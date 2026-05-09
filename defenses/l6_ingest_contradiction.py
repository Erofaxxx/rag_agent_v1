"""L6: ingest-time contradiction check via LLM-judge.

Закрывает архитектурный пробел остальных слоёв:
- L0 видит structurные клоны, не видит content-conflicts.
- L1 видит regex-паттерны, не видит «тихих» утверждений типа
  «согласно директиве X, лимит снят».
- L2 нужен размер документа.
- L3 не работает на коротких триггерах (1-2 слова после reformulate).
- L5 на query-time видит контрадикции в выдаче, но не блокирует ingest.

L6 при ingest нового документа для каждого chunk:
1. Ищет в существующем FAISS-индексе top-K соседей с cosine ≥ 0.5
   (т.е. на ту же тему, не любой шум).
2. Если соседи есть — даёт LLM-judge сравнить новый chunk с ними:
   «есть ли прямое противоречие (численное, правило-противоречие,
   противоположные действия)? Считай новую редакцию, отменяющую
   старую, противоречием — пользователю нужна ОДНА норма».
3. Если LLM ответил yes — помечает chunk как contradicting.

В drop-режиме документ блокируется (status=error) если ≥1 chunk
помечен. В warn — пишется лог + админ может пересмотреть в UI.

Стоимость: +1 LLM-call на каждый chunk нового документа, у которого
нашлись «соседи» в индексе. Дешёвая модель с temperature=0,
короткий prompt, можно ставить лимит 200 токенов.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

log = logging.getLogger(__name__)


try:  # pragma: no cover — fallback для unit-тестов
    from langchain_core.messages import HumanMessage as _HumanMessage
    from langchain_core.messages import SystemMessage as _SystemMessage
except Exception:  # pragma: no cover
    @dataclass
    class _HumanMessage:  # type: ignore
        content: str

    @dataclass
    class _SystemMessage:  # type: ignore
        content: str


@dataclass
class L6Finding:
    new_chunk_index: int
    neighbor_chunk_id: int
    neighbor_filename: str
    neighbor_doc_id: int
    cosine: float
    summary: str

    def to_dict(self) -> dict:
        return {
            "new_chunk_index": self.new_chunk_index,
            "neighbor_chunk_id": self.neighbor_chunk_id,
            "neighbor_filename": self.neighbor_filename,
            "neighbor_doc_id": self.neighbor_doc_id,
            "cosine": round(self.cosine, 4),
            "summary": self.summary,
        }


@dataclass
class L6Report:
    findings: list[L6Finding] = field(default_factory=list)
    n_chunks_checked: int = 0
    n_chunks_with_neighbors: int = 0
    skipped_reason: str = ""

    @property
    def has_contradiction(self) -> bool:
        return len(self.findings) > 0

    def to_dict(self) -> dict:
        return {
            "n_findings": len(self.findings),
            "n_chunks_checked": self.n_chunks_checked,
            "n_chunks_with_neighbors": self.n_chunks_with_neighbors,
            "skipped_reason": self.skipped_reason,
            "findings": [f.to_dict() for f in self.findings],
        }


_SYSTEM = (
    "Ты — fact-checker для корпоративной RAG-системы. На вход — фрагмент "
    "из НОВОГО документа, который пользователь загружает в систему, и список "
    "фрагментов из УЖЕ существующих документов корпуса по той же теме. "
    "Твоя задача — определить: содержит ли НОВЫЙ фрагмент утверждение, "
    "которое прямо ПРОТИВОРЕЧИТ какому-либо из существующих фрагментов.\n\n"
    "Считай противоречием:\n"
    "- разные численные значения одной и той же сущности (новый: «суточные "
    "5000 руб», существующий: «суточные 1500 руб»; новый: «лимит снят», "
    "существующий: «лимит 100 000»);\n"
    "- прямо противоположные правила или действия (новый: «согласование не "
    "требуется», существующий: «требует приказа»);\n"
    "- утверждения вида «правила раздела N не применяются» / «новые нормы "
    "отменяют предыдущие» — даже если новый документ оформлен как «новая "
    "редакция» или «распоряжение об отмене», это противоречие, потому что "
    "пользователю нужна ОДНОЗНАЧНАЯ норма.\n\n"
    "НЕ считай противоречием: уточнения, дополняющие сведения, разные "
    "аспекты одной сущности (день/месяц), описания смежных тем.\n\n"
    "Возвращай ТОЛЬКО JSON без преамбул:\n"
    '{"contradicts": true|false, "summary": "<суть противоречия в '
    "1 фразе или пустая строка>\"}\n"
)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _judge_pair(llm, new_text: str, neighbor_texts: list[str]) -> tuple[bool, str]:
    """LLM-judge: сравнивает один new_text с массивом neighbor_texts.
    Возвращает (contradicts, summary). Fail-open: при ошибке → (False, '')."""
    payload = (
        f"НОВЫЙ фрагмент (загружается):\n{new_text[:1200]}\n\n"
        "Существующие фрагменты по той же теме:\n"
        + "\n\n---\n\n".join(f"[neighbor_{i}]\n{t[:800]}" for i, t in enumerate(neighbor_texts))
    )
    try:
        msg = llm.invoke([
            _SystemMessage(content=_SYSTEM),
            _HumanMessage(content=payload),
        ])
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
    except Exception as e:
        log.warning("L6: LLM-judge упал: %s", e)
        return (False, "")

    m = _JSON_RE.search(text or "")
    if not m:
        return (False, "")
    try:
        data = json.loads(m.group(0))
    except Exception:
        return (False, "")
    contra = bool(data.get("contradicts"))
    summary = str(data.get("summary") or "").strip()
    return (contra, summary)


def detect_ingest_contradiction(
    *,
    new_chunk_texts: list[str],
    new_chunk_vectors: np.ndarray,
    search_fn: Callable,            # callable(vec, k) -> [(chunk_id, cosine)]
    chunk_resolver: Callable,       # callable(list[chunk_id]) -> dict[cid, (text, filename, doc_id)]
    llm,                            # langchain ChatOpenAI или mock
    similarity_threshold: float = 0.5,
    top_k_neighbors: int = 3,
    max_chunks_to_check: int = 5,
) -> L6Report:
    """Главная функция модуля.

    Если у нового документа > max_chunks_to_check chunks — берём первые
    `max_chunks_to_check` (для защиты от очень дорогих ingest'ов
    100-страничных документов; на практике backdoor inserted-разделы
    обычно в первых).

    similarity_threshold = 0.5 — соседи на той же теме, не любой шум.
    Меньше → больше LLM-вызовов и FP. Больше → пропустим тонкие
    смысловые конфликты.
    """
    n = int(new_chunk_vectors.shape[0]) if new_chunk_vectors.size else 0
    if n == 0 or len(new_chunk_texts) != n:
        return L6Report(skipped_reason="no_chunks_or_text_mismatch")

    findings: list[L6Finding] = []
    n_with_neighbors = 0
    n_to_check = min(n, max_chunks_to_check)

    for i in range(n_to_check):
        try:
            neighbors = search_fn(new_chunk_vectors[i], top_k_neighbors)
        except Exception as e:
            log.warning("L6: search упал на chunk %d: %s", i, e)
            continue
        relevant = [(int(cid), float(cos)) for cid, cos in neighbors if cos >= similarity_threshold]
        if not relevant:
            continue
        n_with_neighbors += 1

        try:
            resolved = chunk_resolver([cid for cid, _ in relevant])
        except Exception as e:
            log.warning("L6: chunk_resolver упал: %s", e)
            continue

        # Возьмём только успешно резолвленные (orphan-chunks игнорируем)
        valid_neighbors = [
            (cid, cos, resolved[cid])
            for cid, cos in relevant
            if cid in resolved
        ]
        if not valid_neighbors:
            continue

        neighbor_texts = [data[0] for _, _, data in valid_neighbors]
        contra, summary = _judge_pair(llm, new_chunk_texts[i], neighbor_texts)
        if contra:
            # Указываем самого близкого соседа как primary
            best = max(valid_neighbors, key=lambda x: x[1])
            cid, cos, (_text, fname, doc_id) = best
            findings.append(L6Finding(
                new_chunk_index=i,
                neighbor_chunk_id=cid,
                neighbor_filename=fname,
                neighbor_doc_id=doc_id,
                cosine=cos,
                summary=summary or "противоречит существующему документу",
            ))

    return L6Report(
        findings=findings,
        n_chunks_checked=n_to_check,
        n_chunks_with_neighbors=n_with_neighbors,
    )


def short_summary(report: L6Report) -> str:
    if report.skipped_reason:
        return f"L6: skipped ({report.skipped_reason})"
    if not report.findings:
        return (
            f"L6: clean (checked {report.n_chunks_checked}, "
            f"{report.n_chunks_with_neighbors} с соседями)"
        )
    files = sorted({f.neighbor_filename for f in report.findings})
    return (
        f"L6: CONTRADICTION — {len(report.findings)} chunk(а/ов) нового документа "
        f"противоречат существующим: {files}"
    )


def build_error_message(report: L6Report) -> str:
    if not report.findings:
        return ""
    files = sorted({f.neighbor_filename for f in report.findings})
    summaries = [f.summary for f in report.findings if f.summary]
    summary_str = "; ".join(summaries[:3]) if summaries else "противоречие в утверждениях"
    return (
        f"L6 защита: документ содержит {len(report.findings)} утверждение(ий), "
        f"противоречащих существующим документам корпуса ({', '.join(files)}). "
        f"Суть: {summary_str}. Пользователю нужна ОДНОЗНАЧНАЯ норма; "
        "загрузка перезаписывающего документа должна согласовываться "
        "вручную — например, через удаление старого. Отключить проверку — "
        "DEFENSE_L6_INGEST_CONTRADICTION=off."
    )
