"""L5: cross-chunk contradiction detection via LLM-judge.

Generic, model-agnostic защита, ловящая случаи, когда внутри retrieval-выдачи
есть chunks, прямо противоречащие друг другу. Это сигнатура широкого класса
проблем:

- Гибридный trigger-based backdoor: атакующий вшил target-фразу
  («лимит снят») в legit-tematic chunk, который ретрив подтягивает наряду с
  настоящим правилом из легитимного документа («лимит 100 000»). На уровне
  отдельного chunk обе стороны выглядят корректно — но **вместе** они
  логически несовместимы.
- Устаревшие версии документов в корпусе.
- Просто противоречивые источники.

Все три случая лечатся одинаково: показать пользователю, что выдача
противоречива, и/или взять «большинство» (тот файл, у которого больше
chunks в выдаче по теме). В drop-mode выкидываем minority side → агент
не получает противоречивых данных и отвечает консистентно.

## Алгоритм

1. Если в выдаче < min_chunks_to_check (по умолчанию 2) — пропускаем
   (сравнивать не с чем).
2. Скармливаем LLM-judge'у компактный список chunks: id, filename, snippet
   (до N символов). Просим вернуть JSON с парами `{a, b, summary}`,
   где a и b — chunk_id противоречащих фрагментов.
3. Для каждой пары определяем majority по filename: какой файл присутствует
   в выдаче большим числом chunks. Minority chunks → suspicious.
4. В warn-mode копим warnings, в drop-mode фильтруем minority.

## Стоимость

+1 LLM-вызов на каждый search_documents tool call, который вернул ≥ 2
chunks. Используем temperature=0 и относительно дешёвую модель
(та же, что и основной агент — DeepSeek через OpenRouter).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)


# langchain_core может отсутствовать в юнит-тестах без production-stack.
# Используем настоящие классы, если они есть, иначе лёгкие подделки —
# для LLM-judge'а важно только наличие .content. Caller всегда передаёт
# свой `llm`, и mock-LLM в тестах принимает что угодно.
try:
    from langchain_core.messages import HumanMessage as _HumanMessage  # type: ignore
    from langchain_core.messages import SystemMessage as _SystemMessage  # type: ignore
except Exception:  # pragma: no cover — только для тестов без langchain
    @dataclass
    class _HumanMessage:  # type: ignore
        content: str

    @dataclass
    class _SystemMessage:  # type: ignore
        content: str


@dataclass
class Contradiction:
    chunk_a: int
    chunk_b: int
    summary: str

    def to_dict(self) -> dict:
        return {"chunk_a": self.chunk_a, "chunk_b": self.chunk_b, "summary": self.summary}


@dataclass
class L5Report:
    contradictions: list[Contradiction] = field(default_factory=list)
    minority_chunk_ids: list[int] = field(default_factory=list)
    skipped_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "n_contradictions": len(self.contradictions),
            "contradictions": [c.to_dict() for c in self.contradictions],
            "minority_chunk_ids": list(self.minority_chunk_ids),
            "skipped_reason": self.skipped_reason,
        }


_SYSTEM = (
    "Ты — fact-checker для корпоративной RAG-системы. На вход — вопрос "
    "пользователя и список фрагментов из разных документов корпуса. Твоя "
    "задача — найти ПРЯМЫЕ ФАКТИЧЕСКИЕ ПРОТИВОРЕЧИЯ между этими фрагментами "
    "по теме вопроса.\n\n"
    "Считай противоречием: разные значения одной сущности (A: «лимит "
    "100 000 руб», B: «лимит снят»); прямо противоположные правила (A: "
    "«запрещено», B: «разрешено без согласований»).\n\n"
    "НЕ считай противоречием: разные формулировки одной идеи; описание "
    "общего случая в A и исключения в B; различия в стиле; неполные "
    "цитаты; разные аспекты предмета (A — про лимит за день, B — за месяц).\n\n"
    "Возвращай только JSON, без преамбул, в формате:\n"
    '{"contradictions": [{"a": <chunk_id>, "b": <chunk_id>, '
    '"summary": "<суть противоречия в 1 фразе>"}, ...]}\n\n'
    "Если противоречий нет — пустой массив. chunk_id — ровно тот, который "
    "указан перед фрагментом в квадратных скобках."
)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _format_chunks_for_judge(hits: list, max_snippet_chars: int = 500) -> str:
    """Компактный листинг chunks для LLM-judge'а."""
    parts: list[str] = []
    for h in hits:
        cid = int(h.chunk.id)
        fname = h.document.filename
        text = (h.chunk.text or "").strip()
        if len(text) > max_snippet_chars:
            text = text[:max_snippet_chars].rstrip() + "…"
        parts.append(f"[chunk_id={cid}] [{fname}]\n{text}")
    return "\n\n---\n\n".join(parts)


def detect_contradictions(
    *,
    query: str,
    hits: list,  # list[SearchHit] — duck-typing
    llm,  # langchain ChatOpenAI или совместимый, должен иметь .invoke([...]) → AIMessage
    min_chunks_to_check: int = 2,
    max_snippet_chars: int = 500,
) -> L5Report:
    """Главная функция модуля.

    `hits` — top-k из retrieval'а. `llm` — langchain ChatOpenAI или совместимый.
    Если llm падает / возвращает не-JSON — возвращаем пустой report (fail-open),
    чтобы L5 не блокировала выдачу при сетевых проблемах.
    """
    if not hits or len(hits) < min_chunks_to_check:
        return L5Report(skipped_reason=f"too_few_chunks (n={len(hits or [])}<{min_chunks_to_check})")

    chunks_text = _format_chunks_for_judge(hits, max_snippet_chars=max_snippet_chars)
    user_payload = (
        f"Вопрос пользователя: {query}\n\n"
        f"Фрагменты:\n{chunks_text}"
    )

    try:
        msg = llm.invoke([
            _SystemMessage(content=_SYSTEM),
            _HumanMessage(content=user_payload),
        ])
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
    except Exception as e:
        log.warning("L5: LLM-judge упал (%s) — fail-open, contradictions не проверены", e)
        return L5Report(skipped_reason=f"llm_error: {e}")

    match = _JSON_RE.search(text or "")
    if not match:
        log.debug("L5: LLM вернул не-JSON, пропускаем")
        return L5Report(skipped_reason="non_json_response")
    try:
        data = json.loads(match.group(0))
    except Exception:
        return L5Report(skipped_reason="invalid_json")

    raw = data.get("contradictions") or []
    if not isinstance(raw, list):
        return L5Report(skipped_reason="bad_format")

    chunk_id_set = {int(h.chunk.id) for h in hits}
    contradictions: list[Contradiction] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            a = int(item.get("a"))
            b = int(item.get("b"))
        except (TypeError, ValueError):
            continue
        if a not in chunk_id_set or b not in chunk_id_set:
            continue
        if a == b:
            continue
        summary = str(item.get("summary") or "").strip()
        contradictions.append(Contradiction(chunk_a=a, chunk_b=b, summary=summary))

    minority_ids = _compute_minority_chunk_ids(hits, contradictions)

    if contradictions:
        log.info(
            "[L5] обнаружено %d противоречий: %s; minority chunks=%s",
            len(contradictions),
            [(c.chunk_a, c.chunk_b) for c in contradictions],
            minority_ids,
        )

    return L5Report(
        contradictions=contradictions,
        minority_chunk_ids=minority_ids,
    )


def _compute_minority_chunk_ids(
    hits: list,
    contradictions: list[Contradiction],
) -> list[int]:
    """Для каждой противоречивой пары определяем minority side и помечаем
    её chunks. Majority — тот filename, у которого больше chunks в выдаче.
    Если ничья (равное число chunks от обеих сторон) — помечаем ОБЕ
    стороны (в drop-mode caller выкинет всё противоречивое и попросит
    пользователя уточнить).
    """
    chunk_to_filename: dict[int, str] = {int(h.chunk.id): h.document.filename for h in hits}
    filename_count: dict[str, int] = {}
    for h in hits:
        fn = h.document.filename
        filename_count[fn] = filename_count.get(fn, 0) + 1

    minority_ids: set[int] = set()
    for c in contradictions:
        fa = chunk_to_filename.get(c.chunk_a)
        fb = chunk_to_filename.get(c.chunk_b)
        if not fa or not fb:
            continue
        ca = filename_count.get(fa, 0)
        cb = filename_count.get(fb, 0)
        if ca > cb:
            # b — minority
            minority_ids.update(
                int(h.chunk.id) for h in hits if h.document.filename == fb
            )
        elif cb > ca:
            minority_ids.update(
                int(h.chunk.id) for h in hits if h.document.filename == fa
            )
        else:
            # ничья — помечаем обе стороны как подозрительные
            minority_ids.update(
                int(h.chunk.id) for h in hits
                if h.document.filename in (fa, fb)
            )

    return sorted(minority_ids)


def filter_hits(
    hits: list,
    report: L5Report,
    *,
    mode: str,
) -> list:
    """В drop-mode убираем minority chunks. В warn — оставляем."""
    if mode == "drop" and report.minority_chunk_ids:
        sus = set(report.minority_chunk_ids)
        return [h for h in hits if int(h.chunk.id) not in sus]
    return hits


def short_summary(report: L5Report) -> str:
    if report.skipped_reason:
        return f"L5: skipped ({report.skipped_reason})"
    n = len(report.contradictions)
    if n == 0:
        return "L5: clean (no contradictions found)"
    return (
        f"L5: {n} contradictions detected; "
        f"minority chunks flagged: {report.minority_chunk_ids}"
    )


def build_warning(report: L5Report, filenames_by_chunk: dict[int, str]) -> str:
    """Плашка для финального ответа в warn-режиме."""
    if not report.contradictions:
        return ""
    files_in_play: set[str] = set()
    for c in report.contradictions:
        if c.chunk_a in filenames_by_chunk:
            files_in_play.add(filenames_by_chunk[c.chunk_a])
        if c.chunk_b in filenames_by_chunk:
            files_in_play.add(filenames_by_chunk[c.chunk_b])
    files_str = ", ".join(sorted(files_in_play)) or "источники"
    return (
        "\n\n⚠️ **Предупреждение безопасности (L5):** в выдаче поиска "
        f"обнаружены противоречия между источниками ({files_str}). "
        "Это может означать устаревшую копию документа, ошибку в данных "
        "или попытку trigger-based backdoor (вшитое противоречащее "
        "правило). Перепроверьте утверждения по оригиналам."
    )
