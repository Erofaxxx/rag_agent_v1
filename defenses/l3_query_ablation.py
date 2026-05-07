"""L3: detection of query-specific (trigger-activated) chunks via leave-one-out
query ablation.

Generic, model-agnostic, training-free защита против trigger-based backdoor-атак
(BadNets-style) на RAG. В отличие от L1 (regex) и L2 (per-document anomaly), не
полагается ни на знание шаблонов атаки, ни на знание других чанков того же
документа — поэтому ловит атаки, перефразированные так, чтобы избежать L1, и
стелс-бэкдоры, тематически близкие к легитимному корпусу.

## Идея

Trigger-based backdoor работает так: атакующий внедряет в корпус документ,
который retrieval вытаскивает в top-k ТОЛЬКО когда пользовательский запрос
содержит конкретный триггер (редкое сочетание слов вроде «согласно директиве
17-альфа»). На обычных запросах такой документ в top-k не приходит — поэтому
clean accuracy не страдает (стелс) — а на запросе с триггером агент
послушно цитирует target-ответ.

Сигнатура такой атаки: chunk появляется в top-k **только** при наличии конкретных
слов в запросе. Уберёшь хотя бы одно из этих слов — chunk исчезнет, потому
что embedding запроса больше не попадает в его «бэкдор-зону».

## Алгоритм

Для каждого «значимого» (длиной ≥ N) слова в запросе делаем ablated retrieval —
тот же поиск, но с удалённым словом. Считаем для каждого chunk_id из исходного
top-k, в какой доле ablation-выдач он остался. Если chunk выпал из большинства
ablated-выдач (`presence_ratio` < 1 - threshold), значит он query-specific:
trigger_score = 1 - presence_ratio высокий → подозрение.

Нормальный релевантный chunk выживает почти все ablations: его embedding близок
к семантическому центру запроса, и удаление одного второстепенного слова
не выкидывает его из top-k.

## Стоимость

Только дополнительные FAISS retrievals (≤ MAX_ABLATIONS), без LLM-вызовов.
Локально, миллисекунды. На production — бесплатно.

## Что НЕ ловит

- Атаки без явного триггера в запросе (например, статичная prompt-injection,
  которую RAG вытаскивает по семантической релевантности самой темы) — для
  таких атак нужен L1 / L4 / LLM-judge. L3 их и не должен ловить, он
  специализирован на trigger-based.
- Атаки, где «триггер» — это семантически центральное слово запроса (например,
  имя продукта). Такие тоже выпадают из top-k при ablation, поэтому могут
  давать ложные позитивы. Поэтому threshold по умолчанию консервативный.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

log = logging.getLogger(__name__)


# Слова, которые НЕ имеет смысла абляровать — короткие предлоги/союзы/частицы
# и «вопросительные» слова. Ablation таких слов меняет грамматику, но не
# семантику запроса, и любой нормальный chunk останется в top-k. Пропускаем
# их сразу, чтобы не тратить FAISS-вызовы и не размывать сигнал.
_STOP_WORDS_RU = {
    "и", "а", "но", "или", "что", "как", "где", "когда", "почему", "зачем",
    "кто", "чей", "чья", "чьи", "чтоб", "чтобы", "если", "то", "ли", "же",
    "ведь", "вот", "быть", "был", "была", "было", "были", "есть", "это",
    "этот", "эта", "это", "эти", "тот", "та", "то", "те", "так", "там",
    "тут", "уже", "ещё", "его", "её", "их", "мой", "моя", "моё", "мои",
    "твой", "твоя", "твои", "ваш", "ваша", "ваше", "наши", "себя",
    "по", "за", "из", "от", "до", "над", "под", "при", "про", "для",
    "без", "через", "между", "перед", "после", "около", "среди", "около",
    "не", "ни", "да", "нет", "также", "только", "лишь", "именно",
    "какой", "какая", "какое", "какие", "сколько", "сколько-нибудь",
}
_STOP_WORDS_EN = {
    "the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "with",
    "and", "or", "but", "not", "no", "is", "are", "was", "were", "be",
    "what", "which", "who", "how", "why", "when", "where", "this", "that",
    "these", "those", "do", "does", "did", "have", "has", "had", "from",
    "as", "if", "than", "then", "so", "such",
}
_STOPWORDS = _STOP_WORDS_RU | _STOP_WORDS_EN


_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
# Извлекаем «слова» как подряд идущие word-character'ы (буквы/цифры) без _.
# Это не просто split() по пробелам — нужно отделять «17-альфа» как одно слово
# через простое split, либо как два через regex. Сейчас оставляем split-логику
# в caller'е через re.findall(_WORD_TOKEN_RE, query) для надёжности.


@dataclass
class TriggerReport:
    """Отчёт по chunk-у."""
    chunk_id: int
    trigger_score: float  # 0..1, выше — подозрительнее
    presence_ratio: float  # доля ablated retrieval'ов, в которых chunk остался
    n_ablations: int


@dataclass
class L3Report:
    suspicious_chunk_ids: list[int] = field(default_factory=list)
    per_chunk: dict[int, TriggerReport] = field(default_factory=dict)
    # Метаданные для отладки/логов
    candidates_tested: list[str] = field(default_factory=list)
    threshold: float = 0.0
    skipped_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "suspicious_chunk_ids": list(self.suspicious_chunk_ids),
            "n_suspicious": len(self.suspicious_chunk_ids),
            "n_tested": len(self.candidates_tested),
            "candidates_tested": list(self.candidates_tested),
            "threshold": self.threshold,
            "skipped_reason": self.skipped_reason,
        }


def _tokenize(query: str) -> list[str]:
    """Аккуратная токенизация: возвращает список «слов» в порядке появления.
    Сохраняет дефисированные термы как одно слово (например, «17-альфа»).
    """
    # Простой подход: split по \s+ и затем strip пунктуации по краям.
    raw = query.split()
    cleaned: list[str] = []
    for tok in raw:
        # Чистим краевую пунктуацию, но дефис внутри слова сохраняем.
        t = tok.strip(".,;:!?()[]{}«»\"'`<>/\\")
        if t:
            cleaned.append(t)
    return cleaned


def _word_core(token: str) -> str:
    """Берём только буквенно-цифровую часть для оценки длины слова. Это нужно
    для отсева коротких служебных слов независимо от хвостовой пунктуации."""
    return "".join(_WORD_RE.findall(token.lower()))


def _select_candidates(
    tokens: list[str],
    *,
    min_word_len: int,
    max_ablations: int,
) -> list[int]:
    """Возвращает индексы слов, которые имеет смысл абляровать.

    Приоритет: слова длиннее (более «лексически плотные») и не попавшие в
    стоп-лист. На большие запросы ограничиваем до max_ablations самых
    длинных кандидатов — этого достаточно, чтобы поймать триггер
    (триггеры обычно содержат редкие/длинные слова).
    """
    scored: list[tuple[int, int, str]] = []
    for i, tok in enumerate(tokens):
        core = _word_core(tok)
        if len(core) < min_word_len:
            continue
        if core in _STOPWORDS:
            continue
        # Score = длина core, побеждают длинные. При равной длине — порядок.
        scored.append((len(core), i, core))
    # Берём самые длинные, потом сортируем по index для стабильности логов.
    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = scored[:max_ablations]
    selected.sort(key=lambda x: x[1])
    return [i for _len, i, _core in selected]


def detect_query_specific_chunks(
    query: str,
    original_hits: list,  # list[SearchHit] — но не импортим search/, чтобы не было циклов
    retrieve_fn: Callable[[str], list],
    *,
    threshold: float = 0.7,
    max_ablations: int = 8,
    min_word_len: int = 4,
) -> L3Report:
    """Главная функция модуля.

    `original_hits` — top-k для оригинального query (уже посчитан caller'ом).
    `retrieve_fn(q) -> hits` — callback для ablated retrieval'ов. Должен быть
        single-query (БЕЗ multi-query/HyDE), иначе ablation размоется.

    Возвращает L3Report с trigger_score'ами для каждого chunk_id из
    original_hits. Caller сам решает, что делать с suspicious_chunk_ids
    (drop / warn / просто залогировать).
    """
    if not original_hits:
        return L3Report(threshold=threshold, skipped_reason="no_original_hits")

    tokens = _tokenize(query)
    if len(tokens) < 2:
        return L3Report(threshold=threshold, skipped_reason="query_too_short")

    cand_idxs = _select_candidates(
        tokens, min_word_len=min_word_len, max_ablations=max_ablations
    )
    if not cand_idxs:
        return L3Report(threshold=threshold, skipped_reason="no_candidate_words")

    original_ids = [_chunk_id(h) for h in original_hits]
    presence: dict[int, int] = {cid: 0 for cid in original_ids}
    n_ablations = 0
    candidates_tested: list[str] = []

    for idx in cand_idxs:
        ablated_tokens = tokens[:idx] + tokens[idx + 1:]
        ablated_q = " ".join(ablated_tokens).strip()
        if not ablated_q:
            continue
        try:
            ablated_hits = retrieve_fn(ablated_q)
        except Exception as e:
            log.warning("L3 ablation retrieval упал на %r: %s", ablated_q, e)
            continue
        ablated_ids = {_chunk_id(h) for h in (ablated_hits or [])}
        n_ablations += 1
        candidates_tested.append(tokens[idx])
        for cid in original_ids:
            if cid in ablated_ids:
                presence[cid] += 1

    if n_ablations == 0:
        return L3Report(
            threshold=threshold,
            candidates_tested=candidates_tested,
            skipped_reason="no_successful_ablations",
        )

    per_chunk: dict[int, TriggerReport] = {}
    suspicious: list[int] = []
    for cid in original_ids:
        ratio = presence[cid] / n_ablations
        score = 1.0 - ratio
        per_chunk[cid] = TriggerReport(
            chunk_id=cid,
            trigger_score=round(score, 3),
            presence_ratio=round(ratio, 3),
            n_ablations=n_ablations,
        )
        if score >= threshold:
            suspicious.append(cid)

    if suspicious:
        log.info(
            "[L3] query=%r — подозрительные chunks: %s "
            "(threshold=%.2f, ablations=%d, candidates=%s)",
            query, suspicious, threshold, n_ablations, candidates_tested,
        )

    return L3Report(
        suspicious_chunk_ids=suspicious,
        per_chunk=per_chunk,
        candidates_tested=candidates_tested,
        threshold=threshold,
    )


def filter_hits(
    original_hits: list,
    report: L3Report,
    *,
    mode: str,
    escalate_to_document: bool = True,
) -> list:
    """Применяет вердикт L3 к списку hits.

    mode:
        'off'  — не трогаем (caller должен был и не вызывать L3, но на всякий)
        'warn' — оставляем все hits, caller сам приляпает предупреждение
        'drop' — выкидываем suspicious chunks

    escalate_to_document=True (по умолчанию) — если хотя бы один chunk документа
    помечен как trigger-activated, выкидываем ВСЕ chunks этого документа.
    Защищает от «размазывания» target-фразы по нескольким соседним чанкам
    через CHUNK_OVERLAP: атакующий мог сделать так, что часть chunks-носителей
    target-фразы приходят в top-k и без триггера (по тематике запроса) и
    не ловятся chunk-level метрикой; document-level escalation закрывает этот
    обход. False-positive риск: один query-specific chunk целого документа
    выкидывает весь документ — но если у документа есть хоть один такой chunk,
    документ почти наверняка содержит backdoor, и риск ложного дропа
    легитимного документа невелик (для нормальных файлов trigger_score у всех
    chunks ≈ 0).
    """
    if mode != "drop" or not report.suspicious_chunk_ids:
        return original_hits

    sus_chunks = set(report.suspicious_chunk_ids)
    if escalate_to_document:
        sus_doc_ids = {_doc_id(h) for h in original_hits if _chunk_id(h) in sus_chunks}
        return [h for h in original_hits if _doc_id(h) not in sus_doc_ids]
    return [h for h in original_hits if _chunk_id(h) not in sus_chunks]


def short_summary(report: L3Report) -> str:
    """Однострочный summary для логов."""
    if report.skipped_reason:
        return f"L3: skipped ({report.skipped_reason})"
    n_sus = len(report.suspicious_chunk_ids)
    n_total = len(report.per_chunk)
    return (
        f"L3: {n_sus}/{n_total} chunks помечены как trigger-activated "
        f"(threshold={report.threshold:.2f}, ablations tested={len(report.candidates_tested)})"
    )


def build_warning(report: L3Report, filenames_by_chunk: dict[int, str]) -> str:
    """Текст плашки для finalного ответа агента — в стиле L4."""
    if not report.suspicious_chunk_ids:
        return ""
    files = sorted({filenames_by_chunk.get(cid, "?") for cid in report.suspicious_chunk_ids})
    files_str = ", ".join(files)
    return (
        "\n\n⚠️ **Предупреждение безопасности (L3):** найдены фрагменты, "
        f"которые попадают в выдачу только при конкретных словах запроса "
        f"(возможный trigger-based backdoor). Источники: {files_str}. "
        "Проверьте содержимое этих документов вручную."
    )


def _chunk_id(h) -> int:
    """Извлекает chunk.id из SearchHit. Не импортим SearchHit, чтобы не было
    циклов — полагаемся на duck-typing."""
    return int(h.chunk.id)


def _doc_id(h) -> int:
    """Извлекает document.id из SearchHit. Используется для document-level
    escalation в filter_hits."""
    return int(h.document.id)
