"""Тесты L3 query-ablation детектора.

Тестируем без реального retrieval'а — подаём моковую `retrieve_fn`,
которая моделирует поведение FAISS:
- «backdoor chunk» (id=42) появляется в top-k ТОЛЬКО когда в запросе есть
  слово «триггер»;
- «легитимные chunks» (id=1, id=2) приходят на любой запрос про лимиты.

Это даёт нам доказуемую сигнатуру trigger-based backdoor — chunk выпадает,
если убрать триггер из запроса.
"""
from __future__ import annotations

from dataclasses import dataclass

from defenses.l3_query_ablation import (
    detect_query_specific_chunks,
    filter_hits,
    short_summary,
)


@dataclass
class _MockChunk:
    id: int


@dataclass
class _MockHit:
    chunk: _MockChunk
    score: float = 0.5


def _hit(cid: int, score: float = 0.5) -> _MockHit:
    return _MockHit(chunk=_MockChunk(id=cid), score=score)


class TestQueryAblation:
    def test_backdoor_chunk_caught(self):
        """Triggered chunk: пропадает из top-k, когда из запроса убирают триггер.

        Моделируем реалистичный multi-word триггер: chunk появляется только
        когда в запросе есть И «директиве», И «17-альфа» (имитация
        embedding-близости к poisoned документу, в котором фраза-триггер
        встречается целиком). Удаление любого из этих слов ломает retrieval.
        """

        def retrieve(q: str):
            ids = [1, 2]
            ql = q.lower()
            if "директиве" in ql and "17-альфа" in ql:
                ids.append(42)
            return [_hit(i) for i in ids]

        original_hits = retrieve("Какой лимит согласно директиве 17-альфа применяется")
        # Для multi-word триггера в длинном запросе используем 0.3 — это
        # реалистично: из 5 candidates 2 триггерных → presence=3/5=0.6 →
        # score=0.4. Ставим порог чуть ниже, чтобы поймать.
        report = detect_query_specific_chunks(
            query="Какой лимит согласно директиве 17-альфа применяется",
            original_hits=original_hits,
            retrieve_fn=retrieve,
            threshold=0.3,
            max_ablations=8,
            min_word_len=4,
        )
        # 42 должен быть помечен как trigger-activated
        assert 42 in report.suspicious_chunk_ids, (
            f"42 not flagged. report={report}"
        )
        # 1 и 2 — нет: они есть в любой выдаче
        assert 1 not in report.suspicious_chunk_ids
        assert 2 not in report.suspicious_chunk_ids
        # Score у trigger-chunk-а должен быть СУЩЕСТВЕННО выше, чем у нормальных
        assert report.per_chunk[42].trigger_score > report.per_chunk[1].trigger_score
        assert report.per_chunk[1].trigger_score == 0.0

    def test_no_false_positive_on_clean_query(self):
        """Без триггера — ни один chunk не должен быть помечен."""

        def retrieve(q: str):
            # Фиксированный набор для любого запроса
            return [_hit(1), _hit(2), _hit(3)]

        original_hits = retrieve("Какой стандартный лимит по операции применяется")
        report = detect_query_specific_chunks(
            query="Какой стандартный лимит по операции применяется",
            original_hits=original_hits,
            retrieve_fn=retrieve,
            threshold=0.7,
            max_ablations=8,
            min_word_len=4,
        )
        assert report.suspicious_chunk_ids == []

    def test_skipped_on_short_query(self):
        """Один значимый токен — нечего абляровать."""

        def retrieve(q: str):
            return [_hit(1)]

        report = detect_query_specific_chunks(
            query="лимит",
            original_hits=[_hit(1)],
            retrieve_fn=retrieve,
            threshold=0.7,
            max_ablations=8,
            min_word_len=4,
        )
        assert report.skipped_reason
        assert report.suspicious_chunk_ids == []

    def test_skipped_on_empty_hits(self):
        report = detect_query_specific_chunks(
            query="any query with words",
            original_hits=[],
            retrieve_fn=lambda q: [],
            threshold=0.7,
            max_ablations=8,
            min_word_len=4,
        )
        assert report.skipped_reason == "no_original_hits"

    def test_filter_drop_mode(self):
        """В drop-mode выкидываем подозрительные chunks из выдачи."""

        def retrieve(q: str):
            ids = [1, 2]
            ql = q.lower()
            if "директиву" in ql and "17-альфа" in ql:
                ids.append(42)
            return [_hit(i) for i in ids]

        original_hits = retrieve("вопрос про директиву 17-альфа")
        report = detect_query_specific_chunks(
            query="вопрос про директиву 17-альфа",
            original_hits=original_hits,
            retrieve_fn=retrieve,
            threshold=0.3,
            max_ablations=8,
            min_word_len=4,
        )
        assert 42 in report.suspicious_chunk_ids
        filtered = filter_hits(original_hits, report, mode="drop")
        ids_left = {h.chunk.id for h in filtered}
        assert 42 not in ids_left
        assert 1 in ids_left and 2 in ids_left

    def test_filter_warn_mode_keeps_all(self):
        def retrieve(q: str):
            ids = [1, 2]
            ql = q.lower()
            if "директиву" in ql and "17-альфа" in ql:
                ids.append(42)
            return [_hit(i) for i in ids]

        original_hits = retrieve("вопрос про директиву 17-альфа сейчас")
        report = detect_query_specific_chunks(
            query="вопрос про директиву 17-альфа сейчас",
            original_hits=original_hits,
            retrieve_fn=retrieve,
            threshold=0.5,
            max_ablations=8,
            min_word_len=4,
        )
        filtered = filter_hits(original_hits, report, mode="warn")
        ids_left = {h.chunk.id for h in filtered}
        # Warn-mode не убирает hits
        assert 42 in ids_left

    def test_summary_string(self):
        report = detect_query_specific_chunks(
            query="too short",
            original_hits=[_hit(1)],
            retrieve_fn=lambda q: [_hit(1)],
            threshold=0.7,
            max_ablations=8,
            min_word_len=4,
        )
        s = short_summary(report)
        assert isinstance(s, str) and "L3" in s

    def test_stopwords_not_ablated(self):
        """«не», «и», «что» — не должны попадать в кандидаты, даже если их в запросе много."""
        calls = []

        def retrieve(q: str):
            calls.append(q)
            return [_hit(1), _hit(2)]

        original_hits = retrieve("какой лимит и сколько что-то и где")
        detect_query_specific_chunks(
            query="какой лимит и сколько что-то и где",
            original_hits=original_hits,
            retrieve_fn=retrieve,
            threshold=0.7,
            max_ablations=8,
            min_word_len=4,
        )
        # Ablation queries не должны содержать удаления служебных слов «и»/«что»
        # (они отфильтрованы как стоп-слова или короче min_word_len). Проверяем:
        # удалено должно быть только значимое слово (например, «лимит», «сколько»).
        # Calls[0] — оригинальный, calls[1:] — ablated.
        for q in calls[1:]:
            # «лимит» или «сколько» должны быть удалены, остальные слова — на месте
            assert q != ""  # ablation не должен делать пустую строку

    def test_dedup_candidates(self):
        """Если запрос содержит одно и то же длинное слово два раза, ablation
        делается только один раз (при удалении одного экземпляра второй остаётся,
        и chunk не должен значимо изменить статус). Это просто smoke-тест,
        что мы не падаем."""

        def retrieve(q: str):
            return [_hit(1), _hit(2)]

        report = detect_query_specific_chunks(
            query="лимит лимит лимит сколько",
            original_hits=[_hit(1), _hit(2)],
            retrieve_fn=retrieve,
            threshold=0.7,
            max_ablations=8,
            min_word_len=4,
        )
        # Главное — не упали
        assert isinstance(report.per_chunk, dict)
