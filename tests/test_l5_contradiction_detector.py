"""Тесты L5 cross-chunk contradiction detector.

Без реального LLM — подаём моковую llm-обёртку, которая возвращает
заранее заготовленный JSON. Проверяем парсинг, majority-rule, фильтрацию.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from defenses.l5_contradiction_detector import (
    Contradiction,
    detect_contradictions,
    filter_hits,
    short_summary,
    build_warning,
)


# ---- moсkи ----

@dataclass
class _MockChunk:
    id: int
    text: str


@dataclass
class _MockDoc:
    id: int
    filename: str


@dataclass
class _MockHit:
    chunk: _MockChunk
    document: _MockDoc
    score: float = 0.5


def _hit(cid: int, text: str, fname: str, doc_id: int | None = None) -> _MockHit:
    return _MockHit(
        chunk=_MockChunk(id=cid, text=text),
        document=_MockDoc(id=doc_id if doc_id is not None else cid, filename=fname),
    )


@dataclass
class _MockAIMessage:
    content: str


class _MockLLM:
    """Возвращает фиксированный JSON ответ. Для проверки fail-mode — задать None."""

    def __init__(self, response: str | None = None, raise_exc: Exception | None = None):
        self.response = response
        self.raise_exc = raise_exc
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.raise_exc:
            raise self.raise_exc
        return _MockAIMessage(content=self.response or "")


# ---- тесты ----

class TestL5Contradiction:
    def test_no_contradictions_returns_empty(self):
        llm = _MockLLM(response='{"contradictions": []}')
        hits = [
            _hit(1, "лимит 100 000", "clean.md"),
            _hit(2, "согласование финансового директора", "clean.md"),
        ]
        report = detect_contradictions(query="лимит?", hits=hits, llm=llm)
        assert report.contradictions == []
        assert report.minority_chunk_ids == []

    def test_skipped_on_too_few_hits(self):
        llm = _MockLLM(response="bad")
        report = detect_contradictions(query="x", hits=[_hit(1, "y", "z.md")], llm=llm)
        assert "too_few_chunks" in report.skipped_reason
        assert llm.calls == [], "LLM не должен быть вызван"

    def test_finds_contradiction_and_marks_minority(self):
        """Сценарий: 3 chunks из clean.md vs 1 chunk из poisoned.md
        с противоречащим утверждением. Majority — clean (3 chunks),
        minority — poisoned. Все chunks poisoned файла → minority."""
        llm = _MockLLM(response=json.dumps({
            "contradictions": [{"a": 1, "b": 2, "summary": "лимит 100k vs снят"}]
        }))
        hits = [
            _hit(1, "лимит 100 000 рублей", "clean.md"),
            _hit(2, "лимит снят", "poisoned.md"),
            _hit(3, "превышение согласует финдиректор", "clean.md"),
            _hit(4, "лимиты пересматриваются ежеквартально", "clean.md"),
        ]
        report = detect_contradictions(query="лимит?", hits=hits, llm=llm)
        assert len(report.contradictions) == 1
        c = report.contradictions[0]
        assert c.chunk_a == 1 and c.chunk_b == 2
        # poisoned.md имеет 1 chunk, clean.md — 3 → poisoned minority
        assert report.minority_chunk_ids == [2]

    def test_filter_drop_removes_minority(self):
        llm = _MockLLM(response=json.dumps({
            "contradictions": [{"a": 1, "b": 2, "summary": "x"}]
        }))
        hits = [
            _hit(1, "A", "majority.md"),
            _hit(2, "Anti-A", "minor.md"),
            _hit(3, "B", "majority.md"),
        ]
        report = detect_contradictions(query="?", hits=hits, llm=llm)
        filtered = filter_hits(hits, report, mode="drop")
        ids = {h.chunk.id for h in filtered}
        assert 2 not in ids
        assert 1 in ids and 3 in ids

    def test_tie_marks_both_sides(self):
        """Ничья по числу chunks → помечаем ОБА."""
        llm = _MockLLM(response=json.dumps({
            "contradictions": [{"a": 1, "b": 2, "summary": "tie"}]
        }))
        hits = [
            _hit(1, "A", "side_a.md"),
            _hit(2, "B", "side_b.md"),
        ]
        report = detect_contradictions(query="?", hits=hits, llm=llm)
        assert set(report.minority_chunk_ids) == {1, 2}

    def test_invalid_json_fails_open(self):
        llm = _MockLLM(response="not json at all")
        hits = [_hit(1, "x", "a.md"), _hit(2, "y", "b.md")]
        report = detect_contradictions(query="?", hits=hits, llm=llm)
        # fail-open: пустой report, никого не помечаем
        assert report.contradictions == []
        assert report.minority_chunk_ids == []

    def test_llm_exception_fails_open(self):
        llm = _MockLLM(raise_exc=RuntimeError("network down"))
        hits = [_hit(1, "x", "a.md"), _hit(2, "y", "b.md")]
        report = detect_contradictions(query="?", hits=hits, llm=llm)
        assert "llm_error" in report.skipped_reason

    def test_invalid_chunk_ids_in_response_skipped(self):
        """LLM сослался на chunk_id, которого нет в выдаче — игнорируем."""
        llm = _MockLLM(response=json.dumps({
            "contradictions": [{"a": 999, "b": 1, "summary": "ghost"}]
        }))
        hits = [_hit(1, "x", "a.md"), _hit(2, "y", "b.md")]
        report = detect_contradictions(query="?", hits=hits, llm=llm)
        assert report.contradictions == []

    def test_summary_string(self):
        llm = _MockLLM(response='{"contradictions": []}')
        hits = [_hit(1, "x", "a.md"), _hit(2, "y", "b.md")]
        report = detect_contradictions(query="?", hits=hits, llm=llm)
        assert "L5" in short_summary(report)

    def test_warning_message_contains_filenames(self):
        llm = _MockLLM(response=json.dumps({
            "contradictions": [{"a": 1, "b": 2, "summary": "x"}]
        }))
        hits = [
            _hit(1, "A", "alpha.md"),
            _hit(2, "B", "beta.md"),
            _hit(3, "C", "alpha.md"),
        ]
        report = detect_contradictions(query="?", hits=hits, llm=llm)
        fname_by_cid = {h.chunk.id: h.document.filename for h in hits}
        warning = build_warning(report, fname_by_cid)
        assert "alpha.md" in warning
        assert "beta.md" in warning
        assert "L5" in warning
