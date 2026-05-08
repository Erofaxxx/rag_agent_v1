"""Тесты L0 near-duplicate detection.

Без реального FAISS / БД — подаём моковые `search_fn` и `chunk_to_doc_resolver`,
которые имитируют существующий индекс. Проверяем, что:
- чистый новый документ (не похож ни на что) → not near-duplicate
- ещё-не-загруженный полный клон существующего → near-duplicate, но БЕЗ inserted
  (не считается атакой, это просто дубль)
- клон + 1-2 inserted раздела → near-duplicate=True, inserted_chunk_indices != []
- слишком короткий документ (1 chunk) → skipped
"""
from __future__ import annotations

import numpy as np

from defenses.l0_corpus_consistency import (
    detect_near_duplicate_document,
    short_summary,
    build_error_message,
)


def _fake_index(chunks_in_index: dict[int, tuple[np.ndarray, int, str]]):
    """Возвращает (search_fn, resolver_fn) для моков.

    chunks_in_index: chunk_id → (vector, doc_id, filename)
    """

    def search_fn(vec, k):
        # Linear scan, cosine on L2-normalized → inner product
        v = np.asarray(vec, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-12)
        scored: list[tuple[int, float]] = []
        for cid, (cvec, _did, _fn) in chunks_in_index.items():
            cv = np.asarray(cvec, dtype=np.float32)
            cv = cv / (np.linalg.norm(cv) + 1e-12)
            scored.append((int(cid), float(np.dot(v, cv))))
        scored.sort(key=lambda p: -p[1])
        return scored[:k]

    def resolver(chunk_id_list):
        return {
            int(cid): (chunks_in_index[cid][1], chunks_in_index[cid][2])
            for cid in chunk_id_list
            if int(cid) in chunks_in_index
        }

    return search_fn, resolver


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    """Детерминированный псевдо-вектор для теста."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v


class TestL0CorpusConsistency:
    def test_clean_doc_no_near_duplicate(self):
        """Новый документ, не похожий ни на что → not near-duplicate."""
        index = {
            10: (_vec(10), 1, "hr.md"),
            11: (_vec(11), 1, "hr.md"),
            20: (_vec(20), 2, "policy.md"),
        }
        search_fn, resolver = _fake_index(index)
        new_vecs = np.stack([_vec(100), _vec(101), _vec(102)])
        report = detect_near_duplicate_document(
            new_vecs,
            search_fn=search_fn,
            chunk_to_doc_resolver=resolver,
            similarity_threshold=0.92,
            duplicate_ratio_threshold=0.7,
        )
        assert report.is_near_duplicate is False

    def test_full_duplicate_caught(self):
        """Полный клон документа → near-duplicate=True.

        Раньше требовали inserted ≥ 1, но в реальной атаке с большим
        CHUNK_SIZE inserted раздел может «размазаться» по соседним chunks
        и не дать ни одного chunk с cosine ниже similarity_threshold.
        Поэтому полный клон тоже flag'аем — либо это безобидный дубль
        (юзер увидит warning), либо subtle data-poisoning, неуловимый
        на уровне embedding."""
        v1, v2, v3, v4 = _vec(1), _vec(2), _vec(3), _vec(4)
        index = {
            10: (v1, 1, "regulation.md"),
            11: (v2, 1, "regulation.md"),
            12: (v3, 1, "regulation.md"),
            13: (v4, 1, "regulation.md"),
        }
        search_fn, resolver = _fake_index(index)
        new_vecs = np.stack([v1, v2, v3, v4])
        report = detect_near_duplicate_document(
            new_vecs,
            search_fn=search_fn,
            chunk_to_doc_resolver=resolver,
            similarity_threshold=0.92,
            duplicate_ratio_threshold=0.7,
        )
        assert report.duplicate_ratio == 1.0
        assert report.inserted_chunk_indices == []
        assert report.is_near_duplicate is True, (
            "полный клон должен помечаться (defensive default)"
        )

    def test_clone_with_inserted_section_caught(self):
        """Клон с 1 inserted разделом → near-duplicate=True, в нашей атаке
        этот один inserted и есть payload."""
        v1, v2, v3 = _vec(1), _vec(2), _vec(3)
        index = {
            10: (v1, 1, "regulation_clean.md"),
            11: (v2, 1, "regulation_clean.md"),
            12: (v3, 1, "regulation_clean.md"),
        }
        search_fn, resolver = _fake_index(index)
        # Новый документ: 3 идентичных chunks + 1 уникальный (inserted backdoor)
        backdoor_vec = _vec(999)  # очень далёкий от v1/v2/v3
        new_vecs = np.stack([v1, v2, v3, backdoor_vec])
        report = detect_near_duplicate_document(
            new_vecs,
            search_fn=search_fn,
            chunk_to_doc_resolver=resolver,
            similarity_threshold=0.92,
            duplicate_ratio_threshold=0.7,
        )
        assert report.is_near_duplicate is True
        assert report.duplicate_ratio == 0.75  # 3 из 4 — duplicates
        assert report.inserted_chunk_indices == [3], "inserted на 4-й позиции"
        assert report.primary_match_filename == "regulation_clean.md"
        assert "regulation_clean.md" in build_error_message(report)

    def test_partial_overlap_below_ratio(self):
        """Документ совпадает с другим только на 50% chunks → ниже порога 70%.
        Это не клон, не помечаем."""
        v1, v2 = _vec(1), _vec(2)
        index = {
            10: (v1, 1, "policy.md"),
            11: (v2, 1, "policy.md"),
        }
        search_fn, resolver = _fake_index(index)
        # 2 chunks совпадают, 2 — уникальные
        new_vecs = np.stack([v1, v2, _vec(50), _vec(51)])
        report = detect_near_duplicate_document(
            new_vecs,
            search_fn=search_fn,
            chunk_to_doc_resolver=resolver,
            similarity_threshold=0.92,
            duplicate_ratio_threshold=0.7,
        )
        assert report.duplicate_ratio == 0.5
        assert report.is_near_duplicate is False

    def test_too_few_chunks_skipped(self):
        index = {10: (_vec(1), 1, "x.md")}
        search_fn, resolver = _fake_index(index)
        new_vecs = _vec(100).reshape(1, -1)
        report = detect_near_duplicate_document(
            new_vecs,
            search_fn=search_fn,
            chunk_to_doc_resolver=resolver,
            min_chunks_to_check=2,
        )
        assert "too_few_chunks" in report.skipped_reason

    def test_empty_index_skipped(self):
        search_fn, resolver = _fake_index({})
        new_vecs = np.stack([_vec(1), _vec(2), _vec(3)])
        report = detect_near_duplicate_document(
            new_vecs,
            search_fn=search_fn,
            chunk_to_doc_resolver=resolver,
        )
        assert report.is_near_duplicate is False

    def test_summary_string(self):
        v1 = _vec(1)
        index = {10: (v1, 1, "x.md"), 11: (_vec(2), 1, "x.md"), 12: (_vec(3), 1, "x.md")}
        search_fn, resolver = _fake_index(index)
        new = np.stack([v1, _vec(2), _vec(3), _vec(999)])
        report = detect_near_duplicate_document(
            new, search_fn=search_fn, chunk_to_doc_resolver=resolver
        )
        s = short_summary(report)
        assert isinstance(s, str) and "L0" in s
