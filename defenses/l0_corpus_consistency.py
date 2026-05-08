"""L0: corpus consistency / near-duplicate document detection at ingest time.

Generic, embedding-based, training-free защита против гибридной trigger-based
backdoor-атаки, в которой атакующий клонирует легитимный документ корпуса
и добавляет 1-2 вшитых раздела с target-фразой. Такой poisoned chunk:
- тематически легитимный (его embedding близок к нормальной выдаче);
- содержит target-payload «невыделяющимся» слогом (нет regex-сигнатуры);
- retrievable не только при триггере (chunk-level ablation L3 его не ловит).

Сама атака легко обнаруживается на структурном уровне: poisoned документ
почти идентичен chunkам уже проиндексированного «оригинала» по большей части
своей массы, и отличается только 1-2 inserted разделами. Это очень
характерный паттерн.

## Алгоритм

1. После того как chunks нового документа эмбеджены, но ДО того как они
   попадают в FAISS индекс, для каждого нового embedding делаем поиск
   топ-K в существующем индексе.
2. Для каждого chunk фиксируем (а) лучший cosine, (б) document_id того,
   с кем совпало.
3. Группируем chunks нового документа по document_id «двойника» среди
   существующих. Если у одного и того же существующего документа набралось
   ≥ duplicate_ratio (например, 0.7) chunks нового, чей лучший cosine ≥
   similarity_threshold (≈ 0.92), это near-duplicate-копия.
4. При этом если в новом документе есть chunks без хорошего match
   (cosine < threshold) — это inserted разделы. Они и есть подозрительная
   часть.

В drop-режиме: документ помечается как `error`, чанки в FAISS не попадают,
файл доступен только для аудита. В warn-режиме: чанки попадают, отчёт
логируется, в admin-UI можно увидеть флаг.

## Что L0 НЕ ловит

- Атаку с уникальным документом, написанным с нуля (не клон). Для этого
  нужен L5 contradiction detector на ответе.
- Атаку через мелкое редактирование легитимного документа (одно слово
  заменили). Inserted chunks не появятся, similarity всех chunks высокая.
  Это уже не backdoor, а tampering, требует другую защиту.

## Стоимость

Только дополнительные FAISS-поиски при ingest (по числу chunks нового
документа, top-K ≤ 5). Без LLM. Локально, миллисекунды.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class L0Report:
    """Отчёт L0 по проверке нового документа."""
    is_near_duplicate: bool = False
    n_chunks: int = 0
    n_duplicates: int = 0  # chunks нового, у которых нашёлся «двойник» в индексе
    duplicate_ratio: float = 0.0  # n_duplicates / n_chunks
    primary_match_doc_id: int = -1  # ID документа, чьим клоном выглядит новый
    primary_match_filename: str = ""
    inserted_chunk_indices: list[int] = field(default_factory=list)
    # Список (new_chunk_index, best_neighbor_chunk_id, cosine) — для отладки
    matches: list[dict] = field(default_factory=list)
    similarity_threshold: float = 0.0
    duplicate_ratio_threshold: float = 0.0
    skipped_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "is_near_duplicate": self.is_near_duplicate,
            "n_chunks": self.n_chunks,
            "n_duplicates": self.n_duplicates,
            "duplicate_ratio": round(self.duplicate_ratio, 3),
            "primary_match_doc_id": self.primary_match_doc_id,
            "primary_match_filename": self.primary_match_filename,
            "n_inserted": len(self.inserted_chunk_indices),
            "inserted_chunk_indices": list(self.inserted_chunk_indices),
            "similarity_threshold": self.similarity_threshold,
            "duplicate_ratio_threshold": self.duplicate_ratio_threshold,
            "skipped_reason": self.skipped_reason,
        }


def detect_near_duplicate_document(
    new_chunk_vectors: np.ndarray,
    *,
    search_fn,  # callable(vec, k) -> list[(chunk_id, score)]
    chunk_to_doc_resolver,  # callable(list[chunk_id]) -> dict[chunk_id, (document_id, filename)]
    similarity_threshold: float = 0.92,
    duplicate_ratio_threshold: float = 0.7,
    top_k_neighbors: int = 5,
    min_chunks_to_check: int = 2,
) -> L0Report:
    """Главная функция модуля.

    new_chunk_vectors: (N, D) L2-нормализованные эмбеддинги chunks нового
        документа. Если они ещё не попали в FAISS — отлично, ничего фильтровать
        не надо. Если как-то попали — caller должен передать chunk_to_doc_resolver,
        который сам вернёт корректные mapping'и без shadow-self-match'ей.

    search_fn(vec, k) -> list[(chunk_id, cosine_score)]:
        обёртка над faiss_index.search для одного вектора.

    chunk_to_doc_resolver(list_of_chunk_ids) -> dict[chunk_id, (document_id, filename)]:
        получить document_id и filename для каждого chunk_id.

    Декаплинг: модуль не импортирует ни faiss_index, ни storage.db напрямую,
    чтобы было легко юнит-тестировать с моками.
    """
    n = int(new_chunk_vectors.shape[0]) if new_chunk_vectors.size else 0
    if n == 0:
        return L0Report(skipped_reason="no_chunks")
    if n < min_chunks_to_check:
        # Документ из 1 чанка — структурно не отличим от любого другого
        # короткого документа, near-duplicate detection бессмысленен.
        return L0Report(
            n_chunks=n,
            similarity_threshold=similarity_threshold,
            duplicate_ratio_threshold=duplicate_ratio_threshold,
            skipped_reason=f"too_few_chunks_to_check (n={n}<{min_chunks_to_check})",
        )

    # Поиск neighbors для каждого chunk
    all_neighbor_chunk_ids: set[int] = set()
    raw_results: list[list[tuple[int, float]]] = []
    for i in range(n):
        try:
            neighbors = search_fn(new_chunk_vectors[i], top_k_neighbors)
        except Exception as e:
            log.warning("L0: search упал на chunk %d, пропускаем: %s", i, e)
            neighbors = []
        raw_results.append(list(neighbors))
        for cid, _score in neighbors:
            all_neighbor_chunk_ids.add(int(cid))

    if not all_neighbor_chunk_ids:
        # Индекс пуст или поиск ничего не вернул → точно не near-duplicate.
        return L0Report(
            n_chunks=n,
            similarity_threshold=similarity_threshold,
            duplicate_ratio_threshold=duplicate_ratio_threshold,
            skipped_reason="empty_index_or_no_neighbors",
        )

    # Получаем mapping chunk_id -> (document_id, filename) одним батчем
    try:
        doc_info = chunk_to_doc_resolver(list(all_neighbor_chunk_ids))
    except Exception as e:
        log.warning("L0: chunk_to_doc_resolver упал: %s", e)
        return L0Report(
            n_chunks=n,
            similarity_threshold=similarity_threshold,
            duplicate_ratio_threshold=duplicate_ratio_threshold,
            skipped_reason=f"resolver_error: {e}",
        )

    # Для каждого нового chunk — лучший match из соседей (с группировкой по doc)
    duplicate_doc_counter: dict[int, int] = {}  # doc_id -> сколько new chunks нашли копию в этом doc
    duplicate_doc_filenames: dict[int, str] = {}
    inserted: list[int] = []
    matches: list[dict] = []

    for i, neighbors in enumerate(raw_results):
        best_cosine = -1.0
        best_neighbor_cid = -1
        best_doc_id = -1
        best_filename = ""
        for cid, cos in neighbors:
            cid_int = int(cid)
            if cid_int not in doc_info:
                continue
            d_id, d_name = doc_info[cid_int]
            if cos > best_cosine:
                best_cosine = float(cos)
                best_neighbor_cid = cid_int
                best_doc_id = int(d_id)
                best_filename = str(d_name)

        matches.append({
            "new_chunk_index": i,
            "best_neighbor_chunk_id": best_neighbor_cid,
            "best_neighbor_doc_id": best_doc_id,
            "best_filename": best_filename,
            "cosine": round(best_cosine, 4),
        })

        if best_cosine >= similarity_threshold and best_doc_id >= 0:
            duplicate_doc_counter[best_doc_id] = duplicate_doc_counter.get(best_doc_id, 0) + 1
            duplicate_doc_filenames[best_doc_id] = best_filename
        else:
            inserted.append(i)

    if not duplicate_doc_counter:
        return L0Report(
            n_chunks=n,
            n_duplicates=0,
            duplicate_ratio=0.0,
            inserted_chunk_indices=inserted,
            matches=matches,
            similarity_threshold=similarity_threshold,
            duplicate_ratio_threshold=duplicate_ratio_threshold,
        )

    # Документ с максимальным числом «дубликатных» chunks — primary match
    primary_doc_id = max(duplicate_doc_counter, key=lambda k: duplicate_doc_counter[k])
    primary_count = duplicate_doc_counter[primary_doc_id]
    primary_ratio = primary_count / n

    is_near_duplicate = (
        primary_ratio >= duplicate_ratio_threshold
        # Должны быть И клонированные части (primary_ratio высокий), И inserted
        # (хотя бы один уникальный chunk). Иначе это просто полная копия — её мы
        # не считаем backdoor (это просто загрузка дубля, не атака; для этого
        # нужен дедуп при загрузке, не security defense).
        and len(inserted) >= 1
    )

    return L0Report(
        is_near_duplicate=is_near_duplicate,
        n_chunks=n,
        n_duplicates=primary_count,
        duplicate_ratio=primary_ratio,
        primary_match_doc_id=primary_doc_id,
        primary_match_filename=duplicate_doc_filenames.get(primary_doc_id, ""),
        inserted_chunk_indices=inserted,
        matches=matches,
        similarity_threshold=similarity_threshold,
        duplicate_ratio_threshold=duplicate_ratio_threshold,
    )


def short_summary(report: L0Report) -> str:
    if report.skipped_reason:
        return f"L0: skipped ({report.skipped_reason})"
    if not report.is_near_duplicate:
        return (
            f"L0: clean (n_chunks={report.n_chunks}, "
            f"max_match_ratio={report.duplicate_ratio:.2f})"
        )
    return (
        f"L0: NEAR-DUPLICATE — {report.n_duplicates}/{report.n_chunks} "
        f"chunks ({report.duplicate_ratio:.0%}) идентичны документу "
        f"{report.primary_match_filename!r} (doc_id={report.primary_match_doc_id}); "
        f"inserted chunks: {len(report.inserted_chunk_indices)} → "
        "потенциальный backdoor (копия с вшитыми разделами)"
    )


def build_error_message(report: L0Report) -> str:
    """Сообщение для статуса документа в БД."""
    if not report.is_near_duplicate:
        return ""
    return (
        f"L0 защита: документ выглядит как копия {report.primary_match_filename!r} "
        f"с {len(report.inserted_chunk_indices)} вшитыми разделами "
        f"({report.duplicate_ratio:.0%} chunks идентичны). "
        "Это типичный паттерн trigger-based backdoor-атаки. "
        "Проверьте документ и удалите inserted разделы либо отключите "
        "DEFENSE_L0_CORPUS_CONSISTENCY, чтобы загрузить как есть."
    )
