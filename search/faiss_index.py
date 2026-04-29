import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from config import settings
from embeddings import embedding_service
from storage import db, ChunkRow, DocumentRow

log = logging.getLogger(__name__)


class FaissIndex:
    """Обёртка над FAISS IndexFlatIP внутри IndexIDMap2: поддерживает
    add_with_ids и remove_ids. Векторы нормализованы на стороне эмбеддера,
    так что inner product == косинус.

    Размерность определяется автоматически: либо из сохранённого индекса при
    загрузке, либо из первого батча векторов при вставке. Это позволяет менять
    модель эмбеддингов через .env без правки кода."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._index: Optional[faiss.IndexIDMap2] = None

    def load_from_disk(self) -> bool:
        """Пробует загрузить сохранённый индекс. Возвращает True, если получилось."""
        with self._lock:
            if not self.path.exists():
                return False
            try:
                self._index = faiss.read_index(str(self.path))
                log.info(
                    "FAISS индекс загружен: %d векторов, dim=%d",
                    self._index.ntotal,
                    self._index.d,
                )
                return True
            except Exception as e:
                log.error("Не удалось загрузить FAISS индекс: %s", e)
                return False

    def _ensure_initialized(self, dim: int) -> faiss.IndexIDMap2:
        if self._index is None:
            base = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIDMap2(base)
            log.info("Создан новый FAISS индекс (dim=%d)", dim)
        elif self._index.d != dim:
            raise RuntimeError(
                f"Несовпадение размерности: индекс имеет dim={self._index.d}, "
                f"а векторы — dim={dim}. Удалите {self.path} или верните прежнюю модель."
            )
        return self._index

    def persist(self) -> None:
        with self._lock:
            if self._index is None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.path))

    def add(self, vectors: np.ndarray, ids: list[int]) -> None:
        if len(ids) == 0 or vectors.size == 0:
            return
        with self._lock:
            idx = self._ensure_initialized(int(vectors.shape[1]))
            ids_arr = np.asarray(ids, dtype=np.int64)
            idx.add_with_ids(vectors.astype(np.float32), ids_arr)

    def remove(self, ids: list[int]) -> int:
        if not ids:
            return 0
        with self._lock:
            if self._index is None:
                return 0
            return int(self._index.remove_ids(np.asarray(ids, dtype=np.int64)))

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
            if q.shape[1] != self._index.d:
                log.error(
                    "Запрос dim=%d не совпадает с индексом dim=%d",
                    q.shape[1],
                    self._index.d,
                )
                return []
            scores, ids = self._index.search(q, min(k, self._index.ntotal))
            out: list[tuple[int, float]] = []
            for i, s in zip(ids[0].tolist(), scores[0].tolist()):
                if i < 0:
                    continue
                out.append((int(i), float(s)))
            return out

    @property
    def size(self) -> int:
        with self._lock:
            return self._index.ntotal if self._index is not None else 0

    @property
    def dim(self) -> Optional[int]:
        with self._lock:
            return self._index.d if self._index is not None else None


faiss_index = FaissIndex(path=settings.faiss_path)


@dataclass
class SearchHit:
    chunk: ChunkRow
    document: DocumentRow
    score: float


class SearchService:
    """Высокоуровневый поиск: вопрос → эмбеддинг → FAISS → подтягивает чанки и
    документы из SQLite. Если включён BM25, объединяет результаты через RRF."""

    def __init__(self) -> None:
        self._bm25 = None
        self._bm25_ids: list[int] = []
        self._bm25_lock = threading.Lock()

    def _ensure_bm25(self) -> None:
        if not settings.SEARCH_USE_BM25:
            return
        with self._bm25_lock:
            if self._bm25 is not None:
                return
            try:
                from rank_bm25 import BM25Okapi
            except Exception as e:
                log.warning("rank_bm25 не установлен: %s", e)
                return
            data = db.get_all_chunk_ids_with_text()
            if not data:
                return
            self._bm25_ids = [i for i, _ in data]
            tokenized = [t.lower().split() for _, t in data]
            self._bm25 = BM25Okapi(tokenized)

    def invalidate_bm25(self) -> None:
        with self._bm25_lock:
            self._bm25 = None
            self._bm25_ids = []

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        owner_user_id: Optional[int] = None,
        notebook_id: Optional[int] = None,
    ) -> list[SearchHit]:
        """Векторный поиск с опциональной изоляцией по владельцу и ноутбуку.

        - owner_user_id: только чанки документов этого юзера (multi-user изоляция)
        - notebook_id: только чанки документов из этого ноутбука (workspace-изоляция)
        """
        k = k or settings.SEARCH_TOP_K
        if not query.strip():
            return []

        # Фильтрация по owner и/или notebook — over-fetch кратно больше из FAISS,
        # чтобы после фильтрации осталось k. Для 5000 векторов это всё ещё мс.
        any_filter = owner_user_id is not None or notebook_id is not None
        over_fetch = max(k * 6, 30) if any_filter else max(k, 15)

        q_vec = embedding_service.encode_query(query)
        dense_hits = faiss_index.search(q_vec, k=over_fetch)
        scored: dict[int, float] = {}
        ranked_dense = [cid for cid, _ in dense_hits]
        for rank, cid in enumerate(ranked_dense):
            scored[cid] = scored.get(cid, 0.0) + 1.0 / (60 + rank + 1)

        if settings.SEARCH_USE_BM25:
            self._ensure_bm25()
            if self._bm25 is not None and self._bm25_ids:
                tokens = query.lower().split()
                scores = self._bm25.get_scores(tokens)
                top_idx = np.argsort(scores)[::-1][:over_fetch]
                ranked_bm = [self._bm25_ids[i] for i in top_idx if scores[i] > 0]
                for rank, cid in enumerate(ranked_bm):
                    scored[cid] = scored.get(cid, 0.0) + 1.0 / (60 + rank + 1)

        ordered_ids = sorted(scored.keys(), key=lambda i: scored[i], reverse=True)
        if not ordered_ids:
            return []

        # Фильтр по владельцу — один SQL JOIN вместо N round-trips
        if owner_user_id is not None:
            owners = db.get_chunk_owners(ordered_ids)
            ordered_ids = [
                cid for cid in ordered_ids
                if owners.get(cid) == owner_user_id
            ]
        # Фильтр по ноутбуку — аналогично
        if notebook_id is not None:
            nb_map = db.get_chunk_notebooks(ordered_ids)
            ordered_ids = [
                cid for cid in ordered_ids
                if nb_map.get(cid) == notebook_id
            ]

        ordered_ids = ordered_ids[:k]
        if not ordered_ids:
            return []

        chunks = db.get_chunks_by_ids(ordered_ids)
        # Загружаем документы пакетом, чтобы не дёргать БД каждый раз
        doc_ids = list({c.document_id for c in chunks})
        docs: dict[int, DocumentRow] = {}
        for did in doc_ids:
            d = db.get_document(did)
            if d:
                docs[did] = d

        hits: list[SearchHit] = []
        dense_score_by_id = {cid: s for cid, s in dense_hits}
        for c in chunks:
            doc = docs.get(c.document_id)
            if doc is None:
                continue
            hits.append(
                SearchHit(
                    chunk=c,
                    document=doc,
                    score=dense_score_by_id.get(c.id, scored.get(c.id, 0.0)),
                )
            )
        return hits


search_service = SearchService()
