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
    так что inner product == косинус."""

    def __init__(self, dim: int, path: Path) -> None:
        self.dim = dim
        self.path = path
        self._lock = threading.RLock()
        self._index: Optional[faiss.IndexIDMap2] = None

    def load_or_create(self) -> None:
        with self._lock:
            if self.path.exists():
                try:
                    self._index = faiss.read_index(str(self.path))
                    log.info(
                        "FAISS индекс загружен: %d векторов, dim=%d",
                        self._index.ntotal,
                        self.dim,
                    )
                    return
                except Exception as e:
                    log.error("Не удалось загрузить FAISS индекс: %s. Создаю новый.", e)
            base = faiss.IndexFlatIP(self.dim)
            self._index = faiss.IndexIDMap2(base)
            log.info("Создан новый пустой FAISS индекс (dim=%d)", self.dim)

    def _ensure(self) -> faiss.IndexIDMap2:
        if self._index is None:
            self.load_or_create()
        assert self._index is not None
        return self._index

    def persist(self) -> None:
        with self._lock:
            idx = self._ensure()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(idx, str(self.path))

    def add(self, vectors: np.ndarray, ids: list[int]) -> None:
        if len(ids) == 0:
            return
        with self._lock:
            idx = self._ensure()
            ids_arr = np.asarray(ids, dtype=np.int64)
            idx.add_with_ids(vectors.astype(np.float32), ids_arr)

    def remove(self, ids: list[int]) -> int:
        if not ids:
            return 0
        with self._lock:
            idx = self._ensure()
            removed = idx.remove_ids(np.asarray(ids, dtype=np.int64))
            return int(removed)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        with self._lock:
            idx = self._ensure()
            if idx.ntotal == 0:
                return []
            q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
            scores, ids = idx.search(q, min(k, idx.ntotal))
            out: list[tuple[int, float]] = []
            for i, s in zip(ids[0].tolist(), scores[0].tolist()):
                if i < 0:
                    continue
                out.append((int(i), float(s)))
            return out

    @property
    def size(self) -> int:
        with self._lock:
            return self._ensure().ntotal


faiss_index = FaissIndex(dim=settings.EMBEDDING_DIM, path=settings.faiss_path)


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

    def search(self, query: str, k: Optional[int] = None) -> list[SearchHit]:
        k = k or settings.SEARCH_TOP_K
        if not query.strip():
            return []

        q_vec = embedding_service.encode_one(query)
        dense_hits = faiss_index.search(q_vec, k=max(k, 15))
        scored: dict[int, float] = {}
        ranked_dense = [cid for cid, _ in dense_hits]
        for rank, cid in enumerate(ranked_dense):
            scored[cid] = scored.get(cid, 0.0) + 1.0 / (60 + rank + 1)

        if settings.SEARCH_USE_BM25:
            self._ensure_bm25()
            if self._bm25 is not None and self._bm25_ids:
                tokens = query.lower().split()
                scores = self._bm25.get_scores(tokens)
                top_idx = np.argsort(scores)[::-1][: max(k, 15)]
                ranked_bm = [self._bm25_ids[i] for i in top_idx if scores[i] > 0]
                for rank, cid in enumerate(ranked_bm):
                    scored[cid] = scored.get(cid, 0.0) + 1.0 / (60 + rank + 1)

        ordered_ids = sorted(scored.keys(), key=lambda i: scored[i], reverse=True)[:k]
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
