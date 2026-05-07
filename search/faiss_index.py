import logging
import re
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

    @staticmethod
    def _tokenize_bm25(text: str) -> list[str]:
        """Простая нормализация: \\w+ ловит русские/латинские слова и цифры,
        отбрасывая знаки препинания. .split() этого не делал — пунктуация
        прилипала к токенам и BM25 промахивался."""
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

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
            tokenized = [self._tokenize_bm25(t) for _, t in data]
            self._bm25 = BM25Okapi(tokenized)

    def warmup(self) -> None:
        """Вызывается из lifespan на старте: строит BM25 заранее, чтобы
        первый запрос после рестарта не платил O(N) и не получал
        thundering-herd из N параллельных запросов, ждущих один лок."""
        self._ensure_bm25()

    def invalidate_bm25(self) -> None:
        with self._bm25_lock:
            self._bm25 = None
            self._bm25_ids = []

    # ---- внутренние kernels ----

    def _bm25_top(self, query: str, k: int) -> list[int]:
        """BM25 top-k id (только если включён и индекс собран)."""
        if not settings.SEARCH_USE_BM25:
            return []
        self._ensure_bm25()
        # Под локом снимаем локальные ссылки. Иначе invalidate_bm25() из
        # параллельного потока (после удаления документа) может обнулить
        # self._bm25 между проверкой None и get_scores → AttributeError.
        with self._bm25_lock:
            bm25 = self._bm25
            ids = list(self._bm25_ids)
        if bm25 is None or not ids:
            return []
        tokens = self._tokenize_bm25(query)
        if not tokens:
            return []
        scores = bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [ids[i] for i in top_idx if scores[i] > 0]

    def _dense_top(self, query: str, k: int, hyde_text: Optional[str] = None) -> list[tuple[int, float]]:
        """Dense top-k id со score. Если задан hyde_text, эмбеддим запрос +
        гипотетический ответ и берём усреднённый L2-нормализованный вектор —
        это даёт точку, близкую и к запросу, и к нужному фрагменту."""
        q_vec = embedding_service.encode_query(query)
        if hyde_text:
            try:
                h_vec = embedding_service.encode_query(hyde_text)
                vec = q_vec + h_vec
                n = float(np.linalg.norm(vec))
                if n > 0:
                    vec = vec / n
                q_vec = vec.astype(np.float32)
            except Exception as e:
                log.debug("HyDE embedding не удался (%s), используем чистый query", e)
        return faiss_index.search(q_vec, k=k)

    def _filter_by_scope(
        self,
        ordered_ids: list[int],
        owner_user_id: Optional[int],
        notebook_id: Optional[int],
    ) -> list[int]:
        if owner_user_id is not None:
            owners = db.get_chunk_owners(ordered_ids)
            ordered_ids = [cid for cid in ordered_ids if owners.get(cid) == owner_user_id]
        if notebook_id is not None:
            nb_map = db.get_chunk_notebooks(ordered_ids)
            ordered_ids = [cid for cid in ordered_ids if nb_map.get(cid) == notebook_id]
        return ordered_ids

    def _hits_for_ids(
        self,
        ordered_ids: list[int],
        score_by_id: dict[int, float],
    ) -> list[SearchHit]:
        if not ordered_ids:
            return []
        chunks = db.get_chunks_by_ids(ordered_ids)
        doc_ids = list({c.document_id for c in chunks})
        docs: dict[int, DocumentRow] = {}
        for did in doc_ids:
            d = db.get_document(did)
            if d:
                docs[did] = d
        out: list[SearchHit] = []
        # Сохраняем порядок ordered_ids
        chunks_by_id = {c.id: c for c in chunks}
        for cid in ordered_ids:
            c = chunks_by_id.get(cid)
            if c is None:
                continue
            doc = docs.get(c.document_id)
            if doc is None:
                continue
            out.append(SearchHit(chunk=c, document=doc, score=score_by_id.get(cid, 0.0)))
        return out

    # ---- public search ----

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        owner_user_id: Optional[int] = None,
        notebook_id: Optional[int] = None,
        *,
        ablation_mode: bool = False,
    ) -> list[SearchHit]:
        """Главная точка входа. Конвейер:

            entity-detection → (BM25-only short-circuit)
            или: multi-query + HyDE → dense + BM25 → RRF
            → фильтр по scope → reformulate-on-low-score (опц.)
            → rerank top-N → top-K

        Все слои опциональны и контролируются настройками. Если всё выключить,
        получим прежний dense-only поиск.

        ablation_mode=True — «голый» режим для L3 query-ablation детектора:
        отключает multi-query, HyDE и reformulate-on-low (они размывают эффект
        одиночной ablation), оставляет dense+BM25+RRF+rerank. Используется
        изнутри defenses/l3_query_ablation, обычные вызовы должны передавать
        False (по умолчанию).
        """
        from search.entity_detector import adaptive_top_k, detect_entities, detect_intent
        from search.query_expansion import hyde, multi_query, reformulate
        from search.reranker import rerank

        if not query.strip():
            return []

        # Short-circuit: ничего не индексировано — все слои выше бессмысленны,
        # и не хочется тратить LLM/embedding запросы на пустой корпус.
        if faiss_index.size == 0 and not settings.SEARCH_USE_BM25:
            return []

        base_k = k or settings.SEARCH_TOP_K
        if k is None and settings.SEARCH_ADAPTIVE_K:
            base_k = adaptive_top_k(query, settings.SEARCH_TOP_K)

        # Entity-heavy → BM25-only. Embedding на «2024-Q1 NDS-3401» не работает,
        # точное совпадение токена даёт BM25.
        profile = detect_entities(query)
        if (
            settings.SEARCH_ENTITY_BM25_FALLBACK
            and settings.SEARCH_USE_BM25
            and profile.is_entity_heavy
        ):
            log.info("Entity-heavy запрос %r → BM25-only path", query)
            hits = self._search_bm25_only(query, base_k, owner_user_id, notebook_id)
            return rerank(query, hits, base_k) if hits else hits

        # Multi-query + HyDE — выключаем в ablation_mode: caller хочет видеть
        # эффект удаления одного слова, а multi-query восстановит «забытый»
        # триггер через перефразировку и спрячет сигнатуру.
        intent = detect_intent(query)
        queries = [query]
        if (
            not ablation_mode
            and settings.SEARCH_MULTI_QUERY
            and settings.SEARCH_MULTI_QUERY_COUNT > 1
        ):
            queries = multi_query(query, n_extra=settings.SEARCH_MULTI_QUERY_COUNT - 1)
            if not queries:
                queries = [query]
        hyde_text: Optional[str] = None
        if not ablation_mode and settings.SEARCH_HYDE and intent == "definition":
            hyde_text = hyde(query)

        hits = self._search_with_queries(
            queries=queries,
            hyde_text=hyde_text,
            base_k=base_k,
            owner_user_id=owner_user_id,
            notebook_id=notebook_id,
        )

        # Reformulate-on-low тоже отключаем в ablation_mode — оно ссылается на
        # «hint terms» из исходных hits и может вернуть obfuscated query, что
        # снова восстановит триггер.
        if (
            not ablation_mode
            and settings.SEARCH_REFORMULATE_ON_LOW
            and hits
            and _avg_top_score(hits, n=3) < settings.SEARCH_LOW_SCORE_THRESHOLD
        ):
            hint_terms = _extract_hint_terms(hits)
            new_q = reformulate(query, hint_terms=hint_terms)
            if new_q:
                log.info("Низкий score %.3f → переформулировка %r → %r",
                         _avg_top_score(hits, n=3), query, new_q)
                alt_hits = self._search_with_queries(
                    queries=[new_q],
                    hyde_text=None,
                    base_k=base_k,
                    owner_user_id=owner_user_id,
                    notebook_id=notebook_id,
                )
                if alt_hits and _avg_top_score(alt_hits, n=3) > _avg_top_score(hits, n=3):
                    hits = alt_hits

        # Реранк top-N → top-K. В ablation_mode тоже реранкаем, потому что
        # top-k из L3-ablation сравниваются с top-k оригинального запроса
        # (тоже после rerank) — должно быть в одной шкале.
        return rerank(query, hits, base_k)

    def _search_with_queries(
        self,
        queries: list[str],
        hyde_text: Optional[str],
        base_k: int,
        owner_user_id: Optional[int],
        notebook_id: Optional[int],
    ) -> list[SearchHit]:
        """Прогоняет N запросов через dense + BM25, объединяет всё через RRF.
        Возвращает top-N кандидатов (RERANKER_CANDIDATES) для последующего реранка."""
        any_filter = owner_user_id is not None or notebook_id is not None
        # Сколько брать из каждого источника. Над фильтром берём кратно больше,
        # потому что часть отвалится после scope-фильтра.
        per_source_k = max(base_k * 6, 30) if any_filter else max(base_k, 15)
        candidate_n = max(settings.RERANKER_CANDIDATES, base_k)

        # RRF-агрегация по всем (query, источник) парам.
        rrf: dict[int, float] = {}
        # Для совместимости с прежним поведением — лучший dense score сохраняем
        # отдельно, чтобы класть в SearchHit.score (его видит UI и LLM).
        dense_score_by_id: dict[int, float] = {}

        for qi, q in enumerate(queries):
            # Первый запрос — потенциально с HyDE; остальные — без, чтобы не
            # тратить лишние эмбеддинги.
            ht = hyde_text if qi == 0 else None
            dense_hits = self._dense_top(q, per_source_k, hyde_text=ht)
            for rank, (cid, sc) in enumerate(dense_hits):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60 + rank + 1)
                if sc > dense_score_by_id.get(cid, -1.0):
                    dense_score_by_id[cid] = sc
            for rank, cid in enumerate(self._bm25_top(q, per_source_k)):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60 + rank + 1)

        if not rrf:
            return []

        ordered = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)
        ordered = self._filter_by_scope(ordered, owner_user_id, notebook_id)
        ordered = ordered[:candidate_n]

        score_by_id = {cid: dense_score_by_id.get(cid, rrf.get(cid, 0.0)) for cid in ordered}
        return self._hits_for_ids(ordered, score_by_id)

    def _search_bm25_only(
        self,
        query: str,
        base_k: int,
        owner_user_id: Optional[int],
        notebook_id: Optional[int],
    ) -> list[SearchHit]:
        any_filter = owner_user_id is not None or notebook_id is not None
        per_source_k = max(base_k * 6, 30) if any_filter else max(base_k, 15)
        ids = self._bm25_top(query, per_source_k)
        if not ids:
            return []
        ids = self._filter_by_scope(ids, owner_user_id, notebook_id)
        ids = ids[:max(settings.RERANKER_CANDIDATES, base_k)]
        # У BM25 свои безразмерные scores; для UI выводим RRF-rank-derived score.
        score_by_id = {cid: 1.0 / (i + 1) for i, cid in enumerate(ids)}
        return self._hits_for_ids(ids, score_by_id)


def _avg_top_score(hits: list[SearchHit], n: int = 3) -> float:
    if not hits:
        return 0.0
    top = hits[:n]
    return sum(h.score for h in top) / max(1, len(top))


_TERM_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9\-]{3,}")


def _extract_hint_terms(hits: list[SearchHit]) -> list[str]:
    """Достаёт отличающиеся слова из найденных чанков как hint для reformulate.
    Это даёт LLM словарь, под который реально что-то находится в индексе."""
    terms: dict[str, int] = {}
    for h in hits[:5]:
        for m in _TERM_RE.finditer(h.chunk.text[:1000]):
            t = m.group(0)
            if t.isnumeric():
                continue
            terms[t] = terms.get(t, 0) + 1
    # Возьмём топ-10 по частоте
    return [t for t, _ in sorted(terms.items(), key=lambda p: p[1], reverse=True)[:10]]


search_service = SearchService()
