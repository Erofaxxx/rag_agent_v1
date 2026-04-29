import logging
import threading
from typing import Iterable

import numpy as np

from config import settings

log = logging.getLogger(__name__)


class EmbeddingService:
    """Singleton эмбеддера. По умолчанию BGE-M3 (~2 GB RAM, dim=1024).
    Меняется через EMBEDDING_MODEL в .env. Для e5-моделей задайте префиксы
    EMBEDDING_QUERY_PREFIX="query: " и EMBEDDING_PASSAGE_PREFIX="passage: "."""

    def __init__(self) -> None:
        self._model = None
        self._impl: str | None = None
        self._dim: int | None = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.load()
        assert self._dim is not None
        return self._dim

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            log.info("Загружаю модель эмбеддингов %s ...", settings.EMBEDDING_MODEL)

            # BGE-M3 имеет специфичный API; для остальных моделей берём
            # обычный sentence-transformers.
            if "bge-m3" in settings.EMBEDDING_MODEL.lower():
                try:
                    from FlagEmbedding import BGEM3FlagModel

                    self._model = BGEM3FlagModel(
                        settings.EMBEDDING_MODEL,
                        use_fp16=settings.EMBEDDING_USE_FP16,
                    )
                    self._impl = "flag"
                except Exception as e:
                    log.warning("FlagEmbedding не загрузился (%s), fallback на sentence-transformers", e)
                    self._load_sentence_transformer()
            else:
                self._load_sentence_transformer()

            # Определяем размерность реальным энкодом
            sample = self._encode_raw(["dim probe"])
            self._dim = int(sample.shape[1])
            log.info(
                "Модель эмбеддингов готова: %s (impl=%s, dim=%d)",
                settings.EMBEDDING_MODEL,
                self._impl,
                self._dim,
            )

    def _load_sentence_transformer(self) -> None:
        from sentence_transformers import SentenceTransformer

        kwargs = {}
        if settings.EMBEDDING_USE_FP16:
            try:
                import torch
                kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
            except Exception:
                pass
        self._model = SentenceTransformer(settings.EMBEDDING_MODEL, **kwargs)
        self._impl = "st"

    def _encode_raw(self, texts: list[str]) -> np.ndarray:
        if self._impl == "flag":
            out = self._model.encode(
                texts,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                max_length=settings.EMBEDDING_MAX_LENGTH,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            return np.asarray(out["dense_vecs"], dtype=np.float32)
        return self._model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def encode(
        self,
        texts: list[str] | Iterable[str],
        prefix: str = "",
    ) -> np.ndarray:
        if self._model is None:
            self.load()
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if prefix:
            texts = [prefix + t for t in texts]
        return self._normalize(self._encode_raw(texts))

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts, prefix=settings.EMBEDDING_PASSAGE_PREFIX)

    def encode_query(self, text: str) -> np.ndarray:
        vecs = self.encode([text], prefix=settings.EMBEDDING_QUERY_PREFIX)
        return vecs[0]

    # Совместимость со старым кодом (не указывает префикс)
    def encode_one(self, text: str) -> np.ndarray:
        return self.encode_query(text)


embedding_service = EmbeddingService()
