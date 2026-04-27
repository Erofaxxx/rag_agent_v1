import logging
import threading
from typing import Iterable

import numpy as np

from config import settings

log = logging.getLogger(__name__)


class EmbeddingService:
    """Синглтон BGE-M3. Загружается один раз при старте и держится в памяти.
    Энкодит батчами; возвращает L2-нормализованные векторы float32."""

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._dim = settings.EMBEDDING_DIM

    @property
    def dim(self) -> int:
        return self._dim

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            log.info("Загружаю модель эмбеддингов %s ...", settings.EMBEDDING_MODEL)
            try:
                from FlagEmbedding import BGEM3FlagModel

                self._model = BGEM3FlagModel(
                    settings.EMBEDDING_MODEL,
                    use_fp16=settings.EMBEDDING_USE_FP16,
                )
                self._impl = "flag"
            except Exception as e:
                log.warning("FlagEmbedding не загрузился (%s), пробую sentence-transformers", e)
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
                self._impl = "st"
            log.info("Модель эмбеддингов готова (impl=%s)", self._impl)

    def encode(self, texts: list[str] | Iterable[str]) -> np.ndarray:
        if self._model is None:
            self.load()

        texts = [t if isinstance(t, str) else str(t) for t in texts]
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        if self._impl == "flag":
            out = self._model.encode(
                texts,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                max_length=8192,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vecs = np.asarray(out["dense_vecs"], dtype=np.float32)
        else:
            vecs = self._model.encode(
                texts,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                normalize_embeddings=False,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)

        # Нормализуем до единичной длины — для FAISS IndexFlatIP это даёт косинус
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


embedding_service = EmbeddingService()
