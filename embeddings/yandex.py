import logging
import threading
import time
from typing import Optional

import numpy as np

from config import settings

log = logging.getLogger(__name__)


class YandexEmbeddingService:
    """Эмбеддинги через Yandex AI Studio (text-search-doc / text-search-query).

    Ключевая особенность — АСИММЕТРИЧНОСТЬ: модель для индексации (doc) и для
    запроса (query) РАЗНЫЕ. Перепутать = просадка качества retrieval. Поэтому
    публичный API сервиса разделён на encode_passages() и encode_query() и не
    позволяет их смешать.

    Лимит на длину текста ≈ 2048 токенов (≈ 6-8 тыс. русских символов). Длиннее
    — Yandex обрежет на сервере без предупреждения. Сервис сам урезает по
    YANDEX_EMBEDDING_MAX_CHARS и логирует warning.
    """

    DEFAULT_DIM = 256

    def __init__(self) -> None:
        self._sdk = None
        self._lock = threading.Lock()
        self._dim: Optional[int] = None
        # Throttle: минимальный интервал между вызовами (sec)
        self._min_interval = 60.0 / max(1, settings.YANDEX_EMBEDDING_RPM)
        self._last_call_ts = 0.0
        self._throttle_lock = threading.Lock()
        # Sanity: doc/query модели не должны совпадать (asymmetric retrieval)
        if (
            settings.YANDEX_EMBEDDING_DOC_MODEL
            == settings.YANDEX_EMBEDDING_QUERY_MODEL
        ):
            log.warning(
                "YANDEX_EMBEDDING_DOC_MODEL и QUERY_MODEL совпадают (%s). "
                "Yandex использует асимметричные модели — обычно %s vs %s. "
                "Проверьте .env, иначе качество поиска просядет.",
                settings.YANDEX_EMBEDDING_DOC_MODEL,
                "text-search-doc",
                "text-search-query",
            )

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.load()
        assert self._dim is not None
        return self._dim

    def load(self) -> None:
        """Инициализирует SDK и определяет размерность через probe-запрос."""
        with self._lock:
            if self._sdk is not None:
                return
            if not settings.YANDEX_API_KEY:
                raise RuntimeError(
                    "EMBEDDING_PROVIDER=yandex, но YANDEX_API_KEY не задан в .env. "
                    "Получите ключ: console.yandex.cloud → IAM → Сервисные аккаунты "
                    "→ создать с ролью ai.languageModels.user → API-ключ."
                )
            if not settings.YANDEX_FOLDER_ID:
                raise RuntimeError(
                    "EMBEDDING_PROVIDER=yandex, но YANDEX_FOLDER_ID не задан в .env."
                )
            try:
                from yandex_ai_studio_sdk import AIStudio
            except ImportError as e:
                raise RuntimeError(
                    "Пакет yandex-ai-studio-sdk не установлен. "
                    "pip install -r requirements.txt"
                ) from e

            self._sdk = AIStudio(
                folder_id=settings.YANDEX_FOLDER_ID,
                auth=settings.YANDEX_API_KEY,
            )
            log.info(
                "Yandex AI Studio SDK инициализирован (folder=%s, doc=%s, query=%s)",
                settings.YANDEX_FOLDER_ID[:8] + "...",
                settings.YANDEX_EMBEDDING_DOC_MODEL,
                settings.YANDEX_EMBEDDING_QUERY_MODEL,
            )

            # Probe: определяем dim через query-модель (вызов короткий и
            # дешёвый). Если dim в .env переопределён — используем его без probe.
            if settings.YANDEX_EMBEDDING_DIMENSIONS > 0:
                self._dim = settings.YANDEX_EMBEDDING_DIMENSIONS
                log.info("Yandex embeddings dim=%d (override из .env)", self._dim)
            else:
                vec = self._call_embed_raw(
                    "probe", settings.YANDEX_EMBEDDING_QUERY_MODEL
                )
                self._dim = int(vec.shape[0])
                log.info("Yandex embeddings dim=%d (probe)", self._dim)

    # ---- public API ----

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Эмбеддинг документов (для индексации). Использует doc-модель."""
        return self._encode_batch(texts, settings.YANDEX_EMBEDDING_DOC_MODEL, kind="doc")

    def encode_query(self, text: str) -> np.ndarray:
        """Эмбеддинг запроса. Использует query-модель."""
        vec = self._encode_batch([text], settings.YANDEX_EMBEDDING_QUERY_MODEL, kind="query")
        return vec[0]

    # Для совместимости со старым интерфейсом (не использовать в новом коде)
    def encode_one(self, text: str) -> np.ndarray:
        return self.encode_query(text)

    # ---- internals ----

    def _truncate(self, text: str) -> str:
        max_chars = settings.YANDEX_EMBEDDING_MAX_CHARS
        if len(text) > max_chars:
            log.warning(
                "Yandex embedding: текст %d символов > лимита %d, обрезаю. "
                "Возможно, чанкер настроен слишком агрессивно (CHUNK_SIZE=%d).",
                len(text),
                max_chars,
                settings.CHUNK_SIZE,
            )
            return text[:max_chars]
        return text

    def _encode_batch(self, texts: list[str], model_name: str, kind: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._sdk is None:
            self.load()

        vectors: list[np.ndarray] = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                text = str(text)
            text = self._truncate(text)
            v = self._call_embed_raw(text, model_name)
            vectors.append(v)
            if (i + 1) % 50 == 0:
                log.info(
                    "Yandex %s embed: %d/%d", kind, i + 1, len(texts)
                )

        arr = np.vstack(vectors).astype(np.float32)
        return self._normalize(arr)

    def _throttle(self) -> None:
        with self._throttle_lock:
            elapsed = time.time() - self._last_call_ts
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call_ts = time.time()

    def _call_embed_raw(self, text: str, model_name: str) -> np.ndarray:
        """Один HTTP-вызов к Yandex с retry/backoff. Возвращает 1D-вектор без
        нормализации (нормализация — на стороне batch-метода)."""
        assert self._sdk is not None
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                self._throttle()
                # SDK принимает 'doc' / 'query' либо полный URI; передаём
                # короткое имя — оно дефолтно резолвится в text-search-doc/query.
                # Если в .env пользователь задал кастомное имя (например, на
                # будущее), используем его.
                short = "doc" if "doc" in model_name else "query"
                m = self._sdk.models.text_embeddings(short)
                if settings.YANDEX_EMBEDDING_DIMENSIONS > 0:
                    m = m.configure(dimensions=settings.YANDEX_EMBEDDING_DIMENSIONS)
                res = m.run(text)
                vec = np.asarray(res.embedding, dtype=np.float32)
                if vec.ndim != 1:
                    raise RuntimeError(f"Неожиданная размерность от Yandex: {vec.shape}")
                return vec
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # Распознаём ошибки, которые имеет смысл ретраить
                retriable = any(
                    s in msg for s in ("429", "rate", "timeout", "503", "502", "unavail")
                )
                if attempt == 4 or not retriable:
                    raise
                wait = 2 ** attempt
                log.warning(
                    "Yandex embed retry #%d через %ds (модель=%s): %s",
                    attempt + 1, wait, model_name, e,
                )
                time.sleep(wait)
        # Сюда дойдём только если raise не сработал — для типизации
        raise RuntimeError(f"Yandex embed failed: {last_err}")

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms
