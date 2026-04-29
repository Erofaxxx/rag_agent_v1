"""Фабрика эмбеддинг-сервисов. Провайдер выбирается через EMBEDDING_PROVIDER:

- 'yandex' (по умолчанию) — Yandex AI Studio text-search-doc/query.
  Требует YANDEX_API_KEY и YANDEX_FOLDER_ID в .env.
- 'bge' — локальный BGE-M3 / multilingual-e5 через sentence-transformers.
  Требует доустановки: pip install -r requirements-bge-fallback.txt

Контракт обоих сервисов:
    encode_passages(list[str]) -> np.ndarray   # для индексации
    encode_query(str) -> np.ndarray             # для поиска
    .dim -> int                                 # размерность вектора
    .load() -> None                             # ленивая инициализация
"""
import logging

from config import settings

log = logging.getLogger(__name__)


def _make_service():
    provider = (settings.EMBEDDING_PROVIDER or "yandex").lower()
    if provider == "yandex":
        from embeddings.yandex import YandexEmbeddingService
        return YandexEmbeddingService()
    if provider == "bge":
        try:
            from embeddings.bge_m3 import EmbeddingService
        except ImportError as e:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=bge требует доустановки локальных моделей. "
                "Запустите: pip install -r requirements-bge-fallback.txt"
            ) from e
        return EmbeddingService()
    raise ValueError(
        f"Неизвестный EMBEDDING_PROVIDER='{provider}'. Допустимы: yandex, bge"
    )


embedding_service = _make_service()


# Re-exports для совместимости со старым кодом
__all__ = ["embedding_service"]
