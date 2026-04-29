#!/usr/bin/env python3
"""Переиндексация всех чанков из SQLite в FAISS текущим EMBEDDING_PROVIDER.

Зачем нужно:
- Сменили EMBEDDING_PROVIDER (например, bge → yandex). Размерность вектора
  меняется (BGE-M3=1024, Yandex=256), старый FAISS-индекс становится
  несовместим.
- Сменили модель внутри одного провайдера (например, BGE-M3 → e5-small).
- Хотите перепрогнать корпус через новую модель для A/B-теста качества.

Использование:
    cd /opt/rag_agent_v1
    .venv/bin/python -m scripts.reindex                # с интерактивным confirm
    .venv/bin/python -m scripts.reindex --yes          # без подтверждения
    .venv/bin/python -m scripts.reindex --batch 50     # размер батча

Файлы документов в data/uploads/ при этом не трогаются — только эмбеддинги.
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Корень проекта в sys.path, чтобы запуск работал и через `python scripts/reindex.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Не прогружаем эмбеддер-синглтон при старте main.py — он сам инициализируется
os.environ.setdefault("PRELOAD_EMBEDDINGS", "0")

from config import settings  # noqa: E402
from embeddings import embedding_service  # noqa: E402
from search import faiss_index  # noqa: E402
from storage import db  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reindex")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", "-y", action="store_true", help="Не спрашивать подтверждение")
    parser.add_argument("--batch", type=int, default=32, help="Размер батча (default 32)")
    args = parser.parse_args()

    chunks = db.get_all_chunk_ids_with_text()
    if not chunks:
        log.info("В БД нет чанков для переиндексации.")
        return 0

    log.info(
        "Найдено %d чанков. Провайдер эмбеддингов: %s",
        len(chunks),
        settings.EMBEDDING_PROVIDER,
    )
    if settings.EMBEDDING_PROVIDER == "yandex":
        log.info(
            "Yandex: doc=%s query=%s dim=%s",
            settings.YANDEX_EMBEDDING_DOC_MODEL,
            settings.YANDEX_EMBEDDING_QUERY_MODEL,
            settings.YANDEX_EMBEDDING_DIMENSIONS or "default",
        )

    if not args.yes:
        ans = input("Старый FAISS-индекс будет удалён, всё перепрогонится. Продолжить? [y/N]: ").strip().lower()
        if ans not in ("y", "yes", "д", "да"):
            log.info("Отменено пользователем.")
            return 1

    # Удаляем старый индекс, создадим новый под актуальную dim
    if settings.faiss_path.exists():
        log.info("Удаляю старый индекс: %s", settings.faiss_path)
        settings.faiss_path.unlink()
    faiss_index._index = None  # сброс ленивого состояния  # noqa: SLF001

    log.info("Прогружаю модель эмбеддингов...")
    embedding_service.load()
    log.info("Размерность: %d", embedding_service.dim)

    started = time.time()
    total = len(chunks)
    batch_size = max(1, args.batch)
    done = 0
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        ids = [cid for cid, _ in batch]
        texts = [text for _, text in batch]
        try:
            vectors = embedding_service.encode_passages(texts)
        except Exception as e:
            log.exception("Ошибка эмбеддинга батча %d-%d: %s", i, i + len(batch), e)
            return 2
        faiss_index.add(vectors, ids)
        done += len(batch)
        elapsed = time.time() - started
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        log.info(
            "%d/%d чанков (%.1f c/чанк, ETA %.0fs)",
            done, total, 1.0 / rate if rate > 0 else 0, eta,
        )

    faiss_index.persist()
    log.info(
        "Готово: %d векторов в индексе за %.0fs. Файл: %s",
        faiss_index.size,
        time.time() - started,
        settings.faiss_path,
    )
    log.info("Перезапустите сервис: sudo systemctl restart rag-agent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
