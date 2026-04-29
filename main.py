import logging
import logging.handlers
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.auth import require_auth
from api.chat import router as chat_router
from api.conversations import router as conversations_router
from api.documents import router as documents_router
from config import settings
from embeddings import embedding_service
from search import faiss_index


def setup_logging() -> None:
    logs_dir = settings.logs_path
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "rag.log"

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        root.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(formatter)
    root.addHandler(fh)


setup_logging()
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Поднимаем RAG Agent")
    log.info("Data dir: %s", settings.data_path)
    # Сначала пробуем загрузить сохранённый FAISS-индекс с диска (если есть —
    # размерность придёт из файла). Если файла нет, индекс создастся при первой
    # вставке векторов с размерностью реальной модели.
    faiss_index.load_from_disk()
    if os.environ.get("PRELOAD_EMBEDDINGS", "1") == "1":
        log.info("Прогружаю модель эмбеддингов в память (займёт минуту при первом старте)...")
        embedding_service.load()
    log.info(
        "Готово: %d векторов в индексе (model=%s)",
        faiss_index.size,
        settings.EMBEDDING_MODEL,
    )
    yield
    log.info("Останавливаюсь")


app = FastAPI(
    title="RAG Agent",
    description="Корпоративный RAG-помощник по документам (BGE-M3 + FAISS + DeepSeek)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(conversations_router)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "documents": len(__import__("storage").db.list_documents()),
        "faiss_vectors": faiss_index.size,
    }


# --- Статика и индексная страница ---

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index(_: str = Depends(require_auth)) -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        workers=1,
    )
