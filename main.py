import logging
import logging.handlers
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from api.admin import router as admin_router
from api.chat import router as chat_router
from api.conversations import router as conversations_router
from api.documents import router as documents_router
from auth import auth_router
from auth.dependencies import optional_user
from auth.passwords import hash_password
from auth.router import limiter
from config import settings
from embeddings import embedding_service
from search import faiss_index
from storage import db


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


def bootstrap_admin() -> None:
    """Создаёт первого админа из ADMIN_BOOTSTRAP_* при первом запуске."""
    if not settings.ADMIN_BOOTSTRAP_EMAIL or not settings.ADMIN_BOOTSTRAP_PASSWORD:
        return
    if db.count_users() > 0:
        # Если кто-то уже есть, не трогаем — иначе риск перезаписи продакшен-юзера
        return
    if len(settings.ADMIN_BOOTSTRAP_PASSWORD) < settings.PASSWORD_MIN_LENGTH:
        log.warning(
            "ADMIN_BOOTSTRAP_PASSWORD слишком короткий (< %d). Админ НЕ создан.",
            settings.PASSWORD_MIN_LENGTH,
        )
        return
    user_id = db.create_user(
        email=settings.ADMIN_BOOTSTRAP_EMAIL,
        password_hash=hash_password(settings.ADMIN_BOOTSTRAP_PASSWORD),
        display_name="Administrator",
        role="admin",
        is_active=True,
    )
    db.log_audit(event="register", user_id=user_id, details="bootstrap_admin")
    log.warning(
        "Создан bootstrap-админ %s (id=%s). СМЕНИТЕ пароль через UI и удалите "
        "ADMIN_BOOTSTRAP_PASSWORD из .env!",
        settings.ADMIN_BOOTSTRAP_EMAIL,
        user_id,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Поднимаем RAG Agent")
    log.info("Data dir: %s", settings.data_path)
    bootstrap_admin()
    db.cleanup_expired_sessions()
    faiss_index.load_from_disk()
    if os.environ.get("PRELOAD_EMBEDDINGS", "1") == "1":
        log.info("Прогружаю модель эмбеддингов в память (займёт минуту при первом старте)...")
        embedding_service.load()
    log.info(
        "Готово: %d векторов в индексе (model=%s), пользователей в БД: %d",
        faiss_index.size,
        settings.EMBEDDING_MODEL,
        db.count_users(),
    )
    yield
    log.info("Останавливаюсь")


app = FastAPI(
    title="RAG Agent",
    description="Корпоративный RAG-помощник по документам",
    version="2.0.0",
    lifespan=lifespan,
)


# ===== Security middleware =====

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers по best-practice OWASP. CSP позволяет CDN для marked.js
    и Google Fonts; всё остальное — same-origin."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Защита от MIME sniffing
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        # Защита от clickjacking
        response.headers.setdefault("X-Frame-Options", "DENY")
        # Referrer не утекает за пределы origin
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        # Запретить geolocation/microphone/camera по умолчанию
        response.headers.setdefault(
            "Permissions-Policy",
            "geolocation=(), microphone=(), camera=(), payment=()",
        )
        # CSP
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com data:; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "form-action 'self'; "
            "base-uri 'self'",
        )
        # HSTS — только если за HTTPS прокси (Caddy в проде)
        if settings.SESSION_COOKIE_SECURE:
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )
        return response


app.add_middleware(SecurityHeadersMiddleware)

# CORS — по умолчанию same-origin only. Если заданы CORS_ORIGINS, добавим их.
cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "X-Requested-With"],
    )

# Rate limiter (slowapi)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Слишком много запросов, попробуйте позже"},
    )


# ===== Routers =====

app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(admin_router)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "documents": db.count_documents(),
        "users": db.count_users(),
        "faiss_vectors": faiss_index.size,
    }


# ===== Static frontend =====

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _serve_html(name: str) -> FileResponse:
    return FileResponse(
        os.path.join(STATIC_DIR, name),
        media_type="text/html; charset=utf-8",
    )


@app.get("/")
def index(user=Depends(optional_user)):
    # Без сессии — на /login. С сессией — главная страница чата.
    if user is None:
        return RedirectResponse(url="/login", status_code=302)
    if not user.is_active:
        return RedirectResponse(url="/pending", status_code=302)
    return _serve_html("index.html")


@app.get("/login")
def login_page() -> FileResponse:
    return _serve_html("login.html")


@app.get("/register")
def register_page() -> FileResponse:
    if not settings.ALLOW_PUBLIC_REGISTRATION:
        return RedirectResponse(url="/login", status_code=302)
    return _serve_html("register.html")


@app.get("/pending")
def pending_page() -> FileResponse:
    return _serve_html("pending.html")


@app.get("/admin")
def admin_page(user=Depends(optional_user)):
    if user is None:
        return RedirectResponse(url="/login", status_code=302)
    if user.role != "admin":
        return RedirectResponse(url="/", status_code=302)
    return _serve_html("admin.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        workers=1,
    )
