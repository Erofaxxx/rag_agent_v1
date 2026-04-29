import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Request, Response

from config import settings
from storage import db, SessionRow, UserRow

SESSION_COOKIE_NAME = settings.SESSION_COOKIE_NAME

# Длина case-sensitive токена в cookie. 32 байта = 256 бит. token_urlsafe
# даёт base64-без-padding, поэтому строка получается ~43 символа.
TOKEN_BYTES = 32


def _hash_token(token: str) -> str:
    """SHA-256 от cookie value. В БД храним только хеш, чтобы кража БД не давала
    напрямую пригодные cookie."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _client_ip(request: Request) -> Optional[str]:
    # X-Forwarded-For выставляет Caddy/nginx. Берём первый IP в цепочке.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


def _set_cookie(response: Response, token: str, max_age_seconds: int) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=max_age_seconds,
        httponly=True,
        secure=settings.SESSION_COOKIE_SECURE,
        samesite=settings.SESSION_COOKIE_SAMESITE,
        path="/",
    )


def create_session(
    request: Request,
    response: Response,
    user_id: int,
) -> str:
    """Генерит токен, кладёт хеш в БД, ставит httpOnly cookie. Возвращает
    raw-токен (на случай тестов; сервер обычно полагается на cookie)."""
    token = secrets.token_urlsafe(TOKEN_BYTES)
    token_hash = _hash_token(token)
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=settings.SESSION_LIFETIME_DAYS)
    db.create_session(
        token_hash=token_hash,
        user_id=user_id,
        expires_at=expires.isoformat(),
        ip_address=_client_ip(request),
        user_agent=(request.headers.get("user-agent") or "")[:512],
    )
    _set_cookie(response, token, max_age_seconds=settings.SESSION_LIFETIME_DAYS * 86400)
    return token


def revoke_session(response: Response, token: Optional[str]) -> None:
    if token:
        db.delete_session(_hash_token(token))
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        secure=settings.SESSION_COOKIE_SECURE,
        samesite=settings.SESSION_COOKIE_SAMESITE,
        httponly=True,
    )


def revoke_all_sessions_for_user(user_id: int) -> int:
    return db.delete_sessions_for_user(user_id)


def lookup_session(request: Request) -> tuple[Optional[UserRow], Optional[SessionRow], Optional[str]]:
    """Возвращает (user, session, raw_token). Если cookie нет / просрочена /
    пользователь неактивен — (None, None, None)."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None, None, None
    token_hash = _hash_token(token)
    session = db.get_session(token_hash)
    if session is None:
        return None, None, None

    # Проверка срока
    try:
        expires = datetime.fromisoformat(session.expires_at)
    except ValueError:
        db.delete_session(token_hash)
        return None, None, None
    if expires < datetime.now(timezone.utc):
        db.delete_session(token_hash)
        return None, None, None

    user = db.get_user(session.user_id)
    if user is None or not user.is_active:
        # Пользователь удалён или деактивирован — сессия больше не валидна.
        db.delete_session(token_hash)
        return None, None, None

    # Sliding session: обновляем last_seen, но не каждый раз — иначе много write
    # на горячих ручках. Раз в 5 минут хватит.
    try:
        last_seen = datetime.fromisoformat(session.last_seen_at)
        if (datetime.now(timezone.utc) - last_seen).total_seconds() > 300:
            db.touch_session(token_hash, ip_address=_client_ip(request))
    except ValueError:
        db.touch_session(token_hash, ip_address=_client_ip(request))

    return user, session, token


def cleanup_expired() -> int:
    return db.cleanup_expired_sessions()
