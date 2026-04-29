import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field, field_validator
from slowapi import Limiter

from auth.dependencies import csrf_check, require_user
from auth.passwords import (
    hash_password,
    needs_rehash,
    validate_password_strength,
    verify_password,
)
from auth.sessions import (
    create_session,
    lookup_session,
    revoke_session,
    SESSION_COOKIE_NAME,
)
from config import settings
from storage import db, UserRow

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])


def _rate_limit_key(request: Request) -> str:
    """Берём IP из X-Forwarded-For если за прокси (Caddy/nginx). Иначе client.host.
    Это критично: без этого все запросы за прокси шерят один rate-limit."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


limiter = Limiter(key_func=_rate_limit_key)


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=256)
    display_name: Optional[str] = Field(default=None, max_length=120)

    @field_validator("display_name")
    @classmethod
    def _strip_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        return v or None


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=256)


class ChangePasswordIn(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=256)
    new_password: str = Field(..., min_length=1, max_length=256)


class UserOut(BaseModel):
    id: int
    email: str
    display_name: Optional[str]
    role: str
    is_active: bool
    created_at: str
    last_login_at: Optional[str]


def _to_user_out(u: UserRow) -> UserOut:
    return UserOut(
        id=u.id,
        email=u.email,
        display_name=u.display_name,
        role=u.role,
        is_active=u.is_active,
        created_at=u.created_at,
        last_login_at=u.last_login_at,
    )


def _client_ip(request: Request) -> Optional[str]:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


# ---------- register ----------

@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
@limiter.limit(f"{settings.RATE_LIMIT_REGISTER_PER_HOUR}/hour")
def register(request: Request, payload: RegisterIn) -> UserOut:
    if not settings.ALLOW_PUBLIC_REGISTRATION:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Регистрация отключена. Обратитесь к администратору.",
        )

    pwd_problem = validate_password_strength(payload.password)
    if pwd_problem:
        raise HTTPException(status_code=400, detail=pwd_problem)

    email = payload.email.strip().lower()
    if db.get_user_by_email(email) is not None:
        # Не сообщаем «пользователь существует» — это утечка наличия аккаунта.
        # Возвращаем 200/201 с тем же телом, как будто всё ок? Нет, лучше ошибка
        # без раскрытия. Используем generic-сообщение.
        raise HTTPException(status_code=400, detail="Не удалось создать аккаунт. Возможно, такой email уже зарегистрирован.")

    # Первый зарегистрированный пользователь становится активным админом
    # автоматически только если ADMIN_BOOTSTRAP_* НЕ задан и в БД ещё нет
    # ни одного юзера. Это запасной путь, если bootstrap пропустили.
    is_first_ever = db.count_users() == 0
    role = "admin" if is_first_ever else "user"
    is_active = is_first_ever

    user_id = db.create_user(
        email=email,
        password_hash=hash_password(payload.password),
        display_name=payload.display_name,
        role=role,
        is_active=is_active,
    )
    db.log_audit(
        event="register",
        user_id=user_id,
        ip_address=_client_ip(request),
        details=f"role={role} active={is_active}",
    )

    user = db.get_user(user_id)
    assert user is not None
    log.info(
        "Зарегистрирован пользователь id=%s email=%s role=%s active=%s",
        user_id, email, role, is_active,
    )
    return _to_user_out(user)


# ---------- login ----------

@router.post("/login", response_model=UserOut)
@limiter.limit(f"{settings.RATE_LIMIT_LOGIN_PER_MINUTE}/minute")
def login(request: Request, response: Response, payload: LoginIn) -> UserOut:
    ip = _client_ip(request)
    user = db.get_user_by_email(payload.email)

    # Чтобы не отличать «нет пользователя» от «неверный пароль» по timing,
    # всё равно прогоняем верификацию. Используем фиктивный валидный хеш.
    DUMMY_HASH = "$argon2id$v=19$m=19456,t=2,p=1$ZmFrZXNhbHRzaXplMTY$0YQjP3o0YQjP3o0YQjP3o0YQjP3o0YQjP3o0YQjP3o"

    if user is None:
        verify_password(payload.password, DUMMY_HASH)
        db.log_audit(event="login_fail", details=f"email={payload.email}", ip_address=ip)
        raise HTTPException(status_code=401, detail="Неверный email или пароль")

    # Проверка временного лока
    if user.locked_until:
        try:
            until = datetime.fromisoformat(user.locked_until)
            if until > datetime.now(timezone.utc):
                db.log_audit(event="login_fail", user_id=user.id, details="locked", ip_address=ip)
                remaining = int((until - datetime.now(timezone.utc)).total_seconds() / 60) + 1
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Аккаунт временно заблокирован. Попробуйте через {remaining} мин.",
                )
        except ValueError:
            db.update_user(user.id, clear_locked_until=True)

    ok = verify_password(payload.password, user.password_hash)
    if not ok:
        new_count = (user.failed_login_count or 0) + 1
        if new_count >= settings.LOGIN_MAX_FAILED_ATTEMPTS:
            until = (datetime.now(timezone.utc) + timedelta(minutes=settings.LOGIN_LOCKOUT_MINUTES)).isoformat()
            db.update_user(user.id, failed_login_count=new_count, locked_until=until)
            db.log_audit(event="lockout", user_id=user.id, ip_address=ip, details=f"attempts={new_count}")
            log.warning("Аккаунт %s заблокирован после %d попыток с %s", user.email, new_count, ip)
        else:
            db.update_user(user.id, failed_login_count=new_count)
        db.log_audit(event="login_fail", user_id=user.id, ip_address=ip, details=f"attempts={new_count}")
        raise HTTPException(status_code=401, detail="Неверный email или пароль")

    if not user.is_active:
        db.log_audit(event="login_fail", user_id=user.id, ip_address=ip, details="not_approved")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Аккаунт ожидает одобрения администратора.",
        )

    # Успех. Сбросим счётчик неудач, обновим last_login, перехешируем пароль
    # если параметры устарели.
    update_kwargs = dict(
        failed_login_count=0,
        clear_locked_until=True,
        last_login_at=datetime.now(timezone.utc).isoformat(),
        last_login_ip=ip,
    )
    if needs_rehash(user.password_hash):
        update_kwargs["password_hash"] = hash_password(payload.password)
    db.update_user(user.id, **update_kwargs)

    create_session(request, response, user_id=user.id)
    db.log_audit(event="login_success", user_id=user.id, ip_address=ip)
    log.info("Login: %s (id=%s) с %s", user.email, user.id, ip)

    fresh = db.get_user(user.id)
    assert fresh is not None
    return _to_user_out(fresh)


# ---------- logout ----------

@router.post("/logout", status_code=204, dependencies=[Depends(csrf_check)])
def logout(request: Request, response: Response) -> Response:
    user, _session, token = lookup_session(request)
    revoke_session(response, token)
    if user:
        db.log_audit(event="logout", user_id=user.id, ip_address=_client_ip(request))
    return Response(status_code=204)


# ---------- me ----------

@router.get("/me", response_model=UserOut)
def me(user: UserRow = Depends(require_user)) -> UserOut:
    return _to_user_out(user)


@router.post("/change-password", status_code=204, dependencies=[Depends(csrf_check)])
def change_password(
    request: Request,
    payload: ChangePasswordIn,
    user: UserRow = Depends(require_user),
) -> Response:
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Текущий пароль неверный")
    pwd_problem = validate_password_strength(payload.new_password)
    if pwd_problem:
        raise HTTPException(status_code=400, detail=pwd_problem)
    db.update_user(user.id, password_hash=hash_password(payload.new_password))
    # Инвалидируем все сессии пользователя кроме текущей? Проще — все,
    # пусть перелогинится. Безопаснее.
    db.delete_sessions_for_user(user.id)
    db.log_audit(event="password_change", user_id=user.id, ip_address=_client_ip(request))
    return Response(status_code=204)
