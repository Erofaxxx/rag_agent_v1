from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from auth.sessions import lookup_session
from storage import UserRow


# Состояние-нелегкое исключение, которое фронт может ловить и редиректить на /login
def _unauthorized(detail: str = "Не авторизован") -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


def _forbidden(detail: str = "Недостаточно прав") -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def get_current_user(request: Request) -> Optional[UserRow]:
    """Возвращает пользователя, если есть валидная сессия, иначе None.
    Использовать там, где аутентификация опциональна (не часто)."""
    user, _session, _token = lookup_session(request)
    return user


def optional_user(request: Request) -> Optional[UserRow]:
    return get_current_user(request)


def require_user(request: Request) -> UserRow:
    """Любой залогиненный и активный пользователь (admin или user)."""
    user = get_current_user(request)
    if user is None:
        raise _unauthorized()
    return user


def require_admin(user: UserRow = Depends(require_user)) -> UserRow:
    if user.role != "admin":
        raise _forbidden("Требуются права администратора")
    return user


# CSRF-защита: для cookie-аутентификации требуем явный заголовок X-Requested-With
# на любых state-changing запросах (POST/PUT/PATCH/DELETE). Браузер по умолчанию
# не выставляет этот заголовок при cross-site формах, и SameSite=Lax cookie не
# поедет на cross-origin POST. Двойная защита.
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def csrf_check(request: Request) -> None:
    if request.method.upper() in SAFE_METHODS:
        return
    # Если у запроса вообще нет cookie с сессией — скорее всего это легит-логин
    # /api/auth/login или подобное; для них выделим отдельную защиту через
    # rate-limit. Сюда не дойдёт всё равно — этот dependency прикладывается
    # точечно к уже-аутентифицированным ручкам.
    if request.headers.get("x-requested-with") != "fetch":
        raise _forbidden("Отсутствует CSRF-маркер (X-Requested-With: fetch)")
