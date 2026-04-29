import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from auth.dependencies import csrf_check, require_admin
from auth.passwords import hash_password, validate_password_strength
from auth.sessions import revoke_all_sessions_for_user
from storage import db, UserRow

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin), Depends(csrf_check)],
)


class UserAdminOut(BaseModel):
    id: int
    email: str
    display_name: Optional[str]
    role: str
    is_active: bool
    failed_login_count: int
    locked_until: Optional[str]
    created_at: str
    last_login_at: Optional[str]
    last_login_ip: Optional[str]


def _to_admin_out(u: UserRow) -> UserAdminOut:
    return UserAdminOut(
        id=u.id,
        email=u.email,
        display_name=u.display_name,
        role=u.role,
        is_active=u.is_active,
        failed_login_count=u.failed_login_count,
        locked_until=u.locked_until,
        created_at=u.created_at,
        last_login_at=u.last_login_at,
        last_login_ip=u.last_login_ip,
    )


def _client_ip(request: Request) -> Optional[str]:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


# ---------- list / get ----------

@router.get("/users", response_model=list[UserAdminOut])
def list_users(_actor: UserRow = Depends(require_admin)) -> list[UserAdminOut]:
    return [_to_admin_out(u) for u in db.list_users()]


@router.get("/users/{user_id}", response_model=UserAdminOut)
def get_user(user_id: int, _actor: UserRow = Depends(require_admin)) -> UserAdminOut:
    u = db.get_user(user_id)
    if u is None:
        raise HTTPException(404, "Пользователь не найден")
    return _to_admin_out(u)


# ---------- approve / reject ----------

class ApproveIn(BaseModel):
    role: str = Field(default="user", pattern=r"^(admin|user)$")


@router.post("/users/{user_id}/approve", response_model=UserAdminOut)
def approve_user(
    user_id: int,
    payload: ApproveIn,
    request: Request,
    actor: UserRow = Depends(require_admin),
) -> UserAdminOut:
    u = db.get_user(user_id)
    if u is None:
        raise HTTPException(404, "Пользователь не найден")
    db.update_user(user_id, is_active=True, role=payload.role, failed_login_count=0, clear_locked_until=True)
    db.log_audit(
        event="approve",
        user_id=user_id,
        actor_user_id=actor.id,
        details=f"role={payload.role}",
        ip_address=_client_ip(request),
    )
    log.info("User %s одобрен админом %s, role=%s", u.email, actor.email, payload.role)
    fresh = db.get_user(user_id)
    assert fresh is not None
    return _to_admin_out(fresh)


@router.post("/users/{user_id}/deactivate", response_model=UserAdminOut)
def deactivate_user(
    user_id: int,
    request: Request,
    actor: UserRow = Depends(require_admin),
) -> UserAdminOut:
    u = db.get_user(user_id)
    if u is None:
        raise HTTPException(404, "Пользователь не найден")
    if u.id == actor.id:
        raise HTTPException(400, "Нельзя деактивировать свой собственный аккаунт")
    if u.role == "admin" and db.count_admins_active() <= 1:
        raise HTTPException(400, "Нельзя деактивировать последнего активного админа")
    db.update_user(user_id, is_active=False)
    revoked = revoke_all_sessions_for_user(user_id)
    db.log_audit(
        event="reject",
        user_id=user_id,
        actor_user_id=actor.id,
        details=f"sessions_revoked={revoked}",
        ip_address=_client_ip(request),
    )
    log.info("User %s деактивирован админом %s", u.email, actor.email)
    fresh = db.get_user(user_id)
    assert fresh is not None
    return _to_admin_out(fresh)


# ---------- role / unlock ----------

class RoleIn(BaseModel):
    role: str = Field(..., pattern=r"^(admin|user)$")


@router.post("/users/{user_id}/role", response_model=UserAdminOut)
def set_role(
    user_id: int,
    payload: RoleIn,
    request: Request,
    actor: UserRow = Depends(require_admin),
) -> UserAdminOut:
    u = db.get_user(user_id)
    if u is None:
        raise HTTPException(404, "Пользователь не найден")
    # Защита от самопонижения, если ты последний админ
    if u.id == actor.id and payload.role != "admin":
        if db.count_admins_active() <= 1:
            raise HTTPException(400, "Нельзя понизить последнего активного админа")
    db.update_user(user_id, role=payload.role)
    db.log_audit(
        event="role_change",
        user_id=user_id,
        actor_user_id=actor.id,
        details=f"new_role={payload.role}",
        ip_address=_client_ip(request),
    )
    fresh = db.get_user(user_id)
    assert fresh is not None
    return _to_admin_out(fresh)


@router.post("/users/{user_id}/unlock", response_model=UserAdminOut)
def unlock_user(
    user_id: int,
    request: Request,
    actor: UserRow = Depends(require_admin),
) -> UserAdminOut:
    u = db.get_user(user_id)
    if u is None:
        raise HTTPException(404, "Пользователь не найден")
    db.update_user(user_id, failed_login_count=0, clear_locked_until=True)
    db.log_audit(event="unlock", user_id=user_id, actor_user_id=actor.id, ip_address=_client_ip(request))
    fresh = db.get_user(user_id)
    assert fresh is not None
    return _to_admin_out(fresh)


class ResetPasswordIn(BaseModel):
    new_password: str = Field(..., min_length=1, max_length=256)


@router.post("/users/{user_id}/reset-password", response_model=UserAdminOut)
def reset_password(
    user_id: int,
    payload: ResetPasswordIn,
    request: Request,
    actor: UserRow = Depends(require_admin),
) -> UserAdminOut:
    u = db.get_user(user_id)
    if u is None:
        raise HTTPException(404, "Пользователь не найден")
    pwd_problem = validate_password_strength(payload.new_password)
    if pwd_problem:
        raise HTTPException(400, pwd_problem)
    db.update_user(user_id, password_hash=hash_password(payload.new_password))
    revoke_all_sessions_for_user(user_id)
    db.log_audit(
        event="password_change",
        user_id=user_id,
        actor_user_id=actor.id,
        details="reset_by_admin",
        ip_address=_client_ip(request),
    )
    fresh = db.get_user(user_id)
    assert fresh is not None
    return _to_admin_out(fresh)


# ---------- delete ----------

@router.delete("/users/{user_id}", status_code=204)
def delete_user(
    user_id: int,
    request: Request,
    actor: UserRow = Depends(require_admin),
) -> Response:
    u = db.get_user(user_id)
    if u is None:
        raise HTTPException(404, "Пользователь не найден")
    if u.id == actor.id:
        raise HTTPException(400, "Нельзя удалить свой аккаунт")
    if u.role == "admin" and db.count_admins_active() <= 1:
        raise HTTPException(400, "Нельзя удалить последнего активного админа")
    db.delete_user(user_id)
    db.log_audit(
        event="delete_user",
        user_id=user_id,
        actor_user_id=actor.id,
        details=f"email={u.email}",
        ip_address=_client_ip(request),
    )
    log.info("User %s удалён админом %s", u.email, actor.email)
    return Response(status_code=204)


# ---------- audit ----------

class AuditOut(BaseModel):
    id: int
    user_id: Optional[int]
    actor_user_id: Optional[int]
    event: str
    details: Optional[str]
    ip_address: Optional[str]
    created_at: str


@router.get("/audit", response_model=list[AuditOut])
def list_audit(limit: int = 200, _actor: UserRow = Depends(require_admin)) -> list[AuditOut]:
    limit = max(1, min(limit, 1000))
    return [
        AuditOut(
            id=r.id,
            user_id=r.user_id,
            actor_user_id=r.actor_user_id,
            event=r.event,
            details=r.details,
            ip_address=r.ip_address,
            created_at=r.created_at,
        )
        for r in db.list_audit(limit=limit)
    ]


# ===== DB browser (read-only) =====
# Безопасность: НЕ позволяем произвольный SQL. Только белый список таблиц с
# фиксированным набором колонок и обязательным удалением чувствительных полей
# (password_hash, token_hash целиком).

_DB_TABLES: dict[str, dict] = {
    "documents": {
        "columns": ["id", "filename", "file_type", "file_size", "upload_date",
                    "status", "error_message", "chunk_count", "uploaded_by"],
        "default_order": "id DESC",
    },
    "chunks": {
        # text может быть длинным — обрежем при выдаче
        "columns": ["id", "document_id", "chunk_index", "page_number",
                    "sheet_name", "slide_number", "substr(text, 1, 200) AS text_preview"],
        "default_order": "id DESC",
        "raw_select": True,
    },
    "conversations": {
        "columns": ["id", "title", "user_id", "created_at", "updated_at"],
        "default_order": "updated_at DESC",
    },
    "messages": {
        "columns": ["id", "conversation_id", "role",
                    "substr(content, 1, 200) AS content_preview", "created_at"],
        "default_order": "id DESC",
        "raw_select": True,
    },
    "users": {
        # password_hash НЕ возвращаем
        "columns": ["id", "email", "display_name", "role", "is_active",
                    "failed_login_count", "locked_until", "created_at",
                    "last_login_at", "last_login_ip"],
        "default_order": "created_at DESC",
    },
    "sessions": {
        # token_hash сокращаем чтобы по нему нельзя было собрать токен
        "columns": ["substr(token_hash, 1, 12) || '...' AS token_prefix",
                    "user_id", "created_at", "expires_at", "last_seen_at",
                    "ip_address", "substr(user_agent, 1, 80) AS user_agent_short"],
        "default_order": "created_at DESC",
        "raw_select": True,
    },
    "auth_audit": {
        "columns": ["id", "user_id", "actor_user_id", "event",
                    "details", "ip_address", "created_at"],
        "default_order": "id DESC",
    },
}


class DbTablesOut(BaseModel):
    tables: list[dict]


class DbRowsOut(BaseModel):
    table: str
    columns: list[str]
    rows: list[list]
    total: int
    limit: int
    offset: int


@router.get("/db/tables", response_model=DbTablesOut)
def db_list_tables(_actor: UserRow = Depends(require_admin)) -> DbTablesOut:
    out = []
    with db.cursor() as cur:
        for name in _DB_TABLES.keys():
            try:
                cur.execute(f"SELECT COUNT(*) AS c FROM {name}")
                count = int(cur.fetchone()["c"])
            except Exception:
                count = 0
            out.append({"name": name, "rows": count})
    return DbTablesOut(tables=out)


@router.get("/db/{table}", response_model=DbRowsOut)
def db_browse_table(
    table: str,
    limit: int = 50,
    offset: int = 0,
    _actor: UserRow = Depends(require_admin),
) -> DbRowsOut:
    if table not in _DB_TABLES:
        raise HTTPException(404, f"Таблица '{table}' не доступна для просмотра")
    spec = _DB_TABLES[table]
    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    columns = spec["columns"]
    select_list = ", ".join(columns)
    order = spec["default_order"]

    with db.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS c FROM {table}")
        total = int(cur.fetchone()["c"])
        cur.execute(
            f"SELECT {select_list} FROM {table} ORDER BY {order} LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = cur.fetchall()

    # Извлекаем имена колонок из spec (без AS-выражений)
    display_cols = [_strip_alias(c) for c in columns]

    out_rows = []
    for r in rows:
        out_rows.append([_safe_value(r[i]) for i in range(len(display_cols))])

    return DbRowsOut(
        table=table,
        columns=display_cols,
        rows=out_rows,
        total=total,
        limit=limit,
        offset=offset,
    )


def _strip_alias(col_expr: str) -> str:
    """'substr(text, 1, 200) AS text_preview' → 'text_preview'"""
    upper = col_expr.upper()
    if " AS " in upper:
        idx = upper.rfind(" AS ")
        return col_expr[idx + 4:].strip()
    return col_expr.strip()


def _safe_value(v):
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)
