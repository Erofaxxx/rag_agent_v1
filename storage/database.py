import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from config import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    upload_date TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    chunk_count INTEGER DEFAULT 0,
    uploaded_by INTEGER
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    page_number INTEGER,
    sheet_name TEXT,
    slide_number INTEGER,
    metadata_json TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    user_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    cited_chunks_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);

-- Аутентификация и пользователи
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    display_name TEXT,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',          -- 'admin' | 'user'
    is_active INTEGER NOT NULL DEFAULT 0,        -- 1 = одобрен админом
    failed_login_count INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,                           -- ISO timestamp временного лока
    created_at TEXT NOT NULL,
    last_login_at TEXT,
    last_login_ip TEXT
);

CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,                 -- sha256 от cookie value
    user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    ip_address TEXT,
    user_agent TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

CREATE TABLE IF NOT EXISTS auth_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,                             -- может быть NULL (login_fail на несуществующего)
    actor_user_id INTEGER,                       -- кто инициировал (например, админ при approve)
    event TEXT NOT NULL,                         -- 'register', 'login_success', 'login_fail',
                                                 -- 'logout', 'approve', 'reject', 'role_change',
                                                 -- 'lockout', 'unlock', 'delete_user', 'password_change'
    details TEXT,
    ip_address TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_user ON auth_audit(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_event ON auth_audit(event);
CREATE INDEX IF NOT EXISTS idx_audit_created ON auth_audit(created_at);

-- Ноутбуки (workspaces). У каждого пользователя несколько ноутбуков, в каждом
-- свой набор документов и диалогов. Поиск/RAG ограничен текущим ноутбуком.
CREATE TABLE IF NOT EXISTS notebooks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_notebooks_user ON notebooks(user_id);
"""

# Дополнительные миграции для обратной совместимости с уже существующими БД
MIGRATIONS = [
    # (test_query, alter_query). test_query должен бросать исключение, если
    # миграция нужна (например, отсутствует колонка).
    ("SELECT user_id FROM conversations LIMIT 1",
     "ALTER TABLE conversations ADD COLUMN user_id INTEGER"),
    ("SELECT uploaded_by FROM documents LIMIT 1",
     "ALTER TABLE documents ADD COLUMN uploaded_by INTEGER"),
    ("SELECT notebook_id FROM documents LIMIT 1",
     "ALTER TABLE documents ADD COLUMN notebook_id INTEGER"),
    ("SELECT notebook_id FROM conversations LIMIT 1",
     "ALTER TABLE conversations ADD COLUMN notebook_id INTEGER"),
]


@dataclass
class DocumentRow:
    id: int
    filename: str
    file_path: str
    file_type: str
    file_size: int
    upload_date: str
    status: str
    error_message: Optional[str]
    chunk_count: int
    uploaded_by: Optional[int] = None
    notebook_id: Optional[int] = None


@dataclass
class NotebookRow:
    id: int
    user_id: int
    name: str
    created_at: str
    updated_at: str


@dataclass
class ChunkRow:
    id: int
    document_id: int
    chunk_index: int
    text: str
    page_number: Optional[int]
    sheet_name: Optional[str]
    slide_number: Optional[int]
    metadata: dict[str, Any]


@dataclass
class ConversationRow:
    id: int
    title: Optional[str]
    created_at: str
    updated_at: str
    user_id: Optional[int] = None
    notebook_id: Optional[int] = None


@dataclass
class MessageRow:
    id: int
    conversation_id: int
    role: str
    content: str
    cited_chunks: list[dict[str, Any]]
    created_at: str


@dataclass
class UserRow:
    id: int
    email: str
    display_name: Optional[str]
    password_hash: str
    role: str
    is_active: bool
    failed_login_count: int
    locked_until: Optional[str]
    created_at: str
    last_login_at: Optional[str]
    last_login_ip: Optional[str]


@dataclass
class SessionRow:
    token_hash: str
    user_id: int
    created_at: str
    expires_at: str
    last_seen_at: str
    ip_address: Optional[str]
    user_agent: Optional[str]


@dataclass
class AuditRow:
    id: int
    user_id: Optional[int]
    actor_user_id: Optional[int]
    event: str
    details: Optional[str]
    ip_address: Optional[str]
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.executescript(SCHEMA)
                # Best-effort миграции для старых БД
                for test_q, alter_q in MIGRATIONS:
                    try:
                        conn.execute(test_q)
                    except sqlite3.OperationalError:
                        try:
                            conn.execute(alter_q)
                        except sqlite3.OperationalError:
                            pass  # колонка уже есть или таблицы ещё нет

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute("BEGIN")
                yield cur
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    # --- documents ---

    def create_document(
        self,
        filename: str,
        file_path: str,
        file_type: str,
        file_size: int,
        uploaded_by: Optional[int] = None,
        notebook_id: Optional[int] = None,
    ) -> int:
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO documents
                   (filename, file_path, file_type, file_size, upload_date, status, uploaded_by, notebook_id)
                   VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
                (filename, file_path, file_type, file_size, _now(), uploaded_by, notebook_id),
            )
            return int(cur.lastrowid)

    def update_document_status(
        self,
        document_id: int,
        status: str,
        error_message: Optional[str] = None,
        chunk_count: Optional[int] = None,
    ) -> None:
        with self.cursor() as cur:
            if chunk_count is not None:
                cur.execute(
                    "UPDATE documents SET status=?, error_message=?, chunk_count=? WHERE id=?",
                    (status, error_message, chunk_count, document_id),
                )
            else:
                cur.execute(
                    "UPDATE documents SET status=?, error_message=? WHERE id=?",
                    (status, error_message, document_id),
                )

    def get_document(self, document_id: int) -> Optional[DocumentRow]:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE id=?", (document_id,))
            row = cur.fetchone()
            return _row_to_document(row) if row else None

    def list_documents(
        self,
        owner_user_id: Optional[int] = None,
        notebook_id: Optional[int] = None,
    ) -> list[DocumentRow]:
        """Фильтр по владельцу (uploaded_by) и/или по ноутбуку."""
        clauses = []
        params: list = []
        if owner_user_id is not None:
            clauses.append("uploaded_by=?")
            params.append(owner_user_id)
        if notebook_id is not None:
            clauses.append("notebook_id=?")
            params.append(notebook_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM documents {where} ORDER BY upload_date DESC"
        with self.cursor() as cur:
            cur.execute(sql, params)
            return [_row_to_document(r) for r in cur.fetchall()]

    def count_documents(
        self,
        owner_user_id: Optional[int] = None,
        notebook_id: Optional[int] = None,
    ) -> int:
        clauses = []
        params: list = []
        if owner_user_id is not None:
            clauses.append("uploaded_by=?")
            params.append(owner_user_id)
        if notebook_id is not None:
            clauses.append("notebook_id=?")
            params.append(notebook_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS c FROM documents {where}", params)
            return int(cur.fetchone()["c"])

    def get_chunk_owners(self, chunk_ids: list[int]) -> dict[int, Optional[int]]:
        """Возвращает {chunk_id: uploaded_by} для фильтрации поиска по владельцу."""
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        with self.cursor() as cur:
            cur.execute(
                f"""SELECT c.id, d.uploaded_by
                    FROM chunks c JOIN documents d ON c.document_id = d.id
                    WHERE c.id IN ({placeholders})""",
                chunk_ids,
            )
            return {int(r["id"]): r["uploaded_by"] for r in cur.fetchall()}

    def get_chunk_notebooks(self, chunk_ids: list[int]) -> dict[int, Optional[int]]:
        """Возвращает {chunk_id: notebook_id} для фильтрации поиска по ноутбуку."""
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        with self.cursor() as cur:
            cur.execute(
                f"""SELECT c.id, d.notebook_id
                    FROM chunks c JOIN documents d ON c.document_id = d.id
                    WHERE c.id IN ({placeholders})""",
                chunk_ids,
            )
            return {int(r["id"]): r["notebook_id"] for r in cur.fetchall()}

    def reset_processing_at_startup(self) -> int:
        """Все documents в статусе 'pending'/'processing' остались от прошлого
        запуска (BackgroundTasks не переживают рестарт сервера). Помечаем
        как 'error' с пояснением, чтобы пользователь видел и мог перезалить."""
        with self.cursor() as cur:
            cur.execute(
                """UPDATE documents
                   SET status='error',
                       error_message='Обработка прервана при перезапуске сервера. Загрузите файл заново.'
                   WHERE status IN ('pending', 'processing')"""
            )
            return cur.rowcount

    def delete_document(self, document_id: int) -> list[int]:
        with self.cursor() as cur:
            cur.execute("SELECT id FROM chunks WHERE document_id=?", (document_id,))
            chunk_ids = [int(r["id"]) for r in cur.fetchall()]
            cur.execute("DELETE FROM documents WHERE id=?", (document_id,))
            return chunk_ids

    # --- chunks ---

    def insert_chunks(self, document_id: int, chunks: list[dict[str, Any]]) -> list[int]:
        ids: list[int] = []
        with self.cursor() as cur:
            for c in chunks:
                cur.execute(
                    """INSERT INTO chunks
                       (document_id, chunk_index, text, page_number, sheet_name, slide_number, metadata_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        document_id,
                        c["chunk_index"],
                        c["text"],
                        c.get("page_number"),
                        c.get("sheet_name"),
                        c.get("slide_number"),
                        json.dumps(c.get("metadata", {}), ensure_ascii=False),
                    ),
                )
                ids.append(int(cur.lastrowid))
        return ids

    def get_chunks_by_ids(self, ids: list[int]) -> list[ChunkRow]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        with self.cursor() as cur:
            cur.execute(f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids)
            rows = cur.fetchall()
        by_id = {int(r["id"]): _row_to_chunk(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def get_all_chunk_ids_with_text(self) -> list[tuple[int, str]]:
        with self.cursor() as cur:
            cur.execute("SELECT id, text FROM chunks")
            return [(int(r["id"]), str(r["text"])) for r in cur.fetchall()]

    # --- conversations / messages ---

    def create_conversation(
        self,
        title: Optional[str] = None,
        user_id: Optional[int] = None,
        notebook_id: Optional[int] = None,
    ) -> int:
        now = _now()
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO conversations (title, created_at, updated_at, user_id, notebook_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (title, now, now, user_id, notebook_id),
            )
            return int(cur.lastrowid)

    def list_conversations(
        self,
        user_id: Optional[int] = None,
        notebook_id: Optional[int] = None,
    ) -> list[ConversationRow]:
        clauses = []
        params: list = []
        if user_id is not None:
            clauses.append("user_id=?")
            params.append(user_id)
        if notebook_id is not None:
            clauses.append("notebook_id=?")
            params.append(notebook_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM conversations {where} ORDER BY updated_at DESC"
        with self.cursor() as cur:
            cur.execute(sql, params)
            return [_row_to_conversation(r) for r in cur.fetchall()]

    def get_conversation(self, conversation_id: int) -> Optional[ConversationRow]:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,))
            row = cur.fetchone()
            return _row_to_conversation(row) if row else None

    def delete_conversation(self, conversation_id: int) -> None:
        with self.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        cited_chunks: Optional[list[dict[str, Any]]] = None,
    ) -> int:
        now = _now()
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO messages
                   (conversation_id, role, content, cited_chunks_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    conversation_id,
                    role,
                    content,
                    json.dumps(cited_chunks or [], ensure_ascii=False),
                    now,
                ),
            )
            cur.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (now, conversation_id),
            )
            return int(cur.lastrowid)

    def get_messages(
        self, conversation_id: int, limit: Optional[int] = None
    ) -> list[MessageRow]:
        with self.cursor() as cur:
            if limit:
                cur.execute(
                    """SELECT * FROM (
                           SELECT * FROM messages WHERE conversation_id=?
                           ORDER BY id DESC LIMIT ?
                       ) ORDER BY id ASC""",
                    (conversation_id, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM messages WHERE conversation_id=? ORDER BY id ASC",
                    (conversation_id,),
                )
            return [_row_to_message(r) for r in cur.fetchall()]

    # ===== notebooks =====

    def create_notebook(self, user_id: int, name: str) -> int:
        now = _now()
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO notebooks (user_id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (user_id, name.strip()[:120] or "Без названия", now, now),
            )
            return int(cur.lastrowid)

    def list_notebooks(self, user_id: int) -> list[NotebookRow]:
        with self.cursor() as cur:
            cur.execute(
                "SELECT * FROM notebooks WHERE user_id=? ORDER BY created_at ASC",
                (user_id,),
            )
            return [_row_to_notebook(r) for r in cur.fetchall()]

    def get_notebook(self, notebook_id: int) -> Optional[NotebookRow]:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM notebooks WHERE id=?", (notebook_id,))
            row = cur.fetchone()
            return _row_to_notebook(row) if row else None

    def rename_notebook(self, notebook_id: int, name: str) -> None:
        with self.cursor() as cur:
            cur.execute(
                "UPDATE notebooks SET name=?, updated_at=? WHERE id=?",
                (name.strip()[:120] or "Без названия", _now(), notebook_id),
            )

    def delete_notebook(self, notebook_id: int) -> list[int]:
        """Удаляет ноутбук со всеми документами и диалогами. Возвращает список
        chunk_ids для последующей очистки FAISS."""
        with self.cursor() as cur:
            cur.execute(
                """SELECT c.id FROM chunks c JOIN documents d ON c.document_id=d.id
                   WHERE d.notebook_id=?""",
                (notebook_id,),
            )
            chunk_ids = [int(r["id"]) for r in cur.fetchall()]
            # Удаляем документы (CASCADE снимет chunks), диалоги, потом сам ноутбук
            cur.execute("DELETE FROM documents WHERE notebook_id=?", (notebook_id,))
            cur.execute("DELETE FROM conversations WHERE notebook_id=?", (notebook_id,))
            cur.execute("DELETE FROM notebooks WHERE id=?", (notebook_id,))
            return chunk_ids

    def list_documents_in_notebook(self, notebook_id: int) -> list[DocumentRow]:
        return self.list_documents(notebook_id=notebook_id)

    def assign_orphans_to_notebook(self, user_id: int, notebook_id: int) -> tuple[int, int]:
        """Один раз привязывает старые документы и диалоги юзера без notebook_id
        к указанному ноутбуку. Используется при первом создании дефолтного
        ноутбука для существующих юзеров."""
        with self.cursor() as cur:
            cur.execute(
                "UPDATE documents SET notebook_id=? WHERE uploaded_by=? AND notebook_id IS NULL",
                (notebook_id, user_id),
            )
            docs = cur.rowcount
            cur.execute(
                "UPDATE conversations SET notebook_id=? WHERE user_id=? AND notebook_id IS NULL",
                (notebook_id, user_id),
            )
            convs = cur.rowcount
            return docs, convs

    # ===== users =====

    def create_user(
        self,
        email: str,
        password_hash: str,
        display_name: Optional[str] = None,
        role: str = "user",
        is_active: bool = False,
    ) -> int:
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO users
                   (email, display_name, password_hash, role, is_active, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (email.strip().lower(), display_name, password_hash, role, 1 if is_active else 0, _now()),
            )
            return int(cur.lastrowid)

    def get_user(self, user_id: int) -> Optional[UserRow]:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE id=?", (user_id,))
            row = cur.fetchone()
            return _row_to_user(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[UserRow]:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE email=? COLLATE NOCASE", (email.strip().lower(),))
            row = cur.fetchone()
            return _row_to_user(row) if row else None

    def list_users(self) -> list[UserRow]:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM users ORDER BY created_at DESC")
            return [_row_to_user(r) for r in cur.fetchall()]

    def count_users(self) -> int:
        with self.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM users")
            return int(cur.fetchone()["c"])

    def update_user(
        self,
        user_id: int,
        *,
        is_active: Optional[bool] = None,
        role: Optional[str] = None,
        display_name: Optional[str] = None,
        password_hash: Optional[str] = None,
        failed_login_count: Optional[int] = None,
        locked_until: Optional[str] = None,
        last_login_at: Optional[str] = None,
        last_login_ip: Optional[str] = None,
        clear_locked_until: bool = False,
    ) -> None:
        fields: list[str] = []
        values: list[Any] = []
        if is_active is not None:
            fields.append("is_active=?")
            values.append(1 if is_active else 0)
        if role is not None:
            fields.append("role=?")
            values.append(role)
        if display_name is not None:
            fields.append("display_name=?")
            values.append(display_name)
        if password_hash is not None:
            fields.append("password_hash=?")
            values.append(password_hash)
        if failed_login_count is not None:
            fields.append("failed_login_count=?")
            values.append(failed_login_count)
        if locked_until is not None:
            fields.append("locked_until=?")
            values.append(locked_until)
        if clear_locked_until:
            fields.append("locked_until=NULL")
        if last_login_at is not None:
            fields.append("last_login_at=?")
            values.append(last_login_at)
        if last_login_ip is not None:
            fields.append("last_login_ip=?")
            values.append(last_login_ip)
        if not fields:
            return
        values.append(user_id)
        with self.cursor() as cur:
            cur.execute(f"UPDATE users SET {', '.join(fields)} WHERE id=?", values)

    def delete_user(self, user_id: int) -> None:
        with self.cursor() as cur:
            cur.execute("DELETE FROM users WHERE id=?", (user_id,))

    def count_admins_active(self) -> int:
        with self.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM users WHERE role='admin' AND is_active=1")
            return int(cur.fetchone()["c"])

    # ===== sessions =====

    def create_session(
        self,
        token_hash: str,
        user_id: int,
        expires_at: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        now = _now()
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO sessions
                   (token_hash, user_id, created_at, expires_at, last_seen_at, ip_address, user_agent)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (token_hash, user_id, now, expires_at, now, ip_address, user_agent),
            )

    def get_session(self, token_hash: str) -> Optional[SessionRow]:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM sessions WHERE token_hash=?", (token_hash,))
            row = cur.fetchone()
            return _row_to_session(row) if row else None

    def touch_session(self, token_hash: str, ip_address: Optional[str] = None) -> None:
        with self.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET last_seen_at=?, ip_address=COALESCE(?, ip_address) WHERE token_hash=?",
                (_now(), ip_address, token_hash),
            )

    def delete_session(self, token_hash: str) -> None:
        with self.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))

    def delete_sessions_for_user(self, user_id: int) -> int:
        with self.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
            return cur.rowcount

    def cleanup_expired_sessions(self) -> int:
        with self.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE expires_at < ?", (_now(),))
            return cur.rowcount

    # ===== audit =====

    def log_audit(
        self,
        event: str,
        *,
        user_id: Optional[int] = None,
        actor_user_id: Optional[int] = None,
        details: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO auth_audit
                   (user_id, actor_user_id, event, details, ip_address, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, actor_user_id, event, details, ip_address, _now()),
            )

    def list_audit(self, limit: int = 200) -> list[AuditRow]:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM auth_audit ORDER BY id DESC LIMIT ?", (limit,))
            return [_row_to_audit(r) for r in cur.fetchall()]


def _row_to_document(row: sqlite3.Row) -> DocumentRow:
    keys = row.keys() if hasattr(row, "keys") else []
    uploaded_by = row["uploaded_by"] if "uploaded_by" in keys else None
    notebook_id = row["notebook_id"] if "notebook_id" in keys else None
    return DocumentRow(
        id=int(row["id"]),
        filename=str(row["filename"]),
        file_path=str(row["file_path"]),
        file_type=str(row["file_type"]),
        file_size=int(row["file_size"]),
        upload_date=str(row["upload_date"]),
        status=str(row["status"]),
        error_message=row["error_message"],
        chunk_count=int(row["chunk_count"] or 0),
        uploaded_by=uploaded_by,
        notebook_id=notebook_id,
    )


def _row_to_chunk(row: sqlite3.Row) -> ChunkRow:
    metadata = {}
    if row["metadata_json"]:
        try:
            metadata = json.loads(row["metadata_json"])
        except (TypeError, ValueError):
            metadata = {}
    return ChunkRow(
        id=int(row["id"]),
        document_id=int(row["document_id"]),
        chunk_index=int(row["chunk_index"]),
        text=str(row["text"]),
        page_number=row["page_number"],
        sheet_name=row["sheet_name"],
        slide_number=row["slide_number"],
        metadata=metadata,
    )


def _row_to_conversation(row: sqlite3.Row) -> ConversationRow:
    keys = row.keys() if hasattr(row, "keys") else []
    user_id = row["user_id"] if "user_id" in keys else None
    notebook_id = row["notebook_id"] if "notebook_id" in keys else None
    return ConversationRow(
        id=int(row["id"]),
        title=row["title"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        user_id=user_id,
        notebook_id=notebook_id,
    )


def _row_to_notebook(row: sqlite3.Row) -> NotebookRow:
    return NotebookRow(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        name=str(row["name"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_message(row: sqlite3.Row) -> MessageRow:
    cited = []
    if row["cited_chunks_json"]:
        try:
            cited = json.loads(row["cited_chunks_json"])
        except (TypeError, ValueError):
            cited = []
    return MessageRow(
        id=int(row["id"]),
        conversation_id=int(row["conversation_id"]),
        role=str(row["role"]),
        content=str(row["content"]),
        cited_chunks=cited,
        created_at=str(row["created_at"]),
    )


def _row_to_user(row: sqlite3.Row) -> UserRow:
    return UserRow(
        id=int(row["id"]),
        email=str(row["email"]),
        display_name=row["display_name"],
        password_hash=str(row["password_hash"]),
        role=str(row["role"]),
        is_active=bool(row["is_active"]),
        failed_login_count=int(row["failed_login_count"] or 0),
        locked_until=row["locked_until"],
        created_at=str(row["created_at"]),
        last_login_at=row["last_login_at"],
        last_login_ip=row["last_login_ip"],
    )


def _row_to_session(row: sqlite3.Row) -> SessionRow:
    return SessionRow(
        token_hash=str(row["token_hash"]),
        user_id=int(row["user_id"]),
        created_at=str(row["created_at"]),
        expires_at=str(row["expires_at"]),
        last_seen_at=str(row["last_seen_at"]),
        ip_address=row["ip_address"],
        user_agent=row["user_agent"],
    )


def _row_to_audit(row: sqlite3.Row) -> AuditRow:
    return AuditRow(
        id=int(row["id"]),
        user_id=row["user_id"],
        actor_user_id=row["actor_user_id"],
        event=str(row["event"]),
        details=row["details"],
        ip_address=row["ip_address"],
        created_at=str(row["created_at"]),
    )


db = Database(str(settings.db_path))
