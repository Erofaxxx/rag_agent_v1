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
    chunk_count INTEGER DEFAULT 0
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
    updated_at TEXT NOT NULL
);

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
"""


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


@dataclass
class MessageRow:
    id: int
    conversation_id: int
    role: str
    content: str
    cited_chunks: list[dict[str, Any]]
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
    ) -> int:
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO documents
                   (filename, file_path, file_type, file_size, upload_date, status)
                   VALUES (?, ?, ?, ?, ?, 'pending')""",
                (filename, file_path, file_type, file_size, _now()),
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

    def list_documents(self) -> list[DocumentRow]:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM documents ORDER BY upload_date DESC")
            return [_row_to_document(r) for r in cur.fetchall()]

    def count_documents(self) -> int:
        with self.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM documents")
            return int(cur.fetchone()["c"])

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

    def create_conversation(self, title: Optional[str] = None) -> int:
        now = _now()
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, now, now),
            )
            return int(cur.lastrowid)

    def list_conversations(self) -> list[ConversationRow]:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM conversations ORDER BY updated_at DESC")
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


def _row_to_document(row: sqlite3.Row) -> DocumentRow:
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
    return ConversationRow(
        id=int(row["id"]),
        title=row["title"],
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


db = Database(str(settings.db_path))
