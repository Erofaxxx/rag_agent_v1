from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.auth import require_auth
from storage import db

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class ConversationOut(BaseModel):
    id: int
    title: Optional[str]
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    cited_chunks: list[dict[str, Any]] = []
    created_at: str


class ConversationDetail(BaseModel):
    id: int
    title: Optional[str]
    created_at: str
    updated_at: str
    messages: list[MessageOut]


@router.get("", response_model=list[ConversationOut])
def list_conversations(_: str = Depends(require_auth)) -> list[ConversationOut]:
    return [
        ConversationOut(id=c.id, title=c.title, created_at=c.created_at, updated_at=c.updated_at)
        for c in db.list_conversations()
    ]


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
def create_conversation(_: str = Depends(require_auth)) -> ConversationOut:
    cid = db.create_conversation(title=None)
    c = db.get_conversation(cid)
    assert c is not None
    return ConversationOut(id=c.id, title=c.title, created_at=c.created_at, updated_at=c.updated_at)


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: int, _: str = Depends(require_auth)) -> ConversationDetail:
    c = db.get_conversation(conversation_id)
    if not c:
        raise HTTPException(404, "Диалог не найден")
    msgs = db.get_messages(conversation_id)
    return ConversationDetail(
        id=c.id,
        title=c.title,
        created_at=c.created_at,
        updated_at=c.updated_at,
        messages=[
            MessageOut(
                id=m.id,
                role=m.role,
                content=m.content,
                cited_chunks=m.cited_chunks,
                created_at=m.created_at,
            )
            for m in msgs
        ],
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: int, _: str = Depends(require_auth)) -> None:
    if db.get_conversation(conversation_id) is None:
        raise HTTPException(404, "Диалог не найден")
    db.delete_conversation(conversation_id)
