from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from auth.dependencies import csrf_check, require_user
from storage import db, UserRow

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class ConversationOut(BaseModel):
    id: int
    title: Optional[str]
    created_at: str
    updated_at: str
    notebook_id: Optional[int] = None


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


def _can_access(conv, user: UserRow) -> bool:
    """Админ видит всё. Юзер — только свои или legacy без user_id."""
    if user.role == "admin":
        return True
    return conv.user_id is None or conv.user_id == user.id


@router.get("", response_model=list[ConversationOut])
def list_conversations(
    notebook_id: Optional[int] = None,
    user: UserRow = Depends(require_user),
) -> list[ConversationOut]:
    user_filter = None if user.role == "admin" else user.id
    if notebook_id is not None and user.role != "admin":
        nb = db.get_notebook(notebook_id)
        if nb is None or nb.user_id != user.id:
            raise HTTPException(404, "Ноутбук не найден")
    items = db.list_conversations(user_id=user_filter, notebook_id=notebook_id)
    return [
        ConversationOut(
            id=c.id, title=c.title,
            created_at=c.created_at, updated_at=c.updated_at,
            notebook_id=c.notebook_id,
        )
        for c in items
    ]


class CreateConversationIn(BaseModel):
    notebook_id: Optional[int] = None


@router.post(
    "",
    response_model=ConversationOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(csrf_check)],
)
def create_conversation(
    payload: CreateConversationIn = CreateConversationIn(),
    user: UserRow = Depends(require_user),
) -> ConversationOut:
    notebook_id = payload.notebook_id
    if notebook_id is not None and user.role != "admin":
        nb = db.get_notebook(notebook_id)
        if nb is None or nb.user_id != user.id:
            raise HTTPException(404, "Ноутбук не найден")
    if notebook_id is None:
        from api.notebooks import ensure_default_notebook
        notebook_id = ensure_default_notebook(user).id
    cid = db.create_conversation(title=None, user_id=user.id, notebook_id=notebook_id)
    c = db.get_conversation(cid)
    assert c is not None
    return ConversationOut(
        id=c.id, title=c.title,
        created_at=c.created_at, updated_at=c.updated_at,
        notebook_id=c.notebook_id,
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: int, user: UserRow = Depends(require_user)) -> ConversationDetail:
    c = db.get_conversation(conversation_id)
    if not c or not _can_access(c, user):
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


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(csrf_check)],
)
def delete_conversation(conversation_id: int, user: UserRow = Depends(require_user)) -> Response:
    c = db.get_conversation(conversation_id)
    if not c or not _can_access(c, user):
        raise HTTPException(404, "Диалог не найден")
    db.delete_conversation(conversation_id)
    return Response(status_code=204)
