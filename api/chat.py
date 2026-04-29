import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from auth.dependencies import csrf_check, require_user
from config import settings
from llm import answer_question
from storage import db, UserRow

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[int] = None


class CitedChunk(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None
    slide_number: Optional[int] = None
    score: float
    snippet: str


class ChatResponse(BaseModel):
    conversation_id: int
    message_id: int
    answer: str
    cited_chunks: list[CitedChunk]


@router.post("", response_model=ChatResponse, dependencies=[Depends(csrf_check)])
async def chat(req: ChatRequest, user: UserRow = Depends(require_user)) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(400, "Пустое сообщение")

    if req.conversation_id is not None:
        conv = db.get_conversation(req.conversation_id)
        if conv is None:
            raise HTTPException(404, "Диалог не найден")
        # Изоляция: пользователь видит только свои диалоги. Админ — все.
        if user.role != "admin" and conv.user_id is not None and conv.user_id != user.id:
            raise HTTPException(404, "Диалог не найден")
        conversation_id = req.conversation_id
    else:
        title = req.message[:60]
        conversation_id = db.create_conversation(title=title, user_id=user.id)

    db.add_message(conversation_id, "user", req.message)

    history_rows = db.get_messages(conversation_id, limit=settings.MAX_HISTORY_MESSAGES + 1)
    # Не передаём в LLM последнее user-сообщение через history, оно уйдёт как
    # самостоятельный question; иначе будет дублироваться.
    history = [
        {"role": m.role, "content": m.content}
        for m in history_rows
        if m.id != history_rows[-1].id
    ]

    result: dict[str, Any] = await run_in_threadpool(answer_question, req.message, history)
    answer = result.get("answer") or "Не удалось получить ответ от модели."
    cited = result.get("cited_chunks") or []

    msg_id = db.add_message(conversation_id, "assistant", answer, cited_chunks=cited)
    return ChatResponse(
        conversation_id=conversation_id,
        message_id=msg_id,
        answer=answer,
        cited_chunks=[CitedChunk(**c) for c in cited],
    )
