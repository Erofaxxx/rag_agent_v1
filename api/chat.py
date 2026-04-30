import asyncio
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
    notebook_id: Optional[int] = None


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
    # Результат пост-сверки ответа: verified=False означает, что часть
    # утверждений не подтверждена найденными фрагментами (предупреждение
    # уже встроено в текст answer). Фронт может нарисовать badge.
    verified: bool = True
    unsupported: list[str] = []


async def _do_llm_work(
    *,
    conversation_id: int,
    user_message: str,
    history: list[dict[str, str]],
    user_id: int,
    notebook_id: Optional[int],
) -> tuple[int, str, list[dict[str, Any]], dict[str, Any]]:
    """Зашильженная LLM-работа: считает ответ и СОХРАНЯЕТ его в БД даже если
    клиент дисконнектнулся. Возвращает (message_id, answer, cited_chunks, verification)."""
    verification: dict[str, Any] = {"verified": True, "unsupported": []}
    try:
        result: dict[str, Any] = await run_in_threadpool(
            answer_question, user_message, history, user_id, notebook_id
        )
        answer = result.get("answer") or "Не удалось получить ответ от модели."
        cited = result.get("cited_chunks") or []
        verification = result.get("verification") or verification
    except Exception as e:
        log.exception("Ошибка генерации ответа: %s", e)
        answer = f"Ошибка при обращении к LLM: {e}"
        cited = []
    msg_id = db.add_message(conversation_id, "assistant", answer, cited_chunks=cited)
    return msg_id, answer, cited, verification


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
        notebook_id = conv.notebook_id  # уже зафиксирован при создании
    else:
        # Новый диалог. notebook_id из тела запроса; если не передан — дефолтный.
        notebook_id = req.notebook_id
        if notebook_id is not None:
            nb = db.get_notebook(notebook_id)
            if nb is None or (user.role != "admin" and nb.user_id != user.id):
                raise HTTPException(404, "Ноутбук не найден")
        else:
            from api.notebooks import ensure_default_notebook
            notebook_id = ensure_default_notebook(user).id
        title = req.message[:60]
        conversation_id = db.create_conversation(title=title, user_id=user.id, notebook_id=notebook_id)

    db.add_message(conversation_id, "user", req.message)

    history_rows = db.get_messages(conversation_id, limit=settings.MAX_HISTORY_MESSAGES + 1)
    # Не передаём в LLM последнее user-сообщение через history — оно уйдёт как
    # самостоятельный question; иначе будет дублироваться.
    history = [
        {"role": m.role, "content": m.content}
        for m in history_rows
        if m.id != history_rows[-1].id
    ]

    # asyncio.shield защищает работу LLM от отмены при дисконнекте клиента.
    # Если юзер закрыл вкладку или отрубился вайфай — серверная корутина
    # доработает, ответ ляжет в БД, и при следующем открытии диалога юзер его увидит.
    work = _do_llm_work(
        conversation_id=conversation_id,
        user_message=req.message,
        history=history,
        user_id=user.id,
        notebook_id=notebook_id,
    )
    try:
        msg_id, answer, cited, verification = await asyncio.shield(work)
    except asyncio.CancelledError:
        # Клиент отрубился. shield()-задача всё равно доработает в фоне и
        # запишет результат в БД — нам только перевыбросить отмену.
        log.info(
            "Клиент дисконнектнулся в чате %s, ответ сохранится в БД",
            conversation_id,
        )
        raise

    return ChatResponse(
        conversation_id=conversation_id,
        message_id=msg_id,
        answer=answer,
        cited_chunks=[CitedChunk(**c) for c in cited],
        verified=bool(verification.get("verified", True)),
        unsupported=list(verification.get("unsupported") or []),
    )
