import logging
import threading
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import settings
from llm.prompts import SYSTEM_PROMPT
from search import search_service, SearchHit

log = logging.getLogger(__name__)


# --- thread-local: текущие найденные чанки за один turn ---
# Чтобы не таскать их через сообщения и не плодить токены, агент пишет
# сюда из тула, а ручка чата забирает после .invoke().
_thread_local = threading.local()


def _reset_thread_state(user_id: Optional[int]) -> None:
    _thread_local.hits = []
    _thread_local.user_id = user_id


def _get_hits() -> list[SearchHit]:
    return getattr(_thread_local, "hits", [])


def _get_user_id() -> Optional[int]:
    return getattr(_thread_local, "user_id", None)


def _add_hits(hits: list[SearchHit]) -> None:
    existing = getattr(_thread_local, "hits", [])
    seen = {h.chunk.id for h in existing}
    for h in hits:
        if h.chunk.id not in seen:
            existing.append(h)
            seen.add(h.chunk.id)
    _thread_local.hits = existing


def _format_source(hit: SearchHit) -> str:
    parts = [hit.document.filename]
    c = hit.chunk
    if c.page_number:
        parts.append(f"стр. {c.page_number}")
    if c.sheet_name:
        parts.append(f"лист «{c.sheet_name}»")
    if c.slide_number:
        parts.append(f"слайд {c.slide_number}")
    return ", ".join(parts)


def _format_hits_for_llm(hits: list[SearchHit]) -> str:
    if not hits:
        return "По этому запросу в документах ничего не найдено."
    parts = []
    for h in hits:
        source = _format_source(h)
        parts.append(
            f"[chunk_id={h.chunk.id}] [{source}] [score={h.score:.3f}]\n{h.chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


@tool
def search_documents(query: str) -> str:
    """Поиск по загруженным документам. Возвращает релевантные фрагменты с указанием
    имени файла и страницы / листа / слайда. Используй разные формулировки, если
    первая выдача не дала ответа.

    Args:
        query: поисковый запрос на естественном языке (русский или английский).
    """
    log.info("[tool] search_documents(%r)", query)
    user_id = _get_user_id()
    hits = search_service.search(query, k=settings.SEARCH_TOP_K, owner_user_id=user_id)
    _add_hits(hits)
    return _format_hits_for_llm(hits)


# --- ленивая инициализация LLM и агента (чтобы импорт модуля был дешёвым) ---

_agent = None
_agent_lock = threading.Lock()


def _build_agent():
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": settings.OPENROUTER_HTTP_REFERER,
            "X-Title": settings.OPENROUTER_X_TITLE,
        },
    )
    return create_react_agent(
        model=llm,
        tools=[search_documents],
        prompt=SYSTEM_PROMPT,
    )


def get_agent():
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = _build_agent()
    return _agent


def _truncate_history(history: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    """Берём последние `limit` сообщений, но всегда сохраняем парность user/assistant."""
    if len(history) <= limit:
        return history
    cut = history[-limit:]
    # Если первое сообщение — assistant, дропаем его, чтобы начиналось с user
    while cut and cut[0]["role"] == "assistant":
        cut = cut[1:]
    return cut


def _to_lc_messages(history: list[dict[str, str]]) -> list[Any]:
    out: list[Any] = []
    for m in history:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
    return out


def answer_question(
    question: str,
    history: Optional[list[dict[str, str]]] = None,
    user_id: Optional[int] = None,
) -> dict[str, Any]:
    """Главный вход. Возвращает {'answer': str, 'cited_chunks': [...]}.

    user_id используется внутри search_documents tool для изоляции корпуса:
    юзер видит только свои документы, админ — все. Передаётся через
    thread-local, потому что tool-функция не принимает контекст напрямую."""
    history = history or []
    history = _truncate_history(history, settings.MAX_HISTORY_MESSAGES)

    _reset_thread_state(user_id)
    agent = get_agent()

    messages = _to_lc_messages(history) + [HumanMessage(content=question)]

    try:
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 8},
        )
    except Exception as e:
        log.exception("Ошибка при вызове агента: %s", e)
        return {
            "answer": f"Ошибка при обращении к LLM: {e}",
            "cited_chunks": [],
        }

    # Финальный ответ — последнее AIMessage без tool_calls
    answer = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    cited = [
        {
            "chunk_id": h.chunk.id,
            "document_id": h.document.id,
            "filename": h.document.filename,
            "page_number": h.chunk.page_number,
            "sheet_name": h.chunk.sheet_name,
            "slide_number": h.chunk.slide_number,
            "score": h.score,
            "snippet": h.chunk.text[:500],
        }
        for h in _get_hits()
    ]

    return {"answer": answer, "cited_chunks": cited}
