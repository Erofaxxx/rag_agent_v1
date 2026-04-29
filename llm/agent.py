import logging
import re
import threading
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import settings
from llm.prompts import SYSTEM_PROMPT
from search import search_service, SearchHit
from storage import db


# Tool-message в результате содержит блоки `[chunk_id=N] [source] [score=X.XXX]`.
# В langgraph 0.2.x sync-тулы выполняются в отдельном thread у ToolNode, и
# threading.local() из вызывающего потока туда не пробрасывается. Поэтому
# вместо thread-local берём chunk_id напрямую из текста ToolMessage.
_CITED_RE = re.compile(r"\[chunk_id=(\d+)\] \[([^\]]+)\] \[score=([0-9.]+)\]")

log = logging.getLogger(__name__)


# --- thread-local: текущие найденные чанки за один turn ---
# Чтобы не таскать их через сообщения и не плодить токены, агент пишет
# сюда из тула, а ручка чата забирает после .invoke().
_thread_local = threading.local()


def _reset_thread_state(user_id: Optional[int], notebook_id: Optional[int] = None) -> None:
    _thread_local.hits = []
    _thread_local.user_id = user_id
    _thread_local.notebook_id = notebook_id
    _thread_local.tool_calls = 0


def _get_hits() -> list[SearchHit]:
    return getattr(_thread_local, "hits", [])


def _get_user_id() -> Optional[int]:
    return getattr(_thread_local, "user_id", None)


def _get_notebook_id() -> Optional[int]:
    return getattr(_thread_local, "notebook_id", None)


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
    # Hard cap на количество tool-вызовов за один turn — защита от runaway-итераций.
    # Соответствует паттерну BudgetMiddleware из magnetto_agent_v2: вместо того
    # чтобы упасть с recursion_limit, мягко отдаём LLM «лимит исчерпан, отвечай
    # на основе того, что есть» и оставляем шанс сформировать финальный ответ.
    used = getattr(_thread_local, "tool_calls", 0) + 1
    _thread_local.tool_calls = used
    cap = settings.MAX_TOOL_CALLS_PER_QUESTION

    log.info("[tool] search_documents(%r) — call %d/%d", query, used, cap)

    if used > cap:
        return (
            f"Лимит поисковых запросов исчерпан ({cap}). Дай финальный ответ "
            f"на основе уже найденных фрагментов или скажи, что данных в "
            f"документах недостаточно. НЕ вызывай search_documents снова."
        )

    user_id = _get_user_id()
    notebook_id = _get_notebook_id()
    hits = search_service.search(
        query,
        k=settings.SEARCH_TOP_K,
        owner_user_id=user_id,
        notebook_id=notebook_id,
    )
    _add_hits(hits)

    formatted = _format_hits_for_llm(hits)
    # Soft warning при приближении к лимиту — даёт LLM шанс заранее группировать
    # запросы или решить, что найденного достаточно.
    remaining = cap - used
    if remaining == 0:
        formatted += "\n\n[Это был последний доступный поиск. Сейчас формируй финальный ответ.]"
    elif remaining == 1:
        formatted += "\n\n[Остался ещё 1 поиск. Используй его только если ответ ещё неполный.]"
    return formatted


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
        state_modifier=SYSTEM_PROMPT,
    )


def get_agent():
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = _build_agent()
    return _agent


def _log_usage(messages: list) -> None:
    """Логирует token usage всех AIMessage в ответе. Для DeepSeek через
    OpenRouter в `prompt_tokens_details.cached_tokens` приходит количество
    кэш-хитов (implicit caching). Видеть это полезно: если в одном диалоге
    cached_tokens=0 раз за разом — значит префикс плывёт (вставляются
    динамические данные, меняется порядок tools и т.п.) и кэш не работает."""
    total_prompt = 0
    total_completion = 0
    total_cached = 0
    calls = 0
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        meta = getattr(msg, "response_metadata", None) or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
        if not usage:
            # langchain-openai иногда кладёт в .usage_metadata
            um = getattr(msg, "usage_metadata", None) or {}
            if um:
                usage = {
                    "prompt_tokens": um.get("input_tokens", 0),
                    "completion_tokens": um.get("output_tokens", 0),
                    "prompt_tokens_details": {
                        "cached_tokens": (um.get("input_token_details") or {}).get("cache_read", 0),
                    },
                }
        if not usage:
            continue
        calls += 1
        total_prompt += int(usage.get("prompt_tokens") or 0)
        total_completion += int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        total_cached += int(details.get("cached_tokens") or 0)

    if calls:
        cache_pct = (100.0 * total_cached / total_prompt) if total_prompt else 0
        log.info(
            "LLM usage: %d calls, prompt=%d, completion=%d, cached=%d (%.0f%% hit)",
            calls, total_prompt, total_completion, total_cached, cache_pct,
        )


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
    notebook_id: Optional[int] = None,
) -> dict[str, Any]:
    """Главный вход. Возвращает {'answer': str, 'cited_chunks': [...]}.

    user_id и notebook_id используются внутри search_documents tool для
    изоляции корпуса: юзер видит только свои документы из указанного ноутбука.
    Передаётся через thread-local, потому что tool-функция не принимает
    контекст напрямую."""
    history = history or []
    history = _truncate_history(history, settings.MAX_HISTORY_MESSAGES)

    _reset_thread_state(user_id, notebook_id)
    agent = get_agent()

    messages = _to_lc_messages(history) + [HumanMessage(content=question)]

    # recursion_limit считает узлы графа: каждый цикл «model → tools → model»
    # это 2 узла. Плюс начальный шаг model. Формула: 1 + N×2 + запас.
    recursion_limit = 1 + settings.MAX_TOOL_CALLS_PER_QUESTION * 2 + 3

    try:
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": recursion_limit},
        )
    except Exception as e:
        log.exception("Ошибка при вызове агента: %s", e)
        return {
            "answer": f"Ошибка при обращении к LLM: {e}",
            "cited_chunks": [],
        }

    # Логируем usage для мониторинга prompt cache hit-rate. У DeepSeek через
    # OpenRouter кэш implicit — никаких маркеров не шлём, но при стабильном
    # префиксе (system+tools+история) cached_tokens должны быть > 0 на 2-м
    # и далее запросе в одном диалоге.
    _log_usage(result["messages"])

    # Финальный ответ — последнее AIMessage без tool_calls
    answer = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    cited = _collect_cited(result["messages"]) or [
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


def _collect_cited(messages: list) -> list[dict[str, Any]]:
    """Собирает cited_chunks из ToolMessage-ов."""
    scores: dict[int, float] = {}
    order: list[int] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        for m in _CITED_RE.finditer(text):
            cid = int(m.group(1))
            score = float(m.group(3))
            if cid not in scores:
                order.append(cid)
            scores[cid] = max(scores.get(cid, 0.0), score)

    if not order:
        return []

    chunks = db.get_chunks_by_ids(order)
    chunks_by_id = {c.id: c for c in chunks}
    doc_ids = list({c.document_id for c in chunks})
    docs: dict[int, Any] = {}
    for did in doc_ids:
        d = db.get_document(did)
        if d:
            docs[did] = d

    out: list[dict[str, Any]] = []
    for cid in order:
        c = chunks_by_id.get(cid)
        if c is None:
            continue
        doc = docs.get(c.document_id)
        if doc is None:
            continue
        out.append({
            "chunk_id": c.id,
            "document_id": doc.id,
            "filename": doc.filename,
            "page_number": c.page_number,
            "sheet_name": c.sheet_name,
            "slide_number": c.slide_number,
            "score": scores.get(cid, 0.0),
            "snippet": c.text[:500],
        })
    return out
