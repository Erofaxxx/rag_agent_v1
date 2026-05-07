import functools
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import settings
from llm.prompts import SYSTEM_PROMPT, build_system_prompt
from llm.verifier import (
    append_strict_warning,
    append_verification_warning,
    strict_verify,
    verify_answer,
)
from search import search_service, SearchHit
from storage import db


# Tool-message в результате содержит блоки `[chunk_id=N] [source] [score=X.XXX]`.
# Из ToolMessage достаём chunk_id для построения cited_chunks.
_CITED_RE = re.compile(r"\[chunk_id=(\d+)\] \[([^\]]+)\] \[score=([0-9.]+)\]")

log = logging.getLogger(__name__)


@dataclass
class _RequestState:
    """Состояние одного вызова агента. Строится в answer_question, передаётся
    в tool через замыкание. Раньше было thread-local, но langgraph 0.2.x
    выполняет sync-тулы на воркер-треде из пула — threading.local() оттуда
    не виден, и хуже того, тред мог обслуживать запрос другого пользователя
    в прошлый раз, что приводило к утечке корпуса между арендаторами.

    `lock` синхронизирует tool_calls/hits между параллельными tool-вызовами:
    в обычном create_react_agent они сериализуются, но parallel_tool_calls и
    другие будущие изменения LangGraph могут запустить два search_documents
    одновременно — без лока счётчик ловит race и cap MAX_TOOL_CALLS перестаёт
    защищать."""

    user_id: Optional[int]
    notebook_id: Optional[int]
    hits: list[SearchHit] = field(default_factory=list)
    tool_calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


def _add_hits(state: _RequestState, hits: list[SearchHit]) -> None:
    with state.lock:
        seen = {h.chunk.id for h in state.hits}
        for h in hits:
            if h.chunk.id not in seen:
                state.hits.append(h)
                seen.add(h.chunk.id)


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


def _build_search_tool(state: _RequestState):
    """Создаёт tool с замыканием на per-request state. Скоуп пользователя/ноутбука
    зашит в замыкание, поэтому даже если langgraph выполнит тул на чужом треде —
    параметры берутся из state, а не из глобала."""

    @tool
    def search_documents(query: str) -> str:
        """Поиск по загруженным документам. Возвращает релевантные фрагменты с указанием
        имени файла и страницы / листа / слайда. Используй разные формулировки, если
        первая выдача не дала ответа.

        Args:
            query: поисковый запрос на естественном языке (русский или английский).
        """
        # Hard cap на количество tool-вызовов за один turn — защита от runaway-итераций.
        with state.lock:
            state.tool_calls += 1
            used = state.tool_calls
        cap = settings.MAX_TOOL_CALLS_PER_QUESTION

        log.info("[tool] search_documents(%r) — call %d/%d", query, used, cap)

        if used > cap:
            return (
                f"Лимит поисковых запросов исчерпан ({cap}). Дай финальный ответ "
                f"на основе уже найденных фрагментов или скажи, что данных в "
                f"документах недостаточно. НЕ вызывай search_documents снова."
            )

        hits = search_service.search(
            query,
            k=None,
            owner_user_id=state.user_id,
            notebook_id=state.notebook_id,
        )

        # ---- Defense L3: query-ablation детектор trigger-based backdoor'ов ----
        # Generic защита, работает на любых неизвестных триггерах: сравниваем
        # top-k оригинального запроса с top-k запросов с удалённым по очереди
        # каждым «значимым» словом. Чанк, который выпадает из top-k при
        # ablation, активирован конкретными словами запроса → подозрителен.
        # Все ablations — single-query через ablation_mode=True (без multi-query/
        # HyDE/reformulate), иначе эффект размывается.
        l3_warning = ""
        if settings.DEFENSE_L3_QUERY_ABLATION != "off" and hits:
            try:
                from defenses.l3_query_ablation import (
                    build_warning as _l3_warning,
                    detect_query_specific_chunks,
                    filter_hits as _l3_filter,
                    short_summary as _l3_summary,
                )

                def _ablation_retrieve(q: str):
                    return search_service.search(
                        q,
                        k=None,
                        owner_user_id=state.user_id,
                        notebook_id=state.notebook_id,
                        ablation_mode=True,
                    )

                l3_report = detect_query_specific_chunks(
                    query=query,
                    original_hits=hits,
                    retrieve_fn=_ablation_retrieve,
                    threshold=settings.DEFENSE_L3_TRIGGER_THRESHOLD,
                    max_ablations=settings.DEFENSE_L3_MAX_ABLATIONS,
                    min_word_len=settings.DEFENSE_L3_MIN_WORD_LEN,
                )
                log.info("[L3] %s", _l3_summary(l3_report))
                if l3_report.suspicious_chunk_ids:
                    fname_by_cid = {h.chunk.id: h.document.filename for h in hits}
                    if settings.DEFENSE_L3_QUERY_ABLATION == "drop":
                        hits = _l3_filter(hits, l3_report, mode="drop")
                        log.info(
                            "[L3] выкинуто %d chunks из выдачи (drop mode)",
                            len(l3_report.suspicious_chunk_ids),
                        )
                    else:  # warn
                        # Плашку склеим в самом конце ответа агента — но
                        # вернуть в тул-выходе нельзя (LLM её увидит как
                        # часть выдачи). Поэтому копим в state и подмешиваем
                        # в answer_question post-hoc.
                        l3_warning = _l3_warning(l3_report, fname_by_cid)
                        if l3_warning:
                            with state.lock:
                                # Имя поля — l3_warnings (список), на случай
                                # нескольких search-вызовов в одном turn.
                                if not hasattr(state, "l3_warnings"):
                                    state.l3_warnings = []
                                state.l3_warnings.append(l3_warning)
            except Exception as e:
                log.warning("[L3] упал, пропускаем (атака пройдёт незамеченной): %s", e)

        _add_hits(state, hits)

        formatted = _format_hits_for_llm(hits)
        remaining = cap - used
        if remaining == 0:
            formatted += "\n\n[Это был последний доступный поиск. Сейчас формируй финальный ответ.]"
        elif remaining == 1:
            formatted += "\n\n[Остался ещё 1 поиск. Используй его только если ответ ещё неполный.]"
        return formatted

    return search_documents


@functools.lru_cache(maxsize=1)
def _get_llm() -> ChatOpenAI:
    """LLM-клиент кэшируется на процесс (он stateless и thread-safe). Агент же
    собирается per-request, потому что его tool несёт скоуп текущего юзера.
    lru_cache даёт thread-safe one-shot init без ручного double-checked locking.
    Если нужно сменить ключ или модель в рантайме — вызвать _get_llm.cache_clear()."""
    return ChatOpenAI(
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


_INLINE_CITE_RE = re.compile(
    r"\[[^\]]*\.(?:pdf|docx?|xlsx?|pptx|md|markdown|txt|csv)[^\]]*\]",
    re.IGNORECASE,
)


def _strip_inline_citations(text: str) -> str:
    """Удаляет из текста ассистента inline-цитаты вида `[file.docx, стр. 23]`.

    Без этого LLM, видя в истории свой предыдущий ответ с такими маркерами,
    лениво рехэширует их без вызова search_documents и тащит за собой
    выдуманные номера страниц. Цитаты мы и так показываем юзеру отдельным
    блоком из cited_chunks — в текст истории они не нужны.
    """
    return _INLINE_CITE_RE.sub("", text or "")


def _to_lc_messages(history: list[dict[str, str]]) -> list[Any]:
    out: list[Any] = []
    for m in history:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            # Из ответов ассистента вычищаем inline-цитаты — они для UI, не для LLM.
            out.append(AIMessage(content=_strip_inline_citations(content)))
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

    state = _RequestState(user_id=user_id, notebook_id=notebook_id)
    # Агент собирается per-request: tool несёт замыкание на state со скоупом
    # пользователя. LLM-клиент при этом переиспользуется (см. _get_llm).
    agent = create_react_agent(model=_get_llm(), tools=[_build_search_tool(state)])

    # Динамический system prompt: правила + актуальный список документов
    # текущего ноутбука. На мета-вопросах ("какие документы загружены") агент
    # отвечает по этому списку без вызова search_documents. На контентных —
    # имена файлов помогают подобрать осмысленные query.
    docs_for_prompt: list[dict[str, Any]] = []
    try:
        docs = db.list_documents(owner_user_id=user_id, notebook_id=notebook_id)
        for d in docs:
            if d.status == "ready":
                docs_for_prompt.append({
                    "filename": d.filename,
                    "chunk_count": d.chunk_count,
                    "file_type": d.file_type,
                })
    except Exception as e:
        log.warning("Не удалось получить список документов для prompt: %s", e)
    system_prompt = build_system_prompt(docs_for_prompt)

    messages = (
        [SystemMessage(content=system_prompt)]
        + _to_lc_messages(history)
        + [HumanMessage(content=question)]
    )

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

    cited = _collect_cited(result["messages"], user_id=user_id, notebook_id=notebook_id) or [
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
        for h in state.hits
    ]

    # Пост-сверка ответа с найденными фрагментами. Стоит +1 LLM-вызов на
    # вопрос — выгодно: ловит галлюцинации, на которые сам агент не падает
    # (например, придуманные цифры рядом с правильно процитированной страницей).
    verification: dict[str, Any] = {"unsupported": [], "verified": True}
    if settings.ANSWER_VERIFICATION and answer and cited:
        try:
            verification = verify_answer(question, answer, cited)
        except Exception as e:
            log.warning("Сбой верификации ответа: %s", e)
        if verification.get("unsupported"):
            answer = append_verification_warning(answer, verification["unsupported"])
            log.info("Verification: %d неподтверждённых утверждений",
                     len(verification["unsupported"]))

    # ---- Defense L4: strict verifier на самих cited chunks ----
    # Проверяем найденные фрагменты на injection-паттерны. Если LLM выдала
    # ответ с цитированием отравленного чанка — добавляем плашку безопасности.
    # Не делает LLM-вызовов, всё на регексах.
    strict: dict[str, Any] = {"suspicious": False, "findings": []}
    if settings.DEFENSE_L4_STRICT_VERIFIER and cited:
        try:
            strict = strict_verify(cited)
            if strict.get("suspicious"):
                answer = append_strict_warning(answer, strict)
                log.info("[L4] suspicious findings: %d", len(strict.get("findings") or []))
        except Exception as e:
            log.warning("[L4] strict verifier failed: %s", e)

    # ---- Defense L3: warn-режим — приклеиваем накопленные плашки в конец ----
    # В drop-режиме L3 уже выкинул подозрительные чанки из выдачи на этапе
    # search_documents tool, и сюда они не попали. В warn-режиме чанки остались
    # в выдаче (LLM могла их использовать в ответе), но мы добавляем плашку
    # с информацией пользователю.
    l3_warnings = list(getattr(state, "l3_warnings", []) or [])
    if l3_warnings:
        # Дедуп: один и тот же документ часто всплывает в нескольких search-
        # вызовах одного turn'а. Склеиваем в одну плашку.
        seen = set()
        unique = []
        for w in l3_warnings:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        answer = (answer.rstrip() + "\n" + "\n".join(unique)).strip()

    return {
        "answer": answer,
        "cited_chunks": cited,
        "verification": verification,
        "strict": strict,
    }


def _collect_cited(
    messages: list,
    user_id: Optional[int] = None,
    notebook_id: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Собирает cited_chunks из ToolMessage-ов. Проверяет, что каждый chunk_id
    принадлежит текущему пользователю/ноутбуку: текст ToolMessage может быть
    подделан (отравленный чанк с литералом «[chunk_id=42]» в теле, либо LLM,
    повторяющий маркер из истории) — без owner-фильтра отсюда могут утечь
    чанки чужих документов в response.cited_chunks."""
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

    if user_id is not None:
        owners = db.get_chunk_owners(order)
        order = [cid for cid in order if owners.get(cid) == user_id]
    if notebook_id is not None and order:
        notebooks = db.get_chunk_notebooks(order)
        order = [cid for cid in order if notebooks.get(cid) == notebook_id]

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
