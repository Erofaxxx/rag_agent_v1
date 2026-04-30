"""Расширение запроса перед retrieval'ом:

- multi_query(): LLM генерирует 2 альтернативные формулировки. Все варианты
  ищутся параллельно, результаты объединяются через RRF в SearchService.
- hyde(): для определительных запросов LLM генерирует короткий гипотетический
  ответ. Эмбеддинг этого ответа гораздо ближе к нужному фрагменту в документе,
  чем эмбеддинг короткого запроса.
- reformulate(): если первая выдача оказалась слабой, просим LLM перефразировать
  иначе (с учётом найденных терминов).

LLM-клиент — отдельный, лёгкий ChatOpenAI. Не используем основного агента —
он живёт в llm/agent.py и нагружен tool-вызовами; для query rewrite это
overkill и риск рекурсии.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings

log = logging.getLogger(__name__)


_helper_lock = threading.Lock()
_helper_llm: Optional[ChatOpenAI] = None


def _get_helper() -> Optional[ChatOpenAI]:
    """Лёгкий ChatOpenAI для вспомогательных текстовых задач (rewrite/HyDE/
    verify). Используем тот же провайдер и модель, но температуру повыше для
    rewrite (0.3) — нужны разнообразные формулировки."""
    global _helper_llm
    if not settings.OPENROUTER_API_KEY:
        return None
    if _helper_llm is not None:
        return _helper_llm
    with _helper_lock:
        if _helper_llm is None:
            _helper_llm = ChatOpenAI(
                model=settings.LLM_MODEL,
                temperature=0.3,
                max_tokens=400,
                api_key=settings.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": settings.OPENROUTER_HTTP_REFERER,
                    "X-Title": settings.OPENROUTER_X_TITLE,
                },
                timeout=20,
            )
    return _helper_llm


_LIST_LINE_RE = re.compile(r"^\s*(?:\d+[\.\)]|[-*•])\s*(.+?)\s*$")


def _parse_list(text: str, max_items: int) -> list[str]:
    """Достаёт пункты из ответа LLM. Принимает форматы:
        1. ...
        - ...
        ...
    Возвращает уникальные строки до max_items."""
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        m = _LIST_LINE_RE.match(line)
        cand = m.group(1).strip() if m else line.strip()
        if not cand:
            continue
        # Срезаем кавычки, если LLM их налепила
        cand = cand.strip(' "“”«»\'')
        key = cand.lower()
        if key in seen or len(cand) < 3:
            continue
        seen.add(key)
        out.append(cand)
        if len(out) >= max_items:
            break
    return out


_REWRITE_SYSTEM = (
    "Ты помогаешь поисковому движку по корпоративным документам. На вход "
    "получаешь вопрос пользователя и возвращаешь {n} АЛЬТЕРНАТИВНЫХ "
    "формулировок того же запроса для лучшего retrieval'а. Сохраняй язык "
    "оригинала (русский или английский). Никаких преамбул, только список:\n"
    "1. ...\n2. ...\nКаждая формулировка должна сохранять смысл, но менять "
    "лексику: синонимы, развёрнутые/сжатые версии, термины, которые могут "
    "встретиться в документе. Не добавляй новых сущностей и фактов."
)


def multi_query(query: str, n_extra: int = 2) -> list[str]:
    """Возвращает [query] + до n_extra LLM-переформулировок. На ошибке LLM
    возвращает только оригинал — поиск не сломается."""
    query = (query or "").strip()
    if not query or n_extra <= 0:
        return [query] if query else []
    llm = _get_helper()
    if llm is None:
        return [query]
    try:
        msg = llm.invoke([
            SystemMessage(content=_REWRITE_SYSTEM.replace("{n}", str(n_extra))),
            HumanMessage(content=query),
        ])
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        extras = _parse_list(text, max_items=n_extra)
    except Exception as e:
        log.warning("multi_query: LLM ошибся (%s), используем оригинал", e)
        return [query]

    # Отбрасываем формулировки, дублирующие оригинал по нормализованному виду
    norm_orig = re.sub(r"\W+", "", query.lower())
    out = [query]
    for e in extras:
        if re.sub(r"\W+", "", e.lower()) == norm_orig:
            continue
        out.append(e)
    log.debug("multi_query: %s → %s", query, out)
    return out


_HYDE_SYSTEM = (
    "Сгенерируй короткий (1-3 предложения) гипотетический фрагмент документа, "
    "который мог бы являться ПРЯМЫМ ОТВЕТОМ на вопрос пользователя. Пиши "
    "на языке вопроса в стиле сухой документации — как будто это абзац из "
    "учебника или корпоративного руководства. Никаких преамбул, никаких "
    "оговорок «если бы», только сам фрагмент."
)


def hyde(query: str) -> Optional[str]:
    """Генерирует гипотетический ответ-документ для embed-поиска. Подходит для
    «что такое X» / definitional intents, где сам запрос слишком короткий."""
    q = (query or "").strip()
    if not q:
        return None
    llm = _get_helper()
    if llm is None:
        return None
    try:
        msg = llm.invoke([
            SystemMessage(content=_HYDE_SYSTEM),
            HumanMessage(content=q),
        ])
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        text = text.strip().strip('"“”«»')
        return text or None
    except Exception as e:
        log.warning("hyde: LLM ошибся (%s)", e)
        return None


_REFORMULATE_SYSTEM = (
    "Поисковая выдача оказалась слабой — переформулируй запрос так, чтобы "
    "найти релевантные фрагменты в документах. Используй другие синонимы, "
    "более конкретные термины или, наоборот, более общие. Сохрани язык. "
    "Верни ТОЛЬКО одну новую формулировку, без преамбул и кавычек."
)


def reformulate(query: str, hint_terms: list[str] | None = None) -> Optional[str]:
    """Генерирует одну альтернативную формулировку. hint_terms — слова из
    названий документов / уже найденных чанков, которые стоит попробовать."""
    q = (query or "").strip()
    if not q:
        return None
    llm = _get_helper()
    if llm is None:
        return None
    user_payload = q
    if hint_terms:
        terms = ", ".join(t for t in hint_terms[:8] if t)
        if terms:
            user_payload = f"{q}\n\nВозможные термины из документов: {terms}"
    try:
        msg = llm.invoke([
            SystemMessage(content=_REFORMULATE_SYSTEM),
            HumanMessage(content=user_payload),
        ])
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        text = text.strip().strip('"“”«»').splitlines()[0].strip()
        if not text or text.lower() == q.lower():
            return None
        return text
    except Exception as e:
        log.warning("reformulate: LLM ошибся (%s)", e)
        return None
