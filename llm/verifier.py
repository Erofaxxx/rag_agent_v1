"""Пост-генерационная сверка ответа с найденными чанками.

Идея простая: после того как агент сформировал ответ, мы делаем ещё один
дешёвый LLM-вызов с инструкцией «возьми каждое фактическое утверждение из
ответа и проверь, есть ли оно в чанках». Возвращается список
unsupported claims (если есть).

Без retry — слишком дорого и сложно. На уровне UI к ответу мы лишь
добавляем дисклеймер о неподтверждённых утверждениях. Пользователю это
честнее, чем молчаливая галлюцинация.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings

log = logging.getLogger(__name__)


_lock = threading.Lock()
_llm: Optional[ChatOpenAI] = None


def _get_llm() -> Optional[ChatOpenAI]:
    global _llm
    if _llm is not None:
        return _llm
    if not settings.OPENROUTER_API_KEY:
        return None
    with _lock:
        if _llm is None:
            _llm = ChatOpenAI(
                model=settings.LLM_MODEL,
                temperature=0.0,
                max_tokens=500,
                api_key=settings.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": settings.OPENROUTER_HTTP_REFERER,
                    "X-Title": settings.OPENROUTER_X_TITLE,
                },
                timeout=25,
            )
    return _llm


_SYSTEM = (
    "Ты — fact-checker. На вход ты получаешь:\n"
    "  - Вопрос пользователя\n"
    "  - Ответ ассистента (на основе документов)\n"
    "  - Найденные фрагменты документов\n"
    "Раздели ответ на отдельные фактические утверждения и для каждого скажи, "
    "поддержано ли оно фрагментами. Игнорируй мета-фразы («в документах "
    "указано», «согласно файлу X»), вводные обороты, общие связки. "
    "Не считай неподтверждёнными общеизвестные структуры русского языка — "
    "только конкретные факты, цифры, даты, имена.\n"
    "Верни ТОЛЬКО JSON без преамбул:\n"
    '{"unsupported": ["короткое описание неподтверждённого утверждения", ...]}\n'
    "Если всё ОК — пустой массив."
)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _format_chunks(cited: list[dict[str, Any]], char_budget: int = 6000) -> str:
    """Компактный листинг чанков с обрезкой по бюджету. Берём snippet из cited."""
    parts: list[str] = []
    used = 0
    for c in cited:
        snippet = (c.get("snippet") or "").strip()
        if not snippet:
            continue
        head = c.get("filename") or "?"
        if c.get("page_number"):
            head += f", стр. {c['page_number']}"
        block = f"[{head}] {snippet}"
        if used + len(block) > char_budget:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def verify_answer(
    question: str,
    answer: str,
    cited_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Возвращает {"unsupported": [...], "verified": bool}.

    Если LLM недоступна или ответ нечитаем — считаем verified=True и пустой
    список (fail-open: лучше показать ответ как есть, чем падать)."""
    if not answer or not cited_chunks:
        return {"unsupported": [], "verified": True}
    llm = _get_llm()
    if llm is None:
        return {"unsupported": [], "verified": True}

    chunks_text = _format_chunks(cited_chunks)
    if not chunks_text:
        return {"unsupported": [], "verified": True}

    user_payload = (
        f"Вопрос: {question}\n\n"
        f"Ответ ассистента:\n{answer}\n\n"
        f"Найденные фрагменты:\n{chunks_text}"
    )

    try:
        msg = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_payload),
        ])
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
    except Exception as e:
        log.warning("verify_answer: LLM ошибся (%s), пропускаем верификацию", e)
        return {"unsupported": [], "verified": True}

    match = _JSON_RE.search(text or "")
    if not match:
        return {"unsupported": [], "verified": True}
    try:
        data = json.loads(match.group(0))
    except Exception:
        return {"unsupported": [], "verified": True}

    raw = data.get("unsupported") or []
    if not isinstance(raw, list):
        return {"unsupported": [], "verified": True}
    unsupported = [str(x).strip() for x in raw if str(x).strip()]
    return {"unsupported": unsupported, "verified": not unsupported}


def append_verification_warning(answer: str, unsupported: list[str]) -> str:
    """Дописывает к ответу аккуратную плашку с неподтверждёнными утверждениями.
    Если их нет — возвращает ответ без изменений."""
    if not unsupported:
        return answer
    lines = "\n".join(f"  • {item}" for item in unsupported[:5])
    return (
        f"{answer.rstrip()}\n\n"
        f"⚠️ Часть утверждений напрямую не подтверждена найденными фрагментами; "
        f"перепроверьте по источникам:\n{lines}"
    )
