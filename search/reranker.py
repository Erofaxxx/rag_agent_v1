"""Реранкер top-N кандидатов из retrieval-этапа.

Зачем: top-N от FAISS / BM25 / RRF — это «можно посмотреть», но порядок
внутри топа стохастичен и часто просто неверен. Cross-encoder (или хороший
LLM) переоценивает пары (query, chunk) и даёт реальный rerank, что заметно
поднимает качество ответа.

Поддерживаемые провайдеры:
- "off"  — не реранкить, просто обрезать до top_k.
- "llm"  — реранк через основную DeepSeek (без доп. установки). +1 LLM вызов.
- "ce"   — локальный cross-encoder (BGE reranker), нужен sentence-transformers.

Lazy-init: модель / клиент создаётся при первом обращении и переиспользуется.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from search.faiss_index import SearchHit

log = logging.getLogger(__name__)


_lock = threading.Lock()
_ce_model = None        # cross-encoder
_llm: Optional[ChatOpenAI] = None


def _get_llm() -> Optional[ChatOpenAI]:
    """Отдельный LLM-клиент с temperature=0 — для детерминированного скоринга."""
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
                max_tokens=600,
                api_key=settings.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": settings.OPENROUTER_HTTP_REFERER,
                    "X-Title": settings.OPENROUTER_X_TITLE,
                },
                timeout=30,
            )
    return _llm


def _get_ce():
    """Lazy-load локального cross-encoder. Возвращает None, если недоступен."""
    global _ce_model
    if _ce_model is not None:
        return _ce_model
    with _lock:
        if _ce_model is not None:
            return _ce_model
        try:
            from sentence_transformers import CrossEncoder
        except Exception as e:
            log.warning(
                "RERANKER_PROVIDER=ce, но sentence-transformers не установлен (%s). "
                "Установи requirements-bge-fallback.txt или переключись на llm/off.",
                e,
            )
            return None
        try:
            _ce_model = CrossEncoder(settings.RERANKER_CE_MODEL, max_length=512)
            log.info("Cross-encoder загружен: %s", settings.RERANKER_CE_MODEL)
        except Exception as e:
            log.warning("Не удалось загрузить cross-encoder %s: %s",
                        settings.RERANKER_CE_MODEL, e)
            _ce_model = None
    return _ce_model


# ---- LLM rerank ----

_LLM_SYSTEM = (
    "Ты — реранкер для поисковой системы по документам. На вход — запрос "
    "пользователя и пронумерованные фрагменты-кандидаты. Для каждого верни "
    "целочисленный score релевантности 0..10 (10 = прямой ответ на вопрос, "
    "0 = совершенно не относится). Учитывай: совпадение конкретики, дат, "
    "имён, терминов; смысловое соответствие; полноту ответа. Игнорируй "
    "красивость языка.\n"
    "Возвращай ТОЛЬКО JSON-массив объектов вида "
    '[{"id": <номер>, "score": <0..10>}, ...] без преамбул и markdown.'
)


def _llm_rerank(query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
    llm = _get_llm()
    if llm is None or not hits:
        return hits[:top_k]

    # Готовим компактный листинг — обрезаем сниппеты, чтобы влезть по токенам.
    # ~400 chars × 20 = 8K chars ≈ 2.5K токенов на DeepSeek — приемлемо.
    payload_lines: list[str] = []
    for i, h in enumerate(hits):
        snippet = h.chunk.text.replace("\n", " ").strip()
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        payload_lines.append(f"[{i}] {snippet}")
    payload = f"Вопрос: {query}\n\nКандидаты:\n" + "\n".join(payload_lines)

    try:
        msg = llm.invoke([
            SystemMessage(content=_LLM_SYSTEM),
            HumanMessage(content=payload),
        ])
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
    except Exception as e:
        log.warning("LLM rerank упал (%s), fallback на исходный порядок", e)
        return hits[:top_k]

    scores = _parse_llm_scores(text, n=len(hits))
    if not scores:
        log.debug("LLM rerank вернул нечитабельный JSON, fallback на исходный порядок")
        return hits[:top_k]

    # Сохраняем dense-score в SearchHit, но используем LLM-score для сортировки.
    # Для отображения в UI это не важно — main score остаётся прежним.
    paired = sorted(
        enumerate(hits),
        key=lambda p: scores.get(p[0], -1.0),
        reverse=True,
    )
    return [h for _, h in paired[:top_k]]


_JSON_BLOCK_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_llm_scores(text: str, n: int) -> dict[int, float]:
    """Достаёт оценки из ответа LLM. Терпим к мусору вокруг JSON."""
    if not text:
        return {}
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except Exception:
        return {}
    out: dict[int, float] = {}
    if not isinstance(data, list):
        return {}
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("id"))
            sc = float(item.get("score", 0))
        except Exception:
            continue
        if 0 <= idx < n:
            out[idx] = sc
    return out


# ---- Cross-encoder rerank ----

def _ce_rerank(query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
    model = _get_ce()
    if model is None or not hits:
        return hits[:top_k]
    pairs = [(query, h.chunk.text[:2000]) for h in hits]
    try:
        scores = model.predict(pairs).tolist()
    except Exception as e:
        log.warning("Cross-encoder rerank упал (%s)", e)
        return hits[:top_k]
    paired = sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)
    return [h for h, _ in paired[:top_k]]


# ---- Public ----

def rerank(query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
    """Реранк top-N до top-K. Провайдер выбирается из settings.RERANKER_PROVIDER."""
    if not hits:
        return []
    provider = (settings.RERANKER_PROVIDER or "off").lower()
    # Защита: если кандидатов уже мало — реранк не нужен.
    if provider == "off" or len(hits) <= top_k:
        return hits[:top_k]
    if provider == "llm":
        return _llm_rerank(query, hits, top_k)
    if provider == "ce":
        return _ce_rerank(query, hits, top_k)
    log.warning("Неизвестный RERANKER_PROVIDER=%r, fallback на off", provider)
    return hits[:top_k]
