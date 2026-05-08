"""L1: ingest-time chunk sanitization.

Детектит prompt-injection паттерны и фейковые цитаты в тексте чанка ДО того,
как чанк попадёт в эмбеддинг и индекс. Это первая линия обороны: если
атакующий встроил «Игнорируй все предыдущие инструкции» прямо текстом —
такой чанк должен быть либо помечен (warn), либо отброшен (drop).

Формат отчёта (для каждого чанка):

    {
      "risk_score": float in [0, 1],   # суммарная оценка риска
      "categories": ["instruction_override", "role_switch", ...],
      "matched_phrases": ["игнорируй все предыдущие", ...],
      "action": "keep" | "warn" | "drop",
    }

Регексы покрывают:
- Override инструкций (рус/англ): "ignore previous", "забудь все"
- Role-switching: "you are now X", "теперь ты — X"
- Утечка system prompt: "system prompt", "</system>", "[INST]"
- Фейковые inline-цитаты: [file.pdf, стр. 23] — это формат самого агента,
  если такое встретилось в чанке, кто-то его подделывает
- Подозрительный Unicode: zero-width chars, RTL override, tag chars
- Большие base64/hex-блобы > N символов
- Прямые «выдай: ...» инструкции: "respond with", "ответь:"

Намеренно НЕ детектируем:
- Естественный язык про инструкции ("в инструкции написано Х") — много FP
- Code blocks (могут быть легитимны в технических документах)

Тесты этой логики живут в tests/test_l1_sanitize.py.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# ---- Категории и шаблоны ----

# Каждая категория = (вес, список регексов).
# Веса подобраны так, что один сильный матч + один слабый ≥ 0.6 (default threshold).

_PATTERNS: dict[str, tuple[float, list[re.Pattern]]] = {
    "instruction_override": (
        0.5,
        [
            re.compile(r"\bignore\s+(all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)", re.IGNORECASE),
            re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?)", re.IGNORECASE),
            re.compile(r"\bforget\s+(?:everything|all|the\s+above|previous)", re.IGNORECASE),
            re.compile(r"игнорируй(?:те)?\s+(?:все\s+)?(?:предыдущ\w+|выш\w+|ранее)", re.IGNORECASE),
            re.compile(r"забудь(?:те)?\s+(?:всё|все|выш\w+|предыдущ\w+)", re.IGNORECASE),
            re.compile(r"\bне\s+слушай\s+(?:предыдущ\w+|систем\w+)", re.IGNORECASE),
        ],
    ),
    "role_switch": (
        0.4,
        [
            re.compile(r"\byou\s+are\s+now\s+(?:a\s+|an\s+)?[a-z]", re.IGNORECASE),
            re.compile(r"\bfrom\s+now\s+on\s+you", re.IGNORECASE),
            re.compile(r"\bact\s+as\s+(?:a\s+|an\s+)?[a-z]+", re.IGNORECASE),
            re.compile(r"теперь\s+ты\s+[—–-]?\s*[a-zа-я]", re.IGNORECASE),
            re.compile(r"отныне\s+ты", re.IGNORECASE),
            re.compile(r"представь(?:ся|те)?,?\s+что\s+ты", re.IGNORECASE),
        ],
    ),
    "system_prompt_leak": (
        0.5,
        [
            re.compile(r"\b(?:system|developer)\s*[:>]\s*[a-zа-я]", re.IGNORECASE),
            re.compile(r"</?(?:system|instruction|prompt|s)\s*>", re.IGNORECASE),
            re.compile(r"\[INST\]|\[/INST\]"),
            re.compile(r"<\|im_start\|>|<\|im_end\|>"),
            re.compile(r"<<SYS>>|<</SYS>>"),
            re.compile(r"systemPrompt|system_prompt", re.IGNORECASE),
        ],
    ),
    "direct_instruction": (
        0.5,
        [
            re.compile(r"\brespond\s+(?:only\s+)?with\s+[\"']", re.IGNORECASE),
            re.compile(r"\boutput\s+(?:exactly|only)\s+[\"']", re.IGNORECASE),
            re.compile(r"всегда\s+отвечай(?:те)?\s*[:«\"]", re.IGNORECASE),
            re.compile(r"в\s+ответ(?:е)?\s+(?:всегда\s+)?(?:напиши|пиши|укажи)\b", re.IGNORECASE),
            re.compile(r"при\s+(?:вопрос\w+|запрос\w+)\s+про\s+\S+\s+(?:всегда\s+)?(?:отвечай|пиши)", re.IGNORECASE),
        ],
    ),
    "fake_citation": (
        0.4,
        [
            # Формат самого агента: [имя_файла.ext, стр. N] — если такое в тексте чанка,
            # значит кто-то имитирует вывод инструмента. Совпадение с _INLINE_CITE_RE
            # из llm/agent.py.
            re.compile(
                r"\[[^\]]*\.(?:pdf|docx?|xlsx?|pptx|md|markdown|txt|csv)[^\]]*\]",
                re.IGNORECASE,
            ),
            # Псевдо-маркер инструмента: [chunk_id=N]
            re.compile(r"\[chunk_id\s*=\s*\d+\]", re.IGNORECASE),
        ],
    ),
    "suspicious_unicode": (
        0.3,
        [
            # Zero-width spaces / joiners / BOM. Часто прячут инструкции
            # «между» словами. Перечисляем явно: коды не образуют непрерывный
            # диапазон, и литеральные range-выражения раньше захватывали
            # нерелевантную часть таблицы и/или пропускали часть нужных кодпоинтов.
            re.compile(r"[​‌‍⁠﻿]"),
            # Bidi override / isolate: U+202A..U+202E + U+2066..U+2069.
            re.compile(r"[‪-‮⁦-⁩]"),
            # Tag chars (Plane 14) — современный prompt-injection через emoji-tags.
            re.compile(r"[\U000E0000-\U000E007F]"),
        ],
    ),
}


# Безопасный «whitelist»: фразы, после которых instruction-override не считается
# атакой (в технических текстах могут встречаться легитимно).
_BENIGN_PREFIXES = [
    re.compile(r"в\s+инструкции\s+(?:сказано|указано|написано)", re.IGNORECASE),
    re.compile(r"the\s+manual\s+states", re.IGNORECASE),
    re.compile(r"according\s+to\s+the\s+(?:instruction|guideline|manual)", re.IGNORECASE),
]


@dataclass
class ChunkRisk:
    risk_score: float
    categories: list[str] = field(default_factory=list)
    matched_phrases: list[str] = field(default_factory=list)
    action: str = "keep"  # keep / warn / drop

    def to_dict(self) -> dict:
        return {
            "risk_score": round(self.risk_score, 3),
            "categories": self.categories,
            "matched_phrases": self.matched_phrases[:10],  # не раздуваем audit_log
            "action": self.action,
        }


def analyze_chunk(text: str) -> ChunkRisk:
    """Возвращает ChunkRisk для одного куска текста.

    risk_score — взвешенная сумма категорий (cap до 1.0). Категории
    суммируются, повторные матчи внутри одной категории — нет (чтобы
    одно длинное «игнорируй всё» не накручивало больше двух разных
    инъекций)."""
    if not text or not text.strip():
        return ChunkRisk(risk_score=0.0)

    # Нормализация: NFKC чтобы атакующий не использовал ﬁ вместо fi для обхода.
    norm = unicodedata.normalize("NFKC", text)

    matched_categories: dict[str, list[str]] = {}
    for cat, (weight, patterns) in _PATTERNS.items():
        for pat in patterns:
            for m in pat.finditer(norm):
                phrase = m.group(0)
                # Whitelist для instruction_override
                if cat == "instruction_override":
                    pre_ctx = norm[max(0, m.start() - 60) : m.start()]
                    if any(b.search(pre_ctx) for b in _BENIGN_PREFIXES):
                        continue
                matched_categories.setdefault(cat, []).append(phrase)

    risk = 0.0
    for cat, phrases in matched_categories.items():
        weight = _PATTERNS[cat][0]
        # Доп. вклад от количества матчей в категории, но плавно: log-like
        n = len(phrases)
        risk += weight * (1.0 + 0.15 * min(n - 1, 4))

    risk = min(risk, 1.0)

    return ChunkRisk(
        risk_score=risk,
        categories=sorted(matched_categories.keys()),
        matched_phrases=[p for ps in matched_categories.values() for p in ps],
    )


def decide_action(risk: ChunkRisk, mode: str, threshold: float) -> str:
    """Перевод (risk, mode) → action.

    mode из settings.DEFENSE_L1_SANITIZE: off / warn / drop.
    threshold из settings.DEFENSE_L1_RISK_THRESHOLD.
    """
    if mode == "off" or risk.risk_score < threshold:
        return "keep"
    if mode == "warn":
        return "warn"
    if mode == "drop":
        return "drop"
    return "keep"


def sanitize_chunks(
    chunks: list[dict],
    *,
    mode: str = "off",
    threshold: float = 0.6,
) -> tuple[list[dict], list[dict]]:
    """Пропускает список чанков (формат из chunking.chunker — dict с 'text')
    через L1-детектор.

    Возвращает (kept_chunks, report). report — список {chunk_index, ...} по
    ВСЕМ чанкам (включая kept), удобно складывать в audit_log.
    """
    kept: list[dict] = []
    report: list[dict] = []
    for i, ch in enumerate(chunks):
        text = ch.get("text", "") or ""
        risk = analyze_chunk(text)
        action = decide_action(risk, mode, threshold)
        risk.action = action
        report.append({"chunk_index": i, **risk.to_dict()})
        if action != "drop":
            kept.append(ch)
    return kept, report


def short_summary(report: list[dict]) -> str:
    """Краткая строка для логов: 'L1: 3/50 warn, 1/50 drop'."""
    n = len(report)
    n_warn = sum(1 for r in report if r["action"] == "warn")
    n_drop = sum(1 for r in report if r["action"] == "drop")
    if n_warn == 0 and n_drop == 0:
        return f"L1: {n} chunks clean"
    return f"L1: {n_warn}/{n} warn, {n_drop}/{n} drop"
