"""Эвристики над текстом запроса:

- detect_entities(): даты, версии, артикулы, цифровые коды, заглавные
  аббревиатуры, токены с цифрами. Используется чтобы на «энтити-запросах»
  переключаться в BM25-only — embedding на коротких терминах с цифрами
  стабильно теряет точные токены.

- detect_intent(): грубая классификация запроса: definition / comparison /
  list / temporal / general. Используется для адаптивного top_k и решения,
  стоит ли применять HyDE.

Намеренно простая логика на regex'ах — без ML, без NER. Этого достаточно,
чтобы вытащить очевидные кейсы и не делать поиск глупее, чем он мог бы быть.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# --- Сущности ---

# Дата в любом разумном формате: 2024-01-15, 15.01.2024, 01/15/2024, 2024 г.,
# Q1 2024, 2024-Q1, 2024 год.
_DATE_PATTERNS = [
    re.compile(r"\b\d{4}[-./]\d{1,2}[-./]\d{1,2}\b"),
    re.compile(r"\b\d{1,2}[-./]\d{1,2}[-./]\d{2,4}\b"),
    re.compile(r"\b\d{4}\s*(?:г\.|год[ау]?)\b", re.IGNORECASE),
    re.compile(r"\b(?:Q|К|кв)\s*[1-4]\s*[-' ]?\s*\d{2,4}\b", re.IGNORECASE),
    re.compile(r"\b\d{4}\s*[-' ]\s*Q\s*[1-4]\b", re.IGNORECASE),
]

# Код / артикул / версия: GOST-12345, ISO-9001, v1.2.3, 1С-Бухгалтерия,
# CVE-2024-1234, Q4-2024-FIN-001
_CODE_PATTERNS = [
    re.compile(r"\b[A-ZА-Я]{2,8}[- ]?\d{2,}\b"),
    re.compile(r"\bv\.?\s*\d+(?:\.\d+){1,3}\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+){2,}\b"),                  # 1.2.3.4
    re.compile(r"\b№\s*\d+(?:[-/]\d+)*\b"),
    re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE),
]

# Аббревиатура заглавными буквами 2-8 символов: API, JSON, ФНС, НДС, RAG-агент.
_ACRONYM_RE = re.compile(r"\b[A-ZА-Я][A-ZА-Я0-9]{1,7}\b")

# Токены, которые содержат и буквы, и цифры — почти всегда это идентификатор.
_ALNUM_MIX_RE = re.compile(r"\b(?:[A-Za-zА-Яа-я]+\d+|\d+[A-Za-zА-Яа-я]+)[A-Za-zА-Яа-я0-9]*\b")

# Чистые числа длины ≥3 (не обязательно сущность, но в коротких запросах
# сильный сигнал для BM25). Не считаем 1-2-значные.
_LONG_NUMBER_RE = re.compile(r"\b\d{3,}\b")

# Слова-обороты, которые превращают вопрос в семантический даже при наличии
# сущностей: «когда выпустили», «расскажи о», «опиши». При них даже ID-запрос
# нужен через embedding.
# Берём только корни — этого достаточно, чтобы поймать любую падежную форму
# (какой/какого/каком/какую/какие → «как»; расскажи/расскажите → «расскаж»).
_QUESTION_WORDS_RE = re.compile(
    r"\b(что|кто|где|когда|как\w*|зачем|почему|сколько|опиш\w*|"
    r"расскаж\w*|объясн\w*|сравни\w*|перечисл\w*|"
    r"what|who|where|when|how|why|describe|explain|compare|list)\b",
    re.IGNORECASE,
)


@dataclass
class EntityProfile:
    entities: list[str]
    has_question_word: bool
    n_words: int

    @property
    def is_entity_heavy(self) -> bool:
        """Запрос — это в основном сущности (даты/коды), без вопросительных
        оборотов. Такой имеет смысл искать BM25-only."""
        if not self.entities:
            return False
        if self.has_question_word:
            return False
        if self.n_words == 0:
            return False
        # Оценка: уникальных сущностей >= 1, и они занимают ≥ 30% слов.
        return len(self.entities) / max(1, self.n_words) >= 0.3


def detect_entities(query: str) -> EntityProfile:
    """Извлекает узнаваемые сущности из запроса. Возвращает EntityProfile с
    флагом is_entity_heavy — эвристикой для роутинга в BM25-only."""
    text = query.strip()
    found: list[str] = []
    for pat in _DATE_PATTERNS:
        found.extend(m.group(0) for m in pat.finditer(text))
    for pat in _CODE_PATTERNS:
        found.extend(m.group(0) for m in pat.finditer(text))
    found.extend(m.group(0) for m in _ALNUM_MIX_RE.finditer(text))
    # Аббревиатуры — только если их 1-2 в запросе и нет вопросительных слов
    acronyms = [m.group(0) for m in _ACRONYM_RE.finditer(text)]
    if 0 < len(acronyms) <= 2:
        found.extend(acronyms)
    found.extend(m.group(0) for m in _LONG_NUMBER_RE.finditer(text))

    # Убираем дубли с сохранением порядка
    seen: set[str] = set()
    uniq: list[str] = []
    for f in found:
        key = f.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(f)

    return EntityProfile(
        entities=uniq,
        has_question_word=bool(_QUESTION_WORDS_RE.search(text)),
        n_words=len(re.findall(r"\w+", text)),
    )


# --- Интент ---

_DEFINITION_PATTERNS = [
    re.compile(r"\bчто\s+так(?:ое|ой|ая|ие)\b", re.IGNORECASE),
    re.compile(r"\bопределени[ея]\b", re.IGNORECASE),
    re.compile(r"\bчто\s+знач(?:ит|ат)\b", re.IGNORECASE),
    re.compile(r"\bкак\s+понимать\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\b", re.IGNORECASE),
    re.compile(r"\bdefin(?:e|ition)\b", re.IGNORECASE),
    re.compile(r"\bmeaning\s+of\b", re.IGNORECASE),
]

_COMPARISON_PATTERNS = [
    re.compile(r"\bсравни\w*\b", re.IGNORECASE),
    re.compile(r"\b(?:в\s+чем\s+)?различ\w+\b", re.IGNORECASE),
    re.compile(r"\bотличи\w+\b", re.IGNORECASE),
    re.compile(r"\bпротив\b", re.IGNORECASE),
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\bdifferenc\w+\b", re.IGNORECASE),
]

_LIST_PATTERNS = [
    re.compile(r"\bперечисл\w+\b", re.IGNORECASE),
    re.compile(r"\bвсе\s+\w+", re.IGNORECASE),
    re.compile(r"\bкакие\s+\w+", re.IGNORECASE),
    re.compile(r"\bсписок\b", re.IGNORECASE),
    re.compile(r"\blist\s+(?:all|of)\b", re.IGNORECASE),
    re.compile(r"\benumerate\b", re.IGNORECASE),
]

_TEMPORAL_PATTERNS = [
    re.compile(r"\bкогда\b", re.IGNORECASE),
    re.compile(r"\bв\s+как(?:ом|ие|ой)\s+(?:год|период|квартал|месяц)\w*\b", re.IGNORECASE),
    re.compile(r"\bдат[аы]\b", re.IGNORECASE),
    re.compile(r"\bwhen\b", re.IGNORECASE),
]


def detect_intent(query: str) -> str:
    """Возвращает один из: 'definition', 'comparison', 'list', 'temporal',
    'general'. Используется для подстройки top_k и активации HyDE."""
    q = query.strip()
    if not q:
        return "general"
    for p in _DEFINITION_PATTERNS:
        if p.search(q):
            return "definition"
    for p in _COMPARISON_PATTERNS:
        if p.search(q):
            return "comparison"
    for p in _LIST_PATTERNS:
        if p.search(q):
            return "list"
    for p in _TEMPORAL_PATTERNS:
        if p.search(q):
            return "temporal"
    return "general"


def adaptive_top_k(query: str, base_k: int) -> int:
    """Подгоняет k под форму запроса. Никаких magic-чисел: правила прозрачные.

    - Сравнения и перечисления → нужно больше контекста (k * 2, до 15).
    - Очень короткие (≤ 3 слова) → больше контекста, чтобы компенсировать
      слабый сигнал в эмбеддинге (k + 3, до 12).
    - Длинные конкретные (≥ 15 слов) → меньше шума (max(base_k - 2, 4)).
    - Иначе — base_k."""
    intent = detect_intent(query)
    n_words = len(re.findall(r"\w+", query))
    if intent in ("comparison", "list"):
        return min(base_k * 2, 15)
    if n_words <= 3:
        return min(base_k + 3, 12)
    if n_words >= 15:
        return max(base_k - 2, 4)
    return base_k
