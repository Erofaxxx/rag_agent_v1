import logging
from pathlib import Path

from parsers.base import ParsedSegment

log = logging.getLogger(__name__)


def parse_text(path: str | Path) -> list[ParsedSegment]:
    """Просто читаем текст. Чанкер дальше разобьёт по абзацам/предложениям."""
    text = _read_text_robust(Path(path))
    return [ParsedSegment(text=text.strip(), metadata={"source_type": "txt"})]


def parse_markdown(path: str | Path) -> list[ParsedSegment]:
    """Markdown — тот же текст, но в metadata помечаем для возможной разной
    логики чанкинга в будущем (сейчас сепараторы '\n## ', '\n### ' уже
    приоритетны в общем чанкере)."""
    text = _read_text_robust(Path(path))
    return [ParsedSegment(text=text.strip(), metadata={"source_type": "md"})]


def _read_text_robust(p: Path) -> str:
    """Пробуем UTF-8, потом cp1251 (типичная Windows-кодировка для русских txt),
    потом latin-1 как последний fallback (никогда не падает)."""
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")
