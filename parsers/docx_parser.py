import logging
from pathlib import Path

import mammoth

from parsers.base import ParsedSegment

log = logging.getLogger(__name__)


def parse_docx(path: str | Path) -> list[ParsedSegment]:
    """Конвертирует DOCX в Markdown через mammoth. Заголовки, списки и
    таблицы сохраняют структуру, что улучшает дальнейший чанкинг."""
    with open(path, "rb") as f:
        result = mammoth.convert_to_markdown(f)
    text = (result.value or "").strip()
    if result.messages:
        for m in result.messages:
            log.debug("mammoth: %s", m.message)
    return [ParsedSegment(text=text, metadata={"source_type": "docx"})]
