from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ParsedSegment:
    """Логический сегмент документа: страница PDF, лист XLSX, слайд PPTX или
    весь DOCX как один сегмент. Чанкер дальше разбивает text на чанки и
    переносит метаданные сегмента в каждый чанк."""

    text: str
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None
    slide_number: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
