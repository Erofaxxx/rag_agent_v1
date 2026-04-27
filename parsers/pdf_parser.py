import logging
from pathlib import Path

import fitz

from config import settings
from parsers.base import ParsedSegment

log = logging.getLogger(__name__)


def parse_pdf(path: str | Path) -> list[ParsedSegment]:
    """Извлекает текст постранично через PyMuPDF. Если средний объём текста
    на страницу мал — считаем, что это скан, и прогоняем OCR."""
    doc = fitz.open(str(path))
    segments: list[ParsedSegment] = []
    total_chars = 0
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            text = text.strip()
            total_chars += len(text)
            segments.append(
                ParsedSegment(
                    text=text,
                    page_number=i + 1,
                    metadata={"source_type": "pdf"},
                )
            )
    finally:
        doc.close()

    avg_per_page = total_chars / max(1, len(segments))
    if avg_per_page < settings.OCR_MIN_CHARS_PER_PAGE:
        log.info(
            "PDF %s выглядит как скан (avg %.1f chars/page), включаю OCR",
            path,
            avg_per_page,
        )
        return _ocr_pdf(path)
    return segments


def _ocr_pdf(path: str | Path) -> list[ParsedSegment]:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception as e:
        log.error("OCR недоступен: %s. Установите poppler-utils и tesseract", e)
        return []

    segments: list[ParsedSegment] = []
    images = convert_from_path(str(path), dpi=200)
    for i, img in enumerate(images):
        try:
            text = pytesseract.image_to_string(img, lang=settings.OCR_LANGUAGES) or ""
        except Exception as e:
            log.warning("OCR ошибся на стр. %d: %s", i + 1, e)
            text = ""
        segments.append(
            ParsedSegment(
                text=text.strip(),
                page_number=i + 1,
                metadata={"source_type": "pdf", "ocr": True},
            )
        )
    return segments
