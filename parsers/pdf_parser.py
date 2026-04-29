import logging
from pathlib import Path
from typing import Any

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


def extract_pdf_spans(path: str | Path) -> dict[int, dict[str, Any]]:
    """Возвращает словарь {page_num: {width, height, spans: [{t, b}]}} с
    координатами text-spans в PDF user-space (origin bottom-left, y вверх).

    Используется для подсветки фрагментов в кастомном PDF-viewer (pdf.js
    рендерит страницу на canvas, мы рисуем overlay-прямоугольники по bbox).

    Формат каждого span:
        t — текст (полный, без strip — нужен для матчинга)
        b — [x0, y0, x1, y1] в pt, в системе координат pdf.js viewport.

    PyMuPDF возвращает y-down (origin top-left) — конвертируем в y-up чтобы
    дальше можно было звать `viewport.convertToViewportRectangle(...)` без
    дополнительных преобразований.
    """
    out: dict[int, dict[str, Any]] = {}
    try:
        doc = fitz.open(str(path))
    except Exception as e:
        log.warning("Не удалось открыть %s для извлечения spans: %s", path, e)
        return out

    try:
        for i, page in enumerate(doc):
            page_h = float(page.rect.height)
            page_w = float(page.rect.width)
            spans: list[dict[str, Any]] = []
            try:
                page_dict = page.get_text("dict") or {}
                for block in page_dict.get("blocks", []):
                    if block.get("type") != 0:  # 0 = text
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "")
                            if not text or not text.strip():
                                continue
                            x0, y0_top, x1, y1_top = span.get("bbox", [0, 0, 0, 0])
                            # PyMuPDF y-down → PDF user-space y-up (для pdf.js)
                            y0 = page_h - y1_top
                            y1 = page_h - y0_top
                            spans.append({
                                "t": text,
                                "b": [
                                    round(float(x0), 2),
                                    round(float(y0), 2),
                                    round(float(x1), 2),
                                    round(float(y1), 2),
                                ],
                            })
            except Exception as e:
                log.warning("Не удалось извлечь spans на стр. %d: %s", i + 1, e)
            out[i + 1] = {
                "width": round(page_w, 2),
                "height": round(page_h, 2),
                "spans": spans,
            }
    finally:
        doc.close()
    return out


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
