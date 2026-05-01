import logging
from pathlib import Path
from typing import Any

import fitz

from config import settings
from parsers.base import ParsedSegment

log = logging.getLogger(__name__)


def parse_pdf(path: str | Path) -> list[ParsedSegment]:
    """Извлекает текст постранично через PyMuPDF. Если средний объём текста
    на страницу мал — считаем, что это скан, и прогоняем OCR. Дополнительно
    через pdfplumber вытаскиваем таблицы — они идут отдельными сегментами,
    отрендеренными в Markdown, чтобы при поиске не размывались между строками."""
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

    # Таблицы — отдельные сегменты, чтобы при чанкинге не порвать строку
    # между двумя чанками. Из текстового потока PyMuPDF их тоже вытащит, но
    # без структуры.
    table_segments = _extract_tables(path)
    if table_segments:
        log.info("PDF %s: добавлено %d таблиц", path, len(table_segments))
        segments.extend(table_segments)
    return segments


def _extract_tables(path: str | Path) -> list[ParsedSegment]:
    """Достаёт таблицы постранично через pdfplumber и рендерит в Markdown.
    Если pdfplumber не установлен — возвращаем пустой список (мягкий фолбэк)."""
    try:
        import pdfplumber
    except Exception as e:
        log.debug("pdfplumber не установлен (%s), таблицы PDF не извлекаются", e)
        return []

    out: list[ParsedSegment] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables() or []
                except Exception as e:
                    log.debug("pdfplumber.extract_tables(стр.%d) ошибся: %s", i + 1, e)
                    continue
                for ti, table in enumerate(tables):
                    md = _table_to_markdown(table)
                    if not md:
                        continue
                    out.append(
                        ParsedSegment(
                            text=f"[Таблица {ti + 1} на стр. {i + 1}]\n{md}",
                            page_number=i + 1,
                            metadata={"source_type": "pdf", "kind": "table"},
                        )
                    )
    except Exception as e:
        log.warning("pdfplumber падает на %s: %s", path, e)
    return out


def _table_to_markdown(rows: list[list[Any]] | None) -> str:
    """Минимальный рендер списка строк в Markdown-таблицу. Пустые/нечитаемые
    ячейки заменяем пустой строкой, чтобы не ломать ширину."""
    if not rows or len(rows) < 2:
        # 1-строчная «таблица» обычно ложно-положительная (одна большая ячейка).
        return ""
    max_cols = max(len(r) for r in rows if r)
    if max_cols < 2:
        return ""

    def cell(x: Any) -> str:
        if x is None:
            return ""
        s = str(x).replace("\n", " ").replace("|", "/").strip()
        return s

    header = rows[0]
    body = rows[1:]
    head_cells = [cell(c) for c in header] + [""] * (max_cols - len(header))
    md = "| " + " | ".join(head_cells) + " |\n"
    md += "| " + " | ".join(["---"] * max_cols) + " |\n"
    for r in body:
        cells = [cell(c) for c in r] + [""] * (max_cols - len(r))
        md += "| " + " | ".join(cells) + " |\n"
    return md.rstrip()


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
