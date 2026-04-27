import logging
from pathlib import Path

from parsers.base import ParsedSegment
from parsers.docx_parser import parse_docx
from parsers.pdf_parser import parse_pdf
from parsers.pptx_parser import parse_pptx
from parsers.xlsx_parser import parse_xlsx

log = logging.getLogger(__name__)


# magic-байты + расширение → канонический тип
EXT_MAP = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".pptx": "pptx",
}

MIME_MAP = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel.sheet.macroenabled.12": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
}


def detect_file_type(path: str | Path) -> str | None:
    """Определяет тип по magic-байтам через python-magic. Если magic недоступен
    или вернул не то, fallback на расширение."""
    p = Path(path)
    try:
        import magic

        mime = magic.from_file(str(p), mime=True)
        if mime in MIME_MAP:
            return MIME_MAP[mime]
        # Если magic вернул octet-stream, всё ещё попробуем расширение
        log.debug("magic mime '%s' для %s — fallback на расширение", mime, p.name)
    except Exception as e:
        log.debug("python-magic недоступен (%s), fallback на расширение", e)

    ext = p.suffix.lower()
    return EXT_MAP.get(ext)


def parse_file(path: str | Path, file_type: str) -> list[ParsedSegment]:
    if file_type == "pdf":
        return parse_pdf(path)
    if file_type == "docx":
        return parse_docx(path)
    if file_type == "xlsx":
        return parse_xlsx(path)
    if file_type == "pptx":
        return parse_pptx(path)
    raise ValueError(f"Неподдерживаемый тип файла: {file_type}")
