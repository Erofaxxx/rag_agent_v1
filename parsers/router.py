import logging
from pathlib import Path

from parsers.base import ParsedSegment
from parsers.csv_parser import parse_csv
from parsers.docx_parser import parse_docx
from parsers.legacy_office import parse_doc, parse_xls
from parsers.pdf_parser import parse_pdf
from parsers.pptx_parser import parse_pptx
from parsers.text_parser import parse_markdown, parse_text
from parsers.xlsx_parser import parse_xlsx

log = logging.getLogger(__name__)


# Расширение → канонический тип
EXT_MAP = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".xls": "xls",
    ".pptx": "pptx",
    ".txt": "txt",
    ".md": "md",
    ".markdown": "md",
    ".csv": "csv",
}

# magic mime → канонический тип
MIME_MAP = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel.sheet.macroenabled.12": "xlsx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "text/plain": "txt",
    "text/markdown": "md",
    "text/csv": "csv",
    "text/x-csv": "csv",
    "application/csv": "csv",
}


def detect_file_type(path: str | Path) -> str | None:
    """Определяет тип по magic-байтам через python-magic. Если magic недоступен
    или вернул не то, fallback на расширение.

    Важная деталь: для текстовых форматов (txt, md, csv) magic часто возвращает
    text/plain — тогда расширение становится единственным источником правды
    о подтипе. Поэтому сначала пробуем расширение, потом magic как уточнение."""
    p = Path(path)
    ext_type = EXT_MAP.get(p.suffix.lower())

    try:
        import magic
        mime = magic.from_file(str(p), mime=True)
        magic_type = MIME_MAP.get(mime)
    except Exception as e:
        log.debug("python-magic недоступен (%s)", e)
        magic_type = None

    # Особый случай: для txt/md/csv доверяем расширению, magic в этом классе
    # размывает (может вернуть text/plain для всех трёх).
    if ext_type in {"txt", "md", "csv"}:
        return ext_type

    # Иначе — magic приоритетнее (защита от подмены расширения)
    return magic_type or ext_type


def parse_file(path: str | Path, file_type: str) -> list[ParsedSegment]:
    if file_type == "pdf":
        return parse_pdf(path)
    if file_type == "docx":
        return parse_docx(path)
    if file_type == "doc":
        return parse_doc(path)
    if file_type == "xlsx":
        return parse_xlsx(path)
    if file_type == "xls":
        return parse_xls(path)
    if file_type == "pptx":
        return parse_pptx(path)
    if file_type == "txt":
        return parse_text(path)
    if file_type == "md":
        return parse_markdown(path)
    if file_type == "csv":
        return parse_csv(path)
    raise ValueError(f"Неподдерживаемый тип файла: {file_type}")
