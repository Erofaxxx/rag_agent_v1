"""Парсеры старых бинарных форматов Word/Excel через LibreOffice headless.

LibreOffice устанавливается в install.sh. На рабочей машине поставить:
  brew install --cask libreoffice          # macOS
  apt install libreoffice                  # Ubuntu

Конвертим .doc → .docx, .xls → .xlsx во временную папку и переотдаём
существующим парсерам (mammoth и openpyxl). Это простой и проверенный
путь — нативные либы для .doc/.xls в Python работают плохо или платные.
"""
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from parsers.base import ParsedSegment
from parsers.docx_parser import parse_docx
from parsers.xlsx_parser import parse_xlsx

log = logging.getLogger(__name__)


def _convert(input_path: Path, target_format: str) -> Path | None:
    """Запускает libreoffice --headless --convert-to. Возвращает путь к
    сконвертированному файлу или None при ошибке."""
    if not shutil.which("libreoffice") and not shutil.which("soffice"):
        log.error("LibreOffice не установлен (libreoffice/soffice не найден в PATH)")
        return None

    bin_name = "libreoffice" if shutil.which("libreoffice") else "soffice"
    out_dir = Path(tempfile.mkdtemp(prefix="rag_loconv_"))
    try:
        # --safe-mode защищает от user-profile-corruption если несколько
        # параллельных конверсий
        result = subprocess.run(
            [
                bin_name, "--headless", "--safe-mode",
                "--convert-to", target_format,
                "--outdir", str(out_dir),
                str(input_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            log.error(
                "LibreOffice convert failed (rc=%d): stdout=%s stderr=%s",
                result.returncode, result.stdout[:500], result.stderr[:500],
            )
            return None

        # Имя выходного файла = stem.target_format
        candidates = list(out_dir.glob(f"*.{target_format}"))
        if not candidates:
            log.error("LibreOffice convert не создал .%s в %s", target_format, out_dir)
            return None
        return candidates[0]
    except subprocess.TimeoutExpired:
        log.error("LibreOffice convert timeout (120s) для %s", input_path)
        return None
    except Exception as e:
        log.exception("Ошибка LibreOffice convert: %s", e)
        return None


def parse_doc(path: str | Path) -> list[ParsedSegment]:
    """Старый .doc → конвертим в .docx → парсим через mammoth."""
    p = Path(path)
    converted = _convert(p, "docx")
    if not converted:
        log.warning("Не удалось сконвертировать %s, возвращаю пусто", p)
        return []
    try:
        segments = parse_docx(converted)
        for s in segments:
            s.metadata["source_type"] = "doc"
            s.metadata["converted_from"] = "doc"
        return segments
    finally:
        try:
            shutil.rmtree(converted.parent, ignore_errors=True)
        except Exception:
            pass


def parse_xls(path: str | Path) -> list[ParsedSegment]:
    """Старый .xls → конвертим в .xlsx → парсим через openpyxl/pandas."""
    p = Path(path)
    converted = _convert(p, "xlsx")
    if not converted:
        log.warning("Не удалось сконвертировать %s, возвращаю пусто", p)
        return []
    try:
        segments = parse_xlsx(converted)
        for s in segments:
            s.metadata["source_type"] = "xls"
            s.metadata["converted_from"] = "xls"
        return segments
    finally:
        try:
            shutil.rmtree(converted.parent, ignore_errors=True)
        except Exception:
            pass
