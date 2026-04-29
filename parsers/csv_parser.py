import logging
from pathlib import Path

import pandas as pd

from parsers.base import ParsedSegment

log = logging.getLogger(__name__)

SMALL_CSV_THRESHOLD = 100  # строк


def parse_csv(path: str | Path) -> list[ParsedSegment]:
    """CSV — пробуем sep=',' и ';', выбираем тот, что даёт больше колонок.
    Маленькие — целиком как Markdown-таблица; большие — построчно."""
    p = Path(path)
    df = _read_csv_robust(p)
    if df is None or df.empty:
        log.warning("CSV %s пустой или нечитаемый", p)
        return []

    df = df.fillna("")
    if len(df) <= SMALL_CSV_THRESHOLD:
        text = "# CSV: " + p.name + "\n\n" + df.to_markdown(index=False)
        return [
            ParsedSegment(
                text=text,
                metadata={"source_type": "csv", "rows": len(df)},
            )
        ]

    segments: list[ParsedSegment] = []
    headers = [str(c) for c in df.columns]
    for row_idx, row in df.iterrows():
        parts = [f"{h}: {row[h]}" for h in headers if str(row[h]).strip()]
        if not parts:
            continue
        row_text = f"Строка {int(row_idx) + 2}: " + ", ".join(parts)
        segments.append(
            ParsedSegment(
                text=row_text,
                metadata={"source_type": "csv", "row_index": int(row_idx) + 2},
            )
        )
    return segments


def _read_csv_robust(p: Path) -> pd.DataFrame | None:
    """Пробуем разные кодировки и разделители."""
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        for sep in (None, ",", ";", "\t"):  # None = pandas auto-detect
            try:
                df = pd.read_csv(p, encoding=enc, sep=sep, engine="python", dtype=str)
                if df is not None and len(df.columns) > 0:
                    return df
            except Exception:
                continue
    log.error("Не удалось прочитать CSV %s ни в одной комбинации", p)
    return None
