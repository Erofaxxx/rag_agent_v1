import logging
from pathlib import Path

import pandas as pd

from parsers.base import ParsedSegment

log = logging.getLogger(__name__)

SMALL_SHEET_THRESHOLD = 50


def parse_xlsx(path: str | Path) -> list[ParsedSegment]:
    """Каждый лист обрабатывается отдельно. Маленькие листы — целиком как
    Markdown-таблица; большие — построчно как 'Заголовок: значение'."""
    segments: list[ParsedSegment] = []
    try:
        xls = pd.ExcelFile(str(path))
    except Exception as e:
        log.error("Не удалось открыть xlsx %s: %s", path, e)
        return []

    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
        except Exception as e:
            log.warning("Не удалось прочитать лист %s: %s", sheet_name, e)
            continue
        df = df.fillna("")
        if df.empty:
            continue

        if len(df) <= SMALL_SHEET_THRESHOLD:
            text = f"# Лист: {sheet_name}\n\n" + df.to_markdown(index=False)
            segments.append(
                ParsedSegment(
                    text=text,
                    sheet_name=sheet_name,
                    metadata={"source_type": "xlsx", "rows": len(df)},
                )
            )
        else:
            headers = [str(c) for c in df.columns]
            for row_idx, row in df.iterrows():
                parts = [f"{h}: {row[h]}" for h in headers if str(row[h]).strip()]
                if not parts:
                    continue
                row_text = (
                    f"Лист '{sheet_name}', строка {int(row_idx) + 2}: " + ", ".join(parts)
                )
                segments.append(
                    ParsedSegment(
                        text=row_text,
                        sheet_name=sheet_name,
                        metadata={
                            "source_type": "xlsx",
                            "row_index": int(row_idx) + 2,
                        },
                    )
                )
    return segments
