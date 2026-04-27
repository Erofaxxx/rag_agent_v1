from typing import Any

from config import settings
from parsers.base import ParsedSegment


SEPARATORS: list[str] = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Рекурсивно разбивает текст по приоритетным сепараторам, чтобы куски
    были близки к chunk_size с перекрытием overlap. Размер считаем в символах
    (для русского языка ~3-4 символа на токен BGE-M3)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    for sep in SEPARATORS:
        if sep == "":
            return _split_by_size(text, chunk_size, overlap)
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks: list[str] = []
        current = ""
        for part in parts:
            piece = (sep + part) if current else part
            if len(current) + len(piece) <= chunk_size:
                current += piece
            else:
                if current.strip():
                    chunks.append(current.strip())
                if len(part) > chunk_size:
                    chunks.extend(_split_text(part, chunk_size, overlap))
                    current = ""
                else:
                    current = part
        if current.strip():
            chunks.append(current.strip())
        return _add_overlap(chunks, overlap)
    return _split_by_size(text, chunk_size, overlap)


def _split_by_size(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = out[-1][-overlap:] if len(out[-1]) > overlap else out[-1]
        merged = (prev_tail + " " + chunks[i]).strip()
        out.append(merged)
    return out


def chunk_segments(
    segments: list[ParsedSegment],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict[str, Any]]:
    """На вход — сегменты от парсеров. На выход — плоский список словарей
    готовых для записи в БД и эмбеддинга."""
    cs = chunk_size or settings.CHUNK_SIZE
    ov = chunk_overlap or settings.CHUNK_OVERLAP
    out: list[dict[str, Any]] = []
    idx = 0
    for seg in segments:
        if not seg.text or not seg.text.strip():
            continue
        for piece in _split_text(seg.text, cs, ov):
            out.append(
                {
                    "chunk_index": idx,
                    "text": piece,
                    "page_number": seg.page_number,
                    "sheet_name": seg.sheet_name,
                    "slide_number": seg.slide_number,
                    "metadata": dict(seg.metadata),
                }
            )
            idx += 1
    return out
