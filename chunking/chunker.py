import logging
from typing import Any, Optional

from config import settings
from parsers.base import ParsedSegment

log = logging.getLogger(__name__)


SEPARATORS: list[str] = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]


def adaptive_chunk_params(
    file_type: Optional[str],
    total_chars: int,
) -> tuple[int, int]:
    """Возвращает (chunk_size, overlap) под тип документа.

    Коротко: длинные нарративы (PDF) выигрывают от больших чанков с щедрым
    overlap'ом; табличные данные (XLSX/CSV) — от маленьких чанков с
    минимальным overlap'ом, чтобы строка не размывалась через границу;
    презентации (PPTX) — короткие, потому что слайд сам по себе атомарная
    единица, и переплетать их не надо."""
    if not settings.CHUNK_ADAPTIVE:
        return settings.CHUNK_SIZE, settings.CHUNK_OVERLAP

    ft = (file_type or "").lower()
    if ft in {"xlsx", "xls", "csv"}:
        return 1200, 150
    if ft == "pptx":
        return 1500, 200
    if ft in {"docx", "doc", "md", "markdown", "txt"}:
        return 2400, 400
    if ft == "pdf":
        # Для длинных PDF берём более крупные чанки — иначе одна тема
        # размазывается на 6-8 фрагментов и retrieval приносит дубликаты.
        if total_chars > 200_000:
            return 3000, 500
        return 2400, 400
    return settings.CHUNK_SIZE, settings.CHUNK_OVERLAP


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


def _semantic_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Разбивает текст по абзацам, эмбеддит соседние и склеивает их в чанки,
    пока косинусная схожесть выше порога ИЛИ пока не превышен chunk_size.

    Это снимает классический разрыв «посреди объяснения»: если два абзаца
    говорят про одно — они окажутся в одном чанке, даже если суммарно
    почти 2*chunk_size. Если про разное — граница пройдёт между ними.

    Стоит времени: на каждый абзац — 1 запрос в embedding API. На большом
    документе это десятки секунд. Поэтому фича опциональна.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        return _split_text(text, chunk_size, overlap)

    # Длинные параграфы сразу режем структурно — embed на 5K-абзаце смысла мало
    expanded: list[str] = []
    for p in paragraphs:
        if len(p) > chunk_size:
            expanded.extend(_split_text(p, chunk_size, overlap))
        else:
            expanded.append(p)
    paragraphs = expanded
    if len(paragraphs) <= 1:
        return paragraphs

    try:
        # Локальный импорт — чтобы chunker не тащил эмбеддинг при инициализации
        from embeddings import embedding_service
        import numpy as np  # noqa: WPS433 — используется только при включённом семантическом
        vecs = embedding_service.encode_passages(paragraphs)
    except Exception as e:
        log.warning("CHUNK_SEMANTIC=true, но embedding для чанкинга упал (%s). "
                    "Fallback на структурный сплит.", e)
        return _split_text(text, chunk_size, overlap)

    threshold = settings.CHUNK_SEMANTIC_THRESHOLD
    chunks: list[str] = []
    current = paragraphs[0]
    for i in range(1, len(paragraphs)):
        sim = float(np.dot(vecs[i - 1], vecs[i]))  # vecs нормализованы → cosine
        next_p = paragraphs[i]
        if sim >= threshold and len(current) + len(next_p) + 2 <= chunk_size:
            current = f"{current}\n\n{next_p}"
        else:
            chunks.append(current.strip())
            current = next_p
    if current.strip():
        chunks.append(current.strip())
    return _add_overlap(chunks, overlap)


def chunk_segments(
    segments: list[ParsedSegment],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    file_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """На вход — сегменты от парсеров. На выход — плоский список словарей
    готовых для записи в БД и эмбеддинга.

    Если chunk_size/chunk_overlap не заданы:
    - используется adaptive_chunk_params(file_type, total_chars), если
      CHUNK_ADAPTIVE=true;
    - иначе берутся CHUNK_SIZE / CHUNK_OVERLAP из settings.

    Если CHUNK_SEMANTIC=true — для каждого сегмента используется
    semantic-сплит через эмбеддинги соседних абзацев (см. _semantic_split).
    """
    total_chars = sum(len(s.text or "") for s in segments)
    if chunk_size is None or chunk_overlap is None:
        cs_auto, ov_auto = adaptive_chunk_params(file_type, total_chars)
        cs = chunk_size if chunk_size is not None else cs_auto
        ov = chunk_overlap if chunk_overlap is not None else ov_auto
    else:
        cs, ov = chunk_size, chunk_overlap

    log.info(
        "Chunking: file_type=%s, total_chars=%d → chunk_size=%d, overlap=%d, semantic=%s",
        file_type, total_chars, cs, ov, settings.CHUNK_SEMANTIC,
    )

    splitter = _semantic_split if settings.CHUNK_SEMANTIC else _split_text
    out: list[dict[str, Any]] = []
    idx = 0
    for seg in segments:
        if not seg.text or not seg.text.strip():
            continue
        for piece in splitter(seg.text, cs, ov):
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
