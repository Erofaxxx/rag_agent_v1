import json
import logging
import re
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from auth.dependencies import csrf_check, require_admin, require_user
from auth.router import limiter
from storage import UserRow
from chunking import chunk_segments
from config import settings
from embeddings import embedding_service
from parsers import detect_file_type, parse_file
from search import faiss_index, search_service
from storage import db

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


# Глобальный lock на ingest pipeline. FastAPI BackgroundTasks могут запускать
# несколько _process_document одновременно (если несколько uploads почти
# одновременно), а L0 (corpus-consistency) делает FAISS-search по уже
# проиндексированным chunks. Без сериализации L0 на «параллельных близнецах»
# (poisoned копия + clean оригинал, загруженные одним вызовом) видит пустой
# индекс и пропускает атаку. Lock сериализует L0+L1+L2+faiss.add в один
# критический раздел; при загрузке N документов время = N × O(1 ingest) —
# хвост одного user'а, не глобальный bottleneck (индексирование и так упирается
# в embedding-RPM провайдера).
_ingest_lock = threading.Lock()


_FILENAME_BAD_CHARS = re.compile(r"[\x00-\x1f/\\<>:\"|?*]")


def sanitize_filename(name: str) -> str:
    """Убирает path-traversal и опасные символы. Возвращает 'unnamed' если пусто."""
    if not name:
        return "unnamed"
    # Только базовое имя — без директорий
    name = Path(name).name
    name = _FILENAME_BAD_CHARS.sub("_", name)
    # Не оставляем точки в начале (.htaccess, .env и т.п.)
    name = name.lstrip(".")
    # Ограничим длину
    if len(name) > 200:
        stem, dot, ext = name.rpartition(".")
        if dot and len(ext) <= 8:
            name = stem[: 200 - len(ext) - 1] + "." + ext
        else:
            name = name[:200]
    return name or "unnamed"


class DocumentOut(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    upload_date: str
    status: str
    error_message: Optional[str] = None
    chunk_count: int


class UploadResponse(BaseModel):
    documents: list[DocumentOut]


def _to_out(d) -> DocumentOut:
    return DocumentOut(
        id=d.id,
        filename=d.filename,
        file_type=d.file_type,
        file_size=d.file_size,
        upload_date=d.upload_date,
        status=d.status,
        error_message=d.error_message,
        chunk_count=d.chunk_count,
    )


def _process_document(document_id: int) -> None:
    """Фоновый воркер: парсит файл, режет на чанки, эмбеддит, кладёт в FAISS.

    Сериализован глобальным `_ingest_lock` — это нужно, чтобы L0
    corpus-consistency корректно работал на параллельной загрузке нескольких
    документов сразу (см. комментарий к _ingest_lock). Без этого
    одновременно стартующие background-обработки видят пустой индекс друг
    относительно друга, и L0 не находит «двойников»."""
    with _ingest_lock:
        _process_document_locked(document_id)


def _process_document_locked(document_id: int) -> None:
    doc = db.get_document(document_id)
    if not doc:
        log.error("Документ %s не найден для обработки", document_id)
        return
    db.update_document_status(document_id, "processing")
    started = time.time()
    try:
        log.info("Начинаю обработку %s (%s)", doc.filename, doc.file_type)
        segments = parse_file(doc.file_path, doc.file_type)
        if not segments:
            raise ValueError("Не удалось извлечь текст из документа")

        chunks = chunk_segments(segments, file_type=doc.file_type)
        if not chunks:
            raise ValueError("После чанкинга не осталось ни одного фрагмента")

        # ---- Defense L1: ingest-time sanitization (security research) ----
        # По умолчанию выключено (settings.DEFENSE_L1_SANITIZE='off'). Если
        # включено — пропускаем чанки через regex-детектор инъекций. В режиме
        # 'warn' просто логируем, в 'drop' — отбрасываем чанки с risk выше
        # порога. Никогда не подменяет содержимое чанка молча.
        l1_report = None
        if settings.DEFENSE_L1_SANITIZE != "off":
            from defenses.l1_sanitize import sanitize_chunks, short_summary as l1_summary
            chunks, l1_report = sanitize_chunks(
                chunks,
                mode=settings.DEFENSE_L1_SANITIZE,
                threshold=settings.DEFENSE_L1_RISK_THRESHOLD,
            )
            log.info("[L1] Документ %s: %s", document_id, l1_summary(l1_report))
            if not chunks:
                raise ValueError(
                    "L1 sanitization отбросил все чанки документа "
                    "(возможно, документ полностью состоит из prompt-injection)"
                )

        chunk_ids = db.insert_chunks(document_id, chunks)
        log.info("Документ %s: %d чанков, эмбеддю...", document_id, len(chunks))

        texts = [c["text"] for c in chunks]
        vectors = embedding_service.encode_passages(texts)

        # ---- Defense L0: corpus consistency / near-duplicate detection ----
        # Перед добавлением в индекс смотрим, не клон ли новый документ уже
        # существующего (≥ ratio chunks с cosine ≥ threshold к chunks одного
        # документа из индекса) с inserted разделами. Это типичная сигнатура
        # стелс-бэкдора через клонирование легитимного файла.
        l0_report = None
        if settings.DEFENSE_L0_CORPUS_CONSISTENCY != "off":
            from defenses.l0_corpus_consistency import (
                build_error_message as l0_error_msg,
                detect_near_duplicate_document,
                short_summary as l0_summary,
            )

            def _l0_search(vec, k):
                return faiss_index.search(vec, k)

            def _l0_resolver(chunk_ids_list):
                # Разрешаем chunk_id → (document_id, filename) одним батчем.
                # chunks этого нового документа УЖЕ в БД, но ЕЩЁ не в FAISS
                # (мы добавим их через faiss_index.add ниже), поэтому
                # FAISS-search их и не вернёт — self-shadow невозможен.
                rows = db.get_chunks_by_ids(list(chunk_ids_list))
                doc_id_set = {r.document_id for r in rows}
                docs = {d.id: d for d in (db.get_document(did) for did in doc_id_set) if d}
                return {
                    r.id: (r.document_id, docs[r.document_id].filename)
                    for r in rows if r.document_id in docs
                }

            l0_report = detect_near_duplicate_document(
                vectors,
                search_fn=_l0_search,
                chunk_to_doc_resolver=_l0_resolver,
                similarity_threshold=settings.DEFENSE_L0_SIMILARITY_THRESHOLD,
                duplicate_ratio_threshold=settings.DEFENSE_L0_DUPLICATE_RATIO_THRESHOLD,
            )
            log.info("[L0] Документ %s: %s", document_id, l0_summary(l0_report))
            if l0_report.is_near_duplicate and settings.DEFENSE_L0_CORPUS_CONSISTENCY == "drop":
                # Откатываем то, что уже сделали: chunks вставлены в БД,
                # но в FAISS их ещё нет. Очищаем chunks, помечаем документ
                # как error.
                db.delete_chunks_for_document(document_id)
                err = l0_error_msg(l0_report)
                db.update_document_status(document_id, "error", error_message=err)
                log.warning("[L0] Документ %s ЗАБЛОКИРОВАН: %s", document_id, err)
                return  # документ не попадёт в индекс — атака предотвращена

        # ---- Defense L2: per-document embedding anomaly detection ----
        # Считаем z-score cosine-расстояния каждого чанка до центроида
        # документа. Чанки-outliers логируем (а в режиме 'drop' выкидываем).
        l2_report = None
        if settings.DEFENSE_L2_ANOMALY != "off":
            from defenses.l2_embedding_anomaly import detect_anomalies, short_summary as l2_summary
            l2_report = detect_anomalies(
                vectors,
                z_threshold=settings.DEFENSE_L2_ZSCORE_THRESHOLD,
            )
            log.info("[L2] Документ %s: %s", document_id, l2_summary(l2_report))
            if settings.DEFENSE_L2_ANOMALY == "drop" and l2_report.n_flagged > 0:
                # Фильтруем chunks, chunk_ids и vectors параллельно по флагам.
                kept_idx = [i for i, f in enumerate(l2_report.flags) if not f]
                if kept_idx:
                    import numpy as _np
                    chunk_ids = [chunk_ids[i] for i in kept_idx]
                    vectors = _np.asarray([vectors[i] for i in kept_idx])
                    log.info("[L2] %d чанков отброшено", len(l2_report.flags) - len(kept_idx))
                else:
                    raise ValueError("L2 пометил все чанки как аномалии — документ выглядит подозрительно целиком")

        # ---- Defense L6: ingest-time contradiction check (LLM-judge) ----
        # Закрывает атаки, не пойманные L0/L1/L2: «новая редакция»,
        # одиночные триггер-утверждения, content-conflicts. Делает
        # LLM-judge для каждого chunk c соседями в индексе (cosine ≥
        # similarity_threshold). +1 LLM-вызов на chunk; обычно их 1-3.
        if settings.DEFENSE_L6_INGEST_CONTRADICTION != "off":
            try:
                from defenses.l6_ingest_contradiction import (
                    build_error_message as l6_error_msg,
                    detect_ingest_contradiction,
                    short_summary as l6_summary,
                )
                from llm.verifier import _get_llm as _l6_get_llm

                def _l6_search(vec, k):
                    return faiss_index.search(vec, k)

                def _l6_resolver(cids):
                    rows = db.get_chunks_by_ids(list(cids))
                    doc_id_set = {r.document_id for r in rows}
                    docs = {d.id: d for d in (db.get_document(did) for did in doc_id_set) if d}
                    return {
                        r.id: (r.text, docs[r.document_id].filename, r.document_id)
                        for r in rows if r.document_id in docs
                    }

                l6_llm = _l6_get_llm()
                if l6_llm is not None:
                    l6_report = detect_ingest_contradiction(
                        new_chunk_texts=[c["text"] for c in chunks],
                        new_chunk_vectors=vectors,
                        search_fn=_l6_search,
                        chunk_resolver=_l6_resolver,
                        llm=l6_llm,
                        similarity_threshold=settings.DEFENSE_L6_SIMILARITY_THRESHOLD,
                        top_k_neighbors=settings.DEFENSE_L6_TOP_K_NEIGHBORS,
                        max_chunks_to_check=settings.DEFENSE_L6_MAX_CHUNKS_TO_CHECK,
                    )
                    log.info("[L6] Документ %s: %s", document_id, l6_summary(l6_report))
                    if l6_report.has_contradiction and settings.DEFENSE_L6_INGEST_CONTRADICTION == "drop":
                        db.delete_chunks_for_document(document_id)
                        err = l6_error_msg(l6_report)
                        db.update_document_status(document_id, "error", error_message=err)
                        log.warning("[L6] Документ %s ЗАБЛОКИРОВАН: %s", document_id, err)
                        return
            except Exception as e:
                log.warning("[L6] упал, пропускаем: %s", e)

        faiss_index.add(vectors, chunk_ids)
        faiss_index.persist()
        search_service.invalidate_bm25()

        db.update_document_status(document_id, "ready", chunk_count=len(chunks))
        log.info(
            "Документ %s готов за %.1fs (%d чанков)",
            document_id,
            time.time() - started,
            len(chunks),
        )

        # Для PDF — рядом с оригиналом сохраняем spans.json для подсветки
        # цитат в боковом viewer'е. Сами цитаты живут в SQLite (chunks.text),
        # этот файл — только для frontend overlay по bbox координатам.
        if doc.file_type == "pdf":
            try:
                from parsers.pdf_parser import extract_pdf_spans
                spans_data = extract_pdf_spans(doc.file_path)
                spans_path = Path(doc.file_path).parent / "spans.json"
                with open(spans_path, "w", encoding="utf-8") as f:
                    json.dump(spans_data, f, ensure_ascii=False)
                log.debug("PDF spans для %s: %d страниц", document_id, len(spans_data))
            except Exception as e:
                log.warning("Не удалось извлечь PDF spans для %s: %s", document_id, e)
        # Оригинал по умолчанию остаётся — нужен для перепарсинга при будущих
        # апгрейдах парсера, для скачивания пользователем, для compliance.
        # Удалить можно через KEEP_ORIGINAL_FILES=false в .env (если очень мало
        # диска или жёсткие privacy-требования).
        if not settings.KEEP_ORIGINAL_FILES:
            try:
                target_dir = Path(doc.file_path).parent.resolve()
                uploads_root = settings.uploads_path.resolve()
                # is_relative_to + неравенство корню: исключаем прямой rmtree(uploads/).
                if (
                    target_dir.exists()
                    and target_dir != uploads_root
                    and target_dir.is_relative_to(uploads_root)
                ):
                    shutil.rmtree(target_dir, ignore_errors=True)
                    log.debug("Удалён оригинал %s (KEEP_ORIGINAL_FILES=false)", doc.file_path)
            except Exception as e:
                log.warning("Не удалось удалить оригинал %s: %s", doc.file_path, e)
    except Exception as e:
        log.exception("Ошибка обработки документа %s: %s", document_id, e)
        db.update_document_status(document_id, "error", error_message=str(e))


def _can_access(doc, user: UserRow) -> bool:
    """Админ — все. Юзер — только свои (по uploaded_by)."""
    if user.role == "admin":
        return True
    return doc.uploaded_by == user.id


@router.get("", response_model=list[DocumentOut])
def list_documents(
    notebook_id: Optional[int] = None,
    user: UserRow = Depends(require_user),
) -> list[DocumentOut]:
    owner = None if user.role == "admin" else user.id
    # Если фильтр по notebook задан, проверим, что юзер имеет к нему доступ
    if notebook_id is not None and user.role != "admin":
        nb = db.get_notebook(notebook_id)
        if nb is None or nb.user_id != user.id:
            raise HTTPException(404, "Ноутбук не найден")
    return [_to_out(d) for d in db.list_documents(owner_user_id=owner, notebook_id=notebook_id)]


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, user: UserRow = Depends(require_user)) -> DocumentOut:
    doc = db.get_document(document_id)
    if not doc or not _can_access(doc, user):
        raise HTTPException(404, "Документ не найден")
    return _to_out(doc)


@router.get("/{document_id}/status", response_model=DocumentOut)
def get_document_status(document_id: int, user: UserRow = Depends(require_user)) -> DocumentOut:
    doc = db.get_document(document_id)
    if not doc or not _can_access(doc, user):
        raise HTTPException(404, "Документ не найден")
    return _to_out(doc)


@router.get("/{document_id}/page/{page}/spans")
def get_page_spans(
    document_id: int,
    page: int,
    user: UserRow = Depends(require_user),
) -> dict[str, Any]:
    """Spans (текст + bbox в PDF user-space) для указанной страницы PDF.
    Используется фронтом для подсветки фрагментов цитаты в pdf.js viewer'е."""
    doc = db.get_document(document_id)
    if not doc or not _can_access(doc, user):
        raise HTTPException(404, "Документ не найден")
    if doc.file_type != "pdf":
        raise HTTPException(400, "Подсветка по bbox доступна только для PDF")
    if not doc.file_path:
        raise HTTPException(404, "Оригинал не сохранён")
    spans_path = Path(doc.file_path).parent / "spans.json"
    if not spans_path.exists():
        raise HTTPException(404, "Spans не извлечены — попробуйте перезагрузить документ")
    try:
        with open(spans_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(500, f"Не удалось прочитать spans: {e}")
    page_data = data.get(str(page)) or data.get(page)
    if not page_data:
        raise HTTPException(404, f"Нет данных для страницы {page}")
    return page_data


@router.get("/{document_id}/file")
def get_document_file(document_id: int, user: UserRow = Depends(require_user)):
    """Отдаёт оригинальный файл для просмотра (PDF в iframe для source highlights).
    Доступ — только владельцу документа или админу. Файл должен существовать
    на диске (KEEP_ORIGINAL_FILES=true)."""
    doc = db.get_document(document_id)
    if not doc or not _can_access(doc, user):
        raise HTTPException(404, "Документ не найден")
    if not doc.file_path:
        raise HTTPException(404, "Оригинал не сохранён")
    p = Path(doc.file_path)
    if not p.exists():
        raise HTTPException(404, "Файл не найден на диске")
    # Безопасность: не отдадим файл за пределами uploads_path. Используем
    # is_relative_to, а не str.startswith — иначе путь типа /data/uploads_evil/...
    # ложно проходил проверку для uploads_path=/data/uploads.
    resolved = p.resolve()
    uploads_root = settings.uploads_path.resolve()
    if not resolved.is_relative_to(uploads_root):
        raise HTTPException(403, "Доступ запрещён")

    media_types = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "doc": "application/msword",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls": "application/vnd.ms-excel",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "txt": "text/plain; charset=utf-8",
        "md": "text/markdown; charset=utf-8",
        "csv": "text/csv; charset=utf-8",
    }
    # Content-Disposition НЕ ставим вручную: starlette сам формирует его из
    # filename= с RFC 5987-кодированием (filename*=UTF-8''...) — нужно для
    # имён с кириллицей, иначе latin-1 кодирование заголовков падает в
    # UnicodeEncodeError → 500. content_disposition_type="inline" даёт
    # iframe-показ PDF вместо принудительного скачивания.
    return FileResponse(
        path=str(p),
        media_type=media_types.get(doc.file_type, "application/octet-stream"),
        filename=doc.filename,
        content_disposition_type="inline",
        headers={
            "X-Frame-Options": "SAMEORIGIN",  # переопределяем глобальный DENY
        },
    )


@router.get("/{document_id}/html")
def get_document_html(document_id: int, user: UserRow = Depends(require_user)) -> dict[str, Any]:
    """Конвертирует не-PDF документы в HTML/markdown/text для рендеринга в
    боковой панели. PDF имеет свой канвас-вьюер с подсветкой bbox, для других
    форматов раньше был iframe с FileResponse, но Word/Excel/Markdown в iframe
    показываются как «скачать или сырой текст».

    DOCX/DOC → mammoth (заголовки, списки, таблицы сохраняются).
    MD/MARKDOWN → возвращаем raw markdown (фронт уже умеет marked.js).
    TXT/CSV → plain text, фронт обернёт в <pre> с HTML-экранированием.
    Остальное — 415: фронт показывает fallback с кнопкой «скачать».
    """
    doc = db.get_document(document_id)
    if not doc or not _can_access(doc, user):
        raise HTTPException(404, "Документ не найден")
    if not doc.file_path:
        raise HTTPException(404, "Оригинал не сохранён")
    p = Path(doc.file_path)
    if not p.exists():
        raise HTTPException(404, "Файл не найден на диске")
    if not p.resolve().is_relative_to(settings.uploads_path.resolve()):
        raise HTTPException(403, "Доступ запрещён")

    ft = (doc.file_type or "").lower()
    try:
        if ft in ("docx", "doc"):
            import mammoth
            with open(p, "rb") as f:
                result = mammoth.convert_to_html(f)
            return {"format": "html", "content": result.value}
        if ft in ("md", "markdown"):
            text = p.read_text(encoding="utf-8", errors="replace")
            return {"format": "markdown", "content": text}
        if ft in ("txt", "csv"):
            text = p.read_text(encoding="utf-8", errors="replace")
            return {"format": "text", "content": text}
    except Exception as e:
        log.exception("Ошибка конвертации документа %s в HTML: %s", document_id, e)
        raise HTTPException(500, f"Не удалось подготовить просмотр: {e}")

    raise HTTPException(415, f"Просмотр для типа {ft!r} не поддерживается")


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(csrf_check)],
)
@limiter.limit(f"{settings.RATE_LIMIT_UPLOAD_PER_MINUTE}/minute")
async def upload_documents(
    request: Request,
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    notebook_id: Optional[int] = Form(None),
    actor: UserRow = Depends(require_user),
) -> UploadResponse:
    # Если notebook_id не передан — кладём в дефолтный ноутбук пользователя
    # (создаётся автоматически при первом обращении к /api/notebooks).
    from api.notebooks import ensure_default_notebook
    if notebook_id is None:
        nb = ensure_default_notebook(actor)
        notebook_id = nb.id
    else:
        nb = db.get_notebook(notebook_id)
        if nb is None or (actor.role != "admin" and nb.user_id != actor.id):
            raise HTTPException(404, "Ноутбук не найден")

    # Лимит — per-user (не глобальный), чтобы один юзер не блокировал других
    own_count = db.count_documents(owner_user_id=actor.id) if actor.role != "admin" else db.count_documents()
    if own_count + len(files) > settings.MAX_DOCUMENTS:
        raise HTTPException(
            400,
            f"Превышен лимит документов ({settings.MAX_DOCUMENTS})",
        )

    out: list[DocumentOut] = []
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    for upload in files:
        original_name = sanitize_filename(upload.filename or "unnamed")
        # Сначала сохраняем файл во временное место, чтобы определить тип
        tmp_dir = settings.uploads_path / "_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # Уникальное имя: ms-таймстамп даёт коллизии при HTTP/2 multiplexing
        # или batch upload, поэтому добавляем 8 байт энтропии — collision-free
        # на любых разумных нагрузках.
        tmp_path = tmp_dir / f"{int(time.time() * 1000)}_{secrets.token_hex(8)}_{original_name}"

        try:
            written = 0
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        f.close()
                        tmp_path.unlink(missing_ok=True)
                        raise HTTPException(
                            413,
                            f"Файл {original_name} больше лимита {settings.MAX_FILE_SIZE_MB} MB",
                        )
                    f.write(chunk)

            file_type = detect_file_type(tmp_path)
            if not file_type:
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(
                    400,
                    f"Неподдерживаемый формат: {original_name}. Принимаются PDF/DOCX/XLSX/PPTX",
                )

            file_size = tmp_path.stat().st_size
            doc_id = db.create_document(
                filename=original_name,
                file_path="",
                file_type=file_type,
                file_size=file_size,
                uploaded_by=actor.id,
                notebook_id=notebook_id,
            )

            try:
                target_dir = settings.uploads_path / str(doc_id)
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / original_name
                shutil.move(str(tmp_path), str(target_path))
                db.update_document_status(doc_id, "pending")
                with db.cursor() as cur:
                    cur.execute(
                        "UPDATE documents SET file_path=? WHERE id=?",
                        (str(target_path), doc_id),
                    )
            except Exception:
                # Файл не доехал до uploads/{doc_id}/ — нельзя оставить запись с
                # пустым file_path, иначе документ висит в list_documents и
                # учитывается в MAX_DOCUMENTS до следующего рестарта (где watchdog
                # пометит error). Каскад удалит и потенциально созданные чанки.
                try:
                    db.delete_document(doc_id)
                except Exception:
                    pass
                raise

            background.add_task(_process_document, doc_id)
            doc = db.get_document(doc_id)
            assert doc is not None
            out.append(_to_out(doc))
        except HTTPException:
            raise
        except Exception as e:
            log.exception("Ошибка загрузки %s: %s", original_name, e)
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(500, f"Ошибка загрузки {original_name}: {e}")

    return UploadResponse(documents=out)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(csrf_check)],
)
def delete_document(document_id: int, user: UserRow = Depends(require_user)) -> None:
    """Удаление документа — может удалить только владелец (uploaded_by) или админ.

    Что чистится синхронно:
    - SQLite: документ + все его чанки (CASCADE)
    - FAISS: векторы по chunk_ids (remove_ids)
    - Файловая система: data/uploads/{document_id}/

    В Yandex AI Studio мы НИЧЕГО не храним — только дёргаем embedding API.
    Удалять там нечего.
    """
    doc = db.get_document(document_id)
    if not doc or not _can_access(doc, user):
        raise HTTPException(404, "Документ не найден")
    chunk_ids = db.delete_document(document_id)
    if chunk_ids:
        faiss_index.remove(chunk_ids)
        faiss_index.persist()
        search_service.invalidate_bm25()
        log.info("Удалён документ %s (%s), чанков: %d", document_id, doc.filename, len(chunk_ids))
    # Audit: для security-research-сервиса удаление документа — событие интереса
    # (вектор «удалить отравленный документ, чтобы скрыть следы»). Без записи
    # forensics-цепочка обрывается.
    db.log_audit(
        event="delete_document",
        user_id=doc.uploaded_by,
        actor_user_id=user.id,
        details=f"document_id={document_id} filename={doc.filename} chunks={len(chunk_ids)}",
    )
    try:
        target_dir = Path(doc.file_path).parent.resolve()
        uploads_root = settings.uploads_path.resolve()
        if (
            target_dir.exists()
            and target_dir != uploads_root
            and target_dir.is_relative_to(uploads_root)
        ):
            shutil.rmtree(target_dir, ignore_errors=True)
    except Exception as e:
        log.warning("Не удалось удалить файлы документа %s: %s", document_id, e)
