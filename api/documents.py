import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from auth.dependencies import csrf_check, require_admin, require_user
from storage import UserRow
from chunking import chunk_segments
from config import settings
from embeddings import embedding_service
from parsers import detect_file_type, parse_file
from search import faiss_index, search_service
from storage import db

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


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
    """Фоновый воркер: парсит файл, режет на чанки, эмбеддит, кладёт в FAISS."""
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

        chunks = chunk_segments(segments)
        if not chunks:
            raise ValueError("После чанкинга не осталось ни одного фрагмента")

        chunk_ids = db.insert_chunks(document_id, chunks)
        log.info("Документ %s: %d чанков, эмбеддю...", document_id, len(chunks))

        texts = [c["text"] for c in chunks]
        vectors = embedding_service.encode_passages(texts)
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
                target_dir = Path(doc.file_path).parent
                if target_dir.exists() and str(target_dir).startswith(str(settings.uploads_path)):
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
    # Безопасность: не отдадим файл за пределами uploads_path
    if not str(p.resolve()).startswith(str(settings.uploads_path.resolve())):
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
    return FileResponse(
        path=str(p),
        media_type=media_types.get(doc.file_type, "application/octet-stream"),
        filename=doc.filename,
        headers={
            # Inline для PDF чтобы открывалось в iframe, а не предлагало скачать
            "Content-Disposition": f'inline; filename="{doc.filename}"',
            "X-Frame-Options": "SAMEORIGIN",  # переопределяем глобальный DENY
        },
    )


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(csrf_check)],
)
async def upload_documents(
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
        tmp_path = tmp_dir / f"{int(time.time() * 1000)}_{original_name}"

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
    try:
        target_dir = Path(doc.file_path).parent
        if target_dir.exists() and str(target_dir).startswith(str(settings.uploads_path)):
            shutil.rmtree(target_dir, ignore_errors=True)
    except Exception as e:
        log.warning("Не удалось удалить файлы документа %s: %s", document_id, e)
