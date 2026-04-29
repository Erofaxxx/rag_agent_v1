import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

from auth.dependencies import csrf_check, require_user
from config import settings
from search import faiss_index, search_service
from storage import db, NotebookRow, UserRow

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/notebooks", tags=["notebooks"])

DEFAULT_NOTEBOOK_NAME = "Документы"


class NotebookOut(BaseModel):
    id: int
    name: str
    created_at: str
    updated_at: str
    document_count: int
    conversation_count: int


class CreateNotebookIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


class RenameNotebookIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


def _to_out(nb: NotebookRow) -> NotebookOut:
    return NotebookOut(
        id=nb.id,
        name=nb.name,
        created_at=nb.created_at,
        updated_at=nb.updated_at,
        document_count=db.count_documents(notebook_id=nb.id),
        conversation_count=len(db.list_conversations(notebook_id=nb.id)),
    )


def ensure_default_notebook(user: UserRow) -> NotebookRow:
    """Гарантирует, что у юзера есть хотя бы один ноутбук. Если ноутбуков нет —
    создаёт дефолтный и привязывает к нему все «бесхозные» документы и
    диалоги (для backward compat с пользователями, у которых уже что-то лежит)."""
    notebooks = db.list_notebooks(user.id)
    if notebooks:
        return notebooks[0]
    nb_id = db.create_notebook(user.id, DEFAULT_NOTEBOOK_NAME)
    nb = db.get_notebook(nb_id)
    assert nb is not None
    docs, convs = db.assign_orphans_to_notebook(user.id, nb_id)
    if docs or convs:
        log.info(
            "Привязал %d документов и %d диалогов user_id=%s к дефолтному ноутбуку %s",
            docs, convs, user.id, nb_id,
        )
    return nb


def _can_access(nb: NotebookRow, user: UserRow) -> bool:
    if user.role == "admin":
        return True
    return nb.user_id == user.id


# ===== endpoints =====

@router.get("", response_model=list[NotebookOut])
def list_notebooks(user: UserRow = Depends(require_user)) -> list[NotebookOut]:
    # Гарантируем дефолт при первом обращении
    ensure_default_notebook(user)
    notebooks = db.list_notebooks(user.id)
    return [_to_out(nb) for nb in notebooks]


@router.post(
    "",
    response_model=NotebookOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(csrf_check)],
)
def create_notebook(
    payload: CreateNotebookIn,
    user: UserRow = Depends(require_user),
) -> NotebookOut:
    nb_id = db.create_notebook(user.id, payload.name)
    nb = db.get_notebook(nb_id)
    assert nb is not None
    log.info("Создан ноутбук '%s' (id=%s) для user_id=%s", payload.name, nb_id, user.id)
    return _to_out(nb)


@router.get("/{notebook_id}", response_model=NotebookOut)
def get_notebook(
    notebook_id: int,
    user: UserRow = Depends(require_user),
) -> NotebookOut:
    nb = db.get_notebook(notebook_id)
    if nb is None or not _can_access(nb, user):
        raise HTTPException(404, "Ноутбук не найден")
    return _to_out(nb)


@router.patch(
    "/{notebook_id}",
    response_model=NotebookOut,
    dependencies=[Depends(csrf_check)],
)
def rename_notebook(
    notebook_id: int,
    payload: RenameNotebookIn,
    user: UserRow = Depends(require_user),
) -> NotebookOut:
    nb = db.get_notebook(notebook_id)
    if nb is None or not _can_access(nb, user):
        raise HTTPException(404, "Ноутбук не найден")
    db.rename_notebook(notebook_id, payload.name)
    fresh = db.get_notebook(notebook_id)
    assert fresh is not None
    return _to_out(fresh)


@router.delete(
    "/{notebook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(csrf_check)],
)
def delete_notebook(
    notebook_id: int,
    user: UserRow = Depends(require_user),
) -> Response:
    nb = db.get_notebook(notebook_id)
    if nb is None or not _can_access(nb, user):
        raise HTTPException(404, "Ноутбук не найден")
    # Защита от удаления последнего ноутбука (юзер должен иметь хотя бы один)
    if len(db.list_notebooks(nb.user_id)) <= 1:
        raise HTTPException(400, "Нельзя удалить единственный ноутбук. Создайте новый и перенесите документы.")

    # Перед удалением соберём пути к файлам, чтобы потом потереть с диска
    docs = db.list_documents(notebook_id=notebook_id)
    file_paths = [Path(d.file_path) for d in docs if d.file_path]

    chunk_ids = db.delete_notebook(notebook_id)
    if chunk_ids:
        faiss_index.remove(chunk_ids)
        faiss_index.persist()
        search_service.invalidate_bm25()
    # Чистим оригиналы
    for fp in file_paths:
        try:
            target_dir = fp.parent
            if target_dir.exists() and str(target_dir).startswith(str(settings.uploads_path)):
                shutil.rmtree(target_dir, ignore_errors=True)
        except Exception as e:
            log.warning("Не удалось удалить файл %s при удалении ноутбука: %s", fp, e)

    log.info(
        "Удалён ноутбук '%s' (id=%s, user_id=%s): %d чанков, %d файлов",
        nb.name, nb.id, nb.user_id, len(chunk_ids), len(file_paths),
    )
    return Response(status_code=204)
