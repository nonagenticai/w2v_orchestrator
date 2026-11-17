"""Annotation routes — get / patch / delete by id, scoped via stage_output → run → tenant."""

# IMPORTANT: do NOT add `from __future__ import annotations` to this file.
# The submodule name `annotations.py` collides with the future-import flag if
# the package eagerly evaluates `from __future__ import annotations`. The
# package __init__ avoids that, but this submodule should be defensive too —
# leaving the future flag off keeps everything consistent.

from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes._common import SessionDep, TenantIdDep
from src.api.schemas import AnnotationResponse, AnnotationUpdate
from src.database.models import Annotation, Deliverable, Project, Run, StageOutput
from src.database.repositories.annotation_repository import AnnotationRepository

router = APIRouter(prefix="/api/annotations", tags=["annotations"])

# The mutating handlers below end with an explicit `await session.commit()`:
# `AnnotationRepository` only FLUSHes and `get_db_session` commits in its ``yield``
# teardown, which FastAPI runs AFTER the response is sent — see CLAUDE.md
# § "A write handler MUST commit before it responds".


async def _scoped_get(
    session: AsyncSession, annotation_id: UUID, tenant_id: UUID
) -> Annotation | None:
    """Tenant-scope an annotation via stage_output → run → deliverable → project."""
    stmt = (
        select(Annotation)
        .join(StageOutput, Annotation.stage_output_id == StageOutput.id)
        .join(Run, StageOutput.run_id == Run.id)
        .join(Deliverable, Run.deliverable_id == Deliverable.id)
        .join(Project, Deliverable.project_id == Project.id)
        .where(Annotation.id == annotation_id, Project.tenant_id == tenant_id)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


def _to_resp(a: Annotation) -> AnnotationResponse:
    return AnnotationResponse.model_validate(a, from_attributes=True)


@router.get(
    "/{annotation_id}",
    response_model=AnnotationResponse,
    operation_id="annotations.get",
)
async def get_annotation(
    annotation_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> AnnotationResponse:
    a = await _scoped_get(session, annotation_id, tenant_id)
    if a is None:
        raise HTTPException(404, "annotation not found")
    return _to_resp(a)


@router.patch(
    "/{annotation_id}",
    response_model=AnnotationResponse,
    operation_id="annotations.update",
)
async def patch_annotation(
    annotation_id: UUID,
    body: AnnotationUpdate,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> AnnotationResponse:
    a = await _scoped_get(session, annotation_id, tenant_id)
    if a is None:
        raise HTTPException(404, "annotation not found")
    fields = body.model_dump(exclude_unset=True)
    if fields.get("status") == "applied":
        # 'applied' is set only by the regenerate transaction; routes can't force it.
        raise HTTPException(400, "status='applied' is set by the engine, not directly")
    updated = await AnnotationRepository(session).update(annotation_id, fields=fields)
    if updated is None:
        raise HTTPException(404, "annotation not found")
    await session.commit()
    return _to_resp(updated)


@router.delete(
    "/{annotation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="annotations.delete",
)
async def delete_annotation(
    annotation_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> None:
    a = await _scoped_get(session, annotation_id, tenant_id)
    if a is None:
        raise HTTPException(404, "annotation not found")
    await AnnotationRepository(session).delete(annotation_id)
    await session.commit()
