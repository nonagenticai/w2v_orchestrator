"""Verbatim-item routes — project-scoped CRUD."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes._common import SessionDep, TenantIdDep
from src.api.routes.runs import _project_id_for_deliverable, _tenant_owns_run
from src.api.schemas import (
    VerbatimItemCreate,
    VerbatimItemListResponse,
    VerbatimItemResponse,
    VerbatimItemUpdate,
)
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.database.repositories.verbatim_item_repository import VerbatimItemRepository
from src.services.verbatim_placement import (
    VERBATIM_CARRIER_PATHS,
    Placement,
    derive_placements,
)

router = APIRouter(tags=["verbatim"])

# The mutating handlers below end with an explicit `await session.commit()`:
# `VerbatimItemRepository` only FLUSHes and `get_db_session` commits in its
# ``yield`` teardown, which FastAPI runs AFTER the response is sent — see
# CLAUDE.md § "A write handler MUST commit before it responds".


def _resp(item, *, project=None, placement: Placement | None = None) -> VerbatimItemResponse:  # noqa: ANN001
    # Identity (tenant_id/subject_id/visibility) is inherited from the parent
    # Project; lifecycle fields are DERIVED (see the VerbatimItemResponse
    # docstring) — `placement` is None everywhere except a `?run_id=` listing,
    # and the schema's `pending` default then stands.
    payload = {
        "id": item.id,
        "project_id": item.project_id,
        "kind": item.kind,
        "text": item.text,
        "hint": item.hint,
        "scope": item.scope,
        "scope_target": item.scope_target,
        "position": item.position,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }
    if project is not None:
        payload["tenant_id"] = project.tenant_id
        payload["subject_id"] = project.owner_subject_id
        payload["visibility"] = project.visibility
    if placement is not None:
        payload["status"] = placement.status
        payload["placed_in_stage"] = placement.stage
        payload["placed_in_element_id"] = placement.element_id
    return VerbatimItemResponse.model_validate(payload)


async def _carrier_outputs(session: AsyncSession, run_id: UUID) -> dict[str, dict]:
    """Load ONLY the verbatim-carrier stage outputs for a run.

    One indexed ``(run_id, stage_name)`` lookup per carrier via
    ``StageOutputRepository.get`` — deliberately NOT ``list_for_run``, which
    hydrates every stage's JSONB blob for the run. Those blobs are multi-MB
    (shotlists, critique reports, per-clip manifests); pulling them all to
    substring-match a handful of script lines is the exact pathology that forced
    ``RunStatusLite`` into existence. The carrier set is small and fixed, and
    this only runs when the caller opted in with ``?run_id=``.
    """
    repo = StageOutputRepository(session)
    outputs: dict[str, dict] = {}
    for stage in VERBATIM_CARRIER_PATHS:
        row = await repo.get(run_id, stage)
        if row is not None and row.output:
            outputs[stage] = row.output
    return outputs


@router.get(
    "/api/projects/{project_id}/verbatim-items",
    response_model=VerbatimItemListResponse,
    operation_id="verbatim.list",
)
async def list_items(
    project_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
    run_id: Annotated[
        UUID | None,
        Query(
            description=(
                "Derive each item's placement against this run's persisted stage "
                "outputs. The run must belong to the caller's tenant AND to this "
                "project (404 otherwise). Omitted: every item reports `pending`, "
                "which is the honest answer — placement is a property of a run."
            )
        ),
    ] = None,
) -> VerbatimItemListResponse:
    project = await ProjectRepository(session).get(project_id, tenant_id=tenant_id)
    if project is None:
        raise HTTPException(404, "project not found")
    items = await VerbatimItemRepository(session).list_for_project(project_id, tenant_id=tenant_id)

    placements: dict[UUID, Placement] = {}
    if run_id is not None:
        # Same tenant gate every run-scoped route uses (run -> deliverable ->
        # project.tenant_id), then a project-match check: a run this tenant owns
        # but under a DIFFERENT project must not be usable to derive placement
        # for these phrases.
        run = await _tenant_owns_run(session, run_id, tenant_id)
        if run is None:
            raise HTTPException(404, "run not found")
        if await _project_id_for_deliverable(session, run.deliverable_id) != project_id:
            raise HTTPException(404, "run not found")
        placements = derive_placements(items, await _carrier_outputs(session, run_id))

    responses = [_resp(i, project=project, placement=placements.get(i.id)) for i in items]
    return VerbatimItemListResponse(items=responses, total=len(responses))


@router.post(
    "/api/projects/{project_id}/verbatim-items",
    response_model=VerbatimItemResponse,
    status_code=status.HTTP_201_CREATED,
    operation_id="verbatim.create",
)
async def create_item(
    project_id: UUID,
    body: VerbatimItemCreate,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> VerbatimItemResponse:
    project = await ProjectRepository(session).get(project_id, tenant_id=tenant_id)
    if project is None:
        raise HTTPException(404, "project not found")
    item = await VerbatimItemRepository(session).create(
        project_id=project_id,
        text=body.text,
        position=body.position,
        kind=body.kind,
        hint=body.hint,
        scope=body.scope,
        scope_target=body.scope_target,
    )
    await session.commit()
    return _resp(item, project=project)


@router.patch(
    "/api/verbatim-items/{item_id}",
    response_model=VerbatimItemResponse,
    operation_id="verbatim.update",
)
async def patch_item(
    item_id: UUID,
    body: VerbatimItemUpdate,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> VerbatimItemResponse:
    fields = body.model_dump(exclude_unset=True)
    item = await VerbatimItemRepository(session).update(item_id, tenant_id=tenant_id, fields=fields)
    if item is None:
        raise HTTPException(404, "verbatim item not found")
    await session.commit()
    return _resp(item)


@router.delete(
    "/api/verbatim-items/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="verbatim.delete",
)
async def delete_item(
    item_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> None:
    ok = await VerbatimItemRepository(session).delete(item_id, tenant_id=tenant_id)
    if not ok:
        raise HTTPException(404, "verbatim item not found")
    await session.commit()
