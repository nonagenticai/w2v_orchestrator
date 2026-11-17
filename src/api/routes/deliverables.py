"""Deliverables routes — get, patch, delete, start run."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status

from src.api.routes._common import DeliverableRepoDep, SessionDep, SubjectIdDep, TenantIdDep
from src.api.schemas import DeliverableResponse, DeliverableUpdate, RunStart, RunSummary
from src.database.models import Deliverable
from src.database.repositories.deliverable_repository import DeliverableRepository
from src.services.stages.website_review.order import OrderBelowFloorError

router = APIRouter(prefix="/api/deliverables", tags=["deliverables"])

# The mutating handlers below take `session: SessionDep` only to commit before they
# respond: the repository FLUSHes and `get_db_session` commits in its ``yield``
# teardown, which FastAPI runs AFTER the response is sent — see CLAUDE.md
# § "A write handler MUST commit before it responds".


async def _to_response(
    d: Deliverable, deliverable_repo: DeliverableRepository
) -> DeliverableResponse:
    # `deliverable.status` is derived on read from the deliverable's latest run.
    statuses = await deliverable_repo.derived_statuses([d.id])
    resp = DeliverableResponse.model_validate(d, from_attributes=True)
    resp.status = statuses[d.id]
    return resp


@router.get(
    "/{deliverable_id}",
    response_model=DeliverableResponse,
    operation_id="deliverables.get",
)
async def get_deliverable(
    deliverable_id: UUID,
    tenant_id: TenantIdDep,
    deliverable_repo: DeliverableRepoDep,
) -> DeliverableResponse:
    d = await deliverable_repo.get(deliverable_id, tenant_id=tenant_id)
    if d is None:
        raise HTTPException(404, detail="deliverable not found")
    return await _to_response(d, deliverable_repo)


@router.patch(
    "/{deliverable_id}",
    response_model=DeliverableResponse,
    operation_id="deliverables.update",
)
async def patch_deliverable(
    deliverable_id: UUID,
    body: DeliverableUpdate,
    tenant_id: TenantIdDep,
    deliverable_repo: DeliverableRepoDep,
    session: SessionDep,
) -> DeliverableResponse:
    fields = body.model_dump(exclude_unset=True)
    d = await deliverable_repo.update(deliverable_id, tenant_id=tenant_id, fields=fields)
    if d is None:
        raise HTTPException(404, detail="deliverable not found")
    await session.commit()
    return await _to_response(d, deliverable_repo)


@router.delete(
    "/{deliverable_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="deliverables.delete",
)
async def delete_deliverable(
    deliverable_id: UUID,
    tenant_id: TenantIdDep,
    deliverable_repo: DeliverableRepoDep,
    session: SessionDep,
) -> None:
    ok = await deliverable_repo.delete(deliverable_id, tenant_id=tenant_id)
    if not ok:
        raise HTTPException(404, detail="deliverable not found")
    await session.commit()


@router.post(
    "/{deliverable_id}/runs",
    response_model=RunSummary,
    status_code=status.HTTP_201_CREATED,
    operation_id="deliverables.runs.start",
)
async def start_run(
    deliverable_id: UUID,
    body: RunStart,
    request: Request,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
) -> RunSummary:
    from src.api.routes.runs import _project_id_for_deliverable
    from src.api.routes.runs import _summary as _run_summary

    engine = request.app.state.run_engine
    try:
        run = await engine.start_run(
            deliverable_id=deliverable_id,
            tenant_id=tenant_id,
            subject_id=subject_id,
            parent_run_id=body.parent_run_id,
            pause_after_stages=body.pause_after_stages,
            config_overrides=body.config_overrides,
        )
    except OrderBelowFloorError as exc:
        # Before the generic arm: the floor refusal gets the TYPED body (floor +
        # ordered value), not `str(exc)`. Must stay FIRST — it is a ValueError.
        raise HTTPException(400, detail=exc.refusal.model_dump()) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    project_id = await _project_id_for_deliverable(session, run.deliverable_id)
    return _run_summary(run, project_id)
