"""Tenant offboarding (P6 ORCH-6): export + delete. Engine: ``services.tenant_offboarding``.

Owner rules: a tenant_admin may only REQUEST; a super_admin executes.

Tenant side (``/api/tenant/offboarding``, ``TenantAdminDep``):

- ``POST`` records a request (``status='requested'``) + an ``offboarding.request`` audit
  row, nothing else. One open request per kind (409 otherwise).
- ``GET`` lists the tenant's own jobs; a finished export carries a ``download_url``
  (``GET /{id}/download``, below). The deciding operator is not named.
- ``GET /{id}/download`` streams the tenant's own finished export zip from MinIO with
  the orchestrator's credentials (``api.minio_stream``). Not a presigned URL: the
  presign host is ``minio.test``, unreachable from a browser.

Both are reachable WITHOUT a live licence (no ``LICENSE_GATE_ONLY``): asking for your
data back, or for its erasure, must keep working after a licence lapses. They are
listed with that reason in ``tests/unit/test_tenant_gate_coverage.py``.

Operator side (``/api/admin``, ``SuperAdminDep``; ungated by licence like
``PATCH /api/admin/tenants/{id}`` — the operator acts on any tenant):

- ``GET /offboarding`` (queue, filterable), ``GET /offboarding/{id}`` and
  ``GET /offboarding/{id}/download`` (any tenant's finished export).
- ``POST /offboarding/{id}/approve`` executes a requested job (same rules as below);
  ``POST /offboarding/{id}/reject`` closes it (``offboarding.reject``).
- ``POST /tenants/{t}/export`` → 202; a detached in-process task
  (``svc.spawn``) writes ``exports/tenants/{t}/{job}.zip``.
- ``GET /tenants/{t}/offboarding-plan`` — step 1 of the two-step delete: row counts,
  object prefixes, blockers, and the ``confirm_token`` (the tenant's name/slug, or its
  id when it has none).
- ``DELETE /tenants/{t}?confirm=<token>[&force=true]`` — step 2. Requires
  ``status=suspended`` and no active run (never overridable), plus — unless
  ``force=true``, which is audited — a completed export, no live (or unverifiable)
  w2v licence, and not a partner tenant. Refusals are synchronous; an accepted delete
  answers **202** with the ``running`` job and runs detached — it can outlive
  Cloudflare's 100 s origin timeout. Poll ``GET /offboarding/{id}`` for
  progress/status and the final report. Idempotent: a deleted tenant returns its
  finished job (200); a failed one resumes from its recorded progress on re-DELETE.

Every job heartbeats ``updated_at``; one whose pod died (``running`` past
``RUNNING_LEASE_SECONDS``) is re-claimed and resumed by
``lifecycle.offboarding_resume`` (at startup, then every minute), or by re-POSTing /
re-DELETEing.
"""

from __future__ import annotations

from typing import Literal, cast
from uuid import UUID, uuid7

from fastapi import APIRouter, Query, Request, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.minio_stream import stream_object
from src.api.routes._common import (
    AuditRepoDep,
    LicensesClientDep,
    SessionDep,
    SubjectIdDep,
    SuperAdminDep,
    TenantAdminDep,
)
from src.api.routes.tenancy.errors import TenancyError
from src.api.schemas.tenant_offboarding import (
    OffboardingApprove,
    OffboardingBlocker,
    OffboardingJob,
    OffboardingJobList,
    OffboardingPlan,
    OffboardingReject,
    OffboardingRequestCreate,
)
from src.config import get_settings
from src.database.models import Tenant, TenantOffboarding
from src.services import tenant_offboarding as svc
from src.services.tts_storage import _get_storage

router = APIRouter(prefix="/api/tenant/offboarding", tags=["tenancy"])
admin_router = APIRouter(prefix="/api/admin", tags=["admin"])

_UNIQUE_VIOLATION = "23505"


def _err(http: int, code: str, message: str, **details: object) -> TenancyError:
    return TenancyError(http, code, message, details or None)


async def _storage() -> svc.OffboardingStorage:
    storage = svc.as_offboarding_storage(await _get_storage())
    if storage is None:
        raise _err(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "STORAGE_UNAVAILABLE",
            "artifact storage does not support offboarding",
        )
    return storage


def _factory(request: Request) -> async_sessionmaker[AsyncSession]:
    return cast(async_sessionmaker[AsyncSession], request.app.state.session_factory)


def _export_ready(job: TenantOffboarding) -> bool:
    """A finished export whose stored key sits under its own tenant's export prefix."""
    return bool(
        job.kind == "export"
        and job.status == "done"
        and job.artifact_key
        and job.artifact_key.startswith(svc.export_prefix(job.tenant_id))
    )


def _download_path(job: TenantOffboarding, *, operator_view: bool) -> str | None:
    if not _export_ready(job):
        return None
    if operator_view:
        return f"/api/admin/offboarding/{job.id}/download"
    return f"/api/tenant/offboarding/{job.id}/download"


async def _stream_export(job: TenantOffboarding, request: Request) -> Response:
    if not _export_ready(job):
        raise _err(status.HTTP_404_NOT_FOUND, "EXPORT_NOT_FOUND", "no finished export")
    response = await stream_object(
        get_settings().storage.minio_bucket,
        cast(str, job.artifact_key),
        request,
        media_type="application/zip",
    )
    response.headers["content-disposition"] = (
        f'attachment; filename="tenant-{job.tenant_id}-export-{job.id}.zip"'
    )
    return response


def _job_out(job: TenantOffboarding, *, operator_view: bool) -> OffboardingJob:
    return OffboardingJob(
        id=job.id,
        tenant_id=job.tenant_id,
        kind=cast(Literal["export", "delete"], job.kind),
        status=cast(Literal["requested", "running", "done", "failed", "rejected"], job.status),
        requested_by=job.requested_by,
        decided_by=job.decided_by if operator_view else None,
        reason=job.reason,
        force=job.force,
        error=job.error,
        progress=dict(job.progress or {}),
        result=job.result,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        download_url=_download_path(job, operator_view=operator_view),
    )


async def _locked_tenant(session: AsyncSession, tenant_id: UUID) -> Tenant:
    tenant = (
        await session.execute(select(Tenant).where(Tenant.id == tenant_id).with_for_update())
    ).scalar_one_or_none()
    if tenant is None:
        raise _err(status.HTTP_404_NOT_FOUND, "TENANT_NOT_FOUND", "tenant is not provisioned")
    return tenant


def _confirm_token(tenant: Tenant) -> str:
    return tenant.name or str(tenant.id)


async def _licence_state(licenses: LicensesClientDep, tenant_id: UUID) -> str:
    return await svc.fresh_licence_state(licenses, tenant_id)


# ─── tenant side ─────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=OffboardingJob,
    status_code=status.HTTP_201_CREATED,
    operation_id="tenant.offboarding.request",
)
async def request_offboarding(
    body: OffboardingRequestCreate,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> OffboardingJob:
    await _locked_tenant(session, tenant_id)
    job = TenantOffboarding(
        id=uuid7(),
        tenant_id=tenant_id,
        kind=body.kind,
        status="requested",
        requested_by=actor,
        reason=body.reason,
        force=False,
        progress={},
        created_at=svc.now_utc(),
        updated_at=svc.now_utc(),
    )
    session.add(job)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        if getattr(exc.orig, "sqlstate", None) == _UNIQUE_VIOLATION:
            raise _err(
                status.HTTP_409_CONFLICT,
                "OFFBOARDING_ALREADY_REQUESTED",
                f"an open {body.kind} request already exists for this tenant",
            ) from None
        raise
    await audit.record(
        actor_subject_id=actor,
        action="offboarding.request",
        target_tenant_id=tenant_id,
        detail={"job_id": str(job.id), "kind": body.kind, "reason": body.reason},
    )
    await session.commit()
    return _job_out(job, operator_view=False)


@router.get("", response_model=OffboardingJobList, operation_id="tenant.offboarding.list")
async def list_own_offboarding(
    tenant_id: TenantAdminDep, session: SessionDep
) -> OffboardingJobList:
    rows = (
        (
            await session.execute(
                select(TenantOffboarding)
                .where(TenantOffboarding.tenant_id == tenant_id)
                .order_by(TenantOffboarding.created_at.desc(), TenantOffboarding.id.desc())
                .limit(100)
            )
        )
        .scalars()
        .all()
    )
    return OffboardingJobList(items=[_job_out(r, operator_view=False) for r in rows])


@router.get("/{job_id}/download", operation_id="tenant.offboarding.download")
async def download_own_export(
    job_id: UUID, request: Request, tenant_id: TenantAdminDep, session: SessionDep
) -> Response:
    """The tenant's own finished export. Another tenant's job is the same 404."""
    job = (
        await session.execute(
            select(TenantOffboarding).where(
                TenantOffboarding.id == job_id, TenantOffboarding.tenant_id == tenant_id
            )
        )
    ).scalar_one_or_none()
    if job is None:
        raise _err(status.HTTP_404_NOT_FOUND, "OFFBOARDING_NOT_FOUND", "offboarding job not found")
    return await _stream_export(job, request)


# ─── operator side ───────────────────────────────────────────────────────────


@admin_router.get(
    "/offboarding", response_model=OffboardingJobList, operation_id="admin.offboarding.list"
)
async def list_offboarding(
    _: SuperAdminDep,
    session: SessionDep,
    job_status: Literal["requested", "running", "done", "failed", "rejected"] | None = Query(
        None, alias="status"
    ),
    kind: Literal["export", "delete"] | None = Query(None),
    tenant_id: UUID | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> OffboardingJobList:
    stmt = select(TenantOffboarding)
    if job_status is not None:
        stmt = stmt.where(TenantOffboarding.status == job_status)
    if kind is not None:
        stmt = stmt.where(TenantOffboarding.kind == kind)
    if tenant_id is not None:
        stmt = stmt.where(TenantOffboarding.tenant_id == tenant_id)
    rows = (
        (
            await session.execute(
                stmt.order_by(
                    TenantOffboarding.created_at.desc(), TenantOffboarding.id.desc()
                ).limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return OffboardingJobList(items=[_job_out(r, operator_view=True) for r in rows])


async def _job_or_404(
    session: AsyncSession, job_id: UUID, *, lock: bool = False
) -> TenantOffboarding:
    job = await session.get(TenantOffboarding, job_id, with_for_update=lock)
    if job is None:
        raise _err(status.HTTP_404_NOT_FOUND, "OFFBOARDING_NOT_FOUND", "offboarding job not found")
    return job


@admin_router.get(
    "/offboarding/{job_id}", response_model=OffboardingJob, operation_id="admin.offboarding.get"
)
async def get_offboarding(job_id: UUID, _: SuperAdminDep, session: SessionDep) -> OffboardingJob:
    return _job_out(await _job_or_404(session, job_id), operator_view=True)


@admin_router.get("/offboarding/{job_id}/download", operation_id="admin.offboarding.download")
async def download_export(
    job_id: UUID, request: Request, _: SuperAdminDep, session: SessionDep
) -> Response:
    return await _stream_export(await _job_or_404(session, job_id), request)


@admin_router.post(
    "/offboarding/{job_id}/reject",
    response_model=OffboardingJob,
    operation_id="admin.offboarding.reject",
)
async def reject_offboarding(
    job_id: UUID,
    body: OffboardingReject,
    _: SuperAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> OffboardingJob:
    job = await _job_or_404(session, job_id, lock=True)
    if job.status != "requested":
        raise _err(
            status.HTTP_409_CONFLICT,
            "OFFBOARDING_NOT_PENDING",
            f"only a requested job can be rejected (status is {job.status!r})",
        )
    job.status, job.decided_by = "rejected", actor
    rejection: dict[str, object] = {"rejection_reason": body.reason}
    job.result = rejection
    job.finished_at = job.updated_at = svc.now_utc()
    await audit.record(
        actor_subject_id=actor,
        action="offboarding.reject",
        target_tenant_id=job.tenant_id,
        detail={"job_id": str(job.id), "kind": job.kind, "reason": body.reason},
    )
    await session.commit()
    return _job_out(job, operator_view=True)


@admin_router.post(
    "/offboarding/{job_id}/approve",
    response_model=OffboardingJob,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="admin.offboarding.approve",
)
async def approve_offboarding(
    job_id: UUID,
    body: OffboardingApprove,
    request: Request,
    response: Response,
    _: SuperAdminDep,
    actor: SubjectIdDep,
    licenses: LicensesClientDep,
    session: SessionDep,
) -> OffboardingJob:
    job = await _job_or_404(session, job_id)
    if job.status != "requested":
        raise _err(
            status.HTTP_409_CONFLICT,
            "OFFBOARDING_NOT_PENDING",
            f"only a requested job can be approved (status is {job.status!r})",
        )
    if job.kind == "export":
        return await _start_export(request, session, job.tenant_id, actor, job_id)
    return await _delete(
        request,
        response,
        session,
        licenses,
        job.tenant_id,
        actor,
        confirm=body.confirm,
        force=body.force,
        job_hint=job_id,
    )


@admin_router.post(
    "/tenants/{tenant_id}/export",
    response_model=OffboardingJob,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="admin.tenants.export",
)
async def export_tenant(
    tenant_id: UUID,
    request: Request,
    _: SuperAdminDep,
    actor: SubjectIdDep,
    session: SessionDep,
) -> OffboardingJob:
    return await _start_export(request, session, tenant_id, actor, None)


async def _start_export(
    request: Request,
    session: AsyncSession,
    tenant_id: UUID,
    actor: UUID,
    job_hint: UUID | None,
) -> OffboardingJob:
    storage = await _storage()
    tenant = await _locked_tenant(session, tenant_id)
    try:
        job = await svc.start_tenant_export(session, tenant, actor, job_hint=job_hint)
    except svc.StartRefusedError as exc:
        raise _refused(exc) from None
    svc.spawn(
        svc.execute_job(_factory(request), storage, job.id, "export"),
        name=f"offboarding-export-{job.id}",
    )
    return _job_out(job, operator_view=True)


@admin_router.get(
    "/tenants/{tenant_id}/offboarding-plan",
    response_model=OffboardingPlan,
    operation_id="admin.tenants.offboarding_plan",
)
async def offboarding_plan(
    tenant_id: UUID, _: SuperAdminDep, licenses: LicensesClientDep, session: SessionDep
) -> OffboardingPlan:
    tenant = await session.get(Tenant, tenant_id)
    if tenant is None:
        raise _err(status.HTTP_404_NOT_FOUND, "TENANT_NOT_FOUND", "tenant is not provisioned")
    if tenant.status == "deleted":
        blockers = [svc.Blocker("ALREADY_DELETED", "the tenant was deleted", overridable=False)]
    else:
        blockers = await svc.delete_blockers(
            session,
            tenant,
            licence_state=await _licence_state(licenses, tenant_id),
            partner_licence_ids=get_settings().w2v_partner_license_ids,
        )
    return OffboardingPlan(
        tenant_id=tenant_id,
        tenant_name=tenant.name,
        tenant_status=tenant.status,
        confirm_token=_confirm_token(tenant),
        deletable=not blockers,
        blockers=[OffboardingBlocker(**b.__dict__) for b in blockers],
        rows=await svc.row_counts(session, tenant_id),
        object_prefixes=await svc.all_object_prefixes(session, tenant_id),
        retained=svc.retained(tenant_id),
        not_deleted=list(svc.DOWNSTREAM_NOT_DELETED),
    )


@admin_router.delete(
    "/tenants/{tenant_id}",
    response_model=OffboardingJob,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="admin.tenants.delete",
)
async def delete_tenant(
    tenant_id: UUID,
    request: Request,
    response: Response,
    _: SuperAdminDep,
    actor: SubjectIdDep,
    licenses: LicensesClientDep,
    session: SessionDep,
    confirm: str = Query(..., description="the plan's confirm_token"),
    force: bool = Query(False),
) -> OffboardingJob:
    return await _delete(
        request,
        response,
        session,
        licenses,
        tenant_id,
        actor,
        confirm=confirm,
        force=force,
        job_hint=None,
    )


def _refused(exc: svc.StartRefusedError) -> TenancyError:
    return _err(exc.http, exc.code, exc.message, **exc.details)


async def _delete(
    request: Request,
    response: Response,
    session: AsyncSession,
    licenses: LicensesClientDep,
    tenant_id: UUID,
    actor: UUID,
    *,
    confirm: str | None,
    force: bool,
    job_hint: UUID | None,
) -> OffboardingJob:
    storage = await _storage()
    tenant = await _locked_tenant(session, tenant_id)
    if confirm != _confirm_token(tenant):
        raise _err(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "CONFIRMATION_MISMATCH",
            "confirm must equal the plan's confirm_token "
            "(GET /api/admin/tenants/{id}/offboarding-plan)",
        )
    try:
        started = await svc.start_tenant_delete(
            session, licenses, tenant, actor, force=force, job_hint=job_hint
        )
    except svc.StartRefusedError as exc:
        raise _refused(exc) from None
    if not started.started:
        response.status_code = status.HTTP_200_OK  # idempotent re-run: nothing started
        return _job_out(started.job, operator_view=True)
    # 202: the delete runs detached (it can outlive Cloudflare's 100 s origin
    # timeout); poll GET /api/admin/offboarding/{id} for progress and the report.
    svc.spawn(
        svc.execute_job(_factory(request), storage, started.job.id, "delete"),
        name=f"offboarding-delete-{started.job.id}",
    )
    return _job_out(started.job, operator_view=True)
