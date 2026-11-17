"""``POST /api/runs/{run_id}/usage`` — service-reported LLM usage for a run.

Since services call the platform vLLM directly (w2v_common ``LLM_BACKEND=vllm``),
their LLM calls pass neither WM nor billing, so the billing mirror
(``RunUsageReconciler``) never sees them and the run showed 0 tokens. The calling
service reports each call here instead; the row is the orchestrator's own
``run_usage`` ledger (``funding = local_gpu``, cost 0 — the platform GPU), tagged
``metadata.source = service_reported`` so the billing mirror leaves it alone.

Auth: the normal AuthMiddleware authenticates the caller. The run must belong to
the EFFECTIVE tenant — unknown or another tenant's run is a 404 and nothing is
written. Idempotent on ``(run_id, call_id)``: a retried report answers
``recorded=false`` and writes no second row.

Route-local delegation. A w2v service reports with its SERVICE-ACCOUNT licence,
which introspects to the SA tenant, never the run's. w2v_common's delegation
(``AUTH_DELEGATING_SUBJECTS``) would fix that but is GLOBAL — a listed subject could
act as any tenant on every orchestrator route — so it stays off. Instead THIS route
alone honours ``X-Effective-Tenant-Id`` when the caller is on the licence lane and
its INTROSPECTED subject (``request.state.license["subject_id"]``, never a header)
is on ``USAGE_REPORT_DELEGATING_SUBJECTS``. Anyone else's header is ignored, as
w2v_common ignores it. That is why this router is mounted WITHOUT the router-level
``TENANT_GATE`` (whose RLS bind / known-tenant / licence checks would run as the SA
tenant): the handler binds RLS to the effective tenant itself, and run ownership
implies a known tenant. No licence gate: this records tokens already spent on an
existing run, it grants nothing.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from uuid import UUID, uuid7

from fastapi import APIRouter, HTTPException, Request, status
from w2v_common.auth.identity_context import get_subject_id_from_context
from w2v_common.auth.outbound import EFFECTIVE_TENANT_HEADER

from src.api.routes._common import SessionDep, TenantIdDep
from src.api.schemas import RunUsageReport, RunUsageReportResult
from src.config import get_settings
from src.database.models import RunUsage
from src.database.repositories.run_usage_repository import (
    FUNDING_LOCAL_GPU,
    SERVICE_REPORTED_USAGE_SOURCE,
    RunUsageRepository,
    attribution_subject,
)
from src.database.session import bind_session_tenant

from ._shared import _tenant_owns_run

log = logging.getLogger(__name__)

#: Mounted in ``asgi.py`` WITHOUT ``TENANT_GATE`` — see the module docstring.
usage_router = APIRouter(prefix="/api/runs", tags=["runs"])


def _delegating_subjects() -> frozenset[str]:
    raw = get_settings().usage_report_delegating_subjects
    return frozenset(s.strip() for s in raw.split(",") if s.strip())


def effective_report_tenant(request: Request, caller_tenant: UUID) -> tuple[UUID, bool]:
    """``(tenant, delegated)`` this report acts for.

    The caller's own tenant, unless the caller authenticated on the LICENCE lane
    with an introspected subject on ``USAGE_REPORT_DELEGATING_SUBJECTS`` and names
    another tenant in ``X-Effective-Tenant-Id``. A malformed header from such a
    caller is a 422 — never a silent fall-back onto the SA tenant."""
    raw = (request.headers.get(EFFECTIVE_TENANT_HEADER) or "").strip()
    if not raw or raw == str(caller_tenant):
        return caller_tenant, False
    introspection = getattr(request.state, "license", None)
    subject = introspection.get("subject_id") if isinstance(introspection, dict) else None
    if not subject or str(subject) not in _delegating_subjects():
        return caller_tenant, False
    try:
        return UUID(raw), True
    except ValueError:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"{EFFECTIVE_TENANT_HEADER} is not a UUID"
        ) from None


def _reported_row(run_id: UUID, body: RunUsageReport, subject: UUID | None) -> RunUsage:
    metadata: dict[str, object] = {
        "source": SERVICE_REPORTED_USAGE_SOURCE,
        "funding": FUNDING_LOCAL_GPU,
        "call_id": body.call_id,
        "service": body.service,
        "total_tokens": body.total_tokens,
    }
    if subject is not None:
        metadata["subject_id"] = str(subject)
    tokens = body.input_tokens + body.output_tokens
    return RunUsage(
        id=uuid7(),
        run_id=run_id,
        # stage_name is NOT NULL: a call outside a named stage is attributed to the
        # reporting service, so it still groups somewhere legible.
        stage_name=body.stage or body.service,
        model=body.model,
        provider=body.provider,
        input_tokens=body.input_tokens,
        output_tokens=body.output_tokens,
        cost_usd=Decimal(0),
        metadata_=metadata,
        slot=None,
        subject_id=subject,
        funding=FUNDING_LOCAL_GPU,
        units=Decimal(tokens),
        unit_kind="tokens",
    )


@usage_router.post(
    "/{run_id}/usage",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RunUsageReportResult,
    operation_id="runs.report_usage",
)
async def report_run_usage(
    run_id: UUID,
    body: RunUsageReport,
    request: Request,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> RunUsageReportResult:
    effective, delegated = effective_report_tenant(request, tenant_id)
    await bind_session_tenant(session, effective)
    run = await _tenant_owns_run(session, run_id, effective)
    if run is None:
        raise HTTPException(404, "run not found")
    snapshot_subject = attribution_subject((run.config_snapshot or {}).get("subject_id"))
    # A delegated report's caller identity is the SERVICE account: attribute to the
    # run's own subject (pinned at creation, as the billing mirror does). Otherwise
    # the caller's subject, falling back to the run's.
    subject = (
        snapshot_subject
        if delegated
        else attribution_subject(
            getattr(request.state, "subject_id", None) or get_subject_id_from_context()
        )
        or snapshot_subject
    )
    recorded = await RunUsageRepository(session).insert_reported(
        _reported_row(run_id, body, subject)
    )
    await session.commit()
    if not recorded:
        log.info(
            "duplicate usage report ignored run=%s call_id=%s service=%s",
            run_id,
            body.call_id,
            body.service,
        )
    return RunUsageReportResult(recorded=recorded)
