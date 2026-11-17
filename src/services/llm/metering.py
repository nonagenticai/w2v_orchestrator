"""Record in-process LLM usage (tenant BYO, and the platform vLLM) into ``run_usage``.

WM-path usage reaches ``run_usage`` by the billing mirror (``RunUsageReconciler``):
WM emits cost events, billing prices them, the reconciler copies them in. A call
the orchestrator makes itself on the TENANT's key never passes WM or billing, so it
is written here directly — one row per call, ``cost_usd`` 0 (the tenant pays their
provider), ``metadata.funding = tenant_byo`` (memory: usage-funding taxonomy) and
``metadata.source = IN_PROCESS_USAGE_SOURCE`` so the billing mirror's
delete-then-insert leaves these rows alone.

The orchestrator's own platform vLLM-direct calls (``LLM_BACKEND=vllm``) skip WM and
billing the same way, so they are written here too — ``funding = local_gpu`` (the
``usage_metrics.funding = platform_direct`` the vLLM backend reports), no slot, no
connection: :func:`record_platform_vllm_usage`.

Best-effort: the LLM call has already succeeded and been paid for by the tenant;
failing the stage because the ledger write failed would throw that answer away.
A failed write is logged at ERROR with everything needed to backfill it.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Final
from uuid import UUID, uuid7

from w2v_common.auth.identity_context import (
    get_actor_id_from_context,
    get_correlation_id_from_context,
    get_stage_from_context,
    get_subject_id_from_context,
    get_tenant_id_from_context,
    get_w2v_project_id_from_context,
)
from w2v_common.clients import WMResponse

from src.database.models import RunUsage
from src.database.repositories.run_usage_repository import (
    FUNDING_LOCAL_GPU,
    IN_PROCESS_USAGE_SOURCE,
    attribution_subject,
    known_funding,
)
from src.database.session import bind_session_tenant, get_db_session
from src.services.llm.in_process import LLMResult
from src.services.providers.resolver import ResolvedSlot

log = logging.getLogger(__name__)

FUNDING_TENANT_BYO: Final = "tenant_byo"

#: w2v_common's vLLM-direct ``usage_metrics.funding`` vocabulary → the ``run_usage``
#: taxonomy. ``platform_direct`` = the platform's own GPU (pgx vLLM).
_USAGE_METRICS_FUNDING: Final[dict[str, str]] = {"platform_direct": FUNDING_LOCAL_GPU}


def funding_from_usage_metrics(raw: object) -> str:
    """The ``run_usage.funding`` for a platform vLLM call's reported funding.

    ``platform_direct`` → ``local_gpu``; a value already in the taxonomy is kept;
    anything else (absent / unknown) is the platform GPU — the only thing the
    vLLM-direct backend can be."""
    if isinstance(raw, str) and raw in _USAGE_METRICS_FUNDING:
        return _USAGE_METRICS_FUNDING[raw]
    return known_funding(raw) or FUNDING_LOCAL_GPU


def _run_id() -> UUID | None:
    raw = get_correlation_id_from_context()
    if not raw:
        return None
    try:
        return UUID(raw)
    except ValueError:
        return None


def _identity_metadata(metadata: dict[str, object]) -> object:
    """Add project / actor / subject from context to ``metadata``; returns the subject.

    Per-user usage must be queryable (memory: usage-funding-taxonomy-and-byo). Both
    identities, never one substituted for the other: ``actor_id`` is WHO acted
    (Keycloak ``sub``; absent for license-token / service work), ``subject_id`` is the
    bound subject the rest of this service attributes writes to. Empty → omitted."""
    project = get_w2v_project_id_from_context()
    if project:
        metadata["project_id"] = project
    actor = get_actor_id_from_context()
    if actor:
        metadata["actor_id"] = actor
    subject = get_subject_id_from_context()
    if subject:
        metadata["subject_id"] = subject
    return subject


def _int(raw: object) -> int:
    try:
        return max(int(raw), 0)  # ty: ignore[invalid-argument-type]  # why: usage_metrics values are JSON scalars; a non-numeric one falls to 0 below
    except TypeError, ValueError:
        return 0


def platform_vllm_usage_row(
    *, run_id: UUID, stage_name: str, worker_name: str, response: WMResponse
) -> RunUsage:
    """One ``run_usage`` row for an orchestrator platform vLLM-direct call."""
    metrics = response.usage_metrics or {}
    input_tokens = _int(metrics.get("input_tokens"))
    output_tokens = _int(metrics.get("output_tokens"))
    funding = funding_from_usage_metrics(metrics.get("funding"))
    model = metrics.get("model")
    provider = metrics.get("provider")
    metadata: dict[str, object] = {
        "source": IN_PROCESS_USAGE_SOURCE,
        "funding": funding,
        "worker_name": worker_name,
        "requests": _int(metrics.get("requests")),
    }
    subject = _identity_metadata(metadata)
    return RunUsage(
        id=uuid7(),
        run_id=run_id,
        stage_name=stage_name,
        model=model if isinstance(model, str) else None,
        provider=provider if isinstance(provider, str) else "vllm",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=Decimal(0),
        metadata_=metadata,
        slot=None,
        subject_id=attribution_subject(subject),
        funding=funding,
        units=Decimal(input_tokens + output_tokens),
        unit_kind="tokens",
    )


def usage_row(
    *,
    run_id: UUID,
    stage_name: str,
    worker_name: str,
    resolved: ResolvedSlot,
    result: LLMResult,
) -> RunUsage:
    metadata: dict[str, object] = {
        "source": IN_PROCESS_USAGE_SOURCE,
        "funding": FUNDING_TENANT_BYO,
        "slot": resolved.ref.slot,
        "provider_kind": result.provider_kind,
        "connection_id": str(resolved.ref.connection_id),
        "connection_version": resolved.ref.version,
        "worker_name": worker_name,
    }
    subject = _identity_metadata(metadata)
    if result.overrides_ignored:
        metadata["overrides_ignored"] = dict(result.overrides_ignored)
    return RunUsage(
        id=uuid7(),
        run_id=run_id,
        stage_name=stage_name,
        model=result.model,
        provider=result.provider_kind,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=Decimal(0),
        metadata_=metadata,
        slot=resolved.ref.slot,
        subject_id=attribution_subject(subject),
        funding=FUNDING_TENANT_BYO,
        units=Decimal(result.input_tokens + result.output_tokens),
        unit_kind="tokens",
        connection_id=resolved.ref.connection_id,
        connection_version=resolved.ref.version,
    )


# TODO(usage-funding-taxonomy-and-byo.md): also EMIT a billing usage event for these
# calls (funding=tenant_byo, cost 0, project + actor) so billing's per-project/per-user
# usage views include BYO traffic. Today the row lives only in the local ``run_usage``.
async def record_in_process_usage(
    *, tenant_id: UUID, worker_name: str, resolved: ResolvedSlot, result: LLMResult
) -> bool:
    """Write one ``run_usage`` row for the bound run; ``False`` when not written."""
    run_id = _run_id()
    if run_id is None:
        log.warning(
            "in-process LLM usage not recorded: no run bound (worker=%s tokens in=%d out=%d)",
            worker_name,
            result.input_tokens,
            result.output_tokens,
        )
        return False
    row = usage_row(
        run_id=run_id,
        stage_name=get_stage_from_context() or worker_name,
        worker_name=worker_name,
        resolved=resolved,
        result=result,
    )
    try:
        async with get_db_session() as session:
            await bind_session_tenant(session, tenant_id)
            session.add(row)
    except Exception:
        log.exception(
            "in-process LLM usage write FAILED run=%s stage=%s worker=%s model=%s "
            "provider=%s tokens in=%d out=%d — backfill needed",
            run_id,
            row.stage_name,
            worker_name,
            result.model,
            result.provider_kind,
            result.input_tokens,
            result.output_tokens,
        )
        return False
    return True


def _bound_tenant() -> UUID | None:
    raw = get_tenant_id_from_context()
    if not raw:
        return None
    try:
        return UUID(str(raw))
    except ValueError:
        return None


async def record_platform_vllm_usage(*, worker_name: str, response: WMResponse) -> bool:
    """Write one ``run_usage`` row for a platform vLLM-direct call of the bound run.

    ``False`` (logged, never raised) when no run or tenant is bound — a call outside
    a run (e.g. an ad-hoc API proxy) has no ledger to land in — or the write failed."""
    run_id = _run_id()
    tenant_id = _bound_tenant()
    metrics = response.usage_metrics or {}
    if run_id is None or tenant_id is None:
        log.warning(
            "platform vLLM usage not recorded: no %s bound (worker=%s tokens in=%s out=%s)",
            "run" if run_id is None else "tenant",
            worker_name,
            metrics.get("input_tokens"),
            metrics.get("output_tokens"),
        )
        return False
    row = platform_vllm_usage_row(
        run_id=run_id,
        stage_name=get_stage_from_context() or worker_name,
        worker_name=worker_name,
        response=response,
    )
    try:
        async with get_db_session() as session:
            await bind_session_tenant(session, tenant_id)
            session.add(row)
    except Exception:
        log.exception(
            "platform vLLM usage write FAILED run=%s stage=%s worker=%s model=%s "
            "tokens in=%d out=%d — backfill needed",
            run_id,
            row.stage_name,
            worker_name,
            row.model,
            row.input_tokens,
            row.output_tokens,
        )
        return False
    return True
