"""Admin tier — platform + tenant aggregates, the tenant registry, and the
persistent admin audit log (Phase B).

- GET   /api/admin/tenants        (super_admin)  → every known tenant + name/status.
- PATCH /api/admin/tenants/{id}   (super_admin)  → rename / set status (upsert) + audit.
- POST  /api/admin/scope-switch   (super_admin)  → record a deliberate scope switch.
- GET   /api/admin/audit          (super_admin)  → the admin audit log (paginated).
- GET   /api/admin/request-audit  (super_admin)  → the per-request audit (paginated).
- GET   /api/admin/overview       (tenant_admin) → the caller's-tenant aggregate.
- POST  /api/admin/runs/{id}/cancel (super_admin) → cancel ANY tenant's run tree + audit.
- POST  /api/admin/provision      (super_admin)  → new tenant + first-user invite + licences.
- PUT   /api/admin/tenants/{id}/partner (super_admin) → bind an existing tenant to a
  partner licence (P9-5) so the partner can mint its machine keys.
- GET   /api/admin/tenants/{id}/llm-access (super_admin) → the tenant's LLM/GPU posture
  (licences ``byo_mode`` for app ``w2v``) + its source.
- PUT   /api/admin/tenants/{id}/llm-access (super_admin) → set it (per-app override) + audit.

P2 tenancy (WP4b): the member roster and role management moved to w2v-owned
membership (``/api/tenant/members``, WP4a); the Keycloak realm-role route and
the ``tenant_id``/``tenant_slug`` user attributes are gone — w2v, not Keycloak,
says which tenant a person belongs to.

The ``tenants`` registry is a name + status annotation layer keyed on the
claim-derived tenant_id (w2v does not mint tenant ids — see
``admin_repository``). A super-admin scope-switch into an unknown or suspended
tenant is rejected fail-closed in ``require_tenant_admin``.

Deliberate admin actions (tenant updates, scope switches) are persisted to
``admin_audit_log`` (migration 0022) — one row per action, written inside the
acting route's session so the audit row commits atomically with the action.

The high-volume per-request firehose (mutations / auth failures / cross-tenant
access) is persisted separately to ``request_audit`` (migration 0026) by
``AuditMiddleware`` off the hot path, and read back via GET /api/admin/request-audit.

``request_audit`` retention/pruning is shipped (a periodic batched DELETE in
``asgi.py``, gated by ``request_audit_retention_days``). The Phase D RLS backstop
is also shipped but INERT — the policies/GUC plumbing are in place while the app
still connects as the table owner with ``TENANT_RLS_MODE=off``; it activates only
at the operator-gated cutover (restricted no-``BYPASSRLS`` role + the flag flipped
atomically).
"""

import datetime
import logging
from typing import Literal
from uuid import UUID, uuid7

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, status
from psycopg.errors import InsufficientPrivilege
from sqlalchemy import Row, func, select, text, union
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes._common import (
    LICENSE_GATE_ONLY,
    AdminRepoDep,
    AuditRepoDep,
    IdentityInviterDep,
    LicenceKeyringDep,
    LicensesClientDep,
    ProvisioningRepoDep,
    RequestAuditRepoDep,
    RunUsageReconcilerDep,
    SessionDep,
    SubjectIdDep,
    SuperAdminDep,
    TenantAdminDep,
)
from src.api.routes.runs.lifecycle import cancel_run_tree
from src.api.routes.tenancy.errors import (
    TenancyError,
    identity_error,
    licences_refused,
    tenant_not_onboarded,
)
from src.api.routes.tenancy.licence_token import (
    keyring_unavailable,
    licences_unavailable,
)
from src.api.schemas import (
    AdminAuditEntry,
    AdminAuditList,
    AdminOverview,
    AdminProvisionLlmAccess,
    AdminProvisionRequest,
    AdminProvisionResult,
    AdminRunList,
    AdminRunListItem,
    AdminTenantList,
    AdminTenantLlmAccess,
    AdminTenantLlmAccessUpdate,
    AdminTenantSummary,
    AdminTenantUpdate,
    AdminUsageDetail,
    RequestAuditEntry,
    RequestAuditList,
    ScopeSwitchAuditRequest,
)
from src.api.schemas.tenancy_keys import TenantPartnerBind, TenantPartnerBinding
from src.api.schemas.tenant_audit import AuditAction
from src.clients.downstream_credential import DownstreamCredentialMissingError
from src.clients.identity_inviter import IdentityInviteError
from src.clients.licenses_client import (
    TENANT_NOT_ACCESSIBLE,
    LicencesAccessRefusedError,
    LicensesClientError,
    TenantNotOnboardedError,
)
from src.config import get_settings
from src.database.models import TenantInvite, TenantMember
from src.database.repositories.admin_repository import AdminRepository
from src.database.repositories.run_repository import RunRepository
from src.database.session import bind_session_tenant
from src.services.run_usage_reconciler import (
    COST_UNAVAILABLE_BILLING_CREDENTIAL_MISSING,
    RunUsageReconciler,
)
from src.services.tenant_founding import found_tenant, license_new_tenant
from src.services.tenant_licence import LicencesTenantNotOnboardedError
from src.services.tenant_llm_access import (
    TenantLlmAccess,
    UnknownLlmAccessModeError,
    apply_tenant_llm_access,
    read_tenant_llm_access,
)
from src.services.tenant_partner_bind import bind_tenant_to_partner
from src.services.tenant_provisioning import invite_user

_log = logging.getLogger("w2v_orchestrator.admin")

router = APIRouter(prefix="/api/admin", tags=["admin"])

# Every mutating handler below takes `session: SessionDep` purely to commit before
# it responds. The repositories only FLUSH and `get_db_session` commits in its
# ``yield`` teardown, which FastAPI runs AFTER the response is sent — see
# CLAUDE.md § "A write handler MUST commit before it responds". `session` is the
# SAME session the repo deps were built on (FastAPI dedups `Depends(get_db_session)`),
# so the commit covers the repo writes and the audit row atomically.


def _summary(row: Row, *, switchable: bool) -> AdminTenantSummary:
    # The list query carries run_count/total_cost_usd; the PATCH path
    # (get_tenant_summary) does not — fall back to the schema defaults there.
    return AdminTenantSummary(
        tenant_id=row.tenant_id,
        name=row.name,
        status=row.status,
        project_count=row.project_count,
        last_activity=row.last_activity,
        run_count=getattr(row, "run_count", 0),
        total_cost_usd=float(getattr(row, "total_cost_usd", 0) or 0),
        partner_licence_id=getattr(row, "partner_licence_id", None),
        switchable=switchable,
    )


async def _switchable(request: Request, tenant_ids: list[UUID]) -> dict[UUID, bool]:
    """``AdminTenantSummary.switchable`` for each tenant, decided by
    ``AdminRepository.tenant_switchable`` — the SAME function
    ``TenancyMiddleware`` (and ``require_tenant_admin``) uses to accept or refuse a
    super_admin scope-switch, so the picker can never offer a tenant the switch
    refuses, nor hide one it accepts. No parallel rule.

    Runs in its own short-lived session: for a registry-less tenant
    ``tenant_switchable`` binds the RLS tenant GUC to look for its projects, and
    that binding must not leak into the route's own session."""
    async with request.app.state.session_factory() as session:
        repo = AdminRepository(session)
        return {tid: await repo.tenant_switchable(tid) for tid in tenant_ids}


def _run_item(row: Row, lease_present: bool | None) -> AdminRunListItem:
    return AdminRunListItem(
        run_id=row.run_id,
        tenant_id=row.tenant_id,
        project_id=row.project_id,
        project_title=row.project_title,
        deliverable_id=row.deliverable_id,
        blueprint_id=row.blueprint_id,
        status=row.status,
        current_stage=row.current_stage,
        started_at=row.started_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
        # A cancel/fail does not clear ``runs.paused_at``; only a live pause has one.
        paused_at=row.paused_at if row.status == "paused" else None,
        pause_reason=row.pause_reason if row.status == "paused" else None,
        lease_present=lease_present,
    )


# ─── Licence gate: PER-ROUTE, because this router mixes two kinds ────────
# The data routers in asgi.py are gated wholesale; this one cannot be.
#
#   EXEMPT — legitimately circular. These routes are how a tenant COMES TO HAVE
#   a licence, or how an operator reaches a tenant that does not have one yet:
#   POST /provision, GET /tenants, PATCH /tenants/{id}, POST /scope-switch,
#   GET|PUT /tenants/{id}/llm-access (set during onboarding, before any run).
#   Gating them on entitlement would deadlock
#   onboarding — provisioning mints the very licence the gate would demand.
#   (GET /request-audit is also ungated: it is the platform's own cross-tenant
#   security-forensics stream, not a per-tenant product capability.)
#
#   GATED — product capability, must stop when entitlement stops:
#   GET /overview, /usage, /runs, /audit.
#
# All of them take LICENSE_GATE_ONLY, never the full TENANT_GATE. See that symbol
# in _common.py for why the other two deps are deliberately omitted here (in
# short: require_known_tenant would 403 a freshly-licensed tenant with no
# projects, and an RLS bind would collapse the cross-tenant /runs + /audit panes
# to a single tenant).
#
# SuperAdminDep/TenantAdminDep answer *who are you*; this gate answers *is this
# tenant sanctioned*. A caller can satisfy one and fail the other.


@router.get(
    "/runs",
    response_model=AdminRunList,
    operation_id="admin.runs.list",
    dependencies=LICENSE_GATE_ONLY,
)
async def list_runs(
    request: Request,
    _: SuperAdminDep,
    repo: AdminRepoDep,
    status_filter: Literal["running", "recent"] = Query("running", alias="status"),
    limit: int = Query(100, ge=1, le=500),
) -> AdminRunList:
    """Cross-tenant operator run pane (super-admin only).

    ``?status=running`` (default) lists every in-flight (pending/running/paused) run
    across ALL tenants — the "what is running right now" view — each enriched
    with ``leasePresent`` (a running row with no lease is an orphan no pod is
    executing). ``?status=recent`` lists the most-recently-updated terminal runs.
    Backed by the ``admin_active_runs`` / ``admin_recent_runs`` SECURITY DEFINER
    resolvers; the data-plane RLS policies are unchanged.
    """
    try:
        if status_filter == "running":
            rows = await repo.list_active_runs()
            dispatcher = request.app.state.dispatcher
            items: list[AdminRunListItem] = []
            for row in rows:
                # Lease check is only meaningful for an actually-running row; a
                # pending row hasn't been leased yet. Degraded (no dispatcher) → None.
                lease: bool | None = None
                if dispatcher is not None and row.status == "running":
                    lease = await dispatcher.lease_present(row.run_id)
                items.append(_run_item(row, lease))
            return AdminRunList(items=items, total=len(items))

        rows = await repo.list_recent_runs(limit)
        return AdminRunList(items=[_run_item(row, None) for row in rows], total=len(rows))
    except ProgrammingError as exc:
        # The admin run-view resolvers are SECURITY DEFINER functions whose EXECUTE
        # grant to the runtime role is codified in migration 0043 but historically
        # lived out-of-band. If that grant is ever missing (fresh DB not yet
        # migrated, role recreated), psycopg raises InsufficientPrivilege — surface
        # a clean 503, never a raw 500. Any other ProgrammingError is a real bug.
        if isinstance(exc.orig, InsufficientPrivilege):
            _log.error(
                "admin run-view resolver EXECUTE not granted to the runtime role "
                "(apply migration 0043 / the phase-d grant): %s",
                exc,
            )
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "operator run view temporarily unavailable",
            ) from exc
        raise


@router.post(
    "/runs/{run_id}/cancel",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="admin.runs.cancel",
    dependencies=LICENSE_GATE_ONLY,
)
async def cancel_any_run(
    run_id: UUID,
    request: Request,
    actor: SubjectIdDep,
    _: SuperAdminDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> None:
    """Cancel ANY tenant's run (and every run it spawned) — super_admin only.

    The operator twin of ``POST /api/runs/{id}/cancel`` (owner decision 2026-10-20:
    the cross-tenant run pane must be able to stop a run, not only show it). Same
    semantics, same code: ``cancel_run_tree``. Already-terminal is NOT an error —
    the tenant route answers 204 there too (the descendant sweep still runs, which
    is what stops a completed cant_converge parent's paid relief child).

    Cross-tenant WITHOUT ``row_security=off``: the run's tenant comes from the
    ``run_tenant`` SECURITY DEFINER resolver (the dispatcher-bootstrap sibling of
    ``admin_active_runs``), and the session is then bound to THAT tenant, so every
    read/write below runs under the data-plane RLS policies unchanged. An unknown
    run is 404. The audit row shares the session and commits with the cancel.
    """
    tenant = (
        await session.execute(text("SELECT run_tenant(:rid)"), {"rid": str(run_id)})
    ).scalar_one_or_none()
    if tenant is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    tenant_id = tenant if isinstance(tenant, UUID) else UUID(str(tenant))
    await bind_session_tenant(session, tenant_id)
    run = await RunRepository(session).get(run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    # Recorded BEFORE the cancel so whichever commit the cancel path makes (the
    # mark-cancelled fallback, the degraded path) carries the audit row with it.
    await audit.record(
        actor_subject_id=actor,
        action="run.cancel",
        target_tenant_id=tenant_id,
        detail={"run_id": str(run_id), "prior_status": run.status},
    )
    delivered = await cancel_run_tree(session, request.app.state.dispatcher, run)
    await session.commit()
    if not delivered:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "cancellation recorded but live interrupt unavailable — dispatcher not ready (Redis down)",
        )


@router.get("/tenants", response_model=AdminTenantList, operation_id="admin.tenants.list")
async def list_tenants(request: Request, _: SuperAdminDep, repo: AdminRepoDep) -> AdminTenantList:
    rows = await repo.list_tenants()
    switchable = await _switchable(request, [row.tenant_id for row in rows])
    items = [_summary(row, switchable=switchable[row.tenant_id]) for row in rows]
    return AdminTenantList(items=items, total=len(items))


@router.patch(
    "/tenants/{tenant_id}",
    response_model=AdminTenantSummary,
    operation_id="admin.tenants.update",
)
async def update_tenant(
    request: Request,
    tenant_id: UUID,
    body: AdminTenantUpdate,
    actor: SubjectIdDep,
    _: SuperAdminDep,
    repo: AdminRepoDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> AdminTenantSummary:
    if body.name is None and body.status is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "nothing to update: provide name and/or status",
        )
    # Snapshot the *before* state for the audit trail (None if the tenant was
    # only known via its projects — this PATCH creates the first registry row).
    existing = await repo.get_tenant(tenant_id)
    if existing is not None and existing.status == "deleted":
        # The offboarding tombstone is final: no rename, no "un-delete" to active.
        raise HTTPException(status.HTTP_409_CONFLICT, "tenant was deleted")
    before = {"name": existing.name, "status": existing.status} if existing is not None else None
    row = await repo.upsert_tenant(tenant_id, name=body.name, status=body.status)
    # The audit insert shares this request's session (FastAPI dedups the
    # get_db_session dependency), so it commits atomically with the upsert.
    await audit.record(
        actor_subject_id=actor,
        action="tenant.update",
        target_tenant_id=tenant_id,
        detail={
            "before": before,
            "after": {"name": row.name, "status": row.status},
        },
    )
    await session.commit()
    switchable = await _switchable(request, [tenant_id])
    return _summary(row, switchable=switchable[tenant_id])


# ─── Tenant LLM/GPU access (licences byo_mode, app w2v) ──────────────────


def _llm_access_failure(exc: Exception) -> TenancyError:
    """Map a licences failure on the llm-access routes. A tenant another app owns
    is 403 ``TENANT_NOT_ONBOARDED``. Any other 401/403 is a definite
    refusal (502 ``LICENCES_REFUSED``, with licences' status + machine code — never
    its body); 404 means licences knows no such tenant; an unknown posture is a
    bad upstream answer; anything else (5xx, transport) is 503 retryable."""
    if isinstance(exc, UnknownLlmAccessModeError):
        return TenancyError(502, "LICENCES_BAD_RESPONSE", str(exc))
    if isinstance(exc, TenantNotOnboardedError):
        # licences 403 ``tenant_not_accessible``: another app owns the tenant and
        # granted w2v nothing — an answer about the TENANT, not a fault of ours.
        return tenant_not_onboarded()
    if isinstance(exc, LicensesClientError) and exc.status_code in (401, 403):
        return TenancyError(
            502,
            "LICENCES_REFUSED",
            f"The licences service refused the byo-mode call (HTTP {exc.status_code}"
            + (f", {exc.code}" if exc.code else "")
            + "). This is a permission/configuration fault, not an outage.",
            details={"upstream_status": exc.status_code, "upstream_code": exc.code},
        )
    if isinstance(exc, LicensesClientError) and exc.status_code == 404:
        return TenancyError(
            404,
            "TENANT_NOT_FOUND",
            "The licences service knows no such tenant (or no Studio app for it).",
            details={"upstream_code": exc.code},
        )
    status_code = exc.status_code if isinstance(exc, LicensesClientError) else None
    return licences_unavailable(
        "Could not reach the licences service for the tenant's LLM access"
        + (f" (HTTP {status_code})." if status_code else f" ({type(exc).__name__}).")
    )


def _llm_access_view(access: TenantLlmAccess) -> AdminTenantLlmAccess:
    return AdminTenantLlmAccess(mode=access.mode, source=access.source)


@router.get(
    "/tenants/{tenant_id}/llm-access",
    response_model=AdminTenantLlmAccess,
    operation_id="admin.tenants.llm_access.get",
)
async def get_tenant_llm_access(
    tenant_id: UUID, _: SuperAdminDep, licenses: LicensesClientDep
) -> AdminTenantLlmAccess:
    """The tenant's effective w2v LLM/GPU posture and where it came from."""
    try:
        access = await read_tenant_llm_access(tenant_id, licenses_client=licenses)
    except (LicensesClientError, UnknownLlmAccessModeError, httpx.HTTPError) as exc:
        raise _llm_access_failure(exc) from exc
    return _llm_access_view(access)


@router.put(
    "/tenants/{tenant_id}/llm-access",
    response_model=AdminTenantLlmAccess,
    operation_id="admin.tenants.llm_access.set",
)
async def set_tenant_llm_access(
    tenant_id: UUID,
    body: AdminTenantLlmAccessUpdate,
    actor: SubjectIdDep,
    _: SuperAdminDep,
    licenses: LicensesClientDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> AdminTenantLlmAccess:
    """Set the tenant's w2v posture (licences per-app ``w2v`` override) and audit it.

    The audit action is ``tenant.update`` with an ``llm_access`` detail: a
    dedicated ``tenant.llm_access`` action needs the ``ck_admin_audit_action``
    CHECK widened by a migration."""
    try:
        access = await apply_tenant_llm_access(tenant_id, body.mode, licenses_client=licenses)
    except (LicensesClientError, UnknownLlmAccessModeError, httpx.HTTPError) as exc:
        raise _llm_access_failure(exc) from exc
    await audit.record(
        actor_subject_id=actor,
        action="tenant.update",
        target_tenant_id=tenant_id,
        detail={
            "llm_access": {
                "app": "w2v",
                "before_override": access.prior_mode,
                "after": access.mode,
            }
        },
    )
    await session.commit()
    return _llm_access_view(access)


@router.put(
    "/tenants/{tenant_id}/partner",
    response_model=TenantPartnerBinding,
    operation_id="admin.tenants.partner_bind",
)
async def bind_tenant_partner(
    tenant_id: UUID,
    body: TenantPartnerBind,
    actor: SubjectIdDep,
    _: SuperAdminDep,
    audit: AuditRepoDep,
    licences: LicensesClientDep,
    keyring: LicenceKeyringDep,
    session: SessionDep,
) -> TenantPartnerBinding:
    """Bind an EXISTING tenant to an allow-listed partner licence (P9-5).

    The partner lane's ``POST /api/tenants`` only adopts EMPTY tenants, so a
    tenant that already holds data (merqi's dogfood tenant) needs this operator
    step before its partner can mint keys for it. Rules, under the tenant lock:

    * ``partner_licence_id`` must be in ``W2V_PARTNER_LICENSE_IDS`` → else 422
      ``NOT_A_PARTNER_LICENCE`` (a typo must not bind a tenant to nobody);
    * the tenant must exist (registry row, or data without one — then the row
      is created ``created_via='super_admin'``) → else 404 ``TENANT_NOT_FOUND``;
      a deleted tenant is 409 ``TENANT_DELETED``;
    * ``licence_token`` (optional) is INTROSPECTED: active ∧ this tenant ∧ app
      w2v, else 422 ``LICENCE_TOKEN_MISMATCH``; licences unreachable → 503
      ``LICENCES_UNAVAILABLE``, a definite gateway 4xx → 502 ``LICENCES_REFUSED``.
      It is stored sealed (lease + CAS — 409 ``licence_busy`` /
      ``licence_changed_concurrently``, audit ``tenant.licence_token_set``) and
      its licence becomes ``licence_ref`` (bills the API-key lane);
    * already bound to THIS partner → 200 ``bound=false`` (a given token is still
      verified and installed);
    * bound to ANOTHER partner → 409 ``TENANT_BOUND_TO_OTHER_PARTNER`` (unbinding
      is a separate, deliberate decision — never an overwrite);
    * else bind + audit ``tenant.partner_bind``.
    """
    outcome = await bind_tenant_to_partner(
        session,
        audit=audit,
        licences=licences,
        keyring=keyring,
        tenant_id=tenant_id,
        partner_licence_id=body.partner_licence_id,
        token=body.licence_token.get_secret_value() if body.licence_token is not None else None,
        actor=actor,
    )
    return TenantPartnerBinding(
        tenant_id=tenant_id,
        partner_licence_id=body.partner_licence_id,
        bound=outcome.bound,
        licence_ref=outcome.licence_ref,
    )


@router.post(
    "/scope-switch",
    response_model=AdminAuditEntry,
    operation_id="admin.scope_switch.record",
)
async def record_scope_switch(
    request: Request,
    body: ScopeSwitchAuditRequest,
    actor: SubjectIdDep,
    _: SuperAdminDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> AdminAuditEntry:
    """Record a deliberate super-admin scope switch (the UI toggle).

    Called WITHOUT X-Active-Tenant-Id, so the actor + ``from_tenant`` are the
    super-admin's own claim. ``to_tenant_id`` None = reset back to self. This is
    the durable record-of-intent; the per-request inline log line still covers
    raw-API switches.
    """
    from_tenant = getattr(request.state, "tenant_id", None)
    action = "scope.switch" if body.to_tenant_id is not None else "scope.reset"
    detail = {"from_tenant": str(from_tenant) if from_tenant is not None else None}
    entry = await audit.record(
        actor_subject_id=actor,
        action=action,
        target_tenant_id=body.to_tenant_id,
        detail=detail,
    )
    await session.commit()
    return AdminAuditEntry(
        id=entry.id,
        created_at=entry.created_at,
        actor_subject_id=entry.actor_subject_id,
        action=entry.action,
        target_tenant_id=entry.target_tenant_id,
        detail=entry.detail,
    )


@router.get(
    "/audit",
    response_model=AdminAuditList,
    operation_id="admin.audit.list",
    dependencies=LICENSE_GATE_ONLY,
)
async def list_audit(
    _: SuperAdminDep,
    audit: AuditRepoDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    # The FULL stored vocabulary (== the ck_admin_audit_action CHECK, pinned by
    # tests/unit/test_tenant_audit_vocabulary.py) — an inline copy here had drifted
    # and could not filter the offboarding.* actions.
    action: AuditAction | None = Query(None),
    tenant_id: UUID | None = Query(None),
    actor_subject_id: UUID | None = Query(None),
) -> AdminAuditList:
    rows, total = await audit.list_events(
        limit=limit,
        offset=offset,
        action=action,
        tenant_id=tenant_id,
        actor_subject_id=actor_subject_id,
    )
    items = [
        AdminAuditEntry(
            id=row.id,
            created_at=row.created_at,
            actor_subject_id=row.actor_subject_id,
            action=row.action,
            target_tenant_id=row.target_tenant_id,
            detail=row.detail or {},
        )
        for row in rows
    ]
    return AdminAuditList(items=items, total=total, limit=limit, offset=offset)


@router.get(
    "/request-audit",
    response_model=RequestAuditList,
    operation_id="admin.request_audit.list",
    # Same tier as its sibling /audit (#1003 gated that one and missed this):
    # an operator pane is product capability, not provisioning.
    dependencies=LICENSE_GATE_ONLY,
)
async def list_request_audit(
    _: SuperAdminDep,
    audit: RequestAuditRepoDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    event: Literal["mutation", "auth_failure", "read"] | None = Query(None),
    effective_tenant_id: UUID | None = Query(None),
    actor_subject_id: UUID | None = Query(None),
    method: str | None = Query(None),
    cross_tenant: bool | None = Query(None),
    since: datetime.datetime | None = Query(None),
    until: datetime.datetime | None = Query(None),
    min_status: int | None = Query(None, ge=100, le=599),
) -> RequestAuditList:
    """The middleware-wide per-request audit (mutations / auth failures /
    cross-tenant access), newest-first, paginated + filtered. Super-admin only."""
    rows, total = await audit.list_events(
        limit=limit,
        offset=offset,
        event=event,
        effective_tenant_id=effective_tenant_id,
        actor_subject_id=actor_subject_id,
        method=method.upper() if method else None,
        cross_tenant=cross_tenant,
        since=since,
        until=until,
        min_status=min_status,
    )
    items = [
        RequestAuditEntry(
            id=row.id,
            created_at=row.created_at,
            actor_subject_id=row.actor_subject_id,
            claim_tenant_id=row.claim_tenant_id,
            effective_tenant_id=row.effective_tenant_id,
            cross_tenant=row.cross_tenant,
            event=row.event,
            method=row.method,
            path=row.path,
            resource=row.resource,
            status_code=row.status_code,
            ip=row.ip,
            duration_ms=row.duration_ms,
            credential_fingerprint=row.credential_fingerprint,
            request_id=row.request_id,
            detail=row.detail or {},
        )
        for row in rows
    ]
    return RequestAuditList(items=items, total=total, limit=limit, offset=offset)


async def _reconcile_tenant_cost(reconciler: RunUsageReconciler, tenant_id: UUID) -> str | None:
    """Read-through billing mirror for a tenant; the cost-unavailable reason, or None.

    Billing configured with no credential (enforce, no token) must neither 500
    the admin page nor pass silently: the response says cost is unavailable.
    """
    try:
        await reconciler.reconcile_unsettled_for_tenant(tenant_id)
    except DownstreamCredentialMissingError:
        return COST_UNAVAILABLE_BILLING_CREDENTIAL_MISSING
    return None


@router.get(
    "/overview",
    response_model=AdminOverview,
    operation_id="admin.overview",
    dependencies=LICENSE_GATE_ONLY,
)
async def overview(
    effective_tenant: TenantAdminDep,
    repo: AdminRepoDep,
    reconciler: RunUsageReconcilerDep,
) -> AdminOverview:
    # `effective_tenant` is BOTH the authz gate and the scoped tenant UUID —
    # no separate TenantIdDep, so authz and scope can never diverge.
    # Read-through: mirror billing-priced cost into run_usage for this tenant's
    # terminal runs that haven't been mirrored yet, so the aggregate shows real $.
    unavailable = await _reconcile_tenant_cost(reconciler, effective_tenant)
    data = await repo.tenant_overview(effective_tenant)
    return AdminOverview(
        tenant_id=effective_tenant,
        **data,
        cost_status="unavailable" if unavailable else "ok",
        cost_unavailable_reason=unavailable,
    )


@router.get(
    "/usage",
    response_model=AdminUsageDetail,
    operation_id="admin.usage.detail",
    dependencies=LICENSE_GATE_ONLY,
)
async def usage(
    effective_tenant: TenantAdminDep,
    repo: AdminRepoDep,
    reconciler: RunUsageReconcilerDep,
    bucket: Literal["day", "week", "month"] = Query("day"),
    window_days: int | None = Query(None, ge=1, le=730),
) -> AdminUsageDetail:
    # The drill-down behind /overview. Same gate/scope chokepoint as overview
    # (`TenantAdminDep` honours the X-Active-Tenant-Id scope-switch), so a
    # super-admin acting into another tenant reads that tenant's usage.
    # `window_days` scopes the WHOLE response (totals + by-* breakdowns + the
    # time_series) to metered usage in the window; `bucket` controls only the
    # time_series granularity.
    # Read-through: fill run_usage from billing for not-yet-mirrored terminal runs
    # so /usage shows real cost instead of $0.
    unavailable = await _reconcile_tenant_cost(reconciler, effective_tenant)
    data = await repo.usage_detail(effective_tenant, bucket=bucket, window_days=window_days)
    return AdminUsageDetail(
        tenant_id=effective_tenant,
        **data,
        cost_status="unavailable" if unavailable else "ok",
        cost_unavailable_reason=unavailable,
    )


# ─── Provisioning (Phase C, C-2; P2 tenancy WP4b) ─────────────────────────


def _licences_ownership_refusal(exc: Exception) -> TenancyError | None:
    """A clear answer for licences' ownership / issuer refusals (migration 050),
    or None to re-raise ``exc`` unchanged. Never carries the upstream body."""
    if isinstance(exc, (TenantNotOnboardedError, LicencesTenantNotOnboardedError)):
        return tenant_not_onboarded()
    if not isinstance(exc, LicensesClientError) or exc.status_code != status.HTTP_403_FORBIDDEN:
        return None
    if exc.code == "app_not_issuable_by_caller":
        return TenancyError(
            403,
            "LICENCE_APP_NOT_ISSUABLE",
            "w2v may not issue a licence for one of the requested apps "
            "(licences: app_not_issuable_by_caller). Request only apps w2v issues.",
        )
    if exc.code == TENANT_NOT_ACCESSIBLE:
        return tenant_not_onboarded()
    if isinstance(exc, LicencesAccessRefusedError):  # unknown_api_client / tenant_owner_only
        return licences_refused(exc.status_code or status.HTTP_403_FORBIDDEN)
    return None


async def _invited_subject(session: AsyncSession, email: str) -> UUID | None:
    """The subject id w2v ALREADY knows for the invited email — from its own
    ``tenant_members`` / accepted ``tenant_invites`` (an identity that signed in
    to, or accepted an invite into, any w2v tenant under that address) — else
    None. Recorded only as the audit row's ``target_subject_id``:
    the tenant's licences are held by ``svc:<tenant>:<app>``, never by this person.

    Provisioning's identity invite answers ``202`` with no id and w2v holds no
    Keycloak read power, so a brand-new person has no subject yet: membership
    binds when they log in with the verified email and accept the invite
    (``POST /api/invites/{id}/accept``) — the invite row is the trail until then.
    Several distinct subjects on one address (an account was recreated) are not
    guessed between: None."""
    address = email.strip().lower()
    known = union(
        select(TenantMember.subject_id.label("subject_id")).where(
            func.lower(TenantMember.email) == address
        ),
        select(TenantInvite.accepted_subject_id.label("subject_id")).where(
            func.lower(TenantInvite.email) == address,
            TenantInvite.accepted_subject_id.is_not(None),
        ),
    ).subquery()
    rows = (await session.execute(select(known.c.subject_id).limit(2))).all()
    return rows[0][0] if len(rows) == 1 else None


@router.post(
    "/provision",
    response_model=AdminProvisionResult,
    status_code=status.HTTP_201_CREATED,
    operation_id="admin.provision",
)
async def provision(
    body: AdminProvisionRequest,
    actor: SubjectIdDep,
    _: SuperAdminDep,
    inviter: IdentityInviterDep,
    licenses: LicensesClientDep,
    keyring: LicenceKeyringDep,
    prov: ProvisioningRepoDep,
    session: SessionDep,
) -> AdminProvisionResult:
    """Provision a new tenant + invite its first user (super_admin).

    Mints a fresh ``tenant_id``, creates the ``tenants`` row through the shared
    ``tenant_provisioning.create_tenant`` (``created_via='super_admin'``), and
    invites ``email`` into it with ``workspace_role`` via ``invite_user`` — the
    Keycloak identity is found or created by provisioning's identity invite API
    (no attributes, no realm role; P2: membership lives in w2v), which also sends
    the action email the account needs. Provisioning returns no account state,
    so neither does this response. The invitee becomes a member by accepting
    the invite. The tenant's licences are TENANT-level — ensured through the
    licences ``w2v`` bundle and held by ``svc:<tenant>:<app>``, never by the
    invited person; the audit row's subject is the one w2v's own ``tenant_members``
    already knows for the email, else none until the invitee logs in and accepts
    (w2v reads nothing from Keycloak). A ``pending`` provisioning_audit row is committed first; on
    any failure after that it flips to ``partial`` (KC/licenses side effects are
    NOT rolled back — an operator inspects the row) and the error surfaces.
    Returns the per-app licence tokens ONCE (no password is ever minted).
    """
    if not keyring.configured:
        # The tenant's dedicated w2v licence token is stored sealed; without a
        # keyring it could not be, so refuse BEFORE any side effect.
        raise keyring_unavailable()
    audit_id = await prov.start(
        actor_subject_id=actor, target_email=body.email, payload=body.model_dump(mode="json")
    )
    tenant_id = uuid7()
    user_uuid: UUID | None = None
    try:
        # licences migration 050: register the new tenant as w2v's own BEFORE any
        # licence is minted on it (licences mints only on a registered tenant its
        # caller owns or holds a grant on).
        await found_tenant(
            session, licenses, tenant_id=tenant_id, name=body.tenant_slug, actor_subject_id=actor
        )
        try:
            invite = await invite_user(
                session,
                inviter,
                tenant_id=tenant_id,
                email=body.email,
                role=body.workspace_role,
                invited_by=actor,
            )
        except IdentityInviteError as exc:
            raise identity_error(exc) from exc
        user_uuid = await _invited_subject(session, body.email)

        # The licences, then the LLM access posture (licences' new-tenant default
        # is byo_required, which blocks every run): one shared core with the
        # early-access approve. A posture failure is REPORTED, never raised.
        licensed = await license_new_tenant(
            session,
            licenses,
            keyring,
            tenant_id=tenant_id,
            app_keys=body.apps or get_settings().provisioning.license_app_keys,
            llm_access=body.llm_access,
            mint_tokens=True,
        )
        tokens, license_ids, llm_access = (
            licensed.tokens,
            licensed.license_ids,
            licensed.llm_access,
        )
    except Exception as exc:
        # `prov.finish` is a DB update+commit on its own session and can itself
        # throw (pool exhaustion, a dropped connection). Unguarded, that would
        # REPLACE `exc` on the `raise` below — the caller (and the HTTP response)
        # would see a masking DB error instead of the real provisioning failure
        # (e.g. the 404 for an unknown licenses app). Admin-only, per-request:
        # contain the write failure so it's logged, never masks. Cost of the
        # degraded state: the audit row stays `pending` instead of flipping to
        # `partial` — an operator reading provisioning_audit sees no record of
        # this attempt; the KC user / any minted licenses already created are
        # unaffected (they're external side effects, not rolled back either way).
        try:
            await prov.finish(
                audit_id,
                status="partial",
                error=str(exc),
                target_subject_id=user_uuid,
                tenant_id=tenant_id,
            )
        except Exception:  # noqa: BLE001 — best-effort bookkeeping, logged below
            _log.exception(
                "provisioning audit_id=%s: failed to flip 'partial' after the real "
                "provisioning error above — audit row stays 'pending'",
                audit_id,
            )
        refusal = _licences_ownership_refusal(exc)
        if refusal is not None:
            raise refusal from exc
        raise

    # The local `tenants` + `tenant_invites` rows above were only FLUSHed. Commit
    # them BEFORE the audit row is flipped to `success` and before the one-time
    # credentials are handed out: the Keycloak user, its password and the minted
    # licenses are already committed in EXTERNAL systems and are explicitly not
    # rolled back, so a teardown-commit failure would record success for a tenant
    # `require_known_tenant` has never heard of — with license tokens that exist
    # nowhere else and an invite that was never stored. Deliberately NOT on the `except` → `partial` path: a partial
    # provision must leave no local registry row (the audit row is the operator's
    # trail), and that path is a rollback path.
    await session.commit()
    # A failed LLM-access posture write leaves the tenant on licences'
    # byo_required default (every run blocked) even though everything else was
    # provisioned. The audit row must NOT read plain `success` then: `partial`
    # (the only other terminal status ck_provisioning_audit_status allows) with
    # the reason in `error`, and the full result so an operator can retry just
    # that step (PUT /api/admin/tenants/{id}/llm-access). The response stays 201.
    llm_access_failed = llm_access.status != "ok"
    await prov.finish(
        audit_id,
        status="partial" if llm_access_failed else "success",
        error=(
            f"llm_access {llm_access.mode!r} not set: {llm_access.detail}"
            if llm_access_failed
            else None
        ),
        result={
            "user_id": str(user_uuid) if user_uuid is not None else None,
            "tenant_id": str(tenant_id),
            "license_ids": license_ids,
            "invite_id": str(invite.invite_id) if invite.invite_id else None,
            "llm_access": llm_access.as_record(),
        },
        target_subject_id=user_uuid,
        tenant_id=tenant_id,
    )
    return AdminProvisionResult(
        tenant_id=tenant_id,
        invite_id=invite.invite_id,
        tokens=tokens,
        license_ids=license_ids,
        llm_access=AdminProvisionLlmAccess(
            mode=llm_access.mode, status=llm_access.status, detail=llm_access.detail
        ),
    )
