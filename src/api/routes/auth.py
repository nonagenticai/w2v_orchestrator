"""Auth identity surface — GET /api/auth/me."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from src.api.routes._common import (
    W2V_APP_KEY,
    LicensesClientDep,
    MembersRepoDep,
    OptionalTenantIdDep,
    SessionDep,
    SubjectIdDep,
    TenantIdDep,
    membership_resolved,
)
from src.api.routes.tenancy.me import memberships_and_pending_invites
from src.api.schemas import AuthMe, LicenseStatus
from src.config import get_settings
from src.services.license_provisioning import ensure_active_license
from src.services.license_status import get_license_status, invalidate_license_status

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Separate /api/license surface — mounted WITHOUT the tenant gate so an
# unentitled / unknown-tenant user can still read their own license status and
# render the renewal block instead of a hard 403.
license_router = APIRouter(prefix="/api/license", tags=["license"])

_log = logging.getLogger("w2v_orchestrator.auth")


@license_router.get("/status", response_model=LicenseStatus, operation_id="license.status")
async def license_status(
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    licenses: LicensesClientDep,
) -> LicenseStatus:
    """Rich license status for the SPA gate (mirrors fr's /api/license/status).

    The per-request license-token auth is the real enforcer; this is the
    fail-soft status hint the frontend polls to render active / expiring /
    blocked. Ships behind ``W2V_LICENSE_GATE_ENABLED`` (default OFF → always
    ``active``)."""
    settings = get_settings()
    gate_enabled = settings.license_gate_enabled
    tenant_str = str(tenant_id) if tenant_id else None
    subject_str = str(subject_id) if subject_id else None
    result = await get_license_status(
        client=licenses,
        app_key=W2V_APP_KEY,
        tenant_id=tenant_str,
        subject_id=subject_str,
        gate_enabled=gate_enabled,
    )

    # Self-serve lazy auto-issue: a portal tenant that self-registered but was
    # never licensed resolves state='none'. With the flag on (and the gate on),
    # mint a license on the spot and re-resolve so the SPA sees 'active' instead
    # of a hard block. Only on 'none' — never resurrect expired/revoked. Fail-soft:
    # a licenses blip must never 500 the status poll (the next poll retries).
    if (
        gate_enabled
        and settings.license_autoissue_enabled
        and result.state == "none"
        and tenant_str
        and subject_str
    ):
        try:
            # Tenant-level licence, held by ``svc:<tenant>:w2v`` — never by the
            # member who happened to poll first.
            await ensure_active_license(licenses, app_key=W2V_APP_KEY, tenant_id=tenant_str)
            # The pre-mint resolve cached 'none'; drop it so the re-resolve below
            # reads the freshly-minted license instead of the stale block.
            invalidate_license_status(
                tenant_id=tenant_str, app_key=W2V_APP_KEY, subject_id=subject_str
            )
            result = await get_license_status(
                client=licenses,
                app_key=W2V_APP_KEY,
                tenant_id=tenant_str,
                subject_id=subject_str,
                gate_enabled=gate_enabled,
            )
        except Exception:  # noqa: BLE001 — auto-issue is best-effort; status must not 500
            _log.warning(
                "license auto-issue failed for tenant=%s subject=%s", tenant_str, subject_str
            )

    return LicenseStatus(
        app_key=W2V_APP_KEY,
        gate_enabled=gate_enabled,
        has_license=result.has_license,
        state=result.state,
        expires_at=result.expires_at,
        days_remaining=result.days_remaining,
        plan_code=result.plan_code,
    )


@router.get("/me", response_model=AuthMe, operation_id="auth.me")
async def me(
    request: Request,
    tenant_id: OptionalTenantIdDep,
    subject_id: SubjectIdDep,
    members: MembersRepoDep,
    session: SessionDep,
) -> AuthMe:
    identity = AuthMe(
        tenant_id=tenant_id,
        subject_id=subject_id,
        email=getattr(request.state, "email", None),
        name=getattr(request.state, "name", None),
        roles=list(getattr(request.state, "roles", []) or []),
        role=getattr(request.state, "membership_role", None),
        active_tenant_refused=getattr(request.state, "active_tenant_refused", None),
    )
    # P2 WP4a: list the memberships and the pending invites for the caller's
    # VERIFIED email. Read-only — /me never accepts an invite (the caller does,
    # explicitly). Never fatal to /me (roll back so the session stays clean).
    try:
        identity.memberships, identity.pending_invites = await memberships_and_pending_invites(
            session, request, subject_id
        )
    except Exception:  # noqa: BLE001 — tenancy extras are a side effect, never fatal to /me
        await session.rollback()
        _log.warning(
            "membership / pending-invite listing failed for subject=%s", subject_id, exc_info=True
        )
    if tenant_id is None:
        # Membership was consulted (membership mode, or a claimless token under
        # shadow) and the caller has none: authenticated, no workspace yet.
        # Nothing to record.
        return identity
    # Roster write (Phase C, C-1; P2 WP3). /me is called once per SPA bootstrap,
    # not on the per-request hot path.
    #   * A visit to the caller's OWN claim tenant (claim/shadow mode) may create
    #     the row — that is how the pre-cutover roster fills.
    #   * A super_admin scope-switch, and EVERY visit in membership mode, only
    #     refreshes an existing active row: a visit is not a membership (WP1
    #     review — a scope-switch used to mint an active `member` row in every
    #     tenant a super_admin looked at). In membership mode the bump of
    #     last_seen_at is what makes "most recently used" pick this tenant next.
    #   * A MEMBERSHIP-RESOLVED request (membership mode, or a claimless token
    #     under shadow) never creates: request.state.tenant_id is then
    #     membership's answer, not a claim — "own claim tenant" would be true by
    #     construction and a claimless super_admin's scope-switch visit would
    #     mint a membership.
    # The auth identity is the contract — a roster write must NEVER break /me, so
    # swallow any failure (and roll back so get_db_session's commit stays clean).
    claim = getattr(request.state, "tenant_id", None)
    own_claim_tenant = claim is not None and str(claim) == str(tenant_id)
    create = (
        get_settings().w2v_tenancy_source != "membership"
        and not membership_resolved(request)
        and own_claim_tenant
    )
    try:
        await members.record_seen(
            subject_id=subject_id,
            tenant_id=tenant_id,
            email=identity.email,
            display_name=identity.name,
            roles=identity.roles,
            create=create,
        )
    except Exception:  # noqa: BLE001 — roster is a side effect, never fatal to /me
        await session.rollback()
        _log.warning("roster upsert failed for subject=%s tenant=%s", subject_id, tenant_id)
    return identity
