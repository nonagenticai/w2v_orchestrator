"""Shared FastAPI dependencies — strict UUID auth + repo factories."""

import logging
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from cachetools import TTLCache
from fastapi import Depends, HTTPException, Request, params, status
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.auth.identity_context import get_subject_id_from_context
from w2v_common.discovery import service_url

from src.auth.membership import RESOLVED_BY_MEMBERSHIP, RESOLVED_NO_MEMBERSHIP
from src.auth.tenancy_middleware import ACTOR_TYPE_API_KEY
from src.clients.billing_client import BillingServiceClient
from src.clients.identity_inviter import IdentityInviter, get_identity_inviter
from src.clients.keycloak_admin import KeycloakAdminClient, get_keycloak_admin
from src.clients.licenses_client import LicensesClient
from src.clients.service_auth import get_service_authenticator
from src.config import get_settings
from src.database.repositories.admin_audit_repository import AuditRepository
from src.database.repositories.admin_repository import AdminRepository
from src.database.repositories.deliverable_repository import DeliverableRepository
from src.database.repositories.members_repository import MembersRepository
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.provisioning_repository import ProvisioningRepository
from src.database.repositories.request_audit_repository import RequestAuditRepository
from src.database.repositories.run_repository import RunRepository
from src.database.repositories.tenant_licence_repository import TenantLicenceRepository
from src.database.session import bind_session_tenant
from src.observability.metrics import api_key_unattributable_total
from src.services.license_status import (
    _CACHE_MAXSIZE,
    _STALE_TTL_SECONDS,
    _STATUS_TTL_SECONDS,
    ENTITLED_STATES,
    NOT_ONBOARDED,
    LicenseStatusResult,
    derive_active_state,
    get_license_status,
)
from src.services.providers.crypto import Keyring, get_keyring
from src.services.run_usage_reconciler import RunUsageReconciler
from src.services.tenant_licence import (
    NOT_ONBOARDED_STATE,
    Entitlement,
    LicencesRefusedError,
    LicencesTenantNotOnboardedError,
    LicencesUnavailableError,
    LicenceTokenUnreadableError,
    tenant_entitlement,
)

# Reuses the audit logger name the AuditMiddleware uses, so a scope-switch lands
# in the same security-audit stream (JSON-formatted in prod). The durable
# per-request audit table (request_audit, migration 0026) is written off the hot
# path by AuditMiddleware; this log line stays for the prod log stream.
_audit = logging.getLogger("w2v_orchestrator.audit")
logger = logging.getLogger(__name__)

# Super-admin scope-switch: a super_admin may act INTO another tenant by sending
# this header (matched case-insensitively by Starlette). Non-super callers are
# silently ignored (probe-resistant); malformed value from a super_admin → 422.
ACTIVE_TENANT_HEADER = "x-active-tenant-id"

#: License ``apps.key`` gated for the w2v product. Single source of truth shared
#: by the status endpoint (``routes/auth.py``) and the enforcement gate below.
W2V_APP_KEY = "w2v"

#: License states that are an AUTHORITATIVE refusal: we reached the licenses
#: service, it answered, and the answer was "not entitled". These earn a 402.
_LICENSE_BLOCKING_STATES: frozenset[str] = frozenset({"none", "expired", "revoked"})

#: Error code for an INDETERMINATE verdict (``unknown``, or any state a future
#: licenses release introduces that this build does not recognise). Distinct
#: from every ``license_<state>`` 402 code on purpose — see below.
LICENSE_UNAVAILABLE_ERROR = "license_unavailable"


def _enforce_license_state(result: LicenseStatusResult) -> None:
    """Deny unless the derived state is POSITIVELY entitled. Single chokepoint
    shared by the request-local fast path and the upstream-lookup path, so the
    two can never disagree about what counts as entitled.

    Fail-CLOSED, and with the two refusals kept distinguishable:

    * entitled (``active`` / ``expiring_soon``) → pass.
    * an authoritative refusal (``none`` / ``expired`` / ``revoked``) → **402**
      ``license_<state>``. 402 Payment Required is a statement about the
      TENANT's licence: buy or renew one and the request succeeds. It is
      terminal for the caller — retrying changes nothing.
    * anything else — ``unknown`` from an upstream failure, or a state this
      build does not recognise — → **503** ``license_unavailable`` with
      ``Retry-After``. We are not saying the tenant is unlicensed (we do not
      know); we are saying WE could not check. 503 is the honest code: it is
      server-side, transient, and retryable, and it keeps a monitored
      licences-outage signal from hiding inside the 402 counter. Emitting 402
      here would tell an entitled customer they have no licence and would push
      the SPA into a renewal block over our own outage.

    This is an ALLOWLIST on purpose. The previous blocklist admitted every state
    it had not been taught to refuse, which is how an introspection outage became
    a fleet-wide ALLOW. Availability is bought by the layers in FRONT of this
    deny (the request-local fast path, the 60s positive cache and the 900s
    stale-while-error window in ``license_status``), never by admitting traffic
    whose entitlement was never established.
    """
    if result.state in ENTITLED_STATES:
        return
    if result.state == NOT_ONBOARDED:
        # licences 403 ``tenant_not_accessible``: a definite answer about the
        # TENANT (not onboarded to w2v) — 403 ``TENANT_NOT_ONBOARDED``, neither
        # "buy a licence" (402) nor "we could not check" (503).
        raise _tenant_not_onboarded()
    if result.state in _LICENSE_BLOCKING_STATES:
        raise HTTPException(
            status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error": f"license_{result.state}",
                "message": "This tenant has no active Studio license.",
            },
        )
    raise HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": LICENSE_UNAVAILABLE_ERROR,
            "message": (
                "Could not verify this tenant's Studio license. This is not a "
                "statement about your entitlement — retry shortly."
            ),
            "state": result.state,
        },
        headers={"Retry-After": "30"},
    )


def _tenant_not_onboarded() -> Exception:
    """The 403 ``TENANT_NOT_ONBOARDED`` TenancyError. Imported lazily: the tenancy
    package's ``__init__`` imports route modules that import THIS module."""
    from src.api.routes.tenancy.errors import tenant_not_onboarded

    return tenant_not_onboarded()


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    factory = request.app.state.session_factory
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def _claim_uuid(request: Request, claim: str) -> UUID:
    val = getattr(request.state, claim, None)
    if val is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"missing {claim} claim")
    try:
        return UUID(str(val))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, f"malformed {claim} claim: not a UUID"
        ) from exc


def resolve_active_tenant(claim: UUID, override: str | None, roles: set[str]) -> tuple[UUID, bool]:
    """Resolve the effective tenant from the claim + an optional scope-switch.

    ``override`` is the raw value from whichever channel carried it — the
    ``X-Active-Tenant-Id`` header or the ``active_tenant`` query param; the two
    are equivalent and are read together by the callers (``get_tenant_id`` for
    requests, ``AuditMiddleware`` for the audit write).

    Returns ``(effective_tenant, switched)``. **Tolerant**: a non-super caller, an
    absent override, or a malformed one all yield ``(claim, False)``. The strict
    422-on-malformed for a super_admin lives in ``get_tenant_id`` (the request
    dep); ``AuditMiddleware`` uses this tolerant form so a bad override never
    breaks the fail-safe audit write. Single source of the scope-switch rule so
    the dep and the middleware can never drift.

    `SUPER_ADMIN` is defined lower in this module; resolved at call time.
    """
    if not override or SUPER_ADMIN not in roles:
        return claim, False
    try:
        effective = UUID(override)
    except TypeError, ValueError:
        return claim, False
    return effective, effective != claim


def get_tenant_id(request: Request) -> UUID:
    """Resolve the EFFECTIVE tenant for this request — the ONE chokepoint.

    Normally the caller's ``tenant_id`` claim. A ``super_admin`` may override it
    to act into another tenant, and the override arrives on one of TWO channels:

    * ``X-Active-Tenant-Id: <uuid>`` — every request the SPA makes via ``fetch()``.
    * ``?active_tenant=<uuid>`` — the only channel available to requests the
      BROWSER issues itself: ``EventSource`` streams and markup-driven loads
      (``<img>``/``<video>``/``<audio>``/``<a href download>``/``<iframe>``).
      None of those can set a request header at all.

    The header wins when both are present. Non-super callers' override is
    silently ignored on both channels (probe-resistant); a super_admin's
    malformed override is a strict 422. Strictly opt-in: with neither channel
    present the behaviour is identical to the claim.

    WHY BOTH CHANNELS RESOLVE HERE, IN ONE FUNCTION — do not re-split this.
    The tenant-scoped data routers are mounted in ``asgi.py`` with ROUTER-LEVEL
    dependencies (``require_known_tenant`` → ``apply_tenant_rls`` → …) that take
    ``TenantIdDep`` too. When the query-param channel lived in a separate dep
    that only the handlers adopted (the deleted ``get_query_or_header_tenant_id``
    / ``BrowserTenantIdDep``, PR #762), a ``?active_tenant=`` switch was seen by
    the HANDLER but not by the gates: ``apply_tenant_rls`` bound the Postgres
    ``app.current_tenant`` GUC to the admin's OWN claim, so the handler's
    ownership SELECT ran under RLS scoped to the wrong tenant, found no row, and
    404'd every caption/artifact/redesign download for a scope-switched admin.
    Header requests worked; query-param requests silently didn't. Any new
    consumer — handler, router-level gate, middleware — must go through this
    function so the two channels can never resolve to different tenants again.
    (``AuditMiddleware`` shares the rule via ``resolve_active_tenant``.)

    `SUPER_ADMIN`/`_roles` are defined lower in this module; Python resolves the
    names at call time, so the ordering is fine.
    """
    if membership_resolved(request):
        # W2V_TENANCY_SOURCE=membership: TenancyMiddleware already applied BOTH
        # channels (membership switch, super_admin any-tenant, audited) and
        # bound the result. Re-applying the scope-switch here would be a second,
        # weaker resolution of the same request.
        return _claim_uuid(request, "tenant_id")
    active = request.headers.get(ACTIVE_TENANT_HEADER) or request.query_params.get("active_tenant")
    claim = _claim_uuid(request, "tenant_id")  # 401 if missing/malformed claim
    roles = _roles(request)
    # Strict: a super_admin's malformed override is a 422 here (the dep rejects it;
    # resolve_active_tenant tolerantly ignores it for the middleware). Non-super
    # callers' override is silently ignored (probe-resistant).
    if active and SUPER_ADMIN in roles:
        try:
            UUID(active)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "malformed active tenant: not a UUID",
            ) from exc
    effective, switched = resolve_active_tenant(claim, active, roles)
    if switched:
        # Audit every effective cross-tenant request (not just the UI toggle).
        subject = getattr(request.state, "subject_id", None) or get_subject_id_from_context()
        _audit.info(
            "SCOPE_SWITCH: %s",
            {
                "subject_id": str(subject) if subject else "unknown",
                "from_tenant": str(claim),
                "to_tenant": str(effective),
                "path": request.url.path,
            },
        )
    return effective


def membership_resolved(request: Request) -> bool:
    """True when ``TenancyMiddleware`` (``W2V_TENANCY_SOURCE=membership``, or a
    claimless token under ``shadow``) resolved this request's tenant from an ACTIVE membership in a non-suspended
    tenant — or a super_admin's audited, switchable choice. Its roles on
    ``request.state.roles`` are then the MEMBERSHIP role (+ realm super_admin),
    not the realm-wide tenant_admin/member roles."""
    return getattr(request.state, "tenant_resolved_by", None) == RESOLVED_BY_MEMBERSHIP


def get_optional_tenant_id(request: Request) -> UUID | None:
    """``get_tenant_id``, except a caller whose membership resolution found NO
    usable membership yields ``None`` instead of a 401.

    Keyed on what ``TenancyMiddleware`` decided for THIS request
    (``tenant_resolved_by == RESOLVED_NO_MEMBERSHIP``), never on
    ``W2V_TENANCY_SOURCE``: in ``shadow`` mode a claimless token is resolved by
    membership exactly as in ``membership`` mode, and a mode check here 401'd it
    (``missing tenant_id claim``) on ``/api/auth/me`` — live 2026-10-15.

    Only for the routes ``TenancyMiddleware`` lets through tenant-less
    (``NO_MEMBERSHIP_EXEMPT_PATHS``: ``/api/auth/me``) — the SPA must still
    learn who the caller is to render its no-workspace screen. Every other
    route uses ``TenantIdDep`` and never sees such a request (403 upstream).
    """
    if (
        getattr(request.state, "tenant_resolved_by", None) == RESOLVED_NO_MEMBERSHIP
        and getattr(request.state, "tenant_id", None) is None
    ):
        return None
    return get_tenant_id(request)


def get_subject_id(request: Request) -> UUID:
    # AuthMiddleware (w2v_common) stores subject_id in a ContextVar via
    # set_identity_context, NOT on request.state. State-first lookup keeps
    # the test conftest's _SeedAuthState path working; ContextVar fallback
    # fires for real traffic (license + JWT paths both populate it).
    val = getattr(request.state, "subject_id", None)
    if val is None:
        val = get_subject_id_from_context() or None
    if val is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing subject_id claim")
    try:
        return UUID(str(val))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "malformed subject_id claim: not a UUID"
        ) from exc


#: The ONLY effective-tenant dep. Honours both scope-switch channels (header and
#: ``?active_tenant=``) — see get_tenant_id for why they must not be split again.
TenantIdDep = Annotated[UUID, Depends(get_tenant_id)]
OptionalTenantIdDep = Annotated[UUID | None, Depends(get_optional_tenant_id)]
SubjectIdDep = Annotated[UUID, Depends(get_subject_id)]


def get_project_repo(session: SessionDep) -> ProjectRepository:
    return ProjectRepository(session)


def get_deliverable_repo(session: SessionDep) -> DeliverableRepository:
    return DeliverableRepository(session)


def get_run_repo(session: SessionDep) -> RunRepository:
    return RunRepository(session)


def get_admin_repo(session: SessionDep) -> AdminRepository:
    return AdminRepository(session)


def get_audit_repo(session: SessionDep) -> AuditRepository:
    # Shares the request session with the route's other repos (FastAPI dedups
    # Depends(get_db_session)), so an audit row commits atomically with the
    # action it records. NOTE: the per-request X-Active-Tenant-Id scope-switch
    # (resolved in get_tenant_id) is intentionally NOT persisted here — only the
    # deliberate UI switch is recorded (POST /api/admin/scope-switch). Raw-API
    # switches stay log-line-only this increment (no DB write on the hot path).
    return AuditRepository(session)


def get_members_repo(session: SessionDep) -> MembersRepository:
    return MembersRepository(session)


def get_request_audit_repo(session: SessionDep) -> RequestAuditRepository:
    # Read-side only (GET /api/admin/request-audit). The WRITE side is the
    # AuditMiddleware on its own fresh session off the hot path — never via DI.
    return RequestAuditRepository(session)


# ─── Provisioning clients (admin tier Phase C-2) ─────────────────────────
# The identity-invite + licenses clients authorize a Keycloak client-credentials
# Bearer (the w2v-backend SA) via a module-level ServiceAuthenticator whose token
# cache survives across requests. The client objects themselves are cheap to
# build per request. Tests override these factories with fakes (no live calls).
# w2v holds NO Keycloak user-write power: provisioning creates/emails users.
def get_inviter() -> IdentityInviter:
    return get_identity_inviter()


def get_signup_keycloak_admin() -> KeycloakAdminClient:
    """The Keycloak admin client the early-access approve mints its identity with
    (``src/services/signup/identity.py``). Tests override."""
    return get_keycloak_admin()


def get_licenses_client() -> LicensesClient:
    return LicensesClient(
        authenticator=get_service_authenticator(),
        base_url=service_url("licenses"),
        ensure_timeout_seconds=get_settings().licences_ensure_timeout_seconds,
    )


def get_licence_keyring() -> Keyring:
    """The keyring that seals tenant licence tokens (the provider keyring; the
    AAD ``("tenant-licence", tenant_id)`` separates the two uses). Tests override."""
    return get_keyring()


LicenceKeyringDep = Annotated[Keyring, Depends(get_licence_keyring)]


def get_provisioning_repo(request: Request) -> ProvisioningRepository:
    # Its own session factory (NOT the request session): the pending audit row is
    # committed before the side effects so a mid-flight crash leaves a trail.
    return ProvisioningRepository(request.app.state.session_factory)


def get_billing_client() -> BillingServiceClient:
    # Read client for billing's per-correlation priced cost. base URL resolves via
    # discovery inside the client; tests override this factory with a fake.
    return BillingServiceClient()


BillingClientDep = Annotated[BillingServiceClient, Depends(get_billing_client)]


def get_run_usage_reconciler(
    session: SessionDep,
    billing_client: BillingClientDep,
) -> RunUsageReconciler:
    # Mirrors billing-priced cost into run_usage on read (see RunUsageReconciler).
    return RunUsageReconciler(session, billing_client)


RunUsageReconcilerDep = Annotated[RunUsageReconciler, Depends(get_run_usage_reconciler)]


ProjectRepoDep = Annotated[ProjectRepository, Depends(get_project_repo)]
DeliverableRepoDep = Annotated[DeliverableRepository, Depends(get_deliverable_repo)]
RunRepoDep = Annotated[RunRepository, Depends(get_run_repo)]
AdminRepoDep = Annotated[AdminRepository, Depends(get_admin_repo)]
AuditRepoDep = Annotated[AuditRepository, Depends(get_audit_repo)]
RequestAuditRepoDep = Annotated[RequestAuditRepository, Depends(get_request_audit_repo)]
MembersRepoDep = Annotated[MembersRepository, Depends(get_members_repo)]
IdentityInviterDep = Annotated[IdentityInviter, Depends(get_inviter)]
SignupKeycloakAdminDep = Annotated[KeycloakAdminClient, Depends(get_signup_keycloak_admin)]
LicensesClientDep = Annotated[LicensesClient, Depends(get_licenses_client)]
ProvisioningRepoDep = Annotated[ProvisioningRepository, Depends(get_provisioning_repo)]


# ─── Role guards (admin tier, Phase B) ──────────────────────────────────
# Portal-realm roles: super_admin > tenant_admin > member. super_admin is a
# Keycloak composite OF tenant_admin (the JWT normally lists both), but the
# guards treat super_admin as *implying* tenant_admin defensively so a
# composite-only token still passes tenant-admin gates. Roles arrive on
# request.state.roles via w2v_common>=0.18.10 AuthMiddleware (JWT path);
# license/anon callers get [].
SUPER_ADMIN = "super_admin"
TENANT_ADMIN = "tenant_admin"


def _roles(request: Request) -> set[str]:
    return set(getattr(request.state, "roles", []) or [])


def require_super_admin(request: Request, tenant_id: TenantIdDep) -> None:
    # `tenant_id: TenantIdDep` is a sub-dependency: get_tenant_id runs FIRST and
    # raises 401 for an anon/missing-claim caller, short-circuiting before the
    # role check — 401-before-403 with no manual ordering.
    if SUPER_ADMIN not in _roles(request):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "super_admin role required")


async def require_tenant_admin(
    request: Request, tenant_id: TenantIdDep, repo: AdminRepoDep
) -> UUID:
    roles = _roles(request)
    if SUPER_ADMIN not in roles and TENANT_ADMIN not in roles:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "tenant_admin role required")
    # `tenant_id` already carries the X-Active-Tenant-Id scope-switch resolution
    # from get_tenant_id (a super_admin acting into another tenant). Fail-closed
    # ONLY on a real cross-tenant switch (effective != claim): a super_admin may
    # not act into a tenant that is unknown (no registry row, no projects) or
    # suspended. A user's OWN claim is left fail-open exactly as before — the
    # every-request existence posture waits on Phase C onboarding (no tenant row
    # is created at provisioning yet, so enforcing it would lock out a brand-new
    # licensed tenant with zero projects). `repo` shares the request session with
    # the route's own AdminRepoDep (FastAPI dedups), so no extra session/round-
    # trip on the common no-switch path (the DB check below is skipped).
    if membership_resolved(request):
        # Membership mode (closes B21): `roles` above IS the caller's membership
        # role in THIS tenant (+ realm super_admin) — a realm-wide tenant_admin
        # grant no longer counts — and the tenant was validated as an active
        # membership / switchable super_admin target by TenancyMiddleware.
        return tenant_id
    claim = _claim_uuid(request, "tenant_id")
    if tenant_id != claim and not await repo.tenant_switchable(tenant_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "tenant not provisioned")
    return tenant_id


def require_any_admin(request: Request, tenant_id: TenantIdDep) -> None:
    roles = _roles(request)
    if SUPER_ADMIN not in roles and TENANT_ADMIN not in roles:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin role required")


SuperAdminDep = Annotated[None, Depends(require_super_admin)]
TenantAdminDep = Annotated[UUID, Depends(require_tenant_admin)]
AnyAdminDep = Annotated[None, Depends(require_any_admin)]


# ─── Fail-closed tenant-existence (admin tier Phase C-3) ─────────────────
# Every tenant-scoped data route validates that the caller's EFFECTIVE tenant
# (claim, or a super-admin X-Active-Tenant-Id switch) is *known* — a `tenants`
# registry row OR ≥1 project, and not suspended (exactly AdminRepository.
# tenant_switchable) OR, with no registry row, holds an entitled TENANT-LEVEL w2v
# licence (see _tenant_licence_state). Cluster-standard provisioning
# (licenses `provision_tenant_bundle.py`) writes ONLY that licence — nothing in
# w2v — so admission is licence-based, same as fr (require-fr-license.ts). Only
# the super-admin POST /api/admin/provision (C-2) or a PATCH writes a registry
# row; the request path never does. Mode (off|warn|enforce, default
# enforce) is a gitops-ConfigMap dial-back valve. Positive results are cached
# (the handful of live tenants stay hot); an UNKNOWN tenant is never cached, so a
# just-provisioned tenant is recognised on its very next request.
_KNOWN_TENANT_TTL_SECONDS = 60.0
_known_tenant_cache: dict[UUID, float] = {}


def _tenant_known_cached(tenant_id: UUID) -> bool:
    expiry = _known_tenant_cache.get(tenant_id)
    if expiry is None:
        return False
    if time.monotonic() >= expiry:
        _known_tenant_cache.pop(tenant_id, None)
        return False
    return True


async def _tenant_licence_state(
    tenant_id: UUID, repo: AdminRepository, licenses: LicensesClient
) -> str | None:
    """Admit an unregistered tenant on its TENANT-LEVEL w2v licence (fr parity).

    Mirrors fr ``requireFrLicense``: ``GET /licenses/active?tenant_id=&app_key=``
    with NO subject — the licence belongs to the tenant, not to whoever is calling.
    Nothing is written: no local registry row on the request path.

    A registry row, when present, stays authoritative — it is w2v's explicit
    super-admin annotation, and ``suspended`` must still deny whatever the licence
    says. Returns the licence state (None when a registry row decided: the
    tenant is suspended). Only a POSITIVE verdict (``ENTITLED_STATES``) admits;
    ``none`` / ``expired`` / ``revoked`` / ``not_onboarded`` / ``unknown`` (an
    upstream error folds into ``unknown``) keep the caller's fail-closed behaviour.
    """
    if await repo.get_tenant(tenant_id) is not None:
        # A row exists yet tenant_switchable said no ⇒ suspended. Skip the lookup.
        return None
    # gate_enabled=True regardless of W2V_LICENSE_GATE_ENABLED: this is an
    # admission verdict, and the gate-off "everyone is active" answer must never
    # admit an arbitrary tenant.
    result = await get_license_status(
        client=licenses,
        app_key=W2V_APP_KEY,
        tenant_id=str(tenant_id),
        subject_id=None,
        gate_enabled=True,
    )
    return result.state


async def require_known_tenant(
    request: Request, tenant_id: TenantIdDep, repo: AdminRepoDep, licenses: LicensesClientDep
) -> None:
    """Fail-closed gate: serve only requests whose effective tenant is known.

    Applied as a router-level dependency to tenant-scoped data routers (see
    ``asgi.py``). ``tenant_id`` is the effective tenant from ``get_tenant_id``
    (claim, or a scope-switch on EITHER channel — header or ``?active_tenant=``),
    so this validates the tenant the caller actually operates in. Resolving it
    through the same single dep as the handlers is load-bearing: a gate that saw
    only one channel would admit/deny a different tenant than the handler serves
    (see get_tenant_id). ``off`` short-circuits before any DB work."""
    mode = get_settings().tenant_enforcement_mode
    if mode == "off":
        return
    # S95: a tenant asserted via X-Effective-Tenant-Id by a subject already in
    # w2v_common's AUTH_DELEGATING_SUBJECTS is "known" by construction —
    # delegation implies provisioning. This is how marketing's cost-attribution
    # fix (sending the REAL end tenant, not the service-account tenant) stopped
    # 403ing: the real tenant legitimately has no `tenants` registry row or
    # `projects` row yet (AdminRepository.tenant_switchable requires one of
    # those), which is exactly the same shape as "genuinely unprovisioned" —
    # ≥37 of 39 licensed tenants are in this position. w2v_common's
    # `_apply_effective_tenant` stamps this marker ONLY on an accepted
    # delegated re-scope (allowlisted subject, header present, tenant differs)
    # — see its docstring for the full provenance argument. `getattr` with a
    # default is deliberate: an orchestrator running against an OLDER
    # w2v_common (pre-S95) has no such attribute, and this gate must degrade
    # to the pre-existing fail-closed behaviour, never raise.
    #
    # ⛔ MUST return here, before `_tenant_known_cached` / the cache write
    # below. `_known_tenant_cache` is a process-global dict keyed ONLY by
    # tenant_id, with no notion of caller or provenance. If this delegated
    # path wrote tenant_id into that cache, a LATER, non-delegated request for
    # the same still-unprovisioned tenant (e.g. a spoofed or misrouted direct
    # customer-token call) would be waved through by `_tenant_known_cached`
    # with no re-check — a cache-poisoning privilege leak. So this branch
    # must be a pure pass-through with no cache side effect whatsoever.
    if getattr(request.state, "tenant_delegated", False):
        return
    # P2 membership mode (closes B21): "known" = an ACTIVE membership in a
    # REGISTERED, non-suspended tenant (or a super_admin's switchable choice), which
    # TenancyMiddleware established for exactly this tenant on this request.
    # No cache write: membership is per caller, the cache is per tenant.
    if membership_resolved(request):
        return
    if _tenant_known_cached(tenant_id):
        return
    if await repo.tenant_switchable(tenant_id):
        _known_tenant_cache[tenant_id] = time.monotonic() + _KNOWN_TENANT_TTL_SECONDS
        return
    # Not known locally — but a tenant-level w2v licence IS provisioning. Caching
    # is safe here: this is a non-delegated request (the delegated branch returned
    # above) and the tenant carried a positive licence verdict of its own.
    licence_state = await _tenant_licence_state(tenant_id, repo, licenses)
    if licence_state in ENTITLED_STATES:
        _known_tenant_cache[tenant_id] = time.monotonic() + _KNOWN_TENANT_TTL_SECONDS
        return
    # Not a known tenant.
    if mode == "enforce":
        if licence_state == NOT_ONBOARDED:
            # licences 403 ``tenant_not_accessible``: say WHY, distinctly.
            raise _tenant_not_onboarded()
        raise HTTPException(status.HTTP_403_FORBIDDEN, "tenant not provisioned")
    # warn: allow but record the would-be-block for blast-radius telemetry.
    subject = getattr(request.state, "subject_id", None) or get_subject_id_from_context()
    _audit.warning(
        "TENANT_UNKNOWN: %s",
        {
            "subject_id": str(subject) if subject else "unknown",
            "tenant_id": str(tenant_id),
            "path": request.url.path,
        },
    )


KnownTenantDep = Annotated[None, Depends(require_known_tenant)]


# ─── Phase D RLS — bind app.current_tenant per request ───────────────────
async def apply_tenant_rls(tenant_id: TenantIdDep, session: SessionDep) -> None:
    """Bind ``app.current_tenant`` for the request transaction so the Postgres
    RLS policies on tenant-scoped tables (``projects`` directly, plus the
    run-tree / ``verbatim_items`` tables via join policies) filter by it.

    Applied as a router-level dependency AFTER ``require_known_tenant`` on the
    tenant-scoped data routers (see ``asgi.py``). ``tenant_id`` is the effective
    tenant from ``get_tenant_id`` (claim, or a super-admin scope-switch on EITHER
    channel — header or ``?active_tenant=``). ``session`` is the same
    request-cached ``get_db_session`` the handler queries on, so the binding
    lands in the right transaction.

    This is where a split resolution turns into a 404: bind the GUC to the
    admin's own claim while the handler queries as the switched tenant and RLS
    hides the row. Both must come from ``get_tenant_id`` — see its docstring.

    Gated by ``TENANT_RLS_MODE`` (off → no-op; enforce → bind). Shares the
    single-source ``bind_session_tenant`` helper with every background session
    (run dispatcher / engine / sub-DAG / reconciler) so the request and background
    paths bind the GUC identically. Dormant until a later increment adds the
    no-``BYPASSRLS`` runtime role and flips the flag (atomic with the role switch)."""
    await bind_session_tenant(session, tenant_id)


TenantRlsDep = Annotated[None, Depends(apply_tenant_rls)]


# ─── License enforcement (Gate 2) ────────────────────────────────────────
# Real server-side enforcement of the w2v product license, mirroring
# commonpractice's `assertTenantLicensed` (backend/src/middleware/license-gate.ts)
# but as a FastAPI router-level dependency. WHY: w2v_common AuthMiddleware admits
# a Keycloak-JWT-only caller (no/expired license token → request.state.license =
# None) and proceeds; the gate flag previously only changed what
# /api/license/status REPORTS, so a JWT user with no w2v license was never
# blocked. This closes that hole on the tenant-scoped data routers.
#
# Behaviour:
#   * gate ON (W2V_LICENSE_GATE_ENABLED unset or true — the DEFAULT) → only a
#     positively-entitled state passes. 402 license_<state> for an authoritative
#     {none, expired, revoked}; 503 license_unavailable when entitlement could
#     not be established at all. FAIL-CLOSED (owner ruling: unsanctioned usage
#     is blocked), matching commonpractice's Gate 2. The cost of an outage is
#     bounded by the three layers in front of the deny: the request-local fast
#     path below (no hop at all), the 60s positive cache, and the 900s
#     stale-while-error window that serves last-known-good.
#   * gate OFF (W2V_LICENSE_GATE_ENABLED=false) → no-op, the break-glass switch:
#     require_license returns before any lookup. Flip it via the gitops
#     ConfigMap, no rebuild — e.g. if a tenant-provisioning gap starts 402ing
#     real customers.
#
# Resolution order:
#   1. `license_state_from_request` — the inbound license token was ALREADY
#      introspected by AuthMiddleware, so when it proves both the app and the
#      tenant we derive the state locally and skip the HTTP hop entirely.
#   2. `get_license_status` — the upstream lookup (cached 60s, single-flighted).
#      Note its /licenses/active route 403s for this service account today; it
#      falls back to the subject inventory and logs `license_gate_misconfigured`.
#
# Mounted alongside `_tenant_gate` on the data routers (see asgi.py). EXEMPT (not
# mounted): health, auth (/api/auth/me must load identity), license
# (/api/license/status must render the block screen), blueprints, admin
# (onboarding/super-admin act on still-unlicensed tenants), and the SSE routers.
def license_state_from_request(request: Request, tenant_id: UUID) -> LicenseStatusResult | None:
    """Derive the gate state from the ALREADY-introspected inbound license token.

    The license-token lane of ``w2v_common.auth.middleware`` stashes the whole
    introspection response on ``request.state.license`` (middleware.py — it is
    the licenses service's ``TokenIntrospectResponse``, carrying ``active``,
    ``status``, ``expires_at``, ``tenant_id`` and an ``app`` blob whose ``key``
    comes straight from ``App.to_dict()``). When that is present the gate
    already has an authoritative answer and can skip the HTTP hop entirely.

    Returns ``None`` — meaning "fall through to the upstream lookup" — for the
    JWT-only lane, an inactive token, or any token we cannot POSITIVELY prove
    belongs to this tenant AND to w2v. Both guards are mandatory:

    * **app**: a license minted for a DIFFERENT product must never entitle w2v.
    * **tenant**: the effective tenant can be a super-admin ``X-Active-Tenant-Id``
      scope-switch, so the token's own tenant is not automatically the tenant
      being gated.

    Falling through is always safe (it costs a round-trip, never entitlement),
    so every ambiguous shape returns ``None`` rather than guessing.
    """
    detail = getattr(request.state, "license", None)
    if not isinstance(detail, dict) or not detail.get("active"):
        return None
    app = detail.get("app")
    if not isinstance(app, dict) or app.get("key") != W2V_APP_KEY:
        return None
    token_tenant = detail.get("tenant_id")
    if not token_tenant or str(token_tenant) != str(tenant_id):
        return None
    # Same pure derivation the /licenses/active 200 path uses, so the
    # expiring-soon window and the stale-200 hardening cannot drift.
    return derive_active_state(detail, datetime.now(UTC))


async def require_license(
    request: Request,
    tenant_id: TenantIdDep,
    licenses: LicensesClientDep,
    keyring: LicenceKeyringDep,
) -> None:
    settings = get_settings()
    if not settings.license_gate_enabled:
        return
    local = license_state_from_request(request, tenant_id)
    if local is not None:
        _enforce_license_state(local)
        return
    # Subject is optional here — it only refines a 404 into expired/revoked/none
    # from the inventory; resolve it tolerantly (never 401 from the gate itself).
    subject = getattr(request.state, "subject_id", None) or get_subject_id_from_context()
    result = await get_license_status(
        client=licenses,
        app_key=W2V_APP_KEY,
        tenant_id=str(tenant_id),
        subject_id=str(subject) if subject else None,
        gate_enabled=True,
    )
    _enforce_license_state(result)
    await _require_attributable_api_key(request, tenant_id, licenses, keyring)


#: API-key lane licence verification caches (P9-6), with the SAME shape and
#: constants as the licence gate (``license_status``): a fresh positive verdict
#: serves for ``_STATUS_TTL_SECONDS``; when licences are UNREACHABLE a
#: last-known-good verdict serves up to ``_STALE_TTL_SECONDS``. A CONFIRMED
#: refusal (revoked / expired / foreign / unknown licence) evicts both and is
#: refused at once. Bounded (``_CACHE_MAXSIZE`` entries, LRU-evicting TTLCache).
#: Key: (tenant, licence).
_api_key_licence_ok: TTLCache[tuple[str, str], bool] = TTLCache(
    maxsize=_CACHE_MAXSIZE, ttl=_STATUS_TTL_SECONDS
)
_api_key_licence_stale: TTLCache[tuple[str, str], bool] = TTLCache(
    maxsize=_CACHE_MAXSIZE, ttl=_STALE_TTL_SECONDS
)


def clear_api_key_licence_cache() -> None:
    """Test seam / safety valve (mirrors ``clear_license_status_cache``)."""
    _api_key_licence_ok.clear()
    _api_key_licence_stale.clear()


def _api_key_unattributable(message: str, state: str, reason: str) -> HTTPException:
    """503 ``license_unavailable``; ``reason`` labels
    ``api_key_unattributable_total`` (bounded: ``API_KEY_UNATTRIBUTABLE_REASONS``)."""
    api_key_unattributable_total.labels(reason=reason).inc()
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"error": LICENSE_UNAVAILABLE_ERROR, "message": message, "state": state},
        headers={"Retry-After": "60"},
    )


#: API-key lane refusal codes (402-style, the commonpractice split): ``unreadable``
#: needs an operator (the stored token will not open); ``unavailable`` is a
#: retryable could-not-check; ``inactive`` is a definite "this licence does not
#: entitle the tenant" (incl. another tenant's / app's token, a stale ref).
LICENCE_UNREADABLE = "licence_unreadable"
LICENCE_UNAVAILABLE = "licence_unavailable"
LICENCE_INACTIVE = "licence_inactive"


def _api_key_licence_refusal(
    code: str, message: str, state: str, *, retry: bool = False
) -> HTTPException:
    return HTTPException(
        status.HTTP_402_PAYMENT_REQUIRED,
        detail={"error": code, "message": message, "state": state},
        headers={"Retry-After": "60"} if retry else None,
    )


def _api_key_licence_refused(status_code: int) -> HTTPException:
    """licences / its gateway definitely refused (c344f9dc): 502, no Retry-After."""
    return HTTPException(
        status.HTTP_502_BAD_GATEWAY,
        detail={
            "error": "LICENCES_REFUSED",
            "message": (
                f"The licences service refused this service's lookup (HTTP {status_code}). "
                "A configuration fault of this w2v deployment; retrying will not help."
            ),
            "state": "refused",
        },
    )


#: ``_ref_verdict`` state: the tenant is entitled but ``/licenses/active`` cannot
#: single the ref out (several candidates, or the legacy 409 multi-active). Also
#: the 503's response ``state`` and its ``api_key_unattributable_total`` reason.
REF_AMBIGUOUS = "ref_ambiguous"


def _ref_verdict(ent: Entitlement, ref: UUID) -> str | None:
    """None when ``ent`` PROVES ``ref`` billable, else the refusal state.

    :data:`REF_AMBIGUOUS` when the fallback cannot single the ref out: the
    caller fails closed exactly like a tenant with no ref (P9-6) — an entitled
    tenant is not proof that THIS ref is live and the tenant's own."""
    if not ent.entitled:
        return ent.reason or ent.state
    if ent.source == "token":
        return None if ent.licence_ref == ref else "licence_ref_mismatch"
    # Fallback: the id is provably THE tenant's licence only with one candidate.
    if ent.licence_ref is None:
        return REF_AMBIGUOUS
    return None if ent.licence_ref == ref else "licence_ref_not_live"


def _ref_refusal(refusal: str) -> Exception:
    """The response for a :func:`_ref_verdict` refusal state."""
    if refusal == NOT_ONBOARDED_STATE:
        return _tenant_not_onboarded()
    if refusal == REF_AMBIGUOUS:
        return _api_key_unattributable(
            "This tenant holds several Studio licences and none is recorded as its "
            "own, so API-key usage cannot be billed. Store the tenant's licence "
            "token (PUT /api/tenants/{id}/licence-token).",
            REF_AMBIGUOUS,
            REF_AMBIGUOUS,
        )
    return _api_key_licence_refusal(
        LICENCE_INACTIVE,
        "This tenant's recorded Studio licence does not entitle it, so API-key usage "
        "cannot be billed.",
        refusal,
    )


async def _require_attributable_api_key(
    request: Request, tenant_id: UUID, licenses: LicensesClient, keyring: Keyring
) -> None:
    """P9-6: an ENTITLED tenant API key must also be BILLABLE — to a licence that
    is LIVE and the tenant's own.

    A key carries no licence token; ``TenancyMiddleware`` binds the tenant's
    ``licence_ref`` as the request's licence instead, and that id is what run
    snapshots and outbound ``X-License-Id`` carry. ``licence_ref`` is a stored
    pointer, so it is RE-VERIFIED here before any spend — by INTROSPECTING the
    tenant's stored licence token (active ∧ own tenant ∧ app w2v ∧
    ``license_id == licence_ref``), or, with no (unexpired) token stored, by
    ``GET /licenses/active`` (a single candidate must BE the ref). No ref → 503
    ``license_unavailable`` state ``unattributable``; a ref the fallback cannot
    single out (several candidates / legacy 409) → the same 503 with state
    ``ref_ambiguous`` (fixed by storing the licence token, not by retrying alone):
    never billed unverified. Both count in ``api_key_unattributable_total{reason}``. Otherwise 402 ``licence_unreadable`` /
    ``licence_unavailable`` (+ Retry-After; last-known-good serves within the
    gate's stale window) / ``licence_inactive``; a definite gateway 4xx is 502
    ``LICENCES_REFUSED``. Same caches and constants as the licence gate."""
    state = request.state
    if getattr(state, "actor_type", None) != ACTOR_TYPE_API_KEY:
        return
    ref = getattr(state, "api_key_licence_id", None)
    if not ref:
        raise _api_key_unattributable(
            "This tenant's Studio licence is not recorded yet, so API-key usage "
            "cannot be billed. Retry shortly.",
            "unattributable",
            "no_ref",
        )
    key = (str(tenant_id), str(ref))
    if key in _api_key_licence_ok:
        return
    async with request.app.state.session_factory() as session:
        stored = await TenantLicenceRepository(session).stored(tenant_id)
    try:
        ent = await tenant_entitlement(
            licenses,
            keyring,
            tenant_id,
            stored.ciphertext if stored is not None else None,
            token_expires_at=stored.token_expires_at if stored is not None else None,
            site="api_key_gate",
        )
    except LicenceTokenUnreadableError as exc:
        logger.error("tenant %s: stored licence token cannot be opened", tenant_id)
        raise _api_key_licence_refusal(
            LICENCE_UNREADABLE,
            "This tenant's stored Studio licence cannot be read. This needs an operator.",
            "unreadable",
        ) from exc
    except LicencesTenantNotOnboardedError as exc:
        raise _tenant_not_onboarded() from exc
    except LicencesRefusedError as exc:
        raise _api_key_licence_refused(exc.status_code) from exc
    except LicencesUnavailableError as exc:
        if key in _api_key_licence_stale:  # last-known-good, the gate's window
            logger.warning(
                "api-key licence verification failed for tenant=%s; serving last-known-good",
                tenant_id,
            )
            return
        raise _api_key_licence_refusal(
            LICENCE_UNAVAILABLE,
            "Could not verify this tenant's Studio licence. Retry shortly.",
            "unknown",
            retry=True,
        ) from exc
    refusal = _ref_verdict(ent, UUID(str(ref)))
    if refusal is not None:
        _api_key_licence_ok.pop(key, None)
        _api_key_licence_stale.pop(key, None)
        raise _ref_refusal(refusal)
    _api_key_licence_ok[key] = True
    _api_key_licence_stale[key] = True


LicenseGateDep = Annotated[None, Depends(require_license)]


# ─── The gate as a REUSABLE dependency list ──────────────────────────────
# `asgi.py` mounts TENANT_GATE router-wide on the tenant-scoped data routers.
# A router that MIXES must-gate and legitimately-exempt routes (admin: the read
# surfaces must stop when entitlement stops, but /provision mints the very
# licence the gate would ask for) cannot flip the whole router, so those routes
# take one of these lists per-route via `@router.get(..., dependencies=...)`.
# Defined ONCE here so a router-level mount and a per-route mount can never
# drift into enforcing different things.


#: Full fail-closed gate: tenant must be KNOWN, RLS bound to it, and ENTITLED.
TENANT_GATE: list[params.Depends] = [
    Depends(require_known_tenant),
    Depends(apply_tenant_rls),
    Depends(require_license),
]

#: ENTITLEMENT ONLY — deliberately no `require_known_tenant`, no RLS bind.
#:
#: This is the correct gate for the ADMIN tier, and the omissions are load-
#: bearing, not laziness:
#:
#: * `require_known_tenant` would import a DIFFERENT policy than the admin tier
#:   has chosen. `require_tenant_admin` runs the same `tenant_switchable` check
#:   but ONLY on a real cross-tenant scope-switch; a caller's OWN claim is
#:   deliberately left fail-open, because a freshly-licensed tenant with no
#:   registry row and no projects yet is indistinguishable from an unprovisioned
#:   one. Adding it here 403s exactly that tenant's admin out of /overview and
#:   /usage — measured, not hypothesised: it turns `test_usage_empty_tenant_
#:   returns_zeros` red. Entitlement is the hole being closed; tenant-existence
#:   posture for the admin tier is a separate decision and is NOT changed here.
#: * RLS-binding `app.current_tenant` to the caller's own claim would, under
#:   TENANT_RLS_MODE=enforce, filter the deliberately CROSS-tenant operator
#:   panes (/runs, /audit) down to one tenant — silently breaking super-admin.
#:
#: What it DOES enforce is the thing the owner ruled on: an unentitled tenant
#: (none/expired/revoked) is refused 402, and an unverifiable one 503.
LICENSE_GATE_ONLY: list[params.Depends] = [Depends(require_license)]
