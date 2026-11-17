"""Membership resolution on the request path (P2 tenancy, WP3).

Mounted DIRECTLY INSIDE w2v_common's ``AuthMiddleware`` (``install_tenancy_middleware``)
and outside everything else that needs a tenant (``IdempotencyMiddleware``, the
routes, every outbound client). Auth proves WHO the caller is; this decides
WHICH TENANT a human caller acts in, per ``W2V_TENANCY_SOURCE``:

* ``claim`` — no change: the JWT ``tenant_id`` claim (bound by Auth) stands.
* ``shadow`` — serve by the claim, compute the membership answer alongside, log
  + count ``TENANCY_SHADOW_MISMATCH`` when they disagree, and lazily create the
  membership for the caller's OWN claim tenant (never for a super_admin
  scope-switch — a visit is not a membership). A token with NO claim is served
  by membership exactly as in ``membership`` mode and counted
  ``tenancy_shadow_checked_total{result="claim_missing"}``.
* ``membership`` — the claim is ignored. ``sub`` → active memberships →
  tenant (``src/auth/membership.py`` rules) → ``request.state.tenant_id`` /
  ``roles`` and the identity context with ``TenantSource.MEMBERSHIP``. No
  usable membership → 403 ``NO_MEMBERSHIP`` (``/api/auth/me`` exempt so the SPA
  can render its no-workspace screen; it gets NO tenant).

Why a pure-ASGI middleware and not a dependency: the identity ContextVars must
be bound BEFORE anything downstream runs. A (sync) dependency's ContextVar
writes do not reach the endpoint, and a route or inner middleware that calls a
peer with the service-account Bearer before the tenant is bound raises
``MissingTenantError`` under ``AUTH_JWT_TENANT_REQUIRED=false`` (w2v_common
0.90.0) — or, worse, under a claim-carrying token, would act as the CLAIM
tenant the membership mode is supposed to ignore.

``set_identity_context`` clears the project/actor bindings, so after
re-seeding the tenant this re-binds ``actor_id=sub`` and calls
``bind_attribution_from_headers`` again — exactly what Auth's JWT lane did.

Only the human JWT lane is resolved. Licence-token, delegated, trusted-network
callers keep the tenant their credential named. A tenant API key
(``w2v_sk_``, ``TenantSource.TENANT_API_KEY``) is tenant-bound by its key row
and always acts with the ``member`` role.

P9-6 — the API-key lane is BILLABLE: a key carries no licence token, so the
identity context binds ``license_id = tenants.licence_ref`` (the tenant's
entitling w2v licence, kept current by ``licence_refresh``). Run snapshots
(``run_setup``) and every outbound ``X-License-Id`` then carry the tenant's
licence, exactly as the licence-token lane does. A tenant with no
``licence_ref`` binds none and ``require_license`` refuses the key (503
``license_unavailable``) rather than letting unattributable spend through.

The membership (human JWT) lane carries no licence token either: it binds the
resolved tenant's own licence (``run_licence.resolve_request_licence`` — the
same ``licence_ref`` resolution a run's snapshot uses) as ``license_id``, so
request-time cost events bill that tenant. None → ``""`` + ERROR + counted
(``w2v_orchestrator_request_licence_unattributed_total``); never an umbrella
licence, never the nil UUID.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Literal
from uuid import UUID, uuid5, uuid7

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from w2v_common.auth.identity_context import (
    bind_attribution_from_headers,
    get_tenant_source_from_context,
    set_identity_context,
)
from w2v_common.auth.middleware import ApiKeyVerifier, AuthMiddleware
from w2v_common.auth.models import User
from w2v_common.tenant import TenantSource

from src.auth.membership import (
    ACTIVE_TENANT_HEADER,
    ACTIVE_TENANT_QUERY,
    MEMBER,
    NO_MEMBERSHIP,
    RESOLVED_BY_MEMBERSHIP,
    RESOLVED_NO_MEMBERSHIP,
    SUPER_ADMIN,
    MalformedRequestedTenantError,
    MembershipRow,
    Refusal,
    Resolution,
    lazy_membership_role,
    membership_roles,
    parse_requested_tenant,
    resolve_tenant,
)
from src.config import get_settings
from src.database.repositories.admin_repository import AdminRepository
from src.database.repositories.api_keys_repository import ApiKeysRepository
from src.database.repositories.members_repository import MembersRepository
from src.database.session import bind_session_tenant
from src.observability.metrics import (
    tenancy_shadow_checked_total,
    tenancy_shadow_mismatch_total,
)
from src.services.run_licence import request_licence_unattributed, resolve_request_licence

logger = logging.getLogger(__name__)
# Same stream as the claim-mode scope-switch line (_common.get_tenant_id).
_audit = logging.getLogger("w2v_orchestrator.audit")

#: Paths a JWT caller with NO usable membership may still reach in
#: ``membership`` mode — with no tenant bound. Everything else is 403
#: ``NO_MEMBERSHIP``. (WP4 adds its pre-membership routes here: invite
#: acceptance, the caller's own tenant list.) EXACT match only — a ``{param}``
#: segment matches exactly one segment that parses as a UUID, never a prefix.
NO_MEMBERSHIP_EXEMPT_PATHS: frozenset[str] = frozenset(
    {
        "/api/auth/me",
        "/api/invites/accept",
        "/api/invites/{invite_id}/accept",
        "/api/me/tenants",
    }
)


def _segment_matches(template: str, value: str) -> bool:
    if template.startswith("{") and template.endswith("}"):
        return _uuid_or_none(value) is not None
    return template == value


def is_no_membership_exempt(path: str) -> bool:
    """Exact match of ``path`` against ``NO_MEMBERSHIP_EXEMPT_PATHS`` (a
    trailing slash is ignored; ``{param}`` = one UUID segment)."""
    path = path.rstrip("/")
    if path in NO_MEMBERSHIP_EXEMPT_PATHS:
        return True
    parts = path.split("/")
    for template in NO_MEMBERSHIP_EXEMPT_PATHS:
        if "{" not in template:
            continue
        tparts = template.split("/")
        if len(tparts) == len(parts) and all(
            _segment_matches(t, v) for t, v in zip(tparts, parts, strict=True)
        ):
            return True
    return False


Lane = Literal["anonymous", "jwt", "api_key", "other"]

#: ``tenancy_shadow_checked_total{result}``: a shadow-mode JWT with no tenant
#: claim, served by membership (nothing to compare against).
CLAIM_MISSING = "claim_missing"

#: Fixed uuid5 namespace for tenant API-key subjects (P2 WP4b decision). NEVER
#: change it: every key's subject — and so its audit/billing attribution — would
#: move. = uuid5(NAMESPACE_URL, "https://w2v.nonagentic.ai/tenant-api-key").
W2V_API_KEY_NS = UUID("51a4e103-0237-5058-b22a-d10310b07c43")
ACTOR_TYPE_API_KEY = "api_key"  # pragma: allowlist secret — an actor-type label


def api_key_subject(key_id: str) -> UUID:
    """The synthetic subject a tenant API key acts as: ``uuid5(W2V_API_KEY_NS, key_id)``."""
    return uuid5(W2V_API_KEY_NS, key_id)


def _lane(request: Request) -> Lane:
    state = request.state
    if getattr(state, "user", None) is None:
        return "anonymous"  # public path, or Auth already answered
    source = get_tenant_source_from_context()
    if source is TenantSource.TENANT_API_KEY or getattr(state, "api_key_id", None):
        return "api_key"
    if source is TenantSource.JWT_CLAIM:
        return "jwt"
    if (
        source is None
        and getattr(state, "tenant_id", None) is None
        and getattr(state, "license", None) is None
        and getattr(state, "license_token", None) is None
    ):
        # AUTH_JWT_TENANT_REQUIRED=false: a verified JWT with no claim.
        return "jwt"
    return "other"


def _error(
    request: Request,
    status: int,
    code: str,
    message: str,
    details: dict[str, str] | None = None,
) -> JSONResponse:
    """The w2v_common AuthMiddleware error envelope, so the SPA parses one shape."""
    request_id = (
        getattr(request.state, "request_id", None)
        or request.headers.get("x-request-id")
        or str(uuid7())
    )
    error: dict[str, object] = {
        "code": code,
        "message": message,
        "status": status,
        "request_id": request_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if details:
        error["details"] = details
    return JSONResponse(status_code=status, content={"error": error})


def _uuid_or_none(raw: object) -> UUID | None:
    if raw is None:
        return None
    try:
        return UUID(str(raw))
    except ValueError:
        return None


class TenancyMiddleware:
    """See the module docstring. Install with :func:`install_tenancy_middleware`."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        lane = _lane(request)
        if lane == "api_key":
            licence_ref = await _tenant_licence_ref(
                scope, getattr(request.state, "tenant_id", None)
            )
            self._bind_api_key_role(request, licence_ref)
        elif lane == "jwt":
            response = await self._jwt(request, scope)
            if response is not None:
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)

    # ── lanes ──────────────────────────────────────────────────────────

    @staticmethod
    def _bind_api_key_role(request: Request, licence_ref: UUID | None = None) -> None:
        """A tenant API key acts as a plain ``member`` of its key's tenant —
        never an admin, whatever the key's creator was — and as its OWN
        synthetic subject :func:`api_key_subject` (never the creating user).

        w2v_common names the key ``api-key:<id>``, which is not a UUID, so every
        route needing a subject (``SubjectIdDep``, the request audit) would 401.
        Re-seed the identity with the synthetic UUID as subject AND actor;
        ``request.state.actor_type = "api_key"`` tells a human from a key.

        ``licence_ref`` (P9-6) is the key tenant's ``tenants.licence_ref``: it is
        bound as the request's ``license_id`` so the key's spend is attributed
        to the tenant's licence (``request.state.api_key_licence_id`` too, for
        ``require_license``). The KEY itself is never bound as a licence token."""
        state = request.state
        state.roles = [MEMBER]
        state.membership_role = MEMBER
        state.actor_type = ACTOR_TYPE_API_KEY
        state.api_key_licence_id = str(licence_ref) if licence_ref is not None else None
        key_id = getattr(state, "api_key_id", None)
        subject = str(api_key_subject(str(key_id))) if key_id else None
        user = getattr(state, "user", None)
        if isinstance(user, User):
            update: dict[str, object] = {"roles": [MEMBER]}
            if subject is not None:
                update["sub"] = subject
            state.user = user.model_copy(update=update)
        if subject is None:
            return
        state.subject_id = subject
        set_identity_context(
            tenant_id=getattr(state, "tenant_id", None),
            license_id=state.api_key_licence_id,
            subject_id=subject,
            caller_license_token=None,  # NEVER the key (it would ride outbound hops)
            tenant_source=TenantSource.TENANT_API_KEY,
            actor_id=subject,
        )
        # set_identity_context cleared the project binding; X-Actor-Id is not
        # honoured — a key names a tenant, not a human.
        bind_attribution_from_headers(request.headers, trusted=False)

    async def _jwt(self, request: Request, scope: Scope) -> JSONResponse | None:
        mode = get_settings().w2v_tenancy_source
        if mode == "membership":
            return await self._membership(request, scope)
        if getattr(request.state, "tenant_id", None) is None:
            if mode == "shadow":
                # A verified identity with NO claim (a provisioning invitee, an
                # operator account — portal users without a ``tenant_id``
                # attribute): there is nothing for the claim to serve, so w2v's
                # own membership is the only answer — served, not 401'd, and
                # counted as its own shadow outcome so users who exist only in
                # w2v's DB move the shadow denominator. No membership → the
                # membership mode's 403 ``NO_MEMBERSHIP`` (``/api/auth/me`` exempt).
                tenancy_shadow_checked_total.labels(result=CLAIM_MISSING).inc()
                logger.info("tenancy shadow: no tenant claim — resolved by membership")
                return await self._membership(request, scope)
            # claim with AUTH_JWT_TENANT_REQUIRED=false: the claim IS the tenant
            # source here, and there is none. Same answer the default JWT lane
            # gives — never let a tenant-less request reach a route.
            return _error(request, 401, "JWT_INVALID", "JWT has no valid tenant")
        if mode == "shadow":
            await self._shadow(request, scope)
        return None

    # ── membership mode ────────────────────────────────────────────────

    async def _membership(self, request: Request, scope: Scope) -> JSONResponse | None:
        user: User = request.state.user
        realm_roles = set(user.roles or [])
        is_super = SUPER_ADMIN in realm_roles
        try:
            requested = parse_requested_tenant(_requested_raw(request), is_super_admin=is_super)
        except MalformedRequestedTenantError as exc:
            return _error(request, 422, "INVALID_ACTIVE_TENANT", str(exc))

        factory = getattr(scope["app"].state, "session_factory", None)
        if factory is None:
            logger.error("membership tenancy: no session_factory on app.state")
            return _error(request, 503, "TENANCY_UNAVAILABLE", "tenant membership unavailable")
        sub = _uuid_or_none(user.sub)
        try:
            outcome = await _resolve(factory, sub, requested, is_super_admin=is_super)
        except Exception as exc:  # noqa: BLE001 — "could not check" is a 503, never a guess
            logger.warning("membership tenancy: resolution failed: %s", type(exc).__name__)
            return _error(request, 503, "TENANCY_UNAVAILABLE", "tenant membership unavailable")

        if outcome == "unknown_tenant":
            fallback = await _refused_active_tenant_fallback(
                request, factory, sub, requested, is_super_admin=is_super
            )
            if isinstance(fallback, JSONResponse):
                return fallback
            outcome = fallback
        if isinstance(outcome, Refusal):
            if is_no_membership_exempt(request.url.path):
                self._bind_no_tenant(request, user)
                return None
            return _error(
                request,
                403,
                NO_MEMBERSHIP,
                "You are not a member of this workspace"
                if outcome.reason == "not_a_member"
                else "You are not a member of any workspace",
            )

        roles = membership_roles(outcome, realm_roles)
        if outcome.via == "super_admin":
            _audit.info(
                "SCOPE_SWITCH: %s",
                {
                    "subject_id": user.sub,
                    "to_tenant": str(outcome.tenant_id),
                    "path": request.url.path,
                    "tenancy_source": "membership",
                },
            )
        licence = await _membership_licence(factory, outcome.tenant_id)
        self._bind_tenant(request, user, outcome, roles, licence)
        return None

    @staticmethod
    def _bind_tenant(
        request: Request, user: User, outcome: Resolution, roles: list[str], licence: str
    ) -> None:
        """``licence`` is the resolved tenant's own licence (``run_licence``'s
        resolution, ``""`` when it has none) — bound as the request's
        ``license_id`` so request-time cost events bill THAT tenant."""
        state = request.state
        state.tenant_id = str(outcome.tenant_id)
        state.roles = roles
        state.membership_role = outcome.role
        state.tenant_resolved_by = RESOLVED_BY_MEMBERSHIP
        # A super_admin in a tenant they are not a member of: AuditMiddleware
        # records this as cross-tenant (it cannot infer it — tenant_id above
        # already IS the requested tenant).
        state.cross_tenant_scope = outcome.via == "super_admin"
        state.user = user.model_copy(update={"tenant_id": outcome.tenant_id, "roles": roles})
        set_identity_context(
            tenant_id=outcome.tenant_id,
            license_id=licence,
            subject_id=user.sub,
            caller_license_token=None,
            # VOUCHED (w2v_common 0.90.0): sub verified by Auth against the realm
            # JWKS, tenant read from w2v's own membership table.
            tenant_source=TenantSource.MEMBERSHIP,
            actor_id=user.sub,
        )
        # set_identity_context cleared the project binding Auth made; re-bind.
        # X-Actor-Id is NOT honoured: the verified sub IS the actor.
        bind_attribution_from_headers(request.headers, trusted=False)

    @staticmethod
    def _bind_no_tenant(request: Request, user: User) -> None:
        """Exempt path, no membership: drop whatever tenant Auth bound from a
        claim — in membership mode a claim names nothing — so any outbound hop
        fails closed instead of acting as the claim tenant.

        Stamps ``tenant_resolved_by = RESOLVED_NO_MEMBERSHIP`` so the route deps
        know the missing tenant is membership's ANSWER, not a missing claim —
        the same in ``membership`` mode and for a claimless ``shadow`` token."""
        realm_roles = set(user.roles or [])
        state = request.state
        state.tenant_id = None
        state.tenant_resolved_by = RESOLVED_NO_MEMBERSHIP
        state.roles = [SUPER_ADMIN] if SUPER_ADMIN in realm_roles else []
        state.membership_role = None
        state.user = user.model_copy(update={"tenant_id": None, "roles": state.roles})
        set_identity_context(
            tenant_id=None,
            license_id=None,
            subject_id=user.sub,
            caller_license_token=None,
            tenant_source=None,
            actor_id=user.sub,
        )
        bind_attribution_from_headers(request.headers, trusted=False)

    # ── shadow mode ────────────────────────────────────────────────────

    async def _shadow(self, request: Request, scope: Scope) -> None:
        """Never changes the response. Any failure is logged + counted.

        ``tenancy_shadow_checked_total`` counts each comparison exactly once:
        ``match`` / ``mismatch`` when a verdict was reached (even if the lazy
        upsert after it then fails), ``error`` only when it was not.
        ``tenancy_shadow_mismatch_total{reason="error"}`` likewise means "the
        CHECK failed" — a lazy-upsert failure after a verdict is not a failed
        check (its verdict was already counted), so it is logged, not counted
        as an error mismatch.
        """
        verdict: list[str] = []
        try:
            await self._shadow_compare(request, scope, verdict)
        except Exception as exc:  # noqa: BLE001 — shadow must never break a request
            if not verdict:
                tenancy_shadow_mismatch_total.labels(reason="error").inc()
                tenancy_shadow_checked_total.labels(result="error").inc()
                logger.warning("TENANCY_SHADOW_MISMATCH check failed: %s", type(exc).__name__)
            else:
                logger.warning(
                    "tenancy shadow: lazy membership upsert failed after verdict %r: %s",
                    verdict[0],
                    type(exc).__name__,
                )

    async def _shadow_compare(self, request: Request, scope: Scope, verdict: list[str]) -> None:
        user: User = request.state.user
        realm_roles = set(user.roles or [])
        is_super = SUPER_ADMIN in realm_roles
        claim = UUID(str(request.state.tenant_id))
        try:
            requested = parse_requested_tenant(_requested_raw(request), is_super_admin=is_super)
        except MalformedRequestedTenantError:
            requested = None
        # What claim mode serves: the claim, or a super_admin's scope-switch.
        served = requested if (is_super and requested is not None) else claim
        sub = _uuid_or_none(user.sub)
        factory = scope["app"].state.session_factory
        async with factory() as session:
            members = MembersRepository(session)
            rows = await members.list_resolvable_memberships(sub) if sub else []
            memberships = [MembershipRow(r.tenant_id, r.role, r.last_seen_at) for r in rows]
            outcome = resolve_tenant(memberships, requested, is_super_admin=is_super)
            reason: str | None = None
            would_serve: UUID | None = None
            if isinstance(outcome, Refusal):
                reason = outcome.reason
            elif outcome.via == "super_admin" and not await AdminRepository(
                session
            ).tenant_switchable(outcome.tenant_id):
                reason = "super_admin_unknown_tenant"
            else:
                would_serve = outcome.tenant_id
                if would_serve != served:
                    reason = "tenant_differs"
            verdict.append("match" if reason is None else "mismatch")
            tenancy_shadow_checked_total.labels(result=verdict[0]).inc()
            if reason is not None:
                tenancy_shadow_mismatch_total.labels(reason=reason).inc()
                logger.warning(
                    "TENANCY_SHADOW_MISMATCH: %s",
                    {
                        "subject_id": user.sub,
                        "reason": reason,
                        "claim_tenant": str(claim),
                        "served_tenant": str(served),
                        "membership_tenant": str(would_serve) if would_serve else None,
                        "path": request.url.path,
                    },
                )
            # Lazy upsert for the caller's OWN claim tenant only, and only into a
            # tenant w2v has provisioned (an active/licence_lapsed registry row).
            # A super_admin scope-switch is a visit, not a membership; a bare fr
            # claim must never make its holder a resolvable member after cutover.
            if (
                sub is not None
                and served == claim
                and all(m.tenant_id != claim for m in memberships)
                and await members.tenant_accepts_lazy_members(claim)
            ):
                inserted = await members.ensure_member(
                    subject_id=sub,
                    tenant_id=claim,
                    role=lazy_membership_role(realm_roles),
                    email=user.email,
                    display_name=user.preferred_username,
                )
                await session.commit()
                if inserted:
                    logger.info(
                        "tenancy shadow: lazily created membership subject=%s tenant=%s",
                        user.sub,
                        claim,
                    )


async def _membership_licence(factory: async_sessionmaker[AsyncSession], tenant_id: UUID) -> str:
    """The licence a membership-resolved request of ``tenant_id`` is billed to —
    the same resolution a run uses (``resolve_request_licence``: the tenant's
    ``licence_ref``). No ref, no row, or a failed read → ``""`` logged at ERROR
    and counted: never an umbrella licence, never the nil UUID. A failed read
    does not refuse the request (the membership itself was resolved)."""
    try:
        async with factory() as session:
            return await resolve_request_licence(session, tenant_id)
    except Exception as exc:  # noqa: BLE001 — could-not-read = unattributed, loudly
        logger.warning("membership tenancy: licence_ref read failed: %s", type(exc).__name__)
        return request_licence_unattributed(tenant_id, "read_failed")


async def _tenant_licence_ref(scope: Scope, tenant_id: object) -> UUID | None:
    """``tenants.licence_ref`` of an API key's tenant (P9-6), or None — no
    registry row, no ref yet, or the read failed (logged; the licence gate then
    refuses the key, so a failed read can never turn into unbilled spend)."""
    tenant = _uuid_or_none(tenant_id)
    factory = getattr(scope["app"].state, "session_factory", None)
    if tenant is None or factory is None:
        return None
    try:
        async with factory() as session:
            return await AdminRepository(session).tenant_licence_ref(tenant)
    except Exception as exc:  # noqa: BLE001 — could-not-read = no licence bound (fail closed)
        logger.warning("api key lane: licence_ref read failed: %s", type(exc).__name__)
        return None


def _requested_raw(request: Request) -> str | None:
    """Header wins over the query param — the claim-mode scope-switch's rule."""
    return request.headers.get(ACTIVE_TENANT_HEADER) or request.query_params.get(
        ACTIVE_TENANT_QUERY
    )


async def _resolve(
    factory: async_sessionmaker[AsyncSession],
    sub: UUID | None,
    requested: UUID | None,
    *,
    is_super_admin: bool,
) -> Resolution | Refusal | Literal["unknown_tenant"]:
    """Membership mode's I/O half: read memberships, apply the rules, and for a
    super_admin acting into a non-member tenant confirm it is switchable."""
    memberships: list[MembershipRow] = []
    async with factory() as session:
        if sub is not None:
            rows = await MembersRepository(session).list_resolvable_memberships(sub)
            memberships = [MembershipRow(r.tenant_id, r.role, r.last_seen_at) for r in rows]
        outcome = resolve_tenant(memberships, requested, is_super_admin=is_super_admin)
        if (
            isinstance(outcome, Resolution)
            and outcome.via == "super_admin"
            and not await AdminRepository(session).tenant_switchable(outcome.tenant_id)
        ):
            return "unknown_tenant"
        return outcome


async def _refused_active_tenant_fallback(
    request: Request,
    factory: async_sessionmaker[AsyncSession],
    sub: UUID | None,
    requested: UUID | None,
    *,
    is_super_admin: bool,
) -> Resolution | Refusal | JSONResponse:
    """A super_admin's requested tenant is not switchable (unknown, suspended or
    deleted). Data routes answer 404 ``TENANT_NOT_FOUND``. An identity route
    (``NO_MEMBERSHIP_EXEMPT_PATHS``: ``/api/auth/me`` …) must still answer: a
    404 there hid the caller's own roles from the SPA, so a super_admin who
    picked a dead tenant lost the very picker that could undo it. Resolve it as
    if no tenant were requested and record which choice was refused
    (``AuthMe.active_tenant_refused``) so the SPA drops it."""
    if not is_no_membership_exempt(request.url.path):
        # ``details.active_tenant`` tells the SPA the refusal is about its
        # stored ACTIVE tenant choice (not a target tenant in the path), so it
        # can drop the choice instead of rendering an empty shell.
        return _error(
            request,
            404,
            "TENANT_NOT_FOUND",
            "tenant not provisioned",
            {"active_tenant": str(requested)},
        )
    request.state.active_tenant_refused = requested
    try:
        outcome = await _resolve(factory, sub, None, is_super_admin=is_super_admin)
    except Exception as exc:  # noqa: BLE001 — "could not check" is a 503, never a guess
        logger.warning("membership tenancy: resolution failed: %s", type(exc).__name__)
        return _error(request, 503, "TENANCY_UNAVAILABLE", "tenant membership unavailable")
    if outcome == "unknown_tenant":
        # Unreachable (no requested tenant never resolves via super_admin), but
        # fail closed rather than assume.
        return Refusal(reason="no_membership")
    return outcome


def make_api_key_verifier(
    session_factory: async_sessionmaker[AsyncSession],
) -> ApiKeyVerifier:
    """The ``w2v_sk_`` verifier registered with w2v_common (lifespan startup).

    w2v_common hands every ``Bearer w2v_sk_…`` here instead of licence
    introspection. ``ApiKeysRepository.resolve`` parses strictly, looks the key
    up by its public prefix and constant-time-compares the SHA-256; unknown,
    revoked and expired keys are all ``None`` → 401 ``INVALID_API_KEY``. A DB
    failure propagates → w2v_common answers 503 (could not check ≠ invalid).
    ``last_used_at`` is bumped on every accepted use, best-effort: a failed
    bump is logged and the verified key is still accepted.
    """

    async def _verify(token: str) -> tuple[UUID, str] | None:
        async with session_factory() as session:
            keys = ApiKeysRepository(session)
            row = await keys.resolve(token)
            if row is None:
                return None
            key_id, tenant_id = row.id, row.tenant_id
            try:
                # The resolve ran tenant-less through the SECURITY DEFINER
                # resolver; the bump is an ordinary RLS-scoped write, so bind
                # the key's own tenant first (else enforce makes it UPDATE 0).
                await bind_session_tenant(session, tenant_id)
                await keys.touch_last_used(key_id)
                await session.commit()
            except Exception as exc:  # noqa: BLE001 — bookkeeping must not fail a valid key
                # Best-effort: the key already verified. Log the type only —
                # never the token or the driver message (may echo a DSN).
                await session.rollback()
                logger.warning(
                    "api key last_used_at update failed key_id=%s: %s",
                    key_id,
                    type(exc).__name__,
                )
            return tenant_id, str(key_id)

    return _verify


#: w2v_common's switch for "a verified JWT with no ``tenant_id`` claim is a 401"
#: (read per request; anything but an explicit false value means required).
JWT_TENANT_REQUIRED_ENV = "AUTH_JWT_TENANT_REQUIRED"
_ENV_FALSE = frozenset({"0", "false", "no", "off"})


def align_jwt_tenant_requirement(mode: str) -> bool:
    """In ``membership`` (and ``shadow``) mode the JWT ``tenant_id`` claim is not
    REQUIRED: in ``membership`` it names nothing; in ``shadow`` a claimless
    identity is served by membership (shadow outcome ``claim_missing``) instead
    of being 401'd before the shadow comparison could ever see it. In
    ``membership``: Keycloak only authenticates, w2v's own
    ``tenant_members`` table decides the tenant (CP pattern). Left at
    w2v_common's default (required), a portal user with no ``tenant_id``
    attribute — e.g. one created by a provisioning invite — is 401'd at the
    auth edge before membership resolution ever runs, i.e. ``membership`` mode
    would still be gated on the claim. Forces ``AUTH_JWT_TENANT_REQUIRED=false``
    for this process and returns True when it changed anything.

    Only the ABSENT claim is relaxed; w2v_common still 401s a present-but-
    malformed claim (its contract). ``claim`` is left untouched: there the claim
    IS the tenant source (a claimless token stays a 401).
    """
    if mode not in ("membership", "shadow"):
        return False
    current = (os.environ.get(JWT_TENANT_REQUIRED_ENV) or "").strip().lower()
    if current in _ENV_FALSE:
        return False
    os.environ[JWT_TENANT_REQUIRED_ENV] = "false"
    logger.warning(
        "W2V_TENANCY_SOURCE=%s: forcing %s=false (was %r) — a claimless identity "
        "is resolved from tenant_members, never 401'd for the missing claim",
        mode,
        JWT_TENANT_REQUIRED_ENV,
        current or None,
    )
    return True


def install_tenancy_middleware(app: Starlette) -> None:
    """Insert :class:`TenancyMiddleware` DIRECTLY inside ``AuthMiddleware``.

    ``app.user_middleware`` is ordered outer→inner, so index(Auth)+1 is the
    first thing Auth's ``call_next`` reaches: the tenant is resolved before
    Idempotency keys on it and before any route or client runs. Must be called
    before the app serves its first request (Starlette builds the stack lazily).

    Also aligns w2v_common's claim requirement with ``W2V_TENANCY_SOURCE``
    (:func:`align_jwt_tenant_requirement`), so a ``membership`` deployment can
    never still depend on the claim at the auth edge.
    """
    align_jwt_tenant_requirement(get_settings().w2v_tenancy_source)
    for index, entry in enumerate(app.user_middleware):
        cls = entry.cls
        if isinstance(cls, type) and issubclass(cls, AuthMiddleware):
            app.user_middleware.insert(index + 1, Middleware(TenancyMiddleware))
            return
    # Auth disabled (AUTH_ENABLED=false): there is no authenticated caller to
    # resolve, and nothing for this middleware to do.
    logger.warning("AuthMiddleware not registered; tenancy middleware not installed")
