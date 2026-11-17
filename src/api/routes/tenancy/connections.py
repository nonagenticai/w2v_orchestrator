"""Tenant provider connections (P3 WP4) — a tenant_admin brings its own provider.

- GET    /api/tenant/connections               → one entry per registry slot.
- PUT    /api/tenant/connections               → P8 bulk subtree write (all or nothing);
  tenant_admin OR a tenant API key with scope ``connections:write``.
- PUT    /api/tenant/connections/{slot}        → policy + SSRF → LIVE TEST → new version.
- POST   /api/tenant/connections/{slot}/test   → test a draft (body) or the current version.
- DELETE /api/tenant/connections/{slot}        → soft delete; the slot falls back to
  the deployment default.

Secrets are write-only: no response carries the secret or its ciphertext, only
``has_secret`` + the 12-hex ``key_fingerprint``. Audit details carry slot, kind,
version and fingerprint — never the secret, never the URL's userinfo.

Every route is ``TenantAdminDep`` + ``LICENSE_GATE_ONLY`` (the tenant-admin tier
posture, see ``tests/unit/test_tenant_gate_coverage.py``) + ``_CONNECTIONS_GATE``'s
RLS bind. The repository filters every statement on the acting tenant explicitly
AND the session's ``app.current_tenant`` is bound to the AUTHORIZED effective
tenant (``bind_admin_tenant_rls``): without that bind, in claim mode the
``after_begin`` listener falls back to the token's CLAIM tenant, so a super_admin
scope-switched into another tenant read 0 rows and failed every write on the RLS
``WITH CHECK``. Unlike the cross-tenant admin panes (``/api/admin/runs``, …),
these routes are strictly single-tenant, so binding is correct here.

Live tests (``POST …/test`` and the test inside ``PUT``) share ONE per-tenant
budget (``ProviderTestRateLimiter``, default 10 per 60 s); past it → 429
``PROVIDER_TEST_RATE_LIMITED`` with ``Retry-After``, checked only once the request
would really go out (validation failures cost nothing).

PUT ordering is the security property: the version guard, the policy, the SSRF
check and the "keep the secret across an endpoint change" refusal ALL run before
the live test, because the test sends the tenant's key to ``base_url``. A failed
test persists no connection or version — only a ``connection.test_failed`` audit
row (slot, kind, upstream status; owner decision). The read transaction is ended
before the (up to 10 s) network test so no pooled connection idles in a
transaction meanwhile; ``create_version`` re-checks the version under a row lock.
Two FIRST writes can both lock-read "no active row" (``FOR UPDATE`` locks nothing
that does not exist); the partial unique index rejects the loser, which is the
same 409 ``VERSION_CONFLICT`` as a stale ``expected_version``.

LLM access posture (licences ``byo_mode``): ``GET`` carries it as top-level
``llm_access``; under ``platform_only`` ("Local only") a write that SETS a
connection on ANY model-provider slot (``byo_governed`` in the registry: llm, vlm,
tts, asr, image, video_gen, ocr) is 409 ``tenant_byo_disabled`` (a delete is still allowed — it only returns the slot to
the platform).

Scope (owner decision 2026-10-20): these connections are w2v's OWN provider
store and are honoured only by the orchestrator pipeline. They are NOT mirrored
into worker_management any more — satellite services use the platform model.

``POST /{slot}/models`` lists the provider's models (``GET {base}/models``) for a
draft or the stored secret, under the same SSRF policy, test budget and 10 s
deadline as a live test; 422 ``models_unavailable`` otherwise.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi import params as fastapi_params
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes._common import (
    LICENSE_GATE_ONLY,
    AdminRepoDep,
    AuditRepoDep,
    SessionDep,
    SubjectIdDep,
    TenantAdminDep,
    TenantIdDep,
    require_tenant_admin,
)
from src.api.routes.tenancy.errors import TenancyError
from src.api.schemas.connections import (
    BulkSlotResult,
    ConnectionDraft,
    ConnectionPut,
    ConnectionsBulkPut,
    ConnectionsBulkResult,
    ConnectionTestResult,
    ConnectionView,
    KindRequirements,
    LlmAccessView,
    ModelsRequest,
    ModelsResult,
    SlotList,
    SlotView,
)
from src.api.schemas.tenancy_keys import SCOPE_CONNECTIONS_WRITE
from src.auth.tenancy_middleware import ACTOR_TYPE_API_KEY
from src.cache.raw import raw_redis
from src.database.models import TenantConnection, TenantConnectionVersion
from src.database.repositories.admin_audit_repository import AuditRepository
from src.database.repositories.api_keys_repository import ApiKeysRepository
from src.database.repositories.connections_repository import (
    KEEP_SECRET,
    ConnectionsRepository,
    ConnectionVersionConflictError,
    SecretReentryRequiredError,
    secret_may_carry,
)
from src.database.session import bind_session_tenant
from src.services.providers.access import required_slots, tenant_writes_refused
from src.services.providers.byo_posture import ByoMode, ByoPostureReader, get_byo_posture
from src.services.providers.crypto import (
    Keyring,
    KeyringNotConfiguredError,
    SecretDecryptError,
    get_keyring,
)
from src.services.providers.errors import (
    ByoPostureUnavailableError,
    SlotEgressRefusedError,
    SlotUnavailableError,
)
from src.services.providers.policy import check_egress, check_kind, check_not_self
from src.services.providers.rate_limit import ProviderTestRateLimiter
from src.services.providers.registry import BYO_GOVERNED_SLOTS, SLOT_NAMES, SLOTS, SlotSpec
from src.services.providers.settings import ProviderPolicySettings, get_provider_policy
from src.services.providers.tester import (
    HttpProviderTester,
    ModelLister,
    ProviderTester,
    ProviderTestResult,
)

router = APIRouter(prefix="/api/tenant/connections", tags=["tenancy"])
logger = logging.getLogger(__name__)

#: SQLSTATE ``unique_violation``.
_UNIQUE_VIOLATION = "23505"


# ─── dependencies (tests override these) ─────────────────────────────────


def get_connections_policy() -> ProviderPolicySettings:
    return get_provider_policy()


def get_connections_keyring() -> Keyring:
    return get_keyring()


PolicyDep = Annotated[ProviderPolicySettings, Depends(get_connections_policy)]
KeyringDep = Annotated[Keyring, Depends(get_connections_keyring)]


def get_connections_byo_posture() -> ByoPostureReader:
    return get_byo_posture()


ByoPostureDep = Annotated[ByoPostureReader, Depends(get_connections_byo_posture)]


def get_provider_tester(policy: PolicyDep) -> ProviderTester:
    return HttpProviderTester(policy)


TesterDep = Annotated[ProviderTester, Depends(get_provider_tester)]


def get_model_lister(policy: PolicyDep) -> ModelLister:
    return HttpProviderTester(policy)


ModelListerDep = Annotated[ModelLister, Depends(get_model_lister)]


def get_provider_test_limiter(request: Request, policy: PolicyDep) -> ProviderTestRateLimiter:
    """Per-tenant live-test limiter on the app's Redis (``app.state.redis``)."""
    redis = getattr(request.app.state, "redis", None)
    raw = None
    if redis is not None:
        try:
            raw = raw_redis(redis)
        except RuntimeError:  # wrapper never connected — same as no Redis
            raw = None
    return ProviderTestRateLimiter(
        raw, limit=policy.test_rate_limit, window_seconds=policy.test_rate_window_seconds
    )


LimiterDep = Annotated[ProviderTestRateLimiter, Depends(get_provider_test_limiter)]


async def bind_admin_tenant_rls(tenant_id: TenantAdminDep, session: SessionDep) -> None:
    """Bind ``app.current_tenant`` to the AUTHORIZED effective tenant (module doc).

    Depends on ``TenantAdminDep`` (not the bare ``TenantIdDep`` that
    ``apply_tenant_rls`` takes) so the bind happens only after the role check and
    the scope-switch ``tenant_switchable`` check passed. ``bind_session_tenant``
    also records the tenant in ``session.info`` so the ``after_begin`` listener
    re-binds it after this module's mid-request rollback/commit."""
    await bind_session_tenant(session, tenant_id)


#: Entitlement + the RLS bind; see the module doc for why this tier binds.
_CONNECTIONS_GATE: list[fastapi_params.Depends] = [
    *LICENSE_GATE_ONLY,
    Depends(bind_admin_tenant_rls),
]


@dataclass(frozen=True, slots=True)
class ConnectionsWriter:
    """Who may bulk-write: a tenant_admin (``api_key_id`` None) or a tenant API key
    carrying the ``connections:write`` scope."""

    tenant_id: UUID
    api_key_id: UUID | None


async def require_connections_writer(
    request: Request, tenant_id: TenantIdDep, admin_repo: AdminRepoDep, session: SessionDep
) -> ConnectionsWriter:
    """``TenantAdminDep`` OR an API key with ``connections:write`` (P8).

    A key acts as a ``member`` (``TenancyMiddleware``), so the admin check would
    always refuse it; the scope is read from the key's LIVE row, scoped to the key's
    own tenant — a revoked/expired key, or one without the scope, is 403."""
    if getattr(request.state, "actor_type", None) == ACTOR_TYPE_API_KEY:
        try:
            key_id = UUID(str(getattr(request.state, "api_key_id", None)))
        except ValueError:
            key_id = None
        scopes = (
            await ApiKeysRepository(session).live_scopes(key_id, tenant_id)
            if key_id is not None
            else None
        )
        if scopes is None or SCOPE_CONNECTIONS_WRITE not in scopes:
            raise _err(
                status.HTTP_403_FORBIDDEN,
                "API_KEY_SCOPE_REQUIRED",
                f"this API key lacks the {SCOPE_CONNECTIONS_WRITE!r} scope",
                required_scope=SCOPE_CONNECTIONS_WRITE,
            )
        return ConnectionsWriter(tenant_id=tenant_id, api_key_id=key_id)
    admin_tenant = await require_tenant_admin(request, tenant_id, admin_repo)
    return ConnectionsWriter(tenant_id=admin_tenant, api_key_id=None)


WriterDep = Annotated[ConnectionsWriter, Depends(require_connections_writer)]


async def bind_writer_tenant_rls(writer: WriterDep, session: SessionDep) -> None:
    """``bind_admin_tenant_rls`` for the bulk route: bind the AUTHORIZED tenant."""
    await bind_session_tenant(session, writer.tenant_id)


_BULK_GATE: list[fastapi_params.Depends] = [
    *LICENSE_GATE_ONLY,
    Depends(bind_writer_tenant_rls),
]


# ─── errors ──────────────────────────────────────────────────────────────


def _err(http: int, code: str, message: str, **details: object) -> TenancyError:
    return TenancyError(http, code, message, details or None)


def _keyring_unavailable() -> TenancyError:
    return _err(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "PROVIDER_KEYRING_UNAVAILABLE",
        "Provider credentials cannot be stored or read on this deployment right now.",
    )


async def read_byo_mode(posture: ByoPostureReader, tenant_id: UUID) -> ByoMode:
    """The tenant's licences ``byo_mode``, or a 503.

    Fails CLOSED and LOUD: an unreadable posture is a 503, never a silent
    ``byo_required: false`` that tells a byo_required tenant the platform is theirs.
    """
    try:
        return await posture.byo_mode(tenant_id, slot="tts")
    except ByoPostureUnavailableError as exc:
        raise _err(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "BYO_POSTURE_UNAVAILABLE",
            "The organisation's provider policy could not be read right now; retry shortly.",
        ) from exc


def _byo_required(mode: ByoMode) -> frozenset[str]:
    """The slots a tenant in ``mode`` must bring itself (``providers/access.py``)."""
    return required_slots(mode)


#: The ``llm_access`` labels the Providers page shows (licences ``byo_mode``).
LLM_ACCESS_LABELS: dict[ByoMode, str] = {
    "local_allowed": "Platform models allowed",
    "platform_only": "Local only",
    "byo_required": "Bring your own keys",
}
#: Slots a ``platform_only`` tenant may not set (its own keys are disabled): every
#: ``byo_governed`` registry slot, video_gen and ocr included (``providers/access.py``).
BYO_DISABLED_SLOTS: frozenset[str] = frozenset(BYO_GOVERNED_SLOTS)
TENANT_BYO_DISABLED = "tenant_byo_disabled"
MODELS_UNAVAILABLE = "models_unavailable"


def _byo_disabled(slot: str) -> TenancyError:
    return _err(
        status.HTTP_409_CONFLICT,
        TENANT_BYO_DISABLED,
        "This organisation is set to local (platform) models only; its own provider "
        f"keys are disabled, so {slot!r} cannot be configured.",
        slot=slot,
    )


async def _spend_test_budget(limiter: ProviderTestRateLimiter, tenant_id: UUID) -> None:
    decision = await limiter.hit(tenant_id)
    if not decision.allowed:
        raise TenancyError(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "PROVIDER_TEST_RATE_LIMITED",
            "Too many provider tests for this tenant; retry later.",
            {"retry_after": decision.retry_after},
            headers={"Retry-After": str(decision.retry_after)},
        )


def _reentry_required() -> TenancyError:
    return _err(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "SECRET_REENTRY_REQUIRED",
        str(SecretReentryRequiredError()),
    )


def _spec(slot: str, *, writable: bool) -> SlotSpec:
    spec = SLOTS.get(slot)
    if spec is None:
        raise _err(status.HTTP_404_NOT_FOUND, "SLOT_NOT_FOUND", f"unknown provider slot {slot!r}")
    if writable and spec.slot_type == "grant":
        raise _err(
            status.HTTP_409_CONFLICT,
            "SLOT_NOT_WRITABLE",
            f"provider {slot!r} is connected through its own grant flow, not stored here",
        )
    if writable and not spec.writable:
        raise _err(
            status.HTTP_409_CONFLICT,
            "SLOT_NOT_WRITABLE",
            f"provider {slot!r} is platform provided and cannot be configured",
        )
    return spec


# ─── views ───────────────────────────────────────────────────────────────


def _connection_view(conn: TenantConnection, ver: TenantConnectionVersion) -> ConnectionView:
    return ConnectionView(
        id=conn.id,
        kind=ver.kind,
        base_url=ver.base_url,
        model=ver.model,
        options=dict(ver.options or {}),
        version=ver.version,
        has_secret=ver.secret_ciphertext is not None,
        key_fingerprint=ver.secret_fingerprint if ver.secret_ciphertext is not None else None,
        last_test_at=conn.last_test_at,
        last_test_ok=conn.last_test_ok,
        last_test_error=conn.last_test_error,
    )


def _slot_view(
    policy: ProviderPolicySettings,
    slot: str,
    current: tuple[TenantConnection, TenantConnectionVersion] | None,
    byo_required: frozenset[str],
) -> SlotView:
    spec = SLOTS[slot]
    allowed = sorted(k for k in spec.tenant_kinds if policy.kind_allowed(slot, k))
    requirements: dict[str, KindRequirements] = {}
    for kind in allowed:
        ks = spec.tenant_kind_specs[kind]
        requirements[kind] = KindRequirements(
            base_url=ks.base_url,
            model_required=ks.model_required,
            secret_required=ks.secret_required,
            required_options=list(ks.required_options),
        )
    return SlotView(
        slot=slot,
        type=spec.slot_type,
        writable=spec.writable,
        allowed_kinds=allowed,
        source="tenant" if current is not None else "deployment",
        default_kind=policy.deployment_kind(slot),
        byo_required=slot in byo_required,
        kind_requirements=requirements,
        connection=_connection_view(*current) if current is not None else None,
    )


# ─── shared validation ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Current:
    """Plain snapshot of the active version (survives the pre-test rollback)."""

    connection_id: UUID
    version: int
    kind: str
    base_url: str | None
    has_secret: bool


def _validate_kind_fields(spec: SlotSpec, draft: ConnectionDraft) -> None:
    """Per-kind field requirements (registry :class:`KindSpec`) — all 422.

    The secret requirement is checked separately (:func:`_require_secret`): whether
    a "keep" draft has one depends on the stored version."""
    ks = spec.kind_spec(draft.kind)
    if ks is None:  # pragma: no cover - check_kind ran first
        return
    if ks.base_url == "forbidden" and draft.base_url is not None:
        raise _err(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "BASE_URL_NOT_ALLOWED",
            f"kind {draft.kind!r} has a fixed endpoint; do not set a base URL",
        )
    if ks.model_required and not draft.model:
        raise _err(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "MODEL_REQUIRED",
            f"kind {draft.kind!r} requires a model",
        )
    options = draft.options or {}
    missing = [
        key
        for key in ks.required_options
        if not isinstance(options.get(key), str) or not str(options.get(key)).strip()
    ]
    if missing:
        raise _err(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "OPTION_REQUIRED",
            f"kind {draft.kind!r} requires options: {', '.join(missing)}",
            missing=missing,
        )


def _require_secret(spec: SlotSpec, draft: ConnectionDraft, secret: str | None) -> None:
    """422 when the kind needs a credential and the effective secret is empty."""
    ks = spec.kind_spec(draft.kind)
    if ks is not None and ks.secret_required and not secret:
        raise _err(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "SECRET_REQUIRED",
            f"kind {draft.kind!r} requires a secret (API key / access key)",
        )


async def _validate_draft(
    policy: ProviderPolicySettings, slot: str, spec: SlotSpec, draft: ConnectionDraft
) -> None:
    """Kind allow-list, per-kind fields, base URL shape and SSRF/egress — all 422."""
    try:
        check_kind(policy, slot, draft.kind)
    except SlotUnavailableError as exc:
        raise _err(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "PROVIDER_KIND_NOT_ALLOWED", exc.reason
        ) from None
    _validate_kind_fields(spec, draft)
    await _validate_base_url(policy, slot, spec, draft)


async def _validate_base_url(
    policy: ProviderPolicySettings, slot: str, spec: SlotSpec, draft: ConnectionDraft
) -> None:
    """Base URL presence, shape and SSRF/egress — all 422 (the URL is never echoed)."""
    if draft.base_url is None:
        if draft.kind in spec.base_url_required_kinds:
            raise _err(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "BASE_URL_REQUIRED",
                f"kind {draft.kind!r} requires a base URL",
            )
        return
    try:
        parts = urlsplit(draft.base_url)
        bad = (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or bool(parts.query)
            or bool(parts.fragment)
        )
    except ValueError:
        bad = True
    if bad:
        # Never echo the URL: it may embed a credential.
        raise _err(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "INVALID_BASE_URL",
            "base URL must be http(s)://host[:port][/path] with no credentials, query or fragment",
        )
    try:
        check_not_self(slot, draft.base_url)
        await check_egress(policy, slot, draft.base_url)
    except SlotEgressRefusedError as exc:
        raise _err(status.HTTP_422_UNPROCESSABLE_CONTENT, "EGRESS_REFUSED", exc.reason) from None


async def _load_current(
    repo: ConnectionsRepository, tenant_id: UUID, slot: str
) -> tuple[_Current | None, TenantConnectionVersion | None]:
    current = await repo.get_current(tenant_id, slot)
    if current is None:
        return None, None
    conn, ver = current
    snap = _Current(
        connection_id=conn.id,
        version=ver.version,
        kind=ver.kind,
        base_url=ver.base_url,
        has_secret=ver.secret_ciphertext is not None,
    )
    return snap, ver


def _secret_for_test(
    draft: ConnectionDraft,
    tenant_id: UUID,
    current_ver: TenantConnectionVersion | None,
    keyring: Keyring,
) -> str | None:
    """The plaintext the live test should send (None = no credential).

    KEEP across a kind/endpoint change is refused HERE, before any request leaves:
    otherwise the test itself would carry the stored key to the new host."""
    if draft.secret is None:  # keep
        if current_ver is None or current_ver.secret_ciphertext is None:
            return None
        if not secret_may_carry(current_ver, kind=draft.kind, base_url=draft.base_url):
            raise _reentry_required()
        try:
            return ConnectionsRepository.decrypt_secret(tenant_id, current_ver, keyring)
        except KeyringNotConfiguredError:
            raise _keyring_unavailable() from None
        except SecretDecryptError:
            raise _err(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "SECRET_UNREADABLE",
                "the stored secret cannot be read; re-enter it",
            ) from None
    if draft.secret == "":  # clear
        return None
    if not keyring.configured:
        raise _keyring_unavailable()
    return draft.secret


# ─── the write service (per-slot PUT and bulk PUT share it) ──────────────

#: Live tests of one bulk write run at most this many at a time.
BULK_TEST_CONCURRENCY = 4

_SlotChange = Literal["created", "updated", "deleted", "unchanged"]


@dataclass(frozen=True, slots=True)
class _WriteContext:
    session: AsyncSession
    audit: AuditRepository
    policy: ProviderPolicySettings
    keyring: Keyring
    tester: ProviderTester
    limiter: ProviderTestRateLimiter
    tenant_id: UUID
    #: The tenant's licences ``byo_mode`` (``platform_only`` refuses BYO writes).
    byo_mode: ByoMode
    #: ``created_by`` of new versions (the human, or the key's synthetic subject).
    subject: UUID
    #: ``actor_subject_id`` of the bulk audit row: the key id for an API key.
    audit_actor: UUID
    api_key_id: UUID | None


@dataclass(slots=True)
class _Plan:
    """One slot of a write after validation, before anything leaves the process."""

    slot: str
    spec: SlotSpec
    body: ConnectionPut | None  # None = delete
    current: _Current | None
    secret: str | None = None
    noop: bool = False
    result: ProviderTestResult | None = None


@dataclass(frozen=True, slots=True)
class _Outcome:
    slot: str
    change: _SlotChange
    view: SlotView
    result: ProviderTestResult | None


def _is_noop(body: ConnectionPut, ver: TenantConnectionVersion | None, keyring: Keyring) -> bool:
    """The draft equals the stored version: same kind / endpoint / model / options and
    the secret is kept, or re-sent with the SAME fingerprint (or cleared on a version
    that has none). A no-op writes no version and runs no live test."""
    if ver is None:
        return False
    if (ver.kind, ver.base_url, ver.model) != (body.kind, body.base_url, body.model):
        return False
    if dict(ver.options or {}) != dict(body.options or {}):
        return False
    if body.secret is None:
        return True
    if body.secret == "":
        return ver.secret_ciphertext is None
    if ver.secret_ciphertext is None or not keyring.configured:
        return False
    return keyring.fingerprint(body.secret) == ver.secret_fingerprint


def _version_conflict(expected: int, current: int, slot: str | None = None) -> TenancyError:
    details: dict[str, object] = {"expected": expected, "current": current}
    if slot is not None:
        details["slot"] = slot
    return _err(
        status.HTTP_409_CONFLICT,
        "VERSION_CONFLICT",
        f"expected version {expected}, current is {current}",
        **details,
    )


async def _plan(
    ctx: _WriteContext,
    repo: ConnectionsRepository,
    slot: str,
    body: ConnectionPut | None,
    *,
    bulk: bool,
) -> _Plan:
    """Everything that can refuse a slot WITHOUT sending a request: the slot, the
    policy, field rules, SSRF/egress, the version guard and the secret rules."""
    if bulk:
        spec = SLOTS.get(slot)
        if spec is None or not spec.writable:
            why = (
                "unknown provider slot"
                if spec is None
                else "is connected through its own grant flow"
                if spec.slot_type == "grant"
                else "is platform provided and cannot be configured"
            )
            raise _err(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "SLOT_NOT_WRITABLE" if spec is not None else "SLOT_NOT_FOUND",
                f"provider {slot!r}: {why}",
            )
    else:
        spec = _spec(slot, writable=True)
    if body is None:
        current, _ver = await _load_current(repo, ctx.tenant_id, slot)
        if current is not None:
            return _Plan(slot=slot, spec=spec, body=None, current=current)
        return _Plan(slot=slot, spec=spec, body=None, current=None, noop=True)
    if tenant_writes_refused(ctx.byo_mode, slot):
        raise _byo_disabled(slot)
    await _validate_draft(ctx.policy, slot, spec, body)
    current, current_ver = await _load_current(repo, ctx.tenant_id, slot)
    live_version = current.version if current is not None else 0
    if body.expected_version != live_version:
        raise _version_conflict(body.expected_version, live_version, slot if bulk else None)
    if bulk and _is_noop(body, current_ver, ctx.keyring):
        return _Plan(slot=slot, spec=spec, body=body, current=current, noop=True)
    # Also the keyring gate: a kept secret is decrypted here (empty keyring → 503).
    secret = _secret_for_test(body, ctx.tenant_id, current_ver, ctx.keyring)
    _require_secret(spec, body, secret)
    return _Plan(slot=slot, spec=spec, body=body, current=current, secret=secret)


async def _plan_all(
    ctx: _WriteContext,
    repo: ConnectionsRepository,
    changes: dict[str, ConnectionPut | None],
    *,
    bulk: bool,
) -> list[_Plan]:
    """Single PUT: the first refusal is the answer (the P3 contract). Bulk: every
    slot is validated; 422 refusals are reported together, per slot. A 409 / 429 /
    503 is about the whole request and wins immediately."""
    plans: list[_Plan] = []
    errors: dict[str, dict[str, object]] = {}
    for slot, body in changes.items():
        try:
            plans.append(await _plan(ctx, repo, slot, body, bulk=bulk))
        except TenancyError as exc:
            if not bulk or exc.status != status.HTTP_422_UNPROCESSABLE_CONTENT:
                raise
            errors[slot] = {"code": exc.code, "message": exc.message, **(exc.details or {})}
    if errors:
        raise _err(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "BULK_VALIDATION_FAILED",
            f"{len(errors)} slot(s) failed validation; nothing was saved",
            slots=errors,
        )
    return plans


async def _run_tests(ctx: _WriteContext, plans: list[_Plan]) -> None:
    """Live-test every changed draft (bounded concurrency), each spending budget."""
    to_test = [p for p in plans if p.body is not None and not p.noop]
    if not to_test:
        return
    for _ in to_test:
        await _spend_test_budget(ctx.limiter, ctx.tenant_id)
    # End the read transaction before the network calls (module doc).
    await ctx.session.rollback()
    gate = asyncio.Semaphore(BULK_TEST_CONCURRENCY)

    async def _one(plan: _Plan) -> None:
        body = plan.body
        assert body is not None  # noqa: S101 — filtered above
        async with gate:
            plan.result = await ctx.tester.test(
                slot=plan.slot,
                kind=body.kind,
                base_url=body.base_url,
                model=body.model,
                options=dict(body.options or {}),
                secret=plan.secret,
            )

    await asyncio.gather(*(_one(p) for p in to_test))


async def _refuse_failed_tests(ctx: _WriteContext, plans: list[_Plan], *, bulk: bool) -> None:
    failed = [p for p in plans if p.result is not None and not p.result.ok]
    if not failed:
        return
    # Owner decision: a failed attempt is audited (never the secret) and committed
    # explicitly — the error response would otherwise roll it back. No connection
    # or version of ANY slot is written.
    for p in failed:
        assert p.body is not None and p.result is not None  # noqa: S101
        await ctx.audit.record(
            actor_subject_id=ctx.audit_actor,
            action="connection.test_failed",
            target_tenant_id=ctx.tenant_id,
            detail={
                "slot": p.slot,
                "kind": p.body.kind,
                "upstream_status": p.result.upstream_status,
                **({"bulk": True} if bulk else {}),
            },
        )
    await ctx.session.commit()
    if not bulk:
        result = failed[0].result
        assert result is not None  # noqa: S101
        raise _err(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "PROVIDER_TEST_FAILED",
            result.message,
            upstream_status=result.upstream_status,
        )
    raise _err(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "PROVIDER_TEST_FAILED",
        f"{len(failed)} provider test(s) failed; nothing was saved",
        slots={
            p.slot: {
                "ok": False,
                "message": p.result.message,
                "upstream_status": p.result.upstream_status,
            }
            for p in failed
            if p.result is not None
        },
    )


async def _persist(
    ctx: _WriteContext, repo: ConnectionsRepository, plan: _Plan
) -> tuple[_SlotChange, tuple[TenantConnection, TenantConnectionVersion] | None]:
    """Write one planned slot (no commit). Raises the repository's errors."""
    if plan.body is None:
        # Version guard: delete only the version seen at planning — a concurrent
        # update made during the probe window is a 409, never silently deleted.
        seen = plan.current.version if plan.current is not None else 0
        await repo.soft_delete(
            ctx.tenant_id, plan.slot, updated_by=ctx.subject, expected_version=seen
        )
        return "deleted", None
    body = plan.body
    conn, ver = await repo.create_version(
        tenant_id=ctx.tenant_id,
        slot=plan.slot,
        kind=body.kind,
        base_url=body.base_url,
        model=body.model,
        options=dict(body.options or {}),
        secret=KEEP_SECRET if body.secret is None else body.secret,
        keyring=ctx.keyring,
        created_by=ctx.subject,
        expected_version=body.expected_version,
    )
    result = plan.result
    assert result is not None  # noqa: S101 — every non-noop put was tested
    # A kind with no auth-only endpoint is stored "not probed" (last_test_ok NULL,
    # the explanation in last_test_error) — never recorded as a pass.
    if result.probed:
        await repo.record_test(ctx.tenant_id, conn.id, ok=True)
    else:
        await repo.record_test(ctx.tenant_id, conn.id, ok=None, error=result.message)
    return ("created" if plan.current is None else "updated"), (conn, ver)


def _audit_entry(
    plan: _Plan,
    change: _SlotChange,
    written: tuple[TenantConnection, TenantConnectionVersion] | None,
) -> dict[str, object]:
    entry: dict[str, object] = {"slot": plan.slot, "change": change}
    if written is not None:
        _conn, ver = written
        entry.update(
            kind=ver.kind,
            version=ver.version,
            key_fingerprint=ver.secret_fingerprint if ver.secret_ciphertext is not None else None,
            probed=plan.result.probed if plan.result is not None else None,
            cleared_secret=bool(
                plan.body is not None
                and plan.body.secret == ""
                and plan.current is not None
                and plan.current.has_secret
            ),
        )
    elif plan.current is not None:
        entry.update(connection_id=str(plan.current.connection_id), version=plan.current.version)
    return entry


async def apply_connection_changes(
    ctx: _WriteContext,
    changes: dict[str, ConnectionPut | None],
    *,
    bulk: bool,
) -> list[_Outcome]:
    """THE connection write: validate all → test all changed → persist all in ONE
    transaction, or nothing.

    ``bulk=False`` is the per-slot ``PUT /{slot}`` (P3 contract unchanged: unknown
    slot 404, grant/platform slot 409, every PUT re-tests, audit ``connection.set``).
    ``bulk=True`` is ``PUT /api/tenant/connections``: those refusals are 422 per slot,
    a draft identical to the stored version is a no-op (no test, no version — a
    partner re-saving its whole tree must not re-probe and bill every provider), and
    the write is audited as ONE ``connection.bulk_set`` row."""
    repo = ConnectionsRepository(ctx.session)
    plans = await _plan_all(ctx, repo, changes, bulk=bulk)
    await _run_tests(ctx, plans)
    await _refuse_failed_tests(ctx, plans, bulk=bulk)
    changed, written = await _persist_all(ctx, repo, plans, bulk=bulk)
    if bulk:
        await _audit_bulk(ctx, plans, changed, written)
    else:
        await _audit_single(ctx, plans, written)
    await ctx.session.commit()
    return await _outcomes(ctx, repo, plans, changed, written)


_Written = dict[str, tuple[TenantConnection, TenantConnectionVersion] | None]


async def _persist_all(
    ctx: _WriteContext, repo: ConnectionsRepository, plans: list[_Plan], *, bulk: bool
) -> tuple[dict[str, _SlotChange], _Written]:
    """Write every changed plan in the caller's ONE transaction; any failure rolls
    back all of them."""
    written: _Written = {}
    changed: dict[str, _SlotChange] = {}
    slot: str | None = None
    try:
        for plan in plans:
            slot = plan.slot
            if plan.noop:
                changed[plan.slot] = "unchanged"
                continue
            changed[plan.slot], written[plan.slot] = await _persist(ctx, repo, plan)
    except ConnectionVersionConflictError as exc:
        await ctx.session.rollback()
        raise _version_conflict(exc.expected, exc.current, slot if bulk else None) from None
    except IntegrityError as exc:
        await ctx.session.rollback()
        # Only a UNIQUE violation is a lost race (the active-(tenant, slot) index or
        # the (connection_id, version) PK); an FK/CHECK failure is not a conflict.
        if getattr(exc.orig, "sqlstate", None) != _UNIQUE_VIOLATION:
            raise
        lost = next((p for p in plans if p.slot not in written and not p.noop), plans[0])
        winner = await repo.get_current(ctx.tenant_id, lost.slot)
        live = winner[1].version if winner is not None else 0
        expected = lost.body.expected_version if lost.body is not None else 0
        raise _version_conflict(expected, live, lost.slot if bulk else None) from None
    except SecretReentryRequiredError:
        await ctx.session.rollback()
        raise _reentry_required() from None
    except KeyringNotConfiguredError:
        await ctx.session.rollback()
        raise _keyring_unavailable() from None
    return changed, written


async def _audit_bulk(
    ctx: _WriteContext, plans: list[_Plan], changed: dict[str, _SlotChange], written: _Written
) -> None:
    """ONE ``connection.bulk_set`` row (actor = the key id for an API key); none when
    every slot was unchanged."""
    entries = [_audit_entry(p, changed[p.slot], written.get(p.slot)) for p in plans]
    if all(e["change"] == "unchanged" for e in entries):
        return
    await ctx.audit.record(
        actor_subject_id=ctx.audit_actor,
        action="connection.bulk_set",
        target_tenant_id=ctx.tenant_id,
        detail={
            "actor_type": "api_key" if ctx.api_key_id is not None else "user",
            "key_id": str(ctx.api_key_id) if ctx.api_key_id is not None else None,
            "changes": entries,
        },
    )


async def _audit_single(ctx: _WriteContext, plans: list[_Plan], written: _Written) -> None:
    """The P3 per-slot audit: ``connection.set`` (+ ``connection.clear_secret``)."""
    for plan in plans:
        pair = written.get(plan.slot)
        if pair is None or plan.body is None or plan.result is None:
            continue
        _conn, ver = pair
        await ctx.audit.record(
            actor_subject_id=ctx.audit_actor,
            action="connection.set",
            target_tenant_id=ctx.tenant_id,
            detail={
                "slot": plan.slot,
                "kind": ver.kind,
                "version": ver.version,
                "key_fingerprint": (
                    ver.secret_fingerprint if ver.secret_ciphertext is not None else None
                ),
                "probed": plan.result.probed,
            },
        )
        if plan.body.secret == "" and plan.current is not None and plan.current.has_secret:
            await ctx.audit.record(
                actor_subject_id=ctx.audit_actor,
                action="connection.clear_secret",
                target_tenant_id=ctx.tenant_id,
                detail={"slot": plan.slot, "version": ver.version},
            )


async def _outcomes(
    ctx: _WriteContext,
    repo: ConnectionsRepository,
    plans: list[_Plan],
    changed: dict[str, _SlotChange],
    written: _Written,
) -> list[_Outcome]:
    outcomes: list[_Outcome] = []
    for plan in plans:
        pair = written.get(plan.slot)
        if pair is not None:
            await ctx.session.refresh(pair[0])  # last_test_* were written by an UPDATE
            current = pair
        elif changed[plan.slot] == "deleted":
            current = None
        else:
            current = await repo.get_current(ctx.tenant_id, plan.slot)
        outcomes.append(
            _Outcome(
                slot=plan.slot,
                change=changed[plan.slot],
                view=_slot_view(ctx.policy, plan.slot, current, _byo_required(ctx.byo_mode)),
                result=plan.result,
            )
        )
    return outcomes


# ─── routes ──────────────────────────────────────────────────────────────


@router.get(
    "",
    response_model=SlotList,
    operation_id="tenant.connections.list",
    dependencies=_CONNECTIONS_GATE,
)
async def list_connections(
    tenant_id: TenantAdminDep, session: SessionDep, policy: PolicyDep, posture: ByoPostureDep
) -> SlotList:
    mode = await read_byo_mode(posture, tenant_id)
    byo = _byo_required(mode)
    repo = ConnectionsRepository(session)
    active = {conn.slot: (conn, ver) for conn, ver in await repo.list_for_tenant(tenant_id)}
    return SlotList(
        items=[_slot_view(policy, slot, active.get(slot), byo) for slot in SLOT_NAMES],
        llm_access=LlmAccessView(mode=mode, label=LLM_ACCESS_LABELS[mode]),
    )


@router.put(
    "",
    response_model=ConnectionsBulkResult,
    operation_id="tenant.connections.bulk_put",
    dependencies=_BULK_GATE,
)
async def bulk_put_connections(
    body: ConnectionsBulkPut,
    writer: WriterDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
    policy: PolicyDep,
    keyring: KeyringDep,
    tester: TesterDep,
    limiter: LimiterDep,
    posture: ByoPostureDep,
) -> ConnectionsBulkResult:
    """P8 subtree PUT: every named slot set (or deleted with ``null``) atomically.

    validate all → live-test every CHANGED slot (bounded concurrency; each test
    spends the per-tenant budget) → persist all in one transaction, or nothing.
    Unknown / grant / platform slot or a non-empty ``dependencies`` → 422."""
    if body.dependencies:
        raise _err(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "DEPENDENCY_NOT_CONFIGURABLE",
            "w2v has no configurable dependencies (webscraper and media-discovery are platform-provided)",
            dependencies=sorted(body.dependencies),
        )
    outcomes = await apply_connection_changes(
        _WriteContext(
            session=session,
            audit=audit,
            policy=policy,
            keyring=keyring,
            tester=tester,
            limiter=limiter,
            tenant_id=writer.tenant_id,
            byo_mode=await read_byo_mode(posture, writer.tenant_id),
            subject=actor,
            audit_actor=writer.api_key_id or actor,
            api_key_id=writer.api_key_id,
        ),
        body.connections,
        bulk=True,
    )
    items: list[BulkSlotResult] = []
    for o in outcomes:
        test = None
        if o.result is not None:
            conn = o.view.connection
            test = ConnectionTestResult(
                ok=o.result.ok if o.result.probed else None,
                probed=o.result.probed,
                message=o.result.message,
                upstream_status=o.result.upstream_status,
                version=conn.version if conn is not None else None,
            )
        items.append(BulkSlotResult(**o.view.model_dump(), change=o.change, test=test))
    return ConnectionsBulkResult(items=items)


@router.put(
    "/{slot}",
    response_model=SlotView,
    operation_id="tenant.connections.put",
    dependencies=_CONNECTIONS_GATE,
)
async def put_connection(
    slot: str,
    body: ConnectionPut,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
    policy: PolicyDep,
    keyring: KeyringDep,
    tester: TesterDep,
    limiter: LimiterDep,
    posture: ByoPostureDep,
) -> SlotView:
    [outcome] = await apply_connection_changes(
        _WriteContext(
            session=session,
            audit=audit,
            policy=policy,
            keyring=keyring,
            tester=tester,
            limiter=limiter,
            tenant_id=tenant_id,
            byo_mode=await read_byo_mode(posture, tenant_id),
            subject=actor,
            audit_actor=actor,
            api_key_id=None,
        ),
        {slot: body},
        bulk=False,
    )
    return outcome.view


@router.post(
    "/{slot}/test",
    response_model=ConnectionTestResult,
    operation_id="tenant.connections.test",
    dependencies=_CONNECTIONS_GATE,
)
async def run_connection_test(
    slot: str,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
    policy: PolicyDep,
    keyring: KeyringDep,
    tester: TesterDep,
    limiter: LimiterDep,
    body: ConnectionDraft | None = None,
) -> ConnectionTestResult:
    """Test a draft (body) or, with no body, the CURRENT stored version.

    Always 200 with ``ok``: a failed test is a result, not an error. Testing the
    stored version records ``last_test_*`` on it; a draft records nothing but the
    audit row."""
    spec = _spec(slot, writable=True)
    repo = ConnectionsRepository(session)
    current, current_ver = await _load_current(repo, tenant_id, slot)
    if body is None:
        if current is None or current_ver is None:
            raise _err(
                status.HTTP_404_NOT_FOUND,
                "CONNECTION_NOT_FOUND",
                f"no connection is configured for {slot!r}",
            )
        draft = ConnectionDraft(
            kind=current_ver.kind,
            base_url=current_ver.base_url,
            model=current_ver.model,
            options=dict(current_ver.options or {}),
        )
        tested_version: int | None = current.version
    else:
        draft = body
        tested_version = None
    await _validate_draft(policy, slot, spec, draft)
    secret = _secret_for_test(draft, tenant_id, current_ver, keyring)
    _require_secret(spec, draft, secret)
    await _spend_test_budget(limiter, tenant_id)
    await session.rollback()

    result = await tester.test(
        slot=slot,
        kind=draft.kind,
        base_url=draft.base_url,
        model=draft.model,
        options=dict(draft.options or {}),
        secret=secret,
    )
    verdict: bool | None = result.ok if result.probed else None
    if tested_version is not None and current is not None:
        await repo.record_test(tenant_id, current.connection_id, ok=verdict, error=result.message)
    await audit.record(
        actor_subject_id=actor,
        action="connection.test" if result.ok else "connection.test_failed",
        target_tenant_id=tenant_id,
        detail={
            "slot": slot,
            "kind": draft.kind,
            "version": tested_version,
            "draft": tested_version is None,
            "upstream_status": result.upstream_status,
            "probed": result.probed,
        },
    )
    await session.commit()
    return ConnectionTestResult(
        ok=verdict,
        probed=result.probed,
        message=result.message,
        upstream_status=result.upstream_status,
        version=tested_version,
    )


@router.delete(
    "/{slot}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="tenant.connections.delete",
    dependencies=_CONNECTIONS_GATE,
)
async def delete_connection(
    slot: str,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> Response:
    _spec(slot, writable=True)
    repo = ConnectionsRepository(session)
    current = await repo.get_current(tenant_id, slot)
    if current is None or not await repo.soft_delete(tenant_id, slot, updated_by=actor):
        raise _err(
            status.HTTP_404_NOT_FOUND,
            "CONNECTION_NOT_FOUND",
            f"no connection is configured for {slot!r}",
        )
    conn, ver = current
    conn_id = conn.id
    await audit.record(
        actor_subject_id=actor,
        action="connection.delete",
        target_tenant_id=tenant_id,
        detail={"slot": slot, "connection_id": str(conn_id), "version": ver.version},
    )
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{slot}/models",
    response_model=ModelsResult,
    operation_id="tenant.connections.models",
    dependencies=_CONNECTIONS_GATE,
)
async def list_provider_models(
    slot: str,
    body: ModelsRequest,
    tenant_id: TenantAdminDep,
    session: SessionDep,
    policy: PolicyDep,
    keyring: KeyringDep,
    lister: ModelListerDep,
    limiter: LimiterDep,
    posture: ByoPostureDep,
) -> ModelsResult:
    """The provider's model ids (``GET {base}/models``) for a draft endpoint.

    ``secret`` omitted/``null`` = the STORED secret — only while ``kind`` and the
    endpoint equal the stored version's (the key was entrusted to that endpoint;
    otherwise 422 ``SECRET_REENTRY_REQUIRED``). Kind allow-list, base URL shape and
    SSRF/egress are checked before anything is sent; the call spends the live-test
    budget. Upstream failure → 422 ``models_unavailable``. The secret is never echoed.
    Under ``platform_only`` a BYO slot is 409 ``tenant_byo_disabled``, as for PUT —
    nothing is sent anywhere.
    """
    spec = _spec(slot, writable=True)
    if spec.byo_governed and tenant_writes_refused(await read_byo_mode(posture, tenant_id), slot):
        raise _byo_disabled(slot)
    draft = ConnectionDraft(kind=body.kind, base_url=body.base_url, secret=body.secret)
    try:
        check_kind(policy, slot, draft.kind)
    except SlotUnavailableError as exc:
        raise _err(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "PROVIDER_KIND_NOT_ALLOWED", exc.reason
        ) from None
    await _validate_base_url(policy, slot, spec, draft)
    _current, current_ver = await _load_current(ConnectionsRepository(session), tenant_id, slot)
    secret = _secret_for_test(draft, tenant_id, current_ver, keyring)
    await _spend_test_budget(limiter, tenant_id)
    # End the read transaction before the (up to 10 s) network call.
    await session.rollback()
    result = await lister.list_models(
        slot=slot, kind=draft.kind, base_url=draft.base_url, secret=secret
    )
    if result.models is None:
        raise _err(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            MODELS_UNAVAILABLE,
            result.message or "the provider did not return a model listing",
            upstream_status=result.upstream_status,
        )
    return ModelsResult(models=result.models)
