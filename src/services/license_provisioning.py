"""Idempotent tenant-licence provisioning (the write side of the license gate).

The tenant's w2v licence is TENANT-level: it is held by the deterministic
service subject ``svc:<tenant-uuid>:w2v`` (licences ``models.service_subject_sub``,
realm ``service``), never by the person who happened to provision or first use
the tenant. A person-held licence dies with that person's membership, counts as
a seat, and attributes the tenant's spend to one member.

Licences mints that holder through exactly one route —
``POST /internal/tenants/{id}/bundles/w2v/ensure`` (licences migrations 050 +
052: w2v owns the tenant and is a ``bundle_ensurer`` of the ``w2v`` bundle,
whose apps are ``w2v`` + ``w2v-image``). It is idempotent per (tenant, app):
``exists`` / ``adopt`` (a sole tenant-level unbundled licence) / ``insert``. A
direct ``POST /licenses/user/{sub}`` cannot state realm ``service``, so it is
never used for the tenant licence.

``ensure_tenant_licences`` is the shared primitive behind these call sites:

- **super-admin provisioning** (``POST /api/admin/provision`` in ``admin.py``);
- **partner tenant creation** (``POST /api/tenants`` in ``tenancy/partner.py``)
  — via :func:`ensure_tenant_licence_ref`, so a tenant is never created with a
  null ``licence_ref`` (live 2026-10-15: a partner-created tenant registered in
  licences but held no licence, unusable with auto-issue off);
- **the operator CLI** ``scripts/ensure_tenant_licences.py`` (repairs an
  existing tenant through the same function);
- **self-serve lazy auto-issue** (``GET /api/license/status`` in ``auth.py``),
  behind ``W2V_LICENSE_AUTOISSUE_ENABLED``.

A NEW tenant (super-admin provision, the early-access approve) goes through
:func:`provision_tenant_licences` instead: the same ensure, then the tenant's LLM
access posture (licences ``byo_mode`` for app ``w2v``). Licences' default for a
new tenant is ``byo_required``, which blocks every run, so the posture is ALWAYS
set explicitly — ``local_allowed`` unless the caller chose another posture.
The repair / lazy paths above never touch the posture (they would overwrite an
operator's later choice).

A refused app (``member_licence`` — the tenant still holds a legacy person-held
licence, migrate it with licences' ``scripts/migrate_member_licences_to_tenant.py``;
``several_candidates``; ``no_plan``; …) is reported, never papered over by
minting for a person.

**A client timeout is not a failure.** Since licences #275 the ensure also
provisions the bundle's dependency licences (live 2026-10-15: 15.8 s, which
the old 15 s client budget gave up on while licences went on to commit). The
ensure has its own longer budget, and when it still times out
:func:`ensure_tenant_licences` re-reads the tenant-level holders once (``GET
/licenses/user/svc:<tenant>:<app>``) before reporting the failure: licences
writes the whole bundle in ONE transaction, so a ``w2v`` licence held by
``svc:<tenant>:w2v`` means the ensure finished.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal
from uuid import UUID

import httpx

from src.clients.licenses_client import LicensesClientError

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.clients.licenses_client import LicensesClient

_log = logging.getLogger("w2v_orchestrator.license_provisioning")

#: The licences bundle w2v ensures on its tenants (apps ``w2v`` + ``w2v-image``).
W2V_BUNDLE_KEY = "w2v"
#: The app whose tenant-level licence entitles (and bills) a w2v tenant.
W2V_LICENCE_APP_KEY = "w2v"
#: The apps of licences' ``w2v`` bundle (migration 028) — what a re-read after an
#: ensure timeout looks for. The ensure's own answer stays the source of truth.
W2V_BUNDLE_APP_KEYS = (W2V_LICENCE_APP_KEY, "w2v-image")
#: The refusal recorded for a bundle app the post-timeout re-read did not find
#: while the ``w2v`` licence was there (licences' own reason is unknowable then).
NOT_CONFIRMED_AFTER_TIMEOUT = "not_confirmed_after_timeout"
#: licences ``bundle_ensure.DEFAULT_PLAN_CODE`` — the last plan preference.
_DEFAULT_PLAN_CODE = "standard"
_SERVICE_SUBJECT_PREFIX = "svc:"


def tenant_licence_subject(tenant_id: UUID | str, app_key: str) -> str:
    """``svc:<canonical-tenant-uuid>:<app-key>`` — the tenant-level licence holder."""
    return f"{_SERVICE_SUBJECT_PREFIX}{UUID(str(tenant_id))}:{app_key}"


class TenantLicenceRefusedError(RuntimeError):
    """licences ensured the bundle but REFUSED an app w2v needs (nothing was
    written for it). ``reason`` is licences' refusal code."""

    def __init__(self, tenant_id: UUID | str, app_key: str, reason: str) -> None:
        super().__init__(
            f"licences refused the tenant-level {app_key!r} licence for tenant "
            f"{tenant_id}: {reason}"
        )
        self.app_key = app_key
        self.reason = reason


@dataclass(frozen=True, slots=True)
class TenantLicences:
    """The tenant's ensured licences: ``app_key -> licence id`` (all held by
    ``svc:<tenant>:<app>``), and the apps licences refused (``app_key -> reason``)."""

    license_ids: dict[str, str] = field(default_factory=dict)
    refused: dict[str, str] = field(default_factory=dict)


async def _plan_preference(licenses: LicensesClient, *, tenant_id: UUID | str) -> list[str]:
    """The ordered plan codes to insert on: the w2v app's non-revoked plans in
    licences' order (the plan the old per-person mint picked comes first), then
    licences' default. De-duplicated (licences 422s a duplicate)."""
    codes: list[str] = []
    apps = await licenses.list_apps(tenant_id=tenant_id)
    app = next((a for a in apps if a.key == W2V_BUNDLE_KEY), None)
    if app is not None:
        plans = await licenses.list_plans(app.id, tenant_id=tenant_id)
        codes = [p.code for p in plans if not p.revoked]
    if _DEFAULT_PLAN_CODE not in codes:
        codes.append(_DEFAULT_PLAN_CODE)
    return list(dict.fromkeys(codes))


async def _held_licence_id(
    licenses: LicensesClient, *, tenant_id: UUID | str, app_key: str
) -> str | None:
    """The one live licence held by ``svc:<tenant>:<app_key>``, else None.

    None also for a subject licences does not know yet (404) and for more than
    one live row (not what an ensure leaves behind — not a confirmation)."""
    subject = tenant_licence_subject(tenant_id, app_key)
    try:
        rows = await licenses.list_user_licenses(subject, tenant_id=tenant_id)
    except LicensesClientError as exc:
        if exc.status_code == 404:
            return None
        raise
    live = [
        row.id
        for row in rows
        if row.revoked_at is None
        and (row.status is None or row.status == "active")
        and (row.app is None or row.app.key == app_key)
    ]
    return live[0] if len(live) == 1 else None


async def _reread_after_timeout(
    licenses: LicensesClient, *, tenant_id: UUID | str
) -> TenantLicences | None:
    """After an ensure timed out client-side: did licences finish anyway?

    Returns the ensured licences when the tenant's ``w2v`` licence is held by
    ``svc:<tenant>:w2v`` (the bundle is one transaction, so it committed), None
    when it is not — or when the re-read itself fails — so the caller reports
    the original timeout. A bundle app absent while ``w2v`` is present is
    recorded as refused :data:`NOT_CONFIRMED_AFTER_TIMEOUT`."""
    try:
        found = {
            key: await _held_licence_id(licenses, tenant_id=tenant_id, app_key=key)
            for key in W2V_BUNDLE_APP_KEYS
        }
    except Exception:  # noqa: BLE001 — the timeout is what gets reported
        _log.warning(
            "re-read after a timed-out bundle ensure failed for tenant=%s",
            tenant_id,
            exc_info=True,
        )
        return None
    if found[W2V_LICENCE_APP_KEY] is None:
        return None
    return TenantLicences(
        license_ids={k: v for k, v in found.items() if v is not None},
        refused={k: NOT_CONFIRMED_AFTER_TIMEOUT for k, v in found.items() if v is None},
    )


async def ensure_tenant_licences(
    licenses: LicensesClient, *, tenant_id: UUID | str
) -> TenantLicences:
    """Ensure ``tenant_id`` holds the tenant-level ``w2v`` bundle licences, held by
    ``svc:<tenant>:<app>``. Idempotent. Raises ``LicensesClientError`` on an
    upstream failure; refused apps come back in ``refused`` (callers decide).
    A client-side timeout is re-read once (:func:`_reread_after_timeout`) and
    re-raised only when licences did not finish."""
    plans = await _plan_preference(licenses, tenant_id=tenant_id)
    try:
        result = await licenses.ensure_tenant_bundle(tenant_id, W2V_BUNDLE_KEY, plans=plans)
    except httpx.TimeoutException as exc:
        reread = await _reread_after_timeout(licenses, tenant_id=tenant_id)
        if reread is None:
            raise
        _log.warning(
            "licences bundle ensure for tenant=%s timed out client-side (%s); the "
            "re-read found the tenant-level licences, so licences finished: %s",
            tenant_id,
            type(exc).__name__,
            reread.license_ids,
        )
        return reread
    license_ids = {key: app.license_id for key, app in result.apps.items() if app.license_id}
    refused = {r.app: r.reason for r in result.refused}
    for key, reason in refused.items():
        _log.warning(
            "licences refused tenant-level %s licence for tenant=%s (holder %s): %s",
            key,
            tenant_id,
            tenant_licence_subject(tenant_id, key),
            reason,
        )
    _log.info(
        "tenant-level licences ensured for tenant=%s: %s",
        tenant_id,
        {k: f"{v} ({result.apps[k].decision})" for k, v in license_ids.items()},
    )
    return TenantLicences(license_ids=license_ids, refused=refused)


async def ensure_active_license(
    licenses: LicensesClient, *, app_key: str, tenant_id: UUID | str
) -> str:
    """The tenant's active tenant-level ``app_key`` licence id, ensuring it first.

    Raises :class:`TenantLicenceRefusedError` when licences refused that app (or
    did not ensure it), ``LicensesClientError`` on an upstream failure."""
    ensured = await ensure_tenant_licences(licenses, tenant_id=tenant_id)
    license_id = ensured.license_ids.get(app_key)
    if license_id is None:
        raise TenantLicenceRefusedError(
            tenant_id, app_key, ensured.refused.get(app_key, "not_in_bundle")
        )
    return license_id


class TenantNotRegisteredError(RuntimeError):
    """The tenant has no w2v ``tenants`` row, so there is nowhere to record its
    ``licence_ref``. Nothing was written in w2v."""

    def __init__(self, tenant_id: UUID | str) -> None:
        super().__init__(f"tenant {tenant_id} has no w2v registry row")


async def record_ensured_licence_ref(
    session: AsyncSession, tenant_id: UUID, licence_id: str
) -> UUID:
    """Record the ensured tenant-level ``w2v`` licence as the tenant's
    ``licence_ref`` and return the tenant's ``licence_ref`` afterwards.

    A STORED licence token owns ``licence_ref`` (only the token path, lease +
    CAS, may change it) — then that ref is kept and returned. Flush only; the
    caller commits. Raises :class:`TenantNotRegisteredError` (no row) and
    :class:`TenantLicenceRefusedError` for a licence id that is not a UUID."""
    from src.database.repositories.tenant_licence_repository import TenantLicenceRepository

    try:
        ref = UUID(licence_id)
    except ValueError as exc:
        raise TenantLicenceRefusedError(
            tenant_id, W2V_LICENCE_APP_KEY, "malformed_licence_id"
        ) from exc
    repo = TenantLicenceRepository(session)
    if await repo.record_fallback_ref(tenant_id, ref):
        return ref
    current = await repo.stored(tenant_id)
    if current is None:
        raise TenantNotRegisteredError(tenant_id)
    # Token-owned (or under a licence-change lease): the token's licence stands.
    return current.licence_ref if current.licence_ref is not None else ref


async def ensure_tenant_licence_ref(
    session: AsyncSession, licenses: LicensesClient, *, tenant_id: UUID
) -> UUID:
    """Ensure the tenant's ``w2v`` bundle (idempotent) and record its ``w2v``
    licence as ``licence_ref``. Never returns None: a refused app raises
    :class:`TenantLicenceRefusedError`, an upstream failure the client's error.
    Flush only; the caller commits."""
    licence_id = await ensure_active_license(
        licenses, app_key=W2V_LICENCE_APP_KEY, tenant_id=tenant_id
    )
    return await record_ensured_licence_ref(session, tenant_id, licence_id)


# ─── New-tenant provisioning: licences + LLM access posture ─────────────────

#: The postures an onboarding path (provision, early-access approve / invite) may
#: give a NEW tenant — all three (owner decision 2026-10-22: the posture is decided
#: at provisioning, CommonPractice-style). ``byo_required`` = bring every model
#: provider; ``platform_only`` = platform models only.
NewTenantLlmAccess = Literal["local_allowed", "byo_required", "platform_only"]
NEW_TENANT_LLM_ACCESS_MODES: Final[tuple[NewTenantLlmAccess, ...]] = (
    "local_allowed",
    "byo_required",
    "platform_only",
)


@dataclass(frozen=True, slots=True)
class LlmAccessStep:
    """The outcome of setting a new tenant's LLM access. ``failed`` ⇒ the tenant
    is still on licences' ``byo_required`` default (runs blocked); ``detail``
    says why (licences' status + machine code, never its body). Retry with
    :func:`set_new_tenant_llm_access`."""

    mode: NewTenantLlmAccess
    status: Literal["ok", "failed"]
    detail: str | None = None

    def as_record(self) -> dict[str, str | None]:
        return {"mode": self.mode, "status": self.status, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ProvisionedTenantLicences:
    licences: TenantLicences
    llm_access: LlmAccessStep


def _require_new_tenant_mode(llm_access: str) -> NewTenantLlmAccess:
    if llm_access not in NEW_TENANT_LLM_ACCESS_MODES:
        raise ValueError(
            f"llm_access {llm_access!r} is not allowed for a new tenant "
            f"(one of {NEW_TENANT_LLM_ACCESS_MODES})"
        )
    return llm_access


#: Backoff between posture-write attempts: 3 attempts, ~1.5s of sleep at most, so
#: a flaky licences write is absorbed inside the provision request's budget.
LLM_ACCESS_RETRY_BACKOFF_SECONDS: Final[tuple[float, ...]] = (0.5, 1.0)
#: Indirection so tests do not really sleep.
_sleep = asyncio.sleep


def _is_transient(exc: Exception) -> bool:
    """A licences REFUSAL (4xx other than 429) is deterministic — retrying it only
    burns the request budget. Everything else (5xx, 429, transport) may pass."""
    if isinstance(exc, LicensesClientError):
        code = exc.status_code
        return code is None or code == 429 or code >= 500
    return True


def _failure_detail(exc: Exception) -> str:
    if isinstance(exc, LicensesClientError):
        return f"licences answered HTTP {exc.status_code}" + (f" ({exc.code})" if exc.code else "")
    return type(exc).__name__


async def set_new_tenant_llm_access(
    licenses: LicensesClient,
    *,
    tenant_id: UUID,
    llm_access: NewTenantLlmAccess = "local_allowed",
) -> LlmAccessStep:
    """Set a new tenant's LLM access (licences per-app ``w2v`` override). Never
    raises for a licences failure — it is RETURNED as ``status='failed'`` so a
    caller holding other side effects (one-time tokens) can report and retry it.
    A transient failure is retried (:data:`LLM_ACCESS_RETRY_BACKOFF_SECONDS`, the
    write is an idempotent PUT); a licences refusal is not.
    Raises ``ValueError`` for a mode outside :data:`NEW_TENANT_LLM_ACCESS_MODES`."""
    from src.services.tenant_llm_access import apply_tenant_llm_access

    mode = _require_new_tenant_mode(llm_access)
    backoffs = LLM_ACCESS_RETRY_BACKOFF_SECONDS
    attempts = len(backoffs) + 1
    for attempt in range(1, attempts + 1):
        try:
            await apply_tenant_llm_access(tenant_id, mode, licenses_client=licenses)
        except Exception as exc:  # noqa: BLE001 — returned as a failed step, logged
            detail = _failure_detail(exc)
            if attempt < attempts and _is_transient(exc):
                _log.warning(
                    "tenant=%s: setting llm_access=%s failed (%s), attempt %d/%d; retrying",
                    tenant_id,
                    mode,
                    detail,
                    attempt,
                    attempts,
                )
                await _sleep(backoffs[attempt - 1])
                continue
            _log.error(
                "tenant=%s: setting llm_access=%s FAILED after %d attempt(s) (%s); the tenant "
                "stays on licences' byo_required default until it is retried",
                tenant_id,
                mode,
                attempt,
                detail,
            )
            return LlmAccessStep(mode=mode, status="failed", detail=detail)
        return LlmAccessStep(mode=mode, status="ok")
    raise AssertionError("unreachable")  # pragma: no cover


async def provision_tenant_licences(
    licenses: LicensesClient,
    *,
    tenant_id: UUID,
    llm_access: NewTenantLlmAccess = "local_allowed",
) -> ProvisionedTenantLicences:
    """The NEW-tenant licences step shared by super-admin provision and the
    early-access approve: :func:`ensure_tenant_licences`, then
    :func:`set_new_tenant_llm_access` (AFTER the licences exist). The mode is
    validated before any side effect (``ValueError``). An ensure failure raises
    as before; a posture failure is returned in ``llm_access``, never raised."""
    _require_new_tenant_mode(llm_access)
    ensured = await ensure_tenant_licences(licenses, tenant_id=tenant_id)
    step = await set_new_tenant_llm_access(licenses, tenant_id=tenant_id, llm_access=llm_access)
    return ProvisionedTenantLicences(licences=ensured, llm_access=step)
