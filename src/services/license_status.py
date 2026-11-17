"""App-scoped license STATUS derivation for the w2v SPA (backs GET /api/license/status).

Mirrors fr's ``backend/src/lib/license-status.ts``. The per-request license-token
auth (``w2v_common.auth.middleware.AuthMiddleware``) is the real enforcer; this
module adds the detail the frontend needs to render an accurate experience —
distinguish active / expiring_soon / expired / revoked / none and surface the
expiry date — so an expired tenant gets a clear "renew" block instead of a
broken shell.

Two-call shape (same as fr): the licenses service ``GET /licenses/active`` returns
200 ONLY for a currently-active license (with ``expires_at``); expired, revoked
and never-had all collapse to 404. To tell those apart we fall back to the
subject's inventory (``GET /licenses/user/{sub}``) and read each record's
``status`` / ``expires_at`` / ``revoked_at``.

``GET /licenses/active`` is gated by the licenses service's
``require_internal_or_reader`` (``ACTIVE_LICENSE_READ_ROLES`` =
``INTERNAL_SERVICE_ROLES`` — ``service-internal`` / ``onboarding-internal`` /
``license-admin`` — plus ``license-reader``). A least-privilege SA holding only
``license-reader`` IS admitted; ``license-reader`` does NOT open the
``/internal/licenses*`` listing routes. The m4 cell client ``w2v-cell-m4``
(realm ``secure-apps``) holds exactly ``identity-inviter`` + ``license-reader``;
licenses migration 085 registers it as an ``api_client`` with owner ``w2v`` and
``cross_owner=false``, so its ``X-Tenant-Id`` retarget is honoured only for
w2v-owned tenants or tenants with an active grant to w2v — it can read nothing
else. Historically the route required an internal-service role and the w2v SA
got **403 on every call** (the gate saw ``unknown`` and enforced nothing); the
fallback below stays as a guard against a regression of that admission:
rather than collapse to ``unknown``, a 401/403 derives entitlement from the
subject INVENTORY route (``GET /licenses/user/{sub}``), which IS authorized for ``license-reader``. That
route can legitimately yield ``active``/``expiring_soon`` — see
:func:`derive_inventory_state`. Do NOT route a 401/403 into
:func:`derive_inactive_state`: it has no active exit, so a paying customer would
fall out as ``none`` and get 402'd.

On any upstream error this returns ``state='unknown'`` — an explicitly
INDETERMINATE answer, not an entitled one. ``unknown`` is a *reporting* state:
``GET /api/license/status`` renders it as "we could not check", and the caches
above deliberately never store it. What it must NEVER do is entitle. The
enforcement chokepoint (``api.routes._common._enforce_license_state``) treats
``unknown`` as a DENY and answers **503 ``license_unavailable``** — distinct
from the 402 ``license_<state>`` a real licence refusal earns, so a caller can
tell "you have no licence" from "we could not check". Availability is preserved
by the layers in front of the deny, not by admitting unverified traffic: the
request-local fast path (an already-introspected token needs no hop), the 60s
positive cache, and the 900s stale-while-error window that serves last-known-good
across a blip. Fail-CLOSED is the owner ruling — unsanctioned usage is blocked.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Literal, Protocol
from uuid import UUID

from cachetools import TTLCache

from src.clients.licenses_client import LicensesClientError, TenantNotOnboardedError
from src.clients.schemas.licenses import ActiveLicenseView, LicenseView


class _LicensesReader(Protocol):
    """The read-only slice of ``LicensesClient`` this module consults.

    Typed as a Protocol so the pure-derivation tests can pass a lightweight fake
    without constructing a real HTTP client, while still rejecting a client that
    is missing either method.
    """

    async def get_active_license(
        self, *, tenant_id: str, app_key: str
    ) -> ActiveLicenseView | None: ...

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str) -> list[LicenseView]: ...


_log = logging.getLogger("w2v_orchestrator.license_status")

LicenseStatusState = Literal[
    "active", "expiring_soon", "expired", "revoked", "none", "not_onboarded", "unknown"
]

#: Licences answered 403 ``tenant_not_accessible``: the tenant is owned by another
#: app that granted w2v no access — it is NOT ONBOARDED to w2v. A definitive,
#: cached answer (never ``unknown``, never retried), kept apart from ``none`` so
#: the gate can tell the caller "this tenant is not onboarded" (403) instead of
#: "buy a licence" (402).
NOT_ONBOARDED: Final = "not_onboarded"

#: Days before expiry at which an active license is flagged expiring-soon.
EXPIRING_SOON_DAYS = 30

#: The ONLY states that entitle. Public because the enforcement chokepoint
#: (``api.routes._common._enforce_license_state``) gates on it as an
#: ALLOWLIST — a state absent from this set can never be silently admitted,
#: so ``has_license`` and the 402/503 verdict can never drift apart.
ENTITLED_STATES: frozenset[str] = frozenset({"active", "expiring_soon"})

#: Upstream statuses that mean "this service account may not call that route" —
#: a PERMANENT misconfiguration, not a transient blip. These divert to the
#: inventory fallback (and log loudly) instead of degrading to ``unknown``.
_AUTHZ_STATUS_CODES: frozenset[int] = frozenset({401, 403})

#: A licenses ``status`` value that still denotes a live entitlement. Anything
#: else (``expired``/``revoked``/``suspended``/``pending``/…) is NOT active — an
#: unrecognised status must never entitle, so this is an allowlist.
_LIVE_RECORD_STATUSES: frozenset[str] = frozenset({"active"})

#: Positive/negative TTL for a derived status, and the longer stale-while-error
#: window. Mirrors the proven shape in ``w2v_common.auth.middleware``: a hit
#: serves immediately, a miss is single-flighted, and an upstream failure with a
#: warm ``stale`` entry serves last-known-good rather than ``unknown``.
_STATUS_TTL_SECONDS = 60.0
_STALE_TTL_SECONDS = 900.0
_CACHE_MAXSIZE = 2048

#: ``(tenant_id, app_key, subject_id)`` — subject participates because the
#: inventory fallback merges the caller's own subject inventory, so two subjects in one tenant can
#: legitimately derive different states.
type _StatusKey = tuple[str, str, str | None]
# A licenses-service wire record (``LicenseView.model_dump()``); consumed by
# .get() throughout, so a plain wire dict rather than a re-parsed model.
type _Record = dict[str, Any]

_status_cache: TTLCache[_StatusKey, LicenseStatusResult] = TTLCache(
    maxsize=_CACHE_MAXSIZE, ttl=_STATUS_TTL_SECONDS
)
_status_stale_cache: TTLCache[_StatusKey, LicenseStatusResult] = TTLCache(
    maxsize=_CACHE_MAXSIZE, ttl=_STALE_TTL_SECONDS
)
_status_locks: dict[_StatusKey, asyncio.Lock] = {}


def clear_license_status_cache() -> None:
    """Drop every cached status. Test seam + a safety valve for a live flag flip."""
    _status_cache.clear()
    _status_stale_cache.clear()
    _status_locks.clear()


def invalidate_license_status(
    *, tenant_id: str | None, app_key: str, subject_id: str | None
) -> None:
    """Evict the cached status for one identity after a KNOWN state change.

    Auto-issue mints a license mid-request; without this eviction the immediate
    re-resolve would read back the pre-mint ``none`` from the 60s cache and the
    SPA would still see a hard block. A mint (or any deliberate mutation) must
    drop both the fresh and the stale entry so the next resolve hits upstream.
    """
    key: _StatusKey = (str(tenant_id), app_key, str(subject_id) if subject_id else None)
    _status_cache.pop(key, None)
    _status_stale_cache.pop(key, None)


def _status_code_of(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


@dataclass(frozen=True)
class LicenseStatusResult:
    state: LicenseStatusState
    has_license: bool
    expires_at: str | None
    days_remaining: int | None
    plan_code: str | None


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_iso(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError, TypeError:
        return None
    # Treat a naive timestamp as UTC so the arithmetic below stays tz-aware.
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _days_until(iso: str, now: datetime) -> int | None:
    """Whole days until ``iso`` (negative = days since expiry). None if unparseable."""
    dt = _parse_iso(iso)
    if dt is None:
        return None
    return math.ceil((dt - now).total_seconds() / 86400)


def _result(
    state: LicenseStatusState,
    *,
    expires_at: str | None = None,
    days_remaining: int | None = None,
    plan_code: str | None = None,
) -> LicenseStatusResult:
    return LicenseStatusResult(
        state=state,
        has_license=state in ENTITLED_STATES,
        expires_at=expires_at,
        days_remaining=days_remaining,
        plan_code=plan_code,
    )


def _plan_code(record: _Record) -> str | None:
    plan = record.get("plan")
    return plan.get("code") if isinstance(plan, dict) else None


def derive_active_state(
    detail: _Record,
    now: datetime,
    expiring_soon_days: int = EXPIRING_SOON_DAYS,
) -> LicenseStatusResult:
    """Pure: derive status from an ACTIVE license detail (200 from /licenses/active).

    A 200 whose expiry is already past (stale 200 / clock skew) is treated as
    ``expired`` — not a dismissible ``expiring_soon`` banner — mirroring fr's
    negative-days hardening.
    """
    expires_at = detail.get("expires_at")
    days_remaining = _days_until(expires_at, now) if expires_at else None
    if days_remaining is None:
        state: LicenseStatusState = "active"
    elif days_remaining < 0:
        state = "expired"
    elif days_remaining <= expiring_soon_days:
        state = "expiring_soon"
    else:
        state = "active"
    return _result(
        state,
        expires_at=expires_at,
        days_remaining=days_remaining,
        plan_code=_plan_code(detail),
    )


def derive_inactive_state(
    records: list[_Record],
    app_key: str,
    now: datetime,
) -> LicenseStatusResult:
    """Pure: derive status with NO active license (404), from the subject's inventory.

    Precedence: revoked > expired > none (pending / future-dated count as none —
    there is no active entitlement yet).
    """
    app_records = [r for r in records if (r.get("app") or {}).get("key") == app_key]
    if not app_records:
        return _result("none")

    revoked = next(
        (r for r in app_records if r.get("status") == "revoked" or r.get("revoked_at")),
        None,
    )
    if revoked is not None:
        exp = revoked.get("expires_at")
        return _result(
            "revoked",
            expires_at=exp,
            days_remaining=_days_until(exp, now) if exp else None,
            plan_code=_plan_code(revoked),
        )

    def _is_expired(r: _Record) -> bool:
        if r.get("status") == "expired":
            return True
        exp = r.get("expires_at")
        if not exp:
            return False
        dt = _parse_iso(exp)
        return dt is not None and dt <= now

    expired = sorted(
        (r for r in app_records if _is_expired(r)),
        key=lambda r: _parse_iso(r.get("expires_at") or "") or datetime.min.replace(tzinfo=UTC),
        reverse=True,  # most-recently-expired first
    )
    if expired:
        exp = expired[0].get("expires_at")
        return _result(
            "expired",
            expires_at=exp,
            days_remaining=_days_until(exp, now) if exp else None,
            plan_code=_plan_code(expired[0]),
        )

    # A record exists but is neither active (we got a 404) nor expired/revoked —
    # e.g. pending / future-dated. No active entitlement → none.
    return _result("none")


def _in_gate_scope(record: _Record, app_key: str, tenant_id: str) -> bool:
    """Does ``record`` entitle THIS (tenant, app)?

    ``GET /licenses/user/{sub}`` deliberately returns the subject's licenses
    across **every tenant** (licenses/main.py — "List all licenses for this
    subject regardless of tenant_id"), so an app-key-only filter would let a
    license the subject holds in tenant A entitle their request in tenant B.
    Both dimensions are mandatory. A record with no ``tenant_id`` cannot be
    proven in-scope and therefore never entitles.
    """
    if (record.get("app") or {}).get("key") != app_key:
        return False
    record_tenant = record.get("tenant_id")
    return bool(record_tenant) and str(record_tenant) == str(tenant_id)


def _is_live_record(record: _Record, now: datetime) -> bool:
    """Is this inventory record a currently-live entitlement?

    Conservative on every axis: revoked loses, an unrecognised ``status`` loses,
    a not-yet-started license loses, and an expiry at/before ``now`` loses. A
    record with no ``expires_at`` is perpetual and passes (the same reading
    :func:`derive_active_state` gives a 200 with a null expiry).
    """
    if record.get("revoked_at") or record.get("status") == "revoked":
        return False
    status = record.get("status")
    if status is not None and status not in _LIVE_RECORD_STATUSES:
        return False
    starts_at = record.get("starts_at")
    if starts_at:
        starts = _parse_iso(starts_at)
        if starts is not None and starts > now:
            return False  # future-dated: no entitlement YET
    expires_at = record.get("expires_at")
    if expires_at:
        expires = _parse_iso(expires_at)
        # Unparseable expiry → cannot prove it is in the future → not live.
        if expires is None or expires <= now:
            return False
    return True


def _expiry_rank(record: _Record) -> tuple[int, datetime]:
    """Sort key picking the longest-lived record; perpetual (no expiry) wins."""
    expires_at = record.get("expires_at")
    if not expires_at:
        return (1, datetime.max.replace(tzinfo=UTC))
    return (0, _parse_iso(expires_at) or datetime.min.replace(tzinfo=UTC))


def derive_inventory_state(
    records: list[_Record],
    app_key: str,
    tenant_id: str,
    now: datetime,
    expiring_soon_days: int = EXPIRING_SOON_DAYS,
) -> LicenseStatusResult:
    """Pure: derive status for ``(tenant_id, app_key)`` from the subject inventory.

    Unlike :func:`derive_inactive_state` (which presumes a 404 from
    ``/licenses/active`` already PROVED there is no active license, and so has no
    active exit), this is used when the primary route was never consulted —
    it answered 401/403. It must therefore be able to return ``active`` /
    ``expiring_soon``, otherwise a fully-paid tenant derives ``none`` and is
    402'd. It routes the winning record through :func:`derive_active_state` so
    the expiring-soon window and the negative-days hardening are applied by the
    SAME code that governs the 200 path — the two can never drift.

    Falls through to :func:`derive_inactive_state` (over the tenant-scoped
    subset) when nothing is live, so revoked / expired / none still resolve.
    """
    scoped = [r for r in records if _in_gate_scope(r, app_key, tenant_id)]
    live = [r for r in scoped if _is_live_record(r, now)]
    if live:
        return derive_active_state(max(live, key=_expiry_rank), now, expiring_soon_days)
    return derive_inactive_state(scoped, app_key, now)


#: Tenant-level (non-member) licences are held by the deterministic service subject
#: ``svc:<tenant-uuid>:<app-key>`` (licenses ``models.service_subject_sub`` /
#: ``scripts/provision_tenant_bundle.py``). Licences are TENANT-level (fr is the
#: reference) — the caller's own subject usually holds nothing.
_SERVICE_SUBJECT_PREFIX = "svc:"


def _tenant_subject(tenant_id: str, app_key: str) -> str | None:
    """``svc:<canonical-tenant-uuid>:<app-key>``, or None when ``tenant_id`` is not a UUID."""
    try:
        tenant = str(UUID(str(tenant_id)))
    except ValueError:
        return None
    return f"{_SERVICE_SUBJECT_PREFIX}{tenant}:{app_key}"


#: The licenses service's body for an unknown subject (``main.py``
#: ``list_user_licenses``: ``HTTPException(404, detail="Subject not found")``).
#: Only THIS 404 is authoritative absence; a bare route-missing 404
#: (``{"detail":"Not Found"}``) is a failure and must fail soft to ``unknown``.
_SUBJECT_NOT_FOUND_DETAIL = "Subject not found"


def _is_subject_not_found(exc: LicensesClientError) -> bool:
    return _status_code_of(exc) == 404 and _SUBJECT_NOT_FOUND_DETAIL in str(exc)


async def _fetch_inventory(
    client: _LicensesReader, subject_id: str | None, tenant_id: str, app_key: str
) -> list[_Record]:
    """Tenant-level (+ caller's own) inventory as plain wire dicts. Raises on upstream failure.

    Keyed by TENANT first: the tenant's ``svc:<tenant>:<app>`` holder is read, so a
    tenant whose only licence is tenant-level still resolves (previously it read
    only ``/licenses/user/{sub}`` and derived ``none`` ⇒ 402). A 404 whose body is
    the licenses service's ``Subject not found`` — no tenant-level licence was ever
    minted — is an EMPTY tenant inventory, but only authoritative when the caller's
    own inventory is also read (no subject ⇒ raise ⇒ ``unknown``, since licences
    are still mostly minted per member); any other 404 (route missing, proxy) is a
    failure. The
    caller's own subject inventory (legacy per-member licences) is merged in when
    known; its failures still raise (⇒ ``unknown`` fail-soft), as before.

    When neither key is usable (no subject AND a non-UUID tenant) we cannot look
    the entitlement up at all — raising routes that to ``unknown`` instead of an
    empty list that would derive ``none`` and 402 the caller.
    """
    tenant_subject = _tenant_subject(tenant_id, app_key)
    if tenant_subject is None and not subject_id:
        raise LicensesClientError("cannot derive entitlement without a tenant or subject")
    records: list[LicenseView] = []
    if tenant_subject is not None:
        try:
            records.extend(await client.list_user_licenses(tenant_subject, tenant_id=tenant_id))
        except LicensesClientError as exc:
            if not _is_subject_not_found(exc):
                raise
            if not subject_id:
                # No tenant-level holder AND no caller subject to consult. Licence
                # minting is still largely per-MEMBER, so a missing svc holder does
                # not prove the tenant is unlicensed — its licences may sit on its
                # members, which we cannot see from here (the S2S require_license
                # path). Absence is only authoritative alongside an empty caller
                # inventory; without one this is "cannot tell" ⇒ unknown, not 402.
                raise LicensesClientError(
                    "no tenant-level licence holder and no subject to consult"
                ) from exc
    if subject_id:
        records.extend(await client.list_user_licenses(subject_id, tenant_id=tenant_id))
    return [r.model_dump() for r in records]


async def _resolve_uncached(
    *,
    client: _LicensesReader,
    app_key: str,
    tenant_id: str,
    subject_id: str | None,
    when: datetime,
) -> LicenseStatusResult:
    """One full upstream resolution. Raises on failure (caller fails soft)."""
    try:
        return await _resolve_upstream(
            client=client, app_key=app_key, tenant_id=tenant_id, subject_id=subject_id, when=when
        )
    except TenantNotOnboardedError:
        # 403 ``tenant_not_accessible`` (licences migration 050) from ANY of the
        # reads below — ``/licenses/active`` or the inventory: the tenant is owned
        # by another app that has not granted w2v access. That is an ANSWER — the
        # tenant is not onboarded to w2v — not a misconfigured service account (no
        # inventory fallback: it would be refused the same way) and not an outage
        # (no 'unknown'). Definitive ⇒ cached.
        _log.info(
            "tenant_not_onboarded_to_w2v: licences reports no w2v grant on tenant=%s "
            "app_key=%s — state 'not_onboarded'",
            tenant_id,
            app_key,
        )
        return _result(NOT_ONBOARDED)


async def _resolve_upstream(
    *,
    client: _LicensesReader,
    app_key: str,
    tenant_id: str,
    subject_id: str | None,
    when: datetime,
) -> LicenseStatusResult:
    try:
        detail = await client.get_active_license(tenant_id=tenant_id, app_key=app_key)
    except TenantNotOnboardedError:
        raise
    except LicensesClientError as exc:
        status_code = _status_code_of(exc)
        if status_code not in _AUTHZ_STATUS_CODES:
            raise
        # PERMANENT misconfiguration, not a blip: the SA is not permitted on
        # /licenses/active. ERROR (not WARNING) with a distinct, greppable
        # marker — as a WARNING this was indistinguishable from a transient
        # timeout and the gate sat inert for days without anyone noticing.
        _log.error(
            "license_gate_misconfigured: /licenses/active rejected the w2v service "
            "account with %s for tenant=%s app_key=%s. The license gate CANNOT use "
            "its primary route and is deriving entitlement from the tenant-level "
            "inventory instead. Fix: grant the SA an internal-service role on the "
            "licenses service (service-internal / license-admin).",
            status_code,
            tenant_id,
            app_key,
        )
        records = await _fetch_inventory(client, subject_id, tenant_id, app_key)
        return derive_inventory_state(records, app_key, tenant_id, when)

    if detail is not None and detail.conflict:
        # Multiple active licenses (data conflict) — entitled, expiry unknown.
        return _result("active")
    if detail is not None:
        # derive_* are pure dict-based gate logic; hand them the wire dict.
        return derive_active_state(detail.model_dump(), when)
    # No active license (404) — an AUTHORITATIVE "not active". Refine to
    # expired / revoked / none from the subject's inventory when we know who
    # they are. derive_inactive_state (no active exit) is correct HERE and only
    # here, precisely because the 404 already ruled active out.
    records = await client.list_user_licenses(subject_id, tenant_id=tenant_id) if subject_id else []
    return derive_inactive_state([r.model_dump() for r in records], app_key, when)


async def _resolve_and_cache(
    *,
    client: _LicensesReader,
    app_key: str,
    tenant_id: str,
    subject_id: str | None,
    when: datetime,
    cache_key: _StatusKey | None,
) -> LicenseStatusResult:
    """Resolve once, fail soft, and populate the caches.

    ``cache_key=None`` means "do not read or write the cache" — used by the
    pinned-``now`` seam so a time-travelling caller can neither read an entry
    derived at a different instant nor poison one for real traffic.
    """
    try:
        result = await _resolve_uncached(
            client=client,
            app_key=app_key,
            tenant_id=tenant_id,
            subject_id=subject_id,
            when=when,
        )
    except Exception:  # noqa: BLE001 — fail-soft, never hard-block
        stale = _status_stale_cache.get(cache_key) if cache_key is not None else None
        if stale is not None:
            # Stale-while-error: this tenant resolved cleanly moments ago, so a
            # transient failure must not downgrade them to 'unknown'.
            _log.warning(
                "license-status lookup failed for tenant=%s app=%s; serving stale state=%s",
                tenant_id,
                app_key,
                stale.state,
                exc_info=True,
            )
            return stale
        _log.warning(
            "license-status lookup failed for tenant=%s app=%s; reporting state='unknown'",
            tenant_id,
            app_key,
            exc_info=True,
        )
        return _result("unknown")

    # Only DEFINITIVE results are cached. 'unknown' is an error state: caching
    # it would keep serving a degraded answer after the upstream recovers.
    if cache_key is not None and result.state != "unknown":
        _status_cache[cache_key] = result
        _status_stale_cache[cache_key] = result
    return result


async def get_license_status(
    *,
    client: _LicensesReader,
    app_key: str,
    tenant_id: str | None,
    subject_id: str | None,
    gate_enabled: bool,
    now: datetime | None = None,
) -> LicenseStatusResult:
    """Resolve the rich license status for the current request's tenant + subject.

    - gate disabled ⇒ everyone entitled (``state='active'``) — zero functional
      change until the gate is flipped on.
    - no tenant ⇒ ``none`` (the route's own auth gate handles unauthenticated
      callers; this is only the status hint).
    - 401/403 from ``/licenses/active`` ⇒ derive from the tenant-level (+ subject) inventory
      (which CAN yield active) and log ``license_gate_misconfigured``.
    - any other upstream error ⇒ last-known-good if warm, else ``unknown``
      (non-blocking; the per-request auth is the real enforcer).

    Results are cached for ``_STATUS_TTL_SECONDS`` per
    ``(tenant, app_key, subject)`` and misses are single-flighted — this runs
    synchronously on EVERY gated request, so an uncached implementation puts a
    licenses round-trip in front of the entire tenant-scoped API.

    Passing an explicit ``now`` bypasses the cache entirely: it is the
    pure-derivation test seam, and a caller pinning time must not read (or
    poison) an entry derived at a different instant.

    ``client`` is a :class:`~src.clients.licenses_client.LicensesClient` exposing
    ``get_active_license(tenant_id, app_key)`` and
    ``list_user_licenses(subject_id, tenant_id)``.
    """
    if not gate_enabled:
        return _result("active")
    if not tenant_id:
        return _result("none")

    cache_key: _StatusKey = (str(tenant_id), app_key, str(subject_id) if subject_id else None)

    if now is not None:
        return await _resolve_and_cache(
            client=client,
            app_key=app_key,
            tenant_id=tenant_id,
            subject_id=subject_id,
            when=now,
            cache_key=None,
        )

    hit = _status_cache.get(cache_key)
    if hit is not None:
        return hit

    # Single-flight: a concurrent burst for one cold tenant collapses into ONE
    # upstream resolution instead of stampeding the licenses service. Re-check
    # inside the lock — a prior waiter may have populated the cache.
    if len(_status_locks) > _CACHE_MAXSIZE:
        # Unbounded growth guard. Dropping a held lock only risks a duplicate
        # upstream call, never a wrong answer.
        _status_locks.clear()
    lock = _status_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        hit = _status_cache.get(cache_key)
        if hit is not None:
            return hit
        return await _resolve_and_cache(
            client=client,
            app_key=app_key,
            tenant_id=tenant_id,
            subject_id=subject_id,
            when=_now(),
            cache_key=cache_key,
        )
