"""The tenant's BYO posture (licences ``byo_mode``) — the source of truth for #1106.

Owner ruling 2026-10-15: the platform Mac GPUs (mflux image, kokoro/kitten TTS,
whisper ASR) serve only the owner's tenants; every other tenant brings its own
provider. Licences holds that as ``byo_mode`` — tenant-wide, with a per-app
override (``tenant_app_byo_mode``) — and w2v reads it for app key ``w2v``:

* ``byo_required`` — never the platform for any model-provider slot: no tenant
  connection ⇒ :class:`SlotNotConfiguredError` (preflight + resolver).
* ``local_allowed`` — tenant connection if any, else the platform default.
* ``platform_only`` — the platform default; tenant connections ignored + unwritable.

The per-slot table is ``providers/access.py`` (driven by the registry's
``byo_governed`` flag, owner decision 2026-10-22).

LAST-KNOWN ON OUTAGE (CommonPractice's posture cache): a licences outage, a non-200
or an unparseable body falls back to the tenant's LAST successfully-read posture in
this process (logged), so a licences blip does not refuse every model call of every
tenant. Only a tenant NEVER seen by this process fails closed: it raises
:class:`ByoPostureUnavailableError` (transient) — never ``local_allowed``. A posture
this code does not know is never cached, so it can never become a last-known value. A 404 ``tenant_not_found_in_registry`` is licences' own "not
provisioned" answer and maps to ``byo_required``, exactly as licences' introspect
path treats a tenant with no row ("no row ⇒ unprovisioned ⇒ no platform GPU").

Only successful answers are cached (per process, ``BYO_POSTURE_CACHE_SECONDS``,
default 60) — a failure is retried on the next call, never remembered.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Final, Literal, Protocol
from uuid import UUID

from src.services.providers.errors import ByoPostureUnavailableError

if TYPE_CHECKING:
    # Runtime imports are local: ``src.clients`` (package init) imports the slot
    # payload builder, which imports this package — a top-level import is circular.
    from src.clients.licenses_client import LicensesClient

__all__ = [
    "BYO_APP_KEY",
    "ByoMode",
    "ByoPostureReader",
    "LicencesByoPosture",
    "StaticByoPosture",
    "clear_byo_posture_cache",
    "get_byo_posture",
    "invalidate_byo_posture",
]

logger = logging.getLogger(__name__)

ByoMode = Literal["local_allowed", "byo_required", "platform_only"]
_KNOWN: Final[dict[str, ByoMode]] = {
    "local_allowed": "local_allowed",
    "byo_required": "byo_required",
    "platform_only": "platform_only",
}

#: The licences app key whose per-app override applies to w2v.
BYO_APP_KEY: Final = "w2v"
_NOT_IN_REGISTRY: Final = "tenant_not_found_in_registry"


class ByoPostureReader(Protocol):
    """What the resolver needs: the tenant's posture, or a raise (never a guess)."""

    async def byo_mode(self, tenant_id: UUID, *, slot: str) -> ByoMode: ...


class StaticByoPosture:
    """A fixed posture — for tests and for callers that already resolved it."""

    def __init__(self, mode: ByoMode) -> None:
        self.mode: ByoMode = mode
        self.calls: list[UUID] = []

    async def byo_mode(self, tenant_id: UUID, *, slot: str) -> ByoMode:
        self.calls.append(tenant_id)
        return self.mode


def _cache_seconds() -> float:
    raw = os.getenv("BYO_POSTURE_CACHE_SECONDS", "60")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 60.0


#: Keyed by (tenant, app key): the per-app override makes the answer app-specific.
_cache: dict[tuple[UUID, str], tuple[ByoMode, float]] = {}
#: The last SUCCESSFUL answer per (tenant, app key), no TTL — the outage fallback.
_last_known: dict[tuple[UUID, str], ByoMode] = {}


def clear_byo_posture_cache() -> None:
    """Drop every cached posture (tests; an operator flip is otherwise ≤ TTL late)."""
    _cache.clear()
    _last_known.clear()


def invalidate_byo_posture(tenant_id: UUID, *, app_key: str = BYO_APP_KEY) -> None:
    """Drop ONE tenant's cached posture — called right after an operator write
    (``apply_tenant_llm_access``) so this process sees the new mode at once.
    Other pods still serve their cached answer for up to the TTL."""
    _cache.pop((tenant_id, app_key), None)
    _last_known.pop((tenant_id, app_key), None)


class LicencesByoPosture:
    """:class:`ByoPostureReader` over ``LicensesClient.get_tenant_byo_mode``."""

    def __init__(self, client: LicensesClient, *, app_key: str = BYO_APP_KEY) -> None:
        self._client = client
        self._app_key = app_key

    async def byo_mode(self, tenant_id: UUID, *, slot: str) -> ByoMode:
        now = time.monotonic()
        key = (tenant_id, self._app_key)
        hit = _cache.get(key)
        if hit is not None and hit[1] > now:
            return hit[0]
        try:
            mode = await self._fetch(tenant_id, slot=slot)
        except ByoPostureUnavailableError as exc:
            last = _last_known.get(key)
            if last is None:
                raise  # never seen: fail closed
            logger.warning(
                "byo posture: licences unavailable for tenant %s (%s); using the last-known "
                "posture %s",
                tenant_id,
                exc.reason,
                last,
            )
            return last
        _last_known[key] = mode
        ttl = _cache_seconds()
        if ttl > 0:
            _cache[key] = (mode, now + ttl)
        return mode

    async def _fetch(self, tenant_id: UUID, *, slot: str) -> ByoMode:
        from src.clients.licenses_client import LicensesClientError

        try:
            view = await self._client.get_tenant_byo_mode(
                tenant_id=tenant_id, app_key=self._app_key
            )
        except LicensesClientError as exc:
            if exc.status_code == 404 and exc.code == _NOT_IN_REGISTRY:
                logger.warning(
                    "byo posture: tenant %s is not in the licences registry; treating as "
                    "byo_required (licences' own reading of an unprovisioned tenant)",
                    tenant_id,
                )
                return "byo_required"
            raise ByoPostureUnavailableError(slot, f"licences answered {exc.status_code}") from exc
        except Exception as exc:
            raise ByoPostureUnavailableError(slot, type(exc).__name__) from exc
        mode = _KNOWN.get(view.byo_mode)
        if mode is None or view.resolution != "row":
            raise ByoPostureUnavailableError(
                slot, f"unusable posture {view.byo_mode!r} (resolution={view.resolution!r})"
            )
        return mode


def get_byo_posture() -> ByoPostureReader:
    """The production reader (licences via the w2v-backend service account)."""
    from w2v_common.discovery import service_url

    from src.clients.licenses_client import LicensesClient
    from src.clients.service_auth import get_service_authenticator

    return LicencesByoPosture(
        LicensesClient(authenticator=get_service_authenticator(), base_url=service_url("licenses"))
    )
