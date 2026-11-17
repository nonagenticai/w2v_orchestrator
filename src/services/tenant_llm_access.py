"""A tenant's LLM/GPU access posture for w2v — read and set by the platform operator.

The posture IS licences' ``byo_mode`` for app key ``w2v`` (see
``src.services.providers.byo_posture`` for how the resolver enforces it):

* ``local_allowed`` — "Platform models allowed": the platform's local models
  (local-llm, kokoro, local ASR, mflux) are the default; the tenant MAY add keys.
* ``platform_only`` — "Local only": platform local models only; own keys disabled.
* ``byo_required`` — "Bring your own keys": no platform fall-through.

Licences' default for a NEW tenant is ``byo_required``, so provisioning (and the
early-access approve path) must set the mode explicitly via
:func:`apply_tenant_llm_access` — or the tenant's first run is blocked.

The write is the PER-APP override (``PUT /tenants/{id}/apps/w2v/byo-mode``), so
the tenant's posture for other apps is untouched. Licences errors propagate as
``LicensesClientError`` (callers decide; the admin route maps 401/403 to 502).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal
from uuid import UUID

from src.clients.licenses_client import LicensesClientError
from src.services.providers.byo_posture import BYO_APP_KEY, ByoMode, invalidate_byo_posture

if TYPE_CHECKING:
    from src.clients.licenses_client import LicensesClient

__all__ = [
    "DEFAULT_LLM_ACCESS",
    "LLM_ACCESS_LABELS",
    "LLM_ACCESS_MODES",
    "LlmAccessSource",
    "TenantLlmAccess",
    "UnknownLlmAccessModeError",
    "apply_tenant_llm_access",
    "read_tenant_llm_access",
]

LLM_ACCESS_MODES: Final[tuple[ByoMode, ...]] = ("local_allowed", "platform_only", "byo_required")
#: The mode a new tenant is born with on the w2v side (owner decision 2026-10-19).
DEFAULT_LLM_ACCESS: Final[ByoMode] = "local_allowed"
LLM_ACCESS_LABELS: Final[dict[str, str]] = {
    "local_allowed": "Platform models allowed",
    "platform_only": "Local only",
    "byo_required": "Bring your own keys",
}

#: Where the effective mode came from: the per-app ``w2v`` override, the
#: tenant-wide value, or licences' "not in registry" default (``byo_required``).
LlmAccessSource = Literal["tenant", "app_override", "default"]
_NOT_IN_REGISTRY: Final = "tenant_not_found_in_registry"


class UnknownLlmAccessModeError(ValueError):
    """A mode outside :data:`LLM_ACCESS_MODES` (refused before any licences call),
    or a licences answer this code does not know."""


@dataclass(frozen=True, slots=True)
class TenantLlmAccess:
    mode: ByoMode
    source: LlmAccessSource
    #: Only on a write: the per-app override the write replaced (None = none existed).
    prior_mode: str | None = None


def _known(mode: str) -> ByoMode:
    if mode not in LLM_ACCESS_MODES:
        raise UnknownLlmAccessModeError(f"unknown llm_access mode {mode!r}")
    return mode


async def read_tenant_llm_access(
    tenant_id: UUID, *, licenses_client: LicensesClient
) -> TenantLlmAccess:
    """The tenant's effective w2v posture and its source. Raises
    ``LicensesClientError`` on any licences failure other than "tenant not in
    the registry" (which is ``byo_required`` / ``default`` — licences' own
    reading of an unprovisioned tenant)."""
    override = await licenses_client.get_tenant_app_byo_override(
        tenant_id=tenant_id, app_key=BYO_APP_KEY
    )
    if override is not None:
        return TenantLlmAccess(mode=_known(override.byo_mode), source="app_override")
    try:
        view = await licenses_client.get_tenant_byo_mode(tenant_id=tenant_id, app_key=BYO_APP_KEY)
    except LicensesClientError as exc:
        if exc.status_code == 404 and exc.code == _NOT_IN_REGISTRY:
            return TenantLlmAccess(mode="byo_required", source="default")
        raise
    return TenantLlmAccess(mode=_known(view.byo_mode), source="tenant")


async def apply_tenant_llm_access(
    tenant_id: UUID, mode: str, *, licenses_client: LicensesClient
) -> TenantLlmAccess:
    """Set the tenant's w2v posture (the per-app override) and drop this
    process's cached posture for it. Raises :class:`UnknownLlmAccessModeError`
    for an unknown ``mode`` (no call made) and ``LicensesClientError`` when
    licences refuses or fails — never swallowed: a tenant left on licences'
    ``byo_required`` default cannot run."""
    wanted = _known(mode)
    result = await licenses_client.set_tenant_app_byo_mode(
        tenant_id=tenant_id, app_key=BYO_APP_KEY, byo_mode=wanted
    )
    invalidate_byo_posture(tenant_id)
    return TenantLlmAccess(
        mode=_known(result.byo_mode), source="app_override", prior_mode=result.prior_byo_mode
    )
