"""Tenant production defaults: the precedence rules, in one place (P6 ORCH-1/ORCH-2).

Every default resolves by the same explicit ladder:

    run / project / brief explicit value  >  tenant default  >  deployment default

* **run / project / brief** — ``config_overrides``, the project row's ``brand`` /
  ``voice`` blocks, a brief-authored ``voice.direction`` or ``direction.retention``,
  a deliverable's explicit ``captions_mode``. A value that is present and not ``None``
  always wins.
* **tenant default** — the ``tenant_defaults`` row (migration 0081). A ``None`` field
  (or no row) means "no tenant default".
* **deployment default** — the ``PipelineSettings`` env value or the code constant
  that applied before P6.

The run reads the row ONCE, at creation (``build_config_snapshot``), and folds the
resolved values into its config snapshot. Stages only ever read the snapshot, so a
running (or paused) run keeps the defaults it started with when the tenant changes
them, and the next run picks up the change.

The raw row is pinned too (``TENANT_DEFAULTS_KEY``) so that a run SPAWNED BY a run
(course lessons, structural relief, every regenerate / re-stitch child) inherits its
parent's defaults instead of re-reading the tenant's current ones: a job pins for its
whole lifetime, children included. A parent without the key (it ran on deployment
defaults, or predates P6) passes "no tenant defaults" down — never a live read.

With no row, every helper returns exactly what the pre-P6 code produced, so the
snapshot is byte-identical (the golden test pins that).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import TenantDefaults
from src.database.repositories.tenant_defaults_repository import TenantDefaultsRepository

#: Snapshot key holding the tenant-defaults row the run pinned (``version`` + every
#: non-null field; absent when the tenant had no row). Server-owned (stripped from
#: overrides) and run-scoped (excluded from ``order_fingerprint`` — the resolved
#: VALUES it produced are what the order compares).
TENANT_DEFAULTS_KEY = "tenant_defaults"
#: Snapshot key carrying the resolved karaoke flag (present only when the tenant set
#: one; absent → the stage falls back to the deployment env, as before P6).
CAPTIONS_KARAOKE_KEY = "captions_karaoke"

#: Deployment defaults that were code constants before P6.
DEPLOYMENT_VOICE_DIRECTION_TONE = "confident"
DEPLOYMENT_RETENTION_MODE = "critique_only"
DEPLOYMENT_CAPTIONS_MODE = "none"


@dataclass(frozen=True, slots=True)
class ResolvedTenantDefaults:
    """A plain, session-free copy of one ``tenant_defaults`` row."""

    version: int
    voice: Mapping[str, object]
    brand: Mapping[str, object]
    captions: Mapping[str, object]
    voice_direction: Mapping[str, object]
    retention_mode: str | None
    model_tier: Mapping[str, object]

    @classmethod
    def from_row(cls, row: TenantDefaults) -> ResolvedTenantDefaults:
        return cls(
            version=row.version,
            voice=_non_null(row.voice),
            brand=_non_null(row.brand),
            captions=_non_null(row.captions),
            voice_direction=_non_null(row.voice_direction),
            retention_mode=row.retention_mode,
            model_tier=_non_null(row.model_tier),
        )

    def to_snapshot(self) -> dict[str, object]:
        """The pin stored under :data:`TENANT_DEFAULTS_KEY` (JSON-safe, nulls dropped)."""
        out: dict[str, object] = {"version": self.version}
        for name in ("voice", "brand", "captions", "voice_direction", "model_tier"):
            block = getattr(self, name)
            if block:
                out[name] = dict(block)
        if self.retention_mode is not None:
            out["retention_mode"] = self.retention_mode
        return out

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, object]) -> ResolvedTenantDefaults | None:
        """The defaults a parent run pinned; ``None`` when it pinned none."""
        pin = snapshot.get(TENANT_DEFAULTS_KEY)
        if not isinstance(pin, Mapping):
            return None
        version = pin.get("version")
        mode = pin.get("retention_mode")
        return cls(
            version=version if isinstance(version, int) else 0,
            voice=_block(pin.get("voice")),
            brand=_block(pin.get("brand")),
            captions=_block(pin.get("captions")),
            voice_direction=_block(pin.get("voice_direction")),
            retention_mode=mode if isinstance(mode, str) else None,
            model_tier=_block(pin.get("model_tier")),
        )


def _block(value: object) -> dict[str, object]:
    return _non_null(value) if isinstance(value, Mapping) else {}


async def resolve_run_defaults(
    session: AsyncSession,
    tenant_id: UUID,
    inherit_from: Mapping[str, object] | None,
) -> ResolvedTenantDefaults | None:
    """The defaults a NEW run pins: its parent's (``inherit_from`` = the parent's
    snapshot) for a run spawned by a run, else the tenant's current row."""
    if inherit_from is not None:
        return ResolvedTenantDefaults.from_snapshot(inherit_from)
    return await load_tenant_defaults(session, tenant_id)


async def load_tenant_defaults(
    session: AsyncSession, tenant_id: UUID
) -> ResolvedTenantDefaults | None:
    row = await TenantDefaultsRepository(session).get(tenant_id)
    return ResolvedTenantDefaults.from_row(row) if row is not None else None


def _non_null(block: Mapping[str, object] | None) -> dict[str, object]:
    return {k: v for k, v in (block or {}).items() if v is not None}


# ─── the ladder, per default ─────────────────────────────────────────────


def merge_block(
    project_block: Mapping[str, object] | None, tenant_block: Mapping[str, object] | None
) -> dict[str, object]:
    """``brand`` / ``voice``: tenant keys underneath, the project's non-``None`` keys on
    top. No tenant block → ``dict(project_block)`` exactly (byte-identical)."""
    project = dict(project_block or {})
    if not tenant_block:
        return project
    merged = dict(tenant_block)
    for key, value in project.items():
        if value is not None or key not in merged:
            merged[key] = value
    return merged


def _pinned_tenant_brand(snapshot: Mapping[str, object]) -> dict[str, object]:
    pin = snapshot.get(TENANT_DEFAULTS_KEY)
    return _block(pin.get("brand")) if isinstance(pin, Mapping) else {}


def authored_brand(snapshot: Mapping[str, object]) -> dict[str, object] | None:
    """``snapshot["brand"]`` WITHOUT the keys the tenant default filled in.

    ``build_config_snapshot`` merges the tenant brand UNDER the project's, so a stage
    with a higher-precedence brand source of its own (a brief-authored ``brand``, the
    reference site's extracted palette) must not read the merged block as "authored":
    the tenant default would shadow both. A key whose value equals the tenant's is
    treated as tenant-filled; :func:`fill_tenant_brand` puts it back underneath.
    No pinned tenant brand → ``snapshot["brand"]`` unchanged (byte-identical)."""
    brand = snapshot.get("brand")
    if not isinstance(brand, Mapping):
        return None
    tenant = _pinned_tenant_brand(snapshot)
    return {k: v for k, v in brand.items() if not (k in tenant and tenant[k] == v)}


def brief_brand(brief: Mapping[str, object]) -> dict[str, object] | None:
    """A brief-authored ``brand`` block (a non-object value is ignored)."""
    brand = brief.get("brand")
    return dict(brand) if isinstance(brand, Mapping) else None


def fill_tenant_brand(
    brand: Mapping[str, object] | None, snapshot: Mapping[str, object]
) -> dict[str, object] | None:
    """The run's pinned tenant brand underneath ``brand`` (gap-filler only). No pinned
    tenant brand → ``brand`` as given."""
    tenant = _pinned_tenant_brand(snapshot)
    if not tenant:
        return dict(brand) if brand is not None else None
    return {**tenant, **(brand or {})}


def voice_direction_tone(defaults: ResolvedTenantDefaults | None) -> str:
    """Seeded ``voice.direction.tone`` (only when the run/brief authored none)."""
    tone = defaults.voice_direction.get("tone") if defaults else None
    return tone if isinstance(tone, str) and tone else DEPLOYMENT_VOICE_DIRECTION_TONE


def retention_seed_mode(defaults: ResolvedTenantDefaults | None) -> str | None:
    """Seeded ``direction.retention.mode`` (only when the run/brief authored none).

    ``None`` means seed nothing: the tenant chose ``off``, so ``retention_plan`` skips
    exactly as it does for a run that never had the block."""
    mode = defaults.retention_mode if defaults else None
    if mode is None:
        return DEPLOYMENT_RETENTION_MODE
    return None if mode == "off" else mode


def captions_mode(requested: str | None, defaults: ResolvedTenantDefaults | None) -> str:
    """A new deliverable's ``captions_mode``: the request's explicit value, else the
    tenant's, else ``"none"``."""
    if requested is not None:
        return requested
    mode = defaults.captions.get("mode") if defaults else None
    return mode if isinstance(mode, str) and mode else DEPLOYMENT_CAPTIONS_MODE


def tenant_captions_karaoke(defaults: ResolvedTenantDefaults | None) -> bool | None:
    """The tenant's karaoke choice, or ``None`` (→ the deployment env applies)."""
    value = defaults.captions.get("karaoke") if defaults else None
    return value if isinstance(value, bool) else None


def captions_karaoke(snapshot: Mapping[str, object], deployment: bool) -> bool:
    """The run's karaoke flag: the value pinned in its snapshot, else the env."""
    value = snapshot.get(CAPTIONS_KARAOKE_KEY)
    return value if isinstance(value, bool) else deployment


def timing_director_tier(
    defaults: ResolvedTenantDefaults | None, deployment: str | None
) -> str | None:
    """The run's ``model_tier.timing_director`` pin: tenant alias, else the env."""
    tier = defaults.model_tier.get("timing_director") if defaults else None
    return tier if isinstance(tier, str) and tier else deployment
