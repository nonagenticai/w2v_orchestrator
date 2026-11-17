"""Tenant production defaults (P6 ORCH-1, migration 0081).

One row per tenant. Writes are compare-and-set on ``version``, done in ONE statement
each, so two concurrent admins cannot both win:

* first write (``expected_version == 0``): ``INSERT … ON CONFLICT DO NOTHING
  RETURNING`` — a lost race returns no row;
* later writes: ``UPDATE … WHERE version = :expected RETURNING`` with
  ``version = version + 1`` — a stale version returns no row.

The caller maps "no row" to 409. The table is under RLS ``tenant_isolation``, and every
statement here still filters on ``tenant_id`` explicitly: the owner role (tests,
``TENANT_RLS_MODE=off``) bypasses the policy, so the WHERE clause is the guarantee and
the policy the backstop. Nothing here commits; the caller's session owns the
transaction.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import TenantDefaults

#: The setting columns a PUT replaces (``version``/``updated_*`` are server-owned).
SETTING_COLUMNS: tuple[str, ...] = (
    "voice",
    "brand",
    "captions",
    "voice_direction",
    "retention_mode",
    "model_tier",
)


class TenantDefaultsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, tenant_id: UUID) -> TenantDefaults | None:
        return (
            await self.session.execute(
                select(TenantDefaults).where(TenantDefaults.tenant_id == tenant_id)
            )
        ).scalar_one_or_none()

    async def create(
        self, tenant_id: UUID, values: dict[str, object], *, actor: UUID | None
    ) -> TenantDefaults | None:
        """Insert version 1. ``None`` when a row already exists (lost race)."""
        stmt = (
            insert(TenantDefaults)
            .values(tenant_id=tenant_id, version=1, updated_by=actor, **_settings(values))
            .on_conflict_do_nothing(index_elements=[TenantDefaults.tenant_id])
            .returning(TenantDefaults)
            .execution_options(populate_existing=True)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def update_if_version(
        self,
        tenant_id: UUID,
        expected_version: int,
        values: dict[str, object],
        *,
        actor: UUID | None,
    ) -> TenantDefaults | None:
        """Replace every setting and bump ``version``. ``None`` on a stale version."""
        stmt = (
            update(TenantDefaults)
            .where(
                TenantDefaults.tenant_id == tenant_id,
                TenantDefaults.version == expected_version,
            )
            .values(
                version=TenantDefaults.version + 1,
                updated_by=actor,
                updated_at=func.now(),
                **_settings(values),
            )
            .returning(TenantDefaults)
            .execution_options(populate_existing=True)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()


def _settings(values: dict[str, object]) -> dict[str, object]:
    """Exactly the setting columns: every one present (a PUT replaces the document),
    nothing else (a caller cannot set ``version`` or ``tenant_id`` through here)."""
    unknown = set(values) - set(SETTING_COLUMNS)
    if unknown:
        raise ValueError(f"unknown tenant_defaults columns: {sorted(unknown)}")
    return {col: values.get(col) for col in SETTING_COLUMNS}
