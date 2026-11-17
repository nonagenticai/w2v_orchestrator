"""P6 ORCH-4 — ``tenant_sound_assets`` RLS under the prod runtime posture.

Only a non-owner ``NOBYPASSRLS`` role (prod ``w2v_orch_app``) makes rows depend on
``app.current_tenant``. The POLICY, not a WHERE clause, isolates tenants: an
unfiltered read bound to A sees none of B's tracks, and a write of B's row while bound
to A is refused.
"""

from __future__ import annotations

from uuid import UUID, uuid7

import pytest
from sqlalchemy import insert, select, text
from sqlalchemy.exc import DBAPIError

from src.database.models import Tenant, TenantSoundAsset

# Re-exported fixtures (pytest resolves them by name in this module).
from tests.api.routes.test_tenancy_connections_rls import (  # noqa: F401 — fixtures
    _enforce,
    restricted_factory,
)

pytestmark = pytest.mark.asyncio


def _values(tenant: UUID, n: int) -> dict[str, object]:
    return {
        "id": uuid7(),
        "tenant_id": tenant,
        "sound_id": f"tenant_bed_{n}",
        "category": "music",
        "object_key": f"sound_library/tenants/{tenant}/music/{n}.wav",
        "duration_s": 1.0,
        "bytes": 10,
        "content_type": "audio/wav",
        "rights_attested_by": uuid7(),
    }


async def _bind(session, tenant: UUID) -> None:
    await session.execute(
        text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(tenant)}
    )


async def test_policy_isolates_tenant_tracks_without_any_where_clause(
    db_session_factory,
    restricted_factory,  # noqa: F811 — the imported fixture
) -> None:
    a, b = uuid7(), uuid7()
    async with db_session_factory() as owner:
        for tid in (a, b):
            await owner.execute(insert(Tenant).values(id=tid, name=f"t-{tid}"))
        await owner.execute(insert(TenantSoundAsset).values(**_values(b, 1)))
        await owner.commit()

    async with restricted_factory() as s:
        await _bind(s, a)
        assert (await s.execute(select(TenantSoundAsset))).scalars().all() == []
        await s.rollback()
        await _bind(s, b)  # control: the same unfiltered read bound to B
        rows = (await s.execute(select(TenantSoundAsset))).scalars().all()
        assert [r.tenant_id for r in rows] == [b]
        await s.rollback()

    async with restricted_factory() as s:
        await _bind(s, a)
        with pytest.raises(DBAPIError, match="row-level security"):
            await s.execute(insert(TenantSoundAsset).values(**_values(b, 2)))
        await s.rollback()
    async with restricted_factory() as s:
        await _bind(s, a)
        deleted = await s.execute(
            text("UPDATE tenant_sound_assets SET deleted_at = now() WHERE tenant_id = :b"),
            {"b": b},
        )
        assert deleted.rowcount == 0, "A must not be able to retire B's track"
        await s.commit()
