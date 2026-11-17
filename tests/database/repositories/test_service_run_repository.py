"""``service_runs`` repository against REAL postgres.

Each test starts from TRUNCATEd tables (``db_engine``). The point of most of these is
the tenant scoping: the repository filters on ``tenant_id`` explicitly, so it holds
even on the owner role that bypasses the RLS policy (tests, ``TENANT_RLS_MODE=off``),
and the policy is the backstop under the restricted runtime role in prod.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4, uuid7

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.database.models import ServiceRun
from src.database.repositories.admin_repository import AdminRepository
from src.database.repositories.service_run_repository import ServiceRunRepository

pytestmark = pytest.mark.asyncio

_PROVIDER: dict[str, object] = {
    "source": "tenant",
    "kind": "openai-compatible",
    "model": "m",
}


async def _tenant(session) -> UUID:
    tenant_id = uuid7()
    await AdminRepository(session).ensure_tenant(tenant_id, created_via="super_admin")
    return tenant_id


async def _image_run(
    repo: ServiceRunRepository,
    tenant_id: UUID,
    *,
    created_at: datetime | None = None,
) -> ServiceRun:
    """The one row shape most tests need: a succeeded image with a provider."""
    return await repo.insert(
        tenant_id=tenant_id,
        slot="image",
        status="succeeded",
        output_kind="image",
        request={"prompt": "a red fox"},
        media_content_type="image/png",
        duration_ms=1234,
        provider=_PROVIDER,
        created_at=created_at,
    )


async def test_insert_flushes_a_listable_row(db_session) -> None:
    tenant_id = await _tenant(db_session)
    repo = ServiceRunRepository(db_session)

    row = await _image_run(repo, tenant_id)

    assert row.id is not None
    listed = await repo.list_by_slot(tenant_id, "image")
    assert [r.id for r in listed] == [row.id]
    assert listed[0].provider["kind"] == "openai-compatible"
    assert listed[0].request == {"prompt": "a red fox"}


async def test_list_is_newest_first_and_scoped_to_the_slot(db_session) -> None:
    tenant_id = await _tenant(db_session)
    repo = ServiceRunRepository(db_session)
    now = datetime.now(UTC)
    first = await _image_run(repo, tenant_id, created_at=now - timedelta(minutes=5))
    second = await _image_run(repo, tenant_id, created_at=now)
    await repo.insert(tenant_id=tenant_id, slot="tts", status="failed")

    listed = await repo.list_by_slot(tenant_id, "image")
    assert [r.id for r in listed] == [second.id, first.id]

    assert len(await repo.list_by_slot(tenant_id, "tts")) == 1


async def test_get_is_scoped_to_the_tenant(db_session) -> None:
    mine = await _tenant(db_session)
    theirs = await _tenant(db_session)
    repo = ServiceRunRepository(db_session)
    row = await _image_run(repo, mine)

    assert (await repo.get(mine, row.id)) is not None
    assert await repo.get(theirs, row.id) is None, "another tenant's run must read as absent"


async def test_delete_older_than_drops_only_rows_past_the_horizon(db_session) -> None:
    tenant_id = await _tenant(db_session)
    repo = ServiceRunRepository(db_session)
    stale = await _image_run(repo, tenant_id, created_at=datetime.now(UTC) - timedelta(days=31))
    fresh = await _image_run(repo, tenant_id)

    assert await repo.delete_older_than(tenant_id, 30) == 1

    remaining = (await db_session.execute(select(ServiceRun.id))).scalars().all()
    assert remaining == [fresh.id]
    assert stale.id not in remaining


async def test_delete_older_than_refuses_a_nonsense_horizon(db_session) -> None:
    with pytest.raises(ValueError, match="at least 1 day"):
        await ServiceRunRepository(db_session).delete_older_than(uuid4(), 0)


async def test_the_module_level_entry_points_the_route_uses(db_session) -> None:
    """``insert_run`` / ``list_runs`` / ``get_run`` — the names the route imports."""
    from src.database.repositories.service_run_repository import (
        get_run,
        insert_run,
        list_runs,
    )

    tenant_id = await _tenant(db_session)
    row = await insert_run(
        session=db_session,
        tenant_id=tenant_id,
        slot="llm",
        status="succeeded",
        output_kind="text",
        text_output="hello",
        provider=None,
    )

    assert [r.id for r in await list_runs(db_session, tenant_id, "llm", 10)] == [row.id]
    assert (await get_run(db_session, tenant_id, "llm", row.id)) is not None
    assert await get_run(db_session, tenant_id, "image", row.id) is None
    assert row.provider == {}, "a NULL provider is stored as an empty object"


async def test_a_media_key_outside_the_tenant_prefix_is_refused_by_the_database(
    db_session,
) -> None:
    """The CHECK is the DB's own guarantee: a row cannot point into another tenant."""
    tenant_id = await _tenant(db_session)
    repo = ServiceRunRepository(db_session)

    with pytest.raises(IntegrityError):
        await repo.insert(
            tenant_id=tenant_id,
            slot="image",
            status="succeeded",
            media_object_key=f"media-cache/tenants/{uuid7()}/service-lab/aa/x.png",
        )
