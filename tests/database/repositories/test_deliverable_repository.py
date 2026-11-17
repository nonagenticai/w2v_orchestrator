from uuid import UUID, uuid4, uuid7

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.repositories.deliverable_repository import (
    DeliverableRepository,
    derive_deliverable_status,
)
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository

pytestmark = pytest.mark.asyncio


async def _add_run(session: AsyncSession, deliverable_id: UUID, *, mark: str | None = None) -> UUID:
    rr = RunRepository(session)
    run = await rr.create(
        deliverable_id=deliverable_id,
        blueprint_id="promo.v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    if mark == "started":
        await rr.mark_started(run.id)
    elif mark == "paused":
        await rr.mark_started(run.id)
        await rr.mark_paused(run.id, reason="checkpoint")
    elif mark == "paused_manual":
        await rr.mark_started(run.id)
        await rr.mark_paused(run.id, reason="manual")
    elif mark == "completed":
        await rr.mark_completed(run.id)
    elif mark == "failed":
        await rr.mark_failed(run.id, "boom")
    elif mark == "cancelled":
        await rr.mark_cancelled(run.id)
    await session.commit()
    return run.id


async def _make_project(session, tenant_id, deliverable_count=1):
    repo = ProjectRepository(session)
    proj = await repo.create_with_deliverables(
        tenant_id=tenant_id,
        owner_subject_id=uuid4(),
        type_="generic",
        title="x",
        brief={},
        brand={},
        voice={},
        theme={},
        visibility="tenant",
        deliverables=[
            {
                "name": f"d{i}",
                "format": "16x9",
                "captions_mode": "none",
                "target_duration_seconds": 60,
            }
            for i in range(deliverable_count)
        ],
    )
    await session.commit()
    return proj


async def test_get_by_id(db_session: AsyncSession) -> None:
    tenant = uuid4()
    proj = await _make_project(db_session, tenant)
    repo = DeliverableRepository(db_session)
    deliv = proj.deliverables[0]
    fetched = await repo.get(deliv.id, tenant_id=tenant)
    assert fetched is not None
    assert fetched.id == deliv.id


async def test_cross_tenant_get_returns_none(db_session: AsyncSession) -> None:
    t1, t2 = uuid4(), uuid4()
    proj = await _make_project(db_session, t1)
    repo = DeliverableRepository(db_session)
    fetched = await repo.get(proj.deliverables[0].id, tenant_id=t2)
    assert fetched is None


async def test_update_fields(db_session: AsyncSession) -> None:
    tenant = uuid4()
    proj = await _make_project(db_session, tenant)
    repo = DeliverableRepository(db_session)
    d = proj.deliverables[0]
    updated = await repo.update(
        d.id, tenant_id=tenant, fields={"name": "renamed", "format": "9x16"}
    )
    await db_session.commit()
    assert updated is not None
    assert updated.name == "renamed"
    assert updated.format == "9x16"


async def test_delete_cascades_runs(db_session: AsyncSession) -> None:
    """ON DELETE CASCADE on runs.deliverable_id is enforced by FK."""
    tenant = uuid4()
    proj = await _make_project(db_session, tenant)
    repo = DeliverableRepository(db_session)
    d = proj.deliverables[0]
    deleted = await repo.delete(d.id, tenant_id=tenant)
    await db_session.commit()
    assert deleted is True
    assert await repo.get(d.id, tenant_id=tenant) is None


# ── derived (read-time) status from the latest run ──────────────────────────


async def test_derive_mapping() -> None:  # asyncio_mode=auto collects all tests as async
    assert derive_deliverable_status(None) == "idle"
    assert derive_deliverable_status("pending") == "running"
    assert derive_deliverable_status("running") == "running"
    # Only a CHECKPOINT pause is a review wait; no reason / unknown is just paused.
    assert derive_deliverable_status("paused") == "paused"
    assert derive_deliverable_status("paused", "unknown") == "paused"
    assert derive_deliverable_status("paused", "checkpoint") == "awaiting_review"
    # Someone pressed Pause: not a review wait (0102).
    assert derive_deliverable_status("paused", "manual") == "paused"
    # A stale reason on a non-paused run changes nothing.
    assert derive_deliverable_status("running", "manual") == "running"
    assert derive_deliverable_status("completed") == "completed"
    assert derive_deliverable_status("failed") == "failed"
    assert derive_deliverable_status("cancelled") == "failed"


async def test_derived_statuses_no_run_is_idle(db_session: AsyncSession) -> None:
    proj = await _make_project(db_session, uuid4())
    did = proj.deliverables[0].id
    repo = DeliverableRepository(db_session)
    assert await repo.derived_statuses([did]) == {did: "idle"}


async def test_derived_statuses_unknown_id_and_empty(db_session: AsyncSession) -> None:
    repo = DeliverableRepository(db_session)
    missing = uuid7()
    assert await repo.derived_statuses([missing]) == {missing: "idle"}
    assert await repo.derived_statuses([]) == {}


async def test_derived_statuses_latest_run_wins(db_session: AsyncSession) -> None:
    proj = await _make_project(db_session, uuid4())
    did = proj.deliverables[0].id
    # Older completed run, then a newer in-flight one — latest (running) wins.
    await _add_run(db_session, did, mark="completed")
    await _add_run(db_session, did, mark="started")
    repo = DeliverableRepository(db_session)
    assert (await repo.derived_statuses([did]))[did] == "running"


async def test_derived_statuses_paused_is_awaiting_review(db_session: AsyncSession) -> None:
    proj = await _make_project(db_session, uuid4())
    did = proj.deliverables[0].id
    await _add_run(db_session, did, mark="paused")
    repo = DeliverableRepository(db_session)
    assert (await repo.derived_statuses([did]))[did] == "awaiting_review"


async def test_derived_statuses_manual_pause_is_paused_not_review(db_session: AsyncSession) -> None:
    proj = await _make_project(db_session, uuid4())
    did = proj.deliverables[0].id
    await _add_run(db_session, did, mark="paused_manual")
    repo = DeliverableRepository(db_session)
    assert (await repo.derived_statuses([did]))[did] == "paused"


async def test_derived_statuses_batches_and_maps_terminal(db_session: AsyncSession) -> None:
    proj = await _make_project(db_session, uuid4(), deliverable_count=3)
    d0, d1, d2 = (d.id for d in proj.deliverables)
    await _add_run(db_session, d0, mark="completed")
    await _add_run(db_session, d1, mark="failed")
    await _add_run(db_session, d2, mark="cancelled")
    repo = DeliverableRepository(db_session)
    assert await repo.derived_statuses([d0, d1, d2]) == {
        d0: "completed",
        d1: "failed",
        d2: "failed",
    }
