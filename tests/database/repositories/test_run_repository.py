from uuid import uuid4

import pytest

from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository

pytestmark = pytest.mark.asyncio


async def _setup_deliverable(session, tenant):
    repo = ProjectRepository(session)
    proj = await repo.create_with_deliverables(
        tenant_id=tenant,
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
                "name": "d",
                "format": "16x9",
                "captions_mode": "none",
                "target_duration_seconds": 60,
            }
        ],
    )
    await session.commit()
    return proj.deliverables[0].id


async def test_create_run_pending(db_session) -> None:
    tenant = uuid4()
    did = await _setup_deliverable(db_session, tenant)
    repo = RunRepository(db_session)
    run = await repo.create(
        deliverable_id=did,
        blueprint_id="generic_v1",
        config_snapshot={"voice": "crisp"},
        pause_after_stages=["script_draft"],
    )
    await db_session.commit()
    assert run.status == "pending"
    assert run.blueprint_id == "generic_v1"
    assert run.pause_after_stages == ["script_draft"]


async def test_status_transitions(db_session) -> None:
    tenant = uuid4()
    did = await _setup_deliverable(db_session, tenant)
    repo = RunRepository(db_session)
    run = await repo.create(
        deliverable_id=did,
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    await db_session.commit()
    await repo.mark_started(run.id)
    await db_session.commit()
    fresh = await repo.get(run.id)
    assert fresh is not None
    assert fresh.status == "running"
    await repo.mark_completed(run.id)
    await db_session.commit()
    fresh = await repo.get(run.id)
    assert fresh is not None
    assert fresh.status == "completed"
    assert fresh.completed_at is not None


_ONE_DELIV = [
    {"name": "d", "format": "16x9", "captions_mode": "none", "target_duration_seconds": 60},
]


async def _make_project(session, tenant, *, title="p", deliverables=_ONE_DELIV):
    repo = ProjectRepository(session)
    proj = await repo.create_with_deliverables(
        tenant_id=tenant,
        owner_subject_id=uuid4(),
        type_="generic",
        title=title,
        brief={},
        brand={},
        voice={},
        theme={},
        visibility="tenant",
        deliverables=deliverables,
    )
    await session.commit()
    return proj


async def test_list_for_project_orders_newest_first_across_deliverables(db_session) -> None:
    tenant = uuid4()
    proj = await _make_project(
        db_session,
        tenant,
        deliverables=[
            {
                "name": "d1",
                "format": "16x9",
                "captions_mode": "none",
                "target_duration_seconds": 60,
            },
            {
                "name": "d2",
                "format": "9x16",
                "captions_mode": "none",
                "target_duration_seconds": 30,
            },
        ],
    )
    d1, d2 = proj.deliverables[0].id, proj.deliverables[1].id
    repo = RunRepository(db_session)
    # Created in one transaction so they share Postgres' transaction-stable
    # now() — the uuid7 id tie-break decides order (newest id first).
    r1 = await repo.create(
        deliverable_id=d1, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    r2 = await repo.create(
        deliverable_id=d2, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    r3 = await repo.create(
        deliverable_id=d1, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await db_session.commit()

    listed = await repo.list_for_project(proj.id, tenant_id=tenant)
    assert [r.id for r in listed] == [r3.id, r2.id, r1.id]


async def test_list_for_project_excludes_other_projects(db_session) -> None:
    tenant = uuid4()
    proj = await _make_project(db_session, tenant, title="mine")
    other = await _make_project(db_session, tenant, title="other")
    repo = RunRepository(db_session)
    mine = await repo.create(
        deliverable_id=proj.deliverables[0].id,
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    await repo.create(
        deliverable_id=other.deliverables[0].id,
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    await db_session.commit()

    listed = await repo.list_for_project(proj.id, tenant_id=tenant)
    assert [r.id for r in listed] == [mine.id]


async def test_list_for_project_tenant_scoped(db_session) -> None:
    tenant = uuid4()
    proj = await _make_project(db_session, tenant)
    repo = RunRepository(db_session)
    await repo.create(
        deliverable_id=proj.deliverables[0].id,
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    await db_session.commit()

    # A different tenant must not see another tenant's runs.
    assert await repo.list_for_project(proj.id, tenant_id=uuid4()) == []


async def test_list_for_deliverable_orders_by_created_desc(db_session) -> None:
    tenant = uuid4()
    did = await _setup_deliverable(db_session, tenant)
    repo = RunRepository(db_session)
    a = await repo.create(
        deliverable_id=did,
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    b = await repo.create(
        deliverable_id=did,
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    await db_session.commit()
    runs = await repo.list_for_deliverable(did)
    assert [r.id for r in runs][:2] == [b.id, a.id]


async def test_mark_paused_is_noop_on_completed_run(db_session) -> None:
    """TOCTOU guard: pausing a run that already reached a terminal state must
    NOT flip it back to paused (a racing pause vs a completion on another pod)."""
    tenant = uuid4()
    did = await _setup_deliverable(db_session, tenant)
    repo = RunRepository(db_session)
    run = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await repo.mark_started(run.id)
    await repo.mark_completed(run.id)
    await db_session.commit()

    await repo.mark_paused(run.id, reason="checkpoint")  # must be a no-op
    await db_session.commit()
    fresh = await repo.get(run.id)
    assert fresh is not None
    assert fresh.status == "completed"  # not flipped to paused


async def test_mark_cancelled_is_noop_on_completed_run_but_works_on_paused(db_session) -> None:
    """Cancel must not override a terminal state, but a paused run is still
    cancellable (the guard excludes only completed/cancelled/failed)."""
    tenant = uuid4()
    did = await _setup_deliverable(db_session, tenant)
    repo = RunRepository(db_session)

    done = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await repo.mark_started(done.id)
    await repo.mark_completed(done.id)
    await db_session.commit()
    await repo.mark_cancelled(done.id)  # no-op on a completed run
    await db_session.commit()
    fresh = await repo.get(done.id)
    assert fresh is not None and fresh.status == "completed"

    paused = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await repo.mark_started(paused.id)
    await repo.mark_paused(paused.id, reason="checkpoint")
    await db_session.commit()
    await repo.mark_cancelled(paused.id)  # paused → cancellable
    await db_session.commit()
    fresh = await repo.get(paused.id)
    assert fresh is not None and fresh.status == "cancelled"


async def test_mark_completed_does_not_clobber_a_cancelled_run(db_session) -> None:
    """Terminal-state guard: a late completion (e.g. the run finished a final
    stage on one pod just as a cancel landed on another) must NOT overwrite the
    cancel. First terminal write wins."""
    tenant = uuid4()
    did = await _setup_deliverable(db_session, tenant)
    repo = RunRepository(db_session)
    run = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await repo.mark_started(run.id)
    await repo.mark_cancelled(run.id)
    await db_session.commit()

    # Bool contract: an already-terminal (cancelled) run does not transition.
    assert await repo.mark_completed(run.id) is False  # must be a no-op
    await db_session.commit()
    fresh = await repo.get(run.id)
    assert fresh is not None
    assert fresh.status == "cancelled"  # not flipped to completed


async def test_mark_failed_does_not_clobber_a_completed_run(db_session) -> None:
    """A late failure must not overwrite an already-completed run."""
    tenant = uuid4()
    did = await _setup_deliverable(db_session, tenant)
    repo = RunRepository(db_session)
    run = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await repo.mark_started(run.id)
    await repo.mark_completed(run.id)
    await db_session.commit()

    # Bool contract: an already-terminal (completed) run does not transition.
    assert await repo.mark_failed(run.id, "late error") is False  # must be a no-op
    await db_session.commit()
    fresh = await repo.get(run.id)
    assert fresh is not None
    assert fresh.status == "completed"
    assert fresh.error is None


async def test_mark_completed_returns_true_only_on_the_first_transition(db_session) -> None:
    """The write-once guard now reports whether THIS call transitioned the row:
    the first mark_completed on a running run returns True; a second call (now
    terminal) returns False; and mark_completed on an already-cancelled run
    returns False. The dispatcher gates its terminal SSE publish + runs_total
    metric on this bool, so double-counting on a finalize↔inline-gate race is
    impossible."""
    tenant = uuid4()
    did = await _setup_deliverable(db_session, tenant)
    repo = RunRepository(db_session)

    run = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await repo.mark_started(run.id)
    await db_session.commit()

    # First transition wins.
    assert await repo.mark_completed(run.id) is True
    await db_session.commit()
    # Second call is a no-op — the row is already terminal.
    assert await repo.mark_completed(run.id) is False
    await db_session.commit()

    # mark_completed on an already-cancelled run also returns False.
    cancelled = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await repo.mark_started(cancelled.id)
    await repo.mark_cancelled(cancelled.id)
    await db_session.commit()
    assert await repo.mark_completed(cancelled.id) is False
    await db_session.commit()


async def test_mark_paused_returns_true_only_on_the_first_transition(db_session) -> None:
    """Same write-once bool contract as mark_completed/mark_failed: the first
    mark_paused on a running run transitions the row and returns True; a second
    call (now already paused) transitions 0 rows and returns False. The
    dispatcher gates its ``run_paused`` SSE + runs_total{paused} increment on
    this bool, so a raced reconciler re-dispatch of an already-paused run can
    neither double-publish nor double-count."""
    tenant = uuid4()
    did = await _setup_deliverable(db_session, tenant)
    repo = RunRepository(db_session)

    run = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await repo.mark_started(run.id)
    await db_session.commit()

    # First pause wins.
    assert await repo.mark_paused(run.id, reason="checkpoint") is True
    await db_session.commit()
    # Second call is a no-op — the row is already paused.
    assert await repo.mark_paused(run.id, reason="checkpoint") is False
    await db_session.commit()
    fresh = await repo.get(run.id)
    assert fresh is not None and fresh.status == "paused"

    # mark_paused on an already-terminal (completed) run also returns False.
    done = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await repo.mark_started(done.id)
    await repo.mark_completed(done.id)
    await db_session.commit()
    assert await repo.mark_paused(done.id, reason="checkpoint") is False
    await db_session.commit()


async def test_mark_cancelled_returns_true_only_on_the_first_transition(db_session) -> None:
    """Write-once bool contract for cancel: the first mark_cancelled on a
    running run transitions the row and returns True; a second call (now already
    cancelled) returns False; and mark_cancelled on an already-completed run
    returns False. The dispatcher's user-cancel path gates its ``run_cancelled``
    SSE + runs_total{cancelled} increment on this bool so a two-replica cancel
    race publishes/counts exactly once. A paused run is still cancellable
    (returns True) — the guard excludes only completed/cancelled/failed."""
    tenant = uuid4()
    did = await _setup_deliverable(db_session, tenant)
    repo = RunRepository(db_session)

    run = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await repo.mark_started(run.id)
    await db_session.commit()

    # First cancel wins.
    assert await repo.mark_cancelled(run.id) is True
    await db_session.commit()
    # Second call is a no-op — the row is already cancelled.
    assert await repo.mark_cancelled(run.id) is False
    await db_session.commit()
    fresh = await repo.get(run.id)
    assert fresh is not None and fresh.status == "cancelled"

    # A paused run is still cancellable → True.
    paused = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await repo.mark_started(paused.id)
    await repo.mark_paused(paused.id, reason="checkpoint")
    await db_session.commit()
    assert await repo.mark_cancelled(paused.id) is True
    await db_session.commit()

    # mark_cancelled on an already-completed run returns False.
    done = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await repo.mark_started(done.id)
    await repo.mark_completed(done.id)
    await db_session.commit()
    assert await repo.mark_cancelled(done.id) is False
    await db_session.commit()


async def test_mark_resumed_only_resumes_a_paused_run(db_session) -> None:
    """Resume must be a no-op on a terminal run — a cancel/complete landing
    between the route's read-side ``status != paused`` check and this write
    must not resurrect the run back to ``running``."""
    tenant = uuid4()
    did = await _setup_deliverable(db_session, tenant)
    repo = RunRepository(db_session)

    # A cancelled run can never be resumed.
    cancelled = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await repo.mark_started(cancelled.id)
    await repo.mark_cancelled(cancelled.id)
    await db_session.commit()
    await repo.mark_resumed(cancelled.id)  # no-op
    await db_session.commit()
    fresh = await repo.get(cancelled.id)
    assert fresh is not None and fresh.status == "cancelled"

    # A paused run resumes to running.
    paused = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await repo.mark_started(paused.id)
    await repo.mark_paused(paused.id, reason="checkpoint")
    await db_session.commit()
    await repo.mark_resumed(paused.id)
    await db_session.commit()
    fresh = await repo.get(paused.id)
    assert fresh is not None and fresh.status == "running"


async def test_mark_started_cannot_resurrect_a_cancelled_run(db_session) -> None:
    """``mark_started`` was the ONE status writer with no CAS, and dispatch calls
    it on every resume / reconciler re-dispatch. A cancel landing between the
    reconciler's orphan SELECT and that write was silently reverted — the run
    went `cancelled` -> `running` and billed to completion."""
    tenant = uuid4()
    did = await _setup_deliverable(db_session, tenant)
    repo = RunRepository(db_session)
    run = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await db_session.commit()
    run_id = run.id
    assert await repo.mark_cancelled(run_id) is True
    await db_session.commit()

    assert await repo.mark_started(run_id) is False
    await db_session.commit()
    db_session.expire_all()
    fresh = await repo.get(run_id)
    assert fresh is not None
    assert fresh.status == "cancelled"


async def test_mark_started_cannot_resurrect_a_paused_run(db_session) -> None:
    """Same window on a `paused` row: flipping it back to `running` runs the DAG
    straight past the human-review gate."""
    tenant = uuid4()
    did = await _setup_deliverable(db_session, tenant)
    repo = RunRepository(db_session)
    run = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await db_session.commit()
    run_id = run.id
    assert await repo.mark_paused(run_id, reason="checkpoint") is True
    await db_session.commit()

    assert await repo.mark_started(run_id) is False
    await db_session.commit()
    db_session.expire_all()
    fresh = await repo.get(run_id)
    assert fresh is not None
    assert fresh.status == "paused"


async def test_statuses_returns_readable_runs_in_one_query(db_session) -> None:
    """``statuses`` maps each existing id to its status; an unknown id is absent."""
    tenant = uuid4()
    did = await _setup_deliverable(db_session, tenant)
    repo = RunRepository(db_session)
    a = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    b = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await db_session.commit()
    await repo.mark_completed(b.id)
    await db_session.commit()
    ghost = uuid4()

    got = await repo.statuses([a.id, b.id, ghost])

    assert got == {a.id: "pending", b.id: "completed"}
    assert await repo.statuses([]) == {}


async def test_resume_clears_the_pause_reason(db_session) -> None:
    """0102: a resumed run is not paused, so it has no pause reason — a stale
    ``checkpoint`` would make the next manual pause (or any reader) lie."""
    from src.database.models import Run

    repo = RunRepository(db_session)
    did = await _setup_deliverable(db_session, uuid4())
    run = await repo.create(
        deliverable_id=did,
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    await repo.mark_started(run.id)
    assert await repo.mark_paused(run.id, reason="checkpoint")
    await db_session.commit()
    assert (await db_session.get(Run, run.id, populate_existing=True)).pause_reason == "checkpoint"

    assert await repo.mark_resumed(run.id)
    await db_session.commit()
    row = await db_session.get(Run, run.id, populate_existing=True)
    assert row.status == "running"
    assert row.pause_reason is None
    assert row.paused_at is None
