"""The reconciler's stale/wedged branch must COMPARE-AND-SET against the
terminal set, exactly like every other run-status mutator.

Why a real database and not a mocked session: the defect is a lost-update race
between two orchestrator replicas, and its whole surface is the WHERE clause of
one UPDATE. A mocked ``session.execute`` can only prove which SQLAlchemy object
was passed, which is the shape of the fix, not its effect. These tests write a
genuinely terminal row, run the branch against it, and read the row back
through an INDEPENDENT session — so what is asserted is the outcome a user
sees, and a regression that keeps the ``.where()`` but breaks the terminal set
still goes red.

The interleaving being reproduced (the reconciler walks its orphan SNAPSHOT
sequentially with several network awaits per row, so a row can go terminal
underneath it):

    pod A: SELECT FROM orphan_runs()      -> run X is 'running'
    pod B: POST /api/runs/X/cancel        -> mark_cancelled CAS wins, commits,
                                             releases the execution lease
    pod A: _resolve_orphan_run(row=X)     -> lease_present False,
                                             db_claim_fresh False,
                                             age > cutoff  ==> failed?  NO.
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.database.models import Run
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository

pytestmark = pytest.mark.asyncio

# The row must look genuinely aged so the stale branch (not the resume branch)
# is the one under test. redis is passed as None throughout, so `attempts` is 0
# and `wedged` is False -- age is the sole trigger, which keeps these tests
# about the UPDATE rather than about the Redis tally.
_AGE = datetime.timedelta(hours=3)
_CUTOFF = datetime.timedelta(seconds=1)


async def _seed_run(session) -> tuple[RunRepository, Run, object]:
    """A committed run row with a real tenant/project/deliverable behind it."""
    tenant = uuid4()
    project = await ProjectRepository(session).create_with_deliverables(
        tenant_id=tenant,
        owner_subject_id=uuid4(),
        type_="generic",
        title="reconciler-cas",
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

    repo = RunRepository(session)
    run = await repo.create(
        deliverable_id=project.deliverables[0].id,
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    await repo.mark_started(run.id)
    await session.commit()
    return repo, run, tenant


def _stale_row(run: Run, tenant_id, *, status: str, now: datetime.datetime) -> SimpleNamespace:
    """The reconciler's view of the row: what the orphan SELECT saw, which by
    construction is the PRE-cancel status, not what is in the table now."""
    return SimpleNamespace(
        id=run.id,
        tenant_id=tenant_id,
        status=status,
        updated_at=now - _AGE,
    )


def _leaseless_dispatcher() -> MagicMock:
    """A dispatcher that reports the run as a genuine orphan: DAG not complete,
    no Redis lease, no fresh DB claim. This is what routes execution into the
    stale/wedged branch."""
    dispatcher = MagicMock()
    dispatcher.finalize_if_dag_complete = AsyncMock(return_value=None)
    dispatcher.lease_present = AsyncMock(return_value=False)
    dispatcher.db_claim_fresh = AsyncMock(return_value=False)
    dispatcher.reap_terminalized_run = AsyncMock()
    dispatcher.dispatch = AsyncMock()
    return dispatcher


async def _resolve(session, dispatcher, row, now):
    from src.lifecycle.reconciler import _resolve_orphan_run

    outcome = await _resolve_orphan_run(
        session,
        dispatcher,
        None,
        row,
        now=now,
        cutoff=_CUTOFF,
        max_age_seconds=int(_CUTOFF.total_seconds()),
    )
    await session.commit()
    return outcome


async def _read_back(db_session_factory, run_id) -> Run:
    """Read the row through a session that never saw the write, so nothing can
    be answered out of the identity map (the test sessionmaker sets
    expire_on_commit=False)."""
    async with db_session_factory() as verify:
        fresh = await verify.get(Run, run_id)
    assert fresh is not None
    return fresh


async def test_cancelled_run_is_not_flipped_to_failed(db_session, db_session_factory) -> None:
    """Pod B's cancel landed after pod A's snapshot. Pod A must not clobber it."""
    repo, run, tenant_id = await _seed_run(db_session)

    assert await repo.mark_cancelled(run.id) is True
    await db_session.commit()

    now = datetime.datetime.now(datetime.UTC)
    dispatcher = _leaseless_dispatcher()
    outcome = await _resolve(
        db_session,
        dispatcher,
        _stale_row(run, tenant_id, status="running", now=now),
        now,
    )

    fresh = await _read_back(db_session_factory, run.id)
    assert fresh.status == "cancelled"
    assert fresh.error is None
    assert fresh.failure_class is None
    # The reap is the cleanup half of a transition this site did not make: it
    # bulk-cancels the run's stage rows and its in-flight compile, and that is
    # what turns a mislabelled run into an unrecoverable one.
    dispatcher.reap_terminalized_run.assert_not_awaited()
    assert outcome != "stale"


async def test_completed_run_is_not_flipped_to_failed(db_session, db_session_factory) -> None:
    """Same race against a run that genuinely finished with its artifact stored."""
    repo, run, tenant_id = await _seed_run(db_session)

    assert await repo.mark_completed(run.id) is True
    await db_session.commit()
    completed_at = (await _read_back(db_session_factory, run.id)).completed_at

    now = datetime.datetime.now(datetime.UTC)
    dispatcher = _leaseless_dispatcher()
    outcome = await _resolve(
        db_session,
        dispatcher,
        _stale_row(run, tenant_id, status="running", now=now),
        now,
    )

    fresh = await _read_back(db_session_factory, run.id)
    assert fresh.status == "completed"
    assert fresh.error is None
    assert fresh.failure_class is None
    # The incoherence the unguarded write produced: status='failed' sitting next
    # to the completion timestamp of the success it overwrote.
    assert fresh.completed_at == completed_at
    dispatcher.reap_terminalized_run.assert_not_awaited()
    assert outcome != "stale"


async def test_genuine_running_orphan_is_still_failed_and_reaped(
    db_session, db_session_factory
) -> None:
    """The guard must not disable the feature: a still-non-terminal aged orphan
    is failed as before, and the cleanup half still fires."""
    _repo, run, tenant_id = await _seed_run(db_session)
    assert (await _read_back(db_session_factory, run.id)).status == "running"

    now = datetime.datetime.now(datetime.UTC)
    dispatcher = _leaseless_dispatcher()
    outcome = await _resolve(
        db_session,
        dispatcher,
        _stale_row(run, tenant_id, status="running", now=now),
        now,
    )

    fresh = await _read_back(db_session_factory, run.id)
    assert fresh.status == "failed"
    assert fresh.failure_class == "unknown"
    assert fresh.error is not None and "max-age" in fresh.error
    dispatcher.reap_terminalized_run.assert_awaited_once()
    dispatcher.dispatch.assert_not_awaited()
    assert outcome == "stale"


async def test_pending_orphan_is_still_failed(db_session, db_session_factory) -> None:
    """The other non-terminal status the orphan scan returns."""
    repo, run, tenant_id = await _seed_run(db_session)
    # Back to the pre-dispatch state the orphan scan also selects for.
    fresh_run = await repo.get(run.id)
    assert fresh_run is not None
    fresh_run.status = "pending"
    await db_session.commit()

    now = datetime.datetime.now(datetime.UTC)
    dispatcher = _leaseless_dispatcher()
    outcome = await _resolve(
        db_session,
        dispatcher,
        _stale_row(run, tenant_id, status="pending", now=now),
        now,
    )

    fresh = await _read_back(db_session_factory, run.id)
    assert fresh.status == "failed"
    assert fresh.failure_class == "unknown"
    dispatcher.reap_terminalized_run.assert_awaited_once()
    assert outcome == "stale"


async def test_guard_reuses_the_repository_non_terminal_set() -> None:
    """The guard reads ``RunRepository._NON_TERMINAL`` rather than retyping the
    strings, so a status added there cannot silently escape this site.

    It must be the ALLOW-list (``in_(_NON_TERMINAL)``), not the deny-list
    (``notin_(_TERMINAL)``): ``paused`` belongs to neither set, so the deny-list
    form would let the reaper fail a run a reviewer had just paused.

    Comments are STRIPPED before the check: the prose above the UPDATE names the
    constant too, so a plain substring search over the source stays green with
    the predicate deleted -- it would be testing the comment, not the code.
    """
    import inspect
    import io
    import tokenize

    from src.lifecycle import reconciler

    source = inspect.getsource(reconciler._resolve_orphan_run)
    code = "".join(
        tok.string
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type not in (tokenize.COMMENT, tokenize.STRING)
    )
    assert "RunRepository._NON_TERMINAL" in code
    assert "notin_" not in code
    assert set(RunRepository._NON_TERMINAL) == {"running", "pending"}
    # paused is in neither set -- that is precisely why the allow-list is required.
    assert "paused" not in RunRepository._NON_TERMINAL
    assert "paused" not in RunRepository._TERMINAL


async def test_paused_run_is_not_failed_by_the_reconciler(db_session, db_session_factory) -> None:
    """``paused`` is in NEITHER ``_TERMINAL`` nor ``_NON_TERMINAL``.

    A ``notin_(_TERMINAL)`` guard looks equivalent but is strictly weaker: it lets
    the reaper fail a run a reviewer had just paused, destroying the exact state
    the pause exists to hold and reaping its stages. The guard must be
    ``in_(_NON_TERMINAL)``, which is also what the orphan scan itself selects on.
    """
    repo, run, tenant_id = await _seed_run(db_session)

    assert await repo.mark_paused(run.id, reason="checkpoint") is True
    await db_session.commit()

    now = datetime.datetime.now(datetime.UTC)
    dispatcher = _leaseless_dispatcher()
    await _resolve(
        db_session, dispatcher, _stale_row(run, tenant_id, status="running", now=now), now
    )

    fresh = await _read_back(db_session_factory, run.id)
    assert fresh.status == "paused"
    assert fresh.failure_class is None
    dispatcher.reap_terminalized_run.assert_not_awaited()
