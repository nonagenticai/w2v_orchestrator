"""Paused time must not count against the run deadline (migration 0094).

A course paused for human review longer than ``run_deadline_seconds`` used to be
"expired" the instant it resumed, because the deadline was the bare wall clock
``started_at + run_deadline_seconds``. Now ``mark_paused`` stamps ``paused_at``,
``mark_resumed`` folds the pause into ``paused_seconds``, and every deadline
computation adds ``paused_seconds`` back.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import text

from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository
from src.services import run_dispatcher
from src.services.runs import sub_dag
from tests.services.runs.test_sub_dag import _patched_runs, _session_factory

_FIVE_HOURS = 5 * 3600
_BUDGET = 14400  # the production run_deadline_seconds (4h)


def _settings(budget: int) -> SimpleNamespace:
    return SimpleNamespace(pipeline=SimpleNamespace(run_deadline_seconds=budget))


async def _new_run(session):
    tenant = uuid4()
    project = await ProjectRepository(session).create_with_deliverables(
        tenant_id=tenant,
        owner_subject_id=uuid4(),
        type_="course",
        title="paused-course",
        brief={},
        brand={},
        voice={},
        theme={},
        visibility="tenant",
        deliverables=[{"name": "c", "format": "16x9", "captions_mode": "none"}],
    )
    await session.commit()
    repo = RunRepository(session)
    run = await repo.create(
        deliverable_id=project.deliverables[0].id,
        blueprint_id="course_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    await session.commit()
    return repo, run.id


@pytest.mark.asyncio
async def test_pause_five_hours_then_resume_folds_the_pause_into_paused_seconds(
    db_session,
) -> None:
    repo, run_id = await _new_run(db_session)
    assert await repo.mark_started(run_id)
    assert await repo.mark_paused(run_id, reason="checkpoint")
    await db_session.commit()

    row = await repo.get(run_id)
    assert row is not None
    await db_session.refresh(row)
    assert row.paused_at is not None, "mark_paused must start the pause clock"
    assert row.paused_seconds == 0

    # Simulate a reviewer who took 5h: started 5h05m ago, paused for the last 5h.
    await db_session.execute(
        text(
            "UPDATE runs SET started_at = now() - interval '5 hours 5 minutes', "
            "paused_at = now() - interval '5 hours' WHERE id = :id"
        ),
        {"id": run_id},
    )
    await db_session.commit()

    assert await repo.mark_resumed(run_id)
    await db_session.commit()
    await db_session.refresh(row)
    assert row.status == "running"
    assert row.paused_at is None, "resume must stop the pause clock"
    assert _FIVE_HOURS - 5 <= row.paused_seconds <= _FIVE_HOURS + 5

    # The deadline the dispatcher builds from this row is NOT expired.
    budget_patch = pytest.MonkeyPatch()
    budget_patch.setattr(run_dispatcher, "get_settings", lambda: _settings(_BUDGET))
    try:
        deadline = run_dispatcher._run_deadline_at(row.started_at, row.paused_seconds)
        bare = run_dispatcher._run_deadline_at(row.started_at)
    finally:
        budget_patch.undo()
    assert deadline is not None and bare is not None
    now = datetime.now(UTC)
    assert bare < now, "without the pause credit the 4h budget is already spent"
    assert deadline > now + timedelta(hours=3), "5h05m - 5h paused = 5 min of a 4h budget"


@pytest.mark.asyncio
async def test_a_second_pause_accumulates_and_a_repause_keeps_the_original_stamp(
    db_session,
) -> None:
    repo, run_id = await _new_run(db_session)
    await repo.mark_started(run_id)
    await repo.mark_paused(run_id, reason="checkpoint")
    await db_session.commit()
    await db_session.execute(
        text("UPDATE runs SET paused_at = now() - interval '1 hour' WHERE id = :id"),
        {"id": run_id},
    )
    await db_session.commit()
    # A no-op re-pause (row already paused) must not reset the clock.
    assert await repo.mark_paused(run_id, reason="checkpoint") is False
    await repo.mark_resumed(run_id)
    await db_session.commit()
    await repo.mark_paused(run_id, reason="checkpoint")
    await db_session.commit()
    await db_session.execute(
        text("UPDATE runs SET paused_at = now() - interval '2 hours' WHERE id = :id"),
        {"id": run_id},
    )
    await db_session.commit()
    await repo.mark_resumed(run_id)
    await db_session.commit()
    row = await repo.get(run_id)
    assert row is not None
    await db_session.refresh(row)
    assert 3 * 3600 - 5 <= row.paused_seconds <= 3 * 3600 + 5


def test_dispatcher_deadline_adds_paused_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_dispatcher, "get_settings", lambda: _settings(_BUDGET))
    started = datetime(2026, 10, 19, 8, 0, tzinfo=UTC)
    assert run_dispatcher._run_deadline_at(started) == started + timedelta(seconds=_BUDGET)
    assert run_dispatcher._run_deadline_at(started, _FIVE_HOURS) == started + timedelta(
        seconds=_BUDGET + _FIVE_HOURS
    )


@pytest.mark.asyncio
async def test_child_wait_budget_credits_the_parents_paused_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A course resumed after a 5h review must not give its lessons the 60s floor."""
    parent = uuid4()
    monkeypatch.setattr(sub_dag, "get_settings", lambda: _settings(_BUDGET))
    row = SimpleNamespace(
        status="running",
        error=None,
        started_at=datetime.now(UTC) - timedelta(seconds=_FIVE_HOURS + 300),
        paused_seconds=_FIVE_HOURS,
    )
    with _patched_runs({parent: row}):
        bound, source = await sub_dag._derive_child_wait_s(parent, _session_factory, None, 1)
    assert bound == sub_dag._FALLBACK_CHILD_WAIT_S, source
    assert f"+{_FIVE_HOURS}s paused" in source
