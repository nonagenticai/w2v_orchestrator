"""Unit tests for the SSE catch-up snapshot (D6 — SSE early-event loss).

The snapshot replays current run state as the same event shapes the dispatcher
publishes live, so a client connecting mid-run (or after a fast run finished) is
brought up to date instead of waiting on events Redis pub/sub already dropped.
These exercise the pure event-shaping logic with a mocked repository — no DB or
Redis needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.routes.runs import _build_run_event_snapshot

pytestmark = pytest.mark.asyncio


def _so(stage: str, status: str, error: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(stage_name=stage, status=status, error=error)


async def _snapshot(run: SimpleNamespace, stage_rows: list[SimpleNamespace]) -> list[dict]:
    repo = AsyncMock()
    repo.list_for_run.return_value = stage_rows
    with patch("src.api.routes.runs.events.StageOutputRepository", return_value=repo):
        # AsyncMock session + SimpleNamespace run stand in for the real
        # AsyncSession/Run — the helper only touches run.id/.status/.error and
        # the (patched) repository, so the runtime types are irrelevant.
        return await _build_run_event_snapshot(AsyncMock(), run)  # ty: ignore[invalid-argument-type]


async def test_pending_run_yields_empty_snapshot() -> None:
    run = SimpleNamespace(id=uuid4(), status="pending", error=None, current_stage=None)
    assert await _snapshot(run, []) == []


async def test_completed_run_replays_started_stages_then_terminal() -> None:
    rid = uuid4()
    run = SimpleNamespace(id=rid, status="completed", error=None, current_stage=None)
    events = await _snapshot(run, [_so("a", "completed"), _so("b", "completed")])
    names = [e["event"] for e in events]
    assert names[0] == "run_started"
    assert names[-1] == "run_completed"
    assert names.count("stage_completed") == 2
    assert all(e["run_id"] == str(rid) for e in events)


async def test_running_stage_gets_started_but_not_completed_and_no_terminal() -> None:
    run = SimpleNamespace(id=uuid4(), status="running", error=None, current_stage="b")
    events = await _snapshot(run, [_so("a", "completed"), _so("b", "running"), _so("c", "pending")])
    stage_events = {(e["stage_name"], e["event"]) for e in events if e["event"].startswith("stage")}
    assert ("a", "stage_started") in stage_events
    assert ("a", "stage_completed") in stage_events
    assert ("b", "stage_started") in stage_events
    assert ("b", "stage_completed") not in stage_events
    # A pending stage has not been announced yet — nothing for it.
    assert not any(name == "c" for name, _ in stage_events)
    # Still running → no terminal event.
    assert not any(
        e["event"] in ("run_completed", "run_failed", "run_paused", "run_cancelled") for e in events
    )


async def test_failed_stage_and_run_carry_error() -> None:
    run = SimpleNamespace(id=uuid4(), status="failed", error="boom", current_stage="b")
    events = await _snapshot(
        run, [_so("a", "completed"), _so("b", "failed", error="downstream 500")]
    )
    stage_failed = next(e for e in events if e["event"] == "stage_failed")
    assert stage_failed["stage_name"] == "b"
    assert stage_failed["error"] == "downstream 500"
    run_failed = next(e for e in events if e["event"] == "run_failed")
    assert run_failed["error"] == "boom"


async def test_cancelled_and_paused_terminals() -> None:
    for status, terminal in (("cancelled", "run_cancelled"), ("paused", "run_paused")):
        run = SimpleNamespace(id=uuid4(), status=status, error=None, current_stage="a")
        events = await _snapshot(run, [_so("a", "completed")])
        assert events[-1]["event"] == terminal
        assert events[0]["event"] == "run_started"


async def test_paused_snapshot_names_the_reviewable_checkpoint_not_a_skipped_stage() -> None:
    """The paused terminal event must carry the stage under review.

    The replay emits a ``stage_started`` for EVERY non-pending row, so a client
    that reconnects ends up with "current stage" = whichever row came last —
    frequently a sibling that started in the same concurrent wave and then
    SKIPPED. Measured live: a demo_video run paused at ``demo_video_gen`` but the
    UI offered ``source_ingest`` (skipped, never ran, no draft) for review. The
    dispatcher re-points ``current_stage`` at the triggering checkpoint on pause,
    and the snapshot must forward it.
    """
    run = SimpleNamespace(id=uuid4(), status="paused", error=None, current_stage="demo_video_gen")
    events = await _snapshot(
        run,
        [_so("source_ingest", "skipped"), _so("demo_video_gen", "completed")],
    )
    run_paused = events[-1]
    assert run_paused["event"] == "run_paused"
    assert run_paused["stage_name"] == "demo_video_gen"


async def test_non_paused_terminals_carry_no_stage_name() -> None:
    # Only the pause gate names a review target; completion/cancellation do not.
    run = SimpleNamespace(id=uuid4(), status="cancelled", error=None, current_stage="a")
    events = await _snapshot(run, [_so("a", "completed")])
    assert "stage_name" not in events[-1]


# ---------------------------------------------------------------------------
# _project_id_for_deliverable — the orphaned-run 500 → clean 404 fix.
# A vanished deliverable/project chain (deleted mid-flight, or a genuinely
# orphaned run) used to raise an uncaught NoResultFound (scalar_one) → 500.
# It now returns a clean 404. Pure: the DB layer is mocked.
# ---------------------------------------------------------------------------
def _session_returning(scalar_value: object) -> AsyncMock:
    result = MagicMock()
    result.scalar_one_or_none.return_value = scalar_value
    session = AsyncMock()
    session.execute.return_value = result
    return session
