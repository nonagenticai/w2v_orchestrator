"""BOTH dispatcher pause sites record ``reason="checkpoint"`` (migration 0102).

The dispatcher pauses a run in two places: the dispatch-entry gate (a raced orphan
re-dispatched after its checkpoint committed) and the post-execution decision (the
normal "a checkpoint just completed" path). Both are review pauses. Flipping either
to ``"manual"`` would make every surface say "Paused" for a run that is waiting on
its reviewer — exactly the run 01a1770d confusion this field exists to end — so each
site is pinned separately.
"""

from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

import pytest

from src.services.runs.run_finalizer import RunDecision
from tests.unit.test_run_dispatcher_child_status import _drive_dispatcher

pytestmark = pytest.mark.asyncio


async def _paused_with(decisions: list[RunDecision]):
    with (
        patch("src.services.run_dispatcher.decide_run_terminal_state", side_effect=decisions),
        patch("src.services.run_dispatcher.pause_trigger_stage", return_value="script_draft"),
    ):
        run_repo, _redis = await _drive_dispatcher(
            executor_side_effect=None, parent_project_id=None, child_project_id=uuid4()
        )
    return run_repo


async def test_entry_gate_pause_is_a_checkpoint_pause() -> None:
    run_repo = await _paused_with([RunDecision.PAUSED])
    run_repo.mark_paused.assert_awaited_once()
    assert run_repo.mark_paused.await_args.kwargs == {"reason": "checkpoint"}
    run_repo.mark_started.assert_not_awaited()  # paused BEFORE executing anything


async def test_post_execution_pause_is_a_checkpoint_pause() -> None:
    # Entry gate says "carry on", the executor runs, then the verdict is PAUSED.
    run_repo = await _paused_with([RunDecision.CONTINUE, RunDecision.PAUSED])
    run_repo.mark_started.assert_awaited()
    run_repo.mark_paused.assert_awaited_once()
    assert run_repo.mark_paused.await_args.kwargs == {"reason": "checkpoint"}
