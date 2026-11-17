"""Unit tests for the dispatcher's cross-pod run-control channel.

A cancel/pause that lands on a replica which doesn't hold the run's execution
task publishes to ``_CONTROL_CHANNEL``; every dispatcher subscribes and cancels
the task if IT owns it. These pin the two halves (publish + handle) without a
real Redis or two live event loops.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

pytestmark = pytest.mark.asyncio


def _dispatcher(redis):
    from src.services.run_dispatcher import RunDispatcher

    return RunDispatcher(session_factory=MagicMock(), registry=MagicMock(), redis=redis)


async def test_broadcast_interrupt_publishes_run_id() -> None:
    from src.services.run_dispatcher import _CONTROL_CHANNEL

    redis = AsyncMock()
    disp = _dispatcher(redis)
    run_id = uuid4()

    await disp.broadcast_interrupt(run_id)

    redis.publish.assert_awaited_once()
    channel, raw = redis.publish.await_args.args
    assert channel == _CONTROL_CHANNEL
    assert json.loads(raw) == {"run_id": str(run_id)}


async def test_handle_control_cancels_owned_task() -> None:
    disp = _dispatcher(AsyncMock())
    run_id = uuid4()
    task = MagicMock()
    disp._tasks[run_id] = task

    disp._handle_control(json.dumps({"run_id": str(run_id)}))

    task.cancel.assert_called_once()


async def test_handle_control_ignores_unowned_run() -> None:
    disp = _dispatcher(AsyncMock())
    # A task we own, for a DIFFERENT run — must not be cancelled.
    other = MagicMock()
    disp._tasks[uuid4()] = other

    disp._handle_control(json.dumps({"run_id": str(uuid4())}))

    other.cancel.assert_not_called()


async def test_handle_control_tolerates_malformed_messages() -> None:
    disp = _dispatcher(AsyncMock())
    task = MagicMock()
    disp._tasks[uuid4()] = task

    # None of these should raise or cancel anything.
    disp._handle_control(b"not json")
    disp._handle_control(json.dumps({"nope": 1}))
    disp._handle_control(json.dumps({"run_id": "not-a-uuid"}))

    task.cancel.assert_not_called()
