"""``child_status_changed`` mirror of run lifecycle onto the parent's SSE channel.

PR #68 added ``projects:<id>`` as a sibling SSE channel that the
project-detail FE subscribes to. ``RunDispatcher`` now mirrors each run's
lifecycle transition onto the parent project's channel (when the run's
project is a child of another project). These tests pin that mirror:

* a run on a child project publishes ``child_status_changed`` on the
  parent's channel for terminal + intermediate states;
* a run on a top-level project (``parent_project_id IS NULL``) does NOT
  publish anything to a ``projects:`` channel.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

pytestmark = pytest.mark.asyncio


class _FakeRedis:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def publish(self, channel: str, message: str) -> int:
        self.published.append((channel, json.loads(message)))
        return 1


def _make_run_obj() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        blueprint_id="bp-1",
        deliverable_id=uuid4(),
        config_snapshot={"tenant_id": str(uuid4()), "subject_id": str(uuid4())},
        pause_after_stages=[],
        # Real column; the dispatcher reads it to anchor the run deadline.
        started_at=None,
    )


def _session_factory(session: AsyncMock):
    @asynccontextmanager
    async def _factory():
        yield session

    return _factory


async def _drive_dispatcher(
    executor_side_effect,
    *,
    parent_project_id: UUID | None,
    child_project_id: UUID | None,
    cancel_before_complete: bool = False,
) -> tuple[AsyncMock, _FakeRedis]:
    from src.services.run_dispatcher import RunDispatcher

    run_obj = _make_run_obj()
    session = AsyncMock()
    session.commit = AsyncMock()

    # Pin session.execute(...).first() to return our (child, parent) row so
    # the dispatcher's parent-project lookup produces deterministic values.
    proj_row = SimpleNamespace(id=child_project_id, parent_project_id=parent_project_id)
    exec_result = MagicMock()
    exec_result.first = MagicMock(return_value=proj_row)
    session.execute = AsyncMock(return_value=exec_result)

    run_repo_instance = AsyncMock()
    run_repo_instance.get = AsyncMock(return_value=run_obj)
    run_repo_instance.mark_started = AsyncMock()
    run_repo_instance.mark_failed = AsyncMock()
    run_repo_instance.mark_cancelled = AsyncMock()
    run_repo_instance.mark_completed = AsyncMock()
    run_repo_instance.mark_paused = AsyncMock()
    run_repo_instance.mark_current_stage = AsyncMock()

    stage_repo_instance = AsyncMock()
    stage_repo_instance.list_for_run = AsyncMock(return_value=[])

    registry = MagicMock()
    blueprint = MagicMock()
    blueprint.topological_order.return_value = []
    registry.resolve_by_id.return_value = blueprint

    redis = _FakeRedis()

    executor_instance = MagicMock()
    executor_instance.run = AsyncMock(side_effect=executor_side_effect)

    with (
        patch("src.services.run_dispatcher.RunRepository", return_value=run_repo_instance),
        patch(
            "src.services.run_dispatcher.StageOutputRepository",
            return_value=stage_repo_instance,
        ),
        # Project-status recompute issues real aggregation queries; these tests
        # mock the session, so stub the repo. Recompute behavior is covered by
        # the repository + dispatcher integration tests.
        patch("src.services.run_dispatcher.ProjectRepository", return_value=AsyncMock()),
        patch("src.services.run_dispatcher.DAGExecutor", return_value=executor_instance),
    ):
        dispatcher = RunDispatcher(
            session_factory=_session_factory(session),
            registry=registry,
            redis=redis,  # ty: ignore[invalid-argument-type]
        )
        await dispatcher.dispatch(run_obj.id)
        if cancel_before_complete:
            await asyncio.sleep(0)
            await dispatcher.shutdown()
        else:
            task = dispatcher._tasks.get(run_obj.id)
            if task is not None:
                await asyncio.wait_for(task, timeout=2.0)

    return run_repo_instance, redis


def _child_status_events(redis: _FakeRedis, parent_project_id: UUID) -> list[dict]:
    expected_channel = f"projects:{parent_project_id}"
    return [
        msg
        for channel, msg in redis.published
        if channel == expected_channel and msg.get("event") == "child_status_changed"
    ]


def _any_project_channel_publishes(redis: _FakeRedis) -> list[tuple[str, dict]]:
    return [(c, m) for c, m in redis.published if c.startswith("projects:")]


async def test_run_completion_publishes_child_status_changed() -> None:
    """Empty-blueprint run on a child project → completed event on parent's channel."""
    parent = uuid4()
    child = uuid4()
    _run_repo, redis = await _drive_dispatcher(
        executor_side_effect=None,  # AsyncMock(side_effect=None) returns None cleanly
        parent_project_id=parent,
        child_project_id=child,
    )
    statuses = [m["status"] for m in _child_status_events(redis, parent)]
    # ``running`` is fired at dispatch start; ``completed`` is fired when the
    # empty blueprint's topological_order ([]) matches the empty
    # completed-stages set (this is the happy "blueprint with no stages"
    # path the existing failure-tests file leans on).
    assert "running" in statuses
    assert "completed" in statuses
    for msg in _child_status_events(redis, parent):
        assert msg["project_id"] == str(child)


async def test_run_failure_publishes_child_status_changed_failed() -> None:
    """Executor raises → ``failed`` mirror on parent's channel."""
    parent = uuid4()
    child = uuid4()
    _run_repo, redis = await _drive_dispatcher(
        executor_side_effect=RuntimeError("upstream blew up"),
        parent_project_id=parent,
        child_project_id=child,
    )
    statuses = [m["status"] for m in _child_status_events(redis, parent)]
    assert "running" in statuses
    assert "failed" in statuses
    # ``completed`` must NOT be published on a failed run.
    assert "completed" not in statuses


async def test_shutdown_drain_does_not_publish_cancelled() -> None:
    """Shutdown drain leaves runs as ``running`` for the next pod's startup
    reconciler — no ``cancelled`` publish, no terminal mirror.

    Pre-Phase-1b this test asserted ``cancelled`` was published. The
    new behaviour: ``RunDispatcher.shutdown()`` flips ``_draining=True``
    BEFORE force-cancelling tasks, and the CancelledError handler then
    leaves the run row in ``running`` rather than marking it
    ``cancelled``. This is what makes durable-resume possible — without
    it, a deploy would still kill every in-flight run.
    """

    async def block_forever(*_args, **_kwargs):
        await asyncio.sleep(60)

    parent = uuid4()
    child = uuid4()
    run_repo, redis = await _drive_dispatcher(
        executor_side_effect=block_forever,
        parent_project_id=parent,
        child_project_id=child,
        cancel_before_complete=True,
    )
    statuses = [m["status"] for m in _child_status_events(redis, parent)]
    # `running` is still mirrored at dispatch start; `cancelled` is NOT.
    assert "running" in statuses
    assert "cancelled" not in statuses
    # The runs:<id> channel must also not see `run_cancelled` on a drain.
    run_channel_events = [m.get("event") for c, m in redis.published if c.startswith("runs:")]
    assert "run_cancelled" not in run_channel_events
    # mark_cancelled must not be called on a drain — that's the entire
    # point of distinguishing drain-cancel from explicit-cancel.
    run_repo.mark_cancelled.assert_not_called()


async def test_explicit_cancel_still_publishes_cancelled() -> None:
    """``dispatcher.cancel(run_id)`` (cancel endpoint, not shutdown drain)
    still drives the run to terminal ``cancelled`` — explicit user cancel
    is the path Phase 1b deliberately preserved."""
    from src.services.run_dispatcher import RunDispatcher

    async def block_forever(*_args, **_kwargs):
        await asyncio.sleep(60)

    parent = uuid4()
    child = uuid4()
    run_obj = _make_run_obj()
    session = AsyncMock()
    session.commit = AsyncMock()
    proj_row = SimpleNamespace(id=child, parent_project_id=parent)
    exec_result = MagicMock()
    exec_result.first = MagicMock(return_value=proj_row)
    session.execute = AsyncMock(return_value=exec_result)

    run_repo_instance = AsyncMock()
    run_repo_instance.get = AsyncMock(return_value=run_obj)
    run_repo_instance.mark_started = AsyncMock()
    run_repo_instance.mark_cancelled = AsyncMock()
    run_repo_instance.mark_completed = AsyncMock()
    run_repo_instance.mark_paused = AsyncMock()
    run_repo_instance.mark_failed = AsyncMock()
    run_repo_instance.mark_current_stage = AsyncMock()

    stage_repo_instance = AsyncMock()
    stage_repo_instance.list_for_run = AsyncMock(return_value=[])

    registry = MagicMock()
    blueprint = MagicMock()
    blueprint.topological_order.return_value = []
    registry.resolve_by_id.return_value = blueprint

    redis = _FakeRedis()

    executor_instance = MagicMock()
    executor_instance.run = AsyncMock(side_effect=block_forever)

    with (
        patch("src.services.run_dispatcher.RunRepository", return_value=run_repo_instance),
        patch(
            "src.services.run_dispatcher.StageOutputRepository",
            return_value=stage_repo_instance,
        ),
        patch("src.services.run_dispatcher.ProjectRepository", return_value=AsyncMock()),
        patch("src.services.run_dispatcher.DAGExecutor", return_value=executor_instance),
    ):
        dispatcher = RunDispatcher(
            session_factory=_session_factory(session),
            registry=registry,
            redis=redis,  # ty: ignore[invalid-argument-type]
        )
        await dispatcher.dispatch(run_obj.id)
        await asyncio.sleep(0)
        # `cancel()` does NOT touch _draining, so the CancelledError
        # handler hits the explicit-cancel branch.
        cancelled_ok = await dispatcher.cancel(run_obj.id)
        assert cancelled_ok is True

    statuses = [m["status"] for m in _child_status_events(redis, parent)]
    assert "cancelled" in statuses
    run_repo_instance.mark_cancelled.assert_called_once_with(run_obj.id)


async def test_top_level_project_run_does_not_publish_to_parent() -> None:
    """Run on a project with ``parent_project_id IS NULL`` → no projects:* publishes."""
    _run_repo, redis = await _drive_dispatcher(
        executor_side_effect=None,
        parent_project_id=None,
        child_project_id=uuid4(),
    )
    assert _any_project_channel_publishes(redis) == []
    # The runs: channel must still see lifecycle events — only the
    # projects:* mirror is suppressed for top-level runs.
    run_events = [msg["event"] for channel, msg in redis.published if channel.startswith("runs:")]
    assert "run_started" in run_events
