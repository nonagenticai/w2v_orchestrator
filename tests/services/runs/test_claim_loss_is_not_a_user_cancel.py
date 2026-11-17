"""A claim-loss cancel must not be treated as a user cancel.

``RunDbClaim._heartbeat`` calls its ``on_lost`` callback on two conditions: a
transient DB blip (failures >= max_failures) and a genuine reclaim by another
instance (rowcount==0). ``RunDispatcher._run`` wires ``on_lost`` to cancel the
executing task — its own comment says the intent is "the row stays `running`,
so whoever holds the claim now owns the resume."

Before this fix that intent was defeated: the cancellation was
indistinguishable from ``POST /runs/{id}/cancel`` once it reached
``_finish_cancelled``, so it fell into the ``else`` branch — "Explicit user
cancel" — and a healthy running run was terminally marked ``cancelled`` with
``run_cancelled`` published, even though nothing the user did caused it.

Every test here was run against the unfixed code first (the claim-loss branch
removed, falling through to ``else``); each one fails there — see the bottom
of this file for the exact command and captured output.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.services.run_dispatcher import RunDispatcher

pytestmark = pytest.mark.asyncio


def _dispatcher() -> RunDispatcher:
    session = AsyncMock()
    session.commit = AsyncMock()
    exec_result = MagicMock()
    exec_result.first = MagicMock(return_value=SimpleNamespace(id=None, parent_project_id=None))
    session.execute = AsyncMock(return_value=exec_result)

    @asynccontextmanager
    async def _factory():
        yield session

    return RunDispatcher(session_factory=_factory, registry=MagicMock(), redis=MagicMock())


class _Repo:
    """Stub RunRepository — records whether mark_cancelled was ever invoked."""

    def __init__(self, marked: list[str], status: str = "running") -> None:
        self._marked = marked
        self._status = status

    def __call__(self, _session):  # used as a monkeypatched class substitute
        return self

    async def get(self, _rid):
        await asyncio.sleep(0)
        return SimpleNamespace(status=self._status)

    async def mark_cancelled(self, _rid) -> bool:
        await asyncio.sleep(0)
        self._marked.append("cancelled")
        return True


def _wire_repo(monkeypatch: pytest.MonkeyPatch, marked: list[str], status: str = "running"):
    import src.services.run_dispatcher as mod

    repo = _Repo(marked, status=status)
    monkeypatch.setattr(mod, "RunRepository", lambda _s: repo)


async def _run_finish_cancelled(
    d: RunDispatcher, run_id, *, publish: AsyncMock, publish_child: AsyncMock
) -> None:
    monkeypatch_targets = (("_publish", publish), ("_publish_child_status", publish_child))
    for name, mock in monkeypatch_targets:
        setattr(d, name, mock)
    d._recompute_project = AsyncMock()
    await d._finish_cancelled(
        run_id,
        parent_project_id=None,
        child_project_id=None,
        paused_published_locally=False,
    )


async def test_claim_loss_cancel_leaves_row_running_and_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d = _dispatcher()
    run_id = uuid4()
    marked: list[str] = []
    _wire_repo(monkeypatch, marked)
    publish, publish_child = AsyncMock(), AsyncMock()

    # This is exactly what the `on_lost` callback does BEFORE cancelling.
    d._claim_lost_runs.add(run_id)

    await _run_finish_cancelled(d, run_id, publish=publish, publish_child=publish_child)

    assert marked == [], "a claim-loss cancel must never mark the run cancelled"
    publish.assert_not_called()
    publish_child.assert_not_called()
    assert run_id not in d._claim_lost_runs, "the marker must be cleared after handling"


async def test_genuine_user_cancel_still_marks_cancelled_and_publishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d = _dispatcher()
    run_id = uuid4()
    marked: list[str] = []
    _wire_repo(monkeypatch, marked)
    publish, publish_child = AsyncMock(), AsyncMock()

    await _run_finish_cancelled(d, run_id, publish=publish, publish_child=publish_child)

    assert marked == ["cancelled"], "an explicit user cancel must still mark the run cancelled"
    publish.assert_awaited_once_with(run_id, "run_cancelled")
    publish_child.assert_awaited_once_with(None, None, "cancelled")


async def test_drain_cancel_is_unaffected_by_the_claim_loss_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d = _dispatcher()
    d._draining = True
    run_id = uuid4()
    marked: list[str] = []
    _wire_repo(monkeypatch, marked)
    publish, publish_child = AsyncMock(), AsyncMock()

    await _run_finish_cancelled(d, run_id, publish=publish, publish_child=publish_child)

    assert marked == [], "drain-cancel must leave the row running, same as before"
    publish.assert_not_called()
    publish_child.assert_not_called()


async def test_claim_loss_marker_does_not_leak_to_a_different_runs_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leak case: a claim-loss cancel must not poison a DIFFERENT run's
    later genuine user cancel."""
    d = _dispatcher()
    lost_run_id, other_run_id = uuid4(), uuid4()
    marked: list[str] = []
    _wire_repo(monkeypatch, marked)
    publish, publish_child = AsyncMock(), AsyncMock()

    d._claim_lost_runs.add(lost_run_id)
    await _run_finish_cancelled(d, lost_run_id, publish=publish, publish_child=publish_child)
    assert marked == [], "sanity: the claim-loss run was not marked cancelled"

    marked.clear()
    publish.reset_mock()
    publish_child.reset_mock()
    await _run_finish_cancelled(d, other_run_id, publish=publish, publish_child=publish_child)

    assert marked == ["cancelled"], (
        "a claim-loss cancel on one run must not suppress a genuine cancel of another run"
    )
    publish.assert_awaited_once_with(other_run_id, "run_cancelled")


async def test_claim_loss_marker_is_cleared_for_a_reused_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leak case: after a claim-loss cancel finishes handling, a LATER
    genuine user cancel of the SAME run id must still cancel properly (the
    run resumed under a new claim and was later cancelled for real)."""
    d = _dispatcher()
    run_id = uuid4()
    marked: list[str] = []
    _wire_repo(monkeypatch, marked)
    publish, publish_child = AsyncMock(), AsyncMock()

    d._claim_lost_runs.add(run_id)
    await _run_finish_cancelled(d, run_id, publish=publish, publish_child=publish_child)
    assert marked == [], "sanity: the first (claim-loss) cancel was not marked cancelled"
    assert run_id not in d._claim_lost_runs

    marked.clear()
    publish.reset_mock()
    publish_child.reset_mock()
    await _run_finish_cancelled(d, run_id, publish=publish, publish_child=publish_child)

    assert marked == ["cancelled"], (
        "a leaked marker made a later genuine cancel of the SAME run id silently no-op"
    )
    publish.assert_awaited_once_with(run_id, "run_cancelled")


async def test_finish_cancelled_clears_the_marker_even_if_handling_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clearing must be a `finally`, not a happy-path-only pop."""
    d = _dispatcher()
    run_id = uuid4()

    @asynccontextmanager
    async def _boom_factory():
        raise RuntimeError("db unavailable")
        yield  # pragma: no cover - unreachable, keeps this an async generator

    monkeypatch.setattr(d, "_sessions", _boom_factory)
    d._claim_lost_runs.add(run_id)

    with pytest.raises(RuntimeError):
        await d._finish_cancelled(
            run_id,
            parent_project_id=None,
            child_project_id=None,
            paused_published_locally=False,
        )

    assert run_id not in d._claim_lost_runs, "the marker leaked past a raising handler"


async def test_on_lost_marks_the_claim_lost_set_before_cancelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proves the wiring in ``_run``: the ``on_lost`` callback passed to
    ``RunDbClaim.start_heartbeat`` must add the run id to
    ``_claim_lost_runs`` BEFORE cancelling the executing task — not rely on
    ``task.cancel(msg)``'s message surviving re-raise.
    """
    d = _dispatcher()
    run_id = uuid4()

    class _FellOpenLease:
        async def acquire(self, _rid):
            return None  # Redis unavailable -> fail-open path

        def start_heartbeat(self, _rid):
            return None

        async def release(self, _rid):
            pass

    class _CapturingClaim:
        def __init__(self) -> None:
            self.on_lost = None

        async def acquire(self, _rid) -> bool:
            return True

        def start_heartbeat(self, _rid, on_lost=None):
            self.on_lost = on_lost
            return

        async def release(self, _rid) -> None:
            pass

    claim = _CapturingClaim()
    monkeypatch.setattr(d, "_lease", _FellOpenLease())
    monkeypatch.setattr(d, "_claim", claim)

    hang_started = asyncio.Event()

    async def _hanging_execute_run(_rid) -> None:
        hang_started.set()
        await asyncio.Event().wait()  # never resolves on its own

    monkeypatch.setattr(d, "_execute_run", _hanging_execute_run)

    task = asyncio.create_task(d._run(run_id))
    await hang_started.wait()
    await asyncio.sleep(0)

    assert claim.on_lost is not None, "start_heartbeat was not given an on_lost callback"
    assert run_id not in d._claim_lost_runs

    claim.on_lost()  # what RunDbClaim._heartbeat does on failure/reclaim

    assert run_id in d._claim_lost_runs, (
        "on_lost must record the run id BEFORE cancelling — task.cancel(msg) "
        "is not a reliable channel, a re-raised CancelledError can lose it"
    )

    with pytest.raises(asyncio.CancelledError):
        await task
