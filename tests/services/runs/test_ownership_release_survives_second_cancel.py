"""A cancelled run must still hand back its lease, its DB claim, and its row.

Three defects, one shape: the dispatcher's teardown ran as bare awaits inside a
`finally:` / `except CancelledError:` that is reached *precisely when a cancel is
in flight*. A SECOND cancel — `POST /runs/{id}/cancel` followed by pod SIGTERM,
which cancels the same task again — landed on one of those awaits and abandoned
the rest.

What each abandonment costs:

* the Redis lease outlives its TTL and `exec_claimed_at` stays fresh, so BOTH
  ownership guards report "alive" for a run that has no owner — the reconciler
  then refuses to re-dispatch it and it sits `running`;
* `mark_cancelled` never commits, so the run the user cancelled is picked up by
  that same reconciler, resumes, and keeps spending.

`except asyncio.CancelledError` is not the fix — it is a `BaseException`, and by
the time it is raised the await has already been interrupted. `asyncio.shield` is.

Every test here was run against the unfixed code first; each one fails there.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.services.run_dispatcher import RunDispatcher


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


class _SlowLease:
    """Yields control on the release await, so a cancel lands where a real one would."""

    def __init__(self) -> None:
        self.released = False

    async def release(self, _run_id) -> None:
        await asyncio.sleep(0)
        self.released = True


class _SlowClaim(_SlowLease):
    pass


@pytest.mark.asyncio
async def test_lease_and_db_claim_are_released_even_when_teardown_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d = _dispatcher()
    lease, claim = _SlowLease(), _SlowClaim()
    monkeypatch.setattr(d, "_lease", lease)
    monkeypatch.setattr(d, "_claim", claim)

    async def _teardown() -> None:
        await asyncio.shield(
            d._release_ownership(
                uuid4(), heartbeat=None, claim_heartbeat=None, own_lease=True, db_claimed=True
            )
        )

    task = asyncio.create_task(_teardown())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert lease.released, "the Redis lease was abandoned — the run has no owner but reads alive"
    assert claim.released, "the durable DB claim was abandoned — the reconciler will skip this run"


@pytest.mark.asyncio
async def test_a_heartbeat_that_died_of_a_non_cancel_error_does_not_skip_the_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`suppress(CancelledError)` covered only cancellation.

    A heartbeat task that had already died of e.g. a `RedisError` re-raised at the
    `await heartbeat` and skipped BOTH releases below it. `_FakeRedis` raising
    `AttributeError` from `raw.set` is a live example of a non-cancelled death.
    """
    d = _dispatcher()
    lease, claim = _SlowLease(), _SlowClaim()
    monkeypatch.setattr(d, "_lease", lease)
    monkeypatch.setattr(d, "_claim", claim)

    async def _doomed() -> None:
        raise AttributeError("'_FakeRedis' object has no attribute 'set'")

    hb = asyncio.create_task(_doomed())
    await asyncio.sleep(0)  # let it die before we cancel-and-await it

    await d._release_ownership(
        uuid4(), heartbeat=hb, claim_heartbeat=None, own_lease=True, db_claimed=True
    )

    assert lease.released, "a dead heartbeat's exception skipped the lease release"
    assert claim.released, "a dead heartbeat's exception skipped the DB claim release"


@pytest.mark.asyncio
async def test_a_db_error_releasing_the_claim_does_not_mask_the_run_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d = _dispatcher()

    class _Exploding:
        async def release(self, _run_id) -> None:
            raise RuntimeError("pgbouncer said no")

    lease = _SlowLease()
    monkeypatch.setattr(d, "_lease", lease)
    monkeypatch.setattr(d, "_claim", _Exploding())

    # This runs in a `finally`, where an exception would REPLACE whatever the run
    # actually failed with.
    raised: BaseException | None = None
    try:
        await d._release_ownership(
            uuid4(), heartbeat=None, claim_heartbeat=None, own_lease=True, db_claimed=True
        )
    except BaseException as exc:  # noqa: BLE001 — the assertion IS "nothing escapes"
        raised = exc

    assert raised is None, f"a claim-release error escaped and would mask the cause: {raised!r}"
    assert lease.released, "the lease release did not run"


@pytest.mark.asyncio
async def test_a_second_cancel_still_marks_the_run_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The money one: without the shield the row stays `running` and resumes."""
    d = _dispatcher()
    run_id = uuid4()
    marked: list[str] = []

    class _Repo:
        def __init__(self, _s) -> None:
            pass

        async def get(self, _rid):
            await asyncio.sleep(0)
            return SimpleNamespace(status="running")

        async def mark_cancelled(self, _rid) -> bool:
            await asyncio.sleep(0)
            marked.append("cancelled")
            return True

    import src.services.run_dispatcher as mod

    monkeypatch.setattr(mod, "RunRepository", _Repo)
    monkeypatch.setattr(d, "_publish", AsyncMock())
    monkeypatch.setattr(d, "_publish_child_status", AsyncMock())
    monkeypatch.setattr(d, "_recompute_project", AsyncMock())

    async def _handler() -> None:
        await asyncio.shield(
            d._finish_cancelled(
                run_id,
                parent_project_id=None,
                child_project_id=None,
                paused_published_locally=False,
            )
        )

    task = asyncio.create_task(_handler())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(5):
        await asyncio.sleep(0)

    assert marked == ["cancelled"], (
        "the second cancel abandoned the transition — the row stays 'running', "
        "the reconciler re-dispatches it, and the cancelled run keeps spending"
    )


@pytest.mark.asyncio
async def test_shielded_cleanup_is_retained_and_drained_by_shutdown() -> None:
    """A shield protects cleanup from cancellation but does NOT keep it alive.

    `await asyncio.shield(x)` stops the caller's cancel reaching `x`, then the
    caller stops awaiting. asyncio holds only a WEAK reference to a task, so `x`
    can be collected mid-transaction, or simply cut off when the lifespan closes
    the loop — "Task was destroyed but it is pending". The shielded work here is
    the terminal transition of a cancelled run, so losing it reintroduces the
    exact defect the shield was added to fix.

    `_shielded` therefore retains the task, and `shutdown` drains the set.
    """
    d = _dispatcher()
    finished: list[str] = []

    async def _slow_cleanup() -> None:
        await asyncio.sleep(0.05)
        finished.append("done")

    async def _caller() -> None:
        await d._shielded(_slow_cleanup())

    task = asyncio.create_task(_caller())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert d._cleanup_tasks, "the shielded cleanup was not retained — it can be collected"

    await d.shutdown()

    assert finished == ["done"], (
        "shutdown returned while cleanup was still in flight; the lifespan would "
        "close the engine underneath a half-committed cancel"
    )


def test_both_production_call_sites_are_actually_shielded() -> None:
    """The tests above shield inside the harness — that proves the coroutine is
    SAFE to shield, not that the dispatcher shields it. Without this guard,
    deleting `asyncio.shield` at either call site leaves all four green while
    reintroducing the exact defect. Read the source and assert the wrapping.
    """
    import ast
    import inspect

    import src.services.run_dispatcher as mod

    tree = ast.parse(inspect.getsource(mod))
    shielded: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        # `_shielded` is the wrapper that shields AND retains; a bare
        # `asyncio.shield` would protect the cleanup from cancellation but leave
        # it unreferenced, which loses it a different way.
        if not (isinstance(fn, ast.Attribute) and fn.attr in {"shield", "_shielded"}):
            continue
        for arg in node.args:
            # asyncio.shield(self._release_ownership(...)) → grab the inner name
            if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute):
                shielded.add(arg.func.attr)

    for required in ("_release_ownership", "_finish_cancelled"):
        assert required in shielded, (
            f"{required} is called without asyncio.shield — a second cancel "
            f"abandons it again. See this module's docstring for what that costs."
        )
