"""Unit tests for ``_reconcile_orphan_runs`` in src/asgi.py.

The reconciler is the lifespan-time entry point that re-dispatches runs
the previous pod left in ``running``/``pending`` after a drain (see
RunDispatcher._draining). These tests pin the four shapes the function
has to handle correctly:

1. No orphans → no-op (no dispatch, no mark_failed).
2. Recent orphan → dispatch called with the run id.
3. Stale orphan (updated_at older than max_age_seconds) → row marked
   ``failed`` with a descriptive error, NOT re-dispatched.
4. ``pg_try_advisory_xact_lock`` returns False (another replica holds
   it) → reconciler returns immediately, no dispatch, no update.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.asyncio


def _session_factory(session):
    @asynccontextmanager
    async def _factory():
        yield session

    return _factory


def _begin_nested():
    """``AsyncSession.begin_nested()`` is a SYNC call returning an async context
    manager (the SAVEPOINT); an ``AsyncMock`` attribute would return a coroutine."""

    @asynccontextmanager
    async def _savepoint():
        yield None

    return _savepoint()


def _execute_results(*results):
    """Build an AsyncMock for ``session.execute`` whose successive calls
    return ``results[0]``, ``results[1]``, ..."""
    return AsyncMock(side_effect=list(results))


def _scalar_result(value):
    """Wrap a scalar in the result-object shape ``.scalar()`` expects."""
    r = MagicMock()
    r.scalar = MagicMock(return_value=value)
    return r


def _rows_result(rows):
    """Wrap a list of rows in the result-object shape ``.all()`` expects."""
    r = MagicMock()
    r.all = MagicMock(return_value=rows)
    return r


def _update_result():
    """Placeholder result for a fire-and-forget UPDATE."""
    return MagicMock()


async def test_periodic_reconciler_ticks_and_survives_a_failed_tick(monkeypatch) -> None:
    """The periodic loop re-runs the reconciler each interval and a single
    raising tick (transient DB/Redis error) must NOT kill the loop — it logs
    and keeps sweeping. Cancellation tears it down cleanly."""
    from src.lifecycle import reconciler

    calls: list[int] = []

    async def fake_reconcile(
        _sf: object, _disp: object, _max_age: int, *, redis: object = None
    ) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient blip on the first tick")

    monkeypatch.setattr(reconciler, "_reconcile_orphan_runs", fake_reconcile)

    real_sleep = asyncio.sleep

    async def instant_sleep(_seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(reconciler.asyncio, "sleep", instant_sleep)

    task = asyncio.create_task(
        reconciler._run_periodic_reconciler(
            cast("Any", None), cast("Any", None), max_age_seconds=7200, interval_seconds=90
        )
    )
    # Yield enough times for several ticks to accumulate.
    for _ in range(20):
        await real_sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Survived the first-tick exception and kept ticking.
    assert len(calls) >= 2


def _fakeredis():
    from fakeredis.aioredis import FakeRedis

    return FakeRedis()
