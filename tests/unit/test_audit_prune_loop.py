"""``_run_periodic_audit_prune`` must prune BEFORE it sleeps.

Regression guard: the loop used to ``await asyncio.sleep(interval_seconds)`` at
the top of its body. ``interval_seconds`` defaults to 24h and the orchestrator
pod rolls far more often than that, so the retention prune never completed a
single pass in production -- leaving ``request_audit`` (a row per mutation and
per auth failure) as the unbounded disk-fill vector the loop exists to bound.

Each test drives the real loop and asserts on an ordered event log of what
actually executed, not on how many times a mock was called.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
from unittest.mock import AsyncMock, MagicMock, patch

from src.lifecycle.audit_prune import _run_periodic_audit_prune

_INTERVAL = 86400
_RETENTION_DAYS = 30


async def _drive(prune_outcome: object = 7) -> list[str]:
    """Run one pass of the loop, recording what it executed, in order.

    The patched sleep raises ``CancelledError`` so the loop stops at its first
    sleep -- meaning anything recorded before ``sleep:`` genuinely ran ahead of
    it, which is the property under test.
    """
    events: list[str] = []

    async def fake_prune(*, cutoff: dt.datetime) -> int:
        events.append(f"prune:{cutoff.isoformat()}")
        if isinstance(prune_outcome, BaseException):
            raise prune_outcome
        assert isinstance(prune_outcome, int)
        return prune_outcome

    async def fake_sleep(seconds: float) -> None:
        events.append(f"sleep:{seconds}")
        raise asyncio.CancelledError

    session = AsyncMock()
    session.commit = AsyncMock(side_effect=lambda: events.append("commit"))
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    factory = MagicMock(return_value=ctx)

    repo = MagicMock()
    repo.return_value.prune_older_than = fake_prune

    with (
        patch(
            "src.database.repositories.request_audit_repository.RequestAuditRepository",
            repo,
        ),
        patch("src.lifecycle.audit_prune.asyncio.sleep", fake_sleep),
        contextlib.suppress(asyncio.CancelledError),
    ):
        await _run_periodic_audit_prune(
            factory, retention_days=_RETENTION_DAYS, interval_seconds=_INTERVAL
        )
    return events


async def test_prunes_before_the_first_sleep() -> None:
    events = await _drive()

    assert events[0].startswith("prune:"), f"first action was {events[0]!r}, expected a prune"
    assert f"sleep:{_INTERVAL}" in events
    assert events.index("commit") < events.index(f"sleep:{_INTERVAL}")


async def test_cutoff_reflects_retention_days() -> None:
    events = await _drive(prune_outcome=0)

    cutoff = dt.datetime.fromisoformat(events[0].removeprefix("prune:"))
    expected = dt.datetime.now(dt.UTC) - dt.timedelta(days=_RETENTION_DAYS)
    assert abs((cutoff - expected).total_seconds()) < 60


async def test_a_failing_tick_still_reaches_the_sleep() -> None:
    """A raising pass must be swallowed and still wait out the interval.

    If the exception escaped, the loop would die; if it skipped the sleep, the
    loop would spin hot against a failing database.
    """
    events = await _drive(prune_outcome=RuntimeError("boom"))

    assert events[0].startswith("prune:")
    assert events[-1] == f"sleep:{_INTERVAL}"
    assert "commit" not in events
