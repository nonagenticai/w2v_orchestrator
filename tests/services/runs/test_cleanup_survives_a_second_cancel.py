"""Cleanup that is itself cancellable never runs, and leaks what it was closing.

Both sites here run their teardown in a `finally:` reached on the way OUT of a
cancellation — a stage timeout (`middleware/timeout.py` wraps every handler in
`wait_for`) or an explicit `task.cancel()` from `dag_executor`'s batch abort. A
SECOND cancel delivered while the first unwinds lands on the bare `await` in the
`finally` and propagates straight out of it, so the release never happens.

`except asyncio.CancelledError` is not the fix: it is a `BaseException`, so the
existing `except (RedisError, ConnectionError, OSError)` cannot catch it, and by
the time it is raised the close has already been interrupted. `asyncio.shield` is.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.services.runs import sub_dag


class _RecordingPubSub:
    """Records what actually completed, and yields control on each await so a
    cancellation delivered mid-teardown lands exactly where a real one would."""

    def __init__(self) -> None:
        self.unsubscribed = False
        self.closed = False

    async def unsubscribe(self, *_channels: str) -> None:
        await asyncio.sleep(0)
        self.unsubscribed = True

    async def aclose(self) -> None:
        await asyncio.sleep(0)
        self.closed = True


@pytest.mark.asyncio
async def test_await_runs_releases_the_pubsub_when_a_second_cancel_lands_mid_teardown() -> None:
    """The production site, with the cancellation shape that actually breaks it.

    ``await_runs`` is reached from ``course/compile_lessons.py`` inside a stage
    handler, which ``middleware/timeout.py`` runs under ``wait_for`` and which
    ``dag_executor``'s batch abort may additionally ``task.cancel()``. ONE cancel is
    survivable unshielded — it is delivered at the loop's await and the `finally`
    then runs with no cancellation pending. It is the SECOND cancel, arriving while
    the teardown is in flight, that lands on the bare await and abandons the close.
    So this test delivers exactly that, from inside the teardown itself.
    """
    import contextlib
    from types import SimpleNamespace
    from uuid import uuid4

    from src.services.runs.sub_dag import await_runs

    class _CancelDuringTeardown(_RecordingPubSub):
        """Fires the second cancel at the moment the release begins."""

        def __init__(self) -> None:
            super().__init__()
            self.task: asyncio.Task[None] | None = None

        async def subscribe(self, *_channels: str) -> None:
            return None

        async def get_message(self, *, ignore_subscribe_messages: bool, timeout: float) -> None:
            await asyncio.sleep(60)  # blocks until the first cancel
            return

        async def unsubscribe(self, *channels: str) -> None:
            if self.task is not None:
                self.task.cancel()  # the second cancel, mid-teardown
            await super().unsubscribe(*channels)

    pubsub = _CancelDuringTeardown()
    redis = SimpleNamespace(pubsub=lambda: pubsub)

    @contextlib.asynccontextmanager
    async def _session():
        yield object()

    rid = uuid4()

    async def _drive() -> None:
        with patch.object(sub_dag, "_seed_initial_status", new=AsyncMock(return_value=None)):
            await await_runs(
                run_ids=[rid],
                session_factory=_session,
                redis=redis,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                poll_interval_s=0.01,
                timeout_s=60,
            )

    task = asyncio.create_task(_drive())
    pubsub.task = task
    await asyncio.sleep(0.01)
    task.cancel()  # the first cancel — the stage timeout
    with contextlib.suppress(asyncio.CancelledError):
        await task

    for _ in range(5):  # let the shielded release finish after the awaiter is gone
        await asyncio.sleep(0)
    assert pubsub.unsubscribed, "the second cancel abandoned the unsubscribe"
    assert pubsub.closed, "the second cancel leaked the pubsub connection"
