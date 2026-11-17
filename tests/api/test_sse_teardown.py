"""SSE teardown ordering (src/api/sse.py).

When a client disconnects mid-stream, ``stream_channel``'s finally block cancels
the in-flight ``pubsub.get_message`` task. A bare ``.cancel()`` only *schedules*
cancellation — the task is still pending on the SAME Redis PubSub connection
when ``unsubscribe``/``aclose`` run, producing "Task was destroyed but it is
pending" plus a dirty connection. This test pins the fix: the cancelled tasks
must be AWAITED (cancellation actually delivered) before the pubsub connection
is unsubscribed and closed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast

from fastapi import Request

from src.api import sse


class _RecordingPubSub:
    """Fake Redis PubSub that records teardown ordering.

    ``get_message`` blocks until cancelled and records the moment the
    CancelledError is actually DELIVERED — which only happens when the
    cancelled task is awaited, not when ``.cancel()`` is merely called.
    ``unsubscribe``/``aclose`` record synchronously (no yield point), matching
    the real client where teardown proceeds without ceding the event loop.
    """

    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def subscribe(self, channel: str) -> None:
        self.events.append("subscribe")

    async def get_message(
        self, *, ignore_subscribe_messages: bool = True, timeout: float = 1.0
    ) -> dict | None:
        try:
            await asyncio.Event().wait()  # block until cancelled
        except asyncio.CancelledError:
            self.events.append("get_message_cancelled")
            raise
        return None

    async def unsubscribe(self, channel: str) -> None:
        self.events.append("unsubscribe")

    async def aclose(self) -> None:
        self.events.append("aclose")


class _FakeRedis:
    def __init__(self, pubsub: _RecordingPubSub) -> None:
        self._pubsub = pubsub

    def pubsub(self) -> _RecordingPubSub:
        return self._pubsub


class _ImmediatelyDisconnectedRequest:
    """Request stub whose client disconnects on the first check, forcing the
    stream to break out of its race loop while ``get_message`` is still
    pending — the exact teardown race being pinned."""

    def __init__(self, redis: _FakeRedis) -> None:
        self.app = SimpleNamespace(state=SimpleNamespace(redis=redis))

    async def is_disconnected(self) -> bool:
        return True


class _RaisingUnsubscribePubSub(_RecordingPubSub):
    """A dropped-socket ``unsubscribe`` raises ConnectionError (a RedisError), the
    common case when the stream ends because the client's connection died."""

    async def unsubscribe(self, channel: str) -> None:
        from redis.exceptions import ConnectionError as RedisConnectionError

        self.events.append("unsubscribe_attempt")
        raise RedisConnectionError("Connection closed by server")


async def test_teardown_closes_pubsub_even_when_unsubscribe_fails() -> None:
    # A ConnectionError on unsubscribe (dead socket — the usual disconnect case)
    # must NOT skip aclose: the dedicated pubsub connection must still be returned
    # to the pool, and nothing may raise out of the finally. Guarded-beside-
    # unguarded: the sibling _listen_control already suppresses RedisError here.
    events: list[str] = []
    pubsub = _RaisingUnsubscribePubSub(events)
    request = cast(Request, _ImmediatelyDisconnectedRequest(_FakeRedis(pubsub)))

    response = await sse.stream_channel(request, channel="runs:test-unsub-fail")
    async for _ev in response.body_iterator:  # must NOT raise
        pass
    for _ in range(3):
        await asyncio.sleep(0)

    assert "unsubscribe_attempt" in events
    assert "aclose" in events  # the leak fix: aclose runs even though unsubscribe raised


async def test_teardown_awaits_cancelled_get_message_before_closing_pubsub() -> None:
    events: list[str] = []
    pubsub = _RecordingPubSub(events)
    request = cast(Request, _ImmediatelyDisconnectedRequest(_FakeRedis(pubsub)))

    response = await sse.stream_channel(request, channel="runs:test-teardown")

    # Drain the stream: the immediate disconnect breaks the loop and runs the
    # generator's finally block inline before iteration completes.
    async for _ev in response.body_iterator:
        pass

    # Give the event loop a couple of ticks so that, on broken code, the
    # still-pending cancelled get_message task gets its CancelledError
    # delivered *now* — i.e. AFTER aclose — and the ordering assert below
    # fails deterministically instead of the marker being absent.
    for _ in range(3):
        await asyncio.sleep(0)

    assert "get_message_cancelled" in events
    assert "unsubscribe" in events
    assert "aclose" in events
    # The cancelled get_message task must be fully retired BEFORE the pubsub
    # connection is unsubscribed/closed.
    assert events.index("get_message_cancelled") < events.index("unsubscribe")
    assert events.index("get_message_cancelled") < events.index("aclose")
