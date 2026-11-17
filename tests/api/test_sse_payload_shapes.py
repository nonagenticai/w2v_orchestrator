"""SSE payload-shape robustness (src/api/sse.py).

``stream_channel`` names each SSE event from the published JSON's ``event`` key. It
used to do that INSIDE the ``try`` that guards ``json.loads``:

    parsed = json.loads(payload)
    yield {"event": parsed.get("event", "message"), ...}

A payload that is valid JSON but not an OBJECT (``123``, ``null``, ``"str"``) parses
fine and then raises ``AttributeError`` on ``.get`` — which escapes the generator and
kills that client's stream. Every current publisher (``RunDispatcher._publish``) sends
an object, so this is robustness rather than a live bug; these tests pin it so a future
publisher (or a stray hand-published message on ``runs:<id>``) cannot drop a stream.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import cast

from fastapi import Request

from src.api import sse


class _ScriptedPubSub:
    """Fake Redis PubSub that replays a fixed list of raw payloads, then goes quiet."""

    def __init__(self, payloads: list[str]) -> None:
        self.pending: deque[str] = deque(payloads)
        # Set only once an EMPTY poll has happened, so the final payload is still yielded
        # before the request stub reports a disconnect (both tasks resolve in the same
        # loop iteration, and the disconnect check wins).
        self.drained = False

    async def subscribe(self, channel: str) -> None:
        return None

    async def get_message(
        self, *, ignore_subscribe_messages: bool = True, timeout: float = 1.0
    ) -> dict | None:
        if not self.pending:
            self.drained = True
            return None
        return {"data": self.pending.popleft()}

    async def unsubscribe(self, channel: str) -> None:
        return None

    async def aclose(self) -> None:
        return None


class _DisconnectsWhenDrainedRequest:
    """Stays connected until the scripted pubsub has been polled dry, then disconnects
    so the stream terminates deterministically (no wall-clock waiting)."""

    def __init__(self, pubsub: _ScriptedPubSub) -> None:
        self._pubsub = pubsub
        self.app = SimpleNamespace(state=SimpleNamespace(redis=SimpleNamespace(pubsub=self._get)))

    def _get(self) -> _ScriptedPubSub:
        return self._pubsub

    async def is_disconnected(self) -> bool:
        return self._pubsub.drained


async def _collect(payloads: list[str]) -> list[dict[str, object]]:
    pubsub = _ScriptedPubSub(payloads)
    request = cast(Request, _DisconnectsWhenDrainedRequest(pubsub))
    response = await sse.stream_channel(request, channel="runs:test-payload-shapes")
    # ``body_iterator`` is typed as yielding ``object``; stream_channel only ever yields
    # the ``{"event": ..., "data": ...}`` dicts asserted on below.
    return [cast("dict[str, object]", ev) async for ev in response.body_iterator]


async def test_non_object_json_payloads_do_not_kill_the_stream() -> None:
    # Each of these is valid JSON but has no ``.get`` — the old code raised
    # AttributeError out of the generator and the client's stream died.
    events = await _collect(['{"event": "stage", "n": 1}', "123", "null", '"a string"', "[1, 2]"])
    assert [ev["event"] for ev in events] == ["stage", "message", "message", "message", "message"]
    # The raw payload is always forwarded verbatim, whatever its shape.
    assert [ev["data"] for ev in events][1:] == ["123", "null", '"a string"', "[1, 2]"]


async def test_invalid_json_and_missing_event_fall_back_to_message() -> None:
    events = await _collect(["not json at all", "{}", '{"event": null}'])
    assert [ev["event"] for ev in events] == ["message", "message", "message"]
    assert [ev["data"] for ev in events] == ["not json at all", "{}", '{"event": null}']
