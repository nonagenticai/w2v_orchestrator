"""Per-resource Server-Sent Events. Subscribes to a Redis channel and forwards
every JSON message as one SSE event."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator, Iterable
from contextlib import suppress

from fastapi import Request
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sse_starlette.sse import EventSourceResponse

log = logging.getLogger(__name__)

# Hard cap on a single SSE connection's lifetime. This is now hygiene, not the
# pool's safety net: the SSE routes build their catch-up snapshot on a SHORT-LIVED
# session and are mounted WITHOUT the tenant-gate dependencies (require_known_tenant
# / apply_tenant_rls), so they no longer pin a request-scoped DB session for the
# stream's lifetime. (Previously every router on the tenant-scoped routers pulled a
# ``SessionDep`` held for the *whole response* — i.e. the entire stream — so an
# abnormally-killed client whose ``request.is_disconnected()`` never fires pinned a
# pool connection until the pod restarted; that is exactly how the pool exhausted in
# prod.) Bounding the stream still forces a periodic reconnect (fresh snapshot, drop
# stale Redis subscriptions). Runs complete in well under this window, so an active
# run is never interrupted. Tunable via env for safety.
_MAX_STREAM_SECONDS = float(os.getenv("SSE_MAX_STREAM_SECONDS", "900"))


async def stream_channel(  # noqa: C901 — generator + race loop; further splitting hurts clarity
    request: Request,
    *,
    channel: str,
    ping_interval: float = 15.0,
    catchup: Iterable[dict] | None = None,
) -> EventSourceResponse:
    redis: Redis = request.app.state.redis
    pubsub = redis.pubsub()
    await pubsub.subscribe(channel)

    # Catch-up snapshot. Redis pub/sub keeps no backlog, so a client that
    # connects after early events were published (a fast run, or an SSE
    # reconnect mid-run) would otherwise never learn the current state and
    # would hang waiting for events that already fired. We subscribe FIRST
    # (above) so nothing published while the route built the snapshot is lost —
    # at worst a stage event is replayed in both the snapshot and the live
    # stream, which is an idempotent state update on the consumer.
    #
    # The snapshot is materialised by the CALLER on a short-lived DB session that
    # is already closed by the time we get here — the SSE route must NOT hold a
    # request-scoped session for the stream's lifetime (that pinned a pool
    # connection for the whole ~stream and is how the pool exhausted in prod).
    catchup_events: list[dict] = list(catchup or [])

    async def gen() -> AsyncGenerator[dict]:  # noqa: C901
        # Race ``pubsub.get_message`` against ``request.is_disconnected`` so a
        # client TCP-close is observed within the same event-loop tick instead
        # of waiting for the next 1s ``get_message`` timeout to fire. Without
        # this, every disconnected client pins a Redis subscription slot for
        # ~1 s after the browser tab is closed.
        pubsub_task: asyncio.Task | None = None
        disconnect_task: asyncio.Task | None = None
        try:
            for ev in catchup_events:
                yield {"event": ev.get("event", "message"), "data": json.dumps(ev)}
            loop = asyncio.get_running_loop()
            last_ping = loop.time()
            deadline = loop.time() + _MAX_STREAM_SECONDS
            while True:
                # Bounded lifetime: return so the response completes and the
                # browser's EventSource transparently reconnects for a fresh
                # snapshot. With the session no longer pinned to the stream this
                # is belt-and-suspenders (Redis-subscription / idle-reconnect
                # hygiene) rather than the pool's last line of defence. The
                # ``while`` re-checks at least every ~1s (the get_message timeout),
                # so the cap is honoured promptly even with no live traffic.
                if loop.time() >= deadline:
                    log.info(
                        "SSE channel %s reached max lifetime (%.0fs); closing for client reconnect",
                        channel,
                        _MAX_STREAM_SECONDS,
                    )
                    break
                if pubsub_task is None or pubsub_task.done():
                    pubsub_task = asyncio.create_task(
                        pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    )
                if disconnect_task is None or disconnect_task.done():
                    disconnect_task = asyncio.create_task(request.is_disconnected())
                done, _pending = await asyncio.wait(
                    {pubsub_task, disconnect_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disconnect_task in done and disconnect_task.result():
                    break
                if pubsub_task not in done:
                    # Disconnect fired False; loop and re-await disconnect.
                    continue
                msg = pubsub_task.result()
                if msg is None:
                    if asyncio.get_running_loop().time() - last_ping > ping_interval:
                        last_ping = asyncio.get_running_loop().time()
                        yield {"event": "ping", "data": "{}"}
                    continue
                payload = msg["data"]
                if isinstance(payload, bytes):
                    payload = payload.decode("utf-8")
                # Only a JSON OBJECT carries an event name. A payload that is valid JSON
                # but a scalar/array (``123``, ``null``, ``"str"``) parses fine and then
                # blows up on ``.get`` with an AttributeError — which escapes the generator
                # and kills THIS client's stream. Every current publisher sends an object,
                # so this is robustness, not a live bug: keep ``.get`` outside the try and
                # fall back to a plain ``message`` event for any non-object payload.
                try:
                    parsed: object = json.loads(payload)
                except json.JSONDecodeError:
                    parsed = None
                event = parsed.get("event") if isinstance(parsed, dict) else None
                yield {"event": str(event) if event else "message", "data": payload}
        finally:
            for t in (pubsub_task, disconnect_task):
                if t is not None and not t.done():
                    t.cancel()
            # Await the cancelled tasks BEFORE touching the pubsub connection:
            # a cancelled-but-still-pending ``get_message`` may hold the same
            # Redis connection, and racing unsubscribe/aclose against it leaves
            # a dirty connection plus "Task was destroyed but it is pending".
            # ``return_exceptions=True`` swallows the CancelledError results so
            # nothing can raise out of this finally block.
            await asyncio.gather(
                *(t for t in (pubsub_task, disconnect_task) if t is not None),
                return_exceptions=True,
            )
            # The stream almost always ends because the client connection dropped, so
            # unsubscribe writes to a dead socket and raises ConnectionError (a
            # RedisError). SEPARATE suppress blocks so a failing unsubscribe never
            # skips aclose — aclose MUST run to return the dedicated pubsub connection
            # to the pool (else this reproduces the documented pool exhaustion). The
            # sibling _listen_control guards the same pair; sse must too.
            with suppress(RedisError, OSError):
                await pubsub.unsubscribe(channel)
            with suppress(RedisError, OSError):
                await pubsub.aclose()

    return EventSourceResponse(gen())
