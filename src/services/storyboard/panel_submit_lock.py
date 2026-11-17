"""Estate-wide mutual exclusion for storyboard panel-image submissions.

The image runner behind ACG's local provider is synchronous and single-threaded,
and it already carries the live ``ai_image_gen`` run stage. A whole-board panel
generate must therefore not fan out — one panel submission at a time, or board
previews head-of-line block real runs.

**A per-process ``asyncio.Semaphore`` cannot deliver that.** The orchestrator
runs ``replicas: 2`` with an HPA up to 5, so an in-pod gate bounds concurrency at
the replica count, not at one. This lock is the cross-replica half: a single
Redis key (there is exactly one image runner estate-wide, so one global key is
the right granularity — this is NOT per-tenant).

Shape copied from :class:`src.services.runs.run_lease.RunLease`, which solves the
same problem for run execution: ``SET NX EX`` to acquire, a heartbeat to renew
while held, compare-and-delete to release. The TTL is what makes a hard-killed
pod's lock recoverable — without it, one ``SIGKILL`` mid-generation would wedge
panel generation estate-wide until someone deleted the key by hand.

Two deliberate failure postures:

* **Redis unreachable → fall OPEN** (proceed with in-pod serialisation only, and
  log it). Falling closed would take the whole feature down on a Redis blip, and
  nothing is double-*billed* by proceeding — the idempotency key owns
  correctness; this lock owns load.
* **Waited past ``max_wait`` → fail LOUDLY** (:class:`PanelSubmitBusyError`). A queue
  that silently barges is not a queue.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import suppress
from types import TracebackType
from typing import TYPE_CHECKING

from redis.exceptions import RedisError

from src.cache.raw import raw_redis

if TYPE_CHECKING:
    from redis.asyncio import Redis

log = logging.getLogger(__name__)

#: The one key. One image runner estate-wide ⇒ one lock, not one per tenant.
SUBMIT_LOCK_KEY = "storyboard:panel-image:submit"

_TTL_SECONDS = 90  # a hard-killed holder's lock auto-expires after this
_HEARTBEAT_SECONDS = 30  # renew interval, comfortably < TTL
_POLL_SECONDS = 0.25  # how often a waiter retries
#: A whole board queues behind this lock by design, and one generation is
#: minutes — so the ceiling is generous. It exists to bound a pathological wait,
#: not to bound a normal one.
_MAX_WAIT_SECONDS = 1800.0


class PanelSubmitBusyError(Exception):
    """Waited past ``max_wait`` for the estate-wide submission slot."""


class PanelSubmitLock:
    """Async context manager holding the estate-wide panel-submission slot."""

    def __init__(
        self,
        redis: Redis,
        *,
        ttl: int = _TTL_SECONDS,
        heartbeat: float = _HEARTBEAT_SECONDS,
        poll: float = _POLL_SECONDS,
        max_wait: float = _MAX_WAIT_SECONDS,
    ) -> None:
        self._redis = redis
        self._ttl = ttl
        self._heartbeat_s = heartbeat
        self._poll = poll
        self._max_wait = max_wait
        self._token = ""
        self._heartbeat: asyncio.Task[None] | None = None
        #: True when Redis was unreachable and we proceeded ungated.
        self.degraded = False

    def _raw(self) -> Redis:
        """The raw ``redis.asyncio`` client under the w2v_common wrapper.

        Same reason as ``RunLease.raw_redis``: the wrapper JSON-serialises
        values, which would break the symmetric token compare in ``release``,
        and it swallows ``RedisError`` where we need it to raise. The
        unwrapping itself lives in :func:`src.cache.raw.raw_redis` — one copy
        for the whole service.
        """
        return raw_redis(self._redis)

    async def __aenter__(self) -> PanelSubmitLock:
        self._token = secrets.token_hex(8)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._max_wait
        raw = self._raw()
        while True:
            try:
                acquired = await raw.set(SUBMIT_LOCK_KEY, self._token, nx=True, ex=self._ttl)
            except (RedisError, OSError, AttributeError) as exc:
                log.warning(
                    "panel submit lock unavailable (%s); proceeding with per-pod "
                    "serialisation only — concurrent submissions are possible across replicas",
                    exc,
                )
                self.degraded = True
                return self
            if acquired:
                self._heartbeat = asyncio.create_task(self._renew(), name="panel-submit-lock-hb")
                return self
            if loop.time() >= deadline:
                raise PanelSubmitBusyError(
                    f"waited {self._max_wait:.0f}s for the image runner; it is still busy"
                )
            await asyncio.sleep(self._poll)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat
            self._heartbeat = None
        if self.degraded or not self._token:
            return
        raw = self._raw()
        try:
            # Compare-and-delete so a TTL-lapse-then-reacquire by another pod is
            # never clobbered. (fakeredis has no ``eval``, so this is GET-then-DEL
            # like RunLease.release; the window is only reachable if our heartbeat
            # already lapsed for a full TTL.)
            current = await raw.get(SUBMIT_LOCK_KEY)
            if isinstance(current, (bytes, bytearray)):
                current = current.decode()
            if current == self._token:
                await raw.delete(SUBMIT_LOCK_KEY)
        except (RedisError, OSError) as exc_release:
            log.warning(
                "panel submit lock release failed (%s); it will TTL-expire in %ss",
                exc_release,
                self._ttl,
            )

    async def _renew(self) -> None:
        """Extend the TTL while we hold it; stop if we no longer own the key."""
        raw = self._raw()
        while True:
            await asyncio.sleep(self._heartbeat_s)
            try:
                current = await raw.get(SUBMIT_LOCK_KEY)
                if isinstance(current, (bytes, bytearray)):
                    current = current.decode()
                if current != self._token:
                    return
                await raw.expire(SUBMIT_LOCK_KEY, self._ttl)
            except (RedisError, OSError) as exc:
                log.warning("panel submit lock heartbeat failed: %s", exc)
