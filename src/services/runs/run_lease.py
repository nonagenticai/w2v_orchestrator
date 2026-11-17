"""Per-run execution lease — a run executes on at most one pod at a time.

The reconciler (``src.lifecycle.reconciler._reconcile_orphan_runs``) re-dispatches runs left
``running`` by a drained pod; without this lease a pod that boots while a prior
pod is still draining would re-execute a still-live run, double-firing its
non-idempotent submit stages (render/TTS/upload).

Extracted from ``RunDispatcher`` so the lease state machine is testable in
isolation. The dispatcher holds one ``RunLease`` (its owner token ≈ this pod),
acquires before executing, renews via a heartbeat while in flight, and releases
on exit; the reconciler reads ``present`` to tell a live run from an orphan.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from uuid import UUID

from redis.asyncio import Redis
from redis.exceptions import RedisError

from src.cache.raw import raw_redis

log = logging.getLogger(__name__)

_LEASE_TTL = 60  # seconds; a crashed pod's lease auto-expires after this
_LEASE_HEARTBEAT = 30  # renew interval (< TTL) while the run is in flight


class RunLease:
    def __init__(self, redis: Redis, *, instance_id: str) -> None:
        self._redis = redis
        self._instance_id = instance_id
        # Retained so the lease-renewal tasks aren't GC'd mid-flight.
        self._heartbeat_tasks: set[asyncio.Task] = set()

    @staticmethod
    def key(run_id: UUID) -> str:
        return f"runexec:{run_id}"

    def raw_redis(self) -> Redis:
        """Return the underlying raw ``redis.asyncio`` client.

        The lease deliberately bypasses the w2v_common ``RedisClient`` wrapper and
        talks to the raw client. This is INTENTIONAL — do not "simplify" it onto the
        wrapper's public methods (``get`` / ``set`` / the new public ``set_nx``),
        which would reintroduce two correctness bugs:

        1. **Fail-closed presence.** The wrapper swallows ``RedisError`` and returns
           ``None`` from ``get``. ``present`` (below) relies on the raw ``get``
           *raising* on a Redis outage so its ``except`` can fall **closed** (assume
           the run is alive → skip). Routed through the wrapper, an outage would look
           like "key absent" → fall **open** → the reconciler re-dispatches the whole
           fleet at once, double-executing non-idempotent stages. This is the exact
           storm ``present``'s docstring warns about.
        2. **Un-serialized value compare.** The wrapper JSON-serializes on write and
           deserializes on read; the lease stores/reads the bare ``instance_id`` on the
           raw client so ``_owned``'s equality check is symmetric. Mixing a wrapper
           write (``"abc"`` → ``'"abc"'``) with a raw read (or vice-versa) breaks
           ownership and leaks the lease until TTL.

        So the atomic acquire here uses the raw client's native ``SET NX EX`` rather
        than the wrapper's public ``set_nx`` (whose tri-state matches ``acquire`` but
        whose serialization does not fit the lease). Production injects the wrapper
        subclass (drop to its raw client via ``_ensure_connected``); unit tests inject
        a raw ``FakeRedis`` directly (no wrapper), so use it as-is. Both cases
        are handled by :func:`src.cache.raw.raw_redis`, the single copy of that
        unwrapping.
        """
        return raw_redis(self._redis)

    async def acquire(self, run_id: UUID) -> bool | None:
        """Try to claim the per-run execution lease.

        ``True`` — this instance now owns the run. ``False`` — another live
        instance owns it; the caller must NOT execute. ``None`` — Redis is
        unavailable; fall open and execute without a lease rather than wedging
        every run on a transient Redis error.
        """
        raw = self.raw_redis()
        lease_key = self.key(run_id)
        try:
            result = await raw.set(lease_key, self._instance_id, nx=True, ex=_LEASE_TTL)
        except (RedisError, OSError, AttributeError) as exc:
            log.warning(
                "run lease acquire failed for %s (%s); proceeding without lease", lease_key, exc
            )
            return None
        return bool(result)

    async def _owned(self, raw: Redis, lease_key: str) -> bool:
        val = await raw.get(lease_key)
        if val is None:
            return False
        if isinstance(val, (bytes, bytearray)):
            val = val.decode()
        return val == self._instance_id

    async def present(self, run_id: UUID) -> bool:
        """True if *any* instance currently holds this run's execution lease.

        Distinct from ownership: the periodic reconciler uses this to tell a
        live run (some pod holds the lease and is renewing it) from an orphan
        (lease released on graceful drain, or TTL-expired after a hard kill).

        On a Redis error this falls **closed** (returns ``True`` → "assume
        alive, skip"). That is deliberate and the opposite of ``acquire``
        falling open: the reaper sweeps every running run each tick, so falling
        open here during a Redis outage would re-dispatch the whole fleet at
        once — and because ``acquire`` also falls open under the same outage,
        those re-dispatches would double-execute non-idempotent stages.
        """
        raw = self.raw_redis()
        try:
            return await raw.get(self.key(run_id)) is not None
        except (RedisError, OSError, AttributeError) as exc:
            log.warning(
                "lease presence check failed for %s (%s); assuming alive (skip)", run_id, exc
            )
            return True

    async def release(self, run_id: UUID) -> None:
        """Compare-and-delete: drop the lease only if we still own it, so a
        TTL-expiry-then-reacquire by another pod is never clobbered.

        (fakeredis has no ``eval``, so this is a GET-then-DEL rather than an
        atomic Lua CAS; the tiny race is harmless given the TTL/heartbeat
        margins — the lease is only un-owned here if our heartbeat already
        lapsed for a full TTL.)
        """
        raw = self.raw_redis()
        lease_key = self.key(run_id)
        try:
            if await self._owned(raw, lease_key):
                await raw.delete(lease_key)
        except (RedisError, OSError, AttributeError) as exc:
            # ``AttributeError`` matches the tuple `acquire` and `present` already
            # use: `_owned` reaches the same `raw.get`/`raw.delete` surface, and a
            # stub/rebound client raising it here escapes into the dispatcher's
            # `finally` and skips the DB-claim release that follows.
            log.warning("run lease release failed for %s (%s); will TTL-expire", lease_key, exc)

    def start_heartbeat(
        self, run_id: UUID, on_lost: Callable[[], object] | None = None
    ) -> asyncio.Task:
        """Start (and retain) a task that renews the lease TTL while in flight.

        ``on_lost`` is invoked (once) the moment this instance can no longer prove
        it owns the lease — either the key is gone / owned by someone else, or the
        renew has been failing long enough that the key MUST already have
        TTL-expired and is therefore reacquirable. Exactly the contract
        ``RunDbClaim.start_heartbeat`` already has, for exactly the same reason:
        losing the lease means another replica is entitled to execute this same
        run, so stopping the heartbeat alone is NOT enough. The caller has to stop
        the WORK, or both replicas run the run's non-idempotent submit stages
        (render / TTS / upload, all ``retries=0``) and the tenant is billed twice.

        Optional so the existing lease unit tests, which drive the state machine
        directly, keep working unchanged.
        """
        lease_key = self.key(run_id)
        task = asyncio.create_task(
            self._heartbeat(run_id, lease_key, on_lost), name=f"lease-hb:{run_id}"
        )
        self._heartbeat_tasks.add(task)
        task.add_done_callback(self._heartbeat_tasks.discard)
        return task

    def _lost(self, run_id: UUID, on_lost: Callable[[], object] | None, reason: str) -> None:
        log.warning("run lease LOST for %s (%s); halting execution on this pod", run_id, reason)
        if on_lost is None:
            return
        try:
            on_lost()
        except Exception as exc:  # noqa: BLE001 -- the halt callback must never kill the heartbeat
            log.warning("run lease on_lost callback failed for %s (%s)", run_id, exc)

    async def _heartbeat(
        self, run_id: UUID, lease_key: str, on_lost: Callable[[], object] | None = None
    ) -> None:
        """Renew the lease TTL while the run executes — but only while we still
        own it; if the lease lapsed and another pod reacquired, HALT the run.

        Both exits used to be silent. Returning on "not owned" left this pod
        executing a run a second pod had already legitimately reacquired, and the
        error branch looped forever without ever concluding anything — so a Redis
        outage longer than ``_LEASE_TTL`` (two failed renews) let the key expire
        under a still-running pod, the reconciler saw a leaseless orphan and
        re-dispatched it, and both pods then ran the same non-idempotent submit
        stages. Note the DB claim cannot backstop this: ``RunDispatcher._run``
        takes it ONLY when ``acquire`` fell open, so a run that started under a
        healthy lease has no claim at all.

        ``max_failures`` mirrors ``RunDbClaim._heartbeat``: a renew that has been
        failing for a whole TTL is indistinguishable, from Redis's point of view,
        from a pod that died — the key is gone and another replica may take it.
        """
        raw = self.raw_redis()
        max_failures = max(1, _LEASE_TTL // _LEASE_HEARTBEAT)
        failures = 0
        while True:
            await asyncio.sleep(_LEASE_HEARTBEAT)
            try:
                if not await self._owned(raw, lease_key):
                    self._lost(run_id, on_lost, "lease absent or reacquired by another instance")
                    return
                await raw.expire(lease_key, _LEASE_TTL)
            except (RedisError, OSError) as exc:
                failures += 1
                log.warning(
                    "run lease heartbeat failed for %s (%s) [%d/%d]",
                    lease_key,
                    exc,
                    failures,
                    max_failures,
                )
                if failures >= max_failures:
                    self._lost(run_id, on_lost, "renew failing for a full TTL")
                    return
                continue
            failures = 0
