"""Durable per-run execution claim — the DB-backed second guard behind the
Redis execution lease (``run_lease.RunLease``).

``RunLease.acquire`` falls *open* (returns ``None`` → execute anyway) on any
Redis error so a transient blip can't wedge every run. The cost is that a *full*
Redis outage lets all 2-5 replicas fail open on the same run at once and
double-run its non-idempotent submit stages (render / TTS / upload, ``retries=0``)
→ duplicate paid artifacts.

This claim closes that window. On the fail-open path the dispatcher takes an
atomic conditional claim on the run row; exactly one pod wins, the rest skip. A
heartbeat refreshes ``exec_claimed_at`` while the run executes, and
finalize/drain clears it. A claim older than ``_CLAIM_TTL`` is *stale* and
reclaimable, so a hard-killed pod's run resumes — the same liveness model as the
Redis lease, but durable and Redis-independent (it must survive a Redis outage).

The claim runs under the run's tenant GUC (``bind_run_tenant`` resolves the
tenant from the run_id via the SECURITY DEFINER bypass) so it is correct under
RLS enforce.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from uuid import UUID

from sqlalchemy import func, or_, select, update

from src.database.models import Run
from src.database.session import bind_run_tenant

log = logging.getLogger(__name__)

# Mirror the Redis lease's TTL/heartbeat so both guards expire on the same clock:
# a claim not refreshed within _CLAIM_TTL is reclaimable; the heartbeat renews it
# at a shorter interval while the run is in flight.
_CLAIM_TTL = 60  # seconds; a crashed pod's claim goes stale (reclaimable) after this
_CLAIM_HEARTBEAT = 30  # refresh interval (< TTL) while the run executes


def _stale_threshold():  # noqa: ANN202 -- SQLAlchemy expression, not a value
    """``now() - _CLAIM_TTL`` computed on the DB server (skew-free)."""
    return func.now() - func.make_interval(0, 0, 0, 0, 0, 0, _CLAIM_TTL)


class RunDbClaim:
    def __init__(self, session_factory, *, instance_id: str) -> None:  # noqa: ANN001
        self._sessions = session_factory
        self._instance_id = instance_id
        # Retain heartbeat tasks so they aren't GC'd mid-flight (RUF006).
        self._heartbeat_tasks: set[asyncio.Task] = set()

    async def acquire(self, run_id: UUID) -> bool:
        """Atomically claim the run for this instance.

        ``True`` — this instance now owns the durable claim (was unclaimed or the
        prior claim is stale). ``False`` — another instance holds a fresh claim;
        the caller must NOT execute.

        Uses ``UPDATE ... WHERE exec_claimed_at IS NULL OR stale RETURNING`` so
        two racing replicas serialize on the row lock: under READ COMMITTED the
        loser re-evaluates the predicate against the winner's committed row, no
        longer matches, and gets 0 rows back.
        """
        async with self._sessions() as session:
            await bind_run_tenant(session, run_id)
            claimed = (
                await session.execute(
                    update(Run)
                    .where(
                        Run.id == run_id,
                        or_(
                            Run.exec_claimed_at.is_(None), Run.exec_claimed_at < _stale_threshold()
                        ),
                    )
                    .values(exec_pod=self._instance_id, exec_claimed_at=func.now())
                    .returning(Run.id)
                )
            ).scalar_one_or_none()
            await session.commit()
            return claimed is not None

    async def fresh(self, run_id: UUID) -> bool:
        """True if the run currently carries a NON-stale execution claim (held by
        any instance). The orphan reconciler uses this — alongside the Redis
        lease — to tell a run still executing lease-less on some pod from a true
        orphan, so a recovered-Redis reconcile tick doesn't re-dispatch it."""
        async with self._sessions() as session:
            await bind_run_tenant(session, run_id)
            count = (
                await session.execute(
                    select(func.count())
                    .select_from(Run)
                    .where(
                        Run.id == run_id,
                        Run.exec_claimed_at.is_not(None),
                        Run.exec_claimed_at >= _stale_threshold(),
                    )
                )
            ).scalar_one()
            return count > 0

    async def release(self, run_id: UUID) -> None:
        """Drop our claim so the next pod can resume immediately — but only if we
        still own it, so a stale-reclaim by another pod is never clobbered."""
        async with self._sessions() as session:
            await bind_run_tenant(session, run_id)
            await session.execute(
                update(Run)
                .where(Run.id == run_id, Run.exec_pod == self._instance_id)
                .values(exec_pod=None, exec_claimed_at=None)
            )
            await session.commit()

    def start_heartbeat(
        self, run_id: UUID, *, on_lost: Callable[[], object] | None = None
    ) -> asyncio.Task:
        """Start (and retain) a task that refreshes ``exec_claimed_at`` while the
        run is in flight, so a long run's claim never goes stale under it.

        ``on_lost`` is invoked (once) the moment this instance can no longer prove
        it owns the claim — either the refresh matched no row (another pod
        stale-reclaimed us) or the refresh has been failing long enough that the
        claim MUST already be stale and therefore reclaimable. Losing the claim
        means a second replica is entitled to execute the same run, so stopping
        the heartbeat alone is not enough: the caller has to stop the WORK, or
        both replicas run the run's non-idempotent submit stages and the tenant
        is billed twice.
        """
        task = asyncio.create_task(self._heartbeat(run_id, on_lost), name=f"claim-hb:{run_id}")
        self._heartbeat_tasks.add(task)
        task.add_done_callback(self._heartbeat_tasks.discard)
        return task

    def _lost(self, run_id: UUID, on_lost: Callable[[], object] | None, reason: str) -> None:
        log.warning("run DB-claim LOST for %s (%s); halting execution on this pod", run_id, reason)
        if on_lost is None:
            return
        try:
            on_lost()
        except Exception as exc:  # noqa: BLE001 -- the halt callback must never kill the heartbeat
            log.warning("run DB-claim on_lost callback failed for %s (%s)", run_id, exc)

    async def _heartbeat(self, run_id: UUID, on_lost: Callable[[], object] | None = None) -> None:
        # A refresh that keeps erroring is indistinguishable, from the DB's point
        # of view, from a pod that died: after _CLAIM_TTL the claim is stale and
        # another replica may take it. The 2026-08-20 leaked `row_security=off`
        # incident is exactly this shape — every statement 500'd while the run
        # was live. Count consecutive failures and treat "we could not renew for
        # a whole TTL" as a lost claim rather than looping forever.
        max_failures = max(1, _CLAIM_TTL // _CLAIM_HEARTBEAT)
        failures = 0
        while True:
            await asyncio.sleep(_CLAIM_HEARTBEAT)
            try:
                async with self._sessions() as session:
                    await bind_run_tenant(session, run_id)
                    result = await session.execute(
                        update(Run)
                        .where(Run.id == run_id, Run.exec_pod == self._instance_id)
                        .values(exec_claimed_at=func.now())
                    )
                    await session.commit()
            except Exception as exc:  # noqa: BLE001 -- a single miss is covered by the TTL
                failures += 1
                log.warning(
                    "run DB-claim heartbeat failed for %s (%s) [%d/%d]",
                    run_id,
                    exc,
                    failures,
                    max_failures,
                )
                if failures >= max_failures:
                    self._lost(run_id, on_lost, "refresh failing for a full TTL")
                    return
                continue
            failures = 0
            # We lost the claim (stale-reclaimed by another pod) → stop the RUN.
            if result.rowcount == 0:
                self._lost(run_id, on_lost, "claim reclaimed by another instance")
                return
