"""Orphan-run reconciler — the lifespan's durable-resume machinery.

Carved verbatim out of ``src.asgi`` (behaviour-preserving) so the lifespan
module stays focused on wiring. These functions re-dispatch runs a prior pod
left in ``running``/``pending`` after a drain (see ``RunDispatcher._draining``),
honouring the per-run execution lease AND the durable DB claim (PR #414) so a
recovered-Redis tick never re-dispatches a run another pod still owns.

``src.asgi.lifespan`` imports ``_reconcile_orphan_runs`` (startup one-shot) and
``_run_periodic_reconciler`` (the timer loop) from here and calls them exactly
as before.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from contextlib import suppress
from typing import TYPE_CHECKING

from redis.exceptions import RedisError

from src.observability.metrics import db_errors_total, runs_total

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy import Row
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.cache.redis_client import RedisClient
    from src.services.run_dispatcher import RunDispatcher

# Stable, w2v_orchestrator-private namespace for the reconciler's brief
# inter-replica mutex. Any int fits; this one is just a recognisable
# constant so an operator hitting it in pg_locks knows what owns it.
_RECONCILER_LOCK_KEY = -7234567891

# How many times the reconciler will re-dispatch the SAME still-running run
# before giving up and failing it. The `updated_at` max-age backstop cannot
# bound a re-dispatch loop on its own: every re-dispatch calls `mark_started`,
# which refreshes `updated_at`, so a run that keeps wedging never ages out.
# The count lives in Redis (`reconcile:attempts:<run_id>`), SHARED across
# replicas — the orchestrator runs 2-5 pods and the reconcile advisory lock
# alternates which pod does the work, so a pod-local counter would never
# accumulate. A genuinely wedged run is failed after this many re-dispatches.
_RECONCILER_MAX_REDISPATCH = 5
# TTL on the counter key. Run ids are unique (uuid7), so a key left behind by a
# run that later succeeded is harmless; the TTL just keeps Redis from
# accumulating dead keys. Comfortably longer than any realistic stuck-loop
# window at the reconcile interval.
_RECONCILE_ATTEMPT_TTL_SECONDS = 3600


async def _note_redispatch(redis: RedisClient, run_id: UUID) -> int:
    """Increment and return this run's consecutive re-dispatch count (shared via
    Redis). Best-effort: a Redis blip returns 0 so the reconciler still resumes
    rather than failing a run on the counter's own error."""
    key = f"reconcile:attempts:{run_id}"
    try:
        n = int(await redis.incr(key))
        if n == 1:
            await redis.expire(key, _RECONCILE_ATTEMPT_TTL_SECONDS)
        return n
    except (RedisError, OSError) as exc:
        logging.getLogger(__name__).warning("reconciler: attempt-count incr failed: %s", exc)
        return 0


async def _clear_redispatch(redis: RedisClient, run_id: UUID) -> None:
    with suppress(RedisError, OSError):
        await redis.delete(f"reconcile:attempts:{run_id}")


async def _reconcile_orphan_runs(
    session_factory: async_sessionmaker[AsyncSession],
    dispatcher: RunDispatcher,
    max_age_seconds: int,
    *,
    redis: RedisClient | None = None,
) -> None:
    """Re-dispatch runs left in ``running``/``pending`` by a prior pod.

    Companion to ``RunDispatcher._draining``: when a pod drains, the
    CancelledError handler leaves in-flight runs in ``running`` instead
    of marking them ``cancelled``, so the next pod's startup can resume
    them from the last completed stage (the dispatcher's ``_run``
    already rehydrates ``prior_run_outputs`` from ``stage_outputs``).

    ``pg_try_advisory_xact_lock`` is a brief, transaction-scoped mutex
    so only ONE of the 2 replicas does the reconcile work on a
    simultaneous-startup race — the other finds no orphans (the first
    has already dispatched them) and no-ops.

    Cross-pod double-execution is prevented one level down by
    ``RunDispatcher``'s per-run execution lease (Redis ``runexec:<id>``):
    if this reconciler re-dispatches a run that is still live on a
    draining pod, that pod still owns the lease, so the redispatched
    ``_run`` no-ops instead of double-firing the run's non-idempotent
    submit stages. A genuinely orphaned run (drained pod released its
    lease, or crashed pod's lease TTL-expired) is acquirable and
    resumes from its last completed stage.

    Runs whose ``updated_at`` is older than ``max_age_seconds`` are
    treated as stale (pre-existing orphans from before durable resume,
    or the user has long since moved on) and marked ``failed`` with a
    descriptive error rather than re-dispatched forever.
    """
    from sqlalchemy import text

    log = logging.getLogger(__name__)
    async with session_factory() as session:
        locked = (
            await session.execute(
                text("SELECT pg_try_advisory_xact_lock(:k)"),
                {"k": _RECONCILER_LOCK_KEY},
            )
        ).scalar()
        if not locked:
            log.info("reconciler: another replica holds the lock; skipping")
            return

        # Cross-tenant scan via the SECURITY DEFINER ``orphan_runs()`` resolver:
        # the reconciler spans ALL tenants but, under enforce, runs as the
        # restricted no-BYPASSRLS role, so a plain ``SELECT FROM runs`` would be
        # RLS-scoped to a single (here unset) tenant → 0 rows. The function (owned
        # by the migration superuser → bypasses FORCE) returns every orphan plus
        # its tenant, which ``_resolve_orphan_run`` then binds before each UPDATE.
        # Owner-bypass keeps it returning all rows when RLS is off, too.
        rows = (
            await session.execute(
                text(
                    "SELECT run_id AS id, tenant_id, status, updated_at "
                    "FROM orphan_runs() ORDER BY updated_at"
                )
            )
        ).all()
        if not rows:
            log.info("reconciler: no orphan runs to resume")
            await session.commit()
            return

        now = datetime.datetime.now(datetime.UTC)
        cutoff = datetime.timedelta(seconds=max_age_seconds)
        tally = {"finalized": 0, "resumed": 0, "stale": 0, "alive": 0, "errored": 0}
        for row in rows:
            # Per-row isolation. Every row shares ONE transaction (the one holding
            # the advisory xact lock), so before this a single orphan raising —
            # a transient error in its reap, an RLS surprise on its UPDATE —
            # propagated out of the loop, skipped every later orphan for this
            # tick, and left the shared transaction aborted (every later
            # statement fails ``InFailedSQLTransaction``). A SAVEPOINT per row
            # means a failure rolls back only that row's writes; the lock and the
            # other rows' work survive and commit.
            try:
                async with session.begin_nested():
                    outcome = await _resolve_orphan_run(
                        session,
                        dispatcher,
                        redis,
                        row,
                        now=now,
                        cutoff=cutoff,
                        max_age_seconds=max_age_seconds,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                db_errors_total.labels(op="reconciler_row").inc()
                log.exception("reconciler: orphan run %s failed to resolve; continuing", row.id)
                outcome = "errored"
            tally[outcome] += 1

        await session.commit()
        log.info(
            "reconciler: finalized=%d resumed=%d stale=%d alive=%d errored=%d total=%d",
            tally["finalized"],
            tally["resumed"],
            tally["stale"],
            tally["alive"],
            tally["errored"],
            len(rows),
        )


async def _resolve_orphan_run(
    session: AsyncSession,
    dispatcher: RunDispatcher,
    redis: RedisClient | None,
    row: Row,
    *,
    now: datetime.datetime,
    cutoff: datetime.timedelta,
    max_age_seconds: int,
) -> str:
    """Decide one orphan row's fate.

    Returns 'finalized' | 'resumed' | 'stale' | 'alive'.
    """
    from sqlalchemy import update

    from src.database.models import Run
    from src.database.repositories.run_repository import RunRepository
    from src.database.session import bind_session_tenant
    from src.models.failure_class import UNKNOWN

    log = logging.getLogger(__name__)

    # Backstop a stranded-but-finished run FIRST, before the lease check. If the
    # DAG is already fully terminal but the run row is still 'running' (owning
    # dispatch task interrupted after the last stage persisted terminal but
    # before the completion gate committed), the run has no pending work and its
    # lease — if any — is meaningless. Finalize it now instead of waiting out the
    # lease TTL or the max-age backstop (up to 2h): this is what wedges the
    # e2e-post-promote promo poll at a stale 'running'. finalize_if_dag_complete
    # is a no-op (returns None) for any run that is not yet DAG-terminal.
    finalized = await dispatcher.finalize_if_dag_complete(row.id)
    if finalized is not None:
        if redis is not None:
            await _clear_redispatch(redis, row.id)
        return "finalized"

    # A live execution lease (heartbeat-renewed by its owning pod) is the
    # authoritative liveness signal — it outranks updated_at age. Never fail a
    # legitimately long-running stage, never re-dispatch a run still executing
    # on another pod. Only leaseless runs are orphans.
    if await dispatcher.lease_present(row.id):
        # ⚠️ Deliberately does NOT clear the redispatch tally (issue #945).
        #
        # This branch used to call `_clear_redispatch` here on the theory that
        # "the run is alive under a lease" means "it is making real progress".
        # That is false for exactly the run this counter exists to catch: a
        # run wedged on ONE stage that always times out re-acquires the lease
        # on every redispatch (it runs for the length of that stage's timeout
        # before releasing), so at least one reconcile tick almost always lands
        # while it is "alive" — and that tick wiped the tally back to 0 right
        # before it was needed. Measured on run `01a0698e-1789…`: the dispatcher
        # logged stage-timeout "resume attempt 1" then "resume attempt 2" (that
        # counter, `StageOutputRepository.increment_attempts`, DOES persist —
        # it is a DB column), yet this reconciler logged `attempt=1` on BOTH
        # redispatches, because the tally was reset in between. No number of
        # redispatches could ever reach `_RECONCILER_MAX_REDISPATCH`, so the
        # ceiling never tripped and the run retried until the outer run
        # deadline (`DAGExecutor._run_deadline_at`) finally cut it off.
        #
        # The tally is now cleared ONLY on `finalized` (the run reached a real
        # terminal state) — see above — so "alive" merely skips this tick's
        # redispatch without erasing the count of tries so far. The tradeoff
        # this accepts: a HEALTHY run interrupted by several drains/rollouts
        # within the TTL window can now also accumulate toward the cap and, in
        # the rare case of 6 drains inside one hour, be failed rather than
        # resumed indefinitely. That is the correct trade for a ceiling meant
        # to bound retries: a cap that can be silently reset by the exact
        # failure mode it exists to catch is not a cap.
        return "alive"

    # Durable-claim fallback: a run dispatched while Redis was down holds NO
    # Redis lease but DOES carry a heartbeat-refreshed DB claim. Once Redis
    # recovers this reconciler would otherwise see it as leaseless and
    # re-dispatch it onto a second pod — re-introducing the double-execution the
    # claim exists to prevent. A fresh claim is an equally-authoritative liveness
    # signal, so treat it as alive too. Same non-reset reasoning as the lease
    # branch above (issue #945) — do NOT clear the redispatch tally here either.
    if await dispatcher.db_claim_fresh(row.id):
        return "alive"

    # updated_at is server-default + onupdate=func.now(), so it is always
    # populated for any row that reached the dispatcher.
    age = now - row.updated_at
    attempts = await _note_redispatch(redis, row.id) if redis is not None else 0
    wedged = redis is not None and attempts > _RECONCILER_MAX_REDISPATCH
    if age > cutoff or wedged:
        reason = (
            (
                f"orphan run re-dispatched {_RECONCILER_MAX_REDISPATCH} times without "
                f"reaching a terminal state; wedged"
            )
            if wedged
            else (f"orphan run exceeded reconciler max-age ({max_age_seconds}s); age={age}")
        )
        # Bind this run's tenant before the UPDATE so it passes RLS under enforce
        # (the cross-tenant scan came from the bypass function, which doesn't bind
        # the GUC). No-op until enforce. dispatch() below re-binds via run_tenant.
        await bind_session_tenant(session, row.tenant_id)
        # ``failure_class`` is written EXPLICITLY rather than inherited from a
        # default, because there is none: this is the one run-failure writer that
        # bypasses ``RunRepository.mark_failed`` (it works from the cross-tenant
        # bypass scan's row, not a per-run repository). Without it the row would
        # land 'failed' with a NULL class and read as "never classified" rather
        # than "classified, and the evidence did not attribute".
        #
        # UNKNOWN is the honest answer and must stay that way. A run that aged
        # out or wedged past max-redispatch died WITHOUT any pod publishing a
        # terminal event, which is precisely the state we know least about: it is
        # equally consistent with a node eviction (external), a dispatcher
        # deadlock (ours), and a stage hung on a customer URL (theirs). Nothing
        # reachable from here separates those, so this site may not choose.
        #
        # ``Run.status.in_(RunRepository._NON_TERMINAL)`` is the compare-and-set
        # EVERY other run-status mutator carries (``mark_completed``,
        # ``mark_failed``, ``mark_cancelled``, and ``mark_started``, which was
        # fixed for this exact reason). This site bypasses ``RunRepository``
        # entirely and did not carry the guard across.
        #
        # The race it closes: the orphan scan above is a SNAPSHOT
        # (``status IN ('pending','running')``), and this function then walks the
        # rows sequentially with several network awaits per row, so by the time a
        # late row is reached its status may be minutes stale. Meanwhile the OTHER
        # replica services a user cancel: ``mark_cancelled`` CAS's the row to
        # ``cancelled``, commits, and releases the lease — so ``lease_present``
        # and ``db_claim_fresh`` are both False here and the age backstop fires.
        # Unguarded, this UPDATE overwrote a genuine ``cancelled``/``completed``
        # with ``failed``/``unknown`` while ``completed_at`` still held the
        # earlier terminal timestamp: an internally incoherent row, and (via the
        # reap below) an unrecoverable one.
        #
        # The terminal set is imported rather than retyped so this site can never
        # drift from the repository's definition of "terminal".
        result = await session.execute(
            update(Run)
            .where(Run.id == row.id, Run.status.in_(RunRepository._NON_TERMINAL))
            .values(status="failed", error=reason, failure_class=UNKNOWN)
        )
        won = int(getattr(result, "rowcount", 0) or 0) > 0
        if not won:
            # Another pod reached a terminal state first. The run IS finished —
            # so clear the redispatch tally (same reasoning as the
            # ``finalize_if_dag_complete`` branch: a real terminal state is the
            # ONE thing that resets it) and report 'finalized', not 'stale'.
            # Emphatically do NOT reap: reaping is the cleanup half of a
            # transition THIS site made, and the winner already ran its own.
            if redis is not None:
                await _clear_redispatch(redis, row.id)
            log.info(
                "reconciler: orphan run %s reached a terminal state concurrently "
                "(scan saw status=%s); leaving it alone",
                row.id,
                row.status,
            )
            return "finalized"
        # The CLEANUP half of the transition. This site bypasses
        # ``RunRepository.mark_failed`` (it works from the cross-tenant bypass
        # scan), and it also bypassed everything the dispatcher's failure gates do
        # AFTER it: the stranded stage rows were never swept, the PAID sub-DAG
        # children (course lessons, parent_run_id=None) were never reaped, the
        # run's own in-flight compile kept rendering and billing, and the owning
        # project was never re-derived so it sat 'producing' with no run left to
        # move it. See ``RunDispatcher.reap_terminalized_run``. Issued AFTER the
        # UPDATE so the project recompute reads the now-failed row.
        await dispatcher.reap_terminalized_run(session, row.id)
        if redis is not None:
            await _clear_redispatch(redis, row.id)
        # A run the reconciler gave up on (wedged past max-redispatch, or aged
        # out) dies WITHOUT any pod publishing a terminal SSE event, so it never
        # reaches the dispatcher's runs_total counter. Count it here so these
        # silent run deaths are visible + alertable.
        runs_total.labels(outcome="stale_failed").inc()
        log.warning(
            "reconciler: marked orphan run %s failed (was status=%s, age=%s): %s",
            row.id,
            row.status,
            age,
            reason,
        )
        return "stale"

    log.info(
        "reconciler: resuming orphan run %s (status=%s, age=%s, attempt=%d)",
        row.id,
        row.status,
        age,
        attempts,
    )
    await dispatcher.dispatch(row.id)
    return "resumed"


async def _run_periodic_reconciler(
    session_factory: async_sessionmaker[AsyncSession],
    dispatcher: RunDispatcher,
    max_age_seconds: int,
    interval_seconds: int,
    redis: RedisClient | None = None,
) -> None:
    """Re-run the orphan reconciler every ``interval_seconds`` for the pod's
    whole lifetime.

    The startup-only reconcile has a hole: a run wedged by a *hard-killed* pod
    (SIGKILL/OOM, no graceful lease release) keeps its Redis lease until the
    60s TTL expires. If the next pod boots and reconciles inside that window,
    the re-dispatch hits the still-present lease, logs "already executing …
    skipping", and never retries — the run hangs in ``running`` until some
    unrelated future restart happens to land after the lease expired. Running
    the reconciler on a timer closes that hole: once the dead lease TTL-expires,
    the next tick finds the run leaseless and resumes it (self-heal in ≲2
    intervals). The per-run lease at ``_run`` still prevents double-execution.
    """
    log = logging.getLogger(__name__)
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await _reconcile_orphan_runs(
                session_factory,
                dispatcher,
                max_age_seconds,
                redis=redis,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # A single bad tick (transient DB/Redis error) must never kill the
            # loop — log and keep sweeping on the next interval. Count it so a
            # persistently-failing reconciler (which would silently let orphan
            # runs pile up) is alertable instead of buried in pod logs.
            db_errors_total.labels(op="reconciler_tick").inc()
            log.exception("periodic reconciler tick failed; continuing")
