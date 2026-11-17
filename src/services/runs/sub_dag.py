"""Sub-DAG primitives — spawn N child runs and await all completions.

Used by CourseBlueprint's compile_lessons stage to dispatch one lesson run
per child lesson project, await the lessons' run_completed events on
``runs:<id>`` channels, and collect their lesson_compile outputs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from redis.asyncio import Redis
from redis.asyncio.client import PubSub
from redis.exceptions import (
    AuthenticationError,
    AuthenticationWrongNumberOfArgsError,
    RedisError,
)
from redis.exceptions import ConnectionError as RedisConnectionError

from src.config import get_settings
from src.database.repositories.run_repository import RunRepository
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.database.session import bind_session_tenant
from src.models.failure_class import FAILURE_CLASSES, UNKNOWN, FailureClass

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)

SessionFactory = Callable[[], AbstractAsyncContextManager[Any]]
TerminalRunStatus = {"completed", "failed", "cancelled"}
#: Status we record for a child whose ``runs`` row we have never been able to
#: read. Distinct from the DB's own ``pending`` (dispatched, not yet started) so
#: a timeout can tell a DISPATCH failure from a child that is merely slow.
_UNKNOWN_STATUS = "unknown"
_MISSING_STATUS = "missing"
#: Non-terminal statuses meaning "this child never got going". Everything else
#: non-terminal (``running``, ``paused``, ...) is a child that IS in progress.
NotStartedRunStatus = {_UNKNOWN_STATUS, _MISSING_STATUS, "pending"}
_EVENT_TO_STATUS = {
    "run_completed": "completed",
    "run_failed": "failed",
    "run_cancelled": "cancelled",
}


class ChildRunError(Exception):
    """A child run did not complete, so its parent cannot either.

    ``failure_class`` INHERITS the child's, because a cascade has no fault of its
    own: the parent did nothing wrong, and attributing this error on its own type
    would file every sub-DAG cascade as ``unknown`` no matter how well the child
    was attributed. The child's class is reachable — ``_raise_on_non_completed``
    already loads the child's ``runs`` row to read ``error``, and
    ``failure_class`` is the column written beside it by ``mark_failed``.

    ``None`` (a child that never failed, e.g. ``cancelled``) and ``'unknown'``
    both leave the attribute at ``UNKNOWN``, which ``attribute_exception`` skips
    rather than accepting — so an unattributed child cannot veto a definite
    answer from a cause deeper in the parent's own chain.
    """

    def __init__(
        self,
        run_id: UUID,
        status: str,
        error: str | None,
        *,
        failure_class: FailureClass = UNKNOWN,
    ) -> None:
        super().__init__(f"child run {run_id} ended in {status}: {error}")
        self.run_id = run_id
        self.status = status
        self.error = error
        self.failure_class = failure_class


# Backwards-compat alias for callers using the historical name.
ChildRunFailed = ChildRunError

# The only sub-DAG stage that spawns independent child RUNS today. Its children
# are tracked in the stage output's ``children_by_deliverable`` map, NOT via the
# ``runs.parent_run_id`` FK (they are spawned with ``parent_run_id=None``).
_SUBDAG_CHILD_STAGE = "course_compile_lessons"


async def subdag_child_run_ids(session: AsyncSession, parent_run_id: UUID) -> list[UUID]:
    """Child RUN ids a sub-DAG stage (``course_compile_lessons``) spawned under
    ``parent_run_id``.

    Read from that stage output's ``children_by_deliverable`` map. Each id is an
    independent PAID lesson render spawned with ``parent_run_id=None`` — so
    ``RunRepository.list_active_children`` (the FK walk the cancel cascade uses)
    cannot see them, and this map is the ONLY handle on them. Returns ``[]`` for a
    non-course parent (no such stage output) or before the stage recorded any
    child. ``session`` must be bound to the parent's tenant (the children share
    it). Malformed entries are skipped, never raised.
    """
    so = await StageOutputRepository(session).get(parent_run_id, _SUBDAG_CHILD_STAGE)
    if so is None or not so.output:
        return []
    mapping = (so.output or {}).get("children_by_deliverable") or {}
    out: list[UUID] = []
    for value in mapping.values():
        try:
            out.append(UUID(str(value)))
        except ValueError, TypeError:
            continue
    return out


async def _seed_initial_status(
    pending: dict[UUID, str], session_factory: SessionFactory, tenant_id: UUID | None
) -> None:
    await _refresh_status(pending, list(pending.keys()), session_factory, tenant_id)


async def _poll_db_for_terminal(
    pending: dict[UUID, str], session_factory: SessionFactory, tenant_id: UUID | None
) -> None:
    open_ids = [rid for rid, s in pending.items() if s not in TerminalRunStatus]
    if not open_ids:
        return
    await _refresh_status(pending, open_ids, session_factory, tenant_id)


async def _refresh_status(
    pending: dict[UUID, str],
    run_ids: Sequence[UUID],
    session_factory: SessionFactory,
    tenant_id: UUID | None,
) -> None:
    """Record each child's CURRENT status, not only its terminal one.

    Recording the live non-terminal status (``pending`` vs ``running``) is what
    lets a timeout say whether a child was never dispatched or is just slow; the
    loop still only exits on ``TerminalRunStatus``, so widening what we store
    does not widen what we treat as done. A row we cannot read at all is
    ``missing`` — the strongest available signal of a failed dispatch.
    """
    async with session_factory() as session:
        await bind_session_tenant(session, tenant_id)
        # One ``WHERE id IN (...)`` per poll, not one SELECT per child: a course
        # parent polls every open lesson every ``poll_interval_s``.
        found = await RunRepository(session).statuses(run_ids)
        for rid in run_ids:
            pending[rid] = found.get(rid, _MISSING_STATUS)


def _apply_pubsub_message(pending: dict[UUID, str], msg: dict) -> None:
    try:
        data = json.loads(msg["data"])
        rid = UUID(data["run_id"])
        ev = data["event"]
    except json.JSONDecodeError, KeyError, ValueError, TypeError:
        return
    new_status = _EVENT_TO_STATUS.get(ev)
    if new_status is not None and rid in pending:
        pending[rid] = new_status


#: Wait bound used when the parent's REMAINING budget cannot be computed (no
#: ``parent_run_id`` passed, the run row is unreadable, or RUN_DEADLINE_SECONDS
#: is disabled). 4800s = 80 min, chosen against three live numbers rather than
#: as a round constant:
#:   * strictly below ``run_deadline_seconds`` (14400) — a stuck child must
#:     surface as a named ChildRunError long before the RUN deadline fires, so
#:     the run dies pointing at the child instead of at the clock;
#:   * strictly below the E2E scenario deadline (5400) — otherwise the scenario
#:     always wins the race and the failure presents as an opaque
#:     "STALLED STAGE course_compile_lessons", which is exactly the observed
#:     symptom (run 32927710090) this bound exists to remove;
#:   * just under the longest EVER successful run in this system (1.44h =
#:     5184s). The old constant was 7200 — above the ENTIRE successful
#:     distribution, so it could never fire on anything that was still alive and
#:     protected nothing. Sitting slightly below the p100 outlier is the accepted
#:     trade: a child still open at 80 minutes is, empirically, dead.
_FALLBACK_CHILD_WAIT_S = 4800.0
#: Extra wait granted per child BEYOND the first. ``_FALLBACK_CHILD_WAIT_S`` was
#: sized against ONE child run; a course spawns N lessons at once and they share
#: the same GPU pool, so they render largely in SERIES — a flat 4800s cap
#: terminal-failed a healthy 5-lesson course whose lessons were simply queued.
#: The cap is ``_FALLBACK_CHILD_WAIT_S + (n - 1) * _PER_EXTRA_CHILD_WAIT_S`` and is
#: still bounded by the parent's remaining run budget (minus the safety margin),
#: so scaling it can never let a child wait outlive the run it protects.
_PER_EXTRA_CHILD_WAIT_S = 1800.0
#: Subtracted from the parent's remaining budget so the child-wait raises BEFORE
#: the run deadline does, leaving room for the ChildRunError to be recorded and
#: the run finalised rather than being cut off mid-unwind.
_CHILD_WAIT_SAFETY_MARGIN_S = 120.0
#: Floor. If the parent is already past (or within a margin of) its deadline we
#: still give the children one short slice so the raise names a CHILD rather
#: than degenerating into an instant timeout with nothing observed.
_MIN_CHILD_WAIT_S = 60.0


def _now() -> float:
    """The single monotonic clock this module measures its budgets against.

    Extracted so the reconnect budget below is provable in milliseconds of wall
    time: a 300-second ladder cannot be waited out in a unit test, and an
    attempt-COUNT assertion would not pin the property that matters — the budget
    is denominated in SECONDS because what it has to outlast is a pod rollout
    measured in seconds. Mirrors ``_FakeClock`` in
    ``tests/unit/test_e2e_validate.py``.
    """
    return asyncio.get_event_loop().time()


# --- redis pubsub reconnect (E2E #11 root cause) ----------------------------
# ``await_runs`` subscribes to ``runs:<child_id>`` for the WHOLE lifetime of
# ``course_compile_lessons`` — tens of minutes, because each child is a full
# lesson render. Over that window a redis restart is not exceptional, it is
# EXPECTED: `clusters/strato/services/pipeline-events-redis/redis.yaml` is
# `replicas: 1` with `strategy: type: Recreate`, so the only pod is destroyed
# before its replacement is created, and every merge/image-bump/node event
# causes one.
#
# Until this ladder existed, `pubsub.get_message()` was UNGUARDED. A restart
# raised `redis.exceptions.ConnectionError("Connection closed by server.")`
# straight out of the stage handler, and `dag_executor` wrote a TERMINAL
# `failed` for a course whose lesson children were all still running happily.
# That is exactly E2E attempt #11 (run 33864937267): 38 success / 1 failure,
# the one failure being `course_v1` with
# `stage_outputs.course_compile_lessons = failed,
#  ConnectionError: Connection closed by server.`
#
# Why retrying here is UNCONDITIONALLY safe — this loop is a pure OBSERVER.
# It spawns nothing, bills nothing and mutates nothing; the child runs are
# independent, durable rows dispatched BEFORE the wait begins. The pubsub is a
# latency optimisation over `_poll_db_for_terminal`, which alone is already
# sufficient for correctness — so a reconnect can neither duplicate paid work
# nor lose an event (we re-read every open child's status from the DB the
# moment we are back). Contrast the HTTP seam, where replaying a POST without
# an idempotency key WOULD double-bill.
#
# The budget. Redis's own Recreate roll is ~75s worst case (30s default
# terminationGracePeriodSeconds, no initContainer, + schedule/pull/AOF-load,
# then readiness `initialDelaySeconds: 5` / `periodSeconds: 5` => <=10s to
# Ready). But the observed outage was COMPOUND: the harness also logged
# `[Errno -3] Temporary failure in name resolution` six times over ~60s, and a
# merge rolls every deployment in the namespace at once, so the orchestrator's
# own ~215s roll (see `_POLL_RECONNECT_BACKOFF` in `scripts/e2e_validate.py`)
# can overlap. 300s covers 75 + 60 with headroom and matches the number #944
# already justified for the harness->orchestrator leg, so the two layers now
# tolerate the SAME outage rather than one absorbing what the other drops.
#
# Why 300s cannot swallow a real outage: it is a CEILING, not a floor. The
# child-wait `deadline` stays authoritative (see the `_now() + delay >=
# deadline` gate), and 300s is at most 6.25% of `_FALLBACK_CHILD_WAIT_S`
# (4800) and 2.1% of `run_deadline_seconds` (14400) — a genuinely dead redis
# still fails the stage promptly and loudly instead of being laundered into a
# timeout. The ladder is also PER-EPISODE, reset by any successful poll: a
# disconnect at t=100 and another at t=3000 are unrelated events.
#
# 1s, 2s, 4s, 8s, then 15s — 23 delays summing to exactly 300s across 24
# connect attempts. An explicit schedule (rather than "loop until the clock
# says stop") keeps exhaustion deterministic and unit-testable.
_PUBSUB_RECONNECT_BACKOFF: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0) + (15.0,) * 19

#: Connection-class redis failures: the socket died, or was never established.
#: Scoped DELIBERATELY narrow. ``RedisError`` at large would sweep in
#: ``ResponseError`` (a healthy server REFUSING us — a real defect that must
#: fail fast, the analogue of a 5xx) and turn a clear failure into a 300s one.
#: ``OSError`` is listed because a DNS failure can surface raw from the socket
#: layer before redis-py wraps it; note redis-py's ``ConnectionError`` is NOT
#: an ``OSError`` subclass (it descends from ``RedisError``), so both are
#: needed.
_PUBSUB_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (RedisConnectionError, OSError)

#: Authentication failures subclass redis-py's ``ConnectionError`` and would
#: otherwise be caught by the tuple above. A rejected credential is not a
#: transient: retrying it for 300s hammers a healthy server and converts an
#: instant, actionable error into a slow one. Checked FIRST.
_PUBSUB_FATAL_ERRORS: tuple[type[Exception], ...] = (
    AuthenticationError,
    AuthenticationWrongNumberOfArgsError,
)


async def _open_pubsub(
    redis: Redis,
    channels: Sequence[str],
    *,
    deadline: float,
    cause: BaseException | None = None,
) -> PubSub:
    """Return a pubsub subscribed to ``channels``, surviving a redis restart.

    Retries connection-class failures on a FRESH pubsub (the old connection is
    dead and its pool entry unusable) with the ``_PUBSUB_RECONNECT_BACKOFF``
    ladder. Fatal errors — auth, and anything outside the connection class —
    propagate on the first attempt, unretried.

    ``deadline`` is a ``_now()`` instant past which no further wait is
    permitted, so the reconnect budget can never outlive the caller's own
    child-wait: the deadline decides the stage's fate, not this ladder. Mirrors
    ``_request_with_retry``'s ``not_after`` in ``scripts/e2e_validate.py``.

    ``cause`` is the error that triggered a RECONNECT (``None`` on the initial
    connect); it is re-raised if the very first re-attempt is what exhausts the
    budget, so the stage still fails naming the original disconnect.
    """
    attempt = 0
    last: BaseException | None = cause
    while True:
        attempt += 1
        pubsub_attempt = redis.pubsub()
        try:
            await pubsub_attempt.subscribe(*channels)
        except _PUBSUB_FATAL_ERRORS:
            await _release_pubsub(pubsub_attempt, channels)
            raise
        except _PUBSUB_RETRYABLE_ERRORS as exc:
            last = exc
            # redis-py checks a connection OUT of the pool inside `subscribe`
            # and keeps it on the PubSub even when the command then fails, so a
            # discarded attempt holds one. Across a full 24-attempt ladder that
            # is 24 connections gone — the pool exhausts before the outage ends
            # and the reconnect can never succeed. Release every attempt we
            # throw away; `_release_pubsub` never raises.
            await _release_pubsub(pubsub_attempt, channels)
            total = len(_PUBSUB_RECONNECT_BACKOFF) + 1
            if attempt >= total:
                log.error(
                    "sub_dag: redis pubsub unreachable after %d attempts (~%.0fs); giving up",
                    total,
                    sum(_PUBSUB_RECONNECT_BACKOFF),
                )
                raise
            delay = _PUBSUB_RECONNECT_BACKOFF[attempt - 1]
            if _now() + delay >= deadline:
                # Waiting would cross the child-wait deadline. Stop here so the
                # deadline — not this ladder — decides the stage's fate; a
                # budget that outran it would only convert a clear connection
                # failure into an opaque timeout.
                log.error(
                    "sub_dag: pubsub reconnect budget would outlive the child-wait "
                    "deadline; giving up after attempt %d/%d",
                    attempt,
                    total,
                )
                raise
            log.warning(
                "sub_dag: redis pubsub connect failed (attempt %d/%d), retrying in %.0fs: %r",
                attempt,
                total,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
        else:
            if attempt > 1 or last is not None:
                # Make a redis restart VISIBLE. A silently-absorbed reconnect
                # is how a real outage would hide.
                log.warning(
                    "sub_dag: redis pubsub RECONNECTED on attempt %d after %r",
                    attempt,
                    last,
                )
            return pubsub_attempt


def _child_wait_cap_s(child_count: int) -> float:
    """Upper bound on the fan-out wait for ``child_count`` concurrently-spawned children."""
    return _FALLBACK_CHILD_WAIT_S + max(child_count - 1, 0) * _PER_EXTRA_CHILD_WAIT_S


async def _derive_child_wait_s(
    parent_run_id: UUID | None,
    session_factory: SessionFactory,
    tenant_id: UUID | None,
    child_count: int = 1,
) -> tuple[float, str]:
    """Effective child-wait bound + a short string explaining where it came from.

    Derived from the parent run's REMAINING budget (``run_deadline_seconds``
    measured from ``runs.started_at``, the same anchor ``RunDispatcher`` uses)
    so this guard is always strictly inside the run it is protecting. Any
    failure to compute it degrades to the child-count cap (``_child_wait_cap_s``,
    which is ``_FALLBACK_CHILD_WAIT_S`` for a single child) — this is a bound,
    never a reason to fail a run, so the except is deliberately wide.

    ``child_count`` scales the cap: N children spawned at once on a shared GPU
    pool render largely in series, so a flat single-child cap kills healthy
    multi-lesson courses. Time the parent spent PAUSED (``runs.paused_seconds``)
    is added back to its budget, matching ``run_dispatcher._run_deadline_at``.
    """
    cap = _child_wait_cap_s(child_count)
    if parent_run_id is None:
        return cap, "fallback:no-parent-run-id"
    budget = get_settings().pipeline.run_deadline_seconds
    if budget <= 0:
        return cap, "fallback:run-deadline-disabled"
    try:
        async with session_factory() as session:
            await bind_session_tenant(session, tenant_id)
            parent = await RunRepository(session).get(parent_run_id)
            # Read the column while the row is still ATTACHED: `started_at` is
            # stamped by mark_started's bare UPDATE, so it is exactly the kind of
            # attribute that raises DetachedInstanceError once the session closes
            # (same trap documented on run_dispatcher._run_deadline_at).
            started_at: datetime | None = getattr(parent, "started_at", None)
            paused_seconds = int(getattr(parent, "paused_seconds", 0) or 0)
    except Exception as exc:  # noqa: BLE001 - a bound must never fail the run
        log.warning("sub_dag: could not read parent %s for child wait: %s", parent_run_id, exc)
        return cap, "fallback:parent-unreadable"
    if started_at is None:
        return cap, "fallback:no-started-at"
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=UTC)
    remaining = budget + paused_seconds - (datetime.now(UTC) - started_at).total_seconds()
    bound = min(remaining - _CHILD_WAIT_SAFETY_MARGIN_S, cap)
    return (
        max(bound, _MIN_CHILD_WAIT_S),
        f"remaining={remaining:.0f}s of {budget}s (+{paused_seconds}s paused); "
        f"cap={cap:.0f}s for {child_count} children",
    )


def _pending_breakdown(pending: dict[UUID, str]) -> str:
    """Timeout detail that separates a DISPATCH failure from a slow child."""
    not_started = {rid: s for rid, s in pending.items() if s in NotStartedRunStatus}
    in_progress = {
        rid: s
        for rid, s in pending.items()
        if s not in NotStartedRunStatus and s not in TerminalRunStatus
    }
    done = {rid: s for rid, s in pending.items() if s in TerminalRunStatus}
    return (
        f"never_started={ {str(k): v for k, v in not_started.items()} }; "
        f"in_progress={ {str(k): v for k, v in in_progress.items()} }; "
        f"terminal={ {str(k): v for k, v in done.items()} }"
    )


async def _raise_on_non_completed(
    pending: dict[UUID, str],
    session_factory: SessionFactory,
    tenant_id: UUID | None,
) -> None:
    """Raise ``ChildRunError`` for the first child that did not complete.

    Extracted from ``await_runs`` purely to keep it under the C901 ceiling; the
    behaviour — first non-completed child wins, error text read from its own
    ``runs`` row — is unchanged.
    """
    for rid, status in pending.items():
        if status != "completed":
            async with session_factory() as session:
                await bind_session_tenant(session, tenant_id)
                run = await RunRepository(session).get(rid)
                # Read the child's OWN attribution off the same row as its error
                # text. Coerced through the vocabulary tuple rather than trusted:
                # the column is CHECK-constrained, but a NULL (never-failed
                # child) and a legacy pre-arming row both have to land on UNKNOWN.
                child_class: FailureClass = UNKNOWN
                # ``getattr`` with a default, not attribute access: this must never
                # be the thing that turns "the child failed" into an AttributeError
                # raised from the reporting path. The column is real on ``Run``, but
                # the raise site is on a failure path and has to stay total.
                persisted = getattr(run, "failure_class", None) if run is not None else None
                for member in FAILURE_CLASSES:
                    if persisted == member:
                        child_class = member
                        break
                raise ChildRunError(
                    rid,
                    status,
                    run.error if run is not None else None,
                    failure_class=child_class,
                )


async def await_runs(
    *,
    run_ids: Sequence[UUID],
    session_factory: SessionFactory,
    redis: Redis,
    tenant_id: UUID | None = None,
    parent_run_id: UUID | None = None,
    fail_on_any: bool = True,
    poll_interval_s: float = 2.0,
    timeout_s: float | None = None,
    child_count: int | None = None,
) -> dict[UUID, str]:
    """Block until every run_id is completed/failed/cancelled. Returns final status
    dict. If fail_on_any and any non-completed, raise ChildRunError.

    ``tenant_id`` is the parent run's tenant; the child runs share it, so it scopes
    the DB-poll sessions under RLS enforce (no-op until then).

    ``timeout_s`` defaults to the parent run's REMAINING budget (see
    ``_derive_child_wait_s``) rather than a constant. It was a fixed 7200s, which
    sat between the E2E scenario deadline (5400s) and ``run_deadline_seconds``
    (14400s): above every successful run this system has ever produced, so it
    never fired first, and a stuck child parked its parent for two hours and then
    presented as a stalled STAGE instead of a named child. Pass ``timeout_s``
    explicitly to override (callers with their own budget, and tests).

    ``child_count`` scales the derived cap (see ``_child_wait_cap_s``); a
    fan-out caller passes how many children it spawned onto the shared pool.
    ``None`` means one child — the single-child cap every caller had before."""
    if timeout_s is None:
        timeout_s, bound_source = await _derive_child_wait_s(
            parent_run_id, session_factory, tenant_id, child_count or 1
        )
    else:
        bound_source = "explicit"
    pending: dict[UUID, str] = {rid: _UNKNOWN_STATUS for rid in run_ids}
    channels = [f"runs:{rid}" for rid in run_ids]
    deadline = _now() + timeout_s
    pubsub: PubSub | None = await _open_pubsub(redis, channels, deadline=deadline)
    try:
        await _seed_initial_status(pending, session_factory, tenant_id)
        while any(s not in TerminalRunStatus for s in pending.values()):
            if _now() >= deadline:
                raise TimeoutError(
                    f"sub_dag await_runs timed out after {timeout_s:.0f}s "
                    f"(bound={bound_source}); {_pending_breakdown(pending)}"
                )
            try:
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=poll_interval_s
                )
            except _PUBSUB_FATAL_ERRORS:
                raise
            except _PUBSUB_RETRYABLE_ERRORS as exc:
                # Redis went away mid-wait (a Recreate roll, a node event, a DNS
                # blip). The children are untouched and still rendering — this
                # loop only WATCHES them — so reconnect instead of terminal-
                # failing a course whose lessons are all fine. See the
                # `_PUBSUB_RECONNECT_BACKOFF` block for why this is safe and how
                # the budget is sized.
                log.warning(
                    "sub_dag: redis pubsub dropped mid-wait, reconnecting: %r",
                    exc,
                )
                # Return the dead connection to the pool before replacing it,
                # so a flapping redis cannot leak one entry per reconnect.
                # `pubsub` is cleared FIRST: if the reconnect below then
                # exhausts its budget, the `finally` must not release this same
                # object a second time. A double release is not harmless —
                # redis-py's `unsubscribe` re-acquires a connection from the
                # pool when it no longer holds one, so the second call would
                # dial redis again purely to unsubscribe a dead subscription.
                dead, pubsub = pubsub, None
                await _release_pubsub(dead, channels)
                pubsub = await _open_pubsub(redis, channels, deadline=deadline, cause=exc)
                # Close the observation gap: any `run_completed` published while
                # we were disconnected was delivered to nobody. The DB is the
                # durable record, so re-read every open child rather than
                # waiting for an event that has already been and gone.
                await _poll_db_for_terminal(pending, session_factory, tenant_id)
                continue
            if msg is not None:
                _apply_pubsub_message(pending, msg)
            else:
                await _poll_db_for_terminal(pending, session_factory, tenant_id)
    finally:
        # SHIELDED. This block runs on the way out of a cancellation — a stage
        # timeout (middleware/timeout.py wraps every handler in wait_for) or an
        # explicit task.cancel() from dag_executor's batch-abort. A SECOND cancel
        # delivered while the first is unwinding lands on the bare `await` below,
        # propagates straight out of the `finally`, and `_close_pubsub` never runs:
        # the connection is never returned to the pool and stays subscribed to
        # `runs:<id>` forever. Under a run that times out and gets re-dispatched
        # every 90s that leaks one connection per attempt until the pool is gone.
        # `except CancelledError` would not help — it is a BaseException and the
        # tuple below cannot catch it; shielding is what makes the cleanup finish.
        if pubsub is not None:
            await asyncio.shield(_release_pubsub(pubsub, channels))
    if fail_on_any:
        await _raise_on_non_completed(pending, session_factory, tenant_id)
    return pending


class _Unsubscribable(Protocol):
    """What ``_release_pubsub`` needs of a pubsub, and nothing more.

    Structural rather than ``redis.asyncio.client.PubSub`` so the teardown tests can
    drive the real function with a recording stub instead of a ``ty: ignore``.

    Declared as a sync method returning an awaitable, not ``async def``: that is how
    redis-py types ``PubSub.unsubscribe`` (``(*args) -> Awaitable``, and it is NOT a
    coroutine function), and an ``async def`` here is not structurally satisfied by it.
    """

    def unsubscribe(self, *channels: str) -> Awaitable[object]: ...


async def _release_pubsub(pubsub: _Unsubscribable, channels: Sequence[str]) -> None:
    """Unsubscribe and close, never raising on a teardown-only failure.

    Kept as one coroutine so ONE ``asyncio.shield`` covers both awaits: shielding
    them separately would leave the gap between them cancellable, which is the
    same hole one line narrower.
    """
    try:
        await pubsub.unsubscribe(*channels)
    except (RedisError, ConnectionError, OSError) as exc:
        log.debug("sub_dag pubsub unsubscribe error: %s", exc)
    try:
        await _close_pubsub(pubsub)
    except (RedisError, ConnectionError, OSError) as exc:
        # Was unguarded, which broke this function's own "never raising on a
        # teardown-only failure" contract: the `finally` that shields this call
        # sits under the child-wait, so a close error REPLACED the real
        # ChildRunError/TimeoutError and the parent run failed naming redis
        # instead of the child that actually died.
        log.debug("sub_dag pubsub close error: %s", exc)


async def _close_pubsub(pubsub: object) -> None:
    """Close the pubsub via whichever method the redis-py version exposes."""
    aclose: Callable[[], Awaitable[None]] | None = getattr(pubsub, "aclose", None)
    if aclose is not None:
        await aclose()
        return
    close = getattr(pubsub, "close", None)
    if close is not None:
        result = close()
        if asyncio.iscoroutine(result):
            await result


async def collect_outputs(
    *,
    run_ids: Sequence[UUID],
    stage_name: str,
    session_factory: SessionFactory,
    tenant_id: UUID | None = None,
) -> dict[UUID, dict[str, Any]]:
    """Read ``stage_outputs.output`` for ``stage_name`` from each run."""
    out: dict[UUID, dict[str, Any]] = {}
    async with session_factory() as session:
        await bind_session_tenant(session, tenant_id)
        repo = StageOutputRepository(session)
        for rid in run_ids:
            so = await repo.get(rid, stage_name)
            if so is None or so.status != "completed" or so.output is None:
                raise RuntimeError(f"child run {rid} missing completed output for {stage_name}")
            out[rid] = dict(so.output)
    return out
