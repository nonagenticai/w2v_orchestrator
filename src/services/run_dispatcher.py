"""Background run executor.

Receives a run_id, loads everything fresh in its own session, runs the DAG,
emits SSE events via Redis pub/sub, persists stage outputs.

The per-run execution lease (one pod at a time) lives in ``run_lease.RunLease``;
the completion accounting + terminal-artifact verification live in
``run_finalizer``. This module is the lifecycle orchestrator that wires them
together with the DAG executor and the SSE publishers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid7

from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import select
from w2v_common.auth.identity_context import (
    clear_identity_context,
    set_identity_context,
    set_w2v_project_context,
)
from w2v_common.tenant import TenantError

from src.auth.run_attribution import bind_run_attribution
from src.auth.run_providers import bind_run_provider_slots, clear_run_provider_slots
from src.auth.run_tenant import billable_tenant_id
from src.auth.tenant_provenance import TENANT_SOURCE_KEY, tenant_source_from_snapshot
from src.config import get_settings
from src.database.models import Deliverable, Project, Run
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.database.session import bind_run_tenant
from src.models.failure_class import OUR_DEFECT, attribute_exception, failure_detail_for
from src.observability.logging import bound_correlation_id
from src.observability.metrics import active_runs, notify_errors_total, runs_total
from src.services.blueprints.registry import BlueprintRegistry
from src.services.compile_cancel import cancel_compiles_for_run
from src.services.runs.dag_executor import DAGExecutor, StageResumableTimeoutError
from src.services.runs.run_claim import RunDbClaim
from src.services.runs.run_context import RunContext
from src.services.runs.run_finalizer import (
    FAILED_STAGE_STATUSES,
    RunDecision,
    decide_run_terminal_state,
    pause_trigger_stage,
    verify_terminal_artifact,
)
from src.services.runs.run_lease import RunLease
from src.services.runs.run_setup import reconcile_blueprint_stage_rows
from src.services.runs.structural_relief import maybe_spawn_structural_relief_run
from src.services.runs.sub_dag import TerminalRunStatus, subdag_child_run_ids
from src.services.stages.shared.step_ledger import step_ledger_factory
from src.utils import describe_exception

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)

# Cross-pod run-control fan-out channel. A cancel/pause that lands on a replica
# which doesn't hold the run's execution task publishes here; every dispatcher
# subscribes and cancels the task if IT owns it. The pause-vs-cancel intent is
# NOT carried on the wire — the originating route commits the run row
# (paused/cancelled) BEFORE broadcasting, and the owner's CancelledError handler
# reads that row to decide. So this channel only needs the run id.
_CONTROL_CHANNEL = "orchestrator:run_control"

# SSE event name → run-outcome label for w2v_orchestrator_runs_total. Only
# terminal events appear here; everything else (run_started, stage_*) is
# uncounted. (Reconciler stale-fails are counted separately in asgi.py with
# outcome="stale_failed", since no pod publishes an SSE event for them.)
_TERMINAL_EVENT_OUTCOMES = {
    "run_completed": "completed",
    "run_failed": "failed",
    "run_paused": "paused",
    "run_cancelled": "cancelled",
}


def _run_deadline_at(started_at: datetime | None, paused_seconds: int = 0) -> datetime | None:
    """When this run must stop resuming, or ``None`` if it never must.

    Takes the anchor as a plain value, NOT the ``Run`` row. The row is loaded
    inside the dispatcher's session block and the executor is constructed after
    it closes, so reading a column off the instance here raised
    ``DetachedInstanceError`` for any attribute not already in ``__dict__`` —
    ``started_at`` is stamped by ``mark_started``'s bare UPDATE, so it is exactly
    that kind of attribute. Every in-process DAG test went red on it.

    ``None`` when the knob is 0 or the caller has no anchor. The caller supplies
    ``now`` for a first dispatch, mirroring ``mark_started``'s
    ``COALESCE(started_at, now())`` — so the budget is measured from the run's
    FIRST slice, not from whichever resume happens to read it.

    ``paused_seconds`` (``runs.paused_seconds``) extends the deadline by the time
    the run sat paused for review: a course paused for longer than the whole
    budget would otherwise be "expired" the moment it was resumed.
    """
    budget = get_settings().pipeline.run_deadline_seconds
    if budget <= 0 or started_at is None:
        return None
    started = started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return started + timedelta(seconds=budget + max(int(paused_seconds or 0), 0))


class RunDispatcher:
    def __init__(
        self,
        *,
        session_factory,  # noqa: ANN001
        registry: BlueprintRegistry,
        redis: Redis,
    ) -> None:
        self._sessions = session_factory
        self._registry = registry
        self._redis = redis
        self._tasks: dict[UUID, asyncio.Task] = {}
        # A unique owner token for this dispatcher instance (≈ this pod), shared
        # by the Redis lease and the durable DB claim so both guards agree on
        # ownership.
        self._instance_id = uuid7().hex
        # Per-run execution lease (fast path, Redis).
        self._lease = RunLease(redis, instance_id=self._instance_id)
        # Durable second guard used ONLY when the Redis lease falls open (Redis
        # unavailable) — an atomic DB claim so a full Redis outage can't fan a
        # run out across every replica. See RunDbClaim.
        self._claim = RunDbClaim(session_factory, instance_id=self._instance_id)
        # When True, the CancelledError handler leaves the run row in
        # `running` so the next pod's startup reconciler picks it up
        # instead of marking it `cancelled`. Distinguishes drain-cancel
        # (pod going down) from user-cancel (POST /runs/{id}/cancel).
        # See ``shutdown`` below; flipped back to False is intentionally
        # not supported — once a dispatcher has drained, it doesn't reopen.
        self._draining: bool = False
        # Run ids whose executing task was cancelled because an OWNERSHIP GUARD
        # was LOST — either the durable DB claim (heartbeat failures /
        # rowcount==0 in RunDbClaim) or the Redis lease (RunLease's own
        # consecutive-failure / reacquired-by-another-holder detection) — not
        # because of a user cancel. asyncio.Task.cancel(msg) is not a reliable
        # channel for this: a re-raised CancelledError can lose the message,
        # so `_finish_cancelled` needs an explicit, out-of-band signal to tell
        # the two apart. Populated by the `on_lost` callback in `_run` BEFORE
        # it cancels the task; consumed and cleared (in a `finally`, so a
        # crash mid-handling still clears it) by `_finish_cancelled` — this
        # dispatcher instance is long-lived and shared across runs, so a
        # leaked entry would make a LATER genuine user cancel of the same run
        # id silently behave like a claim loss instead.
        self._claim_lost_runs: set[UUID] = set()
        # Background subscriber to _CONTROL_CHANNEL (cross-pod cancel/pause).
        # Started by start_control_listener(), torn down in shutdown().
        self._control_task: asyncio.Task | None = None
        # Late-bound by app startup once both objects exist (avoids circular
        # import). Sub-DAG stages (course-level) pull this from handler_deps.
        self._run_engine: Any | None = None
        # STRONG references to shielded cleanup still in flight.
        # ``await asyncio.shield(x)`` protects ``x`` from the caller's
        # cancellation, but it does NOT keep it alive: the caller stops awaiting,
        # and asyncio holds only a weak reference to a task, so ``x`` can be
        # garbage-collected mid-transaction or cut off when the lifespan closes
        # the loop ("Task was destroyed but it is pending"). Since the shielded
        # work here is exactly the terminal transition of a cancelled run, losing
        # it reintroduces the defect the shield was added to fix. ``shutdown``
        # drains this set.
        self._cleanup_tasks: set[asyncio.Task] = set()

    def bind_run_engine(self, engine: Any) -> None:
        """Late-bind RunEngine reference for sub-DAG stage construction."""
        self._run_engine = engine

    async def dispatch(self, run_id: UUID) -> None:
        if run_id in self._tasks:
            log.warning("run %s already dispatched", run_id)
            return
        task = asyncio.create_task(self._run(run_id), name=f"run:{run_id}")
        self._tasks[run_id] = task
        task.add_done_callback(lambda _t: self._tasks.pop(run_id, None))

    async def cancel(self, run_id: UUID) -> bool:
        """True only if the task's own CancelledError handler ran to completion —
        i.e. the cancel is RECORDED. False means the caller must do the DB
        fallback itself (``_cancel_one`` marks the row cancelled).

        The handler is unshielded and touches the DB. A blip in there used to
        propagate straight out of here: the cancel route 500'd, ``mark_cancelled``
        never committed, and neither descendant sweep ran — so every paid
        course-lesson child kept rendering while the user was told the cancel
        failed. Returning False instead routes the caller onto the fallback,
        which is exactly what the never-dispatched case already does.
        ``suppress`` covered only ``CancelledError``, which was the narrower half
        of the same bug.
        """
        task = self._tasks.get(run_id)
        if task is None:
            return False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            # Our own task being cancelled from outside would land here too, but
            # that is the pre-existing contract of this method (see `suppress`
            # above) and the caller is going away regardless.
            pass
        except Exception as exc:  # noqa: BLE001 -- fall back to the DB-side cancel
            log.warning(
                "run %s cancel handler raised (%s); falling back to marking the row cancelled",
                run_id,
                exc,
            )
            return False
        return True

    async def _cancel_subdag_children(self, session: AsyncSession, run_id: UUID) -> int:
        """Cancel every non-terminal PAID child RUN a sub-DAG stage
        (``course_compile_lessons``) spawned under a run that just terminalized
        failed/cancelled.

        Those children carry ``parent_run_id=None`` — invisible to the FK-based
        descendant cascade — so nothing else reaps them and each keeps spending on
        its own DAG until it finishes. Best-effort + idempotent: an already-terminal
        or absent child is skipped; a per-child cancel error is logged, never raised
        into the parent's terminalization. ``session`` is bound to the parent's
        tenant (children share it); ``mark_cancelled`` on the child rides the caller's
        transaction, while the live/cross-pod interrupt is a side effect. Returns the
        number of children acted on. Mirrors the route cancel's ``_cancel_one``.
        """
        try:
            child_ids = await subdag_child_run_ids(session, run_id)
        except Exception as exc:  # noqa: BLE001 — child-cancel must NEVER abort the parent terminalize
            log.warning("failed to read sub-DAG children of run %s for cancel: %s", run_id, exc)
            return 0
        if not child_ids:
            return 0
        rr = RunRepository(session)
        acted = 0
        for child_id in child_ids:
            # The whole per-child body (read + cancel) is guarded: a DB or interrupt
            # error on one child must not abort the parent's fail-commit nor skip the
            # remaining children.
            try:
                child = await rr.get(child_id)
                if child is None or getattr(child, "status", None) in TerminalRunStatus:
                    continue
                acted += 1
                # Same-pod fast interrupt; else terminalize the row so it stops on
                # the owning pod (broadcast) or the reconciler backstop.
                if not await self.cancel(child_id):
                    await rr.mark_cancelled(child_id)
                # Provenance: a course retry re-spawns a CASCADE-cancelled lesson
                # but keeps a USER-cancelled one terminal (course_compile_lessons).
                await rr.mark_cancelled_by_parent(child_id, run_id)
                await self.broadcast_interrupt(child_id)
                # Stopping the child's DAG does not stop the compile it already
                # submitted — that render keeps going on the video-compile service and
                # keeps being billed. A course fans out one paid render PER LESSON, so
                # this is the largest single spend a cancel can leak. Best-effort by
                # construction (never raises, hard-timeout), which is what lets it sit
                # inside this per-child guard without endangering the parent's commit.
                await cancel_compiles_for_run(child_id)
            except Exception as exc:  # noqa: BLE001 — never abort the parent terminalize
                log.warning(
                    "failed to cancel orphaned sub-DAG child run %s of %s: %s",
                    child_id,
                    run_id,
                    exc,
                )
        if acted:
            log.info(
                "run %s terminalized: cancelled %d orphaned sub-DAG child render(s)",
                run_id,
                acted,
            )
        return acted

    async def reap_terminalized_run(self, session: AsyncSession, run_id: UUID) -> None:
        """The CLEANUP half of a terminal run transition, for the one writer that
        flips the run row itself instead of going through ``RunRepository``.

        Every other run-failure site in the estate pairs its ``mark_failed`` with
        four things (see ``_execute_run``'s two failure gates):

        1. ``bulk_cancel_non_terminal`` — the run is terminal and will never be
           resumed, so its leftover ``pending``/``running`` stage rows strand
           forever and surface as ghost stages in the UI/SSE.
        2. ``_cancel_subdag_children`` — course lessons are spawned with
           ``parent_run_id=None``, so the FK cascade cannot see them and each one
           keeps spending on its own DAG.
        3. ``cancel_compiles_for_run`` — stopping the orchestrator's task does not
           stop the render already submitted to the compile service, which keeps
           being billed until it finishes on its own.
        4. ``_recompute_project`` — the owning project (and its course parent)
           otherwise sits at ``producing`` forever with no run left to move it.

        ``lifecycle.reconciler._resolve_orphan_run``'s stale/wedged fail does none
        of them: it works from the cross-tenant bypass scan's row and issues a bare
        ``UPDATE runs``. That is the WORST path to skip them on — a run only reaches
        it after its owning pod died without publishing anything, i.e. exactly the
        state in which a submitted render is still running and still billing.

        ``session`` must already be bound to the run's tenant, and the run row must
        already have been flipped terminal in this transaction (``_recompute_project``
        re-derives the project from it). Ordering mirrors the dispatcher's gates:
        sweep, reap, then recompute.

        The two compile cancels below (the run's own, and each sub-DAG child's)
        run under the RUN's identity, not the reconciler's. This method is
        reached only from the reconciler loop, which has no inbound request, so
        ``AuthMiddleware`` never populated the identity ContextVars — and
        ``bind_session_tenant`` above it is the DB GUC only. ``VideoCompileClient``
        bakes ``X-Tenant-Id: Bearer <SA token>``, so with no tenant bound
        ``_apply_outbound_tenant`` takes the ``service_credentialed`` early return
        and the cancel goes out as the orchestrator SA tenant. Once video_compile
        is a DELEGATING destination (w2v_common#190) every compile job is filed
        under the CUSTOMER tenant the dispatcher stamped at submit (``_execute_run``
        binds identity at ~:648), so an SA-scoped ``cancel_run_jobs`` matches zero
        of them and the orphaned render runs to completion, billing the customer.
        Same idiom as ``finalize_if_dag_complete``: bind from the run's persisted
        snapshot via ``billable_tenant_id`` + ``tenant_source_from_snapshot``, and
        ``try/finally: clear_identity_context()`` so this long-lived reconciler
        task never carries one run's tenant into the next row it reaps.
        """
        swept = await StageOutputRepository(session).bulk_cancel_non_terminal(run_id)
        if swept:
            log.info(
                "run %s terminalized by the reconciler: swept %d stranded stage row(s) "
                "to 'cancelled'",
                run_id,
                swept,
            )
        # The reconciler's cross-tenant scan row carries only id/tenant_id/status/
        # updated_at; the snapshot (tenant + license + subject + provenance) is on
        # the run row, which ``session`` — already tenant-bound — can read.
        run = await RunRepository(session).get(run_id)
        _snap = dict(run.config_snapshot or {}) if run is not None else {}
        try:
            _run_tenant: UUID | None = billable_tenant_id(_snap.get("tenant_id"))
        except TenantError as exc:
            # No honest identity to bind, and inventing one is exactly the
            # fail-open the provenance ladder exists to close. The compile cancel
            # is SKIPPED rather than sent undelegated: an SA-scoped cancel cannot
            # reach a customer-filed job anyway, and the compiler's own work-
            # timeout/reaper backstops still bound the render.
            log.error(
                "run %s: reaper could not bind identity — snapshot has no billable "
                "tenant (%s); skipping the compile cancel for this run",
                run_id,
                exc,
            )
            _run_tenant = None
        if _run_tenant is not None:
            set_identity_context(
                tenant_id=_run_tenant,
                license_id=_snap.get("license_id") or None,
                subject_id=_snap.get("subject_id") or None,
                # FOURTH piece: absent for a pre-0.67 row → ``None`` → the
                # delegated hop refuses (``UnvouchedTenantError``, swallowed by
                # ``cancel_compiles_for_run``'s guard and logged) instead of
                # delegating an unvouched tenant.
                tenant_source=tenant_source_from_snapshot(_snap.get(TENANT_SOURCE_KEY)),
            )
            bind_run_attribution(_snap)
            bind_run_provider_slots(_snap)
        try:
            await self._cancel_subdag_children(session, run_id)
            # The run's OWN compile. ``_cancel_subdag_children`` covers each child's;
            # nothing covered this one. Never raises, hard-timeout by construction.
            if _run_tenant is not None:
                await cancel_compiles_for_run(run_id)
            proj_row = (
                await session.execute(
                    select(Project.id, Project.parent_project_id)
                    .join(Deliverable, Deliverable.project_id == Project.id)
                    .join(Run, Run.deliverable_id == Deliverable.id)
                    .where(Run.id == run_id)
                )
            ).first()
            if proj_row is not None:
                await self._recompute_project(session, proj_row.id, proj_row.parent_project_id)
        finally:
            # ``finally`` rather than a clear before the return: a ``return`` added
            # later inside this block could not silently reintroduce the leak.
            clear_identity_context()
            clear_run_provider_slots()

    async def broadcast_interrupt(self, run_id: UUID) -> None:
        """Tell whichever replica owns ``run_id``'s task to cancel it.

        Cancel/pause routes call ``cancel()`` first (the fast same-pod path);
        this fans the request out to the OTHER replicas so a run executing on a
        non-receiving pod is also stopped. The execution lease is exclusive, so
        at most one pod acts on it; everyone else no-ops. Best-effort — a Redis
        blip must not fail the cancel route (the row is already committed and
        the periodic reconciler is the backstop).
        """
        try:
            await self._redis.publish(_CONTROL_CHANNEL, json.dumps({"run_id": str(run_id)}))
        except (RedisError, OSError) as exc:
            log.warning("failed to broadcast interrupt for run %s: %s", run_id, exc)

    async def start_control_listener(self) -> None:
        """Begin consuming _CONTROL_CHANNEL. Idempotent; no-op without Redis."""
        if self._control_task is None:
            self._control_task = asyncio.create_task(
                self._listen_control(), name="run-control-listener"
            )

    async def _listen_control(self) -> None:
        while True:
            pubsub = None
            try:
                pubsub = self._redis.pubsub()
                await pubsub.subscribe(_CONTROL_CHANNEL)
                while True:
                    msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30.0)
                    if msg is None or msg.get("type") != "message":
                        continue
                    self._handle_control(msg["data"])
            except asyncio.CancelledError:
                break
            except (RedisError, OSError) as exc:
                log.warning("run-control listener error, reconnecting in 5s: %s", exc)
                await asyncio.sleep(5)
            finally:
                if pubsub is not None:
                    with suppress(RedisError, OSError):
                        await pubsub.unsubscribe(_CONTROL_CHANNEL)
                        await pubsub.aclose()

    def _handle_control(self, data: Any) -> None:
        """Cancel the local task for a broadcast run id, if this pod owns it."""
        try:
            payload = json.loads(data)
            run_id = UUID(payload["run_id"])
        except ValueError, KeyError, TypeError:
            log.warning("run-control: ignoring malformed message: %r", data)
            return
        task = self._tasks.get(run_id)
        if task is not None:
            # The intent (pause vs cancel) lives in the already-committed run
            # row; the CancelledError handler reads it. We just stop the task.
            log.info("run-control: cancelling local task for run %s (cross-pod request)", run_id)
            task.cancel()

    def active_run_count(self) -> int:
        """Return the number of in-flight dispatch tasks.

        Used by the asgi lifespan drain to decide when the grace window can end
        early (every run finished naturally) versus when we have to fall through
        to ``shutdown()`` and force-cancel.
        """
        return len(self._tasks)

    async def shutdown(self) -> None:
        # Flip _draining BEFORE cancelling so every in-flight task's
        # CancelledError handler sees True and leaves its run row in
        # `running` for the next pod's startup reconciler to resume,
        # rather than marking `cancelled`. Explicit cancel() calls
        # (cancel endpoint, pause endpoint) don't go through here, so
        # they still drive the run to `cancelled`/`paused`.
        self._draining = True
        if self._control_task is not None:
            self._control_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._control_task
            self._control_task = None
        for task in list(self._tasks.values()):
            task.cancel()
        for task in list(self._tasks.values()):
            with suppress(asyncio.CancelledError, Exception):
                await task
        # Drain shielded cleanup LAST. Cancelling a run task above makes its
        # handler start a shielded terminal transition and then stop awaiting it;
        # returning here without draining lets the lifespan close the engine
        # underneath a half-committed cancel.
        for task in list(self._cleanup_tasks):
            with suppress(asyncio.CancelledError, Exception):
                await task

    async def lease_present(self, run_id: UUID) -> bool:
        """True if *any* instance currently holds this run's execution lease.

        Public delegate over ``RunLease.present`` — the asgi reconciler uses it
        to tell a live run from an orphan before re-dispatching.
        """
        return await self._lease.present(run_id)

    async def db_claim_fresh(self, run_id: UUID) -> bool:
        """True if the run carries a non-stale durable execution claim.

        Public delegate over ``RunDbClaim.fresh`` — the reconciler consults it
        AFTER the Redis lease check so a run executing lease-less (Redis was down
        at dispatch) under a fresh DB claim is recognised as alive and NOT
        re-dispatched once Redis recovers. Almost always False (claims exist only
        on the fail-open path); a Redis error inside falls closed (returns True →
        assume alive, skip) so a blip never triggers a double-dispatch.
        """
        try:
            return await self._claim.fresh(run_id)
        except Exception as exc:  # noqa: BLE001 -- fall closed: treat as alive, don't re-dispatch
            log.warning("db_claim_fresh check failed for %s (%s); assuming alive", run_id, exc)
            return True

    async def _run(self, run_id: UUID) -> None:
        """Run-scoped logging wrapper around :meth:`_run_owned`.

        Binds the run id as the log ``correlation_id`` for everything this run
        task does — lease/claim heartbeats, the DAG, and every stage task the
        executor spawns (``create_task`` copies the context). Without it, stage
        log lines could not be attributed to a run: an investigation grepping a
        pod log for a run id found only HTTP access lines, and could not tell
        whether a stage's ERROR guard had fired.

        Bound HERE, inside the per-run task ``dispatch`` creates, and never in
        ``dispatch`` itself: that runs in the caller's context (the HTTP handler
        or the reconciler loop), so a bind there would leak one run's id onto
        whatever that caller logs next — including the next run it dispatches.
        """
        with bound_correlation_id(str(run_id)):
            try:
                await self._run_owned(run_id)
            finally:
                # ``set_w2v_project_context`` returns no reset token; the bind in
                # ``_execute_run`` is already confined to this task's context
                # copy, and this clear keeps it from outliving the run if this
                # coroutine is ever awaited outside its own task.
                set_w2v_project_context(None)

    async def _run_owned(self, run_id: UUID) -> None:
        """Lease-guarded entry point: execute a run on at most one pod.

        Acquires the per-run lease, runs the DAG under a renewing heartbeat, and
        releases the lease on exit (including drain-cancel — which lets the next
        pod's reconciler resume the run cleanly). A run already owned by another
        live instance is skipped without touching its row.
        """
        acquired = await self._lease.acquire(run_id)
        if acquired is False:
            log.warning("run %s already executing on another instance; skipping", run_id)
            return
        own_lease = acquired is True
        # Redis fell open (acquire returned None → Redis unavailable). Without a
        # lease, every replica would proceed and double-run the run's
        # non-idempotent submit stages. Take the durable DB claim as the second
        # guard: exactly one pod wins, the rest skip. Only reached on the
        # fail-open path, so the normal Redis-lease path is unchanged.
        db_claimed = False
        if not own_lease:
            db_claimed = await self._claim.acquire(run_id)
            if not db_claimed:
                log.warning("run %s already DB-claimed by another instance; skipping", run_id)
                return
        # The halt callback both heartbeats share. Losing EITHER guard means a
        # second replica is now entitled to execute this same run, so the answer is
        # identical: stop the work. The row stays `running`, so whoever holds the
        # guard now owns the resume.
        executing = asyncio.current_task()

        def _halt() -> None:
            # Mark BEFORE cancelling (#955): `_finish_cancelled` reads this set to
            # tell an ownership-loss cancel from a user cancel, and it must see the
            # marker by the time the CancelledError it triggers is handled. This is
            # load-bearing for the LEASE path too, not just the DB claim: without
            # it, a lost Redis lease cancels the task, falls through to
            # `_finish_cancelled`'s final `else` ("explicit user cancel") and
            # terminally marks a perfectly healthy run `cancelled` — the exact
            # defect #955 fixed for the claim, reintroduced through the other guard.
            self._claim_lost_runs.add(run_id)
            if executing is not None:
                executing.cancel()

        # ``on_lost`` is not optional here, and the DB claim does NOT backstop it:
        # the claim is taken ONLY when ``acquire`` fell open (``if not own_lease``),
        # so a run that started under a HEALTHY lease has no claim at all. Without
        # this, a Redis outage longer than the 60s lease TTL let the key expire
        # under a still-executing pod; the reconciler then saw a leaseless orphan,
        # re-dispatched it onto a second pod, and both ran the run's non-idempotent
        # submit stages (render / TTS / upload, `retries=0`) — billed twice.
        heartbeat = self._lease.start_heartbeat(run_id, _halt) if own_lease else None
        # Losing the durable claim means another replica is now entitled to run
        # this same run. Stopping only the heartbeat (the old behaviour) left THIS
        # pod executing the DAG alongside it — every non-idempotent submit stage
        # (render / TTS / upload) then runs twice and is paid for twice. Halt the
        # executing task instead; the row stays `running`, so whoever holds the
        # claim now owns the resume.
        claim_heartbeat = self._claim.start_heartbeat(run_id, on_lost=_halt) if db_claimed else None
        # Gauge tracks runs actively executing on THIS pod. Increment whenever we
        # proceed to execute — under the Redis lease OR the durable DB claim; only
        # the "owned by another instance" / "already DB-claimed" skips above do not.
        active_runs.inc()
        try:
            await self._execute_run(run_id)
        finally:
            active_runs.dec()
            # Shielded: every await below is a cancellation point, and this
            # `finally` runs precisely when a cancel is in flight. See
            # `_release_ownership` for what a lost release costs.
            await self._shielded(
                self._release_ownership(
                    run_id,
                    heartbeat=heartbeat,
                    claim_heartbeat=claim_heartbeat,
                    own_lease=own_lease,
                    db_claimed=db_claimed,
                )
            )

    async def _shielded(self, coro: Awaitable[None]) -> None:
        """``await`` ``coro`` so a cancel of THIS task cannot interrupt it.

        Wraps rather than calling ``asyncio.shield`` directly so the inner task
        is also RETAINED (see ``_cleanup_tasks``) — shielding alone leaves it
        unreferenced, which is a different way to lose the same cleanup.
        """
        task = asyncio.ensure_future(coro)
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
        await asyncio.shield(task)

    async def _release_ownership(
        self,
        run_id: UUID,
        *,
        heartbeat: asyncio.Task | None,
        claim_heartbeat: asyncio.Task | None,
        own_lease: bool,
        db_claimed: bool,
    ) -> None:
        """Stop the heartbeats and hand back the lease + DB claim. Never raises.

        ONE coroutine so a SINGLE ``asyncio.shield`` covers every await —
        shielding them individually would leave the gaps between them
        cancellable, which is the same hole a few lines narrower
        (`runs/sub_dag.py::_release_pubsub` makes the same argument).

        What a lost release costs, and why this is not mere hygiene: the Redis
        lease survives its 60s TTL and `exec_claimed_at` stays fresh, so
        `lease_present` and the DB claim BOTH report "an owner is alive" for a
        run that has none. The reconciler then refuses to re-dispatch it, and the
        run sits `running` until a human notices.

        Two ways it used to be lost. (1) A second `cancel()` — a user cancel
        followed by pod SIGTERM lands one — aborted the bare awaits here.
        (2) `suppress(CancelledError)` around `await heartbeat` covers ONLY
        cancellation: a heartbeat task that had already died of, say, a
        `RedisError` re-raised it at the await and skipped BOTH releases below.
        `_FakeRedis` raising `AttributeError` from `raw.set` is a live example
        of a heartbeat dying non-cancelled.
        """
        for hb in (heartbeat, claim_heartbeat):
            if hb is None:
                continue
            hb.cancel()
            # Swallow whatever the heartbeat task carries — its cancellation, or
            # the error that already killed it. Neither is a reason to skip the
            # releases, which is what this method exists to guarantee.
            with suppress(asyncio.CancelledError, Exception):
                await hb
        if own_lease:
            await self._lease.release(run_id)
        if db_claimed:
            # ``RunClaim.release`` opens its own session; a DB blip there would
            # otherwise propagate out of the `finally` and replace the exception
            # the run actually failed with.
            try:
                await self._claim.release(run_id)
            except Exception as exc:  # noqa: BLE001 — teardown must not mask the cause
                log.warning("run %s: DB claim release failed (%s); will stale-expire", run_id, exc)

    async def _execute_run(self, run_id: UUID) -> None:  # noqa: C901
        # Cached at dispatch time so each lifecycle transition can mirror its
        # SSE onto the parent project's channel without re-querying. Both
        # remain None for top-level (non-child) projects — in which case
        # _publish_child_status is a no-op.
        parent_project_id: UUID | None = None
        child_project_id: UUID | None = None
        # Set True once an inline pause gate below has already published run_paused
        # for this dispatch, so the CancelledError pause branch does not re-publish
        # (+ re-increment runs_total) if the cancel lands inside the inline publish's
        # await. Survives into the except clause (no block scoping in Python).
        paused_published_locally = False
        try:
            async with self._sessions() as session:
                # Bootstrap the RLS GUC from the run's tenant BEFORE the first
                # query — under enforce + the restricted role, reading the run row
                # is itself tenant-scoped, so the dispatcher (which holds only a
                # run_id) resolves the tenant via the SECURITY DEFINER bypass.
                # No-op until enforce. Every later block re-binds the same way.
                await bind_run_tenant(session, run_id)
                run_repo = RunRepository(session)
                stage_repo = StageOutputRepository(session)
                run = await run_repo.get(run_id)
                if run is None:
                    log.error("dispatch: run %s missing", run_id)
                    return
                blueprint = self._registry.resolve_by_id(run.blueprint_id)
                config_snapshot = dict(run.config_snapshot)
                # The run's BILLING tenant, refused rather than defaulted. This
                # value is bound into the identity ContextVar below and
                # ``identity_event_hook`` stamps it verbatim as
                # ``X-Effective-Tenant-Id`` on every metered hop — i.e. on every
                # worker_management LLM call every stage in the DAG makes
                # (growth_bundle, promo_*, course_*, demo_video_*, …). There is no
                # per-call tenant argument to get wrong: ``WMClient.message()``
                # takes none, so THIS is the single point where the whole run's
                # spend is attributed, and the only point where a bad one can be
                # stopped.
                #
                # ``UUID(...) if ... else None`` used to accept two unattributable
                # values here. Measured against w2v-common 0.67.2's real hook (see
                # ``src/auth/run_tenant``): the nil uuid rides out as
                # ``X-Effective-Tenant-Id: 00000000-0000-0000-0000-000000000000``,
                # and an absent tenant sends no header at all, billing the shared
                # w2v service-account tenant. Neither raised, so neither was
                # visible until the cost event dead-lettered.
                try:
                    tenant_id = billable_tenant_id(config_snapshot.get("tenant_id"))
                except TenantError as exc:
                    await self._fail_unbillable(
                        session, run_repo, run_id, exc, deliverable_id=run.deliverable_id
                    )
                    return
                subject_id = (
                    UUID(config_snapshot["subject_id"])
                    if config_snapshot.get("subject_id")
                    else None
                )
                regenerate_notes = config_snapshot.get("regenerate_notes") or {}
                # Bind the three-piece identity for the whole run task. Dispatch
                # (fresh + reconciler-resumed orphans) runs as a background task
                # with no inbound request, so AuthMiddleware never populated the
                # identity ContextVars. Without this, every S2S call a stage
                # handler makes goes out with empty X-Tenant-Id/X-Subject-Id/
                # X-License-Id — downstream attribution + audit lose the subject.
                # ContextVars are task-local, so this scopes to this run only.
                set_identity_context(
                    # The VALIDATED uuid. Honest about what this line is NOT: it
                    # is not what enforces the refusal, and passing the raw
                    # ``config_snapshot["tenant_id"]`` here would be an
                    # EQUIVALENT mutant, not a defect. ``set_identity_context``
                    # re-runs whatever it is handed through ``parse_tenant_id``
                    # and stores a ``uuid.UUID`` (w2v-common 0.67.2,
                    # ``auth/identity_context.py`` — the ContextVar is
                    # ``ContextVar[uuid.UUID | None]``), so every spelling
                    # normalises to the same value; and any value that reaches
                    # this line already survived the guard above, which
                    # ``return``s on refusal. The guard is the whole enforcement
                    # — do not read this argument as a second one. Bound as the
                    # parsed uuid anyway so the type matches ``RunContext``
                    # below and a reader sees one canonical form.
                    tenant_id=tenant_id,
                    license_id=config_snapshot.get("license_id") or None,
                    subject_id=config_snapshot.get("subject_id") or None,
                    # FOURTH piece, added by w2v-common 0.67.0: without it every
                    # metered S2S hop (worker_management = every LLM call) raises
                    # UnvouchedTenantError, because a delegated tenant of
                    # unrecorded provenance may not be spent as.
                    tenant_source=tenant_source_from_snapshot(
                        config_snapshot.get(TENANT_SOURCE_KEY)
                    ),
                )
                bind_run_attribution(config_snapshot)
                # P3: the run's PINNED provider slots + model tiers for every stage.
                bind_run_provider_slots(config_snapshot)
                stage_rows = await stage_repo.list_for_run(run.id)
                # A run created before its blueprint gained a stage has no row for
                # that stage, and every mark_* writer is a bare UPDATE — so the
                # stage would stay invisible to the completion gate and the run
                # would sit 'running' forever. Seed the missing rows here, before
                # prior_run_outputs and the pause gate read them. Rides the
                # tenant-bound session and the commit below.
                stage_rows = await reconcile_blueprint_stage_rows(
                    stage_repo, run_id=run.id, blueprint=blueprint, stage_rows=stage_rows
                )
                ctx = RunContext(
                    run_id=run.id,
                    deliverable_id=run.deliverable_id,
                    config_snapshot=config_snapshot,
                    prior_run_outputs={
                        # A 'skipped' stage is terminal-SUCCESS (see
                        # run_finalizer.TERMINAL_STAGE_STATUSES) but is written with
                        # NO output — seed it into the completed-set with an empty
                        # dict so resume does not needlessly RE-RUN it. 'completed'
                        # still requires its real output; only skips carry {}.
                        so.stage_name: (
                            dict(so.output) if (so.status == "completed" and so.output) else {}
                        )
                        for so in stage_rows
                        if (so.status == "completed" and so.output) or so.status == "skipped"
                    },
                    open_annotations_by_stage={
                        str(stage_name): list(notes)
                        for stage_name, notes in regenerate_notes.items()
                    },
                    tenant_id=tenant_id,
                    subject_id=subject_id,
                    correlation_id=str(run.id),
                    # Durable per-sub-step results, so a stage re-entered after a
                    # pod death reuses its finished sub-steps (ctx.step_ledger).
                    step_ledger_factory=step_ledger_factory(
                        self._sessions, run_id=run.id, tenant_id=tenant_id
                    ),
                )
                proj_row = (
                    await session.execute(
                        select(Project.id, Project.parent_project_id)
                        .join(Deliverable, Deliverable.project_id == Project.id)
                        .where(Deliverable.id == run.deliverable_id)
                    )
                ).first()
                if proj_row is not None:
                    child_project_id = proj_row.id
                    parent_project_id = proj_row.parent_project_id
                # Attribute every worker_management call this run makes to the
                # run's OWN w2v project (orchestrator ``projects.id``): WMClient
                # sends it as ``X-Project-Id`` and the identity hook forwards it as
                # ``X-W2V-Project-Id`` to downstream w2v hops. Bound AFTER
                # ``set_identity_context`` (which clears it on a tenant change).
                # Never an fr project — those arrive on inbound X-Project-Id and
                # are a separate concept. ``None`` (no project row) clears it, so
                # WMClient falls back to the tenant-level project and counts it.
                # Task-scoped: this runs inside the per-run task ``dispatch``
                # creates, and ``_run`` clears it on exit.
                set_w2v_project_context(child_project_id)
                # Crash-safe pause gate. If a checkpoint was already consumed (a
                # completed pause-stage) but the DAG is not complete, this
                # dispatch is a raced-orphan re-dispatch of an about-to-pause run
                # — drained after the checkpoint committed but before mark_paused,
                # so the row is still `running` and the reconciler picked it up.
                # It is NOT an approved resume (the resume route strips consumed
                # checkpoints from pause_after_stages). Re-pause instead of
                # running past the human-review gate — otherwise the executor
                # seeds the completed checkpoint, never re-fires the pause, drains
                # downstream, and DAG_COMPLETE silently completes the run without
                # review. decide_run_terminal_state returns PAUSED in exactly this
                # case; DAG_COMPLETE (Part-1 backstop) / CONTINUE fall through.
                dispatch_pause_set = set(run.pause_after_stages or [])
                if (
                    decide_run_terminal_state(blueprint, stage_rows, dispatch_pause_set)
                    is RunDecision.PAUSED
                ):
                    # Name the gate the human must review (see the completion
                    # gate below for why current_stage cannot be trusted here).
                    paused_stage = pause_trigger_stage(blueprint, stage_rows, dispatch_pause_set)
                    if paused_stage is not None:
                        await run_repo.mark_current_stage(run_id, paused_stage)
                    transitioned = await run_repo.mark_paused(run_id, reason="checkpoint")
                    await self._recompute_project(session, child_project_id, parent_project_id)
                    await session.commit()
                    if transitioned:
                        paused_published_locally = True
                        await self._publish(run_id, "run_paused", stage_name=paused_stage)
                        await self._publish_child_status(
                            parent_project_id, child_project_id, "paused"
                        )
                    return
                # Capture the deadline anchor while the row is still attached to
                # this session. ``mark_started`` COALESCEs, so a run that already
                # started keeps its original stamp and a fresh one starts now —
                # reproduced here rather than re-read, because the UPDATE below
                # does not refresh the in-memory instance.
                run_started_at = run.started_at or datetime.now(UTC)
                # Same reason as started_at: read it while attached. It was
                # folded in by mark_resumed before this dispatch loaded the row.
                run_paused_seconds = int(getattr(run, "paused_seconds", 0) or 0)
                if not await run_repo.mark_started(run_id):
                    # The row left ("pending", "running") between the reconciler's
                    # orphan SELECT and here — a user cancel, or a pause. That
                    # decision WINS: executing anyway would resurrect a cancelled
                    # run and bill it to completion, or run a paused run past its
                    # human-review gate. Nothing to commit; drop the dispatch.
                    log.warning(
                        "dispatch: run %s is no longer startable (cancelled/paused "
                        "under us); abandoning this dispatch",
                        run_id,
                    )
                    return
                await self._recompute_project(session, child_project_id, parent_project_id)
                await session.commit()
                await self._publish(run_id, "run_started")
                await self._publish_child_status(parent_project_id, child_project_id, "running")

            async def emit(event: str, **kw: Any) -> None:
                await self._publish(run_id, event, **kw)

            # Stages this dispatch recorded 'failed', in the order they failed.
            # ``current_stage`` is written only at stage START, so after a
            # concurrent wave it names the last sibling to START — not the one
            # that failed. The executor persists 'failed' immediately before it
            # re-raises, and gather propagates the first raise, so the FIRST
            # entry is the stage whose exception reaches the except path below.
            failed_stages: list[str] = []

            async def persist(
                stage_name: str,
                status: str,
                output: dict | None = None,
                error: str | None = None,
            ) -> None:
                async with self._sessions() as s:
                    await bind_run_tenant(s, run_id)
                    sor = StageOutputRepository(s)
                    rr = RunRepository(s)
                    if status == "running":
                        await sor.mark_running(run_id, stage_name)
                        await rr.mark_current_stage(run_id, stage_name)
                    elif status == "completed":
                        # A False here means the stage_outputs row was absent, so
                        # the computed output was silently dropped. Signal only —
                        # failing the run on a bookkeeping gap would be worse than
                        # the gap itself; the reconciler owns the repair.
                        if not await sor.mark_completed(run_id, stage_name, output or {}):
                            log.warning(
                                "persist(completed): stage_outputs row unexpectedly absent — "
                                "run_id=%s stage_name=%s; the stage output was NOT persisted",
                                run_id,
                                stage_name,
                            )
                    elif status == "failed":
                        failed_stages.append(stage_name)
                        # A False here means the row was absent or already
                        # terminal (write-once guard). Signal only — the
                        # reconciler owns repair; failing the run on a
                        # bookkeeping gap would be worse than the gap.
                        if not await sor.mark_failed(run_id, stage_name, error or "unknown"):
                            log.warning(
                                "persist(failed): stage_outputs row absent or already "
                                "terminal — run_id=%s stage_name=%s; the 'failed' write "
                                "was a no-op",
                                run_id,
                                stage_name,
                            )
                    elif status == "skipped":
                        # Opt-in stage whose precondition was absent — terminal,
                        # not a failure. Counts toward run completion below.
                        if not await sor.mark_skipped(run_id, stage_name, error or ""):
                            log.warning(
                                "persist(skipped): stage_outputs row unexpectedly absent — "
                                "run_id=%s stage_name=%s; the terminal 'skipped' was NOT "
                                "persisted and the completion gate will not count it",
                                run_id,
                                stage_name,
                            )
                    elif status == "pending":
                        # Stage cancelled mid-flight — reset so a resumed run
                        # re-executes it instead of seeing a stuck 'running' row.
                        await sor.mark_pending(run_id, stage_name)
                    await s.commit()

            async def bump_attempts(stage_name: str) -> int:
                # Durably increment the stage's timeout-resume counter (P1). Its own
                # short-lived session + commit, mirroring ``persist``.
                async with self._sessions() as s:
                    await bind_run_tenant(s, run_id)
                    attempts = await StageOutputRepository(s).increment_attempts(run_id, stage_name)
                    await s.commit()
                    return attempts

            async def note_interrupted(stage_name: str) -> int:
                # Durably record a DRAIN/CANCEL reset. Its own short-lived session +
                # commit, mirroring ``bump_attempts`` — deliberately NOT the caller's
                # session, because this runs while the stage task is unwinding under
                # cancellation and must not ride on a session that is being torn down.
                async with self._sessions() as s:
                    await bind_run_tenant(s, run_id)
                    count = await StageOutputRepository(s).increment_interruptions(
                        run_id, stage_name
                    )
                    await s.commit()
                    return count

            async def record_config_reads(stage_name: str, reads: list[str]) -> None:
                # What the stage read from the snapshot (restart_plan.py). Own short
                # session, like the two hooks above; the executor swallows a failure
                # here (NULL reads = "unknown", the conservative answer).
                async with self._sessions() as s:
                    await bind_run_tenant(s, run_id)
                    await StageOutputRepository(s).set_config_reads(run_id, stage_name, reads)
                    await s.commit()

            executor = DAGExecutor(
                blueprint,
                emit=emit,
                persist=persist,
                handler_deps={
                    "run_engine": self._run_engine,
                    "redis": self._redis,
                    "session_factory": self._sessions,
                },
                bump_attempts=bump_attempts,
                note_interrupted=note_interrupted,
                max_stage_attempts=get_settings().pipeline.max_stage_attempts,
                stage_base_ceiling_seconds=get_settings().pipeline.llm_stage_base_ceiling_s,
                run_deadline_at=_run_deadline_at(run_started_at, run_paused_seconds),
                record_config_reads=record_config_reads,
            )
            try:
                await executor.run(ctx, pause_after_stages=set(run.pause_after_stages or []))
            except StageResumableTimeoutError as exc:
                # A stage timed out but is under its resume budget. Do NOT terminal-
                # fail: leave the run row in 'running' (the lease is released in
                # ``_run``'s finally) so the orphan reconciler re-dispatches it and it
                # resumes from the last completed stage — the same "left running for
                # resume" contract as the drain path below. No terminal publish; the
                # FE's SSE/poll resyncs and sees 'running' (the timed-out stage was
                # reset to 'pending' by the executor).
                log.warning(
                    "run %s: %s — left 'running' for reconciler resume",
                    run_id,
                    exc,
                )
                return
            except Exception as exc:
                log.exception("run %s failed during DAG execution", run_id)
                async with self._sessions() as session:
                    await bind_run_tenant(session, run_id)
                    # The ONLY run-failure site with the exception object in
                    # hand, so the only one that can attribute anything richer
                    # than 'unknown'. ``attribute_exception`` is an allowlist over
                    # the cause chain (an upstream-stamped class, or a Python
                    # programming-error type); anything it does not recognise
                    # comes back 'unknown' rather than being forced into a bucket.
                    transitioned = await RunRepository(session).mark_failed(
                        run_id,
                        describe_exception(exc),
                        failure_class=attribute_exception(exc),
                        failed_stage=failed_stages[0] if failed_stages else None,
                        failure_detail=failure_detail_for(exc),
                    )
                    if transitioned:
                        # Only sweep on a real transition to 'failed'. mark_failed is
                        # write-once-guarded, so a losing racer gets transitioned=False;
                        # sweeping then could clobber stage rows a concurrent worker is
                        # legitimately resuming. On the winning transition the run is
                        # terminal and never resumed, so its leftover 'pending'/'running'
                        # stage rows (independent siblings / never-scheduled downstream)
                        # would strand forever — terminalize them to 'cancelled'.
                        swept = await StageOutputRepository(session).bulk_cancel_non_terminal(
                            run_id
                        )
                        if swept:
                            log.info(
                                "run %s failed: swept %d stranded stage row(s) to 'cancelled'",
                                run_id,
                                swept,
                            )
                        # Cancel any PAID sub-DAG child renders (course lessons)
                        # this run spawned — they carry parent_run_id=None so the
                        # FK descendant cascade cannot see them, and would keep
                        # spending until they finish on their own.
                        await self._cancel_subdag_children(session, run_id)
                    await self._recompute_project(session, child_project_id, parent_project_id)
                    await session.commit()
                if transitioned:
                    await self._publish(run_id, "run_failed", error=describe_exception(exc))
                    await self._publish_child_status(parent_project_id, child_project_id, "failed")
                return

            async with self._sessions() as session:
                await bind_run_tenant(session, run_id)
                rr = RunRepository(session)
                run = await rr.get(run_id)
                if run is None:
                    log.error("dispatch: run %s vanished before completion", run_id)
                    return
                pause_set = set(run.pause_after_stages or [])
                stage_rows = await StageOutputRepository(session).list_for_run(run_id)
                # Pure verdict: PAUSED (a pause gate hit), DAG_COMPLETE (every
                # stage terminal — 'skipped' counts), or CONTINUE (leave running).
                decision = decide_run_terminal_state(blueprint, stage_rows, pause_set)
                if decision is RunDecision.PAUSED:
                    # Re-point current_stage at the checkpoint that tripped the
                    # gate. Until the pause it tracks "the stage that started
                    # most recently", which after a concurrent wave can be a
                    # sibling that SKIPPED — the FE builds the pause banner and
                    # the stage-editor href from this field, so leaving it there
                    # sends the reviewer to a stage that never ran.
                    paused_stage = pause_trigger_stage(blueprint, stage_rows, pause_set)
                    if paused_stage is not None:
                        await rr.mark_current_stage(run_id, paused_stage)
                    transitioned = await rr.mark_paused(run_id, reason="checkpoint")
                    await self._recompute_project(session, child_project_id, parent_project_id)
                    await session.commit()
                    if transitioned:
                        paused_published_locally = True
                        await self._publish(run_id, "run_paused", stage_name=paused_stage)
                        await self._publish_child_status(
                            parent_project_id, child_project_id, "paused"
                        )
                    return
                if decision is RunDecision.DAG_COMPLETE:
                    # The complete-vs-fail split needs a storage probe, so it
                    # stays here (I/O) rather than in the pure decision.
                    artifact_problem = await verify_terminal_artifact(blueprint, stage_rows)
                    if artifact_problem is not None:
                        # The DAG finished but the deliverable artifact is not
                        # actually in storage — fail loud instead of a green lie.
                        # OUR_DEFECT, unambiguously: every stage reported terminal-
                        # good and the deliverable is still not in storage. No
                        # customer input and no upstream status can make "we said we
                        # finished and produced nothing" anyone else's fault.
                        transitioned = await rr.mark_failed(
                            run_id, artifact_problem, failure_class=OUR_DEFECT
                        )
                        await self._recompute_project(session, child_project_id, parent_project_id)
                        await session.commit()
                        if transitioned:
                            await self._publish(run_id, "run_failed", error=artifact_problem)
                            await self._publish_child_status(
                                parent_project_id, child_project_id, "failed"
                            )
                        return
                    transitioned = await rr.mark_completed(run_id)
                    await self._recompute_project(session, child_project_id, parent_project_id)
                    await session.commit()
                    if transitioned:
                        await self._publish(run_id, "run_completed")
                        await self._publish_child_status(
                            parent_project_id, child_project_id, "completed"
                        )
                        # PR-C: only the CAS winner (transitioned) reaches here, so the
                        # bounded structural-relief regenerate is spawned at most once per
                        # completion. The hook never raises — it cannot un-complete the run.
                        if self._run_engine is not None:
                            await maybe_spawn_structural_relief_run(
                                run_engine=self._run_engine,
                                parent_run_id=run_id,
                                deliverable_id=run.deliverable_id,
                                config_snapshot=run.config_snapshot,
                                stage_rows=stage_rows,
                            )
                    return
                if decision is RunDecision.FAILED:
                    # A terminal-BAD stage row (failed/cancelled) exists but the
                    # executor returned normally — the resumable-timeout drain let
                    # an independent sibling persist 'failed' on a transient error
                    # (the timeout path does NOT cancel siblings), and its later
                    # success write is a guarded no-op so the row stays 'failed'.
                    # Such a run can never reach DAG_COMPLETE; fail it loud + sweep
                    # the leftover non-terminal rows, mirroring the except-Exception
                    # path, instead of stranding it for the ~2h max-age backstop.
                    topo = set(blueprint.topological_order())
                    failed_row = next(
                        (
                            r
                            for r in stage_rows
                            if r.status in FAILED_STAGE_STATUSES and r.stage_name in topo
                        ),
                        None,
                    )
                    reason = json.dumps(
                        {
                            "reason": "stage_failed_post_drain",
                            "stage": getattr(failed_row, "stage_name", None),
                            "error": getattr(failed_row, "error", None),
                        }
                    )
                    # No ``failure_class``: deliberately left at ``mark_failed``'s
                    # 'unknown' default. All this site has is the failed stage
                    # row's own free-text ``error`` — the exception, and with it
                    # every structural signal, was consumed inside the drained
                    # executor. Prose is exactly what this column exists to
                    # replace, so re-deriving an attribution from it here would
                    # reintroduce the bug. Attributing this route properly needs
                    # the stage row to carry its own class; that is a follow-up,
                    # not a guess to make now.
                    transitioned = await rr.mark_failed(
                        run_id, reason, failed_stage=getattr(failed_row, "stage_name", None)
                    )
                    if transitioned:
                        swept = await StageOutputRepository(session).bulk_cancel_non_terminal(
                            run_id
                        )
                        if swept:
                            log.info(
                                "run %s failed (post-drain stage failure): swept %d "
                                "stranded stage row(s) to 'cancelled'",
                                run_id,
                                swept,
                            )
                        # Reap orphaned PAID sub-DAG child renders (see the
                        # except-Exception path) — the same leak on this failure route.
                        await self._cancel_subdag_children(session, run_id)
                    await self._recompute_project(session, child_project_id, parent_project_id)
                    await session.commit()
                    if transitioned:
                        await self._publish(run_id, "run_failed", error=reason)
                        await self._publish_child_status(
                            parent_project_id, child_project_id, "failed"
                        )
                    return
                # decision is CONTINUE. The executor returns only on a
                # completed-pause gate or a fully-drained DAG, so a naturally
                # returning executor can never land here — CONTINUE means the
                # executor exited with non-terminal stages and no real pause
                # (the old skipped-pause-stage wedge). Leaving the row in
                # `running` silently strands it (the orphan reconciler then
                # re-dispatches it forever). Log loudly so any regression is
                # visible instead of a silent infinite loop; the reconciler's
                # max-age backstop will eventually mark it failed.
                log.error(
                    "run %s returned CONTINUE: executor exited with work remaining and "
                    "no completed pause-gate (invariant breach) — left running for the "
                    "reconciler backstop",
                    run_id,
                )
        except asyncio.CancelledError:
            # Shielded: this whole handler is the ONLY place a user-cancelled run
            # is transitioned out of `running`, and every await in it is a
            # cancellation point. A SECOND cancel — `POST /runs/{id}/cancel`
            # followed by pod SIGTERM (`shutdown()` cancels the same task) — used
            # to abort it mid-body, so `mark_cancelled` never committed, the row
            # stayed `running`, and the reconciler re-dispatched the run the user
            # had just cancelled. It then resumed and kept spending.
            await self._shielded(
                self._finish_cancelled(
                    run_id,
                    parent_project_id=parent_project_id,
                    child_project_id=child_project_id,
                    paused_published_locally=paused_published_locally,
                )
            )
            raise

    async def _fail_unbillable(
        self,
        # Typed, not ``Any``: this method is new, and ``tests/test_any_ratchet.py``
        # is a one-way budget (43) that the surrounding legacy ``session: Any``
        # signatures already spend in full.
        session: AsyncSession,
        run_repo: RunRepository,
        run_id: UUID,
        exc: TenantError,
        *,
        deliverable_id: UUID,
    ) -> None:
        """Terminal-fail a run whose snapshot carries no billable tenant.

        Deliberately NOT a bare ``raise``. An exception escaping ``_execute_run``
        leaves the row in ``running``, and the orphan reconciler re-dispatches a
        ``running`` row — so raising would spin this run forever against a
        condition that can never resolve on its own. ``mark_failed`` is the state
        that says "a human must fix the project's tenant and start a new run",
        which is the only recovery there is (the provenance ladder deliberately
        offers no way to invent one; see ``src/auth/tenant_provenance``).

        Runs BEFORE the DAG starts, so there are no stage rows to sweep and no
        sub-DAG children to reap — the two things the post-execution failure path
        below exists to clean up. Nothing has been spent yet either, which is the
        whole point of refusing here rather than at the first LLM call.

        It must still do the OTHER half of every terminal transition in this
        file: ``_recompute_project`` (before the commit, so project status lands
        atomically with the run status) then the gated ``_publish_child_status``.
        Skipping it was worse than the billing defect it replaced: the refusal
        fires on exactly the legacy nil-uuid rows this guard targets, and
        ``recompute_status`` has only two callers estate-wide (here and
        ``api/routes/runs/_shared.py``), so nothing else would ever re-derive the
        project — the owning project (and its course parent) would sit at
        ``producing`` forever with no run left to move them.

        The project ids are resolved HERE rather than passed in: the refusal
        happens before ``_execute_run``'s own project lookup, so its
        ``child_project_id``/``parent_project_id`` are both still ``None`` at
        this point. Same query and same ``first()``-may-be-``None`` handling as
        ``finalize_if_dag_complete``.
        """
        log.error("run %s refused at dispatch: %s", run_id, exc)
        proj_row = (
            await session.execute(
                select(Project.id, Project.parent_project_id)
                .join(Deliverable, Deliverable.project_id == Project.id)
                .where(Deliverable.id == deliverable_id)
            )
        ).first()
        child_project_id = proj_row.id if proj_row is not None else None
        parent_project_id = proj_row.parent_project_id if proj_row is not None else None
        reason = json.dumps({"reason": "unbillable_tenant", "error": describe_exception(exc)})
        # OUR_DEFECT: this run was accepted, persisted and dispatched by our own
        # API, and is refused only HERE, at spend time, because our provenance
        # ladder cannot produce a billable tenant for a snapshot we ourselves
        # wrote. Accepting a run we cannot bill is the defect; this refusal is
        # the symptom of it.
        transitioned = await run_repo.mark_failed(run_id, reason, failure_class=OUR_DEFECT)
        await self._recompute_project(session, child_project_id, parent_project_id)
        await session.commit()
        if transitioned:
            await self._publish(run_id, "run_failed", error=reason)
            await self._publish_child_status(parent_project_id, child_project_id, "failed")

    async def _finish_cancelled(
        self,
        run_id: UUID,
        *,
        parent_project_id: UUID | None,
        child_project_id: UUID | None,
        paused_published_locally: bool,
    ) -> None:
        """Transition + publish for a run whose task was cancelled.

        Extracted from the `except CancelledError` handler purely so one
        ``asyncio.shield`` can cover all of it — see the call site.
        """
        # Snapshot-and-clear via `finally` below: this dispatcher instance is
        # long-lived and shared across runs, so the marker must not survive
        # past this one handling, or a LATER genuine user cancel of the same
        # run id would be silently swallowed as a claim loss too.
        claim_lost = run_id in self._claim_lost_runs
        try:
            await self._finish_cancelled_body(
                run_id,
                parent_project_id=parent_project_id,
                child_project_id=child_project_id,
                paused_published_locally=paused_published_locally,
                claim_lost=claim_lost,
            )
        finally:
            self._claim_lost_runs.discard(run_id)

    async def _finish_cancelled_body(
        self,
        run_id: UUID,
        *,
        parent_project_id: UUID | None,
        child_project_id: UUID | None,
        paused_published_locally: bool,
        claim_lost: bool,
    ) -> None:
        async with self._sessions() as s:
            await bind_run_tenant(s, run_id)
            rr = RunRepository(s)
            current = await rr.get(run_id)
            # ``getattr`` for ``status`` so unit-test stubs that omit the
            # field don't trip an AttributeError and replace the
            # CancelledError mid-handler.
            if current is not None and getattr(current, "status", None) == "paused":
                # The pause endpoint (routes/runs/lifecycle.py::pause_run)
                # ALREADY transitioned the row running→paused (compare-and-set,
                # RunRepository.mark_paused) BEFORE cancelling this task — that
                # pre-mark is the intent signal this branch reads to tell a
                # user-pause from a user-cancel. So the transition has already
                # happened elsewhere; this branch is the sole *live* publisher
                # of ``run_paused`` for a user-initiated pause and must re-emit
                # it for FE resync. A compare-and-set gate here is degenerate
                # (the row is already `paused`, so mark_paused would return
                # False on EVERY user-pause and silently drop the event). The
                # real double-publish/double-count guard for pause lives at the
                # two inline pause gates above, which call mark_paused and
                # publish only on a real (rowcount>0) transition — so a raced
                # reconciler re-dispatch of an already-paused row no longer
                # re-publishes. See run_repository.mark_paused (now -> bool).
                #
                # ...unless THIS dispatch already published run_paused inline (an
                # inline pause gate ran, then the run was cancelled inside/after
                # that publish's await): _publish already incremented
                # runs_total{outcome=paused} once, so re-publishing here would
                # double-count + double-emit. paused_published_locally closes that
                # window; the FE still resyncs via runs.get() on reconnect.
                if not paused_published_locally:
                    await self._publish(run_id, "run_paused")
                    await self._publish_child_status(parent_project_id, child_project_id, "paused")
            elif self._draining:
                # Drain-cancel (pod going down). Leave the run row in
                # ``running`` so the next pod's startup reconciler
                # re-dispatches and resumes from the last completed stage.
                # No terminal publish — the FE's SSE reconnect to
                # ``runs:<run_id>`` will resync via runs.get() and see
                # ``running`` (same contract as a brief network blip).
                # Stage-output rehydration in _run handles the actual resume.
                log.info(
                    "run %s left in 'running' for resume (dispatcher draining)",
                    run_id,
                )
            elif claim_lost:
                # An ownership guard was lost — the durable DB claim (heartbeat
                # failures for a full TTL, or rowcount==0 because another
                # instance reclaimed the row) or the Redis lease (same two
                # shapes) — and the shared `_halt` callback cancelled this task
                # to halt execution —
                # see the comment at the `_run` call site. This is NOT a user
                # cancel: no other branch above matched (not a pre-marked
                # pause, not a drain), so without this branch it would fall
                # into the `else` below and terminally mark a healthy run
                # `cancelled`. Same handling as drain: leave the row
                # `running`, no terminal publish — whoever holds the claim
                # now (possibly this same pod on a future reconcile, possibly
                # another instance) owns the resume.
                log.warning(
                    "run %s ownership guard lost (not a user cancel) — left in 'running' "
                    "for the new claim holder to resume; no publish",
                    run_id,
                )
            else:
                # Explicit user cancel (POST /runs/{id}/cancel routed
                # through dispatcher.cancel()); _draining is False. The
                # transition happens HERE (co-located with the publish, same
                # shape as the completed/failed gates): mark_cancelled is a
                # compare-and-set that only flips a non-terminal row. Gate the
                # terminal SSE + implicit runs_total{cancelled} increment on
                # its bool so a second racer (e.g. a cross-pod cancel that
                # already committed `cancelled`, or a re-entry) whose write
                # transitions 0 rows does NOT double-publish/double-count.
                transitioned = await rr.mark_cancelled(run_id)
                await self._recompute_project(s, child_project_id, parent_project_id)
                await s.commit()
                if transitioned:
                    await self._publish(run_id, "run_cancelled")
                    await self._publish_child_status(
                        parent_project_id, child_project_id, "cancelled"
                    )

    async def finalize_if_dag_complete(self, run_id: UUID) -> str | None:
        """Backstop the reconciler calls for a stranded-but-finished run.

        If every stage is terminal ('skipped' counts) but the run row is still
        ``running``, the owning dispatch task was interrupted *after* the last
        stage persisted terminal (``dag_executor`` ``persist('completed')``) but
        *before* the completion gate committed ``mark_completed`` — so nothing
        ever flipped the run-level status. Recover it HERE, using the exact same
        primitives as the inline gate (``decide_run_terminal_state`` →
        ``verify_terminal_artifact`` → ``mark_completed``/``mark_failed``), so a
        DAG-complete run finalizes within one reconciler tick instead of waiting
        out the lease TTL or the 2h max-age backstop.

        Called *regardless of any still-present execution lease*: a DAG-complete
        run has no pending work, so the lease is meaningless, and
        ``mark_completed``/``mark_failed`` are write-once-guarded — racing the
        inline gate (or a doomed re-execution) is a benign no-op. Returns
        ``'completed'`` | ``'failed'`` if it finalized, else ``None`` (the run is
        not yet DAG-terminal — leave it to the normal reconciler flow).
        """
        async with self._sessions() as session:
            await bind_run_tenant(session, run_id)
            rr = RunRepository(session)
            run = await rr.get(run_id)
            # Only a run still stuck in 'running' is a candidate — a row already
            # terminal (the inline gate won the race) or paused must be left alone.
            if run is None or getattr(run, "status", None) != "running":
                return None
            blueprint = self._registry.resolve_by_id(run.blueprint_id)
            stage_rows = await StageOutputRepository(session).list_for_run(run_id)
            pause_set = set(run.pause_after_stages or [])
            if decide_run_terminal_state(blueprint, stage_rows, pause_set) is not (
                RunDecision.DAG_COMPLETE
            ):
                return None

            proj_row = (
                await session.execute(
                    select(Project.id, Project.parent_project_id)
                    .join(Deliverable, Deliverable.project_id == Project.id)
                    .where(Deliverable.id == run.deliverable_id)
                )
            ).first()
            child_project_id = proj_row.id if proj_row is not None else None
            parent_project_id = proj_row.parent_project_id if proj_row is not None else None

            # Bind the run's identity BEFORE anything in this method makes an
            # outbound call. ``bind_run_tenant`` above is the DB GUC only — it
            # never touches the identity ContextVars, and this task has none
            # (the reconciler has no inbound request, so AuthMiddleware never
            # populated them). Two consumers below need it, and the FIRST is the
            # terminal-artifact probe, not the relief spawn:
            #
            # * ``verify_terminal_artifact``'s ``design_job`` probe calls
            #   html_render, which w2v-common 0.73.0 moved into the
            #   metered/delegating destination set. ``DesignClient`` bakes
            #   ``X-Tenant-Id: Bearer <SA token>``, so with no tenant bound
            #   ``_apply_outbound_tenant`` takes the ``service_credentialed``
            #   early return and sends NO ``X-Effective-Tenant-Id``. html_render
            #   then filters by the SA tenant, cannot see a redesign_job created
            #   under the RUN's tenant, and 404s — which ``design_job_exists``
            #   maps to ``False``, i.e. a verdict, i.e. ``mark_failed``/
            #   OUR_DEFECT on a run whose design EXISTS. The inline gate never
            #   had this hole because it binds at run-task start (~:648).
            # * the structural-relief spawn, which inherits license_id/subject_id
            #   from the ContextVars (FIX B, unchanged — only hoisted).
            #
            # Cannot turn a working finalize into an exception. The only raiser
            # is ``billable_tenant_id``, caught here; ``set_identity_context``
            # takes an already-parsed uuid, and an unvouched provenance raises
            # only at the outbound hop — inside ``design_job_exists``, whose bare
            # except maps it to ``None`` (INDETERMINATE, never a verdict).
            _snap = run.config_snapshot or {}
            try:
                _run_tenant = billable_tenant_id(_snap.get("tenant_id"))
            except TenantError as exc:
                # No honest identity to bind. The identity-clearing ``finally``
                # below (not this branch) is what guarantees we don't leave
                # whatever the PREVIOUS run finalized by this long-lived
                # reconciler task bound — an unattributed probe is
                # indeterminate, a wrong-tenant one lies.
                log.error(
                    "run %s: reconciler-finalize could not bind identity — snapshot has no "
                    "billable tenant (%s); the terminal-artifact probe goes out "
                    "undelegated and the structural-relief spawn is skipped",
                    run_id,
                    exc,
                )
                _run_tenant = None
            if _run_tenant is not None:
                set_identity_context(
                    tenant_id=_run_tenant,
                    license_id=_snap.get("license_id") or None,
                    subject_id=_snap.get("subject_id") or None,
                    # FOURTH piece: absent for a pre-0.67 row → ``None`` →
                    # the delegated hop refuses instead of delegating an
                    # unvouched tenant. Degrades to INDETERMINATE, not to a
                    # false "absent".
                    tenant_source=tenant_source_from_snapshot(_snap.get(TENANT_SOURCE_KEY)),
                )
                bind_run_attribution(_snap)
                bind_run_provider_slots(_snap)

            # Everything below this point may run under the identity just
            # bound above (or may run with nothing bound, if ``_run_tenant``
            # is ``None``). Either way, this is a long-lived reconciler task
            # that will call this method again for the NEXT run — a ``return``
            # (or any future early-exit added inside this block) must not
            # leave this run's tenant bound for that next call to inherit.
            # ``try/finally`` guarantees the clear on every exit path
            # (``mark_failed``'s return, ``mark_completed``'s return, and any
            # exception), which a clear-before-each-``return`` cannot: a
            # ``return`` added later inside this block would silently skip it.
            try:
                # Same complete-vs-fail split as the inline gate: even a fully
                # terminal DAG must have its deliverable actually in storage.
                artifact_problem = await verify_terminal_artifact(blueprint, stage_rows)
                if artifact_problem is not None:
                    # OUR_DEFECT, unambiguously: every stage reported terminal-
                    # good and the deliverable is still not in storage. No
                    # customer input and no upstream status can make "we said we
                    # finished and produced nothing" anyone else's fault.
                    transitioned = await rr.mark_failed(
                        run_id, artifact_problem, failure_class=OUR_DEFECT
                    )
                    await self._recompute_project(session, child_project_id, parent_project_id)
                    await session.commit()
                    if transitioned:
                        await self._publish(run_id, "run_failed", error=artifact_problem)
                        await self._publish_child_status(
                            parent_project_id, child_project_id, "failed"
                        )
                    log.warning(
                        "reconciler-finalize: run %s DAG-complete but artifact missing → failed",
                        run_id,
                    )
                    return "failed"

                transitioned = await rr.mark_completed(run_id)
                await self._recompute_project(session, child_project_id, parent_project_id)
                await session.commit()
                if transitioned:
                    await self._publish(run_id, "run_completed")
                    await self._publish_child_status(
                        parent_project_id, child_project_id, "completed"
                    )
                    # PR-C: same structural-relief spawn as the inline gate. Only the CAS
                    # winner reaches here; the ``uq_runs_parent_relief_pass`` unique index is
                    # the backstop against this path racing the inline gate. Never raises.
                    #
                    # FIX B, now satisfied by the hoisted bind above: without the
                    # parent's identity on the ContextVars the spawned child's
                    # start_run would build a snapshot with an EMPTY license_id
                    # (get_license_id_from_context()=="") and every paid S2S call (bdd /
                    # record / compile / scrape) would go out with no X-License-Id → license-
                    # authz 402/403 or lost attribution.
                    #
                    # The spawn stays gated on the SAME refusal as before: the child
                    # inherits this snapshot, so an unbillable parent tenant would spawn
                    # a run whose every LLM call is delegated as the nil uuid (or, when
                    # absent, billed to the w2v service account). Skipping the spawn
                    # loses only the structural-relief re-render — a discretionary
                    # backstop — where spawning it would spend real money nobody can be
                    # invoiced for.
                    if self._run_engine is not None and _run_tenant is not None:
                        await maybe_spawn_structural_relief_run(
                            run_engine=self._run_engine,
                            parent_run_id=run_id,
                            deliverable_id=run.deliverable_id,
                            config_snapshot=run.config_snapshot,
                            stage_rows=stage_rows,
                        )
                log.info(
                    "reconciler-finalize: run %s was DAG-complete but stranded in 'running' "
                    "(owning task interrupted before the completion gate) → completed",
                    run_id,
                )
                return "completed"
            finally:
                clear_identity_context()
                clear_run_provider_slots()

    async def _publish(self, run_id: UUID, event: str, **payload: Any) -> None:
        # Count the run-level outcome BEFORE the publish, so the metric reflects
        # the actual terminal transition even if SSE delivery fails. Terminal
        # events are uniquely named; non-terminal (run_started/stage_*) map to
        # None and are not counted. A run that pauses then resumes+completes
        # legitimately records two transitions (paused, then completed).
        outcome = _TERMINAL_EVENT_OUTCOMES.get(event)
        if outcome is not None:
            runs_total.labels(outcome=outcome).inc()
        # Best-effort: a Redis blip must never mask a run's real outcome. This
        # is called on the run-failed path (inside the DAG-failure except), so
        # an unguarded publish here would replace the original error with a
        # RedisError. Swallow + log + COUNT (so a silent SSE outage is alertable
        # instead of invisible); the FE resyncs via runs.get() on SSE reconnect
        # (same contract as _publish_child_status below).
        msg = {"event": event, "run_id": str(run_id), **payload}
        try:
            await self._redis.publish(f"runs:{run_id}", json.dumps(msg))
        except (RedisError, OSError) as exc:
            notify_errors_total.inc()
            log.warning("Failed to publish run event %s for run %s: %s", event, run_id, exc)

    @staticmethod
    async def _recompute_project(
        session: Any, child_project_id: UUID | None, parent_project_id: UUID | None = None
    ) -> None:
        """Re-derive the owning project's status after a run transition, then
        roll up to its parent (course) so a lesson's outcome reflects on the
        course.

        Runs on the caller's session *before* its commit so project status
        commits atomically with the run-status change. The child is recomputed
        first; its UPDATE is visible to the parent's children query within the
        same transaction. No-op when the run's project id wasn't resolved
        (defensive — it always is in practice).
        """
        if child_project_id is None:
            return
        repo = ProjectRepository(session)
        await repo.recompute_status(child_project_id)
        if parent_project_id is not None:
            await repo.recompute_status(parent_project_id)

    async def _publish_child_status(
        self,
        parent_project_id: UUID | None,
        child_project_id: UUID | None,
        status: str,
    ) -> None:
        # Best-effort mirror of a run's lifecycle onto the parent project's
        # SSE channel. No-op for top-level projects (no parent). Redis
        # failures are logged and swallowed — the DB row was already
        # committed, and the FE resyncs via list/get on EventSource
        # reconnect (same contract as projects._publish_project_event).
        if parent_project_id is None or child_project_id is None:
            return
        msg = {
            "event": "child_status_changed",
            "project_id": str(child_project_id),
            "status": status,
        }
        try:
            await self._redis.publish(f"projects:{parent_project_id}", json.dumps(msg))
        except (RedisError, OSError) as exc:
            log.warning(
                "Failed to publish child_status_changed (parent=%s child=%s status=%s): %s",
                parent_project_id,
                child_project_id,
                status,
                exc,
            )
