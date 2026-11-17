from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from w2v_common.auth import set_correlation_context, set_w2v_job_context
from w2v_common.stages import StageTimeoutError

from src.config import LLM_STAGE_BASE_CEILING_S
from src.models.failure_class import FAILURE_CLASSES, UNKNOWN, FailureClass
from src.services.blueprints.base import DAGBlueprint, StageSpec
from src.services.middleware.chain import build_chain
from src.services.middleware.metrics import MetricsMiddleware
from src.services.middleware.timeout import TimeoutMiddleware
from src.services.runs.config_reads import RecordingSnapshot
from src.services.runs.run_context import RunContext
from src.services.stages.base import (
    StageHandler,
    StageOutput,
    StageProducedNothingError,
    StageReportedFailureError,
    StageSkippedError,
    missing_produced_fields,
    raised_failure_flags,
)
from src.services.stages.shared.degradation import ensure_degradation_codes
from src.utils import describe_exception

log = logging.getLogger(__name__)

EmitFn = Callable[..., Awaitable[None]]
PersistFn = Callable[..., Awaitable[None]]
# Bump the stage's timeout-resume counter and return the new value.
BumpAttemptsFn = Callable[[str], Awaitable[int]]
# Durably record that a stage was reset by a DRAIN/CANCEL rather than a timeout.
NoteInterruptedFn = Callable[[str], Awaitable[int]]
#: ``(stage_name, config_reads)`` — persists the snapshot keys a stage read
#: (``stage_outputs.config_reads``, see ``services/runs/config_reads.py``).
RecordConfigReadsFn = Callable[[str, list[str]], Awaitable[None]]
# Every stage's output by stage name — what ``DAGExecutor.run`` returns.
StageOutputs = dict[str, dict[str, Any]]

# A genuine timeout (the outer stage ceiling or the inner stage_call budget) that
# is treated as *slowness, not a deterministic failure* and re-dispatched/resumed
# rather than terminal-failing the run — as long as the stage stays under
# ``max_stage_attempts``. Both the outer ``TimeoutMiddleware`` (``asyncio.wait_for``
# → builtin ``TimeoutError``) and the inner ``stage_call`` all-attempts-timeout
# (``StageTimeoutError``) count; a ``StageRetriesExhaustedError`` from bad output is
# NOT a timeout and stays terminal.
_RESUMABLE_TIMEOUT_EXC: tuple[type[BaseException], ...] = (TimeoutError, StageTimeoutError)


# How long the batch drain keeps waiting for its stages AFTER it has absorbed an
# outer cancel (pod SIGTERM / user cancel / lease loss). The only work left at
# that point is each cancelled stage's short "reset my row to pending" DB write,
# so this is generous; it exists ONLY so an absorb-and-wait can never become an
# unbounded hang, which would be a worse bug than the strand it prevents.
_DRAIN_GRACE_AFTER_CANCEL_S = 10.0
# The wall-clock bound above cannot bound a cancel that is re-delivered on every
# await: each delivery returns instantly, so the clock never advances. Bound the
# number of absorbs too.
_MAX_ABSORBED_CANCELS = 3


class _DrainState:
    """Bookkeeping for :func:`_drain_batch` — how the drain reacts to a cancel.

    Split out of the drain loop only to keep that loop under the complexity gate;
    every decision here is described in ``_drain_batch``'s docstring.
    """

    def __init__(self, *, batch_cancelled: bool) -> None:
        self.batch_cancelled = batch_cancelled
        self.absorbed: asyncio.CancelledError | None = None
        self._count = 0
        self._deadline: float | None = None

    def remaining(self, now: float) -> float | None:
        """Seconds left of the post-cancel grace, or ``None`` while uncancelled."""
        return None if self._deadline is None else self._deadline - now

    def absorb(
        self, exc: asyncio.CancelledError, pending: set[asyncio.Task[Any]], now: float
    ) -> bool:
        """Swallow one cancel; return whether the drain should keep waiting."""
        self.absorbed = self.absorbed or exc
        self._count += 1
        # Clear the request so the next await actually waits instead of re-raising
        # instantly (py3.11+ keeps a cancelling counter and a `_must_cancel` flag
        # that would otherwise spin the drain loop without advancing the clock).
        current = asyncio.current_task()
        if current is not None:
            current.uncancel()
        if not self.batch_cancelled:
            # We were letting the batch drain naturally (resumable-timeout path).
            # The run itself is now going down, so a still-running sibling has
            # nothing to drain FOR — cancel it ONCE so it runs its own stage-row
            # reset instead of being abandoned detached.
            for task in pending:
                task.cancel()
            self.batch_cancelled = True
        if self._deadline is None:
            self._deadline = now + _DRAIN_GRACE_AFTER_CANCEL_S
        return self._count < _MAX_ABSORBED_CANCELS


async def _drain_batch(
    tasks: list[asyncio.Task[Any]], *, cancel_first: bool
) -> asyncio.CancelledError | None:
    """Wait out a stage batch, surviving a SECOND, independent cancel.

    Returns the first ``CancelledError`` delivered to THIS task while draining
    (the caller re-raises it so a cancelled run is not misreported as a plain
    stage failure), or ``None`` if the drain was never cancelled.

    Why this is not ``await asyncio.gather(*tasks, return_exceptions=True)``:
    that await is a cancellation point, and ``gather`` re-cancels its children on
    every ``cancel()`` it receives. ``_run_one``'s ``CancelledError`` branch resets
    its stage row from ``running`` back to ``pending`` with a plain ``await`` —
    correct only while cancel is delivered ONCE. A cancel of the batch followed by
    an independent cancel of the run task (pod SIGTERM, user cancel, lost lease)
    killed that reset half-way; those stage rows stayed ``running`` forever and
    were later swept to ``cancelled``, which is exactly the operator-blind state
    the W1-1 comment in ``_run_one`` exists to prevent.

    Why not a bare ``asyncio.shield`` either: ``await shield(fut)`` raises the
    CancelledError *immediately* and lets the caller walk away while the inner
    work continues detached — it protects the coroutine, not the wait. So the
    cancellation is absorbed and the wait is RESUMED, bounded in both wall time
    (``_DRAIN_GRACE_AFTER_CANCEL_S``) and deliveries (``_MAX_ABSORBED_CANCELS``);
    an unbounded absorb-and-retry would hang the pod, a worse bug than the strand.

    ``asyncio.wait`` is the wait primitive on purpose: unlike ``gather``, having
    its own await cancelled does NOT cancel the tasks it is waiting on. Each stage
    is therefore cancelled at most ONCE by this function (``cancel_first``, or the
    first absorbed cancel), which is what keeps ``_run_one``'s single-delivery
    assumption true.
    """
    loop = asyncio.get_running_loop()
    if cancel_first:
        for task in tasks:
            task.cancel()
    state = _DrainState(batch_cancelled=cancel_first)
    pending: set[asyncio.Task[Any]] = set(tasks)

    while pending:
        timeout = state.remaining(loop.time())
        if timeout is not None and timeout <= 0:
            break
        try:
            _done, pending = await asyncio.wait(pending, timeout=timeout)
        except asyncio.CancelledError as exc:
            if not state.absorb(exc, pending, loop.time()):
                break

    if pending:
        log.error(
            "stage batch drain gave up with %d task(s) still running after an outer "
            "cancel — their stage rows may be left 'running'",
            len(pending),
        )
    return state.absorbed


class StageResumableTimeoutError(Exception):
    """A stage timed out but is under its resume budget — resume, don't fail.

    Raised by ``DAGExecutor._run_one`` and caught by ``RunDispatcher._execute_run``,
    which leaves the run in ``running`` so the reconciler re-dispatches it and it
    resumes from the last completed stage (the timed-out stage is reset to
    ``pending``). Distinct from every real failure, which stays terminal.
    """

    def __init__(self, stage_name: str, attempts: int) -> None:
        super().__init__(f"stage {stage_name!r} timed out (resume attempt {attempts})")
        self.stage_name = stage_name
        self.attempts = attempts


class DAGExecutor:
    def __init__(
        self,
        blueprint: DAGBlueprint,
        *,
        emit: EmitFn,
        persist: PersistFn,
        handler_deps: dict[str, Any] | None = None,
        bump_attempts: BumpAttemptsFn | None = None,
        note_interrupted: NoteInterruptedFn | None = None,
        max_stage_attempts: int = 0,
        stage_base_ceiling_seconds: int = LLM_STAGE_BASE_CEILING_S,
        run_deadline_at: datetime | None = None,
        record_config_reads: RecordConfigReadsFn | None = None,
    ) -> None:
        self.blueprint = blueprint
        self.emit = emit
        self.persist = persist
        self._handler_deps: dict[str, Any] = dict(handler_deps or {})
        # Outer per-stage ceiling for stages whose StageSpec.default_timeout_seconds
        # is None (the "inherit the runtime LLM base ceiling" sentinel). The dispatcher
        # injects PipelineSettings.llm_stage_base_ceiling_s; a bare executor (unit tests)
        # falls back to the module-constant default so old semantics hold.
        self._stage_base_ceiling_seconds = stage_base_ceiling_seconds
        # Resume-on-timeout wiring (durability P1). ``bump_attempts`` durably
        # increments the stage's timeout counter; ``max_stage_attempts`` is the cap
        # past which a repeatedly-slow stage terminal-fails. When bump_attempts is
        # not wired (defaults), timeouts fall through to terminal failure exactly
        # as before — so unit tests that construct a bare executor keep old semantics.
        self._bump_attempts = bump_attempts
        # Durably records a DRAIN/CANCEL reset, the twin of ``bump_attempts``'s
        # timeout counter. Optional and defaulted to None for exactly the reason
        # ``bump_attempts`` is: a bare executor (unit tests) keeps old semantics and
        # simply does not record. It is deliberately NOT folded into ``persist``:
        # ``persist(name, 'pending')`` is issued by both the timeout and the
        # cancellation paths, so the distinction has to be made HERE, at the only
        # place that still knows which exception it caught.
        self._note_interrupted = note_interrupted
        self._max_stage_attempts = max_stage_attempts
        # Wall-clock the whole RUN may not outlive. `None` (a bare executor, or the
        # knob set to 0) keeps the pre-existing behaviour exactly: the resume ladder is
        # bounded only per-stage, by `max_stage_attempts`.
        #
        # Why this exists, measured over 30 days of production runs: the resume ladder
        # is per-STAGE, so nothing bounded the run. Five runs burned over two hours
        # each, every one of them already lost — the worst was 9.7 HOURS
        # (`website_review_v1`, `website_redesign.wait`, "all 8 attempts timed out").
        # The longest run that ever SUCCEEDED in the same window was 5176s (1.44h). So
        # the failed tail was 6.7x the successful maximum, and every hour of it was
        # paid LLM time spent re-issuing a call that had already timed out repeatedly.
        self._run_deadline_at = run_deadline_at
        # Persists which snapshot keys each completed stage READ, so a later Restart
        # re-runs only the stages a changed input affects (restart_plan.py). Optional
        # like the two hooks above: a bare executor records nothing, and an
        # unrecorded row is treated as "reads everything" — the safe default.
        self._record_config_reads = record_config_reads

    def _run_deadline_passed(self) -> float | None:
        """Seconds the run is PAST its deadline, or ``None`` if it is not.

        ``None`` also when no deadline is configured — the caller treats that and
        "still inside the budget" identically on purpose, so the knob at 0 is a true
        no-op rather than a different code path.
        """
        if self._run_deadline_at is None:
            return None
        overrun = (datetime.now(UTC) - self._run_deadline_at).total_seconds()
        return overrun if overrun > 0 else None

    def _build_chain(
        self, spec: StageSpec
    ) -> Callable[[Callable[[], Awaitable[Any]]], Awaitable[Any]]:
        # Metrics + Timeout only. Retry is deliberately NOT applied here: the
        # handlers already retry at the call boundary where idempotency is known
        # per call (`stage_call(retries=...)`, `stage_gather(per_item_retries=...)`,
        # and BaseServiceClient's transport/5xx retry). A blind outer retry on the
        # whole handler stacked a 3x2x2 cascade on top of those and — worse —
        # re-ran non-idempotent compile/render handlers that set `retries=0` on
        # purpose, causing duplicate renders. The old per-(stage,tenant) circuit
        # breaker was a process-global dict not shared across pods, so it gave no
        # real cross-pod protection; both were removed rather than re-libraried.
        # Timeout stays as the run-level budget ceiling that covers the inner
        # retry worst-case (see the per-stage default_timeout_seconds reasoning
        # in the blueprints).
        # None => inherit the runtime base LLM ceiling (env-configurable); an explicit
        # int on the spec (redesign, record, short deterministic stages) wins.
        timeout_seconds = (
            spec.default_timeout_seconds
            if spec.default_timeout_seconds is not None
            else self._stage_base_ceiling_seconds
        )
        return build_chain(
            [
                MetricsMiddleware(stage_name=spec.name),
                TimeoutMiddleware(seconds=timeout_seconds),
            ]
        )

    @staticmethod
    def _enforce_degradation_codes(stage_name: str, output_dict: dict[str, object]) -> None:
        """``degraded`` ⇒ at least one CODED entry (see ``ensure_degradation_codes``)."""
        if ensure_degradation_codes(stage_name, output_dict):
            log.warning(
                "stage %s degraded via prose notes only — stamped the "
                "unclassified_note code; give this producer a real degradation code",
                stage_name,
            )

    async def _run_one(self, spec: StageSpec, ctx: RunContext) -> tuple[dict[str, Any], bool]:
        """Run one stage. Returns ``(output, was_skipped)``.

        ``was_skipped`` is True only when the stage raised ``StageSkippedError``
        (an opt-in/gated stage the run didn't request). The caller uses it so a
        *skipped* stage never trips a ``pause_after`` review gate — there is no
        output to review, and pausing there wedges the run (the finalizer's
        pause check counts only ``completed`` stages, so executor and finalizer
        would disagree). See ``run``'s pause guard.
        """
        # Stamp cost-correlation for this stage so every LLM cost event emitted by
        # worker_management / inference-router during the stage is attributable to
        # this run + stage (correlation_id=run_id, stage=spec.name). ContextVars
        # are task-local — parallel-group siblings run in separate gather tasks,
        # each carrying its own stage value. w2v_common's identity_event_hook
        # forwards these on the outbound WM call.
        set_correlation_context(str(ctx.run_id), spec.name)
        await self.emit("stage_started", stage_name=spec.name)
        await self.persist(spec.name, "running")
        handler = spec.handler_factory(self._handler_deps)
        chain = self._build_chain(spec)
        try:
            _h = handler
            _ctx = ctx

            async def _call(_h: StageHandler = _h, _ctx: RunContext = _ctx) -> StageOutput:
                return await _h.run({}, _ctx)

            result = await chain(_call)
            # INSIDE the try on purpose. `_enforce_produces` can raise
            # StageProducedNothingError for a `produces_required` stage, and that
            # raise has to reach the same terminal path as any other stage failure
            # — persist 'failed' + emit stage_failed. Sitting outside the try (as
            # it did until W1-1) it propagated straight past every handler: the
            # dispatcher still marked the RUN failed, but the guilty stage row was
            # left 'running' and later swept to 'cancelled' by the failure sweeper,
            # and no stage_failed event was ever emitted. The operator then sees a
            # failed run whose stages all read 'cancelled' — nothing points at the
            # stage that produced nothing. model_dump() moves in with it so the
            # dump and the gate stay adjacent; a dump that blows up is a real
            # failure too and belongs on the same path.
            output_dict = result.model_dump()
            # Same placement, same reasoning, different question: `produces` asks
            # "is the output empty?", this asks "did the stage say it failed?".
            # A stage with a legitimate empty mode can only answer the second.
            #
            # ORDER MATTERS, and this one is deliberate. A stage may declare both
            # (`promo_slides` does, "belt AND braces"), and then both fire on the
            # same output — but only this gate can say WHY. It reads a flag the
            # stage set at the exact point it swallowed the downstream error and
            # forwards that error's text out of `notes` into the message;
            # `_enforce_produces` can only infer a defect from an empty field and
            # ends with "look at the stage handler for a swallowed downstream
            # error", the cause discarded. A stated failure outranks an inferred
            # one — the docstring below puts it as "an empty declared field is a
            # guess about intent, but a raised flag was set by the stage itself".
            #
            # Running produces first is not a style question, it lost real time:
            # in the 2026-09-08/09 promo outage every failed run recorded the
            # cause-free message, and three separate investigations read it as
            # naming the orchestrator when the fault was a slide_service template
            # lookup. The cause sat in `notes` the whole time, one gate away.
            self._enforce_failure_flags(spec.name, type(result), output_dict)
            self._enforce_produces(spec.name, type(result), output_dict)
            # LAST, after `_enforce_produces` may have appended a note: a stage
            # that reads `degraded` must carry at least one CODED entry, or the run
            # record shows `degraded=true, degradation_codes=[]` (prod 01a09f82's
            # media_discovery). One chokepoint for every stage of every blueprint.
            self._enforce_degradation_codes(spec.name, output_dict)
        except StageSkippedError as exc:
            # An OPT-IN stage the run didn't request (e.g. no ai_image_gen config).
            # NOT a failure: persist 'skipped' and return an empty output so the
            # stage counts as terminal at the completion gate and its dependents
            # stay eligible (they read `(prior_output(name) or {}).get(...)` and
            # tolerate absence). Only StageSkippedError lands here — a
            # StagePreconditionError ("no shots to compile") is a real failure and
            # deliberately falls through to `except Exception` → 'failed'.
            log.info("stage %s skipped: %s", spec.name, exc.reason)
            # A skip is a DECISION made from the snapshot (an absent opt-in block):
            # record what it read so a Restart re-evaluates it — and re-runs its
            # dependents — when that input changes.
            await self._persist_config_reads(spec.name, ctx)
            await self.persist(spec.name, "skipped", error=exc.reason)
            await self.emit("stage_skipped", stage_name=spec.name, reason=exc.reason)
            return {}, True
        except asyncio.CancelledError:
            # Cancellation (pod drain / pause / cancel) is a BaseException, so it
            # bypasses the `except Exception` below — without this branch the
            # stage row persisted "running" above is never moved off "running"
            # and the run shows a stage stuck running forever. Best-effort reset
            # to "pending" so a resumed run re-executes it, then propagate. The
            # reset is awaited (not shielded/detached) so it actually lands.
            # That relies on cancel() being delivered to THIS task exactly once —
            # which is a guarantee of `_drain_batch`, not of asyncio: the drain
            # cancels each stage task at most once and absorbs any further cancel
            # aimed at the run. Do not reintroduce a bare `gather` there.
            with contextlib.suppress(Exception):
                await self.persist(spec.name, "pending")
            # Record the drain durably, AFTER the reset and in its own suppress.
            # Order matters: the status reset is correctness (a row left 'running'
            # wedges the resume), this counter is bookkeeping, so if a second
            # cancel lands between the two we lose the metric and not the reset.
            #
            # Why it is recorded at all: `mark_pending` leaves a row identical to
            # the one a resumable TIMEOUT leaves, and only the timeout path bumps
            # `attempts`. A drained stage therefore resumed with `attempts == 0`
            # and its original `started_at`, so every duration derived from that
            # row silently included the burnt execution and the pod downtime while
            # looking like a single clean run. Pods roll constantly, so that is the
            # common case, not the edge case.
            if self._note_interrupted is not None:
                with contextlib.suppress(Exception):
                    await self._note_interrupted(spec.name)
            raise
        except _RESUMABLE_TIMEOUT_EXC as exc:
            # A genuine timeout — the outer stage ceiling or the inner stage_call
            # all-attempts-timeout — is SLOWNESS, not a deterministic failure.
            # Bounded resume: durably bump the per-stage counter; while under the
            # cap, reset the stage to 'pending' (a re-dispatch then re-runs only
            # this stage; completed stages are preserved) and raise
            # StageResumableTimeoutError so the dispatcher leaves the run 'running' for
            # the reconciler to resume. Over the cap — or when resume isn't wired
            # (bare executor / old semantics) — fall through to terminal failure so
            # an always-slow stage can't loop forever.
            attempts = await self._bump_attempts(spec.name) if self._bump_attempts else 0
            # The run deadline is checked HERE, at the resume decision, and nowhere
            # else. Resuming is the only thing that can extend a run past its stages'
            # own ceilings, so this is the single point where an unbounded burn is
            # authorised — and refusing here fails the run through the SAME terminal
            # path as an exhausted resume budget, carrying the timeout that caused it.
            # A stage already in flight is never cancelled mid-work by this.
            overran = self._run_deadline_passed()
            if overran is not None:
                log.error(
                    "stage %s timed out and the RUN deadline has passed by %.0fs "
                    "(resume attempt %d/%d) — failing instead of resuming",
                    spec.name,
                    overran,
                    attempts,
                    self._max_stage_attempts,
                )
            if (
                self._bump_attempts is not None
                and attempts <= self._max_stage_attempts
                and overran is None
            ):
                log.warning(
                    "stage %s timed out (resume attempt %d/%d) — resuming",
                    spec.name,
                    attempts,
                    self._max_stage_attempts,
                )
                with contextlib.suppress(Exception):
                    await self.persist(spec.name, "pending")
                raise StageResumableTimeoutError(spec.name, attempts) from exc
            err = describe_exception(exc)
            if overran is not None:
                err = (
                    f"RunDeadlineExceeded: run exceeded its wall-clock deadline by "
                    f"{overran:.0f}s while stage {spec.name!r} was being resumed "
                    f"(attempt {attempts}/{self._max_stage_attempts}); "
                    f"last stage error: {err}"
                )
            else:
                log.error(
                    "stage %s timed out and exhausted its resume budget (%d) — failing",
                    spec.name,
                    self._max_stage_attempts,
                )
            # Same hazard as the `except Exception` chokepoint below: `persist`/
            # `emit` are caller-supplied I/O and can themselves throw. Unguarded,
            # that would REPLACE `exc` (the exhausted-resume timeout) on the bare
            # `raise` — the dispatcher's failure classifier would then see a
            # persist/emit error instead of the real timeout. Contain both so the
            # bookkeeping failure is logged, never masks. Rare branch (only when
            # the resume budget is exhausted), same mechanism as W1-3/#924.
            await self._persist_failed_bookkeeping(
                spec.name, err, context="exhausting its resume budget"
            )
            raise
        except Exception as exc:
            err = describe_exception(exc)
            log.exception("stage %s failed", spec.name)
            # `persist`/`emit` are caller-supplied I/O (DB write, event publish) and
            # can themselves throw. Unguarded, that exception would REPLACE `exc` on
            # the `raise` below — every caller upstack (the dispatcher's `except
            # Exception`, the run-failure classifier) would then see a persist/emit
            # error instead of the real stage failure that caused it. Contain both
            # calls so the bookkeeping best-effort failure is logged, never masks.
            await self._persist_failed_bookkeeping(
                spec.name, err, context="the real stage error above"
            )
            raise
        await self._persist_config_reads(spec.name, ctx)
        await self.persist(spec.name, "completed", output=output_dict)
        await self.emit("stage_completed", stage_name=spec.name)
        return output_dict, False

    async def _persist_config_reads(self, stage_name: str, ctx: RunContext) -> None:
        """Record which snapshot keys ``stage_name`` read (restart_plan.py).

        Called BEFORE the completion write: a crash between the two leaves the row
        non-terminal (it re-runs), never completed with reads it did not record.
        Best-effort — a failed write leaves ``config_reads`` NULL, which the restart
        planner reads as "unknown, re-run on any input change".
        """
        if self._record_config_reads is None or not isinstance(
            ctx.config_snapshot, RecordingSnapshot
        ):
            return
        try:
            await self._record_config_reads(stage_name, ctx.config_snapshot.reads())
        except Exception:
            log.warning("could not record config reads for stage %s", stage_name, exc_info=True)

    async def _persist_failed_bookkeeping(self, stage_name: str, err: str, *, context: str) -> None:
        """Best-effort ``persist(..., 'failed')`` + ``emit('stage_failed')`` pair,
        shared by both terminal-failure exits of ``_run_one`` (a genuine stage
        exception, and an exhausted resume-timeout budget). Both callers already
        contain this in their OWN try/except before a bare ``raise`` — pulled out
        once the duplication pushed ``_run_one`` over the complexity gate — so a
        persist/emit failure here is logged and swallowed, never propagated: the
        caller's `raise` must always carry the real stage error, not this one.
        """
        try:
            await self.persist(stage_name, "failed", error=err)
            await self.emit("stage_failed", stage_name=stage_name, error=err)
        except Exception:
            log.exception(
                "stage %s: failed to persist/emit 'failed' bookkeeping after %s — "
                "the run may show stale 'running' state",
                stage_name,
                context,
            )

    @staticmethod
    def _enforce_produces(
        stage_name: str, output_model: type[StageOutput], output: MutableMapping[str, object]
    ) -> None:
        """The "did this stage actually do its job?" gate.

        This is deliberately HERE and nowhere else. Every stage of every
        blueprint records its success through the two lines above, so one check
        placed between them covers all of them — whereas the failure mode it
        catches (a stage swallowing its own error, writing an empty artifact and
        reporting ``completed``) has been re-implemented independently in a dozen
        stage modules, each locally reasonable. Fixing those one at a time does
        not converge; a new stage just reintroduces it.

        Default is DEGRADE, not fail: a note is appended, and
        ``api/routes/runs/_shared.py`` turns non-empty notes into a ``degraded``
        run. That makes adopting ``produces`` on a stage safe — it can surface a
        defect without turning a working pipeline red. ``produces_required``
        opts a stage into hard failure where an empty output is unambiguous.
        """
        produces = getattr(output_model, "produces", ())
        if not produces:
            return
        empty = missing_produced_fields(output, produces)
        if not empty:
            return

        if getattr(output_model, "produces_required", False):
            raise StageProducedNothingError(stage_name, empty)

        joined = ", ".join(empty)
        log.warning(
            "stage %s completed but produced nothing for declared field(s): %s",
            stage_name,
            joined,
        )
        # `notes` is the run-level degrade channel. A stage that declares
        # `produces` without a `notes` field would have nowhere to say so, which
        # is why the ratchet test requires the two to go together.
        notes = output.get("notes")
        if isinstance(notes, list):
            message = (
                f"{stage_name}: completed without producing {joined} — "
                f"the stage reported success but this output is empty"
            )
            # Rebuilt rather than appended in place: the dumped value is typed
            # `object`, so `isinstance` narrows it to `list[Never]` and `.append`
            # does not type-check without a cast. Rebuilding needs no cast and
            # keeps the mutation to this one key.
            output["notes"] = [*notes, message]

    @staticmethod
    def _enforce_failure_flags(
        stage_name: str, output_model: type[StageOutput], output: MutableMapping[str, object]
    ) -> None:
        """The "did this stage say its own work failed?" gate.

        Sibling of :meth:`_enforce_produces`, and here for the same reason: one
        check between the two lines that record every stage's success covers
        every blueprint, where a per-stage guard converges on nothing.

        It exists because `produces` structurally CANNOT cover the stages that
        need it most. `showcase_slides` and `website_review_slides` both have a
        legitimate empty deck (an all-footage demo, a url-only shotlist authors
        no slides), so `produces=("slides",)` would fail correct runs — which is
        exactly why neither declared it, and exactly why a total slide_service
        outage completed clean with every branded card missing from a playable,
        billable video. `fails_when_flagged` keys on the FAILURE, not on the
        emptiness, so the benign empty and the outage stop being the same fact.

        No degrade tier, unlike `produces`: an empty declared field is a guess
        about intent, but a raised flag was set by the stage itself at the point
        it swallowed the error. There is nothing ambiguous left to soften.
        """
        flags = getattr(output_model, "fails_when_flagged", ())
        if not flags:
            return
        raised = raised_failure_flags(output, flags)
        if not raised:
            return
        # `notes` carries the swallowed downstream cause (the soft-fail boundary
        # writes it there). Passed into the error so the operator gets the real
        # reason — a bare "the flag was true" would just relocate the mystery.
        notes = output.get("notes")
        # The stage already attributed the fault it swallowed (see
        # `shared/slide_stage.py`, which stamps `attribute_exception(exc)` onto
        # its result and whose consumers surface it as a field). It arrives here
        # only as a DUMPED value typed `object`, so it is narrowed back to
        # `FailureClass` by membership over the vocabulary — the same pattern as
        # `models/failure_class.attribute_exception`, and for the same reason: a
        # persisted output is arbitrary data and must never widen the enum. An
        # absent or unrecognised value stays `unknown`, which is the honest
        # answer rather than a guess.
        dumped_class = output.get("failure_class")
        failure_class: FailureClass = UNKNOWN
        for member in FAILURE_CLASSES:
            if dumped_class == member:
                failure_class = member
                break
        raise StageReportedFailureError(
            stage_name,
            raised,
            notes if isinstance(notes, list) else (),
            failure_class=failure_class,
        )

    async def run(self, ctx: RunContext, *, pause_after_stages: set[str]) -> StageOutputs:
        """Run the DAG with the run id bound as the w2v job for cost attribution.

        ``set_w2v_job_context(run_id)`` makes every outbound w2v call a stage
        makes carry ``X-W2V-Job-Id`` (``identity_event_hook``), and every direct
        cost event w2v_common emits in-process default its ``job_id`` to the run,
        so per-run spend is attributable — the job sibling of the project bound
        by ``run_dispatcher``. Bound HERE, before any stage task exists, so each
        ``create_task`` copies it into every stage (parallel siblings included).
        The executor runs inside the per-run task ``dispatch`` creates, so the
        bind is confined to that task's context; the clear in ``finally`` keeps
        it from outliving the DAG (the dispatcher's post-DAG bookkeeping is not
        stage work), mirroring ``run_dispatcher._run``'s project clear.
        """
        set_w2v_job_context(ctx.run_id)
        try:
            return await self._run_dag(ctx, pause_after_stages=pause_after_stages)
        finally:
            set_w2v_job_context(None)

    async def _run_dag(self, ctx: RunContext, *, pause_after_stages: set[str]) -> StageOutputs:
        """Schedule stages as deps are satisfied; run every co-eligible stage concurrently."""
        outputs: StageOutputs = dict(ctx.prior_run_outputs)
        completed: set[str] = set(outputs.keys())
        all_stages = {s.name: s for s in self.blueprint.stages}
        remaining = set(all_stages) - completed
        paused = False

        async def step_for(name: str) -> tuple[str, dict[str, Any], bool]:
            # Each stage gets its OWN recording copy of the snapshot, so the keys it
            # reads are attributable to it even while siblings run concurrently.
            ctx_view = replace(
                ctx,
                prior_run_outputs={**outputs},
                config_snapshot=RecordingSnapshot(ctx.config_snapshot),
            )
            out, skipped = await self._run_one(all_stages[name], ctx_view)
            return name, out, skipped

        while remaining and not paused:
            eligible = sorted(
                n for n in remaining if all(dep in completed for dep in all_stages[n].depends_on)
            )
            if not eligible:
                raise RuntimeError(
                    f"no eligible stages but {sorted(remaining)} remain — DAG deadlock?"
                )
            # Run EVERY co-eligible stage concurrently. ``depends_on`` is the only
            # ordering constraint, so any stage whose deps are all satisfied may run
            # this round — there is no benefit to draining stages one batch at a time.
            # (``parallel_group`` is a display-only label on the blueprint API; it does
            # NOT throttle scheduling. A stage that must run after another carries a
            # ``depends_on`` edge — never rely on round batching for ordering.)
            #
            # A bare asyncio.gather re-raises the first sibling's failure but leaves
            # the others running — a non-idempotent render/TTS/upload would keep going
            # detached after the run is already marked failed (duplicate artifacts,
            # wasted spend, a late stage_completed SSE emitted after the terminal
            # event). Cancel + drain the whole batch before propagating so nothing
            # outlives the failure. The original exception (not an ExceptionGroup) is
            # re-raised unchanged, so the dispatcher's failure handling is preserved.
            tasks = [asyncio.create_task(step_for(n)) for n in eligible]
            try:
                results = list(await asyncio.gather(*tasks))
            except StageResumableTimeoutError as timeout_exc:
                # A resumable timeout is SLOWNESS, not a failure. Let the in-flight siblings
                # DRAIN to their natural terminal state (do NOT cancel): a sibling that already
                # reached 'completed' is then preserved and skipped on resume, so only the
                # timed-out stage — already reset to 'pending' in _run_one before it raised —
                # re-runs. Cancelling here would reset a mid-flight non-idempotent sibling to
                # 'pending' and duplicate its side effect on resume. gather propagates the
                # first-raised exception, so a genuine failure raised before a timeout still
                # lands in the cancel branch below; only a resumable-timeout-first batch drains.
                # The drain ABSORBS a second, independent cancel (pod SIGTERM / user
                # cancel / lease loss) rather than letting it abandon the batch: this
                # branch does not cancel, so it can otherwise sit here for the slowest
                # sibling's full stage ceiling with every row still 'running'.
                cancelled = await _drain_batch(tasks, cancel_first=False)
                if cancelled is not None:
                    # The run was CANCELLED. Report that, not the timeout that got us
                    # here — a bare `raise` would hand the dispatcher a resumable
                    # timeout and it would leave the run 'running' for the reconciler.
                    raise cancelled from timeout_exc
                raise
            except BaseException as batch_exc:
                # A genuine failure aborts the batch: cancel + drain so no detached non-idempotent
                # sibling runs on after the run is already failing (duplicate artifacts / late SSE).
                cancelled = await _drain_batch(tasks, cancel_first=True)
                if cancelled is not None:
                    # A cancel landed DURING the drain. Propagate the cancellation, not
                    # the stage failure, so the dispatcher does not misclassify a
                    # cancelled run as a plain stage failure.
                    raise cancelled from batch_exc
                raise
            for name, out, skipped in results:
                outputs[name] = out
                completed.add(name)
                remaining.discard(name)
                # A skipped pause-stage produced nothing to review, so it must
                # NOT pause — pausing there leaves the run wedged because the
                # finalizer only treats a *completed* pause-stage as a real
                # pause (`pause_set & completed`). Only a completed pause-stage
                # halts the DAG for human review.
                if name in pause_after_stages and not skipped:
                    paused = True
        return outputs
