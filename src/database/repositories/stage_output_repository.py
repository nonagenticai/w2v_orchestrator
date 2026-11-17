from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID, uuid7

from sqlalchemy import Result, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from src.database.models import StageOutput

logger = logging.getLogger(__name__)

# The non-terminal stage states. Every terminal/status-mutating writer guards on
# ``status IN _NON_TERMINAL`` so it can only mutate a row that has NOT yet reached
# a terminal state (completed/failed/skipped). This makes the terminal writes
# write-once: a racing/stale write across the 2 replicas, or a resume re-issuing
# a mark on an already-terminal row, is a benign no-op instead of a silent
# clobber of a stage's real result. ``mark_pending`` keeps its narrower
# ``status == 'running'`` guard (it only ever legitimately resets a running row).
_NON_TERMINAL = ("pending", "running")

# Run statuses that are TERMINAL and never resumed (``sub_dag.TerminalRunStatus``
# minus ``completed``, which cannot own a non-terminal stage row by construction).
# ``paused`` is deliberately NOT here: a paused run is resumable
# (``RunRepository.mark_resumed`` / ``POST /runs/{id}/resume``), so its
# not-yet-executed stages are still genuinely going to run and ``pending`` is the
# TRUE status for them.
TERMINATED_RUN_STATUSES: tuple[str, ...] = ("cancelled", "failed")


def stranded_stage_reason(run_status: str) -> str:
    """The ``stage_outputs.error`` text written on a stage the run outran.

    Deliberately phrased as a statement about the RUN, not about the stage. The
    only other "did not run" vocabulary in this table is ``skipped``, whose
    reason strings say the stage's own precondition was absent (e.g.
    ``site_capture: no brief.source_url (opt-in stage not requested)``). Those
    are different facts: an opt-out stage was never wanted, a stranded stage was
    wanted and the run died first. Conflating them would make the ``skipped``
    reasons untrustworthy — and ``skipped`` counts as terminal-SUCCESS in
    ``run_finalizer.TERMINAL_STAGE_STATUSES``, so labelling these ``skipped``
    would additionally let the completion gate declare a cancelled run's DAG
    complete.
    """
    return f"run terminated ({run_status}) before this stage ran; it never ran and never will"


def _wrote(result: Result[Any], writer: str, run_id: UUID, stage_name: str) -> bool:
    """Return whether a guarded terminal UPDATE matched a row, warning if it did not.

    The ``bool`` return is the only signal a no-op write leaves behind, and some
    call sites discard it — so a dropped stage result would otherwise be
    completely invisible. Emitting the warning HERE (rather than at each call
    site) makes every no-op observable regardless of what the caller does with
    the return value. ``False`` means the row was absent (an invariant breach —
    ``runs.run_setup.reconcile_blueprint_stage_rows`` guarantees a row per
    blueprint stage) or already terminal (the write-once guard fired, benign on
    a resume). Both are log-only: raising here would abort live dispatch.
    """
    if int(getattr(result, "rowcount", 0) or 0) > 0:
        return True
    logger.warning(
        "%s: no non-terminal stage_outputs row updated — run_id=%s stage_name=%s; "
        "the row is absent (invariant breach) or already terminal, so this write "
        "was a silent no-op and its result is dropped",
        writer,
        run_id,
        stage_name,
    )
    return False


class StageOutputRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_pending(self, run_id: UUID, stage_name: str) -> StageOutput:
        stmt = (
            pg_insert(StageOutput)
            .values(
                id=uuid7(),
                run_id=run_id,
                stage_name=stage_name,
                status="pending",
                # Seeded at INSERT rather than by a server default, so that NULL
                # keeps meaning "written before 0067, interruption history UNKNOWN"
                # and 0 means "recorded, never drained". A server default would
                # have stamped 0 onto every historical row and asserted a clean
                # history the database does not have.
                interruptions=0,
            )
            .on_conflict_do_nothing(index_elements=["run_id", "stage_name"])
        )
        await self.session.execute(stmt)
        result = await self.get(run_id, stage_name)
        if result is None:
            raise RuntimeError(
                f"upsert_pending: row missing after insert for {run_id}/{stage_name}"
            )
        return result

    async def insert_pending_bulk(self, run_id: UUID, stage_names: Sequence[str]) -> None:
        """Insert the given stages as ``pending`` in ONE statement; never raises.

        The reconciliation path for a run whose blueprint gained a stage after
        the run was created (``runs.run_setup.reconcile_blueprint_stage_rows``).
        Deliberately unlike ``upsert_pending``: no read-back and no
        ``RuntimeError``, because this runs inside live dispatch where raising
        would abort the run into the generic failure path. A conflict on
        ``uq_stage_outputs_run_stage`` is ignored, so a second dispatch racing
        the same reconcile is a benign no-op.
        """
        if not stage_names:
            return
        await self.session.execute(
            pg_insert(StageOutput)
            .values(
                [
                    {
                        "id": uuid7(),
                        "run_id": run_id,
                        "stage_name": name,
                        "status": "pending",
                        # See upsert_pending: NULL is reserved for pre-0067 rows.
                        "interruptions": 0,
                    }
                    for name in stage_names
                ]
            )
            .on_conflict_do_nothing(index_elements=["run_id", "stage_name"])
        )

    async def insert_skipped_bulk(self, run_id: UUID, entries: Sequence[tuple[str, str]]) -> None:
        """Insert the given stages as already-``skipped`` in ONE statement; never raises.

        The retrofit-skip counterpart to :meth:`insert_pending_bulk`
        (``runs.run_setup.reconcile_blueprint_stage_rows``): a stage ADDED to a
        blueprint upstream of a dependent that has already started on THIS run
        must never seed as ``pending`` — the frozen ``config_snapshot`` predates
        the stage, so a real execution would produce output no downstream
        consumer was validated against, and a completed dependent would never be
        re-run to consume it anyway. Marking it ``skipped`` (terminal-SUCCESS,
        per ``run_finalizer.TERMINAL_STAGE_STATUSES``) lets the completion gate
        close exactly as it does for any other opt-out stage, and every
        downstream reader already has to tolerate a skipped dependency (any
        stage can raise ``StageSkippedError`` at runtime), so nothing downstream
        needs special-casing.

        ``entries`` is ``(stage_name, reason)`` pairs; the reason is written
        verbatim into ``error``, matching :meth:`mark_skipped`. Deliberately
        unlike ``upsert_pending``: no read-back and no ``RuntimeError``, for the
        same live-dispatch reason as :meth:`insert_pending_bulk`. A conflict on
        ``uq_stage_outputs_run_stage`` is ignored, so a second dispatch racing
        the same reconcile is a benign no-op.
        """
        if not entries:
            return
        await self.session.execute(
            pg_insert(StageOutput)
            .values(
                [
                    {
                        "id": uuid7(),
                        "run_id": run_id,
                        "stage_name": name,
                        "status": "skipped",
                        "error": reason,
                        "completed_at": func.now(),
                        # See upsert_pending: NULL is reserved for pre-0067 rows.
                        "interruptions": 0,
                    }
                    for name, reason in entries
                ]
            )
            .on_conflict_do_nothing(index_elements=["run_id", "stage_name"])
        )

    async def get(self, run_id: UUID, stage_name: str) -> StageOutput | None:
        stmt = select(StageOutput).where(
            StageOutput.run_id == run_id, StageOutput.stage_name == stage_name
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_for_update(self, run_id: UUID, stage_name: str) -> StageOutput | None:
        """``get`` + ``SELECT ... FOR UPDATE`` — the read half of a read-modify-write
        on the ``output`` JSONB.

        The Studio PATCH routes load the WHOLE ``output`` blob, mutate it in
        Python and write it back. With a plain ``get`` two concurrent edits of
        two DIFFERENT elements both read the same base document and the second
        write silently discards the first — HTTP 200 on both, one edit gone, no
        signal anywhere. ``stage_outputs`` carries no ``lock_version`` (the
        storyboard tables got one in 0055/0056), so the row lock is the guard:
        the second request blocks here until the first commits and then re-reads
        the merged document. It serializes only concurrent writers of the SAME
        stage row and needs no schema change.
        """
        stmt = (
            select(StageOutput)
            .where(StageOutput.run_id == run_id, StageOutput.stage_name == stage_name)
            .with_for_update()
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_for_run(self, run_id: UUID) -> list[StageOutput]:
        """All stage rows for a run, in a DETERMINISTIC order.

        Callers treat this as an ordered list — Studio serialises it into
        ``RunDetail.stage_outputs`` and picks "the LAST stage carrying a clip
        array" — so an unordered SELECT returns physical heap order and makes
        that pick flip between requests (live run 01a08f79: clip reordering
        failed intermittently). ``created_at`` alone does NOT disambiguate:
        rows are seeded in ONE transaction and share an identical timestamp
        (all 16 rows of that run at 2026-09-11 09:58:30.332457+02), so
        ``stage_name`` is the key that actually makes the order stable.

        This is NOT a semantic ordering. It is not topological and carries no
        information about pipeline order — no caller may infer execution order,
        dependency order, or "latest stage" semantics from the position of a
        row in this list.
        """
        stmt = (
            select(StageOutput)
            .where(StageOutput.run_id == run_id)
            .order_by(StageOutput.created_at, StageOutput.stage_name)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def set_config_reads(self, run_id: UUID, stage_name: str, reads: list[str]) -> None:
        """Record the snapshot keys a stage read (migration 0102, restart_plan.py).

        Guarded to a NON-terminal row: it is written just before the completion write,
        and a terminal row's read set describes the execution that produced its output
        — a late/stale write must not replace it.
        """
        await self.session.execute(
            update(StageOutput)
            .where(
                StageOutput.run_id == run_id,
                StageOutput.stage_name == stage_name,
                StageOutput.status.in_(_NON_TERMINAL),
            )
            .values(config_reads=list(reads))
        )

    async def copy_config_reads_from(
        self, run_id: UUID, *, parent_run_id: UUID, stage_names: Sequence[str]
    ) -> None:
        """Copy the parent's ``config_reads`` onto this run's PRESERVED stage rows.

        A preserved row carries the parent's output, so what that output was computed
        from is the parent's read set. Without the copy a later Restart of the child
        would see NULL ("unknown") on every inherited stage and re-run all of them on
        any input change.
        """
        if not stage_names:
            return
        parent = aliased(StageOutput)
        await self.session.execute(
            update(StageOutput)
            .where(
                StageOutput.run_id == run_id,
                StageOutput.stage_name.in_(list(stage_names)),
                parent.run_id == parent_run_id,
                parent.stage_name == StageOutput.stage_name,
            )
            .values(config_reads=parent.config_reads)
        )

    async def mark_running(self, run_id: UUID, stage_name: str) -> None:
        """Mark a stage 'running'. Guarded to a non-terminal row so a resume that
        re-dispatches an already-terminal stage (e.g. a deterministically
        'skipped' one, which is not seeded into the resume completed-set) cannot
        reopen it. ``pending`` and ``running`` are both allowed: first dispatch,
        idempotent re-dispatch, and post-``mark_pending`` resume.

        COALESCE on ``started_at`` so it is stamped once, on the FIRST attempt —
        the same rule ``RunRepository.mark_started`` already applies at the run
        level, for the same reason. A stage that hits a resumable timeout is
        reset to 'pending' and re-dispatched; an unconditional ``now()`` here
        erased the burnt attempt entirely, so ``completed_at - started_at``
        counted only the last attempt AND the burnt one rendered as dead air
        between the predecessor's completion and this stage's start. That
        misread is what nightly action_reel run 01a01d41 looked like: a 1935s
        segment_select attempt reported as a 1935s dispatch stall with nothing
        running."""
        await self.session.execute(
            update(StageOutput)
            .where(
                StageOutput.run_id == run_id,
                StageOutput.stage_name == stage_name,
                StageOutput.status.in_(_NON_TERMINAL),
            )
            .values(
                status="running",
                started_at=func.coalesce(StageOutput.started_at, func.now()),
                # UNCONDITIONAL, unlike ``started_at`` above — this is the OTHER
                # half of the pair, and the two must not be conflated. Every entry
                # into 'running' begins a new EXECUTION of the handler, so this is
                # re-stamped each time, and ``completed_at - last_execution_started_at``
                # is a per-execution runtime that can legitimately be compared with a
                # stage ceiling. ``completed_at - started_at`` cannot: it spans burnt
                # executions, re-dispatch gaps and pod downtime by design.
                last_execution_started_at=func.now(),
            )
        )

    async def mark_pending(self, run_id: UUID, stage_name: str) -> None:
        """Reset a stage row back to 'pending' so it can be re-executed.

        Used when a stage is cancelled mid-flight (pod drain / pause / cancel)
        or hits a resumable timeout: the row must not be left stuck in
        'running'. A resumed run rebuilds its completed-set from 'completed'
        rows only, so a 'pending' stage is re-executed cleanly. Only resets rows
        that are currently 'running' so a late cancellation can never clobber a
        stage that already completed.

        ``started_at`` is deliberately PRESERVED (``attempts`` already was). The
        stage is awaiting resume, not un-started, and the abandoned attempt is
        usually the largest block of wall clock the run spent — nulling it here
        (paired with an unconditional ``now()`` in :meth:`mark_running`) deleted
        the only record that the attempt happened, leaving a stage table in
        which a burnt 32-minute attempt is indistinguishable from a scheduler
        that never dispatched anything. ``completed_at``/``error`` still clear:
        those describe an outcome the stage no longer has.

        ``last_execution_started_at`` is likewise preserved: it dates the execution
        that was just abandoned, which is exactly the thing an audit wants to see.
        The next :meth:`mark_running` re-stamps it for the new execution.

        This method is reached from BOTH reset paths — a resumable timeout and a
        cancellation — and cannot tell them apart, which is why it does not bump a
        counter itself. The callers record which one happened:
        :meth:`increment_attempts` for a timeout, :meth:`increment_interruptions`
        for a drain.
        """
        await self.session.execute(
            update(StageOutput)
            .where(
                StageOutput.run_id == run_id,
                StageOutput.stage_name == stage_name,
                StageOutput.status == "running",
            )
            .values(status="pending", completed_at=None, error=None)
        )

    async def record_progress(self, run_id: UUID, stage_name: str, output: dict[str, Any]) -> None:
        """Persist in-flight bookkeeping into a stage's output JSONB WITHOUT
        changing its status.

        The one durable channel a stage has for state that must survive a
        resumable-timeout re-dispatch: ``mark_pending`` resets status/timestamps
        but deliberately leaves ``output`` intact, and the stage can read it back
        with :meth:`get` on its next attempt. Used to re-attach to work already
        submitted downstream instead of re-submitting it — sub-DAG child run_ids
        (``course_compile_lessons``) and the in-flight html_render design job
        (``web_design``), both of which are expensive and non-idempotent to
        re-issue.

        Overwritten by ``mark_completed`` with the real stage output once the
        stage succeeds. Guarded to a non-terminal row so a resume that
        re-attaches can never write this bookkeeping OVER a completed row's real
        output. NOTE: this REPLACES ``output`` wholesale — a caller with several
        keys must pass the merged dict.
        """
        await self.session.execute(
            update(StageOutput)
            .where(
                StageOutput.run_id == run_id,
                StageOutput.stage_name == stage_name,
                StageOutput.status.in_(_NON_TERMINAL),
            )
            .values(output=output)
        )

    async def increment_attempts(self, run_id: UUID, stage_name: str) -> int:
        """Atomically bump the stage's timeout-resume counter; return the new value.

        Durability P1: a genuine timeout increments this before deciding whether
        to resume (attempts <= cap → leave the run running for the reconciler) or
        terminal-fail. Uses ``attempts = attempts + 1 ... RETURNING`` so concurrent
        writers can't lose a bump. Returns 0 if the row is unexpectedly absent
        (mark_running seeds it first, so this is defensive) — a 0 sorts below any
        positive cap, so the stage still gets its resume rather than a spurious
        terminal fail on a missing row.

        That defensive 0 is now an INVARIANT BREACH, not a benign fallback:
        ``runs.run_setup.reconcile_blueprint_stage_rows`` guarantees a row exists
        for every blueprint stage before dispatch. Because ``0 <= cap`` for any
        positive cap, a missing row silently grants an UNBOUNDED resume loop
        instead of failing terminally. The return type and behaviour are
        deliberately unchanged — raising here would abort live dispatch into the
        generic failure path — so the breach is surfaced as a ``logger.warning``
        naming the run and stage instead.

        Guarded to ``status IN _NON_TERMINAL`` for symmetry with every sibling
        writer: a terminal row — in particular one already swept to 'cancelled'
        by :meth:`bulk_cancel_non_terminal` — must stay unwritable, so a stray
        bump can no longer mutate a run that is already terminal-failed. This is
        LATENT hardening, not a live bug fix: the sole caller
        (``run_dispatcher.bump_attempts``) runs strictly before the run-failure
        sweep, so no reachable path bumps a cancelled row today. Consequence for
        the ``None`` branch: it now means the row was absent OR already terminal,
        so the warning covers both.
        """
        result = await self.session.execute(
            update(StageOutput)
            .where(
                StageOutput.run_id == run_id,
                StageOutput.stage_name == stage_name,
                StageOutput.status.in_(_NON_TERMINAL),
            )
            .values(attempts=StageOutput.attempts + 1)
            .returning(StageOutput.attempts)
        )
        row = result.scalar_one_or_none()
        if row is None:
            logger.warning(
                "increment_attempts: no non-terminal stage_outputs row to bump — "
                "run_id=%s stage_name=%s; the row is absent (invariant breach) or "
                "already terminal. Returning 0, which grants an unbounded resume "
                "instead of a terminal fail",
                run_id,
                stage_name,
            )
            return 0
        return int(row)

    async def increment_interruptions(self, run_id: UUID, stage_name: str) -> int:
        """Atomically record that this stage was DRAINED (pod roll / pause / cancel);
        return the new count.

        The durable twin of :meth:`increment_attempts`. Both a genuine timeout and a
        cancellation reset the stage with :meth:`mark_pending`, so the row they leave
        behind is byte-identical — and only the timeout path bumped a counter. A
        drained stage therefore resumed with ``attempts == 0`` and an untouched
        ``started_at``, which reads as "ran once, cleanly, for the whole span". Since
        orchestrator pods roll constantly that is the COMMON case, which is why
        ``attempts == 0`` is a false filter for "this duration is trustworthy" and why
        this is recorded rather than inferred.

        ``coalesce(interruptions, 0) + 1`` because the column is nullable with no
        server default: pre-0067 rows carry NULL for "never recorded" and must not be
        silently converted into a 0 that claims they were never drained.

        Guarded to ``status IN _NON_TERMINAL`` for symmetry with every sibling writer,
        so a stray bump cannot mutate a row already swept terminal. Returns 0 when no
        row matched — this is bookkeeping, and raising here would abort a drain that
        is already unwinding.
        """
        result = await self.session.execute(
            update(StageOutput)
            .where(
                StageOutput.run_id == run_id,
                StageOutput.stage_name == stage_name,
                StageOutput.status.in_(_NON_TERMINAL),
            )
            .values(interruptions=func.coalesce(StageOutput.interruptions, 0) + 1)
            .returning(StageOutput.interruptions)
        )
        row = result.scalar_one_or_none()
        if row is None:
            logger.warning(
                "increment_interruptions: no non-terminal stage_outputs row to bump — "
                "run_id=%s stage_name=%s; the row is absent or already terminal, so this "
                "drain goes unrecorded and the stage's duration stays ambiguous",
                run_id,
                stage_name,
            )
            return 0
        return int(row)

    async def mark_completed(self, run_id: UUID, stage_name: str, output: dict[str, Any]) -> bool:
        """Mark a stage 'completed'; return whether a row was actually updated.

        ``False`` means the ``stage_outputs`` row for ``(run_id, stage_name)``
        was absent, so the write was a silent no-op. That is an invariant breach
        now that ``runs.run_setup.reconcile_blueprint_stage_rows`` guarantees a
        row exists for every blueprint stage before dispatch — a stage's result
        has been computed and then dropped on the floor. Callers must not change
        control flow on it; the no-op is warned about centrally in :func:`_wrote`
        so it stays observable even where a caller discards the bool.

        Guarded to a non-terminal row (``status IN _NON_TERMINAL``): the terminal
        write is write-once, so a racing/stale completion — or a resume
        re-issuing it on an already-terminal row — is a benign no-op. ``False``
        therefore now means EITHER the row was absent (invariant breach) OR it
        was already terminal (guard fired); both are log-only, no branching.
        """
        result = await self.session.execute(
            update(StageOutput)
            .where(
                StageOutput.run_id == run_id,
                StageOutput.stage_name == stage_name,
                StageOutput.status.in_(_NON_TERMINAL),
            )
            .values(status="completed", output=output, completed_at=func.now())
        )
        return _wrote(result, "mark_completed", run_id, stage_name)

    async def mark_failed(self, run_id: UUID, stage_name: str, error: str) -> bool:
        """Mark a stage 'failed'; return whether a row was actually updated.

        Guarded and self-reporting exactly like ``mark_completed``: the
        non-terminal guard makes the failure write-once so a stale duplicate can
        never flip an already-``completed`` stage to ``failed``, and ``False``
        (row absent OR already terminal) is log-only, never a control-flow
        branch — raising here would abort live dispatch.
        """
        result = await self.session.execute(
            update(StageOutput)
            .where(
                StageOutput.run_id == run_id,
                StageOutput.stage_name == stage_name,
                StageOutput.status.in_(_NON_TERMINAL),
            )
            .values(status="failed", error=error, completed_at=func.now())
        )
        return _wrote(result, "mark_failed", run_id, stage_name)

    async def mark_skipped(self, run_id: UUID, stage_name: str, reason: str) -> bool:
        """Mark a stage 'skipped' — an opt-in path the run did not request.

        Distinct from 'failed': nothing went wrong, the stage's precondition
        (e.g. an ``ai_image_gen`` config block) was simply absent. The
        completion gate counts 'skipped' alongside 'completed' as terminal, so
        a run that legitimately skips an optional stage still reaches
        'completed' instead of hanging in 'running'. The reason is stored in
        ``error`` for surfacing; the column is shared, the status disambiguates.

        Returns whether a row was actually updated. ``False`` means the
        ``stage_outputs`` row for ``(run_id, stage_name)`` was absent and the
        write was a silent no-op — an invariant breach now that
        ``runs.run_setup.reconcile_blueprint_stage_rows`` guarantees a row
        exists for every blueprint stage before dispatch. A missing row here
        also means the completion gate never sees the terminal 'skipped', so the
        run can hang in 'running'. Callers log it; they do not branch on it.

        Guarded to a non-terminal row for symmetry with the other terminal
        writers: a resume re-running a deterministically-'skipped' stage
        re-issues this on an already-'skipped' row as a harmless no-op.
        """
        result = await self.session.execute(
            update(StageOutput)
            .where(
                StageOutput.run_id == run_id,
                StageOutput.stage_name == stage_name,
                StageOutput.status.in_(_NON_TERMINAL),
            )
            .values(status="skipped", error=reason, completed_at=func.now())
        )
        return _wrote(result, "mark_skipped", run_id, stage_name)

    async def bulk_cancel_non_terminal(self, run_id: UUID, *, run_status: str = "failed") -> int:
        """Terminalize a terminated run's leftover stage rows; return rows swept.

        Invoked from ``RunRepository.mark_failed``/``mark_cancelled`` on the winning
        CAS (the chokepoint that makes the invariant unconditional) and directly by
        the reconciler's terminalizer for a run that went terminal on another pod,
        where no CAS is won here.

        ``run_status`` is the terminal run status that stranded these rows; it only
        selects the ``error`` prose (see :func:`stranded_stage_reason`). The reason
        is written UNCONDITIONALLY over any prior value: a non-terminal row's
        ``error`` is at best a stale in-flight note, and the operative fact an
        operator needs is that the RUN ended.
        When a stage hard-fails, the DAG executor cancels only the *current*
        ``asyncio.gather`` batch (resetting those siblings to 'pending'); any
        independent sibling branch or never-scheduled downstream stage is left
        'pending'/'running' under a now-terminal 'failed' run. Failed runs are
        never resumed (the write-once contract), so those rows strand forever —
        they pollute the ``stage_outputs`` group-by the stranded-stage detector
        relies on and surface as ghost 'pending' stages in the UI/SSE.

        Sweeping them to 'cancelled' — DB-terminal (admitted by migration 0059's
        widened ``ck_stage_outputs_status``) and guarded against re-sweep by
        ``_NON_TERMINAL``, but deliberately NOT counted by the run-completion gate
        (``run_finalizer.TERMINAL_STAGE_STATUSES`` is ``{'completed','skipped'}``),
        because the run is already terminal-failed and 'cancelled' means the stage
        never ran — is pure data hygiene: pipeline behaviour is unchanged, the run is
        already fail-fast terminal. Guarded to ``status IN _NON_TERMINAL`` for
        symmetry with the terminal writers — an already-'completed'/'failed'/
        'skipped' row is never clobbered.
        """
        result = await self.session.execute(
            update(StageOutput)
            .where(
                StageOutput.run_id == run_id,
                StageOutput.status.in_(_NON_TERMINAL),
            )
            .values(
                status="cancelled",
                error=stranded_stage_reason(run_status),
                completed_at=func.now(),
            )
        )
        return int(getattr(result, "rowcount", 0) or 0)
