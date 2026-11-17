from __future__ import annotations

import datetime
import logging
from collections.abc import Sequence
from typing import Any, Literal
from uuid import UUID, uuid7

from sqlalchemy import ColumnElement, Integer, Row, and_, cast, desc, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Deliverable, Project, Run, StageOutput
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.models.failure_class import UNKNOWN, FailureClass

#: Why a run is paused, as a WRITER records it (``runs.pause_reason``, 0102).
#: ``unknown`` exists only on rows 0102 backfilled; nothing writes it.
PauseReason = Literal["checkpoint", "manual"]

log = logging.getLogger(__name__)

# The JSON values that are FALSEY in Python, i.e. the ones for which
# ``bool((output or {}).get("notes"))`` is False even though the key is present.
# Listing them (rather than testing ``jsonb_array_length > 0``) keeps the SQL
# derivation semantically IDENTICAL to the Python one in
# ``api/routes/runs/_shared.py::_stage_response`` for EVERY JSON type, not just
# arrays — so the detail view and the list/lite views can never disagree.
_FALSEY_NOTES: tuple[object, ...] = ([], {}, "", False, 0)

#: ``runs.failure_detail.code`` on a sub-DAG child (course lesson) that the PARENT's
#: cascade cancelled — as opposed to one the USER cancelled, which carries no
#: detail. ``course_compile_lessons`` re-spawns the former on a course retry and
#: keeps the latter terminal. Written only by :meth:`RunRepository.mark_cancelled_by_parent`.
CANCELLED_BY_PARENT: str = "cancelled_by_parent"


def cancelled_by_parent(run: Run) -> bool:
    """True when ``run`` is a child the parent cascade cancelled (see above)."""
    detail = run.failure_detail or {}
    return run.status == "cancelled" and detail.get("code") == CANCELLED_BY_PARENT


def _truthy_json_key(key: str) -> ColumnElement[bool]:
    """``bool((output or {}).get(key))``, evaluated in Postgres.

    ``output -> key`` is SQL NULL when ``output`` is NULL or the key is absent
    (every row written before that key existed), and ``jsonb_typeof(NULL)`` is
    NULL, so both cases fall out of the ``NOT IN`` (NULL is not true) — the same
    False the Python expression produces. A jsonb ``'null'`` is excluded
    explicitly because it would otherwise survive the value list.
    """
    value = StageOutput.output[key]
    return and_(
        func.jsonb_typeof(value).notin_(("null",)),
        value.notin_(_FALSEY_NOTES),
    )


def degraded_stage_clause() -> ColumnElement[bool]:
    """SQL predicate for "this ``stage_outputs`` row soft-failed downstream work".

    SQL twin of ``services/stages/shared/degradation.py::is_degraded`` (used by
    ``api/routes/runs/_shared.py::_stage_response``), which the detail view
    applies in Python: ``status == "completed"`` AND either channel non-empty —
    the explicit machine-readable ``degradations`` list OR the legacy prose
    ``notes`` list. The two definitions must be changed together or the detail
    view and the list/lite views disagree about the same run.

    Evaluated entirely in Postgres against the ``output`` JSONB — the blob is
    never selected, so callers on the cheap list/lite paths get the flag without
    shipping (or even reading out) multi-MB stage payloads.
    """
    return and_(
        StageOutput.status == "completed",
        or_(_truthy_json_key("degradations"), _truthy_json_key("notes")),
    )


class RunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        deliverable_id: UUID,
        blueprint_id: str,
        config_snapshot: dict[str, Any],
        pause_after_stages: list[str],
        parent_run_id: UUID | None = None,
        structural_relief_pass: int | None = None,
        seeded_pause_after_stages: list[str] | None = None,
    ) -> Run:
        run = Run(
            id=uuid7(),
            deliverable_id=deliverable_id,
            parent_run_id=parent_run_id,
            structural_relief_pass=structural_relief_pass,
            blueprint_id=blueprint_id,
            status="pending",
            current_stage=None,
            config_snapshot=config_snapshot,
            pause_after_stages=pause_after_stages,
            seeded_pause_after_stages=seeded_pause_after_stages,
        )
        self.session.add(run)
        await self.session.flush()
        return run

    async def get(self, run_id: UUID) -> Run | None:
        return (
            await self.session.execute(select(Run).where(Run.id == run_id))
        ).scalar_one_or_none()

    async def statuses(self, run_ids: Sequence[UUID]) -> dict[UUID, str]:
        """``{run_id: status}`` for every READABLE run in ``run_ids``, in ONE query.

        A run absent from the result is unreadable (never written, or outside the
        session's RLS tenant) — callers decide what that means. Selects only the
        two columns, never whole rows: the sub-DAG wait polls this every couple of
        seconds for every open child.
        """
        if not run_ids:
            return {}
        rows = await self.session.execute(
            select(Run.id, Run.status).where(Run.id.in_(list(run_ids)))
        )
        return {rid: str(status) for rid, status in rows.all()}

    async def list_for_deliverable(self, deliverable_id: UUID) -> list[Run]:
        # Tie-break on id (uuid7 = time-ordered) so two runs created in the same
        # millisecond have a deterministic order — created_at alone flaked in CI.
        stmt = (
            select(Run)
            .where(Run.deliverable_id == deliverable_id)
            .order_by(desc(Run.created_at), desc(Run.id))
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def list_active_children(self, parent_run_id: UUID) -> list[Run]:
        """Runs spawned from ``parent_run_id`` (e.g. structural-relief children)
        that are not yet terminal.

        Feeds the cancel cascade: a ``cant_converge`` parent spawns a bounded
        child render, so cancelling the parent must also interrupt any child (or
        grandchild) still rendering, or we leak an orphaned PAID compile. The
        ``notin_(_TERMINAL)`` predicate intentionally also catches ``paused``.
        """
        result = await self.session.execute(
            select(Run).where(Run.parent_run_id == parent_run_id, Run.status.notin_(self._TERMINAL))
        )
        return list(result.scalars().all())

    async def list_for_project(
        self,
        project_id: UUID,
        *,
        tenant_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Run]:
        # All runs of a project, across every deliverable, via the
        # Run→Deliverable→Project join (tenant-scoped on Project, same shape as
        # runs.py::_tenant_owns_run). Newest-first with the uuid7 id tie-break —
        # created_at alone flaked in CI (see list_for_deliverable).
        stmt = (
            select(Run)
            .join(Deliverable, Deliverable.id == Run.deliverable_id)
            .join(Project, Project.id == Deliverable.project_id)
            .where(Project.id == project_id, Project.tenant_id == tenant_id)
            .order_by(desc(Run.created_at), desc(Run.id))
            .limit(limit)
            .offset(offset)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def count_for_project(self, project_id: UUID, *, tenant_id: UUID) -> int:
        """Total runs of a project — the denominator ``list_for_project`` pages over.

        Same tenant-scoped join as :meth:`list_for_project`, so the count can never
        include a row the list could not return.
        """
        stmt = (
            select(func.count(Run.id))
            .join(Deliverable, Deliverable.id == Run.deliverable_id)
            .join(Project, Project.id == Deliverable.project_id)
            .where(Project.id == project_id, Project.tenant_id == tenant_id)
        )
        return int((await self.session.execute(stmt)).scalar_one())

    async def list_active_for_project(self, project_id: UUID, *, tenant_id: UUID) -> list[Run]:
        """Every run of a project that can still spend: pending, running or PAUSED.

        Paused is included on purpose — a paused run resumes into paid stages, so a
        project delete must stop it too (UI review 2026-10-17, "cancel on delete").
        Unbounded: a project has a handful of active runs, never a page of them.
        """
        stmt = (
            select(Run)
            .join(Deliverable, Deliverable.id == Run.deliverable_id)
            .join(Project, Project.id == Deliverable.project_id)
            .where(
                Project.id == project_id,
                Project.tenant_id == tenant_id,
                Run.status.in_((*self._NON_TERMINAL, "paused")),
            )
            .order_by(Run.created_at, Run.id)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def degraded_run_ids(self, run_ids: Sequence[UUID]) -> set[UUID]:
        """The subset of ``run_ids`` that finished green but degraded.

        ONE aggregate query for the whole page (no N+1, no blob load): it selects
        ``run_id`` only, with :func:`degraded_stage_clause` pushed into the WHERE,
        so the list endpoints can flag "completed with warnings" at the cost of a
        single indexed scan over ``idx_stage_outputs_run``.
        """
        if not run_ids:
            return set()
        stmt = (
            select(StageOutput.run_id)
            .where(StageOutput.run_id.in_(run_ids), degraded_stage_clause())
            .distinct()
        )
        return set((await self.session.execute(stmt)).scalars().all())

    async def degraded_codes_by_run(self, run_ids: Sequence[UUID]) -> dict[UUID, list[str]]:
        """WHICH degradation codes each of ``run_ids`` shipped with.

        The by-code companion to :meth:`degraded_run_ids`. That method answers
        "was this run degraded?"; nine distinct codes exist
        (``services/stages/shared/degradation.py::DEGRADATION_CODES``), all nine
        are emitted, and until this method nothing could ask WHICH — so
        "which runs shipped without their music bed last week" had no answer
        short of loading every stage blob and grepping it.

        Same shape and same cost as :meth:`degraded_run_ids`: one aggregate
        query over ``idx_stage_outputs_run`` with :func:`degraded_stage_clause`
        pushed into the WHERE. The multi-MB ``output`` payload is NEVER
        selected — only the ``degradations`` SUBTREE is projected out of the
        JSONB in Postgres, so the wire cost is proportional to the number of
        degradations, not the size of the stage outputs.

        Ordering is deterministic — ``started_at`` then the uuid7 ``id`` as the
        tie-break (the same tie-break ``list_for_deliverable`` needs, for the
        same reason: two rows in one millisecond flaked in CI) — and codes are
        de-duplicated per run in first-seen order, matching
        ``api/routes/runs/_shared.py::_degradation_codes``.

        Runs with no degradation are ABSENT from the mapping rather than mapped
        to ``[]``, so ``len(result)`` is the degraded count and the keys are a
        superset of nothing. Callers wanting a total should use
        ``.get(run_id, [])``.

        ⚠️ REPORTING ONLY. Nothing here fails, alerts on, or gates a run: eight
        of the nine codes have no measured base rate, and this repo has already
        shipped one signal that fired on 96.7% of masters. Arming any of these
        is a separate, evidence-backed decision.
        """
        if not run_ids:
            return {}
        stmt = (
            select(StageOutput.run_id, StageOutput.output["degradations"].label("degradations"))
            .where(StageOutput.run_id.in_(run_ids), degraded_stage_clause())
            .order_by(StageOutput.started_at.nulls_last(), StageOutput.id)
        )
        by_run: dict[UUID, list[str]] = {}
        for run_id, entries in (await self.session.execute(stmt)).all():
            if not isinstance(entries, list):
                # A notes-only degraded stage: the legacy prose channel has no
                # stable identity to report. It still counts as degraded (that is
                # `degraded_run_ids`' job) — it just has no code, so it must not
                # manufacture one.
                continue
            codes = by_run.setdefault(run_id, [])
            for entry in entries:
                # Tolerant on read for the same reason `degradation.code` is typed
                # `str` and not a `Literal`: a rolling deploy can land a newer
                # pod's code in a row this build reads back, and rejecting it
                # would break the endpoint rather than report the fact.
                if not isinstance(entry, dict):
                    continue
                code = entry.get("code")
                if isinstance(code, str) and code != "" and code not in codes:
                    codes.append(code)
            if not codes:
                del by_run[run_id]
        return by_run

    async def stage_statuses(
        self, run_id: UUID
    ) -> Sequence[
        Row[tuple[str, str, datetime.datetime | None, datetime.datetime | None, int, bool]]
    ]:
        """Per-stage lifecycle state + degraded flag for the live-monitor poll.

        A PROJECTION, deliberately not ``StageOutputRepository.list_for_run``:
        that selects whole ORM rows including each stage's ``output`` JSONB, which
        the lite endpoint then throws away. Here the degraded bit is computed in
        Postgres and only the five scalar columns come back, so the poll's cost
        stops growing with the size of the stage payloads.
        """
        stmt = select(
            StageOutput.stage_name,
            StageOutput.status,
            StageOutput.started_at,
            StageOutput.completed_at,
            StageOutput.attempts,
            degraded_stage_clause().label("degraded"),
        ).where(StageOutput.run_id == run_id)
        return (await self.session.execute(stmt)).all()

    async def mark_started(self, run_id: UUID) -> bool:
        # COALESCE so started_at is stamped once, on the FIRST execution slice.
        # mark_started runs at the top of every _execute_run — including resume
        # and reconciler re-dispatch — so an unconditional now() overwrote the
        # original start and made `completed_at - started_at` report only the
        # final slice, under-counting a paused/resumed run's real duration.
        #
        # `_NON_TERMINAL` is the CAS the comment block below claims EVERY status
        # mutator carries; mark_started was the one that did not, and it is the
        # one dispatch calls on every resume/reconciler re-dispatch. Without it a
        # cancel (or pause) that lands between the reconciler's orphan SELECT and
        # this write is silently reverted: the row goes `cancelled` → `running`
        # and the run bills to completion, or a `paused` row is flipped back to
        # `running` and sails past its human-review gate. Restricting to
        # ("running", "pending") — rather than merely excluding the terminal
        # states — is what closes the paused half; a legitimate resume has
        # already been CAS'd to `running` by `mark_resumed`.
        #
        # Returns whether this dispatch may proceed, like every sibling mutator.
        result = await self.session.execute(
            update(Run)
            .where(Run.id == run_id, Run.status.in_(self._NON_TERMINAL))
            .values(status="running", started_at=func.coalesce(Run.started_at, func.now()))
        )
        return int(getattr(result, "rowcount", 0) or 0) > 0

    async def mark_current_stage(self, run_id: UUID, stage_name: str) -> None:
        await self.session.execute(
            update(Run).where(Run.id == run_id).values(current_stage=stage_name)
        )

    # Terminal-state guards on mark_paused/mark_cancelled close a TOCTOU race:
    # the cancel/pause routes check status against a snapshot read at request
    # start, then write unconditionally. If the run reached a terminal state on
    # another pod in between, an unguarded write would flip e.g. `completed` →
    # `paused`/`cancelled` (a later resume would then re-run a finished run).
    # The predicate makes the write a no-op once the run is terminal.
    _NON_TERMINAL = ("running", "pending")
    # Terminal states are write-once: the first pod to reach one wins. Every
    # status mutator guards against overwriting a terminal state so a
    # concurrent cancel on another pod is never clobbered by a late
    # complete/fail, and a finished run can never be resurrected by resume.
    _TERMINAL = ("completed", "cancelled", "failed")

    async def _terminalize_stage_rows(self, run_id: UUID, *, run_status: str) -> int:
        """Sweep this run's not-yet-executed stage rows terminal; return rows swept.

        THE INVARIANT: once a run reaches a status it can never leave, no stage of
        it may still read ``pending`` — ``pending`` means "this stage will run",
        and for a cancelled/failed run that is false and permanently false. A
        measured population on 2026-09-08 held 1066 such rows across 131 terminal
        runs (939 under 89 ``cancelled`` runs, 127 under 42 ``failed``), because
        the sweep lived at two of the failure CALL SITES rather than at the
        transition itself, and every cancel route bypassed it.

        Placing it HERE — inside the compare-and-set, in the same transaction —
        is what makes it unconditional: any present or future caller of
        ``mark_failed``/``mark_cancelled`` gets the sweep, because the sweep is
        part of what the transition MEANS. Gated on the CAS actually winning: a
        losing racer must not sweep rows a concurrent worker is legitimately
        resuming.

        ``mark_paused`` deliberately does NOT call this — a paused run is
        resumable, so its ``pending`` stages are telling the truth.
        """
        swept = await StageOutputRepository(self.session).bulk_cancel_non_terminal(
            run_id, run_status=run_status
        )
        if swept:
            log.info(
                "run %s -> %s: swept %d stranded stage row(s) to 'cancelled'",
                run_id,
                run_status,
                swept,
            )
        return swept

    async def mark_paused(self, run_id: UUID, *, reason: PauseReason) -> bool:
        """Pause a non-terminal run, recording WHY.

        ``reason`` is ``'checkpoint'`` when a review gate tripped (a person must
        approve the stage output — the only kind that is "needs review") or
        ``'manual'`` for ``POST /runs/{id}/pause``. Keyword-
        only and required, so no caller can pause a run without saying which.
        """
        result = await self.session.execute(
            update(Run)
            .where(Run.id == run_id, Run.status.in_(self._NON_TERMINAL))
            # Start the pause clock (COALESCE: never move an existing stamp).
            # Folded into ``paused_seconds`` by ``mark_resumed`` so the run
            # deadline does not charge review time to the run's budget.
            .values(
                status="paused",
                paused_at=func.coalesce(Run.paused_at, func.now()),
                pause_reason=reason,
            )
        )
        return int(getattr(result, "rowcount", 0) or 0) > 0

    async def mark_resumed(self, run_id: UUID, pause_after_stages: list[str] | None = None) -> bool:
        # Only a paused run may resume. Guarding on `paused` makes the write
        # atomic with the route's read-side `status != "paused"` check, so a
        # cancel/complete/fail landing on another pod between that read and
        # this write turns resume into a no-op instead of resurrecting a
        # terminal run back to `running`.
        #
        # ``pause_after_stages`` (when given) is rewritten in the SAME atomic
        # update: the resume route strips every checkpoint the human just
        # approved past (already-consumed pause-stages), making the row itself
        # the durable approval ledger. A later reconciler re-dispatch (e.g. a
        # drain mid-downstream) then sees the cleared list and does NOT re-pause
        # at an already-approved gate. Passing None leaves the list untouched.
        #
        # Returns whether the CAS actually won, like every sibling mutator here.
        # It used to return None, so the caller had no way to tell a resume from
        # a no-op and answered 204 either way -- and then dispatched the run
        # regardless. The route's read-side check makes that a narrow race
        # rather than the common path, but in that race the row stays terminal
        # while `_execute_run` is spawned against it, and nothing downstream
        # re-checks the status on entry.
        #
        # The pause clock is folded in by the SAME compare-and-set: the time the
        # row sat paused is added to ``paused_seconds`` and ``paused_at`` cleared,
        # so the run deadline (``started_at + budget + paused_seconds``) does not
        # count a human review against the run. A row paused before migration
        # 0094 has ``paused_at`` NULL and contributes 0 — the old behaviour.
        paused_for = func.coalesce(
            cast(func.floor(func.extract("epoch", func.now() - Run.paused_at)), Integer),
            0,
        )
        values: dict[str, Any] = {
            "status": "running",
            "paused_seconds": Run.paused_seconds + func.greatest(paused_for, 0),
            "paused_at": None,
            # This pause is over; so are its reason and its notice's dismissal (0104),
            # so the run's next checkpoint is a fresh notice.
            "pause_reason": None,
            "review_notice_dismissed_at": None,
        }
        if pause_after_stages is not None:
            values["pause_after_stages"] = pause_after_stages
        result = await self.session.execute(
            update(Run).where(Run.id == run_id, Run.status == "paused").values(**values)
        )
        return int(getattr(result, "rowcount", 0) or 0) > 0

    async def mark_completed(self, run_id: UUID) -> bool:
        result = await self.session.execute(
            update(Run)
            .where(Run.id == run_id, Run.status.notin_(self._TERMINAL))
            .values(status="completed", completed_at=func.now(), current_stage=None)
        )
        return int(getattr(result, "rowcount", 0) or 0) > 0

    async def mark_failed(
        self,
        run_id: UUID,
        error: str,
        *,
        failure_class: FailureClass = UNKNOWN,
        failed_stage: str | None = None,
        failure_detail: dict[str, str] | None = None,
    ) -> bool:
        """Terminal-fail a run, recording WHOSE fault it was alongside the prose.

        ``failed_stage`` (when known) re-points ``current_stage`` at the stage
        that FAILED, in the same compare-and-set. Until now ``current_stage`` was
        only ever written at stage START, so a wave that ran its stages
        concurrently left it naming whichever sibling started LAST — on prod run
        01a0a054 a failed ``website_review_record`` reported
        ``current_stage='website_review_visual_director'``, a sibling that
        COMPLETED. ``None`` leaves the column untouched (the caller does not know
        which stage failed, and guessing is worse than the stale value).

        ``failure_class`` is written in the SAME statement as ``status='failed'``
        because this method is the single chokepoint for that transition — every
        run-level failure route in ``run_dispatcher`` funnels through here — so
        an attributed run and an unattributed one can never diverge, and the
        ``ck_runs_failure_class_only_on_failed`` CHECK holds by construction.

        ⚠️ The default is ``UNKNOWN``, deliberately, and it is what makes this
        safe to roll out incrementally: a caller that has no evidence (or a
        future caller nobody remembered to update) records "not attributed"
        rather than inheriting somebody else's bucket. A wrong attribution is
        worse than an absent one — it sends an engineer hunting a bug that does
        not exist. There is no code path here that infers a class.
        """
        values: dict[str, object] = {
            "status": "failed",
            "error": error,
            "failure_class": failure_class,
            # The structured what-to-do (e.g. ``byo_provider_required``) beside
            # the prose; NULL when the failure carried none.
            "failure_detail": failure_detail,
            "completed_at": func.now(),
        }
        if failed_stage is not None:
            values["current_stage"] = failed_stage
        result = await self.session.execute(
            update(Run).where(Run.id == run_id, Run.status.notin_(self._TERMINAL)).values(**values)
        )
        transitioned = int(getattr(result, "rowcount", 0) or 0) > 0
        if transitioned:
            await self._terminalize_stage_rows(run_id, run_status="failed")
        return transitioned

    async def mark_cancelled_by_parent(self, run_id: UUID, parent_run_id: UUID) -> bool:
        """Record that ``run_id`` was cancelled by its PARENT's cascade, not a user.

        Called by both sub-DAG cascades (``RunDispatcher._cancel_subdag_children``
        and the cancel route's ``_sweep_deliverable_children``) right after they
        cancel a non-terminal child. Guarded on ``status='cancelled'`` (a child
        that raced to ``failed``/``completed`` is left alone) and on an absent
        detail (never overwrite). Without this a cascade-cancelled lesson and a
        user-cancelled one are indistinguishable rows.
        """
        result = await self.session.execute(
            update(Run)
            .where(Run.id == run_id, Run.status == "cancelled", Run.failure_detail.is_(None))
            .values(
                failure_detail={
                    "code": CANCELLED_BY_PARENT,
                    "message": f"Cancelled because its parent run {parent_run_id} ended",
                }
            )
        )
        return int(getattr(result, "rowcount", 0) or 0) > 0

    async def mark_cancelled(self, run_id: UUID) -> bool:
        # Cancel may apply to a paused run too, so the guard excludes only the
        # already-terminal states rather than restricting to _NON_TERMINAL.
        result = await self.session.execute(
            update(Run)
            .where(Run.id == run_id, Run.status.notin_(self._TERMINAL))
            .values(status="cancelled", completed_at=func.now())
        )
        transitioned = int(getattr(result, "rowcount", 0) or 0) > 0
        if transitioned:
            await self._terminalize_stage_rows(run_id, run_status="cancelled")
        return transitioned
