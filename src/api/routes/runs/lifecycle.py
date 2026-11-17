"""Run lifecycle-control routes: cancel, pause, resume, checkpoints, regenerate,
reorder-clips, regenerate-clip."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import HTTPException, Request, status
from sqlalchemy import update

from src.api.routes._common import (
    SessionDep,
    SubjectIdDep,
    TenantIdDep,
)
from src.api.schemas import (
    ApplyClipEdits,
    CheckpointsResponse,
    CheckpointsUpdate,
    MusicEdit,
    RegenerateClip,
    RegenerateRequest,
    Renarrate,
    ReorderClips,
    RestartPlanReason,
    RestartPlanResponse,
    RestartPlanStage,
    RunSummary,
)
from src.database.models import Run
from src.database.repositories.annotation_repository import AnnotationRepository
from src.database.repositories.deliverable_repository import DeliverableRepository
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.services.blueprints.registry import get_registry
from src.services.compile_cancel import cancel_compiles_for_run
from src.services.object_key_guard import (
    KeyConfinementError,
    decode_ref_path,
    key_segments,
    normalize_key,
)
from src.services.run_engine import (
    EditDirective,
    RegenClipDirective,
    RenarrateDirective,
    ReorderDirective,
    RestartDirective,
)
from src.services.runs.elements import (
    compile_cut_stage,
    music_rewind_stage,
    stage_supports_element_pinning,
)
from src.services.runs.regenerate import (
    EditError,
    RegenClipError,
    RegenerateError,
    RegenerateOrderMismatchError,
    RegenerateScopeError,
    RenarrateError,
    ReorderError,
)
from src.services.runs.restart_plan import (
    PLANNABLE_PARENT_STATUSES,
    Reason,
    RestartPlan,
    StagePlan,
    compute_restart_plan,
)
from src.services.runs.resume_inputs import RESUME_INPUTS_KEY, build_resume_inputs
from src.services.runs.run_finalizer import terminal_stage_names
from src.services.runs.run_setup import build_config_snapshot
from src.services.runs.sub_dag import TerminalRunStatus, subdag_child_run_ids
from src.services.stages.action_reel.music_override import MUSIC_OVERRIDE_KEY
from src.services.stages.sound_catalog import SoundLibrary, load_sound_library

from ._shared import (
    _project_id_for_deliverable,
    _recompute_project_status_for_run,
    _require_dispatcher,
    _summary,
    _tenant_owns_run,
    router,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.run_dispatcher import RunDispatcher

# Depth cap on the cancel cascade's descendant walk. A cant_converge parent
# spawns bounded relief children (N<=2 in practice); the cap + a visited-set
# guarantee the walk terminates even though parent_run_id is not constrained
# acyclic at write time (mirrors the ancestor-walk guard in _shared).
_MAX_CANCEL_CASCADE_DEPTH = 3

_log = logging.getLogger(__name__)

#: ``error.details.code`` of the 409 ``POST /runs/{id}/resume`` answers for any run
#: that is not ``paused`` (failed, cancelled, completed, running, pending).
RUN_NOT_RESUMABLE = "run_not_resumable"


async def _collect_active_descendants(repo: RunRepository, root_id: UUID) -> list[Run]:
    """Breadth-first walk of the non-terminal runs spawned (transitively) from
    ``root_id`` — the structural-relief child, its child, ….

    Collected in one pass BEFORE any cancellation so every descendant is read
    while still active (``expire_on_commit=False`` keeps the loaded attributes
    valid across the per-run commits the cancel does afterwards). A ``visited``
    set plus ``_MAX_CANCEL_CASCADE_DEPTH`` bound a pathological parent-cycle.
    """
    visited: set[UUID] = {root_id}
    out: list[Run] = []
    frontier: list[UUID] = [root_id]
    depth = 0
    while frontier and depth < _MAX_CANCEL_CASCADE_DEPTH:
        next_frontier: list[UUID] = []
        for parent_id in frontier:
            for child in await repo.list_active_children(parent_id):
                if child.id in visited:
                    continue
                visited.add(child.id)
                out.append(child)
                next_frontier.append(child.id)
        frontier = next_frontier
        depth += 1
    return out


async def _sweep_active_descendants(
    repo: RunRepository,
    root_id: UUID,
    cancel: Callable[[Run], Awaitable[object]],
) -> None:
    """Cancel every active run spawned (transitively) from ``root_id``, re-collecting
    after each pass so a child spawned in the cancel race window (the reconciler-
    finalize spawn) is still caught. ``cancel`` is idempotent on an already-terminal
    run, so re-cancelling is a harmless no-op. Bounded by ``_MAX_CANCEL_CASCADE_DEPTH``
    passes; exits as soon as a pass finds no active descendant.
    """
    for _ in range(_MAX_CANCEL_CASCADE_DEPTH):
        descendants = await _collect_active_descendants(repo, root_id)
        if not descendants:
            return
        for descendant in descendants:
            await cancel(descendant)


async def _cancel_one(session: AsyncSession, dispatcher: RunDispatcher, run: Run) -> None:
    """Cancel a SINGLE run: live-interrupt via the dispatcher, else mark the row
    cancelled + recompute the owning project's status + broadcast a cross-pod
    interrupt.

    Extracted verbatim from the original ``cancel_run`` fallback so the parent's
    behaviour is unchanged; the cascade reuses it per active descendant. Commit
    BEFORE broadcasting so whichever replica owns the task reads ``cancelled`` in
    its CancelledError handler. The recompute is idempotent (safe if a remote
    owner also recomputes) and covers the never-dispatched / truly-orphaned run
    the dispatcher's own transition handler would miss.

    Stopping the orchestrator's task is only HALF a cancel: the run's compile is a job on
    another service that keeps rendering — and keeps costing — until it finishes on its
    own. ``cancel_compiles_for_run`` is that second half. It runs on BOTH branches
    (a same-pod task cancel and the mark-cancelled fallback), because the compile is in
    flight either way, and it is best-effort by construction so a down compiler cannot
    strand this run in ``cancelling``.
    """
    if not await dispatcher.cancel(run.id):
        await RunRepository(session).mark_cancelled(run.id)
        await _recompute_project_status_for_run(session, run.deliverable_id)
        await session.commit()
        await dispatcher.broadcast_interrupt(run.id)
    await cancel_compiles_for_run(run.id)


async def _sweep_deliverable_children(
    session: AsyncSession,
    repo: RunRepository,
    root_id: UUID,
    cancel: Callable[[Run], Awaitable[object]],
) -> None:
    """Cancel every non-terminal PAID child RUN a sub-DAG stage
    (``course_compile_lessons``) spawned under ``root_id``.

    These children are spawned with ``parent_run_id=None`` and tracked only in the
    stage output's ``children_by_deliverable`` map (see
    ``sub_dag.subdag_child_run_ids``), so the FK-based ``_sweep_active_descendants``
    cascade cannot see them — without this a user who cancels a course run keeps
    paying for every in-flight lesson render. ``cancel`` is idempotent on an
    already-terminal run; a child already terminal is skipped. ``session`` is the
    request's tenant-bound session (the children share the tenant). Non-course runs
    have no such stage output → a harmless no-op.
    """
    for child_id in await subdag_child_run_ids(session, root_id):
        child = await repo.get(child_id)
        if child is None or child.status in TerminalRunStatus:
            continue
        await cancel(child)
        # Provenance: a course retry re-spawns a CASCADE-cancelled lesson but keeps
        # a USER-cancelled one terminal (course_compile_lessons).
        await repo.mark_cancelled_by_parent(child_id, root_id)


async def cancel_run_tree(
    session: AsyncSession, dispatcher: RunDispatcher | None, run: Run
) -> bool:
    """Cancel ``run`` and every run it spawned; ``False`` ⇒ degraded (no live interrupt).

    The whole of ``POST /runs/{id}/cancel`` after its ownership check, extracted so a
    project delete stops spend through the SAME path (UI review 2026-10-17: deleting
    a project only archived it and its runs kept rendering — and billing). ``False``
    means the cancel was RECORDED (rows marked cancelled, compiles cancelled,
    committed) but no live cross-pod interrupt could be delivered because the
    dispatcher is down; the caller answers 503 so the client retries.
    """
    repo = RunRepository(session)
    run_id = run.id
    # The descendant cascade runs REGARDLESS of the parent's own status. A
    # cant_converge parent spawns its bounded structural-relief child only AFTER
    # it COMPLETES, so a `completed` (terminal) parent can still own a running —
    # PAID — child render that must be interrupted, or it becomes an orphan.
    parent_terminal = run.status in ("completed", "cancelled", "failed")

    if dispatcher is None:
        # Degraded mode (no dispatcher → Redis down): we can't deliver a live
        # cross-pod interrupt, but best-effort record the cancel on the parent
        # AND every active descendant so nothing is wedged `running` in the UI;
        # it converges on caller retry or reconciler sweep. mark_cancelled is a
        # terminal-guarded no-op on an already-terminal parent, so for a
        # completed cant_converge parent the descendant sweep is what matters.
        # The compile-cancel does NOT go through the dispatcher — it is a plain HTTP call
        # to another service — so it still works with Redis down. This branch is in fact
        # where it matters most: no live interrupt is possible, so without it the run's
        # render is guaranteed to run to completion and be billed.
        async def _degraded_cancel(target: Run) -> None:
            await repo.mark_cancelled(target.id)
            await cancel_compiles_for_run(target.id)

        if not parent_terminal:
            await repo.mark_cancelled(run_id)
            await cancel_compiles_for_run(run_id)
        # Same race guard as the live path; within the single uncommitted tx a
        # cancelled row drops out of the next SELECT, so the sweep converges.
        await _sweep_active_descendants(repo, run.id, _degraded_cancel)
        # PAID sub-DAG children (course lessons) are parent_run_id=None, so the FK
        # cascade above misses them — reap them too or the cancel keeps spending.
        await _sweep_deliverable_children(session, repo, run.id, _degraded_cancel)
        await _recompute_project_status_for_run(session, run.deliverable_id)
        await session.commit()
        return False

    # Cancel the parent FIRST, then sweep descendants. Ordering matters for the
    # reconciler-finalize race: a stranded parent's relief child is spawned by
    # `run_dispatcher.finalize_if_dag_complete` inside its `mark_completed(parent)`
    # + `if transitioned:` gate. Marking the parent terminal-`cancelled` up front
    # makes that write-once `mark_completed` no-op, so the reconciler can no longer
    # spawn a fresh paid child underneath us. Then RE-COLLECT after each cancel
    # pass: a child spawned in the narrow window before the parent went terminal is
    # caught on the next pass (mark_cancelled / dispatcher.cancel is idempotent on
    # an already-terminal run). Bounded by _MAX_CANCEL_CASCADE_DEPTH passes; the
    # loop exits as soon as a pass finds no active descendant.
    if not parent_terminal:
        await _cancel_one(session, dispatcher, run)
    await _sweep_active_descendants(repo, run.id, lambda d: _cancel_one(session, dispatcher, d))
    # PAID sub-DAG children (course lessons) carry parent_run_id=None, so the FK
    # cascade above never sees them — reap them too or the cancelled course keeps
    # spending on every in-flight lesson render.
    await _sweep_deliverable_children(
        session, repo, run.id, lambda d: _cancel_one(session, dispatcher, d)
    )
    return True


@router.post(
    "/{run_id}/cancel",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="runs.cancel",
)
async def cancel_run(
    run_id: UUID,
    request: Request,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> None:
    run = await _tenant_owns_run(session, run_id, tenant_id)
    if run is None:
        raise HTTPException(404, "run not found")
    if not await cancel_run_tree(session, request.app.state.dispatcher, run):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "cancellation recorded but live interrupt unavailable — dispatcher not ready (Redis down)",
        )


@router.post(
    "/{run_id}/pause",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="runs.pause",
)
async def pause_run(
    run_id: UUID,
    request: Request,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> None:
    run = await _tenant_owns_run(session, run_id, tenant_id)
    if run is None:
        raise HTTPException(404, "run not found")
    if run.status not in ("running", "pending"):
        raise HTTPException(409, f"cannot pause {run.status}")
    # Resolve the dispatcher BEFORE mutating: if it's down (degraded mode) we
    # must not leave the row `paused` with no live interrupt to back it up.
    dispatcher = _require_dispatcher(request)
    # Persist the paused status BEFORE cancelling the dispatcher task. The
    # dispatcher's CancelledError handler checks the current status and skips
    # the cancel-overwrite when the row is already marked paused, so the
    # ordering matters.
    await RunRepository(session).mark_paused(run_id, reason="manual")
    await session.commit()
    # Same-pod fast path, then fan out to other replicas: a run executing on a
    # non-receiving pod is stopped by its control listener, and that pod's
    # CancelledError handler reads the now-`paused` row and honours it.
    await dispatcher.cancel(run_id)
    await dispatcher.broadcast_interrupt(run_id)


def _drop_removed_seeded_gates(
    blueprint_id: str, pauses: list[str], seeded: list[str] | None
) -> list[str]:
    """Drop the pauses the BLUEPRINT seeded that it no longer declares as gates.

    A pause is dropped only if it is BOTH in ``seeded`` (the run's
    ``seeded_pause_after_stages``, the blueprint-seeded subset recorded at
    creation) AND absent from the blueprint's current ``default_pause_after``
    stages. Everything else -- in particular a checkpoint a USER chose (explicit
    create list, ``PATCH /checkpoints``, CheckpointPicker) -- is kept: a missed
    user pause lets paid stages run unreviewed, an extra pause costs a click.

    ``seeded is None`` (legacy row, provenance unknown) keeps the list unchanged;
    the remedy for those stays ``PATCH /runs/{id}/checkpoints []``. An unknown
    ``blueprint_id`` also keeps it unchanged (fail safe).
    """
    if not seeded:
        return pauses
    try:
        blueprint = get_registry().resolve_by_id(blueprint_id)
    except KeyError:
        return pauses
    gates = {s.name for s in blueprint.stages if s.default_pause_after}
    stale = set(seeded) - gates
    return [s for s in pauses if s not in stale]


@router.post(
    "/{run_id}/resume",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="runs.resume",
)
async def resume_run(
    run_id: UUID,
    request: Request,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> None:
    run = await _tenant_owns_run(session, run_id, tenant_id)
    if run is None:
        raise HTTPException(404, "run not found")
    if run.status != "paused":
        # A STRUCTURED 409 (``error.details.code``), not a bare string: the run page
        # offered "Resume" on a FAILED run (UI review 2026-10-17, 0b) and the user
        # read the outcome as "nothing happened". Only a paused run resumes in place;
        # a failed/cancelled one must be restarted (``POST /runs/{id}/regenerate``),
        # which the FE can now route to by branching on the code.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail={
                "code": RUN_NOT_RESUMABLE,
                "status": run.status,
                "message": (
                    f"Only a paused run can be resumed; this run is {run.status}. "
                    "Restart it to run the pipeline again."
                ),
            },
        )
    # Resolve the dispatcher BEFORE mutating: marking the row `resumed` without
    # a dispatcher to actually re-dispatch would wedge it (resumed but not running).
    dispatcher = _require_dispatcher(request)
    # Resuming approves every checkpoint already consumed (a completed
    # pause-stage). Strip those from pause_after_stages so the row is the durable
    # approval ledger: a later reconciler re-dispatch (drain mid-downstream) then
    # knows the gate was approved and must NOT re-pause. Without this, an
    # about-to-pause run interrupted before mark_paused and an approved resume are
    # indistinguishable at dispatch entry (see run_dispatcher._execute_run guard).
    stage_rows = await StageOutputRepository(session).list_for_run(run_id)
    consumed = terminal_stage_names(stage_rows)
    remaining_pauses = [s for s in (run.pause_after_stages or []) if s not in consumed]
    # BACKLOG #6: a gate the blueprint SEEDED at creation and has since removed
    # (website_review, 2026-09-15) must not keep re-pausing runs created before
    # the removal. Only blueprint-seeded entries (runs.seeded_pause_after_stages)
    # are eligible to drop; user-chosen checkpoints are always kept.
    remaining_pauses = _drop_removed_seeded_gates(
        run.blueprint_id, remaining_pauses, run.seeded_pause_after_stages
    )
    resumed = await RunRepository(session).mark_resumed(run_id, pause_after_stages=remaining_pauses)
    if not resumed:
        # BEFORE the commit, deliberately. The status check above already 409s the
        # ordinary case, so losing the CAS means the run terminalised between that
        # read and this write, and the UPDATE matched no rows -- there is nothing
        # to commit and nothing to keep. Answering 204 here would be worse than
        # the error: a reviewer told their approval landed will not look again,
        # and the dispatch below would spawn an executor against a row that can
        # never accept its results.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail={
                "code": RUN_NOT_RESUMABLE,
                "status": "unknown",
                "message": "The run is no longer paused; reload to see its current state.",
            },
        )
    await session.commit()
    await dispatcher.dispatch(run_id)


@router.get(
    "/{run_id}/checkpoints",
    response_model=CheckpointsResponse,
    operation_id="runs.checkpoints.get",
)
async def get_checkpoints(
    run_id: UUID, session: SessionDep, tenant_id: TenantIdDep
) -> CheckpointsResponse:
    run = await _tenant_owns_run(session, run_id, tenant_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return CheckpointsResponse(run_id=run.id, pause_after_stages=list(run.pause_after_stages or []))


@router.patch(
    "/{run_id}/checkpoints",
    response_model=CheckpointsResponse,
    operation_id="runs.checkpoints.update",
)
async def update_checkpoints(
    run_id: UUID,
    body: CheckpointsUpdate,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> CheckpointsResponse:
    run = await _tenant_owns_run(session, run_id, tenant_id)
    if run is None:
        raise HTTPException(404, "run not found")
    # Never write to a terminal run, matching every other status mutator here.
    if run.status in ("completed", "cancelled", "failed"):
        raise HTTPException(409, f"cannot change checkpoints on a {run.status} run")
    # `pause_after_stages` is not a free-form list — `resume_run` uses it as the
    # durable APPROVAL LEDGER, stripping every gate the reviewer just approved
    # past. An unconditional write here races that CAS: land after it and an
    # already-approved gate is re-armed, so the run re-pauses the instant it
    # resumes (the dispatch-entry guard sees a completed pause-stage and calls
    # `mark_paused`). Strip the consumed gates on this side too and the outcome
    # stops depending on which write lands last — a gate whose stage is already
    # terminal can never legitimately fire again in THIS run anyway.
    consumed = terminal_stage_names(await StageOutputRepository(session).list_for_run(run_id))
    effective = [s for s in body.pause_after_stages if s not in consumed]
    # Provenance (BACKLOG #6): every stage the user explicitly sent is user-owned
    # from now on, so it leaves the blueprint-seeded set and a later blueprint
    # gate removal can never drop it at resume. NULL (legacy, provenance unknown)
    # stays NULL. Only this run's own column is touched -- it never changes again
    # after creation except here, so there is no race with resume to resolve.
    values: dict[str, list[str] | None] = {"pause_after_stages": effective}
    if run.seeded_pause_after_stages is not None:
        patched = set(body.pause_after_stages)
        values["seeded_pause_after_stages"] = [
            s for s in run.seeded_pause_after_stages if s not in patched
        ]
    await session.execute(update(Run).where(Run.id == run_id).values(**values))
    await session.commit()
    return CheckpointsResponse(run_id=run_id, pause_after_stages=effective)


@router.post(
    "/{run_id}/regenerate",
    response_model=RunSummary,
    status_code=status.HTTP_201_CREATED,
    operation_id="runs.regenerate",
)
async def regenerate_run(
    run_id: UUID,
    body: RegenerateRequest,
    request: Request,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
) -> RunSummary:
    """Create a child run that reuses what is still valid and re-runs the rest.

    ``from_stage`` given — the "rerun this stage" action: re-run that stage and
    everything depending on it, reuse every other completed stage. 422 if it is not
    a stage of this run's blueprint.

    No ``from_stage`` (the Studio's Restart / regenerate) on a failed, cancelled or
    completed parent: a per-stage plan (``services/runs/restart_plan.py``, previewed
    by ``GET /runs/{id}/restart-plan``) re-runs the stages that did not finish, have
    open feedback, read a run input that changed, or depend on a stage the user
    edited — and reuses everything else. Where the plan cannot reuse anything (or the
    blueprint cannot resume safely) the legacy annotation-driven partition applies:
    cut at the stages with open feedback, else re-run everything."""
    parent = await _tenant_owns_run(session, run_id, tenant_id)
    if parent is None:
        raise HTTPException(404, "run not found")
    # Every sibling spawn route (reorder-clips, apply-edits, regenerate-clip,
    # music rewind) 409s on an ACTIVE parent; regenerate was the only one that
    # did not, and each child it spawns is a full PAID run. Unlike those
    # siblings a regenerate is legitimate on a paused/failed/cancelled parent (the
    # annotation-driven review flow, and the Studio's Restart of a dead run), so
    # the gate is "not still active" rather than "completed".
    if parent.status in ("running", "pending"):
        raise HTTPException(
            409, f"cannot regenerate a {parent.status} run; pause or wait for completion"
        )
    # ...and 409 the second POST of a double-submit. Nothing else dedupes this:
    # `run_engine.start_run` re-checks nothing, so two clicks (or a retried
    # request) create and DISPATCH two full paid child runs off one parent.
    if await RunRepository(session).list_active_children(parent.id):
        raise HTTPException(
            409,
            "this run already has an active child run; wait for it to finish or cancel it",
        )
    from_stage = body.from_stage
    # Only an explicit narrow request refuses on an empty preserved set — see
    # RegenerateScopeError; the implicit paths keep their "re-run everything" fallback.
    from_stage_explicit = body.from_stage is not None
    restart: RestartDirective | None = None
    if from_stage is None and parent.status in PLANNABLE_PARENT_STATUSES:
        plan = await _restart_plan_for(
            session,
            parent,
            config_overrides=body.config_overrides,
            tenant_id=tenant_id,
            subject_id=subject_id,
        )
        if plan.mode == "partial":
            restart = RestartDirective(rerun_roots=plan.rerun_roots, rerun_after=plan.edited_stages)
    # subject_id comes from SubjectIdDep (the canonical auth dep), which reads
    # the ContextVar set by AuthMiddleware on real traffic — request.state is
    # only populated by the test conftest, so the old inline read 401'd in prod.
    engine = request.app.state.run_engine
    try:
        child = await engine.start_run(
            deliverable_id=parent.deliverable_id,
            tenant_id=tenant_id,
            subject_id=subject_id,
            parent_run_id=parent.id,
            pause_after_stages=list(parent.pause_after_stages or []),
            # The parent's list is a DRAINED approval ledger (resume_run strips
            # each approved gate); re-arm the blueprint's gates on the stages this
            # child actually re-runs — see restore_child_pause_after_stages.
            restore_drained_checkpoints=True,
            config_overrides=body.config_overrides,
            from_stage=from_stage,
            from_stage_explicit=from_stage_explicit,
            restart=restart,
        )
    except RegenerateScopeError as exc:
        # BEFORE the RegenerateError/ValueError arms — it is a ValueError too. A typed
        # 409 naming the requested stage: the request is well-formed (422 would be a
        # lie) but cannot be honoured narrowly against THIS parent's state.
        raise HTTPException(
            409,
            detail={
                "reason": "regenerate_scope_unavailable",
                "cause": exc.cause,
                "from_stage": exc.from_stage,
                "dropped_stages": exc.stages,
                "message": str(exc),
            },
        ) from exc
    except RegenerateOrderMismatchError as exc:
        # Also before the ValueError arm. 409, same reasoning as the scope refusal: the
        # request is well-formed, it is this PARENT's frozen narration that cannot be
        # honoured against the new order. The body names every input that moved so the
        # Studio can say which field to put back.
        raise HTTPException(
            409,
            detail={
                "reason": "regenerate_order_mismatch",
                "inherited_narration_stages": exc.narration_stages,
                "changed_duration_inputs": {
                    path: {"parent": before, "requested": after}
                    for path, (before, after) in exc.changes.items()
                },
                "message": str(exc),
            },
        ) from exc
    except RegenerateError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    project_id = await _project_id_for_deliverable(session, child.deliverable_id)
    return _summary(child, project_id)


async def _restart_plan_for(
    session: AsyncSession,
    parent: Run,
    *,
    config_overrides: Mapping[str, object] | None,
    tenant_id: UUID,
    subject_id: UUID,
) -> RestartPlan:
    """The per-stage Restart plan for ``parent`` (``services/runs/restart_plan.py``).

    Gathers the plan's inputs; the decision itself is the pure
    :func:`compute_restart_plan`. The "would-be" snapshot is re-built through
    ``build_config_snapshot`` — the same assembler ``start_run`` persists — with the
    request's ``config_overrides`` and the parent's pinned tenant defaults, plus the
    live-input fingerprint, so parent and would-be child are compared by the SAME
    functions. Only stages that read a changed key re-run (``config_reads``); a parent
    row that predates the recording re-runs on any change.

    Read-only: shared by ``POST /regenerate`` and the ``GET /restart-plan`` preview.
    """
    try:
        blueprint = get_registry().resolve_by_id(parent.blueprint_id)
    except KeyError:
        return _unplannable("this run's pipeline is no longer registered", [])
    deliverable = await DeliverableRepository(session).get(
        parent.deliverable_id, tenant_id=tenant_id
    )
    project = (
        await ProjectRepository(session).get(deliverable.project_id, tenant_id=tenant_id)
        if deliverable is not None
        else None
    )
    if deliverable is None or project is None:
        # Nothing to plan against; the route's start_run raises the real error.
        return _unplannable("the run's project could not be read", blueprint.topological_order())
    would_be = await build_config_snapshot(
        session,
        proj=project,
        deliv=deliverable,
        tenant_id=tenant_id,
        subject_id=subject_id,
        config_overrides=dict(config_overrides or {}),
        # The child a Restart creates inherits the parent's pinned tenant defaults
        # (start_run, P6), so the would-be snapshot must too.
        inherit_defaults_from=parent.config_snapshot or {},
    )
    would_be[RESUME_INPUTS_KEY] = await build_resume_inputs(
        session, blueprint=blueprint, project_id=project.id, config_snapshot=would_be
    )
    rows = await StageOutputRepository(session).list_for_run(parent.id)
    annotated = await AnnotationRepository(session).list_open_by_stage_for_run(parent.id)
    return compute_restart_plan(
        blueprint=blueprint,
        parent_status=parent.status,
        rows=rows,
        parent_snapshot=parent.config_snapshot or {},
        would_be_snapshot=would_be,
        annotated_stages=set(annotated),
    )


def _unplannable(reason: str, order: list[str]) -> RestartPlan:
    why = Reason("not_plannable", reason)
    return RestartPlan(
        mode="full",
        stages=tuple(StagePlan(n, "rerun", (why,)) for n in order),
        full_reasons=(why,),
    )


@router.get(
    "/{run_id}/restart-plan",
    response_model=RestartPlanResponse,
    operation_id="runs.restartPlan.get",
)
async def get_restart_plan(
    run_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
) -> RestartPlanResponse:
    """Preview what ``POST /runs/{id}/regenerate {}`` would reuse and re-run.

    Read-only. 409 for a run that is still active (``pending``/``running``) or
    ``paused`` (a paused run is resumed in place, not restarted). The answer is
    computed from the CURRENT project state, so it moves when the user edits.
    """
    parent = await _tenant_owns_run(session, run_id, tenant_id)
    if parent is None:
        raise HTTPException(404, "run not found")
    if parent.status not in PLANNABLE_PARENT_STATUSES:
        raise HTTPException(
            409,
            detail={
                "code": "restart_plan_unavailable",
                "status": parent.status,
                "message": (
                    "A paused run continues where it stopped; resume it instead."
                    if parent.status == "paused"
                    else f"This run is {parent.status}; wait for it to finish or cancel it."
                ),
            },
        )
    plan = await _restart_plan_for(
        session, parent, config_overrides=None, tenant_id=tenant_id, subject_id=subject_id
    )
    return RestartPlanResponse(
        run_id=parent.id,
        mode=plan.mode,
        full_reasons=[_reason(r) for r in plan.full_reasons],
        changed_inputs=list(plan.changed_inputs),
        stages=[
            RestartPlanStage(
                stage_name=s.stage_name,
                action=s.action,
                reasons=[_reason(r) for r in s.reasons],
                edited=s.edited,
            )
            for s in plan.stages
        ],
    )


def _reason(r: Reason) -> RestartPlanReason:
    return RestartPlanReason(code=r.code, message=r.message, stage=r.stage, inputs=list(r.inputs))


@router.post(
    "/{run_id}/reorder-clips",
    response_model=RunSummary,
    status_code=status.HTTP_201_CREATED,
    operation_id="runs.reorderClips",
)
async def reorder_clips(
    run_id: UUID,
    body: ReorderClips,
    request: Request,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
) -> RunSummary:
    """Rearrange the clip timeline and re-stitch the video as a new version.

    Spawns a child run that preserves every stage before the blueprint's compile
    cut (with ``stage_name``'s clip array reordered to ``ordered_element_ids``)
    and re-runs only the compile stage onward — reusing every generated clip
    asset. Returns the child ``RunSummary`` so the UI navigates to the new
    version (its lineage shows on ``GET /runs/{id}/ancestors``).

    422 if the stage is not a clip spine, the blueprint has no compile stage
    downstream, or the parent run is not completed (only a finished timeline can
    be re-stitched); 400 if ``ordered_element_ids`` is not a permutation of the
    stage's ids; 409 if the parent run is still running."""
    parent = await _tenant_owns_run(session, run_id, tenant_id)
    if parent is None:
        raise HTTPException(404, "run not found")
    if not stage_supports_element_pinning(body.stage_name):
        raise HTTPException(422, f"stage '{body.stage_name}' does not carry a clip timeline")
    if parent.status == "running":
        raise HTTPException(409, "cannot reorder on a running run; pause or wait for completion")
    if parent.status != "completed":
        # A reorder re-stitches an existing timeline: every clip asset must
        # already exist. A paused/failed run has no finished timeline — honouring
        # it would silently re-run the (expensive) recording stages instead of
        # reusing clips, breaking the "reuse every generated clip" contract.
        raise HTTPException(
            422,
            f"cannot reorder a {parent.status} run; only a completed run has a "
            "finished timeline to re-stitch",
        )
    try:
        blueprint = get_registry().resolve_by_id(parent.blueprint_id)
    except KeyError as exc:
        raise HTTPException(500, f"unknown blueprint '{parent.blueprint_id}'") from exc
    if compile_cut_stage(blueprint, body.stage_name) is None:
        raise HTTPException(
            422, f"stage '{body.stage_name}' has no downstream compile stage to re-stitch"
        )

    engine = request.app.state.run_engine
    try:
        child = await engine.start_run(
            deliverable_id=parent.deliverable_id,
            tenant_id=tenant_id,
            subject_id=subject_id,
            parent_run_id=parent.id,
            pause_after_stages=list(parent.pause_after_stages or []),
            # The parent's list is a DRAINED approval ledger (resume_run strips
            # each approved gate); re-arm the blueprint's gates on the stages this
            # child actually re-runs — see restore_child_pause_after_stages.
            restore_drained_checkpoints=True,
            reorder=ReorderDirective(
                spine_stage=body.stage_name, ordered_ids=body.ordered_element_ids
            ),
        )
    # The "no downstream compile stage" ReorderError is unreachable here: the
    # 422 pre-check above (compile_cut_stage is None) already rejects it, so the
    # only ReorderError that surfaces is the permutation / missing-output case → 400.
    except ReorderError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    project_id = await _project_id_for_deliverable(session, child.deliverable_id)
    return _summary(child, project_id)


def _reject_foreign_voiceover_urls(edits: dict[str, dict[str, object]], tenant_id: str) -> None:
    """422 if any ClipEdit ``audio_url`` isn't a voiceover THIS tenant uploaded.

    ``audio_url`` is the only client-settable asset URL on a ClipEdit, and the
    compile-time ``refresh_shot_asset_urls`` pass re-signs it with OUR MinIO
    credentials (``resign_presigned_url`` accepts any object in a bucket we own —
    it checks "is it ours", NOT "is it yours"). An unvalidated value would let a
    tenant point at ANOTHER tenant's object and have us fetch it and mix it into
    their video: a cross-tenant read primitive. Clients only ever echo back a URL
    we just minted for them, so require this tenant's own voiceover prefix.
    """
    for clip_id, edit in edits.items():
        if not _is_own_voiceover_url(edit.get("audio_url"), tenant_id):
            raise HTTPException(
                422, f"clip '{clip_id}': audio_url is not a voiceover this tenant uploaded"
            )


def _is_own_voiceover_url(audio_url: object, tenant_id: str) -> bool:
    """True when ``audio_url`` is absent/null, or is a voiceover THIS tenant uploaded.

    ``store_voiceover_override`` writes to ``voiceovers/{tenant}/clip_voiceover_override/…``,
    so the tenant is carried in the object key. We match on the key path only — the
    host and the query string (the presign signature) are irrelevant and change on
    every re-sign.

    An explicit ``None`` is the "clear back to TTS" delete and is always allowed.

    **The comparison is segment-wise on a NORMALIZED key, never a substring.** This
    used to be ``f"/voiceovers/{tenant_id}/clip_voiceover_override/" in path``, and a
    substring test cannot see traversal: ::

        …/voiceovers/{me}/clip_voiceover_override/../../{victim}/clip_voiceover_override/x.wav

    contains the caller's own prefix verbatim — so the old check PASSED it — while the
    path resolves to the victim's object. RFC 3986 §5.2.4 dot-segment removal is
    mandatory for URI resolution, so a downstream fetch of that URL is entitled to
    resolve it exactly that way. Since the whole point of this gate is that the
    compile-time ``refresh_shot_asset_urls`` pass re-signs the stored value with OUR
    MinIO credentials, letting one through is a cross-tenant read primitive — the very
    thing the caller docstring above says this must prevent. ``normalize_key`` collapses
    ``..`` BEFORE the comparison and ``is_within`` compares whole segments, so neither
    traversal nor a sibling prefix that merely shares a string (``{tenant}evil``) can
    pass. Same primitives, same reason, as ``api/panel_image_url.panel_object_ref``.
    """
    if audio_url is None:
        return True
    if not isinstance(audio_url, str):
        return False
    path = urlsplit(audio_url).path.lstrip("/")
    if not path:
        return False
    try:
        normalized = normalize_key(decode_ref_path(path))
    except KeyConfinementError:
        # Undecodable, or a decode that re-segments the key. Refuse rather than fall
        # back to a weaker test: an unparseable key is not this tenant's object.
        return False
    # Match the segment TRIPLE anywhere in the key, not anchored at the start. How
    # many segments precede it varies with the URL shape — a path-style presigned URL
    # leads with the bucket, other refs lead with ``artifacts/``, and a relative ref
    # leads with neither. Anchoring would 422 legitimate edits. Searching by whole
    # segments keeps the old check's positional tolerance while removing the two ways
    # a substring could be fooled: traversal (collapsed by ``normalize_key`` above)
    # and a sibling that merely shares a string prefix (``{tenant}evil`` is a
    # different segment, so it cannot match).
    want = ["voiceovers", tenant_id, "clip_voiceover_override"]
    segments = key_segments(normalized)
    return any(segments[i : i + len(want)] == want for i in range(len(segments) - len(want) + 1))


def _music_rewind_for(parent: Run, body: ApplyClipEdits) -> str | None:
    """The stage this run's music change must rewind to, or ``None``, gates included.

    ``None`` for every request without a music change and for every blueprint that
    does not time its cut to the bed — those keep the ordinary ``sound_effects`` +
    edit-directive re-stitch, and take the SAME spine gate they always took, in the
    same order. An unresolvable blueprint also reads ``None`` so the caller's own
    resolve still raises the 500 it always did.

    Raises 422 when the request cannot be honoured: a non-rewind request whose
    stage is not a clip spine (the pre-existing gate, moved verbatim), or a rewind
    request that ALSO carries per-clip edits — a grid-owning pipeline has no
    clip-spine stage at all (``_ELEMENT_PATHS`` lists none of ``action_reel``'s),
    so applying the music half and dropping the clip half would be exactly the
    half-honoured edit this path exists to stop.
    """
    if body.music is None:
        return None
    try:
        blueprint = get_registry().resolve_by_id(parent.blueprint_id)
    except KeyError:
        return None
    rewind = music_rewind_stage(blueprint)
    if rewind is not None and body.edits:
        raise HTTPException(
            422,
            "this pipeline cuts to the music and has no per-clip timeline to edit; "
            "submit the music change on its own",
        )
    return rewind


async def _require_known_music_track(music: MusicEdit | None, tenant_id: UUID) -> None:
    """422 unless a music edit's truthy ``track`` is a real music id for this tenant.

    Both downstream consumers (``sound_design.resolve_audio_bed`` and action_reel's
    ``music_select``) validate the id against the SAME tenant library and, on a miss,
    drop the bed — the re-stitch then completes green with the old (or no) bed. The
    Studio hit exactly that by sending the track's LABEL ("smooth lounge jazz
    background bed") instead of its id (``jazz_lounge``). An unknown id is a client
    error the caller can fix; answer it here, before a child run is spent on it.
    A falsy ``track`` (music OFF) needs no lookup."""
    if music is None or not music.track:
        return
    try:
        library = await load_sound_library(tenant_id=str(tenant_id))
    except Exception:  # noqa: BLE001 -- storage outage must not 500 an edit
        # The storage backend (manifest layers) is unreachable. Judge against the
        # built-in baseline — the bottom layer of every tenant's library — rather
        # than 500 the edit or wave an unknown id through to a silent no-op.
        _log.warning("sound library unavailable; validating music track against built-ins")
        library = SoundLibrary.builtin()
    if library.is_valid(music.track, category="music"):
        return
    known = ", ".join(library.ids_in("music")) or "none"
    raise HTTPException(
        422,
        f"unknown music track {music.track!r}: `music.track` must be a music track id "
        f"from GET /api/music/tracks (available: {known})",
    )


def _require_restitchable(parent: Run) -> None:
    """409/422 unless ``parent`` is a completed run there is a timeline to re-stitch.

    Extracted verbatim from ``apply_clip_edits`` (same statuses, same wording) so the
    handler stays under the cyclomatic gate; the reasoning is unchanged: an edit
    re-stitches an EXISTING timeline, so every clip asset must already exist. A
    paused/failed run has none, and honouring it would silently re-run the expensive
    recording stages instead of reusing clips.
    """
    if parent.status == "running":
        raise HTTPException(409, "cannot edit clips on a running run; pause or wait for completion")
    if parent.status != "completed":
        raise HTTPException(
            422,
            f"cannot edit clips on a {parent.status} run; only a completed run has a "
            "finished timeline to re-stitch",
        )


async def _spawn_music_rewind_child(
    *,
    request: Request,
    session: AsyncSession,
    parent: Run,
    tenant_id: UUID,
    subject_id: UUID,
    music: MusicEdit,
    from_stage: str,
) -> RunSummary:
    """Spawn the child run for a grid-owning pipeline's music change."""
    engine = request.app.state.run_engine
    try:
        child = await engine.start_run(
            deliverable_id=parent.deliverable_id,
            tenant_id=tenant_id,
            subject_id=subject_id,
            parent_run_id=parent.id,
            pause_after_stages=list(parent.pause_after_stages or []),
            # Same reasoning as every other re-stitch route: the parent's list is
            # a DRAINED approval ledger, so re-arm the blueprint's gates on the
            # stages this child actually re-runs.
            restore_drained_checkpoints=True,
            config_overrides=_reel_music_overrides(parent.config_snapshot, music),
            from_stage=from_stage,
        )
    except RegenerateError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    project_id = await _project_id_for_deliverable(session, child.deliverable_id)
    return _summary(child, project_id)


def _reel_music_overrides(
    parent_config: Mapping[str, object] | None, music: MusicEdit
) -> dict[str, object]:
    """The ``config_overrides`` block for a GRID-OWNING pipeline's music change.

    Deliberately NOT ``sound_effects``. ``grep -rn sound_effects
    src/services/stages/action_reel/`` returns nothing: ``ActionReelCompileStage``
    overrides ``build_audio_bed``, so the shared ``sound_design.resolve`` path that
    consumes ``sound_effects`` for every other deliverable type is bypassed and a
    music edit written there is a silent no-op — 201, green child run, identical
    soundtrack.

    So the choice is landed on the top-level ``music`` block, which is where
    ``music_select`` reads a pin (the same place ``brief.music.track_id`` lands,
    and it already skips the LLM outright for one), and the mix knobs ride along
    for ``compile``.

    MERGED onto the parent's block, not written fresh. ``MusicEdit``'s contract is
    "each omitted field keeps the run's current value", and the sibling
    ``_music_config_overrides`` already honours it by carrying the parent's
    ``sound_effects`` forward. Honouring it here too is the choice made over
    rewriting the docstring, for one reason: ``build_config_snapshot`` SPREADS
    ``config_overrides`` wholesale over a snapshot the child rebuilds from the
    brief, so a fresh block does not merge with anything — a second edit that
    changes only the track would silently revert a ``gain_db`` set by the first,
    and the operator's only clue would be the mix. One contract, both paths.

    ``enabled`` is always explicit: a falsy ``track`` is a request to turn the bed
    OFF, and ``music_select`` needs to tell that apart from "no opinion". Turning
    it off also drops the inherited ``track_id`` (the sibling instead writes
    ``music: False``, because there an absent key means "auto") so a later
    re-enable cannot resurrect a stale pin,
    while the mix knobs survive — same as the sibling leaves transitions/ambient.
    """
    raw_parent = (parent_config or {}).get(MUSIC_OVERRIDE_KEY)
    block: dict[str, object] = (
        dict(cast("dict[str, object]", raw_parent)) if isinstance(raw_parent, dict) else {}
    )
    block["enabled"] = bool(music.track)
    if music.track:
        block["track_id"] = music.track
    else:
        block.pop("track_id", None)
    if music.gain_db is not None:
        block["gain_db"] = music.gain_db
    if music.fade_in_s is not None:
        block["fade_in_s"] = music.fade_in_s
    if music.fade_out_s is not None:
        block["fade_out_s"] = music.fade_out_s
    if music.duck_db is not None:
        block["duck_db"] = music.duck_db
    return {MUSIC_OVERRIDE_KEY: block}


def _music_config_overrides(
    parent_config: Mapping[str, object] | None, music: MusicEdit | None
) -> dict[str, object] | None:
    """Merge a run-level music override onto the parent's resolved ``sound_effects``.

    ``parent_config`` is the parent run's config snapshot. Returns a
    ``config_overrides`` dict (``{"sound_effects": {...}}``) or ``None`` when there
    is no music change. A truthy ``track`` enables the bed and sets the catalog id
    (+ any supplied gain/fades); a falsy ``track`` turns music OFF by writing an
    explicit ``music: False`` while leaving transitions / ambient untouched. It must
    NOT drop the key: an ABSENT ``music`` means "auto" to ``resolve_audio_bed``, which
    then hands the child the music supervisor's pick — so "No music" shipped a bed.
    A later truthy ``track`` overwrites the ``False``. The parent's sound
    design is carried forward because a child run rebuilds its config snapshot from
    the brief, not the parent.

    An OMITTED ``music`` (a clip-only edit) is "keep the current bed", not "no
    opinion": the child rebuilds ``sound_effects`` from the brief
    (``build_config_snapshot``), so an earlier Studio choice — a track with its
    gain/fades/duck, or the explicit ``music: False`` — would silently revert to the
    brief's bed on the next caption or trim edit. So when the parent's block carries
    an explicit music choice it is carried forward unchanged; a parent that never had
    one (auto) returns ``None`` and the child resolves exactly as before."""
    raw_sound = (parent_config or {}).get("sound_effects")
    sound: dict[str, object] = (
        dict(cast("dict[str, object]", raw_sound)) if isinstance(raw_sound, dict) else {}
    )
    if music is None:
        return {"sound_effects": sound} if _has_explicit_music_choice(sound) else None
    if music.track:
        sound["enabled"] = True
        sound["music"] = music.track
        if music.gain_db is not None:
            sound["music_gain_db"] = music.gain_db
        if music.fade_in_s is not None:
            sound["fade_in_s"] = music.fade_in_s
        if music.fade_out_s is not None:
            sound["fade_out_s"] = music.fade_out_s
        if music.duck_db is not None:
            sound["duck_db"] = music.duck_db
    else:
        sound["music"] = False
    return {"sound_effects": sound}


#: ``sound_effects`` keys a Studio music edit writes (see ``_music_config_overrides``).
_MUSIC_CHOICE_KEYS = ("music_gain_db", "fade_in_s", "fade_out_s", "duck_db")


def _has_explicit_music_choice(sound: Mapping[str, object]) -> bool:
    """Whether a resolved ``sound_effects`` block pins the music bed.

    An explicit track id or ``music: False`` pins it; an absent key or a bare
    ``True`` means "auto" (the supervisor's pick), which the child re-derives on its
    own. A mix knob alone (gain/fades/duck) is also a choice worth keeping.
    """
    choice = sound.get("music")
    if choice is False or (isinstance(choice, str) and choice):
        return True
    return any(sound.get(key) is not None for key in _MUSIC_CHOICE_KEYS)


@router.post(
    "/{run_id}/apply-edits",
    response_model=RunSummary,
    status_code=status.HTTP_201_CREATED,
    operation_id="runs.applyClipEdits",
)
async def apply_clip_edits(
    run_id: UUID,
    body: ApplyClipEdits,
    request: Request,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
) -> RunSummary:
    """Apply per-clip edits (trim / anchor / caption / fit) and/or a music-bed
    change; re-stitch the video as a version.

    Spawns a child run that preserves every stage before the blueprint's compile
    cut — with ``stage_name``'s clip array carrying each edited clip's override
    fields — and re-runs only the compile stage onward, reusing every generated
    clip asset. Per-clip edits live on the spine stage output (survive down the
    run lineage); a run-level ``music`` change is merged into the run's
    ``sound_effects`` config and passed as a config override (never mutates the
    project brief). Returns the child ``RunSummary`` so the UI navigates to the
    new version.

    **A pipeline whose CUT is timed to the bed rewinds further.** ``action_reel``
    snaps every shot boundary to the selected track's beat grid, so re-stitching a
    new bed over the old cut would ship a reel cut to a track it does not play. For
    those blueprints the music change instead re-runs from ``music_select`` (see
    ``runs/elements.music_rewind_stage``) so the grid and the cut are re-derived,
    the footage branch is preserved, and the choice rides the ``music`` config block
    the stage actually reads — NOT ``sound_effects``, which no ``action_reel`` stage
    reads. Such a request must carry no per-clip ``edits`` (that pipeline has no
    clip spine) → 422.

    Same completed-only gate as reorder (only a finished timeline can be
    re-stitched): 422 if the stage is not a clip spine, the blueprint has no
    compile stage downstream, or the parent run is not completed; 400 if none of
    the edited clip ids exist on the current timeline; 409 if the parent run is
    still running."""
    parent = await _tenant_owns_run(session, run_id, tenant_id)
    if parent is None:
        raise HTTPException(404, "run not found")
    await _require_known_music_track(body.music, tenant_id)
    # Resolved BEFORE the spine gate, but only far enough to ask "is this a
    # grid-owning pipeline whose music change must rewind?". An unknown blueprint
    # reads as None here and falls through to the ordinary path below, whose own
    # resolve still 500s — so the error precedence every other deliverable type
    # sees is byte-identical.
    music_rewind = _music_rewind_for(parent, body)
    if music_rewind is None and not stage_supports_element_pinning(body.stage_name):
        raise HTTPException(422, f"stage '{body.stage_name}' does not carry a clip timeline")
    _require_restitchable(parent)
    try:
        blueprint = get_registry().resolve_by_id(parent.blueprint_id)
    except KeyError as exc:
        raise HTTPException(500, f"unknown blueprint '{parent.blueprint_id}'") from exc
    if music_rewind is None and compile_cut_stage(blueprint, body.stage_name) is None:
        raise HTTPException(
            422, f"stage '{body.stage_name}' has no downstream compile stage to re-stitch"
        )

    # `body.music is not None` is a precondition of `music_rewind` being set (it
    # returns None without one); restated here so the type narrows without a cast.
    if music_rewind is not None and body.music is not None:
        # The whole point of the rewind: `from_stage` makes the child re-run the
        # grid owner's DESCENDANT CLOSURE (music_select → cut_plan → graphics /
        # narrate → compile), so the new bed's beat grid is fetched and the cut is
        # re-snapped to it, while the footage branch is preserved and never
        # re-billed. Re-stitching at the compile cut instead would render a reel
        # cut to the OLD track's beats — every stage green, `beat_locked` true,
        # and wrong.
        return await _spawn_music_rewind_child(
            request=request,
            session=session,
            parent=parent,
            tenant_id=tenant_id,
            subject_id=subject_id,
            music=body.music,
            from_stage=music_rewind,
        )

    # Only the fields the client actually sent are patched (exclude_unset) — an
    # omitted field leaves the clip's current value untouched; an explicit null
    # clears a prior override. Snake_case keys (by_alias=False) match the spine
    # element fields.
    edits = {
        clip_id: edit.model_dump(exclude_unset=True, by_alias=False)
        for clip_id, edit in body.edits.items()
    }

    _reject_foreign_voiceover_urls(edits, str(tenant_id))

    # A run-level music change merges into the PARENT's resolved sound_effects
    # (child runs rebuild config from the brief, so we carry the current sound
    # design forward and only touch music) and rides `config_overrides` — which
    # wins over the brief lift and reaches resolve_audio_bed on the child.
    config_overrides = _music_config_overrides(parent.config_snapshot, body.music)

    engine = request.app.state.run_engine
    try:
        child = await engine.start_run(
            deliverable_id=parent.deliverable_id,
            tenant_id=tenant_id,
            subject_id=subject_id,
            parent_run_id=parent.id,
            pause_after_stages=list(parent.pause_after_stages or []),
            # The parent's list is a DRAINED approval ledger (resume_run strips
            # each approved gate); re-arm the blueprint's gates on the stages this
            # child actually re-runs — see restore_child_pause_after_stages.
            restore_drained_checkpoints=True,
            edit=EditDirective(spine_stage=body.stage_name, edits=edits),
            config_overrides=config_overrides,
        )
    # The "no downstream compile stage" EditError is unreachable (the 422
    # pre-check above already rejects it), so the only EditError that surfaces is
    # the missing-output / no-live-id case → 400.
    except EditError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    project_id = await _project_id_for_deliverable(session, child.deliverable_id)
    return _summary(child, project_id)


@router.post(
    "/{run_id}/renarrate-clip",
    response_model=RunSummary,
    status_code=status.HTTP_201_CREATED,
    operation_id="runs.renarrateClip",
)
async def renarrate_clip(
    run_id: UUID,
    body: Renarrate,
    request: Request,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
) -> RunSummary:
    """Re-narrate a single line (new text and/or voice) and re-stitch as a version.

    Spawns a child run that re-synthesizes EXACTLY ONE TTS clip — the addressed
    line's — patches that line's text on the stage the narrate handler reads (so the
    captions match the new audio) plus that one clip entry on the narrate stage's
    preserved output, and re-runs only the compile stage onward. Every sibling
    clip's audio is reused byte-identically. Returns the child ``RunSummary``.

    Same completed-only gate as apply-edits / reorder (only a finished timeline can
    be re-stitched): 422 if the stage is not a clip spine, the parent run is not
    completed, the pipeline has no narration stage between the spine and the compile
    cut, the clip cannot be resolved to a source line, or the clip already carries an
    UPLOADED voiceover (which would win over the TTS in ``assemble_shots``, making the
    re-narrate silently inaudible — remove it first). 409 if the parent is running."""
    parent = await _tenant_owns_run(session, run_id, tenant_id)
    if parent is None:
        raise HTTPException(404, "run not found")
    if not stage_supports_element_pinning(body.stage_name):
        raise HTTPException(422, f"stage '{body.stage_name}' does not carry a clip timeline")
    if parent.status == "running":
        raise HTTPException(409, "cannot re-narrate on a running run; pause or wait for completion")
    if parent.status != "completed":
        # A re-narrate re-stitches an existing timeline: every sibling clip asset
        # must already exist, or we would silently re-run the expensive narrate +
        # recording stages instead of reusing them. Mirrors the reorder/edit gate.
        raise HTTPException(
            422,
            f"cannot re-narrate a line on a {parent.status} run; only a completed run "
            "has a finished timeline to re-stitch",
        )

    engine = request.app.state.run_engine
    try:
        child = await engine.start_run(
            deliverable_id=parent.deliverable_id,
            tenant_id=tenant_id,
            subject_id=subject_id,
            parent_run_id=parent.id,
            pause_after_stages=list(parent.pause_after_stages or []),
            # The parent's list is a DRAINED approval ledger (resume_run strips
            # each approved gate); re-arm the blueprint's gates on the stages this
            # child actually re-runs — see restore_child_pause_after_stages.
            restore_drained_checkpoints=True,
            renarrate=RenarrateDirective(
                spine_stage=body.stage_name,
                element_id=body.element_id,
                narration_text=body.narration_text,
                voice=body.voice,
            ),
        )
    except RenarrateError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    project_id = await _project_id_for_deliverable(session, child.deliverable_id)
    return _summary(child, project_id)


@router.post(
    "/{run_id}/regenerate-clip",
    response_model=RunSummary,
    status_code=status.HTTP_201_CREATED,
    operation_id="runs.regenerateClip",
)
async def regenerate_clip(
    run_id: UUID,
    body: RegenerateClip,
    request: Request,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
) -> RunSummary:
    """Regenerate a single clip's image and re-stitch as a new version.

    Spawns a child run that preserves the spine (and every sibling clip's assets)
    verbatim and re-runs only the per-clip image stage onward, folding this clip's
    open feedback into just its prompt — so exactly one image regenerates while
    every other clip is reused. Returns the child ``RunSummary``.

    422 if the stage is not a clip spine, the parent run is not completed (its
    sibling clips must already exist to be reused), or the pipeline cannot target
    a single clip surgically (no per-clip image stage, static round-robin
    prompts, no line_id, or no open feedback on the clip) — the UI then falls
    back to the annotation-driven ``regenerate``. 409 if the parent run is still
    running."""
    parent = await _tenant_owns_run(session, run_id, tenant_id)
    if parent is None:
        raise HTTPException(404, "run not found")
    if not stage_supports_element_pinning(body.stage_name):
        raise HTTPException(422, f"stage '{body.stage_name}' does not carry a clip timeline")
    if parent.status == "running":
        raise HTTPException(409, "cannot regenerate on a running run; pause or wait for completion")
    if parent.status != "completed":
        # Surgical single-clip regen reuses every sibling clip's asset from
        # storage; a paused/failed run has not produced them, so honouring it
        # would re-generate everything. Reject → the UI falls back to the
        # annotation-driven regenerate, which handles partial runs.
        raise HTTPException(
            422,
            f"cannot regenerate a clip on a {parent.status} run; only a completed "
            "run has the sibling clips to reuse",
        )

    engine = request.app.state.run_engine
    try:
        child = await engine.start_run(
            deliverable_id=parent.deliverable_id,
            tenant_id=tenant_id,
            subject_id=subject_id,
            parent_run_id=parent.id,
            pause_after_stages=list(parent.pause_after_stages or []),
            # The parent's list is a DRAINED approval ledger (resume_run strips
            # each approved gate); re-arm the blueprint's gates on the stages this
            # child actually re-runs — see restore_child_pause_after_stages.
            restore_drained_checkpoints=True,
            regen_clip=RegenClipDirective(spine_stage=body.stage_name, element_id=body.element_id),
        )
    except RegenClipError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    project_id = await _project_id_for_deliverable(session, child.deliverable_id)
    return _summary(child, project_id)
