"""RunEngine — single entry point for starting a run.

Steps:
  1. Resolve the deliverable + parent project (tenant-scoped).
  2. Resolve the project_type's DAGBlueprint.
  3. INSERT a `runs` row (status=pending) + initial stage_outputs (status=pending)
     in one transaction. If parent_run_id is given, run the regenerate logic:
     preserve completed parent stage outputs strictly BEFORE the first stage
     with open annotations, mark them completed on the child, then atomically
     flip the parent's open annotations to status='applied' with
     applied_in_run_id=child.id (per spec §4 invariant 4 + §8.6).
  4. Build a snapshot of the project's brief/brand/voice/theme into runs.config_snapshot.
     For course-typed projects, also fold in `lessons[]` (each child's title +
     brief + default_deliverable_id) so the course_compile_lessons stage can
     dispatch child runs without re-resolving children. For regenerate runs,
     fold open annotations into config_snapshot["regenerate_notes"] so the
     dispatcher can seed RunContext.open_annotations_by_stage.
  5. Hand off to the dispatcher (asyncio task) which then resolves the engine
     in-process and runs the DAG.

The per-step assembly (config snapshot, course lessons, initial stage rows)
lives in ``run_setup`` so each piece is independently testable; ``start_run``
is the orchestrator.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Container, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, TypedDict
from uuid import UUID

from src.database.models import Run
from src.database.repositories.deliverable_repository import DeliverableRepository
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.database.session import bind_session_tenant
from src.services.blueprints.registry import BlueprintRegistry
from src.services.providers import adapters
from src.services.providers.preflight import preflight_byo
from src.services.run_licence import resolve_run_licence
from src.services.runs.regenerate import (
    apply_open_annotations_in_transaction,
    assert_inherited_narration_matches_the_order,
    build_edit_state,
    build_regen_clip_state,
    build_regenerate_state,
    build_renarrate_state,
    build_reorder_state,
)
from src.services.runs.resume_inputs import RESUME_INPUTS_KEY, build_resume_inputs
from src.services.runs.run_finalizer import terminal_stage_names
from src.services.runs.run_setup import (
    REGENERATE_FROM_STAGE_KEY,
    build_config_snapshot,
    insert_initial_stage_rows,
)
from src.services.stages.website_review.order import validate_website_review_order

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.database.models import Project
    from src.services.blueprints.base import DAGBlueprint

log = logging.getLogger(__name__)

DispatchFn = Callable[[UUID], Awaitable[None]]


class ReorderDirective(TypedDict):
    """Studio clip-reorder request routed into ``start_run``.

    ``spine_stage`` is the clip-spine stage whose element array is reordered;
    ``ordered_ids`` is the full permutation of that stage's element ids.
    """

    spine_stage: str
    ordered_ids: list[str]


class RegenClipDirective(TypedDict):
    """Studio single-clip regenerate request routed into ``start_run``.

    ``spine_stage`` is the clip-spine stage the feedback is pinned to;
    ``element_id`` is the target clip's spine element id (``shot.id`` / ``line.id``).
    """

    spine_stage: str
    element_id: str


class RenarrateDirective(TypedDict):
    """Studio single-line re-narrate request routed into ``start_run``.

    ``spine_stage`` / ``element_id`` address the clip; ``narration_text`` and/or
    ``voice`` are the new values (at least one is set — the schema enforces it; an
    omitted field keeps the line's current text / the run's configured voice).
    Re-synthesizes exactly ONE clip and re-stitches — see ``build_renarrate_state``.
    """

    spine_stage: str
    element_id: str
    narration_text: str | None
    voice: str | None


class EditDirective(TypedDict):
    """Studio per-clip edit request routed into ``start_run``.

    ``spine_stage`` is the clip-spine stage whose elements carry the edited
    fields; ``edits`` maps clip id → a patch of override fields (trim / anchor).
    Re-stitches only (preserve every pre-compile stage with the edited spine),
    reusing every generated clip asset — no annotation notes.
    """

    spine_stage: str
    edits: dict[str, dict[str, Any]]


class RestartDirective(TypedDict):
    """A planned Restart (``services/runs/restart_plan.py``) routed into ``start_run``.

    ``rerun_roots`` re-run together with every stage depending on them;
    ``rerun_after`` are stages the user edited — reused, their dependents re-run.
    Every other completed parent stage is inherited.
    """

    rerun_roots: frozenset[str]
    rerun_after: frozenset[str]


def _seed_pause_after_stages(
    blueprint: DAGBlueprint, proj: Project, parent_run_id: UUID | None
) -> list[str]:
    """Default pause checkpoints for a fresh run (None → seed from blueprint).

    Regenerate runs (parent_run_id set) inherit the parent's checkpoints, so
    they seed empty.

    audiobook declares two review gates (chapter review after ingest, casting
    review after cast) so a book pauses for human review BEFORE paid TTS spend.
    The one-command lift (spec §5) lets a brief opt OUT: an explicit
    ``brief.audiobook.pause_for_review: false`` drops both gates so the book
    renders end-to-end from a single brief. Absent/true preserve today's
    behaviour (both gates active) — the safe default, since silently skipping
    the gates would run straight into paid TTS narration without review.

    ⚠️ **website_review has NO gates any more** (owner decision 2026-09-15 —
    the UI never announced them, so a gated run read as stalled), so its seed is
    ``[]`` for every brief. Its two opt-out branches — ``generate_video: false``
    and a top-level ``pause_for_review: false`` — were DELETED with them, not
    left in place: with no ``default_pause_after`` stage in the blueprint they
    could only ever return ``[]`` from a ``seed`` that was already ``[]``, and
    dead code no test can distinguish reads as a live guarantee. An audit proved
    exactly that — the whole ``pause_for_review`` branch could be removed with
    all 14 blueprint tests still green.

    ``WebsiteReviewBrief.pause_for_review`` is therefore INERT for
    website_review. It is still typed at the door (removing it would break the
    generated FE client) and is documented there as a no-op. If the gates ever
    come back, the opt-out has to come back WITH a test that reds when it is
    removed — restoring this branch alone would be a lie again.
    """
    if parent_run_id is not None:
        return []
    seed = [s.name for s in blueprint.stages if s.default_pause_after]
    if (
        proj.type == "audiobook"
        and ((proj.brief or {}).get("audiobook") or {}).get("pause_for_review") is False
    ):
        return []
    return seed


def restore_child_pause_after_stages(
    *,
    blueprint: DAGBlueprint,
    proj: Project,
    parent_pause_after_stages: list[str] | None,
    rerun_stages: Container[str],
    approved_stages: Container[str],
) -> list[str]:
    """Re-arm the blueprint's review gates on the stages a CHILD run re-runs.

    WHY this exists: ``runs.pause_after_stages`` is the durable *approval
    ledger*, not a static gate list. ``resume_run`` STRIPS each checkpoint the
    human has already approved (so the reconciler's re-dispatch does not
    re-pause a drained gate — see ``run_dispatcher``). A parent whose gates were
    all approved therefore ends with ``[]``. Naively handing that ``[]`` to a
    regenerate / reorder / clip-edit child made the child run with ZERO review
    gates — a paid render with no human in the loop. **A drained ledger is not
    an empty gate list.**

    So: recompute the blueprint's ORIGINAL checkpoint set (``parent_run_id=None``
    — the project-level opt-outs still apply), keep only the gates that fall on
    stages the child actually RE-RUNS, and union that with whatever the parent
    still had outstanding. Stages whose outputs were PRESERVED are not re-run and
    so are never re-paused; a parent whose gates were never drained is unaffected,
    because the restored set is a subset of the seed it already carries.

    ⚠️ A gate is restored ONLY if the parent actually REACHED it. That third
    condition is what separates the two ways a name can be missing from the
    parent's list, which are otherwise byte-identical ``[]``:

    * the human approved past it — ``resume_run`` stripped it, and the stage has
      a TERMINAL row in the parent's ``stage_outputs``
      (``run_finalizer.terminal_stage_names``, the same predicate ``resume_run``
      itself uses to decide what to strip);
    * an operator deliberately removed it — ``PATCH /runs/{id}/checkpoints`` with
      ``[]``, or ``POST /projects`` with ``autopilot``. The stage was never run,
      so it has no terminal row.

    ⚠️ THIS IS A PROXY, AND IT IS ONLY PARTIAL. It separates the two cases only
    while the parent NEVER REACHED the gate. An autopilot parent that ran to
    COMPLETION has a terminal row for every stage, so its child does get the
    gates re-armed — the operator's decision is still overridden there. Closing
    that needs a real signal (an ``autopilot`` flag on the run), not a
    terminal-row proxy, and that is a migration this change does not make.

    Shipping anyway is a deliberate weighting, not an oversight: an unwanted
    pause is VISIBLE (the run sits ``paused``, ``GET /checkpoints`` shows it) and
    recoverable in one operator round-trip (PATCH the child's checkpoints, then
    resume). The failure it prevents is a paid render with no human review and
    nothing anywhere saying so. Between a recoverable pause and a silent paid
    render, the pause is the correct default.

    Returned in blueprint stage order for determinism. NOTE: this is the
    blueprint's DECLARATION order, not ``topological_order()``; only
    ``GET /checkpoints`` echoes the list and nothing depends on the ordering.
    """
    seed = set(_seed_pause_after_stages(blueprint, proj, None))
    keep = set(parent_pause_after_stages or [])
    restored = keep | {name for name in seed if name in rerun_stages and name in approved_stages}
    ordered = [s.name for s in blueprint.stages if s.name in restored]
    # Preserve any parent entry that is not a stage of this blueprint (defensive:
    # an operator-set checkpoint via PATCH /checkpoints) rather than dropping it.
    known = {s.name for s in blueprint.stages}
    return ordered + sorted(restored - known)


def _validate_website_review_outputs(project_type: str, brief: dict[str, Any]) -> None:
    """Reject a website_review whose every output is disabled (no-op otherwise).

    Mirrors ``video_gate``'s defaults (the source of truth): critique & video
    default ON, redesign defaults OFF but is implied by video. With all three
    explicitly false, every stage skips and the run "completes" with no
    deliverable — a silent green empty run. Raise ``ValueError`` (→ HTTP 400 at
    the route boundary) so the caller learns nothing would be produced.
    """
    if project_type != "website_review":
        return
    critique = bool(brief.get("generate_critique", True))
    redesign = bool(brief.get("generate_redesign", False)) or bool(
        brief.get("generate_video", True)
    )
    if not (critique or redesign):
        raise ValueError(
            "website_review needs at least one output enabled (critique, redesign, or video)"
        )


#: Parent states a child run may be spawned from: every state in which the parent is
#: no longer executing. ``pending``/``running`` stay refused — a child off a live
#: parent would race it for the same deliverable.
_SPAWNABLE_PARENT_STATUSES: frozenset[str] = frozenset(
    {"paused", "completed", "failed", "cancelled"}
)


async def _resolve_parent_state(
    *,
    session: AsyncSession,
    run_repo: RunRepository,
    blueprint: DAGBlueprint,
    parent_run_id: UUID,
    deliverable_id: UUID,
    reorder: ReorderDirective | None,
    edit: EditDirective | None = None,
    regen_clip: RegenClipDirective | None = None,
    renarrate: RenarrateDirective | None = None,
    from_stage: str | None = None,
    from_stage_explicit: bool = False,
    restart: RestartDirective | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], list[UUID], Run]:
    """Resolve preserved outputs (+ regenerate notes) for a child run.

    Returns the resolved PARENT alongside them: ``start_run`` needs the parent's
    persisted ``config_snapshot`` to check the child's order against what the
    inherited narration was authored for, and this is where the (validated) parent
    row is already in hand — re-reading it there would be a second query and a
    second chance for the two to disagree.

    A ``reorder`` directive re-stitches only: preserve every stage before the
    compile cut with the spine reordered, no annotation notes. An ``edit``
    directive is the same re-stitch with per-clip edit fields merged onto the
    spine instead of a reorder. A ``renarrate`` directive is the same re-stitch
    with ONE line's text re-written and ONE TTS clip re-synthesized (the narrate
    stage's preserved output is patched in place — see ``build_renarrate_state``).
    A ``regen_clip`` directive regenerates a single clip's image: preserve every
    stage before the image stage (spine verbatim) with the clip's feedback folded
    into that stage. Otherwise run the annotation-driven regenerate partition,
    optionally forced to cut at ``from_stage`` (the "rerun this stage" action).
    """
    parent = await run_repo.get(parent_run_id)
    if parent is None or parent.deliverable_id != deliverable_id:
        raise ValueError("parent_run_id invalid")
    # ``cancelled`` is a dead run exactly like ``failed``: the Studio offers Restart on
    # both (w2v_frontend ``canRestartRun``) and the regenerate route refuses only an
    # ACTIVE parent. Leaving it out here 400'd every Restart of a cancelled run (prod
    # run 01a1770d, 2026-10-27: a stuck run was cancelled, then Restart → 400, three times).
    if parent.status not in _SPAWNABLE_PARENT_STATUSES:
        raise ValueError(
            f"parent run must be paused/completed/failed/cancelled (is {parent.status})"
        )
    if reorder is not None:
        preserved = await build_reorder_state(
            session=session,
            parent_run=parent,
            blueprint=blueprint,
            spine_stage=reorder["spine_stage"],
            ordered_ids=reorder["ordered_ids"],
        )
        return preserved, {}, [], parent
    if edit is not None:
        preserved = await build_edit_state(
            session=session,
            parent_run=parent,
            blueprint=blueprint,
            spine_stage=edit["spine_stage"],
            edits=edit["edits"],
        )
        return preserved, {}, [], parent
    if renarrate is not None:
        preserved = await build_renarrate_state(
            session=session,
            parent_run=parent,
            blueprint=blueprint,
            spine_stage=renarrate["spine_stage"],
            element_id=renarrate["element_id"],
            narration_text=renarrate["narration_text"],
            voice=renarrate["voice"],
        )
        return preserved, {}, [], parent
    if regen_clip is not None:
        clip_state = await build_regen_clip_state(
            session=session,
            parent_run=parent,
            blueprint=blueprint,
            spine_stage=regen_clip["spine_stage"],
            element_id=regen_clip["element_id"],
        )
        return (*clip_state, parent)
    regen_state = await build_regenerate_state(
        session=session,
        parent_run=parent,
        blueprint=blueprint,
        from_stage=from_stage,
        from_stage_explicit=from_stage_explicit,
        rerun_roots=restart["rerun_roots"] if restart is not None else None,
        rerun_after=restart["rerun_after"] if restart is not None else (),
    )
    return (*regen_state, parent)


def _record_regenerate_request(
    config_snapshot: MutableMapping[str, object],
    *,
    notes_by_stage: Mapping[str, object],
    from_stage: str | None,
) -> None:
    """Stamp a child run's regenerate request onto its (run-scoped) snapshot keys.

    ``from_stage`` is passed only when the CALLER named it; a stage reads it back via
    :data:`REGENERATE_FROM_STAGE_KEY` to know it was the explicit re-run target.
    """
    if notes_by_stage:
        config_snapshot["regenerate_notes"] = notes_by_stage
    if from_stage is not None:
        config_snapshot[REGENERATE_FROM_STAGE_KEY] = from_stage


class RunEngine:
    def __init__(
        self,
        *,
        session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
        registry: BlueprintRegistry,
        dispatch: DispatchFn,
    ) -> None:
        self._sessions = session_factory
        self._registry = registry
        self._dispatch = dispatch

    async def start_run(
        self,
        *,
        deliverable_id: UUID,
        tenant_id: UUID,
        subject_id: UUID,
        parent_run_id: UUID | None,
        pause_after_stages: list[str] | None,
        config_overrides: dict[str, Any] | None = None,
        reorder: ReorderDirective | None = None,
        edit: EditDirective | None = None,
        regen_clip: RegenClipDirective | None = None,
        renarrate: RenarrateDirective | None = None,
        from_stage: str | None = None,
        from_stage_explicit: bool = False,
        restart: RestartDirective | None = None,
        structural_relief_pass: int | None = None,
        restore_drained_checkpoints: bool = False,
        inherit_slots: Mapping[str, Any] | None = None,
        inherit_defaults_from: Mapping[str, Any] | None = None,
    ) -> Run:
        """Create + dispatch a run.

        ``inherit_slots``: the PARENT run's ``provider_slots`` for a SYSTEM-spawned
        child (course lessons, structural relief) so it keeps the parent's provider
        pins. User-initiated restarts / regenerates leave it ``None`` and resolve
        fresh — the user is starting new work on the tenant's current providers.

        Tenant defaults (P6) are stricter — a job pins them for its whole lifetime:
        EVERY run spawned by a run inherits the parent's pinned defaults. With
        ``parent_run_id`` set that is automatic (the parent's snapshot); a spawned run
        with no parent row link (course lessons) passes ``inherit_defaults_from`` =
        the spawning run's snapshot. Only a run started fresh reads the tenant's
        current defaults.
        """
        async with self._sessions() as session:
            # Bind the RLS GUC before any read/INSERT so the run + initial stage
            # rows pass the tenant policies under enforce (no-op until then).
            await bind_session_tenant(session, tenant_id)
            deliv_repo = DeliverableRepository(session)
            proj_repo = ProjectRepository(session)
            run_repo = RunRepository(session)
            stage_repo = StageOutputRepository(session)

            deliv = await deliv_repo.get(deliverable_id, tenant_id=tenant_id)
            if deliv is None:
                raise ValueError("deliverable not found or not visible to tenant")
            proj = await proj_repo.get(deliv.project_id, tenant_id=tenant_id)
            if proj is None:
                raise ValueError("project missing for deliverable")
            blueprint = self._registry.resolve(proj.type)

            # When the caller omits pause_after_stages (None — the SPA default
            # for runs.start() with no body), seed from the blueprint's declared
            # defaults so e.g. promo runs pause after script + storyboard for
            # creative review by default. An explicit empty list keeps autopilot.
            # Regenerate runs (parent_run_id set) skip the seeding — the resumed
            # run inherits the parent's checkpoint set via checkpoint propagation.
            # Provenance (BACKLOG #6): ``seeded`` is the subset of the checkpoint
            # list the BLUEPRINT put there, persisted as
            # ``runs.seeded_pause_after_stages``. An explicit caller list (create
            # body, autopilot ``[]``) is a USER choice and contributes nothing.
            seeded: list[str] = []
            if pause_after_stages is None:
                pause_after_stages = _seed_pause_after_stages(blueprint, proj, parent_run_id)
                seeded = list(pause_after_stages)

            preserved_outputs: dict[str, dict[str, Any]] = {}
            notes_by_stage: dict[str, list[dict[str, Any]]] = {}
            affected_so_ids: list[UUID] = []
            parent_run: Run | None = None
            if parent_run_id is not None:
                (
                    preserved_outputs,
                    notes_by_stage,
                    affected_so_ids,
                    parent_run,
                ) = await _resolve_parent_state(
                    session=session,
                    run_repo=run_repo,
                    blueprint=blueprint,
                    parent_run_id=parent_run_id,
                    deliverable_id=deliverable_id,
                    reorder=reorder,
                    edit=edit,
                    regen_clip=regen_clip,
                    renarrate=renarrate,
                    from_stage=from_stage,
                    from_stage_explicit=from_stage_explicit,
                    restart=restart,
                )
                # A child inherits its parent's provenance for every entry it
                # carries over; a legacy parent (NULL) contributes nothing, so
                # its entries are treated as user picks and always kept.
                parent_seeded = set(parent_run.seeded_pause_after_stages or [])
                seeded = [s for s in pause_after_stages or [] if s in parent_seeded]
                if restore_drained_checkpoints:
                    before_restore = set(pause_after_stages or [])
                    # The parent's list is a DRAINED approval ledger, not a gate
                    # list — re-arm the blueprint's gates on the re-run region.
                    # Opt-in: system-spawned children (structural relief) and the
                    # explicit-list API on /deliverables deliberately keep theirs.
                    pause_after_stages = restore_child_pause_after_stages(
                        blueprint=blueprint,
                        proj=proj,
                        parent_pause_after_stages=pause_after_stages,
                        rerun_stages={
                            s.name for s in blueprint.stages if s.name not in preserved_outputs
                        },
                        # Only gates the parent actually REACHED. A name missing
                        # from the parent's list because an operator cleared it
                        # (autopilot) has no terminal row here, so it stays out.
                        approved_stages=terminal_stage_names(
                            await StageOutputRepository(session).list_for_run(parent_run_id)
                        ),
                    )
                    # Every gate the restore re-armed came from the blueprint seed.
                    seeded += [s for s in pause_after_stages if s not in before_restore]

            config_snapshot = await build_config_snapshot(
                session,
                proj=proj,
                deliv=deliv,
                tenant_id=tenant_id,
                subject_id=subject_id,
                config_overrides=config_overrides,
                inherit_slots=inherit_slots,
                inherit_defaults_from=(
                    inherit_defaults_from
                    if inherit_defaults_from is not None
                    else ((parent_run.config_snapshot or {}) if parent_run is not None else None)
                ),
            )
            # Cost attribution: the run's S2S hops carry THIS tenant's own licence
            # (tenants.licence_ref), never an empty one that downstream would bill
            # to the shared umbrella licence. See src/services/run_licence.py.
            config_snapshot["license_id"] = await resolve_run_licence(
                session, tenant_id, config_snapshot.get("license_id")
            )
            # What the stages read LIVE (video_sources, uploaded documents), frozen
            # at start so a later Restart of this run can tell whether they moved.
            config_snapshot[RESUME_INPUTS_KEY] = await build_resume_inputs(
                session,
                blueprint=blueprint,
                project_id=proj.id,
                config_snapshot=config_snapshot,
            )
            _record_regenerate_request(
                config_snapshot,
                notes_by_stage=notes_by_stage,
                from_stage=from_stage if parent_run is not None and from_stage_explicit else None,
            )

            if parent_run is not None:
                # A child that INHERITS its parent's narration must be judged against
                # the order that narration was authored for. The preserved set is
                # resolved above; the ordered duration is read LIVE off the deliverable
                # by `build_config_snapshot` just above — so this is the first and only
                # point where both halves exist, and it sits BEFORE `run_repo.create`
                # so a refusal costs no paid child run. Raises
                # RegenerateOrderMismatchError (a ValueError); the regenerate route
                # answers a typed 409.
                assert_inherited_narration_matches_the_order(
                    parent_snapshot=parent_run.config_snapshot or {},
                    child_snapshot=config_snapshot,
                    preserved_outputs=preserved_outputs,
                )

            _validate_website_review_outputs(proj.type, config_snapshot.get("brief") or {})
            if proj.type == "website_review":
                # 10s is the product floor (MINIMUM_ORDER_SECONDS), enforced whether or
                # not title cards are on and re-checked HERE because a deliverable's
                # order can be shortened after the create door passed it. Raises
                # OrderBelowFloorError (a ValueError); the route answers a typed 400.
                validate_website_review_order(config_snapshot)

            # #1106 BYO preflight: a byo_required tenant whose run needs a
            # platform-GPU modality (image/tts/asr) it has no connection for is
            # refused HERE — before the run row exists and before any LLM stage
            # spends money. Raises SlotNotConfiguredError (402 byo_provider_required)
            # or, on a licences outage, ByoPostureUnavailableError (503, retryable).
            await preflight_byo(
                blueprint,
                config_snapshot,
                tenant_id,
                posture=adapters.get_byo_posture(),
                skip_stages=preserved_outputs,
            )

            run = await run_repo.create(
                deliverable_id=deliverable_id,
                blueprint_id=blueprint.id,
                config_snapshot=config_snapshot,
                pause_after_stages=pause_after_stages,
                parent_run_id=parent_run_id,
                structural_relief_pass=structural_relief_pass,
                seeded_pause_after_stages=seeded,
            )

            await insert_initial_stage_rows(
                session,
                stage_repo,
                run_id=run.id,
                blueprint=blueprint,
                preserved_outputs=preserved_outputs,
            )
            if parent_run_id is not None and preserved_outputs:
                # An inherited output was computed from the PARENT's read set; carry
                # it so a later Restart of this child can still tell which inputs
                # those stages depend on (NULL would re-run them on any change).
                await stage_repo.copy_config_reads_from(
                    run.id, parent_run_id=parent_run_id, stage_names=list(preserved_outputs)
                )

            if parent_run_id is not None and affected_so_ids:
                # A surgical single-clip regen flips ONLY the target clip's open
                # feedback: the spine stage_output can carry sibling clips' open
                # annotations that must stay open. reorder + whole-run regenerate
                # re-run the whole spine, so they flip wholesale (element_id=None).
                await apply_open_annotations_in_transaction(
                    session=session,
                    affected_stage_output_ids=affected_so_ids,
                    new_run_id=run.id,
                    element_id=regen_clip["element_id"] if regen_clip is not None else None,
                )

            await session.commit()
            await self._dispatch(run.id)
            return run
