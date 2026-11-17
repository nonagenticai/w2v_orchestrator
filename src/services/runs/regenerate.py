"""Regenerate semantics — preserve completed stages before the first annotated
stage; re-run from there with notes folded into prompts.

Called by ``RunEngine.start_run`` when ``parent_run_id`` is given. Runs inside
the caller's transaction so annotation-flipping is atomic with the new run
insert.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Collection, Mapping
from typing import Any, cast
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.run_providers import bound_existing_run_scope
from src.clients.tts_client import TTSClient
from src.database.models import Run, StageOutput
from src.database.repositories.annotation_repository import AnnotationRepository
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.services.blueprints.base import DAGBlueprint
from src.services.runs.elements import (
    compile_cut_stage,
    element_ids_in,
    element_in,
    element_line_id,
    patch_elements_in_output,
    reorder_elements_in_output,
)
from src.services.stages.cro_audit.narrate import CroAuditNarrateStage
from src.services.stages.demo_video.narrate import DemoVideoNarrateStage
from src.services.stages.generic.narrate import NarrateStage
from src.services.stages.lesson.narrate import LessonNarrateStage
from src.services.stages.primitives.narration import NarrationStageBase, synthesize_and_store
from src.services.stages.promo.narrate import PromoNarrateStage
from src.services.stages.seo_audit.narrate import SeoAuditNarrateStage
from src.services.stages.shared.clip_edit_fields import EDITABLE_CLIP_FIELDS
from src.services.stages.website_review.record import WebsiteReviewRecordStage

# Per-clip edit fields the Studio may override on a spine line. Each threads
# spine-line → compositor ``Shot`` in ``assemble_shots``, so an edit re-stitches
# by preserving every pre-compile stage (edited spine) and re-running only the
# compile cut — exactly like a reorder.
#   clip_lead_seconds: skip N leading seconds of the recorded footage (drop the
#     app's blank load-in) — ``_attach_clip_framing`` → ``Shot.clip_lead_seconds``.
#   clip_out_seconds: the window's OUT-point on the source timeline, closing the
#     window ``clip_lead_seconds`` opens — ``_attach_clip_framing`` →
#     ``Shot.clip_out_seconds``. Listed here for the same reason as
#     ``audio_seconds``: a field missing from the frozenset is SILENTLY DROPPED on
#     merge, so an out-point authored in the Studio would 200 and then do nothing.
#   clip_anchor: anchor the footage window at its "start" or "end".
#   caption: burned-in caption text; an authored value wins over the computed
#     map AND emits even when captions_mode != burned_in (``assemble_shots``).
#   fit: compositor fit ("contain"/"cover"/"scroll"); authored value wins.
#   speed: per-clip playback rate; transition: entering mix/wipe cut
#     (``_attach_clip_playback`` → ``Shot.speed``/``Shot.transition``).
#   adjust: per-clip colour grade (brightness/contrast/saturation/gamma + preset);
#     overlays: burned-in text titles — both via ``_attach_clip_look`` →
#     ``Shot.adjust``/``Shot.overlays``.
#   audio: per-clip audio level (gain/mute/fade-in/fade-out) on the shot's primary
#     audio — via ``_attach_clip_audio`` → ``Shot.audio``.
#   audio_url / audio_seconds: an uploaded voiceover REPLACING this shot's TTS
#     narration + its server-probed duration (``POST /api/uploads/voiceover``).
#     ``assemble_shots`` prefers the spine's ``audio_url`` over the narrate clip's
#     and lets ``audio_seconds`` win the duration reconcile. A field missing from
#     the frozenset below is SILENTLY DROPPED on merge (``patch_elements_in_output``),
#     so BOTH must be listed or the upload lands nowhere.
# DELIBERATELY NOT editable — the A22/C6 timeline fields (``time_lock`` /
# ``min_seconds`` / ``max_seconds`` / ``vo_rate``). They ARE authoring controls and
# ``reconcile_timeline`` does honour an authored value over its computed default (C6),
# but NOT via this per-clip path, because a clip edit re-runs only from the compile cut
# (``compile_cut_stage`` → ``terminal_artifact_stage``) and ``reconcile_timeline`` sits
# strictly UPSTREAM of it. So a per-clip edit here would be a control that lies:
#   * ``time_lock`` — read only by ``reconcile_timeline``, which never re-runs ⇒ a pure
#     silent no-op.
#   * ``min_seconds`` / ``max_seconds`` — compile WOULD clamp that one shot (they reach
#     ``assemble_shots`` via ``_min_max_by_line``), but the solver never re-runs, so no
#     other beat gives the seconds back: the total duration silently drifts off target.
#   * ``vo_rate`` — a COMPUTED OUTPUT of the solver (the speaking rate needed to fit
#     narration into the allotted seconds), never an authoring control; setting it would
#     fight the solver and desync the TTS from the cut.
# Author them on the brief/storyboard spine instead (``authored_storyboard``), where the
# full pipeline — solver included — runs and the whole timeline rebalances.
# TODO(C13): to make these live per-clip, the clip-edit cut point must move upstream to
# ``reconcile_timeline`` whenever a timeline field changed, so the budget rebalances.
_EDITABLE_CLIP_FIELDS = EDITABLE_CLIP_FIELDS

# Per-clip image stages: they generate one visual per clip and reuse unchanged
# clips by content-addressed media id, so re-running one regenerates only the
# clip whose prompt changed. Surgical single-clip regen cuts at this stage.
_IMAGE_PRODUCER_STAGES = ("ai_image_gen",)

# Every ``NarrationStageBase`` subclass: one TTS clip per source-stage line,
# keyed by that line's id. Re-narrate resolves the run's narration stage from
# this table so it can patch that stage's PRESERVED output in place (see
# ``build_renarrate_state``) — the classes themselves are the table, so a new
# narrate stage cannot drift out of it by forgetting to add its name here.
_NARRATION_PRODUCERS: tuple[type[NarrationStageBase[Any, Any]], ...] = (
    NarrateStage,
    PromoNarrateStage,
    LessonNarrateStage,
    DemoVideoNarrateStage,
    WebsiteReviewRecordStage,
    CroAuditNarrateStage,
    SeoAuditNarrateStage,
)
_NARRATION_PRODUCER_STAGES: tuple[str, ...] = tuple(c.name for c in _NARRATION_PRODUCERS)
_NARRATION_BY_STAGE: dict[str, type[NarrationStageBase[Any, Any]]] = {
    c.name: c for c in _NARRATION_PRODUCERS
}


log = logging.getLogger(__name__)


#: Every snapshot path a blueprint's own resolver reads to learn what length was
#: ORDERED. The union of ``shared/duration_conformance``'s per-blueprint resolvers
#: (``promo_ordered_seconds`` / ``lesson_ordered_seconds`` / ``review_ordered_seconds``
#: / ``demo_video_ordered_seconds``), ``website_review/order.website_review_ordered_
#: seconds`` and ``action_reel/target_duration.target_seconds``.
#:
#: It is deliberately a UNION of raw paths rather than a call into those resolvers.
#: This module does not need to know which scalar a given blueprint would resolve to
#: — it only needs to know whether ANY input the order is read from moved between a
#: parent and its child. A union over-reports at worst (a blueprint that ignores a
#: key still refuses when only that key moved), and a per-blueprint dispatch here
#: would be a SECOND copy of a precedence that has already drifted once
#: (``promo_ordered_seconds``' docstring records exactly that bug).
#:
#: ``brief.duration_minutes`` is carried in its authored unit, not scaled to seconds:
#: nothing compares these values across paths, only the same path across two
#: snapshots, so converting would add arithmetic with no reader.
ORDERED_DURATION_PATHS: tuple[tuple[str, ...], ...] = (
    ("deliverable", "target_duration_seconds"),
    ("brief", "target_duration_seconds"),
    ("brief", "duration_seconds"),
    ("brief", "duration_minutes"),
    ("direction", "timing", "target_seconds"),
    ("target_duration_seconds",),
)


def _numeric(value: object) -> float | None:
    """``value`` as a float, or ``None`` when it is not a usable number.

    ``bool`` is excluded explicitly (it is an ``int`` subclass, so ``True`` would
    otherwise read as a 1-second order).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def ordered_duration_inputs(config_snapshot: Mapping[str, Any]) -> dict[str, float]:
    """``dotted.path -> value`` for every :data:`ORDERED_DURATION_PATHS` entry present.

    An absent, null or non-numeric key is OMITTED rather than mapped to a default:
    "the caller did not state a length" and "the caller ordered the blueprint's
    fallback" must not compare equal, because only the first of the two can be
    satisfied by inherited narration authored under the other.
    """
    found: dict[str, float] = {}
    for path in ORDERED_DURATION_PATHS:
        section: Mapping[str, object] = config_snapshot
        for segment in path[:-1]:
            child = section.get(segment)
            if not isinstance(child, Mapping):
                section = {}
                break
            section = cast("Mapping[str, object]", child)
        seconds = _numeric(section.get(path[-1]))
        if seconds is not None:
            found[".".join(path)] = seconds
    return found


def ordered_duration_changes(
    parent_snapshot: Mapping[str, Any], child_snapshot: Mapping[str, Any]
) -> dict[str, tuple[float | None, float | None]]:
    """``dotted.path -> (parent value, child value)`` for every ordered-duration move.

    Empty when the two snapshots order the same length. A key present on one side
    only shows as ``None`` on the other — see :func:`ordered_duration_inputs`.
    """
    parent_inputs = ordered_duration_inputs(parent_snapshot)
    child_inputs = ordered_duration_inputs(child_snapshot)
    return {
        path: (parent_inputs.get(path), child_inputs.get(path))
        for path in sorted(set(parent_inputs) | set(child_inputs))
        if parent_inputs.get(path) != child_inputs.get(path)
    }


class RegenerateOrderMismatchError(ValueError):
    """A child that INHERITS its parent's narration but was handed a DIFFERENT order.

    A regenerate freezes every stage before its cut point, and for every blueprint in
    :data:`_NARRATION_PRODUCERS` that cut sits downstream of the narrate stage — so the
    child's spoken track is the parent's, verbatim, byte for byte. The ordered duration
    is NOT frozen with it: ``build_config_snapshot`` reads ``deliverable
    .target_duration_seconds`` LIVE off the row, so a deliverable edited between the two
    runs hands the child a length its inherited narration was never written for.

    Nothing downstream reconciles the two. **A film is exactly its narration** — the
    compiler adds only the title cards — so the delivered duration cannot move, and the
    only lever that could move it (re-authoring the script to a new word count) sits
    UPSTREAM of the cut and does not re-run.

    Measured in prod, one ``website_review`` family, all four runs delivering the same
    film to within 0.088s across three days::

        01a09f82-7179 (parent)  ordered 10  delivered 15.050  +50.50%
        01a0aa08-4282           ordered 10  delivered 15.138  +51.38%
        01a0aeae-db0d           ordered 50  delivered 15.138  -69.72%
        01a0b00d-f914           ordered 10  delivered 15.058  +50.58%

    The two children 120 percentage points apart delivered 15.138s and 15.058s: the
    differing input is the ORDER FIELD, not the film. And -69.72% is the unrecoverable
    direction — no extend lever exists downstream of the cut, so the run cannot even
    fail usefully. It completed, ``output->>'degraded'`` NULL, with
    ``website_review_compile`` having recorded ``duration_out_of_tolerance`` as a note
    no caller vetoes on.

    The blueprint itself is fine: an ORIGINAL (non-regenerate) run of the same
    blueprint ordered 50s delivered 46.737s, -6.53%. Only the regenerate cannot reach a
    new order, because it never re-authors to it.

    WHY REFUSING, AND NOT ONE OF THE TWO REPAIRS
    -------------------------------------------
    * Inheriting the parent's order instead (pin the snapshot to the parent's number)
      would persist a run whose ``config_snapshot`` contradicts the deliverable row —
      and the deliverable row is what the Studio shows, what a later Restart
      fingerprints (``order_fingerprint``) and what the next run reads. It swaps a
      visible incoherence for a silent one and gives the customer no signal that the
      50s they typed was discarded.
    * Moving the cut point upstream so the narration re-authors would convert a cheap
      narrow regenerate into a full re-render. Peers bill per HTTP call on any 2xx and
      the orchestrator emits no veto, so the caller would be charged for fresh TTS and
      fresh LLM work they did not ask for — exactly the harm ``RegenerateScopeError``
      exists to refuse, in the opposite direction.

    So the door answers, and names both ways forward: restore the order, or ask for a
    full re-run (omit ``from_stage``), which DOES re-author narration to the new
    length. The caller decides which, because only the caller knows whether they meant
    to change the length or clicked the wrong button.
    """

    def __init__(
        self,
        *,
        changes: Mapping[str, tuple[float | None, float | None]],
        narration_stages: list[str],
    ) -> None:
        self.changes = dict(changes)
        self.narration_stages = narration_stages
        moved = "; ".join(
            f"{path}: {_render(before)} -> {_render(after)}"
            for path, (before, after) in sorted(self.changes.items())
        )
        super().__init__(
            f"this regenerate inherits the parent's narration ({', '.join(narration_stages)}) "
            f"but was handed a different ordered duration ({moved}). The inherited audio "
            f"fixes the film's length, and no stage downstream of the cut can re-author it, "
            f"so the child would deliver the parent's film against the new order. Restore "
            f"the ordered duration, or omit 'from_stage' to request a full re-run that "
            f"re-authors the narration to the new length."
        )


def _render(value: float | None) -> str:
    """A duration input for the refusal message: ``<unset>``, or trimmed of ``.0``."""
    return "<unset>" if value is None else f"{value:g}"


def assert_inherited_narration_matches_the_order(
    *,
    parent_snapshot: Mapping[str, Any],
    child_snapshot: Mapping[str, Any],
    preserved_outputs: Mapping[str, Any],
) -> None:
    """Raise :class:`RegenerateOrderMismatchError` on a frozen-narration order change.

    Both conditions are required, and each on its own is legitimate:

    * narration inherited, order unchanged — the ordinary regenerate / reorder / clip
      edit. Nothing to reconcile.
    * order changed, narration NOT inherited — the cut sits at or above the narrate
      stage, so the child re-authors to the new length. That is the flow that measured
      50s -> 46.737s, and it must keep working.

    Only their intersection is incoherent, so only their intersection is refused. And
    within that intersection, only a BOTH-SET-AND-DIFFERENT move counts as a
    contradiction. ``config_snapshot`` is not always a full record of what was
    ordered — a parent seeded (or persisted) before an order field was ever written
    carries no key for it at all, and ``build_config_snapshot`` reads the child's
    value LIVE off the deliverable row regardless. ``<unset> -> 30`` there is a first
    order being placed where none existed, not a changed order contradicting a frozen
    narration: there is no prior order for the inherited narration to contradict, so
    nothing is refused. Only when BOTH sides name a duration and they differ is there
    an actual prior order the child's narration was never written for — the measured
    defect (parent ordered 10, child ordered 50, same frozen ~15.1s film, child scored
    -69.72%). A side that is merely absent must not be treated as a difference.
    """
    inherited = [name for name in _NARRATION_PRODUCER_STAGES if name in preserved_outputs]
    if not inherited:
        return
    changes = {
        path: (before, after)
        for path, (before, after) in ordered_duration_changes(
            parent_snapshot, child_snapshot
        ).items()
        if before is not None and after is not None
    }
    if not changes:
        return
    raise RegenerateOrderMismatchError(changes=changes, narration_stages=inherited)


class RegenerateError(ValueError):
    """A regenerate directive that names a stage outside the run's blueprint."""


class RegenerateScopeError(ValueError):
    """An EXPLICIT narrow regenerate that would silently widen to a full re-run.

    A caller that names ``from_stage`` is asking for a NARROW re-run: everything the
    parent completed outside the stage's descendant subtree must be inherited. When
    that inheritance does not happen, the child re-runs — and re-bills — work the
    caller never asked to redo: fresh TTS, fresh LLM, a DIFFERENT script. Peers bill
    per HTTP call on any 2xx, so the widening is invisible and expensive.

    Measured in prod 2026-09-16: ``regenerate {"fromStage": "website_review_compile"}``
    came back re-running ``website_redesign`` with script/record/slides/visual_director
    all ``pending``. The parent's stage_outputs read back empty (the probe ran under a
    cross-tenant scope switch, so ``list_for_run`` saw nothing under the wrong RLS GUC).

    ⚠️ THE BASELINE IS THE PARENT'S OWN ROWS, NOT THE BLUEPRINT. The first version of
    this guard compared ``preserved`` against ``set(topological_order) - rerun_region``
    — what the blueprint SAYS should exist. That is wrong and it broke a real, common
    flow: a reviewer paused at the first gate clicks "rerun this stage", and the parent
    has legitimately completed nothing outside the subtree, so the guard announced that
    N stages "should have been inherited" about N stages that never ran (CI:
    ``test_regenerate_from_stage_creates_child``, 409 instead of 201). What the parent
    ACTUALLY recorded is the only sound baseline.

    Two shapes are refused, each with its own ``cause``:

    - ``parent_never_ran`` — the parent is cancelled/failed and has NO stage_output
      rows: it stopped before dispatch wrote any (cancelled while pending, or failed at
      start). Nothing failed to read — there is nothing to inherit.
    - ``parent_outputs_unreadable`` — the parent is paused/completed (so it ran) and
      yet NOT ONE stage_output row is readable. That is the prod shape: the read
      itself failed, and nothing about it can be trusted, so no narrow promise can be
      kept.
    - ``preserved_outputs_incomplete`` — a stage OUTSIDE the re-run region is recorded
      ``completed`` on the parent but produced no preserved entry (a NULL output). That
      stage is about to be silently re-run and re-billed.

    WHY THE TOTAL CASE IS THE ONLY ONE THAT NEEDS CATCHING — this is an invariant, not
    an accepted gap. An earlier version of this docstring called "RLS hides 8 of 9 rows"
    an undetectable residual. It is better than that: **that shape is not producible.**
    Migration ``0028_rls_runtree`` makes the ``stage_outputs`` policy a join-based
    ``EXISTS`` that climbs to ``projects.tenant_id`` — the predicate is a property of the
    RUN, not of the individual row — so visibility is all-or-nothing per run. A wrong
    tenant GUC hides every row of the parent or none of them, which is exactly what
    ``parent_outputs_unreadable`` catches. A partial read would require a per-row
    predicate, and there is none.
    (``preserved_outputs_incomplete`` covers the different, real case: rows are visible
    but a completed one carries no usable output.)

    The endpoint REFUSES rather than guessing: a caller who genuinely wants a full
    re-run omits ``from_stage``. Only an EXPLICIT request raises — the implicit
    annotation-driven / failed-parent-resume path keeps its "preserve nothing, re-run
    all" fallback unchanged, because it never promised anything narrow.
    """

    def __init__(self, from_stage: str, *, cause: str, stages: list[str]) -> None:
        self.from_stage = from_stage
        self.cause = cause
        self.stages = stages
        if cause == "parent_never_ran":
            detail = (
                "the run stopped before any of its steps ran, so there is nothing to "
                "keep. Restart it without choosing a step to run it from the beginning"
            )
            super().__init__(f"cannot regenerate from stage '{from_stage}': {detail}.")
            return
        if cause == "parent_outputs_unreadable":
            detail = (
                "not one of the parent run's stage outputs is readable, so nothing "
                "could be inherited"
            )
        else:
            detail = (
                f"the parent completed {', '.join(stages)} outside '{from_stage}' but "
                f"none of those outputs could be inherited"
            )
        super().__init__(
            f"cannot regenerate from stage '{from_stage}': {detail}. This narrow request "
            f"would silently re-run (and re-bill) work you did not ask to redo. Omit "
            f"'from_stage' to request a full re-run deliberately."
        )


async def build_regenerate_state(
    *,
    session: AsyncSession,
    parent_run: Run,
    blueprint: DAGBlueprint,
    from_stage: str | None = None,
    from_stage_explicit: bool = False,
    rerun_roots: Collection[str] | None = None,
    rerun_after: Collection[str] = (),
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    list[UUID],
]:
    """Return ``(preserved_outputs, notes_by_stage, affected_stage_output_ids)``.

    ``rerun_roots`` / ``rerun_after`` carry a planned Restart
    (``services/runs/restart_plan.py``): the re-run region is the descendant closure
    of ``rerun_roots`` (plus any stage with open feedback) UNION the strict
    descendants of ``rerun_after`` (stages the user edited — reused themselves, their
    dependents re-run). Mutually exclusive with ``from_stage``.

    ``preserved_outputs``: ``{stage_name: output_dict}`` for every completed
    stage that is NOT in the re-run descendant-region. The new run inherits these
    immutably; the engine inserts them as completed stage_outputs on the child.

    The re-run region is the DESCENDANT CLOSURE of the roots — ``{roots}`` plus
    every stage that (in)directly depends on a root — NOT the topological prefix.
    A parallel no-dependency root (e.g. ``source_ingest``, a paid scrape) sorts
    AFTER a gen stage in topological order without depending on it, so the old
    ``order[:partition]`` prefix wrongly excluded it and re-ran (re-billed) it on
    every pass. The roots are:
    - ``from_stage`` when given — the "rerun this stage" action: ``{from_stage}``.
      Its whole descendant subtree re-runs; everything else completed is preserved.
      Raises ``RegenerateError`` if ``from_stage`` is not a stage of the blueprint.
      When ``from_stage_explicit`` (the CALLER named it, as opposed to it being
      derived from a failed parent), raises ``RegenerateScopeError`` if the
      preserved set comes back empty while the blueprint says stages outside the
      subtree should have been inherited — a narrow request must never silently
      widen into a full, re-billed re-run.
    - otherwise every stage with open annotations (annotation-driven regenerate);
      if there are none, preserve nothing and re-run everything.

    ``notes_by_stage``: ``{stage_name: [{text, element_id}]}`` — open
    annotations at stages in the re-run region, grouped by parent stage_name.
    Persisted on ``runs.config_snapshot["regenerate_notes"]`` so the dispatcher
    can fold them into ``RunContext.open_annotations_by_stage``. Annotations on
    preserved (out-of-region) stages stay open and are NOT applied — their
    stage output is carried over verbatim.

    ``affected_stage_output_ids``: the parent stage_outputs in the re-run region
    that have open annotations, for the bulk-apply UPDATE.
    """
    so_repo = StageOutputRepository(session)
    ann_repo = AnnotationRepository(session)
    parent_outputs = await so_repo.list_for_run(parent_run.id)
    by_stage = {so.stage_name: so for so in parent_outputs}
    open_by_stage = await ann_repo.list_open_by_stage_for_run(parent_run.id)

    order = blueprint.topological_order()

    resolved = _regenerate_roots(
        order=order,
        open_by_stage=open_by_stage,
        from_stage=from_stage,
        rerun_roots=rerun_roots,
        rerun_after=rerun_after,
    )
    if resolved is None:
        # No notes, no target — vanilla re-run. Preserve nothing; re-run all.
        return {}, {}, []
    roots, after = resolved

    # Re-run region = descendant closure of the roots. INVARIANT: on a LINEAR
    # chain a→b→c→d, descendants(from_stage) == order[order.index(from_stage)+1:],
    # so rerun_region == order[partition:] and preserved == order[:partition] —
    # BYTE-IDENTICAL to the old topological-prefix behaviour. Only DAGs with a
    # parallel root (a stage topo-sorted after a root it does NOT depend on) or
    # parallel sibling branches change: those out-of-subtree stages are now
    # preserved instead of needlessly (and expensively) re-run.
    rerun_region: set[str] = set()
    for r in roots:
        rerun_region.add(r)
        rerun_region |= blueprint.descendants(r)
    for e in after:
        rerun_region |= blueprint.descendants(e)

    preserved: dict[str, dict[str, Any]] = {
        stage_name: dict(so.output)
        for stage_name, so in by_stage.items()
        if stage_name not in rerun_region and so.status == "completed" and so.output is not None
    }

    # Only fold/flip annotations on stages that actually re-run. Notes on
    # preserved (out-of-region) stages stay open — their output is carried over
    # verbatim, so applying them now would be a lie.
    notes_by_stage = {
        stage_name: [{"text": a.text, "element_id": a.element_id} for a in anns]
        for stage_name, anns in open_by_stage.items()
        if stage_name in rerun_region
    }
    affected_so_ids = [
        by_stage[stage_name].id
        for stage_name in open_by_stage
        if stage_name in by_stage and stage_name in rerun_region
    ]

    # A narrow re-run promises that everything the PARENT completed outside the subtree
    # is inherited. Detect a break of that promise here, once, so EVERY caller is
    # covered — then split on who asked:
    #   * an EXPLICIT caller (a human clicking "rerun this stage") gets a typed 4xx;
    #   * the AUTOMATIC callers get a WARNING and proceed (see below).
    if from_stage is not None:
        widening = _detect_silent_widening(
            parent_status=parent_run.status,
            from_stage=from_stage,
            order=order,
            rerun_region=rerun_region,
            by_stage=by_stage,
            preserved=preserved,
        )
        if widening is not None:
            cause, dropped = widening
            if from_stage_explicit:
                raise RegenerateScopeError(from_stage, cause=cause, stages=dropped)
            # ⛔ REPORT ONLY for the automatic callers — `_spawn_music_rewind_child`
            # (lifecycle.py) and `structural_relief.py`'s self-healing loop, plus the
            # failed-parent resume that derives a from_stage the caller never sent.
            # Refusing there could STALL the pipeline mid-run with no human present, so
            # the widening is made VISIBLE rather than fatal. Owner decision: do not
            # make an automatic loop refuse.
            log.warning(
                "regenerate scope widened silently (report-only, automatic caller): "
                "run=%s from_stage=%s cause=%s dropped=%s — work outside the re-run "
                "region will be re-run and re-billed",
                parent_run.id,
                from_stage,
                cause,
                dropped or "<all>",
            )

    return preserved, notes_by_stage, affected_so_ids


def _regenerate_roots(
    *,
    order: list[str],
    open_by_stage: Mapping[str, object],
    from_stage: str | None,
    rerun_roots: Collection[str] | None,
    rerun_after: Collection[str],
) -> tuple[set[str], set[str]] | None:
    """``(roots, after)`` for :func:`build_regenerate_state`, or ``None`` = re-run all.

    ``roots`` re-run with every descendant; ``after`` (edited stages) only drag
    their descendants in. Precedence: a planned restart, then ``from_stage``, then
    the stages with open feedback.
    """
    if rerun_roots is not None:
        if from_stage is not None:
            raise RegenerateError("a planned restart cannot also name from_stage")
        unknown = sorted((set(rerun_roots) | set(rerun_after)) - set(order))
        if unknown:
            raise RegenerateError(f"stage(s) {unknown} are not stages of this run's blueprint")
        return set(rerun_roots) | set(open_by_stage), set(rerun_after)
    if from_stage is not None:
        if from_stage not in order:
            raise RegenerateError(f"stage '{from_stage}' is not a stage of this run's blueprint")
        return {from_stage}, set()
    if not open_by_stage:
        return None
    return set(open_by_stage), set()


def _detect_silent_widening(
    *,
    parent_status: str,
    from_stage: str,
    order: list[str],
    rerun_region: set[str],
    by_stage: Mapping[str, StageOutput],
    # `Mapping[str, object]`, not `Mapping[str, dict[str, Any]]`: this helper only
    # tests MEMBERSHIP, and the tighter spelling would add a `dict[str, Any]` against
    # the ratchet in tests/test_any_ratchet.py for no type safety gained.
    preserved: Mapping[str, object],
) -> tuple[str, list[str]] | None:
    """``(cause, dropped_stages)`` when a narrow re-run would silently widen, else ``None``.

    Measured against the parent's OWN rows, never against the blueprint — see
    ``RegenerateScopeError`` for the flow the blueprint-derived baseline broke.
    """
    if not set(order) - rerun_region:
        # A DAG root's subtree IS the whole blueprint: nothing could be inherited, so a
        # full re-run is the literal correct answer to what was asked. Not a widening.
        return None
    if not by_stage:
        if parent_status in ("cancelled", "failed"):
            # Stopped BEFORE dispatch wrote a single stage row (cancelled while
            # pending, or failed at start). No read failed: there is simply nothing
            # to inherit, so a narrow "from this stage" cannot be honoured.
            return ("parent_never_ran", [])
        # The parent is paused/completed — it RAN — yet not one stage_output row came
        # back. The read itself failed; no narrow promise can be kept on it.
        return ("parent_outputs_unreadable", [])
    # Not just "preserved is empty": ANY stage the parent recorded as completed outside
    # the re-run region that failed to become a preserved output is about to be
    # silently re-run and re-billed.
    dropped = sorted(
        stage_name
        for stage_name, so in by_stage.items()
        if stage_name not in rerun_region
        and so.status == "completed"
        and stage_name not in preserved
    )
    return ("preserved_outputs_incomplete", dropped) if dropped else None


class ReorderError(ValueError):
    """A reorder directive that cannot be honoured (bad stage, order, or ids)."""


def _mirror_array_keys(output: dict[str, object], ids: set[str]) -> list[str]:
    """Top-level keys of ``output`` holding a list of ``{"id": ...}`` dicts covering ``ids``.

    Deliberately keyed on the DATA (an array whose id set is exactly ``ids``) and
    not on ``elements._ELEMENT_PATHS``: the mirror stages this exists for are not
    — and must not become — registered element paths (see
    ``_reorder_mirror_stages``). Returns ``[]`` for an empty ``ids``.
    """
    if not ids:
        return []
    keys: list[str] = []
    for key, value in output.items():
        if not isinstance(value, list) or not value:
            continue
        node_ids = {
            node["id"]
            for node in value
            if isinstance(node, dict) and isinstance(node.get("id"), str)
        }
        if node_ids == ids:
            keys.append(key)
    return keys


def _reorder_mirror_stages(
    preserved: dict[str, dict[str, object]],
    *,
    spine_stage: str,
    spine_ids: set[str],
    spine_keys: list[str],
    ordered_ids: list[str],
) -> None:
    """Apply the spine's new element order to every preserved stage that MIRRORS it.

    WHY this exists — the precedence-vs-spine inversion
    ---------------------------------------------------
    A reorder rewrites the SPINE stage's element array and preserves every other
    pre-cut stage verbatim. That is only correct while the spine is the array the
    compile stage actually reads. On the ``demo_video`` family it is not.

    Four optional stages (``script_doctor``, ``claim_finish``, ``retention_plan``,
    ``reconcile_timeline``) each RE-EMIT the ``demo_video_gen`` shape — the same
    ``lines`` list, same ids, "order preserved by construction"
    (``stages/demo_video/spine.py``). Because they are re-emissions, downstream
    consumers read the LATEST one via ``resolve_plan_output()``, whose
    ``PLAN_STAGE_PRECEDENCE`` ranks ``reconcile_timeline`` highest and the raw
    ``demo_video_gen`` LOWEST. But ``demo_video_gen`` is the only stage a client
    may legally pass as the spine (``resolve_clip_spine_stage``).

    So spine and reader are inverted: the reorder lands on the lowest-precedence
    array, the compile stage stitches from the highest-precedence one, and the
    child run completes GREEN with a byte-identical video. Nothing fails; the
    feature is simply inert. "ids / order preserved by construction" is exactly
    the assumption a reorder violates.

    The fix, and why it is shaped like this
    ---------------------------------------
    Mirror the reorder onto every preserved stage that carries the SAME ARRAY
    under the SAME KEY with the SAME SET OF IDS as the spine. That match is the
    definition of "re-emits the spine shape", so it is self-limiting: it catches
    all four demo_video mirrors, and the equivalent mirrors in promo / lesson /
    app_review, without this generic module importing a single blueprint-specific
    stage name. A stage carrying a different id set (a different array entirely,
    or a stale/partial subset) is left untouched — it is not a mirror and
    reordering it would be a guess.

    It is matched on the array KEY rather than through
    ``elements.stage_supports_element_pinning`` on purpose: three of the four
    mirrors are NOT in ``_ELEMENT_PATHS`` and must stay out of it, because that
    registry is also what makes a stage a LEGAL reorder TARGET. They must be
    reorder*ed*, never reorder*able*.

    Mutates ``preserved`` in place. Do not "simplify" this back to preserving
    non-spine stages verbatim — that is the bug.
    """
    rank = {eid: i for i, eid in enumerate(ordered_ids)}
    for stage_name, out in preserved.items():
        if stage_name == spine_stage:
            continue
        for key in spine_keys:
            nodes = out.get(key)
            if not isinstance(nodes, list) or not nodes:
                continue
            node_ids = {
                node["id"]
                for node in nodes
                if isinstance(node, dict) and isinstance(node.get("id"), str)
            }
            if node_ids != spine_ids:
                continue
            # `.get`-guarded like every sibling helper in ``elements.py``: a
            # malformed/degraded node without an id sorts to the end rather than
            # crashing the re-stitch.
            out[key] = sorted(
                nodes,
                key=lambda n: (
                    rank.get(n.get("id"), len(rank)) if isinstance(n, dict) else len(rank)
                ),
            )


async def build_reorder_state(
    *,
    session: AsyncSession,
    parent_run: Run,
    blueprint: DAGBlueprint,
    spine_stage: str,
    ordered_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Return ``preserved_outputs`` for a clip-reorder re-stitch.

    Reordering the timeline reuses every generated clip asset: only the compiled
    video needs rebuilding. So preserve every completed parent stage strictly
    BEFORE the blueprint's compile cut (``terminal_artifact_stage``), with the
    ``spine_stage`` output's element array reordered to ``ordered_ids``. The
    child run inherits these as completed stage_outputs, so the DAG executor
    (which runs only stages absent from the seeded outputs) re-runs just the
    compile stage onward — re-stitching in the new order.

    The new order is ALSO applied to every preserved stage that mirrors the
    spine's element array (same key, same id set) — see
    ``_reorder_mirror_stages`` for why reordering the spine alone leaves the
    reorder inert on any blueprint whose compile stage reads a higher-precedence
    re-emission of that array.

    Raises ``ReorderError`` (→ HTTP 4xx at the route) if the blueprint has no
    compile stage downstream of the spine, the spine stage output is missing or
    not completed, or ``ordered_ids`` is not a permutation of the spine's ids.
    """
    cut = compile_cut_stage(blueprint, spine_stage)
    if cut is None:
        raise ReorderError(f"stage '{spine_stage}' has no downstream compile stage to re-stitch")

    so_repo = StageOutputRepository(session)
    parent_outputs = await so_repo.list_for_run(parent_run.id)
    by_stage = {so.stage_name: so for so in parent_outputs}

    spine_so = by_stage.get(spine_stage)
    if spine_so is None or spine_so.status != "completed" or spine_so.output is None:
        raise ReorderError(f"stage '{spine_stage}' has no completed output to reorder")

    spine_output = dict(spine_so.output)
    spine_ids = element_ids_in(spine_output, spine_stage)
    reordered = reorder_elements_in_output(spine_output, spine_stage, ordered_ids)
    if reordered is None:
        raise ReorderError("ordered_element_ids must be a permutation of the stage's element ids")

    # The array key(s) the spine reorder actually landed on — the handle used to
    # find the same array on the mirror stages below.
    spine_keys = _mirror_array_keys(spine_output, spine_ids)

    order = blueprint.topological_order()
    preserved: dict[str, dict[str, Any]] = {}
    for stage_name in order[: order.index(cut)]:
        so = by_stage.get(stage_name)
        if so is None or so.status != "completed" or so.output is None:
            continue
        preserved[stage_name] = reordered if stage_name == spine_stage else dict(so.output)
    _reorder_mirror_stages(
        preserved,
        spine_stage=spine_stage,
        spine_ids=spine_ids,
        spine_keys=spine_keys,
        ordered_ids=ordered_ids,
    )
    return preserved


class EditError(ValueError):
    """A clip-edit directive that cannot be honoured (bad stage, or no clip hit)."""


async def build_edit_state(
    *,
    session: AsyncSession,
    parent_run: Run,
    blueprint: DAGBlueprint,
    spine_stage: str,
    edits: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return ``preserved_outputs`` for a per-clip edit re-stitch.

    The exact reorder story, one substitution: instead of reordering the spine's
    element array, shallow-merge each clip's edit fields (``_EDITABLE_CLIP_FIELDS``)
    onto the matching spine element. Every generated clip asset is reused — only
    the compiled video re-renders — so preserve every completed parent stage
    strictly BEFORE the compile cut (``terminal_artifact_stage``) with the edited
    spine output substituted; the DAG executor re-runs just the compile stage
    onward, honouring the new per-clip framing.

    ``edits`` may be empty — a music-only re-stitch (the run-level override rides
    ``config_overrides`` into the child's config snapshot, not the spine); the spine
    is then preserved verbatim and the compile stage re-runs against the new config.

    Raises ``EditError`` (→ HTTP 4xx at the route) if the blueprint has no compile
    stage downstream of the spine, the spine output is missing / not completed, or
    a NON-EMPTY edit set names no id that exists on the spine (a fully-stale set).
    """
    cut = compile_cut_stage(blueprint, spine_stage)
    if cut is None:
        raise EditError(f"stage '{spine_stage}' has no downstream compile stage to re-stitch")

    so_repo = StageOutputRepository(session)
    parent_outputs = await so_repo.list_for_run(parent_run.id)
    by_stage = {so.stage_name: so for so in parent_outputs}

    spine_so = by_stage.get(spine_stage)
    if spine_so is None or spine_so.status != "completed" or spine_so.output is None:
        raise EditError(f"stage '{spine_stage}' has no completed output to edit")

    if edits:
        edited = patch_elements_in_output(
            dict(spine_so.output), spine_stage, edits, allowed_fields=_EDITABLE_CLIP_FIELDS
        )
        if edited is None:
            raise EditError("none of the edited clip ids exist on this stage's timeline")
    else:
        # Music-only re-stitch: spine unchanged, compile re-runs with the new config.
        edited = dict(spine_so.output)

    order = blueprint.topological_order()
    preserved: dict[str, dict[str, Any]] = {}
    for stage_name in order[: order.index(cut)]:
        so = by_stage.get(stage_name)
        if so is None or so.status != "completed" or so.output is None:
            continue
        preserved[stage_name] = edited if stage_name == spine_stage else dict(so.output)
    return preserved


class RegenClipError(ValueError):
    """A single-clip regenerate directive that cannot be honoured surgically."""


def _producer_stage_between(
    blueprint: DAGBlueprint, spine_stage: str, candidates: tuple[str, ...]
) -> str | None:
    """The first stage of ``candidates`` strictly between the spine and the compile cut.

    Returns ``None`` when the blueprint has no compile cut, the spine is not
    strictly upstream of it, or no candidate sits in between.
    """
    cut = blueprint.compile_cut_stage or blueprint.terminal_artifact_stage
    if cut is None:
        return None
    order = blueprint.topological_order()
    if spine_stage not in order or cut not in order or order.index(spine_stage) >= order.index(cut):
        return None
    window = order[order.index(spine_stage) + 1 : order.index(cut)]
    for name in window:
        if name in candidates:
            return name
    return None


def _image_producer_stage(blueprint: DAGBlueprint, spine_stage: str) -> str | None:
    """The per-clip image stage strictly between the spine and the compile cut.

    ``None`` when no per-clip image stage sits in between (e.g. demo_video, whose
    clips are recorded footage rather than generated images).
    """
    return _producer_stage_between(blueprint, spine_stage, _IMAGE_PRODUCER_STAGES)


def _narration_producer_stage(blueprint: DAGBlueprint, spine_stage: str) -> str | None:
    """The per-line narration stage strictly between the spine and the compile cut."""
    return _producer_stage_between(blueprint, spine_stage, _NARRATION_PRODUCER_STAGES)


def _narrate_read_stage(narrate_stage: str, preserved: dict[str, dict[str, Any]]) -> str:
    """The stage whose output the narrate handler actually READS for its line text.

    Usually the handler's declared ``source_stage``. But a stage whose spine may be
    re-emitted downstream (demo_video: ``script_doctor`` → ``retention_plan`` →
    ``reconcile_timeline`` each re-emit the ``demo_video_gen`` shape, ids and order
    preserved) overrides ``resolve_script`` to prefer the LAST re-emission that ran,
    and declares that precedence on ``source_stage_chain``. We have no ``RunContext``
    here, so resolve the same precedence against the preserved outputs: the first
    chain stage that actually produced lines wins; a skipped stage stores a falsy
    ``{}`` and falls through. This is load-bearing — the caption/subtitle text is
    read from THIS stage, so writing the new line text anywhere else would leave the
    subtitle saying the OLD line while the voice says the NEW one.
    """
    cls = _NARRATION_BY_STAGE[narrate_stage]
    for name in cls.source_stage_chain:
        output = preserved.get(name)
        if output and output.get(cls.source_list_key):
            return name
    return cls.source_stage


async def build_regen_clip_state(
    *,
    session: AsyncSession,
    parent_run: Run,
    blueprint: DAGBlueprint,
    spine_stage: str,
    element_id: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    list[UUID],
]:
    """Return ``(preserved_outputs, notes_by_stage, affected_stage_output_ids)``
    for a surgical single-clip regenerate.

    Unlike the annotation-driven regenerate (which cuts at the FIRST annotated
    stage — here the spine/storyboard — and re-runs it wholesale, letting the LLM
    perturb every sibling clip), this cuts at the per-clip IMAGE stage and
    preserves the spine verbatim, so sibling clips are byte-identical. The
    target clip's open feedback is folded into the image stage keyed by the
    clip's ``line_id``, so ``ai_image_gen`` appends it to only that line's prompt
    → its content-addressed ``media_id`` busts → exactly that image regenerates
    while every sibling is reused from storage. The compile stage then
    re-stitches.

    Raises ``RegenClipError`` (→ HTTP 4xx / caller fallback) when a single clip
    cannot be targeted surgically: no per-clip image stage downstream, the image
    stage does not use per-line prompts (``prompts_from_stage`` unset — e.g. the
    generic round-robin pipeline), the spine element is missing / has no
    ``line_id``, or the clip has no open feedback to apply.
    """
    producer = _image_producer_stage(blueprint, spine_stage)
    if producer is None:
        raise RegenClipError(
            f"stage '{spine_stage}' has no downstream per-clip image stage to regenerate"
        )

    # Surgical regen only works when the image stage draws a per-line prompt
    # (``prompts_from_stage``): the fold changes that line's prompt to bust its
    # media id. With static round-robin prompts there is no per-line prompt to
    # change, so we refuse (the caller falls back to annotation-driven regen).
    producer_cfg = (parent_run.config_snapshot or {}).get(producer) or {}
    if not producer_cfg.get("prompts_from_stage"):
        raise RegenClipError(
            f"stage '{producer}' has no per-line prompts; cannot regenerate a single clip surgically"
        )

    so_repo = StageOutputRepository(session)
    ann_repo = AnnotationRepository(session)
    parent_outputs = await so_repo.list_for_run(parent_run.id)
    by_stage = {so.stage_name: so for so in parent_outputs}

    spine_so = by_stage.get(spine_stage)
    if spine_so is None or spine_so.status != "completed" or spine_so.output is None:
        raise RegenClipError(f"stage '{spine_stage}' has no completed output")

    line_id = element_line_id(spine_so.output, spine_stage, element_id)
    if line_id is None:
        raise RegenClipError(
            f"clip '{element_id}' has no line_id; cannot target a single image surgically"
        )

    open_by_stage = await ann_repo.list_open_by_stage_for_run(parent_run.id)
    clip_notes = [a for a in open_by_stage.get(spine_stage, []) if a.element_id == element_id]
    if not clip_notes:
        raise RegenClipError(f"clip '{element_id}' has no open feedback to apply")

    order = blueprint.topological_order()
    preserved: dict[str, dict[str, Any]] = {}
    for stage_name in order[: order.index(producer)]:
        so = by_stage.get(stage_name)
        if so is not None and so.status == "completed" and so.output is not None:
            preserved[stage_name] = dict(so.output)

    # Fold the clip's feedback into the image stage, keyed to the line_id so
    # ai_image_gen appends it to only that line's prompt.
    notes_by_stage = {producer: [{"text": a.text, "element_id": line_id} for a in clip_notes]}
    affected_so_ids = [spine_so.id]
    return preserved, notes_by_stage, affected_so_ids


class RenarrateError(ValueError):
    """A re-narrate directive that cannot be honoured (bad stage/clip, or a
    voiceover upload already owns this line's audio)."""


def _replace_clip_in_narrate_output(
    narrate_output: dict[str, Any],
    *,
    id_field: str,
    line_id: str,
    clip: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a new narrate output with the clip for ``line_id`` swapped for ``clip``.

    Every sibling clip entry is carried over by identity — byte-identical — so the
    recompiled video reuses their stored audio verbatim. ``None`` when the narrate
    stage emitted no clip for that line (nothing to replace).
    """
    clips = narrate_output.get("clips")
    if not isinstance(clips, list):
        return None
    new_clips: list[Any] = []
    found = False
    for entry in clips:
        if isinstance(entry, dict) and entry.get(id_field) == line_id:
            new_clips.append(clip)
            found = True
        else:
            new_clips.append(entry)
    if not found:
        return None
    return {**narrate_output, "clips": new_clips}


def _replace_line_text(
    source_output: dict[str, Any], *, list_key: str, line_id: str, text: str
) -> dict[str, Any]:
    """Return a new source-stage output with ``line_id``'s ``text`` set to ``text``."""
    lines = source_output.get(list_key)
    if not isinstance(lines, list):
        return source_output
    new_lines = [
        {**item, "text": text} if isinstance(item, dict) and item.get("id") == line_id else item
        for item in lines
    ]
    return {**source_output, list_key: new_lines}


def _completed_output(by_stage: dict[str, Any], stage_name: str) -> dict[str, Any]:
    """The stage's completed output dict, or raise ``RenarrateError``."""
    so = by_stage.get(stage_name)
    if so is None or so.status != "completed" or so.output is None:
        raise RenarrateError(f"stage '{stage_name}' has no completed output")
    return dict(so.output)


def _renarrate_line_id(
    *,
    spine_output: dict[str, Any],
    spine_stage: str,
    element_id: str,
    narrate_source_stage: str,
) -> str:
    """The narration line id the spine clip maps to, or raise ``RenarrateError``.

    The narrate fan-out keys each clip by its SOURCE item's ``id``. When the handler
    reads the spine itself (or a re-emission of it — same ids by construction), that
    id IS the spine element id; otherwise the spine element carries a ``line_id``
    join key onto the script it narrates.

    Also the guard for the slice-2 uploaded voiceover: ``assemble_shots`` prefers the
    spine's ``audio_url`` over the narrate clip's, so a re-narrate here would
    synthesize a clip nobody plays. Make that conflict explicit (→ 422) rather than
    silently doing nothing audible.
    """
    element = element_in(spine_output, spine_stage, element_id)
    if element is None:
        raise RenarrateError(f"clip '{element_id}' is not on this stage's timeline")
    if element.get("audio_url"):
        raise RenarrateError("Remove the uploaded voiceover before re-narrating this line.")
    line_id = (
        element_id
        if narrate_source_stage == spine_stage
        else element_line_id(spine_output, spine_stage, element_id)
    )
    if not line_id:
        raise RenarrateError(f"clip '{element_id}' has no line_id; cannot target a single line")
    return line_id


async def build_renarrate_state(
    *,
    session: AsyncSession,
    parent_run: Run,
    blueprint: DAGBlueprint,
    spine_stage: str,
    element_id: str,
    narration_text: str | None = None,
    voice: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return ``preserved_outputs`` for a single-line re-narrate (new text and/or voice).

    ⚠️ We cut at the COMPILE stage, NOT at the narrate stage — and patch the two
    preserved outputs the recompile reads. Cutting at narrate would look surgical
    and is a trap: ``store_tts_clip`` / ``find_existing_clip`` are keyed by
    ``run_id``, and a regenerate always spawns a NEW child run, so
    ``find_existing_clip(child_run_id, …)`` would miss on EVERY line and the whole
    narration would be re-synthesized to change one sentence.

    So instead:
      1. Resolve the narration stage between the spine and the compile cut.
      2. Synthesize EXACTLY ONE clip, stored under the PARENT run id (the clip id
         is content-addressed, so it cannot collide with the parent's own clips).
      3. Patch the line's ``text`` in the stage the narrate handler actually READS
         (``_narrate_read_stage``) — load-bearing, because the captions/subtitles
         derive from that text: changing the audio without the text would leave the
         subtitle on the old line.
      4. Patch the narrate stage's preserved output, swapping only this line's clip
         entry; every sibling clip entry stays byte-identical.
      5. Preserve every completed stage strictly before the compile cut. The DAG
         then re-runs only compile, which re-stitches from ``audio_by_line`` and
         re-derives the shot's ``seconds`` from the new clip duration.

    ⚠️ Known caveat: a per-line ``voice`` override lives ONLY in the synthesized
    audio — it is not persisted on the spine (there is no per-line voice field), so
    a later FULL narrate re-run (an annotation-driven regenerate that cuts upstream
    of the narrate stage) reverts that line to the run's configured voice. Same for
    the fact that we do not forward a per-line ``speed``/``vo_rate``: the re-narrated
    clip is synthesized at the backend default rate.

    Raises ``RenarrateError`` (→ HTTP 422 at the route) when there is no compile cut
    / no narration stage downstream of the spine, the spine or narrate output is
    missing, the clip cannot be resolved to a source line, or the clip already
    carries an uploaded voiceover (whose audio would silently win over the new TTS).
    """
    cut = compile_cut_stage(blueprint, spine_stage)
    if cut is None:
        raise RenarrateError(f"stage '{spine_stage}' has no downstream compile stage to re-stitch")
    narrate_stage = _narration_producer_stage(blueprint, spine_stage)
    if narrate_stage is None:
        raise RenarrateError(f"stage '{spine_stage}' has no downstream narration stage")

    so_repo = StageOutputRepository(session)
    parent_outputs = await so_repo.list_for_run(parent_run.id)
    by_stage = {so.stage_name: so for so in parent_outputs}

    spine_so = _completed_output(by_stage, spine_stage)
    narrate_so = _completed_output(by_stage, narrate_stage)

    cls = _NARRATION_BY_STAGE[narrate_stage]
    line_id = _renarrate_line_id(
        spine_output=spine_so,
        spine_stage=spine_stage,
        element_id=element_id,
        narrate_source_stage=cls.source_stage,
    )

    order = blueprint.topological_order()
    preserved: dict[str, dict[str, Any]] = {}
    for stage_name in order[: order.index(cut)]:
        so = by_stage.get(stage_name)
        if so is not None and so.status == "completed" and so.output is not None:
            preserved[stage_name] = dict(so.output)

    read_stage = _narrate_read_stage(narrate_stage, preserved)
    source_output = preserved.get(read_stage)
    if not source_output:
        raise RenarrateError(f"stage '{read_stage}' has no completed output to re-narrate")

    source_line = next(
        (
            item
            for item in (source_output.get(cls.source_list_key) or [])
            if isinstance(item, dict) and item.get("id") == line_id
        ),
        None,
    )
    if source_line is None:
        raise RenarrateError(f"line '{line_id}' is not on stage '{read_stage}'")

    # Defaults: omitted text ⇒ keep the current line; omitted voice ⇒ the run's
    # configured voice (exactly what the original narrate pass used).
    text = narration_text if narration_text is not None else str(source_line.get("text") or "")
    resolved_voice = (
        voice
        if voice is not None
        else ((parent_run.config_snapshot or {}).get("voice") or {}).get("voice_id")
    )

    # EXACTLY ONE synthesize call. Stored under the PARENT run id so the clip lands
    # next to its siblings and the child run's compile can fetch it; the clip id is
    # content-addressed, so it can never collide with an existing clip.
    # TTSClient lazily opens a pooled httpx.AsyncClient; a bare ``TTSClient()``
    # inline would leak that pool on every single-line re-narrate. It exposes no
    # async-context-manager (BaseServiceClient teardown is ``close()``, not
    # ``aclose`` / ``__aenter__``), so close it in a finally.
    # The synth runs in REQUEST context, so act as the PARENT run: its pinned
    # ``tts`` slot (a tenant BYO voice must not get one platform clip spliced in)
    # and its cost correlation (usage metered to the run that owns the clip).
    tts_client = TTSClient()
    try:
        with bound_existing_run_scope(
            parent_run.config_snapshot or {}, run_id=str(parent_run.id), stage=narrate_stage
        ):
            clip_id, artifact_url, seconds = await synthesize_and_store(
                tts_client,
                text=text,
                voice=resolved_voice,
                run_id=str(parent_run.id),
                stage_name=narrate_stage,
            )
    finally:
        # SHIELDED for the same reason as sub_dag's pubsub release: this runs on the
        # way out of a cancellation, and a second cancel landing on a bare `await`
        # here abandons the pooled httpx.AsyncClient the comment above exists to
        # close. Shield is what makes the close finish; `except CancelledError`
        # cannot, because the close would already have been interrupted.
        await asyncio.shield(tts_client.close())

    patched_narrate = _replace_clip_in_narrate_output(
        dict(narrate_so),
        id_field=cls.source_id_field,
        line_id=line_id,
        clip={
            "id": clip_id,
            cls.source_id_field: line_id,
            "artifact_url": artifact_url,
            "seconds": seconds,
        },
    )
    if patched_narrate is None:
        raise RenarrateError(f"stage '{narrate_stage}' has no clip for line '{line_id}'")

    preserved[read_stage] = _replace_line_text(
        source_output, list_key=cls.source_list_key, line_id=line_id, text=text
    )
    preserved[narrate_stage] = patched_narrate
    return preserved


async def apply_open_annotations_in_transaction(
    *,
    session: AsyncSession,
    affected_stage_output_ids: list[UUID],
    new_run_id: UUID,
    element_id: str | None = None,
) -> int:
    """Atomic-flip open → applied with applied_in_run_id=new_run_id.

    When ``element_id`` is given (a surgical single-clip regenerate), only that
    clip's feedback flips; sibling clips' still-open annotations on the same
    stage_output are left untouched. When ``None`` (reorder + whole-run
    regenerate), every open annotation on the affected stage_outputs flips.

    Caller MUST commit (or roll back) afterwards. The flip and the new run row
    must commit together to honour the §8.6 'double-application impossible'
    invariant.
    """
    if not affected_stage_output_ids:
        return 0
    return await AnnotationRepository(session).bulk_apply(
        stage_output_ids=affected_stage_output_ids,
        applied_in_run_id=new_run_id,
        element_id=element_id,
    )
