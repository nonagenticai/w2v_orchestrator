"""ActionReelBlueprint — raw footage in, beat-cut reel out, reviewable at every seam.

The product: the operator uploads raw camera/drone/watersport footage, an LLM
picks the segments worth keeping, and those are cut to a music beat with
per-clip time-warp, under a royalty-free bed and animated graphics.

DAG (two branches that join at ``cut_plan``)::

    source_ingest ─┬─ footage_probe → shot_analysis → segment_select ─┐
                   └─ music_select ───────────────────────────────────┤
                                                                      ▼
                                                                  cut_plan
                                                                   ┌──┴──┐
                                                    graphics_design     narrate
                                                                   └──┬──┘
                                                                      ▼
                                                       compile → render_manifest

The footage branch and the music branch are genuinely independent — a bed is
chosen from the brief's mood, not from the clips — so they are declared as
separate chains off ``source_ingest`` and the analyzer round-trip for the music
grid overlaps the (much longer) per-source video analysis instead of queueing
behind it. ``cut_plan`` is the join, and it is the only stage that needs both.

**Why FIVE pauses.** Every stage that makes an EDITORIAL decision pauses, and
none of the mechanical ones do. That is not caution, it is cost asymmetry: the
render is the expensive, slow, non-idempotent step at the bottom, and every one
of these decisions is free to change above it and a full re-render to change
below it. The reviewable seams are: what the footage is worth (shot_analysis),
which seconds survive (segment_select), what it is cut to (music_select), the
shot list itself (cut_plan), and the overlay layer (graphics_design).

Two stages are reused rather than re-implemented: ``source_ingest`` (the
universal multi-source grounding leaf) VERBATIM, and the terminal provenance
ledger via ``ActionReelRenderManifestStage`` — the shared handler with its
artifact source pointed at this blueprint's ``compile`` stage and nothing else
changed. ``cut_plan`` carries the real beat-fit solver, and ``graphics_design`` now
designs and renders a real transparent overlay rather than emitting an empty card
list — its ``cards_render_failed`` flag is what a consumer reads to tell "the
designer chose no cards" (False) apart from "a card was designed and its render
failed" (True), since both ship an empty card list plus a note.
"""

from __future__ import annotations

from src.services.blueprints._factories import FOOTAGE_PROBE_FACTORY, SOURCE_INGEST_FACTORY
from src.services.blueprints.base import DAGBlueprint, StageSpec, handler
from src.services.blueprints.registry import get_registry
from src.services.stages.action_reel.compile import ActionReelCompileStage
from src.services.stages.action_reel.cut_plan import CutPlanStage
from src.services.stages.action_reel.graphics_design import (
    GRAPHICS_DESIGN_TIMEOUT_SECONDS,
    GraphicsDesignStage,
)
from src.services.stages.action_reel.manifest import ActionReelRenderManifestStage
from src.services.stages.action_reel.music_select import MusicSelectStage
from src.services.stages.action_reel.narrate import (
    NARRATE_TIMEOUT_SECONDS,
    ActionReelNarrateStage,
)
from src.services.stages.action_reel.segment_select import SegmentSelectStage
from src.services.stages.action_reel.shot_analysis import (
    SHOT_ANALYSIS_TIMEOUT_SECONDS,
    ShotAnalysisStage,
)
from src.services.stages.primitives.compile_stage import COMPILE_STAGE_OUTER_TIMEOUT_SECONDS

ACTION_REEL_BLUEPRINT = DAGBlueprint(
    id="action_reel_v1",
    project_type="action_reel",
    # `compile` is the deliverable the run finalizer probes for an artifact_url
    # before it may report `completed` — a reel that "succeeded" pointing at a
    # non-existent object is the exact silent-degradation lie that gate exists
    # to catch. There is no post-compile accept-best loop in this blueprint, so
    # the terminal artifact and the compile cut are the same stage.
    terminal_artifact_stage="compile",
    # The re-stitch anchor for surgical timeline edits (reorder / edit-clips
    # re-run this stage onward — see runs/elements.compile_cut_stage). Pinned to
    # `cut_plan`, NOT to `compile`: a reorder must re-run the PLAN and the render,
    # because the shot list IS the timeline for this product. Anchoring on
    # `compile` would re-render the stale order; anchoring any higher (on
    # segment_select) would re-draw a completely different edit and throw away the
    # selection a human already approved at that pause.
    compile_cut_stage="cut_plan",
    stages=[
        StageSpec(
            # Reused VERBATIM from the generic stage set. OPT-IN (skips clean
            # without brief.sources) and a NO-DEP LEAF, so the scrape/parse
            # fan-out overlaps everything below it. Both branches hang off it:
            # the footage branch needs nothing from it but must not start before
            # the run's grounding exists, and music_select reads the brief mood it
            # can enrich. A skipped source_ingest never blocks either branch —
            # `skipped` is terminal-but-fine and satisfies a depends_on.
            name="source_ingest",
            handler_factory=SOURCE_INGEST_FACTORY,
            default_timeout_seconds=900,
        ),
        StageSpec(
            # Depends on source_ingest only for ORDERING (it reads the DB, not the
            # ingest output): it is the head of the footage branch, and hanging it
            # here rather than making it a second leaf keeps the two branches
            # rooted in one place so a future grounding-aware probe has the edge
            # already. No pause: it reports facts about files the operator just
            # uploaded — there is no decision here to review.
            name="footage_probe",
            depends_on=("source_ingest",),
            handler_factory=FOOTAGE_PROBE_FACTORY,
            # Pure DB read + presigns. Explicitly short so a hung storage backend
            # surfaces in seconds instead of inheriting the 700s LLM ceiling on a
            # stage that does no LLM work.
            default_timeout_seconds=120,
        ),
        StageSpec(
            # One video_analyzer job per source, fanned out (stage_gather,
            # concurrency 4). Depends on footage_probe for the presigned URLs it
            # feeds the analyzer.
            #
            # PAUSES: the first stage that says anything about the operator's
            # actual footage. Every decision below it is made from these numbers,
            # so "the analyzer never saw clip 3" costs minutes here and a wasted
            # render if it is caught after the compile.
            name="shot_analysis",
            depends_on=("footage_probe",),
            handler_factory=handler(ShotAnalysisStage),
            # Imported, never a literal: the budget is derived in the handler
            # module from its OWN fan-out constants, so the two cannot drift.
            #
            # It is a WAVE budget: ceil(sources / concurrency 4) * 600s + 100s.
            # This used to be a hardcoded 1400s justified by PER-SOURCE arithmetic
            # ("(1 + 1 retries) * 600s = 1200s + 100s headroom"). That was wrong on
            # both terms — it assumed a SINGLE wave (the outer ceiling is enforced
            # per STAGE, not per attempt, so a 10-clip shoot needs 3 waves = 1800s
            # against a 1400s budget), and the retry it doubled by can never fire
            # (_analyze_one swallows Exception into an error-carrying result, so
            # stage_gather's retry_on never sees one). Do not re-derive 1400.
            default_timeout_seconds=SHOT_ANALYSIS_TIMEOUT_SECONDS,
            default_pause_after=True,
        ),
        StageSpec(
            # The editorial LLM call: which seconds of which clip survive, with a
            # score and a rationale. Depends on shot_analysis because it selects
            # against MEASURED signal (scene cuts + motion curve), not against
            # filenames.
            #
            # PAUSES: this is the first output a human can read as an EDIT. A
            # wrong selection reviewed here is free; the same selection reviewed
            # after the compile is a full re-render.
            name="segment_select",
            depends_on=("shot_analysis",),
            handler_factory=handler(SegmentSelectStage),
            # None => inherit the env-configurable base LLM ceiling, which the DAG
            # executor resolves against the same knob the inner WM budget comes
            # from. Pinning a literal here would make that GitOps knob inert.
            default_pause_after=True,
        ),
        StageSpec(
            # The music branch. Depends on source_ingest ONLY — a bed is chosen
            # from the brief's mood, not from the footage — so the LLM pick and the
            # analyzer's beat-grid round-trip run CONCURRENTLY with the long
            # per-source video analysis instead of behind it. Making it depend on
            # the footage chain would serialise two independent minutes for no
            # editorial gain.
            #
            # PAUSES: music is the most subjective choice in the product and the
            # cheapest to change. Approving the bed before the cut is planned costs
            # a click.
            name="music_select",
            depends_on=("source_ingest",),
            handler_factory=handler(MusicSelectStage),
            default_pause_after=True,
        ),
        StageSpec(
            # THE JOIN: the selected segments meet the beat grid and become an
            # ordered shot list. Both edges are required and neither is optional —
            # this is the only stage that needs footage decisions AND music
            # decisions, which is why the two branches exist at all.
            #
            # PAUSES: the shot list is the reel. Reordering or dropping a shot here
            # is free; doing it after the compile is a re-render. It is also the
            # `compile_cut_stage`, so a surgical timeline edit re-enters the DAG
            # exactly here.
            #
            # The beat-fit solver lives in stages/action_reel/cut_plan.py: each
            # shot gets a contiguous window of BEATS and a per-clip speed, and
            # every cut time is a difference of two absolute beat times (never a
            # running sum) so the cuts cannot drift off the music.
            name="cut_plan",
            depends_on=("segment_select", "music_select"),
            handler_factory=handler(CutPlanStage),
            # A bounded solve over two prior outputs: a walk of the beat list with
            # a fixed candidate set per shot. Short and explicit — no LLM, no
            # downstream call.
            default_timeout_seconds=300,
            default_pause_after=True,
        ),
        StageSpec(
            # The overlay layer (title, per-shot callouts, end card). Depends on
            # cut_plan because a card is timed against a SHOT, so it cannot be
            # designed before the shot list exists.
            #
            # PAUSES: graphics are the most visible layer and the cheapest to
            # reject before a render. The beat is wired NOW so the review exists
            # the day the designer ships, rather than being retrofitted into a
            # blueprint people already depend on.
            #
            # Designs a title treatment on the opening shot and renders it as a
            # transparent overlay. Never a StageSkippedError: this stage is not
            # opt-in, so a render that fails degrades to an empty card list plus
            # a note saying the reel ships with no graphics — and sets
            # `cards_render_failed=True`, which is what distinguishes "could not
            # render them" from a legitimate "chose no cards" (flag False).
            #
            # This ceiling is HALF a ladder: the stage's own recorder poll
            # (`graphics_design._RECORD_TIMEOUT_S`) must stay strictly BELOW it,
            # or the outer TimeoutMiddleware (`asyncio.wait_for`) cancels with a
            # BaseException the stage's `except Exception` cannot catch — and the
            # degrade promised two paragraphs up never runs. Lowering this number
            # without lowering the inner one silently re-breaks that promise; a
            # test pins the pair.
            name="graphics_design",
            depends_on=("cut_plan",),
            handler_factory=handler(GraphicsDesignStage),
            # 300s while this stage was title-only and deterministic. It now holds a
            # WM client at the shared LLM client rung (the per-shot callout
            # designer), so the ceiling has to clear `llm_budget_seconds x attempts`
            # BEFORE any of the recorder polls below it — otherwise the outer
            # wrapper cancels the stage with a BaseException and every degrade path
            # documented above becomes unreachable. The arithmetic (LLM 1930s +
            # 4 sequential 240s renders + 100s headroom) lives on the constant, so
            # this file cannot drift from the module that spends the time.
            default_timeout_seconds=GRAPHICS_DESIGN_TIMEOUT_SECONDS,
            default_pause_after=True,
        ),
        StageSpec(
            # OPT-IN, skip-clean voice-over. Most reels have none — the footage and
            # the bed are the edit — so with no `brief.narration.lines` this raises
            # StageSkippedError and `compile`'s depends_on never blocks a run that
            # did not ask for narration. Depends on cut_plan so a voice-over is
            # written against an APPROVED cut; narrating an unapproved one is work
            # thrown away the moment the reviewer reorders a shot.
            #
            # NO pause: the lines are AUTHORED by the operator (they ride in on the
            # brief), so there is nothing generated to review — the same reasoning
            # showcase applies to its authored demo_video_gen.
            name="narrate",
            depends_on=("cut_plan",),
            handler_factory=handler(ActionReelNarrateStage),
            # Per-line TTS fan-out, budgeted by WAVES rather than by one line.
            # `stage_gather(concurrency=4, per_item_timeout_seconds=600,
            # per_item_retries=1)` ⇒ worst case ceil(N / 4) * 1200s, so the old
            # literal 1300 ("(1 + 1 retries) * 600s = 1200s") only ever covered
            # N <= 4 and this PAID stage could be cancelled mid-fan-out at N = 5.
            # The number is now computed in the stage module from the fan-out's own
            # constants and the line cap it enforces (12 lines ⇒ 3 waves ⇒ 3600s
            # + 100s headroom = 3700s), so the ceiling cannot drift from the
            # arithmetic it is meant to cover.
            default_timeout_seconds=NARRATE_TIMEOUT_SECONDS,
        ),
        StageSpec(
            # The canonical compile path (CompileStageBase -> clients/video_compile),
            # so this stage owns only the shot join. Depends on BOTH producers of
            # what goes on screen: graphics_design (the overlay layer — the edge is
            # declared today even though the card list is empty, so shipping the
            # designer touches one stage and not the topology) and narrate (which
            # skips clean, so the edge never blocks a music-only reel).
            #
            # Note it does NOT list cut_plan: graphics_design already depends on it,
            # so the ordering is implied and a redundant edge would just be a second
            # place to keep in sync.
            #
            # NO pause: the render is the terminal artifact; it is reviewed by
            # watching it.
            name="compile",
            depends_on=("graphics_design", "narrate"),
            handler_factory=handler(ActionReelCompileStage),
            # Outer must cover the compile-wait backstop — matches every other
            # compile stage in the fleet (see compile_stage.py).
            default_timeout_seconds=COMPILE_STAGE_OUTER_TIMEOUT_SECONDS,
        ),
        StageSpec(
            # The terminal provenance/override ledger; a pure audit artifact that
            # never mutates anything, so the delivered reel is byte-identical
            # whether or not it runs. The shared handler is reused whole — the
            # subclass only points its artifact source at THIS blueprint's render
            # stage (`compile`), since the base chain names demo_video stages that
            # do not exist here and would leave `artifact_url`/`duration_seconds`
            # empty on every reel.
            name="render_manifest",
            depends_on=("compile",),
            handler_factory=handler(ActionReelRenderManifestStage),
            default_timeout_seconds=120,
        ),
    ],
)


get_registry().register(ACTION_REEL_BLUEPRINT)
