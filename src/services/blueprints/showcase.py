"""ShowcaseBlueprint — a native, reusable "product showcase" video.

The richest demo video: it interleaves real form-recording footage (drag / type /
click captured by the frontend_test_runner) with branded motion cards (an intro, a
divider before each beat, a value-prop, an outro) and narrates the whole thing as a
two-voice dialogue. Reuses the demo_video stage classes — the gen / narrate /
record / compile stages all honour the authored brief fields (``brief.lines`` +
``brief.package_url`` + ``brief.dialogue.enabled``) — and adds one new stage,
``showcase_slides``, which renders the authored ``card`` beats. Everything joins on
the authored line id, so ``demo_video_compile`` muxes cards, footage, and dialogue
into one timeline.

DAG: gen → (narrate ∥ record ∥ showcase_slides) → compile → growth_bundle.

``demo_video_gen`` is authored (it consumes ``brief.lines`` and skips bdd
generation), so — unlike the demo_video blueprint — there is nothing generated to
review and the stage does not pause. The stage *names* are shared with demo_video
on purpose: narrate / record / compile look their inputs up by those names.
"""

from __future__ import annotations

from src.services.blueprints._factories import SOURCE_INGEST_FACTORY
from src.services.blueprints.base import DAGBlueprint, StageSpec, handler
from src.services.blueprints.registry import get_registry
from src.services.stages.demo_video.claim_finish import ClaimFinishStage
from src.services.stages.demo_video.compile import DemoVideoCompileStage
from src.services.stages.demo_video.encode_master import EncodeMasterStage
from src.services.stages.demo_video.final_critique import FinalCritiqueStage
from src.services.stages.demo_video.footage_critique import DemoVideoFootageCritiqueStage
from src.services.stages.demo_video.footage_direction import DemoVideoFootageDirectionStage
from src.services.stages.demo_video.gen import DemoVideoGenStage
from src.services.stages.demo_video.manifest import RenderManifestStage
from src.services.stages.demo_video.narrate import DemoVideoNarrateStage
from src.services.stages.demo_video.reconcile_timeline import ReconcileTimelineStage
from src.services.stages.demo_video.record import DemoVideoRecordStage
from src.services.stages.demo_video.retention_plan import RetentionPlanStage
from src.services.stages.demo_video.slides import ShowcaseSlidesStage
from src.services.stages.demo_video.sound_director import DemoVideoSoundDirectorStage
from src.services.stages.demo_video.voice_director import VoiceDirectorStage
from src.services.stages.growth_bundle.handler import GrowthBundleStage
from src.services.stages.primitives.compile_stage import COMPILE_STAGE_OUTER_TIMEOUT_SECONDS
from src.services.stages.shared.script_doctor import ScriptDoctorStage

SHOWCASE_BLUEPRINT = DAGBlueprint(
    id="showcase_v1",
    project_type="showcase",
    # The compile stage is the terminal artifact: it is the "compile cut" a timeline
    # reorder must re-run (`compile_cut_stage` → `terminal_artifact_stage`, see
    # runs/regenerate.py) and the stage the finalizer probes for existence. The L4
    # footage-critique loop that runs AFTER it may re-render a better video, but that
    # accepted-best artifact reaches the deliverable/provenance via `render_manifest`
    # (which depends on the critique stage), NOT by moving the cut — making critique
    # the cut would wrongly preserve the stale stitch on a reorder.
    terminal_artifact_stage="demo_video_compile",
    # …and BECAUSE the cut cannot move, the compile is not the delivered file.
    # Same chain as demo_video_v1: compile → `demo_video_footage_critique` →
    # `demo_video_encode_master` → `final_critique`, each a skip-clean
    # pass-through that may republish a re-mastered `artifact_url`.
    # `final_critique` is the last of them that declares the field, so it is what
    # ships. The compile stays the re-stitch anchor and the finalizer probe.
    delivered_artifact_stage="final_critique",
    stages=[
        StageSpec(
            name="demo_video_gen",
            handler_factory=handler(DemoVideoGenStage),
            # Authored mode resolves instantly (no bdd generation); the larger
            # envelope only matters if a showcase is run without authored lines.
            default_timeout_seconds=1200,
            # Authored input — nothing generated to review, so run straight through.
            default_pause_after=False,
        ),
        StageSpec(
            # OPT-IN, skip-clean line-doctor (ORCH-1.2): re-emits the demo_video_gen
            # shape with only weak+unlocked line TEXT rewritten (ids/order/bdd_job_id/
            # package_url/features_generated preserved), else raises StageSkippedError
            # so a run with no enabled `script_doctor` block falls straight through.
            # Downstream text-consumers read the spine via resolve_plan_output (spine.py),
            # which prefers this stage's output and falls back to demo_video_gen — so a
            # skipped doctor is byte-identical to the pre-ORCH-1.2 topology.
            name="script_doctor",
            depends_on=("demo_video_gen",),
            handler_factory=handler(ScriptDoctorStage),
            # 2026-08-19: 900 -> 5800. The old 900 was a hand-copy of "a bit above
            # the 660s budget" and was inverted the moment the budget went to 950.
            # This stage does NOT have the base-ceiling shape, so inheriting would
            # be wrong too: run_evaluator_optimizer issues up to THREE sequential
            # stage_calls (critic pass 1, optimizer, critic pass 2 — max_passes is
            # hard-clamped to 2 in loop_engine.py), each of llm_retries + 1
            # attempts at llm_budget_seconds. Worst case 3 x 2 x 950 = 5700s; +100s
            # headroom (the sibling convention in test_stage_timeout_invariants)
            # = 5800. The product rule the config validator applies to the BASE
            # ceiling therefore applies here too, multiplied by the loop's call
            # count — the outer wrapper must never fire first, because a
            # TimeoutMiddleware cut raises CancelledError (a BaseException) which
            # this stage's soft-degrade `except Exception` cannot catch: it would
            # hard-fail a paid run after the human gen-gate instead of shipping the
            # un-doctored draft. A clean inner StageTimeoutError degrades correctly.
            default_timeout_seconds=5800,
            # The doctored script is worth a review beat before the render fan-out.
            default_pause_after=True,
        ),
        # source_ingest is OPT-IN (skips → completed-empty without brief.sources).
        # No-dep leaf so the scrape/parse fan-out overlaps demo_video_gen; it is
        # the grounding corpus that claim_finish (and the growth_bundle claim lint)
        # read via source_ingest.grounding_sentences(ctx).
        StageSpec(
            name="source_ingest",
            handler_factory=SOURCE_INGEST_FACTORY,
            default_timeout_seconds=900,
        ),
        StageSpec(
            # OPT-IN, skip-clean claim gate (ORCH-4.4 / A18): checks each authored
            # line's factual claims against the source_ingest grounding corpus and
            # re-emits the spine shape with claim_status/claim_source stamped (and
            # unsupported, unpinned lines softened), else raises StageSkippedError so
            # a run with no grounding corpus / no enabled `claim_finish` block falls
            # straight through. Depends on source_ingest as well as the doctored spine
            # — without that edge the corpus may not have landed yet and the gate would
            # read `{}`; the dep never blocks because source_ingest skips clean.
            name="claim_finish",
            depends_on=("script_doctor", "source_ingest"),
            handler_factory=handler(ClaimFinishStage),
            # WM-loop stage like retention_plan: outer must cover the LLM ladder inner
            # budget (660s) — see PipelineSettings.llm_budget_seconds.
        ),
        StageSpec(
            # OPT-IN, skip-clean retention pass (ORCH-1.3): re-emits the spine shape
            # with hook_role/pin_position per-line hints (and any weak-line rewrites),
            # else raises StageSkippedError so a run with no enabled `retention_plan`
            # block falls straight through. Downstream text-consumers read the spine
            # via resolve_plan_output (spine.py), which now prefers this stage's output
            # and falls back to claim_finish / script_doctor / demo_video_gen — so a
            # skipped pass is byte-identical to the pre-ORCH-1.3 topology.
            name="retention_plan",
            depends_on=("claim_finish",),
            handler_factory=handler(RetentionPlanStage),
        ),
        StageSpec(
            # OPT-IN, skip-clean timeline reconciliation (ORCH-1.4): the LAST plan
            # mutator. Needs BOTH the doctored/retained plan spine AND the real
            # recorded footage lengths, so it joins retention_plan with
            # demo_video_record (which hangs off demo_video_gen, a parallel sibling of
            # the plan chain — no cycle). It stamps a per-line `vo_rate` that narrate
            # consumes, which is why the whole render fan-out sits BELOW it. Skip-clean:
            # no `direction.timing` block ⇒ StageSkippedError ⇒ resolve_plan_output
            # falls through to retention_plan ⇒ byte-identical to pre-ORCH-1.4.
            name="reconcile_timeline",
            depends_on=("retention_plan", "demo_video_record"),
            handler_factory=handler(ReconcileTimelineStage),
            # WM-loop stage like retention_plan: outer must cover the LLM ladder inner
            # budget (660s) — see PipelineSettings.llm_budget_seconds.
        ),
        StageSpec(
            # OPT-IN, skip-clean voice-performance director (ORCH-1.4): reads the
            # reconciled spine and emits a PerformanceDirective per line (provider
            # tier / delivery note / emphasis / pause / rate_hint / ssml). It does
            # NOT rewrite line text or order — so it is NOT in PLAN_STAGE_PRECEDENCE
            # (spine.py); narrate reads it via ctx.prior_output("voice_director") to
            # resolve the per-line delivery. Sits between reconcile_timeline (its
            # input spine) and narrate (its consumer). Skips clean ⇒ narrate falls
            # back to the vo_rate path, byte-identical to the pre-ORCH-1.4 topology.
            name="voice_director",
            depends_on=("reconcile_timeline",),
            handler_factory=handler(VoiceDirectorStage),
            # WM-LLM stage like the sibling plan passes: outer must cover the LLM
            # ladder inner budget (660s) — see PipelineSettings.llm_budget_seconds.
        ),
        StageSpec(
            name="demo_video_narrate",
            depends_on=("voice_director",),
            handler_factory=handler(DemoVideoNarrateStage),
            default_timeout_seconds=1300,
        ),
        StageSpec(
            name="demo_video_record",
            depends_on=("demo_video_gen",),
            handler_factory=handler(DemoVideoRecordStage),
            default_timeout_seconds=2200,
        ),
        StageSpec(
            name="showcase_slides",
            # Also depends on narrate so each card's MOTION clip can be rendered for
            # its full narration length (animated the whole time on screen, no
            # last-frame freeze). Off the critical path — record (FTR, ~15min) is
            # the long pole, so the serialization behind narrate is hidden.
            depends_on=("reconcile_timeline", "demo_video_narrate"),
            handler_factory=handler(ShowcaseSlidesStage),
            # Deck render (≤600s) + opt-in branded motion phase (≤1100s) + overhead.
            default_timeout_seconds=1900,
        ),
        StageSpec(
            # OPT-IN, skip-clean: picks per-line accent SFX when brief.sound_effects
            # is enabled, else raises StageSkippedError so demo_video_compile's
            # depends_on never blocks a run that didn't opt in. Reads narration TEXT
            # from gen + real per-line DURATIONS from narrate for accent placement.
            name="demo_video_sound_director",
            depends_on=("reconcile_timeline", "demo_video_narrate"),
            handler_factory=handler(DemoVideoSoundDirectorStage),
            # Outer must cover the LLM ladder inner budget (660s) — see
            # PipelineSettings.llm_budget_seconds.
        ),
        StageSpec(
            # OPT-IN, skip-clean footage shot-planner (ORCH-3.3): picks a per-line
            # framing plan (zoom on/off, callout, reframe) when direction.footage is
            # requested, else raises StageSkippedError so demo_video_compile's
            # depends_on never blocks a run that didn't opt in. Reads the spine TEXT +
            # narrate DURATIONS + record FOCUS TRACKS (the has-click target signal) so a
            # line with no captured click never zooms/callouts at nothing. The record dep
            # is REQUIRED — without it has_click is always false and the feature dies.
            name="demo_video_footage_direction",
            depends_on=("reconcile_timeline", "demo_video_narrate", "demo_video_record"),
            handler_factory=handler(DemoVideoFootageDirectionStage),
            # Outer must cover the LLM ladder inner budget (660s) — see
            # PipelineSettings.llm_budget_seconds.
        ),
        StageSpec(
            name="demo_video_compile",
            depends_on=(
                "demo_video_narrate",
                "demo_video_record",
                "showcase_slides",
                "demo_video_sound_director",
                "demo_video_footage_direction",
            ),
            handler_factory=handler(DemoVideoCompileStage),
            # Outer timeout must cover the compile-wait backstop below it — see
            # compile_stage.COMPILE_STAGE_OUTER_TIMEOUT_SECONDS.
            default_timeout_seconds=COMPILE_STAGE_OUTER_TIMEOUT_SECONDS,
        ),
        StageSpec(
            # Post-compile SOFT critique loop (L4): scores the rendered video
            # shot-by-shot (video_analyzer measured numbers → the `footage-critic` WM
            # evaluator), nudges the footage-direction plan for weak shots and
            # re-renders via the canonical compile path (≤2 passes, accept-best). It
            # NEVER hard-blocks the deliverable. OPT-IN on `direction.footage.critique`;
            # absent ⇒ StageSkippedError ⇒ skipped with empty output so downstream
            # reads demo_video_compile's artifact unchanged (byte-identical).
            name="demo_video_footage_critique",
            depends_on=("demo_video_compile",),
            handler_factory=handler(DemoVideoFootageCritiqueStage),
            # 2500 clears the single-stage_call invariant (950 x 2 + 30 = 1930); it does
            # NOT cover the 3-call evaluator-optimizer loop (5800s) — a PRE-EXISTING,
            # fleet-wide shortfall tracked in _LOOP_STAGE_CALLS_BACKLOG in
            # tests/services/blueprints/test_llm_timeout_ladder.py.
            default_timeout_seconds=2500,
        ),
        StageSpec(
            # S19 encode/mux mastering evaluator-optimizer (SOFT, spec 13 §5): scores
            # the compiler's deterministic MasteringReport (encode-critic WM worker),
            # proposes a CLAMPED mastering profile (mux-mastering-advisor WM worker) and
            # — strictly GUARDED (only when footage_critique did NOT re-render and a gate
            # failed) — re-compiles ONCE via the canonical compile path (≤2 passes,
            # accept-best). NEVER hard-blocks. DISABLED-BY-DEFAULT: absent `mastering.critic`
            # ⇒ zero WM calls and the UNCHANGED artifact passes through, so downstream reads
            # compile's/footage-critique's artifact unchanged (byte-identical).
            name="demo_video_encode_master",
            depends_on=("demo_video_footage_critique",),
            handler_factory=handler(EncodeMasterStage),
            # 2500 clears the single-stage_call invariant (950 x 2 + 30 = 1930); it does
            # NOT cover the 3-call evaluator-optimizer loop (5800s) — a PRE-EXISTING,
            # fleet-wide shortfall tracked in _LOOP_STAGE_CALLS_BACKLOG in
            # tests/services/blueprints/test_llm_timeout_ladder.py.
            default_timeout_seconds=2500,
        ),
        StageSpec(
            # A11 advisory ship gate (SOFT): reads whole-film gestalt off the
            # video_analyzer output + the compiler's final-mux loudness, scores it
            # (optionally via the `film-critic` WM worker) and hands a quality badge +
            # chapters downstream. NEVER hard-blocks; ships the accepted-best artifact.
            # Under `mode="auto"` it runs a BOUNDED (≤2-pass) in-stage re-drive that
            # re-runs an owning director and RE-COMPILES, so `rerun_stages` reports the
            # stages actually executed. OPT-IN on `ship_gate`; absent / `mode=off`
            # ⇒ StageSkippedError ⇒ skipped with `{}` output so growth_bundle /
            # render_manifest read the upstream artifact unchanged (byte-identical).
            # ⚠️ Today only LOUDNESS is genuinely measurable: the analyzer emits no
            # per-frame visual features, so continuity / cohesion / tonal drift come back
            # `null` + `insufficient_evidence` on real renders. That is deliberate honesty,
            # not a regression — see final_critique's module docstring.
            name="final_critique",
            depends_on=("demo_video_encode_master",),
            handler_factory=handler(FinalCritiqueStage),
            # MUST be explicit — see the demo_video blueprint for the full rationale:
            # inheriting the base ceiling lets the outer wrapper trip on the stage's own
            # LLM ladder plus an `auto` re-drive's full recompile, which dag_executor turns
            # into a re-dispatch and ultimately a `failed` run — a soft gate hard-blocking
            # the deliverable. Matches the compile siblings and clears the
            # single-stage_call invariant (950 x 2 + 30 = 1930); it does NOT cover this
            # stage's 4-call ladder (7700s) — a PRE-EXISTING, fleet-wide shortfall tracked
            # in _LOOP_STAGE_CALLS_BACKLOG in
            # tests/services/blueprints/test_llm_timeout_ladder.py.
            default_timeout_seconds=2500,
        ),
        StageSpec(
            name="growth_bundle",
            depends_on=("final_critique",),
            handler_factory=handler(GrowthBundleStage),
        ),
        StageSpec(
            # Reads the delivered artifact from the A11 ship-gate output (pass-through
            # best) when it ran, falling back through footage_critique to
            # demo_video_compile when the earlier stages skipped-clean — so it depends
            # on final_critique to observe its output.
            name="render_manifest",
            depends_on=("final_critique",),
            handler_factory=handler(RenderManifestStage),
            default_timeout_seconds=120,
        ),
    ],
)


get_registry().register(SHOWCASE_BLUEPRINT)
