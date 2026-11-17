"""DemoVideoBlueprint — gen → (narrate ∥ record) → compile → growth_bundle.

Turns a website/app URL into a narrated, real-interaction (type/drag/click) demo
video. ``demo_video_gen`` drives bdd_playwright_service to generate a Playwright-
BDD walkthrough suite + per-scenario narration script, then pauses so a human can
review the scenarios + narration before the costly TTS + record + compile steps.
``script_doctor`` then sits directly below gen (opt-in, skip-clean) and rewrites
only weak+unlocked line TEXT, so every downstream text consumer reads the doctored
spine. ``demo_video_record`` (frontend_test_runner runs the suite with video recording)
fans out off the gen output; ``reconcile_timeline`` then joins the plan spine with
the REAL recorded footage lengths and stamps the per-line ``vo_rate`` that
``demo_video_narrate`` (TTS) speaks at. Both join on the scenario ``clip-<n>`` id,
then ``demo_video_compile`` muxes them via the canonical compile path.
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
from src.services.stages.demo_video.voice_director import VoiceDirectorStage
from src.services.stages.growth_bundle.handler import GrowthBundleStage
from src.services.stages.primitives.compile_stage import COMPILE_STAGE_OUTER_TIMEOUT_SECONDS
from src.services.stages.shared.script_doctor import ScriptDoctorStage

DEMO_VIDEO_BLUEPRINT = DAGBlueprint(
    id="demo_video_v1",
    project_type="demo_video",
    # The compile stage is the terminal artifact: it is the "compile cut" that a
    # timeline reorder must re-run (`compile_cut_stage` → `terminal_artifact_stage`,
    # see runs/regenerate.py) and the stage the finalizer probes for existence. The
    # L4 footage-critique loop that runs AFTER it may re-render a better video, but
    # that accepted-best artifact is carried to the deliverable/provenance by
    # `render_manifest` (which depends on the critique stage), NOT by moving the cut —
    # making critique the cut would wrongly preserve the stale stitch on a reorder.
    terminal_artifact_stage="demo_video_compile",
    # …and BECAUSE the cut cannot move, the compile is not the delivered file.
    # Three pass-through stages follow it — `demo_video_footage_critique`,
    # `demo_video_encode_master`, `final_critique` — and each one may republish a
    # re-mastered `artifact_url` (or skip clean and pass the previous through).
    # `final_critique` is the LAST stage in that chain that declares
    # `artifact_url`, so it is what actually ships. Named here so nothing has to
    # infer the deliverable from output shape; the compile stays the re-stitch
    # anchor and the finalizer probe, untouched.
    delivered_artifact_stage="final_critique",
    stages=[
        StageSpec(
            name="demo_video_gen",
            handler_factory=handler(DemoVideoGenStage),
            # Outer envelope above the inner per-poll cap (600s) + start/result
            # overhead. bdd generation = inspect + features + step defs + narration.
            default_timeout_seconds=1200,
            # Pause so a human can review the generated scenarios + narration
            # before TTS + record + compile spend.
            default_pause_after=True,
        ),
        StageSpec(
            # OPT-IN, skip-clean line-doctor (ORCH-1.2): re-emits the demo_video_gen
            # shape with only weak+unlocked line TEXT rewritten (ids/order/bdd_job_id/
            # package_url/features_generated preserved), else raises StageSkippedError
            # so a run with no enabled `script_doctor` block falls straight through.
            # Downstream text-consumers read the spine via resolve_plan_output (spine.py),
            # which already ranks this stage above demo_video_gen — so a skipped doctor
            # is byte-identical to the pre-registration topology.
            #
            # It was built + tested for BOTH families but only ever registered in
            # showcase_v1, so on demo_video runs it had no StageSpec at all and could
            # never execute (zero rows in stage_outputs). This spec is that fix; it
            # mirrors showcase's slot exactly: hangs off demo_video_gen and sits ABOVE
            # claim_finish, because every text consumer (claim grounding, retention
            # rewrite/reorder, timeline reconcile, voice direction, narrate/TTS,
            # footage direction) must see the DOCTORED line text, not the raw draft.
            # demo_video_record stays wired to demo_video_gen (it reads clip ids /
            # bdd_job_id / expected_text, never line TEXT), so it remains a parallel
            # sibling of the plan chain and the reconcile_timeline join stays acyclic.
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
            # NO second human gate. This family already pauses at demo_video_gen
            # (see above) — showcase_v1 does NOT (its gen-equivalent is
            # `default_pause_after=False`, showcase.py) and pauses at script_doctor
            # instead. So "mirror showcase" here would mean DOUBLE-gating, not
            # matching it. Today the gate would be inert anyway (a SKIPPED stage
            # never trips a pause_after gate — dag_executor.run's `and not skipped`
            # guard — and script_doctor skips clean while `script_doctor.enabled` is
            # unset), but WS-1 is about to default-inject
            # `script_doctor: {"enabled": True}`, which makes this stage LIVE on
            # every demo_video run and the gate would then fire every time.
            # Live evidence that a second gate is actively harmful: 68 of 91
            # demo_video runs are already stranded at the EXISTING demo_video_gen
            # gate, while showcase (single-gate) completes 22 of 25. Stranding more
            # runs would defeat the entire point of activating this stage. The
            # operator still reviews the script — at the gen gate, and in the final
            # output.
            default_pause_after=False,
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
            # OPT-IN, skip-clean claim gate (ORCH-4.4 / A18): checks each line's factual
            # claims against the source_ingest grounding corpus and re-emits the spine
            # shape with claim_status/claim_source stamped (and unsupported, unpinned
            # lines softened), else raises StageSkippedError so a run with no grounding
            # corpus falls straight through. It grounds the DOCTORED spine, so it hangs
            # off script_doctor (which itself hangs off demo_video_gen) rather than
            # demo_video_gen directly — same slot as showcase_v1. It also depends on
            # source_ingest, without which the corpus may not have landed yet and the
            # gate would read `{}`. No dep blocks — all skip clean.
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
            # via resolve_plan_output (spine.py), which prefers this stage's output and
            # falls back to claim_finish / demo_video_gen — so a skipped pass is
            # byte-identical to the pre-ORCH-1.3 topology.
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
            # consumes, which is why narrate sits BELOW it. Skip-clean: no
            # `direction.timing` block ⇒ StageSkippedError ⇒ resolve_plan_output falls
            # through to retention_plan ⇒ byte-identical to the pre-ORCH-1.4 topology.
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
            # Per-line TTS fan-out; one bad line is 2 attempts x 600s = 1200s.
            default_timeout_seconds=1300,
        ),
        StageSpec(
            name="demo_video_record",
            depends_on=("demo_video_gen",),
            handler_factory=handler(DemoVideoRecordStage),
            # Outer envelope above the stage's re-polled wait: up to 3 polls x 600s
            # (1 + _WAIT_RETRIES) plus start_execution + per-clip fetch/store + the
            # stage_call backoff. FTR runs bddgen + npm install + N recorded
            # scenarios; a large suite can need the full re-poll budget.
            default_timeout_seconds=2200,
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
            # WM-LLM stage: outer must cover the LLM ladder inner budget (660s) — see
            # PipelineSettings.llm_budget_seconds.
        ),
        StageSpec(
            name="demo_video_compile",
            depends_on=(
                "demo_video_narrate",
                "demo_video_record",
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
            # Outer must cover a full re-render (compile worst case, 2500s) plus the
            # per-pass analyzer + LLM critic; the loop is hard-bounded to ≤2 passes.
            # 2500 clears the single-stage_call invariant (950 x 2 + 30 = 1930); it does
            # NOT cover the 3-call evaluator-optimizer loop (5800s) — a PRE-EXISTING,
            # fleet-wide shortfall tracked in _LOOP_STAGE_CALLS_BACKLOG in
            # tests/services/blueprints/test_llm_timeout_ladder.py.
            default_timeout_seconds=2500,
        ),
        StageSpec(
            # S19/S20 encode/mux mastering evaluator-optimizer (SOFT, spec 13 §5): scores
            # the compiler's deterministic MasteringReport (encode-critic WM worker),
            # proposes a CLAMPED mastering profile (mux-mastering-advisor WM worker) and —
            # when a gate failed or a rerun was asked — re-compiles via the canonical
            # compile path (≤2 passes, accept-best). It composes with footage_critique:
            # the critique persists its nudged plans (`footage_plans`), which compile
            # prefers, so the re-compile reproduces the improved footage AND the mastering
            # delta. NEVER hard-blocks. DISABLED-BY-DEFAULT: absent `mastering.critic`
            # ⇒ zero WM calls and the UNCHANGED artifact passes through, so downstream reads
            # compile's/footage-critique's artifact unchanged (byte-identical).
            name="demo_video_encode_master",
            depends_on=("demo_video_footage_critique",),
            handler_factory=handler(EncodeMasterStage),
            # Outer must cover a full re-compile (compile worst case, 2500s) plus the
            # per-pass encode-critic + advisor WM calls; the loop is bounded to ≤2 passes.
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
            # MUST be explicit, like the compile/critique siblings. Inheriting the
            # base LLM ceiling HARD-FAILS the soft gate: tripping the outer wrapper makes
            # dag_executor raise StageResumableTimeoutError and re-dispatch from scratch
            # (a new analyzer job, possibly a SECOND full compile), and past
            # `max_stage_attempts` the run persists `failed` — a "soft gate that never
            # hard-blocks" hard-blocking the deliverable. 2500s matches
            # demo_video_compile / _footage_critique / _encode_master, all of which this
            # stage can re-invoke, and clears the single-stage_call invariant
            # (950 x 2 + 30 = 1930).
            # It does NOT cover this stage's 4-call ladder (7700s) — a PRE-EXISTING,
            # fleet-wide shortfall tracked in _LOOP_STAGE_CALLS_BACKLOG in
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


get_registry().register(DEMO_VIDEO_BLUEPRINT)
