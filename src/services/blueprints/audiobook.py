from __future__ import annotations

from src.services.blueprints.base import DAGBlueprint, StageSpec, handler
from src.services.blueprints.registry import get_registry
from src.services.stages.audiobook.cast import AudiobookCastStage
from src.services.stages.audiobook.direct import AudiobookDirectStage
from src.services.stages.audiobook.ingest import AudiobookIngestStage
from src.services.stages.audiobook.narrate import AudiobookNarrateStage
from src.services.stages.audiobook.plan import AudiobookPlanStage
from src.services.stages.audiobook.qa import AudiobookQAStage
from src.services.stages.audiobook.segment import AudiobookSegmentStage
from src.services.stages.audiobook.stitch import AudiobookStitchStage

AUDIOBOOK_BLUEPRINT = DAGBlueprint(
    id="audiobook_v1",
    project_type="audiobook",
    terminal_artifact_stage="audiobook_stitch",
    stages=[
        StageSpec(
            name="audiobook_ingest",
            handler_factory=handler(AudiobookIngestStage),
            # 3600s (not 120s) because the opt-in pronunciation-proposer makes a
            # WM/LLM call: the outer stage timeout must cover the LLM ladder inner
            # budget (660s) + retries, or TimeoutMiddleware would fire OUTSIDE the
            # proposer's soft-degrade guard and fail a paid ingest. Matches the
            # other LLM-making audiobook stages (direct/cast/narrate/qa).
            default_timeout_seconds=3600,
            # Pause so the user can review detected chapters before TTS spend.
            default_pause_after=True,
        ),
        StageSpec(
            name="audiobook_plan",
            depends_on=("audiobook_ingest",),
            handler_factory=handler(AudiobookPlanStage),
            default_timeout_seconds=60,
        ),
        StageSpec(
            name="audiobook_segment",
            depends_on=("audiobook_plan",),
            handler_factory=handler(AudiobookSegmentStage),
            # Sub-chunk speaker attribution for span-granularity voice acting.
            # No-op (empty output, no LLM) unless brief.acting_granularity="span",
            # so off/chunk runs pay nothing and the DAG shape stays stable.
            # Outer must cover the LLM ladder inner budget (660s) — see
            # PipelineSettings.llm_budget_seconds.
        ),
        StageSpec(
            name="audiobook_cast",
            depends_on=("audiobook_segment",),
            handler_factory=handler(AudiobookCastStage),
            # One LLM pass over the manuscript to cast voices per character.
            # No pin: inherits LLM_STAGE_BASE_CEILING_S, which covers the whole
            # stage (llm_retries + 1 attempts x llm_budget_seconds).  See
            # PipelineSettings.llm_budget_seconds.
            # Pause so the user reviews the casting + chunk plan before TTS
            # spend (the higher-value gate; replaces the old plan pause).
            default_pause_after=True,
        ),
        StageSpec(
            name="audiobook_direct",
            depends_on=("audiobook_cast",),
            handler_factory=handler(AudiobookDirectStage),
            # LLM per-chunk performance director (delivery/pause/emphasis/rate).
            # Skip-clean when no brief.audiobook.direction, so a no-direction run
            # is behaviour-preserving and this never blocks narrate.
            # 2026-08-19: the 720s pin is GONE (None = inherit the base ceiling).
            # It only ever existed to restate "outer must cover the inner LLM
            # budget", which is exactly what LLM_STAGE_BASE_CEILING_S is for — one
            # ``stage_call`` of ``llm_retries + 1`` attempts (2 x 950 = 1900s under
            # a 1950s ceiling). The batched WM loop inside ``_attempt`` runs under
            # that single per-attempt budget, so this stage has the base shape and
            # no independent requirement. Inheriting also re-arms the GitOps knob
            # PIPELINE_LLM_STAGE_BASE_CEILING_S, which the literal made inert.
        ),
        StageSpec(
            name="audiobook_narrate",
            # Depends on cast (voices) + direct (delivery). direct is skip-clean,
            # so a no-direction run still narrates with cast voices + speed only.
            depends_on=("audiobook_cast", "audiobook_direct"),
            handler_factory=handler(AudiobookNarrateStage),
            # Must sit ABOVE the gateway path's own poll budget
            # (narrate._GATEWAY_POLL_BUDGET_SECONDS = 7200) plus submit/assembly
            # margin: at 3600 a full novel was killed mid-poll by the stage
            # timeout while the workers were still draining it. The direct path
            # (2 attempts x 600s per chunk, concurrency 2) fits inside it too.
            default_timeout_seconds=7500,
        ),
        StageSpec(
            name="audiobook_qa",
            # Listen-back QA (PR-7): transcribe the rendered chunk clips + WM-judge
            # each against the manuscript, flag mispronunciation / wrong-word /
            # dropped-text. Runs BEFORE stitch so a re-rolled (corrected) clip can
            # reach the delivered book — stitch prefers a qa-corrected clip when
            # present. Needs the per-chunk narrate artifacts + the manuscript spine.
            # OPT-IN + SOFT: skip-clean without brief.audiobook.qa; a skipped stage
            # still counts as a satisfied dependency (dag_executor), so a non-qa run
            # stitches the narrate clips byte-identically. It only ever DEGRADES a
            # run (via notes), never blocks the ship. Outer must cover the ASR
            # fan-out + the surgical re-roll's re-synth + the LLM ladder.
            depends_on=("audiobook_narrate",),
            handler_factory=handler(AudiobookQAStage),
            # 3600 clears the single-stage_call invariant (950 x 2 + 30 = 1930). It does
            # NOT cover this stage's 3-call evaluator-optimizer loop (5800s) — that is a
            # PRE-EXISTING, fleet-wide shortfall tracked in _LOOP_STAGE_CALLS_BACKLOG in
            # tests/services/blueprints/test_llm_timeout_ladder.py, not this PR's to fix.
            default_timeout_seconds=3600,
        ),
        StageSpec(
            name="audiobook_stitch",
            # narrate for the base clips + qa for any corrected clips (qa is
            # opt-in/skip-clean; a skipped qa still satisfies this edge, and stitch
            # then uses the narrate clips — byte-identical to the pre-re-roll DAG).
            depends_on=("audiobook_narrate", "audiobook_qa"),
            handler_factory=handler(AudiobookStitchStage),
            # One concat per chapter (sequential, ~15-25s each on the single
            # video_compiler) THEN a combined pass — so wall-clock scales with
            # chapter count. A 40-chapter book overran the old 900s budget
            # (TimeoutError mid-stitch); widen to match narrate so long books
            # finish. (Resumable per-chapter stitch is the deeper fix — tracked
            # separately.)
            default_timeout_seconds=3600,
        ),
    ],
)


get_registry().register(AUDIOBOOK_BLUEPRINT)
