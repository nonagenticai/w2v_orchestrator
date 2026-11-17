"""WebsiteReviewBlueprint — critique → redesign → script → record → compile.

Drives analysis_service (the live critique + redesign engines proven in W1)
into the standard A/V pipeline: the redesign before/after screenshots back a
narrated walkthrough of the critique findings.

Runs straight through: there are NO default review gates. It used to pause after
redesign and after the script, but the UI never announced either checkpoint, so the
owner's runs looked stalled rather than waiting for an answer. Owner decision
2026-09-15 — removed.
"""

from __future__ import annotations

from src.services.blueprints._factories import SOURCE_INGEST_FACTORY
from src.services.blueprints.base import DAGBlueprint, StageFactory, StageSpec, handler
from src.services.blueprints.registry import get_registry
from src.services.stages.base import StageHandler
from src.services.stages.generic.media_discovery import MediaDiscoveryStage
from src.services.stages.generic.site_capture import SiteCaptureStage
from src.services.stages.growth_bundle.handler import GrowthBundleStage
from src.services.stages.primitives.compile_stage import COMPILE_STAGE_OUTER_TIMEOUT_SECONDS
from src.services.stages.website_review.compile import WebsiteReviewCompileStage
from src.services.stages.website_review.critique import WebsiteCritiqueStage
from src.services.stages.website_review.record import WebsiteReviewRecordStage
from src.services.stages.website_review.redesign import WebsiteRedesignStage
from src.services.stages.website_review.script import WebsiteReviewScriptStage
from src.services.stages.website_review.slide_plan import WebsiteReviewSlidePlanStage
from src.services.stages.website_review.slides import WebsiteReviewSlidesStage
from src.services.stages.website_review.visual_director import (
    WebsiteReviewVisualDirectorStage,
)


def _website_redesign(deps: dict) -> StageHandler:
    # Deps-aware (not bare ``handler(...)``) purely for the session_factory: the
    # stage stashes its in-flight html_render job id in its own stage_outputs row
    # so a resumable-timeout re-dispatch RE-ATTACHES instead of submitting a second
    # job and throwing the first render away. ``.get`` — absence degrades to the old
    # resubmit behaviour rather than failing the run.
    return WebsiteRedesignStage(session_factory=deps.get("session_factory"))


WEBSITE_REVIEW_BLUEPRINT = DAGBlueprint(
    id="website_review_v1",
    project_type="website_review",
    terminal_artifact_stage="website_review_compile",
    stages=[
        StageSpec(
            name="website_critique",
            handler_factory=handler(WebsiteCritiqueStage),
            # Worst case = start(2 attempts x 60s) + wait(2 attempts x 600s) = 1320s;
            # budget must exceed it or the dispatcher kills the stage mid-retry.
            default_timeout_seconds=1400,
        ),
        StageSpec(
            name="website_redesign",
            depends_on=("website_critique",),
            handler_factory=StageFactory(WebsiteRedesignStage, _website_redesign),
            # ⚠️ RAISED 5400 -> 7200, 2026-09-02. This is the outer Timeout the
            # dispatcher enforces on ONE attempt of the handler, and 5400 did not
            # cover what one attempt can legitimately spend — so a stage that was
            # still polling healthily was killed mid-retry.
            #
            # One-attempt worst case, all four terms:
            #   start          1 x 60                        =   60   (retries=0)
            #   claim queue    1 x 1810                      = 1810   (design_queue)
            #   wait           (1+7) x (590+10)              = 4800
            #   wait backoff   1.25 x (2**7 - 1)             =  159   (jitter at max)
            #   result         2 x 60                        =  120
            #                                                 ------
            #                                                  6949
            # 7200 covers it with 251s of margin and stays BELOW html_render's
            # REDESIGN_MAX_JOB_LIFETIME_SECONDS (9000s after gitops #3319), so the
            # stage can never outlive the job it is watching. Derived, not guessed:
            # tests/services/stages/website_review/test_redesign.py recomputes every
            # term from the constants and reds if any of them moves.
            #
            # This is a ONE-ATTEMPT budget on purpose. The stage's TOTAL coverage of a
            # slow render comes from the resume ladder (pipeline.max_stage_attempts),
            # which now RE-ATTACHES to the same job instead of orphaning it — see
            # stages/shared/design_job_stash.py. Raising redesign.py _WAIT_RETRIES to
            # buy the same coverage inside one attempt is blocked by stage_call's
            # hard cap of 10 and its unbounded exponential backoff.
            default_timeout_seconds=7200,
            # Owner decision 2026-09-15: NO review gate here. The owner's runs
            # "paused a couple of times" and read as stalled — the UI never announced
            # the checkpoint, so a gate that exists to be answered by a human was
            # invisible to the only human who could answer it. A website_review now
            # runs straight through.
            default_pause_after=False,
        ),
        StageSpec(
            name="website_review_script",
            # source_ingest is opt-in (skips → completed-empty without
            # brief.sources), so this dep never blocks a plain review run; when
            # the operator grounds the review in extra sources the script
            # consumes grounding_prompt_block(ctx).
            depends_on=("website_critique", "website_redesign", "source_ingest"),
            handler_factory=handler(WebsiteReviewScriptStage),
            # Owner decision 2026-09-15 — see website_redesign above. Removed, not
            # opted out: ``brief.pause_for_review`` could already drop both gates,
            # but every run the owner started took the default and stopped.
            default_pause_after=False,
        ),
        StageSpec(
            name="website_review_record",
            depends_on=("website_review_script",),
            handler_factory=handler(WebsiteReviewRecordStage),
            # Outer Timeout must cover the inner stage_gather worst case: a
            # single bad line takes 2 attempts x 600s = 1200s. The previous
            # 900s truncated the second attempt mid-flight on the first TTS
            # transport blip (E2E `1a2b52eb`, 2026-05-30 06:26 UTC).
            default_timeout_seconds=1300,
        ),
        # Opt-in (skips cleanly without brief.source_url): records the LIVE site in
        # per-screen mode (brief.site_capture.screens + motion:"clips") so the
        # BEFORE act shows discrete settled screens + reveal clips instead of a
        # full-page scroll. A no-dep leaf; compile soft-consumes its output.
        StageSpec(
            name="site_capture",
            handler_factory=handler(SiteCaptureStage),
            default_timeout_seconds=1300,
        ),
        # Opt-in (skips cleanly without brief.sources): ingests extra operator
        # sources (websites / docs / pinned media) and exposes
        # grounding_text — consumed by website_review_script. A no-dep leaf;
        # never blocks a plain review run.
        StageSpec(
            name="source_ingest",
            handler_factory=SOURCE_INGEST_FACTORY,
            default_timeout_seconds=900,
        ),
        # Opt-in (skips unless brief.media_discovery is set): searches media-discovery
        # media-discovery and records ranked candidates. A no-dep leaf consumed
        # by website_review_compile, which overlays the re-published candidates
        # on the LOW-VALUE full-page scroll-fallback lines — never the settled
        # cover screens / redesign sections / reveal clips.
        StageSpec(
            name="media_discovery",
            handler_factory=handler(MediaDiscoveryStage),
            default_timeout_seconds=300,
        ),
        # Maps each AFTER-act narration line to the redesign section it discusses
        # (LLM + structural carry-forward) so compile shows discrete settled
        # sections synced to the words instead of a full-page scroll. Soft: skips
        # cleanly when no redesign-section assets / after lines exist, and compile
        # falls back to the scroll pools.
        StageSpec(
            name="website_review_visual_director",
            depends_on=("website_review_script", "website_redesign", "site_capture"),
            handler_factory=handler(WebsiteReviewVisualDirectorStage),
            # Outer must cover the LLM ladder inner budget (660s) — see
            # PipelineSettings.llm_budget_seconds.
        ),
        # Authors the BEFORE-act deck for the fully-AI path: one card per before-act
        # narration line, grounded in the critique findings the line cites. Stays out
        # of the way when the operator authored `slide` shots by hand, and falls back
        # to a deterministic critique projection (+ an advisory) when the model's
        # reply is unusable — it never returns a hollow deck silently.
        StageSpec(
            name="website_review_slide_plan",
            depends_on=("website_critique", "website_review_script"),
            handler_factory=handler(WebsiteReviewSlidePlanStage),
            # Outer must cover the LLM ladder inner budget (660s) — see
            # PipelineSettings.llm_budget_seconds. Same shape as the visual director.
        ),
        # Renders the BEFORE-act `slide` shots — operator-authored ones from
        # `brief.shotlist`, else the cards from `website_review_slide_plan`. Returns
        # an empty deck when there are neither, so it never blocks a plain review.
        # compile binds each rendered slide to its line_id.
        StageSpec(
            name="website_review_slides",
            depends_on=("website_review_script", "website_review_slide_plan"),
            handler_factory=handler(WebsiteReviewSlidesStage),
            default_timeout_seconds=600,
        ),
        StageSpec(
            name="website_review_compile",
            # source_ingest + media_discovery are opt-in skip-clean leaves; the
            # dep is ordering only (the leaf must complete before compile reads
            # its re-published media) and never blocks a plain review run. When
            # grounded, compile overlays that media on the scroll-fallback lines.
            depends_on=(
                "website_review_record",
                "site_capture",
                "website_review_visual_director",
                "website_review_slides",
                "source_ingest",
                "media_discovery",
            ),
            handler_factory=handler(WebsiteReviewCompileStage),
            # Outer Timeout must cover the compile-wait backstop below it — see
            # compile_stage.COMPILE_STAGE_OUTER_TIMEOUT_SECONDS.
            default_timeout_seconds=COMPILE_STAGE_OUTER_TIMEOUT_SECONDS,
        ),
        StageSpec(
            name="growth_bundle",
            depends_on=("website_review_compile",),
            handler_factory=handler(GrowthBundleStage),
        ),
    ],
)


get_registry().register(WEBSITE_REVIEW_BLUEPRINT)
