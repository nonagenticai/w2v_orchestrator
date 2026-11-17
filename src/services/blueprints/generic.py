from __future__ import annotations

from src.services.blueprints._factories import SOURCE_INGEST_FACTORY
from src.services.blueprints.base import DAGBlueprint, StageFactory, StageSpec, handler
from src.services.blueprints.registry import get_registry
from src.services.stages.generic.ai_image_gen import (
    GenericAiImageGenStage,
    make_generic_ai_image_gen_stage,
)
from src.services.stages.generic.ai_video_gen import AiVideoGenStage
from src.services.stages.generic.brief_intake import BriefIntakeStage
from src.services.stages.generic.media_discovery import MediaDiscoveryStage
from src.services.stages.generic.motion_graphics_gen import MotionGraphicsGenStage
from src.services.stages.generic.narrate import NarrateStage
from src.services.stages.generic.script_draft import ScriptDraftStage
from src.services.stages.generic.storyboard_draft import StoryboardDraftStage
from src.services.stages.generic.video_render import VideoRenderStage
from src.services.stages.primitives.compile_stage import COMPILE_STAGE_OUTER_TIMEOUT_SECONDS

GENERIC_BLUEPRINT = DAGBlueprint(
    id="generic_v1",
    project_type="generic",
    terminal_artifact_stage="video_render",
    stages=[
        StageSpec(
            name="brief_intake",
            handler_factory=handler(BriefIntakeStage),
            default_timeout_seconds=30,
        ),
        StageSpec(
            name="script_draft",
            # source_ingest is opt-in (skips → completed-empty without
            # brief.sources), so this dep never blocks a plain generic run.
            depends_on=("brief_intake", "source_ingest"),
            handler_factory=handler(ScriptDraftStage),
            default_pause_after=True,
        ),
        StageSpec(
            name="storyboard_draft",
            depends_on=("script_draft",),
            handler_factory=handler(StoryboardDraftStage),
            default_timeout_seconds=60,
            default_pause_after=True,
        ),
        StageSpec(
            name="narrate",
            depends_on=("script_draft",),
            handler_factory=handler(NarrateStage),
            # Inner stage_gather worst case: 2 attempts x 600s = 1200s per
            # item. Mirrors website_review_record's tuning.
            default_timeout_seconds=1300,
        ),
        StageSpec(
            name="video_render",
            # source_ingest + media_discovery + ai_video_gen are opt-in skip-clean
            # leaves, so these deps never block a plain generic run (same as
            # script_draft's source_ingest dep); they let video_render composite
            # real media and AI-generated clips.
            depends_on=(
                "storyboard_draft",
                "ai_image_gen",
                "narrate",
                "source_ingest",
                "media_discovery",
                "ai_video_gen",
                "motion_graphics_gen",
            ),
            handler_factory=handler(VideoRenderStage),
            # The stage Timeout middleware wraps the entire retry loop, so it
            # must cover the compile-wait backstop below it — see
            # compile_stage.COMPILE_STAGE_OUTER_TIMEOUT_SECONDS.
            default_timeout_seconds=COMPILE_STAGE_OUTER_TIMEOUT_SECONDS,
        ),
        StageSpec(
            name="ai_image_gen",
            # Reads storyboard_draft's per-line `prompts` (run_setup seeds
            # `ai_image_gen.prompts_from_stage = "storyboard_draft"` for every
            # generic run), so each shot gets its own still. Depending on
            # script_draft alone left it nothing to read and every shot a text card.
            # It ALSO waits on the opt-in media stages video_render composites:
            # a line they cover never shows its AI still, so GenericAiImageGenStage
            # skips it (same rule as video_render, `assign_generic_line_media`).
            # All four skip-clean when not requested, so a plain run is unchanged.
            depends_on=(
                "storyboard_draft",
                "source_ingest",
                "media_discovery",
                "ai_video_gen",
                "motion_graphics_gen",
            ),
            # session_factory: reads the parent run's images on a regenerate.
            handler_factory=StageFactory(GenericAiImageGenStage, make_generic_ai_image_gen_stage),
            # Inner stage_gather worst case: 2 attempts x 600s = 1200s per item.
            default_timeout_seconds=1300,
        ),
        # Opt-in generative video. Empty result unless `ai_video_gen` is set
        # in the run config snapshot; see ai_video_gen.AiVideoGenStage.
        StageSpec(
            name="ai_video_gen",
            depends_on=("script_draft",),
            handler_factory=handler(AiVideoGenStage),
            # Inner stage_gather worst case: 2 attempts x 1200s = 2400s per item.
            default_timeout_seconds=2500,
        ),
        # Opt-in deterministic branded motion graphics. Empty result unless
        # `motion_graphics` is set in the run config snapshot. video_render lays
        # the clips above generative AI video. See MotionGraphicsGenStage.
        StageSpec(
            name="motion_graphics_gen",
            depends_on=("script_draft",),
            handler_factory=handler(MotionGraphicsGenStage),
            # Inner stage_gather worst case: 2 attempts x 900s = 1800s per item.
            default_timeout_seconds=1900,
        ),
        # source_ingest is OPT-IN (skips without brief.sources). No-dep leaf so
        # the scrape/parse fan-out overlaps brief_intake; grounds script_draft.
        StageSpec(
            name="source_ingest",
            handler_factory=SOURCE_INGEST_FACTORY,
            default_timeout_seconds=900,
        ),
        # media_discovery is OPT-IN (skips without brief.media_discovery); parallel
        # leaf consumed by video_render (real-media compositing). Mirrors promo.
        StageSpec(
            name="media_discovery",
            depends_on=("brief_intake",),
            handler_factory=handler(MediaDiscoveryStage),
            default_timeout_seconds=300,
        ),
    ],
)


get_registry().register(GENERIC_BLUEPRINT)
