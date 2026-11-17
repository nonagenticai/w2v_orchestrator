"""LessonBlueprint — script → slides → (narrate ‖ ai_image_gen) → record → compile.

``ai_image_gen`` is default-enabled for lessons (see ``build_config_snapshot``):
``lesson_slides`` emits one image prompt per slide, ``ai_image_gen`` renders the
hero images, and ``lesson_record`` composites them into the slide PNGs. Narration
(TTS) and image generation both depend only on ``lesson_slides`` so the DAG
scheduler runs the two long poles concurrently.
"""

from __future__ import annotations

from src.services.blueprints._factories import SOURCE_INGEST_FACTORY
from src.services.blueprints.base import DAGBlueprint, StageSpec, handler
from src.services.blueprints.registry import get_registry
from src.services.stages.generic.ai_video_gen import AiVideoGenStage
from src.services.stages.generic.media_discovery import MediaDiscoveryStage
from src.services.stages.lesson.ai_image_gen import LessonAiImageGenStage
from src.services.stages.lesson.compile import LessonCompileStage
from src.services.stages.lesson.narrate import LessonNarrateStage
from src.services.stages.lesson.record import LessonRecordStage
from src.services.stages.lesson.script import LessonScriptStage
from src.services.stages.lesson.slides import LessonSlidesStage
from src.services.stages.primitives.compile_stage import COMPILE_STAGE_OUTER_TIMEOUT_SECONDS

LESSON_BLUEPRINT = DAGBlueprint(
    id="lesson_v1",
    project_type="lesson",
    terminal_artifact_stage="lesson_compile",
    stages=[
        StageSpec(
            name="lesson_script",
            # source_ingest is opt-in (skips → completed-empty without
            # brief.sources), so this dep never blocks a plain lesson run; when
            # the lesson (or its parent course, propagated per-child by the
            # wizard) carries extra sources the script consumes
            # grounding_prompt_block(ctx).
            depends_on=("source_ingest",),
            handler_factory=handler(LessonScriptStage),
            default_pause_after=True,
        ),
        StageSpec(
            name="lesson_slides",
            depends_on=("lesson_script",),
            handler_factory=handler(LessonSlidesStage),
            default_timeout_seconds=60,
        ),
        StageSpec(
            name="lesson_narrate",
            # Depends on lesson_slides (not lesson_script) so narrate and
            # ai_image_gen become eligible in the SAME scheduler round and the
            # two long poles (TTS ‖ image gen) run concurrently.
            depends_on=("lesson_slides",),
            handler_factory=handler(LessonNarrateStage),
            # Inner stage_gather worst case: 2 attempts x 600s = 1200s per item.
            default_timeout_seconds=1300,
        ),
        StageSpec(
            name="ai_image_gen",
            depends_on=("lesson_slides",),
            # Lesson-scoped: a failure completes with zero images + a degradation,
            # so lesson_record (which depends on it) still renders plain slides
            # instead of failing the lesson and with it the whole course.
            handler_factory=handler(LessonAiImageGenStage),
            # Inner stage_gather worst case: 2 attempts x 600s = 1200s per item.
            default_timeout_seconds=1300,
        ),
        StageSpec(
            name="lesson_record",
            depends_on=("lesson_slides", "ai_image_gen"),
            handler_factory=handler(LessonRecordStage),
            # Serial deck render (wait 600s + per-slide PNG fetches); 900s is
            # adequate — BaseServiceClient handles transport retries internally.
            default_timeout_seconds=900,
        ),
        StageSpec(
            name="lesson_compile",
            # source_ingest + media_discovery + ai_video_gen are opt-in skip-clean
            # leaves; the dep is ordering only (the leaf must complete before
            # compile reads its re-published media) and never blocks a plain
            # lesson run. When the lesson is grounded, lesson_compile lays the
            # pinned/discovered media over the leading lines as full-frame
            # cutaways; AI-generated clips fill remaining lines (lowest tier).
            depends_on=(
                "lesson_narrate",
                "lesson_record",
                "source_ingest",
                "media_discovery",
                "ai_video_gen",
            ),
            handler_factory=handler(LessonCompileStage),
            # Outer Timeout must cover the compile-wait backstop below it (see
            # compile_stage.COMPILE_STAGE_OUTER_TIMEOUT_SECONDS).
            default_timeout_seconds=COMPILE_STAGE_OUTER_TIMEOUT_SECONDS,
        ),
        # Opt-in generative video. Empty result unless `ai_video_gen` is set
        # in the run config snapshot; see ai_video_gen.AiVideoGenStage.
        StageSpec(
            name="ai_video_gen",
            depends_on=("lesson_script",),
            handler_factory=handler(AiVideoGenStage),
            # Inner stage_gather worst case: 2 attempts x 1200s = 2400s per item.
            default_timeout_seconds=2500,
        ),
        # Opt-in (skips cleanly without brief.sources): ingests extra operator
        # sources (websites / docs / pinned media) and exposes
        # grounding_text — consumed by lesson_script. A no-dep leaf; never
        # blocks a plain lesson run. A grounded course propagates its sources
        # onto each lesson child (wizard buildLessonChildren), so the lesson
        # videos are grounded too.
        StageSpec(
            name="source_ingest",
            handler_factory=SOURCE_INGEST_FACTORY,
            # Scrape/parse worst case ~180s/source over a small fan-out.
            default_timeout_seconds=900,
        ),
        # Opt-in (skips unless brief.media_discovery is set): searches media-discovery
        # media-discovery and records ranked candidates. A no-dep leaf consumed
        # by lesson_compile, which lays the re-published candidates over the
        # leading lines as full-frame cutaways (mirrors generic/promo render).
        StageSpec(
            name="media_discovery",
            handler_factory=handler(MediaDiscoveryStage),
            default_timeout_seconds=300,
        ),
    ],
)


get_registry().register(LESSON_BLUEPRINT)
