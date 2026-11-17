"""CourseBlueprint — outline → compile_lessons (sub-DAG dispatcher)."""

from __future__ import annotations

from src.services.blueprints._factories import SOURCE_INGEST_FACTORY
from src.services.blueprints.base import DAGBlueprint, StageFactory, StageSpec, handler
from src.services.blueprints.registry import get_registry
from src.services.stages.base import StageHandler
from src.services.stages.course.compile_lessons import CourseCompileLessonsStage
from src.services.stages.course.outline import CourseOutlineStage
from src.services.stages.course.stitch import CourseStitchStage
from src.services.stages.course.thumbnail import CourseThumbnailStage
from src.services.stages.generic.ai_image_gen import AiImageGenStage
from src.services.stages.generic.media_discovery import MediaDiscoveryStage
from src.services.stages.growth_bundle.handler import GrowthBundleStage
from src.services.stages.primitives.compile_stage import COMPILE_STAGE_OUTER_TIMEOUT_SECONDS


def _compile_lessons(deps: dict) -> StageHandler:
    try:
        return CourseCompileLessonsStage(
            run_engine=deps["run_engine"],
            redis=deps["redis"],
            session_factory=deps["session_factory"],
        )
    except KeyError as exc:
        raise RuntimeError(
            "course_compile_lessons requires run_engine + redis + session_factory "
            "in handler_deps; ensure RunDispatcher passes them when constructing "
            "DAGExecutor."
        ) from exc


COURSE_BLUEPRINT = DAGBlueprint(
    id="course_v1",
    project_type="course",
    terminal_artifact_stage="course_stitch",
    stages=[
        StageSpec(
            name="course_outline",
            # source_ingest is opt-in (skips → completed-empty without
            # brief.sources), so this dep never blocks a plain course run; when
            # the operator grounds the course in extra sources the outline
            # consumes grounding_prompt_block(ctx).
            depends_on=("source_ingest",),
            handler_factory=handler(CourseOutlineStage),
            default_pause_after=True,
        ),
        # Opt-in (skips cleanly without brief.sources): ingests extra operator
        # sources (websites / docs / pinned media) and exposes
        # grounding_text — consumed by course_outline. A no-dep leaf; never
        # blocks a plain course run.
        StageSpec(
            name="source_ingest",
            handler_factory=SOURCE_INGEST_FACTORY,
            default_timeout_seconds=900,
        ),
        # Opt-in (skips unless brief.media_discovery is set): searches media-discovery
        # media-discovery and records ranked candidates. A no-dep leaf with no
        # consumer at the COURSE level — course_stitch only concatenates the
        # child lesson videos. Course grounding (incl. real-media compositing)
        # reaches the screen via each lesson child instead: the wizard
        # propagates the course's sources onto every lesson, and lesson_compile
        # now composites them as cutaways. So this course-level leaf stays
        # recorded-but-not-composited (lessons do the compositing).
        StageSpec(
            name="media_discovery",
            handler_factory=handler(MediaDiscoveryStage),
            default_timeout_seconds=300,
        ),
        StageSpec(
            name="course_compile_lessons",
            depends_on=("course_outline",),
            handler_factory=StageFactory(CourseCompileLessonsStage, _compile_lessons),
            default_timeout_seconds=14400,  # 4h — child-run fanout for N lessons
        ),
        StageSpec(
            name="course_stitch",
            depends_on=("course_compile_lessons",),
            handler_factory=handler(CourseStitchStage),
            # Outer Timeout must cover the inner compile wait
            # (COMPILE_WAIT_CEILING_SECONDS + backstop margin) or it abandons a
            # stitch the compiler still considers healthy. Same ceiling every
            # CompileStageBase compile stage uses.
            default_timeout_seconds=COMPILE_STAGE_OUTER_TIMEOUT_SECONDS,
        ),
        StageSpec(
            name="growth_bundle",
            depends_on=("course_stitch",),
            handler_factory=handler(GrowthBundleStage),
        ),
        StageSpec(
            name="ai_image_gen",
            depends_on=("course_outline",),
            handler_factory=handler(AiImageGenStage),
            # Inner stage_gather worst case: 2 attempts x 600s = 1200s per item.
            default_timeout_seconds=1300,
        ),
        StageSpec(
            name="course_thumbnail",
            depends_on=("course_outline",),
            handler_factory=handler(CourseThumbnailStage),
            # One AI cover image; runs parallel to the lesson fanout, read
            # best-effort by course_stitch. NOT a dependency of course_stitch —
            # a flaky/skipped cover must never gate the render.
            default_timeout_seconds=1300,
        ),
    ],
)


get_registry().register(COURSE_BLUEPRINT)
