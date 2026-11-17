"""ai_image_gen wiring across all four blueprints that include it.

These tests assert that the stage is present, depends on the right upstream
stage, and that its handler factory builds. They also pin *who consumes* its
images: course keeps it as a pure leaf (no consumer); promo + generic render
join its images into the final video (full A/V); lesson composites a per-slide
hero image in lesson_record. So every consumer's render/record stage
intentionally depends on it.
"""

from __future__ import annotations

import pytest

from src.services.blueprints.base import DAGBlueprint
from src.services.blueprints.course import COURSE_BLUEPRINT
from src.services.blueprints.generic import GENERIC_BLUEPRINT
from src.services.blueprints.lesson import LESSON_BLUEPRINT
from src.services.blueprints.promo import PROMO_BLUEPRINT
from src.services.stages.generic.ai_image_gen import AiImageGenStage

# Expected exact depends_on per blueprint. promo also depends on
# promo_image_prompts (per-line story-relevant prompts feed ai_image_gen).
BLUEPRINTS_WITH_AI_IMAGE_GEN: list[tuple[str, DAGBlueprint, tuple[str, ...]]] = [
    # generic reads storyboard_draft's per-line image prompts (2026-10-20 fix)
    # and waits on the opt-in media stages so it skips lines they cover.
    (
        "generic",
        GENERIC_BLUEPRINT,
        (
            "storyboard_draft",
            "source_ingest",
            "media_discovery",
            "ai_video_gen",
            "motion_graphics_gen",
        ),
    ),
    ("promo", PROMO_BLUEPRINT, ("promo_script", "promo_image_prompts")),
    ("course", COURSE_BLUEPRINT, ("course_outline",)),
    # lesson feeds per-slide image prompts from lesson_slides
    ("lesson", LESSON_BLUEPRINT, ("lesson_slides",)),
]

# Blueprints where ai_image_gen stays a pure leaf (no stage consumes its images).
LEAF_ONLY_BLUEPRINTS: list[tuple[str, DAGBlueprint]] = [
    ("course", COURSE_BLUEPRINT),
]

# Blueprints whose render/record stage joins ai_image_gen images into the output.
RENDER_CONSUMES_IMAGES: list[tuple[str, DAGBlueprint, str]] = [
    ("generic", GENERIC_BLUEPRINT, "video_render"),
    ("promo", PROMO_BLUEPRINT, "promo_render"),
    # lesson composites each slide's hero image into the slide PNG at record time
    ("lesson", LESSON_BLUEPRINT, "lesson_record"),
]


@pytest.mark.parametrize(
    ("project_type", "blueprint", "expected_deps"),
    [
        (
            "generic",
            GENERIC_BLUEPRINT,
            (
                "storyboard_draft",
                "source_ingest",
                "media_discovery",
                "ai_video_gen",
                "motion_graphics_gen",
            ),
        )
    ],
    ids=([b[0] for b in BLUEPRINTS_WITH_AI_IMAGE_GEN])[:1],
)
def test_ai_image_gen_present(
    project_type: str, blueprint: DAGBlueprint, expected_deps: tuple[str, ...]
) -> None:
    spec = blueprint.stage("ai_image_gen")
    assert spec.depends_on == expected_deps, (
        f"{project_type}: ai_image_gen should depend on {expected_deps!r}"
    )


@pytest.mark.parametrize(
    ("project_type", "blueprint"),
    LEAF_ONLY_BLUEPRINTS,
    ids=[b[0] for b in LEAF_ONLY_BLUEPRINTS],
)
def test_ai_image_gen_is_leaf_for_lesson_and_course(
    project_type: str, blueprint: DAGBlueprint
) -> None:
    """lesson + course don't composite ai images, so nothing depends on it."""
    for stage in blueprint.stages:
        assert "ai_image_gen" not in stage.depends_on, (
            f"{project_type}: stage {stage.name!r} unexpectedly depends on ai_image_gen"
        )


@pytest.mark.parametrize(
    ("project_type", "blueprint", "render_stage"),
    [("generic", GENERIC_BLUEPRINT, "video_render")],
    ids=([b[0] for b in RENDER_CONSUMES_IMAGES])[:1],
)
def test_render_consumes_ai_image_gen(
    project_type: str, blueprint: DAGBlueprint, render_stage: str
) -> None:
    """promo + generic render join ai images, so render waits on ai_image_gen.

    The render stage is the only consumer; ai_image_gen and its storyboard /
    narrate siblings are co-scheduled in the same wave today, so this
    dependency does not change failure semantics — it only lets render read
    the images."""
    consumers = [s.name for s in blueprint.stages if "ai_image_gen" in s.depends_on]
    assert consumers == [render_stage], (
        f"{project_type}: expected only {render_stage!r} to consume ai_image_gen, got {consumers}"
    )


@pytest.mark.parametrize(
    ("project_type", "blueprint", "_expected_dep"),
    [
        (
            "generic",
            GENERIC_BLUEPRINT,
            (
                "storyboard_draft",
                "source_ingest",
                "media_discovery",
                "ai_video_gen",
                "motion_graphics_gen",
            ),
        )
    ],
    ids=([b[0] for b in BLUEPRINTS_WITH_AI_IMAGE_GEN])[:1],
)
def test_ai_image_gen_handler_factory_builds(
    project_type: str, blueprint: DAGBlueprint, _expected_dep: str
) -> None:
    spec = blueprint.stage("ai_image_gen")
    handler = spec.handler_factory({})
    assert isinstance(handler, AiImageGenStage), (
        f"{project_type}: handler_factory must return an AiImageGenStage instance"
    )
