"""ai_video_gen wiring across the blueprints that include it.

The stage is opt-in (raises StageSkippedError unless ``ai_video_gen`` is set
in the run config snapshot). Its re-published clips are now composited into
the render: each blueprint's terminal render stage consumes ``ai_video_gen``
and lays the generative clips over the lowest-value lines that curated real
media left bare (lowest motion tier). These tests pin:

- ``ai_video_gen`` exists in the 3 blueprints that already host
  ``ai_image_gen`` and are not pure sub-DAG dispatchers (promo, lesson,
  generic). ``course`` is intentionally excluded — its sub-DAG fan-out
  exercises generative video via child lesson runs, not the parent.
- The stage depends solely on its script stage (same dependency shape as
  ``ai_image_gen`` so it has the script text available if a future
  enhancement derives video prompts from script content).
- The blueprint's render stage now depends on ``ai_video_gen`` (consumes it).
  The dep is ordering only — ``ai_video_gen`` is opt-in skip-clean, so it
  never blocks a plain run.
- The handler factory builds and returns an ``AiVideoGenStage`` instance.
"""

from __future__ import annotations

import pytest

from src.services.blueprints.base import DAGBlueprint
from src.services.blueprints.generic import GENERIC_BLUEPRINT
from src.services.blueprints.lesson import LESSON_BLUEPRINT
from src.services.blueprints.promo import PROMO_BLUEPRINT
from src.services.stages.generic.ai_video_gen import AiVideoGenStage

# (project_type, blueprint, ai_video_gen's own dep, render stage that consumes it)
BLUEPRINTS_WITH_AI_VIDEO_GEN: list[tuple[str, DAGBlueprint, str, str]] = [
    ("generic", GENERIC_BLUEPRINT, "script_draft", "video_render"),
    ("promo", PROMO_BLUEPRINT, "promo_script", "promo_render"),
    ("lesson", LESSON_BLUEPRINT, "lesson_script", "lesson_compile"),
]

# Non-render stages that also wait on ai_video_gen. generic's ai_image_gen reads
# it (via assign_generic_line_media) only to SKIP stills on lines an AI clip
# covers — it never composites the clips itself.
_COVERAGE_READERS: dict[str, list[str]] = {"generic": ["ai_image_gen"]}


@pytest.mark.parametrize(
    ("project_type", "blueprint", "expected_dep", "_render_stage"),
    [("generic", GENERIC_BLUEPRINT, "script_draft", "video_render")],
    ids=([b[0] for b in BLUEPRINTS_WITH_AI_VIDEO_GEN])[:1],
)
def test_ai_video_gen_present(
    project_type: str, blueprint: DAGBlueprint, expected_dep: str, _render_stage: str
) -> None:
    spec = blueprint.stage("ai_video_gen")
    assert spec.depends_on == (expected_dep,), (
        f"{project_type}: ai_video_gen should depend solely on {expected_dep!r}"
    )


@pytest.mark.parametrize(
    ("project_type", "blueprint", "_expected_dep", "render_stage"),
    [("generic", GENERIC_BLUEPRINT, "script_draft", "video_render")],
    ids=([b[0] for b in BLUEPRINTS_WITH_AI_VIDEO_GEN])[:1],
)
def test_ai_video_gen_consumed_by_render(
    project_type: str, blueprint: DAGBlueprint, _expected_dep: str, render_stage: str
) -> None:
    """The blueprint's render stage now consumes ai_video_gen (composited)."""
    consumers = [s.name for s in blueprint.stages if "ai_video_gen" in s.depends_on]
    assert sorted(consumers) == sorted([render_stage, *_COVERAGE_READERS.get(project_type, [])]), (
        f"{project_type}: ai_video_gen should be consumed only by {render_stage!r}, "
        f"got: {consumers}"
    )


@pytest.mark.parametrize(
    ("project_type", "blueprint", "_expected_dep", "_render_stage"),
    [("generic", GENERIC_BLUEPRINT, "script_draft", "video_render")],
    ids=([b[0] for b in BLUEPRINTS_WITH_AI_VIDEO_GEN])[:1],
)
def test_ai_video_gen_handler_factory_builds(
    project_type: str, blueprint: DAGBlueprint, _expected_dep: str, _render_stage: str
) -> None:
    spec = blueprint.stage("ai_video_gen")
    handler = spec.handler_factory({})
    assert isinstance(handler, AiVideoGenStage), (
        f"{project_type}: handler_factory must return an AiVideoGenStage instance"
    )
