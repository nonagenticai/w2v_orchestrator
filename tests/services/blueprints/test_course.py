import pytest

from src.services.blueprints.course import COURSE_BLUEPRINT
from src.services.blueprints.registry import get_registry


def test_course_registered() -> None:
    assert get_registry().resolve("course") is COURSE_BLUEPRINT


def test_course_topology() -> None:
    # course_stitch concatenates the per-lesson MP4s into one course video,
    # between compile_lessons and growth_bundle. ai_image_gen + course_thumbnail
    # stay leaves off course_outline. The opt-in source_ingest leaf runs before
    # course_outline (which consumes its grounding) but never gates a plain run.
    order = COURSE_BLUEPRINT.topological_order()
    spine = [
        "course_outline",
        "course_compile_lessons",
        "course_stitch",
        "growth_bundle",
    ]
    assert [s for s in order if s in spine] == spine
    assert {"ai_image_gen", "course_thumbnail", "source_ingest"} <= set(order)
    assert order.index("source_ingest") < order.index("course_outline")


def test_course_outline_grounds_in_source_ingest() -> None:
    # Opt-in: source_ingest skips cleanly without brief.sources, so this dep
    # never blocks a plain course run; course_outline appends grounding_prompt_block.
    assert "source_ingest" in COURSE_BLUEPRINT.stage("course_outline").depends_on
    assert COURSE_BLUEPRINT.stage("source_ingest").depends_on == ()


def test_course_media_discovery_is_an_opt_in_leaf() -> None:
    # A no-dep leaf with no consumer in the course render path (lessons render
    # independently, no real-media compositor) — recorded but not composited.
    assert COURSE_BLUEPRINT.stage("media_discovery").depends_on == ()
    consumers = [s.name for s in COURSE_BLUEPRINT.stages if "media_discovery" in s.depends_on]
    assert consumers == []


def test_course_stitch_follows_compile_lessons() -> None:
    assert COURSE_BLUEPRINT.stage("course_stitch").depends_on == ("course_compile_lessons",)
    assert COURSE_BLUEPRINT.stage("growth_bundle").depends_on == ("course_stitch",)


def test_course_thumbnail_is_a_leaf_off_outline() -> None:
    # Runs parallel to the lesson fanout; course_stitch must NOT depend on it
    # (a flaky/skipped cover can never gate the render).
    assert COURSE_BLUEPRINT.stage("course_thumbnail").depends_on == ("course_outline",)
    assert "course_thumbnail" not in COURSE_BLUEPRINT.stage("course_stitch").depends_on


def test_course_compile_lessons_factory_requires_deps() -> None:
    spec = next(s for s in COURSE_BLUEPRINT.stages if s.name == "course_compile_lessons")
    with pytest.raises(RuntimeError, match="run_engine"):
        spec.handler_factory({})
