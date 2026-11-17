from src.services.blueprints.lesson import LESSON_BLUEPRINT
from src.services.blueprints.registry import get_registry


def test_lesson_registered() -> None:
    assert get_registry().resolve("lesson") is LESSON_BLUEPRINT


def test_narrate_and_image_gen_co_scheduled() -> None:
    """Narration (TTS) and image generation are the two long poles. Both depend
    only on lesson_slides so the DAG scheduler makes them eligible in the same
    round and runs them concurrently."""
    narrate = LESSON_BLUEPRINT.stage("lesson_narrate")
    image_gen = LESSON_BLUEPRINT.stage("ai_image_gen")
    assert narrate.depends_on == ("lesson_slides",)
    assert image_gen.depends_on == ("lesson_slides",)


def test_lesson_topology_respects_deps() -> None:
    order = LESSON_BLUEPRINT.topological_order()
    assert order.index("lesson_script") < order.index("lesson_slides")
    assert order.index("lesson_slides") < order.index("lesson_narrate")
    assert order.index("lesson_slides") < order.index("ai_image_gen")
    assert order.index("lesson_slides") < order.index("lesson_record")
    # record composites the per-slide hero images, so it waits on ai_image_gen
    assert order.index("ai_image_gen") < order.index("lesson_record")
    assert order.index("lesson_narrate") < order.index("lesson_compile")
    assert order.index("lesson_record") < order.index("lesson_compile")
    # The opt-in source_ingest leaf runs before lesson_script (which grounds in
    # it) but never gates a plain lesson run.
    assert order.index("source_ingest") < order.index("lesson_script")


def test_lesson_script_grounds_in_source_ingest() -> None:
    # Opt-in: source_ingest skips cleanly without brief.sources, so this dep never
    # blocks a plain lesson run; lesson_script appends grounding_prompt_block. A
    # grounded course propagates its sources onto each lesson child.
    assert "source_ingest" in LESSON_BLUEPRINT.stage("lesson_script").depends_on
    assert LESSON_BLUEPRINT.stage("source_ingest").depends_on == ()


def test_lesson_media_discovery_is_an_opt_in_leaf_consumed_by_compile() -> None:
    # A no-dep, skip-clean opt-in leaf now consumed by lesson_compile, which
    # composites the re-published candidates as full-frame cutaways.
    assert LESSON_BLUEPRINT.stage("media_discovery").depends_on == ()
    consumers = [s.name for s in LESSON_BLUEPRINT.stages if "media_discovery" in s.depends_on]
    assert consumers == ["lesson_compile"]


def test_lesson_compile_grounds_in_source_ingest_and_media_discovery() -> None:
    # Ordering-only deps on the two opt-in skip-clean leaves so their
    # re-published media is ready before lesson_compile composites it. Both stay
    # no-dep leaves, so a plain lesson run is unaffected.
    compile_deps = LESSON_BLUEPRINT.stage("lesson_compile").depends_on
    assert "source_ingest" in compile_deps
    assert "media_discovery" in compile_deps
    # ai_video_gen (opt-in leaf) is also consumed — its clips fill leftover lines.
    assert "ai_video_gen" in compile_deps
    assert LESSON_BLUEPRINT.stage("source_ingest").depends_on == ()
    assert LESSON_BLUEPRINT.stage("media_discovery").depends_on == ()
