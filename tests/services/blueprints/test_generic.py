from src.services.blueprints.generic import GENERIC_BLUEPRINT
from src.services.blueprints.registry import get_registry


def test_generic_blueprint_registered() -> None:
    bp = get_registry().resolve("generic")
    assert bp is GENERIC_BLUEPRINT


def test_generic_blueprint_topology() -> None:
    order = GENERIC_BLUEPRINT.topological_order()
    # narrate (TTS) and ai_video_gen fan out from script_draft; ai_image_gen
    # reads storyboard_draft's per-line prompts, so it follows the storyboard.
    # video_render needs storyboard + images + audio.
    # ai_video_gen is an opt-in leaf (no consumer); included to pin its
    # post-script ordering. source_ingest (opt-in multi-source grounding) is a
    # no-dep leaf feeding script_draft; media_discovery (opt-in) hangs off
    # brief_intake.
    assert order == [
        "brief_intake",
        "source_ingest",
        "media_discovery",
        "script_draft",
        "storyboard_draft",
        "narrate",
        "ai_video_gen",
        "motion_graphics_gen",
        "ai_image_gen",
        "video_render",
    ]
    # ...and on the opt-in media stages, so it can skip lines they cover.
    assert GENERIC_BLUEPRINT.stage("ai_image_gen").depends_on == (
        "storyboard_draft",
        "source_ingest",
        "media_discovery",
        "ai_video_gen",
        "motion_graphics_gen",
    )
    # source_ingest grounds the script, so it must precede script_draft.
    assert order.index("source_ingest") < order.index("script_draft")
    assert "source_ingest" in GENERIC_BLUEPRINT.stage("script_draft").depends_on


def test_generic_video_render_joins_storyboard_images_and_audio() -> None:
    render = GENERIC_BLUEPRINT.stage("video_render")
    # video_render also depends on the opt-in source_ingest + media_discovery +
    # ai_video_gen leaves so it can composite real media and AI-generated clips
    # (all skip-clean → no plain-run impact).
    assert set(render.depends_on) == {
        "storyboard_draft",
        "ai_image_gen",
        "narrate",
        "source_ingest",
        "media_discovery",
        "ai_video_gen",
        "motion_graphics_gen",
    }
    assert GENERIC_BLUEPRINT.stage("narrate").depends_on == ("script_draft",)
