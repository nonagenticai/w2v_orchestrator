from src.services.blueprints.promo import PROMO_BLUEPRINT
from src.services.blueprints.registry import get_registry
from src.services.stages.primitives.compile_stage import COMPILE_STAGE_OUTER_TIMEOUT_SECONDS


def test_promo_registered_under_correct_type() -> None:
    assert get_registry().resolve("promo") is PROMO_BLUEPRINT


def test_promo_topology() -> None:
    order = PROMO_BLUEPRINT.topological_order()
    # site_capture is an opt-in no-dep leaf: it co-schedules with promo_analyze
    # at run open and feeds promo_script (scraped copy) + promo_render (footage).
    # promo_narrate (TTS), ai_image_gen and ai_video_gen all fan out from
    # promo_script. promo_render needs storyboard + images + audio + capture.
    # Assert relative ordering, not one fixed order: site_capture and
    # promo_analyze are sibling roots whose tie-break order is incidental.
    pos = {name: i for i, name in enumerate(order)}
    assert set(order) == {
        "promo_analyze",
        "site_capture",
        # source_ingest (opt-in multi-source grounding) is a no-dep leaf feeding
        # analyze + script; media_discovery (opt-in) hangs off analyze.
        "source_ingest",
        "media_discovery",
        "promo_script",
        "promo_storyboard",
        "promo_narrate",
        # promo_image_prompts writes per-line image prompts from the storyboard;
        # ai_image_gen consumes them so the accents are story-relevant.
        "promo_image_prompts",
        "ai_image_gen",
        "ai_video_gen",
        # promo_motion_brief (opt-in) designs one motion card per script line;
        # motion_graphics_gen (opt-in) renders deterministic branded motion clips.
        "promo_motion_brief",
        "motion_graphics_gen",
        # promo_sound_director (opt-in) picks per-line accent SFX cues.
        "promo_sound_director",
        # promo_visual_director maps footage lines → captured demo screens so
        # promo_render frame-syncs the video to the narration.
        "promo_visual_director",
        # promo_footage_direction (opt-in) plans per-footage-line cursor/zoom/callout
        # INTENT; promo_render threads the plans into the compositor shots.
        "promo_footage_direction",
        # promo_slides (slide_service) + promo_html (reused redesign) fan out
        # from promo_script and feed promo_render's per-line visual plan.
        "promo_slides",
        "promo_html",
        "promo_render",
        # promo_footage_critique (opt-in L4 loop) is the terminal artifact stage:
        # it re-renders weak shots or passes promo_render's artifact through unchanged.
        "promo_footage_critique",
        "promo_review",
        "growth_bundle",
    }
    assert pos["promo_analyze"] < pos["promo_script"]
    assert pos["site_capture"] < pos["promo_script"]
    assert pos["promo_script"] < pos["promo_storyboard"]
    assert pos["promo_storyboard"] < pos["promo_render"]
    assert pos["site_capture"] < pos["promo_render"]
    assert pos["promo_script"] < pos["promo_slides"] < pos["promo_render"]
    assert pos["promo_script"] < pos["promo_html"] < pos["promo_render"]
    assert pos["promo_storyboard"] < pos["promo_image_prompts"] < pos["ai_image_gen"]
    # visual_director needs script + storyboard + capture, and feeds render.
    assert pos["promo_script"] < pos["promo_visual_director"] < pos["promo_render"]
    assert pos["promo_storyboard"] < pos["promo_visual_director"]
    assert pos["site_capture"] < pos["promo_visual_director"]
    # Linear delivery tail: render → footage_critique (terminal artifact) → review → bundle.
    assert (
        pos["promo_render"]
        < pos["promo_footage_critique"]
        < pos["promo_review"]
        < pos["growth_bundle"]
    )
    # source_ingest grounds analyze + script; media_discovery runs after analyze.
    assert pos["source_ingest"] < pos["promo_analyze"] < pos["promo_script"]
    assert pos["promo_analyze"] < pos["media_discovery"]
    # promo_sound_director fans out from the script and feeds render.
    assert pos["promo_script"] < pos["promo_sound_director"] < pos["promo_render"]
    # promo_footage_direction needs storyboard/script/narrate/capture/visual_director
    # and feeds render (the L1 plan seam).
    assert pos["promo_storyboard"] < pos["promo_footage_direction"] < pos["promo_render"]
    assert pos["promo_visual_director"] < pos["promo_footage_direction"]
    assert pos["site_capture"] < pos["promo_footage_direction"]


def test_promo_footage_critique_is_terminal_artifact_stage() -> None:
    # (a) the L4 footage-critique loop is the delivered artifact, not promo_render.
    assert PROMO_BLUEPRINT.terminal_artifact_stage == "promo_footage_critique"
    # (b) the critique stage depends only on promo_render (re-render source).
    assert PROMO_BLUEPRINT.stage("promo_footage_critique").depends_on == ("promo_render",)
    # (c) promo_review now scores the delivered (critique) artifact, not promo_render.
    assert PROMO_BLUEPRINT.stage("promo_review").depends_on == ("promo_footage_critique",)
    # growth_bundle still hangs off promo_review, unchanged.
    assert PROMO_BLUEPRINT.stage("growth_bundle").depends_on == ("promo_review",)
    # (d) the blueprint constructor validates the DAG (acyclic + terminal stage exists);
    # topological_order raises if the graph is cyclic, so a clean call proves validity.
    assert PROMO_BLUEPRINT.topological_order()
    # (e) the surgical-edit re-stitch cut stays at the REAL compile (promo_render),
    # decoupled from the post-compile deliverable so reorder/renarrate re-render
    # correctly instead of stranding promo_render as seeded-completed.
    assert PROMO_BLUEPRINT.compile_cut_stage == "promo_render"


def test_promo_render_joins_storyboard_images_audio_and_capture() -> None:
    render = PROMO_BLUEPRINT.stage("promo_render")
    assert set(render.depends_on) == {
        "promo_storyboard",
        "ai_image_gen",
        "promo_narrate",
        "site_capture",
        "promo_visual_director",
        "promo_slides",
        "promo_html",
        # Real-media sources composited over image-suitable lines (opt-in, skip-clean).
        "source_ingest",
        "media_discovery",
        # AI-generated clips composited over leftover low-value lines (opt-in).
        "ai_video_gen",
        # Branded motion-graphics clips, above AI video (opt-in, skip-clean).
        "motion_graphics_gen",
        # Per-line accent SFX cues (opt-in, skip-clean).
        "promo_sound_director",
        # Per-footage-line cursor/zoom/callout INTENT plan (opt-in, skip-clean).
        "promo_footage_direction",
    }
    assert PROMO_BLUEPRINT.stage("promo_narrate").depends_on == ("promo_script",)


def test_promo_visual_director_deps() -> None:
    assert set(PROMO_BLUEPRINT.stage("promo_visual_director").depends_on) == {
        "promo_script",
        "promo_storyboard",
        "site_capture",
    }


def test_promo_slides_and_html_fan_out_from_script() -> None:
    assert PROMO_BLUEPRINT.stage("promo_slides").depends_on == ("promo_script",)
    assert PROMO_BLUEPRINT.stage("promo_html").depends_on == ("promo_script",)


def test_promo_script_grounds_on_analyze_and_capture() -> None:
    assert set(PROMO_BLUEPRINT.stage("promo_script").depends_on) == {
        "promo_analyze",
        "site_capture",
        "source_ingest",
    }


def test_promo_analyze_grounds_on_source_ingest() -> None:
    assert PROMO_BLUEPRINT.stage("promo_analyze").depends_on == ("source_ingest",)


def test_source_ingest_and_site_capture_are_optin_leaves() -> None:
    assert PROMO_BLUEPRINT.stage("site_capture").depends_on == ()
    assert PROMO_BLUEPRINT.stage("source_ingest").depends_on == ("site_capture",)


def test_promo_pause_defaults() -> None:
    assert PROMO_BLUEPRINT.stage("promo_analyze").default_pause_after is False
    assert PROMO_BLUEPRINT.stage("promo_script").default_pause_after is True
    assert PROMO_BLUEPRINT.stage("promo_storyboard").default_pause_after is True
    assert PROMO_BLUEPRINT.stage("promo_render").default_pause_after is False
    assert PROMO_BLUEPRINT.stage("promo_review").default_pause_after is False


def test_promo_timeout_budget() -> None:
    """Render gets the long timeout; LLM stages get a generous budget."""
    # Render is video-compile (heaviest stage). The outer Timeout must cover the
    # compile-wait backstop (defect-1 fix, 2026-09-17 incident) — see
    # compile_stage.COMPILE_STAGE_OUTER_TIMEOUT_SECONDS.
    assert (
        PROMO_BLUEPRINT.stage("promo_render").default_timeout_seconds
        == COMPILE_STAGE_OUTER_TIMEOUT_SECONDS
    )
    # Review/analyze are LLM calls; they inherit the env-configurable base LLM ceiling
    # (None sentinel, was a hardcoded 700) so a slow-vLLM raise in GitOps reaches them.
    assert PROMO_BLUEPRINT.stage("promo_review").default_timeout_seconds is None
    assert PROMO_BLUEPRINT.stage("promo_analyze").default_timeout_seconds is None


def test_promo_motion_brief_feeds_motion_graphics_gen() -> None:
    """The brief needs the script, the measured narration and the storyboard (the
    fill plan's holds); motion_graphics_gen renders what it authored."""
    order = PROMO_BLUEPRINT.topological_order()
    pos = {name: i for i, name in enumerate(order)}
    assert set(PROMO_BLUEPRINT.stage("promo_motion_brief").depends_on) == {
        "promo_script",
        "promo_storyboard",
        "promo_narrate",
    }
    assert set(PROMO_BLUEPRINT.stage("motion_graphics_gen").depends_on) == {
        "promo_script",
        "promo_motion_brief",
    }
    assert pos["promo_narrate"] < pos["promo_motion_brief"] < pos["motion_graphics_gen"]
    assert pos["motion_graphics_gen"] < pos["promo_render"]
