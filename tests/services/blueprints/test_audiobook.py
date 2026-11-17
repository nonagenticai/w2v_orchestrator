from src.database.models import Project
from src.services.blueprints.audiobook import AUDIOBOOK_BLUEPRINT
from src.services.blueprints.registry import get_registry


def test_audiobook_registered() -> None:
    assert get_registry().resolve("audiobook") is AUDIOBOOK_BLUEPRINT
    assert AUDIOBOOK_BLUEPRINT.id == "audiobook_v1"


def test_audiobook_topology() -> None:
    assert AUDIOBOOK_BLUEPRINT.topological_order() == [
        "audiobook_ingest",
        "audiobook_plan",
        "audiobook_segment",
        "audiobook_cast",
        # audiobook_direct (performance director) sits between cast and narrate;
        # skip-clean without brief.audiobook.direction.
        "audiobook_direct",
        "audiobook_narrate",
        # audiobook_qa (PR-7 listen-back QA) runs BEFORE stitch so a re-rolled
        # (corrected) clip can reach the delivered book; opt-in + SOFT, skip-clean
        # without brief.audiobook.qa (a skipped dep still satisfies stitch, which
        # then uses the narrate clips — byte-identical).
        "audiobook_qa",
        "audiobook_stitch",
    ]


def test_audiobook_terminal_artifact_stage() -> None:
    assert AUDIOBOOK_BLUEPRINT.terminal_artifact_stage == "audiobook_stitch"


def test_audiobook_pauses_for_review_before_narration() -> None:
    pausing = {s.name for s in AUDIOBOOK_BLUEPRINT.stages if s.default_pause_after}
    # The chunk-plan pause moved to the cast stage (review casting before TTS spend).
    assert pausing == {"audiobook_ingest", "audiobook_cast"}


def test_audiobook_narrate_depends_on_cast_and_direct() -> None:
    by_name = {s.name: s for s in AUDIOBOOK_BLUEPRINT.stages}
    # segment sits between plan and cast (sub-chunk speaker attribution).
    assert by_name["audiobook_segment"].depends_on == ("audiobook_plan",)
    assert by_name["audiobook_cast"].depends_on == ("audiobook_segment",)
    # direct (performance director) depends on cast; narrate now depends on both
    # cast (voices) and direct (delivery). direct is skip-clean so a no-direction
    # run still narrates with cast voices + speed only.
    assert by_name["audiobook_direct"].depends_on == ("audiobook_cast",)
    assert by_name["audiobook_narrate"].depends_on == ("audiobook_cast", "audiobook_direct")
    # qa (listen-back) depends on narrate (needs the per-chunk clip artifacts to
    # transcribe) and now runs BEFORE stitch so a corrected clip reaches the book.
    assert by_name["audiobook_qa"].depends_on == ("audiobook_narrate",)
    # stitch now depends on narrate + qa: a re-rolled clip from qa overrides the
    # narrate clip. qa is opt-in/skip-clean, and a skipped dep still counts as
    # satisfied (dag_executor), so a non-qa run stitches the narrate clips as before.
    assert by_name["audiobook_stitch"].depends_on == ("audiobook_narrate", "audiobook_qa")


# --- one-command lift (spec §5): conditional review pauses --------------------


def _audiobook_project(brief: dict) -> Project:
    """In-memory (unflushed) audiobook Project — _seed_pause_after_stages only
    reads ``.type`` + ``.brief``, so no session / DB round-trip is needed."""
    return Project(type="audiobook", brief=brief)
