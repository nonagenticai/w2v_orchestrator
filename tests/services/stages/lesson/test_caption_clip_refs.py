"""LessonCompileStage.caption_clip_refs — the word-timed caption seam (ORCH-4.2).

Proves the lesson override joins the SAME text source ``build_shots`` uses for
the static caption map (``lesson_script`` paragraphs, keyed by paragraph id ==
each shot's line_id) to the ``lesson_narrate`` clips, and stays inert unless the
deliverable opted into ``captions_mode=burned_in``.
"""

from unittest.mock import AsyncMock
from uuid import uuid4

from src.services.runs.run_context import RunContext
from src.services.stages.lesson.compile import LessonCompileStage
from src.services.word_timed_captions import ClipRef


def _ctx(
    captions_mode: str,
    *,
    paragraphs: list[dict[str, str]],
    clips: list[dict[str, object]],
) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"deliverable": {"format": "16x9", "captions_mode": captions_mode}},
        prior_run_outputs={
            "lesson_script": {"paragraphs": paragraphs},
            "lesson_narrate": {"clips": clips},
        },
    )


def test_gate_off_returns_empty() -> None:
    ctx = _ctx(
        "none",
        paragraphs=[{"id": "p1", "text": "Hello world"}],
        clips=[{"id": "c1", "paragraph_id": "p1", "seconds": 30.0}],
    )
    assert LessonCompileStage(client=AsyncMock()).caption_clip_refs(ctx) == {}


def test_burned_in_maps_script_text_by_paragraph_id() -> None:
    ctx = _ctx(
        "burned_in",
        paragraphs=[{"id": "p1", "text": "Hello world"}],
        clips=[{"id": "c1", "paragraph_id": "p1", "seconds": 30.0}],
    )
    result = LessonCompileStage(client=AsyncMock()).caption_clip_refs(ctx)

    assert set(result) == {"p1"}
    ref = result["p1"]
    assert isinstance(ref, ClipRef)
    # Caption surface is the SCRIPT text, never anything on the clip.
    assert ref.text == "Hello world"
    assert ref.narrate_stage == "lesson_narrate"
    assert ref.clip_filename == "c1.wav"
    assert ref.seconds == 30.0
    # Keyed by paragraph_id proves the paragraph_id == line_id mapping.
    assert ref.line_id == "p1"


def test_clip_with_no_script_text_is_skipped() -> None:
    ctx = _ctx(
        "burned_in",
        paragraphs=[{"id": "p1", "text": "Hello world"}],
        clips=[
            {"id": "c1", "paragraph_id": "p1", "seconds": 30.0},
            # p9 has no matching paragraph in lesson_script → no ClipRef.
            {"id": "c9", "paragraph_id": "p9", "seconds": 12.0},
        ],
    )
    result = LessonCompileStage(client=AsyncMock()).caption_clip_refs(ctx)

    assert set(result) == {"p1"}
    assert "p9" not in result
