from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageError
from src.services.stages.lesson.script import LessonScriptParagraph, LessonScriptStage
from src.services.stages.primitives.dialogue import assign_dialogue_speakers


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


@pytest.mark.asyncio
async def test_script_returns_paragraphs_with_ids() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "paragraphs": [
                {"text": "Open with the problem.", "seconds": 30.0},
                {"text": "Show the diagnostic.", "seconds": 60.0},
            ]
        }
    )
    stage = LessonScriptStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {
                "duration_minutes": 6,
                "learning_objective": "Diagnose churn drivers.",
            },
            "voice": {"descriptor": "calm"},
        },
    )

    out = await stage.run({}, ctx=ctx)

    assert len(out.paragraphs) == 2
    assert all(p.id for p in out.paragraphs)
    assert out.total_seconds == pytest.approx(90.0)
    assert out.learning_objective == "Diagnose churn drivers."
    # Regression (E2E 2026-06-02): lesson uses a FRESH worker name, not the
    # base-default "lesson-script" (poisoned to a lines shape; lesson needs
    # paragraphs).
    _, kwargs = fake.message.call_args
    assert kwargs["worker_name"] == "lesson-script-v2"
    assert kwargs["output_schema"]["properties"]["paragraphs"]["items"]["required"] == [
        "text",
        "seconds",
        "speaker",
    ]


@pytest.mark.asyncio
async def test_script_raises_on_malformed_llm() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(None)
    stage = LessonScriptStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"duration_minutes": 2, "learning_objective": "x"},
            "voice": {"descriptor": "v"},
        },
    )

    with pytest.raises(StageError):
        await stage.run({}, ctx=ctx)


@pytest.mark.asyncio
async def test_script_includes_reviewer_notes_in_prompt() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp({"paragraphs": [{"text": "x", "seconds": 1.0}]})
    stage = LessonScriptStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"duration_minutes": 2, "learning_objective": "x"},
            "voice": {"descriptor": "v"},
        },
        open_annotations_by_stage={
            "lesson_script": [{"text": "more concrete examples"}],
        },
    )

    await stage.run({}, ctx=ctx)
    _, kwargs = fake.message.call_args
    assert "more concrete examples" in kwargs["message"]


@pytest.mark.asyncio
async def test_script_grounds_in_source_ingest_when_present() -> None:
    # When the lesson (or its parent course, propagated per-child) is grounded in
    # extra sources, the script prompt carries the SOURCE MATERIAL block.
    fake = AsyncMock()
    fake.message.return_value = _resp({"paragraphs": [{"text": "x", "seconds": 1.0}]})
    stage = LessonScriptStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"duration_minutes": 2, "learning_objective": "x"},
            "voice": {"descriptor": "v"},
        },
        prior_run_outputs={
            "source_ingest": {"grounding_text": "Churn drivers: pricing, onboarding."}
        },
    )

    await stage.run({}, ctx=ctx)
    _, kwargs = fake.message.call_args
    assert "SOURCE MATERIAL" in kwargs["message"]
    assert "Churn drivers: pricing, onboarding." in kwargs["message"]


@pytest.mark.asyncio
async def test_script_prompt_unchanged_without_source_ingest() -> None:
    # Plain-run-safe: no source_ingest output → no grounding block appended.
    fake = AsyncMock()
    fake.message.return_value = _resp({"paragraphs": [{"text": "x", "seconds": 1.0}]})
    stage = LessonScriptStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"duration_minutes": 2, "learning_objective": "x"},
            "voice": {"descriptor": "v"},
        },
    )

    await stage.run({}, ctx=ctx)
    _, kwargs = fake.message.call_args
    assert "SOURCE MATERIAL" not in kwargs["message"]


def test_assign_dialogue_speakers_by_question_role() -> None:
    """Questions → student 'a'; everything else → instructor 'b'."""
    paragraphs = [
        LessonScriptParagraph(id="1", text="What is gradient descent?", seconds=2.0),
        LessonScriptParagraph(id="2", text="It is an optimization method.", seconds=4.0),
        LessonScriptParagraph(id="3", text="Why does the step size matter?", seconds=2.0),
        LessonScriptParagraph(id="4", text="Too large a step overshoots.", seconds=4.0),
    ]
    assert [p.speaker for p in assign_dialogue_speakers(paragraphs)] == ["a", "b", "a", "b"]


def test_assign_dialogue_speakers_parity_fallback() -> None:
    """No question marks → fall back to strict a/b parity so two voices still appear."""
    paragraphs = [
        LessonScriptParagraph(id="1", text="First teaching point.", seconds=4.0),
        LessonScriptParagraph(id="2", text="Second teaching point.", seconds=4.0),
    ]
    assert [p.speaker for p in assign_dialogue_speakers(paragraphs)] == ["a", "b"]


@pytest.mark.asyncio
async def test_script_dialogue_mode_prompts_socratic_and_rederives_speakers() -> None:
    """brief.dialogue.enabled → Socratic prompt + speakers re-derived from role.

    The LLM mis-tags every paragraph 'a' (the known gotcha); the output must be
    corrected to alternate student-question / instructor-answer by role.
    """
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "paragraphs": [
                {"text": "What is overfitting?", "seconds": 2.0, "speaker": "a"},
                {"text": "It is memorizing noise.", "seconds": 5.0, "speaker": "a"},
            ]
        }
    )
    stage = LessonScriptStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {
                "duration_minutes": 6,
                "learning_objective": "Understand overfitting.",
                "dialogue": {"enabled": True, "personas": {"a": "Robin the learner"}},
            },
            "voice": {"descriptor": "calm"},
        },
    )

    out = await stage.run({}, ctx=ctx)

    _, kwargs = fake.message.call_args
    prompt = kwargs["message"]
    assert "SOCRATIC" in prompt
    assert "student" in prompt and "instructor" in prompt
    assert "Robin the learner" in prompt  # custom persona flows in
    # Speaker re-derived from question-role despite the LLM tagging both "a".
    assert [p.speaker for p in out.paragraphs] == ["a", "b"]


@pytest.mark.asyncio
async def test_script_single_narrator_prompt_pins_instructor() -> None:
    """No dialogue flag → single-narrator prompt, speaker pinned to instructor 'b'."""
    fake = AsyncMock()
    fake.message.return_value = _resp({"paragraphs": [{"text": "Teach.", "seconds": 4.0}]})
    stage = LessonScriptStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"duration_minutes": 6, "learning_objective": "x"},
            "voice": {"descriptor": "calm"},
        },
    )

    out = await stage.run({}, ctx=ctx)

    _, kwargs = fake.message.call_args
    prompt = kwargs["message"]
    assert "SINGLE narrator" in prompt
    assert "SOCRATIC" not in prompt
    assert all(p.speaker == "b" for p in out.paragraphs)


@pytest.mark.asyncio
async def test_lesson_source_script_dialogue_rederives_speakers() -> None:
    """Verbatim source-script bypass still re-derives speakers when dialogue is on."""
    fake = AsyncMock()
    stage = LessonScriptStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {
                "learning_objective": "Learn X",
                "source_script": "What is X?\n\nX is a thing.",
                "dialogue": {"enabled": True},
            },
            "voice": {"descriptor": "calm"},
        },
    )

    out = await stage.run({}, ctx=ctx)

    assert [p.speaker for p in out.paragraphs] == ["a", "b"]
    fake.message.assert_not_called()


@pytest.mark.asyncio
async def test_lesson_source_script_passes_through_verbatim() -> None:
    fake = AsyncMock()
    stage = LessonScriptStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {
                "learning_objective": "Learn X",
                "source_script": "First block.\n\nSecond block.",
            },
            "voice": {"descriptor": "calm"},
        },
    )

    out = await stage.run({}, ctx=ctx)

    assert [p.text for p in out.paragraphs] == ["First block.", "Second block."]
    assert out.learning_objective == "Learn X"
    fake.message.assert_not_called()


@pytest.mark.asyncio
async def test_lesson_script_verbatim_present_succeeds_and_injects_prompt() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"paragraphs": [{"text": "Remember: drink water daily.", "seconds": 5.0}]}
    )
    stage = LessonScriptStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"duration_minutes": 2, "learning_objective": "Understand X."},
            "voice": {"descriptor": "calm"},
            "verbatim_items": [{"text": "drink water"}],
        },
    )

    out = await stage.run({}, ctx=ctx)

    assert out.paragraphs[0].text.startswith("Remember")
    _, kwargs = fake.message.call_args
    assert "drink water" in kwargs["message"]


@pytest.mark.asyncio
async def test_lesson_script_verbatim_missing_raises() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"paragraphs": [{"text": "Some unrelated content.", "seconds": 5.0}]}
    )
    stage = LessonScriptStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"duration_minutes": 2, "learning_objective": "Understand X."},
            "voice": {"descriptor": "calm"},
            "verbatim_items": [{"text": "drink water"}],
        },
    )

    with pytest.raises(StageError):
        await stage.run({}, ctx=ctx)
