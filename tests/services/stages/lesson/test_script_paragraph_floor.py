"""A lesson draw with too few paragraphs is split on sentence boundaries.

Run 01a156a8-978b (2026-10-20): a 60s lesson drew ONE 183-word paragraph, which
became one narration clip, one slide held 70s, and a draw the length trim could not
touch (it can only drop whole paragraphs). The prompt now asks for one paragraph
per ~20s, and ``enforce_paragraph_floor`` guarantees the floor deterministically
without changing a word.
"""

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.lesson.script import (
    LessonScriptParagraph,
    LessonScriptStage,
    enforce_paragraph_floor,
    paragraph_count_floor,
    paragraph_plan_sentence,
)

_SENTENCES = [
    "Photosynthesis turns light into chemical energy.",
    "Chlorophyll absorbs red and blue light while reflecting green.",
    "Inside the chloroplast, water is split and oxygen is released.",
    "The Calvin cycle then fixes carbon dioxide into simple sugars.",
    "Plants store that sugar as starch for the night.",
    "Without this process almost no food chain on Earth would exist.",
]
_ONE_PARAGRAPH = " ".join(_SENTENCES)


def _para(text: str, pid: str = "p0", seconds: float = 60.0) -> LessonScriptParagraph:
    return LessonScriptParagraph(id=pid, text=text, seconds=seconds)


@pytest.mark.parametrize(("seconds", "floor"), [(60.0, 2)])
def test_the_floor_is_one_paragraph_per_thirty_seconds_at_least_two(
    seconds: float, floor: int
) -> None:
    assert paragraph_count_floor(seconds) == floor


def test_a_single_paragraph_is_split_without_changing_a_word() -> None:
    out = enforce_paragraph_floor([_para(_ONE_PARAGRAPH)], 90.0)

    assert len(out) == 3
    assert " ".join(p.text for p in out) == _ONE_PARAGRAPH
    assert all(p.text for p in out)
    assert out[0].id == "p0", "annotations pinned to the drawn id stay pinned"
    assert len({p.id for p in out}) == 3
    assert sum(p.seconds for p in out) == pytest.approx(60.0)
    # balanced on words, not 1 sentence + 5
    assert max(len(p.text.split()) for p in out) <= 2 * min(len(p.text.split()) for p in out)


def test_split_ids_are_deterministic_across_a_replay() -> None:
    first = enforce_paragraph_floor([_para(_ONE_PARAGRAPH)], 90.0)
    again = enforce_paragraph_floor([_para(_ONE_PARAGRAPH)], 90.0)

    assert [p.id for p in first] == [p.id for p in again]


def test_a_draw_already_at_the_floor_is_untouched() -> None:
    paragraphs = [_para(_SENTENCES[0], "a"), _para(_SENTENCES[1], "b")]

    assert enforce_paragraph_floor(paragraphs, 60.0) == paragraphs


def test_a_single_sentence_cannot_be_split_and_is_left_alone() -> None:
    paragraphs = [_para(_SENTENCES[0])]

    assert enforce_paragraph_floor(paragraphs, 120.0) == paragraphs


def test_the_speaker_survives_the_split() -> None:
    para = LessonScriptParagraph(id="p0", text=_ONE_PARAGRAPH, seconds=60.0, speaker="a")

    assert {p.speaker for p in enforce_paragraph_floor([para], 60.0)} == {"a"}


def test_the_prompt_asks_for_a_paragraph_per_twenty_seconds() -> None:
    assert paragraph_plan_sentence(60.0, 160.0).startswith("Write 3 paragraphs")
    assert paragraph_plan_sentence(360.0, 160.0).startswith("Write 18 paragraphs")


def test_the_paragraph_plan_states_each_paragraphs_word_share() -> None:
    """60s at 160 wpm = 160 words over 3 paragraphs = ~53 each, not "2-3 sentences".

    Run 01a158a6-0f2c (2026-10-20): "2-3 sentences each" drew 20-word paragraphs
    against a 198-word budget; the share is stated in words so the two agree.
    """
    plan = paragraph_plan_sentence(60.0, 160.0)
    assert "about 53 words each" in plan
    assert "2-3 sentences" not in plan
    assert "(4-5 sentences)" in plan


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj, output_type="object", output_schema_used=None, metadata=None, usage_metrics=None
    )


@pytest.mark.asyncio
async def test_the_stage_ships_several_paragraphs_from_a_one_paragraph_draw() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"paragraphs": [{"text": _ONE_PARAGRAPH, "seconds": 60.0, "speaker": "b"}]}
    )
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"duration_minutes": 1, "learning_objective": "Photosynthesis."},
            "voice": {"descriptor": "calm"},
        },
    )

    out = await LessonScriptStage(client=fake).run({}, ctx=ctx)

    assert len(out.paragraphs) >= 2
    assert " ".join(p.text for p in out.paragraphs) == _ONE_PARAGRAPH
    _, kwargs = fake.message.call_args_list[0]
    assert "Write 3 paragraphs" in kwargs["message"]
