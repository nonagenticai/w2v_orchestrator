from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.lesson.slides import LessonSlidesStage


def _ctx(paragraphs: list[dict], brand: dict | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brand": brand or {}},
        prior_run_outputs={
            "lesson_script": {
                "paragraphs": paragraphs,
                "total_seconds": 70.0,
                "learning_objective": "x",
            },
        },
    )


@pytest.mark.asyncio
async def test_slides_groups_every_three_paragraphs() -> None:
    paragraphs = [
        {"id": f"p{i}", "text": f"Paragraph {i} body.", "seconds": 10.0} for i in range(7)
    ]
    stage = LessonSlidesStage()

    out = await stage.run({}, ctx=_ctx(paragraphs))

    assert len(out.slides) == 3
    assert all(s.id for s in out.slides)
    assert out.slides[0].paragraph_ids == ["p0", "p1", "p2"]
    assert out.slides[2].paragraph_ids == ["p6"]
    assert "Paragraph 0" in out.slides[0].title
    # one bullet per grouped paragraph — EXCEPT the lead, whose only sentence is
    # already the title (it is never repeated as a bullet)
    assert out.slides[0].bullets == ["Paragraph 1 body.", "Paragraph 2 body."]
    # one image prompt per slide, keyed by slide id for ai_image_gen
    assert len(out.prompts) == 3
    assert out.prompts[0].line_id == out.slides[0].id
    assert out.prompts[0].prompt == out.slides[0].image_prompt
    assert "no text" in out.slides[0].image_prompt.lower()
    # the topic/title must NOT appear in the image prompt — feeding it made the
    # image model render garbled title text into the picture.
    assert "Paragraph 0" not in out.slides[0].image_prompt
    # images vary per slide (different abstract motif) so a deck isn't 3 clones
    assert out.slides[0].image_prompt != out.slides[1].image_prompt


@pytest.mark.asyncio
async def test_title_and_bullets_trim_on_word_boundary_not_mid_word() -> None:
    """Regression: titles/bullets used to be sliced at a fixed char count,
    producing mid-word cuts like '...intelligence. T'. Now they trim to the
    first sentence / whole words."""
    long_lead = (
        "Welcome to this brief overview of artificial intelligence and the way "
        "modern systems actually reason about the world around them today. "
        "A second sentence that must not appear in the title."
    )
    paragraphs = [{"id": "p0", "text": long_lead, "seconds": 12.0}]
    stage = LessonSlidesStage()

    out = await stage.run({}, ctx=_ctx(paragraphs))

    title = out.slides[0].title
    assert len(title) <= 72
    assert not title.endswith("T")  # the old mid-word artifact
    assert "second sentence" not in title  # first sentence only
    # whole words only: no partial trailing token
    assert long_lead.startswith(title) or title in long_lead.split(". ")[0]


@pytest.mark.asyncio
async def test_brand_palette_flows_into_image_prompt() -> None:
    paragraphs = [{"id": "p0", "text": "Intro.", "seconds": 5.0}]
    stage = LessonSlidesStage()

    out = await stage.run({}, ctx=_ctx(paragraphs, brand={"primary_color": "#0B1F3A"}))

    assert "#0B1F3A" in out.slides[0].image_prompt


@pytest.mark.asyncio
async def test_slides_raises_without_script() -> None:
    stage = LessonSlidesStage()
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    with pytest.raises(RuntimeError, match="lesson_script"):
        await stage.run({}, ctx=ctx)


_ONE_PARAGRAPH = (
    "Photosynthesis turns light into chemical energy. Chlorophyll absorbs red and "
    "blue light. Water is split and oxygen is released. The Calvin cycle fixes "
    "carbon dioxide into sugar. Plants store that sugar as starch."
)


@pytest.mark.asyncio
async def test_a_one_paragraph_lesson_never_repeats_its_title_as_the_only_bullet() -> None:
    """Run 01a156a8-978b (2026-10-20): one 183-word paragraph rendered one slide whose
    only bullet was its own title. The bullets now come from the paragraph's LATER
    sentences, in narration order."""
    out = await LessonSlidesStage().run(
        {}, ctx=_ctx([{"id": "p0", "text": _ONE_PARAGRAPH, "seconds": 60.0}])
    )

    (slide,) = out.slides
    assert slide.title == "Photosynthesis turns light into chemical energy."
    assert slide.title not in slide.bullets
    assert slide.bullets == [
        "Chlorophyll absorbs red and blue light.",
        "Water is split and oxygen is released.",
        "The Calvin cycle fixes carbon dioxide into sugar.",
        "Plants store that sugar as starch.",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("count", "expected"),
    [(1, [1])],
)
async def test_a_short_lesson_gets_several_slides(count: int, expected: list[int]) -> None:
    """A flat 3-per-slide gave a 60s lesson of 2-3 paragraphs ONE slide for the film."""
    paragraphs = [
        {"id": f"p{i}", "text": f"Point {i} matters. Because {i}.", "seconds": 15.0}
        for i in range(count)
    ]

    out = await LessonSlidesStage().run({}, ctx=_ctx(paragraphs))

    assert [len(s.paragraph_ids) for s in out.slides] == expected
    for slide in out.slides:
        assert slide.title not in slide.bullets
