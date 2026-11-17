"""lesson_slides — group paragraphs into branded slides (title + bullets + a
per-slide on-brand image prompt).

Pure-deterministic: groups up to ~3 paragraphs into one slide (fewer per slide on
a short lesson, so a 60s lesson still gets several slides — see
:func:`_paragraphs_per_slide`), derives a clean slide title and bullet points from
the paragraph text on **whole-word / whole-sentence** boundaries (never a mid-word
cut, which is what produced "…intelligence. T" titles), and emits one
image-generation prompt per slide.

The title is the lead paragraph's first sentence, so that sentence is NEVER also a
bullet (run 01a156a8-978b, 2026-10-20: a one-paragraph lesson rendered one slide
whose only bullet was its own title). A slide with fewer than
:data:`_MIN_BULLETS` paragraph bullets is topped up from the following sentences
of its own paragraphs, in narration order.

The ``prompts`` list (keyed by slide id as ``line_id``) is what ``ai_image_gen``
consumes when the run config sets
``ai_image_gen.prompts_from_stage = "lesson_slides"`` — so each slide gets its
own hero image, which ``lesson_record`` then composites into the slide PNG.
No LLM call, so no ``stage_call`` wrapper is needed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any
from uuid import uuid7

from src.services.stages.base import StageHandler, StageInput, StageOutput

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

_PARAS_PER_SLIDE = 3
#: A slide whose paragraphs yield fewer bullets than this is topped up from the
#: later sentences of those same paragraphs, up to :data:`_MAX_BULLETS`.
_MIN_BULLETS = 2
_MAX_BULLETS = 4
_TITLE_MAX = 72
_BULLET_MAX = 140
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _normalize(text: str) -> str:
    return " ".join(str(text).split())


def _sentences(text: str) -> list[str]:
    text = _normalize(text)
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()] if text else []


def _paragraphs_per_slide(count: int) -> int:
    """How many paragraphs share a slide: 1 up to three paragraphs, then up to 3.

    A flat 3-per-slide gave a short lesson (60s, 2-4 paragraphs) ONE slide held for
    the whole film. ``min(3, ceil(n / 3))`` keeps the long-lesson grouping (7 → 3,3,1)
    and gives a short one a slide per paragraph or two.
    """
    if count <= 3:
        return 1
    return min(_PARAS_PER_SLIDE, -(-count // 3))


def _same_text(a: str, b: str) -> bool:
    def key(value: str) -> str:
        return re.sub(r"[^\w]+", " ", value.casefold()).strip()

    return key(a) == key(b)


def _slide_text(group: Sequence[Mapping[str, Any]]) -> tuple[str, list[str]]:
    """``(title, bullets)`` for one slide — the title sentence is never a bullet."""
    lead = group[0]["text"]
    per_paragraph = [_sentences(p["text"]) for p in group]
    title = _clip_words(per_paragraph[0][0] if per_paragraph[0] else lead, _TITLE_MAX)
    # The lead's first sentence IS the title; its bullet is its NEXT sentence.
    per_paragraph[0] = per_paragraph[0][1:]
    bullets = [sentences[0] for sentences in per_paragraph if sentences]
    if len(bullets) < _MIN_BULLETS:
        # Too few paragraphs to make a list: use the slide's sentences in order.
        bullets = [s for sentences in per_paragraph for s in sentences][:_MAX_BULLETS]
    clipped: list[str] = []
    for bullet in bullets:
        text = _clip_words(bullet, _BULLET_MAX)
        if text and not _same_text(text, title) and not any(_same_text(text, c) for c in clipped):
            clipped.append(text)
    return title, clipped


def _clip_words(text: str, limit: int) -> str:
    """Trim to ``limit`` chars on a word boundary — never mid-word."""
    text = _normalize(text)
    if len(text) <= limit:
        return text
    head = text[:limit].rsplit(" ", 1)[0].rstrip(",;:-—")
    return head or text[:limit]


# Purely abstract motifs cycled per slide so a deck's images vary without ever
# describing the topic. Feeding the slide *title* to the image model made it
# render that title as (garbled) text into the picture — so the prompt carries
# NO content phrase at all, only an on-brand abstract texture.
_ABSTRACT_MOTIFS = (
    "flowing translucent waves",
    "soft layered gradients",
    "gentle geometric facets",
    "drifting points of light",
    "smooth rippling silk",
    "calm bokeh light orbs",
)


def _slide_image_prompt(brand: dict[str, Any], index: int) -> str:
    palette = ", ".join(
        c
        for c in (
            brand.get("primary_color"),
            brand.get("secondary_color"),
            brand.get("accent_color"),
        )
        if c
    )
    palette_hint = f" Brand palette: {palette}." if palette else ""
    motif = _ABSTRACT_MOTIFS[index % len(_ABSTRACT_MOTIFS)]
    return (
        f"A purely abstract, wordless background of {motif}. Cinematic soft "
        f"lighting, shallow depth of field, smooth gradients, elegant and minimal."
        f"{palette_hint} No subject, no people, no text, no words, no letters, "
        f"no numbers, no logos, no UI."
    )


class LessonSlidesInput(StageInput):
    pass


class LessonSlide(StageOutput):
    id: str
    paragraph_ids: list[str]
    title: str
    bullets: list[str]
    image_prompt: str


class LessonSlideImagePrompt(StageOutput):
    id: str
    line_id: str  # the slide id — ai_image_gen keys each image back to its slide
    prompt: str


class LessonSlidesOutput(StageOutput):
    slides: list[LessonSlide]
    prompts: list[LessonSlideImagePrompt]


class LessonSlidesStage(StageHandler[LessonSlidesInput, LessonSlidesOutput]):
    name = "lesson_slides"
    input_model = LessonSlidesInput
    output_model = LessonSlidesOutput

    async def handle(
        self,
        payload: LessonSlidesInput,
        ctx: RunContext,
    ) -> LessonSlidesOutput:
        script = ctx.prior_output("lesson_script")
        if not script:
            raise RuntimeError("lesson_script output missing")
        paragraphs = script["paragraphs"]
        brand = ctx.config_snapshot.get("brand") or {}

        slides: list[LessonSlide] = []
        prompts: list[LessonSlideImagePrompt] = []
        per_slide = _paragraphs_per_slide(len(paragraphs))
        groups = [paragraphs[i : i + per_slide] for i in range(0, len(paragraphs), per_slide)]
        for index, grp in enumerate(groups):
            slide_id = str(uuid7())
            title, bullets = _slide_text(grp)
            image_prompt = _slide_image_prompt(brand, index)
            slides.append(
                LessonSlide(
                    id=slide_id,
                    paragraph_ids=[p["id"] for p in grp],
                    title=title,
                    bullets=bullets,
                    image_prompt=image_prompt,
                )
            )
            prompts.append(
                LessonSlideImagePrompt(id=str(uuid7()), line_id=slide_id, prompt=image_prompt)
            )
        return LessonSlidesOutput(slides=slides, prompts=prompts)
