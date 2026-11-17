"""Storyboard draft — maps each script line to a shot AND an image prompt.

Pure-deterministic projection over the prior `script_draft` output; no
downstream client call. The constructor still accepts a WMClient for
forward-compatibility with a planned LLM-backed storyboarding mode.

``prompts`` is the per-line source ``ai_image_gen`` reads
(``prompts_from_stage: "storyboard_draft"``, seeded for every generic run by
``run_setup.build_config_snapshot``), so each shot gets a generated still keyed
to its own narration line. Before 2026-10-20 nothing fed ``ai_image_gen`` on a
generic run: it skipped, and every shot rendered its ``description`` — then the
literal placeholder ``"Visual for: <line>"`` — as a text card (live run
``01a156a4-e906``). ``description`` is now the narration line itself, i.e. an
honest text card when no image is available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from uuid import uuid7

from pydantic import Field

from src.services.llm import WMClient
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
)

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


class StoryboardDraftInput(StageInput):
    pass


class StoryboardShot(StageOutput):
    id: str
    line_id: str
    description: str
    seconds: float


class StoryboardImagePrompt(StageOutput):
    """One per-line image prompt, the shape ``ai_image_gen._resolve_prompts`` reads."""

    line_id: str
    prompt: str


def image_prompt_for_line(text: str, *, title: str | None) -> str:
    """A wordless still for one narration line, anchored to the film's subject.

    Deterministic (same line + title → same prompt → same ``deterministic_media_id``),
    so a resumed run reuses its stored image instead of paying ACG again.
    """
    subject = f' for a video titled "{title}"' if title else ""
    return (
        f"Editorial illustration{subject}, depicting: {text.strip()} "
        "Cinematic composition, rich detail, no text, no lettering, no captions."
    )


class StoryboardDraftOutput(StageOutputWithNotes):
    # `shots` IS the render spine — `video_render` subscripts `story["shots"]`
    # and `assemble_shots` iterates it. Empty makes `video_render` raise
    # StagePreconditionError one stage LATER, where the operator sees "no shots
    # to compile" on the compile stage and nothing points back here.
    produces: ClassVar[tuple[str, ...]] = ("shots",)

    shots: list[StoryboardShot]
    #: Per-line image prompts for ``ai_image_gen`` (one per shot, keyed by line_id).
    prompts: list[StoryboardImagePrompt] = Field(default_factory=list)


class StoryboardDraftStage(StageHandler[StoryboardDraftInput, StoryboardDraftOutput]):
    name = "storyboard_draft"
    input_model = StoryboardDraftInput
    output_model = StoryboardDraftOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)

    def __init__(self, client: WMClient | None = None) -> None:
        self._wm = client or WMClient()

    async def handle(
        self,
        payload: StoryboardDraftInput,
        ctx: RunContext,
    ) -> StoryboardDraftOutput:
        script = ctx.prior_output("script_draft")
        if not script:
            raise RuntimeError("script_draft output missing")
        title = ctx.config_snapshot.get("title")
        shots: list[StoryboardShot] = []
        prompts: list[StoryboardImagePrompt] = []
        for line in script["lines"]:
            text = str(line["text"])
            shots.append(
                StoryboardShot(
                    id=str(uuid7()),
                    line_id=line["id"],
                    description=text,
                    seconds=line["seconds"],
                )
            )
            if text.strip():
                prompts.append(
                    StoryboardImagePrompt(
                        line_id=str(line["id"]),
                        prompt=image_prompt_for_line(text, title=str(title) if title else None),
                    )
                )
        return StoryboardDraftOutput(shots=shots, prompts=prompts)
