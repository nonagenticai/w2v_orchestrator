"""promo_image_prompts — per-line, story-grounded AI image prompts.

Reads ``promo_script`` + ``promo_storyboard`` and, for each line the storyboard
plans as an ``ai_image`` accent, asks the LLM (via worker_management) for one
short, concrete image prompt that visually evokes THAT line's meaning — so the
generated accents illustrate the story instead of generic atmospheric stock.

Output is keyed by ``line_id``. ``ai_image_gen`` consumes it when the run config
sets ``ai_image_gen.prompts_from_stage = "promo_image_prompts"``, and
``promo_render`` keys each image to its narration line.

Soft by design: if there are no ai_image shots, or the LLM call fails, the stage
returns an empty list and ``ai_image_gen`` falls back to its static
``prompts`` — the run still renders, just with generic accents.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid7

from w2v_common.stages import stage_call

from src.services.llm import WMClient
from src.services.stages.base import SlotUse, StageHandler, StageInput, StageOutput
from src.services.stages.llm_budget import (
    LLM_RETRY_ON,
    MEDIUM_ANSWER_TOKENS,
    downshift_reasoning,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.shared.motion_slides import motion_per_line_requested
from src.services.stages.upstream import guarded

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

_BRAND_HINT = (
    "Style: cinematic, editorial, abstract-but-evocative. Brand palette deep "
    "navy #0B1F3A, teal #2BB6A3, warm orange #F26B3A. No text, no logos, no UI "
    "chrome."
)
#: Appended to every generated-image prompt of a run that opted in to per-line
#: motion (``direction.motion == "per_line"``), whatever the model wrote. The
#: instruction in ``_BRAND_HINT`` only reaches the prompt-writing model; the
#: image model sees the prompt alone, and run 01a11082 (2026-10-06) shipped a
#: still with garbled letters ("PLIC") baked into it. A run without the flag
#: sends the model's prompt verbatim, as before.
NO_TEXT_CLAUSE = "No text, no letters, no words, no numbers, no signage, no logos, no UI."


def textless(prompt: str) -> str:
    """``prompt`` ending in :data:`NO_TEXT_CLAUSE` (exactly once)."""
    body = prompt.strip()
    if body.endswith(NO_TEXT_CLAUSE):
        return body
    return f"{body.rstrip('.')}. {NO_TEXT_CLAUSE}"


_SYSTEM_PROMPT = (
    "You write concise image-generation prompts. For each narration line you are "
    "given, return exactly one vivid, concrete image prompt that visually evokes "
    "that line's meaning and mood. Return only JSON matching the schema."
)
_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line_id": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["line_id", "prompt"],
            },
        }
    },
    "required": ["prompts"],
}


# Reasoning-model output budget (see stages/llm_budget.py — a FLOOR, not a cap).
# ``{prompts[]}`` is one ~one-sentence image prompt per ai_image line. Unbatched,
# but the ai_image lines are a SUBSET of an already-short promo script, so this is a
# modest list of one-sentence strings ⇒ MEDIUM.
_PROMPT_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)


class PromoImagePromptsInput(StageInput):
    pass


class PromoImagePrompt(StageOutput):
    id: str
    line_id: str
    prompt: str


class PromoImagePromptsOutput(StageOutput):
    prompts: list[PromoImagePrompt]
    notes: list[str] = []  # noqa: RUF012 — pydantic copies field defaults per-instance


class PromoImagePromptsStage(StageHandler[PromoImagePromptsInput, PromoImagePromptsOutput]):
    name = "promo_image_prompts"
    input_model = PromoImagePromptsInput
    output_model = PromoImagePromptsOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    @staticmethod
    def _ai_image_lines(ctx: RunContext) -> list[dict[str, str]]:
        """The (line_id, text) of each line the storyboard plans as an ai_image."""
        script = ctx.prior_output("promo_script") or {}
        story = ctx.prior_output("promo_storyboard") or {}
        text_by_line = {str(ln["id"]): str(ln["text"]) for ln in script.get("lines", [])}
        out: list[dict[str, str]] = []
        for shot in story.get("shots", []):
            if str(shot.get("visual_type")) == "ai_image":
                lid = str(shot.get("line_id"))
                if lid in text_by_line:
                    out.append({"line_id": lid, "text": text_by_line[lid]})
        return out

    async def handle(
        self,
        payload: PromoImagePromptsInput,
        ctx: RunContext,
    ) -> PromoImagePromptsOutput:
        targets = self._ai_image_lines(ctx)
        if not targets:
            return PromoImagePromptsOutput(prompts=[], notes=["no ai_image lines in the plan"])

        valid_ids = {t["line_id"] for t in targets}
        lines_block = "\n".join(f'- line_id {t["line_id"]}: "{t["text"]}"' for t in targets)
        message = (
            "Write one image-generation prompt per narration line below. Each must "
            "visually capture that specific line's meaning and emotion (not a generic "
            "backdrop), in about one sentence.\n"
            f"{_BRAND_HINT}\n\nLines:\n{lines_block}"
        )

        finish = textless if motion_per_line_requested(ctx.config_snapshot) else str

        async def _attempt(enable_reasoning: bool | None) -> list[PromoImagePrompt]:
            resp = await self._wm.message(
                enable_reasoning=enable_reasoning,
                worker_name="promo-image-prompts",
                message=message,
                system_prompt=_SYSTEM_PROMPT,
                output_schema=_OUTPUT_SCHEMA,
                max_tokens=_PROMPT_MAX_TOKENS,
            )
            raw = list((resp.output or {}).get("prompts") or [])
            built = [
                PromoImagePrompt(
                    id=str(uuid7()), line_id=str(p["line_id"]), prompt=finish(str(p["prompt"]))
                )
                for p in raw
                if p.get("prompt") and str(p.get("line_id")) in valid_ids
            ]
            if not built:
                raise ValueError("promo_image_prompts: LLM returned no usable per-line prompts")
            return built

        try:
            # Soft-fail path: guarded() makes a permanent WM failure degrade to the
            # static prompts immediately instead of after the full retry ladder.
            prompts = await stage_call(
                downshift_reasoning(lambda think: guarded(lambda: _attempt(think))),
                timeout_seconds=self._llm_timeout,
                retries=self._llm_retries,
                retry_on=LLM_RETRY_ON,
                stage_name=self.name,
            )
        except Exception as exc:  # noqa: BLE001 -- soft-fail: degrade to ai_image_gen's static prompts
            log.warning("promo_image_prompts: degrading to static prompts: %s", exc)
            return PromoImagePromptsOutput(prompts=[], notes=[f"prompt generation failed: {exc}"])

        return PromoImagePromptsOutput(prompts=prompts, notes=[])
