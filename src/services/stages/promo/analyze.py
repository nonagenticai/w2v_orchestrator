"""promo_analyze — extract typed insights from the project brief.

Output exposes ``pain_points[*].id`` / ``hook_candidates[*].id`` so PR-6
can pin annotations onto individual insights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid7

from w2v_common.stages import stage_call

from src.services.llm import WMClient
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
)
from src.services.stages.generic.grounding import grounding_prompt_block
from src.services.stages.llm_budget import (
    LLM_RETRY_ON,
    downshift_reasoning,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.shared.placeholder_text import text_or_empty
from src.services.stages.upstream import guarded

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# WM worker name (regex ^[a-zA-Z0-9_-]+$); provider/model live on the WM worker
# config, not here. WM returns ``resp.output`` already parsed to the schema.
_WORKER_NAME = "promo-analyze"

_SYSTEM_PROMPT = (
    "You analyze a product-promo brief and extract pain points, hook candidates "
    "and tone hints. Return only JSON matching the schema."
)

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pain_points": {"type": "array", "items": {"type": "string"}},
        "hook_candidates": {"type": "array", "items": {"type": "string"}},
        "tone_hints": {"type": "array", "items": {"type": "string"}},
    },
}


# Worst-case size of the ANSWER alone: three short string arrays (pain points,
# hook candidates, tone hints). Observed live is ~138 tokens; 700 covers a
# verbose draw. This is deliberately NOT the whole budget — the served model is a
# reasoning model and its thinking tokens are charged against the same
# ``max_tokens``, so ``output_token_budget`` adds the reasoning headroom on top
# and applies the router's chain-deadline cap (see ``stages/llm_budget.py``).
_ANALYZE_ANSWER_TOKENS = 700
_ANALYZE_MAX_TOKENS = output_token_budget(answer_tokens=_ANALYZE_ANSWER_TOKENS)


class PromoAnalyzeInput(StageInput):
    pass


class PromoInsight(StageOutput):
    id: str
    text: str


class PromoAnalyzeOutput(StageOutputWithNotes):
    # `promo_script` subscripts both lists straight into the script prompt, and
    # `runs/elements.py` pins reviewer annotations to their ids. The handler
    # substitutes literal fallbacks ("unspecified pain", the brief title) when
    # the LLM returns nothing, so an EMPTY list can only mean drift — there is
    # no mode in which it is the intended answer.
    # `tone_hints` is deliberately NOT declared: it has zero readers in src/.
    produces: ClassVar[tuple[str, ...]] = ("pain_points", "hook_candidates")

    pain_points: list[PromoInsight]
    hook_candidates: list[PromoInsight]
    tone_hints: list[str]


class PromoAnalyzeStage(StageHandler[PromoAnalyzeInput, PromoAnalyzeOutput]):
    name = "promo_analyze"
    input_model = PromoAnalyzeInput
    output_model = PromoAnalyzeOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    async def handle(
        self,
        payload: PromoAnalyzeInput,
        ctx: RunContext,
    ) -> PromoAnalyzeOutput:
        cfg = ctx.config_snapshot
        brief = cfg.get("brief") or {}
        voice = cfg.get("voice") or {}
        prompt = self._build_prompt(cfg, brief, voice) + grounding_prompt_block(ctx)
        # ``max_tokens`` is EXPLICIT. This comment used to say the omission was
        # "DELIBERATE, not an oversight" because the observed output is ~138 tokens,
        # so WM's 6000 default "is never the binding limit". That reasoning is about
        # ANSWER SIZE ONLY and is wrong under the real mechanism: the served model is
        # a reasoning model (``--reasoning-parser qwen3``) whose THINKING tokens are
        # charged against the same budget, so a call can burn the whole budget before
        # emitting a single byte of JSON — WM then answers 422, which classifies as
        # PermanentUpstreamError (absent from LLM_RETRY_ON) and fails the stage
        # terminally on attempt 1. A small answer does not make the call safe; the
        # budget is a FLOOR, not a cap. Derivation + live measurements:
        # ``stages/llm_budget.py``.
        resp = await stage_call(
            downshift_reasoning(
                lambda think: guarded(
                    lambda: self._wm.message(
                        enable_reasoning=think,
                        worker_name=_WORKER_NAME,
                        message=prompt,
                        system_prompt=_SYSTEM_PROMPT,
                        output_schema=_OUTPUT_SCHEMA,
                        max_tokens=_ANALYZE_MAX_TOKENS,
                    )
                )
            ),
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            retry_on=LLM_RETRY_ON,
            stage_name="promo_analyze",
        )
        data = resp.output if isinstance(resp.output, dict) else {}
        return PromoAnalyzeOutput(
            pain_points=[
                PromoInsight(id=str(uuid7()), text=str(t))
                for t in (data.get("pain_points") or ["unspecified pain"])
            ],
            hook_candidates=[
                PromoInsight(id=str(uuid7()), text=str(t))
                for t in (data.get("hook_candidates") or [cfg.get("title") or "Hook"])
            ],
            tone_hints=[
                str(t) for t in (data.get("tone_hints") or [voice.get("descriptor") or "neutral"])
            ],
        )

    @staticmethod
    def _build_prompt(cfg: dict, brief: dict, voice: dict) -> str:
        return (
            "You are analyzing a brief for a PRODUCT promo video — a short commercial "
            "for a software product. Emit JSON of "
            '{"pain_points":[...], "hook_candidates":[...], "tone_hints":[...]}.\n'
            "pain_points: the concrete, specific PROBLEMS this audience has today that "
            "the product solves — phrase each as the user's real-world frustration "
            "(time/cost/bottleneck), not a generic benefit.\n"
            "hook_candidates: opening lines for the script; bias them PROBLEM-FIRST "
            "(name the pain before naming the product), short and punchy.\n"
            "tone_hints: short adjectives guiding delivery.\n"
            "If the value prop names distinct offering/capability TYPES, treat each as a "
            "concrete capability worth surfacing.\n" + _brief_facts(cfg, brief, voice)
        )


def _brief_facts(cfg: dict, brief: dict, voice: dict) -> str:
    """The brief's known facts, one per line — an ABSENT field emits no line.

    Interpolating an absent field rendered ``CTA: None``: the model reads that as
    a CTA whose text is "None", and the script stage downstream obeyed the same
    defect into a spoken "None".
    """
    facts = (
        ("Title", text_or_empty(cfg.get("title"))),
        ("Audience", text_or_empty(brief.get("audience"))),
        ("Value prop", text_or_empty(brief.get("value_proposition"))),
        ("CTA", text_or_empty(brief.get("call_to_action"))),
        ("Voice", text_or_empty(voice.get("descriptor"))),
    )
    lines = [f"{label}: {value}\n" for label, value in facts if value]
    duration = text_or_empty(brief.get("target_duration_seconds"))
    if duration:
        lines.append(f"Duration: {duration}s\n")
    return "".join(lines)
