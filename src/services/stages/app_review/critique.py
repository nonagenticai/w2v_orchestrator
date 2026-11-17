"""app_critique — pros/cons/recommendations from inspect facts."""

from __future__ import annotations

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
from src.services.stages.upstream import guarded

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# WM worker name (regex ^[a-zA-Z0-9_-]+$); provider/model live on the WM worker
# config, not here. WM returns ``resp.output`` already parsed to the schema.
_WORKER_NAME = "app-critique"

_SYSTEM_PROMPT = (
    "You critique an app/product from inspection facts, producing pros, cons, "
    "recommendations and an overall take. Return only JSON matching the schema."
)

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pros": {"type": "array", "items": {"type": "string"}},
        "cons": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "overall_take": {"type": "string"},
    },
}


# Reasoning-model output budget (see stages/llm_budget.py — a FLOOR, not a cap).
# ``{pros[], cons[], recommendations[], overall_take}`` — three short lists of
# sentence-length strings plus one prose paragraph. Their length is driven by the
# reviewer's judgement, not by the input size ⇒ MEDIUM.
_CRITIQUE_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)


class AppCritiqueInput(StageInput):
    pass


class CritiqueItem(StageOutput):
    id: str
    kind: str  # "pro" | "con" | "recommendation"
    text: str


class AppCritiqueOutput(StageOutput):
    pros: list[CritiqueItem]
    cons: list[CritiqueItem]
    recommendations: list[CritiqueItem]
    overall_take: str


class AppCritiqueStage(StageHandler[AppCritiqueInput, AppCritiqueOutput]):
    name = "app_critique"
    input_model = AppCritiqueInput
    output_model = AppCritiqueOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    async def handle(
        self,
        payload: AppCritiqueInput,
        ctx: RunContext,
    ) -> AppCritiqueOutput:
        inspect = ctx.prior_output("app_inspect")
        if not inspect:
            raise RuntimeError("app_inspect output missing")
        notes = ctx.annotations_for("app_critique")
        notes_block = (
            "\n\nApply reviewer notes:\n" + "\n".join(f"- {n['text']}" for n in notes)
            if notes
            else ""
        )
        brief = ctx.config_snapshot.get("brief") or {}
        prompt = (
            f"Critique this app/product through the lens of "
            f'"{brief.get("review_lens", "general usability")}".\n'
            f"Audience: {brief.get('audience', 'general')}.\n"
            f"Facts: {inspect['facts']}\n"
            f"Competitors: {inspect.get('competitors')}\n"
            "Output JSON: "
            '{"pros":[...], "cons":[...], "recommendations":[...], "overall_take":"..."}.'
            f"{notes_block}"
        )
        resp = await stage_call(
            downshift_reasoning(
                lambda think: guarded(
                    lambda: self._wm.message(
                        enable_reasoning=think,
                        worker_name=_WORKER_NAME,
                        message=prompt,
                        system_prompt=_SYSTEM_PROMPT,
                        output_schema=_OUTPUT_SCHEMA,
                        max_tokens=_CRITIQUE_MAX_TOKENS,
                    )
                )
            ),
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            retry_on=LLM_RETRY_ON,
            stage_name="app_critique",
        )
        data = resp.output if isinstance(resp.output, dict) else {}
        return AppCritiqueOutput(
            pros=[
                CritiqueItem(id=str(uuid7()), kind="pro", text=str(t))
                for t in (data.get("pros") or [])
            ],
            cons=[
                CritiqueItem(id=str(uuid7()), kind="con", text=str(t))
                for t in (data.get("cons") or [])
            ],
            recommendations=[
                CritiqueItem(id=str(uuid7()), kind="recommendation", text=str(t))
                for t in (data.get("recommendations") or [])
            ],
            overall_take=str(data.get("overall_take", "—")),
        )
