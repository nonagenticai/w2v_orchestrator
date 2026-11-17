"""AuditScriptStageBase — narrated voiceover over an audit scorecard result.

An opt-in (``brief.generate_video``) LLM script stage: turns the deterministic
``<type>_scan`` output (overall score + grade + quick-wins + strategic findings +
per-breakdown scores) into a short spoken walkthrough — a score headline, the top
quick-wins to fix, and a closing CTA. Off by default (``require_audit_video``
skips it), so a plain scan-only audit run is unchanged.

Reuses ``LLMScriptStageBase`` (the shared WM-worker script tail); only the prompt
varies per audit type, and that variance is the class-level config below
(``breakdown_key`` / ``item_key`` / labels), set by the per-type subclass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from src.services.stages.audit_video.gate import require_audit_video
from src.services.stages.audit_video.models import (
    AuditScriptInput,
    AuditScriptLine,
    AuditScriptOutput,
)
from src.services.stages.base import StageOutput
from src.services.stages.primitives.script_stage import LLMScriptStageBase

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


class AuditScriptStageBase(LLMScriptStageBase[AuditScriptInput, AuditScriptOutput]):
    """Shared audit-script stage. Per-type subclass sets ``name`` + the config below."""

    input_model = AuditScriptInput
    output_model = AuditScriptOutput
    element_key = "lines"
    min_elements = 1

    #: Scan-output key holding the per-category breakdown list (``groups`` for SEO,
    #: ``subsections`` for CRO, …) and the per-item category key inside it.
    breakdown_key: ClassVar[str]
    item_key: ClassVar[str]
    #: Prompt labels. ``walkthrough_noun`` → "…-audit walkthrough"; ``score_label``
    #: → "Overall <score_label>: …"; ``headline_metric`` → "the overall <metric>
    #: and grade"; ``quick_win_rationale`` → "what it is and <rationale>".
    walkthrough_noun: ClassVar[str]
    score_label: ClassVar[str]
    headline_metric: ClassVar[str]
    quick_win_rationale: ClassVar[str]

    async def handle(self, payload: AuditScriptInput, ctx: RunContext) -> AuditScriptOutput:
        # Off by default: a scan-only audit skips the whole video sub-chain.
        require_audit_video(ctx)
        return await super().handle(payload, ctx)

    def build_prompt(self, ctx: RunContext) -> str:
        prefix = self.name.removesuffix("_script")
        scan = ctx.prior_output(f"{prefix}_scan")
        if not scan:
            raise RuntimeError(f"{prefix}_scan output required for the audit script")
        voice = ctx.config_snapshot.get("voice") or {}
        url = str(scan.get("url") or "the site")
        overall = scan.get("overall_score")
        grade = str(scan.get("grade") or "")
        quick_wins = [
            {"title": str(w.get("title") or ""), "severity": str(w.get("severity") or "")}
            for w in (scan.get("quick_wins") or [])[:6]
        ]
        strategic = [str(s.get("title") or "") for s in (scan.get("strategic") or [])[:4]]
        breakdown = [
            {self.item_key: str(g.get(self.item_key) or ""), "score": g.get("score")}
            for g in (scan.get(self.breakdown_key) or [])
        ]
        notes = ctx.annotations_for(f"{prefix}_script")
        notes_block = (
            "\n\nApply reviewer notes:\n" + "\n".join(f"- {n['text']}" for n in notes)
            if notes
            else ""
        )
        return (
            f"Write a short spoken {self.walkthrough_noun} walkthrough for {url}. "
            f"Voice/tone: {voice.get('descriptor', 'clear and encouraging')}.\n"
            f"Overall {self.score_label}: {overall} (grade {grade}).\n"
            f"Per-{self.item_key} scores: {breakdown}\n"
            f"Top quick-wins (title, severity): {quick_wins}\n"
            f"Strategic recommendations: {strategic}\n"
            "Structure the narration as 6-10 short lines:\n"
            f"1. A headline line stating the overall {self.headline_metric} and grade.\n"
            f"2. One line per top quick-win: what it is and {self.quick_win_rationale}.\n"
            "3. A closing call-to-action line encouraging the fixes.\n"
            "Keep each line conversational and under ~25 words. Set `seconds` to a "
            "realistic spoken duration for each line.\n"
            'Output JSON: {"lines":[{"text":"...","seconds":<num>}, ...]}.'
            f"{notes_block}"
        )

    def make_element(self, raw: dict, *, element_id: str) -> AuditScriptLine:
        return AuditScriptLine(
            id=element_id,
            text=str(raw["text"]),
            seconds=float(raw["seconds"]),
        )

    def build_output(self, elements: list[StageOutput], ctx: RunContext) -> AuditScriptOutput:
        lines = cast("list[AuditScriptLine]", elements)
        return AuditScriptOutput(
            lines=lines,
            total_seconds=sum(line.seconds for line in lines),
        )
