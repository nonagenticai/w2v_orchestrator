"""Brief intake — pure transform of project brief into a typed normalized form.

No downstream client calls. Reads the brief from RunContext.config_snapshot,
returns a structured BriefIntakeOutput. Other stages depend on this output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.stages.base import StageHandler, StageInput, StageOutput
from src.services.stages.shared.duration_conformance import generic_ordered_seconds

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


class BriefIntakeInput(StageInput):
    pass  # All inputs come from ctx.config_snapshot.


class BriefIntakeOutput(StageOutput):
    title: str
    angle: str
    audience: str
    target_duration_seconds: int
    voice_descriptor: str
    source_script: str | None = None


class BriefIntakeStage(StageHandler[BriefIntakeInput, BriefIntakeOutput]):
    name = "brief_intake"
    input_model = BriefIntakeInput
    output_model = BriefIntakeOutput

    async def handle(
        self,
        payload: BriefIntakeInput,
        ctx: RunContext,
    ) -> BriefIntakeOutput:
        cfg = ctx.config_snapshot
        brief = cfg.get("brief") or {}
        voice = cfg.get("voice") or {}
        return BriefIntakeOutput(
            title=cfg.get("title", "Untitled"),
            angle=brief.get("angle", "informational"),
            audience=brief.get("audience", "general"),
            # brief → deliverable → 60s. Reading only ``brief`` asked the script
            # model for a 60s film on a 30s DELIVERABLE order (where the wizard
            # writes the length) — and ``video_render`` grades against this ladder.
            target_duration_seconds=round(generic_ordered_seconds(cfg)),
            voice_descriptor=voice.get("descriptor", "neutral"),
            source_script=(brief.get("source_script") or None),
        )
