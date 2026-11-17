"""AuditCompileStageBase — composite audit slide images + narration into a video.

Opt-in (``brief.generate_video``) and TERMINAL: emits one shot per narrated audit
line (the still image of the slide that line belongs to, with the line's narration
muxed over it), and the video-compile service stitches the shots into a single
narrated MP4. Mirrors ``lesson_compile``'s line→slide→image join minus the
lesson-specific real-media / motion / ai-video cutaway overlays (an audit deck is
plain narrated slides). Skips cleanly when the video variant is off — the run then
completes on the ``<type>_scan`` output (the scan-only deliverable).

Per-type subclass sets ``name`` only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.stages.audit_video.gate import require_audit_video
from src.services.stages.audit_video.models import AuditCompileInput, AuditCompileOutput
from src.services.stages.primitives.compile_stage import CompileStageBase
from src.services.stages.shot_assembly import assemble_shots

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


class AuditCompileStageBase(CompileStageBase[AuditCompileInput, AuditCompileOutput]):
    """Shared audit-compile stage. Per-type subclass sets ``name`` only."""

    input_model = AuditCompileInput
    output_model = AuditCompileOutput
    no_shots_message = "no narrated shots to compile"

    async def handle(self, payload: AuditCompileInput, ctx: RunContext) -> AuditCompileOutput:
        require_audit_video(ctx)
        return await super().handle(payload, ctx)

    def build_shots(self, ctx: RunContext) -> list[dict[str, Any]]:
        prefix = self.name.removesuffix("_compile")
        slides = ctx.prior_output(f"{prefix}_slides")
        narrate = ctx.prior_output(f"{prefix}_narrate")
        record = ctx.prior_output(f"{prefix}_record")
        if not (slides and narrate and record):
            raise RuntimeError(f"{prefix}_slides + {prefix}_narrate + {prefix}_record required")

        image_by_slide = {rec["slide_id"]: rec["image_url"] for rec in record["recordings"]}
        slide_by_line = {
            line_id: slide["id"] for slide in slides["slides"] for line_id in slide["line_ids"]
        }
        # The line→slide→image two-hop expressed as a keyed image join.
        image_by_line: dict[str, str] = {}
        for clip in narrate["clips"]:
            line_id = clip["line_id"]
            slide_id = slide_by_line.get(line_id)
            image_url = image_by_slide.get(slide_id) if slide_id else None
            if image_url:
                image_by_line[line_id] = image_url

        # Spine = narrated lines; each clip is its own audio (joined by the line id
        # used as line_id). No storyboard, so no `description`.
        storyboard_shots = [
            {"id": c["id"], "line_id": c["line_id"], "seconds": float(c["seconds"] or 0.0)}
            for c in narrate["clips"]
        ]
        return assemble_shots(
            storyboard_shots=storyboard_shots,
            images=[],
            narration_clips=[
                {
                    "line_id": c["line_id"],
                    "artifact_url": c["artifact_url"],
                    "seconds": c["seconds"],
                }
                for c in narrate["clips"]
            ],
            image_by_line=image_by_line,
        )

    def build_output(self, result: dict[str, Any], ctx: RunContext) -> AuditCompileOutput:
        return AuditCompileOutput(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=float(result["duration_seconds"]),
        )
