"""web_design_v2.assemble — the terminal deliverable summary.

Gathers the verified design into the run's final output: the live/interactive
preview URL, the generated HTML + downloadable PDF, representative screenshots,
and the verification report. Field names mirror the redesign viewer's contract
(``redesign_job_id`` / ``redesigned_html_path`` / ``visuals``) so the frontend
run-view renders it with the existing viewer + download wiring, plus the new
live-preview + verification affordances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.stages.base import StageError, StageHandler, StageInput, StageOutput
from src.services.stages.website_review.redesign import RedesignVisual

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


class AssembleInput(StageInput):
    pass


class AssembleOutput(StageOutput):
    redesign_job_id: str
    redesigned_html_path: str | None = None
    live_html_url: str
    pdf_url: str
    interactivity: str = "static"
    accessibility_score: float | None = None
    verification_status: str | None = None
    verification_report: dict | None = None
    visuals: list[RedesignVisual] = []  # noqa: RUF012 — pydantic copies field defaults per-instance


class WebDesignAssembleStage(StageHandler[AssembleInput, AssembleOutput]):
    name = "assemble"
    input_model = AssembleInput
    output_model = AssembleOutput

    async def handle(self, payload: AssembleInput, ctx: RunContext) -> AssembleOutput:
        design = ctx.prior_output("design") or {}
        job_id = design.get("redesign_job_id")
        if not job_id:
            raise StageError("assemble: no design job id from the design stage")
        verify = ctx.prior_output("verify") or {}
        return AssembleOutput(
            redesign_job_id=str(job_id),
            redesigned_html_path=design.get("redesigned_html_path"),
            live_html_url=str(design.get("live_html_url") or f"/api/redesign/jobs/{job_id}/html"),
            pdf_url=f"/api/redesign/jobs/{job_id}/pdf",
            interactivity=str(design.get("interactivity") or "static"),
            accessibility_score=design.get("accessibility_score"),
            verification_status=verify.get("status") or design.get("verification_status"),
            verification_report=design.get("verification_report"),
            visuals=[RedesignVisual.model_validate(v) for v in design.get("visuals") or []],
        )
