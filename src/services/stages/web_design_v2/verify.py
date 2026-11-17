"""web_design_v2.verify — surface the verification report as an explicit gate.

html_render already fails a design that doesn't pass its verification lane (so a
broken design never reaches here). This stage makes the gate a first-class,
visible run step: it re-asserts the design passed and surfaces the structured
report for the UI's Verification tab. Two ways it fails loud:

* the report is present but not ``passed`` (defense-in-depth), and
* the report is ABSENT (``skipped``/None) — web_design_v2 always drives
  html_render in ``mode="assembly"``, so a verified result is expected; a missing
  one means the verify lane silently didn't run (e.g. an html_render rollback to
  a pre-verification image), and the "only verified designs complete" guarantee
  must not be satisfied by an unverified design. Gated on
  ``web_design_require_verification`` (default ON) so a degraded verify lane can
  be consciously tolerated via the gitops ConfigMap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import get_settings
from src.services.stages.base import StageError, StageHandler, StageInput, StageOutput

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


class VerifyInput(StageInput):
    pass


class VerifyOutput(StageOutput):
    status: str
    hard_findings: list[dict] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    report: dict | None = None


class WebDesignVerifyStage(StageHandler[VerifyInput, VerifyOutput]):
    name = "verify"
    input_model = VerifyInput
    output_model = VerifyOutput

    async def handle(self, payload: VerifyInput, ctx: RunContext) -> VerifyOutput:
        design = ctx.prior_output("design") or {}
        status = design.get("verification_status")
        report = design.get("verification_report")

        # When the service ran verification, status must be "passed" to proceed.
        # (A failed design would already have failed the design stage; this guards
        # the case where a report is present but not passing.)
        if status is not None and status != "passed" and status != "skipped":
            hard = [f for f in (report or {}).get("findings", []) if f.get("severity") == "hard"]
            summary = "; ".join(f"[{f.get('check')}] {f.get('message')}" for f in hard)
            raise StageError(f"design failed verification: {summary or status}")

        # Absent verification (None/"skipped"): the assembly verify lane didn't
        # run. Refuse to complete an unverified design unless explicitly allowed —
        # "only verified designs complete".
        if (
            status is None or status == "skipped"
        ) and get_settings().web_design_require_verification:
            raise StageError(
                "web_design_v2 design produced no verification result "
                "(verification lane did not run); refusing to complete an unverified "
                "design. Set W2V_WEB_DESIGN_REQUIRE_VERIFICATION=false to tolerate."
            )

        hard_findings = [
            f for f in (report or {}).get("findings", []) if f.get("severity") == "hard"
        ]
        return VerifyOutput(status=status or "skipped", hard_findings=hard_findings, report=report)
