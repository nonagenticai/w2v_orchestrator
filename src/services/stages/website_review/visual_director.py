"""website_review_visual_director — map each AFTER-act line to the redesign section it discusses.

The after act presents the redesign. Instead of panning the whole redesign page
top→bottom on every line (the unwatchable scroll), html_render now captures a
discrete still per landmark section (``redesign_section:<name>`` visual assets) and
this stage maps each after-act narration line to the section it's about — so when
the designer says "the new chat composer", the composer section is on screen.

Mirrors ``promo_visual_director`` but scoped to the AFTER act only (the BEFORE act
keeps its proven deterministic live-screen keying inside compile). The LLM does the
semantic line→section match; the shared ``resolve_assignments`` then enforces the
structural rule — only *answer* lines (the designer pointing at a section) switch
the visual, questions hold the prior one. Soft by design: with no section assets,
no after lines, or an LLM failure it returns a best-effort/empty mapping and compile
falls back to the full-page scroll.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid7

from src.services.llm import WMClient
from src.services.stages.base import SlotUse, StageHandler, StageInput, StageOutput
from src.services.stages.llm_budget import (
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
)
from src.services.stages.shared.video_gate import require_video
from src.services.stages.shared.visual_sync import resolve_assignments, run_assignment_llm
from src.services.stages.website_review.grounding import rendered_section_visuals

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# A redesign-section visual carries this role prefix (html_render
# _build_visual_assets → role=f"redesign_section:{name}").
_SECTION_ROLE_PREFIX = "redesign_section:"
_SYSTEM_PROMPT = (
    "You are a video editor syncing a designer's walkthrough of a REDESIGNED web "
    "page to screenshots of that page's sections. You get the sections in "
    "top-to-bottom page order (id + heading) and the narration lines in order. For "
    "each line return the id of the ONE section that should be on screen while it is "
    "spoken.\n"
    "Rules:\n"
    "- A walkthrough moves across the page, so DISTRIBUTE the lines across DIFFERENT "
    "sections — do NOT map most lines to a single section. Advance to a new section "
    "as the narration turns to a new topic.\n"
    "- A line about a UI element maps to the section that CONTAINS it: a search bar "
    "or navigation to the top header/hero section; a chat widget, message box or "
    "contact form to the contact section; project or work cards to the projects/work "
    "section; site links or legal text to the footer.\n"
    "- A general remark ('take a look at the new redesign') keeps the current section "
    "or uses the top one.\n"
    "Return only JSON matching the schema."
)


class WebsiteReviewVisualDirectorInput(StageInput):
    pass


class LineSection(StageOutput):
    id: str
    line_id: str
    section_id: str


class WebsiteReviewVisualDirectorOutput(StageOutput):
    assignments: list[LineSection]
    notes: list[str] = []  # noqa: RUF012 — pydantic copies field defaults per-instance


class WebsiteReviewVisualDirectorStage(
    StageHandler[WebsiteReviewVisualDirectorInput, WebsiteReviewVisualDirectorOutput]
):
    name = "website_review_visual_director"
    input_model = WebsiteReviewVisualDirectorInput
    output_model = WebsiteReviewVisualDirectorOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    @staticmethod
    def _after_lines(ctx: RunContext) -> list[dict[str, Any]]:
        """The (line_id, text, speaker, section_id) of each AFTER-act narration line."""
        script = ctx.prior_output("website_review_script") or {}
        out: list[dict[str, Any]] = []
        for ln in script.get("lines", []):
            if str(ln.get("phase") or "before").strip().lower() != "after":
                continue
            if not ln.get("id"):
                continue
            out.append(
                {
                    "line_id": str(ln["id"]),
                    "text": str(ln.get("text") or ""),
                    "speaker": str(ln.get("speaker") or "a"),
                    "section_id": str(ln.get("section_id") or ""),
                }
            )
        return out

    @staticmethod
    def _sections(ctx: RunContext) -> list[dict[str, str]]:
        """The discrete redesign-section visuals (id + label) to map lines onto.

        Ids are the grounding inventory's (page order, a repeated visual id renamed
        ``-2``) — the same ids the script authored ``section_id`` against.
        """
        redesign = ctx.prior_output("website_redesign") or {}
        return [
            {"id": uid, "label": str(v.get("label") or uid)}
            for uid, v in rendered_section_visuals(redesign)
        ]

    async def handle(
        self,
        payload: WebsiteReviewVisualDirectorInput,
        ctx: RunContext,
    ) -> WebsiteReviewVisualDirectorOutput:
        require_video(ctx)  # video-only: no after-act to sync on a critique-only run
        after_lines = self._after_lines(ctx)
        sections = self._sections(ctx)
        if not after_lines or not sections:
            return WebsiteReviewVisualDirectorOutput(
                assignments=[],
                notes=["no after-act lines or no redesign-section assets to map"],
            )

        section_ids = [s["id"] for s in sections]
        # The script already names the rendered section each after-act line is about
        # (it was prompted with the section inventory and its ids are validated
        # against the rendered sections). Honour it: re-guessing a line's section from
        # its text is how one of nine sections got shown for a whole after act.
        authored = {
            ln["line_id"]: ln["section_id"] for ln in after_lines if ln["section_id"] in section_ids
        }
        # Emitted AS IS: a renamed duplicate (``sec-2``) is the only handle that
        # tells two sections sharing a visual id apart, and compile keys its stills
        # by the same inventory ids (grounding.rendered_section_visuals).
        if len(authored) == len(after_lines):
            return WebsiteReviewVisualDirectorOutput(
                assignments=[
                    LineSection(
                        id=str(uuid7()),
                        line_id=ln["line_id"],
                        section_id=authored[ln["line_id"]],
                    )
                    for ln in after_lines
                ],
                notes=[],
            )
        valid_lines = {ln["line_id"] for ln in after_lines}
        # Number the sections so the LLM sees their top-to-bottom page order.
        sections_block = "\n".join(
            f'{i + 1}. section_id {s["id"]}: "{s["label"]}"' for i, s in enumerate(sections)
        )
        lines_block = "\n".join(f'- line_id {ln["line_id"]}: "{ln["text"]}"' for ln in after_lines)
        message = (
            "Sync each narration line to the redesign section it discusses. The "
            "sections are listed in top-to-bottom page order; distribute the lines "
            "across different sections as the walkthrough moves down the page.\n\n"
            f"Redesign sections (top to bottom):\n{sections_block}\n\n"
            f"Narration lines:\n{lines_block}"
        )

        notes: list[str] = []
        llm_map = await run_assignment_llm(
            self._wm,
            worker_name="website-review-visual-director",
            message=message,
            system_prompt=_SYSTEM_PROMPT,
            target_key="section_id",
            valid_lines=valid_lines,
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            stage_name=self.name,
            notes=notes,
        )

        resolved = resolve_assignments(
            after_lines, section_ids, {**llm_map, **authored}, spread_on_collapse=True
        )
        # An authored section is the script's own statement — it wins over the
        # question-holds rule and the anti-collapse spread for ITS line.
        resolved.update(authored)
        assignments = [
            LineSection(id=str(uuid7()), line_id=lid, section_id=sid)
            for lid, sid in resolved.items()
        ]
        return WebsiteReviewVisualDirectorOutput(assignments=assignments, notes=notes)
