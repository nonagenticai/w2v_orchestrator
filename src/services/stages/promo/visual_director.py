"""promo_visual_director — map each footage narration line to the demo screen it describes.

The promo's narration is a two-hander Q&A walking through a live product demo.
For the video to *follow* the narration, every footage line must show the exact
screen it talks about ("see this leaderboard" → the leaderboard screenshot). This
stage produces that mapping: ``{line_id → screen_id}`` over the discrete demo
screens captured by ``site_capture`` (per-screen mode).

The LLM does the semantic match (line text → best screen by label); a
deterministic post-pass then enforces the structural rules the LLM can't be
trusted with (mirrors the speaker-derivation lesson in ``promo/script.py``):

* only *answer* lines switch the screen — the answerer is the one pointing at it
  ("look at this leaderboard"), so a *question* line (asker "a") holds the
  previously visible screen and the switch lands on the answer, not before it;
* any line the LLM left unmapped carries the previous line's screen (or the first
  screen), so every footage line always resolves to a real screen.

``promo_render`` consumes this to key each footage line's screenshot. Soft by
design: with no screens, no footage lines, or an LLM failure it returns an empty/
best-effort mapping and render falls back to its slide universal-fallback.
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
from src.services.stages.shared.shotlist import normalize_shotlist

# Lifted into the shared module so website_review's visual director uses the same
# structural sync rules + LLM-match half; ``resolve_assignments`` is re-imported
# here under its original private name to keep promo (and its tests) identical.
from src.services.stages.shared.visual_sync import (
    resolve_assignments as _resolve_assignments,
)
from src.services.stages.shared.visual_sync import (
    run_assignment_llm,
)

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

_SYSTEM_PROMPT = (
    "You are a video editor syncing narration to a product demo. For each "
    "narration line you are given the list of demo screens (id + description). "
    "Return, for each line, the id of the ONE screen that line is most clearly "
    "about — what should be on screen while that line is spoken. Return only JSON "
    "matching the schema."
)


class PromoVisualDirectorInput(StageInput):
    pass


class LineScreen(StageOutput):
    id: str
    line_id: str
    screen_id: str


class PromoVisualDirectorOutput(StageOutput):
    assignments: list[LineScreen]
    notes: list[str] = []  # noqa: RUF012 — pydantic copies field defaults per-instance


class PromoVisualDirectorStage(StageHandler[PromoVisualDirectorInput, PromoVisualDirectorOutput]):
    name = "promo_visual_director"
    input_model = PromoVisualDirectorInput
    output_model = PromoVisualDirectorOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    @staticmethod
    def _footage_lines(ctx: RunContext) -> list[dict[str, Any]]:
        """The (line_id, text, speaker) of each line the storyboard plans as footage."""
        script = ctx.prior_output("promo_script") or {}
        story = ctx.prior_output("promo_storyboard") or {}
        by_line = {str(ln["id"]): ln for ln in script.get("lines", [])}
        out: list[dict[str, Any]] = []
        for shot in story.get("shots", []):
            if str(shot.get("visual_type")) == "footage":
                lid = str(shot.get("line_id"))
                ln = by_line.get(lid)
                if ln:
                    out.append(
                        {
                            "line_id": lid,
                            "text": str(ln.get("text") or ""),
                            "speaker": str(ln.get("speaker") or "a"),
                        }
                    )
        return out

    @staticmethod
    def _screens(ctx: RunContext) -> list[dict[str, str]]:
        capture = ctx.prior_output("site_capture") or {}
        return [
            {"id": str(s["id"]), "label": str(s.get("label") or s["id"])}
            for s in capture.get("screens", [])
            if s.get("id")
        ]

    async def handle(
        self,
        payload: PromoVisualDirectorInput,
        ctx: RunContext,
    ) -> PromoVisualDirectorOutput:
        # Authored shotlist: the binding is deterministic — site_capture used each
        # URL shot's line id as the screen capture_id, so screen_id == line_id.
        # Skip the LLM entirely; map each URL shot to its own screen 1:1.
        brief = ctx.config_snapshot.get("brief") or {}
        shots = normalize_shotlist(brief)
        if shots:
            assignments = [
                LineScreen(id=str(uuid7()), line_id=s.line_id, screen_id=s.line_id)
                for s in shots
                if s.kind == "url"
            ]
            # No note: ``notes`` is the DEGRADED channel (``is_degraded``), so an
            # informational "deterministic shotlist binding" line turned every
            # shotlist run's visual director into a user-facing "Unclassified note"
            # warning (prod run 01a17e6f) for a stage that did exactly its job.
            return PromoVisualDirectorOutput(assignments=assignments)

        footage_lines = self._footage_lines(ctx)
        screens = self._screens(ctx)
        if not footage_lines or not screens:
            return PromoVisualDirectorOutput(
                assignments=[],
                notes=["no footage lines or no captured screens to map"],
            )

        screen_ids = [s["id"] for s in screens]
        valid_lines = {ln["line_id"] for ln in footage_lines}
        screens_block = "\n".join(f'- screen_id {s["id"]}: "{s["label"]}"' for s in screens)
        lines_block = "\n".join(
            f'- line_id {ln["line_id"]}: "{ln["text"]}"' for ln in footage_lines
        )
        message = (
            "Match each narration line to the demo screen it is most about.\n\n"
            f"Demo screens:\n{screens_block}\n\nNarration lines:\n{lines_block}"
        )

        notes: list[str] = []
        llm_map = await run_assignment_llm(
            self._wm,
            worker_name="promo-visual-director",
            message=message,
            system_prompt=_SYSTEM_PROMPT,
            target_key="screen_id",
            valid_lines=valid_lines,
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            stage_name=self.name,
            notes=notes,
        )

        resolved = _resolve_assignments(footage_lines, screen_ids, llm_map)
        assignments = [
            LineScreen(id=str(uuid7()), line_id=lid, screen_id=sid) for lid, sid in resolved.items()
        ]
        return PromoVisualDirectorOutput(assignments=assignments, notes=notes)
