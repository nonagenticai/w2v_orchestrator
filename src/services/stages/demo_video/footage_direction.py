"""demo_video_footage_direction — LLM footage shot-planner for demo_video/showcase.

A thin subclass of the shared :class:`FootageDirectionStage`. Reads the ordered
line TEXT from the spine (``resolve_plan_output`` — the doctored/reconciled output
when those stages ran, else ``demo_video_gen``), the real per-line DURATIONS from
``demo_video_narrate`` (``clips[].seconds``), and — critically — whether each
line's recording captured a real click target from ``demo_video_record``
(``clips[].focus_track``). A line binds its clip the SAME way ``demo_video_compile``
does (``line.get("clip") or line["id"]``), so several dialogue lines sharing one
footage beat all inherit that clip's has-click signal. ``demo_video_compile`` then
threads the resulting per-line ``plans`` into the compositor shots — turning off a
zoom / dropping a cursor / adding a callout — resolving the callout bbox from the
same captured focus track (never from the plan).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from src.services.stages.base import SlotUse, StagePreconditionError
from src.services.stages.demo_video.spine import resolve_plan_output
from src.services.stages.shared.footage_direction import (
    DIRECTOR_LINE_CHARS,
    FootageDirectionStage,
    PlannerLine,
)
from src.services.stages.shot_assembly import normalize_focus_track, normalize_footage_viewport

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


class DemoVideoFootageDirectionStage(FootageDirectionStage):
    name = "demo_video_footage_direction"
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)
    worker_name = "footage-director"

    def planner_lines(self, ctx: RunContext) -> list[PlannerLine]:
        # Doctored/reconciled spine when those stages ran, else demo_video_gen.
        gen = resolve_plan_output(ctx)
        if not gen or not gen.get("lines"):
            raise StagePreconditionError(
                "demo_video_footage_direction: demo_video_gen output missing"
            )
        # A line has a real zoom/callout target only when its bound recording
        # clip captured a focus track (a click bbox over time). Keyed by clip id,
        # so the has-click signal fans onto every line bound to that clip.
        record = ctx.prior_output("demo_video_record") or {}
        clips = record.get("clips", [])
        # A target is an INTERACTED, visible element (`interacted_bbox`), not merely a
        # non-empty track: a track of zero-filled / off-screen keyframes has nothing
        # to zoom at, so its line must plan zoom OFF like a no-click line.
        focus_clips = {
            str(c["line_id"])
            for c in clips
            if normalize_focus_track(
                c.get("focus_track"), viewport=normalize_footage_viewport(c.get("viewport"))
            )
        }
        # Real narrated durations when available, else the authored fallback.
        narrate = ctx.prior_output("demo_video_narrate") or {}
        seconds_by_line = {
            str(c["line_id"]): float(c.get("seconds") or 0.0)
            for c in narrate.get("clips", [])
            if c.get("line_id")
        }
        out: list[PlannerLine] = []
        for line in gen["lines"]:
            # An author-pinned split bypasses the LLM planner for this line: it is
            # applied at compile (`pin_authored_splits`) and the director never sees it.
            if isinstance(line.get("split"), dict) and line["split"].get("second_clip"):
                continue
            key = str(line.get("clip") or line["id"])
            line_id = str(line["id"])
            out.append(
                {
                    "line_id": line_id,
                    "text": str(line.get("text") or "")[:DIRECTOR_LINE_CHARS],
                    "seconds": seconds_by_line.get(line_id, float(line.get("seconds") or 0.0)),
                    "has_click": key in focus_clips,
                }
            )
        return out
