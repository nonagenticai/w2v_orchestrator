"""promo_footage_direction — LLM footage shot-planner for the promo pipeline.

A thin subclass of the shared :class:`FootageDirectionStage`, mirroring
``demo_video_footage_direction`` but reading promo's data sources. It plans over
the FOOTAGE lines only (``promo_storyboard`` shots whose ``visual_type`` is
``"footage"``): the line TEXT from ``promo_script`` (keyed by ``id``), the real
per-line DURATIONS from ``promo_narrate`` (``clips[].seconds``), and — critically —
whether the line's bound demo screen captured a real click target
(``site_capture`` screen ``focus_track``).

``has_click`` is derived from the ``promo_visual_director`` ``assignments``
line→screen map (``line_id → screen_id``) fed through ``footage_tracks_by_line``:
a line has a click only when its screen carries a non-empty focus track. This
omits ``promo_render``'s extra ``screen_id in still_by_screen`` filter (render-only
display state); the difference is inert — a line whose screen has a focus track but
no fetchable still is dropped from render's footage shots, so any plan for it never
reaches the compositor. ``promo_render`` threads the resulting per-line ``plans``
into the compositor shots.

Opt-in + skip-clean is owned by the shared base: with no ``direction.footage``
config block the base ``handle`` raises ``StageSkippedError`` before
``planner_lines`` is ever called, so ``promo_render``'s dependency on this stage
never blocks a run that didn't opt in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from src.services.stages.base import SlotUse, StagePreconditionError
from src.services.stages.shared.footage_direction import (
    DIRECTOR_LINE_CHARS,
    FootageDirectionStage,
    PlannerLine,
)
from src.services.stages.shot_assembly import footage_tracks_by_line, normalize_focus_track

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


class PromoFootageDirectionStage(FootageDirectionStage):
    name = "promo_footage_direction"
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)
    worker_name = "footage-director"

    def planner_lines(self, ctx: RunContext) -> list[PlannerLine]:
        story = ctx.prior_output("promo_storyboard") or {}
        shots = story.get("shots") or []
        if not shots:
            raise StagePreconditionError("promo_footage_direction: promo_storyboard output missing")
        # Line text from the script (keyed by id), with the shot's own text as a
        # fallback for a footage line the script no longer carries.
        script = ctx.prior_output("promo_script") or {}
        text_by_line = {
            str(ln["id"]): str(ln.get("text") or "")
            for ln in script.get("lines", [])
            if ln.get("id")
        }
        # Real narrated durations when available, else the authored fallback.
        narrate = ctx.prior_output("promo_narrate") or {}
        seconds_by_line = {
            str(c["line_id"]): float(c.get("seconds") or 0.0)
            for c in narrate.get("clips", [])
            if c.get("line_id")
        }
        # The SAME line→screen map promo_render builds (render.py:496-500), fed to
        # footage_tracks_by_line so has_click is derived exactly as compile does.
        director = ctx.prior_output("promo_visual_director") or {}
        screen_by_line = {
            str(a["line_id"]): str(a["screen_id"])
            for a in director.get("assignments", [])
            if a.get("line_id") and a.get("screen_id")
        }
        capture = ctx.prior_output("site_capture") or {}
        focus_by_line, _cursor_by_line, viewport_by_line = footage_tracks_by_line(
            capture.get("screens", []), screen_by_line, footage=True
        )
        out: list[PlannerLine] = []
        for shot in shots:
            if str(shot.get("visual_type") or "") != "footage":
                continue
            line_id = str(shot["line_id"])
            text = text_by_line.get(line_id, str(shot.get("text") or ""))
            out.append(
                {
                    "line_id": line_id,
                    "text": text[:DIRECTOR_LINE_CHARS],
                    "seconds": seconds_by_line.get(line_id, float(shot.get("seconds") or 0.0)),
                    # A screen whose captured focus track holds an INTERACTED,
                    # visible element ⇒ the compiler has a bbox to zoom/callout at.
                    "has_click": bool(
                        normalize_focus_track(
                            focus_by_line.get(line_id),
                            viewport=viewport_by_line.get(line_id),
                        )
                    ),
                }
            )
        return out
