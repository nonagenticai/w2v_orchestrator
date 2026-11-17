"""demo_video_sound_director — LLM accent-cue director for the showcase/demo_video.

A thin subclass of the shared :class:`SoundDirectorStage`. Unlike the promo
director (whose ``promo_script`` lines already carry estimated durations), the
authored demo_video lines carry no timing until narration is synthesised — so
this reads the ordered narration TEXT from ``demo_video_gen`` and the real
per-line DURATIONS from ``demo_video_narrate`` (``clips[].seconds``), giving the
LLM accurate line lengths to place accents against. ``demo_video_compile`` threads
the resulting per-line ``cues`` into the compositor shots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.services.stages.base import SlotUse, StagePreconditionError
from src.services.stages.demo_video.spine import resolve_plan_output
from src.services.stages.shared.sound_director import DIRECTOR_LINE_CHARS, SoundDirectorStage

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


class DemoVideoSoundDirectorStage(SoundDirectorStage):
    name = "demo_video_sound_director"
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)
    worker_name = "demo-sound-director"

    def director_lines(self, ctx: RunContext) -> list[dict[str, Any]]:
        # Doctored spine when script_doctor ran, else demo_video_gen (skip-clean).
        gen = resolve_plan_output(ctx)
        if not gen or not gen.get("lines"):
            raise StagePreconditionError("demo_video_sound_director: demo_video_gen output missing")
        narrate = ctx.prior_output("demo_video_narrate") or {}
        seconds_by_line = {
            str(clip["line_id"]): float(clip.get("seconds") or 0.0)
            for clip in narrate.get("clips", [])
            if clip.get("line_id")
        }
        return [
            {
                "line_id": str(line["id"]),
                "text": str(line.get("text") or "")[:DIRECTOR_LINE_CHARS],
                # Real narrated duration when available, else the authored fallback.
                "seconds": seconds_by_line.get(str(line["id"]), float(line.get("seconds") or 0.0)),
            }
            for line in gen["lines"]
        ]
