"""promo_sound_director — LLM picks per-line accent SFX cues for the promo.

A thin subclass of the shared :class:`SoundDirectorStage`: the whole LLM /
validation / clamp / opt-in-skip core lives in
``src.services.stages.shared.sound_director``; this class only supplies the promo
source (``promo_script`` lines) and the WM worker name. ``promo_render`` threads
the resulting per-line ``cues`` into the compositor shots (resolved to presigned
sound-library URLs at compile time).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.services.stages.base import SlotUse, StagePreconditionError
from src.services.stages.shared.sound_director import (
    DIRECTOR_LINE_CHARS,
    LineSfxCue,
    SoundDirectorInput,
    SoundDirectorOutput,
    SoundDirectorStage,
)

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# Back-compat aliases (importers / tests reference these names).
PromoSoundDirectorInput = SoundDirectorInput
PromoSoundDirectorOutput = SoundDirectorOutput

__all__ = [
    "LineSfxCue",
    "PromoSoundDirectorInput",
    "PromoSoundDirectorOutput",
    "PromoSoundDirectorStage",
]


class PromoSoundDirectorStage(SoundDirectorStage):
    name = "promo_sound_director"
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)
    worker_name = "promo-sound-director"

    def director_lines(self, ctx: RunContext) -> list[dict[str, Any]]:
        script = ctx.prior_output("promo_script")
        if not script or not script.get("lines"):
            raise StagePreconditionError("promo_sound_director: promo_script output missing")
        return [
            {
                "line_id": str(line["id"]),
                "text": str(line.get("text") or "")[:DIRECTOR_LINE_CHARS],
                "seconds": float(line.get("seconds") or 0.0),
            }
            for line in script["lines"]
        ]
