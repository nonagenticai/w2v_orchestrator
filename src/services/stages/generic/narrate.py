"""narrate — TTS each script line for the generic blueprint.

One sync ``TTSClient.synthesize`` per line (via the shared NarrationStageBase),
bytes persisted via ``store_tts_clip`` so ``video_render`` can mux a narration
track keyed by ``line_id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from src.services.stages.base import (
    ArtifactUrl,
    SlotUse,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
)
from src.services.stages.primitives.narration import NarrationStageBase

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


class NarrateInput(StageInput):
    pass


class GenericAudioClip(StageOutput):
    id: str
    line_id: str
    artifact_url: ArtifactUrl
    seconds: float
    # Speed the kept clip was synthesized at (None = backend default) and its
    # spoken word count, filled by NarrationStageBase. Optional so outputs
    # persisted before these fields existed still validate.
    speed: float | None = None
    words: int | None = None


class NarrateOutput(StageOutputWithNotes):
    # `clips` is the audio spine: `shot_assembly` keys `audio_by_line` off
    # `clip["line_id"]` and takes each shot's duration from the clip, and
    # `captions.py` builds the SRT from it. Zero clips renders a SILENT video
    # with storyboard-estimated durations and no captions — and still completes.
    # Not opt-in, and `script_draft` guarantees >=1 line, so empty is never a
    # legitimate outcome here.
    produces: ClassVar[tuple[str, ...]] = ("clips",)

    clips: list[GenericAudioClip]


class NarrateStage(NarrationStageBase[NarrateInput, NarrateOutput]):
    name = "narrate"
    input_model = NarrateInput
    output_model = NarrateOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("tts"),)
    source_stage = "script_draft"
    source_list_key = "lines"
    source_id_field = "line_id"
    clip_model = GenericAudioClip

    def speed_for(self, item: dict, ctx: RunContext) -> float | None:
        """The speed ``script_draft``'s per-line fit MEASURED this line at.

        ``deterministic_clip_id`` hashes voice + speed + text, so requesting the
        persisted speed is what makes narrate replay the measured clip instead of
        synthesizing a different one — measured == shipped. Absent / ``None`` (an
        unfitted line, or an output persisted before the field existed) omits speed
        exactly as before. Anything outside tts_backend's legal 0.5-2.0 is ignored
        rather than sent and 4xx'd.
        """
        raw = item.get("speed")
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            return None
        speed = float(raw)
        return speed if 0.5 <= speed <= 2.0 else None
