"""lesson_narrate — TTS each paragraph; collect audio-clip artifact URLs.

One sync ``TTSClient.synthesize`` per paragraph (via the shared
NarrationStageBase), bytes persisted via ``store_tts_clip`` so downstream stages
can reference a stable ``artifact_url`` keyed by ``paragraph_id``.

In the opt-in Socratic two-voice mode (``brief.dialogue.enabled``) each paragraph
is cast to its speaker's voice — student "a" vs instructor "b" — via the shared
``voice_for_speaker`` helper. When the flag is off every paragraph uses the single
``voice.voice_id`` and the inert ``speaker`` field is ignored (behaviour identical
to single-narrator lessons).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from src.services.stages.base import ArtifactUrl, SlotUse, StageInput, StageOutput
from src.services.stages.primitives.dialogue import voice_for_speaker
from src.services.stages.primitives.narration import NarrationStageBase

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# Socratic two-voice cast: student "a" in a warmer/lighter voice (af_heart),
# instructor "b" in a calm authoritative voice (am_michael). Distinct gender keeps
# the two unmistakably separate by ear. Overridable via config.voice.cast. The ids
# coincide with the shared promo default, but the role mapping is lesson-specific
# (a=student, b=instructor), so it lives here as a named constant.
_DEFAULT_CAST: dict[str, str] = {"a": "af_heart", "b": "am_michael"}


class LessonNarrateInput(StageInput):
    pass


class LessonAudioClip(StageOutput):
    id: str
    paragraph_id: str
    artifact_url: ArtifactUrl
    seconds: float
    # Speed the kept clip was synthesized at (None = backend default) and its
    # spoken word count, filled by NarrationStageBase. Optional so outputs
    # persisted before these fields existed still validate.
    speed: float | None = None
    words: int | None = None


class LessonNarrateOutput(StageOutput):
    clips: list[LessonAudioClip]


class LessonNarrateStage(NarrationStageBase[LessonNarrateInput, LessonNarrateOutput]):
    name = "lesson_narrate"
    input_model = LessonNarrateInput
    output_model = LessonNarrateOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("tts"),)
    source_stage = "lesson_script"
    source_list_key = "paragraphs"
    source_id_field = "paragraph_id"
    clip_model = LessonAudioClip

    def voice_for(self, item: dict, voice: dict, ctx: RunContext) -> str | None:
        brief = ctx.config_snapshot.get("brief") or {}
        return voice_for_speaker(item, voice, brief, default_cast=_DEFAULT_CAST)
