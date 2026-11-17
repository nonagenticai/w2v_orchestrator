"""website_review_record — TTS each review-script line.

One sync ``TTSClient.synthesize`` per line (via the shared NarrationStageBase),
bytes persisted via ``store_tts_clip`` keyed by ``line_id`` so the compile stage
can mux a narration track. When the run is a two-hander (``dialogue_cast``:
``brief.dialogue.enabled`` explicitly set AND two distinct voices resolve) each line
is cast to its speaker's voice; otherwise every line is read in ``voice.voice_id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from src.services.stages.base import ArtifactUrl, SlotUse, StageInput, StageOutput
from src.services.stages.primitives.dialogue import voice_for_speaker
from src.services.stages.primitives.narration import NarrationStageBase
from src.services.stages.shared.video_gate import require_video

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


class WebsiteReviewRecordInput(StageInput):
    pass


class ReviewAudioClip(StageOutput):
    id: str
    line_id: str
    artifact_url: ArtifactUrl
    seconds: float
    # Speed the kept clip was synthesized at (None = backend default) and its
    # spoken word count, filled by NarrationStageBase. Optional so outputs
    # persisted before these fields existed still validate.
    speed: float | None = None
    words: int | None = None


class WebsiteReviewRecordOutput(StageOutput):
    clips: list[ReviewAudioClip]


class WebsiteReviewRecordStage(
    NarrationStageBase[WebsiteReviewRecordInput, WebsiteReviewRecordOutput]
):
    name = "website_review_record"
    input_model = WebsiteReviewRecordInput
    output_model = WebsiteReviewRecordOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("tts"),)
    source_stage = "website_review_script"
    source_list_key = "lines"
    source_id_field = "line_id"
    clip_model = ReviewAudioClip

    async def handle(
        self, payload: WebsiteReviewRecordInput, ctx: RunContext
    ) -> WebsiteReviewRecordOutput:
        # critique-only runs (generate_video=false) skip TTS; the script it
        # narrates was skipped too, so running on would fail loud.
        require_video(ctx)
        return await super().handle(payload, ctx)

    def voice_for(self, item: dict, voice: dict, ctx: RunContext) -> str | None:
        """Cast a dialogue line to its speaker's voice (two-hander), else the
        single configured voice. Gated on ``dialogue_cast`` — the SAME predicate the
        script stage writes the prompt and derives speakers from."""
        return voice_for_speaker(item, voice, ctx.config_snapshot.get("brief") or {})
