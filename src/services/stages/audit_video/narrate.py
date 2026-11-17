"""AuditNarrateStageBase — TTS each audit-script line into an audio clip.

Opt-in (``brief.generate_video``) NarrationStageBase subclass, mirroring
``lesson_narrate``: one sync ``TTSClient.synthesize`` per ``<type>_script`` line
(via the shared narration fan-out), each WAV persisted by ``store_tts_clip``
under a stable ``artifact_url`` keyed by ``line_id``. Single-narrator (the audit
has no dialogue casting). Skips cleanly when the video variant is off.

Per-type subclass sets ``name`` + ``source_stage`` (its ``<type>_script``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.stages.audit_video.gate import require_audit_video
from src.services.stages.audit_video.models import (
    AuditAudioClip,
    AuditNarrateInput,
    AuditNarrateOutput,
)
from src.services.stages.primitives.narration import NarrationStageBase

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


class AuditNarrateStageBase(NarrationStageBase[AuditNarrateInput, AuditNarrateOutput]):
    """Shared audit-narrate stage. Per-type subclass sets ``name`` + ``source_stage``."""

    input_model = AuditNarrateInput
    output_model = AuditNarrateOutput
    source_list_key = "lines"
    source_id_field = "line_id"
    clip_model = AuditAudioClip

    async def handle(self, payload: AuditNarrateInput, ctx: RunContext) -> AuditNarrateOutput:
        require_audit_video(ctx)
        return await super().handle(payload, ctx)
