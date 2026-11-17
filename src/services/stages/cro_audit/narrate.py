"""cro_audit_narrate — TTS each audit-script line into an audio clip.

Thin per-type declaration over :class:`AuditNarrateStageBase`; only the source
script stage name is CRO-specific.
"""

from __future__ import annotations

from typing import ClassVar

from src.services.stages.audit_video.narrate import AuditNarrateStageBase
from src.services.stages.base import SlotUse


class CroAuditNarrateStage(AuditNarrateStageBase):
    name = "cro_audit_narrate"
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("tts"),)
    source_stage = "cro_audit_script"
