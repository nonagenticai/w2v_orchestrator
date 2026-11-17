"""seo_audit_narrate — TTS each audit-script line into an audio clip.

Thin per-type declaration over :class:`AuditNarrateStageBase`; only the source
script stage name is SEO-specific.
"""

from __future__ import annotations

from typing import ClassVar

from src.services.stages.audit_video.narrate import AuditNarrateStageBase
from src.services.stages.base import SlotUse


class SeoAuditNarrateStage(AuditNarrateStageBase):
    name = "seo_audit_narrate"
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("tts"),)
    source_stage = "seo_audit_script"
