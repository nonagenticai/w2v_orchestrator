"""cro_audit_slides — project the audit narration lines onto branded slides.

Thin per-type declaration over :class:`AuditSlidesStageBase` (fully generic —
the slide projection is identical across audit types).
"""

from __future__ import annotations

from src.services.stages.audit_video.slides import AuditSlidesStageBase


class CroAuditSlidesStage(AuditSlidesStageBase):
    name = "cro_audit_slides"
