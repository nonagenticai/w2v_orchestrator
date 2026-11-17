"""seo_audit_record — render the audit slides to per-slide PNG images.

Thin per-type declaration over :class:`AuditRecordStageBase`; only the deck-title
fallback is SEO-specific.
"""

from __future__ import annotations

from src.services.stages.audit_video.record import AuditRecordStageBase


class SeoAuditRecordStage(AuditRecordStageBase):
    name = "seo_audit_record"
    deck_title = "SEO Audit"
