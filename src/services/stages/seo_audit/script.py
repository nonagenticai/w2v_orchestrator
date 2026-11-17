"""seo_audit_script — narrated voiceover over the SEO audit result.

Thin per-type declaration over :class:`AuditScriptStageBase`; only the scan
breakdown key (``groups``/``group``) and the prompt labels are SEO-specific.
"""

from __future__ import annotations

from typing import ClassVar

from src.services.stages.audit_video.script import AuditScriptStageBase
from src.services.stages.base import SlotUse


class SeoAuditScriptStage(AuditScriptStageBase):
    name = "seo_audit_script"
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)
    breakdown_key = "groups"
    item_key = "group"
    walkthrough_noun = "SEO-audit"
    score_label = "SEO Health Score"
    headline_metric = "score"
    quick_win_rationale = "why it matters"
