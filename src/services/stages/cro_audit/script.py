"""cro_audit_script — narrated voiceover over the CRO conversion audit result.

Thin per-type declaration over :class:`AuditScriptStageBase`; the cro scan output
is keyed by ``subsections``/``section`` (not seo's ``groups``/``group``) and the
prompt labels frame conversion rather than discoverability.
"""

from __future__ import annotations

from typing import ClassVar

from src.services.stages.audit_video.script import AuditScriptStageBase
from src.services.stages.base import SlotUse


class CroAuditScriptStage(AuditScriptStageBase):
    name = "cro_audit_script"
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)
    breakdown_key = "subsections"
    item_key = "section"
    walkthrough_noun = "CRO / conversion-audit"
    score_label = "CRO Conversion Score"
    headline_metric = "conversion score"
    quick_win_rationale = "why it lifts conversion"
