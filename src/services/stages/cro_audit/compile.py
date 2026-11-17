"""cro_audit_compile — composite audit slide images + narration into one video.

Thin per-type declaration over :class:`AuditCompileStageBase` (fully generic —
the line→slide→image join is identical across audit types).
"""

from __future__ import annotations

from src.services.stages.audit_video.compile import AuditCompileStageBase


class CroAuditCompileStage(AuditCompileStageBase):
    name = "cro_audit_compile"
