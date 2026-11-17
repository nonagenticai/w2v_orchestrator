"""Typed responses for analysis_service's critique API (port 6303).

Source of truth (read, not guessed) — analysis_service
``src/domains/critique/schemas.py``:
- ``FindingResponse`` (:101-117) — one critique finding.
- ``CritiqueResultResponse`` (:219-234) — the full result returned by
  ``GET /api/critique/{job_id}`` (which ``wait_for_critique`` polls).
- ``ComparisonResponse`` (:267-278) — ``GET /api/critique/compare``.

These mirrors are deliberately FOCUSED: only the fields the orchestrator's
``website_critique`` stage reads are declared, each with the server's own
nullability + default. The server models carry extra optional fields
(``confidence_scores``, ``scan_duration``, per-finding ``selector`` /
``wcag_criterion`` / …) that the orchestrator ignores; ``extra="ignore"``
(pydantic's default) drops them without re-introducing ``Any``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CritiqueFinding(BaseModel):
    """Focused mirror of ``FindingResponse`` — the 6 fields the stage reads.

    Server ``schemas.py:101-117``: ``finding_id``/``category``/``severity``/
    ``title``/``description`` are required; ``recommendation`` defaults to "".
    """

    finding_id: str  # server :104
    category: str  # server :105
    severity: str  # server :106
    title: str  # server :107
    description: str  # server :108
    recommendation: str = ""  # server :110


class CritiqueResult(BaseModel):
    """Focused mirror of ``CritiqueResultResponse`` (server ``schemas.py:219-234``).

    Nullable fields stay ``| None`` to match the server, so the stage's
    ``... or <default>`` fallbacks remain meaningful rather than masking absence.
    """

    job_id: str  # server :222
    status: str  # server :223
    url: str  # server :224
    findings: list[CritiqueFinding] | None = None  # server :225
    category_scores: dict[str, float] | None = None  # server :226
    overall_score: float | None = None  # server :227
    summary: str | None = None  # server :229


class CritiqueComparison(BaseModel):
    """Full mirror of ``ComparisonResponse`` (server ``schemas.py:267-278``).

    Flat scalars only — no nested ``dict[str, Any]`` — so mirroring the whole
    wire shape costs nothing on the ratchet.
    """

    job_id_1: str  # server :270
    job_id_2: str  # server :271
    url_1: str  # server :272
    url_2: str  # server :273
    score_deltas: dict[str, float] = Field(default_factory=dict)  # server :274
    overall_delta: float | None = None  # server :275
    new_findings: int = 0  # server :276
    resolved_findings: int = 0  # server :277
    persistent_findings: int = 0  # server :278
