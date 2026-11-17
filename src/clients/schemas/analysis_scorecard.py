"""Typed responses for analysis_service's synchronous scorecard surfaces (port 6303).

Source of truth (read, not guessed) — analysis_service ``src/domains/``:
- ``POST /api/cro/score``            -> ``CroScoreResponse``           (cro/schemas.py:81),
  nested ``CroSubsectionSummary`` (:24), ``CroFindingRef`` (:50), ``CroLlmJudgment`` (:72).
- ``POST /api/seo/audit``            -> ``SeoAuditResponse``           (seo_audit/schemas.py:104),
  nested ``SeoAuditGroupSummary`` (:59), ``SeoAuditFindingRef`` (:83).
- ``POST /api/marketing/scorecard``  -> ``MarketingScorecardResponse`` (marketing/schemas.py:97),
  nested ``MarketingDimensionSummary`` (:39), ``MarketingFindingRef`` (:63),
  ``MarketingLlmAssessment`` (:80).

The "great report" enrichment (analysis_service PR1-3) is additive and
presence-gated: each ``*FindingRef`` may carry ``priority`` / ``rank`` /
``evidence`` / ``selector`` / ``recommendation``, and each ``*ScoreResponse`` may
carry ``projection`` / ``grounding`` / ``exec_summary`` / ``critique``. All default
to ``None`` and appear only when the request opts into the matching
``include_*`` flag — so a default request's response is byte-identical to today.
The shared nested models (``Recommendation`` … ``CritiqueResult``) mirror
``scorecards/schemas.py`` on the server.

Each route declares its ``response_model``, so every field below is guaranteed on
the wire and any undeclared key is stripped by FastAPI — nothing here invents a
field the server never emits. ``grade`` and ``top_severity`` stay ``str`` (a new
service-side grade/severity string must not hard-fail a running pipeline);
collection defaults use ``Field(default_factory=...)``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# --- Shared "great report" enrichment models (mirror scorecards/schemas.py) ---


class Recommendation(BaseModel):
    """Evidence-grounded fix prose for one finding — mirrors ``Recommendation``."""

    fix_prose: str
    rationale: str = ""
    example: str = ""
    effort: str = ""


class ScoreProjection(BaseModel):
    """Projected score lift from fixing the top findings — mirrors ``ScoreProjection``."""

    baseline: float
    projected: float
    delta: float
    removed: list[str] = Field(default_factory=list)


class GroundingResult(BaseModel):
    """One recommendation's claim-grounding verdict (A18) — mirrors ``GroundingResult``."""

    check_id: str
    supported: bool
    evidence_ref: str = ""


class ExecSummary(BaseModel):
    """Executive summary + id-locked roadmap (A2) — mirrors ``ExecSummary``."""

    summary: str
    so_what: str = ""
    roadmap: list[str] = Field(default_factory=list)


class CritiqueResult(BaseModel):
    """Holistic report-critic verdict (A11) — mirrors ``CritiqueResult``."""

    score: float
    generic_check_ids: list[str] = Field(default_factory=list)
    ungrounded_check_ids: list[str] = Field(default_factory=list)
    rerun_stages: list[str] = Field(default_factory=list)


# --- Audit-report contract (analysis_service scorecards/schemas.py, design 2026-10-17
# §3.2 / §4 X-G1..X-G4): ``ai`` / ``findings`` / ``checks`` / ``facts`` on all three
# responses. ``status`` / ``layer`` stay ``str`` so a new server-side literal cannot
# hard-fail a running pipeline; ``facts`` stays an open mapping for the same reason
# (its per-type shape is rendered presence-gated by ``shared/audit_report``).


class AiLayerStatus(BaseModel):
    """One WM-backed layer's outcome — mirrors ``AiLayerStatus``."""

    layer: str
    worker: str = ""
    status: str
    reason: str | None = None


class AiAnnotation(BaseModel):
    """One structured AI observation (SEO intent / E-E-A-T / gaps) — mirrors ``AiAnnotation``."""

    layer: str
    kind: str
    title: str = ""
    summary: str = ""
    data: dict[str, object] = Field(default_factory=dict)


class AiStatus(BaseModel):
    """Machine-readable AI-analysis status — mirrors ``AiStatus``.

    ``None`` on the response (an older analysis_service pod) is NOT "complete":
    the scan stages treat it as unavailable when AI was requested (design §3.3).
    """

    requested: bool
    status: str
    reason: str | None = None
    layers: list[AiLayerStatus] = Field(default_factory=list)
    annotations: list[AiAnnotation] = Field(default_factory=list)


class AuditFindingBase(BaseModel):
    """A full finding (every severity) — mirrors ``AuditFindingBase``."""

    check_id: str
    title: str
    severity: str
    description: str = ""
    fix: str = ""
    effort: str = ""
    evidence: str | None = None
    selector: str | None = None
    rank: int | None = None
    priority: float | None = None
    action_bucket: str = "none"
    recommendation: Recommendation | None = None


class AuditCheckBase(BaseModel):
    """One catalogued check with pass / fail / not_run — mirrors ``AuditCheckBase``."""

    check_id: str
    title: str = ""
    status: str
    severity: str | None = None
    measured: str | None = None
    target: str | None = None
    reason: str | None = None


class CroExperiment(BaseModel):
    """Authored A/B experiment for a CRO check — mirrors ``CroExperiment``."""

    hypothesis: str
    primary_metric: str
    variant: str


class CroFinding(AuditFindingBase):
    section: str
    experiment: CroExperiment | None = None


class CroCheck(AuditCheckBase):
    section: str


class SeoAuditFinding(AuditFindingBase):
    group: str


class SeoAuditCheck(AuditCheckBase):
    group: str


class MarketingFinding(AuditFindingBase):
    dimension: str


class MarketingCheck(AuditCheckBase):
    dimension: str


class CroSubsectionSummary(BaseModel):
    """One CRO sub-section's score, weight, and finding rollup — mirrors ``CroSubsectionSummary``."""

    section: str
    score: float | None
    weight: float
    count: int = 0
    top_severity: str = "info"
    measured: bool = True
    note: str | None = None


class CroFindingRef(BaseModel):
    """Lightweight finding reference for action buckets — mirrors ``CroFindingRef``.

    ``priority`` / ``rank`` / ``evidence`` / ``selector`` / ``recommendation`` are
    additive and appear only when the request opts into the matching enrichment.
    """

    check_id: str
    section: str
    title: str
    severity: str
    priority: float | None = None
    rank: int | None = None
    evidence: str | None = None
    selector: str | None = None
    recommendation: Recommendation | None = None


class CroLlmJudgment(BaseModel):
    """One Tier-3 LLM micro-judgment (advisory only) — mirrors ``CroLlmJudgment``."""

    element: str
    score: float
    assessment: str = ""
    suggestion: str = ""


class CroScoreResponse(BaseModel):
    """``POST /api/cro/score`` — mirrors ``CroScoreResponse``."""

    overall_score: float
    grade: str
    subsection_scores: dict[str, float | None] = Field(default_factory=dict)
    subsections: list[CroSubsectionSummary] = Field(default_factory=list)
    unmeasured_sections: list[str] = Field(default_factory=list)
    coverage: float | None = None
    quick_wins: list[CroFindingRef] = Field(default_factory=list)
    strategic: list[CroFindingRef] = Field(default_factory=list)
    source: str = ""
    llm_enriched: bool = False
    llm_judgments: list[CroLlmJudgment] = Field(default_factory=list)
    projection: ScoreProjection | None = None
    grounding: list[GroundingResult] | None = None
    exec_summary: ExecSummary | None = None
    critique: CritiqueResult | None = None
    findings: list[CroFinding] = Field(default_factory=list)
    checks: list[CroCheck] = Field(default_factory=list)
    facts: dict[str, object] | None = None
    ai: AiStatus | None = None
    scored_at: str = ""


class SeoAuditGroupSummary(BaseModel):
    """One SEO audit group's score, weight, and finding rollup — mirrors ``SeoAuditGroupSummary``."""

    group: str
    score: float | None
    weight: float
    count: int = 0
    top_severity: str = "info"
    measured: bool = True
    note: str | None = None


class SeoAuditFindingRef(BaseModel):
    """Lightweight finding reference for action buckets — mirrors ``SeoAuditFindingRef``.

    ``priority`` / ``rank`` / ``evidence`` / ``selector`` / ``recommendation`` are
    additive and appear only when the request opts into the matching enrichment.
    """

    check_id: str
    group: str
    title: str
    severity: str
    priority: float | None = None
    rank: int | None = None
    evidence: str | None = None
    selector: str | None = None
    recommendation: Recommendation | None = None


class SeoAuditResponse(BaseModel):
    """``POST /api/seo/audit`` — mirrors ``SeoAuditResponse``."""

    overall_score: float
    grade: str
    group_scores: dict[str, float | None] = Field(default_factory=dict)
    groups: list[SeoAuditGroupSummary] = Field(default_factory=list)
    unmeasured_groups: list[str] = Field(default_factory=list)
    coverage: float | None = None
    quick_wins: list[SeoAuditFindingRef] = Field(default_factory=list)
    strategic: list[SeoAuditFindingRef] = Field(default_factory=list)
    source: str = ""
    llm_enriched: bool = False
    projection: ScoreProjection | None = None
    grounding: list[GroundingResult] | None = None
    exec_summary: ExecSummary | None = None
    critique: CritiqueResult | None = None
    findings: list[SeoAuditFinding] = Field(default_factory=list)
    checks: list[SeoAuditCheck] = Field(default_factory=list)
    facts: dict[str, object] | None = None
    ai: AiStatus | None = None
    scored_at: str = ""


class MarketingDimensionSummary(BaseModel):
    """One scorecard dimension's score, weight, and finding rollup — mirrors ``MarketingDimensionSummary``."""

    dimension: str
    score: float | None
    weight: float
    count: int = 0
    top_severity: str = "info"


class MarketingFindingRef(BaseModel):
    """Lightweight finding reference for action buckets — mirrors ``MarketingFindingRef``.

    ``priority`` / ``rank`` / ``evidence`` / ``selector`` / ``recommendation`` are
    additive and appear only when the request opts into the matching enrichment.
    """

    check_id: str
    dimension: str
    title: str
    severity: str
    priority: float | None = None
    rank: int | None = None
    evidence: str | None = None
    selector: str | None = None
    recommendation: Recommendation | None = None


class MarketingLlmAssessment(BaseModel):
    """One Tier-3 LLM copy assessment (advisory only) — mirrors ``MarketingLlmAssessment``."""

    element: str
    dimension: str = "content"
    score: float
    assessment: str = ""
    suggestion: str = ""


class MarketingScorecardResponse(BaseModel):
    """``POST /api/marketing/scorecard`` — mirrors ``MarketingScorecardResponse``."""

    overall_score: float
    grade: str
    dimension_scores: dict[str, float | None] = Field(default_factory=dict)
    dimensions: list[MarketingDimensionSummary] = Field(default_factory=list)
    quick_wins: list[MarketingFindingRef] = Field(default_factory=list)
    strategic: list[MarketingFindingRef] = Field(default_factory=list)
    source: str = ""
    llm_enriched: bool = False
    llm_assessments: list[MarketingLlmAssessment] = Field(default_factory=list)
    projection: ScoreProjection | None = None
    grounding: list[GroundingResult] | None = None
    exec_summary: ExecSummary | None = None
    critique: CritiqueResult | None = None
    findings: list[MarketingFinding] = Field(default_factory=list)
    checks: list[MarketingCheck] = Field(default_factory=list)
    facts: dict[str, object] | None = None
    ai: AiStatus | None = None
    scored_at: str = ""
