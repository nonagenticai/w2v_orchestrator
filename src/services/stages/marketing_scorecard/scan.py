"""marketing_scorecard_scan — run the analysis_service marketing scorecard on the URL.

Calls the synchronous ``POST /api/marketing/scorecard`` (deterministic composite
over the CRO + SEO + marketing scanners) for ``brief.target_url`` and emits the
six-dimension weighted marketing score, per-dimension breakdown, and quick-win /
strategic finding buckets. When ``brief.include_llm`` is set, advisory Tier-3
copy assessments are attached without moving the deterministic score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from w2v_common.stages import stage_call

from src.clients.analysis_scorecard_client import (
    AnalysisScorecardClient,
    ScorecardEnrichment,
)
from src.clients.schemas.analysis_scorecard import (
    MarketingFinding as MarketingFindingWire,
)
from src.clients.schemas.analysis_scorecard import (
    MarketingFindingRef as MarketingFindingRefWire,
)
from src.clients.schemas.analysis_scorecard import (
    MarketingLlmAssessment as MarketingLlmAssessmentWire,
)
from src.services.stages.base import (
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithDegradations,
    StagePreconditionError,
)
from src.services.stages.shared.audit_enrichment import (
    AiStatusOut,
    AuditCheckOut,
    AuditFindingOut,
    CritiqueResultOut,
    ExecSummaryOut,
    GroundingResultOut,
    RecommendationOut,
    ScoreProjectionOut,
    ai_degradations,
    check_fields,
    finding_fields,
    map_ai,
    map_critique,
    map_exec_summary,
    map_grounding,
    map_projection,
    map_recommendation,
    read_grounding_sources,
    resolve_intent,
)

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

_SCORE_TIMEOUT_SECONDS = 60.0
# The AI-analysis path (recommendations, grounding, exec summary, critic) runs
# several LLM calls server-side (~60s live), so it gets a longer per-attempt
# budget and no retry — inside the 180s stage budget, and a retry would re-charge
# the LLM work.
_LLM_SCORE_TIMEOUT_SECONDS = 150.0


class MarketingScorecardScanInput(StageInput):
    pass


class MarketingDimension(StageOutput):
    dimension: str
    # None exactly when the dimension was not measured (analysis_service reports
    # ``measured=False`` + a ``note``) — never dressed up as a clean 100.
    score: float | None
    weight: float
    count: int = 0
    top_severity: str = "info"


class MarketingFindingRef(StageOutput):
    check_id: str
    dimension: str
    title: str
    severity: str
    priority: float | None = None
    rank: int | None = None
    evidence: str | None = None
    selector: str | None = None
    recommendation: RecommendationOut | None = None


class MarketingLlmAssessment(StageOutput):
    element: str
    dimension: str = "content"
    score: float
    assessment: str = ""
    suggestion: str = ""


class MarketingFinding(AuditFindingOut):
    dimension: str


class MarketingCheck(AuditCheckOut):
    dimension: str


class MarketingScorecardScanOutput(StageOutputWithDegradations):
    url: str
    overall_score: float
    grade: str
    llm_enriched: bool = False
    dimension_scores: dict[str, float | None] = {}  # noqa: RUF012 — pydantic copies field defaults per-instance
    dimensions: list[MarketingDimension] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    quick_wins: list[MarketingFindingRef] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    strategic: list[MarketingFindingRef] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    llm_assessments: list[MarketingLlmAssessment] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    projection: ScoreProjectionOut | None = None
    grounding: list[GroundingResultOut] | None = None
    exec_summary: ExecSummaryOut | None = None
    critique: CritiqueResultOut | None = None
    # --- audit-report contract (design 2026-10-17 §3.2 / §4 / §5.1) ---
    project_type: Literal["marketing_scorecard"] = "marketing_scorecard"
    scored_at: str = ""
    findings: list[MarketingFinding] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    checks: list[MarketingCheck] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    facts: dict[str, object] = {}  # noqa: RUF012 — pydantic copies field defaults per-instance
    ai: AiStatusOut | None = None


class MarketingScorecardScanStage(
    StageHandler[MarketingScorecardScanInput, MarketingScorecardScanOutput]
):
    name = "marketing_scorecard_scan"
    input_model = MarketingScorecardScanInput
    output_model = MarketingScorecardScanOutput

    def __init__(self, client: AnalysisScorecardClient | None = None) -> None:
        self._client = client or AnalysisScorecardClient(service_name="marketing")

    async def handle(
        self,
        payload: MarketingScorecardScanInput,
        ctx: RunContext,
    ) -> MarketingScorecardScanOutput:
        brief = ctx.config_snapshot.get("brief") or {}
        target = brief.get("target_url")
        if not target:
            raise StagePreconditionError("brief.target_url required for marketing_scorecard")
        include_llm = bool(brief.get("include_llm"))
        enrichment = ScorecardEnrichment.from_brief(
            include_llm=include_llm,
            intent=resolve_intent(ctx.config_snapshot, brief),
        )
        # Only forward grounding sources when the enrichment that consumes them is
        # on (include_llm gates include_grounding via ScorecardEnrichment.from_brief);
        # otherwise a brief.sources run would POST a payload analysis can't use.
        sources = read_grounding_sources(ctx) if include_llm else None

        url_str = str(target)
        result = await stage_call(
            lambda: self._client.score_marketing(
                url=url_str, include_llm=include_llm, enrichment=enrichment, sources=sources
            ),
            timeout_seconds=_LLM_SCORE_TIMEOUT_SECONDS if include_llm else _SCORE_TIMEOUT_SECONDS,
            retries=0 if include_llm else 1,  # deterministic scan: idempotent, safe to retry
            stage_name="marketing_scorecard_scan.score",
        )

        dimensions = [
            MarketingDimension(
                dimension=d.dimension,
                score=d.score,
                weight=d.weight,
                count=d.count,
                top_severity=d.top_severity,
            )
            for d in result.dimensions
        ]
        return MarketingScorecardScanOutput(
            url=result.source or url_str,
            overall_score=result.overall_score,
            grade=result.grade,
            llm_enriched=result.llm_enriched,
            dimension_scores=dict(result.dimension_scores),
            dimensions=dimensions,
            quick_wins=[_ref(r) for r in result.quick_wins],
            strategic=[_ref(r) for r in result.strategic],
            llm_assessments=[_assessment(a) for a in result.llm_assessments],
            projection=map_projection(result.projection),
            grounding=map_grounding(result.grounding),
            exec_summary=map_exec_summary(result.exec_summary),
            critique=map_critique(result.critique),
            scored_at=result.scored_at,
            findings=[_finding(f) for f in result.findings],
            checks=[
                MarketingCheck.model_validate({**check_fields(c), "dimension": c.dimension})
                for c in result.checks
            ],
            facts=dict(result.facts or {}),
            ai=map_ai(result.ai),
            degradations=ai_degradations(include_llm=include_llm, ai=result.ai),
        )


def _ref(r: MarketingFindingRefWire) -> MarketingFindingRef:
    return MarketingFindingRef(
        check_id=r.check_id,
        dimension=r.dimension,
        title=r.title,
        severity=r.severity,
        priority=r.priority,
        rank=r.rank,
        evidence=r.evidence,
        selector=r.selector,
        recommendation=map_recommendation(r.recommendation),
    )


def _assessment(a: MarketingLlmAssessmentWire) -> MarketingLlmAssessment:
    return MarketingLlmAssessment(
        element=a.element,
        dimension=a.dimension,
        score=a.score,
        assessment=a.assessment,
        suggestion=a.suggestion,
    )


def _finding(f: MarketingFindingWire) -> MarketingFinding:
    return MarketingFinding.model_validate({**finding_fields(f), "dimension": f.dimension})
