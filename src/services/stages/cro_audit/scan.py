"""cro_audit_scan — run the analysis_service CRO score on the target URL.

Calls the synchronous ``POST /api/cro/score`` (deterministic 8-section scanner)
for ``brief.target_url`` and emits the CRO conversion score, per-sub-section
breakdown, and quick-win / strategic finding buckets. When ``brief.include_llm``
is set, the opt-in Tier-3 micro-judge adds advisory per-snippet judgments
without moving the deterministic score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from w2v_common.stages import stage_call

from src.clients.analysis_scorecard_client import (
    AnalysisScorecardClient,
    ScorecardEnrichment,
)
from src.clients.schemas.analysis_scorecard import (
    CroFinding as CroFindingWire,
)
from src.clients.schemas.analysis_scorecard import (
    CroFindingRef as CroFindingRefWire,
)
from src.clients.schemas.analysis_scorecard import (
    CroLlmJudgment as CroLlmJudgmentWire,
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


class CroAuditScanInput(StageInput):
    pass


class CroSubsection(StageOutput):
    section: str
    # None exactly when the section was not measured (analysis_service reports
    # ``measured=False`` + a ``note``) — never dressed up as a clean 100.
    score: float | None
    weight: float
    count: int = 0
    top_severity: str = "info"
    measured: bool = True
    note: str | None = None


class CroFindingRef(StageOutput):
    check_id: str
    section: str
    title: str
    severity: str
    priority: float | None = None
    rank: int | None = None
    evidence: str | None = None
    selector: str | None = None
    recommendation: RecommendationOut | None = None


class CroLlmJudgment(StageOutput):
    element: str
    score: float
    assessment: str = ""
    suggestion: str = ""


class CroExperimentOut(StageOutput):
    hypothesis: str
    primary_metric: str
    variant: str


class CroFinding(AuditFindingOut):
    section: str
    experiment: CroExperimentOut | None = None


class CroCheck(AuditCheckOut):
    section: str


class CroAuditScanOutput(StageOutputWithDegradations):
    url: str
    overall_score: float
    grade: str
    llm_enriched: bool = False
    subsection_scores: dict[str, float | None] = {}  # noqa: RUF012 — pydantic copies field defaults per-instance
    subsections: list[CroSubsection] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    quick_wins: list[CroFindingRef] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    strategic: list[CroFindingRef] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    llm_judgments: list[CroLlmJudgment] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    projection: ScoreProjectionOut | None = None
    grounding: list[GroundingResultOut] | None = None
    exec_summary: ExecSummaryOut | None = None
    critique: CritiqueResultOut | None = None
    # --- audit-report contract (design 2026-10-17 §3.2 / §4 / §5.1) ---
    project_type: Literal["cro_audit"] = "cro_audit"
    scored_at: str = ""
    coverage: float | None = None
    unmeasured_sections: list[str] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    findings: list[CroFinding] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    checks: list[CroCheck] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    facts: dict[str, object] = {}  # noqa: RUF012 — pydantic copies field defaults per-instance
    ai: AiStatusOut | None = None


class CroAuditScanStage(StageHandler[CroAuditScanInput, CroAuditScanOutput]):
    name = "cro_audit_scan"
    input_model = CroAuditScanInput
    output_model = CroAuditScanOutput

    def __init__(self, client: AnalysisScorecardClient | None = None) -> None:
        self._client = client or AnalysisScorecardClient(service_name="cro")

    async def handle(
        self,
        payload: CroAuditScanInput,
        ctx: RunContext,
    ) -> CroAuditScanOutput:
        brief = ctx.config_snapshot.get("brief") or {}
        target = brief.get("target_url")
        if not target:
            raise StagePreconditionError("brief.target_url required for cro_audit")
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
            lambda: self._client.score_cro(
                url=url_str, include_llm=include_llm, enrichment=enrichment, sources=sources
            ),
            timeout_seconds=_LLM_SCORE_TIMEOUT_SECONDS if include_llm else _SCORE_TIMEOUT_SECONDS,
            retries=0 if include_llm else 1,  # deterministic scan: idempotent, safe to retry
            stage_name="cro_audit_scan.score",
        )

        subsections = [
            CroSubsection(
                section=s.section,
                score=s.score,
                weight=s.weight,
                count=s.count,
                top_severity=s.top_severity,
                measured=s.measured,
                note=s.note,
            )
            for s in result.subsections
        ]
        return CroAuditScanOutput(
            url=result.source or url_str,
            overall_score=result.overall_score,
            grade=result.grade,
            llm_enriched=result.llm_enriched,
            subsection_scores=dict(result.subsection_scores),
            subsections=subsections,
            quick_wins=[_ref(r) for r in result.quick_wins],
            strategic=[_ref(r) for r in result.strategic],
            llm_judgments=[_judgment(j) for j in result.llm_judgments],
            projection=map_projection(result.projection),
            grounding=map_grounding(result.grounding),
            exec_summary=map_exec_summary(result.exec_summary),
            critique=map_critique(result.critique),
            scored_at=result.scored_at,
            coverage=result.coverage,
            unmeasured_sections=list(result.unmeasured_sections),
            findings=[_finding(f) for f in result.findings],
            checks=[
                CroCheck.model_validate({**check_fields(c), "section": c.section})
                for c in result.checks
            ],
            facts=dict(result.facts or {}),
            ai=map_ai(result.ai),
            degradations=ai_degradations(include_llm=include_llm, ai=result.ai),
        )


def _ref(r: CroFindingRefWire) -> CroFindingRef:
    return CroFindingRef(
        check_id=r.check_id,
        section=r.section,
        title=r.title,
        severity=r.severity,
        priority=r.priority,
        rank=r.rank,
        evidence=r.evidence,
        selector=r.selector,
        recommendation=map_recommendation(r.recommendation),
    )


def _judgment(j: CroLlmJudgmentWire) -> CroLlmJudgment:
    return CroLlmJudgment(
        element=j.element,
        score=j.score,
        assessment=j.assessment,
        suggestion=j.suggestion,
    )


def _finding(f: CroFindingWire) -> CroFinding:
    exp = f.experiment
    return CroFinding.model_validate(
        {
            **finding_fields(f),
            "section": f.section,
            "experiment": None if exp is None else exp.model_dump(),
        }
    )
