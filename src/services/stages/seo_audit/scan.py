"""seo_audit_scan — run the analysis_service SEO audit on the target URL.

Calls the synchronous ``POST /api/seo/audit`` (deterministic, Tier 0-2 by
default) for ``brief.target_url`` and emits the SEO Health Score, per-group
breakdown, and prioritized findings. When ``brief.include_llm`` is set, the two
opt-in micro-LLM analyzers add info-level annotations without moving the score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from w2v_common.stages import stage_call

from src.clients.analysis_scorecard_client import (
    AnalysisScorecardClient,
    ScorecardEnrichment,
)
from src.clients.schemas.analysis_scorecard import (
    SeoAuditFinding as SeoAuditFindingWire,
)
from src.clients.schemas.analysis_scorecard import (
    SeoAuditFindingRef as SeoAuditFindingRefWire,
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

_AUDIT_TIMEOUT_SECONDS = 60.0
# Longer per-attempt budget when the opt-in advanced analyzers run (competitor
# content-gap + a sitemap-crawl of up to max_pages). Kept under the 180s
# seo_audit_scan stage timeout (with retries=0 on this path) so the outer stage
# never truncates a live attempt.
_ADVANCED_AUDIT_TIMEOUT_SECONDS = 150.0


class SeoAuditScanInput(StageInput):
    pass


class SeoAuditGroup(StageOutput):
    group: str
    # None exactly when the group was not measured (analysis_service reports
    # ``measured=False`` + a ``note``) — never dressed up as a clean 100.
    score: float | None
    weight: float
    count: int = 0
    top_severity: str = "info"
    measured: bool = True
    note: str | None = None


class SeoAuditFindingRef(StageOutput):
    check_id: str
    group: str
    title: str
    severity: str
    priority: float | None = None
    rank: int | None = None
    evidence: str | None = None
    selector: str | None = None
    recommendation: RecommendationOut | None = None


class SeoAuditFinding(AuditFindingOut):
    group: str


class SeoAuditCheck(AuditCheckOut):
    group: str


class SeoAuditScanOutput(StageOutputWithDegradations):
    url: str
    overall_score: float
    grade: str
    llm_enriched: bool = False
    group_scores: dict[str, float | None] = {}  # noqa: RUF012 — pydantic copies field defaults per-instance
    groups: list[SeoAuditGroup] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    quick_wins: list[SeoAuditFindingRef] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    strategic: list[SeoAuditFindingRef] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    projection: ScoreProjectionOut | None = None
    grounding: list[GroundingResultOut] | None = None
    exec_summary: ExecSummaryOut | None = None
    critique: CritiqueResultOut | None = None
    # --- audit-report contract (design 2026-10-17 §3.2 / §4 / §5.1) ---
    project_type: Literal["seo_audit"] = "seo_audit"
    scored_at: str = ""
    coverage: float | None = None
    unmeasured_groups: list[str] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    findings: list[SeoAuditFinding] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    checks: list[SeoAuditCheck] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    facts: dict[str, object] = {}  # noqa: RUF012 — pydantic copies field defaults per-instance
    ai: AiStatusOut | None = None


class SeoAuditScanStage(StageHandler[SeoAuditScanInput, SeoAuditScanOutput]):
    name = "seo_audit_scan"
    input_model = SeoAuditScanInput
    output_model = SeoAuditScanOutput

    def __init__(self, client: AnalysisScorecardClient | None = None) -> None:
        self._client = client or AnalysisScorecardClient(service_name="seo")

    async def handle(
        self,
        payload: SeoAuditScanInput,
        ctx: RunContext,
    ) -> SeoAuditScanOutput:
        brief = ctx.config_snapshot.get("brief") or {}
        target = brief.get("target_url")
        if not target:
            raise StagePreconditionError("brief.target_url required for seo_audit")
        include_llm = bool(brief.get("include_llm"))
        # Opt-in advanced analyzers (off by default → default request unchanged):
        # competitor_urls feeds the content-gap analyzer (only fires when include_llm
        # is on downstream); cross_page_linking opts into the sitemap-crawl audit.
        competitor_urls = [
            str(u).strip() for u in (brief.get("competitor_urls") or []) if str(u).strip()
        ]
        cross_page_linking = bool(brief.get("cross_page_linking"))
        raw_max_pages = brief.get("max_pages")
        max_pages = int(raw_max_pages) if isinstance(raw_max_pages, int) else None
        enrichment = ScorecardEnrichment.from_brief(
            include_llm=include_llm,
            intent=resolve_intent(ctx.config_snapshot, brief),
        )
        # Only forward grounding sources when the enrichment that consumes them is
        # on (include_llm gates include_grounding via ScorecardEnrichment.from_brief);
        # otherwise a brief.sources run would POST a payload analysis can't use.
        sources = read_grounding_sources(ctx) if include_llm else None

        # The advanced analyzers (competitor content-gap + cross-page sitemap crawl)
        # and the AI-analysis layers (include_llm)
        # legitimately take longer than the plain deterministic scan, so give them more
        # per-attempt headroom (still inside the 180s stage budget) and skip the retry —
        # both are fail-soft downstream, and a retry would re-charge the LLM/crawl.
        advanced = bool(competitor_urls) or cross_page_linking or include_llm
        url_str = str(target)
        result = await stage_call(
            lambda: self._client.score_seo(
                url=url_str,
                include_llm=include_llm,
                competitor_urls=competitor_urls or None,
                cross_page_linking=cross_page_linking,
                max_pages=max_pages,
                enrichment=enrichment,
                sources=sources,
            ),
            timeout_seconds=_ADVANCED_AUDIT_TIMEOUT_SECONDS if advanced else _AUDIT_TIMEOUT_SECONDS,
            retries=0 if advanced else 1,  # plain scan: idempotent GET-like, safe to retry
            stage_name="seo_audit_scan.audit",
        )

        groups = [
            SeoAuditGroup(
                group=g.group,
                score=g.score,
                weight=g.weight,
                count=g.count,
                top_severity=g.top_severity,
                measured=g.measured,
                note=g.note,
            )
            for g in result.groups
        ]
        return SeoAuditScanOutput(
            url=result.source or url_str,
            overall_score=result.overall_score,
            grade=result.grade,
            llm_enriched=result.llm_enriched,
            group_scores=dict(result.group_scores),
            groups=groups,
            quick_wins=[_ref(r) for r in result.quick_wins],
            strategic=[_ref(r) for r in result.strategic],
            projection=map_projection(result.projection),
            grounding=map_grounding(result.grounding),
            exec_summary=map_exec_summary(result.exec_summary),
            critique=map_critique(result.critique),
            scored_at=result.scored_at,
            coverage=result.coverage,
            unmeasured_groups=list(result.unmeasured_groups),
            findings=[_finding(f) for f in result.findings],
            checks=[
                SeoAuditCheck.model_validate({**check_fields(c), "group": c.group})
                for c in result.checks
            ],
            facts=dict(result.facts or {}),
            ai=map_ai(result.ai),
            degradations=ai_degradations(include_llm=include_llm, ai=result.ai),
        )


def _ref(r: SeoAuditFindingRefWire) -> SeoAuditFindingRef:
    return SeoAuditFindingRef(
        check_id=r.check_id,
        group=r.group,
        title=r.title,
        severity=r.severity,
        priority=r.priority,
        rank=r.rank,
        evidence=r.evidence,
        selector=r.selector,
        recommendation=map_recommendation(r.recommendation),
    )


def _finding(f: SeoAuditFindingWire) -> SeoAuditFinding:
    return SeoAuditFinding.model_validate({**finding_fields(f), "group": f.group})
