"""Shared StageOutput models + wire→output mappers for the audit "great report" enrichment.

analysis_service PR1-3 attaches presence-gated enrichment to every scorecard response:
per-finding ``priority`` / ``rank`` / ``evidence`` / ``selector`` / ``recommendation``,
and top-level ``projection`` / ``grounding`` / ``exec_summary`` / ``critique``. The
cro / seo / marketing scan stages thread these through identically, so the nested output
models and the wire→output mappers live here once.

Every field defaults to ``None`` / empty, and every mapper returns ``None`` for a ``None``
input — so a scan whose request opted into no enrichment (``include_llm`` off) carries no
enrichment VALUES. (The additive fields still serialize as ``null`` keys in the stage
output, so the JSON is not byte-for-byte identical to pre-PR4; every consumer presence-gates
on them, and the rendered scorecard / PDF data is unchanged.)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from pydantic import Field

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

from src.clients.schemas.analysis_scorecard import (
    AiStatus as AiStatusWire,
)
from src.clients.schemas.analysis_scorecard import (
    AuditCheckBase as AuditCheckWire,
)
from src.clients.schemas.analysis_scorecard import (
    AuditFindingBase as AuditFindingWire,
)
from src.clients.schemas.analysis_scorecard import (
    CritiqueResult as CritiqueResultWire,
)
from src.clients.schemas.analysis_scorecard import (
    ExecSummary as ExecSummaryWire,
)
from src.clients.schemas.analysis_scorecard import (
    GroundingResult as GroundingResultWire,
)
from src.clients.schemas.analysis_scorecard import (
    Recommendation as RecommendationWire,
)
from src.clients.schemas.analysis_scorecard import (
    ScoreProjection as ScoreProjectionWire,
)
from src.services.stages.base import StageOutput
from src.services.stages.shared.degradation import (
    AUDIT_AI_ANALYSIS_UNAVAILABLE,
    Degradation,
    degradation,
)


class RecommendationOut(StageOutput):
    """Evidence-grounded fix prose for one finding — mirrors ``Recommendation``."""

    fix_prose: str
    rationale: str = ""
    example: str = ""
    effort: str = ""


class ScoreProjectionOut(StageOutput):
    """Projected score lift from fixing the top findings — mirrors ``ScoreProjection``."""

    baseline: float
    projected: float
    delta: float
    removed: list[str] = Field(default_factory=list)


class GroundingResultOut(StageOutput):
    """One recommendation's claim-grounding verdict (A18) — mirrors ``GroundingResult``."""

    check_id: str
    supported: bool
    evidence_ref: str = ""


class ExecSummaryOut(StageOutput):
    """Executive summary + id-locked roadmap (A2) — mirrors ``ExecSummary``."""

    summary: str
    so_what: str = ""
    roadmap: list[str] = Field(default_factory=list)


class CritiqueResultOut(StageOutput):
    """Holistic report-critic verdict (A11) — mirrors ``CritiqueResult``."""

    score: float
    generic_check_ids: list[str] = Field(default_factory=list)
    ungrounded_check_ids: list[str] = Field(default_factory=list)
    rerun_stages: list[str] = Field(default_factory=list)


def map_recommendation(r: RecommendationWire | None) -> RecommendationOut | None:
    """Wire → output for a single finding's fix prose (``None`` passes through)."""
    if r is None:
        return None
    return RecommendationOut(
        fix_prose=r.fix_prose, rationale=r.rationale, example=r.example, effort=r.effort
    )


def map_projection(p: ScoreProjectionWire | None) -> ScoreProjectionOut | None:
    """Wire → output for the top-level score projection (``None`` passes through)."""
    if p is None:
        return None
    return ScoreProjectionOut(
        baseline=p.baseline, projected=p.projected, delta=p.delta, removed=list(p.removed)
    )


def map_grounding(g: list[GroundingResultWire] | None) -> list[GroundingResultOut] | None:
    """Wire → output for the top-level grounding verdicts (``None`` passes through)."""
    if g is None:
        return None
    return [
        GroundingResultOut(check_id=x.check_id, supported=x.supported, evidence_ref=x.evidence_ref)
        for x in g
    ]


def map_exec_summary(e: ExecSummaryWire | None) -> ExecSummaryOut | None:
    """Wire → output for the top-level exec-summary (``None`` passes through)."""
    if e is None:
        return None
    return ExecSummaryOut(summary=e.summary, so_what=e.so_what, roadmap=list(e.roadmap))


def map_critique(c: CritiqueResultWire | None) -> CritiqueResultOut | None:
    """Wire → output for the top-level report-critic verdict (``None`` passes through)."""
    if c is None:
        return None
    return CritiqueResultOut(
        score=c.score,
        generic_check_ids=list(c.generic_check_ids),
        ungrounded_check_ids=list(c.ungrounded_check_ids),
        rerun_stages=list(c.rerun_stages),
    )


def resolve_intent(
    config_snapshot: Mapping[str, object], brief: Mapping[str, object]
) -> str | None:
    """Optional page intent (audience / goal): ``brief.intent`` → ``direction.intent`` → None.

    Steers the opt-in recommendation-writer + exec-summary prose only; ignored on the
    deterministic path. Absent → ``None`` (generic recs), preserving today's behaviour.
    """
    # Only a string intent is meaningful — a non-string (list/int/dict) is ignored
    # rather than coerced into a garbage repr and forwarded to the writer prompt.
    raw = brief.get("intent")
    if not isinstance(raw, str) or not raw:
        raw = None
        direction = config_snapshot.get("direction")
        if isinstance(direction, dict):
            candidate = cast("Mapping[str, object]", direction).get("intent")
            raw = candidate if isinstance(candidate, str) else None
    text = raw.strip() if raw else ""
    return text or None


def read_grounding_sources(ctx: RunContext) -> list[str] | None:
    """The upstream ``source_ingest`` grounding text as a scorecard ``sources`` list.

    Returns the run's ``source_ingest`` ``grounding_text`` (the aggregated, scraped
    ``brief.sources`` corpus) wrapped as a single-element list, or ``None`` when
    ``source_ingest`` didn't run / found nothing — so a run without ``brief.sources``
    passes no ``sources`` and its request stays byte-identical.
    """
    out = ctx.prior_output("source_ingest") or {}
    text = str(out.get("grounding_text") or "").strip()
    return [text] if text else None


# --- Audit-report contract (design 2026-10-17 §3.2 / §4): ai / findings / checks ---


class AiLayerStatusOut(StageOutput):
    """One AI layer's outcome — mirrors ``AiLayerStatus``."""

    layer: str
    worker: str = ""
    status: str
    reason: str | None = None


class AiAnnotationOut(StageOutput):
    """One structured AI observation — mirrors ``AiAnnotation``."""

    layer: str
    kind: str
    title: str = ""
    summary: str = ""
    data: dict[str, object] = Field(default_factory=dict)


class AiStatusOut(StageOutput):
    """The AI-analysis status the report reads (never infers) — mirrors ``AiStatus``."""

    requested: bool
    status: str
    reason: str | None = None
    layers: list[AiLayerStatusOut] = Field(default_factory=list)
    annotations: list[AiAnnotationOut] = Field(default_factory=list)


class AuditFindingOut(StageOutput):
    """A full finding (every severity). Each scan subclasses it with its bucket key."""

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
    recommendation: RecommendationOut | None = None


class AuditCheckOut(StageOutput):
    """One catalogued check with pass / fail / not_run. Subclassed with the bucket key."""

    check_id: str
    title: str = ""
    status: str
    severity: str | None = None
    measured: str | None = None
    target: str | None = None
    reason: str | None = None


def map_ai(a: AiStatusWire | None) -> AiStatusOut | None:
    """Wire → output for the ``ai`` status block (``None`` passes through)."""
    if a is None:
        return None
    return AiStatusOut.model_validate(a.model_dump())


def finding_fields(f: AuditFindingWire) -> dict[str, object]:
    """The shared finding fields, for ``<Finding>.model_validate`` (the scan adds its bucket key)."""
    return {
        "check_id": f.check_id,
        "title": f.title,
        "severity": f.severity,
        "description": f.description,
        "fix": f.fix,
        "effort": f.effort,
        "evidence": f.evidence,
        "selector": f.selector,
        "rank": f.rank,
        "priority": f.priority,
        "action_bucket": f.action_bucket,
        "recommendation": None if f.recommendation is None else f.recommendation.model_dump(),
    }


def check_fields(c: AuditCheckWire) -> dict[str, object]:
    """The shared check fields, for ``<Check>.model_validate`` (the scan adds its bucket key)."""
    return {
        "check_id": c.check_id,
        "title": c.title,
        "status": c.status,
        "severity": c.severity,
        "measured": c.measured,
        "target": c.target,
        "reason": c.reason,
    }


def ai_degradations(*, include_llm: bool, ai: AiStatusWire | None) -> list[Degradation]:
    """``AUDIT_AI_ANALYSIS_UNAVAILABLE`` when AI was asked for and did not deliver.

    Design §3.3: ``partial`` / ``unavailable`` degrade the run; so does a response
    with NO ``ai`` block (an older analysis_service pod) — a missing status is never
    read as complete. AI not requested → no degradation, whatever the block says.
    """
    if not include_llm:
        return []
    if ai is None:
        return [
            degradation(AUDIT_AI_ANALYSIS_UNAVAILABLE, "analysis_service did not report AI status")
        ]
    if ai.status not in {"partial", "unavailable"}:
        return []
    bad = [
        f"{layer.layer}: {layer.reason or layer.status}"
        for layer in ai.layers
        if layer.status in {"failed", "empty"}
    ]
    detail = "; ".join(bad) or ai.reason or f"AI analysis {ai.status}"
    return [degradation(AUDIT_AI_ANALYSIS_UNAVAILABLE, f"AI analysis {ai.status} — {detail}")]
