"""Fixes routes — authoritative server-side ROI/cost ranking (F7).

``POST /api/fixes/rank`` ranks critique-driven fix proposals by impact-vs-cost
ROI. This model used to run in the browser (``fix-roi-ranking.ts``), shipping
the pricing constants + scoring weights in the tamperable client bundle. It now
runs server-side behind the API trust boundary. Pure compute — no DB, no
downstream calls; the standard tenant auth still applies (via ``TenantIdDep``,
plus the router-level tenant gate mounted in ``asgi.py``).
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.routes._common import TenantIdDep
from src.api.schemas import FixRankRequest, FixRankResponse
from src.services.fix_ranking import (
    assess_fix_ranking_confidence,
    rank_fixes_by_cost_vs_impact,
)

router = APIRouter(prefix="/api/fixes", tags=["fixes"])


@router.post("/rank", response_model=FixRankResponse, operation_id="fixes.rank")
async def rank_fixes(
    body: FixRankRequest,
    tenant_id: TenantIdDep,
) -> FixRankResponse:
    """Rank fix proposals by ROI (impact vs. estimated implementation cost).

    Byte-exact port of the former in-browser model. Returns the ranked list
    (roi_score desc, then video-impact desc, then effort asc) plus a confidence
    summary describing how much to trust the ordering.
    """
    proposals = [p.model_dump() for p in body.proposals]
    findings = [f.model_dump() for f in body.critique_findings]
    ranked = rank_fixes_by_cost_vs_impact(proposals, findings, body.output_cost_per_million_usd)
    confidence = assess_fix_ranking_confidence(ranked, findings)
    return FixRankResponse.model_validate({"ranked": ranked, "confidence": confidence})
