"""Schemas for the server-side fix ROI/cost ranking endpoint (F7).

Request/response contract for ``POST /api/fixes/rank`` — the authoritative
server-side port of the frontend's in-browser ROI model. Field names are
snake_case (Python/OpenAPI convention); the FE consumes these via codegen.
Re-exported from the ``src.api.schemas`` facade.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FeatureSuggestionInput(BaseModel):
    """A single fix/feature proposal — mirror of the FE ``FeatureSuggestion``."""

    id: int
    title: str
    category: str = ""
    impact: str = ""
    complexity: str = ""
    estimated_effort: str = ""
    description: str = ""
    rationale: str = ""


class CritiqueFindingInput(BaseModel):
    """A critique finding used to weight proposal severity — mirror of the FE
    ``CritiqueFinding`` (only the fields the model reads are required)."""

    finding_id: str
    category: str = ""
    severity: str = ""
    title: str = ""
    description: str = ""
    url: str | None = None
    recommendation: str | None = None
    selector: str | None = None


class FixRankRequest(BaseModel):
    proposals: list[FeatureSuggestionInput]
    critique_findings: list[CritiqueFindingInput] = Field(default_factory=list)
    output_cost_per_million_usd: float = 15


class RankedFixSuggestion(BaseModel):
    proposal: FeatureSuggestionInput
    critique_severity: str
    critique_severity_score: float
    matched_finding_title: str | None
    video_impact_score: float
    redesign_effort_days: float
    redesign_effort_score: float
    estimated_output_tokens: int
    output_cost_per_million_usd: float
    estimated_token_cost_usd: float
    roi_score: float
    priority_label: str


class RankingConfidenceSummary(BaseModel):
    level: Literal["strong", "moderate", "weak"]
    score: float
    summary: str
    uncertainty_signals: list[str]
    proposal_count: int
    finding_count: int
    severity_diversity: int
    matched_proposal_ratio: float
    roi_spread: float


class FixRankResponse(BaseModel):
    ranked: list[RankedFixSuggestion]
    confidence: RankingConfidenceSummary
