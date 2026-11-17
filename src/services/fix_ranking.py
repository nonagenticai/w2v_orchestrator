"""Server-side ROI / cost ranking model for critique-driven fix proposals (F7).

Authoritative Python port of the frontend model that used to run in the
browser (``src/lib/utils/fix-roi-ranking.ts``). Moving it server-side puts the
pricing constants and the scoring model behind the API trust boundary instead
of shipping them in the tamperable browser bundle.

Pure functions — no DB, no I/O. Every constant/table lives at module level.

Parity is byte-exact with the TS original and is locked by a captured-TS
fixture oracle (``tests/services/test_fix_ranking.py`` over
``tests/fixtures/fix_ranking_parity.json``). Two JS-isms are reproduced
deliberately:

* ``_js_to_fixed_2`` reproduces ``Number(x.toFixed(2))`` — JS ``toFixed``
  rounds on the EXACT double value with ties resolved toward the larger
  scaled integer. ``Decimal(x)`` is the exact value of the double and
  ``ROUND_HALF_UP`` matches that tie rule for the (always-positive) roi score.
  Python's built-in ``round`` is bankers' rounding and would DIFFER on ``.5``.
* ``_js_round`` reproduces ``Math.round`` as ``floor(x + 0.5)`` (round half
  up), NOT Python's bankers ``round``.

All float arithmetic uses the same operations in the same order as the TS, so
the intermediate doubles are bit-identical to the browser's.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

CLAUDE_CODE_OUTPUT_COST_PER_MILLION_USD = 15
TOKENS_PER_EFFORT_DAY = 30_000

SEVERITY_SCORES: dict[str, float] = {
    "critical": 1.0,
    "high": 0.8,
    "medium": 0.55,
    "low": 0.3,
    "info": 0.15,
}

VIDEO_IMPACT_SCORES: dict[str, float] = {
    "critical": 1.0,
    "high": 1.0,
    "medium": 0.65,
    "low": 0.35,
}

COMPLEXITY_DAYS: dict[str, float] = {
    "low": 2,
    "simple": 2,
    "medium": 7,
    "moderate": 7,
    "high": 15,
    "complex": 15,
}

COMPLEXITY_TOKEN_MULTIPLIER: dict[str, float] = {
    "low": 0.8,
    "simple": 0.8,
    "medium": 1,
    "moderate": 1,
    "high": 1.35,
    "complex": 1.35,
}

STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "your",
        "will",
        "has",
        "have",
        "are",
        "was",
        "were",
        "you",
        "they",
        "their",
        "can",
        "should",
        "could",
        "about",
        "page",
        "site",
        "user",
        "users",
        "using",
        "more",
        "less",
    }
)

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)")
_SINGLE_RE = re.compile(r"(\d+(?:\.\d+)?)")

_TWO_PLACES = Decimal("0.01")


def _js_to_fixed_2(value: float) -> float:
    """Reproduce ``Number(value.toFixed(2))`` from JS exactly.

    ``toFixed`` rounds the EXACT mathematical value of the double, ties toward
    the larger scaled integer. ``Decimal(value)`` is that exact value and
    ``ROUND_HALF_UP`` (away from zero) matches the tie rule for positive values
    (roi score is always positive). Python's built-in ``round`` uses bankers'
    rounding and would give different results on exact ``.5`` boundaries.
    """
    return float(Decimal(value).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP))


def _js_round(value: float) -> int:
    """Reproduce JS ``Math.round`` (round half up) as ``floor(value + 0.5)``.

    Python's built-in ``round`` is bankers' rounding, which differs on ``.5``.
    All inputs here are positive, so ``floor(value + 0.5)`` is exact.
    """
    return math.floor(value + 0.5)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalize_label(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    return _WHITESPACE_RE.sub("_", value.lower().strip())


def to_severity_score(severity: str) -> float:
    return SEVERITY_SCORES.get(normalize_label(severity, "medium"), SEVERITY_SCORES["medium"])


def severity_from_score(score: float) -> str:
    if score >= 0.9:
        return "critical"
    if score >= 0.72:
        return "high"
    if score >= 0.5:
        return "medium"
    if score >= 0.24:
        return "low"
    return "info"


def impact_score(impact: str) -> float:
    return VIDEO_IMPACT_SCORES.get(normalize_label(impact, "medium"), VIDEO_IMPACT_SCORES["medium"])


def complexity_days(complexity: str) -> float:
    return COMPLEXITY_DAYS.get(normalize_label(complexity, "medium"), COMPLEXITY_DAYS["medium"])


def complexity_token_multiplier(complexity: str) -> float:
    return COMPLEXITY_TOKEN_MULTIPLIER.get(
        normalize_label(complexity, "medium"), COMPLEXITY_TOKEN_MULTIPLIER["medium"]
    )


def tokenize(value: str) -> set[str]:
    cleaned = _NON_ALNUM_RE.sub(" ", value.lower())
    parts = _WHITESPACE_RE.split(cleaned)
    return {part for part in parts if len(part) > 3 and part not in STOP_WORDS}


def overlap_count(left: set[str], right: set[str]) -> int:
    return sum(1 for token in left if token in right)


def estimate_effort_days(estimated_effort: str, complexity: str) -> float:
    raw = (estimated_effort or "").lower().strip()
    if not raw:
        return complexity_days(complexity)

    range_match = _RANGE_RE.search(raw)
    single_match = _SINGLE_RE.search(raw)
    if range_match:
        value = (float(range_match.group(1)) + float(range_match.group(2))) / 2
    elif single_match:
        value = float(single_match.group(1))
    else:
        return complexity_days(complexity)

    days = value
    if "hour" in raw:
        days = value / 8
    elif "week" in raw:
        days = value * 5
    elif "month" in raw:
        days = value * 22
    elif "day" in raw:
        days = value

    return clamp(days, 0.5, 60)


def _match_critique_severity(
    proposal: Mapping[str, Any],
    critique_findings: Sequence[Mapping[str, Any]],
) -> tuple[str, float, str | None]:
    if not critique_findings:
        return "medium", SEVERITY_SCORES["medium"], None

    proposal_text = (
        f"{proposal.get('title', '')} "
        f"{proposal.get('description', '')} "
        f"{proposal.get('rationale', '')}"
    )
    proposal_tokens = tokenize(proposal_text)

    best_overlap = 0
    best_severity_score = 0.0
    best_severity = "medium"
    best_title: str | None = None
    severity_score_total = 0.0

    for finding in critique_findings:
        normalized_severity = normalize_label(finding.get("severity"), "medium")
        finding_severity_score = to_severity_score(normalized_severity)
        severity_score_total += finding_severity_score

        recommendation = finding.get("recommendation")
        finding_text = (
            f"{finding.get('title', '')} "
            f"{finding.get('description', '')} "
            f"{recommendation if recommendation is not None else ''}"
        )
        finding_tokens = tokenize(finding_text)
        overlap = overlap_count(proposal_tokens, finding_tokens)

        if overlap > best_overlap or (
            overlap == best_overlap and finding_severity_score > best_severity_score
        ):
            best_overlap = overlap
            best_severity_score = finding_severity_score
            best_severity = normalized_severity
            best_title = finding.get("title") or None

    if best_overlap >= 2:
        return best_severity, best_severity_score, best_title

    average_severity_score = severity_score_total / len(critique_findings)
    return severity_from_score(average_severity_score), average_severity_score, None


def priority_label(roi_score: float) -> str:
    if roi_score >= 1.75:
        return "Best ROI"
    if roi_score >= 1.25:
        return "High ROI"
    if roi_score >= 0.9:
        return "Medium ROI"
    return "Lower ROI"


def rank_fixes_by_cost_vs_impact(
    proposals: Sequence[Mapping[str, Any]],
    critique_findings: Sequence[Mapping[str, Any]] | None = None,
    output_cost_per_million_usd: float = CLAUDE_CODE_OUTPUT_COST_PER_MILLION_USD,
) -> list[dict[str, Any]]:
    """Rank fix proposals by impact-vs-cost ROI. Faithful port of the TS model.

    Returns one ranked entry per proposal, sorted by roi_score desc, then
    video_impact_score desc, then redesign_effort_days asc (stable).
    """
    findings = list(critique_findings or [])
    ranked: list[dict[str, Any]] = []
    for proposal in proposals:
        critique_severity, critique_severity_score, matched_title = _match_critique_severity(
            proposal, findings
        )
        video_impact_score = impact_score(proposal.get("impact", ""))
        effort_days = estimate_effort_days(
            proposal.get("estimated_effort", ""), proposal.get("complexity", "")
        )
        effort_score = clamp(effort_days / 20, 0.1, 1.25)
        output_tokens = max(
            8_000,
            _js_round(
                effort_days
                * TOKENS_PER_EFFORT_DAY
                * complexity_token_multiplier(proposal.get("complexity", ""))
            ),
        )
        token_cost_usd = (output_tokens / 1_000_000) * output_cost_per_million_usd
        token_cost_score = clamp(token_cost_usd / 1.5, 0.05, 1.25)
        impact_composite = 0.55 * critique_severity_score + 0.45 * video_impact_score
        cost_composite = 0.7 * effort_score + 0.3 * token_cost_score
        roi_score = impact_composite / max(0.1, cost_composite)

        ranked.append(
            {
                "proposal": dict(proposal),
                "critique_severity": critique_severity,
                "critique_severity_score": critique_severity_score,
                "matched_finding_title": matched_title,
                "video_impact_score": video_impact_score,
                "redesign_effort_days": effort_days,
                "redesign_effort_score": effort_score,
                "estimated_output_tokens": output_tokens,
                "output_cost_per_million_usd": output_cost_per_million_usd,
                "estimated_token_cost_usd": token_cost_usd,
                # roi_score field is rounded to 2 dp; priority_label uses the
                # UNROUNDED roi_score (matching the TS, which reads the local
                # variable before Number(...toFixed(2))).
                "roi_score": _js_to_fixed_2(roi_score),
                "priority_label": priority_label(roi_score),
            }
        )

    ranked.sort(
        key=lambda item: (
            -item["roi_score"],
            -item["video_impact_score"],
            item["redesign_effort_days"],
        )
    )
    return ranked


def assess_fix_ranking_confidence(
    ranked: Sequence[Mapping[str, Any]],
    critique_findings: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarise confidence in a ranking. Faithful port of the TS model."""
    findings = list(critique_findings or [])
    proposal_count = len(ranked)
    finding_count = len(findings)
    severity_diversity = len({normalize_label(f.get("severity"), "medium") for f in findings})
    matched_proposal_count = sum(1 for item in ranked if item.get("matched_finding_title"))
    matched_proposal_ratio = matched_proposal_count / proposal_count if proposal_count > 0 else 0
    if proposal_count > 1:
        roi_scores = [item["roi_score"] for item in ranked]
        roi_spread = max(roi_scores) - min(roi_scores)
    else:
        roi_spread = 0

    score = 1.0
    uncertainty_signals: list[str] = []

    if proposal_count <= 1:
        score -= 0.2
        uncertainty_signals.append("Only one proposal is available, so rank confidence is limited.")
    if finding_count == 0:
        score -= 0.5
        uncertainty_signals.append(
            "No critique findings were available; severity fallback was used."
        )
    else:
        if finding_count < 3:
            score -= 0.2
            uncertainty_signals.append("Findings are sparse, which weakens ranking evidence.")
        if severity_diversity < 2:
            score -= 0.18
            uncertainty_signals.append(
                "Severity diversity is low, so prioritization contrast is weaker."
            )
        if proposal_count > 0 and matched_proposal_ratio < 0.35:
            score -= 0.14
            uncertainty_signals.append("Few proposals directly matched specific critique findings.")
    if proposal_count > 1 and roi_spread < 0.2:
        score -= 0.16
        uncertainty_signals.append("ROI scores are tightly clustered, making order less stable.")

    score = clamp(score, 0.05, 1)
    level = "strong" if score >= 0.75 else "moderate" if score >= 0.45 else "weak"
    if level == "strong":
        summary = "Ranking signal is strong: multiple, diverse findings support this order."
    elif level == "moderate":
        summary = "Ranking signal is moderate: useful direction, but validate top items manually."
    else:
        summary = (
            "Ranking signal is weak: treat ordering as provisional and review supporting evidence."
        )

    return {
        "level": level,
        "score": score,
        "summary": summary,
        "uncertainty_signals": uncertainty_signals,
        "proposal_count": proposal_count,
        "finding_count": finding_count,
        "severity_diversity": severity_diversity,
        "matched_proposal_ratio": matched_proposal_ratio,
        "roi_spread": roi_spread,
    }
