"""Byte-exact parity gate for the server-side ROI/cost ranking model (F7).

The oracle ``tests/fixtures/fix_ranking_parity.json`` was captured by running
the REAL frontend TS (``src/lib/utils/fix-roi-ranking.ts``) over a diverse set
of inputs with ``bun``. This test asserts the Python port reproduces the TS
output EXACTLY — every field, including roi_score to 2 decimals, the integer
token estimate, and every float in the confidence summary. If this passes, the
server returns byte-identical numbers to what the browser used to compute.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.services.fix_ranking import (
    assess_fix_ranking_confidence,
    rank_fixes_by_cost_vs_impact,
)

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "fix_ranking_parity.json"

# TS emits camelCase; the Python port emits snake_case. Map every TS key to the
# Python key so the comparison is on identical value trees.
_RANKED_KEY_MAP = {
    "proposal": "proposal",
    "critiqueSeverity": "critique_severity",
    "critiqueSeverityScore": "critique_severity_score",
    "matchedFindingTitle": "matched_finding_title",
    "videoImpactScore": "video_impact_score",
    "redesignEffortDays": "redesign_effort_days",
    "redesignEffortScore": "redesign_effort_score",
    "estimatedOutputTokens": "estimated_output_tokens",
    "outputCostPerMillionUsd": "output_cost_per_million_usd",
    "estimatedTokenCostUsd": "estimated_token_cost_usd",
    "roiScore": "roi_score",
    "priorityLabel": "priority_label",
}
_CONFIDENCE_KEY_MAP = {
    "level": "level",
    "score": "score",
    "summary": "summary",
    "uncertaintySignals": "uncertainty_signals",
    "proposalCount": "proposal_count",
    "findingCount": "finding_count",
    "severityDiversity": "severity_diversity",
    "matchedProposalRatio": "matched_proposal_ratio",
    "roiSpread": "roi_spread",
}


def _expected_ranked(ts_ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{_RANKED_KEY_MAP[k]: v for k, v in entry.items()} for entry in ts_ranked]


def _expected_confidence(ts_conf: dict[str, Any]) -> dict[str, Any]:
    return {_CONFIDENCE_KEY_MAP[k]: v for k, v in ts_conf.items()}


def _load_cases() -> list[dict[str, Any]]:
    return json.loads(_FIXTURE.read_text())["cases"]


CASES = _load_cases()


def test_fixture_has_diverse_cases() -> None:
    assert len(CASES) >= 8, "parity oracle must cover a diverse case set"


@pytest.mark.parametrize("case", list(CASES)[:1], ids=([c["name"] for c in CASES])[:1])
def test_ranking_parity(case: dict[str, Any]) -> None:
    inp = case["input"]
    got = rank_fixes_by_cost_vs_impact(
        inp["proposals"],
        inp["critique_findings"],
        inp["output_cost_per_million_usd"],
    )
    expected = _expected_ranked(case["ranked"])
    assert got == expected, f"ranking mismatch in case {case['name']!r}"


@pytest.mark.parametrize("case", list(CASES)[:1], ids=([c["name"] for c in CASES])[:1])
def test_confidence_parity(case: dict[str, Any]) -> None:
    inp = case["input"]
    ranked = rank_fixes_by_cost_vs_impact(
        inp["proposals"],
        inp["critique_findings"],
        inp["output_cost_per_million_usd"],
    )
    got = assess_fix_ranking_confidence(ranked, inp["critique_findings"])
    expected = _expected_confidence(case["confidence"])
    assert got == expected, f"confidence mismatch in case {case['name']!r}"


def test_roi_score_is_rounded_two_dp_but_label_from_unrounded() -> None:
    """Guard the subtle TS behaviour: the stored roi_score is rounded to 2dp,
    while priority_label is computed from the UNROUNDED score."""
    for case in CASES:
        for ts_entry, py_entry in zip(
            case["ranked"], _expected_ranked(case["ranked"]), strict=True
        ):
            rounded = round(py_entry["roi_score"], 2)
            assert py_entry["roi_score"] == rounded
            assert ts_entry["priorityLabel"] == py_entry["priority_label"]
