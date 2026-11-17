"""Route test for POST /api/fixes/rank (F7).

Exercises the real app wiring (auth seed + router mount) and asserts the
response shape plus a known ranking. The exhaustive numeric parity lives in
tests/services/test_fix_ranking.py; here we confirm the endpoint threads the
request through the ported model and returns the schema-shaped response.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "fix_ranking_parity.json"


def _case(name: str) -> dict:
    cases = json.loads(_FIXTURE.read_text())["cases"]
    return next(c for c in cases if c["name"] == name)


@pytest.mark.asyncio
async def test_rank_endpoint_returns_ranked_and_confidence(client) -> None:
    case = _case("matched-proposals-diverse-severities")
    res = await client.post("/api/fixes/rank", json=case["input"])
    assert res.status_code == 200, res.text
    body = res.json()

    assert set(body) == {"ranked", "confidence"}
    ranked = body["ranked"]
    assert len(ranked) == 3

    # Known ordering (captured from the real TS oracle): proposal 1 (critical +
    # cheap) ranks first as Best ROI, and the list is roi_score-descending.
    assert ranked[0]["proposal"]["id"] == 1
    assert ranked[0]["priority_label"] == "Best ROI"
    assert ranked[0]["roi_score"] == 7.04
    assert ranked[0]["critique_severity"] == "critical"
    assert ranked[0]["matched_finding_title"] == "Checkout button contrast failure"
    roi_scores = [r["roi_score"] for r in ranked]
    assert roi_scores == sorted(roi_scores, reverse=True)

    confidence = body["confidence"]
    assert confidence["level"] == "strong"
    assert confidence["proposal_count"] == 3
    assert confidence["finding_count"] == 4


@pytest.mark.asyncio
async def test_rank_endpoint_no_findings_uses_fallback(client) -> None:
    case = _case("single-proposal-no-findings")
    res = await client.post("/api/fixes/rank", json=case["input"])
    assert res.status_code == 200, res.text
    body = res.json()
    assert len(body["ranked"]) == 1
    # Fallback severity path + single-proposal both dock confidence.
    assert body["confidence"]["finding_count"] == 0
    assert body["confidence"]["level"] in {"weak", "moderate"}
    assert any("No critique findings" in s for s in body["confidence"]["uncertainty_signals"])


@pytest.mark.asyncio
async def test_rank_endpoint_rejects_malformed_body(client) -> None:
    # proposals is required; a missing id inside a proposal must 422 at the door.
    res = await client.post("/api/fixes/rank", json={"proposals": [{"title": "x"}]})
    assert res.status_code == 422
