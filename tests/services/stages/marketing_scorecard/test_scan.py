from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.schemas.analysis_scorecard import MarketingScorecardResponse
from src.services.runs.run_context import RunContext
from src.services.stages.marketing_scorecard.scan import MarketingScorecardScanStage

_RESULT = {
    "overall_score": 78.5,
    "grade": "C",
    "llm_enriched": False,
    "dimension_scores": {
        "content": 90.0,
        "conversion": 80.0,
        "seo": 70.0,
        "competitive": 75.0,
        "brand": 85.0,
        "growth": 60.0,
    },
    "dimensions": [
        {"dimension": "content", "score": 90.0, "weight": 0.25, "count": 0, "top_severity": "info"},
        {
            "dimension": "growth",
            "score": 60.0,
            "weight": 0.10,
            "count": 2,
            "top_severity": "medium",
        },
    ],
    "quick_wins": [
        {
            "check_id": "mkt-growth-no-pricing-visibility",
            "dimension": "growth",
            "title": "No visible pricing",
            "severity": "medium",
        }
    ],
    "strategic": [],
    "llm_assessments": [],
    "source": "https://acme.test",
    "scored_at": "2026-07-05T00:00:00Z",
}


def _ctx(
    target: str | None = "https://acme.test",
    include_llm: bool = False,
    intent: str | None = None,
    grounding_text: str | None = None,
) -> RunContext:
    brief: dict = {}
    if target is not None:
        brief["target_url"] = target
    if include_llm:
        brief["include_llm"] = True
    if intent is not None:
        brief["intent"] = intent
    prior = (
        {"source_ingest": {"grounding_text": grounding_text}} if grounding_text is not None else {}
    )
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": brief},
        prior_run_outputs=prior,
    )


@pytest.mark.asyncio
async def test_scan_emits_composite_and_dimensions() -> None:
    fake = AsyncMock()
    fake.score_marketing.return_value = MarketingScorecardResponse.model_validate(_RESULT)
    stage = MarketingScorecardScanStage(client=fake)

    out = await stage.run({}, ctx=_ctx())

    assert out.overall_score == 78.5
    assert out.grade == "C"
    assert out.url == "https://acme.test"
    assert out.dimension_scores["growth"] == 60.0
    assert len(out.dimensions) == 2
    assert out.dimensions[0].dimension == "content"
    assert out.quick_wins[0].check_id == "mkt-growth-no-pricing-visibility"
    fake.score_marketing.assert_awaited_once()
    assert fake.score_marketing.await_args.kwargs["include_llm"] is False


@pytest.mark.asyncio
async def test_scan_forwards_include_llm_and_assessments() -> None:
    fake = AsyncMock()
    fake.score_marketing.return_value = MarketingScorecardResponse.model_validate(
        {
            **_RESULT,
            "llm_enriched": True,
            "llm_assessments": [
                {
                    "element": "headline",
                    "dimension": "content",
                    "score": 80.0,
                    "assessment": "ok",
                    "suggestion": "tighten",
                }
            ],
        }
    )
    stage = MarketingScorecardScanStage(client=fake)

    out = await stage.run({}, ctx=_ctx(include_llm=True))

    assert out.llm_enriched is True
    assert len(out.llm_assessments) == 1
    assert out.llm_assessments[0].element == "headline"
    assert out.llm_assessments[0].dimension == "content"
    assert fake.score_marketing.await_args.kwargs["include_llm"] is True


# --- "great report" enrichment threading (analysis_service PR1-3) ---


@pytest.mark.asyncio
async def test_scan_threads_enrichment_when_include_llm() -> None:
    fake = AsyncMock()
    fake.score_marketing.return_value = MarketingScorecardResponse.model_validate(
        {
            **_RESULT,
            "quick_wins": [
                {
                    "check_id": "mkt-growth-no-pricing-visibility",
                    "dimension": "growth",
                    "title": "No visible pricing",
                    "severity": "medium",
                    "rank": 1,
                    "priority": 6.0,
                    "evidence": "no pricing block",
                    "selector": ".pricing",
                    "recommendation": {
                        "fix_prose": "Show a pricing table above the fold.",
                        "rationale": "Buyers self-qualify on price.",
                        "example": "$29/mo",
                        "effort": "low",
                    },
                }
            ],
            "projection": {
                "baseline": 78.5,
                "projected": 84.0,
                "delta": 5.5,
                "removed": ["mkt-growth-no-pricing-visibility"],
            },
            "grounding": [
                {
                    "check_id": "mkt-growth-no-pricing-visibility",
                    "supported": True,
                    "evidence_ref": "no price element",
                }
            ],
            "exec_summary": {
                "summary": "Growth lags the other dimensions.",
                "so_what": "Surfacing pricing is the fastest win.",
                "roadmap": ["mkt-growth-no-pricing-visibility"],
            },
            "critique": {
                "score": 82.0,
                "generic_check_ids": [],
                "ungrounded_check_ids": [],
                "rerun_stages": [],
            },
        }
    )
    stage = MarketingScorecardScanStage(client=fake)

    out = await stage.run({}, ctx=_ctx(include_llm=True, intent="B2B devs"))

    # enrichment is forwarded to the client (flags on + intent resolved from the brief)
    enr = fake.score_marketing.await_args.kwargs["enrichment"]
    assert enr.include_recommendations is True
    assert enr.include_grounding is True
    assert enr.intent == "B2B devs"
    # every response block is mapped onto the scan output
    assert out.quick_wins[0].rank == 1
    assert out.quick_wins[0].priority == 6.0
    assert out.quick_wins[0].evidence == "no pricing block"
    assert out.quick_wins[0].recommendation is not None
    assert out.quick_wins[0].recommendation.fix_prose == "Show a pricing table above the fold."
    assert out.projection is not None and out.projection.delta == 5.5
    assert out.grounding is not None and out.grounding[0].supported is True
    assert out.exec_summary is not None and out.exec_summary.summary.startswith("Growth lags")
    assert out.critique is not None and out.critique.score == 82.0


@pytest.mark.asyncio
async def test_scan_carries_no_enrichment_when_include_llm_off() -> None:
    # include_llm off → only the deterministic flags (design §2.0) and no AI field.
    fake = AsyncMock()
    fake.score_marketing.return_value = MarketingScorecardResponse.model_validate(_RESULT)
    stage = MarketingScorecardScanStage(client=fake)

    out = await stage.run({}, ctx=_ctx(include_llm=False))

    enr = fake.score_marketing.await_args.kwargs["enrichment"]
    assert enr.as_payload() == {
        "include_priority": True,
        "include_projection": True,
        "include_evidence": True,
        "include_unbucketed": True,
    }
    assert out.degradations == []
    assert out.exec_summary is None
    assert out.projection is None
    assert out.grounding is None
    assert out.critique is None
    assert out.quick_wins[0].recommendation is None
    assert out.quick_wins[0].rank is None


@pytest.mark.asyncio
async def test_scan_sends_no_sources_without_source_ingest() -> None:
    # No brief.sources → source_ingest skipped → no grounding_text → sources None
    # (default request byte-identical).
    fake = AsyncMock()
    fake.score_marketing.return_value = MarketingScorecardResponse.model_validate(_RESULT)
    stage = MarketingScorecardScanStage(client=fake)

    await stage.run({}, ctx=_ctx(include_llm=True))

    assert fake.score_marketing.await_args.kwargs["sources"] is None


@pytest.mark.asyncio
async def test_scan_gates_sources_on_include_llm() -> None:
    # Grounding sources are only forwarded when include_llm (→ include_grounding) is
    # on. With include_llm off, a brief.sources run must NOT POST a payload the
    # deterministic-only scorer can't consume (avoids a wasted scrape+payload).
    fake = AsyncMock()
    fake.score_marketing.return_value = MarketingScorecardResponse.model_validate(_RESULT)
    stage = MarketingScorecardScanStage(client=fake)

    await stage.run({}, ctx=_ctx(include_llm=False, grounding_text="INGESTED SOURCE CORPUS"))

    assert fake.score_marketing.await_args.kwargs["sources"] is None
