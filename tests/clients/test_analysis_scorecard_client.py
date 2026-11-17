"""AnalysisScorecardClient payload contract — one client, one typed method per
scorecard surface (cro/seo/marketing). The SEO advanced-analyzer keys are added
only when set, so the default-path request to analysis_service stays
byte-identical across all surfaces."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.clients.analysis_scorecard_client import (
    AnalysisScorecardClient,
    ScorecardEnrichment,
)
from src.clients.schemas.analysis_scorecard import (
    CroScoreResponse,
    MarketingScorecardResponse,
    SeoAuditResponse,
)

# A minimal, model-validatable payload (overall_score + the required grade).
_OK = {"overall_score": 1.0, "grade": "A"}


def _client(service_name: str = "seo") -> AnalysisScorecardClient:
    return AnalysisScorecardClient(
        service_name=service_name,
        base_url="http://analysis.test",
        license_token="tok",
    )


@pytest.mark.asyncio
async def test_score_cro_posts_to_cro_path(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client("cro")
    req = AsyncMock(return_value=_OK)
    monkeypatch.setattr(client, "_request", req)

    out = await client.score_cro(url="https://acme.test", include_llm=True)

    assert isinstance(out, CroScoreResponse)
    args, kwargs = req.call_args
    assert args[0] == "POST"
    assert args[1] == "/api/cro/score"
    assert kwargs["json"] == {"url": "https://acme.test", "include_llm": True}


@pytest.mark.asyncio
async def test_score_seo_posts_to_seo_path(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client("seo")
    req = AsyncMock(return_value=_OK)
    monkeypatch.setattr(client, "_request", req)

    out = await client.score_seo(url="https://acme.test", include_llm=True)

    assert isinstance(out, SeoAuditResponse)
    args, kwargs = req.call_args
    assert args[1] == "/api/seo/audit"
    assert kwargs["json"] == {"url": "https://acme.test", "include_llm": True}


@pytest.mark.asyncio
async def test_score_marketing_posts_to_marketing_path(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client("marketing")
    req = AsyncMock(return_value=_OK)
    monkeypatch.setattr(client, "_request", req)

    out = await client.score_marketing(url="https://acme.test")

    assert isinstance(out, MarketingScorecardResponse)
    assert req.call_args.args[1] == "/api/marketing/scorecard"
    assert req.call_args.kwargs["json"] == {"url": "https://acme.test", "include_llm": False}


@pytest.mark.asyncio
async def test_seo_advanced_keys_added_only_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client("seo")
    req = AsyncMock(return_value=_OK)
    monkeypatch.setattr(client, "_request", req)

    await client.score_seo(
        url="https://acme.test",
        include_llm=True,
        competitor_urls=["https://rival.test"],
        cross_page_linking=True,
        max_pages=15,
    )

    assert req.call_args.kwargs["json"] == {
        "url": "https://acme.test",
        "include_llm": True,
        "competitor_urls": ["https://rival.test"],
        "cross_page_linking": True,
        "max_pages": 15,
    }


@pytest.mark.asyncio
async def test_seo_empty_competitor_list_and_false_flags_are_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client("seo")
    req = AsyncMock(return_value=_OK)
    monkeypatch.setattr(client, "_request", req)

    await client.score_seo(
        url="https://acme.test",
        competitor_urls=[],
        cross_page_linking=False,
        max_pages=None,
    )

    # Falsy advanced inputs must not leak into the payload.
    assert req.call_args.kwargs["json"] == {"url": "https://acme.test", "include_llm": False}


# --- "great report" enrichment (analysis_service PR1-3) ---


_DETERMINISTIC = {
    "include_priority": True,
    "include_projection": True,
    "include_evidence": True,
    "include_unbucketed": True,
}


def test_from_brief_off_sends_only_the_deterministic_flags() -> None:
    # Design 2026-10-17 §2.0 / F13: priority/projection/evidence/unbucketed are
    # deterministic and always on; include_llm off adds NO WM-backed flag.
    payload = ScorecardEnrichment.from_brief(include_llm=False).as_payload()
    assert payload == _DETERMINISTIC
    for wm_flag in (
        "include_recommendations",
        "include_grounding",
        "include_exec_summary",
        "include_critic",
    ):
        assert wm_flag not in payload


def test_from_brief_on_sets_every_flag_and_intent() -> None:
    payload = ScorecardEnrichment.from_brief(include_llm=True, intent="B2B devs").as_payload()
    assert payload == {
        "include_priority": True,
        "include_projection": True,
        "include_evidence": True,
        "include_unbucketed": True,
        "include_recommendations": True,
        "include_grounding": True,
        "include_exec_summary": True,
        "include_critic": True,
        "intent": "B2B devs",
    }


def test_all_off_enrichment_emits_no_keys() -> None:
    # NEG control for the "add-only-when-set" contract: every flag off → {}.
    assert ScorecardEnrichment().as_payload() == {}


def test_empty_intent_is_omitted() -> None:
    # A blank intent must not leak a key (falsy → skipped).
    assert "intent" not in ScorecardEnrichment(include_priority=True, intent="").as_payload()


@pytest.mark.asyncio
async def test_enrichment_none_keeps_request_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client("cro")
    req = AsyncMock(return_value=_OK)
    monkeypatch.setattr(client, "_request", req)

    await client.score_cro(url="https://acme.test", include_llm=False, enrichment=None)

    # No enrichment → today's exact two-key payload.
    assert req.call_args.kwargs["json"] == {"url": "https://acme.test", "include_llm": False}


@pytest.mark.asyncio
async def test_enrichment_flags_forwarded_to_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client("marketing")
    req = AsyncMock(return_value=_OK)
    monkeypatch.setattr(client, "_request", req)

    await client.score_marketing(
        url="https://acme.test",
        include_llm=True,
        enrichment=ScorecardEnrichment.from_brief(include_llm=True, intent="devs"),
    )

    assert req.call_args.kwargs["json"] == {
        "url": "https://acme.test",
        "include_llm": True,
        "include_priority": True,
        "include_projection": True,
        "include_evidence": True,
        "include_unbucketed": True,
        "include_recommendations": True,
        "include_grounding": True,
        "include_exec_summary": True,
        "include_critic": True,
        "intent": "devs",
    }


@pytest.mark.asyncio
async def test_sources_added_only_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client("cro")
    req = AsyncMock(return_value=_OK)
    monkeypatch.setattr(client, "_request", req)

    # None/empty sources → no key (default request byte-identical)
    await client.score_cro(url="https://acme.test", sources=None)
    assert "sources" not in req.call_args.kwargs["json"]
    await client.score_cro(url="https://acme.test", sources=[])
    assert "sources" not in req.call_args.kwargs["json"]

    # Present sources → forwarded verbatim
    await client.score_cro(url="https://acme.test", sources=["corpus text"])
    assert req.call_args.kwargs["json"]["sources"] == ["corpus text"]


@pytest.mark.asyncio
async def test_http_timeout_covers_the_longest_scan_budget() -> None:
    """The httpx read timeout must not undercut the scan stages' per-attempt budget.

    Live 2026-10-18: an AI-analysis marketing scorecard took ~61s server-side, and
    the inherited 30s ``BaseServiceClient`` default raised ReadTimeout twice
    (StageRetriesExhaustedError) although the stage allowed 60s per attempt.
    """
    from src.services.stages.cro_audit import scan as cro_scan
    from src.services.stages.marketing_scorecard import scan as mkt_scan
    from src.services.stages.seo_audit import scan as seo_scan

    longest = max(
        cro_scan._LLM_SCORE_TIMEOUT_SECONDS,
        mkt_scan._LLM_SCORE_TIMEOUT_SECONDS,
        seo_scan._ADVANCED_AUDIT_TIMEOUT_SECONDS,
    )
    client = _client("marketing")
    http = await client._get_client()
    try:
        assert http.timeout.read is not None
        assert http.timeout.read >= longest
    finally:
        await http.aclose()
