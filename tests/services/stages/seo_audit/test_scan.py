from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.schemas.analysis_scorecard import SeoAuditResponse
from src.services.runs.run_context import RunContext
from src.services.stages.base import StagePreconditionError
from src.services.stages.seo_audit.scan import SeoAuditScanStage

_RESULT = {
    "overall_score": 78.5,
    "grade": "B",
    "llm_enriched": False,
    "group_scores": {"on_page": 90.0, "technical": 70.0},
    "groups": [
        {"group": "on_page", "score": 90.0, "weight": 0.22, "count": 1, "top_severity": "low"},
        {"group": "technical", "score": 70.0, "weight": 0.18, "count": 2, "top_severity": "medium"},
    ],
    "quick_wins": [
        {
            "check_id": "seo-og-tags-missing",
            "group": "technical",
            "title": "OG tags",
            "severity": "medium",
        }
    ],
    "strategic": [],
    "source": "https://acme.test",
    "scored_at": "2026-07-05T00:00:00Z",
}


def _ctx(
    target: str | None = "https://acme.test",
    include_llm: bool = False,
    *,
    competitor_urls: list[str] | None = None,
    cross_page_linking: bool = False,
    max_pages: int | None = None,
) -> RunContext:
    brief: dict = {}
    if target is not None:
        brief["target_url"] = target
    if include_llm:
        brief["include_llm"] = True
    if competitor_urls is not None:
        brief["competitor_urls"] = competitor_urls
    if cross_page_linking:
        brief["cross_page_linking"] = True
    if max_pages is not None:
        brief["max_pages"] = max_pages
    return RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={"brief": brief})


@pytest.mark.asyncio
async def test_scan_emits_score_and_groups() -> None:
    fake = AsyncMock()
    fake.score_seo.return_value = SeoAuditResponse.model_validate(_RESULT)
    stage = SeoAuditScanStage(client=fake)

    out = await stage.run({}, ctx=_ctx())

    assert out.overall_score == 78.5
    assert out.grade == "B"
    assert out.url == "https://acme.test"
    assert out.group_scores["technical"] == 70.0
    assert len(out.groups) == 2
    assert out.groups[0].group == "on_page"
    assert out.quick_wins[0].check_id == "seo-og-tags-missing"
    fake.score_seo.assert_awaited_once()
    kwargs = fake.score_seo.await_args.kwargs
    assert kwargs["include_llm"] is False
    # Default (plain) path: the advanced analyzers are off → not requested.
    assert kwargs["competitor_urls"] is None
    assert kwargs["cross_page_linking"] is False
    assert kwargs["max_pages"] is None


@pytest.mark.asyncio
async def test_scan_forwards_competitor_and_cross_page() -> None:
    fake = AsyncMock()
    fake.score_seo.return_value = SeoAuditResponse.model_validate(_RESULT)
    stage = SeoAuditScanStage(client=fake)

    await stage.run(
        {},
        ctx=_ctx(
            include_llm=True,
            competitor_urls=["https://rival.test", "  ", "https://rival2.test"],
            cross_page_linking=True,
            max_pages=10,
        ),
    )

    kwargs = fake.score_seo.await_args.kwargs
    # Blank entries are trimmed out before forwarding.
    assert kwargs["competitor_urls"] == ["https://rival.test", "https://rival2.test"]
    assert kwargs["cross_page_linking"] is True
    assert kwargs["max_pages"] == 10


@pytest.mark.asyncio
async def test_scan_ignores_non_int_max_pages() -> None:
    # A malformed max_pages (e.g. a stringly-typed brief value) is dropped, not passed.
    fake = AsyncMock()
    fake.score_seo.return_value = SeoAuditResponse.model_validate(_RESULT)
    stage = SeoAuditScanStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {"target_url": "https://acme.test", "max_pages": "oops"}},
    )
    await stage.run({}, ctx=ctx)
    assert fake.score_seo.await_args.kwargs["max_pages"] is None


@pytest.mark.asyncio
async def test_scan_forwards_include_llm() -> None:
    fake = AsyncMock()
    fake.score_seo.return_value = SeoAuditResponse.model_validate({**_RESULT, "llm_enriched": True})
    stage = SeoAuditScanStage(client=fake)

    out = await stage.run({}, ctx=_ctx(include_llm=True))

    assert out.llm_enriched is True
    assert fake.score_seo.await_args.kwargs["include_llm"] is True


@pytest.mark.asyncio
async def test_scan_raises_without_target_url() -> None:
    stage = SeoAuditScanStage(client=AsyncMock())
    with pytest.raises(StagePreconditionError, match="target_url"):
        await stage.run({}, ctx=_ctx(target=None))


@pytest.mark.asyncio
async def test_scan_threads_and_maps_enrichment() -> None:
    fake = AsyncMock()
    fake.score_seo.return_value = SeoAuditResponse.model_validate(
        {
            **_RESULT,
            "quick_wins": [
                {
                    "check_id": "seo-og-tags-missing",
                    "group": "technical",
                    "title": "OG tags",
                    "severity": "medium",
                    "rank": 2,
                    "recommendation": {"fix_prose": "Add OpenGraph title + image tags."},
                }
            ],
            "grounding": [
                {"check_id": "seo-og-tags-missing", "supported": True, "evidence_ref": "no og:*"}
            ],
        }
    )
    stage = SeoAuditScanStage(client=fake)

    out = await stage.run({}, ctx=_ctx(include_llm=True))

    assert fake.score_seo.await_args.kwargs["enrichment"].include_grounding is True
    assert out.quick_wins[0].rank == 2
    assert out.quick_wins[0].recommendation is not None
    assert out.quick_wins[0].recommendation.fix_prose == "Add OpenGraph title + image tags."
    assert out.grounding is not None and out.grounding[0].supported is True


@pytest.mark.asyncio
async def test_scan_no_enrichment_when_include_llm_off() -> None:
    fake = AsyncMock()
    fake.score_seo.return_value = SeoAuditResponse.model_validate(_RESULT)
    stage = SeoAuditScanStage(client=fake)

    out = await stage.run({}, ctx=_ctx(include_llm=False))

    # Deterministic flags are always on (design §2.0); no WM-backed flag is sent.
    assert fake.score_seo.await_args.kwargs["enrichment"].as_payload() == {
        "include_priority": True,
        "include_projection": True,
        "include_evidence": True,
        "include_unbucketed": True,
    }
    assert out.degradations == []
    assert out.grounding is None
    assert out.quick_wins[0].recommendation is None
    assert out.quick_wins[0].rank is None
