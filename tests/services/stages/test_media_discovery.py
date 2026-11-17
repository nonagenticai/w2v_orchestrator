"""media_discovery — opt-in media-discovery search + ranking."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.schemas.media_discovery import MediaSearchResponse
from src.services.runs.run_context import RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.generic.media_discovery import MediaDiscoveryStage


def _ctx(brief: dict, prior: dict | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": brief, "title": "Acme"},
        prior_run_outputs=prior or {},
    )


def _search_resp(items: list[dict], **over: object) -> MediaSearchResponse:
    """A typed search response from partial item dicts (full wire shape:
    SearchResponse requires all five fields; each item requires the five
    identity strings, which the server's response_model guarantees)."""
    results = [{"title": "", "url": "", "media_type": "", "source": "", **item} for item in items]
    body: dict = {
        "results": results,
        "total": len(results),
        "sources_queried": [],
        "sources_failed": [],
        "search_time_ms": 5.0,
    }
    body.update(over)
    return MediaSearchResponse.model_validate(body)


@pytest.mark.asyncio
async def test_skips_when_not_requested():
    stage = MediaDiscoveryStage(client=AsyncMock())
    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=_ctx({}))


@pytest.mark.asyncio
async def test_searches_and_ranks_results():
    md = AsyncMock()
    md.search.return_value = _search_resp(
        [
            {"id": "a", "title": "Low", "quality_score": 0.2, "thumbnail_url": "t1"},
            {"id": "b", "title": "High", "quality_score": 0.9, "thumbnail_url": "t2"},
            {"id": "", "title": "no-id"},  # empty id → skipped
        ],
        sources_queried=["unsplash"],
    )
    stage = MediaDiscoveryStage(client=md)
    out = await stage.run(
        {}, ctx=_ctx({"media_discovery": {}, "value_proposition": "great widgets"})
    )
    assert [m.id for m in out.items] == ["b", "a"]  # ranked by quality desc
    assert out.query == "great widgets"
    md.search.assert_awaited_once()


# no source url


@pytest.mark.asyncio
async def test_republish_failure_is_soft(monkeypatch):
    md = AsyncMock()
    md.search.return_value = _search_resp(
        [{"id": "img", "url": "https://cdn/p.jpg", "media_type": "image", "quality_score": 0.9}]
    )
    monkeypatch.setattr(
        "src.services.stages.generic.media_discovery.republish_image",
        AsyncMock(side_effect=RuntimeError("egress blocked")),
    )
    stage = MediaDiscoveryStage(client=md)
    out = await stage.run({}, ctx=_ctx({"media_discovery": {}, "value_proposition": "x"}))
    assert out.items[0].composite_url == ""  # kept for the wizard, not composited
    assert out.items[0].composite_status == "failed"
    assert out.notes == []
    assert [d["code"] for d in out.degradations] == ["media_not_composited"]
    assert "egress blocked" in out.degradations[0]["detail"]


@pytest.mark.asyncio
async def test_search_failure_is_soft():
    md = AsyncMock()
    md.search.side_effect = RuntimeError("media-discovery down")
    stage = MediaDiscoveryStage(client=md)
    out = await stage.run({}, ctx=_ctx({"media_discovery": {}, "value_proposition": "x"}))
    assert out.items == []
    assert out.notes == []
    assert [d["code"] for d in out.degradations] == ["media_discovery_incomplete"]
    assert "media-discovery down" in out.degradations[0]["detail"]


@pytest.mark.asyncio
async def test_youtube_watch_page_is_classified_up_front_not_degraded(monkeypatch):
    """Prod run 01a09f82: media search returned a YouTube result as media_type=video with its
    WATCH-page url. The stage downloaded it, got text/html, and wrote a
    failure-looking note that flagged the stage degraded with NO code. A watch page
    is not a downloadable clip: classify it without a request, record why on the
    item, and do not degrade."""
    md = AsyncMock()
    md.search.return_value = _search_resp(
        [
            {
                "id": "yt",
                "url": "https://www.youtube.com/watch?v=abc123",
                "media_type": "video",
                "source": "youtube",
                "quality_score": 0.9,
            }
        ]
    )
    video = AsyncMock(side_effect=AssertionError("must not download a watch page"))
    monkeypatch.setattr("src.services.stages.generic.media_discovery.republish_video", video)
    stage = MediaDiscoveryStage(client=md)
    out = await stage.run({}, ctx=_ctx({"media_discovery": {}, "value_proposition": "x"}))
    item = out.items[0]
    assert item.composite_status == "not_compositable"
    assert "youtube.com watch page" in item.composite_detail
    assert item.composite_url == ""
    assert out.notes == [] and out.degradations == []
    assert [a["code"] for a in out.advisories] == ["media_not_compositable"]
    assert "youtube.com/watch" in out.advisories[0]["detail"]
    video.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_media_content_type_is_a_classification_not_a_degradation(monkeypatch):
    """An unknown host answering text/html to the video fetch is not media."""
    from src.services.visual_media import NotMediaContentError

    url = "https://cdn.example/clip"
    md = AsyncMock()
    md.search.return_value = _search_resp(
        [{"id": "v", "url": url, "media_type": "video", "quality_score": 0.5}]
    )
    monkeypatch.setattr(
        "src.services.stages.generic.media_discovery.republish_video",
        AsyncMock(side_effect=NotMediaContentError(url, "text/html", expected="video")),
    )
    out = await MediaDiscoveryStage(client=md).run(
        {}, ctx=_ctx({"media_discovery": {}, "value_proposition": "x"})
    )
    assert out.items[0].composite_status == "not_compositable"
    assert "text/html" in out.items[0].composite_detail
    assert "not a video file" in out.items[0].composite_detail
    assert out.notes == [] and out.degradations == []
    assert [a["code"] for a in out.advisories] == ["media_not_compositable"]


@pytest.mark.asyncio
async def test_failed_sources_and_missing_query_carry_a_code():
    md = AsyncMock()
    md.search.return_value = _search_resp([], sources_failed=["pexels"])
    out = await MediaDiscoveryStage(client=md).run(
        {}, ctx=_ctx({"media_discovery": {}, "value_proposition": "x"})
    )
    assert [d["code"] for d in out.degradations] == ["media_discovery_incomplete"]
    assert "pexels" in out.degradations[0]["detail"]

    no_query = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {"media_discovery": {}}},
        prior_run_outputs={},
    )
    out = await MediaDiscoveryStage(client=AsyncMock()).run({}, ctx=no_query)
    assert [d["code"] for d in out.degradations] == ["media_discovery_incomplete"]
    assert out.notes == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("brief_limit", "sent"),
    [(500, 100)],
)
async def test_brief_limit_is_clamped_to_media_discoverys_bound(
    brief_limit: object, sent: int
) -> None:
    """media-discovery's SearchRequest is ``limit: Field(ge=1, le=100)`` — 101+ is a 422."""
    md = AsyncMock()
    md.search.return_value = _search_resp([])
    stage = MediaDiscoveryStage(client=md)
    await stage.run({}, ctx=_ctx({"media_discovery": {"query": "dogs", "limit": brief_limit}}))
    assert md.search.call_args.kwargs["limit"] == sent
