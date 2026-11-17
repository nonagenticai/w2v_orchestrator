"""Motion-render phase is self-bounded with graceful degradation.

The motion clips render inside a stage with its own timeout (e.g. promo_render =
2500s). Before this, a deck with many slides on the single throughput-limited
video-production pod blew that budget and HARD-FAILED the whole stage — the
~9-12 motion-slide cap that forced SECTION mode. ``render_deck_motion_clips``
now bounds the phase to ``phase_budget_s``: clips that finish in time get their
motion video; the rest gracefully keep their still PNG (absent from the returned
map). The stage always completes — no hard cap, no TimeoutError.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import pytest

from src.services.stages.shared import motion_slides

if TYPE_CHECKING:
    from src.clients.video_production_client import VideoProductionClient


def _deck(n: int) -> tuple[list[str], list[dict[str, Any]]]:
    line_ids = [f"line-{i}" for i in range(n)]
    deck_slides = [{"content": {"title": f"Slide {i}", "subtitle": "s"}} for i in range(n)]
    return line_ids, deck_slides


class _FakeVideoClient:
    """video_production client stub. Clips with index < ``fast_below`` resolve
    immediately; the rest hang forever (simulating a renderer that can't keep up),
    so the phase budget is what bounds them."""

    def __init__(self, fast_below: int) -> None:
        self.fast_below = fast_below

    async def record_animation(self, *, part_id: str, **_kw: Any) -> str:
        # part_id is f"slide-{line_id}" → recover the index from line-{i}
        idx = int(part_id.rsplit("-", 1)[-1])
        return f"job-{idx}"

    async def wait_for_content_recording(self, job_id: str, timeout: float) -> None:
        idx = int(job_id.rsplit("-", 1)[-1])
        if idx >= self.fast_below:
            await asyncio.Event().wait()  # never resolves → bounded only by the phase budget

    async def get_recording_video(self, job_id: str) -> tuple[bytes, str, str]:
        return (b"mp4-bytes", "video/mp4", "inline")


@pytest.fixture(autouse=True)
def _stub_store(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_store(*, video_id: str, **_kw: Any) -> str:
        return f"https://minio.test/{video_id}.mp4"

    monkeypatch.setattr(motion_slides, "store_generated_video", _fake_store)


async def _run(client: _FakeVideoClient, n: int, budget: float) -> motion_slides.DeckMotionResult:
    line_ids, deck_slides = _deck(n)
    # The stub implements only the few methods render_deck_motion_clips calls;
    # cast to the real type (ty rejects a structural fake for a concrete annotation).
    return await motion_slides.render_deck_motion_clips(
        cast("VideoProductionClient", client),
        line_ids=line_ids,
        deck_slides=deck_slides,
        deck_brand={},
        motion_cfg={},
        deliverable={"format": "16x9"},
        run_id="run-1",
        stage_name="promo_slides",
        phase_budget_s=budget,
    )


async def test_all_clips_render_within_budget() -> None:
    # Every clip is fast → all rendered, full map, no degradation.
    result = await _run(_FakeVideoClient(fast_below=8), n=8, budget=30.0)
    assert set(result.urls.keys()) == set(range(8))
    assert all(url for url in result.urls.values())
    assert result.budget_exceeded is False
    assert result.failed_indices == ()


async def test_phase_budget_degrades_gracefully_no_hard_fail() -> None:
    # 10 slides, only the first 2 are fast; the rest hang. A tiny budget must
    # return the completed clips (graceful) instead of raising — the slides whose
    # clips didn't finish are simply absent and keep their still PNG.
    result = await _run(_FakeVideoClient(fast_below=2), n=10, budget=0.4)
    assert set(result.urls.keys()) == {0, 1}
    assert len(result.urls) < 10  # the cap is graceful, not a hard failure
    # The drop is now diagnosable: the phase hit its budget and the missing
    # indices are reported so a notes-carrying caller can degrade the run.
    assert result.budget_exceeded is True
    assert result.failed_indices == tuple(range(2, 10))


async def test_empty_deck_returns_empty() -> None:
    result = await _run(_FakeVideoClient(fast_below=0), n=0, budget=1.0)
    assert result.urls == {}
    assert result.budget_exceeded is False
    assert result.failed_indices == ()
