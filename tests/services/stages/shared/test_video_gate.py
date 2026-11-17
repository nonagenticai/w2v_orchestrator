"""Per-output opt-in gates for website_review (critique / redesign / video).

Each output is an independent brief flag; the stages skip themselves
(StageSkippedError → "skipped") when their output wasn't requested. Absent
defaults reproduce the original full pipeline. The only coupling is
video⟶redesign (the narrated review needs a redesign to show).
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.growth_bundle.handler import GrowthBundleStage
from src.services.stages.shared.video_gate import (
    critique_requested,
    redesign_requested,
    require_critique,
    require_redesign,
    require_video,
    video_requested,
)
from src.services.stages.website_review.compile import WebsiteReviewCompileStage
from src.services.stages.website_review.critique import WebsiteCritiqueStage
from src.services.stages.website_review.record import WebsiteReviewRecordStage
from src.services.stages.website_review.redesign import WebsiteRedesignStage
from src.services.stages.website_review.script import WebsiteReviewScriptStage
from src.services.stages.website_review.slides import WebsiteReviewSlidesStage
from src.services.stages.website_review.visual_director import WebsiteReviewVisualDirectorStage


def _ctx(brief: dict) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": brief, "title": "Acme"},
        prior_run_outputs={},
    )


# ── gate predicates ──────────────────────────────────────────────────────────


def test_critique_requested_defaults_on():
    assert critique_requested(_ctx({})) is True
    assert critique_requested(_ctx({"generate_critique": False})) is False


def test_video_requested_defaults_on():
    assert video_requested(_ctx({})) is True
    assert video_requested(_ctx({"generate_video": False})) is False


def test_redesign_requested_off_by_default_but_implied_by_video():
    assert redesign_requested(_ctx({"generate_video": False})) is False  # neither
    assert redesign_requested(_ctx({"generate_redesign": True, "generate_video": False})) is True
    assert redesign_requested(_ctx({"generate_video": True})) is True  # video implies redesign
    assert redesign_requested(_ctx({})) is True  # absent video defaults on → redesign on


def test_require_helpers_raise_only_when_not_requested():
    require_critique(_ctx({}))  # absent → on
    with pytest.raises(StageSkippedError):
        require_critique(_ctx({"generate_critique": False}))
    require_redesign(_ctx({"generate_redesign": True, "generate_video": False}))
    with pytest.raises(StageSkippedError):
        require_redesign(_ctx({"generate_video": False}))
    require_video(_ctx({}))
    with pytest.raises(StageSkippedError):
        require_video(_ctx({"generate_video": False}))


# ── stage-level skips ────────────────────────────────────────────────────────

# Every video-subgraph stage skips when neither redesign nor video is requested.
_VIDEO_STAGES = [
    WebsiteRedesignStage,
    WebsiteReviewScriptStage,
    WebsiteReviewRecordStage,
    WebsiteReviewCompileStage,
    WebsiteReviewVisualDirectorStage,
    WebsiteReviewSlidesStage,
    GrowthBundleStage,
]


@pytest.mark.parametrize("stage_cls", [WebsiteRedesignStage])
@pytest.mark.asyncio
async def test_video_stage_skips_when_redesign_and_video_off(stage_cls):
    stage = stage_cls(client=AsyncMock())
    ctx = _ctx({"generate_video": False, "generate_redesign": False})
    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=ctx)


@pytest.mark.asyncio
async def test_critique_stage_skips_when_critique_off():
    stage = WebsiteCritiqueStage(client=AsyncMock())
    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=_ctx({"generate_critique": False, "target_url": "https://x.test"}))
