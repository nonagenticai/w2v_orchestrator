"""Verbatim injection + verification tests for the app_review_script stage.

The narration rate is stubbed for the whole module. Nothing here is about
durations, and unstubbed these tests reach the REAL tts backend twice over (the
Phase-A probe, then a full-script measurement per draft) — 60s of timeouts on a
laptop, and on an in-cluster CI runner a live synthesis nobody asked for.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

import src.services.stages.app_review.script as app_review_script
from src.services.runs.run_context import RunContext
from src.services.stages.app_review.script import AppReviewScriptStage
from src.services.stages.base import StageError
from tests.services.stages.app_review.rate_stubs import stub_speech_rate

_PRIOR = {
    "app_inspect": {"facts": ["fast", "clean UI"]},
    "app_critique": {"pros": ["pro1"], "cons": ["con1"], "recommendations": ["rec1"]},
}


@pytest.fixture(autouse=True)
def _no_tts(monkeypatch: pytest.MonkeyPatch) -> None:
    """One known rate, no backend — see the module docstring."""
    stub_speech_rate(monkeypatch, app_review_script, 150.0)


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _ctx(verbatim_items: list[dict]) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"duration_seconds": 120},
            "voice": {"descriptor": "neutral"},
            "verbatim_items": verbatim_items,
        },
        prior_run_outputs=_PRIOR,
    )


@pytest.mark.asyncio
async def test_app_review_script_verbatim_present_succeeds_and_injects_prompt() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "lines": [
                {
                    "text": "Download it today, you will love it.",
                    "seconds": 3.0,
                    "fact_id": None,
                    "critique_item_id": None,
                }
            ]
        }
    )
    stage = AppReviewScriptStage(client=fake)
    out = await stage.run({}, ctx=_ctx([{"text": "Download it today"}]))
    assert out.lines[0].text.startswith("Download it today")
    _, kwargs = fake.message.call_args
    assert "Download it today" in kwargs["message"]


@pytest.mark.asyncio
async def test_app_review_script_verbatim_missing_raises() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "lines": [
                {
                    "text": "It is a fine app.",
                    "seconds": 3.0,
                    "fact_id": None,
                    "critique_item_id": None,
                }
            ]
        }
    )
    stage = AppReviewScriptStage(client=fake)
    with pytest.raises(StageError):
        await stage.run({}, ctx=_ctx([{"text": "Download it today"}]))
