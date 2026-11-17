"""app_review_script: the LLM contract — schema, id links, and the error paths.

The narration rate is stubbed for the whole module. Nothing here is about
durations, and unstubbed these tests reach the REAL tts backend twice over (the
Phase-A probe, then a full-script measurement per draft): measured 2026-09-13
they took 120s of pure network timeout, and on an in-cluster CI runner the
"timeout" becomes a live synthesis nobody asked for.
"""

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

import src.services.stages.app_review.script as app_review_script
from src.services.runs.run_context import RunContext
from src.services.stages.app_review.script import AppReviewScriptStage
from tests.services.stages.app_review.rate_stubs import stub_speech_rate


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


def _ctx_with_priors() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"duration_seconds": 10},
            "voice": {"descriptor": "fast"},
        },
        prior_run_outputs={
            "app_inspect": {"facts": [{"id": "f1", "kind": "title", "text": "T"}]},
            "app_critique": {
                "pros": [{"id": "p1", "text": "fast"}],
                "cons": [{"id": "c1", "text": "sparse"}],
                "recommendations": [{"id": "r1", "text": "ship"}],
                "overall_take": "ok",
            },
        },
    )


@pytest.mark.asyncio
async def test_output_schema_names_fact_and_critique_ids() -> None:
    # Regression: WM only emits schema-named props, so the link fields must be
    # declared or fact_id/critique_item_id are silently dropped from output.
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": [{"text": "hi", "seconds": 2.0}]})
    await AppReviewScriptStage(client=fake).run({}, ctx=_ctx_with_priors())
    _, kwargs = fake.message.call_args
    props = kwargs["output_schema"]["properties"]["lines"]["items"]["properties"]
    assert {"text", "seconds", "fact_id", "critique_item_id"} <= set(props)


@pytest.mark.asyncio
async def test_script_lines_carry_fact_or_critique_ids() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "lines": [
                {
                    "text": "Fast triage.",
                    "seconds": 5.0,
                    "fact_id": "f1",
                    "critique_item_id": None,
                },
                {
                    "text": "Pros vs cons.",
                    "seconds": 5.0,
                    "fact_id": None,
                    "critique_item_id": "p1",
                },
            ]
        }
    )
    stage = AppReviewScriptStage(client=fake)

    out = await stage.run({}, ctx=_ctx_with_priors())

    assert len(out.lines) == 2
    assert all(line.id for line in out.lines)
    assert out.lines[0].fact_id == "f1"
    assert out.lines[1].critique_item_id == "p1"
    assert out.total_seconds == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_script_raises_without_inspect_or_critique() -> None:
    fake = AsyncMock()
    stage = AppReviewScriptStage(client=fake)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    with pytest.raises(RuntimeError):
        await stage.run({}, ctx=ctx)
