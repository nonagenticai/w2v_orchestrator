from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.app_review.critique import AppCritiqueStage


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _ctx_with_inspect() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"audience": "engs", "review_lens": "speed"},
        },
        prior_run_outputs={
            "app_inspect": {
                "target_url": "http://target",
                "facts": [
                    {"id": "f1", "kind": "title", "text": "Target"},
                    {"id": "f2", "kind": "feature", "text": "Triage"},
                ],
                "competitors": [{"url": "http://c", "title": "C"}],
            }
        },
    )


@pytest.mark.asyncio
async def test_critique_returns_pros_cons_recs_with_ids() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "pros": ["Fast triage."],
            "cons": ["Sparse PM."],
            "recommendations": ["Use for issues; pair with PM tool."],
            "overall_take": "Solid for our lens.",
        }
    )
    stage = AppCritiqueStage(client=fake)

    out = await stage.run({}, ctx=_ctx_with_inspect())

    assert len(out.pros) == 1
    assert len(out.cons) == 1
    assert len(out.recommendations) == 1
    assert all(item.id for item in out.pros + out.cons + out.recommendations)
    assert out.pros[0].kind == "pro"
    assert out.cons[0].kind == "con"
    assert out.recommendations[0].kind == "recommendation"
    assert out.overall_take == "Solid for our lens."


@pytest.mark.asyncio
async def test_critique_raises_without_inspect() -> None:
    fake = AsyncMock()
    stage = AppCritiqueStage(client=fake)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    with pytest.raises(RuntimeError, match="app_inspect"):
        await stage.run({}, ctx=ctx)
