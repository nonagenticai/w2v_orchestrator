from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.llm_budget import MIN_OUTPUT_TOKEN_BUDGET
from src.services.stages.promo.analyze import _ANALYZE_MAX_TOKENS, PromoAnalyzeStage


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "Anti-churn",
            "brief": {
                "angle": "growth",
                "audience": "saas",
                "value_proposition": "vp",
                "call_to_action": "cta",
                "target_duration_seconds": 45,
            },
            "voice": {"descriptor": "crisp"},
        },
    )


@pytest.mark.asyncio
async def test_analyze_emits_typed_insights() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "pain_points": ["high CAC", "long sales cycles"],
            "hook_candidates": ["Stop bleeding cash on dead leads."],
            "tone_hints": ["urgent", "founder-to-founder"],
        }
    )
    stage = PromoAnalyzeStage(client=fake)

    out = await stage.run({}, ctx=_ctx())

    assert len(out.pain_points) == 2
    assert len(out.hook_candidates) == 1
    assert all(p.id for p in out.pain_points)  # element ids minted
    assert all(h.id for h in out.hook_candidates)
    assert out.tone_hints == ["urgent", "founder-to-founder"]


@pytest.mark.asyncio
async def test_analyze_falls_back_to_defaults_on_non_dict_output() -> None:
    """Tolerant stage: a non-dict WM output yields seeded defaults, never a raise."""
    fake = AsyncMock()
    fake.message.return_value = _resp(None)
    stage = PromoAnalyzeStage(client=fake)

    out = await stage.run({}, ctx=_ctx())

    assert [p.text for p in out.pain_points] == ["unspecified pain"]
    assert [h.text for h in out.hook_candidates] == ["Anti-churn"]  # cfg title fallback
    assert out.tone_hints == ["crisp"]  # voice descriptor fallback


@pytest.mark.asyncio
async def test_analyze_calls_promo_analyze_worker() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp({"pain_points": [], "hook_candidates": [], "tone_hints": []})
    stage = PromoAnalyzeStage(client=fake)

    await stage.run({}, ctx=_ctx())

    _, kwargs = fake.message.call_args
    assert kwargs["worker_name"] == "promo-analyze"
    assert isinstance(kwargs["message"], str) and kwargs["message"]


@pytest.mark.asyncio
async def test_analyze_passes_explicit_max_tokens_floor() -> None:
    """Pins the EXPLICIT ``max_tokens`` so the omission cannot come back.

    This test previously pinned the omission, on the reasoning that promo-analyze emits
    ~138 tokens so WM's 6000 default is never binding. That is about ANSWER SIZE only:
    the served model is a reasoning model whose thinking tokens are charged against the
    same budget, and exhausting it yields a WM 422 -> PermanentUpstreamError -> terminal
    stage failure. The budget is a floor (``stages/llm_budget.py``), never a cap.
    """
    fake = AsyncMock()
    fake.message.return_value = _resp({"pain_points": [], "hook_candidates": [], "tone_hints": []})
    stage = PromoAnalyzeStage(client=fake)

    await stage.run({}, ctx=_ctx())

    args, kwargs = fake.message.call_args
    assert not args, "WMClient.message must be called with keyword args only"
    assert kwargs["max_tokens"] == _ANALYZE_MAX_TOKENS
    assert kwargs["max_tokens"] >= MIN_OUTPUT_TOKEN_BUDGET
