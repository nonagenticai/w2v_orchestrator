"""growth_bundle blueprint — registration + standalone execution.

The growth_bundle stage is already exercised by
``tests/services/stages/growth_bundle/test_handler.py``; here we assert the NEW
standalone product type is registered and that the stage runs with no prior
stages (the standalone path: ``script`` prior-output degrades to ``{}``) and the
``require_video`` gate passing when ``generate_video`` is left unset.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.blueprints.growth_bundle import GROWTH_BUNDLE_BLUEPRINT
from src.services.blueprints.registry import get_registry
from src.services.runs.run_context import RunContext
from src.services.stages.errors import StageSkippedError
from src.services.stages.growth_bundle.handler import GrowthBundleStage


def test_registered() -> None:
    assert get_registry().resolve("growth_bundle") is GROWTH_BUNDLE_BLUEPRINT
    assert GROWTH_BUNDLE_BLUEPRINT.id == "growth_bundle_v1"
    assert GROWTH_BUNDLE_BLUEPRINT.project_type == "growth_bundle"


def test_single_stage_no_artifact_gate() -> None:
    bp = GROWTH_BUNDLE_BLUEPRINT
    assert bp.topological_order() == ["growth_bundle"]
    assert bp.stage("growth_bundle").depends_on == ()
    # Emits text snippets, not an artifact_url → no run-level artifact gate.
    assert bp.terminal_artifact_stage is None


def test_outer_timeout_clears_llm_ladder() -> None:
    # LLM-budget stage: outer ceiling must sit above the 660s inner budget
    # (see test_llm_timeout_ladder). Guard against a tighter value sneaking in.
    from src.config import get_settings
    from src.services.stages.llm_budget import llm_budget_seconds

    # None => inherits the env-configurable base ceiling; resolve it to the value the
    # executor would actually use, then assert the ladder still clears the inner budget.
    spec_outer = GROWTH_BUNDLE_BLUEPRINT.stage("growth_bundle").default_timeout_seconds
    outer = (
        spec_outer if spec_outer is not None else get_settings().pipeline.llm_stage_base_ceiling_s
    )
    assert outer >= llm_budget_seconds() + 30.0


def _wm_response(output: dict) -> WMResponse:
    return WMResponse(
        output=output,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _standalone_ctx() -> RunContext:
    """A standalone growth_bundle run: NO prior stages, generate_video unset."""
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "Anti-Churn",
            "brief": {"call_to_action": "Book a demo"},
            "voice": {"descriptor": "crisp"},
        },
        prior_run_outputs={},
    )


@pytest.mark.asyncio
async def test_runs_standalone_without_prior_stages() -> None:
    fake = AsyncMock()
    fake.message.return_value = _wm_response(
        {
            "linkedin": {"text": "L", "cta_url": "https://x"},
            "email": {"text": "E", "cta_url": None},
        }
    )
    stage = GrowthBundleStage(client=fake)

    out = await stage.run({}, ctx=_standalone_ctx())

    assert {s.channel for s in out.snippets} == {"linkedin", "email"}
    assert all(s.id for s in out.snippets)
    # No prior 'promo_script'/'course_outline' → the prompt's "Prior content" is {}.
    prompt = fake.message.await_args.kwargs["message"]
    assert "Prior content: {}" in prompt


@pytest.mark.asyncio
async def test_skips_when_generate_video_false() -> None:
    fake = AsyncMock()
    stage = GrowthBundleStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"title": "x", "brief": {"generate_video": False}, "voice": {}},
        prior_run_outputs={},
    )

    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=ctx)
    fake.message.assert_not_awaited()
