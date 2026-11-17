"""Regression: script_draft must FAIL LOUD when the LLM returns an empty `lines`
array (instead of silently substituting synthetic lines).

An empty `lines` list would otherwise map downstream to zero shots and a 422 from
video-compile (`shots` field required, ge=1). Rather than ship a fabricated script,
the stage raises ``StageError`` so the run fails with a clear reason.

See bug-bash B1+B2 (2026-05-09).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageError
from src.services.stages.generic.script_draft import ScriptDraftStage


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _brief_ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={
            "brief_intake": {
                "title": "T",
                "angle": "growth",
                "audience": "saas",
                "target_duration_seconds": 30,
                "voice_descriptor": "crisp",
            }
        },
    )


@pytest.mark.asyncio
async def test_script_draft_raises_when_llm_returns_empty_lines() -> None:
    """LLM returning `{"lines":[]}` must fail loud, not synthesize lines."""
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": []})
    stage = ScriptDraftStage(client=fake)

    with pytest.raises(StageError):
        await stage.run({}, ctx=_brief_ctx())


@pytest.mark.asyncio
async def test_script_draft_raises_when_llm_returns_missing_lines_key() -> None:
    """LLM returning `{}` must fail loud (KeyError path → StageError)."""
    fake = AsyncMock()
    fake.message.return_value = _resp({})
    stage = ScriptDraftStage(client=fake)

    with pytest.raises(StageError):
        await stage.run({}, ctx=_brief_ctx())
