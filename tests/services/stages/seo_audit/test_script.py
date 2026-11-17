from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.errors import StageSkippedError
from src.services.stages.seo_audit.script import SeoAuditScriptStage

_SCAN = {
    "url": "https://acme.test",
    "overall_score": 72.0,
    "grade": "C",
    "groups": [{"group": "on_page", "score": 90.0}, {"group": "technical", "score": 60.0}],
    "quick_wins": [
        {"check_id": "seo-og-tags-missing", "title": "Add OpenGraph tags", "severity": "medium"},
        {"check_id": "seo-alt-text", "title": "Add image alt text", "severity": "low"},
    ],
    "strategic": [{"title": "Build topical authority"}],
}


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _ctx(*, generate_video: bool = True, scan: dict | None = _SCAN) -> RunContext:
    prior = {"seo_audit_scan": scan} if scan is not None else {}
    brief: dict = {"target_url": "https://acme.test"}
    if generate_video:
        brief["generate_video"] = True
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": brief, "voice": {"descriptor": "clear"}},
        prior_run_outputs=prior,
    )


@pytest.mark.asyncio
async def test_script_skips_when_video_off() -> None:
    fake = AsyncMock()
    stage = SeoAuditScriptStage(client=fake)
    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=_ctx(generate_video=False))
    fake.message.assert_not_called()


@pytest.mark.asyncio
async def test_script_builds_lines_from_scan() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "lines": [
                {"text": "Your site scored 72, a grade C.", "seconds": 4.0},
                {"text": "First, add OpenGraph tags for better sharing.", "seconds": 5.0},
                {"text": "Start improving today.", "seconds": 3.0},
            ]
        }
    )
    stage = SeoAuditScriptStage(client=fake)

    out = await stage.run({}, ctx=_ctx())

    assert len(out.lines) == 3
    assert all(line.id for line in out.lines)
    assert out.total_seconds == pytest.approx(12.0)
    # The scan facts flow into the prompt.
    _, kwargs = fake.message.call_args
    prompt = kwargs["message"]
    assert "72" in prompt and "grade C" in prompt
    assert "Add OpenGraph tags" in prompt
    assert "Build topical authority" in prompt
    # Default kebab worker name.
    assert kwargs["worker_name"] == "seo-audit-script"


@pytest.mark.asyncio
async def test_script_raises_without_scan() -> None:
    fake = AsyncMock()
    stage = SeoAuditScriptStage(client=fake)
    with pytest.raises(RuntimeError, match="seo_audit_scan"):
        await stage.run({}, ctx=_ctx(scan=None))
