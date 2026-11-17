from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from w2v_common.stages.errors import StageRetriesExhaustedError, StageTimeoutError

from src.clients.schemas.critique import CritiqueResult
from src.services.runs.run_context import RunContext
from src.services.stages.website_review.critique import WebsiteCritiqueStage


def _ctx(target: str | None = "https://acme.test") -> RunContext:
    brief: dict = {}
    if target is not None:
        brief["target_url"] = target
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": brief},
    )


@pytest.mark.asyncio
async def test_critique_emits_findings_and_job_id() -> None:
    fake = AsyncMock()
    fake.start_critique.return_value = "job-1"
    fake.wait_for_critique.return_value = CritiqueResult.model_validate(
        {
            "job_id": "job-1",
            "status": "completed",
            "url": "https://acme.test",
            "overall_score": 72.5,
            "summary": "Decent but slow.",
            "category_scores": {"seo": 80.0, "performance": 55.0},
            "findings": [
                {
                    "finding_id": "F1",
                    "category": "seo",
                    "severity": "high",
                    "title": "Missing meta description",
                    "description": "No meta tag.",
                    "recommendation": "Add one.",
                }
            ],
        }
    )
    stage = WebsiteCritiqueStage(client=fake)

    out = await stage.run({}, ctx=_ctx())

    assert out.critique_job_id == "job-1"
    assert out.url == "https://acme.test"
    assert out.overall_score == 72.5
    assert out.category_scores["performance"] == 55.0
    assert len(out.findings) == 1
    assert out.findings[0].id == "F1"
    assert out.findings[0].recommendation == "Add one."
    fake.start_critique.assert_awaited_once()


@pytest.mark.asyncio
async def test_critique_raises_without_target_url() -> None:
    fake = AsyncMock()
    stage = WebsiteCritiqueStage(client=fake)

    with pytest.raises(RuntimeError, match="target_url"):
        await stage.run({}, ctx=_ctx(target=None))


@pytest.mark.asyncio
async def test_critique_omits_auth_info_when_brief_has_none() -> None:
    """Absent brief.auth_info → no auth_info kwarg on start_critique
    (so analysis_service falls through to its existing unauthenticated path).
    """
    fake = AsyncMock()
    fake.start_critique.return_value = "job-3"
    fake.wait_for_critique.return_value = CritiqueResult.model_validate(
        {
            "job_id": "job-3",
            "status": "completed",
            "url": "https://acme.test",
            "overall_score": 80.0,
            "summary": "",
            "category_scores": {},
            "findings": [],
        }
    )
    stage = WebsiteCritiqueStage(client=fake)

    await stage.run({}, ctx=_ctx())

    call = fake.start_critique.await_args
    assert "auth_info" not in call.kwargs


@pytest.mark.asyncio
async def test_critique_wait_fails_immediately_on_terminal_job_failure() -> None:
    """A critique job the service already marked `failed` must fail the stage on the
    first poll with the service's own error text — not be re-polled (retry_on=())."""
    fake = AsyncMock()
    fake.start_critique.return_value = "job-dead"
    terminal = RuntimeError("critique job job-dead failed: scan crashed")
    fake.wait_for_critique.side_effect = terminal
    stage = WebsiteCritiqueStage(client=fake)

    with pytest.raises(RuntimeError, match="scan crashed") as exc_info:
        await stage.run({}, ctx=_ctx())

    assert not isinstance(exc_info.value, StageRetriesExhaustedError)
    assert exc_info.value is terminal
    assert fake.wait_for_critique.await_count == 1


@pytest.mark.asyncio
async def test_critique_wait_still_retries_on_poll_timeout() -> None:
    """retry_on=() must not eat the poll-timeout retry: a still-running critique keeps
    its 2 poll attempts (stage_call's TimeoutError branch ignores retry_on)."""
    fake = AsyncMock()
    fake.start_critique.return_value = "job-slow"
    fake.wait_for_critique.side_effect = TimeoutError
    stage = WebsiteCritiqueStage(client=fake)

    with (
        patch("w2v_common.stages.execution.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(StageTimeoutError),
    ):
        await stage.run({}, ctx=_ctx())

    assert fake.wait_for_critique.await_count == 2  # 1 initial + 1 retry
