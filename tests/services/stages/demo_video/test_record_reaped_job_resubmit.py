"""A REAPED FTR job (its pod died) must be re-submitted; a real failure must not.

The defect: an FTR pod deleted mid-job leaves its row ``processing`` with nobody
writing to it. FTR's reaper (``cleanup_service.py``, ``reap_stuck_test_execution_jobs``)
flips it to ``failed`` after 30 min without a heartbeat. ``poll_job`` treats ANY
``failed`` as terminal and raises, so the orchestrator threw away a run it could
safely have re-submitted — ~324 min/week of E2E failures.

These tests pin BOTH halves of the contract, because a retry that cannot tell the
two apart is worse than the bug it fixes:

  * a reaped job IS re-submitted, under a DIFFERENT idempotency key (FTR's dedupe
    is unconditional — reusing the key hands back the same corpse);
  * a genuine execution failure is NOT re-submitted, and attempt 0's key is
    unchanged so a resumable-timeout re-dispatch still dedupes as before.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.services.stages.demo_video.record import (
    _SUBMIT_ATTEMPTS,
    DemoVideoRecordStage,
)

from .test_stages import _ctx, _passing_analyzer, _qa_settings

# The EXACT message frontend_test_runner_service's reaper stamps
# (`cleanup_service.py`: reap_stuck_test_execution_jobs), wrapped the way
# BaseServiceClient.poll_job raises it: the job's `error` field (which FTR's
# status route sets verbatim from `error_message`) inside a RuntimeError.
_REAPED = (
    "frontend_test_runner job ftr-1 failed: Job stuck (no heartbeat — producing pod likely crashed)"
)
# What JobManager.mark_failed writes for a REAL execution error, via
# _process_job's except handler: f"{type(exc).__name__}: {exc}".
_REAL_FAILURE = (
    "frontend_test_runner job ftr-1 failed: "
    "RuntimeError: package download failed: 500 from bdd_playwright"
)

_GOOD_RESULT: dict[str, Any] = {"scenario_clips": [{"clip_key": "clip-1", "duration_ms": 1000}]}


def _client(*, wait_errors: list[str | None]) -> AsyncMock:
    """FTR client whose Nth ``wait_for_execution`` raises ``wait_errors[N]``.

    A ``None`` entry means that attempt succeeds. ``start_execution`` returns a
    distinct job id per call so the test can see a genuinely NEW job, and records
    the Idempotency-Key it was handed.
    """
    client = AsyncMock()
    calls = {"n": 0}

    async def _start(_payload: object, *, idempotency_key: str | None = None) -> str:
        calls["n"] += 1
        return f"ftr-job-{calls['n']}"

    async def _wait(*_a: object, **_kw: object) -> dict[str, Any]:
        err = wait_errors[client.wait_for_execution.await_count - 1]
        if err is not None:
            raise RuntimeError(err)
        return {}

    client.start_execution.side_effect = _start
    client.wait_for_execution.side_effect = _wait
    client.get_job_result.return_value = _GOOD_RESULT
    client.get_clip_video.return_value = b"mp4"
    return client


async def _run(client: AsyncMock) -> Any:
    gen = {"bdd_job_id": "bdd-1", "lines": [{"id": "clip-1"}]}
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    with (
        patch(
            "src.services.stages.demo_video.record.get_settings",
            return_value=_qa_settings(demo_require_footage=True),
        ),
        patch(
            "src.services.stages.demo_video.record.store_capture_clip",
            new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
        ),
        # Collapse the re-submit pause (and stage_call's backoff) so the test is fast.
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        return await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://app.example"}, prior={"demo_video_gen": gen}),
        )


def _keys(client: AsyncMock) -> list[str | None]:
    return [c.kwargs.get("idempotency_key") for c in client.start_execution.await_args_list]


@pytest.mark.asyncio
async def test_reaped_job_is_resubmitted_and_the_run_survives():
    """THE regression. Without the fix the reaped RuntimeError propagates and the
    stage dies on attempt 1."""
    client = _client(wait_errors=[_REAPED, None])
    out = await _run(client)

    assert client.start_execution.await_count == 2, "the reaped job was not re-submitted"
    assert client.get_job_result.await_count == 1
    assert out.clips, "the surviving attempt's footage was not harvested"


@pytest.mark.asyncio
async def test_the_resubmit_uses_a_different_idempotency_key():
    """FTR's create_job upserts ON CONFLICT DO NOTHING over the partial unique index
    (tenant_id, idempotency_key) and returns the EXISTING row for any repeat — a
    reaped `failed` one included. An identical key would hand back the corpse, so the
    retry would be a guaranteed no-op. Attempt 0 must stay byte-identical to the
    historical key so a resume still dedupes onto the original job."""
    client = _client(wait_errors=[_REAPED, None])
    await _run(client)

    keys = _keys(client)
    assert keys[0] is not None
    assert not keys[0].endswith(":r1"), "attempt 0 must keep the historical key shape"
    assert keys[1] == f"{keys[0]}:r1"
    assert keys[0] != keys[1], "re-submit reused the key; FTR would return the dead job"


@pytest.mark.asyncio
async def test_a_genuine_execution_failure_is_never_resubmitted():
    """The constraint that matters: a real failure must NOT be retried. If this ever
    goes green with 2 submissions, the discriminator has gone blind and the retry is
    masking real defects while burning the pool."""
    client = _client(wait_errors=[_REAL_FAILURE, None])
    with pytest.raises(RuntimeError, match="package download failed"):
        await _run(client)

    assert client.start_execution.await_count == 1


@pytest.mark.asyncio
async def test_retries_are_bounded_and_exhaustion_raises_the_real_error():
    """An unbounded retry against a pool that is evicting pods is worse than failing."""
    client = _client(wait_errors=[_REAPED] * _SUBMIT_ATTEMPTS)
    with pytest.raises(RuntimeError, match="Job stuck"):
        await _run(client)

    assert client.start_execution.await_count == _SUBMIT_ATTEMPTS
    # Exhaustion must surface the FTR error itself, not a synthetic wrapper.
    assert client.get_job_result.await_count == 0
