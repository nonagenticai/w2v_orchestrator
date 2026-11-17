"""Defect 1 mutation-proof tests (2026-09-17 incident).

The orchestrator used to abandon a compile job the compiler itself still
considered healthy: ``_COMPILE_TIMEOUT_SECONDS`` (1200s) + a 120s stage_call
backstop gave up at 1320s, well inside the compiler's own worst case (a single
ffmpeg subprocess is allowed 1800s, plus reaper redispatch on top —
``video_compiler_service`` ``Settings.reclaim_max_lifetime_seconds`` caps the
whole thing at 7200s). These tests simulate ELAPSED WALL CLOCK by patching the
running loop's own ``time()`` — no real sleeping (``asyncio.sleep`` is already
a no-op fixture, matching ``tests/unit/test_video_compile_poll.py``'s
convention) — so a "1320s+ has passed" scenario runs in milliseconds.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx
import pytest

from src.clients.video_compile import _MAX_LOST_POLLS, VideoCompileClient
from src.services.stages.primitives.compile_stage import _COMPILE_TIMEOUT_SECONDS

pytestmark = pytest.mark.asyncio

_IDEM_KEY = "11111111-1111-7111-8111-111111111111:demo_video_compile"

# The OLD deadline this defect shipped with (compile_stage.py, pre-fix):
# _COMPILE_TIMEOUT_SECONDS=1200.0, stage_call timeout_seconds=+120 => 1320.0.
# Production incident 2026-09-17, compile job 1602: "attempt 1/1 timed out
# after 1320.0s" on a compile the compiler was still heartbeating.
_OLD_STAGE_CALL_TIMEOUT_SECONDS = 1320.0


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse retry/backoff sleeps so the poll loop runs instantly."""

    async def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _noop)


class _FakeClock:
    """A fake ``loop.time()`` that advances by ``step`` seconds per call.

    Lets ``_await_job``'s ``loop.time() >= deadline`` check see elapsed wall
    clock without any real sleeping.
    """

    def __init__(self, step: float) -> None:
        self._t = 0.0
        self._step = step

    def __call__(self) -> float:
        self._t += self._step
        return self._t


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> VideoCompileClient:
    client = VideoCompileClient(base_url="http://compile.test", license_token="t")
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        timeout=client.timeout,
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    return client


def _processing_then_completed(
    n_processing: int,
) -> Callable[[httpx.Request], httpx.Response]:
    """POST -> job-1; the first ``n_processing`` GETs answer 'processing',
    every GET after that answers 'completed'."""
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"job_id": "job-1"})
        state["i"] += 1
        if state["i"] <= n_processing:
            return httpx.Response(200, json={"job_id": "job-1", "status": "processing"})
        return httpx.Response(
            200,
            json={
                "job_id": "job-1",
                "status": "completed",
                "artifact_url": "https://m.test/w2v-compiled-videos/out.mp4",
                "duration_seconds": 10.0,
            },
        )

    return handler


async def test_a_job_healthy_past_the_old_1320s_ceiling_is_not_abandoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) A job that stays 'processing' (advancing liveness — every poll gets a
    fresh 200) for a SIMULATED 2200s — comfortably past the OLD 1320s
    stage_call ceiling that shipped the incident — must NOT be abandoned. The
    new ceiling is bound to the compiler's own worst case, not a number picked
    on this side.

    MUTATION PROOF: reverting ``_COMPILE_TIMEOUT_SECONDS`` to the old 1200.0
    turns this red (``_await_job`` raises ``TimeoutError`` once simulated time
    crosses 1320s, well before this test's 22nd poll).
    """
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "time", _FakeClock(step=100.0))

    # 22 x 100s = 2200s of simulated elapsed time before the job completes.
    assert _OLD_STAGE_CALL_TIMEOUT_SECONDS < 22 * 100.0
    client = _client(_processing_then_completed(n_processing=22))
    try:
        result = await client.compile(
            shots=[], timeout=_COMPILE_TIMEOUT_SECONDS, idempotency_key=_IDEM_KEY
        )
    finally:
        await client.close()

    assert result["status"] == "completed"


async def test_liveness_loss_is_abandoned_within_a_few_polls_not_the_full_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) A job whose liveness genuinely STOPS (every poll 404s — the durable
    job store has no record of it, matching ``_JobLostError``'s contract) must
    still be abandoned PROMPTLY: within ``_MAX_LOST_POLLS`` polls per attempt
    (initial + one resubmit), never anywhere near riding out the new, much
    larger ``_COMPILE_TIMEOUT_SECONDS`` ceiling. Raising the ceiling to cover a
    HEALTHY compile's worst case must not turn a genuinely dead job into an
    hours-long hang.

    MUTATION PROOF: removing the 404-specific branch in
    ``video_compile._poll_status_grace`` (so a 404 falls through to the
    generic transient-status grace, `_MAX_SERVER_ERROR_POLLS=5`, or — with
    that also removed — to the raw wall-clock deadline) makes this test's
    poll-count assertion fail; it currently bounds strictly on the
    liveness-loss counters, not on elapsed time.
    """
    loop = asyncio.get_running_loop()
    # A tiny step: if bailing here secretly depended on the wall-clock deadline
    # (rather than the 404-loss counters), it would take ~75 iterations to
    # reach the new ~7500s ceiling at 100s/step. It doesn't get anywhere close.
    monkeypatch.setattr(loop, "time", _FakeClock(step=1.0))

    posts = {"n": 0}
    gets = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            posts["n"] += 1
            return httpx.Response(200, json={"job_id": f"job-{posts['n']}"})
        gets["n"] += 1
        return httpx.Response(404, json={"detail": "job not found"})

    client = _client(handler)
    try:
        with pytest.raises(RuntimeError, match="lost"):
            await client.compile(
                shots=[], timeout=_COMPILE_TIMEOUT_SECONDS, idempotency_key=_IDEM_KEY
            )
    finally:
        await client.close()

    assert posts["n"] == 2  # original submit + exactly one resubmit, then give up
    assert gets["n"] == 2 * _MAX_LOST_POLLS  # bounded by the loss counter, not the clock
