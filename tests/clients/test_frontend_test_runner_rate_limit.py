"""FrontendTestRunnerClient honours a 429 ``Retry-After`` on the execute POST.

This is the consumer-side proof for w2v-common 0.58.2. The measured failure was
22 production runs (burst 2026-07-20..24) where ``demo_video_record``'s
``POST /api/test-runner/execute`` met a 429 rate-limit with a 60-SECOND window
and burned its whole retry budget in ~3 seconds: 3 attempts on a linear 1s
ladder (1s + 2s), then surfaced the 429 and failed a paid recording stage.

The library fix is inert unless three things hold HERE, which is what these
assert against an ``httpx.MockTransport``:

1. The POST is retryable at all. ``BaseServiceClient`` excludes POST from
   retries by default (replaying a partial write double-creates). It opts in
   ONLY when an ``Idempotency-Key`` is supplied — which ``demo_video_record``
   does (``f"{ctx.run_id}:{self.name}"``, record.py). Drop that key and every
   other assertion here is unreachable in production, so the key is carried
   through ``start_execution`` in each test rather than mocked away.
2. A server-supplied ``Retry-After`` is actually honoured, not just the new
   longer built-in ladder. Asserted with a header LARGER than the ladder step
   so the two sources are distinguishable.
3. The 429 budget outlasts a 60s rate-limit window end-to-end.

``asyncio.sleep`` is patched to record its ARGUMENT rather than to elapse: the
delay the client asks for is the behaviour under test, and a test that actually
waited 75s would be its own bug.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from src.clients.frontend_test_runner_client import FrontendTestRunnerClient

pytestmark = pytest.mark.asyncio

EXECUTE = "/api/test-runner/execute"
# Shape produced by demo_video_record: f"{ctx.run_id}:{self.name}".
IDEMPOTENCY_KEY = "run-abc:demo_video_record"


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record every retry delay the client ASKS for, without elapsing it."""
    recorded: list[float] = []

    async def _record(delay: float, *_a: object, **_k: object) -> None:
        recorded.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _record)
    return recorded


def _client(handler) -> FrontendTestRunnerClient:
    client = FrontendTestRunnerClient(base_url="http://ftr.test", license_token="t")
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        timeout=client.timeout,
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    return client


def _rate_limited_then_ok(retry_after: str | None, *, keys: list[str | None], limit_first: int = 1):
    """429 (optionally with ``Retry-After``) for the first N POSTs, then 200."""

    def handler(request: httpx.Request) -> httpx.Response:
        keys.append(request.headers.get("Idempotency-Key"))
        if len(keys) <= limit_first:
            headers = {"Retry-After": retry_after} if retry_after is not None else {}
            return httpx.Response(429, json={"detail": "rate limited"}, headers=headers)
        return httpx.Response(200, json={"job_id": "ftr-job-1"})

    return handler


async def test_retry_after_5s_replaces_the_old_1s_ladder(slept: list[float]) -> None:
    """The regression case: a 429 + ``Retry-After: 5`` now waits 5s, not 1s."""
    keys: list[str | None] = []
    client = _client(_rate_limited_then_ok("5", keys=keys))

    job_id = await client.start_execution({"brief": {}}, idempotency_key=IDEMPOTENCY_KEY)

    assert job_id == "ftr-job-1", "the retried execute POST must still yield the job id"
    assert slept == [5.0], f"expected a single 5s wait, got {slept} (1.0 == unfixed client)"
    assert keys == [IDEMPOTENCY_KEY, IDEMPOTENCY_KEY], (
        "the replay must carry the SAME Idempotency-Key or FTR starts a second paid run"
    )
    await client.close()


async def test_retry_after_longer_than_the_ladder_wins(slept: list[float]) -> None:
    """Proves the HEADER is read, not merely the new 5s built-in ladder.

    ``Retry-After: 30`` on the first retry outranks the ladder's 5s step. Were
    the header ignored, this would still sleep 5.0 and pass the test above —
    which is why that test alone cannot distinguish the two mechanisms.
    """
    keys: list[str | None] = []
    client = _client(_rate_limited_then_ok("30", keys=keys))

    await client.start_execution({"brief": {}}, idempotency_key=IDEMPOTENCY_KEY)

    assert slept == [30.0], f"server-supplied Retry-After must win over the ladder, got {slept}"
    await client.close()


async def test_retry_after_never_makes_the_client_more_aggressive(slept: list[float]) -> None:
    """``Retry-After: 0`` is floored at the ladder — no free amplification."""
    keys: list[str | None] = []
    client = _client(_rate_limited_then_ok("0", keys=keys))

    await client.start_execution({"brief": {}}, idempotency_key=IDEMPOTENCY_KEY)

    assert slept == [5.0], f"a 0/negative Retry-After must floor at the ladder, got {slept}"
    await client.close()


async def test_429_budget_outlasts_a_60_second_window(slept: list[float]) -> None:
    """End-to-end: an unrelenting 429 is now ridden out past a 60s window.

    The old budget spent 1s + 2s = 3s against a 60s window and gave up. The
    ladder must now total more than 60s so a full window can elapse.
    """
    keys: list[str | None] = []
    # Never recovers: exercises the complete ladder.
    client = _client(_rate_limited_then_ok(None, keys=keys, limit_first=99))

    with pytest.raises(httpx.HTTPStatusError):
        await client.start_execution({"brief": {}}, idempotency_key=IDEMPOTENCY_KEY)

    assert slept == [5.0, 10.0, 15.0, 20.0, 25.0], f"unexpected 429 ladder: {slept}"
    assert sum(slept) > 60.0, "the 429 budget must outlast a full 60s rate-limit window"
    await client.close()


async def test_without_an_idempotency_key_the_post_is_not_retried(slept: list[float]) -> None:
    """The key is load-bearing: without it the execute POST stays one-shot.

    Guards the wiring in ``demo_video_record`` — if that key is ever dropped,
    the library fix silently stops applying to this call site.
    """
    keys: list[str | None] = []
    client = _client(_rate_limited_then_ok("5", keys=keys))

    with pytest.raises(httpx.HTTPStatusError):
        await client.start_execution({"brief": {}})

    assert slept == [], "a keyless POST must not be replayed"
    assert keys == [None], "expected exactly one attempt"
    await client.close()
