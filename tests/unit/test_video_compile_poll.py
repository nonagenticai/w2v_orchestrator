"""VideoCompileClient._await_job resilience.

The poll loop must survive the compile service rolling under it without failing
the stage, and must honour the full terminal contract (completed /
completed_with_errors / partial = success). These drive the client through an
``httpx.MockTransport`` whose handler scripts the POST submit then a sequence of
poll responses (or raised transport errors).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx
import pytest

from src.clients.video_compile import (
    _MAX_LOST_POLLS,
    _MAX_SERVER_ERROR_POLLS,
    VideoCompileClient,
)
from src.services.stages.upstream import (
    PermanentUpstreamError,
    TransientUpstreamError,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse retry/backoff sleeps so the poll loop runs instantly."""

    async def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _noop)


_IDEM_KEY = "11111111-1111-7111-8111-111111111111:demo_video_compile"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> VideoCompileClient:
    client = VideoCompileClient(base_url="http://compile.test", license_token="t")
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        timeout=client.timeout,
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    return client


def _scripted(
    poll_steps: list[httpx.Response | Exception],
) -> Callable[[httpx.Request], httpx.Response]:
    """POST → submit (job_id); each GET pops the next step (clamped to the last).

    A step that is an ``Exception`` is raised (transport error); a ``Response``
    is returned (including 4xx, which ``_request`` turns into an
    ``HTTPStatusError``).
    """
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"job_id": "job-1"})
        step = poll_steps[min(state["i"], len(poll_steps) - 1)]
        state["i"] += 1
        if isinstance(step, Exception):
            raise step
        return step

    return handler


def _completed(status: str = "completed") -> httpx.Response:
    return httpx.Response(
        200,
        json={"status": status, "artifact_url": "http://x/y.mp4", "duration_seconds": 1.0},
    )


async def test_transient_transport_error_is_retried_not_fatal() -> None:
    """A sustained-but-transient blip (past _request's own retries) keeps polling."""
    # 3 raised GETs exhaust _request's connection-reset retries for one poll;
    # the next poll succeeds — the stage must not fail.
    steps: list[httpx.Response | Exception] = [
        httpx.ReadError("connection reset"),
        httpx.ReadError("connection reset"),
        httpx.ReadError("connection reset"),
        _completed(),
    ]
    client = _client(_scripted(steps))
    try:
        result = await client.compile(shots=[], timeout=30.0, idempotency_key=_IDEM_KEY)
    finally:
        await client.close()
    assert result["status"] == "completed"


async def test_bounded_404_grace_then_success() -> None:
    """A couple of 404s (poll hit a not-yet-warm replica) are tolerated."""
    steps: list[httpx.Response | Exception] = [
        httpx.Response(404, json={"detail": "job not found"}),
        httpx.Response(404, json={"detail": "job not found"}),
        _completed(),
    ]
    client = _client(_scripted(steps))
    try:
        result = await client.compile(shots=[], timeout=30.0, idempotency_key=_IDEM_KEY)
    finally:
        await client.close()
    assert result["status"] == "completed"


async def test_persistent_404_below_floor_declares_lost_no_resubmit() -> None:
    """Past the grace window with too little budget left, the job is declared lost
    (fail fast, not hang) and is NOT resubmitted (timeout 30s < resubmit floor)."""
    posts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            posts["n"] += 1
            return httpx.Response(200, json={"job_id": f"job-{posts['n']}"})
        return httpx.Response(404, json={"detail": "gone"})

    client = _client(handler)
    try:
        with pytest.raises(RuntimeError, match="lost"):
            await client.compile(shots=[], timeout=30.0, idempotency_key=_IDEM_KEY)
    finally:
        await client.close()
    assert _MAX_LOST_POLLS >= 1
    assert posts["n"] == 1  # the loss was re-raised, not resubmitted


async def test_confirmed_loss_resubmits_once_and_succeeds() -> None:
    """A confirmed loss with budget to spare resubmits exactly once: the second
    job (a fresh job_id) is polled to completion. The lost first job produced
    nothing, so the single resubmit cannot duplicate output."""
    posts = {"n": 0}
    polled: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            posts["n"] += 1
            return httpx.Response(200, json={"job_id": f"job-{posts['n']}"})
        polled.append(request.url.path)
        # While only the first job exists, every poll 404s (pod wiped its store);
        # once the resubmit lands (2nd POST), the new job completes.
        if posts["n"] >= 2:
            return _completed()
        return httpx.Response(404, json={"detail": "gone"})

    client = _client(handler)
    try:
        result = await client.compile(shots=[], timeout=300.0, idempotency_key=_IDEM_KEY)
    finally:
        await client.close()
    assert result["status"] == "completed"
    assert posts["n"] == 2  # exactly one resubmit
    assert polled[-1].endswith("/job-2")  # success came from the new job


async def test_second_consecutive_loss_propagates() -> None:
    """Bounded to one resubmit: if the resubmitted job is also lost, the loss
    propagates (no infinite resubmit loop)."""
    posts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            posts["n"] += 1
            return httpx.Response(200, json={"job_id": f"job-{posts['n']}"})
        return httpx.Response(404, json={"detail": "gone"})  # always lost

    client = _client(handler)
    try:
        with pytest.raises(RuntimeError, match="lost"):
            await client.compile(shots=[], timeout=300.0, idempotency_key=_IDEM_KEY)
    finally:
        await client.close()
    assert posts["n"] == 2  # original + exactly one resubmit, then give up


@pytest.mark.parametrize("status", ["completed_with_errors"])
async def test_partial_terminal_status_is_success(status: str) -> None:
    """A with-errors / partial compile still produced an artifact — not a hang."""
    client = _client(_scripted([_completed(status)]))
    try:
        result = await client.compile(shots=[], timeout=30.0, idempotency_key=_IDEM_KEY)
    finally:
        await client.close()
    assert result["status"] == status


async def test_failed_status_raises() -> None:
    client = _client(
        _scripted([httpx.Response(200, json={"status": "failed", "error": "ffmpeg boom"})])
    )
    try:
        with pytest.raises(RuntimeError, match="ffmpeg boom"):
            await client.compile(shots=[], timeout=30.0, idempotency_key=_IDEM_KEY)
    finally:
        await client.close()


@pytest.mark.parametrize("method", ["compile"])
async def test_submit_carries_idempotency_key(method: str) -> None:
    """The submit POST sends an Idempotency-Key so a conn-reset replay is deduped
    server-side instead of starting a second render."""
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            captured["key"] = request.headers.get("Idempotency-Key")
            return httpx.Response(200, json={"job_id": "j1"})
        return _completed()

    client = _client(handler)
    try:
        if method == "compile":
            await client.compile(shots=[], timeout=30.0, idempotency_key=_IDEM_KEY)
        else:
            await client.concat_audio(
                clips=[],
                timeout=30.0,
                idempotency_key="11111111-1111-7111-8111-111111111111:audiobook_stitch:combined",
            )
    finally:
        await client.close()
    assert captured.get("key")  # present and non-empty


# ── a 5xx mid-poll: bounded in-loop retry, permanent still aborts ─────────────
async def test_transient_5xx_mid_poll_is_retried_not_fatal() -> None:
    """A compile pod that is UP but unhappy (503 while a replica warms) must not
    abort a render that is still going to finish.

    Before: the poll loop retried only ``TransportError``; a 503 escaped
    ``_request``'s own 3-attempt status budget and killed the whole wait — which
    the caller cannot safely retry (``stage_call(retries=0)``), so the render was
    lost. After: the wait survives and returns the completed artifact."""
    # THREE consecutive 503s exhaust ``_request``'s OWN 3-attempt retryable-status
    # budget for one poll — anything fewer never reaches the poll loop's handler and
    # would test the wrong layer entirely. The next poll then succeeds.
    steps: list[httpx.Response | Exception] = [
        httpx.Response(503, json={"detail": "warming up"}),
        httpx.Response(503, json={"detail": "warming up"}),
        httpx.Response(503, json={"detail": "warming up"}),
        _completed(),
    ]
    client = _client(_scripted(steps))
    try:
        result = await client.compile(shots=[], timeout=300.0, idempotency_key=_IDEM_KEY)
    finally:
        await client.close()
    assert result["status"] == "completed"


async def test_persistent_5xx_is_bounded_and_re_raises_the_status() -> None:
    """The in-loop 5xx grace is BOUNDED: a compiler answering 503 forever must fail
    with the compiler's own status long before the 300s render budget, and must
    never be resubmitted (a 5xx is not a confirmed loss)."""
    posts = {"n": 0}
    polls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            posts["n"] += 1
            return httpx.Response(200, json={"job_id": "job-1"})
        polls["n"] += 1
        return httpx.Response(503, json={"detail": "still broken"})

    client = _client(handler)
    try:
        with pytest.raises(TransientUpstreamError) as excinfo:
            await client.compile(shots=[], timeout=300.0, idempotency_key=_IDEM_KEY)
    finally:
        await client.close()
    # The CLASSIFIED type reaches the caller (#870: a 503 queue blip, a 402 billing
    # stop and a 422 contract error must not arrive as one undifferentiated
    # HTTPStatusError) — and the compiler's real status still survives underneath it
    # as __cause__, so this is not a synthesised TimeoutError the submit/poll wrapper
    # would have to treat as "may still be running".
    cause = excinfo.value.__cause__
    assert isinstance(cause, httpx.HTTPStatusError)
    assert cause.response.status_code == 503
    # BOUNDED, exactly: each in-loop round is 3 attempts deep inside _request's own
    # status budget, and the loop allows _MAX_SERVER_ERROR_POLLS rounds. Asserting the
    # product (not just an upper bound) pins BOTH ends — that the grace exists at all,
    # and that it stops well before the 300s render deadline.
    assert polls["n"] == _MAX_SERVER_ERROR_POLLS * 3
    assert posts["n"] == 1  # a 5xx is NOT a confirmed loss ⇒ never resubmitted


async def test_permanent_status_mid_poll_aborts_immediately() -> None:
    """A permanent status (403 — the license token was rejected) is not a blip:
    it must abort on the FIRST poll, spending no grace at all."""
    polls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"job_id": "job-1"})
        polls["n"] += 1
        return httpx.Response(403, json={"detail": "forbidden"})

    client = _client(handler)
    try:
        with pytest.raises(PermanentUpstreamError) as excinfo:
            await client.compile(shots=[], timeout=300.0, idempotency_key=_IDEM_KEY)
    finally:
        await client.close()
    # Classified, per #870 — "this will never work" must be distinguishable from
    # "try again" at the caller. The 403 remains reachable as __cause__.
    cause = excinfo.value.__cause__
    assert isinstance(cause, httpx.HTTPStatusError)
    assert cause.response.status_code == 403
    assert polls["n"] == 1  # zero grace: one poll, then out


async def test_a_successful_poll_resets_the_5xx_grace() -> None:
    """The grace counts CONSECUTIVE failures. A long render that hiccups repeatedly
    (503, ok, 503, ok, ...) far more than ``_MAX_SERVER_ERROR_POLLS`` times in total
    must still complete — otherwise the bound would cap total render length."""
    seq: list[httpx.Response] = []
    for _ in range(_MAX_SERVER_ERROR_POLLS * 3):
        # One hiccup = 3 x 503 (past _request's own budget) ⇒ one in-loop grace tick.
        seq.extend(httpx.Response(503, json={"detail": "hiccup"}) for _ in range(3))
        seq.append(httpx.Response(200, json={"status": "processing"}))
    seq.append(_completed())
    client = _client(_scripted(list(seq)))
    try:
        result = await client.compile(shots=[], timeout=600.0, idempotency_key=_IDEM_KEY)
    finally:
        await client.close()
    assert result["status"] == "completed"
