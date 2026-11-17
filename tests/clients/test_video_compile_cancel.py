"""Cancelling a run must cancel the render it is paying for.

``dispatcher.cancel`` stops the orchestrator's own task; it says nothing to the
video-compile service, so before this the compile ran to completion and was billed in
full. These drive the client + the best-effort wrapper through an ``httpx.MockTransport``
and pin the two things the cancel path depends on: the RIGHT key reaches the wire, and no
compiler failure of any shape can propagate into the caller.
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid7

import httpx
import pytest

from src.clients.video_compile import VideoCompileClient
from src.services import compile_cancel

pytestmark = pytest.mark.asyncio

_RUN_ID = UUID("0198f0f0-0000-7000-8000-000000000001")


def _client(handler) -> VideoCompileClient:  # type: ignore[no-untyped-def]
    client = VideoCompileClient(base_url="http://compile.test", license_token="t")
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        timeout=client.timeout,
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    return client


async def test_cancel_posts_the_run_prefix_key() -> None:
    """The prefix must be ``{run_id}:`` — exactly what ``_compile_idempotency_key``
    builds from. A bare run_id would match nothing and the cancel would be a silent
    no-op; anything shorter would reach other runs' renders."""
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen.append((request.url.path, _json.loads(request.content)))
        return httpx.Response(200, json={"cancelled": ["job-1", "job-2"], "killed": 1})

    client = _client(handler)
    n = await client.cancel_run_jobs(str(_RUN_ID))
    await client.close()

    assert seen == [("/api/jobs/cancel", {"idempotency_key_prefix": f"{_RUN_ID}:"})]
    assert n == 2


@pytest.mark.parametrize(
    "handler",
    [pytest.param(lambda r: httpx.Response(500, json={"detail": "boom"}), id="500")],
)
async def test_cancel_propagation_never_raises(monkeypatch: pytest.MonkeyPatch, handler) -> None:  # type: ignore[no-untyped-def]
    """A compiler that is down, missing the route (old image mid-rollout), or answering
    garbage must NOT fail the user's cancel — that would strand the run in ``cancelling``,
    which is strictly worse than a render that keeps going."""
    client = _client(handler)
    monkeypatch.setattr(compile_cancel, "VideoCompileClient", lambda **_k: client)

    assert await compile_cancel.cancel_compiles_for_run(_RUN_ID) == 0


async def test_cancel_propagation_is_hard_timeout_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wedged compiler must not park the cancel (and the DB transaction behind it) for
    the client's full 600s request timeout."""

    async def _hangs(*_a: object, **_k: object) -> int:
        await asyncio.sleep(3600)
        return 1

    closed = False

    class _Hanging:
        def __init__(self, **_k: object) -> None: ...

        cancel_run_jobs = staticmethod(_hangs)

        async def close(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr(compile_cancel, "VideoCompileClient", _Hanging)
    monkeypatch.setattr(compile_cancel, "_CANCEL_TIMEOUT_SECONDS", 0.05)

    n = await asyncio.wait_for(compile_cancel.cancel_compiles_for_run(_RUN_ID), timeout=5)

    assert n == 0
    assert closed, "the http client must be closed even on the timeout path (pool leak)"


async def test_cancel_propagation_survives_a_client_that_cannot_be_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing the client resolves discovery + reads the SA license token. On a
    misconfigured pod that raises, and it must not take the user's cancel down with it —
    which is why the constructor sits inside the guard, not above it."""

    def _explodes(**_k: object) -> VideoCompileClient:
        raise RuntimeError("no license token configured")

    monkeypatch.setattr(compile_cancel, "VideoCompileClient", _explodes)

    assert await compile_cancel.cancel_compiles_for_run(_RUN_ID) == 0


async def test_cancel_propagation_closes_the_client_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One cancelled run must not leak one connection pool."""
    client = _client(lambda r: httpx.Response(200, json={"cancelled": ["j"], "killed": 1}))
    monkeypatch.setattr(compile_cancel, "VideoCompileClient", lambda **_k: client)

    inner = client._client
    assert await compile_cancel.cancel_compiles_for_run(_RUN_ID) == 1
    # BaseServiceClient.close() drops its httpx client; assert on the underlying object
    # (the attribute itself is None afterwards, which alone would not prove a close).
    assert inner is not None and inner.is_closed
    assert client._client is None


async def test_poll_treats_cancelled_as_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A poll that meets ``cancelled`` must end there. An unrecognised status is not
    terminal in either direction, so the loop would spin to the full render budget and
    the stage would report a TIMEOUT on a job that was provably never going to finish."""

    async def _noop(*_a: object, **_k: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _noop)

    polls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        if request.method == "POST":
            return httpx.Response(200, json={"job_id": "job-1"})
        polls += 1
        return httpx.Response(200, json={"status": "cancelled", "error": "cancelled by caller"})

    client = _client(handler)
    with pytest.raises(RuntimeError, match="cancelled by caller"):
        await client.compile(
            shots=[], timeout=1200.0, idempotency_key=f"{uuid7()}:demo_video_compile"
        )
    await client.close()
    assert polls == 1, "must not keep polling a cancelled job"
