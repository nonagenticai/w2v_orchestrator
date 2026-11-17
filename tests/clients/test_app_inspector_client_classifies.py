"""`_fetch_result` must classify every non-409 status instead of re-raising it raw.

WHY THIS FILE EXISTS

`AppInspectorClient._fetch_result` bounded-retries a 409 (the `/result` write-race) and
used to `raise` everything else as a bare `httpx.HTTPStatusError`. Every caller then had
to re-interpret that status for itself — which is precisely the "N private retry policies"
defect the one policy in `stages/upstream.py` exists to remove. A 402 and a 503 arrived
looking identical, so a caller could not tell "no retry will help" from "worth retrying".

⚠️ THE COUPLED HALF. Routing this through `classify` is only safe because
`stages/app_review/inspect.py` widened its per-competitor `except` to include
`PermanentUpstreamError` / `TransientUpstreamError`. Those subclass plain `Exception` and
match none of `httpx.HTTPError` / `OSError` / `TimeoutError` / `RuntimeError` /
`ValidationError`, so without that widening one competitor 503 would escape `stage_gather`
and fail the whole PAID `app_inspect` stage *after the target had already succeeded*.
`test_inspect.py` owns the proof for that half; this file owns the client half.
"""

from __future__ import annotations

import httpx
import pytest

from src.clients.app_inspector_client import AppInspectorClient
from src.services.stages.upstream import PermanentUpstreamError, TransientUpstreamError


def _client(handler: httpx.MockTransport) -> AppInspectorClient:
    client = AppInspectorClient(base_url="http://app-inspector.test")
    client._client = httpx.AsyncClient(transport=handler, base_url="http://app-inspector.test")
    return client


def _transport(status: int, body: str = '{"detail":"upstream said so"}') -> httpx.MockTransport:
    def _handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body, headers={"content-type": "application/json"})

    return httpx.MockTransport(_handle)


@pytest.mark.asyncio
async def test_a_402_becomes_a_permanent_upstream_error() -> None:
    """402 is payment/config — no retry can help, and the caller must be able to know it."""
    client = _client(_transport(402))
    with pytest.raises(PermanentUpstreamError):
        await client.get_inspection("job-1")


@pytest.mark.asyncio
async def test_a_503_becomes_a_transient_upstream_error() -> None:
    """503 is worth retrying. As a bare HTTPStatusError it was indistinguishable from 402."""
    client = _client(_transport(503))
    with pytest.raises(TransientUpstreamError):
        await client.get_inspection("job-1")


@pytest.mark.asyncio
async def test_the_409_write_race_is_still_retried_and_not_classified() -> None:
    """The one status this method already handled must behave EXACTLY as before.

    409 means "the result row is not readable yet" — a bounded local retry, not an upstream
    fault. Routing it through the policy would change a working path; the bug was only ever
    what happened to every OTHER status.
    """
    attempts = 0

    def _handle(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(409, text='{"detail":"not ready"}')

    client = _client(httpx.MockTransport(_handle))
    with pytest.raises(RuntimeError, match="still 409"):
        await client.get_inspection("job-1")
    assert attempts > 1, "the 409 branch must still retry, not fail on the first attempt"
