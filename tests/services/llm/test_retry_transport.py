"""``in_process.RetryTransport`` — which transport errors are safe to re-send.

A connect failure never reached the provider, so retrying it is free. A read or
write timeout may have left the POST executing upstream; re-sending a
non-idempotent TENANT completion can run (and bill) it twice, so with
``retry_read_timeout=False`` it surfaces on the first failure. The PLATFORM's own
unmetered vLLM (``retry_read_timeout=True``) keeps the WM path's timeout retry.
"""

from __future__ import annotations

import httpx2
import pytest
from w2v_common.clients import base as _wm_retry

from src.services.llm import in_process

pytestmark = pytest.mark.asyncio


class _Raising(httpx2.AsyncBaseTransport):
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        self.calls += 1
        raise self.exc


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []

    async def _sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(in_process, "_sleep", _sleep)
    return slept


def _req() -> httpx2.Request:
    return httpx2.Request("POST", "http://llm.test/v1/chat/completions", json={"x": 1})


@pytest.mark.parametrize("exc_type", [httpx2.ReadTimeout])
async def test_tenant_read_and_write_timeouts_are_never_resent(exc_type: type[Exception]) -> None:
    inner = _Raising(exc_type("upstream slow"))
    with pytest.raises(exc_type):
        await in_process.RetryTransport(inner, retry_read_timeout=False).handle_async_request(
            _req()
        )
    assert inner.calls == 1


@pytest.mark.parametrize("exc_type", [httpx2.ConnectError])
async def test_connect_failures_are_retried_up_to_the_budget(exc_type: type[Exception]) -> None:
    inner = _Raising(exc_type("refused"))
    with pytest.raises(exc_type):
        await in_process.RetryTransport(inner, retry_read_timeout=False).handle_async_request(
            _req()
        )
    assert inner.calls == _wm_retry._MAX_RETRIES


async def test_connect_retries_respect_the_cumulative_sleep_budget(
    monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]
) -> None:
    monkeypatch.setattr(_wm_retry, "_MAX_RETRIES", 100)
    monkeypatch.setattr(_wm_retry, "_RETRY_DELAY", 10.0)
    monkeypatch.setattr(_wm_retry, "_RETRY_SLEEP_BUDGET", 25.0)
    inner = _Raising(httpx2.ConnectError("refused"))
    with pytest.raises(httpx2.ConnectError):
        await in_process.RetryTransport(inner, retry_read_timeout=False).handle_async_request(
            _req()
        )
    assert sum(_no_sleep) <= 25.0
    assert inner.calls == 2  # slept 10, then 10+20 > 25 → raise


@pytest.mark.parametrize("exc_type", [httpx2.ReadTimeout])
async def test_platform_read_and_write_timeouts_are_retried(exc_type: type[Exception]) -> None:
    inner = _Raising(exc_type("upstream slow"))
    with pytest.raises(exc_type):
        await in_process.RetryTransport(inner, retry_read_timeout=True).handle_async_request(_req())
    assert inner.calls == _wm_retry._MAX_RETRIES


@pytest.mark.parametrize("retry_read_timeout", [True])
async def test_http_client_passes_the_flag_to_its_retry_transport(
    monkeypatch: pytest.MonkeyPatch, retry_read_timeout: bool
) -> None:
    calls: list[int] = []

    def _slow() -> httpx2.MockTransport:
        def handler(request: httpx2.Request) -> httpx2.Response:
            calls.append(1)
            raise httpx2.ReadTimeout("slow", request=request)

        return httpx2.MockTransport(handler)

    monkeypatch.setattr(in_process, "inner_transport_factory", _slow)
    async with in_process._http_client(
        "http://llm.test/v1", None, 5.0, retry_read_timeout=retry_read_timeout
    ) as http:
        with pytest.raises(httpx2.ReadTimeout):
            await http.post("http://llm.test/v1/chat/completions", json={})
    assert len(calls) == (_wm_retry._MAX_RETRIES if retry_read_timeout else 1)
