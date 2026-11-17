"""A failed inspection that carried a BrowserStack key must never echo it.

app_inspector answers a malformed ``device_farm`` with a pydantic 422 that echoes
the whole object — access key included — under ``input``. w2v_common logs that body
before raising and its redaction (``\\baccess[_-]?key``) does not match
``browserstack_access_key``, so the key landed in the pod log AND in the
``HTTPStatusError`` message that becomes the stage error.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from src.clients.app_inspector_client import AppInspectorClient, DeviceFarmInspectionError

FARM_KEY = "bsk-p5-device-farm-fixture-0123456789"  # pragma: allowlist secret
FARM_CONFIG = {
    "device_farm": {
        "provider": "browserstack",
        "browserstack_username": "tenant-user",
        "browserstack_access_key": FARM_KEY,
    }
}


def _echoing_client() -> AppInspectorClient:
    """A fake app_inspector that 422s and echoes the request's device_farm verbatim."""

    def _handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        detail = [
            {
                "type": "value_error",
                "loc": ["body", "device_farm"],
                "msg": "Value error, unsupported device",
                "input": body.get("device_farm"),
            }
        ]
        return httpx.Response(422, json={"detail": detail})

    client = AppInspectorClient(base_url="http://app-inspector.test")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handle), base_url="http://app-inspector.test"
    )
    return client


def _all_log_text(caplog: pytest.LogCaptureFixture) -> str:
    formatter = logging.Formatter("%(name)s %(message)s")
    return "\n".join(formatter.format(r) for r in caplog.records)


@pytest.mark.asyncio
async def test_a_422_echo_never_leaks_the_access_key(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    client = _echoing_client()

    with pytest.raises(DeviceFarmInspectionError) as raised:
        await client.inspect_app("https://app.test", FARM_CONFIG)

    assert raised.value.status_code == 422
    assert FARM_KEY not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    logged = _all_log_text(caplog)
    assert "returned 422" in logged  # the failure is still visible...
    assert FARM_KEY not in logged  # ...without the key
    # No partial key either (a truncated body can end mid-secret).
    assert FARM_KEY[:12] not in logged


@pytest.mark.asyncio
async def test_without_a_farm_the_upstream_error_is_untouched(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No farm secret in flight → today's behaviour: the raw status error, body logged."""
    caplog.set_level(logging.DEBUG)
    client = _echoing_client()

    with pytest.raises(httpx.HTTPStatusError):
        await client.inspect_app("https://app.test", {"platform": "web"})
    assert "unsupported device" in _all_log_text(caplog)
