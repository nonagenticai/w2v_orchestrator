"""Tests for service clients."""

from __future__ import annotations

import itertools
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from w2v_common.auth.identity_context import clear_identity_context, set_identity_context

from src.clients.base import BaseServiceClient

#: w2v-common 0.67.0 refuses an outbound call with no tenant bound and no
#: service-account token ("every message must carry the requestor's tenant_id
#: and there is no default"). Real callers always run inside a bound context.
_TENANT = "dddddddd-dddd-dddd-dddd-dddddddddddd"


@pytest.fixture(autouse=True)
def _bound_tenant():
    set_identity_context(tenant_id=_TENANT, license_id="lic", subject_id="sub")
    try:
        yield
    finally:
        clear_identity_context()


@pytest.mark.asyncio
async def test_base_client_init():
    """BaseServiceClient stores base_url and service_name correctly."""
    client = BaseServiceClient(
        base_url="http://localhost:6303",
        service_name="test-service",
    )
    assert client.base_url == "http://localhost:6303"
    assert client.service_name == "test-service"
    assert client.timeout == 30.0


@pytest.mark.asyncio
async def test_base_client_health_check_failure():
    """health_check returns unhealthy when service is unreachable."""
    client = BaseServiceClient(
        base_url="http://localhost:19999",
        service_name="test-service",
    )
    result = await client.health_check()
    assert result["status"] == "unhealthy"
    assert result["service"] == "test-service"
    assert "error" in result


@pytest.mark.asyncio
async def test_base_client_close_after_request():
    """close() should properly clean up an opened httpx client."""
    client = BaseServiceClient(
        base_url="http://localhost:6303",
        service_name="test-service",
    )
    # Force creation of the internal httpx client
    internal = await client._get_client()
    assert internal is not None
    assert not internal.is_closed

    await client.close()
    assert client._client is None


@pytest.mark.asyncio
async def test_base_client_license_token():
    """License token is stored on the client."""
    client = BaseServiceClient(
        base_url="http://localhost:6303",
        service_name="test-service",
        license_token="my-token-123",
    )
    assert client.license_token == "my-token-123"
    # The internal httpx client should carry the token via X-Tenant-Id
    # (w2v_common 0.6.7+: AuthMiddleware reads X-Tenant-Id, not Authorization)
    internal = await client._get_client()
    assert internal.headers.get("x-tenant-id") == "Bearer my-token-123"
    assert internal.headers.get("authorization") is None
    await client.close()


@pytest.mark.asyncio
async def test_base_client_get_client_reuses():
    """_get_client returns the same client on repeated calls."""
    client = BaseServiceClient(
        base_url="http://localhost:6303",
        service_name="test-service",
    )
    c1 = await client._get_client()
    c2 = await client._get_client()
    assert c1 is c2
    await client.close()


# ============================================================================
# _request - retry on 500 then success
# ============================================================================


@pytest.mark.asyncio
async def test_base_client_request_retries_on_500():
    """_request retries on 500 status then succeeds on second attempt."""
    client = BaseServiceClient(base_url="http://localhost:9999", service_name="test")

    fail_response = MagicMock()
    fail_response.status_code = 500

    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {"result": "recovered"}
    success_response.raise_for_status = MagicMock()

    mock_httpx = AsyncMock()
    mock_httpx.request.side_effect = [fail_response, success_response]
    mock_httpx.is_closed = False
    client._client = mock_httpx

    with patch("w2v_common.clients.base.asyncio.sleep", new_callable=AsyncMock):
        result = await client._request("GET", "/api/test")

    assert result == {"result": "recovered"}
    assert mock_httpx.request.call_count == 2
    await client.close()


# ============================================================================
# _request - exhaust retries on 503
# ============================================================================


@pytest.mark.asyncio
async def test_base_client_request_exhausts_retries():
    """_request raises HTTPStatusError after exhausting retries on persistent 503."""
    client = BaseServiceClient(base_url="http://localhost:9999", service_name="test")

    fail_response = MagicMock()
    fail_response.status_code = 503
    fail_response.text = "503 service unavailable"
    fail_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("503 error", request=MagicMock(), response=fail_response)
    )

    mock_httpx = AsyncMock()
    mock_httpx.request.return_value = fail_response
    mock_httpx.is_closed = False
    client._client = mock_httpx

    with patch("w2v_common.clients.base.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(httpx.HTTPStatusError):
            await client._request("GET", "/api/failing")

    # All 3 retries should have been attempted
    assert mock_httpx.request.call_count == 3
    await client.close()


# ============================================================================
# poll_job - timeout via time.monotonic
# ============================================================================


@pytest.mark.asyncio
async def test_base_client_poll_job_timeout():
    """poll_job raises TimeoutError when time.monotonic exceeds timeout."""
    client = BaseServiceClient(base_url="http://localhost:9999", service_name="test")

    # Always return "processing" so it never completes
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "processing"}
    mock_response.raise_for_status = MagicMock()

    mock_httpx = AsyncMock()
    mock_httpx.request.return_value = mock_response
    mock_httpx.is_closed = False
    client._client = mock_httpx

    # Simulate time passing beyond the timeout on the second poll iteration.
    # time is imported locally inside poll_job, so we patch the stdlib time
    # module. Use a non-exhausting sequence (start + first elapsed read are 0,
    # everything after is past the timeout) so the test is robust to however
    # many monotonic() reads the client + its instrumentation make per request.
    with (
        patch("time.monotonic", side_effect=itertools.chain([0.0, 0.0], itertools.repeat(100.0))),
        patch("w2v_common.clients.base.asyncio.sleep", new_callable=AsyncMock),
    ):
        with pytest.raises(TimeoutError, match="timed out"):
            await client.poll_job("/api/jobs", "j-timeout", timeout=10.0)

    await client.close()
