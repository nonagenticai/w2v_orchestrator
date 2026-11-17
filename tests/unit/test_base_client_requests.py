"""Tests for BaseServiceClient _request, submit_job, and poll_job methods."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from w2v_common.clients import JobSubmissionError

from src.clients.base import _MAX_RETRIES, _RETRY_DELAY, _RETRYABLE_STATUS_CODES, BaseServiceClient

# ============================================================================
# Constants
# ============================================================================


def test_retryable_status_codes():
    """Retryable status codes include expected server errors."""
    assert 429 in _RETRYABLE_STATUS_CODES
    assert 500 in _RETRYABLE_STATUS_CODES
    assert 502 in _RETRYABLE_STATUS_CODES
    assert 503 in _RETRYABLE_STATUS_CODES
    assert 504 in _RETRYABLE_STATUS_CODES
    assert 200 not in _RETRYABLE_STATUS_CODES


def test_max_retries():
    """Max retries default is 3."""
    assert _MAX_RETRIES == 3


def test_retry_delay():
    """Retry delay default is 1.0 seconds."""
    assert _RETRY_DELAY == 1.0


# ============================================================================
# _request - success
# ============================================================================


@pytest.mark.asyncio
async def test_request_get_success():
    """_request returns parsed JSON on successful GET."""
    client = BaseServiceClient(base_url="http://localhost:9999", service_name="test")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": "ok"}
    mock_response.raise_for_status = MagicMock()

    mock_httpx = AsyncMock()
    mock_httpx.request.return_value = mock_response
    mock_httpx.is_closed = False
    client._client = mock_httpx

    result = await client._request("GET", "/api/jobs/1")
    assert result == {"result": "ok"}
    await client.close()


@pytest.mark.asyncio
async def test_request_post_success():
    """_request returns parsed JSON on successful POST."""
    client = BaseServiceClient(base_url="http://localhost:9999", service_name="test")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"job_id": "j-1"}
    mock_response.raise_for_status = MagicMock()

    mock_httpx = AsyncMock()
    mock_httpx.request.return_value = mock_response
    mock_httpx.is_closed = False
    client._client = mock_httpx

    result = await client._request("POST", "/api/jobs", json={"url": "https://example.com"})
    assert result == {"job_id": "j-1"}
    await client.close()


# ============================================================================
# _request - retryable status codes
# ============================================================================


@pytest.mark.asyncio
async def test_request_retries_on_503():
    """_request retries on 503 status then succeeds."""
    client = BaseServiceClient(base_url="http://localhost:9999", service_name="test")

    fail_response = MagicMock()
    fail_response.status_code = 503

    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {"status": "ok"}
    success_response.raise_for_status = MagicMock()

    mock_httpx = AsyncMock()
    mock_httpx.request.side_effect = [fail_response, success_response]
    mock_httpx.is_closed = False
    client._client = mock_httpx

    with patch("w2v_common.clients.base.asyncio.sleep", new_callable=AsyncMock):
        result = await client._request("GET", "/test")
    assert result == {"status": "ok"}
    assert mock_httpx.request.call_count == 2
    await client.close()


@pytest.mark.asyncio
async def test_request_raises_after_max_retries():
    """_request raises after exhausting retries on retryable status codes."""
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
            await client._request("GET", "/test")
    assert mock_httpx.request.call_count == _MAX_RETRIES
    await client.close()


# ============================================================================
# _request - transport error with retry
# ============================================================================


@pytest.mark.asyncio
async def test_request_retries_on_transport_error():
    """_request retries on TransportError then succeeds."""
    client = BaseServiceClient(base_url="http://localhost:9999", service_name="test")

    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {"data": "ok"}
    success_response.raise_for_status = MagicMock()

    mock_httpx = AsyncMock()
    mock_httpx.request.side_effect = [
        httpx.ConnectError("Connection refused"),
        success_response,
    ]
    mock_httpx.is_closed = False
    client._client = mock_httpx

    with patch("w2v_common.clients.base.asyncio.sleep", new_callable=AsyncMock):
        result = await client._request("GET", "/test")
    assert result == {"data": "ok"}
    await client.close()


@pytest.mark.asyncio
async def test_request_raises_transport_error_after_retries():
    """_request raises TransportError after exhausting retries."""
    client = BaseServiceClient(base_url="http://localhost:9999", service_name="test")

    mock_httpx = AsyncMock()
    mock_httpx.request.side_effect = httpx.ConnectError("Connection refused")
    mock_httpx.is_closed = False
    client._client = mock_httpx

    with patch("w2v_common.clients.base.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(httpx.ConnectError):
            await client._request("GET", "/test")
    assert mock_httpx.request.call_count == _MAX_RETRIES
    await client.close()


# ============================================================================
# submit_job
# ============================================================================


@pytest.mark.asyncio
async def test_submit_job_returns_job_id():
    """submit_job returns the job_id from the response."""
    client = BaseServiceClient(base_url="http://localhost:9999", service_name="test")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"job_id": "j-abc-123"}
    mock_response.raise_for_status = MagicMock()

    mock_httpx = AsyncMock()
    mock_httpx.request.return_value = mock_response
    mock_httpx.is_closed = False
    client._client = mock_httpx

    job_id = await client.submit_job("/api/start", {"input": "test"})
    assert job_id == "j-abc-123"
    await client.close()


@pytest.mark.asyncio
async def test_submit_job_raises_when_the_response_carries_no_job_id():
    """A 2xx with no job_id is refused, not returned as "".

    The job is RUNNING downstream but there is no handle to supervise it.
    Returning "" made every caller poll ``<endpoint>/``, so the run died later
    as a 404/timeout naming the POLL url and never the submit. The raise must
    carry what a human needs to find it: the service, the submit endpoint, and
    the envelope's key shape (keys only -- values may carry secrets).
    """
    client = BaseServiceClient(base_url="http://localhost:9999", service_name="test")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "started"}
    mock_response.raise_for_status = MagicMock()

    mock_httpx = AsyncMock()
    mock_httpx.request.return_value = mock_response
    mock_httpx.is_closed = False
    client._client = mock_httpx

    with pytest.raises(JobSubmissionError) as excinfo:
        await client.submit_job("/api/start", {"input": "test"})

    message = str(excinfo.value)
    # Names the submit, not a later poll: service + method + endpoint.
    assert "test" in message
    assert "POST /api/start" in message
    # Names what was actually there, so the shape of the bad envelope is
    # debuggable without re-running the call.
    assert "status" in message
    # Values must NOT be echoed -- only the key shape.
    assert "started" not in message

    await client.close()


# ============================================================================
# poll_job
# ============================================================================


@pytest.mark.asyncio
async def test_poll_job_completed_immediately():
    """poll_job returns when job completes on first poll."""
    client = BaseServiceClient(base_url="http://localhost:9999", service_name="test")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "completed", "output": "video.mp4"}
    mock_response.raise_for_status = MagicMock()

    mock_httpx = AsyncMock()
    mock_httpx.request.return_value = mock_response
    mock_httpx.is_closed = False
    client._client = mock_httpx

    result = await client.poll_job("/api/jobs", "j-1")
    assert result["status"] == "completed"
    assert result["output"] == "video.mp4"
    await client.close()


@pytest.mark.asyncio
async def test_poll_job_fails():
    """poll_job raises RuntimeError when job status is 'failed'."""
    client = BaseServiceClient(base_url="http://localhost:9999", service_name="test")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "failed", "error": "out of memory"}
    mock_response.raise_for_status = MagicMock()

    mock_httpx = AsyncMock()
    mock_httpx.request.return_value = mock_response
    mock_httpx.is_closed = False
    client._client = mock_httpx

    with pytest.raises(RuntimeError, match="failed"):
        await client.poll_job("/api/jobs", "j-1")
    await client.close()


@pytest.mark.asyncio
async def test_poll_job_error_status():
    """poll_job raises RuntimeError when job status is 'error'."""
    client = BaseServiceClient(base_url="http://localhost:9999", service_name="test")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "error", "error": "internal"}
    mock_response.raise_for_status = MagicMock()

    mock_httpx = AsyncMock()
    mock_httpx.request.return_value = mock_response
    mock_httpx.is_closed = False
    client._client = mock_httpx

    with pytest.raises(RuntimeError, match="failed"):
        await client.poll_job("/api/jobs", "j-1")
    await client.close()


@pytest.mark.asyncio
async def test_poll_job_timeout():
    """poll_job raises TimeoutError when job exceeds timeout."""
    client = BaseServiceClient(base_url="http://localhost:9999", service_name="test")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "processing"}
    mock_response.raise_for_status = MagicMock()

    mock_httpx = AsyncMock()
    mock_httpx.request.return_value = mock_response
    mock_httpx.is_closed = False
    client._client = mock_httpx

    # w2v-common 0.16.2 enforces timeout >= 1.0s; the test still exercises
    # the timeout-elapsed branch — patched sleep advances monotonic-clock
    # tracking the loop expectation, not wall-clock.
    with pytest.raises(TimeoutError, match="timed out"):
        await client.poll_job("/api/jobs", "j-1", interval=0.1, timeout=1.0)
    await client.close()


@pytest.mark.asyncio
async def test_poll_job_max_attempts_exceeded():
    """poll_job raises TimeoutError when max_attempts is exceeded."""
    client = BaseServiceClient(base_url="http://localhost:9999", service_name="test")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "processing"}
    mock_response.raise_for_status = MagicMock()

    mock_httpx = AsyncMock()
    mock_httpx.request.return_value = mock_response
    mock_httpx.is_closed = False
    client._client = mock_httpx

    # w2v-common 0.16.2 enforces poll_interval > 0 and timeout <= 600s;
    # ``patch(asyncio.sleep)`` keeps wall-clock at zero so max_attempts
    # remains the gating factor.
    with patch("w2v_common.clients.base.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(TimeoutError, match="exceeded 5 poll attempts"):
            await client.poll_job("/api/jobs", "j-1", interval=0.1, timeout=600, max_attempts=5)
    assert mock_httpx.request.call_count == 5
    await client.close()


# ============================================================================
# _request with params
# ============================================================================


@pytest.mark.asyncio
async def test_request_with_params():
    """_request passes query params to the HTTP client."""
    client = BaseServiceClient(base_url="http://localhost:9999", service_name="test")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"items": []}
    mock_response.raise_for_status = MagicMock()

    mock_httpx = AsyncMock()
    mock_httpx.request.return_value = mock_response
    mock_httpx.is_closed = False
    client._client = mock_httpx

    result = await client._request("GET", "/api/list", params={"limit": 10})
    assert result == {"items": []}
    # Verify params were passed
    call_kwargs = mock_httpx.request.call_args
    assert call_kwargs.kwargs["params"] == {"limit": 10}
    await client.close()
