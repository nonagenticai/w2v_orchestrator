"""Tests for middleware components: request_id, audit, metrics.

The frame-policy / security-header behaviour is covered by
``tests/unit/test_frame_policy_middleware.py`` against the LIVE canonical middleware.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.middleware.audit import (
    _AUTH_FAILURE_CODES,
    _MUTATION_METHODS,
    AuditMiddleware,
    fingerprint_api_key,
)

# Promoted OUT of ``src.middleware.audit`` (where it was ``_safe_exc_message``) into
# stdlib-only ``src.utils``: the preview-proxy sweeper needs it and must not import
# the Starlette middleware stack. The assertions below are unchanged — this is the
# same function, and the audit middleware's use of it is what they still cover.
from src.utils import MAX_EXC_MESSAGE_CHARS as _MAX_EXC_MESSAGE_CHARS
from src.utils import safe_exc_message as _safe_exc_message

# ============================================================================
# fingerprint_api_key
# ============================================================================


def test_fingerprint_api_key_returns_short_hash():
    """fingerprint_api_key returns a 16-char hex digest."""
    result = fingerprint_api_key("my-secret-api-key")
    assert result is not None
    assert len(result) == 16
    assert all(c in "0123456789abcdef" for c in result)


def test_fingerprint_api_key_none():
    """fingerprint_api_key returns None for None input."""
    assert fingerprint_api_key(None) is None


def test_fingerprint_api_key_empty():
    """fingerprint_api_key returns None for empty string."""
    assert fingerprint_api_key("") is None


def test_fingerprint_api_key_deterministic():
    """Same input produces same fingerprint."""
    assert fingerprint_api_key("token-xyz") == fingerprint_api_key("token-xyz")


def test_fingerprint_api_key_different_inputs():
    """Different inputs produce different fingerprints."""
    assert fingerprint_api_key("token-a") != fingerprint_api_key("token-b")


# ============================================================================
# _resource_from_path
# ============================================================================


# ============================================================================
# _MUTATION_METHODS and _AUTH_FAILURE_CODES constants
# ============================================================================


def test_mutation_methods():
    """MUTATION_METHODS includes standard state-changing HTTP methods."""
    assert "POST" in _MUTATION_METHODS
    assert "PUT" in _MUTATION_METHODS
    assert "DELETE" in _MUTATION_METHODS
    assert "PATCH" in _MUTATION_METHODS
    assert "GET" not in _MUTATION_METHODS


def test_auth_failure_codes():
    """AUTH_FAILURE_CODES includes 401 and 403."""
    assert 401 in _AUTH_FAILURE_CODES
    assert 403 in _AUTH_FAILURE_CODES
    assert 200 not in _AUTH_FAILURE_CODES


# RequestIdMiddleware is no longer orchestrator-local: it is the shared
# w2v_common.middleware.RequestIdMiddleware installed by create_app (and tested
# in w2v_common). The migration's wiring/ordering is covered by
# tests/unit/test_asgi_middleware_stack.py.


# ============================================================================
# AuditMiddleware
# ============================================================================


def _audit_request(
    method: str,
    path: str,
    headers: dict[str, str],
    *,
    query: dict[str, str] | None = None,
    tenant_id: str | None = None,
    roles: list[str] | None = None,
):
    """A request whose state is realistic but whose app has NO session_factory,
    so AuditMiddleware's persist path cleanly no-ops (logging is what's asserted
    here; the DB persistence is covered in tests/api/routes/test_request_audit.py)."""
    request = MagicMock()
    request.method = method
    request.url.path = path
    request.headers = headers
    # A real dict, never a MagicMock: the middleware reads `active_tenant` off it,
    # and a MagicMock would return a truthy sentinel for every lookup.
    request.query_params = query or {}
    request.client.host = "127.0.0.1"
    # Real attributes (not MagicMock) so roles iteration / uuid parsing behave.
    request.state = SimpleNamespace(
        tenant_id=tenant_id, roles=roles or [], user=None, request_id="r"
    )
    request.app.state = SimpleNamespace()  # no session_factory → persist skips
    return request


_AUDIT_CLAIM = "11111111-1111-4111-8111-111111111111"
_AUDIT_OTHER = "22222222-2222-4222-8222-222222222222"


async def _audit_get(request) -> MagicMock:
    """Drive a 200 GET through AuditMiddleware and return the patched logger."""
    middleware = AuditMiddleware(app=MagicMock())
    response = MagicMock()
    response.status_code = 200
    call_next = AsyncMock(return_value=response)
    with patch("src.middleware.audit.logger") as mock_logger:
        await middleware.dispatch(request, call_next)
    return mock_logger


@pytest.mark.asyncio
async def test_audit_middleware_records_cross_tenant_get_via_query_param():
    """A scope switch carried as `?active_tenant=` must still be audited.

    Browser-initiated loads (<img>/<video>/<a download>/EventSource) cannot set
    the X-Active-Tenant-Id header, so they carry the switch as a query param. If
    this middleware read only the header, such a request would score
    cross_tenant=False and — being a successful GET — would not be recorded at
    all, letting a super-admin read another tenant's artifacts with no durable
    trail. Regression pin for exactly that.
    """
    mock_logger = await _audit_get(
        _audit_request(
            "GET",
            "/api/runs/abc/artifact/w2v-promo/x.png",
            {},
            query={"active_tenant": _AUDIT_OTHER},
            tenant_id=_AUDIT_CLAIM,
            roles=["super_admin"],
        )
    )
    mock_logger.info.assert_called_once()
    payload = mock_logger.info.call_args[0][1]
    assert payload["cross_tenant"] is True
    assert payload["event"] == "read"
    assert payload["tenant_id"] == _AUDIT_OTHER


@pytest.mark.asyncio
async def test_audit_middleware_ignores_query_param_switch_for_non_super():
    """NEGATIVE CONTROL: the param grants a non-super caller nothing.

    Same request, same param — only the role differs. Without this pair the test
    above would pass even if the middleware recorded every GET carrying the param.
    """
    mock_logger = await _audit_get(
        _audit_request(
            "GET",
            "/api/runs/abc/artifact/w2v-promo/x.png",
            {},
            query={"active_tenant": _AUDIT_OTHER},
            tenant_id=_AUDIT_CLAIM,
            roles=["tenant_admin"],
        )
    )
    mock_logger.info.assert_not_called()


@pytest.mark.asyncio
async def test_audit_middleware_same_tenant_query_param_is_not_cross_tenant():
    """NEGATIVE CONTROL: switching to your own tenant is a no-op, not an event."""
    mock_logger = await _audit_get(
        _audit_request(
            "GET",
            "/api/runs/abc/artifact/w2v-promo/x.png",
            {},
            query={"active_tenant": _AUDIT_CLAIM},
            tenant_id=_AUDIT_CLAIM,
            roles=["super_admin"],
        )
    )
    mock_logger.info.assert_not_called()


@pytest.mark.asyncio
async def test_audit_middleware_logs_mutation():
    """AuditMiddleware logs POST/PUT/DELETE requests."""
    middleware = AuditMiddleware(app=MagicMock())
    request = _audit_request("POST", "/api/pipeline", {"x-tenant-id": "test-tenant"})

    response = MagicMock()
    response.status_code = 200
    call_next = AsyncMock(return_value=response)

    with patch("src.middleware.audit.logger") as mock_logger:
        result = await middleware.dispatch(request, call_next)
        assert result is response
        mock_logger.info.assert_called_once()
        assert mock_logger.info.call_args[0][1]["event"] == "mutation"


@pytest.mark.asyncio
async def test_audit_middleware_logs_auth_failure():
    """AuditMiddleware logs auth failures (401/403)."""
    middleware = AuditMiddleware(app=MagicMock())
    request = _audit_request("GET", "/api/pipeline", {"x-tenant-id": "anonymous"})

    response = MagicMock()
    response.status_code = 401
    call_next = AsyncMock(return_value=response)

    with patch("src.middleware.audit.logger") as mock_logger:
        result = await middleware.dispatch(request, call_next)
        assert result is response
        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args[0][1]["event"] == "auth_failure"


@pytest.mark.asyncio
async def test_audit_write_timeout_logs_repr_and_request_id(monkeypatch):
    """A persist that blows the write timeout must log a NON-EMPTY message.

    `asyncio.timeout` raises a BARE `TimeoutError()` (CPython does
    `raise TimeoutError from exc_val` with no args), so `"%s" % exc` is the
    empty string — which is exactly what shipped and produced 15 blind
    `request_audit write failed: ` warnings in 24h with no correlation id.
    This drives the REAL timeout (no mocked exception) and pins both halves:
    the repr renders, and the request id rides along.
    """
    monkeypatch.setattr("src.middleware.audit._AUDIT_WRITE_TIMEOUT_S", 0.01)

    class _HangingSession:
        async def __aenter__(self):
            await asyncio.sleep(1.0)  # never completes inside the 0.01s cap
            return self

        async def __aexit__(self, *exc):
            return False

    request = _audit_request("POST", "/api/projects", {}, tenant_id=_AUDIT_CLAIM)
    request.state.request_id = "req-abc-123"
    request.app.state = SimpleNamespace(session_factory=lambda: _HangingSession())

    middleware = AuditMiddleware(app=MagicMock())
    response = MagicMock()
    response.status_code = 201
    with patch("src.middleware.audit.logger") as mock_logger:
        result = await middleware.dispatch(request, AsyncMock(return_value=response))

    # The swallow is intentional: audit must never break the request it audits.
    assert result is response
    mock_logger.warning.assert_called_once()
    args, kwargs = mock_logger.warning.call_args
    rendered = args[0] % tuple(args[1:])
    assert "TimeoutError()" in rendered, f"empty/blind exception rendering: {rendered!r}"
    assert "req-abc-123" in rendered
    assert kwargs.get("exc_info") is True


# ============================================================================
# _safe_exc_message — what an unhandled exception is allowed to leave behind
# ============================================================================


def test_safe_exc_message_keeps_a_plain_message():
    assert _safe_exc_message(RuntimeError("stage timed out")) == "stage timed out"


def test_safe_exc_message_scrubs_a_bearer_token():
    """An httpx/client error can echo the caller's credential. It must not land
    in the audit row."""
    msg = _safe_exc_message(RuntimeError("401 for header Bearer abc.def.ghi123"))
    assert "abc.def.ghi123" not in msg
    assert "[redacted]" in msg


def test_safe_exc_message_scrubs_a_keyword_secret():
    msg = _safe_exc_message(ValueError("connect failed: password=hunter2 host=db"))
    assert "hunter2" not in msg
    assert "[redacted]" in msg


def test_safe_exc_message_is_truncated():
    msg = _safe_exc_message(RuntimeError("x" * (_MAX_EXC_MESSAGE_CHARS + 50)))
    assert len(msg) == _MAX_EXC_MESSAGE_CHARS


def test_safe_exc_message_falls_back_to_repr_for_a_bare_exception():
    """`str(TimeoutError())` is EMPTY — the same trap the persist-failure log
    line documents. An empty message would make the row useless."""
    assert _safe_exc_message(TimeoutError()) == "TimeoutError()"
