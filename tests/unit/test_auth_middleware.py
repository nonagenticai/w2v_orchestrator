"""Tests for license token authentication middleware."""

from __future__ import annotations

import importlib
import ipaddress
import os
import uuid
from collections.abc import Iterator
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError
from w2v_common.tenant import MissingTenantError

from src.auth.middleware import (
    PUBLIC_ENDPOINTS,
    PUBLIC_PREFIXES,
    AuthMiddleware,
    _get_license_cache,
    _introspect_license,
    _is_trusted_network,
)
from src.auth.models import User

# ---------- User model tests ----------


#: A real tenant uuid. w2v-common 0.67.0 types ``User.tenant_id`` as
#: ``uuid.UUID | None``, so a slug like ``"tenant-1"`` is no longer merely
#: untyped — Pydantic REJECTS it at construction.
_TENANT = uuid.UUID("11111111-2222-3333-4444-555555555555")


def test_user_model_creation():
    """User model stores sub, preferred_username, and tenant_id."""
    user = User(sub="tenant-1", preferred_username="admin", tenant_id=_TENANT)
    assert user.sub == "tenant-1"
    assert user.preferred_username == "admin"
    assert user.tenant_id == _TENANT


def test_user_model_defaults():
    """User model has sensible defaults."""
    user = User(sub="test-sub")
    assert user.preferred_username == "anonymous"
    assert user.tenant_id is None


def test_user_model_rejects_a_non_uuid_tenant():
    """A slug tenant is refused at the boundary, not carried as a plausible string.

    Pre-0.67.0 ``tenant_id`` was a bare ``str``, so ``"tenant-1"`` constructed
    happily and survived all the way into storage keys and outbound headers.
    """
    with pytest.raises(ValidationError):
        User(sub="sub-1", tenant_id="tenant-1")


def test_user_effective_tenant_id_returns_the_real_tenant():
    """effective_tenant_id returns the bound tenant as a UUID."""
    user_with_tenant = User(sub="sub-1", tenant_id=_TENANT)
    assert user_with_tenant.effective_tenant_id == _TENANT


def test_user_effective_tenant_id_refuses_to_fall_back_to_sub():
    """No tenant → RAISE. It must never mint a tenant equal to ``sub``.

    This test previously asserted the substitution itself
    (``effective_tenant_id == "sub-1"``). That fallback was a sentinel by
    another name: it invented a tenant that was never registered, and because
    ``sub`` is free-form it also produced non-UUID tenants. w2v-common 0.67.0
    removed it, so the assertion is inverted to pin the REFUSAL — asserting the
    old substitution would be asserting the bug.
    """
    user_without_tenant = User(sub="sub-1")
    with pytest.raises(MissingTenantError):
        _ = user_without_tenant.effective_tenant_id


# ---------- _is_trusted_network tests ----------
#
# w2v-common 0.89.0 made the trust lanes explicit opt-in: TRUSTED_CIDRS now
# defaults to EMPTY (it used to default to loopback + 172.16/12 + 10/8). The set
# is parsed once at import, so every test below reloads the shared middleware
# module — and this service's re-export shim — with TRUSTED_CIDRS pinned, the
# same way w2v_common's own tests do. ``restore_middleware_module`` reloads both
# again from a clean env so no other test inherits a widened trust set.


def _reload_with_trusted_cidrs(monkeypatch: pytest.MonkeyPatch, value: str | None) -> ModuleType:
    """Reload middleware with TRUSTED_CIDRS set (or unset); return the src shim."""
    import w2v_common.auth.middleware as shared

    import src.auth.middleware as shim

    if value is None:
        monkeypatch.delenv("TRUSTED_CIDRS", raising=False)
    else:
        monkeypatch.setenv("TRUSTED_CIDRS", value)
    importlib.reload(shared)
    return importlib.reload(shim)


@pytest.fixture
def restore_middleware_module() -> Iterator[None]:
    """Reload middleware + shim once more after the test, from a clean env."""
    import w2v_common.auth.middleware as shared

    import src.auth.middleware as shim

    yield
    saved = os.environ.pop("TRUSTED_CIDRS", None)
    try:
        importlib.reload(shared)
        importlib.reload(shim)
    finally:
        if saved is not None:
            os.environ["TRUSTED_CIDRS"] = saved


@pytest.mark.usefixtures("restore_middleware_module")
@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "::1",
        "10.0.0.1",
        "10.42.0.100",  # k3s pod CIDR
        "172.16.0.1",
        "192.168.1.1",  # LAN edge / SSR proxy range (2026-04-18 bypass)
        "8.8.8.8",
    ],
)
def test_nothing_trusted_by_default(monkeypatch: pytest.MonkeyPatch, host: str):
    """With TRUSTED_CIDRS unset, no address — not even loopback — is trusted."""
    mod = _reload_with_trusted_cidrs(monkeypatch, None)
    parsed = mod.TRUSTED_NETWORKS
    assert parsed == []
    assert mod._is_trusted_network(host) is False


@pytest.mark.usefixtures("restore_middleware_module")
def test_is_trusted_network_loopback(monkeypatch: pytest.MonkeyPatch):
    """Loopback is trusted when the operator names it."""
    mod = _reload_with_trusted_cidrs(monkeypatch, "127.0.0.1/32")
    assert mod._is_trusted_network("127.0.0.1") is True


@pytest.mark.usefixtures("restore_middleware_module")
def test_is_trusted_network_ipv6_loopback(monkeypatch: pytest.MonkeyPatch):
    """IPv6 loopback is trusted when the operator names it."""
    mod = _reload_with_trusted_cidrs(monkeypatch, "::1/128")
    assert mod._is_trusted_network("::1") is True
    # Naming ::1 does not implicitly trust the IPv4 loopback.
    assert mod._is_trusted_network("127.0.0.1") is False


@pytest.mark.usefixtures("restore_middleware_module")
def test_is_trusted_network_private_ranges(monkeypatch: pytest.MonkeyPatch):
    """Named in-cluster private ranges (10.0.0.0/8, 172.16.0.0/12) are trusted.

    192.168.0.0/16 stays untrusted even then — that range is the LAN edge
    (where SSR proxies live), so trusting it would let unauth external requests
    slip past via the source-IP fallback path. See
    project_orchestrator_trusted_networks_bypass_2026-04-18.md.
    """
    mod = _reload_with_trusted_cidrs(monkeypatch, "10.0.0.0/8,172.16.0.0/12")
    assert mod._is_trusted_network("10.0.0.1") is True
    assert mod._is_trusted_network("172.16.0.1") is True
    assert mod._is_trusted_network("192.168.1.1") is False
    assert mod._is_trusted_network("127.0.0.1") is False


def test_is_trusted_network_lan_edge_not_trusted():
    """192.168.0.0/16 (LAN edge / SSR proxy range) MUST NOT be trusted."""
    assert _is_trusted_network("192.168.1.1") is False
    assert _is_trusted_network("192.168.0.1") is False


def test_is_trusted_network_public_ip():
    """Public IP addresses are not trusted."""
    assert _is_trusted_network("8.8.8.8") is False


@pytest.mark.usefixtures("restore_middleware_module")
@pytest.mark.parametrize("host", [None, "not-an-ip", ""])
def test_is_trusted_network_unparseable_host(monkeypatch: pytest.MonkeyPatch, host: str | None):
    """None / invalid / empty hosts are never trusted, even under a trust-all set."""
    mod = _reload_with_trusted_cidrs(monkeypatch, "0.0.0.0/0,::/0")
    assert mod._is_trusted_network(host) is False


# ---------- PUBLIC_ENDPOINTS and PUBLIC_PREFIXES ----------


def test_public_endpoints_contains_health():
    """Health endpoints are public."""
    assert "/health" in PUBLIC_ENDPOINTS
    assert "/health/ready" in PUBLIC_ENDPOINTS
    assert "/health/live" in PUBLIC_ENDPOINTS


def test_public_endpoints_contains_docs():
    """Documentation endpoints are public."""
    assert "/docs" in PUBLIC_ENDPOINTS
    assert "/redoc" in PUBLIC_ENDPOINTS
    assert "/openapi.json" in PUBLIC_ENDPOINTS


def test_public_prefixes():
    """Public prefixes include docs and health."""
    assert any(p.startswith("/docs") for p in PUBLIC_PREFIXES)
    assert any(p.startswith("/health/") for p in PUBLIC_PREFIXES)


# ---------- TRUSTED_NETWORKS ----------


@pytest.mark.usefixtures("restore_middleware_module")
def test_trusted_networks_default_is_empty(monkeypatch: pytest.MonkeyPatch):
    """The DEFAULT trust set is empty (w2v-common 0.89.0, explicit opt-in)."""
    mod = _reload_with_trusted_cidrs(monkeypatch, None)
    assert mod.TRUSTED_NETWORKS == []


@pytest.mark.usefixtures("restore_middleware_module")
def test_trusted_networks_parsed(monkeypatch: pytest.MonkeyPatch):
    """An explicit TRUSTED_CIDRS parses into exactly the named IP networks."""
    mod = _reload_with_trusted_cidrs(monkeypatch, " 10.42.0.0/16 , ::1/128 ,")
    parsed = mod.TRUSTED_NETWORKS
    assert parsed == [ipaddress.ip_network("10.42.0.0/16"), ipaddress.ip_network("::1/128")]
    for net in parsed:
        assert isinstance(net, (ipaddress.IPv4Network, ipaddress.IPv6Network))


# ---------- _introspect_license ----------


@pytest.mark.asyncio
async def test_introspect_license_cached():
    """Cached license introspection returns cached result."""
    _get_license_cache().clear()
    token = "test-cached-token"
    import hashlib

    cache_key = f"lic:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
    cached_data = {"active": True, "tenant_id": "t1"}
    _get_license_cache()[cache_key] = cached_data

    result = await _introspect_license(token)
    assert result == cached_data
    _get_license_cache().clear()


@pytest.mark.asyncio
async def test_introspect_license_success():
    """Successful license introspection returns data and caches it."""
    _get_license_cache().clear()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"active": True, "tenant_id": "t1"}

    with patch("w2v_common.auth.middleware.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        result = await _introspect_license("valid-token")
        assert result is not None
        assert result["active"] is True
    _get_license_cache().clear()


@pytest.mark.asyncio
async def test_introspect_license_inactive():
    """Inactive license returns data but is not cached."""
    _get_license_cache().clear()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"active": False, "reason": "expired"}

    with patch("w2v_common.auth.middleware.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        result = await _introspect_license("inactive-token")
        assert result is not None
        assert result["active"] is False
    _get_license_cache().clear()


@pytest.mark.asyncio
async def test_introspect_license_network_error():
    """Network error during introspection returns None."""
    _get_license_cache().clear()
    with patch("w2v_common.auth.middleware.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        result = await _introspect_license("error-token")
        assert result is None
    _get_license_cache().clear()


@pytest.mark.asyncio
async def test_introspect_license_non_200():
    """Non-200 response from introspection returns None."""
    _get_license_cache().clear()
    mock_response = MagicMock()
    mock_response.status_code = 500

    with patch("w2v_common.auth.middleware.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        result = await _introspect_license("bad-response-token")
        assert result is None
    _get_license_cache().clear()


# ---------- AuthMiddleware._extract_license_token ----------


def test_extract_license_token_accepts_opaque_authorization_bearer():
    """An OPAQUE `Authorization: Bearer` IS a license source (w2v_common 0.74.0).

    0.74.0 deliberately widened `_extract_license_token` to accept the platform
    credential shape in addition to w2v's legacy `X-Tenant-Id: Bearer`. This is
    required so w2v can sit behind the `license-authz` ext-authz gate, which
    reads the licence token ONLY from `Authorization`.

    This test previously asserted the inverse ("Authorization is never a license
    source", FR-401 / 0.6.5). That invariant was narrowed, not dropped — see
    `test_extract_license_token_ignores_jwt_shaped_authorization` below, which
    pins the half that still holds and that guarded the 2026-04-26 prod 401s.
    """
    middleware = AuthMiddleware(app=MagicMock())
    request = MagicMock()
    request.headers = {"Authorization": "Bearer test-token-123"}

    assert middleware._extract_license_token(request) == "test-token-123"


def test_extract_license_token_ignores_jwt_shaped_authorization():
    """A JWT-shaped `Authorization: Bearer` is still NEVER a license token.

    This is the surviving half of the old FR-401 invariant and the one worth
    pinning: `dispatch` consumes JWT-shaped bearers on the JWT path, so a JWT
    must never be submitted to licence introspection. w2v_common 0.74.0
    implements this with `_looks_like_jwt` (three non-empty dot-separated
    segments). Regressing it reproduces the 2026-04-26 prod 401s on
    /api/assets/procedures and /api/assets/secrets.
    """
    middleware = AuthMiddleware(app=MagicMock())
    request = MagicMock()
    request.headers = {"Authorization": "Bearer header.payload.signature"}

    assert middleware._extract_license_token(request) is None
    # ...and it still reaches the JWT lane instead.
    assert middleware._extract_jwt_bearer(request) == "header.payload.signature"


def test_extract_license_token_prefers_x_tenant_id_over_authorization():
    """Legacy `X-Tenant-Id: Bearer` wins when both shapes are present (0.74.0).

    `BaseServiceClient` puts the CALLER's delegated token on `X-Tenant-Id` while
    its constructor-baked SERVICE-ACCOUNT token sits on `Authorization`.
    Preferring `Authorization` would silently re-attribute every existing
    delegated call from the customer tenant to the service account's tenant.
    """
    middleware = AuthMiddleware(app=MagicMock())
    request = MagicMock()
    request.headers = {
        "X-Tenant-Id": "Bearer caller-token",
        "Authorization": "Bearer service-account-token",
    }

    assert middleware._extract_license_token(request) == "caller-token"


def test_extract_jwt_bearer_from_authorization():
    """Authorization Bearer is readable on the JWT lane.

    NOTE (w2v_common 0.74.0): "never license introspection" no longer holds for
    an OPAQUE bearer — see
    `test_extract_license_token_accepts_opaque_authorization_bearer`. What this
    pins is only that `_extract_jwt_bearer` parses the header; which lane
    actually consumes it is decided by `_looks_like_jwt`.
    """
    middleware = AuthMiddleware(app=MagicMock())
    request = MagicMock()
    request.headers = {"Authorization": "Bearer test-token-123"}

    assert middleware._extract_jwt_bearer(request) == "test-token-123"


def test_extract_license_token_from_x_tenant_id_bearer():
    """Token can be extracted from X-Tenant-Id with Bearer prefix."""
    middleware = AuthMiddleware(app=MagicMock())
    request = MagicMock()
    request.headers = {"X-Tenant-Id": "Bearer my-license-token"}

    token = middleware._extract_license_token(request)
    assert token == "my-license-token"


def test_extract_license_token_no_token():
    """No token returns None when no auth headers present."""
    middleware = AuthMiddleware(app=MagicMock())
    request = MagicMock()
    request.headers = {}

    token = middleware._extract_license_token(request)
    assert token is None


def test_extract_license_token_x_tenant_id_not_bearer():
    """X-Tenant-Id without Bearer prefix is not treated as a token."""
    middleware = AuthMiddleware(app=MagicMock())
    request = MagicMock()
    request.headers = {"X-Tenant-Id": "plain-tenant-id"}

    token = middleware._extract_license_token(request)
    assert token is None


# ---------- AuthMiddleware.dispatch ----------


@pytest.mark.asyncio
async def test_dispatch_public_endpoint():
    """Public endpoints bypass authentication."""
    middleware = AuthMiddleware(app=MagicMock())
    request = MagicMock()
    request.url.path = "/health"
    request.headers = {}

    expected_response = MagicMock()
    call_next = AsyncMock(return_value=expected_response)

    response = await middleware.dispatch(request, call_next)
    assert response is expected_response
    call_next.assert_awaited_once_with(request)


@pytest.mark.asyncio
async def test_dispatch_public_prefix():
    """Endpoints with public prefixes bypass authentication."""
    middleware = AuthMiddleware(app=MagicMock())
    request = MagicMock()
    request.url.path = "/health/downstream"
    request.headers = {}

    expected_response = MagicMock()
    call_next = AsyncMock(return_value=expected_response)

    response = await middleware.dispatch(request, call_next)
    assert response is expected_response


@pytest.mark.asyncio
async def test_dispatch_trusted_network_no_token_fail_closed_default():
    """w2v_common 0.6.9: trusted-network unauth requests now fail closed (401).

    AUTH_TRUST_INCLUSTER_IDENTITY default flipped to "false" in 0.6.9. Phases 1
    and 2 of the trust-incluster removal shipped, so X-Tenant-Id propagation
    + bare-client license-token plumbing replace the trusted-network fallback.
    """
    middleware = AuthMiddleware(app=MagicMock())
    request = MagicMock()
    request.url.path = "/api/pipeline"
    request.headers = {}
    request.client.host = "127.0.0.1"
    request.state = MagicMock()

    call_next = AsyncMock()

    response = await middleware.dispatch(request, call_next)
    assert response.status_code == 401
    call_next.assert_not_awaited()


def _opt_in_request() -> MagicMock:
    request = MagicMock()
    request.url.path = "/api/pipeline"
    # A real uuid: 0.67.0 parses X-Tenant-Id as a uuid and 401s a slug outright
    # (there is no default tenant to fall back to any more).
    request.headers = {"X-Tenant-Id": str(_TENANT)}
    # 10.42.0.0/16 is the k3s pod CIDR. 192.168.0.0/16 is never trusted — that
    # range is the LAN edge / SSR proxy layer (w2v_common 0.6.8).
    request.client.host = "10.42.0.100"
    request.state = MagicMock()
    return request


@pytest.mark.asyncio
@pytest.mark.usefixtures("restore_middleware_module")
async def test_dispatch_trusted_network_with_x_tenant_id_opt_in(monkeypatch: pytest.MonkeyPatch):
    """Legacy fail-open path needs BOTH opt-ins: the flag AND a named TRUSTED_CIDRS.

    Operators mid-rollback can opt back into the trusted-network fallback with
    AUTH_TRUST_INCLUSTER_IDENTITY=true (sourced from AuthSettings via a function
    accessor since w2v_common 0.6.10, so the accessor is patched). Since
    w2v_common 0.89.0 the peer must also sit inside an explicitly named
    TRUSTED_CIDRS network.
    """
    mod = _reload_with_trusted_cidrs(monkeypatch, "10.42.0.0/16")
    middleware = mod.AuthMiddleware(app=MagicMock())
    request = _opt_in_request()

    expected_response = MagicMock()
    call_next = AsyncMock(return_value=expected_response)

    with patch("w2v_common.auth.middleware._trust_incluster_identity", return_value=True):
        response = await middleware.dispatch(request, call_next)
    assert response is expected_response
    assert request.state.tenant_id == str(_TENANT)


@pytest.mark.asyncio
@pytest.mark.usefixtures("restore_middleware_module")
async def test_dispatch_opt_in_flag_alone_with_default_cidrs_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    """The flag alone trusts nothing: with the default (empty) TRUSTED_CIDRS → 401."""
    mod = _reload_with_trusted_cidrs(monkeypatch, None)
    middleware = mod.AuthMiddleware(app=MagicMock())
    request = _opt_in_request()

    call_next = AsyncMock()

    with patch("w2v_common.auth.middleware._trust_incluster_identity", return_value=True):
        response = await middleware.dispatch(request, call_next)
    assert response.status_code == 401
    call_next.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_untrusted_no_token():
    """Untrusted network without token returns 401."""
    middleware = AuthMiddleware(app=MagicMock())
    request = MagicMock()
    request.url.path = "/api/pipeline"
    request.headers = {}
    request.client.host = "203.0.113.1"
    request.state = MagicMock()

    call_next = AsyncMock()

    response = await middleware.dispatch(request, call_next)
    assert response.status_code == 401
    call_next.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_valid_license_token():
    """Valid license token sets user and tenant on request state."""
    middleware = AuthMiddleware(app=MagicMock())
    request = MagicMock()
    request.url.path = "/api/pipeline"
    request.headers = {"X-Tenant-Id": "Bearer valid-token"}
    request.client.host = "203.0.113.1"
    request.state = MagicMock()

    expected_response = MagicMock()
    call_next = AsyncMock(return_value=expected_response)

    with patch(
        "w2v_common.auth.middleware._introspect_license",
        return_value={"active": True, "tenant_id": str(_TENANT), "subject_id": "user-1"},
    ):
        response = await middleware.dispatch(request, call_next)
        assert response is expected_response
        assert request.state.tenant_id == str(_TENANT)


@pytest.mark.asyncio
async def test_dispatch_invalid_license_token():
    """Invalid license token returns 401."""
    middleware = AuthMiddleware(app=MagicMock())
    request = MagicMock()
    request.url.path = "/api/pipeline"
    request.headers = {"Authorization": "Bearer invalid-token"}
    request.client.host = "203.0.113.1"
    request.state = MagicMock()

    call_next = AsyncMock()

    with patch(
        "w2v_common.auth.middleware._introspect_license",
        return_value={"active": False, "reason": "expired"},
    ):
        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 401
        call_next.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_introspection_returns_none():
    """When introspection returns None, returns 401."""
    middleware = AuthMiddleware(app=MagicMock())
    request = MagicMock()
    request.url.path = "/api/pipeline"
    request.headers = {"Authorization": "Bearer network-fail-token"}
    request.client.host = "203.0.113.1"
    request.state = MagicMock()

    call_next = AsyncMock()

    with patch("w2v_common.auth.middleware._introspect_license", return_value=None):
        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_dispatch_root_path_is_public():
    """Root / path is public."""
    middleware = AuthMiddleware(app=MagicMock())
    request = MagicMock()
    request.url.path = "/"
    request.headers = {}

    expected_response = MagicMock()
    call_next = AsyncMock(return_value=expected_response)

    response = await middleware.dispatch(request, call_next)
    assert response is expected_response


@pytest.mark.asyncio
async def test_dispatch_trailing_slash_stripped():
    """Trailing slash is stripped for path matching."""
    middleware = AuthMiddleware(app=MagicMock())
    request = MagicMock()
    request.url.path = "/health/"
    request.headers = {}

    expected_response = MagicMock()
    call_next = AsyncMock(return_value=expected_response)

    response = await middleware.dispatch(request, call_next)
    assert response is expected_response
