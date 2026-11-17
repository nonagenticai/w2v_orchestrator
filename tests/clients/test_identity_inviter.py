"""IdentityInviter — provisioning's identity invite API, faked at the HTTP boundary."""

from __future__ import annotations

import json
import uuid

import httpx
import pytest

from src.clients import identity_inviter as mod
from src.clients.identity_inviter import (
    IDENTITY_INVITE_NOT_CONFIGURED,
    IDENTITY_INVITE_REJECTED,
    IDENTITY_PROVIDER_FORBIDDEN,
    IDENTITY_RATE_LIMITED,
    KEYCLOAK_UNAVAILABLE,
    IdentityInviteError,
    IdentityInviter,
)
from src.clients.service_auth import ServiceAuthError

pytestmark = pytest.mark.asyncio

_BASE = "https://provisioning.test"
_REDIRECT = "https://w2v.example.test/"


class _Auth:
    async def token(self) -> str:
        return "sa-token"


class _BrokenAuth:
    async def token(self) -> str:
        raise ServiceAuthError("token endpoint returned 401")


def _inviter(
    handler, *, auth=None, base: str = _BASE, redirect: str = _REDIRECT
) -> IdentityInviter:
    return IdentityInviter(
        authenticator=auth or _Auth(),
        base_url=base,
        client_id="w2v-frontend",
        default_redirect_uri=redirect,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def _fails(handler, **kw) -> IdentityInviteError:
    with pytest.raises(IdentityInviteError) as info:
        await _inviter(handler, **kw).invite("a@b.test")
    return info.value


_ACCEPTED = {"accepted": True}


async def test_posts_the_contract_body_with_the_service_token() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(202, json=_ACCEPTED)

    actor = uuid.uuid4()
    tenant = uuid.uuid4()
    out = await _inviter(handler).invite(
        "  New@Example.test ", actor=actor, context=mod.tenant_context(tenant)
    )
    assert out is None  # nothing about the account comes back
    [req] = seen
    assert req.method == "POST"
    assert str(req.url) == f"{_BASE}/api/identity/invites"
    assert req.headers["authorization"] == "Bearer sa-token"
    # Exactly the contract's fields — never attributes/credentials/roles.
    assert json.loads(req.content) == {
        "email": "New@Example.test",
        "client_id": "w2v-frontend",
        "redirect_uri": _REDIRECT,
        "actor": str(actor),
        "context": f"w2v:tenant:{tenant}",
    }


async def test_audit_fields_are_optional_and_redirect_overrides_the_default() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(202, json=_ACCEPTED)

    link = "https://w2v.example.test/invite/accept?token=abc"
    await _inviter(handler).invite("a@b.test", redirect_uri=link)
    assert bodies == [{"email": "a@b.test", "client_id": "w2v-frontend", "redirect_uri": link}]


@pytest.mark.parametrize(
    ("exc", "uncertain"),
    [(httpx.ReadTimeout, True)],
)
async def test_only_a_connect_phase_failure_is_certain(exc: type, uncertain: bool) -> None:
    """Connect-phase failures never reached provisioning (safe to retry); anything
    after the request may have been sent is uncertain (never auto-resent)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise exc("boom", request=request)

    err = await _fails(handler)
    assert err.code == KEYCLOAK_UNAVAILABLE
    assert err.uncertain is uncertain


@pytest.mark.parametrize(
    "resp",
    [httpx.Response(202, json={"accepted": False})],
)
async def test_a_2xx_outside_the_contract_is_unavailable(resp: httpx.Response) -> None:
    err = await _fails(lambda r: resp)
    assert err.code == KEYCLOAK_UNAVAILABLE
    assert err.uncertain, "a 2xx means provisioning processed the request"


@pytest.mark.parametrize("status", [401])
async def test_401_403_is_forbidden(status: int) -> None:
    err = await _fails(lambda r: httpx.Response(status, json={"detail": "not allowed"}))
    assert err.code == IDENTITY_PROVIDER_FORBIDDEN
    assert err.http_status == 502
    assert err.upstream_status == status


@pytest.mark.parametrize(
    "resp",
    [httpx.Response(502, json={"detail": {"code": "IDP_UNAVAILABLE"}})],
)
async def test_5xx_and_unexpected_are_unavailable(resp: httpx.Response) -> None:
    err = await _fails(lambda r: resp)
    assert err.code == KEYCLOAK_UNAVAILABLE
    assert err.http_status == 502
    assert err.upstream_status == resp.status_code
    # A 5xx answered AFTER provisioning got the request: it may have sent the email.
    assert err.uncertain is (resp.status_code >= 500)


@pytest.mark.parametrize("status", [401])
async def test_a_refusal_is_certain(status: int) -> None:
    err = await _fails(lambda r: httpx.Response(status, json={"detail": "no"}))
    assert not err.uncertain, "a refusal sent nothing: safe to retry"


async def test_transport_failure_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    err = await _fails(handler)
    assert err.code == KEYCLOAK_UNAVAILABLE
    assert err.upstream_status is None


async def test_token_mint_failure_is_unavailable_and_never_calls_provisioning() -> None:
    calls: list[httpx.Request] = []
    err = await _fails(
        lambda r: calls.append(r) or httpx.Response(202, json=_ACCEPTED), auth=_BrokenAuth()
    )
    assert err.code == KEYCLOAK_UNAVAILABLE
    assert calls == []


async def test_429_surfaces_retry_after() -> None:
    err = await _fails(lambda r: httpx.Response(429, headers={"Retry-After": "17"}))
    assert err.code == IDENTITY_RATE_LIMITED
    assert err.http_status == 429
    assert err.retry_after == 17

    no_header = await _fails(lambda r: httpx.Response(429))
    assert no_header.retry_after is None


async def test_422_is_rejected() -> None:
    err = await _fails(lambda r: httpx.Response(422, json={"detail": "redirect_uri not allowed"}))
    assert err.code == IDENTITY_INVITE_REJECTED


async def test_malformed_200_is_unavailable_not_a_crash() -> None:
    err = await _fails(lambda r: httpx.Response(200, json=["accepted"]))
    assert err.code == KEYCLOAK_UNAVAILABLE


@pytest.mark.parametrize(("base", "redirect"), [("", _REDIRECT)])
async def test_unconfigured_refuses_without_a_call(base: str, redirect: str) -> None:
    calls: list[httpx.Request] = []
    err = await _fails(
        lambda r: calls.append(r) or httpx.Response(202, json=_ACCEPTED),
        base=base,
        redirect=redirect,
    )
    assert err.code == IDENTITY_INVITE_NOT_CONFIGURED
    assert err.http_status == 503
    assert calls == []


async def test_factory_reads_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import clear_settings_cache

    monkeypatch.setenv("PROVISIONING_URL", "https://provisioning.test/")
    monkeypatch.setenv("IDENTITY_INVITE_CLIENT_ID", "w2v-frontend")
    monkeypatch.delenv("IDENTITY_INVITE_REDIRECT_URI", raising=False)
    monkeypatch.setenv("W2V_FRONTEND_URL", "https://w2v.example.test")
    clear_settings_cache()
    try:
        assert mod.default_redirect_uri() == "https://w2v.example.test/"
        monkeypatch.setenv("IDENTITY_INVITE_REDIRECT_URI", "http://localhost:6183/")
        clear_settings_cache()
        inviter = mod.get_identity_inviter()
        assert inviter._base == "https://provisioning.test"
        assert inviter._default_redirect == "http://localhost:6183/"
        assert inviter._client_id == "w2v-frontend"
    finally:
        clear_settings_cache()
