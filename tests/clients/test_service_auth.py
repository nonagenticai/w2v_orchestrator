"""ServiceAuthenticator — client-credentials token cache (Phase C-2)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from src.clients.service_auth import ServiceAuthenticator, ServiceAuthError

pytestmark = pytest.mark.asyncio


def _auth(handler, **overrides) -> ServiceAuthenticator:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    kwargs = {
        "keycloak_url": "http://kc",
        "realm": "portal",
        "client_id": "w2v-backend",
        "client_secret": "secret",  # pragma: allowlist secret
        "client": client,
    }
    kwargs.update(overrides)
    return ServiceAuthenticator(**kwargs)


async def test_mints_and_caches() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/realms/portal/protocol/openid-connect/token"
        calls["n"] += 1
        return httpx.Response(200, json={"access_token": f"tok-{calls['n']}", "expires_in": 3600})

    auth = _auth(handler)
    assert await auth.token() == "tok-1"
    assert await auth.token() == "tok-1"  # cached, no second mint
    assert calls["n"] == 1


async def test_refreshes_when_expired() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"access_token": f"tok-{calls['n']}", "expires_in": 3600})

    auth = _auth(handler)
    assert await auth.token() == "tok-1"
    auth._expires_at = 0.0  # force staleness
    assert await auth.token() == "tok-2"
    assert calls["n"] == 2


async def test_concurrent_refresh_is_deduped() -> None:
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        await asyncio.sleep(0)  # yield so concurrent callers pile on the lock
        return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})

    auth = _auth(handler)
    tokens = await asyncio.gather(auth.token(), auth.token(), auth.token())
    assert tokens == ["tok", "tok", "tok"]
    assert calls["n"] == 1  # only one mint despite three concurrent callers


async def test_raises_on_non_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad client")

    with pytest.raises(ServiceAuthError, match="401"):
        await _auth(handler).token()


async def test_raises_when_token_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"expires_in": 3600})

    with pytest.raises(ServiceAuthError, match="missing access_token"):
        await _auth(handler).token()


@pytest.mark.parametrize(
    "response",
    [httpx.Response(200, text="<html>bad gateway</html>")],
)
async def test_malformed_token_body_is_a_service_auth_error(response: httpx.Response) -> None:
    auth = _auth(lambda request: response)
    with pytest.raises(ServiceAuthError):
        await auth.token()


async def test_unparseable_expires_in_does_not_crash() -> None:
    auth = _auth(
        lambda request: httpx.Response(200, json={"access_token": "tok", "expires_in": "soon"})
    )
    assert await auth.token() == "tok"
