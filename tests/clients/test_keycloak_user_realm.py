"""Realm-per-app: Studio's USER realm is separate from the service-token realm.

KEYCLOAK_REALM (portal) keeps minting the w2v-backend service token that licenses
and identity invites authorize; KEYCLOAK_USER_REALM decides which realm the
early-access approve / removal / backfill administer, and the optional
KEYCLOAK_USER_REALM_ADMIN_CLIENT_ID/_SECRET pair authenticates there
(commonpractice #480 pattern).
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from pydantic import ValidationError

from src.clients import service_auth
from src.clients.keycloak_admin import get_keycloak_admin
from src.clients.service_auth import (
    ServiceAuthenticator,
    get_service_authenticator,
    get_user_realm_admin_authenticator,
)
from src.config import ProvisioningSettings, clear_settings_cache

_ENV = (
    "KEYCLOAK_URL",
    "KEYCLOAK_REALM",
    "KEYCLOAK_SERVICE_REALM",
    "KEYCLOAK_CLIENT_ID",
    "KEYCLOAK_CLIENT_SECRET",
    "KEYCLOAK_USER_REALM",
    "KEYCLOAK_USER_REALM_ADMIN_CLIENT_ID",
    "KEYCLOAK_USER_REALM_ADMIN_CLIENT_SECRET",
)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KEYCLOAK_URL", "http://kc")
    monkeypatch.setenv("KEYCLOAK_REALM", "portal")
    monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "w2v-backend")
    monkeypatch.setenv("KEYCLOAK_CLIENT_SECRET", "svc-secret")  # pragma: allowlist secret
    monkeypatch.setattr(service_auth, "_authenticator", None)
    monkeypatch.setattr(service_auth, "_user_realm_authenticator", None)
    monkeypatch.setattr(service_auth, "_admin_fallback_authenticator", None)
    clear_settings_cache()
    yield monkeypatch
    clear_settings_cache()


def _settings() -> ProvisioningSettings:
    return ProvisioningSettings()


def test_user_realm_defaults_to_keycloak_realm(env: pytest.MonkeyPatch) -> None:
    s = _settings()
    assert s.keycloak_user_realm == ""
    assert s.user_realm == "portal"
    assert not s.user_realm_has_admin_client


def test_user_realm_reads_its_env(env: pytest.MonkeyPatch) -> None:
    env.setenv("KEYCLOAK_USER_REALM", " w2v ")
    s = _settings()
    assert s.user_realm == "w2v"
    assert s.keycloak_realm == "portal"  # the service-token realm does not move


@pytest.mark.parametrize(
    ("client_id", "secret"),
    [("w2v-admin", "")],
)
def test_user_realm_admin_client_is_both_or_neither(
    env: pytest.MonkeyPatch, client_id: str, secret: str
) -> None:
    env.setenv("KEYCLOAK_USER_REALM_ADMIN_CLIENT_ID", client_id)
    env.setenv("KEYCLOAK_USER_REALM_ADMIN_CLIENT_SECRET", secret)
    with pytest.raises(ValidationError, match="set both or neither"):
        _settings()


def test_user_realm_admin_client_both_set_is_accepted(env: pytest.MonkeyPatch) -> None:
    env.setenv("KEYCLOAK_USER_REALM_ADMIN_CLIENT_ID", "w2v-admin")
    env.setenv(
        "KEYCLOAK_USER_REALM_ADMIN_CLIENT_SECRET", "zq-sentinel-9f"
    )  # pragma: allowlist secret
    s = _settings()
    assert s.user_realm_has_admin_client
    assert "zq-sentinel-9f" not in repr(s)  # SecretStr: never rendered


def test_unset_is_byte_identical_to_before(env: pytest.MonkeyPatch) -> None:
    """No new env ⇒ admin REST on portal, authorized by the one service authenticator."""
    auth = get_user_realm_admin_authenticator()
    assert auth is get_service_authenticator()
    kc = get_keycloak_admin()
    assert kc._base == "http://kc/admin/realms/portal"
    assert kc._auth is get_service_authenticator()


def test_user_realm_without_own_client_moves_only_the_target(env: pytest.MonkeyPatch) -> None:
    env.setenv("KEYCLOAK_USER_REALM", "w2v")
    clear_settings_cache()
    kc = get_keycloak_admin()
    assert kc._base == "http://kc/admin/realms/w2v"
    assert kc._auth is get_service_authenticator()


def test_user_realm_with_own_client_mints_from_the_user_realm(env: pytest.MonkeyPatch) -> None:
    env.setenv("KEYCLOAK_USER_REALM", "w2v")
    env.setenv("KEYCLOAK_USER_REALM_ADMIN_CLIENT_ID", "w2v-admin")
    env.setenv("KEYCLOAK_USER_REALM_ADMIN_CLIENT_SECRET", "adm")  # pragma: allowlist secret
    clear_settings_cache()
    kc = get_keycloak_admin()
    assert kc._base == "http://kc/admin/realms/w2v"
    admin_auth = kc._auth
    assert isinstance(admin_auth, ServiceAuthenticator)
    assert admin_auth is not get_service_authenticator()
    assert admin_auth is get_user_realm_admin_authenticator()  # cached singleton
    assert admin_auth._token_url == "http://kc/realms/w2v/protocol/openid-connect/token"
    assert admin_auth._client_id == "w2v-admin"
    # The licences / identity-invite token stays the portal w2v-backend SA.
    svc = get_service_authenticator()
    assert svc._token_url == "http://kc/realms/portal/protocol/openid-connect/token"
    assert svc._client_id == "w2v-backend"


@pytest.mark.asyncio
async def test_admin_calls_carry_the_user_realm_token(env: pytest.MonkeyPatch) -> None:
    """End to end over a mock transport: the token comes from realm w2v's w2v-admin
    and the admin call targets /admin/realms/w2v."""
    env.setenv("KEYCLOAK_USER_REALM", "w2v")
    env.setenv("KEYCLOAK_USER_REALM_ADMIN_CLIENT_ID", "w2v-admin")
    env.setenv("KEYCLOAK_USER_REALM_ADMIN_CLIENT_SECRET", "adm")  # pragma: allowlist secret
    clear_settings_cache()
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("authorization", "")))
        if request.url.path.endswith("/protocol/openid-connect/token"):
            body = request.content.decode()
            assert "client_id=w2v-admin" in body
            return httpx.Response(200, json={"access_token": "w2v-tok", "expires_in": 300})
        return httpx.Response(404)

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    s = _settings()
    auth = ServiceAuthenticator(
        keycloak_url=s.keycloak_url,
        realm=s.user_realm,
        client_id=s.keycloak_user_realm_admin_client_id,
        client_secret=s.keycloak_user_realm_admin_client_secret.get_secret_value(),
        client=transport,
    )
    env.setattr(service_auth, "_user_realm_authenticator", auth)
    kc = get_keycloak_admin()
    kc._client = transport
    assert await kc.get_user("u-1") is None
    assert seen == [
        ("/realms/w2v/protocol/openid-connect/token", ""),
        ("/admin/realms/w2v/users/u-1", "Bearer w2v-tok"),
    ]


# ─── realm-per-app P7 chain C: KEYCLOAK_SERVICE_REALM ─────────────────────────


def test_service_realm_defaults_to_keycloak_realm(env: pytest.MonkeyPatch) -> None:
    s = _settings()
    assert s.keycloak_service_realm == ""
    assert s.service_realm == "portal"


def test_service_realm_moves_only_the_s2s_token(env: pytest.MonkeyPatch) -> None:
    """licenses / identity invites mint from secure-apps; the admin REST fallback stays
    on KEYCLOAK_REALM (a secure-apps token holds no admin role in any user realm)."""
    env.setenv("KEYCLOAK_SERVICE_REALM", " secure-apps ")
    clear_settings_cache()
    svc = get_service_authenticator()
    assert svc._token_url == "http://kc/realms/secure-apps/protocol/openid-connect/token"
    assert svc._client_id == "w2v-backend"
    admin = get_user_realm_admin_authenticator()
    assert admin is not svc
    assert isinstance(admin, ServiceAuthenticator)
    assert admin._token_url == "http://kc/realms/portal/protocol/openid-connect/token"
    assert admin._client_id == "w2v-backend"
    kc = get_keycloak_admin()
    assert kc._base == "http://kc/admin/realms/portal"
    assert kc._auth is admin


def test_service_realm_with_user_realm_admin_client(env: pytest.MonkeyPatch) -> None:
    """The prod shape: S2S from secure-apps, users in w2v administered by w2v-admin."""
    env.setenv("KEYCLOAK_SERVICE_REALM", "secure-apps")
    env.setenv("KEYCLOAK_USER_REALM", "w2v")
    env.setenv("KEYCLOAK_USER_REALM_ADMIN_CLIENT_ID", "w2v-admin")
    env.setenv("KEYCLOAK_USER_REALM_ADMIN_CLIENT_SECRET", "adm")  # pragma: allowlist secret
    clear_settings_cache()
    assert get_service_authenticator()._token_url.endswith(
        "/realms/secure-apps/protocol/openid-connect/token"
    )
    kc = get_keycloak_admin()
    assert kc._base == "http://kc/admin/realms/w2v"
    admin = kc._auth
    assert isinstance(admin, ServiceAuthenticator)
    assert admin._token_url == "http://kc/realms/w2v/protocol/openid-connect/token"


def test_service_realm_equal_to_keycloak_realm_is_one_singleton(env: pytest.MonkeyPatch) -> None:
    env.setenv("KEYCLOAK_SERVICE_REALM", "portal")
    clear_settings_cache()
    assert get_user_realm_admin_authenticator() is get_service_authenticator()
