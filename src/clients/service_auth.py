"""OAuth2 client-credentials token minter (admin tier Phase C-2).

The Keycloak-Admin and licenses clients authorize a Keycloak **client-credentials**
Bearer token (the ``w2v-backend`` service account), NOT the service-account
license token the license-authz peers use. This mints + caches that token,
refreshing 60s before expiry and de-duplicating concurrent refreshes behind an
``asyncio.Lock`` — mirrors fr's ``ServiceAuthenticator`` (``fr/backend/src/lib/
service-auth.ts``).

Stateless across requests via a module-level singleton (``get_service_authenticator``)
so the cached token survives between requests within a pod.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol

import httpx

from src.config import get_settings


class TokenProvider(Protocol):
    """Anything that can supply a Bearer token (real authenticator or a test fake)."""

    async def token(self) -> str: ...


# Refresh this many seconds before the token's stated expiry, so an in-flight
# call never races a same-instant expiry (matches fr's 60s buffer).
_REFRESH_BUFFER_SECONDS = 60.0
_TOKEN_TIMEOUT_SECONDS = 10.0


class ServiceAuthError(RuntimeError):
    """Raised when a client-credentials token cannot be minted."""


class ServiceAuthenticator:
    """Caches a Keycloak client-credentials access token, refreshing on expiry."""

    def __init__(
        self,
        *,
        keycloak_url: str,
        realm: str,
        client_id: str,
        client_secret: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._token_url = f"{keycloak_url.rstrip('/')}/realms/{realm}/protocol/openid-connect/token"
        self._client_id = client_id
        self._client_secret = client_secret
        self._client = client
        self._lock = asyncio.Lock()
        self._access_token: str | None = None
        self._expires_at: float = 0.0

    def _now(self) -> float:
        return time.monotonic()

    def _fresh(self) -> bool:
        return self._access_token is not None and self._now() < self._expires_at

    async def token(self) -> str:
        """Return a valid access token, minting/refreshing under the lock."""
        if self._fresh() and self._access_token is not None:
            return self._access_token
        async with self._lock:
            # Another coroutine may have refreshed while we waited for the lock.
            if not (self._fresh() and self._access_token is not None):
                await self._refresh()
            if self._access_token is None:
                raise ServiceAuthError("no access token after refresh")
            return self._access_token

    async def _refresh(self) -> None:
        data = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        try:
            if self._client is not None:
                resp = await self._client.post(self._token_url, data=data)
            else:
                async with httpx.AsyncClient(timeout=_TOKEN_TIMEOUT_SECONDS) as client:
                    resp = await client.post(self._token_url, data=data)
        except httpx.HTTPError as exc:
            raise ServiceAuthError(f"token request failed: {exc}") from exc
        if resp.status_code != 200:
            raise ServiceAuthError(f"token endpoint returned {resp.status_code}: {resp.text[:200]}")
        # A 200 with a non-JSON body (a proxy's HTML error page) or a non-object
        # JSON is a token-endpoint failure, not a crash: map it onto the one error
        # type callers handle instead of leaking ValueError/AttributeError.
        try:
            payload = resp.json()
        except ValueError as exc:
            raise ServiceAuthError(f"token endpoint returned non-JSON: {resp.text[:200]}") from exc
        if not isinstance(payload, dict):
            raise ServiceAuthError("token response is not a JSON object")
        token = payload.get("access_token")
        if not token:
            raise ServiceAuthError("token response missing access_token")
        # Refresh slightly before the stated lifetime; clamp so a tiny/absent
        # expires_in still yields a non-negative window.
        try:
            expires_in = float(payload.get("expires_in", 0) or 0)
        except TypeError, ValueError:
            expires_in = 0.0
        self._access_token = token
        self._expires_at = self._now() + max(0.0, expires_in - _REFRESH_BUFFER_SECONDS)


_authenticator: ServiceAuthenticator | None = None


def get_service_authenticator() -> ServiceAuthenticator:
    """The S2S token (licenses, identity invites): ``KEYCLOAK_CLIENT_ID`` minted in
    ``KEYCLOAK_SERVICE_REALM`` (default ``KEYCLOAK_REALM``). Module-level singleton fed
    from settings (token cache survives requests)."""
    global _authenticator
    if _authenticator is None:
        s = get_settings().provisioning
        _authenticator = ServiceAuthenticator(
            keycloak_url=s.keycloak_url,
            realm=s.service_realm,
            client_id=s.keycloak_client_id,
            client_secret=s.keycloak_client_secret.get_secret_value(),
        )
    return _authenticator


_admin_fallback_authenticator: ServiceAuthenticator | None = None


def _keycloak_realm_authenticator() -> ServiceAuthenticator:
    """``KEYCLOAK_CLIENT_ID`` minted in ``KEYCLOAK_REALM`` — the admin-REST fallback.

    The same singleton as :func:`get_service_authenticator` while the two realms are
    equal (``KEYCLOAK_SERVICE_REALM`` unset), so nothing changes until it is set."""
    global _admin_fallback_authenticator
    s = get_settings().provisioning
    if s.service_realm == s.keycloak_realm:
        return get_service_authenticator()
    if _admin_fallback_authenticator is None:
        _admin_fallback_authenticator = ServiceAuthenticator(
            keycloak_url=s.keycloak_url,
            realm=s.keycloak_realm,
            client_id=s.keycloak_client_id,
            client_secret=s.keycloak_client_secret.get_secret_value(),
        )
    return _admin_fallback_authenticator


_user_realm_authenticator: ServiceAuthenticator | None = None


def get_user_realm_admin_authenticator() -> TokenProvider:
    """The token the Keycloak Admin REST of the USER realm is called with.

    ``KEYCLOAK_USER_REALM_ADMIN_CLIENT_ID``/``_SECRET`` set ⇒ that client's own
    client-credentials token, minted from ``KEYCLOAK_USER_REALM``. Unset ⇒
    ``KEYCLOAK_CLIENT_ID`` minted in ``KEYCLOAK_REALM`` (never ``KEYCLOAK_SERVICE_REALM``:
    a secure-apps token holds no admin role in any user realm), exactly as before. Licenses and identity invites never use this one: they stay
    on :func:`get_service_authenticator`.
    """
    global _user_realm_authenticator
    s = get_settings().provisioning
    if not s.user_realm_has_admin_client:
        return _keycloak_realm_authenticator()
    if _user_realm_authenticator is None:
        _user_realm_authenticator = ServiceAuthenticator(
            keycloak_url=s.keycloak_url,
            realm=s.user_realm,
            client_id=s.keycloak_user_realm_admin_client_id.strip(),
            client_secret=s.keycloak_user_realm_admin_client_secret.get_secret_value().strip(),
        )
    return _user_realm_authenticator
