"""Keycloak Admin REST client (``{keycloak_url}/admin/realms/{realm}/...``).

Administers Studio's USER realm (``KEYCLOAK_USER_REALM``, default
``KEYCLOAK_REALM``). Authorized by that realm's own admin client
(``KEYCLOAK_USER_REALM_ADMIN_CLIENT_ID``) when configured, else by the
``w2v-backend`` client-credentials Bearer from :class:`ServiceAuthenticator`.
Callers, nothing else:

* the one-shot tenancy backfill (``scripts/backfill_tenancy.py``): a user's
  representation and effective realm roles (read-only);
* the early-access approve (``src/services/signup/identity.py``, George
  2026-10-20: w2v early access works EXACTLY like CommonPractice and fr): mint
  the requester's credential-less realm user with firstName/lastName (so
  Keycloak shows no "Update account information" form), stamped
  ``w2v_signup_id = [<signup row id>]``, and send its one
  ``execute-actions-email``. Needs the realm-management role ``manage-users``.
* the early-access REMOVAL (``src/services/signup/removal.py``): retire the
  login an approved signup minted — delete it when it never got a credential,
  else disable + mark it (``set_user_state``, CommonPractice
  ``retireRejectedIdentity``). Same role.

The ordinary tenant-member invite does NOT come here: it stays on
provisioning's identity invite API (``src/clients/identity_inviter.py``). A
guard test (``tests/clients/test_keycloak_admin.py``) keeps the write surface
confined to the signup identity module.
"""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import quote

import httpx

from src.clients.schemas.keycloak import KeycloakRoleMappings, KeycloakUser
from src.clients.service_auth import (
    ServiceAuthError,
    TokenProvider,
    get_user_realm_admin_authenticator,
)
from src.config import get_settings

_TIMEOUT_SECONDS = 15.0

#: Transport failures that happen BEFORE a byte of the request is sent: Keycloak
#: cannot have acted. Every other transport failure (read/write timeout, dropped
#: connection, protocol error mid-response) may come after it did — ``uncertain``.
_NOT_SENT = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ProxyError,
    httpx.UnsupportedProtocol,
)


class KeycloakAdminError(RuntimeError):
    """A Keycloak admin call failed.

    ``status_code`` is the admin API's HTTP status when there was one (None for a
    transport failure or a malformed-but-successful response). ``uncertain``: the
    request may have been processed — ANY 5xx (Keycloak, or a gateway in front
    of it, may have acted before answering 500/502/503/504) or a transport
    failure after connecting (read/write timeout, dropped connection). For a
    side-effecting call (the action email) the outcome is UNKNOWN and must not
    be blindly repeated. Definite (retryable): a 4xx, or a transport failure
    before a byte was sent (connect refused / connect timeout / pool timeout).
    """

    def __init__(
        self, message: str, *, status_code: int | None = None, uncertain: bool = False
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.uncertain = uncertain


class KeycloakAdminForbiddenError(KeycloakAdminError):
    """The admin API refused the call (403): the service account lacks the
    realm-management role it needs. Keycloak is up — retrying will not help."""


def _unexpected(op: str, resp: httpx.Response) -> KeycloakAdminError:
    """The error for an admin call that returned ``resp`` unexpectedly."""
    cls = KeycloakAdminForbiddenError if resp.status_code == 403 else KeycloakAdminError
    return cls(
        f"{op} returned {resp.status_code}: {resp.text[:200]}",
        status_code=resp.status_code,
        uncertain=resp.status_code >= 500,
    )


def _q(segment: str) -> str:
    return quote(segment, safe="")


class KeycloakAdminClient:
    def __init__(
        self,
        *,
        authenticator: TokenProvider,
        keycloak_url: str,
        realm: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._auth = authenticator
        self._configured = bool(keycloak_url.strip())
        self._base = f"{keycloak_url.rstrip('/')}/admin/realms/{realm}"
        self._client = client

    async def _request(
        self,
        op: str,
        method: str,
        path: str,
        *,
        params: dict[str, str | int] | None = None,
        json: object = None,
    ) -> httpx.Response:
        if not self._configured:
            raise KeycloakAdminError(f"{op}: KEYCLOAK_URL is not configured")
        try:
            token = await self._auth.token()
        except ServiceAuthError as exc:
            raise KeycloakAdminError(f"{op}: could not mint this service's token: {exc}") from exc
        headers = {"Authorization": f"Bearer {token}"}
        url = f"{self._base}{path}"
        try:
            if self._client is not None:
                return await self._client.request(
                    method, url, params=params, json=json, headers=headers
                )
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                return await client.request(method, url, params=params, json=json, headers=headers)
        except (httpx.HTTPError, OSError) as exc:
            raise KeycloakAdminError(
                f"{op}: Keycloak unreachable ({type(exc).__name__})",
                uncertain=not isinstance(exc, _NOT_SENT),
            ) from exc

    # ─── users: reads ────────────────────────────────────────────────────

    async def get_user(self, user_id: str) -> KeycloakUser | None:
        """The user representation, or None when Keycloak has no such user (404)."""
        resp = await self._request("get_user", "GET", f"/users/{_q(user_id)}")
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise _unexpected("get_user", resp)
        return KeycloakUser.model_validate(resp.json())

    async def find_user_by_email(self, email: str) -> KeycloakUser | None:
        """The ONE realm user whose email is exactly ``email`` (normalized), or
        None. Two matches is an error, never a guess."""
        params: dict[str, str | int] = {
            "email": email.strip().lower(),
            "exact": "true",
            "max": 2,
        }
        resp = await self._request("find_user_by_email", "GET", "/users", params=params)
        if resp.status_code != 200:
            raise _unexpected("find_user_by_email", resp)
        users = resp.json()
        if not isinstance(users, list):
            raise KeycloakAdminError("find_user_by_email: not a list", status_code=200)
        if len(users) > 1:
            raise KeycloakAdminError(
                "find_user_by_email: two realm users carry this address", status_code=200
            )
        return KeycloakUser.model_validate(users[0]) if users else None

    async def count_credentials(self, user_id: str) -> int:
        """How many credentials (password, OTP, …) the user has
        (``GET /users/{id}/credentials``)."""
        return await self._count("count_credentials", f"/users/{_q(user_id)}/credentials")

    async def count_federated_identities(self, user_id: str) -> int:
        """How many identity-provider links the user has
        (``GET /users/{id}/federated-identity``)."""
        return await self._count(
            "count_federated_identities", f"/users/{_q(user_id)}/federated-identity"
        )

    async def _count(self, op: str, path: str) -> int:
        resp = await self._request(op, "GET", path)
        if resp.status_code != 200:
            raise _unexpected(op, resp)
        items = resp.json()
        if not isinstance(items, list):
            raise KeycloakAdminError(f"{op}: not a list", status_code=200)
        return len(items)

    async def count_groups(self, user_id: str) -> int:
        """How many groups the user is a member of (``GET /users/{id}/groups``)."""
        return await self._count("count_groups", f"/users/{_q(user_id)}/groups")

    async def get_user_role_mappings(self, user_id: str) -> KeycloakRoleMappings:
        """The user's DIRECT realm + client role mappings
        (``GET /users/{id}/role-mappings``; composites not expanded)."""
        resp = await self._request(
            "get_user_role_mappings", "GET", f"/users/{_q(user_id)}/role-mappings"
        )
        if resp.status_code != 200:
            raise _unexpected("get_user_role_mappings", resp)
        body = resp.json()
        if not isinstance(body, dict):
            raise KeycloakAdminError("get_user_role_mappings: not an object", status_code=200)
        realm = [
            str(r["name"])
            for r in body.get("realmMappings") or []
            if isinstance(r, dict) and r.get("name")
        ]
        clients: dict[str, list[str]] = {}
        for key, entry in (body.get("clientMappings") or {}).items():
            if not isinstance(entry, dict):
                continue
            names = [
                str(m["name"])
                for m in entry.get("mappings") or []
                if isinstance(m, dict) and m.get("name")
            ]
            clients[str(entry.get("client") or key)] = names
        return KeycloakRoleMappings(realm=sorted(realm), clients=clients)

    # ─── users: writes (early-access identity only) ──────────────────────

    async def create_user(self, representation: Mapping[str, object]) -> str | None:
        """Create a user; its id (from the ``Location`` header), or None on 409
        (username / email already taken)."""
        resp = await self._request("create_user", "POST", "/users", json=representation)
        if resp.status_code == 409:
            return None
        if resp.status_code != 201:
            raise _unexpected("create_user", resp)
        user_id = resp.headers.get("location", "").rstrip("/").rsplit("/", 1)[-1]
        if not user_id or user_id == "users":
            raise KeycloakAdminError("create_user: 201 without a Location id", status_code=201)
        return user_id

    async def update_user_attributes(self, user_id: str, attrs: dict[str, str | None]) -> None:
        """Merge (value) or remove (None) attributes. Read-modify-write of the
        WHOLE representation: Keycloak's PUT replaces it, so every other field
        and attribute (``tenant_id``, …) is carried over unchanged."""
        path = f"/users/{_q(user_id)}"
        read = await self._request("update_user_attributes", "GET", path)
        if read.status_code != 200:
            raise _unexpected("update_user_attributes(read)", read)
        rep = read.json()
        if not isinstance(rep, dict):
            raise KeycloakAdminError("update_user_attributes: not an object", status_code=200)
        merged: dict[str, object] = dict(rep.get("attributes") or {})
        for key, value in attrs.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = [value]
        write = await self._request(
            "update_user_attributes", "PUT", path, json={**rep, "attributes": merged}
        )
        if write.status_code not in (200, 204):
            raise _unexpected("update_user_attributes", write)

    async def execute_actions_email(
        self,
        user_id: str,
        actions: list[str],
        *,
        client_id: str,
        redirect_uri: str,
        lifespan_seconds: int,
    ) -> None:
        """``PUT /users/{id}/execute-actions-email``: Keycloak mails the user a
        link that performs ``actions`` and then lands on ``redirect_uri``
        (validated against ``client_id``'s redirect URIs)."""
        params: dict[str, str | int] = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "lifespan": lifespan_seconds,
        }
        resp = await self._request(
            "execute_actions_email",
            "PUT",
            f"/users/{_q(user_id)}/execute-actions-email",
            params=params,
            json=actions,
        )
        if resp.status_code not in (200, 204):
            raise _unexpected("execute_actions_email", resp)

    async def set_user_state(
        self, user_id: str, *, enabled: bool, attributes: dict[str, str | None]
    ) -> KeycloakUser | None:
        """Set ``enabled`` and merge (value) / remove (None) ``attributes`` in ONE
        read-modify-write of the whole representation (every other field and
        attribute carried over), then READ IT BACK: the stored user is returned
        so the caller can verify the write persisted (a realm user profile that
        does not declare an attribute drops it silently). None when the user is
        gone (404 on the read)."""
        path = f"/users/{_q(user_id)}"
        read = await self._request("set_user_state", "GET", path)
        if read.status_code == 404:
            return None
        if read.status_code != 200:
            raise _unexpected("set_user_state(read)", read)
        rep = read.json()
        if not isinstance(rep, dict):
            raise KeycloakAdminError("set_user_state: not an object", status_code=200)
        merged: dict[str, object] = dict(rep.get("attributes") or {})
        for key, value in attributes.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = [value]
        write = await self._request(
            "set_user_state", "PUT", path, json={**rep, "enabled": enabled, "attributes": merged}
        )
        if write.status_code not in (200, 204):
            raise _unexpected("set_user_state", write)
        return await self.get_user(user_id)

    async def delete_user(self, user_id: str) -> bool:
        """Delete the user; False when it was already gone (404)."""
        resp = await self._request("delete_user", "DELETE", f"/users/{_q(user_id)}")
        if resp.status_code == 404:
            return False
        if resp.status_code not in (200, 204):
            raise _unexpected("delete_user", resp)
        return True

    # ─── realm roles ─────────────────────────────────────────────────────

    async def get_user_effective_realm_roles(self, user_id: str) -> list[str]:
        """The user's EFFECTIVE realm role names (composites expanded), sorted."""
        resp = await self._request(
            "get_user_effective_realm_roles",
            "GET",
            f"/users/{_q(user_id)}/role-mappings/realm/composite",
        )
        if resp.status_code != 200:
            raise _unexpected("get_user_effective_realm_roles", resp)
        roles = resp.json()
        return sorted(r["name"] for r in roles if isinstance(r, dict) and r.get("name"))


def get_keycloak_admin() -> KeycloakAdminClient:
    """The admin client for Studio's USER realm (``KEYCLOAK_USER_REALM``, default
    ``KEYCLOAK_REALM``), authorized as ``KEYCLOAK_USER_REALM_ADMIN_CLIENT_ID`` in that
    realm when configured, else as ``KEYCLOAK_CLIENT_ID`` (today's behaviour)."""
    s = get_settings().provisioning
    return KeycloakAdminClient(
        authenticator=get_user_realm_admin_authenticator(),
        keycloak_url=s.keycloak_url,
        realm=s.user_realm,
    )
