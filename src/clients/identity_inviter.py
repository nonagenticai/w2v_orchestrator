"""Identity invite client: provisioning owns Keycloak user creation for the
ordinary tenant-member invite.

On this path w2v never creates, emails or disables Keycloak users itself. (The
ONE exception is the early-access approve, George 2026-10-20: it mints its own
credential-less, ``w2v_signup_id``-stamped user exactly like CommonPractice and
fr — ``src/services/signup/identity.py``.) A client holding
Keycloak user-create power can set ``tenant_id`` / credentials /
``emailVerified`` and so mint a working login in any tenant of every app that
reads the claim. Provisioning holds that power behind ONE fixed-shape call::

    POST {PROVISIONING_URL}/api/identity/invites
    {"email", "client_id", "redirect_uri", "actor"?, "context"?}
    → 202 {"accepted": true}

It finds the user by exact (normalized) email or creates it unverified with no
attributes/credentials/roles, and sends the Keycloak ``execute-actions-email``
the account needs (``UPDATE_PASSWORD`` / ``VERIFY_EMAIL``, or none). Idempotent.
The answer deliberately says NOTHING about the account (no id, no
created/verified/emailed flags): w2v cannot echo account state to a tenant admin
because it never learns it. Membership binds on the invitee's own login by
verified email (``POST /api/invites/.../accept``), never on a returned id.
``actor`` (the inviting subject) and ``context`` (``w2v:tenant:<uuid>``) are
audit-only.

Authorized by this service's own client-credentials token (the shared
:class:`~src.clients.service_auth.ServiceAuthenticator`; the client needs the
realm role ``identity-inviter`` and an allow-list entry in provisioning).

Every failure is an :class:`IdentityInviteError` carrying a stable w2v code —
the routes map it 1:1 onto their error envelope. No fallback to the Keycloak
admin API exists, on purpose.
"""

from __future__ import annotations

from uuid import UUID

import httpx

from src.clients.service_auth import ServiceAuthError, TokenProvider, get_service_authenticator
from src.config import get_settings

_TIMEOUT_SECONDS = 15.0
_PATH = "/api/identity/invites"

#: Provisioning refused this service (401/403). An operator fix, not an outage.
IDENTITY_PROVIDER_FORBIDDEN = "IDENTITY_PROVIDER_FORBIDDEN"
#: Keycloak (behind provisioning, 502) or provisioning itself is unreachable /
#: 5xx / answered outside the contract.
KEYCLOAK_UNAVAILABLE = "KEYCLOAK_UNAVAILABLE"
#: Provisioning's per-caller rate limit (429); ``retry_after`` says when.
IDENTITY_RATE_LIMITED = "IDENTITY_RATE_LIMITED"
#: Provisioning rejected the request shape (422): a bad email, or this
#: deployment's client_id / redirect_uri is not on its allow-list.
IDENTITY_INVITE_REJECTED = "IDENTITY_INVITE_REJECTED"
#: This deployment has no PROVISIONING_URL / redirect URI configured.
IDENTITY_INVITE_NOT_CONFIGURED = "IDENTITY_INVITE_NOT_CONFIGURED"

_HTTP_STATUS = {
    IDENTITY_PROVIDER_FORBIDDEN: 502,
    KEYCLOAK_UNAVAILABLE: 502,
    IDENTITY_RATE_LIMITED: 429,
    IDENTITY_INVITE_REJECTED: 502,
    IDENTITY_INVITE_NOT_CONFIGURED: 503,
}


class IdentityInviteError(RuntimeError):
    """A failed identity invite. ``code`` is the w2v error code, ``http_status``
    the status a route answers with, ``upstream_status`` / ``upstream_code``
    what provisioning said (None for transport/config failures), and
    ``retry_after`` the seconds from a 429's ``Retry-After``."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        upstream_status: int | None = None,
        upstream_code: str | None = None,
        retry_after: int | None = None,
        uncertain: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = _HTTP_STATUS[code]
        self.upstream_status = upstream_status
        self.upstream_code = upstream_code
        self.retry_after = retry_after
        #: The request may have reached provisioning (any transport failure after
        #: connecting — read/write timeout, dropped connection, protocol error —
        #: a 5xx, or a 2xx outside the contract): the email may or may not have
        #: gone out. A caller that must not send twice records the outcome as
        #: unknown instead of retrying it. Connect-phase failures are not.
        self.uncertain = uncertain


def _upstream_code(resp: httpx.Response) -> str | None:
    """The machine code in provisioning's error body, whatever its envelope:
    ``{"detail": "X"}``, ``{"detail": {"code"|"error": "X"}}``,
    ``{"error": {"code": "X"}}`` or ``{"code"|"error": "X"}``."""
    try:
        body = resp.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    for key in ("detail", "error", "code"):
        value = body.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            for inner in ("code", "error"):
                if isinstance(value.get(inner), str) and value[inner]:
                    return value[inner]
    return None


def _retry_after(resp: httpx.Response) -> int | None:
    raw = resp.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0, int(float(raw)))
    except ValueError:
        return None


#: Transport failures that happen BEFORE a byte of the request is sent: provisioning
#: cannot have acted, so a retry is safe. Every other transport failure (a read /
#: write timeout, a dropped connection, a protocol error mid-response) may come
#: after provisioning processed the request and is ``uncertain``.
_NOT_SENT = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ProxyError,
    httpx.UnsupportedProtocol,
)


def _error_for(resp: httpx.Response) -> IdentityInviteError:
    status = resp.status_code
    code = _upstream_code(resp)
    if status in (401, 403):
        return IdentityInviteError(
            IDENTITY_PROVIDER_FORBIDDEN,
            "Provisioning refused this service's identity invite (HTTP "
            f"{status}): its client lacks the identity-inviter role or allow-list entry.",
            upstream_status=status,
            upstream_code=code,
        )
    if status == 429:
        retry = _retry_after(resp)
        return IdentityInviteError(
            IDENTITY_RATE_LIMITED,
            "Provisioning rate-limited this service's identity invites.",
            upstream_status=status,
            upstream_code=code,
            retry_after=retry,
        )
    if status == 422:
        return IdentityInviteError(
            IDENTITY_INVITE_REJECTED,
            "Provisioning rejected the identity invite (bad email, or this "
            "deployment's client_id / redirect_uri is not allow-listed).",
            upstream_status=status,
            upstream_code=code,
        )
    # A 5xx came back AFTER provisioning received the request: it may have sent
    # the email before failing (e.g. a 502 from a timed-out Keycloak call).
    return IdentityInviteError(
        KEYCLOAK_UNAVAILABLE,
        f"Identity invite failed upstream (HTTP {status}).",
        upstream_status=status,
        upstream_code=code,
        uncertain=status >= 500,
    )


def _accepted(resp: httpx.Response) -> bool:
    try:
        body = resp.json()
    except ValueError:
        return False
    return isinstance(body, dict) and body.get("accepted") is True


class IdentityInviter:
    """``POST {base_url}/api/identity/invites`` with a client-credentials Bearer."""

    def __init__(
        self,
        *,
        authenticator: TokenProvider,
        base_url: str,
        client_id: str,
        default_redirect_uri: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._auth = authenticator
        self._base = base_url.rstrip("/")
        self._client_id = client_id
        self._default_redirect = default_redirect_uri
        self._client = client

    def _body(
        self,
        email: str,
        target: str,
        actor: UUID | None,
        context: str | None,
    ) -> dict[str, str]:
        body: dict[str, str] = {
            "email": email.strip(),
            "client_id": self._client_id,
            "redirect_uri": target,
        }
        optional = {
            "actor": str(actor) if actor is not None else None,
            "context": context,
        }
        body.update({k: v for k, v in optional.items() if v})
        return body

    async def invite(
        self,
        email: str,
        *,
        redirect_uri: str | None = None,
        actor: UUID | None = None,
        context: str | None = None,
    ) -> None:
        """Ask provisioning to find-or-create ``email`` and send whatever action
        email it needs. Returns nothing on acceptance (202): the outcome for the
        account is not w2v's to know. ``redirect_uri`` (default: the configured
        one) must sit under this caller's allow-listed prefix; ``actor`` /
        ``context`` are provisioning's audit fields. (The early-access approve
        does NOT come here: it mints its own Keycloak user —
        ``src/services/signup/identity.py``.)"""
        target = redirect_uri or self._default_redirect
        if not self._base or not target or not self._client_id:
            raise IdentityInviteError(
                IDENTITY_INVITE_NOT_CONFIGURED,
                "Identity invites are not configured on this deployment "
                "(PROVISIONING_URL / IDENTITY_INVITE_REDIRECT_URI / IDENTITY_INVITE_CLIENT_ID).",
            )
        body = self._body(email, target, actor, context)
        try:
            token = await self._auth.token()
            headers = {"Authorization": f"Bearer {token}"}
            url = f"{self._base}{_PATH}"
            if self._client is not None:
                resp = await self._client.post(url, json=body, headers=headers)
            else:
                async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                    resp = await client.post(url, json=body, headers=headers)
        except ServiceAuthError as exc:
            raise IdentityInviteError(
                KEYCLOAK_UNAVAILABLE, f"Could not mint this service's token: {exc}"
            ) from exc
        except (httpx.HTTPError, OSError) as exc:
            raise IdentityInviteError(
                KEYCLOAK_UNAVAILABLE,
                f"Could not reach provisioning: {type(exc).__name__}",
                uncertain=not isinstance(exc, _NOT_SENT),
            ) from exc
        if not 200 <= resp.status_code < 300:
            raise _error_for(resp)
        if not _accepted(resp):
            raise IdentityInviteError(
                KEYCLOAK_UNAVAILABLE,
                "Provisioning answered the identity invite outside the contract "
                '(expected {"accepted": true}).',
                upstream_status=resp.status_code,
                uncertain=True,  # a 2xx: provisioning processed something
            )


def tenant_context(tenant_id: UUID) -> str:
    """The ``context`` audit field for an invite into a w2v tenant."""
    return f"w2v:tenant:{tenant_id}"


def default_redirect_uri() -> str:
    """``IDENTITY_INVITE_REDIRECT_URI``, else the SPA root (``W2V_FRONTEND_URL/``)."""
    s = get_settings().provisioning
    if s.identity_invite_redirect_uri:
        return s.identity_invite_redirect_uri
    return f"{s.w2v_frontend_url.rstrip('/')}/" if s.w2v_frontend_url else ""


def get_identity_inviter() -> IdentityInviter:
    s = get_settings().provisioning
    return IdentityInviter(
        authenticator=get_service_authenticator(),
        base_url=s.provisioning_url,
        client_id=s.identity_invite_client_id,
        default_redirect_uri=default_redirect_uri(),
    )
