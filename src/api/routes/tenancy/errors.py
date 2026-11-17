"""Tenancy error envelope (P2 WP4a).

w2v_common's HTTPException handler renders ``code`` as ``HTTP_<status>`` and
stringifies a dict ``detail``, so a route cannot hand the SPA a stable code
(``LAST_TENANT_ADMIN``, ``INVITE_EMAIL_MISMATCH``, …) through HTTPException.
Tenancy routes raise :class:`TenancyError` instead; its handler emits the same
canonical envelope ``{"error": {"code", "message", "status", "request_id",
"timestamp", "details"?}}`` with the real code.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid7

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from w2v_common.asgi.error_envelope import make_error_payload

from src.clients.identity_inviter import IdentityInviteError
from src.clients.licenses_client import TenantNotOnboardedError
from src.services.tenant_licence import LicencesTenantNotOnboardedError


class TenancyError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details
        #: Extra response headers (e.g. ``Retry-After`` on a 429).
        self.headers = headers


async def _handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, TenancyError)  # noqa: S101 — registered for TenancyError only
    request_id = (
        getattr(request.state, "request_id", None)
        or request.headers.get("x-request-id")
        or str(uuid7())
    )
    return JSONResponse(
        status_code=exc.status,
        content=make_error_payload(
            code=exc.code,
            message=exc.message,
            status=exc.status,
            request_id=request_id,
            details=exc.details,
        ),
        headers=exc.headers,
    )


#: licences (or its gateway) DEFINITELY refused this service's lookup — a
#: configuration fault, so 502 with NO Retry-After (unlike 503 LICENCES_UNAVAILABLE).
LICENCES_REFUSED = "LICENCES_REFUSED"


def licences_refused(status_code: int) -> TenancyError:
    """The 502 for a :class:`~src.services.licence_refresh.LicencesRefusedError`.
    Carries only the upstream status — never the upstream body."""
    return TenancyError(
        502,
        LICENCES_REFUSED,
        f"The licences service refused this service's lookup (HTTP {status_code}). "
        "This is a configuration fault of this w2v deployment, not an outage; "
        "retrying will not help — contact the operator.",
        details={"upstream_status": status_code},
    )


#: licences answered 403 ``tenant_not_accessible``: the tenant is owned by another
#: app that granted w2v no access — it is not onboarded to w2v. A definite answer
#: about the TENANT (403, no Retry-After): not a configuration fault of this
#: deployment (that is 502 ``LICENCES_REFUSED``), not an outage (503), and not
#: "buy a licence" (402 ``license_*``). One code on every surface that talks to
#: licences — the gate, the API-key lane, the partner and operator routes.
TENANT_NOT_ONBOARDED = "TENANT_NOT_ONBOARDED"


def tenant_not_onboarded() -> TenancyError:
    """The 403 for a tenant licences says is not onboarded to w2v."""
    return TenancyError(
        403,
        TENANT_NOT_ONBOARDED,
        "This tenant is not onboarded to Studio: the licences service holds no Studio "
        "grant for it. Its owning app must onboard it to Studio first.",
    )


async def _not_onboarded_handler(request: Request, exc: Exception) -> JSONResponse:
    """Safety net: a licences ``tenant_not_accessible`` that no call site mapped
    still answers the 403 ``TENANT_NOT_ONBOARDED`` — never a 500."""
    return await _handler(request, tenant_not_onboarded())


_IDENTITY_MESSAGES = {
    "IDENTITY_PROVIDER_FORBIDDEN": (
        "The identity provider refused to invite this account: this service lacks "
        "the identity-inviter permission in provisioning. An operator must grant it."
    ),
    "KEYCLOAK_UNAVAILABLE": "Could not reach the identity provider.",
    "IDENTITY_RATE_LIMITED": "Too many invites right now; retry shortly.",
    "IDENTITY_INVITE_REJECTED": (
        "The identity provider rejected this invite (the address or this "
        "deployment's invite settings). Contact the operator."
    ),
    "IDENTITY_INVITE_NOT_CONFIGURED": "Invites are not configured on this deployment.",
}


def identity_error(exc: IdentityInviteError) -> TenancyError:
    """The response for a failed identity invite. Provisioning's body is never
    echoed: only the w2v code, a fixed message and (on 429) ``Retry-After``."""
    headers = (
        {"Retry-After": str(exc.retry_after)}
        if exc.code == "IDENTITY_RATE_LIMITED" and exc.retry_after is not None
        else None
    )
    details = {"retry_after": exc.retry_after} if headers else None
    return TenancyError(
        exc.http_status,
        exc.code,
        _IDENTITY_MESSAGES.get(exc.code, "Identity invite failed."),
        details,
        headers=headers,
    )


def register_tenancy_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(TenancyError, _handler)
    app.add_exception_handler(TenantNotOnboardedError, _not_onboarded_handler)
    app.add_exception_handler(LicencesTenantNotOnboardedError, _not_onboarded_handler)
