"""Route-side helpers for a tenant licence token: verify it, install it.

Shared by ``POST /api/tenants`` (partner create / replay), ``PUT
/api/tenants/{id}/licence-token`` (rotation / handover) and the super_admin
``PUT /api/admin/tenants/{id}/partner`` (bind). Error envelope:

* 422 ``LICENCE_TOKEN_MISMATCH`` — the token is inactive, or of another tenant /
  app (``details.reason``: licences' inactive reason, ``tenant`` or ``app``);
* 503 ``LICENCES_UNAVAILABLE`` + Retry-After — licences could not answer;
* 502 ``LICENCES_REFUSED`` — licences or its gateway definitely refused (c344f9dc);
* 409 ``licence_busy`` — another licence change holds the tenant's lease;
* 409 ``licence_changed_concurrently`` — the stored token changed under us;
* 503 ``PROVIDER_KEYRING_UNAVAILABLE`` — no keyring, nothing can be sealed.

The token is never echoed — not in a message, not in ``details``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.tenancy.errors import TenancyError, licences_refused, tenant_not_onboarded
from src.services.providers.crypto import Keyring, KeyringNotConfiguredError
from src.services.tenant_licence import (
    LicenceIntrospection,
    LicencesRefusedError,
    LicencesTenantNotOnboardedError,
    LicencesUnavailableError,
    TokenVerdict,
    introspect_for_tenant,
)
from src.services.tenant_licence_install import (
    Installed,
    LicenceBusyError,
    LicenceChangedConcurrentlyError,
    install_licence_token,
)

LICENCE_TOKEN_MISMATCH = "LICENCE_TOKEN_MISMATCH"  # noqa: S105  # pragma: allowlist secret — an error code
LICENCE_BUSY = "licence_busy"
LICENCE_CHANGED_CONCURRENTLY = "licence_changed_concurrently"


def licences_unavailable(message: str) -> TenancyError:
    return TenancyError(503, "LICENCES_UNAVAILABLE", message, headers={"Retry-After": "30"})


def keyring_unavailable() -> TenancyError:
    return TenancyError(
        503,
        "PROVIDER_KEYRING_UNAVAILABLE",
        "The tenant licence cannot be stored on this deployment right now.",
    )


async def verify_licence_token(
    licences: LicenceIntrospection, tenant_id: UUID, token: str
) -> TokenVerdict:
    """The ENTITLED verdict of ``token`` for ``tenant_id`` — else a TenancyError."""
    try:
        verdict = await introspect_for_tenant(licences, token, tenant_id)
    except LicencesTenantNotOnboardedError as exc:
        raise tenant_not_onboarded() from exc
    except LicencesRefusedError as exc:
        raise licences_refused(exc.status_code) from exc
    except LicencesUnavailableError as exc:
        raise licences_unavailable(
            "Could not verify the licence token against the licences service; retry."
        ) from exc
    if not verdict.entitled:
        raise TenancyError(
            422,
            LICENCE_TOKEN_MISMATCH,
            "licence_token is not an active Studio licence token of this tenant.",
            details={"reason": verdict.reason or verdict.outcome},
        )
    return verdict


async def install_token(
    session: AsyncSession,
    keyring: Keyring,
    tenant_id: UUID,
    token: str,
    verdict: TokenVerdict,
    *,
    on_swapped: Callable[[Installed], Awaitable[None]] | None = None,
) -> Installed:
    """:func:`install_licence_token` (lease + CAS, COMMITS) with the 409/503 envelope."""
    try:
        return await install_licence_token(
            session, keyring, tenant_id, token, verdict, on_swapped=on_swapped
        )
    except LicenceBusyError as exc:
        raise TenancyError(
            409,
            LICENCE_BUSY,
            "Another licence change for this tenant is in progress. Retry shortly.",
            headers={"Retry-After": "5"},
        ) from exc
    except LicenceChangedConcurrentlyError as exc:
        raise TenancyError(
            409,
            LICENCE_CHANGED_CONCURRENTLY,
            "The tenant's licence changed while this change was being applied. Reload "
            "and try again.",
        ) from exc
    except KeyringNotConfiguredError as exc:
        await session.rollback()
        raise keyring_unavailable() from exc
