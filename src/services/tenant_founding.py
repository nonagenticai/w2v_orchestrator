"""Founding a NEW tenant: the shared core of every tenant-creating onboarding path.

Two callers share these two functions, in this order, so they can never issue
different things:

* ``POST /api/admin/provision`` (super_admin) — ``found_tenant``, then its
  first-user invite, then ``license_new_tenant``;
* the early-access approve (``src/services/signup/core.py``) — ``found_tenant``,
  ``license_new_tenant``, then the requester's invite + one email.

:func:`found_tenant` registers the tenant in licences as w2v's own (licences
migration 050: licences mints only on a registered tenant its caller owns) and
creates the ``tenants`` row (``created_via='super_admin'``). Both are idempotent
on ``tenant_id``.

:func:`license_new_tenant` ensures the TENANT-level ``w2v`` bundle licences +
the LLM access posture (``provision_tenant_licences``), then mints the tenant a
DEDICATED token for its ``w2v`` licence and stores it sealed (entitlement is the
introspection of that token, ``src/services/tenant_licence.py``).

Nothing here commits: the caller's session owns the transaction.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import HTTPException, status

from src.api.routes.tenancy.errors import TenancyError
from src.api.routes.tenancy.partner import register_in_licences
from src.database.repositories.tenant_licence_repository import TenantLicenceRepository
from src.services.license_provisioning import (
    W2V_BUNDLE_KEY,
    W2V_LICENCE_APP_KEY,
    LlmAccessStep,
    NewTenantLlmAccess,
    provision_tenant_licences,
)
from src.services.tenant_licence import seal_licence_token
from src.services.tenant_provisioning import TenantCreation, create_tenant

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.clients.licenses_client import LicensesClient
    from src.services.providers.crypto import Keyring

_log = logging.getLogger("w2v_orchestrator.tenant_founding")


def uuid_or_none(raw: str | None) -> UUID | None:
    try:
        return UUID(raw) if raw else None
    except ValueError:
        _log.warning("licence id %r is not a UUID; tenant licence_ref not recorded", raw)
        return None


def instant_or_none(raw: str | None) -> datetime.datetime | None:
    if not raw:
        return None
    try:
        when = datetime.datetime.fromisoformat(raw)
    except ValueError:
        _log.warning("minted token expiry %r is not ISO 8601; stored as unknown", raw)
        return None
    return when if when.tzinfo is not None else when.replace(tzinfo=datetime.UTC)


async def found_tenant(
    session: AsyncSession,
    licenses: LicensesClient,
    *,
    tenant_id: UUID,
    name: str,
    actor_subject_id: UUID | None,
) -> TenantCreation:
    """Register ``tenant_id`` in licences under ``name`` (as w2v's own), then the
    ``tenants`` row (``created_via='super_admin'``). Idempotent; flush only."""
    await register_in_licences(licenses, tenant_id, name)
    return await create_tenant(
        session,
        tenant_id=tenant_id,
        name=name,
        created_via="super_admin",
        actor_subject_id=actor_subject_id,
    )


@dataclass(frozen=True, slots=True)
class NewTenantLicences:
    """``tokens`` (one per requested app, minted only when asked for — they are
    handed out ONCE) and ``license_ids`` for the requested apps; ``licence_ref``
    = the tenant's ``w2v`` licence whose dedicated token was stored sealed;
    ``llm_access`` = the posture step (a failure is RETURNED, never raised)."""

    license_ids: dict[str, str]
    llm_access: LlmAccessStep
    licence_ref: UUID | None
    tokens: dict[str, str] = field(default_factory=dict)


async def license_new_tenant(
    session: AsyncSession,
    licenses: LicensesClient,
    keyring: Keyring,
    *,
    tenant_id: UUID,
    app_keys: list[str],
    llm_access: NewTenantLlmAccess,
    mint_tokens: bool,
) -> NewTenantLicences:
    """Ensure the tenant's TENANT-LEVEL licences (held by ``svc:<tenant>:<app>``,
    never by a person) + its LLM posture, then seal and store a dedicated token
    for its ``w2v`` licence. Idempotent: a retry answers ``exists`` instead of
    minting a second licence (``create_token`` revokes nothing).

    A requested key licences refused → 409 ``LICENCE_REFUSED`` naming the reason;
    one not in w2v's bundle → 404. ``mint_tokens`` additionally mints one token
    per requested app for the caller to hand out (the super_admin provision's
    one-time response); the early-access approve hands nothing out."""
    provisioned = await provision_tenant_licences(
        licenses, tenant_id=tenant_id, llm_access=llm_access
    )
    ensured = provisioned.licences
    tokens: dict[str, str] = {}
    license_ids: dict[str, str] = {}
    for key in app_keys:
        license_id = ensured.license_ids.get(key)
        if license_id is None:
            reason = ensured.refused.get(key)
            if reason is None:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND,
                    f"licenses app {key!r} not found in the {W2V_BUNDLE_KEY!r} bundle",
                )
            raise TenancyError(
                409,
                "LICENCE_REFUSED",
                f"licences refused the tenant-level {key!r} licence: {reason}",
            )
        if mint_tokens:
            minted = await licenses.create_token(license_id, tenant_id=tenant_id)
            tokens[key] = minted.token
        license_ids[key] = license_id
    # The w2v licence (svc:<tenant>:w2v) entitles the tenant. Mint it a DEDICATED
    # token, seal and store it: entitlement is then the introspection of that
    # token, which an off-cluster deployment can do too. licence_ref = that
    # licence — the one every run's cost events are attributed to.
    licence_ref = uuid_or_none(license_ids.get(W2V_LICENCE_APP_KEY))
    if licence_ref is not None:
        minted = await licenses.create_token(str(licence_ref), tenant_id=tenant_id)
        sealed = seal_licence_token(keyring, tenant_id, minted.token)
        await TenantLicenceRepository(session).store_token(
            tenant_id,
            ciphertext=sealed.ciphertext,
            key_id=sealed.key_id,
            licence_ref=licence_ref,
            token_expires_at=instant_or_none(minted.expires_at),
        )
    return NewTenantLicences(
        license_ids=license_ids,
        llm_access=provisioned.llm_access,
        licence_ref=licence_ref,
        tokens=tokens,
    )
