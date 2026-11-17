"""Install a VERIFIED licence token on a tenant: lease → compare-and-swap.

The one write path for a token on an EXISTING tenant row — the rotation route
(``PUT /api/tenants/{id}/licence-token``), a partner create/replay that carries a
token (provisioning mints a fresh one on every attempt), and the super_admin bind.
CP ``changeTenantLicence`` minus mint/revoke (the partner owns those; licences has
no single-token revoke, so a replaced token is simply dropped):

1. same token as stored → no-op (``changed=False``), no lease taken;
2. take the per-tenant LEASE (one conditional UPDATE, committed at once so every
   other session sees it) — held → :class:`LicenceBusyError` (409 ``licence_busy``);
3. re-read the stored blob under the lease and CAS-swap it (row must still hold
   that blob AND our lease) — lost → :class:`LicenceChangedConcurrentlyError`
   (409 ``licence_changed_concurrently``);
4. the caller's ``on_swapped`` (its audit row) joins the same transaction, which
   commits before the lease is released.

The token itself must already have been introspection-verified for this tenant
(``verdict``): no licences call happens while the lease is held.
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import UUID, uuid7

from sqlalchemy.ext.asyncio import AsyncSession

from src.database.repositories.tenant_licence_repository import TenantLicenceRepository
from src.services.providers.crypto import Keyring
from src.services.tenant_licence import (
    LicenceTokenUnreadableError,
    TokenVerdict,
    open_licence_token,
    seal_licence_token,
)

_log = logging.getLogger("w2v_orchestrator.tenant_licence")


class LicenceBusyError(RuntimeError):
    """Another licence change holds this tenant's lease."""


class LicenceChangedConcurrentlyError(RuntimeError):
    """The stored token changed between read and swap."""


@dataclass(frozen=True, slots=True)
class Installed:
    changed: bool
    licence_ref: UUID | None
    previous_licence_ref: UUID | None
    set_at: datetime.datetime | None


def _same_token(keyring: Keyring, tenant_id: UUID, ciphertext: bytes | None, token: str) -> bool:
    if ciphertext is None:
        return False
    try:
        return open_licence_token(keyring, tenant_id, ciphertext) == token
    except LicenceTokenUnreadableError:
        return False  # an unreadable blob is replaced, never compared


async def install_licence_token(
    session: AsyncSession,
    keyring: Keyring,
    tenant_id: UUID,
    token: str,
    verdict: TokenVerdict,
    *,
    on_swapped: Callable[[Installed], Awaitable[None]] | None = None,
) -> Installed:
    """Install ``token`` (already verified: ``verdict.entitled``) on ``tenant_id``.
    COMMITS. Raises :class:`LicenceBusyError`, :class:`LicenceChangedConcurrentlyError`,
    or ``KeyringNotConfiguredError`` (no keyring → nothing can be sealed)."""
    if not verdict.entitled or verdict.licence_id is None:
        raise ValueError("install_licence_token needs an entitled verdict")
    repo = TenantLicenceRepository(session)
    current = await repo.stored(tenant_id)
    if current is None:
        await session.rollback()
        raise LicenceChangedConcurrentlyError("tenant row vanished")
    if _same_token(keyring, tenant_id, current.ciphertext, token):
        await session.rollback()
        return Installed(False, current.licence_ref, current.licence_ref, None)
    sealed = seal_licence_token(keyring, tenant_id, token)

    op = uuid7()
    if not await repo.acquire_lease(tenant_id, op):
        await session.rollback()
        raise LicenceBusyError(str(tenant_id))
    await session.commit()  # the lease must be visible to every other session now
    try:
        held = await repo.stored(tenant_id)
        old = held.ciphertext if held is not None else None
        set_at = await repo.swap_token(
            tenant_id,
            old_ciphertext=old,
            op_id=op,
            ciphertext=sealed.ciphertext,
            key_id=sealed.key_id,
            licence_ref=verdict.licence_id,
            token_expires_at=verdict.token_expires_at,
        )
        if set_at is None:
            await session.rollback()
            raise LicenceChangedConcurrentlyError(str(tenant_id))
        installed = Installed(
            True, verdict.licence_id, held.licence_ref if held is not None else None, set_at
        )
        if on_swapped is not None:
            await on_swapped(installed)
        await session.commit()
        return installed
    finally:
        try:
            await session.rollback()  # no-op after the commit; clears a failed txn
            await repo.release_lease(tenant_id, op)
            await session.commit()
        except Exception:  # noqa: BLE001 — the lease frees itself at licence_op_until
            _log.exception("tenant %s: licence lease release failed", tenant_id)
