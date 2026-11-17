"""Bind an EXISTING tenant to an allow-listed partner licence (P9-5).

The one implementation behind both callers:

* ``PUT /api/admin/tenants/{tenant_id}/partner`` (super_admin, ``src/api/routes/admin.py``);
* ``python -m scripts.bind_partner`` (in-pod operator CLI — no browser session).

Rules (the route's docstring is the contract), under the tenant lock:

* ``partner_licence_id`` must be in ``W2V_PARTNER_LICENSE_IDS`` → else 422
  ``NOT_A_PARTNER_LICENCE``;
* the tenant must exist (registry row, or data without one — then the row is
  created ``created_via='super_admin'``) → else 404 ``TENANT_NOT_FOUND``; a
  deleted tenant is 409 ``TENANT_DELETED``;
* ``token`` (optional) is INTROSPECTED before the lock (active ∧ this tenant ∧
  app w2v, else 422 ``LICENCE_TOKEN_MISMATCH``; 503 / 502 when licences cannot /
  will not answer), then stored sealed via lease + CAS;
* already bound to THIS partner → ``bound=False``; bound to ANOTHER partner →
  409 ``TENANT_BOUND_TO_OTHER_PARTNER``; else bind + audit ``tenant.partner_bind``.

``dry_run`` runs every check above (including the introspection, which is a
read) and then ROLLS BACK before anything is committed: the registry row an
occupied tenant would get is never kept, no audit row, no token stored.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.database.models import Tenant
from src.database.repositories.admin_audit_repository import AuditRepository
from src.database.repositories.admin_repository import AdminRepository
from src.database.tenant_lock import lock_tenant_id
from src.services.providers.crypto import Keyring
from src.services.tenant_licence import LicenceIntrospection
from src.services.tenant_licence_install import Installed
from src.services.tenant_provisioning import tenant_has_occupants


@dataclass(frozen=True, slots=True)
class PartnerBindOutcome:
    tenant_id: UUID
    partner_licence_id: UUID
    #: False on an idempotent re-bind to the same partner (nothing changed).
    bound: bool
    #: The tenant's billing licence after the call (dry run: what it WOULD be).
    licence_ref: UUID | None
    #: An occupied tenant without a registry row gets one (dry run: would get one).
    created_registry_row: bool
    #: ``Installed.changed`` when a token was given and applied; None otherwise.
    token_changed: bool | None = None


async def bind_tenant_to_partner(
    session: AsyncSession,
    *,
    audit: AuditRepository,
    licences: LicenceIntrospection,
    keyring: Keyring,
    tenant_id: UUID,
    partner_licence_id: UUID,
    token: str | None,
    actor: UUID | None,
    by: str = "super_admin",
    dry_run: bool = False,
) -> PartnerBindOutcome:
    """Bind (see module docstring). Commits on success unless ``dry_run``.

    ``actor`` is the acting subject (None for the operator CLI — there is no
    human subject); ``by`` names the lane in the audit ``detail``. The route's
    ``by='super_admin'`` keeps its ``tenant.partner_bind`` detail unchanged; any
    other lane adds ``"by"`` to it so an operator bind is distinguishable."""
    # Lazy: ``src.api.routes`` imports ``admin``, which imports this module — a
    # top-level import here would be circular for a caller (the CLI) that
    # imports this module first. The error envelope is the route's, on purpose.
    from src.api.routes.tenancy.errors import TenancyError
    from src.api.routes.tenancy.licence_token import (
        install_token,
        keyring_unavailable,
        verify_licence_token,
    )

    if partner_licence_id not in get_settings().w2v_partner_license_ids:
        raise TenancyError(
            422,
            "NOT_A_PARTNER_LICENCE",
            "partner_licence_id is not an allow-listed partner licence.",
        )
    verdict = None
    if token is not None:
        if not keyring.configured:
            raise keyring_unavailable()
        # Before the lock: no external call while the tenant lock is held.
        verdict = await verify_licence_token(licences, tenant_id, token)
    tenant, created_row = await _lock_bind_target(session, tenant_id)
    bound = tenant.partner_licence_id != partner_licence_id
    if bound and tenant.partner_licence_id is not None:
        raise TenancyError(
            409,
            "TENANT_BOUND_TO_OTHER_PARTNER",
            "This tenant is already managed by a different partner.",
        )
    if dry_run:
        # Every refusal above has run; nothing of this transaction is kept.
        current_ref = tenant.licence_ref
        await session.rollback()
        return PartnerBindOutcome(
            tenant_id=tenant_id,
            partner_licence_id=partner_licence_id,
            bound=bound,
            licence_ref=verdict.licence_id if verdict is not None else current_ref,
            created_registry_row=created_row,
        )
    if bound:
        tenant.partner_licence_id = partner_licence_id
        detail: dict[str, object] = {
            "partner_licence_id": str(partner_licence_id),
            "created_registry_row": created_row,
        }
        if by != "super_admin":
            detail["by"] = by
        if verdict is not None:
            detail["licence_ref"] = str(verdict.licence_id)
        await session.flush()
        await audit.record(
            actor_subject_id=actor,
            action="tenant.partner_bind",
            target_tenant_id=tenant_id,
            detail=detail,
        )
    licence_ref = tenant.licence_ref
    await session.commit()
    token_changed = None
    if token is not None and verdict is not None:
        installed = await install_token(
            session,
            keyring,
            tenant_id,
            token,
            verdict,
            on_swapped=_audit_licence_token_set(audit, tenant_id, actor, by),
        )
        licence_ref = installed.licence_ref
        token_changed = installed.changed
    return PartnerBindOutcome(
        tenant_id=tenant_id,
        partner_licence_id=partner_licence_id,
        bound=bound,
        licence_ref=licence_ref,
        created_registry_row=created_row,
        token_changed=token_changed,
    )


def _audit_licence_token_set(
    audit: AuditRepository, tenant_id: UUID, actor: UUID | None, by: str
) -> Callable[[Installed], Awaitable[None]]:
    async def record(installed: Installed) -> None:
        await audit.record(
            actor_subject_id=actor,
            action="tenant.licence_token_set",
            target_tenant_id=tenant_id,
            detail={
                "by": by,
                "licence_ref": str(installed.licence_ref) if installed.licence_ref else None,
                "previous_licence_ref": (
                    str(installed.previous_licence_ref) if installed.previous_licence_ref else None
                ),
            },
        )

    return record


async def _lock_bind_target(session: AsyncSession, tenant_id: UUID) -> tuple[Tenant, bool]:
    """The tenant row under the tenant lock + ``FOR UPDATE``; an occupied tenant
    without a registry row gets one (``created_via='super_admin'``). 404 when the
    tenant does not exist at all, 409 when it was deleted."""
    from src.api.routes.tenancy.errors import TenancyError

    await lock_tenant_id(session, tenant_id)
    tenant = (
        await session.execute(select(Tenant).where(Tenant.id == tenant_id).with_for_update())
    ).scalar_one_or_none()
    created_row = False
    if tenant is None:
        if not await tenant_has_occupants(session, tenant_id):
            raise TenancyError(404, "TENANT_NOT_FOUND", "No such tenant.")
        admin = AdminRepository(session)
        await admin.ensure_tenant(tenant_id, created_via="super_admin")
        tenant = await admin.get_tenant(tenant_id)
        if tenant is None:  # pragma: no cover - the row was just ensured
            raise RuntimeError(f"tenant {tenant_id} vanished after ensure_tenant")
        created_row = True
    if tenant.status == "deleted":
        raise TenancyError(409, "TENANT_DELETED", "This tenant was deleted.")
    return tenant, created_row
