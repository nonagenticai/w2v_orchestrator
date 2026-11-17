"""The tenant's sealed licence token + its per-tenant licence-change lease.

Repository rules as elsewhere: methods only FLUSH (the caller commits), except the
lease, which must be visible to OTHER sessions the moment it is taken — the lease
methods therefore run on their own short transaction via the caller's session and
are documented as such at the call site.

The lease mirrors commonpractice ``withLicenceLease``: ONE conditional UPDATE
(``licence_op_until IS NULL OR < now()``), no connection, transaction or advisory
lock held across the licences HTTP calls. A crashed holder frees itself at
``licence_op_until``.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import case, func, literal, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import OPERATOR_HELD_TENANT_STATUSES, Tenant

#: How long a licence-change lease is held at most (twice the introspection timeout, plus slack).
LICENCE_LEASE_SECONDS = 60


@dataclass(frozen=True, slots=True)
class StoredLicence:
    tenant_id: UUID
    status: str
    licence_ref: UUID | None
    partner_licence_id: UUID | None
    ciphertext: bytes | None
    token_expires_at: datetime.datetime | None = None
    #: ``licence_token_set_at`` as read — with ``ciphertext`` and ``licence_ref``
    #: the snapshot :meth:`TenantLicenceRepository.record_sweep_verdict` CASes on.
    token_set_at: datetime.datetime | None = None

    def __repr__(self) -> str:  # never render the sealed token
        return (
            f"StoredLicence(tenant_id={self.tenant_id!s}, has_token={self.ciphertext is not None})"
        )


def _entitled_status() -> object:
    """``active`` unless an operator holds the tenant (suspended / deleted)."""
    return case(
        (Tenant.status.in_(OPERATOR_HELD_TENANT_STATUSES), Tenant.status),
        else_="active",
    )


class TenantLicenceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def stored(self, tenant_id: UUID) -> StoredLicence | None:
        row = (
            await self.session.execute(
                select(
                    Tenant.id,
                    Tenant.status,
                    Tenant.licence_ref,
                    Tenant.partner_licence_id,
                    Tenant.licence_token_ciphertext,
                    Tenant.licence_token_expires_at,
                    Tenant.licence_token_set_at,
                ).where(Tenant.id == tenant_id)
            )
        ).one_or_none()
        if row is None:
            return None
        return StoredLicence(*row)

    async def sweep_targets(self) -> list[StoredLicence]:
        """Every tenant the refresh sweep judges: one with a stored token or a
        ``licence_ref``, not held by an operator."""
        rows = (
            await self.session.execute(
                select(
                    Tenant.id,
                    Tenant.status,
                    Tenant.licence_ref,
                    Tenant.partner_licence_id,
                    Tenant.licence_token_ciphertext,
                    Tenant.licence_token_expires_at,
                    Tenant.licence_token_set_at,
                ).where(
                    or_(
                        Tenant.licence_ref.is_not(None),
                        Tenant.licence_token_ciphertext.is_not(None),
                    ),
                    Tenant.status.not_in(OPERATOR_HELD_TENANT_STATUSES),
                )
            )
        ).all()
        return [StoredLicence(*r) for r in rows]

    async def record_sweep_verdict(
        self, row: StoredLicence, *, entitled: bool, licence_ref: UUID | None
    ) -> bool:
        """Record a refresh-sweep verdict ONLY if the tenant's licence state is
        still the one the verdict was reached on (compare-and-set): same sealed
        token (NULL-safe), same ``licence_token_set_at``, same ``licence_ref``, and
        no unexpired licence-change lease held. A concurrent rotation / bind wins;
        the sweep's stale verdict is dropped (the next sweep re-judges).

        ``entitled`` moves ``active`` ↔ ``licence_lapsed`` (an operator hold is
        kept); ``licence_ref`` is written only when given. Flush only. False when
        the CAS lost (or the row is gone)."""
        new_status = "active" if entitled else "licence_lapsed"
        result = await self.session.execute(
            update(Tenant)
            .where(
                Tenant.id == row.tenant_id,
                Tenant.licence_token_ciphertext.is_not_distinct_from(row.ciphertext),
                Tenant.licence_token_set_at.is_not_distinct_from(row.token_set_at),
                Tenant.licence_ref.is_not_distinct_from(row.licence_ref),
                or_(Tenant.licence_op_until.is_(None), Tenant.licence_op_until < func.now()),
            )
            .values(
                status=case(
                    (Tenant.status.in_(OPERATOR_HELD_TENANT_STATUSES), Tenant.status),
                    else_=new_status,
                ),
                licence_ref=func.coalesce(
                    literal(licence_ref, type_=Tenant.licence_ref.type), Tenant.licence_ref
                ),
                licence_checked_at=func.now(),
                updated_at=func.now(),
            )
            .returning(Tenant.id)
        )
        return result.first() is not None

    async def record_fallback_ref(self, tenant_id: UUID, licence_ref: UUID) -> bool:
        """Record a TOKEN-LESS ``/licenses/active`` single-candidate ``licence_ref``
        (first-party create / replay) — only while NO token is stored and no
        licence-change lease is held. A stored token owns ``licence_ref``; only the
        token path (lease + CAS) may change it. The tenant is entitled (an operator
        hold is kept). Flush only. False when the row is token-owned / leased."""
        result = await self.session.execute(
            update(Tenant)
            .where(
                Tenant.id == tenant_id,
                Tenant.licence_token_ciphertext.is_(None),
                or_(Tenant.licence_op_until.is_(None), Tenant.licence_op_until < func.now()),
            )
            .values(
                licence_ref=licence_ref,
                licence_checked_at=func.now(),
                status=_entitled_status(),
                updated_at=func.now(),
            )
            .returning(Tenant.id)
        )
        return result.first() is not None

    async def expiring_token_count(self, within: datetime.timedelta) -> int:
        """Tenants whose stored token expires within ``within`` (incl. expired)."""
        return int(
            (
                await self.session.execute(
                    select(func.count()).where(
                        Tenant.licence_token_ciphertext.is_not(None),
                        Tenant.licence_token_expires_at < func.now() + within,
                        Tenant.status != "deleted",
                    )
                )
            ).scalar_one()
        )

    async def store_token(
        self,
        tenant_id: UUID,
        *,
        ciphertext: bytes,
        key_id: str,
        licence_ref: UUID,
        token_expires_at: datetime.datetime | None,
    ) -> bool:
        """Store a VERIFIED token and its licence as the tenant's ref; the tenant
        is entitled (an operator hold is kept). Unconditional — callers that race
        a rotation use :meth:`swap_token`. False when there is no row."""
        result = await self.session.execute(
            update(Tenant)
            .where(Tenant.id == tenant_id)
            .values(
                licence_token_ciphertext=ciphertext,
                licence_token_kid=key_id,
                licence_token_set_at=func.now(),
                licence_token_expires_at=token_expires_at,
                licence_ref=licence_ref,
                licence_checked_at=func.now(),
                status=_entitled_status(),
                updated_at=func.now(),
            )
            .returning(Tenant.licence_token_set_at)
        )
        return result.first() is not None

    async def swap_token(
        self,
        tenant_id: UUID,
        *,
        old_ciphertext: bytes | None,
        op_id: UUID,
        ciphertext: bytes,
        key_id: str,
        licence_ref: UUID,
        token_expires_at: datetime.datetime | None,
    ) -> datetime.datetime | None:
        """Compare-and-swap: replace the token ONLY if the row still holds
        ``old_ciphertext`` (NULL-safe) AND this lease. Returns the new
        ``licence_token_set_at``, or None when the swap lost."""
        result = await self.session.execute(
            update(Tenant)
            .where(
                Tenant.id == tenant_id,
                Tenant.licence_token_ciphertext.is_not_distinct_from(old_ciphertext),
                Tenant.licence_op_id == op_id,
            )
            .values(
                licence_token_ciphertext=ciphertext,
                licence_token_kid=key_id,
                licence_token_set_at=func.now(),
                licence_token_expires_at=token_expires_at,
                licence_ref=licence_ref,
                licence_checked_at=func.now(),
                status=_entitled_status(),
                updated_at=func.now(),
            )
            .returning(Tenant.licence_token_set_at)
        )
        return result.scalar_one_or_none()

    async def acquire_lease(self, tenant_id: UUID, op_id: UUID) -> bool:
        """Take the licence-change lease (flush only — the CALLER commits at once so
        other sessions see it). False when another op holds an unexpired lease."""
        result = await self.session.execute(
            update(Tenant)
            .where(
                Tenant.id == tenant_id,
                or_(Tenant.licence_op_until.is_(None), Tenant.licence_op_until < func.now()),
            )
            .values(
                licence_op_id=op_id,
                licence_op_until=func.now() + datetime.timedelta(seconds=LICENCE_LEASE_SECONDS),
            )
            .returning(Tenant.id)
        )
        return result.first() is not None

    async def release_lease(self, tenant_id: UUID, op_id: UUID) -> None:
        """Release only OUR lease (a takeover after expiry is never cleared)."""
        await self.session.execute(
            update(Tenant)
            .where(Tenant.id == tenant_id, Tenant.licence_op_id == op_id)
            .values(licence_op_id=None, licence_op_until=None)
        )
