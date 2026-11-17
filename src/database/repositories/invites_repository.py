"""Tenant invites (P2 tenancy, migration 0073).

A tenant_admin invites an email with a role; the invitee accepts with the one-time
token (or, WP4a, by logging in with a VERIFIED email that matches a pending
invite). Only ``sha256(token)`` is stored. Lookups by token hash / email are
CROSS-TENANT by design (the table has no RLS) — every tenant-scoped mutation
here takes the acting ``tenant_id`` and filters on it, so a tenant_admin can
never revoke or resend another tenant's invite by id.

Nothing here commits: the caller's session owns the transaction.
"""

from __future__ import annotations

import datetime
import hashlib
import secrets
from typing import Literal
from uuid import UUID, uuid7

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import TenantInvite

InviteRole = Literal["member", "tenant_admin"]
InviteStatus = Literal["pending", "accepted", "revoked", "expired"]

#: 32 random bytes → 43 urlsafe chars. The raw token travels only in the invite
#: link; the DB keeps its SHA-256.
INVITE_TOKEN_BYTES = 32


def new_invite_token() -> tuple[str, bytes]:
    """A fresh one-time invite token and its SHA-256 (what gets stored)."""
    token = secrets.token_urlsafe(INVITE_TOKEN_BYTES)
    return token, hash_invite_token(token)


def hash_invite_token(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


class InvitesRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        tenant_id: UUID,
        email: str,
        role: InviteRole,
        token_hash: bytes,
        expires_at: datetime.datetime,
        invited_by: UUID | None,
        kc_user_created: bool = False,
    ) -> TenantInvite:
        """Insert a PENDING invite. A second pending invite for the same
        ``(tenant_id, lower(email))`` violates ``uq_tenant_invites_pending_email``
        (IntegrityError) — the route points the caller at the existing one. ``email``
        is stored stripped as given; matching is case-insensitive via
        ``lower(email)``."""
        invite = TenantInvite(
            id=uuid7(),
            tenant_id=tenant_id,
            email=email.strip(),
            role=role,
            token_hash=token_hash,
            expires_at=expires_at,
            status="pending",
            invited_by=invited_by,
            kc_user_created=kc_user_created,
        )
        self.session.add(invite)
        await self.session.flush()
        return invite

    async def get(self, invite_id: UUID, tenant_id: UUID) -> TenantInvite | None:
        """One invite of the acting tenant (None for another tenant's id)."""
        return (
            await self.session.execute(
                select(TenantInvite).where(
                    TenantInvite.id == invite_id, TenantInvite.tenant_id == tenant_id
                )
            )
        ).scalar_one_or_none()

    async def get_pending_by_token_hash(
        self, token_hash: bytes, *, now: datetime.datetime | None = None
    ) -> TenantInvite | None:
        """The PENDING, unexpired invite this token opens, else None. Expired and
        settled invites are indistinguishable from unknown ones to the caller."""
        return (
            await self.session.execute(
                select(TenantInvite).where(
                    TenantInvite.token_hash == token_hash,
                    TenantInvite.status == "pending",
                    TenantInvite.expires_at > (now or func.now()),
                )
            )
        ).scalar_one_or_none()

    async def get_pending_for_tenant_email(
        self, tenant_id: UUID, email: str
    ) -> TenantInvite | None:
        """The (at most one) pending invite for this email in this tenant."""
        return (
            await self.session.execute(
                select(TenantInvite).where(
                    TenantInvite.tenant_id == tenant_id,
                    func.lower(TenantInvite.email) == email.strip().lower(),
                    TenantInvite.status == "pending",
                )
            )
        ).scalar_one_or_none()

    async def list_pending_for_email(
        self, email: str, *, now: datetime.datetime | None = None, lock: bool = False
    ) -> list[TenantInvite]:
        """Every pending, unexpired invite for an email ACROSS tenants (the
        ``/api/auth/me`` ``pending_invites`` listing). Served by
        ``idx_tenant_invites_pending_email``.

        ``lock=True`` takes ``FOR UPDATE`` on the rows (in ``created_at``
        order, so two callers lock in the same order): a concurrent revoke /
        resend / token-accept either finishes first — and Postgres re-checks
        ``status = 'pending'`` on the new row version, dropping it — or waits
        for this transaction. Held until the caller commits."""
        stmt = (
            select(TenantInvite)
            .where(
                func.lower(TenantInvite.email) == email.strip().lower(),
                TenantInvite.status == "pending",
                TenantInvite.expires_at > (now or func.now()),
            )
            .order_by(TenantInvite.created_at, TenantInvite.id)
        )
        if lock:
            stmt = stmt.with_for_update()
        return list((await self.session.execute(stmt)).scalars().all())

    async def list_for_tenant(
        self, tenant_id: UUID, *, status: InviteStatus | None = "pending"
    ) -> list[TenantInvite]:
        """The tenant's invites, newest first; ``status=None`` lists all."""
        stmt = select(TenantInvite).where(TenantInvite.tenant_id == tenant_id)
        if status is not None:
            stmt = stmt.where(TenantInvite.status == status)
        stmt = stmt.order_by(TenantInvite.created_at.desc(), TenantInvite.id.desc())
        return list((await self.session.execute(stmt)).scalars().all())

    async def mark_accepted(self, invite_id: UUID, subject_id: UUID) -> bool:
        """pending → accepted, binding the accepting subject. False when the
        invite was no longer pending (a concurrent accept/revoke won)."""
        result = await self.session.execute(
            update(TenantInvite)
            .where(TenantInvite.id == invite_id, TenantInvite.status == "pending")
            .values(status="accepted", accepted_subject_id=subject_id, accepted_at=func.now())
            .returning(TenantInvite.id)
        )
        return result.first() is not None

    async def revoke(self, invite_id: UUID, tenant_id: UUID) -> bool:
        """pending → revoked, scoped to the acting tenant."""
        result = await self.session.execute(
            update(TenantInvite)
            .where(
                TenantInvite.id == invite_id,
                TenantInvite.tenant_id == tenant_id,
                TenantInvite.status == "pending",
            )
            .values(status="revoked")
            .returning(TenantInvite.id)
        )
        return result.first() is not None

    async def resend(
        self,
        invite_id: UUID,
        tenant_id: UUID,
        *,
        token_hash: bytes,
        expires_at: datetime.datetime,
    ) -> bool:
        """Rotate a pending invite's token + expiry (the old link stops working)."""
        result = await self.session.execute(
            update(TenantInvite)
            .where(
                TenantInvite.id == invite_id,
                TenantInvite.tenant_id == tenant_id,
                TenantInvite.status == "pending",
            )
            .values(token_hash=token_hash, expires_at=expires_at, resent_at=func.now())
            .returning(TenantInvite.id)
        )
        return result.first() is not None

    async def expire_stale(self, *, now: datetime.datetime | None = None) -> int:
        """Flip every pending invite past ``expires_at`` to ``expired``; returns
        how many. Housekeeping only — reads already ignore expired rows."""
        result = await self.session.execute(
            update(TenantInvite)
            .where(
                TenantInvite.status == "pending",
                TenantInvite.expires_at <= (now or func.now()),
            )
            .values(status="expired")
            .returning(TenantInvite.id)
        )
        return len(result.all())
