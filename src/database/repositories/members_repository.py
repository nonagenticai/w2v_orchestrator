"""Tenant members roster (Phase C, increment C-1).

Adopts fr's DB-backed roster model: ``record_seen`` upserts the caller into
``tenant_members`` on ``GET /api/auth/me`` (best-effort, once per session
bootstrap — NOT a per-request hot path), and ``list_members`` reads the roster
back for the super-admin/tenant-admin Members screen.

``record_seen`` does NOT commit — it shares the route's session (FastAPI dedups
``get_db_session``), like ``AuditRepository.record``, so the roster write commits
with the request. The Members screen is read-only this increment; role
management (grant/revoke) waits on the Keycloak-Admin client (increment C-2).

P2 tenancy (migration 0072) makes the roster the membership AUTHORITY: the
``role``/``status`` columns + the membership methods below (``add_member``,
``set_role``, ``remove_member``, ``list_active_memberships``, …). ``record_seen``
deliberately never touches ``role``/``status``: a login only refreshes the
observed snapshot, it can neither grant a role nor resurrect a removed member.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from sqlalchemy import ColumnElement, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import OPERATOR_HELD_TENANT_STATUSES, Tenant, TenantMember
from src.database.tenant_lock import lock_tenant_id

# Whitelist of sortable columns — the route validates ``sort`` via a Literal, but
# we map defensively (unknown key → last_seen_at) so the repo can never order by
# an attacker-controlled column.
MemberRole = Literal["member", "tenant_admin"]

_SORT_COLUMNS: dict[str, Any] = {
    "last_seen": TenantMember.last_seen_at,
    "first_seen": TenantMember.first_seen_at,
    "email": TenantMember.email,
    "display_name": TenantMember.display_name,
}


class MembersRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record_seen(
        self,
        *,
        subject_id: UUID,
        tenant_id: UUID,
        email: str | None,
        display_name: str | None,
        roles: list[str],
        create: bool = True,
    ) -> None:
        """Upsert the caller's membership row. Inserts on first sight (server
        defaults stamp ``first_seen_at``/``last_seen_at``); on conflict bumps
        ``last_seen_at`` + ``roles`` and fills email/display_name only when the
        new value is non-null (a license-path call may omit them — never clobber
        a previously-captured name with NULL). No commit (route session owns it).

        ``create=False`` (P2 WP3) only refreshes an EXISTING active row and never
        inserts. Callers pass it for a super_admin scope-switch (a visit is not
        a membership — the realm role is their authority there) and for every
        visit under ``W2V_TENANCY_SOURCE=membership`` (membership rows are
        created only by explicit grants: invite acceptance, partner, backfill).
        """
        if not create:
            await self.session.execute(
                update(TenantMember)
                .where(
                    TenantMember.subject_id == subject_id,
                    TenantMember.tenant_id == tenant_id,
                    TenantMember.status == "active",
                )
                .values(
                    last_seen_at=func.now(),
                    roles=roles,
                    email=func.coalesce(email, TenantMember.email),
                    display_name=func.coalesce(display_name, TenantMember.display_name),
                )
            )
            return
        # Existing row (any status — the upsert below never filtered on it):
        # refresh it and stop. The hot path stays one UPDATE, no lock.
        refreshed = (
            await self.session.execute(
                update(TenantMember)
                .where(
                    TenantMember.subject_id == subject_id,
                    TenantMember.tenant_id == tenant_id,
                )
                .values(
                    last_seen_at=func.now(),
                    roles=roles,
                    email=func.coalesce(email, TenantMember.email),
                    display_name=func.coalesce(display_name, TenantMember.display_name),
                )
                .returning(TenantMember.subject_id)
            )
        ).first()
        if refreshed is not None:
            return
        # About to INSERT a member — that occupies the tenant. Serialise with
        # partner tenant creation (same advisory lock) so the insert cannot slip
        # between its occupancy check and its tenant insert. The upsert stays: a
        # concurrent login for the same subject may have inserted meanwhile.
        await lock_tenant_id(self.session, tenant_id)
        stmt = pg_insert(TenantMember).values(
            subject_id=subject_id,
            tenant_id=tenant_id,
            email=email,
            display_name=display_name,
            roles=roles,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[TenantMember.subject_id, TenantMember.tenant_id],
            set_={
                "last_seen_at": func.now(),
                "roles": stmt.excluded.roles,
                "email": func.coalesce(stmt.excluded.email, TenantMember.email),
                "display_name": func.coalesce(
                    stmt.excluded.display_name, TenantMember.display_name
                ),
            },
        )
        await self.session.execute(stmt)

    async def get_member(self, subject_id: UUID, tenant_id: UUID) -> TenantMember | None:
        """The roster row for one identity in one tenant, or None.

        Used by member role-mgmt (Phase C-2) to confirm a target belongs to the
        acting tenant_admin's effective tenant before granting/revoking a role.
        """
        return (
            await self.session.execute(
                select(TenantMember).where(
                    TenantMember.subject_id == subject_id,
                    TenantMember.tenant_id == tenant_id,
                )
            )
        ).scalar_one_or_none()

    async def set_roles(self, subject_id: UUID, tenant_id: UUID, roles: list[str]) -> None:
        """Overwrite the roster row's ``roles`` snapshot (Phase C-2 role mgmt).

        Keeps the Members screen in sync with a just-applied Keycloak grant/revoke
        without waiting for the member's next ``/api/auth/me``. No commit — shares
        the route session (the audit row + this update commit atomically)."""
        await self.session.execute(
            pg_insert(TenantMember)
            .values(subject_id=subject_id, tenant_id=tenant_id, roles=roles)
            .on_conflict_do_update(
                index_elements=[TenantMember.subject_id, TenantMember.tenant_id],
                set_={"roles": roles},
            )
        )

    async def list_members(
        self,
        tenant_id: UUID,
        *,
        limit: int,
        offset: int,
        search: str | None = None,
        sort: str = "last_seen",
        order: str = "desc",
    ) -> tuple[list[Row[Any]], int]:
        """A page of the tenant's roster + total count, optionally searched/sorted.

        Filters (AND-ed): the tenant scope, plus an optional ``search`` that
        ILIKE-matches the term anywhere in ``email`` OR ``display_name``. The
        chosen ``sort`` column (whitelisted) drives the order, ``subject_id`` is
        always the final tiebreak so the order is stable (subject_id breaks ties
        for rows sharing a sort-key value); ``total`` reflects the same filters
        so pagination stays correct. Defaults reproduce the prior behaviour
        (``last_seen_at DESC, subject_id``).
        """
        conditions: list[ColumnElement[bool]] = [TenantMember.tenant_id == tenant_id]
        if search and search.strip():
            # Escape ILIKE wildcards so a literal % / _ in the term doesn't act as
            # a wildcard; match the term as a substring of either name field.
            term = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{term}%"
            conditions.append(
                or_(
                    TenantMember.email.ilike(pattern, escape="\\"),
                    TenantMember.display_name.ilike(pattern, escape="\\"),
                )
            )

        sort_col = _SORT_COLUMNS.get(sort, TenantMember.last_seen_at)
        ordered = sort_col.asc() if order == "asc" else sort_col.desc()
        # nulls_last keeps un-named members (NULL email/display_name) at the end
        # regardless of direction; a no-op for the NOT-NULL timestamp columns.
        rows = list(
            (
                await self.session.execute(
                    select(
                        TenantMember.subject_id,
                        TenantMember.email,
                        TenantMember.display_name,
                        TenantMember.roles,
                        TenantMember.role,
                        TenantMember.status,
                        TenantMember.first_seen_at,
                        TenantMember.last_seen_at,
                    )
                    .where(*conditions)
                    .order_by(ordered.nulls_last(), TenantMember.subject_id)
                    .limit(limit)
                    .offset(offset)
                )
            ).all()
        )
        total = (
            await self.session.execute(
                select(func.count()).select_from(TenantMember).where(*conditions)
            )
        ).scalar_one()
        return rows, total

    # ─── P2 membership authority (migration 0072) ────────────────────────
    # None of these commit — the caller's session owns the transaction.

    async def list_active_memberships(self, subject_id: UUID) -> list[TenantMember]:
        """Every ACTIVE membership of one identity, most recently used first.

        Served by the partial index ``idx_tenant_members_subject_active``. Does
        not check ``tenants.status`` — the resolver decides what a suspended or
        licence-lapsed tenant means for the caller.
        """
        return list(
            (
                await self.session.execute(
                    select(TenantMember)
                    .where(TenantMember.subject_id == subject_id, TenantMember.status == "active")
                    .order_by(TenantMember.last_seen_at.desc(), TenantMember.tenant_id)
                )
            )
            .scalars()
            .all()
        )

    async def list_resolvable_memberships(self, subject_id: UUID) -> list[Row[Any]]:
        """``(tenant_id, role, last_seen_at)`` of every membership the request
        path may resolve to: ACTIVE, in a tenant that is not ``suspended``.

        A membership whose tenant has NO ``tenants`` registry row is NOT
        resolvable: membership resolution skips ``require_known_tenant``, so the
        registry row is what proves w2v provisioned the tenant (0072 backfills
        it for every pre-P2 tenant). ``licence_lapsed`` stays resolvable — the licence gate
        answers that with a 402 the SPA can render, not a 403. Most recently
        used first, ``tenant_id`` as the stable tiebreak. One indexed query per
        request (``idx_tenant_members_subject_active``).
        """
        return list(
            (
                await self.session.execute(
                    select(TenantMember.tenant_id, TenantMember.role, TenantMember.last_seen_at)
                    .join(Tenant, Tenant.id == TenantMember.tenant_id)
                    .where(
                        TenantMember.subject_id == subject_id,
                        TenantMember.status == "active",
                        Tenant.status.not_in(OPERATOR_HELD_TENANT_STATUSES),
                    )
                    .order_by(TenantMember.last_seen_at.desc(), TenantMember.tenant_id)
                )
            ).all()
        )

    async def tenant_accepts_lazy_members(self, tenant_id: UUID) -> bool:
        """True when a ``tenants`` row exists with status ``active`` or
        ``licence_lapsed`` — the only tenants the shadow-mode lazy upsert may
        write a membership into. Never creates a tenant row."""
        status = (
            await self.session.execute(select(Tenant.status).where(Tenant.id == tenant_id))
        ).scalar_one_or_none()
        return status in ("active", "licence_lapsed")

    async def ensure_member(
        self,
        *,
        subject_id: UUID,
        tenant_id: UUID,
        role: MemberRole,
        email: str | None,
        display_name: str | None,
    ) -> bool:
        """Insert an active membership if the identity has NO row in the tenant.

        The shadow-mode lazy upsert (P2 WP3). ``ON CONFLICT DO NOTHING``: an
        existing row — including a REMOVED one — is never touched, so a lazy
        write can neither change a role nor resurrect a removed member. True
        when a row was inserted.
        """
        result = await self.session.execute(
            pg_insert(TenantMember)
            .values(
                subject_id=subject_id,
                tenant_id=tenant_id,
                role=role,
                status="active",
                email=email,
                display_name=display_name,
                roles=[],
            )
            .on_conflict_do_nothing(
                index_elements=[TenantMember.subject_id, TenantMember.tenant_id]
            )
            .returning(TenantMember.subject_id)
        )
        return result.first() is not None

    async def get_active_membership(self, subject_id: UUID, tenant_id: UUID) -> TenantMember | None:
        return (
            await self.session.execute(
                select(TenantMember).where(
                    TenantMember.subject_id == subject_id,
                    TenantMember.tenant_id == tenant_id,
                    TenantMember.status == "active",
                )
            )
        ).scalar_one_or_none()

    async def add_member(
        self,
        *,
        subject_id: UUID,
        tenant_id: UUID,
        role: MemberRole,
        invited_by: UUID | None = None,
        email: str | None = None,
        display_name: str | None = None,
    ) -> None:
        """Grant (or re-grant) a membership: insert, or reactivate a removed row.

        The explicit membership write (invite acceptance, partner tenant
        creation, backfill) — unlike ``record_seen`` it SETS ``role`` and
        ``status``. Re-adding a removed member clears ``removed_at`` (the
        ``ck_tenant_members_removed_at`` invariant) and keeps the original
        ``created_at``/``first_seen_at``. Email/display_name only fill in, never
        clobber with NULL.
        """
        stmt = pg_insert(TenantMember).values(
            subject_id=subject_id,
            tenant_id=tenant_id,
            role=role,
            status="active",
            invited_by=invited_by,
            email=email,
            display_name=display_name,
            roles=[],
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[TenantMember.subject_id, TenantMember.tenant_id],
            set_={
                "role": stmt.excluded.role,
                "status": "active",
                "removed_at": None,
                "invited_by": func.coalesce(stmt.excluded.invited_by, TenantMember.invited_by),
                "email": func.coalesce(stmt.excluded.email, TenantMember.email),
                "display_name": func.coalesce(
                    stmt.excluded.display_name, TenantMember.display_name
                ),
            },
        )
        await self.session.execute(stmt)

    async def set_role(self, subject_id: UUID, tenant_id: UUID, role: MemberRole) -> bool:
        """Change an ACTIVE member's tenant role. False when no active row matched
        (a removed member must be re-added, not silently re-roled)."""
        result = await self.session.execute(
            update(TenantMember)
            .where(
                TenantMember.subject_id == subject_id,
                TenantMember.tenant_id == tenant_id,
                TenantMember.status == "active",
            )
            .values(role=role)
            .returning(TenantMember.subject_id)
        )
        return result.first() is not None

    async def remove_member(self, subject_id: UUID, tenant_id: UUID) -> bool:
        """Soft-remove an ACTIVE membership (``status='removed'``, ``removed_at``
        stamped). False when there was no active row. The row is kept for audit
        and so a later re-invite can reactivate it."""
        result = await self.session.execute(
            update(TenantMember)
            .where(
                TenantMember.subject_id == subject_id,
                TenantMember.tenant_id == tenant_id,
                TenantMember.status == "active",
            )
            .values(status="removed", removed_at=func.now())
            .returning(TenantMember.subject_id)
        )
        return result.first() is not None

    async def count_active_admins(self, tenant_id: UUID) -> int:
        """Active tenant_admins of a tenant — the "never remove/demote the last
        admin" guard reads this inside the same transaction as the write."""
        return (
            await self.session.execute(
                select(func.count())
                .select_from(TenantMember)
                .where(
                    TenantMember.tenant_id == tenant_id,
                    TenantMember.status == "active",
                    TenantMember.role == "tenant_admin",
                )
            )
        ).scalar_one()
