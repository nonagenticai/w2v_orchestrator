"""Persistent admin audit log (Phase B, 4th increment).

Records deliberate admin actions to ``admin_audit_log`` (migration 0022) and
reads them back for the super-admin viewer. ``record`` does NOT commit — it is
called from inside an acting super-admin route whose ``get_db_session`` commits
on success, so the audit row commits atomically with the action it records (and
is rolled back with it on failure).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid7

from sqlalchemy import (
    ColumnElement,
    and_,
    exists,
    func,
    literal,
    not_,
    or_,
    select,
    tuple_,
)
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import AdminAuditLog, TenantMember


class AuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record(
        self,
        *,
        actor_subject_id: UUID | None,
        action: str,
        target_tenant_id: UUID | None = None,
        detail: dict[str, Any] | None = None,
    ) -> AdminAuditLog:
        """Append an audit row and return it. No commit — the route's session
        owns the txn (so the row commits atomically with the action). ``id`` and
        ``created_at`` are set in Python so the caller can serialise the row
        without a flush/refresh round-trip."""
        entry = AdminAuditLog(
            id=uuid7(),
            created_at=datetime.now(UTC),
            actor_subject_id=actor_subject_id,
            action=action,
            target_tenant_id=target_tenant_id,
            detail=detail or {},
        )
        self.session.add(entry)
        return entry

    async def list_events(
        self,
        *,
        limit: int,
        offset: int,
        action: str | None = None,
        tenant_id: UUID | None = None,
        actor_subject_id: UUID | None = None,
    ) -> tuple[list[Row[Any]], int]:
        """Newest-first page of audit rows + the total count, optionally filtered.

        Filters (all AND-ed, each optional): ``action`` exact match,
        ``tenant_id`` against ``target_tenant_id``, ``actor_subject_id`` against
        the actor. ``total`` reflects the same filters so pagination is correct.
        ``id`` is uuid7 (time-ordered) so ``created_at DESC, id DESC`` is a
        stable total order even for rows sharing a ``created_at`` instant.
        """
        conditions: list[ColumnElement[bool]] = []
        if action is not None:
            conditions.append(AdminAuditLog.action == action)
        if tenant_id is not None:
            conditions.append(AdminAuditLog.target_tenant_id == tenant_id)
        if actor_subject_id is not None:
            conditions.append(AdminAuditLog.actor_subject_id == actor_subject_id)

        rows = list(
            (
                await self.session.execute(
                    select(
                        AdminAuditLog.id,
                        AdminAuditLog.created_at,
                        AdminAuditLog.actor_subject_id,
                        AdminAuditLog.action,
                        AdminAuditLog.target_tenant_id,
                        AdminAuditLog.detail,
                    )
                    .where(*conditions)
                    .order_by(AdminAuditLog.created_at.desc(), AdminAuditLog.id.desc())
                    .limit(limit)
                    .offset(offset)
                )
            ).all()
        )
        total = (
            await self.session.execute(select(func.count(AdminAuditLog.id)).where(*conditions))
        ).scalar_one()
        return rows, total

    async def list_tenant_events(
        self,
        *,
        tenant_id: UUID,
        limit: int,
        before: tuple[datetime, UUID] | None = None,
        action: str | None = None,
        actor_subject_id: UUID | None = None,
        actor_kind: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[Row[Any]]:
        """Newest-first keyset page of the audit rows whose TARGET is ``tenant_id``.

        Each row carries ``is_member`` (the actor is attributable to the tenant, see
        ``member_actor_clause``) and ``member_email``. ``before`` is the
        ``(created_at, id)`` of the last row of the previous page. ``actor_subject_id``
        only ever matches member rows: filtering on a platform operator's id must not
        reveal which rows they wrote. Returns up to ``limit`` rows (the caller asks
        for one extra to know whether another page exists).
        """
        is_member = member_actor_clause(tenant_id)
        conditions: list[ColumnElement[bool]] = [AdminAuditLog.target_tenant_id == tenant_id]
        if before is not None:
            conditions.append(
                tuple_(AdminAuditLog.created_at, AdminAuditLog.id)
                < tuple_(literal(before[0]), literal(before[1]))
            )
        if action is not None:
            conditions.append(AdminAuditLog.action == action)
        if actor_subject_id is not None:
            conditions.append(AdminAuditLog.actor_subject_id == actor_subject_id)
            conditions.append(is_member)
        if actor_kind == "member":
            conditions.append(is_member)
        elif actor_kind == "platform_operator":
            conditions.append(not_(is_member))
        if since is not None:
            conditions.append(AdminAuditLog.created_at >= since)
        if until is not None:
            conditions.append(AdminAuditLog.created_at < until)
        member_email = (
            select(TenantMember.email)
            .where(
                TenantMember.subject_id == AdminAuditLog.actor_subject_id,
                TenantMember.tenant_id == tenant_id,
            )
            .scalar_subquery()
        )
        return list(
            (
                await self.session.execute(
                    select(
                        AdminAuditLog.id,
                        AdminAuditLog.created_at,
                        AdminAuditLog.actor_subject_id,
                        AdminAuditLog.action,
                        AdminAuditLog.target_tenant_id,
                        AdminAuditLog.detail,
                        is_member.label("is_member"),
                        member_email.label("member_email"),
                    )
                    .where(*conditions)
                    .order_by(AdminAuditLog.created_at.desc(), AdminAuditLog.id.desc())
                    .limit(limit)
                )
            ).all()
        )


#: Actions only a super_admin / the platform can perform on a tenant (SuperAdminDep
#: routes, partner provisioning). Their actor is never attributed to the tenant, even
#: when that super_admin also happens to be one of its members.
PLATFORM_ONLY_ACTIONS: tuple[str, ...] = (
    "scope.switch",
    "scope.reset",
    "tenant.create",
    "youtube.platform_channel.set",
    "youtube.tenant_policy.set",
    "offboarding.export",
    "offboarding.delete",
    "offboarding.reject",
    "tenant.partner_bind",
    "run.cancel",
    "signup.remove",
)


def member_actor_clause(tenant_id: UUID) -> ColumnElement[bool]:
    """True when an audit row's actor may be named to ``tenant_id``'s admins.

    The actor must be on the tenant's roster (``tenant_members``, ANY status — a
    removed member's past actions stay attributed). A super_admin scope-switched into
    the tenant is not (a visit never creates a membership, see
    ``MembersRepository.upsert_member``), nor is a partner or the platform. Even a
    member is not named for a platform-only action, nor for a ``tenant.update`` that
    did not come from the tenant-admin rename route (``detail.by = 'tenant_admin'``):
    the other writers are the super_admin PATCH and partner adoption.
    """
    on_roster = exists().where(
        TenantMember.subject_id == AdminAuditLog.actor_subject_id,
        TenantMember.tenant_id == tenant_id,
    )
    # COALESCE: a missing key must be FALSE, not NULL — a NULL would drop the row from
    # both the ``member`` and the ``platform_operator`` filter.
    tenant_admin_rename = func.coalesce(AdminAuditLog.detail["by"].astext, "") == "tenant_admin"
    return and_(
        AdminAuditLog.actor_subject_id.is_not(None),
        on_roster,
        AdminAuditLog.action.not_in(PLATFORM_ONLY_ACTIONS),
        or_(AdminAuditLog.action != "tenant.update", tenant_admin_rename),
    )
