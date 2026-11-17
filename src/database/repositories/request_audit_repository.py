"""High-volume per-request security audit (middleware-wide audit).

Records one ``request_audit`` row (migration 0026) per mutation / auth-failure /
cross-tenant access and reads them back for the super-admin viewer. Unlike
``AuditRepository`` (deliberate admin actions, atomic with the acting route),
``record`` here is called by ``AuditMiddleware`` on a FRESH short session OUTSIDE
the route's transaction and must be fail-safe — the middleware swallows any error
so a failed audit insert never breaks the request. ``id`` + ``created_at`` are set
in Python so the row needs no flush/refresh round-trip.
"""

from __future__ import annotations

import datetime
from typing import Any
from uuid import UUID, uuid7

from sqlalchemy import ColumnElement, delete, func, select
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import RequestAudit


class RequestAuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record(
        self,
        *,
        actor_subject_id: UUID | None,
        claim_tenant_id: UUID | None,
        effective_tenant_id: UUID | None,
        cross_tenant: bool,
        event: str,
        method: str,
        path: str,
        resource: str | None,
        status_code: int,
        ip: str | None,
        duration_ms: float | None,
        credential_fingerprint: str | None,
        request_id: str | None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Append a request-audit row. No commit — the caller (the middleware's
        fresh session) commits. ``id``/``created_at`` set in Python."""
        self.session.add(
            RequestAudit(
                id=uuid7(),
                created_at=datetime.datetime.now(datetime.UTC),
                actor_subject_id=actor_subject_id,
                claim_tenant_id=claim_tenant_id,
                effective_tenant_id=effective_tenant_id,
                cross_tenant=cross_tenant,
                event=event,
                method=method,
                path=path,
                resource=resource,
                status_code=status_code,
                ip=ip,
                duration_ms=duration_ms,
                credential_fingerprint=credential_fingerprint,
                request_id=request_id,
                detail=detail or {},
            )
        )

    async def prune_older_than(
        self,
        *,
        cutoff: datetime.datetime,
        batch_size: int = 10_000,
    ) -> int:
        """Delete ``request_audit`` rows with ``created_at < cutoff``, in batches.

        Returns the total number of rows removed. The DELETE is scoped to a
        ``LIMIT``-ed subselect and looped until a batch removes nothing, so a
        first prune of a large backlog never takes one long row-lock — each
        ``batch_size`` chunk commits independently (the caller commits between
        batches). The ``(created_at, id)`` index supports the cutoff scan.

        No-op (returns 0) when there is nothing older than ``cutoff``.
        """
        total = 0
        while True:
            ids_subq = (
                select(RequestAudit.id)
                .where(RequestAudit.created_at < cutoff)
                .limit(batch_size)
                .scalar_subquery()
            )
            result = await self.session.execute(
                delete(RequestAudit).where(RequestAudit.id.in_(ids_subq))
            )
            deleted = int(getattr(result, "rowcount", 0) or 0)
            if deleted == 0:
                break
            await self.session.commit()
            total += deleted
        return total

    async def list_events(
        self,
        *,
        limit: int,
        offset: int,
        event: str | None = None,
        effective_tenant_id: UUID | None = None,
        actor_subject_id: UUID | None = None,
        method: str | None = None,
        cross_tenant: bool | None = None,
        since: datetime.datetime | None = None,
        until: datetime.datetime | None = None,
        min_status: int | None = None,
    ) -> tuple[list[Row[Any]], int]:
        """Newest-first page of request-audit rows + the filtered total.

        All filters are optional and AND-ed; ``total`` reflects the same filters.
        ``id`` is uuid7 (time-ordered) so ``created_at DESC, id DESC`` is a stable
        total order for rows sharing a ``created_at`` instant.
        """
        conditions: list[ColumnElement[bool]] = []
        if event is not None:
            conditions.append(RequestAudit.event == event)
        if effective_tenant_id is not None:
            conditions.append(RequestAudit.effective_tenant_id == effective_tenant_id)
        if actor_subject_id is not None:
            conditions.append(RequestAudit.actor_subject_id == actor_subject_id)
        if method is not None:
            conditions.append(RequestAudit.method == method)
        if cross_tenant is not None:
            conditions.append(RequestAudit.cross_tenant == cross_tenant)
        if since is not None:
            conditions.append(RequestAudit.created_at >= since)
        if until is not None:
            conditions.append(RequestAudit.created_at < until)
        if min_status is not None:
            conditions.append(RequestAudit.status_code >= min_status)

        rows = list(
            (
                await self.session.execute(
                    select(
                        RequestAudit.id,
                        RequestAudit.created_at,
                        RequestAudit.actor_subject_id,
                        RequestAudit.claim_tenant_id,
                        RequestAudit.effective_tenant_id,
                        RequestAudit.cross_tenant,
                        RequestAudit.event,
                        RequestAudit.method,
                        RequestAudit.path,
                        RequestAudit.resource,
                        RequestAudit.status_code,
                        RequestAudit.ip,
                        RequestAudit.duration_ms,
                        RequestAudit.credential_fingerprint,
                        RequestAudit.request_id,
                        RequestAudit.detail,
                    )
                    .where(*conditions)
                    .order_by(RequestAudit.created_at.desc(), RequestAudit.id.desc())
                    .limit(limit)
                    .offset(offset)
                )
            ).all()
        )
        total = (
            await self.session.execute(select(func.count(RequestAudit.id)).where(*conditions))
        ).scalar_one()
        return rows, total
