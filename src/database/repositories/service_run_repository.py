"""Service lab run history (migration 0107).

One row per bench call: what was asked, which provider served it, where the media
went, and how long it took. The table is under RLS ``tenant_isolation``, and every
statement here still filters on ``tenant_id`` explicitly — the owner role (tests,
``TENANT_RLS_MODE=off``) bypasses the policy, so the WHERE clause is the guarantee
and the policy the backstop.

Nothing here commits: the caller's session owns the transaction (the route handler
commits before it responds — see the repo CLAUDE.md).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Final
from uuid import UUID, uuid7

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import ServiceRun

__all__ = [
    "DEFAULT_LIST_LIMIT",
    "DEFAULT_RETENTION_DAYS",
    "ServiceRunRepository",
    "delete_older_than",
    "get_run",
    "insert_run",
    "list_runs",
]

#: The row horizon. Matches the ``media-cache/`` bucket lifecycle rule
#: (``STORAGE_MEDIA_CACHE_TTL_DAYS``, 30 days) so a row never outlives its object.
DEFAULT_RETENTION_DAYS: Final = 30

#: The newest-first list the bench shows; a bound, never "everything".
DEFAULT_LIST_LIMIT: Final = 50


class ServiceRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def insert(
        self,
        *,
        tenant_id: UUID,
        slot: str,
        status: str,
        output_kind: str | None = None,
        run_id: UUID | None = None,
        subject_id: UUID | None = None,
        request: dict[str, object] | None = None,
        text_output: str | None = None,
        media_object_key: str | None = None,
        media_content_type: str | None = None,
        duration_ms: int | None = None,
        provider: dict[str, object] | None = None,
        error: dict[str, object] | None = None,
        created_at: datetime | None = None,
    ) -> ServiceRun:
        """Add one history row and FLUSH it (the caller commits).

        ``run_id`` may be pre-allocated by the caller so the media object key and the
        row share one id; omitted, a fresh ``uuid7`` is minted here.
        """
        row = ServiceRun(
            id=run_id or uuid7(),
            tenant_id=tenant_id,
            subject_id=subject_id,
            slot=slot,
            status=status,
            output_kind=output_kind,
            request=dict(request or {}),
            text_output=text_output,
            media_object_key=media_object_key,
            media_content_type=media_content_type,
            duration_ms=duration_ms,
            provider=dict(provider or {}),
            error=error,
        )
        if created_at is not None:
            row.created_at = created_at
        self.session.add(row)
        await self.session.flush()
        return row

    async def list_by_slot(
        self, tenant_id: UUID, slot: str, *, limit: int = DEFAULT_LIST_LIMIT
    ) -> Sequence[ServiceRun]:
        """This tenant's runs for ``slot``, newest first (the bench's list query)."""
        rows = await self.session.execute(
            select(ServiceRun)
            .where(ServiceRun.tenant_id == tenant_id, ServiceRun.slot == slot)
            .order_by(ServiceRun.created_at.desc(), ServiceRun.id.desc())
            .limit(limit)
        )
        return rows.scalars().all()

    async def get(self, tenant_id: UUID, run_id: UUID) -> ServiceRun | None:
        """One row, scoped to its tenant (another tenant's id reads as absent)."""
        return (
            await self.session.execute(
                select(ServiceRun).where(ServiceRun.id == run_id, ServiceRun.tenant_id == tenant_id)
            )
        ).scalar_one_or_none()

    async def delete_older_than(self, tenant_id: UUID, days: int = DEFAULT_RETENTION_DAYS) -> int:
        """Delete THIS tenant's rows older than ``days``; returns how many went.

        The DB clock decides the cutoff (``now()``), not the pod's: retention is a
        database-side horizon, and every pod agrees on it. Under RLS this can only
        ever reach the bound tenant's rows; the caller is expected to have bound one.
        """
        if days < 1:
            raise ValueError(f"retention must be at least 1 day, got {days}")
        result = await self.session.execute(
            delete(ServiceRun).where(
                ServiceRun.tenant_id == tenant_id,
                ServiceRun.created_at < func.now() - func.make_interval(0, 0, 0, days),
            )
        )
        # ``getattr`` because ``Result`` only exposes ``rowcount`` on the CursorResult
        # a DML statement returns — the same spelling the other repositories use.
        return int(getattr(result, "rowcount", 0) or 0)


# ── module-level entry points ────────────────────────────────────────────────
# The route layer (`api/routes/service_lab.py`) calls these four by name; the
# class above is where the statements live, so there is exactly ONE copy of each
# query in the service. Keep the two in step.


async def insert_run(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    slot: str,
    status: str,
    output_kind: str | None = None,
    run_id: UUID | None = None,
    subject_id: UUID | None = None,
    request: dict[str, object] | None = None,
    text_output: str | None = None,
    media_object_key: str | None = None,
    media_content_type: str | None = None,
    duration_ms: int | None = None,
    provider: dict[str, object] | None = None,
    error: dict[str, object] | None = None,
    created_at: datetime | None = None,
) -> ServiceRun:
    """One history row, flushed (the caller commits). See ``ServiceRunRepository.insert``."""
    return await ServiceRunRepository(session).insert(
        tenant_id=tenant_id,
        slot=slot,
        status=status,
        output_kind=output_kind,
        run_id=run_id,
        subject_id=subject_id,
        request=request,
        text_output=text_output,
        media_object_key=media_object_key,
        media_content_type=media_content_type,
        duration_ms=duration_ms,
        provider=provider,
        error=error,
        created_at=created_at,
    )


async def list_runs(
    session: AsyncSession,
    tenant_id: UUID,
    slot: str,
    limit: int = DEFAULT_LIST_LIMIT,
) -> Sequence[ServiceRun]:
    """This tenant's runs for ``slot``, newest first (the history list)."""
    return await ServiceRunRepository(session).list_by_slot(tenant_id, slot, limit=limit)


async def get_run(
    session: AsyncSession, tenant_id: UUID, slot: str, run_id: UUID
) -> ServiceRun | None:
    """One row for the tenant AND that slot (a mismatched slot reads as absent)."""
    row = await ServiceRunRepository(session).get(tenant_id, run_id)
    return row if row is not None and row.slot == slot else None


async def delete_older_than(
    session: AsyncSession, tenant_id: UUID, days: int = DEFAULT_RETENTION_DAYS
) -> int:
    """Drop THIS tenant's rows past the retention horizon; returns how many went."""
    return await ServiceRunRepository(session).delete_older_than(tenant_id, days)
