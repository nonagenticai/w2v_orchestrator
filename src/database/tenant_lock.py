"""The per-tenant advisory lock every first-writer of a tenant id serialises on.

Two writers can turn an unused tenant id into an occupied one: a partner
creating the tenant (``POST /api/tenants``) and a claim-mode login inserting the
caller's first ``tenant_members`` row (``record_seen``). Without a shared lock a
member insert can land between the partner's occupancy check and its tenant
insert — the takeover the check exists to stop. A row lock cannot do it: there
may be no row yet. Transaction-scoped: released at COMMIT/ROLLBACK.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


def tenant_lock_key(tenant_id: UUID) -> str:
    """The text hashed into the bigint lock key — one derivation for all callers."""
    return f"w2v.tenant-create:{tenant_id}"


async def lock_tenant_id(session: AsyncSession, tenant_id: UUID) -> None:
    """Take the tenant's advisory lock for the rest of the transaction."""
    await session.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"),
        {"k": tenant_lock_key(tenant_id)},
    )
