"""Retention-prune tests for ``RequestAuditRepository`` (migration 0026).

The middleware-wide audit persists a row per mutation / auth-failure /
cross-tenant access; ``prune_older_than`` is the disk-fill backstop swept by the
pod-lifetime loop in ``src/asgi.py``.
"""

import datetime
from uuid import uuid7

import pytest
from sqlalchemy import text

from src.database.models import RequestAudit
from src.database.repositories.request_audit_repository import RequestAuditRepository

pytestmark = pytest.mark.asyncio


def _row(
    created_at: datetime.datetime, *, event: str = "read", cross_tenant: bool = False
) -> RequestAudit:
    return RequestAudit(
        id=uuid7(),
        created_at=created_at,
        actor_subject_id=None,
        claim_tenant_id=None,
        effective_tenant_id=None,
        cross_tenant=cross_tenant,
        event=event,
        method="GET",
        path="/api/projects",
        resource="projects",
        status_code=200,
        ip=None,
        duration_ms=None,
        credential_fingerprint=None,
        request_id=None,
        detail={},
    )


async def _seed(session, count: int, created_at: datetime.datetime) -> None:
    for _ in range(count):
        session.add(_row(created_at))
    await session.commit()


async def test_prune_removes_only_rows_older_than_cutoff(db_session) -> None:
    now = datetime.datetime.now(datetime.UTC)
    old = now - datetime.timedelta(days=120)
    fresh = now - datetime.timedelta(days=10)
    await _seed(db_session, 3, old)
    await _seed(db_session, 2, fresh)

    repo = RequestAuditRepository(db_session)
    cutoff = now - datetime.timedelta(days=90)
    deleted = await repo.prune_older_than(cutoff=cutoff)

    assert deleted == 3
    rows, total = await repo.list_events(limit=100, offset=0)
    assert total == 2
    assert all(r.created_at >= fresh for r in rows)


async def test_prune_is_noop_when_nothing_expired(db_session) -> None:
    now = datetime.datetime.now(datetime.UTC)
    await _seed(db_session, 4, now - datetime.timedelta(days=5))

    repo = RequestAuditRepository(db_session)
    deleted = await repo.prune_older_than(cutoff=now - datetime.timedelta(days=90))

    assert deleted == 0
    _, total = await repo.list_events(limit=100, offset=0)
    assert total == 4


async def test_prune_batches_past_batch_size(db_session) -> None:
    now = datetime.datetime.now(datetime.UTC)
    old = now - datetime.timedelta(days=200)
    await _seed(db_session, 25, old)

    repo = RequestAuditRepository(db_session)
    # batch_size below the row count exercises the multi-batch loop.
    deleted = await repo.prune_older_than(cutoff=now - datetime.timedelta(days=90), batch_size=10)

    assert deleted == 25
    _, total = await repo.list_events(limit=100, offset=0)
    assert total == 0


async def test_list_events_filters_cross_tenant_only(db_session) -> None:
    now = datetime.datetime.now(datetime.UTC)
    db_session.add(_row(now, cross_tenant=True))
    db_session.add(_row(now, cross_tenant=False))
    await db_session.commit()

    repo = RequestAuditRepository(db_session)
    rows, total = await repo.list_events(limit=100, offset=0, cross_tenant=True)

    assert total == 1
    assert all(r.cross_tenant for r in rows)


async def test_cross_tenant_partial_index_exists(db_session) -> None:
    """The partial index serving the cross-tenant filter ships in the schema.

    Scoped to ``current_schema()`` because CI runs each job in its own schema in
    a shared postgres — an unscoped ``pg_indexes`` lookup would match siblings.
    """
    row = (
        await db_session.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE schemaname = current_schema() "
                "AND tablename = 'request_audit' "
                "AND indexname = 'idx_request_audit_cross_tenant'"
            )
        )
    ).first()
    assert row is not None, "idx_request_audit_cross_tenant missing"
    # Partial — the WHERE clause keeps it scoped to the cross-tenant subset.
    assert "cross_tenant" in row[0].lower()
    assert "where" in row[0].lower()
