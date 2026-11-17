"""P3 — superseded provider-secret GC against REAL postgres.

The GC is one SECURITY DEFINER statement (``purge_unpinned_provider_secrets``,
migration 0079) so the pin scan and the purge cannot drift apart and it works
under the RLS-restricted runtime role. Each test seeds runs whose
``config_snapshot.provider_slots`` pin specific versions and asserts which
ciphertexts survive.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid7

import pytest
from sqlalchemy import select, text, update
from sqlalchemy.exc import DBAPIError, ProgrammingError

from src.database.models import Deliverable, Project, Run, TenantConnectionVersion
from src.database.repositories.admin_repository import AdminRepository
from src.database.repositories.connections_repository import ConnectionsRepository
from src.services.providers.crypto import Keyring
from src.services.providers.gc import DEFAULT_GC_GRACE, purge_unpinned_secrets
from src.services.providers.settings import ProviderPolicySettings
from tests.conftest import require_createrole

pytestmark = pytest.mark.asyncio

LONG_AGO = datetime(2026, 1, 1, tzinfo=UTC)


async def _connection_with_versions(session, n: int) -> tuple[UUID, UUID]:
    """A tenant + an llm connection with versions 1..n (1..n-1 superseded long ago)."""
    tenant = uuid7()
    assert await AdminRepository(session).ensure_tenant(tenant, created_via="super_admin")
    ring = Keyring([("k1", os.urandom(32))])
    repo = ConnectionsRepository(session)
    conn_id: UUID | None = None
    for i in range(n):
        conn, _ = await repo.create_version(
            tenant_id=tenant, slot="llm", kind="anthropic", secret=f"sk-{i}", keyring=ring
        )
        conn_id = conn.id
    assert conn_id is not None
    await session.execute(
        update(TenantConnectionVersion)
        .where(
            TenantConnectionVersion.connection_id == conn_id,
            TenantConnectionVersion.superseded_at.is_not(None),
        )
        .values(superseded_at=LONG_AGO)
    )
    await session.commit()
    return tenant, conn_id


def _pin(conn_id: UUID, version: int) -> dict[str, Any]:
    return {
        "provider_slots": {
            "llm": {
                "source": "tenant",
                "kind": "anthropic",
                "connection_id": str(conn_id),
                "version": version,
            },
            "tts": {"source": "deployment", "kind": "platform", "connection_id": None},
        }
    }


async def _run(
    session,
    tenant: UUID,
    status: str,
    snapshot: dict[str, Any],
    *,
    completed_at: datetime | None = None,
) -> None:
    """Seed a run. A terminal run defaults to having completed LONG_AGO (outside the
    function's 1h recently-terminal grace)."""
    if completed_at is None and status in ("completed", "failed", "cancelled"):
        completed_at = LONG_AGO
    project = Project(
        id=uuid7(),
        tenant_id=tenant,
        owner_subject_id=uuid7(),
        visibility="private",
        type="promo",
        title="p",
    )
    deliverable = Deliverable(id=uuid7(), project_id=project.id, name="d", format="16x9")
    run = Run(
        id=uuid7(),
        deliverable_id=deliverable.id,
        blueprint_id="promo_v1",
        status=status,
        config_snapshot=snapshot,
        completed_at=completed_at,
    )
    session.add_all([project, deliverable, run])
    await session.commit()


async def _has_secret(session, conn_id: UUID) -> dict[int, bool]:
    rows = (
        await session.execute(
            select(TenantConnectionVersion.version, TenantConnectionVersion.secret_ciphertext)
            .where(TenantConnectionVersion.connection_id == conn_id)
            .execution_options(populate_existing=True)
        )
    ).all()
    return {v: ct is not None for v, ct in rows}


async def test_gc_keeps_versions_pinned_by_running_paused_pending_runs(db_session) -> None:
    tenant, conn = await _connection_with_versions(db_session, 6)
    await _run(db_session, tenant, "running", _pin(conn, 1))
    await _run(db_session, tenant, "paused", _pin(conn, 2))
    await _run(db_session, tenant, "pending", _pin(conn, 3))
    await _run(db_session, tenant, "completed", _pin(conn, 4))  # terminal long ago → no protection
    await _run(db_session, tenant, "running", {"type": "promo"})  # legacy run pins nothing

    purged = await purge_unpinned_secrets(db_session)
    await db_session.commit()

    assert purged == 2
    assert await _has_secret(db_session, conn) == {
        1: True,  # running
        2: True,  # paused — a resume next week must find its credential
        3: True,  # pending
        4: False,  # only a terminal run pinned it
        5: False,  # superseded, unpinned
        6: True,  # current version is never superseded
    }
    purged_row = (
        await db_session.execute(
            select(TenantConnectionVersion).where(
                TenantConnectionVersion.connection_id == conn,
                TenantConnectionVersion.version == 5,
            )
        )
    ).scalar_one()
    assert purged_row.purged_at is not None and purged_row.key_id is None
    assert await purge_unpinned_secrets(db_session) == 0, "idempotent: nothing left"


async def test_gc_grace_window_keeps_just_superseded_versions(db_session) -> None:
    _tenant, conn = await _connection_with_versions(db_session, 2)
    now = datetime.now(UTC)
    await db_session.execute(
        update(TenantConnectionVersion)
        .where(TenantConnectionVersion.connection_id == conn, TenantConnectionVersion.version == 1)
        .values(superseded_at=now - timedelta(minutes=5))
    )
    await db_session.commit()

    assert await purge_unpinned_secrets(db_session, grace=timedelta(hours=1), now=now) == 0
    assert (await _has_secret(db_session, conn))[1] is True
    # A shorter caller grace cannot go below the server-side 1h floor.
    assert await purge_unpinned_secrets(db_session, grace=timedelta(minutes=1), now=now) == 0
    assert (await _has_secret(db_session, conn))[1] is True
    await db_session.execute(
        update(TenantConnectionVersion)
        .where(TenantConnectionVersion.connection_id == conn, TenantConnectionVersion.version == 1)
        .values(superseded_at=now - timedelta(hours=2))
    )
    await db_session.commit()
    assert await purge_unpinned_secrets(db_session, grace=timedelta(hours=1), now=now) == 1
    assert (await _has_secret(db_session, conn))[1] is False


async def _call_gc(session, cutoff: Any, limit: int) -> int:
    """Call the SQL function directly — bypassing the Python wrapper's grace math."""
    result = await session.execute(
        text("SELECT purge_unpinned_provider_secrets(CAST(:cutoff AS timestamptz), :limit)"),
        {"cutoff": cutoff, "limit": limit},
    )
    return int(result.scalar_one() or 0)


async def test_gc_infinity_cutoff_still_respects_the_one_hour_floor(db_session) -> None:
    _tenant, conn = await _connection_with_versions(db_session, 3)
    now = datetime.now(UTC)
    await db_session.execute(
        update(TenantConnectionVersion)
        .where(TenantConnectionVersion.connection_id == conn, TenantConnectionVersion.version == 2)
        .values(superseded_at=now - timedelta(minutes=5))
    )
    await db_session.commit()

    assert await _call_gc(db_session, "infinity", 100) == 1  # only v1 (superseded long ago)
    await db_session.commit()
    assert await _has_secret(db_session, conn) == {1: False, 2: True, 3: True}


@pytest.mark.parametrize(("limit", "expected"), [(-5, 0)])
async def test_gc_limit_is_clamped_server_side(db_session, limit: int, expected: int) -> None:
    _tenant, conn = await _connection_with_versions(db_session, 4)
    assert await _call_gc(db_session, "infinity", limit) == expected
    await db_session.commit()
    secrets = await _has_secret(db_session, conn)
    assert sum(1 for v in (1, 2, 3) if not secrets[v]) == expected


async def test_gc_limit_upper_bound_is_ten_thousand() -> None:
    """The 10k cap itself, asserted on the DDL (seeding >10k versions is too slow)."""
    from src.database.rls import provider_secret_gc_statements

    assert "GREATEST(LEAST(p_limit, 10000), 0)" in provider_secret_gc_statements()[0]


async def test_gc_keeps_pins_of_a_run_completed_within_the_grace_window(db_session) -> None:
    """The structural-relief child is spawned AFTER its parent is marked completed; in
    that gap the parent's (inherited) pins must still protect the version."""
    tenant, conn = await _connection_with_versions(db_session, 4)
    now = datetime.now(UTC)
    await _run(
        db_session, tenant, "completed", _pin(conn, 1), completed_at=now - timedelta(minutes=5)
    )
    await _run(db_session, tenant, "failed", _pin(conn, 2), completed_at=now - timedelta(minutes=5))
    await _run(
        db_session, tenant, "completed", _pin(conn, 3), completed_at=now - timedelta(hours=2)
    )

    assert await purge_unpinned_secrets(db_session) == 1
    await db_session.commit()
    assert await _has_secret(db_session, conn) == {1: True, 2: True, 3: False, 4: True}


async def test_gc_grace_default_is_one_hour() -> None:
    assert timedelta(hours=1) == DEFAULT_GC_GRACE
    assert ProviderPolicySettings().gc_grace_seconds == 3600


@pytest.mark.parametrize(
    "snapshot",
    [{"provider_slots": "garbage"}],
)
async def test_gc_refuses_to_purge_when_a_live_pin_is_unreadable(db_session, snapshot) -> None:
    """A pin it cannot read cannot be proven absent → purge NOTHING (fail safe)."""
    tenant, conn = await _connection_with_versions(db_session, 2)
    await _run(db_session, tenant, "paused", snapshot)
    with pytest.raises(DBAPIError):
        await purge_unpinned_secrets(db_session)
    await db_session.rollback()
    assert (await _has_secret(db_session, conn))[1] is True


async def test_gc_sees_every_tenant_under_the_restricted_runtime_role(
    db_session_factory,
) -> None:
    """Prod runs as a NOBYPASSRLS role with no tenant bound: a plain scan would see no
    runs (no pins) — the function must still see both the pins and the versions."""
    role = f"rls_probe_{uuid7().hex[:12]}"
    async with db_session_factory() as owner:
        schema = (await owner.execute(text("SELECT current_schema()"))).scalar_one()
        try:
            await owner.execute(text(f'CREATE ROLE "{role}" NOLOGIN NOSUPERUSER NOBYPASSRLS'))
        except ProgrammingError as exc:
            await owner.rollback()
            require_createrole(exc)
        await owner.execute(text(f'GRANT "{role}" TO CURRENT_USER'))
        await owner.execute(text(f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"'))
        await owner.execute(
            text(
                "GRANT EXECUTE ON FUNCTION purge_unpinned_provider_secrets"
                f'(timestamptz, integer) TO "{role}"'
            )
        )
        await owner.commit()
    try:
        async with db_session_factory() as owner:
            tenant, conn = await _connection_with_versions(owner, 3)
            await _run(owner, tenant, "paused", _pin(conn, 1))

        restricted = db_session_factory()
        try:
            await restricted.execute(text(f'SET LOCAL ROLE "{role}"'))
            assert await purge_unpinned_secrets(restricted) == 1
            await restricted.commit()
        finally:
            await restricted.close()

        async with db_session_factory() as owner:
            assert await _has_secret(owner, conn) == {1: True, 2: False, 3: True}
    finally:
        async with db_session_factory() as owner:
            for stmt in (
                f'REVOKE "{role}" FROM CURRENT_USER',
                f'DROP OWNED BY "{role}"',
                f'DROP ROLE IF EXISTS "{role}"',
            ):
                try:
                    await owner.execute(text(stmt))
                    await owner.commit()
                except ProgrammingError:
                    await owner.rollback()
