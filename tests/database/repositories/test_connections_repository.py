"""Provider connections repository + RLS + rewrap, against REAL postgres.

Each test starts from TRUNCATEd tables (``db_engine``). The cross-tenant cases are
the point: a pinned version must only resolve for its own tenant, and under the
restricted (non-owner, NOBYPASSRLS) role the policy alone must hide another
tenant's connections and versions.
"""

from __future__ import annotations

import os
from uuid import UUID, uuid7

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError, ProgrammingError

from scripts.rewrap_provider_secrets import rewrap
from scripts.rewrap_provider_secrets import run as rewrap_run
from src.database.models import TenantConnection, TenantConnectionVersion
from src.database.repositories.admin_repository import AdminRepository
from src.database.repositories.connections_repository import (
    KEEP_SECRET,
    ConnectionsRepository,
    ConnectionVersionConflictError,
    SecretReentryRequiredError,
    sees_all_tenants,
)
from src.services.providers.crypto import (
    Keyring,
    KeyringNotConfiguredError,
    SecretDecryptError,
    peek_key_id,
)
from tests.conftest import require_createrole

pytestmark = pytest.mark.asyncio

KEY_A = "sk-tenant-a-0001"  # pragma: allowlist secret
KEY_B = "sk-tenant-a-0002"  # pragma: allowlist secret


def _some[T](value: T | None) -> T:
    assert value is not None
    return value


@pytest.fixture
def ring() -> Keyring:
    return Keyring([("k1", os.urandom(32))])


async def _tenant(session) -> UUID:
    tenant_id = uuid7()
    assert await AdminRepository(session).ensure_tenant(tenant_id, created_via="super_admin")
    return tenant_id


async def test_versions_supersede_keep_and_clear_the_secret(db_session, ring) -> None:
    repo = ConnectionsRepository(db_session)
    tenant = await _tenant(db_session)

    conn, v1 = await repo.create_version(
        tenant_id=tenant,
        slot="llm",
        kind="openai-compatible",
        base_url="https://api.example.test/v1",
        model="m-1",
        secret=KEY_A,
        keyring=ring,
        expected_version=0,
    )
    assert (conn.current_version, v1.version, v1.key_id) == (1, 1, "k1")
    assert v1.secret_fingerprint == ring.fingerprint(KEY_A)
    assert repo.decrypt_secret(tenant, v1, ring) == KEY_A

    # Absent secret ⇒ carried forward, RE-SEALED for v2 (never a byte copy: AAD differs).
    conn, v2 = await repo.create_version(
        tenant_id=tenant,
        slot="llm",
        kind="openai-compatible",
        base_url="https://API.example.test:443/v2",  # same endpoint, other path/case
        model="m-2",
        secret=KEEP_SECRET,
        keyring=ring,
        expected_version=1,
    )
    assert (conn.current_version, v2.version) == (2, 2)
    assert v2.secret_ciphertext != v1.secret_ciphertext
    assert repo.decrypt_secret(tenant, v2, ring) == KEY_A
    assert v2.secret_fingerprint == v1.secret_fingerprint
    await db_session.refresh(v1)
    assert v1.superseded_at is not None, "the previous version is stamped superseded"
    assert v2.superseded_at is None

    # Pinned v1 still resolves (a paused run keeps its provider) — config unchanged.
    pinned = _some(await repo.get_version(tenant, conn.id, 1))
    assert (pinned.model, repo.decrypt_secret(tenant, pinned, ring)) == ("m-1", KEY_A)

    # "" clears.
    _, v3 = await repo.create_version(
        tenant_id=tenant, slot="llm", kind="anthropic", secret="", expected_version=2
    )
    assert (v3.secret_ciphertext, v3.key_id, v3.secret_fingerprint) == (None, None, None)
    assert repo.decrypt_secret(tenant, v3, ring) is None

    current = _some(await repo.get_current(tenant, "llm"))
    assert (current[0].id, current[1].version, current[1].kind) == (conn.id, 3, "anthropic")
    assert [c.slot for c, _ in await repo.list_for_tenant(tenant)] == ["llm"]


async def test_expected_version_mismatch_conflicts_and_writes_nothing(db_session, ring) -> None:
    repo = ConnectionsRepository(db_session)
    tenant = await _tenant(db_session)
    await repo.create_version(tenant_id=tenant, slot="llm", kind="anthropic", expected_version=0)
    with pytest.raises(ConnectionVersionConflictError) as caught:
        await repo.create_version(
            tenant_id=tenant, slot="llm", kind="anthropic", expected_version=0
        )
    assert (caught.value.expected, caught.value.current) == (0, 1)
    rows = (await db_session.execute(select(TenantConnectionVersion))).scalars().all()
    assert len(rows) == 1


async def test_secret_without_keyring_fails_before_any_write(db_session) -> None:
    repo = ConnectionsRepository(db_session)
    tenant = await _tenant(db_session)
    with pytest.raises(KeyringNotConfiguredError):
        await repo.create_version(
            tenant_id=tenant, slot="llm", kind="anthropic", secret=KEY_A, keyring=Keyring([])
        )
    assert (await db_session.execute(select(TenantConnection))).first() is None


async def test_versions_are_tenant_bound(db_session, ring) -> None:
    repo = ConnectionsRepository(db_session)
    t_a, t_b = await _tenant(db_session), await _tenant(db_session)
    conn, v1 = await repo.create_version(
        tenant_id=t_a, slot="llm", kind="anthropic", secret=KEY_A, keyring=ring
    )
    assert await repo.get_version(t_b, conn.id, 1) is None
    assert await repo.get_current(t_b, "llm") is None
    assert await repo.list_for_tenant(t_b) == []
    with pytest.raises(SecretDecryptError):
        repo.decrypt_secret(t_b, v1, ring)  # AAD binds tenant A
    assert not await repo.soft_delete(t_b, "llm", updated_by=None)
    assert not await repo.record_test(t_b, conn.id, ok=False, error="x")


async def test_soft_delete_frees_the_slot_and_keeps_pins(db_session, ring) -> None:
    repo = ConnectionsRepository(db_session)
    tenant = await _tenant(db_session)
    conn, _ = await repo.create_version(
        tenant_id=tenant, slot="llm", kind="anthropic", secret=KEY_A, keyring=ring
    )
    assert await repo.record_test(tenant, conn.id, ok=False, error="401 from provider")
    await db_session.refresh(conn)
    assert (conn.last_test_ok, conn.last_test_error) == (False, "401 from provider")

    assert await repo.soft_delete(tenant, "llm", updated_by=None)
    assert await repo.get_current(tenant, "llm") is None
    old = _some(await repo.get_version(tenant, conn.id, 1))
    assert old.superseded_at is not None
    assert not await repo.soft_delete(tenant, "llm", updated_by=None)

    fresh, v1 = await repo.create_version(
        tenant_id=tenant,
        slot="llm",
        kind="anthropic",
        secret=KEY_B,
        keyring=ring,
        expected_version=0,
    )
    assert fresh.id != conn.id and v1.version == 1
    assert repo.decrypt_secret(tenant, v1, ring) == KEY_B


async def test_partial_unique_refuses_two_active_connections(db_session) -> None:
    tenant = await _tenant(db_session)
    for _ in range(2):
        db_session.add(
            TenantConnection(id=uuid7(), tenant_id=tenant, slot="llm", current_version=1)
        )
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_purge_candidates_and_purge(db_session, ring) -> None:
    repo = ConnectionsRepository(db_session)
    tenant = await _tenant(db_session)
    conn, _ = await repo.create_version(
        tenant_id=tenant, slot="llm", kind="anthropic", secret=KEY_A, keyring=ring
    )
    await repo.create_version(
        tenant_id=tenant, slot="llm", kind="anthropic", secret=KEY_B, keyring=ring
    )
    await repo.create_version(
        tenant_id=tenant, slot="llm", kind="anthropic", secret=KEY_B, keyring=ring
    )

    refs = await repo.purge_candidates()
    assert [(r.tenant_id, r.connection_id, r.version) for r in refs] == [
        (tenant, conn.id, 1),
        (tenant, conn.id, 2),
    ], "superseded versions with a secret; never the current one"
    pinned = await repo.purge_candidates(exclude=[(conn.id, 1)])
    assert [r.version for r in pinned] == [2], "a run-pinned version is excluded"

    assert await repo.purge_secret(conn.id, 2)
    assert not await repo.purge_secret(conn.id, 3), "the current version is never purged"
    purged = _some(await repo.get_version(tenant, conn.id, 2))
    await db_session.refresh(purged)
    assert (purged.secret_ciphertext, purged.key_id) == (None, None)
    assert purged.purged_at is not None and purged.secret_fingerprint == ring.fingerprint(KEY_B)
    assert [r.version for r in await repo.purge_candidates()] == [1]


async def test_rewrap_moves_every_secret_to_the_primary_key(db_session) -> None:
    old_key, new_key = os.urandom(32), os.urandom(32)
    old_ring = Keyring([("old", old_key)])
    repo = ConnectionsRepository(db_session)
    t_a, t_b = await _tenant(db_session), await _tenant(db_session)
    conn_a, _ = await repo.create_version(
        tenant_id=t_a, slot="llm", kind="anthropic", secret=KEY_A, keyring=old_ring
    )
    await repo.create_version(
        tenant_id=t_a, slot="llm", kind="anthropic", secret=KEY_B, keyring=old_ring
    )
    conn_b, _ = await repo.create_version(
        tenant_id=t_b, slot="llm", kind="anthropic", secret=KEY_A, keyring=old_ring
    )
    await db_session.commit()

    rotated = Keyring([("new", new_key), ("old", old_key)])
    dry = await rewrap(db_session, rotated, apply=False)
    assert (dry.rewrapped, dry.failed) == (3, [])
    assert {
        k for (k,) in (await db_session.execute(select(TenantConnectionVersion.key_id))).all()
    } == {"old"}

    report = await rewrap(db_session, rotated, apply=True)
    await db_session.commit()
    assert (report.rewrapped, report.failed) == (3, [])

    new_only = Keyring([("new", new_key)])
    for tenant, conn, version, expected in (
        (t_a, conn_a, 1, KEY_A),
        (t_a, conn_a, 2, KEY_B),
        (t_b, conn_b, 1, KEY_A),
    ):
        row = _some(await repo.get_version(tenant, conn.id, version))
        await db_session.refresh(row)
        assert row.key_id == "new" and peek_key_id(_some(row.secret_ciphertext)) == "new"
        # Fingerprint is keyed by the primary → recomputed and persisted on rewrap.
        assert row.secret_fingerprint == rotated.fingerprint(expected)
        assert row.secret_fingerprint != old_ring.fingerprint(expected)
        assert repo.decrypt_secret(tenant, row, new_only) == expected
    assert (await rewrap(db_session, rotated, apply=True)).rewrapped == 0


async def test_rewrap_reports_unopenable_secret_and_writes_nothing(db_session) -> None:
    lost = Keyring([("lost", os.urandom(32))])
    repo = ConnectionsRepository(db_session)
    tenant = await _tenant(db_session)
    await repo.create_version(
        tenant_id=tenant, slot="llm", kind="anthropic", secret=KEY_A, keyring=lost
    )
    await db_session.commit()

    report = await rewrap(db_session, Keyring([("new", os.urandom(32))]), apply=True)
    assert report.rewrapped == 0 and len(report.failed) == 1
    assert "unknown key id 'lost'" in report.failed[0][1]


async def test_rls_isolates_connections_and_versions_under_restricted_role(
    db_session_factory, ring
) -> None:
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
                f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA "{schema}" TO "{role}"'
            )
        )
        await owner.commit()

    try:
        async with db_session_factory() as owner:
            repo = ConnectionsRepository(owner)
            t_a, t_b = await _tenant(owner), await _tenant(owner)
            conn_a, _ = await repo.create_version(
                tenant_id=t_a, slot="llm", kind="anthropic", secret=KEY_A, keyring=ring
            )
            conn_b, _ = await repo.create_version(
                tenant_id=t_b, slot="llm", kind="anthropic", secret=KEY_B, keyring=ring
            )
            await owner.commit()

        async def as_role(tenant: UUID | None):
            s = db_session_factory()
            await s.execute(text(f'SET LOCAL ROLE "{role}"'))
            if tenant is not None:
                await s.execute(
                    text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(tenant)}
                )
            return s

        # GUC = A: only A's connection and version; the policy alone hides B's
        # (the raw SELECT carries no tenant filter).
        s = await as_role(t_a)
        try:
            conns = (await s.execute(text("SELECT id FROM tenant_connections"))).scalars().all()
            vers = (
                (await s.execute(text("SELECT connection_id FROM tenant_connection_versions")))
                .scalars()
                .all()
            )
            assert conns == [conn_a.id] and vers == [conn_a.id]
            # Cross-tenant write of a version under B's connection: WITH CHECK refuses.
            with pytest.raises(ProgrammingError) as caught:
                await s.execute(
                    text(
                        "INSERT INTO tenant_connection_versions (connection_id, version, kind) "
                        "VALUES (:c, 9, 'anthropic')"
                    ),
                    {"c": str(conn_b.id)},
                )
            assert "row-level security" in str(caught.value.orig).lower()
        finally:
            await s.rollback()
            await s.close()

        s = await as_role(t_a)
        try:
            upd = await s.execute(
                text("UPDATE tenant_connection_versions SET model = 'x' WHERE connection_id = :c"),
                {"c": str(conn_b.id)},
            )
            assert upd.rowcount == 0
            dele = await s.execute(
                text("DELETE FROM tenant_connections WHERE id = :c"), {"c": str(conn_b.id)}
            )
            assert dele.rowcount == 0
        finally:
            await s.rollback()
            await s.close()

        # No GUC: fail closed on both tables.
        s = await as_role(None)
        try:
            for table in ("tenant_connections", "tenant_connection_versions"):
                count = (await s.execute(text(f"SELECT count(*) FROM {table}"))).scalar_one()  # noqa: S608
                assert count == 0, f"{table}: unset GUC must be fail-closed"
        finally:
            await s.rollback()
            await s.close()
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


@pytest.mark.parametrize(
    ("kind", "base_url"),
    [
        ("anthropic", "https://api.example.test/v1"),
        ("openai-compatible", "https://attacker.example/v1"),
    ],
)
async def test_keep_secret_refused_when_endpoint_or_kind_changes(
    db_session, ring, kind, base_url
) -> None:
    """A stored key must never follow the URL to another host (key exfiltration)."""
    repo = ConnectionsRepository(db_session)
    tenant = await _tenant(db_session)
    await repo.create_version(
        tenant_id=tenant,
        slot="llm",
        kind="openai-compatible",
        base_url="https://api.example.test/v1",
        secret=KEY_A,
        keyring=ring,
    )
    with pytest.raises(SecretReentryRequiredError, match="re-enter the secret"):
        await repo.create_version(
            tenant_id=tenant, slot="llm", kind=kind, base_url=base_url, keyring=ring
        )
    current = _some(await repo.get_current(tenant, "llm"))
    assert current[1].version == 1, "nothing was written"
    # Re-entering the secret (or clearing it) is allowed on the new endpoint.
    _, v2 = await repo.create_version(
        tenant_id=tenant, slot="llm", kind=kind, base_url=base_url, secret=KEY_B, keyring=ring
    )
    assert repo.decrypt_secret(tenant, v2, ring) == KEY_B


async def test_keep_secret_allowed_for_model_change(db_session, ring) -> None:
    repo = ConnectionsRepository(db_session)
    tenant = await _tenant(db_session)
    await repo.create_version(
        tenant_id=tenant,
        slot="llm",
        kind="openai-compatible",
        base_url="https://api.example.test/v1",
        model="a",
        secret=KEY_A,
        keyring=ring,
    )
    _, v2 = await repo.create_version(
        tenant_id=tenant,
        slot="llm",
        kind="openai-compatible",
        base_url="https://api.example.test/v1",
        model="b",
        keyring=ring,
    )
    assert repo.decrypt_secret(tenant, v2, ring) == KEY_A


async def test_rewrap_refuses_under_a_role_that_cannot_see_every_tenant(
    db_session_factory, ring, capsys
) -> None:
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
            text(f'GRANT SELECT, UPDATE ON ALL TABLES IN SCHEMA "{schema}" TO "{role}"')
        )
        await owner.commit()
    try:
        async with db_session_factory() as owner:
            assert await sees_all_tenants(owner, "tenant_connections", "tenant_connection_versions")

        class _Restricted:
            async def __aenter__(self):
                self.s = db_session_factory()
                await self.s.execute(text(f'SET LOCAL ROLE "{role}"'))
                return self.s

            async def __aexit__(self, *exc):
                await self.s.rollback()
                await self.s.close()

        rc = await rewrap_run(_Restricted, ring, apply=True)
        assert rc == 2
        assert "does not bypass row-level security" in capsys.readouterr().err
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


async def test_rewrap_also_reseals_tenant_licence_tokens(db_session) -> None:
    """The keyring seals tenant licence tokens too (mig 0089): a rotation must
    re-seal them, or dropping the old key de-licenses every such tenant."""
    from sqlalchemy import func
    from sqlalchemy import insert as sa_insert

    from scripts.rewrap_provider_secrets import rewrap_licence_tokens
    from src.database.models import Tenant
    from src.services.tenant_licence import open_licence_token, seal_licence_token

    old_key, new_key = os.urandom(32), os.urandom(32)
    old_ring = Keyring([("lrold", old_key)])
    tid = uuid7()
    sealed = seal_licence_token(old_ring, tid, "lic_tok_rotate_me")
    await db_session.execute(
        sa_insert(Tenant).values(
            id=tid,
            name="t",
            licence_token_ciphertext=sealed.ciphertext,
            licence_token_kid=sealed.key_id,
            licence_token_set_at=func.now(),
        )
    )
    await db_session.flush()
    # Rotated ring that can open every token in this schema (the licence-token
    # tests seal with tests/api/routes/_p9_licences.TEST_KEYRING, kid lt1 = bytes
    # 0..31); the transaction is rolled back, so nothing leaks out.
    rotated = Keyring([("lrnew", new_key), ("lrold", old_key), ("lt1", bytes(range(32)))])
    report = await rewrap_licence_tokens(db_session, rotated, apply=True)
    assert report.failed == [] and report.rewrapped >= 1
    row = (
        await db_session.execute(
            select(Tenant.licence_token_ciphertext, Tenant.licence_token_kid).where(
                Tenant.id == tid
            )
        )
    ).one()
    assert row.licence_token_kid == "lrnew"
    only_new = Keyring([("lrnew", new_key)])
    assert open_licence_token(only_new, tid, bytes(row.licence_token_ciphertext)) == (
        "lic_tok_rotate_me"
    )
    unopenable = await rewrap_licence_tokens(
        db_session, Keyring([("zz", os.urandom(32))]), apply=False
    )
    assert any(t == tid for t, _ in unopenable.failed), "an unopenable token is reported"
    await db_session.rollback()
