"""P2 tenancy migrations 0072-0075 — upgrade over a database that ALREADY HAS DATA.

The from-scratch replay (``test_migration_replay_check``) and the create_all ↔
alembic parity gates prove the chain builds the right SHAPE on an empty database.
They cannot see what 0072 does to EXISTING rows — the part that runs once in prod
against the live ``tenant_members`` roster. So this test:

1. builds a throwaway database at ``0071_stage_step_results`` (the pre-P2 head);
2. seeds the shapes prod holds: plain members, a tenant_admin, a super_admin with
   a home row AND a scope-switch row in another tenant, a member with an EMPTY
   roles array, a row whose ``roles`` is a jsonb OBJECT (not an array), a
   pre-existing suspended+named tenants row, an audit row;
3. runs ``alembic upgrade head`` — the real command, so ``alembic_version`` is
   written exactly as in prod (a too-long revision id fails HERE);
4. asserts the backfilled role/status/created_at, the tenants rows and their
   provenance, and that every new CHECK is live;
5. re-applies 0072's statements on the migrated DB (drift tolerance), then
   downgrades to 0071 and upgrades again (round trip keeps the data).
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError

from tests.conftest import TEST_DATABASE_URL, _fail_or_skip_without_database
from tests.support import run_isolation

_VERSIONS = Path(__file__).resolve().parents[2] / "src/database/migrations/versions"
_PRE = "0071_stage_step_results"

_PROBE_DB = f"tenancy_upgrade_probe_{os.getpid()}"

# Fixed ids so a failure message names the row.
T1 = UUID("11111111-1111-7111-8111-111111111111")  # pre-existing tenants row (suspended)
T2 = UUID("22222222-2222-7222-8222-222222222222")  # only reached via super_admin switch
T3 = UUID("33333333-3333-7333-8333-333333333333")  # member-only tenant, no tenants row
M1 = UUID("aaaaaaaa-0000-7000-8000-000000000001")  # plain member @T1
A1 = UUID("aaaaaaaa-0000-7000-8000-000000000002")  # tenant_admin @T1
S1 = UUID("aaaaaaaa-0000-7000-8000-000000000003")  # super_admin: home T1 + switch T2
U2 = UUID("aaaaaaaa-0000-7000-8000-000000000004")  # empty roles @T3
W1 = UUID("aaaaaaaa-0000-7000-8000-000000000005")  # roles is a jsonb OBJECT @T3


def _load(revision: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"m_{revision}", _VERSIONS / f"{revision}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ─── seeded upgrade ──────────────────────────────────────────────────────


def _server_engine() -> Engine:
    return create_engine(TEST_DATABASE_URL, isolation_level="AUTOCOMMIT")


def _drop_probe(engine: Engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("SET statement_timeout = 30000"))
        try:
            conn.execute(text(f"DROP DATABASE IF EXISTS {_PROBE_DB} WITH (FORCE)"))
        except OperationalError as exc:  # cleanup fallout must not mask the result
            warnings.warn(f"could not drop {_PROBE_DB}: {exc}", stacklevel=2)


@pytest.fixture
def probe() -> Iterator[tuple[Config, Engine]]:
    """A throwaway DATABASE (so ``alembic_version`` behaves as in prod) that
    alembic's env.py targets via ``DATABASE_URL``."""
    server = _server_engine()
    try:
        with server.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:
        _fail_or_skip_without_database(exc)
    _drop_probe(server)
    with server.connect() as conn:
        conn.execute(text(f"CREATE DATABASE {_PROBE_DB}"))

    url = f"{TEST_DATABASE_URL.rsplit('/', 1)[0]}/{_PROBE_DB}"
    cfg = Config()
    cfg.set_main_option("script_location", "src/database/migrations")
    prev_url = os.environ.get("DATABASE_URL")
    prev_pgoptions = os.environ.get("PGOPTIONS")
    os.environ["DATABASE_URL"] = url
    if prev_pgoptions is not None:  # run isolation pins search_path; probe uses public
        os.environ["PGOPTIONS"] = run_isolation.pgoptions("public")
    engine = create_engine(url)
    try:
        yield cfg, engine
    finally:
        engine.dispose()
        if prev_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = prev_url
        if prev_pgoptions is not None:
            os.environ["PGOPTIONS"] = prev_pgoptions
        _drop_probe(server)
        server.dispose()


def _seed(engine: Engine) -> None:
    member_sql = text(
        "INSERT INTO tenant_members (subject_id, tenant_id, email, roles, first_seen_at, "
        "last_seen_at) VALUES (:s, :t, :e, CAST(:r AS jsonb), :f, :f)"
    )
    rows = [
        (M1, T1, "m1@example.test", '["member"]', "2026-01-01T00:00:00Z"),
        (A1, T1, "a1@example.test", '["member","tenant_admin"]', "2026-01-02T00:00:00Z"),
        (S1, T1, "s1@example.test", '["super_admin"]', "2026-01-03T00:00:00Z"),
        (S1, T2, "s1@example.test", '["super_admin"]', "2026-01-04T00:00:00Z"),
        (U2, T3, None, "[]", "2026-01-05T00:00:00Z"),
        (W1, T3, None, '{"tenant_admin": true}', "2026-01-06T00:00:00Z"),
    ]
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO tenants (id, name, status) VALUES (:id, 'acme', 'suspended')"),
            {"id": T1},
        )
        for s, t, e, r, f in rows:
            conn.execute(member_sql, {"s": s, "t": t, "e": e, "r": r, "f": f})
        conn.execute(
            text(
                "INSERT INTO admin_audit_log (id, actor_subject_id, action, target_tenant_id) "
                "VALUES ('01999999-0000-7000-8000-000000000001', :a, 'member.grant_role', :t)"
            ),
            {"a": A1, "t": T1},
        )


def _members(engine: Engine) -> dict[tuple[UUID, UUID], tuple[str, str, bool, bool]]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT subject_id, tenant_id, role, status, removed_at IS NULL, "
                "created_at = first_seen_at FROM tenant_members"
            )
        ).all()
    return {(r[0], r[1]): (r[2], r[3], r[4], r[5]) for r in rows}


def _tenants(engine: Engine) -> dict[UUID, tuple[str | None, str, str]]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id, name, status, created_via FROM tenants")).all()
    return {r[0]: (r[1], r[2], r[3]) for r in rows}


def _rejects(engine: Engine, sql: str, params: dict[str, object] | None = None) -> bool:
    try:
        with engine.begin() as conn:
            conn.execute(text(sql), params or {})
    except IntegrityError:
        return True
    return False


_EXPECTED_MEMBERS = {
    (M1, T1): ("member", "active", True, True),
    (A1, T1): ("tenant_admin", "active", True, True),
    # super_admin-only rows are NEVER tenant_admin: the authority is the realm
    # role at request time. Mapping them would turn every scope-switch row into
    # a cross-tenant admin grant.
    (S1, T1): ("member", "active", True, True),
    # The migration cannot know S1's home tenant; scripts/backfill_tenancy.py
    # removes this scope-switch row. Until then it is a plain member row.
    (S1, T2): ("member", "active", True, True),
    (U2, T3): ("member", "active", True, True),
    # ``?`` on a jsonb OBJECT matches KEYS — guarded by jsonb_typeof = 'array'.
    (W1, T3): ("member", "active", True, True),
}
_EXPECTED_TENANTS = {
    T1: ("acme", "suspended", "backfill"),  # pre-existing row: untouched, stamped backfill
    T2: (None, "active", "backfill"),
    T3: (None, "active", "backfill"),
}


def test_upgrade_over_seeded_roster_backfills_role_status_and_tenants(probe) -> None:
    cfg, engine = probe
    command.upgrade(cfg, _PRE)
    _seed(engine)

    command.upgrade(cfg, "head")

    assert _members(engine) == _EXPECTED_MEMBERS
    assert _tenants(engine) == _EXPECTED_TENANTS

    # A record_seen-shaped insert (no new columns) still works: defaults apply.
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO tenant_members (subject_id, tenant_id, roles) VALUES (:s, :t, '[]')"),
            {"s": UUID("aaaaaaaa-0000-7000-8000-0000000000ff"), "t": T3},
        )
        row = conn.execute(
            text("SELECT role, status FROM tenant_members WHERE subject_id = :s"),
            {"s": UUID("aaaaaaaa-0000-7000-8000-0000000000ff")},
        ).one()
        assert tuple(row) == ("member", "active")
        # New tenants rows default to super_admin provenance (today's only writers).
        conn.execute(
            text("INSERT INTO tenants (id) VALUES ('44444444-4444-7444-8444-444444444444')")
        )
        assert (
            conn.execute(
                text(
                    "SELECT created_via FROM tenants "
                    "WHERE id = '44444444-4444-7444-8444-444444444444'"
                )
            ).scalar_one()
            == "super_admin"
        )

    # Every new CHECK is live.
    assert _rejects(
        engine, "UPDATE tenant_members SET role = 'owner' WHERE subject_id = :s", {"s": M1}
    )
    assert _rejects(
        engine, "UPDATE tenant_members SET status = 'removed' WHERE subject_id = :s", {"s": M1}
    ), "status=removed without removed_at must be refused"
    assert not _rejects(
        engine,
        "UPDATE tenant_members SET status = 'removed', removed_at = now() "
        "WHERE subject_id = :s AND tenant_id = :t",
        {"s": S1, "t": T2},
    )
    assert _rejects(engine, "UPDATE tenants SET created_via = 'magic' WHERE id = :t", {"t": T2})
    assert not _rejects(
        engine, "UPDATE tenants SET status = 'licence_lapsed' WHERE id = :t", {"t": T3}
    )
    assert _rejects(engine, "UPDATE tenants SET status = 'gone' WHERE id = :t", {"t": T3})
    assert not _rejects(
        engine,
        "INSERT INTO admin_audit_log (id, action) "
        "VALUES ('01999999-0000-7000-8000-000000000002', 'member.invite')",
    )
    assert _rejects(
        engine,
        "INSERT INTO admin_audit_log (id, action) "
        "VALUES ('01999999-0000-7000-8000-000000000003', 'bogus.action')",
    )
    assert _rejects(
        engine,
        "INSERT INTO tenant_api_keys (id, tenant_id, name, prefix, key_hash) VALUES "
        "('01999999-0000-7000-8000-000000000004', :t, 'k', 'NOT-B32!', "
        "decode(repeat('ab', 32), 'hex'))",
        {"t": T1},
    )
    assert _rejects(
        engine,
        "INSERT INTO tenant_invites (id, tenant_id, email, token_hash, expires_at) VALUES "
        "('01999999-0000-7000-8000-000000000005', '55555555-5555-7555-8555-555555555555', "
        "'x@example.test', decode(repeat('cd', 32), 'hex'), now() + interval '1 day')",
    ), "invite for an unknown tenant must violate the FK"


def test_reapplied_and_round_tripped_upgrade_is_drift_tolerant(probe) -> None:
    cfg, engine = probe
    command.upgrade(cfg, _PRE)
    _seed(engine)
    command.upgrade(cfg, "head")

    # Drift tolerance: 0072's statements re-run against an already-migrated DB
    # (the "migrated but unstamped" state) without raising or changing data.
    with engine.begin() as conn:
        for statement in _load("0072_tenancy_membership")._UPGRADE:
            conn.execute(text(statement))
        for revision in ("0073_tenant_invites", "0074_tenant_api_keys"):
            for statement in _load(revision)._UPGRADE:
                conn.execute(text(statement))
    assert _members(engine) == _EXPECTED_MEMBERS
    assert _tenants(engine) == _EXPECTED_TENANTS

    # Downgrade back to the pre-P2 head, then forward again: the roster survives,
    # the backfill re-derives the same result.
    command.downgrade(cfg, _PRE)
    with engine.connect() as conn:
        cols = set(
            conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = 'tenant_members'"
                )
            ).scalars()
        )
        assert "role" not in cols and "status" not in cols
        assert conn.execute(text("SELECT to_regclass('tenant_invites')")).scalar_one() is None
    command.upgrade(cfg, "head")
    assert _members(engine) == _EXPECTED_MEMBERS
    assert _tenants(engine) == _EXPECTED_TENANTS
