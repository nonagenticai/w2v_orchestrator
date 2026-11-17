"""P3 provider-slot migrations 0077 (connection tables + RLS) and 0078 (audit vocab).

Independent claims:

1. Revision ids fit ``alembic_version`` and chain linearly; the revisions import no
   application code (frozen literal DDL). DB-free.
2. 0077's frozen RLS text equals the live ``src.database.rls`` generator, so the
   migration-built (prod) and ``create_all``-built (test) tables carry the SAME
   policy. DB-free.
3. ``alembic upgrade head`` and ``create_all`` build the same two tables — columns,
   constraints, indexes, RLS flags (``parity_schemas``).
4. On a real throwaway DATABASE: upgrade 0076 → head with data present, every
   CHECK/FK live; re-upgrade over tables that ALREADY exist (the migrated-but-
   unstamped state) succeeds and keeps the rows; downgrade → 0076 removes the
   tables and restores the old audit CHECK while keeping audit rows written under
   the new vocabulary; upgrade again.
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
from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError

from tests.conftest import TEST_DATABASE_URL, _fail_or_skip_without_database
from tests.support import run_isolation
from tests.support.schema_probe import ParityProbe

_VERSIONS = Path(__file__).resolve().parents[2] / "src/database/migrations/versions"
_PRE = "0076_tenant_is_occupied_fn"
_TABLES = ("tenant_connections", "tenant_connection_versions")
_PROBE_DB = f"connections_upgrade_probe_{os.getpid()}"

TENANT = UUID("11111111-1111-7111-8111-111111111111")
CONN = UUID("cccccccc-0000-7000-8000-000000000001")


def _load(revision: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"m_{revision}", _VERSIONS / f"{revision}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ─── DB-free ─────────────────────────────────────────────────────────────


# ─── create_all ↔ alembic parity ─────────────────────────────────────────


def _shape(conn: Connection, schema: str, table: str) -> tuple[list, list, list, tuple]:
    cols = conn.execute(
        text(
            "SELECT column_name, data_type, is_nullable, column_default "
            "FROM information_schema.columns WHERE table_schema = :s AND table_name = :t "
            "ORDER BY column_name"
        ),
        {"s": schema, "t": table},
    ).all()
    cons = conn.execute(
        text(
            "SELECT c.conname, pg_get_constraintdef(c.oid) FROM pg_constraint c "
            "JOIN pg_class t ON t.oid = c.conrelid JOIN pg_namespace n ON n.oid = t.relnamespace "
            "WHERE n.nspname = :s AND t.relname = :t ORDER BY c.conname"
        ),
        {"s": schema, "t": table},
    ).all()
    idx = conn.execute(
        text(
            "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = :s AND tablename = :t "
            "ORDER BY indexname"
        ),
        {"s": schema, "t": table},
    ).all()
    flags = conn.execute(
        text(
            "SELECT c.relrowsecurity, c.relforcerowsecurity FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = :s AND c.relname = :t"
        ),
        {"s": schema, "t": table},
    ).one()

    def strip(v: object) -> object:
        return v.replace(f"{schema}.", "") if isinstance(v, str) else v

    return (
        [tuple(r) for r in cols],
        [tuple(strip(v) for v in r) for r in cons],
        [tuple(strip(v) for v in r) for r in idx],
        tuple(flags),
    )


def test_alembic_and_create_all_build_the_same_tables(parity_schemas: ParityProbe) -> None:
    probe = parity_schemas
    with probe.engine.connect() as conn:
        for table in _TABLES:
            al = _shape(conn, probe.alembic, table)
            ca = _shape(conn, probe.create_all, table)
            assert al[0], f"{table}: alembic build has no columns"
            assert al == ca, f"{table}: create_all and alembic diverge"
            assert al[3] == (True, False), f"{table}: ENABLE, not FORCE"
        cons = dict(_shape(conn, probe.alembic, "tenant_connection_versions")[1])
        idx = dict(_shape(conn, probe.alembic, "tenant_connections")[2])
    assert cons["tenant_connection_versions_pkey"] == "PRIMARY KEY (connection_id, version)"
    assert "ON DELETE CASCADE" in cons["tenant_connection_versions_connection_id_fkey"]
    assert "WHERE (status = 'active'::text)" in idx["uq_tenant_connections_tenant_slot_active"]
    assert idx["uq_tenant_connections_tenant_slot_active"].startswith("CREATE UNIQUE INDEX")


# ─── real database: upgrade / drift / downgrade / upgrade ────────────────


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


def _rejects(engine: Engine, sql: str, params: dict[str, object] | None = None) -> bool:
    try:
        with engine.begin() as conn:
            conn.execute(text(sql), params or {})
    except IntegrityError:
        return True
    return False


def _seed(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO tenants (id) VALUES (:t)"), {"t": TENANT})
        conn.execute(
            text(
                "INSERT INTO tenant_connections (id, tenant_id, slot, current_version) "
                "VALUES (:c, :t, 'llm', 2)"
            ),
            {"c": CONN, "t": TENANT},
        )
        conn.execute(
            text(
                "INSERT INTO tenant_connection_versions "
                "(connection_id, version, kind, secret_ciphertext, key_id, secret_fingerprint, "
                " superseded_at) VALUES "
                "(:c, 1, 'openai-compatible', '\\x01'::bytea, 'k1', 'abcdef012345', now()), "
                "(:c, 2, 'anthropic', NULL, NULL, NULL, NULL)"
            ),
            {"c": CONN},
        )
        conn.execute(
            text(
                "INSERT INTO admin_audit_log (id, action, target_tenant_id) "
                "VALUES ('01999999-0000-7000-8000-0000000000c1', 'connection.set', :t)"
            ),
            {"t": TENANT},
        )


def _counts(engine: Engine) -> tuple[int, int]:
    with engine.connect() as conn:
        return (
            conn.execute(text("SELECT count(*) FROM tenant_connections")).scalar_one(),
            conn.execute(text("SELECT count(*) FROM tenant_connection_versions")).scalar_one(),
        )


def _version_num(engine: Engine) -> str:
    with engine.connect() as conn:
        return conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()


def test_upgrade_constraints_drift_and_round_trip(probe) -> None:
    cfg, engine = probe
    command.upgrade(cfg, _PRE)
    command.upgrade(cfg, "head")
    _seed(engine)
    assert _counts(engine) == (1, 2)

    # Every constraint is live.
    other = UUID("cccccccc-0000-7000-8000-0000000000ff")
    assert _rejects(
        engine,
        "INSERT INTO tenant_connections (id, tenant_id, slot, current_version) "
        "VALUES (:o, :t, 'llm', 1)",
        {"o": other, "t": TENANT},
    ), "a second ACTIVE connection for the same (tenant, slot) must be refused"
    assert not _rejects(
        engine,
        "INSERT INTO tenant_connections (id, tenant_id, slot, current_version, status, "
        "deleted_at) VALUES (:o, :t, 'llm', 1, 'deleted', now())",
        {"o": other, "t": TENANT},
    ), "a DELETED connection must not block the slot"
    assert _rejects(
        engine,
        "UPDATE tenant_connections SET status = 'deleted' WHERE id = :c",
        {"c": CONN},
    ), "status=deleted without deleted_at must be refused"
    assert _rejects(engine, "UPDATE tenant_connections SET slot = 'LLM' WHERE id = :c", {"c": CONN})
    assert not _rejects(
        engine, "UPDATE tenant_connections SET slot = 'publish.youtube' WHERE id = :c", {"c": CONN}
    )
    assert _rejects(
        engine,
        "INSERT INTO tenant_connections (id, tenant_id, slot, current_version) VALUES "
        "('cccccccc-0000-7000-8000-0000000000fe', '99999999-9999-7999-8999-999999999999', "
        "'llm', 1)",
    ), "unknown tenant must violate the FK"
    assert _rejects(
        engine,
        "UPDATE tenant_connection_versions SET key_id = NULL WHERE connection_id = :c "
        "AND version = 1",
        {"c": CONN},
    ), "ciphertext without key_id must be refused"
    assert _rejects(
        engine,
        "UPDATE tenant_connection_versions SET purged_at = now() WHERE connection_id = :c "
        "AND version = 1",
        {"c": CONN},
    ), "purged_at with ciphertext still present must be refused"
    assert _rejects(
        engine,
        "UPDATE tenant_connection_versions SET secret_fingerprint = 'XYZ' WHERE "
        "connection_id = :c AND version = 1",
        {"c": CONN},
    )
    assert _rejects(
        engine,
        "INSERT INTO admin_audit_log (id, action) "
        "VALUES ('01999999-0000-7000-8000-0000000000c2', 'connection.bogus')",
    )
    assert not _rejects(
        engine,
        "INSERT INTO admin_audit_log (id, action) "
        "VALUES ('01999999-0000-7000-8000-0000000000c3', 'connection.test_failed')",
    )

    # Drift tolerance: tables + policies already present, alembic_version says 0076
    # (migrated but unstamped). The real `upgrade head` must re-apply cleanly.
    with engine.begin() as conn:
        conn.execute(text("UPDATE alembic_version SET version_num = :v"), {"v": _PRE})
    command.upgrade(cfg, "head")
    assert _counts(engine) == (2, 2), "re-apply must keep existing rows"
    with engine.connect() as conn:
        policies = conn.execute(
            text(
                "SELECT tablename FROM pg_policies WHERE policyname = 'tenant_isolation' "
                "AND tablename IN ('tenant_connections', 'tenant_connection_versions')"
            )
        ).scalars()
        assert sorted(policies) == sorted(_TABLES)

    # Downgrade to 0076: tables gone; the old CHECK is back NOT VALID, so the audit
    # rows written under the new vocabulary survive, and new ones are refused.
    command.downgrade(cfg, _PRE)
    with engine.connect() as conn:
        for table in _TABLES:
            assert conn.execute(text("SELECT to_regclass(:t)"), {"t": table}).scalar_one() is None
        assert (
            conn.execute(
                text("SELECT count(*) FROM admin_audit_log WHERE action LIKE 'connection.%'")
            ).scalar_one()
            == 2
        )
    assert _rejects(
        engine,
        "INSERT INTO admin_audit_log (id, action) "
        "VALUES ('01999999-0000-7000-8000-0000000000c4', 'connection.set')",
    )

    command.upgrade(cfg, "head")
    assert _counts(engine) == (0, 0)
