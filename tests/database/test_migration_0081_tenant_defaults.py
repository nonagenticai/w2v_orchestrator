"""P6 migrations 0081 (``tenant_defaults`` + RLS) and 0082 (audit vocabulary).

Independent claims:

1. Revision ids fit ``alembic_version`` and chain linearly onto 0080 (YouTube audit actions); the revisions
   import no application code (frozen literal DDL). DB-free.
2. 0081's frozen RLS text equals the live ``src.database.rls`` generator, so the
   migration-built (prod) and ``create_all``-built (test) tables carry the SAME
   policy; 0082's vocabulary is a strict superset of 0080's and equals the model's
   CHECK. DB-free.
3. ``alembic upgrade head`` and ``create_all`` build the same table — columns,
   constraints, indexes, RLS flags (``parity_schemas``).
4. On a real throwaway DATABASE: upgrade 0080 → head with data present, every
   CHECK/FK live (incl. ON DELETE CASCADE from ``tenants``); re-upgrade over a table
   that ALREADY exists (migrated but unstamped) keeps the rows; downgrade → 0080
   removes the table and restores the old audit CHECK while keeping audit rows written
   under the new vocabulary; upgrade again.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError

from tests.conftest import TEST_DATABASE_URL, _fail_or_skip_without_database
from tests.database.test_migration_0077_tenant_connections import _shape
from tests.support import run_isolation
from tests.support.schema_probe import ParityProbe

_VERSIONS = Path(__file__).resolve().parents[2] / "src/database/migrations/versions"
_PRE = "0080_youtube_audit_actions"
_PROBE_DB = f"defaults_upgrade_probe_{os.getpid()}"

TENANT = UUID("22222222-2222-7222-8222-222222222222")


def _load(revision: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"m_{revision}", _VERSIONS / f"{revision}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _action_set(check: str) -> set[str]:
    return set(check.split("(", 1)[1].rstrip(")").split(","))


# ─── DB-free ─────────────────────────────────────────────────────────────


# ─── create_all ↔ alembic parity ─────────────────────────────────────────


def test_alembic_and_create_all_build_the_same_table(parity_schemas: ParityProbe) -> None:
    probe = parity_schemas
    with probe.engine.connect() as conn:
        al = _shape(conn, probe.alembic, "tenant_defaults")
        ca = _shape(conn, probe.create_all, "tenant_defaults")
    assert al[0], "alembic build has no columns"
    assert al == ca, "create_all and alembic diverge on tenant_defaults"
    assert al[3] == (True, False), "ENABLE, not FORCE"
    cons = dict(al[1])
    assert cons["tenant_defaults_pkey"] == "PRIMARY KEY (tenant_id)"
    assert "ON DELETE CASCADE" in cons["tenant_defaults_tenant_id_fkey"]


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


def _rejects(engine: Engine, sql: str, params: Mapping[str, object] | None = None) -> bool:
    try:
        with engine.begin() as conn:
            conn.execute(text(sql), dict(params or {}))
    except IntegrityError:
        return True
    return False


def _rows(engine: Engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM tenant_defaults")).scalar_one()


def _version_num(engine: Engine) -> str:
    with engine.connect() as conn:
        return conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()


def test_upgrade_constraints_drift_and_round_trip(probe) -> None:
    cfg, engine = probe
    command.upgrade(cfg, _PRE)
    command.upgrade(cfg, "head")
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO tenants (id) VALUES (:t)"), {"t": TENANT})
    t = {"t": TENANT}
    assert not _rejects(
        engine,
        "INSERT INTO tenant_defaults (tenant_id, voice, retention_mode) "
        "VALUES (:t, '{\"voice_id\": \"af_heart\"}', 'off')",
        t,
    )
    with engine.connect() as conn:
        assert conn.execute(text("SELECT version FROM tenant_defaults")).scalar_one() == 1

    # Every constraint is live.
    assert _rejects(engine, "INSERT INTO tenant_defaults (tenant_id) VALUES (:t)", t), "PK"
    assert _rejects(
        engine,
        "INSERT INTO tenant_defaults (tenant_id) VALUES ('99999999-9999-7999-8999-999999999999')",
    ), "unknown tenant must violate the FK"
    assert _rejects(engine, "UPDATE tenant_defaults SET retention_mode = 'sometimes'")
    assert _rejects(engine, "UPDATE tenant_defaults SET version = 0")
    # Static column names from this literal tuple; no input reaches the SQL.
    for col in ("voice", "brand", "captions", "voice_direction", "model_tier"):
        set_ = f"UPDATE tenant_defaults SET {col} ="  # noqa: S608
        assert _rejects(engine, f"{set_} 'null'::jsonb"), col
        assert _rejects(engine, f"{set_} '[]'::jsonb"), col
        assert not _rejects(engine, f"{set_} '{{}}'::jsonb"), col
        assert not _rejects(engine, f"{set_} NULL"), col
    assert _rejects(
        engine,
        "INSERT INTO admin_audit_log (id, action) "
        "VALUES ('01999999-0000-7000-8000-0000000000d1', 'defaults.bogus')",
    )
    assert not _rejects(
        engine,
        "INSERT INTO admin_audit_log (id, action, target_tenant_id) "
        "VALUES ('01999999-0000-7000-8000-0000000000d2', 'defaults.update', :t)",
        t,
    )
    with engine.connect() as conn:
        policy = conn.execute(
            text(
                "SELECT count(*) FROM pg_policies WHERE policyname = 'tenant_isolation' "
                "AND tablename = 'tenant_defaults'"
            )
        ).scalar_one()
    assert policy == 1

    # Drift tolerance: table + policy present, alembic_version says 0080.
    with engine.begin() as conn:
        conn.execute(text("UPDATE alembic_version SET version_num = :v"), {"v": _PRE})
    command.upgrade(cfg, "head")
    assert _rows(engine) == 1, "re-apply must keep existing rows"

    # The tenant row going away takes its defaults with it.
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM tenants WHERE id = :t"), t)
    assert _rows(engine) == 0

    # Downgrade to 0080: table gone; the 0080 CHECK is back NOT VALID, so audit rows
    # written under the new vocabulary survive, and new ones are refused.
    command.downgrade(cfg, _PRE)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT to_regclass('tenant_defaults')")).scalar_one() is None
        assert (
            conn.execute(
                text("SELECT count(*) FROM admin_audit_log WHERE action = 'defaults.update'")
            ).scalar_one()
            == 1
        )
    assert _rejects(
        engine,
        "INSERT INTO admin_audit_log (id, action) "
        "VALUES ('01999999-0000-7000-8000-0000000000d3', 'defaults.update')",
    )

    command.upgrade(cfg, "head")
    assert _rows(engine) == 0
