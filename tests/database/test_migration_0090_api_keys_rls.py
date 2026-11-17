"""Migration 0090 (``tenant_api_keys`` RLS + pre-tenant key resolvers).

1. DB-free: the id fits ``alembic_version`` and chains onto 0089; the revision
   imports no application code; its frozen DDL equals the live generators.
2. ``alembic upgrade head`` and ``create_all`` build the same table + RLS flags.
3. On a real throwaway DATABASE: fresh ``upgrade head`` → RLS ENABLEd (not
   FORCEd), one ``tenant_isolation`` policy, both resolvers SECURITY DEFINER and
   not PUBLIC-executable, ``resolve_api_key`` answers by (prefix, hash); a drift
   re-apply keeps rows; ``downgrade -1`` removes all of it; ``upgrade head`` again.
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
from sqlalchemy.exc import OperationalError

from src.auth.api_key_format import generate_api_key
from src.database import rls
from tests.conftest import TEST_DATABASE_URL, _fail_or_skip_without_database
from tests.database.test_migration_0077_tenant_connections import _shape
from tests.support import run_isolation
from tests.support.schema_probe import ParityProbe

_VERSIONS = Path(__file__).resolve().parents[2] / "src/database/migrations/versions"
_REV = "0090_api_keys_rls"
_PRE = "0089_tenant_licence_token"
_PROBE_DB = f"api_keys_rls_probe_{os.getpid()}"

TENANT = UUID("66666666-6666-7666-8666-666666666666")
KEY_ID = UUID("01999999-0000-7000-8000-000000000090")


def _load(revision: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"m_{revision}", _VERSIONS / f"{revision}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ─── DB-free ─────────────────────────────────────────────────────────────


def test_frozen_ddl_matches_the_live_generators() -> None:
    m = _load(_REV)
    assert tuple(m._RLS_UPGRADE_STATEMENTS) == tuple(rls.tenant_api_keys_upgrade_statements())
    assert tuple(m._FUNCTION_STATEMENTS) == tuple(rls.api_key_functions_statements())
    assert tuple(m._DROP_FUNCTION_STATEMENTS) == tuple(rls.drop_api_key_functions_statements())


def test_resolver_never_returns_the_hash() -> None:
    fn = rls.api_key_functions_statements()[0]
    returns = fn.split("RETURNS TABLE", 1)[1].split("LANGUAGE", 1)[0]
    assert "key_hash" not in returns
    assert "SECURITY DEFINER" in fn and "pg_temp" in fn


# ─── create_all ↔ alembic parity ─────────────────────────────────────────


def test_alembic_and_create_all_build_the_same_table(parity_schemas: ParityProbe) -> None:
    probe = parity_schemas
    with probe.engine.connect() as conn:
        al = _shape(conn, probe.alembic, "tenant_api_keys")
        ca = _shape(conn, probe.create_all, "tenant_api_keys")
    assert al[0], "alembic build has no columns"
    assert al == ca, "create_all and alembic diverge on tenant_api_keys"
    assert al[3] == (True, False), "ENABLE, not FORCE — exactly like the sibling tables"


# ─── real database ───────────────────────────────────────────────────────


def _server_engine() -> Engine:
    return create_engine(TEST_DATABASE_URL, isolation_level="AUTOCOMMIT")


def _drop_probe(engine: Engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("SET statement_timeout = 30000"))
        try:
            conn.execute(text(f"DROP DATABASE IF EXISTS {_PROBE_DB} WITH (FORCE)"))
        except OperationalError as exc:
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
    if prev_pgoptions is not None:
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


def _state(engine: Engine) -> dict[str, object]:
    with engine.connect() as conn:
        flags = conn.execute(
            text(
                "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                "WHERE oid = 'tenant_api_keys'::regclass"
            )
        ).one()
        policies = conn.execute(
            text(
                "SELECT count(*) FROM pg_policies WHERE policyname = 'tenant_isolation' "
                "AND tablename = 'tenant_api_keys'"
            )
        ).scalar_one()
        fns = conn.execute(
            text(
                "SELECT proname, prosecdef, has_function_privilege('public', oid, 'EXECUTE') "
                "FROM pg_proc WHERE proname IN ('resolve_api_key', 'api_key_prefix_taken') "
                "ORDER BY proname"
            )
        ).all()
        rows = conn.execute(text("SELECT count(*) FROM tenant_api_keys")).scalar_one()
    return {
        "flags": tuple(flags),
        "policies": policies,
        "fns": [tuple(f) for f in fns],
        "rows": rows,
    }


_ON: dict[str, object] = {
    "flags": (True, False),
    "policies": 1,
    "fns": [("api_key_prefix_taken", True, False), ("resolve_api_key", True, False)],
}
_OFF: dict[str, object] = {"flags": (False, False), "policies": 0, "fns": []}


def _sub(state: dict[str, object], expected: dict[str, object]) -> dict[str, object]:
    return {k: state[k] for k in expected}


def test_upgrade_head_downgrade_and_upgrade_again(probe) -> None:
    cfg, engine = probe
    command.upgrade(cfg, "head")
    assert _sub(_state(engine), _ON) == _ON

    key = generate_api_key()
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO tenants (id) VALUES (:t)"), {"t": TENANT})
        conn.execute(
            text(
                "INSERT INTO tenant_api_keys (id, tenant_id, name, prefix, key_hash) "
                "VALUES (:id, :t, 'ci', :p, :h)"
            ),
            {"id": KEY_ID, "t": TENANT, "p": key.prefix, "h": key.key_hash},
        )
    with engine.connect() as conn:
        hit = conn.execute(
            text("SELECT key_id, tenant_id FROM resolve_api_key(:p, :h)"),
            {"p": key.prefix, "h": key.key_hash},
        ).all()
        miss = conn.execute(
            text("SELECT count(*) FROM resolve_api_key(:p, :h)"),
            {"p": key.prefix, "h": b"\x00" * 32},
        ).scalar_one()
        taken = conn.execute(
            text("SELECT api_key_prefix_taken(:p)"), {"p": key.prefix}
        ).scalar_one()
    assert [tuple(r) for r in hit] == [(KEY_ID, TENANT)]
    assert miss == 0, "the right prefix with the wrong hash matches nothing"
    assert taken is True

    # Drift: everything present, alembic_version back at 0089 → re-apply is clean.
    with engine.begin() as conn:
        conn.execute(text("UPDATE alembic_version SET version_num = :v"), {"v": _PRE})
    command.upgrade(cfg, "head")
    state = _state(engine)
    assert _sub(state, _ON) == _ON and state["rows"] == 1, "re-apply keeps rows"

    # To _PRE explicitly, not "-1": 0091 now sits on top of this revision.
    command.downgrade(cfg, _PRE)
    state = _state(engine)
    assert _sub(state, _OFF) == _OFF and state["rows"] == 1, "downgrade keeps the keys"

    command.upgrade(cfg, "head")
    state = _state(engine)
    assert _sub(state, _ON) == _ON and state["rows"] == 1
