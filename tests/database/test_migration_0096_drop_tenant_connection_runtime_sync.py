"""0096 drops tenant_connections.runtime_sync — drift-tolerant both ways, on a real throwaway DB.

upgrade head → the column is gone; downgrade re-adds it (nullable JSONB, empty); a
DB whose column was already dropped out-of-band still upgrades cleanly, and one
whose column was re-added out-of-band still downgrades cleanly.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text

from tests.conftest import TEST_DATABASE_URL, _fail_or_skip_without_database
from tests.database.test_migration_0090_api_keys_rls import _drop_probe, _server_engine
from tests.support import run_isolation

_REV = "0096_drop_conn_runtime_sync"
_PRE = "0095_tenant_conn_runtime_sync"
_PROBE_DB = f"conn_runtime_sync_drop_probe_{os.getpid()}"


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Config, Engine]]:
    from sqlalchemy import create_engine
    from sqlalchemy.exc import OperationalError

    import tests.database.test_migration_0090_api_keys_rls as m0090

    monkeypatch.setattr(m0090, "_PROBE_DB", _PROBE_DB)
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
    monkeypatch.setenv("DATABASE_URL", url)
    if os.environ.get("PGOPTIONS") is not None:
        monkeypatch.setenv("PGOPTIONS", run_isolation.pgoptions("public"))
    engine = create_engine(url)
    try:
        yield cfg, engine
    finally:
        engine.dispose()
        _drop_probe(server)
        server.dispose()


def _columns(engine: Engine) -> dict[str, tuple[str, str]]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT column_name, is_nullable, data_type FROM information_schema.columns "
                "WHERE table_name = 'tenant_connections' AND column_name = 'runtime_sync'"
            )
        ).all()
    return {r[0]: (r[1], r[2]) for r in rows}


def test_upgrade_drops_downgrade_restores_and_both_tolerate_drift(probe) -> None:
    cfg, engine = probe
    command.upgrade(cfg, _PRE)
    assert _columns(engine) == {"runtime_sync": ("YES", "jsonb")}
    command.upgrade(cfg, _REV)
    assert _columns(engine) == {}
    with engine.connect() as conn:
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == _REV

    # Downgrade restores the (empty) column; with it re-added out-of-band first,
    # the downgrade is still clean.
    command.downgrade(cfg, _PRE)
    assert _columns(engine) == {"runtime_sync": ("YES", "jsonb")}
    command.upgrade(cfg, _REV)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE tenant_connections ADD COLUMN runtime_sync JSONB"))
    command.downgrade(cfg, _PRE)
    assert _columns(engine) == {"runtime_sync": ("YES", "jsonb")}

    # Drift the other way: column already dropped, stamp behind -> upgrade is clean.
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE tenant_connections DROP COLUMN runtime_sync"))
    command.upgrade(cfg, "head")
    assert _columns(engine) == {}
