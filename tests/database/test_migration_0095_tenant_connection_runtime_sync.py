"""0095 tenant_connections.runtime_sync — drift-tolerant in BOTH directions, on a real throwaway DB.

upgrade to 0095 → the nullable JSONB column exists (0096 drops it again, see
``test_migration_0096_drop_tenant_connection_runtime_sync``); a DB that lost it (drift) still
downgrades cleanly; a re-stamped DB that already has it re-applies cleanly.
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

_REV = "0095_tenant_conn_runtime_sync"
_PRE = "0094_runs_paused_time"
_PROBE_DB = f"conn_runtime_sync_probe_{os.getpid()}"


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


def test_upgrade_drifted_downgrade_and_reapply(probe) -> None:
    cfg, engine = probe
    command.upgrade(cfg, _REV)
    assert _columns(engine) == {"runtime_sync": ("YES", "jsonb")}

    # Drift: the column vanished out-of-band; the downgrade must still succeed.
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE tenant_connections DROP COLUMN runtime_sync"))
    command.downgrade(cfg, _PRE)
    assert _columns(engine) == {}

    # Drift the other way: column present, stamp behind -> re-apply is clean.
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE tenant_connections ADD COLUMN runtime_sync JSONB"))
    command.upgrade(cfg, _REV)
    assert _columns(engine) == {"runtime_sync": ("YES", "jsonb")}
