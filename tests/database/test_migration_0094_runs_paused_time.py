"""0094 runs.paused_at + runs.paused_seconds — drift-tolerant in BOTH directions, on a real throwaway DB.

upgrade head → the column exists; a DB that lost the column (drift) still downgrades
cleanly; a re-stamped DB that already has the column re-applies cleanly.
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

_REV = "0094_runs_paused_time"
_PRE = "0093_run_usage_call_id_unique"
_PROBE_DB = f"runs_paused_time_probe_{os.getpid()}"


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


def _columns(engine: Engine) -> dict[str, tuple[str, str | None]]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT column_name, is_nullable, column_default FROM information_schema.columns "
                "WHERE table_name = 'runs' AND column_name IN ('paused_at', 'paused_seconds')"
            )
        ).all()
    return {r[0]: (r[1], r[2]) for r in rows}


def test_upgrade_drifted_downgrade_and_reapply(probe) -> None:
    cfg, engine = probe
    command.upgrade(cfg, "head")
    cols = _columns(engine)
    assert cols["paused_at"][0] == "YES"
    assert cols["paused_seconds"][0] == "NO" and cols["paused_seconds"][1] == "0"

    # Drift: one column vanished out-of-band; the downgrade must still succeed.
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE runs DROP COLUMN paused_at"))
    command.downgrade(cfg, _PRE)
    assert _columns(engine) == {}

    # Drift the other way: a column present, stamp behind -> re-apply is clean.
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE runs ADD COLUMN paused_at TIMESTAMPTZ"))
    command.upgrade(cfg, "head")
    assert set(_columns(engine)) == {"paused_at", "paused_seconds"}
