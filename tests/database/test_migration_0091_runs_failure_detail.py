"""0091 runs.failure_detail — drift-tolerant in BOTH directions, on a real throwaway DB.

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

_REV = "0091_runs_failure_detail"
_PRE = "0090_api_keys_rls"
_PROBE_DB = f"runs_failure_detail_probe_{os.getpid()}"


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


def _has_column(engine: Engine) -> bool:
    with engine.connect() as conn:
        return bool(
            conn.execute(
                text(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = 'runs' AND column_name = 'failure_detail'"
                )
            ).first()
        )


def test_upgrade_drifted_downgrade_and_reapply(probe) -> None:
    cfg, engine = probe
    command.upgrade(cfg, "head")
    assert _has_column(engine)

    # Drift: the column vanished out-of-band; the downgrade must still succeed.
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE runs DROP COLUMN failure_detail"))
    command.downgrade(cfg, _PRE)
    assert not _has_column(engine)

    # Drift the other way: column present, stamp behind → re-apply is clean.
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE runs ADD COLUMN failure_detail JSONB"))
    command.upgrade(cfg, "head")
    assert _has_column(engine)
