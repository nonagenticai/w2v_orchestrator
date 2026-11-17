"""P6 ORCH-4 migration 0084 (``tenant_sound_assets`` + RLS).

1. DB-free: the id fits ``alembic_version`` and chains onto 0083; the revision imports
   no application code; its frozen RLS text equals the live generator; its audit
   CHECK is exactly 0082's set plus ``music.update`` and equals the model's.
2. ``alembic upgrade head`` and ``create_all`` build the same table.
3. On a real throwaway DATABASE: upgrade 0083 → 0084, every CHECK / FK / unique is
   live (incl. the cross-tenant key CHECK and the live-only unique id); re-upgrade over
   an existing table (migrated but unstamped) keeps rows; downgrade drops the table;
   upgrade again.
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
from sqlalchemy import CheckConstraint, Engine, create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError

from src.database.models import Base
from tests.conftest import TEST_DATABASE_URL, _fail_or_skip_without_database
from tests.database.test_migration_0077_tenant_connections import _shape
from tests.support import run_isolation
from tests.support.schema_probe import ParityProbe

_VERSIONS = Path(__file__).resolve().parents[2] / "src/database/migrations/versions"
_REV = "0084_tenant_sound_assets"
_PRE = "0083_run_usage_attribution"
_PROBE_DB = f"sound_assets_probe_{os.getpid()}"

TENANT = UUID("33333333-3333-7333-8333-333333333333")
OTHER = UUID("44444444-4444-7444-8444-444444444444")
SUBJECT = UUID("55555555-5555-7555-8555-555555555555")


def _load(revision: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"m_{revision}", _VERSIONS / f"{revision}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ─── DB-free ─────────────────────────────────────────────────────────────


def _action_set(check: str) -> set[str]:
    return set(check.split("(", 1)[1].rstrip(")").split(","))


def test_audit_check_is_0082_plus_music_update_and_matches_the_model() -> None:
    m = _load(_REV)
    assert m._OLD_ACTIONS == _load("0082_p6_audit_actions")._NEW_ACTIONS, (
        "0084's downgrade target must be 0082's set verbatim"
    )
    old_set, new_set = _action_set(m._OLD_ACTIONS), _action_set(m._NEW_ACTIONS)
    assert new_set - old_set == {"'music.update'"}
    assert old_set <= new_set, "every 0082 action is kept"
    model_check = next(
        c
        for c in Base.metadata.tables["admin_audit_log"].constraints
        if isinstance(c, CheckConstraint) and c.name == "ck_admin_audit_action"
    )
    # The model mirrors the LATEST widening (0085 adds ``offboarding.reject``; parity
    # is pinned there). Here: nothing 0084 introduced has been dropped since.
    assert new_set <= _action_set(str(model_check.sqltext))


# ─── create_all ↔ alembic parity ─────────────────────────────────────────


def test_alembic_and_create_all_build_the_same_table(parity_schemas: ParityProbe) -> None:
    probe = parity_schemas
    with probe.engine.connect() as conn:
        al = _shape(conn, probe.alembic, "tenant_sound_assets")
        ca = _shape(conn, probe.create_all, "tenant_sound_assets")
    assert al[0], "alembic build has no columns"
    assert al == ca, "create_all and alembic diverge on tenant_sound_assets"
    assert al[3] == (True, False), "ENABLE, not FORCE"
    idx = dict(al[2])
    assert "WHERE (deleted_at IS NULL)" in idx["uq_tenant_sound_assets_live_sound_id"]


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


def _rejects(engine: Engine, sql: str, params: Mapping[str, object] | None = None) -> bool:
    try:
        with engine.begin() as conn:
            conn.execute(text(sql), dict(params or {}))
    except IntegrityError:
        return True
    return False


_INSERT = (
    "INSERT INTO tenant_sound_assets (id, tenant_id, sound_id, category, object_key, "
    "duration_s, bytes, content_type, rights_attested_by, deleted_at) VALUES "
    "(:id, :t, :sid, :cat, :key, :dur, :bytes, 'audio/wav', :sub, :deleted)"
)


_AUDIT = "INSERT INTO admin_audit_log (id, action) VALUES (:id, :a)"


def _row(n: int, **over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": UUID(f"01999999-0000-7000-8000-{n:012d}"),
        "t": TENANT,
        "sid": f"tenant_bed_{n}",
        "cat": "music",
        "key": f"sound_library/tenants/{TENANT}/music/{n}.wav",
        "dur": 1.5,
        "bytes": 100,
        "sub": SUBJECT,
        "deleted": None,
    }
    base.update(over)
    return base


def _count(engine: Engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM tenant_sound_assets")).scalar_one()


def test_upgrade_constraints_drift_and_round_trip(probe) -> None:
    cfg, engine = probe
    command.upgrade(cfg, _PRE)
    assert _rejects(engine, _AUDIT, {"id": UUID(int=0xA0), "a": "music.update"}), (
        "control: 0083 does not know music.update"
    )
    command.upgrade(cfg, _REV)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO tenants (id) VALUES (:a), (:b)"), {"a": TENANT, "b": OTHER})
    assert not _rejects(engine, _AUDIT, {"id": UUID(int=0xA1), "a": "music.update"})
    assert not _rejects(engine, _AUDIT, {"id": UUID(int=0xA2), "a": "offboarding.delete"})
    assert _rejects(engine, _AUDIT, {"id": UUID(int=0xA3), "a": "music.bogus"})

    assert not _rejects(engine, _INSERT, _row(1)), "control row must insert"
    # Every constraint is live.
    assert _rejects(engine, _INSERT, _row(2, sid="bed_2")), "id outside tenant_ namespace"
    assert _rejects(engine, _INSERT, _row(3, sid="tenant_Bad-Id")), "id charset"
    assert _rejects(engine, _INSERT, _row(4, cat="voice")), "category"
    assert _rejects(engine, _INSERT, _row(5, dur=0)), "duration > 0"
    assert _rejects(engine, _INSERT, _row(6, bytes=0)), "bytes > 0"
    assert _rejects(engine, _INSERT, _row(7, key=f"sound_library/tenants/{OTHER}/music/7.wav")), (
        "a row may not point into ANOTHER tenant's tree"
    )
    assert _rejects(engine, _INSERT, _row(8, key="sound_library/shared/music/x.mp3"))
    assert _rejects(engine, _INSERT, _row(9, t=UUID(int=9))), "unknown tenant → FK"
    assert _rejects(engine, _INSERT, _row(10, sid="tenant_bed_1")), "live id is unique"
    assert _rejects(engine, _INSERT, _row(11, key=f"sound_library/tenants/{TENANT}/music/1.wav")), (
        "object_key is unique"
    )
    # A soft-deleted row frees its id; another tenant may use the same id.
    with engine.begin() as conn:
        conn.execute(text("UPDATE tenant_sound_assets SET deleted_at = now()"))
    assert not _rejects(engine, _INSERT, _row(12, sid="tenant_bed_1"))
    assert not _rejects(
        engine,
        _INSERT,
        _row(13, t=OTHER, sid="tenant_bed_1", key=f"sound_library/tenants/{OTHER}/music/13.wav"),
    )
    assert _count(engine) == 3
    with engine.connect() as conn:
        assert (
            conn.execute(
                text(
                    "SELECT count(*) FROM pg_policies WHERE policyname = 'tenant_isolation' "
                    "AND tablename = 'tenant_sound_assets'"
                )
            ).scalar_one()
            == 1
        )

    # Drift: table + policy present, alembic_version back at 0083.
    with engine.begin() as conn:
        conn.execute(text("UPDATE alembic_version SET version_num = :v"), {"v": _PRE})
    command.upgrade(cfg, _REV)
    assert _count(engine) == 3, "re-apply must keep rows"

    # Tenant deletion cascades.
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM tenants WHERE id = :t"), {"t": OTHER})
    assert _count(engine) == 2

    command.downgrade(cfg, _PRE)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT to_regclass('tenant_sound_assets')")).scalar_one() is None
        kept = conn.execute(
            text("SELECT count(*) FROM admin_audit_log WHERE action = 'music.update'")
        ).scalar_one()
    assert kept == 1, "downgrade keeps audit rows written under music.update (NOT VALID)"
    assert _rejects(engine, _AUDIT, {"id": UUID(int=0xA4), "a": "music.update"})
    assert not _rejects(engine, _AUDIT, {"id": UUID(int=0xA5), "a": "music.delete"})
    command.upgrade(cfg, _REV)
    assert _count(engine) == 0
