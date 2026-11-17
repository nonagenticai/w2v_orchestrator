"""P7 / R3 migration 0088 (``run_usage.connection_id`` + ``connection_version``).

1. DB-free: the revision id fits ``alembic_version`` and chains onto 0087; it
   imports no application code; its CHECK text equals the model's.
2. ``alembic upgrade head`` and ``create_all`` build the same ``run_usage``.
3. On a real throwaway DATABASE seeded at 0087: the backfill copies the connection
   only from well-formed in-process metadata, never half-sets a pair, leaves
   platform rows NULL; the CHECK is live; a re-run over migrated-but-unstamped
   columns writes no row; downgrade drops the columns (rows kept); upgrade again.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from uuid import UUID

import pytest
from alembic import command
from sqlalchemy import CheckConstraint, Engine, text
from sqlalchemy.exc import IntegrityError

from src.database.models import Base
from tests.database.test_migration_0077_tenant_connections import _shape
from tests.database.test_migration_0083_run_usage_attribution import (
    RUN_WITH_SUBJECT,
    _version_num,
    _xmins,
)
from tests.database.test_migration_0083_run_usage_attribution import (
    _seed as _seed_runs_0083,
)
from tests.database.test_migration_0083_run_usage_attribution import (
    probe as probe,
)
from tests.support.schema_probe import ParityProbe

_VERSIONS = Path(__file__).resolve().parents[2] / "src/database/migrations/versions"
_REV = "0088_run_usage_connection"
_PRE = "0087_p9_partner_keys"
_NEW_COLS = ("connection_id", "connection_version")

CONN = UUID("99999999-9999-7999-8999-999999999999")
LLM_ROW = UUID("66666666-6666-7666-8666-666666666701")
TTS_ROW = UUID("66666666-6666-7666-8666-666666666702")
BAD_ID_ROW = UUID("66666666-6666-7666-8666-666666666703")
NO_VERSION_ROW = UUID("66666666-6666-7666-8666-666666666704")
PLATFORM_ROW = UUID("66666666-6666-7666-8666-666666666705")
NOT_IN_PROCESS_ROW = UUID("66666666-6666-7666-8666-666666666706")


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("m_0088", _VERSIONS / f"{_REV}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ─── DB-free ─────────────────────────────────────────────────────────────


def test_check_matches_the_model() -> None:
    m = _load()
    migration_check = next(s for s in m._CONSTRAINTS if "ck_run_usage_connection_pair" in s)
    model_check = next(
        c
        for c in Base.metadata.tables["run_usage"].constraints
        if isinstance(c, CheckConstraint) and c.name == "ck_run_usage_connection_pair"
    )
    assert f"CHECK ({model_check.sqltext})" in migration_check


def test_alembic_and_create_all_build_the_same_run_usage(parity_schemas: ParityProbe) -> None:
    probe = parity_schemas
    with probe.engine.connect() as conn:
        al = _shape(conn, probe.alembic, "run_usage")
        ca = _shape(conn, probe.create_all, "run_usage")
    assert {c[0] for c in al[0]} >= set(_NEW_COLS)
    assert al == ca, "create_all and alembic diverge on run_usage"


# ─── real database ───────────────────────────────────────────────────────


def _seed(engine: Engine) -> None:
    in_process = {"source": "orchestrator_in_process", "funding": "tenant_byo"}
    rows = [
        (LLM_ROW, {**in_process, "connection_id": str(CONN), "connection_version": 3}),
        (
            TTS_ROW,
            {
                **in_process,
                "unit": "characters",
                "connection_id": str(CONN),
                "connection_version": 12,
            },
        ),
        (BAD_ID_ROW, {**in_process, "connection_id": "not-a-uuid", "connection_version": 3}),
        # half the pair in metadata → neither column (never half-set)
        (NO_VERSION_ROW, {**in_process, "connection_id": str(CONN)}),
        (PLATFORM_ROW, None),
        # a non-in-process row that happens to carry the keys is NOT backfilled
        (
            NOT_IN_PROCESS_ROW,
            {"source": "billing", "connection_id": str(CONN), "connection_version": 1},
        ),
    ]
    with engine.begin() as conn:
        for row_id, meta in rows:
            conn.execute(
                text(
                    "INSERT INTO run_usage (id, run_id, stage_name, metadata) "
                    "VALUES (:id, :r, 's', CAST(:m AS jsonb))"
                ),
                {
                    "id": row_id,
                    "r": RUN_WITH_SUBJECT,
                    "m": json.dumps(meta) if meta is not None else None,
                },
            )


def _connections(engine: Engine) -> dict[UUID, tuple]:
    with engine.connect() as conn:
        return {
            r[0]: (r[1], r[2])
            for r in conn.execute(
                text(
                    "SELECT id, connection_id, connection_version FROM run_usage "
                    "WHERE id::text LIKE '66666666-6666-7666-8666-6666666667%'"
                )
            ).all()
        }


def test_backfill_drift_and_round_trip(probe) -> None:
    cfg, engine = probe
    command.upgrade(cfg, _PRE)
    _seed_runs_0083(engine)  # tenants/projects/deliverables/runs (+ its own usage rows)
    _seed(engine)
    command.upgrade(cfg, _REV)
    assert _version_num(engine) == _REV

    expected = {
        LLM_ROW: (CONN, 3),
        TTS_ROW: (CONN, 12),
        BAD_ID_ROW: (None, None),
        NO_VERSION_ROW: (None, None),
        PLATFORM_ROW: (None, None),
        NOT_IN_PROCESS_ROW: (None, None),
    }
    assert _connections(engine) == expected

    # The pair CHECK is live.
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            text("UPDATE run_usage SET connection_version = NULL WHERE id = :i"), {"i": LLM_ROW}
        )

    # Drift: columns present, alembic_version back at 0087 → re-apply writes no row
    # (the backfill only fills NULL pairs; the half-pair cleanup matches nothing).
    with engine.begin() as conn:
        conn.execute(text("UPDATE alembic_version SET version_num = :v"), {"v": _PRE})
    xmin_before = _xmins(engine)
    command.upgrade(cfg, _REV)
    assert _xmins(engine) == xmin_before
    assert _connections(engine) == expected

    # Drift: a half pair written while the CHECK was absent is cleared, not fatal.
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE run_usage DROP CONSTRAINT ck_run_usage_connection_pair"))
        conn.execute(
            text("UPDATE run_usage SET connection_version = 4 WHERE id = :i"),
            {"i": PLATFORM_ROW},
        )
        conn.execute(text("UPDATE alembic_version SET version_num = :v"), {"v": _PRE})
    command.upgrade(cfg, _REV)
    assert _connections(engine)[PLATFORM_ROW] == (None, None)

    command.downgrade(cfg, _PRE)
    with engine.connect() as conn:
        cols = set(
            conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = 'run_usage'"
                )
            ).scalars()
        )
        assert not cols & set(_NEW_COLS)

    command.upgrade(cfg, _REV)
    assert _connections(engine) == expected
