"""P6 ORCH-3 migration 0083 (``run_usage`` attribution columns + backfill).

1. DB-free: the revision id fits ``alembic_version`` and chains onto 0082; it
   imports no application code; its CHECK text equals the model's.
2. ``alembic upgrade head`` and ``create_all`` build the same ``run_usage``.
3. On a real throwaway DATABASE seeded at 0082 with BOTH row sources (in-process
   LLM + TTS rows, billing-mirrored rows) plus the unattributable cases: the
   upgrade backfills exactly what the stored data supports and leaves the rest
   NULL; a re-run over migrated-but-unstamped columns is a no-op; downgrade drops
   the columns (rows kept); upgrade again.
"""

from __future__ import annotations

import importlib.util
import json
import os
import warnings
from collections.abc import Iterator
from decimal import Decimal
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
_REV = "0083_run_usage_attribution"
_PRE = "0082_p6_audit_actions"
_PROBE_DB = f"usage_attr_probe_{os.getpid()}"
_NEW_COLS = ("slot", "subject_id", "funding", "units", "unit_kind")

TENANT = UUID("33333333-3333-7333-8333-333333333333")
SUBJECT = UUID("44444444-4444-7444-8444-444444444444")
RUN_WITH_SUBJECT = UUID("55555555-5555-7555-8555-555555555551")
RUN_WITHOUT_SUBJECT = UUID("55555555-5555-7555-8555-555555555552")
# run_usage rows, one per case
LLM_ROW = UUID("66666666-6666-7666-8666-666666666601")
TTS_ROW = UUID("66666666-6666-7666-8666-666666666602")
BAD_SUBJECT_ROW = UUID("66666666-6666-7666-8666-666666666603")
MIRROR_ROW = UUID("66666666-6666-7666-8666-666666666604")
MIRROR_NO_TOKENS_ROW = UUID("66666666-6666-7666-8666-666666666605")
MIRROR_NO_SUBJECT_ROW = UUID("66666666-6666-7666-8666-666666666606")


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("m_0083", _VERSIONS / f"{_REV}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ─── DB-free ─────────────────────────────────────────────────────────────


def test_check_matches_the_model() -> None:
    m = _load()
    migration_check = next(s for s in m._UPGRADE if "ck_run_usage_unit_kind CHECK" in s)
    model_check = next(
        c
        for c in Base.metadata.tables["run_usage"].constraints
        if isinstance(c, CheckConstraint) and c.name == "ck_run_usage_unit_kind"
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


def _drop_probe(engine: Engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("SET statement_timeout = 30000"))
        try:
            conn.execute(text(f"DROP DATABASE IF EXISTS {_PROBE_DB} WITH (FORCE)"))
        except OperationalError as exc:
            warnings.warn(f"could not drop {_PROBE_DB}: {exc}", stacklevel=2)


@pytest.fixture
def probe() -> Iterator[tuple[Config, Engine]]:
    server = create_engine(TEST_DATABASE_URL, isolation_level="AUTOCOMMIT")
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


def _seed(engine: Engine) -> None:
    project = UUID("77777777-7777-7777-8777-777777777777")
    deliverable = UUID("88888888-8888-7888-8888-888888888888")
    in_process = {"source": "orchestrator_in_process", "funding": "tenant_byo"}
    rows = [
        # (id, run, tokens in/out, metadata)
        (
            LLM_ROW,
            RUN_WITH_SUBJECT,
            100,
            20,
            {**in_process, "slot": "llm", "subject_id": str(SUBJECT)},
        ),
        (
            TTS_ROW,
            RUN_WITH_SUBJECT,
            0,
            0,
            {
                **in_process,
                "slot": "tts",
                "subject_id": str(SUBJECT),
                "unit": "characters",
                "characters": 345,
            },
        ),
        (
            BAD_SUBJECT_ROW,
            RUN_WITH_SUBJECT,
            5,
            5,
            # a non-taxonomy funding in metadata is not copied (unknown → NULL)
            {**in_process, "funding": "platform", "slot": "llm", "subject_id": "svc-x"},
        ),
        (MIRROR_ROW, RUN_WITH_SUBJECT, 7, 3, None),
        (MIRROR_NO_TOKENS_ROW, RUN_WITH_SUBJECT, 0, 0, {"stage": "narrate"}),
        (MIRROR_NO_SUBJECT_ROW, RUN_WITHOUT_SUBJECT, 1, 1, None),
    ]
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO tenants (id) VALUES (:t)"), {"t": TENANT})
        conn.execute(
            text(
                "INSERT INTO projects (id, tenant_id, owner_subject_id, visibility, type, title) "
                "VALUES (:p, :t, :s, 'private', 'promo', 'p')"
            ),
            {"p": project, "t": TENANT, "s": SUBJECT},
        )
        conn.execute(
            text(
                "INSERT INTO deliverables (id, project_id, name, format) "
                "VALUES (:d, :p, 'd', '16x9')"
            ),
            {"d": deliverable, "p": project},
        )
        for run, snapshot in (
            (RUN_WITH_SUBJECT, {"subject_id": str(SUBJECT)}),
            (RUN_WITHOUT_SUBJECT, {}),
        ):
            conn.execute(
                text(
                    "INSERT INTO runs (id, deliverable_id, blueprint_id, config_snapshot) "
                    "VALUES (:r, :d, 'promo', CAST(:snap AS jsonb))"
                ),
                {"r": run, "d": deliverable, "snap": json.dumps(snapshot)},
            )
        for row_id, run, tin, tout, meta in rows:
            conn.execute(
                text(
                    "INSERT INTO run_usage (id, run_id, stage_name, input_tokens, output_tokens, "
                    "metadata) VALUES (:id, :r, 's', :i, :o, CAST(:m AS jsonb))"
                ),
                {
                    "id": row_id,
                    "r": run,
                    "i": tin,
                    "o": tout,
                    "m": json.dumps(meta) if meta is not None else None,
                },
            )


def _attribution(engine: Engine) -> dict[UUID, tuple]:
    with engine.connect() as conn:
        return {
            r[0]: tuple(r[1:])
            for r in conn.execute(
                text("SELECT id, slot, subject_id, funding, units, unit_kind FROM run_usage")
            ).all()
        }


def _xmins(engine: Engine) -> dict[UUID, str]:
    """Row id → ``xmin`` (the writing transaction): changes iff the row was rewritten."""
    with engine.connect() as conn:
        return {
            r[0]: r[1] for r in conn.execute(text("SELECT id, xmin::text FROM run_usage")).all()
        }


def _version_num(engine: Engine) -> str:
    with engine.connect() as conn:
        return conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()


def test_backfill_drift_and_round_trip(probe) -> None:
    cfg, engine = probe
    command.upgrade(cfg, _PRE)
    _seed(engine)
    command.upgrade(cfg, _REV)
    assert _version_num(engine) == _REV

    expected = {
        LLM_ROW: ("llm", SUBJECT, "tenant_byo", Decimal(120), "tokens"),
        TTS_ROW: ("tts", SUBJECT, "tenant_byo", Decimal(345), "chars"),
        # malformed subject → NULL (unattributed); non-taxonomy funding → NULL
        BAD_SUBJECT_ROW: ("llm", None, None, Decimal(10), "tokens"),
        # billing-mirrored: subject from the run snapshot, no slot, funding UNKNOWN
        # (billing cannot tell platform-paid from tenant BYO) → NULL, never guessed
        MIRROR_ROW: (None, SUBJECT, None, Decimal(10), "tokens"),
        MIRROR_NO_TOKENS_ROW: (None, SUBJECT, None, None, None),
        # a run whose snapshot has no subject: unattributed to a user too
        MIRROR_NO_SUBJECT_ROW: (None, None, None, Decimal(2), "tokens"),
    }
    assert _attribution(engine) == expected

    # The CHECKs are live: funding is the agreed taxonomy or NULL — the retired
    # 'platform' guess is rejected.
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(text("UPDATE run_usage SET unit_kind = 'bytes' WHERE id = :i"), {"i": LLM_ROW})
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            text("UPDATE run_usage SET funding = 'platform' WHERE id = :i"), {"i": MIRROR_ROW}
        )

    # Drift: columns present, alembic_version back at 0082 → re-apply is a no-op that
    # does not overwrite a value written since (backfill only fills NULLs).
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE run_usage SET funding = 'local_gpu' WHERE id = :i"), {"i": MIRROR_ROW}
        )
        conn.execute(text("UPDATE alembic_version SET version_num = :v"), {"v": _PRE})
    xmin_before = _xmins(engine)
    command.upgrade(cfg, _REV)
    # A no-op re-run writes NO row: every backfill UPDATE is guarded to rows it has
    # not processed yet, so a re-apply on a large table creates no dead tuples.
    assert _xmins(engine) == xmin_before
    after = _attribution(engine)
    assert after[MIRROR_ROW][2] == "local_gpu"
    assert {k: v for k, v in after.items() if k != MIRROR_ROW} == {
        k: v for k, v in expected.items() if k != MIRROR_ROW
    }

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
        assert conn.execute(text("SELECT count(*) FROM run_usage")).scalar_one() == 6

    command.upgrade(cfg, _REV)
    assert _version_num(engine) == _REV
    assert _attribution(engine)[LLM_ROW] == expected[LLM_ROW]

    # Drift from an earlier draft of 0083 that labelled mirrored rows 'platform': a
    # re-apply clears the guess to NULL (unknown) instead of failing the CHECK.
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE run_usage DROP CONSTRAINT ck_run_usage_funding"))
        conn.execute(
            text("UPDATE run_usage SET funding = 'platform' WHERE id = :i"), {"i": MIRROR_ROW}
        )
        conn.execute(text("UPDATE alembic_version SET version_num = :v"), {"v": _PRE})
    command.upgrade(cfg, _REV)
    assert _attribution(engine)[MIRROR_ROW][2] is None
