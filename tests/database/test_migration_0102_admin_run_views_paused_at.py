"""0102: pause reason + widened admin resolvers — on a real throwaway DB.

The probe creates the RUNTIME role ``w2v_orch_app`` itself. Without it the migration's
role-guarded GRANT is a silent no-op in CI (the role exists only in prod), so deleting
the GRANT left every test green — the exact shape of the 0043 incident, where the
missing grant was a raw 500 on ``GET /api/admin/runs`` under enforce. Here the runtime
role must be able to EXECUTE every resolver after upgrade, after downgrade and after a
re-apply, and PUBLIC must never be able to.
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

_REV = "0102_admin_run_views_paused_at"
_PRE = "0101_signup_llm_byo_required"
_PROBE_DB = f"admin_run_views_paused_probe_{os.getpid()}"
_RUNTIME_ROLE = "w2v_orch_app"
_VIEWS = ("admin_active_runs()", "admin_recent_runs(integer)")


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
        created_role = not conn.execute(
            text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": _RUNTIME_ROLE}
        ).first()
        if created_role:
            conn.execute(text(f"CREATE ROLE {_RUNTIME_ROLE} NOLOGIN"))
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
        # Roles are cluster-wide: remove only the one this probe created. Its only
        # privileges lived in the probe DB, which is already gone.
        if created_role:
            with server.connect() as conn:
                conn.execute(text(f"DROP ROLE IF EXISTS {_RUNTIME_ROLE}"))
        server.dispose()


def _privileges(engine: Engine, fn: str) -> tuple[bool, bool]:
    """(runtime role may EXECUTE, PUBLIC may EXECUTE) for one function signature."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT has_function_privilege(:role, CAST(:fn AS regprocedure), 'EXECUTE'), "
                "has_function_privilege('public', CAST(:fn AS regprocedure), 'EXECUTE')"
            ),
            {"role": _RUNTIME_ROLE, "fn": fn},
        ).one()
    return bool(row[0]), bool(row[1])


def _result(engine: Engine, fn: str) -> str:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT pg_get_function_result(CAST(:fn AS regprocedure))"), {"fn": fn}
        ).scalar_one()


def _run_columns(engine: Engine) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'runs' "
                "AND column_name = 'pause_reason'"
            )
        ).all()
    return {r[0] for r in rows}


def _definer_shape(engine: Engine, fn: str) -> tuple[bool, list[str], str, str]:
    """(SECURITY DEFINER?, proconfig, owner, the role running the migration)."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT p.prosecdef, COALESCE(p.proconfig, '{}'), p.proowner::regrole::text, "
                "current_user FROM pg_proc p WHERE p.oid = CAST(:fn AS regprocedure)"
            ),
            {"fn": fn},
        ).one()
    return bool(row[0]), list(row[1]), str(row[2]), str(row[3])


def _assert_definer(engine: Engine) -> None:
    """The resolvers must stay SECURITY DEFINER (that is what lets the RLS-scoped
    runtime role read every tenant), pin their search_path (an unpinned DEFINER
    function is an injection hole), and be owned by the migration role — in prod
    ``w2v_service``, the owner of ``runs``; here whoever ran ``alembic upgrade``."""
    for fn in _VIEWS:
        secdef, config, owner, migrator = _definer_shape(engine, fn)
        assert secdef, f"{fn} lost SECURITY DEFINER"
        assert any(c.startswith("search_path=") for c in config), (fn, config)
        assert owner == migrator, (fn, owner, migrator)


def _assert_head(engine: Engine) -> None:
    _assert_definer(engine)
    for fn in _VIEWS:
        result = _result(engine, fn)
        assert "paused_at timestamp with time zone" in result, (fn, result)
        assert "pause_reason text" in result, (fn, result)
    for fn in _VIEWS:
        runtime, public = _privileges(engine, fn)
        assert runtime, f"{_RUNTIME_ROLE} lost EXECUTE on {fn} (the 0043 incident class)"
        assert not public, f"{fn} is executable by PUBLIC"
    assert _run_columns(engine) == {"pause_reason"}


def test_upgrade_downgrade_reapply(probe) -> None:
    cfg, engine = probe
    command.upgrade(cfg, "head")
    _assert_head(engine)
    with engine.connect() as conn:
        cols = list(conn.execute(text("SELECT * FROM admin_active_runs() LIMIT 0")).keys())
    assert cols[-2:] == ["paused_at", "pause_reason"]

    command.downgrade(cfg, _PRE)
    for fn in _VIEWS:
        result = _result(engine, fn)
        assert "paused_at" not in result and "pause_reason" not in result, (fn, result)
        assert result.endswith("updated_at timestamp with time zone)"), (fn, result)
        runtime, public = _privileges(engine, fn)
        assert runtime and not public, fn
    _assert_definer(engine)
    assert _run_columns(engine) == set()

    command.upgrade(cfg, "head")
    _assert_head(engine)


def test_pause_reason_check(probe) -> None:
    cfg, engine = probe
    command.upgrade(cfg, "head")
    with engine.connect() as conn:
        definition = conn.execute(
            text(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conname = 'ck_runs_pause_reason' AND conrelid = 'runs'::regclass"
            )
        ).scalar_one()
    assert "'checkpoint'" in definition and "'manual'" in definition, definition
    assert "IS NULL" in definition, definition


def test_0092_guard_ignores_a_same_named_function_in_another_schema(probe) -> None:
    """0092 skips only when ITS schema's ``admin_active_runs()`` already has 0102's
    shape. A decoy with ``paused_at`` in another schema must not make it skip."""
    cfg, engine = probe
    command.upgrade(cfg, "0091_runs_failure_detail")
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA decoy"))
        conn.execute(
            text(
                "CREATE FUNCTION decoy.admin_active_runs() RETURNS TABLE (paused_at timestamptz) "
                "LANGUAGE sql AS $$ SELECT now() $$"
            )
        )
    command.upgrade(cfg, "0092_admin_active_runs_paused")
    with engine.connect() as conn:
        body = conn.execute(
            text(
                "SELECT prosrc FROM pg_proc WHERE oid = 'public.admin_active_runs()'::regprocedure"
            )
        ).scalar_one()
    assert "'paused'" in body, "0092 skipped because of a function in ANOTHER schema"
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA decoy CASCADE"))


def test_0092_downgrade_survives_a_restamped_database(probe) -> None:
    """A re-stamped DB (version 0092, schema at 0102) DOWNGRADES instead of aborting.

    ``CREATE OR REPLACE`` cannot change a function's result type, so replaying 0092's
    downgrade over 0102's wider ``admin_active_runs()`` fails with "cannot change
    return type of existing function" — a restore path with no way forward. The guard
    drops and recreates 0035's shape; the privileges the DROP discards must come back.
    """
    cfg, engine = probe
    command.upgrade(cfg, "head")
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE alembic_version SET version_num = '0092_admin_active_runs_paused'")
        )

    command.downgrade(cfg, "0091_runs_failure_detail")

    result = _result(engine, "admin_active_runs()")
    assert "paused_at" not in result and "pause_reason" not in result, result
    assert result.endswith("updated_at timestamp with time zone)"), result
    runtime, public = _privileges(engine, "admin_active_runs()")
    assert runtime, "the recreated function lost its w2v_orch_app grant (the 0043 incident class)"
    assert not public, "the recreated function is executable by PUBLIC"
    # The BODY, not just the signature: 0102 widens the filter to include paused
    # runs, and a downgrade that keeps that WHERE while restoring the old columns
    # would pass every assertion above.
    with engine.connect() as conn:
        body = conn.execute(
            text("SELECT prosrc FROM pg_proc WHERE oid = 'admin_active_runs()'::regprocedure")
        ).scalar_one()
    assert "'paused'" not in body, "the downgraded body still lists paused runs"


def test_backfill_classifies_rows_that_were_already_paused(probe) -> None:
    """Rows paused BEFORE 0102 carry no reason, and they were not all checkpoint
    pauses (POST /runs/{id}/pause predates 0102). Only a row showing a checkpoint's
    exact evidence — current_stage is a declared gate AND that stage completed — is
    ``checkpoint``; everything else is ``unknown``, never guessed into review."""
    cfg, engine = probe
    command.upgrade(cfg, _PRE)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO projects (id, tenant_id, owner_subject_id, visibility, type, title) "
                "VALUES (gen_random_uuid(), gen_random_uuid(), gen_random_uuid(), 'tenant', "
                "'generic', 'p') RETURNING id"
            )
        )
        pid = conn.execute(text("SELECT id FROM projects LIMIT 1")).scalar_one()
        did = conn.execute(
            text(
                "INSERT INTO deliverables (id, project_id, name, format) "
                "VALUES (gen_random_uuid(), :p, 'd', '16x9') RETURNING id"
            ),
            {"p": pid},
        ).scalar_one()

        def run(status: str, stage: str | None, gates: list[str]) -> object:
            return conn.execute(
                text(
                    "INSERT INTO runs (id, deliverable_id, blueprint_id, status, current_stage, "
                    "pause_after_stages) VALUES (gen_random_uuid(), :d, 'generic_v1', :s, :c, :g) "
                    "RETURNING id"
                ),
                {"d": did, "s": status, "c": stage, "g": gates},
            ).scalar_one()

        def stage_row(run_id: object, name: str, status: str) -> None:
            conn.execute(
                text(
                    "INSERT INTO stage_outputs (id, run_id, stage_name, status) "
                    "VALUES (gen_random_uuid(), :r, :n, :s)"
                ),
                {"r": run_id, "n": name, "s": status},
            )

        gate = run("paused", "script_draft", ["script_draft"])
        stage_row(gate, "script_draft", "completed")
        # Pressed Pause mid-stage: the current stage is not a gate.
        mid = run("paused", "narrate", ["script_draft"])
        stage_row(mid, "narrate", "running")
        # A declared gate that never completed: not proof of a checkpoint pause.
        half = run("paused", "script_draft", ["script_draft"])
        stage_row(half, "script_draft", "running")
        bare = run("paused", None, [])
        live = run("running", "script_draft", ["script_draft"])

    command.upgrade(cfg, "head")
    with engine.connect() as conn:
        got = dict(conn.execute(text("SELECT id, pause_reason FROM runs")).all())
    assert got[gate] == "checkpoint"
    assert got[mid] == "unknown"
    assert got[half] == "unknown"
    assert got[bare] == "unknown"
    assert got[live] is None, "a run that is not paused has no pause reason"
