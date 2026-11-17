"""Frozen-migration DDL guard — DB-free static/text invariants.

Several alembic migrations were FROZEN: they inline literal DDL instead of
importing a live generator from ``src.database.rls``, because replaying a
migration through the live generator would emit whatever the function bodies
say TODAY — so a later edit to those bodies would silently rewrite a historical
migration's replayed output, diverging a migration-based prod DB from a fresh
``create_all`` replay.

These tests keep that invariant honest without touching a database:

1. ``0030``'s frozen snapshot must still equal the live generator (it has not
   drifted since the freeze) — a mismatch means someone edited the live
   generator without reconciling the frozen historical snapshot.
2. No frozen migration may re-import the live generator (regression guard
   against accidental un-freezing).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from src.database.rls import (
    drop_run_functions_statements,
    run_functions_statements,
)
from src.database.stage_output_guard import (
    downgrade_statements,
    upgrade_statements,
)

_VERSIONS_DIR = Path(__file__).resolve().parents[2] / "src" / "database" / "migrations" / "versions"


def _load_migration(filename: str) -> ModuleType:
    """Load a migration module by file path (not a normal package import)."""
    path = _VERSIONS_DIR / filename
    spec = importlib.util.spec_from_file_location(f"m_{path.stem}", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_0030_matches_live_run_functions_generator() -> None:
    module = _load_migration("0030_rls_bypass_functions.py")

    drift_msg = (
        "migration 0030's frozen DDL drifted from "
        "src.database.rls.run_functions_statements() — someone edited the live "
        "generator without reconciling the frozen historical snapshot. If the "
        "change is intentional, it needs a NEW forward migration that ALTERs the "
        "functions, not an edit to 0030 (which would rewrite history)."
    )

    assert tuple(module._RUN_FUNCTIONS_STATEMENTS) == tuple(run_functions_statements()), drift_msg
    assert tuple(module._DROP_RUN_FUNCTIONS_STATEMENTS) == tuple(drop_run_functions_statements()), (
        drift_msg
    )


def test_frozen_migrations_do_not_reimport_the_live_generator() -> None:
    # Exact filenames for 0030/0031 are known; 0035/0054 are resolved by glob
    # because their descriptive suffix is not load-bearing here.
    frozen_globs = (
        "0030_rls_bypass_functions.py",
        "0031_rls_admin_rollup_fn.py",
        "0035_*.py",
        "0054_*.py",
    )
    forbidden = "from src.database.rls import"

    for pattern in frozen_globs:
        matches = sorted(_VERSIONS_DIR.glob(pattern))
        if not matches:
            pytest.skip(f"no frozen migration file matched {pattern!r}")
        for path in matches:
            text = path.read_text()
            assert forbidden not in text, (
                f"frozen migration {path.name} re-imports the live generator "
                f"({forbidden!r}) — it must inline literal DDL so replaying it "
                "reproduces the HISTORICAL schema, not today's function bodies. "
                "Re-freeze it by pasting the current generator output as literals."
            )


def test_frozen_0063_matches_live_stage_output_guard_generator() -> None:
    """Migration 0063 inlines a FROZEN copy of the ``stage_outputs`` write-once
    trigger DDL, for the same reason 0030 does: the SAME statements are also
    applied to a ``create_all``-built schema by
    ``src.database.stage_output_guard.register_stage_output_write_once_ddl``, and
    a later edit to the generator would otherwise rewrite what 0063 replays —
    silently giving the migration-built (prod) and create_all-built (test) schemas
    DIFFERENT triggers, with the tests proving the wrong one.
    """
    module = _load_migration("0063_stage_output_write_once.py")

    drift_msg = (
        "migration 0063's frozen DDL drifted from "
        "src.database.stage_output_guard — prod (migration chain) and tests "
        "(create_all registrar) would now build DIFFERENT write-once triggers. "
        "A guard change needs a NEW forward migration, not an edit to 0063."
    )

    assert tuple(module._UPGRADE_STATEMENTS) == tuple(upgrade_statements()), drift_msg
    assert tuple(module._DOWNGRADE_STATEMENTS) == tuple(downgrade_statements()), drift_msg


# --- The 12 revisions frozen 2026-10-05 -------------------------------------
#
# These used to ``from src.database.rls import ...`` and render their RLS DDL from
# the LIVE generators, so any later edit to rls.py silently rewrote what a fresh
# bootstrap replays through these historical revisions. They now inline the exact
# statements the generators emitted at freeze time (verified byte-identical via
# per-revision ``alembic upgrade/downgrade <a>:<b> --sql`` before vs after).
#
# Each entry: (revision file glob, migration constant, live generator name).
_FROZEN_2026_10_05 = [
    ("0027_*.py", "_UPGRADE_STATEMENTS", "upgrade_statements"),
    ("0027_*.py", "_DOWNGRADE_STATEMENTS", "downgrade_statements"),
    ("0028_*.py", "_JOIN_UPGRADE_STATEMENTS", "join_upgrade_statements"),
    ("0028_*.py", "_JOIN_DOWNGRADE_STATEMENTS", "join_downgrade_statements"),
    ("0029_*.py", "_REBUILD_POLICIES_STATEMENTS", "rebuild_policies_statements"),
    ("0043_*.py", "_GRANT_ADMIN_RUN_VIEWS_STATEMENTS", "grant_admin_run_views_statements"),
    ("0043_*.py", "_REVOKE_ADMIN_RUN_VIEWS_STATEMENTS", "revoke_admin_run_views_statements"),
    ("0046_*.py", "_LATE_JOIN_UPGRADE_STATEMENTS", "late_join_upgrade_statements"),
    ("0046_*.py", "_LATE_JOIN_DOWNGRADE_STATEMENTS", "late_join_downgrade_statements"),
    ("0049_*.py", "_LATE_JOIN_EVENTS_UPGRADE_STATEMENTS", "late_join_events_upgrade_statements"),
    (
        "0049_*.py",
        "_LATE_JOIN_EVENTS_DOWNGRADE_STATEMENTS",
        "late_join_events_downgrade_statements",
    ),
    (
        "0050_*.py",
        "_LATE_JOIN_SNAPSHOTS_UPGRADE_STATEMENTS",
        "late_join_snapshots_upgrade_statements",
    ),
    (
        "0050_*.py",
        "_LATE_JOIN_SNAPSHOTS_DOWNGRADE_STATEMENTS",
        "late_join_snapshots_downgrade_statements",
    ),
    (
        "0051_*.py",
        "_LATE_JOIN_PANEL_ASSETS_UPGRADE_STATEMENTS",
        "late_join_panel_assets_upgrade_statements",
    ),
    (
        "0051_*.py",
        "_LATE_JOIN_PANEL_ASSETS_DOWNGRADE_STATEMENTS",
        "late_join_panel_assets_downgrade_statements",
    ),
    (
        "0052_*.py",
        "_LATE_JOIN_BRANCHES_UPGRADE_STATEMENTS",
        "late_join_branches_upgrade_statements",
    ),
    (
        "0052_*.py",
        "_LATE_JOIN_BRANCHES_DOWNGRADE_STATEMENTS",
        "late_join_branches_downgrade_statements",
    ),
    (
        "0060_*.py",
        "_LATE_JOIN_VIDEO_SOURCES_UPGRADE_STATEMENTS",
        "late_join_video_sources_upgrade_statements",
    ),
    (
        "0060_*.py",
        "_LATE_JOIN_VIDEO_SOURCES_DOWNGRADE_STATEMENTS",
        "late_join_video_sources_downgrade_statements",
    ),
    (
        "0062_*.py",
        "_LATE_JOIN_VIDEO_SOURCE_UPLOADS_UPGRADE_STATEMENTS",
        "late_join_video_source_uploads_upgrade_statements",
    ),
    (
        "0062_*.py",
        "_LATE_JOIN_VIDEO_SOURCE_UPLOADS_DOWNGRADE_STATEMENTS",
        "late_join_video_source_uploads_downgrade_statements",
    ),
]


def _load_by_glob(pattern: str) -> ModuleType:
    matches = sorted(_VERSIONS_DIR.glob(pattern))
    assert len(matches) == 1, f"expected exactly one revision for {pattern!r}, got {matches}"
    return _load_migration(matches[0].name)


@pytest.mark.parametrize(
    ("pattern", "const", "generator"),
    _FROZEN_2026_10_05,
    ids=[f"{p[:4]}-{g}" for p, _, g in _FROZEN_2026_10_05],
)
def test_frozen_2026_10_05_snapshot_matches_live_generator(
    pattern: str, const: str, generator: str
) -> None:
    from src.database import rls

    snapshot = tuple(getattr(_load_by_glob(pattern), const))
    live = tuple(getattr(rls, generator)())
    assert snapshot, f"{pattern}:{const} is empty — a vacuous snapshot proves nothing"
    assert snapshot == live, (
        f"{pattern}:{const} drifted from src.database.rls.{generator}(). The live "
        "generator changed after this revision was frozen. That change needs a NEW "
        "forward migration; do NOT edit the historical snapshot (it would rewrite what "
        "a fresh bootstrap replays) and do NOT re-import the generator."
    )


def test_frozen_0045_snapshot_matches_live_helpers() -> None:
    from src.database import rls

    module = _load_by_glob("0045_*.py")
    table = module._TABLE
    assert table == "source_ingest_cache"
    assert rls.enable_sql(table) == module._ENABLE_SQL
    assert rls.policy_sql(table) == module._POLICY_SQL
    assert rls.drop_policy_sql(table) == module._DROP_POLICY_SQL
    assert rls.disable_sql(table) == module._DISABLE_SQL


def test_frozen_0029_downgrade_restores_the_legacy_guc() -> None:
    """0029's downgrade has no live generator to compare against (rls.py only
    knows the hardened NULLIF shape), so pin its shape instead: every policy it
    re-creates uses the pre-NULLIF GUC and none uses NULLIF."""
    module = _load_by_glob("0029_*.py")
    creates = [s for s in module._LEGACY_POLICY_STATEMENTS if s.startswith("CREATE POLICY")]
    drops = [s for s in module._LEGACY_POLICY_STATEMENTS if s.startswith("DROP POLICY")]
    assert len(creates) == len(drops) == len(module._REBUILD_POLICIES_STATEMENTS) // 2
    for stmt in creates:
        assert "NULLIF" not in stmt, stmt
        assert "current_setting('app.current_tenant', true)::uuid" in stmt, stmt


def test_no_revision_imports_the_live_rls_generator() -> None:
    """Repo-wide: NO alembic revision may render DDL from live ``src.database.rls``.

    Checked on the AST (not a substring) so a docstring that merely mentions the
    module does not satisfy or trip the guard.
    """
    import ast

    offenders: list[str] = []
    for path in sorted(_VERSIONS_DIR.glob("[0-9]*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "src.database.rls"
            ):
                offenders.append(path.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "src.database":
                if any(a.name == "rls" for a in node.names):
                    offenders.append(path.name)
            elif isinstance(node, ast.Import) and any(
                a.name.startswith("src.database.rls") for a in node.names
            ):
                offenders.append(path.name)
    assert not offenders, (
        f"revisions import the LIVE rls generator: {offenders}. Inline the rendered "
        "statements as a frozen tuple instead (see the 2026-10-05 block above)."
    )


def test_frozen_0076_matches_live_tenant_occupancy_generator() -> None:
    """0076's frozen ``tenant_is_occupied`` DDL equals the live generator AND
    keeps the SECURITY DEFINER + PUBLIC-revoke + pinned-search_path posture.
    Without SECURITY DEFINER the partner occupancy check is RLS-blind under
    enforce (answers "empty" for a tenant full of data) — the takeover it exists
    to stop; without a pinned path ending in ``pg_temp`` a caller's temp table
    could shadow ``projects`` inside the definer's privileges."""
    from src.database.rls import (
        drop_tenant_occupancy_functions_statements,
        tenant_occupancy_functions_statements,
    )

    module = _load_migration("0076_tenant_is_occupied_fn.py")
    live = tuple(tenant_occupancy_functions_statements())
    assert tuple(module._UPGRADE_STATEMENTS) == live
    assert tuple(module._DOWNGRADE_STATEMENTS) == tuple(
        drop_tenant_occupancy_functions_statements()
    )
    assert "SECURITY DEFINER" in live[0] and "SECURITY INVOKER" not in live[0]
    assert "FROM CURRENT" not in live[0]
    assert (
        "SET search_path = pg_catalog, ' || quote_ident(current_schema()) || ', pg_temp " in live[0]
    )
    for table in ("projects", "tenant_members", "tenant_invites", "tenant_api_keys"):
        assert f"FROM {table} WHERE tenant_id = p_tenant" in live[0], table
    assert live[1] == "REVOKE EXECUTE ON FUNCTION tenant_is_occupied(uuid) FROM PUBLIC"
    assert "GRANT EXECUTE ON FUNCTION tenant_is_occupied(uuid) TO w2v_orch_app" in live[2]
    assert (
        "from src.database.rls import"
        not in (_VERSIONS_DIR / "0076_tenant_is_occupied_fn.py").read_text()
    )


def test_frozen_0079_matches_live_provider_secret_gc_generator() -> None:
    """0079's frozen GC function equals the live generator AND keeps SECURITY
    DEFINER (under the restricted runtime role a plain scan sees no runs → no pins →
    it would purge secrets paused runs still need), the PUBLIC revoke, and a pinned
    search_path ending in ``pg_temp``."""
    from src.database.rls import (
        drop_provider_secret_gc_statements,
        provider_secret_gc_statements,
    )

    module = _load_migration("0079_provider_secret_gc_fn.py")
    live = tuple(provider_secret_gc_statements())
    assert tuple(module._UPGRADE_STATEMENTS) == live
    assert tuple(module._DOWNGRADE_STATEMENTS) == tuple(drop_provider_secret_gc_statements())
    assert "SECURITY DEFINER" in live[0] and "FROM CURRENT" not in live[0]
    assert "quote_ident(current_schema()) || $q$, pg_temp " in live[0]
    assert "NOT IN ('completed', 'failed', 'cancelled')" in live[0]
    assert live[1].startswith("REVOKE EXECUTE ON FUNCTION purge_unpinned_provider_secrets")
    assert "TO w2v_orch_app" in live[2]
    assert (
        "from src.database.rls import"
        not in (_VERSIONS_DIR / "0079_provider_secret_gc_fn.py").read_text()
    )
