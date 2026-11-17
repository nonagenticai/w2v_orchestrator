"""Postgres Row-Level-Security (RLS) — Phase D tenant-isolation backstop.

Single source of truth for the RLS DDL so the two apply paths can never drift:
  * the Alembic migrations (``0027_rls_projects.py`` → ``projects``;
    ``0028_rls_runtree.py`` → the run-tree + ``verbatim_items`` tables) — how
    PROD existing tables get RLS, and
  * a SQLAlchemy ``after_create`` event — how a freshly-``create_all``-ed schema
    (tests + dev) gets the SAME policies.

RLS is ENABLEd but NOT FORCEd, so the orchestrator's own DB user (the table
owner) keeps bypassing it: every policy here is **INERT for the app today** and
changes no behaviour. They only take effect for a future non-owner,
no-``BYPASSRLS`` role (a later Phase D increment), at which point
``SET LOCAL app.current_tenant`` (see ``apply_tenant_rls`` in
``api/routes/_common.py``, gated by ``TENANT_RLS_MODE``) scopes every query.

Two policy shapes, by how a row reaches its tenant:
  * ``projects`` carries ``tenant_id`` directly → a simple equality policy.
  * the run-tree + ``verbatim_items`` tables carry no ``tenant_id``; they reach
    it via an FK chain to ``projects`` → a **join-based** ``EXISTS`` policy that
    climbs to ``projects.tenant_id``. This keeps the increment purely additive
    (no schema change, no backfill, no write-path change). Denormalizing
    ``tenant_id`` onto these tables stays a deferred, perf-driven option.

Deferred to later Phase D increments (documented, not built here):
  * ``FORCE ROW LEVEL SECURITY`` + the no-``BYPASSRLS`` prod DB role + the staged
    rollout that flips ``TENANT_RLS_MODE=enforce`` (and the background-session
    GUC the non-request sessions would then need).

The GUC read uses ``current_setting('app.current_tenant', true)`` (missing-ok),
so an unset GUC yields NULL rather than erroring.
"""

from __future__ import annotations

from sqlalchemy import DDL, MetaData, event

POLICY_NAME = "tenant_isolation"

# The transaction-scoped GUC ``apply_tenant_rls`` binds per request. Factored
# once so the direct and join predicates read the exact same setting.
#
# ``NULLIF(…, '')`` is load-bearing, not cosmetic: a custom placeholder GUC set
# with ``is_local=true`` reverts to the **empty string** (NOT NULL) once its
# transaction ends, so a pooled/PgBouncer connection that bound the GUC on a
# prior transaction reports ``''`` on the next un-bound one. Casting ``''::uuid``
# raises (a 500), defeating fail-closed; ``NULLIF`` folds ``''`` back to NULL so
# the predicate is simply false → 0 rows. ``current_setting(…, true)`` (missing_ok)
# covers the never-set case (NULL) the same way.
_GUC = "NULLIF(current_setting('app.current_tenant', true), '')::uuid"

# Tables that carry ``tenant_id`` directly and get a simple equality policy
# (shipped in migration 0027).
RLS_POLICY_TABLES: tuple[str, ...] = ("projects",)

# Direct-tenant tables added AFTER the 0027/0029 RLS increments, each shipped by
# its own later migration. Kept SEPARATE from ``RLS_POLICY_TABLES`` on purpose:
# migrations 0027 (``upgrade_statements``) and 0029 (``rebuild_policies_statements``
# → ``_all_policy_specs``) iterate ``RLS_POLICY_TABLES`` at run time, so adding a
# table there would make those HISTORICAL migrations emit ENABLE/POLICY DDL for a
# table a fresh ``alembic upgrade head`` has not created yet (they run before this
# table's migration) → a hard failure. Each table here applies its own
# ENABLE+POLICY inline in its own migration and is registered for the
# ``create_all`` (test/dev) path via ``register_late_rls_ddl``.
RLS_POLICY_TABLES_LATE: tuple[str, ...] = ("source_ingest_cache",)

_DIRECT_PREDICATE = f"tenant_id = {_GUC}"

# Tables that reach tenant only via an FK chain to ``projects`` → a join-based
# ``EXISTS`` policy climbing to ``projects.tenant_id`` (shipped in migration
# 0028). Each predicate references its own table by name (the policy is ON that
# table) and is correlated to the row under test.


def _exists_predicate(joins: str, where: str) -> str:
    # Static DDL assembled from hard-coded table/column names + the fixed GUC —
    # no user input ever reaches it, so the SQL-injection heuristic is a false
    # positive (same posture as the hard-coded table list in migration 0004).
    return f"EXISTS (SELECT 1 FROM {joins} WHERE {where} AND p.tenant_id = {_GUC})"  # noqa: S608


RLS_JOIN_PREDICATES: dict[str, str] = {
    # 1-hop: directly own a project_id.
    "deliverables": _exists_predicate("projects p", "p.id = deliverables.project_id"),
    "verbatim_items": _exists_predicate("projects p", "p.id = verbatim_items.project_id"),
    # 2-hop: run → deliverable → project.
    "runs": _exists_predicate(
        "deliverables d JOIN projects p ON p.id = d.project_id",
        "d.id = runs.deliverable_id",
    ),
    # 3-hop: → run → deliverable → project.
    "stage_outputs": _exists_predicate(
        "runs r JOIN deliverables d ON d.id = r.deliverable_id "
        "JOIN projects p ON p.id = d.project_id",
        "r.id = stage_outputs.run_id",
    ),
    "run_usage": _exists_predicate(
        "runs r JOIN deliverables d ON d.id = r.deliverable_id "
        "JOIN projects p ON p.id = d.project_id",
        "r.id = run_usage.run_id",
    ),
    # 4-hop: → stage_output → run → deliverable → project.
    "annotations": _exists_predicate(
        "stage_outputs so JOIN runs r ON r.id = so.run_id "
        "JOIN deliverables d ON d.id = r.deliverable_id "
        "JOIN projects p ON p.id = d.project_id",
        "so.id = annotations.stage_output_id",
    ),
}

RLS_JOIN_TABLES: tuple[str, ...] = tuple(RLS_JOIN_PREDICATES)

# Join-based tables added AFTER the 0028 RLS increment (storyboard Phase 0). Kept
# SEPARATE from ``RLS_JOIN_PREDICATES`` for the SAME reason ``RLS_POLICY_TABLES_LATE``
# is separate from ``RLS_POLICY_TABLES``: historical migration 0028 iterates
# ``RLS_JOIN_PREDICATES`` at run time (``join_upgrade_statements``) and runs BEFORE
# these tables' migration (0046), so adding them there would emit ENABLE/POLICY DDL
# for not-yet-created tables on a fresh ``alembic upgrade head`` → hard failure. Each
# applies its own ENABLE+POLICY inline in migration 0046 (``late_join_upgrade_statements``)
# and is registered for the ``create_all`` (test/dev) path via ``register_late_join_rls_ddl``.
# All 4 are 1-hop: they carry ``project_id`` directly → climb to ``projects.tenant_id``.
RLS_JOIN_PREDICATES_LATE: dict[str, str] = {
    "storyboards": _exists_predicate("projects p", "p.id = storyboards.project_id"),
    "storyboard_scenes": _exists_predicate("projects p", "p.id = storyboard_scenes.project_id"),
    "storyboard_panels": _exists_predicate("projects p", "p.id = storyboard_panels.project_id"),
    "storyboard_assets": _exists_predicate("projects p", "p.id = storyboard_assets.project_id"),
}


def enable_sql(table: str) -> str:
    # ENABLE (not FORCE): the table owner still bypasses, so this is inert for the
    # app until a restricted role + FORCE land in a later increment.
    return f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"


def policy_sql(table: str, predicate: str = _DIRECT_PREDICATE) -> str:
    return f"CREATE POLICY {POLICY_NAME} ON {table} USING ({predicate}) WITH CHECK ({predicate})"


def drop_policy_sql(table: str) -> str:
    return f"DROP POLICY IF EXISTS {POLICY_NAME} ON {table}"


def disable_sql(table: str) -> str:
    return f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY"


def _enable_and_policy(table: str, predicate: str) -> list[str]:
    return [enable_sql(table), policy_sql(table, predicate)]


def _drop_and_disable(table: str) -> list[str]:
    return [drop_policy_sql(table), disable_sql(table)]


def upgrade_statements() -> list[str]:
    """DDL for ``0027_rls_projects``'s ``upgrade()`` — the direct-tenant tables.

    Intentionally scoped to ``RLS_POLICY_TABLES`` (``projects``) and unchanged:
    0028 applies the join tables separately, so a fresh ``alembic upgrade head``
    never double-applies a table's policy.
    """
    out: list[str] = []
    for table in RLS_POLICY_TABLES:
        out += _enable_and_policy(table, _DIRECT_PREDICATE)
    return out


def downgrade_statements() -> list[str]:
    """Reverse of :func:`upgrade_statements` (fully reversible)."""
    out: list[str] = []
    for table in RLS_POLICY_TABLES:
        out += _drop_and_disable(table)
    return out


def join_upgrade_statements() -> list[str]:
    """DDL for ``0028_rls_runtree``'s ``upgrade()`` — the join-based tables."""
    out: list[str] = []
    for table, predicate in RLS_JOIN_PREDICATES.items():
        out += _enable_and_policy(table, predicate)
    return out


def join_downgrade_statements() -> list[str]:
    """Reverse of :func:`join_upgrade_statements` (fully reversible)."""
    out: list[str] = []
    for table in RLS_JOIN_PREDICATES:
        out += _drop_and_disable(table)
    return out


def late_join_upgrade_statements() -> list[str]:
    """DDL for a late join table's OWN migration (0046) — ENABLE + join policy.

    Emitted inline in 0046 (NOT in 0028), because these tables are absent when 0028
    runs on a fresh ``alembic upgrade head`` — see the ``RLS_JOIN_PREDICATES_LATE``
    comment.
    """
    out: list[str] = []
    for table, predicate in RLS_JOIN_PREDICATES_LATE.items():
        out += _enable_and_policy(table, predicate)
    return out


def late_join_downgrade_statements() -> list[str]:
    """Reverse of :func:`late_join_upgrade_statements` (fully reversible)."""
    out: list[str] = []
    for table in RLS_JOIN_PREDICATES_LATE:
        out += _drop_and_disable(table)
    return out


def _all_policy_specs() -> list[tuple[str, str]]:
    """Every RLS table paired with its CURRENT predicate (direct + join)."""
    specs: list[tuple[str, str]] = [(name, _DIRECT_PREDICATE) for name in RLS_POLICY_TABLES]
    specs += list(RLS_JOIN_PREDICATES.items())
    return specs


def rebuild_policies_statements() -> list[str]:
    """DROP + re-CREATE the ``tenant_isolation`` policy on every RLS table with
    the CURRENT predicate, leaving ENABLE/FORCE state untouched.

    Used by a migration to roll a predicate change (e.g. the ``NULLIF`` fail-closed
    hardening) onto existing DBs whose policies were created by 0027/0028 — those
    tables are already ENABLEd, so re-CREATE alone would collide; drop first.
    """
    out: list[str] = []
    for table, predicate in _all_policy_specs():
        out += [drop_policy_sql(table), policy_sql(table, predicate)]
    return out


def register_late_rls_ddl(metadata: MetaData) -> None:
    """Attach ``after_create`` DDL for the direct-tenant tables added after 0027
    (``RLS_POLICY_TABLES_LATE``) so a ``create_all``-ed schema (tests/dev) carries
    the SAME ENABLE+POLICY their own migration applies in prod.

    Separate from :func:`register_rls_ddl` only so the historical migrations that
    iterate ``RLS_POLICY_TABLES`` keep emitting exactly ``projects`` (see the
    ``RLS_POLICY_TABLES_LATE`` comment). The predicate is the plain direct-tenant
    equality — these tables carry ``tenant_id`` on the row, no ancestor join, so
    there is no FK-ordering concern in ``create_all``.
    """
    for name in RLS_POLICY_TABLES_LATE:
        table = metadata.tables.get(name)
        if table is None:
            continue
        event.listen(
            table, "after_create", DDL(f"{enable_sql(name)}; {policy_sql(name, _DIRECT_PREDICATE)}")
        )


def register_late_join_rls_ddl(metadata: MetaData) -> None:
    """Attach ``after_create`` ENABLE+POLICY for the late JOIN tables
    (``RLS_JOIN_PREDICATES_LATE``, storyboard Phase 0) so a ``create_all``-ed schema
    (tests/dev) carries the SAME RLS migration 0046 applies in prod.

    Separate from :func:`register_rls_ddl` so historical migration 0028 keeps
    iterating exactly its own ``RLS_JOIN_PREDICATES`` set (see the
    ``RLS_JOIN_PREDICATES_LATE`` comment). Each predicate references ``projects`` (an
    ancestor via the ``project_id`` FK); ``create_all`` orders by FK dependency, so
    ``projects`` exists before these tables' ``after_create`` fires.
    """
    for name, predicate in RLS_JOIN_PREDICATES_LATE.items():
        table = metadata.tables.get(name)
        if table is None:
            continue
        event.listen(
            table, "after_create", DDL(f"{enable_sql(name)}; {policy_sql(name, predicate)}")
        )


# Join-based table added AFTER the 0046 storyboard increment: the append-only
# ``storyboard_panel_events`` approval trail. Kept SEPARATE from
# ``RLS_JOIN_PREDICATES_LATE`` for the SAME trap reason those are separate from
# ``RLS_JOIN_PREDICATES``: historical migration 0046 iterates
# ``RLS_JOIN_PREDICATES_LATE`` at run time (``late_join_upgrade_statements``) and
# runs BEFORE this table's migration (0049), so adding it there would emit
# ENABLE/POLICY DDL for a not-yet-created table on a fresh ``alembic upgrade head``
# → hard failure. It applies its own ENABLE+POLICY inline in migration 0049
# (``late_join_events_upgrade_statements``) and is registered for the ``create_all``
# (test/dev) path via ``register_late_join_events_rls_ddl``. 1-hop: carries
# ``project_id`` directly → climbs to ``projects.tenant_id``.
RLS_JOIN_PREDICATES_LATE_EVENTS: dict[str, str] = {
    "storyboard_panel_events": _exists_predicate(
        "projects p", "p.id = storyboard_panel_events.project_id"
    ),
}


def late_join_events_upgrade_statements() -> list[str]:
    """DDL for migration 0049's ``upgrade()`` — ENABLE + join policy for the trail."""
    out: list[str] = []
    for t, pred in RLS_JOIN_PREDICATES_LATE_EVENTS.items():
        out += _enable_and_policy(t, pred)
    return out


def late_join_events_downgrade_statements() -> list[str]:
    """Reverse of :func:`late_join_events_upgrade_statements` (fully reversible)."""
    out: list[str] = []
    for t in RLS_JOIN_PREDICATES_LATE_EVENTS:
        out += _drop_and_disable(t)
    return out


def register_late_join_events_rls_ddl(metadata: MetaData) -> None:
    """Attach ``after_create`` ENABLE+POLICY for ``storyboard_panel_events`` so a
    ``create_all``-ed schema (tests/dev) carries the SAME RLS migration 0049 applies
    in prod. Separate registration so historical migration 0046 keeps iterating
    exactly its own ``RLS_JOIN_PREDICATES_LATE`` set (see the
    ``RLS_JOIN_PREDICATES_LATE_EVENTS`` comment). References ``projects`` (an ancestor
    via the ``project_id`` FK); ``create_all`` orders by FK dependency, so ``projects``
    exists before this table's ``after_create`` fires."""
    for name, predicate in RLS_JOIN_PREDICATES_LATE_EVENTS.items():
        table = metadata.tables.get(name)
        if table is None:
            continue
        event.listen(
            table, "after_create", DDL(f"{enable_sql(name)}; {policy_sql(name, predicate)}")
        )


# Join-based table added AFTER the 0049 events increment: the storyboard
# version-snapshot table ``storyboard_snapshots``. Kept SEPARATE from
# ``RLS_JOIN_PREDICATES_LATE`` / ``_LATE_EVENTS`` for the SAME trap reason those are
# separate from ``RLS_JOIN_PREDICATES``: historical migrations 0046/0049 iterate
# THEIR OWN late sets at run time and run BEFORE this table's migration (0050), so
# reusing an existing map would emit ENABLE/POLICY DDL for a not-yet-created table
# on a fresh ``alembic upgrade head`` → hard failure. It applies its own
# ENABLE+POLICY inline in migration 0050 (``late_join_snapshots_upgrade_statements``)
# and is registered for the ``create_all`` (test/dev) path via
# ``register_late_join_snapshots_rls_ddl``. 1-hop: carries ``project_id`` directly →
# climbs to ``projects.tenant_id``.
RLS_JOIN_PREDICATES_SNAPSHOTS: dict[str, str] = {
    "storyboard_snapshots": _exists_predicate(
        "projects p", "p.id = storyboard_snapshots.project_id"
    ),
}


def late_join_snapshots_upgrade_statements() -> list[str]:
    """DDL for migration 0050's ``upgrade()`` — ENABLE + join policy for snapshots."""
    out: list[str] = []
    for t, pred in RLS_JOIN_PREDICATES_SNAPSHOTS.items():
        out += _enable_and_policy(t, pred)
    return out


def late_join_snapshots_downgrade_statements() -> list[str]:
    """Reverse of :func:`late_join_snapshots_upgrade_statements` (fully reversible)."""
    out: list[str] = []
    for t in RLS_JOIN_PREDICATES_SNAPSHOTS:
        out += _drop_and_disable(t)
    return out


def register_late_join_snapshots_rls_ddl(metadata: MetaData) -> None:
    """Attach ``after_create`` ENABLE+POLICY for ``storyboard_snapshots`` so a
    ``create_all``-ed schema (tests/dev) carries the SAME RLS migration 0050 applies
    in prod. Separate registration so historical migrations 0046/0049 keep iterating
    exactly their own late sets (see the ``RLS_JOIN_PREDICATES_SNAPSHOTS`` comment).
    References ``projects`` (an ancestor via the ``project_id`` FK); ``create_all``
    orders by FK dependency, so ``projects`` exists before this table's
    ``after_create`` fires."""
    for name, predicate in RLS_JOIN_PREDICATES_SNAPSHOTS.items():
        table = metadata.tables.get(name)
        if table is None:
            continue
        event.listen(
            table, "after_create", DDL(f"{enable_sql(name)}; {policy_sql(name, predicate)}")
        )


# Join-based table added AFTER the 0050 snapshots increment: the panel→asset join
# table ``storyboard_panel_assets`` (migration 0051). Kept SEPARATE from every other
# late map for the SAME trap reason they are separate from ``RLS_JOIN_PREDICATES``:
# historical migrations 0046/0049/0050 iterate THEIR OWN late maps at run time and
# run BEFORE 0051, so reusing one of them would emit ENABLE/POLICY DDL for a table
# that does not exist yet on a fresh ``alembic upgrade head`` → hard failure. It
# applies its own ENABLE+POLICY inline in migration 0051
# (``late_join_panel_assets_upgrade_statements``) and is registered for the
# ``create_all`` (test/dev) path via ``register_late_join_panel_assets_rls_ddl``.
#
# 2-hop (the first storyboard table that is not 1-hop): the join row carries neither
# ``tenant_id`` nor ``project_id``, so it climbs ``panel_id`` →
# ``storyboard_panels.project_id`` → ``projects.tenant_id``.
RLS_JOIN_PREDICATES_PANEL_ASSETS: dict[str, str] = {
    "storyboard_panel_assets": _exists_predicate(
        "storyboard_panels sp JOIN projects p ON p.id = sp.project_id",
        "sp.id = storyboard_panel_assets.panel_id",
    ),
}


def late_join_panel_assets_upgrade_statements() -> list[str]:
    """DDL for migration 0051's ``upgrade()`` — ENABLE + join policy for the join table."""
    out: list[str] = []
    for t, pred in RLS_JOIN_PREDICATES_PANEL_ASSETS.items():
        out += _enable_and_policy(t, pred)
    return out


def late_join_panel_assets_downgrade_statements() -> list[str]:
    """Reverse of :func:`late_join_panel_assets_upgrade_statements` (fully reversible)."""
    out: list[str] = []
    for t in RLS_JOIN_PREDICATES_PANEL_ASSETS:
        out += _drop_and_disable(t)
    return out


def register_late_join_panel_assets_rls_ddl(metadata: MetaData) -> None:
    """Attach ``after_create`` ENABLE+POLICY for ``storyboard_panel_assets`` so a
    ``create_all``-ed schema (tests/dev) carries the SAME RLS migration 0051 applies
    in prod. Separate registration so historical migrations 0046/0049/0050 keep
    iterating exactly their own late maps (see the
    ``RLS_JOIN_PREDICATES_PANEL_ASSETS`` comment). The predicate references
    ``storyboard_panels`` and ``projects`` — both ancestors via this table's FKs, and
    ``create_all`` orders by FK dependency, so both exist before this table's
    ``after_create`` fires."""
    for name, predicate in RLS_JOIN_PREDICATES_PANEL_ASSETS.items():
        table = metadata.tables.get(name)
        if table is None:
            continue
        event.listen(
            table, "after_create", DDL(f"{enable_sql(name)}; {policy_sql(name, predicate)}")
        )


# Join-based table added AFTER the 0051 panel-assets increment: the branch↔root
# link table ``storyboard_branches`` (migration 0052). Kept SEPARATE from every other
# late map for the SAME trap reason they are separate from ``RLS_JOIN_PREDICATES``:
# historical migrations 0046/0049/0050/0051 iterate THEIR OWN late maps at run time
# and run BEFORE 0052, so reusing one would emit ENABLE/POLICY DDL for a table that
# does not exist yet on a fresh ``alembic upgrade head`` → hard failure. It applies
# its own ENABLE+POLICY inline in migration 0052
# (``late_join_branches_upgrade_statements``) and is registered for the ``create_all``
# (test/dev) path via ``register_late_join_branches_rls_ddl``.
#
# 1-hop: the row carries ``root_project_id`` (an FK to ``projects``), so it climbs
# ``root_project_id -> projects.tenant_id``. The child project inherits the root's
# tenant at fork, so ``child_project_id`` would climb to the same tenant; ``root_`` is
# chosen as the stable, user-visible side of the link.
RLS_JOIN_PREDICATES_BRANCHES: dict[str, str] = {
    "storyboard_branches": _exists_predicate(
        "projects p", "p.id = storyboard_branches.root_project_id"
    ),
}


def late_join_branches_upgrade_statements() -> list[str]:
    """DDL for migration 0052's ``upgrade()`` — ENABLE + join policy for branches."""
    out: list[str] = []
    for t, pred in RLS_JOIN_PREDICATES_BRANCHES.items():
        out += _enable_and_policy(t, pred)
    return out


def late_join_branches_downgrade_statements() -> list[str]:
    """Reverse of :func:`late_join_branches_upgrade_statements` (fully reversible)."""
    out: list[str] = []
    for t in RLS_JOIN_PREDICATES_BRANCHES:
        out += _drop_and_disable(t)
    return out


def register_late_join_branches_rls_ddl(metadata: MetaData) -> None:
    """Attach ``after_create`` ENABLE+POLICY for ``storyboard_branches`` so a
    ``create_all``-ed schema (tests/dev) carries the SAME RLS migration 0052 applies
    in prod. Separate registration so historical migrations 0046/0049/0050/0051 keep
    iterating exactly their own late maps (see the ``RLS_JOIN_PREDICATES_BRANCHES``
    comment). References ``projects`` (an ancestor via the ``root_project_id`` FK);
    ``create_all`` orders by FK dependency, so ``projects`` exists before this table's
    ``after_create`` fires."""
    for name, predicate in RLS_JOIN_PREDICATES_BRANCHES.items():
        table = metadata.tables.get(name)
        if table is None:
            continue
        event.listen(
            table, "after_create", DDL(f"{enable_sql(name)}; {policy_sql(name, predicate)}")
        )


# Join-based table added AFTER the 0052 branches increment: the uploaded-footage
# table ``video_sources`` (migration 0060, the ``action_reel`` input side). Kept
# SEPARATE from every other late map for the SAME trap reason they are separate from
# ``RLS_JOIN_PREDICATES``: historical migrations 0046/0049/0050/0051/0052 iterate
# THEIR OWN late maps at run time and run BEFORE 0060, so reusing one would emit
# ENABLE/POLICY DDL for a table that does not exist yet on a fresh
# ``alembic upgrade head`` → hard failure. It applies its own ENABLE+POLICY inline in
# migration 0060 (``late_join_video_sources_upgrade_statements``) and is registered
# for the ``create_all`` (test/dev) path via ``register_late_join_video_sources_rls_ddl``.
#
# 1-hop: the row carries ``project_id`` directly (and deliberately NO ``tenant_id``,
# so an uploaded source can never disagree with its project's tenant), so it climbs
# ``project_id -> projects.tenant_id``.
RLS_JOIN_PREDICATES_VIDEO_SOURCES: dict[str, str] = {
    "video_sources": _exists_predicate("projects p", "p.id = video_sources.project_id"),
}


def late_join_video_sources_upgrade_statements() -> list[str]:
    """DDL for migration 0060's ``upgrade()`` — ENABLE + join policy for video sources."""
    out: list[str] = []
    for t, pred in RLS_JOIN_PREDICATES_VIDEO_SOURCES.items():
        out += _enable_and_policy(t, pred)
    return out


def late_join_video_sources_downgrade_statements() -> list[str]:
    """Reverse of :func:`late_join_video_sources_upgrade_statements` (fully reversible)."""
    out: list[str] = []
    for t in RLS_JOIN_PREDICATES_VIDEO_SOURCES:
        out += _drop_and_disable(t)
    return out


def register_late_join_video_sources_rls_ddl(metadata: MetaData) -> None:
    """Attach ``after_create`` ENABLE+POLICY for ``video_sources`` so a
    ``create_all``-ed schema (tests/dev) carries the SAME RLS migration 0060 applies
    in prod. Separate registration so historical migrations 0046/0049/0050/0051/0052
    keep iterating exactly their own late maps (see the
    ``RLS_JOIN_PREDICATES_VIDEO_SOURCES`` comment). References ``projects`` (an
    ancestor via the ``project_id`` FK); ``create_all`` orders by FK dependency, so
    ``projects`` exists before this table's ``after_create`` fires."""
    for name, predicate in RLS_JOIN_PREDICATES_VIDEO_SOURCES.items():
        table = metadata.tables.get(name)
        if table is None:
            continue
        event.listen(
            table, "after_create", DDL(f"{enable_sql(name)}; {policy_sql(name, predicate)}")
        )


# Join-based table added AFTER the 0060 video-sources increment: the chunked-upload
# session table ``video_source_uploads`` (migration 0062). Kept SEPARATE from every
# other late map for the SAME trap reason they are separate from
# ``RLS_JOIN_PREDICATES``: historical migrations 0046/0049/0050/0051/0052/0060 iterate
# THEIR OWN late maps at run time and run BEFORE 0062, so reusing one would emit
# ENABLE/POLICY DDL for a table that does not exist yet on a fresh
# ``alembic upgrade head`` → hard failure. It applies its own ENABLE+POLICY inline in
# migration 0062 (``late_join_video_source_uploads_upgrade_statements``) and is
# registered for the ``create_all`` (test/dev) path via
# ``register_late_join_video_source_uploads_rls_ddl``.
#
# 1-hop: the row carries ``project_id`` directly (and deliberately NO ``tenant_id``,
# so an upload session can never disagree with its project's tenant), so it climbs
# ``project_id -> projects.tenant_id``. NOTE it does NOT climb via ``source_id`` —
# that is a PRE-ALLOCATED id whose ``video_sources`` row does not exist until the
# upload completes, so a climb through it would read zero rows for every OPEN session,
# i.e. exactly the state the policy has to protect.
RLS_JOIN_PREDICATES_VIDEO_SOURCE_UPLOADS: dict[str, str] = {
    "video_source_uploads": _exists_predicate(
        "projects p", "p.id = video_source_uploads.project_id"
    ),
}


def late_join_video_source_uploads_upgrade_statements() -> list[str]:
    """DDL for migration 0062's ``upgrade()`` — ENABLE + join policy for upload sessions."""
    out: list[str] = []
    for t, pred in RLS_JOIN_PREDICATES_VIDEO_SOURCE_UPLOADS.items():
        out += _enable_and_policy(t, pred)
    return out


def late_join_video_source_uploads_downgrade_statements() -> list[str]:
    """Reverse of :func:`late_join_video_source_uploads_upgrade_statements`
    (fully reversible)."""
    out: list[str] = []
    for t in RLS_JOIN_PREDICATES_VIDEO_SOURCE_UPLOADS:
        out += _drop_and_disable(t)
    return out


def register_late_join_video_source_uploads_rls_ddl(metadata: MetaData) -> None:
    """Attach ``after_create`` ENABLE+POLICY for ``video_source_uploads`` so a
    ``create_all``-ed schema (tests/dev) carries the SAME RLS migration 0062 applies
    in prod. Separate registration so historical migrations
    0046/0049/0050/0051/0052/0060 keep iterating exactly their own late maps (see the
    ``RLS_JOIN_PREDICATES_VIDEO_SOURCE_UPLOADS`` comment). References ``projects`` (an
    ancestor via the ``project_id`` FK); ``create_all`` orders by FK dependency, so
    ``projects`` exists before this table's ``after_create`` fires.

    Both paths are load-bearing. Registering ONLY the migration would leave tests
    running against an unprotected table — a cross-tenant read would pass there and
    fail in prod; registering ONLY here would ship prod without a policy. And under
    enforce an unbound session reads ZERO rows from this table, which the user
    experiences as "my upload vanished", not as a 403."""
    for name, predicate in RLS_JOIN_PREDICATES_VIDEO_SOURCE_UPLOADS.items():
        table = metadata.tables.get(name)
        if table is None:
            continue
        event.listen(
            table, "after_create", DDL(f"{enable_sql(name)}; {policy_sql(name, predicate)}")
        )


# Join-based table added AFTER the 0062 upload-session increment: the step ledger
# ``stage_step_results`` (migration 0071) — durable per-sub-step results that let a
# stage re-entered after a pod death reuse finished sub-steps. Kept SEPARATE from
# ``RLS_JOIN_PREDICATES`` (where its sibling ``stage_outputs`` lives) for the SAME
# trap reason every late map is separate: historical migrations 0028/0029 iterate
# ``RLS_JOIN_PREDICATES`` at freeze time and run BEFORE 0071, and
# ``tests/database/test_frozen_migration_ddl.py`` pins their frozen snapshots to that
# map — adding a table there would both break the snapshot guard and describe a
# policy on a table that does not exist yet at those revisions. It applies its own
# ENABLE+POLICY inline (FROZEN) in migration 0071 and is registered for the
# ``create_all`` (test/dev) path via ``register_stage_step_results_rls_ddl``.
#
# 3-hop, byte-for-byte the ``stage_outputs`` climb: ``run_id -> runs -> deliverables
# -> projects.tenant_id``. The row deliberately carries NO ``tenant_id``, so a ledger
# entry can never disagree with the tenant of the run that owns it.
RLS_JOIN_PREDICATES_STAGE_STEP_RESULTS: dict[str, str] = {
    "stage_step_results": _exists_predicate(
        "runs r JOIN deliverables d ON d.id = r.deliverable_id "
        "JOIN projects p ON p.id = d.project_id",
        "r.id = stage_step_results.run_id",
    ),
}


def stage_step_results_upgrade_statements() -> list[str]:
    """DDL for migration 0071's ``upgrade()`` — ENABLE + join policy for the ledger."""
    out: list[str] = []
    for t, pred in RLS_JOIN_PREDICATES_STAGE_STEP_RESULTS.items():
        out += _enable_and_policy(t, pred)
    return out


def stage_step_results_downgrade_statements() -> list[str]:
    """Reverse of :func:`stage_step_results_upgrade_statements` (fully reversible)."""
    out: list[str] = []
    for t in RLS_JOIN_PREDICATES_STAGE_STEP_RESULTS:
        out += _drop_and_disable(t)
    return out


def register_stage_step_results_rls_ddl(metadata: MetaData) -> None:
    """Attach ``after_create`` ENABLE+POLICY for ``stage_step_results`` so a
    ``create_all``-ed schema (tests/dev) carries the SAME RLS migration 0071 applies
    in prod. References ``runs``/``deliverables``/``projects`` (ancestors via the
    ``run_id`` FK chain); ``create_all`` orders by FK dependency, so they exist before
    this table's ``after_create`` fires.

    Both paths are load-bearing: migration-only would leave tests running against an
    unprotected table, registrar-only would ship prod without a policy. The frozen
    text in 0071 is pinned to these generators by
    ``tests/database/test_migration_0071_stage_step_results.py``."""
    for name, predicate in RLS_JOIN_PREDICATES_STAGE_STEP_RESULTS.items():
        table = metadata.tables.get(name)
        if table is None:
            continue
        event.listen(
            table, "after_create", DDL(f"{enable_sql(name)}; {policy_sql(name, predicate)}")
        )


# P3 provider slots (migration 0077): a tenant's provider connections and their
# immutable versions. ``tenant_connections`` carries ``tenant_id`` directly (plain
# equality); ``tenant_connection_versions`` climbs ONE hop to its connection's
# ``tenant_id`` — it deliberately has no ``tenant_id`` of its own, so a version can
# never disagree with the connection that owns it. Kept in its own map (not
# ``RLS_POLICY_TABLES_LATE`` / ``RLS_JOIN_PREDICATES``) for the same reason as every
# late set above: historical migrations iterate those at run time.
RLS_TENANT_CONNECTIONS_PREDICATES: dict[str, str] = {
    "tenant_connections": _DIRECT_PREDICATE,
    "tenant_connection_versions": (
        "EXISTS (SELECT 1 FROM tenant_connections c "  # noqa: S608 - static DDL
        "WHERE c.id = tenant_connection_versions.connection_id "
        f"AND c.tenant_id = {_GUC})"
    ),
}


def tenant_connections_upgrade_statements() -> list[str]:
    """ENABLE + ``tenant_isolation`` for both connection tables (migration 0077)."""
    out: list[str] = []
    for t, pred in RLS_TENANT_CONNECTIONS_PREDICATES.items():
        out += _enable_and_policy(t, pred)
    return out


def register_tenant_connections_rls_ddl(metadata: MetaData) -> None:
    """``after_create`` ENABLE+POLICY for the connection tables so a ``create_all``
    schema (tests/dev) carries the SAME RLS 0077 applies in prod. The versions
    policy references ``tenant_connections``, which ``create_all`` creates first
    (FK order). 0077's frozen text is pinned to this generator by
    ``tests/database/test_migration_0077_tenant_connections.py``."""
    for name, predicate in RLS_TENANT_CONNECTIONS_PREDICATES.items():
        table = metadata.tables.get(name)
        if table is None:
            continue
        event.listen(
            table, "after_create", DDL(f"{enable_sql(name)}; {policy_sql(name, predicate)}")
        )


# P6 tenant defaults (migration 0081): one row per tenant, ``tenant_id`` direct.
# Its own map for the same reason as every late set above (historical migrations
# iterate the older maps at run time).
RLS_TENANT_DEFAULTS_PREDICATES: dict[str, str] = {"tenant_defaults": _DIRECT_PREDICATE}


def tenant_defaults_upgrade_statements() -> list[str]:
    """ENABLE + ``tenant_isolation`` for ``tenant_defaults`` (migration 0081)."""
    out: list[str] = []
    for t, pred in RLS_TENANT_DEFAULTS_PREDICATES.items():
        out += _enable_and_policy(t, pred)
    return out


def register_tenant_defaults_rls_ddl(metadata: MetaData) -> None:
    """``after_create`` ENABLE+POLICY for ``tenant_defaults`` so a ``create_all``
    schema (tests/dev) carries the SAME RLS 0081 applies in prod. 0081's frozen text
    is pinned to this generator by
    ``tests/database/test_migration_0081_tenant_defaults.py``."""
    for name, predicate in RLS_TENANT_DEFAULTS_PREDICATES.items():
        table = metadata.tables.get(name)
        if table is None:
            continue
        event.listen(
            table, "after_create", DDL(f"{enable_sql(name)}; {policy_sql(name, predicate)}")
        )


# P6 tenant music library (migration 0084): one row per uploaded sound, ``tenant_id``
# direct. Its own map for the same reason as every late set above.
RLS_TENANT_SOUND_ASSETS_PREDICATES: dict[str, str] = {"tenant_sound_assets": _DIRECT_PREDICATE}


def tenant_sound_assets_upgrade_statements() -> list[str]:
    """ENABLE + ``tenant_isolation`` for ``tenant_sound_assets`` (migration 0084)."""
    out: list[str] = []
    for t, pred in RLS_TENANT_SOUND_ASSETS_PREDICATES.items():
        out += _enable_and_policy(t, pred)
    return out


def register_tenant_sound_assets_rls_ddl(metadata: MetaData) -> None:
    """``after_create`` ENABLE+POLICY for ``tenant_sound_assets`` so a ``create_all``
    schema carries the SAME RLS 0084 applies in prod. 0084's frozen text is pinned to
    this generator by ``tests/database/test_migration_0084_tenant_sound_assets.py``."""
    for name, predicate in RLS_TENANT_SOUND_ASSETS_PREDICATES.items():
        table = metadata.tables.get(name)
        if table is None:
            continue
        event.listen(
            table, "after_create", DDL(f"{enable_sql(name)}; {policy_sql(name, predicate)}")
        )


# Service lab run history (migration 0107): one row per bench call, ``tenant_id``
# direct. Its own map for the same reason as every late set above.
RLS_SERVICE_RUNS_PREDICATES: dict[str, str] = {"service_runs": _DIRECT_PREDICATE}


def service_runs_upgrade_statements() -> list[str]:
    """ENABLE + ``tenant_isolation`` for ``service_runs`` (migration 0107)."""
    out: list[str] = []
    for t, pred in RLS_SERVICE_RUNS_PREDICATES.items():
        out += _enable_and_policy(t, pred)
    return out


def register_service_runs_rls_ddl(metadata: MetaData) -> None:
    """``after_create`` ENABLE+POLICY for ``service_runs`` so a ``create_all``
    schema (tests/dev) carries the SAME RLS 0107 applies in prod. The parity is
    exercised indirectly by the ``create_all`` ↔ ``alembic`` round-trips in
    ``tests/database/test_migration_0077_*`` / ``_0081_*``, which is why 0107's
    DDL is drift-tolerant (``IF NOT EXISTS`` + a policy DROP before CREATE)."""
    for name, predicate in RLS_SERVICE_RUNS_PREDICATES.items():
        table = metadata.tables.get(name)
        if table is None:
            continue
        event.listen(
            table, "after_create", DDL(f"{enable_sql(name)}; {policy_sql(name, predicate)}")
        )


# Tenant API keys (migration 0090): ``tenant_id`` direct, the same shape as every
# direct-tenant table. Shipped without RLS in 0074 because a presented key is
# resolved BEFORE any tenant is known; that pre-tenant lookup now goes through the
# ``resolve_api_key`` SECURITY DEFINER resolver below, so the table itself can be
# isolated like its siblings. Its own map for the same reason as every late set.
RLS_TENANT_API_KEYS_PREDICATES: dict[str, str] = {"tenant_api_keys": _DIRECT_PREDICATE}


def tenant_api_keys_upgrade_statements() -> list[str]:
    """ENABLE + ``tenant_isolation`` for ``tenant_api_keys`` (migration 0090)."""
    out: list[str] = []
    for t, pred in RLS_TENANT_API_KEYS_PREDICATES.items():
        out += _enable_and_policy(t, pred)
    return out


def register_tenant_api_keys_rls_ddl(metadata: MetaData) -> None:
    """``after_create`` ENABLE+POLICY for ``tenant_api_keys`` plus its two
    pre-tenant resolvers, so a ``create_all`` schema carries the SAME RLS and
    functions 0090 applies in prod. 0090's frozen text is pinned to these
    generators by ``tests/database/test_migration_0090_api_keys_rls.py``."""
    for name, predicate in RLS_TENANT_API_KEYS_PREDICATES.items():
        table = metadata.tables.get(name)
        if table is None:
            continue
        event.listen(
            table, "after_create", DDL(f"{enable_sql(name)}; {policy_sql(name, predicate)}")
        )
        event.listen(table, "after_create", DDL("; ".join(api_key_functions_statements())))


def register_rls_ddl(metadata: MetaData) -> None:
    """Attach ``after_create`` DDL so ``Base.metadata.create_all`` (tests/dev)
    applies the same RLS as the migrations apply in prod.

    Fires for every direct (``RLS_POLICY_TABLES``) and join
    (``RLS_JOIN_TABLES``) table present in ``metadata``. ``create_all`` follows a
    ``drop_all`` (which cascades the policy away with the table) in the test
    harness, so a plain CREATE POLICY is safe. A join policy references ancestor
    tables (``projects``/``runs``/…); ``create_all`` orders by FK dependency, so
    every referenced ancestor exists before this table's ``after_create`` fires.
    """
    for name, predicate in _all_policy_specs():
        table = metadata.tables.get(name)
        if table is None:
            continue
        event.listen(
            table, "after_create", DDL(f"{enable_sql(name)}; {policy_sql(name, predicate)}")
        )


# ─── Background-session bypass functions (Phase D increment 3b) ────────────────
# Under enforce + the restricted no-BYPASSRLS runtime role, trusted background
# machinery hits a bootstrap wall: the orphan reconciler must scan runs ACROSS
# all tenants, and the dispatcher receives a bare run_id and must learn that
# run's tenant BEFORE it can set the GUC to read the row. Both are solved by
# ``SECURITY DEFINER`` functions owned by the migration user — in prod that user
# is the postgres superuser (bypasses FORCE), in CI/tests it is the table owner
# (bypasses ENABLE-not-FORCE) — so the function body sees every tenant. EXECUTE
# is revoked from PUBLIC; PR 3 grants it to ``w2v_orch_app`` out-of-band (the
# function owner/definer can always execute its own function, so tests as the
# owner are unaffected). These are read-only resolvers — they never widen what a
# query returns, only what the trusted background code can look up.
# Static DDL assembled from the hard-coded run→deliverable→project join + fixed
# column names — no user input ever reaches it, so S608 is a false positive (same
# posture as ``_exists_predicate`` and migration 0004's hard-coded table list).
_RUN_JOIN = (
    "runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id"
)
_RUN_TENANT_BODY = f"SELECT p.tenant_id FROM {_RUN_JOIN} WHERE r.id = p_run_id"  # noqa: S608
_ORPHAN_RUNS_BODY = f"SELECT r.id, p.tenant_id, r.status, r.updated_at FROM {_RUN_JOIN} WHERE r.status IN ('pending', 'running')"  # noqa: S608
# ``SET search_path FROM CURRENT`` pins the search_path in effect when the function
# is CREATED (the migration / create_all connection) rather than a hard-coded
# schema: prod creates it with ``public`` on the path, while CI runs each job in a
# per-job schema — both must resolve the run-tree tables the caller sees. Capturing
# at create time (not execution time) keeps it injection-safe for SECURITY DEFINER.
_FN_PREAMBLE = "LANGUAGE sql STABLE SECURITY DEFINER SET search_path FROM CURRENT AS $$ "
_RUN_TENANT_FN = (
    f"CREATE OR REPLACE FUNCTION run_tenant(p_run_id uuid) RETURNS uuid {_FN_PREAMBLE}"
    f"{_RUN_TENANT_BODY} $$"
)
_ORPHAN_RUNS_FN = (
    "CREATE OR REPLACE FUNCTION orphan_runs() "
    "RETURNS TABLE (run_id uuid, tenant_id uuid, status text, updated_at timestamptz) "
    f"{_FN_PREAMBLE}{_ORPHAN_RUNS_BODY} $$"
)


def run_functions_statements() -> list[str]:
    """DDL for the ``run_tenant``/``orphan_runs`` SECURITY DEFINER resolvers."""
    return [
        # Single-run tenant lookup — the dispatcher's bootstrap before it can
        # bind the GUC and read the run row under RLS.
        _RUN_TENANT_FN,
        "REVOKE EXECUTE ON FUNCTION run_tenant(uuid) FROM PUBLIC",
        # Cross-tenant orphan scan for the startup/periodic reconciler.
        _ORPHAN_RUNS_FN,
        "REVOKE EXECUTE ON FUNCTION orphan_runs() FROM PUBLIC",
    ]


def drop_run_functions_statements() -> list[str]:
    return [
        "DROP FUNCTION IF EXISTS orphan_runs()",
        "DROP FUNCTION IF EXISTS run_tenant(uuid)",
    ]


def register_run_functions_ddl(metadata: MetaData) -> None:
    """Attach the bypass functions as an ``after_create`` on ``runs`` so a
    ``create_all``-ed schema (tests/dev) carries them too. ``runs`` is created
    after ``projects``/``deliverables`` (FK order), so the join targets exist."""
    runs = metadata.tables.get("runs")
    if runs is None:
        return
    event.listen(runs, "after_create", DDL("; ".join(run_functions_statements())))


# ─── Admin cross-tenant rollup resolver (Phase D increment 4) ──────────────────
# The admin/platform surface is excluded from the per-request ``apply_tenant_rls``
# gate and reads RLS-protected tables (projects/runs/run_usage) for super-admins.
# All but one of those reads operate on a KNOWN tenant_id and simply bind that
# tenant's GUC (see ``admin_repository`` + ``bind_session_tenant``), so RLS scopes
# them correctly. The single GENUINELY cross-tenant read — the platform tenants
# list with its per-tenant project/run/cost rollup — cannot bind one tenant, so it
# goes through this ``SECURITY DEFINER`` resolver (the exact pattern as
# ``orphan_runs``/``run_tenant``): owned by the migration user (postgres superuser
# in prod → bypasses FORCE; the table owner in CI/tests → bypasses ENABLE), so the
# body sees every tenant. Read-only, EXECUTE revoked from PUBLIC. This keeps the
# data-plane policies UNCHANGED (no "bypass GUC" disjunct) → provable isolation
# stays intact. Static SQL (no user input).
_ADMIN_TENANT_ROLLUP_BODY = (
    "SELECT pa.tenant_id, pa.project_count, pa.last_activity, "
    "COALESCE(ua.run_count, 0) AS run_count, COALESCE(ua.cost_usd, 0) AS cost_usd "
    "FROM (SELECT p.tenant_id, count(p.id) AS project_count, "
    "max(COALESCE(p.updated_at, p.created_at)) AS last_activity "
    "FROM projects p WHERE p.is_branch = false GROUP BY p.tenant_id) pa "
    "LEFT JOIN (SELECT p.tenant_id, count(DISTINCT r.id) AS run_count, "
    "COALESCE(sum(ru.cost_usd), 0) AS cost_usd FROM runs r "
    "JOIN deliverables d ON d.id = r.deliverable_id "
    "JOIN projects p ON p.id = d.project_id "
    "LEFT JOIN run_usage ru ON ru.run_id = r.id GROUP BY p.tenant_id) ua "
    "ON ua.tenant_id = pa.tenant_id"
)
_ADMIN_TENANT_ROLLUP_FN = (
    "CREATE OR REPLACE FUNCTION admin_tenant_rollup() "
    "RETURNS TABLE (tenant_id uuid, project_count bigint, last_activity timestamptz, "
    "run_count bigint, cost_usd numeric) "
    f"{_FN_PREAMBLE}{_ADMIN_TENANT_ROLLUP_BODY} $$"
)


def admin_functions_statements() -> list[str]:
    """DDL for the ``admin_tenant_rollup`` SECURITY DEFINER resolver."""
    return [
        _ADMIN_TENANT_ROLLUP_FN,
        "REVOKE EXECUTE ON FUNCTION admin_tenant_rollup() FROM PUBLIC",
    ]


def drop_admin_functions_statements() -> list[str]:
    return ["DROP FUNCTION IF EXISTS admin_tenant_rollup()"]


# ─── Admin cross-tenant ACTIVE/RECENT run views (operator run-visibility) ──────
# The platform operator pane ("what is running right now, across all tenants")
# is a second genuinely cross-tenant super-admin read — like ``admin_tenant_rollup``
# it cannot bind a single tenant's GUC, so it goes through SECURITY DEFINER
# resolvers (same owner/REVOKE/static-SQL guarantees; data-plane policies stay
# UNCHANGED). ``admin_active_runs`` returns every pending/running/PAUSED run (paused
# since 0092 — a run awaiting review is in-flight work an operator must see); the richer
# row (tenant + project title + stage + timestamps) is what an operator needs to
# triage in-flight work. ``admin_recent_runs(limit)`` returns the most-recently
# updated terminal runs for the "recent" view. ``paused_at`` (since 0102) is when a
# paused run started waiting for its reviewer: without it the pane could only age a
# run from ``started_at``, so a run parked at a review gate for a day read as a run
# that had been EXECUTING for a day (run 01a1770d, 2026-10-26/27).
_RUN_VIEW_COLS = (
    "run_id uuid, tenant_id uuid, project_id uuid, project_title text, "
    "deliverable_id uuid, blueprint_id text, status text, current_stage text, "
    "started_at timestamptz, created_at timestamptz, updated_at timestamptz, "
    "paused_at timestamptz, pause_reason text"
)
# Static SQL assembled from hard-coded table/column names + the fixed _RUN_JOIN —
# no user input ever reaches it (same posture as _ORPHAN_RUNS_BODY above).
_RUN_VIEW_SELECT = (
    "SELECT r.id, p.tenant_id, p.id, p.title, r.deliverable_id, r.blueprint_id, "  # noqa: S608
    "r.status, r.current_stage, r.started_at, r.created_at, r.updated_at, r.paused_at, "
    f"r.pause_reason FROM {_RUN_JOIN}"
)
_ADMIN_ACTIVE_RUNS_FN = (
    "CREATE OR REPLACE FUNCTION admin_active_runs() "
    f"RETURNS TABLE ({_RUN_VIEW_COLS}) "
    f"{_FN_PREAMBLE}{_RUN_VIEW_SELECT} WHERE r.status IN ('pending', 'running', 'paused') $$"
)
_ADMIN_RECENT_RUNS_FN = (
    "CREATE OR REPLACE FUNCTION admin_recent_runs(p_limit integer) "
    f"RETURNS TABLE ({_RUN_VIEW_COLS}) "
    f"{_FN_PREAMBLE}{_RUN_VIEW_SELECT} "
    "WHERE r.status IN ('completed', 'failed', 'cancelled') "
    "ORDER BY r.updated_at DESC LIMIT p_limit $$"
)


def admin_run_view_statements() -> list[str]:
    """DDL for the ``admin_active_runs`` / ``admin_recent_runs`` resolvers."""
    return [
        _ADMIN_ACTIVE_RUNS_FN,
        "REVOKE EXECUTE ON FUNCTION admin_active_runs() FROM PUBLIC",
        _ADMIN_RECENT_RUNS_FN,
        "REVOKE EXECUTE ON FUNCTION admin_recent_runs(integer) FROM PUBLIC",
    ]


def drop_admin_run_view_statements() -> list[str]:
    return [
        "DROP FUNCTION IF EXISTS admin_active_runs()",
        "DROP FUNCTION IF EXISTS admin_recent_runs(integer)",
    ]


# ``EXECUTE`` on the admin run-view resolvers was created REVOKEd from PUBLIC with
# the grant to the restricted runtime role ``w2v_orch_app`` deferred out-of-band
# (see 0035 / the comment above). That out-of-band grant was never applied in
# prod, so under ``enforce`` the runtime role hit ``permission denied for function
# admin_active_runs`` — a raw 500 on ``GET /api/admin/runs``. Codify it forward-only
# so every deploy's migration (run as the privileged owner ``w2v_service``, which
# owns these functions) self-heals the grant. Role-GUARDED: ``w2v_orch_app`` exists
# only in prod, not in CI/test DBs (create_all / ephemeral postgres run as the
# owner), so an unconditional GRANT would error there — the ``DO`` block no-ops when
# the role is absent.
_GRANT_ADMIN_RUN_VIEWS = (
    "DO $$ BEGIN "
    "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_orch_app') THEN "
    "GRANT EXECUTE ON FUNCTION admin_active_runs() TO w2v_orch_app; "
    "GRANT EXECUTE ON FUNCTION admin_recent_runs(integer) TO w2v_orch_app; "
    "END IF; END $$"
)
_REVOKE_ADMIN_RUN_VIEWS = (
    "DO $$ BEGIN "
    "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_orch_app') THEN "
    "REVOKE EXECUTE ON FUNCTION admin_active_runs() FROM w2v_orch_app; "
    "REVOKE EXECUTE ON FUNCTION admin_recent_runs(integer) FROM w2v_orch_app; "
    "END IF; END $$"
)


def grant_admin_run_views_statements() -> list[str]:
    """Role-guarded GRANT EXECUTE on the admin run-view resolvers to ``w2v_orch_app``."""
    return [_GRANT_ADMIN_RUN_VIEWS]


def revoke_admin_run_views_statements() -> list[str]:
    """Downgrade counterpart of ``grant_admin_run_views_statements``."""
    return [_REVOKE_ADMIN_RUN_VIEWS]


def register_admin_functions_ddl(metadata: MetaData) -> None:
    """Attach the admin cross-tenant resolvers as an ``after_create`` on
    ``run_usage`` so a ``create_all``-ed schema (tests/dev) carries them too.
    ``run_usage`` is created after projects/deliverables/runs (FK order), so every
    join target exists."""
    run_usage = metadata.tables.get("run_usage")
    if run_usage is None:
        return
    stmts = admin_functions_statements() + admin_run_view_statements()
    event.listen(run_usage, "after_create", DDL("; ".join(stmts)))


# ─── Pending-video-source resolver (the preview-proxy worker's cross-tenant scan) ──
# ``video_sources`` rows land with ``preview_status='pending'`` and a background
# sweeper (``src.lifecycle.preview_proxy``) turns each original into a browser-
# playable h264 proxy. That sweep spans ALL tenants and runs on a session with no
# tenant bound, so under enforce a plain ``SELECT FROM video_sources`` would be
# RLS-scoped to an unset tenant → 0 rows, i.e. the worker would sweep forever and
# find nothing. Exactly the ``orphan_runs()`` situation, solved exactly the same
# way: a ``SECURITY DEFINER`` resolver owned by the migration user (postgres
# superuser in prod → bypasses FORCE; the table owner in CI/tests → bypasses
# ENABLE-not-FORCE) that returns each pending row PLUS the tenant the worker must
# bind before it writes. The data-plane policy on ``video_sources`` is UNCHANGED —
# every UPDATE the worker issues still runs under the bound tenant's policy, so
# provable isolation stays intact. Read-only; EXECUTE revoked from PUBLIC.
#
# ``attempts`` is derived here rather than in Python so the worker never has to
# parse the free-form ffprobe blob just to learn its own retry count. The
# ``jsonb_typeof`` guard is load-bearing: ``probe`` is raw ffprobe output plus
# whatever the worker has written under ``preview_worker``, and an unguarded
# ``::int`` cast on a non-numeric value would ERROR the whole scan (one poisoned
# row would stop every tenant's previews), not just that row.
#
# Static SQL assembled from hard-coded table/column names — no user input ever
# reaches it (same posture as ``_ORPHAN_RUNS_BODY``), so S608 is a false positive.
_PENDING_VIDEO_SOURCES_BODY = (
    "SELECT vs.id, p.tenant_id, vs.project_id, vs.object_key, "
    "CASE WHEN jsonb_typeof(vs.probe->'preview_worker'->'attempts') = 'number' "
    "THEN (vs.probe->'preview_worker'->>'attempts')::int ELSE 0 END "
    "FROM video_sources vs JOIN projects p ON p.id = vs.project_id "
    "WHERE vs.preview_status = 'pending' "
    # The LEASE fence. Without it a row whose worker is mid-transcode is handed
    # straight back to the next tick (the advisory lock only stops the SECOND pod
    # once it has already been scanned and paid for a session), and — worse — a
    # pod SIGKILLed mid-transcode never wrote its attempt counter, so the row was
    # re-picked forever with the ``max_attempts`` cap unable to engage. An expired
    # lease is deliberately re-offered: that IS the crash-recovery path, and it
    # needs no reaper because the claim already durably counted the attempt.
    "AND (vs.preview_leased_until IS NULL OR vs.preview_leased_until < now()) "
    "ORDER BY vs.created_at LIMIT p_limit"
)
_PENDING_VIDEO_SOURCES_FN = (
    "CREATE OR REPLACE FUNCTION pending_video_sources(p_limit integer) "
    "RETURNS TABLE (source_id uuid, tenant_id uuid, project_id uuid, "
    "object_key text, attempts integer) "
    f"{_FN_PREAMBLE}{_PENDING_VIDEO_SOURCES_BODY} $$"
)
# Same role-guarded shape as the admin run-view grant: ``w2v_orch_app`` (the
# restricted, no-BYPASSRLS runtime role) exists only in prod, so an unconditional
# GRANT would error on a CI/test DB where migrations run as the owner.
_GRANT_PENDING_VIDEO_SOURCES = (
    "DO $$ BEGIN "
    "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_orch_app') THEN "
    "GRANT EXECUTE ON FUNCTION pending_video_sources(integer) TO w2v_orch_app; "
    "END IF; END $$"
)


def video_source_functions_statements() -> list[str]:
    """DDL for the ``pending_video_sources`` SECURITY DEFINER resolver (+ grant)."""
    return [
        _PENDING_VIDEO_SOURCES_FN,
        "REVOKE EXECUTE ON FUNCTION pending_video_sources(integer) FROM PUBLIC",
        _GRANT_PENDING_VIDEO_SOURCES,
    ]


def drop_video_source_functions_statements() -> list[str]:
    return ["DROP FUNCTION IF EXISTS pending_video_sources(integer)"]


def register_video_source_functions_ddl(metadata: MetaData) -> None:
    """Attach ``pending_video_sources`` as an ``after_create`` on ``video_sources``
    so a ``create_all``-ed schema (tests/dev) carries it too. ``video_sources`` is
    created after ``projects`` (FK order), so the join target exists."""
    video_sources = metadata.tables.get("video_sources")
    if video_sources is None:
        return
    event.listen(video_sources, "after_create", DDL("; ".join(video_source_functions_statements())))


# ─── Tenant-occupancy resolver (P2 partner tenant creation) ─────────────────────
# ``POST /api/tenants`` lets an allow-listed partner pick the ``tenant_id``. It may
# only claim (or adopt a backfill row for) an id that is NOT in use, so it must ask
# "does tenant X hold anything?" — but the partner's request is bound to the
# PARTNER's own tenant (or none), so under enforce a plain ``SELECT … FROM projects
# WHERE tenant_id = X`` is RLS-scoped to the wrong tenant and answers 0: exactly
# the takeover it exists to stop. Same answer as ``run_tenant``: a read-only
# ``SECURITY DEFINER`` boolean owned by the migration user (bypasses ENABLE-not-
# FORCE). It returns one bit — never a row — so it widens nothing a caller can
# read. EXECUTE revoked from PUBLIC, granted to ``w2v_orch_app`` under the usual
# role guard (that role exists only in prod). Static SQL, no user input.
#
# "Occupied" = any row in a table that holds the tenant's DATA or ACCESS:
#   * ``projects``       — every project-scoped table (runs, storyboards, sources,
#                          …) hangs off a project, so this one covers them all;
#   * ``tenant_members`` — people (any status);
#   * ``tenant_invites`` — people on the way in (any status);
#   * ``tenant_api_keys``— machine access (revoked or not).
# Deliberately NOT: ``source_ingest_cache`` (a re-derivable cache, not data) and the
# audit logs (``admin_audit_log`` / ``provisioning_audit`` / ``request_audit``: a
# record that someone LOOKED at or tried an id does not make it in use).
#
# ``search_path`` is PINNED, with ``pg_temp`` LAST (a SECURITY DEFINER function
# must never resolve an unqualified name through the caller's temp schema — with
# ``FROM CURRENT`` and no explicit ``pg_temp`` it is searched FIRST). The middle
# entry is the schema the function is created in, resolved ONCE at creation via
# ``quote_ident(current_schema())``: ``public`` under the prod migration (so the
# stored config is ``pg_catalog, public, pg_temp``), the per-run schema under the
# test suite's schema-isolated builds. No ``%`` and no ``:`` in the text, so it
# survives both ``DDL`` formatting and ``text()`` bind parsing unescaped.
_TENANT_IS_OCCUPIED_BODY = (
    "SELECT EXISTS (SELECT 1 FROM projects WHERE tenant_id = p_tenant) "
    "OR EXISTS (SELECT 1 FROM tenant_members WHERE tenant_id = p_tenant) "
    "OR EXISTS (SELECT 1 FROM tenant_invites WHERE tenant_id = p_tenant) "
    "OR EXISTS (SELECT 1 FROM tenant_api_keys WHERE tenant_id = p_tenant)"
)
_TENANT_IS_OCCUPIED_FN = (
    "DO $do$ BEGIN EXECUTE "
    "'CREATE OR REPLACE FUNCTION tenant_is_occupied(p_tenant uuid) RETURNS boolean "
    "LANGUAGE sql STABLE SECURITY DEFINER "
    "SET search_path = pg_catalog, ' || quote_ident(current_schema()) || ', pg_temp "
    f"AS $fn$ {_TENANT_IS_OCCUPIED_BODY} $fn$'; END $do$"
)
_GRANT_TENANT_IS_OCCUPIED = (
    "DO $$ BEGIN "
    "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_orch_app') THEN "
    "GRANT EXECUTE ON FUNCTION tenant_is_occupied(uuid) TO w2v_orch_app; "
    "END IF; END $$"
)


def tenant_occupancy_functions_statements() -> list[str]:
    """DDL for the ``tenant_is_occupied`` SECURITY DEFINER resolver (+ grant)."""
    return [
        _TENANT_IS_OCCUPIED_FN,
        "REVOKE EXECUTE ON FUNCTION tenant_is_occupied(uuid) FROM PUBLIC",
        _GRANT_TENANT_IS_OCCUPIED,
    ]


def drop_tenant_occupancy_functions_statements() -> list[str]:
    return ["DROP FUNCTION IF EXISTS tenant_is_occupied(uuid)"]


def register_tenant_occupancy_functions_ddl(metadata: MetaData) -> None:
    """Attach ``tenant_is_occupied`` as a METADATA-level ``after_create`` so a
    ``create_all``-ed schema (tests/dev) carries it too. Metadata-level because the
    body reads four tables and a ``LANGUAGE sql`` body is validated at CREATE —
    all four must exist, which only holds once the whole metadata is built."""
    event.listen(metadata, "after_create", DDL("; ".join(tenant_occupancy_functions_statements())))


# ─── Provider-secret GC (P3 WP3) ──────────────────────────────────────────────
# Purges the ciphertext of SUPERSEDED tenant-connection versions that no
# non-terminal run (pending / running / paused …) still pins via
# ``runs.config_snapshot->'provider_slots'``. Cross-tenant by nature, and the pod
# runs as the restricted ``w2v_orch_app`` role under enforce — a plain session
# would see one tenant or none and purge nothing (or, worse, if it saw versions but
# not runs, purge pinned secrets). So it is a ``SECURITY DEFINER`` function owned
# by the migration user, and it reads the pins and purges in ONE statement: there
# is no window between "which versions are pinned" and "purge the rest".
#
# Fail SAFE: a non-terminal run whose ``provider_slots`` is not an object, or a
# tenant pin whose ids do not cast, RAISES — the GC cannot prove that version
# unreferenced, so nothing is purged. ``NOT IN`` the terminal set (not ``IN`` a
# live set) so a status added later is protected by default. The caller supplies
# ``p_superseded_before`` (now - grace) so a version superseded moments ago — one
# a run may have pinned in a not-yet-committed snapshot — is never touched.
#
# plpgsql (body not validated at CREATE, so table order does not matter) with the
# explicit ``pg_catalog, <schema>, pg_temp`` search_path of ``tenant_is_occupied``.
# The body is dollar-quoted inside the EXECUTE so it may contain single quotes; no
# ``%`` and no single-colon binds, so it survives ``DDL`` and ``text()`` unescaped.
# A run's pins are LIVE while it is non-terminal OR reached a terminal status
# within the last hour: the structural-relief child is spawned AFTER its parent is
# marked completed (``run_dispatcher``), so the inherited pins must survive that
# gap. The cutoff and the batch size are clamped server-side so no caller (an
# ``'infinity'`` cutoff, a huge/negative limit) can bypass the 1h floor.
_PROVIDER_GC_BODY = (
    "DECLARE n integer; "
    "v_cutoff timestamptz := LEAST(p_superseded_before, now() - interval '1 hour'); "
    "v_limit integer := GREATEST(LEAST(p_limit, 10000), 0); "
    "BEGIN "
    "IF EXISTS (SELECT 1 FROM runs r "
    "WHERE (r.status NOT IN ('completed', 'failed', 'cancelled') "
    "OR COALESCE(r.completed_at, r.updated_at) > now() - interval '1 hour') "
    "AND r.config_snapshot->'provider_slots' IS NOT NULL "
    "AND jsonb_typeof(r.config_snapshot->'provider_slots') NOT IN ('object', 'null')) THEN "
    "RAISE EXCEPTION 'malformed provider_slots on a live run; refusing to purge'; "
    "END IF; "
    "WITH pinned AS ("
    "SELECT (e.value->>'connection_id')::uuid AS connection_id, "
    "(e.value->>'version')::integer AS version "
    "FROM runs r CROSS JOIN LATERAL jsonb_each(CASE "
    "WHEN jsonb_typeof(r.config_snapshot->'provider_slots') = 'object' "
    "THEN r.config_snapshot->'provider_slots' ELSE '{}'::jsonb END) e "
    "WHERE (r.status NOT IN ('completed', 'failed', 'cancelled') "
    "OR COALESCE(r.completed_at, r.updated_at) > now() - interval '1 hour') "
    "AND e.value->>'source' = 'tenant'"
    "), victims AS ("
    "SELECT v.connection_id, v.version FROM tenant_connection_versions v "
    "WHERE v.superseded_at < v_cutoff "
    "AND v.secret_ciphertext IS NOT NULL AND v.purged_at IS NULL "
    "AND NOT EXISTS (SELECT 1 FROM pinned p "
    "WHERE p.connection_id = v.connection_id AND p.version = v.version) "
    "ORDER BY v.superseded_at LIMIT v_limit"
    ") "
    "UPDATE tenant_connection_versions t "
    "SET secret_ciphertext = NULL, key_id = NULL, purged_at = now() "
    "FROM victims WHERE t.connection_id = victims.connection_id "
    "AND t.version = victims.version; "
    "GET DIAGNOSTICS n = ROW_COUNT; "
    "RETURN n; "
    "END"
)
_PROVIDER_GC_FN = (
    "DO $do$ BEGIN EXECUTE "
    "$q$CREATE OR REPLACE FUNCTION purge_unpinned_provider_secrets("
    "p_superseded_before timestamptz, p_limit integer) RETURNS integer "
    "LANGUAGE plpgsql VOLATILE SECURITY DEFINER "
    "SET search_path = pg_catalog, $q$ || quote_ident(current_schema()) || $q$, pg_temp "
    f"AS $fn$ {_PROVIDER_GC_BODY} $fn$$q$; END $do$"
)
_GRANT_PROVIDER_GC = (
    "DO $$ BEGIN "
    "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_orch_app') THEN "
    "GRANT EXECUTE ON FUNCTION purge_unpinned_provider_secrets(timestamptz, integer) "
    "TO w2v_orch_app; "
    "END IF; END $$"
)


def provider_secret_gc_statements() -> list[str]:
    """DDL for ``purge_unpinned_provider_secrets`` (SECURITY DEFINER) + grant."""
    return [
        _PROVIDER_GC_FN,
        "REVOKE EXECUTE ON FUNCTION purge_unpinned_provider_secrets(timestamptz, integer) "
        "FROM PUBLIC",
        _GRANT_PROVIDER_GC,
    ]


def drop_provider_secret_gc_statements() -> list[str]:
    return ["DROP FUNCTION IF EXISTS purge_unpinned_provider_secrets(timestamptz, integer)"]


def register_provider_secret_gc_ddl(metadata: MetaData) -> None:
    """METADATA-level ``after_create`` so a ``create_all``-ed schema carries the GC
    function too (it reads ``runs`` and ``tenant_connection_versions``)."""
    event.listen(metadata, "after_create", DDL("; ".join(provider_secret_gc_statements())))


# ─── Pre-tenant API-key resolvers (migration 0090) ────────────────────────────
# A ``Bearer w2v_sk_…`` is verified BEFORE any tenant is known — the key IS what
# names the tenant — so under enforce + the restricted ``w2v_orch_app`` role a
# plain ``SELECT … FROM tenant_api_keys WHERE prefix = …`` returns 0 rows and every
# key would 401. Same answer as ``run_tenant`` / ``tenant_is_occupied``: read-only
# ``SECURITY DEFINER`` functions owned by the migration user (postgres superuser in
# prod → bypasses FORCE; the table owner in CI/tests → bypasses ENABLE-not-FORCE).
#
#   * ``resolve_api_key(prefix, sha256)`` — the key row matching BOTH the public
#     prefix and the SHA-256 of the full presented key. It returns ONLY
#     ``(key_id, tenant_id, scopes, revoked_at, expires_at)`` — never the hash,
#     the name or the creator — so possessing the key is the only way to learn
#     anything about it: a caller that does not already hold the secret matches
#     no row. The digest equality is not constant-time, which is fine: it compares
#     SHA-256 outputs of a 256-bit secret, and a timing leak of a digest prefix
#     does not help forge a preimage. Expiry is returned (not filtered) so the
#     caller keeps its injectable ``now`` (tests) — revoked/expired keys are still
#     refused by the repository, as before.
#   * ``api_key_prefix_taken(prefix)`` — one bit, for ``create``'s prefix re-roll:
#     prefixes are UNIQUE across ALL tenants, so a tenant-scoped pre-check would
#     miss another tenant's prefix and the mint would die on the constraint.
#
# EXECUTE revoked from PUBLIC, granted to ``w2v_orch_app`` under the usual role
# guard. ``search_path`` pinned ``pg_catalog, <schema>, pg_temp`` exactly as
# ``tenant_is_occupied`` (pg_temp LAST). Static SQL, no ``%``/``:``/quotes in the
# bodies, so they survive ``DDL`` formatting, ``text()`` and the EXECUTE literal.
_RESOLVE_API_KEY_BODY = (
    "SELECT k.id, k.tenant_id, k.scopes, k.revoked_at, k.expires_at "
    "FROM tenant_api_keys k WHERE k.prefix = p_prefix AND k.key_hash = p_key_hash"
)
_RESOLVE_API_KEY_FN = (
    "DO $do$ BEGIN EXECUTE "
    "'CREATE OR REPLACE FUNCTION resolve_api_key(p_prefix text, p_key_hash bytea) "
    "RETURNS TABLE (key_id uuid, tenant_id uuid, scopes text[], "
    "revoked_at timestamptz, expires_at timestamptz) "
    "LANGUAGE sql STABLE SECURITY DEFINER "
    "SET search_path = pg_catalog, ' || quote_ident(current_schema()) || ', pg_temp "
    f"AS $fn$ {_RESOLVE_API_KEY_BODY} $fn$'; END $do$"
)
_API_KEY_PREFIX_TAKEN_BODY = (  # pragma: allowlist secret — static SQL, not a secret
    "SELECT EXISTS (SELECT 1 FROM tenant_api_keys WHERE prefix = p_prefix)"
)
_API_KEY_PREFIX_TAKEN_FN = (
    "DO $do$ BEGIN EXECUTE "
    "'CREATE OR REPLACE FUNCTION api_key_prefix_taken(p_prefix text) RETURNS boolean "
    "LANGUAGE sql STABLE SECURITY DEFINER "
    "SET search_path = pg_catalog, ' || quote_ident(current_schema()) || ', pg_temp "
    f"AS $fn$ {_API_KEY_PREFIX_TAKEN_BODY} $fn$'; END $do$"
)
_GRANT_API_KEY_FUNCTIONS = (
    "DO $$ BEGIN "
    "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_orch_app') THEN "
    "GRANT EXECUTE ON FUNCTION resolve_api_key(text, bytea) TO w2v_orch_app; "
    "GRANT EXECUTE ON FUNCTION api_key_prefix_taken(text) TO w2v_orch_app; "
    "END IF; END $$"
)


def api_key_functions_statements() -> list[str]:
    """DDL for ``resolve_api_key`` + ``api_key_prefix_taken`` (SECURITY DEFINER) + grants."""
    return [
        _RESOLVE_API_KEY_FN,
        "REVOKE EXECUTE ON FUNCTION resolve_api_key(text, bytea) FROM PUBLIC",
        _API_KEY_PREFIX_TAKEN_FN,
        "REVOKE EXECUTE ON FUNCTION api_key_prefix_taken(text) FROM PUBLIC",
        _GRANT_API_KEY_FUNCTIONS,
    ]


def drop_api_key_functions_statements() -> list[str]:
    return [
        "DROP FUNCTION IF EXISTS api_key_prefix_taken(text)",
        "DROP FUNCTION IF EXISTS resolve_api_key(text, bytea)",
    ]
