"""storyboard_snapshots — version history + restore for the storyboard graph

Phase 5 storyboard version history: a snapshot captures the WHOLE storyboard
graph (storyboard + scenes + panels + assets) for a project as one self-contained
JSONB ``payload`` (the ``Run.config_snapshot`` blob idiom), so it can be listed
and rolled back to. Adds:

* ``storyboard_snapshots`` — project-scoped, ``Text`` + ``CheckConstraint`` enum
  (``kind IN ('manual','auto')``), ``timestamptz`` server default, JSONB ``payload``,
  a ``(project_id, created_at DESC)`` newest-first index, and RLS ENABLE-not-FORCE +
  the join-based ``tenant_isolation`` policy climbing ``project_id ->
  projects.tenant_id`` — sourced from ``src.database.rls``
  (``RLS_JOIN_PREDICATES_SNAPSHOTS``) so this migration and the ``create_all``
  ``after_create`` path (``register_late_join_snapshots_rls_ddl``) can't drift.
* widens ``ck_storyboard_panel_events_type`` to allow the new ``'restored'`` event
  a restore appends per panel.
* makes ``uq_storyboard_panels_project_shot`` DEFERRABLE INITIALLY IMMEDIATE so a
  restore can upsert shot_key swaps/reorders in one transaction (validated at
  COMMIT via ``SET CONSTRAINTS ALL DEFERRED``); every other write path is
  unchanged (INITIALLY IMMEDIATE = per-statement check unless explicitly deferred).
  Postgres cannot alter a constraint's deferrability in place, so it is dropped
  and re-added.

Like 0049, the snapshots table is registered in ``RLS_JOIN_PREDICATES_SNAPSHOTS``
— NOT the earlier ``_LATE`` / ``_LATE_EVENTS`` maps historical migrations iterate
at run time — so a fresh ``alembic upgrade head`` never emits DDL for a table it
has not created yet; the ENABLE+POLICY is emitted here directly.

Revision ID: 0050
Revises: 0049
Create Date: 2026-07-13
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "0050"
down_revision = "0049"
branch_labels = None
depends_on = None

# FROZEN DDL — inlined snapshot of ``src.database.rls.late_join_snapshots_downgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_LATE_JOIN_SNAPSHOTS_DOWNGRADE_STATEMENTS: tuple[str, ...] = (
    "DROP POLICY IF EXISTS tenant_isolation ON storyboard_snapshots",
    "ALTER TABLE storyboard_snapshots DISABLE ROW LEVEL SECURITY",
)


# FROZEN DDL — inlined snapshot of ``src.database.rls.late_join_snapshots_upgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_LATE_JOIN_SNAPSHOTS_UPGRADE_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE storyboard_snapshots ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON storyboard_snapshots USING (EXISTS (SELECT 1 FROM projects p WHERE p.id = storyboard_snapshots.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM projects p WHERE p.id = storyboard_snapshots.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
)


_EVENTS_CHECK = "ck_storyboard_panel_events_type"
_PANELS_UNIQUE = "uq_storyboard_panels_project_shot"


def upgrade() -> None:
    op.create_table(
        "storyboard_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "storyboard_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("storyboards.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("actor_subject", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.CheckConstraint("kind IN ('manual','auto')", name="ck_storyboard_snapshots_kind"),
    )
    op.create_index(
        "idx_storyboard_snapshots_project_created",
        "storyboard_snapshots",
        ["project_id", "created_at"],
    )

    # Widen the events CHECK to admit the restore trail event.
    op.drop_constraint(_EVENTS_CHECK, "storyboard_panel_events", type_="check")
    op.create_check_constraint(
        _EVENTS_CHECK,
        "storyboard_panel_events",
        "event_type IN ('approved','unapproved','restored')",
    )

    # Re-create the panels unique constraint as DEFERRABLE INITIALLY IMMEDIATE
    # (Postgres cannot alter deferrability in place → drop + add). Raw SQL because
    # alembic's create_unique_constraint does not expose the deferrable clause.
    op.drop_constraint(_PANELS_UNIQUE, "storyboard_panels", type_="unique")
    op.execute(
        f"ALTER TABLE storyboard_panels ADD CONSTRAINT {_PANELS_UNIQUE} "
        "UNIQUE (project_id, shot_key) DEFERRABLE INITIALLY IMMEDIATE"
    )

    # RLS: ENABLE (not FORCE) + join-based tenant_isolation policy (sourced from
    # rls.py, frozen; drift caught by tests/database/test_frozen_migration_ddl.py).
    for stmt in _LATE_JOIN_SNAPSHOTS_UPGRADE_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _LATE_JOIN_SNAPSHOTS_DOWNGRADE_STATEMENTS:
        op.execute(stmt)

    # Restore the non-deferrable unique constraint.
    op.drop_constraint(_PANELS_UNIQUE, "storyboard_panels", type_="unique")
    op.create_unique_constraint(_PANELS_UNIQUE, "storyboard_panels", ["project_id", "shot_key"])

    # Narrow the events CHECK back to the pre-restore vocabulary.
    op.drop_constraint(_EVENTS_CHECK, "storyboard_panel_events", type_="check")
    op.create_check_constraint(
        _EVENTS_CHECK,
        "storyboard_panel_events",
        "event_type IN ('approved','unapproved')",
    )

    op.drop_index("idx_storyboard_snapshots_project_created", table_name="storyboard_snapshots")
    op.drop_table("storyboard_snapshots")
