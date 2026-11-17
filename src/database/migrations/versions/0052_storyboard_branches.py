"""storyboard branch/merge — is_branch + storyboard_branches + origin_id + merged event

Additive-only schema for storyboard branch/merge (design
``05_STORYBOARD_BRANCH_MERGE_DESIGN.md`` §PR2). A branch is a real, HIDDEN child
``projects`` row; this migration adds the machinery that lets one project be forked,
diffed, and merged back:

* ``projects.is_branch`` — ``bool NOT NULL DEFAULT false``. Additive with a constant
  server-default → a metadata-only ``ADD COLUMN`` in pg11+, no table rewrite. Every
  user-facing projects LISTING filters ``is_branch = false`` (enforced in
  ``ProjectRepository``); it is deliberately NOT hung off ``parent_project_id`` (that
  is the course→lesson hierarchy).
* ``origin_id uuid NULL`` on ``storyboard_panels`` / ``storyboard_scenes`` /
  ``storyboard_assets`` — additive nullable, NO FK (it references a row in a DIFFERENT
  project that may be deleted; it is a lineage marker, not a constraint). Metadata-only.
  It is the identity that survives fork's id re-mint so merge can match rows. Each gets
  a PARTIAL btree on ``(project_id, origin_id) WHERE origin_id IS NOT NULL`` — merge
  matches row-by-row on it, and partial keeps it sized to the branches in flight.
* ``storyboard_snapshots.kind`` CHECK gains ``'branch_base'`` — the snapshot a fork
  mints to pin its merge base, given its own kind so neither the auto-prune nor the
  user-facing DELETE can take it away from an open branch.
* ``storyboard_branches`` — the branch↔root link + merge lifecycle. Both project FKs
  ON DELETE CASCADE; ``child_project_id`` UNIQUE (one record per child);
  ``base_snapshot_id`` pins the fork point (the ``base`` side of merge's three-way
  diff), also ON DELETE CASCADE — see the column comment. RLS: 1-hop
  ``root_project_id -> projects.tenant_id``, sourced
  from ``src.database.rls`` (``late_join_branches_*_statements``) so this migration and
  the ``create_all`` ``after_create`` path can't drift, and emitted here directly (the
  earlier late maps belong to 0046/0049/0050/0051, which iterate them at run time
  *before* this table exists).
* ``storyboard_panel_events.event_type`` CHECK gains ``'merged'`` — a branch MERGE
  appends one ``merged`` trail event per touched panel.

Runtime-role table privileges ride ``ALTER DEFAULT PRIVILEGES`` (gitops), like every
other table — no GRANT here (identical to 0046/0049/0050/0051).

Revision ID: 0052
Revises: 0051
Create Date: 2026-07-13
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "0052"
down_revision = "0051"
branch_labels = None
depends_on = None

# FROZEN DDL — inlined snapshot of ``src.database.rls.late_join_branches_downgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_LATE_JOIN_BRANCHES_DOWNGRADE_STATEMENTS: tuple[str, ...] = (
    "DROP POLICY IF EXISTS tenant_isolation ON storyboard_branches",
    "ALTER TABLE storyboard_branches DISABLE ROW LEVEL SECURITY",
)


# FROZEN DDL — inlined snapshot of ``src.database.rls.late_join_branches_upgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_LATE_JOIN_BRANCHES_UPGRADE_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE storyboard_branches ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON storyboard_branches USING (EXISTS (SELECT 1 FROM projects p WHERE p.id = storyboard_branches.root_project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM projects p WHERE p.id = storyboard_branches.root_project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
)


_ORIGIN_ID_TABLES = ("storyboard_panels", "storyboard_scenes", "storyboard_assets")

_EVENT_TYPE_CONSTRAINT = "ck_storyboard_panel_events_type"

_SNAPSHOT_KIND_CONSTRAINT = "ck_storyboard_snapshots_kind"


def upgrade() -> None:
    # 1. projects.is_branch — constant default → metadata-only ADD COLUMN, no rewrite.
    op.add_column(
        "projects",
        sa.Column("is_branch", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )

    # 2. origin_id lineage — additive nullable, no FK (references a possibly-deleted
    #    row in another project; a lineage marker, not a constraint).
    for table in _ORIGIN_ID_TABLES:
        op.add_column(
            table,
            sa.Column("origin_id", postgresql.UUID(as_uuid=True), nullable=True),
        )
        # Merge resolves every row of both sides BY ``origin_id`` within a project —
        # unindexed that is a seq scan per row. PARTIAL on ``origin_id IS NOT NULL``:
        # only forked rows carry lineage, so the index tracks the branches in flight,
        # not the table size (and the index is empty until the first fork).
        op.create_index(
            f"idx_{table}_origin",
            table,
            ["project_id", "origin_id"],
            postgresql_where=sa.text("origin_id IS NOT NULL"),
        )

    # 3. storyboard_branches — the branch↔root link + merge lifecycle.
    op.create_table(
        "storyboard_branches",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "root_project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "child_project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "base_snapshot_id",
            postgresql.UUID(as_uuid=True),
            # CASCADE: with kind='branch_base' (step 5 below) the base snapshot is
            # unreachable by the auto-prune AND by a hand DELETE, so the only thing
            # left that can delete it is deleting the whole project — which already
            # reaps this branch row via the two project FKs. A non-cascading FK would
            # not save the branch there; it would only risk a NO ACTION violation
            # mid-cascade, depending on which side postgres reaps first.
            sa.ForeignKey("storyboard_snapshots.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.Text(), nullable=False, server_default="open"),
        sa.Column("actor_subject", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("merged_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('open','merged','abandoned')",
            name="ck_storyboard_branches_status",
        ),
        sa.UniqueConstraint("child_project_id", name="uq_storyboard_branches_child"),
    )
    op.create_index(
        "idx_storyboard_branches_root",
        "storyboard_branches",
        ["root_project_id", "created_at"],
    )

    # RLS: ENABLE-not-FORCE + the 1-hop join-based tenant_isolation policy (sourced
    # frozen from rls.py; drift is caught by tests/database/test_frozen_migration_ddl.py).
    for stmt in _LATE_JOIN_BRANCHES_UPGRADE_STATEMENTS:
        op.execute(stmt)

    # 4. event_type CHECK gains 'merged'.
    op.drop_constraint(_EVENT_TYPE_CONSTRAINT, "storyboard_panel_events", type_="check")
    op.create_check_constraint(
        _EVENT_TYPE_CONSTRAINT,
        "storyboard_panel_events",
        "event_type IN ('approved','unapproved','restored','merged')",
    )

    # 5. snapshot kind CHECK gains 'branch_base' — the snapshot a FORK mints to pin the
    #    branch's merge base. It has to be its OWN kind: an 'auto' base would be swept
    #    by ``StoryboardSnapshotRepository.prune_auto_for_project`` (it deletes every
    #    'auto' beyond the newest N and shields exactly ONE ``protect_id``, not the
    #    arbitrarily many open branches), and a 'manual' base is hand-deletable (the
    #    DELETE route only 409s on 'auto'). Either way the branch would lose its diff
    #    base. A third value is immune to the prune for free (it filters on kind='auto')
    #    and the DELETE route's 409 gate is widened to match.
    op.drop_constraint(_SNAPSHOT_KIND_CONSTRAINT, "storyboard_snapshots", type_="check")
    op.create_check_constraint(
        _SNAPSHOT_KIND_CONSTRAINT,
        "storyboard_snapshots",
        "kind IN ('manual','auto','branch_base')",
    )


def downgrade() -> None:
    # Reverse 5: restore the two-value vocabulary. Unlike the event_type CHECK below,
    # this one CANNOT just be re-narrowed: postgres validates a new CHECK against the
    # existing rows, so any surviving branch_base snapshot would abort the downgrade.
    # They are deleted first — the branch rows they served are dropped below anyway,
    # so a base with no branch is dead weight.
    op.execute("DELETE FROM storyboard_snapshots WHERE kind = 'branch_base'")
    op.drop_constraint(_SNAPSHOT_KIND_CONSTRAINT, "storyboard_snapshots", type_="check")
    op.create_check_constraint(
        _SNAPSHOT_KIND_CONSTRAINT,
        "storyboard_snapshots",
        "kind IN ('manual','auto')",
    )

    # Reverse 4: drop the widened CHECK, restore the pre-0052 vocabulary. Lossy only
    # if a 'merged' event exists (a real rollback with merged data has bigger
    # problems); mirrors 0050's 'restored' add.
    op.drop_constraint(_EVENT_TYPE_CONSTRAINT, "storyboard_panel_events", type_="check")
    op.create_check_constraint(
        _EVENT_TYPE_CONSTRAINT,
        "storyboard_panel_events",
        "event_type IN ('approved','unapproved','restored')",
    )

    # Reverse 3: drop RLS policy BEFORE the table (a policy cannot outlive its table).
    for stmt in _LATE_JOIN_BRANCHES_DOWNGRADE_STATEMENTS:
        op.execute(stmt)
    op.drop_index("idx_storyboard_branches_root", table_name="storyboard_branches")
    op.drop_table("storyboard_branches")

    # Reverse 2: the partial index goes BEFORE its column (dropping the column would
    # drop the index implicitly, but an explicit drop keeps the reversal symmetric and
    # readable in the SQL log).
    for table in reversed(_ORIGIN_ID_TABLES):
        op.drop_index(f"idx_{table}_origin", table_name=table)
        op.drop_column(table, "origin_id")

    # Reverse 1.
    op.drop_column("projects", "is_branch")
