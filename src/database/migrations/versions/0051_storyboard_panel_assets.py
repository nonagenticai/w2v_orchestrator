"""storyboard_panel_assets — normalize panel→asset refs into a real join table

``storyboard_panels`` carried three *undeclared* foreign keys to
``storyboard_assets`` (0046 L138-150): ``characters`` / ``props`` (JSONB arrays of
asset-id strings) and ``location_asset_id`` (a bare ``uuid``, no FK). Deleting an
asset silently gutted every panel referencing it. This migration replaces all three
with ``storyboard_panel_assets``:

* ``panel_id`` FK → ``storyboard_panels`` **ON DELETE CASCADE**; ``asset_id`` FK →
  ``storyboard_assets`` **ON DELETE NO ACTION DEFERRABLE INITIALLY IMMEDIATE**.
  NOT ``RESTRICT``: Postgres checks RESTRICT immediately and *cannot* defer it even
  under ``SET CONSTRAINTS ALL DEFERRED``, which is exactly what
  ``storyboard_snapshot_service.restore()`` does — RESTRICT would 500 every restore
  of a snapshot older than an asset-attach. ``NO ACTION`` honours the deferral; with
  no ``SET CONSTRAINTS`` in scope (the normal delete path) ``INITIALLY IMMEDIATE``
  still raises ``23503`` at the DELETE statement, which is the 409 signal.
* ``role`` is ``Text`` + ``CheckConstraint`` (no native pg ENUM — same convention as
  0046), ``UNIQUE(panel_id, asset_id, role)``, plus a partial unique on ``panel_id``
  ``WHERE role = 'location'`` so location stays scalar.
* RLS: ENABLE (not FORCE) + a join-based ``tenant_isolation`` policy — a **2-hop**
  climb ``panel_id → storyboard_panels.project_id → projects.tenant_id`` (the join
  table has no ``project_id`` of its own). Sourced from ``src.database.rls``
  (``late_join_panel_assets_*_statements``) so this migration and the ``create_all``
  ``after_create`` path can't drift, and emitted here directly — the earlier
  ``_LATE`` / ``_LATE_EVENTS`` / ``_SNAPSHOTS`` maps belong to 0046/0049/0050, which
  iterate them at run time *before* this table exists.

Runtime-role table privileges ride ``ALTER DEFAULT PRIVILEGES`` (gitops), like every
other table — no GRANT here (identical to 0046/0049/0050).

The backfill stages every raw ref into a temp table and then drops the **dangling**
ones, loudly (``RAISE NOTICE`` per triple). ``characters`` is ``list[str]`` and the
API accepts arbitrary strings today (``["hero"]``), so non-UUID-shaped values exist
in prod; they cannot satisfy the new FK. The dangling set is computed in **two
separate statements** — shape-invalid first (deleted out of the staging table), then
absent-from-``storyboard_assets`` over the shape-valid remainder — because Postgres
does not guarantee ``OR``/``AND`` evaluation order and a ``::uuid`` cast against a
non-UUID string would abort the whole migration.

Revision ID: 0051
Revises: 0050
Create Date: 2026-07-13
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "0051"
down_revision = "0050"
branch_labels = None
depends_on = None

# FROZEN DDL — inlined snapshot of ``src.database.rls.late_join_panel_assets_downgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_LATE_JOIN_PANEL_ASSETS_DOWNGRADE_STATEMENTS: tuple[str, ...] = (
    "DROP POLICY IF EXISTS tenant_isolation ON storyboard_panel_assets",
    "ALTER TABLE storyboard_panel_assets DISABLE ROW LEVEL SECURITY",
)


# FROZEN DDL — inlined snapshot of ``src.database.rls.late_join_panel_assets_upgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_LATE_JOIN_PANEL_ASSETS_UPGRADE_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE storyboard_panel_assets ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON storyboard_panel_assets USING (EXISTS (SELECT 1 FROM storyboard_panels sp JOIN projects p ON p.id = sp.project_id WHERE sp.id = storyboard_panel_assets.panel_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM storyboard_panels sp JOIN projects p ON p.id = sp.project_id WHERE sp.id = storyboard_panel_assets.panel_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
)


# A static, self-contained DDL/DML script — no interpolation, no parameters, nothing
# caller-supplied (the uuid shape regex is inlined below rather than f-string-spliced,
# so ruff's S608 has nothing to flag and there is no `noqa` to explain away).
# The regex is the canonical uuid text shape, matched case-insensitively (``~*``): a
# value that IS castable to uuid must never be misfiled as corruption.
_BACKFILL = """
DO $$
DECLARE
    v_dangling integer;
    r record;
BEGIN
    CREATE TEMP TABLE _pa_stage (
        panel_id uuid NOT NULL,
        role     text NOT NULL,
        raw      text NOT NULL,
        position integer NOT NULL
    ) ON COMMIT DROP;

    CREATE TEMP TABLE _pa_dangling (
        panel_id uuid NOT NULL,
        role     text NOT NULL,
        raw      text NOT NULL
    ) ON COMMIT DROP;

    -- characters / props: ordinality-1 becomes the position. The array guard lives
    -- INSIDE the lateral (a set-returning function in FROM is evaluated before the
    -- WHERE clause, so a WHERE-side jsonb_typeof guard would not protect the call).
    INSERT INTO _pa_stage (panel_id, role, raw, position)
    SELECT p.id, 'character', e.value #>> '{}', (e.ord - 1)::integer
    FROM storyboard_panels p
    CROSS JOIN LATERAL jsonb_array_elements(
        CASE WHEN jsonb_typeof(p.characters) = 'array' THEN p.characters ELSE '[]'::jsonb END
    ) WITH ORDINALITY AS e(value, ord)
    WHERE jsonb_typeof(e.value) = 'string';

    INSERT INTO _pa_stage (panel_id, role, raw, position)
    SELECT p.id, 'prop', e.value #>> '{}', (e.ord - 1)::integer
    FROM storyboard_panels p
    CROSS JOIN LATERAL jsonb_array_elements(
        CASE WHEN jsonb_typeof(p.props) = 'array' THEN p.props ELSE '[]'::jsonb END
    ) WITH ORDINALITY AS e(value, ord)
    WHERE jsonb_typeof(e.value) = 'string';

    -- location: already a uuid column, so always shape-valid; one row, position 0.
    INSERT INTO _pa_stage (panel_id, role, raw, position)
    SELECT p.id, 'location', p.location_asset_id::text, 0
    FROM storyboard_panels p
    WHERE p.location_asset_id IS NOT NULL;

    -- (a) shape-invalid refs. DELETED out of the staging table, so that (b) below
    -- can cast unconditionally — no statement in this block ever casts a string it
    -- has not already shape-checked in a PRIOR statement.
    WITH bad AS (
        DELETE FROM _pa_stage s
        WHERE s.raw !~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        RETURNING s.panel_id, s.role, s.raw
    )
    INSERT INTO _pa_dangling (panel_id, role, raw)
    SELECT panel_id, role, raw FROM bad;

    -- (b) shape-valid but absent from storyboard_assets.
    WITH bad AS (
        DELETE FROM _pa_stage s
        WHERE NOT EXISTS (
            SELECT 1 FROM storyboard_assets a WHERE a.id = s.raw::uuid
        )
        RETURNING s.panel_id, s.role, s.raw
    )
    INSERT INTO _pa_dangling (panel_id, role, raw)
    SELECT panel_id, role, raw FROM bad;

    SELECT count(*) INTO v_dangling FROM _pa_dangling;
    IF v_dangling > 0 THEN
        RAISE NOTICE
            'storyboard_panel_assets backfill: dropping % dangling panel->asset '
            'reference(s) (pre-existing corruption; cannot satisfy the new FK)',
            v_dangling;
        FOR r IN
            SELECT panel_id, role, raw FROM _pa_dangling ORDER BY panel_id, role, raw
        LOOP
            RAISE NOTICE '  dangling ref: panel_id=% role=% asset_id=%',
                r.panel_id, r.role, r.raw;
        END LOOP;
    END IF;

    -- DISTINCT ON dedupes a same-asset-twice array, which would otherwise abort on
    -- uq_storyboard_panel_assets_triple. Lowest position wins.
    INSERT INTO storyboard_panel_assets (id, panel_id, asset_id, role, position, created_at)
    SELECT DISTINCT ON (s.panel_id, s.raw::uuid, s.role)
        gen_random_uuid(), s.panel_id, s.raw::uuid, s.role, s.position, now()
    FROM _pa_stage s
    ORDER BY s.panel_id, s.raw::uuid, s.role, s.position;
END
$$;
"""

_REMATERIALIZE = """
UPDATE storyboard_panels p
SET characters = COALESCE(
        (
            SELECT jsonb_agg(pa.asset_id::text ORDER BY pa.position)
            FROM storyboard_panel_assets pa
            WHERE pa.panel_id = p.id AND pa.role = 'character'
        ),
        '[]'::jsonb
    ),
    props = COALESCE(
        (
            SELECT jsonb_agg(pa.asset_id::text ORDER BY pa.position)
            FROM storyboard_panel_assets pa
            WHERE pa.panel_id = p.id AND pa.role = 'prop'
        ),
        '[]'::jsonb
    ),
    location_asset_id = (
        SELECT pa.asset_id
        FROM storyboard_panel_assets pa
        WHERE pa.panel_id = p.id AND pa.role = 'location'
        ORDER BY pa.position
        LIMIT 1
    );
"""


def upgrade() -> None:
    op.create_table(
        "storyboard_panel_assets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "panel_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("storyboard_panels.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "asset_id",
            postgresql.UUID(as_uuid=True),
            # NO ACTION (not RESTRICT) so snapshot restore's SET CONSTRAINTS ALL
            # DEFERRED can actually defer the check; INITIALLY IMMEDIATE keeps every
            # other write path checking per-statement (→ 23503 → 409).
            sa.ForeignKey(
                "storyboard_assets.id",
                ondelete="NO ACTION",
                deferrable=True,
                initially="IMMEDIATE",
            ),
            nullable=False,
        ),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "role IN ('character','prop','location')",
            name="ck_storyboard_panel_assets_role",
        ),
        sa.UniqueConstraint(
            "panel_id",
            "asset_id",
            "role",
            name="uq_storyboard_panel_assets_triple",
        ),
    )
    # Location is scalar on the wire (UUID | None) — one location row per panel.
    op.create_index(
        "uq_storyboard_panel_assets_one_location",
        "storyboard_panel_assets",
        ["panel_id"],
        unique=True,
        postgresql_where=sa.text("role = 'location'"),
    )
    # The FK check and the 409 "which panels use this asset" listing both scan by asset_id.
    op.create_index(
        "idx_storyboard_panel_assets_asset",
        "storyboard_panel_assets",
        ["asset_id"],
    )
    op.create_index(
        "idx_storyboard_panel_assets_panel_role",
        "storyboard_panel_assets",
        ["panel_id", "role", "position"],
    )

    # RLS: ENABLE-not-FORCE + the 2-hop join-based tenant_isolation policy (sourced
    # frozen from rls.py; drift is caught by tests/database/test_frozen_migration_ddl.py).
    for stmt in _LATE_JOIN_PANEL_ASSETS_UPGRADE_STATEMENTS:
        op.execute(stmt)

    # Backfill BEFORE the columns are dropped — same transaction.
    op.execute(_BACKFILL)

    op.drop_column("storyboard_panels", "characters")
    op.drop_column("storyboard_panels", "props")
    op.drop_column("storyboard_panels", "location_asset_id")


def downgrade() -> None:
    op.add_column(
        "storyboard_panels",
        sa.Column(
            "characters",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "storyboard_panels",
        sa.Column(
            "props",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "storyboard_panels",
        sa.Column("location_asset_id", postgresql.UUID(as_uuid=True), nullable=True),
    )

    # Re-materialize the legacy columns from the join table. Lossy only for the
    # dangling refs the upgrade already dropped — correct, they were corruption.
    op.execute(_REMATERIALIZE)

    for stmt in _LATE_JOIN_PANEL_ASSETS_DOWNGRADE_STATEMENTS:
        op.execute(stmt)

    op.drop_index("idx_storyboard_panel_assets_panel_role", table_name="storyboard_panel_assets")
    op.drop_index("idx_storyboard_panel_assets_asset", table_name="storyboard_panel_assets")
    op.drop_index("uq_storyboard_panel_assets_one_location", table_name="storyboard_panel_assets")
    op.drop_table("storyboard_panel_assets")
