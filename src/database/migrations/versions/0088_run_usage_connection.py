"""run_usage: which tenant connection (id + version) served the row (P7 / R3)

Adds two nullable columns to ``run_usage``:

    connection_id       uuid     the tenant provider connection that served the call
    connection_version  integer  the connection VERSION pinned for the run

Both NULL = the platform path (WM / gateway / billing-mirrored rows); both set =
an in-process tenant-slot call. ``ck_run_usage_connection_pair`` makes "one without
the other" unrepresentable. No FK: a connection can be offboarded/purged while its
historical usage must survive.

Backfill (data only, from what the in-process writers already stored in
``metadata``): rows with ``metadata->>'source' = 'orchestrator_in_process'`` whose
``metadata->>'connection_id'`` is a well-formed UUID AND whose
``metadata->>'connection_version'`` is a non-negative integer get both columns;
anything else stays NULL (never half-set, never guessed). Guarded to rows still
NULL, so a re-run writes no row. Prod ``run_usage`` is a few hundred rows (0083
measured 239 on 2026-10-09): one transaction, well under a second.

Drift-tolerant: ``ADD COLUMN IF NOT EXISTS``; the CHECK is dropped-if-present and
re-added AFTER the backfill (which only ever sets both or neither). Frozen literal
SQL; no application code.

Revision ID: 0088_run_usage_connection
Revises: 0087_p9_partner_keys
Create Date: 2026-10-13
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 25 characters.
revision = "0088_run_usage_connection"
down_revision = "0087_p9_partner_keys"
branch_labels = None
depends_on = None

_UPGRADE = (
    "ALTER TABLE run_usage ADD COLUMN IF NOT EXISTS connection_id uuid",
    "ALTER TABLE run_usage ADD COLUMN IF NOT EXISTS connection_version integer",
    "ALTER TABLE run_usage DROP CONSTRAINT IF EXISTS ck_run_usage_connection_pair",
)

_BACKFILL = (
    "UPDATE run_usage SET "
    "connection_id = (metadata->>'connection_id')::uuid, "
    "connection_version = (metadata->>'connection_version')::integer "
    "WHERE metadata->>'source' = 'orchestrator_in_process' "
    "AND metadata->>'connection_id' ~ "
    "'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$' "
    "AND metadata->>'connection_version' ~ '^[0-9]{1,9}$' "
    "AND connection_id IS NULL AND connection_version IS NULL",
)

_CONSTRAINTS = (
    # Drift: a hand-written half pair would fail the CHECK — clear it to unknown.
    "UPDATE run_usage SET connection_id = NULL, connection_version = NULL "
    "WHERE (connection_id IS NULL) <> (connection_version IS NULL)",
    "ALTER TABLE run_usage ADD CONSTRAINT ck_run_usage_connection_pair "
    "CHECK ((connection_id IS NULL) = (connection_version IS NULL))",
)

_DOWNGRADE = (
    "ALTER TABLE run_usage DROP CONSTRAINT IF EXISTS ck_run_usage_connection_pair",
    "ALTER TABLE run_usage DROP COLUMN IF EXISTS connection_version",
    "ALTER TABLE run_usage DROP COLUMN IF EXISTS connection_id",
)


def upgrade() -> None:
    for statement in (*_UPGRADE, *_BACKFILL, *_CONSTRAINTS):
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
