"""run_usage: per-user / per-slot attribution columns (P6 ORCH-3)

Adds five nullable columns to ``run_usage``:

    slot        text     provider slot of an in-process (tenant BYO) call; NULL = platform path
    subject_id  uuid     who the usage is attributed to (the run's subject)
    funding     text     local_gpu | platform_external | tenant_byo; NULL = not known
    units       numeric  billable quantity in ``unit_kind``
    unit_kind   text     tokens | chars | images | seconds

plus ``idx_run_usage_subject (subject_id)``, ``ck_run_usage_unit_kind`` and
``ck_run_usage_funding`` (the agreed taxonomy or NULL).

Backfill (data only, from values already stored — no guessing):

* in-process rows (``metadata->>'source' = 'orchestrator_in_process'``): ``slot``,
  ``subject_id`` from ``metadata``, ``funding`` from ``metadata`` only if it is a
  taxonomy value (else NULL); TTS rows (``metadata->>'unit' =
  'characters'``) get ``units = metadata->>'characters'``, ``unit_kind = 'chars'``;
  LLM rows get ``units = input + output tokens``, ``unit_kind = 'tokens'``.
* billing-mirrored rows (everything else): ``subject_id`` from the run's
  ``config_snapshot->>'subject_id'``; ``funding`` and ``slot`` stay NULL — billing's
  per-run cost carries no funding dimension (a gateway/WM call may be tenant BYO), so
  it is reported "unattributed", never guessed;
  rows with tokens get ``units``/``unit_kind = 'tokens'``.

A value that is not a well-formed UUID is left NULL instead of failing the cast. Any
row the backfill cannot attribute keeps NULL and the API reports it as
"unattributed" — never dropped. Every UPDATE touches only rows it has not processed
(its target columns still NULL), so a re-run (migrated but unstamped) writes no row.
Not batched: prod ``run_usage`` held 239 rows (608 kB) on 2026-10-09, so the whole
upgrade — which alembic runs in ONE transaction, holding the ADD COLUMN's ACCESS
EXCLUSIVE lock through the backfill — finishes in well under a second.

Drift-tolerant: ``ADD COLUMN IF NOT EXISTS``, ``CREATE INDEX IF NOT EXISTS``, the
CHECKs are dropped-if-present then re-added. Frozen literal SQL; no application code.

Revision ID: 0083_run_usage_attribution
Revises: 0082_p6_audit_actions
Create Date: 2026-10-09
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 26 characters.
revision = "0083_run_usage_attribution"
down_revision = "0082_p6_audit_actions"
branch_labels = None
depends_on = None

_UPGRADE = (
    "ALTER TABLE run_usage ADD COLUMN IF NOT EXISTS slot text",
    "ALTER TABLE run_usage ADD COLUMN IF NOT EXISTS subject_id uuid",
    "ALTER TABLE run_usage ADD COLUMN IF NOT EXISTS funding text",
    "ALTER TABLE run_usage ADD COLUMN IF NOT EXISTS units numeric",
    "ALTER TABLE run_usage ADD COLUMN IF NOT EXISTS unit_kind text",
    "ALTER TABLE run_usage DROP CONSTRAINT IF EXISTS ck_run_usage_unit_kind",
    "ALTER TABLE run_usage ADD CONSTRAINT ck_run_usage_unit_kind "
    "CHECK (unit_kind IS NULL OR unit_kind IN ('tokens','chars','images','seconds'))",
    "ALTER TABLE run_usage DROP CONSTRAINT IF EXISTS ck_run_usage_funding",
    # Drift: a DB that ran an earlier draft of this revision holds the retired
    # 'platform' label (a guess) — clear any non-taxonomy value to NULL (unknown).
    "UPDATE run_usage SET funding = NULL WHERE funding IS NOT NULL "
    "AND funding NOT IN ('local_gpu','platform_external','tenant_byo')",
    "ALTER TABLE run_usage ADD CONSTRAINT ck_run_usage_funding "
    "CHECK (funding IS NULL OR funding IN ('local_gpu','platform_external','tenant_byo'))",
    "CREATE INDEX IF NOT EXISTS idx_run_usage_subject ON run_usage (subject_id)",
)

_BACKFILL = (
    # In-process rows: identity + slot + funding from the metadata the writers stored.
    "UPDATE run_usage SET "
    "slot = COALESCE(slot, metadata->>'slot'), "
    "funding = COALESCE(funding, CASE WHEN metadata->>'funding' IN "
    "('local_gpu','platform_external','tenant_byo') THEN metadata->>'funding' END), "
    "subject_id = COALESCE(subject_id, CASE WHEN metadata->>'subject_id' ~ "
    "'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$' THEN (metadata->>'subject_id')::uuid END) "
    "WHERE metadata->>'source' = 'orchestrator_in_process' "
    # Only rows no backfill/writer has attributed yet: a re-run rewrites nothing.
    "AND slot IS NULL AND funding IS NULL AND subject_id IS NULL",
    # In-process TTS: characters.
    "UPDATE run_usage SET units = (metadata->>'characters')::numeric, unit_kind = 'chars' "
    "WHERE units IS NULL AND metadata->>'source' = 'orchestrator_in_process' "
    "AND metadata->>'unit' = 'characters' AND metadata->>'characters' ~ '^[0-9]+$'",
    # Billing-mirrored rows: the run's subject only. funding stays NULL (unknown).
    "UPDATE run_usage u SET "
    "subject_id = (r.config_snapshot->>'subject_id')::uuid "
    "FROM runs r WHERE r.id = u.run_id "
    "AND u.metadata->>'source' IS DISTINCT FROM 'orchestrator_in_process' "
    # Only rows still unattributed whose run pins a well-formed subject: a re-run
    # (or a run with no/malformed subject) writes no row.
    "AND u.subject_id IS NULL AND r.config_snapshot->>'subject_id' ~ "
    "'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'",
    # Any row that carries tokens and has no unit yet: tokens.
    "UPDATE run_usage SET units = input_tokens + output_tokens, unit_kind = 'tokens' "
    "WHERE units IS NULL AND input_tokens + output_tokens > 0",
)

_DOWNGRADE = (
    "DROP INDEX IF EXISTS idx_run_usage_subject",
    "ALTER TABLE run_usage DROP CONSTRAINT IF EXISTS ck_run_usage_unit_kind",
    "ALTER TABLE run_usage DROP CONSTRAINT IF EXISTS ck_run_usage_funding",
    "ALTER TABLE run_usage DROP COLUMN IF EXISTS unit_kind",
    "ALTER TABLE run_usage DROP COLUMN IF EXISTS units",
    "ALTER TABLE run_usage DROP COLUMN IF EXISTS funding",
    "ALTER TABLE run_usage DROP COLUMN IF EXISTS subject_id",
    "ALTER TABLE run_usage DROP COLUMN IF EXISTS slot",
)


def upgrade() -> None:
    for statement in (*_UPGRADE, *_BACKFILL):
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
