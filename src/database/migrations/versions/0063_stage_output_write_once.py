"""stage_outputs.output is write-once once the stage is terminal — DB-enforced

Before this, the write-once contract lived ONLY in Python: every writer in
``StageOutputRepository`` carries a ``status IN ('pending','running')`` WHERE
clause. The table's only constraints were ``uq_stage_outputs_run_stage`` and
NOT NULLs, so any path that did not go through that repository could silently
overwrite a finished stage's result and the run would report an output nobody
produced.

Installs a ``BEFORE UPDATE ... FOR EACH ROW WHEN (...)`` trigger. The WHEN
clause is scoped as narrowly as the invariant: it fires only when the row was
ALREADY terminal and the update actually changes ``output``. ``status``,
``error``, ``attempts`` and the timestamps stay updatable on a terminal row;
non-terminal rows are untouched (in-flight ``record_progress`` must keep
working); INSERTs are not covered (a regenerated child run legitimately
INSERTs preserved stages already ``completed``).

Escape hatch, required by four live Studio routes that edit a *completed*
stage's output on purpose (element patch, course-stitch chapter override,
course-stitch thumbnail override, element restamp + its downstream reset):
``SET LOCAL w2v.allow_stage_output_edit = 'on'`` in the same transaction.
``SET LOCAL``, not ``SET`` — pgbouncer runs transaction pooling, so a
session-scoped GUC would leak the exemption to the next request on that
connection.

Purely additive and non-destructive: no existing row is touched, no rewrite,
no lock beyond the brief ``CREATE TRIGGER`` ACCESS EXCLUSIVE. An OLDER image
that never sets the GUC boots fine against the migrated DB — it only ever
issues the non-terminal-guarded writes, which the trigger never sees.

⚠ The DDL below is a FROZEN literal copy of
``src.database.stage_output_guard.upgrade_statements()`` /
``downgrade_statements()``, per this repo's freeze convention (see
``tests/database/test_frozen_migration_ddl.py``): replaying a migration through
a live generator would emit whatever the generator says TODAY, so a later edit
would retroactively rewrite this migration's replayed output.
``tests/database/test_stage_output_write_once.py`` asserts the copy has not
drifted; if the guard must change, that is a NEW forward migration, not an edit
here.

Revision ID: 0063_stage_output_write_once
Revises: 0062_video_source_uploads
Create Date: 2026-08-20
"""

from alembic import op

revision = "0063_stage_output_write_once"
down_revision = "0062_video_source_uploads"
branch_labels = None
depends_on = None

_UPGRADE_STATEMENTS: tuple[str, ...] = (
    """CREATE OR REPLACE FUNCTION reject_terminal_stage_output_rewrite()
RETURNS trigger
LANGUAGE plpgsql
AS $guard$
BEGIN
    -- Deliberate, per-transaction opt-out (studio edits). current_setting with
    -- missing_ok=true yields NULL when the GUC was never set; a pooled
    -- connection whose SET LOCAL has ended reports '' instead. Neither is 'on'.
    IF coalesce(current_setting('w2v.allow_stage_output_edit', true), '') = 'on' THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION
        'stage_outputs.output is write-once: stage % of run % is already %, '
        'refusing to overwrite a terminal stage output',
        OLD.stage_name, OLD.run_id, OLD.status
        USING ERRCODE = 'W2V01',
              HINT = 'A deliberate post-terminal edit must '
                     'SET LOCAL w2v.allow_stage_output_edit = ''on'' in the same transaction.';
END;
$guard$""",
    "DROP TRIGGER IF EXISTS trg_stage_outputs_write_once ON stage_outputs",
    """CREATE TRIGGER trg_stage_outputs_write_once
BEFORE UPDATE ON stage_outputs
FOR EACH ROW
WHEN (
    OLD.status IN ('completed', 'failed', 'skipped', 'cancelled')
    AND NEW.output IS DISTINCT FROM OLD.output
)
EXECUTE FUNCTION reject_terminal_stage_output_rewrite()""",
)

_DOWNGRADE_STATEMENTS: tuple[str, ...] = (
    "DROP TRIGGER IF EXISTS trg_stage_outputs_write_once ON stage_outputs",
    "DROP FUNCTION IF EXISTS reject_terminal_stage_output_rewrite()",
)


def upgrade() -> None:
    for statement in _UPGRADE_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE_STATEMENTS:
        op.execute(statement)
