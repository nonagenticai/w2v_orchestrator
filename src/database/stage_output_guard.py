"""DB-level write-once guard for a terminal stage's ``stage_outputs.output``.

Until this shipped, "a completed stage's output is never overwritten" was a
convention enforced ONLY in Python: every writer in
``StageOutputRepository`` carries a ``status IN ('pending','running')`` WHERE
clause. The table itself had exactly two constraints —
``uq_stage_outputs_run_stage`` and NOT NULLs — so anything that did not go
through that repository (a repair script, a future handler, an interactive
``psql``, a retry that re-enters by a different path) could silently rewrite a
finished stage's result and the run would then report an output nobody
produced.

**Mechanism — a ``BEFORE UPDATE`` row trigger with a WHEN clause, nothing
wider.** The WHEN clause is the whole design:

    OLD.status IN (terminal) AND NEW.output IS DISTINCT FROM OLD.output

so the function is not even invoked unless the update has exactly the illegal
shape. Consequences, all deliberate:

* Only ``output`` freezes. ``status``, ``error``, ``attempts``,
  ``completed_at``, ``updated_at`` stay freely updatable on a terminal row —
  a CHECK constraint or a column-level grant could not express that.
* A non-terminal row is untouched: ``record_progress`` on a ``running`` stage
  is normal in-flight bookkeeping and must keep working.
* An UPDATE that leaves ``output`` byte-identical is not an overwrite and is
  allowed (``IS DISTINCT FROM`` on ``jsonb`` compares values, not text).
* INSERTs are not covered at all — ``run_setup.seed_stage_rows`` legitimately
  INSERTs a regenerated child run's preserved stages already ``completed``.

**The escape hatch, and why one is required.** The Studio editing surface has
four routes (``src/api/routes/runs/stages.py``) that mutate a *completed*
stage's output on purpose, at an authenticated + ownership-checked caller's
request: the element patch, the ``course_stitch`` chapter-title/hidden
override, the ``course_stitch`` thumbnail override, and the element restamp
(which also resets its downstream stages from ``completed`` back to
``pending`` with ``output = NULL``). A blanket freeze would take all four out.

They opt in per transaction with

    SET LOCAL w2v.allow_stage_output_edit = 'on'

``SET LOCAL`` — never a bare ``SET`` — because the app talks to postgres
through pgbouncer in transaction pooling mode: a session-scoped GUC would leak
the exemption onto whatever request got the connection next. Transaction scope
also means the restamp's downstream reset, issued in the same transaction, is
covered by the one opt-in.

The two apply paths (an Alembic migration for prod, a SQLAlchemy
``after_create`` registrar for ``create_all``-built test/dev schemas) read the
statements from HERE so they cannot drift; migration 0063 inlines a frozen copy
per this repo's freeze convention, and
``tests/database/test_stage_output_write_once.py`` asserts the copy still
matches.
"""

from __future__ import annotations

from sqlalchemy import DDL, MetaData, event

#: Stage statuses after which ``output`` is frozen. Mirrors the terminal half of
#: ``ck_stage_outputs_status`` (see migration 0059 for 'cancelled') and the
#: complement of ``StageOutputRepository._NON_TERMINAL``.
TERMINAL_STATUSES: tuple[str, ...] = ("completed", "failed", "skipped", "cancelled")

#: Transaction-scoped GUC a deliberate post-terminal edit sets to ``'on'``.
EDIT_GUC = "w2v.allow_stage_output_edit"

#: SQL to opt the CURRENT TRANSACTION out of the guard. ``SET LOCAL`` is
#: load-bearing under pgbouncer transaction pooling — see the module docstring.
ALLOW_EDIT_SQL = f"SET LOCAL {EDIT_GUC} = 'on'"

#: Custom SQLSTATE the guard raises with, so a caller (and the test) can assert
#: on the error identity rather than on message text. Class 'W2' is outside
#: every postgres-defined class, so it can never collide with a real DB error.
ERRCODE = "W2V01"

FUNCTION_NAME = "reject_terminal_stage_output_rewrite"
TRIGGER_NAME = "trg_stage_outputs_write_once"

_TERMINAL_SQL_LIST = ", ".join(f"'{s}'" for s in TERMINAL_STATUSES)

_CREATE_FUNCTION = f"""
CREATE OR REPLACE FUNCTION {FUNCTION_NAME}()
RETURNS trigger
LANGUAGE plpgsql
AS $guard$
BEGIN
    -- Deliberate, per-transaction opt-out (studio edits). current_setting with
    -- missing_ok=true yields NULL when the GUC was never set; a pooled
    -- connection whose SET LOCAL has ended reports '' instead. Neither is 'on'.
    IF coalesce(current_setting('{EDIT_GUC}', true), '') = 'on' THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION
        'stage_outputs.output is write-once: stage % of run % is already %, '
        'refusing to overwrite a terminal stage output',
        OLD.stage_name, OLD.run_id, OLD.status
        USING ERRCODE = '{ERRCODE}',
              HINT = 'A deliberate post-terminal edit must '
                     'SET LOCAL {EDIT_GUC} = ''on'' in the same transaction.';
END;
$guard$
""".strip()

_DROP_TRIGGER = f"DROP TRIGGER IF EXISTS {TRIGGER_NAME} ON stage_outputs"

_CREATE_TRIGGER = f"""
CREATE TRIGGER {TRIGGER_NAME}
BEFORE UPDATE ON stage_outputs
FOR EACH ROW
WHEN (
    OLD.status IN ({_TERMINAL_SQL_LIST})
    AND NEW.output IS DISTINCT FROM OLD.output
)
EXECUTE FUNCTION {FUNCTION_NAME}()
""".strip()

_DROP_FUNCTION = f"DROP FUNCTION IF EXISTS {FUNCTION_NAME}()"


def upgrade_statements() -> list[str]:
    """DDL that installs the guard. Idempotent: safe to replay."""
    return [_CREATE_FUNCTION, _DROP_TRIGGER, _CREATE_TRIGGER]


def downgrade_statements() -> list[str]:
    """DDL that removes the guard. Trigger first — the function is its dependency."""
    return [_DROP_TRIGGER, _DROP_FUNCTION]


def register_stage_output_write_once_ddl(metadata: MetaData) -> None:
    """Attach the guard to ``stage_outputs``' ``after_create`` so a
    ``create_all``-built schema (tests + dev) carries the SAME trigger migration
    0063 installs in prod.

    Without this the guard would exist only in prod and every DB-level test —
    which runs against a ``create_all`` schema, see ``tests/conftest.py`` — would
    be testing an ungated table.
    """
    table = metadata.tables.get("stage_outputs")
    if table is None:
        return
    for statement in upgrade_statements():
        # SQLAlchemy's DDL construct %-formats the statement against a context
        # dict before executing it, so the ``%`` placeholders in the trigger
        # function's RAISE would blow up with "%o format: an integer is
        # required". Doubling them here (and ONLY here) keeps
        # ``upgrade_statements()`` the canonical, migration-comparable text.
        event.listen(table, "after_create", DDL(statement.replace("%", "%%")))
