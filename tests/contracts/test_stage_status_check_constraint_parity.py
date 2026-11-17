"""Contract guards around the stage-status vocabulary.

Three things must agree, and each disagreement is a shipped bug:

1. **The MIGRATED database's ``ck_stage_outputs_status`` CHECK** ↔
   ``src/api/schemas/_shared.py``'s ``StageStatus`` Literal.
2. **Every ``ck_*`` CHECK constraint** in the ``create_all`` (models) schema ↔
   the ``alembic upgrade head`` (prod) schema.
3. **Every terminal ``StageStatus``** ↔ the SSE replay's
   ``_stage_terminal_event`` vocabulary.

(1) is the parity that broke in prod. Migration 0059 widened
``ck_stage_outputs_status`` to admit ``'cancelled'`` and
``StageOutputRepository.bulk_cancel_non_terminal`` started writing it, but the
``StageStatus`` Literal was never widened. Every read of an affected run then
died inside ``StageOutputResponse.model_validate`` with
``input_value='cancelled'`` → HTTP 500. Set algebra over the two authorities
catches the whole class, and a drift in EITHER direction is a bug:

- **DB wider than the schema** — a writer can persist a row the API cannot
  serialize (the 500 above).
- **Schema wider than the DB** — the API advertises a status the DB will reject
  with an ``IntegrityError`` on write, so the value is a dead branch.

⚠️ The authority for (1) is the **migration-replayed** schema, NOT the test
database. The test database is built by ``tests/conftest.py``'s
``_db_schema_initialized`` with ``Base.metadata.create_all`` — i.e. from
``src/database/models.py``. Reading ``pg_constraint`` there measures the MODELS,
not prod. The original version of this guard did exactly that, which left the
bug it was written to prevent fully reproducible: a migration that widens a CHECK
without touching ``models.py`` keeps the guard green while prod diverges. Nothing
else covered that either — ``alembic check``
(``tests/database/test_migration_replay_check.py``) uses autogenerate, which does
**not** detect CHECK constraints, and ``test_schema_parity`` compares only
``pg_policies`` + ``pg_proc``.

(2) is the generalisation that makes the models-vs-migrations gap structural
rather than one-column-deep: it holds for every CHECK in the schema, so the next
``models.py``-only or migration-only CHECK fails here rather than in prod. It is
what caught ``ck_verbatim_items_kind`` / ``ck_verbatim_items_scope`` living in
migration 0017 but never in ``VerbatimItem.__table_args__`` — prod rejected a bad
``kind``, the test database happily accepted it.

(3) closes the replay side of the same class. ``_stage_terminal_event`` returns
``None`` for any status it does not recognise, while its caller appends
``stage_started`` unconditionally for every non-``pending`` row. So a 7th status
— added to the CHECK and the Literal, keeping (1) and (2) green — would replay
``stage_started`` with no terminal event, sticking the UI badge on *running*
forever: the ghost-pending stage the hard-fail sweep to ``'cancelled'`` exists to
eliminate. Asserting handler coverage structurally means a future author cannot
forget.

Both DB-backed tests take the session-scoped ``parity_schemas`` fixture, which
FAILS loudly on an unreachable database via ``_fail_or_skip_without_database``.
Neither catches ``OperationalError`` to skip.
"""

from __future__ import annotations

import re
from typing import get_args

from src.api.routes.runs.events import _stage_terminal_event
from src.api.schemas._shared import StageStatus
from tests.support.schema_probe import ParityProbe, snapshot_check_constraints

_STAGE_OUTPUTS_TABLE = "stage_outputs"
_CONSTRAINT_NAME = "ck_stage_outputs_status"

# Matches each single-quoted literal inside e.g.
#   CHECK ((status = ANY (ARRAY['pending'::text, 'running'::text, ...])))
_QUOTED = re.compile(r"'([^']*)'")

# The two statuses that are, by definition, NOT terminal: a 'pending' stage
# replays nothing at all and a 'running' one replays only `stage_started`.
# EVERY other StageStatus must produce a terminal SSE event.
_NON_TERMINAL_STATUSES = frozenset({"pending", "running"})


def test_stage_status_literal_matches_migrated_check_constraint(
    parity_schemas: ParityProbe,
) -> None:
    """``StageStatus`` must admit exactly what the MIGRATED (prod) CHECK admits."""
    probe = parity_schemas
    with probe.engine.connect() as conn:
        checks = snapshot_check_constraints(conn, probe.alembic)

    definition = checks.get((_STAGE_OUTPUTS_TABLE, _CONSTRAINT_NAME))
    assert definition is not None, (
        f"CHECK constraint {_CONSTRAINT_NAME!r} not found on {_STAGE_OUTPUTS_TABLE!r} in "
        "the migration-replayed schema. Either the constraint was renamed/dropped "
        "(update this guard and the API Literal together) or the chain no longer "
        "builds it — a missing constraint means stage_outputs.status is unvalidated "
        "in PROD, which is worse than the drift this test hunts.\n"
        f"CHECK constraints found: {sorted(checks)}"
    )

    db_values = set(_QUOTED.findall(definition))
    assert db_values, f"could not parse any quoted status values out of: {definition!r}"

    schema_values = set(get_args(StageStatus))

    assert db_values == schema_values, (
        "stage-status drift between the migrated database and the API schema.\n"
        f"  CHECK {_CONSTRAINT_NAME}: {sorted(db_values)}\n"
        f"  StageStatus Literal:      {sorted(schema_values)}\n"
        f"  writable-but-unreadable (DB only): {sorted(db_values - schema_values)}\n"
        f"  advertised-but-unwritable (schema only): {sorted(schema_values - db_values)}\n"
        "A DB-only value 500s every read of a row carrying it "
        "(StageOutputResponse.model_validate); a schema-only value is a dead "
        "branch the DB rejects on write. Widen both, in the same PR."
    )


def test_check_constraints_match_between_models_and_migrations(
    parity_schemas: ParityProbe,
) -> None:
    """Every CHECK constraint must be identical in the models and the migrations.

    A migration-only CHECK means prod enforces an invariant the test database does
    not: a bad write passes the whole suite and raises ``IntegrityError`` in prod.
    A models-only CHECK means the reverse — tests reject a write prod accepts, so
    the suite is stricter than reality and the invariant is unprotected where it
    matters. Both directions fail here.
    """
    probe = parity_schemas
    with probe.engine.connect() as conn:
        models_checks = snapshot_check_constraints(conn, probe.create_all)
        migrated_checks = snapshot_check_constraints(conn, probe.alembic)

    # Guard against a silent both-empty pass (e.g. a build that no-oped).
    assert models_checks, "create_all produced no CHECK constraints — build no-oped?"
    assert migrated_checks, "alembic replay produced no CHECK constraints — build no-oped?"

    assert set(models_checks) == set(migrated_checks), (
        "CHECK-constraint SET diverges between models.py and the migration chain.\n"
        f"  only in models.py (create_all): {sorted(set(models_checks) - set(migrated_checks))}\n"
        f"  only in migrations (prod):      {sorted(set(migrated_checks) - set(models_checks))}\n"
        "A migration-only CHECK is enforced in prod but not in tests (a bad write "
        "ships green and 500s live); a models-only CHECK is enforced in tests but "
        "not in prod (the invariant is unprotected where it matters). Add the "
        "constraint to whichever side is missing it — copy the predicate verbatim."
    )

    for key in sorted(models_checks):
        assert models_checks[key] == migrated_checks[key], (
            f"CHECK constraint {key} has a DIFFERENT predicate in models.py than in "
            "the migration chain — the two schemas admit different values.\n"
            f"  models.py (create_all): {models_checks[key]}\n"
            f"  migrations (prod):      {migrated_checks[key]}\n"
            "Note postgres renders an ``IN (...)`` list in DECLARATION ORDER, so the "
            "value order must match too — this repo's convention is that a model "
            "predicate is a byte-for-byte mirror of the migration's. Copy, don't retype."
        )


def test_every_terminal_stage_status_replays_a_terminal_sse_event() -> None:
    """Each non-pending/running ``StageStatus`` must map to a terminal SSE event.

    ``_build_run_event_snapshot`` appends ``stage_started`` for every non-pending
    stage row unconditionally, so an unhandled status replays a start with no
    finish and the consumer's badge sticks on *running* forever.
    """
    terminal_statuses = sorted(set(get_args(StageStatus)) - _NON_TERMINAL_STATUSES)
    assert terminal_statuses, "StageStatus has no terminal values — Literal gutted?"

    unhandled: list[str] = []
    for status in terminal_statuses:
        event = _stage_terminal_event("run-id", "some_stage", status, "boom")
        if event is None:
            unhandled.append(status)
            continue
        assert event.get("event"), f"terminal event for {status!r} has no 'event' key: {event}"
        assert event.get("run_id") == "run-id", f"{status!r} event lost run_id: {event}"
        assert event.get("stage_name") == "some_stage", f"{status!r} event lost stage_name: {event}"

    assert not unhandled, (
        "_stage_terminal_event returns None for these terminal stage statuses: "
        f"{unhandled}\n"
        "The SSE catch-up replay appends 'stage_started' for EVERY non-pending "
        "stage row, so a status with no terminal event leaves the client's badge "
        "stuck on 'running' after a reconnect — the ghost-pending stage bug. Add "
        "a branch to _stage_terminal_event in src/api/routes/runs/events.py "
        "(and a matching live publish site) for each status listed."
    )


def test_non_terminal_stage_statuses_replay_no_terminal_event() -> None:
    """The converse: 'pending'/'running' must NOT produce a terminal event.

    Without this, a future edit could satisfy the coverage test above by mapping
    everything to some event — including 'running', which would replay a finish
    for a stage that is still going and freeze a live badge as done.
    """
    for status in sorted(_NON_TERMINAL_STATUSES):
        assert _stage_terminal_event("run-id", "some_stage", status, None) is None, (
            f"_stage_terminal_event returned a TERMINAL event for {status!r}; a stage "
            "in that state has not finished, so replaying one would mark a live "
            "stage as done in the UI."
        )
