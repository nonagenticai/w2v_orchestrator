"""Contract guard: the ``failure_class`` vocabulary has THREE authorities.

They must all admit exactly the same four values:

1. the MIGRATED database's ``ck_runs_failure_class`` CHECK (what PROD enforces),
2. the ``FailureClass`` Literal (what the API serializes and what ``ty`` checks),
3. the ``FAILURE_CLASSES`` runtime tuple (what ``attribute_exception`` iterates).

This is not a hypothetical class of bug in this repo — it has already shipped
once on the neighbouring vocabulary. Migration 0059 widened
``ck_stage_outputs_status`` to admit ``'cancelled'`` and a writer started
producing it, but the ``StageStatus`` Literal was never widened, so every read of
an affected run died inside ``model_validate`` with a 500. Both directions of
drift are bugs:

* **DB wider than the Literal** — a writer persists a value the API cannot
  serialize (``RunDetail.failure_class``), i.e. the 500 above, on the run-detail
  view of exactly the failed runs somebody is trying to diagnose.
* **Literal wider than the DB** — the API advertises a class the CHECK rejects on
  write, so ``mark_failed`` raises ``IntegrityError`` and the run cannot be failed
  AT ALL. That is strictly worse than the read-side break: it turns a bad
  attribution into a stuck run.
* **Tuple out of step with the Literal** — ``attribute_exception`` matches a
  stamped attribute by iterating ``FAILURE_CLASSES``, so a member missing there is
  silently unreachable: the class is computed, discarded, and the run records
  ``'unknown'`` with nothing anywhere saying why.

⚠️ The DB authority is the **migration-replayed** schema, never the test database.
The suite's own database is built by ``conftest``'s ``_db_schema_initialized``
with ``Base.metadata.create_all`` — reading ``pg_constraint`` there measures
``models.py``, not prod. ``parity_schemas`` builds both, and its sibling
``test_check_constraints_match_between_models_and_migrations`` (in
``test_stage_status_check_constraint_parity.py``) already holds models ↔
migrations for every ``ck_*``, this one included.
"""

from __future__ import annotations

import re
from typing import get_args

import pytest

from src.api.schemas._shared import FailureClass as ApiFailureClass
from src.models.failure_class import FAILURE_CLASSES, FailureClass
from tests.support.schema_probe import ParityProbe, snapshot_check_constraints

_RUNS_TABLE = "runs"
_VOCAB_CONSTRAINT = "ck_runs_failure_class"
_ONLY_ON_FAILED_CONSTRAINT = "ck_runs_failure_class_only_on_failed"

# Matches each single-quoted literal inside e.g.
#   CHECK ((failure_class IS NULL) OR (failure_class = ANY (ARRAY['our_defect'::text, ...])))
_QUOTED = re.compile(r"'([^']*)'")


@pytest.mark.parametrize(
    ("source", "literal"),
    [
        ("src/models/failure_class.py (canonical)", FailureClass),
        ("src/api/schemas/_shared.py (API export)", ApiFailureClass),
    ],
)
def test_failure_class_literal_matches_migrated_check_constraint(
    source: str,
    literal: object,
    parity_schemas: ParityProbe,
) -> None:
    """Each Python authority must admit exactly what the MIGRATED (prod) CHECK admits.

    Run against BOTH the canonical Literal and the API package's re-export. The
    re-export is an alias today, and asserting ``ApiFailureClass is FailureClass``
    would be a VACUOUS way to pin that: ``typing`` interns Literal, so an
    identical hand-retyped copy in ``_shared.py`` compares ``is``-equal to the
    original and such a test passes on the broken code (verified — that is
    exactly how the first version of this guard failed its own mutation). Testing
    the MEMBERS of each authority against the database is the property that
    actually matters, and it kills a retyped copy the moment it diverges.
    """
    probe = parity_schemas
    with probe.engine.connect() as conn:
        checks = snapshot_check_constraints(conn, probe.alembic)

    definition = checks.get((_RUNS_TABLE, _VOCAB_CONSTRAINT))
    assert definition is not None, (
        f"CHECK constraint {_VOCAB_CONSTRAINT!r} not found on {_RUNS_TABLE!r} in the "
        "migration-replayed schema. Either it was renamed/dropped (update this guard "
        "and the Literal together) or the chain no longer builds it — a missing "
        "constraint means runs.failure_class is UNVALIDATED in prod, so a typo'd "
        "attribution persists silently and the whole column stops being trustworthy.\n"
        f"CHECK constraints found: {sorted(checks)}"
    )

    db_values = set(_QUOTED.findall(definition))
    assert db_values, f"could not parse any quoted values out of: {definition!r}"

    literal_values = set(get_args(literal))

    assert db_values == literal_values, (
        "failure-class drift between the migrated database and the Python type.\n"
        f"  authority checked:        {source}\n"
        f"  CHECK {_VOCAB_CONSTRAINT}: {sorted(db_values)}\n"
        f"  FailureClass Literal:     {sorted(literal_values)}\n"
        f"  writable-but-unreadable (DB only): {sorted(db_values - literal_values)}\n"
        f"  advertised-but-unwritable (Literal only): {sorted(literal_values - db_values)}\n"
        "A DB-only value 500s every read of a run carrying it; a Literal-only value "
        "makes mark_failed raise IntegrityError, which leaves the run UNFAILABLE. "
        "Widen the CHECK (a new migration), src/models/failure_class.py and "
        "Run.__table_args__ in the SAME PR."
    )


def test_failure_classes_tuple_matches_the_literal() -> None:
    """The runtime tuple and the static Literal are the same vocabulary.

    ``FAILURE_CLASSES`` is spelled out rather than derived with ``get_args`` so
    ``ty`` keeps each element at its literal type. That is a deliberate
    duplication, and this is the guard that makes it safe: a member added to one
    and not the other is silently unreachable in ``attribute_exception``, which
    iterates the tuple.
    """
    assert set(FAILURE_CLASSES) == set(get_args(FailureClass)), (
        "FAILURE_CLASSES and the FailureClass Literal disagree.\n"
        f"  tuple only:   {sorted(set(FAILURE_CLASSES) - set(get_args(FailureClass)))}\n"
        f"  Literal only: {sorted(set(get_args(FailureClass)) - set(FAILURE_CLASSES))}\n"
        "attribute_exception matches stamped attributes by iterating the tuple, so a "
        "Literal-only member can never be returned — the classification is computed "
        "and then thrown away."
    )
    assert len(FAILURE_CLASSES) == len(set(FAILURE_CLASSES)), (
        f"FAILURE_CLASSES contains duplicates: {FAILURE_CLASSES}"
    )


def test_failure_class_is_constrained_to_failed_runs(
    parity_schemas: ParityProbe,
) -> None:
    """A fault attribution must be impossible on a run that did not fail.

    Without this CHECK the column could carry an attribution for a completed or
    cancelled run, which makes every aggregate over it wrong in a way no reader
    would suspect (``GROUP BY failure_class`` silently counting successes).
    """
    probe = parity_schemas
    with probe.engine.connect() as conn:
        checks = snapshot_check_constraints(conn, probe.alembic)

    definition = checks.get((_RUNS_TABLE, _ONLY_ON_FAILED_CONSTRAINT))
    assert definition is not None, (
        f"CHECK constraint {_ONLY_ON_FAILED_CONSTRAINT!r} is missing from the "
        "migration-replayed schema; a non-failed run could carry a fault "
        "attribution.\n"
        f"CHECK constraints found: {sorted(checks)}"
    )
    assert "failed" in definition, (
        f"{_ONLY_ON_FAILED_CONSTRAINT} no longer references the 'failed' status — it "
        f"cannot be enforcing what its name claims: {definition!r}"
    )
