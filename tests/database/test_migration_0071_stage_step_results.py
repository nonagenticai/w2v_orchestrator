"""Migration 0071 (``stage_step_results``, the step ledger) — frozen-DDL + parity.

Three claims, each of which can fail independently:

1. The RLS text FROZEN inside the revision still equals what the live
   ``src.database.rls`` generators emit — so the migration-built (prod) table and
   the ``create_all``-built (test) table carry the same policy. DB-free.
2. That policy is the ``stage_outputs`` run-join, table name aside — the ledger
   must not invent its own tenant climb. DB-free.
3. ``alembic upgrade head`` and ``create_all`` build the SAME table: columns,
   nullability, the unique key and the cascade. Needs the test database.

(The repo-wide "no revision imports live rls" guard and the policy-surface parity
gate already cover this revision generically — ``test_frozen_migration_ddl.py`` and
``test_schema_parity.py``.)
"""

from __future__ import annotations

from sqlalchemy import text

from src.database import rls
from tests.support.schema_probe import ParityProbe

_TABLE = "stage_step_results"


def test_policy_is_the_stage_outputs_run_join() -> None:
    ledger = rls.RLS_JOIN_PREDICATES_STAGE_STEP_RESULTS[_TABLE]
    sibling = rls.RLS_JOIN_PREDICATES["stage_outputs"]

    assert ledger == sibling.replace("stage_outputs.run_id", f"{_TABLE}.run_id")
    assert "stage_outputs" not in ledger
    # Kept OUT of the map the frozen 0028/0029 snapshots are pinned to.
    assert _TABLE not in rls.RLS_JOIN_PREDICATES


def _columns(conn, schema: str) -> list[tuple[str, str, str, str | None]]:
    rows = conn.execute(
        text(
            "SELECT column_name, data_type, is_nullable, column_default "
            "FROM information_schema.columns "
            "WHERE table_schema = :s AND table_name = :t ORDER BY column_name"
        ),
        {"s": schema, "t": _TABLE},
    ).all()
    return [tuple(r) for r in rows]


def _constraints(conn, schema: str) -> list[tuple[str, str, str]]:
    rows = conn.execute(
        text(
            "SELECT c.conname, c.contype::text, pg_get_constraintdef(c.oid) "
            "FROM pg_constraint c JOIN pg_class t ON t.oid = c.conrelid "
            "JOIN pg_namespace n ON n.oid = t.relnamespace "
            "WHERE n.nspname = :s AND t.relname = :t ORDER BY c.conname"
        ),
        {"s": schema, "t": _TABLE},
    ).all()
    # The FK's rendered target is schema-qualified only when it is off the
    # search_path; strip the probe schema so the two builds compare on shape.
    return [(name, kind, defn.replace(f"{schema}.", "")) for name, kind, defn in rows]


def test_alembic_and_create_all_build_the_same_table(parity_schemas: ParityProbe) -> None:
    probe = parity_schemas
    with probe.engine.connect() as conn:
        al_cols = _columns(conn, probe.alembic)
        ca_cols = _columns(conn, probe.create_all)
        al_cons = _constraints(conn, probe.alembic)
        ca_cons = _constraints(conn, probe.create_all)
        rls_flags = conn.execute(
            text(
                "SELECT c.relrowsecurity, c.relforcerowsecurity FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = :s AND c.relname = :t"
            ),
            {"s": probe.alembic, "t": _TABLE},
        ).one()

    assert [c[0] for c in al_cols] == [
        "created_at",
        "id",
        "input_hash",
        "result",
        "run_id",
        "stage_name",
        "step_key",
        "updated_at",
    ]
    assert al_cols == ca_cols
    assert all(nullable == "NO" for _, _, nullable, _ in al_cols), "every column is NOT NULL"
    assert dict((c[0], c[1]) for c in al_cols)["result"] == "jsonb"

    assert al_cons == ca_cons
    by_name = {name: (kind, defn) for name, kind, defn in al_cons}
    assert by_name["uq_stage_step_results_run_stage_step"] == (
        "u",
        "UNIQUE (run_id, stage_name, step_key)",
    )
    fks = [defn for kind, defn in by_name.values() if kind == "f"]
    assert fks == ["FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE"]

    assert tuple(rls_flags) == (True, False), "ENABLE, not FORCE (owner stays unbound)"
