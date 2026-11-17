"""Offboarding covers EVERY orchestrator table — a new table cannot slip past it.

``services.tenant_offboarding.TENANT_TABLES`` decides, per table, how a tenant's rows are
exported and deleted. If a table exists that the plan does not name, an offboarded
tenant's rows in it would survive the delete (and be missing from the export) while
every other test stays green. These tests compare the plan against the models, the
``create_all`` schema the suite runs on, AND the schema ``alembic upgrade head`` builds
(what prod runs). Adding a table fails here until it is classified.
"""

from __future__ import annotations

from src.database.models import Base
from src.services import tenant_offboarding as svc

_PLANNED = [s.table for s in svc.TENANT_TABLES]


def test_plan_names_each_table_once_and_every_model_table() -> None:
    assert len(_PLANNED) == len(set(_PLANNED)), "a table is planned twice"
    missing = set(Base.metadata.tables) - set(_PLANNED)
    stale = set(_PLANNED) - set(Base.metadata.tables)
    assert not missing, f"tables with no offboarding rule (export + delete): {sorted(missing)}"
    assert not stale, f"plan names tables that do not exist: {sorted(stale)}"


def test_only_the_registry_and_the_job_trail_survive_a_delete() -> None:
    kept = {s.table: s.mode for s in svc.TENANT_TABLES if s.mode in ("retain", "tombstone")}
    assert kept == {"tenant_offboarding": "retain", "tenants": "tombstone"}
    assert all(s.why for s in svc.TENANT_TABLES if s.mode != "delete"), "say WHY it is kept"


def test_deletion_order_is_leaf_first() -> None:
    """A child's selector climbs through its parents, so every FK child must be
    deleted BEFORE the table it references (else its selector finds nothing)."""
    position = {t: i for i, t in enumerate(_PLANNED)}
    for table in Base.metadata.tables.values():
        for fk in table.foreign_keys:
            parent = fk.column.table.name
            if parent == table.name or parent == "tenants":
                continue  # self-reference / the tombstone, which is never deleted
            assert position[table.name] < position[parent], (
                f"{table.name} must be deleted before {parent}"
            )


def test_redacted_export_columns_exist() -> None:
    for table, columns in svc._EXPORT_DROP_COLUMNS.items():
        assert set(columns) <= set(Base.metadata.tables[table].columns.keys()), table
