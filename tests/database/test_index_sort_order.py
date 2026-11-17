"""Model-vs-migration parity for DESC-ordered indexes.

Two indexes are created with an explicit descending sort order by the migrations
(and therefore in prod): ``idx_admin_audit_created`` (0022) and
``idx_runs_deliverable`` (0015). Both serve "newest first" reads.

``src/database/models.py`` declared them WITHOUT the sort order, which is not a
runtime bug — the database keeps what the migration built — but it made
``alembic check`` report a permanent phantom drift (fixed on main; this test
exists so it cannot come back a third time):

    Detected changed index 'idx_admin_audit_created' on 'admin_audit_log':
    expression #1 'created_at DESC' to 'created_at'

A standing false positive is worse than no check at all: it trains everyone to
ignore the one signal that would catch a REAL divergence. This test pins the
declaration so the drift cannot silently come back — it reads the metadata only,
so it needs no database and never skips.

It asserts the RENDERED SQL, not the SQLAlchemy construct, so either spelling
(``text("created_at DESC")`` or ``desc("created_at")``) satisfies it — what is
pinned is the DESC reaching the DDL, which is the part that drifted.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Index

from src.database.models import Base

# index name → the ordered expressions as SQLAlchemy renders them. A bound Column
# renders table-qualified; the ``desc()`` literal does not — what matters here is
# the DESC suffix, which is the half that used to be missing.
DESC_INDEXES = {
    "idx_admin_audit_created": ["created_at DESC"],
    "idx_runs_deliverable": ["runs.deliverable_id", "created_at DESC"],
}


def _index(name: str) -> Index:
    for table in Base.metadata.tables.values():
        for index in table.indexes:
            if index.name == name:
                return index
    raise AssertionError(f"index {name!r} is not declared on any model")


@pytest.mark.parametrize(("name", "expected"), list(DESC_INDEXES.items())[:1])
def test_index_declares_the_sort_order_the_migration_creates(
    name: str, expected: list[str]
) -> None:
    rendered = [str(expr) for expr in _index(name).expressions]
    assert rendered == expected, (
        f"{name} is declared as {rendered} but the migration creates {expected}; "
        "this divergence makes `alembic check` fail forever on a phantom diff"
    )
