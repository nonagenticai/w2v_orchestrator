"""Migration 0100 (``signup_approvals.removal``, the early-access removal).

1. DB-free: the id fits varchar(32) and chains onto 0099; no application imports;
   the audit CHECK widens 0097's set by exactly ``signup.remove``.
2. ``alembic upgrade head`` and ``create_all`` build the same table.
3. A real throwaway DATABASE: upgrade 0099 → 0100; ``removed`` is admitted only
   with a ``removal``; a removed row RESERVES the address until its removal is
   ``done`` / ``resolved_manually``; ``signup.remove`` is a legal audit action;
   downgrade REFUSES while a removed row exists, drops the column on a table
   without one; upgrade again.
"""

from __future__ import annotations

import json
from uuid import UUID

import pytest
from alembic import command
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from tests.database.test_migration_0077_tenant_connections import _shape
from tests.database.test_migration_0084_tenant_sound_assets import (  # noqa: F401 — fixture
    _VERSIONS,
    _load,
    _rejects,
    probe,
)
from tests.database.test_migration_0085_tenant_offboarding import _model_check
from tests.support.schema_probe import ParityProbe

_REV = "0100_signup_removal"
_PRE = "0099_signup_llm_access"
_INSERT = (
    "INSERT INTO signup_approvals (id, email, name, source, status, tenant_slug, tenant_id, "
    "decided_at, removal) VALUES (:id, :email, 'N', 'web', :status, :slug, :tid, :decided, "
    "CAST(:removal AS jsonb))"
)
_AUDIT = (
    "INSERT INTO admin_audit_log (id, created_at, action, detail) "
    "VALUES (:id, now(), :a, '{}'::jsonb)"
)


def _row(n: int, **over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": UUID(int=n),
        "email": "ann@example.test",
        "status": "removed",
        "slug": None,
        "tid": UUID(int=0xA0 + n),
        "decided": "2026-10-21T00:00:00Z",
        "removal": json.dumps({"state": "pending"}),
    }
    base.update(over)
    return base


def test_revision_id_fits_chains_and_imports_no_application_code() -> None:
    m = _load(_REV)
    assert m.revision == _REV and m.down_revision == _PRE and len(_REV) <= 32
    source = (_VERSIONS / f"{_REV}.py").read_text()
    for banned in ("from src", "import src", "Base.metadata"):
        assert banned not in source, banned
    old = set(m._OLD_ACTIONS.removeprefix("action IN (").rstrip(")").split(","))
    new = set(m._NEW_ACTIONS.removeprefix("action IN (").rstrip(")").split(","))
    assert new - old == {"'signup.remove'"} and old < new
    prev = _load("0097_audit_run_cancel")
    assert m._OLD_ACTIONS == prev._NEW_ACTIONS
    model = _model_check("admin_audit_log", "ck_admin_audit_action")
    assert set(model.removeprefix("action IN (").rstrip(")").split(",")) == new


def test_alembic_and_create_all_build_the_same_table(parity_schemas: ParityProbe) -> None:
    probe_ = parity_schemas
    with probe_.engine.connect() as conn:
        al = _shape(conn, probe_.alembic, "signup_approvals")
        ca = _shape(conn, probe_.create_all, "signup_approvals")
    assert al[0], "alembic build has no columns"
    assert al == ca, "create_all and alembic diverge on signup_approvals"


def test_removed_rows_reserve_the_address_until_done_and_downgrade(probe) -> None:  # noqa: F811
    cfg, engine = probe
    command.upgrade(cfg, _PRE)
    # 0099 refuses 'removed' and the audit action.
    assert _rejects(
        engine,
        _INSERT.replace(", removal)", ")").replace(", CAST(:removal AS jsonb))", ")"),
        {k: v for k, v in _row(1).items() if k != "removal"},
    )
    assert _rejects(engine, _AUDIT, {"id": UUID(int=0xF0), "a": "signup.remove"})
    command.upgrade(cfg, _REV)
    # a removed row needs its removal record
    assert _rejects(engine, _INSERT, _row(1, removal=None))
    assert not _rejects(engine, _INSERT, _row(1))
    # …which RESERVES the address while the removal is not finished
    assert _rejects(
        engine, _INSERT, _row(2, status="pending", decided=None, tid=None, removal=None)
    )
    with engine.begin() as conn:
        conn.execute(
            text('UPDATE signup_approvals SET removal = \'{"state": "done"}\' WHERE id = :id'),
            {"id": UUID(int=1)},
        )
    assert not _rejects(
        engine, _INSERT, _row(2, status="pending", decided=None, tid=None, removal=None)
    )
    # a resolved removal frees it too; a blocked one does not
    assert not _rejects(
        engine, _INSERT, _row(3, email="b@example.test", removal=json.dumps({"state": "blocked"}))
    )
    assert _rejects(
        engine,
        _INSERT,
        _row(4, email="b@example.test", status="pending", decided=None, tid=None, removal=None),
    )
    with engine.begin() as conn:
        conn.execute(
            text(
                'UPDATE signup_approvals SET removal = \'{"state": "resolved_manually"}\' '
                "WHERE id = :id"
            ),
            {"id": UUID(int=3)},
        )
    assert not _rejects(
        engine,
        _INSERT,
        _row(4, email="b@example.test", status="pending", decided=None, tid=None, removal=None),
    )
    assert not _rejects(engine, _AUDIT, {"id": UUID(int=0xF1), "a": "signup.remove"})
    # downgrade refuses while a removed row exists
    with pytest.raises(DBAPIError):
        command.downgrade(cfg, _PRE)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM signup_approvals WHERE status = 'removed'"))
    command.downgrade(cfg, _PRE)
    with engine.connect() as conn:
        cols = conn.execute(
            text(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_name = 'signup_approvals' AND column_name = 'removal'"
            )
        ).scalar()
        kept = conn.execute(
            text("SELECT count(*) FROM admin_audit_log WHERE action = 'signup.remove'")
        ).scalar()
    assert cols == 0 and kept == 1  # NOT VALID: the audit row survives the downgrade
    assert _rejects(engine, _AUDIT, {"id": UUID(int=0xF2), "a": "signup.remove"})
    command.upgrade(cfg, _REV)
