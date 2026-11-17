"""Migration 0097 (audit action ``run.cancel`` for the super_admin run cancel).

1. DB-free: the id fits varchar(32) and chains onto 0096; no application imports;
   the audit CHECK is exactly 0089's set plus ``run.cancel`` and equals the model's.
2. A real throwaway DATABASE: the action is refused at 0096 and accepted at 0097;
   downgrade keeps the row (NOT VALID) and refuses new ones; upgrade again.
"""

from __future__ import annotations

from uuid import UUID

from alembic import command

from tests.database.test_migration_0084_tenant_sound_assets import (  # noqa: F401 — fixture
    _VERSIONS,
    _action_set,
    _load,
    _rejects,
    probe,
)
from tests.database.test_migration_0085_tenant_offboarding import _model_check

_REV = "0097_audit_run_cancel"
_PRE = "0096_drop_conn_runtime_sync"
_AUDIT = "INSERT INTO admin_audit_log (id, action) VALUES (:id, :a)"


def test_audit_check_is_0089_plus_run_cancel_and_matches_the_model() -> None:
    m = _load(_REV)
    assert m._OLD_ACTIONS == _load("0089_tenant_licence_token")._NEW_ACTIONS
    old_set, new_set = _action_set(m._OLD_ACTIONS), _action_set(m._NEW_ACTIONS)
    assert new_set - old_set == {"'run.cancel'"} and old_set < new_set
    # The model carries the LATEST set (0100 widened it again; that test pins equality).
    assert new_set <= _action_set(_model_check("admin_audit_log", "ck_admin_audit_action"))


def test_upgrade_accepts_run_cancel_and_downgrade_keeps_rows(probe) -> None:  # noqa: F811
    cfg, engine = probe
    command.upgrade(cfg, _PRE)
    assert _rejects(engine, _AUDIT, {"id": UUID(int=0xF0), "a": "run.cancel"})
    command.upgrade(cfg, _REV)
    assert not _rejects(engine, _AUDIT, {"id": UUID(int=0xF1), "a": "run.cancel"})
    command.downgrade(cfg, _PRE)
    with engine.connect() as conn:
        kept = conn.exec_driver_sql(
            "SELECT count(*) FROM admin_audit_log WHERE action = 'run.cancel'"
        ).scalar_one()
    assert kept == 1, "downgrade keeps rows written under the new action"
    assert _rejects(engine, _AUDIT, {"id": UUID(int=0xF2), "a": "run.cancel"})
    command.upgrade(cfg, _REV)
    assert not _rejects(engine, _AUDIT, {"id": UUID(int=0xF3), "a": "run.cancel"})
