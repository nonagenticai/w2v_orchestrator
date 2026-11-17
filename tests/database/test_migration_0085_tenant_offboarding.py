"""P6 ORCH-6 migration 0085 (``tenant_offboarding`` + ``deleted`` tombstone + reject).

1. DB-free: the id fits and chains onto 0084; no application imports; the audit CHECK
   is exactly 0084's set plus ``offboarding.reject`` (0086 widened it again); the tenant
   status CHECK equals the model's.
2. ``alembic upgrade head`` and ``create_all`` build the same table / tenants CHECK.
3. A real throwaway DATABASE: upgrade 0084 → 0085, every constraint is live, a re-apply
   over an existing (unstamped) table keeps rows, downgrade keeps tombstones and
   reject rows (NOT VALID) and drops the table, upgrade again.
"""

from __future__ import annotations

from uuid import UUID

from alembic import command
from sqlalchemy import CheckConstraint, Engine, text

from src.database.models import Base
from tests.database.test_migration_0077_tenant_connections import _shape
from tests.database.test_migration_0084_tenant_sound_assets import (  # noqa: F401 — fixture
    _VERSIONS,
    _action_set,
    _load,
    _rejects,
    probe,
)
from tests.support.schema_probe import ParityProbe

_REV = "0085_tenant_offboarding"
_PRE = "0084_tenant_sound_assets"

TENANT = UUID("66666666-6666-7666-8666-666666666666")
OTHER = UUID("77777777-7777-7777-8777-777777777777")


def _model_check(table: str, name: str) -> str:
    return next(
        str(c.sqltext)
        for c in Base.metadata.tables[table].constraints
        if isinstance(c, CheckConstraint) and c.name == name
    )


def test_audit_check_is_0084_plus_reject() -> None:
    # The model now carries 0086's set; tests/database/test_migration_0086_* pins it.
    m = _load(_REV)
    assert m._OLD_ACTIONS == _load(_PRE)._NEW_ACTIONS, "downgrade target is 0084 verbatim"
    old_set, new_set = _action_set(m._OLD_ACTIONS), _action_set(m._NEW_ACTIONS)
    assert new_set - old_set == {"'offboarding.reject'"} and old_set < new_set


def test_tenant_status_check_matches_the_model() -> None:
    m = _load(_REV)
    assert _action_set(m._NEW_TENANT_STATUS) - _action_set(m._OLD_TENANT_STATUS) == {"'deleted'"}
    assert _action_set(_model_check("tenants", "ck_tenants_status")) == _action_set(
        m._NEW_TENANT_STATUS
    )


def test_alembic_and_create_all_build_the_same_table(parity_schemas: ParityProbe) -> None:
    schemas = parity_schemas
    with schemas.engine.connect() as conn:
        for table in ("tenant_offboarding", "tenants"):
            al = _shape(conn, schemas.alembic, table)
            ca = _shape(conn, schemas.create_all, table)
            assert al[0], f"alembic build of {table} has no columns"
            assert al == ca, f"create_all and alembic diverge on {table}"
    with schemas.engine.connect() as conn:
        shape = _shape(conn, schemas.alembic, "tenant_offboarding")
    assert shape[3] == (False, False), "no RLS: the operator queue reads across tenants"
    assert "WHERE (status = ANY" in dict(shape[2])["uq_tenant_offboarding_open"]


_JOB = (
    "INSERT INTO tenant_offboarding (id, tenant_id, kind, status, artifact_key) "
    "VALUES (:id, :t, :kind, :status, :key)"
)
_AUDIT = "INSERT INTO admin_audit_log (id, action) VALUES (:id, :a)"


def _job(n: int, **over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": UUID(f"01999999-0000-7000-8000-{n:012d}"),
        "t": TENANT,
        "kind": "export",
        "status": "done",
        "key": None,
    }
    base.update(over)
    return base


def _count(engine: Engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM tenant_offboarding")).scalar_one()


def test_upgrade_constraints_drift_and_round_trip(probe) -> None:  # noqa: F811
    cfg, engine = probe
    command.upgrade(cfg, _PRE)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO tenants (id) VALUES (:a), (:b)"), {"a": TENANT, "b": OTHER})
    tomb = "UPDATE tenants SET status = 'deleted' WHERE id = :t"
    assert _rejects(engine, tomb, {"t": OTHER}), "control: 0084 has no 'deleted' status"
    assert _rejects(engine, _AUDIT, {"id": UUID(int=0xB0), "a": "offboarding.reject"})

    command.upgrade(cfg, _REV)
    assert not _rejects(engine, tomb, {"t": OTHER})
    assert _rejects(engine, "UPDATE tenants SET status = 'gone' WHERE id = :t", {"t": TENANT})
    assert not _rejects(engine, _AUDIT, {"id": UUID(int=0xB1), "a": "offboarding.reject"})
    assert not _rejects(engine, _AUDIT, {"id": UUID(int=0xB2), "a": "music.update"})

    key = f"exports/tenants/{TENANT}/x.zip"
    assert not _rejects(engine, _JOB, _job(1, key=key)), "control row must insert"
    assert _rejects(engine, _JOB, _job(2, kind="purge")), "kind"
    assert _rejects(engine, _JOB, _job(3, status="paused")), "status"
    assert _rejects(engine, _JOB, _job(4, key=f"exports/tenants/{OTHER}/x.zip")), (
        "an archive key may not point into ANOTHER tenant's export tree"
    )
    assert _rejects(engine, _JOB, _job(5, key="tenants/x.zip"))
    assert _rejects(engine, _JOB, _job(6, t=UUID(int=6))), "unknown tenant → FK"
    # At most one OPEN job per (tenant, kind); closed ones do not count.
    assert not _rejects(engine, _JOB, _job(7, status="requested"))
    assert _rejects(engine, _JOB, _job(8, status="running")), "second open export"
    assert not _rejects(engine, _JOB, _job(9, status="requested", kind="delete"))
    assert not _rejects(engine, _JOB, _job(10, status="failed"))
    # No cascade: a tenant with job rows cannot be hard-deleted (it is tombstoned).
    assert _rejects(engine, "DELETE FROM tenants WHERE id = :t", {"t": TENANT})
    assert _count(engine) == 4

    with engine.begin() as conn:  # drift: table present, version back at 0084
        conn.execute(text("UPDATE alembic_version SET version_num = :v"), {"v": _PRE})
    command.upgrade(cfg, _REV)
    assert _count(engine) == 4, "re-apply must keep rows"

    command.downgrade(cfg, _PRE)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT to_regclass('tenant_offboarding')")).scalar_one() is None
        kept = conn.execute(
            text("SELECT count(*) FROM tenants WHERE status = 'deleted'")
        ).scalar_one()
        rejects = conn.execute(
            text("SELECT count(*) FROM admin_audit_log WHERE action = 'offboarding.reject'")
        ).scalar_one()
    assert kept == 1 and rejects == 1, "downgrade keeps rows written under the new values"
    assert _rejects(engine, _AUDIT, {"id": UUID(int=0xB3), "a": "offboarding.reject"})
    assert _rejects(engine, tomb, {"t": TENANT})
    command.upgrade(cfg, _REV)
    assert _count(engine) == 0
