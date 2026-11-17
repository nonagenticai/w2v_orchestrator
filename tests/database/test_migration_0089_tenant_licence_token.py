"""Migration 0089 (tenant licence token, sealed + licence-change lease + audit action).

1. DB-free: the id fits varchar(32) and chains onto 0088; no application imports;
   the token CHECK equals the model's; the audit CHECK is exactly 0087's set plus
   ``tenant.licence_token_set`` and equals the model's.
2. ``alembic upgrade head`` and ``create_all`` build the same ``tenants``.
3. A real throwaway DATABASE: upgrade 0088 → 0089 with a tenant present (columns
   NULL), a half token triple is refused, the new audit action is accepted; drift
   (columns present, version back at 0088, a half triple written while the CHECK
   was absent) re-applies cleanly and clears the half triple; downgrade drops the
   columns and keeps the audit row (NOT VALID); upgrade again.
"""

from __future__ import annotations

from uuid import UUID

from alembic import command
from sqlalchemy import CheckConstraint, text

from src.database.models import Base
from tests.database.test_migration_0077_tenant_connections import _shape
from tests.database.test_migration_0084_tenant_sound_assets import (  # noqa: F401 — fixture
    _VERSIONS,
    _action_set,
    _load,
    _rejects,
    probe,
)
from tests.database.test_migration_0085_tenant_offboarding import _model_check
from tests.support.schema_probe import ParityProbe

_REV = "0089_tenant_licence_token"
_PRE = "0088_run_usage_connection"
_COLS = {
    "licence_token_ciphertext",
    "licence_token_kid",
    "licence_token_set_at",
    "licence_token_expires_at",
    "licence_op_id",
    "licence_op_until",
}
TENANT = UUID("88888888-8888-7888-8888-888888888889")
_AUDIT = "INSERT INTO admin_audit_log (id, action) VALUES (:id, :a)"
_HALF = (
    "UPDATE tenants SET licence_token_ciphertext = '\\x01'::bytea, licence_token_kid = NULL, "
    "licence_token_set_at = NULL WHERE id = :t"
)
_FULL = (
    "UPDATE tenants SET licence_token_ciphertext = '\\x01'::bytea, licence_token_kid = 'k1', "
    "licence_token_set_at = now() WHERE id = :t"
)


def test_token_check_matches_the_model() -> None:
    m = _load(_REV)
    model = next(
        c
        for c in Base.metadata.tables["tenants"].constraints
        if isinstance(c, CheckConstraint) and c.name == "ck_tenants_licence_token_complete"
    )
    assert str(model.sqltext) == m._TRIPLE_CHECK
    assert any(f"CHECK ({m._TRIPLE_CHECK})" in s for s in m._CONSTRAINTS)


def test_audit_check_is_0087_plus_licence_token_set_and_matches_the_model() -> None:
    m = _load(_REV)
    assert m._OLD_ACTIONS == _load("0087_p9_partner_keys")._NEW_ACTIONS
    old_set, new_set = _action_set(m._OLD_ACTIONS), _action_set(m._NEW_ACTIONS)
    assert new_set - old_set == {"'tenant.licence_token_set'"} and old_set < new_set
    # ``<=``: later migrations (0097 ``run.cancel``) widen the model past 0089's set.
    assert new_set <= _action_set(_model_check("admin_audit_log", "ck_admin_audit_action"))


def test_alembic_and_create_all_build_the_same_tenants(parity_schemas: ParityProbe) -> None:
    schemas = parity_schemas
    with schemas.engine.connect() as conn:
        al = _shape(conn, schemas.alembic, "tenants")
        ca = _shape(conn, schemas.create_all, "tenants")
    assert {c[0] for c in al[0]} >= _COLS
    assert al == ca, "create_all and alembic diverge on tenants"


def _cols(engine) -> set[str]:  # type: ignore[no-untyped-def]
    with engine.connect() as conn:
        return set(
            conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = 'tenants'"
                )
            ).scalars()
        )


def test_upgrade_constraints_drift_and_round_trip(probe) -> None:  # noqa: F811
    cfg, engine = probe
    command.upgrade(cfg, _PRE)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO tenants (id) VALUES (:t)"), {"t": TENANT})
    assert _rejects(engine, _AUDIT, {"id": UUID(int=0xE0), "a": "tenant.licence_token_set"})

    command.upgrade(cfg, _REV)
    assert _cols(engine) >= _COLS
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT licence_token_ciphertext, licence_op_id FROM tenants WHERE id = :t"),
            {"t": TENANT},
        ).one()
    assert tuple(row) == (None, None), "an existing tenant starts with no token"
    assert _rejects(engine, _HALF, {"t": TENANT}), "a half token triple is refused"
    assert not _rejects(engine, _FULL, {"t": TENANT})
    assert not _rejects(engine, _AUDIT, {"id": UUID(int=0xE1), "a": "tenant.licence_token_set"})

    # Drift: CHECK absent, a half triple written, version back at 0088 → cleared, re-applied.
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE tenants DROP CONSTRAINT ck_tenants_licence_token_complete"))
        conn.execute(text(_HALF), {"t": TENANT})
        conn.execute(text("UPDATE alembic_version SET version_num = :v"), {"v": _PRE})
    command.upgrade(cfg, _REV)
    with engine.connect() as conn:
        blob = conn.execute(
            text("SELECT licence_token_ciphertext FROM tenants WHERE id = :t"), {"t": TENANT}
        ).scalar_one()
    assert blob is None, "the half triple was cleared, never left to fail the CHECK"

    command.downgrade(cfg, _PRE)
    assert not (_cols(engine) & _COLS)
    with engine.connect() as conn:
        kept = conn.execute(
            text("SELECT count(*) FROM admin_audit_log WHERE action = 'tenant.licence_token_set'")
        ).scalar_one()
    assert kept == 1, "downgrade keeps rows written under the new action"
    assert _rejects(engine, _AUDIT, {"id": UUID(int=0xE2), "a": "tenant.licence_token_set"})
    command.upgrade(cfg, _REV)
    assert _cols(engine) >= _COLS
