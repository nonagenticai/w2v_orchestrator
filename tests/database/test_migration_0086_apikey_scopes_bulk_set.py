"""P8 migration 0086 (``tenant_api_keys.scopes`` + ``connection.bulk_set``).

1. DB-free: the id fits varchar(32) and chains onto 0085; no application imports; the
   audit CHECK is exactly 0085's set plus ``connection.bulk_set`` and is still in the
   model's (0087 widened it again); the scopes CHECK equals the model's.
2. ``alembic upgrade head`` and ``create_all`` build the same ``tenant_api_keys``.
3. A real throwaway DATABASE: upgrade 0085 → 0086 with a key present (it gets
   ``'{}'`` — no silent grant), the scopes vocabulary is closed, a re-apply over an
   existing (unstamped) column keeps the data, downgrade keeps bulk_set audit rows
   (NOT VALID), upgrade again.
"""

from __future__ import annotations

import re
from uuid import UUID

from alembic import command
from sqlalchemy import Engine, text

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

_REV = "0086_apikey_scopes_bulk_set"
_PRE = "0085_tenant_offboarding"

TENANT = UUID("88888888-8888-7888-8888-888888888888")
KEY = UUID("01999999-0000-7000-8000-000000000086")
_AUDIT = "INSERT INTO admin_audit_log (id, action) VALUES (:id, :a)"


def test_audit_check_is_0085_plus_bulk_set_and_still_in_the_model() -> None:
    # The model now carries 0087's set; tests/database/test_migration_0087_* pins it.
    # Here: nothing 0086 introduced has been dropped since.
    m = _load(_REV)
    assert m._OLD_ACTIONS == _load(_PRE)._NEW_ACTIONS, "downgrade target is 0085 verbatim"
    old_set, new_set = _action_set(m._OLD_ACTIONS), _action_set(m._NEW_ACTIONS)
    assert new_set - old_set == {"'connection.bulk_set'"} and old_set < new_set
    assert new_set <= _action_set(_model_check("admin_audit_log", "ck_admin_audit_action"))


def test_scopes_check_matches_the_model() -> None:
    model = _model_check("tenant_api_keys", "ck_tenant_api_keys_scopes")
    assert " ".join(model.split()) == " ".join(_load(_REV)._SCOPES_CHECK.split())
    assert re.findall(r"'([^']+)'", model) == ["connections:write"]


def test_alembic_and_create_all_build_the_same_table(parity_schemas: ParityProbe) -> None:
    schemas = parity_schemas
    with schemas.engine.connect() as conn:
        al = _shape(conn, schemas.alembic, "tenant_api_keys")
        ca = _shape(conn, schemas.create_all, "tenant_api_keys")
    assert al[0], "alembic build of tenant_api_keys has no columns"
    assert al == ca, "create_all and alembic diverge on tenant_api_keys"


def _scopes(engine: Engine) -> list[str] | None:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT scopes FROM tenant_api_keys WHERE id = :k"), {"k": KEY}
        ).scalar_one()


def test_upgrade_constraints_drift_and_round_trip(probe) -> None:  # noqa: F811
    cfg, engine = probe
    command.upgrade(cfg, _PRE)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO tenants (id) VALUES (:t)"), {"t": TENANT})
        conn.execute(
            text(
                "INSERT INTO tenant_api_keys (id, tenant_id, name, prefix, key_hash) "
                "VALUES (:k, :t, 'k', 'abcdefgh', decode(repeat('00', 32), 'hex'))"
            ),
            {"k": KEY, "t": TENANT},
        )
    assert _rejects(engine, _AUDIT, {"id": UUID(int=0xC0), "a": "connection.bulk_set"})

    command.upgrade(cfg, _REV)
    assert _scopes(engine) == [], "an existing key gains no grant"
    grant = "UPDATE tenant_api_keys SET scopes = :s WHERE id = :k"
    assert not _rejects(engine, grant, {"s": ["connections:write"], "k": KEY})
    assert _rejects(engine, grant, {"s": ["connections:admin"], "k": KEY}), "closed vocabulary"
    assert _rejects(engine, "UPDATE tenant_api_keys SET scopes = NULL WHERE id = :k", {"k": KEY})
    assert not _rejects(engine, _AUDIT, {"id": UUID(int=0xC1), "a": "connection.bulk_set"})

    with engine.begin() as conn:  # drift: column present, version back at 0085
        conn.execute(text("UPDATE alembic_version SET version_num = :v"), {"v": _PRE})
    command.upgrade(cfg, _REV)
    assert _scopes(engine) == ["connections:write"], "re-apply must keep data"

    command.downgrade(cfg, _PRE)
    with engine.connect() as conn:
        kept = conn.execute(
            text("SELECT count(*) FROM admin_audit_log WHERE action = 'connection.bulk_set'")
        ).scalar_one()
        has_col = conn.execute(
            text(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_name = 'tenant_api_keys' AND column_name = 'scopes'"
            )
        ).scalar_one()
    assert kept == 1, "downgrade keeps rows written under the new action"
    assert has_col == 0
    assert _rejects(engine, _AUDIT, {"id": UUID(int=0xC2), "a": "connection.bulk_set"})
    command.upgrade(cfg, _REV)
    assert _scopes(engine) == []
