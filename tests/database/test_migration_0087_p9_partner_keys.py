"""P9 migration 0087 (``tenant_api_keys.partner_licence_id`` + ``tenant.partner_bind``).

1. DB-free: the id fits varchar(32) and chains onto 0086; no application imports; the
   audit CHECK is exactly 0086's set plus ``tenant.partner_bind`` and equals the model's.
2. ``alembic upgrade head`` and ``create_all`` build the same ``tenant_api_keys``
   (column + the partial unique index).
3. A real throwaway DATABASE: upgrade 0086 → 0087 with a key present (it stays a
   tenant_admin key: ``partner_licence_id`` NULL), two LIVE partner keys of one name
   are refused while a revoked one and a tenant_admin duplicate are not, a re-apply
   over an existing (unstamped) column keeps the data, downgrade keeps partner_bind
   audit rows (NOT VALID), upgrade again.
"""

from __future__ import annotations

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

_REV = "0087_p9_partner_keys"
_PRE = "0086_apikey_scopes_bulk_set"

TENANT = UUID("88888888-8888-7888-8888-888888888887")
PARTNER = UUID("99999999-9999-4999-8999-999999999987")
KEY = UUID("01999999-0000-7000-8000-000000000087")
_AUDIT = "INSERT INTO admin_audit_log (id, action) VALUES (:id, :a)"
_KEY = (
    "INSERT INTO tenant_api_keys (id, tenant_id, name, prefix, key_hash, partner_licence_id, "
    "revoked_at) VALUES (:id, :t, :n, :p, decode(repeat('00', 32), 'hex'), :pl, :r)"
)


def test_audit_check_is_0086_plus_partner_bind_and_still_in_the_model() -> None:
    # The model now carries 0089's set; tests/database/test_migration_0089_* pins it.
    # Here: nothing 0087 introduced has been dropped since.
    m = _load(_REV)
    assert m._OLD_ACTIONS == _load(_PRE)._NEW_ACTIONS, "downgrade target is 0086 verbatim"
    old_set, new_set = _action_set(m._OLD_ACTIONS), _action_set(m._NEW_ACTIONS)
    assert new_set - old_set == {"'tenant.partner_bind'"} and old_set < new_set
    assert new_set <= _action_set(_model_check("admin_audit_log", "ck_admin_audit_action"))


def test_alembic_and_create_all_build_the_same_table(parity_schemas: ParityProbe) -> None:
    schemas = parity_schemas
    with schemas.engine.connect() as conn:
        al = _shape(conn, schemas.alembic, "tenant_api_keys")
        ca = _shape(conn, schemas.create_all, "tenant_api_keys")
    assert any(c[0] == "partner_licence_id" for c in al[0]), "alembic build lacks the column"
    assert any(i[0] == "uq_tenant_api_keys_partner_live_name" for i in al[2])
    assert al == ca, "create_all and alembic diverge on tenant_api_keys"


def _partner_of(engine: Engine, key: UUID) -> UUID | None:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT partner_licence_id FROM tenant_api_keys WHERE id = :k"), {"k": key}
        ).scalar_one()


def _key(i: int, name: str, *, partner: UUID | None, revoked: bool = False) -> dict[str, object]:
    return {
        "id": UUID(int=0x870 + i),
        "t": TENANT,
        "n": name,
        "p": f"abcdefg{'abcdefgh'[i]}",
        "pl": partner,
        "r": "2026-10-01T00:00:00Z" if revoked else None,
    }


def test_upgrade_constraints_drift_and_round_trip(probe) -> None:  # noqa: F811
    cfg, engine = probe
    command.upgrade(cfg, _PRE)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO tenants (id) VALUES (:t)"), {"t": TENANT})
        conn.execute(
            text(
                "INSERT INTO tenant_api_keys (id, tenant_id, name, prefix, key_hash) "
                "VALUES (:k, :t, 'merqi-marketing', 'zzzzzzzz', decode(repeat('00', 32), 'hex'))"
            ),
            {"k": KEY, "t": TENANT},
        )
    assert _rejects(engine, _AUDIT, {"id": UUID(int=0xD0), "a": "tenant.partner_bind"})

    command.upgrade(cfg, _REV)
    assert _partner_of(engine, KEY) is None, "an existing key stays a tenant_admin key"
    # One LIVE partner key per (tenant, name) …
    assert not _rejects(engine, _KEY, _key(0, "merqi-sales", partner=PARTNER))
    assert _rejects(engine, _KEY, _key(1, "merqi-sales", partner=PARTNER)), "duplicate live"
    # … but a revoked one, and a tenant_admin key of the same name, do not collide.
    assert not _rejects(engine, _KEY, _key(2, "merqi-sales", partner=PARTNER, revoked=True))
    assert not _rejects(engine, _KEY, _key(3, "merqi-sales", partner=None))
    assert not _rejects(engine, _KEY, _key(4, "merqi-marketing", partner=PARTNER))
    assert not _rejects(engine, _AUDIT, {"id": UUID(int=0xD1), "a": "tenant.partner_bind"})

    with engine.begin() as conn:  # drift: column + index present, version back at 0086
        conn.execute(text("UPDATE alembic_version SET version_num = :v"), {"v": _PRE})
    command.upgrade(cfg, _REV)
    assert _partner_of(engine, UUID(int=0x870)) == PARTNER, "re-apply must keep data"

    command.downgrade(cfg, _PRE)
    with engine.connect() as conn:
        kept = conn.execute(
            text("SELECT count(*) FROM admin_audit_log WHERE action = 'tenant.partner_bind'")
        ).scalar_one()
        has_col = conn.execute(
            text(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_name = 'tenant_api_keys' AND column_name = 'partner_licence_id'"
            )
        ).scalar_one()
    assert kept == 1, "downgrade keeps rows written under the new action"
    assert has_col == 0
    assert _rejects(engine, _AUDIT, {"id": UUID(int=0xD2), "a": "tenant.partner_bind"})
    command.upgrade(cfg, _REV)
    assert _partner_of(engine, KEY) is None
