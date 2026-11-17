"""Migration 0098 (``signup_approvals``, early-access signup v2).

1. DB-free: the id fits varchar(32) and chains onto 0097; no application imports.
2. ``alembic upgrade head`` and ``create_all`` build the same table.
3. A real throwaway DATABASE: upgrade 0097 → 0098; the CHECKs and the two partial
   uniques are live (one OPEN row per address); downgrade REFUSES while a row exists and drops an empty
   table; upgrade again.
"""

from __future__ import annotations

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
from tests.support.schema_probe import ParityProbe

_REV = "0098_signup_approvals"
_PRE = "0097_audit_run_cancel"
_INSERT = (
    "INSERT INTO signup_approvals (id, email, name, source, status, locale, message, "
    "tenant_slug, tenant_id, decided_at) VALUES (:id, :email, 'N', :source, :status, :locale, "
    ":message, :slug, :tid, :decided)"
)


def _row(n: int, **over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": UUID(int=n),
        "email": f"p{n}@example.test",
        "source": "web",
        "status": "pending",
        "locale": None,
        "message": None,
        "slug": None,
        "tid": None,
        "decided": None,
    }
    base.update(over)
    return base


def test_alembic_and_create_all_build_the_same_table(parity_schemas: ParityProbe) -> None:
    probe_ = parity_schemas
    with probe_.engine.connect() as conn:
        al = _shape(conn, probe_.alembic, "signup_approvals")
        ca = _shape(conn, probe_.create_all, "signup_approvals")
    assert al[0], "alembic build has no columns"
    assert al == ca, "create_all and alembic diverge on signup_approvals"
    assert al[3] == (False, False), "no RLS: a platform table"


def test_constraints_and_downgrade_guard(probe) -> None:  # noqa: F811
    cfg, engine = probe
    command.upgrade(cfg, _PRE)
    command.upgrade(cfg, _REV)
    assert not _rejects(engine, _INSERT, _row(1))
    # one PENDING row per lower(email)
    assert _rejects(engine, _INSERT, _row(2, email="P1@example.test"))
    assert _rejects(engine, _INSERT, _row(3, locale="xx"))
    assert _rejects(engine, _INSERT, _row(4, source="email"))
    assert _rejects(engine, _INSERT, _row(5, message="x" * 2001))
    assert _rejects(engine, _INSERT, _row(6, slug="Bad Slug"))
    # approved needs a tenant + a decision time; pending must not carry one
    assert _rejects(engine, _INSERT, _row(7, status="approved", decided="2026-10-20"))
    assert _rejects(engine, _INSERT, _row(8, decided="2026-10-20"))
    tid = UUID(int=0xAA)
    ok = _row(9, status="approved", decided="2026-10-20", slug="acme", tid=tid)
    assert not _rejects(engine, _INSERT, ok)
    # the slug is reserved once
    assert _rejects(
        engine, _INSERT, _row(10, status="approved", decided="2026-10-20", slug="acme", tid=tid)
    )
    # an APPROVED row blocks a new one for the same address (one workspace per person)
    assert _rejects(engine, _INSERT, _row(11, email="P9@example.test"))
    # a REJECTED row does not block a new pending one
    rejected = _row(13, status="rejected", decided="2026-10-20", email="p14@example.test")
    assert not _rejects(engine, _INSERT, rejected)
    assert not _rejects(engine, _INSERT, _row(14))

    with pytest.raises(DBAPIError, match="refusing to drop signup_approvals"):
        command.downgrade(cfg, _PRE)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM signup_approvals"))
    command.downgrade(cfg, _PRE)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT to_regclass('signup_approvals')")).scalar() is None
    command.upgrade(cfg, _REV)
    assert not _rejects(engine, _INSERT, _row(12))
