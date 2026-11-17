"""``scripts/ensure_tenant_licences.py`` — the operator CLI that ensures an
existing tenant's w2v licences and records ``licence_ref``.

``run`` is driven against the real test database with the licences fake, so
"dry run writes nothing and calls no licences write", "apply records the ref", idempotence, a token-owned ref, and refusals are
proven on rows. It shares ``ensure_tenant_licence_ref`` with the partner create
route (whose behaviour is pinned in ``tests/api/routes/test_p9_partner_keys.py``).
"""

from __future__ import annotations

import argparse
import datetime
from typing import Any, cast
from uuid import UUID, uuid7

import pytest
from sqlalchemy import insert, select

from scripts import ensure_tenant_licences
from scripts.ensure_tenant_licences import run
from src.database.models import AdminAuditLog, Tenant
from tests.api.routes._p9_licences import P9Licences

pytestmark = pytest.mark.asyncio


async def _registered(factory, *, licence_ref: UUID | None = None, token: bool = False) -> UUID:
    """A registry row with a null licence_ref — the live 2026-10-15 shape."""
    tid = uuid7()
    values: dict[str, Any] = {"id": tid, "name": f"t-{tid}", "created_via": "partner"}
    if licence_ref is not None:
        values["licence_ref"] = licence_ref
    if token:
        values["licence_token_ciphertext"] = b"sealed-not-a-real-token"
        values["licence_token_kid"] = "lt1"
        values["licence_token_set_at"] = datetime.datetime.now(datetime.UTC)
    async with factory() as s:
        await s.execute(insert(Tenant).values(**values))
        await s.commit()
    return tid


async def _tenant(factory, tid: UUID) -> Tenant:
    async with factory() as s:
        return (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one()


async def _audit(factory, tid: UUID) -> list[tuple[str, UUID | None, dict]]:
    async with factory() as s:
        rows = await s.execute(
            select(AdminAuditLog.action, AdminAuditLog.actor_subject_id, AdminAuditLog.detail)
            .where(AdminAuditLog.target_tenant_id == tid)
            .order_by(AdminAuditLog.created_at)
        )
        return [(r.action, r.actor_subject_id, r.detail) for r in rows]


async def _run(factory, lic: P9Licences, tid: UUID, *, apply: bool) -> tuple[int, str]:
    lines: list[str] = []
    code = await run(factory, cast("Any", lic), tenant_id=tid, apply=apply, out=lines.append)
    return code, "\n".join(lines)


async def test_dry_run_reports_and_writes_nothing(db_session_factory) -> None:
    tid = await _registered(db_session_factory)
    lic = P9Licences()

    code, out = await _run(db_session_factory, lic, tid, apply=False)

    assert code == 0, out
    assert "licence_ref=None" in out and "nothing written" in out
    assert lic.ensure_calls == [], "a dry run makes no licences write"
    assert (await _tenant(db_session_factory, tid)).licence_ref is None
    assert await _audit(db_session_factory, tid) == []


async def test_apply_ensures_and_records_the_ref(
    db_session_factory,
) -> None:
    tid = await _registered(db_session_factory)
    lic = P9Licences()

    code, out = await _run(db_session_factory, lic, tid, apply=True)

    assert code == 0, out
    [ensured] = lic.live[tid]
    assert (await _tenant(db_session_factory, tid)).licence_ref == ensured
    assert f"RECORDED {tid}: licence_ref={ensured}" in out
    assert await _audit(db_session_factory, tid) == [], "no admin_audit_log row (closed set)"


async def test_apply_is_idempotent(db_session_factory) -> None:
    tid = await _registered(db_session_factory)
    lic = P9Licences()
    assert (await _run(db_session_factory, lic, tid, apply=True))[0] == 0
    first = (await _tenant(db_session_factory, tid)).licence_ref

    code, out = await _run(db_session_factory, lic, tid, apply=True)

    assert code == 0, out
    assert f"UNCHANGED {tid}: licence_ref={first}" in out
    assert (await _tenant(db_session_factory, tid)).licence_ref == first
    assert lic.live[tid] == [first], "no second licence"


async def test_a_token_owned_ref_is_kept(db_session_factory) -> None:
    token_ref = uuid7()
    tid = await _registered(db_session_factory, licence_ref=token_ref, token=True)
    lic = P9Licences()

    code, out = await _run(db_session_factory, lic, tid, apply=True)

    assert code == 0, out
    assert lic.ensure_calls == [str(tid)], "the bundle is still ensured"
    assert (await _tenant(db_session_factory, tid)).licence_ref == token_ref
    assert "a stored token owns licence_ref; kept" in out


async def test_a_refused_licence_is_exit_1_and_records_nothing(db_session_factory) -> None:
    tid = await _registered(db_session_factory)
    lic = P9Licences()
    lic.ensure_refuse = {"w2v": "member_licence"}

    code, out = await _run(db_session_factory, lic, tid, apply=True)

    assert code == 1, out
    assert "member_licence" in out
    assert (await _tenant(db_session_factory, tid)).licence_ref is None
    assert await _audit(db_session_factory, tid) == []


async def test_an_unknown_tenant_is_refused_without_a_licences_call(db_session_factory) -> None:
    lic = P9Licences()
    tid = uuid7()

    code, out = await _run(db_session_factory, lic, tid, apply=True)

    assert code == 1, out
    assert "no live w2v registry row" in out
    assert lic.ensure_calls == []


async def test_main_aborts_with_exit_2_when_licences_is_unreachable(
    monkeypatch, capsys, db_session_factory
) -> None:
    tid = await _registered(db_session_factory)
    from src.api.routes import _common
    from src.database import connection, session

    async def _maker() -> Any:
        return db_session_factory

    async def _no_close() -> None:
        return None

    monkeypatch.setattr(session, "get_session_maker", _maker)
    monkeypatch.setattr(connection, "close_engine", _no_close)
    monkeypatch.setattr(_common, "get_licenses_client", lambda: P9Licences(down=True))

    code = await ensure_tenant_licences._main(argparse.Namespace(tenant_id=tid, apply=True))

    assert code == 2
    assert "ABORTED, nothing recorded: OSError" in capsys.readouterr().err
    assert (await _tenant(db_session_factory, tid)).licence_ref is None
