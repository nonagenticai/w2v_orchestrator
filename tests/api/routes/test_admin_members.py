"""Tenant members roster — the ``GET /api/auth/me`` roster write.

The roster is populated best-effort on ``GET /api/auth/me``. The old
``GET /api/admin/members`` read route was deleted in P2 WP4b (its semantics are
WP4a's ``/api/tenant/members``), so the roster is read straight from the table.
"""

from __future__ import annotations

from uuid import uuid4, uuid7

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes.auth import router as auth_router
from src.database.models import TenantMember
from src.database.repositories.members_repository import MembersRepository

pytestmark = pytest.mark.asyncio


def _build_app(
    db_session_factory,
    *,
    tenant_id: str,
    roles: list[str],
    subject_id: str,
    email: str | None = None,
    name: str | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = db_session_factory
    app.include_router(auth_router)

    class _SeedIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = tenant_id
            request.state.roles = roles
            request.state.subject_id = subject_id
            request.state.email = email
            request.state.name = name
            return await call_next(request)

    app.add_middleware(_SeedIdentity)
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


# ─── /me populates the roster ────────────────────────────────────────────


async def test_me_records_member(db_session_factory) -> None:
    tenant = uuid4()
    subject = uuid4()
    app = _build_app(
        db_session_factory,
        tenant_id=str(tenant),
        roles=["tenant_admin"],
        subject_id=str(subject),
        email="ada@example.com",
        name="Ada Lovelace",
    )
    async with _client(app) as c:
        me1 = await c.get("/api/auth/me")
        assert me1.status_code == 200, me1.text
        # A second /me is an upsert, not a duplicate row.
        await c.get("/api/auth/me")
    async with db_session_factory() as s:
        rows = (
            (await s.execute(select(TenantMember).where(TenantMember.tenant_id == tenant)))
            .scalars()
            .all()
        )
    assert len(rows) == 1
    member = rows[0]
    assert member.subject_id == subject
    assert member.email == "ada@example.com"
    assert member.display_name == "Ada Lovelace"
    assert member.roles == ["tenant_admin"]
    assert member.last_seen_at >= member.first_seen_at


async def test_me_roster_write_is_best_effort(db_session_factory, monkeypatch) -> None:
    """A failing roster write must NOT break /me — identity is the contract."""

    async def _boom(self, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated roster write failure")

    monkeypatch.setattr(MembersRepository, "record_seen", _boom)
    subject = uuid4()
    app = _build_app(
        db_session_factory,
        tenant_id=str(uuid4()),
        roles=["member"],
        subject_id=str(subject),
        email="grace@example.com",
        name="Grace",
    )
    async with _client(app) as c:
        me = await c.get("/api/auth/me")
    assert me.status_code == 200, me.text
    body = me.json()
    assert body["subject_id"] == str(subject)
    assert body["email"] == "grace@example.com"


# ─── Repository unit ─────────────────────────────────────────────────────


async def test_repo_record_seen_upserts(db_session) -> None:
    repo = MembersRepository(db_session)
    subject, tenant = uuid7(), uuid7()
    await repo.record_seen(
        subject_id=subject,
        tenant_id=tenant,
        email="a@b.com",
        display_name="Alice",
        roles=["member"],
    )
    # Second sight: roles change, email omitted (None) must NOT clobber.
    await repo.record_seen(
        subject_id=subject,
        tenant_id=tenant,
        email=None,
        display_name=None,
        roles=["member", "tenant_admin"],
    )
    await db_session.flush()

    rows = (
        (await db_session.execute(select(TenantMember).where(TenantMember.tenant_id == tenant)))
        .scalars()
        .all()
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.email == "a@b.com"  # coalesced, not clobbered
    assert row.display_name == "Alice"
    assert row.roles == ["member", "tenant_admin"]
