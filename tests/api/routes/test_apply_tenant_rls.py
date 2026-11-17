"""Phase D RLS — the flag-gated ``apply_tenant_rls`` request dependency.

``off`` (default) is a no-op; ``enforce`` issues a parameterized,
transaction-scoped ``set_config('app.current_tenant', <tenant>, true)`` on the
request session. No DB needed — a recording fake session is enough.
"""

from __future__ import annotations

from typing import cast
from uuid import uuid7

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes._common import apply_tenant_rls
from src.config import clear_settings_cache

pytestmark = pytest.mark.asyncio


class _RecordingSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None]] = []
        # Mirrors ``Session.info`` — bind_session_tenant records the bound tenant
        # there so the after_begin re-bind listener can prefer it over the (claim)
        # identity ContextVar. See src/database/session.py:SESSION_TENANT_KEY.
        self.info: dict[str, object] = {}

    async def execute(self, stmt, params=None):
        self.calls.append((str(stmt), params))


@pytest.fixture(autouse=True)
def _reset_settings():
    clear_settings_cache()
    yield
    clear_settings_cache()


async def test_off_is_a_noop(monkeypatch) -> None:
    monkeypatch.setenv("TENANT_RLS_MODE", "off")
    clear_settings_cache()
    session = _RecordingSession()
    await apply_tenant_rls(tenant_id=uuid7(), session=cast(AsyncSession, session))
    assert session.calls == []


async def test_enforce_sets_local_guc(monkeypatch) -> None:
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    session = _RecordingSession()
    tenant = uuid7()
    await apply_tenant_rls(tenant_id=tenant, session=cast(AsyncSession, session))
    assert len(session.calls) == 1
    sql, params = session.calls[0]
    assert "set_config('app.current_tenant'" in sql
    assert "true" in sql  # is_local — transaction-scoped
    assert params == {"tenant": str(tenant)}
    # The EFFECTIVE tenant is also recorded on the session so the after_begin
    # listener re-binds it (not the token claim) after any intra-request commit.
    assert session.info["w2v_bound_tenant"] == str(tenant)
