"""The backfill used to scan nothing under RLS and call it success.

Measured against prod 2026-08-28: `scanned=0 would_change=0`, exit 0, while the
database held 183 projects (164 non-archived). `projects` carries
`tenant_id = current_setting('app.current_tenant')` and `w2v_orch_app` has no
BYPASSRLS, so an unbound tenant admits zero rows. The old log line was
indistinguishable from a genuinely reconciled estate.

These tests pin the two properties that make that impossible to repeat:
  1. the walk BINDS a tenant before querying, once per tenant;
  2. scanning zero projects is a FAILURE (exit 1), never a quiet success.

They deliberately stub the DB layer: the defect is in the script's control flow,
not in SQL, and a test that needs a live RLS-enabled Postgres to prove a control
flow bug is a test that will be skipped in CI and prove nothing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "backfill_project_status.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("backfill_project_status", SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot load {SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["backfill_project_status"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def script():
    return _load_script()


@pytest.mark.asyncio
async def test_zero_scanned_across_real_tenants_is_a_failure(script, monkeypatch):
    """The exact prod shape: tenants exist, projects invisible, nothing scanned.

    Before this change the script logged `scanned=0 would_change=0` and exited 0.
    """
    tenants = [uuid4(), uuid4()]

    async def _fake_backfill(_maker, _tenant, _dry):
        return (0, 0)

    monkeypatch.setattr(script, "get_session_maker", _unused_session_maker)
    monkeypatch.setattr(script, "_backfill_tenant", _fake_backfill)

    rc = await script.main(dry_run=True, tenants=tenants)
    assert rc == 1, "scanning zero projects while tenants exist must FAIL, not pass"


@pytest.mark.asyncio
async def test_no_tenants_at_all_is_a_failure(script, monkeypatch):
    monkeypatch.setattr(script, "get_session_maker", _unused_session_maker)

    async def _no_tenants(_maker):
        return []

    monkeypatch.setattr(script, "_known_tenants", _no_tenants)
    rc = await script.main(dry_run=True, tenants=None)
    assert rc == 1


@pytest.mark.asyncio
async def test_a_real_walk_succeeds_and_visits_every_tenant(script, monkeypatch):
    """Positive control — without it the two failure tests above pass vacuously
    against a script that returns 1 unconditionally."""
    tenants = [uuid4(), uuid4(), uuid4()]
    seen: list[UUID] = []

    async def _fake_backfill(_maker, tenant, _dry):
        seen.append(tenant)
        return (7, 1)

    monkeypatch.setattr(script, "get_session_maker", _unused_session_maker)
    monkeypatch.setattr(script, "_backfill_tenant", _fake_backfill)

    rc = await script.main(dry_run=True, tenants=tenants)
    assert rc == 0
    assert seen == tenants, "every tenant must be walked, not just the first"


@pytest.mark.asyncio
async def test_backfill_binds_the_tenant_before_the_first_query(script, monkeypatch):
    """The whole defect in one assertion: the RLS bind must happen, and it must
    happen BEFORE the project query, or the policy admits nothing."""
    order: list[str] = []
    tenant = uuid4()

    async def _bind(_session, tid):
        order.append(f"bind:{tid}")

    class _Result:
        def scalars(self):
            order.append("query")
            return SimpleNamespace(all=lambda: [])

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, *_a, **_k):
            return _Result()

        async def rollback(self):
            order.append("rollback")

        async def commit(self):
            order.append("commit")

    monkeypatch.setattr(script, "bind_session_tenant", _bind)
    scanned, changed = await script._backfill_tenant(lambda: _Session(), tenant, True)

    assert scanned == 0 and changed == 0
    assert order[0] == f"bind:{tenant}", f"tenant must be bound first, got {order}"
    assert "query" in order and order.index(f"bind:{tenant}") < order.index("query")


async def _unused_session_maker():
    """Stands in for `get_session_maker`, which `main()` awaits before doing anything.

    Every test here stubs whatever would consume the result, so the sentinel it hands
    back must never actually be called — if it is, the test is exercising a real DB
    path it was written to avoid, and it says so instead of quietly connecting.
    """

    def _maker():
        raise AssertionError("session maker must not be used in these tests")

    return _maker
