"""``POST /api/runs/{run_id}/usage`` through the REAL auth stack, with the S2S header
shape w2v_common's ``VLLMUsageReporter`` sends.

w2v_common ``AuthMiddleware`` (licence lane, introspection faked at the function
seam) → ``TenancyMiddleware`` (membership mode, as prod) → the routers mounted as
``asgi.py`` mounts them (usage router WITHOUT the gate, runs router WITH
``TENANT_GATE``) → real postgres.

GLOBAL delegation (``AUTH_DELEGATING_SUBJECTS``) is OFF. The service account is only
on the ROUTE-LOCAL ``USAGE_REPORT_DELEGATING_SUBJECTS``. Pins:

* SA licence + ``X-Effective-Tenant-Id: <run tenant>`` → 202 on the usage route, one
  row, attributed to the run's subject (not the service account);
* the SAME headers on ``GET /api/runs/{id}`` act as the SA's own tenant → 404: the
  allowlist re-scopes nothing outside the usage route;
* no delegation header / a tenant that does not own the run / a non-allowlisted
  subject asserting the tenant → 404, nothing written; a malformed header → 422.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from sqlalchemy import select
from w2v_common.asgi import create_app
from w2v_common.auth import middleware as auth_mw

from src.api.routes._common import TENANT_GATE
from src.api.routes.runs import router as runs_router
from src.api.routes.runs import usage_router
from src.auth.tenancy_middleware import install_tenancy_middleware
from src.config import clear_settings_cache, get_settings
from src.database.models import Deliverable, Project, Run, RunUsage
from src.database.repositories.admin_repository import AdminRepository

pytestmark = pytest.mark.asyncio

_SA_TOKEN = "sa-licence-token-not-a-jwt"  # pragma: allowlist secret - fake
_CUSTOMER_TOKEN = "customer-licence-token-not-a-jwt"  # pragma: allowlist secret - fake
_PUBLIC_PEER = ("203.0.113.9", 50000)  # outside TRUSTED_NETWORKS
SA_SUBJECT = str(uuid.uuid7())
SA_TENANT = uuid.uuid7()
CUSTOMER_SUBJECT = str(uuid.uuid7())


@pytest.fixture
def s2s_app(monkeypatch: pytest.MonkeyPatch, db_session_factory: Any) -> Iterator[Any]:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("W2V_TENANCY_SOURCE", "membership")
    monkeypatch.delenv("AUTH_DELEGATING_SUBJECTS", raising=False)  # global: OFF
    monkeypatch.setenv("USAGE_REPORT_DELEGATING_SUBJECTS", f" {SA_SUBJECT} , other")
    monkeypatch.setenv("CELL_AUTH_MODE", "dual")
    # Tenant scoping is what is under test; the w2v licence gate has its own suite.
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    clear_settings_cache()

    async def _introspect(token: str) -> dict[str, object] | None:
        subjects = {_SA_TOKEN: SA_SUBJECT, _CUSTOMER_TOKEN: CUSTOMER_SUBJECT}
        if token not in subjects:
            return None
        return {"active": True, "tenant_id": str(SA_TENANT), "subject_id": subjects[token]}

    monkeypatch.setattr(auth_mw, "_introspect_license", _introspect)
    app = create_app(get_settings(), routers=[usage_router], cors_enabled=False)
    app.include_router(runs_router, dependencies=TENANT_GATE)
    install_tenancy_middleware(app)
    app.state.session_factory = db_session_factory
    yield app
    clear_settings_cache()


async def _run_for(factory: Any, tenant: uuid.UUID, subject: uuid.UUID | None = None) -> uuid.UUID:
    async with factory() as s:
        project = Project(
            id=uuid.uuid7(),
            tenant_id=tenant,
            owner_subject_id=uuid.uuid7(),
            visibility="private",
            type="generic",
            title="s2s",
        )
        deliverable = Deliverable(id=uuid.uuid7(), project_id=project.id, name="d", format="16x9")
        run = Run(
            id=uuid.uuid7(),
            deliverable_id=deliverable.id,
            blueprint_id="generic_v1",
            status="running",
            config_snapshot={"subject_id": str(subject)} if subject else {},
            pause_after_stages=[],
        )
        s.add(project)
        await s.flush()
        s.add(deliverable)
        await s.flush()
        s.add(run)
        await s.commit()
        return run.id


def _headers(run_id: uuid.UUID, *, token: str = _SA_TOKEN, effective: object = None) -> dict:
    """The reporter's wire shape: SA licence + identity_event_hook's forwarded set."""
    headers = {
        "X-Tenant-Id": f"Bearer {token}",
        "X-Subject-Id": str(uuid.uuid7()),
        "X-Correlation-Id": str(run_id),
        "X-W2V-Job-Id": str(run_id),
        "X-Stage": "analysis_summary",
    }
    if effective is not None:
        headers["X-Effective-Tenant-Id"] = str(effective)
    return headers


_BODY = {
    "call_id": "s2s-call-1",
    "stage": "analysis_summary",
    "model": "local-llm",
    "provider": "vllm",
    "input_tokens": 12,
    "output_tokens": 3,
    "total_tokens": 15,
    "service": "analysis-service",
}


async def _call(app: Any, method: str, path: str, headers: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, client=_PUBLIC_PEER)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        body = _BODY if method == "POST" else None
        return await c.request(method, path, json=body, headers=headers)


async def _post(app: Any, run_id: uuid.UUID, headers: dict) -> httpx.Response:
    return await _call(app, "POST", f"/api/runs/{run_id}/usage", headers)


async def _rows(factory: Any, run_id: uuid.UUID) -> list[RunUsage]:
    async with factory() as s:
        return list(
            (await s.execute(select(RunUsage).where(RunUsage.run_id == run_id))).scalars().all()
        )


async def test_delegated_report_for_the_runs_tenant_is_recorded(
    s2s_app: Any, db_session_factory: Any
) -> None:
    tenant, run_subject = uuid.uuid7(), uuid.uuid7()
    run_id = await _run_for(db_session_factory, tenant, run_subject)

    resp = await _post(s2s_app, run_id, _headers(run_id, effective=tenant))

    assert resp.status_code == 202, resp.text
    assert resp.json() == {"recorded": True}
    (row,) = await _rows(db_session_factory, run_id)
    assert (row.input_tokens, row.output_tokens, row.funding) == (12, 3, "local_gpu")
    assert row.subject_id == run_subject, "a delegated report is the run's, not the SA's"


async def test_same_delegated_headers_do_not_rescope_another_run_route(
    s2s_app: Any, db_session_factory: Any
) -> None:
    """The allowlist is route-local: GET /api/runs/{id} still acts as the SA tenant."""
    async with db_session_factory() as s:
        await AdminRepository(s).ensure_tenant(SA_TENANT, created_via="super_admin")
        await s.commit()
    tenant = uuid.uuid7()
    run_id = await _run_for(db_session_factory, tenant)
    headers = _headers(run_id, effective=tenant)

    detail = await _call(s2s_app, "GET", f"/api/runs/{run_id}", headers)
    status_lite = await _call(s2s_app, "GET", f"/api/runs/{run_id}/status", headers)

    assert detail.status_code == 404, detail.text
    assert "run not found" in detail.text, detail.text
    assert status_lite.status_code == 404, status_lite.text
    # ...while the usage route, with the very same headers, is served.
    assert (await _post(s2s_app, run_id, headers)).status_code == 202


async def test_without_delegation_the_sa_tenant_stands_and_nothing_is_written(
    s2s_app: Any, db_session_factory: Any
) -> None:
    """Forwarded correlation / job / subject headers never name a tenant."""
    run_id = await _run_for(db_session_factory, uuid.uuid7())

    resp = await _post(s2s_app, run_id, _headers(run_id))

    assert resp.status_code == 404, resp.text
    assert await _rows(db_session_factory, run_id) == []


async def test_delegation_to_a_tenant_that_does_not_own_the_run_is_404(
    s2s_app: Any, db_session_factory: Any
) -> None:
    run_id = await _run_for(db_session_factory, uuid.uuid7())

    resp = await _post(s2s_app, run_id, _headers(run_id, effective=uuid.uuid7()))

    assert resp.status_code == 404, resp.text
    assert await _rows(db_session_factory, run_id) == []


async def test_non_allowlisted_subject_cannot_assert_the_runs_tenant(
    s2s_app: Any, db_session_factory: Any
) -> None:
    tenant = uuid.uuid7()
    run_id = await _run_for(db_session_factory, tenant)

    resp = await _post(s2s_app, run_id, _headers(run_id, token=_CUSTOMER_TOKEN, effective=tenant))

    assert resp.status_code == 404, resp.text
    assert await _rows(db_session_factory, run_id) == []


async def test_malformed_effective_tenant_from_an_allowlisted_subject_is_422(
    s2s_app: Any, db_session_factory: Any
) -> None:
    run_id = await _run_for(db_session_factory, uuid.uuid7())

    resp = await _post(s2s_app, run_id, _headers(run_id, effective="not-a-uuid"))

    assert resp.status_code == 422, resp.text
    assert await _rows(db_session_factory, run_id) == []


async def test_empty_allowlist_honours_no_delegation(
    s2s_app: Any, db_session_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USAGE_REPORT_DELEGATING_SUBJECTS", "")
    clear_settings_cache()
    tenant = uuid.uuid7()
    run_id = await _run_for(db_session_factory, tenant)

    resp = await _post(s2s_app, run_id, _headers(run_id, effective=tenant))

    assert resp.status_code == 404, resp.text
    assert await _rows(db_session_factory, run_id) == []
