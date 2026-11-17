"""API tests for POST /api/audit — the one-command url→audit sugar route.

Real postgres (conftest ``client``/``app``), so these cannot silently skip. The run
engine's ``start_run`` is mocked so the route's own wiring — project + deliverable
creation, brief assembly, defaults, and the response shape — is exercised without
dispatching the whole audit DAG.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


def _mock_start_run(app: FastAPI) -> SimpleNamespace:
    run = SimpleNamespace(id=uuid7(), pause_after_stages=[])
    app.state.run_engine.start_run = AsyncMock(return_value=run)
    return run


async def test_audit_one_command_creates_and_starts(client: AsyncClient, app: FastAPI) -> None:
    run = _mock_start_run(app)
    r = await client.post("/api/audit", json={"url": "https://example.com", "intent": "B2B devs"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["runId"] == str(run.id)
    assert body["reportPdfUrl"] == f"/api/runs/{run.id}/report/pdf"
    assert body["eventsUrl"] == f"/api/runs/{run.id}/events"
    # The project + deliverable were really created (real DB); the brief carries the
    # audit inputs and the one-command defaults (marketing_scorecard + include_llm on).
    proj = await client.get(f"/api/projects/{body['projectId']}")
    assert proj.status_code == 200
    pj = proj.json()
    assert pj["type"] == "marketing_scorecard"
    assert pj["deliverables"][0]["id"] == body["deliverableId"]
    assert pj["brief"]["target_url"] == "https://example.com"
    assert pj["brief"]["include_llm"] is True
    assert pj["brief"]["intent"] == "B2B devs"


async def test_audit_type_override_and_no_intent(client: AsyncClient, app: FastAPI) -> None:
    _mock_start_run(app)
    r = await client.post(
        "/api/audit",
        json={"url": "https://x.test", "auditType": "seo_audit", "includeLlm": False},
    )
    assert r.status_code == 201, r.text
    pj = (await client.get(f"/api/projects/{r.json()['projectId']}")).json()
    assert pj["type"] == "seo_audit"
    assert pj["brief"]["include_llm"] is False
    assert "intent" not in pj["brief"]  # omitted when not provided


@pytest.mark.parametrize(
    "bad",
    [{}],
)
async def test_audit_missing_blank_or_non_http_url_422(
    client: AsyncClient, app: FastAPI, bad: dict
) -> None:
    _mock_start_run(app)
    r = await client.post("/api/audit", json=bad)
    assert r.status_code == 422, r.text
