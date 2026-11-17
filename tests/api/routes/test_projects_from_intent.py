"""API tests for the A0 authoring front door — POST /api/projects/from-intent.

Real postgres (conftest `client`/`app`), so these cannot silently skip. The run
engine's ``start_run`` is mocked so the route's own wiring — project + deliverable
creation, the fail-fast 422 guard, and the response shape — is exercised without
dispatching the whole DAG (the plan-materialization itself is covered by the pure
``tests/services/runs/test_run_setup_intent.py``).
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


async def test_from_intent_showcase_with_package_starts_run(
    client: AsyncClient, app: FastAPI
) -> None:
    run = _mock_start_run(app)
    r = await client.post(
        "/api/projects/from-intent",
        json={
            "intent": {
                "type": "showcase",
                "title": "Meet the editor",
                "url": "https://example.com",
                "package": "minio://w2v/pkg.zip",
                "must_show": ["home", "editor"],
                "must_say": ["Meet the editor."],
            }
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["run_id"] == str(run.id)
    assert body["events_url"] == f"/api/runs/{run.id}/events"
    assert body["plan_preview_url"] == f"/api/runs/{run.id}"
    # The project + deliverable were really created (real DB).
    proj = await client.get(f"/api/projects/{body['project_id']}")
    assert proj.status_code == 200
    assert proj.json()["deliverables"][0]["id"] == body["deliverable_id"]


async def test_from_intent_showcase_without_package_is_422(
    client: AsyncClient, app: FastAPI
) -> None:
    _mock_start_run(app)
    r = await client.post(
        "/api/projects/from-intent",
        json={
            "intent": {
                "type": "showcase",
                "title": "No footage",
                "url": "https://example.com",
                "must_show": ["home"],
            }
        },
    )
    assert r.status_code == 422
    assert "package" in r.text.lower()


async def test_from_intent_generic_without_package_ok(client: AsyncClient, app: FastAPI) -> None:
    """A generated type needs no footage package — the generated pipeline builds it
    from target_url, so no 422 and the run starts."""
    run = _mock_start_run(app)
    r = await client.post(
        "/api/projects/from-intent",
        json={
            "intent": {
                "type": "generic",
                "title": "A generic explainer",
                "url": "https://example.com",
            }
        },
    )
    assert r.status_code == 201, r.text
    assert r.json()["run_id"] == str(run.id)
