"""Studio v3 — /api/deliverables route tests (T9)."""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def _create_project_and_deliv(client: AsyncClient) -> str:
    r = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "for deliv",
            "deliverables": [
                {
                    "name": "d",
                    "format": "16x9",
                    "captions_mode": "none",
                    "target_duration_seconds": 60,
                }
            ],
        },
    )
    assert r.status_code == 201
    pid = r.json()["id"]
    detail = await client.get(f"/api/projects/{pid}")
    return detail.json()["deliverables"][0]["id"]


async def test_get_deliverable(client: AsyncClient) -> None:
    did = await _create_project_and_deliv(client)
    r = await client.get(f"/api/deliverables/{did}")
    assert r.status_code == 200
    assert r.json()["id"] == did


async def test_fresh_deliverable_status_is_idle(client: AsyncClient) -> None:
    """No run yet → derived status is 'idle' on both the deliverable and project surfaces."""
    r = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "status wiring",
            "deliverables": [
                {
                    "name": "d",
                    "format": "16x9",
                    "captions_mode": "none",
                    "target_duration_seconds": 60,
                }
            ],
        },
    )
    assert r.status_code == 201
    pid = r.json()["id"]
    # Create response surfaces idle.
    assert r.json()["deliverables"][0]["status"] == "idle"
    did = r.json()["deliverables"][0]["id"]
    # Project-detail surface.
    detail = await client.get(f"/api/projects/{pid}")
    assert detail.json()["deliverables"][0]["status"] == "idle"
    # Single-deliverable surface.
    one = await client.get(f"/api/deliverables/{did}")
    assert one.json()["status"] == "idle"


async def test_patch_deliverable(client: AsyncClient) -> None:
    did = await _create_project_and_deliv(client)
    r = await client.patch(f"/api/deliverables/{did}", json={"name": "renamed"})
    assert r.status_code == 200
    assert r.json()["name"] == "renamed"


async def test_delete_deliverable(client: AsyncClient) -> None:
    did = await _create_project_and_deliv(client)
    r = await client.delete(f"/api/deliverables/{did}")
    assert r.status_code == 204
    follow = await client.get(f"/api/deliverables/{did}")
    assert follow.status_code == 404


async def test_get_unknown_deliverable_returns_404(client: AsyncClient) -> None:
    from uuid import uuid4

    r = await client.get(f"/api/deliverables/{uuid4()}")
    assert r.status_code == 404


async def test_runs_sub_resource_creates_pending_run(client: AsyncClient) -> None:
    from unittest.mock import AsyncMock, patch

    from w2v_common.clients import WMResponse

    did = await _create_project_and_deliv(client)
    fake_lr = AsyncMock()
    fake_lr.message.return_value = WMResponse(
        output={"lines": [{"text": "hi", "seconds": 5.0}]},
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )
    fake_render = AsyncMock()
    fake_render.compile.return_value = {"artifact_url": "http://a/x.mp4", "duration_seconds": 30.0}
    with (
        patch(
            "src.services.stages.primitives.script_stage.WMClient",
            return_value=fake_lr,
        ),
        patch(
            "src.services.stages.primitives.compile_stage.VideoCompileClient",
            return_value=fake_render,
        ),
    ):
        r = await client.post(f"/api/deliverables/{did}/runs", json={})
    assert r.status_code == 201
    assert r.json()["status"] == "pending"
