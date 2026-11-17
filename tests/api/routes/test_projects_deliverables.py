"""POST /api/projects/{id}/deliverables — the create-project wizard's finalize step.

The wizard pre-creates a ``drafting`` project (``POST /api/projects/draft``), then at
Review PATCHes it and adds the deliverable here before starting runs. That last call
had NO route: it 404'd with a bare "Not Found" after the user had already uploaded
footage, so the work was lost. Every test below fails without the route.

Additive: the atomic ``POST /api/projects`` door is untouched (see
``test_projects.py::test_post_project_rejects_zero_deliverables``) and so are the
``/api/deliverables/{id}`` routes (see ``test_deliverables.py``).
"""

import pytest
from httpx import AsyncClient

from tests.conftest import SeededClient

pytestmark = pytest.mark.asyncio

_DELIVERABLE = {
    "name": "hero cut",
    "format": "16x9",
    "captions_mode": "burned_in",
    "target_duration_seconds": 45,
}


async def _draft(client: AsyncClient, title: str = "wizard draft") -> str:
    r = await client.post("/api/projects/draft", json={"type": "promo", "title": title})
    assert r.status_code == 201, r.text
    return str(r.json()["id"])


async def test_add_deliverable_to_drafting_project(client: SeededClient) -> None:
    """THE regression test: the wizard's finalize call must create + return the row."""
    pid = await _draft(client)
    r = await client.post(f"/api/projects/{pid}/deliverables", json=_DELIVERABLE)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["project_id"] == pid
    assert body["name"] == "hero cut"
    assert body["format"] == "16x9"
    assert body["captions_mode"] == "burned_in"
    assert body["target_duration_seconds"] == 45
    # Derived-on-read status: no run yet ⇒ idle.
    assert body["status"] == "idle"
    assert body["id"]


async def test_added_deliverable_matches_project_detail(client: SeededClient) -> None:
    """The wizard reads the project back — the two surfaces must agree field-for-field."""
    pid = await _draft(client, "readback")
    created = await client.post(f"/api/projects/{pid}/deliverables", json=_DELIVERABLE)
    assert created.status_code == 201, created.text

    detail = await client.get(f"/api/projects/{pid}")
    assert detail.status_code == 200, detail.text
    delivs = detail.json()["deliverables"]
    assert len(delivs) == 1
    assert delivs[0] == created.json()


async def test_added_deliverable_is_startable_target(client: SeededClient) -> None:
    """The returned id must be a real, durable deliverable — readable via its own route.

    Commit-before-responding matters here: the wizard's next call uses this id.
    """
    pid = await _draft(client, "durable")
    created = await client.post(f"/api/projects/{pid}/deliverables", json=_DELIVERABLE)
    assert created.status_code == 201, created.text
    did = created.json()["id"]
    r = await client.get(f"/api/deliverables/{did}")
    assert r.status_code == 200, r.text
    assert r.json() == created.json()


async def test_defaults_match_the_atomic_create_path(client: SeededClient) -> None:
    """Omitted optionals default exactly as ``ProjectCreate``'s deliverables do."""
    pid = await _draft(client, "defaults")
    r = await client.post(
        f"/api/projects/{pid}/deliverables",
        json={"name": "minimal", "format": "9x16"},
    )
    assert r.status_code == 201, r.text
    assert r.json()["captions_mode"] == "none"
    assert r.json()["target_duration_seconds"] is None


async def test_other_tenant_project_returns_404(
    client: SeededClient, other_tenant_client: AsyncClient
) -> None:
    """Cross-tenant is 404, not 403 and never a success — no project-existence probe.

    The owner's 201 is a POSITIVE CONTROL: without it a missing route would satisfy
    the 404 assertion for the wrong reason and this test would pass on the bug.
    """
    pid = await _draft(client, "private")
    owned = await client.post(f"/api/projects/{pid}/deliverables", json=_DELIVERABLE)
    assert owned.status_code == 201, owned.text

    cross = await other_tenant_client.post(
        f"/api/projects/{pid}/deliverables",
        json={"name": "intruder", "format": "1x1"},
    )
    assert cross.status_code == 404, cross.text
    # Nothing of the foreign tenant's was written into the owner's project.
    detail = await client.get(f"/api/projects/{pid}")
    assert [d["name"] for d in detail.json()["deliverables"]] == ["hero cut"]


async def test_unknown_project_returns_404(client: SeededClient) -> None:
    # Positive control first (same reason as the cross-tenant test).
    pid = await _draft(client, "known")
    assert (
        await client.post(f"/api/projects/{pid}/deliverables", json=_DELIVERABLE)
    ).status_code == 201

    r = await client.post(
        "/api/projects/00000000-0000-0000-0000-000000000000/deliverables",
        json=_DELIVERABLE,
    )
    assert r.status_code == 404, r.text


async def test_invalid_format_returns_422(client: SeededClient) -> None:
    pid = await _draft(client, "bad format")
    r = await client.post(
        f"/api/projects/{pid}/deliverables",
        json={"name": "x", "format": "21x9"},
    )
    assert r.status_code == 422, r.text
