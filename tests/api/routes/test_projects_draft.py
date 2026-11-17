"""POST /api/projects/draft — the create-project wizard's EARLY draft path.

Additive endpoint: creates a real project row with a chosen ``type`` but NO
deliverables yet (status server-defaults to ``drafting``; deliverables are a
child table, so zero rows is legal). The strict ``ProjectCreate`` used by the
normal final-submit / n8n path is deliberately left untouched — see
``test_projects.py::test_post_project_rejects_zero_deliverables`` which still
guards that door.
"""

import pytest
from httpx import AsyncClient

from tests.conftest import SeededClient

pytestmark = pytest.mark.asyncio


async def test_draft_creates_project_with_zero_deliverables(client: SeededClient) -> None:
    r = await client.post("/api/projects/draft", json={"type": "promo", "title": "wizard draft"})
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["type"] == "promo"
    assert data["title"] == "wizard draft"
    assert data["status"] == "drafting"
    # No deliverables yet — both on the create response and on a follow-up GET.
    assert data["deliverables"] == []
    detail = await client.get(f"/api/projects/{data['id']}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["deliverables"] == []


async def test_draft_invalid_type_returns_422(client: SeededClient) -> None:
    # Unknown type is rejected at the ProjectDraftCreate Literal door (DB CHECK backstop).
    r = await client.post("/api/projects/draft", json={"type": "not_a_real_type", "title": "x"})
    assert r.status_code == 422, r.text


async def test_draft_omitting_title_uses_default(client: SeededClient) -> None:
    r = await client.post("/api/projects/draft", json={"type": "generic"})
    assert r.status_code == 201, r.text
    assert r.json()["title"] == "Untitled"


async def test_draft_owned_by_caller_tenant(client: SeededClient) -> None:
    r = await client.post("/api/projects/draft", json={"type": "generic", "title": "mine"})
    assert r.status_code == 201, r.text
    assert r.json()["tenant_id"] == client.tenant_id
    assert r.json()["owner_subject_id"] == client.subject_id


async def test_draft_cross_tenant_get_returns_404(
    client: SeededClient, other_tenant_client: AsyncClient
) -> None:
    r = await client.post("/api/projects/draft", json={"type": "generic", "title": "private-draft"})
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    cross = await other_tenant_client.get(f"/api/projects/{pid}")
    assert cross.status_code == 404
