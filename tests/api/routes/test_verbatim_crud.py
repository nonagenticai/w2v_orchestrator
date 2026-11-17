"""DB-backed integration tests for verbatim-items CRUD.

Covers /api/projects/{project_id}/verbatim-items (list, create) and
/api/verbatim-items/{item_id} (patch, delete). Tenant scoping is enforced
via VerbatimItem.project_id -> Project.tenant_id; cross-tenant reads return
404 (or [] for the listing endpoint).
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from src.database.models import Project

pytestmark = pytest.mark.asyncio


async def _seed_project(db_session, *, tenant_id: UUID, subject_id: UUID) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="verbatim-crud",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def test_create_then_list_round_trip_with_auto_position(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    a = await client.post(
        f"/api/projects/{project.id}/verbatim-items",
        json={"text": "first", "kind": "slide"},
    )
    b = await client.post(
        f"/api/projects/{project.id}/verbatim-items",
        json={"text": "second", "kind": "tts", "hint": "say it slow"},
    )
    assert a.status_code == 201 and b.status_code == 201, (a.text, b.text)
    a_body = a.json()
    assert a_body["position"] == 0
    assert a_body["kind"] == "slide"
    # `pending` is the CORRECT answer here and stays asserted: placement is a
    # property of a RUN, and neither create nor a run-less list has one to match
    # against. The derived `placed` path is exercised in
    # tests/api/routes/test_verbatim_run_placement.py (`?run_id=`).
    assert a_body["status"] == "pending"
    assert a_body["retry_count"] == 0
    assert b.json()["position"] == 1
    assert b.json()["hint"] == "say it slow"

    listing = await client.get(f"/api/projects/{project.id}/verbatim-items")
    assert listing.status_code == 200
    body = listing.json()
    # SPA envelope contract: {items, total} matches $lib/types VerbatimItemListResponse.
    assert body["total"] == 2
    items = body["items"]
    assert [i["text"] for i in items] == ["first", "second"]
    assert [i["position"] for i in items] == [0, 1]
    assert [i["kind"] for i in items] == ["slide", "tts"]
    # tenant identity is inherited from the parent Project on responses.
    assert items[0]["tenant_id"] == client.tenant_id


async def test_create_with_explicit_position(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    r = await client.post(
        f"/api/projects/{project.id}/verbatim-items",
        json={"text": "pinned", "position": 7},
    )
    assert r.status_code == 201
    assert r.json()["position"] == 7


async def test_patch_text_and_position(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    created = await client.post(
        f"/api/projects/{project.id}/verbatim-items",
        json={"text": "v1"},
    )
    item_id = created.json()["id"]

    patched = await client.patch(
        f"/api/verbatim-items/{item_id}",
        json={"text": "v2", "position": 5},
    )
    assert patched.status_code == 200, patched.text
    body = patched.json()
    assert body["text"] == "v2"
    assert body["position"] == 5


async def test_delete_round_trip(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    created = await client.post(
        f"/api/projects/{project.id}/verbatim-items",
        json={"text": "doomed"},
    )
    item_id = created.json()["id"]

    deleted = await client.delete(f"/api/verbatim-items/{item_id}")
    assert deleted.status_code == 204

    again = await client.delete(f"/api/verbatim-items/{item_id}")
    assert again.status_code == 404


async def test_list_unknown_project_returns_404(client) -> None:
    r = await client.get(f"/api/projects/{uuid4()}/verbatim-items")
    assert r.status_code == 404


async def test_cross_tenant_verbatim_isolation(
    client, other_tenant_client: AsyncClient, db_session
) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    created = await client.post(
        f"/api/projects/{project.id}/verbatim-items",
        json={"text": "secret"},
    )
    assert created.status_code == 201
    item_id = created.json()["id"]

    # Cross-tenant list of someone else's project: 404 from project-scope check.
    bleed_list = await other_tenant_client.get(f"/api/projects/{project.id}/verbatim-items")
    assert bleed_list.status_code == 404

    # Cross-tenant patch / delete: 404 (item exists, but tenant scoping hides it).
    bleed_patch = await other_tenant_client.patch(
        f"/api/verbatim-items/{item_id}", json={"text": "stolen"}
    )
    assert bleed_patch.status_code == 404
    bleed_del = await other_tenant_client.delete(f"/api/verbatim-items/{item_id}")
    assert bleed_del.status_code == 404

    # Original tenant still sees the unmodified item.
    mine = await client.get(f"/api/projects/{project.id}/verbatim-items")
    assert mine.json()["items"][0]["text"] == "secret"
