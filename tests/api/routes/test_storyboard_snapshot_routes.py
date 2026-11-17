"""HTTP surface for storyboard version snapshots + restore (Phase 5).

Exercises the three routes mounted by ``src/api/routes/storyboard.py``:

* ``POST   /api/projects/{project_id}/storyboard/snapshots`` — create manual (201)
* ``GET    /api/projects/{project_id}/storyboard/snapshots`` — list, newest-first,
  bounded by ``limit``/``offset`` with a project-wide ``total``
* ``DELETE /api/projects/{project_id}/storyboard/snapshots/{snapshot_id}`` — delete a
  manual snapshot (204); ``auto`` snapshots are machine-managed → 409
* ``POST   /api/storyboard/snapshots/{snapshot_id}/restore`` — restore (200, both ids)

plus the cross-tenant gates (list → 404 on a foreign project, restore → 404 on a
foreign snapshot). Mirrors ``test_storyboard_approval_routes.py`` (client seeds a
tenant/subject pair; ``other_tenant_client`` proves isolation).
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
        title="snapshot-routes",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def _seed_panel(client, project_id, *, vo_text: str = "x") -> str:
    r = await client.post(
        f"/api/projects/{project_id}/storyboard/panels", json={"vo_text": vo_text}
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def _board_map(client, project_id) -> dict[str, int]:
    """The whole-board ``{row id: lock_version}`` map that ``restore`` now REQUIRES (400
    without it, 409 when stale — pinned in test_storyboard_destructive_lock.py)."""
    merged: dict[str, int] = {}
    for kind in ("panels", "scenes", "assets"):
        r = await client.get(f"/api/projects/{project_id}/storyboard/{kind}")
        assert r.status_code == 200, r.text
        merged.update({row["id"]: row["lock_version"] for row in r.json()["items"]})
    return merged


async def test_create_list_restore_roundtrip(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _seed_panel(client, project.id, vo_text="original")

    # create (201)
    created = await client.post(
        f"/api/projects/{project.id}/storyboard/snapshots", json={"note": "v1"}
    )
    assert created.status_code == 201, created.text
    snap = created.json()
    snapshot_id = snap["id"]
    assert snap["kind"] == "manual"
    assert snap["note"] == "v1"
    assert snap["actor_subject"] == client.subject_id
    assert snap["panel_count"] == 1

    # list (contains it)
    listing = await client.get(f"/api/projects/{project.id}/storyboard/snapshots")
    assert listing.status_code == 200, listing.text
    body = listing.json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == snapshot_id

    # mutate then restore (200, returns both ids)
    panels = (await client.get(f"/api/projects/{project.id}/storyboard/panels")).json()
    panel_id = panels["items"][0]["id"]
    await client.patch(f"/api/storyboard/panels/{panel_id}", json={"vo_text": "changed"})

    restored = await client.post(
        f"/api/storyboard/snapshots/{snapshot_id}/restore",
        json={"expected_lock_versions": await _board_map(client, project.id)},
    )
    assert restored.status_code == 200, restored.text
    rbody = restored.json()
    assert rbody["restored_snapshot_id"] == snapshot_id
    assert UUID(rbody["auto_snapshot_id"]) != UUID(snapshot_id)

    # content rolled back
    after = (await client.get(f"/api/projects/{project.id}/storyboard/panels")).json()
    assert after["items"][0]["vo_text"] == "original"

    # the auto safety snapshot now shows up in the listing (manual + auto)
    listing2 = (await client.get(f"/api/projects/{project.id}/storyboard/snapshots")).json()
    assert listing2["total"] == 2
    assert {i["kind"] for i in listing2["items"]} == {"manual", "auto"}


async def test_list_newest_first(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _seed_panel(client, project.id)
    first = await client.post(
        f"/api/projects/{project.id}/storyboard/snapshots", json={"note": "first"}
    )
    second = await client.post(
        f"/api/projects/{project.id}/storyboard/snapshots", json={"note": "second"}
    )
    assert first.status_code == 201 and second.status_code == 201

    listing = (await client.get(f"/api/projects/{project.id}/storyboard/snapshots")).json()
    assert [i["id"] for i in listing["items"]] == [second.json()["id"], first.json()["id"]]


async def test_cross_tenant_list_404(client, other_tenant_client: AsyncClient, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _seed_panel(client, project.id)
    await client.post(f"/api/projects/{project.id}/storyboard/snapshots", json={})

    bleed = await other_tenant_client.get(f"/api/projects/{project.id}/storyboard/snapshots")
    assert bleed.status_code == 404, bleed.text


async def test_cross_tenant_restore_404(
    client, other_tenant_client: AsyncClient, db_session
) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _seed_panel(client, project.id)
    created = await client.post(f"/api/projects/{project.id}/storyboard/snapshots", json={})
    snapshot_id = created.json()["id"]

    bleed = await other_tenant_client.post(
        f"/api/storyboard/snapshots/{snapshot_id}/restore",
        json={"expected_lock_versions": {}},
    )
    assert bleed.status_code == 404, bleed.text


async def test_restore_unknown_snapshot_404(client) -> None:
    r = await client.post(
        f"/api/storyboard/snapshots/{uuid4()}/restore", json={"expected_lock_versions": {}}
    )
    assert r.status_code == 404, r.text


# ── bounded listing ──────────────────────────────────────────────────────────
async def test_list_is_paginated_with_a_true_total(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _seed_panel(client, project.id)
    ids = []
    for i in range(3):
        created = await client.post(
            f"/api/projects/{project.id}/storyboard/snapshots", json={"note": f"v{i}"}
        )
        assert created.status_code == 201, created.text
        ids.append(created.json()["id"])
    newest_first = list(reversed(ids))

    first = (
        await client.get(f"/api/projects/{project.id}/storyboard/snapshots?limit=1&offset=0")
    ).json()
    assert [i["id"] for i in first["items"]] == [newest_first[0]]
    assert first["total"] == 3  # the project-wide total, not len(items)

    second = (
        await client.get(f"/api/projects/{project.id}/storyboard/snapshots?limit=2&offset=1")
    ).json()
    assert [i["id"] for i in second["items"]] == newest_first[1:]
    assert second["total"] == 3

    # limit is bounded (le=100)
    bad = await client.get(f"/api/projects/{project.id}/storyboard/snapshots?limit=101")
    assert bad.status_code == 422, bad.text


# ── delete ───────────────────────────────────────────────────────────────────
async def test_delete_manual_snapshot_204_then_gone(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _seed_panel(client, project.id)
    created = await client.post(f"/api/projects/{project.id}/storyboard/snapshots", json={})
    snapshot_id = created.json()["id"]

    deleted = await client.delete(f"/api/projects/{project.id}/storyboard/snapshots/{snapshot_id}")
    assert deleted.status_code == 204, deleted.text

    listing = (await client.get(f"/api/projects/{project.id}/storyboard/snapshots")).json()
    assert listing["total"] == 0
    assert listing["items"] == []

    # second delete → 404
    again = await client.delete(f"/api/projects/{project.id}/storyboard/snapshots/{snapshot_id}")
    assert again.status_code == 404, again.text


async def test_delete_auto_snapshot_409(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _seed_panel(client, project.id)
    created = await client.post(f"/api/projects/{project.id}/storyboard/snapshots", json={})
    snapshot_id = created.json()["id"]

    # a restore mints the machine-managed `auto` safety snapshot
    restored = await client.post(
        f"/api/storyboard/snapshots/{snapshot_id}/restore",
        json={"expected_lock_versions": await _board_map(client, project.id)},
    )
    auto_id = restored.json()["auto_snapshot_id"]

    conflict = await client.delete(f"/api/projects/{project.id}/storyboard/snapshots/{auto_id}")
    assert conflict.status_code == 409, conflict.text

    listing = (await client.get(f"/api/projects/{project.id}/storyboard/snapshots")).json()
    assert auto_id in {i["id"] for i in listing["items"]}


async def test_delete_snapshot_of_another_project_404(client, db_session) -> None:
    tenant_id = UUID(client.tenant_id)
    subject_id = UUID(client.subject_id)
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    other = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    await _seed_panel(client, project.id)
    created = await client.post(f"/api/projects/{project.id}/storyboard/snapshots", json={})
    snapshot_id = created.json()["id"]

    bleed = await client.delete(f"/api/projects/{other.id}/storyboard/snapshots/{snapshot_id}")
    assert bleed.status_code == 404, bleed.text


async def test_cross_tenant_delete_404(
    client, other_tenant_client: AsyncClient, db_session
) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _seed_panel(client, project.id)
    created = await client.post(f"/api/projects/{project.id}/storyboard/snapshots", json={})
    snapshot_id = created.json()["id"]

    bleed = await other_tenant_client.delete(
        f"/api/projects/{project.id}/storyboard/snapshots/{snapshot_id}"
    )
    assert bleed.status_code == 404, bleed.text
