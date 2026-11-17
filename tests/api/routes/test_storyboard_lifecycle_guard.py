"""Lifecycle guard: an ``archived`` project's storyboard is READ-ONLY (→ 409).

``projects.status == 'archived'`` is the sole terminal/read-only state (a merged or
abandoned branch child, or a root the user soft-deleted). Every storyboard MUTATION is
refused with ``409 {"detail": "project is archived and read-only"}``; reads stay open.
The guard lives at the repository chokepoint (the by-id routes carry no project_id) and
is surfaced by the ONE ``ProjectArchivedError`` ASGI handler. Covered here end-to-end
over HTTP on real postgres — no skips, asserting the delivered status + body.

Also pins the low-hanging fixes shipped with the guard: ``reorder`` on a
missing/cross-tenant project is a 404 (not a 400).

Mirrors ``test_storyboard_routes.py`` fixtures (``client`` + ``db_session``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import Depends, FastAPI
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.clients import WMResponse

from src.api.routes._common import get_db_session
from src.api.routes.storyboard import get_assist_service
from src.database.models import Project
from src.services.storyboard.assist import StoryboardAssistService

pytestmark = pytest.mark.asyncio


async def _board_map(client, project_id) -> dict[str, int]:
    """Current ``{panel id: lock_version}`` map — reorder now REQUIRES one."""
    r = await client.get(f"/api/projects/{project_id}/storyboard/panels")
    assert r.status_code == 200, r.text
    return {row["id"]: row["lock_version"] for row in r.json()["items"]}


_ARCHIVED_DETAIL = "project is archived and read-only"


async def _seed_project(
    db_session, *, tenant_id: UUID, subject_id: UUID, status: str = "drafting"
) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="lifecycle-guard",
        brief={"goal": "ship"},
        brand={},
        voice={},
        theme={},
        status=status,
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def _archive(db_session, project_id: UUID) -> None:
    """Flip a project to ``archived`` directly in the DB (the app reads it on the next
    request via a fresh session), so the guard is exercised without depending on the
    project soft-delete route."""
    await db_session.execute(
        update(Project).where(Project.id == project_id).values(status="archived")
    )
    await db_session.commit()


async def _panel(client, project_id, *, shot_key: str | None = None, vo_text: str = "x") -> str:
    body: dict[str, Any] = {"vo_text": vo_text}
    if shot_key is not None:
        body["shot_key"] = shot_key
    r = await client.post(f"/api/projects/{project_id}/storyboard/panels", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def _scene(client, project_id) -> str:
    r = await client.post(f"/api/projects/{project_id}/storyboard/scenes", json={"name": "Opening"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def _asset(client, project_id, *, name: str = "Hero") -> str:
    r = await client.post(
        f"/api/projects/{project_id}/storyboard/assets", json={"kind": "character", "name": name}
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _install_fake_assist(app: FastAPI, fake: AsyncMock) -> None:
    def _factory(session: AsyncSession = Depends(get_db_session)) -> StoryboardAssistService:
        return StoryboardAssistService(session, client=fake)

    app.dependency_overrides[get_assist_service] = _factory


def _assert_archived_409(resp) -> None:
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == _ARCHIVED_DETAIL, resp.text


# ─── by-id mutators (no project_id in path → guarded at the repo chokepoint) ──
async def test_archived_blocks_every_by_id_mutator(client, db_session) -> None:
    tid, sid = UUID(client.tenant_id), UUID(client.subject_id)
    project = await _seed_project(db_session, tenant_id=tid, subject_id=sid)
    panel_id = await _panel(client, project.id)
    scene_id = await _scene(client, project.id)
    asset_id = await _asset(client, project.id)

    await _archive(db_session, project.id)

    # panel patch / approve / unapprove / delete
    _assert_archived_409(
        await client.patch(f"/api/storyboard/panels/{panel_id}", json={"vo_text": "nope"})
    )
    _assert_archived_409(
        await client.post(f"/api/storyboard/panels/{panel_id}/approve", json={"note": "n"})
    )
    _assert_archived_409(
        await client.post(f"/api/storyboard/panels/{panel_id}/unapprove", json={"note": "n"})
    )
    _assert_archived_409(
        await client.delete(f"/api/storyboard/panels/{panel_id}?expected_lock_version=0")
    )
    # scene patch / delete
    _assert_archived_409(
        await client.patch(f"/api/storyboard/scenes/{scene_id}", json={"name": "nope"})
    )
    _assert_archived_409(
        await client.delete(f"/api/storyboard/scenes/{scene_id}?expected_lock_version=0")
    )
    # asset patch / delete
    _assert_archived_409(
        await client.patch(f"/api/storyboard/assets/{asset_id}", json={"name": "nope"})
    )
    _assert_archived_409(
        await client.delete(f"/api/storyboard/assets/{asset_id}?expected_lock_version=0")
    )

    # …and nothing was actually mutated: the panel keeps its original vo_text.
    listing = await client.get(f"/api/projects/{project.id}/storyboard/panels")
    assert listing.status_code == 200, listing.text
    assert next(p for p in listing.json()["items"] if p["id"] == panel_id)["vo_text"] == "x"


# ─── project-scoped create + reorder ─────────────────────────────────────────
async def test_archived_blocks_creates_and_reorder(client, db_session) -> None:
    tid, sid = UUID(client.tenant_id), UUID(client.subject_id)
    project = await _seed_project(db_session, tenant_id=tid, subject_id=sid)
    a = await _panel(client, project.id, shot_key="clip-1")
    b = await _panel(client, project.id, shot_key="clip-2")

    await _archive(db_session, project.id)

    _assert_archived_409(
        await client.post(f"/api/projects/{project.id}/storyboard/panels", json={"vo_text": "new"})
    )
    _assert_archived_409(
        await client.post(f"/api/projects/{project.id}/storyboard/scenes", json={"name": "New"})
    )
    _assert_archived_409(
        await client.post(
            f"/api/projects/{project.id}/storyboard/assets",
            json={"kind": "character", "name": "New"},
        )
    )
    _assert_archived_409(
        await client.patch(
            f"/api/projects/{project.id}/storyboard/panels/reorder",
            json={"ordered_panel_ids": [b, a]},
        )
    )


# ─── snapshots: create / delete / restore ────────────────────────────────────
async def test_archived_blocks_snapshot_create_delete_restore(client, db_session) -> None:
    tid, sid = UUID(client.tenant_id), UUID(client.subject_id)
    project = await _seed_project(db_session, tenant_id=tid, subject_id=sid)
    await _panel(client, project.id)

    # A manual snapshot to later attempt to delete + restore, taken while still editable.
    snap = await client.post(
        f"/api/projects/{project.id}/storyboard/snapshots", json={"note": "v1"}
    )
    assert snap.status_code == 201, snap.text
    snapshot_id = snap.json()["id"]

    await _archive(db_session, project.id)

    _assert_archived_409(
        await client.post(f"/api/projects/{project.id}/storyboard/snapshots", json={"note": "v2"})
    )
    _assert_archived_409(
        await client.delete(f"/api/projects/{project.id}/storyboard/snapshots/{snapshot_id}")
    )
    _assert_archived_409(
        await client.post(
            f"/api/storyboard/snapshots/{snapshot_id}/restore",
            json={"expected_lock_versions": {}},
        )
    )


# ─── rewrite (LLM) — guarded at its terminal repo.update ─────────────────────
async def test_archived_blocks_rewrite(client, app, db_session) -> None:
    tid, sid = UUID(client.tenant_id), UUID(client.subject_id)
    project = await _seed_project(db_session, tenant_id=tid, subject_id=sid)
    panel_id = await _panel(client, project.id)
    await _archive(db_session, project.id)

    fake = AsyncMock()
    fake.message.return_value = WMResponse(
        output={"vo_text": "rewritten"},
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )
    _install_fake_assist(app, fake)
    try:
        resp = await client.post(
            f"/api/storyboard/panels/{panel_id}/rewrite",
            json={"instruction": "punch it up", "fields": ["vo_text"]},
        )
    finally:
        app.dependency_overrides.clear()
    _assert_archived_409(resp)


# ─── draft (LLM) — guarded at ensure_for_project, before the WM roundtrip ─────
async def test_archived_blocks_draft(client, app, db_session) -> None:
    """``draft`` (AI storyboard authoring) 409s on an archived project. The guard
    fires at ``ensure_for_project`` BEFORE any WM/LLM call — asserted by the fake
    client never being awaited (an archived project must not burn an LLM roundtrip)."""
    tid, sid = UUID(client.tenant_id), UUID(client.subject_id)
    project = await _seed_project(db_session, tenant_id=tid, subject_id=sid)
    await _archive(db_session, project.id)

    fake = AsyncMock()  # must NOT be reached — the guard precedes the LLM call
    _install_fake_assist(app, fake)
    try:
        resp = await client.post(
            f"/api/projects/{project.id}/storyboard/draft",
            json={"instructions": "make a video about cats"},
        )
    finally:
        app.dependency_overrides.clear()
    _assert_archived_409(resp)
    fake.message.assert_not_awaited()


# ─── branches: fork-from + merge-into an archived root ───────────────────────
async def test_fork_from_archived_root_returns_409(client, db_session) -> None:
    tid, sid = UUID(client.tenant_id), UUID(client.subject_id)
    project = await _seed_project(db_session, tenant_id=tid, subject_id=sid)
    await _panel(client, project.id)
    await _archive(db_session, project.id)

    _assert_archived_409(
        await client.post(
            f"/api/projects/{project.id}/storyboard/branches", json={"name": "alt-cut"}
        )
    )


async def test_merge_into_archived_root_returns_409(client, db_session) -> None:
    tid, sid = UUID(client.tenant_id), UUID(client.subject_id)
    project = await _seed_project(db_session, tenant_id=tid, subject_id=sid)
    await _panel(client, project.id, shot_key="clip-1", vo_text="one")
    await _panel(client, project.id, shot_key="clip-2", vo_text="two")

    # Fork while the root is still editable, edit the child, read the diff.
    fork = await client.post(
        f"/api/projects/{project.id}/storyboard/branches", json={"name": "alt-cut"}
    )
    assert fork.status_code == 201, fork.text
    branch = fork.json()

    child_panels = await client.get(f"/api/projects/{branch['child_project_id']}/storyboard/panels")
    clip1 = next(p for p in child_panels.json()["items"] if p["shot_key"] == "clip-1")["id"]
    edit = await client.patch(f"/api/storyboard/panels/{clip1}", json={"vo_text": "one-edited"})
    assert edit.status_code == 200, edit.text

    diff = await client.get(f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/diff")
    assert diff.status_code == 200, diff.text
    fingerprint = diff.json()["root_fingerprint"]

    # Now archive the ROOT and attempt to merge into it.
    await _archive(db_session, project.id)

    merged = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={"root_fingerprint": fingerprint, "resolutions": []},
    )
    _assert_archived_409(merged)


# ─── negative controls: the guard must NOT over-fire ─────────────────────────
async def test_producing_project_is_still_fully_mutable(client, db_session) -> None:
    """A non-archived (``producing``) project is unaffected — every mutator succeeds."""
    tid, sid = UUID(client.tenant_id), UUID(client.subject_id)
    project = await _seed_project(db_session, tenant_id=tid, subject_id=sid, status="producing")

    # create panel/scene/asset
    panel_id = await _panel(client, project.id, shot_key="clip-1")
    await _scene(client, project.id)
    asset_id = await _asset(client, project.id)

    # patch panel, approve, unapprove
    patched = await client.patch(f"/api/storyboard/panels/{panel_id}", json={"vo_text": "revised"})
    assert patched.status_code == 200, patched.text
    assert patched.json()["vo_text"] == "revised"
    assert (
        await client.post(f"/api/storyboard/panels/{panel_id}/approve", json={"note": "ok"})
    ).status_code == 200
    assert (
        await client.post(f"/api/storyboard/panels/{panel_id}/unapprove", json={"note": "ok"})
    ).status_code == 200

    # patch asset, reorder, snapshot create
    assert (
        await client.patch(f"/api/storyboard/assets/{asset_id}", json={"name": "Renamed"})
    ).status_code == 200
    second = await _panel(client, project.id, shot_key="clip-2")
    reordered = await client.patch(
        f"/api/projects/{project.id}/storyboard/panels/reorder",
        json={
            "ordered_panel_ids": [second, panel_id],
            # NOT hardcoded zeros: panel_id was patched + approved + unapproved above,
            # so its token has moved. Read the live map.
            "expected_lock_versions": await _board_map(client, project.id),
        },
    )
    assert reordered.status_code == 200, reordered.text
    snap = await client.post(
        f"/api/projects/{project.id}/storyboard/snapshots", json={"note": "v1"}
    )
    assert snap.status_code == 201, snap.text


async def test_reorder_missing_project_is_404_not_400(client, db_session) -> None:
    """Low-hanging fix: a missing/cross-tenant project on reorder is a 404 (the resource
    does not exist at this address), NOT the 400 that is reserved for a bad ORDERING."""
    resp = await client.patch(
        f"/api/projects/{uuid4()}/storyboard/panels/reorder",
        json={"ordered_panel_ids": [str(uuid4())], "expected_lock_versions": {}},
    )
    assert resp.status_code == 404, resp.text


async def test_reorder_bad_ordering_is_still_400(client, db_session) -> None:
    """The 404 change must not swallow the permutation 400: a real project with a
    malformed ordering still returns 400."""
    tid, sid = UUID(client.tenant_id), UUID(client.subject_id)
    project = await _seed_project(db_session, tenant_id=tid, subject_id=sid)
    await _panel(client, project.id)
    resp = await client.patch(
        f"/api/projects/{project.id}/storyboard/panels/reorder",
        json={
            "ordered_panel_ids": [str(uuid4())],  # unknown id → not a permutation
            "expected_lock_versions": await _board_map(client, project.id),
        },
    )
    assert resp.status_code == 400, resp.text
