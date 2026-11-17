"""Branch CONTEXT on the project detail read (``GET /api/projects/{id}``).

PR #475 made a storyboard branch a real but HIDDEN project row: excluded from
``GET /api/projects`` (``is_branch`` filter) yet perfectly readable by id. That left a
dead end — a user could open a branch with nothing telling them they were in one and no
way back to the root. These tests pin the fix:

* a branch project's detail carries ``is_branch: true`` + a ``branch`` block naming the
  root (id + title) and the branch's own id / name / lifecycle status,
* a NORMAL project carries ``is_branch: false`` + ``branch: null`` (no false banner),
* the branch stays out of the LIST (the leak PR #475 closed must not reopen), and
* a branch of another tenant is a 404, never a context leak.

Mirrors ``test_storyboard_branch_routes.py`` (route-level, real postgres) and the
tenant-isolation shape of ``tests/database/test_storyboard_branches_rls.py``.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from src.database.models import Project

pytestmark = pytest.mark.asyncio


async def _seed_project(db_session, *, tenant_id: UUID, subject_id: UUID, title: str) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title=title,
        brief={"goal": "ship"},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def _seed_panel(client, project_id) -> None:
    r = await client.post(
        f"/api/projects/{project_id}/storyboard/panels", json={"vo_text": "original"}
    )
    assert r.status_code == 201, r.text


async def _open_branch(client, project_id, *, name: str = "alt-cut") -> dict:
    r = await client.post(f"/api/projects/{project_id}/storyboard/branches", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()


async def test_branch_project_detail_carries_branch_context(client, db_session) -> None:
    root = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
        title="Root Cut",
    )
    await _seed_panel(client, root.id)
    branch = await _open_branch(client, root.id, name="alt-cut")
    child_id = branch["child_project_id"]

    detail = await client.get(f"/api/projects/{child_id}")
    assert detail.status_code == 200, detail.text
    body = detail.json()

    # The flag the FE needs to know it is NOT looking at a normal project…
    assert body["is_branch"] is True
    # …and the block that makes the branch escapable: who am I, and where is home.
    ctx = body["branch"]
    assert ctx is not None
    assert ctx["branch_id"] == branch["id"]
    assert ctx["name"] == "alt-cut"
    assert ctx["status"] == "open"
    assert ctx["root_project_id"] == str(root.id)
    # Denormalised so the banner renders "branch of <root>" with no second round-trip.
    assert ctx["root_title"] == "Root Cut"
    # A direct branch's lineage is a single hop: the root itself, flagged as a normal
    # project (not a branch) so the FE knows the chain terminates here.
    assert ctx["ancestors"] == [
        {"project_id": str(root.id), "title": "Root Cut", "is_branch": False}
    ]


async def test_branch_of_branch_carries_the_full_ancestor_chain(client, db_session) -> None:
    """A grandchild branch (fork of a branch) reports its WHOLE lineage, root-ward.

    Branch-of-branch is supported (#483): the fork is 1-hop whatever it forks, so a chain
    R ← B ← C is a linked list up ``storyboard_branches.root_project_id`` (the immediate
    parent). The context walks it in one recursive CTE so the banner can say "branch of B,
    which is a branch of A" instead of dead-ending one hop up. Forking the branch here also
    pins the fix to the old blanket 409: ``_open_branch`` asserts 201.
    """
    root = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
        title="Root Cut",
    )
    await _seed_panel(client, root.id)
    mid = await _open_branch(client, root.id, name="mid")
    mid_pid = mid["child_project_id"]
    # Fork the BRANCH itself — this 201s now; it used to be a blanket 409.
    leaf = await _open_branch(client, mid_pid, name="leaf")
    leaf_pid = leaf["child_project_id"]

    detail = await client.get(f"/api/projects/{leaf_pid}")
    assert detail.status_code == 200, detail.text
    ctx = detail.json()["branch"]

    # Immediate parent (backward-compatible root_* fields) is the MID branch, not the root.
    assert ctx["root_project_id"] == mid_pid
    assert ctx["root_title"] == "mid"
    # Full lineage, immediate-parent first → ultimate root last. The intermediate hop is a
    # branch; the terminal is the true root (a normal project).
    assert ctx["ancestors"] == [
        {"project_id": mid_pid, "title": "mid", "is_branch": True},
        {"project_id": str(root.id), "title": "Root Cut", "is_branch": False},
    ]


async def test_branch_context_tracks_lifecycle_status(client, db_session) -> None:
    root = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
        title="Root Cut",
    )
    await _seed_panel(client, root.id)
    branch = await _open_branch(client, root.id, name="doomed")
    child_id = branch["child_project_id"]

    gone = await client.delete(f"/api/projects/{root.id}/storyboard/branches/{branch['id']}")
    assert gone.status_code == 204, gone.text

    # ABANDON archives the child rather than destroying it — so the project is still
    # readable, and its context must report the branch's real state, not a stale 'open'.
    detail = await client.get(f"/api/projects/{child_id}")
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["status"] == "archived"
    assert body["is_branch"] is True
    assert body["branch"]["status"] == "abandoned"


async def test_normal_project_detail_has_no_branch_context(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
        title="Plain",
    )
    detail = await client.get(f"/api/projects/{project.id}")
    assert detail.status_code == 200, detail.text
    body = detail.json()
    # No false banner on a project that is nobody's branch.
    assert body["is_branch"] is False
    assert body["branch"] is None

    # A ROOT project that HAS branches is still not itself a branch.
    await _seed_panel(client, project.id)
    await _open_branch(client, project.id, name="side-quest")
    root_detail = await client.get(f"/api/projects/{project.id}")
    assert root_detail.json()["is_branch"] is False
    assert root_detail.json()["branch"] is None


async def test_list_still_excludes_branches_and_flags_are_false(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
        title="Listed",
    )
    await _seed_panel(client, project.id)
    child_id = (await _open_branch(client, project.id))["child_project_id"]

    listed = await client.get("/api/projects")
    items = listed.json()["items"]
    ids = [p["id"] for p in items]
    assert str(project.id) in ids
    # Surfacing is_branch on ProjectResponse must NOT be mistaken for "the list now
    # carries branches" — the list still filters them out at the repository.
    assert child_id not in ids
    assert all(p["is_branch"] is False for p in items)


async def test_cross_tenant_branch_detail_is_404(
    client, other_tenant_client: AsyncClient, db_session
) -> None:
    root = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
        title="Secret Root",
    )
    await _seed_panel(client, root.id)
    child_id = (await _open_branch(client, root.id, name="secret-branch"))["child_project_id"]

    # 404, not 200-with-null-branch: the branch project itself belongs to another tenant,
    # so neither its existence nor its root's title may leak.
    bleed = await other_tenant_client.get(f"/api/projects/{child_id}")
    assert bleed.status_code == 404
    assert (await other_tenant_client.get(f"/api/projects/{root.id}")).status_code == 404

    # The owner still sees the full context.
    mine = await client.get(f"/api/projects/{child_id}")
    assert mine.json()["branch"]["root_title"] == "Secret Root"
