"""HTTP surface for storyboard BRANCHES (fork + lifecycle, migration 0052).

Exercises the three routes mounted by ``src/api/routes/storyboard.py``:

* ``POST   /api/projects/{project_id}/storyboard/branches`` — fork (201)
* ``GET    /api/projects/{project_id}/storyboard/branches`` — bounded, newest-first
  page + a filter-consistent ``total`` (NOT ``len(items)``)
* ``DELETE /api/projects/{project_id}/storyboard/branches/{branch_id}`` — abandon (204)

plus the state-vs-existence status codes that carry the design's contract:
branching a BRANCH is 409 (the target exists; its STATE refuses), abandoning a
``merged`` branch is 409, a hand-DELETE of the ``branch_base`` snapshot is 409 (it
would CASCADE the branch away and strand the fork), and every cross-tenant call is a
404 — never a 403, which would be an existence oracle. The forked child project must
be invisible to ``GET /api/projects``.

Mirrors ``test_storyboard_snapshot_routes.py``.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from src.database.models import Project, StoryboardBranch
from src.database.repositories.storyboard_repository import StoryboardBranchRepository

pytestmark = pytest.mark.asyncio


async def _seed_project(db_session, *, tenant_id: UUID, subject_id: UUID) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="branch-routes",
        brief={"goal": "ship"},
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


async def _open_branch(client, project_id, *, name: str = "alt-cut") -> dict:
    r = await client.post(f"/api/projects/{project_id}/storyboard/branches", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()


async def _branch_row(db_session, branch_id: UUID) -> StoryboardBranch:
    return (
        await db_session.execute(
            select(StoryboardBranch)
            .where(StoryboardBranch.id == branch_id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


async def test_create_branch_201_and_child_is_hidden_from_project_list(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _seed_panel(client, project.id, vo_text="original")

    body = await _open_branch(client, project.id, name="alt-cut")
    assert body["name"] == "alt-cut"
    assert body["status"] == "open"
    assert body["root_project_id"] == str(project.id)
    assert body["child_project_id"] != str(project.id)
    assert body["base_snapshot_id"]  # the fork pinned its merge base
    assert body["merged_at"] is None

    child_id = body["child_project_id"]

    # THE projects-page leak the branch design calls out: the child is a real project
    # row, so only the ``is_branch`` filter keeps it out of the generic list.
    listed = await client.get("/api/projects")
    ids = [p["id"] for p in listed.json()["items"]]
    assert str(project.id) in ids
    assert child_id not in ids
    listed_all = await client.get("/api/projects?include_archived=true")
    assert child_id not in [p["id"] for p in listed_all.json()["items"]]

    # …nor in the root's CHILDREN list (that column is the course→lesson hierarchy).
    children = await client.get(f"/api/projects/{project.id}/children")
    assert children.status_code == 200
    assert child_id not in [c["id"] for c in children.json()]

    # The fork copied the graph: the child carries its own panel with the same vo_text.
    child_panels = await client.get(f"/api/projects/{child_id}/storyboard/panels")
    assert child_panels.status_code == 200
    items = child_panels.json()["items"]
    assert [p["vo_text"] for p in items] == ["original"]
    assert items[0]["project_id"] == child_id  # a COPY — not the root's row


async def test_branching_a_branch_is_allowed_and_the_parent_cannot_merge_until_it_closes(
    client, db_session
) -> None:
    """Forking a branch is a 201 now (it used to be a blanket 409).

    The refusal moved to the way back UP: merging the parent while the child is still open
    would mark the parent terminal and leave the child merging into a dead end — its work
    silently stranded. So the fork is allowed and the OUT-OF-ORDER MERGE is what 409s.
    """
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _seed_panel(client, project.id)
    parent = await _open_branch(client, project.id, name="first")
    child_id = parent["child_project_id"]

    # The fork OF A BRANCH — 201, and chained to its immediate parent.
    nested = await client.post(
        f"/api/projects/{child_id}/storyboard/branches", json={"name": "second"}
    )
    assert nested.status_code == 201, nested.text

    # The parent now refuses to merge: its child is still open.
    base = f"/api/projects/{project.id}/storyboard/branches/{parent['id']}"
    diff = await client.get(f"{base}/diff")
    assert diff.status_code == 200, diff.text
    blocked = await client.post(
        f"{base}/merge",
        json={"root_fingerprint": diff.json()["root_fingerprint"], "resolutions": []},
    )
    assert blocked.status_code == 409, blocked.text
    assert "open child" in blocked.json()["detail"]

    # …and so does abandon, for the same reason.
    assert (await client.delete(base)).status_code == 409

    # Forking the ROOT again still works too.
    again = await client.post(
        f"/api/projects/{project.id}/storyboard/branches", json={"name": "third"}
    )
    assert again.status_code == 201, again.text


async def test_list_branches_paginates_with_a_real_total(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _seed_panel(client, project.id)
    for n in range(3):
        await _open_branch(client, project.id, name=f"b{n}")

    page = await client.get(f"/api/projects/{project.id}/storyboard/branches?limit=2")
    assert page.status_code == 200
    body = page.json()
    assert len(body["items"]) == 2
    # ``total`` is the project-wide count under the same filter, NOT len(items) — a
    # len() "total" silently caps the UI's pager at one page.
    assert body["total"] == 3
    assert [i["name"] for i in body["items"]] == ["b2", "b1"]  # newest-first

    page2 = await client.get(f"/api/projects/{project.id}/storyboard/branches?limit=2&offset=2")
    assert [i["name"] for i in page2.json()["items"]] == ["b0"]
    assert page2.json()["total"] == 3

    # ``status`` filters BOTH the page and the total, or the pager describes a set it
    # is not paging.
    open_only = await client.get(f"/api/projects/{project.id}/storyboard/branches?status=open")
    assert open_only.json()["total"] == 3
    merged = await client.get(f"/api/projects/{project.id}/storyboard/branches?status=merged")
    assert merged.json()["items"] == [] and merged.json()["total"] == 0


async def test_abandon_branch_204_then_merged_is_409(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _seed_panel(client, project.id)
    body = await _open_branch(client, project.id, name="doomed")
    branch_id, child_id = UUID(body["id"]), UUID(body["child_project_id"])

    gone = await client.delete(f"/api/projects/{project.id}/storyboard/branches/{branch_id}")
    assert gone.status_code == 204, gone.text

    row = await _branch_row(db_session, branch_id)
    assert row.status == "abandoned"
    assert row.merged_at is None
    # DELETE is the verb, but this is an ABANDON: the child is archived, not destroyed.
    child = (
        await db_session.execute(
            select(Project).where(Project.id == child_id).execution_options(populate_existing=True)
        )
    ).scalar_one()
    assert child.status == "archived"
    assert child.archived_at is not None

    # Re-abandoning an abandoned branch is idempotent (204) — only ``merged`` conflicts.
    assert (
        await client.delete(f"/api/projects/{project.id}/storyboard/branches/{branch_id}")
    ).status_code == 204

    # A MERGED branch refuses: the root already holds its content, and abandoning would
    # archive the very project the merge's provenance points at.
    await StoryboardBranchRepository(db_session).mark_merged(row)
    await db_session.commit()
    conflict = await client.delete(f"/api/projects/{project.id}/storyboard/branches/{branch_id}")
    assert conflict.status_code == 409, conflict.text

    assert (
        await client.delete(f"/api/projects/{project.id}/storyboard/branches/{uuid4()}")
    ).status_code == 404


async def test_branch_base_snapshot_cannot_be_hand_deleted(client, db_session) -> None:
    """Hand-deleting the base would CASCADE the branch row away and silently strand the
    fork — so it is 409, exactly like an ``auto`` snapshot. A ``manual`` snapshot on the
    same project still deletes (204), which is what makes this a real gate and not a
    blanket refusal."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _seed_panel(client, project.id)
    base_id = (await _open_branch(client, project.id))["base_snapshot_id"]

    refused = await client.delete(f"/api/projects/{project.id}/storyboard/snapshots/{base_id}")
    assert refused.status_code == 409, refused.text

    # It is still listed as the branch's base, with the machine-managed kind.
    listing = await client.get(f"/api/projects/{project.id}/storyboard/snapshots")
    kinds = {s["id"]: s["kind"] for s in listing.json()["items"]}
    assert kinds[base_id] == "branch_base"

    # Control: a manual snapshot on the same project deletes fine.
    manual = await client.post(
        f"/api/projects/{project.id}/storyboard/snapshots", json={"note": "mine"}
    )
    assert manual.status_code == 201, manual.text
    ok = await client.delete(
        f"/api/projects/{project.id}/storyboard/snapshots/{manual.json()['id']}"
    )
    assert ok.status_code == 204


async def test_cross_tenant_branch_routes_are_404(
    client, other_tenant_client: AsyncClient, db_session
) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _seed_panel(client, project.id)
    body = await _open_branch(client, project.id, name="mine")
    branch_id = body["id"]

    # Fork somebody else's project → 404, not 403: a distinguishable 403 is an existence
    # oracle across tenants, which is itself a leak.
    theft = await other_tenant_client.post(
        f"/api/projects/{project.id}/storyboard/branches", json={"name": "theft"}
    )
    assert theft.status_code == 404

    bleed_list = await other_tenant_client.get(f"/api/projects/{project.id}/storyboard/branches")
    assert bleed_list.status_code == 404

    bleed_del = await other_tenant_client.delete(
        f"/api/projects/{project.id}/storyboard/branches/{branch_id}"
    )
    assert bleed_del.status_code == 404

    # The owner's branch is untouched and still open.
    mine = await client.get(f"/api/projects/{project.id}/storyboard/branches")
    assert [b["id"] for b in mine.json()["items"]] == [branch_id]
    assert mine.json()["items"][0]["status"] == "open"
