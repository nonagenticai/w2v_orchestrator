"""HTTP surface for the storyboard branch three-way DIFF + MERGE.

* ``GET  /api/projects/{project_id}/storyboard/branches/{branch_id}/diff``  → BranchDiffResponse
* ``POST /api/projects/{project_id}/storyboard/branches/{branch_id}/merge`` → MergeResponse

plus the status codes that ARE the contract: a stale ``root_fingerprint`` is 409 (the root
moved while the human read the diff — merging anyway would clobber it), an unresolved
``conflict`` is 422 (the REQUEST is incomplete; picking a side is the decision the human
opened the diff to make), a ``merged``/``abandoned`` branch is 409 (both terminal), a branch
addressed through the WRONG project is 404, and every cross-tenant call is 404 — never a
403, which would be an existence oracle.

The approval trail is asserted end-to-end through the API, because that is the invariant
the merge design exists for (``storyboard_panel_events.panel_id`` is ON DELETE CASCADE, so
a merge that reinserted the panel would silently reap it).

Mirrors ``test_storyboard_branch_routes.py``.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy import text as sa_text
from sqlalchemy.exc import IntegrityError, OperationalError

from src.api.routes.storyboard import get_merge_service
from src.database.models import Project
from src.database.repositories.storyboard_repository import StoryboardRepository

pytestmark = pytest.mark.asyncio


async def _seed_project(db_session, *, tenant_id: UUID, subject_id: UUID) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="merge-routes",
        brief={"goal": "ship"},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def _seed_panel(client, project_id, *, shot_key: str, vo_text: str) -> str:
    r = await client.post(
        f"/api/projects/{project_id}/storyboard/panels",
        json={"shot_key": shot_key, "vo_text": vo_text},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def _open_branch(client, project_id, *, name: str = "alt-cut") -> dict:
    r = await client.post(f"/api/projects/{project_id}/storyboard/branches", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()


async def _panels(client, project_id) -> dict[str, dict]:
    r = await client.get(f"/api/projects/{project_id}/storyboard/panels")
    assert r.status_code == 200, r.text
    return {p["shot_key"]: p for p in r.json()["items"]}


async def _edit(client, panel_id: str, **fields) -> dict:
    r = await client.patch(f"/api/storyboard/panels/{panel_id}", json=fields)
    assert r.status_code == 200, r.text
    return r.json()


async def _diff(client, project_id, branch_id: str) -> dict:
    r = await client.get(f"/api/projects/{project_id}/storyboard/branches/{branch_id}/diff")
    assert r.status_code == 200, r.text
    return r.json()


async def _fork_with_edit(client, db_session) -> tuple[Project, dict, dict]:
    """Root with clip-1..3, forked, with clip-1 re-voiced IN THE BRANCH. The setup most of
    these tests want."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    for i, text in enumerate(("one", "two", "three")):
        await _seed_panel(client, project.id, shot_key=f"clip-{i + 1}", vo_text=text)
    branch = await _open_branch(client, project.id)
    child_panels = await _panels(client, branch["child_project_id"])
    await _edit(client, child_panels["clip-1"]["id"], vo_text="one-edited")
    return project, branch, child_panels


async def test_diff_returns_the_three_way_matrix_and_a_fingerprint(client, db_session) -> None:
    project, branch, child_panels = await _fork_with_edit(client, db_session)
    root_panels = await _panels(client, project.id)

    # A second branch-side change (added) and a root-side one on a DIFFERENT panel, so the
    # response has to distinguish "mergeable" from "merely moved on main".
    await _seed_panel(client, branch["child_project_id"], shot_key="clip-4", vo_text="new")
    await _edit(client, root_panels["clip-3"]["id"], vo_text="main only")

    body = await _diff(client, project.id, branch["id"])
    assert body["branch_id"] == branch["id"]
    assert body["root_project_id"] == str(project.id)
    assert body["child_project_id"] == branch["child_project_id"]
    assert body["status"] == "open"
    assert body["base_snapshot_id"] == branch["base_snapshot_id"]
    assert body["root_fingerprint"]
    assert body["conflict_count"] == 0

    kinds = {e["kind"]: e for e in body["entries"]}
    assert set(kinds) == {"modified", "added"}, "a main-only edit is not a mergeable entry"
    # Entries are keyed by LINEAGE — the ROOT's panel id — not by the child's row id.
    assert kinds["modified"]["key"] == root_panels["clip-1"]["id"]
    assert kinds["modified"]["key"] != child_panels["clip-1"]["id"]
    assert kinds["modified"]["changed_fields"] == ["vo_text"]
    assert kinds["modified"]["base"]["vo_text"] == "one"
    assert kinds["modified"]["main"]["vo_text"] == "one"
    assert kinds["modified"]["branch"]["vo_text"] == "one-edited"
    assert kinds["added"]["base"] is None and kinds["added"]["main"] is None


async def test_merge_applies_and_the_approval_trail_survives(client, db_session) -> None:
    """The whole point of the design, end to end over HTTP: approve a ROOT panel, merge a
    modification onto it, and the ``approved`` event must still be in its trail. A merge
    that delete-then-reinserted the panel would CASCADE the trail away while every content
    assertion still passed."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    for i, text in enumerate(("one", "two")):
        await _seed_panel(client, project.id, shot_key=f"clip-{i + 1}", vo_text=text)
    root_panels = await _panels(client, project.id)
    target = root_panels["clip-1"]["id"]

    approve = await client.post(
        f"/api/storyboard/panels/{target}/approve", json={"note": "ship it"}
    )
    assert approve.status_code == 200, approve.text

    branch = await _open_branch(client, project.id)
    child_panels = await _panels(client, branch["child_project_id"])
    await _edit(client, child_panels["clip-1"]["id"], vo_text="re-voiced")

    body = await _diff(client, project.id, branch["id"])
    merged = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={"root_fingerprint": body["root_fingerprint"], "resolutions": []},
    )
    assert merged.status_code == 200, merged.text
    payload = merged.json()
    assert payload["status"] == "merged"
    assert payload["merged_at"] is not None
    assert payload["applied"] == [target]
    assert payload["skipped"] == []
    assert payload["auto_snapshot_id"]  # the merge is reversible
    assert payload["root_fingerprint"] != body["root_fingerprint"]  # the root moved

    after = await _panels(client, project.id)
    assert after["clip-1"]["id"] == target, "same row — merge UPSERTs, it never reinserts"
    assert after["clip-1"]["vo_text"] == "re-voiced"
    assert after["clip-1"]["revision"] == 2

    trail = await client.get(f"/api/storyboard/panels/{target}/events")
    assert trail.status_code == 200
    assert [e["event_type"] for e in trail.json()["items"]] == ["approved", "merged"]

    # Approval is the ROOT's governance record: merge carried CONTENT and left it alone.
    assert after["clip-1"]["status"] == "approved"
    assert after["clip-1"]["approved_at"] is not None

    # The branch is closed, and the pre-merge auto snapshot is listed.
    listed = await client.get(f"/api/projects/{project.id}/storyboard/branches")
    assert [b["status"] for b in listed.json()["items"]] == ["merged"]
    snapshots = await client.get(f"/api/projects/{project.id}/storyboard/snapshots")
    kinds = {s["id"]: s["kind"] for s in snapshots.json()["items"]}
    assert kinds[payload["auto_snapshot_id"]] == "auto"


async def test_merge_archives_the_child_project(client, db_session) -> None:
    """A merged branch's child project is ARCHIVED (not destroyed) in the same transaction,
    and it never resurfaces on the projects page. ``abandon`` archives its child but refuses
    an already-merged branch, so this is the ONLY path that archives a merged branch's child.
    """
    project, branch, _child = await _fork_with_edit(client, db_session)
    child_id = UUID(branch["child_project_id"])

    body = await _diff(client, project.id, branch["id"])
    merged = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={"root_fingerprint": body["root_fingerprint"], "resolutions": []},
    )
    assert merged.status_code == 200, merged.text
    assert merged.json()["status"] == "merged"

    # Re-read the child through the DB (it is hidden from every API listing) — archived,
    # not destroyed: its panels remain the record of the merge.
    child = (
        await db_session.execute(
            select(Project).where(Project.id == child_id).execution_options(populate_existing=True)
        )
    ).scalar_one()
    assert child.status == "archived", "merge must archive the child project"
    assert child.archived_at is not None
    assert child.is_branch is True

    # …and it never leaks onto the projects page, with OR without include_archived.
    listed = await client.get("/api/projects?include_archived=true")
    assert str(child_id) not in [p["id"] for p in listed.json()["items"]]


async def test_merging_a_branch_with_an_open_sub_branch_is_409(client, db_session) -> None:
    """Bottom-up or not at all. Merging the parent branch now ARCHIVES its child project —
    the root any sub-branch forks from — so an open sub-branch would be stranded, merging
    into a dead end. Refuse it (409), exactly as ``abandon`` does, and leave the parent open
    and its child un-archived (a refused merge writes nothing).
    """
    project, branch, _child = await _fork_with_edit(client, db_session)
    child_id = branch["child_project_id"]

    # A sub-branch rooted at the parent branch's OWN child project.
    nested = await client.post(
        f"/api/projects/{child_id}/storyboard/branches", json={"name": "sub"}
    )
    assert nested.status_code == 201, nested.text

    body = await _diff(client, project.id, branch["id"])
    blocked = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={"root_fingerprint": body["root_fingerprint"], "resolutions": []},
    )
    assert blocked.status_code == 409, blocked.text
    assert "open child" in blocked.json()["detail"]

    # The refused merge wrote nothing: the parent is still open and its child is still live.
    listed = await client.get(f"/api/projects/{project.id}/storyboard/branches")
    statuses = {b["id"]: b["status"] for b in listed.json()["items"]}
    assert statuses[branch["id"]] == "open"
    child = (
        await db_session.execute(
            select(Project)
            .where(Project.id == UUID(child_id))
            .execution_options(populate_existing=True)
        )
    ).scalar_one()
    assert child.status != "archived", "a refused merge must not archive the child"


async def test_stale_fingerprint_is_409(client, db_session) -> None:
    project, branch, _child = await _fork_with_edit(client, db_session)
    body = await _diff(client, project.id, branch["id"])

    # Somebody else moves the root after the diff was read.
    root_panels = await _panels(client, project.id)
    await _edit(client, root_panels["clip-3"]["id"], vo_text="meanwhile")

    stale = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={"root_fingerprint": body["root_fingerprint"]},
    )
    assert stale.status_code == 409, stale.text
    detail = stale.json()["detail"]
    assert detail["expected_fingerprint"] == body["root_fingerprint"]
    assert detail["actual_fingerprint"] != body["root_fingerprint"]

    # The root is untouched — a refused merge writes NOTHING.
    unchanged = await _panels(client, project.id)
    assert unchanged["clip-1"]["vo_text"] == "one"
    still_open = await client.get(f"/api/projects/{project.id}/storyboard/branches")
    assert still_open.json()["items"][0]["status"] == "open"

    # Re-read the diff → fresh token → the merge goes through, and the concurrent edit
    # survives (it was never in the merge's path).
    fresh = await _diff(client, project.id, branch["id"])
    ok = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={"root_fingerprint": fresh["root_fingerprint"]},
    )
    assert ok.status_code == 200, ok.text
    final = await _panels(client, project.id)
    assert final["clip-1"]["vo_text"] == "one-edited"
    assert final["clip-3"]["vo_text"] == "meanwhile"


async def test_unresolved_conflict_is_422_then_resolvable(client, db_session) -> None:
    project, branch, _child = await _fork_with_edit(client, db_session)
    root_panels = await _panels(client, project.id)
    # Main edits the SAME panel the branch did → conflict.
    await _edit(client, root_panels["clip-1"]["id"], vo_text="main wins?")

    body = await _diff(client, project.id, branch["id"])
    assert body["conflict_count"] == 1
    entry = body["entries"][0]
    assert entry["kind"] == "conflict"
    assert entry["changed_fields"] == ["vo_text"] and entry["main_changed_fields"] == ["vo_text"]

    # 422, not 409: nothing about the server's state refuses — the REQUEST is incomplete.
    unresolved = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={"root_fingerprint": body["root_fingerprint"]},
    )
    assert unresolved.status_code == 422, unresolved.text
    assert unresolved.json()["detail"]["keys"] == [entry["key"]]

    # Resolving it to ``main`` keeps the root's version and closes the branch.
    resolved = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={
            "root_fingerprint": body["root_fingerprint"],
            "resolutions": [{"key": entry["key"], "resolution": "main"}],
        },
    )
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["skipped"] == [entry["key"]]
    assert resolved.json()["applied"] == []

    after = await _panels(client, project.id)
    assert after["clip-1"]["vo_text"] == "main wins?"
    # A skipped panel is not touched at all — no revision bump, no trail event.
    assert after["clip-1"]["revision"] == 1
    trail = await client.get(f"/api/storyboard/panels/{root_panels['clip-1']['id']}/events")
    assert trail.json()["items"] == []


async def test_merge_resolved_to_branch_wins_the_conflict(client, db_session) -> None:
    project, branch, _child = await _fork_with_edit(client, db_session)
    root_panels = await _panels(client, project.id)
    await _edit(client, root_panels["clip-1"]["id"], vo_text="main version")

    body = await _diff(client, project.id, branch["id"])
    key = body["entries"][0]["key"]
    won = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={
            "root_fingerprint": body["root_fingerprint"],
            "resolutions": [{"key": key, "resolution": "branch"}],
        },
    )
    assert won.status_code == 200, won.text
    assert won.json()["applied"] == [key]

    after = await _panels(client, project.id)
    assert after["clip-1"]["vo_text"] == "one-edited"
    assert after["clip-1"]["id"] == key  # …written onto the SAME root row


async def test_merge_resolved_to_keep_both_keeps_both_cuts(client, db_session) -> None:
    """``keep_both`` is a known resolution verb (not an enum-422), and it lands BOTH the
    root's version and the branch's version — the branch's on a fresh, non-colliding key."""
    project, branch, _child = await _fork_with_edit(client, db_session)
    root_panels = await _panels(client, project.id)
    await _edit(client, root_panels["clip-1"]["id"], vo_text="main version")

    body = await _diff(client, project.id, branch["id"])
    key = body["entries"][0]["key"]
    won = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={
            "root_fingerprint": body["root_fingerprint"],
            "resolutions": [{"key": key, "resolution": "keep_both"}],
        },
    )
    assert won.status_code == 200, won.text
    assert won.json()["applied"] == [key]

    after = await _panels(client, project.id)
    # Main keeps clip-1 verbatim (its own row); the branch's cut lands on clip-1-2.
    assert after["clip-1"]["vo_text"] == "main version"
    assert after["clip-1"]["id"] == key
    assert "clip-1-2" in after
    assert after["clip-1-2"]["vo_text"] == "one-edited"
    assert after["clip-1-2"]["id"] != key


async def test_keep_both_on_a_non_conflict_entry_is_422(client, db_session) -> None:
    """keep_both is conflict-only: aimed at a clean (branch-only) modification, it is a 422,
    not a silent no-op — there is no second version to keep."""
    project, branch, _child = await _fork_with_edit(client, db_session)
    # clip-1 was edited on the branch only (no root edit) → a 'modified' entry, not a conflict.
    body = await _diff(client, project.id, branch["id"])
    key = body["entries"][0]["key"]
    resp = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={
            "root_fingerprint": body["root_fingerprint"],
            "resolutions": [{"key": key, "resolution": "keep_both"}],
        },
    )
    assert resp.status_code == 422, resp.text
    assert key in resp.json()["detail"]["keys"]


async def test_merging_a_closed_branch_is_409(client, db_session) -> None:
    project, branch, _child = await _fork_with_edit(client, db_session)
    body = await _diff(client, project.id, branch["id"])
    first = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={"root_fingerprint": body["root_fingerprint"]},
    )
    assert first.status_code == 200, first.text

    # A closed branch is still DIFFABLE — that is how a merge is audited after the fact.
    audit = await _diff(client, project.id, branch["id"])
    assert audit["status"] == "merged"

    # …but re-merging replays stale content onto a root that has moved on: 409.
    again = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={"root_fingerprint": audit["root_fingerprint"]},
    )
    assert again.status_code == 409, again.text

    # An ABANDONED branch is terminal in the other direction.
    dead = await _open_branch(client, project.id, name="dead")
    assert (
        await client.delete(f"/api/projects/{project.id}/storyboard/branches/{dead['id']}")
    ).status_code == 204
    dead_diff = await _diff(client, project.id, dead["id"])
    refused = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{dead['id']}/merge",
        json={"root_fingerprint": dead_diff["root_fingerprint"]},
    )
    assert refused.status_code == 409, refused.text


async def test_branch_addressed_through_the_wrong_project_is_404(client, db_session) -> None:
    """The tenant scope is the SECURITY boundary; the path is the ADDRESSING one. Without
    the ``root_project_id`` check a branch of project A could be diffed/merged through
    project B's URL by anyone in the same tenant."""
    project, branch, _child = await _fork_with_edit(client, db_session)
    other = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )

    wrong_diff = await client.get(
        f"/api/projects/{other.id}/storyboard/branches/{branch['id']}/diff"
    )
    assert wrong_diff.status_code == 404

    wrong_merge = await client.post(
        f"/api/projects/{other.id}/storyboard/branches/{branch['id']}/merge",
        json={"root_fingerprint": "x"},
    )
    assert wrong_merge.status_code == 404

    unknown = await client.get(f"/api/projects/{project.id}/storyboard/branches/{uuid4()}/diff")
    assert unknown.status_code == 404


async def test_cross_tenant_diff_and_merge_are_404(
    client, other_tenant_client: AsyncClient, db_session
) -> None:
    project, branch, _child = await _fork_with_edit(client, db_session)
    body = await _diff(client, project.id, branch["id"])

    bleed_diff = await other_tenant_client.get(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/diff"
    )
    assert bleed_diff.status_code == 404, "404, never 403 — a 403 is an existence oracle"

    bleed_merge = await other_tenant_client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={"root_fingerprint": body["root_fingerprint"]},
    )
    assert bleed_merge.status_code == 404

    # The owner's branch is untouched and still open — the foreign merge changed nothing.
    mine = await _diff(client, project.id, branch["id"])
    assert mine["status"] == "open"
    assert mine["root_fingerprint"] == body["root_fingerprint"]
    untouched = await _panels(client, project.id)
    assert untouched["clip-1"]["vo_text"] == "one"


async def test_merge_requires_a_fingerprint_and_a_known_resolution(client, db_session) -> None:
    """The concurrency token is not optional — a merge with no fingerprint is not a merge,
    it is a race. And an unknown resolution verb is a 422, not a silent default."""
    project, branch, _child = await _fork_with_edit(client, db_session)
    body = await _diff(client, project.id, branch["id"])

    missing = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge", json={}
    )
    assert missing.status_code == 422, missing.text

    bogus = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={
            "root_fingerprint": body["root_fingerprint"],
            "resolutions": [{"key": body["entries"][0]["key"], "resolution": "whatever"}],
        },
    )
    assert bogus.status_code == 422, bogus.text


# ─── merge-vs-create_panel race + resolution-key hardening (PR: merge-race) ───
async def test_merge_duplicate_resolution_keys_is_422(client, db_session) -> None:
    """Two verdicts for the same panel key are ambiguous. Silently last-winning the dict
    comprehension would apply a resolution the caller never chose — reject it as 422."""
    project, branch, _child = await _fork_with_edit(client, db_session)
    root_panels = await _panels(client, project.id)
    await _edit(client, root_panels["clip-1"]["id"], vo_text="main version")

    body = await _diff(client, project.id, branch["id"])
    key = body["entries"][0]["key"]

    dup = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={
            "root_fingerprint": body["root_fingerprint"],
            "resolutions": [
                {"key": key, "resolution": "branch"},
                {"key": key, "resolution": "main"},
            ],
        },
    )
    assert dup.status_code == 422, dup.text
    assert dup.json()["detail"]["message"] == "duplicate resolution keys"
    assert key in dup.json()["detail"]["keys"]


async def test_merge_integrity_error_maps_to_409_not_500(app, client, db_session) -> None:
    """Belt-and-braces: any residual constraint collision during a merge surfaces as a
    retryable 409 (re-read the diff and retry), never an uncaught 500."""
    project, branch, _child = await _fork_with_edit(client, db_session)
    body = await _diff(client, project.id, branch["id"])

    class _RaisingMerge:
        async def merge(self, *args, **kwargs):
            raise IntegrityError(
                "INSERT INTO storyboard_panels ...", {}, Exception("duplicate key value")
            )

    app.dependency_overrides[get_merge_service] = lambda: _RaisingMerge()
    try:
        resp = await client.post(
            f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
            json={"root_fingerprint": body["root_fingerprint"], "resolutions": []},
        )
    finally:
        app.dependency_overrides.pop(get_merge_service, None)

    assert resp.status_code == 409, resp.text
    assert "raced" in resp.json()["detail"]


async def test_ensure_for_project_locks_root_row_serializing_merge(
    db_session, db_session_factory
) -> None:
    """The create chokepoint (``ensure_for_project``) takes FOR UPDATE on the projects row —
    the SAME row a storyboard MERGE row-locks via ``MergeService._lock_root``. So a
    concurrent create_panel and merge serialize on that row instead of independently minting
    a colliding ``clip-N`` that would 500 at COMMIT on the deferred shot_key uniqueness.

    Proven deterministically with ``FOR UPDATE NOWAIT``: while a second session holds the
    ensure_for_project lock, a probe of the same row fails immediately with a lock error;
    once that session's transaction ends, the probe succeeds."""
    tenant_id, subject_id = uuid4(), uuid4()
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="merge-race-lock",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    # Capture the id as a plain UUID: the aborted-probe rollback below expires the ORM
    # instance, so a later ``project.id`` read would trigger a lazy reload (async IO
    # outside the greenlet → MissingGreenlet).
    pid = project.id

    # Pre-create the storyboard row (committed) so the ensure_for_project UNDER TEST hits
    # ``on_conflict_do_nothing`` — no INSERT, hence no incidental FK KEY-SHARE lock on the
    # projects row. The lock the probe contends on then comes SOLELY from the explicit
    # FOR UPDATE this test validates (the steady-state race: storyboard row already exists).
    async with db_session_factory() as setup:
        await StoryboardRepository(setup).ensure_for_project(pid, tenant_id=tenant_id)
        await setup.commit()

    probe = sa_text("SELECT id FROM projects WHERE id = :pid FOR UPDATE NOWAIT")

    async with db_session_factory() as session_b:
        # Session B takes (and holds) the create-path lock on the projects row.
        sb = await StoryboardRepository(session_b).ensure_for_project(pid, tenant_id=tenant_id)
        assert sb is not None

        # Session A (a would-be merge) cannot lock the same row → deterministic failure.
        with pytest.raises(OperationalError):
            await db_session.execute(probe, {"pid": pid})
        await db_session.rollback()  # clear the aborted probe transaction

    # B's transaction has ended (session close → rollback) → the lock is released.
    got = await db_session.execute(probe, {"pid": pid})
    assert got.scalar_one() == pid
    await db_session.rollback()


# ─── per-field conflict resolution over HTTP ────────────────────────────────────


async def _two_field_conflict(client, db_session):
    """A conflict on clip-1 where BOTH lanes edited vo_text AND dialogue. Returns
    ``(project, branch, key)`` where ``key`` is the conflict's lineage key."""
    project, branch, child = await _fork_with_edit(client, db_session)
    # Branch already re-voiced clip-1; give it a second edited field.
    await _edit(client, child["clip-1"]["id"], vo_text="theirs", dialogue="theirs-d")
    root_panels = await _panels(client, project.id)
    await _edit(client, root_panels["clip-1"]["id"], vo_text="ours", dialogue="ours-d")
    body = await _diff(client, project.id, branch["id"])
    entry = next(e for e in body["entries"] if e["kind"] == "conflict")
    return project, branch, entry["key"], body["root_fingerprint"]


async def test_merge_field_resolution_splits_per_column_and_applies(client, db_session) -> None:
    project, branch, key, fp = await _two_field_conflict(client, db_session)

    # A field-ONLY resolution (no whole-row verdict) must APPLY, not silently skip — the route
    # must not fold a null ``resolution`` into the whole-row map (the hole-A guard).
    ok = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={
            "root_fingerprint": fp,
            "resolutions": [
                {"key": key, "field_resolutions": {"vo_text": "branch", "dialogue": "main"}}
            ],
        },
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["applied"] == [key]  # applied, NOT skipped

    after = await _panels(client, project.id)
    assert after["clip-1"]["vo_text"] == "theirs"  # branch pick
    assert after["clip-1"]["dialogue"] == "ours-d"  # main pick
    assert after["clip-1"]["id"] == key  # same root row


async def test_merge_field_resolution_partial_coverage_is_422(client, db_session) -> None:
    project, branch, key, fp = await _two_field_conflict(client, db_session)
    r = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={
            "root_fingerprint": fp,
            "resolutions": [
                {"key": key, "field_resolutions": {"vo_text": "branch"}}
            ],  # dialogue missing
        },
    )
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["missing_fields"] == {key: ["dialogue"]}


async def test_merge_resolution_and_field_resolutions_together_is_422(client, db_session) -> None:
    project, branch, key, fp = await _two_field_conflict(client, db_session)
    # XOR is a request-schema invariant → 422 at validation, before the service runs.
    r = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={
            "root_fingerprint": fp,
            "resolutions": [
                {"key": key, "resolution": "branch", "field_resolutions": {"vo_text": "branch"}}
            ],
        },
    )
    assert r.status_code == 422, r.text


async def test_merge_field_resolution_naming_a_fenced_field_is_422(client, db_session) -> None:
    project, branch, key, fp = await _two_field_conflict(client, db_session)
    # shot_key is never per-field mergeable → rejected at schema validation.
    r = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{branch['id']}/merge",
        json={
            "root_fingerprint": fp,
            "resolutions": [{"key": key, "field_resolutions": {"shot_key": "branch"}}],
        },
    )
    assert r.status_code == 422, r.text


# ─── sibling-to-sibling merge (WS3) ──────────────────────────────────────────
async def _two_siblings(client, db_session) -> tuple[Project, dict, dict]:
    """A root with clip-1..3 and TWO branches forked off it (A then B)."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    for i, t in enumerate(("one", "two", "three")):
        await _seed_panel(client, project.id, shot_key=f"clip-{i + 1}", vo_text=t)
    a = await _open_branch(client, project.id, name="alt-A")
    b = await _open_branch(client, project.id, name="alt-B")
    return project, a, b


async def test_sibling_diff_and_merge_round_trip(client, db_session) -> None:
    project, a, b = await _two_siblings(client, db_session)
    a_panels = await _panels(client, a["child_project_id"])
    await _edit(client, a_panels["clip-1"]["id"], vo_text="A-one")

    # Sibling diff is branch-centric on the SOURCE → A's one edit, no conflict.
    r = await client.get(
        f"/api/projects/{project.id}/storyboard/branches/{a['id']}/diff-against/{b['id']}"
    )
    assert r.status_code == 200, r.text
    diff = r.json()
    assert diff["conflict_count"] == 0
    assert [e["kind"] for e in diff["entries"]] == ["modified"]

    r = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{a['id']}/merge-into",
        json={
            "target_branch_id": b["id"],
            "target_fingerprint": diff["root_fingerprint"],
            "resolutions": [],
        },
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["source_branch_id"] == a["id"]
    assert payload["target_branch_id"] == b["id"]
    assert payload["auto_snapshot_id"]
    assert payload["target_fingerprint"]

    # A's edit landed on B, and B is still a live, editable branch project.
    b_panels = await _panels(client, b["child_project_id"])
    assert b_panels["clip-1"]["vo_text"] == "A-one"


async def test_merge_into_a_non_sibling_is_422(client, db_session) -> None:
    p1, a, _b = await _two_siblings(client, db_session)
    p2 = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _seed_panel(client, p2.id, shot_key="clip-1", vo_text="x")
    other = await _open_branch(client, p2.id, name="other-root")

    r = await client.post(
        f"/api/projects/{p1.id}/storyboard/branches/{a['id']}/merge-into",
        json={"target_branch_id": other["id"], "target_fingerprint": "x", "resolutions": []},
    )
    assert r.status_code == 422, r.text


async def test_sibling_merge_stale_target_fingerprint_is_409(client, db_session) -> None:
    project, a, b = await _two_siblings(client, db_session)
    a_panels = await _panels(client, a["child_project_id"])
    await _edit(client, a_panels["clip-1"]["id"], vo_text="A-one")

    r = await client.post(
        f"/api/projects/{project.id}/storyboard/branches/{a['id']}/merge-into",
        json={"target_branch_id": b["id"], "target_fingerprint": "stale", "resolutions": []},
    )
    assert r.status_code == 409, r.text


async def test_diff_advertises_the_server_mergeable_field_allowlist(client, db_session) -> None:
    """The diff response carries the server's ALLOWLIST so the frontend derives its fence
    from the server, not a hand-mirrored denylist. Both the branch->root diff and the
    sibling diff advertise the same allowlist."""
    from src.services.storyboard.merge_service import FIELD_MERGEABLE

    expected = sorted(FIELD_MERGEABLE)
    fenced = ("shot_key", "position", "scene", "characters", "props", "location")

    # branch->root diff
    project, branch, _ = await _fork_with_edit(client, db_session)
    body = await _diff(client, project.id, branch["id"])
    assert body["mergeable_fields"] == expected
    assert "vo_text" in body["mergeable_fields"]  # a plain content column IS mergeable
    for f in fenced:
        assert f not in body["mergeable_fields"]  # the fence: never per-field mergeable

    # sibling diff (same schema, separate route)
    project2, a, b = await _two_siblings(client, db_session)
    a_panels = await _panels(client, a["child_project_id"])
    await _edit(client, a_panels["clip-1"]["id"], vo_text="A-one")
    r = await client.get(
        f"/api/projects/{project2.id}/storyboard/branches/{a['id']}/diff-against/{b['id']}"
    )
    assert r.status_code == 200, r.text
    assert r.json()["mergeable_fields"] == expected
