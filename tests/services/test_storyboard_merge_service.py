"""Storyboard branch MERGE — the three-way diff and its apply.

The fork (``test_storyboard_branch_service.py``) established lineage; this pins what
lineage is FOR. Against real postgres (``db_session``), the whole matrix:

* **the diff** — added / modified / deleted / conflict, keyed by ``origin_id`` lineage and
  NOT by panel id (ids are re-minted per project; an id-keyed diff is 100% noise);
* **APPROVAL-TRAIL SURVIVAL** — the invariant this PR exists to protect. Approve a ROOT
  panel, merge a modification onto it, and the ``approved`` event must still be there.
  ``storyboard_panel_events.panel_id`` is ON DELETE CASCADE, so a merge that
  delete-then-reinserted the panel would silently destroy the trail while every other
  assertion in this file still passed. That is the bug (PR #468) this test exists to catch;
* **the pre-merge auto snapshot** — a merge is reversible or it is not shippable;
* **the stale-fingerprint refusal** — a merge computed against a root that has since moved
  must not clobber what landed on main in between;
* **tenant isolation** and a **``lesson``-typed root** (whose ``parent_project_id`` rules
  merge must never route through).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from uuid import UUID, uuid4, uuid7

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError

from src.database.models import Project, StoryboardPanel, StoryboardSnapshot
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.storyboard_repository import (
    StoryboardAssetRepository,
    StoryboardBranchRepository,
    StoryboardPanelEventRepository,
    StoryboardPanelRepository,
    StoryboardRepository,
    StoryboardSceneRepository,
    StoryboardSnapshotRepository,
)
from src.services.storyboard.branch_service import (
    BranchNotFoundError,
    OpenChildBranchesError,
    StoryboardBranchService,
)
from src.services.storyboard.merge_service import (
    BranchNotOpenError,
    FieldResolutionTargetError,
    KeepBothTargetError,
    NotSiblingsError,
    SharedBaseMissingError,
    StaleRootError,
    StoryboardMergeService,
    UnresolvedConflictError,
)

pytestmark = pytest.mark.asyncio


def _side(view: dict[str, Any] | None) -> dict[str, Any]:
    """Assert a diff side is PRESENT, then hand back its view.

    ``base`` / ``main`` / ``branch`` are ``None`` on the side where the panel does not
    exist (that absence IS ``added`` / ``deleted``). Every subscript below is on a side the
    test has already established exists — this states that, instead of typing it away.
    """
    assert view is not None
    return view


async def _seed_root(
    db_session,
    *,
    tenant_id: UUID,
    subject_id: UUID,
    type_: str = "generic",
    parent: UUID | None = None,
) -> Project:
    project = Project(
        id=uuid7(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type=type_,
        title="merge-root",
        brief={"goal": "ship"},
        brand={},
        voice={},
        theme={},
        parent_project_id=parent,
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def _seed_graph(
    db_session, *, tenant_id: UUID, subject_id: UUID, **kw: Any
) -> dict[str, Any]:
    """Root + 1 scene + 3 panels (clip-1..3) + a character. The board every test forks."""
    project = await _seed_root(db_session, tenant_id=tenant_id, subject_id=subject_id, **kw)
    storyboard = await StoryboardRepository(db_session).ensure_for_project(
        project.id, tenant_id=tenant_id
    )
    assert storyboard is not None

    scene = await StoryboardSceneRepository(db_session).create(
        project_id=project.id, storyboard_id=storyboard.id, position=0, name="scene-1"
    )
    hero = await StoryboardAssetRepository(db_session).create(
        project_id=project.id, position=0, kind="character", name="Hero"
    )
    panels = StoryboardPanelRepository(db_session)
    created = []
    for i, vo in enumerate(("one", "two", "three")):
        created.append(
            await panels.create(
                project_id=project.id,
                storyboard_id=storyboard.id,
                position=i,
                shot_key=f"clip-{i + 1}",
                scene_id=scene.id,
                vo_text=vo,
                characters=[hero.id] if i == 0 else None,
            )
        )
    await db_session.commit()
    return {
        "project": project,
        "project_id": project.id,
        "storyboard_id": storyboard.id,
        "scene_id": scene.id,
        "hero_id": hero.id,
        "panels": [p.id for p in created],
    }


async def _fork(db_session, root_id: UUID, *, tenant_id: UUID, subject_id: UUID, name: str = "alt"):
    branch, child = await StoryboardBranchService(db_session).create_branch(
        root_id, tenant_id=tenant_id, name=name, actor_subject=subject_id
    )
    ids = (branch.id, child.id)
    await db_session.commit()
    return ids


async def _panels_by_shot(db_session, project_id: UUID, *, tenant_id: UUID) -> dict[str, Any]:
    rows = await StoryboardPanelRepository(db_session).list_for_project(
        project_id, tenant_id=tenant_id
    )
    return {r.shot_key: r for r in rows}


async def _root_panel(db_session, panel_id: UUID) -> StoryboardPanel | None:
    return (
        await db_session.execute(
            select(StoryboardPanel)
            .where(StoryboardPanel.id == panel_id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()


async def _events(db_session, panel_id: UUID, *, tenant_id: UUID) -> list[str]:
    trail = await StoryboardPanelEventRepository(db_session).list_for_panel(
        panel_id, tenant_id=tenant_id
    )
    return [e.event_type for e in (trail or [])]


# ─── 1. The diff matrix ──────────────────────────────────────────────────────
async def test_diff_of_an_untouched_branch_is_empty(db_session) -> None:
    """A fresh fork is byte-identical to its base by construction (the fork seeds from the
    base snapshot's payload, not from a second live read). Anything else here would mean
    the diff is comparing raw ids somewhere instead of lineage."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, _child = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )

    diff = await StoryboardMergeService(db_session).diff(branch_id, tenant_id=tenant_id)
    assert diff.entries == []
    assert diff.conflict_count == 0
    assert diff.status == "open"
    assert diff.root_fingerprint  # a stable content hash, not an empty string


async def test_diff_reports_modified_keyed_by_lineage_not_panel_id(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )

    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    edited = child_panels["clip-1"]
    await StoryboardPanelRepository(db_session).update(
        edited.id, tenant_id=tenant_id, fields={"vo_text": "one-edited"}
    )
    await db_session.commit()

    diff = await StoryboardMergeService(db_session).diff(branch_id, tenant_id=tenant_id)
    assert len(diff.entries) == 1
    entry = diff.entries[0]
    assert entry.kind == "modified"
    # THE contract: the key is the ROOT's panel id (the lineage), not the child's row id.
    assert entry.key == root["panels"][0]
    assert entry.key != edited.id
    assert entry.changed_fields == ["vo_text"]
    assert entry.main_changed_fields == []
    assert _side(entry.base)["vo_text"] == "one"
    assert _side(entry.main)["vo_text"] == "one"
    assert _side(entry.branch)["vo_text"] == "one-edited"
    # The scene + cast are compared as LINEAGE keys, so a forked panel that changed nothing
    # about them reads as unchanged rather than as an edit on every reference.
    assert _side(entry.base)["scene"] == _side(entry.branch)["scene"]
    assert (
        _side(entry.base)["characters"]
        == _side(entry.branch)["characters"]
        == [str(root["hero_id"])]
    )


async def test_diff_reports_added_and_deleted(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )

    panels = StoryboardPanelRepository(db_session)
    child_storyboard = await StoryboardRepository(db_session).ensure_for_project(
        child_id, tenant_id=tenant_id
    )
    assert child_storyboard is not None
    fresh = await panels.create(
        project_id=child_id,
        storyboard_id=child_storyboard.id,
        position=3,
        shot_key="clip-4",
        vo_text="brand new",
    )
    by_shot = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await panels.delete(
        by_shot["clip-3"].id,
        tenant_id=tenant_id,
        expected_lock_version=by_shot["clip-3"].lock_version,
    )
    await db_session.commit()

    diff = await StoryboardMergeService(db_session).diff(branch_id, tenant_id=tenant_id)
    kinds = {e.kind: e for e in diff.entries}
    assert set(kinds) == {"added", "deleted"}
    # A panel born in the branch has no lineage, so it keys to its OWN id — which by
    # construction exists on no other side. That is what makes it an ``added``.
    assert kinds["added"].key == fresh.id
    assert kinds["added"].base is None and kinds["added"].main is None
    assert _side(kinds["added"].branch)["vo_text"] == "brand new"
    # A deleted panel keys to the ROOT id it was forked from.
    assert kinds["deleted"].key == root["panels"][2]
    assert kinds["deleted"].branch is None
    assert _side(kinds["deleted"].base)["vo_text"] == "three"


async def test_diff_reports_conflict_when_both_sides_touched_the_same_panel(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)

    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await panels.update(
        child_panels["clip-2"].id, tenant_id=tenant_id, fields={"vo_text": "branch says two"}
    )
    # …and main moved on the SAME panel, independently.
    await panels.update(root["panels"][1], tenant_id=tenant_id, fields={"vo_text": "main says two"})
    # A panel only MAIN touched is not an entry at all — there is nothing to merge from it.
    await panels.update(root["panels"][2], tenant_id=tenant_id, fields={"vo_text": "main only"})
    await db_session.commit()

    diff = await StoryboardMergeService(db_session).diff(branch_id, tenant_id=tenant_id)
    assert len(diff.entries) == 1, "a main-only edit must not appear as a mergeable entry"
    entry = diff.entries[0]
    assert entry.kind == "conflict"
    assert diff.conflict_count == 1
    assert entry.key == root["panels"][1]
    assert entry.changed_fields == ["vo_text"]
    assert entry.main_changed_fields == ["vo_text"]
    assert _side(entry.base)["vo_text"] == "two"
    assert _side(entry.main)["vo_text"] == "main says two"
    assert _side(entry.branch)["vo_text"] == "branch says two"


async def test_identical_edit_on_both_sides_is_not_a_conflict(db_session) -> None:
    """Both sides changed the panel, to the SAME thing. There is nothing to decide, so it
    is a plain ``modified`` — a conflict here would be a prompt with one possible answer."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await panels.update(child_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "same"})
    await panels.update(root["panels"][0], tenant_id=tenant_id, fields={"vo_text": "same"})
    await db_session.commit()

    diff = await StoryboardMergeService(db_session).diff(branch_id, tenant_id=tenant_id)
    assert [e.kind for e in diff.entries] == ["modified"]
    assert diff.conflict_count == 0


# ─── 2. THE approval-trail survival invariant ────────────────────────────────
async def test_merge_upserts_and_the_root_approval_trail_survives(db_session) -> None:
    """THE invariant of this PR (the PR #468 bug).

    ``storyboard_panel_events.panel_id`` is ON DELETE CASCADE. A merge that
    delete-then-reinserted the root panel — even with identical content, even with the same
    id, even inside one transaction — would silently reap its whole approval trail while
    every other assertion about the merged CONTENT still passed. Approve a root panel, merge
    a modification onto it, and demand the panel row and its ``approved`` event are both
    still there.
    """
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    panels = StoryboardPanelRepository(db_session)
    events = StoryboardPanelEventRepository(db_session)

    target = root["panels"][0]
    approved = await panels.approve(
        target, tenant_id=tenant_id, actor_subject_id=subject_id, note="ship it", events=events
    )
    assert approved is not None and approved.status == "approved"
    await db_session.commit()

    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await panels.update(
        child_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "re-voiced"}
    )
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    result = await svc.merge(
        branch_id,
        tenant_id=tenant_id,
        root_fingerprint=diff.root_fingerprint,
        actor_subject=subject_id,
    )
    await db_session.commit()

    # The panel row is the SAME row — same id — carrying the branch's content.
    merged = await _root_panel(db_session, target)
    assert merged is not None, "the root panel was destroyed; merge must UPSERT, never reinsert"
    assert merged.id == target
    assert merged.vo_text == "re-voiced"
    assert merged.revision == 2  # the content-edit counter moved

    # …and the trail is intact. THIS is the assertion the whole design is for.
    trail = await _events(db_session, target, tenant_id=tenant_id)
    assert "approved" in trail, "the approval trail was CASCADE-reaped — merge reinserted"
    assert trail == ["approved", "merged"]  # append-only, in order

    # Approval is the ROOT's governance record: the merge carried CONTENT, and left the
    # approval exactly where it was granted. It neither forged one nor revoked one.
    assert merged.status == "approved"
    assert merged.approved_by == str(subject_id)
    assert merged.approved_at is not None

    assert result["applied"] == [target]
    assert result["status"] == "merged"


# ─── 2c. The merged branch's child project is ARCHIVED ───────────────────────
async def test_merge_archives_the_child_project(db_session) -> None:
    """A merged branch's child project must be archived in the SAME transaction.

    ``merge`` lands the branch's content on the ROOT, so the child project is now a stale,
    forkable duplicate of already-merged work. ``abandon`` archives its child, but
    ``abandon`` refuses an already-merged branch — so without this the child would be stuck
    live+editable forever. The design (``05_STORYBOARD_BRANCH_MERGE_DESIGN.md``) requires it
    archived, NOT destroyed: its panels remain the record of what was merged.
    """
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await StoryboardPanelRepository(db_session).update(
        child_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "merged edit"}
    )
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    await svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=diff.root_fingerprint)
    await db_session.commit()

    child = (
        await db_session.execute(
            select(Project).where(Project.id == child_id).execution_options(populate_existing=True)
        )
    ).scalar_one()
    assert child.status == "archived", "merge must archive the child project"
    assert child.archived_at is not None
    # Archived, NOT destroyed — the child's panels remain as the record of the merge.
    assert child.is_branch is True
    surviving = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    assert set(surviving) == {"clip-1", "clip-2", "clip-3"}


async def test_merged_child_is_absent_from_the_projects_list(db_session) -> None:
    """Regression: the archived merged child never surfaces in the generic project list,
    with or without ``include_archived``. It is ``is_branch=true`` (always hidden) AND
    archived — the projects page must not leak it either way."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await StoryboardPanelRepository(db_session).update(
        child_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "merged edit"}
    )
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    await svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=diff.root_fingerprint)
    await db_session.commit()

    projects = ProjectRepository(db_session)
    visible = await projects.list(tenant_id=tenant_id, limit=50, offset=0)
    archived_too = await projects.list(
        tenant_id=tenant_id, include_archived=True, limit=50, offset=0
    )
    assert child_id not in {p.id for p in visible.items}
    assert child_id not in {p.id for p in archived_too.items}


async def test_merge_mints_a_pre_merge_auto_snapshot(db_session) -> None:
    """A merge is reversible or it is not shippable: restore ``auto_snapshot_id`` and the
    root is back. Same ``kind='auto'`` safety copy restore mints, for the same reason."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await StoryboardPanelRepository(db_session).update(
        child_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "after"}
    )
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    result = await svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=diff.root_fingerprint)
    await db_session.commit()

    snapshots = StoryboardSnapshotRepository(db_session)
    auto = await snapshots.get(result["auto_snapshot_id"], tenant_id=tenant_id)
    assert auto is not None
    assert auto.kind == "auto"
    assert auto.project_id == root["project_id"]
    # It captured the PRE-merge board — that is what makes the merge undoable.
    captured = {p["id"]: p for p in auto.payload["panels"]}
    assert captured[str(root["panels"][0])]["vo_text"] == "one"


async def test_merge_caps_the_roots_auto_snapshots(db_session) -> None:
    """The cap is a property of MINTING an auto, not of restoring.

    ``restore`` capped its autos from day one; ``merge`` minted one per merge and never
    pruned — so the configured cap was a no-op on the busiest auto producer we have. A root
    merged N times grew by N whole graph blobs, forever. Two merges under a cap of 1 must
    leave exactly ONE auto, and it must be the LAST merge's undo handle (protecting the id
    it just minted is what stops a cap of 1 from evicting the very snapshot that makes the
    merge reversible).

    The ``branch_base`` snapshots are asserted to SURVIVE. They are not ``kind='auto'``, and
    ``storyboard_branches.base_snapshot_id`` is ON DELETE CASCADE — a prune that reached one
    would silently take the branch row (its status / merged_at / lineage) with it.
    """
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    root_id = root["project_id"]

    # Two branches, each editing a DIFFERENT panel, so both merges apply cleanly and the
    # test is measuring retention rather than conflict handling.
    first_id, first_child = await _fork(
        db_session, root_id, tenant_id=tenant_id, subject_id=subject_id, name="alt-1"
    )
    second_id, second_child = await _fork(
        db_session, root_id, tenant_id=tenant_id, subject_id=subject_id, name="alt-2"
    )
    for child_id, shot in ((first_child, "clip-1"), (second_child, "clip-2")):
        child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
        await StoryboardPanelRepository(db_session).update(
            child_panels[shot].id, tenant_id=tenant_id, fields={"vo_text": f"{shot} edited"}
        )
    await db_session.commit()

    svc = StoryboardMergeService(db_session, max_auto_per_project=1)
    minted: list[UUID] = []
    for branch_id in (first_id, second_id):
        # Re-diff before each merge: the first merge moves the root, so the second branch's
        # fingerprint must be taken against the root as it stands NOW or it 409s.
        diff = await svc.diff(branch_id, tenant_id=tenant_id)
        result = await svc.merge(
            branch_id, tenant_id=tenant_id, root_fingerprint=diff.root_fingerprint
        )
        minted.append(result["auto_snapshot_id"])
        await db_session.commit()

    rows = (
        (
            await db_session.execute(
                select(StoryboardSnapshot).where(StoryboardSnapshot.project_id == root_id)
            )
        )
        .scalars()
        .all()
    )
    autos = [r for r in rows if r.kind == "auto"]
    assert len(autos) == 1, "merge minted an auto per merge and never pruned them"
    assert autos[0].id == minted[-1], "the prune evicted the newest auto — the merge's undo"
    assert sum(1 for r in rows if r.kind == "branch_base") == 2, "the prune reached branch_base"


# ─── 2b. Chains: branch-of-branch merges bottom-up ───────────────────────────
async def test_merging_a_branch_with_open_children_is_refused(db_session) -> None:
    """The guard that makes branch-of-branch safe, and the one this feature turns on.

    ``mark_merged`` is TERMINAL. Merge ``B`` into ``R`` while ``C`` is still open and ``C``
    is left able to merge only into ``B`` — a branch that can never carry it upward again.
    The API would report a successful merge and ``C``'s work would never reach the root:
    silently stranded. Refuse the out-of-order merge; close the children first.
    """
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    parent_id, parent_child = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id, name="B"
    )
    await _fork(db_session, parent_child, tenant_id=tenant_id, subject_id=subject_id, name="C")

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(parent_id, tenant_id=tenant_id)
    with pytest.raises(OpenChildBranchesError) as caught:
        await svc.merge(parent_id, tenant_id=tenant_id, root_fingerprint=diff.root_fingerprint)
    assert caught.value.open_children == 1
    await db_session.rollback()

    # It refused BEFORE writing: no auto snapshot, and the branch is still open.
    branch = await StoryboardBranchRepository(db_session).get(parent_id, tenant_id=tenant_id)
    assert branch is not None
    assert branch.status == "open"
    autos = [
        s
        for s in (
            await StoryboardSnapshotRepository(db_session).list_for_project(
                root["project_id"], tenant_id=tenant_id, limit=50
            )
        )
        if s.kind == "auto"
    ]
    assert autos == []


async def test_a_chain_merges_bottom_up_and_the_edit_reaches_the_root(db_session) -> None:
    """Two hops, end to end: an edit made in ``C`` reaches ``R`` via ``B``.

    This is what the chain is FOR, and it is the proof that lineage survives a re-fork.
    ``C``'s panel carries ``R``'s origin id (not ``B``'s re-minted one), so the SAME key
    matches on both hops — ``C``→``B`` upserts onto ``B``'s panel, then ``B``→``R`` upserts
    onto ``R``'s. Had the fork re-rooted lineage at each hop, the second merge would have
    reported the panel as ADDED and landed a duplicate on the root instead.
    """
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    root_id = root["project_id"]

    parent_id, parent_child = await _fork(
        db_session, root_id, tenant_id=tenant_id, subject_id=subject_id, name="B"
    )
    grand_id, grand_child = await _fork(
        db_session, parent_child, tenant_id=tenant_id, subject_id=subject_id, name="C"
    )

    # The edit is made ONLY in the grandchild.
    grand_panels = await _panels_by_shot(db_session, grand_child, tenant_id=tenant_id)
    await StoryboardPanelRepository(db_session).update(
        grand_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "from the grandchild"}
    )
    await db_session.commit()

    svc = StoryboardMergeService(db_session)

    # Hop 1: C → B.
    diff_c = await svc.diff(grand_id, tenant_id=tenant_id)
    await svc.merge(grand_id, tenant_id=tenant_id, root_fingerprint=diff_c.root_fingerprint)
    await db_session.commit()
    middle = await _panels_by_shot(db_session, parent_child, tenant_id=tenant_id)
    assert middle["clip-1"].vo_text == "from the grandchild"
    # The root has NOT seen it yet — the chain is not a broadcast.
    assert (await _panels_by_shot(db_session, root_id, tenant_id=tenant_id))[
        "clip-1"
    ].vo_text == "one"

    # Hop 2: B → R. Only legal now that C is closed.
    diff_b = await svc.diff(parent_id, tenant_id=tenant_id)
    await svc.merge(parent_id, tenant_id=tenant_id, root_fingerprint=diff_b.root_fingerprint)
    await db_session.commit()

    root_panels = await _panels_by_shot(db_session, root_id, tenant_id=tenant_id)
    assert root_panels["clip-1"].vo_text == "from the grandchild"
    # Upserted onto the ROOT's existing panel by lineage — not landed as a duplicate.
    assert root_panels["clip-1"].id == root["panels"][0]
    assert len(root_panels) == 3


# ─── 3. Apply: added / deleted / conflict resolutions ────────────────────────
async def test_merge_added_panel_lands_on_the_root_with_lineage(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    child_sb = await StoryboardRepository(db_session).ensure_for_project(
        child_id, tenant_id=tenant_id
    )
    assert child_sb is not None
    fresh = await StoryboardPanelRepository(db_session).create(
        project_id=child_id,
        storyboard_id=child_sb.id,
        position=3,
        shot_key="clip-4",
        vo_text="the new one",
    )
    fresh_id = fresh.id
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    await svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=diff.root_fingerprint)
    await db_session.commit()

    root_panels = await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id)
    assert set(root_panels) == {"clip-1", "clip-2", "clip-3", "clip-4"}
    landed = root_panels["clip-4"]
    assert landed.vo_text == "the new one"
    assert landed.project_id == root["project_id"]
    assert landed.id != fresh_id  # a COPY — ids are per-project
    # Lineage is stamped so a SECOND merge recognises it instead of duplicating it.
    assert landed.origin_id == fresh_id
    # A branch author's approval does not travel — the root has no trail event for it.
    assert landed.status == "draft"


async def test_merge_added_panel_renames_on_a_shot_key_collision(db_session) -> None:
    """Two projects can legally hold conflicting ``clip-4``s — the unique constraint is
    per-project, which is the whole reason a branch is its own project. On merge they land
    in ONE project, so the incoming panel is renamed. Refusing would strand the branch;
    overwriting would destroy a root panel that appears in nobody's diff."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)

    child_sb = await StoryboardRepository(db_session).ensure_for_project(
        child_id, tenant_id=tenant_id
    )
    assert child_sb is not None
    await panels.create(
        project_id=child_id,
        storyboard_id=child_sb.id,
        position=3,
        shot_key="clip-4",
        vo_text="branch clip-4",
    )
    # Meanwhile main grew its OWN clip-4.
    await panels.create(
        project_id=root["project_id"],
        storyboard_id=root["storyboard_id"],
        position=3,
        shot_key="clip-4",
        vo_text="main clip-4",
    )
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    await svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=diff.root_fingerprint)
    await db_session.commit()  # the deferred unique check settles here, or this raises

    root_panels = await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id)
    assert root_panels["clip-4"].vo_text == "main clip-4", "the root's panel kept its key"
    assert root_panels["clip-4-2"].vo_text == "branch clip-4"


async def test_merge_deleted_panel_removes_it_from_the_root(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await StoryboardPanelRepository(db_session).delete(
        child_panels["clip-3"].id,
        tenant_id=tenant_id,
        expected_lock_version=child_panels["clip-3"].lock_version,
    )
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    await svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=diff.root_fingerprint)
    await db_session.commit()

    # The ONE sanctioned delete: the author removed the panel and asked for that to land.
    assert await _root_panel(db_session, root["panels"][2]) is None
    remaining = await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id)
    assert set(remaining) == {"clip-1", "clip-2"}


async def test_unresolved_conflict_refuses_and_writes_nothing(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await panels.update(
        child_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "theirs"}
    )
    await panels.update(root["panels"][0], tenant_id=tenant_id, fields={"vo_text": "ours"})
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    with pytest.raises(UnresolvedConflictError) as exc:
        await svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=diff.root_fingerprint)
    assert exc.value.keys == [root["panels"][0]]
    await db_session.rollback()

    # Nothing was written on the way to the raise — including no auto snapshot.
    panel = await _root_panel(db_session, root["panels"][0])
    assert panel is not None and panel.vo_text == "ours"
    branch = await StoryboardBranchRepository(db_session).get(branch_id, tenant_id=tenant_id)
    assert branch is not None and branch.status == "open"


async def test_conflict_resolved_to_main_keeps_the_root_version(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    # A conflict on clip-1 …
    await panels.update(
        child_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "theirs"}
    )
    await panels.update(root["panels"][0], tenant_id=tenant_id, fields={"vo_text": "ours"})
    # … and a clean modification on clip-2, which must still land.
    await panels.update(child_panels["clip-2"].id, tenant_id=tenant_id, fields={"vo_text": "clean"})
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    result = await svc.merge(
        branch_id,
        tenant_id=tenant_id,
        root_fingerprint=diff.root_fingerprint,
        resolutions={root["panels"][0]: "main"},
    )
    await db_session.commit()

    assert result["skipped"] == [root["panels"][0]]
    assert result["applied"] == [root["panels"][1]]
    kept = await _root_panel(db_session, root["panels"][0])
    assert kept is not None and kept.vo_text == "ours"  # main won
    assert kept.revision == 1  # …and was not even touched
    landed = await _root_panel(db_session, root["panels"][1])
    assert landed is not None and landed.vo_text == "clean"
    # Only the applied panel gets a trail event.
    assert await _events(db_session, root["panels"][0], tenant_id=tenant_id) == []
    assert await _events(db_session, root["panels"][1], tenant_id=tenant_id) == ["merged"]


async def _seed_conflict_on_clip1(db_session, *, tenant_id, subject_id):
    """A both-lanes edit on clip-1 (main='ours', branch='theirs') → one conflict entry."""
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await panels.update(
        child_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "theirs"}
    )
    await panels.update(root["panels"][0], tenant_id=tenant_id, fields={"vo_text": "ours"})
    await db_session.commit()
    return root, branch_id, child_id


async def test_conflict_resolved_to_keep_both_mints_a_noncolliding_shot_key(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root, branch_id, _child = await _seed_conflict_on_clip1(
        db_session, tenant_id=tenant_id, subject_id=subject_id
    )

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    result = await svc.merge(
        branch_id,
        tenant_id=tenant_id,
        root_fingerprint=diff.root_fingerprint,
        resolutions={root["panels"][0]: "keep_both"},
    )
    await db_session.commit()

    assert result["applied"] == [root["panels"][0]]
    by_shot = await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id)
    # BOTH cuts survive: main keeps clip-1 (its own row, verbatim), the branch's version
    # lands on a fresh, non-colliding shot_key as a NEW panel.
    assert by_shot["clip-1"].vo_text == "ours"
    assert by_shot["clip-1"].id == root["panels"][0]
    assert "clip-1-2" in by_shot
    assert by_shot["clip-1-2"].vo_text == "theirs"
    assert by_shot["clip-1-2"].id != root["panels"][0]


async def test_keep_both_does_not_touch_mains_row_or_its_trail(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root, branch_id, _child = await _seed_conflict_on_clip1(
        db_session, tenant_id=tenant_id, subject_id=subject_id
    )

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    await svc.merge(
        branch_id,
        tenant_id=tenant_id,
        root_fingerprint=diff.root_fingerprint,
        resolutions={root["panels"][0]: "keep_both"},
    )
    await db_session.commit()

    kept = await _root_panel(db_session, root["panels"][0])
    assert kept is not None and kept.vo_text == "ours"
    assert kept.revision == 1  # never rewritten (an upsert would have bumped it)
    # Main's kept row gets NO merged event (it was not touched); the minted panel does.
    assert await _events(db_session, root["panels"][0], tenant_id=tenant_id) == []
    by_shot = await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id)
    assert await _events(db_session, by_shot["clip-1-2"].id, tenant_id=tenant_id) == ["merged"]


async def test_keep_both_new_row_has_independent_origin_id(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root, branch_id, _child = await _seed_conflict_on_clip1(
        db_session, tenant_id=tenant_id, subject_id=subject_id
    )

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    await svc.merge(
        branch_id,
        tenant_id=tenant_id,
        root_fingerprint=diff.root_fingerprint,
        resolutions={root["panels"][0]: "keep_both"},
    )
    await db_session.commit()

    by_shot = await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id)
    new_row = await _root_panel(db_session, by_shot["clip-1-2"].id)
    # The minted row must NOT share main's lineage key (root["panels"][0]) or a later
    # diff/merge would collapse the two into one. A fresh native root self-lineages.
    assert new_row is not None and new_row.origin_id is None
    kept = await _root_panel(db_session, root["panels"][0])
    assert kept is not None and kept.origin_id != new_row.id


async def test_keep_both_on_branch_deleted_conflict_collapses_to_main(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    # Branch DELETED clip-3, main EDITED it → a conflict whose branch side is None.
    await panels.delete(
        child_panels["clip-3"].id,
        tenant_id=tenant_id,
        expected_lock_version=child_panels["clip-3"].lock_version,
    )
    await panels.update(
        root["panels"][2], tenant_id=tenant_id, fields={"vo_text": "main kept editing"}
    )
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    result = await svc.merge(
        branch_id,
        tenant_id=tenant_id,
        root_fingerprint=diff.root_fingerprint,
        resolutions={root["panels"][2]: "keep_both"},
    )
    await db_session.commit()

    # No branch row to bring across → keep_both degenerates to "keep main": mint nothing.
    assert result["applied"] == []
    assert result["skipped"] == [root["panels"][2]]
    kept = await _root_panel(db_session, root["panels"][2])
    assert kept is not None and kept.vo_text == "main kept editing"
    by_shot = await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id)
    assert set(by_shot) == {"clip-1", "clip-2", "clip-3"}  # no clip-3-2 minted


async def test_keep_both_on_a_non_conflict_entry_is_422(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    # A clean branch-only edit on clip-2 — a 'modified' entry, NOT a conflict.
    await panels.update(child_panels["clip-2"].id, tenant_id=tenant_id, fields={"vo_text": "solo"})
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    with pytest.raises(KeepBothTargetError) as exc:
        await svc.merge(
            branch_id,
            tenant_id=tenant_id,
            root_fingerprint=diff.root_fingerprint,
            resolutions={root["panels"][1]: "keep_both"},
        )
    assert root["panels"][1] in exc.value.keys
    await db_session.rollback()
    # Nothing written on the way to the raise.
    landed = await _root_panel(db_session, root["panels"][1])
    assert landed is not None and landed.vo_text == "two"


async def test_branch_deleted_main_edited_conflict_resolved_to_branch_really_deletes(
    db_session,
) -> None:
    """The branch DELETED the panel; main EDITED it. That is a ``conflict`` whose branch
    side is ``None`` — and resolving it to "branch" means "yes, delete it".

    The apply used to route on the KIND string (``deleted`` → delete, anything else →
    write). This entry is spelled ``conflict``, so it went to the writer, which found no
    branch row to write and skipped it — while ``merge`` had already listed its key in
    ``applied``. The API answered 200 "merged", and the panel the caller asked to remove
    was still sitting on the root. Silent data loss, in the direction nobody checks.
    """
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)

    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await panels.delete(
        child_panels["clip-3"].id,
        tenant_id=tenant_id,
        expected_lock_version=child_panels["clip-3"].lock_version,
    )
    # …and main kept working on the very panel the branch removed.
    await panels.update(
        root["panels"][2], tenant_id=tenant_id, fields={"vo_text": "main kept editing"}
    )
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    assert [e.kind for e in diff.entries] == ["conflict"]
    entry = diff.entries[0]
    assert entry.key == root["panels"][2]
    assert entry.branch is None, "the branch side does not exist — that absence IS the delete"
    assert _side(entry.main)["vo_text"] == "main kept editing"

    result = await svc.merge(
        branch_id,
        tenant_id=tenant_id,
        root_fingerprint=diff.root_fingerprint,
        resolutions={root["panels"][2]: "branch"},
    )
    await db_session.commit()

    assert result["applied"] == [root["panels"][2]]
    assert result["skipped"] == []
    # The whole point: ``applied`` and REALITY agree.
    assert await _root_panel(db_session, root["panels"][2]) is None
    remaining = await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id)
    assert set(remaining) == {"clip-1", "clip-2"}


async def test_branch_deleted_main_edited_conflict_resolved_to_main_keeps_the_panel(
    db_session,
) -> None:
    """The mirror decision on the same conflict: keep main's panel. It must SURVIVE with
    main's content, untouched and un-trailed — a merge that deletes a panel the caller
    explicitly voted to keep is the same data loss with the sign flipped."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)

    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await panels.delete(
        child_panels["clip-3"].id,
        tenant_id=tenant_id,
        expected_lock_version=child_panels["clip-3"].lock_version,
    )
    await panels.update(root["panels"][2], tenant_id=tenant_id, fields={"vo_text": "main wins"})
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    result = await svc.merge(
        branch_id,
        tenant_id=tenant_id,
        root_fingerprint=diff.root_fingerprint,
        resolutions={root["panels"][2]: "main"},
    )
    await db_session.commit()

    assert result["skipped"] == [root["panels"][2]]
    assert result["applied"] == []
    kept = await _root_panel(db_session, root["panels"][2])
    assert kept is not None and kept.vo_text == "main wins"
    assert kept.revision == 1  # not touched at all
    assert await _events(db_session, root["panels"][2], tenant_id=tenant_id) == []
    remaining = await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id)
    assert set(remaining) == {"clip-1", "clip-2", "clip-3"}


# ─── 4. Concurrency: the fingerprint gate ────────────────────────────────────
async def test_stale_fingerprint_refuses_the_merge(db_session) -> None:
    """The root moved while the human was reading the diff. Merging anyway would clobber
    whatever landed on main — so it is a refusal, not a last-write-wins race."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await panels.update(child_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "mine"})
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    stale = (await svc.diff(branch_id, tenant_id=tenant_id)).root_fingerprint

    # Somebody else edits the root — a panel this merge does not even touch.
    await panels.update(root["panels"][2], tenant_id=tenant_id, fields={"vo_text": "meanwhile"})
    await db_session.commit()

    with pytest.raises(StaleRootError) as exc:
        await svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=stale)
    assert exc.value.expected == stale
    assert exc.value.actual != stale
    await db_session.rollback()

    # Re-reading the diff yields a fresh token, and the merge then goes through.
    fresh = await svc.diff(branch_id, tenant_id=tenant_id)
    await svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=fresh.root_fingerprint)
    await db_session.commit()

    merged = await _root_panel(db_session, root["panels"][0])
    assert merged is not None and merged.vo_text == "mine"
    # The concurrent edit survived — it was never in the merge's path.
    other = await _root_panel(db_session, root["panels"][2])
    assert other is not None and other.vo_text == "meanwhile"


async def test_fingerprint_is_content_addressed_not_a_clock(db_session) -> None:
    """Two reads of an unchanged board hash identically; the response's post-merge
    fingerprint chains straight into a second merge with no re-read."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    svc = StoryboardMergeService(db_session)
    b1, c1 = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id, name="one"
    )
    b2, c2 = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id, name="two"
    )

    first = (await svc.diff(b1, tenant_id=tenant_id)).root_fingerprint
    second = (await svc.diff(b2, tenant_id=tenant_id)).root_fingerprint
    assert first == second, "an unchanged board must hash the same — this is not a timestamp"

    panels = StoryboardPanelRepository(db_session)
    c1_panels = await _panels_by_shot(db_session, c1, tenant_id=tenant_id)
    await panels.update(c1_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "from-b1"})
    c2_panels = await _panels_by_shot(db_session, c2, tenant_id=tenant_id)
    await panels.update(c2_panels["clip-2"].id, tenant_id=tenant_id, fields={"vo_text": "from-b2"})
    await db_session.commit()

    result = await svc.merge(b1, tenant_id=tenant_id, root_fingerprint=first)
    await db_session.commit()

    # The second branch's cached token is now stale…
    with pytest.raises(StaleRootError):
        await svc.merge(b2, tenant_id=tenant_id, root_fingerprint=second)
    await db_session.rollback()
    # …but the first merge's RESPONSE fingerprint describes the root as it now is.
    await svc.merge(b2, tenant_id=tenant_id, root_fingerprint=result["root_fingerprint"])
    await db_session.commit()

    final = await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id)
    assert final["clip-1"].vo_text == "from-b1"
    assert final["clip-2"].vo_text == "from-b2"


async def test_two_concurrent_merges_of_one_root_serialise_and_the_loser_409s(
    db_session, db_session_factory
) -> None:
    """The fingerprint gate is a compare-and-set whose compare and set are two statements
    under READ COMMITTED — and the orchestrator runs ``replicas:2`` + HPA. Two REAL
    postgres sessions, two branches of one root, both holding the same (valid) token:

    * without the root row lock both pass the check, and the loser then upserts clip-1 from
      an in-memory copy of a root row the winner has already rewritten. Last write wins, the
      winner's content is gone, and the 409 never fires — a lost update.
    * with it, the loser BLOCKS on ``SELECT … FOR UPDATE`` before it reads a single board,
      wakes up after the winner's COMMIT, hashes the board it actually finds, and 409s.

    The blocking assertion is the load-bearing one: it can only hold if the lock is real and
    is taken BEFORE the read it protects.
    """
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    b1, c1 = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id, name="one"
    )
    b2, c2 = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id, name="two"
    )

    # Both branches edit the SAME panel — that is what makes the second write a CLOBBER
    # rather than an independent one.
    panels = StoryboardPanelRepository(db_session)
    c1_panels = await _panels_by_shot(db_session, c1, tenant_id=tenant_id)
    await panels.update(c1_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "from-b1"})
    c2_panels = await _panels_by_shot(db_session, c2, tenant_id=tenant_id)
    await panels.update(c2_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "from-b2"})
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    token = (await svc.diff(b1, tenant_id=tenant_id)).root_fingerprint

    # Replica A merges b1 and does NOT commit yet — its transaction (and the root lock) is
    # exactly the window replica B races into.
    await svc.merge(b1, tenant_id=tenant_id, root_fingerprint=token)

    async with db_session_factory() as other:
        loser = asyncio.create_task(
            StoryboardMergeService(other).merge(b2, tenant_id=tenant_id, root_fingerprint=token)
        )
        try:
            await asyncio.sleep(0.5)
            assert not loser.done(), (
                "the second merge read the root while the first held it — no row lock, so both "
                "will pass the fingerprint gate and the later write silently wins"
            )

            await db_session.commit()  # A lands; the lock releases here and B wakes up

            with pytest.raises(StaleRootError):
                await asyncio.wait_for(loser, timeout=10)
        finally:
            # B is a REAL second connection blocked on a REAL row lock. If anything above
            # fails, the ``async with`` would close ``other`` out from under a still-pending
            # task and strand its transaction — an `idle in transaction` backend holding the
            # lock, which then blocks the next run's TRUNCATE and hangs the WHOLE suite (not
            # just this test). Drain the task before the session can go.
            loser.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await loser
            await other.rollback()

    # A's content is intact: B refused rather than overwriting it.
    merged = await _root_panel(db_session, root["panels"][0])
    assert merged is not None and merged.vo_text == "from-b1"
    assert merged.revision == 2  # bumped exactly once — B wrote nothing


# ─── 5. Lifecycle ────────────────────────────────────────────────────────────
async def test_merging_a_closed_branch_raises(db_session) -> None:
    """``merged`` and ``abandoned`` are both terminal. Re-merging replays stale content over
    a root that has moved on; merging an abandoned branch resurrects discarded work."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await StoryboardPanelRepository(db_session).update(
        child_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "once"}
    )
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    await svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=diff.root_fingerprint)
    await db_session.commit()

    again = await svc.diff(branch_id, tenant_id=tenant_id)
    assert again.status == "merged"  # a CLOSED branch is still diffable — that is the audit
    with pytest.raises(BranchNotOpenError) as exc:
        await svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=again.root_fingerprint)
    assert exc.value.status == "merged"
    await db_session.rollback()

    abandoned_id, _ = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id, name="dead"
    )
    await StoryboardBranchService(db_session).abandon(abandoned_id, tenant_id=tenant_id)
    await db_session.commit()
    fresh = await svc.diff(abandoned_id, tenant_id=tenant_id)
    with pytest.raises(BranchNotOpenError):
        await svc.merge(abandoned_id, tenant_id=tenant_id, root_fingerprint=fresh.root_fingerprint)


# ─── 6. Tenant isolation (RLS) ───────────────────────────────────────────────
async def test_cross_tenant_diff_and_merge_raise_branch_not_found(db_session) -> None:
    """Missing and cross-tenant collapse to the SAME error: a distinguishable one would be
    an existence oracle across tenants, which is itself the leak."""
    tenant_a, tenant_b = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_a, subject_id=uuid7())
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_a, subject_id=uuid7()
    )
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_a)
    await StoryboardPanelRepository(db_session).update(
        child_panels["clip-1"].id, tenant_id=tenant_a, fields={"vo_text": "private"}
    )
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    with pytest.raises(BranchNotFoundError):
        await svc.diff(branch_id, tenant_id=tenant_b)
    with pytest.raises(BranchNotFoundError):
        await svc.merge(branch_id, tenant_id=tenant_b, root_fingerprint="whatever")
    with pytest.raises(BranchNotFoundError):
        await svc.diff(uuid4(), tenant_id=tenant_a)

    # The owner's branch is untouched and still mergeable.
    mine = await svc.diff(branch_id, tenant_id=tenant_a)
    assert mine.status == "open"
    assert [e.kind for e in mine.entries] == ["modified"]


# ─── 7. A lesson-typed root merges (it must not route through create_with_deliverables) ──
async def test_merge_of_a_lesson_rooted_branch(db_session) -> None:
    """``ProjectRepository.create_with_deliverables`` raises for ``type='lesson'`` with a
    NULL ``parent_project_id`` — and a branch's child project is EXACTLY that (its
    ``parent_project_id`` stays NULL because that column is the course→lesson hierarchy,
    not the branch link). Merge creates no project at all, so it can never route through
    that factory; this test is the guard that keeps it that way."""
    tenant_id, subject_id = uuid7(), uuid7()
    course = await _seed_root(
        db_session, tenant_id=tenant_id, subject_id=subject_id, type_="course"
    )
    root = await _seed_graph(
        db_session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        type_="lesson",
        parent=course.id,
    )
    assert root["project"].type == "lesson"

    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    # The forked child is a lesson with a NULL parent — the shape the factory rejects.
    child = (await db_session.execute(select(Project).where(Project.id == child_id))).scalar_one()
    assert child.type == "lesson" and child.parent_project_id is None and child.is_branch

    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await StoryboardPanelRepository(db_session).update(
        child_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "lesson edit"}
    )
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    result = await svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=diff.root_fingerprint)
    await db_session.commit()

    assert result["status"] == "merged"
    merged = await _root_panel(db_session, root["panels"][0])
    assert merged is not None and merged.vo_text == "lesson edit"
    # The lesson root kept its place in the course tree — merge touched no project row.
    reloaded = (
        await db_session.execute(
            select(Project)
            .where(Project.id == root["project_id"])
            .execution_options(populate_existing=True)
        )
    ).scalar_one()
    assert reloaded.parent_project_id == course.id
    assert reloaded.type == "lesson"


# ─── 8. Assets a branch panel gained come along ──────────────────────────────
async def test_merge_materialises_a_new_branch_asset_the_panel_references(db_session) -> None:
    """A panel that gained a cast member in the branch has NO root-side asset to point at.
    Dropping the reference (or merging it dangling) would silently lose the edit, so the
    counterpart is materialised on the root WITH lineage."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )

    villain = await StoryboardAssetRepository(db_session).create(
        project_id=child_id, position=1, kind="character", name="Villain"
    )
    villain_id = villain.id
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    child_hero = child_panels["clip-1"].characters[0]
    await StoryboardPanelRepository(db_session).update(
        child_panels["clip-1"].id,
        tenant_id=tenant_id,
        fields={"characters": [child_hero, str(villain_id)]},
    )
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    assert diff.entries[0].changed_fields == ["characters"]
    await svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=diff.root_fingerprint)
    await db_session.commit()

    root_assets = await StoryboardAssetRepository(db_session).list_for_project(
        root["project_id"], tenant_id=tenant_id
    )
    by_name = {a.name: a for a in root_assets}
    assert set(by_name) == {"Hero", "Villain"}
    assert by_name["Villain"].id != villain_id  # a copy, in the root's namespace
    assert by_name["Villain"].origin_id == villain_id  # …with lineage, so a re-merge finds it

    merged = await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id)
    cast = [UUID(c) for c in merged["clip-1"].characters]
    assert cast == [by_name["Hero"].id, by_name["Villain"].id]  # ORDER is content, and it held


# ─── 6. merge-vs-PATCH lost-update (the board row lock) ───────────────────────
#
# The ``projects``-row lock (``_lock_root``) only serialises merge-vs-merge; a
# ``PATCH /panels/{id}`` never touches the ``projects`` row, so it slips past that lock,
# commits between merge's UNLOCKED read and merge's ``ON CONFLICT DO UPDATE SET <all cols>``
# upsert, and is silently reverted — a lost update the fingerprint gate never sees (it
# compares merge's OWN stale snapshot). ``_lock_root_board`` takes ``FOR UPDATE`` on the
# root's panel/scene/asset rows so the PATCH's UPDATE blocks until merge commits, then
# re-applies on top.


def _install_apply_seam(
    svc: StoryboardMergeService, monkeypatch: pytest.MonkeyPatch
) -> tuple[asyncio.Event, asyncio.Event]:
    """Wrap ``svc._apply`` in a deterministic seam so a test can interleave a concurrent
    PATCH exactly while merge holds the board locks.

    ``_lock_root_board`` acquires the ``FOR UPDATE`` locks BEFORE ``_apply`` runs, so by the
    time the seam fires ``locks_held`` the locks are already held; ``release`` gates the real
    apply so the PATCH launched in between provably blocks on those locks. ``monkeypatch``
    (not ``svc._apply = …``) rebinds the instance method without a lint/type suppression and
    auto-restores after the test."""
    real_apply = svc._apply
    locks_held = asyncio.Event()
    release = asyncio.Event()

    async def _seam(**kwargs: Any) -> None:
        locks_held.set()
        await release.wait()
        await real_apply(**kwargs)

    monkeypatch.setattr(svc, "_apply", _seam)
    return locks_held, release


async def _patch_position(session: Any, panel_id: UUID, *, tenant_id: UUID, position: int) -> Any:
    """The concurrent writer: the exact repo path ``PATCH /panels/{id}`` takes for a
    position edit. Returns the panel, or ``None`` (→ 404) if the row was deleted mid-flush."""
    return await StoryboardPanelRepository(session).update(
        panel_id, tenant_id=tenant_id, fields={"position": position}
    )


async def test_lock_root_board_blocks_a_concurrent_panel_update(
    db_session, db_session_factory
) -> None:
    """The lock primitive, isolated: while session A holds ``_lock_root_board``'s
    ``FOR UPDATE`` on the root board, a second REAL connection's ``UPDATE`` of a root panel
    must BLOCK (surfaced as a 55P03 lock timeout), then succeed once A commits. This is the
    load-bearing behaviour merge-vs-PATCH safety is built on — if the UPDATE did not block,
    nothing downstream would serialise."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    root_id = root["project_id"]
    target = root["panels"][0]

    # Session A takes the board lock and HOLDS it — no commit.
    await StoryboardMergeService(db_session)._lock_root_board(root_id)

    async with db_session_factory() as other:
        try:
            # Session B: bound the wait so a real conflict surfaces as an error, not a hang.
            await other.execute(text("SET LOCAL lock_timeout='500ms'"))
            with pytest.raises(OperationalError) as exc_info:
                await other.execute(
                    text("UPDATE storyboard_panels SET position = 99 WHERE id = :pid"),
                    {"pid": str(target)},
                )
            assert getattr(exc_info.value.orig, "sqlstate", None) == "55P03"  # lock_not_available
            await other.rollback()

            # A commits → the board locks release → B's UPDATE now goes straight through.
            await db_session.commit()
            await other.execute(text("SET LOCAL lock_timeout='500ms'"))
            result = await other.execute(
                text("UPDATE storyboard_panels SET position = 99 WHERE id = :pid"),
                {"pid": str(target)},
            )
            assert result.rowcount == 1
            await other.commit()
        finally:
            await other.rollback()


async def test_merge_does_not_revert_a_concurrent_panel_patch(
    db_session, db_session_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end lost-update: a root panel is APPROVED at position 3; the branch edits only
    its CONTENT. A ``PATCH position=7`` lands WHILE the merge holds the board locks. Under the
    fix the PATCH blocks, merge commits its content edit, then the PATCH re-applies on top —
    so BOTH survive: ``vo_text`` is the branch's, ``position`` is the PATCH's 7. On pre-fix
    code merge's stale snapshot upserts ``position=3`` back over the PATCH → this fails."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    root_id = root["project_id"]
    target = root["panels"][0]

    panels = StoryboardPanelRepository(db_session)
    events = StoryboardPanelEventRepository(db_session)
    # Root panel: approved + at position 3 (the governed state merge must not clobber).
    await panels.update(target, tenant_id=tenant_id, fields={"position": 3})
    await panels.approve(target, tenant_id=tenant_id, actor_subject_id=subject_id, events=events)
    await db_session.commit()

    branch_id, child_id = await _fork(
        db_session, root_id, tenant_id=tenant_id, subject_id=subject_id
    )
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await panels.update(
        child_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "branch-edit"}
    )
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    token = (await svc.diff(branch_id, tenant_id=tenant_id)).root_fingerprint
    locks_held, release = _install_apply_seam(svc, monkeypatch)

    merge_task = asyncio.create_task(
        svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=token)
    )
    async with db_session_factory() as patch_sess:
        patch_task: asyncio.Task[Any] | None = None
        try:
            await asyncio.wait_for(locks_held.wait(), timeout=10)
            # PATCH races in while merge holds the board locks — its flush must BLOCK.
            patch_task = asyncio.create_task(
                _patch_position(patch_sess, target, tenant_id=tenant_id, position=7)
            )
            await asyncio.sleep(0.5)
            assert not patch_task.done(), "the PATCH did not block on the board lock"

            release.set()  # let merge's real _apply run
            await asyncio.wait_for(merge_task, timeout=10)
            await db_session.commit()  # merge lands, the board locks release

            patched = await asyncio.wait_for(patch_task, timeout=10)
            assert patched is not None and patched.position == 7
            await patch_sess.commit()
        finally:
            if patch_task is not None:
                patch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await patch_task
            merge_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await merge_task
            await patch_sess.rollback()

    # A fresh session proves it on disk: content merged, PATCH's position survived.
    async with db_session_factory() as verify:
        final = (
            await verify.execute(
                select(StoryboardPanel)
                .where(StoryboardPanel.id == target)
                .execution_options(populate_existing=True)
            )
        ).scalar_one_or_none()
        assert final is not None
        assert final.vo_text == "branch-edit"  # merge's content edit landed
        assert final.position == 7  # the concurrent PATCH was NOT reverted
        assert final.status == "approved"  # …and approval was preserved


async def test_concurrent_patch_of_a_merge_deleted_panel_is_404_not_500(
    db_session, db_session_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete-race: the branch DELETES a panel; a ``PATCH`` of that same root panel lands
    while merge holds the board locks. The PATCH's UPDATE blocks, merge commits the delete,
    the UPDATE unblocks to match 0 rows → SQLAlchemy raises ``StaleDataError``. The repo
    ``update`` catches it and returns ``None`` (which the route maps to 404) rather than
    letting an unhandled 500 escape."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    root_id = root["project_id"]
    victim = root["panels"][2]  # clip-3

    branch_id, child_id = await _fork(
        db_session, root_id, tenant_id=tenant_id, subject_id=subject_id
    )
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await StoryboardPanelRepository(db_session).delete(
        child_panels["clip-3"].id,
        tenant_id=tenant_id,
        expected_lock_version=child_panels["clip-3"].lock_version,
    )
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    token = (await svc.diff(branch_id, tenant_id=tenant_id)).root_fingerprint
    locks_held, release = _install_apply_seam(svc, monkeypatch)

    merge_task = asyncio.create_task(
        svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=token)
    )
    async with db_session_factory() as patch_sess:
        patch_task: asyncio.Task[Any] | None = None
        try:
            await asyncio.wait_for(locks_held.wait(), timeout=10)
            patch_task = asyncio.create_task(
                _patch_position(patch_sess, victim, tenant_id=tenant_id, position=7)
            )
            await asyncio.sleep(0.5)
            assert not patch_task.done(), "the PATCH did not block on the board lock"

            release.set()
            await asyncio.wait_for(merge_task, timeout=10)
            await db_session.commit()  # merge (incl. the delete) lands, locks release

            # The PATCH unblocks onto a now-deleted row: repo.update returns None (→ 404),
            # NOT an unhandled StaleDataError → 500.
            result = await asyncio.wait_for(patch_task, timeout=10)
            assert result is None
            await patch_sess.rollback()
        finally:
            if patch_task is not None:
                patch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await patch_task
            merge_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await merge_task
            await patch_sess.rollback()

    # The panel really is gone on disk.
    async with db_session_factory() as verify:
        gone = (
            await verify.execute(select(StoryboardPanel).where(StoryboardPanel.id == victim))
        ).scalar_one_or_none()
        assert gone is None


# ─── per-field conflict resolution ──────────────────────────────────────────────
# A conflict where BOTH lanes edited an overlapping set of plain content columns can be
# resolved branch-vs-main PER COLUMN instead of taking a whole side. The invariants:
#   * the caller's per-column pick lands (branch value for X, main value for Y in ONE panel);
#   * a field only MAIN edited is PRESERVED (the blanket branch-overlay would silently revert
#     it — the reason a true 3-way apply exists);
#   * a delete-vs-edit conflict / a main-only lineage edit is NOT field-resolvable (→ 422),
#     because per-field cannot express "keep main's deletion / scene" — those need a whole-row
#     verdict, and honouring a field pick there would silently delete or revert.


async def _seed_two_field_conflict(
    db_session, *, tenant_id, subject_id, branch_fields, main_fields
):
    """Fork clip-1, then edit it differently on both lanes → one conflict entry.

    ``branch_fields`` / ``main_fields`` are the field dicts applied to the child and root
    copies of clip-1. Returns ``(root, branch_id, child_id)``; the conflict's lineage key is
    ``root["panels"][0]``.
    """
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await panels.update(child_panels["clip-1"].id, tenant_id=tenant_id, fields=branch_fields)
    await panels.update(root["panels"][0], tenant_id=tenant_id, fields=main_fields)
    await db_session.commit()
    return root, branch_id, child_id


async def test_field_resolution_splits_branch_and_main_per_column(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root, branch_id, _child = await _seed_two_field_conflict(
        db_session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        branch_fields={"vo_text": "theirs", "dialogue": "theirs-d"},
        main_fields={"vo_text": "ours", "dialogue": "ours-d"},
    )

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    result = await svc.merge(
        branch_id,
        tenant_id=tenant_id,
        root_fingerprint=diff.root_fingerprint,
        field_resolutions={root["panels"][0]: {"vo_text": "branch", "dialogue": "main"}},
    )
    await db_session.commit()

    assert result["applied"] == [root["panels"][0]]
    by_shot = await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id)
    merged = by_shot["clip-1"]
    # ONE panel carries the branch's vo_text AND main's dialogue …
    assert merged.vo_text == "theirs"
    assert merged.dialogue == "ours-d"
    # … upserted onto MAIN's existing row (id preserved → approval trail intact).
    assert merged.id == root["panels"][0]


async def test_field_resolution_preserves_a_main_only_edit(db_session) -> None:
    """The load-bearing no-data-loss proof: a field only MAIN changed survives.

    Branch edits vo_text + action_description; main edits vo_text + lighting_mood. Only
    vo_text conflicts. Resolving vo_text=branch must NOT revert main's lighting_mood — the
    blanket branch overlay would copy branch's lighting_mood (== base == None), silently
    discarding main's edit. A true 3-way overlay keeps it.
    """
    tenant_id, subject_id = uuid7(), uuid7()
    root, branch_id, _child = await _seed_two_field_conflict(
        db_session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        branch_fields={"vo_text": "theirs", "action_description": "branch-action"},
        main_fields={"vo_text": "ours", "lighting_mood": "main-noir"},
    )

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    await svc.merge(
        branch_id,
        tenant_id=tenant_id,
        root_fingerprint=diff.root_fingerprint,
        field_resolutions={root["panels"][0]: {"vo_text": "branch"}},
    )
    await db_session.commit()

    merged = (await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id))["clip-1"]
    assert merged.vo_text == "theirs"  # conflicting field, resolved to branch
    assert merged.action_description == "branch-action"  # branch-only edit lands
    assert merged.lighting_mood == "main-noir"  # main-only edit PRESERVED (the D-fix)
    assert merged.id == root["panels"][0]


async def test_field_resolution_partial_coverage_is_unresolved(db_session) -> None:
    """A field_resolutions that misses a conflicting field is as incomplete as none —
    merging would default the uncovered field to branch. → UnresolvedConflictError + the
    missing field is reported, and nothing is written."""
    tenant_id, subject_id = uuid7(), uuid7()
    root, branch_id, _child = await _seed_two_field_conflict(
        db_session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        branch_fields={"vo_text": "theirs", "dialogue": "theirs-d"},
        main_fields={"vo_text": "ours", "dialogue": "ours-d"},
    )

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    with pytest.raises(UnresolvedConflictError) as exc:
        await svc.merge(
            branch_id,
            tenant_id=tenant_id,
            root_fingerprint=diff.root_fingerprint,
            field_resolutions={root["panels"][0]: {"vo_text": "branch"}},  # dialogue missing
        )
    assert exc.value.keys == [root["panels"][0]]
    assert exc.value.missing == {root["panels"][0]: ["dialogue"]}
    await db_session.rollback()

    merged = (await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id))["clip-1"]
    assert merged.vo_text == "ours" and merged.dialogue == "ours-d"  # untouched


async def test_field_resolution_on_delete_vs_edit_is_rejected_panel_survives(db_session) -> None:
    """ADVERSARIAL A1 — the silent-delete guard. Branch DELETES clip-1, main edits it: a
    delete-vs-edit conflict has an EMPTY conflicting-field set, so it is not field-resolvable.
    A field_resolutions aimed at it must 422 (FieldResolutionTargetError), never vacuously
    'resolve' and route the panel to deletion. The panel must still stand afterwards."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await panels.delete(
        child_panels["clip-1"].id,
        tenant_id=tenant_id,
        expected_lock_version=child_panels["clip-1"].lock_version,
    )  # branch deletes
    await panels.update(
        root["panels"][0], tenant_id=tenant_id, fields={"vo_text": "ours"}
    )  # main edits
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    with pytest.raises(FieldResolutionTargetError) as exc:
        await svc.merge(
            branch_id,
            tenant_id=tenant_id,
            root_fingerprint=diff.root_fingerprint,
            field_resolutions={root["panels"][0]: {"vo_text": "branch"}},
        )
    assert exc.value.keys == [root["panels"][0]]
    await db_session.rollback()

    survivor = await _root_panel(db_session, root["panels"][0])
    assert survivor is not None and survivor.vo_text == "ours"  # NOT deleted


async def test_field_resolution_fenced_by_a_main_only_lineage_edit(db_session) -> None:
    """A content conflict where main ALSO made a main-only LINEAGE change (removed the
    character) is not field-resolvable: the content overlay cannot carry lineage, so honouring
    a field pick would revert main's cast. → FieldResolutionTargetError; use a whole-row verdict."""
    tenant_id, subject_id = uuid7(), uuid7()
    root, branch_id, _child = await _seed_two_field_conflict(
        db_session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        branch_fields={"vo_text": "theirs"},
        main_fields={"vo_text": "ours", "characters": []},  # main-only lineage change
    )

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    with pytest.raises(FieldResolutionTargetError):
        await svc.merge(
            branch_id,
            tenant_id=tenant_id,
            root_fingerprint=diff.root_fingerprint,
            field_resolutions={root["panels"][0]: {"vo_text": "branch"}},
        )
    await db_session.rollback()

    # A whole-row verdict is still accepted for the same conflict.
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    await svc.merge(
        branch_id,
        tenant_id=tenant_id,
        root_fingerprint=diff.root_fingerprint,
        resolutions={root["panels"][0]: "main"},
    )
    await db_session.commit()
    merged = (await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id))["clip-1"]
    assert merged.vo_text == "ours"


@pytest.mark.parametrize(
    ("main_extra", "assert_field", "kept"),
    [({"shot_key": "clip-1-renamed"}, "shot_key", "clip-1-renamed")],
)
async def test_field_resolution_fenced_by_a_main_only_shot_key_or_position(
    db_session, main_extra, assert_field, kept
) -> None:
    """``shot_key``/``position`` are re-sourced from the BRANCH in the apply (they are not
    carried by the per-field content overlay), so a main-ONLY rename/reorder cannot be honoured
    per-field — the conflict is fenced to a whole-row verdict. Without the fence a field
    resolution would silently revert main's rename/reorder to the branch (== base) value."""
    tenant_id, subject_id = uuid7(), uuid7()
    root, branch_id, _child = await _seed_two_field_conflict(
        db_session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        branch_fields={"vo_text": "theirs"},
        main_fields={"vo_text": "ours", **main_extra},
    )

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    with pytest.raises(FieldResolutionTargetError):
        await svc.merge(
            branch_id,
            tenant_id=tenant_id,
            root_fingerprint=diff.root_fingerprint,
            field_resolutions={root["panels"][0]: {"vo_text": "branch"}},
        )
    await db_session.rollback()

    # main's rename/reorder is intact — the merge refused rather than reverting it.
    survivor = await _root_panel(db_session, root["panels"][0])
    assert survivor is not None and getattr(survivor, assert_field) == kept


# ─── pure-disjoint auto-merge ────────────────────────────────────────────────
# Both lanes edit the SAME panel but touch DISJOINT fields (no field changed on both). That
# is not a disagreement — it is both edits, and it merges with NO human input, exactly as a
# line-based VCS auto-merges non-conflicting hunks. Before this, such a divergence was a
# ``conflict`` whose only lossless escape was ``keep_both`` (duplicating the panel); a
# whole-row ``branch`` verdict silently reverted main's field.
async def test_disjoint_field_edits_are_modified_not_conflict(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root, branch_id, _child = await _seed_two_field_conflict(
        db_session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        branch_fields={"action_description": "branch-action"},
        main_fields={"lighting_mood": "main-noir"},
    )

    diff = await StoryboardMergeService(db_session).diff(branch_id, tenant_id=tenant_id)
    assert [e.kind for e in diff.entries] == ["modified"], "disjoint edits are not a conflict"
    assert diff.conflict_count == 0
    entry = diff.entries[0]
    assert entry.key == root["panels"][0]
    assert entry.changed_fields == ["action_description"]
    assert entry.main_changed_fields == ["lighting_mood"]


async def test_disjoint_field_edits_auto_merge_preserves_both_sides(db_session) -> None:
    """THE no-data-loss proof: with NO resolution supplied, both sides' disjoint edits land
    on one panel — branch's ``action_description`` overlays, main's ``lighting_mood`` is kept.
    A blanket branch overlay would revert ``lighting_mood`` to base (None), discarding main."""
    tenant_id, subject_id = uuid7(), uuid7()
    root, branch_id, _child = await _seed_two_field_conflict(
        db_session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        branch_fields={"action_description": "branch-action"},
        main_fields={"lighting_mood": "main-noir"},
    )

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    result = await svc.merge(  # no resolutions / field_resolutions — it auto-merges
        branch_id, tenant_id=tenant_id, root_fingerprint=diff.root_fingerprint
    )
    await db_session.commit()

    assert result["applied"] == [root["panels"][0]]
    merged = (await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id))["clip-1"]
    assert merged.action_description == "branch-action"  # branch's disjoint edit lands
    assert merged.lighting_mood == "main-noir"  # main's disjoint edit PRESERVED
    assert merged.id == root["panels"][0]  # upserted onto main's row → trail intact
    assert merged.revision == 2  # the root row was touched exactly once
    assert await _events(db_session, root["panels"][0], tenant_id=tenant_id) == ["merged"]


async def test_disjoint_auto_merge_is_load_bearing(db_session, monkeypatch) -> None:
    """Negative control: neuter ``_auto_mergeable`` and the SAME disjoint scenario reverts to
    an unresolvable conflict — proving the reclassification, not some pre-existing path, is
    what lets it merge without input (so the success test above is not a tautology)."""
    from src.services.storyboard import merge_service as merge_module

    monkeypatch.setattr(merge_module, "_auto_mergeable", lambda _entry: False)

    tenant_id, subject_id = uuid7(), uuid7()
    _root, branch_id, _child = await _seed_two_field_conflict(
        db_session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        branch_fields={"action_description": "branch-action"},
        main_fields={"lighting_mood": "main-noir"},
    )

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    assert [e.kind for e in diff.entries] == ["conflict"]  # no longer demoted
    with pytest.raises(UnresolvedConflictError):
        await svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=diff.root_fingerprint)


async def test_disjoint_edit_with_a_main_only_fenced_field_stays_a_conflict(db_session) -> None:
    """A disjoint divergence the content overlay CANNOT carry stays a whole-row conflict.
    ``shot_key`` is re-sourced from the branch unconditionally in the apply, so a main-only
    ``shot_key`` change would be silently reverted by an auto-merge — it must not auto-merge."""
    tenant_id, subject_id = uuid7(), uuid7()
    _root, branch_id, _child = await _seed_two_field_conflict(
        db_session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        branch_fields={"vo_text": "theirs"},
        main_fields={"shot_key": "clip-1-renamed"},  # fenced, branch-sourced in the apply
    )

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    assert [e.kind for e in diff.entries] == ["conflict"]
    assert diff.conflict_count == 1
    with pytest.raises(UnresolvedConflictError):  # still needs a human whole-row pick
        await svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=diff.root_fingerprint)


# ─── sibling-to-sibling merge (WS3) ──────────────────────────────────────────
# Two branches forked off the SAME parent share ONE common ancestor — the parent's state
# before either diverged, pinned as ``parent_base_snapshot_id``. Merging A into B applies
# A's changes onto B three-way against that shared base, leaving BOTH branches open: a
# cross-pollinate, not a lifecycle close (source is not 'merged', neither child archived).
async def _fork_two_siblings(db_session, root_id, *, tenant_id, subject_id):
    """Fork two branches (A then B) off the same root. Returns (a_id, a_child, b_id, b_child)."""
    a_branch, a_child = await StoryboardBranchService(db_session).create_branch(
        root_id, tenant_id=tenant_id, name="alt-A", actor_subject=subject_id
    )
    b_branch, b_child = await StoryboardBranchService(db_session).create_branch(
        root_id, tenant_id=tenant_id, name="alt-B", actor_subject=subject_id
    )
    await db_session.commit()
    return a_branch.id, a_child.id, b_branch.id, b_child.id


async def test_siblings_share_one_parent_base_snapshot(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    a_id, _ac, b_id, _bc = await _fork_two_siblings(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    repo = StoryboardBranchRepository(db_session)
    a = await repo.get(a_id, tenant_id=tenant_id)
    b = await repo.get(b_id, tenant_id=tenant_id)
    assert a is not None and b is not None
    # The FIRST fork establishes the shared ancestor (its own base); every later fork reuses
    # it — so both siblings pin the SAME parent base.
    assert a.parent_base_snapshot_id is not None
    assert a.parent_base_snapshot_id == b.parent_base_snapshot_id == a.base_snapshot_id
    # …which is NOT B's own per-branch base (B forked later, off a possibly-moved parent).
    assert b.parent_base_snapshot_id != b.base_snapshot_id


async def test_sibling_merge_applies_source_edits_and_keeps_target_edits(db_session) -> None:
    """The core: A's changes land on B, and B's OWN edits survive (branch-centric on the
    source, so a panel only the target touched is not even an entry)."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    a_id, a_child, b_id, b_child = await _fork_two_siblings(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)
    a_panels = await _panels_by_shot(db_session, a_child, tenant_id=tenant_id)
    b_panels = await _panels_by_shot(db_session, b_child, tenant_id=tenant_id)
    await panels.update(a_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "A-one"})
    await panels.update(b_panels["clip-2"].id, tenant_id=tenant_id, fields={"vo_text": "B-two"})
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff_sibling(a_id, b_id, tenant_id=tenant_id)
    # Only A's clip-1 edit is an entry; B's own clip-2 edit is not (nothing to merge FROM A).
    assert [e.kind for e in diff.entries] == ["modified"]
    assert diff.conflict_count == 0

    result = await svc.merge_into_sibling(
        a_id, b_id, tenant_id=tenant_id, target_fingerprint=diff.root_fingerprint
    )
    await db_session.commit()
    assert result["applied"] == [root["panels"][0]]

    b_after = await _panels_by_shot(db_session, b_child, tenant_id=tenant_id)
    assert b_after["clip-1"].vo_text == "A-one"  # A's edit landed on B
    assert b_after["clip-2"].vo_text == "B-two"  # B's own edit PRESERVED
    # A's own board is untouched — a sibling merge writes only the target.
    a_after = await _panels_by_shot(db_session, a_child, tenant_id=tenant_id)
    assert a_after["clip-1"].vo_text == "A-one"
    assert a_after["clip-2"].vo_text == "two"


async def test_sibling_merge_leaves_both_branches_open_and_children_live(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    a_id, a_child, b_id, b_child = await _fork_two_siblings(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)
    a_panels = await _panels_by_shot(db_session, a_child, tenant_id=tenant_id)
    await panels.update(a_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "A-one"})
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff_sibling(a_id, b_id, tenant_id=tenant_id)
    await svc.merge_into_sibling(
        a_id, b_id, tenant_id=tenant_id, target_fingerprint=diff.root_fingerprint
    )
    await db_session.commit()

    repo = StoryboardBranchRepository(db_session)
    a = await repo.get(a_id, tenant_id=tenant_id)
    b = await repo.get(b_id, tenant_id=tenant_id)
    assert a is not None and a.status == "open"  # source NOT closed
    assert b is not None and b.status == "open"  # target stays open
    for child_id in (a_child, b_child):
        proj = (
            await db_session.execute(select(Project).where(Project.id == child_id))
        ).scalar_one()
        assert proj.status != "archived"


async def test_sibling_merge_conflict_needs_resolution_then_applies(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    a_id, a_child, b_id, b_child = await _fork_two_siblings(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)
    a_panels = await _panels_by_shot(db_session, a_child, tenant_id=tenant_id)
    b_panels = await _panels_by_shot(db_session, b_child, tenant_id=tenant_id)
    # BOTH edit clip-1, differently → a genuine conflict against the shared base.
    await panels.update(a_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "A-one"})
    await panels.update(b_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "B-one"})
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff_sibling(a_id, b_id, tenant_id=tenant_id)
    assert [e.kind for e in diff.entries] == ["conflict"]
    with pytest.raises(UnresolvedConflictError):
        await svc.merge_into_sibling(
            a_id, b_id, tenant_id=tenant_id, target_fingerprint=diff.root_fingerprint
        )
    await db_session.rollback()

    diff = await svc.diff_sibling(a_id, b_id, tenant_id=tenant_id)
    key = diff.entries[0].key
    await svc.merge_into_sibling(
        a_id,
        b_id,
        tenant_id=tenant_id,
        target_fingerprint=diff.root_fingerprint,
        resolutions={key: "branch"},  # take the source (A)
    )
    await db_session.commit()
    b_after = await _panels_by_shot(db_session, b_child, tenant_id=tenant_id)
    assert b_after["clip-1"].vo_text == "A-one"


async def test_sibling_merge_stale_target_fingerprint_is_409(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    a_id, a_child, b_id, _bc = await _fork_two_siblings(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)
    a_panels = await _panels_by_shot(db_session, a_child, tenant_id=tenant_id)
    await panels.update(a_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "A-one"})
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    with pytest.raises(StaleRootError):
        await svc.merge_into_sibling(
            a_id, b_id, tenant_id=tenant_id, target_fingerprint="not-the-real-fingerprint"
        )


async def test_merge_into_a_non_sibling_is_rejected(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root1 = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    root2 = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    a_id, _ac = await _fork(
        db_session, root1["project_id"], tenant_id=tenant_id, subject_id=subject_id, name="A"
    )
    x_id, _xc = await _fork(
        db_session, root2["project_id"], tenant_id=tenant_id, subject_id=subject_id, name="X"
    )
    svc = StoryboardMergeService(db_session)
    # Different parents → no shared base to three-way against.
    with pytest.raises(NotSiblingsError):
        await svc.diff_sibling(a_id, x_id, tenant_id=tenant_id)
    # A branch is not its own sibling.
    with pytest.raises(NotSiblingsError):
        await svc.diff_sibling(a_id, a_id, tenant_id=tenant_id)


async def test_sibling_merge_without_a_shared_base_is_409(db_session) -> None:
    """A branch predating the 0057 backfill has a NULL parent base; a sibling merge from it
    must refuse rather than degrade into 'take everything from the source' (data loss)."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    a_id, _ac, b_id, _bc = await _fork_two_siblings(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    repo = StoryboardBranchRepository(db_session)
    a = await repo.get(a_id, tenant_id=tenant_id)
    assert a is not None
    a.parent_base_snapshot_id = None  # simulate a pre-0057 branch
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    with pytest.raises(SharedBaseMissingError):
        await svc.diff_sibling(a_id, b_id, tenant_id=tenant_id)
