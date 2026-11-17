"""Storyboard BRANCH (fork) service — the deep-copy into a hidden child project.

A fork is a snapshot read back out into a DIFFERENT, empty project, so every
invariant that makes a later three-way merge possible is established HERE and
nowhere else. These tests pin all of them against real postgres (``db_session``,
owner-bypass RLS — tenant scoping is the repo project-join, same as the sibling
snapshot tests):

* **Re-minted ids** — no scene/panel/asset/join-row id is shared with the root.
  A uuid PK is global; a shared id would mean the two projects are the same rows.
* **``origin_id`` lineage** — each copied row points at its SOURCE, and a row that
  ALREADY carries lineage keeps the ORIGINAL id rather than re-rooting at the
  intermediate hop (the "if set, else id" rule in the service).
* **Reference remap** — child panels' ``scene_id`` and child join rows resolve
  ONLY inside the child. Any surviving root id is a cross-project leak.
* **``tenant_id`` inheritance** + cross-tenant fork → ``ProjectNotFoundError``.
* **``branch_base`` survives the auto-prune** — the bug the design originally had:
  an ``auto`` base would be swept out from under an open branch, leaving a branch
  that can never be merged.
* **abandon** — archives the child; a ``merged`` branch refuses.

Mirrors ``test_storyboard_snapshot_service.py`` (its inverse direction).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4, uuid7

import pytest
from sqlalchemy import select

from src.database.models import Project, StoryboardPanelAsset
from src.database.repositories.storyboard_repository import (
    StoryboardAssetRepository,
    StoryboardBranchRepository,
    StoryboardPanelRepository,
    StoryboardRepository,
    StoryboardSceneRepository,
    StoryboardSnapshotRepository,
)
from src.services.storyboard.branch_service import (
    BranchAlreadyMergedError,
    BranchNotFoundError,
    OpenChildBranchesError,
    ProjectNotFoundError,
    StoryboardBranchService,
)
from src.services.storyboard.storyboard_snapshot_service import StoryboardSnapshotService
from tests.support.storyboard_locks import board_lock_versions_for_snapshot

pytestmark = pytest.mark.asyncio


async def _seed_project(db_session, *, tenant_id: UUID, subject_id: UUID) -> Project:
    """A root project with NON-EMPTY creative context — brief/brand/voice/theme are
    what the fork must carry over, so seeding ``{}`` would make that test vacuous."""
    project = Project(
        id=uuid7(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="branch-root",
        brief={"goal": "sell the thing"},
        brand={"palette": ["#000", "#fff"]},
        voice={"tone": "wry"},
        theme={"font": "Inter"},
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def _seed_graph(db_session, *, tenant_id: UUID, subject_id: UUID) -> dict[str, Any]:
    """Root project + 2 scenes + 3 panels (distinct shot_keys) + a full cast:
    2 characters, 1 prop and 1 LOCATION — i.e. every ``role`` the join table carries,
    including the one guarded by the ``uq_storyboard_panel_assets_one_location``
    partial index. This is the graph the fork has to reproduce.
    """
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    storyboard = await StoryboardRepository(db_session).ensure_for_project(
        project.id, tenant_id=tenant_id
    )
    assert storyboard is not None

    scenes = StoryboardSceneRepository(db_session)
    panels = StoryboardPanelRepository(db_session)
    assets = StoryboardAssetRepository(db_session)

    s1 = await scenes.create(
        project_id=project.id, storyboard_id=storyboard.id, position=0, name="scene-1"
    )
    s2 = await scenes.create(
        project_id=project.id, storyboard_id=storyboard.id, position=1, name="scene-2"
    )
    a = await assets.create(project_id=project.id, position=0, kind="character", name="A")
    b = await assets.create(project_id=project.id, position=1, kind="character", name="B")
    prop = await assets.create(project_id=project.id, position=2, kind="prop", name="P")
    loc = await assets.create(project_id=project.id, position=3, kind="location", name="L")

    p1 = await panels.create(
        project_id=project.id,
        storyboard_id=storyboard.id,
        position=0,
        shot_key="clip-1",
        scene_id=s1.id,
        vo_text="one",
        characters=[a.id, b.id],
        props=[prop.id],
        location_asset_id=loc.id,
    )
    p2 = await panels.create(
        project_id=project.id,
        storyboard_id=storyboard.id,
        position=1,
        shot_key="clip-2",
        scene_id=s1.id,
        vo_text="two",
    )
    # An UNSCENED panel: the fork's scene remap is None-safe, and only a panel with a
    # NULL scene_id can prove it.
    p3 = await panels.create(
        project_id=project.id,
        storyboard_id=storyboard.id,
        position=2,
        shot_key="clip-3",
        vo_text="three",
    )
    await db_session.commit()
    return {
        "project": project,
        "storyboard_id": storyboard.id,
        "scenes": [s1.id, s2.id],
        "panels": [p1.id, p2.id, p3.id],
        "assets": [a.id, b.id, prop.id, loc.id],
        "scened_panels": {p1.id: s1.id, p2.id: s1.id},
        "unscened_panel": p3.id,
    }


async def _join_rows(db_session, panel_ids: list[UUID]) -> list[StoryboardPanelAsset]:
    rows = (
        await db_session.execute(
            select(StoryboardPanelAsset)
            .where(StoryboardPanelAsset.panel_id.in_(panel_ids))
            .execution_options(populate_existing=True)
        )
    ).scalars()
    return list(rows)


async def _child_graph(db_session, child_id: UUID, *, tenant_id: UUID) -> dict[str, Any]:
    scenes = await StoryboardSceneRepository(db_session).list_for_project(
        child_id, tenant_id=tenant_id
    )
    panels = await StoryboardPanelRepository(db_session).list_for_project(
        child_id, tenant_id=tenant_id
    )
    assets = await StoryboardAssetRepository(db_session).list_for_project(
        child_id, tenant_id=tenant_id
    )
    joins = await _join_rows(db_session, [p.id for p in panels])
    return {"scenes": scenes, "panels": panels, "assets": assets, "joins": joins}


# ─── 1. Every id is re-minted ────────────────────────────────────────────────
async def test_fork_re_mints_every_id(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)

    _branch, child = await StoryboardBranchService(db_session).create_branch(
        root["project"].id, tenant_id=tenant_id, name="alt-cut", actor_subject=subject_id
    )
    child_id = child.id
    await db_session.commit()

    graph = await _child_graph(db_session, child_id, tenant_id=tenant_id)
    # Same SHAPE …
    assert len(graph["scenes"]) == 2
    assert len(graph["panels"]) == 3
    assert len(graph["assets"]) == 4
    assert len(graph["joins"]) == 4  # 2 characters + 1 prop + 1 location

    root_ids = {
        *root["scenes"],
        *root["panels"],
        *root["assets"],
        *(j.id for j in await _join_rows(db_session, root["panels"])),
    }
    child_ids = {
        *(s.id for s in graph["scenes"]),
        *(p.id for p in graph["panels"]),
        *(a.id for a in graph["assets"]),
        *(j.id for j in graph["joins"]),
    }
    # … but NOT one shared id. A uuid PK is global: a shared id would mean the two
    # projects are literally the same row, and an edit to one would be an edit to both.
    assert root_ids & child_ids == set()
    assert child_id != root["project"].id


# ─── 2. origin_id lineage ────────────────────────────────────────────────────
async def test_fork_stamps_origin_id_on_every_copied_row(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)

    _branch, child = await StoryboardBranchService(db_session).create_branch(
        root["project"].id, tenant_id=tenant_id, name="lineage", actor_subject=subject_id
    )
    child_id = child.id
    await db_session.commit()

    graph = await _child_graph(db_session, child_id, tenant_id=tenant_id)
    # A fresh root row has no lineage of its own, so the child's origin_id must be
    # exactly the SOURCE row's id — the merge key.
    assert {s.origin_id for s in graph["scenes"]} == set(root["scenes"])
    assert {p.origin_id for p in graph["panels"]} == set(root["panels"])
    assert {a.origin_id for a in graph["assets"]} == set(root["assets"])
    # …and the origin is never the row's own (re-minted) id.
    assert all(p.origin_id != p.id for p in graph["panels"])
    # Root rows are untouched: the fork is a copy, not a move.
    root_panels = await StoryboardPanelRepository(db_session).list_for_project(
        root["project"].id, tenant_id=tenant_id
    )
    assert all(p.origin_id is None for p in root_panels)


async def test_fork_preserves_pre_existing_origin_id_rather_than_re_rooting(db_session) -> None:
    """The "``origin_id`` if set, else ``id``" rule.

    A source row that ALREADY carries lineage keeps the ORIGINAL id — the copy does not
    re-root lineage at the intermediate row. Branch-of-branch is refused outright (see
    below), so the only way to reach this state today is a row whose lineage was stamped
    by an earlier fork+merge; we simulate it by stamping ``origin_id`` on a root row.
    Getting this wrong would silently break merge for any row that has ever been forked.
    """
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    panels = StoryboardPanelRepository(db_session)

    ancestral = uuid7()  # an id from an earlier generation
    p1 = root["panels"][0]
    await panels.update(p1, tenant_id=tenant_id, fields={"origin_id": ancestral})
    await db_session.commit()

    _branch, child = await StoryboardBranchService(db_session).create_branch(
        root["project"].id, tenant_id=tenant_id, name="re-fork", actor_subject=subject_id
    )
    child_id = child.id
    await db_session.commit()

    child_panels = await panels.list_for_project(child_id, tenant_id=tenant_id)
    copied = next(p for p in child_panels if p.shot_key == "clip-1")
    assert copied.origin_id == ancestral, "lineage must NOT re-root at the intermediate row"
    assert copied.origin_id != p1
    # The panels with no prior lineage still root at their direct source.
    other = next(p for p in child_panels if p.shot_key == "clip-2")
    assert other.origin_id == root["panels"][1]


# ─── 3. Reference remap — zero cross-project leakage ─────────────────────────
async def test_fork_remaps_every_reference_into_the_child(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)

    _branch, child = await StoryboardBranchService(db_session).create_branch(
        root["project"].id, tenant_id=tenant_id, name="remap", actor_subject=subject_id
    )
    child_id = child.id
    await db_session.commit()

    graph = await _child_graph(db_session, child_id, tenant_id=tenant_id)
    child_scene_ids = {s.id for s in graph["scenes"]}
    child_panel_ids = {p.id for p in graph["panels"]}
    child_asset_ids = {a.id for a in graph["assets"]}

    # Every row belongs to the child project, and to the child's own storyboard.
    assert {s.project_id for s in graph["scenes"]} == {child_id}
    assert {p.project_id for p in graph["panels"]} == {child_id}
    assert {a.project_id for a in graph["assets"]} == {child_id}
    assert len({p.storyboard_id for p in graph["panels"]}) == 1
    assert {p.storyboard_id for p in graph["panels"]} != {root["storyboard_id"]}

    # scene_id points at CHILD scenes only — never at a root scene — and the unscened
    # panel stays unscened (the remap is None-safe).
    scened = [p for p in graph["panels"] if p.scene_id is not None]
    assert len(scened) == 2
    assert all(p.scene_id in child_scene_ids for p in scened)
    assert not any(p.scene_id in set(root["scenes"]) for p in scened)
    unscened = [p for p in graph["panels"] if p.scene_id is None]
    assert len(unscened) == 1 and unscened[0].shot_key == "clip-3"

    # Join rows reference only child panels + child assets.
    assert all(j.panel_id in child_panel_ids for j in graph["joins"])
    assert all(j.asset_id in child_asset_ids for j in graph["joins"])
    root_ids = set(root["panels"]) | set(root["assets"])
    assert not any(j.panel_id in root_ids or j.asset_id in root_ids for j in graph["joins"])

    # The ROOT's graph is untouched — the copy did not steal its rows.
    root_joins = await _join_rows(db_session, root["panels"])
    assert len(root_joins) == 4
    assert all(j.asset_id in set(root["assets"]) for j in root_joins)


# ─── 4. tenant_id inheritance + the cross-tenant 404 ─────────────────────────
async def test_child_inherits_tenant_id(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)

    _branch, child = await StoryboardBranchService(db_session).create_branch(
        root["project"].id, tenant_id=tenant_id, name="tenant", actor_subject=subject_id
    )
    await db_session.commit()

    # RLS keys off tenant_id — a child that did not inherit it is a cross-tenant leak,
    # not a cosmetic bug.
    assert child.tenant_id == root["project"].tenant_id == tenant_id
    assert child.is_branch is True
    assert child.parent_project_id is None  # NOT the course→lesson hierarchy


async def test_cross_tenant_fork_raises_project_not_found(db_session) -> None:
    """Tenant B forking tenant A's project is a 404, never a leak — and never a
    distinguishable 403, which would itself be an existence oracle."""
    tenant_a, tenant_b = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_a, subject_id=uuid7())
    # Read the id BEFORE the rollback below: a rollback expires every ORM instance in
    # the session, and re-reading an expired attribute would lazy-load — which on an
    # AsyncSession is a MissingGreenlet, not a query.
    root_id = root["project"].id

    with pytest.raises(ProjectNotFoundError):
        await StoryboardBranchService(db_session).create_branch(
            root_id, tenant_id=tenant_b, name="theft", actor_subject=uuid7()
        )
    await db_session.rollback()

    # Nothing was created on the way to the raise.
    branches = await StoryboardBranchRepository(db_session).list_for_root(
        root_id, tenant_id=tenant_a
    )
    assert branches == []


async def test_fork_of_unknown_project_raises(db_session) -> None:
    with pytest.raises(ProjectNotFoundError):
        await StoryboardBranchService(db_session).create_branch(
            uuid4(), tenant_id=uuid7(), name="ghost", actor_subject=uuid7()
        )


# ─── 5. Branch is hidden from every project listing ──────────────────────────
async def test_branch_child_is_hidden_from_project_repository_list(db_session) -> None:
    from src.database.repositories.project_repository import ProjectRepository

    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)

    _branch, child = await StoryboardBranchService(db_session).create_branch(
        root["project"].id, tenant_id=tenant_id, name="hidden", actor_subject=subject_id
    )
    child_id = child.id
    await db_session.commit()

    repo = ProjectRepository(db_session)
    listed = await repo.list(tenant_id=tenant_id, limit=50, offset=0)
    ids = {p.id for p in listed.items}
    assert root["project"].id in ids
    assert child_id not in ids, "a branch must never surface in the generic project list"
    assert listed.total == 1  # the total agrees with the filter, not with the raw table

    # …not even with include_archived (the branch is hidden, not merely un-archived).
    listed_all = await repo.list(tenant_id=tenant_id, include_archived=True, limit=50, offset=0)
    assert child_id not in {p.id for p in listed_all.items}

    # …and not in the root's CHILDREN list either (that is the course→lesson tree).
    with_children = await repo.get_with_children(root["project"].id, tenant_id=tenant_id)
    assert with_children is not None
    assert child_id not in {c.id for c in with_children.children}


# ─── 6. Branching a branch chains to its IMMEDIATE parent ────────────────────
async def test_branching_a_branch_chains_to_its_immediate_parent(db_session) -> None:
    """A fork of a branch is supported, and it is 1-hop like every other fork.

    ``C``'s link row points at ``B``'s PROJECT — not at the ultimate root — so
    diff/merge/lock/RLS all resolve ``C``'s "root" as its immediate parent with no special
    case, and ``branch_context`` shows ``B`` as where ``C`` came from. Re-rooting ``C`` at
    ``R`` instead would conflate ``B``'s unmerged edits with ``C``'s own on the way back
    up. The ordering rule that keeps the chain sound is asserted separately (a branch with
    open children refuses to merge/abandon).
    """
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    root_id = root["project"].id

    svc = StoryboardBranchService(db_session)
    first, child = await svc.create_branch(
        root_id, tenant_id=tenant_id, name="first", actor_subject=subject_id
    )
    child_id = child.id
    await db_session.commit()
    assert first.root_project_id == root_id

    # The fork of the BRANCH — this used to raise BranchOfBranchError.
    second, grandchild = await svc.create_branch(
        child_id, tenant_id=tenant_id, name="second", actor_subject=subject_id
    )
    await db_session.commit()

    assert second.status == "open"
    # Chained to B, NOT flattened onto R.
    assert second.root_project_id == child_id
    assert second.root_project_id != root_id
    assert grandchild.is_branch is True
    assert grandchild.tenant_id == tenant_id

    # Lineage survived the RE-fork: the grandchild's panels still carry the ROOT's origin
    # ids (not the middle branch's re-minted ones), which is what lets the same key match
    # on both hops back up.
    panels_repo = StoryboardPanelRepository(db_session)
    root_origins = {p.id for p in await panels_repo.list_for_project(root_id, tenant_id=tenant_id)}
    grand_panels = await panels_repo.list_for_project(grandchild.id, tenant_id=tenant_id)
    assert {p.origin_id for p in grand_panels} == root_origins
    assert {p.id for p in grand_panels}.isdisjoint(root_origins)


async def test_abandoning_a_branch_with_open_children_is_refused(db_session) -> None:
    """Abandoning ``B`` strands ``C`` exactly as thoroughly as merging it does.

    ``B``'s child project is ARCHIVED by abandon, and ``B`` goes terminal — so ``C``, which
    can only ever merge into ``B``, would be merging into a dead end. Refuse, and say so.
    Abandoning ``C`` first then frees ``B``.
    """
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)

    svc = StoryboardBranchService(db_session)
    first, child = await svc.create_branch(
        root["project"].id, tenant_id=tenant_id, name="first", actor_subject=subject_id
    )
    first_id = first.id
    second, _grandchild = await svc.create_branch(
        child.id, tenant_id=tenant_id, name="second", actor_subject=subject_id
    )
    second_id = second.id
    await db_session.commit()

    with pytest.raises(OpenChildBranchesError) as caught:
        await svc.abandon(first_id, tenant_id=tenant_id)
    assert caught.value.open_children == 1
    await db_session.rollback()

    # The parent is still open — the refusal wrote nothing.
    repo = StoryboardBranchRepository(db_session)
    still = await repo.get(first_id, tenant_id=tenant_id)
    assert still is not None
    assert still.status == "open"

    # Close the child, and the parent is free to abandon.
    await svc.abandon(second_id, tenant_id=tenant_id)
    await db_session.commit()
    freed = await svc.abandon(first_id, tenant_id=tenant_id)
    await db_session.commit()
    assert freed.status == "abandoned"


# ─── 7. The creative context comes along ─────────────────────────────────────
async def test_fork_copies_the_creative_context(db_session) -> None:
    """A branch is an alternative CUT of the same project, not a new project: dropping
    brief/brand/voice/theme would render the branch's panels against a DIFFERENT brand
    and voice, so the diff would measure the fork's own data loss, not the author's edits.
    """
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    parent = root["project"]

    _branch, child = await StoryboardBranchService(db_session).create_branch(
        parent.id, tenant_id=tenant_id, name="context", actor_subject=subject_id
    )
    await db_session.commit()

    assert child.brief == parent.brief == {"goal": "sell the thing"}
    assert child.brand == parent.brand == {"palette": ["#000", "#fff"]}
    assert child.voice == parent.voice == {"tone": "wry"}
    assert child.theme == parent.theme == {"font": "Inter"}
    assert child.type == parent.type
    assert child.visibility == parent.visibility
    assert child.owner_subject_id == parent.owner_subject_id
    assert child.title == "context"  # the ONE field that is the branch's own
    assert child.status == "drafting"


# ─── 8. branch_base survives the auto-prune ──────────────────────────────────
async def test_base_snapshot_is_branch_base_kind(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)

    branch, _child = await StoryboardBranchService(db_session).create_branch(
        root["project"].id, tenant_id=tenant_id, name="base", actor_subject=subject_id
    )
    base_id = branch.base_snapshot_id
    await db_session.commit()

    base = await StoryboardSnapshotRepository(db_session).get_meta(
        base_id, project_id=root["project"].id, tenant_id=tenant_id
    )
    assert base is not None
    assert base.kind == "branch_base"  # NOT 'auto' (prunable) and NOT 'manual' (deletable)


async def test_branch_base_survives_an_auto_prune(db_session) -> None:
    """THE bug the design originally had.

    An ``auto`` base would be swept by ``prune_auto_for_project`` — whose ``protect_id``
    shields exactly ONE id, not the arbitrarily many open branches — leaving a branch
    whose merge base has vanished, i.e. a branch that can never be merged. Churn the root
    hard enough to blow past the retention cap and assert the base is still there.
    """
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    project_id = root["project"].id

    branch, _child = await StoryboardBranchService(db_session).create_branch(
        project_id, tenant_id=tenant_id, name="prune-me-if-you-can", actor_subject=subject_id
    )
    base_id = branch.base_snapshot_id
    await db_session.commit()

    snapshots = StoryboardSnapshotService(db_session)
    manual = await snapshots.create_snapshot(
        project_id, tenant_id=tenant_id, kind="manual", note="keep"
    )
    manual_id = manual.id
    for _ in range(4):
        await snapshots.create_snapshot(project_id, tenant_id=tenant_id, kind="auto", note="churn")
    await db_session.commit()

    repo = StoryboardSnapshotRepository(db_session)
    deleted = await repo.prune_auto_for_project(project_id, keep=1)
    await db_session.commit()

    assert deleted == 3, "only the surplus AUTO snapshots are prunable"
    remaining = await repo.list_for_project(project_id, tenant_id=tenant_id, limit=50)
    kinds = [s.kind for s in remaining]
    assert kinds.count("auto") == 1
    assert kinds.count("branch_base") == 1
    assert kinds.count("manual") == 1
    assert base_id in {s.id for s in remaining}, "the branch's merge base was pruned away"
    assert manual_id in {s.id for s in remaining}

    # The branch row still resolves through its base — the FK was never orphaned.
    still = await StoryboardBranchRepository(db_session).get(branch.id, tenant_id=tenant_id)
    assert still is not None and still.base_snapshot_id == base_id


async def test_branch_base_survives_restore_driven_auto_churn(db_session) -> None:
    """Same guard through the REAL churn path: every ``restore`` mints an ``auto``
    safety copy and then prunes to the cap. Pin the cap at 1 and restore repeatedly —
    the base must outlive all of it."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    project_id = root["project"].id

    branch, _child = await StoryboardBranchService(db_session).create_branch(
        project_id, tenant_id=tenant_id, name="churn", actor_subject=subject_id
    )
    base_id = branch.base_snapshot_id
    await db_session.commit()

    capped = StoryboardSnapshotService(db_session, max_auto_per_project=1)
    manual = await capped.create_snapshot(project_id, tenant_id=tenant_id, kind="manual", note="v1")
    manual_id = manual.id
    await db_session.commit()
    for _ in range(3):
        await capped.restore(
            manual_id,
            tenant_id=tenant_id,
            actor_subject=subject_id,
            expected_lock_versions=await board_lock_versions_for_snapshot(db_session, manual_id),
        )
        await db_session.commit()

    repo = StoryboardSnapshotRepository(db_session)
    remaining = await repo.list_for_project(project_id, tenant_id=tenant_id, limit=50)
    kinds = [s.kind for s in remaining]
    assert kinds.count("auto") == 1  # the cap held
    assert base_id in {s.id for s in remaining}  # …and it never touched the base
    assert kinds.count("branch_base") == 1


# ─── 10. abandon ─────────────────────────────────────────────────────────────
async def test_abandon_marks_branch_and_archives_the_child(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)

    svc = StoryboardBranchService(db_session)
    branch, child = await svc.create_branch(
        root["project"].id, tenant_id=tenant_id, name="doomed", actor_subject=subject_id
    )
    branch_id, child_id = branch.id, child.id
    await db_session.commit()

    abandoned = await svc.abandon(branch_id, tenant_id=tenant_id)
    await db_session.commit()

    assert abandoned.status == "abandoned"
    assert abandoned.merged_at is None  # it records a MERGE; this branch never had one

    # The child is ARCHIVED, not deleted — its panels are the only record of the
    # discarded work.
    reloaded = (
        await db_session.execute(
            select(Project).where(Project.id == child_id).execution_options(populate_existing=True)
        )
    ).scalar_one()
    assert reloaded.status == "archived"
    assert reloaded.archived_at is not None
    assert reloaded.is_branch is True

    # Re-abandoning is idempotent, not an error.
    again = await svc.abandon(branch_id, tenant_id=tenant_id)
    await db_session.commit()
    assert again.status == "abandoned"


async def test_abandon_a_merged_branch_raises(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)

    svc = StoryboardBranchService(db_session)
    branch, child = await svc.create_branch(
        root["project"].id, tenant_id=tenant_id, name="merged", actor_subject=subject_id
    )
    branch_id, child_id = branch.id, child.id
    await db_session.commit()

    repo = StoryboardBranchRepository(db_session)
    await repo.mark_merged(branch)
    await db_session.commit()

    # ``merged`` is terminal: the root already holds the content, and abandoning would
    # archive the very project the merge's provenance still points at.
    with pytest.raises(BranchAlreadyMergedError):
        await svc.abandon(branch_id, tenant_id=tenant_id)
    await db_session.rollback()

    child_row = (
        await db_session.execute(
            select(Project).where(Project.id == child_id).execution_options(populate_existing=True)
        )
    ).scalar_one()
    assert child_row.status != "archived", "a refused abandon must not archive the child"


async def test_abandon_cross_tenant_or_unknown_raises_branch_not_found(db_session) -> None:
    tenant_a, tenant_b = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_a, subject_id=uuid7())

    svc = StoryboardBranchService(db_session)
    branch, _child = await svc.create_branch(
        root["project"].id, tenant_id=tenant_a, name="mine", actor_subject=uuid7()
    )
    branch_id = branch.id
    await db_session.commit()

    with pytest.raises(BranchNotFoundError):
        await svc.abandon(branch_id, tenant_id=tenant_b)
    with pytest.raises(BranchNotFoundError):
        await svc.abandon(uuid4(), tenant_id=tenant_a)


# ─── 12. A full cast forks without a constraint deferral ─────────────────────
async def test_fork_with_location_and_multi_role_cast_needs_no_deferral(db_session) -> None:
    """The ordered insert (scenes → assets → panels → joins) into an EMPTY child cannot
    transiently violate ``uq_storyboard_panels_project_shot``, the ``asset_id`` FK, or
    the ``uq_storyboard_panel_assets_one_location`` partial index — which is exactly why
    the fork, unlike ``restore``, ships NO ``SET CONSTRAINTS ALL DEFERRED``. If that
    reasoning is ever wrong, THIS is the test that raises an IntegrityError.
    """
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)

    _branch, child = await StoryboardBranchService(db_session).create_branch(
        root["project"].id, tenant_id=tenant_id, name="cast", actor_subject=subject_id
    )
    child_id = child.id
    await db_session.commit()  # the ordered insert commits clean, or this raises

    graph = await _child_graph(db_session, child_id, tenant_id=tenant_id)
    by_shot = {p.shot_key: p for p in graph["panels"]}
    assert set(by_shot) == {"clip-1", "clip-2", "clip-3"}  # shot_keys copied verbatim

    cast_panel = by_shot["clip-1"]
    child_assets = {a.id: a for a in graph["assets"]}
    by_role: dict[str, list[tuple[str, int]]] = {}
    for j in graph["joins"]:
        assert j.panel_id == cast_panel.id  # only the cast panel carries refs
        by_role.setdefault(j.role, []).append((child_assets[j.asset_id].name, j.position))

    assert sorted(by_role["character"], key=lambda t: t[1]) == [("A", 0), ("B", 1)]  # ORDER kept
    assert by_role["prop"] == [("P", 0)]
    assert by_role["location"] == [("L", 0)]  # exactly ONE location — the partial index held

    # The projected refs resolve to CHILD assets, not the root's.
    assert cast_panel.location_asset_id in child_assets
    assert cast_panel.location_asset_id not in set(root["assets"])
    assert [UUID(c) for c in cast_panel.characters] != [root["assets"][0], root["assets"][1]]
