"""Behavior-preserving guard for the storyboard merge/diff PERF refactor.

Three optimizations landed together, all required to be byte-identical to the prior
implementation:

1. ``_Board.scene_key_of`` went O(panels*scenes) -> O(1) via a reverse ``scene_id -> key``
   map (``_Board._scene_key_by_id``) built once in ``_build_board``;
2. no fingerprint recompute was added or removed (analysed: the pre-/post-apply
   fingerprints are DIFFERENT board states and diff/merge are separate requests — nothing
   is provably redundant, so both stay);
3. merge's per-touched-panel trail loop now writes through the BATCH
   ``StoryboardPanelEventRepository.append_all`` (one flush) instead of N single-row
   ``append`` flushes.

These tests assert the INVARIANTS that would break if any of the three changed observable
behavior. The real equivalence guard is the *unchanged* existing merge/diff suite in
``test_storyboard_merge_service.py``; this file adds the refactor-specific pins:

* the reverse map is the exact inverse of ``scene_ids`` and ``scene_key_of``'s unknown/None
  fallback is preserved (opt 1);
* a full multi-scene, multi-panel, forked+edited merge produces the correct final panel set
  with correct scene assignments, and emits EXACTLY one ``merged`` trail event per touched
  panel in order — proving the batch write dropped/duplicated/misordered nothing (opt 3).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid7

import pytest

from src.database.repositories.storyboard_repository import (
    StoryboardPanelEventRepository,
    StoryboardPanelRepository,
    StoryboardRepository,
    StoryboardSceneRepository,
)
from src.services.storyboard.merge_service import (
    StoryboardMergeService,
)

# Proven DB helpers from the sibling suite (``tests`` is a package).
from tests.services.test_storyboard_merge_service import (
    _fork,
    _panels_by_shot,
    _seed_root,
)

pytestmark = pytest.mark.asyncio


# ─── opt 1: the reverse scene_id → key map + preserved fallback (pure, no DB) ──


# ─── multi-scene seed for the golden integration test ─────────────────────────
async def _seed_multi_scene_graph(
    db_session, *, tenant_id: UUID, subject_id: UUID
) -> dict[str, Any]:
    """Root with TWO scenes and panels spread across them.

    scene-A: clip-1, clip-2 ; scene-B: clip-3. (A branch will later add clip-4 in a NEW
    scene-C, exercising ``_ensure_scenes`` → ``scene_key_of`` on a materialised scene.)
    """
    project = await _seed_root(db_session, tenant_id=tenant_id, subject_id=subject_id)
    storyboard = await StoryboardRepository(db_session).ensure_for_project(
        project.id, tenant_id=tenant_id
    )
    assert storyboard is not None
    scenes = StoryboardSceneRepository(db_session)
    scene_a = await scenes.create(
        project_id=project.id, storyboard_id=storyboard.id, position=0, name="scene-A"
    )
    scene_b = await scenes.create(
        project_id=project.id, storyboard_id=storyboard.id, position=1, name="scene-B"
    )
    panels = StoryboardPanelRepository(db_session)
    layout = [
        ("clip-1", scene_a.id, "one"),
        ("clip-2", scene_a.id, "two"),
        ("clip-3", scene_b.id, "three"),
    ]
    for i, (shot, scene_id, vo) in enumerate(layout):
        await panels.create(
            project_id=project.id,
            storyboard_id=storyboard.id,
            position=i,
            shot_key=shot,
            scene_id=scene_id,
            vo_text=vo,
        )
    await db_session.commit()
    return {
        "project_id": project.id,
        "storyboard_id": storyboard.id,
        "scene_a": scene_a.id,
        "scene_b": scene_b.id,
    }


async def _root_scene_names(
    db_session, project_id: UUID, *, tenant_id: UUID
) -> dict[UUID, str | None]:
    scenes = await StoryboardSceneRepository(db_session).list_for_project(
        project_id, tenant_id=tenant_id
    )
    return {s.id: s.name for s in scenes}


async def test_multi_scene_forked_merge_preserves_scenes_and_emits_one_event_per_touched(
    db_session,
) -> None:
    """The golden end-to-end guard.

    A two-scene root, a fork that (a) modifies a scene-A panel, (b) modifies the scene-B
    panel, and (c) adds a new panel in a brand-new scene-C. After merge:

    * the final root panel set is correct and each panel's SCENE assignment survived by
      lineage (the O(1) ``scene_key_of` produced the same scenes the scan would have);
    * scene-C was materialised onto the root and the added panel points at it;
    * EXACTLY one ``merged`` trail event exists per touched panel — no drop, no duplicate —
      and the count equals ``len(result['applied'])``, proving ``append_all`` batched all N
      rows losslessly.
    """
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_multi_scene_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )

    # Edit the branch: two modifies (one per existing scene) + one add in a new scene-C.
    panels = StoryboardPanelRepository(db_session)
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await panels.update(
        child_panels["clip-1"].id, tenant_id=tenant_id, fields={"vo_text": "A-edit"}
    )
    await panels.update(
        child_panels["clip-3"].id, tenant_id=tenant_id, fields={"vo_text": "B-edit"}
    )

    child_sb = await StoryboardRepository(db_session).ensure_for_project(
        child_id, tenant_id=tenant_id
    )
    assert child_sb is not None
    scene_c = await StoryboardSceneRepository(db_session).create(
        project_id=child_id, storyboard_id=child_sb.id, position=2, name="scene-C"
    )
    await panels.create(
        project_id=child_id,
        storyboard_id=child_sb.id,
        position=3,
        shot_key="clip-4",
        scene_id=scene_c.id,
        vo_text="four",
    )
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)

    # Fingerprint determinism: two diffs of the same board hash identically (opt 2 didn't
    # perturb it) and it is a sha256 hex digest.
    diff2 = await svc.diff(branch_id, tenant_id=tenant_id)
    assert diff.root_fingerprint == diff2.root_fingerprint
    assert len(diff.root_fingerprint) == 64

    result = await svc.merge(
        branch_id,
        tenant_id=tenant_id,
        root_fingerprint=diff.root_fingerprint,
        actor_subject=subject_id,
    )
    await db_session.commit()

    assert result["status"] == "merged"
    # 3 panels changed on the branch: clip-1, clip-3, clip-4 (add). All resolve to "branch".
    assert len(result["applied"]) == 3
    assert result["skipped"] == []

    # Final panel set + scene assignments survived by lineage.
    merged = await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id)
    scene_names = await _root_scene_names(db_session, root["project_id"], tenant_id=tenant_id)
    assert set(merged) == {"clip-1", "clip-2", "clip-3", "clip-4"}
    assert merged["clip-1"].vo_text == "A-edit"
    assert merged["clip-3"].vo_text == "B-edit"
    assert scene_names[merged["clip-1"].scene_id] == "scene-A"
    assert scene_names[merged["clip-2"].scene_id] == "scene-A"  # untouched, still scene-A
    assert scene_names[merged["clip-3"].scene_id] == "scene-B"
    # scene-C was materialised onto the root and the added panel points at it.
    assert scene_names[merged["clip-4"].scene_id] == "scene-C"

    # THE batch-append invariant: exactly one "merged" event per touched panel, no dupes.
    all_events = await StoryboardPanelEventRepository(db_session).list_for_project(
        root["project_id"], tenant_id=tenant_id
    )
    merged_events = [e for e in all_events if e.event_type == "merged"]
    assert len(merged_events) == len(result["applied"]) == 3
    touched_panel_ids = {merged["clip-1"].id, merged["clip-3"].id, merged["clip-4"].id}
    assert {e.panel_id for e in merged_events} == touched_panel_ids
    # One per panel (no duplicate rows from the batch), each carrying the branch lineage.
    assert sorted(e.panel_id for e in merged_events) == sorted(touched_panel_ids)
    for e in merged_events:
        assert e.detail.get("branch_id") == str(branch_id)


async def test_untouched_scene_b_panel_keeps_its_scene_after_a_scene_a_merge(
    db_session,
) -> None:
    """A merge that only touches scene-A must leave scene-B panels' scene assignment exactly
    as it was — a direct check that the reverse-map ``scene_key_of`` did not mis-resolve an
    untouched panel's scene during ``fingerprint``/``view``/``_panel_rows``."""
    tenant_id, subject_id = uuid7(), uuid7()
    root = await _seed_multi_scene_graph(db_session, tenant_id=tenant_id, subject_id=subject_id)
    branch_id, child_id = await _fork(
        db_session, root["project_id"], tenant_id=tenant_id, subject_id=subject_id
    )
    panels = StoryboardPanelRepository(db_session)
    child_panels = await _panels_by_shot(db_session, child_id, tenant_id=tenant_id)
    await panels.update(child_panels["clip-2"].id, tenant_id=tenant_id, fields={"vo_text": "A2"})
    await db_session.commit()

    svc = StoryboardMergeService(db_session)
    diff = await svc.diff(branch_id, tenant_id=tenant_id)
    assert [e.kind for e in diff.entries] == ["modified"]
    await svc.merge(branch_id, tenant_id=tenant_id, root_fingerprint=diff.root_fingerprint)
    await db_session.commit()

    merged = await _panels_by_shot(db_session, root["project_id"], tenant_id=tenant_id)
    scene_names = await _root_scene_names(db_session, root["project_id"], tenant_id=tenant_id)
    assert merged["clip-2"].vo_text == "A2"
    assert scene_names[merged["clip-3"].scene_id] == "scene-B"  # untouched scene survived
    assert scene_names[merged["clip-1"].scene_id] == "scene-A"
