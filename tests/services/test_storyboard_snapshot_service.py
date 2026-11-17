"""Storyboard version-snapshot + restore service (Phase 5).

Seeds a real storyboard graph (≥2 scenes, ≥3 panels with distinct shot_keys,
≥1 asset, one approved panel with a live approval event), snapshots it, then
mutates the live graph HARD — content edit, a shot_key SWAP (which only survives
because the restore path defers the DEFERRABLE unique constraint), a scene delete,
and a NEW panel — and restores. Proves the whole graph identity comes back, the
post-snapshot panel is gone, the PRIOR approval event is preserved, exactly one
``restored`` event is appended per restored panel, and an ``auto`` safety snapshot
was taken first. Real postgres via ``db_session`` (owner-bypass RLS; tenant scoping
is the repo project-join, same as the sibling tests).
"""

from __future__ import annotations

import copy
from typing import Any, cast
from uuid import UUID, uuid4, uuid7

import pytest
from sqlalchemy import select

from src.database.models import Project, StoryboardPanelAsset
from src.database.repositories.storyboard_repository import (
    StoryboardAssetRepository,
    StoryboardPanelEventRepository,
    StoryboardPanelRepository,
    StoryboardRepository,
    StoryboardSceneRepository,
    StoryboardSnapshotRepository,
)
from src.services.storyboard.storyboard_snapshot_service import (
    SnapshotNotFoundError,
    StoryboardSnapshotService,
)
from tests.support.storyboard_locks import board_lock_versions_for_snapshot

pytestmark = pytest.mark.asyncio


async def _seed_project(db_session, *, tenant_id: UUID, subject_id: UUID) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="snapshot-svc",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def test_snapshot_then_restore_roundtrips_full_graph(db_session) -> None:
    tenant_id = uuid4()
    subject_id = uuid4()
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)

    storyboard = await StoryboardRepository(db_session).ensure_for_project(
        project.id, tenant_id=tenant_id
    )
    assert storyboard is not None

    scenes = StoryboardSceneRepository(db_session)
    panels = StoryboardPanelRepository(db_session)
    assets = StoryboardAssetRepository(db_session)
    events = StoryboardPanelEventRepository(db_session)

    s1 = await scenes.create(
        project_id=project.id, storyboard_id=storyboard.id, position=0, name="scene-1"
    )
    s2 = await scenes.create(
        project_id=project.id, storyboard_id=storyboard.id, position=1, name="scene-2"
    )
    p1 = await panels.create(
        project_id=project.id,
        storyboard_id=storyboard.id,
        position=0,
        shot_key="clip-1",
        scene_id=s1.id,
        vo_text="one",
    )
    p2 = await panels.create(
        project_id=project.id,
        storyboard_id=storyboard.id,
        position=1,
        shot_key="clip-2",
        scene_id=s1.id,
        vo_text="two",
    )
    p3 = await panels.create(
        project_id=project.id,
        storyboard_id=storyboard.id,
        position=2,
        shot_key="clip-3",
        scene_id=s2.id,
        vo_text="three",
    )
    await assets.create(project_id=project.id, position=0, kind="character", name="Hero")

    # Approve p1 → sets approved_at/by AND writes an approval event.
    await panels.approve(p1.id, tenant_id=tenant_id, actor_subject_id=subject_id, events=events)
    await db_session.commit()

    svc = StoryboardSnapshotService(db_session)
    snap = await svc.create_snapshot(
        project.id, tenant_id=tenant_id, kind="manual", note="v1", actor_subject=subject_id
    )
    snapshot_id = snap.id
    await db_session.commit()

    # ── Mutate the live graph HARD ──────────────────────────────────────────
    # content edit
    await panels.update(p2.id, tenant_id=tenant_id, fields={"vo_text": "TWO-EDITED"})
    # SWAP p1/p2 shot_keys (three-step to dodge the constraint in test setup):
    await panels.update(p1.id, tenant_id=tenant_id, fields={"shot_key": "clip-tmp"})
    await panels.update(p2.id, tenant_id=tenant_id, fields={"shot_key": "clip-1"})
    await panels.update(p1.id, tenant_id=tenant_id, fields={"shot_key": "clip-2"})
    # delete a scene (p3.scene_id → NULL via ON DELETE SET NULL)
    await scenes.delete(s2.id, tenant_id=tenant_id, expected_lock_version=s2.lock_version)
    # add a brand-new panel
    p_new = await panels.create(
        project_id=project.id,
        storyboard_id=storyboard.id,
        position=3,
        shot_key="clip-4",
        vo_text="new-post-snapshot",
    )
    p_new_id = p_new.id
    await db_session.commit()

    # ── Restore ─────────────────────────────────────────────────────────────
    result = await svc.restore(
        snapshot_id,
        tenant_id=tenant_id,
        actor_subject=subject_id,
        expected_lock_versions=await board_lock_versions_for_snapshot(db_session, snapshot_id),
    )
    await db_session.commit()

    assert result["restored_snapshot_id"] == snapshot_id
    assert result["auto_snapshot_id"] != snapshot_id

    # ── Assert full graph identity restored ─────────────────────────────────
    restored_panels = await panels.list_for_project(project.id, tenant_id=tenant_id)
    by_id = {p.id: p for p in restored_panels}
    # post-snapshot panel is gone
    assert p_new_id not in by_id
    assert set(by_id) == {p1.id, p2.id, p3.id}
    # shot_keys swapped back
    assert by_id[p1.id].shot_key == "clip-1"
    assert by_id[p2.id].shot_key == "clip-2"
    assert by_id[p3.id].shot_key == "clip-3"
    # content edit reverted
    assert by_id[p2.id].vo_text == "two"
    # scene_id FK restored (scene s2 recreated)
    assert by_id[p3.id].scene_id == s2.id
    # approval state restored verbatim
    assert by_id[p1.id].status == "approved"
    assert by_id[p1.id].approved_at is not None
    assert by_id[p1.id].approved_by == str(subject_id)

    restored_scenes = await scenes.list_for_project(project.id, tenant_id=tenant_id)
    assert {s.id for s in restored_scenes} == {s1.id, s2.id}

    # ── Events: prior approval preserved + exactly one restored per panel ────
    all_events = await events.list_for_project(project.id, tenant_id=tenant_id)
    approved = [e for e in all_events if e.event_type == "approved"]
    restored = [e for e in all_events if e.event_type == "restored"]
    assert len(approved) == 1  # the original approval event survived
    assert approved[0].panel_id == p1.id
    assert len(restored) == 3  # one per restored panel
    assert {e.panel_id for e in restored} == {p1.id, p2.id, p3.id}
    assert all(e.detail.get("snapshot_id") == str(snapshot_id) for e in restored)

    # ── An auto snapshot was taken before the restore ───────────────────────
    snaps = await StoryboardSnapshotRepository(db_session).list_for_project(
        project.id, tenant_id=tenant_id
    )
    kinds = [s.kind for s in snaps]
    assert "auto" in kinds and "manual" in kinds
    assert any(s.id == result["auto_snapshot_id"] and s.kind == "auto" for s in snaps)


async def test_restore_unknown_snapshot_raises(db_session) -> None:
    svc = StoryboardSnapshotService(db_session)
    with pytest.raises(SnapshotNotFoundError):
        await svc.restore(
            uuid4(), tenant_id=uuid4(), actor_subject=uuid4(), expected_lock_versions={}
        )


# ── auto-snapshot retention (prune) ──────────────────────────────────────────
async def _seed_snapshotable(db_session, *, tenant_id: UUID, subject_id: UUID) -> Project:
    """A project with a storyboard + one panel — enough to snapshot."""
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    storyboard = await StoryboardRepository(db_session).ensure_for_project(
        project.id, tenant_id=tenant_id
    )
    assert storyboard is not None
    await StoryboardPanelRepository(db_session).create(
        project_id=project.id,
        storyboard_id=storyboard.id,
        position=0,
        shot_key="clip-1",
        vo_text="one",
    )
    await db_session.commit()
    return project


async def _mint(svc, project_id: UUID, *, tenant_id: UUID, kind: str, db_session) -> UUID:
    """Create one snapshot and return its id (read BEFORE the commit expires it)."""
    row = await svc.create_snapshot(project_id, tenant_id=tenant_id, kind=kind, note=kind)
    snapshot_id = row.id
    await db_session.commit()
    return snapshot_id


async def test_prune_keeps_newest_autos_and_never_touches_manual(db_session) -> None:
    tenant_id = uuid4()
    subject_id = uuid4()
    project = await _seed_snapshotable(db_session, tenant_id=tenant_id, subject_id=subject_id)

    svc = StoryboardSnapshotService(db_session)
    manual_id = await _mint(
        svc, project.id, tenant_id=tenant_id, kind="manual", db_session=db_session
    )
    auto_ids = [
        await _mint(svc, project.id, tenant_id=tenant_id, kind="auto", db_session=db_session)
        for _ in range(4)
    ]

    repo = StoryboardSnapshotRepository(db_session)
    deleted = await repo.prune_auto_for_project(project.id, keep=2)
    await db_session.commit()

    assert deleted == 2  # the two OLDEST autos
    remaining = await repo.list_for_project(project.id, tenant_id=tenant_id, limit=50)
    assert {s.id for s in remaining} == {manual_id, auto_ids[-1], auto_ids[-2]}
    # the manual snapshot is untouched by retention
    assert [s.kind for s in remaining].count("manual") == 1


async def test_prune_keep_zero_disables_pruning(db_session) -> None:
    tenant_id = uuid4()
    subject_id = uuid4()
    project = await _seed_snapshotable(db_session, tenant_id=tenant_id, subject_id=subject_id)

    svc = StoryboardSnapshotService(db_session)
    auto_ids = [
        await _mint(svc, project.id, tenant_id=tenant_id, kind="auto", db_session=db_session)
        for _ in range(3)
    ]

    repo = StoryboardSnapshotRepository(db_session)
    # keep=0 means DISABLED, not "delete them all".
    assert await repo.prune_auto_for_project(project.id, keep=0) == 0
    await db_session.commit()

    remaining = await repo.list_for_project(project.id, tenant_id=tenant_id, limit=50)
    assert {s.id for s in remaining} == set(auto_ids)


async def test_restore_prunes_autos_down_to_the_cap(db_session) -> None:
    tenant_id = uuid4()
    subject_id = uuid4()
    project = await _seed_snapshotable(db_session, tenant_id=tenant_id, subject_id=subject_id)

    svc = StoryboardSnapshotService(db_session, max_auto_per_project=1)
    manual_id = await _mint(
        svc, project.id, tenant_id=tenant_id, kind="manual", db_session=db_session
    )

    # Three restores → three auto snapshots minted, but the cap keeps only the newest.
    for _ in range(3):
        await svc.restore(
            manual_id,
            tenant_id=tenant_id,
            actor_subject=subject_id,
            expected_lock_versions=await board_lock_versions_for_snapshot(db_session, manual_id),
        )
        await db_session.commit()

    repo = StoryboardSnapshotRepository(db_session)
    remaining = await repo.list_for_project(project.id, tenant_id=tenant_id, limit=50)
    kinds = [s.kind for s in remaining]
    assert kinds.count("auto") == 1  # capped
    assert kinds.count("manual") == 1  # the restore source survives
    assert any(s.id == manual_id for s in remaining)
    assert await repo.count_for_project(project.id, tenant_id=tenant_id) == 2


async def test_restore_of_an_auto_snapshot_never_prunes_its_own_source(db_session) -> None:
    tenant_id = uuid4()
    subject_id = uuid4()
    project = await _seed_snapshotable(db_session, tenant_id=tenant_id, subject_id=subject_id)

    svc = StoryboardSnapshotService(db_session)
    oldest_auto = await _mint(
        svc, project.id, tenant_id=tenant_id, kind="auto", db_session=db_session
    )
    for _ in range(2):
        await _mint(svc, project.id, tenant_id=tenant_id, kind="auto", db_session=db_session)

    # Cap of 1: the prune would drop every auto but the freshly minted safety copy —
    # except the snapshot being restored FROM, which must survive the restore.
    capped = StoryboardSnapshotService(db_session, max_auto_per_project=1)
    await capped.restore(
        oldest_auto,
        tenant_id=tenant_id,
        actor_subject=subject_id,
        expected_lock_versions=await board_lock_versions_for_snapshot(db_session, oldest_auto),
    )
    await db_session.commit()

    repo = StoryboardSnapshotRepository(db_session)
    remaining = await repo.list_for_project(project.id, tenant_id=tenant_id, limit=50)
    ids = {s.id for s in remaining}
    assert oldest_auto in ids  # protected
    assert len(ids) == 2  # the source + the new safety copy


async def test_list_for_project_pagination_and_count(db_session) -> None:
    tenant_id = uuid4()
    subject_id = uuid4()
    project = await _seed_snapshotable(db_session, tenant_id=tenant_id, subject_id=subject_id)

    svc = StoryboardSnapshotService(db_session)
    ids = [
        await _mint(svc, project.id, tenant_id=tenant_id, kind="manual", db_session=db_session)
        for _ in range(3)
    ]
    newest_first = list(reversed(ids))

    repo = StoryboardSnapshotRepository(db_session)
    page = await repo.list_for_project(project.id, tenant_id=tenant_id, limit=2, offset=0)
    assert [s.id for s in page] == newest_first[:2]
    page2 = await repo.list_for_project(project.id, tenant_id=tenant_id, limit=2, offset=2)
    assert [s.id for s in page2] == newest_first[2:]
    assert await repo.count_for_project(project.id, tenant_id=tenant_id) == 3
    # cross-tenant sees neither rows nor a count
    assert await repo.count_for_project(project.id, tenant_id=uuid4()) == 0


# ─── Panel → asset refs across snapshot/restore (migration 0051) ──────────────
# The refs moved out of the panel row into ``storyboard_panel_assets``, so
# ``_row_to_dict`` (which walks ``__table__.columns``) no longer captures them with
# the panel: ``serialize`` carries them in a flat top-level ``panel_assets`` section
# instead, and ``restore`` wipes + re-inserts from it. The section's PRESENCE is the
# version flag — a payload WITHOUT the key is a pre-0051 snapshot whose panel dicts
# still carry the legacy ``characters`` / ``props`` / ``location_asset_id`` keys, and
# restore must RECONSTRUCT the join rows from those (else an old snapshot would
# restore the prose while leaving today's cast attached — a hybrid the user never saw).


async def _cast_project(db_session, *, tenant_id: UUID, subject_id: UUID) -> dict[str, Any]:
    """A project + storyboard + one panel wearing every role: characters ``[a, b]``
    (in that order), props ``[p]``, location ``loc``."""
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    storyboard = await StoryboardRepository(db_session).ensure_for_project(
        project.id, tenant_id=tenant_id
    )
    assert storyboard is not None
    assets = StoryboardAssetRepository(db_session)
    a = await assets.create(project_id=project.id, position=0, kind="character", name="A")
    b = await assets.create(project_id=project.id, position=1, kind="character", name="B")
    p = await assets.create(project_id=project.id, position=2, kind="prop", name="P")
    loc = await assets.create(project_id=project.id, position=3, kind="location", name="L")
    panel = await StoryboardPanelRepository(db_session).create(
        project_id=project.id,
        storyboard_id=storyboard.id,
        position=0,
        shot_key="clip-1",
        vo_text="one",
        characters=[a.id, b.id],
        props=[p.id],
        location_asset_id=loc.id,
    )
    ids = {
        "project": project,
        "storyboard_id": storyboard.id,
        "panel_id": panel.id,
        "a": a.id,
        "b": b.id,
        "p": p.id,
        "loc": loc.id,
    }
    await db_session.commit()
    return ids


async def _panel_refs(db_session, project_id: UUID, *, tenant_id: UUID, panel_id: UUID):
    """``(characters, props, location_asset_id)`` re-projected from the DB."""
    panels = await StoryboardPanelRepository(db_session).list_for_project(
        project_id, tenant_id=tenant_id
    )
    panel = next(p for p in panels if p.id == panel_id)
    return panel.characters, panel.props, panel.location_asset_id


async def _join_row_count(db_session, panel_id: UUID) -> int:
    rows = (
        await db_session.execute(
            select(StoryboardPanelAsset)
            .where(StoryboardPanelAsset.panel_id == panel_id)
            .execution_options(populate_existing=True)
        )
    ).scalars()
    return len(list(rows))


async def test_serialize_always_emits_panel_assets_section(db_session) -> None:
    """The key is ALWAYS present (possibly ``[]``) — its absence is what marks a
    pre-0051 payload, so an empty graph must still emit it or restore would take the
    legacy branch on a modern snapshot."""
    tenant_id, subject_id = uuid7(), uuid7()
    project = await _seed_snapshotable(db_session, tenant_id=tenant_id, subject_id=subject_id)

    payload = await StoryboardSnapshotService(db_session).serialize(project.id, tenant_id=tenant_id)
    assert "panel_assets" in payload
    assert payload["panel_assets"] == []  # a panel with no refs


async def test_snapshot_restore_roundtrips_panel_asset_refs_in_order(db_session) -> None:
    tenant_id, subject_id = uuid7(), uuid7()
    ids = await _cast_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    project, panel_id = ids["project"], ids["panel_id"]

    svc = StoryboardSnapshotService(db_session)
    snap = await svc.create_snapshot(
        project.id, tenant_id=tenant_id, kind="manual", note="cast", actor_subject=subject_id
    )
    snapshot_id = snap.id
    assert len(snap.payload["panel_assets"]) == 4  # 2 characters + 1 prop + 1 location
    await db_session.commit()

    # Mutate every role: reverse the cast, drop the prop, clear the location.
    await StoryboardPanelRepository(db_session).update(
        panel_id,
        tenant_id=tenant_id,
        fields={"characters": [ids["b"], ids["a"]], "props": [], "location_asset_id": None},
    )
    await db_session.commit()
    assert (await _panel_refs(db_session, project.id, tenant_id=tenant_id, panel_id=panel_id)) == (
        [str(ids["b"]), str(ids["a"])],
        [],
        None,
    )

    await svc.restore(
        snapshot_id,
        tenant_id=tenant_id,
        actor_subject=subject_id,
        expected_lock_versions=await board_lock_versions_for_snapshot(db_session, snapshot_id),
    )
    await db_session.commit()

    characters, props, location = await _panel_refs(
        db_session, project.id, tenant_id=tenant_id, panel_id=panel_id
    )
    assert characters == [str(ids["a"]), str(ids["b"])]  # ORDER restored, not just membership
    assert props == [str(ids["p"])]
    assert location == ids["loc"]
    assert await _join_row_count(db_session, panel_id) == 4


async def test_restore_prunes_join_rows_the_snapshot_does_not_have(db_session) -> None:
    """Snapshot with FEWER refs than live → the extra join rows are pruned (a restore
    is a rollback, not a merge)."""
    tenant_id, subject_id = uuid7(), uuid7()
    ids = await _cast_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    project, panel_id = ids["project"], ids["panel_id"]

    panels = StoryboardPanelRepository(db_session)
    # Snapshot a LEAN state: one character, nothing else.
    await panels.update(
        panel_id,
        tenant_id=tenant_id,
        fields={"characters": [ids["a"]], "props": [], "location_asset_id": None},
    )
    await db_session.commit()

    svc = StoryboardSnapshotService(db_session)
    snap = await svc.create_snapshot(project.id, tenant_id=tenant_id, kind="manual", note="lean")
    snapshot_id = snap.id
    assert len(snap.payload["panel_assets"]) == 1
    await db_session.commit()

    # Live gains refs the snapshot never had.
    await panels.update(
        panel_id,
        tenant_id=tenant_id,
        fields={
            "characters": [ids["a"], ids["b"]],
            "props": [ids["p"]],
            "location_asset_id": ids["loc"],
        },
    )
    await db_session.commit()
    assert await _join_row_count(db_session, panel_id) == 4

    await svc.restore(
        snapshot_id,
        tenant_id=tenant_id,
        actor_subject=subject_id,
        expected_lock_versions=await board_lock_versions_for_snapshot(db_session, snapshot_id),
    )
    await db_session.commit()

    characters, props, location = await _panel_refs(
        db_session, project.id, tenant_id=tenant_id, panel_id=panel_id
    )
    assert characters == [str(ids["a"])]
    assert props == []
    assert location is None
    assert await _join_row_count(db_session, panel_id) == 1  # the extras are GONE


def _to_legacy(payload: dict[str, Any], refs: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a modern payload into the PRE-0051 shape: no top-level ``panel_assets``
    key, and the three legacy keys back inside each panel dict (which is exactly what
    ``_row_to_dict`` captured when they were still columns)."""
    legacy = copy.deepcopy(payload)
    del legacy["panel_assets"]
    for panel in legacy["panels"]:
        panel["characters"] = list(refs["characters"])
        panel["props"] = list(refs["props"])
        panel["location_asset_id"] = refs["location_asset_id"]
    return legacy


async def _write_snapshot(
    db_session, project, storyboard_id: UUID, payload: dict[str, Any]
) -> UUID:
    """Persist a hand-crafted payload as a real snapshot row (bypassing ``serialize``,
    which can only ever emit the modern shape)."""
    row = await StoryboardSnapshotRepository(db_session).create(
        project_id=project.id,
        storyboard_id=storyboard_id,
        kind="manual",
        payload=payload,
        note="pre-0051",
    )
    snapshot_id = row.id
    await db_session.commit()
    return snapshot_id


async def test_restore_of_pre_0051_snapshot_reconstructs_join_rows(db_session) -> None:
    """The legacy branch: a payload with NO ``panel_assets`` key must rebuild the join
    rows from the panel dicts' legacy keys — right assets, right roles, right positions,
    no 500."""
    tenant_id, subject_id = uuid7(), uuid7()
    ids = await _cast_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    project, panel_id = ids["project"], ids["panel_id"]

    svc = StoryboardSnapshotService(db_session)
    modern = await svc.serialize(project.id, tenant_id=tenant_id)
    legacy = _to_legacy(
        modern,
        {
            "characters": [str(ids["a"]), str(ids["b"])],
            "props": [str(ids["p"])],
            "location_asset_id": str(ids["loc"]),
        },
    )
    snapshot_id = await _write_snapshot(db_session, project, ids["storyboard_id"], legacy)

    # Live drifts away from the snapshot entirely.
    await StoryboardPanelRepository(db_session).update(
        panel_id,
        tenant_id=tenant_id,
        fields={"characters": [], "props": [], "location_asset_id": None},
    )
    await db_session.commit()
    assert await _join_row_count(db_session, panel_id) == 0

    await svc.restore(
        snapshot_id,
        tenant_id=tenant_id,
        actor_subject=subject_id,
        expected_lock_versions=await board_lock_versions_for_snapshot(db_session, snapshot_id),
    )
    await db_session.commit()

    characters, props, location = await _panel_refs(
        db_session, project.id, tenant_id=tenant_id, panel_id=panel_id
    )
    assert characters == [str(ids["a"]), str(ids["b"])]  # order == legacy list order
    assert props == [str(ids["p"])]
    assert location == ids["loc"]

    rows = (
        await db_session.execute(
            select(StoryboardPanelAsset)
            .where(StoryboardPanelAsset.panel_id == panel_id)
            .execution_options(populate_existing=True)
        )
    ).scalars()
    by_role: dict[str, list[tuple[str, int]]] = {}
    for row in rows:
        by_role.setdefault(row.role, []).append((str(row.asset_id), row.position))
    assert sorted(by_role["character"], key=lambda t: t[1]) == [
        (str(ids["a"]), 0),
        (str(ids["b"]), 1),
    ]
    assert by_role["prop"] == [(str(ids["p"]), 0)]
    assert by_role["location"] == [(str(ids["loc"]), 0)]


async def test_restore_of_pre_0051_snapshot_skips_refs_absent_from_payload_assets(
    db_session,
) -> None:
    """A legacy ref pointing at an asset that is NOT in the payload's own ``assets``
    set was ALREADY dangling at capture time (legal back then — there was no FK). It
    cannot satisfy the new FK, so it is skipped, not resurrected into a 23503; the
    other refs land intact. A non-UUID-shaped legacy string (``"hero"`` — the pre-0051
    API accepted free text) is skipped the same way."""
    tenant_id, subject_id = uuid7(), uuid7()
    ids = await _cast_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    project, panel_id = ids["project"], ids["panel_id"]

    svc = StoryboardSnapshotService(db_session)
    modern = await svc.serialize(project.id, tenant_id=tenant_id)
    dangling = str(uuid7())  # never an asset of this (or any) project
    legacy = _to_legacy(
        modern,
        {
            "characters": [str(ids["a"]), dangling, "hero", str(ids["b"])],
            "props": [dangling],
            "location_asset_id": dangling,
        },
    )
    snapshot_id = await _write_snapshot(db_session, project, ids["storyboard_id"], legacy)

    await svc.restore(
        snapshot_id,
        tenant_id=tenant_id,
        actor_subject=subject_id,
        expected_lock_versions=await board_lock_versions_for_snapshot(db_session, snapshot_id),
    )
    await db_session.commit()

    characters, props, location = await _panel_refs(
        db_session, project.id, tenant_id=tenant_id, panel_id=panel_id
    )
    # The two real characters survive, contiguously positioned; the junk is dropped.
    assert characters == [str(ids["a"]), str(ids["b"])]
    assert props == []
    assert location is None
    assert await _join_row_count(db_session, panel_id) == 2


async def test_restore_of_empty_panel_assets_prunes_and_ignores_legacy_keys(db_session) -> None:
    """``panel_assets: []`` PRESENT is not the same as ABSENT: an explicitly empty
    modern section is authoritative — it prunes — even when stale legacy keys happen to
    sit in the panel dicts. Only ABSENCE triggers the reconstruction branch."""
    tenant_id, subject_id = uuid7(), uuid7()
    ids = await _cast_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    project, panel_id = ids["project"], ids["panel_id"]

    svc = StoryboardSnapshotService(db_session)
    payload = await svc.serialize(project.id, tenant_id=tenant_id)
    payload["panel_assets"] = []
    for panel in cast("list[dict[str, Any]]", payload["panels"]):
        # Legacy keys present AND non-empty — they must be ignored.
        panel["characters"] = [str(ids["a"]), str(ids["b"])]
        panel["props"] = [str(ids["p"])]
        panel["location_asset_id"] = str(ids["loc"])
    snapshot_id = await _write_snapshot(db_session, project, ids["storyboard_id"], payload)

    # Live still carries all 4 refs from _cast_project.
    assert await _join_row_count(db_session, panel_id) == 4

    await svc.restore(
        snapshot_id,
        tenant_id=tenant_id,
        actor_subject=subject_id,
        expected_lock_versions=await board_lock_versions_for_snapshot(db_session, snapshot_id),
    )
    await db_session.commit()

    characters, props, location = await _panel_refs(
        db_session, project.id, tenant_id=tenant_id, panel_id=panel_id
    )
    assert characters == []
    assert props == []
    assert location is None
    assert await _join_row_count(db_session, panel_id) == 0
