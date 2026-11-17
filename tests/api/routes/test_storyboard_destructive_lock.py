"""The DESTRUCTIVE + REORDER storyboard routes REQUIRE an optimistic-concurrency token.

Before this suite, six mutating routes took no lock at all — ``panels/reorder``,
``scenes/reorder``, ``DELETE`` panel/scene/asset, and ``snapshots/{id}/restore``. The two
reorders even BUMPED ``lock_version`` without ever COMPARING it. Every one of them was
last-write-wins, and ``restore`` is the sharp one: it rewrites scenes + panels + assets
wholesale, so a restore fired against a stale read silently discards everything a sibling
landed in the meantime.

The contract these tests pin, per route:

* **missing token → 400** with a message naming the fix (not a silent success);
* **stale token → 409** in the EXISTING conflict shape
  (``{"detail": "<entity> was modified concurrently"}``, from the app-level
  ``StoryboardVersionConflictError`` handler — no new error shape was invented);
* and — the part a bare 400-on-missing-field test does NOT prove — **the concurrent
  write SURVIVES**: after the 409 the board still holds the sibling's state, not the
  stale caller's. Every conflict case below asserts the data, not just the status code.

Two clients are NOT needed: "concurrent" here means "a write landed between the caller's
read and its write", which one client reproduces exactly by capturing the token map,
writing, and then replaying the captured map.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from src.database.models import Project

pytestmark = pytest.mark.asyncio


# ─── helpers ─────────────────────────────────────────────────────────────
async def _seed_project(db_session, *, tenant_id: UUID, subject_id: UUID) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="destructive-lock",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def _panel(client, project_id, *, vo_text: str) -> dict[str, Any]:
    r = await client.post(
        f"/api/projects/{project_id}/storyboard/panels", json={"vo_text": vo_text}
    )
    assert r.status_code == 201, r.text
    return r.json()


async def _scene(client, project_id, *, title: str) -> dict[str, Any]:
    r = await client.post(f"/api/projects/{project_id}/storyboard/scenes", json={"name": title})
    assert r.status_code == 201, r.text
    return r.json()


async def _asset(client, project_id, *, name: str) -> dict[str, Any]:
    r = await client.post(
        f"/api/projects/{project_id}/storyboard/assets", json={"kind": "prop", "name": name}
    )
    assert r.status_code == 201, r.text
    return r.json()


async def _panels(client, project_id) -> list[dict[str, Any]]:
    r = await client.get(f"/api/projects/{project_id}/storyboard/panels")
    assert r.status_code == 200, r.text
    return r.json()["items"]


async def _scenes(client, project_id) -> list[dict[str, Any]]:
    r = await client.get(f"/api/projects/{project_id}/storyboard/scenes")
    assert r.status_code == 200, r.text
    return r.json()["items"]


def _lock_map(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {row["id"]: row["lock_version"] for row in rows}


async def _board_lock_map(client, project_id) -> dict[str, int]:
    """The flat ``{row id: lock_version}`` map ``restore`` requires: scenes + panels +
    assets in ONE namespace (uuid7 ids are unique across the three tables)."""
    merged = _lock_map(await _panels(client, project_id))
    merged.update(_lock_map(await _scenes(client, project_id)))
    r = await client.get(f"/api/projects/{project_id}/storyboard/assets")
    assert r.status_code == 200, r.text
    merged.update(_lock_map(r.json()["items"]))
    return merged


# ─── panels/reorder ──────────────────────────────────────────────────────
async def test_reorder_panels_without_lock_map_is_400(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    a = await _panel(client, project.id, vo_text="a")
    b = await _panel(client, project.id, vo_text="b")

    r = await client.patch(
        f"/api/projects/{project.id}/storyboard/panels/reorder",
        json={"ordered_panel_ids": [b["id"], a["id"]]},
    )
    assert r.status_code == 400, r.text
    assert "expected_lock_versions" in r.json()["detail"]
    # The order is UNCHANGED — a rejected reorder must not half-apply.
    assert [p["id"] for p in await _panels(client, project.id)] == [a["id"], b["id"]]


async def test_reorder_panels_with_stale_map_loses_to_the_concurrent_write(
    client, db_session
) -> None:
    """The data-loss case, not just the missing-field case.

    Caller A reads the board, B reorders it, then A replays its stale map. A must 409 AND
    B's order must still be the live order. Under the old last-write-wins code A's ordering
    silently won and B's reorder was gone with no error anywhere.
    """
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    a = await _panel(client, project.id, vo_text="a")
    b = await _panel(client, project.id, vo_text="b")
    c = await _panel(client, project.id, vo_text="c")

    stale = _lock_map(await _panels(client, project.id))  # A's read

    # B's write: reverse the board.
    b_order = [c["id"], b["id"], a["id"]]
    r = await client.patch(
        f"/api/projects/{project.id}/storyboard/panels/reorder",
        json={"ordered_panel_ids": b_order, "expected_lock_versions": stale},
    )
    assert r.status_code == 200, r.text

    # A replays its pre-B map with a DIFFERENT ordering.
    r = await client.patch(
        f"/api/projects/{project.id}/storyboard/panels/reorder",
        json={
            "ordered_panel_ids": [b["id"], a["id"], c["id"]],
            "expected_lock_versions": stale,
        },
    )
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == "storyboard was modified concurrently"
    # THE proof: B's order survived.
    assert [p["id"] for p in await _panels(client, project.id)] == b_order


async def test_reorder_panels_409s_when_a_sibling_edited_a_panel(client, db_session) -> None:
    """A plain PATCH bumps the edited panel's token, so a reorder computed before that
    edit is stale too — the reorder response is how the FE refreshes its tokens, and
    accepting a stale map would hand back stale tokens as if they were fresh."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    a = await _panel(client, project.id, vo_text="a")
    b = await _panel(client, project.id, vo_text="b")
    stale = _lock_map(await _panels(client, project.id))

    r = await client.patch(f"/api/storyboard/panels/{a['id']}", json={"vo_text": "edited by B"})
    assert r.status_code == 200, r.text

    r = await client.patch(
        f"/api/projects/{project.id}/storyboard/panels/reorder",
        json={"ordered_panel_ids": [b["id"], a["id"]], "expected_lock_versions": stale},
    )
    assert r.status_code == 409, r.text
    rows = await _panels(client, project.id)
    assert [p["id"] for p in rows] == [a["id"], b["id"]]  # not reordered
    assert next(p for p in rows if p["id"] == a["id"])["vo_text"] == "edited by B"  # not lost


async def test_reorder_panels_with_fresh_map_succeeds(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    a = await _panel(client, project.id, vo_text="a")
    b = await _panel(client, project.id, vo_text="b")

    r = await client.patch(
        f"/api/projects/{project.id}/storyboard/panels/reorder",
        json={
            "ordered_panel_ids": [b["id"], a["id"]],
            "expected_lock_versions": _lock_map(await _panels(client, project.id)),
        },
    )
    assert r.status_code == 200, r.text
    assert [p["id"] for p in r.json()] == [b["id"], a["id"]]


async def test_reorder_panels_non_permutation_is_still_400(client, db_session) -> None:
    """A CURRENT map with a malformed ordering keeps its 400 — the board-CAS runs first
    but must not swallow the permutation contract."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    a = await _panel(client, project.id, vo_text="a")
    await _panel(client, project.id, vo_text="b")

    r = await client.patch(
        f"/api/projects/{project.id}/storyboard/panels/reorder",
        json={
            "ordered_panel_ids": [a["id"]],
            "expected_lock_versions": _lock_map(await _panels(client, project.id)),
        },
    )
    assert r.status_code == 400, r.text


# ─── scenes/reorder ──────────────────────────────────────────────────────
async def test_reorder_scenes_without_lock_map_is_400(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    a = await _scene(client, project.id, title="a")
    b = await _scene(client, project.id, title="b")

    r = await client.patch(
        f"/api/projects/{project.id}/storyboard/scenes/reorder",
        json={"ordered_scene_ids": [b["id"], a["id"]]},
    )
    assert r.status_code == 400, r.text
    assert "expected_lock_versions" in r.json()["detail"]
    assert [s["id"] for s in await _scenes(client, project.id)] == [a["id"], b["id"]]


async def test_reorder_scenes_with_stale_map_loses_to_the_concurrent_write(
    client, db_session
) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    a = await _scene(client, project.id, title="a")
    b = await _scene(client, project.id, title="b")
    c = await _scene(client, project.id, title="c")

    stale = _lock_map(await _scenes(client, project.id))
    b_order = [c["id"], b["id"], a["id"]]
    r = await client.patch(
        f"/api/projects/{project.id}/storyboard/scenes/reorder",
        json={"ordered_scene_ids": b_order, "expected_lock_versions": stale},
    )
    assert r.status_code == 200, r.text

    r = await client.patch(
        f"/api/projects/{project.id}/storyboard/scenes/reorder",
        json={"ordered_scene_ids": [b["id"], a["id"], c["id"]], "expected_lock_versions": stale},
    )
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == "storyboard was modified concurrently"
    assert [s["id"] for s in await _scenes(client, project.id)] == b_order


# ─── DELETE panel / scene / asset ────────────────────────────────────────
async def test_delete_panel_without_lock_is_400_and_the_panel_survives(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _panel(client, project.id, vo_text="keep me")

    r = await client.delete(f"/api/storyboard/panels/{panel['id']}")
    assert r.status_code == 400, r.text
    assert "expected_lock_version" in r.json()["detail"]
    assert [p["id"] for p in await _panels(client, project.id)] == [panel["id"]]


async def test_delete_panel_with_stale_lock_409s_and_keeps_the_concurrent_edit(
    client, db_session
) -> None:
    """A delete cannot be merged, so an unguarded one destroys a sibling's edit outright."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _panel(client, project.id, vo_text="original")
    stale = panel["lock_version"]

    r = await client.patch(f"/api/storyboard/panels/{panel['id']}", json={"vo_text": "edited by B"})
    assert r.status_code == 200, r.text

    r = await client.delete(f"/api/storyboard/panels/{panel['id']}?expected_lock_version={stale}")
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == "panel was modified concurrently"
    rows = await _panels(client, project.id)
    assert [p["vo_text"] for p in rows] == ["edited by B"]


async def test_delete_panel_with_fresh_lock_succeeds(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _panel(client, project.id, vo_text="bye")
    r = await client.delete(
        f"/api/storyboard/panels/{panel['id']}?expected_lock_version={panel['lock_version']}"
    )
    assert r.status_code == 204, r.text
    assert await _panels(client, project.id) == []


async def test_delete_scene_requires_the_lock(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    scene = await _scene(client, project.id, title="original")

    r = await client.delete(f"/api/storyboard/scenes/{scene['id']}")
    assert r.status_code == 400, r.text
    assert "expected_lock_version" in r.json()["detail"]

    r = await client.patch(f"/api/storyboard/scenes/{scene['id']}", json={"name": "edited by B"})
    assert r.status_code == 200, r.text

    r = await client.delete(
        f"/api/storyboard/scenes/{scene['id']}?expected_lock_version={scene['lock_version']}"
    )
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == "scene was modified concurrently"
    survivors = await _scenes(client, project.id)
    assert [s["name"] for s in survivors] == ["edited by B"]

    fresh = survivors[0]["lock_version"]
    r = await client.delete(f"/api/storyboard/scenes/{scene['id']}?expected_lock_version={fresh}")
    assert r.status_code == 204, r.text


async def test_delete_asset_requires_the_lock(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    asset = await _asset(client, project.id, name="original")

    r = await client.delete(f"/api/storyboard/assets/{asset['id']}")
    assert r.status_code == 400, r.text
    assert "expected_lock_version" in r.json()["detail"]

    r = await client.patch(f"/api/storyboard/assets/{asset['id']}", json={"name": "edited by B"})
    assert r.status_code == 200, r.text

    r = await client.delete(
        f"/api/storyboard/assets/{asset['id']}?expected_lock_version={asset['lock_version']}"
    )
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == "asset was modified concurrently"
    listed = await client.get(f"/api/projects/{project.id}/storyboard/assets")
    assert [a["name"] for a in listed.json()["items"]] == ["edited by B"]


# ─── snapshots/{id}/restore — the sharp one ──────────────────────────────
async def test_restore_without_lock_map_is_400_and_mints_no_auto_snapshot(
    client, db_session
) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _panel(client, project.id, vo_text="original")
    snap = await client.post(f"/api/projects/{project.id}/storyboard/snapshots", json={})
    assert snap.status_code == 201, snap.text
    snapshot_id = snap.json()["id"]

    r = await client.post(f"/api/storyboard/snapshots/{snapshot_id}/restore")
    assert r.status_code == 400, r.text
    assert "expected_lock_versions" in r.json()["detail"]
    # A refused restore writes NOTHING — not even the "auto before restore" safety blob.
    listed = await client.get(f"/api/projects/{project.id}/storyboard/snapshots")
    assert [s["kind"] for s in listed.json()["items"]] == ["manual"]


async def test_restore_with_stale_map_loses_and_the_sibling_edit_survives(
    client, db_session
) -> None:
    """THE silent-data-loss case this whole change exists for.

    A snapshots the board, B edits a panel and adds another, then A restores against its
    pre-B map. Old behaviour: the restore succeeded and B's edit AND B's new panel were
    gone — no 409, no trace. New behaviour: 409, and B's work is still there.
    """
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _panel(client, project.id, vo_text="original")
    snap = await client.post(f"/api/projects/{project.id}/storyboard/snapshots", json={})
    assert snap.status_code == 201, snap.text
    snapshot_id = snap.json()["id"]

    stale = await _board_lock_map(client, project.id)  # A's read

    # B's work, both flavours: an edit to an existing row AND a brand-new row.
    r = await client.patch(f"/api/storyboard/panels/{panel['id']}", json={"vo_text": "edited by B"})
    assert r.status_code == 200, r.text
    added = await _panel(client, project.id, vo_text="added by B")

    r = await client.post(
        f"/api/storyboard/snapshots/{snapshot_id}/restore",
        json={"expected_lock_versions": stale},
    )
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == "storyboard was modified concurrently"

    rows = {p["id"]: p for p in await _panels(client, project.id)}
    assert rows[panel["id"]]["vo_text"] == "edited by B"  # edit NOT rolled back
    assert added["id"] in rows  # new panel NOT deleted
    listed = await client.get(f"/api/projects/{project.id}/storyboard/snapshots")
    assert [s["kind"] for s in listed.json()["items"]] == ["manual"]  # no auto minted


async def test_restore_with_fresh_map_succeeds(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _panel(client, project.id, vo_text="original")
    snap = await client.post(f"/api/projects/{project.id}/storyboard/snapshots", json={})
    assert snap.status_code == 201, snap.text

    r = await client.patch(f"/api/storyboard/panels/{panel['id']}", json={"vo_text": "later"})
    assert r.status_code == 200, r.text

    r = await client.post(
        f"/api/storyboard/snapshots/{snap.json()['id']}/restore",
        json={"expected_lock_versions": await _board_lock_map(client, project.id)},
    )
    assert r.status_code == 200, r.text
    rows = await _panels(client, project.id)
    assert [p["vo_text"] for p in rows] == ["original"]


async def test_restore_of_a_second_concurrent_restore_conflicts(client, db_session) -> None:
    """Two siblings restoring different snapshots: the second must 409 against its
    pre-first map rather than silently discarding the first restore."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _panel(client, project.id, vo_text="v1")
    first = await client.post(f"/api/projects/{project.id}/storyboard/snapshots", json={})
    assert first.status_code == 201, first.text

    r = await client.patch(f"/api/storyboard/panels/{panel['id']}", json={"vo_text": "v2"})
    assert r.status_code == 200, r.text
    second = await client.post(f"/api/projects/{project.id}/storyboard/snapshots", json={})
    assert second.status_code == 201, second.text

    shared = await _board_lock_map(client, project.id)  # both siblings read here

    r = await client.post(
        f"/api/storyboard/snapshots/{first.json()['id']}/restore",
        json={"expected_lock_versions": shared},
    )
    assert r.status_code == 200, r.text

    r = await client.post(
        f"/api/storyboard/snapshots/{second.json()['id']}/restore",
        json={"expected_lock_versions": shared},
    )
    assert r.status_code == 409, r.text
    assert [p["vo_text"] for p in await _panels(client, project.id)] == ["v1"]
