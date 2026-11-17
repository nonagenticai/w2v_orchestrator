"""DB-backed integration tests for storyboard CRUD (Phase 0).

Covers the project-scoped panel/scene/asset surfaces mounted by
``src/api/routes/storyboard.py``:

* ``/api/projects/{project_id}/storyboard/{panels,scenes,assets}`` (list, create)
* ``/api/projects/{project_id}/storyboard/panels/reorder`` (permutation-exact)
* ``/api/storyboard/{panels,scenes,assets}/{item_id}`` (patch, delete)

Tenant scoping is enforced via ``<table>.project_id -> Project.tenant_id``;
cross-tenant reads return 404 (item routes) or 404 (list route via the
``_require_project`` ownership gate). Mirrors ``test_verbatim_crud.py``.
"""

from __future__ import annotations

from uuid import UUID, uuid4, uuid7

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from src.database.models import Project, StoryboardPanelAsset
from src.database.repositories.storyboard_repository import StoryboardAssetRepository

pytestmark = pytest.mark.asyncio


async def _seed_project(db_session, *, tenant_id: UUID, subject_id: UUID) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="storyboard-crud",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


# ─── Panels ──────────────────────────────────────────────────────────────
async def test_panel_create_list_patch_delete(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    # ``location_asset_id`` is a REAL FK into ``storyboard_assets`` since migration
    # 0051 (it was a bare, unvalidated UUID column before), so the id must name an
    # asset of THIS project or the create is a 422. Everything asserted below —
    # including the ``location_asset_id`` passthrough — is unchanged.
    loc_id = UUID(await _seed_asset(client, project.id))
    a = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={
            "vo_text": "hello world",
            "shot_size": "MS",
            "location_asset_id": str(loc_id),
            "duration_seconds": 3.5,
        },
    )
    b = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"shot_key": "clip-9", "action_description": "second"},
    )
    assert a.status_code == 201 and b.status_code == 201, (a.text, b.text)

    a_body = a.json()
    # Repo assigns clip-<position+1> when shot_key is omitted (position 0 → clip-1).
    assert a_body["shot_key"] == "clip-1"
    assert a_body["position"] == 0
    assert a_body["status"] == "draft"
    assert a_body["revision"] == 1
    assert a_body["vo_text"] == "hello world"
    assert a_body["shot_size"] == "MS"
    assert a_body["duration_seconds"] == 3.5
    # snake_case JSON keys + column-name passthrough (NOT camelCase / NOT `location`).
    assert a_body["project_id"] == str(project.id)
    assert a_body["location_asset_id"] == str(loc_id)
    # JSONB columns come back with their DB defaults.
    assert a_body["media_ref"] == {}
    assert a_body["gen"] == {}
    assert a_body["characters"] == []
    assert a_body["props"] == []

    b_body = b.json()
    assert b_body["shot_key"] == "clip-9"
    assert b_body["position"] == 1

    listing = await client.get(f"/api/projects/{project.id}/storyboard/panels")
    assert listing.status_code == 200
    body = listing.json()
    # SPA envelope contract: {items, total}.
    assert body["total"] == 2
    items = body["items"]
    assert [i["position"] for i in items] == [0, 1]
    assert [i["shot_key"] for i in items] == ["clip-1", "clip-9"]

    panel_id = a_body["id"]
    patched = await client.patch(
        f"/api/storyboard/panels/{panel_id}",
        json={"vo_text": "revised", "shot_size": "CU"},
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["vo_text"] == "revised"
    assert patched.json()["shot_size"] == "CU"

    # DELETE now REQUIRES the lock token (400 without it, 409 when stale — see
    # test_storyboard_destructive_lock.py); the PATCH above handed us the fresh one.
    lv = patched.json()["lock_version"]
    deleted = await client.delete(f"/api/storyboard/panels/{panel_id}?expected_lock_version={lv}")
    assert deleted.status_code == 204
    again = await client.delete(f"/api/storyboard/panels/{panel_id}?expected_lock_version={lv}")
    assert again.status_code == 404


async def test_panel_create_duplicate_shot_key_409(client, db_session) -> None:
    # uq_storyboard_panels_project_shot (migration 0048) → clean 409, not a leaked 500.
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    first = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"shot_key": "clip-1"},
    )
    assert first.status_code == 201, first.text

    dup = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"shot_key": "clip-1"},
    )
    assert dup.status_code == 409, dup.text
    assert "clip-1" in dup.json()["detail"]


async def test_panel_create_duplicate_auto_shot_key_409(client, db_session) -> None:
    # When shot_key is omitted the repo auto-assigns clip-<position+1>; forcing the
    # same position twice collides on the same auto key. The 409 must not fabricate a
    # (possibly wrong) key in its message — it reports the generic auto-assigned case.
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    first = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"position": 0},
    )
    assert first.status_code == 201, first.text

    dup = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"position": 0},
    )
    assert dup.status_code == 409, dup.text
    assert "shot_key" in dup.json()["detail"]


async def test_panel_reorder_happy_path(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    ids: list[str] = []
    for n in range(3):
        r = await client.post(
            f"/api/projects/{project.id}/storyboard/panels",
            json={"vo_text": f"line {n}"},
        )
        assert r.status_code == 201, r.text
        ids.append(r.json()["id"])

    reordered = list(reversed(ids))
    resp = await client.patch(
        f"/api/projects/{project.id}/storyboard/panels/reorder",
        json={
            "ordered_panel_ids": reordered,
            "expected_lock_versions": dict.fromkeys(ids, 0),
        },
    )
    assert resp.status_code == 200, resp.text
    # The reorder response echoes the panels in their NEW order with fresh lock_version
    # tokens, so the client can refresh its cache without a follow-up GET.
    assert [p["id"] for p in resp.json()] == reordered
    assert [p["position"] for p in resp.json()] == [0, 1, 2]

    listing = await client.get(f"/api/projects/{project.id}/storyboard/panels")
    assert [i["id"] for i in listing.json()["items"]] == reordered
    assert [i["position"] for i in listing.json()["items"]] == [0, 1, 2]


async def test_panel_reorder_permutation_400(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    ids: list[str] = []
    for n in range(3):
        r = await client.post(
            f"/api/projects/{project.id}/storyboard/panels",
            json={"vo_text": f"line {n}"},
        )
        ids.append(r.json()["id"])

    url = f"/api/projects/{project.id}/storyboard/panels/reorder"

    # Missing id → not a permutation → 400.
    lv = dict.fromkeys(ids, 0)  # a CURRENT map, so the 400 under test is the permutation one
    missing = await client.patch(
        url, json={"ordered_panel_ids": ids[:2], "expected_lock_versions": lv}
    )
    assert missing.status_code == 400, missing.text

    # Extra (unknown) id → 400.
    extra = await client.patch(
        url, json={"ordered_panel_ids": [*ids, str(uuid4())], "expected_lock_versions": lv}
    )
    assert extra.status_code == 400, extra.text

    # Duplicate id → 400.
    dupe = await client.patch(
        url, json={"ordered_panel_ids": [ids[0], ids[0], ids[1]], "expected_lock_versions": lv}
    )
    assert dupe.status_code == 400, dupe.text


async def test_list_panels_unknown_project_returns_404(client) -> None:
    r = await client.get(f"/api/projects/{uuid4()}/storyboard/panels")
    assert r.status_code == 404


# ─── Scenes ──────────────────────────────────────────────────────────────
async def test_scene_crud_smoke(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    created = await client.post(
        f"/api/projects/{project.id}/storyboard/scenes",
        json={"name": "Opening", "int_ext": "INT", "location": "Office", "time_of_day": "day"},
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["name"] == "Opening"
    assert body["int_ext"] == "INT"
    assert body["time_of_day"] == "day"
    assert body["position"] == 0
    assert body["project_id"] == str(project.id)

    listing = await client.get(f"/api/projects/{project.id}/storyboard/scenes")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1

    scene_id = body["id"]
    patched = await client.patch(f"/api/storyboard/scenes/{scene_id}", json={"name": "Renamed"})
    assert patched.status_code == 200, patched.text
    assert patched.json()["name"] == "Renamed"

    lv = patched.json()["lock_version"]
    deleted = await client.delete(f"/api/storyboard/scenes/{scene_id}?expected_lock_version={lv}")
    assert deleted.status_code == 204
    assert (
        await client.delete(f"/api/storyboard/scenes/{scene_id}?expected_lock_version={lv}")
    ).status_code == 404


async def _seed_scenes(client, project_id, n: int) -> list[str]:
    ids: list[str] = []
    for i in range(n):
        r = await client.post(
            f"/api/projects/{project_id}/storyboard/scenes",
            json={"name": f"Scene {i}"},
        )
        assert r.status_code == 201, r.text
        ids.append(r.json()["id"])
    return ids


async def test_scene_reorder_happy_path(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    ids = await _seed_scenes(client, project.id, 3)

    reordered = list(reversed(ids))
    resp = await client.patch(
        f"/api/projects/{project.id}/storyboard/scenes/reorder",
        json={"ordered_scene_ids": reordered, "expected_lock_versions": dict.fromkeys(ids, 0)},
    )
    assert resp.status_code == 200, resp.text
    assert [s["id"] for s in resp.json()] == reordered
    assert [s["position"] for s in resp.json()] == [0, 1, 2]

    listing = await client.get(f"/api/projects/{project.id}/storyboard/scenes")
    assert [i["id"] for i in listing.json()["items"]] == reordered
    assert [i["position"] for i in listing.json()["items"]] == [0, 1, 2]


async def test_scene_reorder_permutation_400_leaves_order_untouched(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    ids = await _seed_scenes(client, project.id, 3)
    url = f"/api/projects/{project.id}/storyboard/scenes/reorder"

    # Missing id → not a permutation → 400.
    lv = dict.fromkeys(ids, 0)  # CURRENT map → the 400 under test is the permutation one
    assert (
        await client.patch(url, json={"ordered_scene_ids": ids[:2], "expected_lock_versions": lv})
    ).status_code == 400
    # Extra (unknown) id → 400.
    assert (
        await client.patch(
            url, json={"ordered_scene_ids": [*ids, str(uuid4())], "expected_lock_versions": lv}
        )
    ).status_code == 400
    # Duplicate id → 400.
    assert (
        await client.patch(
            url,
            json={"ordered_scene_ids": [ids[0], ids[0], ids[1]], "expected_lock_versions": lv},
        )
    ).status_code == 400

    # The permutation check runs BEFORE any position mutation, so a rejected reorder
    # leaves the original order intact — no partial write.
    listing = await client.get(f"/api/projects/{project.id}/storyboard/scenes")
    assert [i["id"] for i in listing.json()["items"]] == ids


async def test_scene_reorder_unknown_project_returns_404(client) -> None:
    resp = await client.patch(
        f"/api/projects/{uuid4()}/storyboard/scenes/reorder",
        json={"ordered_scene_ids": [str(uuid4())], "expected_lock_versions": {}},
    )
    assert resp.status_code == 404, resp.text


# ─── Assets ──────────────────────────────────────────────────────────────
async def test_asset_crud_smoke(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    created = await client.post(
        f"/api/projects/{project.id}/storyboard/assets",
        json={"kind": "character", "name": "Hero", "canonical_descriptor": "tall, brave"},
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["kind"] == "character"
    assert body["name"] == "Hero"
    assert body["position"] == 0
    assert body["project_id"] == str(project.id)
    assert body["reference_images"] == []
    assert body["attributes"] == {}

    listing = await client.get(f"/api/projects/{project.id}/storyboard/assets")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1

    asset_id = body["id"]
    patched = await client.patch(
        f"/api/storyboard/assets/{asset_id}", json={"canonical_descriptor": "short, wise"}
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["canonical_descriptor"] == "short, wise"

    lv = patched.json()["lock_version"]
    deleted = await client.delete(f"/api/storyboard/assets/{asset_id}?expected_lock_version={lv}")
    assert deleted.status_code == 204
    assert (
        await client.delete(f"/api/storyboard/assets/{asset_id}?expected_lock_version={lv}")
    ).status_code == 404


# ─── Cross-tenant isolation ──────────────────────────────────────────────
async def test_cross_tenant_panel_isolation(
    client, other_tenant_client: AsyncClient, db_session
) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    created = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"vo_text": "secret"},
    )
    assert created.status_code == 201
    panel_id = created.json()["id"]

    # Cross-tenant list of someone else's project: 404 from the project-scope gate.
    bleed_list = await other_tenant_client.get(f"/api/projects/{project.id}/storyboard/panels")
    assert bleed_list.status_code == 404

    # Cross-tenant patch / delete: 404 (row exists, tenant scoping hides it).
    bleed_patch = await other_tenant_client.patch(
        f"/api/storyboard/panels/{panel_id}", json={"vo_text": "stolen"}
    )
    assert bleed_patch.status_code == 404
    bleed_del = await other_tenant_client.delete(
        f"/api/storyboard/panels/{panel_id}?expected_lock_version=0"
    )
    assert bleed_del.status_code == 404

    # Original tenant still sees the unmodified panel.
    mine = await client.get(f"/api/projects/{project.id}/storyboard/panels")
    assert mine.json()["items"][0]["vo_text"] == "secret"


# ─── Explicit-null PATCH on NOT-NULL-backed fields → 422 (was 500) ─────────
# Snake_case JSON keys per the existing tests (see the line-77 comment above:
# "snake_case JSON keys ... NOT camelCase"). An explicit ``null`` for a
# NOT-NULL column must be a pydantic 422, never a DB IntegrityError → 500.
async def _seed_panel(client, project_id) -> str:
    r = await client.post(
        f"/api/projects/{project_id}/storyboard/panels",
        json={"vo_text": "x"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def _seed_asset(client, project_id, *, canonical_descriptor: str | None = None) -> str:
    body = {"kind": "character", "name": "Hero"}
    if canonical_descriptor is not None:
        body["canonical_descriptor"] = canonical_descriptor
    r = await client.post(f"/api/projects/{project_id}/storyboard/assets", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def _seed_scene(client, project_id) -> str:
    r = await client.post(
        f"/api/projects/{project_id}/storyboard/scenes",
        json={"name": "Opening"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


@pytest.mark.parametrize(
    "field",
    ["characters"],
)
async def test_panel_patch_explicit_null_not_null_field_returns_422(
    client, db_session, field
) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    panel_id = await _seed_panel(client, project.id)
    resp = await client.patch(f"/api/storyboard/panels/{panel_id}", json={field: None})
    assert resp.status_code == 422, resp.text


@pytest.mark.parametrize(
    "field",
    ["reference_images"],
)
async def test_asset_patch_explicit_null_not_null_field_returns_422(
    client, db_session, field
) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    asset_id = await _seed_asset(client, project.id)
    resp = await client.patch(f"/api/storyboard/assets/{asset_id}", json={field: None})
    assert resp.status_code == 422, resp.text


async def test_scene_patch_explicit_null_position_returns_422(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    scene_id = await _seed_scene(client, project.id)
    resp = await client.patch(f"/api/storyboard/scenes/{scene_id}", json={"position": None})
    assert resp.status_code == 422, resp.text


async def test_asset_patch_explicit_null_canonical_descriptor_clears_200(
    client, db_session
) -> None:
    """CONTROL: canonical_descriptor is a nullable column → null clears it, 200."""
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    asset_id = await _seed_asset(client, project.id, canonical_descriptor="tall, brave")
    resp = await client.patch(
        f"/api/storyboard/assets/{asset_id}", json={"canonical_descriptor": None}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["canonical_descriptor"] is None


async def test_asset_patch_omitting_not_null_fields_is_noop_200(client, db_session) -> None:
    """CONTROL: a PATCH that OMITS the NOT-NULL fields still succeeds (no-op)."""
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    asset_id = await _seed_asset(client, project.id)
    resp = await client.patch(f"/api/storyboard/assets/{asset_id}", json={"name": "renamed"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "renamed"
    # Omitted NOT-NULL fields keep their DB defaults, untouched.
    assert body["reference_images"] == []
    assert body["attributes"] == {}


# ─── Positive controls: the null-guard must NOT over-fire on legit values ──
# ``characters`` used to accept arbitrary strings (``["hero"]``); since migration
# 0051 every entry is a real FK into ``storyboard_assets``, so the case cannot name
# its id at parametrize time — it carries this sentinel, and the test body swaps in a
# freshly seeded asset id. The null-guard behaviour under test is unchanged.
_ASSET = "__seeded_asset__"


def _sub(value, asset_id: str):
    """Replace the ``_ASSET`` sentinel with a real id (returns new objects — the
    parametrize payloads are shared across cases and must never be mutated)."""
    if isinstance(value, dict):
        return {k: _sub(v, asset_id) for k, v in value.items()}
    if isinstance(value, list):
        return [_sub(v, asset_id) for v in value]
    return asset_id if value == _ASSET else value


@pytest.mark.parametrize(
    ("payload", "field", "expected"),
    [({"position": 3}, "position", 3)],
)
async def test_panel_patch_valid_not_null_value_persists_200(
    client, db_session, payload, field, expected
) -> None:
    """A non-null value on a guarded field patches through (200) and persists."""
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    panel_id = await _seed_panel(client, project.id)
    asset_id = await _seed_asset(client, project.id)
    payload = _sub(payload, asset_id)
    expected = _sub(expected, asset_id)
    resp = await client.patch(f"/api/storyboard/panels/{panel_id}", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()[field] == expected
    # Re-GET confirms persistence (not just an echoed response).
    listing = await client.get(f"/api/projects/{project.id}/storyboard/panels")
    row = next(i for i in listing.json()["items"] if i["id"] == panel_id)
    assert row[field] == expected


@pytest.mark.parametrize(
    ("payload", "field", "expected"),
    [({"name": "Hero"}, "name", "Hero")],
)
async def test_asset_patch_valid_not_null_value_persists_200(
    client, db_session, payload, field, expected
) -> None:
    """A non-null value on a guarded asset field patches through (200) and persists."""
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    asset_id = await _seed_asset(client, project.id)
    resp = await client.patch(f"/api/storyboard/assets/{asset_id}", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()[field] == expected
    listing = await client.get(f"/api/projects/{project.id}/storyboard/assets")
    row = next(i for i in listing.json()["items"] if i["id"] == asset_id)
    assert row[field] == expected


async def test_panel_patch_explicit_null_422_body_names_offending_field(client, db_session) -> None:
    """The 422 must be raised FOR the guarded field (loc names it), not an
    unrelated validation error that would pass a bare status-code check."""
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    panel_id = await _seed_panel(client, project.id)
    resp = await client.patch(f"/api/storyboard/panels/{panel_id}", json={"characters": None})
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    # FastAPI RequestValidationError loc for a body field is ["body", "<field>"];
    # snake_case JSON key → loc tail is the snake_case field name.
    assert any(err["loc"][-1] == "characters" for err in detail), detail


# ─── Panel → asset references (migration 0051, storyboard_panel_assets) ───────
# The WIRE contract is unchanged — ``characters: list[str]`` / ``props: list[str]`` /
# ``location_asset_id: UUID|None`` on create, patch and response — but they are now
# PROJECTIONS over the ``storyboard_panel_assets`` join table, not columns. These
# tests pin the projection (order, partial-PATCH isolation), the 422 on an unknown /
# foreign-project id (the cross-tenant guard the old undeclared FK never had), and
# the 409 on deleting a referenced asset.
async def _named_asset(client, project_id, *, name: str, kind: str = "character") -> str:
    r = await client.post(
        f"/api/projects/{project_id}/storyboard/assets", json={"kind": kind, "name": name}
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def _join_rows(db_session, panel_id: str) -> list[StoryboardPanelAsset]:
    """The panel's join rows, straight from postgres (``populate_existing`` so the
    session's identity map can't serve a stale copy of a row the API just rewrote)."""
    rows = (
        await db_session.execute(
            select(StoryboardPanelAsset)
            .where(StoryboardPanelAsset.panel_id == UUID(panel_id))
            .execution_options(populate_existing=True)
        )
    ).scalars()
    return list(rows)


async def _panel_by_id(client, project_id, panel_id: str) -> dict:
    listing = await client.get(f"/api/projects/{project_id}/storyboard/panels")
    assert listing.status_code == 200, listing.text
    return next(i for i in listing.json()["items"] if i["id"] == panel_id)


async def test_panel_asset_refs_roundtrip_projection(client, db_session) -> None:
    """POST with characters/props/location → 201 echoes them, and a fresh GET
    re-projects the SAME values out of the join table (not an echoed request body)."""
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    a = await _named_asset(client, project.id, name="A")
    b = await _named_asset(client, project.id, name="B")
    p = await _named_asset(client, project.id, name="Prop", kind="prop")
    loc = await _named_asset(client, project.id, name="Loc", kind="location")

    created = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={
            "vo_text": "cast me",
            "characters": [a, b],
            "props": [p],
            "location_asset_id": loc,
        },
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["characters"] == [a, b]
    assert body["props"] == [p]
    assert body["location_asset_id"] == loc

    row = await _panel_by_id(client, project.id, body["id"])
    assert row["characters"] == [a, b]
    assert row["props"] == [p]
    assert row["location_asset_id"] == loc

    # 4 join rows: 2 characters + 1 prop + 1 location.
    assert len(await _join_rows(db_session, body["id"])) == 4


async def test_panel_asset_refs_preserve_author_order(client, db_session) -> None:
    """Author order is data, not incidental: create ``[b, a]`` → ``[b, a]``; PATCH to
    ``[a, b]`` → ``[a, b]``; the join rows' ``position`` is 0,1 in that same order."""
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    a = await _named_asset(client, project.id, name="A")
    b = await _named_asset(client, project.id, name="B")

    created = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"characters": [b, a]},
    )
    assert created.status_code == 201, created.text
    panel_id = created.json()["id"]
    assert created.json()["characters"] == [b, a]

    rows = sorted(await _join_rows(db_session, panel_id), key=lambda r: r.position)
    assert [(str(r.asset_id), r.position) for r in rows] == [(b, 0), (a, 1)]

    patched = await client.patch(f"/api/storyboard/panels/{panel_id}", json={"characters": [a, b]})
    assert patched.status_code == 200, patched.text
    assert patched.json()["characters"] == [a, b]
    assert (await _panel_by_id(client, project.id, panel_id))["characters"] == [a, b]

    rows = sorted(await _join_rows(db_session, panel_id), key=lambda r: r.position)
    assert [(str(r.asset_id), r.position) for r in rows] == [(a, 0), (b, 1)]


async def test_panel_partial_patch_does_not_clobber_other_roles(client, db_session) -> None:
    """A PATCH that names ONE role must not touch the others — the whole point of
    the diff-and-apply. ``{"props": []}`` clears props only; a later explicit
    ``{"location_asset_id": null}`` clears location only."""
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    a = await _named_asset(client, project.id, name="A")
    p = await _named_asset(client, project.id, name="Prop", kind="prop")
    loc = await _named_asset(client, project.id, name="Loc", kind="location")

    created = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"characters": [a], "props": [p], "location_asset_id": loc},
    )
    assert created.status_code == 201, created.text
    panel_id = created.json()["id"]

    cleared_props = await client.patch(f"/api/storyboard/panels/{panel_id}", json={"props": []})
    assert cleared_props.status_code == 200, cleared_props.text
    body = cleared_props.json()
    assert body["props"] == []
    assert body["characters"] == [a]  # UNTOUCHED
    assert body["location_asset_id"] == loc  # UNTOUCHED

    cleared_loc = await client.patch(
        f"/api/storyboard/panels/{panel_id}", json={"location_asset_id": None}
    )
    assert cleared_loc.status_code == 200, cleared_loc.text
    body = cleared_loc.json()
    assert body["location_asset_id"] is None
    assert body["characters"] == [a]  # still UNTOUCHED
    assert body["props"] == []

    # Re-projected from the DB, not echoed: one surviving join row (the character).
    row = await _panel_by_id(client, project.id, panel_id)
    assert row["characters"] == [a]
    assert row["props"] == []
    assert row["location_asset_id"] is None
    rows = await _join_rows(db_session, panel_id)
    assert [(str(r.asset_id), r.role) for r in rows] == [(a, "character")]


async def test_panel_create_unknown_asset_id_returns_422(client, db_session) -> None:
    """An id that names no asset of this project is the CALLER's fault → 422 (never a
    500 FK violation), and nothing is half-written."""
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    ghost = str(uuid7())
    resp = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"vo_text": "ghost cast", "characters": [ghost]},
    )
    assert resp.status_code == 422, resp.text

    listing = await client.get(f"/api/projects/{project.id}/storyboard/panels")
    assert listing.json()["total"] == 0  # no panel was created


async def test_panel_patch_unknown_asset_id_422_leaves_panel_unchanged(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    a = await _named_asset(client, project.id, name="A")
    created = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"vo_text": "keep me", "characters": [a]},
    )
    panel_id = created.json()["id"]

    resp = await client.patch(
        f"/api/storyboard/panels/{panel_id}",
        json={"vo_text": "mutated", "characters": [str(uuid7())]},
    )
    assert resp.status_code == 422, resp.text

    row = await _panel_by_id(client, project.id, panel_id)
    assert row["characters"] == [a]  # refs unchanged
    assert row["vo_text"] == "keep me"  # and the rest of the PATCH rolled back too
    assert [str(r.asset_id) for r in await _join_rows(db_session, panel_id)] == [a]


async def test_panel_create_foreign_project_asset_returns_422(client, db_session) -> None:
    """The FK alone would happily accept ANOTHER project's (another TENANT's) asset —
    the repo's project-scoped validation is the cross-tenant guard. 422, no join row."""
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    # A project owned by a DIFFERENT tenant, with a real asset of its own.
    foreign = await _seed_project(db_session, tenant_id=uuid7(), subject_id=uuid7())
    foreign_asset = await StoryboardAssetRepository(db_session).create(
        project_id=foreign.id, position=0, kind="character", name="Foreign Hero"
    )
    foreign_asset_id = foreign_asset.id
    await db_session.commit()

    resp = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"vo_text": "steal", "characters": [str(foreign_asset_id)]},
    )
    assert resp.status_code == 422, resp.text

    # Nothing written: no panel, and not a single join row anywhere.
    assert (await client.get(f"/api/projects/{project.id}/storyboard/panels")).json()["total"] == 0
    total = (
        await db_session.execute(
            select(StoryboardPanelAsset).execution_options(populate_existing=True)
        )
    ).scalars()
    assert list(total) == []

    # Same guard on PATCH.
    panel_id = await _seed_panel(client, project.id)
    patched = await client.patch(
        f"/api/storyboard/panels/{panel_id}",
        json={"location_asset_id": str(foreign_asset_id)},
    )
    assert patched.status_code == 422, patched.text
    assert await _join_rows(db_session, panel_id) == []


async def test_delete_referenced_asset_returns_409_naming_the_panels(client, db_session) -> None:
    """Deleting an asset a panel still references is a 409 that NAMES the panels
    (``detail.panels[] = [{id, shot_key}]``) — never the silent data corruption the
    pre-0051 undeclared FK allowed, and never a leaked 500."""
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    a = await _named_asset(client, project.id, name="A")
    created = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"shot_key": "clip-7", "characters": [a]},
    )
    assert created.status_code == 201, created.text
    panel_id = created.json()["id"]

    conflict = await client.delete(f"/api/storyboard/assets/{a}?expected_lock_version=0")
    assert conflict.status_code == 409, conflict.text
    panels = conflict.json()["detail"]["panels"]
    assert panels == [{"id": panel_id, "shot_key": "clip-7"}]

    # The asset is still there (the 409 did not partially delete it).
    listing = await client.get(f"/api/projects/{project.id}/storyboard/assets")
    assert [i["id"] for i in listing.json()["items"]] == [a]

    # Clear the reference → the delete now succeeds.
    cleared = await client.patch(f"/api/storyboard/panels/{panel_id}", json={"characters": []})
    assert cleared.status_code == 200, cleared.text
    assert await _join_rows(db_session, panel_id) == []

    deleted = await client.delete(f"/api/storyboard/assets/{a}?expected_lock_version=0")
    assert deleted.status_code == 204, deleted.text
    assert (await client.get(f"/api/projects/{project.id}/storyboard/assets")).json()["total"] == 0


async def test_delete_unreferenced_asset_returns_204(client, db_session) -> None:
    """CONTROL: the 409 must not over-fire — an asset no panel references deletes."""
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    used = await _named_asset(client, project.id, name="Used")
    unused = await _named_asset(client, project.id, name="Unused")
    created = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"shot_key": "clip-1", "characters": [used]},
    )
    assert created.status_code == 201, created.text

    assert (
        await client.delete(f"/api/storyboard/assets/{unused}?expected_lock_version=0")
    ).status_code == 204
    # …and the referenced one is still protected.
    assert (
        await client.delete(f"/api/storyboard/assets/{used}?expected_lock_version=0")
    ).status_code == 409


async def test_panel_patch_swaps_location_to_a_different_asset(client, db_session) -> None:
    """Swapping location A → B in ONE patch must succeed (200), not 500.

    The old join row is removed and a new one appended in the same flush; SQLAlchemy
    emits the INSERT before the DELETE, so both rows carry role='location' for the
    same panel mid-flush. ``uq_storyboard_panel_assets_one_location`` is a partial
    unique INDEX (not a deferrable constraint), so it fires immediately — the row
    must be recycled in place instead.
    """
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    panel_id = await _seed_panel(client, project.id)
    loc_a = await _seed_asset(client, project.id)
    loc_b = await _seed_asset(client, project.id)

    resp = await client.patch(
        f"/api/storyboard/panels/{panel_id}", json={"location_asset_id": loc_a}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["location_asset_id"] == loc_a

    resp = await client.patch(
        f"/api/storyboard/panels/{panel_id}", json={"location_asset_id": loc_b}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["location_asset_id"] == loc_b


# ─── scene_id validation + PATCH duplicate shot_key (500-guard PR) ────────
async def test_patch_panel_duplicate_shot_key_returns_409(client, db_session) -> None:
    # PATCH panel B's shot_key to panel A's key → uq_storyboard_panels_project_shot must
    # surface a clean 409, NOT a leaked 500 (the create path already 409s; patch didn't).
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    panel_a = await _seed_panel(client, project.id)
    panel_b = await _seed_panel(client, project.id)

    # Discover panel A's auto-assigned shot_key (clip-N) from the list.
    listing = await client.get(f"/api/projects/{project.id}/storyboard/panels")
    assert listing.status_code == 200, listing.text
    by_id = {p["id"]: p for p in listing.json()["items"]}
    a_shot_key = by_id[panel_a]["shot_key"]
    assert by_id[panel_b]["shot_key"] != a_shot_key

    dup = await client.patch(f"/api/storyboard/panels/{panel_b}", json={"shot_key": a_shot_key})
    assert dup.status_code == 409, dup.text
    assert a_shot_key in dup.json()["detail"]


async def test_create_panel_unknown_scene_id_returns_422(client, db_session) -> None:
    # A scene_id that isn't a scene of this project → 422 (was a FK-violation 500).
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    resp = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"vo_text": "x", "scene_id": str(uuid4())},
    )
    assert resp.status_code == 422, resp.text


async def test_patch_panel_unknown_scene_id_returns_422(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    panel_id = await _seed_panel(client, project.id)
    resp = await client.patch(f"/api/storyboard/panels/{panel_id}", json={"scene_id": str(uuid4())})
    assert resp.status_code == 422, resp.text


async def test_panel_cross_project_scene_id_rejected(client, db_session) -> None:
    # A scene owned by project A must NOT silently bind to a panel of project B (same
    # tenant). The FK alone would accept it — the project-scoped guard makes it a 422.
    project_a = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    project_b = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    scene_a = await _seed_scene(client, project_a.id)
    panel_b = await _seed_panel(client, project_b.id)

    resp = await client.patch(f"/api/storyboard/panels/{panel_b}", json={"scene_id": scene_a})
    assert resp.status_code == 422, resp.text


async def test_patch_panel_valid_same_project_scene_id_ok(client, db_session) -> None:
    # Guard against over-rejecting the legitimate move-to-scene flow.
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    panel_id = await _seed_panel(client, project.id)
    scene_id = await _seed_scene(client, project.id)

    resp = await client.patch(f"/api/storyboard/panels/{panel_id}", json={"scene_id": scene_id})
    assert resp.status_code == 200, resp.text
    assert resp.json()["scene_id"] == scene_id
