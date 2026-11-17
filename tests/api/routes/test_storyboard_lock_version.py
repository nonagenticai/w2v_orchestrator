"""DB-backed integration tests for the storyboard panel ``lock_version`` token.

Exercises the opt-in optimistic-concurrency column added to ``storyboard_panels``
through the real FastAPI app (``client`` + ``app`` fixtures, real postgres via
``db_session``). The contract:

* Every panel WRITE — ``PATCH``, ``approve``, ``unapprove``, AI ``rewrite`` —
  advances ``lock_version`` by exactly one (a plain counter, NOT ``revision``).
* A write that carries ``expected_lock_version`` gets compare-and-set: it succeeds
  and bumps only if the token still matches; a stale token is a ``409`` with detail
  ``"panel was modified concurrently"``.
* A write with NO ``expected_lock_version`` is last-write-wins — it always succeeds,
  even against a row whose token has since moved (backward compatible).
* ``lock_version`` is present in the panel response body.

Mirrors the fixtures of ``test_storyboard_approval_routes.py`` (project + panel
seeding) and ``test_storyboard_ai_routes.py`` (fake-WM assist override for rewrite).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.clients import WMResponse

from src.api.routes._common import get_db_session
from src.api.routes.storyboard import get_assist_service
from src.database.models import Project
from src.services.storyboard.assist import StoryboardAssistService

pytestmark = pytest.mark.asyncio


# ─── helpers ─────────────────────────────────────────────────────────────
async def _seed_project(db_session, *, tenant_id: UUID, subject_id: UUID) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="storyboard-lock-version",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def _seed_panel(client, project_id) -> dict[str, Any]:
    r = await client.post(
        f"/api/projects/{project_id}/storyboard/panels",
        json={"vo_text": "x"},
    )
    assert r.status_code == 201, r.text
    return r.json()


async def _named_asset(client, project_id, *, name: str, kind: str = "character") -> str:
    # Mirrors ``_named_asset`` in test_storyboard_routes.py: assets are referenced by id
    # from a panel's characters/props/location. An unseeded id → 422, so seed first.
    r = await client.post(
        f"/api/projects/{project_id}/storyboard/assets", json={"kind": kind, "name": name}
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _fake(output: dict[str, Any]) -> AsyncMock:
    fake = AsyncMock()
    fake.message.return_value = WMResponse(
        output=output,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )
    return fake


def _install_fake(app: FastAPI, fake: AsyncMock) -> None:
    def _factory(session: AsyncSession = Depends(get_db_session)) -> StoryboardAssistService:
        return StoryboardAssistService(session, client=fake)

    app.dependency_overrides[get_assist_service] = _factory


# ─── response surface ────────────────────────────────────────────────────
async def test_lock_version_present_and_starts_at_zero(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    assert "lock_version" in panel
    assert panel["lock_version"] == 0


# ─── every write bumps ───────────────────────────────────────────────────
async def test_every_write_increments_lock_version(client, app, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    panel_id = panel["id"]
    assert panel["lock_version"] == 0

    # PATCH → 1
    r = await client.patch(
        f"/api/storyboard/panels/{panel_id}", json={"action_description": "edited"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["lock_version"] == 1

    # approve → 2
    r = await client.post(f"/api/storyboard/panels/{panel_id}/approve", json={})
    assert r.status_code == 200, r.text
    assert r.json()["lock_version"] == 2

    # unapprove → 3
    r = await client.post(f"/api/storyboard/panels/{panel_id}/unapprove", json={})
    assert r.status_code == 200, r.text
    assert r.json()["lock_version"] == 3

    # AI rewrite (routes through repo.update) → 4
    _install_fake(app, _fake({"dialogue": "punchy"}))
    try:
        r = await client.post(
            f"/api/storyboard/panels/{panel_id}/rewrite",
            json={"instruction": "make it punchy"},
        )
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["lock_version"] == 4
    assert body["dialogue"] == "punchy"


async def test_idempotent_approve_does_not_bump(client, db_session) -> None:
    # A no-op approve (already approved) is not a write → token unchanged.
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel_id = (await _seed_panel(client, project.id))["id"]

    first = await client.post(f"/api/storyboard/panels/{panel_id}/approve", json={})
    assert first.status_code == 200
    assert first.json()["lock_version"] == 1

    second = await client.post(f"/api/storyboard/panels/{panel_id}/approve", json={})
    assert second.status_code == 200
    # Idempotent no-op → NO further bump.
    assert second.json()["lock_version"] == 1


# ─── compare-and-set ─────────────────────────────────────────────────────
async def test_cas_success_bumps(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    panel_id = panel["id"]
    assert panel["lock_version"] == 0

    r = await client.patch(
        f"/api/storyboard/panels/{panel_id}",
        json={"action_description": "cas edit", "expected_lock_version": 0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["lock_version"] == 1
    assert body["action_description"] == "cas edit"


async def test_cas_conflict_returns_409(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    panel_id = panel["id"]
    # Two readers both observe version 0.
    version_n = panel["lock_version"]
    assert version_n == 0

    # First writer wins the CAS at N → token is now 1.
    first = await client.patch(
        f"/api/storyboard/panels/{panel_id}",
        json={"action_description": "first", "expected_lock_version": version_n},
    )
    assert first.status_code == 200, first.text
    assert first.json()["lock_version"] == 1

    # Second writer replays the now-stale N → 409.
    second = await client.patch(
        f"/api/storyboard/panels/{panel_id}",
        json={"action_description": "second", "expected_lock_version": version_n},
    )
    assert second.status_code == 409, second.text
    assert second.json()["detail"] == "panel was modified concurrently"

    # The losing write did NOT land: the first writer's content survives.
    current = await client.get(f"/api/projects/{project.id}/storyboard/panels")
    assert current.status_code == 200
    row = next(p for p in current.json()["items"] if p["id"] == panel_id)
    assert row["action_description"] == "first"
    assert row["lock_version"] == 1


async def test_cas_conflict_on_approve_returns_409(client, db_session) -> None:
    # CAS also guards the approve/unapprove state flip.
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    panel_id = panel["id"]

    # Advance the token out from under the stale approve.
    bump = await client.patch(
        f"/api/storyboard/panels/{panel_id}", json={"action_description": "moved"}
    )
    assert bump.status_code == 200
    assert bump.json()["lock_version"] == 1

    # approve echoing the stale version 0 → 409.
    stale = await client.post(
        f"/api/storyboard/panels/{panel_id}/approve", json={"expected_lock_version": 0}
    )
    assert stale.status_code == 409, stale.text
    assert stale.json()["detail"] == "panel was modified concurrently"

    # Fresh version succeeds and bumps.
    ok = await client.post(
        f"/api/storyboard/panels/{panel_id}/approve", json={"expected_lock_version": 1}
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["status"] == "approved"
    assert ok.json()["lock_version"] == 2


# ─── backward compatibility ──────────────────────────────────────────────
async def test_no_expected_version_is_last_write_wins(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    panel_id = panel["id"]

    # Move the token to 1 via a CAS write.
    first = await client.patch(
        f"/api/storyboard/panels/{panel_id}",
        json={"action_description": "first", "expected_lock_version": 0},
    )
    assert first.status_code == 200
    assert first.json()["lock_version"] == 1

    # A write WITHOUT expected_lock_version must still succeed against the bumped row
    # (last-write-wins) and advance the token to 2.
    second = await client.patch(
        f"/api/storyboard/panels/{panel_id}", json={"action_description": "second"}
    )
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["action_description"] == "second"
    assert body["lock_version"] == 2


# ─── join-only edits (cast/props/location) share the token ────────────────
# ``characters``/``props``/``location_asset_id`` are read-only projections of the
# ``storyboard_panel_assets`` join table — a PATCH naming only these writes child
# rows, NOT a scalar panel column. These tests prove that path still advances
# ``lock_version`` (last-write-wins) and is compare-and-set guarded, closing the
# "a join-only edit may not bump the token" residual with executed assertions.
async def test_join_only_lww_bump(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    panel_id = panel["id"]
    assert panel["lock_version"] == 0
    a = await _named_asset(client, project.id, name="A")
    p = await _named_asset(client, project.id, name="Prop", kind="prop")
    loc = await _named_asset(client, project.id, name="Loc", kind="location")

    # characters-only → bump to 1.
    r = await client.patch(f"/api/storyboard/panels/{panel_id}", json={"characters": [a]})
    assert r.status_code == 200, r.text
    assert r.json()["lock_version"] == 1
    assert r.json()["characters"] == [a]

    # props-only → bump to 2; characters left untouched (partial-PATCH safety).
    r = await client.patch(f"/api/storyboard/panels/{panel_id}", json={"props": [p]})
    assert r.status_code == 200, r.text
    assert r.json()["lock_version"] == 2
    assert r.json()["props"] == [p]
    assert r.json()["characters"] == [a]

    # location-only set → bump to 3.
    r = await client.patch(f"/api/storyboard/panels/{panel_id}", json={"location_asset_id": loc})
    assert r.status_code == 200, r.text
    assert r.json()["lock_version"] == 3
    assert r.json()["location_asset_id"] == loc

    # location-only clear (null is a legitimate clear) → still a write → bump to 4.
    r = await client.patch(f"/api/storyboard/panels/{panel_id}", json={"location_asset_id": None})
    assert r.status_code == 200, r.text
    assert r.json()["lock_version"] == 4
    assert r.json()["location_asset_id"] is None


async def test_join_only_cas_success_bumps(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    panel_id = panel["id"]
    assert panel["lock_version"] == 0
    a = await _named_asset(client, project.id, name="A")
    b = await _named_asset(client, project.id, name="B")

    # Join-only edit with a matching token → single in-DB bump, refresh reports N+1.
    r = await client.patch(
        f"/api/storyboard/panels/{panel_id}",
        json={"characters": [a], "expected_lock_version": 0},
    )
    assert r.status_code == 200, r.text
    assert r.json()["lock_version"] == 1
    assert r.json()["characters"] == [a]

    # A second join-only CAS against the fresh token → 2 (no double-bump/stale writeback).
    r = await client.patch(
        f"/api/storyboard/panels/{panel_id}",
        json={"characters": [a, b], "expected_lock_version": 1},
    )
    assert r.status_code == 200, r.text
    assert r.json()["lock_version"] == 2
    assert r.json()["characters"] == [a, b]


async def test_join_only_cas_conflict_returns_409(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    panel_id = panel["id"]
    a = await _named_asset(client, project.id, name="A")
    b = await _named_asset(client, project.id, name="B")
    # Two readers both observe version 0.
    version_n = panel["lock_version"]
    assert version_n == 0

    # First join-only writer wins the CAS at N → token is now 1.
    first = await client.patch(
        f"/api/storyboard/panels/{panel_id}",
        json={"characters": [a], "expected_lock_version": version_n},
    )
    assert first.status_code == 200, first.text
    assert first.json()["lock_version"] == 1

    # Second join-only writer replays the now-stale N → 409.
    second = await client.patch(
        f"/api/storyboard/panels/{panel_id}",
        json={"characters": [b], "expected_lock_version": version_n},
    )
    assert second.status_code == 409, second.text
    assert second.json()["detail"] == "panel was modified concurrently"

    # The losing write did NOT land — the CAS ran BEFORE the join rows were applied,
    # so the winner's cast survives and nothing was half-written.
    current = await client.get(f"/api/projects/{project.id}/storyboard/panels")
    assert current.status_code == 200
    row = next(p for p in current.json()["items"] if p["id"] == panel_id)
    assert row["characters"] == [a]
    assert row["lock_version"] == 1


async def test_join_only_edit_bumps_token_then_stale_scalar_409(client, db_session) -> None:
    # A join-only edit and a scalar edit share ONE token: the join bump must invalidate
    # a subsequent scalar CAS that echoes the pre-join version.
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    panel_id = panel["id"]
    a = await _named_asset(client, project.id, name="A")

    # Last-write-wins join-only edit moves the shared token 0 → 1.
    joined = await client.patch(f"/api/storyboard/panels/{panel_id}", json={"characters": [a]})
    assert joined.status_code == 200, joined.text
    assert joined.json()["lock_version"] == 1

    # A scalar edit echoing the stale pre-join version 0 → 409.
    stale = await client.patch(
        f"/api/storyboard/panels/{panel_id}",
        json={"action_description": "x", "expected_lock_version": 0},
    )
    assert stale.status_code == 409, stale.text
    assert stale.json()["detail"] == "panel was modified concurrently"

    # The fresh version succeeds; the join edit is still present.
    ok = await client.patch(
        f"/api/storyboard/panels/{panel_id}",
        json={"action_description": "x", "expected_lock_version": 1},
    )
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert body["lock_version"] == 2
    assert body["action_description"] == "x"
    assert body["characters"] == [a]


async def test_join_only_noop_still_bumps(client, db_session) -> None:
    # An accepted join-only write advances the token even when the asset set is unchanged
    # (parity with a same-value scalar PATCH) — so it still 409s a concurrent stale writer.
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    panel_id = panel["id"]
    a = await _named_asset(client, project.id, name="A")

    r = await client.patch(f"/api/storyboard/panels/{panel_id}", json={"characters": [a]})
    assert r.status_code == 200, r.text
    assert r.json()["lock_version"] == 1

    # Identical set under last-write-wins → still bumps to 2.
    r = await client.patch(f"/api/storyboard/panels/{panel_id}", json={"characters": [a]})
    assert r.status_code == 200, r.text
    assert r.json()["lock_version"] == 2

    # Identical set under CAS with the fresh token → bumps to 3.
    r = await client.patch(
        f"/api/storyboard/panels/{panel_id}",
        json={"characters": [a], "expected_lock_version": 2},
    )
    assert r.status_code == 200, r.text
    assert r.json()["lock_version"] == 3


# ─── scenes: lock_version + CAS (migration 0056) ─────────────────────────
async def _seed_scene(client, project_id, *, name: str = "s1") -> dict[str, Any]:
    r = await client.post(f"/api/projects/{project_id}/storyboard/scenes", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()


async def test_scene_lock_version_present_and_starts_at_zero(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    scene = await _seed_scene(client, project.id)
    assert "lock_version" in scene
    assert scene["lock_version"] == 0


async def test_scene_every_write_increments_lock_version(client, db_session) -> None:
    # Last-write-wins PATCH advances the token by exactly one each time.
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    scene_id = (await _seed_scene(client, project.id))["id"]

    r = await client.patch(f"/api/storyboard/scenes/{scene_id}", json={"name": "renamed"})
    assert r.status_code == 200, r.text
    assert r.json()["lock_version"] == 1
    assert r.json()["name"] == "renamed"

    r = await client.patch(f"/api/storyboard/scenes/{scene_id}", json={"location": "INT"})
    assert r.status_code == 200, r.text
    assert r.json()["lock_version"] == 2


async def test_scene_cas_success_bumps(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    scene = await _seed_scene(client, project.id)
    scene_id = scene["id"]
    assert scene["lock_version"] == 0

    r = await client.patch(
        f"/api/storyboard/scenes/{scene_id}",
        json={"description": "cas edit", "expected_lock_version": 0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["lock_version"] == 1
    assert body["description"] == "cas edit"


async def test_scene_cas_conflict_returns_409(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    scene = await _seed_scene(client, project.id)
    scene_id = scene["id"]
    version_n = scene["lock_version"]
    assert version_n == 0

    first = await client.patch(
        f"/api/storyboard/scenes/{scene_id}",
        json={"name": "first", "expected_lock_version": version_n},
    )
    assert first.status_code == 200, first.text
    assert first.json()["lock_version"] == 1

    # Replay of the now-stale token → 409, entity named "scene".
    second = await client.patch(
        f"/api/storyboard/scenes/{scene_id}",
        json={"name": "second", "expected_lock_version": version_n},
    )
    assert second.status_code == 409, second.text
    assert second.json()["detail"] == "scene was modified concurrently"

    # The losing write did NOT land: the first writer's content survives.
    current = await client.get(f"/api/projects/{project.id}/storyboard/scenes")
    assert current.status_code == 200
    row = next(s for s in current.json()["items"] if s["id"] == scene_id)
    assert row["name"] == "first"
    assert row["lock_version"] == 1


# ─── assets: lock_version + CAS (migration 0056) ─────────────────────────
async def _seed_asset(client, project_id, *, name: str = "hero") -> dict[str, Any]:
    r = await client.post(
        f"/api/projects/{project_id}/storyboard/assets",
        json={"kind": "character", "name": name},
    )
    assert r.status_code == 201, r.text
    return r.json()


async def test_asset_lock_version_present_and_starts_at_zero(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    asset = await _seed_asset(client, project.id)
    assert "lock_version" in asset
    assert asset["lock_version"] == 0


async def test_asset_every_write_increments_lock_version(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    asset_id = (await _seed_asset(client, project.id))["id"]

    r = await client.patch(
        f"/api/storyboard/assets/{asset_id}", json={"canonical_descriptor": "tall"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["lock_version"] == 1
    assert r.json()["canonical_descriptor"] == "tall"

    r = await client.patch(f"/api/storyboard/assets/{asset_id}", json={"style_token": "noir"})
    assert r.status_code == 200, r.text
    assert r.json()["lock_version"] == 2


async def test_asset_cas_success_bumps(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    asset = await _seed_asset(client, project.id)
    asset_id = asset["id"]
    assert asset["lock_version"] == 0

    r = await client.patch(
        f"/api/storyboard/assets/{asset_id}",
        json={"canonical_descriptor": "cas edit", "expected_lock_version": 0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["lock_version"] == 1
    assert body["canonical_descriptor"] == "cas edit"


async def test_asset_cas_conflict_returns_409(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    asset = await _seed_asset(client, project.id)
    asset_id = asset["id"]
    version_n = asset["lock_version"]
    assert version_n == 0

    first = await client.patch(
        f"/api/storyboard/assets/{asset_id}",
        json={"name": "first", "expected_lock_version": version_n},
    )
    assert first.status_code == 200, first.text
    assert first.json()["lock_version"] == 1

    # Replay of the now-stale token → 409, entity named "asset".
    second = await client.patch(
        f"/api/storyboard/assets/{asset_id}",
        json={"name": "second", "expected_lock_version": version_n},
    )
    assert second.status_code == 409, second.text
    assert second.json()["detail"] == "asset was modified concurrently"

    # The losing write did NOT land: the first writer's content survives.
    current = await client.get(f"/api/projects/{project.id}/storyboard/assets")
    assert current.status_code == 200
    row = next(a for a in current.json()["items"] if a["id"] == asset_id)
    assert row["name"] == "first"
    assert row["lock_version"] == 1


# ─── reorder advances lock_version (CAS-blind reorder revert, PR: reorder-cas) ───
async def test_reorder_bumps_lock_version_on_moved_panels_only(client, db_session) -> None:
    """A reorder MUTATES a moved panel's position, so it advances that row's
    ``lock_version``; a panel that keeps its slot keeps its token. Without the bump a
    stale-token PATCH could silently revert the reorder (see the 409 test below)."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    a = (await _seed_panel(client, project.id))["id"]  # position 0
    b = (await _seed_panel(client, project.id))["id"]  # position 1
    c = (await _seed_panel(client, project.id))["id"]  # position 2

    # Reverse: a→2, c→0 (both move); b stays at position 1 (unmoved).
    resp = await client.patch(
        f"/api/projects/{project.id}/storyboard/panels/reorder",
        json={"ordered_panel_ids": [c, b, a], "expected_lock_versions": {a: 0, b: 0, c: 0}},
    )
    assert resp.status_code == 200, resp.text
    by_id = {p["id"]: p for p in resp.json()}
    assert [p["id"] for p in resp.json()] == [c, b, a]
    assert by_id[a]["lock_version"] == 1  # moved 0→2
    assert by_id[c]["lock_version"] == 1  # moved 2→0
    assert by_id[b]["lock_version"] == 0  # unmoved → token untouched


async def test_stale_token_position_patch_after_reorder_409_no_silent_revert(
    client, db_session
) -> None:
    """The core bug: a client holding a pre-reorder token PATCHes ``position`` with its
    now-stale ``expected_lock_version``. Because the reorder advanced the token, the CAS
    no longer matches → 409 (not a silent revert of the reorder)."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    a = await _seed_panel(client, project.id)  # position 0, lock_version 0
    b = (await _seed_panel(client, project.id))["id"]  # position 1
    a_id = a["id"]
    assert a["lock_version"] == 0

    # Reorder moves A to position 1 (token bumped to 1) — the stale client never saw this.
    reordered = await client.patch(
        f"/api/projects/{project.id}/storyboard/panels/reorder",
        json={"ordered_panel_ids": [b, a_id], "expected_lock_versions": {a_id: 0, b: 0}},
    )
    assert reordered.status_code == 200, reordered.text

    # Stale client replays the pre-reorder token trying to yank A back to position 0.
    stale = await client.patch(
        f"/api/storyboard/panels/{a_id}",
        json={"position": 0, "expected_lock_version": 0},
    )
    assert stale.status_code == 409, stale.text
    assert stale.json()["detail"] == "panel was modified concurrently"

    # The reorder SURVIVED — A is still at its new position, not silently reverted.
    listing = await client.get(f"/api/projects/{project.id}/storyboard/panels")
    assert [i["id"] for i in listing.json()["items"]] == [b, a_id]

    # The fresh (post-reorder) token still works — the bump is not a deadlock.
    fresh = await client.patch(
        f"/api/storyboard/panels/{a_id}",
        json={"action_description": "ok", "expected_lock_version": 1},
    )
    assert fresh.status_code == 200, fresh.text
    assert fresh.json()["lock_version"] == 2


async def test_scene_stale_token_position_patch_after_reorder_409(client, db_session) -> None:
    """Scene mirror of the panel reorder-CAS fix."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    s1 = await _seed_scene(client, project.id, name="s1")  # position 0, lock_version 0
    s2 = (await _seed_scene(client, project.id, name="s2"))["id"]  # position 1
    s1_id = s1["id"]
    assert s1["lock_version"] == 0

    reordered = await client.patch(
        f"/api/projects/{project.id}/storyboard/scenes/reorder",
        json={"ordered_scene_ids": [s2, s1_id], "expected_lock_versions": {s1_id: 0, s2: 0}},
    )
    assert reordered.status_code == 200, reordered.text
    by_id = {s["id"]: s for s in reordered.json()}
    assert by_id[s1_id]["lock_version"] == 1  # moved → bumped

    stale = await client.patch(
        f"/api/storyboard/scenes/{s1_id}",
        json={"position": 0, "expected_lock_version": 0},
    )
    assert stale.status_code == 409, stale.text
    assert stale.json()["detail"] == "scene was modified concurrently"

    listing = await client.get(f"/api/projects/{project.id}/storyboard/scenes")
    assert [i["id"] for i in listing.json()["items"]] == [s2, s1_id]
