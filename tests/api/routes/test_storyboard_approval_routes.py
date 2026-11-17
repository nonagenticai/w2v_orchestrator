"""DB-backed integration tests for the storyboard per-panel approval trail.

Covers the Boords-pattern approve/unapprove surface mounted by
``src/api/routes/storyboard.py``:

* ``POST /api/storyboard/panels/{panel_id}/{approve,unapprove}`` — flip current
  state + append an append-only event; idempotent no-ops write NO event.
* ``GET  /api/storyboard/panels/{panel_id}/events`` — the per-panel trail.
* ``GET  /api/projects/{project_id}/storyboard/events`` — the per-project trail.
* the generic ``PATCH /api/storyboard/panels/{panel_id}`` status interception —
  a direct ``status`` PATCH (the FE dropdown path) MUST write the trail too.

The panel row holds the CURRENT state (``status`` + ``approved_at`` /
``approved_by``); the event table is the immutable history. ``revision`` (the
content-edit counter owned by AI rewrite) is NEVER bumped by an approval — the
event only snapshots it. Mirrors ``test_storyboard_routes.py``.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from src.database.models import Project

pytestmark = pytest.mark.asyncio


async def _seed_project(db_session, *, tenant_id: UUID, subject_id: UUID) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="storyboard-approval",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def _seed_panel(client, project_id) -> str:
    r = await client.post(
        f"/api/projects/{project_id}/storyboard/panels",
        json={"vo_text": "x"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def _panel_events(client, panel_id):
    r = await client.get(f"/api/storyboard/panels/{panel_id}/events")
    assert r.status_code == 200, r.text
    return r.json()


# ─── Approve ─────────────────────────────────────────────────────────────
async def test_approve_sets_state_and_writes_event(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel_id = await _seed_panel(client, project.id)

    resp = await client.post(
        f"/api/storyboard/panels/{panel_id}/approve", json={"note": "looks good"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "approved"
    assert body["approved_by"] == client.subject_id
    assert body["approved_at"] is not None
    # Approval snapshots revision but never bumps it.
    assert body["revision"] == 1

    events = await _panel_events(client, panel_id)
    assert events["total"] == 1
    ev = events["items"][0]
    assert ev["event_type"] == "approved"
    assert ev["note"] == "looks good"
    assert ev["revision"] == 1
    assert ev["panel_id"] == panel_id
    assert ev["project_id"] == str(project.id)
    assert ev["actor_subject_id"] == client.subject_id


async def test_approve_is_idempotent(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel_id = await _seed_panel(client, project.id)

    first = await client.post(f"/api/storyboard/panels/{panel_id}/approve", json={})
    second = await client.post(f"/api/storyboard/panels/{panel_id}/approve", json={})
    assert first.status_code == 200 and second.status_code == 200, (first.text, second.text)
    assert second.json()["status"] == "approved"

    # Second approve is a no-op → NO second event.
    events = await _panel_events(client, panel_id)
    assert events["total"] == 1


async def test_revision_not_bumped_by_approval(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel_id = await _seed_panel(client, project.id)
    resp = await client.post(f"/api/storyboard/panels/{panel_id}/approve", json={})
    assert resp.status_code == 200, resp.text
    assert resp.json()["revision"] == 1
    # Re-GET confirms the persisted panel revision is untouched.
    listing = await client.get(f"/api/projects/{project.id}/storyboard/panels")
    row = next(i for i in listing.json()["items"] if i["id"] == panel_id)
    assert row["revision"] == 1


# ─── Unapprove ───────────────────────────────────────────────────────────
async def test_unapprove_roundtrip(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel_id = await _seed_panel(client, project.id)

    await client.post(f"/api/storyboard/panels/{panel_id}/approve", json={})
    resp = await client.post(f"/api/storyboard/panels/{panel_id}/unapprove", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "draft"
    assert body["approved_at"] is None
    assert body["approved_by"] is None

    events = await _panel_events(client, panel_id)
    assert events["total"] == 2
    assert [e["event_type"] for e in events["items"]] == ["approved", "unapproved"]


async def test_unapprove_never_approved_is_noop(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel_id = await _seed_panel(client, project.id)

    resp = await client.post(f"/api/storyboard/panels/{panel_id}/unapprove", json={})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "draft"

    # No-op → NO event.
    events = await _panel_events(client, panel_id)
    assert events["total"] == 0


# ─── Not-found + cross-tenant ────────────────────────────────────────────
async def test_approve_unknown_panel_404(client, db_session) -> None:
    resp = await client.post(f"/api/storyboard/panels/{uuid4()}/approve", json={})
    assert resp.status_code == 404, resp.text


async def test_approve_cross_tenant_404(
    client, other_tenant_client: AsyncClient, db_session
) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel_id = await _seed_panel(client, project.id)

    bleed = await other_tenant_client.post(f"/api/storyboard/panels/{panel_id}/approve", json={})
    assert bleed.status_code == 404, bleed.text

    # The panel stays draft for its real tenant — the cross-tenant call did nothing.
    listing = await client.get(f"/api/projects/{project.id}/storyboard/panels")
    assert listing.json()["items"][0]["status"] == "draft"


# ─── Project-wide trail listing ──────────────────────────────────────────
async def test_project_events_listing_is_chronological(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel_a = await _seed_panel(client, project.id)
    panel_b = await _seed_panel(client, project.id)

    ra = await client.post(f"/api/storyboard/panels/{panel_a}/approve", json={})
    rb = await client.post(f"/api/storyboard/panels/{panel_b}/approve", json={})
    assert ra.status_code == 200 and rb.status_code == 200, (ra.text, rb.text)

    resp = await client.get(f"/api/projects/{project.id}/storyboard/events")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 2
    # Chronological: panel_a approved before panel_b (separate request transactions).
    assert [e["panel_id"] for e in body["items"]] == [panel_a, panel_b]
    assert all(e["event_type"] == "approved" for e in body["items"])


async def test_project_events_unknown_project_404(client) -> None:
    r = await client.get(f"/api/projects/{uuid4()}/storyboard/events")
    assert r.status_code == 404


# ─── Interception: the generic PATCH status route writes the trail ────────
async def test_patch_status_via_generic_route_writes_trail(client, db_session) -> None:
    """The FE dropdown still PATCHes ``status`` on the generic panel route; that
    path MUST flip approved_at/by AND append the trail, exactly like /approve."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel_id = await _seed_panel(client, project.id)

    resp = await client.patch(f"/api/storyboard/panels/{panel_id}", json={"status": "approved"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "approved"
    assert body["approved_at"] is not None
    assert body["approved_by"] == client.subject_id

    events = await _panel_events(client, panel_id)
    assert events["total"] == 1
    assert events["items"][0]["event_type"] == "approved"


async def test_patch_status_to_draft_via_generic_route_unapproves(client, db_session) -> None:
    """PATCH status back to draft flows through unapprove (clears state + trail)."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel_id = await _seed_panel(client, project.id)
    await client.post(f"/api/storyboard/panels/{panel_id}/approve", json={})

    resp = await client.patch(f"/api/storyboard/panels/{panel_id}", json={"status": "draft"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "draft"
    assert body["approved_at"] is None

    events = await _panel_events(client, panel_id)
    assert events["total"] == 2
    assert [e["event_type"] for e in events["items"]] == ["approved", "unapproved"]


async def test_patch_non_status_field_still_works(client, db_session) -> None:
    """CONTROL: a PATCH that omits status keeps the unchanged behaviour (no event)."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel_id = await _seed_panel(client, project.id)
    resp = await client.patch(f"/api/storyboard/panels/{panel_id}", json={"vo_text": "revised"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["vo_text"] == "revised"
    assert resp.json()["status"] == "draft"

    events = await _panel_events(client, panel_id)
    assert events["total"] == 0
