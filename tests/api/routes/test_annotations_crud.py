"""DB-backed integration tests for annotations CRUD.

Covers /api/runs/{run_id}/stages/{stage_name}/annotations (list + create) and
/api/annotations/{id} (get + patch + delete). Tenant scoping is enforced via
stage_output -> run -> deliverable -> project.tenant_id.

Rows are seeded directly through the db_session fixture rather than driven
through the run engine, so stage_outputs are guaranteed stable while the
test exercises the annotation endpoints.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from src.database.models import Deliverable, Project, Run, StageOutput

pytestmark = pytest.mark.asyncio


async def _seed_run_with_stage_output(
    db_session,
    *,
    tenant_id: UUID,
    subject_id: UUID,
    stage_name: str = "script_draft",
    output: dict | None = None,
) -> tuple[Project, Deliverable, Run, StageOutput]:
    """Seed Project + Deliverable + Run + StageOutput in the test DB.

    Uses a stage that supports element pinning (script_draft -> lines[]) so
    element_id paths can be exercised when callers pass an output payload.
    """
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="annotations-crud",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    deliverable = Deliverable(
        id=uuid4(),
        project_id=project.id,
        name="d",
        format="16x9",
        captions_mode="none",
        target_duration_seconds=30,
    )
    run = Run(
        id=uuid4(),
        deliverable_id=deliverable.id,
        blueprint_id="generic_v1",
        status="paused",
        config_snapshot={},
        pause_after_stages=[],
    )
    so = StageOutput(
        id=uuid4(),
        run_id=run.id,
        stage_name=stage_name,
        status="completed" if output else "pending",
        output=output,
    )
    db_session.add_all([project, deliverable, run, so])
    await db_session.commit()
    return project, deliverable, run, so


async def test_create_then_list_annotations_round_trip(client, db_session) -> None:
    _, _, run, _ = await _seed_run_with_stage_output(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    create = await client.post(
        f"/api/runs/{run.id}/stages/script_draft/annotations",
        json={"text": "tighten the hook"},
    )
    assert create.status_code == 201, create.text
    body = create.json()
    assert body["text"] == "tighten the hook"
    assert body["status"] == "open"
    assert body["element_id"] is None
    assert body["applied_in_run_id"] is None

    listing = await client.get(f"/api/runs/{run.id}/stages/script_draft/annotations")
    assert listing.status_code == 200, listing.text
    items = listing.json()["items"]
    assert len(items) == 1
    assert items[0]["id"] == body["id"]


async def test_create_with_invalid_element_id_returns_400(client, db_session) -> None:
    _, _, run, _ = await _seed_run_with_stage_output(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
        output={"lines": [{"id": "L1", "text": "hi"}]},
    )
    bad = await client.post(
        f"/api/runs/{run.id}/stages/script_draft/annotations",
        json={"text": "fix this", "element_id": "L99"},
    )
    assert bad.status_code == 400, bad.text
    assert "element_id" in bad.text

    ok = await client.post(
        f"/api/runs/{run.id}/stages/script_draft/annotations",
        json={"text": "fix this", "element_id": "L1"},
    )
    assert ok.status_code == 201, ok.text
    assert ok.json()["element_id"] == "L1"


async def test_get_patch_delete_annotation_by_id(client, db_session) -> None:
    _, _, run, _ = await _seed_run_with_stage_output(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    created = await client.post(
        f"/api/runs/{run.id}/stages/script_draft/annotations",
        json={"text": "v1"},
    )
    aid = created.json()["id"]

    fetched = await client.get(f"/api/annotations/{aid}")
    assert fetched.status_code == 200
    assert fetched.json()["text"] == "v1"

    patched = await client.patch(f"/api/annotations/{aid}", json={"text": "v2"})
    assert patched.status_code == 200, patched.text
    assert patched.json()["text"] == "v2"

    deleted = await client.delete(f"/api/annotations/{aid}")
    assert deleted.status_code == 204
    gone = await client.get(f"/api/annotations/{aid}")
    assert gone.status_code == 404


async def test_patch_status_applied_rejected_with_400(client, db_session) -> None:
    """Only the regenerate transaction can mark an annotation 'applied'."""
    _, _, run, _ = await _seed_run_with_stage_output(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    created = await client.post(
        f"/api/runs/{run.id}/stages/script_draft/annotations",
        json={"text": "v1"},
    )
    aid = created.json()["id"]
    blocked = await client.patch(f"/api/annotations/{aid}", json={"status": "applied"})
    assert blocked.status_code == 400
    assert "applied" in blocked.text


async def test_cross_tenant_annotation_returns_404(
    client, other_tenant_client: AsyncClient, db_session
) -> None:
    _, _, run, _ = await _seed_run_with_stage_output(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    created = await client.post(
        f"/api/runs/{run.id}/stages/script_draft/annotations",
        json={"text": "secret"},
    )
    assert created.status_code == 201, created.text
    aid = created.json()["id"]

    bleed = await other_tenant_client.get(f"/api/annotations/{aid}")
    assert bleed.status_code == 404


async def test_list_filters_by_status(client, db_session) -> None:
    _, _, run, so = await _seed_run_with_stage_output(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    a1 = await client.post(
        f"/api/runs/{run.id}/stages/script_draft/annotations",
        json={"text": "open one"},
    )
    a2 = await client.post(
        f"/api/runs/{run.id}/stages/script_draft/annotations",
        json={"text": "to dismiss"},
    )
    assert a1.status_code == 201 and a2.status_code == 201

    dismissed = await client.patch(
        f"/api/annotations/{a2.json()['id']}", json={"status": "dismissed"}
    )
    assert dismissed.status_code == 200
    assert dismissed.json()["status"] == "dismissed"

    open_only = await client.get(f"/api/runs/{run.id}/stages/script_draft/annotations")
    assert {i["id"] for i in open_only.json()["items"]} == {a1.json()["id"]}

    all_status = await client.get(f"/api/runs/{run.id}/stages/script_draft/annotations?status=all")
    assert {i["id"] for i in all_status.json()["items"]} == {
        a1.json()["id"],
        a2.json()["id"],
    }
    # so id round-trip
    assert UUID(open_only.json()["stage_output_id"]) == so.id
