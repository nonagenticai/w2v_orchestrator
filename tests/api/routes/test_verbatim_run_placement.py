"""``GET /api/projects/{id}/verbatim-items?run_id=`` — read-time placement.

Without ``run_id`` the response is byte-identical to the historical shape
(everything ``pending``). With ``run_id`` the lifecycle fields are derived by
re-matching each phrase against that run's currently-persisted carrier stage
outputs. Tenant scoping reuses ``_tenant_owns_run`` (run -> deliverable ->
project.tenant_id), exactly like every other run-scoped route.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from src.database.models import Deliverable, Project, Run, StageOutput

pytestmark = pytest.mark.asyncio


async def _seed_project(db_session, *, tenant_id: UUID, subject_id: UUID) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="verbatim-placement",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def _seed_run(
    db_session,
    project: Project,
    *,
    stage_name: str = "script_draft",
    output: dict | None = None,
) -> Run:
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
        status="completed",
        config_snapshot={},
        pause_after_stages=[],
    )
    rows: list[object] = [deliverable, run]
    if output is not None:
        rows.append(
            StageOutput(
                id=uuid4(),
                run_id=run.id,
                stage_name=stage_name,
                status="completed",
                output=output,
            )
        )
    db_session.add_all(rows)
    await db_session.commit()
    return run


_LINES = {
    "lines": [
        {"id": "L1", "text": "Welcome to the product tour."},
        {"id": "L2", "text": "Remember: results may vary. That is the disclaimer."},
    ]
}


async def test_run_id_derives_real_placement(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    run = await _seed_run(db_session, project, output=_LINES)
    placed = await client.post(
        f"/api/projects/{project.id}/verbatim-items",
        json={"text": "results may vary"},
    )
    missing = await client.post(
        f"/api/projects/{project.id}/verbatim-items",
        json={"text": "money back guaranteed"},
    )
    assert placed.status_code == 201 and missing.status_code == 201

    r = await client.get(f"/api/projects/{project.id}/verbatim-items?run_id={run.id}")
    assert r.status_code == 200, r.text
    by_text = {i["text"]: i for i in r.json()["items"]}

    hit = by_text["results may vary"]
    assert hit["status"] == "placed"
    assert hit["placed_in_stage"] == "script_draft"
    assert hit["placed_in_element_id"] == "L2"

    miss = by_text["money back guaranteed"]
    assert miss["status"] == "pending"
    assert miss["placed_in_stage"] is None
    assert miss["placed_in_element_id"] is None


async def test_straddling_phrase_is_placed_with_null_element_id(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    run = await _seed_run(db_session, project, output=_LINES)
    await client.post(
        f"/api/projects/{project.id}/verbatim-items",
        json={"text": "product tour. Remember"},
    )

    r = await client.get(f"/api/projects/{project.id}/verbatim-items?run_id={run.id}")
    item = r.json()["items"][0]
    assert item["status"] == "placed"
    assert item["placed_in_stage"] == "script_draft"
    assert item["placed_in_element_id"] is None


async def test_without_run_id_everything_stays_pending(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _seed_run(db_session, project, output=_LINES)
    await client.post(
        f"/api/projects/{project.id}/verbatim-items",
        json={"text": "results may vary"},
    )

    r = await client.get(f"/api/projects/{project.id}/verbatim-items")
    assert r.status_code == 200
    item = r.json()["items"][0]
    assert item["status"] == "pending"
    assert item["placed_in_stage"] is None
    assert item["placed_in_element_id"] is None


async def test_run_with_no_carrier_stage_output_stays_pending(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    # A completed run whose only stage output is a NON-carrier stage.
    run = await _seed_run(
        db_session,
        project,
        stage_name="storyboard_draft",
        output={"shots": [{"id": "S1", "text": "results may vary"}]},
    )
    await client.post(
        f"/api/projects/{project.id}/verbatim-items",
        json={"text": "results may vary"},
    )

    r = await client.get(f"/api/projects/{project.id}/verbatim-items?run_id={run.id}")
    assert r.json()["items"][0]["status"] == "pending"


async def test_unknown_run_id_returns_404(client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    r = await client.get(f"/api/projects/{project.id}/verbatim-items?run_id={uuid4()}")
    assert r.status_code == 404, r.text


async def test_cross_tenant_run_id_returns_404(client, db_session) -> None:
    """A run under ANOTHER tenant's project must not leak its stage output through
    this project's verbatim listing."""
    mine = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    theirs = await _seed_project(db_session, tenant_id=uuid4(), subject_id=uuid4())
    foreign_run = await _seed_run(db_session, theirs, output=_LINES)
    await client.post(
        f"/api/projects/{mine.id}/verbatim-items",
        json={"text": "results may vary"},
    )

    r = await client.get(f"/api/projects/{mine.id}/verbatim-items?run_id={foreign_run.id}")
    assert r.status_code == 404, r.text


async def test_run_from_another_project_of_the_same_tenant_returns_404(client, db_session) -> None:
    """A run the caller's tenant owns, but under a DIFFERENT project, must not
    be usable to derive placement for this project's phrases."""
    mine = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    sibling = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    other_run = await _seed_run(db_session, sibling, output=_LINES)

    r = await client.get(f"/api/projects/{mine.id}/verbatim-items?run_id={other_run.id}")
    assert r.status_code == 404, r.text


async def test_slide_item_reports_rejected_not_pending(client, db_session) -> None:
    """A `slide` phrase has no consumer anywhere in the pipeline (every stage in
    STAGE_VERBATIM_KIND consumes `tts`), so it must not sit on the same amber
    "Pending" badge as a tts phrase that simply has not landed yet — even when the
    narration coincidentally contains the words."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    run = await _seed_run(db_session, project, output=_LINES)
    await client.post(
        f"/api/projects/{project.id}/verbatim-items",
        json={"text": "results may vary", "kind": "slide"},
    )
    await client.post(
        f"/api/projects/{project.id}/verbatim-items",
        json={"text": "results may vary", "kind": "tts"},
    )

    r = await client.get(f"/api/projects/{project.id}/verbatim-items?run_id={run.id}")
    assert r.status_code == 200, r.text
    by_kind = {i["kind"]: i for i in r.json()["items"]}

    assert by_kind["slide"]["status"] == "rejected"
    assert by_kind["slide"]["placed_in_stage"] is None
    assert by_kind["slide"]["placed_in_element_id"] is None

    # The tts twin, same text, IS placed — proving the split is by kind.
    assert by_kind["tts"]["status"] == "placed"
    assert by_kind["tts"]["placed_in_stage"] == "script_draft"


async def test_superseded_spine_does_not_report_a_dropped_phrase(client, db_session) -> None:
    """Finding 3 at the API boundary: script_doctor is the spine that ships."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
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
        blueprint_id="demo_video_v1",
        status="completed",
        config_snapshot={},
        pause_after_stages=[],
    )
    db_session.add_all(
        [
            deliverable,
            run,
            StageOutput(
                id=uuid4(),
                run_id=run.id,
                stage_name="demo_video_gen",
                status="completed",
                output={"lines": [{"id": "clip-1", "text": "draft with the disclaimer here"}]},
            ),
            StageOutput(
                id=uuid4(),
                run_id=run.id,
                stage_name="script_doctor",
                status="completed",
                output={"lines": [{"id": "clip-1", "text": "doctored, disclaimer removed"}]},
            ),
        ]
    )
    await db_session.commit()
    await client.post(
        f"/api/projects/{project.id}/verbatim-items",
        json={"text": "the disclaimer"},
    )

    r = await client.get(f"/api/projects/{project.id}/verbatim-items?run_id={run.id}")
    item = r.json()["items"][0]
    assert item["status"] == "pending", "the superseded gen draft must not rescue it"
    assert item["placed_in_stage"] is None
