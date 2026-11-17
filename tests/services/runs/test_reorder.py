"""DB-backed tests for clip-reorder re-stitch (studio timeline).

A reorder spawns a child run that preserves every stage before the blueprint's
compile cut (``terminal_artifact_stage``) with the spine stage's clip array
reordered, and re-runs only the compile stage onward — reusing every generated
clip asset. These tests drive ``build_reorder_state`` directly and end-to-end
through ``RunEngine.start_run(reorder=...)`` against real postgres.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from src.database.models import Deliverable, Project, Run, StageOutput
from src.services.blueprints.registry import get_registry
from src.services.run_engine import RunEngine
from src.services.runs.regenerate import ReorderError, build_reorder_state

pytestmark = pytest.mark.asyncio

GENERIC_STAGES = (
    "brief_intake",
    "script_draft",
    "storyboard_draft",
    "narrate",
    "video_render",
    "ai_image_gen",
    "ai_video_gen",
    "motion_graphics_gen",
    "source_ingest",
    "media_discovery",
)
SHOTS = [
    {"id": "s1", "seconds": 3, "description": "one"},
    {"id": "s2", "seconds": 4, "description": "two"},
    {"id": "s3", "seconds": 5, "description": "three"},
]


async def _seed_parent(db_session) -> tuple[Project, Deliverable, Run]:
    tenant_id, subject_id = uuid4(), uuid4()
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="reorder-parent",
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
    parent = Run(
        id=uuid4(),
        deliverable_id=deliverable.id,
        blueprint_id="generic_v1",
        status="completed",
        config_snapshot={
            "project_id": str(project.id),
            "tenant_id": str(tenant_id),
            "subject_id": str(subject_id),
            "title": project.title,
            "type": "generic",
            "brief": {},
            "brand": {},
            "voice": {},
            "theme": {},
            "deliverable": {
                "id": str(deliverable.id),
                "name": deliverable.name,
                "format": deliverable.format,
                "captions_mode": deliverable.captions_mode,
                "target_duration_seconds": deliverable.target_duration_seconds,
            },
        },
        pause_after_stages=[],
    )
    sos = []
    for name in GENERIC_STAGES:
        output = (
            {"shots": [dict(s) for s in SHOTS]} if name == "storyboard_draft" else {"stage": name}
        )
        sos.append(
            StageOutput(
                id=uuid4(), run_id=parent.id, stage_name=name, status="completed", output=output
            )
        )
    db_session.add_all([project, deliverable, parent, *sos])
    await db_session.commit()
    return project, deliverable, parent


async def test_build_reorder_state_preserves_all_but_compile_and_reorders_spine(db_session) -> None:
    _, _, parent = await _seed_parent(db_session)
    blueprint = get_registry().resolve("generic")

    preserved = await build_reorder_state(
        session=db_session,
        parent_run=parent,
        blueprint=blueprint,
        spine_stage="storyboard_draft",
        ordered_ids=["s3", "s1", "s2"],
    )

    # video_render is the compile cut → NOT preserved (child re-runs it).
    assert "video_render" not in preserved
    assert set(preserved) == set(GENERIC_STAGES) - {"video_render"}
    # Spine reordered; other stages untouched.
    assert [s["id"] for s in preserved["storyboard_draft"]["shots"]] == ["s3", "s1", "s2"]
    assert preserved["script_draft"] == {"stage": "script_draft"}


async def test_build_reorder_state_rejects_bad_inputs(db_session) -> None:
    _, _, parent = await _seed_parent(db_session)
    blueprint = get_registry().resolve("generic")

    with pytest.raises(ReorderError):  # not a permutation
        await build_reorder_state(
            session=db_session,
            parent_run=parent,
            blueprint=blueprint,
            spine_stage="storyboard_draft",
            ordered_ids=["s1", "s2"],
        )
    with pytest.raises(ReorderError):  # spine == terminal, no downstream compile
        await build_reorder_state(
            session=db_session,
            parent_run=parent,
            blueprint=blueprint,
            spine_stage="video_render",
            ordered_ids=["s1", "s2", "s3"],
        )


async def test_run_engine_reorder_restitches_only_compile(db_session, db_session_factory) -> None:
    project, deliverable, parent = await _seed_parent(db_session)
    dispatched: list[UUID] = []

    async def _record_dispatch(run_id: UUID) -> None:
        dispatched.append(run_id)

    engine = RunEngine(
        session_factory=db_session_factory, registry=get_registry(), dispatch=_record_dispatch
    )
    child = await engine.start_run(
        deliverable_id=deliverable.id,
        tenant_id=project.tenant_id,
        subject_id=project.owner_subject_id,
        parent_run_id=parent.id,
        pause_after_stages=[],
        reorder={"spine_stage": "storyboard_draft", "ordered_ids": ["s3", "s1", "s2"]},
    )

    assert dispatched == [child.id]
    assert child.parent_run_id == parent.id

    child_sos = (
        (await db_session.execute(select(StageOutput).where(StageOutput.run_id == child.id)))
        .scalars()
        .all()
    )
    by_name = {so.stage_name: so for so in child_sos}
    # Only the compile stage is pending; everything upstream is preserved.
    assert by_name["video_render"].status == "pending"
    assert by_name["video_render"].output is None
    assert by_name["storyboard_draft"].status == "completed"
    assert [s["id"] for s in by_name["storyboard_draft"].output["shots"]] == ["s3", "s1", "s2"]
    assert by_name["narrate"].status == "completed"
    assert by_name["ai_image_gen"].status == "completed"
