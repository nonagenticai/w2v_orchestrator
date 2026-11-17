"""A child run must NOT inherit a DRAINED approval ledger as "no gates".

``runs.pause_after_stages`` is the durable approval ledger: ``resume_run``
strips each checkpoint the human approved, so a fully-reviewed parent ends at
``[]``. Handing that ``[]`` straight to a regenerate / reorder / clip-edit child
made the child render — at real cost — with ZERO review gates.

These tests drive ``RunEngine.start_run`` against the real postgres backend and
assert the child's persisted ``pause_after_stages``, i.e. the row the DAG
executor actually reads (``dag_executor``: ``if name in pause_after_stages``).
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from src.database.models import Deliverable, Project, Run, StageOutput
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.services.blueprints.registry import get_registry
from src.services.run_engine import RunEngine
from src.services.runs.run_finalizer import terminal_stage_names

pytestmark = pytest.mark.asyncio

# generic_v1's declared review gates (``default_pause_after=True``).
GENERIC_GATES = ("script_draft", "storyboard_draft")

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


async def _seed_parent(
    db_session_factory,
    *,
    pause_after_stages: list[str],
    terminal_stages: tuple[str, ...] = GENERIC_STAGES,
    status: str = "completed",
):
    """Seed a generic parent run.

    ``terminal_stages`` is the set that gets a COMPLETED ``stage_outputs`` row —
    i.e. exactly what ``run_finalizer.terminal_stage_names`` will report as
    "the parent reached this stage". Stages outside it get NO row at all, which
    is how an autopilot run that never reached a gate looks on disk. This is
    load-bearing: it is the only signal separating "the human approved past this
    gate" from "an operator deleted this gate", both of which leave
    ``pause_after_stages == []``.
    """
    tenant_id, subject_id = uuid4(), uuid4()
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="checkpoint-restore-parent",
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
        status=status,
        config_snapshot={"tenant_id": str(tenant_id), "subject_id": str(subject_id)},
        pause_after_stages=list(pause_after_stages),
    )
    outputs = [
        StageOutput(
            id=uuid4(),
            run_id=parent.id,
            stage_name=name,
            status="completed",
            output={"stage": name},
        )
        for name in terminal_stages
    ]
    async with db_session_factory() as session:
        session.add_all([project, deliverable, parent, *outputs])
        await session.commit()
    return tenant_id, subject_id, deliverable.id, parent.id


async def _regenerate_child(
    db_session_factory, *, tenant_id, subject_id, deliverable_id, parent_id, from_stage
) -> Run:
    engine = RunEngine(
        session_factory=db_session_factory,
        registry=get_registry(),
        dispatch=lambda _rid: asyncio.sleep(0),
    )
    return await engine.start_run(
        deliverable_id=deliverable_id,
        tenant_id=tenant_id,
        subject_id=subject_id,
        parent_run_id=parent_id,
        pause_after_stages=[],  # what the route passes for a DRAINED parent
        from_stage=from_stage,
        restore_drained_checkpoints=True,
    )


async def test_drained_parent_child_regains_gate_on_rerun_stage(db_session_factory) -> None:
    """1. Every gate approved (ledger drained to []) → the child still gates the
    stage it re-runs. Without this the regenerated storyboard would flow straight
    into the paid render with no human review."""
    tenant_id, subject_id, deliverable_id, parent_id = await _seed_parent(
        db_session_factory, pause_after_stages=[]
    )
    # PREMISE CHECK: the gate is restorable only because the parent REACHED it.
    # Asserted with the production predicate, so this test cannot pass by
    # accident of how the fixture happens to seed rows — if the seeding stopped
    # producing a terminal row for the gate, this fails here rather than
    # silently turning the assertion below into a test of nothing.
    async with db_session_factory() as session:
        rows = await StageOutputRepository(session).list_for_run(parent_id)
    assert "storyboard_draft" in terminal_stage_names(rows)

    child = await _regenerate_child(
        db_session_factory,
        tenant_id=tenant_id,
        subject_id=subject_id,
        deliverable_id=deliverable_id,
        parent_id=parent_id,
        from_stage="storyboard_draft",
    )
    assert "storyboard_draft" in (child.pause_after_stages or [])


async def test_preserved_stage_is_not_repaused(db_session_factory) -> None:
    """3. ``script_draft`` is upstream of the cut, so its output is PRESERVED —
    it never re-runs, and re-pausing it would strand the child on a gate whose
    stage produces nothing new."""
    tenant_id, subject_id, deliverable_id, parent_id = await _seed_parent(
        db_session_factory, pause_after_stages=[]
    )
    child = await _regenerate_child(
        db_session_factory,
        tenant_id=tenant_id,
        subject_id=subject_id,
        deliverable_id=deliverable_id,
        parent_id=parent_id,
        from_stage="storyboard_draft",
    )
    assert "script_draft" not in (child.pause_after_stages or [])


async def test_undrained_parent_behaviour_unchanged(db_session_factory) -> None:
    """2. REGRESSION FENCE: a parent whose gates were never drained hands its
    full ledger to the child exactly as before. The restored set is a subset of
    the blueprint seed the parent already carries, so the union is a no-op."""
    tenant_id, subject_id, deliverable_id, parent_id = await _seed_parent(
        db_session_factory, pause_after_stages=list(GENERIC_GATES)
    )
    engine = RunEngine(
        session_factory=db_session_factory,
        registry=get_registry(),
        dispatch=lambda _rid: asyncio.sleep(0),
    )
    child = await engine.start_run(
        deliverable_id=deliverable_id,
        tenant_id=tenant_id,
        subject_id=subject_id,
        parent_run_id=parent_id,
        pause_after_stages=list(GENERIC_GATES),  # what the route passes, un-drained
        from_stage="storyboard_draft",
        restore_drained_checkpoints=True,
    )
    assert sorted(child.pause_after_stages or []) == sorted(GENERIC_GATES)


async def test_opt_out_caller_keeps_its_explicit_list(db_session_factory) -> None:
    """A system-spawned child (structural relief) passes ``[]`` deliberately and
    does NOT opt in — it must stay autopilot."""
    tenant_id, subject_id, deliverable_id, parent_id = await _seed_parent(
        db_session_factory, pause_after_stages=[]
    )
    engine = RunEngine(
        session_factory=db_session_factory,
        registry=get_registry(),
        dispatch=lambda _rid: asyncio.sleep(0),
    )
    child = await engine.start_run(
        deliverable_id=deliverable_id,
        tenant_id=tenant_id,
        subject_id=subject_id,
        parent_run_id=parent_id,
        pause_after_stages=[],
        from_stage="storyboard_draft",
    )
    assert list(child.pause_after_stages or []) == []


async def test_autopilot_parent_gate_never_reached_is_not_restored(db_session_factory) -> None:
    """5. REGRESSION FENCE for the operator's deliberate ``[]``.

    ``PATCH /runs/{id}/checkpoints`` with ``pause_after_stages: []`` and
    ``POST /projects`` with ``autopilot: true`` both persist the SAME ``[]`` a
    fully-approved run drains to. Re-arming gates on the child would override a
    deliberate operator decision.

    The discriminator is whether the parent ever REACHED the gate. Here the run
    died at ``brief_intake``: ``script_draft`` / ``storyboard_draft`` have no
    ``stage_outputs`` row at all, so ``terminal_stage_names`` does not report
    them and neither gate may be restored — even though both are in the
    blueprint seed AND in the child's re-run region (which is what makes this a
    real discriminator test rather than a re-run-region test).
    """
    tenant_id, subject_id, deliverable_id, parent_id = await _seed_parent(
        db_session_factory,
        pause_after_stages=[],
        terminal_stages=("brief_intake",),
        status="failed",
    )
    async with db_session_factory() as session:
        rows = await StageOutputRepository(session).list_for_run(parent_id)
    reached = terminal_stage_names(rows)
    assert not reached & set(GENERIC_GATES), "premise: parent reached neither gate"

    child = await _regenerate_child(
        db_session_factory,
        tenant_id=tenant_id,
        subject_id=subject_id,
        deliverable_id=deliverable_id,
        parent_id=parent_id,
        from_stage="script_draft",
    )
    # Both gates are downstream of the cut, so both re-run; neither was reached.
    assert list(child.pause_after_stages or []) == []
