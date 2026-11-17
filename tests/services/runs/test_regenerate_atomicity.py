"""DB-backed integration tests for regenerate atomicity.

Spec invariant (§4 #4 + §8.6): when a child run is started with
``parent_run_id``, three things commit together:
  1. the new ``runs`` row,
  2. ``stage_outputs`` rows for the child (preserved-as-completed for stages
     before the first dirty stage, pending for the rest),
  3. every open annotation on the parent's affected stage_outputs flipped to
     ``status='applied'`` with ``applied_in_run_id=child.id``.

These tests exercise the helpers directly + drive the RunEngine end-to-end so
the atomicity guarantee is validated against the real postgres backend.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from src.database.models import Annotation, Deliverable, Project, Run, StageOutput
from src.services.blueprints.base import DAGBlueprint, StageSpec
from src.services.blueprints.registry import get_registry
from src.services.run_engine import RunEngine
from src.services.runs.regenerate import (
    RegenerateError,
    apply_open_annotations_in_transaction,
    build_regenerate_state,
)
from src.services.stages.base import StageHandler, StageInput
from src.services.stages.base import StageOutput as StageHandlerOutput

pytestmark = pytest.mark.asyncio


class _NoopIn(StageInput):
    pass


class _NoopOut(StageHandlerOutput):
    pass


class _NoopHandler(StageHandler[_NoopIn, _NoopOut]):
    name = "noop"
    input_model = _NoopIn
    output_model = _NoopOut

    async def handle(self, payload: _NoopIn, ctx: object) -> _NoopOut:
        return _NoopOut()


def _noop_factory(_deps: dict) -> _NoopHandler:
    return _NoopHandler()


def _parallel_root_blueprint() -> DAGBlueprint:
    """demo_video-shaped DAG with a PARALLEL ROOT.

    ``gen`` and ``ingest`` are both no-dependency roots; ``ingest`` (a stand-in
    for a paid ``source_ingest`` KAG crawl) is Kahn-sorted AFTER ``gen`` without
    depending on it, and ``tail`` joins both branches::

        gen ─▶ mid ─▶ tail
        ingest ───────▲
    """
    return DAGBlueprint(
        id="parallel_root_v1",
        project_type="parallel_root_test",
        stages=[
            StageSpec(name="gen", handler_factory=_noop_factory),
            StageSpec(name="ingest", handler_factory=_noop_factory),
            StageSpec(name="mid", depends_on=("gen",), handler_factory=_noop_factory),
            StageSpec(name="tail", depends_on=("mid", "ingest"), handler_factory=_noop_factory),
        ],
    )


async def _seed_parent_with_stages(
    db_session,
    *,
    stage_names: tuple[str, ...],
    blueprint_id: str = "parallel_root_v1",
    dirty_stage: str | None = None,
    open_annotations: int = 0,
) -> tuple[Project, Deliverable, Run, dict[str, StageOutput]]:
    """Seed a paused parent run with ``stage_names`` all completed.

    Optional ``open_annotations`` open annotations attach to ``dirty_stage``.
    Mirrors ``_seed_completed_parent_run`` but for an arbitrary stage set so the
    tests can exercise custom (parallel-root / linear) blueprints.
    """
    tenant_id = uuid4()
    subject_id = uuid4()
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="regenerate-custom-parent",
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
        blueprint_id=blueprint_id,
        status="paused",
        config_snapshot={"tenant_id": str(tenant_id), "subject_id": str(subject_id)},
        pause_after_stages=[],
    )
    sos = {
        name: StageOutput(
            id=uuid4(),
            run_id=parent.id,
            stage_name=name,
            status="completed",
            output={"stage": name, "demo": True},
        )
        for name in stage_names
    }
    db_session.add_all([project, deliverable, parent, *sos.values()])
    await db_session.flush()

    if dirty_stage is not None:
        for i in range(open_annotations):
            db_session.add(
                Annotation(
                    id=uuid4(),
                    stage_output_id=sos[dirty_stage].id,
                    element_id=None,
                    text=f"note {i}",
                    status="open",
                )
            )
    await db_session.commit()
    return project, deliverable, parent, sos


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


async def _seed_completed_parent_run(
    db_session,
    *,
    tenant_id: UUID | None = None,
    subject_id: UUID | None = None,
    dirty_stage: str = "storyboard_draft",
    open_annotations: int = 2,
) -> tuple[Project, Deliverable, Run, dict[str, StageOutput]]:
    """Seed a paused parent run with all 4 generic stages completed.

    ``open_annotations`` open annotations are attached to the parent's
    ``dirty_stage`` stage_output. Returns ``(project, deliverable, parent_run,
    {stage_name: stage_output})``.
    """
    tenant_id = tenant_id or uuid4()
    subject_id = subject_id or uuid4()
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="regenerate-parent",
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
        status="paused",
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
    sos = {
        name: StageOutput(
            id=uuid4(),
            run_id=parent.id,
            stage_name=name,
            status="completed",
            output={"stage": name, "demo": True},
        )
        for name in GENERIC_STAGES
    }
    db_session.add_all([project, deliverable, parent, *sos.values()])
    await db_session.flush()

    for i in range(open_annotations):
        db_session.add(
            Annotation(
                id=uuid4(),
                stage_output_id=sos[dirty_stage].id,
                element_id=None,
                text=f"note {i}",
                status="open",
            )
        )
    await db_session.commit()
    return project, deliverable, parent, sos


async def test_build_regenerate_state_partitions_at_first_dirty_stage(db_session) -> None:
    _, _, parent, sos = await _seed_completed_parent_run(db_session, dirty_stage="storyboard_draft")
    blueprint = get_registry().resolve("generic")

    preserved, notes_by_stage, affected = await build_regenerate_state(
        session=db_session, parent_run=parent, blueprint=blueprint
    )

    # Re-run region = descendant closure of the dirty storyboard_draft = {storyboard_draft,
    # ai_image_gen, video_render} — ai_image_gen renders storyboard_draft's per-line
    # prompts, so it is downstream of it. Everything else completed is preserved —
    # INCLUDING the sibling branches narrate / ai_video_gen / motion_graphics_gen (they
    # depend only on script_draft, NOT on storyboard_draft, so a storyboard regenerate
    # must not re-run them) plus the parallel roots source_ingest + media_discovery.
    assert set(preserved.keys()) == set(GENERIC_STAGES) - {
        "storyboard_draft",
        "ai_image_gen",
        "video_render",
    }
    assert preserved["script_draft"] == {"stage": "script_draft", "demo": True}
    assert list(notes_by_stage.keys()) == ["storyboard_draft"]
    assert len(notes_by_stage["storyboard_draft"]) == 2
    assert affected == [sos["storyboard_draft"].id]


async def test_build_regenerate_state_no_notes_returns_empty(db_session) -> None:
    _, _, parent, _ = await _seed_completed_parent_run(db_session, open_annotations=0)
    blueprint = get_registry().resolve("generic")

    preserved, notes_by_stage, affected = await build_regenerate_state(
        session=db_session, parent_run=parent, blueprint=blueprint
    )

    assert preserved == {}
    assert notes_by_stage == {}
    assert affected == []


#: Descendant closure of generic ``storyboard_draft``: ai_image_gen renders its
#: per-line prompts, video_render composites both.
_STORYBOARD_REGION = {"storyboard_draft", "ai_image_gen", "video_render"}


async def test_build_regenerate_state_from_stage_forces_partition(db_session) -> None:
    """``from_stage`` (the 'rerun this stage' action) preserves exactly the
    stages strictly before it, independent of where annotations sit."""
    # Annotations live on a LATE stage (video_render), but from_stage cuts at
    # storyboard_draft → only stages strictly before storyboard_draft preserved.
    _, _, parent, _ = await _seed_completed_parent_run(
        db_session, dirty_stage="video_render", open_annotations=1
    )
    blueprint = get_registry().resolve("generic")

    preserved, notes_by_stage, affected = await build_regenerate_state(
        session=db_session, parent_run=parent, blueprint=blueprint, from_stage="storyboard_draft"
    )

    # Descendant-region of storyboard_draft = {storyboard_draft, ai_image_gen,
    # video_render}; every other completed stage (siblings + roots) is preserved.
    assert set(preserved.keys()) == set(GENERIC_STAGES) - _STORYBOARD_REGION
    # video_render is a descendant of storyboard_draft, so its note IS folded + flipped.
    assert list(notes_by_stage.keys()) == ["video_render"]
    assert len(affected) == 1


async def test_build_regenerate_state_from_stage_preserves_with_no_notes(db_session) -> None:
    """Unlike the annotation-driven path (no notes → re-run everything),
    ``from_stage`` with no open feedback still preserves the earlier stages."""
    _, _, parent, _ = await _seed_completed_parent_run(db_session, open_annotations=0)
    blueprint = get_registry().resolve("generic")

    preserved, notes_by_stage, affected = await build_regenerate_state(
        session=db_session, parent_run=parent, blueprint=blueprint, from_stage="storyboard_draft"
    )

    assert set(preserved.keys()) == set(GENERIC_STAGES) - _STORYBOARD_REGION
    assert notes_by_stage == {}
    assert affected == []


async def test_build_regenerate_state_from_stage_keeps_upstream_notes_open(db_session) -> None:
    """A note on a stage BEFORE from_stage is not applied: that stage is
    preserved verbatim, so its feedback stays open for a later pass."""
    _, _, parent, _ = await _seed_completed_parent_run(
        db_session, dirty_stage="brief_intake", open_annotations=2
    )
    blueprint = get_registry().resolve("generic")

    preserved, notes_by_stage, affected = await build_regenerate_state(
        session=db_session, parent_run=parent, blueprint=blueprint, from_stage="storyboard_draft"
    )

    # brief_intake is preserved (before the cut) → its notes are NOT folded/flipped.
    assert "brief_intake" in preserved
    assert notes_by_stage == {}
    assert affected == []


async def test_build_regenerate_state_from_stage_unknown_raises(db_session) -> None:
    _, _, parent, _ = await _seed_completed_parent_run(db_session, open_annotations=0)
    blueprint = get_registry().resolve("generic")

    with pytest.raises(RegenerateError, match="not a stage of this run's blueprint"):
        await build_regenerate_state(
            session=db_session, parent_run=parent, blueprint=blueprint, from_stage="does_not_exist"
        )


async def test_build_regenerate_state_preserves_parallel_root(db_session) -> None:
    """A no-dependency parallel root (e.g. a paid ``source_ingest`` crawl) is
    Kahn-sorted AFTER the gen stage but does NOT depend on it, so a regenerate
    from gen must PRESERVE it — not re-run (re-bill) it every pass. This is the
    exact bug the descendant-region partition fixes.
    """
    blueprint = _parallel_root_blueprint()
    # Pure-method proof: ingest is a parallel root, not a descendant of gen.
    assert blueprint.descendants("gen") == {"mid", "tail"}
    assert "ingest" not in blueprint.descendants("gen")

    _, _, parent, _ = await _seed_parent_with_stages(
        db_session, stage_names=("gen", "ingest", "mid", "tail")
    )

    preserved, _notes, _affected = await build_regenerate_state(
        session=db_session, parent_run=parent, blueprint=blueprint, from_stage="gen"
    )

    # ingest is preserved (not re-run); gen + its descendant subtree are not.
    assert "ingest" in preserved
    assert set(preserved.keys()).isdisjoint({"gen", "mid", "tail"})


async def test_build_regenerate_state_linear_chain_regression(db_session) -> None:
    """On a pure linear chain the descendant-region partition is byte-identical
    to the old topological-prefix behaviour: from_stage='c' preserves exactly
    {a, b}."""
    blueprint = DAGBlueprint(
        id="linear_v1",
        project_type="linear_test",
        stages=[
            StageSpec(name="a", handler_factory=_noop_factory),
            StageSpec(name="b", depends_on=("a",), handler_factory=_noop_factory),
            StageSpec(name="c", depends_on=("b",), handler_factory=_noop_factory),
            StageSpec(name="d", depends_on=("c",), handler_factory=_noop_factory),
        ],
    )
    # descendants(c) == the topological suffix strictly after c on a linear chain.
    assert blueprint.descendants("c") == {"d"}

    _, _, parent, _ = await _seed_parent_with_stages(
        db_session, stage_names=("a", "b", "c", "d"), blueprint_id="linear_v1"
    )

    preserved, _notes, _affected = await build_regenerate_state(
        session=db_session, parent_run=parent, blueprint=blueprint, from_stage="c"
    )

    assert set(preserved.keys()) == {"a", "b"}


async def test_build_regenerate_state_annotation_driven_parallel_root(db_session) -> None:
    """Annotation-driven regenerate with a parallel root: a note on ``mid`` re-runs
    mid + its descendant tail, keeps the parallel root ``ingest`` and the upstream
    ``gen`` preserved, and keys notes/affected to the re-run region only."""
    blueprint = _parallel_root_blueprint()
    _, _, parent, sos = await _seed_parent_with_stages(
        db_session,
        stage_names=("gen", "ingest", "mid", "tail"),
        dirty_stage="mid",
        open_annotations=2,
    )

    preserved, notes_by_stage, affected = await build_regenerate_state(
        session=db_session, parent_run=parent, blueprint=blueprint
    )

    assert "ingest" in preserved
    assert "gen" in preserved
    assert set(preserved.keys()).isdisjoint({"mid", "tail"})
    assert list(notes_by_stage.keys()) == ["mid"]
    assert len(notes_by_stage["mid"]) == 2
    assert affected == [sos["mid"].id]


async def test_apply_open_annotations_flips_open_to_applied(db_session) -> None:
    _, deliverable, parent, sos = await _seed_completed_parent_run(
        db_session, dirty_stage="script_draft", open_annotations=3
    )
    # apply_open_annotations_in_transaction's UPDATE sets applied_in_run_id,
    # which is a FK into runs.id. Seed a real child run row so the FK holds.
    child = Run(
        id=uuid4(),
        deliverable_id=deliverable.id,
        parent_run_id=parent.id,
        blueprint_id="generic_v1",
        status="pending",
        config_snapshot={},
        pause_after_stages=[],
    )
    db_session.add(child)
    await db_session.flush()

    flipped = await apply_open_annotations_in_transaction(
        session=db_session,
        affected_stage_output_ids=[sos["script_draft"].id],
        new_run_id=child.id,
    )
    await db_session.commit()
    assert flipped == 3

    rows = (
        (
            await db_session.execute(
                select(Annotation).where(Annotation.stage_output_id == sos["script_draft"].id)
            )
        )
        .scalars()
        .all()
    )
    assert {a.status for a in rows} == {"applied"}
    assert {a.applied_in_run_id for a in rows} == {child.id}

    # Re-running is a no-op once nothing is left in 'open' status.
    second_child = Run(
        id=uuid4(),
        deliverable_id=deliverable.id,
        parent_run_id=parent.id,
        blueprint_id="generic_v1",
        status="pending",
        config_snapshot={},
        pause_after_stages=[],
    )
    db_session.add(second_child)
    await db_session.flush()
    again = await apply_open_annotations_in_transaction(
        session=db_session,
        affected_stage_output_ids=[sos["script_draft"].id],
        new_run_id=second_child.id,
    )
    await db_session.commit()
    assert again == 0


async def test_apply_open_annotations_empty_list_short_circuits(db_session) -> None:
    flipped = await apply_open_annotations_in_transaction(
        session=db_session,
        affected_stage_output_ids=[],
        new_run_id=uuid4(),
    )
    assert flipped == 0


async def test_run_engine_start_run_atomic_regenerate(db_session, db_session_factory) -> None:
    """End-to-end: child run row + preserved stage_outputs + applied
    annotations all visible after RunEngine.start_run commits.
    """
    project, deliverable, parent, parent_sos = await _seed_completed_parent_run(
        db_session, dirty_stage="storyboard_draft", open_annotations=2
    )

    dispatched: list[UUID] = []

    async def _record_dispatch(run_id: UUID) -> None:
        dispatched.append(run_id)

    engine = RunEngine(
        session_factory=db_session_factory,
        registry=get_registry(),
        dispatch=_record_dispatch,
    )

    child = await engine.start_run(
        deliverable_id=deliverable.id,
        tenant_id=project.tenant_id,
        subject_id=project.owner_subject_id,
        parent_run_id=parent.id,
        pause_after_stages=[],
    )

    assert dispatched == [child.id]
    assert child.parent_run_id == parent.id

    # Child run row exists, persisted.
    persisted_child = (
        await db_session.execute(select(Run).where(Run.id == child.id))
    ).scalar_one_or_none()
    assert persisted_child is not None
    assert persisted_child.status == "pending"

    # Child stage outputs: preserved-as-completed for the two stages before
    # the dirty one; pending for storyboard_draft + video_render.
    child_sos = (
        (await db_session.execute(select(StageOutput).where(StageOutput.run_id == child.id)))
        .scalars()
        .all()
    )
    by_name = {so.stage_name: so for so in child_sos}
    assert set(by_name) == set(GENERIC_STAGES)
    assert by_name["brief_intake"].status == "completed"
    assert by_name["brief_intake"].output == {"stage": "brief_intake", "demo": True}
    assert by_name["script_draft"].status == "completed"
    assert by_name["storyboard_draft"].status == "pending"
    assert by_name["storyboard_draft"].output is None
    assert by_name["video_render"].status == "pending"

    # Parent's open annotations are now applied, pinned to the child id.
    applied = (
        (
            await db_session.execute(
                select(Annotation).where(
                    Annotation.stage_output_id == parent_sos["storyboard_draft"].id
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(applied) == 2
    assert {a.status for a in applied} == {"applied"}
    assert {a.applied_in_run_id for a in applied} == {child.id}

    # regenerate notes folded into child config_snapshot for the dispatcher.
    notes = persisted_child.config_snapshot.get("regenerate_notes")
    assert notes is not None
    assert list(notes.keys()) == ["storyboard_draft"]
    assert len(notes["storyboard_draft"]) == 2


async def test_run_engine_rejects_running_parent(db_session, db_session_factory) -> None:
    project, deliverable, parent, _ = await _seed_completed_parent_run(db_session)
    parent.status = "running"
    await db_session.commit()

    engine = RunEngine(
        session_factory=db_session_factory,
        registry=get_registry(),
        dispatch=lambda _rid: _noop(),
    )
    with pytest.raises(ValueError, match=r"paused/completed/failed/cancelled \(is running\)"):
        await engine.start_run(
            deliverable_id=deliverable.id,
            tenant_id=project.tenant_id,
            subject_id=project.owner_subject_id,
            parent_run_id=parent.id,
            pause_after_stages=[],
        )


async def _noop() -> None:  # pragma: no cover - helper for the lambda above
    return None
