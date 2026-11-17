"""Tests for surgical single-clip regeneration (studio "Regenerate this clip").

A regen-clip spawns a child run that preserves the spine (and every sibling
clip's assets) verbatim and re-runs only the per-clip image stage onward,
folding the clip's open feedback into just that clip's prompt — so exactly one
image regenerates while every other clip is reused from content-addressed
storage. Pure-unit tests cover producer detection + the ai_image_gen note fold;
DB-backed tests drive ``build_regen_clip_state`` and ``RunEngine.start_run``.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from src.database.models import Annotation, Deliverable, Project, Run, StageOutput
from src.services.blueprints.registry import get_registry
from src.services.run_engine import RunEngine
from src.services.runs.regenerate import (
    RegenClipError,
    build_regen_clip_state,
)
from src.services.runs.run_context import RunContext
from src.services.stages.generic.ai_image_gen import AiImageGenStage

# ── Pure unit: image-producer detection ────────────────────────────────────


# ── Pure unit: ai_image_gen per-line note fold ─────────────────────────────


def _ctx(open_annotations: dict) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        open_annotations_by_stage=open_annotations,
    )


def test_apply_clip_notes_folds_into_matching_line_only() -> None:
    prompts: list[tuple[str, str | None]] = [("a sunset", "L1"), ("a city", "L2")]
    ctx = _ctx({"ai_image_gen": [{"text": "make it brighter", "element_id": "L1"}]})
    out = AiImageGenStage._apply_clip_notes(prompts, ctx)
    assert out[0][0] == "a sunset\n\nApply this reviewer feedback: make it brighter"
    assert out[0][1] == "L1"
    # Sibling untouched → same prompt → same media id → reused.
    assert out[1] == ("a city", "L2")


def test_apply_clip_notes_noop_without_matching_line() -> None:
    prompts: list[tuple[str, str | None]] = [("a sunset", "L1"), ("a city", None)]
    # Note for a line_id not present, and a static (line_id=None) prompt.
    ctx = _ctx({"ai_image_gen": [{"text": "x", "element_id": "L9"}]})
    assert AiImageGenStage._apply_clip_notes(prompts, ctx) == prompts
    assert AiImageGenStage._apply_clip_notes(prompts, _ctx({})) == prompts


# ── DB-backed: build_regen_clip_state + engine ─────────────────────────────

PROMO_STAGES = (
    "promo_analyze",
    "promo_script",
    "promo_storyboard",
    "promo_image_prompts",
    "ai_image_gen",
    "promo_narrate",
    "promo_render",
)
SHOTS = [
    {"id": "s1", "line_id": "L1", "seconds": 3, "description": "one"},
    {"id": "s2", "line_id": "L2", "seconds": 4, "description": "two"},
]


async def _seed_promo_parent(
    db_session,
    *,
    prompts_from_stage: str | None = "promo_image_prompts",
    note_on: str | None = "s1",
) -> tuple[Project, Deliverable, Run, dict[str, StageOutput]]:
    tenant_id, subject_id = uuid4(), uuid4()
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="promo",
        title="regen-clip-parent",
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
    ai_cfg = {"prompts_from_stage": prompts_from_stage} if prompts_from_stage else {}
    parent = Run(
        id=uuid4(),
        deliverable_id=deliverable.id,
        blueprint_id="promo_v1",
        status="completed",
        config_snapshot={
            "project_id": str(project.id),
            "tenant_id": str(tenant_id),
            "subject_id": str(subject_id),
            "title": project.title,
            "type": "promo",
            "brief": {},
            "brand": {},
            "voice": {},
            "theme": {},
            "ai_image_gen": ai_cfg,
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
    by_name: dict[str, StageOutput] = {}
    for name in PROMO_STAGES:
        output = (
            {"shots": [dict(s) for s in SHOTS]} if name == "promo_storyboard" else {"stage": name}
        )
        so = StageOutput(
            id=uuid4(), run_id=parent.id, stage_name=name, status="completed", output=output
        )
        by_name[name] = so
    notes = (
        [
            Annotation(
                id=uuid4(),
                stage_output_id=by_name["promo_storyboard"].id,
                element_id=note_on,
                text="make it brighter",
                status="open",
            )
        ]
        if note_on is not None
        else []
    )
    db_session.add_all([project, deliverable, parent, *by_name.values(), *notes])
    await db_session.commit()
    return project, deliverable, parent, by_name


async def test_build_regen_clip_state_preserves_spine_and_folds_note_at_image_stage(
    db_session,
) -> None:
    _, _, parent, by_name = await _seed_promo_parent(db_session)
    blueprint = get_registry().resolve("promo")

    preserved, notes_by_stage, affected = await build_regen_clip_state(
        session=db_session,
        parent_run=parent,
        blueprint=blueprint,
        spine_stage="promo_storyboard",
        element_id="s1",
    )

    # Spine preserved verbatim (siblings untouched); image stage + compile re-run.
    assert "promo_storyboard" in preserved
    assert [s["id"] for s in preserved["promo_storyboard"]["shots"]] == ["s1", "s2"]
    assert "promo_image_prompts" in preserved
    assert "ai_image_gen" not in preserved
    assert "promo_render" not in preserved
    # Feedback folded into the image stage, keyed by the clip's line_id (not shot id).
    assert notes_by_stage == {"ai_image_gen": [{"text": "make it brighter", "element_id": "L1"}]}
    # The storyboard annotations flip to applied.
    assert affected == [by_name["promo_storyboard"].id]


async def test_build_regen_clip_state_rejects_unsurgical_cases(db_session) -> None:
    blueprint = get_registry().resolve("promo")

    # No open feedback on the clip.
    _, _, p_none, _ = await _seed_promo_parent(db_session, note_on=None)
    with pytest.raises(RegenClipError):
        await build_regen_clip_state(
            session=db_session,
            parent_run=p_none,
            blueprint=blueprint,
            spine_stage="promo_storyboard",
            element_id="s1",
        )

    # Image stage has no per-line prompts (static round-robin) → fall back.
    _, _, p_static, _ = await _seed_promo_parent(db_session, prompts_from_stage=None)
    with pytest.raises(RegenClipError):
        await build_regen_clip_state(
            session=db_session,
            parent_run=p_static,
            blueprint=blueprint,
            spine_stage="promo_storyboard",
            element_id="s1",
        )

    # Unknown element id → no line_id → refuse.
    _, _, p_ok, _ = await _seed_promo_parent(db_session)
    with pytest.raises(RegenClipError):
        await build_regen_clip_state(
            session=db_session,
            parent_run=p_ok,
            blueprint=blueprint,
            spine_stage="promo_storyboard",
            element_id="does-not-exist",
        )


async def test_run_engine_regen_clip_reruns_only_image_and_compile(
    db_session, db_session_factory
) -> None:
    project, deliverable, parent, _ = await _seed_promo_parent(db_session)
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
        regen_clip={"spine_stage": "promo_storyboard", "element_id": "s1"},
    )

    assert dispatched == [child.id]
    assert child.parent_run_id == parent.id

    child_sos = (
        (await db_session.execute(select(StageOutput).where(StageOutput.run_id == child.id)))
        .scalars()
        .all()
    )
    by_name = {so.stage_name: so for so in child_sos}
    # Spine + sibling assets preserved; only the image stage onward re-runs.
    assert by_name["promo_storyboard"].status == "completed"
    assert [s["id"] for s in by_name["promo_storyboard"].output["shots"]] == ["s1", "s2"]
    assert by_name["ai_image_gen"].status == "pending"
    assert by_name["ai_image_gen"].output is None
    assert by_name["promo_render"].status == "pending"
    # The regenerate note is carried on the child's config for the image stage.
    assert child.config_snapshot["regenerate_notes"] == {
        "ai_image_gen": [{"text": "make it brighter", "element_id": "L1"}]
    }


async def test_run_engine_regen_clip_applies_only_target_clip_feedback(
    db_session, db_session_factory
) -> None:
    """Surgical single-clip regen must flip ONLY the target clip's open feedback.

    The storyboard stage_output carries open annotations for BOTH clips (s1 + s2).
    Regenerating s1 must apply s1's note (→ ``applied`` pinned to the child) while
    s2's note stays ``open`` — otherwise a later regen of s2 would find its
    feedback already consumed. Guards against the wholesale ``bulk_apply`` bug.
    """
    project, deliverable, parent, by_name = await _seed_promo_parent(db_session, note_on="s1")
    storyboard_so_id = by_name["promo_storyboard"].id
    # Add a SECOND open annotation on the sibling clip s2 (same stage_output).
    db_session.add(
        Annotation(
            id=uuid4(),
            stage_output_id=storyboard_so_id,
            element_id="s2",
            text="sibling note",
            status="open",
        )
    )
    await db_session.commit()

    async def _record_dispatch(run_id: UUID) -> None:
        return None

    engine = RunEngine(
        session_factory=db_session_factory, registry=get_registry(), dispatch=_record_dispatch
    )
    child = await engine.start_run(
        deliverable_id=deliverable.id,
        tenant_id=project.tenant_id,
        subject_id=project.owner_subject_id,
        parent_run_id=parent.id,
        pause_after_stages=[],
        regen_clip={"spine_stage": "promo_storyboard", "element_id": "s1"},
    )

    anns = (
        (
            await db_session.execute(
                select(Annotation).where(Annotation.stage_output_id == storyboard_so_id)
            )
        )
        .scalars()
        .all()
    )
    by_clip = {a.element_id: a for a in anns}
    # Target clip s1: flipped to applied, pinned to the child run.
    assert by_clip["s1"].status == "applied"
    assert by_clip["s1"].applied_in_run_id == child.id
    # Sibling clip s2: still open, untouched.
    assert by_clip["s2"].status == "open"
    assert by_clip["s2"].applied_in_run_id is None
