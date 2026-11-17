"""The stage-output contract enforced at the DAG executor's chokepoint.

`_enforce_produces` runs between `result.model_dump()` and
`persist(..., "completed", ...)` — the two lines every stage of every blueprint
records its success through. These tests pin all four behaviours: degrade by
default, fail when required, stay silent when the stage produced something, and
stay silent for stages that declare nothing.

The emptiness tests matter as much as the violation tests: treating `0` or
`False` as "produced nothing" would fail stages that legitimately produce a
zero, which is a worse bug than the one this closes.
"""

from __future__ import annotations

import contextlib
import inspect
from typing import ClassVar
from uuid import uuid4

import pytest
from pydantic import Field

from src.services.blueprints.base import DAGBlueprint, StageSpec
from src.services.runs.dag_executor import DAGExecutor
from src.services.runs.run_context import RunContext
from src.services.stages.action_reel.compile import ActionReelCompileOutput
from src.services.stages.action_reel.cut_plan import CutPlanOutput
from src.services.stages.app_review.compile import AppReviewCompileOutput
from src.services.stages.audiobook.narrate import AudiobookNarrateOutput
from src.services.stages.audiobook.plan import AudiobookPlanOutput
from src.services.stages.audiobook.stitch import AudiobookStitchOutput
from src.services.stages.audit_video.models import AuditCompileOutput, AuditNarrateOutput
from src.services.stages.base import (
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StageProducedNothingError,
    missing_produced_fields,
)
from src.services.stages.course.stitch import CourseStitchOutput
from src.services.stages.demo_video.compile import DemoVideoCompileOutput
from src.services.stages.demo_video.encode_master import EncodeMasterOutput
from src.services.stages.demo_video.final_critique import FinalCritiqueOutput
from src.services.stages.demo_video.narrate import DemoVideoNarrateOutput
from src.services.stages.generic.storyboard_draft import StoryboardDraftOutput
from src.services.stages.generic.video_render import VideoRenderOutput
from src.services.stages.lesson.compile import LessonCompileOutput
from src.services.stages.promo.render import PromoRenderOutput
from src.services.stages.promo.storyboard import PromoStoryboardOutput
from src.services.stages.shared.footage_critique import FootageCritiqueOutput
from src.services.stages.shared.footage_direction import FootageDirectionOutput
from src.services.stages.website_review.compile import WebsiteReviewCompileOutput


class _Declaring(StageOutputWithNotes):
    produces: ClassVar[tuple[str, ...]] = ("lines",)
    lines: list[str] = Field(default_factory=list)


class _Required(StageOutputWithNotes):
    produces: ClassVar[tuple[str, ...]] = ("artifact_url",)
    produces_required: ClassVar[bool] = True
    artifact_url: str = ""


class _Silent(StageOutput):
    """Declares nothing — the pre-existing majority."""

    lines: list[str] = Field(default_factory=list)


class _Counting(StageOutputWithNotes):
    produces: ClassVar[tuple[str, ...]] = ("count",)
    count: int = 0


def _enforce(model: type[StageOutput], output: dict[str, object]) -> None:
    DAGExecutor._enforce_produces("some_stage", model, output)


def test_empty_declared_field_degrades_with_a_note() -> None:
    output: dict[str, object] = {"lines": [], "notes": []}

    _enforce(_Declaring, output)

    notes = output["notes"]
    assert isinstance(notes, list)
    assert len(notes) == 1
    note = str(notes[0])
    assert "producing" in note and "lines" in note


def test_a_produced_field_is_silent() -> None:
    output: dict[str, object] = {"lines": ["a line"], "notes": []}

    _enforce(_Declaring, output)

    assert output["notes"] == []


def test_required_stage_raises_instead_of_degrading() -> None:
    output: dict[str, object] = {"artifact_url": "", "notes": []}

    with pytest.raises(StageProducedNothingError) as exc:
        _enforce(_Required, output)

    assert exc.value.empty_fields == ("artifact_url",)
    assert "some_stage" in str(exc.value)
    # It raised rather than degrading, so nothing was appended.
    assert output["notes"] == []


def test_undeclared_stage_is_never_touched() -> None:
    """The 70-odd stages that declare nothing must behave exactly as before."""
    output: dict[str, object] = {"lines": [], "notes": []}

    _enforce(_Silent, output)

    assert output["notes"] == []


def test_zero_is_a_real_value_not_an_empty_one() -> None:
    """A stage that legitimately produces 0 must not be called empty."""
    output: dict[str, object] = {"count": 0, "notes": []}

    _enforce(_Counting, output)

    assert output["notes"] == []


def test_a_renamed_field_reads_as_missing() -> None:
    """A declared name absent from the dump is drift — exactly what to catch."""
    output: dict[str, object] = {"renamed_lines": ["a"], "notes": []}

    _enforce(_Declaring, output)

    notes = output["notes"]
    assert isinstance(notes, list)
    assert len(notes) == 1


@pytest.mark.parametrize(
    ("value", "empty"),
    [(None, True)],
)
def test_emptiness_rule(value: object, empty: bool) -> None:
    missing = missing_produced_fields({"f": value}, ("f",))
    assert bool(missing) is empty


# ── The WIRING, not just the helper ───────────────────────────────────────────
# Every test above calls `_enforce_produces` directly, so deleting its call site
# in `_run_one` leaves them all green — a gate that is present but unreachable.
# These two drive a real stage through the executor instead, so the chokepoint
# call itself is what is under test.


class _EmptyIn(StageInput):
    pass


def _stage_returning(model: type[StageOutput], **kwargs: object) -> type[StageHandler]:
    async def _handle(self: object, payload: object, ctx: object) -> StageOutput:
        return model(**kwargs)

    return type(
        "_ProducesStage",
        (StageHandler,),
        {
            "name": "s",
            "input_model": _EmptyIn,
            "output_model": model,
            "handle": _handle,
        },
    )


async def _run_single_stage(handler_cls: type[StageHandler]) -> list[dict[str, object]]:
    """Run a one-stage blueprint, returning what was persisted as completed."""
    persisted: list[dict[str, object]] = []

    async def emit(event: str, **kw: object) -> None:
        return None

    async def persist(
        stage_name: str,
        status: str,
        output: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        if status == "completed" and output is not None:
            persisted.append(output)

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[StageSpec(name="s", handler_factory=lambda deps: handler_cls())],
    )
    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    await executor.run(ctx, pause_after_stages=set())
    return persisted


@pytest.mark.asyncio
async def test_executor_degrades_an_empty_stage_before_persisting_it() -> None:
    """The note must be ON the persisted output — that is what makes a run degraded."""
    persisted = await _run_single_stage(_stage_returning(_Declaring, lines=[]))

    assert len(persisted) == 1
    notes = persisted[0]["notes"]
    assert isinstance(notes, list)
    assert len(notes) == 1, "the executor did not enforce the contract before persisting"
    assert "producing" in str(notes[0])


@pytest.mark.asyncio
async def test_executor_persists_a_real_output_untouched() -> None:
    persisted = await _run_single_stage(_stage_returning(_Declaring, lines=["real"]))

    assert len(persisted) == 1
    assert persisted[0]["notes"] == []


# ── The seven REAL terminal artifact producers (W1-1) ─────────────────────────
# The tests above use throwaway models, so they prove the mechanism works while
# proving nothing about whether any shipping stage is wired into it. Until W1-1
# not one production model set `produces_required`, so the whole contract could
# only ever append a note — a compile stage that produced NO artifact still
# reached `completed`. These drive the actual `DemoVideoCompileOutput` /
# `VideoRenderOutput` classes through the executor and assert the run FAILS.
#
# `model_construct` is deliberate: `artifact_url` is an `ArtifactUrl`, whose
# AfterValidator already rejects `""` at construction, so a normally-built empty
# output is unreachable. The reachable shapes are the ones that skip validation —
# a renamed/dropped field (the dump no longer carries `artifact_url` at all) and
# any `model_construct` path — and `model_construct` reproduces both without
# pretending the validator does not exist. The two guards are complementary:
# ArtifactUrl catches a bad value, `produces_required` catches a MISSING one.


def _stage_returning_unvalidated(model: type[StageOutput], **kwargs: object) -> type[StageHandler]:
    """A stage whose output bypasses pydantic validation — see the note above."""

    async def _handle(self: object, payload: object, ctx: object) -> StageOutput:
        # `None` is passed positionally for `_fields_set`: without it the type
        # checker matches our first `**kwargs` entry against that positional
        # parameter and rejects it.
        return model.model_construct(None, **kwargs)

    return type(
        "_UnvalidatedProducesStage",
        (StageHandler,),
        {
            "name": "s",
            "input_model": _EmptyIn,
            "output_model": model,
            "handle": _handle,
        },
    )


async def _run_capturing_statuses(
    handler_cls: type[StageHandler],
) -> tuple[list[tuple[str, str, str | None]], BaseException | None]:
    """Run a one-stage blueprint, returning every persisted status + what escaped."""
    statuses: list[tuple[str, str, str | None]] = []

    async def emit(event: str, **kw: object) -> None:
        statuses.append(("__event__", event, None))

    async def persist(
        stage_name: str,
        status: str,
        output: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        statuses.append((stage_name, status, error))

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[StageSpec(name="s", handler_factory=lambda deps: handler_cls())],
    )
    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    raised: BaseException | None = None
    try:
        await executor.run(ctx, pause_after_stages=set())
    except BaseException as exc:  # noqa: BLE001 — the escape IS the assertion
        raised = exc
    return statuses, raised


@pytest.mark.asyncio
async def test_demo_video_compile_with_no_artifact_fails_the_run() -> None:
    handler = _stage_returning_unvalidated(
        DemoVideoCompileOutput, artifact_url="", duration_seconds=0.0, notes=[]
    )

    statuses, raised = await _run_capturing_statuses(handler)

    assert isinstance(raised, StageProducedNothingError), (
        "an empty demo_video artifact must FAIL the run, not degrade it — the "
        "exception has to escape the executor for the dispatcher to mark it failed"
    )
    assert raised.empty_fields == ("artifact_url",)
    # It must never have been recorded as a success.
    assert not [s for s in statuses if s[1] == "completed"]
    # …and the guilty stage must be the one marked failed, with a message an
    # operator can act on without opening this file.
    failed = [s for s in statuses if s[0] == "s" and s[1] == "failed"]
    assert len(failed) == 1, f"the stage row was not marked failed: {statuses}"
    error = failed[0][2] or ""
    assert "StageProducedNothingError" in error
    assert "'s'" in error and "artifact_url" in error
    assert ("__event__", "stage_failed", None) in statuses


@pytest.mark.asyncio
async def test_video_render_with_a_dropped_artifact_field_fails_the_run() -> None:
    """The drift shape: the declared name is not in the dump at all."""
    handler = _stage_returning_unvalidated(VideoRenderOutput, duration_seconds=12.0, notes=[])

    statuses, raised = await _run_capturing_statuses(handler)

    assert isinstance(raised, StageProducedNothingError)
    assert raised.empty_fields == ("artifact_url",)
    assert not [s for s in statuses if s[1] == "completed"]


@pytest.mark.asyncio
async def test_a_real_artifact_still_completes() -> None:
    """The counterweight: the gate must not fail a stage that DID produce."""
    handler = _stage_returning_unvalidated(
        VideoRenderOutput,
        artifact_url="https://minio.test/b/final.mp4",
        duration_seconds=12.0,
        notes=[],
    )

    statuses, raised = await _run_capturing_statuses(handler)

    assert raised is None
    assert ("s", "completed", None) in statuses


def test_every_terminal_artifact_producer_is_required() -> None:
    """All seven, not just the two driven through the executor above.

    Named explicitly rather than counted: the ratchet in
    `tests/test_stage_produces_ratchet.py` holds the floor, but a refactor that
    swapped one of these for a different model would keep the count at seven
    while quietly un-requiring a deliverable.

    action_reel + audiobook joined the list in T1-1 (both were zero-declared).
    """
    for model in (
        ActionReelCompileOutput,
        AppReviewCompileOutput,
        AudiobookStitchOutput,
        CourseStitchOutput,
        DemoVideoCompileOutput,
        LessonCompileOutput,
        PromoRenderOutput,
        VideoRenderOutput,
        WebsiteReviewCompileOutput,
        # T1-2: the four remaining stages whose artifact IS what ships — the two
        # audit blueprints' terminal compile, promo_v1's terminal
        # `footage_critique`, and the two demo_video pass-throughs that
        # `render_manifest` resolves the delivered artifact from.
        AuditCompileOutput,
        EncodeMasterOutput,
        FinalCritiqueOutput,
        FootageCritiqueOutput,
    ):
        assert model.produces == ("artifact_url",), model.__name__
        assert model.produces_required is True, (
            f"{model.__name__} produces the deliverable itself — an empty one must "
            f"fail the run, not degrade it"
        )


# ── The two customer-startable blueprints wired in (T1-1) ─────────────────────
# action_reel and audiobook declared NOTHING until T1-1: every stage of both
# could report `completed` holding an empty timeline / an empty chapter plan and
# the run stayed green all the way to a deliverable nobody could use. These drive
# the REAL models through the executor, because a declaration nobody watched fail
# is an assumption.


@pytest.mark.asyncio
async def test_cut_plan_with_no_shots_degrades_the_run() -> None:
    """`cut_plan.shots` IS the reel — compile/graphics_design/narrate all read it."""
    persisted = await _run_single_stage(_stage_returning(CutPlanOutput, shots=[]))

    assert len(persisted) == 1
    notes = persisted[0]["notes"]
    assert isinstance(notes, list)
    assert len(notes) == 1, (
        "an empty cut_plan reached `completed` with no violation recorded — the "
        "`produces` declaration on CutPlanOutput is not being enforced"
    )
    assert "shots" in str(notes[0])


@pytest.mark.asyncio
async def test_audiobook_narrate_with_no_clips_degrades_the_run() -> None:
    """`audiobook_narrate.clips` is what stitch and qa consume."""
    persisted = await _run_single_stage(_stage_returning(AudiobookNarrateOutput, clips=[]))

    assert len(persisted) == 1
    notes = persisted[0]["notes"]
    assert isinstance(notes, list)
    assert len(notes) == 1
    assert "clips" in str(notes[0])


@pytest.mark.asyncio
async def test_audiobook_plan_with_no_chapters_degrades_the_run() -> None:
    persisted = await _run_single_stage(
        _stage_returning(AudiobookPlanOutput, chapters=[], total_chunks=0)
    )

    assert len(persisted) == 1
    notes = persisted[0]["notes"]
    assert isinstance(notes, list)
    assert len(notes) == 1
    assert "chapters" in str(notes[0])


@pytest.mark.asyncio
async def test_action_reel_compile_with_no_artifact_fails_the_run() -> None:
    """The reel itself: an empty artifact must FAIL, matching every sibling compile."""
    handler = _stage_returning_unvalidated(
        ActionReelCompileOutput, artifact_url="", duration_seconds=0.0, notes=[]
    )

    statuses, raised = await _run_capturing_statuses(handler)

    assert isinstance(raised, StageProducedNothingError)
    assert raised.empty_fields == ("artifact_url",)
    assert not [s for s in statuses if s[1] == "completed"]


@pytest.mark.asyncio
async def test_audiobook_stitch_with_no_artifact_fails_the_run() -> None:
    handler = _stage_returning_unvalidated(
        AudiobookStitchOutput, artifact_url="", duration_seconds=0.0, chapters=[], notes=[]
    )

    statuses, raised = await _run_capturing_statuses(handler)

    assert isinstance(raised, StageProducedNothingError)
    assert raised.empty_fields == ("artifact_url",)
    assert not [s for s in statuses if s[1] == "completed"]


# ── The five remaining blueprint clusters wired in (T1-2) ────────────────────
# generic / promo / demo_video / shared / audit_video declared nothing until
# T1-2. As above, these drive the REAL models through the executor rather than a
# throwaway twin, because a declaration nobody watched fail is an assumption.


@pytest.mark.asyncio
async def test_storyboard_draft_with_no_shots_degrades_the_run() -> None:
    """`shots` is the render spine; empty surfaces one stage LATE, at compile."""
    persisted = await _run_single_stage(_stage_returning(StoryboardDraftOutput, shots=[]))

    assert len(persisted) == 1
    notes = persisted[0]["notes"]
    assert isinstance(notes, list)
    assert len(notes) == 1, (
        "an empty storyboard reached `completed` with no violation recorded — the "
        "`produces` declaration on StoryboardDraftOutput is not being enforced"
    )
    assert "shots" in str(notes[0])


@pytest.mark.asyncio
async def test_demo_video_narrate_with_no_clips_degrades_the_run() -> None:
    """Zero clips compiles a SILENT film and still reports completed."""
    persisted = await _run_single_stage(_stage_returning(DemoVideoNarrateOutput, clips=[]))

    assert len(persisted) == 1
    notes = persisted[0]["notes"]
    assert isinstance(notes, list)
    assert len(notes) == 1
    assert "clips" in str(notes[0])


@pytest.mark.asyncio
async def test_demo_video_narrate_with_clips_is_left_alone() -> None:
    """The counterweight — the gate must not annotate a stage that DID produce."""
    clip = {
        "id": "c1",
        "line_id": "l1",
        "artifact_url": "https://minio.test/b/l1.wav",
        "seconds": 2.0,
    }
    persisted = await _run_single_stage(_stage_returning(DemoVideoNarrateOutput, clips=[clip]))

    assert len(persisted) == 1
    assert persisted[0]["notes"] == []


@pytest.mark.asyncio
async def test_audit_narrate_with_no_clips_degrades_the_run() -> None:
    persisted = await _run_single_stage(_stage_returning(AuditNarrateOutput, clips=[]))

    assert len(persisted) == 1
    notes = persisted[0]["notes"]
    assert isinstance(notes, list)
    assert len(notes) == 1
    assert "clips" in str(notes[0])


@pytest.mark.asyncio
async def test_promo_storyboard_with_no_shots_degrades_the_run() -> None:
    persisted = await _run_single_stage(_stage_returning(PromoStoryboardOutput, shots=[]))

    assert len(persisted) == 1
    notes = persisted[0]["notes"]
    assert isinstance(notes, list)
    assert len(notes) == 1
    assert "shots" in str(notes[0])


@pytest.mark.asyncio
async def test_footage_direction_with_no_plans_degrades_the_run() -> None:
    """The module's own header names this: a timed-out planner returns no plans."""
    persisted = await _run_single_stage(_stage_returning(FootageDirectionOutput, plans=[]))

    assert len(persisted) == 1
    notes = persisted[0]["notes"]
    assert isinstance(notes, list)
    assert len(notes) == 1
    assert "plans" in str(notes[0])


@pytest.mark.asyncio
async def test_audit_compile_with_no_artifact_fails_the_run() -> None:
    """The audit blueprints' terminal artifact — an empty one must FAIL."""
    handler = _stage_returning_unvalidated(
        AuditCompileOutput, artifact_url="", duration_seconds=0.0, notes=[]
    )

    statuses, raised = await _run_capturing_statuses(handler)

    assert isinstance(raised, StageProducedNothingError)
    assert raised.empty_fields == ("artifact_url",)
    assert not [s for s in statuses if s[1] == "completed"]


@pytest.mark.asyncio
async def test_footage_critique_with_a_dropped_artifact_field_fails_the_run() -> None:
    """promo_v1's terminal artifact stage, in the drift shape (field absent)."""
    handler = _stage_returning_unvalidated(FootageCritiqueOutput, duration_seconds=4.0, notes=[])

    statuses, raised = await _run_capturing_statuses(handler)

    assert isinstance(raised, StageProducedNothingError)
    assert raised.empty_fields == ("artifact_url",)
    assert not [s for s in statuses if s[1] == "completed"]


# ── A stated failure outranks an inferred emptiness ───────────────────────────
# `promo_slides` is the one shipping stage that declares BOTH gates ("belt AND
# braces", by its own comment). The two answer different questions and only one
# of them carries a cause: `_enforce_produces` infers a defect from `slides == []`
# and can say nothing about why, while `_enforce_failure_flags` reads a flag the
# stage set at the exact point it swallowed the downstream error, and forwards
# that error's text out of `notes` into the message.
#
# Run the emptiness gate first and the flag gate never executes, so the operator
# gets "produced nothing ... look at the stage handler for a swallowed downstream
# error" with the swallowed error discarded. That is not hypothetical: it is what
# the live 2026-09-08/09 promo outage recorded, and three separate investigations
# read that message as naming the orchestrator when the cause was in
# slide_service. The cause was in `notes` the whole time.


class _BothGates(StageOutputWithNotes):
    """The `promo_slides` shape: a required output AND a self-reported failure."""

    produces: ClassVar[tuple[str, ...]] = ("slides",)
    produces_required: ClassVar[bool] = True
    fails_when_flagged: ClassVar[tuple[str, ...]] = ("slides_render_failed",)
    slides: list[str] = Field(default_factory=list)
    slides_render_failed: bool = False


@pytest.mark.asyncio
async def test_a_flagged_failure_reports_its_cause_not_just_its_emptiness() -> None:
    """The operator must get the swallowed cause, not 'this field is empty'.

    Both gates fire on this output. The flag one wins because it knows something
    the other cannot: WHY. Asserting on the cause text rather than on the
    exception type is deliberate — the type could change; losing the cause is the
    defect.
    """
    cause = "slide render failed: slide template 'title' not found"
    errors: list[str] = []

    async def emit(event: str, **kw: object) -> None:
        return None

    async def persist(
        stage_name: str,
        status: str,
        output: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        if status == "failed" and error is not None:
            errors.append(error)

    handler_cls = _stage_returning(_BothGates, slides=[], slides_render_failed=True, notes=[cause])
    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[StageSpec(name="s", handler_factory=lambda deps: handler_cls())],
    )
    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    with contextlib.suppress(Exception):
        await executor.run(ctx, pause_after_stages=set())

    assert errors, "the stage did not persist a failure"
    assert cause in errors[0], (
        f"the failure the operator reads does not carry the swallowed cause — got: {errors[0]!r}"
    )


def test_the_flag_gate_is_consulted_before_the_emptiness_gate() -> None:
    """Pins the ORDER directly, so a refactor cannot silently restore the bug.

    The executor-level test above proves the observable behaviour; this proves
    the reason for it, and fails loudly if someone swaps the two calls back.
    """
    # Anchor on the `self.` call syntax, NOT the bare name: the surrounding
    # comments discuss both gates by name, and matching those would let this
    # test pass against prose while the calls stayed in the wrong order.
    source = inspect.getsource(DAGExecutor._run_one)
    assert source.count("self._enforce_failure_flags(") == 1
    assert source.count("self._enforce_produces(") == 1
    flags_at = source.index("self._enforce_failure_flags(")
    produces_at = source.index("self._enforce_produces(")
    assert flags_at < produces_at, (
        "_enforce_produces runs first, so a stage declaring both gates loses the "
        "cause its flag carried"
    )
