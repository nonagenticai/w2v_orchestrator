"""action_reel blueprint topology + review-gate invariants."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from src.services.blueprints.action_reel import ACTION_REEL_BLUEPRINT
from src.services.blueprints.registry import get_registry
from src.services.runs.run_context import RunContext
from src.services.stages.action_reel.narrate import (
    MAX_NARRATION_LINES,
    NARRATE_TIMEOUT_SECONDS,
    ActionReelNarrateInput,
    ActionReelNarrateOutput,
    ActionReelNarrateStage,
)
from src.services.stages.action_reel.shot_analysis import SHOT_ANALYSIS_TIMEOUT_SECONDS
from src.services.stages.base import StagePreconditionError, StageSkippedError
from src.services.stages.primitives.compile_stage import COMPILE_STAGE_OUTER_TIMEOUT_SECONDS
from src.services.stages.primitives.narration import (
    _CONCURRENCY,
    _PER_ITEM_TIMEOUT_SECONDS,
    NarrationStageBase,
)


def test_action_reel_registered() -> None:
    assert get_registry().resolve("action_reel") is ACTION_REEL_BLUEPRINT
    assert ACTION_REEL_BLUEPRINT.id == "action_reel_v1"


def test_stage_list_and_order() -> None:
    """The declared stage list, pinned. This is the FE/codegen contract as much as
    an internal one — /api/blueprints serves it verbatim."""
    assert [s.name for s in ACTION_REEL_BLUEPRINT.stages] == [
        "source_ingest",
        "footage_probe",
        "shot_analysis",
        "segment_select",
        "music_select",
        "cut_plan",
        "graphics_design",
        "narrate",
        "compile",
        "render_manifest",
    ]
    order = ACTION_REEL_BLUEPRINT.topological_order()
    assert len(order) == len(ACTION_REEL_BLUEPRINT.stages)
    assert order[0] == "source_ingest"
    assert order[-1] == "render_manifest"


def test_two_branches_join_at_cut_plan() -> None:
    """The footage branch and the music branch are independent until cut_plan.

    Load-bearing: if music_select ever grew an edge onto the footage chain, the
    analyzer round-trip for the beat grid would serialise behind the (much
    longer) per-source video analysis for no editorial gain.
    """
    stage = ACTION_REEL_BLUEPRINT.stage
    assert stage("footage_probe").depends_on == ("source_ingest",)
    assert stage("shot_analysis").depends_on == ("footage_probe",)
    assert stage("segment_select").depends_on == ("shot_analysis",)
    # Music hangs off the grounding leaf ONLY — never off the footage chain.
    assert stage("music_select").depends_on == ("source_ingest",)
    assert set(stage("cut_plan").depends_on) == {"segment_select", "music_select"}


def test_compile_joins_graphics_and_narration() -> None:
    # narrate skips clean, so its edge never blocks a music-only reel; the
    # graphics edge is declared today (with an empty card list) so shipping the
    # designer is a one-stage change, not a topology change.
    assert set(ACTION_REEL_BLUEPRINT.stage("compile").depends_on) == {
        "graphics_design",
        "narrate",
    }
    assert ACTION_REEL_BLUEPRINT.stage("render_manifest").depends_on == ("compile",)
    order = ACTION_REEL_BLUEPRINT.topological_order()
    for stage_name in ("graphics_design", "narrate"):
        assert order.index("cut_plan") < order.index(stage_name)
        assert order.index(stage_name) < order.index("compile")


def test_every_editorial_decision_pauses() -> None:
    """Exactly the five decision stages pause; every mechanical stage does not.

    The asymmetry is the point: each of these is free to change above the render
    and a full re-render to change below it.
    """
    paused = {s.name for s in ACTION_REEL_BLUEPRINT.stages if s.default_pause_after}
    assert paused == {
        "shot_analysis",
        "segment_select",
        "music_select",
        "cut_plan",
        "graphics_design",
    }
    for mechanical in ("source_ingest", "footage_probe", "narrate", "compile", "render_manifest"):
        assert not ACTION_REEL_BLUEPRINT.stage(mechanical).default_pause_after


def test_compile_cut_is_the_plan_not_the_render() -> None:
    """A timeline reorder must re-enter at cut_plan.

    `compile` is the deliverable, but re-running only `compile` would re-render
    the stale shot order, and re-running from `segment_select` would re-draw a
    different edit entirely — discarding the selection a human approved.
    """
    assert ACTION_REEL_BLUEPRINT.terminal_artifact_stage == "compile"
    assert ACTION_REEL_BLUEPRINT.compile_cut_stage == "cut_plan"
    assert "compile" in ACTION_REEL_BLUEPRINT.descendants("cut_plan")


def test_stage_timeouts_are_explicit_where_they_must_be() -> None:
    # The LLM stages carry the None inherit-sentinel so the GitOps ceiling knob
    # reaches them; the fan-out / compile stages pin an explicit budget that
    # covers their inner worst case.
    stage = ACTION_REEL_BLUEPRINT.stage
    assert stage("segment_select").default_timeout_seconds is None
    assert stage("music_select").default_timeout_seconds is None
    # ceil(_TIMEOUT_SOURCE_CEILING / _CONCURRENCY) WAVES * 600s + 100s headroom.
    # History, so nobody re-derives a superseded figure: 1400s (per-source arithmetic
    # that assumed a single wave and a retry that cannot fire) -> 1900s (3 waves at
    # _CONCURRENCY=4) -> 3700s (6 waves at _CONCURRENCY=2).
    #
    # _CONCURRENCY dropped to 2 because 4 concurrent 4K analyses were MEASURED to
    # OOMKill the analyzer pod (exitCode 137) once this stage started asking for a
    # real frame budget; 3 peaked at 1850/4096 MiB and 2 is the safe worst-case skew.
    # The literal is kept ALONGSIDE the imported constant on purpose: the next line
    # proves the blueprint and the module agree, but only this hardcoded number makes
    # a change to the budget show up in a diff and demand a deliberate decision.
    assert stage("shot_analysis").default_timeout_seconds == 3700
    assert stage("shot_analysis").default_timeout_seconds == SHOT_ANALYSIS_TIMEOUT_SECONDS
    # ceil(MAX_NARRATION_LINES / 4) waves * (1 + 1 retry) * 600s + 100s headroom.
    assert stage("narrate").default_timeout_seconds == 3700
    assert stage("narrate").default_timeout_seconds == NARRATE_TIMEOUT_SECONDS
    # Must cover the compile-wait backstop (defect-1 fix, 2026-09-17 incident) —
    # see compile_stage.COMPILE_STAGE_OUTER_TIMEOUT_SECONDS.
    assert stage("compile").default_timeout_seconds == COMPILE_STAGE_OUTER_TIMEOUT_SECONDS


def test_the_narrate_budget_covers_every_wave_the_line_cap_allows() -> None:
    """THE regression. The old budget was 1300s with a comment claiming a 1200s
    worst case — true only for a SINGLE concurrency wave.

    ``stage_gather`` runs ``_CONCURRENCY`` lines at a time with one retry each, so
    the stage's wall clock is ``ceil(N / _CONCURRENCY) * 2 * _PER_ITEM_TIMEOUT``.
    At N = 5 that is 2400s against a 1300s ceiling: the outer TimeoutMiddleware
    cancels a PAID stage mid-fan-out. This asserts the ceiling covers the largest
    line count the stage will actually accept — recomputed from the fan-out's own
    constants, so lowering either the budget or raising the cap alone fails here.
    """
    waves = math.ceil(MAX_NARRATION_LINES / _CONCURRENCY)
    worst_case = waves * 2 * _PER_ITEM_TIMEOUT_SECONDS

    budget = ACTION_REEL_BLUEPRINT.stage("narrate").default_timeout_seconds
    assert budget is not None, "narrate must pin its own ceiling, not inherit the LLM one"
    assert budget >= worst_case
    # ...and the old single-wave budget provably did NOT — spelled out rather than
    # implied, so a revert to 1300 cannot read as a harmless number change.
    assert worst_case > 1300


def _narrate_ctx(line_count: int) -> RunContext:
    lines: list[dict[str, Any]] = [
        {"id": f"vo-{n}", "text": f"line {n}"} for n in range(1, line_count + 1)
    ]
    return RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot={"brief": {"narration": {"lines": lines}}},
        prior_run_outputs={},
    )


def _stub_fanout(monkeypatch: pytest.MonkeyPatch, reached: list[int]) -> None:
    """Replace the base class's fan-out so these tests exercise the GATE only.

    The real ``NarrationStageBase.handle`` would run ``stage_gather`` against live
    TTS + storage — 600s per item before it even retries. Recording the line count
    it was reached with is the whole signal: this is a test about what gets past
    the gate, not about synthesis.
    """

    async def _fake_handle(
        _self: NarrationStageBase[ActionReelNarrateInput, ActionReelNarrateOutput],
        _payload: ActionReelNarrateInput,
        ctx: RunContext,
    ) -> ActionReelNarrateOutput:
        brief = ctx.config_snapshot["brief"]
        reached.append(len(brief["narration"]["lines"]))
        return ActionReelNarrateOutput(clips=[])

    monkeypatch.setattr(NarrationStageBase, "handle", _fake_handle)


async def test_a_brief_the_budget_cannot_serve_is_refused_before_any_paid_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the budget fix, and the reason it is a cap and not just a
    bigger number: the line count is AUTHOR-controlled and unbounded, so no finite
    ceiling is sufficient without one.

    Refusal has to happen at the top of the stage. ``narrate`` is PAID (live TTS),
    so the alternative is a TimeoutMiddleware cancellation ~22 minutes into a
    fan-out that has already spent — hence the assertion that the fan-out is never
    entered at all.
    """
    reached: list[int] = []
    _stub_fanout(monkeypatch, reached)

    with pytest.raises(StagePreconditionError) as caught:
        await ActionReelNarrateStage(client=AsyncMock()).handle(
            ActionReelNarrateInput(), _narrate_ctx(MAX_NARRATION_LINES + 1)
        )

    assert str(MAX_NARRATION_LINES) in str(caught.value)
    assert reached == []


async def test_the_cap_admits_a_brief_that_exactly_fills_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cap that refuses the count it was SIZED for is the same defect the other
    way round — so the boundary is pinned from both sides. At exactly
    ``MAX_NARRATION_LINES`` the fan-out must be entered with every line."""
    reached: list[int] = []
    _stub_fanout(monkeypatch, reached)

    out = await ActionReelNarrateStage(client=AsyncMock()).handle(
        ActionReelNarrateInput(), _narrate_ctx(MAX_NARRATION_LINES)
    )

    assert reached == [MAX_NARRATION_LINES]
    assert out.clips == []


async def test_no_narration_still_skips_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-in gate must survive the new one above it: most reels carry no
    voice-over at all, and that is a skip, never a precondition failure."""
    reached: list[int] = []
    _stub_fanout(monkeypatch, reached)

    with pytest.raises(StageSkippedError):
        await ActionReelNarrateStage(client=AsyncMock()).handle(
            ActionReelNarrateInput(), _narrate_ctx(0)
        )

    assert reached == []
