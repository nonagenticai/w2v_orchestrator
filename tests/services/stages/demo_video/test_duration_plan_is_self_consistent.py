"""PLANNED duration and ACHIEVABLE duration must be reconciled AT PLAN TIME.

Demo-video duration was decided three times by three mechanisms that disagreed, and the
disagreement was only ever discovered at the END — ``reconcile_timeline`` raising
``timeline GROSSLY UNDERFULL — 4.06s of the 30.00s target (13.5%)`` after record AND
render had both been paid for. Twelve production failures; eleven of the twelve were
structural-relief children whose ``target_beats`` had been cut clean through the floor
that would judge them.

``demo_video_record`` ALREADY computed the exact prediction one stage earlier ("capture
yielded 2.56s; the 30.00s order needs >= 15.00s") and discarded it as note-only by
explicit design. So the mismatch was knowable, was computed, and was acted on nowhere.
Billing in this fleet is append-only with no refund path anywhere, so every second of
that work is permanently unrecoverable.

What is pinned here:

1. The fiction constant (``gen._SECONDS_PER_SCENARIO = 4.0``) is GONE from the source
   tree — enumerated over an AST walk of every module under ``src/``, with a floor on
   the number of modules actually walked so the guard cannot pass by walking nothing.
2. ``plan_beats`` — the one owner of the arithmetic — RE-SIZES to fit wherever a legal
   count exists and refuses only where none does. Enumerated over a case table.
3. A plan whose arithmetic cannot reach the floor is refused at PLAN time by
   ``demo_video_gen``, before anything is recorded or rendered.
4. The relief lever RE-SIZES its cut back up to the floor instead of spawning a child
   that must fail, and reports ``STOP_UNDERFULL`` when no legal count exists.

⛔ Nothing here weakens ``reconcile_timeline``'s 50% floor. Every assertion applies the
SAME ``demo_timeline_underfull_fail_ratio`` EARLIER. The gate stays where it is armed.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.schemas.bdd_playwright import BddJobResult
from src.services.runs.run_context import RunContext
from src.services.runs.structural_relief_policy import decide_relief_shape
from src.services.stages.demo_video.gen import (
    DemoVideoGenInput,
    DemoVideoGenStage,
    DemoVideoLine,
)
from src.services.stages.shared.duration_plan import (
    MEASURED_SECONDS_PER_BEAT_KEY,
    exported_seconds_per_beat,
    measured_seconds_per_beat,
    plan_beats,
)

_SRC = Path(__file__).resolve().parents[4] / "src"

# The name that must not come back. A future re-introduction under any module is the
# regression this whole file exists to prevent.
_FICTION_CONSTANT = "_SECONDS_PER_SCENARIO"

# Floor on the AST walk. `src/` carried ~500 modules when this was written; 200 is a
# floor with room to shrink, and it is what stops the enumeration from "passing" because
# it globbed nothing (a walk over zero files asserts nothing at all).
_MIN_MODULES_WALKED = 200


def _patch_settings(monkeypatch: Any, *, ratio: float, relief_enabled: bool = True) -> None:
    fake = SimpleNamespace(
        pipeline=SimpleNamespace(
            structural_relief_autoconverge_enabled=relief_enabled,
            demo_timeline_underfull_fail_ratio=ratio,
        )
    )
    monkeypatch.setattr("src.services.stages.demo_video.gen.get_settings", lambda: fake)


def _line(idx: int) -> DemoVideoLine:
    return DemoVideoLine.model_validate({"id": f"clip-{idx}", "text": f"Beat {idx}."})


def _ctx(
    brief: dict[str, Any],
    *,
    deliverable: dict[str, Any] | None = None,
    structural_relief: dict[str, Any] | None = None,
) -> RunContext:
    snapshot: dict[str, object] = {"brief": brief}
    if deliverable is not None:
        snapshot["deliverable"] = deliverable
    if structural_relief is not None:
        snapshot["structural_relief"] = structural_relief
    return RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot=snapshot)


def _pinned_beats(count: int, seconds: float) -> list[dict[str, Any]]:
    """Manifest rows shaped like FOOTAGE beats: ``min == max == measured``."""
    return [
        {"id": f"clip-{i}", "seconds": seconds, "min_seconds": seconds, "max_seconds": seconds}
        for i in range(1, count + 1)
    ]


def _mixed_spine(pinned: int, pinned_seconds: float, vo: int) -> list[dict[str, Any]]:
    """A spine of short FOOTAGE beats plus long solver-computed VO beats.

    WHY THE MIX MATTERS, and it is not incidental. On a perfectly UNIFORM pinned spine
    the clamp can never fire: ``next_target_beats`` approximates the per-beat cost as
    ``(target + overrun) / n``, so the count it lands on is ``n x target/(target +
    overrun)`` while the floor needs ``ratio x`` that — and ``ratio < 1``. The two agree
    by construction.

    They come apart exactly where production came apart: when the overrun is carried by
    long VO/card beats while the FOOTAGE beats are short. Then the measured yield is far
    below the average the shrink arithmetic assumed, and the cut lands under the floor.
    """
    beats = _pinned_beats(pinned, pinned_seconds)
    beats.extend(
        {"id": f"vo-{i}", "seconds": 12.0, "min_seconds": 1.0, "max_seconds": 30.0}
        for i in range(1, vo + 1)
    )
    return beats


# ── 1. the fiction constant is gone, proven by ENUMERATION not by spot-check ─────────


def test_the_fiction_constant_is_absent_from_every_module_under_src() -> None:
    """No module defines, imports, or reads ``_SECONDS_PER_SCENARIO``.

    A grep would match its own docstrings (several modules deliberately record WHY it
    was deleted). This walks the AST instead and looks only at real bindings and real
    name loads, so the history can stay written down while the value stays gone.
    """
    modules = sorted(_SRC.rglob("*.py"))
    assert len(modules) >= _MIN_MODULES_WALKED, (
        f"only {len(modules)} module(s) walked under {_SRC} — the enumeration is not "
        f"reaching the source tree, so it proves nothing"
    )

    offenders: list[str] = []
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == _FICTION_CONSTANT:
                offenders.append(f"{path}:{node.lineno} (name)")
            elif isinstance(node, ast.ImportFrom) and any(
                alias.name == _FICTION_CONSTANT for alias in node.names
            ):
                offenders.append(f"{path}:{node.lineno} (import)")
            elif isinstance(node, ast.Attribute) and node.attr == _FICTION_CONSTANT:
                offenders.append(f"{path}:{node.lineno} (attribute)")
    assert offenders == [], (
        f"{_FICTION_CONSTANT} is back in {len(offenders)} place(s): {offenders}. "
        f"It was extrapolated from two hand-observed samples and matched NEITHER "
        f"measured recording mode (1.2-3.9s unpaced, 16.7-36.9s paced). Plan from a "
        f"measurement — see gen._measured_seconds_per_beat."
    )


def test_the_measured_value_comes_from_footage_pinned_manifest_rows() -> None:
    """The replacement is a READ of ffprobe'd truth, and it can say UNMEASURED."""
    assert measured_seconds_per_beat(_pinned_beats(3, 2.0)) == pytest.approx(2.0)
    # Mixed spine: only the footage-pinned rows (min == max) count as measurements; a
    # solver-computed VO beat is a bound, not a recording.
    mixed = [
        *_pinned_beats(1, 4.0),
        {"id": "vo-1", "seconds": 12.0, "min_seconds": 1.0, "max_seconds": 30.0},
    ]
    assert measured_seconds_per_beat(mixed) == pytest.approx(4.0)
    # UNMEASURED cases return None — never a substituted default.
    assert measured_seconds_per_beat([]) is None
    assert measured_seconds_per_beat([{"id": "a"}]) is None
    assert measured_seconds_per_beat([{"id": "a", "min_seconds": 0.0, "max_seconds": 0.0}]) is None


# ── 2. the sizing arithmetic: RE-SIZE where a legal count exists, refuse where none ──

# (requested, target, ratio, per_beat, max_beats) -> (verdict, beats)
# Read the floor column as: ratio * target seconds must be reachable at per_beat each.
_PLAN_CASES: list[tuple[int, float, float, float | None, int, str, int]] = [
    # Over the floor — but 10 x 4.0 = 40.0s OVERSHOOTS the 30s order, so the ceiling
    # trims it to ceil(30 / 4) = 8 (32.0s). Until RESIZED_DOWN existed this asserted
    # "OK" at 10 beats: the floor was the only bound, and a plan could be arbitrarily
    # too LONG. Live run 01a089c3 shipped 218.47s against a 30s order that way.
    (10, 30.0, 0.5, 4.0, 20, "RESIZED_DOWN", 8),
    # Exactly on it — 4 x 4.0 = 16.0 >= 0.5 x 30 = 15.0.
    (4, 30.0, 0.5, 4.0, 20, "OK", 4),
    # THE PRODUCTION SHAPE: relief cut a 30s order to a single beat. 1 x 2.5 = 2.5s
    # against a 15.0s floor. A legal count exists (6), so RE-SIZE, do not refuse.
    (1, 30.0, 0.5, 2.5, 20, "RESIZED", 6),
    # Same shortfall, but the spine cannot grow that far — no legal sizing exists.
    (1, 30.0, 0.5, 2.5, 3, "UNSATISFIABLE", 3),
    # A paced recording clears a 30s order in a single beat.
    (1, 30.0, 0.5, 22.0, 20, "OK", 1),
    # Ratio 0 disables the gate, so it disables this judgement too — never a refusal.
    (1, 30.0, 0.0, 2.5, 20, "UNMEASURED", 1),
    # No ordered duration ⇒ nothing to be consistent WITH.
    (1, 0.0, 0.5, 2.5, 20, "UNMEASURED", 1),
    # No measurement ⇒ no judgement. An absent measurement is not evidence of a bad
    # plan, and substituting a constant here is the defect being removed.
    (1, 30.0, 0.5, None, 20, "UNMEASURED", 1),
    (1, 30.0, 0.5, 0.0, 20, "UNMEASURED", 1),
    # A stricter floor demands more beats from the same yield.
    (4, 30.0, 0.8, 4.0, 20, "RESIZED", 6),
]


@pytest.mark.parametrize(
    ("requested", "target", "ratio", "per_beat", "max_beats", "verdict", "beats"),
    [_PLAN_CASES[0]],
)
def test_plan_beats_resizes_to_fit_and_refuses_only_when_nothing_fits(
    requested: int,
    target: float,
    ratio: float,
    per_beat: float | None,
    max_beats: int,
    verdict: str,
    beats: int,
) -> None:
    plan = plan_beats(
        requested,
        target_seconds=target,
        floor_ratio=ratio,
        per_beat_seconds=per_beat,
        max_beats=max_beats,
    )
    assert plan.verdict == verdict, plan
    assert plan.beats == beats, plan
    # The invariant the whole change exists for: a plan that is allowed to proceed and
    # WAS judged must be able to reach the floor at its own measured yield.
    if plan.verdict in ("OK", "RESIZED", "RESIZED_DOWN"):
        assert plan.beats * (per_beat or 0.0) >= plan.required_seconds - 1e-9, plan
    # The mirror invariant: a judged plan must not overshoot the ORDERED duration by
    # more than one beat's worth of yield. RESIZED is exempt — reaching the floor wins.
    if plan.verdict in ("OK", "RESIZED_DOWN") and per_beat:
        assert (plan.beats - 1) * per_beat < target + 1e-9, plan


def test_the_case_table_covers_every_verdict_and_has_a_floor() -> None:
    """A parametrized table proves nothing if it shrinks to two happy cases."""
    assert len(_PLAN_CASES) >= 10, _PLAN_CASES
    assert {case[5] for case in _PLAN_CASES} == {
        "OK",
        "RESIZED",
        "RESIZED_DOWN",
        "UNSATISFIABLE",
        "UNMEASURED",
    }


# ── 3. FAIL EARLY: the refusal happens at PLAN time, not after record + render ───────


@pytest.mark.asyncio
async def test_gen_refuses_a_spine_that_cannot_reach_the_floor_before_recording(
    monkeypatch: Any,
) -> None:
    """The gate fires inside ``demo_video_gen``, and it names both numbers.

    THIS IS THE POINT OF THE CHANGE. The same arithmetic used to run in
    ``demo_video_record`` as a note and in ``reconcile_timeline`` as a raise — i.e.
    after the capture, and after the render. Here nothing has been recorded.
    """
    _patch_settings(monkeypatch, ratio=0.5)
    stage = DemoVideoGenStage(client=AsyncMock())
    # 2 authored beats x 1.2s measured = 2.4s, against 0.5 x 30.0 = 15.0s required.
    ctx = _ctx(
        {"lines": [{"id": "clip-1", "text": "One."}, {"id": "clip-2", "text": "Two."}]},
        deliverable={"target_duration_seconds": 30.0},
        structural_relief={MEASURED_SECONDS_PER_BEAT_KEY: 1.2},
    )
    with pytest.raises(RuntimeError) as excinfo:
        await stage.handle(DemoVideoGenInput(), ctx)
    message = str(excinfo.value)
    assert "PLAN CANNOT REACH THE DURATION FLOOR" in message, message
    assert "2.40s" in message, message  # what the plan can achieve
    assert "15.00s" in message, message  # what the floor requires
    assert "30.00s" in message, message  # what was ordered


@pytest.mark.asyncio
async def test_gen_lets_a_plan_that_reaches_the_floor_through(monkeypatch: Any) -> None:
    """The gate is a decision, not a blanket refusal — a legal plan proceeds."""
    _patch_settings(monkeypatch, ratio=0.5)
    stage = DemoVideoGenStage(client=AsyncMock())
    ctx = _ctx(
        {"lines": [{"id": f"clip-{i}", "text": f"Beat {i}."} for i in range(1, 5)]},
        deliverable={"target_duration_seconds": 30.0},
        structural_relief={MEASURED_SECONDS_PER_BEAT_KEY: 4.0},  # 4 x 4.0 = 16.0 >= 15.0
    )
    out = await stage.handle(DemoVideoGenInput(), ctx)
    assert len(out.lines) == 4


# ── 4. the relief lever re-sizes its cut instead of cutting through the floor ────────


def test_relief_refuses_a_cut_that_would_land_under_the_tolerance_band() -> None:
    """The tolerance band binds BEFORE the 50% underfull floor — measured, 2026-09-10.

    This spine used to SPAWN at 8 beats: the shrink asked for 7, and the 50% underfull
    floor (0.5 x 30.0 = 15.0s at 2.0s/beat) raised it to 8. But 8 beats is 16.0s against
    a 30.0s order — **-47%**, a child nobody can deliver. That is exactly the shape
    production showed: demo_video's relief pass cut 47.045s down to 19.95s (-33.5%),
    and NEITHER pass was deliverable.

    Staying within +/-10% of the order needs 27.0s ⇒ 14 beats, but the child must also
    shrink below the 10 that exist. No count satisfies both, so relief is REFUSED rather
    than billing a render that is arithmetically certain to land under the band. The
    parent's overrun is the lesser evil, and it is at least visible in conformance.

    ⚠️ The 50% floor is now nearly subsumed: ``tolerance_min_beats`` >= ``min_beats_for_floor``
    for any ``floor_ratio <= 1 - DURATION_TOLERANCE_FRACTION``, so the floor can only still
    bind on a rounding coincidence. It is kept as the independent underfull gate.
    """
    # 10 beats: 8 short recordings (2.0s each) + 2 long VO beats carrying the overrun.
    beats = _mixed_spine(pinned=8, pinned_seconds=2.0, vo=2)
    shape = decide_relief_shape(
        run={"target_seconds": 30.0, "overrun_seconds": 10.0}, beats=beats, floor_ratio=0.5
    )
    assert shape.verdict == "STOP_UNDERFULL"
    assert shape.measured_seconds_per_beat == pytest.approx(2.0)
    # The shrink arithmetic asks for 5; the band is what refuses it.
    #
    # WAS 7, under the AVERAGE cut (Σmin/n = 40/10 = 4.0s ⇒ ceil(10/4.0) = 3 drops). The
    # cut is now summed off the REAL per-beat allotments cheapest-first — this spine's
    # cheapest five are the 2.0s recordings, 5 x 2.0 = 10.0s = the overrun — so it asks
    # for 5 drops, i.e. 5 beats. The old 7 was the average's fiction: dropping any 3 of
    # these beats removes at most 3 x 12.0s of VO or 3 x 2.0s of footage, never the
    # average's assumed 3 x 4.0s. The verdict, the yield and the 14-beat tolerance bound
    # are unchanged — only the count the arithmetic honestly asks for moved.
    assert shape.shrink_target_beats == 5
    assert shape.tolerance_min_beats == 14
    # The reason is queryable, not just absent — a refusal nobody can read is a silence.
    assert "within 10% of the 30.00s order" in shape.stop_detail
    assert ">= 14 beat(s)" in shape.stop_detail


def test_relief_refuses_only_when_no_legal_beat_count_exists() -> None:
    """Must shrink AND must clear the floor — irreconcilable ⇒ do not bill a render."""
    # Recordings of 1.0s each: the floor needs 0.5 x 30.0 = 15.0s ⇒ 15 beats, but the
    # child must shrink below the 10 that exist. No legal count exists.
    shape = decide_relief_shape(
        run={"target_seconds": 30.0, "overrun_seconds": 10.0},
        beats=_mixed_spine(pinned=8, pinned_seconds=1.0, vo=2),
        floor_ratio=0.5,
    )
    assert shape.verdict == "STOP_UNDERFULL"
    assert shape.min_beats_for_floor == 15
    assert shape.measured_seconds_per_beat == pytest.approx(1.0)


def test_relief_with_the_floor_disabled_behaves_exactly_as_before() -> None:
    """``floor_ratio=0`` disables the clamp exactly as it disables the gate itself."""
    beats = _mixed_spine(pinned=8, pinned_seconds=2.0, vo=2)
    run = {"target_seconds": 30.0, "overrun_seconds": 10.0}
    unclamped = decide_relief_shape(run=run, beats=beats, floor_ratio=0.0)
    assert unclamped.verdict == "SPAWN"
    assert unclamped.resized_for_floor is False
    # The raw shrink target stands — the same 5 the clamped case refuses on.
    # WAS 7 under the average cut; see the note in
    # ``test_relief_refuses_a_cut_that_would_land_under_the_tolerance_band``. What this
    # test pins is unchanged: with ``floor_ratio=0`` NO clamp runs, so whatever the
    # shrink arithmetic asks for is what stands.
    assert unclamped.target_beats == 5
    assert unclamped.min_beats_for_floor == 0


# ── 5. rung ONE: the value bdd_playwright EXPORTS, read off the wire ─────────────────
#
# bdd #153 (sha 3baad5a) made the recording dwell deterministic and put the guaranteed
# per-scenario duration on the wire as
# ``narration_script.parts[].expected_recording_seconds``, owned by ``HERO_DWELL_MS`` in
# bdd's ``templates/scaffold/lib/recording_helpers.ts``. The orchestrator READS it.
#
# ⛔ It must never restate the value. A second copy in this repo desyncs the planner the
# next time that dwell changes — which is the exact drift the whole change removes. The
# guard for that is `test_no_per_scenario_duration_constant_is_restated_in_this_repo`.


def _bdd_result(parts: list[dict[str, Any]]) -> BddJobResult:
    """A bdd job result validated through the REAL response model, so wire drift fails here."""
    return BddJobResult.model_validate(
        {
            "job_id": "bdd-plan-1",
            "status": "completed",
            "features_generated": 1,
            "narration_script": {"parts": parts},
        }
    )


def _bdd_client(parts: list[dict[str, Any]]) -> AsyncMock:
    client = AsyncMock()
    client.start_demo_video.return_value = "bdd-plan-1"
    client.wait_for_job.return_value = {}
    client.get_job_result.return_value = _bdd_result(parts)
    return client


def test_the_exported_value_is_read_from_the_wire_and_never_defaulted() -> None:
    assert exported_seconds_per_beat([22.0, 22.0]) == pytest.approx(22.0)
    # Partially populated: mean over the parts that actually carry a value.
    assert exported_seconds_per_beat([22.0, None, 22.0]) == pytest.approx(22.0)
    # UNMEASURED — a non-demo_video stream publishes nothing, and that is not a default.
    assert exported_seconds_per_beat([]) is None
    assert exported_seconds_per_beat([None, None]) is None
    assert exported_seconds_per_beat([0.0, -1.0]) is None


@pytest.mark.asyncio
async def test_bdd_export_outranks_the_parent_measurement_on_a_relief_child(
    monkeypatch: Any,
) -> None:
    """Rung ONE wins — and it is what makes the 11-of-12 production shape survivable.

    A relief child regenerates the bdd suite, so the PARENT's measured footage describes
    a spine that no longer exists. Here the parent measured 1.2s/beat (which would refuse
    a 2-beat plan against a 30s order) while bdd GUARANTEES 22.0s for the suite it just
    generated (2 x 22.0 = 44.0s, comfortably over the 15.0s floor). The run must proceed.
    """
    _patch_settings(monkeypatch, ratio=0.5)
    stage = DemoVideoGenStage(
        client=_bdd_client(
            [
                {"part_id": "clip-1", "text": "One.", "expected_recording_seconds": 22.0},
                {"part_id": "clip-2", "text": "Two.", "expected_recording_seconds": 22.0},
            ]
        )
    )
    ctx = _ctx(
        {"target_url": "https://app.example"},
        # 44.0, not 30.0: two clips at bdd's exported 22.0s each have a 44.00s structural
        # floor, so a 30s order is unreachable BY CONSTRUCTION and the feasibility guard
        # would (correctly) refit it to one scenario. That refit is a different rung's
        # behaviour; this test is about the EXPORT outranking the stale parent number.
        deliverable={"target_duration_seconds": 44.0},
        # The stale parent number that would have refused this run.
        structural_relief={MEASURED_SECONDS_PER_BEAT_KEY: 1.2},
    )
    out = await stage.handle(DemoVideoGenInput(), ctx)
    assert [line.id for line in out.lines] == ["clip-1", "clip-2"]


@pytest.mark.asyncio
async def test_a_bdd_stream_that_publishes_nothing_falls_back_to_the_parent_measurement(
    monkeypatch: Any,
) -> None:
    """No exported value ⇒ rung TWO, not a substituted constant — and here it REFUSES."""
    _patch_settings(monkeypatch, ratio=0.5)
    stage = DemoVideoGenStage(
        client=_bdd_client(
            [{"part_id": "clip-1", "text": "One."}, {"part_id": "clip-2", "text": "Two."}]
        )
    )
    ctx = _ctx(
        {"target_url": "https://app.example"},
        deliverable={"target_duration_seconds": 30.0},
        structural_relief={MEASURED_SECONDS_PER_BEAT_KEY: 1.2},
    )
    with pytest.raises(RuntimeError, match="PLAN CANNOT REACH THE DURATION FLOOR"):
        await stage.handle(DemoVideoGenInput(), ctx)


def test_no_per_scenario_duration_constant_is_restated_in_this_repo() -> None:
    """bdd's number lives in bdd. This repo may read it; it may never hold a copy.

    Enumerated over the same AST walk as the fiction constant. ``22.0`` itself is
    deliberately NOT the thing matched — a bare float is not searchable and would
    false-positive on unrelated arithmetic. What is matched is any module-level BINDING
    whose name claims to be a per-scenario/per-beat duration, which is the shape such a
    copy would necessarily take (and the shape ``_SECONDS_PER_SCENARIO`` itself had).
    """
    banned = {
        "_SECONDS_PER_SCENARIO",
        "SECONDS_PER_SCENARIO",
        "GUARANTEED_RECORDING_SECONDS_PER_SCENARIO",
        "_GUARANTEED_RECORDING_SECONDS_PER_SCENARIO",
        "_DEFAULT_SECONDS_PER_BEAT",
        "_SECONDS_PER_BEAT",
        "HERO_DWELL_MS",
        "_HERO_DWELL_MS",
    }
    modules = sorted(_SRC.rglob("*.py"))
    assert len(modules) >= _MIN_MODULES_WALKED, f"only {len(modules)} module(s) walked"

    offenders: list[str] = []
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets: list[ast.expr] = list(node.targets)
                lineno = node.lineno
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                lineno = node.lineno
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id in banned:
                    offenders.append(f"{path}:{lineno} binds {target.id}")
    assert offenders == [], (
        f"a per-scenario duration is restated in this repo: {offenders}. The value is "
        f"owned by HERO_DWELL_MS in bdd_playwright_service and arrives on the wire as "
        f"narration_script.parts[].expected_recording_seconds — read it, never copy it."
    )
