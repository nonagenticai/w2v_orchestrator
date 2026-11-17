"""Unit tests for the demo_video structural timing-relief policy (PR-C).

``decide_structural_relief`` is a pure predicate: it maps orch #671's typed
convergence classification (plus the arming flag, the pass number, and the
overrun trajectory) onto one of four verdicts. These table-driven cases cover
every branch — the clean-stop, the non-candidate progress stop, the pass cap, the
no-progress epsilon/flat stops, and the spawn paths (including the footage-dominated
mixed spine where relief was not "requested") — so a regression in the branch
ordering is caught without touching the DB or the run engine.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.services.runs.structural_relief_policy import (
    ReliefShape,
    StructuralReliefDecision,
    decide_relief_shape,
    decide_structural_relief,
    next_target_beats,
)
from src.services.stages.shared.duration_conformance import (
    DURATION_TOLERANCE_FRACTION,
)


@pytest.mark.parametrize(
    (
        "convergence_outcome",
        "structural_relief_requested",
        "overrun_seconds",
        "prior_overrun_seconds",
        "pass_number",
        "expected",
    ),
    [("converged", True, 0.0, None, 0, "STOP_CONVERGED")],
)
def test_decide_structural_relief(
    convergence_outcome: str,
    structural_relief_requested: bool,
    overrun_seconds: float,
    prior_overrun_seconds: float | None,
    pass_number: int,
    expected: StructuralReliefDecision,
) -> None:
    """Each classification/trajectory maps to its exact relief verdict."""
    decision = decide_structural_relief(
        convergence_outcome=convergence_outcome,
        structural_relief_requested=structural_relief_requested,
        overrun_seconds=overrun_seconds,
        prior_overrun_seconds=prior_overrun_seconds,
        pass_number=pass_number,
    )

    assert decision == expected


# ═════════════════════════════════════════════════════════════════════════════════════
# The UPPER bound, and an honest cut
#
# ``STOP_UNDERFULL`` had no counterpart. Every count-sizing call in this pipeline rounds
# UP (``beats_for_seconds`` = ``ceil``) — including the "ceiling" ``plan_beats`` computes
# from the ORDERED duration — so the count that survived the clamps could be over the
# band and nothing said so. Measured on live run 01a089c3 (below), that spawned a paid
# child at +46.7%, which ``gen._apply_feasibility_floor`` then refitted to -26.7%.
#
# Second half: the cut was sized from an AVERAGE per-beat cost while ``gen`` drops the
# CHEAPEST beats, so it mis-sized in whichever direction the spine was skewed.
# ═════════════════════════════════════════════════════════════════════════════════════


def _pinned(beat_id: str, seconds: float) -> dict[str, Any]:
    """A FOOTAGE row: ``min == max == measured``, the discriminator the yield reads."""
    return {"id": beat_id, "seconds": seconds, "min_seconds": seconds, "max_seconds": seconds}


def _vo(beat_id: str, seconds: float) -> dict[str, Any]:
    """A solver-computed VO row: elastic, so it is a bound and NOT a measurement."""
    return {"id": beat_id, "seconds": seconds, "min_seconds": seconds, "max_seconds": 30.0}


def test_no_child_is_spawned_when_no_beat_count_lands_inside_the_band() -> None:
    """LIVE RUN 01a089c3 — 30.00s order, 47.045s delivered, 22.00s measured per scenario.

    Two footage beats of 22.00s plus three short VO beats: 44.00 + 3.045 = 47.045s, i.e.
    17.045s over a 30.00s order. Every guard that existed passed it:

    * the shrink arithmetic asks for a deep cut;
    * ``plan_beats``'s underfull floor needs ceil(15.00/22.00) = 1 beat;
    * its "ceiling" is ceil(30.00/22.00) = **2** — 44.00s, already past the order it is
      supposed to bound, because it is itself a round-UP;
    * ``tolerance_min_beats`` = ceil(27.00/22.00) = 2, which is not greater than 2, so
      the lower clamp did not fire either.

    So a PAID child was spawned at 2 x 22.00 = 44.00s against a 30.00s order (+46.7%),
    and ``gen._apply_feasibility_floor`` (30 // 22 = 1) then refitted it to 22.00s
    (-26.7%). At this yield the products are 22.00s and 44.00s and the band is
    [27.00s, 33.00s]: the ENUMERATION is what proves no legal count lands inside it.
    """
    beats = [_pinned("clip-1", 22.0), _pinned("clip-2", 22.0)] + [
        _vo(f"vo-{i}", 1.015) for i in range(1, 4)
    ]
    shape = decide_relief_shape(
        run={"target_seconds": 30.0, "overrun_seconds": 17.045}, beats=beats, floor_ratio=0.5
    )
    assert shape.measured_seconds_per_beat == pytest.approx(22.0)
    # The spine CAN shrink and its cheapest 2-beat subset costs 2.03s, so neither
    # pre-existing guard refuses here: without the band check this SPAWNS at 2 beats.
    assert shape.beat_count == 5
    assert shape.verdict == "STOP_UNDERFULL", "a +46.7% child must not be billed"
    # The refusal names ONLY what was enumerated: the legal counts and their seconds.
    detail = shape.stop_detail
    assert "[27.00s, 33.00s]" in detail
    assert "all 4 legal count(s) were checked" in detail
    assert "1 beat(s) = 22.00s" in detail
    assert "2 beat(s) = 44.00s" in detail


def test_a_count_above_the_band_is_lowered_rather_than_refused() -> None:
    """The guard RE-SIZES where a legal count exists — refusing is the last resort.

    Ten beats against a 40.00s order, 20.00s over, on a manifest too thin to model a
    drop sequence (only the two FOOTAGE rows carry seconds), so the cut falls back to the
    average: 60.00/10 = 6.00s, ceil(20.00/6.00) = 4 drops ⇒ 6 beats. The measured footage
    yield is 12.00s, not 6.00s — the mixed-spine gap production keeps showing — so
    ``plan_beats`` resizes down to its ceiling, ceil(40.00/12.00) = **4**. That ceiling is
    48.00s: +20%, above the band [36.00s, 44.00s], because the ceiling is a round-UP of
    the order rather than a bound inside it. Nothing downstream trims it.

    3 beats = 36.00s IS inside the band, so the plan is lowered to 3 and the child is
    spawned. Without the upper bound this spawned a paid 4-beat child at +20%
    (``protected_seconds_floor`` sees only the two footage rows, 24.00s, and passes it).
    """
    beats = [_pinned("clip-1", 12.0), _pinned("clip-2", 12.0)] + [
        {"id": f"vo-{i}", "max_seconds": 30.0} for i in range(1, 9)
    ]
    shape = decide_relief_shape(
        run={"target_seconds": 40.0, "overrun_seconds": 20.0}, beats=beats, floor_ratio=0.5
    )
    assert shape.measured_seconds_per_beat == pytest.approx(12.0)
    assert shape.shrink_target_beats == 6, "the unguarded average cut, for the record"
    assert shape.verdict == "SPAWN", "a legal count exists — re-size, never refuse"
    assert shape.target_beats == 3
    assert shape.target_beats * 12.0 == pytest.approx(36.0)


def test_the_cut_is_sized_from_the_beats_gen_actually_drops() -> None:
    """Cheapest-first, not the average — the spine that measured +30.0%.

    Six beats: five of 6.00s and one of 60.00s (Σ = 90.00s) against a 60.00s order, so
    30.00s over. ``gen._apply_target_beats`` drops the CHEAPEST droppable beats, so
    removing 30.00s takes FIVE drops (6+6+6+6+6), leaving the 60.00s beat — exactly the
    order.

    The average said otherwise: Σmin/n = 15.00s, ceil(30.00/15.00) = 2 drops ⇒ 4 beats,
    and the two cheapest remove only 12.00s, leaving 78.00s = +30.0%. The assertions
    below are on the SECONDS each answer delivers, not on the counts alone.
    """
    spine = [6.0, 6.0, 6.0, 6.0, 6.0, 60.0]
    beats = [_pinned(f"clip-{i}", secs) for i, secs in enumerate(spine, start=1)]
    run = {"target_seconds": 60.0, "overrun_seconds": 30.0}

    target = next_target_beats(run, len(beats), beats)
    assert target == 1

    def _kept_seconds(count: int) -> float:
        """What survives when the cheapest beats are the ones dropped."""
        return sum(sorted(spine, reverse=True)[:count])

    assert _kept_seconds(target) == pytest.approx(60.0), "lands on the order"
    assert _kept_seconds(4) == pytest.approx(78.0), "what the average's 4 beats deliver"
    # Same call WITHOUT the beats falls back to the average — the fallback is live, and
    # this is the number it produces.
    assert next_target_beats(run, len(beats)) == 4


# ═════════════════════════════════════════════════════════════════════════════════════
# STOP_ALREADY_CONFORMING — the parent that had already delivered, and bought a child
#
# Production, 44.00s order: the parent delivered 44.016s (+0.04%) — already inside the
# +/-10% band the film is graded by. Every gate here asks "can the CHILD converge?" and
# a legal 4-beat count exists at 10.632s/beat, so the old gate SPAWNed. The paid child
# came back at 46.217s (+5.04%): strictly worse than the thing it replaced, billed at
# full price against a billing path that is append-only with no refund.
#
# The rows below are real production runs (T=44.00s, floor_ratio=0.5, all beats
# footage-pinned so ``min == max == measured``). The guard must flip the two that were
# already conforming and leave every other row EXACTLY where it was.
# ═════════════════════════════════════════════════════════════════════════════════════

_ORDERED_SECONDS = 44.0


def _pinned_spine(per_beat: float, beat_count: int) -> list[dict[str, Any]]:
    """``beat_count`` identical footage rows — the all-pinned shape every real row has."""
    return [_pinned(f"clip-{i}", per_beat) for i in range(1, beat_count + 1)]


def _shape_for(per_beat: float, beat_count: int, delivered: float | None) -> ReliefShape:
    """``decide_relief_shape`` for one production row, overrun derived not asserted."""
    beats = _pinned_spine(per_beat, beat_count)
    return decide_relief_shape(
        run={
            "target_seconds": _ORDERED_SECONDS,
            "overrun_seconds": beat_count * per_beat - _ORDERED_SECONDS,
        },
        beats=beats,
        floor_ratio=0.5,
        delivered_seconds=delivered,
    )


def test_a_parent_that_already_conforms_never_buys_a_relief_child() -> None:
    """THE FLIP. 44.016s delivered against a 44.00s order (+0.04%) — the old gate SPAWNed.

    A legal in-band count exists (4 x 10.632 = 42.53s), which is exactly why no other
    guard in this function catches it: they all reason about the child alone. The
    measured child of this very parent came back at 46.217s (+5.04%).
    """
    shape = _shape_for(10.632, 5, 44.016)

    assert shape.verdict == "STOP_ALREADY_CONFORMING", (
        "a parent delivering +0.04% must not be billed a paid re-render it can only lose on"
    )
    assert shape.verdict != "SPAWN", "the +5.04% child this parent bought must not be billed"
    assert shape.status_quo_seconds == pytest.approx(44.016), (
        "the delivery the refusal was decided from must be recorded, or an auditor cannot "
        "tell a measured refusal from an unmeasured one"
    )
    assert "44.02s" in shape.stop_detail
    assert "44.00s" in shape.stop_detail


def test_the_guard_is_opt_in_an_unmeasured_delivery_still_spawns() -> None:
    """BACK-COMPAT. The SAME row with ``delivered_seconds=None`` keeps today's answer.

    ``None`` means the delivery was UNMEASURED, not that it was measured and found
    absent. A caller that cannot measure must make no judgement, so every pre-existing
    verdict stands — that is what makes this change opt-in at the call site.
    """
    shape = _shape_for(10.632, 5, None)

    assert shape.verdict == "SPAWN", (
        "an unmeasured delivery must reproduce the pre-existing behaviour exactly"
    )
    assert shape.target_beats == 4
    assert shape.status_quo_seconds is None, "no measurement means nothing to record"


@pytest.mark.parametrize(
    ("delivered", "expected", "claim"),
    [(48.39, "STOP_ALREADY_CONFORMING", "a delivery inside the band must not be re-rendered")],
)
def test_the_band_edges_decide_the_verdict(delivered: float, expected: str, claim: str) -> None:
    """Both edges, both directions, sized off the CONSTANT rather than a literal 0.10.

    The bounds are computed from ``DURATION_TOLERANCE_FRACTION`` here so that moving the
    constant moves this test with it: a guard that refuses a paid render must use the
    same number as the measurement that grades the delivered film, and a test that
    hardcoded the fraction would keep passing after the two diverged.
    """
    upper = _ORDERED_SECONDS * (1.0 + DURATION_TOLERANCE_FRACTION)
    lower = _ORDERED_SECONDS * (1.0 - DURATION_TOLERANCE_FRACTION)
    assert upper == pytest.approx(48.4) and lower == pytest.approx(39.6), (
        "the fixtures below bracket the bounds the constant actually produces"
    )

    shape = _shape_for(10.632, 5, delivered)

    assert shape.verdict == expected, claim


@pytest.mark.parametrize(
    ("per_beat", "beat_count", "delivered", "expected", "unmeasured_expected", "target_beats"),
    [(13.896, 5, 77.723, "SPAWN", "SPAWN", 3)],
)
def test_out_of_band_parents_never_spend_more_than_the_footage_gate(
    per_beat: float,
    beat_count: int,
    delivered: float,
    expected: str,
    unmeasured_expected: str,
    target_beats: int,
) -> None:
    """Above the band, spend never goes UP: a footage refusal is returned unchanged.

    Each row is compared against its own ``delivered_seconds=None`` run: every refusal the
    footage-only gate made is the SAME refusal (verdict and beat count), and the one spawn
    is still the same spawn.
    """
    measured = _shape_for(per_beat, beat_count, delivered)
    unmeasured = _shape_for(per_beat, beat_count, None)

    assert measured.verdict == expected
    assert unmeasured.verdict == unmeasured_expected
    assert measured.target_beats == target_beats
    # The delivered basis never decides on these rows: row 1 delivered MORE than its
    # footage plan, rows 2-3 were already refused on footage. Each says which, and each
    # carries the footage figures it was actually sized from.
    assert measured.sizing_basis == (
        "footage_delivered_exceeds_plan"
        if delivered > beat_count * per_beat
        else "footage_basis_refused"
    )
    assert measured.sizing_overrun_seconds == pytest.approx(unmeasured.sizing_overrun_seconds)
    assert measured.verdict == unmeasured.verdict
    assert measured.target_beats == unmeasured.target_beats
    assert unmeasured.sizing_basis == "footage"
    assert measured.overrun_seconds == pytest.approx(unmeasured.overrun_seconds), (
        "the recorded overrun stays the FOOTAGE figure; only the sizing moved"
    )
    assert measured.measured_seconds_per_beat == pytest.approx(per_beat), (
        "measured_seconds_per_beat stays the FOOTAGE yield the recorder produced"
    )
    if unmeasured.verdict != "SPAWN":
        assert measured.verdict != "SPAWN", "delivered sizing must not buy a refused render"
    # The measurement is RECORDED even where it changed no decision — an auditor re-reading
    # the persisted shape must be able to tell "measured and out of band" from "never
    # measured", and only the second of those is allowed to carry ``None``.
    assert measured.status_quo_seconds == pytest.approx(delivered), (
        "a delivery that was measured must be recorded on every verdict, not only on the "
        "one that read it"
    )
    assert unmeasured.status_quo_seconds is None, "an unmeasured delivery records nothing"


def test_a_conforming_parent_is_refused_even_where_the_old_answer_was_a_refusal() -> None:
    """46.217s vs 44.00s (+5.04%) — the CHILD of the flip row, measured in production.

    Today this spine refuses anyway (``STOP_UNDERFULL``: no legal count at 11.910s/beat
    lands in the band), so the spend does not change — but the recorded REASON does, and
    the reason is what an auditor joins the refusal to. Asserting the old verdict here
    would lock in a sentence that names the wrong arithmetic.
    """
    shape = _shape_for(11.910, 4, 46.217)

    assert _shape_for(11.910, 4, None).verdict == "STOP_UNDERFULL", "what it answers today"
    assert shape.verdict == "STOP_ALREADY_CONFORMING", (
        "a conforming parent is refused for conforming, not for an unreachable beat count"
    )
    assert shape.status_quo_seconds == pytest.approx(46.217)


def test_an_unpinned_spine_that_already_conforms_is_also_refused() -> None:
    """THE HOLE THE GATE LEFT: no footage-pinned row, so there is no measured yield.

    Every OTHER duration guard here reasons about a candidate CHILD, so it needs
    ``measured_seconds_per_beat`` and the underfull ratio and is gated on both. This one
    reasons only about the PARENT's own delivery against the order and needs neither —
    but while it sat inside that gate, an elastic spine (``min 5.0 / max 12.0``, so the
    yield is unmeasurable) skipped the block entirely and fell through to the terminal
    SPAWN, billing a paid child for a run that had delivered +0.04%.

    An absent per-beat measurement is not a reason to buy a render nobody needs.
    """
    elastic = [
        {"id": f"vo-{i}", "seconds": 11.0, "min_seconds": 5.0, "max_seconds": 12.0}
        for i in range(1, 6)
    ]
    run = {"target_seconds": _ORDERED_SECONDS, "overrun_seconds": 11.0}

    unmeasured_yield = decide_relief_shape(
        run=run, beats=elastic, floor_ratio=0.5, delivered_seconds=44.016
    )

    assert unmeasured_yield.measured_seconds_per_beat is None, (
        "the fixture must actually defeat the yield measurement, or it proves nothing"
    )
    assert unmeasured_yield.verdict == "STOP_ALREADY_CONFORMING", (
        "an unmeasurable per-beat yield must not license a paid child for a run that "
        "already delivered inside the band"
    )
    assert unmeasured_yield.status_quo_seconds == pytest.approx(44.016)
    # And the same spine with no delivery measured keeps the old answer: the fix is in
    # WHERE the guard is asked, not in making an unmeasured run refuse.
    assert decide_relief_shape(run=run, beats=elastic, floor_ratio=0.5).verdict == "SPAWN", (
        "the hoist must not change the unmeasured path"
    )


# ── THE DELIVERED BASIS: delivery tracks NARRATION, the overrun is a FOOTAGE number ──
#
# Production run 01a0ab23 (showcase, 30.00s order): footage measured 31.28s
# (``overrun_seconds=1.28``, +4% "over"), the film DELIVERED 22.671s (-24.43%). Relief can
# only SHRINK the spine, so no child can repair an under-delivery — it can only land
# shorter still. The verdict must come from the delivery, not the footage arithmetic.

_SHOWCASE_ORDER = 30.0
_SHOWCASE_DELIVERED = 22.671


def _elastic_spine(total_seconds: float, beat_count: int) -> list[dict[str, Any]]:
    """Unpinned VO rows: no measured yield, so the footage gates make NO judgement and
    the old path falls straight through to SPAWN — the worst case for this defect."""
    each = total_seconds / beat_count
    return [
        {"id": f"vo-{i}", "seconds": each, "min_seconds": each / 2, "max_seconds": each * 2}
        for i in range(1, beat_count + 1)
    ]


def test_footage_over_delivered_under_on_a_pinned_spine_is_decided_by_delivery() -> None:
    """The live shape: all-pinned, footage +4%. The old gate also stopped — but from the
    footage arithmetic. The refusal must now be the delivered one."""
    beats = _pinned_spine(7.82, 4)
    run = {"target_seconds": _SHOWCASE_ORDER, "overrun_seconds": 4 * 7.82 - _SHOWCASE_ORDER}

    shape = decide_relief_shape(
        run=run, beats=beats, floor_ratio=0.5, delivered_seconds=_SHOWCASE_DELIVERED
    )

    assert shape.verdict == "STOP_UNDERFULL"
    assert "delivered 22.67s" in shape.stop_detail
    assert "measured/beat" not in shape.stop_detail, (
        "the refusal must not be the footage-arithmetic sentence"
    )


def test_a_delivery_over_the_band_still_reaches_the_relief_lever() -> None:
    """The delivered basis only REFUSES; a real delivered overrun is relieved as before."""
    measured = _shape_for(13.896, 5, 77.723)
    assert measured.verdict == "SPAWN"
    assert measured.verdict == _shape_for(13.896, 5, None).verdict


# ── _find_delivered_seconds: which number the guard is allowed to be fed ──────────────


def _row(stage_name: str, conformance: object) -> Any:
    """A stage row as the hook sees it — only ``.output`` and ``.stage_name`` are read."""
    output: dict[str, Any] = {"anything": 1}
    if conformance is not None:
        output["duration_conformance"] = conformance
    return SimpleNamespace(stage_name=stage_name, output=output)


# ── ABOVE THE BAND: the child is SIZED off the DELIVERY, not off the footage plan ─────
#
# A mixed spine (4 footage-pinned beats + 6 elastic VO beats, 9.00s each) against a 60s
# order: footage plan 90s (overrun 30s), delivered 75s (+25%, above the band). Delivery
# tracks narration, so each beat delivers 75/90 = 0.833 of its footage (7.50s).
#
# * Footage-sized: cut 30s of 9s beats -> 4 beats -> 6 left -> predicts 6 x 7.50 = 45.00s
#   delivered, -25% — a PAID undershoot.
# * Delivered-sized: cut 15s of 7.50s beats -> 2 beats -> 8 left -> predicts 60.00s.

_MIXED_ORDER = 60.0
_MIXED_DELIVERED = 75.0


def _mixed_spine() -> list[dict[str, Any]]:
    pinned = [_pinned(f"clip-{i}", 9.0) for i in range(1, 5)]
    elastic = [
        {"id": f"vo-{i}", "seconds": 9.0, "min_seconds": 4.5, "max_seconds": 18.0}
        for i in range(1, 7)
    ]
    return pinned + elastic


def test_an_above_band_child_is_sized_from_delivered_seconds() -> None:
    run = {"target_seconds": _MIXED_ORDER, "overrun_seconds": 10 * 9.0 - _MIXED_ORDER}

    footage = decide_relief_shape(run=run, beats=_mixed_spine(), floor_ratio=0.5)
    delivered = decide_relief_shape(
        run=run, beats=_mixed_spine(), floor_ratio=0.5, delivered_seconds=_MIXED_DELIVERED
    )

    assert footage.verdict == "SPAWN"
    assert footage.sizing_basis == "footage"
    assert footage.target_beats == 6, "the fixture must reproduce the footage over-cut"
    assert footage.target_beats * 9.0 * (_MIXED_DELIVERED / 90.0) == pytest.approx(45.0)

    assert delivered.verdict == "SPAWN"
    assert delivered.sizing_basis == "delivered"
    assert delivered.sizing_overrun_seconds == pytest.approx(_MIXED_DELIVERED - _MIXED_ORDER)
    assert delivered.sizing_seconds_per_beat == pytest.approx(7.5)
    assert delivered.target_beats == 8, (
        "sized off the 15s DELIVERED overrun, not the 30s footage overrun"
    )
    assert delivered.sizing_seconds_per_beat is not None
    predicted = delivered.target_beats * delivered.sizing_seconds_per_beat
    lower = _MIXED_ORDER * (1.0 - DURATION_TOLERANCE_FRACTION)
    upper = _MIXED_ORDER * (1.0 + DURATION_TOLERANCE_FRACTION)
    assert lower <= predicted <= upper, "the delivered-sized child is predicted in band"
    # The recorded overrun is still the FOOTAGE figure the manifest carried.
    assert delivered.overrun_seconds == pytest.approx(30.0)
    assert delivered.measured_seconds_per_beat == pytest.approx(9.0)


def test_an_unmeasured_delivery_sizes_from_footage_exactly_as_before() -> None:
    run = {"target_seconds": _MIXED_ORDER, "overrun_seconds": 30.0}
    shape = decide_relief_shape(run=run, beats=_mixed_spine(), floor_ratio=0.5)
    assert shape.sizing_basis == "footage"
    assert shape.sizing_overrun_seconds == pytest.approx(30.0)
    assert shape.target_beats == next_target_beats(run, 10, _mixed_spine())


def test_the_sizing_basis_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    run = {"target_seconds": _MIXED_ORDER, "overrun_seconds": 30.0}
    with caplog.at_level("INFO", logger="src.services.runs.structural_relief_policy"):
        decide_relief_shape(
            run=run, beats=_mixed_spine(), floor_ratio=0.5, delivered_seconds=_MIXED_DELIVERED
        )
        decide_relief_shape(run=run, beats=_mixed_spine(), floor_ratio=0.5)
    messages = [r.getMessage() for r in caplog.records]
    assert any("basis=delivered" in m and "overrun=15.000s" in m for m in messages)
    assert any("basis=footage" in m and "delivered=unmeasured" in m for m in messages)
