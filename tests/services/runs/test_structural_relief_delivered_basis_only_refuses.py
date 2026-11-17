"""The delivered sizing basis may only ever REFUSE MORE than the footage basis.

A relief child is a PAID re-render, so the contract of ``_sizing_basis`` is absolute: it
must never spawn a child the pre-change, footage-sized gate refused. It was violated
when the parent's delivery EXCEEDED its footage plan (``delivered > target + overrun``):
the delivered overrun is then larger than the footage one, the cut deepens, the shrink
target drops, the footage convergence floor drops with it, and a
``STOP_UNCONVERGEABLE`` became a ``SPAWN``.

Two proofs live here:

* the concrete counter-example an auditor produced, pinned as a regression;
* a deterministic DIFFERENTIAL test against a frozen copy of the pre-change decision
  function (:func:`_oracle_pre_delivered_sizing`).

The contract holds BY CONSTRUCTION: ``decide_relief_shape`` evaluates the footage basis
first and returns any refusal it makes unchanged; the delivered basis only decides where
footage would spawn. The differential test is the check on that construction.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

import pytest

from src.services.runs.structural_relief_policy import (
    decide_relief_shape,
    next_target_beats,
    protected_seconds_floor,
)
from src.services.stages.shared.duration_conformance import (
    DURATION_TOLERANCE_FRACTION,
)
from src.services.stages.shared.duration_conformance import (
    _usable as finite_or_none,
)
from src.services.stages.shared.duration_plan import (
    band_seconds,
    beat_counts_within_band,
    beats_for_seconds,
    measured_seconds_per_beat,
    plan_beats,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# ═════════════════════════════════════════════════════════════════════════════════════
# FROZEN REFERENCE ORACLE — TEST-ONLY. DO NOT "KEEP IN SYNC" WITH THE PRODUCTION CODE.
#
# A verdict-only port of ``decide_relief_shape`` + ``_delivered_basis_refusal`` +
# ``_already_conforming`` exactly as they stood at commit ``3f58cd2a`` — the last commit
# BEFORE the child was sized from ``delivered_seconds``
# (``git show 3f58cd2a:src/services/runs/structural_relief_policy.py``). It is the
# footage-only answer the delivered basis is contractually forbidden to out-spend.
#
# It returns ``(verdict, target_beats)`` and drops every field that is only reporting.
# The helpers it calls (``next_target_beats``, ``protected_seconds_floor`` and the
# ``duration_plan`` functions) are imported, not ported: they are byte-identical at
# ``3f58cd2a`` and at the commit that introduced delivered sizing.
# ═════════════════════════════════════════════════════════════════════════════════════


def _oracle_pre_delivered_sizing(  # noqa: C901 - a faithful port, not refactored
    *,
    run: Mapping[str, Any],
    beats: Sequence[object],
    floor_ratio: float = 0.0,
    delivered_seconds: float | None = None,
) -> tuple[str, int]:
    beat_count = len(beats)
    delivered = finite_or_none(delivered_seconds)
    target_beats = next_target_beats(run, beat_count, beats)
    overrun = finite_or_none(run.get("overrun_seconds")) or 0.0
    target_seconds = finite_or_none(run.get("target_seconds")) or 0.0
    if overrun <= 0 or target_beats >= beat_count:
        return "STOP_CANNOT_SHRINK", target_beats
    per_beat = measured_seconds_per_beat(beats)

    # ── _delivered_basis_refusal / _already_conforming, inlined ──
    if (
        delivered is not None
        and target_seconds > 0.0
        and abs(delivered - target_seconds) <= target_seconds * DURATION_TOLERANCE_FRACTION
    ):
        return "STOP_ALREADY_CONFORMING", target_beats
    if (
        delivered is not None
        and target_seconds > 0.0
        and delivered < target_seconds * (1.0 - DURATION_TOLERANCE_FRACTION)
    ):
        return "STOP_UNDERFULL", target_beats

    plan = plan_beats(
        target_beats,
        target_seconds=target_seconds,
        floor_ratio=floor_ratio,
        per_beat_seconds=per_beat,
        max_beats=beat_count - 1,
    )
    if plan.verdict == "UNSATISFIABLE":
        return "STOP_UNDERFULL", target_beats
    target_beats = plan.beats

    if floor_ratio > 0.0 and per_beat is not None and per_beat > 0.0 and target_seconds > 0.0:
        lower_bound = target_seconds * (1.0 - DURATION_TOLERANCE_FRACTION)
        tolerance_min_beats = beats_for_seconds(lower_bound, per_beat)
        if tolerance_min_beats > target_beats:
            if tolerance_min_beats >= beat_count:
                return "STOP_UNDERFULL", target_beats
            target_beats = tolerance_min_beats
        band = band_seconds(target_seconds, DURATION_TOLERANCE_FRACTION)
        in_band = beat_counts_within_band(
            per_beat_seconds=per_beat, max_beats=beat_count - 1, band=band
        )
        if not in_band:
            return "STOP_UNDERFULL", target_beats
        if target_beats not in in_band:
            pick = target_beats
            target_beats = min(in_band, key=lambda k: abs(k - pick))

    seconds_floor = protected_seconds_floor(beats, target_beats)
    if target_seconds > 0 and seconds_floor is not None and seconds_floor > target_seconds:
        return "STOP_UNCONVERGEABLE", target_beats
    return "SPAWN", target_beats


# ═════════════════════════════════════════════════════════════════════════════════════
# 1. The auditor's counter-example.
# ═════════════════════════════════════════════════════════════════════════════════════

_CX_RUN = {"target_seconds": 45.0, "overrun_seconds": 28.8}
_CX_FLOOR_RATIO = 0.5
_CX_DELIVERED = 90.0


def _cx_beats() -> list[dict[str, Any]]:
    return [
        {"id": "b1", "seconds": 34.6, "min_seconds": 26.1},
        {"id": "b2", "seconds": 31.0, "min_seconds": 24.4},
        {"id": "b3", "seconds": 34.5, "min_seconds": 23.3},
    ]


def test_a_delivery_above_the_footage_plan_never_buys_a_child_footage_refused() -> None:
    """90.0s delivered against a 73.8s footage plan: sized from delivery the cut deepened
    to 1 beat (floor 23.3s <= 45s) and SPAWNED a paid child; sized from footage it is a
    2-beat child whose 47.7s floor cannot fit the 45s order."""
    oracle = _oracle_pre_delivered_sizing(
        run=_CX_RUN, beats=_cx_beats(), floor_ratio=_CX_FLOOR_RATIO, delivered_seconds=_CX_DELIVERED
    )
    assert oracle == ("STOP_UNCONVERGEABLE", 2), "the fixture must reproduce the old refusal"
    assert _CX_RUN["target_seconds"] + _CX_RUN["overrun_seconds"] < _CX_DELIVERED

    shape = decide_relief_shape(
        run=_CX_RUN, beats=_cx_beats(), floor_ratio=_CX_FLOOR_RATIO, delivered_seconds=_CX_DELIVERED
    )

    assert shape.verdict == "STOP_UNCONVERGEABLE"
    assert shape.target_beats == 2
    assert shape.seconds_floor == pytest.approx(47.7)
    # The record says WHICH basis sized it and WHY, and the figures are footage figures.
    assert shape.sizing_basis == "footage_delivered_exceeds_plan"
    assert shape.sizing_overrun_seconds == pytest.approx(28.8)
    assert shape.status_quo_seconds == pytest.approx(_CX_DELIVERED)


def test_the_above_plan_guard_is_strict_and_both_sides_of_it_refuse() -> None:
    """The guard is ``delivered > plan``. AT the plan the delivery is eligible, but this
    parent is refused on footage first, so the row records that; a hair over, the row
    records the above-plan fallback. Same footage refusal either way."""
    plan_seconds = _CX_RUN["target_seconds"] + _CX_RUN["overrun_seconds"]
    at_plan = decide_relief_shape(
        run=_CX_RUN, beats=_cx_beats(), floor_ratio=_CX_FLOOR_RATIO, delivered_seconds=plan_seconds
    )
    just_over = decide_relief_shape(
        run=_CX_RUN,
        beats=_cx_beats(),
        floor_ratio=_CX_FLOOR_RATIO,
        delivered_seconds=plan_seconds + 0.01,
    )
    assert at_plan.sizing_basis == "footage_basis_refused"
    assert just_over.sizing_basis == "footage_delivered_exceeds_plan"
    assert at_plan.verdict == just_over.verdict == "STOP_UNCONVERGEABLE"


# ═════════════════════════════════════════════════════════════════════════════════════
# 2. Differential property test against the frozen oracle.
# ═════════════════════════════════════════════════════════════════════════════════════

_SEED = 20261005
_CASES = 20_000
_NAN = float("nan")

Case = tuple[dict[str, Any], list[object], float, float | None, str]


def _beat(rng: random.Random, index: int) -> object:
    seconds = round(rng.uniform(1.0, 40.0), 1)
    kind = rng.random()
    if kind < 0.40:  # footage-pinned: min == max == measured
        return {
            "id": f"b{index}",
            "seconds": seconds,
            "min_seconds": seconds,
            "max_seconds": seconds,
        }
    if kind < 0.75:  # elastic
        return {
            "id": f"b{index}",
            "seconds": seconds,
            "min_seconds": round(seconds * rng.uniform(0.3, 1.0), 1),
            "max_seconds": round(seconds * rng.uniform(1.0, 2.0), 1),
        }
    if kind < 0.90:  # no max
        return {
            "id": f"b{index}",
            "seconds": seconds,
            "min_seconds": round(seconds * rng.uniform(0.3, 1.0), 1),
        }
    if kind < 0.96:  # thin: nothing to judge from
        return {"id": f"b{index}"}
    return rng.choice([None, "junk", {"id": f"b{index}", "seconds": _NAN, "min_seconds": True}])


def _delivered(rng: random.Random, target: float, plan: float) -> tuple[float | None, str]:
    """A delivery and the REGION it was drawn from, so coverage can be asserted."""
    region = rng.choice(
        ["none", "zero", "negative", "nan", "inf", "under", "inside", "above_in_plan", "above_plan"]
    )
    band_top = target * (1.0 + DURATION_TOLERANCE_FRACTION)
    value: float | None
    if region == "none":
        value = None
    elif region == "zero":
        value = 0.0
    elif region == "negative":
        value = -rng.uniform(0.1, 100.0)
    elif region == "nan":
        value = _NAN
    elif region == "inf":
        value = rng.choice([math.inf, -math.inf])
    elif region == "under":
        value = target * rng.uniform(0.05, 0.899)
    elif region == "inside":
        value = target * rng.uniform(0.9, 1.1)
    elif region == "above_in_plan":
        value = rng.uniform(band_top, max(plan, band_top))
    else:
        value = max(plan, band_top) * rng.uniform(1.0001, 3.0)
    return value, region


def _case(rng: random.Random) -> Case:
    beats = [_beat(rng, i) for i in range(rng.randint(0, 8))]
    total = sum(
        float(b["seconds"])
        for b in beats
        if isinstance(b, dict)
        and isinstance(b.get("seconds"), float)
        and math.isfinite(b["seconds"])
    )
    shape = rng.random()
    if shape < 0.55 and total > 2.0:
        # The manifest identity: sum(seconds) == target + overrun.
        target = round(total * rng.uniform(0.2, 1.05), 1)
        overrun = total - target
    else:
        target = round(rng.uniform(5.0, 120.0), 1)
        overrun = round(rng.uniform(-10.0, 80.0), 1)
    run: dict[str, Any] = {"target_seconds": target, "overrun_seconds": overrun}
    corrupt = rng.random()
    if corrupt < 0.02:
        run["target_seconds"] = rng.choice([0.0, -5.0, _NAN, None])
    elif corrupt < 0.04:
        run["overrun_seconds"] = rng.choice([0.0, _NAN, None])
    elif corrupt < 0.05:
        del run[rng.choice(["target_seconds", "overrun_seconds"])]
    floor_ratio = rng.choice([0.0, 0.5, 0.5, 0.5, round(rng.uniform(0.05, 0.95), 2)])
    delivered, region = _delivered(rng, target, target + overrun)
    return run, beats, floor_ratio, delivered, region


def _cases() -> list[Case]:
    rng = random.Random(_SEED)  # noqa: S311 - a reproducible test corpus, not a secret
    return [_case(rng) for _ in range(_CASES)]


def _flips() -> list[tuple[Case, str, Any]]:
    """Every corpus case the frozen pre-change gate REFUSED and the new gate SPAWNS."""
    out: list[tuple[Case, str, Any]] = []
    for case in _cases():
        run, beats, floor_ratio, delivered, _region = case
        old_verdict, _old_beats = _oracle_pre_delivered_sizing(
            run=run, beats=beats, floor_ratio=floor_ratio, delivered_seconds=delivered
        )
        new = decide_relief_shape(
            run=run, beats=beats, floor_ratio=floor_ratio, delivered_seconds=delivered
        )
        if old_verdict != "SPAWN" and new.verdict == "SPAWN":
            out.append((case, old_verdict, new))
    return out


def _describe(flip: tuple[Case, str, Any]) -> str:
    (run, beats, floor_ratio, delivered, _region), old_verdict, new = flip
    return (
        f"{old_verdict} -> SPAWN({new.target_beats} beats) basis={new.sizing_basis} "
        f"run={run} floor_ratio={floor_ratio} delivered={delivered} beats={beats}"
    )


def test_the_corpus_can_observe_a_flip() -> None:
    """A zero below proves nothing unless the corpus populates both oracle outcomes,
    every delivery region and all three sizing bases — and the delivered basis is not
    inert (it does refuse renders the footage gate would have bought)."""
    oracle: dict[str, int] = {}
    regions: dict[str, int] = {}
    bases: dict[str, int] = {}
    newly_refused = 0
    for run, beats, floor_ratio, delivered, region in _cases():
        old_verdict, _ = _oracle_pre_delivered_sizing(
            run=run, beats=beats, floor_ratio=floor_ratio, delivered_seconds=delivered
        )
        new = decide_relief_shape(
            run=run, beats=beats, floor_ratio=floor_ratio, delivered_seconds=delivered
        )
        kind = "spawn" if old_verdict == "SPAWN" else "refuse"
        oracle[kind] = oracle.get(kind, 0) + 1
        regions[region] = regions.get(region, 0) + 1
        bases[new.sizing_basis] = bases.get(new.sizing_basis, 0) + 1
        newly_refused += old_verdict == "SPAWN" and new.verdict != "SPAWN"
    assert sum(oracle.values()) == _CASES
    assert oracle["refuse"] > 1_000, oracle
    assert oracle["spawn"] > 1_000, oracle
    assert len(regions) == 9, regions
    assert all(count > 500 for count in regions.values()), regions
    assert bases.get("delivered", 0) > 500, bases
    assert bases.get("footage_delivered_exceeds_plan", 0) > 500, bases
    assert bases.get("footage", 0) > 500, bases
    assert bases.get("footage_basis_refused", 0) > 100, bases
    assert newly_refused > 0


def test_the_delivered_basis_never_spawns_a_child_the_footage_gate_refused() -> None:
    """THE CONTRACT, WHOLE: zero refusal -> SPAWN flips across the corpus."""
    flips = _flips()
    assert not flips, f"{len(flips)} refusal->SPAWN flip(s); first: {_describe(flips[0])}"


def test_every_footage_refusal_is_returned_unchanged() -> None:
    """Stronger than "still a refusal": wherever the frozen footage gate refuses, the new
    gate returns the SAME verdict and beat count — the delivery never rewrites it."""
    checked = 0
    for run, beats, floor_ratio, delivered, _region in _cases():
        old = _oracle_pre_delivered_sizing(
            run=run, beats=beats, floor_ratio=floor_ratio, delivered_seconds=delivered
        )
        if old[0] == "SPAWN":
            continue
        new = decide_relief_shape(
            run=run, beats=beats, floor_ratio=floor_ratio, delivered_seconds=delivered
        )
        assert (new.verdict, new.target_beats) == old, (
            f"run={run} floor_ratio={floor_ratio} delivered={delivered} beats={beats}"
        )
        assert new.sizing_basis != "delivered"
        checked += 1
    assert checked > 1_000


_IN_PLAN_RUN = {"target_seconds": 57.3, "overrun_seconds": 63.6}
_IN_PLAN_DELIVERED = 117.5


def _in_plan_beats() -> list[dict[str, Any]]:
    return [
        {"id": "b0", "seconds": 30.4, "min_seconds": 30.4, "max_seconds": 30.4},
        {"id": "b1", "seconds": 32.8, "min_seconds": 32.4, "max_seconds": 65.6},
        {"id": "b2", "seconds": 24.8, "min_seconds": 22.0, "max_seconds": 28.3},
        {"id": "b3", "seconds": 32.9, "min_seconds": 32.9, "max_seconds": 32.9},
    ]


def test_an_in_plan_delivery_never_buys_a_child_the_footage_band_refused() -> None:
    """The second flip class, with the delivery INSIDE the footage plan. Footage 120.9s
    (= 57.3 + 63.6), delivered 117.5s. Footage yield 31.65s/beat: 1 beat = 31.65s, 2 =
    63.30s, band [51.57s, 63.03s] -> nothing fits -> STOP_UNDERFULL. Scaled by
    117.5/120.9 the yield is 30.76s/beat, 2 beats = 61.52s lands in band and the 2-beat
    footage floor (52.4s) fits the order — sized from delivery alone this SPAWNED."""
    oracle = _oracle_pre_delivered_sizing(
        run=_IN_PLAN_RUN,
        beats=_in_plan_beats(),
        floor_ratio=0.5,
        delivered_seconds=_IN_PLAN_DELIVERED,
    )
    assert oracle == ("STOP_UNDERFULL", 2), "the fixture must reproduce the old refusal"

    shape = decide_relief_shape(
        run=_IN_PLAN_RUN,
        beats=_in_plan_beats(),
        floor_ratio=0.5,
        delivered_seconds=_IN_PLAN_DELIVERED,
    )

    assert shape.verdict == "STOP_UNDERFULL"
    assert shape.target_beats == 2
    # Recorded as the FOOTAGE refusal it is, with the reason the delivery was not used.
    assert shape.sizing_basis == "footage_basis_refused"
    assert shape.sizing_overrun_seconds == pytest.approx(63.6)
    assert shape.sizing_seconds_per_beat == pytest.approx(31.65)
    assert "31.65s measured/beat" in shape.stop_detail
    assert shape.status_quo_seconds == pytest.approx(_IN_PLAN_DELIVERED)


def test_where_footage_spawns_the_child_is_still_sized_from_delivery() -> None:
    """NOT VACUOUS: the delivered basis still decides wherever footage allows a spawn.
    10 x 8.0s elastic beats (floor 2.0s each) against a 50s order, delivered 64s of the
    80s footage plan. Footage cuts 30s -> 4 beats dropped -> 6 left; delivery cuts only
    14s of 6.4s beats -> 3 dropped -> 7 left. Both SPAWN; the delivered-sized child keeps
    one more beat."""
    run = {"target_seconds": 50.0, "overrun_seconds": 30.0}
    beats = [
        {"id": f"b{i}", "seconds": 8.0, "min_seconds": 2.0, "max_seconds": 16.0} for i in range(10)
    ]
    assert _oracle_pre_delivered_sizing(run=run, beats=beats, floor_ratio=0.5) == ("SPAWN", 6)

    footage = decide_relief_shape(run=run, beats=beats, floor_ratio=0.5)
    delivered = decide_relief_shape(run=run, beats=beats, floor_ratio=0.5, delivered_seconds=64.0)

    assert (footage.verdict, footage.sizing_basis, footage.target_beats) == ("SPAWN", "footage", 6)
    assert delivered.verdict == "SPAWN"
    assert delivered.sizing_basis == "delivered"
    assert delivered.sizing_overrun_seconds == pytest.approx(14.0)
    assert delivered.target_beats == 7
    assert delivered.target_beats != footage.target_beats


def test_where_footage_spawns_the_delivered_basis_may_still_refuse() -> None:
    """The narrowing direction is real on the corpus: some footage SPAWNs are refused on
    the delivered basis, and every such row says the delivery decided it."""
    narrowed = 0
    for run, beats, floor_ratio, delivered, _region in _cases():
        new = decide_relief_shape(
            run=run, beats=beats, floor_ratio=floor_ratio, delivered_seconds=delivered
        )
        if new.sizing_basis != "delivered":
            continue
        old = _oracle_pre_delivered_sizing(
            run=run, beats=beats, floor_ratio=floor_ratio, delivered_seconds=delivered
        )
        assert old[0] == "SPAWN", "the delivered basis only ever runs where footage spawns"
        narrowed += new.verdict != "SPAWN"
    assert narrowed > 0


def test_a_delivery_above_the_footage_plan_decides_exactly_as_the_footage_gate() -> None:
    """Above the plan the fallback is the pre-change code, not merely "no more spend":
    same verdict AND same beat count as the oracle, on every such case."""
    checked = 0
    for run, beats, floor_ratio, delivered, _region in _cases():
        new = decide_relief_shape(
            run=run, beats=beats, floor_ratio=floor_ratio, delivered_seconds=delivered
        )
        if new.sizing_basis != "footage_delivered_exceeds_plan":
            continue
        checked += 1
        assert (new.verdict, new.target_beats) == _oracle_pre_delivered_sizing(
            run=run, beats=beats, floor_ratio=floor_ratio, delivered_seconds=delivered
        ), f"run={run} floor_ratio={floor_ratio} delivered={delivered} beats={beats}"
    assert checked > 500


# ═════════════════════════════════════════════════════════════════════════════════════
# 3. The pre-existing delivered verdicts still hold.
# ═════════════════════════════════════════════════════════════════════════════════════


def _ordered_seconds(run: Mapping[str, Any]) -> float | None:
    """The ordered seconds when the run can be judged against a delivery at all."""
    target = finite_or_none(run.get("target_seconds")) or 0.0
    return target if target > 0.0 else None


def test_the_band_verdicts_survive_on_every_generated_parent() -> None:
    """Inside +/-10% => STOP_ALREADY_CONFORMING; under it => STOP_UNDERFULL — on every
    generated parent that reaches the delivered guard (i.e. is not STOP_CANNOT_SHRINK)."""
    conforming = 0
    underfull = 0
    for run, beats, floor_ratio, delivered, region in _cases():
        target = _ordered_seconds(run)
        if target is None or region not in {"inside", "under", "zero", "negative"}:
            continue
        new = decide_relief_shape(
            run=run, beats=beats, floor_ratio=floor_ratio, delivered_seconds=delivered
        )
        if new.verdict == "STOP_CANNOT_SHRINK":
            continue
        assert delivered is not None
        if abs(delivered - target) <= target * DURATION_TOLERANCE_FRACTION:
            assert new.verdict == "STOP_ALREADY_CONFORMING", (run, delivered)
            conforming += 1
        else:
            assert new.verdict == "STOP_UNDERFULL", (run, delivered)
            underfull += 1
        assert new.sizing_basis == "footage"
    assert conforming > 300
    assert underfull > 300


def test_an_absent_or_unusable_delivery_is_identical_to_the_oracle() -> None:
    """None / 0 / negative / NaN / inf delivered: verdict AND beat count match the
    pre-change gate exactly, and nothing is sized from a delivery."""
    seen: dict[str, int] = {}
    for run, beats, floor_ratio, delivered, region in _cases():
        if region not in {"none", "zero", "negative", "nan", "inf"}:
            continue
        new = decide_relief_shape(
            run=run, beats=beats, floor_ratio=floor_ratio, delivered_seconds=delivered
        )
        assert (new.verdict, new.target_beats) == _oracle_pre_delivered_sizing(
            run=run, beats=beats, floor_ratio=floor_ratio, delivered_seconds=delivered
        ), f"run={run} floor_ratio={floor_ratio} delivered={delivered} beats={beats}"
        assert new.sizing_basis == "footage"
        seen[region] = seen.get(region, 0) + 1
    assert all(seen.get(r, 0) > 500 for r in ("none", "zero", "negative", "nan", "inf")), seen
