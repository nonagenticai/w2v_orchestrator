"""A non-finite duration is UNMEASURED — never a crash, never an impossibility claim.

``json.loads("1e400")`` is a legal JSON number that parses to ``inf``, and the
orchestrator's ``config_overrides`` is an unvalidated ``dict`` that round-trips whole
into ``direction`` and from there into this arithmetic. Nothing in the stack declares
``allow_inf_nan=False``, so ``nan``/``inf`` are input-reachable on an AUTHENTICATED
request with no DB corruption required.

Every ``<= 0.0`` guard in this arithmetic was blind to that: each comparison against
``nan`` is False, so the guard passes the value straight through to ``math.ceil``, which
raises ``ValueError`` (``nan``) / ``OverflowError`` (``inf``) in the middle of a
billing-relevant decision.

Two properties are pinned here, and the second matters more than the first:

1. The arithmetic does not raise.
2. The answer is ``UNMEASURED`` / "no band" / the empty flag set — "I could not
   measure", which is NOT "it cannot be done". ``UNSATISFIABLE``, a full-timeline
   infeasibility flag, and an empty in-band enumeration are all claims of impossibility,
   and this repo has shipped a false one before. A corrupt input must never manufacture
   another.

⛔ And no substituted default: a non-finite order becomes "no order", never a
zero-second order and never an assumed duration. Manufacturing an order nobody placed
is the exact defect ``duration_plan`` exists to delete.
"""

from __future__ import annotations

import math

import pytest

from src.services.stages.shared.duration_plan import (
    band_seconds,
    beat_counts_within_band,
    beats_for_seconds,
    exported_seconds_per_beat,
    floor_seconds,
    measured_seconds_per_beat,
    plan_beats,
)
from src.services.stages.shared.time_budget import (
    TimeDemand,
    infeasible_beats,
    tolerance_seconds,
    vo_rate_for,
)

NAN = float("nan")
INF = float("inf")


def _demand(beat_id: str, *, minimum: float = 1.0, pinned: bool = False) -> TimeDemand:
    return TimeDemand(
        id=beat_id,
        min=minimum,
        ideal=minimum,
        max=minimum * 2.0,
        elasticity=1.0,
        pinned=pinned,
        word_count=3,
        kind="vo",
    )


# ── duration_plan: the coercion and the count arithmetic ─────────────────────────────


@pytest.mark.parametrize("bad", [NAN])
def test_beats_for_seconds_cannot_size_a_nonfinite_order(bad: float) -> None:
    """0 = "cannot size". Before the guard this raised out of ``math.ceil``."""
    assert beats_for_seconds(bad, 4.0) == 0
    assert beats_for_seconds(30.0, bad) == 0


@pytest.mark.parametrize("bad", [NAN])
def test_plan_beats_reports_unmeasured_not_unsatisfiable(bad: float) -> None:
    """The verdict for an unreadable number is UNMEASURED — no judgement made.

    ``UNSATISFIABLE`` would assert that no beat count can deliver the order. Nothing
    here measured anything, so that claim would be manufactured out of a corrupt input.
    """
    by_target = plan_beats(
        3, target_seconds=bad, floor_ratio=0.5, per_beat_seconds=4.0, max_beats=5
    )
    assert by_target.verdict == "UNMEASURED"
    assert by_target.satisfiable is True
    assert by_target.beats == 3  # requested count untouched

    by_yield = plan_beats(
        3, target_seconds=30.0, floor_ratio=0.5, per_beat_seconds=bad, max_beats=5
    )
    assert by_yield.verdict == "UNMEASURED"
    assert by_yield.satisfiable is True


@pytest.mark.parametrize("bad", [NAN])
def test_floor_and_band_refuse_to_describe_a_nonfinite_order(bad: float) -> None:
    """ "No floor" / "no band" — the same answer an ABSENT target gets."""
    assert floor_seconds(bad, 0.5) == 0.0
    assert floor_seconds(30.0, bad) == 0.0
    assert band_seconds(bad, 0.1) == (0.0, 0.0)


def test_beat_counts_within_band_never_licenses_every_count_on_an_unmeasurable_band() -> None:
    """The FALSE-ACCEPTANCE direction — and the reason this guard is not cosmetic.

    An earlier draft of this file argued no test belonged here, on the grounds that a
    non-finite yield already enumerated to ``[]`` and that the empty list is the problem
    rather than the fix. That reasoning covers only HALF the input space and the other
    half is the dangerous one: with an UNBOUNDED UPPER EDGE, ``k * per_beat <= high`` is
    True for every ``k``, so the unguarded function returns ``[1 … max_beats]`` — a band
    nobody could measure certifying EVERY possible spine as conformant. A differential
    sweep over {nan, ±inf, 0, ±finite}³ finds 19 input combinations where the guard
    changes the answer, and in all 19 the unguarded answer is the over-permissive one.

    ``[]`` — "no legal count could be enumerated" — is the only honest reply. Note it is
    also the empty answer the caller reads as a refusal, which is exactly why the
    non-finite target is stopped UPSTREAM (``band_seconds`` returns the "no band"
    ``(0.0, 0.0)``) rather than being allowed to reach a refusal it never measured.
    """
    # Unbounded upper edge: the unguarded form answers [1, 2, 3, 4, 5] to all three.
    assert beat_counts_within_band(per_beat_seconds=7.0, max_beats=5, band=(-INF, INF)) == []
    assert beat_counts_within_band(per_beat_seconds=7.0, max_beats=5, band=(0.0, INF)) == []
    assert beat_counts_within_band(per_beat_seconds=7.0, max_beats=5, band=(27.0, INF)) == []
    # Unbounded LOWER edge against a finite ceiling: unguarded answers [1].
    assert beat_counts_within_band(per_beat_seconds=7.0, max_beats=5, band=(-INF, 10.0)) == []
    # A non-finite yield enumerates to nothing either way; pinned so the guard cannot
    # regress into returning a band-shaped answer for it.
    assert beat_counts_within_band(per_beat_seconds=INF, max_beats=5, band=(27.0, 33.0)) == []
    assert beat_counts_within_band(per_beat_seconds=7.0, max_beats=5, band=(NAN, NAN)) == []


def test_measured_seconds_per_beat_drops_nonfinite_rows_before_averaging() -> None:
    """``math.isclose(inf, inf)`` is True, so an ``inf`` row used to PIN as measured.

    One such row made the mean ``inf`` — a "measurement" of a yield nobody observed,
    which then propagated into the relief sizing as though it were ffprobe truth.
    """
    rows = [
        {"min_seconds": INF, "max_seconds": INF},
        {"min_seconds": 2.0, "max_seconds": 2.0},
        {"min_seconds": 4.0, "max_seconds": 4.0},
    ]
    measured = measured_seconds_per_beat(rows)
    assert measured == 3.0  # (2.0 + 4.0) / 2 — the inf row contributed nothing
    assert measured is not None and math.isfinite(measured)

    only_bad = measured_seconds_per_beat([{"min_seconds": INF, "max_seconds": INF}])
    assert only_bad is None  # UNMEASURED, not a yield of infinity


def test_exported_seconds_per_beat_drops_nonfinite_rows_before_averaging() -> None:
    assert exported_seconds_per_beat([INF, 2.0]) == 2.0
    assert exported_seconds_per_beat([INF]) is None
    assert exported_seconds_per_beat([NAN, 4.0]) == 4.0


# ── time_budget: the signals a corrupt target used to fake ────────────────────────────


def test_infeasible_beats_makes_no_judgement_against_a_nonfinite_target() -> None:
    """Not "every beat is infeasible" — that list DRIVES structural relief.

    Without the guard: ``sum(min) <= nan`` is False so the early-out is skipped,
    ``fair_share`` is ``nan``, ``d.min > nan`` is False for every beat, and the function
    falls through to its last line and returns the WHOLE elastic set.
    """
    demands = [_demand("a"), _demand("b"), _demand("c")]
    assert infeasible_beats(demands, NAN) == []
    assert infeasible_beats(demands, INF) == []
    # …and the real signal is untouched: Σmin = 3.0 over a 0.5s target is a real overrun.
    assert infeasible_beats(demands, 0.5) == ["a", "b", "c"]


def test_tolerance_seconds_reports_no_band_rather_than_a_nan_one() -> None:
    """A ``nan`` band width silently DISARMS ``is_underfull`` (every ``<`` is False)."""
    assert tolerance_seconds(NAN) == 0.0
    assert tolerance_seconds(INF) == 0.0
    assert tolerance_seconds(30.0, NAN) == 0.0


def test_vo_rate_for_never_returns_an_unflagged_nan_rate() -> None:
    """``(nan, False, False)`` is a rate no synthesizer accepts, carrying no fault flag.

    A non-finite allotment means there are no usable seconds to speak in — the same
    condition the zero/negative branch already names ``unsayable`` and routes to
    trim/compress.
    """
    rate, unsayable, underspoken = vo_rate_for(NAN, 10)
    assert math.isfinite(rate)
    assert unsayable is True
    assert underspoken is False

    rate, unsayable, _ = vo_rate_for(5.0, 10, wpm_natural=NAN)
    assert math.isfinite(rate)
    assert unsayable is True

    # Nothing to say is still neutral, non-finite allotment or not.
    assert vo_rate_for(NAN, 0) == (1.0, False, False)


# ── the no-behaviour-change half ──────────────────────────────────────────────────────


def test_finite_inputs_are_unchanged() -> None:
    """A guard that changes a FINITE answer is a regression, not a fix."""
    assert beats_for_seconds(30.0, 4.0) == 8
    assert floor_seconds(30.0, 0.5) == 15.0
    assert band_seconds(30.0, 0.1) == (27.0, 33.0)
    assert beat_counts_within_band(per_beat_seconds=10.0, max_beats=5, band=(27.0, 33.0)) == [3]
    assert (
        measured_seconds_per_beat(
            [{"min_seconds": 2.0, "max_seconds": 2.0}, {"min_seconds": 4.0, "max_seconds": 4.0}]
        )
        == 3.0
    )
    assert exported_seconds_per_beat([2.0, 4.0]) == 3.0
    assert tolerance_seconds(30.0) == pytest.approx(0.6)
    # 10 words / 150 wpm = 4.00s natural in a 5.00s allotment ⇒ raw rate 0.80: the line
    # FITS, so it is spoken at natural 1.0 (never slowed) and raises `underspoken`.
    assert vo_rate_for(5.0, 10) == pytest.approx((1.0, False, True))
    assert infeasible_beats([_demand("a"), _demand("b")], 10.0) == []

    plan = plan_beats(4, target_seconds=30.0, floor_ratio=0.5, per_beat_seconds=4.0, max_beats=8)
    assert plan.verdict == "OK"
    assert plan.beats == 4
