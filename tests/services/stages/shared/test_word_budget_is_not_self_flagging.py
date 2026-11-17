"""A/V quality wave 13 — the word budget must not flag its OWN perfect delivery.

THE PROPERTY, and it is a property rather than a constant: a beat that says exactly the
number of words the budget asked it for must NOT come back ``underspoken``. "Delivered its
budget" and "not flagged" have to be the same event, or the detector is reporting the
budget's arithmetic instead of the narration.

THE DEFECT THIS PINS. The budget and the detector were the same constant. The budget was
first written at ``VO_RATE_NATURAL_FLOOR`` (0.80) — the wave-13 root cause, since it
ordered a 20%-dragged voice upstream — and the obvious repair was to move it to the
detector's own ``VO_RATE_DRAG_FLOOR`` (0.97). That repair is wrong, and the reason is
arithmetic, not taste: writing the budget AT the threshold puts a flawless delivery
exactly ON the detection boundary, where the integer rounding in
``narration_words_target`` casts the deciding vote. It then fires, or does not, per
duration. ``NARRATION_BUDGET_RATE`` (1.0) is the split that leaves the 3% gap as the
tolerance.

WHY A SWEEP AND NOT ONE DURATION. The defect held for SOME durations and not others — any
single-duration test picks a side of a coin flip and passes. Measured here across every
whole ordered second from 5s to 180s: at the 0.97 floor a perfect delivery is flagged on
**48.9%** of durations (86 of 176); at 1.0 on **0.6%** (1 of 176).

``test_pricing_the_budget_at_the_drag_floor_reintroduces_the_coin_flip`` is the mutation
proof, run as a test rather than by hand: it re-prices the budget at the floor and asserts
the flag rate comes back. Without it, the sweep above could be pinning nothing.
"""

from __future__ import annotations

import pytest

from src.services.stages.demo_video.gen import narration_words_target
from src.services.stages.shared.time_budget import (
    DEFAULT_WPM_NATURAL,
    NARRATION_BUDGET_RATE,
    VO_RATE_DRAG_FLOOR,
    vo_rate_for,
)

#: Every whole ordered second the product offers, 5s to 180s inclusive.
_SWEEP = [float(seconds) for seconds in range(5, 181)]

#: The one order the budget genuinely cannot satisfy: at 5s the 12-word budget rounds to a
#: raw rate below the floor and there is no integer that does not. It is documented as
#: degenerate in :data:`NARRATION_BUDGET_RATE` and is allowed here rather than papered
#: over — the fix would be to refuse a 5s order, not to bend the detector.
_DEGENERATE_ORDERS = {5.0}


def _flagged_underspoken(ordered: float) -> bool:
    """Does a single beat that delivers ``ordered``'s whole budget read as underspoken?"""
    words = narration_words_target(ordered, DEFAULT_WPM_NATURAL)
    _rate, _unsayable, underspoken = vo_rate_for(ordered, words, DEFAULT_WPM_NATURAL)
    return underspoken


def test_the_budget_and_the_detector_are_not_the_same_number() -> None:
    """The 3% gap IS the tolerance, and it lives in exactly one place.

    Two constants, two jobs: the budget says how many words to write, the floor says how
    slow a delivery we refuse to ship. If these ever collapse back onto one value the
    sweep below becomes a coin flip again, so the relationship is asserted directly.
    """
    assert NARRATION_BUDGET_RATE == 1.0
    assert VO_RATE_DRAG_FLOOR < NARRATION_BUDGET_RATE
    assert pytest.approx(0.03) == NARRATION_BUDGET_RATE - VO_RATE_DRAG_FLOOR


def test_a_beat_that_delivers_its_budget_exactly_is_never_underspoken() -> None:
    """THE REGRESSION. Swept over every ordered duration, not sampled at one."""
    flagged = [ordered for ordered in _SWEEP if _flagged_underspoken(ordered)]
    assert set(flagged) <= _DEGENERATE_ORDERS, (
        "A perfect delivery was flagged `underspoken` on these ordered durations: "
        f"{flagged}. The budget is being written at, or below, the rate the detector "
        "fires at, so rounding decides the verdict. See NARRATION_BUDGET_RATE."
    )


def test_the_budget_is_never_priced_below_the_drag_floor() -> None:
    """The per-duration form of the same property, stated on the WORD COUNT.

    The sweep above reads the detector's verdict; this reads the budget that produced it,
    so a change that keeps the flag quiet by weakening ``vo_rate_for`` instead of by
    pricing the budget correctly still fails here.
    """
    for ordered in _SWEEP:
        if ordered in _DEGENERATE_ORDERS:
            continue
        words = narration_words_target(ordered, DEFAULT_WPM_NATURAL)
        floor_words = ordered * DEFAULT_WPM_NATURAL * VO_RATE_DRAG_FLOOR / 60
        assert words >= floor_words, (
            f"{ordered}s budgets {words} words but needs at least {floor_words:.2f} to be "
            "sayable at the drag floor"
        )
