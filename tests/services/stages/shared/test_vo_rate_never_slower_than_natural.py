"""A line that fits its slot is spoken at natural speed 1.0 — never slowed.

Owner complaint: "narration slows down". ``vo_rate_for`` used to clamp any line with
slack to ``VO_RATE_DRAG_FLOOR`` (0.97) and pass a raw rate in ``[0.97, 1.0)``
through verbatim — both slower than natural. The seconds a line cannot fill at 1.0
belong to the PICTURE (``hold_seconds``), never to the voice.

The ``underspoken`` DETECTOR stays at ``natural_floor`` (0.97 default): it arms a paid
re-ask and ``gen.py`` sizes its word splits against it, so moving it to 1.0 would flag
rounding residue on flawless deliveries. Only the delivered RATE is floored at 1.0.
"""

from __future__ import annotations

import pytest

from src.services.stages.shared.time_budget import (
    DEFAULT_FREEZE_PAD_CEILING,
    DEFAULT_WPM_NATURAL,
    TimeDemand,
    reconcile_timeline,
    underspoken_hold,
    vo_rate_for,
)

_WORDS = 20
_NATURAL = _WORDS / DEFAULT_WPM_NATURAL * 60.0  # 8.00s


def test_thirty_percent_slack_is_spoken_at_one_and_flagged() -> None:
    allotment = _NATURAL / 0.70
    assert vo_rate_for(allotment, _WORDS) == (1.0, False, True)


def test_a_raw_rate_just_under_one_is_spoken_at_one() -> None:
    rate, unsayable, underspoken = vo_rate_for(_NATURAL / 0.985, _WORDS)
    assert rate == 1.0
    assert unsayable is False
    # Inside the detector's dead band: not a re-ask-worthy fault.
    assert underspoken is False


@pytest.mark.parametrize("raw", [0.25])
def test_no_line_with_slack_is_ever_delivered_below_natural(raw: float) -> None:
    rate, _, _ = vo_rate_for(_NATURAL / raw, _WORDS)
    assert rate == 1.0


def test_a_configured_floor_below_one_moves_the_detector_not_the_voice() -> None:
    rate, _, underspoken = vo_rate_for(_NATURAL / 0.85, _WORDS, natural_floor=0.80)
    assert rate == 1.0
    assert underspoken is False


def test_speed_up_side_is_unchanged() -> None:
    assert vo_rate_for(_NATURAL / 1.2, _WORDS) == pytest.approx((1.2, False, False))
    assert vo_rate_for(_NATURAL / 3.0, _WORDS)[:2] == (2.0, True)


def test_the_hold_covers_every_second_the_natural_voice_leaves() -> None:
    """The hold is sized against the DELIVERED rate (1.0), so a raw-0.985 beat's
    1.5% gap is held by the picture instead of silently shipping short."""
    allotment = _NATURAL / 0.985
    assert underspoken_hold(allotment, _WORDS) == pytest.approx(allotment - _NATURAL)
    demand = TimeDemand(
        id="b",
        kind="vo",
        min=allotment,
        ideal=allotment,
        max=allotment,
        elasticity=0.0,
        pinned=True,
        word_count=_WORDS,
    )
    (a,) = reconcile_timeline([demand], target=allotment)
    assert a.vo_rate == 1.0
    assert a.hold_seconds == pytest.approx(min(allotment - _NATURAL, DEFAULT_FREEZE_PAD_CEILING))
    # spoken at 1.0 + hold == the allotment, to the millisecond
    assert _NATURAL + a.hold_seconds + a.uncovered_seconds == pytest.approx(allotment)
