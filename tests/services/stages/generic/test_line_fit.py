"""Per-line closed loop (``generic/line_fit.py``) + its wiring through script_draft
and narrate: budget split, the three bands, the speed clamp, final-line
absorption, and narrate replaying the measured clip at the persisted speed."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.clients.schemas.tts import TTSSynthesis
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.narration_fit import fit_summary
from src.services.stages.generic import line_fit
from src.services.stages.generic.line_fit import (
    ACCEPT_BAND,
    MAX_REWRITES_PER_LINE,
    SPEED_EXPONENT,
    SPEED_MAX,
    SPEED_MIN,
    Attempt,
    LineTake,
    achievable,
    band_for,
    clamp_speed,
    corrective_speed,
    fit_line,
    fit_script,
    fit_speed,
    line_budgets,
    local_exponent,
    secant_ask,
    select_attempt,
    speed_seconds,
    target_words,
)
from src.services.stages.generic.narrate import NarrateStage
from src.services.stages.generic.script_draft import ScriptDraftStage
from src.services.tts_storage import deterministic_clip_id

#: The fake narrator: 2.5 words/second at speed 1.0, and kokoro's MEASURED response to a
#: speed — duration scales as ``speed ** -SPEED_EXPONENT``, not ``1/speed``.
WPS = 2.5


def _words(n: int) -> str:
    return " ".join(["word"] * n)


def _take(text: str, speed: float | None = None) -> LineTake:
    seconds = len(text.split()) / WPS * (speed or 1.0) ** -SPEED_EXPONENT
    return LineTake(text=text, seconds=seconds, speed=speed)


class _Recorder:
    """A fake ``synth`` / ``rewrite`` pair that records every paid call."""

    def __init__(self, rewrites: list[str | None] | None = None) -> None:
        self.synth_calls: list[tuple[str, float | None]] = []
        self.rewrite_calls: list[tuple[int, str, float, float, int]] = []
        self._rewrites = list(rewrites or [])

    async def synth(self, text: str, speed: float | None) -> LineTake:
        self.synth_calls.append((text, speed))
        return _take(text, speed)

    async def rewrite(
        self, index: int, text: str, measured: float, budget: float, want: int
    ) -> str | None:
        self.rewrite_calls.append((index, text, measured, budget, want))
        return self._rewrites.pop(0) if self._rewrites else None


# --- budget split -------------------------------------------------------------


def test_budgets_are_the_word_share_of_the_order_and_sum_to_it() -> None:
    budgets = line_budgets([_words(10), "", _words(30)], 20.0)
    assert budgets == pytest.approx([5.0, 0.0, 15.0])
    assert sum(budgets) == pytest.approx(20.0)


def test_no_order_or_no_words_budgets_nothing() -> None:
    assert line_budgets([_words(3)], 0.0) == [0.0]
    assert line_budgets(["", " "], 30.0) == [0.0, 0.0]


# --- the three bands ----------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "band"),
    [(10.0, "accept")],
)
def test_band_for(seconds: float, band: str) -> None:
    assert band_for(seconds, 10.0) == band


async def test_accept_band_keeps_the_measured_clip_and_pays_nothing() -> None:
    rec = _Recorder()
    fit = await fit_line(
        0, _take(_words(26)), 10.0, synth=rec.synth, rewrite=rec.rewrite, rewritable=True
    )  # 10.4s of 10s: +4%
    assert fit.take.speed is None and fit.take.seconds == pytest.approx(10.4)
    assert rec.synth_calls == [] and rec.rewrite_calls == []
    assert fit.initial_band == "accept"


async def test_speed_band_resynthesizes_once_at_the_calibrated_speed() -> None:
    rec = _Recorder()
    text = _words(27)  # 10.8s of 10.25s: +5.4%
    fit = await fit_line(
        0, _take(text), 10.25, synth=rec.synth, rewrite=rec.rewrite, rewritable=True
    )
    assert rec.rewrite_calls == []
    # (10.8 / 10.25) ** (1 / 0.6) = 1.091 — the 1/speed model would have asked 1.054,
    # which kokoro turns into 10.46s (+2.1%), not 10.25s.
    assert rec.synth_calls == [(text, 1.091)]
    assert fit.take.speed == 1.091 and fit.resynthesized and not fit.corrected
    assert fit.take.seconds == pytest.approx(10.25, rel=1e-3)


async def test_rewrite_band_rewrites_the_line_with_the_exact_word_target_then_clamps() -> None:
    rec = _Recorder(rewrites=[_words(26)])
    fit = await fit_line(
        2, _take(_words(40)), 10.0, synth=rec.synth, rewrite=rec.rewrite, rewritable=True
    )  # 16s of 10s: +60%
    # Told its measured length, its budget and the words it should carry.
    assert rec.rewrite_calls == [(2, _words(40), 16.0, 10.0, 25)]
    assert fit.take.text == _words(26) and fit.rewrites == 1
    # 10.4s after the rewrite: inside +/-5%, so no speed re-synth.
    assert rec.synth_calls == [(_words(26), None)]
    assert fit.take.speed is None


async def test_rewrites_are_bounded_then_the_speed_clamp_takes_the_rest() -> None:
    rec = _Recorder(rewrites=[_words(36), _words(33), _words(26)])
    fit = await fit_line(
        0, _take(_words(40)), 10.0, synth=rec.synth, rewrite=rec.rewrite, rewritable=True
    )
    assert len(rec.rewrite_calls) == MAX_REWRITES_PER_LINE == fit.rewrites
    # 33 words = 13.2s (+32%) after two rewrites -> clamped to 1.1, never faster.
    assert fit.take.text == _words(33)
    assert fit.take.speed == SPEED_MAX
    assert rec.synth_calls[-1] == (_words(33), SPEED_MAX)


async def test_a_rewrite_that_lands_further_away_is_discarded() -> None:
    rec = _Recorder(rewrites=[_words(60), None])
    fit = await fit_line(
        0, _take(_words(40)), 10.0, synth=rec.synth, rewrite=rec.rewrite, rewritable=True
    )
    assert fit.take.text == _words(40)
    assert fit.take.speed == SPEED_MAX


async def test_a_verbatim_line_is_never_rewritten_only_clamped() -> None:
    rec = _Recorder(rewrites=[_words(25)])
    fit = await fit_line(
        0, _take(_words(40)), 10.0, synth=rec.synth, rewrite=rec.rewrite, rewritable=False
    )
    assert rec.rewrite_calls == []
    assert fit.take.text == _words(40) and fit.take.speed == SPEED_MAX


async def test_a_failed_rewrite_and_a_failed_resynth_keep_the_take_in_hand() -> None:
    async def _boom(*_: Any) -> Any:
        raise RuntimeError("down")

    fit = await fit_line(0, _take(_words(40)), 10.0, synth=_boom, rewrite=_boom, rewritable=True)
    assert fit.take == _take(_words(40))
    assert not fit.resynthesized


# --- the speed clamp ----------------------------------------------------------


@pytest.mark.parametrize(("raw", "clamped"), [(0.5, SPEED_MIN)])
def test_speed_clamp(raw: float, clamped: float) -> None:
    assert clamp_speed(raw) == clamped


def test_fit_speed_subtracts_reported_fixed_silence() -> None:
    # 10.5s with 1.0s of fixed silence against 10s: only the 9.5s of speech moves,
    # so it must land on 9.0s -> (9.5/9) ** (1/0.6) = 1.094 (whole clip: 1.085).
    assert fit_speed(LineTake(text="x", seconds=10.5, pause_seconds=1.0), 10.0) == 1.094
    assert fit_speed(LineTake(text="x", seconds=10.5), 10.0) == 1.085
    assert fit_speed(LineTake(text="x", seconds=20.0), 10.0) == SPEED_MAX


def test_target_words_follows_the_line_s_own_pace() -> None:
    assert target_words(_words(40), 16.0, 10.0) == 25
    assert target_words(_words(10), 2.0, 4.0) == 20


# --- selection after the clamp + secant re-ask (live 01a15e2b-2a88 / 01a15e77-e3a4) ---


class _Table:
    """A narrator whose clip lengths are given per TEXT (at speed 1.0); a speed take
    scales by 1/speed. Lets a test fix each rewrite's measurement independently of its
    word count — exactly what the live model did (asked 24 words, delivered 31)."""

    def __init__(self, seconds: dict[str, float], rewrites: list[str]) -> None:
        self.seconds = seconds
        self.rewrites = list(rewrites)
        self.synth_calls: list[tuple[str, float | None]] = []
        self.rewrite_calls: list[tuple[str, int]] = []

    async def synth(self, text: str, speed: float | None) -> LineTake:
        self.synth_calls.append((text, speed))
        return LineTake(text=text, seconds=self.seconds[text] / (speed or 1.0), speed=speed)

    async def rewrite(
        self, _index: int, text: str, _measured: float, _budget: float, want: int
    ) -> str | None:
        self.rewrite_calls.append((text, want))
        return self.rewrites.pop(0) if self.rewrites else None


def _attempt(number: int, seconds: float, *, words: int, ask: int, source: int | None) -> Attempt:
    take = LineTake(text=_words(words), seconds=seconds, speed=1.0)
    speed, error = achievable(take, 10.0)
    return Attempt(
        number=number, take=take, words=words, ask=ask, source=source, speed=speed, error=error
    )


def test_achievable_is_the_miss_after_the_clamped_speed() -> None:
    # Under kokoro's response: -20% raw reaches only -14.8% at the 0.9 floor; +21% raw
    # reaches +14.3% at 1.1 (the 1/speed model claimed -11.1% / +10.0%).
    speed, err = achievable(LineTake(text="a", seconds=8.0, speed=1.0), 10.0)
    assert speed == SPEED_MIN and err == pytest.approx(8.0 * 0.9**-SPEED_EXPONENT / 10.0 - 1)
    speed, err = achievable(LineTake(text="b", seconds=12.1, speed=1.0), 10.0)
    assert speed == SPEED_MAX and err == pytest.approx(12.1 * 1.1**-SPEED_EXPONENT / 10.0 - 1)
    # In band: kept as synthesized, the raw miss is the miss.
    assert achievable(LineTake(text="c", seconds=10.3, speed=1.0), 10.0) == (
        1.0,
        pytest.approx(0.03),
    )


def test_selection_ranks_by_the_clamped_error_not_the_raw_one() -> None:
    first = _attempt(0, 8.0, words=20, ask=20, source=None)  # raw -20%, clamped -11.1%
    over = _attempt(1, 12.1, words=31, ask=25, source=0)  # raw +21%, clamped +10.0%
    assert select_attempt([first, over]) is over
    # A tie keeps the earlier take (no paid text change for nothing).
    assert select_attempt([first, first]) is first


def test_secant_interpolates_the_ask_between_an_under_and_an_over() -> None:
    # Live 01a15e77-e3a4 clip-5: 19 words = 7.267s, asked 24 -> 11.012s, window 9.04s.
    first = Attempt(0, LineTake(_words(19), 7.267, 1.0), 19, 19, None, 0.9, -0.1)
    over = Attempt(1, LineTake(_words(31), 11.012, 1.0), 31, 24, 0, 1.1, 0.1)
    assert secant_ask(first, [first, over], 9.04) == 21  # 19 + 1.773 * 5 / 3.745
    # No bracket (both under) -> no secant; the pace target stays in charge.
    under = Attempt(1, LineTake(_words(22), 8.0, 1.0), 22, 24, 0, 0.9, -0.05)
    assert secant_ask(first, [first, under], 9.04) is None
    # Only measurements OF the source count: a rewrite of another take is not its family.
    foreign = Attempt(2, LineTake(_words(40), 12.0, 1.0), 40, 30, 1, 1.1, 0.1)
    assert secant_ask(first, [first, under, foreign], 9.04) is None


def test_a_rewrite_used_as_source_anchors_the_secant_at_its_own_word_count() -> None:
    # Live 01a15e77-e3a4 clip-4: rw1 (28 words, asked 31) = 9.134s was rewritten again,
    # asked 33 -> 14.122s. The source's point is (28, 9.134), never its own ask (31).
    rw1 = Attempt(1, LineTake(_words(28), 9.134, 1.0), 28, 31, 0, 0.9, -0.07)
    rw2 = Attempt(2, LineTake(_words(44), 14.122, 1.0), 44, 33, 1, 1.1, 0.19)
    assert secant_ask(rw1, [rw1, rw2], 10.84) == 30  # 28 + 1.706 * 5 / 4.988


def test_secant_never_re_asks_the_count_the_source_already_has() -> None:
    first = Attempt(0, LineTake(_words(20), 9.0, 1.0), 20, 20, None, 0.9, -0.1)
    over = Attempt(1, LineTake(_words(40), 30.0, 1.0), 40, 30, 0, 1.1, 0.5)
    assert secant_ask(first, [first, over], 10.0) == 21  # 20.47 rounds to 20 -> bumped


async def test_an_overshooting_rewrite_re_asks_the_interpolated_target_not_the_same_one() -> None:
    orig, r1, r2 = _words(19), _words(31), _words(24)
    table = _Table({orig: 7.267, r1: 11.012, r2: 8.95}, rewrites=[r1, r2])
    fit = await fit_line(
        4,
        LineTake(orig, 7.267, 1.0),
        9.04,
        synth=table.synth,
        rewrite=table.rewrite,
        rewritable=True,
    )
    # First ask: the line's own pace (unchanged). Second: the secant, FROM THE ORIGINAL —
    # it used to re-send the identical 24-word ask and re-roll the same overshoot.
    assert table.rewrite_calls == [(orig, 24), (orig, 21)]
    assert fit.rewrites == 2 and fit.take.text == r2
    assert [a.source for a in fit.attempts] == [None, 0, 0]
    assert [a.ask for a in fit.attempts] == [19, 24, 21]


async def test_no_take_in_band_ships_the_least_clamped_error_not_the_original() -> None:
    orig, r1, r2 = _words(20), _words(31), _words(33)
    # raw: orig -20% (closest), r1 +21%, r2 +30%. Clamped: orig -11.1%, r1 +10.0%.
    table = _Table({orig: 8.0, r1: 12.1, r2: 13.0}, rewrites=[r1, r2])
    fit = await fit_line(
        0,
        LineTake(orig, 8.0, 1.0),
        10.0,
        synth=table.synth,
        rewrite=table.rewrite,
        rewritable=True,
    )
    assert fit.take.text == r1 and fit.take.speed == SPEED_MAX and fit.resynthesized
    assert fit.take.seconds == pytest.approx(11.0)
    assert table.synth_calls[-1] == (r1, SPEED_MAX)


async def test_a_speed_take_worse_than_a_raw_take_never_ships() -> None:
    orig = _words(20)

    async def _sluggish(text: str, speed: float | None) -> LineTake:
        # The narrator ignores the speed: the "re-synth" comes back LONGER.
        return LineTake(text=text, seconds=9.0 if speed else 8.0, speed=speed)

    fit = await fit_line(
        0,
        LineTake(orig, 8.0, 1.0),
        7.0,
        synth=_sluggish,
        rewrite=AsyncMock(return_value=None),
        rewritable=True,
    )
    assert fit.take.seconds == 8.0 and not fit.resynthesized


async def test_the_summary_maps_every_measurement_to_its_line() -> None:
    orig, r1, r2 = _words(19), _words(31), _words(24)
    table = _Table({orig: 7.267, r1: 11.012, r2: 8.95}, rewrites=[r1, r2])
    fit = await fit_line(
        0,
        LineTake(orig, 7.267, 1.0),
        9.04,
        synth=table.synth,
        rewrite=table.rewrite,
        rewritable=True,
    )
    summary = fit_summary([("clip-5", fit)])
    assert summary.startswith("clip-5 9.04s:")
    assert "#1 rw1<-#0 ask 24 31w 11.01s" in summary
    assert "#2 rw2<-#0 ask 21 24w 8.95s" in summary


# --- calibrated speed response + the corrective re-synth ----------------------

#: Measured same-text kokoro takes (live 2026-10-20/21): (seconds at 1.0, speed, seconds).
_KOKORO_TAKES = [
    (9.134, 0.900, 9.703),  # 01a15e77-e3a4 showcase clip-4
    (7.267, 0.900, 7.763),  # 01a15e77-e3a4 clip-5
    (9.879, 1.063, 9.493),  # 01a15d7c-ca2d clip-1
    (11.203, 1.088, 10.701),  # 01a15d7c-ca2d clip-4
    (5.709, 1.100, 5.413),  # 01a15b57-8bb4 generic
    (5.378, 0.939, 5.599),  # 01a15b57-8ed1 generic
]


@pytest.mark.parametrize(("natural", "speed", "measured"), [(9.134, 0.900, 9.703)])
def test_the_calibrated_response_predicts_measured_kokoro_takes(
    natural: float, speed: float, measured: float
) -> None:
    """Within 2.5% of what kokoro delivered — where ``1/speed`` missed by up to 5.4%."""
    predicted = speed_seconds(LineTake("x", natural, 1.0), speed)
    naive = natural / speed
    assert abs(predicted - measured) / measured < 0.025
    assert abs(predicted - measured) < abs(naive - measured)


def test_speed_seconds_composes_from_any_base_speed() -> None:
    at_one = LineTake("x", 10.0, 1.0)
    at_09 = LineTake("x", speed_seconds(at_one, 0.9), 0.9)
    assert speed_seconds(at_09, 1.1) == pytest.approx(speed_seconds(at_one, 1.1))


def test_local_exponent_is_the_line_s_own_response() -> None:
    first = LineTake("x", 10.0, 1.0)
    assert local_exponent(first, LineTake("x", 10.0 * 0.9**-0.4, 0.9)) == pytest.approx(0.4)
    # Same speed, ignored speed, inverted speed: no exponent, so no extra call.
    assert local_exponent(first, LineTake("x", 9.0, 1.0)) is None
    assert local_exponent(first, LineTake("x", 10.0, 1.1)) is None
    assert local_exponent(first, LineTake("x", 10.5, 1.1)) is None


def test_corrective_speed_resolves_from_the_line_s_own_exponent() -> None:
    # A speed take already in band is never corrected.
    assert corrective_speed(LineTake("x", 8.6, 1.0), LineTake("x", 9.0, 0.95), 9.04) is None
    # A stiff line (k=0.3): 9.0s at 1.0, asked 0.95 -> 9.14s, -8.6% of a 10s window.
    # Its own k says even 0.70 would be needed: the clamp floor, one call.
    stiff_first = LineTake("x", 9.0, 1.0)
    stiff_take = LineTake("x", 9.0 * 0.95**-0.3, 0.95)
    assert corrective_speed(stiff_first, stiff_take, 10.0) == SPEED_MIN
    # Already at the clamp the take was measured at: a repeat buys the same clip.
    floor = LineTake("x", 9.0 * 0.9**-0.3, 0.9)
    assert corrective_speed(stiff_first, floor, 10.0) is None


class _Narrator:
    """A fake kokoro whose response to speed is ``speed ** -exponent``."""

    def __init__(self, exponent: float, natural: float) -> None:
        self.exponent, self.natural = exponent, natural
        self.calls: list[float | None] = []

    async def synth(self, text: str, speed: float | None) -> LineTake:
        self.calls.append(speed)
        return LineTake(text, self.natural * (speed or 1.0) ** -self.exponent, speed)


async def test_a_speed_take_that_misses_gets_one_corrective_resynth() -> None:
    """A line far more responsive than the fleet (k=1.4; the measured spread is
    0.42-0.83): 9.45s of a 10s window, the calibrated 0.91 overshoots to +7.8%, and the
    line's OWN exponent re-solves the speed — one extra call, measured, in band."""
    narrator = _Narrator(1.4, 9.45)
    fit = await fit_line(
        0,
        LineTake(_words(25), 9.45, 1.0),
        10.0,
        synth=narrator.synth,
        rewrite=AsyncMock(return_value=None),
        rewritable=False,
    )
    assert len(narrator.calls) == 2  # one speed take + ONE corrective, never more
    first_speed, fix = narrator.calls
    assert first_speed == fit_speed(LineTake(_words(25), 9.45, 1.0), 10.0) == 0.91
    assert fix is not None and first_speed is not None and first_speed < fix < 1.0
    assert fit.corrected and fit.resynthesized and fit.take.speed == fix
    assert abs(fit.miss) <= ACCEPT_BAND
    # What ships is a MEASURED clip at the speed narrate will request.
    assert fit.take.seconds == pytest.approx(9.45 * fix**-1.4)


async def test_a_corrective_take_worse_than_the_first_never_ships() -> None:
    calls: list[float | None] = []

    async def _erratic(text: str, speed: float | None) -> LineTake:
        calls.append(speed)
        # 9.42s (-5.8%) first; the speed take misses at +5.5%; the corrective is worse.
        return LineTake(text, 10.55 if len(calls) == 1 else 11.0, speed)

    fit = await fit_line(
        0,
        LineTake(_words(25), 9.42, 1.0),
        10.0,
        synth=_erratic,
        rewrite=AsyncMock(return_value=None),
        rewritable=False,
    )
    assert len(calls) == 2
    assert fit.take.seconds == 10.55 and fit.take.speed == calls[0]
    assert fit.resynthesized and fit.corrected


async def test_no_corrective_call_when_the_first_speed_take_lands() -> None:
    narrator = _Narrator(SPEED_EXPONENT, 10.0)
    fit = await fit_line(
        0,
        LineTake(_words(25), 10.0, 1.0),
        9.48,
        synth=narrator.synth,
        rewrite=AsyncMock(return_value=None),
        rewritable=False,
    )
    assert len(narrator.calls) == 1 and not fit.corrected
    assert abs(fit.miss) <= 0.005


async def test_a_take_the_clamp_cannot_reach_is_rewritten_even_inside_the_speed_band() -> None:
    """+11.5% raw is inside the +/-12% speed band, but 1.1 only reaches +5.3% under the
    calibrated response — so the line is rewritten rather than shipped 5.3% long."""
    rec = _Recorder(rewrites=[_words(25)])
    fit = await fit_line(
        0, _take(_words(28)), 10.04, synth=rec.synth, rewrite=rec.rewrite, rewritable=True
    )  # 11.2s of 10.04s: +11.6%
    assert band_for(11.2, 10.04) == "speed"
    assert len(rec.rewrite_calls) == 1
    assert fit.take.text == _words(25) and abs(fit.miss) <= ACCEPT_BAND


# --- whole script: no-op band + final-line absorption -------------------------


async def test_a_script_already_within_five_percent_is_untouched() -> None:
    rec = _Recorder()
    takes = [_take(_words(20)), _take(_words(60)), _take(_words(19))]  # 39.6s of 40s
    fits = await fit_script(
        takes, 40.0, synth=rec.synth, rewrite=rec.rewrite, rewritable=[True] * 3
    )
    assert [f.take for f in fits] == takes
    assert rec.synth_calls == [] and rec.rewrite_calls == []


async def test_the_last_lines_absorb_what_the_head_actually_came_to() -> None:
    """Head line 0 is verbatim-pinned and +60% (16s of 10s): it can only be clamped
    to 1.1 (15.11s under kokoro's response). The two tail lines are then budgeted from what is LEFT of the
    order (30 - 14.545s), not their nominal 10s each, and fitted to it."""
    rec = _Recorder()
    takes = [_take(_words(40)), _take(_words(25)), _take(_words(25))]  # 16 + 10 + 10
    fits = await fit_script(
        takes, 30.0, synth=rec.synth, rewrite=rec.rewrite, rewritable=[False, False, False]
    )
    head = fits[0].take.seconds
    assert fits[0].take.speed == SPEED_MAX and head == pytest.approx(16.0 * 1.1**-SPEED_EXPONENT)
    left = 30.0 - head
    assert [f.budget for f in fits[1:]] == pytest.approx([left / 2, left / 2])
    # 10s against a 7.44s slot is +34%: not rewritable, so each is clamped to 1.1.
    assert all(f.take.speed == SPEED_MAX for f in fits[1:])
    total = sum(f.take.seconds for f in fits)
    assert total < sum(t.seconds for t in takes)  # 36s -> ~34s, as far as 1.1 reaches


async def test_absorption_lands_the_total_inside_the_band_when_the_clamp_allows() -> None:
    rec = _Recorder()
    # +7% overall: every head line goes to speed; the tail lines mop up the rest.
    takes = [_take(_words(w)) for w in (27, 27, 27, 26)]  # 10.8 * 3 + 10.4 = 42.8
    fits = await fit_script(
        takes, 40.0, synth=rec.synth, rewrite=rec.rewrite, rewritable=[True] * 4
    )
    total = sum(f.take.seconds for f in fits)
    assert abs(total - 40.0) / 40.0 <= ACCEPT_BAND
    assert rec.rewrite_calls == []
    assert all(1.0 < (f.take.speed or 1.0) <= SPEED_MAX for f in fits)


async def test_single_line_script_is_its_own_absorber() -> None:
    rec = _Recorder()
    fits = await fit_script(
        [_take(_words(27))], 10.0, synth=rec.synth, rewrite=rec.rewrite, rewritable=[True]
    )
    assert fits[0].budget == pytest.approx(10.0)
    # +8% needs 1.08 ** (1/0.6) = 1.14 under kokoro's response: clamped to 1.1 (+2.0%).
    assert fits[0].take.speed == SPEED_MAX


def test_module_constants_are_the_owner_approved_bands() -> None:
    assert (line_fit.ACCEPT_BAND, line_fit.SPEED_BAND) == (0.05, 0.12)
    assert (line_fit.SPEED_MIN, line_fit.SPEED_MAX) == (0.9, 1.1)
    assert line_fit.MAX_REWRITES_PER_LINE == 2
    assert line_fit.ABSORB_LINES == 2


# --- wiring: script_draft persists the speed, narrate replays that exact clip --


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj, output_type="object", output_schema_used=None, metadata=None, usage_metrics=None
    )


async def test_narrate_reuses_the_clip_id_measured_with_the_chosen_speed(
    _stub_generic_line_synth: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """script_draft fits a +5.3% line at speed 1.09 and persists it; narrate asks for
    the SAME (text, voice, speed) id — a cache hit — and synthesizes nothing."""
    text = _words(79)  # 79 words at 150 wpm = 31.6s of a 30s order: +5.3%
    wm = AsyncMock()
    wm.message.return_value = _resp({"lines": [{"text": text, "seconds": 30.0}]})
    voice = {"voice_id": "af_heart"}
    brief = {
        "title": "T",
        "angle": "a",
        "audience": "x",
        "target_duration_seconds": 30,
        "voice_descriptor": "calm",
    }
    run_id = uuid4()
    draft_ctx = RunContext(
        run_id=run_id,
        deliverable_id=uuid4(),
        config_snapshot={"voice": voice},
        prior_run_outputs={"brief_intake": brief},
    )
    draft = await ScriptDraftStage(client=wm).run({}, ctx=draft_ctx)
    (line,) = draft.lines
    assert line.speed == 1.09
    assert line.seconds == pytest.approx(30.0, rel=1e-3)
    stored = {
        deterministic_clip_id(c.kwargs["text"], c.kwargs["voice"], c.kwargs["speed"])
        for c in _stub_generic_line_synth.await_args_list
        if c.kwargs["stage_name"] == "narrate"
    }
    assert deterministic_clip_id(text, "af_heart", 1.09) in stored

    async def _find(run: str, stage: str, clip_id: str, audio_format: str = "wav") -> Any:
        assert (run, stage) == (str(run_id), "narrate")
        return (f"https://minio.test/narrate/{clip_id}.wav", 30.0) if clip_id in stored else None

    monkeypatch.setattr("src.services.stages.primitives.narration.find_existing_clip", _find)
    tts = AsyncMock()
    tts.synthesize.return_value = TTSSynthesis(
        audio_bytes=b"\x01", sample_rate=0, duration_seconds=99.0, provider="", voice=""
    )
    narrate_ctx = RunContext(
        run_id=run_id,
        deliverable_id=uuid4(),
        config_snapshot={"voice": voice},
        prior_run_outputs={"script_draft": draft.model_dump()},
    )
    out = await NarrateStage(client=tts).run({}, ctx=narrate_ctx)
    assert tts.synthesize.await_count == 0  # measured == shipped
    assert out.clips[0].id == deterministic_clip_id(text, "af_heart", 1.09)
    assert out.clips[0].speed == 1.09
    assert out.clips[0].seconds == pytest.approx(30.0)


def test_narrate_speed_for_ignores_absent_and_illegal_values() -> None:
    stage = NarrateStage(client=AsyncMock())
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    assert stage.speed_for({"id": "a", "text": "t"}, ctx) is None
    assert stage.speed_for({"speed": None}, ctx) is None
    assert stage.speed_for({"speed": True}, ctx) is None
    assert stage.speed_for({"speed": 3.0}, ctx) is None
    assert stage.speed_for({"speed": 0.95}, ctx) == 0.95
