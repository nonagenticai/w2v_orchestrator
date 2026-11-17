"""action_reel cut_plan — the beat-fit solver.

The centre of gravity here is the DRIFT property: a solver that sums per-shot
durations passes a three-shot test and fails a twenty-shot one, so the headline
test is deliberately long. Everything else (speed bounds, the repair ladder, the
no-grid degradation) hangs off the same fixture style: plain lists in, plain
dicts out, no RunContext — which is exactly why the solver is a pure function.

Boundaries are snapped to the output frame grid, so "this cut is on the beat"
is asserted to within HALF A FRAME (~20.9 ms at 23.976 fps) rather than to the
float. That is not a weakened assertion: the tolerance is a fixed, per-cut
bound, and the property these tests actually defend is that it never GROWS with
shot index. A tolerance that accumulated would be the bug.
"""

from __future__ import annotations

import math
import re
from itertools import pairwise
from typing import Any
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.action_reel.cut_plan import (
    COMPILER_SPEED_MIN,
    DEFAULT_FRAME_RATE,
    MIN_SHOT_SECONDS,
    PREFERRED_SPEED_MAX,
    PREFERRED_SPEED_MIN,
    SPEED_MAX,
    SPEED_MIN,
    CutPlanInput,
    CutPlanStage,
    solve_cut_plan,
)
from src.services.stages.base import StagePreconditionError

#: How far a "no warp" (repair (b)) speed may sit from exactly 1.0. Not slack in
#: the edit: source in/out points are emitted at millisecond resolution and
#: ``speed`` is derived FROM that emitted pair (so the renderer's
#: ``(out - in) / speed`` reproduces the beat window exactly), which puts at most
#: half a millisecond of the trim's rounding into the speed. Over the shortest
#: legal shot that is <0.1%, i.e. invisible — where a millisecond of over-run,
#: landing in the only channel that sets shot length, is not.
_TRIM_SPEED_TOLERANCE = 0.001 / MIN_SHOT_SECONDS

_BPM = 120.0
_IDEAL_BEAT = 60.0 / _BPM  # 0.5s
_FRAME = 1.0 / DEFAULT_FRAME_RATE  # 33.3ms at the 30 fps default
#: The worst a single boundary may sit from the beat it represents: one snap to
#: the nearest frame, and nothing more, however long the reel gets.
_HALF_FRAME = _FRAME / 2.0


def _snap(seconds: float) -> float:
    """The frame-grid position the solver is expected to emit for ``seconds``."""
    return round(seconds * DEFAULT_FRAME_RATE) / DEFAULT_FRAME_RATE


def _grid(count: int = 240, *, jitter: float = 0.0) -> list[float]:
    """A 120 BPM beat list. ``jitter`` displaces each beat deterministically so
    the grid is NOT a uniform ``60/bpm`` ruler — real analyzer output never is,
    and a solver that idealises the grid must not survive these tests."""
    return [round(i * _IDEAL_BEAT + jitter * math.sin(i * 1.7), 6) for i in range(count)]


def _grid_at(beat_seconds: float, *, count: int = 160) -> list[float]:
    """A grid at an arbitrary tempo. ``_grid`` is pinned to 120 BPM (0.5s beats),
    which only offers 1.0/2.0/4.0s phrase windows — too coarse to build a
    fixture whose reel lands just past a 45s ask on a chosen number of shots.
    Same deterministic jitter, because a uniform ruler is not a real grid."""
    return [round(i * beat_seconds + 0.012 * math.sin(i * 1.7), 6) for i in range(count)]


def _downbeats(beats: list[float]) -> list[float]:
    return beats[::4]


def _segments(lengths: list[float], *, source_id: str = "src-1", score: float = 0.6) -> list[dict]:
    """Segments at MICROSECOND resolution, deliberately.

    This used to round to 3 decimals, which made every fixture in the file a
    whole number of milliseconds — and the solver emits its trim ms-FLOORED, so a
    whole-ms source rounds to itself and no test could ever see the millisecond
    the floor takes off. That is precisely how the speed fence below missed a
    sub-floor speed: not because the sweep was too narrow, but because the
    fixtures could not express the input that breaks it. Upstream in/out points
    come from an analyzer's float seconds and carry no such alignment.
    """
    out: list[dict] = []
    cursor = 0.0
    for i, length in enumerate(lengths):
        out.append(
            {
                "id": f"seg-{i}",
                "source_id": source_id,
                "in_seconds": round(cursor, 6),
                "out_seconds": round(cursor + length, 6),
                "seconds": round(length, 6),
                "score": score,
                "rationale": f"moment {i}",
            }
        )
        cursor += length + 1.0
    return out


def _f(shot: dict[str, object], key: str) -> float:
    """Typed read of a solver-emitted number. The solver returns plain
    ``dict[str, object]`` on purpose (no pydantic in the hot path), so the tests
    narrow rather than blind-cast — a wrong type here is itself a failure."""
    value = shot[key]
    assert isinstance(value, int | float), f"{key} is {type(value).__name__}"
    return float(value)


def _i(shot: dict[str, object], key: str) -> int:
    value = shot[key]
    assert isinstance(value, int), f"{key} is {type(value).__name__}"
    return value


def _nearest_beat_error(beats: list[float], moment: float) -> float:
    return min(abs(b - moment) for b in beats)


def test_cut_times_are_absolute_beat_times_and_do_not_drift() -> None:
    """THE regression test. Twenty shots on a jittery grid: every cut time must
    still be an actual beat time, and the error must not grow with shot index —
    which is precisely what a cumulative (sum-of-durations) solver cannot do."""
    beats = _grid(jitter=0.012)
    segments = _segments([2.0] * 20)

    shots, notes, _, *_ = solve_cut_plan(segments, beats, downbeats=_downbeats(beats))

    assert len(shots) == 20, notes
    errors: list[float] = []
    # Reconstruct the timeline the way a naive consumer would — by ADDING the
    # emitted durations — and demand it still lands on real beats.
    clock = _f(shots[0], "start_seconds")
    for i, shot in enumerate(shots):
        # Each boundary is its own beat time snapped to a frame — so it sits
        # within half a frame of the beat, and the shot INDEX does not appear in
        # that bound anywhere.
        assert shot["start_seconds"] == pytest.approx(
            beats[_i(shot, "beat_index")], abs=_HALF_FRAME
        ), i
        error = abs(clock - _f(shot, "start_seconds"))
        assert error < 1e-3, f"shot {i} cut time drifted {error * 1000:.3f}ms off the grid"
        assert _nearest_beat_error(beats, clock) <= _HALF_FRAME + 1e-9
        errors.append(error)
        clock += _f(shot, "seconds")

    # Error must be flat, not a ramp: the last shots may not be worse than the
    # first ones by anything beyond float noise.
    assert max(errors[-5:]) <= max(errors[:5]) + 1e-6


def test_seconds_is_always_a_difference_of_two_frame_snapped_beat_times() -> None:
    beats = _grid(jitter=0.02)
    shots, *_ = solve_cut_plan(_segments([1.9, 3.8, 2.1, 4.2, 1.0, 7.6]), beats)

    assert shots
    for shot in shots:
        start_index = _i(shot, "beat_index")
        end = _f(shot, "start_seconds") + _f(shot, "seconds")
        # The end lands on some LATER beat, snapped to a frame — to the frame,
        # not to a loose tolerance that would hide an idealised grid.
        assert _nearest_beat_error(beats, end) <= _HALF_FRAME + 1e-9
        assert end == pytest.approx(_snap(end), abs=1e-9)
        assert beats[start_index] == pytest.approx(_f(shot, "start_seconds"), abs=_HALF_FRAME)


def _offset_grid(head: float, count: int = 240) -> list[float]:
    """A 120 BPM grid whose FIRST beat is at ``head`` rather than at zero — i.e.
    a bed with a musical intro, which is what a real track has and what the
    zero-anchored fixture above cannot exercise."""
    return [round(head + i * _IDEAL_BEAT, 6) for i in range(count)]


def test_the_reel_is_anchored_at_video_zero_not_at_the_first_beat() -> None:
    """The bed is laid at track t=0 against video t=0 with no offset field, so a
    plan that started shot 0 at ``beats[0]`` would put EVERY cut ``beats[0]``
    early. Shot 0 covers the intro instead, and each later boundary is the beat
    time the viewer actually hears."""
    beats = _offset_grid(1.3)
    shots, notes, _, *_ = solve_cut_plan(_segments([2.0] * 6), beats)

    assert len(shots) == 6, notes
    assert _f(shots[0], "start_seconds") == 0.0
    # The old, off-by-the-first-beat behaviour, named so a regression to it is
    # unmistakable rather than a silent 1.3s slip.
    assert _f(shots[0], "start_seconds") != pytest.approx(beats[0])
    # The opening shot's LENGTH is the first cut boundary — the intro is inside
    # it, not skipped and not silently dropped from the timeline.
    first_cut = _f(shots[1], "start_seconds")
    assert _f(shots[0], "seconds") == pytest.approx(first_cut, abs=1e-9)
    assert first_cut > beats[0]

    for shot in shots[1:]:
        assert _f(shot, "start_seconds") == pytest.approx(
            beats[_i(shot, "beat_index")], abs=_HALF_FRAME
        )


def test_a_first_beat_near_zero_cannot_mint_a_flicker_opening_shot() -> None:
    """The degenerate end of the anchoring: a grid whose first beat is 8 ms in.
    The origin is folded into the window before the minimum-length filter, so
    the opening shot is chosen from the same candidates as any other and can
    never come out as a sub-``MIN_SHOT_SECONDS`` sliver of intro."""
    beats = _offset_grid(0.008)
    shots, notes, _, *_ = solve_cut_plan(_segments([2.0] * 6), beats)

    assert len(shots) == 6, notes
    assert _f(shots[0], "start_seconds") == 0.0
    assert all(_f(s, "seconds") >= MIN_SHOT_SECONDS for s in shots)
    # And it is a normal shot, not a special-cased one: it is the same length as
    # the reel's other 2.0s segments, to within the frame snap.
    assert _f(shots[0], "seconds") == pytest.approx(_f(shots[1], "seconds"), abs=_FRAME)


def test_every_boundary_lands_on_the_output_frame_grid() -> None:
    """Shots are encoded separately and concatenated, so the renderer can only
    make whole frames. Asking for anything else means the renderer rounds — and
    ITS roundings prefix-sum down the reel, which is the drift this module
    exists to prevent, re-entering through the back door."""
    beats = _grid(jitter=0.018)
    shots, *_ = solve_cut_plan(_segments([1.8, 2.0, 4.1, 2.6, 3.9, 2.0]), beats)

    assert shots
    for i, shot in enumerate(shots):
        start = _f(shot, "start_seconds")
        end = start + _f(shot, "seconds")
        assert start == pytest.approx(_snap(start), abs=1e-9), f"shot {i} start off-frame"
        assert end == pytest.approx(_snap(end), abs=1e-9), f"shot {i} end off-frame"
        # Whole frames of duration, which is what the renderer will produce.
        frames = _f(shot, "seconds") * DEFAULT_FRAME_RATE
        assert frames == pytest.approx(round(frames), abs=1e-6)


def test_a_non_monotonic_beat_grid_is_rejected_rather_than_silently_solved() -> None:
    """``bisect_left`` and every ``beats[i + k] - beats[i]`` assume sorted input;
    an unsorted grid does not raise on its own, it yields a plan whose cuts are
    out of order. It is an upstream defect, so it surfaces as a precondition."""
    segments = _segments([2.0, 2.0])
    good = _grid(count=40)

    for broken, label in (
        ([*good[:10], good[8], *good[10:]], "out of order"),
        ([*good[:10], good[9], *good[10:]], "duplicated"),
        ([-0.5, *good[1:]], "negative head"),
    ):
        with pytest.raises(StagePreconditionError) as caught:
            solve_cut_plan(segments, broken)
        assert "beat" in str(caught.value), label


def test_shots_are_contiguous_no_gaps_no_overlaps() -> None:
    beats = _grid(jitter=0.015)
    shots, *_ = solve_cut_plan(_segments([2.0, 4.1, 1.8, 3.9, 2.2]), beats)

    for previous, following in pairwise(shots):
        expected = _f(previous, "start_seconds") + _f(previous, "seconds")
        assert _f(following, "start_seconds") == pytest.approx(expected, abs=1e-9)


def test_jittery_grid_is_tracked_not_idealised() -> None:
    """With 25ms of jitter, at least one emitted duration must differ measurably
    from the ideal ``span * 60/bpm`` — otherwise the solver synthesised a ruler."""
    beats = _grid(jitter=0.025)
    shots, *_ = solve_cut_plan(_segments([2.0] * 12), beats)

    deviations = [
        abs(_f(s, "seconds") - round(_f(s, "seconds") / _IDEAL_BEAT) * _IDEAL_BEAT) for s in shots
    ]
    assert max(deviations) > 0.005


def test_speeds_stay_inside_the_hard_band_and_mostly_inside_the_preferred_one() -> None:
    beats = _grid(jitter=0.01)
    lengths = [1.8, 2.2, 3.9, 4.4, 2.0, 7.8, 1.1, 3.6, 2.5, 4.0, 1.9, 8.2]
    shots, *_ = solve_cut_plan(_segments(lengths), beats)

    assert len(shots) == len(lengths)
    speeds = [_f(s, "speed") for s in shots]
    assert all(SPEED_MIN <= speed <= SPEED_MAX for speed in speeds), speeds
    invisible = [s for s in speeds if PREFERRED_SPEED_MIN <= s <= PREFERRED_SPEED_MAX]
    assert len(invisible) >= len(speeds) * 0.8, speeds


def test_speed_matches_the_source_span_over_the_beat_window() -> None:
    beats = _grid(jitter=0.01)
    shots, *_ = solve_cut_plan(_segments([2.0, 3.3, 4.7]), beats)

    for shot in shots:
        source_len = _f(shot, "out_seconds") - _f(shot, "in_seconds")
        assert _f(shot, "speed") == pytest.approx(source_len / _f(shot, "seconds"), abs=1e-3)


def test_minimum_rendered_shot_length_is_respected() -> None:
    beats = _grid(jitter=0.01)
    shots, *_ = solve_cut_plan(_segments([0.9, 1.0, 0.8, 1.2]), beats)

    assert all(_f(s, "seconds") >= 0.6 for s in shots)


def test_oversized_segment_is_trimmed_rather_than_sped_up_past_the_band() -> None:
    """Repair (b): a source far too long for any phrase span is trimmed inward
    (head kept) and plays without a warp, instead of being run at 3x."""
    beats = _grid(jitter=0.01)
    segments = _segments([12.0])

    shots, notes, _, *_ = solve_cut_plan(segments, beats)

    assert len(shots) == 1
    shot = shots[0]
    trimmed = _f(shot, "out_seconds") - _f(shot, "in_seconds")
    assert trimmed < 12.0
    assert _f(shot, "in_seconds") == pytest.approx(_f(segments[0], "in_seconds"))
    assert _f(shot, "speed") == pytest.approx(1.0, abs=_TRIM_SPEED_TOLERANCE)
    assert shot["beat_locked"] is True
    # A trim is the solver doing its job, not a degradation — no note, so the run
    # does not get flagged degraded for a normal edit.
    assert notes == []


def test_undersized_segment_still_yields_a_valid_plan() -> None:
    """Repair (c): too short for a phrase span, but an off-phrase span lands it
    inside the hard band — still beat-locked, still contiguous."""
    beats = _grid(jitter=0.01)
    shots, *_ = solve_cut_plan(_segments([0.7, 2.0, 0.75]), beats)

    assert len(shots) == 3
    assert all(SPEED_MIN <= _f(s, "speed") <= SPEED_MAX for s in shots)
    assert all(s["beat_locked"] is True for s in shots)


def test_off_beat_fallback_is_flagged_not_silent() -> None:
    """Repair (d): a segment no window can hold at a legal speed is still placed
    on the grid, but stamped ``beat_locked=False`` AND noted.

    The 0.4s source is chosen so the clamp genuinely fires (its 0.4x over the
    narrowest 1.0s window this grid offers is under the editorial ``SPEED_MIN``)
    while the emitted speed still clears the compiler's floor. It read 0.2s until
    the bound below was corrected — see that comment.
    """
    beats = _grid(jitter=0.01)
    shots, notes, _, *_ = solve_cut_plan(_segments([0.4, 2.0]), beats)

    assert shots[0]["beat_locked"] is False
    # HISTORY, so that nobody widens this again. It first read
    # `SPEED_MIN <= speed <= SPEED_MAX`. In 2026-08 the LOWER bound was
    # deliberately widened to `0.0 < speed`, reasoning that it was only ever
    # reachable by trimming PAST the segment's approved out-point and that "a
    # source too short for any window now plays BELOW SPEED_MIN, on a shot that
    # is already flagged and noted". The premise was right and the conclusion was
    # wrong: it weighed ONE floor, the editorial `SPEED_MIN`, and there are two.
    # `compile` forwards `speed` verbatim and the compiler's own contract
    # (`tests/contracts/compile_request.schema.json` `$defs/Shot.speed.minimum`)
    # rejects anything under `COMPILER_SPEED_MIN`, for the WHOLE shots list at
    # once. So `0.0 <` was pinning a shot the renderer refuses to render — this
    # very fixture emitted 0.2x — i.e. a reel that dies after five human review
    # gates approved it. Below the compiler floor there is no legal shot to emit,
    # only a segment to leave unplaced, which is what the solver now does.
    assert COMPILER_SPEED_MIN <= _f(shots[0], "speed") <= SPEED_MAX
    assert any("beat_locked=False" in note for note in notes)
    # The compromise is local: the reel stays on the grid after it.
    assert shots[1]["beat_locked"] is True
    assert _f(shots[1], "start_seconds") == pytest.approx(
        _f(shots[0], "start_seconds") + _f(shots[0], "seconds"), abs=1e-9
    )


def _assert_render_equation(shots: list[dict[str, object]], label: str = "") -> None:
    """THE contract, asserted on the emitted triple.

    The renderer derives a shot's rendered length as ``(out - in) / speed``. If
    that is not ``seconds``, the shot occupies a different slice of the timeline
    than the one the solver allocated on the beat grid — and since the shots are
    concatenated, the difference PREFIX-SUMS into every later cut. Checked to
    1e-6, not to a frame: this is arithmetic the solver fully controls, and any
    slack here is exactly the room a rounding or a clamp needs to hide in.
    """
    assert shots, f"no shots to check {label}"
    for i, shot in enumerate(shots):
        seconds = _f(shot, "seconds")
        speed = _f(shot, "speed")
        rendered = (_f(shot, "out_seconds") - _f(shot, "in_seconds")) / speed
        assert abs(rendered - seconds) < 1e-6, (
            f"{label} shot {i} renders {rendered:.6f}s inside a {seconds:.6f}s beat window "
            f"(in={shot['in_seconds']!r} out={shot['out_seconds']!r} speed={speed!r}) — "
            f"every later cut is {abs(rendered - seconds):.3f}s off the music"
        )


#: A grid whose only offerable span is 3 beats: spans 1 and 2 end before 0.6s so
#: they fall under MIN_SHOT_SECONDS and are never offered, and the grid runs out
#: before span 4. That leaves the phrase set {2, 4, 8} with no candidate, so the
#: trim repair (b) — which only ever looks at phrase spans — cannot fire and a
#: too-long source has to fall all the way to the clamp of repair (d). Starved on
#: purpose: the clamp is the rung a healthy grid never reaches.
_STARVED_GRID = [0.15, 0.3, 0.45, 0.9]


def test_a_clamped_shot_is_re_trimmed_rather_than_re_lengthened() -> None:
    """Repair (d), the too-LONG direction, and the worst bug this module can have.

    A 9s source against a ~0.9s window wants 10x. Clamping the speed to 2.0x
    while keeping the 9s trim makes the compositor play 4.5s of reel inside a
    0.9s beat window — the shot silently steals 3.6s from the music and every
    cut after it lands that much late. The clamp must cost FOOTAGE, not TIME.
    """
    shots, notes, _, *_ = solve_cut_plan(_segments([9.0]), _STARVED_GRID)

    assert len(shots) == 1, notes
    shot = shots[0]
    assert shot["beat_locked"] is False
    assert _f(shot, "speed") == pytest.approx(SPEED_MAX, abs=_TRIM_SPEED_TOLERANCE)
    assert _f(shot, "speed") <= SPEED_MAX
    # The trim was pulled in to what 2.0x can actually cover in the window; the
    # 9s the segment offered is NOT what gets played.
    trimmed = _f(shot, "out_seconds") - _f(shot, "in_seconds")
    assert trimmed == pytest.approx(SPEED_MAX * _f(shot, "seconds"), abs=1e-3)
    assert trimmed < 9.0
    _assert_render_equation(shots, "clamped-too-long")


def test_a_clamped_too_short_shot_still_fills_its_beat_window() -> None:
    """Repair (d), the too-SHORT direction. The window must come out FULL: a
    shot that renders shorter than the slot it was given leaves a hole, and the
    music plays straight through it while every later cut lands early.

    The old form of this test asserted ``_f(shots[0], "speed") >= SPEED_MIN``.
    That bound could only be met by reading footage the segment does not
    contain, so it is replaced by the property it was a proxy for — the window
    is filled, exactly — plus the explicit no-overshoot test below. Nothing here
    is weakened: the render equation is still asserted to 1e-6 across the whole
    plan.

    The source was 0.2s and is now 0.4s, because dropping the editorial floor is
    not the same as dropping BOTH floors: at 0.2s this fixture emitted 0.2x, and
    ``COMPILER_SPEED_MIN`` is where the compiler 422s the entire shots list.
    A segment that cannot clear that floor has no legal placement at all and is
    now left unplaced (asserted below), so a clamped shot that DOES reach the
    reel is one that clears it — which is what makes "the window comes out full"
    a property of something the renderer will actually accept.
    """
    beats = _grid(jitter=0.01)
    segments = _segments([0.4, 2.0, 2.0])
    shots, *_ = solve_cut_plan(segments, beats)

    assert shots[0]["beat_locked"] is False
    assert COMPILER_SPEED_MIN <= _f(shots[0], "speed") <= SPEED_MAX
    # The window is filled to the microsecond: `(out - in) / speed` IS `seconds`.
    rendered = (_f(shots[0], "out_seconds") - _f(shots[0], "in_seconds")) / _f(shots[0], "speed")
    assert rendered == pytest.approx(_f(shots[0], "seconds"), abs=1e-6)
    # ...and it is filled out of the segment's own footage, not past it.
    assert _f(shots[0], "out_seconds") <= _f(segments[0], "out_seconds")
    _assert_render_equation(shots, "clamped-too-short")

    # The other side of the same boundary: a source too short to clear the
    # COMPILER's floor over any window this grid offers has no legal placement,
    # so it is left unplaced with a note naming it — never emitted at a speed
    # that would take the rest of the reel down with it.
    too_short = _segments([0.2, 2.0, 2.0])
    shots, notes, unplaced, *_ = solve_cut_plan(too_short, beats)
    assert not [shot for shot in shots if str(shot["id"]).startswith("seg-0")]
    assert unplaced == 1, notes
    assert any("seg-0" in note for note in notes), notes
    _assert_render_equation(shots, "too-short-for-the-compiler")


def test_a_clamped_shot_never_trims_past_the_approved_segment() -> None:
    """Repair (d) may not invent footage.

    The measured case: a 0.45s segment at 10.0-10.45 against a 120 BPM grid. Its
    natural speed over the shortest offerable window is well under ``SPEED_MIN``,
    so the clamp used to re-derive the source length as ``speed * window`` =
    0.501s and emit ``out_seconds`` 10.501 — 51ms past the out-point a reviewer
    approved, and past the end of the file when the segment sits at the tail of a
    source, where the renderer answers with a frozen or short last frame.
    Nothing downstream catches it: ``compile._source_window`` only checks that
    the window is ordered and non-negative.
    """
    segment: dict[str, object] = {
        "id": "seg-tail",
        "source_id": "src-1",
        "in_seconds": 10.0,
        "out_seconds": 10.45,
        "score": 0.6,
        "rationale": "the last usable half second",
    }

    shots, notes, _, *_ = solve_cut_plan([segment], _grid())

    assert len(shots) == 1, notes
    shot = shots[0]
    assert shot["beat_locked"] is False
    assert _f(shot, "out_seconds") <= 10.45, "the plan reads past the approved segment"
    assert _f(shot, "in_seconds") >= 10.0
    # The compromise is a slower-than-editorial warp, and it is honest arithmetic:
    # the shot still fills its beat window exactly.
    assert _f(shot, "speed") < SPEED_MIN
    _assert_render_equation(shots, "clamped-past-the-source")


@pytest.mark.parametrize("bpm", [40.0])
def test_no_tempo_makes_a_clamped_shot_read_past_its_source(bpm: float) -> None:
    """The same property swept over the tempo range: the worst overshoot was
    0.341s at 40 BPM, because a slower grid means a wider minimum window and so a
    longer invented tail."""
    beat = 60.0 / bpm
    beats = [round(i * beat, 6) for i in range(240)]
    segment: dict[str, object] = {
        "id": "seg-tail",
        "source_id": "src-1",
        "in_seconds": 10.0,
        "out_seconds": 10.45,
        "score": 0.6,
        "rationale": "",
    }

    shots, *_ = solve_cut_plan([segment], beats)

    assert shots
    assert _f(shots[0], "out_seconds") <= 10.45, f"{bpm} BPM plan reads past the source"


def test_a_short_opening_segment_is_dropped_rather_than_sent_below_the_compiler_floor() -> None:
    """The one clamp the render cannot survive.

    Shot 0's origin is quantised to 0.0 rather than to ``beats[0]``, so its
    narrowest offerable window is the whole musical intro — ``beats[1]``. A bed
    with a 2.0s intro and a 0.4s segment (``segment_select`` admits anything from
    0.4s up) therefore reaches repair (d) needing 0.2x, and repair (d) used to
    emit exactly that: the speed it clamped into the editorial band was thrown
    away by the ``min(speed * window, source_len)`` guard, and ``build``
    re-derived 0.2 from the trim that was actually sent.

    ``compile`` forwards ``speed`` verbatim and the compiler's contract floors it
    at ``COMPILER_SPEED_MIN``, validating the shots list as a UNIT — so one 0.2x
    shot 422s every other shot with it. The reel dies after all five human review
    gates approved it and the user is told "render failed" with nothing to watch.
    Losing one approved moment is the cheap half of that trade, so the segment
    goes unplaced with a note and the walk keeps its beat for the next one.
    """
    # A 2.0s musical intro, then 120 BPM. Real beds open on a bar of music.
    beats = [0.0, 2.0] + [round(2.0 + i * _IDEAL_BEAT, 6) for i in range(1, 200)]
    segments: list[dict[str, object]] = [
        {
            "id": "seg-short",
            "source_id": "src-1",
            "in_seconds": 10.0,
            "out_seconds": 10.4,
            "score": 0.9,
            "rationale": "the flash the intro cannot hold",
        },
        {
            "id": "seg-normal",
            "source_id": "src-1",
            "in_seconds": 20.0,
            "out_seconds": 22.0,
            "score": 0.6,
            "rationale": "a shot that fits the intro exactly",
        },
    ]

    shots, notes, unplaced, *_ = solve_cut_plan(segments, beats)

    for shot in shots:
        assert _f(shot, "speed") >= COMPILER_SPEED_MIN, (
            f"shot {shot['id']} would 422 the whole compile at {_f(shot, 'speed'):.3f}x"
        )
    short = [shot for shot in shots if str(shot["id"]).startswith("seg-short")]
    if not short:
        assert unplaced >= 1, "an approved segment vanished without being counted"
        assert any("seg-short" in note for note in notes), (
            "a dropped segment the reviewer approved must say so, and say why"
        )
    # The drop is LOCAL: the beat it declined is still there for the next segment.
    assert shots, notes
    assert str(shots[0]["id"]).startswith("seg-normal")
    assert _f(shots[0], "start_seconds") == pytest.approx(0.0, abs=1e-9)
    _assert_render_equation(shots, "short-opening-segment")


#: Grids that make the FIRST window hostile — the place a too-short segment has
#: no escape from, because shot 0's origin is quantised to 0.0 and so its
#: narrowest offerable window is the whole musical intro. A long intro (a real
#: bed opens on a bar or two of music) and a slow tempo (a wider beat) are the
#: two ways that window gets long enough to sink a short clip.
_HOSTILE_GRIDS: dict[str, list[float]] = {
    "120bpm": _grid(),
    "120bpm-jittered": _grid(jitter=0.017),
    "2s-intro": [0.0, 2.0] + [round(2.0 + i * _IDEAL_BEAT, 6) for i in range(1, 200)],
    "3s-intro-jittered": [0.0, 3.0]
    + [round(3.0 + i * _IDEAL_BEAT + 0.01 * math.sin(i * 1.7), 6) for i in range(1, 200)],
    "40bpm": [round(i * 1.5, 6) for i in range(200)],
    # 61/30 s is 61 FRAMES at the 30 fps default: an intro window that is exactly
    # on the frame grid and NOT on the millisecond grid (2.033333…s). It is the
    # shape that defeats a filter written against the unrounded source length,
    # because the millisecond the emitted trim loses to `floor` is worth ~1e-4 of
    # speed here — small, and on the wrong side of a HARD bound.
    "61-frame-intro-30fps": [0.0] + [round(61.0 / 30.0 + i, 6) for i in range(200)],
}


@pytest.mark.parametrize("grid_name", list(sorted(_HOSTILE_GRIDS))[:1])
# 0.508334 is the paired length: against the 61-frame intro above it passes an
# unrounded filter by 3 microseconds (0.508334/0.25 = 2.033336 >= 2.033333) and
# then emits floor_ms → 0.508/2.033333 = 0.24984x, i.e. under the compiler's
# floor, on a shots list that five human review gates already approved.
@pytest.mark.parametrize("length", [0.1])
@pytest.mark.parametrize("target", [None])
def test_no_input_makes_the_solver_emit_a_speed_the_compiler_rejects(
    grid_name: str, length: float, target: float | None
) -> None:
    """THE fence, and the assertion whose absence let this ship.

    ``speed`` is the one field the solver computes that another service
    VALIDATES: ``compile`` forwards it verbatim and the compiler's contract
    (``tests/contracts/compile_request.schema.json`` ``$defs/Shot.speed``) bounds
    it to ``[0.25, 4.0]``, checking the shots list as a UNIT. One out-of-band
    shot therefore 422s every other shot with it — the render dies AFTER all five
    human review gates approved it and the user sees "render failed" and no reel.

    So this is swept rather than sampled: every combination of a pathologically
    short segment (``segment_select._MIN_SEGMENT_SECONDS`` admits 0.4s, and
    a second take can leave less), a grid whose first window is long, and a
    length target that pulls re-takes off the queue. Whatever the solver decides
    — place it, clamp it, drop it — nothing it EMITS may be unrenderable.
    """
    beats = _HOSTILE_GRIDS[grid_name]
    segments = _segments([length, 2.0, length, 3.0, length])

    shots, notes, unplaced, *_ = solve_cut_plan(segments, beats, target_seconds=target)

    for shot in shots:
        speed = _f(shot, "speed")
        assert COMPILER_SPEED_MIN <= speed <= SPEED_MAX, (
            f"{grid_name}/{length}s/target={target}: shot {shot['id']} at {speed:.3f}x is "
            f"outside the compiler's contract — the WHOLE compile request 422s, "
            f"not just this shot"
        )
    # A segment that cannot be emitted must be ACCOUNTED for, not quietly gone:
    # `unplaced_segments` is what tells the reviewer their approved moment was
    # dropped, and a note is what tells them why.
    placed = {str(shot["id"]).split("-t")[0] for shot in shots}
    missing = [str(segment["id"]) for segment in segments if str(segment["id"]) not in placed]
    assert unplaced >= len(missing), f"{len(missing)} segments vanished, {unplaced} counted"
    for segment_id in missing:
        assert any(segment_id in note for note in notes), f"{segment_id} dropped silently"
    if shots:
        _assert_render_equation(shots, f"{grid_name}/{length}s")


def test_the_speed_filter_tests_the_millisecond_floored_trim_it_will_emit() -> None:
    """The candidate filter and ``build`` must round the SAME way.

    The filter admits a window when the footage can still cover it above the
    compiler's floor; ``build`` then emits an out-point floored to the
    millisecond. Testing the UNROUNDED source length therefore admits a window on
    up to a millisecond of footage the shot never gets — worth ~1e-4 of speed
    over a ~2s window, which is nothing except when the bound is HARD.

    The numbers are exact, not illustrative: a 61-frame (2.033333s) intro window
    at 30 fps against a 0.508334s source. 0.508334/0.25 = 2.033336 clears the
    window by three microseconds and is admitted; the emitted trim is
    floor_ms(10.508334) = 10.508, and 0.508/2.033333 = 0.24984x — rejected by
    ``$defs/Shot.speed.minimum``, which 422s the WHOLE shots list.
    """
    beats = [0.0] + [round(61.0 / 30.0 + i, 6) for i in range(40)]
    segments: list[dict[str, object]] = [
        {
            "id": "seg-0",
            "source_id": "src-1",
            "in_seconds": 10.0,
            "out_seconds": 10.508334,
            "seconds": 0.508334,
            "score": 0.6,
            "rationale": "the moment a reviewer approved",
        }
    ]

    shots, notes, unplaced, *_ = solve_cut_plan(segments, beats, frame_rate=30.0)

    for shot in shots:
        assert _f(shot, "speed") >= COMPILER_SPEED_MIN, (
            f"shot {shot['id']} emitted {_f(shot, 'speed'):.8f}x — under the compiler's "
            f"{COMPILER_SPEED_MIN}x floor, so the whole compile request 422s"
        )
    # Dropping it is the only legal answer here, and a drop must be accounted
    # for: a segment that silently vanishes is an approved moment the reviewer
    # never learns was cut.
    assert unplaced == 1
    assert any("seg-0" in note for note in notes), notes


def test_a_clamped_note_shows_the_speed_that_will_be_validated() -> None:
    """The note is the only place a human sees the number the compiler checks.

    Printed at two decimals, an emitted 0.250327x and an emitted 0.24984x are
    both the string "0.25x" — one is legal and one kills the render, and the
    reviewer approving the plan cannot tell them apart. The evidence has to carry
    more precision than the distance a violation can hide in.
    """
    beats = [0.0] + [round(61.0 / 30.0 + i, 6) for i in range(40)]
    segments: list[dict[str, object]] = [
        {
            "id": "seg-0",
            "source_id": "src-1",
            "in_seconds": 10.0,
            "out_seconds": 10.509,
            "seconds": 0.509,
            "score": 0.6,
            "rationale": "a moment",
        }
    ]

    shots, notes, _, *_ = solve_cut_plan(segments, beats, frame_rate=30.0)

    assert len(shots) == 1, notes
    speed = _f(shots[0], "speed")
    # Not the floor, and not two-decimals-from it: the fixture is chosen so the
    # rounded form would read as exactly the bound.
    assert speed != pytest.approx(COMPILER_SPEED_MIN, abs=1e-6)
    assert f"{speed:.2f}" == f"{COMPILER_SPEED_MIN:.2f}"

    clamped = [note for note in notes if "clamped" in note]
    assert len(clamped) == 1, notes
    printed = re.search(r"clamped ([0-9.]+)x", clamped[0])
    assert printed is not None, clamped[0]
    assert float(printed.group(1)) == pytest.approx(speed, abs=1e-4), (
        f"the note says {printed.group(1)}x for a shot that will be validated at "
        f"{speed:.8f}x — a reviewer reads a legal value on an illegal shot"
    )


def test_speed_is_derived_from_the_emitted_millisecond_trim() -> None:
    """``in``/``out`` are emitted at millisecond resolution, so a ``speed`` taken
    from the UNROUNDED length disagrees with the renderer's own
    ``(out - in) / speed`` by up to a millisecond per shot — in the only channel
    that sets shot length, and it prefix-sums.

    The lengths deliberately include a trimmed one (12.0 → repair (b), whose
    out-point is a beat time and therefore NOT already a whole millisecond) and a
    clamped one (0.2 → repair (d)): a segment whose own in/out are already
    millisecond values rounds to itself and would prove nothing.
    """
    beats = _grid(jitter=0.017)
    shots, *_ = solve_cut_plan(_segments([12.0, 2.0, 0.2, 3.3]), beats)

    for shot in shots:
        for key in ("in_seconds", "out_seconds"):
            scaled = _f(shot, key) * 1000.0
            assert scaled == pytest.approx(round(scaled), abs=1e-6), f"{key} is not whole ms"
        # Exact, not approximate: `speed` must BE the quotient of the emitted
        # numbers, so no rounding can live between the plan and the render.
        assert _f(shot, "speed") * _f(shot, "seconds") == pytest.approx(
            _f(shot, "out_seconds") - _f(shot, "in_seconds"), abs=1e-9
        )


@pytest.mark.parametrize(
    ("lengths", "beats", "label"),
    [([2.0] * 20, _grid(jitter=0.012), "the long jittery reel")],
)
def test_the_render_equation_holds_for_every_shot_of_every_plan(
    lengths: list[float], beats: list[float], label: str
) -> None:
    """The property, over every rung of the repair ladder and the degradation.

    ``speed == (out - in) / seconds`` is not one branch's concern: it is what
    makes the emitted triple mean the same thing to the solver and to the
    renderer, so it is asserted over whole plans rather than at the one call
    site a regression happened to be found in."""
    shots, *_ = solve_cut_plan(_segments(lengths), beats)

    _assert_render_equation(shots, label)


def test_section_change_prefers_a_downbeat() -> None:
    """A new source is a section change; the solver biases the cut onto a bar
    line when it can do so without leaving the invisible band."""
    beats = _grid(jitter=0.004)
    downbeats = _downbeats(beats)
    # Asserted on the beat INDEX, not on the emitted time: the emitted time is
    # snapped to a frame, so comparing it to a raw downbeat timestamp would be
    # testing the quantiser rather than the downbeat preference.
    downbeat_times = set(downbeats)
    downbeat_indices = {i for i, moment in enumerate(beats) if moment in downbeat_times}
    segments = _segments([2.0, 2.0], source_id="src-a") + _segments([2.0, 2.0], source_id="src-b")
    for i, segment in enumerate(segments):
        segment["id"] = f"seg-{i}"

    shots, *_ = solve_cut_plan(segments, beats, downbeats=downbeats)

    changes = [
        shot for previous, shot in pairwise(shots) if shot["source_id"] != previous["source_id"]
    ]
    assert changes
    assert all(_i(s, "beat_index") in downbeat_indices for s in changes)


def test_target_duration_stops_the_walk() -> None:
    beats = _grid(jitter=0.01)
    shots, *_ = solve_cut_plan(_segments([2.0] * 20), beats, target_seconds=8.0)

    total = sum(_f(s, "seconds") for s in shots)
    assert total >= 8.0
    assert total < 16.0
    assert len(shots) < 20


def _clumped_by_source(n_sources: int, per_source: int, length: float) -> list[dict[str, object]]:
    """The selection shape ``_merge_batch_draws`` actually produces: score-DESC,
    which clumps a strong source's moments together rather than round-robining
    them. It matters here because a round-robin list reaches every source in its
    first pass, so the walk can stop early and STILL leave nobody absent — which
    is exactly how the first version of this fix shipped a regression under a
    green suite."""
    return [
        {
            "id": f"seg-{source}-{k}",
            "source_id": f"src-{source}",
            "in_seconds": round(k * (length + 1.0), 3),
            "out_seconds": round(k * (length + 1.0) + length, 3),
            "seconds": length,
            "score": round(0.9 - 0.01 * (per_source * source + k), 3),
            "rationale": f"moment {source}-{k}",
        }
        for source in range(n_sources)
        for k in range(per_source)
    ]


def test_reaching_the_target_reports_absent_sources_without_degrading_the_run() -> None:
    """The regression guard for the FIX ITSELF, which is the sharper of the two edges.

    ``notes`` is a status channel: ``api/routes/runs/_shared.py:96`` sets
    ``degraded = completed and bool(output["notes"])``, so any note here flips the
    whole run. The first cut of this fix noted every absence and turned **181 of
    200** realistic plans (10 clips, 20 segments in score-DESC order, a 45s ask)
    from clean to degraded — nine healthy reels in ten wearing a damage badge,
    which is noise, not honesty.

    Reaching the target EXPLAINS the absence: ``segment_select`` over-selects
    (``_COVERAGE_FACTOR = 2.0``) so the solver can choose, and a reel that is the
    length the brief asked for is the system working. So the fields carry it and
    the notes channel stays empty.

    The fixture is the shape that can actually fail: clumped by source, 20
    segments over 10 sources, so the walk genuinely stops with sources unwalked.
    """
    beats = _grid_at(0.72, count=160)
    segments = _clumped_by_source(10, 2, 3.0)

    shots, notes, unplaced, unused, absent, *_ = solve_cut_plan(
        segments, beats, downbeats=beats[::4], target_seconds=45.0
    )

    # The case is real: the walk stopped early and two sources never reached the
    # reel. Without these the silence assertion below would be vacuous.
    assert sum(_f(s, "seconds") for s in shots) >= 45.0
    assert unused == 4
    assert absent == ("src-8", "src-9")
    assert unplaced == 0, "reaching the target is not the grid running out"
    # ...and the run is NOT badged degraded for it.
    assert notes == [], f"a reel that hit its target must stay clean; got {notes}"


def test_a_plan_that_fell_short_does_name_its_absent_sources() -> None:
    """The other edge: nothing explains this absence away, so it IS a degradation.

    The grid ran out at three shots against a 45s ask. Footage was approved, the
    reel had room for it, and it still did not make the cut — a reviewer can act
    on that (pick a longer bed, re-select), which is precisely what the
    ``degraded`` badge is for.
    """
    beats = _grid(count=12, jitter=0.01)  # ~5.5s of music against a 45s ask
    segments = _one_segment_per_source([2.0] * 5)

    shots, notes, unplaced, unused, absent, *_ = solve_cut_plan(
        segments, beats, target_seconds=45.0
    )

    assert sum(_f(s, "seconds") for s in shots) < 45.0
    assert unplaced >= 1
    assert unused == 0, "the walk never reached the target, so nothing was left BY the target"
    assert absent == ("src-3", "src-4")
    absent_notes = [note for note in notes if note.startswith("SOURCES ABSENT")]
    assert len(absent_notes) == 1, f"a short reel must say what it left out; got {notes}"
    assert "src-3, src-4" in absent_notes[0], "the reviewer needs the ids, not just a count"
    assert "45.0s" in absent_notes[0], "and the ask it fell short of"


def test_a_plan_that_used_every_source_says_nothing_at_all() -> None:
    """The other half of the fix, and the half that keeps it usable.

    ``notes`` badges a completed stage as a ``degraded`` run, so a note on a
    healthy plan would surface EVERY length-targeted reel as degraded and teach
    reviewers to skim the channel — which is how the absence above would have
    stayed invisible even after being reported. Same fixture as the test above
    with the two unwalked sources removed: the walk still stops at the target,
    and the plan is silent.
    """
    beats = _grid(count=60, jitter=0.01)
    segments = _one_segment_per_source([4.0] * 3)

    shots, notes, unplaced, unused, absent, *_ = solve_cut_plan(
        segments, beats, target_seconds=12.0
    )

    assert [s["source_id"] for s in shots] == ["src-0", "src-1", "src-2"]
    assert unplaced == 0
    assert unused == 0
    assert absent == ()
    assert notes == [], f"a plan that used every source must be silent; got {notes}"


def _one_segment_per_source(lengths: list[float]) -> list[dict]:
    """The live corpus shape: N sources, one long segment drawn from each — which
    is what ``segment_select`` actually returned on the run that came out short."""
    return [
        {
            "id": f"seg-{i}",
            "source_id": f"src-{i}",
            "in_seconds": 0.0,
            "out_seconds": round(length, 3),
            "seconds": round(length, 3),
            "score": 0.7,
            "rationale": f"moment {i}",
        }
        for i, length in enumerate(lengths)
    ]


def test_a_long_segment_gives_a_second_shot_rather_than_a_short_reel() -> None:
    """THE regression. Live run: 10 segments totalling 69.5s of approved footage,
    a 58s bed with 130 beats, ``--target-seconds 45`` — and a 33.7s reel with an
    11.3s shortfall. Neither the music nor the footage ran out; the SEGMENT LIST
    did, because the walk took at most one shot per segment and the beat windows
    were ~3.4s. The ask was reachable and was not reached.

    The footage each segment does not use (repair (b) trims a long source to its
    window) is exactly the material that closes the gap.
    """
    beats = _grid(count=120, jitter=0.012)  # ~60s of music, more than the ask
    segments = _one_segment_per_source([6.95] * 10)  # 69.5s of approved footage

    shots, notes, _, *_ = solve_cut_plan(
        segments, beats, downbeats=_downbeats(beats), target_seconds=45.0
    )

    total = sum(_f(s, "seconds") for s in shots)
    # Reached, within the one shot the stopping rule is allowed to overshoot by.
    assert total >= 45.0, f"still short at {total:.2f}s: {notes}"
    assert len(shots) > len(segments), "shot count is still bounded by segment count"
    # Not bought with degradation: every shot still lands on the grid.
    assert all(s["beat_locked"] is True for s in shots), notes
    assert not any("TRUNCATED" in note for note in notes)
    _assert_render_equation(shots, "second-take reel")


def test_every_shot_id_is_accepted_by_the_compilers_id_charset() -> None:
    """A re-take id must survive the compiler, not just be unique.

    ``video_compiler_service`` pins ``Shot.id`` to ``_SAFE_ID_PATTERN`` --
    ``^[A-Za-z0-9_.-]+$`` -- because it interpolates the id into scratch FILE PATHS
    and inline into ffmpeg FILTERGRAPHS. The first take suffix used ``#``, which is
    outside that charset: a live 10-clip run planned 15 beat-locked shots and then
    died on ``POST /api/compile -> 422``, after every earlier stage had succeeded.
    Uniqueness alone is not the contract; the charset is part of it.
    """
    safe = re.compile(r"^[A-Za-z0-9_.-]+$")
    beats = _grid(count=140, jitter=0.01)
    segments = _one_segment_per_source([6.95] * 10)

    shots, *_ = solve_cut_plan(segments, beats, downbeats=_downbeats(beats), target_seconds=45.0)

    retakes = [s for s in shots if str(s["id"]) not in {str(x["id"]) for x in segments}]
    assert retakes, "no re-take was produced, so the charset is untested"
    for shot in shots:
        shot_id = str(shot["id"])
        assert safe.match(shot_id), f"compiler would 422 on shot id {shot_id!r}"
        assert len(shot_id) <= 128, f"compiler caps the id at 128 chars: {shot_id!r}"


def test_a_second_take_shows_unused_footage_never_a_replay() -> None:
    """Filling the reel must not mean showing the same frames twice, and the two
    shots must not collide on the id downstream joins are keyed by."""
    beats = _grid(count=120, jitter=0.012)
    segments = _one_segment_per_source([6.95] * 10)

    shots, *_ = solve_cut_plan(segments, beats, target_seconds=45.0)

    ids = [str(s["id"]) for s in shots]
    # Binds the assertions below to the case they are about: some segment was
    # visited twice. Without it this test passes vacuously on a one-shot walk.
    assert len(shots) > len(segments)
    assert len(ids) == len(set(ids)), "two shots share an id; overlays would double up"
    windows: dict[str, list[tuple[float, float]]] = {}
    for shot in shots:
        windows.setdefault(str(shot["source_id"]), []).append(
            (_f(shot, "in_seconds"), _f(shot, "out_seconds"))
        )
    for source_id, spans in windows.items():
        spans.sort()
        for (_, first_out), (second_in, _) in pairwise(spans):
            assert second_in >= first_out - 1e-9, f"{source_id} replays footage"


def test_without_a_target_the_plan_is_still_one_shot_per_segment() -> None:
    """No requested length means no gap to close: the plan is the approved
    selection, once each, exactly as it was before the queue existed."""
    beats = _grid(count=200, jitter=0.012)
    segments = _one_segment_per_source([6.95] * 10)

    shots, *_ = solve_cut_plan(segments, beats)

    assert len(shots) == len(segments)
    assert [str(s["id"]) for s in shots] == [str(s["id"]) for s in segments]


def test_exhausted_grid_stops_and_notes() -> None:
    beats = _grid(count=12, jitter=0.01)  # ~5.5s of music
    shots, notes, _, *_ = solve_cut_plan(_segments([2.0] * 10), beats)

    assert 0 < len(shots) < 10
    assert any("beat grid exhausted" in note for note in notes)


def test_no_grid_degrades_to_even_spacing() -> None:
    segments = _segments([2.0, 3.5, 1.4])
    shots, notes, _, *_ = solve_cut_plan(segments, [])

    assert [_f(s, "speed") for s in shots] == [1.0, 1.0, 1.0]
    assert [_i(s, "beat_index") for s in shots] == [-1, -1, -1]
    assert [_f(s, "seconds") for s in shots] == pytest.approx([2.0, 3.5, 1.4])
    assert [_f(s, "start_seconds") for s in shots] == pytest.approx([0.0, 2.0, 5.5])
    assert all(s["beat_locked"] is False for s in shots)
    assert any("no beat grid" in note for note in notes)


def test_no_grid_still_honours_the_requested_length() -> None:
    """The measured regression: 8 x 5.0s of selection against a 20s request came
    out as a 40.0s reel — DOUBLE — because the degradation emitted every segment
    and ``segment_select`` over-selects ~2x by design. ``shortfall_seconds``
    stayed 0.0 throughout, because a shortfall gate cannot see an over-run.

    Losing the beat grid costs the reel its beat-sync. It does not license a reel
    of a length nobody asked for.
    """
    segments = _segments([5.0] * 8)

    shots, notes, _, *_ = solve_cut_plan(segments, [], target_seconds=20.0)

    total = sum(_f(s, "seconds") for s in shots)
    assert total == pytest.approx(20.0)
    assert total < 40.0
    assert len(shots) == 4
    # The stopping rule is the solved walk's: emit the shot that crosses the
    # target, then stop — never truncate a shot mid-way.
    assert [_f(s, "seconds") for s in shots] == pytest.approx([5.0] * 4)
    # ...and the degradation is still reported as a degradation.
    assert any("no beat grid" in note for note in notes)


def test_no_grid_without_a_target_still_emits_every_segment() -> None:
    """The other half: an author who asked for no length gets the whole approved
    selection, exactly as before. The stop is the TARGET's, not a new cap."""
    shots, *_ = solve_cut_plan(_segments([5.0] * 8), [])

    assert len(shots) == 8


def _ctx(
    *,
    beats: list[float],
    target: float | None = None,
    segments: list[dict] | None = None,
    sources: list[dict] | None = None,
) -> RunContext:
    config: dict[str, Any] = {}
    if target is not None:
        config["brief"] = {"target_duration_seconds": target}
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=config,
        prior_run_outputs={
            "segment_select": {"segments": segments or _segments([2.0, 4.0, 2.0])},
            "music_select": {
                "beats": beats,
                "downbeats": _downbeats(beats),
                "bpm": _BPM,
                "track_id": "trk-1",
                "track_url": "https://example.invalid/trk-1.mp3",
            },
            "footage_probe": {
                "sources": sources
                or [{"id": "src-1", "original_url": "https://example.invalid/a.mp4"}]
            },
        },
    )


async def test_stage_marks_the_plan_solved_and_joins_source_urls() -> None:
    out = await CutPlanStage().handle(CutPlanInput(), _ctx(beats=_grid(jitter=0.01)))

    assert out.solver_stubbed is False
    assert out.beat_grid_available is True
    assert out.notes == []
    assert len(out.shots) == 3
    assert all(s.video_url == "https://example.invalid/a.mp4" for s in out.shots)
    assert out.total_seconds == pytest.approx(sum(s.seconds for s in out.shots), abs=1e-3)
    assert out.bpm == _BPM
    assert out.music_track_id == "trk-1"


async def test_total_seconds_is_derived_from_the_boundaries_not_summed() -> None:
    """The reel's length is "where the last shot ends minus where the first one
    starts" — one subtraction of two absolute times. Summing the shot durations
    at the last line of the stage would re-introduce exactly the accumulation the
    solver above is built to avoid."""
    # An offset grid on purpose: it lands the reel's end on a frame count that is
    # NOT a whole number of milliseconds, which is what tells the derived value
    # apart from the `round(sum(...), 3)` it replaced.
    beats = [round(0.37 + i * _IDEAL_BEAT, 6) for i in range(240)]
    out = await CutPlanStage().handle(CutPlanInput(), _ctx(beats=beats))

    assert out.shots
    assert out.shots[0].start_seconds == 0.0
    last = out.shots[-1]
    expected = (last.start_seconds + last.seconds) - out.shots[0].start_seconds
    # Exact, not approx: this must BE the subtraction of two boundaries.
    assert out.total_seconds == expected
    # ...and not a value rounded to milliseconds on the way out.
    assert out.total_seconds != round(out.total_seconds, 3)


async def test_stage_does_not_raise_without_a_grid() -> None:
    out = await CutPlanStage().handle(CutPlanInput(), _ctx(beats=[]))

    assert out.beat_grid_available is False
    # The even-spacing fallback IS the stub: no beat-fit solving happened. The
    # flag used to be hard-coded False, which told the e2e proof harness that an
    # unsolved plan had come out of the solver.
    assert out.solver_stubbed is True
    assert [s.speed for s in out.shots] == [1.0, 1.0, 1.0]
    assert any("no beat grid" in note for note in out.notes)


async def test_dropped_segments_are_a_field_not_only_a_note() -> None:
    """A grid that runs out mid-selection must report the drop as a NUMBER.

    Same argument as ``shortfall_seconds``: footage a human approved and the reel
    does not contain is the degradation an operator must not be able to skim
    past, and a count interpolated into a note string is not something the e2e
    harness, the run-status roll-up, or any other consumer can gate on. The count
    was computed and then thrown away.

    Both halves are asserted because half of this is worthless: the field must be
    non-zero AND it must be the SAME number the note quotes. A field that reports
    a plausible-looking count of its own is a second source of truth, which is
    how a "structured" number ends up disagreeing with the prose beside it.
    """
    # ~5.5 s of music against 10 approved 2 s segments: the grid runs out long
    # before the selection does.
    ctx = _ctx(beats=_grid(count=12, jitter=0.01), segments=_segments([2.0] * 10))

    out = await CutPlanStage().handle(CutPlanInput(), ctx)

    exhausted = [n for n in out.notes if "beat grid exhausted" in n]
    assert exhausted, out.notes
    quoted = re.search(r"(\d+) selected segments were not placed", exhausted[0])
    assert quoted is not None, exhausted[0]

    assert out.unplaced_segments == 10 - len(out.shots)
    assert out.unplaced_segments > 0
    assert out.unplaced_segments == int(quoted.group(1)), (
        "the structured count and the note disagree — one of them is lying"
    )


async def test_a_complete_plan_reports_no_unplaced_segments() -> None:
    """The other direction: the field must not be a constant.

    A gate that reads non-zero on a healthy reel is one an operator learns to
    ignore, which costs exactly as much as no gate at all.
    """
    out = await CutPlanStage().handle(CutPlanInput(), _ctx(beats=_grid(jitter=0.01)))

    assert len(out.shots) == 3
    assert out.unplaced_segments == 0
    # Same "not a constant" argument for its sibling: every source resolved, so
    # nothing was dropped before the solve either.
    assert out.unfetchable_segments == 0


async def test_stage_honours_the_brief_target_duration() -> None:
    out = await CutPlanStage().handle(CutPlanInput(), _ctx(beats=_grid(jitter=0.01), target=3.0))

    assert len(out.shots) < 3
    assert out.total_seconds >= 3.0
    # Long enough: nothing was truncated, so the shortfall channel stays quiet.
    assert out.shortfall_seconds == 0.0
    assert not any("TRUNCATED" in note for note in out.notes)


async def test_solver_stubbed_is_false_when_the_solver_actually_ran() -> None:
    """The other half of the flag: it must not become a constant in the other
    direction either — a solved plan has to report itself solved."""
    out = await CutPlanStage().handle(CutPlanInput(), _ctx(beats=_grid(jitter=0.01)))

    assert out.solver_stubbed is False
    assert out.beat_grid_available is True


async def test_the_plan_is_quantised_to_the_rate_compile_will_render_at() -> None:
    """Plan and render must share ONE frame grid.

    The stage used to take ``solve_cut_plan``'s 23.976 default while compile's
    master is resolved from the same footage (24 fps here). Boundaries snapped to
    a grid the master cannot express are re-rounded by the renderer, one-sidedly,
    per segment — which is the prefix-sum drift this module exists to prevent,
    re-entering through the encoder.
    """
    sources = [{"id": "src-1", "original_url": "https://example.invalid/a.mp4", "fps": 23.976}]
    out = await CutPlanStage().handle(
        CutPlanInput(), _ctx(beats=_grid(jitter=0.01), sources=sources)
    )

    # 23.976 measured snaps to the legal broadcast rate 24 (see cadence.py).
    assert out.frame_rate == 24.0
    assert out.frame_rate != pytest.approx(DEFAULT_FRAME_RATE)
    for shot in out.shots:
        for boundary in (shot.start_seconds, shot.start_seconds + shot.seconds):
            frames = boundary * out.frame_rate
            assert frames == pytest.approx(round(frames), abs=1e-6), (
                f"boundary {boundary} is not a whole frame at {out.frame_rate} fps — "
                "the renderer will round it and the roundings prefix-sum"
            )


async def test_a_source_with_no_url_is_excluded_before_the_grid_is_allocated() -> None:
    """A shot compile will drop must never be given a beat window.

    ``compile`` skips a shot whose ``video_url`` is empty. If the solver has
    already spent a window on it, the render loses that window while the music
    plays straight through it, and every later cut is late by exactly its
    ``seconds``. So the segment leaves the walk BEFORE the grid is allocated.
    """
    segments = _segments([2.0, 2.0], source_id="src-1") + _segments([2.0], source_id="src-dead")
    for i, segment in enumerate(segments):
        segment["id"] = f"seg-{i}"
    beats = _grid(jitter=0.01)
    ctx = _ctx(beats=beats, segments=segments)

    out = await CutPlanStage().handle(CutPlanInput(), ctx)

    assert [s.id for s in out.shots] == ["seg-0", "seg-1"]
    assert all(s.video_url for s in out.shots)
    # Never silent: the excluded segment is named, with its source.
    assert any("seg-2" in note and "src-dead" in note for note in out.notes)
    # And never note-ONLY: the same loss as `unplaced_segments`, one step
    # earlier, so it is counted the same way. A note is prose — nothing gates on
    # it — and this drop is footage a human approved that the reel will not
    # contain.
    assert out.unfetchable_segments == 1
    # THE point: the grid was re-allocated to the shots that will actually
    # exist, so the plan is identical to one solved from the playable set alone —
    # no window is left standing empty in the middle of the reel.
    expected, *_ = solve_cut_plan(segments[:2], beats, downbeats=_downbeats(beats), frame_rate=30.0)
    assert [s.start_seconds for s in out.shots] == [_f(e, "start_seconds") for e in expected]
    assert [s.seconds for s in out.shots] == [_f(e, "seconds") for e in expected]


class _Resigner:
    """Storage double: re-mints any URL it recognises, like the real MinIO seam."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def resign_presigned_url(self, url: str) -> str | None:
        self.seen.append(url)
        return f"{url}?X-Amz-Signature=fresh" if "example.invalid" in url else None


async def test_source_urls_are_re_presigned_at_plan_time(monkeypatch: Any) -> None:
    """``footage_probe`` presigns with a 1h TTL, and cut_plan runs after TWO
    human-paced pause gates. An expired presigned URL is still a non-empty
    string, so it passes every ``if not video_url`` guard and dies deep inside
    the compositor's fetch under an unrelated message."""
    storage = _Resigner()
    monkeypatch.setattr(
        "src.services.stages.shot_presign._get_storage",
        lambda: _awaitable(storage),
    )

    out = await CutPlanStage().handle(CutPlanInput(), _ctx(beats=_grid(jitter=0.01)))

    assert storage.seen == ["https://example.invalid/a.mp4"]
    assert all(
        s.video_url == "https://example.invalid/a.mp4?X-Amz-Signature=fresh" for s in out.shots
    )


async def _awaitable(value: object) -> object:
    return value


async def test_an_empty_plan_fails_here_rather_than_four_stages_later() -> None:
    """A grid too short to hold one shot used to return ``shots=[]`` as a
    SUCCESS; the run then died in compile with a message about the render. The
    cause belongs in the precondition channel, named."""
    with pytest.raises(StagePreconditionError) as caught:
        await CutPlanStage().handle(CutPlanInput(), _ctx(beats=[0.1, 0.2]))

    message = str(caught.value)
    assert "beat grid" in message
    assert "empty" in message


async def test_an_all_unplaceable_plan_blames_the_footage_and_names_what_it_dropped() -> None:
    """The other way an empty plan happens, and the one that used to lie.

    Here the grid is perfect — 3s windows, far above ``MIN_SHOT_SECONDS`` — and
    the footage is 0.4s, which cannot fill one of them without falling under the
    compiler's speed floor. Blaming "the beat grid could not hold a single shot"
    sends the next person to fix a grid that was never broken.

    Worse, the solver's notes name WHICH approved moment was dropped and why, and
    the raise threw all of them away: the reviewer got a beat count and no
    diagnosis of a run that had already passed five human gates.
    """
    grid = [round(i * 3.0, 6) for i in range(40)]
    with pytest.raises(StagePreconditionError) as caught:
        await CutPlanStage().handle(
            CutPlanInput(), _ctx(beats=grid, segments=_segments([0.4, 0.4, 0.4]))
        )

    message = str(caught.value)
    # The cause, named as the footage and not as the grid.
    assert "none of the 3 selected segments could be placed" in message
    assert f"{COMPILER_SPEED_MIN:g}x floor" in message
    assert "could not hold a single shot" not in message
    # ...and the evidence rides along: which moments went, and why.
    for segment_id in ("seg-0", "seg-1", "seg-2"):
        assert segment_id in message, message
    assert "cannot fill the narrowest" in message
    # Bounded. A 200-segment selection must not paste 200 notes into an error.
    assert message.count("was not placed") <= 3


async def test_every_selected_segment_unfetchable_is_a_precondition_failure() -> None:
    sources = [{"id": "src-other", "original_url": "https://example.invalid/other.mp4"}]
    with pytest.raises(StagePreconditionError) as caught:
        await CutPlanStage().handle(CutPlanInput(), _ctx(beats=_grid(jitter=0.01), sources=sources))

    assert "fetchable source url" in str(caught.value)


async def test_a_bed_too_short_for_the_requested_length_reports_a_shortfall() -> None:
    """Real beds are ~58s while the catalog claims 120s, so "give me 90s" quietly
    became a 58s reel. The truncation is a NUMBER on the output, not only a note
    among the others: an operator must not be able to skim past it."""
    beats = _grid(count=40, jitter=0.01)  # ~19.5s of music
    segments = _segments([2.0] * 30)
    out = await CutPlanStage().handle(
        CutPlanInput(), _ctx(beats=beats, segments=segments, target=90.0)
    )

    assert out.target_seconds == 90.0
    assert out.shortfall_seconds > 60.0
    assert out.shortfall_seconds == pytest.approx(90.0 - out.total_seconds, abs=1e-3)
    assert any(note.startswith("TRUNCATED") for note in out.notes)


async def test_a_corpus_too_short_for_the_ask_still_reports_a_shortfall() -> None:
    """The other side of the second-take fix, at the stage seam: the walk may come
    back for a segment's UNUSED tail, and when the tails run out it stops. It must
    never buy the requested length by replaying footage — so an impossible ask
    (60s of reel from 6s of selection, against two full minutes of music) still
    comes out short and still says so, loudly and as a number.
    """
    out = await CutPlanStage().handle(
        CutPlanInput(),
        _ctx(beats=_grid(jitter=0.01), segments=_segments([2.0] * 3), target=60.0),
    )

    assert out.total_seconds < 10.0
    assert out.shortfall_seconds > 50.0
    assert out.shortfall_seconds == pytest.approx(60.0 - out.total_seconds, abs=1e-3)
    assert any(note.startswith("TRUNCATED") for note in out.notes)


async def test_the_no_grid_degradation_does_not_double_the_requested_reel() -> None:
    """The regression at the STAGE seam: an operator asking for 20s of reel out of
    a (deliberately over-selected) 40s of footage, on a run whose analyzer died."""
    out = await CutPlanStage().handle(
        CutPlanInput(), _ctx(beats=[], segments=_segments([5.0] * 8), target=20.0)
    )

    assert out.solver_stubbed is True
    assert out.total_seconds == pytest.approx(20.0)
    assert out.shortfall_seconds == 0.0
    assert out.overrun_seconds == 0.0


async def test_a_plan_that_runs_long_reports_an_overrun_number() -> None:
    """The mirror of ``shortfall_seconds``. Both walks stop by emitting the shot
    that CROSSES the target, so a plan can legitimately run up to one shot long —
    but "did this reel deliver the requested length" must be answerable in both
    directions, and the shortfall gate is one-sided.

    No note, deliberately: notes surface the run as ``degraded`` and a sub-one-shot
    over-run is the stopping rule working.
    """
    out = await CutPlanStage().handle(
        CutPlanInput(), _ctx(beats=[], segments=_segments([7.0] * 5), target=20.0)
    )

    assert out.total_seconds == pytest.approx(21.0)
    assert out.overrun_seconds == pytest.approx(1.0)
    assert out.shortfall_seconds == 0.0
    assert not any("TRUNCATED" in note for note in out.notes)
    assert not any("OVERRUN" in note for note in out.notes), (
        "a one-shot overshoot is the stopping rule working; badging it degraded "
        "would badge every length-targeted reel"
    )


async def test_a_runaway_overrun_reaches_the_reviewer_as_a_note() -> None:
    """The other half, and the one that was invisible. The stopping rule bounds
    the over-run by the FINAL SHOT, not by anything the author asked for — so one
    45s clip against a 20s ask ships a reel more than twice the requested length,
    entirely "within the rule". ``overrun_seconds`` carried that, and nothing
    reads ``overrun_seconds``: the reviewer saw a green gate with nothing on it.
    """
    out = await CutPlanStage().handle(
        CutPlanInput(), _ctx(beats=[], segments=_segments([45.0, 30.0]), target=20.0)
    )

    assert out.total_seconds == pytest.approx(45.0)
    assert out.overrun_seconds == pytest.approx(25.0)
    overruns = [note for note in out.notes if note.startswith("OVERRUN")]
    assert len(overruns) == 1, f"the over-run must be said out loud; got {out.notes}"
    assert "45.0s" in overruns[0], "the note names the length that was delivered"
    assert "20.0s" in overruns[0], "and the length that was asked for"


async def test_the_render_equation_survives_the_full_stage() -> None:
    """The stage re-reads its own solver output through pydantic; the contract
    has to hold on what actually LEAVES the stage, not only inside it."""
    out = await CutPlanStage().handle(CutPlanInput(), _ctx(beats=_grid(jitter=0.013)))

    assert out.shots
    for i, shot in enumerate(out.shots):
        rendered = (shot.out_seconds - shot.in_seconds) / shot.speed
        assert rendered == pytest.approx(shot.seconds, abs=1e-6), i


async def test_the_absent_sources_reach_the_stage_output_as_fields() -> None:
    """The reviewer reads the STAGE output, not the solver's tuple.

    ``cut_plan`` has ``default_pause_after=True``: this output is rendered at a
    human review gate, and a note is prose — no gate, no run-status roll-up and
    no e2e proof harness can read one. The same discipline ``shortfall_seconds``
    and ``unfetchable_segments`` already follow applies here: the ids and the
    counts leave the stage as FIELDS.

    And they leave it WITHOUT a note, which is the whole point of the fields
    existing: ``_shared.py:96`` turns any note into ``degraded``, so a lossless
    fact that a reviewer can look up has to travel in a channel that does not
    change the run's status to carry it.
    """
    out = await CutPlanStage().handle(
        CutPlanInput(),
        _ctx(
            beats=_grid(count=60, jitter=0.01),
            segments=_one_segment_per_source([4.0] * 5),
            sources=[
                {"id": f"src-{i}", "original_url": f"https://example.invalid/{i}.mp4"}
                for i in range(5)
            ],
            target=12.0,
        ),
    )

    assert [s.source_id for s in out.shots] == ["src-0", "src-1", "src-2"]
    # Honestly zero, and that is exactly why they could not carry this: neither
    # the grid nor the length fell short.
    assert out.unplaced_segments == 0
    assert out.shortfall_seconds == 0.0
    assert out.segments_unused_at_target == 2
    assert out.sources_absent == ["src-3", "src-4"]
    assert out.sources_absent_count == 2
    # The reel is the length it was asked for, so the run stays clean.
    assert out.notes == [], f"a plan that hit its target must not be degraded; got {out.notes}"


def test_the_no_grid_path_reports_its_absent_sources_as_fields_too() -> None:
    """The identical silent truncation, one path away.

    ``_even_spacing`` stops on the SAME rule as the solved walk (emit, then stop
    once the cursor has reached the target) and reported the same nothing: 0
    unplaced, an empty absent list. Half-fixing this is worse than not fixing it
    — a reviewer who has learned to read ``sources_absent`` on the solved path
    would read the empty list here as "every clip is in there", exactly when the
    grid is missing and the plan is ALREADY degraded to un-synced cuts.

    No SOURCES-ABSENT note fires here and that is structural, not an oversight:
    on this path absence REQUIRES the target break (a walk that fell short has
    emitted every segment it had), and a reached target explains the absence. The
    fields are the whole report. The pre-existing no-grid note is untouched.

    Same fixture as the solved-path test with the grid taken away, so the only
    variable is the path: five sources, one 4s segment each, a 12s ask.
    """
    segments = _one_segment_per_source([4.0] * 5)

    shots, notes, unplaced, unused, absent, *_ = solve_cut_plan(
        [*segments], [], target_seconds=12.0
    )

    # The degradation itself is untouched: same three shots, same natural
    # lengths, same 1.0x. Only the reporting changed.
    assert [s["source_id"] for s in shots] == ["src-0", "src-1", "src-2"]
    assert all(s["beat_locked"] is False for s in shots)
    # There is no grid here, so nothing could have been dropped for want of one.
    assert unplaced == 0
    assert unused == 2
    assert absent == ("src-3", "src-4")
    assert notes == [
        "no beat grid available; shots are spaced evenly at their natural length "
        "at 1.0x and are NOT cut to the music"
    ], f"the no-grid note is the only one this path may carry; got {notes}"


def test_the_live_run_shape_is_not_badged_degraded() -> None:
    """The production run this fix did NOT fire on, pinned.

    Run ``01a01f19-838f-755f-b3a9-9e84723cb23d``: 10 uploaded sources, 18
    selected segments (65.4s of approved footage), 18 shots, 47.08s planned
    against a 45s target. Every segment was placed, every source reached the
    reel, and ``unplaced_segments: 0`` was TRUE — this is a LATENT defect fix,
    not a live-firing one, and that distinction only survives if something
    encodes it.

    The fixture keeps that run's numbers (10 sources, 18 segments, 65.4s
    selected, a 45s ask, a reel just past it) but orders them CLUMPED BY SOURCE
    — the ordering ``_merge_batch_draws`` can equally produce — because the
    run's own near-round-robin order cannot fail this test: the walk reaches
    every source in its first pass, so ``sources_absent`` is empty however the
    note is gated, and a pin that cannot fail on the case it pins is worse than
    no pin at all. Clumped, the same corpus stops early and leaves four sources
    unwalked, so the silence below is load-bearing: it holds ONLY because the
    reel reached its 45s ask.

    What it defends: someone later "hardening" the silence into a warning would
    badge a run of exactly this shape ``degraded`` (``_shared.py:96``) and hand
    the reviewer a false alarm on a plan that delivered the length it was asked
    for — which measured 181 false alarms in 200 plans.
    """
    # ~83 BPM. The run's bed tracked slower than the 120 the other fixtures use,
    # and the phrase window is what decides how many of the 18 segments the walk
    # reaches before the ask is met.
    beats = _grid_at(0.65, count=140)
    segments: list[dict[str, object]] = [
        {
            "id": f"seg-{source}-{k}",
            "source_id": f"src-{source}",
            "in_seconds": round(k * 5.0, 3),
            "out_seconds": round(k * 5.0 + 3.633, 3),
            "seconds": 3.633,
            "score": round(0.9 - 0.01 * source, 3),
            "rationale": f"moment {source}-{k}",
        }
        for source in range(10)
        # 8 sources gave two moments and 2 gave one: 18 segments, 65.4s.
        for k in range(2 if source < 8 else 1)
    ]
    assert len(segments) == 18
    assert sum(_f(seg, "seconds") for seg in segments) == pytest.approx(65.4, abs=0.05)

    shots, notes, unplaced, unused, absent, *_ = solve_cut_plan(
        segments, beats, downbeats=beats[::4], target_seconds=45.0
    )

    total = sum(_f(s, "seconds") for s in shots)
    # The shape that makes the silence non-vacuous: the ask was MET, and the walk
    # did stop with sources unwalked. Remove the gate and this run is degraded.
    assert total >= 45.0
    assert unplaced == 0
    assert unused > 0
    assert absent, "the fixture must actually leave sources out, or it pins nothing"
    assert notes == [], f"a run that delivered its 45s ask must stay clean; got {notes}"


async def test_a_bed_too_short_to_hold_a_shot_blames_the_music_not_the_footage() -> None:
    """The cause that could NEVER be reported, and it is the music bed.

    ``_empty_plan_error`` used to tell its two causes apart on
    ``unplaced_segments`` alone. But the solver's walk guard is
    ``cursor < len(queue)``, so on the grid-exhausted break there is always at
    least one pending take left to count and ``unplaced_segments`` is always
    >= 1 — the "beat grid could not hold a shot" branch was structurally
    unreachable. Every short/degenerate bed (``music_select`` declares
    ``beats: list[float] = []`` with no count or spacing validation, so this is
    a live input, not a fixture) was reported to the reviewer as a FOOTAGE
    problem, and the ``missing=["music_select.beats"]`` signal that gates and
    tooling read never fired once.

    ``missing`` is the assertion, not the prose: it is the machine-readable half
    and it is the half that was wrong.
    """
    # 5s of perfectly good footage against a bed with ONE 0.3s window — well
    # under MIN_SHOT_SECONDS, so nothing can be placed at beat 0.
    with pytest.raises(StagePreconditionError) as caught:
        await CutPlanStage().handle(
            CutPlanInput(), _ctx(beats=[0.0, 0.3], segments=_segments([5.0]))
        )

    assert caught.value.missing == ["music_select.beats"], (
        "a short MUSIC BED was reported as a footage/speed-floor problem"
    )
    message = str(caught.value)
    assert "could not hold a single shot" in message
    assert "speed under the compiler" not in message


def test_the_solver_flags_grid_exhaustion_as_a_field_not_only_as_prose() -> None:
    """``grid_exhausted`` is what makes the branch above decidable.

    It cannot be derived downstream from ``unplaced_segments`` (always >= 1 here)
    nor from the notes (prose is not a contract), so the walk has to carry it.
    """
    solution = solve_cut_plan(_segments([5.0, 5.0]), [0.0, 0.3])

    assert solution.shots == []
    assert solution.grid_exhausted is True
    assert solution.unplaced_segments >= 1, (
        "the premise: the count is non-zero on this path, so it cannot tell the causes apart"
    )
