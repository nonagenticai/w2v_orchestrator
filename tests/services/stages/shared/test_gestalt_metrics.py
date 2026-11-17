"""Pure gestalt-metric functions for the A11 advisory ship gate.

No I/O, no mocks — every function operates on plain vectors, so the tests feed
synthetic fixtures directly. Covers the div-by-zero guards (1-shot / empty inputs)
and the canonical -14.0 LUFS master passing the loudness gate.
"""

from __future__ import annotations

from typing import cast

import pytest

from src.services.stages.shared.gestalt_metrics import (
    MOMENTUM_INSUFFICIENT_EVIDENCE,
    MOMENTUM_PASS,
    MOMENTUM_UNCONFIGURED,
    assembled_momentum_curve,
    build_beat_intervals,
    cohesion_score,
    continuity_deltas,
    continuity_stats,
    final_mux_loudness,
    gestalt_summary,
    loudness_window,
    momentum_gap,
    momentum_verdict,
    sample_frames_at_cuts,
    tonal_variance,
)
from src.services.stages.shared.retention import coerce_beat_seconds, predicted_watch_curve


# ── sample_frames_at_cuts: join cuts to spine ids, zip to shorter ────────────
def test_sample_frames_zips_to_the_shorter() -> None:
    # 3 spine ids but only 2 cuts ⇒ zip to the shorter (2 pairs).
    assert sample_frames_at_cuts([1.0, 2.0], 10.0, ["clip-1", "clip-2", "clip-3"]) == [
        ("clip-1", 1.0),
        ("clip-2", 2.0),
    ]


def test_sample_frames_drops_cut_past_duration() -> None:
    assert sample_frames_at_cuts([1.0, 99.0], 10.0, ["clip-1", "clip-2"]) == [("clip-1", 1.0)]


# ── build_beat_intervals + time-containment cut→beat join ────────────────────
def test_containment_pairs_by_time_on_cardinality_mismatch() -> None:
    # 3 beats, 10s each ⇒ intervals [(clip-1,0,10),(clip-2,10,20),(clip-3,20,30)].
    lines = [
        {"id": "clip-1", "seconds": 10.0},
        {"id": "clip-2", "seconds": 10.0},
        {"id": "clip-3", "seconds": 10.0},
    ]
    intervals = build_beat_intervals(lines)
    assert intervals == [("clip-1", 0.0, 10.0), ("clip-2", 10.0, 20.0), ("clip-3", 20.0, 30.0)]
    # 5 cuts vs 3 ids: containment maps each cut to its CONTAINING beat (no positional
    # mis-pairing — the positional zip would only emit 3 pairs and mis-align them).
    assert sample_frames_at_cuts(
        [5.0, 12.0, 15.0, 22.0, 28.0],
        30.0,
        ["clip-1", "clip-2", "clip-3"],
        beat_intervals=intervals,
    ) == [
        ("clip-1", 5.0),
        ("clip-2", 12.0),
        ("clip-2", 15.0),
        ("clip-3", 22.0),
        ("clip-3", 28.0),
    ]


def test_boundary_tiebreak_assigns_to_later_beat() -> None:
    # Half-open [start, end): a cut exactly on the 10.0 boundary joins clip-2, not clip-1.
    intervals = [("clip-1", 0.0, 10.0), ("clip-2", 10.0, 20.0)]
    assert sample_frames_at_cuts([10.0], 20.0, ["clip-1", "clip-2"], beat_intervals=intervals) == [
        ("clip-2", 10.0)
    ]


def test_beat_intervals_none_falls_back_to_positional() -> None:
    # beat_intervals=None ⇒ byte-identical to the legacy positional zip (no kwarg).
    boundaries = [1.0, 2.0]
    spine = ["clip-1", "clip-2", "clip-3"]
    assert sample_frames_at_cuts(boundaries, 10.0, spine, beat_intervals=None) == (
        sample_frames_at_cuts(boundaries, 10.0, spine)
    )


def test_build_beat_intervals_all_zero_seconds_returns_none() -> None:
    # Every beat has zero width ⇒ whole film has zero width ⇒ None (caller falls back).
    lines = [
        {"id": "clip-1", "seconds": 0.0},
        {"id": "clip-2", "seconds": 0.0},
        {"id": "clip-3", "seconds": 0.0},
    ]
    assert build_beat_intervals(lines) is None


def test_build_beat_intervals_skips_idless_lines() -> None:
    # A line missing `id` is skipped; the remaining ids keep their cumulative offsets.
    lines = [
        {"id": "clip-1", "seconds": 10.0},
        {"seconds": 5.0},  # no id ⇒ skipped, contributes no interval
        {"id": "clip-3", "seconds": 10.0},
    ]
    assert build_beat_intervals(lines) == [("clip-1", 0.0, 10.0), ("clip-3", 10.0, 20.0)]


def test_cut_past_last_end_clamps_to_last_beat() -> None:
    # A cut at/after the last interval's end clamps to the last beat — never dropped by
    # the containment path (29 is inside the film; 30 lands exactly on the last end).
    intervals = [("clip-1", 0.0, 10.0), ("clip-2", 10.0, 20.0), ("clip-3", 20.0, 30.0)]
    assert sample_frames_at_cuts(
        [29.0, 30.0], 30.0, ["clip-1", "clip-2", "clip-3"], beat_intervals=intervals
    ) == [("clip-3", 29.0), ("clip-3", 30.0)]


# ── continuity_deltas: adjacent L2, div-by-zero guard ────────────────────────
def test_continuity_deltas_flags_exactly_the_drift_pair() -> None:
    frames = [
        {"shot_id": "clip-1", "features": [0.0, 0.0, 0.0]},
        {"shot_id": "clip-2", "features": [10.0, 0.0, 0.0]},  # big jump
    ]
    breaks = continuity_deltas(frames, ceiling=1.0)
    assert len(breaks) == 1
    assert breaks[0]["from_id"] == "clip-1"
    assert breaks[0]["to_id"] == "clip-2"


def test_continuity_deltas_single_shot_no_break_no_error() -> None:
    # One shot ⇒ no adjacent pair ⇒ no delta, no divide-by-zero.
    assert continuity_deltas([{"shot_id": "clip-1", "features": [1.0]}], ceiling=0.1) == []
    assert continuity_deltas([], ceiling=0.1) == []


def test_continuity_deltas_below_ceiling_is_not_a_break() -> None:
    frames = [
        {"shot_id": "clip-1", "features": [1.0, 1.0]},
        {"shot_id": "clip-2", "features": [1.1, 1.0]},
    ]
    assert continuity_deltas(frames, ceiling=0.5) == []


# ── cohesion_score: [0,1], 1.0 for identical / degenerate ────────────────────
def test_cohesion_identical_vectors_is_unobservable_not_perfect() -> None:
    # ZERO DISPERSION is not a perfect film — it is the absence of a measurement. This is
    # the LIVE shape: the analyzer emits no per-frame brightness/sharpness/colour, so five
    # of six feature components are constant by construction and a static capture zeroes
    # the sixth. Returning 1.0 here fed a `cohesion_floor` (derived as a p05 of a FLOOR
    # family) of 1.0 into the nightly corpus — a gate no real film can clear.
    assert cohesion_score([{"features": [1.0, 1.0]}, {"features": [1.0, 1.0]}]) is None
    # Control: any real dispersion still scores, and scores BELOW 1.0.
    scored = cohesion_score([{"features": [1.0, 1.0]}, {"features": [1.1, 1.05]}])
    assert scored is not None and 0.0 < scored < 1.0


def test_cohesion_scattered_is_below_one() -> None:
    scattered = cohesion_score([{"features": [0.0, 0.0]}, {"features": [5.0, 5.0]}])
    assert scattered is not None  # genuinely dispersed ⇒ a real measurement
    assert scattered < 1.0


def test_cohesion_single_vector_is_unobservable() -> None:
    # HONESTY: one vector has nothing to disperse, so cohesion is None — NOT the
    # degenerate-perfect 1.0 the whole live corpus persisted (and was then compared
    # against the cohesion floor as if it were a measurement).
    assert cohesion_score([{"features": [1.0]}]) is None
    assert cohesion_score([]) is None
    assert cohesion_score([{"features": []}, {"features": []}]) is None


# ── tonal_variance: lexical proxy + worst adjacent jump ──────────────────────
def test_tonal_variance_flags_the_register_jump() -> None:
    fa = [
        {"shot_id": "clip-1", "description": "welcome friendly enjoy happy"},  # warmth
        {"shot_id": "clip-2", "description": "fast now launch speed action"},  # energy
    ]
    tv = tonal_variance(fa)
    variance = tv["variance"]
    assert isinstance(variance, float)
    assert variance > 0.0
    worst = tv["worst_cut"]
    assert isinstance(worst, dict)
    worst_map = cast("dict[str, object]", worst)
    assert worst_map["from_id"] == "clip-1"
    assert worst_map["to_id"] == "clip-2"


def test_tonal_variance_single_shot_is_unobservable() -> None:
    # HONESTY: with one shot there is no drift to measure. `variance: None`, never the
    # flattering 0.0 ("perfectly consistent tone") the live corpus persisted.
    assert tonal_variance([{"shot_id": "clip-1", "description": "hello"}]) == {
        "variance": None,
        "worst_cut": None,
    }
    assert tonal_variance([]) == {"variance": None, "worst_cut": None}


# ── final_mux_loudness: pass-through / normalize, never re-measures ───────────
def test_final_mux_loudness_passes_through() -> None:
    out = final_mux_loudness(
        {"integrated_lufs": -14.0, "true_peak_dbtp": -1.0, "lra": 7.0, "silence_ratio": 0.1}
    )
    # `lra` / `silence_ratio` are DROPPED (A/V wave 7): no predicate in this module or
    # in `final_critique` ever read them, and neither has a ceiling that is not a guess.
    # The compiler still measures and persists both at the source, so nothing is lost.
    assert out == {"integrated_lufs": -14.0, "true_peak_dbtp": -1.0}


def test_final_mux_loudness_none_and_missing_keys() -> None:
    assert final_mux_loudness(None) is None
    empty = final_mux_loudness({})
    assert empty is not None
    assert empty["integrated_lufs"] is None


# ── loudness window default ──────────────────────────────────────────────────
def test_loudness_window_default() -> None:
    # The default window is DERIVED (target ± tolerance), not restated — see
    # `shared/loudness_contract.py` and `test_loudness_contract.py`. The comment here
    # used to claim the compiler's DEFAULT_MASTER_LUFS was -12.5 and that -14 was
    # "stale": both were false. The compiler ships -14.0 (PR #217, 2026-09-12) and
    # -12.5 is the value that PR abandoned for causing audible gain riding.
    from src.services.stages.shared.loudness_contract import ship_loudness_window

    assert loudness_window({}) == ship_loudness_window()
    assert loudness_window({"loudness_lufs": [-16.0, -12.0]}) == (-16.0, -12.0)


# ── gestalt_summary: guards + the -14.0 master MUST pass ─────────────────────
def test_gestalt_summary_fewer_than_two_frames_is_insufficient_evidence() -> None:
    # THE core honesty inversion. Fewer than two frames ⇒ FOUR of the five gate terms are
    # unobservable, so the gate makes NO claim: every unobservable term is null and
    # ship_ready is False. It used to compute `loud_ok and momentum_ok` — both VACUOUSLY
    # true — which made a blank/frozen film the most confident "ship-ready" verdict the
    # gate could emit, with zero WM calls and a `ship-ready` badge.
    g = gestalt_summary(
        frames=[], frame_analyses=[], loudness={"integrated_lufs": -14.0}, ceilings={}
    )
    assert g["insufficient_evidence"] is True
    assert g["ship_ready"] is False  # in-window -14 LUFS alone is NOT evidence of a film
    assert g["continuity_breaks"] == []  # no shot-to-shot delta exists
    assert g["continuity_stats"] == {"n": 0}
    # No flattering constants anywhere.
    assert g["cohesion"] is None
    assert g["momentum_gap"] is None
    assert cast("dict[str, object]", g["tonal"])["variance"] is None


def test_gestalt_summary_out_of_window_loudness_is_not_ship_ready() -> None:
    g = gestalt_summary(
        frames=[], frame_analyses=[], loudness={"integrated_lufs": -6.0}, ceilings={}
    )
    assert g["ship_ready"] is False


def test_gestalt_summary_continuity_break_drops_ship_ready() -> None:
    frames = [
        {"shot_id": "clip-1", "features": [0.0]},
        {"shot_id": "clip-2", "features": [9.0]},
    ]
    g = gestalt_summary(
        frames=frames,
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={"continuity_l2": 1.0},
    )
    assert g["continuity_breaks"]
    assert g["ship_ready"] is False


def test_gestalt_summary_explicit_zero_continuity_ceiling_is_honored() -> None:
    # Fix #1 revert-verify: an author-set continuity_l2=0.0 must be honored, NOT promoted
    # to the 0.5 default by an `or`-falsy fallback. A tiny 0.1 L2 delta exceeds a 0.0
    # ceiling ⇒ a continuity break ⇒ ship_ready False. Cohesion (~0.95) clears the 0.5
    # floor, so only the continuity ceiling decides. Under the OLD `or` code, 0.0→0.5
    # would suppress the break and ship_ready would be True.
    frames = [
        {"shot_id": "clip-1", "features": [0.0]},
        {"shot_id": "clip-2", "features": [0.1]},  # L2 delta 0.1 > ceiling 0.0
    ]
    g = gestalt_summary(
        frames=frames,
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={"continuity_l2": 0.0},
    )
    assert g["continuity_breaks"]  # 0.1 > 0.0 ⇒ break (would be suppressed at 0.5)
    assert g["ship_ready"] is False


def test_gestalt_summary_coherent_film_with_good_loudness_ships() -> None:
    frames = [
        {"shot_id": "clip-1", "features": [1.0, 1.0]},
        # Small but REAL dispersion (L2 0.112, well under the 0.5 ceiling): a coherent film
        # the gate can actually measure. Identical vectors would be zero dispersion ⇒
        # insufficient_evidence, which is a different test.
        {"shot_id": "clip-2", "features": [1.1, 1.05]},
    ]
    g = gestalt_summary(
        frames=frames, frame_analyses=[], loudness={"integrated_lufs": -14.0}, ceilings={}
    )
    assert g["ship_ready"] is True
    # A MEASURED cohesion, strictly below the degenerate-perfect 1.0.
    assert g["cohesion"] == round(cast("float", cohesion_score(frames)), 4)
    assert cast("float", g["cohesion"]) < 1.0
    assert g["insufficient_evidence"] is False


# ── assembled_momentum_curve: retention-shaped ACTUAL momentum per beat ───────
def test_assembled_momentum_curve_low_motion_sags() -> None:
    # Both beats at offset 0 isolate the motion modulation: full motion → 0.7+0.3 = 1.0,
    # zero motion → 0.7 (the low-motion beat delivers less momentum).
    assert assembled_momentum_curve([0.0, 0.0], [1.0, 0.0], total_duration=10.0) == [1.0, 0.7]


def test_assembled_momentum_curve_clamps_motion_and_aligns_length() -> None:
    # motion clamped into [0, 1] (5.0→1.0, -1.0→0.0); the 3rd cut with no motion
    # partner is dropped by the min(len) alignment.
    assert assembled_momentum_curve([0.0, 0.0, 0.0], [5.0, -1.0], total_duration=10.0) == [1.0, 0.7]


def test_assembled_momentum_curve_empty_is_empty() -> None:
    assert assembled_momentum_curve([], [], total_duration=10.0) == []


# ── momentum_gap: spine-id-aligned + scale-normalized worst under-delivery ────
def test_momentum_gap_flat_healthy_normalizes_to_zero() -> None:
    # A uniformly-scaled assembled curve (each x0.9) has the SAME normalized shape as
    # predicted ⇒ 0.0 gap. Under the OLD positional raw-diff this returned ~0.1 — this
    # is the revert-verify for the motion-baseline bias fix.
    predicted = [("c1", 1.0), ("c2", 0.8), ("c3", 0.6)]
    assembled = [("c1", 0.9), ("c2", 0.72), ("c3", 0.54)]
    assert momentum_gap(assembled, predicted) == 0.0


def test_momentum_gap_positive_on_mid_run_sag() -> None:
    # Same predicted, but assembled dips at c2. Normalized: predicted ratios [1, 0.8, 0.6],
    # assembled ratios [1, 0.4, 0.6]; worst deficit = 0.8 - 0.4 = 0.4.
    predicted = [("c1", 1.0), ("c2", 0.8), ("c3", 0.6)]
    assembled = [("c1", 0.9), ("c2", 0.36), ("c3", 0.54)]
    gap = momentum_gap(assembled, predicted)
    assert gap is not None  # two common beats + a positive anchor ⇒ measurable
    assert gap > 0.0
    assert gap == pytest.approx(0.4)


def test_momentum_gap_joins_by_spine_id_not_position() -> None:
    # assembled lists only c1 + c3, REORDERED (c3 first) and uniformly scaled (x0.5).
    # Anchored on the first common beat (assembled[0]=c3), c3 must match pred[c3]=0.6
    # (NOT positionally to pred[c2]=0.8). Normalized shapes coincide ⇒ 0.0, proving the
    # id-join. A positional diff would mis-pair c3↔c2 and yield a non-zero gap.
    predicted = [("c1", 1.0), ("c2", 0.8), ("c3", 0.6), ("c4", 0.4)]
    assembled = [("c3", 0.3), ("c1", 0.5)]  # 0.6·0.5, 1.0·0.5
    assert momentum_gap(assembled, predicted) == 0.0


def test_momentum_gap_directional_over_delivery_is_zero() -> None:
    # Assembled normalized-above predicted everywhere ⇒ no under-delivery ⇒ 0.0.
    predicted = [("c1", 1.0), ("c2", 1.0)]
    assembled = [("c1", 0.5), ("c2", 1.0)]  # ratios [1, 2] vs predicted [1, 1]
    assert momentum_gap(assembled, predicted) == 0.0


def test_momentum_gap_unobservable_when_fewer_than_two_common_beats() -> None:
    # HONESTY: <2 aligned beats ⇒ None, never 0.0. 0.0 is the FLATTERING value (a
    # perfectly-delivered film) and is indistinguishable from a genuinely measured pass;
    # persisting it made the calibration corpus a corpus of silent zeros.
    assert momentum_gap([("zzz", 1.0), ("c1", 0.9)], [("c1", 1.0)]) is None
    assert momentum_gap([], [("c1", 1.0), ("c2", 1.0)]) is None


def test_momentum_gap_unobservable_on_nonpositive_anchor() -> None:
    # First common beat anchor is 0.0 ⇒ the normalization is undefined ⇒ None (never
    # divides by zero, and never fabricates a 0.0 "no gap" reading).
    predicted = [("c1", 1.0), ("c2", 0.5)]
    assert momentum_gap([("c1", 0.0), ("c2", 0.5)], predicted) is None


def test_momentum_gap_curve_pairs_end_to_end() -> None:
    # Realistic wiring: assembled_momentum_curve values zipped to spine ids, diffed
    # against a flat predicted curve. The zero-motion middle beat sags below the
    # full-motion anchor, so the normalized gap is positive.
    spine = ["c1", "c2", "c3"]
    curve = assembled_momentum_curve([0.0, 0.0, 0.0], [1.0, 0.0, 1.0], total_duration=10.0)
    assembled = list(zip(spine, curve, strict=True))
    predicted = [("c1", 1.0), ("c2", 1.0), ("c3", 1.0)]
    # anchor a0=1.0, p0=1.0; middle beat 0.7 ⇒ deficit 1.0 - 0.7 = 0.3.
    assert momentum_gap(assembled, predicted) == pytest.approx(0.3)


# ── gestalt_summary: momentum_gap is ADVISORY EVIDENCE, never a ship_ready gate ─
def test_gestalt_summary_default_momentum_gap_key_present_as_null() -> None:
    # An UNMEASURED momentum signal rides in as an explicit null, not a default 0.0.
    g = gestalt_summary(
        frames=[], frame_analyses=[], loudness={"integrated_lufs": -14.0}, ceilings={}
    )
    assert "momentum_gap" in g
    assert g["momentum_gap"] is None


def test_gestalt_summary_high_momentum_gap_does_not_change_ship_ready() -> None:
    frames = [
        {"shot_id": "clip-1", "features": [1.0, 1.0]},
        # Small but REAL dispersion (L2 0.112, well under the 0.5 ceiling): a coherent film
        # the gate can actually measure. Identical vectors would be zero dispersion ⇒
        # insufficient_evidence, which is a different test.
        {"shot_id": "clip-2", "features": [1.1, 1.05]},
    ]
    # A coherent, in-window, no-break film stays ship-ready even with a LARGE momentum
    # gap — momentum is advisory evidence only, it never gates (normal >=2-frame branch).
    g = gestalt_summary(
        frames=frames,
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={},
        momentum_gap=0.9,
    )
    assert g["momentum_gap"] == 0.9  # surfaced as evidence
    assert g["ship_ready"] is True  # but NOT gated on


def test_gestalt_summary_high_momentum_gap_rides_through_with_fewer_than_two_frames() -> None:
    # <2 frames: the measured gap still rides in as evidence, but ship_ready is False on
    # insufficient evidence regardless of it (it is not the gap that decides).
    g = gestalt_summary(
        frames=[],
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={},
        momentum_gap=0.9,
    )
    assert g["momentum_gap"] == 0.9
    assert g["insufficient_evidence"] is True
    assert g["ship_ready"] is False


def test_gestalt_summary_ship_ready_byte_identical_regardless_of_momentum_gap() -> None:
    # Byte-identity guard AND gate-detector: on an otherwise ship-ready film (coherent,
    # in-window, no breaks), ship_ready + every non-momentum key are identical whether
    # the gap is 0.0 (absent) or large. If momentum ever gated ship_ready, the large-gap
    # run would flip to False and this equality would break.
    frames = [
        {"shot_id": "clip-1", "features": [1.0, 1.0]},
        {"shot_id": "clip-2", "features": [1.1, 1.05]},  # coherent (real dispersion) ⇒ ship-ready
    ]
    absent = gestalt_summary(
        frames=frames, frame_analyses=[], loudness={"integrated_lufs": -14.0}, ceilings={}
    )  # default momentum_gap=0.0
    large = gestalt_summary(
        frames=frames,
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={},
        momentum_gap=0.9,
    )
    # Both ship-ready; only the momentum_gap value differs, everything else identical.
    assert absent["ship_ready"] is True
    assert large["ship_ready"] is True
    assert {k: v for k, v in absent.items() if k != "momentum_gap"} == {
        k: v for k, v in large.items() if k != "momentum_gap"
    }


# ── gestalt_summary: momentum becomes a HARD gate once a ceiling is configured ─
_SHIPPABLE_FRAMES = [
    {"shot_id": "clip-1", "features": [1.0, 1.0]},
    {"shot_id": "clip-2", "features": [1.1, 1.05]},  # coherent (real dispersion), no breaks
]


def test_gestalt_summary_momentum_gap_over_configured_ceiling_blocks_ship() -> None:
    # An otherwise ship-ready film fails ONLY because the momentum gap (0.9) exceeds the
    # configured ceiling (0.5) — proves the opt-in gate actually gates.
    g = gestalt_summary(
        frames=_SHIPPABLE_FRAMES,
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={"momentum_gap_max": 0.5},
        momentum_gap=0.9,
    )
    assert g["momentum_ok"] is False
    assert g["ship_ready"] is False


def test_gestalt_summary_momentum_gap_under_configured_ceiling_ships() -> None:
    # Same ceiling, a gap within it (0.3 <= 0.5) => momentum_ok, ship_ready unaffected.
    g = gestalt_summary(
        frames=_SHIPPABLE_FRAMES,
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={"momentum_gap_max": 0.5},
        momentum_gap=0.3,
    )
    assert g["momentum_ok"] is True
    assert g["ship_ready"] is True


def test_gestalt_summary_momentum_gap_at_ceiling_is_inclusive() -> None:
    # Half-open the other way: gap == ceiling passes (<=), only strictly-over fails.
    g = gestalt_summary(
        frames=_SHIPPABLE_FRAMES,
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={"momentum_gap_max": 0.5},
        momentum_gap=0.5,
    )
    assert g["momentum_ok"] is True
    assert g["ship_ready"] is True


def test_gestalt_summary_momentum_ceiling_gates_fewer_than_two_frames() -> None:
    # The <2-frame branch (ship_ready rides on loudness) is ALSO gated by momentum once
    # a ceiling is set: good loudness is no longer sufficient if the gap blows the ceiling.
    g = gestalt_summary(
        frames=[],
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={"momentum_gap_max": 0.5},
        momentum_gap=0.9,
    )
    assert g["momentum_ok"] is False
    assert g["ship_ready"] is False


# ── continuity_stats: the UNCENSORED delta distribution (calibration evidence) ─
# Five shots whose four adjacent deltas (0.1, 0.2, 0.05, 0.25) are ALL below the
# default 0.5 ceiling — so continuity_deltas reports nothing at all.
_BELOW_CEILING_FRAMES = [
    {"shot_id": "clip-1", "features": [0.0]},
    {"shot_id": "clip-2", "features": [0.1]},
    {"shot_id": "clip-3", "features": [0.3]},
    {"shot_id": "clip-4", "features": [0.35]},
    {"shot_id": "clip-5", "features": [0.6]},
]


def test_continuity_stats_reports_every_pair_while_breaks_is_empty() -> None:
    # THE anti-censoring proof: continuity_deltas is censored at the ceiling, so at the
    # default 0.5 it emits [] and the persisted evidence would be empty. continuity_stats
    # still sees all four adjacent pairs — that distribution is what a future ceiling
    # calibration needs and could never have been recovered from continuity_breaks.
    g = gestalt_summary(
        frames=_BELOW_CEILING_FRAMES,
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={},
    )
    assert g["continuity_breaks"] == []  # censored view: nothing breached 0.5
    stats = cast(dict[str, object], g["continuity_stats"])
    assert stats["n"] == 4  # uncensored view: every adjacent pair
    assert stats["min"] == 0.05
    assert stats["max"] == 0.25
    assert stats["mean"] == pytest.approx(0.15)


def test_continuity_stats_percentiles_are_ordered() -> None:
    stats = continuity_stats(_BELOW_CEILING_FRAMES)
    lo = cast(float, stats["min"])
    p50 = cast(float, stats["p50"])
    p90 = cast(float, stats["p90"])
    p95 = cast(float, stats["p95"])
    hi = cast(float, stats["max"])
    assert lo <= p50 <= p90 <= p95 <= hi


def test_continuity_stats_degenerate_film_has_no_pairs() -> None:
    # No adjacent pair exists ⇒ {"n": 0}. The key is still emitted by gestalt_summary
    # (computed outside the <2-frame gate branch), so a consumer never KeyErrors.
    assert continuity_stats([]) == {"n": 0}
    assert continuity_stats([{"shot_id": "clip-1", "features": [1.0]}]) == {"n": 0}
    g = gestalt_summary(
        frames=[], frame_analyses=[], loudness={"integrated_lufs": -14.0}, ceilings={}
    )
    assert g["continuity_stats"] == {"n": 0}


def test_continuity_stats_single_pair_has_zero_stdev() -> None:
    # Two frames ⇒ exactly one delta ⇒ population stdev of a single sample is 0.0.
    stats = continuity_stats(
        [{"shot_id": "clip-1", "features": [0.0]}, {"shot_id": "clip-2", "features": [0.3]}]
    )
    assert stats["n"] == 1
    assert stats["stdev"] == 0.0
    assert stats["min"] == stats["max"] == stats["p50"] == stats["p95"] == 0.3


def test_continuity_stats_are_ceiling_independent() -> None:
    # Same film, two ceilings that produce OPPOSITE break verdicts: the stats block is
    # byte-identical either way, proving it summarizes the raw distribution and is not
    # a re-shaped view of the censored break list.
    default = gestalt_summary(
        frames=_BELOW_CEILING_FRAMES,
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={},
    )
    zero = gestalt_summary(
        frames=_BELOW_CEILING_FRAMES,
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={"continuity_l2": 0.0},
    )
    assert default["continuity_breaks"] == []
    assert len(cast(list[object], zero["continuity_breaks"])) == 4  # every pair breaches 0.0
    assert default["continuity_stats"] == zero["continuity_stats"]


def test_ship_ready_byte_identical_with_and_without_continuity_stats() -> None:
    # Gate-inertness proof (mirrors the momentum_gap byte-identity guard above): across
    # widely varying delta distributions — flat, mild drift, one hard jump, all-jumps —
    # ship_ready and continuity_breaks are EXACTLY what the pre-existing math
    # (continuity_deltas + cohesion_score, at the default 0.5 ceiling/floor) yields.
    # Adding continuity_stats perturbs neither, for any distribution.
    # NB the all-flat distribution moved to its own degeneracy test: identical vectors are
    # ZERO DISPERSION, which is now `insufficient_evidence`, not a coherent film. Every
    # distribution here carries real signal.
    distributions = [
        [0.0, 0.1, 0.3, 0.35],  # mild drift, all under the ceiling
        [0.0, 0.05, 5.0, 5.1],  # one hard jump
        [0.0, 3.0, 6.0, 9.0],  # every cut is a break
    ]
    for values in distributions:
        frames = [{"shot_id": f"clip-{i + 1}", "features": [v]} for i, v in enumerate(values)]
        g = gestalt_summary(
            frames=frames, frame_analyses=[], loudness={"integrated_lufs": -14.0}, ceilings={}
        )
        expected_breaks = continuity_deltas(frames, 0.5)
        cohesion = cohesion_score(frames)
        assert cohesion is not None  # every distribution above genuinely disperses
        assert g["continuity_breaks"] == expected_breaks
        assert g["ship_ready"] is (not expected_breaks and cohesion >= 0.5)
        # ...and the advisory block rode along regardless of the verdict.
        assert cast("dict[str, object]", g["continuity_stats"])["n"] == 3


def test_flat_distribution_is_degenerate_not_a_coherent_film() -> None:
    # The case removed from the sweep above, pinned explicitly: four identical frames
    # produce four identical feature vectors ⇒ every adjacent delta is exactly 0.0. That
    # is the absence of a measurement, not a flawless film.
    frames = [{"shot_id": f"clip-{i + 1}", "features": [0.0]} for i in range(4)]
    g = gestalt_summary(
        frames=frames, frame_analyses=[], loudness={"integrated_lufs": -14.0}, ceilings={}
    )
    stats = cast("dict[str, object]", g["continuity_stats"])
    assert stats == {"n": 3, "degenerate": True}  # no min/max/mean/stdev/percentiles
    assert g["cohesion"] is None
    assert g["continuity_ceiling_suggested"] is None
    assert g["insufficient_evidence"] is True
    assert g["ship_ready"] is False  # in-window -14 LUFS is NOT evidence of a film


# ── continuity_ceiling_suggested: ADVISORY two-sigma within-run suggestion (INERT) ────
def test_continuity_ceiling_suggested_is_mean_plus_two_stdev() -> None:
    # >=2 adjacent deltas ⇒ the suggestion is exactly round(mean + 2*stdev, 4), derived
    # from the SAME continuity_stats(frames) the summary computes (no magic number).
    stats = continuity_stats(_BELOW_CEILING_FRAMES)
    mean = cast(float, stats["mean"])
    stdev = cast(float, stats["stdev"])
    expected = round(mean + 2.0 * stdev, 4)
    g = gestalt_summary(
        frames=_BELOW_CEILING_FRAMES,
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={},
    )
    assert g["continuity_ceiling_suggested"] == expected


def test_continuity_ceiling_suggested_none_when_fewer_than_two_frames() -> None:
    # 0 frames and 1 frame ⇒ no adjacent delta ⇒ no distribution ⇒ None (not fabricated).
    empty = gestalt_summary(
        frames=[], frame_analyses=[], loudness={"integrated_lufs": -14.0}, ceilings={}
    )
    assert empty["continuity_ceiling_suggested"] is None
    single = gestalt_summary(
        frames=[{"shot_id": "clip-1", "features": [1.0]}],
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={},
    )
    assert single["continuity_ceiling_suggested"] is None


def test_continuity_ceiling_suggested_is_inert_on_the_gate() -> None:
    # INERTNESS: the suggestion (a mean+2*stdev value well above the 0.5 default) does NOT move
    # continuity_breaks / ship_ready / cohesion — those stay exactly what the pre-existing
    # math yields at the hardcoded 0.5 ceiling/floor. Here the four sub-0.5 deltas breach
    # nothing, so the film is ship-ready even though the suggested ceiling differs from 0.5.
    g = gestalt_summary(
        frames=_BELOW_CEILING_FRAMES,
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={},
    )
    # The suggestion exists and is NOT the live 0.5 ceiling — proving it is a distinct,
    # advisory number rather than the gate value.
    suggested = cast(float, g["continuity_ceiling_suggested"])
    assert suggested != 0.5
    # ...yet the gate evidence is byte-identical to the 0.5-ceiling pre-change math.
    assert g["continuity_breaks"] == continuity_deltas(_BELOW_CEILING_FRAMES, 0.5)
    assert g["continuity_breaks"] == []  # every delta is under 0.5
    _cohesion = cohesion_score(_BELOW_CEILING_FRAMES)
    assert _cohesion is not None
    assert g["cohesion"] == round(_cohesion, 4)
    assert g["ship_ready"] is True  # no breaks + cohesion clears 0.5 + in-window loudness


# ── REGRESSION (item 5): ONE seconds coercer, shared with the retention curve ──
# `build_beat_intervals` used the STRICT `as_float` while `retention._seconds` parsed
# numeric strings. A JSONB spine legitimately round-trips 5.0 as "5.0", so:
#   * an ALL-string spine made every beat 0.0 wide ⇒ offset 0.0 ⇒ None ⇒ the caller
#     silently fell back to the positional zip, and
#   * a MIXED spine was worse — the string beats became ZERO-WIDTH spans that no cut can
#     ever fall inside, so every cut was mis-attributed with no signal that anything broke.
def test_build_beat_intervals_parses_string_seconds_like_the_watch_curve() -> None:
    lines = [{"id": "clip-1", "seconds": "5.0"}, {"id": "clip-2", "seconds": "5.0"}]
    assert build_beat_intervals(lines) == [("clip-1", 0.0, 5.0), ("clip-2", 5.0, 10.0)]
    # ... and it agrees with the curve it is diffed against (same coercer, same widths).
    assert predicted_watch_curve(lines) == predicted_watch_curve(
        [{"id": "clip-1", "seconds": 5.0}, {"id": "clip-2", "seconds": 5.0}]
    )


def test_build_beat_intervals_mixed_spine_has_no_zero_width_spans() -> None:
    # The nastiest shape: some beats numeric, some stringified. Every span must be real.
    lines = [
        {"id": "clip-1", "seconds": 4.0},
        {"id": "clip-2", "seconds": "6.0"},
        {"id": "clip-3", "seconds": 5},
    ]
    intervals = build_beat_intervals(lines)
    assert intervals is not None  # a parseable spine ⇒ real intervals, never the fallback
    assert intervals == [("clip-1", 0.0, 4.0), ("clip-2", 4.0, 10.0), ("clip-3", 10.0, 15.0)]
    assert all(end > start for _id, start, end in intervals)
    # ... and a cut inside the previously zero-width beat now lands on it.
    assert sample_frames_at_cuts(
        [7.0], 15.0, ["clip-1", "clip-2", "clip-3"], beat_intervals=intervals
    ) == [("clip-2", 7.0)]


def test_build_beat_intervals_raises_loudly_on_an_unparseable_duration() -> None:
    # A duration that is neither absent nor parseable is a real wire defect. It raises
    # (the stage's momentum guard logs it with a traceback and falls back) rather than
    # silently fabricating a 0.0-wide beat.
    for bad in ("soon", object(), float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="unparseable"):
            build_beat_intervals([{"id": "clip-1", "seconds": bad}])
    # A MISSING duration is legitimate (an explicitly duration-less beat) ⇒ 0.0, no raise.
    assert build_beat_intervals([{"id": "clip-1"}, {"id": "clip-2", "seconds": 3.0}]) == [
        ("clip-1", 0.0, 0.0),
        ("clip-2", 0.0, 3.0),
    ]


# ── REGRESSION (item 7): the retention cliff applies on ONE shared time axis ───
# `assembled_momentum_curve` took the 0.85 cliff step on the RENDERED, t0-rebased cut
# clock; `predicted_watch_curve` takes it on the INTENDED cumulative beat offset.
# momentum_gap self-normalizes SCALE but cannot normalize away a STEP, so a small render
# overrun that straddles the 30s cliff made one curve drop 15% and not the other —
# fabricating a large gap out of pure clock drift.
def test_cliff_step_on_the_rendered_clock_fabricates_a_gap() -> None:
    # Beat 2 is INTENDED at offset 29.0 (before the cliff) but RENDERS at 31.0 (after it)
    # — a 2s overrun. Motion is uniform and full, so a correct read is ~0.
    intended = [{"id": "c1", "seconds": 29.0}, {"id": "c2", "seconds": 5.0}]
    predicted = list(zip(["c1", "c2"], predicted_watch_curve(intended), strict=True))
    rendered_cuts = [0.0, 31.0]
    motion = [1.0, 1.0]

    # BEFORE (cliff on the rendered clock): the step lands on the assembled curve only.
    skewed = list(
        zip(["c1", "c2"], assembled_momentum_curve(rendered_cuts, motion, 40.0), strict=True)
    )
    skewed_gap = momentum_gap(skewed, predicted)
    assert skewed_gap is not None and skewed_gap > 0.10  # ~0.109, pure artifact

    # AFTER (cliff evaluated on the SHARED intended axis): the step cancels.
    aligned = list(
        zip(
            ["c1", "c2"],
            assembled_momentum_curve(rendered_cuts, motion, 40.0, cliff_times=[0.0, 29.0]),
            strict=True,
        )
    )
    aligned_gap = momentum_gap(aligned, predicted)
    assert aligned_gap is not None and aligned_gap < 0.05  # ~0.020, only real drift left
    assert aligned_gap < skewed_gap


def test_cliff_times_still_fires_when_both_axes_are_past_the_cliff() -> None:
    # Control: the cliff is not disabled — a beat genuinely past 30s on BOTH axes still
    # takes the 0.85 step, so the fix removes the artifact, not the model.
    with_cliff = assembled_momentum_curve([0.0, 31.0], [1.0, 1.0], 40.0, cliff_times=[0.0, 31.0])
    without_cliff = assembled_momentum_curve([0.0, 31.0], [1.0, 1.0], 40.0, cliff_times=[0.0, 29.0])
    assert with_cliff[1] == pytest.approx(without_cliff[1] * 0.85)


def test_coerce_beat_seconds_is_the_single_shared_contract() -> None:
    # The canonical coercer both call sites now use. Pinned here so a future "tidy-up"
    # that re-strictifies one of them fails loudly.
    assert coerce_beat_seconds("5.0") == 5.0
    assert coerce_beat_seconds(5) == 5.0
    assert coerce_beat_seconds(None) == 0.0  # explicitly duration-less beat
    assert coerce_beat_seconds("soon") is None
    assert coerce_beat_seconds(True) is None  # a bool is never a duration
    assert coerce_beat_seconds(float("nan")) is None
    assert coerce_beat_seconds(float("inf")) is None


# ══ REGRESSION: structurally-degenerate evidence is NOT a measurement ══════════
# The analyzer emits no per-frame brightness / sharpness / dominant_color, so
# `_feature_vector` fills five of six components from whole-video constants and only
# `motion` can differ between shots. On a static capture even that is 0, so every sampled
# vector is identical. Fixing the cut spine (correct, and kept) raises the number of
# sampled shots but cannot create signal — so without these guards it simply produces MORE
# fabricated zeros, which the nightly calibration job then turns into a maximally-strict
# gate while `corpus_n` reads healthy.
def test_continuity_stats_all_zero_deltas_is_degenerate_not_a_distribution() -> None:
    frames = [{"shot_id": f"clip-{i}", "features": [1.0, 2.0]} for i in range(1, 5)]
    stats = continuity_stats(frames)
    # The pair count is honest, but NO percentile/summary key is published …
    assert stats == {"n": 3, "degenerate": True}
    # … which is what makes the metric emitter and the ceiling suggester skip it: both
    # read `max` / `mean` / `stdev`, and absent keys cannot be observed or averaged.
    assert "max" not in stats
    assert "mean" not in stats
    assert "stdev" not in stats


def test_continuity_stats_keeps_the_full_distribution_when_signal_exists() -> None:
    # Control: one non-zero delta anywhere and the real distribution is published.
    frames = [
        {"shot_id": "clip-1", "features": [1.0]},
        {"shot_id": "clip-2", "features": [1.0]},
        {"shot_id": "clip-3", "features": [1.4]},
    ]
    stats = continuity_stats(frames)
    assert "degenerate" not in stats
    assert stats["n"] == 2
    assert stats["max"] == pytest.approx(0.4)


def test_tonal_variance_blind_keyword_proxy_reports_unmeasured() -> None:
    # `_tone_vector` is a 3-bucket keyword counter over a ~24-word vocabulary. Real
    # narration prose hits NONE of those buckets, so every shot reduces to [0,0,0] and the
    # computed variance is 0.0 — the counter failing to fire, not a consistent film.
    prose = [
        {"shot_id": "clip-1", "description": "A dashboard loads showing a project board."},
        {"shot_id": "clip-2", "description": "The cursor hovers and a detail drawer opens."},
        {"shot_id": "clip-3", "description": "Rendered clips appear along the filmstrip."},
    ]
    assert tonal_variance(prose) == {"variance": None, "worst_cut": None}


def test_tonal_variance_never_invents_a_worst_cut_from_a_zero_jump() -> None:
    # `worst_jump` seeded at -1.0 meant the FIRST adjacent pair always won, so identical
    # tone vectors still produced a named seam with jump 0.0 presented as the film's worst
    # tonal break. One keyword fires here, so variance IS measured — but every pair is
    # identical, so there is no worst cut to name.
    same = [{"shot_id": f"clip-{i}", "description": "welcome"} for i in range(1, 4)]
    tv = tonal_variance(same)
    assert tv["variance"] == 0.0  # genuinely measured: the proxy fired on every shot
    assert tv["worst_cut"] is None  # ... and found no jump — so it names none


def test_gestalt_summary_zero_dispersion_frames_are_insufficient_evidence() -> None:
    # End-to-end: several sampled shots, all identical in feature space, in-window
    # loudness. The gate must NOT read that as a coherent ship-ready film.
    frames = [{"shot_id": f"clip-{i}", "features": [5.0, 5.0]} for i in range(1, 4)]
    g = gestalt_summary(
        frames=frames, frame_analyses=[], loudness={"integrated_lufs": -14.0}, ceilings={}
    )
    assert g["insufficient_evidence"] is True
    assert g["ship_ready"] is False
    assert g["cohesion"] is None
    assert g["continuity_ceiling_suggested"] is None
    assert cast("dict[str, object]", g["continuity_stats"])["degenerate"] is True


# ── momentum_verdict: the unarmed state is NAMED, not reported as a pass ──────
# Before A/V wave 7, `momentum_ok` was a bare bool that was True whenever no
# `ceilings["momentum_gap_max"]` was configured — and nothing configures it anywhere.
# So an AND-term of `ship_ready` was vacuously True on every master ever produced and
# was indistinguishable in the payload from a term that had measured something.


def test_unconfigured_momentum_is_named_not_a_pass() -> None:
    # THE POINT: no ceiling ⇒ the term observed NOTHING. `momentum_ok` stays True (so
    # `ship_ready` is unchanged) but the payload no longer claims a pass.
    g = gestalt_summary(
        frames=_SHIPPABLE_FRAMES,
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={},
        momentum_gap=0.9,  # a gap that WOULD breach any sane ceiling, and is not judged
    )
    assert g["momentum_verdict"] == "unconfigured"
    assert g["momentum_verdict"] != "pass"
    assert g["momentum_ok"] is True  # byte-identical ship behaviour
    assert g["ship_ready"] is True


def test_configured_and_unmeasured_momentum_is_insufficient_evidence_not_a_breach() -> None:
    # An operator armed the term and the gap could not be measured. That is NOT ok (as
    # before) but it is also NOT a claim that the film's momentum sagged.
    g = gestalt_summary(
        frames=_SHIPPABLE_FRAMES,
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={"momentum_gap_max": 0.5},
        momentum_gap=None,
    )
    assert g["momentum_verdict"] == "insufficient_evidence"
    assert g["momentum_ok"] is False
    assert g["ship_ready"] is False


def test_momentum_verdict_states_are_exhaustive_and_distinct() -> None:
    assert momentum_verdict(0.9, None) == "unconfigured"
    assert momentum_verdict(None, None) == "unconfigured"
    assert momentum_verdict(None, 0.5) == "insufficient_evidence"
    assert momentum_verdict(0.3, 0.5) == "pass"
    assert momentum_verdict(0.5, 0.5) == "pass"  # inclusive, mirrors the ship term
    assert momentum_verdict(0.9, 0.5) == "fail"


def test_configured_breach_still_reads_as_fail() -> None:
    g = gestalt_summary(
        frames=_SHIPPABLE_FRAMES,
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={"momentum_gap_max": 0.5},
        momentum_gap=0.9,
    )
    assert g["momentum_verdict"] == "fail"
    assert g["momentum_ok"] is False


def test_no_default_momentum_ceiling_is_smuggled_in() -> None:
    # ⛔ GUARD: arming this term with a guessed threshold is the failure mode this
    # project keeps hitting. With no ceiling configured, NO gap value may ever produce
    # a pass/fail verdict — only `unconfigured`.
    for gap in (0.0, 0.01, 0.25, 0.5, 0.99, 5.0, None):
        g = gestalt_summary(
            frames=_SHIPPABLE_FRAMES,
            frame_analyses=[],
            loudness={"integrated_lufs": -14.0},
            ceilings={},
            momentum_gap=gap,
        )
        assert g["momentum_verdict"] == "unconfigured", gap
        assert g["momentum_ok"] is True, gap


def test_final_mux_loudness_drops_the_keys_nothing_reads() -> None:
    # `lra` / `silence_ratio` were emitted into the gate's verdict payload and read by
    # NO predicate here or in `final_critique`. They are measured and persisted at the
    # source (the compiler's `parse_loudness` block), so dropping the copy loses nothing.
    out = final_mux_loudness({"integrated_lufs": -14.0, "lra": 7.0, "silence_ratio": 0.1})
    assert out is not None
    assert "lra" not in out
    assert "silence_ratio" not in out
    assert set(out) == {"integrated_lufs", "true_peak_dbtp"}


# ── unobserved_terms: an unarmed term is insufficient evidence, not a pass ────
def test_an_unarmed_momentum_term_is_reported_unobserved_rather_than_passed() -> None:
    """⭐ `momentum_ok` is vacuously True on every master, and now it SAYS so.

    Nothing anywhere sets `ceilings["momentum_gap_max"]`, so `momentum_verdict` is
    `unconfigured` on 100% of the fleet, `_momentum_ok` collapses that to True, and
    `ship_ready` ANDs in a term that observed nothing. `unobserved_terms` is the explicit
    insufficient-evidence list that makes the vacuity readable from the record.

    ⛔ It is REPORTING, not a gate, and that is deliberate: failing an unarmed term closed
    with no calibrated ceiling would make `ship_ready` false on 100% of masters — the
    near-universal-flag failure, not a gate. So `momentum_ok` and `ship_ready` are
    asserted UNCHANGED right beside it.
    """
    frames = [
        {"shot_id": "clip-1", "features": [1.0, 1.0]},
        {"shot_id": "clip-2", "features": [1.1, 1.05]},
    ]
    g = gestalt_summary(
        frames=frames, frame_analyses=[], loudness={"integrated_lufs": -14.0}, ceilings={}
    )

    assert g["momentum_verdict"] == MOMENTUM_UNCONFIGURED
    assert g["unobserved_terms"] == ["momentum"]
    # Unchanged, for ALL inputs — the honesty may not sink a master.
    assert g["momentum_ok"] is True
    assert g["ship_ready"] is True


def test_an_armed_and_measured_momentum_term_is_not_unobserved() -> None:
    """FALSE-POSITIVE GUARD: a term that DID observe its subject must not be listed.

    A list that names every term regardless is exactly the signal-that-fires-on-
    everything this repo has shipped before. With a ceiling configured and a gap
    measured, `momentum` leaves the list; loudness, measured and in-window, was never
    on it.
    """
    frames = [
        {"shot_id": "clip-1", "features": [1.0, 1.0]},
        {"shot_id": "clip-2", "features": [1.1, 1.05]},
    ]
    g = gestalt_summary(
        frames=frames,
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={"momentum_gap_max": 0.25},
        momentum_gap=0.1,
    )

    assert g["momentum_verdict"] == MOMENTUM_PASS
    assert g["unobserved_terms"] == []
    assert g["ship_ready"] is True


def test_an_armed_but_unmeasurable_momentum_term_is_listed_as_unobserved() -> None:
    """The OTHER unobservable state: an operator armed the term and the gap could not be
    measured. That already fails the term (`momentum_ok` False — "unobservable" must never
    pass a term somebody deliberately armed), and it belongs on the list for the same
    reason `unconfigured` does: neither verdict looked at a master."""
    frames = [
        {"shot_id": "clip-1", "features": [1.0, 1.0]},
        {"shot_id": "clip-2", "features": [1.1, 1.05]},
    ]
    g = gestalt_summary(
        frames=frames,
        frame_analyses=[],
        loudness={"integrated_lufs": -14.0},
        ceilings={"momentum_gap_max": 0.25},
        momentum_gap=None,
    )

    assert g["momentum_verdict"] == MOMENTUM_INSUFFICIENT_EVIDENCE
    assert g["unobserved_terms"] == ["momentum"]
    assert g["momentum_ok"] is False
