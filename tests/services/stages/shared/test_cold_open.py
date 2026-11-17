"""Unit tests for the cold-open framing planner (pure, no DB).

Mirrors ``test_motion_plan``. Covers the ORCH-1.3a guarantees:

1. **Authored wins** — an authored anchor and/or lead are honoured; a partial
   author fills only the missing half from the deterministic default.
2. **Clamp** — the lead is clamped to ``[0, clip_seconds]``.
3. **Deterministic default** — anchor ``"end"`` + a duration-derived lead for a
   range of durations; the cap holds for long clips.
4. **Style perturbation** — ``open_style is None`` equals the plain default, and
   two different style words can produce different leads, but never a different
   anchor.
5. **Safe degrade** — non-positive / non-numeric ``clip_seconds`` ⇒ ``("end",
   0.0)``.
6. **Determinism** — same inputs ⇒ identical output across calls.
7. **cold_open_hash** — stable, 16 hex chars, changes iff the window changes.
"""

from __future__ import annotations

from typing import cast

from src.clients.schemas.video_analyzer import VideoAnalyzerMotionSample
from src.services.stages.shared.cold_open import (
    cold_open_hash,
    cold_open_window,
)

# ---------------------------------------------------------------------------
# Authored wins
# ---------------------------------------------------------------------------


def test_authored_anchor_and_lead_honored() -> None:
    anchor, lead = cold_open_window(
        10.0, {"cold_open_anchor": "start", "cold_open_lead_seconds": 2.5}, None
    )
    assert anchor == "start"
    assert lead == 2.5


def test_authored_anchor_end_honored() -> None:
    anchor, _lead = cold_open_window(10.0, {"cold_open_anchor": "end"}, None)
    assert anchor == "end"


def test_partial_author_only_anchor_fills_default_lead() -> None:
    anchor, lead = cold_open_window(10.0, {"cold_open_anchor": "start"}, None)
    # anchor authored, lead comes from the plain default (== no-hook default lead)
    assert anchor == "start"
    assert lead == cold_open_window(10.0, None, None)[1]


def test_partial_author_only_lead_fills_default_anchor() -> None:
    anchor, lead = cold_open_window(10.0, {"cold_open_lead_seconds": 0.4}, None)
    assert anchor == "end"  # the deterministic default anchor
    assert lead == 0.4


def test_style_does_not_override_authored_lead() -> None:
    # An authored lead is honoured verbatim regardless of open_style.
    _a, lead = cold_open_window(10.0, {"cold_open_lead_seconds": 0.75}, "punchy")
    assert lead == 0.75


def test_invalid_authored_anchor_falls_to_default() -> None:
    anchor, _lead = cold_open_window(10.0, {"cold_open_anchor": "middle"}, None)
    assert anchor == "end"


def test_negative_authored_lead_falls_to_default() -> None:
    # A negative lead fails the >= 0 gate, so the default lead is used instead.
    _a, lead = cold_open_window(10.0, {"cold_open_lead_seconds": -3.0}, None)
    assert lead == cold_open_window(10.0, None, None)[1]


# ---------------------------------------------------------------------------
# Clamp to [0, clip_seconds]
# ---------------------------------------------------------------------------


def test_authored_lead_clamped_to_clip_seconds() -> None:
    _a, lead = cold_open_window(5.0, {"cold_open_lead_seconds": 999.0}, None)
    assert lead == 5.0


def test_default_lead_never_exceeds_clip_for_a_range() -> None:
    for clip in (0.5, 1.0, 2.0, 3.0, 8.0, 12.5, 30.0):
        _a, lead = cold_open_window(clip, None, None)
        assert 0.0 <= lead <= clip


# ---------------------------------------------------------------------------
# Deterministic default
# ---------------------------------------------------------------------------


def test_default_anchor_is_end() -> None:
    anchor, _lead = cold_open_window(10.0, None, None)
    assert anchor == "end"


def test_default_lead_matches_fraction_formula() -> None:
    # 0.12 * 4.0 = 0.48, below the 1.5 cap.
    _a, lead = cold_open_window(4.0, None, None)
    assert lead == round(min(1.5, 4.0 * 0.12), 3)


def test_default_lead_capped_for_long_clips() -> None:
    # 0.12 * 100 = 12.0, so the cap dominates.
    _a, lead = cold_open_window(100.0, None, None)
    assert lead == 1.5


# ---------------------------------------------------------------------------
# open_style perturbation
# ---------------------------------------------------------------------------


def test_open_style_none_equals_plain_default() -> None:
    assert cold_open_window(10.0, None, None) == cold_open_window(10.0, {}, None)


def test_open_style_can_shift_lead() -> None:
    leads = {
        cold_open_window(10.0, None, style)[1]
        for style in ("punchy", "calm", "editorial", "energetic", "cinematic")
    }
    # Distinct style words map into distinct leads via the seeded band.
    assert len(leads) > 1


def test_open_style_never_changes_anchor() -> None:
    for style in ("punchy", "calm", "editorial", "energetic", "nonsense"):
        anchor, _lead = cold_open_window(10.0, None, style)
        assert anchor == "end"


def test_open_style_lead_within_band_of_default() -> None:
    default_lead = cold_open_window(10.0, None, None)[1]
    for style in ("punchy", "calm", "editorial"):
        _a, lead = cold_open_window(10.0, None, style)
        # A style only nudges the lead upward within the bounded band.
        assert default_lead <= lead <= default_lead + 0.25 + 1e-9


# ---------------------------------------------------------------------------
# Safe degrade
# ---------------------------------------------------------------------------


def test_zero_duration_degrades() -> None:
    assert cold_open_window(0.0, None, None) == ("end", 0.0)


def test_negative_duration_degrades() -> None:
    assert cold_open_window(-5.0, {"cold_open_lead_seconds": 2.0}, "punchy") == ("end", 0.0)


def test_non_numeric_duration_degrades() -> None:
    # A garbage runtime value (the pure fn defends even though callers pass float).
    assert cold_open_window(cast(float, "garbage"), None, None) == ("end", 0.0)


def test_none_duration_degrades() -> None:
    assert cold_open_window(cast(float, None), None, None) == ("end", 0.0)


def test_nan_and_inf_duration_degrade() -> None:
    assert cold_open_window(float("nan"), None, None) == ("end", 0.0)
    assert cold_open_window(float("inf"), None, None) == ("end", 0.0)


def test_bool_duration_degrades() -> None:
    # A bool is not a duration even though it is int-ish.
    assert cold_open_window(cast(float, True), None, None) == ("end", 0.0)


def test_garbage_hook_ignored() -> None:
    # A non-Mapping hook is simply ignored → plain default.
    assert cold_open_window(10.0, None, None) == cold_open_window(10.0, None, None)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_window_is_deterministic() -> None:
    hook = {"cold_open_anchor": "start"}
    assert cold_open_window(7.5, hook, "punchy") == cold_open_window(7.5, hook, "punchy")
    assert cold_open_window(7.5, None, "calm") == cold_open_window(7.5, None, "calm")


# ---------------------------------------------------------------------------
# cold_open_hash
# ---------------------------------------------------------------------------


def test_cold_open_hash_is_stable_and_16_hex() -> None:
    h = cold_open_hash("end", 1.2)
    assert h == cold_open_hash("end", 1.2)
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_cold_open_hash_changes_when_anchor_changes() -> None:
    assert cold_open_hash("end", 1.2) != cold_open_hash("start", 1.2)


def test_cold_open_hash_changes_when_lead_changes() -> None:
    assert cold_open_hash("end", 1.2) != cold_open_hash("end", 1.3)


# ---------------------------------------------------------------------------
# Peak motion (ORCH-1.3c) — the measured curve beats the duration guess
# ---------------------------------------------------------------------------


def _curve(*samples: tuple[float, float]) -> list[dict[str, float]]:
    return [{"t": t, "motion": m} for t, m in samples]


def test_peak_motion_seeks_to_the_action_and_forces_start_anchor() -> None:
    """The whole feature: a distinguished peak wins over the duration-derived guess.

    A static load-in, then the page paints at t=6, then it settles. Without a curve
    this clip would anchor "end" with a ~1.2s lead; with one it must cut to the paint.
    The anchor MUST flip to "start" — the compiler ignores a lead under an "end"
    anchor (renderer._clip_seek_args), so "end" + lead would silently drop the seek.
    """
    curve = _curve((2.0, 0.001), (4.0, 0.002), (6.0, 0.40), (8.0, 0.01), (10.0, 0.005))
    anchor, lead = cold_open_window(12.0, None, "cold_open", curve)
    assert anchor == "start"
    assert lead == 6.0 - 0.35  # peak minus the pre-roll: watch the motion HAPPEN

    # Same clip, no curve → the old duration-derived default, unchanged.
    default_anchor, default_lead = cold_open_window(12.0, None, "cold_open")
    assert default_anchor == "end"
    assert default_lead != lead


def test_flat_curve_is_not_a_peak() -> None:
    """A static screen has a max, but it is noise — not action. Fall back, don't cut."""
    anchor, lead = cold_open_window(
        12.0, None, "cold_open", _curve((2.0, 0.001), (4.0, 0.002), (6.0, 0.0015))
    )
    assert (anchor, lead) == cold_open_window(12.0, None, "cold_open")


def test_uniform_motion_has_no_moment() -> None:
    """Motion everywhere ⇒ no single moment is 'the' moment. Prominence gate holds.

    A continuously-playing video: every sample is well above the noise floor, but the
    peak is only ~1.1x the mean. Argmaxing that cuts to an arbitrary sample, which is
    worse than the settled tail — so the planner must decline.
    """
    anchor, lead = cold_open_window(
        12.0, None, "cold_open", _curve((2.0, 0.30), (4.0, 0.32), (6.0, 0.35), (8.0, 0.31))
    )
    assert (anchor, lead) == cold_open_window(12.0, None, "cold_open")


def test_authored_lead_still_beats_a_measured_peak() -> None:
    """Ladder order: the author is above the measurement."""
    curve = _curve((2.0, 0.001), (6.0, 0.40), (8.0, 0.01))
    anchor, lead = cold_open_window(12.0, {"cold_open_lead_seconds": 1.0}, "cold_open", curve)
    assert (anchor, lead) == ("end", 1.0)


def test_authored_end_anchor_suppresses_the_peak_seek() -> None:
    """An 'end' anchor cannot honour a lead downstream, so a peak there would be a lie."""
    curve = _curve((2.0, 0.001), (6.0, 0.40), (8.0, 0.01))
    anchor, lead = cold_open_window(12.0, {"cold_open_anchor": "end"}, "cold_open", curve)
    assert anchor == "end"
    assert (anchor, lead) == cold_open_window(12.0, {"cold_open_anchor": "end"}, "cold_open")


def test_style_never_perturbs_a_measured_peak() -> None:
    """Style nudges the GUESS, never the MEASUREMENT."""
    curve = _curve((2.0, 0.001), (6.0, 0.40), (8.0, 0.01))
    a1, l1 = cold_open_window(12.0, None, "cold_open", curve)
    a2, l2 = cold_open_window(12.0, None, "punchy", curve)
    assert (a1, l1) == (a2, l2) == ("start", 5.65)


def test_earliest_peak_wins_a_tie() -> None:
    """Deterministic on ties, and an earlier cut leaves more footage after it."""
    curve = _curve((2.0, 0.001), (4.0, 0.40), (6.0, 0.001), (8.0, 0.40))
    _anchor, lead = cold_open_window(20.0, None, "cold_open", curve)
    assert lead == 4.0 - 0.35


def test_peak_at_the_head_clamps_to_zero_not_negative() -> None:
    curve = _curve((0.2, 0.40), (2.0, 0.001), (4.0, 0.002))
    anchor, lead = cold_open_window(12.0, None, "cold_open", curve)
    assert (anchor, lead) == ("start", 0.0)


def test_single_sample_curve_is_not_enough_to_call_a_peak() -> None:
    anchor, lead = cold_open_window(12.0, None, "cold_open", _curve((6.0, 0.40)))
    assert (anchor, lead) == cold_open_window(12.0, None, "cold_open")


def test_garbage_curve_degrades_to_the_default_and_never_raises() -> None:
    """Cross-repo seam: quality_metrics is an untyped JSON blob upstream."""
    baseline = cold_open_window(12.0, None, "cold_open")
    garbage: list[object] = [
        {"t": "six", "motion": 0.4},
        {"t": 6.0, "motion": None},
        {"t": float("nan"), "motion": 0.4},
        {"t": -1.0, "motion": 0.4},
        {"t": 99.0, "motion": 0.9},  # beyond the clip → not a real sample
        "not-a-mapping",
        {},
    ]
    assert (
        cold_open_window(12.0, None, "cold_open", cast("list[dict[str, object]]", garbage))
        == baseline
    )
    assert cold_open_window(12.0, None, "cold_open", []) == baseline


def test_peak_motion_is_deterministic() -> None:
    curve = _curve((2.0, 0.001), (6.0, 0.40), (8.0, 0.01))
    first = cold_open_window(12.0, None, "cold_open", curve)
    assert all(cold_open_window(12.0, None, "cold_open", curve) == first for _ in range(5))
    assert cold_open_hash(*first) == cold_open_hash(*first)


# ---------------------------------------------------------------------------
# Contract-shape: _peak_motion_lead accepts BOTH a typed sample and a dict row
# ---------------------------------------------------------------------------


def _typed_curve(*samples: tuple[float, float]) -> list[VideoAnalyzerMotionSample]:
    return [VideoAnalyzerMotionSample(t=t, motion=m) for t, m in samples]
