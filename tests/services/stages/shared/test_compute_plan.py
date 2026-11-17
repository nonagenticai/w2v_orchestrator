"""Unit tests for the pure compute-everything planning substrate.

No DB — these are pure functions.
"""

from __future__ import annotations

import pytest

from src.services.stages.shared.compute_plan import (
    MixGainTargets,
    beat_word_budget,
    bind_clips_to_lines,
    mix_gain_targets,
    pick_distinct_voices,
    transition_gain_db,
)
from src.services.stages.shared.provenance import ProvenanceValue

# --- mix_gain_targets: byte-identical neutral guard ------------------------


def test_neutral_mix_targets_are_todays_literals_exactly() -> None:
    # The byte-identical guard: a neutral run pins to the four literals so
    # re-sourcing sound_design's defaults through this seam changes nothing.
    targets = mix_gain_targets()
    assert isinstance(targets, MixGainTargets)
    assert targets.music.value == -18.0
    assert targets.ambient.value == -30.0
    assert targets.duck.value == -12.0
    assert targets.sfx.value == -10.0


@pytest.mark.parametrize(
    ("vo_lufs", "mood"),
    [(None, None)],
)
def test_neutral_when_vo_lufs_or_mood_absent(vo_lufs: float | None, mood: str | None) -> None:
    # Neutral holds when vo_lufs is None OR mood is None/"neutral" (any case).
    targets = mix_gain_targets(vo_lufs, mood)
    assert (
        targets.music.value,
        targets.ambient.value,
        targets.duck.value,
        targets.sfx.value,
    ) == (-18.0, -30.0, -12.0, -10.0)


def test_neutral_source_is_default() -> None:
    targets = mix_gain_targets()
    for pv in (targets.music, targets.ambient, targets.duck, targets.sfx):
        assert isinstance(pv, ProvenanceValue)
        assert pv.source == "default"
        assert pv.computed_by == "mix_gain_targets"


def test_mix_targets_monotone_in_vo_lufs() -> None:
    # With a non-neutral mood, a louder VO (higher LUFS) deepens every target
    # monotonically (more negative). Assert strictly decreasing across a sweep.
    sweep = [mix_gain_targets(lufs, "energetic") for lufs in (-20.0, -16.0, -12.0, -8.0)]
    for field in ("music", "ambient", "duck", "sfx"):
        values = [getattr(t, field).value for t in sweep]
        assert values == sorted(values, reverse=True)
        assert values[0] > values[-1]  # strictly deeper at louder VO


# --- transition_gain_db ----------------------------------------------------


def test_transition_gain_defaults_to_the_neutral_sfx_target() -> None:
    # Empty brief → the same -10dB the accent cues get, NOT full level.
    assert transition_gain_db({}) == -10.0


def test_transition_gain_brief_override_wins() -> None:
    assert transition_gain_db({"sfx_gain_db": -4.0}) == -4.0


def test_transition_gain_zero_db_override_is_honoured() -> None:
    # 0.0 dB is a legitimate setting ("no attenuation") and falsy — a truthiness
    # check would silently discard it and fall back to -10dB.
    assert transition_gain_db({"sfx_gain_db": 0.0}) == 0.0


def test_transition_gain_none_override_falls_back() -> None:
    # An explicitly-null key is "unset", not an override.
    assert transition_gain_db({"sfx_gain_db": None}) == -10.0


def test_transition_gain_tracks_mix_target_when_no_override() -> None:
    # Absent a brief override it follows the computed sfx target: a louder VO
    # deepens the transition just as it deepens the accents.
    cfg = {"vo_lufs": -12.0, "mood": "energetic"}
    assert transition_gain_db(cfg) == mix_gain_targets(-12.0, "energetic").sfx.value
    assert transition_gain_db(cfg) < -10.0


def test_transition_gain_override_beats_the_mix_target() -> None:
    assert transition_gain_db({"vo_lufs": -12.0, "mood": "energetic", "sfx_gain_db": -2.0}) == -2.0


# --- beat_word_budget ------------------------------------------------------


@pytest.mark.parametrize(
    ("duration_s", "wpm"),
    [(60.0, 150.0)],
)
def test_beat_word_budget_matches_formula(duration_s: float, wpm: float) -> None:
    pv = beat_word_budget(duration_s, wpm)
    assert pv.value == pytest.approx(duration_s / 60.0 * wpm)


def test_beat_word_budget_provenance() -> None:
    pv = beat_word_budget(60.0, 150.0)
    assert isinstance(pv, ProvenanceValue)
    assert pv.source == "smart-auto"
    assert pv.computed_by == "beat_word_budget"


# --- bind_clips_to_lines ---------------------------------------------------


def test_bind_clips_fuzzy_and_spine_order_preserved() -> None:
    lines = [
        {"id": "C0", "clip": "@clip-1"},
        {"id": "C1"},  # falls back to its own id as the join key
        {"id": "clip-2", "clip": "clip-2"},
    ]
    # Available footage tags in mixed @-prefixed / bare form (the fuzzy join).
    must_show = ["clip-1", "@clip-2", "C1"]
    bound = bind_clips_to_lines(lines, must_show)
    # Spine ids preserved (C0 present), order follows the spine, not must_show.
    assert list(bound.keys()) == ["C0", "C1", "clip-2"]
    assert bound["C0"] == "clip-1"  # @clip-1 normalized joins bare clip-1
    assert bound["C1"] == "C1"
    assert bound["clip-2"] == "clip-2"


def test_bind_clips_omits_unmatched_and_is_deterministic() -> None:
    lines = [{"id": "C0"}, {"id": "C1"}, {"id": "C2"}]
    must_show = ["C0", "C2"]
    first = bind_clips_to_lines(lines, must_show)
    second = bind_clips_to_lines(lines, must_show)
    assert first == second == {"C0": "C0", "C2": "C2"}
    assert "C1" not in first  # no footage → omitted, spine not reordered


# --- pick_distinct_voices --------------------------------------------------


def test_pick_distinct_voices_are_distinct_and_ordered() -> None:
    catalog = ["v_a", "v_b", "v_c", "v_d"]
    speakers = ["Narrator", "Alice", "Bob"]
    got = pick_distinct_voices(speakers, catalog)
    assert list(got.keys()) == speakers
    assert len(set(got.values())) == 3  # all distinct


def test_pick_distinct_voices_pinned_wins_and_reserves() -> None:
    catalog = ["v_a", "v_b", "v_c"]
    got = pick_distinct_voices(["Alice", "Bob"], catalog, pinned={"Alice": "v_b"})
    assert got["Alice"] == "v_b"
    assert got["Bob"] != "v_b"  # pinned voice is reserved


def test_pick_distinct_voices_deterministic_when_exhausted() -> None:
    catalog = ["v_a", "v_b"]
    speakers = ["A", "B", "C", "D"]
    first = pick_distinct_voices(speakers, catalog)
    second = pick_distinct_voices(speakers, catalog)
    assert first == second
    assert first["A"] == "v_a" and first["B"] == "v_b"
    # Overflow speakers still get a stable catalog voice (per-name fallback).
    assert first["C"] in catalog and first["D"] in catalog
