"""Unit tests for the pure planning levers (composers over compute_plan).

No DB — pure functions.
"""

from __future__ import annotations

from src.services.planning import compute_levers
from src.services.planning.presets import load_preset
from src.services.stages.shared.provenance import ProvenanceValue

# --- beat_count ------------------------------------------------------------


def test_beat_count_grows_with_duration() -> None:
    preset = load_preset("default")
    short = compute_levers.beat_count(30.0, preset).value
    long = compute_levers.beat_count(300.0, preset).value
    assert long > short


def test_beat_count_is_floored_at_one() -> None:
    preset = load_preset("default")
    pv = compute_levers.beat_count(0.0, preset)
    assert isinstance(pv, ProvenanceValue)
    assert pv.value == 1  # a run always has at least one beat
    assert pv.computed_by == "compute_levers"


# --- mix_gains -------------------------------------------------------------


def test_mix_gains_neutral_equals_compute_plan_literals_as_default() -> None:
    preset = load_preset("default")  # vo_lufs -16
    targets = compute_levers.mix_gains(preset, "neutral")
    assert (targets.music.value, targets.ambient.value, targets.duck.value, targets.sfx.value) == (
        -18.0,
        -30.0,
        -12.0,
        -10.0,
    )
    for pv in (targets.music, targets.ambient, targets.duck, targets.sfx):
        assert pv.source == "default"


def test_mix_gains_none_mood_uses_preset_default_mood() -> None:
    # default preset's mood_default is "neutral" → still the pinned literals.
    preset = load_preset("default")
    targets = compute_levers.mix_gains(preset, None)
    assert targets.music.value == -18.0
    assert targets.music.source == "default"


# --- build_beats -----------------------------------------------------------


def test_build_beats_ids_are_sacred_clip_spine() -> None:
    beats = compute_levers.build_beats(["a", "b", "c"], [], 0)
    assert [b["id"] for b in beats] == ["clip-1", "clip-2", "clip-3"]


def test_build_beats_must_say_appears_verbatim() -> None:
    phrases = ["Introducing the fastest way to ship.", "One click. Done."]
    beats = compute_levers.build_beats(["a", "b"], phrases, 0)
    texts = [b["text"] for b in beats]
    assert texts == phrases  # dealt one-per-beat, verbatim


def test_build_beats_overflow_phrases_not_dropped() -> None:
    # More phrases than must_show labels → the extra is appended verbatim.
    beats = compute_levers.build_beats(["only-one"], ["first phrase", "second phrase"], 0)
    assert len(beats) == 1
    assert "first phrase" in beats[0]["text"]
    assert "second phrase" in beats[0]["text"]


def test_build_beats_falls_back_to_n_beats_without_labels() -> None:
    beats = compute_levers.build_beats([], [], 4)
    assert [b["id"] for b in beats] == ["clip-1", "clip-2", "clip-3", "clip-4"]


def test_build_beats_empty_when_no_content() -> None:
    assert compute_levers.build_beats([], [], 0) == []


def test_build_beats_round_robins_speakers() -> None:
    beats = compute_levers.build_beats(["a", "b", "c"], [], 0, speakers=["guide", "new"])
    assert [b["speaker"] for b in beats] == ["guide", "new", "guide"]


# --- cast_assignment -------------------------------------------------------


def test_cast_assignment_is_distinct_and_deterministic() -> None:
    first = compute_levers.cast_assignment(["Narrator", "Guest"], None)
    second = compute_levers.cast_assignment(["Narrator", "Guest"], None)
    assert first == second
    assert len(set(first.values())) == 2  # distinct voices


def test_cast_assignment_honors_pins() -> None:
    got = compute_levers.cast_assignment(["A", "B"], {"A": "bm_george"})
    assert got["A"] == "bm_george"
    assert got["B"] != "bm_george"


# --- checkpoint_set --------------------------------------------------------


def test_checkpoint_set_maps_modes() -> None:
    preset = load_preset("default")
    assert compute_levers.checkpoint_set(preset, "autopilot") == []
    assert compute_levers.checkpoint_set(preset, "preview") == ["demo_video_gen"]
    assert compute_levers.checkpoint_set(preset, "careful") == ["demo_video_gen"]


def test_checkpoint_set_falls_back_to_preset_mode() -> None:
    preset = load_preset("founder_story")  # checkpoint_mode="careful"
    assert compute_levers.checkpoint_set(preset, "") == ["demo_video_gen"]
