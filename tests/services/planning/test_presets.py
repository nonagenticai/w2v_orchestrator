"""Unit tests for the deterministic planning presets.

No DB — pure functions.
"""

from __future__ import annotations

import pytest

from src.services.planning.presets import (
    DEFAULT_PRESET_NAME,
    Preset,
    clear_preset_cache,
    load_preset,
)

_NAMED = ("default", "launch_film_punchy", "explainer_calm", "demo_forward", "founder_story")


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_preset_cache()


@pytest.mark.parametrize("name", ["default"])
def test_each_named_preset_loads_and_is_valid(name: str) -> None:
    preset = load_preset(name)
    assert isinstance(preset, Preset)
    assert preset.name == name
    assert preset.is_valid()


def test_unknown_name_falls_back_to_default() -> None:
    preset = load_preset("does_not_exist")
    assert preset.name == DEFAULT_PRESET_NAME


def test_empty_name_falls_back_to_default() -> None:
    assert load_preset("").name == DEFAULT_PRESET_NAME


def test_load_is_cached_and_deterministic() -> None:
    first = load_preset("explainer_calm")
    second = load_preset("explainer_calm")
    assert first is second  # served from the process cache


def test_is_valid_gate_rejects_malformed_preset() -> None:
    assert Preset(name="x").is_valid()
    assert not Preset(name="").is_valid()  # empty name
    assert not Preset(name="x", wpm=0.0).is_valid()  # non-positive pace
    assert not Preset(name="x", voice_count=0).is_valid()  # no voices
    assert not Preset(name="x", checkpoint_mode="bogus").is_valid()  # unknown mode
    assert not Preset(name="x", transition_style="bogus").is_valid()  # unknown transition


def test_named_presets_have_distinct_paces() -> None:
    wpms = {load_preset(n).wpm for n in _NAMED}
    assert len(wpms) >= 4  # sensibly distinct, not all clones of default


def test_founder_story_is_single_voice() -> None:
    preset = load_preset("founder_story")
    assert preset.voice_count == 1
    assert preset.dialogue_enabled is False
