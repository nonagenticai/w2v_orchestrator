"""Authoring presets — the named starting points a ``VideoIntent`` can pick.

A :class:`Preset` bundles the deterministic knob defaults a planning run reads
(narrator pace, reference VO loudness, cast size, dialogue on/off, transition
style, title-card templates, checkpoint mode). ``resolve_plan`` treats a preset
as the bottom rung of the provenance ladder — an authored brief value overrides
it, a computed smart-auto value overrides the global default, and the preset
supplies the default.

Storage model mirrors :mod:`src.services.stages.sound_catalog`: a small built-in
baseline is the **source of truth** here, wrapped in a short-TTL process cache so
a future tenant/MinIO override layer can slot in without touching call sites. The
``presets/*.yaml`` files ship the same values as documentation + the seam a remote
override would read; this slice is baseline-only (no remote read, no yaml parse).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# Closed sets the ``is_valid`` gate checks against (mirrors sound_catalog's
# category gate). Kept in code so a fresh env still validates the core presets.
_VALID_CHECKPOINT_MODES: frozenset[str] = frozenset({"autopilot", "preview", "careful"})
_VALID_TRANSITIONS: frozenset[str] = frozenset({"cut", "dissolve", "fade", "wipe", "match_cut"})


@dataclass(frozen=True)
class Preset:
    """One named authoring starting point.

    All fields are deterministic knob defaults — no LLM, no stochastic choice.
    ``mood_default`` feeds the mix-gain seam (``"neutral"`` pins the byte-identical
    literals); ``checkpoint_mode`` maps to the human-review gate set.
    """

    name: str
    #: ⛔ AN AUTHORED STYLE, NOT A MEASUREMENT — and deliberately NOT routed through
    #: ``stages/shared/speech_rate.resolve_natural_wpm``, unlike the rubric reference
    #: in ``evaluation/rubrics/script_craft`` which WAS routed there in the same wave.
    #: The two look identical (a wpm number) and are different quantities:
    #:
    #: * ``resolve_natural_wpm`` measures ONE fact — how fast the configured TTS voice
    #:   physically speaks. It returns a single number per run.
    #: * These five values are five DIFFERENT EDITORIAL PACES the author chose between
    #:   (``launch_film_punchy`` 165 vs ``explainer_calm`` 135). The SPREAD is the whole
    #:   product of the field; the preset is what the author picked, before any voice is
    #:   resolved.
    #:
    #: So the naive routing is a regression, not a fix: substituting the measured rate
    #: makes every preset compute the same ``beat_count`` and silently deletes the pace
    #: dimension from the preset system, while the caller's own docstring
    #: (``compute_levers.beat_count``) still claims it varies "at the preset's narration
    #: pace".
    #:
    #: ⚠️ WHAT IS GENUINELY WRONG HERE, STATED RATHER THAN FIXED: the single consumer
    #: ``compute_levers.beat_count`` (``planning/compute_levers.py:41``) reads this as an
    #: ABSOLUTE delivery rate, so a 165 against a voice that delivers ~120 over-counts
    #: beats and contributes to the overrun class. The correct fix is a RATIO — the
    #: measured voice rate scaled by this preset's pace relative to ``default``'s 150 —
    #: not a substitution. It is not made here because ``beat_count`` is a coarse
    #: scaffold (``WORDS_PER_BEAT = 30``, its own comment calls it "rough") that only
    #: chooses how many beats to draft, and the per-line word budget that actually binds
    #: delivered length IS already measured downstream, in each ``*/script.py`` stage via
    #: ``resolve_script_sample_wpm``. Re-tuning a scaffold ahead of the thing it feeds
    #: needs its own measurement; ``beat_count``'s contribution to overrun has never been
    #: measured on its own.
    wpm: float = 150.0
    vo_lufs: float = -16.0
    mood_default: str = "neutral"
    voice_count: int = 2
    dialogue_enabled: bool = True
    transition_style: str = "cut"
    card_templates: tuple[str, ...] = ("title_bullets", "value_prop", "cta")
    checkpoint_mode: str = "autopilot"

    def is_valid(self) -> bool:
        """True when the preset's knobs are self-consistent (the load gate).

        Mirrors ``SoundLibrary.is_valid``: an unknown/malformed preset never
        reaches a run — ``load_preset`` falls back to ``default`` instead.
        """
        return (
            bool(self.name)
            and self.wpm > 0.0
            and self.voice_count >= 1
            and self.checkpoint_mode in _VALID_CHECKPOINT_MODES
            and self.transition_style in _VALID_TRANSITIONS
        )


#: The built-in baseline (source of truth). Distinct, sensible values per name.
_BUILTIN_BASELINE: dict[str, Preset] = {
    "default": Preset(name="default"),
    "launch_film_punchy": Preset(
        name="launch_film_punchy",
        wpm=165.0,
        vo_lufs=-14.0,
        mood_default="energetic",
        voice_count=2,
        dialogue_enabled=True,
        transition_style="cut",
        checkpoint_mode="autopilot",
    ),
    "explainer_calm": Preset(
        name="explainer_calm",
        wpm=135.0,
        vo_lufs=-16.0,
        mood_default="calm",
        voice_count=1,
        dialogue_enabled=False,
        transition_style="dissolve",
        checkpoint_mode="preview",
    ),
    "demo_forward": Preset(
        name="demo_forward",
        wpm=155.0,
        vo_lufs=-15.0,
        mood_default="confident",
        voice_count=2,
        dialogue_enabled=True,
        transition_style="cut",
        checkpoint_mode="preview",
    ),
    "founder_story": Preset(
        name="founder_story",
        wpm=145.0,
        vo_lufs=-16.0,
        mood_default="warm",
        voice_count=1,
        dialogue_enabled=False,
        transition_style="dissolve",
        checkpoint_mode="careful",
    ),
}

#: The name every unknown/invalid lookup resolves to (never raises).
DEFAULT_PRESET_NAME = "default"


# --- Runtime loading (cached; baseline-only for this slice) -----------------

_PRESET_TTL_SECONDS = 60.0
# Process cache keyed by preset name. Short TTL so a future remote override is
# picked up without a restart (mirrors sound_catalog's _LIB_CACHE shape).
_PRESET_CACHE: dict[str, tuple[float, Preset]] = {}


def clear_preset_cache() -> None:
    """Drop the process cache (tests + after a re-seed)."""
    _PRESET_CACHE.clear()


def load_preset(name: str) -> Preset:
    """Resolve a preset by name, never raising — an unknown name → ``default``.

    Source of truth is :data:`_BUILTIN_BASELINE`; the result is cached per name
    for :data:`_PRESET_TTL_SECONDS`.

    SEAM (not implemented in this slice): a future tenant/MinIO override layer
    would read ``presets/<name>.yaml`` (or a bucket manifest) and merge it over
    the baseline right here, exactly as ``load_sound_library`` layers a manifest
    over ``BUILTIN_CATALOG``. Until then the yaml files are documentation only.
    """
    key = name or DEFAULT_PRESET_NAME

    hit = _PRESET_CACHE.get(key)
    if hit is not None and (time.monotonic() - hit[0]) < _PRESET_TTL_SECONDS:
        return hit[1]

    preset = _BUILTIN_BASELINE.get(key) or _BUILTIN_BASELINE[DEFAULT_PRESET_NAME]
    if not preset.is_valid():  # defensive: a malformed baseline still degrades safely
        preset = _BUILTIN_BASELINE[DEFAULT_PRESET_NAME]

    _PRESET_CACHE[key] = (time.monotonic(), preset)
    return preset
