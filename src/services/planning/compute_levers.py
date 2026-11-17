"""Pure planning levers — thin composers over the shared compute substrate.

Every function here is deterministic and side-effect-free. They do NOT
re-derive lever math; they compose the primitives already proven in
:mod:`src.services.stages.shared.compute_plan` (word budget, mix targets, clip
binding, distinct-voice assignment) and wrap the results in the provenance
vocabulary from :mod:`src.services.stages.shared.provenance`. ``resolve_plan``
is the only intended caller.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from src.services.planning.presets import Preset
from src.services.stages.audiobook.voice_catalog import VOICE_CATALOG
from src.services.stages.shared.compute_plan import (
    MixGainTargets,
    beat_word_budget,
    bind_clips_to_lines,
    mix_gain_targets,
    pick_distinct_voices,
)
from src.services.stages.shared.provenance import ProvenanceValue, resolve_with_provenance

#: Rough spoken-word density per beat — a beat is ~one visual/idea. Used only to
#: turn a word budget into a beat count; the real per-line budget stays in
#: ``beat_word_budget``.
WORDS_PER_BEAT = 30


def beat_count(duration_s: float, preset: Preset) -> ProvenanceValue:
    """How many beats fit ``duration_s`` seconds at the preset's narration pace.

    ``words = beat_word_budget(duration_s, preset.wpm)`` then
    ``ceil(words / WORDS_PER_BEAT)``, floored at 1 (a run always has at least one
    beat). Smart-auto — ``computed_by='compute_levers'``.
    """
    words = beat_word_budget(duration_s, preset.wpm).value
    beats = max(1, math.ceil(words / WORDS_PER_BEAT))
    return resolve_with_provenance(
        "beat_count",
        smart_auto=beats,
        default=1,
        computed_by="compute_levers",
    )


def build_beats(
    must_show: list[str],
    must_say: list[str],
    n_beats: int,
    *,
    speakers: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Deterministic beat scaffold — one beat per ``must_show`` label.

    The count is: one beat per ``must_show`` label; absent labels, ``n_beats``
    generic beats. ``must_say`` phrases are dealt across the beats' ``text``
    VERBATIM (one per beat in order; any overflow phrases are appended to the
    last beat so nothing authored is dropped). Each beat gets the sacred spine id
    ``clip-<i+1>`` and, when ``speakers`` is given, a round-robin ``speaker``.

    No narration is invented beyond ``must_show`` / ``must_say``. Returns ``[]``
    when neither labels, phrases, nor ``n_beats > 0`` yield any content.
    """
    labels = list(must_show or [])
    phrases = list(must_say or [])
    speaker_pool = list(speakers or [])

    if labels:
        count = len(labels)
    elif n_beats > 0:
        count = n_beats
    else:
        count = 0
    if count == 0:
        return []

    beats: list[dict[str, Any]] = []
    for i in range(count):
        text = phrases[i] if i < len(phrases) else ""
        speaker = speaker_pool[i % len(speaker_pool)] if speaker_pool else None
        beats.append({"id": f"clip-{i + 1}", "text": text, "speaker": speaker})

    # Overflow phrases (more must_say than beats) appended verbatim to the last
    # beat — never silently dropped.
    if len(phrases) > count:
        leftover = " ".join(phrases[count:])
        tail = beats[-1]["text"]
        beats[-1]["text"] = f"{tail} {leftover}".strip()

    return beats


def mix_gains(preset: Preset, mood: str | None) -> MixGainTargets:
    """The music / ambient / duck / sfx bed gain targets for this preset + mood.

    Delegates straight to ``compute_plan.mix_gain_targets`` — a neutral mood
    (``None`` → ``preset.mood_default``, or an explicit ``"neutral"``) pins the
    byte-identical literals with source ``"default"``.
    """
    return mix_gain_targets(vo_lufs=preset.vo_lufs, mood=mood or preset.mood_default)


def cast_assignment(speakers: Sequence[str], pins: Mapping[str, str] | None) -> dict[str, str]:
    """Assign each speaker a DISTINCT kokoro voice from the authoritative catalog.

    ``VOICE_CATALOG`` is a frozenset (unordered) → ``sorted`` for a deterministic
    catalog order before delegating to ``pick_distinct_voices``. ``pins``
    speaker→voice entries win outright.
    """
    catalog = sorted(VOICE_CATALOG)
    return pick_distinct_voices(speakers, catalog, pinned=pins)


def clip_bindings(lines: Sequence[Mapping[str, Any]], must_show: Iterable[str]) -> dict[str, str]:
    """Bind spine lines to available footage-clip keys (``{line_id: clip_key}``)."""
    return bind_clips_to_lines(lines, must_show)


def checkpoint_set(preset: Preset, mode: str) -> list[str]:
    """Human-review checkpoints for a checkpoint mode.

    ``preview`` / ``careful`` gate before the (expensive, irreversible) demo
    video generation; ``autopilot`` gates nothing. ``mode`` overrides
    ``preset.checkpoint_mode`` when non-empty.
    """
    effective = mode or preset.checkpoint_mode
    mapping: dict[str, list[str]] = {
        "preview": ["demo_video_gen"],
        "autopilot": [],
        "careful": ["demo_video_gen"],
    }
    return mapping.get(effective, [])
