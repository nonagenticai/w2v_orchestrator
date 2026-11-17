"""Compute-everything planning substrate — pure planning heuristics.

The knobs a compile/narrate stage needs (per-line word budgets, the music /
ambient / duck / sfx mix targets, the line↔footage-clip binding, distinct voice
assignment) are today either inlined as literals at the point of use or derived
ad-hoc inside a stage handler. This module lifts those decisions into small,
**pure** free functions so they can be computed once, up front, and audited: no
DB, no HTTP client, no ``RunContext``. It mirrors the frozen-dataclass +
free-function style of :mod:`provenance` and :mod:`loop_engine`.

Every resolved knob is wrapped in a :class:`ProvenanceValue` (via
``resolve_with_provenance`` / ``clamp`` from :mod:`provenance`) so the UI/audit
can explain where each value came from — a pinned literal default, a computed
smart-auto value, or a clamped one.

Byte-identical guarantee: the mix targets for a neutral run (no ``vo_lufs`` /
no mood) resolve to exactly today's literals (music -18, ambient -30, duck -12,
sfx -10 dB), so re-sourcing ``sound_design``'s defaults through
:func:`mix_gain_targets` changes nothing on a run that doesn't opt in.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.models.showcase_line import normalize_clip_key
from src.services.stages.shared.provenance import ProvenanceValue, resolve_with_provenance
from src.services.stages.shared.time_budget import NARRATION_BUDGET_RATE

# Neutral mix targets — today's literals in ``sound_design.resolve_audio_bed`` /
# ``sound_director._DEFAULT_ACCENT_GAIN_DB``. A neutral run pins to these exactly.
_NEUTRAL_MUSIC_DB = -18.0
_NEUTRAL_AMBIENT_DB = -30.0
_NEUTRAL_DUCK_DB = -12.0
_NEUTRAL_SFX_DB = -10.0

# Reference voiceover loudness the neutral literals are tuned for. A louder VO
# (a *higher* LUFS value, e.g. -12 > -16) pushes every bed target deeper so the
# narration keeps the foreground; a quieter VO relaxes them. Monotone in vo_lufs.
_REF_VO_LUFS = -16.0
_DUCK_SLOPE = 0.5


@dataclass(frozen=True)
class MixGainTargets:
    """The four bed gain targets (dB), each provenance-tracked.

    ``music`` / ``ambient`` are the continuous-bed gains; ``duck`` is how far the
    music dips under narration; ``sfx`` is the default per-accent cue gain.
    """

    music: ProvenanceValue
    ambient: ProvenanceValue
    duck: ProvenanceValue
    sfx: ProvenanceValue


def beat_word_budget(duration_s: float, wpm: float) -> ProvenanceValue:
    """Total spoken-word budget for a beat of ``duration_s`` seconds at ``wpm``.

    ``duration_s / 60 * wpm * NARRATION_BUDGET_RATE`` — the number of words a narrator
    speaks in the window. A computed (smart-auto) value;
    ``computed_by='beat_word_budget'``.

    ⭐ WAVE 14, N5 — WHY THE FACTOR IS NAMED HERE EVEN THOUGH IT IS 1.0. This is the
    whole-fleet word budget: ``lesson``, ``promo``, ``website_review`` and ``app_review``
    each wrap it as their ``word_budget``, while ``demo_video`` budgets through
    ``gen.narration_words_target``, which multiplies by
    :data:`~src.services.stages.shared.time_budget.NARRATION_BUDGET_RATE` explicitly.
    Two arithmetics that must agree, and until now only one of them named the constant —
    so wave 13's fix (budget at the natural rate, not at the drag floor) was
    blueprint-local by accident rather than by decision, and the next change to that
    constant would have moved demo_video and left the other four behind.

    ⚠️ CORRECTION to the wave-14 brief, which said these four "still order the drag".
    They do not: an omitted factor is an IMPLICIT 1.0, which is exactly what wave 13
    settled on. Measured at 150 wpm, all six budgets already agree to the word
    (30s -> 75, 60s -> 150, 120s -> 300, 137.5s -> 344). The defect was DRIFT RISK, not
    a live drag, and naming the constant is what closes it.
    """
    budget = duration_s / 60.0 * wpm * NARRATION_BUDGET_RATE
    return resolve_with_provenance(
        "beat_word_budget",
        smart_auto=budget,
        default=0.0,
        computed_by="beat_word_budget",
    )


def mix_gain_targets(
    vo_lufs: float | None = None,
    mood: str | None = None,
) -> MixGainTargets:
    """Music / ambient / duck / sfx bed gain targets (dB).

    For a **neutral** run — ``vo_lufs is None`` or ``mood`` is ``None`` /
    ``"neutral"`` — the targets are pinned to today's literals exactly
    (music -18, ambient -30, duck -12, sfx -10) with source ``"default"``, so
    re-sourcing ``sound_design``'s defaults through this seam is byte-identical.

    Otherwise the targets shift with ``vo_lufs``: a louder voiceover (higher
    LUFS) deepens every target (source ``"smart-auto"``) so the narration stays
    in front. Monotone non-increasing in ``vo_lufs``.
    """
    if vo_lufs is None or mood is None or str(mood).strip().lower() == "neutral":
        return MixGainTargets(
            music=resolve_with_provenance(
                "music_gain_db", default=_NEUTRAL_MUSIC_DB, computed_by="mix_gain_targets"
            ),
            ambient=resolve_with_provenance(
                "ambient_gain_db", default=_NEUTRAL_AMBIENT_DB, computed_by="mix_gain_targets"
            ),
            duck=resolve_with_provenance(
                "duck_db", default=_NEUTRAL_DUCK_DB, computed_by="mix_gain_targets"
            ),
            sfx=resolve_with_provenance(
                "sfx_gain_db", default=_NEUTRAL_SFX_DB, computed_by="mix_gain_targets"
            ),
        )

    # Louder VO → positive shift → every target subtracts it → deeper (more
    # negative), which is monotone non-increasing as vo_lufs rises.
    shift = _DUCK_SLOPE * (vo_lufs - _REF_VO_LUFS)
    return MixGainTargets(
        music=resolve_with_provenance(
            "music_gain_db",
            smart_auto=_NEUTRAL_MUSIC_DB - shift,
            default=_NEUTRAL_MUSIC_DB,
            computed_by="mix_gain_targets",
        ),
        ambient=resolve_with_provenance(
            "ambient_gain_db",
            smart_auto=_NEUTRAL_AMBIENT_DB - shift,
            default=_NEUTRAL_AMBIENT_DB,
            computed_by="mix_gain_targets",
        ),
        duck=resolve_with_provenance(
            "duck_db",
            smart_auto=_NEUTRAL_DUCK_DB - shift,
            default=_NEUTRAL_DUCK_DB,
            computed_by="mix_gain_targets",
        ),
        sfx=resolve_with_provenance(
            "sfx_gain_db",
            smart_auto=_NEUTRAL_SFX_DB - shift,
            default=_NEUTRAL_SFX_DB,
            computed_by="mix_gain_targets",
        ),
    )


def transition_gain_db(cfg: Mapping[str, Any]) -> float:
    """Gain (dB) for the transition cue that rides every cut but the open.

    Mirrors the accent rule in ``sound_design.sfx_by_line`` exactly: a brief-level
    ``sfx_gain_db`` OVERRIDES the computed default, so a brief can trim the
    transition whooshes relative to narration without a code change. Unset falls
    back to :func:`mix_gain_targets`' ``sfx`` target (neutral → -10 dB), the same
    default the accents get.

    ``is not None``, not truthiness: a valid 0.0 dB transition ("no attenuation")
    is falsy and would otherwise be discarded in favour of the -10 dB default.
    """
    override = cfg.get("sfx_gain_db")
    if override is not None:
        return float(override)
    return float(mix_gain_targets(cfg.get("vo_lufs"), cfg.get("mood")).sfx.value)


def bind_clips_to_lines(
    lines: Sequence[Mapping[str, Any]],
    must_show: Iterable[str],
) -> dict[str, str]:
    """Bind each spine line to an available footage-clip key, deterministically.

    ``lines`` is the ordered spine; each line joins on its authored ``clip`` tag
    (``@clip-<n>``) or, absent that, its own ``id`` — normalized to the
    frontend_test_runner ``clip-<n>`` form via
    :func:`~src.models.showcase_line.normalize_clip_key`, so a hand-authored
    ``@clip-1`` still matches recorded ``clip-1`` footage (the fuzzy join).
    ``must_show`` is the set of available clip keys (e.g. the recorded footage
    tags, or a mapping keyed by them — a mapping iterates its keys).

    Returns ``{line_id: clip_key}`` for every line whose key matches an available
    clip, built in spine order so the spine ids are preserved and never reordered
    (the C0 invariant). A line whose key matches nothing is simply omitted.
    """
    available = {normalize_clip_key(str(key)) for key in must_show}
    bound: dict[str, str] = {}
    for line in lines:
        line_id = str(line["id"])
        raw_key = line.get("clip") or line["id"]
        clip_key = normalize_clip_key(str(raw_key))
        if clip_key in available:
            bound[line_id] = clip_key
    return bound


def pick_distinct_voices(
    speakers: Sequence[str],
    catalog: Sequence[str],
    *,
    pinned: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Assign each speaker a DISTINCT catalog voice, deterministically.

    Speakers are processed in order; each takes the first ``catalog`` voice not
    already used. When the catalog is exhausted (more speakers than voices) a
    speaker falls back to a stable per-name pick so the same name always maps to
    the same voice regardless of position — mirroring
    ``audiobook.cast._distinct_voice``. ``pinned`` speaker→voice entries win
    outright (and reserve their voice). Pure + deterministic given the inputs.
    """
    pins = dict(pinned or {})
    used: set[str] = set(pins.values())
    assignment: dict[str, str] = {}
    for speaker in speakers:
        if speaker in pins:
            assignment[speaker] = pins[speaker]
            continue
        chosen: str | None = next((vid for vid in catalog if vid not in used), None)
        if chosen is None and catalog:
            # Catalog exhausted: stable per-name pick (same name → same voice).
            idx = int.from_bytes(speaker.encode("utf-8"), "big") % len(catalog)
            chosen = catalog[idx]
        if chosen is not None:
            assignment[speaker] = chosen
            used.add(chosen)
    return assignment
