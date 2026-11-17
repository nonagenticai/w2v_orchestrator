"""Deterministic music supervisor — the automatic fallback pick when a brief
does not name a track.

Today ``resolve_audio_bed`` (``shared/sound_design.py``) only plays music when
the ``sound_effects`` config explicitly names a ``music`` id — a brief that
opts sound design on but doesn't hand-pick a track ships **silent**. This
module is the deterministic fallback: given a mood/tone hint it picks a
sensible catalog track, table-driven, no model call.

**Why not an LLM.** The library has exactly three music tracks
(``corporate_uplift`` / ``jazz_lounge`` / ``ambient_tech``) and two ambient beds
(``office_hum`` / ``room_tone``) — see ``sound_catalog.BUILTIN_CATALOG``. Asking
a model to choose one of three options is theater; a small mood→id table is
both cheaper and perfectly reproducible. ``06-sound-music.md`` §5 describes a
future WM ``music-supervisor`` that reasons over per-asset mood/energy/tempo
metadata once the catalog carries it — this module is the Phase-0, table-only
predecessor and deliberately does not call ``worker_management``.

**Precedence contract.** This module is the *fallback*, not the whole
decision: the caller (``resolve_audio_bed``) must only invoke :func:`pick_bed`
when the brief did **not** already name a ``music``/``ambient`` id in
``cfg`` — an authored id always wins outright and is never laundered through
this table. ``pick_bed`` itself is unconditional; it does not re-check ``cfg``
for an existing choice.

**Authoritative validation (mirrors ``sound_catalog``'s "advisory model,
authoritative library" contract).** The mood table is advisory: every id it
proposes is checked against the *loaded* :class:`~src.services.stages.sound_catalog.SoundLibrary`
before being returned. An id the table names but the library doesn't carry
(pruned catalog, tenant override, stale table) drops to that category's
default id; if the default is *also* invalid the category resolves to
``None`` rather than ever handing back a broken presign target.

**Offsets.** ``music_offset_lu`` / ``ambient_offset_lu`` are how far (in LU)
the bed sits *below* the measured narration loudness. They ride the compile
payload; the compiler measures the VO bus and solves the bed gain from them
(VO-relative leveling, ``06-sound-music.md`` T1), keeping the absolute
``*_gain_db`` keys only as its measurement-failed fallback. They are resolved
through
:func:`~src.services.stages.shared.provenance.resolve_with_provenance` (same
idiom as :func:`~src.services.stages.shared.compute_plan.mix_gain_targets`) so
they show up in the provenance ledger like every other computed lever, and so
a brief can override them (``sound_effects.music_offset_lu`` /
``sound_effects.ambient_offset_lu``) ahead of the computed default.

``duck_open_lu_below_vo`` rides the same seam but sizes the *sidechain duck
key* rather than a bed gain: it is how far below the measured VO loudness the
music/ambient duck opens, so the duck depth stays constant as narration level
moves instead of collapsing against an absolute threshold. It resolves through
the same :func:`_resolve_offset` helper, lands in the provenance ledger, and a
brief can override it via ``sound_effects.duck_open_lu_below_vo``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from src.services.stages.shared.provenance import ProvenanceValue, resolve_with_provenance
from src.services.stages.sound_catalog import SoundCategory


class _BedLibrary(Protocol):
    """The slice of ``SoundLibrary`` this module needs — structural, so a
    test double satisfies it without importing/constructing the real
    manifest-driven ``SoundLibrary`` (mirrors ``sound_catalog._SoundStorage``)."""

    def is_valid(self, sfx_id: str, *, category: SoundCategory | None = None) -> bool: ...


# Neutral offsets — "sit N LU below the measured narration". These ride the
# compile payload (`music_offset_lu` / `ambient_offset_lu`); the compiler
# measures the VO bus and solves the bed gain from them (spec 06-sound-music
# T1). The absolute `*_gain_db` keys remain as its measurement-failed fallback.
_DEFAULT_MUSIC_OFFSET_LU = 18.0
_DEFAULT_AMBIENT_OFFSET_LU = 30.0

# How far below the MEASURED voiceover loudness the music/ambient sidechain duck
# opens. The compiler solves `dbfs = vo_lufs - open_lu_below_vo` and takes
# `10 ** (dbfs / 20)` as the `sidechaincompress` threshold; without this key it
# falls back to the absolute literal `duck_threshold = 0.03` (= -30.4576 dBFS),
# so duck depth collapses as narration gets quieter.
#
# 12.46 is the exact back-solve of that shipped literal at this bus's VO target:
# every voiceover here is premastered to TARGET_LUFS = -18.0
# (`src/services/audio_premaster.py`), and -18.0 - (-30.4576) = 12.4576 -> 12.46.
#
# That -18.0 premise is only as good as its enforcement point, which is
# `services/stages/primitives/narration.py::ensure_tts_loudness` — the single
# choke point through which every narration clip passes. It normalizes any clip
# tts_backend could not prove the level of, and counts
# `w2v_orchestrator_tts_clip_unnormalized_total` for each one it could not fix.
# A non-zero rate on that counter is precisely the population for which this
# constant's nominal operating point does NOT hold.
# So at the nominal operating point the derived threshold reproduces today's
# literal (12.5 would drift it -0.49%); only OFF-nominal VO changes, which is
# precisely the defect being fixed. Hence no feature flag: robustness at
# nominal, not a product change.
#
# Honest caveat: `vo_lufs` is INTEGRATED loudness while `sidechaincompress` gates
# on INSTANTANEOUS key amplitude, so this constant also absorbs speech crest
# factor. It is legitimate not because it is derived from first principles — it
# is not — but because it is back-solved from the shipped literal at a known,
# code-enforced VO target.
_DEFAULT_DUCK_OPEN_LU_BELOW_VO = 12.46

# The catalog only has 3 music tracks / 2 ambient beds today (see
# sound_catalog.BUILTIN_CATALOG) — a mood table, not a model, is the right
# tool. Defaults are the "safe middle" pick in each category.
_DEFAULT_MUSIC_ID = "corporate_uplift"
_DEFAULT_AMBIENT_ID = "room_tone"

#: mood/tone (lowercased) -> catalog music id. Advisory only — see module
#: docstring; every id is re-validated against the loaded library.
_MOOD_MUSIC_TABLE: dict[str, str] = {
    "upbeat": "corporate_uplift",
    "energetic": "corporate_uplift",
    "confident": "corporate_uplift",
    "punchy": "corporate_uplift",
    "professional": "corporate_uplift",
    "corporate": "corporate_uplift",
    "calm": "jazz_lounge",
    "warm": "jazz_lounge",
    "relaxed": "jazz_lounge",
    "cinematic": "jazz_lounge",
    "playful": "jazz_lounge",
    "technical": "ambient_tech",
    "neutral": "ambient_tech",
    "minimal": "ambient_tech",
}

#: mood/tone (lowercased) -> catalog ambient id. Advisory only, same contract.
_MOOD_AMBIENT_TABLE: dict[str, str] = {
    "technical": "office_hum",
    "corporate": "office_hum",
    "professional": "office_hum",
}


@dataclass(frozen=True)
class BedChoice:
    """The deterministic fallback bed pick, both ids library-validated.

    ``music_id`` / ``ambient_id`` are ``None`` when neither the mood-matched
    nor the default id for that category survives library validation (never a
    broken presign target). ``music_offset_lu`` / ``ambient_offset_lu`` are
    provenance-tracked so the resolved offset — and whether it came from an
    explicit ``cfg`` override or the computed default — is auditable.
    ``duck_open_lu_below_vo`` is tracked the same way; it sizes the sidechain
    duck key VO-relatively instead of against an absolute threshold.
    """

    music_id: str | None
    ambient_id: str | None
    music_offset_lu: ProvenanceValue
    ambient_offset_lu: ProvenanceValue
    duck_open_lu_below_vo: ProvenanceValue


def _resolve_id(
    mood: str | None,
    table: dict[str, str],
    default_id: str,
    library: _BedLibrary,
    *,
    category: SoundCategory,
) -> str | None:
    normalized = (mood or "").strip().lower()
    candidate = table.get(normalized, default_id)
    if library.is_valid(candidate, category=category):
        return candidate
    if candidate != default_id and library.is_valid(default_id, category=category):
        return default_id
    return None


def _resolve_offset(cfg: dict[str, object], key: str, default: float) -> ProvenanceValue:
    """``resolve_with_provenance`` for an offset, honouring an explicit ``cfg``
    override only when the key is actually present (a *missing* key must fall
    through to the computed default, not be mistaken for an explicit
    ``None`` override — ``resolve_with_provenance`` treats a supplied
    ``None`` as a real, winning override)."""
    if key in cfg:
        return resolve_with_provenance(
            key, override=cfg[key], default=default, computed_by="music_supervisor"
        )
    return resolve_with_provenance(key, default=default, computed_by="music_supervisor")


def _omit_if_vo_unproven(resolved: ProvenanceValue, *, vo_level_unproven: bool) -> ProvenanceValue:
    """Null out a COMPUTED (never an explicit override) offset when the run's
    voiceover level is not actually known.

    ``_DEFAULT_MUSIC_OFFSET_LU`` / ``_DEFAULT_DUCK_OPEN_LU_BELOW_VO`` are back-solved
    from an ASSUMED VO bus at ``TARGET_LUFS = -18.0``
    (``audio_premaster.premaster_core`` / ``narration.ensure_tts_loudness``). That
    assumption fails open: a clip whose level ``ensure_tts_loudness`` could not prove
    (``premaster_failed_open`` / ``unmeasurable_level`` — see
    ``degradation.AUDIO_UNNORMALIZED``) still ships, at WHATEVER level the backend
    produced, not -18. Handing the compiler a VO-relative offset computed against a
    premise that clip does not satisfy sizes the duck/gain against a fiction.
    Returning ``value=None`` here (source/provenance otherwise preserved for audit)
    means the CALLER omits the key entirely, so the compiler measures the real VO bus
    instead of trusting the assumed one.

    An explicit ``cfg`` override is NEVER touched — the author asked for a specific
    number and that number is honoured regardless of what any clip's proof status
    is, same precedence rule as every other lever in this module.
    """
    if not vo_level_unproven or resolved.source == "override":
        return resolved
    return replace(resolved, value=None)


def pick_bed(
    cfg: dict[str, object],
    library: _BedLibrary,
    *,
    mood: str | None = None,
    vo_level_unproven: bool = False,
) -> BedChoice:
    """Deterministically pick a fallback music + ambient bed.

    ``library`` is the already-loaded, tenant-scoped ``SoundLibrary`` (the
    caller — ``resolve_audio_bed`` — already awaits ``load_sound_library()``
    for its own bed resolution, so this stays a **sync**, pure function: no
    duplicate I/O, no second library load). ``mood`` is typically
    ``cfg.get("mood")``; the caller decides that mapping.

    ``vo_level_unproven`` — True when ANY narration clip in this run shipped at an
    unproven level (see :func:`_omit_if_vo_unproven`) — nulls out the COMPUTED
    (never an explicit override) ``*_offset_lu`` / ``duck_open_lu_below_vo`` values,
    so the caller can drop those keys and let the compiler measure the real bus
    instead of trusting the assumed -18 LUFS premise they were back-solved from.

    Contract: call this ONLY when the brief hasn't already named a
    ``music``/``ambient`` id in ``cfg`` — see the module docstring. This
    function does not itself check ``cfg`` for an existing choice, so it is
    safe to unit-test in isolation from that precedence rule.
    """
    music_id = _resolve_id(mood, _MOOD_MUSIC_TABLE, _DEFAULT_MUSIC_ID, library, category="music")
    ambient_id = _resolve_id(
        mood, _MOOD_AMBIENT_TABLE, _DEFAULT_AMBIENT_ID, library, category="ambient"
    )
    music_offset = _omit_if_vo_unproven(
        _resolve_offset(cfg, "music_offset_lu", _DEFAULT_MUSIC_OFFSET_LU),
        vo_level_unproven=vo_level_unproven,
    )
    ambient_offset = _omit_if_vo_unproven(
        _resolve_offset(cfg, "ambient_offset_lu", _DEFAULT_AMBIENT_OFFSET_LU),
        vo_level_unproven=vo_level_unproven,
    )
    duck_open = _omit_if_vo_unproven(
        _resolve_offset(cfg, "duck_open_lu_below_vo", _DEFAULT_DUCK_OPEN_LU_BELOW_VO),
        vo_level_unproven=vo_level_unproven,
    )
    return BedChoice(
        music_id=music_id,
        ambient_id=ambient_id,
        music_offset_lu=music_offset,
        ambient_offset_lu=ambient_offset,
        duck_open_lu_below_vo=duck_open,
    )
