"""Shared sound-design helpers for compile stages.

Blueprint-agnostic functions lifted from ``promo_render`` so any compile stage
(promo, showcase/demo_video) resolves the ``sound_effects`` config the same way:

* ``sound_config`` — the run's ``sound_effects`` block (empty when off).
* ``transition_sfx_id`` — the per-cut transition SFX id (pure config decision;
  membership is validated at presign time in ``resolve_shot_sound_urls``).
* ``sfx_by_line`` — the sound-director's per-line accent cues, grouped by
  ``line_id``. Parametrised by the director stage name so each blueprint points
  at its own director output (``promo_sound_director`` /
  ``demo_video_sound_director``).
* ``resolve_audio_bed`` — the continuous music + ambient bed, catalog ids
  resolved to presigned URLs and the ambient cue sized to the final duration.

All are no-ops (empty / ``None``) when sound design is off, so a run that doesn't
opt in is byte-identical to before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.stages.shared.compute_plan import mix_gain_targets
from src.services.stages.shared.degradation import (
    AUDIO_UNNORMALIZED,
    SOUND_BED_MISSING,
    Degradation,
    degradation,
)
from src.services.stages.shared.music_supervisor import BedChoice, pick_bed
from src.services.stages.shared.provenance import ProvenanceValue
from src.services.stages.sound_catalog import (
    DEFAULT_AMBIENT_ID,
    DEFAULT_TRANSITION_ID,
    SoundLibrary,
    load_sound_library,
)
from src.services.tts_storage import presign_sound_asset

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


def sound_config(ctx: RunContext) -> dict[str, Any]:
    """The run's ``sound_effects`` config block (empty when sound design is off)."""
    return ctx.config_snapshot.get("sound_effects") or {}


def transition_sfx_id(cfg: dict[str, Any]) -> str | None:
    """Resolve the per-cut transition SFX id from config.

    ``transitions``: ``false`` disables; ``true``/absent uses the default whoosh;
    an explicit id overrides. Membership is validated at presign time against the
    loaded library (``resolve_shot_sound_urls``) — an id that can't be resolved is
    dropped there, so this stays a pure-config (sync) decision.
    """
    if not cfg.get("enabled"):
        return None
    choice = cfg.get("transitions")
    if choice is False:
        return None
    if choice is True or choice is None:
        return DEFAULT_TRANSITION_ID
    return str(choice)


def sfx_by_line(
    ctx: RunContext, cfg: dict[str, Any], director_stage: str
) -> dict[str, list[dict[str, Any]]]:
    """Group the sound-director's per-line accent cues by ``line_id`` (catalog
    ids; resolved to URLs at compile time). ``director_stage`` is the prior stage
    that produced the cues (``promo_sound_director`` / ``demo_video_sound_director``).
    Empty when sound design is off or the director skipped."""
    if not cfg.get("enabled"):
        return {}
    director = ctx.prior_output(director_stage) or {}
    # A brief-level sfx_gain_db OVERRIDES the director's per-cue gain (the director
    # emits a uniform default, e.g. -10dB), so a brief can trim all accents/
    # transitions relative to narration without a code change. Unset → per-cue gain
    # (or -10dB) as before.
    sfx_override = cfg.get("sfx_gain_db")
    # Centralized default gain seam: neutral → today's -10.0 literal (byte-identical).
    sfx_default = float(mix_gain_targets(cfg.get("vo_lufs"), cfg.get("mood")).sfx.value)
    by_line: dict[str, list[dict[str, Any]]] = {}
    for cue in director.get("cues", []):
        cue_gain = cue.get("gain_db")
        # `is not None`, not truthiness: a valid 0.0 dB accent ("no attenuation")
        # is falsy and would otherwise be overwritten by the -10 dB default.
        gain = (
            float(sfx_override)
            if sfx_override is not None
            else (float(cue_gain) if cue_gain is not None else sfx_default)
        )
        by_line.setdefault(str(cue["line_id"]), []).append(
            {
                "sfx_id": str(cue["sfx_id"]),
                "at_seconds": float(cue.get("at_seconds") or 0.0),
                "gain_db": gain,
            }
        )
    return by_line


def _bed_dropped(
    degradations: list[Degradation] | None, kind: str, asset_id: str, reason: str
) -> None:
    """Record that a CHOSEN bed did not make it into the render.

    The three ``if url:`` / ``if bed:`` guards below are silent drops by
    construction: a music or ambient cue that was selected (by the brief or by
    the music supervisor) and then failed to presign simply does not appear in
    the payload, and the video ships with no bed. There was no log line and no
    field — the only evidence was the absence of a key nobody diffed.

    ``degradations`` is an optional sink so every existing caller and test keeps
    working unchanged; ``None`` means the caller does not collect, not that the
    drop is acceptable.
    """
    if degradations is None:
        return
    degradations.append(
        degradation(
            SOUND_BED_MISSING,
            f"{kind} bed {asset_id!r} was chosen but {reason}; the delivered video "
            f"has no {kind} bed",
        )
    )


def _ambient_id(ambient_choice: object, library: SoundLibrary, fallback: BedChoice) -> str | None:
    """Resolve the ambient catalog id from the three-state ``ambient`` config.

    Extracted verbatim from :func:`resolve_audio_bed` (no behaviour change) to
    keep that function under the C901 ceiling once the degradation sink was
    added. A bare ``True`` is the opt-in default; a real id is honoured when the
    library says the caller can see it; a genuinely ABSENT key falls back to the
    music supervisor's pick; an explicit ``False`` means "no ambient" and is NOT
    treated as absence.
    """
    if ambient_choice is True:
        return DEFAULT_AMBIENT_ID
    # ``resolves`` (not ``is_valid``): the id is pinned in the run's snapshot, so a
    # tenant track deleted after the run started must still play for this run.
    if ambient_choice and library.resolves(str(ambient_choice), category="ambient"):
        return str(ambient_choice)
    if ambient_choice is None:
        return fallback.ambient_id
    return None


def _any_clip_loudness_unproven(ctx: RunContext) -> bool:
    """True when ANY prior stage in this run recorded an :data:`AUDIO_UNNORMALIZED`
    degradation — i.e. at least one narration clip shipped at an unproven level
    (``primitives.narration.ensure_tts_loudness`` failed open: ``premaster_failed_open``
    / ``unmeasurable_level``).

    Blueprint-agnostic by construction: it scans every stage's persisted output
    rather than naming a specific narrate stage, so it works the same for
    demo_video / promo / website_review / action_reel without this shared helper
    needing to know which of them ran. A stage output that is not a mapping, or
    carries no ``degradations`` key, contributes nothing — matches the additive,
    absent-key-is-empty-list contract ``degradation.py`` documents.
    """
    for output in ctx.prior_run_outputs.values():
        if not isinstance(output, dict):
            continue
        degradations = output.get("degradations")
        if not isinstance(degradations, list):
            continue
        for entry in degradations:
            if isinstance(entry, dict) and entry.get("code") == AUDIO_UNNORMALIZED:
                return True
    return False


def _set_offset_if_proven(target: dict[str, object], key: str, resolved: ProvenanceValue) -> None:
    """Set ``target[key]`` from a :class:`ProvenanceValue`, unless
    :func:`music_supervisor._omit_if_vo_unproven` nulled it because the run's VO
    level could not be proven — then the key is left OUT entirely rather than
    sent as ``0.0``/``None``, so the compiler falls back to measuring the real
    VO bus instead of trusting the assumed one."""
    if resolved.value is not None:
        target[key] = float(resolved.value)


async def resolve_audio_bed(
    ctx: RunContext,
    shots: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    degradations: list[Degradation] | None = None,
) -> dict[str, Any] | None:
    """Continuous music + ambient bed from the ``sound_effects`` config. Resolves
    the chosen catalog ids to presigned URLs and sizes the ambient cue to the
    FINAL encoded duration (``shots`` is post title-card + ken-burns). ``None``
    when sound design is off / no bed resolves.

    ``degradations`` is an optional sink — see :func:`_bed_dropped` for why a
    dropped bed had to become recordable."""
    if not cfg.get("enabled"):
        return None
    total = sum(float(shot.get("seconds") or 0.0) for shot in shots)
    bed: dict[str, Any] = {}
    # Centralized default gain seam: an explicit brief gain still WINS via
    # ``cfg.get(<key>, <computed default>)``; neutral computed defaults are exactly
    # today's literals (-18 music / -30 ambient / -12 duck), so a run that doesn't
    # set vo_lufs/mood is byte-identical.
    gains = mix_gain_targets(cfg.get("vo_lufs"), cfg.get("mood"))
    # Tenant-scoped library: a bed id is only honoured if it's a real,
    # category-correct sound the caller can see.
    library = await load_sound_library()

    # ORCH-4.1 T1: an explicit brief-authored id still WINS outright — the
    # supervisor is only consulted when the brief named no track at all, and
    # (being sync + pure) is safe to call unconditionally without a second
    # library load. Reused for both music AND ambient offsets even when the
    # id itself came from the brief, since pick_bed already honours an
    # explicit cfg["music_offset_lu"]/cfg["ambient_offset_lu"] override
    # ahead of its computed default (see music_supervisor._resolve_offset).
    #
    # ``vo_level_unproven``: the VO-relative offsets below are back-solved from an
    # ASSUMED -18 LUFS voiceover bus. When any clip in this run shipped at an
    # unproven level (narration.ensure_tts_loudness failed open), that assumption
    # does not hold for this run — pick_bed nulls out the COMPUTED (never an
    # explicit override) offsets so they are omitted below and the compiler
    # measures the real bus instead.
    fallback: BedChoice = pick_bed(
        cfg,
        library,
        mood=cfg.get("mood"),
        vo_level_unproven=_any_clip_loudness_unproven(ctx),
    )

    # Same three-state contract as `ambient` below: an authored id WINS, an
    # explicit `music: false` means "no music" (now that showcase defaults sound
    # ON, that is the only way to ask for ambience WITHOUT a bed), and only a
    # genuinely absent key falls back to the supervisor's pick.
    music_choice = cfg.get("music")
    # Three-state, mirroring `ambient` below: a bare `True` (opt-in default) OR an
    # absent key both resolve to the supervisor's fallback pick; an explicit
    # `False` means "no music"; an explicit id is honoured as-is. A bare `True`
    # must NOT slip through as the string "True" — `is_valid("True")` fails, so
    # `music: true` shipped SILENCE.
    music_id = fallback.music_id if music_choice is None or music_choice is True else music_choice
    # ``resolves`` below, not ``is_valid``: an authored id is pinned in the run's
    # snapshot, so a tenant track deleted after the run started (RETIRED in the
    # tenant manifest until no run is active) must still play for THIS run. The
    # supervisor's own pick only ever comes from ``ids_in``, which excludes retired.
    if music_choice is not False and not (
        music_id and library.resolves(str(music_id), category="music")
    ):
        # Music was asked for (explicitly or by default) and NOTHING resolved: the
        # brief's id is not a music asset this tenant can see, or the supervisor
        # found no music in the library. This used to fall through to a payload
        # with no `music` key and no trace — a promo with sound design ON shipped
        # without a bed and nothing said so (run 01a10d84, 2026-10-05, where the
        # only evidence was a 0.0004s `audio_bed` phase).
        _bed_dropped(
            degradations,
            "music",
            str(music_id or "(none)"),
            "is not a music asset in this tenant's sound library",
        )
    elif music_id and library.resolves(str(music_id), category="music"):
        url = await presign_sound_asset(str(music_id))
        if not url:
            _bed_dropped(degradations, "music", str(music_id), "could not be presigned")
        if url:
            music_cue: dict[str, object] = {
                "url": url,
                "gain_db": float(cfg.get("music_gain_db", gains.music.value)),
                "fade_in_s": float(cfg.get("fade_in_s", 1.5)),
                "fade_out_s": float(cfg.get("fade_out_s", 2.5)),
                "duck_db": float(cfg.get("duck_db", gains.duck.value)),
            }
            _set_offset_if_proven(music_cue, "music_offset_lu", fallback.music_offset_lu)
            # VO-relative duck key: the compiler opens the sidechain at
            # `vo_lufs - duck_open_lu_below_vo` instead of the absolute
            # `duck_threshold` literal, so duck DEPTH stays put as narration
            # level moves. Must be emitted on the ambient cue below too — a
            # window where music ducks VO-relative but ambient still ducks
            # absolute would be worse than neither.
            _set_offset_if_proven(
                music_cue, "duck_open_lu_below_vo", fallback.duck_open_lu_below_vo
            )
            bed["music"] = music_cue

    ambient_id = _ambient_id(cfg.get("ambient"), library, fallback)
    if ambient_id and total > 0:
        url = await presign_sound_asset(ambient_id)
        if not url:
            _bed_dropped(degradations, "ambient", ambient_id, "could not be presigned")
        if url:
            ambient_cue: dict[str, object] = {
                "url": url,
                "start_seconds": 0.0,
                "end_seconds": total,
                "gain_db": float(cfg.get("ambient_gain_db", gains.ambient.value)),
            }
            # Same VO-unproven omission as `music` above.
            _set_offset_if_proven(ambient_cue, "ambient_offset_lu", fallback.ambient_offset_lu)
            # Same VO-relative duck key as `music` above — the two MUST
            # activate together (see the comment there), including staying
            # OMITTED together when the VO level is unproven.
            _set_offset_if_proven(
                ambient_cue, "duck_open_lu_below_vo", fallback.duck_open_lu_below_vo
            )
            bed["ambient"] = [ambient_cue]
    return bed or None
