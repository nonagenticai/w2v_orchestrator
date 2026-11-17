"""The per-run music override block — one reader for the two stages that act on it.

A Studio "change the music" edit on an action_reel run rides the child run's
``config_snapshot["music"]`` block (written by
``api/routes/runs/lifecycle._reel_music_overrides``, which MERGES it onto the
parent run's block so an omitted knob keeps the run's current value). It is deliberately NOT
the ``sound_effects`` block every other blueprint's bed comes from: nothing in
``stages/action_reel/`` reads ``sound_effects`` — ``ActionReelCompileStage``
overrides ``build_audio_bed`` and never reaches the shared ``sound_design``
resolver — so a music edit written there is a silent no-op on this product.

Two stages consume the block, for two different halves of the edit:

* ``music_select`` reads the CHOICE (``track_id`` / ``enabled``). It has to be
  this stage and not ``compile``: the cut is timed to the bed. ``cut_plan``
  snaps every shot boundary to the selected track's beat grid, so swapping the
  bed at compile time would render a reel cut to a DIFFERENT track's grid —
  ``beat_locked`` per shot, and wrong. The edit therefore rewinds the child run
  to ``music_select`` (see ``runs/elements.music_rewind_stage``) so the new
  track's grid is fetched and the cut re-derived from it.
* ``compile`` reads the MIX knobs (``gain_db`` / ``fade_in_s`` / ``fade_out_s``),
  which are pure render-time values and change nothing about the timing.

``duck_db`` is accepted by the API for every deliverable type but cannot be
honoured on this one — see ``compile._MUSIC_DUCK_DB``, whose duck is keyed off
the per-shot narration bus while this product's voice-over rides the
request-level bed bus. It is surfaced as an explicit ignored-field note rather
than dropped, because a knob that reads as applied and is not is the same class
of lie as the no-op this module exists to fix.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Config-snapshot key the override rides on. Top level, not under ``brief``:
#: ``build_config_snapshot`` spreads ``config_overrides`` over the snapshot, so a
#: top-level key needs no merge semantics, whereas a ``brief`` override would
#: REPLACE the whole brief dict.
MUSIC_OVERRIDE_KEY = "music"

#: Mix knobs this product cannot honour, in the order they are reported.
IGNORED_MIX_FIELDS = ("duck_db",)


class MusicOverride(BaseModel):
    """The run's music-override block, parsed.

    A MODEL rather than the untyped mapping this started as. The block has a
    known shape — it is written by ``lifecycle._reel_music_overrides`` from a
    ``MusicEdit``, whose fields these mirror — so an untyped dict bought nothing
    but three call sites doing their own ``.get()`` guessing, and it tripped
    ``tests/test_any_ratchet.py``. The bounds are NOT repeated here: ``MusicEdit``
    is the validating boundary (``ge=-60, le=12`` on ``gain_db``, mirroring
    ``ClipAudio``), and re-declaring them would put two sources of truth on the
    same number.

    ``extra="ignore"`` and every field optional, because the SNAPSHOT is not a
    validated boundary: ``config_overrides`` reaches it unvalidated, and a stage
    must read a surprising block as "no opinion" rather than raise mid-run.
    """

    model_config = ConfigDict(extra="ignore")

    track_id: str | None = None
    enabled: bool | None = None
    gain_db: float | None = None
    fade_in_s: float | None = None
    fade_out_s: float | None = None
    duck_db: float | None = None

    @property
    def is_off(self) -> bool:
        """True when the override explicitly turns the bed OFF.

        Only an explicit ``enabled: False`` counts. An absent ``enabled`` is "the
        edit did not speak about on/off" — which, for an override carrying a
        ``track_id``, is on.
        """
        return self.enabled is False

    def ignored_mix_fields(self) -> list[str]:
        """The authored mix knobs this product cannot apply (for the run note)."""
        return [f for f in IGNORED_MIX_FIELDS if getattr(self, f, None) is not None]

    def mix_knob(self, key: str, default: float) -> float:
        """One mix knob, falling back to *default* when unauthored OR unusable.

        The non-finite guard is LOAD-BEARING and moved here from ``compile``'s
        ``_mix_knob``: pydantic accepts ``nan``/``inf`` for a ``float`` field by
        default, and either one interpolated into the compiler's filtergraph 422s
        the whole render. ``config_overrides`` reaches the snapshot unvalidated,
        so this is the last place to catch it. Unusable reads as "not authored".
        """
        authored = getattr(self, key, None)
        if authored is None:
            return default
        number = float(authored)
        return number if math.isfinite(number) else default


def read_music_override(config_snapshot: Mapping[str, Any]) -> MusicOverride:
    """The run's music-override block, or an empty one when there is none.

    Defensive about shape: ``config_overrides`` is unvalidated at the engine
    boundary, so a non-dict value — or one whose fields are the wrong type —
    reads as "no override" rather than raising inside a stage.
    """
    raw = config_snapshot.get(MUSIC_OVERRIDE_KEY)
    if not isinstance(raw, dict):
        return MusicOverride()
    try:
        return MusicOverride.model_validate(raw)
    except ValidationError:
        pass
    # PER-FIELD, not all-or-nothing, and that distinction is a real behaviour:
    # whole-block rejection would throw away a VALID `track_id` because the same
    # edit carried one junk mix knob, silently reverting the operator's actual
    # choice — a worse lie than the dropped knob. Each field is re-validated
    # alone and only the unusable ones fall out. Still never raises: a malformed
    # block is an absent opinion, not a mid-run crash on a paid stage.
    # `object`, not `Any`: these values go straight back into
    # `model_validate`, which does its own narrowing, so a permissive alias
    # here would buy nothing and cost a notch on `test_dict_str_any_ratchet`.
    usable: dict[str, object] = {}
    for field in MusicOverride.model_fields:
        if field not in raw:
            continue
        try:
            MusicOverride.model_validate({field: raw[field]})
        except ValidationError:
            continue
        usable[field] = raw[field]
    return MusicOverride.model_validate(usable)
