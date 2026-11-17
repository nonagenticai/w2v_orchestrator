"""Resolve per-shot sound-effect catalog ids to presigned URLs at compile time.

The compositor shots carry sound cues as catalog **ids** through planning
(``assemble_shots`` writes ``shot["sfx"] = [{sfx_id, at_seconds, gain_db}]`` and
``shot["transition_sfx_id"]``). This twin of ``shot_presign.refresh_shot_asset_urls``
runs in ``CompileStageBase.handle`` right before the render and rewrites those
ids into the wire shape the compiler expects:

* ``shot["sfx"]``          → ``[{url, at_seconds, gain_db}]`` (``AccentCue``)
* ``shot["transition_sfx"]`` → ``{url, gain_db}`` (``SoundCue``). The gain comes
  from the planning-only ``shot["transition_sfx_gain_db"]`` key (a brief's
  ``sfx_gain_db``, else the computed mix target); absent ⇒ the neutral -10 dB
  default, the same level the accent cues get.

Resolving at compile time (rather than storing URLs in a prior stage output)
sidesteps presign expiry entirely — the URL is signed seconds before the
compiler fetches it. A cue whose id can't be presigned (unknown id, backend
can't sign, library not yet seeded) is dropped; the keys are removed when no
cue survives so the compiler sees no empty/dead sound fields.
"""

from __future__ import annotations

import logging
from typing import Any

from src.services.stages.shared.compute_plan import _NEUTRAL_SFX_DB
from src.services.stages.shared.degradation import SFX_CUE_DROPPED, Degradation, degradation
from src.services.tts_storage import presign_sound_asset

logger = logging.getLogger(__name__)


async def resolve_shot_sound_urls(
    shots: list[dict[str, Any]],
    *,
    notes: list[str] | None = None,
    degradations: list[Degradation] | None = None,
) -> list[dict[str, Any]]:
    """Rewrite each shot's ``sfx`` id-cues + ``transition_sfx_id`` into presigned
    URL cues in place; return *shots*. Best-effort: an unresolvable id is dropped,
    never failing a compile that would otherwise proceed.

    ``notes`` is the OPTIONAL degrade channel, the same ``list[str]`` sink
    ``shared/title_cards.build_title_card_shots`` uses: a caller that owns a stage
    output with a ``notes`` field passes its list in, and each DROPPED cue appends
    one sentence naming the sfx id and the reason (once per id — the cache dedupes).
    Behaviour is unchanged, the cue is still dropped and the compile still runs, and
    ``None`` (no caller sink) is a no-op.

    Without it the drop was recorded NOWHERE durable: a sound-library or MinIO blip
    shipped a film with its SFX bed silently missing, and the ``log.warning``
    evaporates with the pod. Lower severity than the mastering gate (audio garnish,
    not the master) but the same fail-open shape: an absent measurement — here an
    absent ASSET — must not be indistinguishable from a clean resolve.

    ``degradations`` is the STRUCTURED twin of ``notes``: each dropped cue also
    appends one :data:`SFX_CUE_DROPPED` entry whose ``detail`` is that sentence.
    """
    cache: dict[str, str | None] = {}

    async def url_for(sfx_id: str) -> str | None:
        if sfx_id not in cache:
            reason: str | None = None
            try:
                cache[sfx_id] = await presign_sound_asset(sfx_id)
            except Exception as exc:  # noqa: BLE001 -- a sign failure drops the cue, never fails compile
                cache[sfx_id] = None
                # The exception CLASS is load-bearing: it separates a MinIO/storage
                # outage from a bad id, and the run detail is the only place anyone
                # reads it after the pod rolls.
                reason = (
                    f"shot_sound: presign failed for sfx {sfx_id!r}, cue dropped: "
                    f"{type(exc).__name__}: {exc}"
                )
            if reason is None and cache[sfx_id] is None:
                reason = (
                    f"shot_sound: no sound-library URL for sfx {sfx_id!r} "
                    "(unknown id or library not seeded), cue dropped"
                )
            if reason is not None:
                logger.warning("%s", reason)
                if notes is not None:
                    notes.append(reason)
                if degradations is not None:
                    degradations.append(degradation(SFX_CUE_DROPPED, reason))
        return cache[sfx_id]

    for shot in shots:
        await _resolve_one_shot(shot, url_for)
    return shots


async def _resolve_one_shot(shot: dict[str, Any], url_for: Any) -> None:
    """Resolve a single shot's accent + transition cue ids in place."""
    raw_cues = shot.pop("sfx", None) or []
    resolved: list[dict[str, Any]] = []
    for cue in raw_cues:
        url = await url_for(str(cue.get("sfx_id") or ""))
        if url:
            resolved.append(
                {
                    "url": url,
                    "at_seconds": float(cue.get("at_seconds") or 0.0),
                    "gain_db": float(cue.get("gain_db") or 0.0),
                }
            )
    if resolved:
        shot["sfx"] = resolved
    transition_id = shot.pop("transition_sfx_id", None)
    # Popped UNCONDITIONALLY (before the id check) so the internal planning key
    # never leaks onto the wire, even when the cue is dropped for want of a URL.
    transition_gain = shot.pop("transition_sfx_gain_db", None)
    if transition_id:
        url = await url_for(str(transition_id))
        if url:
            shot["transition_sfx"] = {
                "url": url,
                "gain_db": float(transition_gain)
                if transition_gain is not None
                else _NEUTRAL_SFX_DB,
            }
