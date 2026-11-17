"""compile — render the planned cut through the canonical compile path.

Nothing bespoke: this subclasses the shared ``CompileStageBase``, which owns the
whole invariant tail (empty-shot guard, title cards, Ken Burns, presign refresh,
mastering/finish blocks, the non-idempotent ``retries=0`` submit with a
resume-stable idempotency key, artifact validation, duration auto-QA). Two seams
are action_reel's own, and both are seams the base deliberately leaves open: the
shot JOIN (below) and the master CADENCE (``resolve_cadence`` — the reel's frame
rate is a property of the footage, not of the run config).

The join, per ``cut_plan`` shot:

* ``video_url`` — the source original (the compositor honours ``video_url`` over
  ``image_url``);
* ``clip_lead_seconds`` / ``clip_out_seconds`` — the segment's IN and OUT points.
  BOTH are load-bearing and both are always emitted. The lead alone is not a cut:
  without an out-point the compositor plays the source from the lead to its own
  end, so a 4K drone original contributes almost its whole length and no cut
  lands anywhere. They are the same keys the cold-open reframe uses, so there is
  no new compiler contract here. The compiler's ``Shot`` validator rejects
  ``clip_out_seconds <= clip_lead_seconds``, and ``shots`` is a ``list[Shot]`` —
  ONE bad window 422s the WHOLE render — so a plan entry that cannot describe an
  ordered window is dropped here rather than sent (see ``_source_window``);
* ``seconds`` — the shot's on-screen length;
* ``speed`` — the per-clip time-warp. Presence-is-opt-in in the compositor, and
  emitted only when it is not 1.0 so the current (unsolved) plan sends a
  byte-identical payload to a no-warp render;
* ``audio`` — always ``{"mute": True}``. The sources are drone originals: motor
  whine and wind noise under a scored reel is not "production sound", it is a
  defect, and there is no shot in this product whose source audio is wanted;
* ``overlay_clips`` — the alpha-channel graphics ``graphics_design`` designed for
  THIS shot, joined on ``card.shot_id``. Times ride through UNCHANGED: a card's
  ``at_seconds``/``seconds`` are already shot-local source-timeline values (the
  designer converted them, where the shot's speed was known), which is exactly
  what ``OverlayClip.retime_with_speed=True`` reads. This stage does no timeline
  arithmetic — computing an absolute reel time here would place every card at the
  wrong moment the first time a reviewer reorders a shot.

Note what is NOT in that list: ``audio_url``. Narration on this product is a
request-level VOICE TRACK, never a per-shot mux — see ``_build_voice_track``.

The music bed rides as the request-level continuous ``music`` block via
``build_audio_bed`` (the same seam ``promo_render`` uses), NOT as a per-shot
sound: the bed is one continuous track the whole reel is cut against, and
attaching it per shot would restart it on every cut. The voice track rides the
same request-level seam, for a harder reason (below).

``graphics_design`` is a declared dependency and now a live producer: the edge
that was built empty carries real cards. A run whose designer emitted nothing
still renders — a reel with no graphics is a degraded reel, not a failed one —
and ``graphics_design`` has already written the note that says so.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, ClassVar, cast

from src.services.stages.action_reel.cadence import resolve_master_fps
from src.services.stages.action_reel.music_override import (
    MusicOverride,
    read_music_override,
)
from src.services.stages.base import (
    ArtifactUrl,
    Degradation,
    StageInput,
    StageOutputWithNotes,
)
from src.services.stages.primitives.compile_stage import (
    CompileStageBase,
    output_dimension_fields,
    playable_duration_seconds,
)
from src.services.stages.shared.duration_conformance import (
    DurationConformance,
    action_reel_ordered_seconds,
    duration_conformance_advisories,
    duration_conformance_degradations,
    observe_duration_conformance,
    picture_truncation_degradations,
)
from src.services.stages.shot_presign import refresh_shot_asset_urls
from src.services.tts_storage import presign_sound_asset

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# Music bed level under the footage. Well below unity because the bed is the
# floor of the mix, not the subject — and low enough that an opt-in voice-over
# sits on top of it without needing a ducking pass this skeleton does not have.
_MUSIC_GAIN_DB = -14.0
# How far the bed ducks under an opt-in voice-over. Inert on the normal
# (narration-free) reel — there is nothing to duck under.
#
# ⚠ It is ALSO inert on a narrated reel, and that is a deliberate, documented
# trade: the compiler's duck is a `sidechaincompress` keyed off the PER-SHOT
# narration bus (`renderer.overlay_audio_bed`: `[nar_key]`), and this product
# puts its voice-over on the request-level bed bus instead (see
# `_build_voice_track`), which is the side being ducked, not the key. The knob is
# still sent because it is free, harmless, and correct the day the compiler grows
# a keyable voice track; the separation the mix actually relies on is the ~14 dB
# between `_MUSIC_GAIN_DB` and `_VOICE_GAIN_DB`.
_MUSIC_DUCK_DB = -12.0
# Voice-over level. Full scale: the VO is the subject when it is present, the bed
# is already 14 dB down, and the compiler's loudness master normalises the sum.
_VOICE_GAIN_DB = 0.0


def _as_float(value: object) -> float | None:
    """A usable finite number, or ``None`` for everything else.

    ``None`` rather than ``0.0`` on purpose: this feeds the cut-window guard,
    where "unparseable" and "zero" must not collapse into the same answer — a
    junk out-point that reads as 0.0 would silently become an inverted window
    the compiler 422s on.
    """
    try:
        number = float(value)  # ty: ignore[invalid-argument-type]
    except TypeError, ValueError:
        return None
    return number if math.isfinite(number) else None


def _source_window(planned: dict[str, object]) -> tuple[float, float] | None:
    """The plan entry's SOURCE trim as ``(in_seconds, out_seconds)``.

    ``None`` when the entry does not describe a renderable window: a missing or
    non-finite bound, a negative in-point, or ``out <= in``. The caller DROPS
    such a shot. That is the deliberate choice over repairing it: the compiler
    validates ``clip_out_seconds > clip_lead_seconds`` per ``Shot`` and takes
    ``shots`` as a ``list[Shot]``, so one bad window 422s the ENTIRE render —
    and a repair here would have to invent an out-point, i.e. guess which
    seconds of somebody's footage to show. ``cut_plan`` already guarantees
    ``speed == (out - in) / seconds``, so a violated window means THAT entry is
    broken, not that a boundary is missing. Dropping it matches how a shot with
    no fetchable ``video_url`` is already handled, and the caller writes a note
    so the reviewer sees the shot that went missing.
    """
    start = _as_float(planned.get("in_seconds"))
    end = _as_float(planned.get("out_seconds"))
    if start is None or end is None or start < 0 or end <= start:
        return None
    return start, end


#: The compiler caps ``Shot.overlay_clips`` at two entries (`schemas._MAX_OVERLAY_CLIPS`,
#: enforced as a `max_length`), so a third card on one shot would 422 the WHOLE
#: render. Mirrored here as a TRUNCATION rather than left to the compiler: the
#: designer's over-enthusiasm must cost a missing card, not a dead reel.
_MAX_OVERLAY_CLIPS_PER_SHOT = 2

#: The compiler's own charset/length bound on ``OverlayClip.x``/``y``
#: (`_SAFE_OVERLAY_EXPR_PATTERN`, `max_length=64`), mirrored because the value is
#: INTERPOLATED INTO AN FFMPEG FILTERGRAPH: a ':' or a quote from an authored
#: card would 422 the entire render, and the whole point of this module's guards
#: is that one bad entry never costs the reel. An expression that fails it falls
#: back to the field default rather than being sent.
_SAFE_OVERLAY_EXPR = re.compile(r"^[A-Za-z0-9_ .()+*/-]{0,64}$")


def _placement_expr(value: object, default: str) -> str:
    """An ``x``/``y`` expression the compiler will accept, or *default*."""
    text = str(value or "")
    return text if text and _SAFE_OVERLAY_EXPR.match(text) else default


def _overlay_clip(card: dict[str, object]) -> dict[str, object]:
    """One ``graphics_design`` card projected onto an ``OverlayClip`` payload.

    ``start_seconds``/``duration_seconds`` are the card's own shot-local source
    times, forwarded verbatim (see the module docstring). ``retime_with_speed``
    is sent EXPLICITLY even though True is the compiler's default: it is the
    field that decides which timeline those two numbers are read on, and leaving
    the reader of this payload to look it up in another repo's schema is how a
    warped shot silently gains a mistimed card.
    """
    # `_as_float` (not a bare float()) for the same reason the cut window uses it:
    # an unparseable value must not collapse into a plausible number the compiler
    # would happily accept. Here it falls back to the field's own documented
    # default instead of being dropped — a mistyped opacity is a cosmetic defect,
    # not an unrenderable shot.
    start = _as_float(card.get("at_seconds"))
    opacity = _as_float(card.get("opacity"))
    clip: dict[str, object] = {
        "url": str(card.get("url") or ""),
        "start_seconds": start if start is not None and start >= 0 else 0.0,
        "x": _placement_expr(card.get("x"), "0"),
        "y": _placement_expr(card.get("y"), "0"),
        # The card is rendered at the full canvas size with its own safe-area
        # padding baked in, so it is composited 1:1 — "fit" scales it to the
        # canvas, which for a same-size overlay is the identity.
        "scale": "fit",
        "opacity": opacity if opacity is not None else 1.0,
        "retime_with_speed": True,
    }
    seconds = _as_float(card.get("seconds"))
    if seconds is not None and seconds > 0:
        # Omitted (⇒ the clip's own natural length) rather than sent as 0, which
        # the compiler rejects (`gt=0`) and would 422 the whole render.
        clip["duration_seconds"] = seconds
    return clip


class ActionReelCompileInput(StageInput):
    pass


class ActionReelCompileOutput(StageOutputWithNotes):
    """The delivered reel. ``notes`` is the shared soft-degradation channel the
    base's duration auto-QA writes into (a short/truncated render flags the run
    ``degraded`` without failing it)."""

    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. `produces_required` means the
    # stage FAILS instead of degrading. Degrade was the wrong default here: it let
    # the run finish while handing the user a reference to nothing, which is the
    # exact defect class this contract exists to close.
    produces: ClassVar[tuple[str, ...]] = ("artifact_url",)
    produces_required: ClassVar[bool] = True

    artifact_url: ArtifactUrl
    duration_seconds: float
    width: int | None = None
    height: int | None = None
    #: What actually happened to the music bed, as a MACHINE-READABLE fact rather
    #: than a sentence in ``notes``. One of:
    #:
    #: * ``"present"`` — a live presigned bed rode out on the request;
    #: * ``"unpresignable"`` — a track WAS planned but its URL could not be
    #:   minted, so the delivered reel is silent;
    #: * ``"not_selected"`` — no track was ever planned (``music_select`` found
    #:   nothing). Not a degradation of this stage; ``cut_plan`` noted it.
    #:
    #: The distinction matters because the failure is invisible in a green run:
    #: the bed is request-level and optional at the compile seam
    #: (``compile_stage``: ``... or {}``), so a dead bed renders successfully and
    #: silently. This stage deliberately does NOT hard-fail that (a reel with
    #: fine footage is worth delivering) — but "silent" must be assertable by a
    #: consumer/QA gate without grepping prose.
    music_bed_state: str = "not_selected"
    #: Delivered-vs-ORDERED length, recorded so a reel that missed the requested
    #: duration is visible in the run record rather than only in the encoder's
    #: own self-report. action_reel is SHOT-LIST authored — `cut_plan` solves the
    #: shot windows against `target_duration.target_seconds` — so the delivered
    #: length is the one number that proves the solver's answer survived the
    #: render. Observation ONLY: it never fails the stage. `None` means nothing
    #: usable was ordered, which is NOT the same fact as "on target" (that case
    #: rides out on `degradations` as `DURATION_NOT_ORDERED`).
    duration_conformance: DurationConformance | None = None


class ActionReelCompileStage(CompileStageBase[ActionReelCompileInput, ActionReelCompileOutput]):
    name = "compile"
    input_model = ActionReelCompileInput
    output_model = ActionReelCompileOutput
    no_shots_message = "cut_plan produced no shots to render"

    def __init__(self, client=None) -> None:  # noqa: ANN001 -- mirrors the base signature
        super().__init__(client)
        # Degradations found while BUILDING the render (a dropped shot, a dead
        # music bed) that the base's tail cannot see: `build_shots` and
        # `build_audio_bed` return payloads, not notes, and `build_output` is
        # handed only the compile RESULT. Instance state is safe here because the
        # dispatcher constructs a fresh handler per stage execution
        # (`dag_executor`: `spec.handler_factory(...)` inside `_run_stage`), and
        # `build_shots` — the first hook the base calls — clears it, so an
        # in-stage retry re-derives the list instead of doubling it.
        self._notes: list[str] = []
        # Same lifecycle as `_notes` (reset in `build_shots`, read in
        # `build_output`), but a FLAG rather than prose — see
        # `ActionReelCompileOutput.music_bed_state`.
        self._music_bed_state: str = "not_selected"

    def build_shots(self, ctx: RunContext) -> list[dict[str, object]]:
        plan = ctx.prior_output("cut_plan")
        if not plan:
            raise RuntimeError("cut_plan output missing")
        # First hook of the compile tail: reset so a retried stage does not
        # accumulate the same note twice (see __init__). The bed flag is reset
        # HERE too, not only in __init__: a reused builder whose second run has
        # no bed would otherwise still report the first run's "present".
        self._notes = []
        self._music_bed_state = "not_selected"
        # Narration is deliberately NOT joined here — it is a request-level voice
        # track built in `build_audio_bed`, where the FINAL shot list (and so the
        # reel's real length) is known. A per-shot `audio_url` would silently
        # re-time the cut; see `_build_voice_track`.

        overlays_by_shot = self._overlays_by_shot(ctx)

        shots: list[dict[str, object]] = []
        for planned in plan.get("shots") or []:
            shot_id = str(planned.get("id") or "")
            video_url = str(planned.get("video_url") or "")
            if not video_url:
                # A shot with no fetchable source is dropped rather than sent: the
                # compositor would 422 on the whole render for one bad clip. Noted
                # for the same reason the window-drop below is: a shot the reviewer
                # approved is not in the render, and a shorter reel with no trace of
                # WHY is the silent degradation the notes channel exists to prevent.
                self._notes.append(
                    f"shot {shot_id}: cut_plan gave no fetchable video_url "
                    f"(video_url={planned.get('video_url')!r}); "
                    "shot dropped from the render"
                )
                continue
            window = _source_window(planned)
            if window is None:
                # Same reasoning as the missing-url drop, for the same reason —
                # one unordered clip window 422s the whole batch. Noted, never
                # silent: a shot the reviewer approved is not in the render.
                self._notes.append(
                    f"shot {shot_id}: cut_plan gave no usable source window "
                    f"(in={planned.get('in_seconds')!r}, out={planned.get('out_seconds')!r}); "
                    "shot dropped from the render"
                )
                continue
            lead, out_point = window
            seconds = float(planned.get("seconds") or 0.0)
            if seconds <= 0.0:
                # The 1s fallback below keeps the render alive, but it is NOT free:
                # shots concatenate, so a re-lengthened hold pushes every later cut
                # off the beat for the rest of the reel. Kept, and said out loud.
                self._notes.append(
                    f"shot {shot_id}: cut_plan gave no usable duration "
                    f"(seconds={planned.get('seconds')!r}); "
                    "held for 1.0s instead, which shifts every later cut off the beat"
                )
            shot: dict[str, object] = {
                "id": shot_id,
                # The compositor keys per-shot joins (captions, sound, overlays)
                # off line_id; for a reel the shot IS the line.
                "line_id": shot_id,
                # Shot.seconds must be > 0 or the compositor 422s; a degenerate
                # plan entry falls back to a 1s hold rather than killing the render.
                #
                # Written as an explicit `> 0.0` test and NOT as `seconds or 1.0`:
                # `or` only substitutes on the FALSY 0.0, so a NEGATIVE duration
                # rode through verbatim and 422'd the whole render on the
                # compositor's `Shot.seconds` (gt=0) validator — after all five
                # review gates had been approved, and while the note above
                # already claimed the shot had been "held for 1.0s instead".
                # The guard and the substitution must agree on the same
                # condition, which is the one the guard states: `seconds <= 0.0`.
                "seconds": seconds if seconds > 0.0 else 1.0,
                "video_url": video_url,
                "fit": "cover",
                # Source trim, BOTH ends, unconditionally. `clip_lead_seconds` is
                # no longer gated on `> 0`: a 0.0 in-point is a meaningful value
                # now that it is one half of a window, and omitting it would leave
                # the out-point to be read against an implicit lead.
                "clip_lead_seconds": lead,
                # THE cut. Without it the compositor plays from `lead` to the end
                # of a multi-minute 4K original and the reel is not an edit.
                # `_source_window` has already guaranteed out > in >= 0, which is
                # exactly the compiler's `Shot` validator.
                "clip_out_seconds": out_point,
                # Kill the source audio on every shot. A per-shot dict (not a
                # shared constant) because later passes in the compile tail may
                # project onto a shot's own sub-dicts.
                "audio": {"mute": True},
            }
            speed = float(planned.get("speed") or 1.0)
            if speed and speed != 1.0:
                # Presence-is-opt-in: omitted at 1.0 so an unsolved plan renders a
                # byte-identical payload to a no-warp compile.
                shot["speed"] = speed
            overlays = overlays_by_shot.get(shot_id)
            if overlays:
                shot["overlay_clips"] = overlays
            shots.append(shot)
        # No per-shot `target_fps` is stamped, deliberately. The master rate is a
        # request-level property and it rides via `resolve_cadence` below; a
        # per-shot rate under a differing master is a SECOND conform, not a fix.
        return shots

    def _note_unjoinable_narration(
        self, clips: list[dict[str, object]], cues: list[dict[str, object]]
    ) -> None:
        """Say so when narration was synthesized but nothing it produced can land.

        The original failure this guard was written for was a KEY-SPACE miss: the
        clips were keyed ``vo-N`` and looked up by shot id, so every lookup missed
        without raising — the run paid for the TTS, stored the clips, and shipped a
        reel with no voice-over and a green stage. That join is gone (narration is
        now a request-level voice track, which has no join key to get wrong), but
        the guard is not: a clip can still fail to reach the reel by having no
        fetchable URL, no positive duration, or nothing left of the timeline to
        play over. The failure mode is identical from the outside — paid-for audio,
        silent product, green run — so it is still REPORTED rather than inferred.

        Never hard-failed: footage plus bed is still a deliverable reel, and
        killing the render would throw it away.

        Only the TOTAL miss is noted here; each individually-dropped clip already
        wrote its own note in ``_build_voice_track``, and noting a partial
        placement as a degradation would flag every over-written brief.
        """
        if not clips or cues:
            return
        self._notes.append(
            f"narration produced no placeable audio — voice-over dropped: "
            f"{len(clips)} clip(s) synthesized, 0 laid on the reel; "
            "the reel renders WITHOUT narration"
        )

    async def _build_voice_track(
        self, ctx: RunContext, shots: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """The opt-in narration, as request-level cues on the GLOBAL timeline.

        Why not the obvious thing — a per-shot ``audio_url``. On a narrated shot the
        compiler treats ``seconds`` as a picture FLOOR, never a cap: the segment
        lasts ``max(narration, seconds)`` when the shot carries ``max_seconds`` (a
        reconcile_timeline-bounded beat), and exactly the narration length when it
        does not. The narration is never cut. (Before video_compiler's
        footage-floor fix it lasted the narration length unconditionally and
        ``seconds`` was discarded.) There is no field that caps a shot below its
        narration: ``max_seconds`` bounds only the freeze-pad DRIFT ramp, and
        ``seconds`` can only lengthen. So a 5 s line muxed onto a 1.4 s beat still
        does not truncate —
        it stretches that shot by 3.6 s and pushes EVERY later cut 3.6 s late,
        which is the precise failure ``cut_plan`` exists to prevent, and it does it
        invisibly (``overrun_seconds`` was computed before the stretch). On a
        beat-sync product, breaking the beat is worse than having no voice-over.

        And there is nothing to join on anyway: ``narrate`` reads
        ``brief.narration.lines`` — an authored SCRIPT for the reel, keyed ``vo-N``
        by position — while shots are ``uuid7``s the segment solver invented. Line
        3 is not "the line for shot 3"; no correspondence was ever authored.

        So the script is laid END TO END from the top of the reel as
        ``AmbientCue``s. That block is a global-timeline audio segment: the
        renderer ``atrim``s each cue to ITS OWN window and mixes the bed bus
        against the narration bus with ``amix=duration=first`` — the bed can never
        lengthen the timeline. The cue's ``[start, end]`` is sized from the clip's
        own measured ``seconds`` (never looser: the renderer feeds every cue
        through ``-stream_loop -1``, so a window wider than the asset would repeat
        its head). A cue with no reel left to play over is DROPPED and noted rather
        than quietly stacked at the tail.

        Returns ``[]`` for the normal product (narrate skipped clean) ⇒ no
        ``ambient`` key on the request ⇒ byte-identical to a music-only reel.
        """
        # narrate is opt-in and skip-clean, so a missing/empty output is the
        # NORMAL case (a music-only reel), never an error.
        narrate = ctx.prior_output("narrate") or {}
        clips = [clip for clip in (narrate.get("clips") or []) if isinstance(clip, dict)]
        # The reel's real length, from the FINAL shot list the base hands us
        # (post title-card splice + ken-burns) — the same measure
        # `shared/sound_design` sizes its ambient window against.
        total = sum(_as_float(shot.get("seconds")) or 0.0 for shot in shots)
        cues: list[dict[str, object]] = []
        cursor = 0.0
        for clip in clips:
            clip_id = str(clip.get("line_id") or clip.get("id") or "?")
            url = str(clip.get("artifact_url") or "")
            seconds = _as_float(clip.get("seconds"))
            if not url or seconds is None or seconds <= 0:
                self._notes.append(
                    f"narration clip {clip_id}: no playable audio "
                    f"(url={url!r}, seconds={clip.get('seconds')!r}); it is NOT on the reel"
                )
                continue
            end = min(cursor + seconds, total)
            if end - cursor <= 0:
                self._notes.append(
                    f"narration clip {clip_id}: the script runs past the end of the cut "
                    f"({cursor:.2f}s into a {total:.2f}s reel); it is NOT on the reel"
                )
                continue
            if end < cursor + seconds:
                self._notes.append(
                    f"narration clip {clip_id}: only {end - cursor:.2f}s of its {seconds:.2f}s "
                    f"fits before the end of the {total:.2f}s reel; the tail is cut off"
                )
            cues.append(
                {
                    "url": url,
                    "start_seconds": round(cursor, 3),
                    "end_seconds": round(end, 3),
                    "gain_db": _VOICE_GAIN_DB,
                }
            )
            cursor += seconds
        self._note_unjoinable_narration(clips, cues)
        if not cues:
            return []
        # Same expiry hole the shots go through: these are presigned MinIO URLs
        # stored at narrate time, this blueprint pauses FIVE times before the
        # render, and `refresh_shot_asset_urls` only walks SHOTS. Re-mint them here
        # through that same helper (and its same never-fail-the-compile policy) by
        # handing it each cue URL in the `audio_url` slot it already knows.
        slots = await refresh_shot_asset_urls([{"audio_url": cue["url"]} for cue in cues])
        for cue, slot in zip(cues, slots, strict=True):
            cue["url"] = slot["audio_url"]
        return cues

    def _overlays_by_shot(self, ctx: RunContext) -> dict[str, list[dict[str, object]]]:
        """``graphics_design`` cards grouped by the shot they are pinned to.

        Three guards, and every one of them writes a note, because the compiler's
        overlay layer fails QUIETLY in opposite directions:

        * a card with NO URL is skipped. The renderer logs ``overlay_clip.dropped``
          for an overlay it cannot fetch and renders on, so sending one buys a
          reel with no graphics and a green run. ``graphics_design`` already
          refuses to emit such a card; this is the second line, for a card that
          reached the output some other way (a resumed run, an operator edit).
        * more than two cards on one shot are TRUNCATED. The compiler's
          ``max_length`` would 422 the entire render instead, which trades a
          missing card for a missing reel.

        * a card naming a shot that is not in the plan (a reviewer deleted the
          shot after the graphics pause) is dropped and noted, because that card
          was reviewed and approved and is now not in the film.
        """
        graphics = ctx.prior_output("graphics_design") or {}
        plan = ctx.prior_output("cut_plan") or {}
        planned_ids = {str(shot.get("id") or "") for shot in plan.get("shots") or []}
        grouped: dict[str, list[dict[str, object]]] = {}
        for card in graphics.get("cards") or []:
            shot_id = str(card.get("shot_id") or "")
            card_id = str(card.get("id") or "?")
            if not str(card.get("url") or ""):
                self._notes.append(
                    f"graphics card {card_id}: no rendered overlay to composite; "
                    "it is NOT on the delivered reel"
                )
                continue
            if shot_id not in planned_ids:
                self._notes.append(
                    f"graphics card {card_id}: names shot {shot_id!r}, which is not in the "
                    "cut; the card is dropped from the render"
                )
                continue
            bucket = grouped.setdefault(shot_id, [])
            if len(bucket) >= _MAX_OVERLAY_CLIPS_PER_SHOT:
                self._notes.append(
                    f"graphics card {card_id}: shot {shot_id} already carries "
                    f"{_MAX_OVERLAY_CLIPS_PER_SHOT} overlays (the compiler's cap); "
                    "this card is dropped rather than 422 the whole render"
                )
                continue
            bucket.append(_overlay_clip(card))
        return grouped

    def title_cards_enabled(self, ctx: RunContext) -> bool:
        """Always ``False``: this product declines the SPLICED title card.

        The base splices an intro/outro card as a free-standing shot at the front
        of the list and nothing rebases the timeline afterwards. On a reel that is
        silent and fatal: the music bed is request-level with no start/offset
        field, so track t=0 IS video t=0 (the invariant `action_reel/cut_plan`
        states at its head), and a 3 s card — the `title_cards` default — pushes
        EVERY solved cut 3 s off the beat. Nothing catches it: no compiler error,
        and the duration auto-QA passes because the declared and actual totals
        both include the card.

        That the VOICE track IS compensated (it is sized against the post-splice
        shots the base hands `build_audio_bed` — see `_build_voice_track`) while
        the music bed is NOT is the evidence this is an oversight rather than a
        design; a bed-offset field would be a compiler-contract change for every
        product, so this lane declines instead.

        Nothing is lost: an authored title reaches the reel as a shot-local
        overlay card (`graphics_design._render_title_card`), whose times ride the
        shot's own timeline and so cannot move a cut.

        NOTED, never silent, when the brief actually asked for a card: the
        reviewer authored something and must be told it was declined and why. A
        brief with no `title_card` block writes no note (it is not a degradation
        to decline what nobody asked for).
        """
        cfg = ctx.config_snapshot.get("title_card") or {}
        if cfg.get("enabled"):
            self._notes.append(
                "brief asked for a spliced title card; it is NOT on the reel — a "
                "front-spliced card shifts every beat-locked cut by its own length "
                "(the music bed has no offset field, so bed t=0 is video t=0). The "
                "title rides as a shot-local graphics overlay instead."
            )
        return False

    def resolve_cadence(self, ctx: RunContext) -> int | None:
        """The master frame rate, taken from the FOOTAGE and not just the config.

        This is the whole reason the base exposes a hook. The drone corpus shoots
        24000/1001 (23.976); the compiler's own master default is 30, so an
        un-answered cadence resamples every pan 5:4 — a duplicate-frame judder on
        exactly the slow aerial moves this product sells.

        ``action_reel/cadence.resolve_master_fps`` is the same function
        ``cut_plan`` quantises its cut boundaries with, called with the same two
        inputs, which is the point: planning on a grid the master cannot express
        re-rounds every boundary renderer-side. Two callers, one answer.

        Unlike the base this ALWAYS returns an int — never ``None``. That is not a
        violation of the base's "None means omit" contract so much as this product
        opting out of it: a reel has measured footage by construction
        (``footage_probe`` is a hard dependency), so "nobody expressed an opinion"
        is not a state this stage can be in. The floor case — an unprobed shoot —
        resolves to 30, which is byte-identical in EFFECT to omitting the key,
        since 30 is the compiler's default.
        """
        return resolve_master_fps(ctx.config_snapshot, ctx.prior_output("footage_probe"))

    async def build_audio_bed(
        self,
        ctx: RunContext,
        shots: list[dict[str, object]],
        *,
        degradations: list[Degradation] | None = None,
    ) -> dict[str, object] | None:
        """The whole reel's request-level audio: the music bed and the voice track.

        Both are request-level for the same structural reason — they span the
        timeline rather than belonging to a cut — but the voice track is here for a
        second, harder reason: it is the ONLY placement that cannot re-time a shot
        (see ``_build_voice_track``). ``None`` when neither resolves, which keeps a
        music-less, narration-less reel byte-identical to before.

        ``degradations`` is accepted to match the base hook and is deliberately
        UNUSED: this override builds the reel's bed itself (``_build_music_bed``)
        rather than through ``sound_design.resolve_audio_bed``, so the
        drop-on-presign-failure site the sink exists for is not on this path. A
        reel bed that fails to resolve is a separate, unwired signal — recording
        it here would be guessing at a mechanism this hook does not run.
        """
        bed: dict[str, object] = {}
        music = await self._build_music_bed(ctx)
        if music is not None:
            bed["music"] = music
        # Sized against the FINAL shots (post title-card splice + ken-burns), which
        # is exactly why the base passes them.
        voice = await self._build_voice_track(ctx, shots)
        if voice:
            bed["ambient"] = voice
        return bed or None

    def _note_ignored_music_fields(self, override: MusicOverride) -> None:
        """Say which authored mix knobs this product could not apply.

        ``duck_db`` is accepted by ``MusicEdit`` for every deliverable type, and on
        this one it does nothing: the compiler's duck is a ``sidechaincompress``
        keyed off the per-shot narration bus while this product's voice-over rides
        the request-level bed bus (see ``_MUSIC_DUCK_DB``). Dropping it silently
        would be the same defect this whole path exists to fix — a control that
        reads as applied and is not — so it is reported. A note also turns the run
        ``degraded``, which is the honest reading of "you asked for something I did
        not do".
        """
        ignored = override.ignored_mix_fields()
        if ignored:
            self._notes.append(
                "the music edit set "
                + ", ".join(ignored)
                + ", which this product cannot apply: its voice-over rides the music "
                "bus itself, so there is nothing for the compiler's sidechain duck to "
                "key off. The value was NOT applied"
            )

    async def _build_music_bed(self, ctx: RunContext) -> dict[str, object] | None:
        """The continuous music bed for the whole reel.

        Request-level, not per-shot: the bed is ONE track the cut is timed
        against, so restarting it at every cut would destroy the very alignment
        the product is about.

        The URL is RE-PRESIGNED here from the catalog id rather than reused from
        ``cut_plan.music_url``. That is load-bearing: presigns expire in an hour,
        this blueprint pauses FIVE times before the render, and the compile-time
        ``refresh_shot_asset_urls`` pass only walks SHOTS — a request-level bed
        URL stored an hour ago is a dead signature the compiler 500s on.

        ``None`` (no bed) when the id no longer resolves; the reel then renders
        silent-but-real rather than failing a run whose footage is fine — but it
        is NOTED, because a beat-cut reel with no music is the one degradation
        nobody can spot in a green run summary: every stage succeeded, the
        artifact exists, and the product is silent. Notes surface the run as
        ``degraded``, which is exactly what it is.
        """
        # Read + reported BEFORE the early returns, deliberately. `duck_db` is
        # authored on the EDIT, not on the bed, so whether the compiler could
        # honour it has nothing to do with whether a bed was selected or could be
        # presigned. Reporting it only on the happy path meant the two paths where
        # the operator is MOST likely to be re-reading the notes — silent reel, no
        # bed — were the two that stayed quiet about a knob they dropped.
        override = read_music_override(ctx.config_snapshot)
        self._note_ignored_music_fields(override)
        plan = ctx.prior_output("cut_plan") or {}
        track_id = str(plan.get("music_track_id") or "")
        if not track_id:
            # No bed was ever planned (music_select found nothing). Not a
            # degradation of THIS stage — cut_plan already noted it.
            self._music_bed_state = "not_selected"
            return None
        url = await presign_sound_asset(track_id)
        if not url:
            # The silent-reel case. Flagged as well as noted: the note is for a
            # human reading the run, the flag is for anything that has to DECIDE.
            self._music_bed_state = "unpresignable"
            self._notes.append(
                f"music bed {track_id} could not be presigned; the reel is cut to "
                "a track it does not carry and renders SILENT"
            )
            return None
        # Same block shape the shared sound-design resolver emits for every other
        # blueprint's bed — no new compiler contract. Fades keep the bed from
        # slamming in/out at the head and tail; see `_MUSIC_DUCK_DB` for why the
        # duck knob is sent but inert on this product.
        #
        # A studio music edit may tune the level and the fades; those are pure
        # render-time values, so they are honoured HERE (unlike the track itself,
        # which the cut is timed to and which therefore rewinds the run to
        # `music_select`). `duck_db` is the one knob this product cannot honour —
        # it is reported, not dropped, by `_note_ignored_music_fields` above.
        self._music_bed_state = "present"
        return {
            "url": url,
            "gain_db": override.mix_knob("gain_db", _MUSIC_GAIN_DB),
            "fade_in_s": override.mix_knob("fade_in_s", 1.0),
            "fade_out_s": override.mix_knob("fade_out_s", 2.0),
            "duck_db": _MUSIC_DUCK_DB,
        }

    def build_output(self, result: dict[str, object], ctx: RunContext) -> ActionReelCompileOutput:
        # The build-time degradations ride out on the shared notes channel; the
        # base APPENDS its own duration auto-QA notes to whatever is here, so the
        # two channels compose rather than overwrite.
        delivered = float(cast("float", result["duration_seconds"]))
        # ⚠️ CORRECTED. This comment used to claim `result["duration_seconds"]` IS
        # "the compiler's ffprobe of the muxed master". It is NOT, and that false
        # statement is how the defect below survived review:
        # `primitives/compile_stage.py:192-195` says `duration_seconds` is the
        # NOMINAL SUM OF SHOT DURATIONS — the intended timeline — while the ffprobe
        # of the encoded output is a different field, `actual_duration_seconds`.
        # So passing `delivered` alone made this the plan grading itself: a gate
        # that cannot fail, which is exactly what it was written to avoid.
        # ⛔ AND `actual_duration_seconds` IS NOT THE ANSWER EITHER — it is
        # `min(video_stream, container)`, the PICTURE length, so on a truncated
        # master it under-reports exactly the delivery worth catching. The
        # measured rung is `playable_duration_seconds(result)`: `max` of the two
        # SEPARATELY probed streams from the mastering report. It WINS over
        # `delivered` whenever it is a real measurement; `delivered` stays as the
        # fallback for the unmeasured case (no/degraded mastering block, or the
        # compiler's 0.0 "probe failed" sentinel), where the record is stamped
        # `declared` and carries DURATION_UNMEASURED rather than silently passing
        # for a measured verdict.
        # The ordered side delegates to action_reel's own resolver — the same
        # ladder `cut_plan` solved against — so this is like-for-like. Modelled on
        # `promo/render.py::build_output`, the nearest sibling: promo is shot-list
        # authored too, so neither stage has (or wants) the narration-budget
        # reconciler in the comparison.
        conformance = observe_duration_conformance(
            self.name,
            action_reel_ordered_seconds(ctx),
            delivered,
            playable_seconds=playable_duration_seconds(result),
        )
        return ActionReelCompileOutput(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=delivered,
            notes=list(self._notes),
            music_bed_state=self._music_bed_state,
            duration_conformance=conformance,
            degradations=[
                *duration_conformance_degradations(conformance),
                # A SECOND, independent fact about the same master. The verdict
                # above grades the PLAYABLE length (max of the two streams)
                # against the order; it cannot express a picture that stopped
                # early, because on such a master that max IS the audio. The
                # picture-vs-audio gap therefore gets its own code — one number
                # carrying both facts is the regression this replaces.
                *picture_truncation_degradations(result.get("mastering")),
            ],
            # The unmeasured fact rides the NON-degrading tier: a length nobody
            # probed is an unverified verdict, not a bad artifact.
            advisories=duration_conformance_advisories(conformance),
            **output_dimension_fields(result),
        )
