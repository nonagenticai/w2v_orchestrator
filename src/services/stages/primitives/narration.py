"""TTS narration primitives.

``synthesize_and_store`` is the body every narrate/record stage repeated: one
sync ``TTSClient.synthesize`` call, persist the WAV via ``store_tts_clip``, read
the duration back. ``NarrationStageBase`` wraps it in the shared ``stage_gather``
fan-out for the common case — TTS every element of a prior stage's list into one
clip per element. Stages with a non-trivial per-item shape (app_review pairs a
visual; audiobook flattens chapters→chunks) call ``synthesize_and_store``
directly from their own loop.

Because it is the single choke point, it is also where the timeline's **-18 LUFS
level contract** is enforced: ``ensure_tts_loudness`` runs between the synthesize
call and the store, so every one of the narration consumers gets a deterministic
level (or a counted, sidecar-recorded reason why not) without a single call-site
edit.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, cast

from w2v_common.stages import stage_gather

from src.clients.schemas.tts import TTSSynthesis
from src.clients.tts_client import TTSClient
from src.observability.metrics import tts_clip_loudness_lufs, tts_clip_unnormalized_total
from src.services.audio_premaster import (
    UNMEASURABLE_LEVEL,
    premaster_core,
    probe_seconds,
    tts_premaster_fallback_enabled,
)
from src.services.stages.base import (
    Degradation,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithDegradations,
    StageOutputWithNotes,
)
from src.services.stages.generic.line_fit import speed_multiplier
from src.services.stages.shared.degradation import (
    AUDIO_UNNORMALIZED,
    NARRATION_RATE_FLOOR_CLAMPED,
    degradation,
)
from src.services.stages.shared.time_budget import (
    VO_RATE_DRAG_FLOOR,
    VO_RATE_MIN_DELIVERED,
    VO_RATE_NATURAL_FLOOR,
)
from src.services.tts_storage import (
    ClipLoudness,
    deterministic_clip_id,
    find_existing_clip,
    store_tts_clip,
)

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

logger = logging.getLogger(__name__)

_PER_ITEM_TIMEOUT_SECONDS = 600.0
_CONCURRENCY = 4

# How far the re-encoded clip's PROBED duration may differ from the INPUT clip's
# PROBED duration before the normalized bytes are DISCARDED. Every consumer of
# ``synthesize_and_store`` treats the returned ``seconds`` as the length of the
# stored artifact — it sets shot lengths, freeze-pad targets and caption timings —
# so bytes whose real length disagrees would silently desync the whole timeline.
# 50 ms is far below anything audible as drift.
#
# ⚠️ **Probed-vs-probed, deliberately — NOT probed-vs-``duration_seconds``.** The
# backend-reported duration is the length of the AUDIO; a probed duration is the
# length of the CONTAINER, and lossy containers pad. Measured with real ffmpeg on
# 2.500 s of true audio: mp3 at 24 kHz probes 2.5680 s (68 ms of encoder
# delay/priming) and m4a probes ~2.518 s. Comparing either against the reported
# 2.500 s trips a 50 ms tolerance on EVERY clip, so with ``format="mp3"`` 100% of
# narration would silently revert to un-normalized. Probing both sides the same
# way cancels the container padding out, leaving the guard measuring only what
# loudnorm did — which is what it is actually for. Latent today (the format
# defaults to wav, which has no padding), fatal the day a format flag flips.
_DURATION_TOLERANCE_SECONDS = 0.05

# How far the MEASURED clip duration may sit from the line's allotment before the
# line is re-synthesized at a corrected speed. A second synthesize is a real cost
# (money + latency on the critical path), so it must buy a real improvement:
# inside ±5% the compiler's own shot padding absorbs the difference inaudibly,
# and correcting a 0.2 s error on a 5 s line would pay full price for nothing.
_FIT_TOLERANCE_RATIO = 0.05

# tts_backend rejects a synthesize request whose ``speed`` falls outside this
# band, so a corrected speed is CLAMPED into it rather than sent and 4xx'd. When
# the clamp binds, the line's script is simply too long (or too short) for its
# allotment at any legal rate — a script-length fault, not a rate fault.
#
# ⚠️ The floor a CORRECTION may target is NOT this one. ``_SPEED_FLOOR`` is where
# the API stops accepting; ``VO_RATE_NATURAL_FLOOR`` is where the voice stops
# sounding like narration. Fitting against the API limit is what let a line be
# re-synthesized at 0.5 — a 2.08x uniform stretch of both phonemes and pauses —
# and shipped as though it had merely been "fitted". A correction may slow a line
# only as far as the natural floor; past that the fault is the script's length and
# is reported as such, never absorbed by dragging the voice further.
_SPEED_FLOOR = 0.5
_SPEED_CEILING = 2.0
# ⭐ THE FIT FLOOR IS THE DRAG FLOOR (0.97), NOT THE NATURAL FLOOR (0.80).
#
# This is the half of the owner's "narration weirdly slows down" fix that the
# previous three attempts left out. Every one of them moved the floor from the API
# limit (0.5) to the natural floor (0.80) and stopped there — so a line 12% short of
# its allotment was still RE-SYNTHESIZED at 0.8933 and shipped, because 0.8933 is
# inside [0.80, 2.0] and nothing below 0.80 ever bound. The clamp was not the
# detector, and no detector existed above it.
#
# Raising it to :data:`VO_RATE_DRAG_FLOOR` does two things in one move, which is why
# it is the fix rather than a knob:
#
#   1. It makes the drag IMPOSSIBLE rather than merely reportable. A line needing
#      0.8933 now clamps to 0.97, so the voice is slowed at most ~2% (inaudible)
#      instead of 11%.
#   2. It ARMS THE EXISTING REPORTING for the whole band. `_record_floor_clamp` and
#      the clamp note both fire on `exact < _FIT_FLOOR`; with the floor at 0.97 that
#      predicate finally covers [0.80, 0.97) — the band the complaint lived in — with
#      no new branch. The degradation already names the uncovered seconds.
#
# The consequence is DELIBERATE and is the owner's decision (wave 13): the line comes
# in SHORT of its allotment and the picture holds the remainder, rather than the
# allotment being filled by dragging speech. The ordered duration is a target with a
# real under-delivery allowance, not a quota the voice must stretch to fill.
#
# `_SPEED_FLOOR` (the API limit) and `VO_RATE_NATURAL_FLOOR` (the pathological limit)
# both remain in the `max()` so the floor can never be configured BELOW either.
#
# ⭐ SUPERSEDED AS THE OPERATIVE FLOOR (2026-09-17): even 0.97 is slower than natural, and
# the owner still heard it. A correction now floors at :func:`_slow_floor` — the planned
# rate, capped at natural 1.0 — so a clip that comes back short is KEPT at natural speed
# and the picture holds the rest. `_FIT_FLOOR` survives as the absolute backstop and as
# the threshold the floor-clamp degradation fires on.
_FIT_FLOOR = max(VO_RATE_DRAG_FLOOR, VO_RATE_NATURAL_FLOOR, _SPEED_FLOOR)


def _slow_floor(planned: float) -> float:
    """The slowest speed a correction may choose for a line PLANNED at ``planned``.

    ⭐ NEVER SLOWER THAN THE PLAN, NEVER SLOWER THAN NATURAL. A clip that comes back
    SHORT of its allotment FITS: it is kept (the picture's underspoken hold covers the
    rest), never re-synthesized slower. A line planned faster than natural may relax
    back to 1.0 but not past it; a line planned below 1.0 (an author ``rate`` or a
    voice_director hint) keeps its planned speed as the floor. ``_FIT_FLOOR`` stays the
    absolute backstop beneath all of it.
    """
    return max(_FIT_FLOOR, min(planned, VO_RATE_MIN_DELIVERED))


# A/V quality wave 8: three delivered clips measured live (run
# 01a09e7d-c6da-7450-a7cf-91c43a4aecf3) landed at 237.5, 293.9 and 250.2 wpm over
# their SPEECH-ONLY portion — 1.6x-2.0x a ~150 wpm natural rate — while every
# `<break>` on those lines carried `pause_before_ms=0`. The correction was not
# wrong to run; it was starved of the truth that ~42-49% of the MEASURED clip was
# kokoro's own fixed inter-phrase silence, so `_fit_speed` over-drove the speech
# to compensate for silence speed can never touch. `_SPEED_CEILING` (2.0) still
# permits that: a corrected speed above this threshold is not a "fit", it is the
# same defect signature reproducing. This is advisory only — it produces a NOTE in
# `_fit_to_target` naming the rate actually delivered — never a gate: the sample is
# 3 clips, and turning currently-shipping runs red on a threshold this fresh would
# be arming a signal before it has been calibrated.
_RATE_DEFECT_SPEED_THRESHOLD = 1.3

# ⭐ THE TOO-SLOW MIRROR OF THE ABOVE — the note that did not exist.
#
# `_RATE_DEFECT_SPEED_THRESHOLD` fires only ABOVE 1.3, i.e. on a line RUSHED to fit.
# There was no mirror, so the entire slow direction raised nothing: the only slow
# signal anywhere was `_record_floor_clamp`, which fires on `exact < _FIT_FLOOR`, and
# while `_FIT_FLOOR` was 0.80 that meant a line corrected to 0.8933 produced NO note,
# NO degradation, and a silently dragged delivery. That is the blind spot the owner
# kept hearing and the pipeline kept certifying as a pass.
#
# ⛔ WHY THIS IS NOT THE RECIPROCAL OF 1.3. 1/1.3 = 0.769, which is BELOW
# `_FIT_FLOOR` — and `corrected` is the CLAMPED speed, so `corrected < 0.769` is
# unsatisfiable by construction. A "symmetric" mirror would therefore have been an
# arithmetically unfalsifiable guard: present in the diff, incapable of firing, and
# indistinguishable from a working one in any green test run. The threshold has to
# sit strictly ABOVE the clamp to be observable at all, which is what
# :data:`VO_RATE_DRAG_FLOOR` is.
#
# Set slightly above the floor (`+ _EPS_RATE`) rather than AT it so the note fires on
# the clamped-to-the-floor case too: a line clamped to exactly 0.97 is the worst case
# in this direction, and `corrected < 0.97` would have excluded precisely it — the
# same off-by-the-boundary that lets a guard miss its own headline case.
#
# Advisory, exactly like its fast twin: a note (⇒ degraded run), never a raise. The
# structured `NARRATION_RATE_FLOOR_CLAMPED` degradation carries the machine-readable
# half.
_EPS_RATE = 1e-9
_RATE_DEFECT_SLOW_THRESHOLD = VO_RATE_DRAG_FLOOR + _EPS_RATE

# The speed a synthesize call runs at when none is sent — the backend default.
# Correction arithmetic needs a NUMBER for the first pass, and ``speed_for``
# returns None for every stage that does not override it.
_DEFAULT_SPEED = 1.0

_SSML_TAG = re.compile(r"<[^>]+>")


def _spoken_word_count(text: str, ssml: bool) -> int:
    """Words the backend will actually SPEAK, for the clip's provenance sidecar.

    SSML markup is stripped first: counting ``<break time="300ms"/>`` as a word
    would inflate the count on exactly the lines that carry directives, and the
    sidecar's whole purpose is to let the pipeline observe its own words-per-
    minute from its own history instead of a hardcoded constant.
    """
    return len((_SSML_TAG.sub(" ", text) if ssml else text).split())


def _fit_speed(
    measured: float,
    speed: float,
    target: float,
    pause_seconds: float = 0.0,
    service_pause_seconds: float | None = None,
    *,
    floor: float = _FIT_FLOOR,
) -> tuple[float, float, float]:
    """Return ``(exact_speed, legal_speed, fixed_used)`` to make *measured* land on *target*.

    ``speed`` is applied NATIVELY at synthesis, but kokoro's length does NOT scale
    as ``1/speed``: it answers as ``seconds(s) = seconds(b) * (b / s) ** k`` with
    the calibrated ``k`` = :data:`~src.services.stages.generic.line_fit.SPEED_EXPONENT`
    (0.60 — 20 measured same-text takes, see ``line_fit``; the ``1/speed`` model
    this function used to assume mispredicted by rms 3.7%, max 5.4%). The speed
    that makes a clip measured at ``d`` seconds at speed ``s`` land on an
    allotment of ``A`` is therefore closed-form::

        s* = s * (d / A) ** (1/k)

    — one step, no iteration, computed by the SAME
    :func:`~src.services.stages.generic.line_fit.speed_multiplier` the per-line
    fitter uses, so there is one model of this voice, not two. The second value
    is that speed clamped into the legal band; a caller compares the two to learn
    whether the clamp bound.

    ⚠️ THE RATIO IS ONLY VALID OVER THE PART ``speed`` CAN ACTUALLY MOVE.
    Anything in ``d`` that is FIXED silence does not shrink when ``speed`` rises,
    so dividing it into the ratio makes the solver over-drive the SPEECH to
    compensate for silence it can never compress — and it compounds: faster
    speech → shorter speech portion → silence a LARGER fraction of ``d`` → the
    next fit over-speeds harder.

    There are two sources of that fixed silence, and this function must subtract
    the union of them, never their sum:

    ``pause_seconds`` is the directive-authored SSML ``<break>`` the ORCHESTRATOR
    put in the text (see ``NarrationStageBase.pause_seconds_for``). The backend
    renders it as ``np.zeros(...)``.

    ``service_pause_seconds`` is ``tts_backend``'s own report of ALL silence it
    spliced in (``SynthesizeResponse.pause_seconds``): the inter-phrase pacing
    gap it inserts between every sentence/clause piece, PLUS the authored break.
    This is the one measurement nothing upstream could have known. Three clips
    delivered by run ``01a09e7d-c6da-7450-a7cf-91c43a4aecf3`` were 42.0%, 44.1%
    and 48.8% silence while carrying ``pause_before_ms=0`` — i.e. the authored
    figure said 0.0 and the truth was nearly half the clip.

    ⛔ They are NOT additive: the service figure is a TOTAL that already contains
    the authored break, so adding them double-counts it and over-subtracts. The
    union of "silence the orchestrator knows about" and "silence the service
    reports" is simply the larger of the two, which is also what keeps this
    correct in both degenerate directions — a backend that predates the field
    (``None``) or a provider that inserts no pacing silence (``0.0``) falls back
    to exactly today's authored-break-only arithmetic, and can never *lower* a
    subtraction that already ships.

    Whichever wins, call it ``f``, and subtract it from both sides::

        s* = s * ((d - f) / (A - f)) ** (1/k)

    which reduces to the whole-clip formula when ``f`` is 0. Guarded: an allotment
    at or below its own fixed silence (``target - f <= 0``) or a measurement at
    or below it (a response shorter than the silence it was asked to render)
    falls back to the plain ratio rather than dividing by zero or going negative
    — the pre-existing (mis-attributing, but bounded) behavior.

    The third return value is the ``f`` actually used — 0.0 whenever a guard sent
    the plain ratio — so a caller reporting a real speech-only rate (see the
    over-threshold note in ``_fit_to_target``) need not re-derive this logic.
    """
    reported = (
        service_pause_seconds
        if service_pause_seconds is not None and service_pause_seconds > 0
        else 0.0
    )
    authored = pause_seconds if pause_seconds > 0 else 0.0
    fixed = max(reported, authored)
    if fixed and target - fixed > 0 and measured - fixed > 0:
        exact = speed * speed_multiplier(measured - fixed, target - fixed)
    else:
        exact = speed * speed_multiplier(measured, target)
        fixed = 0.0
    return exact, min(max(exact, floor), _SPEED_CEILING), fixed


def _delivered_wpm(words: int, seconds: float) -> float | None:
    """Words per minute over *seconds*, or ``None`` when that cannot be honest.

    Returns ``None`` rather than a number for a wordless or unmeasurable line —
    a note that states a rate must state a rate that was actually measured, not
    a division by a placeholder.
    """
    if words <= 0 or seconds <= 0:
        return None
    return words * 60.0 / seconds


def _record_floor_clamp(
    degradations: list[Degradation] | None,
    *,
    stage_name: str,
    item_id: str,
    exact: float,
    target: float,
    achieved: float,
) -> None:
    """Record :data:`NARRATION_RATE_FLOOR_CLAMPED` for one dragged line.

    THE SLOW DIRECTION, which nothing reported. ``_RATE_DEFECT_SPEED_THRESHOLD``
    only ever fires above 1.3 — a line RUSHED to fit — while a line that needed
    speed 0.746 and got clamped to the 0.80 floor raised prose and nothing else.
    That clamp is the owner's "the narration weirdly slows down" complaint, and
    it is simultaneously why the beat ships short: at the floor the voice cannot
    stretch far enough, so ``target - achieved`` seconds of the allotment are
    covered by neither speech nor the stretch.

    Those UNCOVERED SECONDS are the number carried in ``detail`` — without them
    a reader cannot tell a 0.2s drag from a 4s hole, which is the same reason
    ``reconcile_timeline._underspoken_note`` reports its residual.

    Nothing here raises or changes a clip: the entry rides the structured
    channel beside the existing note, so the fact is queryable rather than
    grep-able. ``degradations is None`` (a caller on the notes-only path) is a
    no-op, exactly like ``_count_unproven``'s optional sink.
    """
    if degradations is None:
        return
    uncovered = max(0.0, target - achieved)
    degradations.append(
        degradation(
            NARRATION_RATE_FLOOR_CLAMPED,
            f"{stage_name}: line {item_id} needed speed {exact:.3f} to fill its "
            f"{target:.2f}s allotment — below the {_FIT_FLOOR} drag floor, so it was "
            f"kept at no slower than its planned rate and delivers {achieved:.2f}s. "
            f"{uncovered:.2f}s of the allotment is covered by neither speech nor the "
            f"stretch. This is a script-length fault: the line has too few words for its "
            f"slot.",
        )
    )


def _record_slow_note(
    notes: list[str],
    *,
    stage_name: str,
    item_id: str,
    exact: float,
    corrected: float,
    target: float,
    clip: SynthesizedClip,
    fixed_used: float,
) -> None:
    """Note one line whose fit needed a speed under the drag floor.

    The TOO-SLOW MIRROR of the ``_RATE_DEFECT_SPEED_THRESHOLD`` note, which only ever
    fired above 1.3 (a line RUSHED to fit). Nothing reported the slow direction unless
    the clamp bound at 0.80, so the whole ``[0.80, 0.97)`` band — the band the owner's
    "narration weirdly slows down" complaint actually lives in — produced no note at
    all. See :data:`_RATE_DEFECT_SLOW_THRESHOLD`, including why this is NOT the
    reciprocal of 1.3.

    Taken against ``exact``, NOT ``corrected``: ``corrected`` is clamped at
    ``_FIT_FLOOR``, so it cannot say HOW FAR under the floor the line sat — a line
    needing 0.62 and one needing 0.96 both arrive as 0.97, and a note that cannot
    separate those cannot separate a 0.2s hole from a 4s one.

    ⛔ The uncovered SECONDS are deliberately not restated here.
    :func:`_record_floor_clamp` derives them from the ACHIEVED duration, and a second
    estimate of one quantity from a different input is how the note and the solver
    drifted apart last time.

    Advisory, like its fast twin: appends a note (⇒ a ``degraded`` run), never raises,
    never changes a clip.
    """
    if exact >= _RATE_DEFECT_SLOW_THRESHOLD:
        return
    speech = _speech_only_seconds(clip, fixed_used)
    rate = None if speech is None else _delivered_wpm(clip.words, speech)
    measured = (
        ""
        if rate is None or speech is None
        else f"; its {clip.words} words deliver {rate:.1f} wpm over {speech:.2f}s of speech"
    )
    notes.append(
        f"{stage_name}: line {item_id} needs speed {exact:.3f} (under the "
        f"{VO_RATE_DRAG_FLOOR} drag floor) to fill its {target:.2f}s allotment, so it "
        f"is delivered at {corrected:.3f} and comes in SHORT rather than dragging the "
        f"voice{measured}. This is a script-length fault: the line has too few words "
        f"for its slot — the picture holds the remainder."
    )


def _speech_only_seconds(clip: SynthesizedClip, fixed_used: float) -> float | None:
    """The portion of ``clip.seconds`` that is speech, or ``None`` if unknown.

    Prefers tts_backend's OWN ``speech_seconds`` — summed from the samples it
    spliced in, not re-measured from a waveform — then any fixed silence the fit
    actually subtracted.

    ⛔ Returns ``None`` rather than ``clip.seconds`` when NEITHER is available.
    Falling back to the total duration would report a total-duration rate under
    a "speech" label, and that specific substitution is the entire defect this
    wave exists to correct: the three clips of run
    01a09e7d-c6da-7450-a7cf-91c43a4aecf3 read ~140 wpm over their totals and
    237.5-293.9 wpm over their speech. A note that cannot tell those apart is
    worse than no note, so a backend that does not report the split stays
    silent here instead of stating a number it did not measure.
    """
    reported = clip.speech_seconds
    if reported is not None and reported > 0:
        return reported
    if fixed_used > 0:
        return clip.seconds - fixed_used
    return None


def _count_unproven(
    reason: str, stage_name: str, degradations: list[Degradation] | None = None
) -> None:
    """Increment the unproven-level counter for one clip, and record it.

    Split out from ``_unnormalized`` because not every unproven clip is an
    un-normalized one: ``measure_failed`` keeps the normalized bytes (loudnorm
    genuinely ran) yet still has no level to observe on the histogram. Both must
    land here or the "every clip in exactly one series" invariant is a lie
    precisely where an auditor would go looking.

    That makes this the ONE place both unproven-level outcomes pass through, so
    it is also where the degradation entry belongs. A Prometheus counter is a
    fleet aggregate: it can tell you 4% of clips shipped at an unknown level, and
    it can never tell you WHICH RUN did — the run record read clean. The
    ``degradations`` sink is optional so nothing on the metrics-only path
    changes; when the narrate stage passes one, the clip's run says so.
    """
    tts_clip_unnormalized_total.labels(reason=reason, stage=stage_name).inc()
    if degradations is not None:
        degradations.append(
            degradation(
                AUDIO_UNNORMALIZED,
                f"a {stage_name} clip was stored at an UNPROVEN loudness "
                f"(reason={reason}); it hits the timeline at whatever level the "
                "backend produced",
            )
        )


def _unnormalized(
    result: TTSSynthesis,
    reason: str,
    stage_name: str,
    degradations: list[Degradation] | None = None,
) -> tuple[bytes, ClipLoudness]:
    """Record one clip stored at an UNKNOWN level and pass the original bytes through.

    Fail-open is the right behaviour — a level problem must never fail a paid
    narrate — but it must not be *invisible*, which is what a lone WARNING was.
    The counter is the alertable signal; the ``ClipLoudness`` goes into the clip's
    sidecar so the level of any specific artifact stays answerable after the fact.
    """
    _count_unproven(reason, stage_name, degradations)
    logger.warning(
        "%s clip stored WITHOUT a proven loudness (%s) voice=%s %.2fs — it will hit the "
        "timeline at whatever level the backend produced",
        stage_name,
        reason,
        result.voice,
        result.duration_seconds,
    )
    return result.audio_bytes, ClipLoudness(source="none", normalized=False)


async def ensure_tts_loudness(
    result: TTSSynthesis,
    *,
    stage_name: str,
    degradations: list[Degradation] | None = None,
) -> tuple[bytes, ClipLoudness]:
    """Return ``(audio_bytes, loudness)`` for *result*, normalizing it if need be.

    tts_backend reports the level it mastered a clip to in ``X-TTS-LUFS``. When
    that header is present the clip's level is PROVEN and this is a pure
    passthrough — no ffmpeg, no temp files, nothing on the hot path. That is the
    overwhelmingly common case and it must stay free.

    When the header is ABSENT the level is simply unknown (see
    ``tts_client._integrated_lufs``), and an unknown level is a defect: the
    compiler mixes narration at unity gain and the music duck back-solves from a
    -18 LUFS voiceover. So the orchestrator re-masters the clip itself with the
    same two-pass ``loudnorm`` and reports ``source="orchestrator"``.

    ⚠️ It calls ``audio_premaster.premaster_core``, NOT ``premaster``. ``premaster``
    is the upload path's entry point and checks ``VOICEOVER_PREMASTER_ENABLED``;
    routing narration through it made the two kill-switches secretly dependent, so
    turning uploads off un-normalized every narration clip while the counter blamed
    ``premaster_failed_open`` — an ffmpeg fault — for an operator config change.
    This path's ONLY switch is ``TTS_PREMASTER_FALLBACK_ENABLED``.

    Every clip lands in EXACTLY ONE of the two series, and the reason names the
    real cause. A clip is observed on ``tts_clip_loudness_lufs`` when (and only
    when) a level is known; otherwise it increments
    ``tts_clip_unnormalized_total`` with one of:

    * ``disabled`` — ``TTS_PREMASTER_FALLBACK_ENABLED`` is off. Original bytes.
    * ``unmeasurable_level`` — the clip is under ~0.5 s or near-silent, so it has
      no integrated loudness to normalize toward (see
      ``audio_premaster.UNMEASURABLE_LEVEL``). Nothing failed; the original bytes
      pass through untouched, because normalizing silence up to -18 LUFS is worse
      than leaving it alone.
    * ``premaster_failed_open`` — a REAL failure: no ffmpeg, unknown container, a
      non-zero exit, a timeout, an exception. Original bytes.
    * ``duration_drift`` — the re-encode moved the clip's length, or either side
      could not be probed. Normalized bytes DISCARDED, original stored.
    * ``measure_failed`` — loudnorm ran and its output IS kept (``normalized:
      true``, ``source: "orchestrator"``), but the trailing advisory measurement
      produced no number, so there is nothing to observe on the histogram. The
      level is very probably -18; it is simply unproven, and an unproven clip that
      appeared in neither series is a hole in the audit trail.

    ⚠️ **The caller's ``seconds`` never comes from here.** This function returns
    bytes and metadata only; ``result.duration_seconds`` remains the single
    authority on the clip's length. The drift check exists precisely so the bytes
    can never disagree with that number — on drift the normalized bytes are thrown
    away, not the duration adjusted, because every downstream timing decision has
    already been made against the reported duration.
    """
    if result.integrated_lufs is not None:
        tts_clip_loudness_lufs.labels(source="tts_backend").observe(result.integrated_lufs)
        return result.audio_bytes, ClipLoudness(
            integrated_lufs=result.integrated_lufs,
            source="tts_backend",
            normalized=True,
        )

    if not tts_premaster_fallback_enabled():
        return _unnormalized(result, "disabled", stage_name, degradations)

    try:
        outcome = await premaster_core(result.audio_bytes, result.format)
        if not outcome.normalized:
            reason = (
                UNMEASURABLE_LEVEL
                if outcome.reason == UNMEASURABLE_LEVEL
                else "premaster_failed_open"
            )
            return _unnormalized(result, reason, stage_name, degradations)

        # Probed-vs-probed: see _DURATION_TOLERANCE_SECONDS. Both sides go through
        # the same ffprobe on the same container, so codec padding cancels and only
        # what loudnorm did to the length is left.
        source_probed = await probe_seconds(result.audio_bytes, result.format)
        probed = await probe_seconds(outcome.audio, result.format)
        if (
            source_probed is None
            or probed is None
            or abs(probed - source_probed) > _DURATION_TOLERANCE_SECONDS
        ):
            logger.warning(
                "%s discarding normalized clip: probed=%s vs source-probed=%s (reported %.3fs)",
                stage_name,
                "unprobeable" if probed is None else f"{probed:.3f}s",
                "unprobeable" if source_probed is None else f"{source_probed:.3f}s",
                result.duration_seconds,
            )
            return _unnormalized(result, "duration_drift", stage_name, degradations)
    except Exception:  # normalization must never fail a paid narrate
        logger.warning("%s tts premaster fallback raised", stage_name, exc_info=True)
        return _unnormalized(result, "premaster_failed_open", stage_name, degradations)

    metrics = outcome.metrics
    if metrics.integrated_lufs is None:
        # Normalization SUCCEEDED; only the advisory re-measurement did not. Keep
        # the good bytes, but count the clip so it is not absent from both series.
        _count_unproven("measure_failed", stage_name, degradations)
        logger.warning(
            "%s normalized clip kept but its level is UNPROVEN (measure_failed) voice=%s",
            stage_name,
            result.voice,
        )
    else:
        tts_clip_loudness_lufs.labels(source="orchestrator").observe(metrics.integrated_lufs)
    return outcome.audio, ClipLoudness(
        integrated_lufs=metrics.integrated_lufs,
        true_peak_dbtp=metrics.true_peak_dbtp,
        source="orchestrator",
        normalized=True,
    )


class SynthesizedClip(NamedTuple):
    """One stored narration clip plus the provenance the persisted output carries.

    ``speed`` is the speed the clip was SYNTHESIZED at — the requested value, or
    ``None`` meaning the backend default held (never an invented 1.0). ``words`` is
    the spoken word count. Both were previously only in the MinIO sidecar, so a
    line shipped at speed 0.5 was invisible in the DB stage output.
    """

    clip_id: str
    artifact_url: str
    seconds: float
    speed: float | None
    words: int
    # tts_backend's own split of `seconds` into speech vs. the FIXED silence it
    # spliced in — `SynthesizeResponse.speech_seconds` / `.pause_seconds`, passed
    # through by `TTSClient.synthesize`. Named `service_pause_seconds` HERE only
    # to keep it distinguishable from `_fit_to_target(pause_seconds=...)`, which
    # is the ORCHESTRATOR-authored break and a strict subset of it.
    #
    # `None` on a reused/cached clip (the sidecar predates this field) and on any
    # backend that has not shipped it — a correction then falls back to
    # `pause_seconds_for` alone, exactly today's behaviour. Defaulted so every
    # existing construction site (the cache-hit branch) stays untouched.
    speech_seconds: float | None = None
    service_pause_seconds: float | None = None


async def synthesize_and_store(
    client: TTSClient,
    *,
    text: str,
    voice: str | None,
    run_id: str,
    stage_name: str,
    speed: float | None = None,
    ssml: bool = False,
    degradations: list[Degradation] | None = None,
) -> tuple[str, str, float]:
    """TTS one item and persist the clip; return ``(clip_id, artifact_url, seconds)``.

    ``speed`` is forwarded only when given — the video narrate stages omit it
    (relying on the backend default), so the synthesize call stays identical to
    the hand-written version; audiobook passes an explicit speed.

    Thin wrapper over ``synthesize_and_store_clip`` for callers that do not
    persist speed/words; ``NarrationStageBase`` uses the richer form.
    """
    clip = await synthesize_and_store_clip(
        client,
        text=text,
        voice=voice,
        run_id=run_id,
        stage_name=stage_name,
        speed=speed,
        ssml=ssml,
        degradations=degradations,
    )
    return clip.clip_id, clip.artifact_url, clip.seconds


async def synthesize_and_store_clip(
    client: TTSClient,
    *,
    text: str,
    voice: str | None,
    run_id: str,
    stage_name: str,
    speed: float | None = None,
    ssml: bool = False,
    degradations: list[Degradation] | None = None,
) -> SynthesizedClip:
    """``synthesize_and_store`` returning a ``SynthesizedClip`` (adds speed + words)."""
    words = _spoken_word_count(text, ssml)
    # Content-addressed clip id → a resumed/re-run narrate reuses the clip that
    # was already synthesized+stored instead of paying for it again. A pod reap
    # mid-narrate now costs only the in-flight chunk, not the whole book.
    clip_id = deterministic_clip_id(text, voice, speed)
    existing = await find_existing_clip(run_id, stage_name, clip_id)
    if existing is not None:
        artifact_url, seconds = existing
        logger.debug("%s reuse clip=%s %.2fs (skipped synth)", stage_name, clip_id, seconds)
        # The clip id hashes the requested speed, so a reused clip was synthesized
        # at exactly that speed — knowable without reading the sidecar.
        return SynthesizedClip(clip_id, artifact_url, seconds, speed, words)

    kwargs: dict[str, Any] = {"text": text, "voice": voice}
    if speed is not None:
        kwargs["speed"] = speed
    # Only add ssml when True — a plain line's synthesize call stays byte-identical
    # to the pre-SSML version (mirrors how `speed` is conditionally added).
    if ssml:
        kwargs["ssml"] = ssml
    result = await client.synthesize(**kwargs)
    # `seconds` is ALWAYS the duration tts_backend reported. ensure_tts_loudness may
    # swap the bytes for a re-mastered copy, but it guarantees (via its own probe)
    # that the swap did not move the clip's length — so the timeline cannot shift
    # as a side effect of levelling.
    seconds = result.duration_seconds
    audio_bytes, loudness = await ensure_tts_loudness(
        result, stage_name=stage_name, degradations=degradations
    )
    artifact_url = await store_tts_clip(
        run_id=run_id,
        stage_name=stage_name,
        clip_id=clip_id,
        audio_bytes=audio_bytes,
        audio_format=result.format,
        seconds=seconds,
        loudness=loudness,
        # ensure_tts_loudness guarantees the re-mastered copy is the SAME LENGTH
        # (it probes to confirm), so timings measured against the pre-master audio
        # remain valid against the bytes actually stored. If that invariant ever
        # breaks, these timings drift and captions desync — it is asserted there,
        # not assumed here.
        word_timings=[w.model_dump() for w in result.word_timings] if result.word_timings else None,
        # Provenance: WHO spoke this clip and under what request. `voice`/`provider`
        # come off the RESULT, not the request — the request's voice may be None
        # (backend default), and the sidecar must record what actually spoke.
        # `speed` is the requested value (None ⇒ the backend default held), which
        # together with `word_count` is what makes a real words-per-minute
        # observable per voice instead of assumed from a constant.
        voice=result.voice,
        provider=result.provider,
        speed=speed,
        word_count=words,
    )
    logger.debug(
        "%s stored clip=%s bytes=%d %.2fs lufs=%s source=%s",
        stage_name,
        clip_id,
        len(audio_bytes),
        seconds,
        loudness.integrated_lufs,
        loudness.source,
    )
    return SynthesizedClip(
        clip_id, artifact_url, seconds, speed, words, result.speech_seconds, result.pause_seconds
    )


class NarrationStageBase[I: StageInput, O: StageOutput](StageHandler[I, O]):
    """Narrate every element of ``source_stage[source_list_key]`` into a clip.

    Fully declarative: a subclass sets ``name`` / ``input_model`` /
    ``output_model`` plus ``source_stage`` / ``source_list_key`` /
    ``source_id_field`` / ``clip_model``. The output model must take a single
    ``clips`` kwarg; each clip is ``clip_model(id, <source_id_field>, artifact_url,
    seconds, speed, words)`` — ``speed``/``words`` are dropped by a clip model that
    does not declare them.
    """

    source_stage: ClassVar[str]
    source_list_key: ClassVar[str]
    source_id_field: ClassVar[str] = "line_id"
    clip_model: ClassVar[type[StageOutput]]

    # Candidate source stages, HIGHEST precedence first, for consumers that must
    # resolve "which stage's output does this handler actually read?" WITHOUT a
    # RunContext — namely the re-narrate patcher (``runs/regenerate.py``), which
    # rewrites the line's ``text`` in that stage's preserved output so the
    # captions stay in sync with the re-synthesized audio. Empty ⇒ ``source_stage``
    # is the only candidate (the default, true for every stage that does not
    # override ``resolve_script``). A stage that DOES override ``resolve_script``
    # MUST declare the same precedence here or the patcher writes the new text to
    # a stage nobody reads.
    source_stage_chain: ClassVar[tuple[str, ...]] = ()

    def __init__(self, client: TTSClient | None = None) -> None:
        self._client = client or TTSClient()

    def voice_for(self, item: dict, voice: dict, ctx: RunContext) -> str | None:
        """Kokoro voice id for one source item. Default: the single configured
        ``voice.voice_id`` (None → backend default) — preserves the original
        single-narrator behaviour for every stage. Multi-voice stages (e.g. a
        promo dialogue) override this to cast per-item (e.g. by ``item['speaker']``,
        gated on a config flag read from ``ctx``). NOT the free-text descriptor
        (a tone hint for the LLM); TTS 500s on an unknown voice id."""
        return voice.get("voice_id")

    def speed_for(self, item: dict, ctx: RunContext) -> float | None:
        """Per-line TTS speed multiplier for one source item. Default ``None`` ⇒
        ``speed`` is omitted from the synthesize call entirely (the backend default
        holds and ``deterministic_clip_id`` hashes the same key as before), so every
        stage that does not override this is byte-identical. A stage whose plan
        carries a per-line rate (demo_video's ``vo_rate``, stamped by
        ``reconcile_timeline``) overrides this to return it."""
        return None

    def fit_target_for(self, item: dict, ctx: RunContext) -> float | None:
        """Seconds this line's audio must occupy, or ``None`` to accept what TTS returns.

        Default ``None`` ⇒ the measured duration is accepted verbatim and exactly
        ONE synthesize call is made, so every stage that does not override this
        (promo / lesson / course / audiobook) is byte-identical to before the hook
        existed. Those stages send no rate today and must not start now.

        A stage whose plan stamps a per-line ALLOTMENT overrides this to return it.
        Returning a target arms the post-synthesis correction in ``handle``: the
        line is measured against the allotment and, when it misses by more than
        ``_FIT_TOLERANCE_RATIO``, re-synthesized ONCE at the exactly-correct speed.
        ``speed`` is applied natively at synthesis and kokoro's length answers it
        as ``speed ** -SPEED_EXPONENT`` (calibrated in ``line_fit``), so that speed
        is closed-form — see ``_fit_speed``.
        """
        return None

    def measured_fit_target_for(
        self, item: dict, ctx: RunContext, measured_seconds: float
    ) -> float | None:
        """The fit target once the first synthesis has been MEASURED.

        Default: :meth:`fit_target_for`, ignoring the measurement — so every stage that
        does not override this (including every stage overriding only
        ``fit_target_for``) corrects against exactly the same target as before.

        A stage whose target is a RANGE rather than a point overrides this: demo_video's
        underspoken hold makes the voice's window ``[allotment - hold, allotment]``, and
        a line measured inside that range must be kept, not re-synthesized FASTER to
        open a hold the plan only estimated it would need.
        """
        return self.fit_target_for(item, ctx)

    def synth_text_for(self, item: dict, ctx: RunContext) -> tuple[str, bool]:
        """Return (text_to_synthesize, is_ssml). Default: the plain line text,
        not SSML — byte-identical to today. A stage that lowers performance
        directives into SSML overrides this."""
        return str(item["text"]), False

    def pause_seconds_for(self, item: dict, ctx: RunContext) -> float:
        """Fixed SSML ``<break>`` seconds baked into this line's ``synth_text_for``
        output, or ``0.0`` when it carries none. Default ``0.0`` ⇒ every stage that
        does not author SSML pauses (i.e. every stage except the one overriding
        ``synth_text_for`` with a break) is byte-identical to before this hook
        existed: ``_fit_speed`` gets ``pause_seconds=0`` and its formula reduces to
        the original ratio.

        A stage that overrides ``synth_text_for`` to prefix a ``<break time="…ms"/>``
        must override this too, reading the SAME directive it used to build the
        SSML (not re-parsing the markup back out) — the break is fixed silence at
        render time, never scaled by ``speed``, so ``_fit_speed`` needs to know its
        length to avoid mis-attributing it to the speakable portion of the line.
        """
        return 0.0

    def resolve_script(self, ctx: RunContext) -> dict[str, Any] | None:
        """The source spine dict to narrate. Default: this stage's ``source_stage``
        output verbatim. A subclass whose spine may be re-emitted by a later
        skip-clean stage (e.g. demo_video's ``script_doctor``) overrides this to
        prefer the current-best output while keeping the same fan-out below."""
        return ctx.prior_output(self.source_stage)

    async def handle(self, payload: I, ctx: RunContext) -> O:
        script = self.resolve_script(ctx)
        if not script:
            raise RuntimeError(f"{self.source_stage} output missing")
        voice = ctx.config_snapshot.get("voice") or {}
        run_id = str(ctx.run_id)
        notes: list[str] = []
        # Structured twin of `notes`. `stage_gather` below fans out with
        # `asyncio` tasks, which COPY the caller's context — they share this one
        # list object, so per-clip appends from concurrent narrations all land
        # here. (Appending to a list is atomic under the GIL and every task runs
        # on this one event loop, so no lock is needed.)
        degradations: list[Degradation] = []

        async def _narrate(item: dict) -> StageOutput:
            synth_text, is_ssml = self.synth_text_for(item, ctx)
            item_id = str(item["id"])
            speed = self.speed_for(item, ctx)
            first = await synthesize_and_store_clip(
                self._client,
                text=synth_text,
                voice=self.voice_for(item, voice, ctx),
                run_id=run_id,
                stage_name=self.name,
                speed=speed,
                ssml=is_ssml,
                degradations=degradations,
            )
            kept = await self._fit_to_target(
                item,
                ctx,
                item_id=item_id,
                synth_text=synth_text,
                is_ssml=is_ssml,
                voice=voice,
                run_id=run_id,
                speed=speed,
                clip=first,
                notes=notes,
                pause_seconds=self.pause_seconds_for(item, ctx),
                degradations=degradations,
            )
            # model_validate (dict) rather than kwargs: clip_model is a dynamic
            # ClassVar[type[StageOutput]] the type checker can't introspect.
            return self.clip_model.model_validate(
                {
                    "id": kept.clip_id,
                    self.source_id_field: str(item["id"]),
                    "artifact_url": kept.artifact_url,
                    "seconds": kept.seconds,
                    # Speed of the clip actually KEPT (the corrected one after a
                    # fit re-synth) + spoken words, so a slowed/fast line is
                    # visible in the persisted output, not only in the sidecar.
                    # A clip model without these fields ignores them (pydantic
                    # default extra="ignore").
                    "speed": kept.speed,
                    "words": kept.words,
                }
            )

        clips = await stage_gather(
            script[self.source_list_key],
            _narrate,
            concurrency=_CONCURRENCY,
            per_item_timeout_seconds=_PER_ITEM_TIMEOUT_SECONDS,
            per_item_retries=1,
            stage_name=self.name,
        )
        out = cast("O", self.output_model.model_validate({"clips": clips}))
        # Only when a note was actually raised, and only where the output model
        # HAS the channel — an unconditional `notes: []` would change the persisted
        # output of every stage that never fits anything. `isinstance` on the built
        # model rather than an untyped payload mapping keeps this checked: it
        # narrows `out` to the class that declares `notes`, so a wrong key is a
        # type error here rather than a silently-dropped extra.
        if notes and isinstance(out, StageOutputWithNotes):
            out.notes = sorted(notes)
        # Same guard, same reason, for the structured channel: assign only when
        # something was raised AND the model declares the field, so the persisted
        # output of a clean narrate is unchanged. `StageOutputWithNotes` inherits
        # from `StageOutputWithDegradations`, so the narrower isinstance above
        # would also have worked -- this one is deliberately the WIDER base, so a
        # narrate output that carries only the structured channel still records.
        if degradations and isinstance(out, StageOutputWithDegradations):
            out.degradations = degradations
        return out

    async def _fit_to_target(
        self,
        item: dict,
        ctx: RunContext,
        *,
        item_id: str,
        synth_text: str,
        is_ssml: bool,
        voice: dict,
        run_id: str,
        speed: float | None,
        clip: SynthesizedClip,
        notes: list[str],
        pause_seconds: float = 0.0,
        degradations: list[Degradation] | None = None,
    ) -> SynthesizedClip:
        """Re-synthesize *clip* ONCE at a corrected speed if it misses its allotment.

        Returns the clip to keep — the corrected one when a correction happened and
        succeeded, otherwise the one passed in. Never raises: a rate problem must
        not fail a paid narrate, and turning currently-passing runs red is out of
        scope. Every path that keeps a clip which does NOT meet its allotment
        records a note, so a run that shipped a mis-fitted line says so.

        ``pause_seconds`` (from :meth:`pause_seconds_for`) is fixed SSML-break
        silence already baked into ``clip``/``seconds`` that ``speed`` cannot move.
        It is 0.0 for every stage that does not author pauses, so the arithmetic
        below is unchanged for them.
        """
        seconds = clip.seconds
        target = self.measured_fit_target_for(item, ctx, seconds)
        # No allotment (the default hook, a footage/card beat, a skipped reconcile)
        # or an unmeasurable clip ⇒ accept what TTS returned, exactly as before.
        if target is None or target <= 0 or seconds <= 0:
            return clip

        if abs(seconds - target) / target <= _FIT_TOLERANCE_RATIO:
            return clip

        effective = _DEFAULT_SPEED if speed is None else speed
        exact, corrected, fixed_used = _fit_speed(
            seconds,
            effective,
            target,
            pause_seconds,
            clip.service_pause_seconds,
            floor=_slow_floor(effective),
        )
        clamped = corrected != exact

        # A/V quality wave 8, item 3 — HONEST NOTE, NOT A GATE.
        #
        # A correction this large is the signature of the defect this wave fixed,
        # not evidence of a fit: the solver is being asked to push the voice far
        # past natural because the line genuinely has too many words for its slot.
        # The number that makes that legible to a human is not the speed multiplier
        # — it is the words-per-minute the clip ALREADY delivers over its SPEECH,
        # against a ~150 wpm natural rate. Three clips on run
        # 01a09e7d-c6da-7450-a7cf-91c43a4aecf3 read 237.5 / 293.9 / 250.2 wpm that
        # way while their TOTAL-duration rate looked unremarkable, because 42-49%
        # of each clip was silence.
        #
        # Recorded BEFORE the re-synthesis branches below so it survives every
        # outcome — pinned, clamped, raised or clean. It changes no arithmetic and
        # fails nothing; `_SPEED_CEILING` is deliberately left where it is.
        if corrected > _RATE_DEFECT_SPEED_THRESHOLD:
            speech = _speech_only_seconds(clip, fixed_used)
            rate = None if speech is None else _delivered_wpm(clip.words, speech)
            if rate is not None and speech is not None:
                notes.append(
                    f"{self.name}: line {item_id} needs speed {corrected:.3f} "
                    f"(over {_RATE_DEFECT_SPEED_THRESHOLD}) to fit its {target:.2f}s "
                    f"allotment; its {clip.words} words already deliver {rate:.1f} wpm "
                    f"over {speech:.2f}s of speech in a {seconds:.2f}s clip "
                    f"({fixed_used:.2f}s subtracted as fixed silence)"
                )

        # ⭐ A/V quality wave 13 — THE SLOW MIRROR, which is the owner's actual
        # complaint. Same shape as the fast note above and recorded in the same place
        # (before the re-synthesis branches, so it survives every outcome), but armed
        # against `_RATE_DEFECT_SLOW_THRESHOLD` instead of the speed threshold.
        #
        # Taken against `exact`, NOT `corrected`: `corrected` is clamped at
        # `_FIT_FLOOR`, so it can never report HOW FAR below the floor the line
        # actually sat — a line needing 0.62 and one needing 0.96 both arrive here as
        # 0.97, and a note that cannot tell those apart cannot tell a 0.2s hole from a
        # 4s one. `exact` is the unclamped truth. The comparison still fires for the
        # clamped-to-the-floor case because `exact <= corrected` in this direction.
        #
        # This note reports the DRAG; `_record_floor_clamp` reports the UNCOVERED
        # SECONDS that are its consequence. Both, because the two facts land in
        # different channels (prose notes vs. the structured degradation) and a reader
        # of either should not have to infer the other. ⛔ The seconds are NOT restated
        # here: `_record_floor_clamp` derives them from the ACHIEVED duration, and a
        # second estimate of the same quantity from a different input is exactly how
        # the note and the solver drifted apart last time.
        _record_slow_note(
            notes,
            stage_name=self.name,
            item_id=item_id,
            exact=exact,
            corrected=corrected,
            target=target,
            clip=clip,
            fixed_used=fixed_used,
        )

        if corrected == effective:
            # Already pinned at the band edge — a second synthesize would produce a
            # byte-identical clip (the id hashes the speed, so it would even hit the
            # cache) and change nothing. Skip the call, keep the record.
            notes.append(
                f"{self.name}: line {item_id} needs speed {exact:.3f} to fit its "
                f"{target:.2f}s allotment; already at the {_FIT_FLOOR}-{_SPEED_CEILING} "
                f"limit of {effective:.2f}, so it runs {seconds:.2f}s"
            )
            if clamped and exact < _FIT_FLOOR:
                _record_floor_clamp(
                    degradations,
                    stage_name=self.name,
                    item_id=item_id,
                    exact=exact,
                    target=target,
                    achieved=seconds,
                )
            logger.warning(
                "%s line=%s CANNOT fit: target=%.2fs achieved=%.2fs speed pinned at %.2f "
                "(exact %.3f outside %.1f-%.1f)",
                self.name,
                item_id,
                target,
                seconds,
                effective,
                exact,
                _FIT_FLOOR,
                _SPEED_CEILING,
            )
            return clip

        try:
            fitted = await synthesize_and_store_clip(
                self._client,
                text=synth_text,
                voice=self.voice_for(item, voice, ctx),
                run_id=run_id,
                stage_name=self.name,
                speed=corrected,
                ssml=is_ssml,
            )
        except Exception:
            # Keep the FIRST clip — it is already synthesized, stored and paid for.
            logger.warning(
                "%s line=%s rate correction to %.3f raised; keeping the %.2fs clip "
                "for a %.2fs allotment",
                self.name,
                item_id,
                corrected,
                seconds,
                target,
                exc_info=True,
            )
            notes.append(
                f"{self.name}: line {item_id} rate correction to {corrected:.3f} failed; "
                f"kept the {seconds:.2f}s clip for a {target:.2f}s allotment"
            )
            return clip

        if clamped:
            # The clamp bound, so the corrected clip is CLOSER but still not a fit.
            # That is a script-length fault (the line has too many words for its
            # slot at any legal rate), not a rate fault — record it, do not fail.
            notes.append(
                f"{self.name}: line {item_id} needs speed {exact:.3f} to fit its "
                f"{target:.2f}s allotment, outside the {_FIT_FLOOR}-{_SPEED_CEILING} "
                f"band; clamped to {corrected:.2f}, achieving {fitted.seconds:.2f}s"
            )
            if exact < _FIT_FLOOR:
                _record_floor_clamp(
                    degradations,
                    stage_name=self.name,
                    item_id=item_id,
                    exact=exact,
                    target=target,
                    achieved=fitted.seconds,
                )
            logger.warning(
                "%s line=%s CANNOT fit: target=%.2fs achieved=%.2fs speed clamped to %.2f "
                "(exact %.3f outside %.1f-%.1f)",
                self.name,
                item_id,
                target,
                fitted.seconds,
                corrected,
                exact,
                _FIT_FLOOR,
                _SPEED_CEILING,
            )
        else:
            logger.info(
                "%s line=%s refitted %.2fs -> %.2fs (target %.2fs) at speed %.3f",
                self.name,
                item_id,
                seconds,
                fitted.seconds,
                target,
                corrected,
            )
        return fitted
