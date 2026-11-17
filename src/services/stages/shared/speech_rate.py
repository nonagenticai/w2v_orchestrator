"""Measure the voice's REAL natural speaking rate instead of guessing it.

``time_budget.DEFAULT_WPM_NATURAL = 150.0`` is a constant that models how fast a
voice talks. Real narration on the live kokoro voice measures 119.4 wpm (the PRODUCTION
row in the table below; this line cited ~94 wpm, which is only the LOW END of the
94-158 per-clip scatter, not the voice's rate). So a word budget derived from 150 is
~26% too generous: a 15s beat is budgeted 37.5 words when that narration fits 29.9 of
them in the window, and the film overruns (a demo_video ordered at 30s delivered
83.07s — 277%).

Re-tuning the constant to any one measured figure is a better-calibrated guess that
goes stale the moment a voice, provider or engine changes — and it is not even one
figure: the same voice reads the calibration passage at 175.2 wpm and production
narration at 119.4. **The system can simply measure it**: tts_backend
returns ``duration_sec`` on every ``POST /api/tts/synthesize``, so one synthesis of a
fixed passage of known word count yields ``words / (duration / 60)`` — the actual
words-per-minute of the actual voice.

**A probe is only as good as its resemblance to what it predicts.** See
:data:`CALIBRATION_TEXT`: the rate a voice delivers depends on how often it stops, so a
passage with realistic sentence lengths measures a usable rate and a single long
sentence measures an optimistic one.

**Never fails a run.** Every entry point returns ``None`` (or the caller's default)
on ANY failure — client error, timeout, a non-positive duration, an implausible
result. A probe that does not answer leaves the caller byte-identical to the
hardcoded-constant behaviour it had before this module existed.

**Measured once per process, per voice.** The result is memoised on
``(provider, voice)`` behind a lock, so N concurrent runs sharing a voice pay for one
probe, not N. A SUCCESS is cached for the life of the process (a voice's rate does not
change); a FAILURE is cached only for :data:`FAILURE_RETRY_AFTER_SECONDS`, so a
transient backend blip does not pin the default for the lifetime of a multi-day pod,
and a sick backend is not re-probed once per run either.

**Falling back is REPORTED, never silent.** ``tts.test`` is a launchd unit on
mini2 — outside the cluster and outside every dashboard — so a blip there used to
swap the measured rate for a hardcoded constant with nothing written down: every
narrated film produced in that window overshot its ordered duration and reported
``completed`` with no note. The delivered rate measured today is 151.6 wpm against a
150.0 default, so on a GOOD day the fallback is nearly right, which is exactly what
makes its silence dangerous — a different voice or provider moves the true rate a
long way and nothing said which day you had. :class:`SpeechRate` (returned by
:func:`resolve_natural_wpm_detail`) carries the answer AND whether it was measured
or assumed, with a :data:`FallbackReason` naming which operational fact caused the
fallback. :func:`resolve_natural_wpm` is unchanged and still returns a plain float.

**A FIXED PASSAGE CANNOT PREDICT ARBITRARY NARRATION — so sample the script.**
Measured on the live backend from inside the orchestrator pod, one voice; the clip
durations are byte-identical to the first clips persisted by live ``app_review`` run
``01a091bc-eae4-724d-b278-7ee323344bf4``, so this reproduces production exactly::

    CALIBRATION_TEXT :  65w  22.25s  ->  175.2 wpm
    real prod line   :  17w   8.37s  ->  121.9 wpm
    real prod line   :  12w   6.62s  ->  108.8 wpm
    real prod line   :  18w   8.62s  ->  125.3 wpm
    PRODUCTION       :  47w  23.61s  ->  119.4 wpm

The passage synthesizes ~47% FASTER than real narration, and per-clip rates scatter
94-158 wpm on an IDENTICAL voice. The difference is the TEXT — numerals ("100 out of
100"), abbreviations ("a11y"), technical nouns, heavy comma structure — not the voice.
No reword of a fixed passage closes that (PR #984 tried and did not), and neither does a
fixed per-utterance cost: splitting ``CALIBRATION_TEXT`` into its five sentences measured
21.575s against 22.255s as ONE synthesis, i.e. a per-utterance constant of **-0.170s**.
The residual is not a constant. It is content.

So this module is **two-phase**:

* **Phase A — budget the ask.** :func:`resolve_natural_wpm_detail` measures
  :data:`CALIBRATION_TEXT` and the caller writes a word budget from it. That number is
  a PRIOR: a request the model is asked to honour, never a prediction of delivery.
  Unchanged behaviour, unchanged process-global ``(provider, voice)`` memo.
* **Phase B — measure before the verdict.** :func:`resolve_script_sample_wpm`
  synthesizes a few of the ACTUAL drafted lines, divides their real words by their real
  seconds, and hands the caller a rate for THIS content. The caller's existing
  verdict -> re-ask -> ``plan_trim`` loop then re-estimates against a true rate. A cold
  first run behaves exactly as before PLUS one correction; there is no new failure mode.

:class:`SpeechRate` carries :attr:`SpeechRate.basis` saying which phase produced it, so
a caller branches on PROVENANCE and never on a value comparison — a genuine 150.0
measurement is still a measurement.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, TypedDict

from src.clients.tts_client import TTSClient
from src.services.stages.primitives.narration import synthesize_and_store_clip
from src.services.stages.shared.degradation import (
    SPEECH_RATE_UNUSABLE,
    Degradation,
    degradation,
)
from src.services.stages.shared.time_budget import DEFAULT_WPM_NATURAL

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from src.services.runs.run_context import RunContext

logger = logging.getLogger(__name__)

# The calibration passage: ordinary prose, no numerals, abbreviations or proper
# nouns (anything a normalizer might expand changes the spoken word count and would
# silently bias the measurement). Its word count is DERIVED from the text below, so
# the two can never drift apart.
#
# **It must be shaped like what it predicts.** A probe pays the fixed per-utterance
# costs of TTS — a sentence-terminal silence at every full stop, plus one lead-in and
# lead-out — and then amortises them over its own word count. The measured wpm is
# therefore a function of the passage's SENTENCE LENGTH, not only of the voice: a
# single 17-word sentence pays one boundary over 17 words (0.059 boundaries/word),
# while the narration this rate is used to budget runs ~12-14 words per sentence
# (~0.077 boundaries/word). The short-sentence structure of real narration costs real
# seconds that a one-sentence probe never paid, so the probe over-reported the rate
# and every word budget derived from it overran.
#
# So the passage below is five sentences of thirteen words each: words-per-sentence
# inside the 12-14 band measured off delivered narration, and five boundaries rather
# than one so a single noisy inter-sentence gap cannot swing the result. This is a
# STRUCTURAL correction — there is deliberately no coefficient, offset or multiplier
# anywhere in this module. The probe measures something shaped like the thing it
# predicts, and whatever wpm that yields is the answer.
CALIBRATION_TEXT: Final = (
    "The team gathered by the river to watch the slow water move past. "
    "Each of them carried a small notebook and a pencil for the morning. "
    "They wrote down what they noticed about the birds along the far bank. "
    "Later the group walked back through the field and shared the notes aloud. "
    "Nobody agreed on the count but everyone enjoyed the quiet hour of work."
)
CALIBRATION_WORDS: Final = len(CALIBRATION_TEXT.split())  # 65
# Sentence-terminating characters, for the structural assertion that this passage is
# multi-sentence with realistic sentence lengths (see the tests).
CALIBRATION_SENTENCE_TERMINATORS: Final = ".!?"

# Plausible human/TTS speaking rates. A conversational audiobook voice sits near 90,
# an energetic promo read near 190; anything outside this band is not a slow or fast
# voice, it is a broken measurement (truncated audio, a padded/silence-suffixed clip,
# a backend echoing a stub duration) and must be rejected rather than believed.
#
# DELIBERATELY UNCHANGED while fixing the calibration shape. This band did not catch
# the over-reported rate, and narrowing it until it would have is fitting a constant to
# one sample — the next wrong number would be a different one and would pass just the
# same. The band's job is to reject a BROKEN measurement (a truncated clip, a stub
# duration), not to adjudicate a plausible-but-wrong one; that is the calibration
# passage's job, and it is the part that was actually defective. Narrow these only from
# a distribution of real per-voice measurements across providers, never from a residual.
MIN_PLAUSIBLE_WPM: Final = 60.0
MAX_PLAUSIBLE_WPM: Final = 260.0

#: How much slower real narration reads than :data:`CALIBRATION_TEXT`, as a ratio.
#:
#: The Phase-A prior used to BE the passage rate, and the passage reads fast: on
#: 2026-10-20 the passage measured **197.6 wpm** while the same voice delivered the
#: drafted narration at **151-166 wpm** (lesson/showcase runs that day; earlier
#: pairs: 197.6 vs 168.8 showcase, ~198 vs 138-140 website_review). Every script
#: prompt then stated a word budget ~20-30% too large for the order. ``0.80`` maps
#: 197.6 to 158.1, the middle of the measured range, and sits inside every observed
#: pair (0.70-0.85), so it is strictly closer than the raw passage for all of them.
#:
#: A RATIO, not a constant wpm, so a slower/faster voice moves its prior with its own
#: measurement (never hardcode a narration wpm — 94, 150 or 158 — anywhere). It is
#: applied ONCE, here in :func:`resolve_natural_wpm_detail`, the single source every
#: Phase-A consumer reads; :func:`measure_natural_wpm` still returns the raw passage
#: rate it logs. It only sizes the opening ask — Phase B/C replace it with the
#: content's own measured rate before any verdict.
PASSAGE_TO_NARRATION_RATIO: Final = 0.80


def narration_prior_wpm(passage_wpm: float) -> float:
    """The Phase-A narration prior for a measured calibration-passage rate."""
    return float(passage_wpm) * PASSAGE_TO_NARRATION_RATIO


# Probing is a courtesy, not a stage — but a bound BELOW the real synthesis time is not
# a safety margin, it is a silent disable: every probe times out, every caller falls back
# to the hardcoded default, and the measurement this module exists for never happens.
# ``tts_client`` documents a ~2000-char chunk (~330 words) rendering in 20-40s on the
# single-lock kokoro box, i.e. roughly 0.06-0.12s of wall clock per word. The 65-word
# calibration passage is therefore a ~4-8s synthesis before any queueing behind another
# in-flight synth — the previous 5.0s bound, sized for a 17-word line, would have
# rejected most of them. This allows that plus room for one synth ahead of it in the
# queue, and stays far under the client's own 300s request timeout.
#
# The cost of the longer bound is bounded and rare: at most one probe per
# ``(provider, voice)`` for the life of the process, and a failure suppresses the next
# probe for :data:`FAILURE_RETRY_AFTER_SECONDS`, so a dead backend costs this once per
# 15 minutes, not once per run. Also used as the probe client's own HTTP timeout so an
# unroutable host fails inside httpx rather than sitting until the outer `wait_for` fires.
_PROBE_TIMEOUT_SECONDS: Final = 30.0
# One transport attempt. The retry ladder exists for narration the run cannot do
# without; a calibration probe simply falls back to the default.
_PROBE_ATTEMPTS: Final = 1
# How long a FAILED probe suppresses the next one for that voice. Long enough that a
# dead backend is not re-probed once per run, short enough that a blip does not cost a
# long-lived pod a whole day of mis-budgeted films.
FAILURE_RETRY_AFTER_SECONDS: Final = 900.0

# How long a SUCCESSFUL probe is trusted before the next caller re-measures it.
#
# Before this existed, `_measured` had no TTL at all — a SUCCESS was cached for the
# life of the process while a FAILURE was bounded at `FAILURE_RETRY_AFTER_SECONDS`.
# That inverted the risk: tts_backend is a launchd unit outside the cluster and
# outside every dashboard (see the module docstring), so a single probe that landed
# on a temporarily-degraded box (a stub duration that was merely "plausible", a
# voice mid-deploy) poisoned every run sharing that pod for as long as the pod lived
# — days, on this fleet — while a probe that failed outright self-healed in 15
# minutes. A long-lived success is exactly as capable of being wrong as a
# short-lived failure; it should not be trusted for longer just because it happened
# to clear the plausibility band once.
#
# Set well above `FAILURE_RETRY_AFTER_SECONDS` (a good measurement should live
# longer than a bad one is suppressed) but still bounded within a pod's lifetime,
# so a voice that genuinely changes rate (a provider swap, a model upgrade) is
# re-measured within a working day rather than never.
SUCCESS_TTL_SECONDS: Final = 6.0 * 3600.0

#: Why a caller got the default instead of a measurement. A single boolean would
#: say "assumed" without saying WHAT to go and look at; these names each point at a
#: different operational fact and a different owner.
#:
#: * ``probe_timeout`` — the synthesis did not answer inside
#:   :data:`_PROBE_TIMEOUT_SECONDS`. tts_backend is up but slow or queued.
#: * ``probe_unreachable`` — the transport never connected. DNS, the launchd unit,
#:   or the route to mini2.
#: * ``probe_error`` — the backend answered with an error, or the client raised
#:   something that is neither of the above.
#: * ``unusable_duration`` — a synthesis came back, and its ``duration_seconds`` was
#:   zero, negative, NaN or infinite. A DIFFERENT fault from a dead backend: the
#:   audio path works and the reported number does not.
#: * ``implausible_rate`` — a finite duration that implies a rate outside
#:   ``[MIN_PLAUSIBLE_WPM, MAX_PLAUSIBLE_WPM]``: truncated audio, a padded clip, a
#:   stub duration.
#: * ``probe_suppressed`` — no probe was attempted at all, because an earlier one
#:   for this voice failed inside :data:`FAILURE_RETRY_AFTER_SECONDS`. The
#:   ORIGINATING reason is preserved and reported instead whenever it is known, so
#:   this value only surfaces when the suppression outlived the record of its cause.
#: * ``probe_unavailable`` — the probe declined to answer and recorded no reason
#:   (the only way to reach this in-process is a test double standing in for
#:   :func:`measure_natural_wpm`). Named rather than guessed.
#: * ``resolution_error`` — the failure was OUTSIDE the probe: reading the run's
#:   voice block, or constructing the default client. Nothing was ever asked of TTS.
#: * ``sample_no_lines`` — Phase B was asked to sample a draft that offered nothing
#:   to sample (no lines, or every line empty of words). Not a backend fault at all.
#: * ``sample_too_few_measured`` — Phase B synthesized the sample and fewer than
#:   :data:`MIN_SAMPLE_LINES` of those syntheses yielded a usable duration. A rate
#:   off one clip is a clip's rate, not the script's (per-clip scatter is 94-158
#:   wpm), so it is refused rather than believed.
#: * ``sample_lines_too_short`` — the sample landed (enough lines, a plausible pooled
#:   rate) but averaged fewer than :data:`MIN_SAMPLE_AVG_WORDS_PER_LINE` words per
#:   sampled line — degenerate scaffolding text, not real narration. See
#:   :data:`MIN_SAMPLE_AVG_WORDS_PER_LINE` for the production evidence.
FallbackReason = Literal[
    "probe_timeout",
    "probe_unreachable",
    "probe_error",
    "unusable_duration",
    "implausible_rate",
    "probe_suppressed",
    "probe_unavailable",
    "resolution_error",
    "sample_no_lines",
    "sample_too_few_measured",
    "sample_lines_too_short",
]

PROBE_TIMEOUT: Final[FallbackReason] = "probe_timeout"
PROBE_UNREACHABLE: Final[FallbackReason] = "probe_unreachable"
PROBE_ERROR: Final[FallbackReason] = "probe_error"
UNUSABLE_DURATION: Final[FallbackReason] = "unusable_duration"
IMPLAUSIBLE_RATE: Final[FallbackReason] = "implausible_rate"
PROBE_SUPPRESSED: Final[FallbackReason] = "probe_suppressed"
PROBE_UNAVAILABLE: Final[FallbackReason] = "probe_unavailable"
RESOLUTION_ERROR: Final[FallbackReason] = "resolution_error"
SAMPLE_NO_LINES: Final[FallbackReason] = "sample_no_lines"
SAMPLE_TOO_FEW_MEASURED: Final[FallbackReason] = "sample_too_few_measured"
SAMPLE_LINES_TOO_SHORT: Final[FallbackReason] = "sample_lines_too_short"

#: WHICH TEXT was measured to produce a rate — the provenance a caller branches on.
#:
#: * ``calibration_passage`` — Phase A. :data:`CALIBRATION_TEXT`, a fixed passage. A
#:   REQUEST to budget the ask against; it does not predict this script's delivery.
#: * ``script_sample`` — Phase B. A deterministic sample of the script's OWN drafted
#:   lines, synthesized by the voice that will speak them. A MEASUREMENT of this
#:   content.
#:
#: This exists so nothing downstream ever has to infer the phase from the value.
#: ``wpm != DEFAULT_WPM_NATURAL`` is not a test for "measured" — a real voice can
#: measure 150.0 — and this repo has already been burned by exactly that sentinel.
#: * ``measured_all`` — Phase C. EVERY spoken line of the script was synthesized and
#:   its real seconds recorded. Not a rate applied to a word count at all: the
#:   duration is the sum of real clip seconds, so the estimate has no sampling error.
#: * ``measured_partial`` — Phase C, minus the lines whose synthesis did not answer.
#:   Those few are extrapolated at the measured rate of the lines that did.
#: * ``authored`` — the author's own ``direction.timing.wpm.natural``
#:   (:func:`authored_natural_wpm`). Outranks every measurement; never a fallback.
RateBasis = Literal[
    "calibration_passage", "script_sample", "measured_all", "measured_partial", "authored"
]

CALIBRATION_PASSAGE: Final[RateBasis] = "calibration_passage"
SCRIPT_SAMPLE: Final[RateBasis] = "script_sample"
BASIS_MEASURED_ALL: Final[RateBasis] = "measured_all"
BASIS_MEASURED_PARTIAL: Final[RateBasis] = "measured_partial"
AUTHORED: Final[RateBasis] = "authored"


@dataclass(frozen=True, slots=True)
class SpeechRate:
    """A words-per-minute rate AND whether anybody measured it.

    The whole point of this type is that ``wpm`` alone cannot answer that: the
    fallback constant and a genuine measurement are the same Python float, and on a
    good day they are within ~1% of each other. A caller holding one of these can
    ask.

    ``reason`` is ``None`` if and only if :attr:`measured` is true — an assumed rate
    always names why, and a measured one never carries a reason to misread.
    """

    #: The rate to budget with. ALWAYS usable, measured or not — availability of the
    #: TTS probe must never fail a paid run, so this is never ``None`` and never NaN.
    wpm: float
    #: True when :attr:`wpm` came off a real synthesis — of :data:`CALIBRATION_TEXT`
    #: when :attr:`basis` is ``calibration_passage``, of this script's own sampled
    #: lines when it is ``script_sample``.
    measured: bool
    #: Which operational fact caused the fallback; ``None`` when measured.
    reason: FallbackReason | None = None
    #: WHICH TEXT produced this rate — see :data:`RateBasis`. Defaults to the
    #: calibration passage, which is what every Phase-A construction in this module
    #: (and every historical one) means.
    basis: RateBasis = CALIBRATION_PASSAGE

    @property
    def assumed(self) -> bool:
        """True when :attr:`wpm` is the caller's default, not a measurement."""
        return not self.measured


@dataclass(frozen=True, slots=True)
class _ProbeFailure:
    """Internal: the probe declined to answer, and which fact caused that."""

    reason: FallbackReason


#: ``(tts pin identity, provider, voice)``. The pin identity is ``"deployment"`` for
#: the platform TTS (no pin / a deployment pin) and ``tenant:<connection>:<version>``
#: for a tenant BYO connection, so one tenant's calibration of ITS voice "x" is never
#: served to another tenant (or the platform) asking for a same-named voice.
_CacheKey = tuple[str, str | None, str | None]

#: The pin identity of the platform TTS (no pin, or a ``deployment`` pin).
_DEPLOYMENT_PIN: Final = "deployment"
#: A pin that could not be read: never shared with a readable one. Its probe fails
#: the same way (the synth reads the same pin), so nothing is cached under it.
_UNREADABLE_PIN: Final = "unreadable"


def _tts_pin_identity() -> str:
    """The bound run's ``tts`` pin as a cache-key component. Never raises."""
    from src.auth.run_providers import current_run_slot
    from src.services.providers.errors import SlotUnavailableError

    try:
        ref = current_run_slot("tts")
    except SlotUnavailableError:
        return _UNREADABLE_PIN
    if ref is None or ref.source != "tenant":
        return _DEPLOYMENT_PIN
    return f"tenant:{ref.connection_id}:{ref.version}"


def _cache_key(provider: str | None, voice: str | None) -> _CacheKey:
    return (_tts_pin_identity(), provider, voice)


# wpm alongside the monotonic instant it expires — see `SUCCESS_TTL_SECONDS`.
_measured: dict[_CacheKey, tuple[float, float]] = {}
_failed_until: dict[_CacheKey, float] = {}
#: The reason the probe for a key last failed, kept beside its suppression window so
#: a SUPPRESSED caller reports the originating fault rather than "suppressed".
_failed_reason: dict[_CacheKey, FallbackReason] = {}
_lock: asyncio.Lock | None = None


def _probe_lock() -> asyncio.Lock:
    """Process-wide probe lock, lazily bound to the running loop.

    Mirrors ``tts_client._synth_gate``: the orchestrator runs one event loop per pod,
    so a single module-level lock is the right scope. Held across the probe so
    concurrent callers for the same voice queue behind one synthesis instead of each
    paying for their own.
    """
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


def clear_measured_wpm_cache() -> None:
    """Drop every memoised rate, Phase A and Phase B, AND every failure suppression.

    Tests only. Phase B's per-draft cache is cleared here too: it is keyed by run id,
    so it cannot leak BETWEEN runs in production, but a test reusing a run id would
    otherwise see a stale content rate and pass for the wrong reason.
    """
    _measured.clear()
    _failed_until.clear()
    _failed_reason.clear()
    _content_measured.clear()


async def measure_natural_wpm(
    client: TTSClient,
    *,
    voice: str | None,
    provider: str | None = None,
) -> float | None:
    """Measure this voice's natural rate in words per minute, or ``None``.

    Synthesizes :data:`CALIBRATION_TEXT` once at ``speed=1.0`` (the backend applies
    ``speed`` natively at synthesis and kokoro's length answers it as
    ``speed ** -SPEED_EXPONENT`` — not ``1/speed`` — so only 1.0 measures the voice
    rather than the multiplier) and divides the known word count by
    the reported duration.

    Memoised on ``(tts pin, provider, voice)``; a cached value returns without a call. Returns
    ``None`` on any failure and NEVER raises — a paid run must not die because a
    calibration courtesy did not answer.

    The failure REASON is not returned here (this signature is depended on by every
    existing caller and by sibling tests that stub it); it is recorded against the
    key and read back by :func:`resolve_natural_wpm_detail`.
    """
    key = _cache_key(provider, voice)
    hit = _cache_lookup(key)
    if not isinstance(hit, _Miss):
        return hit

    async with _probe_lock():
        # Re-check: a queued caller for the same voice waits for the holder's probe
        # rather than firing a duplicate of it.
        hit = _cache_lookup(key)
        if not isinstance(hit, _Miss):
            return hit

        outcome = await _probe(client, voice=voice, provider=provider)
        if isinstance(outcome, _ProbeFailure):
            _failed_until[key] = time.monotonic() + FAILURE_RETRY_AFTER_SECONDS
            _failed_reason[key] = outcome.reason
            return None
        # A SUCCESS clears the recorded fault: only `_measured` is consulted for a
        # rate, so a stale reason beside a live measurement could otherwise make a
        # later caller report a measured rate as assumed.
        _failed_until.pop(key, None)
        _failed_reason.pop(key, None)
        _measured[key] = (outcome, time.monotonic() + SUCCESS_TTL_SECONDS)
        return outcome


class _Miss:
    """Sentinel: this key has no decided answer, so it must be probed."""


_MISS: Final = _Miss()


def _cache_lookup(key: _CacheKey) -> float | _Miss | None:
    """The memoised answer for ``key``, or :data:`_MISS` when one must be measured.

    ``None`` is a real answer here (a probe that failed inside its suppression
    window), which is exactly why a sentinel is needed instead of ``None``-as-miss.

    A SUCCESS past its :data:`SUCCESS_TTL_SECONDS` is evicted here rather than by a
    background sweep: this module runs on request, not on a timer, so "expired" is
    decided lazily, at the moment something would have used it. The stale entry is
    popped (not merely ignored) so it does not leak forever in a long-lived pod.
    """
    cached = _measured.get(key)
    if cached is not None:
        wpm, expires_at = cached
        if time.monotonic() < expires_at:
            return wpm
        _measured.pop(key, None)
    until = _failed_until.get(key)
    if until is not None and time.monotonic() < until:
        return None
    return _MISS


def _classify(exc: BaseException) -> FallbackReason:
    """Which operational fact this probe exception is.

    Deliberately structural rather than an ``httpx``/``TTSClient`` import: this
    module must classify whatever the injected client raises, including a test
    double and a future transport. ``TimeoutError`` covers ``asyncio.TimeoutError``
    (the same class since 3.11); the class-name checks cover the transport families
    that subclass no builtin — ``httpx.ConnectTimeout`` is a timeout and
    ``httpx.ConnectError`` is unreachable, and neither is an ``OSError``.

    The timeout test runs FIRST because ``ConnectTimeout`` satisfies both, and a
    timeout is the more actionable of the two. ``"Connect" in name`` is a substring
    rather than a prefix on purpose: it also catches the wrapped shapes
    (``ClientConnectorError``, ``_ConnectError`` in a test double), and the cost of
    over-matching here is naming the transport instead of the backend — a strictly
    better pointer than the generic bucket.
    """
    name = type(exc).__name__
    if isinstance(exc, TimeoutError) or "Timeout" in name:
        return PROBE_TIMEOUT
    if isinstance(exc, OSError) or "Connect" in name:
        return PROBE_UNREACHABLE
    return PROBE_ERROR


async def _probe(
    client: TTSClient, *, voice: str | None, provider: str | None
) -> float | _ProbeFailure:
    try:
        result = await asyncio.wait_for(
            client.synthesize(
                text=CALIBRATION_TEXT,
                voice=voice,
                provider=provider,
                speed=1.0,
                attempts=_PROBE_ATTEMPTS,
            ),
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        reason = _classify(exc)
        logger.warning(
            "speech-rate probe failed voice=%s provider=%s reason=%s — falling back to "
            "the configured default wpm",
            voice,
            provider,
            reason,
            exc_info=True,
        )
        return _ProbeFailure(reason)

    duration = float(getattr(result, "duration_seconds", 0.0) or 0.0)
    # `duration <= 0.0` is FALSE for NaN, so the non-finite check has to be explicit:
    # otherwise a NaN duration falls through to the band test and is reported as an
    # implausible RATE, naming the wrong fault (the rate arithmetic is fine; the
    # backend's reported number is not).
    if not math.isfinite(duration) or duration <= 0.0:
        logger.warning(
            "speech-rate probe returned an unusable duration=%r voice=%s", duration, voice
        )
        return _ProbeFailure(UNUSABLE_DURATION)

    wpm = CALIBRATION_WORDS / (duration / 60.0)
    if not (MIN_PLAUSIBLE_WPM <= wpm <= MAX_PLAUSIBLE_WPM):
        logger.warning(
            "speech-rate probe produced an implausible %.1f wpm (%d words in %.2fs, "
            "outside [%.0f, %.0f]) voice=%s — rejecting the measurement",
            wpm,
            CALIBRATION_WORDS,
            duration,
            MIN_PLAUSIBLE_WPM,
            MAX_PLAUSIBLE_WPM,
            voice,
        )
        return _ProbeFailure(IMPLAUSIBLE_RATE)

    logger.info(
        "speech-rate probe measured %.1f wpm (%d words in %.2fs) voice=%s provider=%s",
        wpm,
        CALIBRATION_WORDS,
        duration,
        voice,
        provider,
    )
    return wpm


@dataclass(frozen=True, slots=True)
class _VoiceSelection:
    """The only two things the probe reads out of ``config_snapshot["voice"]``.

    A typed payload rather than the raw config mapping: the snapshot block is
    arbitrary user config, so every read out of it needs coercing anyway — doing it
    once, here, means the rest of the module handles ``str | None`` and nothing else.
    ``None`` on either field is meaningful, not missing: it selects tts_backend's own
    default voice / provider, which is a legitimate thing to measure.
    """

    voice: str | None
    provider: str | None


def _voice_selection(ctx: RunContext) -> _VoiceSelection:
    raw = ctx.config_snapshot.get("voice")
    block: Mapping[str, object] = raw if isinstance(raw, Mapping) else {}
    voice = block.get("voice_id")
    provider = block.get("provider")
    return _VoiceSelection(
        voice=str(voice) if voice is not None else None,
        provider=str(provider) if provider is not None else None,
    )


class _Unspecified:
    """Sentinel: the caller named no voice, so the run's configured voice is probed.

    ``None`` cannot serve as "not given" here — it is a MEANINGFUL voice argument that
    selects tts_backend's own default voice, and a caller that deliberately wants that
    measured must be able to say so.
    """


UNSPECIFIED_VOICE: Final = _Unspecified()


async def resolve_natural_wpm(
    ctx: RunContext,
    *,
    client: TTSClient | None = None,
    default: float,
    voice: str | _Unspecified | None = UNSPECIFIED_VOICE,
) -> float:
    """This run's natural wpm: the measured rate for a voice, else ``default``.

    By default the voice is ``config_snapshot["voice"]["voice_id"]`` — absent means the
    backend's default voice, which is a legitimate probe target (``None`` is a cache key
    like any other), not a reason to skip measuring.

    ``voice`` overrides that. The measurement is memoised per voice, so asking for a
    specific one costs one extra probe, not one per call. Omitting the argument is
    byte-identical to the previous behaviour.

    ⚠️ **CAST-MODE GAP — partially closed.** A cast-mode stage delivers its audio
    through ``voice_for_speaker`` (a per-speaker voice), while a budget taken at
    ``config_snapshot["voice"]["voice_id"]`` describes a voice that may never
    speak. A cast whose speakers are slower than ``voice_id`` overruns its budget
    by exactly that ratio. Measured live on run
    ``01a0918c-4419-7299-8ccc-7289c674d7f9`` (lesson_v1): every paragraph cast to
    ``am_michael`` (156.60 wpm) while the budget was taken at the backend default
    ``af_heart`` (175.24 wpm), so a 377-word draw estimated at 129.1s (+7.6%,
    verdict ``ok``, no trim) and delivered 147.685s (+23.07%). The trim was never
    defective; it was never REACHED.

    Every cast-mode stage now closes it, by resolving each line's voice with the SAME
    ``voice_for_speaker`` call its own narrate/record stage makes — importing that
    stage's cast map rather than copying it, so a budget cannot drift from the
    casting — and then calling :func:`resolve_cast_wpm`. The wired modules are
    ``services/stages/lesson/script.py``, ``services/stages/promo/script.py``,
    ``services/stages/website_review/script.py`` and
    ``services/stages/demo_video/reconcile_timeline.py``.

    ``app_review/script.py`` and ``lesson/record.py`` stay single-voice on purpose:
    neither casts, so one rate is the correct rate for them.

    ⚠️ A wired budget is not an ACCURATE one, and this function cannot make it so:
    it measures a FIXED PASSAGE, which is a prior for the ask, never a prediction of
    delivery. Measured on live ``app_review`` run 01a091bc-eae4 (no casting, so the
    voice rate was already right): 271 words estimated 92.9s and delivered 133.215s
    across 16 clips — a -30.3% miss. Budget against the real content rate
    (271 / (133.215/60) = 122.1 wpm) and the same arithmetic lands on 133.2s.

    ⛔ That residual is NOT a per-utterance constant, and nothing here may model it as
    one. 40.4s over 16 clips is 2.53s/clip, and an earlier revision of this docstring
    reported exactly that number as the per-utterance cost — it is a DISPROVED
    hypothesis, falsified three times independently: splitting ``CALIBRATION_TEXT``
    into its five sentences measured 21.575s against 22.255s as ONE synthesis, i.e. a
    per-utterance constant of **-0.170s**. The residual is content, and the only thing
    that measures content is :func:`resolve_script_sample_wpm` (Phase B).
    ``lesson`` hides the same error behind the ``_motion_durations`` clamp.

    The provider always comes from the run config: a cast names voices, never backends.

    Returns a bare ``float`` for backward compatibility with its six call sites. A
    caller that needs to know whether the number was MEASURED or assumed calls
    :func:`resolve_natural_wpm_detail`, which this function delegates to.
    """
    return (await resolve_natural_wpm_detail(ctx, client=client, default=default, voice=voice)).wpm


async def resolve_natural_wpm_detail(
    ctx: RunContext,
    *,
    client: TTSClient | None = None,
    default: float,
    voice: str | _Unspecified | None = UNSPECIFIED_VOICE,
) -> SpeechRate:
    """:func:`resolve_natural_wpm`, plus whether the rate was measured or assumed.

    Same arguments, same fallback value, same never-raises contract — the ONLY
    difference is that the fallback is now recoverable instead of silent. A caller
    holding the returned :class:`SpeechRate` can record
    :func:`speech_rate_fallback_degradation` when ``assumed`` is true, which is the
    difference between "this film overshot and nobody knows why" and a queryable
    fact naming the voice and the fault.

    ``None`` as the probe voice is NOT a fallback reason here: it deliberately
    selects tts_backend's own default voice, which is a legitimate thing to measure
    (see :func:`resolve_natural_wpm`), so "no voice configured" resolves to a real
    measurement like any other key.
    """
    try:
        selection = _voice_selection(ctx)
        probe_voice = selection.voice if isinstance(voice, _Unspecified) else voice
        measured = await measure_natural_wpm(
            client or TTSClient(timeout=_PROBE_TIMEOUT_SECONDS),
            voice=probe_voice,
            provider=selection.provider,
        )
    except Exception:
        logger.warning("speech-rate resolution failed — using default wpm", exc_info=True)
        return SpeechRate(wpm=default, measured=False, reason=RESOLUTION_ERROR)

    if measured is not None:
        return SpeechRate(wpm=narration_prior_wpm(measured), measured=True)

    # A probe that failed under suppression still reports the ORIGINATING fault: the
    # reason outlives the one probe that produced it, for the whole window. When no
    # reason was recorded at all, say so rather than guessing one.
    reason = _failed_reason.get(_cache_key(selection.provider, probe_voice), PROBE_UNAVAILABLE)
    logger.warning(
        "speech-rate falling back to %.1f wpm (default) voice=%s provider=%s reason=%s",
        default,
        probe_voice,
        selection.provider,
        reason,
    )
    return SpeechRate(wpm=default, measured=False, reason=reason)


@dataclass(frozen=True, slots=True)
class CastSpeechRate:
    """The rate to budget a script with when its lines are spoken by SEVERAL voices.

    A single global rate cannot describe a cast. Two voices differing by 12% make a
    +23% overrun look like +7.6% on paper, which is exactly how the live lesson
    defect hid: the estimate landed inside the tolerance band, the verdict came back
    ``ok``, and the trim machinery that would have fixed it was never reached.

    The honest estimate is the WORD-WEIGHTED sum ``Σ(words_i / wpm_i) * 60`` — each
    line's own words at its own voice's own measured rate.

    :attr:`wpm` is that sum expressed back as a single rate
    (``total_words / (seconds / 60)``), because every downstream consumer
    (``narration_budget.estimate_seconds``, ``budget_words``, ``plan_trim``) takes
    one ``words_per_minute``. It is an EXACT re-expression of :attr:`seconds` for the
    script as drawn, and an approximation only after a trim changes the word mix —
    a far smaller error than budgeting a cast at a voice that never speaks, and one
    that no longer hides the direction of the miss.
    """

    #: Effective blended rate for this exact script. Always usable, never NaN.
    wpm: float
    #: The word-weighted estimate in seconds, ``Σ(words_i / wpm_i) * 60``.
    seconds: float
    #: Per distinct voice, the rate used AND whether anybody measured it. Empty
    #: when the weighting did not run.
    per_voice: dict[str | None, SpeechRate]
    #: True when :attr:`wpm` came off a real per-voice weighted sum. False means the
    #: caller's ``default`` was returned unchanged (no words, or an unusable sum),
    #: i.e. the caller is on its previous single-rate behaviour.
    weighted: bool


async def resolve_cast_wpm(
    ctx: RunContext,
    *,
    segments: Sequence[tuple[str, str | None]],
    default: float,
    client: TTSClient | None = None,
) -> CastSpeechRate:
    """Budget a multi-voice script against the voices that will ACTUALLY speak.

    ``segments`` is ``(text, voice_id)`` per line, in script order — the voice
    resolved by the SAME rule the narrate stage will use at synthesis time, which
    is the caller's job because the cast mapping is stage-specific
    (``lesson/narrate.py`` maps a=student/b=instructor; promo maps the same ids to
    different roles). This function does not guess a cast.

    Each DISTINCT voice is measured once through :func:`resolve_natural_wpm_detail`,
    which memoises per ``(provider, voice)`` — an N-line two-voice script costs two
    probes, not N, and zero if another run already measured them.

    **An unmeasurable voice is unmeasured, not "the default rate".** A voice whose
    probe does not answer contributes at ``default`` — byte-identical to the
    behaviour the caller had before this function existed — and its
    :class:`SpeechRate` in :attr:`CastSpeechRate.per_voice` carries ``assumed`` and
    the :data:`FallbackReason`, so the caller can record
    :func:`speech_rate_fallback_degradation` naming that voice. Nothing is silently
    substituted.

    Never raises: :func:`resolve_natural_wpm_detail` swallows its own failures, and
    a script with no words (or one whose weighted sum is not a positive finite
    number) returns ``default`` with ``weighted=False``.
    """
    words = [len(str(text).split()) for text, _ in segments]
    total_words = sum(words)
    if total_words <= 0:
        return CastSpeechRate(wpm=default, seconds=0.0, per_voice={}, weighted=False)

    # ⚠️ Probe ONLY the voices that actually speak. A segment with zero words
    # contributes nothing to ``seconds`` (the sum below already skips it), so probing
    # its voice buys a synthesis nobody hears — and, worse, a voice whose probe fails
    # lands in ``per_voice`` with ``assumed=True``, which every caller turns into a
    # ``SPEECH_RATE_UNUSABLE`` degradation saying the run was budgeted at the default.
    # On a cast where the silent voice is the only unprobeable one that sentence is
    # arithmetically FALSE — the rate really was measured — and it marks a paid run
    # degraded over nothing. That is the shape this repo already learned to fear: a
    # signal that fires on runs it does not describe is not a signal.
    per_voice: dict[str | None, SpeechRate] = {}
    for count, (_, voice) in zip(words, segments, strict=True):
        if count > 0 and voice not in per_voice:
            per_voice[voice] = await resolve_natural_wpm_detail(
                ctx, client=client, default=default, voice=voice
            )

    # A non-positive rate would be a ZeroDivisionError on a PAID run's happy path.
    # It cannot come from a measurement (those are bounded by MIN/MAX_PLAUSIBLE_WPM)
    # — only from a caller passing a broken ``default`` — so it is the caller's own
    # number that is returned, unweighted, rather than a substitute of ours.
    if any(not math.isfinite(r.wpm) or r.wpm <= 0.0 for r in per_voice.values()):
        logger.warning(
            "cast speech-rate weighting got a non-positive rate for one of %d voices "
            "— falling back to the caller's %r wpm",
            len(per_voice),
            default,
        )
        return CastSpeechRate(wpm=default, seconds=0.0, per_voice=per_voice, weighted=False)

    seconds = sum(
        count / per_voice[voice].wpm * 60.0
        for count, (_, voice) in zip(words, segments, strict=True)
        if count > 0
    )
    if not math.isfinite(seconds) or seconds <= 0.0:
        logger.warning(
            "cast speech-rate weighting produced an unusable %r seconds over %d words "
            "— falling back to the caller's %.1f wpm",
            seconds,
            total_words,
            default,
        )
        return CastSpeechRate(wpm=default, seconds=0.0, per_voice=per_voice, weighted=False)

    return CastSpeechRate(
        wpm=total_words / (seconds / 60.0),
        seconds=seconds,
        per_voice=per_voice,
        weighted=True,
    )


def speech_rate_fallback_degradation(
    *, stage: str, rate: SpeechRate, voice: str | None = None
) -> Degradation:
    """One ``degradations`` entry for a rate that was ASSUMED, not measured.

    Reuses ``degradation.SPEECH_RATE_UNUSABLE`` rather than minting a code: that
    code's documented fact is "the rate used to budget narration was not a sound
    measurement, so a substitute was used", and a no-answer probe is that fact
    exactly — every duration number downstream (the estimate, the word budget, the
    trim) was computed at a rate nobody measured. A query asking "which runs were
    budgeted blind?" wants a rejected 40 wpm and an unanswered probe in the same
    result set, and the run-level consequence (``degraded``) is identical.

    Raises ``ValueError`` on a MEASURED rate. A degradation channel that also
    carries successes is one people learn to ignore (``mastering_gate_failed``
    would have fired on 96.7% of masters), and the caller has ``rate.assumed`` to
    ask with — emitting on a measurement is a caller bug, not an input.
    """
    if rate.measured:
        raise ValueError(
            f"{stage}: speech_rate_fallback_degradation called for a MEASURED "
            f"{rate.wpm:.1f} wpm — guard the call with `rate.assumed`"
        )
    target = f" for voice {voice!r}" if voice is not None else ""
    return degradation(
        SPEECH_RATE_UNUSABLE,
        f"{stage}: the speech-rate calibration probe{target} did not yield a "
        f"measurement (reason={rate.reason}), so every narration duration on this run "
        f"was budgeted at the {rate.wpm:.1f} wpm default instead of a measured rate.",
    )


def authored_natural_wpm(ctx: RunContext) -> float | None:
    """The author's ``direction.timing.wpm.natural``, or ``None`` when unstated.

    THE ONE RATE THAT OUTRANKS A MEASUREMENT. An author who writes a natural rate means
    it, so the probe is not even attempted (it would cost a synthesis whose answer is
    discarded) and every word budget on the run must be taken at THIS number — otherwise
    the budget handed to the producer and the budget the solver later solves against
    disagree, which is precisely the drift that pins ``vo_rate`` at its floor and makes a
    film play uniformly slow.

    Lives here, beside the measurement it overrides, so ``demo_video_gen`` (which writes
    the budget) and ``demo_video_reconcile_timeline`` (which solves against it) read the
    SAME resolution and cannot drift. ``None`` for an absent / non-numeric /
    non-positive value: those are not an authored rate, they are the absence of one.
    """
    direction = ctx.config_snapshot.get("direction") or {}
    timing = direction.get("timing") if isinstance(direction, dict) else None
    raw = timing.get("wpm") if isinstance(timing, dict) else None
    if not isinstance(raw, Mapping):
        return None
    value = raw.get("natural")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


# ---------------------------------------------------------------------------
# PHASE B — measure the rate of the SCRIPT, not of a fixed passage
# ---------------------------------------------------------------------------

#: How many drafted lines to synthesize. The measured standard deviation of the
#: per-line rate across three real production lines was ~7.1%, so the standard error
#: of a k-line pooled rate is ~7.1%/sqrt(k): k=4 gives ~3.5%, comfortably inside the
#: +/-10% conformance band that the verdict this feeds is judged against. k=2 gives
#: ~5.0% — still inside, which is why a PARTIAL sample down to
#: :data:`MIN_SAMPLE_LINES` is accepted rather than discarded.
#:
#: These are COUNTS OF LINES, not rates or ratios. Nothing here is a wpm, a
#: multiplier or an offset — the rate is always divided out of real measurements.
DEFAULT_SAMPLE_LINES: Final = 4
#: Below this many usable durations the sample is refused (see
#: :data:`SAMPLE_TOO_FEW_MEASURED`): one clip's rate is that clip's rate, and the
#: measured per-clip scatter on a single voice is 94-158 wpm.
MIN_SAMPLE_LINES: Final = 2
#: One extra sampled line per this many script lines, so a long script (whose lines
#: vary more, and whose total synthesis cost dwarfs the sample) is measured harder.
SAMPLE_LINES_PER_SCRIPT_LINE: Final = 8
#: Hard ceiling. The sample is paid on the critical path; past this the marginal
#: standard-error gain is under a point and the wall clock is not.
MAX_SAMPLE_LINES: Final = 8

#: Below this many words per sampled line ON AVERAGE, the pooled rate is refused
#: rather than believed (see :data:`SAMPLE_LINES_TOO_SHORT`).
#:
#: **PROVEN, not guessed.** ``demo_video_gen`` samples the bdd-drafted SCAFFOLDING
#: text — machine-generated boilerplate ("First, I am on the ... page.") that gets
#: entirely rewritten by ``script_doctor`` before anything is delivered — and that
#: scaffolding runs far shorter than real narration. Measured on three live
#: production runs of the same blueprint, same voice: samples averaging 7.0 and 9.0
#: words/line measured 131.0 and 197.4 wpm against deliveries of 142.4 and 142.0
#: (+39% on the 9.0-word sample); a fourth sample averaging 15.0 words/line (inside
#: the realistic band, just carrying a spelled-out URL) landed a mere -13.6% off by a
#: DIFFERENT mechanism entirely. This is the same physics that made
#: :data:`CALIBRATION_TEXT` itself get rewritten to 13-word sentences: a probe pays
#: TTS's fixed per-utterance cost (lead-in, lead-out, terminal silence) once per
#: line, and that cost is amortised over however many words the line has — few words
#: means the fixed cost dominates and the pooled rate over-reports (or, with enough
#: sampling noise on a handful of 2-3s clips, simply becomes unreliable in either
#: direction). Real delivered narration in this module's own measurements runs
#: 12-14 words/sentence (see :data:`CALIBRATION_TEXT`'s own docstring); this
#: threshold sits a safety margin below that band, so a genuinely short-but-real
#: line is not falsely rejected, while degenerate scaffolding is.
#:
#: Refusing a too-short sample and falling back to the Phase-A prior is not a
#: downgrade: the prior is a fixed multi-sentence passage sized to the SAME
#: realistic band, which this module already trusts more than a probe that cannot
#: even clear it.
MIN_SAMPLE_AVG_WORDS_PER_LINE: Final = 10.0


def sample_size(line_count: int, *, requested: int | None = None) -> int:
    """How many lines to sample from a script of ``line_count`` lines.

    :data:`DEFAULT_SAMPLE_LINES` for an ordinary script, scaling up for a long one
    and clamped to :data:`MAX_SAMPLE_LINES` and to the script's own length.
    ``requested`` overrides the scaling (still clamped to the script), for a caller
    that knows its own budget.
    """
    if line_count <= 0:
        return 0
    if requested is not None:
        return max(0, min(int(requested), line_count))
    scaled = -(-line_count // SAMPLE_LINES_PER_SCRIPT_LINE)  # ceil division
    return min(MAX_SAMPLE_LINES, line_count, max(DEFAULT_SAMPLE_LINES, scaled))


def select_sample_indices(texts: Sequence[str], *, count: int) -> tuple[int, ...]:
    """Which lines to synthesize — by WORD-COUNT STRATA, never at random.

    Determinism is a hard requirement: this repo asserts identical output across
    ``PYTHONHASHSEED`` values, and a sample drawn with ``random`` would make the
    narration verdict — and therefore whether a re-ask and a trim happen at all —
    irreproducible from the same draft. There is no RNG here, and nothing on the path
    from ``texts`` to the returned tuple iterates a set or a dict keyed by strings.

    The strata are the point, not merely the determinism. Per-clip rate scatters
    94-158 wpm on ONE voice and the largest single driver is line length (a short line
    amortises its lead-in and lead-out over few words). Taking the shortest, the
    longest, and evenly-spaced lengths between them spans that axis, where the first
    ``count`` lines of the script would sample only the opening.

    Empty lines are excluded: they contribute no words, so they cannot contribute to a
    words-per-minute, and synthesizing one buys a clip nobody hears.

    Returns indices into ``texts`` in SCRIPT ORDER.
    """
    candidates = [index for index, text in enumerate(texts) if len(str(text).split()) > 0]
    if not candidates or count <= 0:
        return ()
    # (word count, index): the index breaks ties so two equal-length lines always
    # order the same way, whatever order they arrived in.
    ordered = sorted(candidates, key=lambda index: (len(str(texts[index]).split()), index))
    total = len(ordered)
    take = min(count, total)
    if take == 1:
        # One line can only be the median stratum. The extremes are exactly the two
        # lines a single sample must not be drawn from.
        return (ordered[(total - 1) // 2],)

    picked: list[int] = []
    seen: set[int] = set()
    for step in range(take):
        # +0.5 then truncate, rather than `round`: `round` is banker's rounding, so
        # `round(2.5) == 2` collides with `round(1.5) == 2` and a stratum is silently
        # lost. Explicit half-up cannot collide when the spacing is >= 1.
        position = int(step * (total - 1) / (take - 1) + 0.5)
        if position not in seen:
            seen.add(position)
            picked.append(ordered[position])
    # Half-up cannot collide at spacing >= 1, and `take <= total` is the only thing
    # guaranteeing that spacing — so top up rather than silently returning fewer lines
    # than the caller's standard-error budget assumed.
    for position in range(total):
        if len(picked) >= take:
            break
        if position not in seen:
            seen.add(position)
            picked.append(ordered[position])
    return tuple(sorted(picked))


@dataclass(frozen=True, slots=True)
class ScriptSpeechRate:
    """The rate to budget THIS script with, plus everything needed to explain it.

    :attr:`rate` is what a caller uses. ``rate.basis`` says whether Phase B landed
    (``script_sample``) or the caller is still on the Phase-A prior
    (``calibration_passage``) — branch on THAT, never on the wpm value.
    """

    #: The rate to budget with: either the measured content rate, or :attr:`prior`
    #: unchanged. Always usable, never NaN.
    rate: SpeechRate
    #: The Phase-A passage rate exactly as handed in, kept so a caller can report BOTH
    #: numbers — the passage over-reports by ~47% and saying so is the diagnostic.
    prior: SpeechRate
    #: Which lines were selected for synthesis, in script order. Empty when none were.
    sampled_indices: tuple[int, ...] = ()
    #: Spoken words across the sampled lines that yielded a usable duration.
    sampled_words: int = 0
    #: Real seconds across those same lines.
    sampled_seconds: float = 0.0
    #: The TWO-TERM DECOMPOSITION of those same seconds, pooled over only the sampled
    #: lines whose backend reported it — ``_SampleTotals``' ``*_split`` fields, carried
    #: out of this function instead of dying at a log line.
    #:
    #: ⛔ NOTHING BRANCHES ON THESE, and nothing may start. They are MEASUREMENT
    #: ENABLEMENT ONLY. :func:`_log_two_term_decomposition`'s docstring records why:
    #: a two-term budget (``words / speech_rate + n_pieces x per_piece_padding``) was
    #: costed against real clips and REJECTED, because the padding term scales with
    #: SENTENCE COUNT — which the LLM chooses AFTER the budget is written — so a
    #: +/-2 error in the predicted piece count spans ``s*`` 0.83-1.24. Until a fit
    #: exists that survives that, these fields are evidence and nothing else.
    #:
    #: Until 2026-09-15 they were computed and then dropped at that log line, so the
    #: fleet-wide sample a real fit needs did not accumulate anywhere queryable.
    #: Persisting them IS the deliverable.
    #:
    #: Zero means one of two different things and the record cannot tell them apart:
    #: no line reported a split (a backend that cannot answer), or the sample never
    #: ran. Read :attr:`lines_split` first — ``0`` there means "no evidence", not
    #: "zero pause". And note the speech-side WORD count is not carried separately:
    #: :attr:`sampled_words` is the numerator for the speech rate only when every
    #: sampled line reported the split, i.e. when ``lines_split`` equals the number of
    #: measured lines — which is the homogeneous-backend case, but check it rather
    #: than assume it.
    speech_seconds_split: float = 0.0
    pause_seconds_split: float = 0.0
    approx_pieces_split: int = 0
    lines_split: int = 0
    #: Why Phase B did not land; ``None`` when it did. Never ``None`` beside a
    #: ``calibration_passage`` basis, and never set beside a ``script_sample`` one.
    reason: FallbackReason | None = None

    @property
    def sampled(self) -> bool:
        """True when :attr:`rate` was measured off this script's own lines."""
        return self.rate.basis == SCRIPT_SAMPLE


#: ``(run_id, provider, voice, speed+text digest)`` -> measured content wpm.
#:
#: DELIBERATELY NOT the process-global ``(provider, voice)`` memo. That memo answers
#: "how fast is this voice", a property of the voice, true for the life of the process.
#: This one answers "how fast is THIS TEXT in this voice", a property of one draft of
#: one script: the run id scopes it to the run and the digest to the draft revision, so
#: a re-ask (different text) misses and is measured afresh, and no run can ever be
#: budgeted at another run's content rate.
#:
#: Bounded, because a long-lived pod serves unbounded runs. Insertion-ordered eviction
#: (a plain ``dict`` is insertion-ordered since 3.7) — the oldest run's draft is the
#: one nobody will ask for again.
#:
#: ⭐ WAVE 14, N2 — AND ``run_id`` IS ALSO WHAT MAKES IT SAFE ACROSS A TTS DEPLOY.
#: The brief asked for the ``tts_backend`` padding-trim mode to be folded into this key,
#: so that "a rate measured in the padded era must not budget a trimmed voice". The
#: ``run_id`` first element already delivers that property: an entry can only ever be
#: read back by the run that wrote it, so a measurement taken before a mini2 redeploy
#: cannot reach any run started after it. There is no cross-run contamination path to
#: close, and this repo cannot see the trim mode anyway — ``tts_backend`` exposes no
#: trim/padding field on the synthesize response or on ``clients.schemas.tts`` (checked:
#: zero hits for ``trim``/``padding`` in ``src/clients/``), so any token added here
#: would have to come from an env var nobody sets. A knob that is always the empty
#: string is not a guard; it is the "defined but never called" failure mode with a
#: reassuring name.
#:
#: The residual exposure is a single run that OUTLIVES a tts_backend redeploy and
#: re-asks for the SAME draft text afterwards. That is worth closing the day
#: ``tts_backend`` reports its trim mode on the response — at which point it belongs in
#: the digest below, next to ``speed``, and this comment is the pointer.
_ContentKey = tuple[str, str | None, str | None, str]


@dataclass(frozen=True, slots=True)
class _ContentMeasurement:
    """One cached content-rate measurement — the rate AND the evidence behind it.

    ⭐ WAVE 14, N2 — WHY THIS IS NOT JUST A ``float``. Until now the cache stored the
    wpm alone, so the HIT path rebuilt ``ScriptSpeechRate`` with ``sampled_words=0``,
    ``sampled_seconds=0.0`` and the whole two-term split at zero, while the MISS path
    returned all of it. Those zeros are indistinguishable from the documented "no
    evidence" reading of the same fields (``lines_split == 0`` means "no backend
    answered"), so a second stage in the same run that asked for the same rate was told
    the measurement had produced nothing — and the fleet-wide sample the split fields
    were persisted to accumulate silently lost every cache hit.

    The rate returned on a hit was always correct. It is the EVIDENCE that was dropped,
    and dropping evidence on a path that looks identical to the miss path is exactly the
    shape this repo keeps getting bitten by.
    """

    wpm: float
    sampled_words: int
    sampled_seconds: float
    speech_seconds_split: float
    pause_seconds_split: float
    approx_pieces_split: int
    lines_split: int


_content_measured: dict[_ContentKey, _ContentMeasurement] = {}
_CONTENT_CACHE_MAX: Final = 256


def _content_key(
    *,
    run_id: str,
    provider: str | None,
    voice: str | None,
    texts: Sequence[str],
    speed: float | None,
) -> _ContentKey:
    speed_token = "" if speed is None else format(float(speed), ".3f")
    joined = " ".join(str(text) for text in texts)
    digest = hashlib.sha256(f"{speed_token}|{joined}".encode()).hexdigest()[:40]
    return (run_id, provider, voice, digest)


def _remember_content_rate(key: _ContentKey, wpm: float, totals: _SampleTotals) -> None:
    """Cache the measured rate TOGETHER with the sample it was measured from.

    ``totals`` is not optional: a caller that has a rate has the totals it divided, and
    making the evidence a defaulted argument is how the hit path came to return zeros.
    """
    _content_measured[key] = _ContentMeasurement(
        wpm=wpm,
        sampled_words=totals.words,
        sampled_seconds=totals.seconds,
        **_split_evidence(totals),
    )
    while len(_content_measured) > _CONTENT_CACHE_MAX:
        _content_measured.pop(next(iter(_content_measured)))


#: Characters this module treats as ending one synthesis PIECE.
#:
#: ⚠️ A DOCUMENTED APPROXIMATION, not the backend's own rule. ``tts_backend``
#: segments a request into utterances itself and the orchestrator does not import
#: that logic (another repo, and it changes without notice), so this counts
#: sentence terminators as a proxy for "how many utterances will kokoro render".
#: It is used ONLY to report a per-piece padding figure in the log below — never
#: to budget words, size a beat or feed any arithmetic a run depends on. A wrong
#: count therefore mis-reports a diagnostic and changes no delivery.
_PIECE_TERMINATORS: Final = ".!?;:"


def _approx_pieces(text: str) -> int:
    """Roughly how many utterances ``tts_backend`` will split *text* into.

    Counts runs of sentence terminators, with a floor of 1 (a line with no
    punctuation is still one piece). See :data:`_PIECE_TERMINATORS` for why this
    is deliberately an approximation and what it may and may not be used for.
    """
    pieces = 0
    prev_was_terminator = False
    for char in text:
        is_terminator = char in _PIECE_TERMINATORS
        if is_terminator and not prev_was_terminator:
            pieces += 1
        prev_was_terminator = is_terminator
    # Trailing prose after the last terminator ("Hi. And more") is a piece too.
    if text.rstrip() and text.rstrip()[-1] not in _PIECE_TERMINATORS:
        pieces += 1
    return max(pieces, 1)


@dataclass(frozen=True, slots=True)
class _LineMeasurement:
    """One sampled line's real duration, plus the backend's own speech/pause split.

    ``speech_seconds``/``pause_seconds`` are ``None`` on any ``tts_backend`` that
    does not report them (see ``clients.schemas.tts``). ``None`` means "this
    backend cannot answer", never "zero pause" — the whole point of the split is
    that pause time does NOT scale with ``speed``, so a fabricated zero would
    describe a clip that does not exist.
    """

    duration: float
    speech_seconds: float | None = None
    pause_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class _SampleTotals:
    """Words and seconds actually measured, and how many lines produced them.

    The ``*_split`` fields are the two-term decomposition pooled over only those
    lines whose backend reported it, which is why they carry their own line and
    piece counts rather than reusing ``lines``: a mixed fleet (one tts_backend
    upgraded, one not) would otherwise divide a subset's seconds by every line.
    """

    words: int
    seconds: float
    lines: int
    #: Pooled over lines that reported the split ONLY — see the class docstring.
    speech_seconds_split: float = 0.0
    pause_seconds_split: float = 0.0
    words_split: int = 0
    lines_split: int = 0
    approx_pieces_split: int = 0


async def _synthesize_sample_line(
    client: TTSClient,
    *,
    text: str,
    voice: str | None,
    provider: str | None,
    speed: float | None,
    run_id: str,
    narrate_stage_name: str | None,
) -> _LineMeasurement:
    """Real seconds for one sampled line. Raises on failure — the caller drops it.

    When ``narrate_stage_name`` is given the clip is synthesized AND STORED through the
    same ``synthesize_and_store_clip`` the narrate stage uses, under that stage's name
    at the same ``speed``. ``deterministic_clip_id`` then makes the sample and the
    eventual narration the SAME object: narrate's own ``find_existing_clip`` hits and
    synthesizes nothing, so a sampled line that survives to delivery costs zero extra
    synthesis. A sampled line later re-drafted or trimmed away is genuinely spent, and
    a caller that re-synthesizes at a per-line rate (demo_video's ``vo_rate``) never
    reuses at all — see :func:`resolve_script_sample_wpm`.

    Without it, a plain ``client.synthesize`` measures the line and stores nothing.
    """
    if narrate_stage_name is None:
        # `speed` is OMITTED when None rather than sent as 1.0 — exactly as
        # `narration.synthesize_and_store_clip` omits it — because "the backend default
        # holds" and "explicitly 1.0" are different requests, and only the first is what
        # a narrate stage that passes no speed will actually ask for.
        result = (
            await client.synthesize(text=text, voice=voice, provider=provider)
            if speed is None
            else await client.synthesize(text=text, voice=voice, provider=provider, speed=speed)
        )
        return _LineMeasurement(
            duration=float(getattr(result, "duration_seconds", 0.0) or 0.0),
            # `getattr` for the same reason `duration_seconds` above uses it: this
            # seam is duck-typed (stubs and a possibly-older client both reach it),
            # and an absent field must read as "not reported" — which this module
            # then pools nowhere — never as a zero pause.
            speech_seconds=getattr(result, "speech_seconds", None),
            pause_seconds=getattr(result, "pause_seconds", None),
        )
    clip = await synthesize_and_store_clip(
        client,
        text=text,
        voice=voice,
        run_id=run_id,
        stage_name=narrate_stage_name,
        speed=speed,
    )
    return _LineMeasurement(
        duration=float(clip.seconds),
        speech_seconds=getattr(clip, "speech_seconds", None),
        pause_seconds=getattr(clip, "service_pause_seconds", None),
    )


async def _sample_totals(
    client: TTSClient,
    *,
    texts: Sequence[str],
    indices: Sequence[int],
    voice: str | None,
    provider: str | None,
    speed: float | None,
    run_id: str,
    narrate_stage_name: str | None,
) -> _SampleTotals:
    """Synthesize the sampled lines and total the words and seconds that landed.

    Failures are PER LINE, not per sample: one line that times out costs its own
    contribution and nothing else, because ``k-1`` real measurements still beat a
    passage that is ~47% off. ``asyncio.gather`` is bounded by the client's own
    ``_synth_gate`` semaphore, so this does not stampede the single-lock kokoro box.

    A line whose duration is not a positive finite number is dropped for the reason
    :data:`UNUSABLE_DURATION` exists: the audio path answered and the number it
    reported is not usable, and pooling it would corrupt every other line's
    contribution.
    """
    sampled = [str(texts[index]) for index in indices]
    # PER-LINE timeouts, as the docstring above promises. This used to be ONE
    # `wait_for` over the whole gather: a single hung line raised `TimeoutError` out
    # of the fan-out, threw away every line that HAD landed, and surfaced as an
    # anonymous `resolution_error` — 3 good measurements of 4 discarded to punish one.
    # Each line's budget is scaled by its queue position behind the pod-wide synth
    # gate, exactly as Phase C (`_measure_script_lines`) sizes it, so the tail of a
    # long sample is not failed by a queue the backend is working through.
    gate = _tts_gate_size()
    outcomes = await asyncio.gather(
        *(
            asyncio.wait_for(
                _synthesize_sample_line(
                    client,
                    text=text,
                    voice=voice,
                    provider=provider,
                    speed=speed,
                    run_id=run_id,
                    narrate_stage_name=narrate_stage_name,
                ),
                timeout=_PROBE_TIMEOUT_SECONDS * (1 + position // gate),
            )
            for position, text in enumerate(sampled)
        ),
        return_exceptions=True,
    )

    words = 0
    seconds = 0.0
    lines = 0
    speech_split = 0.0
    pause_split = 0.0
    words_split = 0
    lines_split = 0
    pieces_split = 0
    for text, outcome in zip(sampled, outcomes, strict=True):
        # A collected child cancellation is a cancelled run, not a TTS fault — the same
        # rule `_collect_durations` applies to Phase C's fan-out.
        if isinstance(outcome, asyncio.CancelledError):
            raise outcome
        if isinstance(outcome, BaseException):
            logger.warning(
                "speech-rate script sample: a line failed to synthesize (%s: %s) — "
                "measuring the rate off the lines that did",
                type(outcome).__name__,
                outcome,
            )
            continue
        duration = float(outcome.duration)
        if not math.isfinite(duration) or duration <= 0.0:
            logger.warning(
                "speech-rate script sample: a line reported an unusable duration=%r — dropped",
                duration,
            )
            continue
        line_words = len(text.split())
        words += line_words
        seconds += duration
        lines += 1
        # Pool the two-term split over the lines that actually reported one. A
        # `None` from either field means this backend does not report it, so the
        # line contributes to the rate above and to nothing here — pooling it as
        # zero pause would claim a measurement that was never taken.
        speech, pause = outcome.speech_seconds, outcome.pause_seconds
        if (
            speech is not None
            and pause is not None
            and math.isfinite(speech)
            and math.isfinite(pause)
            and speech > 0.0
            and pause >= 0.0
        ):
            speech_split += speech
            pause_split += pause
            words_split += line_words
            lines_split += 1
            pieces_split += _approx_pieces(text)
    return _SampleTotals(
        words=words,
        seconds=seconds,
        lines=lines,
        speech_seconds_split=speech_split,
        pause_seconds_split=pause_split,
        words_split=words_split,
        lines_split=lines_split,
        approx_pieces_split=pieces_split,
    )


class _SplitEvidence(TypedDict):
    """The kwargs :func:`_split_evidence` hands to ``ScriptSpeechRate``.

    A ``TypedDict`` rather than a plain dict so the ``**`` unpack at each of the
    three return sites is checked by ``ty`` — an untyped mapping would make a
    renamed field a silent ``TypeError`` at runtime instead of a type error here.
    """

    speech_seconds_split: float
    pause_seconds_split: float
    approx_pieces_split: int
    lines_split: int


def _split_evidence(totals: _SampleTotals) -> _SplitEvidence:
    """The two-term decomposition as ``ScriptSpeechRate`` kwargs.

    One place builds it so the three ``ScriptSpeechRate`` returns below cannot drift
    apart, and so a future field is added once. Carried on ALL THREE — including the
    two REJECTING returns (too few lines measured, implausible pooled rate) — because
    the split is the evidence that EXPLAINS a rejection: a sample whose pooled wpm
    came back outside the plausible band is exactly the sample whose speech/pause
    breakdown says whether the audio was truncated or the padding exploded. Dropping
    it on the failure path would withhold the measurement precisely where it is worth
    most. The rejection itself is unaffected — it already happened.
    """
    return _SplitEvidence(
        speech_seconds_split=totals.speech_seconds_split,
        pause_seconds_split=totals.pause_seconds_split,
        approx_pieces_split=totals.approx_pieces_split,
        lines_split=totals.lines_split,
    )


async def resolve_script_sample_wpm(
    ctx: RunContext,
    *,
    texts: Sequence[str],
    prior: SpeechRate,
    client: TTSClient | None = None,
    voice: str | _Unspecified | None = UNSPECIFIED_VOICE,
    speed: float | None = None,
    narrate_stage_name: str | None = None,
    sample_lines: int | None = None,
) -> ScriptSpeechRate:
    """PHASE B: the rate THIS drafted script actually delivers at.

    Call it once the draft exists and BEFORE the narration verdict. It synthesizes a
    deterministic sample of the drafted lines (:func:`select_sample_indices`), divides
    the real words by the real seconds, and returns that as the rate to re-estimate,
    re-ask and trim against. The caller's existing loop absorbs the correction; this
    function decides nothing about the script and changes nothing in it.

    ``prior`` is the Phase-A :class:`SpeechRate` from
    :func:`resolve_natural_wpm_detail`. It is what the caller keeps when the sample
    cannot answer, and it is returned on :attr:`ScriptSpeechRate.prior` either way.

    ``voice`` / ``speed`` must be the ones the narrate stage will REQUEST. A sample
    taken at a different speed measures the multiplier, and one taken at a voice that
    never speaks measures a voice that never speaks — the cast-mode trap this module
    already documents, arriving in a second form.

    ``narrate_stage_name``, when given, stores each sampled clip under that stage's
    name so narrate reuses it instead of re-synthesizing (see
    :func:`_synthesize_sample_line`). **Only pass it when the reuse is real**: the
    sampled text must be the text narrate will speak, at the same speed. Pass ``None``
    and take the cost when it is not — a design contorted to save a synthesis, that
    ships the wrong audio, is far more expensive than four syntheses.

    **NEVER RAISES, and never substitutes silently.** Every failure returns ``prior``
    unchanged with :attr:`ScriptSpeechRate.reason` set, which the caller turns into
    :func:`script_sample_degradation`. A caller that ignores the result is
    byte-identical to its pre-Phase-B behaviour.

    The previous run of the same blueprint is NOT an input here and must not become
    one. Content differs per run, so a prior run's delivered rate is a stale proxy
    wearing a measurement's clothes — the exact class of error this module exists to
    end. Telemetry only.
    """
    try:
        return await _resolve_script_sample_wpm(
            ctx,
            texts=texts,
            prior=prior,
            client=client,
            voice=voice,
            speed=speed,
            narrate_stage_name=narrate_stage_name,
            sample_lines=sample_lines,
        )
    except Exception:
        logger.warning(
            "speech-rate script sample failed outright — keeping the prior rate", exc_info=True
        )
        return ScriptSpeechRate(rate=prior, prior=prior, reason=RESOLUTION_ERROR)


async def _resolve_script_sample_wpm(
    ctx: RunContext,
    *,
    texts: Sequence[str],
    prior: SpeechRate,
    client: TTSClient | None,
    voice: str | _Unspecified | None,
    speed: float | None,
    narrate_stage_name: str | None,
    sample_lines: int | None,
) -> ScriptSpeechRate:
    selection = _voice_selection(ctx)
    sample_voice = selection.voice if isinstance(voice, _Unspecified) else voice

    indices = select_sample_indices(texts, count=sample_size(len(texts), requested=sample_lines))
    if not indices:
        logger.warning(
            "speech-rate script sample: the draft offered no spoken line to sample — "
            "keeping the %.1f wpm %s rate",
            prior.wpm,
            prior.basis,
        )
        return ScriptSpeechRate(rate=prior, prior=prior, reason=SAMPLE_NO_LINES)

    run_id = str(ctx.run_id)
    sampled_texts = [str(texts[index]) for index in indices]
    key = _content_key(
        run_id=run_id,
        provider=selection.provider,
        voice=sample_voice,
        texts=sampled_texts,
        speed=speed,
    )
    cached = _content_measured.get(key)
    if cached is not None:
        # ⭐ WAVE 14, N2. Rebuilt with the SAME fields the miss path returns below —
        # see :class:`_ContentMeasurement` for why a bare wpm here published four zeros
        # that read as "no measurement was taken".
        return ScriptSpeechRate(
            rate=SpeechRate(wpm=cached.wpm, measured=True, basis=SCRIPT_SAMPLE),
            prior=prior,
            sampled_indices=indices,
            sampled_words=cached.sampled_words,
            sampled_seconds=cached.sampled_seconds,
            speech_seconds_split=cached.speech_seconds_split,
            pause_seconds_split=cached.pause_seconds_split,
            approx_pieces_split=cached.approx_pieces_split,
            lines_split=cached.lines_split,
        )

    totals = await _sample_totals(
        client or TTSClient(timeout=_PROBE_TIMEOUT_SECONDS),
        texts=texts,
        indices=indices,
        voice=sample_voice,
        provider=selection.provider,
        speed=speed,
        run_id=run_id,
        narrate_stage_name=narrate_stage_name,
    )

    if totals.lines < MIN_SAMPLE_LINES or totals.words <= 0 or totals.seconds <= 0.0:
        logger.warning(
            "speech-rate script sample measured only %d of %d line(s) (%d words in %.2fs) — "
            "keeping the %.1f wpm %s rate",
            totals.lines,
            len(indices),
            totals.words,
            totals.seconds,
            prior.wpm,
            prior.basis,
        )
        return ScriptSpeechRate(
            rate=prior,
            prior=prior,
            sampled_indices=indices,
            sampled_words=totals.words,
            sampled_seconds=totals.seconds,
            **_split_evidence(totals),
            reason=SAMPLE_TOO_FEW_MEASURED,
        )

    wpm = totals.words / (totals.seconds / 60.0)

    # DEGENERATE SAMPLE, NOT A BROKEN ONE. `totals.lines`/`words`/`seconds` are all
    # positive and `wpm` may sit comfortably inside the plausible band — the measurement
    # is a real number off real audio — but if the sampled lines average too few words
    # each, that number describes TTS's fixed per-utterance overhead amortised over a
    # handful of words, not this script's speaking rate. See
    # :data:`MIN_SAMPLE_AVG_WORDS_PER_LINE` for the production evidence (a 9.0 words/line
    # sample of `demo_video_gen`'s bdd-drafted scaffolding measured +39% high against
    # delivery). Checked BEFORE the plausibility band: a degenerate sample can land
    # anywhere in [60, 260] and still be wrong for this reason.
    avg_words_per_line = totals.words / totals.lines
    if avg_words_per_line < MIN_SAMPLE_AVG_WORDS_PER_LINE:
        logger.warning(
            "speech-rate script sample averaged only %.1f words/line (< %.1f) over %d "
            "line(s) — %.1f wpm is degenerate-scaffolding overhead, not a content rate; "
            "keeping the %.1f wpm %s rate",
            avg_words_per_line,
            MIN_SAMPLE_AVG_WORDS_PER_LINE,
            totals.lines,
            wpm,
            prior.wpm,
            prior.basis,
        )
        return ScriptSpeechRate(
            rate=prior,
            prior=prior,
            sampled_indices=indices,
            sampled_words=totals.words,
            sampled_seconds=totals.seconds,
            **_split_evidence(totals),
            reason=SAMPLE_LINES_TOO_SHORT,
        )

    # THE SAME band, unchanged and for the same reason: it rejects a BROKEN measurement
    # (truncated audio, a padded clip, a stub duration), never a plausible-but-slow one.
    # A content rate is EXPECTED to sit well under the passage rate — that gap is the
    # defect being fixed — so nothing here may be narrowed toward the passage's answer.
    if not (MIN_PLAUSIBLE_WPM <= wpm <= MAX_PLAUSIBLE_WPM):
        logger.warning(
            "speech-rate script sample produced an implausible %.1f wpm (%d words in %.2fs "
            "over %d line(s), outside [%.0f, %.0f]) voice=%s — rejecting the measurement",
            wpm,
            totals.words,
            totals.seconds,
            totals.lines,
            MIN_PLAUSIBLE_WPM,
            MAX_PLAUSIBLE_WPM,
            sample_voice,
        )
        return ScriptSpeechRate(
            rate=prior,
            prior=prior,
            sampled_indices=indices,
            sampled_words=totals.words,
            sampled_seconds=totals.seconds,
            **_split_evidence(totals),
            reason=IMPLAUSIBLE_RATE,
        )

    _remember_content_rate(key, wpm, totals)
    _log_two_term_decomposition(totals, wpm=wpm, voice=sample_voice)
    logger.info(
        "speech-rate script sample measured %.1f wpm (%d words in %.2fs over %d of %d "
        "line(s)) voice=%s provider=%s — the %s prior said %.1f wpm",
        wpm,
        totals.words,
        totals.seconds,
        totals.lines,
        len(indices),
        sample_voice,
        selection.provider,
        prior.basis,
        prior.wpm,
    )
    return ScriptSpeechRate(
        rate=SpeechRate(wpm=wpm, measured=True, basis=SCRIPT_SAMPLE),
        prior=prior,
        sampled_indices=indices,
        sampled_words=totals.words,
        sampled_seconds=totals.seconds,
        **_split_evidence(totals),
    )


def _log_two_term_decomposition(totals: _SampleTotals, *, wpm: float, voice: str | None) -> None:
    """Report the sample's SPEECH rate and its per-piece padding, separately.

    ⛔ DIAGNOSTIC ONLY — nothing branches on this and no budget reads it.

    A clip's duration is not ``words / rate``. It is approximately::

        words / speech_rate  +  n_pieces x per_piece_padding

    and the second term scales with SENTENCE COUNT, not word count, so a single
    words-per-minute scalar cannot express it: the same voice reading the same
    number of words in more sentences takes longer. Three delivered clips of one
    ``af_heart`` run measured 192 / 174 / 138 wpm over their SPEECH at speed 1.0
    while their padding held steady near 0.53s per piece — i.e. the padding term
    was the better-behaved of the two, and the rate term was not a constant
    either.

    That is why this logs the terms SEPARATELY instead of fitting them. A
    two-term budget was costed against those same three clips and, even granted
    perfect padding knowledge, spanned ``s*`` 0.93-1.11 — and 0.83-1.24 once the
    piece count is mis-predicted by +/-2, which the planner cannot avoid because
    the LLM chooses the sentence structure after the budget is written. Shipping
    it would trade an overrun for an underrun at the 0.80 rate floor, which is
    the same complaint from the other side. So this emits the evidence a real fit
    needs, from production, and changes no arithmetic until that fit exists.

    Silent when no sampled line reported the split — a backend that cannot answer
    produces no line here rather than a row of zeros that looks like data.

    ⭐ WAVE 14, N1 — THE ARITHMETIC REASON, which is stronger than the costing above and
    was not written down until now. At the BUDGET site the substitution is a NO-OP.
    ``narration_words_target`` runs SECONDS -> WORDS before any text exists, so the piece
    count must be PREDICTED, and the only predictor available is this sample's own
    pieces-per-word ratio. Substituting it gives::

        words/speech_wpm + per_piece x (pieces_per_word x words)
          == words x (60/speech_wpm + per_piece x pieces_per_word)
          == words x 60/padded_wpm

    identically, for every word count — because ``padded_wpm`` IS the sum of the two
    terms over this same sample. So a two-term budget built from these fields emits
    byte-identical numbers while looking like a fix; it can only differ by using a piece
    count that is a GUESS, which is the +/-2 case costed above. Pinned, so wave 15 does
    not have to re-derive it: ``tests/unit/test_two_term_budget_collapses_to_one_term.py``.

    ⭐ AND WHAT WOULD UNBLOCK IT. The orchestrator already sends bdd a per-beat WORD
    count (``demo_video.gen.narration_words_per_scenario``). A per-beat SENTENCE count
    alongside it turns the piece count from a prediction into a COMMITMENT, at which
    point the two-term budget is implementable and falsifiable. That is a
    bdd_playwright_service contract change, not an edit to this module, and a key added
    here before bdd honours it would be a payload nothing reads.
    """
    if totals.lines_split <= 0 or totals.speech_seconds_split <= 0.0:
        return
    speech_wpm = totals.words_split / (totals.speech_seconds_split / 60.0)
    pieces = max(totals.approx_pieces_split, 1)
    per_piece = totals.pause_seconds_split / pieces
    total_split = totals.speech_seconds_split + totals.pause_seconds_split
    pause_fraction = totals.pause_seconds_split / total_split if total_split > 0 else 0.0
    logger.info(
        "speech-rate two-term decomposition (DIAGNOSTIC, budgets nothing): "
        "speech %.1f wpm over %.2fs of speech vs the %.1f wpm this sample will "
        "budget from; padding %.2fs (%.1f%% of the audio) over ~%d approx piece(s) "
        "=> ~%.3fs per piece; %d line(s) reported the split voice=%s",
        speech_wpm,
        totals.speech_seconds_split,
        wpm,
        totals.pause_seconds_split,
        pause_fraction * 100.0,
        pieces,
        per_piece,
        totals.lines_split,
        voice,
    )


def script_sample_degradation(*, stage: str, sample: ScriptSpeechRate) -> Degradation:
    """One ``degradations`` entry for a script whose OWN rate could not be measured.

    Reuses ``degradation.SPEECH_RATE_UNUSABLE`` rather than minting a code, for that
    code's documented fact: the rate every duration number on this run was computed
    from is not a sound measurement OF THIS NARRATION. A fixed passage synthesizes ~47%
    faster than real lines, so a run that fell back to it was budgeted against text
    nobody is going to speak — the same operational consequence, and a query asking
    "which runs were budgeted blind?" wants this row in the same result set.

    Raises ``ValueError`` when the sample LANDED. A degradation channel that also
    carries successes is one people learn to ignore, and the caller has
    ``sample.reason`` to guard with.
    """
    if sample.reason is None:
        raise ValueError(
            f"{stage}: script_sample_degradation called for a sample that MEASURED "
            f"{sample.rate.wpm:.1f} wpm — guard the call with `sample.reason is not None`"
        )
    return degradation(
        SPEECH_RATE_UNUSABLE,
        f"{stage}: the narration rate could not be measured off this script's own lines "
        f"(reason={sample.reason}; {len(sample.sampled_indices)} line(s) selected, "
        f"{sample.sampled_words} word(s) in {sample.sampled_seconds:.2f}s usable), so every "
        f"duration number on this run was budgeted at the {sample.prior.wpm:.1f} wpm "
        f"{sample.prior.basis} rate, which does not predict this content.",
    )


# ---------------------------------------------------------------------------
# SINGLE OWNER — the ONE rate a run's word budget AND its timeline solver use
# ---------------------------------------------------------------------------
#
# The regression this exists to end: PR #992 added Phase B (this module's
# `resolve_script_sample_wpm`) but wired it into ONLY `reconcile_timeline` (the
# solver). `demo_video_gen`'s word budget kept asking bdd for a script sized at the
# Phase-A passage rate (175.2 wpm); the solver then judged that same script against
# the measured content rate (119.4 wpm) — TWO DIFFERENT NUMBERS for the same script,
# which is a worse defect than the one Phase B was built to fix ("UNDERFULL —
# content fills only 29.08s", clips pinned at the 0.545 floor). PR #998 reverted it.
#
# `resolve_effective_wpm` is the fix: the one function both `demo_video_gen`
# (budget) and `demo_video_reconcile_timeline` (solver) call, so there is no second
# place either can independently arrive at a different rate for the same script.
#
# ⛔ "THE SAME SCRIPT" MUST BE CHECKED, NOT ASSUMED. `demo_video_gen` samples the RAW
# bdd draft; `script_doctor` then rewrites every line before the solver runs. Run
# `01a0afab-24a3-7745-b8b9-9625966f31ca`: the draft embedded a quoted URL per line
# (`… application at "https://demo.playwright.dev/todomvc"`), which `split()` counts as
# ONE word while kokoro spells it out — measured on mini2 at speed 1.0, that line is
# 6.406s with the URL and 3.074s without it. The draft sampled at 111.7 wpm; the doctored
# lines (no URL) are delivered at ~150. The solver reused 111.7 as if it were a
# measurement of the doctored text, estimated clip-4 at 12.35s against a 10.28s clip,
# and planned a 1.5s freeze pad the voice then had to be dragged to fill.
#
# So the rate is stored WITH a digest of the exact texts it was sampled over
# (`spoken_text_digests`), and a consumer holding different text re-samples.


def spoken_texts(texts: Iterable[str]) -> list[str]:
    """The non-blank line texts a Phase-B sample is taken over, in script order.

    The ONE normalisation both the sampler's caller (``demo_video_gen``) and the
    read-back (``reconcile_timeline``) apply, so the same script yields the same
    ``texts`` argument — and therefore the same sample indices and the same content
    cache key — on both sides.
    """
    return [str(text) for text in texts if str(text).strip()]


def spoken_text_digests(texts: Iterable[str]) -> list[str]:
    """One stable digest per spoken line, in script order.

    Stored beside ``resolved_wpm`` so a later stage can tell whether it is about to
    budget text the rate was measured on. PER LINE, not one digest of the whole script,
    because ``demo_video_gen`` legitimately DROPS beats after sampling (relief /
    duration ceiling / feasibility floor): a subset of unchanged lines is still the
    text that was measured, while any rewrite of any line is not.
    """
    return [hashlib.sha256(text.encode()).hexdigest()[:40] for text in spoken_texts(texts)]


async def resolve_effective_wpm(
    ctx: RunContext,
    *,
    texts: Sequence[str],
    client: TTSClient | None = None,
    voice: str | _Unspecified | None = UNSPECIFIED_VOICE,
    speed: float | None = None,
    default: float = DEFAULT_WPM_NATURAL,
    narrate_stage_name: str | None = None,
    sample_lines: int | None = None,
) -> ScriptSpeechRate:
    """THE rate to budget/solve this run's narration with. Composes the whole ladder.

    1. **Authored wins outright.** :func:`authored_natural_wpm` — an author who wrote
       ``direction.timing.wpm.natural`` gets exactly that rate, no probe made, on
       every call regardless of ``texts``.
    2. **No script yet ⇒ Phase A only.** ``texts`` empty means the draft has not been
       written (this IS the chicken-and-egg: a script cannot be sampled before it
       exists), so this is exactly :func:`resolve_natural_wpm_detail` against
       :data:`CALIBRATION_TEXT` — a PRIOR to size the ask, not a prediction.
    3. **A script exists ⇒ Phase B.** ``texts`` non-empty measures a sample of THOSE
       lines (:func:`resolve_script_sample_wpm`), which is the rate this content will
       actually be spoken at.

    Callers that need Phase A's own value even after a Phase-B call lands (to report
    both, or to seed :func:`resolve_script_sample_wpm`'s ``prior``) find it on
    :attr:`ScriptSpeechRate.prior` — this function has already done both probes by
    the time it returns.

    ``voice``/``speed``/``narrate_stage_name``/``sample_lines`` pass straight through
    to whichever phase runs; see their docstrings on :func:`resolve_natural_wpm_detail`
    and :func:`resolve_script_sample_wpm`.

    Never raises: every rung underneath swallows its own failures and returns a usable
    ``default``-backed rate.

    ⚠️ Cast-mode (several distinct voices in one script) is NOT modelled here — this
    function measures ONE voice/rate for the whole ``texts`` argument. A cast budget
    is :func:`resolve_cast_wpm`'s job (word-weighted per distinct voice); a caller
    solving a cast keeps using that function directly and reserves this one for the
    single global rate that a word-count budget is inherently expressed in.
    """
    authored = authored_natural_wpm(ctx)
    if authored is not None:
        rate = SpeechRate(wpm=authored, measured=True, basis=AUTHORED)
        return ScriptSpeechRate(rate=rate, prior=rate)

    prior = await resolve_natural_wpm_detail(ctx, client=client, default=default, voice=voice)
    if not texts:
        return ScriptSpeechRate(rate=prior, prior=prior)

    return await resolve_script_sample_wpm(
        ctx,
        texts=texts,
        prior=prior,
        client=client,
        voice=voice,
        speed=speed,
        narrate_stage_name=narrate_stage_name,
        sample_lines=sample_lines,
    )


# ---------------------------------------------------------------------------
# PHASE C — measure EVERY line, so the estimate stops being an estimate
# ---------------------------------------------------------------------------
#
# Phase B samples ``k`` lines and pools them into a rate. That rate is then applied to
# every OTHER line's word count, which re-imports the error it was built to remove: a
# 4-line sample carries ~5% standard error, and the delivered durations measured
# tonight (n=5 per type) still scattered -20.8%..+12.9% around the ordered length.
#
# The sampling error is unnecessary, because of two facts this repo already relies on:
#
# 1. **A film IS exactly its narration.** An 83.065s film was 41.410 + 41.655s of
#    narrate clips, to the millisecond; the compiler adds nothing.
# 2. **The record stage REUSES clips.** ``deterministic_clip_id`` keys on
#    voice + speed + preprocessing + text, so a clip synthesized here under the
#    narrate stage's own name IS the clip narrate will find and play — the same
#    property :func:`_synthesize_sample_line` already exploits for the sample.
#
# So synthesizing EVERY line rather than four of them replaces the estimate with the
# answer, and costs only the wall clock of doing earlier what the run was going to do
# anyway (~+10s, measured): the syntheses are not extra, they are moved.
#
# This is ADDITIVE. :func:`resolve_script_sample_wpm` and everything it uses are
# untouched — a caller that wants the cheap sample still gets exactly the old bytes.
#
# What is NOT assumed here: that every line answers. A line that fails, times out, or
# reports a duration that is not a positive finite number is EXTRAPOLATED from the
# lines that did answer — preferring the rate of the SAME VOICE, because per-voice
# rate differences are the largest term this module has ever measured and pooling a
# narrator's rate onto a character's line reintroduces the error on purpose.

#: The pod-wide ceiling on in-flight synthesize calls, read exactly as
#: ``tts_client._MAX_CONCURRENT_SYNTHESIZE`` reads it. NOT imported from there: that
#: module snapshots the value at import time into a private name, and this one needs
#: the CURRENT setting to size a timeout (and a test needs to be able to move it).
_DEFAULT_TTS_GATE: Final = 3


def _tts_gate_size() -> int:
    """How many of these syntheses can be in flight at once, pod-wide.

    A value that does not parse, or that is not a positive count, is GARBAGE and
    falls back to :data:`_DEFAULT_TTS_GATE` — not to ``1``. Clamping junk to 1 would
    silently turn every per-line timeout into ``timeout_per_line * (1 + position)``,
    i.e. a queue-depth test on a gate the pod does not actually have, which is the
    precise failure this multiplier exists to prevent.
    """
    raw = os.getenv("TTS_MAX_CONCURRENT_SYNTHESIZE")
    if raw is None:
        return _DEFAULT_TTS_GATE
    try:
        size = int(raw)
    except ValueError:
        return _DEFAULT_TTS_GATE
    return size if size >= 1 else _DEFAULT_TTS_GATE


def _spoken_words(text: object) -> int:
    """Words a TTS engine will actually say for ``text``. Zero for blank."""
    return len(str(text).split())


def _seconds_at(words: int, wpm: float) -> float:
    """Seconds ``words`` take at ``wpm``. Zero rather than NaN/inf on a junk rate."""
    if words <= 0 or not math.isfinite(wpm) or wpm <= 0.0:
        return 0.0
    return words * 60.0 / wpm


@dataclass(frozen=True, slots=True)
class ScriptMeasurement:
    """Real seconds for every line of a script — measured, not inferred.

    :attr:`total_seconds` is the sum of :attr:`per_line_seconds`, and when
    :attr:`basis` is ``measured_all`` every one of those numbers came off a real
    synthesis the record stage will reuse. That makes it the DELIVERED duration of
    the film, not a projection of it.

    :attr:`basis` is the only thing to branch on. ``measured_all`` /
    ``measured_partial`` mean Phase C landed; anything else means it did not and
    :attr:`reason` says which fact stopped it — in which case every non-blank line's
    seconds is :attr:`prior`-derived and no better than the pre-Phase-C estimate.
    """

    #: One entry per line of ``texts``, in script order. Measured seconds for a line
    #: that answered, ``0.0`` for a blank line, an extrapolation otherwise.
    per_line_seconds: tuple[float, ...]
    #: Per line: is :attr:`per_line_seconds` a FACT? True for a line that synthesized,
    #: and for a blank line (whose 0.0s is exact, not estimated). False for every
    #: extrapolated line — including, on a refused measurement, lines that did
    #: synthesize but whose numbers were discarded with the rest.
    measured: tuple[bool, ...]
    #: Non-blank lines whose synthesis raised, timed out, or reported an unusable
    #: duration. These are the lines that cost real wall clock and bought nothing.
    failed_indices: tuple[int, ...]
    #: Lines whose seconds is derived from a rate rather than measured. Equals
    #: :attr:`failed_indices` when the measurement landed; on a refusal it is EVERY
    #: non-blank line, which is exactly how the two differ and why both exist.
    extrapolated_indices: tuple[int, ...]
    #: ``sum(per_line_seconds)``. The number a duration verdict is taken against.
    total_seconds: float
    #: How many non-blank lines yielded a usable duration from TTS. Raw truth: it is
    #: NOT reduced to zero when those measurements are refused, so a reader can tell
    #: "the backend was dead" from "the backend answered and the answer was absurd".
    measured_count: int
    #: The rate :attr:`per_line_seconds` is expressed at — pooled over the measured
    #: lines when the measurement landed, ``prior.wpm`` when it did not.
    wpm: float
    #: See :data:`RateBasis`. ``measured_all`` / ``measured_partial`` on success,
    #: ``prior.basis`` unchanged on a refusal.
    basis: str
    #: The Phase-A/Phase-B rate as handed in, kept so a caller can report BOTH.
    prior: SpeechRate
    #: Why Phase C did not land; ``None`` when it did.
    reason: FallbackReason | None = None

    @property
    def complete(self) -> bool:
        """True only when every spoken line's seconds came off a real synthesis."""
        return self.basis == BASIS_MEASURED_ALL


async def _measure_one_line(
    client: TTSClient,
    *,
    text: str,
    voice: str | None,
    provider: str | None,
    speed: float | None,
    run_id: str,
    stage_name: str,
    timeout: float,
) -> float:
    """One line's real seconds, bounded by its OWN timeout.

    Per line, deliberately: one ``wait_for`` over the whole fan-out would let one slow
    line lose every other line's measurement with it. ``_sample_totals`` bounds its
    lines the same way.

    Returns the DURATION only. ``_synthesize_sample_line`` also carries
    tts_backend's speech/pause split now, but this path (``measure_script_lines``)
    is a per-line duration census with its own contract and callers, and widening
    it is a separate change — the split is consumed by ``_sample_totals``, which
    is where the two-term decomposition is reported.
    """
    measurement = await asyncio.wait_for(
        _synthesize_sample_line(
            client,
            text=text,
            voice=voice,
            provider=provider,
            speed=speed,
            run_id=run_id,
            narrate_stage_name=stage_name,
        ),
        timeout=timeout,
    )
    return measurement.duration


def _per_voice_rates(
    *,
    measured_seconds: Mapping[int, float],
    words: Sequence[int],
    voices: Sequence[str | None],
    pooled: float,
) -> dict[str | None, float]:
    """Words-per-minute per VOICE, over that voice's own measured lines.

    A cast's voices differ by far more than the sampling error this module exists to
    remove, so a failed character line filled from the narrator's rate reintroduces
    the error on purpose. A voice with no usable measurement is simply absent from
    the result — the caller falls back to ``pooled``, which is the only honest answer
    left and is never zero at the point this is called.
    """
    voice_words: dict[str | None, int] = {}
    voice_seconds: dict[str | None, float] = {}
    for index, seconds in measured_seconds.items():
        voice = voices[index]
        voice_words[voice] = voice_words.get(voice, 0) + words[index]
        voice_seconds[voice] = voice_seconds.get(voice, 0.0) + seconds
    rates = {
        voice: count / (voice_seconds[voice] / 60.0)
        for voice, count in voice_words.items()
        if count > 0 and voice_seconds[voice] > 0.0
    }
    return rates or dict.fromkeys(voice_words, pooled)


def _collect_durations(
    indices: Sequence[int], outcomes: Sequence[float | BaseException]
) -> tuple[dict[int, float], list[int]]:
    """Split the fan-out's results into usable durations and failed line indices.

    ``asyncio.CancelledError`` is RE-RAISED rather than counted as a failure:
    ``return_exceptions=True`` collects a child's cancellation like any other
    exception, and a cancelled run must neither proceed to report a measurement nor
    be told its cancellation was a TTS fault.
    """
    measured: dict[int, float] = {}
    failed: list[int] = []
    for index, outcome in zip(indices, outcomes, strict=True):
        if isinstance(outcome, asyncio.CancelledError):
            raise outcome
        if isinstance(outcome, BaseException):
            logger.warning(
                "speech-rate full-script measurement: line %d failed to synthesize "
                "(%s: %s) — it will be extrapolated from the lines that did",
                index,
                type(outcome).__name__,
                outcome,
            )
            failed.append(index)
            continue
        duration = float(outcome)
        if not math.isfinite(duration) or duration <= 0.0:
            logger.warning(
                "speech-rate full-script measurement: line %d reported an unusable "
                "duration=%r — extrapolating it instead",
                index,
                duration,
            )
            failed.append(index)
            continue
        measured[index] = duration
    return measured, failed


def _prior_measurement(
    *,
    texts: Sequence[str],
    prior: SpeechRate,
    reason: FallbackReason,
    failed_indices: tuple[int, ...] = (),
    measured_count: int = 0,
) -> ScriptMeasurement:
    """Phase C could not answer: every spoken line falls back to the prior rate.

    Byte-for-byte the estimate the caller would have computed without Phase C, so a
    caller that ignores :attr:`ScriptMeasurement.reason` is no worse off than before —
    but the reason is there, and :func:`script_measurement_degradation` turns it into
    a row somebody can query.
    """
    per_line = tuple(_seconds_at(_spoken_words(text), prior.wpm) for text in texts)
    blank = tuple(_spoken_words(text) == 0 for text in texts)
    return ScriptMeasurement(
        per_line_seconds=per_line,
        measured=blank,
        failed_indices=failed_indices,
        extrapolated_indices=tuple(i for i, is_blank in enumerate(blank) if not is_blank),
        total_seconds=math.fsum(per_line),
        measured_count=measured_count,
        wpm=prior.wpm,
        basis=prior.basis,
        prior=prior,
        reason=reason,
    )


async def measure_script_lines(
    ctx: RunContext,
    *,
    texts: Sequence[str],
    voices: Sequence[str | None],
    prior: SpeechRate,
    speed: float | None,
    stage_name: str,
    client: TTSClient | None = None,
    timeout_per_line: float = _PROBE_TIMEOUT_SECONDS,
) -> ScriptMeasurement:
    """PHASE C: synthesize EVERY line and return what the film will really run to.

    ``voices`` is parallel to ``texts`` — one voice per line, ``None`` meaning
    tts_backend's own default. A length mismatch is a CALLER bug, and it is reported
    as a fallback rather than raised: a mis-zipped cast must not fail a paid run.

    ``speed`` and ``stage_name`` must be the ones the narrate/record stage will
    REQUEST. They are what makes ``deterministic_clip_id`` collide, and the collision
    is the entire economic argument: a clip measured here is the clip delivered
    later, so the measurement costs the synthesis nobody was going to avoid. Pass a
    speed or a stage name narrate will not use and every synthesis here is genuinely
    wasted AND the numbers describe audio nobody will hear.

    **NEVER RAISES** (``asyncio.CancelledError`` excepted — cancellation is not a
    failure to absorb, and swallowing it would let a cancelled run keep synthesizing).
    Every other failure returns a prior-derived measurement with
    :attr:`ScriptMeasurement.reason` set.
    """
    # Materialized BEFORE anything else, and separately, because the recovery path is
    # not exempt from the guarantee: `_prior_measurement` iterates `texts`, so a
    # `texts` that is not a sequence at all would escape the handler below as a
    # TypeError raised from inside the handler itself. Separately, so that a bad
    # `voices` still yields the full-length prior estimate the caller would have
    # computed on its own. `except Exception` does not catch `asyncio.CancelledError`
    # (it is a BaseException), so a cancelled run is still cancelled here.
    try:
        safe_texts: tuple[str, ...] = tuple(texts)
    except Exception:
        logger.warning(
            "speech-rate full-script measurement got a non-iterable texts=%r — keeping "
            "the %.1f wpm %s prior and measuring nothing",
            type(texts).__name__,
            prior.wpm,
            prior.basis,
            exc_info=True,
        )
        return _prior_measurement(texts=(), prior=prior, reason=RESOLUTION_ERROR)
    try:
        safe_voices: tuple[str | None, ...] = tuple(voices)
    except Exception:
        logger.warning(
            "speech-rate full-script measurement got a non-iterable voices=%r — keeping "
            "the %.1f wpm %s prior for every line",
            type(voices).__name__,
            prior.wpm,
            prior.basis,
            exc_info=True,
        )
        return _prior_measurement(texts=safe_texts, prior=prior, reason=RESOLUTION_ERROR)

    try:
        return await _measure_script_lines(
            ctx,
            texts=safe_texts,
            voices=safe_voices,
            prior=prior,
            speed=speed,
            stage_name=stage_name,
            client=client,
            timeout_per_line=timeout_per_line,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning(
            "speech-rate full-script measurement failed outright — keeping the %.1f wpm "
            "%s prior for every line",
            prior.wpm,
            prior.basis,
            exc_info=True,
        )
        return _prior_measurement(texts=safe_texts, prior=prior, reason=RESOLUTION_ERROR)


async def _measure_script_lines(
    ctx: RunContext,
    *,
    texts: Sequence[str],
    voices: Sequence[str | None],
    prior: SpeechRate,
    speed: float | None,
    stage_name: str,
    client: TTSClient | None,
    timeout_per_line: float,
) -> ScriptMeasurement:
    if len(texts) != len(voices):
        logger.warning(
            "speech-rate full-script measurement got %d line(s) and %d voice(s) — a caller "
            "bug; keeping the %.1f wpm %s prior rather than measuring the wrong voice",
            len(texts),
            len(voices),
            prior.wpm,
            prior.basis,
        )
        return _prior_measurement(texts=texts, prior=prior, reason=RESOLUTION_ERROR)

    words = [_spoken_words(text) for text in texts]

    # A script with NO spoken line is measured EXACTLY, not refused. Every line is
    # 0.0s by arithmetic, the total is 0.0s, and there was never anything to
    # synthesize — so the answer is complete and `measured_all`. Falling through to
    # the `measured_count < MIN_SAMPLE_LINES` gate below would instead stamp
    # `sample_too_few_measured` and raise a degradation reading "0 of 0 spoken line(s)
    # usable": a signal that fires on a CORRECT result, which is no signal at all.
    if not any(count > 0 for count in words):
        logger.info(
            "speech-rate full-script measurement: %d line(s), none of them spoken — "
            "0.00s exactly, nothing to synthesize",
            len(words),
        )
        return ScriptMeasurement(
            per_line_seconds=(0.0,) * len(words),
            measured=(True,) * len(words),
            failed_indices=(),
            extrapolated_indices=(),
            total_seconds=0.0,
            measured_count=0,
            wpm=prior.wpm,
            basis=BASIS_MEASURED_ALL,
            prior=prior,
        )

    provider = _voice_selection(ctx).provider
    run_id = str(ctx.run_id)
    tts = client or TTSClient(timeout=_PROBE_TIMEOUT_SECONDS)
    gate = _tts_gate_size()

    # A blank line is 0.0s by arithmetic, not by measurement: it has no words, so it
    # can neither take time nor contribute to a rate. Synthesizing one buys a clip
    # nobody hears and, worse, occupies a slot behind the gate that a real line needs.
    spoken = [index for index, count in enumerate(words) if count > 0]

    # The gate admits `gate` syntheses at a time, so the line at queue position `p`
    # waits for roughly `p // gate` batches ahead of it before its own synthesis even
    # starts. A flat per-line timeout would therefore not be a timeout at all — it
    # would be a queue-depth test, failing the tail of every long script while the
    # backend was working perfectly. Position is counted over the lines actually
    # SUBMITTED (blank ones never queue), which is the same as the script index
    # whenever the script has no blank lines.
    pending = [
        _measure_one_line(
            tts,
            text=str(texts[index]),
            voice=voices[index],
            provider=provider,
            speed=speed,
            run_id=run_id,
            stage_name=stage_name,
            timeout=timeout_per_line * (1 + position // gate),
        )
        for position, index in enumerate(spoken)
    ]
    outcomes = await asyncio.gather(*pending, return_exceptions=True)

    measured_seconds, failed = _collect_durations(spoken, outcomes)

    measured_count = len(measured_seconds)
    pooled_words = sum(words[index] for index in measured_seconds)
    pooled_seconds = math.fsum(measured_seconds.values())
    pooled = (
        pooled_words / (pooled_seconds / 60.0) if pooled_words > 0 and pooled_seconds > 0.0 else 0.0
    )

    # The SAME two gates Phase B applies, for the same reasons and at the same
    # thresholds: below MIN_SAMPLE_LINES a "rate" is one clip's rate, and outside the
    # plausible band the audio path is broken rather than slow. Nothing is narrowed
    # toward the prior — a content rate is EXPECTED to sit well under the passage's.
    #
    # The too-few gate guards EXTRAPOLATION: a rate pooled off one clip must not be
    # applied to lines that did not answer. When EVERY spoken line answered there is
    # nothing to extrapolate — the per-line seconds ARE the delivered audio — so a
    # one-line script is measured, not refused. Run 01a156a8-978b (2026-10-20): a
    # 60s lesson drew ONE 183-word paragraph, it synthesized fine, and this gate
    # threw that clip away for the 197.6 wpm calibration prior (estimate ~56s, "in
    # band"); the lesson delivered 70.37s (+17.3%) with SPEECH_RATE_UNUSABLE.
    if measured_count < MIN_SAMPLE_LINES and failed:
        logger.warning(
            "speech-rate full-script measurement usable on only %d of %d spoken line(s) — "
            "keeping the %.1f wpm %s prior for every line",
            measured_count,
            len(spoken),
            prior.wpm,
            prior.basis,
        )
        return _prior_measurement(
            texts=texts,
            prior=prior,
            reason=SAMPLE_TOO_FEW_MEASURED,
            failed_indices=tuple(failed),
            measured_count=measured_count,
        )
    if not (MIN_PLAUSIBLE_WPM <= pooled <= MAX_PLAUSIBLE_WPM):
        logger.warning(
            "speech-rate full-script measurement produced an implausible %.1f wpm "
            "(%d words in %.2fs over %d line(s), outside [%.0f, %.0f]) — rejecting it",
            pooled,
            pooled_words,
            pooled_seconds,
            measured_count,
            MIN_PLAUSIBLE_WPM,
            MAX_PLAUSIBLE_WPM,
        )
        return _prior_measurement(
            texts=texts,
            prior=prior,
            reason=IMPLAUSIBLE_RATE,
            failed_indices=tuple(failed),
            measured_count=measured_count,
        )

    # Per-voice rates, for extrapolating the lines that did not answer. A cast's
    # voices differ by far more than the sampling error this whole file is about, so
    # a failed character line is filled from that CHARACTER's measured rate whenever
    # one exists, and only falls through to the pooled rate when it does not.
    rates = _per_voice_rates(
        measured_seconds=measured_seconds, words=words, voices=voices, pooled=pooled
    )
    # `is not None` rather than truthiness: a measured 0.0 is already rejected above,
    # but a dict lookup that silently re-derives a falsy measurement is exactly the
    # kind of thing a later edit turns into a bug nobody can see.
    per_line = tuple(
        measured_seconds[index]
        if index in measured_seconds
        else _seconds_at(count, rates.get(voices[index], pooled))
        for index, count in enumerate(words)
    )
    flags = tuple(count == 0 or index in measured_seconds for index, count in enumerate(words))

    basis = BASIS_MEASURED_PARTIAL if failed else BASIS_MEASURED_ALL
    total = math.fsum(per_line)
    logger.info(
        "speech-rate full-script measurement: %d of %d spoken line(s) measured, "
        "%.2fs total at %.1f wpm (basis=%s) — the %s prior said %.1f wpm",
        measured_count,
        len(spoken),
        total,
        pooled,
        basis,
        prior.basis,
        prior.wpm,
    )
    return ScriptMeasurement(
        per_line_seconds=per_line,
        measured=flags,
        failed_indices=tuple(failed),
        extrapolated_indices=tuple(failed),
        total_seconds=total,
        measured_count=measured_count,
        wpm=pooled,
        basis=basis,
        prior=prior,
    )


def script_measurement_degradation(
    measurement: ScriptMeasurement, *, stage: str
) -> Degradation | None:
    """One ``degradations`` entry for a script Phase C could not measure — or ``None``.

    Unlike :func:`script_sample_degradation` this RETURNS ``None`` on success rather
    than raising, because the caller's shape is different: Phase C runs on every
    narrated run and its result is threaded straight into the verdict, so an
    unconditional call site with no guard is the correct one and a raise there would
    turn a measured run into a crash.

    Reuses ``degradation.SPEECH_RATE_UNUSABLE`` for the same documented fact the
    other two entry points reuse it for: every duration number on this run was
    computed from a rate nobody measured OF THIS NARRATION. A partial measurement is
    NOT degraded — its few extrapolated lines are extrapolated from this script's own
    measured rate, which is the thing being asked for.
    """
    if measurement.reason is None:
        return None
    # On a refusal EVERY spoken line is extrapolated, so this count IS the spoken-line
    # count — derived from the one field that is unambiguous here rather than by
    # re-deriving blankness from a 0.0 that a real (if absurd) clip could also produce.
    spoken = len(measurement.extrapolated_indices)
    return degradation(
        SPEECH_RATE_UNUSABLE,
        f"{stage}: the narration duration could not be measured off this script's own "
        f"lines (reason={measurement.reason}; {measurement.measured_count} of {spoken} "
        f"spoken line(s) usable, {len(measurement.failed_indices)} failed), so every "
        f"duration number on this run was budgeted at the {measurement.prior.wpm:.1f} wpm "
        f"{measurement.prior.basis} rate, which does not predict this content.",
    )
