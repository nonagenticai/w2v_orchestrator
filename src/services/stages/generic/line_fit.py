"""Per-line closed loop: fit every narration line to its own share of the order.

A generic film IS its narration (the compiler adds ~0.1-0.17s), so the film is
the ordered length exactly when the sum of the clips narrate will play is. Until
this module, ``script_draft`` judged length on a POOLED rate projected onto word
counts — and below 10 words/line on average it threw the measurement away for the
calibration prior (run 61ae: 158 wpm assumed vs 138 spoken, estimate 31.1s,
delivered 35.63s, +19%). Every line had already been synthesized; the real
per-line durations were simply discarded into an average.

This loop keeps them. Each line gets a seconds BUDGET (its word share of the
order), its real synthesized duration is compared to it, and:

* within :data:`ACCEPT_BAND` (+/-5%) the clip is kept as synthesized;
* within :data:`SPEED_BAND` (+/-12%) it is re-synthesized ONCE at a speed clamped
  to [:data:`SPEED_MIN`, :data:`SPEED_MAX`] — the speed is returned so it can be
  persisted on the line, and ``deterministic_clip_id`` (voice + speed + text) makes
  that exact clip the one narrate plays;
* further out, the LINE ALONE is rewritten by the LLM, told its measured length and
  the exact words to add or cut (at most :data:`MAX_REWRITES_PER_LINE`), and only
  then speed-clamped.

Selection ranks EVERY measured take (the first and each rewrite) by the miss it can
reach after the speed clamp, not by its raw miss — and once one take of a line lands
under its budget and another over, the next rewrite's word target is interpolated
from those two measurements (secant on the asked word count) instead of re-asking the
target that just overshot. Live 2026-10-21 (``01a15e2b-2a88``, ``01a15e77-e3a4``):
every rewrite overshot, the second re-ask was byte-identical to the first, and the
line fell back to its original text at the 0.9 floor (-14% to -19%).

The speed response is CALIBRATED, not assumed. kokoro (tts_backend) does not shorten a
clip as ``1/speed``: 20 measured same-text speed takes (2026-10-20/21, speeds
0.900-1.100) fit ``seconds(s) = seconds(1) * s**-k`` with k = 0.60 (rms residual 0.9%,
max 2.2%; the ``1/speed`` model it replaces: rms 3.7%, max 5.4%). Fixed pauses do NOT
explain it: tts_backend's spliced sentence gaps (its ``pause_seconds``) are 0-11% of
those clips and do stay fixed, but subtracting them leaves the same k (0.62) — the
spoken words themselves stretch less than asked. So the [0.9, 1.1] clamp reaches only
about +6.5% / -5.6% of length, not +11% / -9%, and a speed take the ``1/speed`` model
predicted on target landed +2.6% to +7.4% over (``01a15e77-e3a4`` clip-4, clip-5).
Two defences:
:data:`SPEED_EXPONENT` in every prediction (:func:`speed_seconds`, :func:`fit_speed`),
and ONE corrective re-synthesis (:func:`corrective_speed`) when the measured speed take
still misses :data:`ACCEPT_BAND`, using the exponent THAT line just exhibited. The
shipped clip is always a measured one.

The residual is absorbed on the last :data:`ABSORB_LINES` lines: they are budgeted
from what the earlier lines ACTUALLY came to, not from the plan.

Two entry points over the same per-line rule (:func:`fit_line`):

* :func:`fit_script` — generic: budgets are each line's word share of the ORDER, and
  the tail absorbs the head's residual;
* :func:`fit_lines` — demo_video / showcase (``reconcile_timeline``): every line has
  its OWN budget, the footage window it narrates, so there is nothing to absorb.

Pure orchestration over two injected callables (``synth`` and ``rewrite``), so the
arithmetic is testable without TTS or an LLM. Billing bound per line: one TTS
synthesis per rewrite (<= 2), plus one speed re-synthesis and at most one corrective
speed re-synthesis; one ``rewrite`` call per rewrite (generic's makes a single model
call; showcase's makes at most two — one recount re-ask when the reply misses its word
count, see ``narration_fit.FIT_WORD_TOLERANCE``).
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

from src.services.stages.verbatim import find_missing_verbatim

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

logger = logging.getLogger(__name__)

#: A line within this fraction of its budget is kept exactly as synthesized.
ACCEPT_BAND: Final = 0.05
#: A line within this fraction (and outside :data:`ACCEPT_BAND`) is re-synthesized
#: once at a corrected speed; further out it is rewritten first.
SPEED_BAND: Final = 0.12
#: The speed clamp (owner-approved). tts_backend accepts 0.5-2.0; outside +/-10% the
#: voice audibly drags or rushes, so a bigger miss is a WORDING fault, not a rate one.
SPEED_MIN: Final = 0.9
SPEED_MAX: Final = 1.1
#: How kokoro's length answers a speed: ``seconds(s) = seconds(b) * (b / s) ** k``.
#: Least squares over 20 measured same-text takes (speeds 0.900-1.100, generic +
#: showcase, 2026-10-20/21): k = 0.591 on the whole clip, 0.616 on the clip minus its
#: spliced pauses (runtime subtracts ``pause_seconds`` when reported); 0.60 is within
#: 0.05% rms of the best fit either way. Slow-only 0.603, fast-only 0.561 — one k holds
#: both directions. Per-take k spreads 0.42-0.83 (rms residual 0.9%, max 2.2%).
SPEED_EXPONENT: Final = 0.60
#: A line's own measured exponent is trusted for the corrective re-synth only inside
#: this range; outside it (a narrator that ignored or inverted the speed) no extra call.
#: Inside the measured 0.42-0.83 spread the first speed take already lands in band, so
#: the corrective call is a guard for an outlier line, not a routine second synthesis.
LOCAL_EXPONENT_RANGE: Final = (0.2, 1.5)
#: LLM rewrites one line may spend before the speed clamp takes what is left.
MAX_REWRITES_PER_LINE: Final = 2
#: How many trailing lines absorb the residual of the lines before them.
ABSORB_LINES: Final = 2
#: A trailing line's re-budget never drops below this fraction of its nominal
#: budget: an over-running head must not hand the closer a 0.0s slot.
MIN_ABSORB_FRACTION: Final = 0.5

Band = Literal["accept", "speed", "rewrite"]


@dataclass(frozen=True, slots=True)
class LineTake:
    """One synthesized clip of one line: its text, the speed asked for, its length.

    ``speed`` is the REQUESTED speed (``None`` = backend default, never an invented
    1.0) — the value narrate must request to reuse this exact clip.
    ``pause_seconds`` is tts_backend's report of the FIXED silence in the clip
    (``None`` when not reported, e.g. a reused clip), which a speed cannot move.
    """

    text: str
    seconds: float
    speed: float | None = None
    pause_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class LineFit:
    """The outcome for one line — the take to ship, and how it was reached."""

    index: int
    take: LineTake
    budget: float
    #: The band of the FIRST take — what the line needed, before any correction.
    initial_band: Band
    rewrites: int = 0
    resynthesized: bool = False
    #: A second, corrective speed take was synthesized (see :func:`corrective_speed`).
    corrected: bool = False
    #: Every measured candidate text, in the order it was synthesized.
    attempts: tuple[Attempt, ...] = ()

    @property
    def miss(self) -> float:
        return miss_ratio(self.take.seconds, self.budget)


def verbatim_rewritable(texts: Sequence[str], items: Sequence[dict]) -> list[bool]:
    """Per line: may the per-line fit rewrite it? Never a line holding a verbatim
    phrase — and when a phrase is held by no single line (it spans lines, or is
    absent) NO line is rewritten (the same rule generic's ``plan_line_trim`` pins by)."""
    pinned: set[int] = set()
    for item in items:
        holders = {li for li, t in enumerate(texts) if not find_missing_verbatim(t, [item])}
        pinned |= holders or set(range(len(texts)))
    return [li not in pinned for li in range(len(texts))]


def spoken_words(text: str) -> int:
    return len(str(text).split())


def line_budgets(texts: Sequence[str], ordered_seconds: float) -> list[float]:
    """Each line's share of the order, proportional to its spoken words.

    The word share is the line's share of the CONTENT; a line whose clip runs long
    for its words is exactly the line that should give time back. Blank lines get
    0.0 (they are never spoken). Sums to ``ordered_seconds`` whenever any line
    speaks and the order is positive.
    """
    words = [spoken_words(t) for t in texts]
    total = sum(words)
    if total <= 0 or not math.isfinite(ordered_seconds) or ordered_seconds <= 0.0:
        return [0.0] * len(words)
    return [ordered_seconds * w / total for w in words]


def miss_ratio(seconds: float, budget: float) -> float:
    """Signed miss: +0.10 = 10% over budget. 0.0 for a non-positive budget."""
    if budget <= 0.0 or not math.isfinite(seconds):
        return 0.0
    return (seconds - budget) / budget


def band_for(seconds: float, budget: float) -> Band:
    miss = abs(miss_ratio(seconds, budget))
    if miss <= ACCEPT_BAND:
        return "accept"
    if miss <= SPEED_BAND:
        return "speed"
    return "rewrite"


def clamp_speed(speed: float) -> float:
    """Into [:data:`SPEED_MIN`, :data:`SPEED_MAX`], rounded to the clip id's 3 decimals.

    Rounded because ``deterministic_clip_id`` formats speed to 3 decimals: two
    floats that differ past the third would still name the same clip, but the
    PERSISTED value must be the one the id was built from, so the round trip
    through JSON cannot drift it.
    """
    return round(min(max(speed, SPEED_MIN), SPEED_MAX), 3)


def _fixed(take: LineTake, *others: float) -> float:
    """Reported fixed silence, or 0.0 when absent or not smaller than every length."""
    fixed = take.pause_seconds if take.pause_seconds and take.pause_seconds > 0 else 0.0
    if fixed and all(v - fixed > 0 for v in (take.seconds, *others)):
        return fixed
    return 0.0


def speed_multiplier(
    moving_seconds: float, moving_budget: float, exponent: float = SPEED_EXPONENT
) -> float:
    """Factor on the current speed that moves ``moving_seconds`` onto ``moving_budget``.

    The ONE inversion of kokoro's calibrated response ``seconds ∝ speed ** -k``:
    ``(moving_seconds / moving_budget) ** (1/k)``. Both arguments are the part of the
    clip a speed CAN move (fixed silence already subtracted) and must be positive —
    the caller owns that guard. Shared by :func:`speed_for` and narrate's
    ``_fit_speed`` so the two fits cannot disagree about the same voice.
    """
    return (moving_seconds / moving_budget) ** (1.0 / exponent)


def speed_for(take: LineTake, budget: float, exponent: float = SPEED_EXPONENT) -> float:
    """The UNCLAMPED speed that lands ``take`` on ``budget`` under ``exponent``.

    Inverts :func:`speed_seconds`: ``s = b * ((seconds - fixed) / (budget - fixed)) ** (1/k)``.
    """
    base = 1.0 if take.speed is None else float(take.speed)
    if budget <= 0.0 or take.seconds <= 0.0 or exponent <= 0.0:
        return base
    fixed = _fixed(take, budget)
    return base * speed_multiplier(take.seconds - fixed, budget - fixed, exponent)


def fit_speed(take: LineTake, budget: float) -> float:
    """The clamped speed that lands ``take`` on ``budget`` — under the CALIBRATED
    response (:data:`SPEED_EXPONENT`), not ``1/speed``.

    Reported fixed silence is subtracted from both sides, falling back to the whole
    clip when it cannot be.
    """
    return clamp_speed(speed_for(take, budget))


def target_words(text: str, seconds: float, budget: float) -> int:
    """Words this line should carry to fill ``budget`` at its OWN measured pace."""
    words = spoken_words(text)
    if seconds <= 0.0 or budget <= 0.0 or words <= 0:
        return words
    return max(1, round(words * budget / seconds))


async def _attempt[T](index: int, what: str, call: Awaitable[T]) -> T | None:
    """Await one paid call; a failure is logged and returns ``None`` (cancel propagates)."""
    try:
        return await call
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("line_fit: %s of line %d failed", what, index, exc_info=True)
        return None


def _closer(candidate: LineTake, best: LineTake, budget: float) -> bool:
    return abs(miss_ratio(candidate.seconds, budget)) < abs(miss_ratio(best.seconds, budget))


def speed_seconds(take: LineTake, speed: float, exponent: float = SPEED_EXPONENT) -> float:
    """Predicted length of ``take`` re-synthesized at ``speed`` — :func:`fit_speed`'s model.

    The part speed can move scales as ``(base / speed) ** exponent``; reported fixed
    silence does not.
    """
    base = 1.0 if take.speed is None else float(take.speed)
    if take.seconds <= 0.0 or speed <= 0.0:
        return take.seconds
    fixed = _fixed(take)
    return fixed + (take.seconds - fixed) * (base / speed) ** exponent


def local_exponent(first: LineTake, second: LineTake) -> float | None:
    """The exponent ONE line exhibited between two measured takes of the same text,
    or ``None`` when the pair cannot define one (same speed, a non-positive length, or
    a narrator that ignored / inverted the speed — outside :data:`LOCAL_EXPONENT_RANGE`).
    """
    b0 = 1.0 if first.speed is None else float(first.speed)
    b1 = 1.0 if second.speed is None else float(second.speed)
    fixed = min(_fixed(first, second.seconds), _fixed(second, first.seconds))
    d0, d1 = first.seconds - fixed, second.seconds - fixed
    if d0 <= 0.0 or d1 <= 0.0 or b0 <= 0.0 or b1 <= 0.0 or abs(math.log(b1 / b0)) < 1e-6:
        return None
    k = math.log(d0 / d1) / math.log(b1 / b0)
    lo, hi = LOCAL_EXPONENT_RANGE
    return k if math.isfinite(k) and lo <= k <= hi else None


def corrective_speed(first: LineTake, speed_take: LineTake, budget: float) -> float | None:
    """The ONE corrective speed for a measured speed take that still misses
    :data:`ACCEPT_BAND` — or ``None`` when no extra synthesis can help.

    Re-solved from ``speed_take`` with the exponent this line just exhibited between
    ``first`` and ``speed_take`` (:func:`local_exponent`), so a line whose words
    stretch less than the fleet's is corrected by its OWN response. ``None`` when the
    take is already in band, the pair defines no exponent, or the clamp leaves the
    speed where a take already measured it (a repeat buys the same clip).
    """
    if abs(miss_ratio(speed_take.seconds, budget)) <= ACCEPT_BAND:
        return None
    k = local_exponent(first, speed_take)
    if k is None:
        return None
    speed = clamp_speed(speed_for(speed_take, budget, k))
    tried = {clamp_speed(1.0 if t.speed is None else t.speed) for t in (first, speed_take)}
    return None if speed in tried else speed


def achievable(take: LineTake, budget: float) -> tuple[float, float]:
    """``(speed, signed miss)`` this take can reach: kept as-is in band, else clamped speed.

    The quantity selection ranks on. A raw miss alone ranks a -20% take above a +16%
    one even though the +16% one reaches +9.6% at the 1.1 clamp and the -20% one only
    -14.8% at 0.9 (calibrated response) — the comparison :func:`fit_line` used to make, which is why a line
    with no take in band always fell back to whichever was rawest-closest (live
    ``01a15e2b-2a88`` / ``01a15e77-e3a4``: the ORIGINAL, at the 0.9 floor).
    """
    current = clamp_speed(1.0 if take.speed is None else take.speed)
    if band_for(take.seconds, budget) == "accept":
        return current, miss_ratio(take.seconds, budget)
    speed = fit_speed(take, budget)
    if speed == current:
        return current, miss_ratio(take.seconds, budget)
    return speed, miss_ratio(speed_seconds(take, speed), budget)


@dataclass(frozen=True, slots=True)
class Attempt:
    """One measured candidate text for a line, and where the speed clamp can land it.

    ``ask`` is the word count the rewrite was ASKED for (the first take: its own count)
    and ``source`` the attempt number it rewrote — the controlled variable of the
    secant, and the mapping the log needs to tell which clip a measurement belongs to.
    """

    number: int
    take: LineTake
    words: int
    ask: int
    source: int | None
    speed: float
    error: float

    def describe(self) -> str:
        origin = (
            "first" if self.source is None else f"rw{self.number}<-#{self.source} ask {self.ask}"
        )
        return (
            f"#{self.number} {origin} {self.words}w {self.take.seconds:.2f}s "
            f"-> sp{self.speed:g} {self.error:+.1%}"
        )


def _attempt_of(
    number: int, take: LineTake, budget: float, *, ask: int, source: int | None
) -> Attempt:
    speed, error = achievable(take, budget)
    return Attempt(
        number=number,
        take=take,
        words=spoken_words(take.text),
        ask=ask,
        source=source,
        speed=speed,
        error=error,
    )


def select_attempt(attempts: Sequence[Attempt]) -> Attempt:
    """The attempt with the smallest error AFTER the speed clamp (first wins a tie)."""
    return min(attempts, key=lambda a: abs(a.error))


def secant_ask(source: Attempt, attempts: Sequence[Attempt], budget: float) -> int | None:
    """Words to ASK of a rewrite of ``source``, interpolated from measurements that
    bracket ``budget`` — or ``None`` when ``source``'s measurements do not bracket it.

    The points are ``(ask, seconds)``: ``source`` itself (asked its own count) and
    every rewrite OF ``source``. The ask, not the delivered count, is the variable the
    loop controls, so the secant corrects the narrator's pace AND the model's habit of
    delivering more than asked (live 2026-10-21: asked +5 words, delivered +12, twice;
    re-asking the same target re-rolled the same overshoot).
    """
    # (ask, seconds): the source asked for its own count IS the source.
    family = [(source.words, source.take.seconds)] + [
        (a.ask, a.take.seconds) for a in attempts if a.source == source.number
    ]
    under = [p for p in family if p[1] < budget]
    over = [p for p in family if p[1] > budget]
    if not under or not over:
        return None
    lo = max(under, key=lambda p: p[1])
    hi = min(over, key=lambda p: p[1])
    if lo[0] == hi[0]:
        return None
    x = lo[0] + (budget - lo[1]) * (hi[0] - lo[0]) / (hi[1] - lo[1])
    want = max(1, round(x))
    if want == source.words:
        # Asking a rewrite for the count it already has buys a re-roll, not a correction.
        want = max(1, source.words + (1 if budget > source.take.seconds else -1))
    return want


def next_rewrite(attempts: Sequence[Attempt], budget: float) -> tuple[Attempt, int]:
    """The attempt to rewrite next and the word count to ask for.

    A bracketing pair (one measurement under the budget, one over, both from the same
    source) is interpolated (:func:`secant_ask`); otherwise the take nearest the budget
    is rewritten to the count its own pace implies (:func:`target_words`).
    """
    for source in attempts:
        want = secant_ask(source, attempts, budget)
        if want is not None:
            return source, want
    nearest = min(attempts, key=lambda a: abs(miss_ratio(a.take.seconds, budget)))
    return nearest, target_words(nearest.take.text, nearest.take.seconds, budget)


async def _rewrite_loop(
    index: int,
    take: LineTake,
    budget: float,
    *,
    synth: Callable[[str, float | None], Awaitable[LineTake]],
    rewrite: Callable[[int, str, float, float, int], Awaitable[str | None]],
) -> tuple[list[Attempt], int]:
    """Up to :data:`MAX_REWRITES_PER_LINE` rewrites while no take is out of the rewrite
    band; EVERY measured take is returned, so selection can rank all of them."""
    attempts = [_attempt_of(0, take, budget, ask=spoken_words(take.text), source=None)]
    rewrites = 0
    # Rewrite while NO take can reach the accept band — judged after the calibrated
    # speed clamp, not by the raw +/-12% band: a +11.5% take reaches only +5.3% at 1.1.
    while rewrites < MAX_REWRITES_PER_LINE and all(abs(a.error) > ACCEPT_BAND for a in attempts):
        rewrites += 1
        source, want = next_rewrite(attempts, budget)
        new_text = await _attempt(
            index,
            "rewrite",
            rewrite(index, source.take.text, source.take.seconds, budget, want),
        )
        if not new_text or not new_text.strip() or new_text.strip() == source.take.text.strip():
            break
        candidate = await _attempt(index, "synthesis", synth(new_text.strip(), None))
        if candidate is None:
            break
        attempts.append(
            _attempt_of(len(attempts), candidate, budget, ask=want, source=source.number)
        )
    return attempts, rewrites


async def fit_line(
    index: int,
    take: LineTake,
    budget: float,
    *,
    synth: Callable[[str, float | None], Awaitable[LineTake]],
    rewrite: Callable[[int, str, float, float, int], Awaitable[str | None]],
    rewritable: bool,
) -> LineFit:
    """Accept / speed / rewrite one line against its budget.

    ``rewrite(index, text, measured_seconds, budget_seconds, target_words)`` returns
    the new text, or ``None`` when no usable rewrite came back. Every measured take
    (the first and each rewrite) is ranked by the error it can reach AFTER the speed
    clamp (:func:`achievable`); the winner is re-synthesized at its speed — and once
    more at :func:`corrective_speed` when that measured speed take still misses the
    accept band — and the shipped take is whichever MEASURED take is closest — so a speed take that came out
    worse than predicted never displaces a better raw one. Never raises on a failed
    synthesis or rewrite — the best take in hand is kept.
    """
    initial = band_for(take.seconds, budget)
    attempts = [_attempt_of(0, take, budget, ask=spoken_words(take.text), source=None)]
    rewrites = 0
    if rewritable:
        attempts, rewrites = await _rewrite_loop(index, take, budget, synth=synth, rewrite=rewrite)
    chosen = select_attempt(attempts)
    best = chosen.take
    measured = [a.take for a in attempts]
    speed = chosen.speed
    speed_takes: list[LineTake] = []
    if band_for(best.seconds, budget) != "accept" and speed != clamp_speed(
        1.0 if best.speed is None else best.speed
    ):
        candidate = await _attempt(index, "speed re-synthesis", synth(best.text, speed))
        if candidate is not None:
            speed_takes.append(replace(candidate, speed=speed))
            fix = corrective_speed(best, speed_takes[0], budget)
            if fix is not None:
                second = await _attempt(index, "corrective re-synthesis", synth(best.text, fix))
                if second is not None:
                    speed_takes.append(replace(second, speed=fix))
    measured.extend(speed_takes)
    # The shipped clip is always a MEASURED take — never a prediction.
    shipped = min(measured, key=lambda t: abs(miss_ratio(t.seconds, budget)))
    resynthesized = any(shipped is t for t in speed_takes)
    corrected = len(speed_takes) > 1
    if rewrites or speed_takes:
        logger.info(
            "line_fit: line %d window %.2fs — %s; chosen #%d; speed takes [%s]; shipped %d "
            "words %.2fs at speed %s (%+.1f%%)",
            index,
            budget,
            "; ".join(a.describe() for a in attempts),
            chosen.number,
            ", ".join(
                f"sp{t.speed:g} pred {speed_seconds(best, float(t.speed or 1.0)):.2f}s "
                f"got {t.seconds:.2f}s ({miss_ratio(t.seconds, budget):+.1%})"
                for t in speed_takes
            ),
            spoken_words(shipped.text),
            shipped.seconds,
            shipped.speed,
            miss_ratio(shipped.seconds, budget) * 100,
        )
    return LineFit(
        index=index,
        take=shipped,
        budget=budget,
        initial_band=initial,
        rewrites=rewrites,
        resynthesized=resynthesized,
        corrected=corrected,
        attempts=tuple(attempts),
    )


async def fit_script(
    takes: Sequence[LineTake],
    ordered_seconds: float,
    *,
    synth: Callable[[str, float | None], Awaitable[LineTake]],
    rewrite: Callable[[int, str, float, float, int], Awaitable[str | None]],
    rewritable: Sequence[bool],
) -> list[LineFit]:
    """Fit every line; the last :data:`ABSORB_LINES` spoken lines absorb the residual.

    ``takes`` are the MEASURED first syntheses, one per line, in script order.
    Nothing is touched when the order is unset, nothing is spoken, or the measured
    total is already within :data:`ACCEPT_BAND` of the order — per-line pacing
    differences then cost money and buy no length.
    """
    texts = [t.text for t in takes]
    budgets = line_budgets(texts, ordered_seconds)
    unchanged = [
        LineFit(index=i, take=t, budget=budgets[i], initial_band=band_for(t.seconds, budgets[i]))
        for i, t in enumerate(takes)
    ]
    spoken = [i for i, t in enumerate(takes) if spoken_words(t.text) > 0]
    if not spoken or ordered_seconds <= 0.0 or not math.isfinite(ordered_seconds):
        return unchanged
    total = math.fsum(t.seconds for t in takes)
    if abs(miss_ratio(total, ordered_seconds)) <= ACCEPT_BAND:
        return unchanged

    absorb = max(1, min(ABSORB_LINES, len(spoken) - 1)) if len(spoken) > 1 else 1
    head, tail = spoken[:-absorb], spoken[-absorb:]
    fits: dict[int, LineFit] = {i: unchanged[i] for i in range(len(takes))}

    async def _fit(i: int, budget: float) -> LineFit:
        return await fit_line(
            i, takes[i], budget, synth=synth, rewrite=rewrite, rewritable=bool(rewritable[i])
        )

    for fit in await asyncio.gather(*(_fit(i, budgets[i]) for i in head)):
        fits[fit.index] = fit

    # The tail is budgeted from what the head ACTUALLY came to.
    remaining = ordered_seconds - math.fsum(fits[i].take.seconds for i in head)
    tail_words = sum(spoken_words(texts[i]) for i in tail)
    tail_budgets = {
        i: max(
            remaining * spoken_words(texts[i]) / tail_words,
            budgets[i] * MIN_ABSORB_FRACTION,
        )
        for i in tail
    }
    for fit in await asyncio.gather(*(_fit(i, tail_budgets[i]) for i in tail)):
        fits[fit.index] = fit
    return [fits[i] for i in range(len(takes))]


async def fit_lines(
    takes: Sequence[LineTake],
    budgets: Sequence[float],
    *,
    synth: Callable[[int, str, float | None], Awaitable[LineTake]],
    rewrite: Callable[[int, str, float, float, int], Awaitable[str | None]],
    rewritable: Sequence[bool],
) -> list[LineFit]:
    """Fit every line to its OWN budget — no pooling, no residual absorption.

    For a spine where each line backs a fixed window (a recorded footage clip), the
    budget is that window and a neighbour's miss is not this line's to repair: the
    picture under it does not move. ``synth(index, text, speed)`` is per line because
    each line may be cast to a different voice. A line with no words or a
    non-positive budget is returned unchanged (nothing to fit it to). Never raises on
    a failed synthesis or rewrite — :func:`fit_line` keeps the best take in hand.
    """

    async def _fit(i: int) -> LineFit:
        def _synth_i(text: str, speed: float | None) -> Awaitable[LineTake]:
            return synth(i, text, speed)

        return await fit_line(
            i,
            takes[i],
            budgets[i],
            synth=_synth_i,
            rewrite=rewrite,
            rewritable=bool(rewritable[i]),
        )

    fits: list[LineFit] = [
        LineFit(index=i, take=t, budget=b, initial_band=band_for(t.seconds, b))
        for i, (t, b) in enumerate(zip(takes, budgets, strict=True))
    ]
    live = [
        i
        for i, (t, b) in enumerate(zip(takes, budgets, strict=True))
        if spoken_words(t.text) > 0 and math.isfinite(b) and b > 0.0
    ]
    for fit in await asyncio.gather(*(_fit(i) for i in live)):
        fits[fit.index] = fit
    return fits
