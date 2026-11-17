"""narration_budget — reconcile a narration script against the ORDERED duration.

For every narration-driven content type the delivered video length IS the sum of
the TTS clip seconds. ``lesson/compile.py`` rescales the slides to fit whatever
audio came back, so the ordered duration has no effect on the master unless the
SCRIPT is made to fit it. Until now the only thing standing between a customer's
order and the delivered file was prompt prose ("N words is a HARD CEILING"),
which an LLM honours approximately at best. Measured live 2026-09-11:

===================  ==========  ==========  ========
content type         ordered     delivered   deviation
===================  ==========  ==========  ========
lesson                    120 s     95.54 s    -20.4%
lesson                    360 s    399.68 s    +11.0%
lesson                    420 s    382.31 s     -9.0%
website_review             60 s    134.20 s   +123.7%
===================  ==========  ==========  ========

Nothing trimmed and nothing extended. This module is the missing arithmetic:
count the words, divide by the MEASURED speech rate, and say whether the draw is
``ok`` / ``long`` / ``short`` against what was ordered — plus a deterministic
plan for cutting a long draw back to the tolerance band.

Deliberate properties
---------------------
* **Pure.** No I/O, no clock, no randomness, no global state. Every function is
  a total function of its arguments, so a plan is reproducible from a persisted
  stage output alone.
* **Deterministic.** Iteration is over ``range``/lists and insertion-ordered
  dicts only — never a set — and ``TrimPlan`` index tuples are emitted in
  ascending order. The same inputs yield an identical plan on every call, in
  every process. Where several plans are equally good the tie is broken by an
  explicit comparison (fewest lines, then the LATEST lines), not by whichever
  one an iteration order happened to reach first.
* **It never invents an order.** ``ordered_seconds <= 0`` means nothing was
  ordered, which is ``"ok"`` here; deciding a default duration belongs to the
  stage that reads the brief, not to this module.
* **It drops WHOLE lines, and it SEARCHES for which ones.** Truncating a
  sentence mid-clause would ship audible garbage; cutting from the tail loses
  the least-load-bearing content and keeps the opening intact, so tail-first is
  the PREFERENCE — but it is no longer the algorithm. A greedy tail-first walk
  cannot reconsider a drop it has already accepted, so a cheap early cut can
  leave the band unreachable with what remains, and the walk then reported the
  ORDER as unreachable: 35/35/20 words at 60 wpm against a 60s order took the
  20s tail, stopped at 70.0s and declared impossibility, while dropping either
  35-word line alone lands at 55.0s. ``plan_trim`` now scores every candidate
  drop set (as a subset-sum table over the droppable word counts) and takes the
  smallest one that lands. Protected indices (a verbatim phrase, an opening
  hook) are never candidates, and the last surviving line is never dropped — an
  empty script is a worse outcome than a long one.
* **It never pads.** There is no "extend" plan: a short draw is reported and
  accepted. Repeating or stretching audio to hit a number is a defect, not a
  fix.
* **It never overshoots, and it never lies about falling short.** The trim is
  bounded on BOTH edges of the band (a drop landing under ``ordered x (1 -
  tolerance)`` is declined), and a plan that could not reach the band says so on
  :attr:`TrimPlan.exceeds_limit` instead of returning an indistinguishable
  "trimmed N line(s)". "Unreachable" is only ever said after the candidates have
  actually been enumerated, and the degradation text names how many were tried.
* **Non-finite input changes nothing.** A NaN/infinite ordered duration or
  tolerance is the no-order NO-OP path, not a licence to cut: every comparison
  against NaN is False, which used to make the trim delete lines until one
  survived and then report ``exceeds_limit=False`` — a silently amputated
  deliverable reported as a success.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.services.stages.shared.degradation import (
    DURATION_OUT_OF_TOLERANCE,
    NARRATION_ESTIMATE_OFF_TARGET,
    ORDERED_DURATION_INFEASIBLE,
    SPEECH_RATE_UNUSABLE,
    Degradation,
    advisory,
    degradation,
)
from src.services.stages.shared.duration_conformance import (
    DURATION_STEER_FRACTION,
    DURATION_TOLERANCE_FRACTION,
)
from src.services.stages.shared.time_budget import NARRATION_BUDGET_RATE

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

#: Sanity bounds on a speech rate, in words per minute. The measured live kokoro
#: rate is ~171 and the cross-type default is 150; a value outside this band is
#: a broken probe or a unit mix-up, not a voice. Substituting (rather than
#: raising) is deliberate: this module sits on a PAID run's happy path, and an
#: unreasonable rate must degrade the estimate, never fail the stage.
WORDS_PER_MINUTE_FLOOR = 60.0
WORDS_PER_MINUTE_CEILING = 300.0

#: The rate substituted for an unusable one — the SAME cross-type default the
#: stages already pass when no probe succeeded (``time_budget.DEFAULT_WPM_NATURAL``,
#: restated here to keep this module import-free of the stage layer; the two are
#: pinned equal by a test).
#:
#: This used to CLAMP to the nearest bound, and that was backwards. The estimate
#: is ``words / wpm * 60``, so a LOW rate INFLATES it: at the 60 floor a
#: legitimate 190-word script estimates at 190s, and against a 60s order the
#: trimmer cuts it to ~66 words and delivers ~26s. The "conservative" edge is the
#: one that destroys the deliverable. The ceiling is the mirror defect (an
#: under-estimate ships a long film, recoverable but wrong). A rate outside the
#: band carries NO information about which direction the truth lies in, so the
#: only defensible substitute is the one this system uses when it has no
#: measurement at all — and it is now reported, never silent.
DEFAULT_WORDS_PER_MINUTE = 150.0

#: Default band around the ordered duration that counts as delivered-as-ordered.
#:
#: This is the SHARED constant, imported rather than restated: a pre-production
#: script budget and the post-production delivered-vs-ordered check that grades
#: it must agree on what "close enough" means, or a script trimmed to the letter
#: of this module can still be reported out of tolerance by
#: ``duration_conformance``. The duplicate literal that used to live here was
#: checked by nothing. No cycle: ``duration_conformance`` imports only
#: ``action_reel.target_duration`` (stdlib-only) and ``shared.degradation``, and
#: imports nothing from this module.
DEFAULT_TOLERANCE = DURATION_TOLERANCE_FRACTION

#: Default LONG-side steering target for a pre-audio estimate, imported for the
#: same reason as :data:`DEFAULT_TOLERANCE` — see ``duration_conformance`` for
#: why it is 4%, and why there is no short-side twin.
DEFAULT_STEER = DURATION_STEER_FRACTION

#: How a per-line SECONDS measurement is turned into the integer weight the trim
#: search sums, and the matching pseudo-rate that converts a weight total back to
#: seconds: ``weight / TRIM_SECONDS_RATE * 60.0 == weight / 10.0 == seconds``.
#: The whole trim (estimate, band comparison, greedy floor) is ``total / rate *
#: 60``, so feeding it deciseconds at 600 wpm reuses that arithmetic UNCHANGED
#: rather than forking a second code path for measured input.
#:
#: DECISECONDS, not centiseconds, and the reason is :data:`MAX_TRIM_SEARCH_STATES`.
#: The subset-sum table holds one entry per ACHIEVABLE weight total, so its size
#: is bounded by the script's total weight, and the bound counts transitions —
#: ``sum(len(table))`` over the droppable lines. Word counts keep that tiny (a
#: 30-line, 9027-word script measured 167 entries). Centiseconds do not: a 10-
#: minute narration is 60,000 centiseconds, and 30 droppable lines against a
#: table growing toward that ceiling costs on the order of 30 x 60,000 = 1.8M
#: transitions — 9x over the 200,000 bound, which would silently drop the search
#: to ``search_exhausted=False`` and hand a paid script to the greedy floor. At
#: deciseconds the same script tops out near 6,000 sums, i.e. ~180,000
#: transitions, inside the bound. The cost is rounding each line to 0.1s: at
#: worst 0.05s per line, ~1.5s across 30 lines, which is under 1.3% of even a
#: 2-minute order and far inside the +/-10% band the plan aims at. An exhaustive
#: search of slightly-rounded seconds beats an exact search that never ran.
TRIM_SECONDS_QUANTUM = 10.0
TRIM_SECONDS_RATE = 600.0

#: Ceiling on the trim search's work, counted in dynamic-programming
#: TRANSITIONS actually performed: before each droppable line the table's REAL
#: size is added to a running total, and the multi-line search stops the moment
#: that total would exceed this. It is a measurement, not an estimate.
#:
#: It used to be an estimate — ``len(droppable) * (droppable_words + 1)``, i.e.
#: the cost of a table holding EVERY word total from 0 to the droppable word
#: count. A real table holds only the ACHIEVABLE sums, and on a 30-line, 9027
#: word audiobook script the two differ by ~1600x: 270,840 estimated against a
#: measured 167 entries built in 0.0007s. The estimate disqualified a trivially
#: tractable script from the search, the greedy walk then amputated 28 of its 30
#: lines, and the plan reported ``candidates_evaluated=0`` — a paid deliverable
#: destroyed by a cost model, not by a cost. A bound that can be wrong by three
#: orders of magnitude is not a bound; count the states you actually build.
#:
#: This gates MULTI-LINE combinations ONLY. Every SINGLE-line drop is scored
#: unconditionally (:func:`_single_drop_sets`) because that sweep is O(n) and is
#: the overwhelmingly common solution — so no value of this constant, and no
#: error in any future cost model, can make the catastrophic case reachable.
#: When the bound IS hit the search is not exhaustive and
#: :attr:`TrimPlan.search_exhausted` says so, which is what keeps the floor
#: degradation honest about what was actually tried.
MAX_TRIM_SEARCH_STATES = 200_000

#: ``"unknown"`` is NOT a fourth outcome of a working comparison — it is the
#: refusal to name a direction when one of the two numbers is not a real number
#: (NaN/inf estimate, or a NaN tolerance). Reporting ``"short"`` for a NaN
#: estimate asserted a direction the arithmetic never established, and callers
#: act on that word: ``"short"`` means ACCEPT-and-ship, ``"long"`` means trim.
#: Every consumer tests ``== "ok"`` / ``== "long"``, so ``"unknown"`` takes the
#: honest path through all of them: no trim, no padding, and a recorded
#: degradation because it is not ``"ok"``.
NarrationVerdict = Literal["ok", "long", "short", "unknown"]


def _is_finite(value: float) -> bool:
    """True only for a real, finite float — NaN and +/-inf are not budgetable."""
    try:
        return math.isfinite(float(value))
    except TypeError, ValueError:
        return False


def wpm_is_usable(words_per_minute: float) -> bool:
    """False when this rate cannot be budgeted against and must be substituted.

    A rate is usable only if it is a real number inside
    ``[WORDS_PER_MINUTE_FLOOR, WORDS_PER_MINUTE_CEILING]``. ``<= 0`` and NaN are
    included (``not rate > 0.0`` catches NaN, which a bare ``max`` does not).

    A caller that estimates at an unusable rate MUST record
    :func:`speech_rate_degradation` — every duration number it then produces was
    computed at a rate nobody measured, and that fact is otherwise invisible.
    """
    rate = float(words_per_minute)
    if not rate > 0.0:  # also catches NaN
        return False
    return WORDS_PER_MINUTE_FLOOR <= rate <= WORDS_PER_MINUTE_CEILING


def _safe_wpm(words_per_minute: float) -> float:
    """``words_per_minute``, or :data:`DEFAULT_WORDS_PER_MINUTE` when unusable.

    THE guard for ``wpm <= 0``: a zero or negative rate would otherwise make
    ``estimate_seconds`` divide by zero (or return a negative duration, which
    then reads as "short" and triggers a pointless re-ask).

    It substitutes the cross-type default rather than clamping to the nearest
    bound — see :data:`DEFAULT_WORDS_PER_MINUTE` for why the floor was the
    DESTRUCTIVE choice, not the conservative one. Substitution is never silent:
    :func:`wpm_is_usable` is the predicate callers check to report it.
    """
    if not wpm_is_usable(words_per_minute):
        return DEFAULT_WORDS_PER_MINUTE
    return float(words_per_minute)


def count_words(texts: Sequence[str]) -> int:
    """Spoken word count (whitespace-split), the same counting rule as the stages."""
    return sum(len(str(t).split()) for t in texts)


def estimate_seconds(texts: Sequence[str], *, words_per_minute: float) -> float:
    """Estimated spoken seconds for ``texts`` at ``words_per_minute``."""
    return count_words(texts) / _safe_wpm(words_per_minute) * 60.0


def budget_words(ordered_seconds: float, *, words_per_minute: float) -> int:
    """How many words the ordered duration can afford at this rate.

    ``ordered_seconds <= 0`` (nothing ordered) affords nothing — 0 — rather than
    a negative count. A non-finite order affords nothing for the same reason,
    and would otherwise make ``round()`` raise on a paid run's happy path.

    ⭐ WAVE 14, N5. ``NARRATION_BUDGET_RATE`` is NAMED rather than left implicit, for
    the reason ``compute_plan.beat_word_budget`` records at length: this is the
    fleet-wide twin of ``demo_video.gen.narration_words_target``, and the two must move
    together. It is 1.0 today, so this changes no number — it removes the way the
    numbers could quietly stop matching.
    """
    ordered = float(ordered_seconds)
    if not _is_finite(ordered) or ordered <= 0.0:
        return 0
    return round(ordered / 60.0 * _safe_wpm(words_per_minute) * NARRATION_BUDGET_RATE)


#: Half-width of the word band a script prompt states around its target. Inside the
#: +/-10% band that grades the master (:data:`DEFAULT_TOLERANCE`), so a draw that
#: honours it needs no re-ask. The SAME figure lesson's prompt states (#1126).
WORD_BAND_FRACTION = 0.05


def word_band(words: int) -> tuple[int, int]:
    """``(low, high)`` words around a target — :data:`WORD_BAND_FRACTION` each way."""
    return (
        math.floor(words * (1.0 - WORD_BAND_FRACTION)),
        math.ceil(words * (1.0 + WORD_BAND_FRACTION)),
    )


def word_target_sentence(target_seconds: float, wpm: float, words: int) -> str:
    """The prompt's whole-script length rule: a TARGET with a floor, never a ceiling.

    The previous wording ("N words is a HARD CEILING … write fewer if the story is
    done sooner, never more") was read literally: lesson run 01a158a6-0f2c
    (2026-10-20, 60s ordered, ceiling 198 words) drew 66, and a replay drew
    57/120/70/58. A film is exactly its narration and is graded at +/-10% in BOTH
    directions, so an instruction that only forbids one of them buys the other.
    One wording for app_review, promo and website_review.
    """
    low, high = word_band(words)
    return (
        f"Total spoken length: at this narrator's measured pace of {wpm:.0f} words per "
        f"minute, {target_seconds:.0f} seconds is {words} words. Write {words} words in "
        f"total (all lines combined) — no fewer than {low} and no more than {high}. "
        "A short script is as wrong as a long one: the customer ordered "
        f"{target_seconds:.0f} seconds. Reach the length with substance — concrete "
        "detail, a worked example, why it matters — never filler or repetition.\n"
    )


def narration_verdict(
    estimated_seconds: float,
    ordered_seconds: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> NarrationVerdict:
    """``"ok"`` when the estimate is within ``tolerance`` of the order.

    ``ordered_seconds <= 0`` (or a non-finite order) is ``"ok"``: nothing was
    ordered, and inventing an order is not this module's job.

    A non-finite ESTIMATE or TOLERANCE is ``"unknown"``, not a direction. Every
    comparison against NaN is False, so the old code fell through to
    ``"long" if deviation > 0.0 else "short"`` and returned ``"short"`` — a
    stated direction derived from an arithmetic that established nothing, on the
    one word that tells a caller to ACCEPT the draw and ship it.
    """
    ordered = float(ordered_seconds)
    if not _is_finite(ordered) or ordered <= 0.0:
        return "ok"
    if not _is_finite(estimated_seconds) or not _is_finite(tolerance):
        return "unknown"
    deviation = (float(estimated_seconds) - ordered) / ordered
    if abs(deviation) <= abs(float(tolerance)):
        return "ok"
    return "long" if deviation > 0.0 else "short"


def closer_to_order(candidate: float, incumbent: float, ordered: float) -> bool:
    """True when ``candidate`` sits STRICTLY closer to ``ordered`` than ``incumbent``.

    THE overcorrection guard, lifted verbatim from ``app_review/script.py``'s
    private ``_closer`` (three stages carry their own copy of this one line).
    Every correction — a re-ask, a trim — is gated on it, so a "fix" that swaps a
    +11% overrun for a -30% underrun is discarded and the original draw ships.
    That is not hypothetical: structural relief on a 30s order went +56.8% ->
    -33.5%, and a lesson went +216% -> -20.4%. Both stopped FAILING and both miss
    the band on the far side.

    **Strict ``<``, and ties keep the INCUMBENT.** Doing nothing is cheaper than
    doing something equivalent, and an equal-distance swap is churn that buys a
    paid run a second LLM draw and a different set of words for the same error.
    The semantics are ``abs`` distance on both sides, so it is symmetric: a
    candidate is judged by how far it lands from the order, never by which side
    of it the candidate is on.
    """
    return abs(candidate - ordered) < abs(incumbent - ordered)


def steer_verdict(
    estimate_seconds: float,
    ordered_seconds: float,
    *,
    steer: float = DEFAULT_STEER,
    tolerance: float = DEFAULT_TOLERANCE,
) -> NarrationVerdict:
    """Where to AIM a pre-audio estimate — a tighter long side, the same short side.

    :func:`narration_verdict` grades against the +/-10% band that also grades the
    delivered master, and a caller steering by it aims at the band's EDGE: +9% is
    ``"ok"``, nothing trims, and the estimate's own several-percent error then
    carries the delivery outside the band. This returns ``"long"`` from
    ``ordered x (1 + steer)`` (4%) so there is room left for that error, while
    ``"short"`` keeps the full ``tolerance`` (10%).

    The asymmetry is deliberate and is NOT a rounding of the same idea: nothing
    downstream pads an underrun, so a tightened short side could only make the
    trim decline cuts it should take (it refuses any drop landing under the lower
    bound) and convert a fixable overrun into an accepted one. Steering short
    would overshoot precisely while the estimate itself runs short.

    Handling of the degenerate inputs MIRRORS :func:`narration_verdict` exactly,
    because a caller switching between them must not change its failure
    behaviour: ``ordered_seconds <= 0`` or a non-finite order is ``"ok"`` (nothing
    was ordered; inventing an order is not this module's job), and a non-finite
    estimate, steer or tolerance is ``"unknown"`` — never a direction the
    arithmetic did not establish, on words whose whole meaning to a caller is
    "trim" or "ship it".
    """
    ordered = float(ordered_seconds)
    if not _is_finite(ordered) or ordered <= 0.0:
        return "ok"
    if not _is_finite(estimate_seconds) or not _is_finite(steer) or not _is_finite(tolerance):
        return "unknown"
    estimate = float(estimate_seconds)
    if estimate > ordered * (1.0 + abs(float(steer))):
        return "long"
    if estimate < ordered * (1.0 - abs(float(tolerance))):
        return "short"
    return "ok"


@dataclass(frozen=True)
class TrimPlan:
    """A deterministic, whole-line trim of a narration script.

    ``keep_indices`` and ``dropped_indices`` partition ``range(len(texts))`` and
    are both in ASCENDING order, so applying a plan is
    ``[lines[i] for i in plan.keep_indices]`` and preserves the script's order.
    """

    keep_indices: tuple[int, ...]
    estimated_seconds_before: float
    estimated_seconds_after: float
    ordered_seconds: float
    dropped_indices: tuple[int, ...]
    #: True when :attr:`estimated_seconds_after` STILL exceeds
    #: ``ordered_seconds x (1 + tolerance)`` — the trim ran and did not reach the
    #: band. Without this a plan that dropped lines and remained 29% long is
    #: textually indistinguishable from one that landed, which is how
    #: ``website_review_v1`` shipped 77.46s against a 60s order and reported
    #: "trimmed 2 line(s)". The field exists so a caller can ASK; the numbers
    #: needed to act on it are :attr:`protected_count` and
    #: :attr:`declined_under_band`.
    exceeds_limit: bool = False
    #: How many indices the caller pinned. THE usual reason :attr:`exceeds_limit`
    #: is true (``website_review`` protects every authored shotlist line, an
    #: unbounded prefix), and the number that makes the report actionable.
    protected_count: int = 0
    #: True when a further droppable line existed and was DECLINED because taking
    #: it would have put the estimate below ``ordered_seconds x (1 - tolerance)``
    #: — i.e. the trim refused to overshoot to the short side. It records a
    #: REFUSAL, not a stop: a declined line is skipped and the walk continues, so
    #: this can be True on a plan that then landed squarely inside the band.
    #: :attr:`exceeds_limit` is the field that says whether the band was reached;
    #: this one only explains WHY, when it was not.
    declined_under_band: bool = False
    #: How many lines the search was ALLOWED to consider dropping (unprotected,
    #: and never the last survivor). ``0`` with :attr:`exceeds_limit` true means
    #: the floor is protection, full stop.
    droppable_count: int = 0
    #: How many distinct candidate drop sets the search actually scored. It is
    #: the number that makes a floor report falsifiable: a claim of "not
    #: reachable" backed by ``0`` candidates is a claim nothing checked.
    candidates_evaluated: int = 0
    #: True when EVERY droppable subset was scored (the normal case). False only
    #: when :data:`MAX_TRIM_SEARCH_STATES` was exceeded and the plan came from
    #: the bounded greedy walk, in which case "unreachable" means "not found
    #: under a bounded search", and the degradation text says exactly that.
    search_exhausted: bool = False


def _better_drop_set(
    candidate: tuple[int, tuple[int, ...]],
    incumbent: tuple[int, tuple[int, ...]],
) -> bool:
    """Fewest lines first; among equals, the one cutting the LATEST lines.

    Dropping line 0 and dropping line 5 shorten the film by the same number of
    seconds, but one of them deletes the opening hook — so the tie-break is the
    lexicographically GREATEST index tuple, which is the same tail-first
    preference the module has always had, now expressed as a comparison instead
    of as an iteration order.
    """
    if candidate[0] != incumbent[0]:
        return candidate[0] < incumbent[0]
    return candidate[1] > incumbent[1]


def _drop_sets_by_dropped_words(
    words: Sequence[int],
    droppable: Sequence[int],
    *,
    max_drops: int,
    max_states: int,
) -> tuple[dict[int, tuple[int, tuple[int, ...]]], bool]:
    """``(dropped_words -> (line_count, indices), exhausted)`` over every subset.

    The estimate depends ONLY on how many words survive, so two drop sets with
    the same word total are interchangeable and the candidate space collapses to
    a subset-sum table: one entry per achievable total, holding the best way to
    reach it under :func:`_better_drop_set`. That makes the search
    ``len(droppable) x distinct sums`` instead of the ``2^n`` a literal
    enumeration of subsets would cost, and it is EXACT, not a heuristic —
    nothing reachable within ``max_drops`` lines is skipped.

    Iteration is over lists and insertion-ordered dicts only, never a set, so
    the table — and therefore the plan — is identical in every process.

    **The bound is measured, not estimated.** ``work`` accumulates the table's
    REAL size before each line — that product IS the number of transitions the
    next line performs — and the walk returns early the moment it would exceed
    ``max_states``. The second element of the return is whether every subset was
    represented: False means the returned table covers only a PREFIX of
    ``droppable`` (still exact for that prefix, still deterministic, just not
    the whole space), and the caller must report ``search_exhausted=False``.
    """
    table: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    work = 0
    for index in droppable:
        work += len(table)
        if work > max_states:
            return table, False
        additions: dict[int, tuple[int, tuple[int, ...]]] = {}
        for total, (count, picked) in table.items():
            if count + 1 > max_drops:
                continue
            candidate = (count + 1, (*picked, index))
            reached = total + words[index]
            best = additions.get(reached, table.get(reached))
            if best is None or _better_drop_set(candidate, best):
                additions[reached] = candidate
        for reached, candidate in additions.items():
            best = table.get(reached)
            if best is None or _better_drop_set(candidate, best):
                table[reached] = candidate
    return table, True


def _single_drop_sets(
    words: Sequence[int],
    droppable: Sequence[int],
    *,
    max_drops: int,
) -> dict[int, tuple[int, tuple[int, ...]]]:
    """Every SINGLE-line drop, scored unconditionally. O(len(droppable)).

    This exists because a bound on the multi-line search must never be able to
    hide the one-line answer. Dropping exactly one line is both the cheapest
    thing to check and the overwhelmingly common solution, so it is checked
    before any cost model gets an opinion — the measured bound in
    :func:`_drop_sets_by_dropped_words` gates COMBINATIONS only.

    A line worth ZERO words never enters the table: it reaches total ``0``,
    where the do-nothing entry already sits and outranks it on line count. That
    is the same rule as "a drop must save time", enforced structurally.
    """
    table: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    if max_drops < 1:
        return table
    for index in droppable:
        candidate = (1, (index,))
        reached = words[index]
        best = table.get(reached)
        if best is None or _better_drop_set(candidate, best):
            table[reached] = candidate
    return table


def _candidate_drop_sets(
    words: Sequence[int],
    droppable: Sequence[int],
    *,
    max_drops: int,
) -> tuple[dict[int, tuple[int, tuple[int, ...]]], bool]:
    """The two tiers, merged. ``(dropped_words -> best drop set, exhausted)``.

    Tier one — every SINGLE-line drop — is scored unconditionally, before any
    cost model is consulted. Tier two — the combinations — is the only part the
    measured bound can cut short, so an over-eager bound (or a wrong one) can
    cost this function multi-line answers but can NEVER cost it the one-line
    answer. That asymmetry is the fix: the catastrophic case was a tractable
    script whose single-line answer was disqualified by an estimate.

    Merging the tiers is order-independent: :func:`_better_drop_set` is a strict
    total order on distinct index tuples, so which tier reached a word total
    first cannot change which entry survives.
    """
    table = _single_drop_sets(words, droppable, max_drops=max_drops)
    combinations, exhausted = _drop_sets_by_dropped_words(
        words, droppable, max_drops=max_drops, max_states=MAX_TRIM_SEARCH_STATES
    )
    for reached, candidate in combinations.items():
        best = table.get(reached)
        if best is None or _better_drop_set(candidate, best):
            table[reached] = candidate
    return table, exhausted


def _greedy_last_first(
    line_count: int,
    words: Sequence[int],
    droppable_desc: Sequence[int],
    *,
    total_words: int,
    words_per_minute: float,
    limit: float,
    lower_bound: float,
    rate: float | None = None,
) -> tuple[tuple[int, ...], float, bool]:
    """The tail-first walk, used ONLY once the band is proven out of reach.

    It is no longer the plan-finder — :func:`plan_trim` runs the exhaustive
    search for that — but it is still the right FLOORED plan: when nothing can
    land in the band the choice is between overruns, and shortening from the
    tail while refusing to cross under the band keeps the opening intact and
    never ships a short deliverable. Returns ``(dropped, after, declined)``.

    **A drop must save time.** A line worth zero words shortens nothing, so
    taking it deletes a beat for 0.0 seconds and moves the walk no closer to the
    band. ``["", "", <200 words>]`` against a 60s order used to come back as
    ``dropped_indices=(0, 1)`` with the estimate unchanged at 200.0s: two beats
    destroyed, the overrun untouched, and the plan reporting two lines "trimmed".
    ``after < current`` is the requirement, not ``after <= current``.

    ``rate`` overrides the words-per-minute conversion for a walk over MEASURED
    per-line weights (see :func:`plan_trim`'s ``line_seconds``), where the
    "words" are deciseconds and the rate is :data:`TRIM_SECONDS_RATE`. ``None``
    — every existing caller — is the unchanged ``_safe_wpm`` path.
    """
    rate = _safe_wpm(words_per_minute) if rate is None else float(rate)
    dropped: list[int] = []
    kept_words = total_words
    current = kept_words / rate * 60.0
    declined = False
    for index in droppable_desc:
        if current <= limit:
            break
        if line_count - len(dropped) <= 1:
            break
        after = (kept_words - words[index]) / rate * 60.0
        if not after < current:
            continue
        if after < lower_bound:
            declined = True
            continue
        dropped.append(index)
        kept_words -= words[index]
        current = after
    return tuple(sorted(dropped)), current, declined


def _trim_weights(
    lines: Sequence[str],
    line_seconds: Sequence[float] | None,
    *,
    words_per_minute: float,
) -> tuple[list[int], float]:
    """``(per-line integer weight, rate)`` for the trim search — words or seconds.

    The search only ever needs a summable integer per line plus a divisor that
    turns a total into seconds, so measured input is a UNIT CHANGE and not a
    second algorithm: deciseconds at :data:`TRIM_SECONDS_RATE`, for which
    ``total / rate * 60`` is exactly ``total / 10``.

    ``None`` reproduces the words path exactly, ``_safe_wpm`` and all. Measured
    input is validated hard instead — a length mismatch or a NaN/negative entry
    raises, because there is no defensible substitute for a measurement that is
    not one, and a silent mis-alignment would cut the script by another line's
    weights.
    """
    if line_seconds is None:
        return [len(line.split()) for line in lines], _safe_wpm(words_per_minute)
    if len(line_seconds) != len(lines):
        raise ValueError(
            f"line_seconds has {len(line_seconds)} entries for {len(lines)} line(s) — "
            "measured seconds must align with the script exactly, index by index"
        )
    for position, value in enumerate(line_seconds):
        if not _is_finite(value) or float(value) < 0.0:
            raise ValueError(
                f"line_seconds[{position}] is {value!r} — every measured line duration "
                "must be a finite, non-negative number of seconds"
            )
    # `round`, never `int`: truncation is a ONE-SIDED error, so it always
    # UNDER-states a line and the understatement accumulates across the script
    # (0.1s per line worst case, and every one of them in the same direction).
    # Rounding to the nearest decisecond keeps the error at worst 0.05s per line
    # and unbiased, which is the docstring's claim above.
    return [round(float(v) * TRIM_SECONDS_QUANTUM) for v in line_seconds], TRIM_SECONDS_RATE


def _kept_seconds(
    kept_indices: Sequence[int],
    *,
    line_seconds: Sequence[float] | None,
    weights: Sequence[int],
    rate: float,
) -> float:
    """The TRUE seconds of the kept lines — never the quantised search weight.

    The search runs on integer deciseconds because a subset-sum table needs
    integers; the REPORT must not. Quantising rounds each line by up to 0.05s,
    and the sum of those roundings can carry the plan across the very edge it
    claims to have landed inside: ``[55.02, 55.02, 40.0]`` against a 100s order
    weighs ``550 + 550`` deciseconds after dropping the tail, i.e. a reported
    ``110.0`` against a ``110.0`` limit with ``exceeds_limit=False``, while the
    kept lines really run ``110.04`` — a band the plan MISSED, reported as a
    band it made. So the weights decide WHICH lines to drop and this decides
    what the plan is allowed to claim about the ones that remain.

    With no measured input there is nothing to disagree with: the weights ARE
    the words, and this reproduces ``total / rate * 60`` exactly.
    """
    if line_seconds is None:
        return sum(weights[i] for i in kept_indices) / rate * 60.0
    return math.fsum(float(line_seconds[i]) for i in kept_indices)


def plan_trim(
    texts: Sequence[str],
    *,
    ordered_seconds: float,
    words_per_minute: float,
    tolerance: float = DEFAULT_TOLERANCE,
    protected_indices: Collection[int] = (),
    line_seconds: Sequence[float] | None = None,
) -> TrimPlan:
    """Plan which whole lines to drop so the estimate lands inside the band.

    It SEARCHES for a plan landing inside
    ``[ordered x (1 - tolerance), ordered x (1 + tolerance)]``, preferring the
    one that drops the FEWEST lines and, among equals, the one cutting the
    LATEST lines. It never drops a protected index, never drops the last
    surviving line, and keeps EVERYTHING when the draw is already within
    tolerance or short — a no-op plan, not an empty one.

    **Why this is a search and not a walk.** It used to be a greedy last-first
    walk that never reconsidered an accepted drop, and such a walk can spend the
    budget on a cheap early cut and then be unable to reach the band with what is
    left — after which it reported the ORDER as unreachable. Measured on 35/35/20
    words at 60 wpm against a 60s order (band 54-66s): the walk took the 20s
    tail, landed at 70.0s, set ``exceeds_limit`` and made its caller say "the
    ordered duration is not reachable by trimming this script" — while dropping
    EITHER 35-word line alone lands at 55.0s, inside the band. That is the third
    time this module's own class of defect has shipped; a greedy rule cannot be
    patched out of it, so the search stops guessing and enumerates.

    **The search, in two tiers.** Tier one is every SINGLE-line drop
    (:func:`_single_drop_sets`), scored unconditionally before any cost model is
    consulted: it is O(len(droppable)) and it is the overwhelmingly common
    answer. Tier two is the combinations — the estimate depends only on the
    surviving word count, so candidate drop sets collapse to a subset-sum table
    over the droppable lines' word counts (:func:`_drop_sets_by_dropped_words`),
    every subset represented at a cost of ``len(droppable) x distinct sums``
    rather than ``2^n``.

    **Only tier two is bounded, and the bound is MEASURED.** The table is grown
    incrementally and the walk stops when the transitions it has actually
    performed would exceed :data:`MAX_TRIM_SEARCH_STATES`; beyond that point
    :attr:`TrimPlan.search_exhausted` is False and the degradation text says the
    order was not reachable UNDER A BOUNDED SEARCH rather than claiming it is
    unreachable. The bound used to be an ESTIMATE from the total word count, and
    it over-counted a real 30-line audiobook script by ~1600x (270,840 predicted
    against 167 states built in 0.0007s): the whole search was skipped, and the
    greedy walk deleted 28 of 30 lines of a paid deliverable when dropping ONE
    landed inside the band. That is why tier one is outside the bound — this
    function must produce the one-line answer even if every cost model in it is
    wrong.

    **It is bounded on BOTH sides.** A candidate landing below
    ``ordered x (1 - tolerance)`` is never chosen — with unequal line lengths a
    drop can leap clean across the band (three lines of 40s against a 60s order:
    120 -> 80 -> 40, i.e. -33.3% reported as a successful trim). Overshooting to
    the far side is this pipeline's documented repeat failure, so such a
    candidate is DECLINED and the refusal is recorded on
    :attr:`TrimPlan.declined_under_band`.

    **When nothing lands.** Only after the whole candidate set has been scored
    and none of it is in the band does the plan come from the tail-first walk
    (:func:`_greedy_last_first`), with :attr:`TrimPlan.exceeds_limit` set — an
    honest overrun, reported alongside how many candidates were actually tried.

    **``line_seconds`` — plan on MEASURED time instead of a word proxy.** Words
    x wpm is the only thing this module could ever see, and it is a proxy: two
    lines of equal word count do not take equal time (a digit string, an
    acronym, a clause the voice pauses after), so the word-based plan can drop
    the wrong pair. Once a stage has real per-line seconds it passes them here
    and the identical search runs over them. The conversion is arithmetic, not a
    second algorithm: each line becomes an integer weight of DECISECONDS and the
    rate becomes :data:`TRIM_SECONDS_RATE`, for which ``weight / rate * 60`` is
    exactly ``weight / 10`` — see :data:`TRIM_SECONDS_QUANTUM` for why
    deciseconds and not centiseconds (the state bound, which centiseconds can
    exceed by ~9x on a 10-minute script, silently demoting the search).

    ``line_seconds=None`` is EVERY existing caller and is bit-for-bit the
    behaviour above. A length that does not match ``texts``, or any value that
    is not a finite ``>= 0`` number, raises ``ValueError``: these are measured
    inputs, and a mismatched or NaN one would silently mis-align seconds with
    lines and cut somebody's script by the wrong weights. Unlike a bad
    ``words_per_minute`` there is no defensible substitute for "the measurement
    you handed me is not a measurement".
    """
    lines = [str(t) for t in texts]
    words, rate = _trim_weights(lines, line_seconds, words_per_minute=words_per_minute)
    total_words = sum(words)
    before = total_words / rate * 60.0
    ordered = float(ordered_seconds)
    all_indices = tuple(range(len(lines)))
    # Out-of-range indices protect NOTHING — they name no line — so counting
    # them made `protected_count`, the number a floor report is acted on,
    # describe a script that does not exist: `protected_indices=(-1, 7)` on a
    # two-line script used to print "2 protected" beside a plan nothing pinned.
    protected = {i for i in (int(x) for x in protected_indices) if 0 <= i < len(lines)}
    no_op = TrimPlan(
        keep_indices=all_indices,
        estimated_seconds_before=before,
        estimated_seconds_after=before,
        ordered_seconds=ordered,
        dropped_indices=(),
        protected_count=len(protected),
    )
    # No order means no band means no judgement: `exceeds_limit` stays False
    # rather than inventing one (the module's "it never invents an order" rule).
    # A NaN or infinite order/tolerance takes the SAME path. Every comparison
    # against NaN is False, so `current <= limit` and `after < lower_bound` were
    # both always False and the walk deleted lines until one survived: three
    # 100-word lines against a 60s order came back as ONE line — 40.0s of a
    # 120.0s draw — with `exceeds_limit=False`, i.e. a silently amputated paid
    # deliverable reported as a success. There is no band to aim at here, so
    # there is nothing to do.
    if not lines or not _is_finite(ordered) or ordered <= 0.0 or not _is_finite(tolerance):
        return no_op

    tol = abs(float(tolerance))
    limit = ordered * (1.0 + tol)
    lower_bound = ordered * (1.0 - tol)
    if before <= limit:
        # Already inside the band, or short. Never pad, never cut for neatness.
        return no_op

    # Sorted lists, never the set itself: nothing here may depend on set order.
    droppable = [i for i in all_indices if i not in protected]
    max_drops = max(0, len(lines) - 1)
    table, exhausted = _candidate_drop_sets(words, droppable, max_drops=max_drops)

    in_band: tuple[int, tuple[int, ...]] | None = None
    in_band_after = before
    declined_under_band = False
    for dropped_words, entry in table.items():
        # `dropped_words == 0` is the do-nothing entry AND any drop set worth
        # zero words: a drop must SAVE time (`after < before`) or it is beats
        # deleted for nothing — the same rule the greedy walk enforces.
        if dropped_words <= 0:
            continue
        after = (total_words - dropped_words) / rate * 60.0
        if after < lower_bound:
            declined_under_band = True
            continue
        if after > limit:
            continue
        if in_band is None or _better_drop_set(entry, in_band):
            in_band, in_band_after = entry, after

    if in_band is not None:
        dropped_sorted = tuple(sorted(in_band[1]))
        kept = tuple(i for i in all_indices if i not in set(dropped_sorted))
        # The search said this lands; the REPORT is re-derived from the true
        # kept seconds, so a quantisation rounding cannot make the plan claim a
        # band it missed by a hundredth of a second (:func:`_kept_seconds`).
        in_band_after = _kept_seconds(kept, line_seconds=line_seconds, weights=words, rate=rate)
        return TrimPlan(
            keep_indices=kept,
            estimated_seconds_before=before,
            estimated_seconds_after=in_band_after,
            ordered_seconds=ordered,
            dropped_indices=dropped_sorted,
            exceeds_limit=in_band_after > limit,
            protected_count=len(protected),
            declined_under_band=declined_under_band,
            droppable_count=len(droppable) if max_drops >= 1 else 0,
            candidates_evaluated=max(0, len(table) - 1),
            search_exhausted=exhausted,
        )

    # Nothing lands. Every candidate has been scored, so the floor below is a
    # measured conclusion and not a greedy walk's guess.
    dropped_sorted, _walk_after, greedy_declined = _greedy_last_first(
        len(lines),
        words,
        [i for i in reversed(all_indices) if i not in protected],
        total_words=total_words,
        words_per_minute=words_per_minute,
        limit=limit,
        lower_bound=lower_bound,
        rate=None if line_seconds is None else rate,
    )
    kept = tuple(i for i in all_indices if i not in set(dropped_sorted))
    # Same rule as the in-band branch: the walk chooses the lines, the true kept
    # seconds decide what is reported and whether the floor exceeds the limit.
    # On the words path this is identical to the walk's own running total.
    floor_after = _kept_seconds(kept, line_seconds=line_seconds, weights=words, rate=rate)
    return TrimPlan(
        keep_indices=kept,
        estimated_seconds_before=before,
        estimated_seconds_after=floor_after,
        ordered_seconds=ordered,
        dropped_indices=dropped_sorted,
        exceeds_limit=floor_after > limit,
        protected_count=len(protected),
        declined_under_band=declined_under_band or greedy_declined,
        droppable_count=len(droppable) if max_drops >= 1 else 0,
        candidates_evaluated=max(0, len(table) - 1),
        search_exhausted=exhausted,
    )


class NarrationEstimate(BaseModel):
    """What the narration budget CONCLUDED, as numbers, for a persisted output.

    The reconciliation currently leaves no trace: a stage re-asks, trims, or
    accepts, and the run record keeps the resulting script and nothing about how
    it got there. So "why did this film come in at +26%?" is answerable only by
    re-deriving the arithmetic from the text — and the two inputs that decide it
    (the rate, and whether the rate was MEASURED or a default) are gone. This
    carries them.

    **Numbers only — never script text.** A stage output is read by the API and
    by the run list; putting lines in here would duplicate the script into a
    second field that drifts from the first, and it is the numbers that a query
    across runs can actually aggregate.

    A field is ``None`` where the fact does not exist, never a stand-in value:
    ``reask_kept`` is ``None`` when there was no re-ask (``False`` would claim a
    draw was discarded), ``measured_lines`` is ``None`` when nothing measured
    per-line audio, and a non-finite seconds/rate input lands as ``None``
    because ``NaN`` is not persistable as JSONB — :attr:`verdict` already
    carries "this was not a real number" as ``"unknown"``.
    """

    #: Final estimated spoken seconds, after any re-ask and any trim.
    estimate_seconds: float | None
    #: What was ordered. ``None``/``<= 0`` means nothing was ordered.
    ordered_seconds: float | None
    #: The rate every duration here was computed at.
    wpm: float | None
    #: WHERE that rate came from — e.g. ``"measured"`` / ``"default"``. The
    #: single most load-bearing field: the same estimate means something
    #: entirely different at a probed rate than at the 150.0 fallback, and the
    #: fallback has been measured to BEAT the probe.
    wpm_basis: str
    #: :func:`narration_verdict` against the OUTER +/-10% band — the same band
    #: that grades the delivered master, so this column and
    #: ``duration_conformance`` are comparable. Steering uses the tighter
    #: :func:`steer_verdict`; that is an aiming decision, not a grade.
    verdict: NarrationVerdict
    #: The estimate of the FIRST draw, before any correction. Equal to
    #: :attr:`estimate_seconds` when nothing was corrected.
    first_draw_estimate_seconds: float | None
    #: Whether the LLM was asked again for a shorter/longer draw.
    reasked: bool
    #: Whether the re-asked draw was KEPT (it is kept only when
    #: :func:`closer_to_order`). ``None`` when there was no re-ask.
    reask_kept: bool | None
    #: How many whole lines the trim dropped. ``0`` is a no-op plan.
    trimmed_lines: int
    #: How many lines had REAL measured seconds behind them. ``None`` is the
    #: pre-measurement world (the estimate is words x wpm end to end);
    #: a number below the line count means the estimate is a mix.
    measured_lines: int | None = None


def build_narration_estimate(
    *,
    estimate_seconds: float,
    ordered_seconds: float,
    wpm: float,
    wpm_basis: str,
    first_draw_estimate_seconds: float | None = None,
    reasked: bool = False,
    reask_kept: bool | None = None,
    trimmed_lines: int = 0,
    measured_lines: int | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> NarrationEstimate:
    """Assemble a :class:`NarrationEstimate`, deriving the verdict here.

    ``verdict`` is derived rather than accepted as an argument on purpose: a
    caller passing its own word would let the recorded verdict disagree with the
    recorded numbers, and this record exists to be trusted after the fact. It is
    :func:`narration_verdict` — the OUTER band — because the persisted verdict
    should be comparable with what grades the delivered master, not with the
    tighter number the stage steered by.

    Tolerant of non-finite input by construction: every seconds/rate field is
    passed through the same finiteness filter, so a NaN estimate is recorded as
    ``None`` with ``verdict="unknown"`` instead of poisoning a JSONB write,
    and ``first_draw_estimate_seconds`` defaults to the final estimate (nothing
    was corrected, so the first draw IS the estimate).
    """

    def _number(value: float | None) -> float | None:
        return float(value) if value is not None and _is_finite(value) else None

    first_draw = (
        estimate_seconds if first_draw_estimate_seconds is None else (first_draw_estimate_seconds)
    )
    return NarrationEstimate(
        estimate_seconds=_number(estimate_seconds),
        ordered_seconds=_number(ordered_seconds),
        wpm=_number(wpm),
        wpm_basis=str(wpm_basis),
        verdict=narration_verdict(estimate_seconds, ordered_seconds, tolerance=tolerance),
        first_draw_estimate_seconds=_number(first_draw),
        reasked=bool(reasked),
        reask_kept=None if reask_kept is None else bool(reask_kept),
        trimmed_lines=int(trimmed_lines),
        measured_lines=None if measured_lines is None else int(measured_lines),
    )


def narration_degradation(
    *,
    stage: str,
    ordered_seconds: float,
    estimated_seconds: float,
    action: str,
    detail: str = "",
) -> Degradation:
    """One entry for the existing ``degradations`` channel.

    Reuses ``degradation.DURATION_OUT_OF_TOLERANCE`` rather than minting a new
    code: the fact being reported IS "this length is outside the ordered
    duration's tolerance band", measured on the narration estimate instead of on
    the delivered master, and the run-level consequence (``degraded``) is
    identical. Minting a code would mean editing
    ``shared/degradation.py``'s ``DEGRADATION_CODES``, which this lane does not
    own — and a code absent from that frozenset makes ``degradation()`` raise.

    The run-specific numbers go in ``detail``, never in ``code`` — that split is
    the whole point of the channel (``code`` is the queryable half).
    """
    deviation = ""
    if float(ordered_seconds) > 0.0:
        pct = (float(estimated_seconds) - float(ordered_seconds)) / float(ordered_seconds) * 100.0
        deviation = f" ({pct:+.1f}%)"
    suffix = f" {detail}" if detail else ""
    return degradation(
        DURATION_OUT_OF_TOLERANCE,
        f"{stage}: narration estimated at {float(estimated_seconds):.1f}s against an "
        f"ordered {float(ordered_seconds):.1f}s{deviation}; action={action}.{suffix}",
    )


def narration_estimate_advisory(
    *,
    stage: str,
    ordered_seconds: float,
    estimated_seconds: float,
    action: str,
    detail: str = "",
) -> Degradation:
    """:func:`narration_degradation`'s text, on the ADVISORY tier.

    Same parameters, same sentence, same numbers — the ONLY difference is the
    code, and therefore which field it rides and whether it flips ``degraded``.

    Why the identical fact changes tier here: this is a PRE-AUDIO estimate.
    ``words / wpm`` carries several percent of its own error, so it grades the
    delivery it cannot see. A lesson run measured 2026-09-11 delivered +0.22% of
    its order — conformant by any standard — while its script stage had already
    stamped ``duration_out_of_tolerance`` from the estimate. The verdict that
    can be trusted is the one the compile stages already raise from the real
    artifact (``duration_conformance.py``), so this stops being a second, worse
    opinion on the same question and becomes what it always was: a note about
    how the script was AIMED.

    Nothing is lost. The entry is still structured, still stably coded, still
    queryable — it simply no longer makes a good film read broken.

    :func:`narration_degradation` is deliberately left in place and untouched:
    stages move to this one by editing their own call, one at a time, and until
    they do their behaviour must not change under them.
    """
    return advisory(
        NARRATION_ESTIMATE_OFF_TARGET,
        _narration_estimate_detail(
            stage=stage,
            ordered_seconds=ordered_seconds,
            estimated_seconds=estimated_seconds,
            action=action,
            detail=detail,
        ),
    )


def _narration_estimate_detail(
    *,
    stage: str,
    ordered_seconds: float,
    estimated_seconds: float,
    action: str,
    detail: str = "",
) -> str:
    """The detail sentence shared by the degradation and the advisory.

    A deliberate copy of :func:`narration_degradation`'s body rather than a
    refactor of it: that function is live on three stages and this lane may not
    change what any existing function emits, not even by a whitespace. The copy
    is safe because it is pinned — a test asserts the two strings are identical
    for the same arguments, so a drift is a failure and not a discovery.
    """
    deviation = ""
    if float(ordered_seconds) > 0.0:
        pct = (float(estimated_seconds) - float(ordered_seconds)) / float(ordered_seconds) * 100.0
        deviation = f" ({pct:+.1f}%)"
    suffix = f" {detail}" if detail else ""
    return (
        f"{stage}: narration estimated at {float(estimated_seconds):.1f}s against an "
        f"ordered {float(ordered_seconds):.1f}s{deviation}; action={action}.{suffix}"
    )


def _floor_reason(plan: TrimPlan) -> str:
    """Name the constraint that ACTUALLY bound this trim, in priority order.

    The old text had exactly two branches — "landed UNDER the band", else
    "N line(s) are protected" — so a floor with nothing protected printed
    ``"0 line(s) dropped, 0 protected, and 0 line(s) are protected from
    trimming"``: self-contradictory (it names protection as the reason while
    reporting there was none) and silent about the real one. A single 500-word
    line against a 60s order floors because the LAST SURVIVING LINE IS NEVER
    DROPPED, which that sentence never says, and it printed beside
    ``1 droppable line(s)`` when the search was allowed to drop none.

    Order matters: a structural impossibility (nothing was droppable at all)
    outranks a refusal (a drop existed and was declined), which outranks
    protection, which outranks "the combinations simply do not reach it".
    """
    total_lines = len(plan.keep_indices) + len(plan.dropped_indices)
    if plan.droppable_count == 0:
        if total_lines <= 1:
            return "the script is a single line and the last survivor is never dropped"
        if plan.protected_count >= total_lines:
            return f"all {total_lines} line(s) are protected from trimming"
        return "no line was droppable"
    if plan.declined_under_band:
        return "every remaining drop would have landed UNDER the band"
    if plan.protected_count > 0:
        return f"{plan.protected_count} of {total_lines} line(s) are protected from trimming"
    if len(plan.keep_indices) <= 1:
        return "the last surviving line alone still exceeds the band"
    return "no combination of the droppable lines lands inside the band"


def trim_floor_degradation(*, stage: str, plan: TrimPlan) -> Degradation:
    """The trim RAN and could not reach the band. One entry naming the floor.

    Reuses ``ORDERED_DURATION_INFEASIBLE`` rather than minting a code: the fact
    IS "the ordered duration is unreachable given a hard floor", the floor here
    being protected lines (or the lower conformance bound) instead of the
    guaranteed per-scenario recording length. The consequence is identical and a
    query asking "which runs could not be made to fit?" wants both.

    Only ever emitted for a plan with ``exceeds_limit`` — a trim that landed in
    the band is not a degradation, and a signal that fires on successes is one
    people learn to ignore.

    The text names WHAT WAS TRIED: how many candidate drop sets the search
    scored, over how many droppable lines, whether that search was exhaustive,
    and the band itself. "Not reachable" is a strong claim about a paid order,
    and it was previously unfalsifiable prose written by a greedy walk that had
    examined one candidate per line and reconsidered none of them. A reader must
    be able to tell the difference between "every subset was scored and none
    lands" and "a bounded search did not find one".
    """
    ordered = float(plan.ordered_seconds)
    after = float(plan.estimated_seconds_after)
    deviation = f" ({(after - ordered) / ordered * 100.0:+.1f}%)" if ordered > 0.0 else ""
    reason = _floor_reason(plan)
    tol = DEFAULT_TOLERANCE
    band = f"[{ordered * (1.0 - tol):.1f}s, {ordered * (1.0 + tol):.1f}s]" if ordered > 0.0 else ""
    searched = (
        f"Scored {plan.candidates_evaluated} candidate drop set(s) over "
        f"{plan.droppable_count} droppable line(s)"
        f"{' (exhaustively)' if plan.search_exhausted else ' (BOUNDED search, not exhaustive)'}"
        f"; none lands inside {band}."
        if band
        else ""
    )
    verdict = (
        "The ordered duration is not reachable by trimming this script."
        if plan.search_exhausted
        else "The ordered duration was not reached under a bounded search of this script."
    )
    return degradation(
        ORDERED_DURATION_INFEASIBLE,
        f"{stage}: narration trim FLOORED at {after:.1f}s against an ordered "
        f"{ordered:.1f}s{deviation} — {len(plan.dropped_indices)} line(s) dropped, "
        f"{plan.protected_count} protected, and {reason}. {searched} {verdict}",
    )


def speech_rate_degradation(*, stage: str, words_per_minute: float) -> Degradation:
    """The rate handed to this module was unusable, so a substitute was used.

    Mints nothing generic: every duration number the stage went on to produce —
    estimate, word budget, trim — was computed at :data:`DEFAULT_WORDS_PER_MINUTE`
    instead of a measured rate, and the entry says so with the value that was
    rejected. A silent substitution here is the defect that let a 60-wpm clamp
    inflate a 190-word script to a 190s estimate and trim a paid deliverable down
    to ~26s.
    """
    return degradation(
        SPEECH_RATE_UNUSABLE,
        f"{stage}: speech rate {float(words_per_minute):.1f} wpm is outside the usable "
        f"band [{WORDS_PER_MINUTE_FLOOR:.0f}, {WORDS_PER_MINUTE_CEILING:.0f}] — every "
        f"narration duration on this run was budgeted at the "
        f"{DEFAULT_WORDS_PER_MINUTE:.0f} wpm default instead of a measured rate.",
    )
