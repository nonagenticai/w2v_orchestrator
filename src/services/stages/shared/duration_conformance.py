"""Delivered-vs-ORDERED duration conformance — the comparison nothing was making.

Nothing anywhere compared the length of the master a stage just shipped to the
duration the customer ORDERED. Measured on live artifacts 2026-09-10 (ffprobe-confirmed):
``website_review`` delivered **273%** of its ordered duration, ``lesson`` 216%,
``app_review`` 140%, ``promo`` 129% — and not one run record contains a single
field, note or log line saying so. A conformance failure that is invisible in
the run record can never be regression-tested, so this module makes it visible.

⚠️ WHICH NUMBER IS "DELIVERED" — this module used to get it wrong
-----------------------------------------------------------------
This docstring used to assert that "the compiler ffprobes the mux and the stage
echoes that number out as ``duration_seconds``". **That is false**, and it is
the belief that made this whole check vacuous. ``primitives/compile_stage.py:192-195``
says it plainly: ``duration_seconds`` **is the nominal sum of shot durations**
— the INTENDED timeline — and the ffprobe of the encoded output is a *different*
field, ``actual_duration_seconds``. Verified on 38 live production runs
2026-09-13: the recorded ``delivered_seconds`` was byte-identical to
``duration_seconds`` on every one, and the probed value was never persisted. So
the verdict was the plan checking itself, and it is structurally incapable of
catching the failure it exists for — a master whose picture stopped early
declares exactly the length it meant to.

Hence the basis ladder (:data:`DELIVERED_BASIS_PLAYABLE` →
:data:`DELIVERED_BASIS_DECLARED`) and the keyword-only ``playable_seconds``: a
caller that has the PLAYABLE measurement passes it and the verdict is MEASURED;
a caller that does not still gets a verdict, but it is stamped ``declared`` and
carries a
:data:`~src.services.stages.shared.degradation.DURATION_UNMEASURED` **advisory**
so an unverified verdict is never byte-identical to a verified one.

⚠️ AND "MEASURED" WAS NOT ENOUGH — the correction this module shipped SECOND
------------------------------------------------------------------------------
Preferring ``actual_duration_seconds`` fixed the plan-grading-itself defect and
introduced its mirror image, live for part of 2026-09-14. That field is
``min(video_stream, container)``
(``video_compiler_service/src/services/renderer.py::_resolve_probed_duration``,
whose own docstring says the stream wins precisely because ``format=duration``
is the MAX over streams), so it is the **PICTURE** length, not the playable one.
On a 120s order whose plan declared 180s and whose picture truncated at 118s
under 180s of narration, the verdict went from ``+50% → OUT_OF_TOLERANCE`` (the
defect CAUGHT, by luck, on the declared number) to ``-1.7% → within_tolerance``
— a broken, over-long master reported as an on-target delivery, by the very
check written to catch it.

The fix is not to pick the other pessimistic scalar. **One number cannot carry
both facts.** ``delivered_seconds`` is now the PLAYABLE length —
``max(video_stream_seconds, audio_stream_seconds)`` from the compiler's
mastering report, which probes the two streams SEPARATELY for exactly this
reason — and "the picture stopped before the audio" is its own signal with its
own code, :func:`picture_truncation_degradations` /
:data:`~src.services.stages.shared.degradation.DURATION_PICTURE_TRUNCATED`. On
that master the two now say, together and truthfully: 180s delivered against a
120s order (+50%, out of tolerance) AND the picture is 62s short of the audio.

⛔ AND THERE IS NO PICTURE-ONLY RUNG — the third and final correction
------------------------------------------------------------------------------
An intermediate revision kept ``actual_duration_seconds`` as a SECOND rung,
below ``playable`` and above ``declared``, on the reasoning that a picture-only
measurement still beats the plan grading itself. **It does not, and the rung is
gone.**

``actual_duration_seconds`` is ``min(video_stream, container)``, so it is
*always* ``<= playable`` and differs from it only when the audio outruns the
video — i.e. **only on a truncated master**. A rung that is identical to the one
above it except on defective files, where it is WRONG, contributes nothing and
costs everything. Worse, the population that falls to it is precisely the
defective one: ``mastering.degraded_report()``
(``video_compiler_service/src/services/mastering.py``) writes ``None`` into BOTH
stream fields when ``post_mux_verify`` crashes, so rung 1 is unavailable for
exactly the masters whose verification failed, and the picture-only rung caught
them all and under-reported every one.

DECLARED is strictly safer than the picture-only probe. The declared plan is the
full INTENDED length, so it still catches an over-long delivery; the picture-only
number makes a truncated over-long master look on-target. "We could not measure
it" is an honest verdict that says so on the record and carries
``DURATION_UNMEASURED``; "we measured the wrong half of it" is not a verdict at
all. When mastering is unusable this module now says it could not measure,
rather than quietly grading the half it could see.

⚠️ ``DURATION_UNMEASURED`` is an ADVISORY, not a degradation, and that tiering is
load-bearing — see :func:`duration_conformance_advisories`. "Nobody probed the
file" is a fact about our instrumentation, not a defect in the artifact, and it
fires on every stage that cannot reach a probe: ``demo_video`` and ``showcase``
route through ``demo_video/encode_master.py``, whose ``EncodeMasterOutput`` has
no ``actual_duration_seconds``, so as a degradation this would flip
``degraded=True`` on 100% of two content types forever, conforming masters
included. That is the near-universal-flag failure the advisory tier exists to
prevent. ``DURATION_OUT_OF_TOLERANCE`` is unaffected and stays a degradation.

⛔ ``0.0`` is the compiler's "unknown" sentinel, NOT a zero-length video:
``video_compiler_service/src/services/renderer.py:668-673`` returns ``0.0`` from
``probe_output_duration`` on ANY probe failure or timeout, deliberately, so a
bad probe cannot fail a good render. Treating it as a real measurement would
report every probe failure as a 0% delivery — a catastrophic false positive
manufactured out of a missing measurement. It falls back to the declared value
and stamps ``declared``. ``None`` arrives the same way from audio-concat
results, which have no video mux to probe.

Scope, deliberately narrow:

* **Never fatal.** Nothing here raises, changes a stage status, or truncates a
  master.
* **It DOES surface the run as degraded** when the delivered length misses the
  ordered one, AND when nothing usable was ordered at all — via
  :func:`duration_conformance_degradations`, which the compile stages record on
  the ``degradations`` channel (``shared/degradation.py``). The "nothing
  ordered" case emits ``DURATION_NOT_ORDERED`` rather than staying silent: a
  ``duration_conformance: null`` field used to be invisible to any
  ``within_tolerance = false`` query, exactly the class of gate-that-cannot-fail
  this module exists to remove.

  ⚠️ This module used to state the opposite, and to give a REASON: it
  "deliberately avoids ``notes`` because that would surface the run as
  ``degraded``". That reasoning was wrong, and it was wrong in the way this
  whole module exists to attack. A 273%-of-ordered master IS a degraded
  delivery; a run record that reports it clean is not being conservative, it is
  lying about a paid artifact. What that argument actually got right is a
  different thing — that this must not turn a run RED — and that distinction is
  now expressible, because degradation has a channel of its own: ``degraded``
  becomes true, ``status`` stays ``completed``, nothing fails. The structured
  ``duration_conformance`` field stays exactly as it is (it carries the
  numbers); the degradation entry is the queryable verdict beside it.
* **"Nothing was ordered" is not "on target".** :func:`duration_conformance`
  returns ``None`` when no usable numeric order exists, so a reader can never
  mistake an absent order for a conforming delivery. That distinction is the
  same one ``_compile_duration_qa_notes`` had to learn the hard way, where an
  unprobed length was byte-identical to a verified one.

Why this is NOT ``evaluation/rubrics/script_craft.py::_duration_budget``
-----------------------------------------------------------------------
That function is the only delivered-vs-ordered arithmetic that existed, and it
is the wrong tool here on three counts, so it is deliberately not reused:

1. It compares **word counts** (``total_words`` vs ``target_seconds * WPM / 60``)
   — a PRE-production proxy for how long a *script* should take to read. This
   module compares **measured encoded seconds** of a finished master. A script
   inside its word budget can still deliver a 273% master.
2. It returns a scored ``MetricResult`` for a rubric weighting, not a plain
   conformance record a stage output can carry.
3. It is reachable only from ``script_doctor``, registered only for
   ``demo_video`` and ``showcase`` — i.e. none of the four blueprints measured
   above.

``_duration_budget`` is not edited or imported by this module.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, NotRequired, TypedDict

from src.services.stages.action_reel.target_duration import target_seconds
from src.services.stages.shared.degradation import (
    DURATION_NOT_ORDERED,
    DURATION_OUT_OF_TOLERANCE,
    DURATION_PICTURE_TRUNCATED,
    DURATION_PLAN_INFEASIBLE,
    DURATION_UNMEASURED,
    ORDERED_DURATION_INFEASIBLE,
    Degradation,
    advisory,
    degradation,
)

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

logger = logging.getLogger(__name__)

# Conformance band for delivered-vs-ordered duration: a master is "on target"
# when its measured length is within ±10% of what was ordered. 10% is the same
# band ``script_craft._duration_budget`` treats as a full-score word budget, so
# the pre-production and post-production checks agree on what "close enough"
# means. It is a REPORTING threshold only — nothing fails outside it.
#
# ⛔ THREE DIFFERENT NUMBERS GRADE THREE DIFFERENT OBJECTS, and they are NOT a
# discrepancy to be unified. A shared constant here would be WRONG:
#
# * THIS one (0.10) grades the PROBED FILE — delivered playable seconds against the
#   ordered seconds, after the render is paid for.
# * ``time_budget.DEFAULT_TOLERANCE_PCT`` (2.0, i.e. 2%) grades the SOLVED TIMELINE
#   from the SHORT side only — ``time_budget.is_underfull`` asks whether the summed
#   allotments fall under target by more than 2%. Different object (a plan, not a
#   file), different direction (underfull only).
# * the INFEASIBLE verdict comes from NEITHER. ``time_budget.infeasible_beats`` is a
#   bare ``Σ min > target`` with NO tolerance at all: a plan whose floors alone
#   exceed the order is refused outright, because seconds can be moved but never
#   created.
#
# The third is why this module now carries :data:`DURATION_PLAN_INFEASIBLE`. A plan
# refused at zero tolerance can still deliver a file inside +/-10%, and reporting
# that as a bare pass is how prod run ``01a0a3c6`` shipped a rejected timeline with a
# reassuring "within +/-10%" beside it. Widening or narrowing any of the three to
# make them agree would only move which of the two verdicts lies.
DURATION_TOLERANCE_FRACTION = 0.10

# Where a PRE-PRODUCTION narration estimate should AIM, which is not the same
# number as the band that grades the delivered master above.
#
# Aiming at the tolerance band means aiming at its EDGE: anything under +10% is
# "ok", so a draw estimated at +9% is accepted and nothing trims it. The
# estimate is words / wpm and carries several percent of error of its own, so an
# accepted +9% plus a 5% under-estimate delivers +14% — outside the band that
# actually grades it. Measured n=5 per type on 2026-09-11: app_review spans
# -20.8%..+12.9%, lesson +4.2%..+26.4%, promo +4.5%..+32.4%. Steering to +4%
# leaves the estimate's own error room to be absorbed inside +/-10%.
#
# LONG SIDE ONLY, and that asymmetry is the point. Nothing in this pipeline pads
# an underrun — ``narration_budget`` has no "extend" plan by design, because
# repeating or stretching audio to hit a number is a defect — so a short
# estimate has no lever to pull anyway. Steering the short side to -4% would
# only make the trim REFUSE cuts it should take (it declines anything landing
# under the lower bound), turning a correctable overrun into an accepted one.
# The short side therefore keeps the full -10% tolerance: see
# ``narration_budget.steer_verdict``, which uses this constant for "long" and
# DURATION_TOLERANCE_FRACTION for "short".
DURATION_STEER_FRACTION = 0.04

#: ``delivered_seconds`` is the PLAYABLE length — ``max(video_stream_seconds,
#: audio_stream_seconds)`` from the compiler's mastering report. This is what the
#: customer's player reports and the only basis that is correct on a truncated
#: master. The container bounds its own streams, so the longer stream IS the
#: playable length. It is the ONLY measured basis; there is deliberately no
#: picture-only rung beneath it (see the module docstring's third correction, and
#: :func:`duration_conformance`).
DELIVERED_BASIS_PLAYABLE = "playable"
#: ``delivered_seconds`` came from the compiler's DECLARED timeline (the nominal
#: sum of shot durations). The plan is checking itself: the verdict is
#: UNVERIFIED, never a pass, and it carries the ``DURATION_UNMEASURED``
#: ADVISORY (:func:`duration_conformance_advisories`) — reported and queryable,
#: never degrading.
DELIVERED_BASIS_DECLARED = "declared"


class DurationConformance(TypedDict):
    """One delivered-vs-ordered comparison, as recorded on a stage output."""

    ordered_seconds: float
    delivered_seconds: float
    #: delivered / ordered — 1.0 is exact, 2.73 is the measured website_review.
    ratio: float
    #: Signed percentage: positive = over-delivered (too long), negative = short.
    deviation_pct: float
    #: ``abs(deviation_pct) <= DURATION_TOLERANCE_FRACTION * 100``.
    within_tolerance: bool
    #: Where ``delivered_seconds`` CAME FROM, best first:
    #: :data:`DELIVERED_BASIS_PLAYABLE` (``max`` of the two separately probed
    #: streams — the length a player reports) or
    #: :data:`DELIVERED_BASIS_DECLARED` (the compiler's nominal timeline, i.e.
    #: the plan grading itself). Without this they are byte-identical on the
    #: record and an unverified verdict reads exactly like a verified one — the
    #: defect this field exists to end.
    #:
    #: ⚠️ ``"probed"`` was a third value here for part of 2026-09-14 and is
    #: RETIRED — nothing writes it any more. It meant ``actual_duration_seconds``,
    #: a PICTURE-ONLY number that under-reports exactly the truncated masters this
    #: record exists to catch. A reader of a persisted row from that window must
    #: treat ``"probed"`` as a LOWER BOUND, not a delivered length.
    delivered_basis: str
    #: True when the PLAN this master was produced from had already been refused
    #: upstream — ``reconcile_timeline`` classified ``convergence_outcome`` as
    #: ``cant_converge``/``progress``/``relief_failed`` and/or raised
    #: ``ordered_duration_infeasible``. It qualifies :attr:`within_tolerance`
    #: rather than replacing it: the band still says what it measured, but a
    #: ``True`` here means the band's pass is NOT a clean one.
    #:
    #: ⚠️ ``NotRequired``, and deliberately so. Rows persisted before 2026-09-15
    #: carry neither key, and a reader that indexes them raises ``KeyError`` on a
    #: replay/regenerate of an old run. Read both through ``.get()``; absent means
    #: "nobody asked", which is NOT the same fact as ``False`` ("asked, and the
    #: plan was fine") — but both are safely non-alarming.
    plan_infeasible: NotRequired[bool]
    #: The upstream token that set :attr:`plan_infeasible` — the raw
    #: ``convergence_outcome`` value, the degradation code, or both joined by
    #: ``"+"``. ``None`` when the plan was not refused. The queryable half lives on
    #: the advisory ``code``; this is the "which upstream said so" detail.
    plan_infeasible_reason: NotRequired[str | None]


def _usable(value: object) -> float | None:
    """Coerce to a finite float, or ``None`` if the value cannot be one.

    ``bool`` is excluded deliberately: ``duration_minutes: true`` is corrupt
    config, not a 1-minute order. The accepted set is narrowed BEFORE the call
    rather than papered over with a type-ignore, so ty checks the conversion.
    """
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        num = float(value)
    except ValueError:
        return None
    return num if math.isfinite(num) else None


def duration_conformance(
    ordered_seconds: float | None,
    delivered_seconds: float,
    *,
    playable_seconds: object = None,
    plan_infeasible: str | None = None,
) -> DurationConformance | None:
    """Compare a delivered master's PLAYABLE length against the ordered length.

    Returns ``None`` — never a zero-valued record and never
    ``within_tolerance=True`` — when no usable numeric duration was ordered
    (absent, non-numeric, non-finite, zero or negative). "Nobody ordered a
    length" and "the length ordered was met" are different facts and only one of
    them is safe to report as conforming; a zero or negative order is also the
    only shape that could divide by zero, and it exits here before the division.

    ⚠️ ONE NUMBER CANNOT CARRY TWO FACTS. ``delivered_seconds`` answers "how long
    does this play for", and nothing else. "The picture stopped before the audio"
    is a SEPARATE signal with its own code
    (:func:`picture_truncation_degradations`). Conflating them is what made this
    function's previous revision able to report a truncated master as on target,
    so the selection below deliberately prefers the LONGEST credible measurement
    rather than the most pessimistic one.

    Selection order — TWO rungs, and only two:

    1. *playable_seconds* — ``max(video_stream_seconds, audio_stream_seconds)``
       from the compiler's mastering report
       (:func:`playable_seconds_from_mastering`), taken iff it is usable and
       strictly ``> 0``. The container bounds its streams, so the longer stream
       is what a player reports. Stamped :data:`DELIVERED_BASIS_PLAYABLE`.
    2. the DECLARED *delivered_seconds* — the plan grading itself. Stamped
       :data:`DELIVERED_BASIS_DECLARED` and carries the ``DURATION_UNMEASURED``
       advisory (:func:`duration_conformance_advisories`).

    ⛔ THERE IS NO PICTURE-ONLY RUNG BETWEEN THEM, and its absence is the fix.
    ``actual_duration_seconds`` is ``min(video_stream, container)``, hence always
    ``<= playable``, hence different from rung 1 ONLY when the audio outruns the
    picture — i.e. only on a truncated master, where it is the wrong number. And
    the population that reaches rung 2 is exactly the suspect one:
    ``mastering.degraded_report()`` nulls BOTH stream fields when
    ``post_mux_verify`` crashes, so a picture-only rung would have graded every
    verification-failed master on its truncated half. **Declared is strictly
    safer:** it is the full INTENDED length, so it still catches an over-long
    delivery, whereas the picture-only number makes a truncated over-long master
    look on-target. "We could not measure it" is an honest verdict; "we measured
    the wrong half of it" is not.

    Anything that is not a usable positive float falls THROUGH to rung 2 —
    ``None`` (audio-concat results have no mux; an absent or degraded mastering
    block), ``0.0`` (the compiler's "unknown" sentinel on any probe failure,
    ``renderer.py:668-673``), NaN, a bool, a non-numeric. The strict ``> 0`` is
    the whole point of the sentinel rule: accepting ``0.0`` would turn every
    failed probe into a reported 0%-of-order delivery, a catastrophic verdict
    manufactured out of a missing measurement. Neither rung is ever taken
    silently — both are named on the record and in the log line, and rung 2
    additionally raises ``DURATION_UNMEASURED``.

    *playable_seconds* is keyword-only, so the existing positional call sites
    stay source-compatible; they simply keep getting ``declared``.

    ⛔ *plan_infeasible* DOES NOT MOVE ``within_tolerance``, and that restraint is
    the design. It is the upstream reason token (see :func:`plan_infeasible_reason`)
    for a plan that was already refused before this file existed — and a refused
    plan can still deliver a file inside the band, which is exactly what prod run
    ``01a0a3c6`` did (+6.79% on a 30.0s order, against a timeline stamped
    ``cant_converge`` with ``feasible: [False, False, False]``). Flipping the boolean
    would raise ``DURATION_OUT_OF_TOLERANCE`` — a DEGRADATION, on a MEASUREMENT that
    did not fail, duplicating an ``ordered_duration_infeasible`` the planning stage
    already raised. Post-rollout sample size for this signal is ONE run. So the fact
    is recorded beside the verdict, carried to the log line
    (:func:`observe_duration_conformance`) and to the ADVISORY tier
    (:func:`duration_conformance_advisories`), and nothing that passed today stops
    passing. Arming it is a separate, evidence-backed decision.
    """
    ordered = _usable(ordered_seconds)
    playable = _usable(playable_seconds)
    if playable is not None and playable > 0:
        delivered = playable
        basis = DELIVERED_BASIS_PLAYABLE
    else:
        delivered = _usable(delivered_seconds)
        basis = DELIVERED_BASIS_DECLARED
    if ordered is None or ordered <= 0 or delivered is None:
        return None
    ratio = delivered / ordered
    # The verdict is taken from the ROUNDED deviation, the same number recorded
    # on the output, so the record can never disagree with the boolean beside
    # it. Unrounded, an exactly-10% delivery lands on 10.000000000000009 and is
    # reported as a violation next to a recorded "10.0%".
    deviation_pct = round((ratio - 1.0) * 100.0, 2)
    return DurationConformance(
        ordered_seconds=round(ordered, 3),
        delivered_seconds=round(delivered, 3),
        ratio=round(ratio, 4),
        deviation_pct=deviation_pct,
        within_tolerance=abs(deviation_pct) <= DURATION_TOLERANCE_FRACTION * 100.0,
        delivered_basis=basis,
        plan_infeasible=bool(plan_infeasible),
        plan_infeasible_reason=plan_infeasible or None,
    )


def observe_duration_conformance(
    stage: str,
    ordered_seconds: float | None,
    delivered_seconds: float,
    *,
    playable_seconds: object = None,
    plan_infeasible: str | None = None,
) -> DurationConformance | None:
    """:func:`duration_conformance` plus the log line, for a stage to call once.

    WARNING when the delivered master is outside the tolerance band, INFO when
    it conforms, DEBUG when nothing numeric was ordered (that is not a defect —
    a brief may legitimately not specify a length). Never raises: a conformance
    observation must not be able to break a render that already succeeded.

    Every non-DEBUG line NAMES THE BASIS (``basis=playable`` / ``basis=declared``,
    see :func:`duration_conformance`), because a reader scrolling a log cannot
    otherwise tell a measured verdict from the plan agreeing with itself — and a
    ``declared`` INFO line saying "within +/-10%" is precisely the reassuring,
    unfounded statement this module exists to stop emitting silently.

    The basis token is emitted RAW, with no qualifier, and that is now safe
    because both surviving tokens mean what they say. An earlier revision had to
    decorate ``probed`` with "(PICTURE-ONLY; playable length unmeasured)" to stop
    it reading as "we measured it" — the decoration was a symptom of a rung that
    should not have existed, and it went with the rung.
    """
    result = duration_conformance(
        ordered_seconds,
        delivered_seconds,
        playable_seconds=playable_seconds,
        plan_infeasible=plan_infeasible,
    )
    if result is None:
        logger.debug(
            "duration conformance: %s ordered no usable duration; delivered %.1fs (not checked)",
            stage,
            _usable(delivered_seconds) or 0.0,
        )
        return None
    if result["within_tolerance"] and result.get("plan_infeasible"):
        # ⛔ THE REASSURING INFO IS SUPPRESSED HERE, and this branch is the whole
        # point of `plan_infeasible`. The band genuinely passed, so the numbers are
        # unchanged and `within_tolerance` is still True — but the plan behind them
        # was REFUSED upstream at zero tolerance, and "within +/-10%" printed alone
        # under that is the whitewash prod run `01a0a3c6` shipped. WARNING, not INFO,
        # because a reader scrolling a log must not be able to take this for a clean
        # pass; the band and the refusal are both named so neither can be quoted
        # without the other.
        logger.warning(
            "duration conformance: %s delivered %.1fs vs %.1fs ordered "
            "(%.0f%% of order, %+.1f%% deviation) — inside +/-%.0f%% but the PLAN was "
            "already refused upstream (%s), so this is NOT a clean pass: the band "
            "grades the probed file, the refusal was a bare sum-of-floors > order with "
            "no band at all [basis=%s]",
            stage,
            result["delivered_seconds"],
            result["ordered_seconds"],
            result["ratio"] * 100.0,
            result["deviation_pct"],
            DURATION_TOLERANCE_FRACTION * 100.0,
            result.get("plan_infeasible_reason") or "reason unrecorded",
            result["delivered_basis"],
        )
    elif result["within_tolerance"]:
        logger.info(
            "duration conformance: %s delivered %.1fs vs %.1fs ordered "
            "(%.0f%% of order, %+.1f%% deviation) — within +/-%.0f%% [basis=%s]",
            stage,
            result["delivered_seconds"],
            result["ordered_seconds"],
            result["ratio"] * 100.0,
            result["deviation_pct"],
            DURATION_TOLERANCE_FRACTION * 100.0,
            result["delivered_basis"],
        )
    else:
        logger.warning(
            "duration conformance: %s delivered %.1fs vs %.1fs ordered "
            "(%.0f%% of order, %+.1f%% deviation) — OUTSIDE +/-%.0f%% [basis=%s]",
            stage,
            result["delivered_seconds"],
            result["ordered_seconds"],
            result["ratio"] * 100.0,
            result["deviation_pct"],
            DURATION_TOLERANCE_FRACTION * 100.0,
            result["delivered_basis"],
        )
    return result


def duration_conformance_degradations(
    result: DurationConformance | None,
) -> list[Degradation]:
    """The delivered-vs-ordered verdict as a structured degradation entry.

    Three outcomes:

    * *result* is ``None`` -> one :data:`~src.services.stages.shared.degradation.DURATION_NOT_ORDERED`
      entry. This is the CHANGE this function exists to make: ``None`` used to
      yield ``[]`` here, which is exactly how a `showcase_v1`/`demo_video_v1`
      run with no usable ordered duration became invisible to any
      ``within_tolerance = false`` query — a JSON ``null`` verdict is neither
      pass nor fail, and a gate that cannot fail is the failure class this
      module exists to remove. :func:`duration_conformance` still returns
      ``None`` in this case (there genuinely is no verdict to give — see its
      docstring), but "nothing was ordered" is itself a queryable fact about a
      ``completed`` delivery, so it now gets its own code rather than silence.
    * *result* exists and ``within_tolerance`` is false -> one
      :data:`~src.services.stages.shared.degradation.DURATION_OUT_OF_TOLERANCE`
      entry, unchanged from before.
    * *result* exists and conforms -> no tolerance entry.

    ⛔ The "nobody probed it" fact is NOT here. It is an ADVISORY and lives on
    :func:`duration_conformance_advisories`, whose docstring gives the reason.
    Emitting it here would be a type error as well as a design one:
    :func:`degradation` now REFUSES
    :data:`~src.services.stages.shared.degradation.DURATION_UNMEASURED` because
    it is an advisory code. The two functions are called together at every
    compile stage, onto ``degradations`` and ``advisories`` respectively.

    Never raises: both codes it can emit are module constants registered in
    :data:`~src.services.stages.shared.degradation.DEGRADATION_CODES`, so
    :func:`degradation`'s membership check can never trip here.
    """
    if result is None:
        return [
            degradation(
                DURATION_NOT_ORDERED,
                "no usable ordered duration was found in config_snapshot for "
                "this deliverable — duration conformance was not evaluated",
            )
        ]
    entries: list[Degradation] = []
    if not result["within_tolerance"]:
        entries.append(
            degradation(
                DURATION_OUT_OF_TOLERANCE,
                f"delivered {result['delivered_seconds']:.1f}s against "
                f"{result['ordered_seconds']:.1f}s ordered "
                f"({result['ratio'] * 100.0:.0f}% of order, "
                f"{result['deviation_pct']:+.1f}% deviation) — outside "
                f"+/-{DURATION_TOLERANCE_FRACTION * 100.0:.0f}%",
            )
        )
    return entries


def duration_conformance_advisories(
    result: DurationConformance | None,
) -> list[Degradation]:
    """The non-degrading facts that qualify the verdict, as ADVISORY entries.

    TWO independent entries, either, both or neither:

    * :data:`~src.services.stages.shared.degradation.DURATION_UNMEASURED` — nobody
      probed the delivered file (basis ``declared``). Described below.
    * :data:`~src.services.stages.shared.degradation.DURATION_PLAN_INFEASIBLE` — the
      verdict PASSED the band while the plan behind it had already been refused
      upstream (see :func:`plan_infeasible_reason`). Raised only beside a pass,
      because a failing verdict already says the delivery missed its order.

    They are orthogonal: one says nobody measured the file, the other says the file
    was measured fine but was built from a rejected plan. Both ride ``advisories``
    for the SAME reason and neither ever flips ``degraded``.

    Returns one
    :data:`~src.services.stages.shared.degradation.DURATION_UNMEASURED` entry
    when the verdict was reached on the DECLARED timeline rather than a probe
    (``delivered_basis == DELIVERED_BASIS_DECLARED``), and ``[]`` otherwise —
    including when *result* is ``None``, because with nothing ordered there is
    no verdict whose basis could be in question; that case is already reported
    as ``DURATION_NOT_ORDERED`` by :func:`duration_conformance_degradations`.

    The test is ``!= DELIVERED_BASIS_DECLARED``, i.e. "the measured rung was NOT
    taken", and that keying is what makes this advisory fire on the whole
    unmeasured population now that ``declared`` is the only fallback. It covers
    the case the retired picture-only rung used to swallow: when
    ``post_mux_verify`` crashes, ``mastering.degraded_report()`` nulls BOTH stream
    fields, so those masters land on ``declared`` and are now reported as
    unverified instead of being graded on their truncated picture and passing.

    Returned SEPARATELY from the degradations, not merged into them, and the
    tier is the whole point. This entry is a statement about OUR
    INSTRUMENTATION — we did not measure the file — not about the artifact. An
    unprobed master is not a bad master; it is an unverified verdict. And the
    entry fires on every stage that cannot reach a probe, which includes two
    blueprints that structurally never can: ``demo_video`` and ``showcase``
    route through ``demo_video/encode_master.py``, whose ``EncodeMasterOutput``
    has no ``actual_duration_seconds`` to pass. Carried on ``degradations``,
    this code would therefore flip ``degraded=True`` on 100% of two content
    types, indefinitely, on perfectly conforming deliveries — the
    96.7%-of-masters near-universal-flag failure the advisory tier exists to
    prevent, and which this repo has already been burned by once.

    So it rides ``advisories`` instead: reported, queryable, ignored by
    ``degradation.is_degraded`` and by ``run_repository.degraded_stage_clause``.
    It is IN ADDITION to, never INSTEAD OF, a tolerance degradation — a run can
    be both out of band and unverified, and those are independent facts: one
    says the length is wrong, the other says nobody checked the length that was
    actually encoded. ``DURATION_OUT_OF_TOLERANCE`` keeps flipping ``degraded``;
    this never does.

    Never raises: the code is a module constant registered in
    :data:`~src.services.stages.shared.degradation.ADVISORY_CODES`, so
    :func:`advisory`'s membership check can never trip here.
    """
    if result is None:
        return []
    entries: list[Degradation] = []
    if result["delivered_basis"] == DELIVERED_BASIS_DECLARED:
        entries.append(
            advisory(
                DURATION_UNMEASURED,
                f"the delivered length was never probed — this verdict compared "
                f"{result['ordered_seconds']:.1f}s ordered against the DECLARED "
                f"timeline ({result['delivered_seconds']:.1f}s), not against the "
                "encoded file, so it cannot detect a master whose picture stopped "
                "early; the verdict is unverified, not a pass",
            )
        )
    # INDEPENDENT of the basis above, and only ever raised beside a PASS. Out of
    # band, `DURATION_OUT_OF_TOLERANCE` already says the delivery missed its order
    # and this would add nothing; the defect being closed is specifically a verdict
    # that reads as reassuring, so the entry exists exactly where the reassurance is.
    if result["within_tolerance"] and result.get("plan_infeasible"):
        entries.append(
            advisory(
                DURATION_PLAN_INFEASIBLE,
                f"delivered {result['delivered_seconds']:.1f}s against "
                f"{result['ordered_seconds']:.1f}s ordered "
                f"({result['deviation_pct']:+.1f}%) is inside +/-"
                f"{DURATION_TOLERANCE_FRACTION * 100.0:.0f}%, but the PLAN it was "
                f"produced from was already refused upstream "
                f"({result.get('plan_infeasible_reason') or 'reason unrecorded'}) — "
                "the two verdicts grade different objects against different bands "
                "(this one grades the probed file at +/-10%; the refusal is a bare "
                "sum-of-minimum-demands > order, with no band), so the pass is real "
                "but it is NOT evidence the order was met as planned",
            )
        )
    return entries


#: ``convergence_outcome`` values that mean the timeline was NOT reconciled inside
#: the order. ``"converged"`` is the only clean one; every other listed one leaves a
#: plan that projects past its target —
#: ``time_budget.TimelineManifestRun.convergence_outcome`` documents each.
#: ``"relief_failed"`` is included deliberately: it means no classification was ever
#: computed because the relief loop RAISED, so "the plan is fine" was never
#: established, and a downstream pass must not be read as establishing it.
#: ``"relief_not_run"`` likewise: relief was armed, the loop never ran, and the final
#: timeline projects past the order.
_UNCONVERGED_OUTCOMES: frozenset[str] = frozenset(
    {"cant_converge", "progress", "relief_failed", "relief_not_run"}
)


def plan_infeasible_reason(prior_output: Mapping[str, object] | None) -> str | None:
    """Did an upstream planning stage already REFUSE this run's timeline?

    Reads a completed stage's persisted output dict — in practice
    ``ctx.prior_output("reconcile_timeline")`` — and returns a short reason token
    when that stage said the order could not be met, or ``None`` when it did not.
    The token is threaded into :func:`observe_duration_conformance` as
    ``plan_infeasible``.

    TWO independent upstream signals, joined with ``"+"`` when both fire, because
    they are emitted by different code paths and either can be present alone:

    1. ``timeline_manifest.run.convergence_outcome`` in
       :data:`_UNCONVERGED_OUTCOMES` — the deterministic re-solve classification
       (``time_budget.build_timeline_manifest``).
    2. an ``ordered_duration_infeasible`` entry on ``degradations`` — the prose+code
       verdict ``reconcile_timeline._ordered_duration_degradation`` raises. Note
       ``demo_video_gen`` and ``narration_budget`` raise the SAME code, so a run that
       reaches here carrying it from either of those is caught too.

    ⚠️ DEFENSIVE BY CONSTRUCTION, and that is not decoration. This reads a
    JSON-round-tripped dict whose shape is owned by another stage, on the exit path
    of a render that already succeeded and has already been PAID for. Every access
    is type-guarded and the function cannot raise: a missing key, a ``None``
    manifest (relief never ran), a list where a mapping was expected, or a
    degradations entry that is not a mapping all yield ``None`` — "we did not learn
    of a refusal", which is the same non-alarming answer as "there was none". A
    reporting observer that can crash a completed run is a worse defect than the
    whitewash it was added to fix.
    """
    if not isinstance(prior_output, Mapping):
        return None
    tokens: list[str] = []
    manifest = prior_output.get("timeline_manifest")
    if isinstance(manifest, Mapping):
        run = manifest.get("run")
        if isinstance(run, Mapping):
            outcome = run.get("convergence_outcome")
            if isinstance(outcome, str) and outcome in _UNCONVERGED_OUTCOMES:
                tokens.append(f"convergence_outcome={outcome}")
    degradations = prior_output.get("degradations")
    if isinstance(degradations, list):
        for entry in degradations:
            if isinstance(entry, Mapping) and entry.get("code") == ORDERED_DURATION_INFEASIBLE:
                tokens.append(ORDERED_DURATION_INFEASIBLE)
                break
    return "+".join(tokens) or None


# --------------------------------------------------------------------------- #
# The two SEPARATELY PROBED stream lengths, and the two different questions they
# answer: how long does this play for, and does the picture cover the audio.
# --------------------------------------------------------------------------- #

# A delivered master is PICTURE-TRUNCATED when its audio outruns its video by
# MORE than this many seconds.
#
# 0.5s, and it is not a number invented here: it is the compiler's own
# ``video_compiler_service/src/services/mastering.py::AV_COVERAGE_TOLERANCE_SECONDS``,
# the line above which it raises ``MasteringGateError`` on the same two numbers.
# Mirroring it is the point — two services grading the same pair of measurements
# must not disagree about the same file, and a threshold of our own would make
# "the compiler passed it, we flagged it" a routine, unexplainable event.
#
# Why it is well above the noise floor, measured rather than assumed. The AAC
# encoder pads the tail of every join (+24.0ms per cut, measured 2026-09-11), and
# the live fleet median ``av_gap_seconds`` is 0.097s — ~5x under this line, and
# that median is over a population whose gaps are dominated by exactly that
# padding. The smallest truncation ever observed here is orders of magnitude
# above it (the regression case is 62s).
#
# ⛔ ONE-SIDED, and this is the half that would otherwise wreck the signal.
# ``gap = audio - video``, so only AUDIO OUTRUNNING PICTURE fires. Picture longer
# than audio is NORMAL and authored: live showcase masters run 2.3-5.6s long, and
# a held final frame or an outro title card is a deliberate choice, not a defect.
# A symmetric guard would fire on healthy showcase runs — "a signal that fires on
# everything is not a signal", the 96.7%-of-masters failure this repo has already
# paid for once.
#
# ESTIMATED FALSE-POSITIVE RATE: ~0% of normal runs. Nothing benign in this
# pipeline produces 0.5s of audio over a finished picture — the only live
# population near the line is AAC tail padding at 0.097s median — and the
# compiler additionally hard-fails this exact condition in BOTH gate modes
# (``schemas.py:1060-1067``: ``av_coverage`` is not advisory), so a report
# reaching us above the line is already anomalous. This guard is therefore
# defence in depth against the cases that gate cannot cover: a stored/replayed
# report, a compiler build predating the gate, and any future path that measures
# the streams without enforcing them. It is NOT unfalsifiable — the regression
# shape (video 118s, audio 180s) fires it, and a test pins that.
PICTURE_TRUNCATION_TOLERANCE_SECONDS = 0.5


def mastering_stream_seconds(mastering: object) -> tuple[float, float] | None:
    """``(video_stream_seconds, audio_stream_seconds)`` from a mastering report.

    ``None`` unless BOTH are present, usable (:func:`_usable`, so a bool, a NaN
    or a non-numeric is rejected) and strictly positive — the same ``> 0``
    sentinel rule the rest of this module applies, because the compiler writes
    ``None`` for an unmeasured stream and ``0.0`` is never a real one.

    BOTH, deliberately, not "whichever we got". Each consumer here needs the
    pair: the playable length is a ``max`` over them, and the truncation verdict
    is a difference between them. One stream alone can answer neither question,
    and substituting it would produce a confident wrong answer rather than an
    honest absence — which for the playable length means silently reinstating the
    picture-only number this ladder exists to demote.
    """
    if not isinstance(mastering, Mapping):
        return None
    video = _usable(mastering.get("video_stream_seconds"))
    audio = _usable(mastering.get("audio_stream_seconds"))
    if video is None or audio is None or video <= 0 or audio <= 0:
        return None
    return video, audio


def playable_seconds_from_mastering(mastering: object) -> float | None:
    """The PLAYABLE length — what the customer's player reports — or ``None``.

    ``max(video_stream_seconds, audio_stream_seconds)``. The container bounds its
    own streams, so the longer stream IS the playable length; a master whose
    picture stopped early still plays for as long as its audio runs, and that is
    the number a delivered-vs-ordered verdict has to grade.

    ``None`` whenever :func:`mastering_stream_seconds` cannot supply the pair
    (absent or malformed mastering block, an unmeasured stream), so the caller
    falls to the next rung of :func:`duration_conformance`'s ladder rather than
    being handed a guess.
    """
    pair = mastering_stream_seconds(mastering)
    return None if pair is None else max(pair)


def picture_truncation_degradations(mastering: object) -> list[Degradation]:
    """ "The picture stops before the audio does", as its own structured entry.

    ONE
    :data:`~src.services.stages.shared.degradation.DURATION_PICTURE_TRUNCATED`
    entry when ``audio_stream_seconds - video_stream_seconds`` exceeds
    :data:`PICTURE_TRUNCATION_TOLERANCE_SECONDS`; ``[]`` otherwise, including
    when the streams were not both measured (an unmeasured gap must never be
    reported as a defect — that is the ``DURATION_UNMEASURED`` distinction one
    tier up, applied to a different measurement).

    ⚠️ ONE-SIDED. Only audio outrunning picture fires; picture longer than audio
    is authored and normal here. The threshold constant's comment carries the
    measurement and the false-positive estimate.

    A DEGRADATION rather than an advisory: this says the ARTIFACT is broken, not
    that our instrumentation is thin. See the code's own doc comment.

    SEPARATE from :func:`duration_conformance_degradations` on purpose, and the
    separation is the fix this pair of functions exists to make. The conformance
    verdict grades the PLAYABLE length against the order; this grades the picture
    against the audio. A truncated master is usually BOTH out of tolerance and
    truncated, and those are two facts a reader needs independently — the whole
    regression was one number being asked to carry both.

    It may coincide with ``mastering_gate``'s structural ``av_coverage`` entry,
    which is derived from the compiler's ``rerun_stages`` verdict. That overlap
    is accepted rather than deduped: this entry is computed from the MEASUREMENTS
    and must not become conditional on the gate agreeing, since the reports where
    the gate did NOT flag it are precisely the ones worth catching.

    Never raises: the code is a module constant registered in
    :data:`~src.services.stages.shared.degradation.DEGRADATION_CODES`.
    """
    pair = mastering_stream_seconds(mastering)
    if pair is None:
        return []
    video, audio = pair
    gap = audio - video
    if gap <= PICTURE_TRUNCATION_TOLERANCE_SECONDS:
        return []
    return [
        degradation(
            DURATION_PICTURE_TRUNCATED,
            f"the delivered picture ends {gap:.1f}s before the audio does "
            f"(video_stream_seconds={video:.1f}, audio_stream_seconds={audio:.1f}, "
            f"tolerance {PICTURE_TRUNCATION_TOLERANCE_SECONDS:.1f}s) — narration "
            "plays over a picture that has already stopped",
        )
    ]


# --------------------------------------------------------------------------- #
# Ordered-duration resolvers — one per blueprint, MIRRORING that blueprint's own
# script stage. These are transcriptions of an existing precedence, not a new
# one: if a script stage's default changes, its resolver here must follow, or the
# conformance check starts measuring against a duration nobody ordered.
# --------------------------------------------------------------------------- #

#: ``promo/script.py:267-272`` — the deliverable is where the FE sets the length;
#: a brief-level override wins; 30s is the blueprint's own fallback.
PROMO_DEFAULT_SECONDS = 30.0
#: ``lesson/script.py:173`` — ``float(brief.get("duration_minutes", 6))``.
LESSON_DEFAULT_MINUTES = 6.0
#: ``website_review/script.py:99`` and ``app_review/script.py:64`` —
#: ``int(brief.get("duration_seconds", 120))``.
REVIEW_DEFAULT_SECONDS = 120.0


def _section(config_snapshot: Mapping[str, object], key: str) -> Mapping[str, object]:
    section = config_snapshot.get(key)
    return section if isinstance(section, Mapping) else {}


def promo_ordered_seconds(config_snapshot: Mapping[str, object]) -> float | None:
    """Ordered promo length, in ``promo/script.py``'s precedence.

    ``brief.target_duration_seconds`` → ``deliverable.target_duration_seconds``
    → 30s. Reading only ``brief`` is the bug that stage already fixed once (it
    defaulted every promo to 30s while ignoring the requested length), so the
    deliverable is read here too.
    """
    brief = _section(config_snapshot, "brief")
    deliverable = _section(config_snapshot, "deliverable")
    return (
        _usable(brief.get("target_duration_seconds"))
        or _usable(deliverable.get("target_duration_seconds"))
        or PROMO_DEFAULT_SECONDS
    )


#: ``generic/brief_intake.py`` — the generic blueprint's own fallback length.
GENERIC_DEFAULT_SECONDS = 60.0


def generic_ordered_seconds(config_snapshot: Mapping[str, object]) -> float:
    """Ordered ``generic`` length — THE ladder ``generic/brief_intake`` budgets with.

    ``brief.target_duration_seconds`` → ``deliverable.target_duration_seconds`` →
    60s. Same shape as :func:`promo_ordered_seconds` (the wizard writes the length
    onto the DELIVERABLE). Unlike every resolver above it is not a transcription:
    ``brief_intake`` CALLS this, so the script is budgeted against, and
    ``video_render`` grades the master against, one number. Before 2026-10-20 the
    generic chain read only ``brief`` and never graded the master at all — run
    ``01a156a4-e906`` delivered 12.3s of a 30s order (-59%) with
    ``duration_conformance = null``.
    """
    brief = _section(config_snapshot, "brief")
    deliverable = _section(config_snapshot, "deliverable")
    return (
        _usable(brief.get("target_duration_seconds"))
        or _usable(deliverable.get("target_duration_seconds"))
        or GENERIC_DEFAULT_SECONDS
    )


def lesson_ordered_seconds(config_snapshot: Mapping[str, object]) -> float | None:
    """Ordered lesson length: ``brief.duration_minutes`` (default 6) x 60."""
    minutes = _usable(_section(config_snapshot, "brief").get("duration_minutes"))
    if minutes is None:
        minutes = LESSON_DEFAULT_MINUTES
    return minutes * 60.0


def review_ordered_seconds(config_snapshot: Mapping[str, object]) -> float | None:
    """Ordered review length: ``brief.duration_seconds`` (default 120).

    ``app_review`` only. ``website_review`` resolves through
    ``website_review/order.py`` (deliverable key first — the wizard writes
    ``deliverable.target_duration_seconds``).
    """
    seconds = _usable(_section(config_snapshot, "brief").get("duration_seconds"))
    return REVIEW_DEFAULT_SECONDS if seconds is None else seconds


def demo_video_ordered_seconds(config_snapshot: Mapping[str, object]) -> float | None:
    """Ordered demo_video length, in the target ladder demo_video's OWN plan uses.

    ``demo_video/reconcile_timeline.py:1005-1009`` — ``direction.timing.target_seconds``
    (the explicit override) wins, else ``deliverable.target_duration_seconds``. That is
    the same ladder ``demo_video_gen._target_seconds`` plans the beat count against
    (``gen.py:360-365``, the deliverable rung), so the delivered master is compared
    against the very number the spine was sized to.

    ⛔ There is NO blueprint fallback here, unlike promo/lesson/review. demo_video has
    none: with neither rung present ``reconcile_timeline`` raises ``StageSkippedError``
    ("no target duration to reconcile to") rather than inventing one, and
    ``_target_seconds`` returns ``None``. Substituting a default would manufacture an
    order nobody placed and report a conformance verdict against it — the exact error
    :func:`duration_conformance` refuses by returning ``None`` for an absent order.
    """
    timing = _section(_section(config_snapshot, "direction"), "timing")
    explicit = _usable(timing.get("target_seconds"))
    if explicit is not None:
        return explicit
    return _usable(_section(config_snapshot, "deliverable").get("target_duration_seconds"))


def action_reel_ordered_seconds(ctx: RunContext) -> float | None:
    """Ordered action_reel length — DELEGATED, not transcribed.

    Every other resolver above is a hand-copy of a script stage's precedence,
    because those stages resolve their target inline and there is nothing to
    call. action_reel is the exception: it already factored its ladder out into
    a leaf module of its own, ``action_reel/target_duration.py``
    (``config_snapshot`` → ``deliverable`` → ``brief``, no invented default),
    precisely so the selector and the planner could not aim at different reels.
    Re-spelling it here would add a THIRD reader of that precedence which drifts
    the moment the other two change — so this calls it instead.

    The import runs shared → blueprint, the opposite direction from the rest of
    this module, and it is safe in exactly one case: ``target_duration`` is a
    leaf (stdlib + a ``TYPE_CHECKING``-only ``RunContext``) and
    ``action_reel/__init__`` is a docstring, so there is no edge back. If that
    module ever grows an import of ``shared/``, invert this instead of copying
    the ladder back in.

    Hence also the signature divergence: the siblings take a ``config_snapshot``
    mapping, this takes the ``RunContext`` its delegate takes. Passing
    ``ctx.config_snapshot`` here would mean re-implementing the ladder, which is
    the thing being avoided.

    ⛔ No blueprint fallback, and that is inherited rather than chosen:
    ``target_seconds`` deliberately invents none ("use every selected segment"),
    and a default invented HERE would manufacture an order nobody placed and
    then grade the delivery against it. ``None`` flows on to
    :func:`duration_conformance`, which returns ``None``, which
    :func:`duration_conformance_degradations` reports as
    ``DURATION_NOT_ORDERED`` — an honest "unmeasured", never "on target".
    """
    return target_seconds(ctx)
