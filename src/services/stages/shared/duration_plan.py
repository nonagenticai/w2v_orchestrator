"""Plan-time arithmetic that keeps an ORDERED duration and an ACHIEVABLE one consistent.

A demo_video run decides its duration THREE times, in three places, and until this
module existed the three could disagree without anybody noticing until the end:

1. **Generation** — how long a recorded scenario actually runs (bdd_playwright).
2. **Planning** — how many scenarios/beats to ask for (``demo_video_gen``, and the
   relief lever's ``target_beats``).
3. **Reconcile** — the hard 50% underfull floor in ``demo_video_reconcile_timeline``,
   which refuses to ship a film below ``demo_timeline_underfull_fail_ratio`` of the
   ordered target.

(2) used to be a fiction: ``gen._SECONDS_PER_SCENARIO = 4.0``, extrapolated from two
hand-observed samples on one 2026-08 E2E run, matching NEITHER mode the pipeline has
since been measured in (1.2-3.9s unpaced, 16.7-36.9s paced). A plan built on it could
not be checked against (3) because it was not a measurement of anything, so the
mismatch was only ever discovered by (3) — AFTER record and render had both been paid
for. Billing in this fleet is append-only with no refund path, so every one of those
seconds is permanently unrecoverable.

**The rule this module encodes: a plan is only legal if its own arithmetic can reach
the floor that will judge it.** ``beats x measured_seconds_per_beat >= ratio x target``.
Where that fails, RE-SIZE the plan to the smallest count that satisfies it (a refused
run delivers nothing); refuse only when no count within the available spine can.

**One owner, and it is a MEASUREMENT.** ``measured_seconds_per_beat`` reads the
ffprobe'd truth the pipeline already carries — the footage-pinned rows of a
``timeline_manifest``, where ``time_budget`` sets ``min == ideal == max == measured``.
It returns ``None`` when nothing was measured, and every function here treats ``None``
as UNMEASURED (make no judgement) rather than substituting a default. Replacing one
invented constant with another invented constant is not a fix.

Pure: no settings, no I/O, deterministic. Both the planner (``demo_video_gen``) and the
relief spawner (``structural_relief_policy``) read these same functions, so the forecast
and the action cannot diverge by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from src.services.stages.shared.duration_conformance import DURATION_TOLERANCE_FRACTION

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Key under which the parent's measured per-beat yield rides on a relief child's
#: ``config_snapshot["structural_relief"]``. One name, written by the spawner and read
#: by ``demo_video_gen`` — never re-derived at the consumer.
MEASURED_SECONDS_PER_BEAT_KEY = "measured_seconds_per_beat"

PlanVerdict = Literal["OK", "UNMEASURED", "RESIZED", "RESIZED_DOWN", "UNSATISFIABLE"]


def _as_float(value: object) -> float | None:
    """``value`` as a FINITE float, or ``None`` — ``bool`` is NOT a number here.

    ``NaN``/``±inf`` are rejected to ``None`` (UNMEASURED), not passed through. They are
    input-reachable: ``json.loads("1e400")`` is ``inf``, and an unvalidated
    ``config_overrides`` blob round-trips through jsonb into this arithmetic. Every
    comparison against ``NaN`` is False, so a bare ``<= 0.0`` guard does not stop one —
    it sails through to ``math.ceil`` and raises ``ValueError``/``OverflowError`` in the
    middle of a billing-relevant decision. ``None`` here means exactly what it has always
    meant in this module: nothing was measured, so make no judgement.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        num = float(value)
        return num if math.isfinite(num) else None
    return None


def measured_seconds_per_beat(beats: Sequence[object]) -> float | None:
    """Mean MEASURED footage seconds per beat, from a ``timeline_manifest``'s rows.

    A footage-pinned beat is one whose length is a measured truth the solver may not
    move: ``shared.time_budget`` builds its demand with ``min == ideal == max ==
    measured``, so the manifest row satisfies ``min_seconds == max_seconds > 0``. That
    equality IS the discriminator between "this is what the recorder actually produced"
    and "this is what the solver computed for a VO/card beat", and it is the same
    signal ``reconcile_timeline._yield_attribution`` reads off the allotments.

    Returns ``None`` when no row qualifies — an old/thin manifest, or a pure text
    timeline with no footage. ``None`` means UNMEASURED, and every caller here must
    make no judgement on it rather than fall back to an assumed number. That is the
    whole point: the assumed number is what this module exists to delete.
    """
    pinned: list[float] = []
    for raw in beats:
        if not isinstance(raw, dict):
            continue
        beat = cast("Mapping[str, Any]", raw)
        low = _as_float(beat.get("min_seconds"))
        high = _as_float(beat.get("max_seconds"))
        if low is None or high is None or low <= 0.0:
            continue
        if math.isclose(low, high, rel_tol=1e-9, abs_tol=1e-9):
            pinned.append(low)
    if not pinned:
        return None
    return sum(pinned) / len(pinned)


def exported_seconds_per_beat(values: Sequence[float | None]) -> float | None:
    """The per-scenario recording duration bdd_playwright EXPORTS, or ``None``.

    ``narration_script.parts[].expected_recording_seconds`` (bdd #153) — the guaranteed
    clip floor, owned by ``HERO_DWELL_MS`` in bdd's ``recording_helpers.ts`` and read off
    the wire. This is rung ONE of the yield ladder and outranks a parent run's measured
    footage, because bdd now *guarantees* the pacing rather than merely having produced
    it once: it is the number the next recording will actually honour.

    ⛔ There is no constant here and there must never be one. Copying bdd's value into
    this repo would desync the planner the moment that dwell changes — which is the exact
    drift this whole change removes. Absent values mean UNMEASURED (``None``), and the
    caller makes no judgement; a non-demo_video stream legitimately publishes nothing.

    Mean over the parts that carry a positive FINITE value, so a partially-populated
    script still yields the figure its recording beats agree on. Non-finite rows are
    dropped BEFORE the average rather than averaged in: one ``inf`` row makes the mean
    ``inf``, which is not a yield anybody measured.
    """
    usable = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
    ]
    if not usable:
        return None
    return sum(usable) / len(usable)


def beats_for_seconds(seconds: float, per_beat_seconds: float) -> int:
    """How many beats of ``per_beat_seconds`` each are needed to cover ``seconds``.

    The count-sizing arithmetic, in one place: ``ceil(seconds / per_beat)``, floored at
    1. A non-positive ``per_beat_seconds`` is not a yield and cannot size anything, so
    it returns 0 — the caller must read that as "cannot size", never as "zero beats".

    A NON-FINITE input is the same answer for the same reason, and must be tested
    SEPARATELY: ``nan <= 0.0`` is False, so the ``<= 0.0`` guard above does not stop it
    and ``math.ceil(nan)`` raises ``ValueError`` (``inf`` raises ``OverflowError``).
    """
    if (
        not math.isfinite(per_beat_seconds)
        or not math.isfinite(seconds)
        or per_beat_seconds <= 0.0
        or seconds <= 0.0
    ):
        return 0
    return max(1, math.ceil(seconds / per_beat_seconds))


def band_seconds(target_seconds: float, tolerance_fraction: float) -> tuple[float, float]:
    """The ``[low, high]`` DELIVERED-seconds band a film of this order is judged by.

    The conformance band, in one place, so a guard that refuses a plan and the
    measurement that later judges the film cannot draw two different lines. Callers pass
    ``duration_conformance.DURATION_TOLERANCE_FRACTION``; this module never names a
    tolerance of its own (a copied constant is the drift this whole module deletes).

    ``(0.0, 0.0)`` when there is no ordered duration — "no band", which every caller must
    read as "make no judgement", never as "an empty band that nothing can satisfy".

    A NON-FINITE ``target_seconds`` is "no ordered duration" for the same reason: a band
    of ``[nan, nan]`` contains nothing and excludes nothing, so every later comparison
    silently answers False. It gets the same "no band" answer, tested separately because
    ``nan <= 0.0`` is False.
    """
    if (
        not math.isfinite(target_seconds)
        or not math.isfinite(tolerance_fraction)
        or target_seconds <= 0.0
    ):
        return (0.0, 0.0)
    tolerance = max(0.0, tolerance_fraction)
    return (target_seconds * (1.0 - tolerance), target_seconds * (1.0 + tolerance))


def beat_counts_within_band(
    *,
    per_beat_seconds: float | None,
    max_beats: int,
    band: tuple[float, float],
) -> list[int]:
    """Every count in ``1..max_beats`` whose achievable seconds land INSIDE ``band``.

    ENUMERATED, never inferred. :func:`beats_for_seconds` answers "how many beats reach
    at least X seconds" and rounds UP, so the count it returns can itself sail past the
    top of the band — which is precisely how a relief child came to be sized at 2 beats x
    22.00s = 44.00s against a 30.00s order. A bound that is computed by rounding cannot
    be checked by the same rounding; it has to be checked against the actual products.

    The list is the evidence a refusal is allowed to cite. A caller that wants to say
    "nothing fits" must get an EMPTY list back from this function first, and may then
    name only the counts it enumerated — this repo has twice shipped a degradation
    asserting an impossibility that was false.

    Empty when there is no yield, no band, or no legal count (``max_beats < 1``).
    """
    low, high = band
    if (
        per_beat_seconds is None
        or not math.isfinite(per_beat_seconds)
        or not math.isfinite(low)
        or not math.isfinite(high)
        or per_beat_seconds <= 0.0
        or high <= 0.0
        or max_beats < 1
    ):
        return []
    return [k for k in range(1, max_beats + 1) if low <= k * per_beat_seconds <= high]


def floor_seconds(target_seconds: float, floor_ratio: float) -> float:
    """The seconds ``reconcile_timeline``'s underfull gate will require of this film.

    Not a second formula for the gate — the gate's own expression, so the predictor and
    the gate cannot disagree (the defect that shipped live run 01a059a2: a note that
    correctly predicted a failure and a gate that then passed the run anyway).

    ``0.0`` — "no floor to require", i.e. no judgement — also covers a NON-FINITE target
    or ratio. Returning ``nan`` here would propagate into ``plan_beats``' ``required``,
    where ``required <= 0.0`` is False for ``nan`` and the UNMEASURED early-out is
    skipped.
    """
    if (
        not math.isfinite(target_seconds)
        or not math.isfinite(floor_ratio)
        or target_seconds <= 0.0
        or floor_ratio <= 0.0
    ):
        return 0.0
    return floor_ratio * target_seconds


@dataclass(frozen=True)
class DurationPlan:
    """A beat count, judged against the floor that will eventually judge the film.

    ``verdict``:

    * ``UNMEASURED`` — no yield measurement and/or no target/ratio, so no judgement was
      possible. ``beats`` is ``requested_beats``, untouched. Never a refusal: an absent
      measurement is not evidence of a bad plan. A NON-FINITE (``nan``/``inf``) yield or
      target lands here too, and deliberately NOT on ``UNSATISFIABLE``: an unreadable
      number is "I could not measure", which is not "it cannot be done". This repo has
      shipped a false impossibility claim before; a corrupt input must never manufacture
      another one.
    * ``OK`` — the requested count already clears the floor arithmetically.
    * ``RESIZED`` — it did not, but a larger count within ``max_beats`` does; ``beats``
      is that count. PREFERRED over refusing: a refused run delivers nothing.
    * ``RESIZED_DOWN`` — the requested count clears the floor but OVERSHOOTS the ordered
      duration; ``beats`` is the largest count that still covers the order. The floor is
      only half the contract — a plan may be too LONG, and until this verdict existed
      nothing said so. See :func:`plan_beats`.
    * ``UNSATISFIABLE`` — even ``max_beats`` beats at the measured yield cannot reach
      the floor. No valid sizing exists, so the work must not be started/billed.
    """

    verdict: PlanVerdict
    beats: int
    requested_beats: int
    min_beats_for_floor: int
    max_beats_for_target: int
    max_beats: int
    per_beat_seconds: float | None
    required_seconds: float
    achievable_seconds: float

    @property
    def satisfiable(self) -> bool:
        """Did a legal sizing exist? (``UNSATISFIABLE`` is the only ``False``.)"""
        return self.verdict != "UNSATISFIABLE"

    def explain(self) -> str:
        """One line naming both numbers, for a log line, a note, or a raised message."""
        if self.verdict == "UNMEASURED":
            return (
                "duration plan UNMEASURED — no footage-pinned beat carries a measured "
                "length, so the planned count cannot be reconciled against the "
                "underfull floor for this run"
            )
        per_beat = self.per_beat_seconds or 0.0
        head = (
            f"{self.requested_beats} beat(s) x {per_beat:.2f}s measured/beat = "
            f"{self.requested_beats * per_beat:.2f}s against the "
            f"{self.required_seconds:.2f}s the underfull floor requires"
        )
        if self.verdict == "OK":
            return f"duration plan OK — {head}"
        if self.verdict == "RESIZED":
            return (
                f"duration plan RESIZED — {head}; re-sized to {self.beats} beat(s) "
                f"({self.achievable_seconds:.2f}s) so the plan can reach the floor"
            )
        if self.verdict == "RESIZED_DOWN":
            return (
                f"duration plan RESIZED_DOWN — {head}, which OVERSHOOTS the ordered "
                f"duration; re-sized to {self.beats} beat(s) "
                f"({self.achievable_seconds:.2f}s), the largest count that still "
                f"covers the order"
            )
        return (
            f"duration plan UNSATISFIABLE — {head}, and even the largest available "
            f"spine ({self.max_beats} beat(s) = {self.achievable_seconds:.2f}s) cannot "
            f"reach it; no valid sizing exists"
        )


def plan_beats(
    requested_beats: int,
    *,
    target_seconds: float,
    floor_ratio: float,
    per_beat_seconds: float | None,
    max_beats: int,
    tolerance_fraction: float | None = DURATION_TOLERANCE_FRACTION,
) -> DurationPlan:
    """Reconcile a PLANNED beat count against the ACHIEVABLE duration it implies.

    This is the early-decision point the pipeline never had. ``demo_video_record``
    already computed exactly this prediction ("capture yielded 2.56s; the 30.00s order
    needs >= 15.00s") and then discarded it as note-only, by explicit design — so the
    mismatch was known one stage after recording and acted on zero stages later, i.e.
    never, until ``reconcile_timeline`` raised past a completed render.

    Prefers RE-SIZING to refusing wherever the arithmetic allows one (see
    :class:`DurationPlan`). ``UNSATISFIABLE`` is reserved for the case where no count
    up to ``max_beats`` works — there the honest answer is that the ordered duration is
    not reachable from this material, and paying for the render cannot change that.

    ⭐ ``tolerance_fraction`` — WHY ``RESIZED_DOWN`` COULD NOT BIND ON ITS OWN CITED CASE
    (wave 13, O4). :class:`DurationPlan` documents ``RESIZED_DOWN`` as the verdict for a
    plan that OVERSHOOTS the ordered duration, and the overshoot it was written for is
    ``target 30.00s, 22.00s measured/beat, 2 beats = 44.00s`` (**+46.7%**). Trace it:
    ``needed = ceil(15.00/22.00) = 1``, ``ceiling = ceil(30.00/22.00) = 2``, and the
    trigger is ``requested_beats > max(ceiling, needed)`` = ``2 > 2`` = **False**. So the
    verdict is ``OK`` and the +46.7% overshoot is certified as a plan that fits.

    The cause is named in :func:`beat_counts_within_band`'s own docstring — *"a bound
    computed by rounding cannot be checked by the same rounding"*. ``ceiling`` is
    ``beats_for_seconds``, which rounds UP, so the ceiling can itself sail past the top
    of the band; comparing a requested count against it can only ever catch a plan that
    overshoots the ROUNDED-UP bound, never one that overshoots the ORDER. The helper was
    written precisely to fix this and ``plan_beats`` never called it.

    So when a ``tolerance_fraction`` is supplied the ceiling is checked against the
    ENUMERATED products (``k * per_beat_seconds`` for every legal ``k``) instead of
    against a rounded quotient, and ``beats`` becomes the largest count whose product
    actually lands inside the band — and, where the band is EMPTY, the largest count
    that at least does not overshoot it (see the ``elif`` below for why missing SHORT is
    the correct direction to miss). Measured against a 30.00s order:

    ==========  =========  ======================  ==========================
    per_beat    requested  before                  after
    ==========  =========  ======================  ==========================
    22.00s      2          ``OK`` 44.00s +46.7%    ``RESIZED_DOWN`` 22.00s -26.7%
    14.00s      3          ``OK`` 42.00s +40.0%    ``RESIZED_DOWN`` 28.00s  -6.7%
    14.00s      2          ``OK`` 28.00s  -6.7%    ``OK`` 28.00s  -6.7% (unchanged)
    10.00s      5          ``RESIZED_DOWN`` 30.00s ``RESIZED_DOWN`` 30.00s (unchanged)
    40.00s      1          ``OK`` 40.00s +33.3%    ``OK`` 40.00s +33.3% (no shorter count)
    ==========  =========  ======================  ==========================

    The first row IS the docstring's cited case, and it is the proof the bound can now
    bind: it was the verdict's own example and the verdict could not reach it.

    ⭐ WAVE 14, N4 — IT NOW DEFAULTS ON, AND THAT IS THE FIX. This parameter shipped
    defaulting to ``None``, i.e. the band check was OPT-IN, and exactly ONE of the four
    call sites opted in (``demo_video/gen.py`` ``_assert_plan_can_reach_floor``). The
    other three — ``gen.py``'s relief lever ``_apply_target_beats``, ``gen.py``'s
    ``_apply_duration_ceiling``, and ``structural_relief_policy.decide_relief_shape``
    — omitted it, so ``in_band`` came back ``[]`` and the rounded-UP quotient stayed in
    charge on every one of them. The +46.7% overshoot class this paragraph exists to
    kill was therefore still live on the PAID relief path.

    A guard whose default disables it is not a guard. ``None`` remains ACCEPTED (a
    caller that has already enumerated the band can still switch it off explicitly),
    but omitting the argument can no longer silently disable the bound.

    ⛔ AND THIS TOLERANCE IS NOT ``time_budget.DEFAULT_TOLERANCE_PCT`` (2.0). That one is
    the UNDERFULL acceptance band on the solver's own allotment sum; this one grades the
    ordered-vs-achievable DURATION and is ``duration_conformance.DURATION_TOLERANCE_FRACTION``
    (0.10), the band the PROBED FILE is judged by. They measure different objects and must
    never be merged — cross-referenced here only so the next reader does not try.
    """
    required = floor_seconds(target_seconds, floor_ratio)
    if (
        per_beat_seconds is None
        or not math.isfinite(per_beat_seconds)
        or per_beat_seconds <= 0.0
        or required <= 0.0
    ):
        return DurationPlan(
            verdict="UNMEASURED",
            beats=requested_beats,
            requested_beats=requested_beats,
            min_beats_for_floor=0,
            max_beats_for_target=0,
            max_beats=max_beats,
            per_beat_seconds=per_beat_seconds,
            required_seconds=required,
            achievable_seconds=0.0,
        )
    needed = beats_for_seconds(required, per_beat_seconds)
    # The CEILING, the mirror of ``needed``: the smallest count that still covers the
    # ORDERED duration (not the floor). Same ``beats_for_seconds`` arithmetic, so the
    # two bounds cannot drift apart. Anything above this is plan the order did not ask
    # for and nobody downstream will trim — reconcile_timeline diagnoses the overshoot
    # and completes anyway.
    ceiling = beats_for_seconds(target_seconds, per_beat_seconds)
    # THE BOUND, CHECKED AGAINST THE PRODUCTS RATHER THAN AGAINST ITSELF. See the
    # ``tolerance_fraction`` paragraph above: `ceiling` is a rounded-UP quotient, so it
    # cannot detect an overshoot of the ORDER — only of itself. `in_band` is the
    # enumeration that can. Empty (no yield, no band, or genuinely nothing fits) leaves
    # `ceiling` in charge exactly as before, which is the conservative direction: this
    # may only ever TIGHTEN the bound, never loosen one that already binds.
    in_band = (
        beat_counts_within_band(
            per_beat_seconds=per_beat_seconds,
            max_beats=max_beats,
            band=band_seconds(target_seconds, tolerance_fraction),
        )
        if tolerance_fraction is not None
        else []
    )
    if in_band:
        ceiling = min(ceiling, max(in_band)) if ceiling > 0 else max(in_band)
    elif tolerance_fraction is not None:
        # ⭐ THE EMPTY-BAND CASE, WHICH IS THE DOCSTRING'S OWN HEADLINE CASE. With
        # ``target 30.00s`` and ``22.00s/beat`` NOTHING conforms: 1 beat is 22.00s
        # (-26.7%), 2 beats is 44.00s (+46.7%). `beat_counts_within_band` correctly
        # returns ``[]``, and an empty enumeration left `ceiling` = 2 in charge — i.e.
        # the rounded-up quotient, which is the LONG side. That is how a +46.7%
        # overshoot was certified ``OK``.
        #
        # When nothing conforms, the choice is WHICH WAY TO MISS, and wave 13 settles
        # that: the ordered duration is a target with a real under-delivery allowance,
        # and a film that comes in short is preferred to one that overruns — the
        # compiler cannot cut audio, and nothing downstream trims an overshoot
        # (reconcile_timeline diagnoses it and completes anyway). So take the largest
        # count that does not exceed the band's TOP, and only fall back to the rounded
        # ceiling when even one beat overshoots it (there, missing long is the only
        # option available and the pre-existing answer is already the right one).
        _, high = band_seconds(target_seconds, tolerance_fraction)
        under = [k for k in range(1, max_beats + 1) if k * per_beat_seconds <= high]
        if under:
            ceiling = min(ceiling, max(under)) if ceiling > 0 else max(under)
    if requested_beats >= needed:
        if ceiling > 0 and requested_beats > max(ceiling, needed):
            resized = max(ceiling, needed)
            return DurationPlan(
                verdict="RESIZED_DOWN",
                beats=resized,
                requested_beats=requested_beats,
                min_beats_for_floor=needed,
                max_beats_for_target=ceiling,
                max_beats=max_beats,
                per_beat_seconds=per_beat_seconds,
                required_seconds=required,
                achievable_seconds=resized * per_beat_seconds,
            )
        return DurationPlan(
            verdict="OK",
            beats=requested_beats,
            requested_beats=requested_beats,
            min_beats_for_floor=needed,
            max_beats_for_target=ceiling,
            max_beats=max_beats,
            per_beat_seconds=per_beat_seconds,
            required_seconds=required,
            achievable_seconds=requested_beats * per_beat_seconds,
        )
    if needed <= max_beats:
        return DurationPlan(
            verdict="RESIZED",
            beats=needed,
            requested_beats=requested_beats,
            min_beats_for_floor=needed,
            max_beats_for_target=ceiling,
            max_beats=max_beats,
            per_beat_seconds=per_beat_seconds,
            required_seconds=required,
            achievable_seconds=needed * per_beat_seconds,
        )
    return DurationPlan(
        verdict="UNSATISFIABLE",
        beats=max_beats,
        requested_beats=requested_beats,
        min_beats_for_floor=needed,
        max_beats_for_target=ceiling,
        max_beats=max_beats,
        per_beat_seconds=per_beat_seconds,
        required_seconds=required,
        achievable_seconds=max(0, max_beats) * per_beat_seconds,
    )
