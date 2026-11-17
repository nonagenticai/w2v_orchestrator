"""Structural timing-relief auto-convergence policy (demo_video, PR-C).

After a ``demo_video`` run completes, its footage-pinned timeline may still be
over budget: the eligible in-place trim/reallocate levers were exhausted yet the
picture is longer than the brief allows. This module decides — purely, from
orch #671's typed convergence classification — whether to spawn a *bounded child
regenerate run* (fewer beats) to converge that over-budget timeline, or to stop.

A gen-rerun is a **paid** render, so spawning is deliberately narrow:

* Only the footage-pinned ``"cant_converge"`` outcome is a candidate. A
  ``"progress"`` outcome means eligible beats still exist and the in-place trim
  loop is the correct (free) lever — never a paid gen-rerun. ``"converged"``
  is already feasible.
* Spawning is bounded by ``max_passes`` (≤3 paid renders total across the chain).
* A pass that did not measurably reduce the overrun (flat or worsening within a
  0.5s epsilon) is treated as non-converging: the last structural cut bought no
  real progress, so we stop rather than burn another render.

The function is side-effect free and dependency-light: it returns one of four
:data:`StructuralReliefDecision` verdicts and the caller performs the spawn.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from src.services.stages.shared.duration_conformance import (
    DURATION_TOLERANCE_FRACTION,
)
from src.services.stages.shared.duration_conformance import (
    # ``_usable`` is "finite float, or None" — one coercion for an ordered duration,
    # shared with the stage boundary rather than restated here.
    _usable as finite_or_none,
)
from src.services.stages.shared.duration_plan import (
    DurationPlan,
    band_seconds,
    beat_counts_within_band,
    beats_for_seconds,
    measured_seconds_per_beat,
    plan_beats,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

log = logging.getLogger(__name__)

StructuralReliefDecision = Literal[
    "SPAWN", "STOP_CONVERGED", "STOP_CANT_CONVERGE", "STOP_CAP", "STOP_RELIEF_OFF"
]

#: ``timeline_manifest.run.convergence_outcome`` for a manifest built ONLY to carry footage
#: out-points (``reconcile_timeline._stamp_footage_out_points``) on a run whose relief is NOT
#: armed (``direction.timing.relief.mode != "auto"``). A relief-ON run whose loop had nothing
#: to do is classified instead (``converged`` when the trimmed timeline fits).
RELIEF_OFF_OUTCOME: Final = "relief_off"

#: ``timeline_manifest.run.convergence_outcome`` for the out-points-only manifest of a
#: relief-ON (``"auto"``) run whose relief loop never RAN (``loop_iterations == 0``) and
#: whose FINAL timeline is NOT in band. The loop only arms on an infeasible or unsayable
#: timeline; the underspoken re-ask runs AFTER it and may lengthen a fitting timeline past
#: the order. Classifying that final state would read ``cant_converge`` and buy a PAID
#: child for an overrun the free re-ask caused, on a run no relief pass ever examined.
#: Mapped EXPLICITLY to ``STOP_CANT_CONVERGE`` in :func:`decide_structural_relief`.
RELIEF_NOT_RUN_OUTCOME: Final = "relief_not_run"


def relief_mode_of_timing(timing: Mapping[str, Any] | None) -> str:
    """``direction.timing.relief.mode`` — absent / malformed ⇒ ``"off"``.

    The ONE reader of the relief opt-in: ``reconcile_timeline`` arms its relief loop on
    ``== "auto"`` from this, and the spawner refuses a paid child unless the SAME read
    says ``"auto"``. A ``direction.timing`` that is absent or ``mode: "off"`` skips the
    reconcile stage entirely, so it reads ``"off"`` here too.
    """
    if not isinstance(timing, dict) or not timing or timing.get("mode") == "off":
        return "off"
    relief = timing.get("relief") or {}
    if not isinstance(relief, dict):
        return "off"
    return str(relief.get("mode", "off"))


def relief_mode_from_snapshot(config_snapshot: Mapping[str, Any]) -> str:
    """:func:`relief_mode_of_timing` read off a run's ``config_snapshot``."""
    direction = config_snapshot.get("direction") or {}
    if not isinstance(direction, dict):
        return "off"
    return relief_mode_of_timing(direction.get("timing"))


# Bound on paid gen-rerun passes: pass 0 (original) + up to 2 children = ≤3 renders.
_DEFAULT_MAX_PASSES = 2

# How much the overrun must shrink for a pass to count as real progress (seconds).
_PROGRESS_EPSILON_SECONDS = 0.5


def decide_structural_relief(
    *,
    convergence_outcome: str,
    structural_relief_requested: bool,
    overrun_seconds: float,
    prior_overrun_seconds: float | None,
    pass_number: int,
    max_passes: int = _DEFAULT_MAX_PASSES,
) -> StructuralReliefDecision:
    """Decide whether to spawn a bounded gen-rerun child to converge the timeline.

    Args:
        convergence_outcome: Orch #671 classification — ``"converged"``,
            ``"progress"``, or ``"cant_converge"``.
        structural_relief_requested: Retained for telemetry/compat only — NO
            longer gates the spawn (see the note in the body). It is
            ``bool(rerun_stages)``, a predicate too narrow for footage-dominated
            mixed spines; the spawn now trusts ``convergence_outcome`` alone.
        overrun_seconds: Current timeline overrun (seconds over budget).
        prior_overrun_seconds: The overrun of the parent pass, or ``None`` on the
            first pass (no prior render to compare against).
        pass_number: Zero-based index of the current pass in the rerun chain.
        max_passes: Maximum number of paid child passes permitted.

    Returns:
        One of the :data:`StructuralReliefDecision` verdicts.
    """
    # 0. The relief loop never RAN — the manifest only carries footage out-points. A
    #    relief-off run must never buy a paid render, whatever its overrun says.
    if convergence_outcome == RELIEF_OFF_OUTCOME:
        return "STOP_RELIEF_OFF"
    #    Relief was armed but the loop never ran, and the final timeline is out of band
    #    (the post-loop underspoken re-ask moved it). No relief pass classified anything,
    #    so nothing here can justify a paid render. Explicit, not left to the
    #    ``!= "cant_converge"`` fall-through below, so a future reorder cannot open it.
    if convergence_outcome == RELIEF_NOT_RUN_OUTCOME:
        return "STOP_CANT_CONVERGE"

    # 1. Feasible already — accept the timeline and stop cleanly.
    if convergence_outcome == "converged":
        return "STOP_CONVERGED"

    # 2. Only the footage-pinned ``cant_converge`` case is a gen-rerun candidate.
    #    ``progress`` means eligible beats exist and the IN-PLACE trim loop is the
    #    lever (free), NOT a paid gen-rerun.
    #
    #    We trust ``convergence_outcome`` ALONE here — deliberately NOT AND-ing it
    #    with ``structural_relief_requested``. That flag is ``bool(rerun_stages)``,
    #    which is only set when a *flagged* beat is ineligible; but ``flagged``
    #    excludes pinned beats, so a footage-dominated MIXED spine (footage floor >
    #    target, plus a few elastic VO beats) leaves ``structural_relief_requested``
    #    False even though ``_classify_convergence`` correctly reports
    #    ``cant_converge`` from the protected-floor test. AND-ing the two therefore
    #    STARVED exactly the case orch #688 rewrote the classifier to catch. The
    #    real arming gate is ``settings.structural_relief_autoconverge_enabled``
    #    (checked at the caller); ``structural_relief_requested`` is retained on the
    #    manifest as telemetry, not a spawn gate.
    if convergence_outcome != "cant_converge":
        return "STOP_CANT_CONVERGE"

    # 3. Bound reached — never exceed the paid-render cap (≤3 renders total).
    if pass_number >= max_passes:
        return "STOP_CAP"

    # 4. Flat/worsening overrun within the epsilon — the last structural cut made
    #    no real progress, so another paid render would just waste money.
    if (
        prior_overrun_seconds is not None
        and overrun_seconds >= prior_overrun_seconds - _PROGRESS_EPSILON_SECONDS
    ):
        return "STOP_CANT_CONVERGE"

    # 5. Footage-pinned, requested, under cap, and still making progress — spawn.
    return "SPAWN"


# ── Shape feasibility: can a relief child actually SHRINK, and can it converge? ──────────
#
# ``decide_structural_relief`` above is only the FIRST of the gates a spawn passes. The
# next two are arithmetic on the timeline manifest — how many beats the child should
# target, and whether ANY subset of that size can fit the budget. Those two lived inline
# in :mod:`src.services.runs.structural_relief` (the spawner) while
# ``reconcile_timeline`` independently forecast the spawn from ``decide_structural_relief``
# alone. The two therefore disagreed exactly where it matters: on an unconvergeable spine
# the spawner refused to bill a render while the stage note told the user "no action
# needed". One decision must have ONE home — it lives here, in the pure module both
# callers already import.

ReliefShapeVerdict = Literal[
    "SPAWN",
    "STOP_CANNOT_SHRINK",
    "STOP_UNCONVERGEABLE",
    "STOP_UNDERFULL",
    "STOP_ALREADY_CONFORMING",
]

#: Which measure SIZED the child (see :func:`_sizing_basis`). ``"footage"`` is the
#: manifest's ffprobe'd footage plan (``target + overrun``); ``"delivered"`` is the
#: parent's measured ``duration_conformance.delivered_seconds``, used ONLY when it lies
#: above the band AND at or below the footage plan. ``"footage_delivered_exceeds_plan"``
#: is the footage basis again, chosen BECAUSE the measured delivery exceeded the footage
#: plan — sized exactly as ``"footage"``, recorded separately so an auditor can tell "no
#: delivery to size from" apart from "a delivery we refused to size from".
#: ``"footage_basis_refused"`` is the footage basis too: a delivery WAS measured and was
#: eligible to size from, but the footage basis had already refused, and a refusal is
#: never re-opened by the delivery (see :func:`decide_relief_shape`).
ReliefSizingBasis = Literal[
    "footage", "delivered", "footage_delivered_exceeds_plan", "footage_basis_refused"
]

#: Each basis as a clause a customer can read, for notes that name it
#: (``encode_master``'s refusal degradation). Never print the identifier itself.
SIZING_BASIS_PHRASES: dict[ReliefSizingBasis, str] = {
    "footage": "the re-render was sized from this run's footage",
    "delivered": "the re-render was sized on the delivered basis, from the film as delivered",
    "footage_delivered_exceeds_plan": (
        "the re-render was sized from this run's footage, because the delivered film ran "
        "longer than the footage plan"
    ),
    "footage_basis_refused": (
        "the re-render was sized from this run's footage, which ruled it out before the "
        "delivered film was considered"
    ),
}

# ── WHY THERE IS NO "is a paid child BETTER than the parent?" COMPARISON HERE. ────────
#
# The obvious missing guard is the symmetric one: when NO legal beat count lands inside
# the band, compare the closest candidate against what the parent ACTUALLY delivered and
# buy the child when it is enough of an improvement. It was written, measured, and
# REMOVED — on this pipeline's real manifests it cannot do the thing it was added for.
#
# THE ARITHMETIC, verified independently (2026-09-16) on 12 of 12 measured production
# demo_video runs, all of which are ALL-PINNED spines (every beat footage-measured, so
# ``min_seconds == max_seconds == per_beat``):
#
#   * :func:`protected_seconds_floor` sums the ``k`` SMALLEST ``min_seconds``. On an
#     all-pinned spine every ``min_seconds`` is the same number, so the floor is EXACTLY
#     ``k x per_beat`` — identical to the candidate duration itself.
#   * The PRE-EXISTING unconvergeable guard at the bottom of :func:`decide_relief_shape`
#     (untouched by this change) refuses any shape whose ``seconds_floor >
#     target_seconds``.
#   * Therefore the reachable candidate set is exactly ``{k : k x per_beat <= T}`` — no
#     count ABOVE the order can ever be spawned, whatever a comparison decides.
#
# So a status-quo comparison can only ever choose an UNDERSHOOT, and on the four real
# out-of-band parents it would have landed at -15.29 / -23.02 / -15.67 / -10.76 %: three
# of them outside the band on the SHORT side. That is the unrecoverable direction —
# there is no extend lever anywhere downstream of here, whereas an overshoot still has
# the in-place trim/reallocate levers in front of it. A guard that is nearly inert and
# whose one firing ships a short film is not an improvement over refusing.
#
# ⛔ Do not re-add it without first refuting the floor arithmetic above. The status quo
# IS recorded (:attr:`ReliefShape.status_quo_seconds`) so a future session can re-measure
# the population rather than re-derive this from scratch.

#: The verdict the OUT-OF-BAND refusal (see :func:`decide_relief_shape`) reports.
#:
#: ⚠️ Deliberately ``STOP_UNDERFULL`` and NOT a new literal, for one concrete reason:
#: :func:`src.services.runs.structural_relief._shape_stop_reason` — the spawner's only
#: consumer of this verdict — matches the four known names and returns ``None`` for
#: anything else, and ``None`` there means SPAWN. A fifth literal would therefore fail
#: OPEN and bill exactly the render this guard exists to refuse. ``STOP_UNDERFULL``
#: already carries the sibling "no count satisfies both bounds" case a few lines below
#: (the tolerance clamp that cannot shrink), so the meaning is consistent rather than
#: stretched; the direction of the miss lives in :attr:`ReliefShape.stop_detail`, which
#: is recorded on the shape itself. Giving the over-band case its own name is a
#: two-line follow-up in ``structural_relief.py`` (a branch in ``_shape_stop_reason``),
#: which is not this change's to make.
_STOP_OUT_OF_BAND: ReliefShapeVerdict = "STOP_UNDERFULL"


def _out_of_band_detail(
    *,
    beat_count: int,
    per_beat: float,
    target_seconds: float,
    band: tuple[float, float],
    yield_label: str = "measured",
) -> str:
    """One sentence naming ONLY what the enumeration proved, with the counts it examined.

    Every claim here is read back off the same candidate list the refusal was decided
    from — the legal counts, their seconds, and the two that bracket the band. No word
    in it asserts an impossibility wider than that list.
    """
    low, high = band
    candidates = [(k, k * per_beat) for k in range(1, beat_count)]
    below = [f"{k} beat(s) = {secs:.2f}s" for k, secs in candidates if secs < low]
    above = [f"{k} beat(s) = {secs:.2f}s" for k, secs in candidates if secs > high]
    bracket = ", ".join(
        part for part in (below[-1] if below else "", above[0] if above else "") if part
    )
    return (
        f"no legal beat count lands inside the +/-{DURATION_TOLERANCE_FRACTION:.0%} band "
        f"[{low:.2f}s, {high:.2f}s] of the {target_seconds:.2f}s order at "
        f"{per_beat:.2f}s {yield_label}/beat: the child must shrink below {beat_count} beat(s), "
        f"and all {len(candidates)} legal count(s) were checked — the closest are "
        f"{bracket} — refusing to bill a render that lands outside the band"
    )


@dataclass(frozen=True)
class ReliefShape:
    """The shape verdict plus every number the callers need to log or explain it."""

    verdict: ReliefShapeVerdict
    beat_count: int
    target_beats: int
    overrun_seconds: float
    target_seconds: float
    seconds_floor: float | None
    #: The parent's MEASURED footage seconds per pinned beat (``None`` = unmeasured).
    #: This is what replaced ``gen._SECONDS_PER_SCENARIO``: a number the recorder
    #: actually produced, not one extrapolated from two hand-observed samples.
    measured_seconds_per_beat: float | None = None
    #: Smallest beat count that can reach ``reconcile_timeline``'s underfull floor at
    #: that measured yield; 0 when unmeasured (no judgement made).
    min_beats_for_floor: int = 0
    #: Set when the shrink target was raised to ``min_beats_for_floor`` — relief cut
    #: deeper than the underfull floor allows, and re-sizing beats refusing.
    resized_for_floor: bool = False
    #: What :func:`next_target_beats` asked for BEFORE the underfull clamp. Carried so a
    #: caller can log the re-size without re-deriving it (two components computing one
    #: quantity two ways is how they end up disagreeing).
    shrink_target_beats: int = 0
    #: Smallest beat count that, at ``measured_seconds_per_beat``, still lands at or
    #: above ``target_seconds x (1 - DURATION_TOLERANCE_FRACTION)`` — the LOWER
    #: conformance bound. 0 when unmeasured / no target (no judgement made).
    tolerance_min_beats: int = 0
    #: Set when the shrink target was raised to :attr:`tolerance_min_beats`, i.e. the
    #: overrun arithmetic asked for a cut that would have UNDERSHOT the order. Distinct
    #: from :attr:`resized_for_floor`, which is the 50% underfull gate, not the +/-10%
    #: conformance band.
    resized_for_tolerance: bool = False
    #: One sentence naming the arithmetic behind a STOP verdict, recorded on the shape
    #: itself so the reason survives to whoever inspects it. Empty on ``SPAWN``.
    stop_detail: str = ""
    #: What the PARENT actually delivered — the caller's
    #: ``duration_conformance.delivered_seconds`` off the master, after the same
    #: finite-or-``None`` coercion every other duration on this path gets. ``None`` means
    #: the delivery was UNMEASURED (no conformance row, or a non-finite figure), never
    #: "measured as zero"; the only guard that reads it is gated on it being a number.
    #:
    #: Carried on EVERY verdict — including the ones that never read it — so a future
    #: session can re-measure the population from the persisted rows instead of
    #: re-deriving it. ``None`` on a row therefore means one thing only: the delivery was
    #: not measured.
    status_quo_seconds: float | None = None
    #: Which measure sized the child's cut and its band/tolerance clamps — ``"delivered"``
    #: only when the parent's delivery was MEASURED, lay ABOVE the band and did NOT exceed
    #: the footage plan; ``"footage_delivered_exceeds_plan"`` when it was measured above
    #: the footage plan (sized from footage, the pre-existing behaviour);
    #: ``"footage_basis_refused"`` when a delivery was eligible but the footage basis had
    #: already refused (this row IS that footage refusal, unchanged); ``"footage"``
    #: otherwise (also the pre-existing behaviour). See :func:`_sizing_basis` and
    #: :func:`decide_relief_shape`.
    sizing_basis: ReliefSizingBasis = "footage"
    #: The overrun the cut was sized from, in ``sizing_basis`` units: ``delivered -
    #: target`` on the delivered basis, the footage ``overrun_seconds`` otherwise.
    #: ``overrun_seconds`` itself stays the FOOTAGE figure on every verdict.
    sizing_overrun_seconds: float = 0.0
    #: Per-beat yield, in ``sizing_basis`` units, that the band/tolerance/underfull clamps
    #: used. On the footage basis it equals :attr:`measured_seconds_per_beat`; on the
    #: delivered basis it is that footage yield scaled by ``delivered / (target +
    #: overrun)``. ``None`` = unmeasured.
    sizing_seconds_per_beat: float | None = None


def _sizing_basis(
    *,
    run: Mapping[str, Any],
    beats: Sequence[object],
    delivered: float | None,
    target_seconds: float,
    overrun: float,
) -> tuple[ReliefSizingBasis, Mapping[str, Any], Sequence[object]]:
    """The ``run``/``beats`` the child is SIZED from, in the units delivery is graded in.

    ``overrun_seconds`` and every per-beat ``seconds`` in the manifest are FOOTAGE
    measures, but a delivered film tracks its NARRATION, which routinely lands well below
    the footage plan (measured: ~7.6s lower). Sizing the cut off the footage overrun
    therefore cuts for an overrun the delivery never had, and the band clamps predict the
    child's length off a per-beat yield the delivery does not show.

    So when the parent's delivery is MEASURED and lies ABOVE the band (inside or under
    it is refused before any sizing runs — :func:`_delivered_basis_refusal`), the whole
    sizing problem is restated in delivered units:

    * the overrun to remove is ``delivered - target``;
    * every beat's ``seconds``/``min_seconds``/``max_seconds`` is scaled by
      ``delivered / (target + overrun)`` — the parent's delivered-per-footage ratio, from
      the manifest identity ``Σseconds = target + overrun`` — so the cheapest-first cut,
      the measured yield, and the band enumeration all predict DELIVERED seconds.

    ⚠️ The proportional scale is a MODEL, not a measurement of the child: it assumes the
    child keeps the parent's delivered/footage ratio. It is the only delivered figure
    available before the child exists.

    Returns the footage inputs UNCHANGED (basis ``"footage"``) whenever the delivery is
    unmeasured, not above the band, or the footage plan is unusable — the pre-existing
    behaviour exactly.

    ⛔ THE DELIVERED BASIS MAY ONLY EVER REFUSE MORE THAN THE FOOTAGE BASIS. It is spend
    logic: a relief child is a PAID re-render. So it is used only while ``delivered <=
    target + overrun`` (the footage plan), where the scale is ``<= 1`` and the delivered
    overrun is no larger than the footage one. ABOVE the footage plan the scale exceeds 1,
    the required cut DEEPENS, the shrink target drops and the footage convergence floor
    drops with it — turning a footage-basis ``STOP_UNCONVERGEABLE`` into a ``SPAWN``
    (measured: target 45 / overrun 28.8 / delivered 90.0 went from a refused 2-beat child
    to a billed 1-beat child). That case returns the footage inputs UNCHANGED under the
    distinct basis ``"footage_delivered_exceeds_plan"``.
    """
    footage_seconds = target_seconds + overrun
    if (
        delivered is None
        or target_seconds <= 0.0
        or footage_seconds <= 0.0
        or delivered <= target_seconds * (1.0 + DURATION_TOLERANCE_FRACTION)
    ):
        return "footage", run, beats
    # OWNER SPEND DECISION NOT MADE: whether paid relief should be allowed at all on a
    # parent that delivers over its footage plan is undecided — do not implement it here.
    if delivered > footage_seconds:
        return "footage_delivered_exceeds_plan", run, beats
    scale = delivered / footage_seconds
    scaled_beats: list[object] = []
    for raw in beats:
        if not isinstance(raw, dict):
            scaled_beats.append(raw)
            continue
        beat = dict(cast("Mapping[str, Any]", raw))
        for key in ("seconds", "min_seconds", "max_seconds"):
            value = beat.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                beat[key] = float(value) * scale
        scaled_beats.append(beat)
    return "delivered", {**run, "overrun_seconds": delivered - target_seconds}, scaled_beats


def _positive(value: object) -> float | None:
    """``value`` as a strictly-positive FINITE float, or ``None`` — ``bool`` is NOT a number.

    ``inf`` is strictly positive and so used to qualify as a per-beat cost. It then sorted
    last in :func:`_cheapest_first_cut`'s ascending sweep and satisfied ``removed >=
    overrun_seconds`` on its own, sizing the cut off a beat length nobody measured.
    ``None`` — "this row cannot support the judgement" — is the answer that branch already
    has for an unreadable row, and it routes to the average fallback rather than to a
    manufactured cut.
    """
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and float(value) > 0.0
    ):
        return float(value)
    return None


def _cheapest_first_cut(beats: Sequence[object], overrun_seconds: float) -> int | None:
    """How many beats gen must DROP to remove ``overrun_seconds``, cheapest beat first.

    ``gen._apply_target_beats`` does not drop an average beat; it drops the CHEAPEST
    droppable ones. Sizing the cut from ``Σmin / n`` therefore mis-sizes it in both
    directions, and the direction depends only on the shape of the spine:

    * expensive-heavy (5 x 6s + 1 x 60s, target 60, overrun 30): the average is 15.0s, so
      2 drops look sufficient — they remove 12s, and the child lands at 78s (+30%).
    * cheap-heavy: the same average over-states what each drop costs the other way, and
      the child lands short. The measured -33.5% pass was this half.

    So the cut is summed off the ACTUAL per-beat cost, ascending, until the running total
    covers the overrun. On a uniform spine this is arithmetically identical
    to the average (every candidate costs the same), which is why the uniform cases keep
    their old answers.

    Returns ``None`` when the manifest cannot support the judgement — no beats, or any
    beat carrying neither a positive ``seconds`` nor a positive ``min_seconds`` — so the
    caller falls back to the average
    rather than modelling a drop sequence off rows that are not all there. When the whole
    spine still does not cover the overrun, every beat is returned: the caller floors the
    resulting target at 1, and the band guards below decide whether that is billable.

    HONESTY NOTE, same as :func:`protected_seconds_floor`: the manifest carries no
    per-beat ``kind``/``protected`` flag, so the cheapest ``k`` here may include beats
    gen is FORCED to keep (index 0, hook roles). The sequence is therefore the cheapest
    LEGAL-OR-NOT one, i.e. a lower bound on what each drop buys and an upper bound on the
    count — it can ask for a deeper cut than gen will manage, never a shallower one, and
    the tolerance band below is what refuses an unbillable result either way.
    """
    if overrun_seconds <= 0.0:
        return None
    costs: list[float] = []
    for raw in beats:
        if not isinstance(raw, dict):
            return None
        beat = cast("Mapping[str, Any]", raw)
        # What dropping this beat REMOVES from the timeline is its current allotment
        # (``seconds``), not its floor: the overrun is measured against the allotments,
        # and ``Σseconds = target + overrun`` is the identity the old average relied on.
        # ``min_seconds`` is the fallback only — on a ``cant_converge`` spine the solver
        # has already pushed every allotment down to its min, so live the two agree.
        cost = _positive(beat.get("seconds")) or _positive(beat.get("min_seconds"))
        if cost is None:
            return None
        costs.append(cost)
    if not costs:
        return None
    costs.sort()
    removed = 0.0
    for index, seconds in enumerate(costs, start=1):
        removed += seconds
        if removed >= overrun_seconds:
            return index
    return len(costs)


def next_target_beats(
    run: Mapping[str, Any], beat_count: int, beats: Sequence[object] | None = None
) -> int:
    """How many beats the relief re-run should target so Σmin can meet the budget.

    Cut enough of the current ``beat_count`` beats to remove the ``overrun``. With
    ``beats`` in hand the cut is summed off the REAL per-beat allotments cheapest-first
    (:func:`_cheapest_first_cut`), matching what ``gen._apply_target_beats`` actually
    drops; without them (or on a manifest too thin to model) it falls back to the old
    average per-beat min (``Σmin / n`` where ``Σmin = target + overrun``). Floored at
    1 beat, and always at least one fewer than now so a spawn always shrinks the spine.

    **No protected-beat clamp.** This used to clamp the result UP to the manifest's
    ``protected_beat_count`` on the premise that gen "physically cannot cut" those beats.
    That premise was FALSE. ``protected_beat_count`` counts
    ``reconcile_timeline._is_protected`` = ``claim_status | locked | time_lock |
    footage-pinned`` — the predicate for what an *in-place* trim/cut/merge inside
    ``reconcile_timeline`` may touch. But the consumer of ``target_beats`` is
    :func:`src.services.stages.demo_video.gen._apply_target_beats`, whose protected set is
    an entirely different, much SMALLER one: index 0 (the positional opener), a
    ``hook_role`` in ``gen._PROTECTED_HOOK_ROLES``, and a ``pin_position``/``time_lock``
    beat that is *not* named in ``relief['over_budget_ids']``. Footage-pinned and
    claim-bearing beats are freely droppable there. So the clamp only ever raised the
    target above what gen could actually deliver: a 12-beat / 10-footage-pinned / 90s
    spine against a 60s target clamped from 7 up to 10, gen dropped two cheap VO beats,
    the child re-rendered at ~90s and did not converge — one wasted PAID render, where
    target 7 would have converged. The lever must be allowed to cut deep; the honest
    protection against a pointless render is the SECONDS feasibility guard in
    :func:`protected_seconds_floor`, not a beat-count clamp read off the wrong predicate.
    """
    n = beat_count
    # Same coercion as :func:`decide_relief_shape`, and it must be HERE rather than only
    # at that caller: this function runs BEFORE the caller's own coercion (it is the
    # ``target_beats`` argument), so a non-finite ``overrun_seconds`` reached the
    # ``math.ceil(overrun / avg_min)`` below unfiltered. With ``overrun = inf`` the
    # average fallback computes ``sigma_min = target + inf = inf``, ``avg_min = inf``,
    # and ``ceil(inf / inf)`` is ``ceil(nan)`` — ``ValueError: cannot convert float NaN
    # to integer``, raised inside the completion path of a run already paid for. Read as
    # ABSENT (0.0), the ``overrun <= 0`` branch below owns it and no judgement is made.
    target = finite_or_none(run.get("target_seconds")) or 0.0
    overrun = finite_or_none(run.get("overrun_seconds")) or 0.0
    if n <= 0:
        return 0  # can't target more beats than exist
    if n == 1:
        return 1
    if overrun <= 0 or target <= 0:
        computed = n - 1
    else:
        cut = _cheapest_first_cut(beats, overrun) if beats is not None else None
        if cut is None:
            sigma_min = target + overrun
            avg_min = sigma_min / n
            cut = max(1, math.ceil(overrun / avg_min)) if avg_min > 0 else 1
        computed = max(1, n - max(1, cut))
    return max(computed, 1)


def protected_seconds_floor(beats: Sequence[object], target_beats: int) -> float | None:
    """Lower bound, in SECONDS, on what a ``target_beats``-beat re-run can possibly cost.

    Any subset of ``target_beats`` beats drawn from this spine costs at least the sum of
    the ``target_beats`` SMALLEST per-beat ``min_seconds``. When that bound already exceeds
    ``target_seconds`` the child render cannot converge no matter which beats gen chooses to
    drop — and a paid render that cannot converge must not be billed.

    Returns ``None`` when the manifest cannot support the judgement (no beats, no positive
    ``min_seconds`` on any beat — an old/thin manifest), so the caller falls back to
    spawning rather than silently suppressing relief on missing data.

    HONESTY NOTE — this is a bound, not a proof. The manifest's per-beat rows carry only
    ``id / seconds / min_seconds / max_seconds / vo_rate / feasible / unsayable`` (see
    ``shared/time_budget.build_timeline_manifest``); they carry NO ``kind`` and no
    per-beat protected flag, so we cannot reconstruct **gen's** protected predicate
    (``i == 0`` / ``hook_role`` / ``pin_position`` / ``time_lock``) here. Two consequences:
    (1) the bound ignores the fact that gen is FORCED to keep specific beats (the opener,
    hook beats), so the true floor is >= this one — the guard is conservative and can only
    fail to skip, never over-skip on that account; (2) the child re-runs gen→record, so a
    regenerated VO beat may come back with a different (usually smaller) ``min_seconds``,
    while a footage-pinned beat's min is its measured clip and is stable. The honest fix is
    for ``build_timeline_manifest`` to emit per-beat ``kind`` + ``protected`` (under gen's
    predicate) so this can sum the truly-unavoidable seconds instead of the k smallest;
    that is a producer change in ``reconcile_timeline`` / ``time_budget`` and is NOT done
    here — do not invent the field at the consumer (repo has a documented dead-signal
    defect class where a reader consumed a field no producer ever wrote).
    """
    if target_beats <= 0:
        return None
    mins: list[float] = []
    for raw in beats:
        if not isinstance(raw, dict):
            continue
        beat = cast("Mapping[str, Any]", raw)
        value = beat.get("min_seconds")
        # A NON-FINITE row is dropped rather than summed. It is not merely noise here:
        # this floor is compared against ``target_seconds`` to decide whether a child
        # render can converge, and BOTH non-finite values fail that comparison OPEN —
        # ``nan > target`` is False, and ``-inf`` drags the sum below any target — so a
        # corrupt row would license a PAID render the real footage cannot deliver. An
        # ``inf`` row fails the other way, manufacturing an unconvergeable verdict out of
        # a length nobody measured. Dropping it lands on this function's existing
        # UNMEASURED answer (``None`` when nothing usable remains), which the caller
        # documents as "fall back to spawning rather than suppress relief on missing
        # data" — a judgement withheld, not an impossibility asserted.
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            mins.append(float(value))
    if not mins or not any(m > 0.0 for m in mins):
        return None  # manifest too thin to judge — do not suppress relief on missing data
    mins.sort()
    return sum(mins[:target_beats])


def _already_conforming(*, delivered: float | None, target_seconds: float) -> float | None:
    """The parent's OWN delivery when it is already inside the +/-10% band, else ``None``.

    Returns the VALUE rather than a bool so the caller can name the seconds in its
    refusal without a second ``is not None`` test (and so a type checker keeps the
    narrowing across the call boundary).

    ``None`` on an unmeasured delivery — "we did not measure it" is not "it conforms",
    and this predicate gates a refusal, so the unmeasured answer must be the one that
    makes no judgement and lets the remaining gates run.

    ``DURATION_TOLERANCE_FRACTION`` is imported from ``duration_conformance``, never
    restated: the guard and the measurement that grades the delivered film have to be
    the same number or this refuses runs the grader would have failed.
    """
    if delivered is None:
        return None
    within = abs(delivered - target_seconds) <= target_seconds * DURATION_TOLERANCE_FRACTION
    return delivered if within else None


def _delivered_basis_refusal(
    *,
    delivered: float | None,
    beats: Sequence[object],
    beat_count: int,
    target_beats: int,
    overrun: float,
    target_seconds: float,
    per_beat: float | None,
    card_seconds: float = 0.0,
) -> ReliefShape | None:
    """Refuse relief off the parent's DELIVERED duration, or ``None`` to fall through.

    See the ``THE DELIVERED BASIS`` note in :func:`decide_relief_shape`. ``None`` on an
    unmeasured delivery or a delivery above the band: only then may the footage-sized
    relief arithmetic decide.
    """
    # ``delivered`` and ``target_seconds`` are BODY figures (title cards taken off, see
    # :func:`decide_relief_shape`); the conformance band is the grader's, on the whole
    # film against the whole order, so the cards go back on for this one comparison.
    order_seconds = target_seconds + card_seconds
    conforming = (
        _already_conforming(
            delivered=None if delivered is None else delivered + card_seconds,
            target_seconds=order_seconds,
        )
        if target_seconds > 0.0
        else None
    )
    if conforming is not None:
        return ReliefShape(
            verdict="STOP_ALREADY_CONFORMING",
            beat_count=beat_count,
            target_beats=target_beats,
            overrun_seconds=overrun,
            target_seconds=target_seconds,
            seconds_floor=protected_seconds_floor(beats, target_beats),
            measured_seconds_per_beat=per_beat,
            shrink_target_beats=target_beats,
            status_quo_seconds=conforming,
            stop_detail=(
                f"the parent already delivered {conforming:.2f}s against a "
                f"{order_seconds:.2f}s order "
                f"({(conforming - order_seconds) / order_seconds:+.2%}), inside the "
                f"+/-{DURATION_TOLERANCE_FRACTION:.0%} conformance band — a paid "
                f"relief child can at best tie it and has come back worse, so "
                f"refusing to bill a render against a delivery that already conforms"
            ),
        )
    if (
        delivered is not None
        and target_seconds > 0.0
        and delivered < target_seconds * (1.0 - DURATION_TOLERANCE_FRACTION)
    ):
        footage_seconds = target_seconds + overrun
        return ReliefShape(
            verdict="STOP_UNDERFULL",
            beat_count=beat_count,
            target_beats=target_beats,
            overrun_seconds=overrun,
            target_seconds=target_seconds,
            seconds_floor=protected_seconds_floor(beats, target_beats),
            measured_seconds_per_beat=per_beat,
            shrink_target_beats=target_beats,
            status_quo_seconds=delivered,
            stop_detail=(
                f"the parent delivered {delivered:.2f}s against a {target_seconds:.2f}s "
                f"order ({(delivered - target_seconds) / target_seconds:+.2%}), already "
                f"UNDER the +/-{DURATION_TOLERANCE_FRACTION:.0%} conformance band, even "
                f"though its footage measured {footage_seconds:.2f}s "
                f"({overrun:+.2f}s over) — delivery tracks narration, not footage, and a "
                f"relief child can only shrink the spine, so refusing to bill a render "
                f"that can only land shorter still"
            ),
        )
    return None


def decide_relief_shape(
    *,
    run: Mapping[str, Any],
    beats: Sequence[object],
    floor_ratio: float = 0.0,
    delivered_seconds: float | None = None,
) -> ReliefShape:
    """Would a relief child actually be spawned for this manifest shape?

    Applies, in the spawner's original order, the two arithmetic guards that follow
    :func:`decide_structural_relief`:

    1. ``STOP_CANNOT_SHRINK`` — nothing to relieve (``overrun <= 0``) or the lever cannot
       shrink the spine any further (a single-beat timeline, or a beat count already at
       the floor). A child would re-render an identical spine and only burn a paid pass.
    2. ``STOP_UNCONVERGEABLE`` — even a perfectly-chosen ``target_beats``-beat spine costs
       at least :func:`protected_seconds_floor` seconds, and that already blows the target.
       The child provably cannot converge and must not be billed. A ``None`` floor means
       the manifest cannot support the judgement, so we spawn as before (never suppress
       relief on missing data).

    ...plus the UNDERFULL guard this pipeline was missing entirely, applied BETWEEN them
    (``floor_ratio`` = ``pipeline.demo_timeline_underfull_fail_ratio``; ``0.0`` disables
    it exactly as it disables the gate itself):

    The relief lever exists to cut an OVERRUN. Nothing stopped it cutting straight
    THROUGH the underfull floor on the way: ``next_target_beats`` floors at 1 beat, so a
    30s order could spawn a child with ``target_beats=1``, which then re-records and
    dies at ``reconcile_timeline``'s 50% floor having paid for a full render. Eleven of
    the twelve production failures of that gate were relief children exactly like this.
    Solving one gate by violating the other is not convergence.

    So the shrink target is CLAMPED UP to :func:`~src.services.stages.shared.duration_plan.plan_beats`'s
    ``min_beats_for_floor`` — the smallest count that, at the parent's MEASURED yield
    (:func:`~src.services.stages.shared.duration_plan.measured_seconds_per_beat`, the
    ffprobe'd footage-pinned rows of this same manifest), can still clear the floor.
    Re-sizing is preferred over refusing throughout: a refused run delivers nothing.
    ``STOP_UNDERFULL`` is returned ONLY when no legal count exists at all — the child
    must shrink (``< beat_count``) and must clear the floor (``>= min_beats_for_floor``)
    and those two are irreconcilable. Billing is append-only with no refund path, so a
    render that provably lands under the floor must never be started.

    Unmeasured manifests make NO judgement here (verdict ``UNMEASURED`` from
    ``plan_beats``): an absent measurement is not evidence of a bad plan, and inventing
    a per-beat constant to fill the gap is the exact defect being removed.

    ...and, when the caller can measure it, the STATUS-QUO guard. ``delivered_seconds``
    is what the parent ACTUALLY shipped (``duration_conformance.delivered_seconds`` off the
    master). Without it every guard above reasons only about the candidate child and never
    about the thing the child is replacing:

    * A parent that had ALREADY delivered +0.04% was billed a child, and that child came
      back WORSE (+5.04%). :data:`STOP_ALREADY_CONFORMING` closes that.
    * A parent whose FOOTAGE measured +4% over delivered -24.43% UNDER (delivery tracks
      narration, not footage). Relief only ever shrinks, so a delivery already under the
      band is refused as ``STOP_UNDERFULL`` off the delivery itself, before any footage
      arithmetic runs.

    ⛔ THERE IS NO SPAWN-SIDE STATUS-QUO GUARD (both guards above only REFUSE), and this docstring asserted one until it was
    caught in the pod. The tempting one — spawn whenever the best legal count is closer to
    the order than the delivery, since four parents shipped at +23.88 / +74.58 / +23.23 /
    +32.34 % while some legal count sat nearer — was BUILT, MEASURED AND REMOVED. It cannot
    work here, and the reason is arithmetic, not taste:

    :func:`protected_seconds_floor` sums the ``target_beats`` smallest ``min_seconds``. On an
    ALL-PINNED spine — which is what ``demo_video`` produces on 12 of 12 measured runs, every
    beat carrying ``min_seconds == max_seconds == seconds`` — that sum IS ``k * per_beat``,
    the candidate itself. The pre-existing ``STOP_UNCONVERGEABLE`` guard below then refuses
    any candidate whose floor exceeds the order. Composed, those two say exactly: *the
    candidate must not exceed the order.* So the reachable counts are ``{k : k * per_beat <=
    target_seconds}`` and NO count above the order is reachable at all — relief can only ever
    UNDERSHOOT. The best reachable candidates for those same four parents are -15.29 / -23.02
    / -15.67 / -10.76 %, all on the side no downstream lever can repair. Both real production
    spawns were likewise below target, and none could have been above it.

    ⇒ Do not re-add it without first refuting that floor arithmetic. The lever is not this
    gate; it is the manifest, which plans a median +39.4% of footage against the order.

    ⛔ ``delivered_seconds=None`` (the default) restores the pre-existing behaviour
    EXACTLY: the status-quo guard is skipped and the band refusal is returned with its
    original wording. A caller that cannot measure the delivery must make no judgement
    here rather than assume one.

    Pure: no settings, no I/O. The spawner performs the spawn and ``reconcile_timeline``
    words its degrade note off the same verdict.

    ⚠️ THE FORECAST AND THE ACTION CAN STILL DIVERGE FOR ONE CALLER. ``encode_master``
    reaches this through :func:`relief_shape_from_manifest` AFTER the master is measured
    and now passes that ``delivered_seconds``, so its verdict matches the spawner's.
    ``reconcile_timeline`` reaches it at PLAN time, before any master exists, so it
    passes no ``delivered_seconds`` and cannot — the number it
    would need is produced by a stage that has not run yet. An already-conforming parent
    is therefore forecast "a shorter-spine re-render will be attempted automatically" and
    then correctly refused by the spawner, which does have the delivery in hand.

    KNOWN AND ACCEPTED. Threading a delivery into a plan-time forecast is not possible,
    and the divergence errs in the safe direction: the note over-promises a free
    correction the customer does not get, rather than the spawner billing a render the
    note never warned about. The alternative — dropping the guard so the two agree — buys
    that agreement with a paid render on a run that already conformed.
    """
    beat_count = len(beats)
    # Same "finite float, or None" coercion as every other duration on this path. A
    # non-finite delivery is read as UNMEASURED, not as a delivery — which routes to the
    # legacy behaviour rather than to a comparison against ``inf``/``nan`` (every ``<=``
    # against ``nan`` is False, so a corrupt figure would silently disable the spawn side
    # of guard 2 while still looking as though the comparison ran).
    delivered = finite_or_none(delivered_seconds)
    # LIKE WITH LIKE. reconcile_timeline reserves title cards INSIDE the order, so the
    # manifest's ``target_seconds`` is the BODY (order minus ``card_seconds``) and its
    # footage figures are body figures — but ``delivered_seconds`` is the whole film,
    # cards included. Compared raw, a carded run delivering exactly its order reads as
    # over by the card seconds and could be billed a PAID child. So the delivery is
    # restated as a BODY delivery here, once, before any guard reads it; the one check
    # that grades the parent against the ORDER (``_already_conforming``) adds the cards
    # back so its band is the grader's band to the millisecond. No ``card_seconds`` on
    # the manifest (pre-card manifests, no title cards) ⇒ 0.0 ⇒ unchanged.
    card_seconds = max(0.0, finite_or_none(run.get("card_seconds")) or 0.0)
    if delivered is not None:
        delivered -= card_seconds
    # Same coercion as the stage boundary (:func:`relief_shape_from_manifest` refuses a
    # non-finite manifest outright; this is the in-function belt for a hand-built ``run``
    # mapping). A non-finite number is read as ABSENT — the 0.0 that ``or 0.0`` already
    # produces for a missing key — which routes to ``plan_beats``' UNMEASURED path and so
    # makes NO judgement. It is not a default duration: 0.0 means "no ordered duration"
    # in this module's vocabulary and every guard below is gated on ``> 0.0``.
    overrun = finite_or_none(run.get("overrun_seconds")) or 0.0
    target_seconds = finite_or_none(run.get("target_seconds")) or 0.0
    # SIZE THE CHILD IN THE UNITS DELIVERY IS GRADED IN (see :func:`_sizing_basis`). A
    # parent with a footage overrun of 9.16s delivered +0.04%; a child cut off that
    # footage overrun is cut for seconds the delivery never had. ``sizing_beats`` feed the
    # cut and the band/tolerance/underfull clamps; the CONVERGENCE floor
    # (:func:`protected_seconds_floor`) stays on the raw footage ``beats``, because it
    # asks whether the child's own ``reconcile_timeline`` solver — which budgets FOOTAGE
    # — can fit the order.
    candidate, sizing_run, sizing_beats = _sizing_basis(
        run=run,
        beats=beats,
        delivered=delivered,
        target_seconds=target_seconds,
        overrun=overrun,
    )
    # ⛔ THE DELIVERED BASIS MAY ONLY NARROW A SPAWN THE FOOTAGE BASIS ALREADY ALLOWS —
    # BY CONSTRUCTION, not by enumerating the ways it could differ. A relief child is a
    # PAID re-render, so the footage basis (the pre-existing decision, exactly) is always
    # evaluated FIRST and any refusal it makes is returned UNCHANGED; the delivery is
    # never consulted to overturn it. Only when footage would SPAWN does the delivered
    # basis decide, and then its answer wins: it may refuse, or spawn a child sized from
    # the delivery. (Sizing from delivery alone turned footage refusals into spawns two
    # separate ways — a deeper cut above the footage plan, and a band enumeration at the
    # scaled yield inside it.)
    footage_shape = _relief_shape_on_basis(
        basis="footage_basis_refused" if candidate == "delivered" else candidate,
        beats=beats,
        sizing_run=run,
        sizing_beats=beats,
        floor_ratio=floor_ratio,
        delivered=delivered,
        overrun=overrun,
        target_seconds=target_seconds,
        card_seconds=card_seconds,
    )
    if candidate != "delivered" or footage_shape.verdict != "SPAWN":
        shape = footage_shape
    else:
        shape = _relief_shape_on_basis(
            basis="delivered",
            beats=beats,
            sizing_run=sizing_run,
            sizing_beats=sizing_beats,
            floor_ratio=floor_ratio,
            delivered=delivered,
            overrun=overrun,
            target_seconds=target_seconds,
            card_seconds=card_seconds,
        )
    log.info(
        "structural relief sizing: basis=%s overrun=%.3fs (footage overrun=%.3fs, "
        "delivered=%s, target=%.3fs, beats=%d) verdict=%s",
        shape.sizing_basis,
        delivered - target_seconds
        if shape.sizing_basis == "delivered" and delivered is not None
        else overrun,
        overrun,
        "unmeasured" if delivered is None else f"{delivered:.3f}s",
        target_seconds,
        beat_count,
        shape.verdict,
    )
    return shape


def _relief_shape_on_basis(
    *,
    basis: ReliefSizingBasis,
    beats: Sequence[object],
    sizing_run: Mapping[str, Any],
    sizing_beats: Sequence[object],
    floor_ratio: float,
    delivered: float | None,
    overrun: float,
    target_seconds: float,
    card_seconds: float = 0.0,
) -> ReliefShape:
    """One pass of :func:`decide_relief_shape`'s guards, sized from ``sizing_run`` /
    ``sizing_beats`` and labelled ``basis``.

    With ``sizing_run is run`` and ``sizing_beats is beats`` this IS the footage-basis
    decision, arithmetic unchanged from before delivered sizing existed. ``beats``,
    ``overrun`` and ``target_seconds`` are always the raw FOOTAGE figures: the recorded
    overrun, the footage yield and the convergence floor never move with the basis.
    """
    beat_count = len(beats)
    sizing_overrun = finite_or_none(sizing_run.get("overrun_seconds")) or 0.0
    target_beats = next_target_beats(sizing_run, beat_count, sizing_beats)
    if overrun <= 0 or target_beats >= beat_count:
        return ReliefShape(
            verdict="STOP_CANNOT_SHRINK",
            beat_count=beat_count,
            target_beats=target_beats,
            overrun_seconds=overrun,
            target_seconds=target_seconds,
            seconds_floor=None,
            status_quo_seconds=delivered,
            sizing_basis=basis,
            sizing_overrun_seconds=sizing_overrun,
        )
    footage_per_beat = measured_seconds_per_beat(beats)
    per_beat = measured_seconds_per_beat(sizing_beats)
    yield_label = "delivered" if basis == "delivered" else "measured"

    # ── THE DELIVERED BASIS: JUDGE THE PARENT BY WHAT IT SHIPPED, NOT BY ITS FOOTAGE. ──
    # ``overrun_seconds`` and ``per_beat`` are FOOTAGE measures — ``per_beat`` is the
    # ffprobe'd footage-pinned rows of this manifest. But a delivered film is exactly its
    # NARRATION length, not its footage length, and the two routinely disagree. Measured
    # live (run 01a0ab23, showcase, 30s order): footage said 31.28s (+4%, ``overrun=1.28``)
    # and the gate refused with a footage-arithmetic ``STOP_UNDERFULL`` sentence, while the
    # film actually DELIVERED 22.671s (-24.43%). The verdict happened to be a stop, but it
    # was decided — and explained — from a number that does not predict delivery; on a
    # spine where a footage count did fit the band, the very same parent would have been
    # billed a SHRINK child against a film that was already 24% short.
    #
    # So when the caller has measured the delivery (``duration_conformance.
    # delivered_seconds`` off the master), the parent's OWN delivery decides first, BEFORE
    # any footage arithmetic is consulted:
    #
    # * GUARD 1 — inside the +/-10% band: the parent already conforms. A child can at best
    #   tie it, and one measured child came back worse (+0.04% parent -> +5.04% child).
    # * GUARD 2 — BELOW the band: relief is a SHRINK lever (``target_beats < beat_count``
    #   always). No shrink can lengthen an under-delivery; every legal child is predicted
    #   shorter still. Refused as ``STOP_UNDERFULL`` (the delivery IS under the band) —
    #   deliberately an existing literal, because ``structural_relief._shape_stop_reason``
    #   maps verdicts by name and the comment there records that a new literal once failed
    #   open and billed.
    # * Above the band: the overrun is real in delivered terms, and the relief lever below
    #   runs SIZED IN DELIVERED UNITS (``_sizing_basis`` above) — not off the footage
    #   overrun, which over-cuts by however far the footage plan exceeds the delivery.
    #
    # Both use the SAME +/-10% the delivered film is graded by (``DURATION_TOLERANCE_
    # FRACTION``, imported, never restated). Deliberately OUTSIDE the ``floor_ratio`` /
    # ``per_beat`` gate: they reason only about the PARENT'S delivery against the order,
    # so an unmeasurable per-beat yield (an unpinned spine) is no reason to skip them.
    # ``delivered_seconds=None`` (plan-time callers, or an unmeasured master) skips both
    # and restores the pre-existing footage-only behaviour exactly.
    refusal = _delivered_basis_refusal(
        delivered=delivered,
        beats=beats,
        beat_count=beat_count,
        target_beats=target_beats,
        overrun=overrun,
        target_seconds=target_seconds,
        per_beat=footage_per_beat,
        card_seconds=card_seconds,
    )
    if refusal is not None:
        return refusal
    # A relief child MUST shrink, so the largest count it may legally plan is one fewer
    # beat than the parent has. If the underfull floor needs more than that, no legal
    # sizing exists — which is what `plan_beats` reports as UNSATISFIABLE.
    plan: DurationPlan = plan_beats(
        target_beats,
        target_seconds=target_seconds,
        floor_ratio=floor_ratio,
        per_beat_seconds=per_beat,
        max_beats=beat_count - 1,
    )
    if plan.verdict == "UNSATISFIABLE":
        return ReliefShape(
            verdict="STOP_UNDERFULL",
            beat_count=beat_count,
            target_beats=target_beats,
            overrun_seconds=overrun,
            target_seconds=target_seconds,
            seconds_floor=protected_seconds_floor(beats, target_beats),
            measured_seconds_per_beat=footage_per_beat,
            sizing_basis=basis,
            sizing_overrun_seconds=sizing_overrun,
            sizing_seconds_per_beat=per_beat,
            min_beats_for_floor=plan.min_beats_for_floor,
            shrink_target_beats=target_beats,
            status_quo_seconds=delivered,
            stop_detail=(
                f"no count <= {beat_count - 1} reaches the underfull floor at "
                f"{per_beat or 0.0:.2f}s {yield_label}/beat against a {target_seconds:.2f}s target"
            ),
        )
    resized = plan.verdict == "RESIZED"
    shrink_target = target_beats
    target_beats = plan.beats

    # ── THE LOWER BOUND. ──────────────────────────────────────────────────────────────
    # The clamp above keeps the child off ``reconcile_timeline``'s 50% underfull FLOOR.
    # It does NOT keep it inside the band the delivered film is actually judged by:
    # ``duration_conformance``'s +/-10%. Those are two different lines and only one of
    # them was drawn.
    #
    # Measured live 2026-09-10: a 30s order delivered 47.045s (+56.8%), relief spawned a
    # child, and the child delivered 19.95s (-33.5%). The undershoot is STRUCTURAL, not
    # bad luck — ``next_target_beats`` computes its ``cut`` from the AVERAGE per-beat
    # minimum while ``gen._apply_target_beats`` drops the CHEAPEST droppable beats, so
    # the correction systematically overshoots, and nothing below it said "too short".
    # Both passes were undeliverable; a relief loop that trades +57% for -34% has
    # converged on nothing.
    #
    # So the shrink target is also clamped UP to the smallest count that still lands at
    # or above ``ordered x (1 - tolerance)``. Re-sizing is preferred over refusing here
    # exactly as it is above — but when the raised count can no longer SHRINK the spine
    # (``>= beat_count``) the two bounds are irreconcilable and NO child is spawned. A
    # paid render that is arithmetically certain to land outside the band must not be
    # billed, in either direction.
    #
    # ``DURATION_TOLERANCE_FRACTION`` is IMPORTED from ``duration_conformance``, never
    # restated: the guard and the measurement that judges it must be the same number.
    #
    # GATED ON THE SAME ``floor_ratio <= 0`` KILL-SWITCH as the clamp above, deliberately.
    # This is a different line (the +/-10% conformance band, not the 50% underfull gate),
    # but ``floor_ratio=0`` is this function's documented "make no duration judgement at
    # all" switch, and a kill-switch that no longer switches everything off is a trap for
    # whoever next reaches for it. Production runs at 0.5, where the guard is live.
    #
    # ⚠️ THIS BLOCK IS WHY THE COERCION AT THE TOP MATTERS, and why it must land on 0.0
    # rather than pass a non-finite value through. With ``target_seconds = inf`` the
    # ``> 0.0`` test below PASSES, ``band_seconds`` returns the "no band" ``(0.0, 0.0)``,
    # ``beat_counts_within_band`` therefore comes back EMPTY — and an empty enumeration is
    # exactly what licenses the ``_STOP_OUT_OF_BAND`` refusal further down. An unreadable
    # number would have produced a confident "no count fits": a false impossibility claim
    # built on no measurement at all. Read as ABSENT (0.0), it makes no judgement instead.
    tolerance_min_beats = 0
    if floor_ratio > 0.0 and per_beat is not None and per_beat > 0.0 and target_seconds > 0.0:
        lower_bound = target_seconds * (1.0 - DURATION_TOLERANCE_FRACTION)
        tolerance_min_beats = beats_for_seconds(lower_bound, per_beat)
        if tolerance_min_beats > target_beats:
            if tolerance_min_beats >= beat_count:
                return ReliefShape(
                    verdict="STOP_UNDERFULL",
                    beat_count=beat_count,
                    target_beats=target_beats,
                    overrun_seconds=overrun,
                    target_seconds=target_seconds,
                    seconds_floor=protected_seconds_floor(beats, target_beats),
                    measured_seconds_per_beat=footage_per_beat,
                    sizing_basis=basis,
                    sizing_overrun_seconds=sizing_overrun,
                    sizing_seconds_per_beat=per_beat,
                    # The binding lower requirement, so the caller's stop message names
                    # the count that actually refused rather than the looser 50% one.
                    min_beats_for_floor=max(plan.min_beats_for_floor, tolerance_min_beats),
                    shrink_target_beats=shrink_target,
                    tolerance_min_beats=tolerance_min_beats,
                    stop_detail=(
                        f"no relief cut satisfies both bounds: the child must shrink below "
                        f"{beat_count} beat(s), but staying within "
                        f"{DURATION_TOLERANCE_FRACTION:.0%} of the {target_seconds:.2f}s "
                        f"order needs >= {tolerance_min_beats} beat(s) at "
                        f"{per_beat:.2f}s {yield_label}/beat (>= {lower_bound:.2f}s delivered) "
                        f"— refusing to bill a render that would land under the band"
                    ),
                )
            target_beats = tolerance_min_beats

        # ── THE UPPER BOUND. ──────────────────────────────────────────────────────────
        # The clamp above is a floor with no ceiling above it. Every count-sizing call in
        # this pipeline rounds UP (``beats_for_seconds`` = ``ceil``), including the one
        # ``plan_beats`` uses as its "ceiling", so the count that survives the clamps can
        # itself be OVER the band — and ``STOP_UNDERFULL`` had no counterpart to catch it.
        #
        # Live run 01a089c3, 30.00s order, 22.00s guaranteed per scenario, 5-beat parent:
        # the shrink asked for 3, ``plan_beats`` RESIZED_DOWN to 2 (its ceiling is
        # ceil(30/22) = 2, i.e. 44.00s — already past the order it is supposed to bound),
        # ``tolerance_min_beats`` is ceil(27/22) = 2 so the lower clamp did not fire, and
        # a paid child was spawned at 2 x 22.00s = 44.00s = +46.7%. The child then hit
        # ``gen._apply_feasibility_floor`` (30 // 22 = 1) and refitted to 22.00s = -26.7%.
        # NEITHER count is inside the band; at this yield NO count is.
        #
        # So the products are ENUMERATED and checked, rather than re-derived by the same
        # rounding that produced the bad count. When the enumeration comes back empty the
        # child is refused; when it does not, the planned count is snapped to the nearest
        # member (by monotonicity that is ``tolerance_min_beats`` from above, or a count
        # below it when the plan had over-shot), so re-sizing still beats refusing.
        band = band_seconds(target_seconds, DURATION_TOLERANCE_FRACTION)
        in_band = beat_counts_within_band(
            per_beat_seconds=per_beat, max_beats=beat_count - 1, band=band
        )
        if not in_band:
            return ReliefShape(
                verdict=_STOP_OUT_OF_BAND,
                beat_count=beat_count,
                target_beats=target_beats,
                overrun_seconds=overrun,
                target_seconds=target_seconds,
                seconds_floor=protected_seconds_floor(beats, target_beats),
                measured_seconds_per_beat=footage_per_beat,
                sizing_basis=basis,
                sizing_overrun_seconds=sizing_overrun,
                sizing_seconds_per_beat=per_beat,
                min_beats_for_floor=max(plan.min_beats_for_floor, tolerance_min_beats),
                shrink_target_beats=shrink_target,
                tolerance_min_beats=tolerance_min_beats,
                status_quo_seconds=delivered,
                stop_detail=_out_of_band_detail(
                    beat_count=beat_count,
                    per_beat=per_beat,
                    target_seconds=target_seconds,
                    band=band,
                    yield_label=yield_label,
                ),
            )
        if target_beats not in in_band:
            target_beats = min(in_band, key=lambda k: abs(k - target_beats))

    # Computed AFTER the clamp, deliberately: re-sizing up raises the unavoidable
    # seconds, so the over-budget feasibility question must be asked about the count
    # that will actually be planned, not the one that was superseded.
    seconds_floor = protected_seconds_floor(beats, target_beats)
    if target_seconds > 0 and seconds_floor is not None and seconds_floor > target_seconds:
        return ReliefShape(
            verdict="STOP_UNCONVERGEABLE",
            beat_count=beat_count,
            target_beats=target_beats,
            overrun_seconds=overrun,
            target_seconds=target_seconds,
            seconds_floor=seconds_floor,
            measured_seconds_per_beat=footage_per_beat,
            sizing_basis=basis,
            sizing_overrun_seconds=sizing_overrun,
            sizing_seconds_per_beat=per_beat,
            min_beats_for_floor=plan.min_beats_for_floor,
            resized_for_floor=resized,
            shrink_target_beats=shrink_target,
            tolerance_min_beats=tolerance_min_beats,
            resized_for_tolerance=target_beats == tolerance_min_beats > plan.beats,
            status_quo_seconds=delivered,
            stop_detail=(
                f"even the cheapest {target_beats}-beat spine costs "
                f"{seconds_floor or 0.0:.2f}s against a {target_seconds:.2f}s target"
            ),
        )
    return ReliefShape(
        verdict="SPAWN",
        beat_count=beat_count,
        target_beats=target_beats,
        overrun_seconds=overrun,
        target_seconds=target_seconds,
        seconds_floor=seconds_floor,
        measured_seconds_per_beat=footage_per_beat,
        sizing_basis=basis,
        sizing_overrun_seconds=sizing_overrun,
        sizing_seconds_per_beat=per_beat,
        min_beats_for_floor=plan.min_beats_for_floor,
        resized_for_floor=resized,
        shrink_target_beats=shrink_target,
        tolerance_min_beats=tolerance_min_beats,
        resized_for_tolerance=target_beats == tolerance_min_beats > plan.beats,
        status_quo_seconds=delivered,
    )


def relief_shape_from_manifest(
    manifest: Mapping[str, object],
    *,
    floor_ratio: float = 0.0,
    delivered_seconds: float | None = None,
) -> ReliefShape | None:
    """:func:`decide_relief_shape` straight off a whole ``timeline_manifest`` dict.

    The convenience entry point for callers holding the manifest rather than its parts
    (``reconcile_timeline``, which has just built it). ``None`` when the manifest cannot
    support the judgement at all — no ``run`` mapping or no ``beats`` list — which the
    caller must read as "cannot promise a spawn", never as "will spawn".

    A run whose ``target_seconds``/``overrun_seconds`` is present but NON-FINITE gets that
    same ``None``. This is the boundary for the in-process path
    ``config_overrides -> direction.timing -> manifest["run"] -> here``: an ordered
    duration that is not a number is no order, so there is nothing to judge a re-render
    against. Refusing to promise a spawn is the conservative direction — it withholds a
    PAID render rather than asserting the render is impossible.

    ``delivered_seconds`` is forwarded unchanged: a caller that has already measured the
    master (``encode_master``) passes it so its forecast is sized and guarded exactly as
    the spawner's will be; a caller that runs before any master exists
    (``reconcile_timeline``) passes nothing and gets the footage-only answer.
    """
    raw_run = manifest.get("run")
    raw_beats = manifest.get("beats")
    if not isinstance(raw_run, dict) or not isinstance(raw_beats, list):
        return None
    run_map = cast("Mapping[str, Any]", raw_run)
    for key in ("target_seconds", "overrun_seconds"):
        raw_value = run_map.get(key)
        if raw_value is not None and finite_or_none(raw_value) is None:
            return None
    return decide_relief_shape(
        run=run_map,
        beats=cast("Sequence[object]", raw_beats),
        floor_ratio=floor_ratio,
        delivered_seconds=delivered_seconds,
    )
