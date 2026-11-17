"""Deterministic retention / pacing analysis (advisory only) — ORCH-0.3.

Pure helpers, no DB, no client, no ``ctx``. Mirrors the frozen-dataclass +
free-function style of ``loop_engine.py``: every function is a total function of
its arguments, so the same beat list always yields byte-identical output and the
module can never issue a render or a network call.

The input is a plain ``list`` of *beats* — narration/footage units carrying the
``AuthoredLine`` / ``DemoVideoLine`` field shape without importing those models
(kept decoupled so any caller with the same duck-typed fields can drive it).
A beat may be a ``dict`` (the JSONB shape ``ctx.prior_output`` returns) or any
object exposing the same attributes. Fields consumed:

- ``id``      — the spine key (``clip-<n>``); read-only, order never changes.
- ``seconds`` — beat duration (float).
- ``clip``    — footage tag; a beat is *footage-pinned* when this is truthy.

Everything is scored around the "31s cliff" — the empirical retention drop-off
early in a video — so a talking-head open with no footage, or a long static run
with no pattern interrupt, is flagged as bounce risk. Purely advisory: nothing
here gates a run; the caller attaches the insight strings to a stage output.

Weights/thresholds resolve through ``resolve_with_provenance`` from a *passed*
dict (never reaching into a snapshot/ctx), so an author's ``direction.retention``
override wins its rung and an absent knob falls to the documented default.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

from src.services.stages.shared.provenance import resolve_with_provenance

# --- Weight/threshold defaults (author-overridable via direction.retention) ----
# Every knob is a duration in seconds. Keep this the single source of truth: both
# the scoring functions and ``resolve_retention_weights`` (used by run_setup to
# stamp the flat ``retention_weights`` block) iterate it.
RETENTION_WEIGHT_DEFAULTS: dict[str, float] = {
    # Latest a viewer should wait before the first footage-pinned beat plays.
    "max_time_to_first_footage_seconds": 8.0,
    # Longest run of consecutive footage-less ("dead") beats before it reads as a
    # static talking-head stretch that bleeds retention.
    "max_dead_beat_run_seconds": 12.0,
    # Longest allowed gap between visual pattern interrupts (footage cut-ins).
    "max_pattern_interrupt_gap_seconds": 15.0,
}

# Watch-curve decay model (deterministic). A single-exponential decay in the beat
# time-offset, with an extra multiplicative drop once the offset crosses the
# empirical early cliff — so the curve visibly bends at ~30s.
_DECAY_TAU_SECONDS = 60.0
_CLIFF_OFFSET_SECONDS = 30.0
_CLIFF_MULTIPLIER = 0.85
_CURVE_PRECISION = 6
_SECONDS_PRECISION = 3


# ------------------------------------------------------------------ accessors --
def _field(beat: object, name: str) -> object:
    """Read ``name`` off a beat that may be a ``dict`` or an attribute-bag."""
    if isinstance(beat, dict):
        return cast("dict[str, object]", beat).get(name)
    return getattr(beat, name, None)


def coerce_beat_seconds(value: object) -> float | None:
    """THE canonical beat-duration coercer — ``None`` when the value cannot be a duration.

    Single source of truth shared with ``gestalt_metrics.build_beat_intervals`` so the
    watch curve and the cut→beat containment join can never disagree about how long a
    beat is. It is deliberately PERMISSIVE about the wire shape (a JSONB spine round-trips
    ``5.0`` as the string ``"5.0"``, and both shapes legitimately occur on the SAME spine),
    but strict about the value: ``None`` ⇒ ``0.0`` (an explicitly duration-less beat),
    a non-finite or unparseable value ⇒ ``None`` so the caller can decide loudly.

    Historically these two call sites diverged: this one parsed ``"5.0"`` while
    ``build_beat_intervals`` used the STRICT ``as_float`` and silently read it as ``0.0``.
    An all-string spine collapsed the join to a positional zip; a MIXED spine was worse —
    it produced zero-width beat spans that no cut can ever fall inside. Keep them unified.
    """
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return None
    try:
        parsed = float(cast("str | float | int", value))
    except TypeError, ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _seconds(beat: object) -> float:
    """Beat duration as a float; a missing/garbage value counts as ``0.0``.

    The TOLERANT wrapper over :func:`coerce_beat_seconds`: retention scoring is purely
    advisory, so an unparseable duration degrades to ``0.0`` rather than aborting a
    watch-curve. ``build_beat_intervals`` deliberately does NOT tolerate it — a silent
    ``0.0`` there fabricates a zero-width span that mis-joins every cut.
    """
    return coerce_beat_seconds(_field(beat, "seconds")) or 0.0


def _is_footage(beat: object) -> bool:
    """True when the beat is pinned to a footage clip (truthy ``clip``)."""
    return bool(_field(beat, "clip"))


def _weight(weights: Mapping[str, object] | None, key: str, default: float) -> float:
    """Resolve one knob down the provenance ladder from a *passed* weights dict.

    An explicit value in ``weights`` is the ``override`` rung; an absent key
    falls through to ``default``. The provenance record itself is discarded here
    (advisory scoring only needs the value), but resolution goes through the
    canonical ladder so overrides behave identically to every other knob.
    """
    weights = weights or {}
    if key in weights:
        resolved = resolve_with_provenance(key, override=weights[key], default=default)
    else:
        resolved = resolve_with_provenance(key, default=default)
    return float(resolved.value)


def resolve_retention_weights(cfg: dict[str, object]) -> dict[str, float]:
    """Resolve every known retention knob from a ``direction.retention`` block.

    Returns a flat ``{key: value}`` dict with all knobs present (author override
    where supplied, else default), suitable for stashing as ``retention_weights``
    on the config snapshot. Resolution routes through ``resolve_with_provenance``.
    """
    return {key: _weight(cfg, key, default) for key, default in RETENTION_WEIGHT_DEFAULTS.items()}


# -------------------------------------------------------------------- report ---
@dataclass(frozen=True)
class BeatBudget:
    """Allotted vs summed beat time for one cut."""

    target_seconds: float
    total_seconds: float
    delta_seconds: float  # total - target (positive ⇒ over)
    over_budget: bool
    under_budget: bool


@dataclass(frozen=True)
class RetentionReport:
    """Deterministic retention read-out for one beat list.

    ``insights`` is the flat, human-readable, JSON-serializable list the caller
    surfaces (advisory). ``watch_curve`` carries one predicted-retention point per
    beat, in spine order. All other fields are the structured inputs behind the
    insight strings, kept for callers/tests that want the raw numbers.
    """

    total_seconds: float
    target_seconds: float | None
    time_to_first_footage: float
    over_budget: bool
    watch_curve: list[float]
    risk_flags: list[str]
    interrupt_gaps: list[dict[str, float]]
    insights: list[str]


# ------------------------------------------------------------------ functions --
def beat_budget(beats: Sequence[object], target_seconds: float) -> BeatBudget:
    """Compare the summed beat duration against an allotted ``target_seconds``."""
    total = sum(_seconds(b) for b in (beats or []))
    target = float(target_seconds)
    delta = total - target
    return BeatBudget(
        target_seconds=round(target, _SECONDS_PRECISION),
        total_seconds=round(total, _SECONDS_PRECISION),
        delta_seconds=round(delta, _SECONDS_PRECISION),
        over_budget=total > target,
        under_budget=total < target,
    )


def time_to_first_footage(beats: Sequence[object]) -> float:
    """Cumulative seconds elapsed before the first footage-pinned beat starts.

    When no beat is footage-pinned the whole cut is talking-head, so this returns
    the total duration (footage never appears within the video).
    """
    offset = 0.0
    for beat in beats or []:
        if _is_footage(beat):
            return round(offset, _SECONDS_PRECISION)
        offset += _seconds(beat)
    return round(offset, _SECONDS_PRECISION)


def predicted_watch_curve(beats: Sequence[object]) -> list[float]:
    """Deterministic predicted-retention fraction at the START of each beat.

    One point per beat, in spine order, in ``[0, 1]``. A single-exponential decay
    in the beat time-offset with an extra drop past the early cliff. Empty beats
    → empty curve.
    """
    curve: list[float] = []
    offset = 0.0
    for beat in beats or []:
        retention = math.exp(-offset / _DECAY_TAU_SECONDS)
        if offset >= _CLIFF_OFFSET_SECONDS:
            retention *= _CLIFF_MULTIPLIER
        curve.append(round(retention, _CURVE_PRECISION))
        offset += _seconds(beat)
    return curve


def _longest_dead_beat_run(beats: Sequence[object]) -> tuple[float, float]:
    """Longest run of consecutive footage-less beats → (run_seconds, start_offset)."""
    longest = 0.0
    longest_start = 0.0
    run = 0.0
    run_start = 0.0
    offset = 0.0
    for beat in beats or []:
        dur = _seconds(beat)
        if _is_footage(beat):
            run = 0.0
            run_start = offset + dur
        else:
            if run == 0.0:
                run_start = offset
            run += dur
            if run > longest:
                longest = run
                longest_start = run_start
        offset += dur
    return round(longest, _SECONDS_PRECISION), round(longest_start, _SECONDS_PRECISION)


def retention_risk(
    beats: Sequence[object], weights: Mapping[str, object] | None = None
) -> list[str]:
    """Flag the 31s-cliff risks: late first footage and long dead-beat runs.

    Empty beats → no flags. Advisory strings only.
    """
    beat_list = list(beats or [])
    if not beat_list:
        return []
    max_ttff = _weight(weights, "max_time_to_first_footage_seconds", 8.0)
    max_dead = _weight(weights, "max_dead_beat_run_seconds", 12.0)

    flags: list[str] = []
    if not any(_is_footage(b) for b in beat_list):
        flags.append(
            "no footage-pinned beat: the whole cut is talking-head — high 31s-cliff bounce risk"
        )
    else:
        ttff = time_to_first_footage(beat_list)
        if ttff > max_ttff:
            flags.append(
                f"first footage at {ttff:.1f}s exceeds the {max_ttff:.1f}s budget: "
                "viewers may bounce before the payoff"
            )

    run_seconds, run_start = _longest_dead_beat_run(beat_list)
    if run_seconds > max_dead:
        flags.append(
            f"dead-beat run of {run_seconds:.1f}s starting at {run_start:.1f}s exceeds "
            f"the {max_dead:.1f}s bound: add a pattern interrupt"
        )
    return flags


def pattern_interrupt_gaps(
    beats: Sequence[object], weights: Mapping[str, object] | None = None
) -> list[dict[str, float]]:
    """Offsets where the gap between visual pattern interrupts exceeds the bound.

    Interrupts are the video start, each footage-pinned beat, and the video end.
    Returns ``[{"start_offset", "gap_seconds"}, ...]`` in spine order. Empty beats
    → empty list.
    """
    beat_list = list(beats or [])
    if not beat_list:
        return []
    bound = _weight(weights, "max_pattern_interrupt_gap_seconds", 15.0)

    interrupts: list[float] = [0.0]
    offset = 0.0
    for beat in beat_list:
        if _is_footage(beat):
            interrupts.append(offset)
        offset += _seconds(beat)
    interrupts.append(offset)  # video end

    # Collapse coincident interrupts (e.g. a footage beat at offset 0.0) so a
    # zero-length "gap" never surfaces.
    ordered: list[float] = []
    for value in interrupts:
        if not ordered or ordered[-1] != value:
            ordered.append(value)

    gaps: list[dict[str, float]] = []
    for prev, nxt in pairwise(ordered):
        gap = nxt - prev
        if gap > bound:
            gaps.append(
                {
                    "start_offset": round(prev, _SECONDS_PRECISION),
                    "gap_seconds": round(gap, _SECONDS_PRECISION),
                }
            )
    return gaps


def analyze_retention(
    beats: Sequence[object],
    *,
    weights: Mapping[str, object] | None = None,
    target_seconds: float | None = None,
) -> RetentionReport:
    """Compose the full deterministic retention read-out for a beat list.

    Empty beats → an all-empty report (``insights == []``), regardless of weights
    or target, so an opted-out / footage-less run is byte-identical to before.
    """
    beat_list = list(beats or [])
    curve = predicted_watch_curve(beat_list)
    risk_flags = retention_risk(beat_list, weights)
    gaps = pattern_interrupt_gaps(beat_list, weights)
    ttff = time_to_first_footage(beat_list) if beat_list else 0.0
    total = round(sum(_seconds(b) for b in beat_list), _SECONDS_PRECISION)

    insights: list[str] = list(risk_flags)
    for gap in gaps:
        insights.append(
            f"no pattern interrupt for {gap['gap_seconds']:.1f}s from {gap['start_offset']:.1f}s: "
            "the cut goes static — cut to footage or a b-roll beat"
        )

    over_budget = False
    target: float | None = None
    if target_seconds is not None:
        budget = beat_budget(beat_list, target_seconds)
        target = budget.target_seconds
        over_budget = budget.over_budget
        # Only an over-budget cut is a retention concern; under-budget is fine.
        if beat_list and budget.over_budget:
            insights.append(
                f"beats sum to {budget.total_seconds:.1f}s vs the {budget.target_seconds:.1f}s "
                f"target (+{budget.delta_seconds:.1f}s): trim to hold pace"
            )

    return RetentionReport(
        total_seconds=total,
        target_seconds=target,
        time_to_first_footage=ttff,
        over_budget=over_budget,
        watch_curve=curve,
        risk_flags=risk_flags,
        interrupt_gaps=gaps,
        insights=insights,
    )
