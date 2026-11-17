"""Global time-budget reconciliation — the master-timeline solver (C13 / aspect A22).

Today four owners each derive timing independently (the authored beat ``seconds``,
the word budget, the TTS clip length, the card dwell) and the last writer wins by
accident: ``shot_assembly`` overwrites every beat's budget with the probed length
of whatever WAV the TTS happened to produce, and the deliverable's
``target_duration_seconds`` buys nothing. This module is the single **arithmetic**
authority that ends that fight: stages declare a :class:`TimeDemand`
(``min``/``ideal``/``max``/``elasticity``) per beat, and :func:`reconcile_timeline`
disposes one per-beat second :class:`Allotment` vector.

**Pure.** No DB, no HTTP, no ``RunContext``, no LLM — same contract as
:mod:`compute_plan`: frozen dataclasses plus free functions, deterministic and
idempotent for a given input. Its only caller, the ``reconcile_timeline`` stage,
is likewise zero-LLM: no model ever produces a second of screen time.

**The water-filling invariant.** Pinned beats (measured footage, an authored
``time_lock``) are paid first, exactly — they are never trimmed (C0). The
remaining budget floods the elastic beats: each starts at its ``min``, and the
slack is poured in proportional to ``elasticity * (ideal - min)``. A beat that
fills to its ``max`` freezes and its leftover re-floods the still-unsaturated
beats, until the slack is gone. So when the target is reachable
(``Σmin <= target <= Σmax``), every allotment respects its own ``[min, max]`` and
``|Σseconds - target| <= target * tolerance_pct/100``.

The solver NEVER fakes a fit, in either direction, and both misses are reportable:

* ``Σmin > target`` (over-demand) — every elastic beat gets its floor, the sum honestly
  overruns, and the offending ids come back flagged ``infeasible``
  (:func:`infeasible_beats`).
* ``Σmax < target`` (under-supply) — every beat saturates at its ``max`` and the sum
  lands honestly UNDER the target; no seconds are invented past a beat's ``max``. Note
  that :func:`infeasible_beats` is empty here (``Σmin <= target`` holds!), so the
  tolerance guarantee above does NOT apply — :func:`budget_drift` / :func:`is_underfull`
  are the signals that catch it.

This module never fails a run — every A22 gate is soft; the stage surfaces both misses
as ``notes`` (⇒ a ``degraded`` run).

**Byte-identical guarantee.** Nothing here is called on a run without
``direction.timing``; the reconcile stage is skip-clean, so a run that does not
opt in is unchanged by this module's existence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final, Literal

from pydantic import BaseModel, Field

from src.models.showcase_line import normalize_clip_key
from src.services.stages.shared.provenance import clamp, input_hash

# The configurable speech band (§6 ``direction.timing.wpm``). The natural rate
# matches the existing ``_WPM`` in ``evaluation/rubrics/script_craft.py``, so a
# word budget computed here agrees with the one the rubric scores against.
DEFAULT_WPM_BAND: tuple[float, float] = (130.0, 170.0)
DEFAULT_WPM_NATURAL: float = 150.0
DEFAULT_TOLERANCE_PCT: float = 2.0
DEFAULT_FREEZE_PAD_CEILING: float = 1.5
# Share of the target that freeze pad may be CREDITED for against the gross-underfill
# floor. Pad is duplicated frames, so it fills a residual — it must never be able to
# carry a run over the floor on its own. See `content_floor_credit` for the measured
# numbers (58.00s/60.00s needs 3.3%; live run 01a059a2 would need 22.0%) behind 0.10.
MAX_PAD_CREDIT_RATIO: float = 0.10
# The rate multipliers kokoro's API will ACCEPT. This is a validity limit, not a
# quality one: a request outside it is rejected with a 4xx. It says nothing about
# whether the result sounds like a person, and conflating the two is what shipped
# half-speed narration for a day (see VO_RATE_NATURAL_FLOOR).
VO_RATE_BAND: tuple[float, float] = (0.5, 2.0)
# The slowest multiplier that still sounds like narration rather than a drag.
#
# kokoro's ``speed`` is a UNIFORM time-stretch — it expands phonemes AND the
# pauses between them — so the delivered clip runs 1/rate times as long and every
# silence grows with it. Measured 2026-09-11 on the live voice (af_heart, three
# sentences, speech-active frames only), as % change from a 1.0 render:
#
#     rate   1.00   0.90   0.85   0.80   0.75   0.60   0.55   0.50
#     dur    1.00x  1.07x  1.12x  1.28x  1.39x  1.69x  1.93x  2.08x
#     HF>10k  0.0%   5.1%   1.6%   4.8%   4.0%   7.1%  12.7%  11.0%
#
# Spectrally the voice is intact far below 0.80; what degrades long before the
# spectrum notices is PACING, which scales straight off the duration column — and
# that column is flat-ish to 0.80 (1.28x) then climbs hard: 1.39x, 1.69x, 2.08x.
# So the floor comes from the duration a listener will tolerate, not from the
# spectral knee, and never from what the API happens to accept.
#
# 0.80 is the ABSOLUTE limit — the rate past which the stretch is pathological
# rather than merely audible. It is NOT the rate a delivery should be allowed to
# land on; see :data:`VO_RATE_DRAG_FLOOR`, which is the operative floor and which
# every caller in this module now defaults to. This constant survives as the
# hard backstop (and as the basis of ``ssml.EMPHASIS_MIN_VO_RATE``, which is
# derived from it and is deliberately NOT re-based on the drag floor).
VO_RATE_NATURAL_FLOOR: float = 0.80

# ⭐ THE OPERATIVE FLOOR — the slowest rate a DELIVERED line may run at, and the
# threshold the underspoken detector fires against.
#
# THE DEFECT THIS NUMBER EXISTS TO CLOSE. For twelve waves the owner's complaint
# was that narration "weirdly slows down". It was reported fixed three times, and
# each fix armed only against ``VO_RATE_NATURAL_FLOOR`` (0.80) — the point where
# the CLAMP binds. So the band ``[0.80, 1.00)`` carried no flag, no hold, no note,
# no degradation and no re-ask: a delivery up to 25% slower than natural was, to
# every signal in the pipeline, indistinguishable from a clean one. The run last
# certified as a PASS delivered 0.8933 / 0.8936 / 0.8650 — 11-13% slow, audible,
# and invisible to all of it. A floor is not a detector; this is the detector.
#
# WHY 0.97, and why not each of the alternatives:
#
# * It MUST exceed 0.8936, or the owner's own reported defect stays undetected.
#   That alone disqualifies every candidate at or below ~0.894.
# * It must exceed ``VO_RATE_NATURAL_FLOOR`` strictly. A threshold at or below the
#   clamp is ARITHMETICALLY UNFALSIFIABLE against a delivered rate, because the
#   delivered rate IS the clamp's output and can never be observed beneath it. The
#   tempting "mirror of ``_RATE_DEFECT_SPEED_THRESHOLD`` (1.3)" — its reciprocal,
#   1/1.3 = 0.769 — is exactly that trap: it sits BELOW 0.80, so it would have read
#   as a guard while being incapable of ever firing.
# * The measured stretch table above gives 1.00 -> 1.00x and 0.90 -> 1.07x of
#   delivered duration, i.e. ~7% of pacing inflation per 0.10 of rate. 0.97 is
#   therefore ~1.02x — a 2% pacing change, under the ~5% just-noticeable difference
#   for tempo, and the same order as ``narration._FIT_TOLERANCE_RATIO`` (0.05)
#   already declines to correct at all as inaudible.
# * It is NOT 1.00, and that is a cost decision rather than a quality one.
#   ``reconcile_timeline`` arms its word-expander on ``any(a.underspoken)``, so every
#   beat over the threshold buys a real LLM re-ask. At 1.00 a single surplus word of
#   rounding residual would arm it on essentially every run. 0.97 leaves a 3% dead
#   band wide enough for ordinary rounding and narrow enough that nothing a listener
#   can hear lives inside it.
#
# Configurable per run via ``direction.timing.vo_rate_drag_floor`` — threaded through
# :func:`reconcile_timeline` as ``natural_floor`` — because the band is calibrated
# against ONE voice and a different cast will want a different number.
VO_RATE_DRAG_FLOOR: float = 0.97

# ⭐ THE SLOWEST RATE A LINE IS EVER DELIVERED AT — natural speed, 1.0.
#
# ⛔ ONE CONSTANT, ONE JOB, again. ``VO_RATE_DRAG_FLOOR`` is the DETECTOR (where
# ``underspoken`` fires and a paid re-ask arms); this is the DELIVERY floor. They were
# one number, so every line with slack was spoken at 0.97 and a raw 0.97-1.0 line was
# spoken at its raw rate — slower than natural either way, which is the owner's
# "narration slows down". A line that fits its slot is now spoken at 1.0 and the
# seconds it leaves are the PICTURE's (:func:`underspoken_hold`, sized against THIS
# rate so spoken + hold == allotment). The detector keeps its 3% dead band (see the
# cost note above and ``demo_video.gen``'s word split, which is sized against it).
VO_RATE_MIN_DELIVERED: float = 1.0

# ⭐ THE RATE THE WORD BUDGET IS WRITTEN AT — and it is deliberately NOT the drag floor.
#
# ⛔ ONE CONSTANT, ONE JOB. The word budget and the underspoken detector are two
# different decisions and must not share a number:
#
#   * the BUDGET answers "how many words should the generator write for this order?"
#   * the FLOOR answers "how slow a delivery do we refuse to ship?"
#
# Reusing the floor for the budget makes a beat that delivers its budget PERFECTLY land
# exactly ON the detection boundary, where integer rounding decides the verdict. Measured
# over every ordered duration from 5s to 180s: with the budget taken at the 0.97 floor, a
# flawless delivery is flagged `underspoken` on **48.9%** of durations — a coin flip, from
# rounding alone. Taken at 1.0 it is flagged on 0.6% (only the degenerate 5s order). A
# signal that fires on half of all PERFECT runs is not a signal; this project has already
# rejected a deband ceiling on exactly that ground.
#
# So the budget is written at the natural rate and the floor sits 3% BELOW it. That gap is
# the tolerance, it lives in exactly one place, and it is what makes "delivered its
# budget" and "not flagged" the same event.
#
# WHY 1.0 IS NOT AN OVERSHOOT RISK. It is not an optimistic target — it is the definition
# of fitting: "the words this voice, as MEASURED (`resolved_wpm`), actually says in the
# ordered time." The old 0.80 was described as a deliberate asymmetry against overshoot,
# but it bought that insurance by budgeting for a voice already dragged 20% — i.e. it
# ORDERED the owner's complaint upstream. The long side is guarded where it belongs and
# not by biasing the budget: `unsayable` on the rate, `plan_beats`' ceiling term, and
# `_apply_duration_ceiling` on the spine.
NARRATION_BUDGET_RATE: float = 1.0

# A card with neither an authored ``seconds`` nor an A16 read-time floor still
# needs to be on screen long enough to register.
_CARD_FLOOR_SECONDS: Final = 1.0
# How far past its ideal a card may be held to absorb slack (the hold ceiling).
_CARD_HOLD_FACTOR: Final = 2.0
_EPS: Final = 1e-9
_COMPUTED_BY: Final = "reconcile_timeline"


@dataclass(frozen=True)
class TimeDemand:
    """One beat's declared timing demand — the solver's input, not its output.

    ``id`` is the spine id verbatim (C0 — never reordered, never re-keyed).
    ``kind`` is ``"footage"`` (a measured clip), ``"vo"`` (has narration) or
    ``"card"`` (a still/title hold). ``elasticity`` is ``0.0`` for a pinned beat
    and ``1.0`` for a fully elastic one; ``pinned`` beats are paid at ``ideal``
    exactly and never trimmed.
    """

    id: str
    kind: str
    min: float
    ideal: float
    max: float
    elasticity: float
    pinned: bool
    word_count: int = 0
    # The MEASURED clip length of a footage beat :func:`trim_footage_tails` may cut — set
    # only on a plain footage-pinned beat (not ``time_lock``, not an author-``locked``
    # line, not one carrying authored ``min_seconds``/``max_seconds``). ``None`` on every
    # other beat. Kept untouched when the tail is trimmed, so ``min/ideal/max`` (the
    # out-point) and the original clip are both on the record.
    clip_seconds: float | None = None

    @property
    def footage_out_point_seconds(self) -> float | None:
        """Where the compile cuts this footage clip's tail, or ``None`` when untrimmed.

        ⛔ FILE TIME: seconds from the HEAD of the recording (the compiler's source
        timeline, the same axis as ``clip_out_seconds``), NOT a duration measured from
        wherever the beat's window starts. The two coincide only because a beat is
        ``clip_seconds``-trimmable solely when its played window already starts at file
        time 0 and runs to the file's end (:func:`has_authored_clip_window` is False) —
        so ``window start (0) + trimmed duration`` IS the file-time out-point. A beat
        with an authored window is never trimmed here, so this never has to add a start.
        """
        if self.clip_seconds is None or self.ideal >= self.clip_seconds - _EPS:
            return None
        return self.ideal


@dataclass(frozen=True)
class Allotment:
    """One beat's reconciled second budget, provenance-tagged.

    ``source`` is one of :data:`~src.services.stages.shared.provenance.VALID_SOURCES`:
    ``"default"`` for a pinned (measured/authored) truth, ``"smart-auto"`` for a
    value the solver computed, ``"llm-clamped"`` where the computed value had to be
    pulled back inside the beat's ``[min, max]``. ``vo_rate`` is ``None`` for a beat
    with no words (footage-only, cards).

    ``underspoken`` is ``unsayable``'s missing mirror: the beat has too FEW words
    for its allotment, so fitting them would need a rate below
    :data:`VO_RATE_NATURAL_FLOOR` — a script-length fault, exactly like
    ``unsayable``, pointing the other way. It is a flag rather than a silent clamp
    because for months this direction had no name: the rate was quietly pinned to
    the band floor and the drag shipped with nothing downstream able to see it.

    ``freeze_pad_seconds`` is the ceiling-capped freeze-hold a **footage** beat is
    allowed past its measured clip (0.0 for every other beat and for footage whose
    narration already fits the bare clip). The caller stamps ``max_seconds = clip +
    freeze_pad_seconds`` so the renderer's freeze-pad is bounded at the ceiling
    instead of stretching to full audio length (freeze-pad Step B). Because the
    compiler really does hold that tail on screen, :func:`budget_drift` ADDS it to the
    beat's delivered seconds (:func:`content_drift`, the gross-floor precondition, does
    not).

    ``hold_seconds`` is the UNDERSPOKEN residual: the seconds this beat's allotment
    holds that its words cannot fill even at the natural floor. See
    :func:`underspoken_hold` for the arithmetic and for why it is a SEPARATE field
    from ``freeze_pad_seconds`` rather than folded into it.

    ⛔ ``hold_seconds`` MUST NOT be added to ``freeze_pad_seconds``, however
    similar "a held frame fills seconds the voice does not" sounds in both. The
    two pads point in OPPOSITE directions and are consumed by different code:
    ``freeze_pad_seconds`` is EXTRA WINDOW GRANTED TO THE VOICE — narrate's
    ``fit_target_for`` returns ``seconds + freeze_pad_seconds`` and re-synthesizes
    the line to fill it — so a footage beat whose narration OVERRUNS its clip is
    allowed to use the pad instead of over-compressing. ``hold_seconds`` is the
    mirror: the beat's words are too FEW, and the residual must NOT become more
    speaking time or narrate drags the delivery down to its ``_FIT_FLOOR`` (0.80)
    trying to fill it — which is precisely the "narration weirdly slows down"
    complaint this field exists to remove. Routing the residual through
    ``freeze_pad_seconds`` would therefore make the symptom WORSE, not better.

    ``uncovered_seconds`` is the part of an underspoken beat's residual that
    ``hold_seconds`` itself does NOT cover — see :func:`underspoken_uncovered`.
    Neither speech nor a held frame occupies these seconds, so ``seconds`` on
    this beat overstates what the master actually delivers by exactly this much;
    :func:`budget_drift` subtracts it rather than counting the full allotment as
    delivered. Always ``0.0`` on a FOOTAGE beat: its allotment is the clip, and the
    clip's picture plays for all of it whatever the narration does.
    """

    id: str
    seconds: float
    vo_rate: float | None
    source: str
    infeasible: bool
    unsayable: bool = False
    freeze_pad_seconds: float = 0.0
    underspoken: bool = False
    hold_seconds: float = 0.0
    uncovered_seconds: float = 0.0


@dataclass(frozen=True)
class DriftReport:
    """Compile-tail reconciliation: measured minus planned, per beat and total."""

    per_beat: dict[str, float]
    total: float
    over_tolerance: bool


def _repair(lo: float, ideal: float, hi: float) -> tuple[float, float, float]:
    """Force ``lo <= ideal <= hi`` so downstream math can never invert."""
    lo = max(lo, 0.0)
    hi = max(hi, lo)
    return lo, min(max(ideal, lo), hi), hi


def _authored_seconds(line: Mapping[str, Any]) -> float | None:
    """The beat's authored ``seconds``, or ``None`` when unset/zero."""
    raw = line.get("seconds")
    if raw is None:
        return None
    value = float(raw)
    return value if value > 0.0 else None


def has_authored_clip_window(line: Mapping[str, Any]) -> bool:
    """Does this line author WHICH part of its recording plays?

    True for an authored ``clip_out_seconds``, a positive ``clip_lead_seconds`` the
    compiler honours (under ``None`` / ``"start"`` — ``"auto"`` is recomputed by the
    compile from the motion curve, ``"end"`` ignores it), or a ``clip_anchor`` whose
    semantics this module does not know. ``"end"`` alone is NOT a window: it plays the
    last seconds BEFORE the usable end, so a file-time cut simply moves that end
    (``compile._bound_window_to_usable`` re-expresses it) and stays consistent.

    The ONE definition both halves of the footage out-point contract read:
    :func:`collect_demands` never makes such a beat tail-trimmable, and
    ``demo_video.compile`` ignores an out-point stamped on one. Its window is the
    author's, so a file-time cut from 0 would land inside it (a 10-15s window of a 20s
    recording "cut to 12" played 10-12s and freeze-held the rest).
    """

    def _positive(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0.0

    anchor = line.get("clip_anchor")
    if _positive(line.get("clip_out_seconds")):
        return True
    if anchor not in (None, "start", "end", "auto"):
        return True
    return anchor in (None, "start") and _positive(line.get("clip_lead_seconds"))


def authored_window_seconds(line: Mapping[str, Any], measured: float) -> float:
    """The footage seconds compile PLAYS for a beat, given its measured file length.

    ``measured`` unchanged unless :func:`has_authored_clip_window` — the compile then
    plays only the author's window and never trims it, so pinning the beat at the FULL
    file length overstated ``Σmin`` / ``overrun_seconds`` (a 5s window on a 20s file
    counted 20s), and on ``relief.mode == "auto"`` that phantom overrun fed the paid
    structural-relief spawn.

    The span mirrors ``video_compiler_service`` ``renderer._clip_seek_args`` over the
    fields ``demo_video.compile`` forwards (``clip_out_seconds`` / ``clip_lead_seconds``
    / ``clip_anchor``):

    * an out-point ⇒ window mode: ``min(out, measured) - lead``, the lead taken verbatim
      (``"end"`` keeps it as the floor of the window). ``"auto"`` recomputes its lead
      from the motion curve at compile time, so only the out-point is known here: the
      span is ``min(out, measured)``, an upper bound — never an under-count.
    * a lead alone (anchor ``None`` / ``"start"``) ⇒ the clip plays from the lead to its
      end: ``measured - lead``.
    * an anchor whose semantics this module does not know ⇒ ``measured`` (no guess).

    A degenerate window (``out <= lead``, which the compiler's request schema rejects)
    also falls back to ``measured``; the span is never longer than ``measured``.
    """
    if not has_authored_clip_window(line):
        return measured

    def _num(value: object) -> float:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0.0:
            return float(value)
        return 0.0

    anchor = line.get("clip_anchor")
    if anchor not in (None, "start", "end", "auto"):
        return measured
    out = _num(line.get("clip_out_seconds"))
    lead = 0.0 if anchor == "auto" else _num(line.get("clip_lead_seconds"))
    end = min(out, measured) if out > 0.0 else measured
    span = end - lead
    return span if 0.0 < span <= measured else measured


def _override(line: Mapping[str, Any], key: str, fallback: float) -> float:
    """An authored bound (C6) always wins over the computed one."""
    raw = line.get(key)
    return fallback if raw is None else float(raw)


def _kind(measured: float | None, word_count: int) -> str:
    if measured is not None:
        return "footage"
    return "vo" if word_count > 0 else "card"


def collect_demands(
    lines: Sequence[Mapping[str, Any]],
    *,
    clip_seconds: Mapping[str, float],
    wpm_band: tuple[float, float] = DEFAULT_WPM_BAND,
    wpm_natural: float = DEFAULT_WPM_NATURAL,
    card_floor_seconds: float | None = None,
    target: float | None = None,
) -> list[TimeDemand]:
    """Turn the spine into one ``TimeDemand`` per beat, in spine order.

    ``target`` (the narration BODY the order leaves after title cards) arms
    :func:`trim_footage_tails`: footage that runs past its own narration is cut back so
    the projected film fits the order. ``None`` ⇒ no trim (the pre-existing behaviour).

    ``clip_seconds`` is the *measured* footage length keyed by the clip key — the
    same join ``compile`` uses (``line["clip"]`` falling back to ``line["id"]``,
    normalized via :func:`~src.models.showcase_line.normalize_clip_key`). It may be
    empty: a beat that carries a ``clip`` tag but has **no measured length is NOT
    pinned** — we never pin a beat to a guess. It falls back to its vo/card demand.

    - **footage** (a measured length exists): ``min == ideal == max == measured``,
      ``elasticity=0``, pinned. C0: never trimmed. EXCEPT an author-``locked`` line
      with words: its verbatim text owns the beat, so it takes the **vo** band below
      (``ideal`` = the authored ``seconds`` clamped into that band) and the clip is
      frozen / trimmed to it — the text is never fitted to the clip.
    - **vo** (non-empty ``text``): ``min = words/wpm_max``, ``ideal = words/wpm_natural``,
      ``max = words/wpm_min`` (all x60), fully elastic.
    - **card** (no text, no measured clip): floors at ``card_floor_seconds`` (the A16
      read-time floor — unbuilt today, so normally ``None``), else the authored
      ``seconds``, else :data:`_CARD_FLOOR_SECONDS`; held at most x2.

    C6 authored overrides always win: ``min_seconds`` / ``max_seconds`` replace the
    computed bounds, and ``time_lock`` pins the beat at its authored (else measured)
    ``seconds`` with ``min == ideal == max``.
    """
    measured_by_key = {normalize_clip_key(str(key)): float(v) for key, v in clip_seconds.items()}
    wpm_min, wpm_max = wpm_band
    demands: list[TimeDemand] = []

    for line in lines:
        trimmable = False
        line_id = str(line["id"])
        clip_key = normalize_clip_key(str(line.get("clip") or line["id"]))
        measured = measured_by_key.get(clip_key)
        if measured is not None:
            # An authored window is what compile plays — size the beat at it, never at
            # the whole file (see ``authored_window_seconds``).
            measured = authored_window_seconds(line, measured)
        word_count = len(str(line.get("text") or "").split())
        authored = _authored_seconds(line)
        kind = _kind(measured, word_count)

        if line.get("time_lock") is True:
            # The author pinned this beat: authored seconds win, else the measured
            # clip, else nothing to pin to (0.0 — an empty lock costs no budget).
            locked = authored if authored is not None else (measured or 0.0)
            demands.append(
                TimeDemand(
                    id=line_id,
                    kind=kind,
                    min=locked,
                    ideal=locked,
                    max=locked,
                    elasticity=0.0,
                    pinned=True,
                    word_count=word_count,
                )
            )
            continue

        if measured is not None and line.get("locked") and word_count > 0:
            # An author-``locked`` line is said VERBATIM, so it is the line — not the
            # recording — that sets this beat's length: the clip is fitted to the words
            # (the compiler freezes the last frame past a short clip and trims a long
            # one), never the words to the clip. Pinning it to the measured clip instead
            # is what made the relief / underspoken passes reach for the text. The
            # authored ``seconds`` is the wish, honoured only where the verbatim words
            # are sayable in it; outside the band the speech band wins over the wish.
            kind = "vo"
            lo = word_count / wpm_max * 60.0
            hi = word_count / wpm_min * 60.0
            natural = word_count / wpm_natural * 60.0
            ideal = min(max(authored, lo), hi) if authored is not None else natural
            elasticity, pinned = 1.0, False
        elif measured is not None:
            lo = ideal = hi = measured
            elasticity, pinned = 0.0, True
            # An authored window is the author's: never tail-trimmed (see
            # ``has_authored_clip_window`` — the out-point is FILE time from 0).
            trimmable = (
                line.get("min_seconds") is None
                and line.get("max_seconds") is None
                and not has_authored_clip_window(line)
            )
        elif word_count > 0:
            lo = word_count / wpm_max * 60.0
            ideal = word_count / wpm_natural * 60.0
            hi = word_count / wpm_min * 60.0
            elasticity, pinned = 1.0, False
        else:
            floor = card_floor_seconds if card_floor_seconds is not None else authored
            lo = floor if floor is not None else _CARD_FLOOR_SECONDS
            ideal = authored if authored is not None else lo
            hi = ideal * _CARD_HOLD_FACTOR
            elasticity, pinned = 1.0, False

        lo = _override(line, "min_seconds", lo)
        hi = _override(line, "max_seconds", hi)
        lo, ideal, hi = _repair(lo, ideal, hi)
        demands.append(
            TimeDemand(
                id=line_id,
                kind=kind,
                min=lo,
                ideal=ideal,
                max=hi,
                elasticity=elasticity,
                pinned=pinned,
                word_count=word_count,
                clip_seconds=measured if trimmable else None,
            )
        )
    if target is not None:
        demands = trim_footage_tails(demands, target=target, wpm_natural=wpm_natural)
    return demands


#: A trimmed footage beat keeps at least this share of its measured clip…
FOOTAGE_TRIM_FLOOR_RATIO: Final = 0.60
#: …and never runs shorter than this (a clip already shorter is never trimmed at all).
FOOTAGE_TRIM_FLOOR_SECONDS: Final = 2.0


def narration_seconds(demand: TimeDemand, wpm_natural: float = DEFAULT_WPM_NATURAL) -> float:
    """The beat's narration at the natural rate — ``0.0`` for a beat with no words."""
    if demand.word_count <= 0 or wpm_natural <= 0.0:
        return 0.0
    return demand.word_count / wpm_natural * 60.0


def _segment_seconds(demand: TimeDemand, wpm_natural: float) -> float:
    """What one beat occupies in the delivered film: ``max(narration, picture)``.

    A footage beat plays as long as the LONGER of its clip and its narration (the
    compiler freeze-holds a short clip under a long line), so footage cut below the
    narration buys nothing. Every other beat is its ``min``: it is elastic, the solver
    fits it to whatever the order leaves, and footage must never be cut to make room for
    seconds an elastic beat would give up on its own.
    """
    if demand.kind == "footage":
        return max(narration_seconds(demand, wpm_natural), demand.ideal)
    return demand.min


def projected_segment_seconds(
    demands: Sequence[TimeDemand], wpm_natural: float = DEFAULT_WPM_NATURAL
) -> float:
    """``Σ max(narration_i, footage_i)`` — the body length the spine projects."""
    return float(sum(_segment_seconds(d, wpm_natural) for d in demands))


def footage_trim_floor(demand: TimeDemand, wpm_natural: float = DEFAULT_WPM_NATURAL) -> float:
    """The shortest a trimmable footage beat may be cut to (never longer than its clip).

    ``max(narration, 60% of the clip, 2.0s)``: below its own narration a cut shortens
    nothing (the segment is narration-bound), and the 60% / 2.0s floors keep the cut a
    TAIL trim rather than a different shot.
    """
    clip = demand.clip_seconds if demand.clip_seconds is not None else demand.ideal
    floor = max(
        narration_seconds(demand, wpm_natural),
        FOOTAGE_TRIM_FLOOR_RATIO * clip,
        FOOTAGE_TRIM_FLOOR_SECONDS,
    )
    return min(floor, clip)


def trim_footage_tails(
    demands: Sequence[TimeDemand],
    *,
    target: float,
    wpm_natural: float = DEFAULT_WPM_NATURAL,
) -> list[TimeDemand]:
    """Cut footage tails that run PAST their narration until the projection fits ``target``.

    Owner ruling: footage over the order is trimmed at the TAIL. Narration-aware (replay of
    31 recorded runs): a beat's segment is ``max(narration_i, footage_i)``, so only the
    footage OVERHANG past a beat's narration can buy seconds back. The excess
    ``projected_segment_seconds - target`` is taken from the beats with the LARGEST
    overhang first, each down to :func:`footage_trim_floor`, and trimming stops the moment
    no excess is left. When narration alone overruns the order, nothing is cut below it —
    the residual is reported (``duration_out_of_tolerance``), never hidden.

    A trimmed beat is re-pinned at its out-point (``min == ideal == max``) with
    ``clip_seconds`` kept, so :attr:`TimeDemand.footage_out_point_seconds` names the cut.
    Pure and deterministic; an untrimmed vector comes back equal to its input.
    """
    out = list(demands)
    if not math.isfinite(target) or target <= 0.0:
        return out
    excess = projected_segment_seconds(out, wpm_natural) - target
    if excess <= _EPS:
        return out
    candidates = sorted(
        (
            (
                d.ideal - footage_trim_floor(d, wpm_natural),
                d.ideal - narration_seconds(d, wpm_natural),
                i,
            )
            for i, d in enumerate(out)
            if d.kind == "footage" and d.pinned and d.clip_seconds is not None
        ),
        # Largest footage-over-narration first; spine order breaks ties (deterministic).
        key=lambda c: (-c[1], c[2]),
    )
    for room, _overhang, idx in candidates:
        if excess <= _EPS:
            break
        if room <= _EPS:
            continue
        cut = min(room, excess)
        demand = out[idx]
        point = demand.ideal - cut
        out[idx] = replace(demand, min=point, ideal=point, max=point)
        excess -= cut
    return out


def footage_cuts(demands: Sequence[TimeDemand]) -> dict[str, float]:
    """Seconds cut from each TRIMMED footage beat's tail, keyed by beat id (spine order)."""
    return {
        d.id: float(d.clip_seconds) - point
        for d in demands
        if d.clip_seconds is not None and (point := d.footage_out_point_seconds) is not None
    }


def infeasible_beats(demands: Sequence[TimeDemand], target: float) -> list[str]:
    """Beat ids that cannot be honored inside ``target`` — the soft-flag set.

    Empty while ``Σmin <= target``: a *single* beat whose floor exceeds its fair
    share is fine as long as the others absorb it, so flagging it there would be a
    false positive. Once ``Σmin > target`` the beats over their fair share are the
    ones a human (or a future ``script_doctor`` round-trip) must shorten; when none
    stands out, the whole elastic set is (collectively) the overrun.

    This detects OVER-demand only. The mirror failure — ``Σmax < target``, where the
    content cannot fill the target — leaves this list EMPTY; :func:`is_underfull` is
    the signal for that one.

    A NON-FINITE ``target`` yields the EMPTY set — no judgement — rather than flagging
    everything. ``sum(min) <= nan`` is False, so without this guard the early-out is
    skipped, ``fair_share`` is ``nan``, ``d.min > nan`` is False for every beat, and the
    function reports the WHOLE elastic set infeasible. That list drives structural relief,
    so a corrupt target would manufacture a full-timeline infeasibility claim out of an
    unreadable number.
    """
    if not demands:
        return []
    if not math.isfinite(target):
        return []
    if sum(d.min for d in demands) <= target:
        return []
    fair_share = target / len(demands)
    elastic = [d for d in demands if not d.pinned]
    over_share = [d.id for d in elastic if d.min > fair_share]
    if over_share:
        return over_share
    return [d.id for d in elastic] or [d.id for d in demands]


def tolerance_seconds(target: float, tolerance_pct: float = DEFAULT_TOLERANCE_PCT) -> float:
    """The acceptance band in seconds: ``target * tolerance_pct / 100``.

    ``0.0`` when either input is NON-FINITE — "no band", the same answer an absent target
    gets. A ``nan`` band width makes every ``< -tolerance`` comparison in
    :func:`is_underfull` False, i.e. it silently disarms the underfull signal instead of
    reporting that it could not be computed.
    """
    if not math.isfinite(target) or not math.isfinite(tolerance_pct):
        return 0.0
    return abs(target) * tolerance_pct / 100.0


def content_drift(allotments: Sequence[Allotment], target: float) -> float:
    """``Σ(allotment - uncovered) - target``: what the CONTENT fills, freeze pad EXCLUDED.

    The pad-blind half of :func:`budget_drift`. It is NOT a measure of the delivered
    film (a footage beat's frozen tail plays too — see :func:`budget_drift`); it is the
    precondition of ``reconcile_timeline``'s gross-underfill HARD floor, which must stay
    pad-blind for the reason :func:`content_floor_credit` states: pad is held frames, so
    it may never, by itself, take a run out of reach of that floor. Gating the floor on
    :func:`budget_drift` instead would let a short order made of a few sub-second clips,
    each padded to the 1.50s ceiling, read as "not underfull" and skip the floor
    entirely — the 01a059a2 blank-master shape, multiplied by beats.
    """
    covered = sum(a.seconds - a.uncovered_seconds for a in allotments)
    return float(covered) - float(target)


def budget_drift(allotments: Sequence[Allotment], target: float) -> float:
    """``Σ(allotment + freeze_pad - uncovered) - target``: what the compiler DELIVERS
    against the target. NEGATIVE means the timeline is under-supplied.

    The counterpart of :func:`infeasible_beats`, which only sees the ``Σmin > target``
    overrun. The opposite failure — ``Σmax < target``, where the content cannot fill the
    target even at every beat's maximum hold — is invisible to it: ``Σmin <= target``
    holds, so it (correctly) reports "feasible", while :func:`_water_fill` saturates
    every beat at its ``max`` and the sum lands far under the target. The solver must NOT
    invent seconds past a beat's ``max``, so the shortfall is REAL — it has to be
    reported, and this is the signal that reports it (see :func:`is_underfull`).

    ``a.seconds`` alone misstates what a beat delivers, in BOTH directions:

    * it OVERSTATES an underspoken beat: the residual beyond the capped freeze-hold
      (``a.uncovered_seconds`` — see :func:`underspoken_uncovered`) is occupied by
      neither speech nor a held frame, so it is subtracted. Without this, a master short
      by exactly Σ``uncovered_seconds`` never registered as underfull.
    * it UNDERSTATES a footage beat whose narration runs past its clip: the caller
      stamps ``max_seconds = clip + freeze_pad_seconds`` and the compiler freeze-holds the
      last frame for that tail, so the beat occupies ``seconds + freeze_pad_seconds``
      (per beat this is :func:`_segment_seconds`' ``max(narration, clip)``, bounded at
      the pad ceiling). Showcase run ``01a15ec2-8b5f`` (2026-10-21): the gate summed
      9.48 + 10.36 + 9.44 = 29.28s and stamped ``timeline_underfull`` +
      ``timeline_off_target`` on a 30s order, while the compiler rendered the beats at
      9.50 / 11.13 / 9.70 = **30.33s** — clip-4's 0.90s and clip-5's 0.21s pads were
      delivered film the sum never counted.

    The two corrections never overlap: ``uncovered_seconds`` is always 0.0 on a footage
    beat and ``freeze_pad_seconds`` is always 0.0 on every other beat. The gross-underfill
    HARD floor deliberately does NOT read this — see :func:`content_drift`.
    """
    pad = sum(a.freeze_pad_seconds for a in allotments)
    return content_drift(allotments, target) + float(pad)


def is_underfull(
    allotments: Sequence[Allotment],
    target: float,
    tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
) -> bool:
    """True when the timeline falls SHORT of ``target`` beyond the tolerance band."""
    return budget_drift(allotments, target) < -tolerance_seconds(target, tolerance_pct)


def _water_fill(
    elastic: Sequence[TimeDemand], slack: float, seconds: dict[str, float]
) -> dict[str, float]:
    """Pour ``slack`` over beats sitting at their floor; returns the *unclamped* wish.

    ``seconds`` is mutated in place to the capped allotment. The returned dict is
    what each beat *wanted* before its ``max`` cut it off, so the caller can hand it
    to :func:`~src.services.stages.shared.provenance.clamp` and get an honest
    ``"llm-clamped"`` provenance on exactly the beats that were actually clamped.
    """
    desired = dict(seconds)
    if slack <= _EPS:
        return desired
    # Each pass either exhausts the slack or saturates at least one beat, so the
    # number of beats + 1 bounds the iteration.
    for _ in range(len(elastic) + 1):
        active = [d for d in elastic if seconds[d.id] < d.max - _EPS]
        if not active or slack <= _EPS:
            break
        weights = {d.id: max(d.elasticity * (d.ideal - d.min), 0.0) for d in active}
        total_weight = sum(weights.values())
        if total_weight <= _EPS:
            # No beat declares any elastic room → share the slack equally.
            weights = dict.fromkeys(weights, 1.0)
            total_weight = float(len(active))
        handed = 0.0
        for demand in active:
            share = slack * weights[demand.id] / total_weight
            desired[demand.id] += share
            given = min(share, demand.max - seconds[demand.id])
            seconds[demand.id] += given
            handed += given
        slack -= handed
        if handed <= _EPS:
            break
    return desired


def reconcile_timeline(
    demands: Sequence[TimeDemand],
    *,
    target: float,
    tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
    wpm_natural: float = DEFAULT_WPM_NATURAL,
    freeze_pad_ceiling_seconds: float = DEFAULT_FREEZE_PAD_CEILING,
    natural_floor: float = VO_RATE_DRAG_FLOOR,
) -> list[Allotment]:
    """Reconcile the demand vector into one per-beat second allotment.

    Pinned beats are paid at ``ideal`` exactly (``source="default"``) and never
    trimmed. The rest is water-filled (see the module docstring). Every beat with
    words also gets its clamped :func:`vo_rate_for` multiplier so narration is fit
    *forward* into its allotment instead of overrunning and being freeze-padded back.
    A **footage** beat is the one exception: its ``vo_rate`` targets ``clip_dur +
    freeze_pad_ceiling`` (the extended window the renderer may freeze-hold into)
    rather than the bare clip length, so its narration breathes into the pad instead
    of over-compressing. The ``seconds`` allotment is unchanged — only the rate moves.

    When ``Σmin > target`` the timeline is infeasible: every elastic beat still gets
    its floor (never below — acceptance #2), the sum honestly overruns the target,
    and :func:`infeasible_beats` ids come back flagged. Nothing raises.

    ``tolerance_pct`` is the caller's acceptance band; the solver saturates the slack
    exactly, so it is carried for API symmetry and asserted by the caller/tests
    rather than consumed here.

    ``natural_floor`` is the operative delivery floor — :data:`VO_RATE_DRAG_FLOOR`
    by default, overridable per run from ``direction.timing.vo_rate_drag_floor``. It
    is threaded to ALL THREE of :func:`vo_rate_for`, :func:`underspoken_hold` and
    :func:`underspoken_uncovered` from this one parameter, never re-defaulted at the
    call sites, because a flag taken against one floor and a residual quantified
    against another is how the note and the solver drifted apart before.
    """
    flagged = set(infeasible_beats(demands, target))
    elastic = [d for d in demands if not d.pinned]
    remaining = target - sum(d.ideal for d in demands if d.pinned)
    seconds: dict[str, float] = {d.id: d.min for d in elastic}
    sources: dict[str, str] = {}

    if sum(d.min for d in elastic) > remaining + _EPS:
        # Infeasible: floors only. The overrun is real and is reported, not hidden.
        sources = {d.id: "smart-auto" for d in elastic}
    else:
        wished = _water_fill(elastic, remaining - sum(d.min for d in elastic), seconds)
        for demand in elastic:
            resolved = clamp(wished[demand.id], demand.min, demand.max, computed_by=_COMPUTED_BY)
            sources[demand.id] = resolved.source
            seconds[demand.id] = float(resolved.value)

    allotments: list[Allotment] = []
    for demand in demands:
        allotted = demand.ideal if demand.pinned else seconds[demand.id]
        rate: float | None = None
        unsayable = False
        underspoken = False
        freeze_pad = 0.0
        hold = 0.0
        uncovered = 0.0
        if demand.word_count > 0:
            rate_target = allotted
            if demand.kind == "footage":
                # Footage clips may freeze-hold up to `freeze_pad_ceiling_seconds`
                # past their measured length; let narration use that extended
                # window instead of over-compressing into the bare clip. The same
                # capped pad is carried onto the allotment so the caller can bound
                # the renderer's freeze-pad at the ceiling (Step B) rather than let
                # it stretch to the full audio length.
                natural = demand.word_count / wpm_natural * 60.0
                freeze_pad = freeze_pad_ceiling(allotted, natural, freeze_pad_ceiling_seconds)
                rate_target = allotted + freeze_pad
            rate, unsayable, underspoken = vo_rate_for(
                rate_target, demand.word_count, wpm_natural, natural_floor=natural_floor
            )
            # The residual the clamped rate leaves unfilled, QUANTIFIED (see
            # `underspoken_hold`). Taken against the SAME `rate_target` the flag was
            # taken against — allotment plus any capped freeze-pad — so the hold and
            # the flag cannot drift apart the way the note and the solver once did.
            # 0.0 on every beat that is not underspoken, so this is a no-op there.
            hold = underspoken_hold(
                rate_target,
                demand.word_count,
                wpm_natural,
                freeze_pad_ceiling_seconds,
                natural_floor=natural_floor,
            )
            # The part of the SAME residual the capped hold above does not cover — see
            # `underspoken_uncovered`. Taken against the SAME `rate_target`/word_count
            # for the same reason: it must not drift from `hold`.
            # ⛔ NOT on a footage beat. Its allotment IS the measured (or tail-trimmed)
            # clip, and that moving picture plays for every allotted second whatever the
            # narration does — nothing is "covered by neither speech nor a held frame".
            # Counting it made `budget_drift` subtract seconds the film really delivers:
            # round-5 P1 (3 footage beats, 29.70s on a 30s order, delivered 29.77s) was
            # stamped `timeline_underfull` "fills only 25.97s" + `timeline_off_target`,
            # P2 (24.00s of a 24s body) "20.87s" — 3.73s / 3.13s of footage-covered
            # narration slack read as missing film.
            if demand.kind != "footage":
                uncovered = underspoken_uncovered(
                    rate_target,
                    demand.word_count,
                    wpm_natural,
                    freeze_pad_ceiling_seconds,
                    natural_floor=natural_floor,
                )
        allotments.append(
            Allotment(
                id=demand.id,
                seconds=allotted,
                vo_rate=rate,
                source="default" if demand.pinned else sources[demand.id],
                infeasible=demand.id in flagged,
                unsayable=unsayable,
                freeze_pad_seconds=freeze_pad,
                underspoken=underspoken,
                hold_seconds=hold,
                uncovered_seconds=uncovered,
            )
        )
    return allotments


def vo_rate_for(
    allotment_seconds: float,
    word_count: int,
    wpm_natural: float = DEFAULT_WPM_NATURAL,
    band: tuple[float, float] = VO_RATE_BAND,
    natural_floor: float = VO_RATE_DRAG_FLOOR,
) -> tuple[float, bool, bool]:
    """The clamped delivery-rate multiplier for a beat, plus its two fault flags.

    The narrator must say ``word_count`` words in ``allotment_seconds``; at the
    natural rate that takes ``word_count / wpm_natural * 60`` seconds, so the
    multiplier is ``natural / allotted`` — a *smaller* allotment means speaking
    *faster* (rate > 1). ``unsayable`` is True when the **unclamped** rate exceeds
    the band's top, i.e. even at max speed the words overrun the beat — route it to
    trim/compress, never clip the audio.

    ``underspoken`` is its mirror, and the reason this function returns three values
    instead of two: the rate needed falls BELOW ``natural_floor``, i.e. the beat has
    too few words for its seconds and the only way to fill them is to drag the voice.
    Both are script-length faults. Only one of them used to have a name — the rate
    was clamped to the band floor, returned as though nothing were wrong, and the
    2.08x-stretched delivery shipped with no flag for any caller to act on.

    The returned rate is floored at :data:`VO_RATE_MIN_DELIVERED` (1.0): a line that
    fits its slot is spoken at natural speed, never slowed. ``natural_floor`` is ONLY
    the ``underspoken`` detector threshold (a paid re-ask arms on it). Surplus seconds
    are the caller's problem to solve honestly (hold the frame, shorten the beat, or
    write more script) — never the voice's.

    Edge cases, kept divide-by-zero safe and HONEST:

    * ``word_count == 0`` ⇒ neutral: nothing to say, so no fault in either direction.
    * ``word_count > 0`` with a ZERO (or negative) allotment ⇒ ``unsayable``.
      Words with no seconds to speak them in is the *worst* case of the very thing
      ``unsayable`` exists to catch; early-returning a neutral rate would swallow it.
    * ``word_count > 0`` with a NON-FINITE allotment or natural rate ⇒ the same branch,
      for the same reason: there are no usable seconds to speak in. Tested separately
      because ``nan <= 0.0`` is False — without it the function returns ``(nan, False,
      False)``, a rate no synthesizer can accept, carrying neither fault flag, i.e. the
      corruption ships silently as "fine".
    """
    _, hi = band
    floor = max(natural_floor, band[0])
    if word_count <= 0:
        return 1.0, False, False
    if (
        not math.isfinite(allotment_seconds)
        or not math.isfinite(wpm_natural)
        or allotment_seconds <= 0.0
        or wpm_natural <= 0.0
    ):
        return hi, True, False
    natural_seconds = word_count / wpm_natural * 60.0
    raw_rate = natural_seconds / allotment_seconds
    # The detector fires at `floor`; the voice is never delivered below natural. A
    # configured `natural_floor` (e.g. `direction.timing.vo_rate_drag_floor`) therefore
    # moves only the flag — it can no longer order a slowed delivery.
    delivered = max(raw_rate, VO_RATE_MIN_DELIVERED)
    return min(delivered, hi), raw_rate > hi, raw_rate < floor


def _underspoken_residual(
    allotment_seconds: float,
    word_count: int,
    wpm_natural: float,
    natural_floor: float,
    band: tuple[float, float],
) -> float:
    """The UNCAPPED gap an underspoken beat's allotment holds that its words
    cannot fill, before either :func:`underspoken_hold` or
    :func:`underspoken_uncovered` splits it into "held" vs. "left uncovered".

    Shared arithmetic so the two callers can never drift apart the way the note
    and the solver once did: ``allotment_seconds - natural_seconds / VO_RATE_MIN_DELIVERED``, the
    same divide-by-zero-safe edge cases as :func:`vo_rate_for` (no words, no
    usable seconds, a non-finite input ⇒ ``0.0``), never negative.
    """
    floor = max(natural_floor, band[0])
    if word_count <= 0 or floor <= 0.0:
        return 0.0
    if (
        not math.isfinite(allotment_seconds)
        or not math.isfinite(wpm_natural)
        or allotment_seconds <= 0.0
        or wpm_natural <= 0.0
    ):
        return 0.0
    natural_seconds = word_count / wpm_natural * 60.0
    # Spoken at the DELIVERED rate (never below natural — see `vo_rate_for`), not at the
    # detector floor: sizing it at 0.97 left ~3% of every slack beat covered by nothing.
    spoken_seconds = natural_seconds / max(VO_RATE_MIN_DELIVERED, band[0])
    return max(allotment_seconds - spoken_seconds, 0.0)


def underspoken_hold(
    allotment_seconds: float,
    word_count: int,
    wpm_natural: float = DEFAULT_WPM_NATURAL,
    ceiling: float = DEFAULT_FREEZE_PAD_CEILING,
    natural_floor: float = VO_RATE_DRAG_FLOOR,
    band: tuple[float, float] = VO_RATE_BAND,
) -> float:
    """Seconds an UNDERSPOKEN beat's allotment holds that its words cannot fill.

    The residual :func:`vo_rate_for`'s ``underspoken`` flag names but does not quantify
    (see :func:`_underspoken_residual`). A beat flagged ``underspoken`` has
    ``raw_rate < floor``, so its rate clamps to ``floor`` and the words are delivered in
    ``natural_seconds / floor`` seconds — which BY CONSTRUCTION is still shorter than
    ``allotment_seconds``.

    WHY IT IS A SEPARATE NUMBER FROM THE RATE. The rate lever is already exhausted at
    the moment this is non-zero — that is what the clamp means — so there is no rate
    that can absorb the gap. It has to be spent as something OTHER than speech, and the
    only honest options are a held frame (this), more words (a re-ask), or a shorter
    beat. The one option that is NOT honest is the one the pipeline took by default:
    hand the residual back to the voice, which drags delivery to the floor and STILL
    ships the beat short.

    Capped at ``ceiling`` (the same per-beat freeze-pad ceiling, 1.5s by default) for
    the reason the freeze pad is capped: a held frame is duplicated frames, not content,
    and an uncapped hold would let one beat sit dead for many seconds. The residual
    BEYOND the ceiling is deliberately left uncovered rather than papered over — see
    :func:`underspoken_uncovered`, which quantifies exactly that leftover so
    :func:`budget_drift` can count it as NOT delivered instead of silently trusting a
    frozen frame to cover seconds nothing actually holds.

    Neutral (``0.0``) on every beat this cannot apply to, and for the same
    divide-by-zero-safe reasons :func:`vo_rate_for` early-returns: no words, no usable
    seconds, or a non-finite input. A beat with no slack at natural speed returns 0.0
    naturally, so the caller may apply it unconditionally. ⚠️ A beat inside the
    detector's dead band (raw rate in ``[natural_floor, 1.0)``) is NOT flagged
    ``underspoken`` yet DOES get a small hold: it is spoken at 1.0, never slowed, and
    the picture holds the few percent the voice leaves.
    """
    residual = _underspoken_residual(
        allotment_seconds, word_count, wpm_natural, natural_floor, band
    )
    return min(residual, ceiling)


def underspoken_uncovered(
    allotment_seconds: float,
    word_count: int,
    wpm_natural: float = DEFAULT_WPM_NATURAL,
    ceiling: float = DEFAULT_FREEZE_PAD_CEILING,
    natural_floor: float = VO_RATE_DRAG_FLOOR,
    band: tuple[float, float] = VO_RATE_BAND,
) -> float:
    """Seconds of an underspoken beat's residual that NEITHER the voice NOR the
    capped freeze-hold covers — ``residual - underspoken_hold(...)``.

    :func:`underspoken_hold` caps its answer at ``ceiling`` on purpose (an
    uncapped hold would let one beat sit dead for many seconds); this is the
    part that cap leaves out. That part is not delivered by ANYTHING — not
    speech (the rate lever is already exhausted), not a held frame (past the
    ceiling) — so a beat whose allotment includes it does not actually occupy
    its full ``allotment_seconds`` on screen. :func:`budget_drift` would otherwise
    count every allotted second as delivered; without subtracting this, a master short by exactly this amount, summed over every beat, never
    registers as underfull (ORCH item 1d). ``0.0`` whenever the residual does
    not exceed the ceiling — i.e. on every beat the hold fully covers, and on
    every beat that is not underspoken at all.
    """
    residual = _underspoken_residual(
        allotment_seconds, word_count, wpm_natural, natural_floor, band
    )
    return max(residual - ceiling, 0.0)


def freeze_pad_ceiling(
    clip_dur: float,
    allotment_seconds: float,
    ceiling: float = DEFAULT_FREEZE_PAD_CEILING,
) -> float:
    """Seconds a short footage clip may freeze-hold to fill its allotment.

    Caps the renderer's today-uncapped ``_freeze_pad_suffix``: the gap the clip has
    to cover, but never more than ``ceiling``. A clip that cannot fill its allotment
    within the ceiling is flagged (extend with B-roll / trim VO), never silently
    stretched.
    """
    return min(max(allotment_seconds - clip_dur, 0.0), ceiling)


def content_floor_credit(
    bare_seconds: float,
    pad_seconds: float,
    target: float,
    pad_credit_ratio: float = MAX_PAD_CREDIT_RATIO,
) -> float:
    """Seconds a timeline may claim against the gross-underfill floor.

    THE INVARIANT: **freeze pad may never, by itself, lift a run over the content
    floor.** Pad is duplicated frames — a held still — so it is a *residual* filler,
    not content. Credit it, but cap the credit at ``pad_credit_ratio`` x target, a
    fraction chosen to sit well BELOW the floor itself.

    WHY A CAP AND NOT A DELETION OF THE CREDIT. The credit exists for a real reason
    and deleting it regresses that: adversarial review measured a 20-beat film at
    **28.00s bare / 58.00s held** against a **60.00s** target — 46.7% bare, i.e. under
    a 50% bare-only floor — and a bare-sum gate would have DESTROYED a 58-second film
    for a 60-second order. That film needs only **3.3 points** of target in pad credit
    to clear the floor.

    WHY A CAP AT ALL, WITH THE MEASURED NUMBERS. :func:`freeze_pad_ceiling` caps pad at
    ``DEFAULT_FREEZE_PAD_CEILING`` = 1.50s **per beat**, so on a SHORT target a single
    beat's pad is a huge share of the order and the gate could not fail at all. Live run
    ``01a059a2-8043-748a-820d-0395db4157b0`` (2026-08-31, landed ``completed``): ordered
    target **5.00s**, one beat ``clip-1`` with ``seconds=1.40`` and ``max_seconds=2.90``
    ⇒ ``freeze_pad_seconds=1.50`` (its ceiling). Bare fill **28%** — far under the 50%
    floor — but ``1.40 + 1.50 = 2.90`` scored **58%** and PASSED. What shipped was a
    10.77s master with ``luma_spread=0.0`` (blank), ``duplicate_frame_ratio=0.6429``,
    ``dead_frame_ratio=0.119``, ``gate_pass=false``. 1.50s is **30% of a 5s target**, so
    *any* single-clip run cleared a 50% floor on padding alone: the gate was structurally
    incapable of failing a short order.

    WHY 0.10. The two measured cases bracket it: the legitimate 58s/60s film needs
    **3.3%** of target in pad, the collapsed 1.4s/5s run would need **22.0%** to be
    rescued. 0.10 leaves the good case ~3x headroom and puts the bad case ~2.2x out of
    reach. Stated as a property rather than a constant: with the default 0.5 ratio the
    gate's effective floor becomes **bare content >= 40% of target**, and pad can never
    supply more than **10% of target** no matter how many beats carry it.

    Both the gate (:func:`~...reconcile_timeline._underfull_failure`) and the harvest-time
    predictor (:func:`~...record._short_clip_notes`) call THIS function, so the two can
    no longer compute the same quantity by different formulas — which is exactly how the
    01a059a2 mismatch shipped: the predictor said "this run cannot satisfy
    reconcile_timeline" and reconcile_timeline then disagreed with its own predictor.
    """
    if target <= 0.0:
        return float(bare_seconds)
    credit = min(max(float(pad_seconds), 0.0), max(pad_credit_ratio, 0.0) * float(target))
    return float(bare_seconds) + credit


def max_content_floor_credit(
    bare_seconds: float, target: float, pad_credit_ratio: float = MAX_PAD_CREDIT_RATIO
) -> float:
    """The BEST case for ``bare_seconds``: every second of pad the cap will ever allow.

    The harvest-time predictor has no allotments yet — no pad has been computed. It can
    still make a *certain* claim by assuming the most generous pad the gate could
    possibly credit, which is exactly the cap. If even that lands under the floor, the
    run provably cannot satisfy :func:`content_floor_credit` later.
    """
    return content_floor_credit(
        bare_seconds, max(pad_credit_ratio, 0.0) * float(target), target, pad_credit_ratio
    )


def drift_report(
    planned: Mapping[str, float],
    measured: Mapping[str, float],
    tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
) -> DriftReport:
    """Compile-tail drift: what the render actually produced minus what we planned.

    ``per_beat`` covers the ids present on both sides (a beat we never measured is
    simply absent). ``total`` is the whole-film drift; ``over_tolerance`` drives the
    **soft** ``rerun_stages[]`` — it never fails the run.
    """
    per_beat = {
        beat_id: float(measured[beat_id]) - float(seconds)
        for beat_id, seconds in planned.items()
        if beat_id in measured
    }
    planned_total = float(sum(planned.values()))
    total = float(sum(measured.values())) - planned_total
    return DriftReport(
        per_beat=per_beat,
        total=total,
        over_tolerance=abs(total) > planned_total * tolerance_pct / 100.0,
    )


def timeline_manifest(
    allotments: Sequence[Allotment], demands: Sequence[TimeDemand], *, target: float
) -> dict[str, Any]:
    """The persisted provenance record for the reconciled timeline (A18/DoD §J).

    Content-addressed by ``(target, demand-vector + elasticity weights)`` — the solver's
    ACTUAL inputs — so a resume with an unchanged input resolves to the cached timeline
    with no re-solve (C13 idempotence-on-resume), and a resume with a CHANGED target
    misses the cache.

    ``target`` is hashed explicitly, not inferred from the resolved total: whenever the
    solver saturates at ``Σmax`` the total equals ``Σmax`` *regardless* of the target, so
    a total-only hash makes ``target=10`` and ``target=1000`` collide — and a resume with
    a raised target would serve a STALE timeline that fills a fraction of the film.
    """
    demand_by_id = {d.id: d for d in demands}
    beats: list[dict[str, Any]] = []
    for allotment in allotments:
        demand = demand_by_id.get(allotment.id)
        beats.append(
            {
                "id": allotment.id,
                "planned": allotment.seconds,
                "demand_min": demand.min if demand else allotment.seconds,
                "demand_max": demand.max if demand else allotment.seconds,
                "allotment": allotment.seconds,
                "source": allotment.source,
                "infeasible": allotment.infeasible,
            }
        )
    total = float(sum(a.seconds for a in allotments))
    vector = [[d.id, d.min, d.ideal, d.max, d.elasticity, d.pinned, d.word_count] for d in demands]
    return {
        "beats": beats,
        "total": total,
        "target": float(target),
        "infeasible_beats": [a.id for a in allotments if a.infeasible],
        "drift": round(budget_drift(allotments, target), 6),
        "input_hash": input_hash(round(float(target), 6), vector),
    }


class TimelineManifestBeat(BaseModel):
    """One beat's line in the C13 relief manifest — the allotment joined to its bounds.

    Typed twin of the per-beat dict :func:`build_timeline_manifest` emits (same keys,
    same meaning): the reconciled ``seconds``, the demand's ``[min_seconds, max_seconds]``
    bounds (0.0 when no matching demand), the clamped ``vo_rate`` (``None`` for a
    footage/card beat with no words), whether the beat is ``feasible`` (its id is NOT
    flagged infeasible), and ``unsayable`` (its words overrun even at max delivery rate).

    ⛔ ``vo_rate`` HERE IS NOT NECESSARILY WHAT NARRATE SPEAKS TO. This row is a SNAPSHOT
    taken when the manifest was built — inside ``reconcile_timeline._relief_loop``, which
    runs BEFORE the UNDERSPOKEN re-ask (``_expand_underspoken``). The re-ask lengthens
    narration and RE-SOLVES, so a beat it fixed keeps its pre-re-ask floor-clamped
    ``0.80`` here while the spine row ``lines[].vo_rate`` — the ONLY rate narrate reads
    (``demo_video/narrate.py`` ``speed_for``) — carries the post-re-ask value. Measured on
    live run ``01a0a582-2531-7145-ba7d-a1295e92b232``: ``0.80`` on three beats here,
    ``1.0383/1.0/1.0`` on the same three in ``lines[]``.

    ``run.beats_superseded_by_underspoken_reask`` names exactly the beats whose rows here
    the re-ask invalidated, so the disagreement can never again be read as a live clamp.

    UPDATE (run ``01a0b007``): on the ``reconcile_timeline`` STAGE OUTPUT, ``vo_rate`` and
    ``unsayable`` are now refreshed from the final allotments
    (``reconcile_timeline._refresh_beat_delivery``) and agree with ``lines[]``;
    ``seconds`` / bounds / ``feasible`` remain the build-time snapshot. The ``notes[]``
    JSON string is still the unrefreshed build-time record.
    """

    id: str
    seconds: float
    min_seconds: float
    max_seconds: float
    vo_rate: float | None = None
    feasible: bool
    unsayable: bool
    # THE COMPILE CONTRACT for a tail-trimmed footage beat (``trim_footage_tails``): the
    # clip is cut at this FILE-TIME out-point — seconds from the recording's head, the
    # same axis as ``clip_out_seconds``. Only a beat whose window starts at file time 0
    # (no authored clip_out / lead / non-start anchor — ``has_authored_clip_window``) is
    # ever trimmed, so window start + trimmed duration == this value. ``None`` ⇒ the clip
    # plays untrimmed. Refreshed from the FINAL demands by ``reconcile_timeline`` —
    # unlike ``seconds``, never a snapshot.
    footage_out_point_seconds: float | None = None


class TimelineManifestRun(BaseModel):
    """Run-level convergence telemetry for the C13 relief loop (spec 23 §7.7).

    ``loop_iterations`` is the loop-engine ``passes`` count (>=1 whenever the relief loop
    ran); ``converged`` is True when the retained draft met the relief score threshold;
    ``infeasible_beat_ids`` are the beats still over-demand after the re-solve;
    ``drift_seconds`` is :func:`budget_drift` — delivered seconds (freeze pad included,
    uncovered excluded) minus the target; read by no spawn decision.

    Convergence classification + structural telemetry (the honest, machine-assertable
    signal an e2e / a future auto-convergence consumer reads):

    * ``convergence_outcome`` — ``"converged"`` (feasible after the re-solve),
      ``"cant_converge"`` (still infeasible AND every over-budget beat is protected —
      claim-bearing / locked / footage-pinned — so no in-place op can help; the run
      DEGRADES honestly, never loops), or ``"progress"`` (still infeasible but ≥1 beat
      is shortenable — a bounded rerun could yet converge), or ``"relief_failed"`` (the
      relief loop RAISED — WM/LLM unavailable — so no classification was ever computed).
      ``"relief_failed"`` exists so a crashed relief run still emits a manifest and cannot
      be mistaken for a non-relief run (a ``None`` manifest is read downstream as "relief
      never ran"). It is safe by construction: ``structural_relief_policy`` maps every
      non-``cant_converge`` outcome to ``STOP_CANT_CONVERGE``, so it never buys a paid rerun.
      ``"relief_off"`` — the relief loop never RAN (relief not opted in): the manifest
      exists only to carry footage out-points to the compile
      (``reconcile_timeline._stamp_footage_out_points``). The policy maps it to
      ``STOP_RELIEF_OFF``; it must never read as a relief run's ``cant_converge``.
      ``"relief_not_run"`` — relief WAS armed (``"auto"``) but the loop never RAN (the
      timeline fit and nothing was unsayable), and the FINAL timeline is out of band (the
      post-loop underspoken re-ask lengthened it). The policy maps it to
      ``STOP_CANT_CONVERGE``: a loop that never ran can never buy a paid rerun.
    * ``structural_relief_requested`` — the loop asked for a ``demo_video_gen`` re-run
      (an infeasible beat could not be shortened in place).
    * ``overrun_seconds`` — ``max(0, Σmin - target)`` after the re-solve (0 when feasible).
    * ``protected_beat_count`` — how many beats are PROTECTED (footage-pinned / claim-bearing /
      author-locked / ``time_lock``) and therefore un-cuttable by any in-place op. This is the
      irreducible floor on the spine's beat count: a downstream structural-relief re-run can
      never target fewer beats than this without dropping a protected beat, so the auto-converge
      hook clamps its ``target_beats`` to it (default 0 ⇒ backward-compatible: an old manifest
      with no field reads as "no known floor").
    * ``structural_rejected_ids`` / ``structural_reallocated_ids`` / ``structural_patches_applied``
      — provenance of the C13 structural pre-pass (which ops were rejected, which beats were
      re-pinned, and how many beats the pre-pass actually cut/merged/pinned).
    * ``beats_superseded_by_underspoken_reask`` — THE STALENESS MARKER, and the one field
      here that is NOT emitted by :func:`build_timeline_manifest`. The manifest is built
      inside ``reconcile_timeline._relief_loop``; the UNDERSPOKEN re-ask
      (``_expand_underspoken``) runs AFTERWARDS, lengthens narration and RE-SOLVES, so
      every ``beats[]`` row it touched is a PRE-re-ask snapshot from that moment on. The
      value cannot be known at build time, so ``reconcile_timeline`` stamps it on the
      model right after the re-ask returns (``_mark_superseded_beats``) and the field
      exists on the model only. Empty ⇒ no re-ask changed anything ⇒ every ``beats[]``
      row still agrees with the spine. Non-empty ⇒ for those ids read
      ``lines[].vo_rate`` / ``lines[].seconds``, never ``beats[]``.

      ⚠️ It marks the ``beats[]`` rows; it does NOT refresh them, so for a named id read
      ``lines[].vo_rate`` / ``lines[].seconds``.

      ⚠️ The RUN-level numbers are a split, and the split is the point. Every field a
      spawn decision reads IS refreshed post-re-ask by
      ``reconcile_timeline._refresh_run_budget``: ``overrun_seconds``,
      ``infeasible_beat_ids`` and ``convergence_outcome`` (except a ``relief_failed``
      outcome, kept verbatim so a crashed relief run can never become a spawn candidate),
      because ``runs/structural_relief.py`` decides and sizes a PAID child render from
      them and the re-ask + rate re-measure move demands in either direction. Still
      pre-re-ask: ``drift_seconds`` (a function of ``allotments``, read by no spawn
      decision).
    """

    target_seconds: float
    loop_iterations: int
    converged: bool
    pacing_score: float | None = None
    infeasible_beat_ids: list[str]
    drift_seconds: float
    convergence_outcome: Literal[
        "converged",
        "cant_converge",
        "progress",
        "relief_failed",
        "relief_off",
        "relief_not_run",
    ] = "converged"
    structural_relief_requested: bool = False
    overrun_seconds: float = 0.0
    protected_beat_count: int = 0
    structural_rejected_ids: list[str] = Field(default_factory=list)
    structural_reallocated_ids: list[str] = Field(default_factory=list)
    structural_patches_applied: int = 0
    # Title-card seconds reserved INSIDE the order. ``target_seconds`` is the BODY
    # (order minus cards); the delivered film carries the cards. Anything comparing a
    # delivered film against ``target_seconds`` (``structural_relief_policy``) must
    # take these off the delivery first, or a carded on-length run reads as over.
    card_seconds: float = 0.0
    beats_superseded_by_underspoken_reask: list[str] = Field(default_factory=list)


class TimelineManifest(BaseModel):
    """Typed, live-assertable twin of :func:`build_timeline_manifest`'s dict.

    The dict is still what rides the ``notes[]`` string (``json.dumps``-serialized for
    humans / log greppers); this model is the STRUCTURED field the stage output carries
    so ``scripts/e2e_validate.py`` can assert on ``run.loop_iterations`` / ``run.converged``
    by dotted path — a string note is unparseable, a typed field is not.

    NOT quite key-identical any more: the model carries one field the dict does not,
    ``run.beats_superseded_by_underspoken_reask``, because its value is only known AFTER
    the manifest is built (see that field's docs). The ``notes[]`` string is therefore the
    build-time record; this model is the one that says which of its rows were later
    superseded.
    """

    beats: list[TimelineManifestBeat]
    run: TimelineManifestRun


def build_timeline_manifest(
    demands: list[TimeDemand],
    allotments: list[Allotment],
    *,
    target_seconds: float,
    loop_iterations: int,
    converged: bool,
    pacing_score: float | None,
    convergence_outcome: Literal[
        "converged",
        "cant_converge",
        "progress",
        "relief_failed",
        "relief_off",
        "relief_not_run",
    ] = "converged",
    structural_relief_requested: bool = False,
    overrun_seconds: float = 0.0,
    protected_beat_count: int = 0,
    structural_rejected_ids: list[str] | None = None,
    structural_reallocated_ids: list[str] | None = None,
    structural_patches_applied: int = 0,
    card_seconds: float = 0.0,
) -> dict[str, Any]:
    """A JSON-serializable provenance manifest for the reconcile_timeline C13 loop.

    Distinct from :func:`timeline_manifest` (the content-addressed resume cache key):
    this is the human/loop-facing record the C13 evaluator-optimizer loop persists —
    the per-beat allotment joined back to its demand's bounds, plus the run-level
    convergence telemetry (§7.7). Every value is a plain ``str``/``float``/``int``/
    ``bool``/``list``/``None`` so the whole dict round-trips through ``json.dumps``
    with no dataclass or custom encoder.

    Per beat (one entry per allotment, in allotment order, joined to its demand by
    ``id``): the allotment's ``seconds``/``vo_rate``/``unsayable``, the demand's
    ``[min, max]`` bounds (0.0 when no matching demand exists), and ``feasible`` —
    the beat id is NOT in :func:`infeasible_beats`. ``vo_rate`` stays ``None`` (JSON
    null) for a beat with no words.

    Pure: no I/O, no LLM, deterministic for a given input.
    """
    demand_by_id = {d.id: d for d in demands}
    infeasible_ids = infeasible_beats(demands, target_seconds)
    flagged = set(infeasible_ids)
    beats: list[dict[str, Any]] = []
    for allotment in allotments:
        demand = demand_by_id.get(allotment.id)
        beats.append(
            {
                "id": allotment.id,
                "seconds": float(allotment.seconds),
                "min_seconds": float(demand.min) if demand else 0.0,
                "max_seconds": float(demand.max) if demand else 0.0,
                "vo_rate": allotment.vo_rate,
                "feasible": allotment.id not in flagged,
                "unsayable": allotment.unsayable,
                "footage_out_point_seconds": demand.footage_out_point_seconds if demand else None,
            }
        )
    return {
        "beats": beats,
        "run": {
            "target_seconds": float(target_seconds),
            "loop_iterations": int(loop_iterations),
            "converged": bool(converged),
            "pacing_score": pacing_score,
            "infeasible_beat_ids": list(infeasible_ids),
            "drift_seconds": budget_drift(allotments, target_seconds),
            "convergence_outcome": convergence_outcome,
            "structural_relief_requested": bool(structural_relief_requested),
            "overrun_seconds": float(overrun_seconds),
            "protected_beat_count": int(protected_beat_count),
            "structural_rejected_ids": list(structural_rejected_ids or []),
            "structural_reallocated_ids": list(structural_reallocated_ids or []),
            "structural_patches_applied": int(structural_patches_applied),
            "card_seconds": float(card_seconds),
        },
    }
