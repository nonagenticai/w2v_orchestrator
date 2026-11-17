"""Pure whole-film gestalt metrics for the A11 advisory ship gate.

A collection of side-effect-free functions over the vectors the
``video_analyzer`` already returns (scene boundaries, per-frame feature vectors,
per-shot text descriptions) plus the compiler's already-measured final-mux
loudness. NOTHING here does I/O, decodes a pixel, or re-measures anything — the
stage (``demo_video/final_critique.py``) gathers the analyzer/compile outputs and
hands plain vectors in; every number below is derived from those.

The gestalt read is deliberately coarse and advisory: it fuses shot-to-shot
continuity, whole-film visual cohesion, tonal drift across the narration, and the
mastered loudness into a single ``ship_ready`` flag. It is NEVER a hard gate — the
stage treats a ``False`` as "ask the film-critic worker", not "block the render".

Feature-vector shape (chosen here, sourced by the stage): a ``frame`` is a mapping
carrying ``shot_id`` (a real ``clip-<n>`` spine id) and ``features`` (a short
sequence of floats — brightness / sharpness / dominant-colour components read off
the analyzer's per-frame record). ``continuity_deltas`` / ``cohesion_score`` reason
purely over those float vectors, so the sourcing can change without touching the
math.

⚠️ **THE FEATURE VECTOR IS CURRENTLY ALMOST ALL CONSTANT.** That "read off the analyzer's
per-frame record" is aspirational: ``video_analyzer_service`` emits
``{index, timestamp, path, description, ocr_text, labels}`` per frame and NO per-frame
``brightness`` / ``sharpness`` / ``dominant_color``. The stage's ``_feature_vector``
therefore substitutes whole-video means for two components and pads ``[0, 0, 0]`` for the
colour triple, so only ``motion`` can vary between shots — and it is 0 across a static
screen capture. Everything in this module is correct; it is simply being fed vectors that
carry no signal on the current wire.

That is why several functions here return ``None`` rather than a number: an unmeasurable
term MUST be visibly unmeasured. ``1.0`` cohesion, ``0.0`` tonal variance and ``0.0``
continuity deltas are all the FLATTERING degenerate values, they are indistinguishable
from a genuinely excellent film, and the nightly calibration job turns a corpus of them
into ``continuity_l2: 0.0`` / ``cohesion_floor: 1.0`` — the maximally strict gate — while
``corpus_n`` grows enough to satisfy its min-samples guard. Callers MUST NOT observe a
``None`` term into a histogram. Do not "simplify" these guards into constants.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from statistics import median, pstdev
from typing import Final

# The ONE loudness number and the window derived from it (see that module). It
# imports nothing from this package, so there is no cycle.
from src.services.stages.shared.loudness_contract import ship_loudness_window

# Reuse the retention watch-curve decay model + rounding so the ASSEMBLED momentum
# curve is directly diffable against ``predicted_watch_curve`` (same tau / cliff /
# precision). Single source of truth — never re-declare these here. ``retention``
# does NOT import this module, so there is no import cycle.
from src.services.stages.shared.retention import (
    _CLIFF_MULTIPLIER,
    _CLIFF_OFFSET_SECONDS,
    _CURVE_PRECISION,
    _DECAY_TAU_SECONDS,
    coerce_beat_seconds,
)

# The STRICT float coercers are canonical in ``src.utils`` (shared with the permissive
# ``coerce_float`` family, which is deliberately NOT the same function — see that
# module). Aliased to the historical private names so every call site below reads
# unchanged.
from src.utils import as_float as _as_float
from src.utils import opt_float as _opt_float

# Loudness ship window (integrated LUFS): DERIVED from the master target the compiler
# actually applies, never restated beside it. See ``loudness_contract`` — the band is
# ``COMPILER_MASTER_LUFS ± SHIP_TOLERANCE_LU``, so it cannot be written in a way that
# excludes the target.
#
# ⚠️ CORRECTION (2026-09-16). This comment previously read "centred on the RATIFIED
# -12.5 LUFS master target (video_compiler_service ``master_lufs``/``DEFAULT_MASTER_LUFS``
# = -12.5 … the roadmap's -14 is stale)". Every clause of that was false on the day it
# was written and is false now: the compiler's value is -14.0, it has been since PR #217
# (2026-09-12), and -12.5 is the value that PR ABANDONED because it drove ~5.5 dB of
# gain riding on a -18 LUFS narration bus — the audible "breathing" complaint. The band
# it produced, ``(-14.0, -11.0)``, therefore sat with its FLOOR exactly ON the real
# target, and convicted three of three live demo_video masters (-14.2/-14.1/-14.1) that
# were correctly mastered. The width is unchanged at ±1.5 LU; only the centre moved.
# Overridable per-run via ``ceilings["loudness_lufs"]``.
_DEFAULT_LOUDNESS_WINDOW: tuple[float, float] = ship_loudness_window()
# Adjacent-shot L2 distance above which a cut reads as a continuity break, and the
# cohesion floor a whole film must clear to be visually coherent. Static defaults
# (baseline-variance ceilings are DEFERRED — see the stage docstring).
_DEFAULT_CONTINUITY_CEILING = 0.5
_DEFAULT_COHESION_FLOOR = 0.5

# Lexical tone proxy: three keyword buckets whose per-shot hit counts form a small
# tone vector. A crude stand-in for a learned tone embedding — enough to flag a
# jarring register jump between adjacent narration beats for an advisory critic.
_TONE_KEYWORDS: dict[str, frozenset[str]] = {
    "energy": frozenset(
        {"fast", "now", "instantly", "boom", "launch", "speed", "action", "click", "power", "boost"}
    ),
    "formality": frozenset(
        {"please", "therefore", "however", "furthermore", "consider", "professional", "regarding"}
    ),
    "warmth": frozenset(
        {"welcome", "love", "friendly", "enjoy", "happy", "great", "wonderful", "help", "thanks"}
    ),
}


def _shot_id(frame: Mapping[str, object]) -> str:
    return str(frame.get("shot_id") or "")


def _features(frame: Mapping[str, object]) -> list[float]:
    """The float feature vector off a frame mapping (defensive; ``[]`` when absent)."""
    raw = frame.get("features")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return [_as_float(x) for x in raw]
    return []


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance over the common prefix (unequal lengths ⇒ zip to shorter)."""
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=False)) ** 0.5


def build_beat_intervals(
    lines: Sequence[Mapping[str, object]],
) -> list[tuple[str, float, float]] | None:
    """Cumulative ``[start, end)`` intervals per beat from ``(id, seconds)``.

    The offset accrues BEFORE adding each beat's ``seconds`` — mirroring
    ``predicted_watch_curve``'s "retention at start-of-beat" accounting
    (``retention.py``) — so a containment join is internally consistent with the curve
    ``momentum_gap`` diffs against. Returns ``None`` when the spine is degenerate (no
    id-bearing lines, or every ``seconds`` is 0 so the whole film has zero width): the
    caller then falls back to the positional zip.

    Durations are parsed with the CANONICAL :func:`retention.coerce_beat_seconds` — the
    same coercer ``predicted_watch_curve`` uses — so the containment join and the curve it
    is diffed against can never disagree about beat width. A JSONB spine legitimately
    round-trips ``5.0`` as ``"5.0"``; the old STRICT ``as_float`` read that as ``0.0``, so
    an all-string spine returned ``None`` (silent fallback to the positional zip) and a
    MIXED spine was worse — zero-width spans that no cut can ever fall inside, silently
    mis-attributing every cut.

    A duration that is neither absent nor parseable is a real wire defect, so it raises
    ``ValueError`` rather than degrading silently: the stage's momentum guard logs it with
    a traceback and falls back, which is LOUD, whereas a fabricated ``0.0`` is invisible.

    Pure — no ctx/IO."""
    out: list[tuple[str, float, float]] = []
    offset = 0.0
    for ln in lines:
        if not isinstance(ln, Mapping) or ln.get("id") is None:
            continue
        raw = ln.get("seconds")
        secs = coerce_beat_seconds(raw)
        if secs is None:
            raise ValueError(
                f"build_beat_intervals: line {ln['id']!r} has an unparseable seconds value {raw!r}"
            )
        out.append((str(ln["id"]), offset, offset + secs))
        offset += secs
    return out if (out and offset > 0.0) else None


def sample_frames_at_cuts(
    scene_boundaries: Sequence[float] | None,
    duration: float,
    spine_ids: Sequence[str],
    *,
    beat_intervals: Sequence[tuple[str, float, float]] | None = None,
) -> list[tuple[str, float]]:
    """Join each scene-cut timestamp to a real spine id — positional by default, or
    by time-containment when ``beat_intervals`` is supplied.

    ``scene_boundaries`` are the analyzer's cut timestamps (seconds). A non-numeric or
    negative cut is skipped, and a cut past ``duration`` is dropped (a stray/garbled
    boundary). Returns ``(spine_id, cut_time)`` pairs in order.

    - **Positional (default, ``beat_intervals`` absent):** ``spine_ids`` are the ordered
      ``clip-<n>`` line ids, zipped to the cut times — fewer cuts than ids (or
      vice-versa) ⇒ zip to the shorter. This path is byte-identical to the legacy join.
    - **Containment (``beat_intervals`` given):** each cut is placed in the beat whose
      ``[start, end)`` interval contains it (half-open ⇒ a boundary cut joins the LATER
      beat); a cut at/after the last interval's end clamps to the last beat (never
      dropped). This defeats the cardinality mis-pairing the positional zip suffers when
      the cut count differs from the id count. ``spine_ids`` is ignored on this path.

    Pure — no ctx/IO.
    """
    dur = _as_float(duration)
    times: list[float] = []
    for raw in scene_boundaries or []:
        t = _opt_float(raw)
        if t is None or t < 0.0:
            continue
        if dur > 0.0 and t > dur:
            continue
        times.append(t)
    if not beat_intervals:  # today's path — byte-identical
        return [(str(sid), t) for sid, t in zip(spine_ids, times, strict=False)]
    pairs: list[tuple[str, float]] = []
    for t in times:
        sid = beat_intervals[-1][0]  # clamp: a cut at/after the last end → last beat (never drop)
        for bid, start, end in beat_intervals:
            if start <= t < end:  # half-open ⇒ a boundary cut joins the LATER beat
                sid = bid
                break
        pairs.append((sid, t))
    return pairs


def assembled_momentum_curve(
    cut_times: Sequence[float],
    beat_motion: Sequence[float],
    total_duration: float,
    *,
    cliff_times: Sequence[float] | None = None,
) -> list[float]:
    """Retention-shaped ACTUAL momentum per beat, directly diffable against ``predicted_watch_curve``.

    Per beat ``i`` the assembled momentum reuses the retention watch-curve decay model
    (single-exponential in the cut time with the extra past-cliff drop), MODULATED by
    the measured motion at that cut so a low-motion cut delivers less momentum than the
    schedule predicted::

        a_i = exp(-cut_times[i] / _DECAY_TAU_SECONDS) * (0.7 + 0.3 * m_i)
        if cliff_times[i] >= _CLIFF_OFFSET_SECONDS: a_i *= _CLIFF_MULTIPLIER

    ``m_i`` is clamped into ``[0, 1]`` (junk ⇒ ``0.0``); when ``cut_times`` and
    ``beat_motion`` differ in length they align to ``min(len)``. Empty input ⇒ ``[]``.
    ``total_duration`` is accepted for a stable call contract with the stage (the cuts
    are already duration-filtered upstream) and does not enter the per-beat math.

    ``cliff_times`` is the AXIS the retention cliff step is evaluated on, defaulting to
    ``cut_times``. It exists because the cliff is a STEP, and ``momentum_gap``
    self-normalizes scale but cannot normalize away a step: this curve runs on the
    RENDERED, t0-rebased clock while ``predicted_watch_curve`` runs on the INTENDED
    cumulative beat offset, so a ~2s render overrun straddling the 30s cliff made ONE
    curve drop 15% and not the other, injecting a spurious ~0.11 gap out of thin air.
    Passing the intended per-beat offsets here puts both cliffs on the SAME axis, so the
    step cancels and only genuine motion under-delivery survives the diff. A shorter
    ``cliff_times`` than ``cut_times`` falls back to the rendered time for the tail.
    """
    _ = total_duration  # duration-filtering happens upstream; kept for a stable contract
    n = min(len(cut_times), len(beat_motion))
    curve: list[float] = []
    for i in range(n):
        t = _as_float(cut_times[i])
        m = max(0.0, min(1.0, _as_float(beat_motion[i])))
        a = math.exp(-t / _DECAY_TAU_SECONDS) * (0.7 + 0.3 * m)
        cliff_t = _as_float(cliff_times[i]) if cliff_times and i < len(cliff_times) else t
        if cliff_t >= _CLIFF_OFFSET_SECONDS:
            a *= _CLIFF_MULTIPLIER
        curve.append(round(a, _CURVE_PRECISION))
    return curve


def momentum_gap(
    assembled: Sequence[tuple[str, float]],  # (spine_id, assembled_momentum), cut order
    predicted: Sequence[tuple[str, float]],  # (spine_id, predicted_retention), beat order
) -> float | None:
    """Worst NORMALIZED under-delivery of the assembled momentum vs A1's predicted watch
    curve, aligned by spine id.

    Both curves are keyed by spine id (NOT positional index) and each is normalized by its
    value at the FIRST COMMON beat before diffing, so a uniform motion level cancels the
    (0.7 + 0.3*m) baseline factor and a healthy film reads ~0, while a genuine mid-run
    motion sag still yields a positive gap. Directional — only under-delivery counts.

    This function only COMPUTES the number; whether it gates ship_ready is decided in
    gestalt_summary and is opt-in (a configured ``ceilings["momentum_gap_max"]``). It is
    spine-id-aligned + scale-normalized (fixing the old positional-index diff + the
    motion-baseline bias), and the cut↔beat join is a time-containment against the intended
    per-beat durations (via ``build_beat_intervals``), not a positional pairing. Residual
    rendered-vs-intended duration drift is why the gate stays off by default and is
    calibrated live (the cliff-step time-base mismatch is FIXED — see
    ``assembled_momentum_curve``'s ``cliff_times``).

    Fewer than two common beats, or a non-positive first-beat anchor ⇒ ``None``, NOT
    ``0.0``: there is nothing to observe, and ``0.0`` is the FLATTERING value (a
    perfectly-delivered film) — indistinguishable from a genuinely measured pass. The
    caller must persist ``None`` and skip the Prometheus observation entirely, so the
    calibration corpus stops being a corpus of silent zeros.
    """
    pred: dict[str, float] = {}
    for sid, v in predicted:
        if sid not in pred:  # first-occurrence wins — do NOT dict()-collapse duplicate ids
            pred[sid] = _as_float(v)
    aligned: list[tuple[float, float]] = []
    for sid, a in assembled:
        if sid in pred:
            aligned.append((_as_float(a), pred[sid]))
    if len(aligned) < 2:
        return None  # no adjacent-beat diff exists — unobservable, NOT a zero gap
    a0, p0 = aligned[0]
    if a0 <= 0.0 or p0 <= 0.0:
        return None  # degenerate anchor ⇒ the normalization is undefined
    worst = 0.0
    for a, p in aligned:
        deficit = (p / p0) - (a / a0)
        if deficit > worst:
            worst = deficit
    # Round to the curve precision so sub-micro float residue on an otherwise-perfect
    # match (e.g. 0.72/0.9 == 0.7999999999999999) reads as a clean 0.0, never a phantom gap.
    return round(max(0.0, worst), _CURVE_PRECISION)


def continuity_deltas(
    frames: Sequence[Mapping[str, object]], ceiling: float
) -> list[dict[str, object]]:
    """Adjacent-pair L2 feature distance; a pair above ``ceiling`` is a break.

    Emits ``{"from_id", "to_id", "delta"}`` for every adjacent shot pair whose
    feature-vector distance exceeds ``ceiling`` (a hard visual discontinuity at the
    cut). Fewer than two frames ⇒ no adjacent pair ⇒ ``[]`` (never divides by zero).
    """
    out: list[dict[str, object]] = []
    for a, b in itertools.pairwise(frames):
        delta = _l2(_features(a), _features(b))
        if delta > ceiling:
            out.append({"from_id": _shot_id(a), "to_id": _shot_id(b), "delta": round(delta, 4)})
    return out


def continuity_stats(frames: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """ADVISORY calibration evidence: the FULL adjacent-shot L2 delta distribution.

    Summarizes EVERY adjacent-pair distance — not only the ones above a ceiling. This
    exists because :func:`continuity_deltas` is CENSORED at the ceiling: it emits only
    pairs that already breached, so the persisted evidence can never justify moving
    ``_DEFAULT_CONTINUITY_CEILING`` off its hardcoded 0.5. These stats are the missing
    baseline-variance prerequisite for that calibration.

    This is calibration evidence ONLY. It NEVER touches ``ship_ready``, gates nothing,
    and is entirely independent of the configured ceiling — ``continuity_breaks``
    remains the sole gate evidence, and the two are kept separate on purpose.

    Returns ``{"n", "min", "max", "mean", "stdev", "p50", "p90", "p95"}`` (floats
    rounded to 4dp; ``stdev`` is the POPULATION stdev, so a single pair reads 0.0).
    Fewer than two frames ⇒ no adjacent pair ⇒ ``{"n": 0}``. Percentiles are
    NEAREST-RANK on a sorted copy, deliberately not ``statistics.quantiles`` — that
    requires ``n >= 2`` and its interpolation varies by method, which would make a
    calibration corpus non-reproducible across runs.

    ⚠️ DEGENERATE DISTRIBUTION. When EVERY adjacent delta is exactly ``0.0`` the frames
    are byte-identical in feature space and there is no distribution at all — only the
    numeric shadow of one. This returns ``{"n": <pairs>, "degenerate": True}`` with NO
    ``min``/``max``/``mean``/``stdev``/percentile keys, so:

      * ``_emit_gestalt_metrics`` finds no ``max`` and observes NOTHING, and
      * ``continuity_ceiling_suggested`` finds no ``mean``/``stdev`` and stays ``None``.

    That is the whole point. The analyzer emits per-frame records of
    ``{index, timestamp, path, description, ocr_text, labels}`` — there is NO per-frame
    ``brightness`` / ``sharpness`` / ``dominant_color`` — so ``_feature_vector`` falls back
    to WHOLE-VIDEO means for two components and pads ``[0,0,0]`` for the colour triple.
    Five of six components are therefore identical for every shot BY CONSTRUCTION and only
    ``motion`` can vary. On a static screen-capture (measured: ``duplicate_frame_ratio``
    0.9754) even that is 0, so every delta is exactly 0.0. Emitting ``max: 0.0`` into the
    nightly calibration corpus would drive the derived ``continuity_l2`` ceiling to 0.0 —
    the maximally strict gate, failing every real film — while ``corpus_n`` grew so the
    min-samples guard read healthy. Absence must stay visible as absence.
    """
    # Same pairwise metric as ``continuity_deltas`` — reused verbatim so the censored
    # gate view and this uncensored view can never drift apart.
    deltas = [_l2(_features(a), _features(b)) for a, b in itertools.pairwise(frames)]
    if not deltas:
        return {"n": 0}
    ordered = sorted(deltas)
    n = len(ordered)
    if ordered[-1] <= 0.0:  # deltas are non-negative ⇒ max 0 means EVERY delta is 0
        return {"n": n, "degenerate": True}

    def _nearest_rank(pct: int) -> float:
        """The ``ceil(pct/100 * n)``-th smallest value (1-indexed), clamped into range."""
        rank = math.ceil(pct / 100.0 * n)
        return ordered[min(max(rank, 1), n) - 1]

    return {
        "n": n,
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
        "mean": round(sum(ordered) / n, 4),
        "stdev": round(pstdev(ordered), 4),
        "p50": round(_nearest_rank(50), 4),
        "p90": round(_nearest_rank(90), 4),
        "p95": round(_nearest_rank(95), 4),
    }


def cohesion_score(frames: Sequence[Mapping[str, object]]) -> float | None:
    """Whole-film visual cohesion in ``[0, 1]`` — 1 minus normalized dispersion.

    Each frame's feature vector is compared to the component-wise MEDIAN vector; the
    mean distance-to-median is squashed into ``[0, 1]`` via ``1 / (mean_dist + 1)``,
    so identical frames score 1.0 and an ever-more-scattered film tends to 0.

    Returns ``None`` — NOT ``1.0`` — whenever there is nothing to disperse:

      * fewer than two non-empty vectors, or a zero-width vector, AND
      * **ZERO DISPERSION**: every vector identical, so ``mean_dist == 0``.

    ``1.0`` is the DEGENERATE PERFECT value. The entire live corpus read
    ``cohesion: 1.0``, and fixing the frame spine did NOT fix that — it made it worse,
    because ``1/(0+1) == 1.0`` is returned just as confidently for two identical vectors
    as for a genuinely flawless film. The analyzer emits no per-frame ``brightness`` /
    ``sharpness`` / ``dominant_color`` (see :func:`continuity_stats`), so five of six
    feature components are identical for every shot BY CONSTRUCTION; on a static capture
    the sixth (motion) is 0 too, and the vectors coincide exactly. A ``1.0`` observed from
    that into the nightly corpus drives the derived ``cohesion_floor`` (a p05 of a FLOOR
    family) to 1.0 — a gate no real film can clear.

    Exact-zero dispersion is the right trigger: real footage with any measurable variation
    never lands on exactly 0.0, so this fires only on the structural case.
    """
    vectors = [v for v in (_features(f) for f in frames) if v]
    if len(vectors) < 2:
        return None
    dim = min(len(v) for v in vectors)
    if dim == 0:
        return None
    trimmed = [v[:dim] for v in vectors]
    center = [median(col) for col in zip(*trimmed, strict=False)]
    mean_dist = sum(_l2(v, center) for v in trimmed) / len(trimmed)
    if mean_dist <= 0.0:  # every vector identical ⇒ no dispersion was MEASURED
        return None
    return max(0.0, min(1.0, 1.0 / (mean_dist + 1.0)))


def _tone_vector(text: str) -> list[float]:
    """Keyword-hit counts across the three tone buckets → a 3-float tone vector."""
    words = [w.strip(".,!?;:\"'()").lower() for w in text.split()]
    counts = {bucket: 0 for bucket in _TONE_KEYWORDS}
    for word in words:
        for bucket, keywords in _TONE_KEYWORDS.items():
            if word in keywords:
                counts[bucket] += 1
    return [float(counts["energy"]), float(counts["formality"]), float(counts["warmth"])]


def tonal_variance(frame_analyses: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Tonal drift across the per-shot narration descriptions (lexical proxy).

    Each shot's text (``description``/``text``) is reduced to a 3-float tone vector
    via :func:`_tone_vector`; ``variance`` is the summed component-wise population
    variance of those vectors, and ``worst_cut`` is the adjacent shot pair with the
    largest tone jump (``None`` when there is no adjacent pair).

    Fewer than two shots ⇒ ``{"variance": None, "worst_cut": None}``. ``None``, not
    ``0.0``: with one shot there is no drift to measure, and ``0.0`` is the flattering
    "perfectly consistent tone" reading — the live corpus persisted exactly that for
    every film whose shot spine had collapsed.

    ⚠️ THE PROXY IS USUALLY BLIND. :func:`_tone_vector` is a 3-bucket KEYWORD COUNTER over
    a ~24-word vocabulary. Real narration prose ("A dashboard loads showing a project
    board…") hits none of those buckets, so every shot reduces to ``[0, 0, 0]`` and the
    computed variance is 0.0 — measured: 30 frames with entirely distinct descriptions
    still produced ``variance: 0.0``. That is the counter failing to fire, NOT a film with
    consistent tone. When EVERY tone vector is the zero vector the proxy observed nothing
    and this returns ``variance: None`` so the term is visibly unmeasured.

    ``worst_cut`` is likewise only reported for a jump that is genuinely > 0. The seed used
    to be ``-1.0``, so the FIRST adjacent pair always "won" and a wall of identical zero
    vectors produced ``{"from_id": "clip-1", "to_id": "clip-2", "jump": 0.0}`` — a named
    seam presented as the film's worst tonal break, invented out of nothing. Prod
    (pre-spine-fix) correctly persisted ``null`` here; this keeps it that way.
    """
    shots: list[tuple[str, list[float]]] = []
    for fa in frame_analyses:
        text = fa.get("description") or fa.get("text") or fa.get("ocr_text") or ""
        shots.append((str(fa.get("shot_id") or ""), _tone_vector(str(text))))
    if len(shots) < 2:
        return {"variance": None, "worst_cut": None}
    if not any(any(v) for _, v in shots):
        # The keyword proxy matched nothing anywhere ⇒ it made no observation at all.
        return {"variance": None, "worst_cut": None}

    vectors = [v for _, v in shots]
    dim = len(vectors[0])
    variance = 0.0
    for i in range(dim):
        col = [v[i] for v in vectors]
        mean = sum(col) / len(col)
        variance += sum((x - mean) ** 2 for x in col) / len(col)

    worst: dict[str, object] | None = None
    # Seeded at 0.0, NOT -1.0: a zero-width "jump" is not a tonal break, and seeding
    # below zero made the first adjacent pair win unconditionally (see the docstring).
    worst_jump = 0.0
    for (from_id, va), (to_id, vb) in itertools.pairwise(shots):
        jump = _l2(va, vb)
        if jump > worst_jump:
            worst_jump = jump
            worst = {"from_id": from_id, "to_id": to_id, "jump": round(jump, 4)}
    return {"variance": round(variance, 4), "worst_cut": worst}


def final_mux_loudness(compile_loudness: Mapping[str, object] | None) -> dict[str, object] | None:
    """Normalize the compiler's threaded final-mux loudness block (NEVER re-measures).

    A pure pass-through of ``{integrated_lufs, true_peak_dbtp}`` (each coerced to
    ``float | None``). ``None`` (or a non-mapping) ⇒ ``None`` — the compiler measured
    no loudness, so the gate makes no loudness claim.

    ⛔ ``lra`` AND ``silence_ratio`` WERE DROPPED FROM THIS BLOCK (A/V wave 7), and the
    deletion loses no measurement. They were copied here and read by NOBODY: the ship
    term is :func:`loudness_verdict`, which reads ``integrated_lufs`` against the
    window, and ``_failed_ceilings``/``_loudness_reasons`` in ``final_critique`` read
    the VERDICT. A number carried into a gate module that no predicate consults is not
    a gate, it is a second copy.

    They are not deleted at the SOURCE and must not be: the compiler measures both in
    ``video_compiler_service/src/services/loudness.py`` (``parse_loudness``) and
    persists the whole block in ``CompileJob.result``, and this orchestrator persists
    the same block again on the compile stage's own DAG output (``_compile_loudness``
    reads it from there). The raw numbers therefore remain available for the
    calibration neither of them has had yet — only the unread copy inside the gate's
    verdict payload is gone.

    ⛔ NOT wired into ``loudness_verdict`` instead, deliberately. Neither has a ceiling
    that is anything but a guess: there is no measured fleet distribution for either,
    ``silence_ratio`` is ``None`` BOTH when silencedetect produced no output AND when
    it found no silent span (so its absence is ambiguous by construction), and an
    opt-in ``ceilings[...]`` entry nobody configures is exactly the fires-on-nothing
    term that ``momentum_verdict`` below had to be rewritten to stop being. Arming
    either one is a calibration decision to be taken against measured numbers, and it
    starts by reading the persisted compiler block, not by re-adding this copy.
    """
    if not isinstance(compile_loudness, Mapping):
        return None
    return {
        "integrated_lufs": _opt_float(compile_loudness.get("integrated_lufs")),
        "true_peak_dbtp": _opt_float(compile_loudness.get("true_peak_dbtp")),
    }


def loudness_window(ceilings: Mapping[str, object]) -> tuple[float, float]:
    """The ``[lo, hi]`` integrated-LUFS ship window (default ``[-15, -13]``)."""
    raw = ceilings.get("loudness_lufs")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)) and len(raw) == 2:
        lo, hi = _opt_float(raw[0]), _opt_float(raw[1])
        if lo is not None and hi is not None:
            return (min(lo, hi), max(lo, hi))
    return _DEFAULT_LOUDNESS_WINDOW


# The three states a loudness term can be in. ``PASS``/``FAIL`` are claims about the
# master; ``INSUFFICIENT_EVIDENCE`` is a claim about the MEASUREMENT, and it is
# deliberately neither of the other two.
# NOTE (S105): "PASS" in the NAME trips bandit's hardcoded-password heuristic. This is a
# verdict string, not a credential.
LOUDNESS_PASS: Final = "pass"  # noqa: S105
LOUDNESS_FAIL: Final = "fail"
LOUDNESS_INSUFFICIENT_EVIDENCE: Final = "insufficient_evidence"


# The FOUR states the momentum term can be in, and the first one is the whole point.
#
# ⛔ ``momentum_ok`` used to be a bare bool that was ``True`` whenever no
# ``ceilings["momentum_gap_max"]`` was configured — and NOTHING configures it, anywhere
# in GitOps or in any run config. So an AND-term of ``ship_ready`` was vacuously true on
# every master ever produced, and ``ship_ready`` read as if momentum had been checked.
# ``final_critique._failed_ceilings`` compounded it: ``if not gestalt["momentum_ok"]``
# can never fire, so "momentum_gap over ceiling" was a reason string no run could reach.
# A term that cannot observe anything must SAY so rather than report a pass.
#
# ⛔ NO DEFAULT CEILING IS INTRODUCED HERE, deliberately. ``momentum_gap`` has no
# measured fleet distribution, and this project has repeatedly shipped signals
# calibrated on nothing (the deband ceiling that fired on 96.7% of masters is the
# reference case). ``UNCONFIGURED`` names the unarmed state honestly; arming it is a
# separate decision that starts with measuring the gap, not with picking a number.
#
# ``_momentum_ok`` collapses these back to the SAME two states the bool had, so
# ``ship_ready`` is byte-identical for ALL inputs.
#
# ⚠️ THAT COLLAPSE IS STILL A VACUOUS PASS, and naming the state did not fix it.
# ``UNCONFIGURED`` maps to ``momentum_ok = True`` and ``ship_ready`` ANDs it, so on every
# master ever produced this term has been SATISFIED without observing anything, and
# ``final_critique._failed_ceilings``'s ``if not gestalt["momentum_ok"]`` still cannot
# fire. The four-state verdict made that legible to a reader who goes looking; it did not
# make it legible to anything that reads the summary as a gate.
#
# ⛔ AND THE FIX IS NOT TO FAIL CLOSED. Wave 8 recorded what that costs: failing a term
# closed with no calibrated ceiling makes ``ship_ready`` false on 100% of masters, which
# is the near-universal-flag failure, not a gate — the same lesson as the deband ceiling
# that fired on 96.7%. So ``ship_ready`` is deliberately UNCHANGED and the honesty rides
# its own NON-GATING key, :data:`UNOBSERVED_TERMS_KEY`: an explicit
# insufficient-evidence list naming every ``ship_ready`` term that could not observe its
# subject. ``ship_ready: true`` can then no longer be read as "all terms checked out"
# without also reading which of them were armed at all.
#: The summary key carrying the insufficient-evidence term list (see above). A
#: REPORTING key: nothing here is a ``ship_ready`` term, by design.
UNOBSERVED_TERMS_KEY: Final = "unobserved_terms"

MOMENTUM_UNCONFIGURED: Final = "unconfigured"
MOMENTUM_PASS: Final = "pass"  # noqa: S105
MOMENTUM_FAIL: Final = "fail"
MOMENTUM_INSUFFICIENT_EVIDENCE: Final = "insufficient_evidence"


def momentum_verdict(momentum_gap: float | None, ceiling: float | None) -> str:
    """The FOUR-state momentum term (see the constants above).

    * no ``ceiling`` ⇒ ``UNCONFIGURED`` — the term is not armed and observed nothing.
      This is the live state on every run today, and it is NOT a pass.
    * ceiling set, ``momentum_gap is None`` ⇒ ``INSUFFICIENT_EVIDENCE`` — an operator
      opted the term in and the gap could not be measured. Distinct from a breach,
      and (as before) it does NOT read as ok: "unobservable" must never pass a term
      somebody deliberately armed.
    * ceiling set and measured ⇒ ``PASS`` / ``FAIL`` on the comparison.
    """
    if ceiling is None:
        return MOMENTUM_UNCONFIGURED
    if momentum_gap is None:
        return MOMENTUM_INSUFFICIENT_EVIDENCE
    return MOMENTUM_PASS if momentum_gap <= ceiling else MOMENTUM_FAIL


def _momentum_ok(verdict: str) -> bool:
    """The ``ship_ready`` term, collapsed back to two states.

    Byte-identical to the pre-four-state bool for ALL inputs: ``UNCONFIGURED`` and
    ``PASS`` are the cases that used to be ``True``; ``FAIL`` and
    ``INSUFFICIENT_EVIDENCE`` are the cases that used to be ``False``. The new
    distinction lives in the reported ``momentum_verdict``, so naming the unarmed state
    cannot, by itself, sink or save a run.
    """
    return verdict in (MOMENTUM_UNCONFIGURED, MOMENTUM_PASS)


# The name(s) the compiler's ``unmeasured`` channel uses for the loudness measurement.
# ``integrated_lufs`` is the live one (a ``_unmeasured_metrics`` kwarg name); ``loudness``
# is tolerated as an alias only. See :func:`_loudness_measured`.
_LOUDNESS_UNMEASURED_NAMES: Final = frozenset({"integrated_lufs", "loudness"})


def _loudness_measured(mastering: Mapping[str, object] | None) -> bool | None:
    """Did the compiler say it MEASURED loudness? ``None`` when it did not say.

    Reads the compiler's ``unmeasured`` channel on its mastering report, tolerating
    every shape it is plausibly emitted in, because this orchestrator does not own it:

    * a **sequence** of names (``["integrated_lufs", "colorspace_ok"]``),
    * a **mapping** of name to truthy flag (``{"integrated_lufs": true}``),
    * a bare **bool** (``true`` ⇒ the whole report is unmeasured).

    ⛔ THE NAMES ARE METRICS, NOT STAGES. The compiler builds this list in
    ``video_compiler_service`` ``mastering.py`` via ``_unmeasured_metrics(**metrics)``,
    so the strings on the wire are its KEYWORD ARGUMENT names — ``integrated_lufs``,
    ``dead_frame_ratio``, ``cfr_ok``, ``av_gap_seconds`` — and the word ``loudness``
    never appears among them. The first cut of this function tested for ``"loudness"``,
    which made ``measured=False`` UNREACHABLE against a real master: the whole
    ``insufficient_evidence`` path was dead in production and passed its test only
    because the test hand-fed it a stage name the compiler does not emit. ``loudness``
    is kept as an accepted alias (harmless, and cheap insurance if that channel is ever
    renamed), but ``integrated_lufs`` is the name that actually arrives.

    ``None`` — the key is absent, or the report is — means THE COMPILER MADE NO
    STATEMENT, which is the case on every master produced before that channel shipped.
    Callers must fall back to today's behaviour there, never to "it was unmeasured":
    inventing an unmeasured verdict for a historical master would relabel a fleet of
    real measurements as unjudgeable.
    """
    if not isinstance(mastering, Mapping):
        return None
    raw = mastering.get("unmeasured")
    if raw is None:
        return None
    if isinstance(raw, bool):
        return not raw
    if isinstance(raw, Mapping):
        return not any(bool(raw.get(name)) for name in _LOUDNESS_UNMEASURED_NAMES)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return _LOUDNESS_UNMEASURED_NAMES.isdisjoint({str(x) for x in raw})
    return None


def loudness_verdict(
    loudness: Mapping[str, object] | None,
    window: tuple[float, float],
    *,
    measured: bool | None = None,
) -> str:
    """Three-way loudness verdict — ``pass`` / ``fail`` / ``insufficient_evidence``.

    ⚠️ THE DEFECT THIS REPLACES was missing-as-pass. The old ``_loudness_ok`` returned
    ``True`` both when there was no loudness block at all and when ``integrated_lufs``
    was non-numeric, on the reasoning "no measured value ⇒ no loudness objection". That
    reasoning is sound for a GATE and wrong for a REPORT: it made an unmeasured master
    indistinguishable from a correctly-mastered one, so the absence of the measurement
    read downstream as evidence of quality. A film nobody listened to scored exactly
    like a film that passed.

    ``measured`` is the compiler's own statement (see :func:`_loudness_measured`):
    ``False`` forces ``insufficient_evidence`` even when a numeric LUFS is present —
    the compiler is telling us that number is not a measurement. ``None`` (it made no
    statement) defers entirely to the payload, which is exactly today's behaviour.

    This is a REPORTING distinction, not a new failure mode: ``insufficient_evidence``
    is NOT ``fail``, and :func:`gestalt_summary` keeps treating it as non-blocking for
    ``ship_ready`` (see ``_loudness_ok``). Turning "we could not measure it" into a
    hard ship failure is a calibration decision to be made against measured fleet
    numbers, not a side effect of naming the state.
    """
    if measured is False:
        return LOUDNESS_INSUFFICIENT_EVIDENCE
    if loudness is None:
        return LOUDNESS_INSUFFICIENT_EVIDENCE
    lufs = loudness.get("integrated_lufs")
    if not isinstance(lufs, (int, float)) or isinstance(lufs, bool):
        return LOUDNESS_INSUFFICIENT_EVIDENCE
    lo, hi = window
    return LOUDNESS_PASS if lo <= float(lufs) <= hi else LOUDNESS_FAIL


def _loudness_ok(verdict: str) -> bool:
    """The ``ship_ready`` term: only an affirmative ``fail`` objects.

    Byte-identical to the pre-three-state behaviour for ALL inputs — every input that
    used to return ``True`` maps to ``pass`` or ``insufficient_evidence``, and only the
    out-of-window case maps to ``fail``. The three-way distinction lives in the
    reported ``loudness_verdict``; this deliberately collapses it back to two so that
    naming the unmeasured state cannot, by itself, sink a run.
    """
    return verdict != LOUDNESS_FAIL


def gestalt_summary(
    *,
    frames: Sequence[Mapping[str, object]],
    frame_analyses: Sequence[Mapping[str, object]],
    loudness: Mapping[str, object] | None,
    ceilings: Mapping[str, object] | None = None,
    momentum_gap: float | None = None,
    mastering: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Fuse the per-shot vectors + loudness into one advisory verdict, carrying the
    measured momentum sag as ADVISORY EVIDENCE (not a gate).

    Returns ``{cohesion, continuity_breaks, continuity_stats,
    continuity_ceiling_suggested, tonal, loudness, loudness_verdict, momentum_gap,
    momentum_ok, momentum_verdict, unobserved_terms, insufficient_evidence,
    ship_ready}``.
    ``momentum_verdict`` is the FOUR-state momentum term (``unconfigured`` / ``pass`` /
    ``fail`` / ``insufficient_evidence``) — see :func:`momentum_verdict`. It is
    reporting only: ``momentum_ok`` (and therefore ``ship_ready``) is unchanged for ALL
    inputs. ``unconfigured`` is the LIVE state — nothing sets ``momentum_gap_max``.
    ``unobserved_terms`` is the LIST form of that fact, and the one a reader can act on
    without knowing each term's private vocabulary: every ``ship_ready`` term that could
    not observe its subject, named. It is REPORTING — ``ship_ready`` is unchanged for ALL
    inputs, because failing an unarmed term closed would sink 100% of masters (wave 8) —
    but a ``ship_ready: true`` beside a non-empty ``unobserved_terms`` is visibly a
    verdict taken on fewer terms than it appears to weigh.
    ``loudness_verdict`` is the THREE-state loudness term (``pass`` / ``fail`` /
    ``insufficient_evidence``) — see :func:`loudness_verdict`. It is reporting only:
    ``ship_ready`` still objects on ``fail`` alone, so an unmeasured master is visibly
    unjudged instead of silently indistinguishable from a clean one.
    ``continuity_stats`` is the UNCENSORED delta distribution (see
    :func:`continuity_stats`) — advisory calibration evidence only, always present, and
    never a ``ship_ready`` term; ``continuity_breaks`` stays the gate evidence.
    ``continuity_ceiling_suggested`` is ADVISORY ONLY: a ``mean + 2*stdev`` (two-sigma)
    within-run suggestion for CALIBRATING the continuity ceiling from this run's own
    delta distribution (``None`` when there are fewer than two adjacent deltas). It does
    NOT change the live ceiling — that stays hardcoded at ``_DEFAULT_CONTINUITY_CEILING``
    — and never enters ``continuity_breaks`` or ``ship_ready``.
    The COHESION floor deliberately gets NO within-run suggestion here: ``cohesion_score``
    is a single scalar per film (one number, no within-run distribution), so its floor is
    only calibratable CROSS-run via fleet-wide telemetry, not from a single run.

    INSUFFICIENT EVIDENCE (the critical guard). Set when the visual terms cannot be
    measured, which happens in TWO ways:

      (a) **fewer than two frames** — no adjacent pair exists at all; or
      (b) **zero dispersion across two-or-more frames** — the sampled feature vectors are
          identical, so every continuity delta is 0.0 and cohesion is undefined.

    Either way this sets ``insufficient_evidence: True``, persists every unobservable term
    as ``None`` (never a flattering constant), marks the continuity distribution
    ``degenerate`` instead of publishing a wall of 0.0 percentiles, and forces
    ``ship_ready = False``.

    ⚠️ **(b) IS THE LIVE CASE, AND NO SPINE FIX CAN REMOVE IT.** The analyzer's per-frame
    records carry only ``{index, timestamp, path, description, ocr_text, labels}`` — there
    is NO per-frame ``brightness`` / ``sharpness`` / ``dominant_color``. ``_feature_vector``
    therefore substitutes WHOLE-VIDEO means for two components and pads ``[0,0,0]`` for the
    colour triple, leaving ``motion`` as the only component that can vary between shots;
    on a static screen capture that is 0 as well. **Continuity, cohesion and tonal drift
    are structurally unmeasurable with the data the analyzer currently emits.** Sourcing
    the cut spine from the compiler's rendered ``shot_timeline`` (correct, and kept) raises
    the number of sampled shots but cannot create signal that never arrived — so this guard
    is what stops the extra pairs turning into extra FABRICATED zeros.

    That matters beyond the badge: the nightly calibration workflow derives
    ``continuity_l2`` as a p95 of a ceiling family and ``cohesion_floor`` as a p05 of a
    floor family. A corpus of 0.0/1.0 samples yields ``continuity_l2: 0.0`` and
    ``cohesion_floor: 1.0`` — the maximally strict gate, failing every real film — while
    ``corpus_n`` grows so the min-samples guard reads healthy. An empty corpus is
    self-evidently uncalibratable; a fabricated one is not. Callers MUST emit no histogram
    sample for a term that came back ``None``.

    This also inverts the older behaviour, which computed
    ``ship_ready = loud_ok and momentum_ok`` — both VACUOUSLY true — so a blank or frozen
    film with a single detected shot was the most confident "ship-ready" verdict the gate
    could emit, with zero WM calls and a ``quality_badge: "ship-ready"``.

    It stays a SOFT gate: ``ship_ready=False`` means "this cannot be judged, ask/degrade",
    never "block the render". The stage still ships the artifact and still runs
    ``growth_bundle``; only the badge degrades.

    ``ship_ready`` is otherwise ``True`` only when the loudness (if measured) is
    in-window AND there are no continuity breaks AND cohesion clears the floor AND
    momentum is ok. A film mastered at the compiler's target passes the loudness gate.

    ``momentum_gap`` (the assembled-vs-predicted watch-curve deficit) rides into the
    returned dict as advisory evidence for the LLM critic — it grounds a
    ``momentum_experienced`` score with a real number — or as ``None`` when it could not
    be measured. Whether it also GATES ``ship_ready`` is opt-in: only a configured
    ``ceilings["momentum_gap_max"]`` turns it into a hard term. With no ceiling set (the
    default), ``momentum_ok`` is True and ``ship_ready`` is byte-identical to a run that
    never measured momentum, for ALL inputs — so activation is a config decision after
    live calibration, not a code change. WITH a ceiling set, an UNMEASURED gap
    (``None``) is ``momentum_ok = False``: an opted-in operator asked for the term to be
    enforced, and "we could not measure it" must not silently read as "it passed". The
    per-beat diff is spine-id-aligned and scale-normalized, the cut↔beat join is a
    time-containment (via ``build_beat_intervals`` / the compiler's rendered timeline),
    and the cliff step is evaluated on a shared axis; the residual rendered-vs-intended
    duration drift is why the ceiling is calibrated live rather than defaulted on.
    """
    ceilings = ceilings or {}
    window = loudness_window(ceilings)
    _c = _opt_float(ceilings.get("continuity_l2"))
    continuity_ceiling = _DEFAULT_CONTINUITY_CEILING if _c is None else _c
    _f = _opt_float(ceilings.get("cohesion_floor"))
    cohesion_floor = _DEFAULT_COHESION_FLOOR if _f is None else _f
    # Momentum is an OPT-IN hard gate: with no ``momentum_gap_max`` ceiling configured
    # it stays advisory (momentum_ok is True), so ship_ready is byte-identical to a run
    # that never measured momentum. Setting the ceiling (via ship_gate.ceilings in GitOps
    # / the run config) turns momentum into a real ship_ready term. A0.25 ceiling means
    # "fail ship when the worst normalized watch-curve under-delivery exceeds 25%".
    # An UNMEASURED gap under a configured ceiling fails the term: the operator opted in
    # to enforcing it, and "unobservable" must never read as "passed".
    # FOUR-STATE, and only the unarmed state is new. `momentum_ok` stays the bool it
    # always was (`final_critique` and the `ship_ready` metric label both read it), and
    # `_momentum_ok` reproduces it exactly; `momentum_verdict` is what makes "no ceiling
    # is configured anywhere, so this term observed NOTHING" visible instead of
    # indistinguishable from a measured pass. See the constants above.
    momentum_gap_max = _opt_float(ceilings.get("momentum_gap_max"))
    momentum_state = momentum_verdict(momentum_gap, momentum_gap_max)
    momentum_ok = _momentum_ok(momentum_state)

    loud = final_mux_loudness(loudness)
    # THREE-STATE, and only the middle state is new. `_loudness_measured` reads the
    # compiler's own `unmeasured` channel off the mastering report; `None` (the key or
    # the report is absent, i.e. every master built before that channel shipped) defers
    # to the payload, so the fallback is today's behaviour and never "nothing is
    # measured". `loud_ok` collapses the verdict back to the original two states, so
    # `ship_ready` is unchanged for ALL inputs.
    loud_verdict = loudness_verdict(loud, window, measured=_loudness_measured(mastering))
    loud_ok = _loudness_ok(loud_verdict)
    cohesion = cohesion_score(frames)
    tonal = tonal_variance(frame_analyses)
    # Uncensored delta distribution, computed UNCONDITIONALLY (so the key is always
    # present — ``{"n": 0}`` for a degenerate film) and OUTSIDE the gate branch below:
    # it is calibration evidence for the hardcoded ceiling, never a ship_ready term.
    stats = continuity_stats(frames)
    # ADVISORY-ONLY within-run suggestion for the continuity ceiling: a two-sigma statistical-
    # outlier threshold over THIS run's adjacent-delta distribution (an adjacent cut
    # beyond mean + 2*stdev reads as an anomaly). This NEVER feeds ``continuity_breaks``
    # or ``ship_ready`` — the live ceiling stays hardcoded at ``_DEFAULT_CONTINUITY_CEILING``;
    # this is calibration guidance for a human/telemetry to act on, not a gate term.
    # ``None`` when there is no distribution to derive from (fewer than two adjacent deltas).
    # Reuses the canonical ``_opt_float`` coercer (bool/junk ⇒ None) so a degenerate
    # ``{"n": 0}`` stats block yields ``None`` rather than fabricating a suggestion.
    _stats_n = _opt_float(stats.get("n"))
    _stats_mean = _opt_float(stats.get("mean"))
    _stats_stdev = _opt_float(stats.get("stdev"))
    continuity_ceiling_suggested: float | None = None
    if (
        _stats_n is not None
        and int(_stats_n) >= 2
        and _stats_mean is not None
        and _stats_stdev is not None
    ):
        continuity_ceiling_suggested = round(_stats_mean + 2.0 * _stats_stdev, 4)

    # UNMEASURABLE, in either of two ways:
    #   (a) fewer than two frames ⇒ no adjacent pair exists at all, or
    #   (b) two-or-more frames whose feature vectors carry ZERO DISPERSION — they are
    #       identical, so every delta is 0.0 and cohesion is undefined. That is the LIVE
    #       case, not a corner case: the analyzer emits no per-frame visual features, so
    #       five of six components are constant by construction and a static capture zeroes
    #       the sixth. Fixing the cut spine (which was necessary and is kept) raises the
    #       PAIR COUNT but cannot manufacture signal that was never in the wire.
    # Either way the gate has no visual evidence and must not claim ship-readiness off the
    # remaining vacuously-true terms (see the docstring).
    degenerate_features = bool(stats.get("degenerate")) or cohesion is None
    insufficient_evidence = len(frames) < 2 or degenerate_features
    if insufficient_evidence:
        continuity_breaks: list[dict[str, object]] = []
        ship_ready = False
    else:
        continuity_breaks = continuity_deltas(frames, continuity_ceiling)
        ship_ready = (
            loud_ok
            and not continuity_breaks
            and cohesion is not None
            and cohesion >= cohesion_floor
            and momentum_ok
        )

    # The insufficient-evidence terms, as an explicit list rather than a state each
    # reader has to decode per-term. NON-GATING on purpose — see the constants block for
    # why failing closed here is the wrong move — and derived from each term's OWN verdict
    # vocabulary, so a new unobservable state lands here by construction instead of
    # silently joining the passes.
    unobserved_terms = [
        name
        for name, unobserved in (
            (
                "momentum",
                momentum_state in (MOMENTUM_UNCONFIGURED, MOMENTUM_INSUFFICIENT_EVIDENCE),
            ),
            ("loudness", loud_verdict == LOUDNESS_INSUFFICIENT_EVIDENCE),
        )
        if unobserved
    ]

    return {
        "cohesion": None if cohesion is None else round(cohesion, 4),
        "continuity_breaks": continuity_breaks,
        "continuity_stats": stats,
        "continuity_ceiling_suggested": continuity_ceiling_suggested,
        "tonal": tonal,
        "loudness": loud,
        "loudness_verdict": loud_verdict,
        "momentum_gap": None if momentum_gap is None else round(momentum_gap, 4),
        "momentum_ok": momentum_ok,
        "momentum_verdict": momentum_state,
        UNOBSERVED_TERMS_KEY: unobserved_terms,
        "insufficient_evidence": insufficient_evidence,
        "ship_ready": ship_ready,
    }
