"""Pure, deterministic cold-open framing planner (ORCH-1.3a).

A "cold open" shows a punchy fragment of the opening footage clip instead of
starting on the slow load-in. This module decides how to *frame* that opening
shot — which end of the clip to anchor on (``start``/``end``) and how many
seconds of lead-in to skip — from three inputs only: the clip's real
``DemoVideoClip.seconds`` duration, an optional authored ``hook`` block, and an
optional director ``open_style`` word.

**SIGNAL (ORCH-1.3c):** the opening clip may now carry a real per-second
``motion_timeline``, measured off the recorded footage by video_analyzer's QA pass
and persisted on ``DemoVideoClip``. When that curve exists AND has a *distinguished*
peak, the lead is the peak itself — we cut to the moment the action actually happens
instead of guessing from duration. When it does not (QA off, older analyzer, or a
clip whose motion is flat/uniform so no single moment is "the" moment), framing
falls back to the original duration + authored-anchor + style-perturbation ladder.
This function returns ``(anchor, lead)`` and NEVER decides shot DURATION
(``seconds``) — narration reconciliation owns that downstream.

**Purity contract (mirrors :mod:`motion_plan` / :mod:`design_system`):** no
``datetime.now``, no ``random``, no env reads, no httpx/worker/LLM import. The
only source of variation is ``sha256(open_style|clip_seconds)`` seeding, so the
same inputs always yield the same window (stable across resumes/retries), yet a
different ``open_style`` can shift the lead within a small band. Every default
is a module-level constant; the ``*_hash`` helper is a 16-hex content address
shaped exactly like ``motion_plan.plan_hash``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Literal, cast

__all__ = [
    "Anchor",
    "_motion_xy",
    "_peak_motion_lead",
    "cold_open_hash",
    "cold_open_window",
]

Anchor = Literal["start", "end"]

# ---------------------------------------------------------------------------
# Tunables (module constants — the deterministic default framing).
# ---------------------------------------------------------------------------

# The settled tail reads punchier than the slow load-in, and it matches the
# showcase "end"-only convention, so the no-author default anchors on the end.
_DEFAULT_ANCHOR: Anchor = "end"

# Default lead-in skip = min(_MAX_LEAD, clip_seconds * _LEAD_FRACTION). The
# fraction (< 1.0) keeps the skip well inside the clip; the cap stops a very long
# opening clip from skipping an absurd amount.
_LEAD_FRACTION = 0.12
_MAX_LEAD = 1.5

# open_style may perturb the lead WITHIN this band (seconds, one-sided upward).
# The seed is bounded to [0, _STYLE_BAND] so a style word only nudges pacing; it
# never flips the anchor and never dominates the duration-derived base.
_STYLE_BAND = 0.25

# Identity map for narrowing an arbitrary author string into an ``Anchor``
# (typed narrowing without a cast/ignore); anything else ⇒ no authored anchor.
_ANCHOR_BY_NAME: dict[str, Anchor] = {"start": "start", "end": "end"}

# --- Peak-motion tunables (ORCH-1.3c) ------------------------------------------
# Cut this many seconds BEFORE the peak sample, so the viewer watches the motion
# happen rather than landing mid-gesture. The curve's own resolution is the frame
# extraction interval (coarse, ~sub-second), so a small pre-roll also absorbs the
# quantisation error of "when" the peak really was.
_PEAK_PRE_ROLL = 0.35

# A peak below this is not action, it is sensor noise on a static screen. Motion is
# a 0..1 mean absolute frame delta, so a still UI sits near 0.
_MIN_PEAK_MOTION = 0.02

# The peak must also STAND OUT: at least this multiple of the curve's mean. A clip
# whose motion is uniform (a video playing throughout, a continuous scroll) has no
# single "moment" — argmaxing it would cut to an arbitrary sample, which is strictly
# worse than the settled-tail default. Prominence is what makes this honest.
_PEAK_PROMINENCE = 1.5


# ---------------------------------------------------------------------------
# Coercion helpers — degrade safely, never raise.
# ---------------------------------------------------------------------------


def _finite_number(value: object) -> float | None:
    """Coerce ``value`` to a finite float, else ``None``.

    ``bool`` is rejected (a flag is not a duration) and non-finite floats
    (``nan``/``inf``) are treated as garbage so callers degrade to the safe
    fallback instead of propagating a poisoned value.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    return None


def _motion_xy(sample: object) -> tuple[float, float] | None:
    """Normalise ONE motion sample to a finite ``(t, motion)`` pair, else ``None``.

    Accepts BOTH shapes the ``motion_timeline`` data arrives in across this codebase:
    a ``Mapping`` (raw JSON row off an untyped analyzer blob, keys ``t``/``motion``)
    and a typed :class:`VideoAnalyzerMotionSample` pydantic object (attributes ``t``/
    ``motion``). Either coerces through :func:`_finite_number`, so a missing key,
    absent attribute, non-numeric, ``bool``, ``NaN``/``inf`` or wrong-type entry
    degrades to ``None`` and the caller skips it — it never raises.
    """
    if isinstance(sample, Mapping):
        row = cast("Mapping[str, object]", sample)
        t_raw, m_raw = row.get("t"), row.get("motion")
    else:
        t_raw, m_raw = getattr(sample, "t", None), getattr(sample, "motion", None)
    t, motion = _finite_number(t_raw), _finite_number(m_raw)
    if t is None or motion is None:
        return None
    return (t, motion)


def _default_lead(clip_seconds: float) -> float:
    """Duration-derived load-in skip: ``min(_MAX_LEAD, clip * _LEAD_FRACTION)``."""
    return round(min(_MAX_LEAD, clip_seconds * _LEAD_FRACTION), 3)


def _peak_motion_lead(
    motion_timeline: Sequence[object] | None,
    clip_seconds: float,
) -> float | None:
    """The lead that lands on the clip's peak of motion, or ``None`` if there isn't one.

    ``None`` — meaning "no usable signal, use the duration-derived default" — is
    returned whenever the curve cannot honestly localise a moment:

    * no curve at all (QA disabled, or a video_analyzer too old to emit one);
    * fewer than 2 samples (nothing to compare a peak against);
    * a peak below :data:`_MIN_PEAK_MOTION` (a static screen — that's noise, not action);
    * a peak under :data:`_PEAK_PROMINENCE` x the mean (motion is *uniform*, so no
      single moment is "the" moment; argmax would pick an arbitrary sample).

    Ties resolve to the EARLIEST peak (strict ``>``), which is both deterministic and
    the better cut: an earlier lead leaves more footage after it. The returned lead is
    the peak minus :data:`_PEAK_PRE_ROLL`, clamped to ``[0, clip_seconds]``.
    """
    if not motion_timeline:
        return None
    samples: list[tuple[float, float]] = []
    for entry in motion_timeline:
        xy = _motion_xy(entry)  # accepts a Mapping row OR a typed motion sample
        if xy is None:
            continue
        t, motion = xy
        if t < 0 or t > clip_seconds:
            continue
        samples.append((t, motion))
    if len(samples) < 2:
        return None

    peak_t, peak_motion = samples[0]
    for t, motion in samples[1:]:
        if motion > peak_motion:  # strict ⇒ earliest peak wins a tie
            peak_t, peak_motion = t, motion

    if peak_motion < _MIN_PEAK_MOTION:
        return None
    mean_motion = sum(motion for _, motion in samples) / len(samples)
    if peak_motion < mean_motion * _PEAK_PROMINENCE:
        return None

    return round(min(max(peak_t - _PEAK_PRE_ROLL, 0.0), clip_seconds), 3)


def _style_offset(open_style: str, clip_seconds: float) -> float:
    """Deterministic bounded lead perturbation in ``[0, _STYLE_BAND]``.

    Keyed on ``sha256(open_style|clip_seconds)`` so distinct style words can
    map to distinct leads while a given ``(style, clip)`` pair is stable. Never
    touches the anchor.
    """
    digest = hashlib.sha256(f"{open_style}|{clip_seconds}".encode()).digest()
    return round((digest[0] / 255.0) * _STYLE_BAND, 3)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def cold_open_window(
    clip_seconds: float,
    hook: Mapping[str, object] | None,
    open_style: str | None,
    motion_timeline: Sequence[Mapping[str, object]] | None = None,
) -> tuple[Anchor, float]:
    """Return ``(clip_anchor, clip_lead_seconds)`` for the opening shot's framing.

    Resolution ladder (fully deterministic, no randomness beyond sha256 seeding):

    1. **Authored wins.** When ``hook`` carries ``cold_open_anchor`` in
       ``{"start", "end"}`` and/or a numeric ``cold_open_lead_seconds`` ≥ 0,
       those are honoured (the lead clamped to ``[0, clip_seconds]``). A partial
       author (only one of the two) fills the missing half from the default.
    2. **Measured peak motion** (ORCH-1.3c), when ``motion_timeline`` localises a
       distinguished peak (see :func:`_peak_motion_lead`) and the author gave no lead:
       lead = that peak (minus a small pre-roll), anchor = ``"start"``.

       The anchor is forced to ``"start"`` and this is **load-bearing, not cosmetic**:
       the compiler honours ``clip_lead_seconds`` only under a ``"start"`` anchor
       (``renderer._clip_seek_args`` — an ``"end"`` anchor takes the tail and ignores
       the lead outright), so emitting a measured peak against the default ``"end"``
       anchor would silently discard it. An author who explicitly asked for ``"end"``
       still gets ``"end"`` — and therefore, correctly, no peak seek.
    3. **Deterministic default** (no author, no usable curve): anchor = ``"end"``
       (the settled tail), lead = :func:`_default_lead` off the real duration.
    4. **open_style** may nudge the *default* lead within ``[0, _STYLE_BAND]``
       via :func:`_style_offset`; ``open_style is None`` equals the plain
       default. Style NEVER changes the anchor, NEVER overrides an authored lead,
       and NEVER perturbs a **measured** peak — a measurement is not a guess, and
       nudging it off the peak by a hash would defeat the entire point.

    Degrades safely: a non-positive or non-numeric ``clip_seconds`` returns
    ``("end", 0.0)``. Only frames the shot — it does NOT decide its duration; the
    compiler clamps the lead to ``clip_duration - shot_duration``, so a late peak can
    never run the shot off the end of the clip.
    """
    cs = _finite_number(clip_seconds)
    if cs is None or cs <= 0:
        return (_DEFAULT_ANCHOR, 0.0)

    upper = cs  # the lead can never exceed the clip we are framing

    author_anchor: Anchor | None = None
    author_lead: float | None = None
    if isinstance(hook, Mapping):
        raw_anchor = hook.get("cold_open_anchor")
        if isinstance(raw_anchor, str):
            author_anchor = _ANCHOR_BY_NAME.get(raw_anchor)
        lead_candidate = _finite_number(hook.get("cold_open_lead_seconds"))
        if lead_candidate is not None and lead_candidate >= 0:
            author_lead = lead_candidate

    if author_lead is not None:
        anchor: Anchor = author_anchor if author_anchor is not None else _DEFAULT_ANCHOR
        lead = author_lead
        return (anchor, round(min(max(lead, 0.0), upper), 3))

    # No authored lead → try the measured curve before falling back to the guess.
    # Skipped when the author pinned the anchor to "end": the compiler cannot seek to
    # a lead under an "end" anchor, so a peak there would be a lie.
    peak_lead = None if author_anchor == "end" else _peak_motion_lead(motion_timeline, cs)
    if peak_lead is not None:
        return ("start", round(min(max(peak_lead, 0.0), upper), 3))

    anchor = author_anchor if author_anchor is not None else _DEFAULT_ANCHOR
    lead = _default_lead(cs)
    if open_style is not None:
        lead += _style_offset(open_style, cs)

    lead = round(min(max(lead, 0.0), upper), 3)
    return (anchor, lead)


def cold_open_hash(anchor: str, lead: float) -> str:
    """Stable 16-hex digest of a cold-open window — a content-address seam.

    Mirrors ``motion_plan.plan_hash``: canonical ``json.dumps(..., sort_keys=
    True)`` over the two fields, then the first 16 hex chars of its sha256. Same
    window ⇒ same id; any change to anchor or lead changes the id.
    """
    payload = {"anchor": anchor, "lead": lead}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
