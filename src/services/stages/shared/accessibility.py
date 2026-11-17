"""Accessibility Tier-1 resolver — deterministic contrast/backplate/safe-area/
silent-gap insights (ORCH-0.8).

Pure helpers, no DB, no client, no ``ctx`` — every function is a total function
of its arguments, so the same shot/clip list always yields byte-identical output
and the module can never issue a render or a network call. Mirrors the frozen
``@dataclass`` + free-function style of :mod:`finish` and :mod:`retention`.

The whole module is **presence-is-opt-in**: :func:`build_a11y_policy` returns
``None`` for a snapshot WITHOUT an ``accessibility`` block, and every consumer
(shot projection, the growth-bundle insight) skips its work when the policy is
absent — so a run that never opted in is byte-identical to before this change.
No function here gates a run: the report is advisory, and the shot-level
``a11y_exempt`` / ``flash_exempt`` flags are only ever stamped ``False`` (never a
gate) and only when a policy is present.

Contrast math is imported verbatim from :mod:`w2v_common.media.contrast` — this
module NEVER forks the WCAG luminance/ratio formulas; it only decides *where* a
backplate is needed and what fill to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from w2v_common.media.contrast import (
    contrast_ratio,
    parse_color,
    passes_wcag,
)

from src.services.stages.shared.provenance import clamp

__all__ = [
    "A11yPolicy",
    "A11yReport",
    "TextRegion",
    "a11y_report",
    "build_a11y_policy",
    "choose_backplate",
    "silent_gaps",
    "text_region_geometry",
]

_COMPUTED_BY = "accessibility"

# Default lower-third caption band (fractions of frame). A shot carries no explicit
# text geometry in the v3 shot dict, so a deterministic default box stands in — the
# same shot always resolves to the same region.
_DEFAULT_REGION = {"x": 0.1, "y": 0.8, "w": 0.8, "h": 0.12}
_DEFAULT_TEXT_COLOR = "#ffffff"
_DEFAULT_BG_COLOR = "#000000"


@dataclass(frozen=True)
class A11yPolicy:
    """Resolved accessibility policy. ``backplate_opacity`` clamps 0..1 on
    construction (via :func:`provenance.clamp`) so an out-of-range value can never
    reach a fill string. Field names/defaults match the ORCH-0.8 contract."""

    wcag_level: str = "AA"
    large_text: bool = False
    min_ratio: float = 4.5
    backplate_opacity: float = 0.6  # clamp 0..1
    backplate_color: str = "#000000"
    min_silent_gap_ms: int = 170

    def __post_init__(self) -> None:
        # Frozen dataclass: write the clamped value back via object.__setattr__.
        resolved = clamp(float(self.backplate_opacity), 0.0, 1.0, computed_by=_COMPUTED_BY)
        object.__setattr__(self, "backplate_opacity", resolved.value)


@dataclass(frozen=True)
class TextRegion:
    """One on-screen text region: geometry (frame fractions) + fg/bg colours."""

    line_id: str
    x: float
    y: float
    w: float
    h: float
    text_color: str
    bg_color: str


@dataclass(frozen=True)
class A11yReport:
    """Deterministic accessibility read-out for one shot/clip list.

    ``insights`` is the flat, human-readable, JSON-serializable advisory list.
    ``failing_line_ids`` are the shots whose text region fails WCAG at the policy
    floor. ``backplates`` maps those line ids → the resolved backplate fill.
    An absent policy or empty shots ⇒ an all-empty report (byte-identical opt-out).
    """

    insights: list[str]
    failing_line_ids: list[str]
    backplates: dict[str, str]


# ------------------------------------------------------------------ geometry ---
def text_region_geometry(shot: dict[str, Any]) -> TextRegion:
    """Resolve a shot's text region deterministically.

    Geometry falls to the default lower-third caption band; colours read from the
    shot's ``text_color`` / ``bg_color`` when present, else the neutral defaults.
    Pure: the same shot dict always yields the same region.
    """
    return TextRegion(
        line_id=str(shot.get("line_id") or shot.get("id") or ""),
        x=float(shot.get("text_x") or _DEFAULT_REGION["x"]),
        y=float(shot.get("text_y") or _DEFAULT_REGION["y"]),
        w=float(shot.get("text_w") or _DEFAULT_REGION["w"]),
        h=float(shot.get("text_h") or _DEFAULT_REGION["h"]),
        text_color=str(shot.get("text_color") or _DEFAULT_TEXT_COLOR),
        bg_color=str(shot.get("bg_color") or _DEFAULT_BG_COLOR),
    )


# ----------------------------------------------------------------- backplate ---
def _backplate_fill(policy: A11yPolicy) -> str:
    """Deterministic ``rgba(...)`` fill from the policy colour + clamped opacity."""
    r, g, b = parse_color(policy.backplate_color)
    return f"rgba({r},{g},{b},{policy.backplate_opacity})"


def choose_backplate(
    fg: str | tuple[int, int, int],
    bg: str | tuple[int, int, int],
    policy: A11yPolicy,
) -> str | None:
    """Return a backplate fill ONLY when ``fg`` on ``bg`` fails the WCAG floor.

    Returns ``None`` when the pair already passes at the policy's level/size (no
    backplate needed); otherwise the ``rgba(...)`` fill that sits behind the text
    to restore legibility. A backplate is emitted strictly below the floor.
    """
    ratio = contrast_ratio(fg, bg)
    if passes_wcag(ratio, large=policy.large_text, level=policy.wcag_level):
        return None
    return _backplate_fill(policy)


# --------------------------------------------------------------- silent gaps ---
def _clip_bounds_ms(clip: dict[str, Any]) -> tuple[float, float] | None:
    """(start_ms, end_ms) for a clip, accepting ``*_ms`` or seconds keys.

    Returns ``None`` when the clip carries no usable timing (so it can't bound a
    gap). ``start_ms``/``end_ms`` win; else ``start``/``end`` (seconds → ms).
    """
    if clip.get("start_ms") is not None and clip.get("end_ms") is not None:
        return float(clip["start_ms"]), float(clip["end_ms"])
    if clip.get("start") is not None and clip.get("end") is not None:
        return float(clip["start"]) * 1000.0, float(clip["end"]) * 1000.0
    return None


def silent_gaps(clips: list[dict[str, Any]], *, min_gap_ms: int) -> list[dict[str, Any]]:
    """Gaps of silence between consecutive timed clips ≥ ``min_gap_ms``.

    A "silent gap" is the interval between one clip's end and the next clip's
    start. Only gaps at or above ``min_gap_ms`` are reported (boundary inclusive).
    Returns ``[{"after_index", "after_line_id", "gap_ms"}, ...]`` in clip order.
    Empty/untimed clips → empty list.
    """
    gaps: list[dict[str, Any]] = []
    prior: tuple[float, float] | None = None
    prior_index = -1
    prior_line = ""
    for index, clip in enumerate(clips or []):
        bounds = _clip_bounds_ms(clip)
        if bounds is None:
            continue
        start, end = bounds
        if prior is not None:
            gap = start - prior[1]
            if gap >= float(min_gap_ms):
                gaps.append(
                    {
                        "after_index": prior_index,
                        "after_line_id": prior_line,
                        "gap_ms": round(gap, 3),
                    }
                )
        prior = (start, end)
        prior_index = index
        prior_line = str(clip.get("line_id") or clip.get("id") or "")
    return gaps


# --------------------------------------------------------------- full report ---
def a11y_report(
    shots: list[dict[str, Any]],
    clips: list[dict[str, Any]],
    policy: A11yPolicy | None,
) -> A11yReport:
    """Compose the full deterministic accessibility read-out.

    An absent policy (``None``) or empty ``shots`` ⇒ an all-empty report, so an
    opted-out run is byte-identical. Otherwise: every shot whose text region fails
    the WCAG floor contributes a failing line id + backplate + insight, and every
    over-threshold silent gap between clips contributes an insight. Advisory only —
    nothing here gates a run.
    """
    if policy is None or not shots:
        return A11yReport(insights=[], failing_line_ids=[], backplates={})

    insights: list[str] = []
    failing_line_ids: list[str] = []
    backplates: dict[str, str] = {}

    for shot in shots:
        region = text_region_geometry(shot)
        fill = choose_backplate(region.text_color, region.bg_color, policy)
        if fill is None:
            continue
        ratio = contrast_ratio(region.text_color, region.bg_color)
        failing_line_ids.append(region.line_id)
        backplates[region.line_id] = fill
        insights.append(
            f"line {region.line_id}: text {region.text_color} on {region.bg_color} "
            f"contrasts {ratio:.2f}:1, below the {policy.wcag_level} floor "
            f"(min {policy.min_ratio:.1f}:1) — backplate applied"
        )

    for gap in silent_gaps(clips or [], min_gap_ms=policy.min_silent_gap_ms):
        insights.append(
            f"silent gap of {gap['gap_ms']:.0f}ms after clip {gap['after_line_id']} "
            f"exceeds the {policy.min_silent_gap_ms}ms bound — consider a caption hold"
        )

    return A11yReport(
        insights=insights,
        failing_line_ids=failing_line_ids,
        backplates=backplates,
    )


# ---------------------------------------------------------------- policy build ---
def build_a11y_policy(config_snapshot: dict[str, Any]) -> A11yPolicy | None:
    """Build an :class:`A11yPolicy` from a run's ``config_snapshot``.

    Presence-is-opt-in: an ABSENT (or non-dict) ``accessibility`` block returns
    ``None`` so every consumer skips its work and the output stays byte-identical.
    A present block yields a policy; each knob reads from the block with the
    contract default when unset. ``backplate_opacity`` clamps 0..1 on construction.
    """
    a11y = config_snapshot.get("accessibility")
    if not isinstance(a11y, dict):
        return None
    return A11yPolicy(
        wcag_level=str(a11y.get("wcag_level", "AA")),
        large_text=bool(a11y.get("large_text", False)),
        min_ratio=float(a11y.get("min_ratio", 4.5)),
        backplate_opacity=float(a11y.get("backplate_opacity", 0.6)),
        backplate_color=str(a11y.get("backplate_color", "#000000")),
        min_silent_gap_ms=int(a11y.get("min_silent_gap_ms", 170)),
    )
