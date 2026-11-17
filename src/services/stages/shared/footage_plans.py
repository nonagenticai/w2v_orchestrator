"""Blueprint-agnostic resolution of the footage-direction shot-plan (ORCH-3.3).

Both the demo_video compile and the promo pipeline capture per-clip focus/cursor
tracks and then apply an authored shot-plan on top of them. The plan is INTENT
only: a zoom-OFF co-removes its focus+cursor tracks, a ``static`` reframe drops
the cursor, a ``split`` names a second clip whose URL is resolved compile-side
(never the LLM), and a callout is placed ONLY at the bbox captured in the
recording's focus track — never at a coordinate the LLM invented. Absent (the
stage skipped ⇒ empty ``plan_by_line``) the whole seam is inert and the captured
tracks ride through byte-identically.

Kept pure so a later L4 critique pass can nudge a plan without touching either
pipeline's compile stage.
"""

from __future__ import annotations

from typing import cast

from src.models.showcase_line import normalize_clip_key
from src.services.stages.shot_assembly import interacted_bbox


def split_second_clip_tags(lines: list[dict[str, object]]) -> set[str]:
    """Footage tags an author-pinned split names as ``second_clip`` that are NOT a line id.

    A ``second_clip`` is a footage-bearing line id OR a recorded ``clip-<n>`` tag. A tag
    no line plays over would be dropped by ``demo_video_record`` (it keeps only the tags a
    line references), so the split's second pane would silently vanish. The record stage
    adds these to the tags it keeps."""
    line_ids = {str(line["id"]) for line in lines}
    out: set[str] = set()
    for line in lines:
        split = line.get("split")
        if isinstance(split, dict) and split.get("second_clip"):
            second = normalize_clip_key(str(split["second_clip"]))
            if second and second not in line_ids:
                out.add(second)
    return out


def _resolve_split(
    raw_split: object,
    video_by_line: dict[str, str],
    line_id: str,
    splits: dict[str, dict[str, object]],
    video_by_clip: dict[str, str] | None = None,
) -> None:
    """Resolve a footage-direction ``split`` plan (T8) into a compiler
    ``Shot.split = {layout, second_source_url}``, stamped into ``splits`` (no-op
    when there is nothing valid to stamp — keeps ``build_shots`` branch-light).

    The director names another ``line_id`` as ``second_clip``; the URL is resolved
    HERE (never the LLM) via ``video_by_line`` (keyed by line id, holds only
    footage-bearing lines). A dangling ref (no such footage line) or a self-split
    (a clip beside itself) is dropped — never emit a source that can't be fetched,
    mirroring the callout-bbox drop. Split is orthogonal to zoom, so this is
    resolved regardless of a line's zoom state.
    """
    if not isinstance(raw_split, dict):
        return
    raw_d = cast(dict[str, object], raw_split)
    second = normalize_clip_key(str(raw_d.get("second_clip") or ""))
    # A line id first; else a recorded clip tag (an authored split may name either).
    url = video_by_line.get(second) or (video_by_clip or {}).get(second)
    if not url or second == line_id:
        return
    splits[line_id] = {"layout": str(raw_d.get("layout") or ""), "second_source_url": url}


def authored_split_plans(lines: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Plans pinned by the AUTHOR: ``{line_id: {"line_id", "split"}}`` for every line
    whose spine entry carries a ``split`` (``AuthoredSplit``: ``layout`` + ``second_clip``).

    Intent only, like any plan: the second clip's URL is still resolved compile-side by
    :func:`_resolve_split`, so a dangling ``second_clip`` is dropped, never fetched.
    """
    out: dict[str, dict[str, object]] = {}
    for line in lines:
        split = line.get("split")
        if isinstance(split, dict) and split.get("layout") and split.get("second_clip"):
            line_id = str(line["id"])
            out[line_id] = {"line_id": line_id, "split": split}
    return out


def pin_authored_splits(
    plan_by_line: dict[str, dict[str, object]],
    authored: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """``plan_by_line`` with each authored split laid over the director's plan for that
    line — the author's ``split`` WINS (the rest of the director's plan for the line,
    zoom/callout, is kept). A new dict; the inputs are not mutated."""
    out = dict(plan_by_line)
    for line_id, pinned in authored.items():
        out[line_id] = {**out.get(line_id, {}), "line_id": line_id, "split": pinned["split"]}
    return out


def resolve_footage_plans(
    *,
    line_ids: list[str],
    plan_by_line: dict[str, dict[str, object]],
    focus_track_by_line: dict[str, list[dict[str, object]]] | None,
    cursor_track_by_line: dict[str, list[dict[str, object]]] | None,
    video_by_line: dict[str, str],
    video_by_clip: dict[str, str] | None = None,
) -> tuple[
    dict[str, list[dict[str, object]]] | None,
    dict[str, list[dict[str, object]]] | None,
    dict[str, dict[str, object]] | None,
    dict[str, dict[str, object]] | None,
]:
    """Apply the authored shot-plan on top of the captured focus/cursor tracks.

    Returns ``(focus_map, cursor_map, callouts_by_line, split_by_line)`` — each
    ``None`` when it holds nothing. With no plan the two input maps are returned
    UNCHANGED (identity, the byte-identical/absent path) and both derived maps are
    ``None``. The callout bbox is read from the ORIGINAL (pre-pop) focus track, so
    a callout on a line whose zoom is left on still lands on its captured target.
    """
    if not plan_by_line:
        return focus_track_by_line, cursor_track_by_line, None, None

    focus_map = dict(focus_track_by_line or {})
    cursor_map = dict(cursor_track_by_line or {})
    callouts: dict[str, dict[str, object]] = {}
    splits: dict[str, dict[str, object]] = {}
    for line_id in line_ids:
        plan = plan_by_line.get(line_id)
        if not plan:
            continue
        # Split is orthogonal to zoom — resolve it BEFORE the zoom-off gate below.
        # `_validate` forces zoom="off" on a no-click line, and a before/after
        # comparison beat has no click, so gating split on zoom would always drop
        # it. Footage-gated compile-side (the ref must resolve to a footage-bearing
        # line via `video_by_line`).
        _resolve_split(plan.get("split"), video_by_line, line_id, splits, video_by_clip)
        # zoom OFF ⇒ co-remove focus + cursor + emit NO callout (a cursor easing to
        # a no-zoom target, or a callout on nothing, is the bug).
        if plan.get("zoom") == "off":
            focus_map.pop(line_id, None)
            cursor_map.pop(line_id, None)
            continue
        # static reframe ⇒ no synthetic cursor for this line.
        if plan.get("reframe") == "static":
            cursor_map.pop(line_id, None)
        callout = plan.get("callout")
        if callout:
            # Coordinates come ONLY from the captured bbox (pre-pop track) — a
            # callout with no captured target is dropped, never placed at a guess.
            # Only an INTERACTED element is a target (see `interacted_bbox`): the
            # first keyframe may be a zero-filled placeholder or an off-screen node.
            track = (focus_track_by_line or {}).get(line_id) or []
            first = next((k for k in track if interacted_bbox(k) is not None), None)
            bbox = first.get("bbox") if first else None
            if first and bbox:
                callout_d = cast(dict[str, object], callout)
                # Onset = the captured click time of that first keyframe, so the
                # callout fires WHEN the element is used, not at shot-start.
                t0 = first.get("t")
                at = float(t0) if isinstance(t0, (int, float)) and not isinstance(t0, bool) else 0.0
                callouts[line_id] = {
                    "type": callout_d["type"],
                    "label": callout_d["label"],
                    "at": at,
                    "bbox": bbox,
                }
    return (focus_map or None, cursor_map or None, callouts or None, splits or None)
