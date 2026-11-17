"""Shotlist normalization — the deterministic, author-driven shot path.

When a brief carries ``brief.shotlist`` (an ordered list authored in the Studio
wizard), the relevant act is built EXACTLY in that order with no LLM
script/storyboard/visual-director guessing: one narration line per shot, URL
shots → captured footage/still, slide shots → a rendered slide, narration bound
1:1 to its shot. ``promo`` drives the WHOLE video from the shotlist;
``website_review`` drives its BEFORE act from the shotlist and lets the AI
generate the AFTER (redesign) act.

The crux is a stable join key. Every consuming stage (``*_script``,
``site_capture``, ``promo_storyboard``, the ``*_visual_director`` / ``*_slides``
/ ``*_compile`` stages) calls :func:`normalize_shotlist` independently
(``site_capture`` is a no-dependency leaf that runs in parallel with the script,
so it cannot read the script's ids). Because the line id is a pure function of
the shot's position (:func:`shot_line_id`), every stage agrees on the binding by
construction — render/compile then join footage/slide → narration with no extra
plumbing.

The brief stays a free-form dict; this module only types and validates the
``shotlist`` sub-tree, soft-dropping malformed entries (same ethos as
``site_capture``/``promo_slides``): a bad shot is skipped, the list stays usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from src.services.stages.verbatim import seconds_for

__all__ = [
    "SHOTLIST_KEY",
    "ShotSpec",
    "normalize_shotlist",
    "normalize_slide",
    "shot_line_id",
]

SHOTLIST_KEY = "shotlist"

# Default pause (ms) on a URL shot so a viewer can read it; overridable per shot.
_DEFAULT_HOLD_MS = 2500
_VIEWPORTS = ("desktop", "tablet", "mobile")
_SLIDE_TEMPLATES = ("title", "bullets", "cta")


def shot_line_id(index: int) -> str:
    """The one true join key for shot ``index`` — identical across every stage."""
    return f"shot-{index:03d}"


@dataclass(frozen=True)
class ShotSpec:
    """One normalized shot. ``index``/``line_id`` are the cross-stage join key."""

    index: int
    line_id: str
    kind: str  # "url" | "slide"
    narration: str
    seconds: float
    url: str | None = None
    directives: dict = field(
        default_factory=dict
    )  # url shots: scroll_to/click/hold_ms/wait_for/viewport
    slide: dict = field(
        default_factory=dict
    )  # slide shots: template/eyebrow/title/body/bullets/cta_url


def _normalize_directives(raw: object) -> dict:
    d: dict[str, Any] = cast("dict[str, Any]", raw) if isinstance(raw, dict) else {}
    hold = d.get("hold_ms")
    out: dict = {
        "hold_ms": int(hold) if isinstance(hold, (int, float)) and hold > 0 else _DEFAULT_HOLD_MS,
        "viewport": str(d.get("viewport") or "desktop").strip().lower(),
    }
    if out["viewport"] not in _VIEWPORTS:
        out["viewport"] = "desktop"
    # Optional selectors — kept only when non-empty so the emitted shape matches
    # the canonical contract (no empty strings).
    for key in ("scroll_to", "click", "wait_for"):
        val = str(d.get(key) or "").strip()
        if val:
            out[key] = val
    return out


def normalize_slide(raw: object) -> dict:
    """Canonicalise a slide sub-tree: a valid template plus only non-empty fields.

    Public because ``website_review_slide_plan`` authors slide content that never
    passes through ``brief.shotlist`` (it is keyed by SCRIPT line id, not by shot
    position) and must still emit the identical canonical shape — otherwise the two
    producers drift and ``_shotlist_deck_slide`` projects them differently.
    """
    s: dict[str, Any] = cast("dict[str, Any]", raw) if isinstance(raw, dict) else {}
    template = str(s.get("template") or "title").strip().lower()
    if template not in _SLIDE_TEMPLATES:
        template = "title"
    out: dict = {"template": template}
    for key in ("eyebrow", "title", "body", "cta_url"):
        val = str(s.get(key) or "").strip()
        if val:
            out[key] = val
    bullets = [str(b).strip() for b in (s.get("bullets") or []) if str(b).strip()]
    if bullets:
        out["bullets"] = bullets
    return out


def normalize_shotlist(brief: dict) -> list[ShotSpec]:
    """Parse ``brief.shotlist`` into ordered :class:`ShotSpec`s (``[]`` if absent).

    Returns ``[]`` when the key is missing/not a list/empty, so every caller can
    cheaply fall back to the legacy LLM path. Entries are kept in order; an entry
    that is not a dict or has blank ``narration`` is dropped. The line id is
    assigned over the KEPT entries (sequential, no gaps), so all stages —
    re-normalizing the same brief — agree on the join key.
    """
    raw = brief.get(SHOTLIST_KEY)
    if not isinstance(raw, list):
        return []
    shots: list[ShotSpec] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        narration = str(entry.get("narration") or "").strip()
        if not narration:
            continue
        kind = str(entry.get("kind") or ("url" if entry.get("url") else "slide")).strip().lower()
        if kind not in ("url", "slide"):
            continue
        index = len(shots)
        line_id = shot_line_id(index)
        seconds = seconds_for(narration)
        if kind == "url":
            shots.append(
                ShotSpec(
                    index=index,
                    line_id=line_id,
                    kind="url",
                    narration=narration,
                    seconds=seconds,
                    url=str(entry.get("url") or "").strip(),
                    directives=_normalize_directives(entry.get("directives")),
                )
            )
        else:
            shots.append(
                ShotSpec(
                    index=index,
                    line_id=line_id,
                    kind="slide",
                    narration=narration,
                    seconds=seconds,
                    slide=normalize_slide(entry.get("slide")),
                )
            )
    return shots
