"""Motion signature + advisory layout planner — pure, deterministic (ORCH-3.1).

The temporal/design sibling of :mod:`design_system`. This module turns *director
intent* (a ``motion.style`` word) and a per-video *seed* into a single motion
**signature** (one easing family, stagger rhythm, entrance/exit direction,
parallax + glow depth) that every card in a video shares, plus a per-card
**timeline** computed from the card's real narration duration so motion fills the
clip instead of freezing at the template's hardcoded ``RENDER_DURATION_MS``.

It also carries an advisory token: a keyword→:func:`pick_icon` glyph. It is
metadata for downstream (a later VP PR); nothing here selects a render template.

**Purity contract (ORCH-3.1 INVARIANT 3):** no ``datetime.now``, no ``random``,
no env reads. The only source of variation is ``seed = sha256(project_id)`` via
:func:`stable_seed`, so a signature is stable across resumes/retries yet differs
between videos. Every dataclass is frozen (sibling convention:
``cadence.CadenceSpec`` / ``compute_plan.MixGainTargets`` — NOT pydantic).

**SCOPE LIMIT (D2/D3):** the advisory ``LayoutKind`` metadata MUST NOT change
which motion template ``motion_slides`` selects — that stays the two real
templates (``animated_slide`` / ``animated_title_card``). The dead ``data_viz`` /
``kinetic`` / ``lower_third`` families do NOT exist in this repo; this module
never references or routes to them. Their activation is a later PR.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "Easing",
    "Entrance",
    "Exit",
    "LayoutKind",
    "MotionPlan",
    "MotionSignature",
    "Style",
    "Timeline",
    "motion_signature",
    "pick_icon",
    "plan_card_timing",
    "plan_hash",
    "stable_seed",
]

# ---------------------------------------------------------------------------
# Vocabulary (spec 04 §T2/§T3 — the "motion" author block enum + layout kinds).
# ---------------------------------------------------------------------------

Style = Literal["calm", "editorial", "punchy", "energetic"]
Easing = Literal["easeOutQuint", "easeOutCubic", "easeOutBack", "easeOutExpo"]
Entrance = Literal["rise", "fade", "wipe", "mask"]
Exit = Literal["fall", "fade", "wipe"]
LayoutKind = Literal[
    "quote",
    "stat",
    "comparison",
    "list",
    "section_divider",
    "timeline",
    "cta",
    "title",
]

_DEFAULT_STYLE: Style = "editorial"

# Identity map for validating an arbitrary ``style`` string into a ``Style``
# (typed narrowing without a cast/ignore); unknown ⇒ the default style.
_STYLE_BY_NAME: dict[str, Style] = {
    "calm": "calm",
    "editorial": "editorial",
    "punchy": "punchy",
    "energetic": "energetic",
}

# Per-style base signature profile. Each style names one easing family, a stagger
# base+jitter span (ms), two candidate entrance/exit directions (the seed picks
# one), and parallax/glow base+span (0..1). The seed only perturbs WITHIN a
# style's band, so a "calm" video always reads calm while two calm videos still
# differ. Fields:  easing, (stagger_base, stagger_span), (entrance_a, entrance_b),
# (exit_a, exit_b), (parallax_base, parallax_span), (glow_base, glow_span).
_STYLE_EASING: dict[Style, Easing] = {
    "calm": "easeOutQuint",
    "editorial": "easeOutCubic",
    "punchy": "easeOutBack",
    "energetic": "easeOutExpo",
}
_STYLE_STAGGER: dict[Style, tuple[int, int]] = {
    "calm": (140, 40),
    "editorial": (100, 40),
    "punchy": (70, 30),
    "energetic": (50, 30),
}
_STYLE_ENTRANCE: dict[Style, tuple[Entrance, Entrance]] = {
    "calm": ("fade", "rise"),
    "editorial": ("rise", "mask"),
    "punchy": ("wipe", "mask"),
    "energetic": ("mask", "wipe"),
}
_STYLE_EXIT: dict[Style, tuple[Exit, Exit]] = {
    "calm": ("fade", "fall"),
    "editorial": ("fade", "wipe"),
    "punchy": ("wipe", "fall"),
    "energetic": ("wipe", "fall"),
}
_STYLE_PARALLAX: dict[Style, tuple[float, float]] = {
    "calm": (0.20, 0.15),
    "editorial": (0.35, 0.20),
    "punchy": (0.55, 0.25),
    "energetic": (0.70, 0.25),
}
_STYLE_GLOW: dict[Style, tuple[float, float]] = {
    "calm": (0.10, 0.10),
    "editorial": (0.20, 0.15),
    "punchy": (0.35, 0.20),
    "energetic": (0.50, 0.25),
}

# Timeline budget split: entrance + exit fractions of the clip; the remainder is
# hold. Constant fractions (< 1.0 combined) so hold is always >= 0 and the three
# integer budgets sum EXACTLY to duration_ms (no-freeze invariant, spec §8.3).
_ENTRANCE_FRAC = 0.22
_EXIT_FRAC = 0.14


@dataclass(frozen=True)
class MotionSignature:
    """One video's motion signature — seeded once, shared by every card.

    ``exit_`` carries a trailing underscore to avoid shadowing the builtin; the
    schema/param layer maps it back to ``exit`` when emitting ``window.PARAMS``.
    """

    easing: Easing
    stagger_ms: int
    entrance: Entrance
    exit_: Exit
    parallax: float
    glow_drift: float


@dataclass(frozen=True)
class Timeline:
    """A single card's computed timeline (absolute ms), derived from clip length.

    Invariant: ``entrance_ms + hold_ms + exit_ms == duration_ms`` (see
    :attr:`total`). ``marks`` are per-element absolute entrance start times so the
    template staggers each layer instead of animating them all at once.
    """

    entrance_ms: int
    hold_ms: int
    exit_ms: int
    stagger_ms: int
    marks: tuple[int, ...]

    @property
    def total(self) -> int:
        """Full timeline span — equals the ``duration_ms`` it was built from."""
        return self.entrance_ms + self.hold_ms + self.exit_ms


@dataclass(frozen=True)
class MotionPlan:
    """The complete per-card motion plan: signature + advisory layout/icon + timeline."""

    signature: MotionSignature
    layout: LayoutKind
    icon: str | None
    timeline: Timeline


# ---------------------------------------------------------------------------
# Seeding — deterministic, no random/env.
# ---------------------------------------------------------------------------


def stable_seed(project_id: str) -> int:
    """Deterministic per-video seed: the first 8 bytes of ``sha256(project_id)``.

    Same project id ⇒ same seed (stable across resumes/retries); distinct project
    ids ⇒ distinct seeds with overwhelming probability. No randomness, no clock.
    """
    digest = hashlib.sha256(project_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _seeded_bytes(style: str, seed: int) -> bytes:
    """A deterministic byte stream keyed by ``(style, seed)`` for signature jitter."""
    return hashlib.sha256(f"{style}|{seed}".encode()).digest()


def motion_signature(style: str, seed: int) -> MotionSignature:
    """Resolve the video-wide motion signature from director ``style`` + ``seed``.

    ``style`` is one of ``calm|editorial|punchy|energetic`` (director intent);
    anything else falls back to ``editorial``. The easing family is fixed per
    style, so two styles always differ; the ``seed`` perturbs stagger, entrance/
    exit direction and parallax/glow WITHIN the style's band, so two videos of the
    same style differ while both still read as that style. Fully deterministic.
    """
    key: Style = _STYLE_BY_NAME.get(style, _DEFAULT_STYLE)
    stream = _seeded_bytes(key, seed)

    stagger_base, stagger_span = _STYLE_STAGGER[key]
    stagger_ms = stagger_base + stream[0] % (stagger_span + 1)

    entrance = _STYLE_ENTRANCE[key][stream[1] % 2]
    exit_ = _STYLE_EXIT[key][stream[2] % 2]

    parallax_base, parallax_span = _STYLE_PARALLAX[key]
    parallax = round(parallax_base + (stream[3] / 255.0) * parallax_span, 3)

    glow_base, glow_span = _STYLE_GLOW[key]
    glow_drift = round(glow_base + (stream[4] / 255.0) * glow_span, 3)

    return MotionSignature(
        easing=_STYLE_EASING[key],
        stagger_ms=stagger_ms,
        entrance=entrance,
        exit_=exit_,
        parallax=parallax,
        glow_drift=glow_drift,
    )


def plan_card_timing(duration_ms: int, n_elements: int, sig: MotionSignature) -> Timeline:
    """Split a card's REAL clip ``duration_ms`` into entrance/hold/exit + stagger.

    Replaces the template's hardcoded ``RENDER_DURATION_MS = 1500 + n*420 + 1200``
    so the internal animation fills the whole clip (no ~4s freeze, spec §8.3). The
    three budgets are integer and sum EXACTLY to ``duration_ms``: ``hold`` absorbs
    the rounding remainder. Per-element ``stagger`` follows the signature's rhythm
    but is capped to the entrance budget so all ``n_elements`` fit inside it.
    """
    d = max(int(duration_ms), 0)
    entrance_ms = round(d * _ENTRANCE_FRAC)
    exit_ms = round(d * _EXIT_FRAC)
    hold_ms = d - entrance_ms - exit_ms

    n = max(int(n_elements), 0)
    stagger_ms = min(sig.stagger_ms, entrance_ms // n) if n > 0 else 0
    marks = tuple(i * stagger_ms for i in range(n))

    return Timeline(
        entrance_ms=entrance_ms,
        hold_ms=hold_ms,
        exit_ms=exit_ms,
        stagger_ms=stagger_ms,
        marks=marks,
    )


# ---------------------------------------------------------------------------
# Iconography (§T6) — deterministic keyword→glyph map, LLM/author override wins.
# ---------------------------------------------------------------------------

# Ordered so the first matching keyword group wins (deterministic). Glyph names
# are the bundled line-icon set ids a later VP PR strokes in ``--accent``.
_ICON_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("secure", "security", "protect", "privacy", "encrypt"), "shield"),
    (("fast", "speed", "instant", "performance", "accelerate"), "bolt"),
    (("grow", "growth", "scale", "increase", "expand"), "trending-up"),
    (("global", "worldwide", "international", "everywhere"), "globe"),
    (("team", "people", "community", "collaborate", "together"), "users"),
    (("money", "cost", "price", "revenue", "budget", "payment"), "dollar"),
    (("time", "schedule", "deadline", "hours"), "clock"),
    (("data", "analytics", "metrics", "insight", "measure"), "chart"),
    (("idea", "innovation", "creative", "smart", "intelligent"), "lightbulb"),
    (("build", "develop", "engineer", "coding", "programming"), "code"),
    (("done", "complete", "success", "verified", "approved"), "check"),
    (("cloud", "hosting", "infrastructure", "server"), "cloud"),
    (("launch", "deploy", "ship", "release"), "rocket"),
    (("premium", "quality", "favorite", "featured"), "star"),
    (("search", "discover", "explore", "find"), "search"),
)


def pick_icon(text: str, override: str | None = None) -> str | None:
    """Deterministic keyword→glyph for a card; an author/planner ``override`` wins.

    Returns the ``override`` verbatim when truthy (an authored or L1-planner icon
    is authoritative), else the first keyword-map hit, else ``None`` (no icon).
    Case-insensitive substring match against ``text``. Pure and order-stable.
    """
    if override:
        return override
    lowered = (text or "").lower()
    for keywords, icon in _ICON_KEYWORDS:
        if any(kw in lowered for kw in keywords):
            return icon
    return None


# ---------------------------------------------------------------------------
# Plan hashing — the seam design_system.token_fingerprint folds on.
# ---------------------------------------------------------------------------


def plan_hash(p: MotionPlan) -> str:
    """Stable 16-hex-char digest of a :class:`MotionPlan` — the media-id seam.

    Consumed by ``design_system.token_fingerprint(...)`` so a deck's
    ``deterministic_media_id`` changes iff the resolved motion plan changes (and
    is byte-identical when no ``motion`` block is present — that path never calls
    this). Deterministic: canonical JSON of every field, sorted keys.
    """
    payload = {
        "signature": {
            "easing": p.signature.easing,
            "stagger_ms": p.signature.stagger_ms,
            "entrance": p.signature.entrance,
            "exit": p.signature.exit_,
            "parallax": p.signature.parallax,
            "glow_drift": p.signature.glow_drift,
        },
        "layout": p.layout,
        "icon": p.icon,
        "timeline": {
            "entrance_ms": p.timeline.entrance_ms,
            "hold_ms": p.timeline.hold_ms,
            "exit_ms": p.timeline.exit_ms,
            "stagger_ms": p.timeline.stagger_ms,
            "marks": list(p.timeline.marks),
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
