"""Pure, deterministic whole-film colorist (ORCH-3.3).

The finishing sibling of :mod:`motion_plan` (temporal signature) and
:mod:`design_system` (design tokens). This module turns *director intent*
(a ``mood`` / ``tone`` word and the ``motion.style`` word) plus the run's
derived design tokens and a per-video seed into a single **whole-film grade** —
one finishing look every shot in the video shares.

**Composition rule (load-bearing).** A per-shot ``ColorAdjust`` (the Studio
re-stitch per-clip edit, ``api/schemas/runs.ColorAdjust`` → ``shot_assembly.
_attach_clip_look``) already owns the tonal dials ``brightness`` / ``contrast`` /
``saturation`` / ``gamma``. The whole-film colorist must NOT touch those or it
would double-apply on top of a user's per-clip grade. It therefore drives ONLY
the *finishing-exclusive* knobs — ``look``, ``shadow_tint``, ``highlight_tint``,
``vignette``, ``grain``, ``glow``, ``sharpen``, ``grade_intensity`` — and leaves
``contrast`` / ``saturation`` / ``gamma`` / ``temperature`` at their identity
defaults on :class:`~src.services.stages.shared.finish.FinishSpec`.

**Purity contract (mirrors :mod:`motion_plan` / :mod:`design_system`):** no
``datetime.now``, no ``random``, no env reads, no httpx/worker/LLM import. The
only source of variation is the ``seed`` (a :func:`provenance.subseed` of the
run's master seed), so the same run always resolves the same grade (stable across
resumes/retries) while two runs of the same mood still differ within a small band.

**Provenance.** Every knob resolves through :func:`provenance.resolve_with_provenance`
with ``computed_by="colorist"``, so a brief-authored knob WINS (``source=
"override"``) and a table-derived one is recorded (``source="default"``). The
resolved :class:`ProvenanceValue` objects are carried on the returned
:class:`Grade` so the audit ledger can explain every finishing decision.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from src.services.stages.shared.provenance import (
    ProvenanceValue,
    master_seed,
    resolve_with_provenance,
    subseed,
)

if TYPE_CHECKING:
    from src.services.stages.shared.design_system import DesignTokens

__all__ = ["Grade", "grade_from_snapshot", "pick_grade"]

_COMPUTED_BY = "colorist"

# The finishing knobs the colorist owns. The tonal dials (contrast/saturation/
# gamma/temperature) are DELIBERATELY absent — per-shot ColorAdjust owns those.
_KNOBS: tuple[str, ...] = (
    "look",
    "shadow_tint",
    "highlight_tint",
    "vignette",
    "grain",
    "glow",
    "sharpen",
    "grade_intensity",
)

# ``look`` stays inside the compiler's existing preset vocabulary (the
# ColorAdjust.preset Literal) plus ``none`` — never an invented look name a
# renderer would silently ignore.
Look = Literal["none", "warm", "cool", "bw", "vivid", "cinematic"]

# The motion-signature style taxonomy, reused verbatim (calm|editorial|punchy|
# energetic) — NOT a new one. Identity map for typed narrowing without a cast.
Style = Literal["calm", "editorial", "punchy", "energetic"]
_DEFAULT_STYLE: Style = "editorial"
_STYLE_BY_NAME: dict[str, Style] = {
    "calm": "calm",
    "editorial": "editorial",
    "punchy": "punchy",
    "energetic": "energetic",
}

# Style scales the OVERALL finishing strength: a calm film wears its grade lightly,
# an energetic one leans in. Applied to grade_intensity + the texture knobs so the
# same mood reads calmer/punchier without changing its character (look/tints).
_STYLE_INTENSITY: dict[Style, float] = {
    "calm": 0.7,
    "editorial": 1.0,
    "punchy": 1.25,
    "energetic": 1.5,
}


@dataclass(frozen=True)
class _GradeProfile:
    """A mood's base finishing look — the pre-style, pre-seed table row."""

    look: Look
    shadow_tint: str
    highlight_tint: str
    vignette: float
    grain: float
    glow: float
    sharpen: float
    grade_intensity: float


# Neutral identity: no look, no film effects. A run whose mood resolves here yields
# a grade FinishSpec.as_finish_payload will not emit (look "none" + all knobs 0).
_NEUTRAL = _GradeProfile(
    look="none",
    shadow_tint="#000000",
    highlight_tint="#ffffff",
    vignette=0.0,
    grain=0.0,
    glow=0.0,
    sharpen=0.0,
    grade_intensity=1.0,
)

# Mood → base finishing look. Keyed off the same director-intent ``mood`` token the
# sound path reads (``mix_gain_targets(..., mood)``); ``style`` (below) modulates
# strength. Tints are neutral-ish film tints (crushed-warm shadows / lifted-cool
# highlights etc.); every ``look`` is in the compiler's preset vocabulary.
_GRADE_BY_MOOD: dict[str, _GradeProfile] = {
    "neutral": _NEUTRAL,
    "warm": _GradeProfile("warm", "#150c02", "#fff3df", 0.16, 0.06, 0.12, 0.10, 0.85),
    "cool": _GradeProfile("cool", "#02060f", "#e7f2ff", 0.16, 0.05, 0.10, 0.14, 0.85),
    "cinematic": _GradeProfile("cinematic", "#04070a", "#f1ead9", 0.32, 0.10, 0.08, 0.12, 0.90),
    "moody": _GradeProfile("cinematic", "#03040a", "#e9e2d6", 0.40, 0.12, 0.06, 0.10, 0.95),
    "vivid": _GradeProfile("vivid", "#000000", "#ffffff", 0.10, 0.02, 0.22, 0.20, 0.80),
    "bold": _GradeProfile("vivid", "#050400", "#fffdf5", 0.14, 0.03, 0.26, 0.24, 0.90),
    "bw": _GradeProfile("bw", "#000000", "#ffffff", 0.22, 0.12, 0.05, 0.16, 1.0),
    "nostalgic": _GradeProfile("warm", "#140a02", "#f7ecd2", 0.28, 0.18, 0.14, 0.06, 0.80),
    "clean": _GradeProfile("none", "#000000", "#ffffff", 0.04, 0.0, 0.06, 0.18, 0.70),
}

# Common author spellings → a canonical mood key (never invents a new look).
_MOOD_ALIASES: dict[str, str] = {
    "none": "neutral",
    "default": "neutral",
    "mono": "bw",
    "monochrome": "bw",
    "b&w": "bw",
    "black-and-white": "bw",
    "cinema": "cinematic",
    "filmic": "cinematic",
    "warmth": "warm",
    "cold": "cool",
    "energetic": "vivid",
    "punchy": "vivid",
    "retro": "nostalgic",
    "vintage": "nostalgic",
    "minimal": "clean",
    "corporate": "clean",
}

# ``grain`` / ``glow`` get a small seeded jitter so two runs of the same mood are
# not pixel-identical in texture, while staying firmly within the mood's band (the
# same "perturb within the band" idea as motion_signature). One-sided upward, small.
_JITTER_BAND = 0.05


def _norm(value: str | None) -> str:
    """Lowercase/strip a token; ``None`` → empty string."""
    return (value or "").strip().lower()


def _resolve_profile(
    mood: str | None, tone: str | None, design_tokens: DesignTokens | None
) -> _GradeProfile:
    """Pick the base :class:`_GradeProfile` from ``mood`` (then ``tone``) tokens.

    ``mood`` is primary (it matches the sound path's director-intent token); when
    it is absent/unknown the secondary ``tone`` word is tried, then the neutral
    identity. The special ``brand`` mood/tone derives its tints from the run's
    :class:`DesignTokens` palette so the film's finishing reads on-brand.
    """
    for word in (_norm(mood), _norm(tone)):
        if not word:
            continue  # absent mood must defer to tone, not short-circuit to neutral
        if word == "brand":
            return _brand_profile(design_tokens)
        canonical = _MOOD_ALIASES.get(word, word)
        profile = _GRADE_BY_MOOD.get(canonical)
        if profile is not None:
            return profile
    return _NEUTRAL


def _brand_profile(design_tokens: DesignTokens | None) -> _GradeProfile:
    """A cinematic grade whose tints follow the run's brand palette.

    Falls back to the plain ``cinematic`` mood when no design tokens are present
    (no ``design`` author-block), so ``brand`` never crashes an un-tokenised run.
    """
    base = _GRADE_BY_MOOD["cinematic"]
    if design_tokens is None:
        return base
    palette = design_tokens.palette
    return _GradeProfile(
        look=base.look,
        # Crush shadows toward the brand surface, lift highlights toward the soft
        # accent — a subtle brand cast, not a full tonal remap (that stays identity).
        shadow_tint=palette.surface,
        highlight_tint=palette.accent_soft,
        vignette=base.vignette,
        grain=base.grain,
        glow=base.glow,
        sharpen=base.sharpen,
        grade_intensity=base.grade_intensity,
    )


@dataclass(frozen=True)
class Grade:
    """A resolved whole-film grade — the finishing-exclusive knobs + its ledger.

    ``look`` / ``shadow_tint`` / ``highlight_tint`` are strings; the rest are 0..1
    floats (``grade_intensity`` the overall strength). ``provenance`` records where
    each knob came from (``override`` for a brief-authored knob, ``default`` for a
    table-derived one) via :class:`ProvenanceValue`. ``is_active`` is the gate the
    finish payload uses: a grade with no look and no film effects emits nothing.
    """

    look: str
    shadow_tint: str
    highlight_tint: str
    vignette: float
    grain: float
    glow: float
    sharpen: float
    grade_intensity: float
    provenance: tuple[ProvenanceValue, ...]

    @property
    def is_active(self) -> bool:
        """True iff this grade actually changes the picture (a look OR any effect)."""
        return self.look != "none" or any(
            v > 0.0 for v in (self.vignette, self.grain, self.glow, self.sharpen)
        )

    def knobs(self) -> dict[str, str | float]:
        """The finishing-exclusive knob map to overlay onto a :class:`FinishSpec`."""
        return {
            "look": self.look,
            "shadow_tint": self.shadow_tint,
            "highlight_tint": self.highlight_tint,
            "vignette": self.vignette,
            "grain": self.grain,
            "glow": self.glow,
            "sharpen": self.sharpen,
            "grade_intensity": self.grade_intensity,
        }


def _seeded_jitter(seed: int, knob: str) -> float:
    """A deterministic ``[0, _JITTER_BAND]`` nudge keyed by ``(seed, knob)``."""
    digest = hashlib.sha256(f"{seed}|{knob}".encode()).digest()
    return round((digest[0] / 255.0) * _JITTER_BAND, 4)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def pick_grade(
    *,
    tone: str | None,
    mood: str | None,
    style: str | None,
    design_tokens: DesignTokens | None,
    brief_grade: Mapping[str, object] | None,
    seed: int,
) -> Grade:
    """Resolve the whole-film :class:`Grade` for a run — pure + deterministic.

    Resolution per knob is the C14 provenance ladder: a brief-authored value in
    ``brief_grade`` WINS (``source="override"``); otherwise the table value
    (``mood``/``tone`` profile, scaled by ``style`` strength, with a small seeded
    texture jitter) is recorded (``source="default"``). Only the finishing-exclusive
    knobs are driven — contrast/saturation/gamma/temperature are never touched.

    ``brief_grade`` is the author's ``finish.grade`` override map (``None`` when the
    author set nothing). ``seed`` is a :func:`provenance.subseed` so re-runs
    reproduce the grade. ``design_tokens`` is consulted only for the ``brand`` mood.
    """
    overrides = brief_grade or {}
    profile = _resolve_profile(mood, tone, design_tokens)

    if profile is _NEUTRAL:
        # A neutral (no-mood/unknown) run stays FULLY identity — no style scaling,
        # no seeded jitter. This is load-bearing: a jitter on grain/glow would make
        # `Grade.is_active` True and leak grade knobs into the payload for a run
        # that never asked for a grade, breaking byte-identity. A brief-authored
        # override below can still turn an individual knob on.
        computed: dict[str, str | float] = {
            "look": _NEUTRAL.look,
            "shadow_tint": _NEUTRAL.shadow_tint,
            "highlight_tint": _NEUTRAL.highlight_tint,
            "vignette": 0.0,
            "grain": 0.0,
            "glow": 0.0,
            "sharpen": 0.0,
            "grade_intensity": _NEUTRAL.grade_intensity,
        }
    else:
        # Style scales overall strength + the film-texture knobs; character
        # (look/tints) is style-independent. Seeded jitter nudges grain/glow within
        # the mood's band so two runs of the same mood differ in texture.
        style_key = _STYLE_BY_NAME.get(_norm(style), _DEFAULT_STYLE)
        factor = _STYLE_INTENSITY[style_key]
        computed = {
            "look": profile.look,
            "shadow_tint": profile.shadow_tint,
            "highlight_tint": profile.highlight_tint,
            "vignette": round(_clamp01(profile.vignette * factor), 4),
            "grain": round(_clamp01(profile.grain * factor + _seeded_jitter(seed, "grain")), 4),
            "glow": round(_clamp01(profile.glow * factor + _seeded_jitter(seed, "glow")), 4),
            "sharpen": round(_clamp01(profile.sharpen * factor), 4),
            "grade_intensity": round(_clamp01(profile.grade_intensity * factor), 4),
        }

    resolved: dict[str, str | float] = {}
    ledger: list[ProvenanceValue] = []
    for knob in _KNOBS:
        if knob in overrides and overrides[knob] is not None:
            pv = resolve_with_provenance(
                knob, override=overrides[knob], default=computed[knob], computed_by=_COMPUTED_BY
            )
        else:
            pv = resolve_with_provenance(knob, default=computed[knob], computed_by=_COMPUTED_BY)
        resolved[knob] = pv.value
        ledger.append(pv)

    return Grade(
        look=str(resolved["look"]),
        shadow_tint=str(resolved["shadow_tint"]),
        highlight_tint=str(resolved["highlight_tint"]),
        vignette=float(resolved["vignette"]),
        grain=float(resolved["grain"]),
        glow=float(resolved["glow"]),
        sharpen=float(resolved["sharpen"]),
        grade_intensity=float(resolved["grade_intensity"]),
        provenance=tuple(ledger),
    )


def _dict(value: object) -> dict[str, Any]:
    """``value`` as a dict, or ``{}`` — degrade safely on unvalidated config blocks."""
    return {str(k): v for k, v in value.items()} if isinstance(value, dict) else {}


def _first_token(*candidates: object) -> str | None:
    """First truthy string in ``candidates`` (director-intent token resolution)."""
    for cand in candidates:
        if isinstance(cand, str) and cand.strip():
            return cand
    return None


def resolve_mood(config_snapshot: dict[str, object]) -> str | None:
    """The run-level director-intent ``mood`` token, or ``None`` when unset.

    Precedence ``finish → direction → sound_effects → top-level`` — the SINGLE
    source of truth for the run's mood, so the colour grade (via
    :func:`grade_from_snapshot`) and the caption style (forwarded to the compiler
    as ``caption_mood``) always read the identical token with no drift.
    """
    cfg = config_snapshot or {}
    return _first_token(
        _dict(cfg.get("finish")).get("mood"),
        _dict(cfg.get("direction")).get("mood"),
        _dict(cfg.get("sound_effects")).get("mood"),
        cfg.get("mood"),
    )


def grade_from_snapshot(config_snapshot: dict[str, Any]) -> Grade:
    """Resolve the whole-film :class:`Grade` from a run's ``config_snapshot``.

    The config-boundary adapter around :func:`pick_grade`: it pulls the same
    director-intent tokens the rest of the pipeline reads — ``mood`` / ``tone``
    (from the ``finish`` block first, then ``direction`` / ``sound_effects`` /
    top-level), the ``motion.style`` word, the C23 design tokens (only when a
    ``design`` author-block is present), the author's ``finish.grade`` override
    map, and a deterministic :func:`subseed` of the run's master seed — and hands
    them to :func:`pick_grade`. Pure/deterministic; no config block is required
    (every token degrades to a neutral default).
    """
    cfg = config_snapshot or {}
    finish = _dict(cfg.get("finish"))
    direction = _dict(cfg.get("direction"))
    motion = _dict(cfg.get("motion"))

    mood = resolve_mood(cfg)
    tone = _first_token(finish.get("tone"), direction.get("tone"), cfg.get("tone"))
    style = _first_token(motion.get("style"))

    design_block = cfg.get("design")
    design_tokens: DesignTokens | None = None
    if isinstance(design_block, dict) and design_block:
        # Lazy import (mirrors design_system.token_fingerprint importing motion_plan
        # lazily) — keeps the shared-module import graph acyclic.
        from src.services.stages.shared.design_system import derive_tokens

        design_tokens = derive_tokens(cfg.get("brand"), cfg.get("theme"), design_block)

    brief_grade = finish.get("grade") if isinstance(finish.get("grade"), dict) else None
    seed = subseed(master_seed(cfg), "grade", str(cfg.get("project_id") or ""))

    return pick_grade(
        tone=tone,
        mood=mood,
        style=style,
        design_tokens=design_tokens,
        brief_grade=brief_grade,
        seed=seed,
    )
