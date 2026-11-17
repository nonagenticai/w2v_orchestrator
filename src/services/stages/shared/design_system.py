"""Pure, deterministic C23 design-token derivation (spec 04 § design system).

A run's optional ``design`` author-block (``type_scale_ratio`` / ``icon_set`` /
``palette``) plus the project ``brand`` (and, when present, the deck ``theme``)
are folded into one immutable :class:`DesignTokens` bundle — a palette, a modular
type scale, and a baseline grid. Everything here is a pure function of its inputs:
no ``datetime.now``, no ``random``, no env reads, so the same inputs always derive
the same tokens (and therefore the same media ids downstream — the whole point of
the C23 seam).

The WCAG contrast helpers (``_hex_to_rgb`` / ``_relative_luminance`` /
``_contrast_ratio``) and the light/dark text defaults are the canonical copies —
they were LIFTED here verbatim from ``motion_slides`` so both modules agree
byte-for-byte; ``motion_slides`` re-imports them from this module (no behaviour
change). Do not fork them.

Presence-is-opt-in throughout: :func:`apply_design_tokens` is a STRICT identity
(the very same object, zero keys added) when the ``design`` block is falsy, so a
run WITHOUT a design block is byte-identical to today. Only a present block moves
any bit.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, cast

from src.services.stages.shared.provenance import resolve_with_provenance

if TYPE_CHECKING:
    # Lazily typed to avoid the hard import cycle with ``motion_plan`` (which
    # ``token_fingerprint`` imports inside the function body at runtime).
    from src.services.stages.shared.motion_plan import MotionPlan

__all__ = [
    "DEFAULT_TOKENS",
    "DesignTokens",
    "Grid",
    "Palette",
    "TypeScale",
    "apply_design_tokens",
    "as_params",
    "derive_palette",
    "derive_tokens",
    "grid_tokens",
    "palette_hash",
    "token_fingerprint",
    "tokens_for_snapshot",
    "type_scale",
]

_COMPUTED_BY = "design_system"

# Common Practice brand defaults — MUST stay identical to
# ``slide_deck._BRAND_DEFAULTS`` (slide_deck.py:39-44) so ``derive_tokens(None,
# None, None) == DEFAULT_TOKENS`` reproduces the shipped brand exactly.
_BRAND_DEFAULTS: dict[str, str] = {
    "primary_color": "#0B1F3A",  # navy
    "secondary_color": "#2BB6A3",  # teal
    "accent_color": "#F26B3A",  # orange
    "font_family": "Inter",
}

# Type-scale + grid defaults. ``_TYPE_BASE`` is the body size in px; the other
# steps are a modular scale off ``type_scale_ratio`` (default below).
_TYPE_BASE = 18
_DEFAULT_TYPE_RATIO = 1.25
_DEFAULT_ICON_SET = "line"
_GRID_COLS = 12
_GRID_GUTTER = 24
_GRID_MARGIN = 64
_GRID_UNIT = 8

# ---------------------------------------------------------------------------
# WCAG helpers — LIFTED VERBATIM from motion_slides. Canonical home is here.
# ---------------------------------------------------------------------------

# WCAG 2.x contrast thresholds: 4.5:1 for body text, 3:1 for large/secondary text.
_MIN_TEXT_CONTRAST = 4.5
_MIN_MUTED_CONTRAST = 3.0

# Auto-text candidates, matching the motion templates' light/dark defaults
# (animated_title_card.html / animated_slide.html). When a theme/brand supplies a
# background but no explicit text colour, we pick whichever of these clears the
# body-text contrast threshold against that background, so a branded surface stays
# readable without the author having to author a text colour.
_LIGHT_TEXT = "#F8FAFC"
_DARK_TEXT = "#0B1437"


def _hex_to_rgb(value: str) -> tuple[int, int, int] | None:
    """Parse a ``#rgb`` / ``#rrggbb`` hex string to an ``(r, g, b)`` triple."""
    s = value.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return None
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return None


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    """WCAG relative luminance of an sRGB colour."""

    def _chan(c: int) -> float:
        x = c / 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * _chan(r) + 0.7152 * _chan(g) + 0.0722 * _chan(b)


def _contrast_ratio(fg: str, bg: str) -> float | None:
    """WCAG contrast ratio between two hex colours; ``None`` if either won't parse."""
    a, b = _hex_to_rgb(fg), _hex_to_rgb(bg)
    if a is None or b is None:
        return None
    la, lb = _relative_luminance(a), _relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# ---------------------------------------------------------------------------
# Deterministic integer-HSL colour maths (for derived surface/accent variants).
# ---------------------------------------------------------------------------


def _rgb_to_hsl(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """sRGB triple → integer HSL (h 0-360, s/l 0-100). Deterministic (rounded)."""
    r, g, b = (c / 255.0 for c in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    lum = (mx + mn) / 2.0
    if mx == mn:
        return (0, 0, round(lum * 100))
    d = mx - mn
    s = d / (2.0 - mx - mn) if lum > 0.5 else d / (mx + mn)
    if mx == r:
        h = (g - b) / d + (6.0 if g < b else 0.0)
    elif mx == g:
        h = (b - r) / d + 2.0
    else:
        h = (r - g) / d + 4.0
    h /= 6.0
    return (round(h * 360), round(s * 100), round(lum * 100))


def _hsl_to_hex(h: int, s: int, lum: int) -> str:
    """Integer HSL → ``#RRGGBB``. Inverse of :func:`_rgb_to_hsl` (lossy round-trip)."""
    hf, sf, lf = h / 360.0, s / 100.0, lum / 100.0
    if sf == 0:
        r = g = b = lf
    else:
        q = lf * (1.0 + sf) if lf < 0.5 else lf + sf - lf * sf
        p = 2.0 * lf - q

        def _hue(t: float) -> float:
            if t < 0:
                t += 1.0
            if t > 1:
                t -= 1.0
            if t < 1 / 6:
                return p + (q - p) * 6.0 * t
            if t < 1 / 2:
                return q
            if t < 2 / 3:
                return p + (q - p) * (2 / 3 - t) * 6.0
            return p

        r, g, b = _hue(hf + 1 / 3), _hue(hf), _hue(hf - 1 / 3)
    return f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"


def _clamp01(v: int, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, v))


def _adjust(hex_color: str, *, d_lum: int = 0, d_sat: int = 0) -> str:
    """Shift a colour's lightness/saturation in integer-HSL space.

    Returns the input unchanged if it does not parse (never invents a colour)."""
    rgb = _hex_to_rgb(hex_color)
    if rgb is None:
        return hex_color
    h, s, lum = _rgb_to_hsl(rgb)
    return _hsl_to_hex(h, _clamp01(s + d_sat), _clamp01(lum + d_lum))


def _mix(fg: str, bg: str, t: float) -> str:
    """Linear sRGB blend of ``fg`` toward ``bg`` by fraction ``t`` (0 = fg, 1 = bg)."""
    a, b = _hex_to_rgb(fg), _hex_to_rgb(bg)
    if a is None or b is None:
        return fg
    r, g, bl = (round(a[i] * (1.0 - t) + b[i] * t) for i in range(3))
    return f"#{r:02X}{g:02X}{bl:02X}"


def _auto_text(bg: str, theme_text: object) -> str:
    """Contrast-safe text colour for ``bg``.

    Honours an authored ``theme_text`` when it clears the body-text threshold,
    else picks the higher-contrast of the light/dark template defaults. Falls
    back to the light default when ``bg`` is unparseable so a colour is always
    returned (the palette never carries a ``None`` token)."""
    if isinstance(theme_text, str) and theme_text:
        ratio = _contrast_ratio(theme_text, bg)
        if ratio is not None and ratio >= _MIN_TEXT_CONTRAST:
            return theme_text
    light = _contrast_ratio(_LIGHT_TEXT, bg)
    dark = _contrast_ratio(_DARK_TEXT, bg)
    if light is None or dark is None:
        return _LIGHT_TEXT
    return _LIGHT_TEXT if light >= dark else _DARK_TEXT


def _first_str(*candidates: object) -> str | None:
    for cand in candidates:
        if isinstance(cand, str) and cand:
            return cand
    return None


# ---------------------------------------------------------------------------
# Token dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    """A resolved, contrast-checked colour palette (all values ``#RRGGBB``)."""

    surface: str
    surface_2: str
    on_surface: str
    on_surface_muted: str
    accent: str
    accent_2: str
    accent_soft: str


@dataclass(frozen=True)
class TypeScale:
    """A modular type scale in px (ints), largest → smallest."""

    display: int
    h1: int
    h2: int
    body: int
    kicker: int
    caption: int


@dataclass(frozen=True)
class Grid:
    """A baseline layout grid."""

    cols: int
    gutter: int
    margin: int
    unit: int


@dataclass(frozen=True)
class DesignTokens:
    """The full immutable C23 token bundle for a run."""

    palette: Palette
    type: TypeScale
    grid: Grid
    font_family: str
    icon_set: str
    type_scale_ratio: float


# ---------------------------------------------------------------------------
# Derivation.
# ---------------------------------------------------------------------------


def _round_half_up(x: float) -> int:
    """Deterministic round-half-up (avoids Python's banker's rounding surprises)."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def derive_palette(
    brand: Mapping[str, object] | None,
    theme: Mapping[str, object] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> Palette:
    """Derive a contrast-checked :class:`Palette` from brand + theme + overrides.

    ``surface`` is taken verbatim from the first usable of the theme's
    gradient/background keys then the brand ``primary_color`` (so it round-trips
    to the same hex). The elevated ``surface_2``, the ``on_surface`` text, its
    ``on_surface_muted`` variant and ``accent_soft`` are DERIVED in integer-HSL
    space and every text token is contrast-checked against ``surface`` (body
    ≥4.5:1, muted ≥3:1) with a light/dark fallback. ``accent`` / ``accent_2`` are
    the brand secondary/accent verbatim. A per-token hex in ``overrides`` (the
    ``design.palette`` block) wins via the C14 provenance ladder."""
    brand = brand or {}
    theme = theme or {}
    overrides = overrides or {}

    surface = (
        _first_str(
            theme.get("gradient_start"),
            theme.get("background_color"),
            theme.get("background"),
            brand.get("primary_color"),
        )
        or _BRAND_DEFAULTS["primary_color"]
    )
    accent = _first_str(brand.get("secondary_color")) or _BRAND_DEFAULTS["secondary_color"]
    accent_2 = _first_str(brand.get("accent_color")) or _BRAND_DEFAULTS["accent_color"]

    surface_2 = _first_str(theme.get("gradient_end")) or _adjust(surface, d_lum=8)
    on_surface = _auto_text(surface, theme.get("text_color"))

    muted = _mix(on_surface, surface, 0.35)
    muted_ratio = _contrast_ratio(muted, surface)
    if muted_ratio is None or muted_ratio < _MIN_MUTED_CONTRAST:
        muted = on_surface  # never drop below the readable body text

    accent_soft = _adjust(accent, d_lum=22, d_sat=-28)

    computed = {
        "surface": surface,
        "surface_2": surface_2,
        "on_surface": on_surface,
        "on_surface_muted": muted,
        "accent": accent,
        "accent_2": accent_2,
        "accent_soft": accent_soft,
    }

    def _resolve(knob: str) -> str:
        # Mirror the mastering_plan resolve idiom: branch on presence, never
        # import the private _UNSET sentinel.
        if knob in overrides:
            return str(
                resolve_with_provenance(
                    knob,
                    override=overrides[knob],
                    default=computed[knob],
                    computed_by=_COMPUTED_BY,
                ).value
            )
        return str(
            resolve_with_provenance(knob, default=computed[knob], computed_by=_COMPUTED_BY).value
        )

    return Palette(
        surface=_resolve("surface"),
        surface_2=_resolve("surface_2"),
        on_surface=_resolve("on_surface"),
        on_surface_muted=_resolve("on_surface_muted"),
        accent=_resolve("accent"),
        accent_2=_resolve("accent_2"),
        accent_soft=_resolve("accent_soft"),
    )


def type_scale(ratio: float = _DEFAULT_TYPE_RATIO, base: int = _TYPE_BASE) -> TypeScale:
    """A modular type scale off ``base`` (body px) and ``ratio`` (round-half-up)."""

    def step(n: int) -> int:
        return _round_half_up(base * (ratio**n))

    return TypeScale(
        display=step(3),
        h1=step(2),
        h2=step(1),
        body=base,
        kicker=step(-1),
        caption=step(-2),
    )


def grid_tokens(design_block: Mapping[str, object] | None = None) -> Grid:
    """Baseline grid, with optional ``design.grid`` integer overrides."""
    grid_block = (design_block or {}).get("grid")
    # ``design_block.get`` returns ``object``; ``isinstance``-narrowing that to
    # ``dict`` yields ``dict[Unknown, Unknown]`` (Never-typed keys), so cast to the
    # opaque-JSON shape for the read-only subscripts below.
    block: Mapping[str, object] = (
        cast("Mapping[str, object]", grid_block) if isinstance(grid_block, dict) else {}
    )

    def _resolve(knob: str, default: int) -> int:
        if knob in block:
            return int(
                resolve_with_provenance(
                    knob, override=block[knob], default=default, computed_by=_COMPUTED_BY
                ).value
            )
        return default

    return Grid(
        cols=_resolve("cols", _GRID_COLS),
        gutter=_resolve("gutter", _GRID_GUTTER),
        margin=_resolve("margin", _GRID_MARGIN),
        unit=_resolve("unit", _GRID_UNIT),
    )


def derive_tokens(
    brand: Mapping[str, object] | None,
    theme: Mapping[str, object] | None = None,
    design_block: Mapping[str, object] | None = None,
) -> DesignTokens:
    """Fold brand + theme + the ``design`` author-block into a :class:`DesignTokens`.

    ``derive_tokens(None, None, None)`` reproduces :data:`DEFAULT_TOKENS` exactly
    (the brand defaults, ratio 1.25, ``line`` icon set). Presence-is-opt-in for
    the design block: an absent block simply uses the computed defaults."""
    # A malformed (non-dict) brand/design block from unvalidated ``config_overrides``
    # is treated as absent — degrade to computed defaults, never crash the run
    # (mirrors ``cadence.py`` isinstance-guarding its author block).
    merged_brand = {**_BRAND_DEFAULTS, **(brand if isinstance(brand, dict) else {})}
    design_block = design_block if isinstance(design_block, dict) else {}

    ratio_raw = design_block.get("type_scale_ratio")
    if "type_scale_ratio" in design_block and ratio_raw is not None:
        ratio = float(
            resolve_with_provenance(
                "type_scale_ratio",
                override=ratio_raw,
                default=_DEFAULT_TYPE_RATIO,
                computed_by=_COMPUTED_BY,
            ).value
        )
    else:
        ratio = _DEFAULT_TYPE_RATIO

    if "icon_set" in design_block and design_block.get("icon_set"):
        icon_set = str(
            resolve_with_provenance(
                "icon_set",
                override=design_block["icon_set"],
                default=_DEFAULT_ICON_SET,
                computed_by=_COMPUTED_BY,
            ).value
        )
    else:
        icon_set = _DEFAULT_ICON_SET

    palette_overrides_raw = design_block.get("palette")
    palette_overrides: Mapping[str, object] | None = (
        cast("Mapping[str, object]", palette_overrides_raw)
        if isinstance(palette_overrides_raw, dict)
        else None
    )

    return DesignTokens(
        palette=derive_palette(merged_brand, theme, palette_overrides),
        type=type_scale(ratio),
        grid=grid_tokens(design_block),
        font_family=str(merged_brand.get("font_family") or _BRAND_DEFAULTS["font_family"]),
        icon_set=icon_set,
        type_scale_ratio=ratio,
    )


def as_params(tokens: DesignTokens) -> dict[str, str]:
    """Flatten a token bundle into a stable ``str→str`` param map for templates."""
    p = tokens.palette
    t = tokens.type
    g = tokens.grid
    return {
        "surface": p.surface,
        "surface_2": p.surface_2,
        "on_surface": p.on_surface,
        "on_surface_muted": p.on_surface_muted,
        "accent": p.accent,
        "accent_2": p.accent_2,
        "accent_soft": p.accent_soft,
        "type_display": str(t.display),
        "type_h1": str(t.h1),
        "type_h2": str(t.h2),
        "type_body": str(t.body),
        "type_kicker": str(t.kicker),
        "type_caption": str(t.caption),
        "grid_cols": str(g.cols),
        "grid_gutter": str(g.gutter),
        "grid_margin": str(g.margin),
        "grid_unit": str(g.unit),
        "font_family": tokens.font_family,
        "icon_set": tokens.icon_set,
        "type_scale_ratio": repr(tokens.type_scale_ratio),
    }


def tokens_for_snapshot(config_snapshot: Mapping[str, object] | None) -> dict[str, str] | None:
    """The run's flat 19-token param map for the renderers, or ``None`` if not opted in.

    The single seam the deck renderers read. Presence-is-opt-in: a run with no
    (or a malformed, non-dict) ``design`` author-block returns ``None``, so the
    caller omits the ``tokens`` key entirely and the render payload stays
    byte-identical to today. Derived from EXACTLY the inputs
    ``motion_slides.token_fp_for_snapshot`` hashes into the media id
    (``derive_tokens(brand, theme, design)``), so the tokens that reach the renderer
    and the tokens a media id is keyed on can never diverge.
    """
    cfg = config_snapshot or {}
    design_block = cfg.get("design")
    if not isinstance(design_block, dict) or not design_block:
        return None
    brand = cfg.get("brand")
    theme = cfg.get("theme")
    return as_params(
        derive_tokens(
            cast("Mapping[str, object]", brand) if isinstance(brand, dict) else None,
            cast("Mapping[str, object]", theme) if isinstance(theme, dict) else None,
            cast("Mapping[str, object]", design_block),
        )
    )


def palette_hash(tokens: DesignTokens | None) -> str:
    """Stable short hash over the WHOLE token bundle (changes iff any token does)."""
    if tokens is None:
        return ""
    canonical = json.dumps(asdict(tokens), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def token_fingerprint(
    design_block: Mapping[str, object] | None,
    motion_block: Mapping[str, object] | None,
    tokens: DesignTokens | None = None,
    plan: MotionPlan | None = None,
) -> str | None:
    """The C23 media-id seam: a short fingerprint of the resolved design + motion.

    ``None`` iff BOTH author-blocks are falsy — keyed on BLOCK PRESENCE, not on
    ``tokens is None`` (``derive_tokens`` never returns ``None``), so the caller
    only folds the fingerprint into a media id when the run actually opted in. The
    hash covers the DERIVED tokens (via :func:`palette_hash`) and the motion plan
    (via ``motion_plan.plan_hash``), so a media id changes iff the resolved tokens
    change — the idempotency invariant. ``motion_plan`` is imported lazily to avoid
    a hard import cycle; if the motion block is present but ``plan`` is ``None`` the
    design side is still hashed on its own."""
    if not design_block and not motion_block:
        return None
    design_part = palette_hash(tokens) if (design_block and tokens is not None) else ""
    plan_part = ""
    if motion_block and plan is not None:
        from src.services.stages.shared.motion_plan import plan_hash

        plan_part = plan_hash(plan)
    canonical = json.dumps([design_part, plan_part], separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def apply_design_tokens[BrandT: Mapping[str, object]](
    brand: BrandT, design_block: Mapping[str, object] | None
) -> BrandT | dict[str, object]:
    """Overlay the resolved design palette onto ``brand`` for the deck renderers.

    STRICT identity when ``design_block`` is falsy: the very SAME ``brand`` object
    is returned and ZERO keys are added, so ``brand_key`` (and every downstream
    media id) is byte-identical to today. Only a present design block updates the
    brand's existing colour/font keys (never adds new keys) to the resolved,
    contrast-checked palette so the deck picks up the derived surface + accents."""
    # Falsy OR malformed (non-dict, e.g. from unvalidated ``config_overrides``) →
    # strict identity: same object, zero keys added, byte-identical media ids.
    if not isinstance(design_block, dict) or not design_block:
        return brand
    tokens = derive_tokens(brand, None, design_block)
    updated = dict(brand)
    updated["primary_color"] = tokens.palette.surface
    updated["secondary_color"] = tokens.palette.accent
    updated["accent_color"] = tokens.palette.accent_2
    updated["font_family"] = tokens.font_family
    return updated


# The canonical default token bundle: exactly what the shipped brand derives to
# with no theme and no design block. Pure/deterministic → a safe module constant.
DEFAULT_TOKENS: DesignTokens = derive_tokens(None, None, None)
