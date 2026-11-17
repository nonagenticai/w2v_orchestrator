"""Tests for the contrast-safe theme mapping onto motion templates."""

from __future__ import annotations

from src.services.stages.shared.motion_slides import (
    motion_brand_params,
    theme_motion_params,
)

_BRAND: dict[str, object] = {"secondary_color": "#14B8A6", "accent_color": "#F97316"}


def test_accents_always_map() -> None:
    params = theme_motion_params(_BRAND, None)
    assert params == {"accent": "#14B8A6", "accent2": "#F97316"}


def test_no_theme_equals_brand_only() -> None:
    assert theme_motion_params(_BRAND, {}) == motion_brand_params(_BRAND)


def test_high_contrast_theme_overrides_surface() -> None:
    # Near-white text on a near-black background clears 4.5:1 easily.
    theme: dict[str, object] = {
        "gradient_start": "#0A0A0A",
        "gradient_end": "#1A1A2E",
        "text_color": "#FAFAFA",
        "subtitle_color": "#B0B0B0",
    }
    params = theme_motion_params(_BRAND, theme)
    assert params["bg"] == "#0A0A0A"
    assert params["bg2"] == "#1A1A2E"
    assert params["text"] == "#FAFAFA"
    assert params["muted"] == "#B0B0B0"
    # Accents still flow through.
    assert params["accent"] == "#14B8A6"


# auto dark text on a light bg


def test_mid_gray_bg_where_no_default_text_clears_falls_back() -> None:
    # A mid-grey background where neither the light nor the dark default text clears
    # 4.5:1 → no surface override; the template keeps its own readable canvas.
    params = theme_motion_params(_BRAND, {"background_color": "#777777"})
    assert "bg" not in params and "text" not in params
    assert params == motion_brand_params(_BRAND)


def test_brand_mark_emitted_only_when_name_present() -> None:
    assert "brand_mark" not in motion_brand_params(_BRAND)
    named = motion_brand_params({**_BRAND, "name": "Specra"})
    assert named["brand_mark"] == "Specra"
    # brand_mark flows through theme_motion_params too.
    assert theme_motion_params({**_BRAND, "name": "Specra"}, None)["brand_mark"] == "Specra"


def test_unparseable_colours_fall_back() -> None:
    theme: dict[str, object] = {"background_color": "not-a-colour", "text_color": "#FAFAFA"}
    params = theme_motion_params(_BRAND, theme)
    assert "bg" not in params


def test_low_contrast_muted_is_dropped_but_surface_kept() -> None:
    # Coherent high-contrast bg/text, but a muted that is too close to bg → drop
    # only muted, keep the themed surface.
    theme: dict[str, object] = {
        "background_color": "#0A0A0A",
        "text_color": "#FAFAFA",
        "subtitle_color": "#0C0C0C",
    }
    params = theme_motion_params(_BRAND, theme)
    assert params["bg"] == "#0A0A0A"
    assert "muted" not in params
