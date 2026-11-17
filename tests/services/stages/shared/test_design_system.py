"""Tests for the pure C23 design-token derivation."""

from __future__ import annotations

from src.services.stages.shared.design_system import (
    DEFAULT_TOKENS,
    apply_design_tokens,
    as_params,
    derive_tokens,
    grid_tokens,
    palette_hash,
    token_fingerprint,
    type_scale,
)

# The shipped brand (mirrors slide_deck._BRAND_DEFAULTS).
_BRAND: dict[str, object] = {
    "primary_color": "#0B1F3A",
    "secondary_color": "#2BB6A3",
    "accent_color": "#F26B3A",
    "font_family": "Inter",
}


def test_default_tokens_reproduced_from_empty_inputs() -> None:
    assert derive_tokens(None, None, None) == DEFAULT_TOKENS


def test_default_tokens_carry_the_shipped_brand() -> None:
    p = DEFAULT_TOKENS.palette
    # Direct brand mappings are verbatim.
    assert p.surface == "#0B1F3A"
    assert p.accent == "#2BB6A3"
    assert p.accent_2 == "#F26B3A"
    assert DEFAULT_TOKENS.font_family == "Inter"
    assert DEFAULT_TOKENS.icon_set == "line"
    assert DEFAULT_TOKENS.type_scale_ratio == 1.25


def test_type_scale_is_a_modular_scale() -> None:
    ts = type_scale(1.25)
    assert ts.body == 18
    assert ts.h2 == 23
    assert ts.h1 == 28
    assert ts.display == 35
    assert ts.kicker == 14
    assert ts.caption == 12
    # Strictly descending display → caption.
    assert ts.display > ts.h1 > ts.h2 > ts.body > ts.kicker > ts.caption


def test_type_scale_ratio_flows_through_derive() -> None:
    tokens = derive_tokens(_BRAND, None, {"type_scale_ratio": 1.5})
    assert tokens.type_scale_ratio == 1.5
    assert tokens.type == type_scale(1.5)


def test_grid_defaults_and_overrides() -> None:
    assert grid_tokens() == grid_tokens(None)
    g = grid_tokens({"grid": {"gutter": 40}})
    assert g.gutter == 40
    assert g.cols == 12  # untouched default


def test_apply_design_tokens_identity_when_block_absent() -> None:
    brand = dict(_BRAND)
    before_keys = set(brand)
    result = apply_design_tokens(brand, None)
    # Same object, zero keys added.
    assert result is brand
    assert set(result) == before_keys


def test_apply_design_tokens_falsy_block_is_identity() -> None:
    brand = dict(_BRAND)
    assert apply_design_tokens(brand, {}) is brand


def test_apply_design_tokens_present_block_updates_without_adding_keys() -> None:
    brand = dict(_BRAND)
    before_keys = set(brand)
    result = apply_design_tokens(brand, {"palette": {"accent": "#112233"}})
    assert result is not brand
    assert set(result) == before_keys  # no NEW keys
    assert result["secondary_color"] == "#112233"  # accent overlaid onto brand


def test_palette_hash_is_stable_and_ignores_none() -> None:
    assert palette_hash(DEFAULT_TOKENS) == palette_hash(derive_tokens(None, None, None))
    assert palette_hash(None) == ""


def test_palette_hash_differs_iff_a_token_differs() -> None:
    base = derive_tokens(_BRAND)
    same = derive_tokens(_BRAND)
    changed = derive_tokens({**_BRAND, "secondary_color": "#654321"})
    assert palette_hash(base) == palette_hash(same)
    assert palette_hash(base) != palette_hash(changed)
    # A design-block ratio change also moves the hash (whole-bundle coverage).
    ratio_change = derive_tokens(_BRAND, None, {"type_scale_ratio": 1.4})
    assert palette_hash(base) != palette_hash(ratio_change)


def test_token_fingerprint_none_iff_both_blocks_falsy() -> None:
    assert token_fingerprint(None, None) is None
    assert token_fingerprint({}, {}) is None
    assert token_fingerprint(None, {}, DEFAULT_TOKENS) is None


def test_token_fingerprint_non_none_when_either_block_present() -> None:
    design_only = token_fingerprint({"palette": {}}, None, DEFAULT_TOKENS)
    motion_only = token_fingerprint(None, {"style": "punchy"})
    assert isinstance(design_only, str) and len(design_only) == 16
    assert isinstance(motion_only, str) and len(motion_only) == 16


def test_token_fingerprint_changes_with_tokens() -> None:
    a = token_fingerprint({"palette": {}}, None, derive_tokens(_BRAND))
    b = token_fingerprint(
        {"palette": {}}, None, derive_tokens({**_BRAND, "secondary_color": "#654321"})
    )
    assert a != b


def test_as_params_is_a_flat_str_map() -> None:
    params = as_params(DEFAULT_TOKENS)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in params.items())
    assert params["surface"] == "#0B1F3A"
    assert params["type_body"] == "18"
    assert params["font_family"] == "Inter"
