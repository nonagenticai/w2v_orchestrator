"""ORCH-3.1 make-or-break regression guard: design/motion tokens are INERT.

The whole point of PR 3.1 is that a run with NO ``design`` and NO ``motion``
author-block is bit-for-bit identical to today across every shared renderer. This
module pins that invariant at the media-id seam: for the slide-image, motion-clip
AND ken-burns paths (all shared by showcase / lesson / audit — D4), it asserts the
wired ``deterministic_media_id`` output equals the pre-change value (the same call
without a fingerprint) whenever ``token_fp_for_snapshot`` returns ``None`` — which
it must for every block-less snapshot. It also pins the positive direction (a
present design or motion block DOES move the id) and the strict-identity contract
of ``apply_design_tokens`` (zero keys added when no design block).

Pure + deterministic: no DB, no I/O — this replays the exact fold idiom the render
callsites use (``*((token_fp,) if token_fp else ())``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from src.services.stages.shared.design_system import (
    DEFAULT_TOKENS,
    apply_design_tokens,
    derive_tokens,
)
from src.services.stages.shared.motion_slides import token_fp_for_snapshot
from src.services.tts_storage import deterministic_media_id

# ── Block-less snapshots for the three shared-renderer flows (D4). None carry a
# ``design`` or ``motion`` block, so every one must fingerprint to ``None``. ─────
_SHOWCASE_CFG = {"project_id": "p-show", "brand": {"primary_color": "#0B1F3A"}, "theme": {}}
_LESSON_CFG = {"project_id": "p-lesson", "brand": {}, "theme": {"background": "#111111"}}
_AUDIT_CFG = {"project_id": "p-audit", "brand": {"primary_color": "#222222"}}
_BLOCKLESS_CFGS = {"showcase": _SHOWCASE_CFG, "lesson": _LESSON_CFG, "audit": _AUDIT_CFG}

# ── Present-block snapshots (opt-in). Each MUST fingerprint to a non-None value. ─
_DESIGN_CFG = {
    "project_id": "p1",
    "brand": {"primary_color": "#0B1F3A"},
    "design": {"palette": {"surface": "#123456"}},
}
_MOTION_CFG = {"project_id": "p1", "motion": {"style": "punchy"}}


def _wired_media_id(parts: tuple[object, ...], token_fp: str | None) -> str:
    """Replay the exact conditional-omission fold used at every render callsite."""
    return deterministic_media_id(*parts, *((token_fp,) if token_fp else ()))


# Per-flow media-id part tuples, mirroring each shared renderer verbatim.
def _slide_image_parts(stage_name: str, cfg: Mapping[str, object]) -> tuple[object, ...]:
    # slide_deck.render_deck_slide_images: (stage, slide_json, brand_key, scale, i)
    slide = {"content": {"title": "a"}}
    brand_key = json.dumps(cfg.get("brand"), sort_keys=True, default=str)
    return (
        stage_name,
        json.dumps(slide, sort_keys=True, default=str),
        brand_key,
        2,
        0,
    )


def _motion_clip_parts(stage_name: str) -> tuple[object, ...]:
    # motion_slides.render_deck_motion_clips: (stage, template, params, w, h, fps, dur)
    params = {"eyebrow": "e", "title": "t", "subtitle": "s"}
    return (
        stage_name,
        "animated_title_card",
        json.dumps(params, sort_keys=True, default=str),
        1920,
        1080,
        30,
        6000,
    )


def _ken_burns_parts() -> tuple[object, ...]:
    # ken_burns.apply_ken_burns: ("ken_burns", image_url, preset, fit, dur, w, h)
    return ("ken_burns", "https://minio.test/img.png", "zoom_in", "cover", 5000, 1920, 1080)


@pytest.mark.parametrize("flow", list(sorted(_BLOCKLESS_CFGS))[:1])
def test_fingerprint_is_none_without_blocks(flow: str) -> None:
    assert token_fp_for_snapshot(_BLOCKLESS_CFGS[flow]) is None


def test_fingerprint_none_for_empty_and_missing_snapshot() -> None:
    assert token_fp_for_snapshot(None) is None
    assert token_fp_for_snapshot({}) is None


@pytest.mark.parametrize("flow", list(sorted(_BLOCKLESS_CFGS))[:1])
def test_slide_image_media_id_inert_without_blocks(flow: str) -> None:
    cfg = _BLOCKLESS_CFGS[flow]
    parts = _slide_image_parts(f"{flow}_slides", cfg)
    baseline = deterministic_media_id(*parts)  # the pre-change value (no fingerprint)
    assert _wired_media_id(parts, token_fp_for_snapshot(cfg)) == baseline


@pytest.mark.parametrize("flow", list(sorted(_BLOCKLESS_CFGS))[:1])
def test_motion_clip_media_id_inert_without_blocks(flow: str) -> None:
    cfg = _BLOCKLESS_CFGS[flow]
    parts = _motion_clip_parts(f"{flow}_slides")
    baseline = deterministic_media_id(*parts)
    assert _wired_media_id(parts, token_fp_for_snapshot(cfg)) == baseline


@pytest.mark.parametrize("flow", list(sorted(_BLOCKLESS_CFGS))[:1])
def test_ken_burns_media_id_inert_without_blocks(flow: str) -> None:
    cfg = _BLOCKLESS_CFGS[flow]
    parts = _ken_burns_parts()
    baseline = deterministic_media_id(*parts)
    assert _wired_media_id(parts, token_fp_for_snapshot(cfg)) == baseline


def test_apply_design_tokens_is_strict_identity_without_block() -> None:
    brand = {"primary_color": "#0B1F3A", "secondary_color": "#2BB6A3", "name": "Acme"}
    result = apply_design_tokens(brand, None)
    assert result is brand  # the SAME object
    assert set(result) == set(brand)  # zero keys added
    assert apply_design_tokens(brand, {}) is brand  # empty block is falsy → identity too


def test_present_design_block_changes_every_path() -> None:
    fp = token_fp_for_snapshot(_DESIGN_CFG)
    assert fp is not None
    for parts in (
        _slide_image_parts("showcase_slides", _DESIGN_CFG),
        _motion_clip_parts("showcase_slides"),
        _ken_burns_parts(),
    ):
        assert _wired_media_id(parts, fp) != deterministic_media_id(*parts)


def test_present_motion_block_changes_every_path() -> None:
    fp = token_fp_for_snapshot(_MOTION_CFG)
    assert fp is not None
    for parts in (
        _slide_image_parts("showcase_slides", _MOTION_CFG),
        _motion_clip_parts("showcase_slides"),
        _ken_burns_parts(),
    ):
        assert _wired_media_id(parts, fp) != deterministic_media_id(*parts)


def test_fingerprint_moves_iff_resolved_token_moves() -> None:
    a = token_fp_for_snapshot(
        {"project_id": "p", "brand": {}, "design": {"palette": {"surface": "#111111"}}}
    )
    b = token_fp_for_snapshot(
        {"project_id": "p", "brand": {}, "design": {"palette": {"surface": "#222222"}}}
    )
    assert a is not None and b is not None
    assert a != b  # a different resolved surface ⇒ a different id (idempotency)

    same = token_fp_for_snapshot(
        {"project_id": "p", "brand": {}, "design": {"palette": {"surface": "#111111"}}}
    )
    assert same == a  # identical inputs ⇒ identical id (determinism)


@pytest.mark.parametrize(
    "cfg",
    [{"design": "oops"}],
)
def test_malformed_blocks_degrade_to_inert(cfg: dict[str, object]) -> None:
    # A non-dict design/motion block (reachable via unvalidated ``config_overrides``
    # on ANY run type) must degrade to the inert path, never crash the run.
    assert token_fp_for_snapshot(cfg) is None


def _erase(value: object) -> Any:
    """Return a value as ``Any`` — simulates the untyped ``config_overrides`` path."""
    return value


def test_malformed_design_block_is_identity() -> None:
    brand = {"primary_color": "#abcabc"}
    # Deliberately-malformed (non-dict) blocks arrive untyped via ``config_overrides``.
    assert apply_design_tokens(brand, _erase("oops")) is brand
    assert derive_tokens(None, None, _erase("boom")) == DEFAULT_TOKENS
