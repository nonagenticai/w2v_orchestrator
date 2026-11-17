"""Pure-function tests for content-derived audiobook pacing (spec L3.1, no I/O).

Covers acceptance #5 (computed pacing over ≥3 boundary kinds, not the legacy
constants) and the caller half of #4 (absent-block byte-identity via a ``None``
return that lets the stitch stage keep its legacy ``700`` ms constant).
"""

from __future__ import annotations

import pytest

from src.services.stages.audiobook.pacing import (
    _BASE_MS,
    compute_pause_ms,
    resolve_chapter_gap_ms,
)


def test_base_pause_per_boundary_kind_neutral() -> None:
    # Acceptance #5: ≥3 distinct boundary kinds map to their distinct bases at
    # neutral tempo + speed 1.0 (rate_factor and tempo_scale both == 1.0).
    assert compute_pause_ms("sentence") == 150
    assert compute_pause_ms("paragraph") == 400
    assert compute_pause_ms("scene") == 900
    assert compute_pause_ms("dialogue_turn") == 250
    assert compute_pause_ms("chapter") == 1400


def test_unknown_boundary_falls_back_to_sentence() -> None:
    assert compute_pause_ms("nonsense") == _BASE_MS["sentence"]


def test_tempo_scales_every_base() -> None:
    # Smart-auto: brisk shortens (*0.8), measured lengthens (*1.25).
    assert compute_pause_ms("chapter", tempo="brisk") == round(1400 * 0.8)
    assert compute_pause_ms("chapter", tempo="measured") == round(1400 * 1.25)
    # Unknown/absent tempo is neutral.
    assert compute_pause_ms("chapter", tempo="whatever") == 1400
    assert compute_pause_ms("chapter", tempo=None) == 1400


def test_tempo_is_case_insensitive() -> None:
    assert compute_pause_ms("scene", tempo="  MEASURED ") == round(900 * 1.25)


def test_speed_rate_factor_clamps() -> None:
    # A slower narrator (speed 0.5 → 1/0.5 = 2.0) is clamped to the 1.4 ceiling.
    assert compute_pause_ms("chapter", speed=0.5) == round(1400 * 1.4)
    # A faster narrator (speed 2.0 → 0.5) is clamped to the 0.7 floor.
    assert compute_pause_ms("chapter", speed=2.0) == round(1400 * 0.7)
    # A mild speed-up scales proportionally within band (1/1.25 = 0.8).
    assert compute_pause_ms("chapter", speed=1.25) == round(1400 * 0.8)
    # Non-positive speed is meaningless → neutral 1.0.
    assert compute_pause_ms("chapter", speed=0.0) == 1400


def test_override_ms_wins_and_is_non_negative() -> None:
    assert compute_pause_ms("chapter", override_ms=250, tempo="brisk", speed=2.0) == 250
    assert compute_pause_ms("chapter", override_ms=0) == 0
    assert compute_pause_ms("chapter", override_ms=-5) == 0


@pytest.mark.parametrize("cfg", [{}])
def test_resolve_chapter_gap_none_when_no_pacing_optin(cfg: dict) -> None:
    # Byte-identity (acceptance #4): no direction/pacing block ⇒ None ⇒ the
    # stitch stage keeps its legacy 700 ms constant, output unchanged.
    assert resolve_chapter_gap_ms(cfg) is None


def test_resolve_chapter_gap_explicit_override_wins() -> None:
    # Explicit chapter_gap_ms wins even over a direction block, and honours 0.
    assert resolve_chapter_gap_ms({"chapter_gap_ms": 1200, "direction": "brisk"}) == 1200
    assert resolve_chapter_gap_ms({"chapter_gap_ms": 0, "direction": "measured"}) == 0


def test_resolve_chapter_gap_pacing_block_override() -> None:
    assert resolve_chapter_gap_ms({"pacing": {"chapter_ms": 2000}}) == 2000
    # A pacing block with no chapter_ms still opts in → computed (not None).
    assert resolve_chapter_gap_ms({"pacing": {"scene_ms": 500}}) == 1400


def test_resolve_chapter_gap_direction_computes_smart_auto() -> None:
    # A freeform direction string carrying a tempo cue drives the computed gap.
    assert resolve_chapter_gap_ms({"direction": "warm literary, brisk pace"}) == round(1400 * 0.8)
    assert resolve_chapter_gap_ms({"direction": "measured, contemplative"}) == round(1400 * 1.25)
    # Direction as a dict with explicit tempo.
    assert resolve_chapter_gap_ms({"direction": {"tempo": "measured"}}) == round(1400 * 1.25)


def test_resolve_chapter_gap_speed_scales_computed() -> None:
    # Computed gap is speed-scaled; a fast narrator gets a shorter chapter gap.
    assert resolve_chapter_gap_ms({"direction": "neutral"}, speed=2.0) == round(1400 * 0.7)


def test_resolve_chapter_gap_ms_ignores_malformed_overrides() -> None:
    # A malformed brief value (authored outside this codebase) must NOT raise out
    # of the terminal stitch stage — it is ignored (treated as absent) and the
    # control ladder falls through, exactly like ingest coerces bad pronunciations.
    # No pacing opt-in at all ⇒ a bad override is ignored and falls through to None.
    assert resolve_chapter_gap_ms({"chapter_gap_ms": "700ms"}) is None
    # A `pacing` dict IS an opt-in, so a malformed chapter_ms is ignored but the
    # computed default is used (not None) — and crucially it does NOT raise.
    assert resolve_chapter_gap_ms({"pacing": {"chapter_ms": "loud"}}) == 1400
    # a malformed override with a valid direction still computes from the direction
    assert resolve_chapter_gap_ms(
        {"chapter_gap_ms": "x", "direction": "measured, contemplative"}
    ) == round(1400 * 1.25)
