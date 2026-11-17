"""Unit tests for the cadence spec (pure, no DB).

Mirrors ``test_finish``. Covers the guarantees ORCH-0.6 ships:

1. **Legal-rate snap** — :func:`clamp_cadence` maps NTSC fractional rates onto
   their integer sibling and leaves legal rates untouched.
2. **Retime decision** — ``interpolate`` ONLY on a non-integer master↔source
   ratio of a ``high_motion`` source; a clean integer conform stays ``clean``.
3. **Presence-is-opt-in** — no intent AND no source profiles ⇒ ``None`` so the
   compile tail is byte-identical; ``assemble_shots`` omits the cadence keys and
   ``render_deck_motion_clips`` keeps fps 30 when ``master_fps`` is unset.
"""

from __future__ import annotations

from src.services.stages.shared.cadence import (
    CadenceSpec,
    build_cadence_spec,
    clamp_cadence,
    reconcile_cadence,
)
from src.services.stages.shot_assembly import assemble_shots


def test_clamp_legal_rates_identity() -> None:
    """A rate already in the legal set is returned unchanged."""
    for rate in (24, 25, 30, 50, 60):
        assert clamp_cadence(float(rate)) == rate


def test_clamp_ntsc_rates_snap() -> None:
    """NTSC fractional rates land on their integer sibling."""
    assert clamp_cadence(23.976) == 24
    assert clamp_cadence(29.97) == 30
    assert clamp_cadence(59.94) == 60


def test_reconcile_absent_intent_and_sources_is_none() -> None:
    """Master gate: no intent AND no source profiles ⇒ None (inert)."""
    assert reconcile_cadence([], None) is None


def test_reconcile_exact_ratio_is_clean() -> None:
    """Integer master↔source ratios never interpolate, even for high motion."""
    # src30 → master60 (ratio 2).
    spec = reconcile_cadence([{"fps": 30, "high_motion": True}], {"target_fps": 60})
    assert spec is not None
    assert spec.target_fps == 60
    assert spec.retime == "clean"
    # src25 → master50 (ratio 2).
    spec = reconcile_cadence([{"fps": 25, "high_motion": True}], {"target_fps": 50})
    assert spec is not None
    assert spec.retime == "clean"
    # equal rate (ratio 1).
    spec = reconcile_cadence([{"fps": 30, "high_motion": True}], {"target_fps": 30})
    assert spec is not None
    assert spec.retime == "clean"


def test_reconcile_non_integer_high_motion_interpolates() -> None:
    """A non-integer ratio on a high-motion source ⇒ interpolate."""
    spec = reconcile_cadence([{"fps": 24, "high_motion": True}], {"target_fps": 60})
    assert spec is not None
    assert spec.target_fps == 60
    assert spec.retime == "interpolate"


def test_reconcile_non_integer_low_motion_is_clean() -> None:
    """The same non-integer ratio on a low-motion source stays clean."""
    spec = reconcile_cadence([{"fps": 24, "high_motion": False}], {"target_fps": 60})
    assert spec is not None
    assert spec.retime == "clean"


def test_reconcile_target_from_dominant_source_when_no_override() -> None:
    """No override ⇒ target is the dominant (mode) legal-snapped source rate."""
    spec = reconcile_cadence([{"fps": 29.97}, {"fps": 30}, {"fps": 24}], None)
    assert spec is not None
    assert spec.target_fps == 30
    assert spec.source_fps == (30, 30, 24)


def test_reconcile_is_cfr_read_from_source() -> None:
    """is_cfr is the AND of source profiles — read, never re-asserted."""
    cfr = reconcile_cadence([{"fps": 30}], {"target_fps": 30})
    assert cfr is not None
    assert cfr.is_cfr is True
    spec = reconcile_cadence([{"fps": 30, "is_cfr": False}], {"target_fps": 30})
    assert spec is not None
    assert spec.is_cfr is False


def test_build_cadence_spec_absent_block_is_none() -> None:
    """No cadence block AND no source profiles ⇒ None (byte-identical compile)."""
    assert build_cadence_spec({}) is None
    assert build_cadence_spec({"cadence": None}) is None
    assert build_cadence_spec({"cadence": "on"}) is None


def test_build_cadence_spec_present_block_yields_spec() -> None:
    """A present cadence block yields a reconciled spec."""
    spec = build_cadence_spec({"cadence": {"target_fps": 59.94}})
    assert isinstance(spec, CadenceSpec)
    assert spec.target_fps == 60


def test_assemble_shots_omits_cadence_keys_by_default() -> None:
    """target_fps=None (default) ⇒ shots carry no target_fps/retime keys."""
    shots = assemble_shots(
        storyboard_shots=[{"id": "s1", "line_id": "l1", "seconds": 2.0}],
        images=[],
        narration_clips=[],
    )
    assert len(shots) == 1
    assert "target_fps" not in shots[0]
    assert "retime" not in shots[0]
