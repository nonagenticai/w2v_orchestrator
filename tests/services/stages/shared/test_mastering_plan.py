"""Unit tests for the mastering profile (pure, no DB).

Covers the two guarantees this PR ships:

1. **Presence-is-opt-in** — an absent (or non-dict) ``mastering`` block yields
   ``None`` from ``build_mastering_profile`` so the compile request is
   byte-identical to today; a present block always yields a profile.
2. **Clamp-on-construct** — :class:`MasteringProfile` pulls its two bounded
   knobs (``crf`` 0..51, ``loudness_target`` -30..-5) back to range at
   construction, but leaves ``None`` knobs absent. The frozen dataclass is
   deterministic (same inputs → equal profiles).
"""

from __future__ import annotations

from dataclasses import replace

from src.services.stages.shared.loudness_contract import COMPILER_MASTER_LUFS
from src.services.stages.shared.mastering_plan import MasteringProfile, build_mastering_profile


def test_absent_mastering_block_returns_none() -> None:
    """No `mastering` key ⇒ None ⇒ compile request byte-identical to today."""
    assert build_mastering_profile({}) is None
    assert build_mastering_profile({"brand": {"primary_color": "#123456"}}) is None


def test_non_dict_mastering_block_returns_none() -> None:
    """A non-dict `mastering` is treated as absent (defensive, still None)."""
    assert build_mastering_profile({"mastering": "on"}) is None
    assert build_mastering_profile({"mastering": None}) is None
    assert build_mastering_profile({"mastering": ["crf"]}) is None


def test_present_block_carries_default_loudness_target() -> None:
    """Any present dict block yields a profile carrying the COMPILER's own default.

    Not -12.5: that literal lived here until 2026-09-16 and disagreed with the -14.0
    the compiler actually applies, so authoring an unrelated knob silently changed a
    run's loudness. See `test_loudness_contract.py`.
    """
    profile = build_mastering_profile({"mastering": {}})
    assert isinstance(profile, MasteringProfile)
    assert profile.loudness_target == COMPILER_MASTER_LUFS
    # Unset optional knobs stay None.
    assert profile.colorspace is None
    assert profile.deband is None
    assert profile.crf is None


def test_explicit_fields_pass_through() -> None:
    """Explicit block fields resolve straight onto the profile."""
    profile = build_mastering_profile(
        {
            "mastering": {
                "colorspace": "bt709",
                "deband": "auto",
                "bitrate_mode": "vbr",
                "maxrate": "12M",
                "crf": 18,
                "loudness_target": -14.0,
                "true_peak": -1.5,
                "platform": "youtube",
            }
        }
    )
    assert profile is not None
    assert profile.colorspace == "bt709"
    assert profile.deband == "auto"
    assert profile.bitrate_mode == "vbr"
    assert profile.maxrate == "12M"
    assert profile.crf == 18
    assert profile.loudness_target == -14.0
    assert profile.true_peak == -1.5
    assert profile.platform == "youtube"


def test_crf_clamps_to_upper_bound() -> None:
    """An out-of-range crf is pulled back to 51 at construction."""
    assert MasteringProfile(crf=99).crf == 51
    built = build_mastering_profile({"mastering": {"crf": 99}})
    assert built is not None
    assert built.crf == 51


def test_crf_none_stays_none() -> None:
    """A None crf is left absent (no clamp), so the compiler keeps its default."""
    assert MasteringProfile().crf is None
    assert MasteringProfile(crf=None).crf is None


def test_loudness_target_clamps_out_of_range() -> None:
    """loudness_target clamps into the sane [-30, -5] window."""
    assert MasteringProfile(loudness_target=-40.0).loudness_target == -30.0
    assert MasteringProfile(loudness_target=0.0).loudness_target == -5.0
    built = build_mastering_profile({"mastering": {"loudness_target": -99.0}})
    assert built is not None
    assert built.loudness_target == -30.0


def test_payload_always_includes_loudness_and_omits_none() -> None:
    """The threaded dict always carries loudness_target and drops unset knobs."""
    payload = MasteringProfile().as_mastering_payload()
    # The default is the COMPILER's own (-14.0), not an orchestrator-local literal —
    # see `test_loudness_contract.py`. This asserted -12.5 until 2026-09-16.
    assert payload == {"loudness_target": COMPILER_MASTER_LUFS}

    full = MasteringProfile(colorspace="bt709", crf=18, platform="tiktok").as_mastering_payload()
    assert full == {
        "loudness_target": COMPILER_MASTER_LUFS,
        "colorspace": "bt709",
        "crf": 18,
        "platform": "tiktok",
    }


def test_frozen_determinism_and_equality() -> None:
    """Same inputs ⇒ equal profiles; a re-clamped round-trip is a fixed point."""
    a = MasteringProfile(crf=99, colorspace="bt709")
    b = MasteringProfile(crf=99, colorspace="bt709")
    assert a == b
    assert replace(a) == a
    assert MasteringProfile(crf=a.crf) == MasteringProfile(crf=51)
