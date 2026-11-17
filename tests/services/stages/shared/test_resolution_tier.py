"""Resolution-tier helpers: tier normalization + format→dims scaling.

Pure-function unit tests (no DB) for the 1080p|2k|4k knob that the run threads
into every motion clip, slide PNG, and the terminal compile canvas.
"""

from __future__ import annotations

from uuid import uuid4

from src.services.runs.run_context import RunContext
from src.services.stages.primitives.compile_stage import (
    output_dimension_fields,
    resolution_tags,
)
from src.services.stages.shared.motion_slides import (
    format_dims,
    quality_tier,
    tier_scale,
)


def _ctx(
    resolution: str | None,
    fmt: str = "16x9",
    mastering: dict[str, object] | None = None,
) -> RunContext:
    snap: dict[str, object] = {"deliverable": {"format": fmt}}
    if resolution is not None:
        snap["resolution"] = resolution
    if mastering is not None:
        snap["mastering"] = mastering
    return RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot=snap)


def test_quality_tier_normalizes_known_values() -> None:
    assert quality_tier({"resolution": "1080p"}) == "1080p"
    assert quality_tier({"resolution": "2K"}) == "2k"  # case-insensitive
    assert quality_tier({"resolution": " 4k "}) == "4k"  # trimmed


def test_quality_tier_defaults_to_2k_when_missing_or_garbage() -> None:
    assert quality_tier({}) == "2k"
    assert quality_tier(None) == "2k"
    assert quality_tier({"resolution": "ultra"}) == "2k"
    assert quality_tier({"resolution": ""}) == "2k"


def test_format_dims_1080p_is_the_base() -> None:
    assert format_dims("16x9", "1080p") == (1920, 1080)
    assert format_dims("9x16", "1080p") == (1080, 1920)
    assert format_dims("1x1", "1080p") == (1080, 1080)


def test_format_dims_scales_to_2k_and_4k_with_even_dims() -> None:
    # 16x9 → exact 2K/4K
    assert format_dims("16x9", "2k") == (2560, 1440)
    assert format_dims("16x9", "4k") == (3840, 2160)
    # vertical scales proportionally, both dims even
    w2k, h2k = format_dims("9x16", "2k")
    assert (w2k, h2k) == (1440, 2560)
    assert w2k % 2 == 0 and h2k % 2 == 0
    # 4x5 (1080x1350) @ 2k → even on both axes (no odd-dim ffmpeg failure)
    w, h = format_dims("4x5", "2k")
    assert w % 2 == 0 and h % 2 == 0


def test_unknown_format_falls_back_to_16x9() -> None:
    assert format_dims("garbage", "4k") == (3840, 2160)
    assert format_dims(None, "1080p") == (1920, 1080)


def test_tier_scale_matches_render_scale_for_slides() -> None:
    assert tier_scale("1080p") == 1.0
    assert tier_scale("4k") == 2.0
    assert tier_scale("2k") == 2560 / 1920
    assert tier_scale("nope") == 1.0


def test_resolution_tags_emit_canvas_and_crf_when_no_authored_crf() -> None:
    """No authored ``mastering.crf`` ⇒ the tier CRF tag rides (byte-identical to
    the pre-mastering-gate behaviour). compile_tags forwards these to
    video_compiler_service to set the canvas + CRF per job. Tag values must be
    strings (the compile tags dict is str→str)."""
    tags_4k = resolution_tags(_ctx("4k"))
    assert tags_4k == {"render_width": "3840", "render_height": "2160", "crf": "18"}

    tags_2k = resolution_tags(_ctx("2k"))
    assert tags_2k == {"render_width": "2560", "render_height": "1440", "crf": "18"}

    # Missing resolution → default tier (2k), never a silent 1080p.
    assert resolution_tags(_ctx(None)) == tags_2k

    # A `mastering` block that does NOT pin crf still emits the tier tag — only an
    # authored crf suppresses it (presence-is-opt-in on the crf field itself).
    assert resolution_tags(_ctx("4k", mastering={"loudness_target": -12.5})) == tags_4k


def test_resolution_tags_omit_crf_when_mastering_crf_authored() -> None:
    """An authored ``mastering.crf`` MUST suppress the tier CRF tag so the
    compiler falls back to the mastering-block crf instead of tag-shadowing it
    (video_compiler prefers an explicit compile-tag: ``if _key not in tags``).
    render_width/render_height are untouched (canvas is never authored via
    mastering)."""
    tags = resolution_tags(_ctx("4k", mastering={"crf": 14}))
    assert tags == {"render_width": "3840", "render_height": "2160"}
    assert "crf" not in tags

    # Holds at the default (missing-resolution) tier too.
    tags_default = resolution_tags(_ctx(None, mastering={"crf": 23}))
    assert tags_default == {"render_width": "2560", "render_height": "1440"}


def test_output_dimension_fields_extracts_real_dims() -> None:
    """The compile result's ffprobed dims become stage-output width/height;
    absent/zero/garbage values are dropped so the output stays None, not 0."""
    assert output_dimension_fields({"width": 3840, "height": 2160}) == {
        "width": 3840,
        "height": 2160,
    }
    # Strings (JSON round-trip) coerce to int.
    assert output_dimension_fields({"width": "2560", "height": "1440"}) == {
        "width": 2560,
        "height": 1440,
    }
    # Absent (older compiler / audio job), zero (probe miss), garbage → dropped.
    assert output_dimension_fields({}) == {}
    assert output_dimension_fields({"width": 0, "height": 0}) == {}
    assert output_dimension_fields({"width": None, "height": "nope"}) == {}


# The compiler's OWN delivery CRF — `video_compiler_service/src/config.py`
# ``render_crf`` default. Duplicated as a literal because that service is a
# separate repo and not importable from here; if it moves, this test is the
# thing that must be re-verified against it by hand.
_COMPILER_DELIVERY_CRF = 18


def test_tier_crf_never_tag_shadows_the_compilers_delivery_crf() -> None:
    """EVERY tier must tag the compiler's own delivery CRF, not a looser one.

    `video_compiler` prefers an explicit compile tag over its env default
    (``if _key not in tags``), so any tier value here SHADOWS ``render_crf``.
    The table used to read ``{1080p: 20, 2k: 19, 4k: 18}`` against a compiler
    default of 18, and because ``quality_tier`` normalises an unset tier to
    ``2k``, every production master without an authored ``mastering.crf``
    encoded at 19 — silently discarding the compiler's 18. It also dragged the
    intermediate polish pass down with it (``encode_ladder`` = ``crf - 6``, so
    13 instead of 12).

    Asserted as an INEQUALITY per tier rather than a table snapshot: going
    SHARPER than the compiler's default stays legal (a deliberate quality
    decision), but going SOFTER is the defect, and a softer cell can never
    re-enter the table unnoticed again.
    """
    for tier in ("1080p", "2k", "4k"):
        crf = int(resolution_tags(_ctx(tier))["crf"])
        assert crf <= _COMPILER_DELIVERY_CRF, (
            f"tier {tier!r} tags crf={crf}, which is SOFTER than the compiler's "
            f"own {_COMPILER_DELIVERY_CRF} and tag-shadows it away"
        )


def test_tier_crf_is_flat_so_1080p_is_never_the_softest_cell() -> None:
    """No tier may be softer than a LARGER one.

    x264's CRF is a constant-QUALITY target and is deliberately
    resolution-independent, so a per-tier ramp encodes an assumption the encoder
    does not share. The old ramp also got its own stated intent backwards: it
    put 1080p at 20, the softest cell in the table, when fewer pixels means each
    quantisation artifact covers a LARGER fraction of the frame — 1080p needs at
    least the same CRF as 2K, never a looser one.
    """
    crf_1080p = int(resolution_tags(_ctx("1080p"))["crf"])
    crf_2k = int(resolution_tags(_ctx("2k"))["crf"])
    crf_4k = int(resolution_tags(_ctx("4k"))["crf"])
    assert crf_1080p <= crf_2k <= crf_4k, (
        f"a smaller canvas is tagged SOFTER than a larger one: "
        f"1080p={crf_1080p}, 2k={crf_2k}, 4k={crf_4k}"
    )


def test_default_tier_crf_is_the_one_production_actually_gets() -> None:
    """The unset-resolution path is the production path — pin it explicitly.

    ``_DEFAULT_TIER`` is ``2k``, so this is the CRF the overwhelming majority of
    delivered masters carry. A regression here is invisible in the per-tier
    tests above if someone "fixes" only the explicitly-named tiers.
    """
    assert resolution_tags(_ctx(None))["crf"] == str(_COMPILER_DELIVERY_CRF)
    assert quality_tier({}) == "2k"


def test_the_unknown_tier_fallback_is_also_pinned_to_the_delivery_crf() -> None:
    """Defense-in-depth on a path that is UNREACHABLE today, asserted directly.

    ``quality_tier`` normalises anything it does not recognise to ``2k``, so
    ``_CRF_BY_TIER.get(tier, _CRF_FALLBACK)`` can never take its default via
    ``resolution_tags`` — a behavioural test cannot reach it, and mutating the
    fallback leaves every other test in this file green. The constant is pinned
    on its own so that a tier added to ``_TIER_SCALE`` WITHOUT a row in
    ``_CRF_BY_TIER`` degrades to the compiler's delivery CRF rather than
    silently reintroducing the soft 20 this change removed.
    """
    from src.services.stages.primitives.compile_stage import (
        _CRF_BY_TIER,
        _CRF_FALLBACK,
    )

    assert _CRF_FALLBACK == _COMPILER_DELIVERY_CRF
    # And the table covers every tier that actually exists, so the fallback stays
    # unreachable rather than becoming load-bearing by accident.
    from src.services.stages.shared.motion_slides import _TIER_SCALE

    assert set(_TIER_SCALE) == set(_CRF_BY_TIER)
