"""Each terminal stage RECORDS its delivered-vs-ordered duration on its output.

The helper's own arithmetic is pinned in
``tests/services/stages/shared/test_duration_conformance.py``. This file pins the
other half — that the four blueprints measured over-delivering on 2026-09-10
actually carry the comparison out of ``build_output``, resolved from the SAME
ordered value their script stage uses. A helper nobody wired in would be exactly
as invisible as the deviation it exists to surface.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.app_review.compile import AppReviewCompileStage
from src.services.stages.lesson.compile import LessonCompileStage
from src.services.stages.promo.render import PromoRenderStage
from src.services.stages.shared.degradation import is_degraded
from src.services.stages.website_review.compile import WebsiteReviewCompileStage

_RESULT_TEMPLATE: dict[str, Any] = {"artifact_url": "http://stub/master.mp4"}


def _ctx(config_snapshot: dict[str, Any]) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=config_snapshot,
        prior_run_outputs={},
    )


def _result(duration_seconds: float) -> dict[str, Any]:
    return {**_RESULT_TEMPLATE, "duration_seconds": duration_seconds}


# (stage factory, config_snapshot ordering 120s, delivered, expected ratio)
_CASES = [
    pytest.param(
        PromoRenderStage,
        {"brief": {"target_duration_seconds": 30}, "deliverable": {"format": "16x9"}},
        38.7,
        1.29,
        id="promo_render",
    ),
    pytest.param(
        LessonCompileStage,
        {"brief": {"duration_minutes": 6}, "deliverable": {"format": "16x9"}},
        777.6,
        2.16,
        id="lesson_compile",
    ),
    pytest.param(
        WebsiteReviewCompileStage,
        {"brief": {"duration_seconds": 120}, "deliverable": {"format": "16x9"}},
        327.6,
        2.73,
        id="website_review_compile",
    ),
    pytest.param(
        AppReviewCompileStage,
        {"brief": {"duration_seconds": 120}, "deliverable": {"format": "16x9"}},
        168.0,
        1.40,
        id="app_review_compile",
    ),
]


@pytest.mark.parametrize(
    ("stage_cls", "config", "delivered", "expected_ratio"),
    [
        pytest.param(
            PromoRenderStage,
            {"brief": {"target_duration_seconds": 30}, "deliverable": {"format": "16x9"}},
            38.7,
            1.29,
            id="promo_render",
        )
    ],
)
def test_over_delivery_reaches_the_stage_output(
    stage_cls: Any, config: dict[str, Any], delivered: float, expected_ratio: float
) -> None:
    """The 2026-09-10 measured over-delivery, per blueprint, now in the record."""
    output = stage_cls().build_output(_result(delivered), _ctx(config))

    conformance = output.duration_conformance
    assert conformance is not None, "the delivered-vs-ordered comparison was not recorded"
    assert conformance["delivered_seconds"] == pytest.approx(delivered)
    assert conformance["ratio"] == pytest.approx(expected_ratio, abs=1e-3)
    assert conformance["within_tolerance"] is False
    # And the delivered figure the stage already shipped is untouched — this is
    # an observation, never a truncation.
    assert output.duration_seconds == pytest.approx(delivered)


@pytest.mark.parametrize(
    ("stage_cls", "config", "delivered", "expected_ratio"),
    [
        pytest.param(
            PromoRenderStage,
            {"brief": {"target_duration_seconds": 30}, "deliverable": {"format": "16x9"}},
            38.7,
            1.29,
            id="promo_render",
        )
    ],
)
def test_on_target_delivery_is_within_tolerance(
    stage_cls: Any, config: dict[str, Any], delivered: float, expected_ratio: float
) -> None:
    """Same wiring, conforming master: the record says so rather than staying empty."""
    ordered = delivered / expected_ratio
    output = stage_cls().build_output(_result(ordered), _ctx(config))

    conformance = output.duration_conformance
    assert conformance is not None
    assert conformance["within_tolerance"] is True
    assert conformance["ratio"] == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize(
    ("stage_cls", "config", "delivered", "expected_ratio"),
    [
        pytest.param(
            PromoRenderStage,
            {"brief": {"target_duration_seconds": 30}, "deliverable": {"format": "16x9"}},
            38.7,
            1.29,
            id="promo_render",
        )
    ],
)
def test_conformance_survives_model_dump(
    stage_cls: Any, config: dict[str, Any], delivered: float, expected_ratio: float
) -> None:
    """The record must reach the PERSISTED output, not just the in-memory model.

    Stage outputs are stored via ``model_dump``; a field that serialises away is
    invisible in the run record, which is the entire defect being fixed.
    """
    dumped = stage_cls().build_output(_result(delivered), _ctx(config)).model_dump()

    assert dumped["duration_conformance"] is not None
    assert dumped["duration_conformance"]["within_tolerance"] is False
    assert dumped["duration_conformance"]["ratio"] == pytest.approx(expected_ratio, abs=1e-3)


def test_promo_reads_the_deliverable_when_the_brief_omits_the_length() -> None:
    """promo_render must follow promo/script.py's precedence, not brief-only.

    Reading only ``brief`` is a bug that stage already shipped once: every promo
    defaulted to 30s while the length the FE set on the deliverable was ignored.
    A 60s deliverable delivered in 60s is ON TARGET; brief-only resolution would
    score it 200% against a phantom 30s order.
    """
    ctx = _ctx({"brief": {}, "deliverable": {"format": "16x9", "target_duration_seconds": 60}})
    conformance = PromoRenderStage().build_output(_result(60.0), ctx).duration_conformance

    assert conformance is not None
    assert conformance["ordered_seconds"] == pytest.approx(60.0)
    assert conformance["within_tolerance"] is True


# --------------------------------------------------------------------------
# The MEASURED length, not the declared one, is what the verdict is taken against
# --------------------------------------------------------------------------
# Until 2026-09-14 every site above passed `result["duration_seconds"]` as the
# delivered length. That field is the compiler's NOMINAL SUM OF SHOT DURATIONS
# (`primitives/compile_stage.py:192-195`) — the intended timeline. Verified on 38
# live production runs: the recorded "delivered" was byte-identical to the
# declared value on every one, so the verdict was the plan checking itself — a
# gate that cannot fail, and structurally blind to the one defect it exists to
# catch (a master whose picture stopped early still DECLARES the length it meant
# to be).
#
# ⛔ The measurement is the PLAYABLE length off the `mastering` block, NOT
# `actual_duration_seconds`. That field is min(video_stream, container) — the
# PICTURE length — and grading it is the second regression, pinned in the block
# further down; it has no rung at all any more.
#
# Every case below is shaped exactly like the original defect: the declared value
# is ON TARGET (a `declared` verdict would be `within_tolerance=True`, ratio 1.0)
# while the measurement says 60% of the order. The two therefore disagree ACROSS
# the +/-10% band, so an assertion here cannot be satisfied by the old wiring.


def _result_with_playable(declared: float, playable: float) -> dict[str, Any]:
    """A compile result whose two probed streams agree at *playable*."""
    return {
        **_RESULT_TEMPLATE,
        "duration_seconds": declared,
        "mastering": {
            "video_stream_seconds": playable,
            "audio_stream_seconds": playable,
        },
    }


@pytest.mark.parametrize(
    ("stage_cls", "config", "delivered", "expected_ratio"),
    [
        pytest.param(
            PromoRenderStage,
            {"brief": {"target_duration_seconds": 30}, "deliverable": {"format": "16x9"}},
            38.7,
            1.29,
            id="promo_render",
        )
    ],
)
def test_the_measured_length_wins_over_the_declared_one(
    stage_cls: Any, config: dict[str, Any], delivered: float, expected_ratio: float
) -> None:
    """A short master: plan says on-target, the streams say 60%. Measurement wins."""
    ordered = delivered / expected_ratio
    playable = ordered * 0.6

    output = stage_cls().build_output(_result_with_playable(ordered, playable), _ctx(config))

    conformance = output.duration_conformance
    assert conformance is not None
    assert conformance["delivered_basis"] == "playable"
    assert conformance["delivered_seconds"] == pytest.approx(playable, abs=1e-3)
    # ... and NOT the declared value, which is what the old wiring recorded.
    assert conformance["delivered_seconds"] != pytest.approx(ordered, abs=1e-3)
    assert conformance["ratio"] == pytest.approx(0.6, abs=1e-3)
    # The declared-basis verdict on this same result would have been a clean pass.
    assert conformance["within_tolerance"] is False
    assert [d["code"] for d in output.degradations] == ["duration_out_of_tolerance"]
    # A MEASURED verdict raises no unmeasured advisory — the tier does not leak.
    assert output.advisories == []


@pytest.mark.parametrize(
    ("stage_cls", "config", "delivered", "expected_ratio"),
    [
        pytest.param(
            PromoRenderStage,
            {"brief": {"target_duration_seconds": 30}, "deliverable": {"format": "16x9"}},
            38.7,
            1.29,
            id="promo_render",
        )
    ],
)
def test_an_unmeasured_result_still_falls_back_to_the_declared_length(
    stage_cls: Any, config: dict[str, Any], delivered: float, expected_ratio: float
) -> None:
    """`0.0` is the compiler's "probe failed" sentinel, never a zero-length video.

    Passing the measurement through must not turn a failed probe into a reported
    0% delivery, so the declared value stays as the fallback and the record is
    stamped `declared` rather than silently reading as a measured verdict.

    And the fallback SAYS SO, on the advisory channel: `duration_unmeasured`
    reaches the built output, so "nobody measured this" is queryable rather than
    byte-identical to a measured pass. It is an ADVISORY, never a degradation —
    it fires on every unmeasured stage, so on `degradations` it would light the
    flag on healthy runs (`shared/degradation.py::ADVISORY_CODES`).
    """
    output = stage_cls().build_output(_result_with_playable(delivered, 0.0), _ctx(config))

    conformance = output.duration_conformance
    assert conformance is not None
    assert conformance["delivered_basis"] == "declared"
    assert conformance["delivered_seconds"] == pytest.approx(delivered)
    assert conformance["ratio"] == pytest.approx(expected_ratio, abs=1e-3)
    assert [a["code"] for a in output.advisories] == ["duration_unmeasured"]
    # ...and it survives to the PERSISTED blob, like the structured field above.
    assert output.model_dump()["advisories"][0]["code"] == "duration_unmeasured"


@pytest.mark.parametrize(
    ("stage_cls", "config", "delivered", "expected_ratio"),
    [
        pytest.param(
            PromoRenderStage,
            {"brief": {"target_duration_seconds": 30}, "deliverable": {"format": "16x9"}},
            38.7,
            1.29,
            id="promo_render",
        )
    ],
)
def test_an_unmeasured_conforming_master_is_not_degraded(
    stage_cls: Any, config: dict[str, Any], delivered: float, expected_ratio: float
) -> None:
    """⭐ THE TIER, at the wiring level: unmeasured must not flip `degraded`.

    `duration_unmeasured` fires on every stage with no usable mastering report,
    which is most of them most of the time. Carried on `degradations` it reported
    conforming masters degraded — for `demo_video`/`showcase`, 100% of them,
    permanently. A conforming unmeasured compile output must therefore read CLEAN
    through `is_degraded` while STILL carrying the advisory: a run that is clean
    because the signal was dropped is the opposite failure, not a fix.
    """
    ordered = delivered / expected_ratio
    output = stage_cls().build_output(_result_with_playable(ordered, 0.0), _ctx(config))

    assert output.duration_conformance is not None
    assert output.duration_conformance["within_tolerance"] is True
    assert output.degradations == []
    assert [a["code"] for a in output.advisories] == ["duration_unmeasured"]
    assert is_degraded(output.model_dump()) is False


# --------------------------------------------------------------------------- #
# ⭐ THE 2026-09-14 REGRESSION, at the wiring level.
#
# The block above wired `actual_duration_seconds` in as the delivered length.
# That field is `min(video_stream, container)`
# (`video_compiler_service/src/services/renderer.py::_resolve_probed_duration`,
# whose docstring says the stream wins precisely because `format=duration` is the
# MAX over streams) — i.e. the PICTURE length. On a master whose picture stopped
# early it IS the truncated number, so the fix inverted into its own failure
# mode: an over-long, broken master graded as an on-target delivery.
#
# `delivered_seconds` is now the PLAYABLE length — `max(video_stream_seconds,
# audio_stream_seconds)` off the compiler's mastering report, which probes the
# two streams separately for exactly this reason — and truncation is a SEPARATE
# code. Every case below is shaped like the live defect: order 120, plan declared
# 180, picture truncated at 118 under 180s of audio.
# --------------------------------------------------------------------------- #

_TRUNCATED_ORDER = 120.0
_TRUNCATED_DECLARED = 180.0
_TRUNCATED_PICTURE = 118.0
_TRUNCATED_AUDIO = 180.0


def _result_truncated(
    declared: float = _TRUNCATED_DECLARED,
    picture: float = _TRUNCATED_PICTURE,
    audio: float = _TRUNCATED_AUDIO,
) -> dict[str, Any]:
    """A compile result whose picture stops before its audio does."""
    return {
        **_RESULT_TEMPLATE,
        "duration_seconds": declared,
        # What `_resolve_probed_duration` reports for such a mux: the picture.
        "actual_duration_seconds": picture,
        "mastering": {
            "video_stream_seconds": picture,
            "audio_stream_seconds": audio,
            "av_gap_seconds": audio - picture,
        },
    }


def _ordered_120(config: dict[str, Any], stage_cls: Any) -> dict[str, Any]:
    """The same config, re-pointed so every blueprint's resolver orders 120s."""
    if stage_cls is PromoRenderStage:
        return {**config, "brief": {"target_duration_seconds": _TRUNCATED_ORDER}}
    if stage_cls is LessonCompileStage:
        return {**config, "brief": {"duration_minutes": _TRUNCATED_ORDER / 60.0}}
    return {**config, "brief": {"duration_seconds": _TRUNCATED_ORDER}}


@pytest.mark.parametrize(
    ("stage_cls", "config", "delivered", "expected_ratio"),
    [
        pytest.param(
            PromoRenderStage,
            {"brief": {"target_duration_seconds": 30}, "deliverable": {"format": "16x9"}},
            38.7,
            1.29,
            id="promo_render",
        )
    ],
)
def test_a_truncated_master_is_graded_on_its_playable_length(
    stage_cls: Any, config: dict[str, Any], delivered: float, expected_ratio: float
) -> None:
    """⭐ THE REGRESSION. 180s of audio over 118s of picture, on a 120s order.

    BEFORE #1007 the declared 180 caught it (+50%) by luck — the plan happened to
    be honest about the length it meant. AFTER #1007 the picture-only probe of
    118 reported -1.7%, `within_tolerance=true`, zero degradations: a truncated,
    50%-over master indistinguishable from a clean delivery.

    Now the playable length (max of the two streams = 180) is graded, so the
    over-delivery is caught on a MEASURED number, and the truncation the length
    alone cannot express rides its own code beside it.
    """
    output = stage_cls().build_output(_result_truncated(), _ctx(_ordered_120(config, stage_cls)))

    conformance = output.duration_conformance
    assert conformance is not None
    assert conformance["delivered_basis"] == "playable"
    assert conformance["delivered_seconds"] == pytest.approx(_TRUNCATED_AUDIO)
    # ...and NOT the picture length, which is what the regression recorded.
    assert conformance["delivered_seconds"] != pytest.approx(_TRUNCATED_PICTURE)
    assert conformance["deviation_pct"] == pytest.approx(50.0)
    assert conformance["within_tolerance"] is False

    codes = [d["code"] for d in output.degradations]
    assert "duration_out_of_tolerance" in codes
    assert "duration_picture_truncated" in codes
    # A measured verdict raises no unmeasured advisory — the tier does not leak.
    assert output.advisories == []
    assert is_degraded(output.model_dump()) is True


@pytest.mark.parametrize(
    ("stage_cls", "config", "delivered", "expected_ratio"),
    [
        pytest.param(
            PromoRenderStage,
            {"brief": {"target_duration_seconds": 30}, "deliverable": {"format": "16x9"}},
            38.7,
            1.29,
            id="promo_render",
        )
    ],
)
def test_a_healthy_master_carries_no_truncation_claim(
    stage_cls: Any, config: dict[str, Any], delivered: float, expected_ratio: float
) -> None:
    """video ~= audio, on target: unchanged from before, and nothing new fires."""
    ordered = _TRUNCATED_ORDER
    result = _result_truncated(declared=ordered, picture=ordered, audio=ordered + 0.04)

    output = stage_cls().build_output(result, _ctx(_ordered_120(config, stage_cls)))

    conformance = output.duration_conformance
    assert conformance is not None
    assert conformance["delivered_basis"] == "playable"
    assert conformance["within_tolerance"] is True
    assert output.degradations == []
    assert output.advisories == []
    assert is_degraded(output.model_dump()) is False


@pytest.mark.parametrize(
    ("stage_cls", "config", "delivered", "expected_ratio"),
    [
        pytest.param(
            PromoRenderStage,
            {"brief": {"target_duration_seconds": 30}, "deliverable": {"format": "16x9"}},
            38.7,
            1.29,
            id="promo_render",
        )
    ],
)
def test_a_showcase_shaped_master_is_not_reported_truncated(
    stage_cls: Any, config: dict[str, Any], delivered: float, expected_ratio: float
) -> None:
    """Picture 5.6s LONGER than audio — measured live on showcase masters.

    The guard is one-sided for this reason: a held final frame or an outro title
    card is authored, not broken, and a symmetric guard would report every
    healthy showcase run as a defective artifact.
    """
    audio = _TRUNCATED_ORDER
    result = _result_truncated(declared=audio, picture=audio + 5.6, audio=audio)

    output = stage_cls().build_output(result, _ctx(_ordered_120(config, stage_cls)))

    assert [d["code"] for d in output.degradations] == []
    conformance = output.duration_conformance
    assert conformance is not None
    # The playable length is the LONGER stream — the picture, here.
    assert conformance["delivered_seconds"] == pytest.approx(audio + 5.6)


@pytest.mark.parametrize(
    ("stage_cls", "config", "delivered", "expected_ratio"),
    [
        pytest.param(
            PromoRenderStage,
            {"brief": {"target_duration_seconds": 30}, "deliverable": {"format": "16x9"}},
            38.7,
            1.29,
            id="promo_render",
        )
    ],
)
def test_an_absent_mastering_block_falls_to_declared_not_to_the_picture(
    stage_cls: Any, config: dict[str, Any], delivered: float, expected_ratio: float
) -> None:
    """⛔ NO mastering report means UNMEASURED — never "graded on the picture".

    The verifier's finding, at the wiring level. `mastering.degraded_report()`
    nulls BOTH stream fields when `post_mux_verify` crashes, so a compile result
    can arrive carrying `actual_duration_seconds` and NOTHING usable in
    `mastering` — precisely the population whose verification failed. A
    picture-only rung graded that population on its truncated half and passed it.

    There is no such rung. Both shapes below fall to `declared` and say so:
    `actual_duration_seconds` present and on-target, and nothing measured at all.
    An unmeasured GAP must never be reported as a defect, and an unmeasured
    LENGTH must stamp `declared` and raise the advisory — two different absences,
    and neither may masquerade as the other.
    """
    ordered = _TRUNCATED_ORDER

    # A picture probe with no mastering block to improve on it: NOT a rung.
    picture_only = stage_cls().build_output(
        {**_RESULT_TEMPLATE, "duration_seconds": ordered, "actual_duration_seconds": ordered},
        _ctx(_ordered_120(config, stage_cls)),
    )
    assert picture_only.duration_conformance is not None
    assert picture_only.duration_conformance["delivered_basis"] == "declared"
    assert [a["code"] for a in picture_only.advisories] == ["duration_unmeasured"]
    assert [d["code"] for d in picture_only.degradations] == []

    # Nothing measured at all — the same verdict, for the same reason.
    declared_only = stage_cls().build_output(
        _result(ordered), _ctx(_ordered_120(config, stage_cls))
    )
    assert declared_only.duration_conformance is not None
    assert declared_only.duration_conformance["delivered_basis"] == "declared"
    assert [a["code"] for a in declared_only.advisories] == ["duration_unmeasured"]
    assert [d["code"] for d in declared_only.degradations] == []


@pytest.mark.parametrize(
    ("video", "audio"),
    [(0.0, 180.0)],
    ids=["zero-video"],
)
def test_an_unusable_stream_pair_falls_back_and_claims_nothing(
    video: object, audio: object
) -> None:
    """0.0 is the UNKNOWN sentinel; a bool or NaN is corrupt wire data.

    Either way the pair is unusable, so the stage falls to the DECLARED length —
    not to the picture probe, which is present here and deliberately ignored —
    and reports no truncation: never a 0%-of-order catastrophe, never a
    truncation claim manufactured out of a missing measurement, and never a
    verdict silently taken on half a master.
    """
    result = {
        **_RESULT_TEMPLATE,
        "duration_seconds": _TRUNCATED_ORDER,
        "actual_duration_seconds": _TRUNCATED_ORDER,
        "mastering": {"video_stream_seconds": video, "audio_stream_seconds": audio},
    }
    config = {"brief": {"duration_seconds": _TRUNCATED_ORDER}, "deliverable": {"format": "16x9"}}

    output = AppReviewCompileStage().build_output(result, _ctx(config))

    assert output.duration_conformance is not None
    assert output.duration_conformance["delivered_basis"] == "declared"
    assert output.duration_conformance["delivered_seconds"] == pytest.approx(_TRUNCATED_ORDER)
    assert [a["code"] for a in output.advisories] == ["duration_unmeasured"]
    assert [d["code"] for d in output.degradations] == []
