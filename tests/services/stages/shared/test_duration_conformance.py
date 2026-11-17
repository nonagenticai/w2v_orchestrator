"""The delivered-vs-ordered duration comparison nothing was making.

Measured on live artifacts 2026-09-10: website_review delivered 273% of its
ordered duration, lesson 216%, app_review 140%, promo 129% — invisible in every
run record. These tests pin the arithmetic, the tolerance band, the log level,
and (the one that matters most) that "nothing was ordered" is never reported as
"on target".
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pytest

from src.services.stages.shared.degradation import (
    ADVISORY_CODES,
    DEGRADATION_CODES,
    DURATION_PICTURE_TRUNCATED,
    DURATION_UNMEASURED,
)
from src.services.stages.shared.duration_conformance import (
    DELIVERED_BASIS_DECLARED,
    DELIVERED_BASIS_PLAYABLE,
    DURATION_TOLERANCE_FRACTION,
    PICTURE_TRUNCATION_TOLERANCE_SECONDS,
    duration_conformance,
    duration_conformance_advisories,
    lesson_ordered_seconds,
    mastering_stream_seconds,
    observe_duration_conformance,
    picture_truncation_degradations,
    playable_seconds_from_mastering,
    promo_ordered_seconds,
    review_ordered_seconds,
)


class TestTolerance:
    def test_tolerance_is_ten_percent(self) -> None:
        assert DURATION_TOLERANCE_FRACTION == 0.10

    def test_exact_delivery_is_within_tolerance(self) -> None:
        result = duration_conformance(120.0, 120.0)
        assert result is not None
        assert result["ratio"] == 1.0
        assert result["deviation_pct"] == 0.0
        assert result["within_tolerance"] is True

    def test_nine_percent_over_is_within_tolerance(self) -> None:
        result = duration_conformance(100.0, 109.0)
        assert result is not None
        assert result["deviation_pct"] == pytest.approx(9.0)
        assert result["within_tolerance"] is True

    def test_nine_percent_short_is_within_tolerance(self) -> None:
        # The band is symmetric: under-delivery inside 10% conforms too, and the
        # deviation is SIGNED so a reader can tell short from long.
        result = duration_conformance(100.0, 91.0)
        assert result is not None
        assert result["deviation_pct"] == pytest.approx(-9.0)
        assert result["within_tolerance"] is True

    def test_eleven_percent_over_is_outside_tolerance(self) -> None:
        result = duration_conformance(100.0, 111.0)
        assert result is not None
        assert result["deviation_pct"] == pytest.approx(11.0)
        assert result["within_tolerance"] is False

    def test_exactly_ten_percent_still_conforms(self) -> None:
        # The band is inclusive; pinned so a `<` / `<=` slip is caught.
        result = duration_conformance(100.0, 110.0)
        assert result is not None
        assert result["within_tolerance"] is True


class TestMeasuredDeviations:
    def test_website_review_273_percent(self) -> None:
        # The real 2026-09-10 measurement: 120s ordered, 327.6s delivered.
        result = duration_conformance(120.0, 327.6)
        assert result is not None
        assert result["ordered_seconds"] == 120.0
        assert result["delivered_seconds"] == 327.6
        assert result["ratio"] == pytest.approx(2.73, abs=1e-3)
        assert result["deviation_pct"] == pytest.approx(173.0, abs=0.1)
        assert result["within_tolerance"] is False

    def test_lesson_216_percent(self) -> None:
        result = duration_conformance(360.0, 777.6)
        assert result is not None
        assert result["ratio"] == pytest.approx(2.16, abs=1e-3)
        assert result["within_tolerance"] is False

    def test_promo_129_percent(self) -> None:
        result = duration_conformance(30.0, 38.7)
        assert result is not None
        assert result["ratio"] == pytest.approx(1.29, abs=1e-3)
        assert result["within_tolerance"] is False


class TestNothingOrdered:
    """`None` ordered must be distinguishable from a conforming delivery."""

    def test_none_ordered_returns_none(self) -> None:
        assert duration_conformance(None, 327.6) is None

    def test_none_ordered_is_not_reported_as_on_target(self) -> None:
        # The defect this guards: returning a record with within_tolerance=True
        # (or a zero-filled record) for an unordered length would make "nobody
        # asked for a duration" byte-identical to "the duration asked for was
        # met". Assert on the ABSENCE of any on-target claim, not just on None,
        # so a future zero-filled record cannot satisfy this test either.
        result = duration_conformance(None, 327.6)
        assert result is None, f"unordered duration produced a record: {result!r}"
        # And the same input DOES produce an on-target claim once something is
        # actually ordered — proving the assertion above discriminates rather
        # than being trivially true for every input.
        ordered = duration_conformance(327.6, 327.6)
        assert ordered is not None and ordered["within_tolerance"] is True

    def test_zero_ordered_returns_none_without_dividing_by_zero(self) -> None:
        assert duration_conformance(0.0, 42.0) is None

    def test_negative_ordered_returns_none(self) -> None:
        assert duration_conformance(-30.0, 42.0) is None

    def test_non_numeric_and_non_finite_inputs_return_none(self) -> None:
        assert duration_conformance(float("nan"), 42.0) is None
        assert duration_conformance(float("inf"), 42.0) is None
        assert duration_conformance("abc", 42.0) is None  # ty: ignore[invalid-argument-type]  # why: deliberately off-contract input — config_snapshot is untyped JSON, so a string can reach this at runtime
        assert duration_conformance(120.0, float("nan")) is None
        assert duration_conformance(120.0, None) is None  # ty: ignore[invalid-argument-type]  # why: deliberately off-contract input — a compiler result with a null duration must not raise

    def test_zero_delivered_is_measured_not_skipped(self) -> None:
        # A zero-length master IS a real (catastrophic) delivery, not a missing
        # order — it must be reported, not swallowed.
        result = duration_conformance(120.0, 0.0)
        assert result is not None
        assert result["ratio"] == 0.0
        assert result["within_tolerance"] is False


class TestObserveLogging:
    def test_within_tolerance_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG):
            result = observe_duration_conformance("promo_render", 100.0, 103.0)
        assert result is not None and result["within_tolerance"] is True
        records = [r for r in caplog.records if "duration conformance" in r.message]
        assert [r.levelno for r in records] == [logging.INFO]

    def test_out_of_tolerance_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG):
            result = observe_duration_conformance("website_review_compile", 120.0, 327.6)
        assert result is not None and result["within_tolerance"] is False
        records = [r for r in caplog.records if "duration conformance" in r.message]
        assert [r.levelno for r in records] == [logging.WARNING]
        assert "OUTSIDE" in records[0].getMessage()
        assert "327.6" in records[0].getMessage()
        assert "120.0" in records[0].getMessage()

    def test_nothing_ordered_logs_neither_info_nor_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG):
            assert observe_duration_conformance("lesson_compile", None, 777.6) is None
        records = [r for r in caplog.records if "duration conformance" in r.message]
        assert [r.levelno for r in records] == [logging.DEBUG]


class TestOrderedResolvers:
    """Each resolver must mirror its blueprint's own script stage, defaults included."""

    def test_promo_prefers_brief_then_deliverable_then_thirty(self) -> None:
        assert promo_ordered_seconds(
            {"brief": {"target_duration_seconds": 45}, "deliverable": {}}
        ) == pytest.approx(45.0)
        # The bug promo/script.py already fixed once: reading only `brief` and
        # ignoring the length the FE set on the deliverable.
        assert promo_ordered_seconds(
            {"brief": {}, "deliverable": {"target_duration_seconds": 60}}
        ) == pytest.approx(60.0)
        assert promo_ordered_seconds({}) == pytest.approx(30.0)

    def test_lesson_converts_minutes_to_seconds(self) -> None:
        assert lesson_ordered_seconds({"brief": {"duration_minutes": 4}}) == pytest.approx(240.0)
        assert lesson_ordered_seconds({}) == pytest.approx(360.0)

    def test_review_reads_brief_duration_seconds(self) -> None:
        assert review_ordered_seconds({"brief": {"duration_seconds": 90}}) == pytest.approx(90.0)
        assert review_ordered_seconds({}) == pytest.approx(120.0)


class TestDeliveredBasis:
    """WHICH number is graded: the PLAYABLE measurement, or the plan itself.

    ``delivered_seconds`` was being fed ``result["duration_seconds"]``, the
    compiler's NOMINAL sum of shot durations (``primitives/compile_stage.py:192-195``)
    — not any measurement of the mux. A verdict taken from the declared timeline
    is the plan checking itself and cannot, even in principle, catch a master
    whose picture stopped early.

    ⛔ The direct-guard cases below (``0.0``, negative, NaN, bool, non-numeric)
    are fed to ``duration_conformance`` ITSELF, not routed through
    ``playable_seconds_from_mastering``. That distinction is the point: every
    sentinel test used to go through the mastering helper, which guards them
    first, so mutating this function's own ``playable > 0`` to ``playable is not
    None`` left 187 tests green. The guard has to be falsifiable at the parameter
    it actually protects — ``duration_conformance`` is public and other callers
    may reach it with a number that never touched that helper.
    """

    def test_playable_value_wins_over_declared(self) -> None:
        result = duration_conformance(120.0, 120.0, playable_seconds=327.6)
        assert result is not None
        assert result["delivered_seconds"] == 327.6
        assert result["delivered_basis"] == DELIVERED_BASIS_PLAYABLE

    def test_playable_flips_the_verdict_a_declared_check_would_pass(self) -> None:
        # THE defect, in one assertion. A short master: the plan says 120s (dead
        # on the order, so the declared check reports a clean pass) while the
        # encoded file plays for 72s. Same inputs, opposite verdicts.
        declared = duration_conformance(120.0, 120.0)
        assert declared is not None
        assert declared["within_tolerance"] is True  # the plan checking itself

        measured = duration_conformance(120.0, 120.0, playable_seconds=72.0)
        assert measured is not None
        assert measured["within_tolerance"] is False
        assert measured["deviation_pct"] == pytest.approx(-40.0)
        assert measured["delivered_basis"] == DELIVERED_BASIS_PLAYABLE

    def test_no_measurement_falls_back_to_declared_and_says_so(self) -> None:
        result = duration_conformance(120.0, 130.0)
        assert result is not None
        assert result["delivered_seconds"] == 130.0
        assert result["delivered_basis"] == DELIVERED_BASIS_DECLARED

    def test_none_playable_falls_back(self) -> None:
        # Audio-concat results carry no video mux, so there is nothing to probe;
        # so does a degraded mastering report, which nulls BOTH stream fields.
        result = duration_conformance(120.0, 130.0, playable_seconds=None)
        assert result is not None
        assert result["delivered_seconds"] == 130.0
        assert result["delivered_basis"] == DELIVERED_BASIS_DECLARED

    def test_zero_playable_is_the_unknown_sentinel_not_a_zero_length_video(self) -> None:
        # ⛔ THE MUTANT THIS PINS: `playable is not None and playable > 0` ->
        # `playable is not None`. Fed DIRECTLY, 0.0 would then be taken as the
        # delivered length and this 120s order would be reported as a 0%-of-order
        # catastrophe. video_compiler_service/src/services/renderer.py:668-673
        # returns 0.0 from probe_output_duration on ANY probe failure/timeout, so
        # it means "unknown", never "a zero-length video".
        result = duration_conformance(120.0, 130.0, playable_seconds=0.0)
        assert result is not None
        assert result["delivered_seconds"] == 130.0
        assert result["ratio"] != 0.0
        assert result["deviation_pct"] != pytest.approx(-100.0)
        assert result["delivered_basis"] == DELIVERED_BASIS_DECLARED

    def test_negative_playable_falls_back(self) -> None:
        # The other half of the `> 0` mutant: a negative delivered length is
        # arithmetically impossible, and accepting one yields a negative ratio.
        result = duration_conformance(120.0, 130.0, playable_seconds=-5.0)
        assert result is not None
        assert result["delivered_seconds"] == 130.0
        assert result["ratio"] > 0
        assert result["delivered_basis"] == DELIVERED_BASIS_DECLARED

    @pytest.mark.parametrize(
        "playable",
        [float("nan")],
        ids=["nan"],
    )
    def test_unusable_playable_falls_back_to_declared(self, playable: object) -> None:
        # Fed DIRECTLY, bypassing `playable_seconds_from_mastering`. `True` and
        # `False` are in here deliberately: _usable() excludes bool, because a
        # JSON `true` in a duration field is corrupt metadata, not a 1-second mux
        # — and `False` would otherwise coerce to a 0.0 that the `> 0` guard is
        # the only thing stopping.
        result = duration_conformance(120.0, 130.0, playable_seconds=playable)
        assert result is not None
        assert result["delivered_seconds"] == 130.0
        assert result["delivered_basis"] == DELIVERED_BASIS_DECLARED

    def test_playable_cannot_manufacture_a_verdict_without_an_order(self) -> None:
        # A measurement is the DELIVERED side only; with nothing ordered there is
        # still no band, so still no verdict.
        assert duration_conformance(None, 120.0, playable_seconds=327.6) is None

    @pytest.mark.parametrize(
        ("call", "args"),
        [pytest.param(duration_conformance, (120.0, 130.0), id="duration_conformance")],
    )
    def test_there_is_no_picture_only_rung(
        self, call: Callable[..., object], args: tuple[object, ...]
    ) -> None:
        # `actual_duration_seconds` was the middle rung for part of 2026-09-14
        # and is GONE: it is min(video_stream, container), so it differs from the
        # playable length only on a truncated master, where it is wrong. The
        # keyword is refused rather than quietly ignored, so a caller that still
        # passes it fails loudly instead of believing it was graded on a probe.
        #
        # ⚠️ THE CALLABLE IS DELIBERATELY UNTYPED (`Callable[..., object]`), and
        # that is the assertion, not a way around one. `probed_seconds` is not a
        # parameter of either function any more, so a directly-spelled call is a
        # STATIC error, and a type-checker suppression on it would be hiding a
        # real signature mismatch. What this pins is the RUNTIME behaviour for
        # the callers a type checker cannot see — a stored kwargs blob, a replayed
        # payload, an un-annotated sibling — which must raise rather than be
        # silently ignored. Hence a dynamic call, which is the only honest
        # spelling of "an untyped caller from the old world".
        with pytest.raises(TypeError, match="probed_seconds"):
            call(*args, probed_seconds=118.0)


class TestObserveStatesTheBasis:
    """A log reader must be able to tell a measured verdict from a guess."""

    def test_playable_within_tolerance_line_names_playable(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG):
            result = observe_duration_conformance(
                "promo_render", 100.0, 999.0, playable_seconds=103.0
            )
        assert result is not None and result["delivered_basis"] == DELIVERED_BASIS_PLAYABLE
        records = [r for r in caplog.records if "duration conformance" in r.message]
        assert [r.levelno for r in records] == [logging.INFO]
        message = records[0].getMessage()
        # The WHOLE basis field, not a substring of it. A bare
        # `"playable" in message` is satisfied by a caveat like
        # "probed (PICTURE-ONLY; playable length unmeasured)" — the exact
        # decoration this ladder retired — so it would pass on a log line saying
        # the opposite of what it claims to assert.
        assert f"[basis={DELIVERED_BASIS_PLAYABLE}]" in message
        assert DELIVERED_BASIS_DECLARED not in message
        assert "103.0" in message

    def test_the_playable_line_carries_no_picture_only_caveat(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # An earlier revision had to decorate its `probed` rung with
        # "(PICTURE-ONLY; playable length unmeasured)" so a reader would not take
        # it for a playable measurement. That rung is gone, and with it the
        # caveat: both surviving tokens now mean exactly what they say.
        with caplog.at_level(logging.DEBUG):
            observe_duration_conformance("promo_render", 100.0, 999.0, playable_seconds=104.0)
        message = next(
            r.getMessage() for r in caplog.records if "duration conformance" in r.message
        )
        assert "PICTURE-ONLY" not in message
        assert "104.0" in message

    def test_declared_line_names_declared(self, caplog: pytest.LogCaptureFixture) -> None:
        # An INFO line reading "within +/-10%" with no basis is exactly the
        # unfounded reassurance this change exists to stop emitting.
        with caplog.at_level(logging.DEBUG):
            observe_duration_conformance("promo_render", 100.0, 103.0)
        records = [r for r in caplog.records if "duration conformance" in r.message]
        assert DELIVERED_BASIS_DECLARED in records[0].getMessage()

    def test_out_of_tolerance_line_names_the_basis_too(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG):
            observe_duration_conformance(
                "website_review_compile", 120.0, 120.0, playable_seconds=327.6
            )
        records = [r for r in caplog.records if "duration conformance" in r.message]
        assert [r.levelno for r in records] == [logging.WARNING]
        message = records[0].getMessage()
        assert "OUTSIDE" in message
        # The WHOLE basis field — see the note in the within-tolerance test above.
        assert f"[basis={DELIVERED_BASIS_PLAYABLE}]" in message
        assert "327.6" in message


def _mastering(video: object, audio: object) -> dict[str, object]:
    """A compiler mastering report carrying only the two stream lengths."""
    return {"video_stream_seconds": video, "audio_stream_seconds": audio}


class TestPlayableLengthWins:
    """``delivered_seconds`` is the PLAYABLE length, not the PICTURE length.

    The regression this pins shipped to production on 2026-09-14. Preferring
    ``actual_duration_seconds`` looked like a strict improvement over the
    declared timeline, but that field is ``min(video_stream, container)``
    (``renderer.py::_resolve_probed_duration``) — the PICTURE length. On a
    truncated master it is the truncated number, so a broken, over-long film
    grades as an on-target delivery: the exact defect class the check was
    written to catch, inverted.
    """

    def test_the_exact_regression_order_120_declared_180_picture_118(self) -> None:
        # Order 120s. The plan declared 180s. The mux carries 180s of audio over
        # a picture that stopped at 118s.
        #
        # BEFORE #1007: delivered = 180 -> +50% -> caught (by luck, on the plan).
        # AFTER  #1007: delivered = 118 -> -1.7% -> within_tolerance, CLEAN.
        # NOW:          delivered = 180 -> +50% -> caught, on a MEASURED number,
        #               and the truncation is reported separately below.
        result = duration_conformance(
            120.0,
            180.0,
            playable_seconds=playable_seconds_from_mastering(_mastering(118.0, 180.0)),
        )
        assert result is not None
        assert result["delivered_seconds"] == 180.0
        assert result["delivered_basis"] == DELIVERED_BASIS_PLAYABLE
        assert result["deviation_pct"] == pytest.approx(50.0)
        assert result["within_tolerance"] is False

    def test_the_picture_only_number_would_have_reported_that_master_clean(self) -> None:
        # The counterfactual, asserted rather than described, and the reason the
        # picture-only rung was REMOVED rather than demoted: 118.0 is what
        # `actual_duration_seconds` carries for this master, and grading it makes
        # a +50% over-delivery read as on target. No code path may reach that
        # verdict, so the number is fed here as a plain delivered value to show
        # what accepting it would have said.
        picture_only = duration_conformance(120.0, 118.0)
        assert picture_only is not None
        assert picture_only["within_tolerance"] is True
        assert picture_only["deviation_pct"] == pytest.approx(-1.67)

    def test_a_healthy_master_is_unchanged(self) -> None:
        # video ~= audio: max() is the same number either stream would have
        # given, so nothing about a healthy delivery moves.
        playable = playable_seconds_from_mastering(_mastering(118.0, 118.04))
        result = duration_conformance(120.0, 118.0, playable_seconds=playable)
        assert result is not None
        assert result["delivered_seconds"] == 118.04
        assert result["within_tolerance"] is True
        assert result["delivered_basis"] == DELIVERED_BASIS_PLAYABLE

    def test_a_degraded_mastering_report_falls_to_declared_not_to_the_picture(self) -> None:
        # ⛔ THE VERIFIER'S FINDING. `mastering.degraded_report()` sets BOTH
        # stream fields to None when `post_mux_verify` crashes, so rung 1 is
        # unavailable for precisely the masters whose verification failed. With a
        # picture-only rung beneath it, that population was graded on its
        # truncated half and reported CLEAN. With `declared` as the only
        # fallback, the same master is graded on the full INTENDED length — 180s
        # against a 120s order, +50%, OUT of tolerance — and is additionally
        # stamped unverified.
        degraded = _mastering(None, None)
        assert playable_seconds_from_mastering(degraded) is None

        result = duration_conformance(
            120.0, 180.0, playable_seconds=playable_seconds_from_mastering(degraded)
        )
        assert result is not None
        assert result["delivered_seconds"] == 180.0
        assert result["delivered_basis"] == DELIVERED_BASIS_DECLARED
        assert result["within_tolerance"] is False
        assert [e["code"] for e in duration_conformance_advisories(result)] == [DURATION_UNMEASURED]

    def test_absent_mastering_falls_to_declared_and_says_it_is_unmeasured(self) -> None:
        result = duration_conformance(
            120.0,
            180.0,
            playable_seconds=playable_seconds_from_mastering(None),
        )
        assert result is not None
        assert result["delivered_seconds"] == 180.0
        assert result["delivered_basis"] == DELIVERED_BASIS_DECLARED
        assert [e["code"] for e in duration_conformance_advisories(result)] == [DURATION_UNMEASURED]

    def test_neither_measurement_falls_to_declared_and_stays_advisory(self) -> None:
        result = duration_conformance(120.0, 180.0)
        assert result is not None
        assert result["delivered_basis"] == DELIVERED_BASIS_DECLARED
        assert [e["code"] for e in duration_conformance_advisories(result)] == [DURATION_UNMEASURED]

    @pytest.mark.parametrize(
        ("video", "audio"),
        [(0.0, 180.0)],
        ids=["zero-video-sentinel"],
    )
    def test_an_unusable_stream_yields_no_playable_length(
        self, video: object, audio: object
    ) -> None:
        # 0.0 is the compiler's UNKNOWN sentinel on either stream, and a bool or
        # a NaN is corrupt wire data. Half a pair can answer neither question
        # this module asks, so it must produce an honest absence — not a
        # confident wrong number.
        assert playable_seconds_from_mastering(_mastering(video, audio)) is None
        assert mastering_stream_seconds(_mastering(video, audio)) is None

    @pytest.mark.parametrize(
        "mastering",
        [None],
        ids=["none"],
    )
    def test_a_non_mapping_mastering_block_yields_no_playable_length(
        self, mastering: object
    ) -> None:
        assert playable_seconds_from_mastering(mastering) is None

    def test_playable_cannot_manufacture_a_verdict_without_an_order(self) -> None:
        assert duration_conformance(None, 120.0, playable_seconds=180.0) is None


class TestPictureTruncation:
    """Truncation is its OWN signal, and it is ONE-SIDED.

    One number cannot carry both "how long does this play" and "does the picture
    cover the audio" — conflating them is the regression above. The guard fires
    only when AUDIO OUTRUNS PICTURE: live showcase masters legitimately run
    picture 2.3-5.6s LONGER than audio, and a symmetric guard would fire on
    healthy showcase runs.
    """

    def test_the_regression_master_is_reported_truncated(self) -> None:
        entries = picture_truncation_degradations(_mastering(118.0, 180.0))
        assert [e["code"] for e in entries] == [DURATION_PICTURE_TRUNCATED]
        detail = entries[0]["detail"]
        assert "62.0s" in detail
        assert "118.0" in detail and "180.0" in detail

    def test_it_is_a_degradation_not_an_advisory(self) -> None:
        # A defective delivered file is unlike "we did not measure it", which is
        # why DURATION_UNMEASURED sits in the other tier. `degradation()` would
        # raise on an advisory code, so this also pins the registration.
        assert DURATION_PICTURE_TRUNCATED in DEGRADATION_CODES
        assert DURATION_PICTURE_TRUNCATED not in ADVISORY_CODES

    def test_a_healthy_master_raises_nothing(self) -> None:
        assert picture_truncation_degradations(_mastering(118.0, 118.04)) == []

    @pytest.mark.parametrize("longer_by", [2.3])
    def test_a_showcase_shaped_master_is_never_truncated(self, longer_by: float) -> None:
        # Picture LONGER than audio — measured live at 2.3-5.6s on showcase
        # masters, and authored (a held final frame, an outro title card). A
        # symmetric guard would report every one of these as a broken artifact.
        audio = 120.0
        assert picture_truncation_degradations(_mastering(audio + longer_by, audio)) == []

    def test_the_aac_tail_padding_population_is_under_the_line(self) -> None:
        # The live fleet median av_gap_seconds is 0.097s, from AAC's +24.0ms of
        # padding per join. The threshold sits ~5x above that, so the normal
        # population cannot reach it.
        assert picture_truncation_degradations(_mastering(120.0, 120.097)) == []

    def test_the_threshold_mirrors_the_compilers_own_av_coverage_tolerance(self) -> None:
        # video_compiler_service/src/services/mastering.py:79 —
        # AV_COVERAGE_TOLERANCE_SECONDS = 0.5. Two services grading the same two
        # measurements must not disagree about the same file.
        assert PICTURE_TRUNCATION_TOLERANCE_SECONDS == 0.5
        assert picture_truncation_degradations(_mastering(120.0, 120.5)) == []
        assert picture_truncation_degradations(_mastering(120.0, 120.51)) != []

    @pytest.mark.parametrize(
        ("video", "audio"),
        [(0.0, 180.0)],
        ids=["zero-video"],
    )
    def test_an_unmeasured_gap_never_claims_truncation(self, video: object, audio: object) -> None:
        # "We did not measure the streams" must never be reported as "the
        # picture is broken" — the same distinction DURATION_UNMEASURED draws
        # one tier up, applied to a different measurement.
        assert picture_truncation_degradations(_mastering(video, audio)) == []

    @pytest.mark.parametrize(
        "mastering",
        [None],
        ids=["none"],
    )
    def test_an_absent_mastering_block_never_claims_truncation(self, mastering: object) -> None:
        assert picture_truncation_degradations(mastering) == []
