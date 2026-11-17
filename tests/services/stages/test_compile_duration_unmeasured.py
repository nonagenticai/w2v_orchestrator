"""An UNPROBED compile duration is not a verified one (fail-open on a measurement gap).

``_compile_duration_qa_notes`` used to return ``[]`` — byte-identical to "checked and
fine" — whenever ``actual_duration_seconds`` was absent, non-numeric or 0.0. But 0.0 is
the compiler's SENTINEL for "ffprobe failed / could not measure"
(``video_compiler_service/src/services/renderer.py:369-381``), and an empty note list
makes ``api/routes/runs/_shared.py`` report the run CLEAN. So a truncated master shipped
green on exactly the runs where the truncation check was the thing that broke.

These tests pin the three outcomes apart: measured-and-fine (silent), measured-and-short
(the existing mismatch note, unchanged), and NOT MEASURED (a distinct unmeasured note ⇒
the run reports ``degraded``).
"""

from __future__ import annotations

import pytest

from src.services.stages.primitives.compile_stage import (
    playable_duration_seconds,
)


class TestPlayableDurationSeconds:
    """The surviving helper: the PLAYABLE length, off the mastering report.

    ``probed_duration_seconds`` stood here until the picture-only rung was
    removed; with all five call sites gone it had no callers, so it was deleted
    rather than left as dead code. Its sentinel semantics are not lost — they live
    on in ``_usable``, which this helper reaches through
    ``playable_seconds_from_mastering``, and every ``None`` below is load-bearing
    for the same reason: the caller uses it to decide that NOTHING measured this
    master, so the verdict falls to ``declared`` and is stamped unverified rather
    than computed against a sentinel.
    """

    def test_the_playable_length_is_the_longer_stream(self) -> None:
        result = {
            "duration_seconds": 180.0,
            "mastering": {"video_stream_seconds": 118.0, "audio_stream_seconds": 180.0},
        }
        assert playable_duration_seconds(result) == 180.0

    def test_it_is_never_the_picture_only_field(self) -> None:
        # `actual_duration_seconds` is present, on-target and IGNORED. If this
        # helper ever reads it again, a truncated master goes back to grading
        # clean on 118.0.
        result = {
            "duration_seconds": 180.0,
            "actual_duration_seconds": 118.0,
            "mastering": {"video_stream_seconds": 118.0, "audio_stream_seconds": 180.0},
        }
        assert playable_duration_seconds(result) == 180.0

    def test_a_picture_only_result_yields_no_measurement(self) -> None:
        # The shape a degraded `post_mux_verify` produces: a picture probe and no
        # usable streams. There is no rung for it — the answer is an honest None.
        assert playable_duration_seconds({"actual_duration_seconds": 118.0}) is None

    def test_a_degraded_mastering_report_yields_none(self) -> None:
        # `mastering.degraded_report()` nulls BOTH stream fields when
        # `post_mux_verify` crashes.
        assert (
            playable_duration_seconds(
                {"mastering": {"video_stream_seconds": None, "audio_stream_seconds": None}}
            )
            is None
        )

    def test_absent_mastering_is_none(self) -> None:
        assert playable_duration_seconds({"duration_seconds": 263.5}) is None

    @pytest.mark.parametrize(
        ("video", "audio"),
        [pytest.param(0.0, 180.0, id="zero_video_sentinel")],
    )
    def test_an_unusable_stream_yields_none(self, video: object, audio: object) -> None:
        # 0.0 is renderer.py's "probe failed" sentinel, never a zero-length video;
        # `True` is corrupt wire data, not a 1-second master. Half a pair can
        # answer neither question the module asks, so it produces an honest
        # absence rather than a confident wrong number.
        result = {
            "mastering": {"video_stream_seconds": video, "audio_stream_seconds": audio},
        }
        assert playable_duration_seconds(result) is None

    def test_numeric_strings_are_coerced_to_float(self) -> None:
        # The wire is JSON via an HTTP client; a stringified number is a real shape.
        playable = playable_duration_seconds(
            {"mastering": {"video_stream_seconds": "118.0", "audio_stream_seconds": 180}}
        )
        assert playable == 180.0
        assert isinstance(playable, float)
