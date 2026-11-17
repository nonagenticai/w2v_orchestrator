"""Unit tests for soft-CC (SRT/VTT) minting — pure functions, no I/O."""

from __future__ import annotations

from src.services.captions import (
    CaptionCue,
    build_caption_cues,
    build_course_caption_cues,
    build_srt,
    build_vtt,
    caption_cues_for_run,
    caption_text_by_line,
    srt_timecode,
    vtt_timecode,
)


class TestTimecode:
    def test_srt_timecode_subsecond(self) -> None:
        assert srt_timecode(0.0) == "00:00:00,000"
        assert srt_timecode(1.5) == "00:00:01,500"
        assert srt_timecode(61.25) == "00:01:01,250"

    def test_srt_timecode_over_an_hour(self) -> None:
        assert srt_timecode(3661.123) == "01:01:01,123"

    def test_vtt_timecode_uses_dot(self) -> None:
        assert vtt_timecode(61.25) == "00:01:01.250"

    def test_negative_clamped_to_zero(self) -> None:
        assert srt_timecode(-5.0) == "00:00:00,000"


class TestBuildSrt:
    def test_structure_and_indexing(self) -> None:
        cues = [
            CaptionCue(0.0, 2.0, "Hello there"),
            CaptionCue(2.0, 5.0, "Second line"),
        ]
        srt = build_srt(cues)
        assert srt == (
            "1\n00:00:00,000 --> 00:00:02,000\nHello there\n"
            "\n"
            "2\n00:00:02,000 --> 00:00:05,000\nSecond line\n"
        )

    def test_blank_text_skipped(self) -> None:
        cues = [CaptionCue(0.0, 1.0, "   "), CaptionCue(1.0, 2.0, "real")]
        srt = build_srt(cues)
        assert "real" in srt
        # only one numbered block, indexed 1
        assert srt.startswith("1\n")
        assert "2\n" not in srt

    def test_empty_cues(self) -> None:
        assert build_srt([]) == ""


class TestBuildVtt:
    def test_header_and_cues(self) -> None:
        vtt = build_vtt([CaptionCue(0.0, 2.0, "hi")])
        assert vtt.startswith("WEBVTT\n")
        assert "00:00:00.000 --> 00:00:02.000\nhi\n" in vtt


class TestBuildCaptionCues:
    def test_orders_by_script_not_clip_order(self) -> None:
        lines = [
            {"id": "a", "text": "first"},
            {"id": "b", "text": "second"},
            {"id": "c", "text": "third"},
        ]
        # clips arrive out of order — timing must still follow script order.
        clips = [
            {"line_id": "c", "seconds": 3.0},
            {"line_id": "a", "seconds": 2.0},
            {"line_id": "b", "seconds": 1.0},
        ]
        cues = build_caption_cues(lines, clips)
        assert [c.text for c in cues] == ["first", "second", "third"]
        assert cues[0] == CaptionCue(0.0, 2.0, "first")
        assert cues[1] == CaptionCue(2.0, 3.0, "second")
        assert cues[2] == CaptionCue(3.0, 6.0, "third")

    def test_line_without_clip_skipped_no_clock_advance(self) -> None:
        lines = [
            {"id": "a", "text": "has audio"},
            {"id": "b", "text": "no audio"},  # no matching clip
            {"id": "c", "text": "after"},
        ]
        clips = [{"line_id": "a", "seconds": 2.0}, {"line_id": "c", "seconds": 1.0}]
        cues = build_caption_cues(lines, clips)
        assert [c.text for c in cues] == ["has audio", "after"]
        # 'after' starts at 2.0 (the gap line didn't advance the clock)
        assert cues[1] == CaptionCue(2.0, 3.0, "after")

    def test_zero_duration_clip_emits_no_cue_but_advances(self) -> None:
        lines = [
            {"id": "a", "text": "zero dur"},
            {"id": "b", "text": "next"},
        ]
        clips = [{"line_id": "a", "seconds": 0.0}, {"line_id": "b", "seconds": 1.0}]
        cues = build_caption_cues(lines, clips)
        assert [c.text for c in cues] == ["next"]

    def test_blank_text_advances_clock(self) -> None:
        lines = [{"id": "a", "text": "  "}, {"id": "b", "text": "real"}]
        clips = [{"line_id": "a", "seconds": 2.0}, {"line_id": "b", "seconds": 1.0}]
        cues = build_caption_cues(lines, clips)
        assert cues == [CaptionCue(2.0, 3.0, "real")]

    def test_custom_keys(self) -> None:
        lines = [{"id": "p1", "text": "para one"}]
        clips = [{"paragraph_id": "p1", "seconds": 2.0}]
        cues = build_caption_cues(lines, clips, clip_join_key="paragraph_id")
        assert cues == [CaptionCue(0.0, 2.0, "para one")]


class TestCaptionCuesForRun:
    def test_promo(self) -> None:
        outputs = {
            "promo_script": {"lines": [{"id": "l1", "text": "promo line"}]},
            "promo_narrate": {"clips": [{"line_id": "l1", "seconds": 2.0}]},
        }
        cues = caption_cues_for_run(outputs)
        assert cues == [CaptionCue(0.0, 2.0, "promo line")]

    def test_lesson_uses_paragraph_join(self) -> None:
        outputs = {
            "lesson_script": {"paragraphs": [{"id": "p1", "text": "lesson para"}]},
            "lesson_narrate": {"clips": [{"paragraph_id": "p1", "seconds": 3.0}]},
        }
        cues = caption_cues_for_run(outputs)
        assert cues == [CaptionCue(0.0, 3.0, "lesson para")]

    def test_website_review(self) -> None:
        outputs = {
            "website_review_script": {"lines": [{"id": "r1", "text": "review line"}]},
            "website_review_record": {"clips": [{"line_id": "r1", "seconds": 1.5}]},
        }
        cues = caption_cues_for_run(outputs)
        assert cues == [CaptionCue(0.0, 1.5, "review line")]

    def test_generic_uses_script_draft_narrate(self) -> None:
        outputs = {
            "script_draft": {
                "lines": [
                    {"id": "g1", "text": "generic first"},
                    {"id": "g2", "text": "generic second"},
                ]
            },
            "narrate": {
                "clips": [
                    {"line_id": "g1", "seconds": 2.5},
                    {"line_id": "g2", "seconds": 1.5},
                ]
            },
        }
        cues = caption_cues_for_run(outputs)
        assert cues == [
            CaptionCue(0.0, 2.5, "generic first"),
            CaptionCue(2.5, 4.0, "generic second"),
        ]

    def test_course_returns_none(self) -> None:
        outputs = {
            "course_stitch": {"course_artifact_url": "http://x/course.mp4", "chapters": []},
        }
        assert caption_cues_for_run(outputs) is None

    def test_no_recognized_pair_returns_none(self) -> None:
        # script present but no narrate clips → not a usable pair
        assert caption_cues_for_run({"promo_script": {"lines": [{"id": "a", "text": "x"}]}}) is None

    def test_empty_outputs_returns_none(self) -> None:
        assert caption_cues_for_run({}) is None


class TestBuildCourseCaptionCues:
    def _lesson(self, *texts_seconds: tuple[str, float]) -> tuple[list[dict], list[dict]]:
        """Build (paragraphs, clips) for a lesson from (text, seconds) pairs."""
        paragraphs = [{"id": f"p{i}", "text": t} for i, (t, _) in enumerate(texts_seconds)]
        clips = [{"paragraph_id": f"p{i}", "seconds": s} for i, (_, s) in enumerate(texts_seconds)]
        return paragraphs, clips

    def test_concatenates_and_offsets_by_chapter_start(self) -> None:
        l1p, l1c = self._lesson(("l1 first", 2.0), ("l1 second", 3.0))
        l2p, l2c = self._lesson(("l2 first", 1.0), ("l2 second", 4.0))
        # lesson 2 starts at 30.0 on the course timeline (its stitched offset)
        cues = build_course_caption_cues([(0.0, l1p, l1c), (30.0, l2p, l2c)])
        assert cues == [
            CaptionCue(0.0, 2.0, "l1 first"),
            CaptionCue(2.0, 5.0, "l1 second"),
            CaptionCue(30.0, 31.0, "l2 first"),
            CaptionCue(31.0, 35.0, "l2 second"),
        ]

    def test_empty_tracks(self) -> None:
        assert build_course_caption_cues([]) == []

    def test_unjoinable_lesson_contributes_nothing_and_does_not_disturb_later(self) -> None:
        # lesson 1's clips don't join its paragraphs → no cues, no clock effect.
        bad_paragraphs = [{"id": "p0", "text": "orphan"}]
        bad_clips = [{"paragraph_id": "nope", "seconds": 9.0}]
        l2p, l2c = self._lesson(("good", 2.0))
        cues = build_course_caption_cues([(0.0, bad_paragraphs, bad_clips), (10.0, l2p, l2c)])
        assert cues == [CaptionCue(10.0, 12.0, "good")]


class TestCaptionTextByLine:
    """The {line_id: text} map the burned-in compile path stamps onto shots.

    Same per-type dispatch as caption_cues_for_run, but no clip join (timing
    comes from the narration clip already on each shot)."""

    def test_promo(self) -> None:
        outputs = {
            "promo_script": {"lines": [{"id": "l1", "text": "A"}, {"id": "l2", "text": "B"}]}
        }
        assert caption_text_by_line(outputs) == {"l1": "A", "l2": "B"}

    def test_lesson_keys_by_paragraph_id(self) -> None:
        # Keyed by paragraph id — the same value lesson_compile uses as line_id.
        outputs = {"lesson_script": {"paragraphs": [{"id": "p1", "text": "Para one."}]}}
        assert caption_text_by_line(outputs) == {"p1": "Para one."}

    def test_website_review(self) -> None:
        outputs = {"website_review_script": {"lines": [{"id": "r1", "text": "Review."}]}}
        assert caption_text_by_line(outputs) == {"r1": "Review."}

    def test_generic_uses_script_draft(self) -> None:
        outputs = {"script_draft": {"lines": [{"id": "g1", "text": "Generic."}]}}
        assert caption_text_by_line(outputs) == {"g1": "Generic."}

    def test_blank_text_kept_as_empty_string(self) -> None:
        # Kept (assemble_shots' truthy gate drops the empty caption at the shot).
        outputs = {"promo_script": {"lines": [{"id": "l1", "text": ""}, {"id": "l2"}]}}
        assert caption_text_by_line(outputs) == {"l1": "", "l2": ""}

    def test_line_without_id_skipped(self) -> None:
        outputs = {"promo_script": {"lines": [{"text": "no id"}, {"id": "l2", "text": "kept"}]}}
        assert caption_text_by_line(outputs) == {"l2": "kept"}

    def test_course_and_audiobook_return_none(self) -> None:
        assert caption_text_by_line({"course_stitch": {"chapters": []}}) is None

    def test_empty_outputs_returns_none(self) -> None:
        assert caption_text_by_line({}) is None
        assert caption_text_by_line({"promo_script": {"lines": []}}) is None


class TestAuditFamilyCaptions:
    """seo_audit / cro_audit narrated videos must produce captions.

    Regression guard for a defect proven live on 2026-08-10: a freshly completed
    narrated SEO audit rendered a "Download captions (.srt)" button that returned
    ``404 no captions for this run``. The audit sub-chain reuses ``audit_video``'s
    shared models, so the shapes were already standard — the stage-name pair was
    simply never listed in ``_CAPTION_SOURCES``, and both the soft-CC track and the
    burned-in path read that one table.
    """

    def test_seo_audit_cues(self) -> None:
        outputs = {
            "seo_audit_script": {
                "lines": [{"id": "s1", "text": "First."}, {"id": "s2", "text": "Second."}]
            },
            "seo_audit_narrate": {
                "clips": [
                    {"line_id": "s1", "seconds": 2.0},
                    {"line_id": "s2", "seconds": 3.0},
                ]
            },
        }
        cues = caption_cues_for_run(outputs)
        assert cues is not None
        assert [c.text for c in cues] == ["First.", "Second."]
        assert cues[0].start == 0.0
        assert cues[1].start == 2.0

    def test_cro_audit_cues(self) -> None:
        outputs = {
            "cro_audit_script": {"lines": [{"id": "c1", "text": "Conversion."}]},
            "cro_audit_narrate": {"clips": [{"line_id": "c1", "seconds": 1.5}]},
        }
        cues = caption_cues_for_run(outputs)
        assert cues is not None
        assert [c.text for c in cues] == ["Conversion."]

    def test_audit_burned_in_text_map(self) -> None:
        """The burned-in path reads the same table, so it was equally blind."""
        assert caption_text_by_line(
            {"seo_audit_script": {"lines": [{"id": "s1", "text": "A"}]}}
        ) == {"s1": "A"}
        assert caption_text_by_line(
            {"cro_audit_script": {"lines": [{"id": "c1", "text": "B"}]}}
        ) == {"c1": "B"}


def test_every_narrate_stage_has_a_caption_source() -> None:
    """No narrated stage may exist without a caption source — the silent-gap guard.

    The audit family shipped a narrated video whose captions 404'd for as long as it
    existed, because adding a ``*_narrate`` stage and adding a ``_CAPTION_SOURCES``
    row are two unrelated edits with nothing tying them together. This ties them:
    every ``*_narrate`` stage reachable from a registered blueprint must appear as
    the clip stage of some caption source, or be listed below as a deliberate
    exemption WITH its reason.

    The exemptions are deliberate, not a backlog: an audiobook has no on-screen
    video to caption, and a course is captioned by ``course_caption_cues`` — a
    DB-coupled path that stitches its child lessons' cues — rather than by this
    table.
    """
    import src.services.blueprints  # noqa: F401  # why: import registers every blueprint
    from src.services.blueprints.registry import get_registry
    from src.services.captions import _CAPTION_SOURCES

    clip_stages = {row[4] for row in _CAPTION_SOURCES}
    exempt = {
        "audiobook_narrate",  # audio-only deliverable: nothing on screen to caption
        "course_narrate",  # captioned via course_caption_cues (stitches child lessons)
    }
    narrate_stages = {
        spec.name
        for bp in get_registry().all()
        for spec in bp.stages
        if spec.name.endswith("_narrate")
    }
    assert narrate_stages, "no *_narrate stages found — the guard would pass vacuously"
    uncovered = sorted(narrate_stages - clip_stages - exempt)
    assert not uncovered, (
        f"narrated stage(s) with no caption source: {uncovered}. Add a row to "
        "_CAPTION_SOURCES (script stage, lines key, id key, text key, clip stage, "
        "clips key, join key), or add an exemption above WITH the reason it has no "
        "captions. A narrated deliverable whose captions 404 is a shipped defect."
    )


# ── captions-only (silent) demo_video: the SRT sidecar has no narration clips ──

_SILENT_GEN = {
    "lines": [
        {"id": "clip-1", "text": "Sophie files a ticket.", "seconds": 4.0},
        {"id": "clip-2", "text": "The manager approves.", "seconds": 0.0},
    ]
}


def test_captions_only_srt_is_timed_by_the_compiled_shot_timeline() -> None:
    from src.services.captions import build_srt, caption_cues_for_run

    outputs = {
        "demo_video_gen": _SILENT_GEN,
        "demo_video_narrate": {"clips": []},
        "demo_video_compile": {
            "shot_timeline": [
                {"line_id": "__title_card", "start": 0.0, "end": 2.0},
                {"line_id": "clip-1", "start": 2.0, "end": 6.0},
                {"line_id": "clip-2", "start": 6.0, "end": 8.5},
            ]
        },
    }
    cues = caption_cues_for_run(outputs)
    assert cues is not None
    assert [(c.start, c.end, c.text) for c in cues] == [
        (2.0, 6.0, "Sophie files a ticket."),
        (6.0, 8.5, "The manager approves."),
    ]
    srt = build_srt(cues)
    assert "Sophie files a ticket." in srt and "The manager approves." in srt


def test_captions_only_srt_without_a_timeline_accumulates_line_seconds() -> None:
    from src.services.captions import caption_cues_for_run
    from src.services.stages.shared.captions_only import reading_seconds

    outputs = {"demo_video_gen": _SILENT_GEN, "demo_video_narrate": {"clips": []}}
    cues = caption_cues_for_run(outputs)
    assert cues is not None
    read = reading_seconds("The manager approves.")
    assert [(c.start, c.end) for c in cues] == [(0.0, 4.0), (4.0, 4.0 + read)]


def test_a_run_with_no_demo_narrate_gets_no_captions_only_cues() -> None:
    from src.services.captions import caption_cues_for_run

    assert caption_cues_for_run({"demo_video_gen": _SILENT_GEN}) is None
