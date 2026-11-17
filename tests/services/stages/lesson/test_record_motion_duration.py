"""lesson_record must size each motion clip to its card's narration, not to the
template's hardcoded RENDER_DURATION_MS.

Live failure 2026-09-10 13:35Z: ``lesson_compile`` died on the video-compiler's
``av_coverage`` guard because the picture was ~7.5s shorter than the audio. Cause:
``_render_motion_slides`` called ``render_deck_motion_clips`` with no
``duration_ms_by_index``, so every clip fell back to the ``feature_spotlight``
template default (``motion_plan.py``: ``1500 + n*420 + 1200`` ms ≈ 4s) — a constant
with no relationship to how long the card is narrated.

These tests assert the PROPERTY (clip length tracks its card's narration length),
not the mechanism, and they are deliberately built so that reverting the single
line that passes ``duration_ms_by_index`` makes them fail.
"""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.lesson.record import (
    LessonRecordStage,
    estimated_narration_seconds_by_slide,
)
from src.services.stages.shared.motion_slides import DeckMotionResult

# The measured rate the probe is stubbed to return. 120 wpm => 2 words/second, so
# a card's expected seconds is simply words / 2 — arithmetic a reader can check.
_WPM = 120.0


class _FakeStorage:
    async def initialize(self) -> None:
        pass

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        return f"artifacts/{job_id}/{stage_name}/{filename}"

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        return f"https://minio.test/{path}"


@pytest.fixture
def fake_storage():
    tts_storage.set_storage_for_tests(_FakeStorage())
    yield
    tts_storage.set_storage_for_tests(None)


def _words(n: int) -> str:
    return " ".join(["word"] * n)


def _ctx() -> RunContext:
    """Two slides. s1's longest paragraph is 24 words (12s at 120 wpm); s2's is
    10 words (5s). Both sit inside slide_stage._motion_durations' [4.5s, 12s]
    clamp, and the two differ, so a single shared constant cannot satisfy both."""
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "deliverable": {"title": "Widgets 101", "format": "16x9"},
            "brand": {},
            "resolution": "1080p",
            "motion_graphics": {"enabled": True},
            "voice": {"voice_id": "af_heart", "provider": "kokoro"},
        },
        prior_run_outputs={
            "lesson_script": {
                "paragraphs": [
                    # s1: the SHORT paragraph comes first on purpose — the clip is
                    # laid over EVERY paragraph line of its slide by lesson_compile,
                    # so the card must be sized by its LONGEST paragraph, not its
                    # first (and not their sum, which would be 34 words / 17s).
                    {"id": "p1", "text": _words(10)},
                    {"id": "p2", "text": _words(24)},
                    {"id": "p3", "text": _words(10)},
                ],
            },
            "lesson_slides": {
                "slides": [
                    {
                        "id": "s1",
                        "paragraph_ids": ["p1", "p2"],
                        "title": "T1",
                        "bullets": ["a", "b"],
                    },
                    {"id": "s2", "paragraph_ids": ["p3"], "title": "T2", "bullets": ["c"]},
                ]
            },
        },
    )


def _paragraph_ids_by_slide() -> dict[str, list[str]]:
    """The first of the two primitives ``LessonRecordStage.handle`` extracts from the
    raw ``lesson_slides`` output before calling the (pure) estimator."""
    return {
        s["id"]: list(s["paragraph_ids"])
        for s in _ctx().prior_run_outputs["lesson_slides"]["slides"]
    }


def _words_by_paragraph() -> dict[str, int]:
    """The second: word counts pulled out of the raw ``lesson_script`` output."""
    return {
        p["id"]: len(p["text"].split())
        for p in _ctx().prior_run_outputs["lesson_script"]["paragraphs"]
    }


def _slide_client() -> AsyncMock:
    fake = AsyncMock()
    fake.resolve_template_id.return_value = "tpl-feature"
    fake.start_deck.return_value = "deck-1"
    fake.wait_for_deck.return_value = {"status": "completed", "slide_count": 2}
    fake.get_slide_png.side_effect = [b"\x89PNG-0", b"\x89PNG-1"]
    return fake


def _video_fake() -> AsyncMock:
    v = AsyncMock()
    v.record_animation.return_value = "anim-1"
    v.wait_for_content_recording.return_value = {"duration_seconds": 4.0}
    v.get_recording_video.return_value = (b"\x00mp4", "video/mp4", None)
    return v


async def _run_and_capture_durations(ctx: RunContext | None = None) -> dict[int, int] | None:
    """Run the stage with the speech-rate probe pinned, returning the
    ``duration_ms_by_index`` lesson_record handed to the motion renderer.

    This drives the REAL ``handle()``, so it also exercises the stage's own
    extraction of paragraph ids / word counts out of the untyped stage outputs.
    """
    stage = LessonRecordStage(client=_slide_client(), video_client=_video_fake())
    mock_render = AsyncMock(
        return_value=DeckMotionResult(
            urls={0: "https://minio.test/clip-0.mp4", 1: "https://minio.test/clip-1.mp4"},
            budget_exceeded=False,
            failed_indices=(),
        )
    )
    with (
        patch("src.services.stages.lesson.record.render_deck_motion_clips", mock_render),
        patch(
            "src.services.stages.lesson.record.resolve_natural_wpm",
            AsyncMock(return_value=_WPM),
        ),
    ):
        await stage.run({}, ctx=ctx if ctx is not None else _ctx())
    return mock_render.call_args.kwargs.get("duration_ms_by_index")


@pytest.mark.asyncio
async def test_motion_clip_length_follows_its_cards_narration(fake_storage) -> None:
    """Each clip is rendered to its own card's narration length.

    The assertion is on the DIFFERENCE between the two cards: card 0 is narrated
    24 words (12.0s) and card 1 is narrated 10 words (5.0s), so no single value —
    template constant or otherwise — can satisfy both.
    """
    durations = await _run_and_capture_durations()

    assert durations is not None, (
        "lesson_record passed no duration_ms_by_index, so every clip falls back to "
        "the template's RENDER_DURATION_MS constant — the av_coverage bug"
    )
    # _motion_durations adds a 1s tail then clamps to [4.5s, 12s].
    assert durations[0] == 12_000  # 24 words / 120wpm = 12.0s (+1s tail, clamped to 12)
    assert durations[1] == 6_000  # 10 words / 120wpm = 5.0s + 1s tail
    # The property that actually matters: the longer-narrated card gets the longer
    # clip. A hardcoded constant makes these equal.
    assert durations[0] > durations[1]


@pytest.mark.asyncio
async def test_motion_clip_covers_the_narration_it_is_laid_over(fake_storage) -> None:
    """The av_coverage invariant itself: for every card, clip length >= the
    narration of the longest paragraph the clip is laid over.

    This is the assertion that would have caught the live failure — audio outran
    picture by ~7.5s because the ~4s template constant was shorter than the ~11.5s
    narrated shot.
    """
    durations = await _run_and_capture_durations()
    assert durations is not None

    expected_seconds = estimated_narration_seconds_by_slide(
        _paragraph_ids_by_slide(), _words_by_paragraph(), _WPM
    )
    for index, slide_id in enumerate(["s1", "s2"]):
        assert durations[index] >= expected_seconds[slide_id] * 1000, (
            f"clip {index} ({durations[index]}ms) is shorter than the narration it "
            f"must cover ({expected_seconds[slide_id]:.1f}s) — av_coverage would fail"
        )


def test_card_seconds_use_the_longest_paragraph_not_the_first_or_the_sum() -> None:
    """lesson_compile lays a slide's single clip over EACH of its paragraph lines,
    each held for its own audio — so the card is sized by its longest paragraph."""
    seconds = estimated_narration_seconds_by_slide(
        _paragraph_ids_by_slide(), _words_by_paragraph(), _WPM
    )
    assert seconds["s1"] == 12.0  # 24 words, not the 10-word first paragraph
    assert seconds["s1"] != 17.0  # and not the 34-word sum
    assert seconds["s2"] == 5.0


def test_card_seconds_scale_with_the_measured_rate() -> None:
    """The estimate is driven by the MEASURED wpm, not a constant: halve the rate
    and every card doubles."""
    by_slide, words = _paragraph_ids_by_slide(), _words_by_paragraph()

    fast = estimated_narration_seconds_by_slide(by_slide, words, _WPM)
    slow = estimated_narration_seconds_by_slide(by_slide, words, _WPM / 2)

    assert slow["s1"] == pytest.approx(fast["s1"] * 2)
    assert slow["s2"] == pytest.approx(fast["s2"] * 2)


def test_uncountable_narration_leaves_the_card_unsized() -> None:
    """A slide whose paragraphs are absent/empty is omitted, which is the
    'unknown duration' signal that keeps the template default for it — a missing
    script must not size every card to zero."""
    by_slide, words = _paragraph_ids_by_slide(), _words_by_paragraph()

    # No word counts at all (no script, or an empty one): every card unsized.
    assert estimated_narration_seconds_by_slide(by_slide, {}, _WPM) == {}
    # A slide whose paragraphs are all zero-word is unsized too — NOT mapped to 0.0.
    assert estimated_narration_seconds_by_slide(by_slide, dict.fromkeys(words, 0), _WPM) == {}
    # A non-positive rate is a broken measurement, not a reason to emit garbage.
    assert estimated_narration_seconds_by_slide(by_slide, words, 0.0) == {}
    # Partial knowledge stays partial: s2's only paragraph is uncounted, so s2 is
    # absent (template default) while s1 is still sized from what IS countable.
    partial = estimated_narration_seconds_by_slide(by_slide, {"p1": 10, "p2": 24}, _WPM)
    assert partial == {"s1": 12.0}


@pytest.mark.asyncio
async def test_stage_joins_slide_paragraph_ids_to_script_word_counts(fake_storage) -> None:
    """The extraction the stage now owns (raw ``lesson_slides`` + ``lesson_script``
    -> the two primitive mappings) must JOIN on paragraph id.

    Point s2 at a paragraph the script never emitted: s2 must come back unsized
    (absent => template default) while s1 is still sized from its real 24 words. An
    extraction that ignored the ids — e.g. zipping slides to paragraphs positionally,
    or counting every word in the script — would size s2 anyway.
    """
    ctx = _ctx()
    ctx.prior_run_outputs["lesson_slides"]["slides"][1]["paragraph_ids"] = ["p-missing"]

    durations = await _run_and_capture_durations(ctx)

    assert durations is not None
    assert durations[0] == 12_000  # s1 unchanged: 24 words -> 12.0s
    assert 1 not in durations, (
        f"s2 references an unknown paragraph, so its length is UNKNOWN and it must be "
        f"absent from duration_ms_by_index; got {durations!r}"
    )
