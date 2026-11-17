"""voice_director — per-line performance directives over the demo_video spine."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse
from w2v_common.stages.errors import StageRetriesExhaustedError

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.demo_video.voice_director import VoiceDirectorStage


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _ctx(*, direction: Any = "Energetic and upbeat.") -> RunContext:
    voice: dict[str, Any] = {}
    if direction is not None:
        voice["direction"] = direction
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": voice} if voice else {},
        prior_run_outputs={
            "demo_video_gen": {
                "lines": [
                    {"id": "b1", "text": "Meet our new dashboard.", "seconds": 0.0},
                    {"id": "b2", "text": "Eleven rooms detected.", "seconds": 0.0},
                ]
            },
        },
    )


async def test_skipped_when_no_direction() -> None:
    with pytest.raises(StageSkippedError):
        await VoiceDirectorStage(client=AsyncMock()).run({}, ctx=_ctx(direction=None))


async def test_skipped_when_direction_empty() -> None:
    with pytest.raises(StageSkippedError):
        await VoiceDirectorStage(client=AsyncMock()).run({}, ctx=_ctx(direction=""))


# 0.1 clamped to min


async def test_provider_tier_forced_free() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"directives": [{"line_id": "b1", "provider_tier": "expressive"}]}
    )
    out = await VoiceDirectorStage(client=fake).run({}, ctx=_ctx())
    assert all(d.provider_tier == "free" for d in out.directives)


async def test_emphasis_count_capped_even_when_all_indices_are_valid() -> None:
    """`_MAX_EMPHASIS_WORDS_PER_LINE` caps the COUNT, not just the range.

    "Our team built this quickly today" has 6 in-bounds words; the model asking
    for four of them must not emphasise more than two — every emphasised word
    becomes its own kokoro utterance at a slower request speed, so an unbounded
    list is a mid-sentence rate drop for each extra word.
    """
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"directives": [{"line_id": "b1", "emphasis_word_indices": [0, 1, 2, 3]}]}
    )
    ctx = _ctx()
    ctx.prior_run_outputs["demo_video_gen"]["lines"][0]["text"] = (
        "Our team built this quickly today"
    )
    out = await VoiceDirectorStage(client=fake).run({}, ctx=ctx)
    by_id = {d.line_id: d for d in out.directives}
    assert len(by_id["b1"].emphasis_word_indices) == 2
    assert by_id["b1"].emphasis_word_indices == [0, 1]


async def test_pause_scaled_when_it_would_eat_the_beat() -> None:
    """A 2000ms pause is legal in isolation but not on a 4s beat: it must be
    scaled down to 10% of the beat and the clamp recorded as a note, never
    silently absorbed."""
    fake = AsyncMock()
    fake.message.return_value = _resp({"directives": [{"line_id": "b1", "pause_before_ms": 2000}]})
    ctx = _ctx()
    ctx.prior_run_outputs["demo_video_gen"]["lines"][0]["seconds"] = 4.0
    out = await VoiceDirectorStage(client=fake).run({}, ctx=ctx)
    by_id = {d.line_id: d for d in out.directives}
    assert by_id["b1"].pause_before_ms == 400  # 10% of 4.0s
    assert any("pause_before_ms" in note and "b1" in note for note in out.notes)


async def test_coverage_fill_for_omitted_line() -> None:
    fake = AsyncMock()
    # Only b1 returned — b2 must be coverage-filled with a neutral default.
    fake.message.return_value = _resp({"directives": [{"line_id": "b1", "delivery_note": "warm"}]})
    out = await VoiceDirectorStage(client=fake).run({}, ctx=_ctx())
    assert [d.line_id for d in out.directives] == ["b1", "b2"]
    b2 = out.directives[1]
    assert b2.provider_tier == "free"
    assert b2.rate_hint == 1.0
    assert b2.delivery_note == ""
    assert b2.pause_before_ms == 0
    assert b2.emphasis_word_indices == []


async def test_no_redrive_feedback_message_is_inert() -> None:
    """Without redrive_feedback the built message carries no critic block — the
    normal DAG run is byte-identical to today (feature inert by default)."""
    fake = AsyncMock()
    fake.message.return_value = _resp({"directives": [{"line_id": "b1"}]})
    await VoiceDirectorStage(client=fake).run({}, ctx=_ctx())
    message = fake.message.call_args.kwargs["message"]
    assert "CRITIC FEEDBACK" not in message

    # Explicit None threads identically to omitting the field entirely.
    fake_none = AsyncMock()
    fake_none.message.return_value = _resp({"directives": [{"line_id": "b1"}]})
    await VoiceDirectorStage(client=fake_none).run({"redrive_feedback": None}, ctx=_ctx())
    assert fake_none.message.call_args.kwargs["message"] == message


async def test_redrive_feedback_appended_to_message() -> None:
    """A non-empty redrive_feedback appends the critic block verbatim so a holistic
    critic stage can re-drive the director from the seams it found."""
    fake = AsyncMock()
    fake.message.return_value = _resp({"directives": [{"line_id": "b1"}]})
    await VoiceDirectorStage(client=fake).run(
        {"redrive_feedback": "continuity break at clip-3"}, ctx=_ctx()
    )
    message = fake.message.call_args.kwargs["message"]
    assert "CRITIC FEEDBACK" in message
    assert "continuity break at clip-3" in message


async def test_lines_split_into_bounded_batches() -> None:
    """More lines than the batch size ⇒ multiple bounded WM calls, and every line's
    directive is merged back (full coverage). Proves the unbounded-array split
    actually splits — a single giant call is the silent-degrade bug it fixes."""
    from src.services.stages.demo_video.voice_director import _DIRECTOR_BATCH_SIZE

    n = _DIRECTOR_BATCH_SIZE * 2 + 1  # 3 batches: sizes 5, 5, 1
    src_lines = [{"id": f"b{i}", "text": f"Line {i}.", "seconds": 0.0} for i in range(n)]
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"direction": "Energetic."}},
        prior_run_outputs={"demo_video_gen": {"lines": src_lines}},
    )

    per_call_counts: list[int] = []

    def _batch_resp(*_: Any, **kwargs: Any) -> WMResponse:
        msg = kwargs["message"]
        ids = [ln["id"] for ln in src_lines if f'"line_id":"{ln["id"]}"' in msg]
        per_call_counts.append(len(ids))
        # Distinctive value so a merged directive is distinguishable from the
        # neutral coverage-fill default.
        return _resp({"directives": [{"line_id": lid, "delivery_note": "hot"} for lid in ids]})

    fake = AsyncMock()
    fake.message.side_effect = _batch_resp
    out = await VoiceDirectorStage(client=fake).run({}, ctx=ctx)

    assert fake.message.call_count == 3  # ceil(11 / 5)
    assert per_call_counts == [5, 5, 1]  # bounded, order-preserving, nothing dropped
    by_id = {d.line_id: d for d in out.directives}
    assert [d.line_id for d in out.directives] == [ln["id"] for ln in src_lines]
    # Every line carries the LLM value (not the coverage-fill default) ⇒ ALL
    # batches — not just the first — were merged back.
    assert all(by_id[str(ln["id"])].delivery_note == "hot" for ln in src_lines)


async def test_malformed_output_reprompts_then_exhausts() -> None:
    """Malformed WM output raises ValueError (in LLM_RETRY_ON), so the stage re-prompts;
    once the retry budget is spent it surfaces StageRetriesExhaustedError whose cause is
    the missing-directives ValueError (never a silent pass)."""
    fake = AsyncMock()
    fake.message.return_value = _resp({"wrong_key": []})
    with pytest.raises(StageRetriesExhaustedError, match="missing directives"):
        await VoiceDirectorStage(client=fake).run({}, ctx=_ctx())
    assert fake.message.call_count >= 2  # re-prompted, not abandoned on attempt 1


async def test_the_model_is_told_not_to_think_before_it_answers() -> None:
    """The stage MUST send ``enable_reasoning=False``, and this is a fence, not a
    preference.

    ``voice_director`` died TWICE in production on 2026-08-20 (11:11 and 10:53)
    with ``StageTimeoutError: stage 'voice_director': all 2 attempts timed out
    (per-attempt budget 950.0s)``, terminal state ``failed`` — a real product
    failure, not a harness deadline.

    Measured through worker-management on the production path (Job
    ``wmprobe-segsel``, 2026-08-20) with the sibling stage's real prompt and
    schema at ``max_tokens=8000``: with reasoning ON the call ran to the stage's
    own 950 s deadline and returned nothing — twice, under BOTH candidate output
    schemas. With reasoning OFF the same call answered in 34-67 s carrying real
    content, every time. So this keyword is the difference between a stage that
    completes and one that cannot, and dropping it degrades SILENTLY: every unit
    test in this file passes without it, because a stubbed client answers
    instantly either way.

    ``w2v_common`` >= 0.61.0 is a hard requirement — worker-management's
    ``MessageRequest`` is ``extra="forbid"``, so an older client cannot send the
    field and a WM without it would 422 the call outright.
    """
    fake = AsyncMock()
    fake.message.return_value = _resp({"directives": [{"line_id": "b1"}]})

    await VoiceDirectorStage(client=fake).run({}, ctx=_ctx())

    assert fake.message.await_args.kwargs["enable_reasoning"] is False


async def test_a_retry_resumes_instead_of_re_paying_the_batches_that_answered() -> None:
    """The measured production death, in miniature.

    ``_DIRECTOR_BATCH_SIZE`` is 5, so 12 lines is 3 WM calls inside ONE
    ``stage_call`` slice. When the accumulator lived inside the attempt factory,
    a failure on batch 3 sent attempt 2 back to batch 1 — three more calls to
    reach the same place. With N batches that cannot all fit in one attempt, the
    stage could never complete no matter how many retries it was given.
    """
    from src.services.stages.demo_video import voice_director as vd

    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"direction": "Energetic."}},
        prior_run_outputs={
            "demo_video_gen": {
                "lines": [{"id": f"b{i}", "text": f"Line {i}.", "seconds": 0.0} for i in range(12)]
            }
        },
    )
    assert vd._DIRECTOR_BATCH_SIZE == 5, "test sizing assumes 3 batches over 12 lines"

    calls: list[int] = []
    fail_third = {"n": 0}

    async def _message(**kwargs: Any) -> WMResponse:
        calls.append(len(calls))
        fail_third["n"] += 1
        if fail_third["n"] == 3:
            raise ValueError("bad draw on the last batch")
        return _resp({"directives": []})

    fake = AsyncMock()
    fake.message = _message
    out = await vd.VoiceDirectorStage(client=fake).run({}, ctx=ctx)

    # 3 batches + ONE re-issue of the batch that failed = 4.
    # Restarting from batch 0 would be 3 + 3 = 6.
    assert len(calls) == 4, f"expected 4 WM calls (3 + 1 resumed), got {len(calls)}"
    assert len(out.directives) == 12, "coverage fill still emits one directive per line"
