from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageError
from src.services.stages.generic.line_fit import SPEED_EXPONENT
from src.services.stages.generic.script_draft import (
    NO_SOURCE_FACT_RULE,
    SOURCE_FACT_RULE,
    ScriptDraftOutput,
    ScriptDraftStage,
    ScriptLine,
    plan_line_trim,
    plan_sentence_trim,
    split_sentences,
)
from tests.services.stages.generic.conftest import measurement_at


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


@pytest.mark.asyncio
async def test_script_draft_uses_brief_and_notes() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": [{"text": "hi", "seconds": 3.0}]})
    stage = ScriptDraftStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={
            "brief_intake": {
                "title": "T",
                "angle": "growth",
                "audience": "saas",
                "target_duration_seconds": 30,
                "voice_descriptor": "crisp",
            }
        },
        open_annotations_by_stage={"script_draft": [{"text": "tighter"}]},
    )
    out = await stage.run({}, ctx=ctx)
    assert len(out.lines) == 1
    assert out.lines[0].text == "hi"
    assert out.lines[0].id  # element id present
    _, kwargs = fake.message.call_args
    assert "tighter" in kwargs["message"]


@pytest.mark.asyncio
async def test_output_schema_constrains_element_fields() -> None:
    # Regression (E2E 2026-06-01): WM structured output only emits properties
    # NAMED in the item schema. A loose ``items: {"type": "object"}`` yielded
    # text-less elements → KeyError("text") → retries exhausted → failed run.
    # The item schema MUST name+require text and seconds.
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": [{"text": "hi", "seconds": 3.0}]})
    stage = ScriptDraftStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={"brief_intake": _BRIEF},
    )
    await stage.run({}, ctx=ctx)
    _, kwargs = fake.message.call_args
    item_schema = kwargs["output_schema"]["properties"]["lines"]["items"]
    assert set(item_schema["properties"]) >= {"text", "seconds"}
    assert set(item_schema["required"]) == {"text", "seconds"}


def test_default_wm_client_is_on_the_llm_ladder_not_the_wm_client_default() -> None:
    """A default-constructed stage must get the LLM CLIENT RUNG, not WMClient's default.

    ⚠️ THE FACT THIS TEST WAS BUILT ON WAS WRONG. Both the name and the comment said
    "WMClient() defaults to a 120s httpx timeout", and asserted ``timeout > 120.0``.
    ``WMClient.__init__`` (w2v_common/clients/wm_client.py) actually defaults to::

        self.timeout = timeout if timeout is not None else _env_float("WM_CLIENT_TIMEOUT_S", 600.0)

    600s, not 120s — so ``> 120.0`` passed for every value from 120.1s upward and could
    not distinguish "on the ladder" from "fell back to the WM default". It was
    near-vacuous, which is worse than absent because it read as coverage.

    The assertion is now EXACT: the rung, and only the rung. Mutation-check: change the
    stage to ``WMClient()`` and this goes red (930.0 != 600.0), where the old form stayed
    green.
    """
    from w2v_common.clients.wm_client import WMClient

    from src.services.stages.llm_budget import llm_budget_seconds, llm_client_timeout_seconds

    # Pin the real default so the claim above cannot rot silently.
    assert WMClient().timeout == 600.0

    timeout = ScriptDraftStage()._wm.timeout
    assert timeout == llm_client_timeout_seconds()
    assert timeout != WMClient().timeout  # the original regression, stated truthfully
    assert timeout < llm_budget_seconds()  # the ladder ordering invariant


@pytest.mark.asyncio
async def test_script_draft_raises_without_brief() -> None:
    fake = AsyncMock()
    stage = ScriptDraftStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
    )
    with pytest.raises(RuntimeError, match="brief_intake"):
        await stage.run({}, ctx=ctx)


@pytest.mark.asyncio
async def test_script_draft_raises_on_malformed_llm_output() -> None:
    # Non-dict WM output (no `lines` key reachable) → ValueError → StageError.
    fake = AsyncMock()
    fake.message.return_value = _resp(None)
    stage = ScriptDraftStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={
            "brief_intake": {
                "title": "T",
                "angle": "growth",
                "audience": "saas",
                "target_duration_seconds": 30,
                "voice_descriptor": "crisp",
            }
        },
    )
    with pytest.raises(StageError):
        await stage.run({}, ctx=ctx)


def _brief_ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={
            "brief_intake": {
                "title": "T",
                "angle": "growth",
                "audience": "saas",
                "target_duration_seconds": 30,
                "voice_descriptor": "crisp",
            }
        },
    )


@pytest.mark.asyncio
async def test_script_draft_retries_transient_empty_then_succeeds() -> None:
    # min_elements=1: a bare `[]` is a transient draw — re-prompt within the retry
    # budget rather than fail on the first attempt. A later non-empty draw recovers.
    fake = AsyncMock()
    # The recovered draw is ON the order (30s at the stubbed 150 wpm = 75 words), so
    # the length re-ask does not fire and the count below is the retry alone.
    fake.message.side_effect = [
        _resp({"lines": []}),
        _resp({"lines": [{"text": " ".join(["word"] * 75), "seconds": 30.0}]}),
    ]
    stage = ScriptDraftStage(client=fake)
    stage._llm_retries = 2  # deterministic: up to 3 attempts, no backoff-count drift
    out = await stage.run({}, ctx=_brief_ctx())
    assert fake.message.call_count == 2  # first empty draw re-prompted, second used
    assert out is not None


@pytest.mark.asyncio
async def test_script_draft_raises_on_persistently_empty_lines() -> None:
    # A PERSISTENTLY empty `lines` list must still fail loud after exhausting
    # retries — never flow into storyboard → zero shots → video-compile 422
    # (bug-bash B1+B2, 2026-05-09). The failure surfaces as StageRetriesExhausted
    # (a StageError) whose message carries the "too few" text.
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": []})
    stage = ScriptDraftStage(client=fake)
    stage._llm_retries = 2  # 3 attempts
    with pytest.raises(StageError, match="too few"):
        await stage.run({}, ctx=_brief_ctx())
    assert fake.message.call_count == 3  # proves it retried, not first-attempt fatal


@pytest.mark.asyncio
async def test_source_script_passes_through_verbatim() -> None:
    fake = AsyncMock()
    stage = ScriptDraftStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={
            "brief_intake": {
                "title": "T",
                "angle": "growth",
                "audience": "saas",
                "target_duration_seconds": 30,
                "voice_descriptor": "crisp",
                "source_script": "Line one.\nLine two.",
            }
        },
    )
    out = await stage.run({}, ctx=ctx)
    assert [line.text for line in out.lines] == ["Line one.", "Line two."]
    fake.message.assert_not_called()


@pytest.mark.asyncio
async def test_source_script_blank_falls_back_to_llm() -> None:
    fake = AsyncMock()
    # On-order reply (75 words = 30s at the stubbed 150 wpm) so the length re-ask
    # stays out of the call count this test is about.
    on_order = " ".join(["hi"] * 75)
    fake.message.return_value = _resp({"lines": [{"text": on_order, "seconds": 30.0}]})
    stage = ScriptDraftStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={
            "brief_intake": {
                "title": "T",
                "angle": "growth",
                "audience": "saas",
                "target_duration_seconds": 30,
                "voice_descriptor": "crisp",
                "source_script": "   \n  ",
            }
        },
    )
    out = await stage.run({}, ctx=ctx)
    assert out.lines[0].text == on_order
    fake.message.assert_called_once()


_BRIEF = {
    "title": "T",
    "angle": "growth",
    "audience": "saas",
    "target_duration_seconds": 30,
    "voice_descriptor": "crisp",
}


@pytest.mark.asyncio
async def test_script_draft_verbatim_present_succeeds_and_injects_prompt() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"lines": [{"text": "Buy now while stocks last", "seconds": 3.0}]}
    )
    stage = ScriptDraftStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"verbatim_items": [{"text": "Buy now"}]},
        prior_run_outputs={"brief_intake": _BRIEF},
    )
    out = await stage.run({}, ctx=ctx)
    assert out.lines[0].text == "Buy now while stocks last"
    _, kwargs = fake.message.call_args
    assert "Buy now" in kwargs["message"]


@pytest.mark.asyncio
async def test_script_draft_verbatim_missing_raises() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": [{"text": "Hello there", "seconds": 2.0}]})
    stage = ScriptDraftStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"verbatim_items": [{"text": "Buy now"}]},
        prior_run_outputs={"brief_intake": _BRIEF},
    )
    with pytest.raises(StageError):
        await stage.run({}, ctx=ctx)


@pytest.mark.asyncio
async def test_script_draft_source_script_verbatim_present_succeeds() -> None:
    fake = AsyncMock()
    stage = ScriptDraftStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"verbatim_items": [{"text": "Buy now"}]},
        prior_run_outputs={
            "brief_intake": {**_BRIEF, "source_script": "Welcome.\nBuy now while you can.\nThanks."}
        },
    )
    out = await stage.run({}, ctx=ctx)
    assert [line.text for line in out.lines] == [
        "Welcome.",
        "Buy now while you can.",
        "Thanks.",
    ]
    fake.message.assert_not_called()


@pytest.mark.asyncio
async def test_script_draft_source_script_verbatim_missing_raises() -> None:
    fake = AsyncMock()
    stage = ScriptDraftStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"verbatim_items": [{"text": "Buy now"}]},
        prior_run_outputs={
            "brief_intake": {**_BRIEF, "source_script": "Welcome.\nThanks for watching."}
        },
    )
    with pytest.raises(StageError):
        await stage.run({}, ctx=ctx)
    fake.message.assert_not_called()


# ---------------------------------------------------------------------------
# Verbatim kind scoping — a `slide`-kind phrase must NOT be enforced against the
# narration script (it belongs on a slide), while `tts`-kind and legacy kindless
# items stay hard-enforced. The `tts` case is the anti-regression guard: if the
# stage→kind filter ever returns [], enforcement vanishes silently and every run
# goes green — which looks like success.
# ---------------------------------------------------------------------------


def _draft_ctx(items: list[dict]) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"verbatim_items": items},
        prior_run_outputs={"brief_intake": _BRIEF},
    )


@pytest.mark.asyncio
async def test_script_draft_slide_kind_verbatim_not_enforced_on_narration() -> None:
    """A slide-kind phrase absent from the narration must NOT fail the script stage."""
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": [{"text": "Hello there", "seconds": 2.0}]})
    stage = ScriptDraftStage(client=fake)

    out = await stage.run({}, ctx=_draft_ctx([{"text": "Buy now", "kind": "slide"}]))

    assert out.lines[0].text == "Hello there"
    # ...and it is not pushed into the narration prompt either.
    _, kwargs = fake.message.call_args
    assert "Buy now" not in kwargs["message"]
    # ...but it is NOT silently lost: the stage records a soft-degrade note, which
    # `runs/_shared.py` surfaces as a `degraded` run. Without this the default UI
    # action (kind='slide') would delete the phrase on a green run.
    assert any("Buy now" in n for n in out.notes)


@pytest.mark.asyncio
async def test_script_draft_tts_kind_verbatim_still_hard_fails_when_missing() -> None:
    """ANTI-REGRESSION: a filter that returns [] would make this pass silently."""
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": [{"text": "Hello there", "seconds": 2.0}]})
    stage = ScriptDraftStage(client=fake)

    with pytest.raises(StageError):
        await stage.run({}, ctx=_draft_ctx([{"text": "Buy now", "kind": "tts"}]))


@pytest.mark.asyncio
async def test_script_draft_kindless_verbatim_still_hard_fails_when_missing() -> None:
    """Backward compat: a pre-0017 snapshot item (no ``kind``) applies everywhere."""
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": [{"text": "Hello there", "seconds": 2.0}]})
    stage = ScriptDraftStage(client=fake)

    with pytest.raises(StageError):
        await stage.run({}, ctx=_draft_ctx([{"text": "Buy now"}]))


@pytest.mark.asyncio
async def test_script_draft_mixed_kinds_enforces_tts_only() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"lines": [{"text": "Buy now while stocks last", "seconds": 3.0}]}
    )
    stage = ScriptDraftStage(client=fake)

    out = await stage.run(
        {},
        ctx=_draft_ctx(
            [{"text": "Buy now", "kind": "tts"}, {"text": "Limited offer", "kind": "slide"}]
        ),
    )

    assert out.lines[0].text == "Buy now while stocks last"
    _, kwargs = fake.message.call_args
    assert "Buy now" in kwargs["message"]
    assert "Limited offer" not in kwargs["message"]


@pytest.mark.asyncio
async def test_script_draft_source_script_slide_kind_not_enforced() -> None:
    """The non-LLM source_script short-circuit must scope by kind too."""
    fake = AsyncMock()
    stage = ScriptDraftStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"verbatim_items": [{"text": "Buy now", "kind": "slide"}]},
        prior_run_outputs={"brief_intake": {**_BRIEF, "source_script": "Welcome.\nThanks."}},
    )

    out = await stage.run({}, ctx=ctx)

    assert [line.text for line in out.lines] == ["Welcome.", "Thanks."]
    fake.message.assert_not_called()


@pytest.mark.asyncio
async def test_script_draft_source_script_tts_kind_still_hard_fails() -> None:
    fake = AsyncMock()
    stage = ScriptDraftStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"verbatim_items": [{"text": "Buy now", "kind": "tts"}]},
        prior_run_outputs={"brief_intake": {**_BRIEF, "source_script": "Welcome.\nThanks."}},
    )

    with pytest.raises(StageError):
        await stage.run({}, ctx=ctx)


@pytest.mark.asyncio
async def test_script_draft_source_script_path_also_records_the_drop() -> None:
    """The non-LLM short-circuit must record the drop too, not just the LLM path."""
    fake = AsyncMock()
    stage = ScriptDraftStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"verbatim_items": [{"text": "Buy now", "kind": "slide"}]},
        prior_run_outputs={"brief_intake": {**_BRIEF, "source_script": "Welcome.\nThanks."}},
    )

    out = await stage.run({}, ctx=ctx)

    fake.message.assert_not_called()
    assert any("Buy now" in n for n in out.notes)


# ---------------------------------------------------------------------------
# Word budget + one bounded length re-ask (live run 01a156a4-e906, 2026-10-20:
# 12.25s of narration for a 30s order, -59%, nothing flagged it).
# ---------------------------------------------------------------------------


def _words(n: int) -> str:
    return " ".join(["word"] * n)


def _rewrite_asks(fake: AsyncMock) -> list[str]:
    """Messages of the per-line fit's single-line rewrites."""
    return [
        c.kwargs["message"]
        for c in fake.message.call_args_list
        if c.kwargs["message"].startswith("Rewrite ONE line")
    ]


def _draft_asks(fake: AsyncMock) -> list[str]:
    """Messages of the whole-script draws (opening draw + length re-asks)."""
    return [
        c.kwargs["message"]
        for c in fake.message.call_args_list
        if not c.kwargs["message"].startswith("Rewrite ONE line")
    ]


@pytest.mark.asyncio
async def test_prompt_carries_word_budget_at_measured_rate() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": [{"text": _words(75), "seconds": 30.0}]})
    await ScriptDraftStage(client=fake).run({}, ctx=_brief_ctx())
    msg = fake.message.call_args.kwargs["message"]
    # 30s x ASK_LONG_FACTOR 1.15 at the stubbed 150 wpm == 86 words, asked LONG.
    assert "86 words" in msg
    assert "150 words per minute" in msg
    assert fake.message.call_count == 1  # on-order draw: no re-ask


@pytest.mark.asyncio
async def test_short_draw_is_reasked_and_closer_draw_ships() -> None:
    # The "model" answers short until it is told the draft was too short, so what
    # ships proves the re-ask carried the corrective instruction to the model.
    asked: list[str] = []

    async def _model(*, message: str, **_: object) -> WMResponse:
        asked.append(message)
        words = 72 if "TOO SHORT" in message else 30  # 30 words = 12s of 30s: -60%
        return _resp({"lines": [{"text": _words(words), "seconds": 12.0}]})

    fake = AsyncMock()
    fake.message.side_effect = _model
    out = await ScriptDraftStage(client=fake).run({}, ctx=_brief_ctx())
    assert sum(len(line.text.split()) for line in out.lines) == 72
    assert any("TOO SHORT" in m and "86 words" in m for m in asked)


@pytest.mark.asyncio
async def test_reask_that_comes_back_shorter_is_discarded_and_ends_the_loop() -> None:
    async def _model(*, message: str, **_: object) -> WMResponse:
        # 20s (-33%) first; the "correction" comes back even shorter, 12s.
        words = 30 if "REVISION REQUIRED" in message else 50
        return _resp({"lines": [{"text": _words(words), "seconds": 20.0}]})

    fake = AsyncMock()
    fake.message.side_effect = _model
    out = await ScriptDraftStage(client=fake).run({}, ctx=_brief_ctx())
    # Two draft draws, then ONE per-line rewrite that came back unchanged (ends the
    # line's rewrites), then the speed clamp: slowed to the 0.9 floor, no further.
    assert len(_draft_asks(fake)) == 2
    assert len(_rewrite_asks(fake)) == 1
    assert sum(len(line.text.split()) for line in out.lines) == 50
    assert out.lines[0].speed == 0.9
    assert out.lines[0].seconds == pytest.approx(20.0 * 0.9**-SPEED_EXPONENT)
    # Short after the re-asks: shipped as the longest draft, and SAID so.
    assert [d["code"] for d in out.degradations] == ["duration_out_of_tolerance"]
    assert "shipped_longest_draft" in out.degradations[0]["detail"]


@pytest.mark.asyncio
async def test_failed_reask_keeps_first_draw() -> None:
    fake = AsyncMock()
    fake.message.side_effect = [
        _resp({"lines": [{"text": _words(30), "seconds": 12.0}]}),
        RuntimeError("wm down"),
        RuntimeError("wm down"),
        RuntimeError("wm down"),
    ]
    stage = ScriptDraftStage(client=fake)
    stage._llm_retries = 0
    out = await stage.run({}, ctx=_brief_ctx())
    assert sum(len(line.text.split()) for line in out.lines) == 30


@pytest.mark.asyncio
async def test_a_truncated_draw_is_redrawn_not_run_fatal() -> None:
    """Live 2026-10-20 (runs 01a158a5-c749 / 01a158a9-5a9b): the platform model looped
    inside one ``text`` string until ``max_tokens`` and the stage failed on attempt 1
    as ``customer_input``. A truncated draw is a retryable model failure: the stage
    draws again and the run continues."""
    from src.models.failure_class import EXTERNAL
    from src.services.stages.upstream import TransientUpstreamError

    truncated = TransientUpstreamError(
        "Platform LLM (inference-router, model local-llm): output token limit reached "
        "before the reply finished (max_tokens=10050, completion_tokens=10050)"
    )
    truncated.failure_class = EXTERNAL
    fake = AsyncMock()
    ok = _resp({"lines": [{"text": "hi", "seconds": 3.0}]})
    # truncated draw, its re-draw, the (short-script) length re-ask, then the
    # per-line fit's rewrite of "hi" (unchanged -> the line keeps its text).
    fake.message.side_effect = [truncated, ok, ok, ok]
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={"brief_intake": _BRIEF},
    )
    out = await ScriptDraftStage(client=fake).run({}, ctx=ctx)
    assert [line.text for line in out.lines] == ["hi"]
    assert fake.message.await_count == 4


def _sentences(*counts: int) -> str:
    """One line of whole sentences, sentence ``i`` holding ``counts[i]`` words."""
    return " ".join(f"S{i} " + " ".join(["word"] * (n - 1)) + "." for i, n in enumerate(counts))


@pytest.mark.asyncio
async def test_verdict_uses_phase_b_rate_measured_on_the_draft(
    _stub_generic_speech_rate: AsyncMock,
    _stub_generic_measurement: AsyncMock,
) -> None:
    """Live 2026-10-20: 91 words for a 30s order looked on-order at the ~198 wpm
    calibration-passage rate (27.6s) but were SPOKEN at ~149 wpm (36.6s, +22%).
    The verdict must use the rate measured on the draft itself — and a LONG draft is
    now trimmed by whole sentences at that rate, never re-asked."""
    from src.services.stages.shared.speech_rate import ScriptSpeechRate, SpeechRate

    async def _rate(_ctx: object, *, texts: object, **kwargs: object) -> ScriptSpeechRate:
        wpm = 150.0 if texts else 198.0  # Phase B (the draft) vs Phase A (passage)
        rate = SpeechRate(wpm=wpm, measured=True)
        return ScriptSpeechRate(rate=rate, prior=rate)

    _stub_generic_speech_rate.side_effect = _rate
    asked: list[str] = []
    draft = [_sentences(10, 10, 10), _sentences(12, 12, 12), _sentences(9, 8, 8)]  # 91

    async def _model(*, message: str, **_: object) -> WMResponse:
        asked.append(message)
        return _resp({"lines": [{"text": t, "seconds": 10.0} for t in draft]})

    fake = AsyncMock()
    fake.message.side_effect = _model
    out = await ScriptDraftStage(client=fake).run({}, ctx=_brief_ctx())

    assert fake.message.call_count == 1  # 91 words at 150 wpm = 36.4s: long -> trim
    assert "198 words per minute" in asked[0]  # opening ask sized at the prior
    words = sum(len(line.text.split()) for line in out.lines)
    assert 27.0 <= words / 150.0 * 60.0 <= 33.0
    assert out.degradations == []
    # The draft is judged by synthesizing EVERY line under narrate's own stage name
    # and speed (Phase C) — never by a Phase-B sample of it.
    assert not [c for c in _stub_generic_speech_rate.await_args_list if c.kwargs["texts"]]
    measured = _stub_generic_measurement.await_args_list
    assert measured and all(c.kwargs["stage_name"] == "narrate" for c in measured)
    assert all(c.kwargs["speed"] is None for c in measured)


def test_budget_sentence_states_a_five_percent_band() -> None:
    prompt = ScriptDraftStage._build_prompt(_BRIEF, [], wpm=150.0)
    # 30s x 1.15 at 150 wpm = 86 words, asked long; +/-5% band = 81..91.
    assert "no fewer than 81 and no more than 91" in prompt
    assert "give or take" not in prompt


def test_prompt_carries_the_brief_audience() -> None:
    """brief_intake normalises ``audience``; the generic prompt used to drop it, so
    every generic film was written for nobody in particular."""
    brief = {**_BRIEF, "audience": "hospital night-shift nurses"}
    prompt = ScriptDraftStage._build_prompt(brief, [], wpm=150.0)
    assert "Audience: hospital night-shift nurses" in prompt


@pytest.mark.asyncio
async def test_each_draft_is_judged_at_its_own_measured_rate_live_01a15968(
    _stub_generic_speech_rate: AsyncMock,
    _stub_generic_measurement: AsyncMock,
) -> None:
    """Live run 01a15968-48fd (2026-10-20, 30s order) delivered 23.47s (-21.8%).

    The real numbers: Phase-A prior 158.1 wpm (passage 197.6 x 0.80) sized the ask at
    79 words; draft 1 = 48 words, measured at 199.9 wpm (14.4s) -> re-asked for 100;
    draft 2 = 65 words, which narrate spoke at 166.8 wpm (23.38s). The verdict reused
    draft 1's 199.9 (19.5s), called it "closer" and shipped it. Each draft must be
    measured on its OWN lines, so the 65-word draft is judged at 166.8 wpm, found
    short, and re-asked again.
    """
    from src.services.stages.shared.speech_rate import ScriptSpeechRate, SpeechRate

    by_words = {48: 199.9, 65: 166.8, 84: 166.8}

    async def _prior(_ctx: object, **_: object) -> ScriptSpeechRate:
        rate = SpeechRate(wpm=158.1, measured=True)
        return ScriptSpeechRate(rate=rate, prior=rate)

    async def _measure(_ctx: object, *, texts: Sequence[str], **_: object) -> object:
        return measurement_at(texts, by_words[sum(len(str(t).split()) for t in texts)])

    _stub_generic_speech_rate.side_effect = _prior
    _stub_generic_measurement.side_effect = _measure
    asked: list[str] = []

    async def _model(*, message: str, **_: object) -> WMResponse:
        asked.append(message)
        words = [48, 65, 84][len(asked) - 1]
        return _resp({"lines": [{"text": _words(words), "seconds": 5.0}]})

    fake = AsyncMock()
    fake.message.side_effect = _model
    out = await ScriptDraftStage(client=fake).run({}, ctx=_brief_ctx())

    assert "91 words" in asked[0]  # opening ask: 34.5s (30s x 1.15) at the 158.1 prior
    assert "TOO SHORT" in asked[1] and "115 words" in asked[1]  # sized at draft 1's 199.9
    # The 65-word draft is NOT shipped: at its own 166.8 wpm it is 23.4s, out of band,
    # so a second re-ask fires, sized LONG at THAT draft's rate (34.5s x 166.8 / 60 = 96).
    assert fake.message.call_count == 3
    assert "TOO SHORT" in asked[2] and "96 words" in asked[2] and "23 seconds" in asked[2]
    assert sum(len(line.text.split()) for line in out.lines) == 84  # 30.2s, in band
    # Each draft measured on its OWN lines — every line, all three drafts.
    assert [len(c.kwargs["texts"]) for c in _stub_generic_measurement.await_args_list] == [1, 1, 1]


@pytest.mark.asyncio
async def test_length_reasks_are_bounded(_stub_generic_speech_rate: AsyncMock) -> None:
    """A model that keeps improving but never reaches the band costs at most
    1 + MAX_LENGTH_REASKS whole-script draws; the per-line fit then spends at most
    MAX_REWRITES_PER_LINE rewrites on the (single) line, keeping each that lands
    closer, and the speed clamp takes the rest."""
    from src.services.stages.generic.line_fit import MAX_REWRITES_PER_LINE, SPEED_MIN
    from src.services.stages.generic.script_draft import MAX_LENGTH_REASKS

    counts = iter([20, 30, 40, 50, 60, 70, 80])

    async def _model(*, message: str, **_: object) -> WMResponse:
        return _resp({"lines": [{"text": _words(next(counts)), "seconds": 5.0}]})

    fake = AsyncMock()
    fake.message.side_effect = _model
    out = await ScriptDraftStage(client=fake).run({}, ctx=_brief_ctx())
    assert len(_draft_asks(fake)) == 1 + MAX_LENGTH_REASKS
    assert len(_rewrite_asks(fake)) == MAX_REWRITES_PER_LINE
    # 40 words (16s) shipped by the drafting loop; rewrites drew 50 (20s) and 60
    # (24s), each closer to 30s; 24s is 20% short -> slowed to the clamp floor.
    assert sum(len(line.text.split()) for line in out.lines) == 60
    assert out.lines[0].speed == SPEED_MIN
    assert out.lines[0].seconds == pytest.approx(24.0 * SPEED_MIN**-SPEED_EXPONENT)


# ---------------------------------------------------------------------------
# Ask long, then trim (owner decision 2026-10-20, live run 01a159b1-785e)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_01a159b1_short_then_long_draft_is_trimmed_into_band(
    _stub_generic_speech_rate: AsyncMock,
    _stub_generic_measurement: AsyncMock,
) -> None:
    """Live run 01a159b1-785e (30s order) drew 18.47s, 39.84s, 21.34s and shipped
    -28.7%. Now: the 18.47s draft is re-asked once, the 39.84s draft stops the
    loop (it is over the floor), and it is trimmed by whole sentences, at ITS
    measured rate, into 27-33s. No third draw."""

    long_lines = [
        "Meet Acme, the fastest way to ship. It halves your deploy time. Teams love it.",
        "Setup takes five minutes from start to finish. You connect a repository and pick "
        "a region. Then every branch you push gets its own preview, automatically.",
        "Security is built in from day one. Secrets never leave your vault. Audit logs "
        "record every change anyone makes.",
        "Rollbacks are a single click away. Nothing is ever lost.",
        "Pricing is simple and fair for every team size. Start free today at acme dot dev.",
    ]
    long_words = sum(len(t.split()) for t in long_lines)
    long_wpm = long_words / 39.84 * 60.0
    short_text = _sentences(10, 10, 10, 10, 7)  # 47 words
    short_wpm = 47 / 18.47 * 60.0

    async def _measure(_ctx: object, *, texts: Sequence[str], **_: object) -> object:
        words = sum(len(str(t).split()) for t in texts)
        # The short draft, then the long draft AND its trimmed text.
        return measurement_at(texts, short_wpm if words == 47 else long_wpm)

    _stub_generic_measurement.side_effect = _measure
    asked: list[str] = []

    async def _model(*, message: str, **_: object) -> WMResponse:
        asked.append(message)
        if len(asked) == 1:
            return _resp({"lines": [{"text": short_text, "seconds": 18.0}]})
        return _resp({"lines": [{"text": t, "seconds": 8.0} for t in long_lines]})

    fake = AsyncMock()
    fake.message.side_effect = _model
    out = await ScriptDraftStage(client=fake).run({}, ctx=_brief_ctx())

    assert fake.message.call_count == 2  # one re-ask, then the long draft stops the loop
    assert "TOO SHORT" in asked[1]
    seconds = sum(len(line.text.split()) for line in out.lines) / long_wpm * 60.0
    assert 27.0 <= seconds <= 33.0, seconds
    assert out.degradations == []
    assert len(out.lines) == len(long_lines)  # no line emptied or dropped
    for line, original in zip(out.lines, long_lines, strict=True):
        kept = split_sentences(line.text)
        source = split_sentences(original)
        assert kept and all(k in source for k in kept)  # whole sentences only
        assert kept == [x for x in source if x in kept]  # order preserved
        assert kept[0] == source[0]  # never empties a line
    assert out.lines[0].text.startswith("Meet Acme, the fastest way to ship.")  # hook
    assert out.lines[-1].text.endswith("Start free today at acme dot dev.")  # CTA
    assert out.total_seconds == pytest.approx(sum(line.seconds for line in out.lines))


@pytest.mark.asyncio
async def test_a_draft_already_in_band_is_untouched() -> None:
    lines = [_sentences(10, 10, 10), _sentences(15, 15, 13)]  # 73 words = 29.2s of 30s
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": [{"text": t, "seconds": 15.0} for t in lines]})
    out = await ScriptDraftStage(client=fake).run({}, ctx=_brief_ctx())
    assert fake.message.call_count == 1
    assert [line.text for line in out.lines] == lines
    # Seconds are the MEASURED clip lengths (30 and 43 words at 150 wpm), not the
    # model's own guesses — and nothing is re-synthesized inside +/-5%.
    assert [line.seconds for line in out.lines] == pytest.approx([12.0, 17.2])
    assert [line.speed for line in out.lines] == [None, None]
    assert out.degradations == []


def test_sentence_trim_keeps_hook_closer_and_cuts_the_longest_line_tail() -> None:
    lines = [
        _sentences(6, 6),
        _sentences(5, 6, 6, 6),  # the longest line: its tail goes first
        _sentences(6, 6),
    ]
    # 47 words at 150 wpm = 18.8s against a 16s order (+17.5%). Every droppable
    # sentence is 6 words, so ONE cut lands at 16.4s; the tie-break picks the tail
    # of the longest line, not the opener's second sentence.
    trimmed, plan = plan_sentence_trim(lines, ordered_seconds=16.0, words_per_minute=150.0)
    assert len(plan.dropped_indices) == 1 and not plan.exceeds_limit
    assert trimmed[0] == lines[0] and trimmed[2] == lines[2]
    assert split_sentences(trimmed[1]) == split_sentences(lines[1])[:-1]
    assert sum(len(t.split()) for t in trimmed) / 150.0 * 60.0 == pytest.approx(16.4)


def test_sentence_trim_never_cuts_a_single_sentence_line_and_reports_the_floor() -> None:
    lines = [_words(60), _words(40)]  # no sentence boundary anywhere
    trimmed, plan = plan_sentence_trim(lines, ordered_seconds=20.0, words_per_minute=150.0)
    assert trimmed == lines
    assert plan.exceeds_limit


def test_splitter_is_abbreviation_aware_english_and_dutch() -> None:
    text = "Ask Dr. Smith, e.g. about 3.5 percent. Bel bijv. de heer Jansen."
    assert split_sentences(text) == [
        "Ask Dr. Smith, e.g. about 3.5 percent.",
        "Bel bijv. de heer Jansen.",
    ]
    assert split_sentences("Meet J. Smith today. Then go.") == ["Meet J. Smith today.", "Then go."]
    assert split_sentences("Mr. Jansen, d.w.z. de baas. Nr. 5 wint.") == [
        "Mr. Jansen, d.w.z. de baas.",
        "Nr. 5 wint.",
    ]


_CODE = "Use code SAVE20 at checkout now to get twenty percent off today."
_CODE_LINES = [
    _sentences(10, 10),
    _sentences(6, 12, 12, 12) + " " + _CODE,  # the code is the longest line's TAIL
    _sentences(10, 10),
]  # 100 words = 40.0s at 150 wpm against a 30s order


def test_sentence_trim_never_drops_a_verbatim_sentence() -> None:
    # Control: unprotected, the code sentence is exactly what the trim cuts first.
    unprotected, _ = plan_sentence_trim(_CODE_LINES, ordered_seconds=30.0, words_per_minute=150.0)
    assert _CODE not in " ".join(unprotected)
    trimmed, plan = plan_sentence_trim(
        _CODE_LINES,
        ordered_seconds=30.0,
        words_per_minute=150.0,
        verbatim_items=[{"text": "use code save20 at checkout now", "kind": "tts"}],
    )
    assert plan.dropped_indices and not plan.exceeds_limit
    assert _CODE in " ".join(trimmed)


@pytest.mark.asyncio
async def test_trim_keeps_verbatim_phrase_through_the_stage() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"lines": [{"text": t, "seconds": 10.0} for t in _CODE_LINES]}
    )
    out = await ScriptDraftStage(client=fake).run(
        {}, ctx=_draft_ctx([{"text": "Use code SAVE20 at checkout now", "kind": "tts"}])
    )
    text = " ".join(line.text for line in out.lines)
    assert "Use code SAVE20 at checkout now" in text
    assert 27.0 <= len(text.split()) / 150.0 * 60.0 <= 33.0  # still trimmed into band


@pytest.mark.asyncio
async def test_trim_that_loses_a_verbatim_phrase_fails_the_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the protection ever misses, the post-trim re-gate fails the stage exactly
    like the base verbatim gate — it never ships green without the phrase."""
    monkeypatch.setattr(
        "src.services.stages.generic.script_draft.verbatim_protected_units",
        lambda *_a, **_k: set(),
    )
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"lines": [{"text": t, "seconds": 10.0} for t in _CODE_LINES]}
    )
    with pytest.raises(StageError, match="required verbatim text missing"):
        await ScriptDraftStage(client=fake).run(
            {}, ctx=_draft_ctx([{"text": "Use code SAVE20 at checkout now", "kind": "tts"}])
        )


# ---------------------------------------------------------------------------
# Tier two: whole-line trim (live run 01a15a56-0306: 9 one-sentence lines,
# 58.3s for a 30s order, "ask-long trim 58.3s -> 58.3s", shipped +94.8%)
# ---------------------------------------------------------------------------

_ONE_SENTENCE_WORDS = [14, 18, 16, 17, 15, 19, 16, 17, 14]  # 146 words = 58.4s @150


def _one_sentence_lines() -> list[str]:
    return [
        f"Line{i} " + " ".join(["word"] * (n - 1)) + "." for i, n in enumerate(_ONE_SENTENCE_WORDS)
    ]


@pytest.mark.asyncio
async def test_live_01a15a56_one_sentence_lines_are_trimmed_by_whole_lines() -> None:
    lines = _one_sentence_lines()
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": [{"text": t, "seconds": 6.0} for t in lines]})
    out = await ScriptDraftStage(client=fake).run({}, ctx=_brief_ctx())

    # 58.4s is already >= 1.5x the order: it is long, so NO re-ask, straight to the trim.
    assert fake.message.call_count == 1
    texts = [line.text for line in out.lines]
    seconds = sum(len(t.split()) for t in texts) / 150.0 * 60.0
    assert 27.0 <= seconds <= 33.0, seconds
    assert out.degradations == []
    assert texts[0] == lines[0] and texts[-1] == lines[-1]  # hook + CTA kept
    assert texts == [t for t in lines if t in texts]  # survivors in original order
    assert all(t in lines for t in texts)  # whole lines, untouched
    assert len({line.id for line in out.lines}) == len(out.lines)
    assert out.total_seconds == pytest.approx(sum(line.seconds for line in out.lines))


def test_line_trim_prefers_the_longest_middle_lines() -> None:
    lines = _one_sentence_lines()
    kept, plan = plan_line_trim(lines, ordered_seconds=30.0, words_per_minute=150.0)
    assert 0 in kept and len(lines) - 1 in kept
    assert kept == sorted(kept)
    assert not plan.exceeds_limit
    assert 27.0 <= sum(len(lines[i].split()) for i in kept) / 150.0 * 60.0 <= 33.0
    assert 5 not in kept  # the 19-word line goes


def test_line_trim_never_drops_a_verbatim_middle_line() -> None:
    lines = _one_sentence_lines()
    lines[5] = "Use code SAVE20 at checkout now " + " ".join(["word"] * 12) + "."  # 18 words
    unprotected, _ = plan_line_trim(lines, ordered_seconds=30.0, words_per_minute=150.0)
    assert 5 not in unprotected  # control: the longest middle line is the first to go
    kept, plan = plan_line_trim(
        lines,
        ordered_seconds=30.0,
        words_per_minute=150.0,
        verbatim_items=[{"text": "use code save20 at checkout now", "kind": "tts"}],
    )
    assert 5 in kept and 0 in kept and len(lines) - 1 in kept
    assert plan.dropped_indices and not plan.exceeds_limit


@pytest.mark.asyncio
async def test_verbatim_middle_line_survives_through_the_stage() -> None:
    lines = _one_sentence_lines()
    lines[5] = "Use code SAVE20 at checkout now " + " ".join(["word"] * 12) + "."
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": [{"text": t, "seconds": 6.0} for t in lines]})
    out = await ScriptDraftStage(client=fake).run(
        {}, ctx=_draft_ctx([{"text": "Use code SAVE20 at checkout now", "kind": "tts"}])
    )
    texts = [line.text for line in out.lines]
    assert lines[5] in texts and texts[0] == lines[0] and texts[-1] == lines[-1]
    assert 27.0 <= sum(len(t.split()) for t in texts) / 150.0 * 60.0 <= 33.0


def test_line_trim_never_reduces_a_two_line_draft() -> None:
    lines = [_words(60) + ".", _words(60) + "."]  # 48s against 20s
    kept, plan = plan_line_trim(lines, ordered_seconds=20.0, words_per_minute=150.0)
    assert kept == [0, 1]
    assert plan.exceeds_limit


@pytest.mark.asyncio
async def test_a_two_line_draft_ships_both_lines_and_records_the_floor() -> None:
    lines = [_words(60) + ".", _words(60) + "."]  # 48s against a 30s order
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": [{"text": t, "seconds": 24.0} for t in lines]})
    out = await ScriptDraftStage(client=fake).run({}, ctx=_brief_ctx())
    assert len(_draft_asks(fake)) == 1
    # The per-line fit tried to rewrite each line; the "model" answers with the
    # whole two-line script again, which is never closer, so both texts stand.
    assert len(_rewrite_asks(fake)) == 4
    assert [line.text for line in out.lines] == lines
    assert "action=sentence_trim_floored" in str(out.degradations)


def test_keep_lines_preserves_ids_and_order() -> None:
    draft = ScriptDraftOutput(
        lines=[ScriptLine(id=f"id{i}", text=f"t{i}.", seconds=float(i + 1)) for i in range(5)],
        total_seconds=15.0,
    )
    out = ScriptDraftStage._keep_lines(draft, [0, 2, 4])
    assert [line.id for line in out.lines] == ["id0", "id2", "id4"]
    assert out.total_seconds == pytest.approx(9.0)


def _prompt_ctx(*, grounding_text: str | None) -> RunContext:
    prior: dict[str, Any] = {"brief_intake": dict(_BRIEF)}
    if grounding_text is not None:
        prior["source_ingest"] = {"grounding_text": grounding_text}
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs=prior,
    )


def test_no_source_prompt_forbids_unsupported_facts() -> None:
    """Round 9: a generic film with no source invented a "London 14-day deposit rule".

    With no ``source_ingest`` content the prompt must forbid falsifiable specifics and
    must not ask for "concrete detail, a worked example" (an instruction to invent).
    """
    prompt = ScriptDraftStage(client=AsyncMock()).build_prompt(_prompt_ctx(grounding_text=None))
    assert NO_SOURCE_FACT_RULE in prompt
    assert "Do NOT state numbers, statistics" in prompt
    assert "laws, regulations, rules, deadlines" in prompt
    assert SOURCE_FACT_RULE not in prompt
    assert "worked example" not in prompt
    assert "concrete detail" not in prompt


def test_empty_source_ingest_counts_as_no_source() -> None:
    prompt = ScriptDraftStage(client=AsyncMock()).build_prompt(_prompt_ctx(grounding_text="  "))
    assert NO_SOURCE_FACT_RULE in prompt
    assert "SOURCE MATERIAL the operator provided" not in prompt


def test_sourced_prompt_restricts_facts_to_the_sources() -> None:
    prompt = ScriptDraftStage(client=AsyncMock()).build_prompt(
        _prompt_ctx(grounding_text="Deposits must be protected within 30 days.")
    )
    assert "Deposits must be protected within 30 days." in prompt
    assert SOURCE_FACT_RULE in prompt
    assert "only state facts" in prompt
    assert NO_SOURCE_FACT_RULE not in prompt
    # The filler-avoidance advice survives, scoped to the sources.
    assert "worked examples taken only from the source material" in prompt


def _rewrite_prompt(ctx: RunContext, *, measured: float, want: int) -> str:
    texts = ["Opening hook line here.", "Renters should know their deposit rights.", "Close."]
    return ScriptDraftStage(client=AsyncMock())._line_rewrite_prompt(
        ctx, texts, index=1, text=texts[1], measured=measured, budget=4.0, want=want
    )


def test_a_short_line_rewrite_without_source_carries_the_no_source_fact_rule() -> None:
    """Review of round 9: "ADD N words" with no fact rule re-invites invented specifics."""
    prompt = _rewrite_prompt(_prompt_ctx(grounding_text=None), measured=2.0, want=12)
    assert "ADD 6" in prompt
    assert NO_SOURCE_FACT_RULE in prompt
    assert SOURCE_FACT_RULE not in prompt
    assert "SOURCE MATERIAL the operator provided" not in prompt
    assert "only explanation or context" in prompt
    assert "no new facts" in prompt


def test_a_short_line_rewrite_with_source_carries_the_sources_and_their_fact_rule() -> None:
    prompt = _rewrite_prompt(
        _prompt_ctx(grounding_text="Deposits must be protected within 30 days."),
        measured=2.0,
        want=12,
    )
    assert "Deposits must be protected within 30 days." in prompt
    assert SOURCE_FACT_RULE in prompt
    assert NO_SOURCE_FACT_RULE not in prompt
    assert "no new facts" in prompt


def test_a_long_line_rewrite_still_carries_the_fact_rule_but_no_add_instruction() -> None:
    prompt = _rewrite_prompt(_prompt_ctx(grounding_text=None), measured=8.0, want=3)
    assert "CUT 3" in prompt
    assert NO_SOURCE_FACT_RULE in prompt
    assert "only explanation or context" not in prompt


@pytest.mark.asyncio
async def test_a_line_rewrite_is_one_model_call_with_no_retry_ladder() -> None:
    """A failed rewrite keeps the best take, so ``redraw_elements`` never retries."""
    fake = AsyncMock()
    fake.message.side_effect = ValueError("unparseable")
    stage = ScriptDraftStage(client=fake)
    stage._llm_retries = 2  # handle's ladder; the rewrite must ignore it
    with pytest.raises(Exception):  # noqa: B017 — stage_call wraps the exhaustion
        await stage.redraw_elements(_prompt_ctx(grounding_text=None), "Rewrite ONE line")
    assert fake.message.await_count == 1
