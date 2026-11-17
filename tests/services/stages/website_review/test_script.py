from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageError
from src.services.stages.website_review.script import WebsiteReviewScriptStage


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {"duration_seconds": 60}, "voice": {"descriptor": "warm"}},
        prior_run_outputs={
            "website_critique": {
                "url": "https://acme.test",
                "overall_score": 70.0,
                "summary": "Slow and unclear.",
                "findings": [
                    {
                        "id": "F1",
                        "category": "performance",
                        "severity": "high",
                        "title": "Slow LCP",
                        "recommendation": "Optimize images.",
                    }
                ],
            },
            "website_redesign": {"style": "modern", "visuals": []},
        },
    )


@pytest.mark.asyncio
async def test_output_schema_names_phase_and_critique_item_id() -> None:
    # Regression: WM only emits schema-named props; critique_item_id AND the
    # two-act phase must be declared in the element schema or they are dropped.
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"lines": [{"text": "hi", "seconds": 2.0, "phase": "before"}]}
    )
    await WebsiteReviewScriptStage(client=fake).run({}, ctx=_ctx())
    _, kwargs = fake.message.call_args
    props = kwargs["output_schema"]["properties"]["lines"]["items"]["properties"]
    assert {"text", "seconds", "critique_item_id", "phase", "speaker"} <= set(props)


@pytest.mark.asyncio
async def test_prompt_requests_two_act_phases() -> None:
    stage = WebsiteReviewScriptStage(client=AsyncMock())
    prompt = stage.build_prompt(_ctx())
    assert "before" in prompt and "after" in prompt
    assert "redesign" in prompt.lower()


@pytest.mark.asyncio
async def test_prompt_is_two_hander_only_when_dialogue_is_enabled() -> None:
    # Dialogue is OFF unless brief.dialogue.enabled — the SAME gate record casts on,
    # so a two-speaker script is never read by one voice (prod run 01a0a054).
    assert "SINGLE narrator" in WebsiteReviewScriptStage(client=AsyncMock()).build_prompt(_ctx())
    ctx = _ctx()
    ctx.config_snapshot["brief"]["dialogue"] = {"enabled": True}
    prompt = WebsiteReviewScriptStage(client=AsyncMock()).build_prompt(ctx)
    assert 'speaker "a"' in prompt and 'speaker "b"' in prompt
    assert '"speaker":"a"|"b"' in prompt


@pytest.mark.asyncio
async def test_prompt_grounds_in_source_ingest_when_present() -> None:
    # When the operator grounded the review in extra sources, the script prompt
    # carries the SOURCE MATERIAL block from source_ingest.
    ctx = _ctx()
    ctx.prior_run_outputs["source_ingest"] = {"grounding_text": "Acme ships in 30 countries."}
    prompt = WebsiteReviewScriptStage(client=AsyncMock()).build_prompt(ctx)
    assert "SOURCE MATERIAL" in prompt
    assert "Acme ships in 30 countries." in prompt


@pytest.mark.asyncio
async def test_prompt_unchanged_without_source_ingest() -> None:
    # Plain-run-safe: no source_ingest output → no grounding block appended.
    prompt = WebsiteReviewScriptStage(client=AsyncMock()).build_prompt(_ctx())
    assert "SOURCE MATERIAL" not in prompt


@pytest.mark.asyncio
async def test_speaker_derived_from_question_role() -> None:
    # The LLM tagged every line "a" (the shipped gotcha); build_output must derive
    # the speaker from the line role so the two voices don't collapse to one.
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "lines": [
                {
                    "text": "Why is this page so slow?",
                    "seconds": 4,
                    "phase": "before",
                    "speaker": "a",
                },
                {
                    "text": "The hero images are huge.",
                    "seconds": 4,
                    "phase": "before",
                    "speaker": "a",
                },
                {
                    "text": "Did the redesign fix it?",
                    "seconds": 4,
                    "phase": "after",
                    "speaker": "a",
                },
            ]
        }
    )
    ctx = _ctx()
    ctx.config_snapshot["brief"]["dialogue"] = {"enabled": True}
    out = await WebsiteReviewScriptStage(client=fake).run({}, ctx=ctx)
    assert [line.speaker for line in out.lines] == ["a", "b", "a"]


@pytest.mark.asyncio
async def test_speaker_all_a_when_dialogue_disabled() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"lines": [{"text": "Slow?", "seconds": 4, "phase": "before", "speaker": "a"}]}
    )
    ctx = _ctx()
    ctx.config_snapshot["brief"]["dialogue"] = {"enabled": False}
    out = await WebsiteReviewScriptStage(client=fake).run({}, ctx=ctx)
    assert [line.speaker for line in out.lines] == ["a"]


@pytest.mark.asyncio
async def test_phase_kept_when_llm_emits_monotonic_arc() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "lines": [
                {"text": "Old page is slow.", "seconds": 5, "phase": "before"},
                {"text": "And cluttered.", "seconds": 5, "phase": "before"},
                {"text": "Here's the redesign.", "seconds": 5, "phase": "after"},
            ]
        }
    )
    out = await WebsiteReviewScriptStage(client=fake).run({}, ctx=_ctx())
    assert [line.phase for line in out.lines] == ["before", "before", "after"]


@pytest.mark.asyncio
async def test_phase_derived_when_llm_interleaves() -> None:
    # The LLM ignored the arc and tagged everything "before" (or interleaved);
    # the deterministic fallback forces a clean ~60/40 before→after split.
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "lines": [
                {"text": "Line 1.", "seconds": 4, "phase": "before"},
                {"text": "Line 2.", "seconds": 4, "phase": "before"},
                {"text": "Line 3.", "seconds": 4, "phase": "before"},
                {"text": "Line 4.", "seconds": 4, "phase": "before"},
                {"text": "Line 5.", "seconds": 4, "phase": "before"},
            ]
        }
    )
    out = await WebsiteReviewScriptStage(client=fake).run({}, ctx=_ctx())
    phases = [line.phase for line in out.lines]
    assert phases == ["before", "before", "before", "after", "after"]


@pytest.mark.asyncio
async def test_phase_boundary_at_redesign_reveal_not_llm_tags() -> None:
    # Regression: the LLM emitted a MONOTONIC arc but tagged the reveal line (and
    # the reaction to it) "before", so the visual switch lagged the narration by
    # ~2 lines (~15s). The before→after boundary must land on the line that
    # PRESENTS the redesign, derived from content — not the LLM's phase enum.
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "lines": [
                {"text": "The current site is slow.", "seconds": 5, "phase": "before"},
                {
                    "text": "Okay, enough critique. Show me the redesign.",
                    "seconds": 4,
                    "phase": "before",
                },
                {
                    "text": "Here is the new modern redesign we built.",
                    "seconds": 5,
                    "phase": "before",
                },
                {"text": "Wow, the cards actually breathe now.", "seconds": 5, "phase": "before"},
                {"text": "Yes, we increased the whitespace.", "seconds": 5, "phase": "after"},
            ]
        }
    )
    out = await WebsiteReviewScriptStage(client=fake).run({}, ctx=_ctx())
    phases = [line.phase for line in out.lines]
    # L1 critique + L2 "show me" request → before; L3 reveal onward → after.
    assert phases == ["before", "before", "after", "after", "after"]


@pytest.mark.asyncio
async def test_script_parses_lines() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "lines": [
                {"text": "This site loads slowly.", "seconds": 6, "critique_item_id": "F1"},
                {
                    "text": "The redesign fixes that.",
                    "seconds": 5,
                    "critique_item_id": None,
                },
            ]
        }
    )
    stage = WebsiteReviewScriptStage(client=fake)

    out = await stage.run({}, ctx=_ctx())

    assert len(out.lines) == 2
    assert out.lines[0].critique_item_id == "F1"
    assert out.total_seconds == 11.0
    assert all(line.id for line in out.lines)


@pytest.mark.asyncio
async def test_script_raises_on_malformed_output() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(None)
    stage = WebsiteReviewScriptStage(client=fake)

    with pytest.raises(StageError):
        await stage.run({}, ctx=_ctx())


@pytest.mark.asyncio
async def test_script_raises_without_redesign() -> None:
    # The redesign is the video's visual spine, so it's required; the critique is
    # optional (a redesign-only-plus-video run has no findings).
    fake = AsyncMock()
    stage = WebsiteReviewScriptStage(client=fake)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    with pytest.raises(RuntimeError, match="website_redesign"):
        await stage.run({}, ctx=ctx)


@pytest.mark.asyncio
async def test_the_model_is_told_not_to_think_before_it_answers() -> None:
    """The shared script primitive MUST send ``enable_reasoning=False``, and this
    is a fence, not a preference.

    ``website_review_script`` died in production on 2026-08-20 with
    ``StageTimeoutError: stage 'website_review_script': all 2 attempts timed out
    (per-attempt budget 950.0s)``, terminal state ``failed`` — a real product
    failure, not a harness deadline. It is also already on the token-limit
    failure ledger in ``llm_budget.py`` (N=6000, 2026-08-19 06:24Z).

    Measured through worker-management on the production path (Job
    ``wmprobe-segsel``, 2026-08-20) with the sibling stage's real prompt and
    schema at ``max_tokens=8000``: with reasoning ON the call ran to the stage's
    own 950 s deadline and returned nothing — twice, under BOTH candidate output
    schemas. With reasoning OFF the same call answered in 34-67 s carrying real
    content, every time. Dropping the keyword degrades SILENTLY: every unit test
    in this file passes without it, because a stubbed client answers instantly
    either way.

    The keyword lives on ``primitives/script_stage.py``, the ONE WM call site all
    seven concrete script stages inherit, so this assertion has teeth for every
    one of them; ``tests/services/stages/primitives/test_script_stage_token_budget.py::
    test_the_roster_covers_every_concrete_subclass`` is the vacuity guard that a
    NEW script stage cannot join the primitive unnoticed.

    ``w2v_common`` >= 0.61.0 is a hard requirement — worker-management's
    ``MessageRequest`` is ``extra="forbid"``, so an older client cannot send the
    field and a WM without it would 422 the call outright.
    """
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"lines": [{"text": "hi", "seconds": 2.0, "phase": "before"}]}
    )

    await WebsiteReviewScriptStage(client=fake).run({}, ctx=_ctx())

    assert fake.message.await_args.kwargs["enable_reasoning"] is False


def test_prompt_pins_the_script_to_the_narrators_language() -> None:
    """Prod 01a09f82: Dutch findings → Dutch script → read by an English-only voice.

    kokoro has no Dutch voice and the backend phonemizes every request as en-us, so
    the script must be written in English regardless of the site's language.
    """
    ctx = _ctx()
    ctx.prior_run_outputs["website_critique"]["findings"] = [
        {"id": "F1", "title": "Onduidelijke navigatie", "recommendation": "Vereenvoudig het menu."}
    ]
    prompt = WebsiteReviewScriptStage(client=None).build_prompt(ctx)
    assert "write EVERY line in English" in prompt
