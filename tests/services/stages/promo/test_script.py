from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

import src.services.stages.promo.script as promo_script
from src.services.runs.run_context import RunContext
from src.services.stages.base import StageError
from src.services.stages.promo.script import (
    PromoScriptStage,
)
from src.services.stages.shared.time_budget import DEFAULT_WPM_NATURAL
from tests.services.stages.promo.rate_stubs import stub_measurement_at_order


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


@pytest.mark.asyncio
async def test_script_uses_insights_and_cta() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "lines": [
                {"text": "Stop the bleeding.", "seconds": 3.0},
                {"text": "Get the playbook.", "seconds": 2.5},
            ]
        }
    )
    stage = PromoScriptStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "Anti-churn",
            "brief": {"call_to_action": "Get the playbook.", "target_duration_seconds": 30},
            "voice": {"descriptor": "crisp"},
        },
        prior_run_outputs={
            "promo_analyze": {
                "pain_points": [{"id": "p1", "text": "high churn"}],
                "hook_candidates": [{"id": "h1", "text": "Stop the bleeding."}],
                "tone_hints": ["urgent"],
            },
        },
    )

    out = await stage.run({}, ctx=ctx)

    assert len(out.lines) == 2
    assert all(line.id for line in out.lines)
    # Not the 5.5 the model claimed for these two lines: `_finalize` persists the
    # MEASURED length of the shipped draw. Six words at the hermetic 150 wpm default
    # is 2.4s, and it agrees with what the run recorded on `narration_estimate`.
    assert out.total_seconds == pytest.approx(2.4)
    assert out.narration_estimate is not None
    assert out.total_seconds == pytest.approx(out.narration_estimate.estimate_seconds)
    _, kwargs = fake.message.call_args
    body = kwargs["message"]
    assert "Stop the bleeding." in body
    assert "Get the playbook." in body
    assert "high churn" in body


@pytest.mark.asyncio
async def test_script_grounds_on_instructions_and_capture_page_text() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": [{"text": "Get the playbook.", "seconds": 2.5}]})
    stage = PromoScriptStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "Anti-churn",
            "brief": {
                "call_to_action": "Get the playbook.",
                "target_duration_seconds": 30,
                "instructions": "Emphasise the one-click onboarding flow.",
            },
            "voice": {"descriptor": "crisp"},
        },
        prior_run_outputs={
            "promo_analyze": {
                "pain_points": [{"id": "p1", "text": "high churn"}],
                "hook_candidates": [{"id": "h1", "text": "Stop the bleeding."}],
                "tone_hints": ["urgent"],
            },
            "site_capture": {
                "clips": [],
                "page_text": "w2v turns manuscripts into narrated videos with one click.",
                "notes": [],
            },
        },
    )

    await stage.run({}, ctx=ctx)

    body = fake.message.call_args.kwargs["message"]
    assert "Emphasise the one-click onboarding flow." in body
    assert "w2v turns manuscripts into narrated videos" in body


@pytest.mark.asyncio
async def test_script_uses_deliverable_target_duration() -> None:
    """The FE sets the duration on the deliverable; the prompt must honour it
    (regression: it previously read only brief and defaulted everything to 30s)."""
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": [{"text": "Buy.", "seconds": 3.0}]})
    stage = PromoScriptStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "t",
            # No target_duration_seconds in brief — it lives on the deliverable.
            "brief": {"call_to_action": "Buy."},
            "deliverable": {"target_duration_seconds": 120},
            "voice": {"descriptor": "v"},
        },
        prior_run_outputs={
            "promo_analyze": {
                "pain_points": [{"id": "p1", "text": "p"}],
                "hook_candidates": [{"id": "h1", "text": "h"}],
                "tone_hints": [],
            },
        },
    )

    await stage.run({}, ctx=ctx)

    body = fake.message.call_args.kwargs["message"]
    assert "approximately 120 seconds" in body


@pytest.mark.asyncio
async def test_script_dialogue_mode_prompts_two_speakers_and_tags_lines() -> None:
    """brief.dialogue.enabled ⇒ the prompt asks for a two-hander and each emitted
    line keeps its speaker tag (so promo_narrate can cast two voices)."""
    fake = AsyncMock()
    # Regression: the LLM tends to tag EVERY line "a" (collapsing the two voices);
    # the stage must re-derive the speaker from each line's role (question = asker
    # "a", answer = answerer "b"), so the output still alternates two voices.
    fake.message.return_value = _resp(
        {
            "lines": [
                {"text": "So anyone can build this?", "seconds": 3.0, "speaker": "a"},
                {"text": "Exactly — watch this. Get the playbook.", "seconds": 3.0, "speaker": "a"},
            ]
        }
    )
    stage = PromoScriptStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "t",
            "brief": {
                "call_to_action": "Get the playbook.",
                "target_duration_seconds": 120,
                "dialogue": {"enabled": True, "personas": {"a": "Maya the skeptic"}},
            },
            "voice": {"descriptor": "conversational"},
        },
        prior_run_outputs={
            "promo_analyze": {
                "pain_points": [{"id": "p1", "text": "p"}],
                "hook_candidates": [{"id": "h1", "text": "So anyone can build this?"}],
                "tone_hints": [],
            },
        },
    )

    out = await stage.run({}, ctx=ctx)

    body = fake.message.call_args.kwargs["message"]
    assert "DIALOGUE" in body
    assert "Maya the skeptic" in body  # custom persona flows into the prompt
    # Natural-dialogue prompt: bans the repetitive "Look at this X" opener, asks
    # for contractions + interruptions, and keeps the screen tour order.
    assert "Look at this" in body  # named as the banned opener
    assert "contractions" in body.lower()
    assert "tour order" in body.lower()
    assert [line.speaker for line in out.lines] == ["a", "b"]


@pytest.mark.asyncio
async def test_script_single_narrator_prompt_pins_speaker_a() -> None:
    """Default (no dialogue flag): the prompt declares a single narrator so the
    schema-required speaker is inert."""
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": [{"text": "Buy.", "seconds": 3.0, "speaker": "a"}]})
    stage = PromoScriptStage(client=fake)

    await stage.run({}, ctx=_script_ctx())

    body = fake.message.call_args.kwargs["message"]
    assert "SINGLE narrator" in body
    assert "DIALOGUE" not in body


@pytest.mark.asyncio
async def test_script_raises_without_analyze_output() -> None:
    fake = AsyncMock()
    stage = PromoScriptStage(client=fake)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    with pytest.raises(RuntimeError, match="promo_analyze"):
        await stage.run({}, ctx=ctx)


@pytest.mark.asyncio
async def test_script_source_script_bypass_emits_exact_lines() -> None:
    """brief.source_script ⇒ no LLM call; each non-blank line is emitted verbatim
    and dialogue speakers are re-derived from role (two voices preserved)."""
    fake = AsyncMock()  # must NOT be called on the bypass path
    stage = PromoScriptStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "t",
            "brief": {
                "call_to_action": "Get the playbook.",
                "source_script": (
                    "So anyone can build this?\n"
                    "Exactly — watch this build sandbox.\n"
                    "And who approves it?\n"
                    "Governance does. Get the playbook."
                ),
                "dialogue": {"enabled": True},
            },
            "voice": {"descriptor": "conversational"},
        },
    )

    out = await stage.run({}, ctx=ctx)

    fake.message.assert_not_called()  # LLM bypassed entirely
    assert [line.text for line in out.lines] == [
        "So anyone can build this?",
        "Exactly — watch this build sandbox.",
        "And who approves it?",
        "Governance does. Get the playbook.",
    ]
    # Questions → "a", answers → "b" (the two voices survive the verbatim bypass).
    assert [line.speaker for line in out.lines] == ["a", "b", "a", "b"]


@pytest.mark.asyncio
async def test_script_source_script_verbatim_enforced() -> None:
    """A required verbatim phrase missing from the source script fails loud."""
    fake = AsyncMock()
    stage = PromoScriptStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "t",
            "brief": {"call_to_action": "cta", "source_script": "Line one.\nLine two."},
            "verbatim_items": [{"text": "this phrase is absent"}],
            "voice": {"descriptor": "v"},
        },
    )

    with pytest.raises(StageError):
        await stage.run({}, ctx=ctx)


@pytest.mark.asyncio
async def test_script_no_source_script_still_uses_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty/absent source_script ⇒ short_circuit returns None ⇒ normal LLM path.

    The estimate is pinned ON the 30s order so the duration loop steers ``ok`` and
    spends no correction: this test is about WHICH branch ran, and a re-ask it is
    not testing would turn its exact call count into noise.
    """
    stub_measurement_at_order(monkeypatch, promo_script, 30.0, DEFAULT_WPM_NATURAL)
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": [{"text": "Buy.", "seconds": 3.0}]})
    stage = PromoScriptStage(client=fake)

    out = await stage.run({}, ctx=_script_ctx())

    fake.message.assert_called_once()
    assert [line.text for line in out.lines] == ["Buy."]


def _script_ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "x",
            "brief": {"call_to_action": "Buy.", "target_duration_seconds": 30},
            "voice": {"descriptor": "x"},
        },
        prior_run_outputs={
            "promo_analyze": {
                "pain_points": [{"id": "p1", "text": "x"}],
                "hook_candidates": [{"id": "h1", "text": "hook"}],
                "tone_hints": [],
            },
        },
    )


@pytest.mark.asyncio
async def test_script_reprompts_when_element_missing_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLMScriptStageBase re-prompts when a line element is missing `text`
    (a transient shape deviation) instead of failing the run on first draw.

    The estimate is pinned ON the order (see the sibling above) so the ONLY thing
    consuming the ``side_effect`` queue is the shape-retry ladder this test pins.
    """
    stub_measurement_at_order(monkeypatch, promo_script, 30.0, DEFAULT_WPM_NATURAL)
    bad = {"lines": [{"seconds": 3.0}]}  # missing required `text`
    good = {"lines": [{"text": "Buy now.", "seconds": 3.0}]}
    fake = AsyncMock()
    fake.message = AsyncMock(side_effect=[_resp(bad), _resp(good)])
    stage = PromoScriptStage(client=fake)

    out = await stage.run({}, ctx=_script_ctx())

    assert fake.message.await_count == 2
    assert [line.text for line in out.lines] == ["Buy now."]


@pytest.mark.asyncio
async def test_script_fails_loud_after_retries_on_persistent_bad_shape() -> None:
    bad = {"lines": [{"seconds": 3.0}]}
    fake = AsyncMock()
    fake.message = AsyncMock(return_value=_resp(bad))
    stage = PromoScriptStage(client=fake)

    with pytest.raises(StageError):
        await stage.run({}, ctx=_script_ctx())
    assert fake.message.await_count == 2  # llm_retries=1 → 2 attempts, all bad
