"""lesson_script: the ordered duration reaches the model as a COUNTABLE budget.

Measured on live artifacts 2026-09-10: a lesson ordered at 360s delivered 778.1s
(216%). The duration only ever reached the model as "Write a 6.0-minute teaching
script" — English an LLM cannot honour, because it has no model of how long its
own words take to say. These tests pin the word budget, its derivation from the
MEASURED voice rate, and the fallback when the rate cannot be measured.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

import src.services.stages.lesson.script as lesson_script
import src.services.stages.shared.speech_rate as speech_rate
from src.services.runs.run_context import RunContext
from src.services.stages.lesson.script import LessonScriptStage
from src.services.stages.shared.speech_rate import (
    resolve_natural_wpm_detail as _real_resolve_natural_wpm_detail,
)
from src.services.stages.shared.time_budget import DEFAULT_WPM_NATURAL
from tests.services.stages.lesson.rate_stubs import stub_speech_rate

MEASURED_KOKORO_WPM = 94.0


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _ctx(brief: dict | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"learning_objective": "Diagnose churn drivers.", **(brief or {})},
            "voice": {"descriptor": "calm"},
        },
    )


def _wm() -> AsyncMock:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"paragraphs": [{"text": "Open with the problem.", "seconds": 30.0}]}
    )
    return fake


async def _run(monkeypatch: pytest.MonkeyPatch, ctx: RunContext, wpm: float) -> tuple[str, Any]:
    """The FIRST prompt, which is the one the word budget is written into.

    ``fake.message.call_args`` is the LAST call, and the stage may now re-ask, so
    the budget sentence is read off ``call_args_list[0]`` — reading the last one
    would silently assert the CORRECTION block's arithmetic instead.
    """
    stub_speech_rate(monkeypatch, lesson_script, wpm)
    fake = _wm()
    out = await LessonScriptStage(client=fake).run({}, ctx=ctx)
    return fake.message.call_args_list[0].kwargs["message"], out


@pytest.mark.asyncio
async def test_prompt_states_the_word_budget_for_the_measured_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """6 minutes (360s) at the MEASURED 94 wpm ⇒ 564 words, a TARGET with a floor.

    Not a ceiling: "a HARD CEILING … write fewer" drew 57-120 words against a
    198-word budget in a live replay (run 01a158a6-0f2c, 2026-10-20).
    """
    prompt, _ = await _run(monkeypatch, _ctx({"duration_minutes": 6}), MEASURED_KOKORO_WPM)
    expected = round(360 * MEASURED_KOKORO_WPM / 60)
    assert expected == 564
    assert f"{expected} words" in prompt
    assert "HARD CEILING" not in prompt
    assert "write fewer" not in prompt
    assert "no fewer than 535 and no more than 593" in prompt
    # The ordered length survives in its original form too.
    assert "6.0-minute teaching script" in prompt


@pytest.mark.asyncio
async def test_a_slower_measured_voice_yields_a_materially_smaller_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same 6-minute order: 564 words at 94 wpm vs 900 at 150."""
    slow, _ = await _run(monkeypatch, _ctx({"duration_minutes": 6}), MEASURED_KOKORO_WPM)
    fast, _ = await _run(monkeypatch, _ctx({"duration_minutes": 6}), DEFAULT_WPM_NATURAL)
    slow_words = round(360 * MEASURED_KOKORO_WPM / 60)
    fast_words = round(360 * DEFAULT_WPM_NATURAL / 60)
    assert f"{slow_words} words" in slow
    assert f"{fast_words} words" in fast
    assert f"{fast_words} words" not in slow
    assert slow_words < fast_words * 0.7


@pytest.mark.asyncio
async def test_unmeasurable_voice_falls_back_to_exactly_the_default_wpm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The REAL fallback path: only the probe is stubbed, not the resolver.

    The autouse ``_hermetic_speech_rate`` fixture pins the Phase-A resolver, so it
    is restored to the real function here — the whole point of this test is that
    ``resolve_natural_wpm_detail``'s OWN fallback lands on exactly the default, and
    a stub answering the default would assert the stub.

    Restored ON ``speech_rate``, and from a reference captured AT IMPORT, because
    that is where the fixture now patches: the stage reaches the resolver through
    ``resolve_effective_wpm``, so re-reading the attribute here would hand back the
    fixture's own ``AsyncMock`` and this test would assert the stub it is trying to
    get rid of.
    """
    speech_rate.clear_measured_wpm_cache()
    monkeypatch.setattr(speech_rate, "resolve_natural_wpm_detail", _real_resolve_natural_wpm_detail)
    monkeypatch.setattr(speech_rate, "TTSClient", MagicMock())
    monkeypatch.setattr(speech_rate, "measure_natural_wpm", AsyncMock(return_value=None))
    fake = _wm()
    await LessonScriptStage(client=fake).run({}, ctx=_ctx({"duration_minutes": 6}))
    prompt = fake.message.call_args_list[0].kwargs["message"]
    assert f"{round(360 * DEFAULT_WPM_NATURAL / 60)} words" in prompt
    assert f"{DEFAULT_WPM_NATURAL:.0f} words per minute" in prompt


@pytest.mark.asyncio
async def test_duration_resolution_precedence_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """brief.duration_minutes, default 6 — and the budget uses the SAME number."""
    wpm = MEASURED_KOKORO_WPM
    ordered, _ = await _run(monkeypatch, _ctx({"duration_minutes": 2}), wpm)
    assert f"{round(120 * wpm / 60)} words" in ordered
    defaulted, _ = await _run(monkeypatch, _ctx(), wpm)
    assert f"{round(360 * wpm / 60)} words" in defaulted


@pytest.mark.asyncio
async def test_output_records_both_the_budget_and_the_actual_word_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recording only: an over-budget script is reported, never rejected."""
    stub_speech_rate(monkeypatch, lesson_script, MEASURED_KOKORO_WPM)
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "paragraphs": [
                {"text": "one two three four five", "seconds": 30.0},
                {"text": "six seven eight", "seconds": 20.0},
            ]
        }
    )
    out = await LessonScriptStage(client=fake).run({}, ctx=_ctx({"duration_minutes": 6}))
    assert out.word_budget == round(360 * MEASURED_KOKORO_WPM / 60)
    assert out.actual_words == 8
    assert len(out.paragraphs) == 2
