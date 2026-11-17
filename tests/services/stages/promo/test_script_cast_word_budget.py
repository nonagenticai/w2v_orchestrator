"""promo_script budgets against the voices that will ACTUALLY speak the promo.

``promo_narrate`` casts a dialogue promo to TWO kokoro voices, and they do not
talk at the same speed — the live default pair measures ``af_heart`` at 175.24
wpm against ``am_michael`` at 156.60. The word budget in the prompt was taken at
``config.voice.voice_id`` alone, i.e. at a voice that speaks half the script at
most. On the sibling ``lesson`` stage that exact gap turned a +23.07% overrun
into a +7.6% estimate, the verdict came back ``ok``, and the trim machinery was
never reached (run ``01a0918c-4419-7299-8ccc-7289c674d7f9``).

This is the PRIOR — the rate the FIRST draft's prompt budget is written from, and
the only rate on the run that may be one. Everything after it (the re-ask target,
the verdict, the trim) runs on the rate measured off the drafted lines; see
``test_script_duration_reconciliation.py``. These tests pin the prior's number,
the untouched single-narrator path, the honest fallback, and the fact that the
verbatim/shotlist bypass still pays for no probe at all.

Rates here are 120 / 60 wpm rather than the live pair: both sit inside
``narration_budget``'s [60, 300] usable band, and a 2x spread keeps every
expectation exact arithmetic. The DIRECTION is the live one — speaker "b" is the
slower voice.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

import src.services.stages.promo.script as promo_script
import src.services.stages.shared.speech_rate as speech_rate
from src.services.runs.run_context import RunContext
from src.services.stages.promo.script import PromoScriptStage
from src.services.stages.shared.degradation import SPEECH_RATE_UNUSABLE
from src.services.stages.verbatim import seconds_for

#: The configured voice, and speaker "a" under promo_narrate's _DEFAULT_CAST.
FAST_VOICE = "af_heart"
#: Speaker "b" under that cast — the voice the budget never asked about.
SLOW_VOICE = "am_michael"

ORDERED_SECONDS = 120
#: 120s at the configured voice's 120 wpm.
SINGLE_VOICE_WORDS = 240
#: 120s at the cast blend of 120 and 60 wpm: 2 words take 0.5 + 1.0 = 1.5s, i.e.
#: 80 wpm, so the order affords 160 words — a THIRD less than the single-rate
#: budget, and the difference between a promo that fits and one that runs over.
CAST_WORDS = 160


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _ctx(*, brief: dict | None = None, voice: dict | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "Anti-churn",
            "brief": {
                "call_to_action": "Get the playbook.",
                "target_duration_seconds": ORDERED_SECONDS,
                "dialogue": {"enabled": True},
                **(brief or {}),
            },
            "voice": {"descriptor": "crisp", "voice_id": FAST_VOICE, **(voice or {})},
        },
        prior_run_outputs={
            "promo_analyze": {
                "pain_points": [{"id": "p1", "text": "high churn"}],
                "hook_candidates": [{"id": "h1", "text": "Stop the bleeding."}],
                "tone_hints": ["urgent"],
            },
        },
    )


def _wm() -> AsyncMock:
    fake = AsyncMock()
    fake.message.return_value = _resp({"lines": [{"text": "Does it stop churn?", "seconds": 3.0}]})
    return fake


def _arm_cast(monkeypatch: pytest.MonkeyPatch, rates: dict[str | None, float | None]) -> AsyncMock:
    """Answer the TTS calibration probe per VOICE; return the probe itself.

    Stubs ``measure_natural_wpm`` — the bottom of the stack — rather than
    ``resolve_cast_wpm``, so the memoisation, the per-voice weighting and the
    unmeasurable-voice fallback are all the shipped ones. Both names the autouse
    hermetic fixture replaces are restored to the REAL ones for the same reason;
    only the full-script measurement (a LATER phase, and not what these tests are
    about) stays stubbed, so no test here reaches tts.test.
    """
    monkeypatch.setattr(
        promo_script, "resolve_natural_wpm_detail", speech_rate.resolve_natural_wpm_detail
    )
    monkeypatch.setattr(promo_script, "resolve_cast_wpm", speech_rate.resolve_cast_wpm)
    speech_rate.clear_measured_wpm_cache()
    monkeypatch.setattr(speech_rate, "TTSClient", MagicMock())
    probe = AsyncMock(side_effect=lambda _client, *, voice, provider=None: rates.get(voice))
    monkeypatch.setattr(speech_rate, "measure_natural_wpm", probe)
    # The rates above are written as the narration prior; the passage->narration
    # ratio is pinned on its own (test_speech_rate_narration_prior.py).
    monkeypatch.setattr(speech_rate, "PASSAGE_TO_NARRATION_RATIO", 1.0)
    return probe


@pytest.mark.asyncio
async def test_a_two_hander_is_budgeted_at_the_cast_rate_not_the_configured_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE defect: the ceiling was set by a voice that speaks half the promo.

    120s at the configured 120 wpm affords 240 words; at the 80 wpm the cast
    actually delivers it affords 160. The prompt must carry the second number —
    the first one buys a promo a third longer than it was ordered.
    """
    probe = _arm_cast(monkeypatch, {FAST_VOICE: 120.0, SLOW_VOICE: 60.0})
    fake = _wm()
    await PromoScriptStage(client=fake).run({}, ctx=_ctx())
    prompt = fake.message.call_args_list[0].kwargs["message"]

    assert "80 words per minute" in prompt, "the blended rate reached the model"
    assert f"{CAST_WORDS} words in total" in prompt
    assert f"{SINGLE_VOICE_WORDS} words" not in prompt, "the single-voice ceiling is gone"
    # A set, not a list: ``measure_natural_wpm`` is stubbed here, so the real
    # per-``(provider, voice)`` memoisation underneath it is stubbed out too and
    # the configured voice is asked twice. WHICH voices were consulted is the
    # contract; how many times the un-memoised stub saw them is not.
    assert {call.kwargs["voice"] for call in probe.await_args_list} == {
        FAST_VOICE,
        SLOW_VOICE,
    }, "both cast voices were measured"


@pytest.mark.asyncio
async def test_a_single_narrator_promo_is_byte_identical_and_probes_one_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dialogue OFF ⇒ promo_narrate speaks one voice ⇒ the budget must too.

    ``config.voice.cast`` is populated here on purpose: the cast is only ever
    applied behind ``brief.dialogue.enabled``, so a budget that read the cast map
    directly — rather than through the ``voice_for_speaker`` rule narrate uses —
    would tighten this prompt by a third for voices that never speak.
    """
    probe = _arm_cast(monkeypatch, {FAST_VOICE: 120.0, SLOW_VOICE: 60.0})
    ctx = _ctx(
        brief={"dialogue": {"enabled": False}},
        voice={"cast": {"a": FAST_VOICE, "b": SLOW_VOICE}},
    )
    fake = _wm()
    out = await PromoScriptStage(client=fake).run({}, ctx=ctx)
    prompt = fake.message.call_args_list[0].kwargs["message"]

    assert "120 words per minute" in prompt, "the configured voice's own rate"
    assert f"{SINGLE_VOICE_WORDS} words in total" in prompt
    assert out.word_budget == SINGLE_VOICE_WORDS
    probed = [call.kwargs["voice"] for call in probe.await_args_list]
    assert probed == [FAST_VOICE], f"no second voice was probed, got {probed}"
    # What this test is about is the RATE: one voice was probed and nothing about
    # it was substituted. The shared ``_wm()`` draw is a single stub line, so it
    # is ~4 words against a 120s order and the duration loop legitimately reports
    # it as short (``test_script_duration_reconciliation.py`` owns that fact);
    # asserting a globally empty list here would pin the ABSENCE of an unrelated
    # signal.
    assert [d for d in out.degradations if d["code"] == SPEECH_RATE_UNUSABLE] == []


@pytest.mark.asyncio
async def test_an_unprobeable_cast_voice_is_named_not_silently_substituted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A voice whose probe does not answer contributes at the default AND says so.

    The budget lands back on 120 wpm — byte-identical to the behaviour before a
    cast was ever consulted — but the run now carries the reason, naming the
    voice. Without it, a promo budgeted blind is indistinguishable from one
    budgeted at a measured rate.
    """
    _arm_cast(monkeypatch, {FAST_VOICE: 120.0, SLOW_VOICE: None})
    fake = _wm()
    out = await PromoScriptStage(client=fake).run({}, ctx=_ctx())
    prompt = fake.message.call_args_list[0].kwargs["message"]

    assert f"{SINGLE_VOICE_WORDS} words in total" in prompt, "fell back, did not invent"
    rate = [d for d in out.degradations if d["code"] == SPEECH_RATE_UNUSABLE]
    assert len(rate) == 1, f"recorded exactly once, got {out.degradations}"
    assert SLOW_VOICE in rate[0]["detail"], "and the voice is NAMED"


@pytest.mark.asyncio
async def test_an_authored_shotlist_pays_for_no_probe_and_keeps_its_own_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bypass builds no prompt, so it must buy no budget — and no probe.

    promo is the duration-sensitive type: an authored shotlist IS the picture
    length, and ``short_circuit`` returns the operator's own ``seconds``
    untouched. Nothing in the cast-rate work may reach this path.
    """
    probe = _arm_cast(monkeypatch, {FAST_VOICE: 120.0, SLOW_VOICE: 60.0})
    narrations = ["First screen, here it is.", "Second screen, and what does it cost?"]
    ctx = _ctx(
        brief={
            "shotlist": [
                {"url": f"https://acme.test/{i}", "narration": text}
                for i, text in enumerate(narrations)
            ]
        }
    )
    fake = _wm()
    out = await PromoScriptStage(client=fake).run({}, ctx=ctx)

    assert fake.message.await_count == 0, "the bypass never asks the model"
    assert probe.await_count == 0, "and never pays for a calibration probe"
    assert [line.text for line in out.lines] == narrations, "the operator's words, verbatim"
    # ``seconds_for`` is the shotlist's OWN per-shot timing rule. The picture is
    # cut against these numbers, so the cast-rate work must leave them alone.
    assert [line.seconds for line in out.lines] == [seconds_for(t) for t in narrations]
    assert out.total_seconds == sum(seconds_for(t) for t in narrations)
    assert out.degradations == []
