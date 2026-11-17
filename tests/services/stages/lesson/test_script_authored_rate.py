"""lesson_script honours the AUTHORED narration rate, and spends nothing proving it.

``speech_rate.resolve_effective_wpm`` puts the operator's own
``direction.timing.wpm.natural`` above both the calibration passage and any
measurement of the drafted paragraphs — it "outranks every measurement; never a
fallback". This stage used to reach past that ladder and probe unconditionally,
so an authored rate could be accepted into the run config and then silently
overruled by whatever the voice happened to measure: two places deciding the rate
for one script, which is the exact defect the shared resolver exists to prevent.

No live scenario authors a rate today (0 of 42 in ``scripts/e2e_scenarios/``), so
these pin a correctness gap rather than a regression — which is precisely why they
have to be here: nothing in production would notice if this broke.
"""

from typing import Any, cast
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

import src.services.stages.lesson.script as lesson_script
import src.services.stages.shared.speech_rate as speech_rate
from src.clients.tts_client import TTSClient
from src.services.runs.run_context import RunContext
from src.services.stages.lesson.script import LessonScriptStage
from src.services.stages.shared.speech_rate import (
    measure_script_lines as _real_measure_script_lines,
)
from src.services.tts_storage import deterministic_clip_id

#: The operator's stated rate. Deliberately none of the other rungs' answers: the
#: directory's autouse fixture pins the Phase-A prior at 150.0 and ``_CountingVoice``
#: below delivers 120.0, so any assertion that passes here passed on the AUTHOR's
#: number and could not have passed on either of the others.
AUTHORED_WPM = 90.0
#: What the fake voice really delivers, if anybody asks it.
CONTENT_WPM = 120.0
ORDERED_SECONDS = 60.0
#: 60 ordered seconds at the authored rate: 60/60*90.
AUTHORED_BUDGET_WORDS = 90


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _words(count: int, *, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(count))


def _draw(word_counts: list[int], *, tag: str) -> WMResponse:
    return _resp(
        {
            "paragraphs": [
                {"text": _words(n, tag=f"{tag}{idx}_"), "seconds": n / 2.0}
                for idx, n in enumerate(word_counts)
            ]
        }
    )


def _ctx(*, authored: float | None = None) -> RunContext:
    """A one-minute lesson, optionally carrying the operator's stated rate."""
    config: dict[str, Any] = {
        "brief": {"learning_objective": "Diagnose churn drivers.", "duration_minutes": 1},
        "voice": {"descriptor": "calm", "voice_id": "af_heart"},
    }
    if authored is not None:
        config["direction"] = {"timing": {"wpm": {"natural": authored}}}
    return RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot=config)


def _wm(*replies: Any) -> Any:
    from unittest.mock import AsyncMock

    fake = AsyncMock()
    queue = list(replies)
    prompts: list[str] = []
    fake.prompts = prompts

    async def _message(**kwargs: Any) -> Any:
        prompts.append(str(kwargs.get("message", "")))
        if not queue:
            raise AssertionError("the stage asked for more draws than the test scripted")
        return queue.pop(0)

    fake.message.side_effect = _message
    return fake


class _CountingVoice:
    """A TTS backend that answers honestly and REMEMBERS every line it was asked for.

    The count is the assertion. A stubbed measurement would record that the stage
    took some code path; this records whether a synthesis was actually requested of
    a backend, which is the thing an authored rate is supposed to make unnecessary.
    """

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def synthesize(self, **kwargs: Any) -> Any:
        text = str(kwargs["text"])
        self.texts.append(text)
        return type("_Synth", (), {"duration_seconds": len(text.split()) / CONTENT_WPM * 60.0})()


def _arrange(monkeypatch: pytest.MonkeyPatch, stored: list[dict[str, Any]]) -> _CountingVoice:
    """Undo the directory's hermetic measurement stub; fake only STORAGE.

    ``conftest._hermetic_speech_rate`` replaces ``measure_script_lines`` wholesale,
    which would make "was anything synthesized?" unanswerable. The real function is
    restored (from a reference captured at import, before the fixture ran) and given
    a fake voice and a fake object store instead — so the only thing missing is the
    network.
    """
    monkeypatch.setattr(lesson_script, "measure_script_lines", _real_measure_script_lines)
    voice = _CountingVoice()

    async def _fake_store(client: Any, **kwargs: Any) -> Any:
        stored.append(dict(kwargs))
        result = await client.synthesize(text=kwargs["text"], voice=kwargs.get("voice"))
        return type(
            "_Clip",
            (),
            {
                "clip_id": deterministic_clip_id(
                    str(kwargs["text"]), kwargs.get("voice"), kwargs.get("speed")
                ),
                "artifact_url": "url",
                "seconds": result.duration_seconds,
                "speed": kwargs.get("speed"),
                "words": len(str(kwargs["text"]).split()),
            },
        )()

    monkeypatch.setattr(speech_rate, "synthesize_and_store_clip", _fake_store)
    return voice


def _as_client(fake: _CountingVoice) -> TTSClient:
    """Duck-typed, cast rather than subclassed — the same idiom as ``test_speech_rate``."""
    return cast("TTSClient", fake)


@pytest.fixture
def stored() -> list[dict[str, Any]]:
    return []


@pytest.mark.asyncio
async def test_an_authored_rate_is_used_verbatim_and_costs_no_synthesis(
    monkeypatch: pytest.MonkeyPatch, stored: list[dict[str, Any]]
) -> None:
    """An author who states a rate gets THAT rate — no probe, no measurement, no TTS.

    Three things, because three things could each be wrong on their own: the budget
    handed to the model is the authored one, the persisted record says the number
    came from the author (``wpm_basis``), and NOTHING was synthesized to arrive at
    it. The last is load-bearing: a measurement taken here could only produce a rate
    the stage is forbidden to use, at the price of a whole lesson of TTS against a
    launchd unit on mini2.
    """
    voice = _arrange(monkeypatch, stored)
    # 90 words is exactly the 60s ordered AT THE AUTHORED RATE, so a stage that
    # honours it ships on one LLM call. A stage that measures instead sees 90 words
    # at 120 wpm = 45.0s, calls it SHORT, and re-asks.
    fake = _wm(_draw([10] * 9, tag="a"))

    out = await LessonScriptStage(client=fake, tts=_as_client(voice)).run(
        {}, ctx=_ctx(authored=AUTHORED_WPM)
    )

    assert voice.texts == [], "an authored rate must not cost a single synthesis"
    assert stored == [], "and nothing was persisted either"
    assert fake.message.await_count == 1, "at the authored rate this draw is exactly in band"

    estimate = out.narration_estimate
    assert estimate is not None
    assert estimate.wpm_basis == "authored", "the record names the AUTHOR, not a probe"
    assert estimate.wpm == AUTHORED_WPM
    assert estimate.estimate_seconds == pytest.approx(ORDERED_SECONDS)
    assert estimate.verdict == "ok"

    # The ASK was written from the authored rate too — the budget and the verdict
    # must not disagree about which number this run is being run at.
    assert f"{AUTHORED_WPM:.0f} words per minute" in fake.prompts[0]
    assert f"{AUTHORED_BUDGET_WORDS} words" in fake.prompts[0]
    assert out.word_budget == AUTHORED_BUDGET_WORDS
    assert out.degradations == [], "declining to measure is not a fault"


@pytest.mark.asyncio
async def test_with_no_authored_rate_the_measurement_still_runs(
    monkeypatch: pytest.MonkeyPatch, stored: list[dict[str, Any]]
) -> None:
    """THE CONTROL — identical fixture and draw, the authored key simply absent.

    Without this, an early return that fired unconditionally would look exactly
    like a correct one: every assertion in the test above would still pass.
    """
    voice = _arrange(monkeypatch, stored)
    fake = _wm(_draw([10] * 9, tag="a"), _draw([10] * 12, tag="b"))

    out = await LessonScriptStage(client=fake, tts=_as_client(voice)).run({}, ctx=_ctx())

    assert len(voice.texts) == 21, "9 paragraphs measured, then 12 of the re-asked draft"
    estimate = out.narration_estimate
    assert estimate is not None
    assert estimate.wpm_basis == "measured_all", "measured off this script's own paragraphs"
    assert estimate.wpm == pytest.approx(CONTENT_WPM)


@pytest.mark.asyncio
async def test_the_prior_is_resolved_without_ever_sampling_a_script(
    monkeypatch: pytest.MonkeyPatch, stored: list[dict[str, Any]]
) -> None:
    """``texts=()`` in the PRIOR call is a cost contract, not a formality.

    ``resolve_effective_wpm`` has three rungs, and the ``texts`` argument alone
    decides between two of them: empty means Phase A (the calibration passage,
    already cached) and non-empty means Phase B, which SYNTHESIZES a sample of the
    lines it was handed. This stage measures every paragraph itself a few lines
    later (Phase C), so a Phase-B sample here is not a second opinion, it is the
    same audio paid for twice — and it is paid before the first draw exists, i.e.
    on text that is not even the script.

    It survived every other test in this directory: passing the real texts here
    changes no output at all, it only spends. What it does change is wall clock —
    the measured cost of that mutant was 0.21s -> 23m30s. So the assertion is the
    argument AND the consequence: Phase B is made to explode if anything reaches
    it, which is the observable the argument is a proxy for.
    """
    voice = _arrange(monkeypatch, stored)
    seen: list[tuple[str, ...]] = []
    synthesized_by_the_time_the_prior_landed: list[int] = []
    real_resolve = lesson_script.resolve_effective_wpm

    async def _spy(ctx: Any, *, texts: Any, **kwargs: Any) -> Any:
        seen.append(tuple(texts))
        result = await real_resolve(ctx, texts=texts, **kwargs)
        synthesized_by_the_time_the_prior_landed.append(len(voice.texts))
        return result

    async def _phase_b_is_forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(
            "the prior must never sample a script — this stage measures every paragraph itself"
        )

    monkeypatch.setattr(lesson_script, "resolve_effective_wpm", _spy)
    monkeypatch.setattr(speech_rate, "resolve_script_sample_wpm", _phase_b_is_forbidden)

    fake = _wm(_draw([10] * 9, tag="a"), _draw([10] * 12, tag="b"))
    out = await LessonScriptStage(client=fake, tts=_as_client(voice)).run({}, ctx=_ctx())

    assert seen == [()], f"the prior is resolved once, on NO texts; got {seen}"
    assert synthesized_by_the_time_the_prior_landed == [0], (
        "resolving the prior cost a synthesis — that is Phase B running"
    )
    # The run still completed normally, so this is not a test that passes by
    # preventing the stage from working.
    assert out.narration_estimate is not None
    assert out.narration_estimate.wpm_basis == "measured_all"


@pytest.mark.asyncio
async def test_an_authored_rate_survives_a_re_ask_and_still_costs_no_synthesis(
    monkeypatch: pytest.MonkeyPatch, stored: list[dict[str, Any]]
) -> None:
    """The RE-MEASURE declines too — an authored rate is not a once-per-run fact.

    ``_reconcile`` measures the first draw, and measures AGAIN after a re-ask,
    because a re-asked draft is new text that must be judged at its own seconds.
    Both calls land in ``_measure``, and the authored early return has to fire in
    both: a guard that fired only the first time would spend a whole second draft
    of TTS to compute a rate the stage is then forbidden to use, and would report
    that measured rate as the run's.

    The first draw is deliberately +50% AT THE AUTHORED RATE so the steer is
    obliged to re-ask; ``await_count == 2`` is what makes the rest non-vacuous.
    """
    voice = _arrange(monkeypatch, stored)
    # 135 words = 90.0s at the authored 90 wpm against a 60s order (+50%) -> re-ask;
    # then 90 words = exactly 60.0s -> ok.
    fake = _wm(_draw([15] * 9, tag="a"), _draw([10] * 9, tag="b"))

    out = await LessonScriptStage(client=fake, tts=_as_client(voice)).run(
        {}, ctx=_ctx(authored=AUTHORED_WPM)
    )

    assert fake.message.await_count == 2, "the first draw must actually have been re-asked"
    assert voice.texts == [], "neither the first draft nor the re-asked one may be synthesized"
    assert stored == [], "and nothing was persisted for either"

    estimate = out.narration_estimate
    assert estimate is not None
    assert estimate.reasked is True
    assert estimate.wpm_basis == "authored", "the re-measure must not re-open the rate question"
    assert estimate.wpm == AUTHORED_WPM
    assert estimate.estimate_seconds == pytest.approx(ORDERED_SECONDS)


@pytest.mark.asyncio
async def test_an_authored_run_claims_no_measured_paragraphs(
    monkeypatch: pytest.MonkeyPatch, stored: list[dict[str, Any]]
) -> None:
    """An ESTIMATE must never be recorded as a MEASUREMENT.

    ``_authored_measurement`` hand-builds a ``ScriptMeasurement``, and every flag on
    it is an assertion about audio: ``measured`` says "this paragraph's seconds came
    off a real clip". On this path nothing was synthesized at all, so every flag on a
    spoken paragraph is false and ``measured_lines`` — the number that reaches the
    persisted ``NarrationEstimate``, and thence anyone asking after the fact whether
    this run's durations were real — is 0.

    Getting this wrong is silent by construction: the seconds would be identical, and
    the run would simply claim to know something it does not.
    """
    voice = _arrange(monkeypatch, stored)
    fake = _wm(_draw([10] * 9, tag="a"))

    out = await LessonScriptStage(client=fake, tts=_as_client(voice)).run(
        {}, ctx=_ctx(authored=AUTHORED_WPM)
    )

    assert voice.texts == []
    estimate = out.narration_estimate
    assert estimate is not None
    assert estimate.measured_lines == 0, (
        "an authored run measured nothing; claiming measured paragraphs would make the "
        "estimate indistinguishable from a full-script measurement"
    )
