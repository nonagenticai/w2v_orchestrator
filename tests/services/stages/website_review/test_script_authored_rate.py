"""website_review_script honours the AUTHORED narration rate, and spends nothing on it.

``speech_rate.resolve_effective_wpm`` puts the operator's own
``direction.timing.wpm.natural`` above both the calibration passage and any
measurement of the drafted lines — it "outranks every measurement; never a
fallback". This stage used to reach past that ladder and probe unconditionally,
so an authored rate could be accepted into the run config and then silently
overruled by whatever the cast happened to measure: two places deciding the rate
for one script, which is the exact defect the shared resolver exists to prevent.

No live scenario authors a rate today (0 of 42 in ``scripts/e2e_scenarios/``), so
these pin a correctness gap rather than a regression — which is precisely why they
have to be here: nothing in production would notice if this broke.
"""

from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

import src.services.stages.shared.speech_rate as speech_rate
import src.services.stages.website_review.script as website_review_script
from src.clients.tts_client import TTSClient
from src.services.runs.run_context import RunContext
from src.services.stages.shared.speech_rate import (
    measure_script_lines as _real_measure_script_lines,
)
from src.services.stages.website_review.script import WebsiteReviewScriptStage
from src.services.tts_storage import deterministic_clip_id

#: The operator's stated rate. Deliberately none of the other rungs' answers: the
#: directory's autouse fixture pins the Phase-A prior at 150.0 and ``_CountingVoice``
#: below delivers 120.0, so any assertion that passes here passed on the AUTHOR's
#: number and could not have passed on either of the others.
AUTHORED_WPM = 90.0
#: What the fake voice really delivers, if anybody asks it.
CONTENT_WPM = 120.0
ORDERED_SECONDS = 60
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
            "lines": [
                {"text": _words(n, tag=f"{tag}{idx}_"), "seconds": n / 2.0, "phase": "before"}
                for idx, n in enumerate(word_counts)
            ]
        }
    )


def _ctx(*, authored: float | None = None) -> RunContext:
    """A 60-second review, optionally carrying the operator's stated rate."""
    config: dict[str, Any] = {
        "brief": {"duration_seconds": ORDERED_SECONDS},
        "voice": {"descriptor": "warm", "voice_id": "af_heart"},
    }
    if authored is not None:
        config["direction"] = {"timing": {"wpm": {"natural": authored}}}
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=config,
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


def _wm(*replies: Any) -> AsyncMock:
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
    monkeypatch.setattr(website_review_script, "measure_script_lines", _real_measure_script_lines)
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
async def test_an_authored_run_claims_no_measured_lines(
    monkeypatch: pytest.MonkeyPatch, stored: list[dict[str, Any]]
) -> None:
    """An ESTIMATE must never be recorded as a MEASUREMENT.

    ``_authored_measurement`` hand-builds a ``ScriptMeasurement``, and every flag on
    it is an assertion about audio: ``measured`` says "this line's seconds came off a
    real clip". On this path nothing was synthesized at all, so every flag on a
    spoken line is false and ``measured_lines`` — the number that reaches the
    persisted ``NarrationEstimate``, and thence anyone asking after the fact whether
    this run's durations were real — is 0.

    Getting this wrong is silent by construction: the seconds would be identical, and
    the run would simply claim to know something it does not.
    """
    voice = _arrange(monkeypatch, stored)
    fake = _wm(_draw([10] * 9, tag="a"))

    out = await WebsiteReviewScriptStage(client=fake, tts=_as_client(voice)).run(
        {}, ctx=_ctx(authored=AUTHORED_WPM)
    )

    assert voice.texts == []
    estimate = out.narration_estimate
    assert estimate is not None
    assert estimate.measured_lines == 0, (
        "an authored run measured nothing; claiming measured lines would make the "
        "estimate indistinguishable from a full-script measurement"
    )
