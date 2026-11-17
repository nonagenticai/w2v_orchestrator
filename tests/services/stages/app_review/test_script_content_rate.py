"""app_review_script budgets against the rate ITS OWN LINES deliver at.

The root cause of every duration miss in this pipeline, measured on the live backend
from inside the orchestrator pod (durations byte-identical to the first clips of live
app_review run ``01a091bc-eae4-724d-b278-7ee323344bf4``)::

    CALIBRATION_TEXT :  65w  22.25s  ->  175.2 wpm
    real prod line   :  17w   8.37s  ->  121.9 wpm
    real prod line   :  12w   6.62s  ->  108.8 wpm
    real prod line   :  18w   8.62s  ->  125.3 wpm
    PRODUCTION       :  47w  23.61s  ->  119.4 wpm

The fixed calibration passage synthesizes ~47% FASTER than real narration, because real
lines carry numerals, abbreviations, technical nouns and heavy comma structure. So the
stage's estimate came in far UNDER the delivery, the verdict read ``ok`` (or, worse,
``short``), and the trim machinery that would have honoured the order was never
reached. That is the same failure shape as the live lesson cast-mode miss: the trim was
never defective, it was never REACHED.

These tests pin the correction end to end: measuring the drafted lines instead of a
fixed passage lowers the rate, the verdict flips, and the re-ask/trim loop absorbs it.

⚠️ The four-line SAMPLE these tests were written against is gone — the stage now
measures EVERY line (``speech_rate.measure_script_lines``), so the estimate is the
film's real length rather than a rate times a word count. The arithmetic below is
unchanged because ``_SlowVoice`` delivers a constant per-WORD rate; what changed is
that nothing is extrapolated from four lines to sixteen.
"""

from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

import src.services.stages.app_review.script as app_review_script
import src.services.stages.shared.speech_rate as speech_rate_module
from src.clients.tts_client import TTSClient
from src.services.runs.run_context import RunContext
from src.services.stages.app_review.record import AppReviewRecordStage
from src.services.stages.app_review.script import (
    NARRATE_SPEED,
    NARRATE_STAGE_NAME,
    AppReviewScriptStage,
)
from src.services.stages.shared.degradation import SPEECH_RATE_UNUSABLE
from src.services.stages.shared.speech_rate import SpeechRate, clear_measured_wpm_cache
from src.services.tts_storage import deterministic_clip_id
from tests.services.stages.app_review.rate_stubs import stub_speech_rate

#: What CALIBRATION_TEXT reports — the PRIOR. Rounded off the live 175.2 so every
#: number below is exact: 240 words at 180 wpm is exactly the 80s ordered.
PRIOR_WPM = 180.0
#: What the drafted lines actually deliver at. The live gap is 175.2 -> 119.4 (1.47x);
#: this is 1.5x, the same defect at a round number.
CONTENT_WPM = 120.0
ORDERED_SECONDS = 80
VOICE_ID = "af_heart"

#: 16 lines, 240 words, deliberately varied in length — the measurement speaks every
#: one of them, so the variation is what proves the total is summed, not scaled.
LINE_WORDS = [12, 18, 10, 20, 14, 16, 13, 17, 11, 19, 15, 15, 14, 16, 10, 20]
#: 240 words: exactly 80.0s at the PRIOR (verdict `ok`), exactly 120.0s at the rate
#: the voice really delivers (verdict `long`, +50%).
TOTAL_WORDS = sum(LINE_WORDS)


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
                {"text": _words(n, tag=f"{tag}{idx}_"), "seconds": n / 2.0}
                for idx, n in enumerate(word_counts)
            ]
        }
    )


def _ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"duration_seconds": ORDERED_SECONDS},
            "voice": {"descriptor": "neutral", "voice_id": VOICE_ID},
        },
        prior_run_outputs={
            "app_inspect": {"facts": [{"id": "f1", "kind": "title", "text": "T"}]},
            "app_critique": {
                "pros": [{"id": "p1", "text": "fast"}],
                "cons": [{"id": "c1", "text": "sparse"}],
                "recommendations": [{"id": "r1", "text": "ship"}],
                "overall_take": "ok",
            },
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


class _SlowVoice:
    """Every line takes the seconds ``CONTENT_WPM`` implies for its word count.

    A constant per-WORD rate, not a constant per-CLIP duration, so the measured total
    is exactly ``words / CONTENT_WPM * 60`` for any subset of lines: these tests assert
    the CORRECTION and never accidentally assert which lines were measured.

    ⚠️ It is therefore the WRONG fixture for asserting that the trim cuts by SECONDS
    rather than by words — the two agree here by construction. That case is pinned in
    ``test_script_narration_conformance.py`` with a voice whose seconds do not follow
    its word counts.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.texts: list[str] = []

    async def synthesize(self, **kwargs: Any) -> Any:
        text = str(kwargs["text"])
        self.texts.append(text)
        if self.fail:
            raise RuntimeError("tts.test is a launchd unit on mini2 and it is down")
        seconds = len(text.split()) / CONTENT_WPM * 60.0
        return type("_Synth", (), {"duration_seconds": seconds})()


def _as_client(fake: _SlowVoice) -> TTSClient:
    """The stage's contract on its TTS client is duck-typed — one async method.

    Same idiom as ``test_speech_rate.py::_client``: cast rather than subclass, so the
    fake stays a dozen lines instead of inheriting a real HTTP client's surface.
    """
    return cast("TTSClient", fake)


@pytest.fixture(autouse=True)
def _clean_rate_cache() -> Any:
    clear_measured_wpm_cache()
    yield
    clear_measured_wpm_cache()


@pytest.fixture
def stored() -> list[dict[str, Any]]:
    """Every clip the full-script measurement persisted, as recorded kwargs."""
    return []


def _arrange(
    monkeypatch: pytest.MonkeyPatch, stored: list[dict[str, Any]], *, fail: bool = False
) -> _SlowVoice:
    """Phase A pinned to the PRIOR, the full-script measurement real, storage faked.

    Only the calibration probe is stubbed. ``measure_script_lines`` itself runs for
    real — a stubbed measurement would prove the wiring and nothing about the
    arithmetic that motivated it.
    """
    monkeypatch.setattr(
        speech_rate_module,
        "resolve_natural_wpm_detail",
        AsyncMock(return_value=SpeechRate(wpm=PRIOR_WPM, measured=True)),
    )
    voice = _SlowVoice(fail=fail)

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

    monkeypatch.setattr(speech_rate_module, "synthesize_and_store_clip", _fake_store)
    return voice


# ── the correction ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_measuring_the_drafted_lines_flips_an_ok_verdict_to_long_and_corrects_the_draw(
    monkeypatch: pytest.MonkeyPatch, stored: list[dict[str, Any]]
) -> None:
    """THE root cause, end to end.

    240 words at the 180 wpm passage prior is exactly the 80s ordered — verdict ``ok``,
    ship it, no correction, and then the film comes out at 120s. The measurement speaks
    the same 240 words at the rate the voice really delivers, the verdict becomes
    ``long``, and the re-ask that was always there now fires.
    """
    voice = _arrange(monkeypatch, stored)
    fake = _wm(_draw(LINE_WORDS, tag="a"), _draw([10] * 16, tag="b"))  # 160 words = 80.0s

    out = await AppReviewScriptStage(client=fake, tts=_as_client(voice)).run({}, ctx=_ctx())

    reasks = [p for p in fake.prompts if "REVISION REQUIRED" in p]
    assert len(reasks) == 1, "the passage prior would have made this draw `ok` and shipped it"
    assert "TOO LONG" in reasks[0]
    # The re-ask states the numbers the MEASURED rate implies, not the prior's.
    assert f"{TOTAL_WORDS} spoken words" in reasks[0]
    assert "120 seconds" in reasks[0], "240 words at the measured 120 wpm"
    assert f"{ORDERED_SECONDS} seconds" in reasks[0]
    assert "160 words" in reasks[0], "what 80s affords at the measured rate"
    assert all(line.text.startswith("b") for line in out.lines), "the corrected draw shipped"
    assert out.actual_words == 160
    # And the word budget RECORDED on the output is the measured one, so the
    # ordered-vs-delivered accounting is no longer computed against a fiction.
    assert out.word_budget == 160


@pytest.mark.asyncio
async def test_without_the_measurement_the_same_draw_ships_uncorrected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CONTROL, and the pre-change behaviour: budgeted at the passage prior, the
    identical draw is inside the band, so nothing happens at all — one LLM call, no
    re-ask, no trim, no degradation — and 120s of narration ships against an 80s order.
    """
    stub_speech_rate(monkeypatch, app_review_script, PRIOR_WPM)
    fake = _wm(_draw(LINE_WORDS, tag="a"))

    out = await AppReviewScriptStage(client=fake).run({}, ctx=_ctx())

    assert fake.message.await_count == 1
    assert [p for p in fake.prompts if "REVISION REQUIRED" in p] == []
    assert out.degradations == []
    assert out.actual_words == TOTAL_WORDS


# ── the fallback ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_failed_measurement_keeps_the_passage_rate_and_records_a_degradation(
    monkeypatch: pytest.MonkeyPatch, stored: list[dict[str, Any]]
) -> None:
    """``tts.test`` is a launchd unit on mini2 — outside the cluster and outside
    every dashboard. A blip there must not silently restore the old, wrong budget:
    the run falls back to the prior AND says which rate it was budgeted at. That
    entry stays on ``degradations``, not ``advisories`` — a run budgeted at a rate
    nobody measured of this content is a real fault, unlike an estimate that missed.
    """
    voice = _arrange(monkeypatch, stored, fail=True)
    fake = _wm(_draw(LINE_WORDS, tag="a"))

    out = await AppReviewScriptStage(client=fake, tts=_as_client(voice)).run({}, ctx=_ctx())

    assert voice.texts, "it did try to measure"
    assert fake.message.await_count == 1, "at the prior rate the draw is in band, as before"
    codes = [entry["code"] for entry in out.degradations]
    assert codes == [SPEECH_RATE_UNUSABLE]
    detail = out.degradations[0]["detail"]
    assert "sample_too_few_measured" in detail
    assert "180.0 wpm calibration_passage" in detail, "names the rate actually used"
    assert "does not predict this content" in detail


@pytest.mark.asyncio
async def test_a_failed_measurement_is_byte_identical_to_the_passage_only_behaviour(
    monkeypatch: pytest.MonkeyPatch, stored: list[dict[str, Any]]
) -> None:
    """A cold path with no TTS must leave the stage exactly where it was. The ONLY
    difference a dead backend may make is the degradation entry that names it.
    """
    voice = _arrange(monkeypatch, stored, fail=True)
    without_tts = await AppReviewScriptStage(
        client=_wm(_draw(LINE_WORDS, tag="a")), tts=_as_client(voice)
    ).run({}, ctx=_ctx())

    stub_speech_rate(monkeypatch, app_review_script, PRIOR_WPM)
    passage_only = await AppReviewScriptStage(client=_wm(_draw(LINE_WORDS, tag="a"))).run(
        {}, ctx=_ctx()
    )

    assert [line.text for line in without_tts.lines] == [line.text for line in passage_only.lines]
    assert without_tts.total_seconds == passage_only.total_seconds
    assert without_tts.word_budget == passage_only.word_budget
    assert without_tts.actual_words == passage_only.actual_words
    assert passage_only.degradations == []


# ── the cost ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_measured_clips_are_stored_where_app_review_record_will_find_them(
    monkeypatch: pytest.MonkeyPatch, stored: list[dict[str, Any]]
) -> None:
    """What makes the sample FREE for every line that survives to delivery.

    ``app_review_record`` calls ``synthesize_and_store`` with the run's ``voice_id``,
    ``stage_name="app_review_record"`` and NO speed — so its clip id is
    ``deterministic_clip_id(text, voice_id, None)`` under that stage. The measurement
    stores under exactly that triple, so ``find_existing_clip`` hits and record
    synthesizes nothing for those lines. THIS is what makes measuring every line
    affordable; break it and the stage pays for the narration twice.
    """
    voice = _arrange(monkeypatch, stored)
    ctx = _ctx()
    fake = _wm(_draw(LINE_WORDS, tag="a"), _draw([10] * 16, tag="b"))

    await AppReviewScriptStage(client=fake, tts=_as_client(voice)).run({}, ctx=ctx)

    assert AppReviewRecordStage.name == NARRATE_STAGE_NAME
    assert NARRATE_SPEED is None, (
        "record omits speed, so the backend default holds; an explicit 1.0 is a "
        "DIFFERENT clip id and would be reused by nothing"
    )
    assert stored, "the measurement persisted its clips"
    assert {call["stage_name"] for call in stored} == {AppReviewRecordStage.name}
    assert {call["run_id"] for call in stored} == {str(ctx.run_id)}
    assert {call["voice"] for call in stored} == {VOICE_ID}
    assert {call["speed"] for call in stored} == {None}
    # The id record will compute for that very text, from record's own arguments.
    for call in stored:
        assert deterministic_clip_id(str(call["text"]), VOICE_ID, None) == deterministic_clip_id(
            str(call["text"]), call["voice"], call["speed"]
        )


@pytest.mark.asyncio
async def test_every_line_is_measured_and_the_shipped_ones_are_not_paid_for_twice(
    monkeypatch: pytest.MonkeyPatch, stored: list[dict[str, Any]]
) -> None:
    """The cost of measuring instead of sampling, stated exactly.

    A 4-line SAMPLE bought a rate, and a rate times a word count is a projection
    carrying several percent of its own error — on the one axis this stage is
    graded on. Measuring every line buys the film's real length instead, and it is
    very nearly free: each clip is stored under the RECORD stage's name at the
    speed record requests, so a line that survives to delivery is synthesized ONCE
    in total, here, rather than once here and once later.

    What it genuinely costs is the draft the re-ask replaced — 16 clips nobody
    hears. That is the honest price, and it is the reason the LLM budget stays at
    ONE re-ask: a third draft would cost a third whole script of synthesis.
    """
    voice = _arrange(monkeypatch, stored)
    fake = _wm(_draw(LINE_WORDS, tag="a"), _draw([10] * 16, tag="b"))

    await AppReviewScriptStage(client=fake, tts=_as_client(voice)).run({}, ctx=_ctx())

    assert len(stored) == 32, f"16 lines per draft over two drafts, got {len(stored)}"
    assert len({call["text"] for call in stored}) == 32, "every line distinct — nothing re-asked"
    shipped = {_words(10, tag=f"b{idx}_") for idx in range(16)}
    assert shipped <= {str(call["text"]) for call in stored}, (
        "every line that SHIPS was measured, i.e. already synthesized and stored"
    )


# ── the AUTHORED rate: the one number that outranks every measurement ────────
#
# `speech_rate.resolve_effective_wpm` puts the operator's own
# `direction.timing.wpm.natural` above both the calibration passage and any
# measurement of the drafted lines. This stage used to reach past that ladder
# and probe unconditionally, so an authored rate was accepted into the config
# and then silently overruled by whatever the voice happened to measure —
# exactly the "two places decide the rate for one script" defect the shared
# resolver exists to prevent. No live scenario authors a rate today (0 of 42 in
# `scripts/e2e_scenarios/`), so these pin a correctness gap, not a regression.

#: A rate no probe would ever return here: `_SlowVoice` delivers CONTENT_WPM
#: (120.0) and the stubbed passage prior is PRIOR_WPM (180.0). 90.0 is neither,
#: so every assertion below fails if ANY other rung of the ladder answers.
AUTHORED_WPM = 90.0
#: What 80 ordered seconds affords at the authored rate: 80/60*90.
AUTHORED_BUDGET_WORDS = 120


def _authored_ctx(wpm: float = AUTHORED_WPM) -> RunContext:
    """``_ctx()`` plus the operator's stated rate, in the shape the resolver reads."""
    ctx = _ctx()
    ctx.config_snapshot["direction"] = {"timing": {"wpm": {"natural": wpm}}}
    return ctx
