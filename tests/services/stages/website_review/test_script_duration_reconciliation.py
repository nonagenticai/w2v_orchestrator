"""website_review_script reconciles its draw against the ORDERED duration.

Measured live 2026-09-12: FIVE consecutive 60 s orders delivered +36.5%, +45.0%,
+36.5%, +52.8% and +51.4% — the worst content type on the board. Root cause, from
12 runs: the word budget AND the overrun check both ran at the CALIBRATION rate
(171-175 wpm) while the cast delivers 120-134 wpm, so the budget was ~20 s too
generous on a 60 s order and the check never fired. The delivered length IS the
sum of the TTS clip seconds (``website_review/compile.py`` slaves the picture to
the voice-over) and this DAG carries NO duration reconciler downstream, so
whatever the script stage returns is what the customer is billed for.

The fix is to stop projecting and MEASURE: every drafted line is synthesized at
the voice ``website_review_record`` will cast it to, and every verdict is taken
against those seconds.

Every test here drives the REAL stage with the WM client faked, so the assertions
bind to the shipped control flow (one re-ask, then a trim, never a pad, never a
raise) rather than to a helper called in isolation.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

import src.services.stages.shared.speech_rate as speech_rate_module
import src.services.stages.website_review.script as website_review_script
from src.services.runs.run_context import RunContext
from src.services.stages.primitives.dialogue import DEFAULT_CAST
from src.services.stages.shared.degradation import (
    DURATION_OUT_OF_TOLERANCE,
    NARRATION_ESTIMATE_OFF_TARGET,
    ORDERED_DURATION_INFEASIBLE,
    SPEECH_RATE_UNUSABLE,
    is_degraded,
)
from src.services.stages.shared.speech_rate import (
    PROBE_UNREACHABLE,
    ScriptMeasurement,
    SpeechRate,
)
from src.services.stages.website_review.record import WebsiteReviewRecordStage
from src.services.stages.website_review.script import WebsiteReviewScriptStage
from tests.services.stages.website_review.rate_stubs import (
    MeasurementRecorder,
    measurement_at,
    refused_measurement,
    stub_speech_rate,
)

#: A round rate keeps every expectation below exact arithmetic rather than a
#: tolerance dance: 1 word = 1 second, so "words" and "seconds" are the same axis.
WPM = 60.0
ORDERED_SECONDS = 60

#: The configured voice, and speaker "a" under the shared cast.
FAST_VOICE = "af_heart"
#: Speaker "b" under the shared cast — the voice the old budget never asked about.
SLOW_VOICE = "am_michael"


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _line(words: int, *, phase: str = "before", speaker: str = "a") -> dict[str, Any]:
    """One line of exactly ``words`` words ⇒ exactly ``words`` seconds at 60 wpm."""
    return {
        "text": " ".join(["word"] * words),
        "seconds": float(words),
        "phase": phase,
        "speaker": speaker,
    }


def _tagged(words: int, tag: str) -> dict[str, Any]:
    """A ``before`` line of ``words`` words, marked so a test can tell WHICH draw shipped."""
    return {
        "text": " ".join([tag] * words),
        "seconds": float(words),
        "phase": "before",
        "speaker": "a",
    }


def _asks(words: int, *, phase: str = "before") -> dict[str, Any]:
    """A line ending in "?" ⇒ ``assign_dialogue_speakers`` casts it to speaker "a"."""
    return {
        "text": " ".join(["word"] * (words - 1) + ["word?"]),
        "seconds": float(words),
        "phase": phase,
        "speaker": "a",
    }


def _reveal(words: int) -> dict[str, Any]:
    """The line that PRESENTS the redesign — pinned by ``_protected_indices``."""
    return {
        "text": "Here is the new redesign " + " ".join(["word"] * (words - 5)),
        "seconds": float(words),
        "phase": "after",
        "speaker": "b",
    }


def _ctx(brief: dict | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"duration_seconds": ORDERED_SECONDS, **(brief or {})},
            "voice": {"descriptor": "warm"},
        },
        prior_run_outputs={
            "website_critique": {
                "url": "https://acme.test",
                "overall_score": 70.0,
                "summary": "Slow and unclear.",
                "findings": [],
            },
            "website_redesign": {"style": "modern", "visuals": []},
        },
    )


def _cast_ctx(brief: dict | None = None) -> RunContext:
    """A ctx whose lines WILL be cast to two voices by website_review_record."""
    ctx = _ctx({"dialogue": {"enabled": True}, **(brief or {})})
    ctx.config_snapshot["voice"] = {"descriptor": "warm", "voice_id": FAST_VOICE}
    return ctx


def _wm(*draws: list[dict[str, Any]]) -> AsyncMock:
    """A WM fake returning one draw per call, repeating the last one thereafter.

    It also RECORDS every outbound prompt on ``fake.prompts``, so assertions can
    find a prompt by the marker that identifies it rather than by its index —
    a position is an implementation detail, the "REWRITE" block is the contract.
    """
    fake = AsyncMock()
    responses = [_resp({"lines": draw}) for draw in draws]
    prompts: list[str] = []
    fake.prompts = prompts

    async def _message(**kwargs: Any) -> WMResponse:
        prompts.append(str(kwargs.get("message", "")))
        return responses[min(len(prompts) - 1, len(responses) - 1)]

    fake.message.side_effect = _message
    return fake


async def _run(
    monkeypatch: pytest.MonkeyPatch,
    fake: AsyncMock,
    ctx: RunContext | None = None,
    rates: dict[str | None, float] | None = None,
) -> Any:
    stub_speech_rate(monkeypatch, website_review_script, WPM, rates)
    return await WebsiteReviewScriptStage(client=fake).run({}, ctx=ctx or _ctx())


async def _run_recorded(
    monkeypatch: pytest.MonkeyPatch,
    fake: AsyncMock,
    ctx: RunContext,
    rates: dict[str | None, float] | None = None,
) -> tuple[Any, MeasurementRecorder]:
    recorder = stub_speech_rate(monkeypatch, website_review_script, WPM, rates)
    out = await WebsiteReviewScriptStage(client=fake).run({}, ctx=ctx)
    return out, recorder


def _advisories(out: Any) -> list[dict[str, str]]:
    """The estimate MISS — advisory tier, because it grades audio nobody heard."""
    return [d for d in out.advisories if d["code"] == NARRATION_ESTIMATE_OFF_TARGET]


def _floors(out: Any) -> list[dict[str, str]]:
    return [d for d in out.degradations if d["code"] == ORDERED_DURATION_INFEASIBLE]


def _rate_entries(out: Any) -> list[dict[str, str]]:
    return [d for d in out.degradations if d["code"] == SPEECH_RATE_UNUSABLE]


# --------------------------------------------------------------------------
# The live shape: a draft over the order is corrected, and what SHIPS is checked.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_live_plus_45_percent_draft_ships_inside_the_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """87 words = 87.0 s against a 60 s order: +45.0%, the live median miss.

    The assertion is on what SHIPS, not on what the loop did: the estimate the
    stage finally stands behind must be inside the +/-10% band, and it must be the
    sum of the per-line seconds actually measured for the lines that survived.
    """
    draw = [_line(30), _reveal(27), _line(30, phase="after", speaker="b")]
    out = await _run(monkeypatch, _wm(draw))

    estimate = out.narration_estimate
    assert estimate is not None
    assert estimate.first_draw_estimate_seconds == pytest.approx(87.0), "the +45% draft"
    assert estimate.estimate_seconds == pytest.approx(57.0), "one 30 s line dropped"
    assert abs(estimate.estimate_seconds - ORDERED_SECONDS) <= ORDERED_SECONDS * 0.1
    assert estimate.verdict == "ok"
    assert _advisories(out) == [], "an in-band outcome raises nothing"
    assert out.degradations == []
    assert any("redesign" in line.text for line in out.lines), "the reveal survived"


@pytest.mark.asyncio
async def test_a_2_2x_overrun_is_re_asked_once_then_trimmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-09-10 failure: 60 s ordered, ~134 s drawn.

    132 words = 132 s = 2.2x the order. Exactly ONE re-ask (2 WM calls, never 3),
    and because the re-draw is no closer the tail is trimmed until the estimate is
    inside the +/-10% band.
    """
    long_draw = [_line(11) for _ in range(12)]  # 132 words = 132s = 2.2x
    fake = _wm(long_draw)
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 2, "exactly one bounded re-ask"
    assert out.actual_words <= round(ORDERED_SECONDS * 1.1), "trimmed into the band"
    assert len(out.lines) < 12, "whole lines were dropped"
    assert out.total_seconds == sum(line.seconds for line in out.lines), "totals re-derived"
    assert _advisories(out) == [], "a trim that landed is not a miss"
    assert out.degradations == []


@pytest.mark.asyncio
async def test_the_re_ask_states_the_countable_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    """The second prompt names produced words, estimated s, ordered s and the target."""
    fake = _wm([_line(11) for _ in range(12)])
    await _run(monkeypatch, fake)

    reasks = [p for p in fake.prompts if "REWRITE" in p]
    assert len(reasks) == 1, f"expected exactly one corrective re-ask, got {len(reasks)}"
    (reask,) = reasks
    assert any("REWRITE" not in p for p in fake.prompts), (
        "the opening draw must be asked WITHOUT a rewrite block"
    )
    assert "132 words" in reask  # produced
    assert "132 seconds" in reask  # MEASURED, not projected
    assert "60 seconds" in reask  # ordered
    assert "60 words" in reask  # target = budget_words(60s, 60wpm)


@pytest.mark.asyncio
async def test_a_within_tolerance_draw_is_untouched_and_records_the_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """58 words = 58 s against 60 s ordered ⇒ ok: no re-ask, no trim, no advisory.

    The estimate itself IS recorded, on a clean run as much as a bad one: "this
    run was measured and it landed" and "nobody ever measured this run" are
    otherwise the same absence.
    """
    draw = [_line(29), _reveal(29)]
    fake = _wm(draw)
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 1, "an in-band draw is never re-asked"
    assert len(out.lines) == 2
    assert out.actual_words == 58
    assert out.degradations == [] and _advisories(out) == []
    estimate = out.narration_estimate
    assert estimate is not None, "a clean run is measured and says so"
    assert estimate.estimate_seconds == pytest.approx(58.0)
    assert estimate.ordered_seconds == pytest.approx(60.0)
    assert estimate.wpm == pytest.approx(WPM)
    assert estimate.wpm_basis == "measured_all"
    assert estimate.measured_lines == 2, "both lines came off a real synthesis"
    assert estimate.reasked is False and estimate.reask_kept is None
    assert estimate.trimmed_lines == 0


@pytest.mark.asyncio
async def test_a_short_draw_is_accepted_and_never_padded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short re-draw ships SHORT. No duplicated lines, no freeze-pad, no filler.

    The model here returns the SAME short draw to every ask, so neither correction
    is closer and the draw ships as drawn — which is the point: exhausting the asks
    does not licence padding.
    """
    short = [_line(6), _reveal(6)]
    fake = _wm(short)
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 3, (
        "short gets TWO asks — the first draw, the re-ask, then (1b)'s second re-ask"
    )
    assert len(out.lines) == 2, "accepted as drawn — nothing added"
    assert out.actual_words == 12
    assert out.total_seconds == 12.0
    texts = [line.text for line in out.lines]
    assert len(set(texts)) == len(texts), "no line was duplicated to fill time"
    assert out.narration_estimate is not None
    assert out.narration_estimate.verdict == "short"
    assert out.narration_estimate.trimmed_lines == 0, "nothing is ever trimmed on the short side"
    assert _advisories(out), "shipping short is recorded, not hidden"
    assert "12.0s" in _advisories(out)[0]["detail"]


async def test_a_draw_still_short_after_one_reask_gets_a_second_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(1b): the short side's second attempt, which is what the live misses lacked.

    Measured 2026-09-15 on three live 60s orders: delivered 45.7s, 51.8s and 51.5s
    (-23.8%, -13.6%, -14.1%) — and all three recorded ``reasked=true,
    reask_kept=true, verdict="short"``. One correction had already fired and been
    KEPT; there was simply no second one to take. Here the second ask lands the
    draw in band, which the first could not.
    """
    short = [_line(6), _reveal(6)]  # 12 words -> 12.0s, far short of 60
    closer = [_line(15), _reveal(15)]  # 30 words -> 30.0s, closer but STILL short
    lands = [_line(29), _reveal(29)]  # 58 words -> 58.0s, inside the +/-10% band
    fake = _wm(short, closer, lands)
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 3, "first draw, re-ask, then the short side's second"
    assert out.total_seconds == 58.0, "the second re-ask is what ships"
    assert out.actual_words == 58
    assert out.narration_estimate is not None
    assert out.narration_estimate.verdict == "ok", "58.0s against 60 is in band"
    assert out.narration_estimate.reask_kept is True
    assert not _advisories(out), "an in-band draw raises no estimate miss"


async def test_the_second_short_reask_is_discarded_when_it_overshoots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It is ``closer_to_order``, not "any second ask wins".

    A second ask that swings past the order is worse than the short draw it was
    correcting, so it is discarded — the same rule that governs the first re-ask.

    The first re-ask must itself stay SHORT here: a re-ask that swung long would
    put a trimmable candidate in the pool and (1b) would correctly stand down,
    which is a different path (see the pre-empt test below).
    """
    short = [_line(25), _reveal(25)]  # 50 words -> 50.0s, short of the 54.0s edge
    better = [_line(26), _reveal(26)]  # 52 words -> 52.0s, closer but still short
    worse = [_line(40), _reveal(40)]  # 80 words -> 80.0s, overshoots by more
    fake = _wm(short, better, worse)
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 3
    assert out.total_seconds == 52.0, "the overshooting second ask is thrown away"
    assert out.narration_estimate is not None
    assert out.narration_estimate.verdict == "short"


async def test_a_second_reask_that_is_not_closer_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected second ask must leave a trace, or the lever cannot be evaluated.

    Shipped 2026-09-15 without this: the first live wave after the second re-ask
    landed had three still-short 60s runs (-30.3%, -15.3%, -10.1%) and NOT ONE row
    carried the "re-asked twice" fingerprint — because only the KEPT path wrote a
    string. "The guard never opened" and "it opened every time and the draw was
    never closer" looked identical in the data. They need different fixes.
    """
    short = [_line(25), _reveal(25)]  # 50.0s
    better = [_line(26), _reveal(26)]  # 52.0s — closer, still short
    worse = [_line(40), _reveal(40)]  # 80.0s — overshoots, rejected
    fake = _wm(short, better, worse)
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 3
    detail = _advisories(out)[0]["detail"]
    assert "second re-ask not closer" in detail, "a rejected second ask must be visible"
    assert "80.0s" in detail and "52.0s" in detail, "it must carry BOTH estimates it judged"


async def test_a_second_reask_that_stands_down_for_the_trim_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standing down is a DECISION, and a silent decision cannot be told from a
    code path that does not exist. It names the long candidate it deferred to."""
    short = [_line(25), _reveal(25)]  # 50.0s, |50-60| = 10
    long_draw = [_line(35), _reveal(35)]  # 70.0s, |70-60| = 10 (a tie, so short leads)
    fake = _wm(short, long_draw, long_draw)
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 2, "no paid second ask — the trim can act for free"
    detail = _advisories(out)[0]["detail"]
    assert "second re-ask stood down for the trim" in detail
    assert "70.0s" in detail, "it names the long candidate it deferred to"


async def test_a_long_candidate_in_the_pool_pre_empts_the_second_short_reask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A paid LLM call must not pre-empt the free deterministic trim.

    When a LONG draw exists anywhere in the pool the trim can reach it for free,
    so (1b) stands down even though the CURRENT best is short. Here the re-ask
    swings long and ties on distance, so the short first draw keeps the lead while
    a trimmable long candidate sits in the pool.
    """
    short = [_line(25), _reveal(25)]  # 50 words -> 50.0s, |50-60| = 10
    long_draw = [_line(35), _reveal(35)]  # 70 words -> 70.0s, |70-60| = 10 (a tie)
    fake = _wm(short, long_draw, long_draw)
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 2, (
        "no second short re-ask: the pool holds a LONG candidate the trim can reach for free"
    )
    assert out.narration_estimate is not None
    assert out.narration_estimate.reask_kept is False, "the tie keeps the first draw"


async def test_a_draw_still_short_after_two_reasks_gets_a_third(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A THIRD short-side round, for a gap one round cannot close.

    Prod run 01a0b2f5-ed32-73f2-8a6e-2158a9ccf37a (website_review, 120s order,
    example.com, 2026-09-18): the first draft came in at -51% (58.5s), the
    directional re-ask plus ONE short-side round only recovered it to -25.9%
    (88.9s) — still outside the +/-10% band — and there was no second short round
    left to take, so the draw shipped short. Each round here keeps landing
    closer while STILL short, so a third round must fire rather than shipping
    the second round's still-short result.
    """
    short = [_line(12), _reveal(12)]  # 24 words -> 24.0s, -60% of 60s
    closer_1 = [_line(20), _reveal(20)]  # 40 words -> 40.0s, -33%, still short
    closer_2 = [_line(25), _reveal(25)]  # 50 words -> 50.0s, -17%, still short
    lands = [_line(29), _reveal(29)]  # 58 words -> 58.0s, inside the +/-10% band
    fake = _wm(short, closer_1, closer_2, lands)
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 4, (
        "first draft, the directional re-ask, then TWO short-side rounds"
    )
    assert out.total_seconds == 58.0, "the third round is what ships"
    assert out.actual_words == 58
    assert out.narration_estimate is not None
    assert out.narration_estimate.verdict == "ok", "58.0s against 60 is in band"
    assert out.narration_estimate.reask_kept is True
    assert not _advisories(out), "an in-band draw raises no estimate miss"


# --------------------------------------------------------------------------
# The CAST. website_review is MULTI-VOICE: record casts each line to its
# speaker's voice, and ``deterministic_clip_id`` hashes (voice, speed, text) —
# so the measurement must ask for exactly what record will ask for, or the run
# pays for every line twice AND measures audio nobody will hear.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_line_is_measured_at_the_voice_record_will_cast_it_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE clip-reuse contract, asserted against ``record.voice_for`` itself.

    ``website_review_record`` resolves each line's voice through
    ``voice_for_speaker``; ``deterministic_clip_id`` then hashes that voice
    together with the speed and the text. Measure a line at any other voice and
    two things break at once: the seconds describe audio nobody will hear, and
    record finds no stored clip under the id it looks up — so every line is
    synthesized a second time, at full price.

    Checked three ways, because the id has three parts: the voices are the
    literal cast ids, they equal what the RECORD stage's own ``voice_for``
    returns for the same lines, and the speed/stage_name are record's too.
    """
    draw = [_asks(20), _line(20, phase="after", speaker="b"), _asks(20, phase="after")]
    ctx = _cast_ctx()
    out, recorder = await _run_recorded(monkeypatch, _wm(draw), ctx, {FAST_VOICE: 60.0})

    speakers = [line.speaker for line in out.lines]
    assert speakers == ["a", "b", "a"], "the derived two-hander really is mixed"
    assert recorder.voices == (FAST_VOICE, SLOW_VOICE, FAST_VOICE), "the literal cast"
    assert DEFAULT_CAST == {"a": FAST_VOICE, "b": SLOW_VOICE}, "…and it is the shared cast"

    # Not a copy of the rule: ask the RECORD stage what it would cast these very
    # lines to. A future edit to either side has to move both or this goes red.
    record = WebsiteReviewRecordStage(client=MagicMock())
    voice_block = ctx.config_snapshot["voice"]
    expected = tuple(
        record.voice_for({"speaker": line.speaker, "text": line.text}, voice_block, ctx)
        for line in out.lines
    )
    assert recorder.voices == expected, "the script measured what record will narrate"

    assert recorder.speeds[-1] is None, "record omits speed; the clip id hashes None"
    assert recorder.stage_names[-1] == WebsiteReviewRecordStage.name, "…and record's own name"


@pytest.mark.asyncio
async def test_each_draw_is_judged_at_its_own_measured_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The re-ask changes the speaker mix, so it is measured on ITS OWN text.

    First draw: 20 "a" words (120 wpm ⇒ 10 s) + 2x40 "b" words (60 wpm ⇒ 80 s) =
    90.0 s, pooled 66.7 wpm. Re-ask: 140 "a" words (70 s) + 10 "b" words (10 s) =
    80.0 s — closer (20 < 30), so it is kept. Graded at the FIRST draw's pooled
    66.7 wpm the same 150 words read as 135 s and the re-draw would be discarded
    as a wild overcorrection.
    """
    first = [_asks(20), *[_line(40, phase="after", speaker="b") for _ in range(2)]]
    reask = [_asks(140), _line(10, phase="after", speaker="b")]
    fake = _wm(first, reask)
    out = await _run(monkeypatch, fake, _cast_ctx(), {FAST_VOICE: 120.0, SLOW_VOICE: 60.0})

    assert fake.message.await_count == 2
    assert out.actual_words == 150, "the re-draw — closer at its OWN seconds — is the shipped one"
    estimate = out.narration_estimate
    assert estimate is not None
    assert estimate.first_draw_estimate_seconds == pytest.approx(90.0)
    assert estimate.estimate_seconds == pytest.approx(80.0)
    assert estimate.reask_kept is True
    (miss,) = _advisories(out)
    assert "re-asked" in miss["detail"] and "not closer" not in miss["detail"]


# --------------------------------------------------------------------------
# Protection: the picture is keyed off specific lines, so the trim may not have
# them — even when that means shipping long and SAYING so.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_reveal_line_survives_a_hard_trim(monkeypatch: pytest.MonkeyPatch) -> None:
    """compile.py keys the redesign to "after" lines — one MUST survive.

    ``plan_trim`` drops last-first, i.e. straight through the after act, so without
    the protected index a hard overrun would deliver a critique and never show the
    redesign at all.
    """
    draw = [
        _line(40),
        _reveal(45),
        *[_line(40, phase="after", speaker="b") for _ in range(4)],
    ]
    out = await _run(monkeypatch, _wm(draw))

    assert len(out.lines) < len(draw), "the draw was trimmed"
    assert any(line.phase == "after" for line in out.lines), "the redesign is still presented"
    assert any("redesign" in line.text for line in out.lines)


@pytest.mark.asyncio
async def test_an_authored_before_act_is_never_trimmed_away(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shotlist lines are the operator's OWN words, keyed to their own captures.

    Sized so the trim MUST want them: 60 words against a 30 s order (limit 33 s)
    where the only AI line is the protected reveal. Everything is protected, so
    nothing is dropped — the stage ships long and SAYS so, rather than deleting
    text a human wrote by hand or orphaning its site_capture screen.
    """
    authored_a = " ".join(["pricing"] * 20)
    authored_b = " ".join(["signup"] * 20)
    ctx = _ctx(
        {
            "duration_seconds": 30,
            "shotlist": [
                {"url": "https://acme.test/pricing", "narration": authored_a},
                {"url": "https://acme.test/signup", "narration": authored_b},
            ],
        }
    )
    fake = _wm([_line(20, phase="after", speaker="b")])
    out = await _run(monkeypatch, fake, ctx)

    texts = [line.text for line in out.lines]
    assert authored_a in texts and authored_b in texts, "authored narration survived"
    assert len(out.lines) == 3, "nothing was dropped — every line is protected"
    assert out.actual_words == 60
    assert _advisories(out), "the un-trimmable overrun is reported, not hidden"
    assert "STILL LONG" in _advisories(out)[0]["detail"], "the trim did not land, and says so"
    # The FLOOR is its own entry, on the DEGRADING tier, with the actionable numbers.
    (floor,) = _floors(out)
    assert "60.0" in floor["detail"], "the seconds it floored at"
    assert "30.0" in floor["detail"], "the ordered seconds"
    assert "3 protected" in floor["detail"], "how much was pinned"


@pytest.mark.asyncio
async def test_a_trim_that_drops_lines_and_stays_long_is_not_reported_as_a_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE live defect: ordered 60 s, delivered 77.46 s (+29.1%), reported "trimmed".

    ``_protected_indices`` pins every authored shotlist line — an UNBOUNDED
    prefix — so a large enough shotlist puts the band out of reach no matter what
    the trim drops. Before this, the action string read "trimmed 2 line(s)",
    which is the exact sentence a trim that LANDED writes: a gate that cannot fail.
    """
    authored = [
        {"url": f"https://acme.test/{n}", "narration": " ".join([f"page{n}"] * 30)}
        for n in range(4)
    ]
    ctx = _ctx({"duration_seconds": ORDERED_SECONDS, "shotlist": authored})
    draw = [_line(20), _reveal(20), _line(20, phase="after", speaker="b")]
    out = await _run(monkeypatch, _wm(draw), ctx)

    estimate = out.narration_estimate
    assert estimate is not None
    assert estimate.estimate_seconds > ORDERED_SECONDS * 1.1, "STILL over the band"
    assert len(out.lines) < 4 + len(draw), "and lines WERE dropped — the ambiguous shape"
    assert estimate.trimmed_lines == 2

    (miss,) = _advisories(out)
    assert "trimmed" in miss["detail"], "it did trim"
    assert "STILL LONG" in miss["detail"], "and it says the trim did not reach the band"
    (floor,) = _floors(out)
    assert f"{float(ORDERED_SECONDS):.1f}s" in floor["detail"], "the ordered seconds"
    # 4 authored shotlist lines + the protected reveal line.
    assert "5 protected" in floor["detail"], "how much was pinned"
    assert "2 line(s) dropped" in floor["detail"]
    assert "not reachable by trimming" in floor["detail"]


# --------------------------------------------------------------------------
# Tiers. An estimate MISS grades audio nobody has heard, so it is ADVISORY; a
# rate or a measurement that did not land means every number is unsound, which
# is a DEGRADATION.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_estimate_miss_stays_an_advisory_and_only_the_short_floor_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The steering note never rides the degrading tier — the short FLOOR does.

    ``narration_estimate_off_target`` is how the script was AIMED and stays an
    advisory. But a draw that is still below the band after every floored re-ask
    is not a steering note: the film is exactly its narration, so 12s against a
    60s order IS the delivered length. Until 2026-10-05 this test pinned
    ``out.degradations == []`` for exactly that draw, which is how a 50s order
    shipped ~15s with nothing above an advisory. Now the short floor is the ONE
    degradation, on ``duration_out_of_tolerance``
    (see ``tests/unit/test_website_review_short_draw_floor.py``).
    """
    short = [_line(6), _reveal(6)]
    out = await _run(monkeypatch, _wm(short))

    assert _advisories(out), "the miss is recorded"
    assert all(d["code"] != NARRATION_ESTIMATE_OFF_TARGET for d in out.degradations)
    assert [d["code"] for d in out.degradations] == [DURATION_OUT_OF_TOLERANCE], (
        "a draw shipped below the floor is degraded — once, and by nothing else"
    )


# --------------------------------------------------------------------------
# The re-ask must NOT overcorrect, and must never be able to lose the run.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_reask_that_overcorrects_keeps_the_first_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First +11.7% (67 words), re-ask -30% (42 words) ⇒ the FIRST draw is kept."""
    first = [_line(34), _reveal(33)]  # 67s
    reask = [_line(21), _reveal(21)]  # 42s
    fake = _wm(first, reask)
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 2, "the re-ask ran"
    assert out.actual_words != 42, "the -30% re-draw must not ship"
    assert out.actual_words == 67, "the closer first draw ships (nothing trimmable in band)"
    assert out.narration_estimate is not None
    assert out.narration_estimate.reask_kept is False
    (miss,) = _advisories(out)
    assert "re-ask not closer; kept first draw" in miss["detail"]


# --------------------------------------------------------------------------
# What a DISCARDED candidate may and may not leave behind, what the trim is
# allowed to reach for, and which rate the measurement is actually taken at.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_discarded_re_asks_failed_measurement_never_degrades_the_shipped_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A draw that was THROWN AWAY does not get to flip the run ``degraded``.

    The shipped script here was measured end to end. The RE-ASK's measurement is
    the one that failed — and that re-ask was then discarded for landing further
    from the order, so not one of its words is in the deliverable. Merging its
    ``SPEECH_RATE_UNUSABLE`` entry into the output made the run degraded and made
    the record say every duration number "was budgeted at the ...
    calibration_passage rate", which is simply false of the draft that shipped.
    Advisories have never leaked between candidates; degradations must not either.
    """
    prior = SpeechRate(wpm=WPM, measured=True)
    monkeypatch.setattr(
        speech_rate_module, "resolve_natural_wpm_detail", AsyncMock(return_value=prior)
    )
    calls: list[int] = []

    async def _measure(_ctx: Any, *, texts: Any, voices: Any, **_kwargs: Any) -> ScriptMeasurement:
        calls.append(len(texts))
        if len(calls) == 1:
            return measurement_at(texts, voices, {}, WPM, prior)
        return refused_measurement(texts, prior, PROBE_UNREACHABLE)

    monkeypatch.setattr(website_review_script, "measure_script_lines", _measure)

    # 64 words = 64.0s = +6.7% ⇒ steered (past the 4% steer band), yet INSIDE the
    # +/-10% band that grades the master, so this draw ships whole.
    shipped = [_line(32), _reveal(32)]
    far = [_line(15), _reveal(15)]  # 30 words = 30.0s ⇒ far worse, discarded
    fake = _wm(shipped, far)

    out = await WebsiteReviewScriptStage(client=fake).run({}, ctx=_ctx())

    assert fake.message.await_count == 2, "the re-ask really was asked for"
    assert len(calls) == 2, "and its text really was measured — the failure is not skipped"
    assert out.actual_words == 64, "the discarded draw is gone"
    assert _rate_entries(out) == [], "the re-ask took its failed measurement with it"
    assert out.degradations == []
    assert not is_degraded(out.model_dump()), "the script that ships was fully measured"
    assert out.narration_estimate is not None
    assert out.narration_estimate.reask_kept is False, "it WAS discarded"
    assert out.narration_estimate.measured_lines == 2, "both shipped lines measured"


@pytest.mark.asyncio
async def test_a_long_sibling_is_trimmed_when_the_incumbent_is_a_short_re_ask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trim scans the whole POOL — restricting it to the incumbent ships -25%.

    The first draft is 80.0s against a 60s order (+33%). The re-ask swings to
    45.0s (-25%), which is CLOSER (15 < 20) and therefore wins — and a trim of the
    winner is impossible, because nothing shortens a short script. The rescue is
    the long draft still sitting in the pool: dropping one of its 20-word lines
    lands it at exactly 60.0s, nearer the order than anything else on offer, at
    the cost of no synthesis and no LLM call.

    Narrow ``_trim_step``'s sources to ``[incumbent]`` and there are no sources at
    all here, so the -25% re-ask ships untouched. That is the whole of what this
    pins: 60.0s versus 45.0s.
    """
    first = [_line(20), _line(20), _line(20), _reveal(20)]  # 80 words = 80.0s
    reask = [_line(22), _reveal(23)]  # 45 words = 45.0s
    out = await _run(monkeypatch, _wm(first, reask))

    assert out.actual_words == 60, "the long first draft, trimmed — not the 45-word re-ask"
    assert len(out.lines) == 3, "one whole line dropped out of the FIRST draft"
    assert out.narration_estimate is not None
    assert out.narration_estimate.estimate_seconds == pytest.approx(60.0)
    assert out.narration_estimate.reask_kept is True, "the re-ask did win on distance first"
    assert out.narration_estimate.trimmed_lines == 1
    assert _advisories(out) == [], "landing on the order exactly is not a miss"
    assert out.degradations == []


@pytest.mark.asyncio
async def test_the_trim_aims_at_the_outer_band_not_at_the_steer_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PINS A DECISION: ``plan_trim`` is given the +/-10% tolerance, never the 4% steer.

    The two constants answer different questions. ``steer_verdict``'s 4% decides
    WHETHER to correct, leaving headroom for the estimate's own error. The trim's
    tolerance decides what a cut is allowed to LAND on — and it declines any drop
    falling below ``ordered x (1 - tolerance)``. Tightening it therefore makes the
    trim refuse cuts it should take.

    Here 70.0s against a 60s order has exactly one useful cut: drop the 15-word
    line and land at 55.0s, conformant at -8.3%. Aimed at the 4% band that cut is
    "under the band" and is DECLINED, the 30-word drop lands at -33% and is
    declined too, and the full 70.0s overrun ships instead. Narrowing the trim
    target must fail here, loudly.
    """
    draw = [_line(30), _line(15), _reveal(25)]  # 70 words = 70.0s = +16.7%
    out = await _run(monkeypatch, _wm(draw, draw))

    assert len(out.lines) == 2, "the 15-word line dropped — the cut the 4% band refuses"
    assert out.actual_words == 55
    assert out.narration_estimate is not None
    assert out.narration_estimate.estimate_seconds == pytest.approx(55.0)
    assert out.narration_estimate.trimmed_lines == 1
    assert _advisories(out) == [], "-8.3% is inside the band that grades the master"
    assert out.degradations == [], "and nothing was floored"


@pytest.mark.asyncio
async def test_the_full_script_measurement_is_taken_at_the_phase_a_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``measure_script_lines`` gets ``self._rate`` — the OBJECT Phase A resolved.

    It is the fallback the measurement extrapolates from when it cannot synthesise,
    so it decides every duration number on a refused run. Every other refusal test
    here stubs the measurement with a prior of its OWN making, which makes the
    argument the stage actually passes invisible: hardcode it to an unmeasured
    150.0 and they all still pass. This stub ECHOES what it was handed, so the
    shipped estimate is evidence of it.

    97 words at the resolved 97.0 wpm is exactly the 60s ordered — verdict ``ok``,
    one WM call, one measurement. At a hardcoded 150.0 the same words read as
    38.8s (-35%) and the stage would re-ask.
    """
    prior = SpeechRate(wpm=97.0, measured=True)
    monkeypatch.setattr(
        speech_rate_module, "resolve_natural_wpm_detail", AsyncMock(return_value=prior)
    )
    priors: list[SpeechRate] = []

    async def _measure(_ctx: Any, *, texts: Any, prior: Any, **_kwargs: Any) -> ScriptMeasurement:
        priors.append(prior)
        return refused_measurement(texts, prior, PROBE_UNREACHABLE)

    monkeypatch.setattr(website_review_script, "measure_script_lines", _measure)
    fake = _wm([_line(50), _reveal(47)])  # 97 words

    out = await WebsiteReviewScriptStage(client=fake).run({}, ctx=_ctx())

    assert priors, "the full-script measurement ran at all"
    assert priors[0] is prior, "the Phase-A rate OBJECT, not a fresh one and not a constant"
    assert fake.message.await_count == 1, "at the resolved 97.0 wpm the draw is in band"
    assert out.narration_estimate is not None
    assert out.narration_estimate.wpm == pytest.approx(97.0), "budgeted at the prior it passed"
    assert out.narration_estimate.estimate_seconds == pytest.approx(60.0)
    assert _advisories(out) == [], "97 words at 97 wpm IS the 60s order"
