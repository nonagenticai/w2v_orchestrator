"""lesson_script reconciles its draw against the ORDERED duration — by MEASURING it.

Measured live 2026-09-13, five consecutive 120s orders: delivered +4.21%, +4.48%,
+11.03%, +24.03% and +26.35%. Two of those ALSO carried a false
``duration_out_of_tolerance`` raised from the pre-audio estimate while the
delivered film was inside the band. Both halves come from one cause: every number
in the loop was a PROJECTION — ``words / wpm`` at a blended cast rate — so the
check fired on lessons that were fine and missed lessons that were not.

The delivered lesson length IS the sum of its TTS clips (``lesson/compile.py``
rescales the slides to whatever audio came back) and ``lesson_narrate`` is the
stage that synthesizes them, so the fix is to stop projecting and MEASURE: every
drafted paragraph is synthesized at the voice ``lesson_narrate`` will cast it to,
under narrate's own stage name and speed, and every verdict is taken against
those seconds.

Every test here drives the REAL stage with the WM client faked, so the assertions
bind to the shipped control flow (one re-ask, then a trim, never a pad, never a
raise) rather than to a helper called in isolation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

import src.services.stages.lesson.script as lesson_script
import src.services.stages.shared.speech_rate as speech_rate_module
from src.services.runs.run_context import RunContext
from src.services.stages.lesson.narrate import _DEFAULT_CAST, LessonNarrateStage
from src.services.stages.lesson.script import LessonScriptStage
from src.services.stages.shared.degradation import (
    NARRATION_ESTIMATE_OFF_TARGET,
    ORDERED_DURATION_INFEASIBLE,
    SPEECH_RATE_UNUSABLE,
)
from src.services.stages.shared.speech_rate import (
    BASIS_MEASURED_ALL,
    PROBE_UNREACHABLE,
    ScriptMeasurement,
    SpeechRate,
)
from tests.services.stages.lesson.rate_stubs import (
    MeasurementRecorder,
    stub_failed_measurement,
    stub_speech_rate,
)

#: A round rate keeps every expectation below exact arithmetic rather than a
#: tolerance dance: 1 word = 1 second, so "words" and "seconds" are the same axis.
WPM = 60.0
#: 2 minutes = 120s, the live order that produced the five misses above.
ORDERED_MINUTES = 2
ORDERED_SECONDS = 120.0

#: The configured voice, and speaker "a" (the student) under lesson's own cast.
STUDENT_VOICE = "af_heart"
#: Speaker "b" (the instructor) — the voice that carries most of a lesson, and the
#: one a single-rate budget never asked about.
INSTRUCTOR_VOICE = "am_michael"


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _para(words: int, *, speaker: str = "b") -> dict[str, Any]:
    """One paragraph of exactly ``words`` words ⇒ exactly ``words`` seconds at 60 wpm."""
    return {"text": " ".join(["word"] * words), "seconds": float(words), "speaker": speaker}


def _asks(words: int) -> dict[str, Any]:
    """A paragraph ending in "?" ⇒ ``assign_dialogue_speakers`` casts it to "a"."""
    return {
        "text": " ".join(["word"] * (words - 1) + ["word?"]),
        "seconds": float(words),
        "speaker": "a",
    }


def _tagged(words: int, tag: str) -> dict[str, Any]:
    """A paragraph of ``words`` words, marked so a test can tell WHICH draw shipped."""
    return {"text": " ".join([tag] * words), "seconds": float(words), "speaker": "b"}


def _alt(words: int) -> dict[str, Any]:
    """Same word count as ``_para``, DIFFERENT text, so a tie is exact in words."""
    return {"text": " ".join(["other"] * words), "seconds": float(words), "speaker": "b"}


def _ctx(brief: dict | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {
                "duration_minutes": ORDERED_MINUTES,
                "learning_objective": "Diagnose churn drivers.",
                **(brief or {}),
            },
            "voice": {"descriptor": "calm"},
        },
    )


def _cast_ctx(brief: dict | None = None) -> RunContext:
    """A ctx whose paragraphs WILL be cast to two voices by lesson_narrate."""
    ctx = _ctx({"dialogue": {"enabled": True}, **(brief or {})})
    ctx.config_snapshot["voice"] = {"descriptor": "calm", "voice_id": STUDENT_VOICE}
    return ctx


def _wm(*draws: list[dict[str, Any]]) -> AsyncMock:
    """A WM fake returning one draw per call, repeating the last one thereafter.

    It also RECORDS every outbound prompt on ``fake.prompts``, so assertions can
    find a prompt by the marker that identifies it rather than by its index — a
    position is an implementation detail, the "CORRECTION" block is the contract.
    """
    fake = AsyncMock()
    responses = [_resp({"paragraphs": draw}) for draw in draws]
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
    stub_speech_rate(monkeypatch, lesson_script, WPM, rates)
    return await LessonScriptStage(client=fake).run({}, ctx=ctx or _ctx())


async def _run_recorded(
    monkeypatch: pytest.MonkeyPatch,
    fake: AsyncMock,
    ctx: RunContext,
    rates: dict[str | None, float] | None = None,
) -> tuple[Any, MeasurementRecorder]:
    recorder = stub_speech_rate(monkeypatch, lesson_script, WPM, rates)
    out = await LessonScriptStage(client=fake).run({}, ctx=ctx)
    return out, recorder


def _advisories(out: Any) -> list[dict[str, str]]:
    """The estimate MISS — advisory tier, because it grades audio nobody heard."""
    return [d for d in out.advisories if d["code"] == NARRATION_ESTIMATE_OFF_TARGET]


def _floors(out: Any) -> list[dict[str, str]]:
    return [d for d in out.degradations if d["code"] == ORDERED_DURATION_INFEASIBLE]


def _rate_entries(out: Any) -> list[dict[str, str]]:
    return [d for d in out.degradations if d["code"] == SPEECH_RATE_UNUSABLE]


def _corrections(prompts: list[str]) -> list[str]:
    """Every recorded prompt carrying a corrective block — found by content."""
    return [p for p in prompts if "CORRECTION" in p]


# --------------------------------------------------------------------------
# The live shape: the worst measured draft is corrected, and what SHIPS is checked.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_live_plus_26_percent_draft_ships_inside_the_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """152 words = 152.0s against a 120s order: +26.7%, the live worst case.

    The assertion is on what SHIPS, not on what the loop did: the estimate the
    stage finally stands behind must be inside the +/-10% band, and it must be the
    sum of the per-paragraph seconds actually measured for the paragraphs that
    survived.
    """
    draw = [_para(38) for _ in range(4)]  # 152 words = 152.0s = +26.7%
    fake = _wm(draw)
    out = await _run(monkeypatch, fake)

    estimate = out.narration_estimate
    assert estimate is not None
    assert estimate.first_draw_estimate_seconds == pytest.approx(152.0), "the +26.7% draft"
    assert estimate.estimate_seconds == pytest.approx(114.0), "one 38s paragraph dropped"
    assert abs(estimate.estimate_seconds - ORDERED_SECONDS) <= ORDERED_SECONDS * 0.1
    assert estimate.verdict == "ok"
    assert _advisories(out) == [], "an in-band outcome raises nothing"
    assert out.degradations == []
    assert out.total_seconds == sum(p.seconds for p in out.paragraphs), "totals re-derived"


@pytest.mark.asyncio
async def test_the_best_draw_ships_and_is_reported_honestly_when_none_land(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One 300-word paragraph cannot be dropped — the script would be empty.

    So the trim runs, reaches nothing, and the run says BOTH things: the estimate
    missed (advisory) and the order was not reachable by trimming (floor, on the
    degrading tier). "Shipped the best draw" and "shipped a conformant draw" must
    never read the same.
    """
    fake = _wm([_para(300)])
    out = await _run(monkeypatch, fake)

    assert len(out.paragraphs) == 1, "the only paragraph survived"
    assert out.narration_estimate is not None
    assert out.narration_estimate.estimate_seconds == pytest.approx(300.0)
    assert out.narration_estimate.verdict == "long"
    (miss,) = _advisories(out)
    assert "STILL LONG" in miss["detail"], "the trim did not land, and says so"
    (floor,) = _floors(out)
    assert "120.0" in floor["detail"], "the ordered seconds"
    assert "0 protected" in floor["detail"], "a lesson pins nothing — the floor is emptiness"


@pytest.mark.asyncio
async def test_an_in_band_draft_is_untouched_and_records_the_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """118 words = 118.0s against 120s ⇒ ok: no re-ask, no trim, no advisory.

    The estimate itself IS recorded, on a clean run as much as a bad one: "this
    run was measured and it landed" and "nobody ever measured this run" are
    otherwise the same absence.
    """
    fake = _wm([_para(59), _para(59)])
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 1, "an in-band draw is never re-asked"
    assert len(out.paragraphs) == 2
    assert out.actual_words == 118
    assert out.degradations == [] and _advisories(out) == []
    estimate = out.narration_estimate
    assert estimate is not None, "a clean run is measured and says so"
    assert estimate.estimate_seconds == pytest.approx(118.0)
    assert estimate.ordered_seconds == pytest.approx(ORDERED_SECONDS)
    assert estimate.wpm == pytest.approx(WPM)
    assert estimate.wpm_basis == "measured_all"
    assert estimate.measured_lines == 2, "both paragraphs came off a real synthesis"
    assert estimate.reasked is False and estimate.reask_kept is None
    assert estimate.trimmed_lines == 0


@pytest.mark.asyncio
async def test_a_plus_6_percent_draft_is_steered_although_the_outer_band_accepts_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """+6.0% is "ok" to the +/-10% band and "long" to the +4% steer.

    THE calibration fix, and the one the live +4.21%/+4.48%/+11.03% runs needed.
    Aiming at the band EDGE leaves nothing for the estimate's own error, which is
    how a draw accepted at +9% delivers outside the band.
    """
    first = [_para(64), _para(63)]  # 127s, +5.8%
    better = [_para(61), _para(61)]  # 122s, +1.7%
    fake = _wm(first, better)
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 2, "the +4% steer corrected a draw the band accepted"
    assert out.actual_words == 122, "the strictly closer re-draw ships"
    assert out.narration_estimate is not None
    assert out.narration_estimate.reask_kept is True
    assert _advisories(out) == [], "an in-band outcome raises nothing"


@pytest.mark.asyncio
async def test_the_correction_states_the_countable_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second prompt names produced words, estimated s, ordered s and the target."""
    fake = _wm([_para(38) for _ in range(4)])
    await _run(monkeypatch, fake)

    corrections = _corrections(fake.prompts)
    assert len(corrections) == 1, f"expected exactly one bounded re-ask, got {len(corrections)}"
    (correction,) = corrections
    assert any("CORRECTION" not in p for p in fake.prompts), (
        "the opening draw must be asked WITHOUT a correction block"
    )
    assert "TOO LONG" in correction
    assert "152 words" in correction, "produced"
    assert "152 seconds" in correction, "MEASURED, not projected"
    assert "120 seconds" in correction, "ordered"
    assert "120 words" in correction, "target = budget_words(120s, 60wpm)"


@pytest.mark.asyncio
async def test_a_short_script_is_accepted_never_padded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short re-draw ships SHORT. No duplicated paragraphs, no filler.

    The short side is ASKED TWICE — the trim is long-only, so without a second ask
    a short draw would get one attempt where a long draw gets two. Both attempts
    here draw the same 40s script, so nothing improves and the stage ships exactly
    what it always shipped: two paragraphs, unpadded, with the advisory. Asking is
    not padding, and this test is the proof of that distinction.
    """
    fake = _wm([_para(20), _alt(20)])
    out = await _run(monkeypatch, fake)

    corrections = _corrections(fake.prompts)
    assert len(corrections) == 2, "short gets the same two attempts long always had"
    assert all("TOO SHORT" in c for c in corrections)
    assert all("Do NOT pad" in c for c in corrections)
    assert len(out.paragraphs) == 2, "accepted as drawn — nothing added"
    assert out.actual_words == 40
    assert out.total_seconds == 40.0
    texts = [p.text for p in out.paragraphs]
    assert len(set(texts)) == len(texts), "no paragraph was duplicated to fill time"
    assert out.narration_estimate is not None
    assert out.narration_estimate.verdict == "short"
    assert out.narration_estimate.trimmed_lines == 0, "nothing is trimmed on the short side"
    assert _advisories(out), "shipping short is recorded, not hidden"
    assert "40.0s" in _advisories(out)[0]["detail"]


@pytest.mark.asyncio
async def test_a_short_correction_quotes_the_draft_and_asks_for_the_missing_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The short re-ask is a DELTA over the best draft so far, never a from-scratch ask.

    Run 01a158a6-0f2c (2026-10-20, 60s ordered): two "rewrite to about N words"
    re-asks drew 115 and 127 words against asks of 164 and 151 and shipped -23.7%.
    A live replay of the delta form grew that 127-word draft to 159-164 (4/4).
    Against 120s at 60 wpm: 70 words first ⇒ 50 missing, 25 per paragraph; the
    100-word re-draw is then the draft the SECOND ask quotes ⇒ 20 missing, 10 each.
    """
    first = [_tagged(35, "alpha"), _tagged(35, "alpha")]
    reask_one = [_tagged(50, "beta"), _tagged(50, "beta")]
    fake = _wm(first, reask_one, reask_one)
    await _run(monkeypatch, fake)

    one, two = _corrections(fake.prompts)
    assert "about 50 MORE words" in one
    assert "EXPAND each one by about 25 words" in one
    assert "1. " + " ".join(["alpha"] * 35) in one, "the draft being extended is quoted"
    assert "about 20 MORE words" in two
    assert "2. " + " ".join(["beta"] * 50) in two, "the second ask extends the BEST draw"
    assert "alpha" not in two.split("CORRECTION", 1)[1]


@pytest.mark.asyncio
async def test_a_second_short_reask_that_lands_in_band_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The case the short side could not reach before: two asks land it in band.

    Against a 120s order: the opening draw is 70s (-41.7%), one re-ask reaches
    100s (-16.7%) — closer, kept, and STILL outside the +/-10% band. That is
    precisely where the three live 120s lessons stopped and shipped at -15.6%,
    -11.9% and -10.2%, because the only remaining attempt was a long-only trim.
    A second ask reaches 118s (-1.7%) and is kept, so the lesson ships CONFORMANT
    and raises no advisory.
    """
    first = [_para(35), _para(35)]
    reask_one = [_alt(50), _alt(50)]
    reask_two = [_para(59), _para(59)]
    fake = _wm(first, reask_one, reask_two)
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 3, "opening draw + two re-asks"
    corrections = _corrections(fake.prompts)
    assert len(corrections) == 2
    assert all("TOO SHORT" in c for c in corrections), "both asks steer the same way"

    assert out.total_seconds == 118.0, "the second ask's draw is the one that ships"
    assert out.narration_estimate is not None
    assert out.narration_estimate.verdict == "ok"
    assert out.narration_estimate.reasked is True
    assert out.narration_estimate.reask_kept is True
    assert out.narration_estimate.trimmed_lines == 0, "nothing was trimmed to get here"
    assert not _advisories(out), "a conformant lesson reports nothing"


@pytest.mark.asyncio
async def test_a_trim_that_would_overshoot_under_the_band_stops_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three 76-word paragraphs = 228s against a 120s order (band 108-132s).

    Dropping one lands at 152s (still long); dropping a second lands at 76s, which
    is -36.7% — the overcorrection this pipeline keeps re-committing. It must stop
    at 152s and report the floor rather than ship a two-thirds-empty lesson.
    """
    fake = _wm([_para(76) for _ in range(3)])
    out = await _run(monkeypatch, fake)

    assert len(out.paragraphs) == 2, "exactly one drop — the second was declined"
    assert out.actual_words == 152
    assert out.narration_estimate is not None
    assert out.narration_estimate.estimate_seconds == pytest.approx(152.0)
    assert out.narration_estimate.trimmed_lines == 1
    (miss,) = _advisories(out)
    assert "STILL LONG" in miss["detail"]
    (floor,) = _floors(out)
    assert "UNDER the band" in floor["detail"]


# --------------------------------------------------------------------------
# The CAST. A Socratic lesson is MULTI-VOICE: lesson_narrate casts each paragraph
# to its speaker's voice, and ``deterministic_clip_id`` hashes (voice, speed,
# text) — so the measurement must ask for exactly what NARRATE will ask for, or
# the run pays for every paragraph twice AND measures audio nobody will hear.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_paragraph_is_measured_at_the_voice_narrate_will_cast_it_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE clip-reuse contract, asserted against ``narrate.voice_for`` itself.

    ``lesson_narrate`` — not ``lesson_record`` — is the stage that synthesizes a
    lesson's audio (``lesson/compile.py`` reads its ``clips``), and
    ``deterministic_clip_id`` hashes that voice together with the speed and the
    text. Measure at any other voice, speed or stage name and two things break at
    once: the seconds describe audio nobody will hear, and narrate finds no stored
    clip under the id it looks up, so every paragraph is synthesized a second time
    at full price.
    """
    draw = [_asks(20), _para(20), _asks(20)]
    ctx = _cast_ctx()
    out, recorder = await _run_recorded(monkeypatch, _wm(draw), ctx, {STUDENT_VOICE: 60.0})

    speakers = [p.speaker for p in out.paragraphs]
    assert speakers == ["a", "b", "a"], "the derived two-hander really is mixed"
    assert recorder.voices == (STUDENT_VOICE, INSTRUCTOR_VOICE, STUDENT_VOICE), "the literal cast"
    assert _DEFAULT_CAST == {"a": STUDENT_VOICE, "b": INSTRUCTOR_VOICE}, "…lesson's own cast"

    # Not a copy of the rule: ask the NARRATE stage what it would cast these very
    # paragraphs to. A future edit to either side has to move both or this goes red.
    narrate = LessonNarrateStage(client=MagicMock())
    voice_block = ctx.config_snapshot["voice"]
    expected = tuple(
        narrate.voice_for({"speaker": p.speaker, "text": p.text}, voice_block, ctx)
        for p in out.paragraphs
    )
    assert recorder.voices == expected, "the script measured what narrate will speak"

    assert recorder.speeds[-1] is None, "narrate omits speed; the clip id hashes None"
    assert recorder.stage_names[-1] == LessonNarrateStage.name, "…and narrate's own name"


@pytest.mark.asyncio
async def test_a_slower_cast_voice_turns_an_ok_verdict_into_a_trim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE live lesson defect: budgeted at a voice that speaks a fifth of the script.

    60 student words + 3x40 instructor words is 180 words: at the CONFIGURED
    student voice's 90 wpm that is EXACTLY the 120s order, verdict "ok" — one LLM
    call, ship it, deliver long. Measured per paragraph at the voices that speak
    them the instructor's 60 wpm dominates: 40 + 3x40 = 160.0s (+33.3%), which
    trims to 120.0s.

    The two rates are chosen so the POOLED rate (67.5 wpm) stays inside
    ``wpm_is_usable``'s 60-300 band: a pool below the floor falls back to the
    word proxy at the substituted 150.0 default, which would make this test assert
    the fallback rather than the cast.
    """
    draw = [_asks(60), *[_para(40) for _ in range(3)]]
    fake = _wm(draw)
    out = await _run(monkeypatch, fake, _cast_ctx(), {STUDENT_VOICE: 90.0, INSTRUCTOR_VOICE: 60.0})

    assert fake.message.await_count == 2, "the cast made the draw long ⇒ one re-ask"
    estimate = out.narration_estimate
    assert estimate is not None
    assert estimate.first_draw_estimate_seconds == pytest.approx(160.0), (
        "a single-rate estimate would have called this 120.0s and returned"
    )
    assert estimate.estimate_seconds == pytest.approx(120.0), "the order, exactly"
    assert len(out.paragraphs) == 3, "one instructor paragraph dropped, the question kept"
    assert out.degradations == [] and _advisories(out) == []


@pytest.mark.asyncio
async def test_the_same_draw_single_voice_is_untouched_and_measures_one_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dialogue off ⇒ ONE voice ⇒ every paragraph measured at the configured voice."""
    ctx = _ctx({"dialogue": {"enabled": False}})
    ctx.config_snapshot["voice"] = {"descriptor": "calm", "voice_id": STUDENT_VOICE}
    draw = [_asks(60), *[_para(20) for _ in range(3)]]
    fake = _wm(draw)
    out, recorder = await _run_recorded(
        monkeypatch, fake, ctx, {STUDENT_VOICE: 60.0, INSTRUCTOR_VOICE: 30.0}
    )

    assert fake.message.await_count == 1, "120 words at 60 wpm is exactly the 120s order"
    assert len(out.paragraphs) == 4, "nothing was dropped"
    assert out.degradations == [] and _advisories(out) == []
    assert recorder.voices == (STUDENT_VOICE,) * 4, "dialogue off casts every paragraph to one"


@pytest.mark.asyncio
async def test_each_draw_is_judged_at_its_own_measured_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The re-ask changes the speaker mix, so it is measured on ITS OWN text.

    First draw: 40 student words (90 wpm ⇒ 26.7s) + 2x60 instructor words
    (60 wpm ⇒ 120.0s) = 146.7s, pooled 65.5 wpm. Re-ask: 150 student words
    (100.0s) + 20 instructor words (20.0s) = 120.0s — the order exactly, so it is
    kept. Graded at the FIRST draw's pooled 65.5 wpm the same 170 words read as
    155.8s, which is FURTHER from the order than the draft it would replace, and
    the re-draw would be discarded as an overcorrection that never happened.
    """
    first = [_asks(40), *[_para(60) for _ in range(2)]]
    reask = [_asks(150), _para(20)]
    fake = _wm(first, reask)
    out = await _run(monkeypatch, fake, _cast_ctx(), {STUDENT_VOICE: 90.0, INSTRUCTOR_VOICE: 60.0})

    assert fake.message.await_count == 2
    assert out.actual_words == 170, "the re-draw — closer at its OWN seconds — ships"
    estimate = out.narration_estimate
    assert estimate is not None
    assert estimate.first_draw_estimate_seconds == pytest.approx(146.667, rel=1e-3)
    assert estimate.estimate_seconds == pytest.approx(120.0)
    assert estimate.reask_kept is True
    assert _advisories(out) == [], "120.0s against 120s is inside the band"


# --------------------------------------------------------------------------
# Tiers. An estimate MISS grades audio nobody has heard, so it is ADVISORY; a
# rate or a measurement that did not land means every number is unsound, which
# is a DEGRADATION.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_estimate_miss_is_an_advisory_and_does_not_degrade_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-audio estimate must not make a good lesson read broken.

    TWO of the five live 120s runs delivered inside the band and were stamped
    ``duration_out_of_tolerance`` anyway, from this very estimate. The verdict that
    can be trusted is the one the compile stages raise from the real artifact.
    """
    out = await _run(monkeypatch, _wm([_para(20), _para(20)]))

    assert _advisories(out), "the miss is recorded"
    assert out.degradations == [], "…on the tier that does NOT flip the run degraded"
    assert all(d["code"] != NARRATION_ESTIMATE_OFF_TARGET for d in out.degradations)


@pytest.mark.asyncio
async def test_a_measurement_that_never_landed_is_degraded_and_said_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unmeasurable script is budgeted blind — a fault, not advice.

    And it is stated ONCE even though two drafts were measured through the same
    dead backend: a reader counting rows must not read "the backend failed twice"
    as a worse outage than "the backend was down".
    """
    stub_failed_measurement(monkeypatch, lesson_script, WPM, PROBE_UNREACHABLE)
    fake = _wm([_para(38) for _ in range(4)])
    out = await LessonScriptStage(client=fake).run({}, ctx=_ctx())

    assert fake.message.await_count == 2, "the re-ask ran — two measurements were refused"
    (entry,) = _rate_entries(out)
    assert PROBE_UNREACHABLE in entry["detail"]
    assert out.narration_estimate is not None
    assert out.narration_estimate.measured_lines == 0, "nothing came off a real synthesis"


@pytest.mark.asyncio
async def test_an_unusable_speech_rate_is_recorded_even_on_an_in_band_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rate outside the sanity band makes EVERY number below unsound.

    Reached the way production reaches it: the full-script measurement refuses, so
    the Phase-A prior is all there is — and here that prior is 0.0 wpm, which is
    not budgetable either. TWO independent faults, and both are recorded.
    """
    stub_failed_measurement(monkeypatch, lesson_script, 0.0, PROBE_UNREACHABLE)
    draw = [_para(150), _para(150)]  # 300 words = 120.0s at the 150.0 substitute
    out = await LessonScriptStage(client=_wm(draw)).run({}, ctx=_ctx())

    details = [d["detail"] for d in _rate_entries(out)]
    assert len(details) == 2, f"the refusal AND the unusable rate, got {details}"
    assert any(PROBE_UNREACHABLE in d for d in details), "the measurement did not land"
    assert any("0.0" in d and "150" in d for d in details), "and 150.0 was substituted"
    assert _advisories(out) == [], "at 150 wpm the draw IS in band — no duration miss"


# --------------------------------------------------------------------------
# The re-ask must NOT overcorrect, and must never be able to lose the run.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_reask_that_overcorrects_keeps_the_first_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First +11.7% (134 words), re-ask -30% (84 words) ⇒ the FIRST draw is kept.

    134s is long to the steer, so the trim is offered the first draw too — and
    declines, because dropping either 67s paragraph lands at 67.0s, far under the
    band. That floored trim is what earns the long side its second ask, so a THIRD
    call goes out; the fake repeats its 84-word draw, which is no closer than the
    134-word incumbent and is therefore DISCARDED. The +11.7% draw ships and the
    run SAYS it missed — the invariant this test exists for: a draw further from
    the order than the incumbent never ships, however many asks paid for it.
    """
    first = [_para(67), _para(67)]
    reask = [_para(42), _para(42)]
    fake = _wm(first, reask)
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 3, "the re-ask ran, then the floored-trim re-ask"
    assert out.actual_words == 134, "the closer first draw ships"
    assert out.narration_estimate is not None
    assert out.narration_estimate.reask_kept is False
    (miss,) = _advisories(out)
    assert "re-ask not closer; kept first draw" in miss["detail"]


@pytest.mark.asyncio
async def test_a_reask_tied_with_the_first_draw_keeps_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First -25% (90 words), re-ask +25% (150 words): equally far ⇒ first kept.

    Distinguished by TEXT, not by count: a tie that shipped the re-ask would be
    invisible to a word-count assertion.
    """
    first = [_para(45), _para(45)]
    reask = [_alt(75), _alt(75)]
    fake = _wm(first, reask)
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 2
    assert out.actual_words == 90, "a tie keeps the incumbent"
    assert all(p.text.startswith("word") for p in out.paragraphs), "the FIRST draw's text"
    (miss,) = _advisories(out)
    assert "re-ask not closer; kept first draw" in miss["detail"]


@pytest.mark.asyncio
async def test_a_failing_reask_keeps_the_first_draw_and_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A paid run must not die because the re-ask failed — trim the first draw."""
    fake = AsyncMock()
    calls = {"n": 0}

    async def _message(**_kwargs: Any) -> WMResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return _resp({"paragraphs": [_para(38) for _ in range(4)]})
        raise RuntimeError("worker-management is down")

    fake.message.side_effect = _message
    out = await _run(monkeypatch, fake)

    assert out.paragraphs, "the first draw survived the failed re-ask"
    assert out.actual_words == 114, "and was still trimmed into the band"
    assert out.degradations == [], "landing in band is not degraded, however it got there"


@pytest.mark.asyncio
async def test_an_empty_reask_keeps_the_first_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    """A corrective re-ask is parsed with no floor, so a bare ``[]`` re-draw is reachable.

    At 0.0s it would be "closer" to a 120s order than a 152s first draw and ship
    an empty lesson. The guard keeps the first draw.
    """
    fake = _wm([_para(38) for _ in range(4)], [])
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 2
    assert out.paragraphs, "an empty re-draw must never replace a real script"
    assert out.actual_words == 114, "first draw kept and trimmed"
    estimate = out.narration_estimate
    assert estimate is not None
    assert (estimate.reasked, estimate.reask_kept) == (True, False), "sent, and discarded"


@pytest.mark.asyncio
async def test_a_broken_reconciliation_never_fails_the_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconciliation is on a paid run's happy path: it degrades, it never raises."""
    monkeypatch.setattr(
        lesson_script,
        "steer_verdict",
        lambda *_a, **_k: (_ for _ in ()).throw(ZeroDivisionError("boom")),
    )
    out = await _run(monkeypatch, _wm([_para(38) for _ in range(4)]))

    assert len(out.paragraphs) == 4, "the unreconciled script ships rather than the run failing"
    assert out.degradations == []
    assert out.narration_estimate is None, "and it claims no measurement it never finished"


@pytest.mark.asyncio
async def test_the_correction_does_not_leak_into_a_later_first_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stage instance is reused by tooling; the correction is per-re-ask."""
    stub_speech_rate(monkeypatch, lesson_script, WPM)
    fake = _wm([_para(38) for _ in range(4)], [_para(120)])
    stage = LessonScriptStage(client=fake)

    await stage.run({}, ctx=_ctx())
    assert _corrections(fake.prompts), "the first run did not re-ask — nothing to leak"
    fake.prompts.clear()  # segment the record, so "first prompt" means the NEW run's
    await stage.run({}, ctx=_ctx())

    assert fake.prompts, "the second run never asked for a draw"
    assert "CORRECTION" not in fake.prompts[0], (
        "the stashed correction leaked into a later run's FIRST draw"
    )


# --------------------------------------------------------------------------
# Verbatim: the customer's own words outrank the ordered duration.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_verbatim_source_script_bypass_is_never_reconciled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Those paragraphs are the CUSTOMER'S OWN WORDS — trimming them deletes text
    they supplied verbatim, and no LLM call happens on that path at all."""
    source = " ".join(f"w{i}" for i in range(600))
    fake = _wm([_para(1)])
    out = await _run(monkeypatch, fake, _ctx({"source_script": source}))

    assert fake.message.await_count == 0
    assert out.actual_words == 0, "the bypass does not compute the budget fields"
    assert sum(len(p.text.split()) for p in out.paragraphs) == 600
    assert out.degradations == []
    assert out.narration_estimate is None, "nothing was measured, and it does not pretend"


@pytest.mark.asyncio
async def test_a_trim_never_drops_the_paragraph_carrying_a_required_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``assert_verbatim_present`` gates the DRAW; the trim must not undo that gate.

    Sized so the trim wants the required paragraph: it is the LAST one, and
    ``plan_trim`` drops last-first. The phrase survives, and the drop the plan
    takes instead is an ordinary paragraph.
    """
    required = "Remember drink water " + " ".join(["word"] * 27)  # 30 words
    draw = [
        *[_para(30) for _ in range(4)],
        {"text": required, "seconds": 30.0, "speaker": "b"},
    ]
    ctx = _ctx()
    ctx.config_snapshot["verbatim_items"] = [{"text": "drink water"}]
    out = await _run(monkeypatch, _wm(draw), ctx)

    texts = [p.text for p in out.paragraphs]
    assert any("drink water" in t for t in texts), "the required phrase survived the trim"
    assert len(out.paragraphs) == 4, "and the trim still ran — it dropped an ordinary paragraph"
    assert out.narration_estimate is not None
    assert out.narration_estimate.estimate_seconds == pytest.approx(120.0)


@pytest.mark.asyncio
async def test_the_trim_plans_on_measured_seconds_not_on_the_word_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two metrics choose DIFFERENT paragraphs, and the measured one is right.

    Three paragraphs, each 60.0s of real audio but wildly unequal in words because
    the student's voice is three times the instructor's: 60 instructor words,
    180 student words, 60 instructor words = 180.0s against a 120s order.

    * On MEASURED seconds every paragraph weighs the same 60.0s, so dropping ONE
      lands on 120.0s exactly — the fewest-drops answer, cutting the latest.
    * On the WORD proxy at the pooled 100 wpm they weigh 36s / 108s / 36s, no
      single drop lands in the band, and the plan takes BOTH instructor
      paragraphs to reach 108.0s — shipping a lesson that is nothing but the
      student's questions.

    So this is not "the same answer computed twice": the word proxy deletes the
    teaching. It is the assertion that survives ``line_seconds=None``.
    """
    draw = [_para(60), _asks(180), _alt(60)]
    fake = _wm(draw)
    out = await _run(monkeypatch, fake, _cast_ctx(), {STUDENT_VOICE: 180.0, INSTRUCTOR_VOICE: 60.0})

    assert fake.message.await_count == 2, "180.0s is long ⇒ one re-ask, which is no closer"
    estimate = out.narration_estimate
    assert estimate is not None
    assert estimate.first_draw_estimate_seconds == pytest.approx(180.0)
    assert estimate.estimate_seconds == pytest.approx(120.0), "the order, exactly"
    assert estimate.trimmed_lines == 1, "ONE drop — the word proxy needs two"
    texts = [p.text for p in out.paragraphs]
    assert len(texts) == 2
    assert texts[0].startswith("word "), "the first instructor paragraph survived"
    assert texts[1].endswith("word?"), "…and so did the student's question"
    assert out.degradations == [] and _advisories(out) == []


@pytest.mark.asyncio
async def test_the_trim_plans_at_the_shipped_draws_own_rate_not_the_last_measured_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``self._wpm`` holds the LAST measurement — which can be a discarded draw's.

    Reaching this needs the word proxy, because ``plan_trim`` ignores the rate
    entirely while it has measured ``paragraph_seconds``. So paragraph 3 of the
    first draft comes back NaN: ``_trimmable_seconds`` refuses it (that is what it
    is for) and the plan falls back to words x rate — at which point WHICH rate
    decides which paragraphs are cut.

    The first draft is measured at 60 wpm, the re-asked one at 240, and the
    re-asked one is discarded for landing 420.0s from a 120s order. At the shipped
    draw's own 60 wpm its 240 words are 240.0s and two paragraphs must go to reach
    the order exactly. At the discarded draw's 240 wpm the very same words read as
    "60.0s" — already UNDER the band — so that trim declines, the only trim left is
    of the 420.0s draft nobody kept, and a single 140.0s paragraph of it ships.
    """
    prior = SpeechRate(wpm=WPM, measured=True)
    monkeypatch.setattr(
        speech_rate_module, "resolve_natural_wpm_detail", AsyncMock(return_value=prior)
    )
    calls: list[int] = []

    async def _measure(_ctx: Any, *, texts: Any, **_kwargs: Any) -> ScriptMeasurement:
        calls.append(len(texts))
        if len(calls) == 1:
            return ScriptMeasurement(
                per_line_seconds=(60.0, 60.0, 60.0, float("nan")),
                measured=(True, True, True, False),
                failed_indices=(3,),
                extrapolated_indices=(3,),
                total_seconds=180.0,
                measured_count=3,
                wpm=WPM,
                basis=BASIS_MEASURED_ALL,
                prior=prior,
            )
        return ScriptMeasurement(
            per_line_seconds=tuple(420.0 / len(texts) for _ in texts),
            measured=(True,) * len(texts),
            failed_indices=(),
            extrapolated_indices=(),
            total_seconds=420.0,
            measured_count=len(texts),
            wpm=240.0,
            basis=BASIS_MEASURED_ALL,
            prior=prior,
        )

    monkeypatch.setattr(lesson_script, "measure_script_lines", _measure)
    #: A lesson protects nothing structurally and no verbatim item is configured,
    #: so both paragraphs the plan wants to drop are droppable.
    fake = _wm([_tagged(60, "a") for _ in range(4)], [_tagged(80, "b") for _ in range(3)])

    out = await LessonScriptStage(client=fake).run({}, ctx=_ctx())

    assert calls == [4, 3], "both draws were measured; the 240 wpm one is the re-ask"
    assert all(p.text.startswith("a") for p in out.paragraphs), "the 240 wpm draw was discarded"
    assert len(out.paragraphs) == 2, "planned at 60 wpm: 240 words is 240.0s and two must go"
    assert out.actual_words == 120
    assert out.narration_estimate is not None
    assert out.narration_estimate.wpm == pytest.approx(WPM), "the SHIPPED draw's own rate"
    assert out.narration_estimate.estimate_seconds == pytest.approx(ORDERED_SECONDS)
    assert _advisories(out) == [], "the trim landed on the order"
    assert _floors(out) == [], "and it was never floored"


# --------------------------------------------------------------------------
# The LONG side's second attempt: when the trim is a NO-OP, ask again with a
# number. Live run 01a0a34a-b050-753c-93a3-fefb5dc6fa99 shipped +14.85% because
# there was nothing left after a trim that could not act.
# --------------------------------------------------------------------------

#: The rate measured on the live run — NOT the round 60.0 the rest of this file
#: uses, because the whole point of the case is that ``plan_trim``'s subset-sum
#: over these four paragraph lengths at THIS rate has no member inside the band.
LIVE_WPM = 148.4814
#: The live re-asked draft: 85/79/84/93 words = 341 words = 137.795s (+14.85%)
#: against the 120s order. Band is [108.0, 132.0]; the fifteen non-empty subsets
#: reach 100.21 / 103.45 / 103.85 / 105.87s and up — the best single drop lands
#: 2.13s UNDER the floor, so the trim declines every one of them.
LIVE_LONG_PARAGRAPHS = (85, 79, 84, 93)


def _live(words: int, tag: str) -> dict[str, Any]:
    return {"text": " ".join([tag] * words), "seconds": float(words), "speaker": "b"}


@pytest.mark.asyncio
async def test_a_long_draw_the_trim_cannot_touch_is_re_asked_at_an_explicit_word_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live +14.85% lesson: trim floored with NOTHING dropped, so ask again.

    Reproduces run ``01a0a34a-b050-753c-93a3-fefb5dc6fa99`` exactly. First draw
    212 words (~85.7s, short) → the re-ask overcorrects to 341 words (137.795s,
    +14.85%, "long" and CLOSER so it is kept) → ``plan_trim`` runs an exhaustive
    subset-sum over 85/79/84/93 words at 148.4814 wpm and finds no subset inside
    [108.0, 132.0]: ``declined_under_band``, ``dropped_indices=()``,
    ``trimmed_lines=0``. Before this, that was the end of the long side and the
    stage shipped +14.85%.

    Now the floored long draw earns the same second attempt the short side got in
    (1b) — one re-ask carrying the WORD COUNT the order affords at THIS draw's own
    measured rate: ``round(120.0 * 148.4814 / 60) = 297`` words, against the 341
    actually written. The rewritten draw is closer, so it is kept and ships.
    """
    stub_speech_rate(monkeypatch, lesson_script, LIVE_WPM)
    first = [_live(53, "short"), _live(53, "short"), _live(53, "short"), _live(53, "short")]
    overshot = [_live(n, "long") for n in LIVE_LONG_PARAGRAPHS]
    corrected = [_live(75, "fix"), _live(74, "fix"), _live(74, "fix"), _live(74, "fix")]
    fake = _wm(first, overshot, corrected)

    out = await LessonScriptStage(client=fake).run({}, ctx=_ctx())

    assert fake.message.await_count == 3, "opening draw + the steer re-ask + the floored re-ask"
    corrections = _corrections(fake.prompts)
    assert len(corrections) == 2, "the long side now gets two attempts, as the short side does"
    assert "TOO SHORT" in corrections[0], "the first draw was short"
    # Slice to the CORRECTION block: the base prompt's own word-budget sentence
    # ALSO names a word count, so a bare ``"297 words" in prompt`` is satisfied by
    # a prompt carrying no target at all. Mutating the target to 298 proved that.
    final = "CORRECTION" + corrections[1].split("CORRECTION")[-1]
    assert "TOO LONG" in final
    assert "It contained 341 words" in final, "what the floored draw actually contained"
    assert "total about 297 words" in final, "the target = round(120.0 * 148.4814 / 60)"

    estimate = out.narration_estimate
    assert estimate is not None
    assert estimate.first_draw_estimate_seconds == pytest.approx(85.67, abs=0.05)
    assert estimate.estimate_seconds != pytest.approx(137.795, abs=0.01), (
        "the +14.85% draw the trim could not touch must NOT be what ships"
    )
    assert estimate.estimate_seconds == pytest.approx(119.99, abs=0.05), "the 297-word rewrite"
    assert estimate.verdict == "ok"
    assert estimate.trimmed_lines == 0, "the trim dropped nothing — that is why this fired"
    assert all(p.text.startswith("fix") for p in out.paragraphs), "the corrected draw shipped"
    assert _advisories(out) == [], "it landed in band, so nothing is reported"
    assert _floors(out) == [], "a floor that no longer describes the shipped script is cleared"


@pytest.mark.asyncio
async def test_a_trim_that_dropped_something_is_never_pre_empted_by_a_paid_re_ask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three 76-word paragraphs: the trim DROPS one and is still long — no third ask.

    The free deterministic lever acted. Spending an LLM call on top of it is the
    exact property (1b)'s short-side guard protects, mirrored: the new long-side
    re-ask fires ONLY when ``trimmed_lines == 0``.
    """
    fake = _wm([_para(76) for _ in range(3)])
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 2, "opening draw + ONE re-ask; the trim did the rest"
    assert len(_corrections(fake.prompts)) == 1
    assert out.narration_estimate is not None
    assert out.narration_estimate.trimmed_lines == 1
    assert _floors(out), "the floor is still reported — the trimmed script is what ships"
