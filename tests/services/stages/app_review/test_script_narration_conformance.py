"""app_review_script RECONCILES its narration against the ordered duration.

Measured live 2026-09-10: an app_review ordered at 120s delivered 132.93s
(+10.8%) — flagged by ``app_review/compile.py`` and then corrected by nobody.
Delivered length IS the sum of the TTS clip seconds, so the only place the order
can actually be honoured is the script.

app_review's miss is MARGINAL, which makes overcorrection the real hazard here:
demo_video once swung +56.8% -> -33.5% and lesson +216% -> -20.4% applying
"fixes" that were never checked for improvement. So every test below pins a
DIRECTION as well as a change, and the within-band case pins that nothing
happens at all.

Measured on five live runs per type (2026-09-12) the stage then delivered
-20.84%, -9.35%, +2.96%, +12.06%, +12.87% against a 120s order, with the
ESTIMATE itself accurate to a few percent. Three defects, all fixed here and all
pinned below:

* it aimed at the BAND EDGE — anything inside +/-10% was "ok", so a draw was
  accepted at +9% with nothing left for the estimate's own error. The decision to
  correct is now ``steer_verdict`` (+4% on the long side);
* the single re-ask only asked whether a draw was CLOSER, never whether it was IN
  BAND: run ``01a09751`` went +34% -> -18.4%, "closer", and delivered -20.84%;
* a draw was judged at the rate SAMPLED OFF THE DRAFT IT REPLACED.

And the estimate MISS moved to ``advisories``: it is a pre-audio number, and the
real verdict is the one ``app_review_compile`` raises off the artifact. What
stays on ``degradations`` is an unusable MEASUREMENT — a real fault.
"""

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

import src.services.stages.app_review.script as app_review_script
import src.services.stages.shared.speech_rate as speech_rate_module
from src.services.runs.run_context import RunContext
from src.services.stages.app_review.script import AppReviewScriptStage
from src.services.stages.shared.degradation import (
    NARRATION_ESTIMATE_OFF_TARGET,
    ORDERED_DURATION_INFEASIBLE,
    SPEECH_RATE_UNUSABLE,
    is_degraded,
)
from src.services.stages.shared.narration_budget import estimate_seconds
from src.services.stages.shared.speech_rate import (
    BASIS_MEASURED_ALL,
    RESOLUTION_ERROR,
    ScriptMeasurement,
    SpeechRate,
)
from tests.services.stages.app_review.rate_stubs import (
    measurement_at,
    stub_failed_measurement,
    stub_speech_rate,
)

#: A round rate so every word count in this file is exactly 0.5s of speech.
WPM = 120.0
ORDERED_SECONDS = 120
#: 120s at 120 wpm.
BUDGET_WORDS = 240


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


def _draw(*word_counts: int, tag: str = "w") -> WMResponse:
    """One WM reply whose lines carry exactly ``word_counts`` words each."""
    return _resp(
        {
            "lines": [
                {"text": _words(n, tag=f"{tag}{idx}_"), "seconds": n / 2.0}
                for idx, n in enumerate(word_counts)
            ]
        }
    )


def _ctx(brief: dict | None = None, verbatim: list[dict] | None = None) -> RunContext:
    snapshot: dict[str, Any] = {
        "brief": dict(brief or {"duration_seconds": ORDERED_SECONDS}),
        "voice": {"descriptor": "neutral"},
    }
    if verbatim is not None:
        snapshot["verbatim_items"] = verbatim
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=snapshot,
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
    """A WM fake that RECORDS every outbound prompt on ``fake.prompts``.

    The record is what the assertions read: a prompt is found by the marker that
    IDENTIFIES it ("REVISION REQUIRED"), never by its position in the call list.
    Position is not a property of this stage — a correct refactor may reorder or
    add calls, and an incorrect one can keep the order perfectly.
    """
    fake = AsyncMock()
    queue = list(replies)
    prompts: list[str] = []
    fake.prompts = prompts

    async def _message(**kwargs: Any) -> Any:
        prompts.append(str(kwargs.get("message", "")))
        if not queue:
            raise AssertionError("the stage asked for more draws than the test scripted")
        reply = queue.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    fake.message.side_effect = _message
    return fake


def _reasks(fake: AsyncMock) -> list[str]:
    """Every recorded prompt that carries the corrective block — found by content."""
    return [p for p in fake.prompts if "REVISION REQUIRED" in p]


async def _run(
    monkeypatch: pytest.MonkeyPatch, fake: AsyncMock, ctx: RunContext | None = None
) -> Any:
    stub_speech_rate(monkeypatch, app_review_script, WPM)
    return await AppReviewScriptStage(client=fake).run({}, ctx=ctx or _ctx())


def _deviation(out: Any) -> float:
    estimated = estimate_seconds([line.text for line in out.lines], words_per_minute=WPM)
    return (estimated - ORDERED_SECONDS) / ORDERED_SECONDS


@pytest.mark.asyncio
async def test_a_marginally_long_draw_is_re_asked_once_and_lands_inside_the_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """+10.8% (the LIVE miss) ⇒ exactly one re-ask, and the reply is adopted."""
    long_draw = _draw(70, 70, 70, 56, tag="a")  # 266 words = 133s = +10.8%
    good_draw = _draw(60, 60, 60, 60, tag="b")  # 240 words = 120s = 0%
    fake = _wm(long_draw, good_draw)

    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 2, "exactly ONE bounded re-ask"
    assert out.actual_words == BUDGET_WORDS
    assert abs(_deviation(out)) <= 0.10
    assert all(line.text.startswith("b") for line in out.lines), "the re-ask reply shipped"


@pytest.mark.asyncio
async def test_the_re_ask_prompt_states_the_counts_and_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model cannot fix a length it is not told — produced/estimated/ordered/target."""
    fake = _wm(_draw(70, 70, 70, 56, tag="a"), _draw(60, 60, 60, 60, tag="b"))

    await _run(monkeypatch, fake)

    reasks = _reasks(fake)
    assert len(reasks) == 1, f"expected exactly one corrective re-ask, got {len(reasks)}"
    (reask,) = reasks
    assert any("REVISION REQUIRED" not in p for p in fake.prompts), (
        "the opening draw must be asked WITHOUT a correction"
    )
    assert "266 spoken words" in reask  # produced
    assert "133 seconds" in reask  # estimated
    assert "120 seconds" in reask  # ordered
    assert f"{BUDGET_WORDS} words" in reask  # target from budget_words
    assert "TOO LONG" in reask


@pytest.mark.asyncio
async def test_a_draw_already_on_target_is_left_completely_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE overcorrection guard: +1.7% must cost no LLM call and no TTS beyond the
    measurement every run takes anyway — no re-ask, no trim, nothing recorded."""
    inside = _draw(61, 61, 61, 61, tag="c")  # 244 words = 122s = +1.7%
    fake = _wm(inside)

    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 1, "an on-target draw must not cost a second LLM call"
    assert out.degradations == []
    assert out.advisories == [], "a conformant film raises nothing, on either tier"
    assert len(out.lines) == 4
    assert out.actual_words == 244
    assert out.total_seconds == pytest.approx(122.0)
    assert out.narration_estimate is not None, "recorded on CLEAN runs too"
    assert out.narration_estimate.reasked is False
    assert out.narration_estimate.reask_kept is None, "None, never False — there was no re-ask"
    assert out.narration_estimate.trimmed_lines == 0


@pytest.mark.asyncio
async def test_a_draw_at_plus_six_percent_is_steered_not_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect the live +12.06% and +12.87% runs came from.

    +6% is comfortably inside the +/-10% band that the OLD gate accepted — and a
    draw accepted at +6% has 4 points of headroom left for every error downstream
    of the estimate. The steer aims at the ORDER, so this is corrected.
    """
    marginal = _draw(64, 64, 64, 63, tag="c")  # 255 words = 127.5s = +6.25%
    on_target = _draw(60, 60, 60, 60, tag="d")  # 240 words = 120s = 0%
    fake = _wm(marginal, on_target)

    out = await _run(monkeypatch, fake)

    assert len(_reasks(fake)) == 1, "the OLD +/-10% gate would have shipped this untouched"
    assert all(line.text.startswith("d") for line in out.lines)
    assert _deviation(out) == pytest.approx(0.0)
    assert out.advisories == [], "steered INTO the band, so nothing to report"


@pytest.mark.asyncio
async def test_a_re_ask_that_overshoots_is_discarded_and_the_marginal_draw_ships(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A +10.8% miss must never be "fixed" into a -37.5% miss."""
    marginal = _draw(70, 70, 70, 56, tag="a")  # +10.8%
    overshoot = _draw(50, 50, 50, tag="b")  # 150 words = 75s = -37.5%
    fake = _wm(marginal, overshoot)

    out = await _run(monkeypatch, fake)

    assert all(line.text.startswith("a") for line in out.lines), "kept the closer draw"
    assert out.actual_words == 266
    assert out.advisories[0]["detail"].endswith("final 133.0s.")
    assert "re-ask not closer" in out.advisories[0]["detail"]


@pytest.mark.asyncio
async def test_a_short_draw_is_accepted_and_never_padded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short is reported, not repaired: no duplicated lines, no invented words.

    The re-ask may ASK for more material — that is honest authorship — but nothing
    mechanical lengthens a short script, and when the re-ask comes back shorter
    still the short draw ships and SAYS SO. Repeated audio is a worse defect than
    a short film.
    """
    short = _draw(45, 45, 45, tag="a")  # 135 words = 67.5s = -43.8%
    shorter = _draw(20, 20, tag="b")  # 40 words = 20s
    fake = _wm(short, shorter)

    out = await _run(monkeypatch, fake)

    assert out.actual_words == 135, "not padded up towards the 240-word budget"
    assert len(out.lines) == 3
    assert len({line.text for line in out.lines}) == 3, "no line duplicated to fill time"
    assert _deviation(out) < 0.0
    assert "TOO SHORT" in _reasks(fake)[0], "it asked for more words; it did not invent any"
    # It SAYS SO — and on the tier that does not call a short script a bad delivery.
    assert out.advisories[0]["code"] == NARRATION_ESTIMATE_OFF_TARGET
    assert "67.5s" in out.advisories[0]["detail"]
    assert out.degradations == []
    assert not is_degraded(out.model_dump()), "a pre-audio estimate never degrades the run"


@pytest.mark.asyncio
async def test_a_failing_re_ask_never_raises_and_keeps_the_first_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This path sits on a PAID run's happy path — it must never fail the stage."""
    fake = _wm(_draw(70, 70, 70, 56, tag="a"), RuntimeError("worker-management exploded"))

    out = await _run(monkeypatch, fake)

    assert out.actual_words == 266
    assert all(line.text.startswith("a") for line in out.lines)
    assert out.advisories[0]["code"] == NARRATION_ESTIMATE_OFF_TARGET
    assert "kept first draw" in out.advisories[0]["detail"]


@pytest.mark.asyncio
async def test_a_still_long_re_ask_is_trimmed_whole_lines_from_the_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Last resort: drop whole lines, tail-first, only while it moves CLOSER."""
    very_long = _draw(120, 120, 120, 120, tag="a")  # 480 words = 240s = +100%
    still_long = _draw(120, 120, 120, tag="b")  # 360 words = 180s = +50%
    fake = _wm(very_long, still_long)

    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 2, "a trim that ACTED is never pre-empted by a paid ask"
    assert all(line.text.startswith("b") for line in out.lines)
    assert len(out.lines) == 2, "one tail line dropped: 240 words = 120s"
    assert out.actual_words == 240
    assert out.total_seconds == pytest.approx(120.0)
    assert abs(_deviation(out)) <= 0.10
    # Nothing recorded: the trim landed INSIDE the band. The entry used to be
    # appended on any non-`ok` first estimate, i.e. on successes too.
    assert out.degradations == []
    assert out.advisories == []
    assert out.narration_estimate is not None
    assert out.narration_estimate.trimmed_lines == 1, "and the record says what it cost"


@pytest.mark.asyncio
async def test_a_trim_never_drops_a_required_verbatim_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The base already asserted the phrase on the FULL draw; the trim must not undo it.

    The phrase sits on the LAST line — exactly the line ``plan_trim`` reaches
    first — so the trim still has to happen, just around it.
    """
    phrase = "ships on every platform"
    long_first = _resp(
        {
            "lines": [
                {"text": _words(240, tag="x"), "seconds": 120.0},
                {"text": _words(60, tag="y"), "seconds": 30.0},
                {"text": phrase, "seconds": 2.0},  # 304 words = 152s = +26.7%
            ]
        }
    )
    fake = _wm(long_first, long_first)
    ctx = _ctx(verbatim=[{"text": phrase, "stages": ["app_review_script"]}])

    out = await _run(monkeypatch, fake, ctx)

    assert any(phrase in line.text for line in out.lines), "the user's exact words survived"
    assert len(out.lines) == 2, "the trim happened AROUND the protected line, not instead of it"
    assert out.lines[-1].text == phrase


@pytest.mark.asyncio
async def test_a_trim_is_abandoned_when_a_phrase_spans_two_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The case per-line protection cannot see.

    ``assert_verbatim_present`` matches against the JOINED text, so a phrase can
    be satisfied by two adjacent lines and satisfied by NEITHER alone — no index
    looks protected, and the only thing standing between the trim and a lost
    phrase is the re-check on the trimmed result. Here the trim would otherwise
    be adopted (150s is closer to 120s than 154s is), so it must be abandoned
    whole: shipping a long film beats shipping one missing the user's words.
    """
    phrase = "ships on every platform"
    draw = _resp(
        {
            "lines": [
                {"text": _words(300, tag="x"), "seconds": 150.0},
                {"text": "ships on", "seconds": 1.0},
                {"text": "every platform", "seconds": 1.0},
                {"text": _words(4, tag="z"), "seconds": 2.0},  # 308 words = 154s
            ]
        }
    )
    fake = _wm(draw, draw)
    ctx = _ctx(verbatim=[{"text": phrase, "stages": ["app_review_script"]}])

    out = await _run(monkeypatch, fake, ctx)

    joined = " ".join(line.text for line in out.lines)
    assert phrase in joined, "the phrase survived across the two lines that carry it"
    assert len(out.lines) == 4, "the whole trim was abandoned, not applied half-way"


@pytest.mark.asyncio
async def test_broken_conformance_arithmetic_never_fails_the_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The outer guard. Whatever breaks in here, the model's draw still ships."""
    monkeypatch.setattr(
        app_review_script,
        "narration_verdict",
        lambda *a, **k: (_ for _ in ()).throw(ZeroDivisionError("boom")),
    )
    fake = _wm(_draw(70, 70, 70, 56, tag="a"))

    out = await _run(monkeypatch, fake)

    assert out.actual_words == 266
    assert len(out.lines) == 4
    assert out.degradations == []


@pytest.mark.asyncio
async def test_an_untrimmable_overrun_reports_the_floor_not_a_clean_trim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single 480-word line cannot be dropped (never an empty script).

    The trim runs, reaches nothing, and the record must say so: "+ trimmed" (or
    its absence) alone reads exactly like a trim that landed in the band.
    """
    # The floored trim earns ONE more re-ask at an explicit word target (step 2b);
    # scripted here as another untrimmable line so the floor still ships.
    fake = _wm(_draw(480, tag="a"), _draw(480, tag="b"), _draw(480, tag="c"))
    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 3, "draw + steer re-ask + floored-trim re-ask"
    assert len(out.lines) == 1, "the last surviving line is never dropped"
    assert out.actual_words == 480
    assert [d["code"] for d in out.advisories] == [NARRATION_ESTIMATE_OFF_TARGET], "the miss"
    codes = [d["code"] for d in out.degradations]
    assert ORDERED_DURATION_INFEASIBLE in codes, "and the FLOOR — which IS a fault"
    miss = out.advisories[0]
    assert "STILL LONG" in miss["detail"], "the action string differs from a success"
    floor = next(d for d in out.degradations if d["code"] == ORDERED_DURATION_INFEASIBLE)
    assert "240.0s" in floor["detail"], "the estimate it floored at"
    assert f"{float(ORDERED_SECONDS):.1f}s" in floor["detail"], "the ordered seconds"
    assert "not reachable by trimming" in floor["detail"]


@pytest.mark.asyncio
async def test_a_verbatim_rejected_trim_is_never_reported_as_a_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ONE coherent story per run: a plan that was not applied did not happen.

    The phrase spans two lines, so no index looks protected and the plan drops
    both of them plus nothing else — and the re-check then refuses the whole
    trim. The script that SHIPS is the untouched 82.0s draw. Reporting the plan's
    floor alongside that put two contradicting entries on one run: "final 82.0s"
    next to "trim FLOORED at 80.0s — 2 line(s) dropped", with zero lines dropped.
    """
    phrase = "ships on every platform"
    draw = _resp(
        {
            "lines": [
                {"text": _words(160, tag="x"), "seconds": 80.0},
                {"text": "ships on", "seconds": 1.0},
                {"text": "every platform", "seconds": 1.0},  # 164 words = 82.0s
            ]
        }
    )
    fake = _wm(draw, draw)
    ctx = _ctx(
        brief={"duration_seconds": 60},
        verbatim=[{"text": phrase, "stages": ["app_review_script"]}],
    )

    out = await _run(monkeypatch, fake, ctx)

    assert len(out.lines) == 3, "the trim was refused — nothing was dropped"
    assert phrase in " ".join(line.text for line in out.lines)
    assert out.degradations == [], "a trim that never ran has no floor"
    assert [d["code"] for d in out.advisories] == [NARRATION_ESTIMATE_OFF_TARGET], "just the miss"
    detail = out.advisories[0]["detail"]
    assert "final 82.0s" in detail, "the length that actually shipped"
    assert "80.0" not in detail, "the un-applied plan's number is nowhere on the record"
    assert "trim REJECTED — it would have dropped a required verbatim phrase" in detail
    # Not vacuous: the plan really did want to drop those two lines.
    assert "STILL LONG" not in detail, "a refusal is not a floor"


@pytest.mark.asyncio
async def test_a_trim_the_closer_guard_refuses_is_reported_as_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other rejection door, forced open.

    ``plan_trim``'s lower-bound guard makes a real draw that is both trimmed and
    NOT closer unreachable (an accepted drop lands at or above ``ordered x 0.9``
    while the draw was above ``ordered x 1.1``), so ``closer_to_order`` is stubbed to
    refuse. What is under test is that the REPORT follows whether the trim was
    ADOPTED — not whether a plan was computed: a refused trim leaves the draw
    intact, so it must not be dressed up as a floored one.
    """
    monkeypatch.setattr(app_review_script, "closer_to_order", lambda *a, **k: False)
    fake = _wm(_draw(120, 120, 120, 120, tag="a"), _draw(120, 120, 120, tag="b"))

    out = await _run(monkeypatch, fake)

    assert all(line.text.startswith("a") for line in out.lines), "the refused draw shipped"
    assert len(out.lines) == 4, "and the refused trim dropped nothing"
    assert ORDERED_DURATION_INFEASIBLE not in [d["code"] for d in out.degradations]
    detail = out.advisories[0]["detail"]
    assert "final 240.0s" in detail
    assert "trim REJECTED — it landed no closer to the ordered duration" in detail


@pytest.mark.asyncio
async def test_a_closer_but_out_of_band_re_ask_does_not_end_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE -20.84% shape, from live run ``01a09751``, and the reason it shipped.

    A +34% first draft was re-asked to -18.4%. That IS closer, so the old loop
    adopted it, found nothing left to do (nothing shortens a short script), and
    delivered -20.84% — a correction that swapped one miss for a bigger one on the
    other side.

    What changes: "closer" no longer ends the loop, only "in band" does. The
    re-asked draw still wins on distance, but the +34% draft is still IN THE POOL
    and it is TRIMMABLE — and a trim costs no LLM call and no synthesis, because
    it selects lines whose seconds are already measured. One dropped line puts it
    exactly on the order, which beats -18.75%, so THAT is what ships.
    """
    overrun = _draw(80, 80, 80, 82, tag="a")  # 322 words = 161.0s = +34.2%
    overshoot = _draw(65, 65, 65, tag="b")  # 195 words = 97.5s = -18.75%
    fake = _wm(overrun, overshoot)

    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 2, "still exactly ONE re-ask — the budget is unchanged"
    assert abs(_deviation(out)) <= 0.10, "the delivered estimate is IN BAND, not merely closer"
    assert all(line.text.startswith("a") for line in out.lines), (
        "the trimmed first draft beat the re-asked one; neither miss shipped"
    )
    assert len(out.lines) == 3, "one whole line dropped"
    assert out.total_seconds == pytest.approx(120.0)
    assert out.advisories == [], "in band ⇒ nothing to report"
    estimate = out.narration_estimate
    assert estimate is not None
    assert estimate.first_draw_estimate_seconds == pytest.approx(161.0)
    assert estimate.estimate_seconds == pytest.approx(120.0)
    assert estimate.reasked is True
    assert estimate.reask_kept is True, "it WAS adopted — and then beaten by the trim"
    assert estimate.trimmed_lines == 1


@pytest.mark.asyncio
async def test_each_draw_is_judged_at_its_own_measured_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-asked draft is NEW TEXT and is measured as such.

    The first draft here is slow (60 wpm) and the re-asked one is fast (240 wpm).
    Judged at the FIRST draft's rate the re-asked 240-word draw estimates at
    240.0s — twice the order, "further away", discarded. Measured on its own it is
    60.0s... also out of band, but SHORT, which is a different decision entirely.
    Judged correctly the first draw (300 words at 60 wpm = 300.0s) is the worse of
    the two and must not survive.
    """
    slow = _draw(100, 100, 100, tag="a")  # 300 words
    fast = _draw(60, 60, 60, 60, tag="b")  # 240 words
    rates = iter([60.0, 240.0])
    prior = SpeechRate(wpm=60.0, measured=True)

    monkeypatch.setattr(
        speech_rate_module, "resolve_natural_wpm_detail", AsyncMock(return_value=prior)
    )

    async def _measure(_ctx: Any, *, texts: Any, **_kwargs: Any) -> Any:
        return measurement_at(texts, next(rates), prior)

    monkeypatch.setattr(app_review_script, "measure_script_lines", _measure)
    fake = _wm(slow, fast)

    out = await AppReviewScriptStage(client=fake).run({}, ctx=_ctx())

    assert all(line.text.startswith("b") for line in out.lines), (
        "at the first draft's rate this draw estimates 240.0s and is discarded"
    )
    estimate = out.narration_estimate
    assert estimate is not None
    assert estimate.estimate_seconds == pytest.approx(60.0), "its OWN measured seconds"
    assert estimate.wpm == pytest.approx(240.0)


@pytest.mark.asyncio
async def test_the_trim_cuts_by_measured_seconds_not_by_word_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Words are a PROXY for time, and the proxy drops the wrong lines.

    Four lines, 220 words, 200.0s measured — but the time is not where the words
    are: line 0 is 100 words in 60.0s, line 3 is 40 words in 100.0s (numerals, an
    acronym, a clause the voice pauses after). The pooled rate is 66 wpm, so the
    WORD proxy believes line 0 is the 90.9s monster and line 3 a 36.4s aside.

    Against a 120s order the two disagree completely:

    * by words, dropping line 0 alone "lands" at 109.1s — inside the band, one
      cut, done. It really delivers 140.0s (+16.7%), outside it.
    * by seconds, no single drop reaches the band (140.0 / 180.0 / 180.0 / 100.0),
      and the answer is TWO lines: 120.0s exactly.

    So this is the test that fails when ``line_seconds`` stops being passed to
    ``plan_trim`` — and nothing else does, because every other fixture here makes
    seconds exactly proportional to words.
    """
    seconds_by_index = (60.0, 20.0, 20.0, 100.0)
    prior = SpeechRate(wpm=66.0, measured=True)
    monkeypatch.setattr(
        speech_rate_module, "resolve_natural_wpm_detail", AsyncMock(return_value=prior)
    )

    async def _measure(_ctx: Any, *, texts: Any, **_kwargs: Any) -> Any:
        assert len(texts) == len(seconds_by_index), "the fixture pins one seconds per line"
        return ScriptMeasurement(
            per_line_seconds=seconds_by_index,
            measured=(True,) * 4,
            failed_indices=(),
            extrapolated_indices=(),
            total_seconds=200.0,
            measured_count=4,
            wpm=66.0,
            basis=BASIS_MEASURED_ALL,
            prior=prior,
        )

    monkeypatch.setattr(app_review_script, "measure_script_lines", _measure)
    draw = _draw(100, 40, 40, 40, tag="a")  # 220 words; 200.0s measured
    fake = _wm(draw, draw)  # the re-ask returns the same text ⇒ no closer, keep it

    out = await AppReviewScriptStage(client=fake).run({}, ctx=_ctx())

    kept = [line.text for line in out.lines]
    assert len(kept) == 2, f"the word proxy keeps THREE lines and delivers 140.0s; got {kept}"
    assert not any(text.startswith("a0_") for text in kept), "the 100-word line went"
    assert any(text.startswith("a3_") for text in kept), (
        "the 40-word line the proxy thinks is cheap is the 100.0s one — it stays"
    )
    assert out.narration_estimate is not None
    assert out.narration_estimate.estimate_seconds == pytest.approx(120.0)
    assert out.advisories == [], "120.0s against a 120s order is not a miss"


@pytest.mark.asyncio
async def test_a_measurement_that_never_landed_is_a_degradation_not_an_advisory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one duration fact that IS evidence of a bad delivery.

    An estimate that misses is advice. A measurement that never happened means
    every duration number on the run describes audio nobody measured — the run is
    budgeted blind, and that belongs on the tier that flips ``degraded``.
    """
    stub_failed_measurement(monkeypatch, app_review_script, WPM, RESOLUTION_ERROR)
    fake = _wm(_draw(60, 60, 60, 60, tag="a"))  # 240 words: on target at the prior

    out = await AppReviewScriptStage(client=fake).run({}, ctx=_ctx())

    assert [d["code"] for d in out.degradations] == [SPEECH_RATE_UNUSABLE]
    assert "could not be measured off this script's own lines" in out.degradations[0]["detail"]
    assert is_degraded(out.model_dump()), "an unmeasured run IS degraded"
    assert out.advisories == [], "the estimate itself is on target — only the basis is bad"
    assert out.narration_estimate is not None
    assert out.narration_estimate.measured_lines == 0, "and the record says so"


@pytest.mark.asyncio
async def test_an_empty_re_ask_is_recorded_as_a_discarded_re_ask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``reasked=True, reask_kept=None`` is a state the field's contract forbids.

    ``NarrationEstimate.reask_kept`` documents ``None`` as "there was no re-ask".
    An empty reply IS a re-ask — it cost the LLM call — and it WAS discarded, so
    the honest record is ``False``. The ambiguous pair made this run
    indistinguishable from one that never re-asked at all.
    """
    fake = _wm(_draw(64, 64, 64, 63, tag="a"), _resp({"lines": []}))

    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 2, "the re-ask was spent"
    assert all(line.text.startswith("a") for line in out.lines)
    assert out.actual_words == 255
    estimate = out.narration_estimate
    assert estimate is not None
    assert estimate.reasked is True
    assert estimate.reask_kept is False, "an empty re-ask was DISCARDED, not never-asked"
    assert estimate.trimmed_lines == 0


@pytest.mark.asyncio
async def test_a_trim_rejection_names_the_draw_it_is_about(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The incumbent is SHORT, so the trim's only source is the OTHER draft.

    ``_trim_step`` reports ``results[0]``, and ``results`` holds only the LONG
    candidates — so when a short re-asked draw is winning, ``results[0]`` is the
    first draft, not the script that ships. An unqualified "trim REJECTED" then
    described a draw the reader has no way to identify, on a run whose shipped
    draft was never trimmable in the first place.
    """
    phrase = "ships on every platform"
    long_first = _resp(
        {
            "lines": [
                {"text": _words(300, tag="x"), "seconds": 150.0},
                {"text": "ships on", "seconds": 1.0},
                {"text": "every platform", "seconds": 1.0},
                {"text": _words(4, tag="z"), "seconds": 2.0},  # 308 words = 154.0s
            ]
        }
    )
    short_second = _resp(
        {
            "lines": [
                {"text": _words(176, tag="y"), "seconds": 88.0},
                {"text": "ships on", "seconds": 1.0},
                {"text": "every platform", "seconds": 1.0},  # 180 words = 90.0s
            ]
        }
    )
    fake = _wm(long_first, short_second)
    ctx = _ctx(verbatim=[{"text": phrase, "stages": ["app_review_script"]}])

    out = await _run(monkeypatch, fake, ctx)

    assert len(out.lines) == 3, "the short re-ask is closer (30s vs 34s) and ships"
    assert phrase in " ".join(line.text for line in out.lines)
    detail = out.advisories[0]["detail"]
    assert (
        "trim of the first draw REJECTED — it would have dropped a required verbatim phrase"
        in detail
    ), detail
    assert "; trim REJECTED" not in detail, "the unqualified note named the wrong draft"


@pytest.mark.asyncio
async def test_an_in_band_incumbent_is_not_displaced_by_a_trimmed_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PINS A DECISION: the loop STOPS once the steer calls the incumbent ``ok``.

    Removing that guard is not a no-op, because the trim scans the whole POOL and
    not just the incumbent. Here the re-asked draw lands at 110.0s — in band, whole,
    nothing cut out of it — while the discarded 162.5s first draft can be trimmed
    to 112.5s, a hair nearer the 120s order. Without the stop-early check that
    amputated draft WINS on distance and ships instead.

    Distance is not the only thing that matters once a draw is conformant: a
    complete script beats a marginally-closer one with a line torn out.
    """
    fake = _wm(_draw(100, 100, 100, 25, tag="a"), _draw(55, 55, 55, 55, tag="b"))

    out = await _run(monkeypatch, fake)

    assert all(line.text.startswith("b") for line in out.lines), (
        "the in-band re-ask ships whole; the trimmed first draft is 112.5s and must not win"
    )
    assert len(out.lines) == 4
    assert out.actual_words == 220
    assert out.narration_estimate is not None
    assert out.narration_estimate.estimate_seconds == pytest.approx(110.0)
    assert out.narration_estimate.trimmed_lines == 0, "nothing was cut once the steer said ok"
    assert out.advisories == []


@pytest.mark.asyncio
async def test_an_unusable_speech_rate_is_recorded_on_an_in_band_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-band draw returns UNTOUCHED — but not silently mis-budgeted.

    At 0 wpm every duration below is computed at the substituted default. The
    substitution used to be invisible, so a run budgeted at a rate nobody
    measured looked identical to one budgeted at the measured rate.
    """
    stub_speech_rate(monkeypatch, app_review_script, 0.0)
    fake = _wm(_draw(150, 150, tag="a"))  # 300 words = 120s at the 150 default
    out = await AppReviewScriptStage(client=fake).run({}, ctx=_ctx())

    assert fake.message.await_count == 1, "in band ⇒ no re-ask"
    assert len(out.lines) == 2, "and no trim"
    codes = [d["code"] for d in out.degradations]
    assert codes == [SPEECH_RATE_UNUSABLE], "the substitution alone is on the record"
    assert "0.0" in out.degradations[0]["detail"]


@pytest.mark.asyncio
async def test_a_floored_re_ask_that_lands_further_away_is_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The (2b) reply is judged by ``closer_to_order`` like every other draw."""
    fake = _wm(_draw(140, 140, tag="a"), _draw(140, 140, tag="b"), _draw(100, 100, 100, tag="c"))

    out = await _run(monkeypatch, fake)

    assert fake.message.await_count == 3
    assert all(line.text.startswith("a") for line in out.lines), "the closer draw ships"
    codes = [d["code"] for d in out.degradations]
    assert ORDERED_DURATION_INFEASIBLE in codes, "the floor still describes what shipped"
