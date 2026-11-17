"""promo_script corrects the drawn narration toward the ORDERED duration.

Measured live over five 45s promos, promo delivered +4.50%, +8.30%, +14.52%,
+19.00% and +32.36%; one run drew 152 words against a 131-word budget. The stage
had no re-ask and no trim — it only compared WORDS to a word budget and recorded
the miss. It refused to act on that on purpose, because the two halves disagreed
on DIRECTION: the run that drew 115 words against a 131-word budget was 12% SHORT
in words and delivered 51.240s against 45.0s, i.e. +13.87% LONG in seconds. A
word-driven re-ask would have asked that run for MORE words.

Measuring SECONDS is what resolves the ambiguity and makes a lever safe, so every
test here is written in seconds. The stub rate is 60 wpm throughout unless stated,
which makes ONE WORD EXACTLY ONE SECOND — so a script's ordered/delivered
arithmetic is readable straight off the word counts rather than re-derived.

Two guards here are promo-specific and are the reason this is not a copy of the
app_review/website_review suites:

* the CALL-TO-ACTION is requested in the prompt only, gated by nothing on the LLM
  path, and ``plan_trim`` prefers dropping the LATEST lines — where the prompt
  tells speaker "b" to put it;
* the lines are DIALOGUE TURNS cast to two kokoro voices, so a line-wise drop
  would leave the same voice speaking twice in a row.
"""

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

import src.services.stages.promo.script as promo_script
from src.services.runs.run_context import RunContext
from src.services.stages.promo.narrate import _DEFAULT_CAST, PromoNarrateStage
from src.services.stages.promo.script import PromoScriptStage
from src.services.stages.shared.degradation import (
    NARRATION_ESTIMATE_OFF_TARGET,
    SPEECH_RATE_UNUSABLE,
)
from src.services.stages.shared.speech_rate import PROBE_TIMEOUT
from tests.services.stages.promo.rate_stubs import (
    MeasurementRecorder,
    stub_failed_measurement,
    stub_speech_rate,
)

#: One word per second. Every duration in this module is a word count.
WPM = 60.0
CTA = "Get the playbook."
#: Speaker "a"/"b" voices under promo_narrate's own ``_DEFAULT_CAST``.
VOICE_A = _DEFAULT_CAST["a"]
VOICE_B = _DEFAULT_CAST["b"]


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _ctx(ordered: float, *, brief: dict | None = None, voice: dict | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "title": "Anti-churn",
            "brief": {
                "call_to_action": CTA,
                "target_duration_seconds": ordered,
                **(brief or {}),
            },
            "voice": {"descriptor": "crisp", **(voice or {})},
        },
        prior_run_outputs={
            "promo_analyze": {
                "pain_points": [{"id": "p1", "text": "high churn"}],
                "hook_candidates": [{"id": "h1", "text": "Stop the bleeding."}],
                "tone_hints": ["urgent"],
            },
        },
    )


def _words(count: int, *, tag: str = "w") -> str:
    """A line of exactly ``count`` whitespace-separated words — i.e. ``count`` seconds."""
    return " ".join(f"{tag}{i}" for i in range(count))


def _draft(*line_words: int, cta_last: bool = True) -> dict:
    """A drawn script: one line per entry, the last one carrying the CTA verbatim.

    The CTA is three words, so a ``cta_last`` line of ``n`` words is the CTA plus
    ``n - 3`` filler — its measured seconds stay exactly ``n``.
    """
    lines = []
    for index, count in enumerate(line_words):
        last = index == len(line_words) - 1
        text = (
            f"{CTA} {_words(count - 3, tag=f'c{index}_')}"
            if last and cta_last
            else _words(count, tag=f"l{index}_")
        )
        lines.append({"text": text, "seconds": 1.0})
    return {"lines": lines}


async def _run(
    monkeypatch: pytest.MonkeyPatch,
    ctx: RunContext,
    *drafts: dict,
    rates: dict[str | None, float] | None = None,
) -> tuple[Any, AsyncMock, MeasurementRecorder]:
    """Run the stage over ``drafts`` (the first draw, then any re-ask)."""
    recorder = stub_speech_rate(monkeypatch, promo_script, WPM, rates)
    fake = AsyncMock()
    if len(drafts) == 1:
        fake.message.return_value = _resp(drafts[0])
    else:
        fake.message = AsyncMock(side_effect=[_resp(d) for d in drafts])
    out = await PromoScriptStage(client=fake).run({}, ctx=ctx)
    return out, fake, recorder


def _advisories(out: Any) -> list[dict]:
    return [a for a in out.advisories if a["code"] == NARRATION_ESTIMATE_OFF_TARGET]


# --------------------------------------------------------------------------- #
# the correction itself
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_live_plus_32_percent_shape_is_corrected_and_ships_in_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE measured defect: 66s against a 50s order (+32.0%) must not ship as drawn.

    The steer fires (anything past +4% does), the re-ask is spent, the re-asked
    draft measures 52s (+4.0%) and is kept because it is closer. What SHIPS is
    inside the band, and the run says so rather than carrying an advisory.
    """
    out, fake, _ = await _run(
        monkeypatch,
        _ctx(50),
        _draft(20, 20, 26),  # 66 words = 66.0s = +32.0%
        _draft(20, 20, 12),  # 52 words = 52.0s = +4.0%
    )

    assert fake.message.await_count == 2, "exactly one re-ask, never a third draw"
    assert out.narration_estimate is not None
    assert out.narration_estimate.first_draw_estimate_seconds == pytest.approx(66.0)
    assert out.narration_estimate.estimate_seconds == pytest.approx(52.0)
    assert out.narration_estimate.reasked is True
    assert out.narration_estimate.reask_kept is True
    assert out.narration_estimate.verdict == "ok"
    assert _advisories(out) == [], f"an in-band ship states nothing, got {out.advisories}"
    assert out.degradations == [], f"and degrades nothing, got {out.degradations}"


@pytest.mark.asyncio
async def test_a_re_ask_that_does_not_land_is_rescued_by_the_trim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The re-ask comes back no closer, so attempt two — the free trim — runs.

    Both drafts are 40s against a 30s order (+33.3%); the second is not closer, so
    the first draw stays the incumbent and is trimmed. Band is [27.0, 33.0], and
    dropping one 10-word line lands on 30.0s exactly.
    """
    out, fake, _ = await _run(
        monkeypatch,
        _ctx(30),
        _draft(10, 10, 10, 10),
        _draft(10, 10, 10, 10),
    )

    assert fake.message.await_count == 2
    assert out.narration_estimate is not None
    assert out.narration_estimate.reask_kept is False, "the re-ask bought nothing"
    assert out.narration_estimate.trimmed_lines == 1
    assert out.narration_estimate.estimate_seconds == pytest.approx(30.0)
    assert len(out.lines) == 3
    assert out.actual_words == 30, "the word total was re-derived over the kept lines"
    # The persisted length is the MEASUREMENT, not the LLM's own arithmetic. The
    # model claimed 1.0s for each of these ten-word lines; they measure 10.0s each
    # at 60 wpm, so a total re-summed from `lines[*].seconds` would persist 3.0 —
    # an order of magnitude under a script this stage just measured end to end.
    assert out.total_seconds == pytest.approx(30.0)
    assert out.total_seconds == pytest.approx(out.narration_estimate.estimate_seconds)
    assert sum(line.seconds for line in out.lines) == pytest.approx(3.0), (
        "per-line seconds stay the model's — promo_storyboard/render time beats off them"
    )


@pytest.mark.asyncio
async def test_an_in_band_draft_is_never_touched(monkeypatch: pytest.MonkeyPatch) -> None:
    """A draw the steer calls ``ok`` costs no second LLM call and loses no line.

    A lever that fires on a good draw doubles the stage's spend on the type that
    completes most often, which is exactly why promo refused to grow one until the
    verdict could be taken on measured seconds.
    """
    out, fake, _ = await _run(monkeypatch, _ctx(45), _draft(15, 15, 15))  # 45 words = 45.0s

    fake.message.assert_called_once()
    assert len(out.lines) == 3
    assert out.narration_estimate is not None
    assert out.narration_estimate.reasked is False
    assert out.narration_estimate.reask_kept is None
    assert out.narration_estimate.trimmed_lines == 0
    assert out.narration_estimate.estimate_seconds == pytest.approx(45.0)


@pytest.mark.asyncio
async def test_a_six_percent_overrun_is_steered_even_though_it_is_in_the_outer_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """+6% is CONFORMANT and still gets corrected — the steer is not the band.

    ``narration_verdict`` (+/-10%) grades the delivered master; a draw accepted at
    +9% has nothing left for the estimate's own error. ``steer_verdict`` aims at
    +4% so there is headroom. Mutate ``steer_verdict`` to ``narration_verdict``
    here and the stage spends no correction at all.
    """
    out, fake, _ = await _run(
        monkeypatch,
        _ctx(50),
        _draft(18, 18, 17),  # 53 words = 53.0s = +6.0%, inside +/-10%
        _draft(17, 17, 16),  # 50 words = 50.0s = dead on
    )

    assert fake.message.await_count == 2, "+6% was steered, not accepted"
    assert out.narration_estimate is not None
    assert out.narration_estimate.estimate_seconds == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_a_short_draw_is_accepted_and_never_padded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing lengthens a short script mechanically — repeated audio is worse.

    The re-ask may ask for more material (and does, in the "TOO SHORT" direction),
    but when it comes back no better the short draft SHIPS with every line intact
    and the miss is reported, not fixed.
    """
    out, fake, _ = await _run(
        monkeypatch,
        _ctx(50),
        _draft(10, 10, 10),  # 30 words = 30.0s = -40%
        _draft(10, 10, 10),
    )

    assert "TOO SHORT" in fake.message.call_args_list[1].kwargs["message"]
    assert len(out.lines) == 3, "no line was added, duplicated or dropped"
    assert out.narration_estimate is not None
    assert out.narration_estimate.estimate_seconds == pytest.approx(30.0)
    assert out.narration_estimate.trimmed_lines == 0
    assert _advisories(out), "a -40% ship is REPORTED"


# --------------------------------------------------------------------------- #
# the two promo-specific trim guards
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_trim_never_drops_the_call_to_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``plan_trim`` drops the LATEST lines, and the CTA is the last line.

    30s drawn against a 20s order, band [18.0, 22.0]: two 5-word lines must go. The
    unprotected tail is lines 3 and 4, so an unguarded trim takes line 5 — the CTA
    — with them. ``protected_items`` pins it, so lines 2 and 3 go instead.
    """
    out, _, _ = await _run(monkeypatch, _ctx(20), _draft(5, 5, 5, 5, 5, 5))

    assert out.narration_estimate is not None
    assert out.narration_estimate.trimmed_lines == 2
    assert out.narration_estimate.estimate_seconds == pytest.approx(20.0)
    assert CTA.lower() in " ".join(line.text for line in out.lines).lower(), (
        "the promo still advertises something"
    )
    assert CTA.lower() in out.lines[-1].text.lower(), "and it is still the closing line"


@pytest.mark.asyncio
async def test_a_trim_that_would_split_the_call_to_action_is_refused_outright(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The protection is per-line; the RE-CHECK is what catches a CTA across two.

    Neither of the last two lines carries the whole phrase, so neither is pinned —
    but the joined script does, and dropping either destroys it. The post-trim
    comparison is against what was missing BEFORE the trim, so this refuses the
    trim rather than the draft.
    """
    ctx = _ctx(20, brief={"call_to_action": "Get the playbook"})
    draft = {
        "lines": [
            {"text": _words(5, tag="a"), "seconds": 1.0},
            {"text": _words(5, tag="b"), "seconds": 1.0},
            {"text": _words(5, tag="c"), "seconds": 1.0},
            {"text": _words(5, tag="d"), "seconds": 1.0},
            {"text": f"{_words(3, tag='e')} Get the", "seconds": 1.0},
            {"text": "playbook now, seriously today", "seconds": 1.0},
        ]
    }
    out, _, _ = await _run(monkeypatch, ctx, draft, draft)

    assert len(out.lines) == 6, "nothing was dropped"
    assert out.narration_estimate is not None
    assert out.narration_estimate.trimmed_lines == 0
    detail = _advisories(out)[0]["detail"]
    assert "REJECTED" in detail and "call-to-action" in detail, (
        f"the refusal is stated, not silent: {detail!r}"
    )


@pytest.mark.asyncio
async def test_the_trim_cannot_drop_a_paraphrased_call_to_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PARAPHRASED CTA has no verbatim match, and used to be trimmed away.

    The brief orders ``"Get the playbook."``; the model closes with
    ``"Grab the playbook today"``. Nothing objects — the CTA is prompt-only and
    ``assert_verbatim_present`` never runs on the LLM path — so this is an ordinary
    draw, not a malformed one.

    60 words against a 20s order at 60 wpm is 60.0s (+200%), band [18.0, 22.0], and
    two of the three lines have to go. Every 2-of-3 drop lands on 20.0s exactly, so
    ``plan_trim`` breaks the tie its documented way — cutting the LATEST lines —
    and takes the closer with it. The post-trim re-check cannot save it either: it
    subtracts what was missing BEFORE the trim, and ``"Get the playbook."`` was
    already missing, so the difference is empty and the trim is approved. Result
    before the fix: one line, no call to action, shipped green.

    The closing block is now protected unconditionally whenever a CTA was ordered,
    so the two droppable lines are 0 and 1 — the trim still reaches the band, it
    just cannot reach the closer.
    """
    paraphrase = f"Grab the playbook today {_words(16, tag='cta_')}"
    draft = {
        "lines": [
            {"text": _words(20, tag="a"), "seconds": 1.0},
            {"text": _words(20, tag="b"), "seconds": 1.0},
            {"text": paraphrase, "seconds": 1.0},
        ]
    }
    out, _, _ = await _run(monkeypatch, _ctx(20), draft, draft)

    joined = " ".join(line.text for line in out.lines).lower()
    assert CTA.lower() not in joined, (
        "the draft never contained the CTA verbatim — so the verbatim protection "
        "demonstrably could not be what kept the closer"
    )
    assert out.narration_estimate is not None
    assert out.narration_estimate.trimmed_lines == 2, "the lever still fired"
    assert out.narration_estimate.estimate_seconds == pytest.approx(20.0)
    assert out.narration_estimate.verdict == "ok"
    assert [line.text for line in out.lines] == [paraphrase], (
        "the promo still asks for something; the two body lines went instead"
    )


@pytest.mark.asyncio
async def test_the_re_ask_target_is_computed_at_the_measured_rate_not_the_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#998: the re-ask asks for the MEASURED rate's word count, never the prior's.

    Every other rate test in this suite stubs the prior EQUAL to the measured rate,
    which makes the invariant arithmetically unfalsifiable — a regression to
    ``self._rate.wpm`` would ship green through all of them. So this one splits the
    two: the Phase-A prior lands at 60 wpm and the full-script measurement pools at
    120 wpm.

    The numbers, each derived here rather than quoted: the first draw's word TARGET
    is written at the prior, ``budget_words(20, 60) = round(20 / 60 x 60) = 20``
    words. The draft then measures 60 words / 120 wpm x 60 = 30.0s against a 20s
    order, which steers long. The re-ask target is
    ``budget_words(20, 120) = round(20 / 60 x 120) = 40`` words — DOUBLE what the
    prior affords, and asking for 20 would tell a narrator who speaks twice as fast
    as assumed to write half the script the order can pay for.
    """
    draft = _draft(20, 20, 20)
    out, fake, _ = await _run(monkeypatch, _ctx(20), draft, draft, rates={None: 120.0})

    assert fake.message.await_count == 2, "the steer fired and a re-ask was spent"
    first, second = (call[1]["message"] for call in fake.message.call_args_list)
    assert "pace of 60 words per minute, 20 seconds is 20 words" in first, (
        "the FIRST draw is budgeted at the prior — that half of #998 is unchanged"
    )
    assert "REWRITE" in second
    assert "i.e. 40 words for the WHOLE script" in second, (
        "the re-ask target is the MEASURED rate's count; the prior's is 20"
    )
    assert "approximately 40 words" in second
    assert "20 words for the WHOLE script" not in second
    assert out.narration_estimate is not None
    assert out.narration_estimate.wpm == pytest.approx(120.0), "and the record says so"


@pytest.mark.asyncio
async def test_a_trim_in_dialogue_mode_keeps_the_a_b_alternation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lines are dialogue TURNS: the trim drops whole a/b pairs, never one line.

    Eight turns of 2/8/5/5/7/3/5/5 words = 40.0s against a 30s order, band
    [27.0, 33.0]. The UNEVEN lengths are the point — they are what separates the
    two plans, and an all-equal script does not:

    * over TURN PAIRS every block is 10.0s, so dropping one lands on 30.0s exactly
      and the survivors still read a, b, a, b, a, b;
    * line by line the cheapest landing is ONE drop — the 7-word turn 4, at 33.0s
      — which leaves turns 3 and 5 adjacent, both speaker "b". The same voice
      answers itself, and ``promo_narrate`` casts it to one kokoro voice twice in
      a row.

    So mutating ``_turn_blocks`` to one line per block makes the SPEAKER assertion
    below fail, which is the only thing that makes that function's docstring
    claim falsifiable. An earlier version of this test used eight equal turns and
    survived that mutation intact: the line-wise plan happened to drop turns 5 and
    6 — an even, adjacent run, i.e. a pair by accident.
    """
    counts = [2, 8, 5, 5, 7, 3, 5, 5]
    lines = []
    for index, count in enumerate(counts):
        ask = index % 2 == 0
        text = (
            f"does {_words(count - 1, tag=f'q{index}_')}?"
            if ask
            else _words(count, tag=f"s{index}_")
        )
        lines.append({"text": text, "seconds": 1.0, "speaker": "a"})
    lines[-1]["text"] = f"{CTA} yes really"  # 5 words, a statement ⇒ speaker "b"

    ctx = _ctx(30, brief={"dialogue": {"enabled": True}}, voice={"voice_id": "af_bella"})
    out, _, _ = await _run(monkeypatch, ctx, {"lines": lines}, {"lines": lines})

    speakers = [line.speaker for line in out.lines]
    assert speakers == ["a", "b", "a", "b", "a", "b"], f"still a clean two-hander: {speakers}"
    assert out.narration_estimate is not None
    assert out.narration_estimate.trimmed_lines == 2, "a whole turn PAIR, not a line"
    assert out.narration_estimate.estimate_seconds == pytest.approx(30.0)
    assert CTA.lower() in out.lines[-1].text.lower()


@pytest.mark.asyncio
async def test_the_trim_plans_on_measured_seconds_not_on_the_word_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal-word turns take UNEQUAL time when two voices speak them.

    Three turn pairs of 24, 24 and 16 words. At the two cast rates (a=120 wpm,
    b=60 wpm) they run 14.0s, 22.0s and 12.0s — 48.0s against a 36s order, band
    [32.4, 39.6]. Dropping pair 0 lands at 34.0s, inside it.

    On the WORD proxy every pair looks like 24/24/16 words at the pooled 80 wpm:
    dropping either 24-word pair lands at 30.0s, UNDER the band and declined, and
    the only pair that would land is the protected CTA pair — so the word plan
    floors and ships the full 48.0s overrun. Mutate ``line_seconds`` to ``None``
    and that is exactly what happens.
    """
    ctx = _ctx(36, brief={"dialogue": {"enabled": True}}, voice={"voice_id": "af_bella"})
    lines = [
        {"text": f"does {_words(19, tag='q0_')}?", "seconds": 1.0},  # a, 20w → 10.0s
        {"text": _words(4, tag="s0_"), "seconds": 1.0},  # b,  4w →  4.0s
        {"text": f"does {_words(3, tag='q1_')}?", "seconds": 1.0},  # a,  4w →  2.0s
        {"text": _words(20, tag="s1_"), "seconds": 1.0},  # b, 20w → 20.0s
        {"text": f"does {_words(7, tag='q2_')}?", "seconds": 1.0},  # a,  8w →  4.0s
        {"text": f"{CTA} {_words(5, tag='c2_')}", "seconds": 1.0},  # b,  8w →  8.0s
    ]
    out, _, _ = await _run(
        monkeypatch,
        ctx,
        {"lines": lines},
        {"lines": lines},
        rates={VOICE_A: 120.0, VOICE_B: 60.0},
    )

    assert out.narration_estimate is not None
    assert out.narration_estimate.first_draw_estimate_seconds == pytest.approx(48.0)
    assert out.narration_estimate.estimate_seconds == pytest.approx(34.0), (
        "the seconds plan dropped the CHEAP pair; the word plan cannot reach the band"
    )
    assert out.narration_estimate.trimmed_lines == 2
    assert [line.text for line in out.lines] == [line["text"] for line in lines[2:]]


# --------------------------------------------------------------------------- #
# what the run RECORDS
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_measurement_requests_exactly_what_promo_narrate_will_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-line VOICE, speed and stage name must match narrate's, or every line
    is synthesized twice.

    ``deterministic_clip_id`` hashes ``(text, voice, speed)`` and the lookup is
    scoped by stage name, so a measurement taken at any other tuple measures audio
    nobody will hear AND stores clips under ids ``promo_narrate`` will not find.
    """
    ctx = _ctx(45, brief={"dialogue": {"enabled": True}}, voice={"voice_id": "af_bella"})
    lines = [
        {"text": f"does {_words(21, tag='q_')}?", "seconds": 1.0},  # a, 22w
        {"text": f"{CTA} {_words(20, tag='s_')}", "seconds": 1.0},  # b, 23w
    ]
    _, _, recorder = await _run(monkeypatch, ctx, {"lines": lines})

    assert len(recorder.calls) == 1, "45 words against a 45s order needs no correction"
    assert recorder.voices == (VOICE_A, VOICE_B), (
        f"each turn measured at the voice promo_narrate casts it to: {recorder.voices}"
    )
    # ``None``, not ``1.0``: promo_narrate inherits ``NarrationStageBase.speed_for``
    # and omits the argument, so ``deterministic_clip_id`` hashes None. A 1.0 here
    # is a different clip id AND a different duration (speed scales it linearly).
    assert promo_script.NARRATE_SPEED is None
    assert recorder.speeds == [None]
    assert recorder.stage_names == [PromoNarrateStage.name]


@pytest.mark.asyncio
async def test_a_single_narrator_promo_measures_every_line_at_the_configured_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dialogue OFF ⇒ narrate speaks ONE voice ⇒ the measurement must too.

    ``config.voice.cast`` is populated on purpose: the cast applies only behind
    ``brief.dialogue.enabled``, so a measurement reading the cast map directly
    would synthesize a pair of voices that never speak.
    """
    ctx = _ctx(
        45,
        brief={"dialogue": {"enabled": False}},
        voice={"voice_id": "af_bella", "cast": {"a": VOICE_A, "b": VOICE_B}},
    )
    _, _, recorder = await _run(monkeypatch, ctx, _draft(15, 15, 15))

    assert recorder.voices == ("af_bella",) * 3


@pytest.mark.asyncio
async def test_a_clean_run_still_records_the_narration_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean run records its numbers too — an absence cannot be read.

    The numbers are persisted on EVERY reconciled run, including the ones that had
    nothing to correct, or the record cannot answer "why did this come in at +26%?"
    after the fact.
    """
    out, _, _ = await _run(monkeypatch, _ctx(45), _draft(15, 15, 15))

    estimate = out.narration_estimate
    assert estimate is not None
    assert estimate.ordered_seconds == pytest.approx(45.0)
    assert estimate.wpm == pytest.approx(WPM)
    assert estimate.wpm_basis == "measured_all"
    assert estimate.measured_lines == 3, "every line's seconds is a real synthesis"
    assert estimate.verdict == "ok"


@pytest.mark.asyncio
async def test_an_estimate_that_misses_the_band_is_an_advisory_not_a_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-audio estimate must not flip a run ``degraded``.

    ``duration_conformance`` grades the delivered master from the real artifact;
    this is a note about how the SCRIPT was aimed. A lesson measured live delivered
    +0.22% of its order while its script stage had already stamped
    ``duration_out_of_tolerance`` from the estimate. Mutate
    ``narration_estimate_advisory`` onto the ``degradations`` channel and this goes
    red on both halves.
    """
    out, _, _ = await _run(
        monkeypatch,
        _ctx(50),
        _draft(10, 10, 10),  # 30.0s = -40%, and nothing pads a short draw
        _draft(10, 10, 10),
    )

    entries = _advisories(out)
    assert len(entries) == 1, f"stated exactly once, got {out.advisories}"
    detail = entries[0]["detail"]
    assert "30.0s" in detail and "50.0s" in detail and "-40.0%" in detail, detail
    assert detail.startswith("promo_script:"), detail
    assert out.degradations == [], (
        f"a conformance opinion formed before any audio degrades nothing: {out.degradations}"
    )


@pytest.mark.asyncio
async def test_an_unusable_measurement_degrades_the_run_and_names_the_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A measurement that did NOT land is a fault, not advice.

    Every duration number on such a run describes audio nobody measured — the
    estimate is the prior rate times a word count again — so it stays on the
    degrading tier while the estimate MISS does not.
    """
    stub_failed_measurement(monkeypatch, promo_script, WPM, PROBE_TIMEOUT)
    fake = AsyncMock()
    fake.message.return_value = _resp(_draft(15, 15, 15))
    out = await PromoScriptStage(client=fake).run({}, ctx=_ctx(45))

    entries = [d for d in out.degradations if d["code"] == SPEECH_RATE_UNUSABLE]
    assert len(entries) == 1, f"recorded exactly once, got {out.degradations}"
    assert PROBE_TIMEOUT in entries[0]["detail"], "the operational fact is NAMED"
    assert out.narration_estimate is not None
    assert out.narration_estimate.measured_lines == 0, "and the record says nothing was measured"


@pytest.mark.asyncio
async def test_an_authored_shotlist_is_not_measured_reconciled_or_probed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bypass IS the picture length, and it is corrected against nothing.

    ``short_circuit`` returns the operator's own narration and seconds before any
    prompt, budget or probe exists. Steering those toward a target they did not ask
    for would manufacture a defect out of their own decision — and reaching the
    reconciliation at all would mean the bypass had started paying for the
    synthesis it deliberately skips.
    """
    recorder = stub_speech_rate(monkeypatch, promo_script, WPM)
    narrations = ["First screen, here it is.", "Second screen, and what does it cost?"]
    fake = AsyncMock()
    out = await PromoScriptStage(client=fake).run(
        {},
        ctx=_ctx(
            45,
            brief={
                "shotlist": [
                    {"url": f"https://acme.test/{i}", "narration": text}
                    for i, text in enumerate(narrations)
                ]
            },
        ),
    )

    assert fake.message.await_count == 0, "the bypass never asks the model"
    assert recorder.calls == [], "and never pays for a synthesis"
    assert [line.text for line in out.lines] == narrations, "the operator's words, verbatim"
    assert out.narration_estimate is None, "nothing was measured, so nothing is claimed"
    assert out.degradations == [] and out.advisories == []


# --------------------------------------------------------------------------- #
# step (2b): a trim that could not act earns one re-ask at a word target
# --------------------------------------------------------------------------- #


def _prompts(fake: AsyncMock) -> list[str]:
    return [str(call.kwargs.get("message", "")) for call in fake.message.call_args_list]


@pytest.mark.asyncio
async def test_a_long_draw_whose_trim_quantum_overshoots_the_band_is_re_asked_at_a_word_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BACKLOG #7: one line's duration is wider than the band, so the trim declines.

    20s + a protected 20s CTA line = 40s against a 30s order (+33.3%), band
    [27.0, 33.0]. The only droppable line lands the script at 20s, UNDER the band,
    so ``plan_trim`` declines and drops NOTHING — and before step (2b) the stage
    shipped +33.3%. Now the floored long draw is re-asked once at
    ``ordered * draw.wpm / 60`` = 30 words, and the in-band reply ships.
    """
    out, fake, _ = await _run(
        monkeypatch,
        _ctx(30),
        _draft(20, 20),  # 40s
        _draft(20, 20),  # steer re-ask, not closer
        _draft(15, 15),  # 30s
    )

    assert fake.message.await_count == 3, "draw + steer re-ask + floored-trim re-ask"
    final = _prompts(fake)[-1]
    assert "REWRITE" in final
    assert "It contained 40 words" in final, "what the floored draw actually contained"
    assert "i.e. 30 words" in final, "the explicit word target"
    assert out.narration_estimate is not None
    assert out.narration_estimate.estimate_seconds == pytest.approx(30.0)
    assert out.narration_estimate.reask_kept is True
    assert out.narration_estimate.trimmed_lines == 0
    assert out.degradations == [], "a floor that no longer describes the shipped script is cleared"
    assert _advisories(out) == []


@pytest.mark.asyncio
async def test_a_trim_that_acted_is_never_pre_empted_by_a_paid_re_ask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The (2b) ask fires only when the trim dropped nothing — the free lever first."""
    out, fake, _ = await _run(monkeypatch, _ctx(30), _draft(10, 10, 10, 10), _draft(10, 10, 10, 10))

    assert fake.message.await_count == 2
    assert out.narration_estimate is not None
    assert out.narration_estimate.trimmed_lines == 1
