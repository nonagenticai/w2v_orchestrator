"""4.2 — the WORD BUDGET reaches bdd_playwright alongside the ordered duration.

A duration alone is not actionable by a SCRIPT WRITER. A film is exactly its
narration, and there is no duration ceiling on the record path, so the only lever
that controls length at the point of authoring is WORD COUNT.

⛔ CONTRACT. ``narration_words_target`` and ``narration_words_per_scenario`` are the
key names a sibling service (bdd_playwright_service) is coded against. Renaming
either is a cross-service break, which is why they are asserted as literals here.

Same harness as ``test_gen_sends_ordered_duration``: mock client, real models.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.schemas.bdd_playwright import BddJobResult
from src.clients.tts_client import TTSClient
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.gen import DemoVideoGenStage
from src.services.stages.shared.speech_rate import (
    CALIBRATION_WORDS,
    clear_measured_wpm_cache,
)
from src.services.stages.shared.time_budget import (
    DEFAULT_WPM_NATURAL,
    NARRATION_BUDGET_RATE,
    VO_RATE_DRAG_FLOOR,
)

_URL = "https://app.example"
_ORDERED = 30.0
# The rate the live voice was MEASURED at (2026-09, kokoro) — deliberately NOT
# DEFAULT_WPM_NATURAL, so a budget taken at the constant and a budget taken at the
# measurement are different integers and the two cannot be confused.
_MEASURED_WPM = 151.6


@pytest.fixture(autouse=True)
def _passage_rate_is_the_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the passage->narration ratio at 1.0: these tests prove the measured-rate
    PLUMBING with arithmetic written at the raw passage rate. The ratio itself is
    pinned by ``tests/services/stages/shared/test_speech_rate_narration_prior.py``."""
    from src.services.stages.shared import speech_rate

    monkeypatch.setattr(speech_rate, "PASSAGE_TO_NARRATION_RATIO", 1.0)


@pytest.fixture(autouse=True)
def _clean_wpm_cache() -> Any:
    """`measure_natural_wpm` memoises per (provider, voice) for the process — without
    this, one test's fake rate leaks into the next one's budget."""
    clear_measured_wpm_cache()
    yield
    clear_measured_wpm_cache()


class _FakeTTS:
    """Answers the calibration probe at a fixed wpm (or refuses to answer)."""

    def __init__(self, wpm: float | None) -> None:
        self.wpm = wpm

    async def synthesize(self, **_: Any) -> Any:
        if self.wpm is None:
            raise TimeoutError("probe timed out")
        seconds = CALIBRATION_WORDS * 60.0 / self.wpm
        return type("_Synth", (), {"duration_seconds": seconds})()


def _tts(wpm: float | None) -> TTSClient:
    return cast("TTSClient", _FakeTTS(wpm))


def _ctx(**sections: Any) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {"target_url": _URL}, **sections},
        prior_run_outputs={},
    )


def _client() -> AsyncMock:
    client = AsyncMock()
    client.start_demo_video.return_value = "bdd-job-1"
    client.wait_for_job.return_value = {}
    client.get_job_result.return_value = BddJobResult.model_validate(
        {
            "job_id": "bdd-job-1",
            "status": "completed",
            "features_generated": 1,
            "narration_script": {
                "parts": [
                    {
                        "part_id": "clip-1",
                        "scenario_name": "Login",
                        "text": "First we log in.",
                        "original_gherkin": "Scenario: Login",
                        "estimated_duration_seconds": 3.2,
                    }
                ]
            },
        }
    )
    return client


async def _sent(ctx: RunContext, wpm: float | None = None) -> dict[str, Any]:
    """Drive the stage and hand back the payload it actually sent bdd.

    ``wpm=None`` means the calibration probe does not answer — the fallback path, where
    the budget is taken at ``DEFAULT_WPM_NATURAL`` exactly as it was before the
    measurement existed. Every test that is not ABOUT the rate uses it, so those
    assertions stay pinned to the constant.
    """
    client = _client()
    stage = DemoVideoGenStage(client=client, tts_client=_tts(wpm))
    await stage.handle(stage.input_model(), ctx)
    return dict(client.start_demo_video.call_args.args[0])


async def test_the_word_target_is_sent_under_the_contracted_key() -> None:
    """The key name, the type, and the ONE rate the budget is priced at.

    HISTORY OF THIS NUMBER, because it has moved twice: 60 words while the budget was
    taken at ``VO_RATE_NATURAL_FLOOR`` (0.80) — that was the wave-13 root cause, since
    budgeting for an already-dragged voice ORDERS the drag upstream; then 73 at
    ``VO_RATE_DRAG_FLOOR`` (0.97), which was the obvious repair and still wrong, because
    pricing the budget AT the detector's own threshold puts a perfect delivery on the
    detection boundary and lets integer rounding cast the deciding vote; now 75 at
    ``NARRATION_BUDGET_RATE`` (1.0), which leaves the 3% gap as the tolerance.

    Derived rather than restated: the hardcoded ``== 73`` is exactly what made this test
    red on a correct change.
    """
    sent = await _sent(_ctx(deliverable={"target_duration_seconds": _ORDERED}))

    # The literal key name IS the cross-service contract.
    assert "narration_words_target" in sent, sent
    # 30s x 150 wpm x 1.0 / 60 = 75 words.
    assert sent["narration_words_target"] == round(
        _ORDERED * DEFAULT_WPM_NATURAL * NARRATION_BUDGET_RATE / 60
    )
    assert isinstance(sent["narration_words_target"], int)


async def test_the_target_is_derived_from_the_repos_constants_not_restated() -> None:
    """Bind the wire value to the constants, so the two budgets cannot drift.

    ⭐ AND TO THE RIGHT ONE OF THEM. This asserted
    ``DEFAULT_WPM_NATURAL * VO_RATE_DRAG_FLOOR`` — i.e. it bound the wire value to the
    DETECTOR's threshold, which is a different decision from the budget and must never be
    the same number again. So the second assertion pins that the two constants DIFFER and
    that the budget is the larger of the two: a future session cannot silently re-merge
    them without reddening this test, which is the whole point of a no-magic-number test.
    """
    sent = await _sent(_ctx(deliverable={"target_duration_seconds": 90.0}))

    assert sent["narration_words_target"] == round(
        90.0 * DEFAULT_WPM_NATURAL * NARRATION_BUDGET_RATE / 60
    )
    # One constant, one job — the budget is priced strictly ABOVE the drag floor.
    assert VO_RATE_DRAG_FLOOR < NARRATION_BUDGET_RATE
    assert sent["narration_words_target"] > round(
        90.0 * DEFAULT_WPM_NATURAL * VO_RATE_DRAG_FLOOR / 60
    )


async def test_a_known_beat_count_gets_a_per_scenario_split_that_covers_the_total() -> None:
    """An AUTHORED scenario count is a known beat count, so the split rides.

    ⭐ WAVE 14, N3 — THE LIVE-PAYLOAD CALL-SITE PROOF. This is the owner's own worked
    example arriving at bdd: a 30.00s order over 4 authored scenarios. The split sent
    here used to be ``[19, 19, 19, 18]``, and its trailing beat, delivered to the word
    against an equal 7.50s allotment, is ``18 / 18.75 = 0.960`` — under the 0.97 drag
    floor, flagged ``underspoken``, one guaranteed paid LLM re-ask per run.

    It is now ``[19, 19, 19, 19]``: the split may OVERSHOOT the total (never undershoot
    it), because undershooting is what orders the drag.
    """
    sent = await _sent(
        _ctx(
            deliverable={"target_duration_seconds": _ORDERED},
            brief={"target_url": _URL, "max_scenarios_per_feature": 4},
        )
    )

    split = sent["narration_words_per_scenario"]
    assert isinstance(split, list)
    assert len(split) == 4
    assert sum(split) >= sent["narration_words_target"]
    assert all(isinstance(n, int) for n in split)
    assert split == [19, 19, 19, 19], "the sub-floor trailing beat is back in the payload"


async def test_a_derived_beat_count_sends_no_split() -> None:
    """⛔ THE FAILURE MODE THIS GUARDS.

    ``_scenarios_for_target`` returns a CEILING, not a quota — its own docstring
    records a derivation that asked for 15 while bdd returned 5. Splitting the budget
    across a ceiling bdd will not reach would hand every beat ~1/3 of its real share,
    i.e. instruct the writer to underfill the film. An absent key is honest.
    """
    sent = await _sent(_ctx(deliverable={"target_duration_seconds": _ORDERED}))

    assert "narration_words_per_scenario" not in sent, sent


async def test_no_order_sends_neither_budget_key() -> None:
    """No order ⇒ no duration ⇒ no word budget. Never a fabricated default."""
    for ctx in (
        _ctx(),
        _ctx(deliverable={}),
        _ctx(deliverable={"target_duration_seconds": 0}),
        _ctx(deliverable={"target_duration_seconds": True}),
    ):
        sent = await _sent(ctx)
        assert "narration_words_target" not in sent, sent
        assert "narration_words_per_scenario" not in sent, sent


async def test_the_budget_is_strictly_additive_to_the_existing_payload() -> None:
    """The duration key and the count ceiling must be untouched by 4.2."""
    sent = await _sent(_ctx(deliverable={"target_duration_seconds": _ORDERED}))

    assert sent["target_duration_seconds"] == _ORDERED
    assert sent["max_scenarios_per_feature"] == 5


# ══ THE BLOCKER: the budget must be taken at the rate the SOLVER solves at ════
#
# `demo_video_reconcile_timeline` solves the same timeline at the MEASURED wpm. When
# this budget was hardcoded to DEFAULT_WPM_NATURAL x 0.80 — an effective 120 wpm —
# against a voice measured at 151.6, bdd was asked for ~21% FEWER words than the solver
# expected. Every beat came back underfilled, `vo_rate` pinned at its 0.80 floor, and the
# delivered film played uniformly slow: the exact symptom this wave exists to remove.


# The calibration passage the voice is probed with reads FASTER than real narration. A
# showcase run measured the passage at 197.6 wpm and the drafted content at 168.8 — so a
# budget priced at the passage asked bdd for ~17% more words than `reconcile_timeline`
# (which solves at the stored CONTENT rate) could fit. website_review hit the same
# mismatch and fixed it in #1076 with `floor_wpm`; gen now prices its budget the same way.
_FAST_CALIBRATION_WPM = 197.6
_SLOW_CALIBRATION_WPM = 120.0


async def test_a_fast_calibration_rate_is_not_the_budget_rate() -> None:
    """A calibration-passage measurement FASTER than ``DEFAULT_WPM_NATURAL`` is held
    to the default — the budget is never priced at the passage's own speed.

    Red on the pre-fix code: it returned the 197.6 measurement as-is, so bdd was asked
    for 198 words for a 60s order instead of 150.
    """
    sent = await _sent(
        _ctx(deliverable={"target_duration_seconds": 60.0}), wpm=_FAST_CALIBRATION_WPM
    )

    assert sent["narration_words_target"] == round(
        60.0 * DEFAULT_WPM_NATURAL * NARRATION_BUDGET_RATE / 60
    )
    # ⛔ And NOT the calibration-derived number — the defect this test pins out.
    assert sent["narration_words_target"] != round(
        60.0 * _FAST_CALIBRATION_WPM * NARRATION_BUDGET_RATE / 60
    )


async def test_a_slow_calibration_rate_still_reaches_the_budget() -> None:
    """``wpm_natural`` is still a LIVE MEASUREMENT when it lies BELOW the default.

    ``floor_wpm`` caps a calibration rate at ``DEFAULT_WPM_NATURAL``; it does not
    replace it. A slow voice keeps its measured rate (fewer words never over-asks), so
    the budget is NOT silently re-pinned to the constant — the wave-13 failure, where a
    hardcoded rate asked bdd for words the solver did not expect.
    """
    sent = await _sent(
        _ctx(deliverable={"target_duration_seconds": 60.0}), wpm=_SLOW_CALIBRATION_WPM
    )

    assert sent["narration_words_target"] == round(
        60.0 * _SLOW_CALIBRATION_WPM * NARRATION_BUDGET_RATE / 60
    )
    assert sent["narration_words_target"] != round(
        60.0 * DEFAULT_WPM_NATURAL * NARRATION_BUDGET_RATE / 60
    )


async def test_an_unanswered_probe_falls_back_to_the_constant_rate() -> None:
    """A sick tts_backend costs the budget its calibration, never the run.

    ⚠️ THE BASELINE GENUINELY MOVED, and this test is renamed rather than quietly
    re-baselined. It used to claim the fallback number is "byte-identical to the
    pre-measurement one" — 60.0 x 150 x 0.97 / 60 = 146 words. That claim is now FALSE in
    magnitude: the budget rate moved to 1.0, so the fallback is 150 words.

    What the test actually protects is unchanged and is the only thing worth protecting
    here: an unanswered probe falls back to ``DEFAULT_WPM_NATURAL`` — the repo's own
    constant — and never to a fabricated number or to a failed run. The WPM fallback is
    the invariant; the rate factor is not part of it. Asserting "identical to the old
    number" would now mean asserting the retired 0.97 budget, i.e. pinning the defect.
    """
    sent = await _sent(_ctx(deliverable={"target_duration_seconds": 60.0}), wpm=None)

    assert sent["narration_words_target"] == round(
        60.0 * DEFAULT_WPM_NATURAL * NARRATION_BUDGET_RATE / 60
    )


async def test_an_authored_natural_rate_outranks_the_measurement() -> None:
    """The same ladder `reconcile_timeline` applies: an author who states a rate means
    it, and the solver will budget at it — so the producer must be asked for words at
    THAT rate, not at a measurement the solver is going to ignore.

    The authored 200 wpm is unchanged; only the rate factor moved, so the expected count
    goes 194 -> 200 words. The measurement is still contrasted against it below, which is
    what makes the ladder's ORDER the thing under test.
    """
    sent = await _sent(
        _ctx(
            deliverable={"target_duration_seconds": 60.0},
            direction={"timing": {"wpm": {"min": 130, "natural": 200, "max": 220}}},
        ),
        wpm=_MEASURED_WPM,
    )

    assert sent["narration_words_target"] == round(60.0 * 200.0 * NARRATION_BUDGET_RATE / 60)
    # ⛔ The authored rate WON — not the measurement the probe answered with.
    assert sent["narration_words_target"] != round(
        60.0 * _MEASURED_WPM * NARRATION_BUDGET_RATE / 60
    )


# ══ Title cards count INSIDE the order (owner ruling) ═════════════════════════
#
# The compile splices intro/outro cards into the same film, so a budget written over the
# whole order delivers `order + cards`. A replay over 31 runs found 15 of 16 overshoots
# had narration ALONE past the order by >10% — no downstream trim can recover that, so
# the budget has to reserve the cards at script time.

_OUTRO_ONLY_6S: dict[str, Any] = {
    "enabled": True,
    "intro": False,
    "outro": True,
    "duration_seconds": 6.0,
}


async def test_a_30s_order_with_a_6s_outro_budgets_24s_of_speech() -> None:
    sent = await _sent(
        _ctx(deliverable={"target_duration_seconds": 30.0}, title_card=_OUTRO_ONLY_6S)
    )

    # 24s x 150 wpm x 1.0 / 60 = 60 words, not the 75 a bare 30s order would get.
    assert sent["narration_words_target"] == round(
        24.0 * DEFAULT_WPM_NATURAL * NARRATION_BUDGET_RATE / 60
    )
    assert sent["narration_words_target"] != round(
        30.0 * DEFAULT_WPM_NATURAL * NARRATION_BUDGET_RATE / 60
    )
    # The film length bdd paces to is still the ORDER — only the words move.
    assert sent["target_duration_seconds"] == 30.0


async def test_the_card_reservation_is_capped_at_20_percent_of_the_order() -> None:
    """A 12s outro on a 30s order is capped by `plan_title_cards` to 6s (20%) — the
    budget reserves what the shared helper plans, never the raw configured seconds."""
    sent = await _sent(
        _ctx(
            deliverable={"target_duration_seconds": 30.0},
            title_card={**_OUTRO_ONLY_6S, "duration_seconds": 12.0},
        )
    )

    assert sent["narration_words_target"] == round(
        24.0 * DEFAULT_WPM_NATURAL * NARRATION_BUDGET_RATE / 60
    )


async def test_no_title_card_block_leaves_the_budget_on_the_whole_order() -> None:
    sent = await _sent(_ctx(deliverable={"target_duration_seconds": 30.0}))

    assert sent["narration_words_target"] == round(
        30.0 * DEFAULT_WPM_NATURAL * NARRATION_BUDGET_RATE / 60
    )


# ══ The LONG side: a total-word ceiling, on authored scripts too ═════════════


def _authored_ctx(words: int) -> RunContext:
    return _ctx(
        deliverable={"target_duration_seconds": 30.0},
        title_card=_OUTRO_ONLY_6S,
        brief={
            "target_url": _URL,
            "lines": [{"id": "clip-1", "text": " ".join(["word"] * words)}],
        },
    )


async def _authored_advisories(words: int) -> list[str]:
    stage = DemoVideoGenStage(client=AsyncMock(), tts_client=_tts(None))
    out = await stage.handle(stage.input_model(), _authored_ctx(words))
    return [str(a["detail"]) for a in out.advisories if "ceiling" in str(a["detail"])]


async def test_an_authored_script_over_the_word_ceiling_is_reported() -> None:
    # Ceiling = (30 - 6)s x 150 wpm / 60 x 1.10 = 66 words.
    over = await _authored_advisories(80)

    assert len(over) == 1
    assert "80 words" in over[0]
    assert "66-word ceiling" in over[0]


async def test_an_authored_script_within_the_word_ceiling_is_not_reported() -> None:
    assert await _authored_advisories(66) == []
