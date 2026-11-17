"""A short ``website_review`` draw may not ship as if it were fine.

The incident: a customer ordered a 50s ``website_review`` film and received ~15s.
In this pipeline a film is exactly as long as its narration (``compile`` slaves the
picture to the voice-over and nothing downstream lengthens it), so a short script
draft IS a short film. Before this file the short side had re-asks, but none of
them stated a MINIMUM — the standing prompt calls the word budget a "HARD CEILING
… write fewer if the story is done sooner", which is an instruction to under-write —
and a draw that survived every ask shipped with only an ADVISORY, the tier that
does not flip the run ``degraded``.

Same class PR #1063 closed for ``app_review`` and ``promo`` (a draw the corrective
levers could not move is re-asked at an explicit word target, and what cannot be
corrected is recorded on the DEGRADING tier, never raised). These tests drive the
REAL stage with only the WM client and the TTS measurement faked.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

import src.services.stages.website_review.script as website_review_script
from src.services.runs.run_context import RunContext
from src.services.stages.shared.degradation import (
    DURATION_OUT_OF_TOLERANCE,
    NARRATION_ESTIMATE_OFF_TARGET,
    is_degraded,
)
from src.services.stages.shared.narration_budget import DEFAULT_TOLERANCE
from src.services.stages.website_review.script import WebsiteReviewScriptStage
from tests.services.stages.website_review.rate_stubs import stub_speech_rate

#: 1 word = 1 second, so "words" and "seconds" are one axis and every number below
#: is exact arithmetic.
WPM = 60.0
ORDERED_SECONDS = 50
#: The customer's draw: ~15s of narration against the 50s order (-70%).
SHORT_WORDS = 15

_FLOOR = re.compile(r"at least (\d+) words", re.IGNORECASE)
#: The floor as the RE-ASK states it (``_reask_prompt_block``). The first-draw prompt
#: states a minimum too ("you must write at least ONE line — … at least N words"),
#: so ``_FLOOR`` alone can no longer tell a re-ask from a first draw.
_REASK_FLOOR = re.compile(
    r"REWRITE — .*?the rewritten script must contain at least (\d+) words", re.DOTALL
)


def _resp(lines: list[dict[str, Any]]) -> WMResponse:
    return WMResponse(
        output={"lines": lines},
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _script(words: int) -> list[dict[str, Any]]:
    """A two-act script of exactly ``words`` words ⇒ ``words`` seconds at 60 wpm."""
    before = words // 2
    after = words - before
    return [
        {
            "text": " ".join(["word"] * before),
            "seconds": float(before),
            "phase": "before",
            "speaker": "a",
        },
        {
            "text": "Here is the new redesign " + " ".join(["word"] * (after - 5)),
            "seconds": float(after),
            "phase": "after",
            "speaker": "b",
        },
    ]


def _ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"duration_seconds": ORDERED_SECONDS},
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


def _ordered(ctx: RunContext) -> float:
    """The seconds the script may SPEAK — read off the stage, never restated."""
    return website_review_script._target_seconds(ctx.config_snapshot)


def _wm_obeying_only_a_stated_floor(prompts: list[str]) -> AsyncMock:
    """A model that under-writes its FIRST draw, then honours a re-ask's MINIMUM.

    That is the measured behaviour the fix is aimed at ("the model habitually
    writes FEWER words than budgeted"), reduced to its discriminating property: the
    first draw comes back ~15s EVEN THOUGH its prompt states a minimum (a stated
    floor is an ask, not a guarantee); the REWRITE block that says "the rewritten
    script must contain at least N words" gets N words.
    """
    fake = AsyncMock()

    async def _message(**kwargs: Any) -> WMResponse:
        prompt = str(kwargs.get("message", ""))
        prompts.append(prompt)
        stated = _REASK_FLOOR.search(prompt)
        return _resp(_script(int(stated.group(1)) if stated else SHORT_WORDS))

    fake.message.side_effect = _message
    return fake


async def _run(monkeypatch: pytest.MonkeyPatch, fake: AsyncMock, ctx: RunContext) -> Any:
    stub_speech_rate(monkeypatch, website_review_script, WPM)
    return await WebsiteReviewScriptStage(client=fake).run({}, ctx=ctx)


def _codes(entries: list[dict[str, str]]) -> list[str]:
    return [entry["code"] for entry in entries]


@pytest.mark.asyncio
async def test_a_50s_order_drawn_at_15s_is_re_asked_with_a_word_floor_and_lands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The incident, end to end: 15s first draw, 50s order — what ships is in band.

    The re-ask must state the floor as a COUNTABLE number at or above the words the
    band's lower edge needs; a model that only lengthens when given a minimum then
    lands, and a draw that landed carries no duration degradation.
    """
    ctx = _ctx()
    ordered = _ordered(ctx)
    floor_seconds = ordered * (1.0 - DEFAULT_TOLERANCE)
    prompts: list[str] = []
    out = await _run(monkeypatch, _wm_obeying_only_a_stated_floor(prompts), ctx)

    assert out.narration_estimate is not None
    assert out.narration_estimate.first_draw_estimate_seconds == pytest.approx(SHORT_WORDS)
    # The first draw stated a minimum and the model under-wrote it anyway.
    assert "HARD MINIMUM" in prompts[0] and _FLOOR.search(prompts[0])
    assert "REWRITE" not in prompts[0]
    floors = [int(m.group(1)) for p in prompts if (m := _REASK_FLOOR.search(p))]
    assert floors, "a short draw must be re-asked with an explicit word floor"
    assert all(words >= floor_seconds * WPM / 60.0 for words in floors), (
        "the stated floor must be enough words to reach the band's lower edge"
    )
    assert out.total_seconds >= floor_seconds, (
        f"a {ORDERED_SECONDS}s order drawn at {SHORT_WORDS}s shipped "
        f"{out.total_seconds}s — below the {floor_seconds:.1f}s floor"
    )
    assert out.narration_estimate.verdict == "ok"
    assert DURATION_OUT_OF_TOLERANCE not in _codes(out.degradations)


@pytest.mark.asyncio
async def test_a_draw_still_below_the_floor_after_every_ask_degrades_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model ignores every ask: 15s ships — DEGRADED, with the floor it missed.

    Mirrors ``app_review``: an uncorrectable draw is not raised (a paid run that
    holds a usable script must not die on a correction) but it is recorded on the
    tier that flips the run ``degraded``. An advisory alone is the silent ship.
    """
    fake = AsyncMock()
    fake.message.return_value = _resp(_script(SHORT_WORDS))
    out = await _run(monkeypatch, fake, _ctx())

    assert fake.message.await_count >= 2, "the short draw was re-asked at all"
    assert out.total_seconds == float(SHORT_WORDS), "never padded to fake the length"
    assert out.narration_estimate is not None
    assert out.narration_estimate.verdict == "short"
    below = [d for d in out.degradations if d["code"] == DURATION_OUT_OF_TOLERANCE]
    assert len(below) == 1, "shipping below the floor is a DEGRADATION, said once"
    assert is_degraded(out.model_dump())
    assert "15.0s" in below[0]["detail"]
    assert _FLOOR.search(below[0]["detail"]), "it names the word floor that was not reached"
    assert NARRATION_ESTIMATE_OFF_TARGET in _codes(out.advisories), "the steering note stays"


@pytest.mark.asyncio
async def test_a_failed_re_ask_on_a_short_draw_degrades_instead_of_shipping_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WM dies after the first draw: the 15s draft is kept, never raised, DEGRADED.

    Before the fix this path logged a warning, kept the first draw and recorded an
    advisory whose action read ``kept first draw`` — indistinguishable from a run
    that never needed correcting.
    """
    fake = AsyncMock()
    calls = {"n": 0}

    async def _message(**_kwargs: Any) -> WMResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return _resp(_script(SHORT_WORDS))
        raise RuntimeError("WM is down")

    fake.message.side_effect = _message
    out = await _run(monkeypatch, fake, _ctx())

    assert out.lines, "the first draw survived the failed re-ask — the run did not die"
    assert out.total_seconds == float(SHORT_WORDS)
    below = [d for d in out.degradations if d["code"] == DURATION_OUT_OF_TOLERANCE]
    assert len(below) == 1
    assert "re-ask failed" in below[0]["detail"], "the failure is named, not swallowed"
    assert is_degraded(out.model_dump())
