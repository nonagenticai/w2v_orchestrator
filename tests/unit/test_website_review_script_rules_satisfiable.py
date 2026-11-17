"""website_review_script — the prompt's rule set must be COMFORTABLY satisfiable.

The incident (live, paid, 2026-10-06, run ``01a11125-a238-7387-95dc-b0a36e8aadee``,
50s order, example.com): the redesign rendered exactly ONE section. The prompt then
said at most ONE after-act line (one per section), "at least HALF of all lines are
after-act lines" — so at most TWO lines in all — under a HARD MINIMUM of 149 words
and a ceiling of 165. Two ~75-word "lines", one of them about a section the prompt
knew only by its slug. Formally satisfiable; the model answered ``{"lines": []}`` on
every attempt (7 completion tokens, temperature 0.1). The same scenario had passed
that morning only because the model BROKE the rules: 6 lines, 2 after-act lines for
1 section.

No model is called here. This is arithmetic over the numbers the prompt states.
"""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid7

import pytest

import src.services.stages.website_review.script as website_review_script
from src.services.runs.run_context import RunContext
from src.services.stages.shared import speech_rate
from src.services.stages.shared.speech_rate import (
    CALIBRATION_TEXT,
    MAX_PLAUSIBLE_WPM,
    MIN_PLAUSIBLE_WPM,
    SpeechRate,
)
from src.services.stages.website_review.grounding import (
    MIN_AFTER_SHARE,
    RedesignInventory,
    RedesignSection,
    ScriptLineView,
    find_script_violations,
)
from src.services.stages.website_review.record import WebsiteReviewRecordStage
from src.services.stages.website_review.script import (
    COMFORTABLE_LINE_WORDS,
    WebsiteReviewScriptStage,
    after_line_allowance,
    first_draw_floor_words,
    max_lines_by_rules,
    word_budget,
)
from tests.services.stages.website_review.rate_stubs import MeasurementRecorder, measurement_at
from tests.services.stages.website_review.test_script_empty_draw_wire import _WireWM

pytestmark = pytest.mark.usefixtures("platform_llm_socket")

#: A spread of rates this stage has really budgeted from: the plausibility band's
#: edges, the low end of the per-clip scatter, measured narration, the class
#: default, the calibration passage. Parameters of the sweep — nothing is tuned to them.
_RATES = (MIN_PLAUSIBLE_WPM, 94.0, 119.4, 138.0, 150.0, 175.2, 197.6, MAX_PLAUSIBLE_WPM)
_ORDERS = tuple(range(10, 121, 5))
_SECTIONS = tuple(range(1, 9))


def _ctx(
    *,
    seconds: int = 50,
    sections: int = 1,
    voice: dict[str, Any] | None = None,
    shotlist: list[dict[str, Any]] | None = None,
) -> RunContext:
    brief: dict[str, Any] = {"duration_seconds": seconds}
    if shotlist:
        brief["shotlist"] = shotlist
    return RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot={"brief": brief, "voice": voice or {"descriptor": "warm"}},
        prior_run_outputs={
            "website_critique": {
                "url": "https://acme.test",
                "overall_score": 70.0,
                "summary": "Slow and unclear.",
                "findings": [],
            },
            "website_redesign": {
                "style": "modern",
                "visuals": [
                    {
                        "id": f"redesign-section-s{i}",
                        "role": f"redesign_section:s{i}",
                        "image_url": f"u/{i:02d}_s{i}.png",
                    }
                    for i in range(sections)
                ],
            },
        },
    )


def _stage(wpm: float | None = None) -> WebsiteReviewScriptStage:
    stage = WebsiteReviewScriptStage(client=_WireWM(empty_calls=0).client())
    if wpm is not None:
        stage._wpm = wpm
    return stage


# --------------------------------------------------------------------------- #
# the invariant
# --------------------------------------------------------------------------- #


def test_the_comfortable_line_is_the_calibration_passages_own_sentence() -> None:
    """Derived from the passage ``speech_rate`` shaped to delivered narration — not chosen."""
    sentences = [s for s in CALIBRATION_TEXT.split(". ") if s]
    assert {len(s.split()) for s in sentences} == {COMFORTABLE_LINE_WORDS}


@pytest.mark.parametrize("wpm", [_RATES[0]])
def test_every_order_can_reach_its_word_minimum_at_a_comfortable_line_length(wpm: float) -> None:
    """floor <= ceiling, and floor / max_lines <= one comfortable line. Everywhere."""
    for seconds in _ORDERS:
        ceiling = word_budget(seconds, wpm)
        floor = first_draw_floor_words(seconds, wpm)
        assert 0 <= floor < ceiling, (seconds, wpm)
        for sections in _SECTIONS:
            allowance = after_line_allowance(sections, ceiling)
            cap = max_lines_by_rules(sections, after_allowance=allowance)
            assert cap is not None
            where = f"{seconds}s order, {sections} section(s), {wpm} wpm"
            assert floor / cap <= COMFORTABLE_LINE_WORDS, (
                f"{where}: {floor} words minimum over at most {cap} lines"
            )
            # The after act can carry its half at that length too — the stated
            # split is one the model can satisfy, not just the total.
            assert allowance * COMFORTABLE_LINE_WORDS >= ceiling * MIN_AFTER_SHARE, where
            assert allowance >= sections, where


@pytest.mark.parametrize("wpm", [_RATES[0]])
def test_an_authored_before_act_leaves_the_after_act_a_reachable_minimum(wpm: float) -> None:
    """The model writes the after act ALONE — so its allowance covers the whole remainder."""
    authored_words = 12
    for seconds in _ORDERS:
        ceiling = word_budget(seconds, wpm)
        floor = first_draw_floor_words(seconds, wpm, authored_words=authored_words)
        for sections in _SECTIONS:
            allowance = after_line_allowance(
                sections, ceiling, authored_before=2, authored_words=authored_words
            )
            cap = max_lines_by_rules(sections, authored_before=2, after_allowance=allowance)
            assert cap == allowance
            assert floor / cap <= COMFORTABLE_LINE_WORDS, (seconds, sections, wpm)


def test_the_incident_one_line_per_section_was_not_comfortably_satisfiable() -> None:
    """The rule the live run was given, at the numbers its prompt stated — and the fix.

    198 wpm, 50s: a 165-word ceiling and a 149-word minimum. One after-act line per
    section capped a one-section page at two lines: 74.5 words a line.
    """
    ceiling = word_budget(50, 198.0)
    floor = first_draw_floor_words(50, 198.0)
    assert (ceiling, floor) == (165, 149)

    one_per_section = max_lines_by_rules(1)
    assert one_per_section == 2
    assert floor / one_per_section > 5 * COMFORTABLE_LINE_WORDS

    allowance = after_line_allowance(1, ceiling)
    assert allowance == 7
    cap = max_lines_by_rules(1, after_allowance=allowance)
    assert cap == 14
    assert floor / cap <= COMFORTABLE_LINE_WORDS


def test_enough_sections_for_the_order_is_still_one_line_per_section() -> None:
    assert after_line_allowance(8, word_budget(50, 198.0)) == 8
    assert after_line_allowance(12, word_budget(120, 150.0)) == 12
    assert after_line_allowance(0, 500) == 0
    assert max_lines_by_rules(0, after_allowance=0) is None


# --------------------------------------------------------------------------- #
# the prompt
# --------------------------------------------------------------------------- #

_LEGACY_SECTION_RULE = (
    "AFTER-ACT RULES: the after act walks the rendered sections above. Write at most "
    "ONE after-act line per section (at most 8 after-act lines), in page order, "
    'and set each after-act line\'s "section_id" to the section_id it describes — '
    'only ids from the list. Before-act lines set "section_id" to null.\n'
)

#: sha256 of the WHOLE prompt for an 8-section page on a 50s order, taken from the
#: commit BEFORE the allowance existed (``4ef978fd``) and compared byte-for-byte
#: against it across 1096 order x section x rate x shotlist combinations when this
#: test was written. A page with enough sections for its order must not move.
#: Re-pinned ONCE, deliberately, when the word budget became a target band (the
#: "HARD CEILING … write fewer" sentence and the HARD MINIMUM's "OVERRIDES the
#: earlier instruction" clause are the only bytes that changed).
_EIGHT_SECTION_PROMPT_SHA256 = (
    "48d1e7e56c94b1ec222d2bcf0f6b3825d334d3cb7cf08ae06534f610c856f873"  # pragma: allowlist secret
)


def test_a_multi_section_prompt_is_byte_identical_to_before() -> None:
    prompt = _stage().build_prompt(_ctx(seconds=50, sections=8))
    assert _LEGACY_SECTION_RULE in prompt
    assert "SEVERAL" not in prompt
    assert "Counted out" not in prompt
    assert hashlib.sha256(prompt.encode()).hexdigest() == _EIGHT_SECTION_PROMPT_SHA256


def test_a_one_section_prompt_says_several_lines_may_cover_the_one_section() -> None:
    """The live shape: 50s order, ONE section, the calibration-passage rate."""
    prompt = _stage(198.0).build_prompt(_ctx(seconds=50, sections=1))

    assert "50 seconds is 165 words" in prompt
    # The FLOOR is stated at the content-basis rate (``floor_wpm``: a calibration
    # 198 is held to the 150 narration default), not at 198 — 149 words there was a
    # ~64s minimum at the 138-140 wpm the content then measured.
    assert "at least 113 words in total" in prompt
    assert "ONE after-act line per section" not in prompt
    assert "(at most 1 after-act lines)" not in prompt
    assert (
        "This page has only ONE section, and this order needs more narration than one "
        "line per section can carry — so SEVERAL after-act lines may describe the SAME "
        "section, each making a different point about it. Write at most 7 after-act lines"
    ) in prompt
    assert "repeating an id across lines is expected here" in prompt
    # The half rule stays, and is counted out so it cannot cap the script at two lines.
    assert "at least HALF of all lines are after-act lines" in prompt
    assert (
        "Counted out: up to 7 after-act lines, so up to 7 before-act lines and 14 lines in all"
    ) in prompt


def test_a_two_section_prompt_names_its_section_count() -> None:
    prompt = _stage(198.0).build_prompt(_ctx(seconds=50, sections=2))
    assert "This page has only 2 sections" in prompt
    assert "Write at most 7 after-act lines" in prompt


def test_an_authored_before_act_states_no_before_split() -> None:
    shot = {"narration": "The hero buries the offer.", "seconds": 4.0, "visual": "screen"}
    prompt = _stage(198.0).build_prompt(_ctx(seconds=50, sections=1, shotlist=[shot]))
    assert "Write ONLY the AFTER act" in prompt
    assert "SEVERAL after-act lines may describe the SAME section" in prompt
    assert "Counted out" not in prompt


def test_the_allowance_never_shrinks_within_one_execution() -> None:
    """A draw is judged against the prompt it answered, not a later, slower rate's."""
    stage = _stage(198.0)
    ctx = _ctx(seconds=50, sections=1)
    assert stage._after_allowance(ctx) == 7
    stage._wpm = 138.0  # the measured rate lands lower than the prior
    assert after_line_allowance(1, word_budget(50, 138.0)) == 5
    assert stage._after_allowance(ctx) == 7


# --------------------------------------------------------------------------- #
# the post-check judges the rule the prompt gave
# --------------------------------------------------------------------------- #

_ONE = RedesignInventory(sections=(RedesignSection(id="redesign-section-about", name="about"),))


def _script(after: int) -> list[ScriptLineView]:
    return [
        ScriptLineView(text="The page has no navigation at all.", phase="before"),
        ScriptLineView(text="There is no heading to anchor it.", phase="before"),
        *[
            ScriptLineView(
                text=f"Point {i} about the single about block.",
                phase="after",
                section_id="redesign-section-about",
            )
            for i in range(after)
        ],
    ]


def test_several_after_lines_on_one_section_are_not_a_violation_under_the_allowance() -> None:
    found = find_script_violations(_script(3), _ONE, after_allowance=7)
    assert not found
    assert found.imbalance == []
    assert found.mapping == []


def test_without_an_allowance_the_one_line_per_section_rule_still_judges() -> None:
    for allowance in (None, 1):
        found = find_script_violations(_script(3), _ONE, after_allowance=allowance)
        assert found.imbalance == [
            "the after act has 3 lines for 1 rendered sections — at most one line per section"
        ]
        assert found.mapping == ["two after-act lines point at the same section"]


def test_more_after_lines_than_the_allowance_is_still_an_imbalance() -> None:
    found = find_script_violations(_script(8), _ONE, after_allowance=7)
    assert found.imbalance == [
        "the after act has 8 lines for 1 rendered section(s) — at most 7 after-act "
        "lines for this order"
    ]


def test_the_half_rule_is_kept_under_the_allowance() -> None:
    lines = [*_script(1), ScriptLineView(text="And no call to action.", phase="before")]
    found = find_script_violations(lines, _ONE, after_allowance=7)
    assert found.imbalance == [
        "the after act (the redesign) has 1 of 4 lines — it must carry at least 50%"
    ]


def test_the_grounding_re_ask_does_not_take_the_allowance_back() -> None:
    legacy = website_review_script._grounding_reask_block(["x"])
    assert "one line per section, each with its section_id" in legacy
    shared = website_review_script._grounding_reask_block(["x"], shared_sections=True)
    assert "one line per section" not in shared
    assert "several lines may share a section, each with its section_id" in shared


# --------------------------------------------------------------------------- #
# the rate the floor and ceiling are computed from is the NARRATOR's
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("voice_id", ["af_heart"])
async def test_the_script_rate_probe_uses_the_voice_narrate_will_speak_in(
    monkeypatch: pytest.MonkeyPatch, voice_id: str | None
) -> None:
    """The Phase-A probe's voice key IS ``website_review_record.voice_for``'s answer.

    ``voice=None`` in a probe log line is not a mismatch: with no ``voice.voice_id``
    configured, narrate requests ``None`` as well (the backend's default voice), so
    probe, per-line measurement and narration all key on the same voice. The gap
    between a ~198 wpm prior and ~138 wpm delivered narration is the calibration
    PASSAGE against real content (``speech_rate`` Phase A vs Phase C), not a voice.
    """
    voice: dict[str, Any] = {"descriptor": "warm"}
    if voice_id is not None:
        voice["voice_id"] = voice_id
    ctx = _ctx(seconds=50, sections=1, voice=voice)
    narrate_voice = WebsiteReviewRecordStage(client=MagicMock()).voice_for(
        {"speaker": "a"}, voice, ctx
    )
    assert narrate_voice == voice_id

    probed: list[tuple[str | None, str | None]] = []

    async def _probe(_client: object, *, voice: str | None, provider: str | None = None) -> float:
        probed.append((voice, provider))
        return 150.0

    monkeypatch.setattr(speech_rate, "measure_natural_wpm", _probe)
    recorder = MeasurementRecorder()

    async def _measure(_ctx: Any, *, texts: Any, voices: Any, **_kw: Any) -> Any:
        recorder.calls.append((tuple(texts), tuple(voices)))
        return measurement_at(texts, voices, {}, 150.0, SpeechRate(wpm=150.0, measured=True))

    monkeypatch.setattr(website_review_script, "measure_script_lines", _measure)

    stage = _stage()
    await stage.run({}, ctx=ctx)

    assert probed, "the script stage never probed a rate"
    assert {v for v, _ in probed} == {narrate_voice}
    assert recorder.calls, "the script stage never measured its lines"
    assert set(recorder.voices) == {narrate_voice}
