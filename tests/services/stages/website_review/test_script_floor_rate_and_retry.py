"""website_review_script — the floor's RATE, the retry's COUNTS, the band-only re-ask.

Live 2026-10-06 (r3-wr50s / r3b kill-capture, 50s order, example.com, ONE rendered
section): the Phase-A calibration passage measured ~198 wpm, so the first draw was
floored at 149 words — while the drafted content then measured 138-140 wpm, i.e. the
minimum ALONE was a ~64s script for a 50s order. Attempt two was attempt one plus a
paragraph restating the same word minimum, at the same temperature 0.1, and the
model answered ``{"lines": []}`` both times.

What is pinned here, each against origin/main before this change:

* (d) a word FLOOR is stated at a content-basis rate (``floor_wpm``), never at the
  calibration-passage rate itself; the CEILING sentence is unchanged.
* (b) the under-floor retry states the minimum as LINE counts, split across acts.
* the grounding re-ask does not demand "ONLY sections and elements listed" when no
  section lists anything (band-only inventory).

Not here, by decision: ``minItems`` on ``lines`` (see the module docstring of
``tests/test_output_schema_bounds.py`` and ``test_script_empty_draw_wire.py`` —
worker-management strips it from the grammar, and the repo bans it).
"""

from __future__ import annotations

import math
from uuid import uuid7

import pytest

import src.services.stages.website_review.script as website_review_script
from src.services.runs.run_context import RunContext
from src.services.stages.shared.narration_budget import DEFAULT_TOLERANCE
from src.services.stages.shared.speech_rate import (
    AUTHORED,
    BASIS_MEASURED_ALL,
    CALIBRATION_PASSAGE,
)
from src.services.stages.shared.time_budget import DEFAULT_WPM_NATURAL
from src.services.stages.website_review.grounding import (
    MIN_AFTER_SHARE,
    RedesignInventory,
    RedesignSection,
    inventory_is_band_only,
)
from src.services.stages.website_review.script import (
    COMFORTABLE_LINE_WORDS,
    WebsiteReviewScriptStage,
    first_draw_floor_words,
    floor_wpm,
    minimum_line_plan,
)
from tests.services.stages.website_review.rate_stubs import stub_speech_rate
from tests.services.stages.website_review.test_script_empty_draw_wire import _WireWM
from tests.support.platform_llm import user_text

pytestmark = pytest.mark.usefixtures("platform_llm_socket")

#: What the calibration passage measured on the live run, and the range the drafted
#: CONTENT then measured at on the same voice. Observations, not tuning.
_CALIBRATION_WPM = 197.6
_CONTENT_WPM = (138.0, 140.0)


def _one_section_ctx(seconds: int = 50) -> RunContext:
    return RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot={"brief": {"duration_seconds": seconds}, "voice": {"descriptor": "warm"}},
        prior_run_outputs={
            "website_critique": {
                "url": "https://example.com",
                "overall_score": 68.4,
                "summary": "Thin page.",
                "findings": [],
            },
            "website_redesign": {
                "style": "modern",
                "visuals": [
                    {
                        "id": "redesign-section-about-this-domain",
                        "role": "redesign_section:about-this-domain",
                        "image_url": "http://minio.test/r/sections/00_about-this-domain.png",
                    }
                ],
            },
        },
    )


def _stated_floor(prompt: str) -> int:
    marker = "at least "
    tail = prompt.split("HARD MINIMUM", 1)[1]
    for chunk in tail.split(marker)[1:]:
        number = chunk.split(" ", 1)[0]
        if number.isdigit() and chunk.split(" ", 2)[1] == "words":
            return int(number)
    raise AssertionError("no word minimum stated")


# --------------------------------------------------------------------------- #
# (d) the floor's rate
# --------------------------------------------------------------------------- #


def test_a_calibration_rate_is_held_to_the_narration_default_for_a_floor() -> None:
    assert floor_wpm(_CALIBRATION_WPM, CALIBRATION_PASSAGE) == DEFAULT_WPM_NATURAL
    # A slower calibration never RAISES the floor.
    assert floor_wpm(120.0, CALIBRATION_PASSAGE) == 120.0
    # Content-measured and authored rates are already the right basis.
    assert floor_wpm(_CALIBRATION_WPM, BASIS_MEASURED_ALL) == _CALIBRATION_WPM
    assert floor_wpm(_CALIBRATION_WPM, AUTHORED) == _CALIBRATION_WPM


def test_the_first_draw_floor_admits_an_in_band_script_at_the_content_rate() -> None:
    """The live shape: the stated minimum, spoken at content pace, fits the band.

    origin/main stated 149 words: 149 / 139 wpm = 64.3s on a 50s order (+29%), so a
    model that obeyed the minimum could only deliver an out-of-band film.
    """
    stage = WebsiteReviewScriptStage(client=_WireWM(empty_calls=0).client())
    stage._wpm = _CALIBRATION_WPM
    stage._wpm_basis = CALIBRATION_PASSAGE
    prompt = stage.build_prompt(_one_section_ctx(50))

    # The CEILING is still the measured prior's — unchanged.
    assert "pace of 198 words per minute, 50 seconds is 165 words" in prompt
    floor = _stated_floor(prompt)
    for content_wpm in _CONTENT_WPM:
        spoken = floor * 60.0 / content_wpm
        assert spoken <= 50 * (1.0 + DEFAULT_TOLERANCE), (floor, content_wpm, spoken)


def test_a_measured_rate_floors_at_itself() -> None:
    stage = WebsiteReviewScriptStage(client=_WireWM(empty_calls=0).client())
    stage._wpm = 139.0
    stage._wpm_basis = BASIS_MEASURED_ALL
    prompt = stage.build_prompt(_one_section_ctx(50))
    assert _stated_floor(prompt) == first_draw_floor_words(50, 139.0)


# --------------------------------------------------------------------------- #
# (b) the retry is a different, countable ask
# --------------------------------------------------------------------------- #


async def test_the_retry_after_an_empty_draw_states_line_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Driven through the real WMClient body: attempt 2 != attempt 1 + the same rule."""
    stub_speech_rate(monkeypatch, website_review_script, _CALIBRATION_WPM)
    wire = _WireWM(empty_calls=1)
    stage = WebsiteReviewScriptStage(client=wire.client())

    await stage.run({}, ctx=_one_section_ctx(50))

    first, retry = user_text(wire.bodies[0]), user_text(wire.bodies[1])
    assert retry.startswith(first)
    added = retry[len(first) :]
    floor = first_draw_floor_words(50, floor_wpm(_CALIBRATION_WPM, CALIBRATION_PASSAGE))
    lines = math.ceil(floor / COMFORTABLE_LINE_WORDS)
    after = math.ceil(lines * MIN_AFTER_SHARE)
    assert 'an empty "lines" list is never a valid answer' in added
    assert f"at least {floor} words in total" in added
    assert f"takes at least {lines} lines" in added
    assert f"{lines - after} before-act line(s) then {after} after-act line(s)" in added
    assert f'Return a "lines" list with at least {lines} elements.' in added
    # And the floor in BOTH bodies is the content-basis one, not 149.
    assert "at least 149 words" not in retry


def test_the_line_plan_respects_the_rule_cap_and_the_authored_path() -> None:
    assert minimum_line_plan(0, cap=None, authored_before=False) == ""
    assert "at least 3 lines — all 3 of them after-act lines" in minimum_line_plan(
        30, cap=None, authored_before=True
    )
    # Never above the cap the prompt's own rules state.
    assert "at least 2 lines" in minimum_line_plan(100, cap=2, authored_before=False)


# --------------------------------------------------------------------------- #
# the grounding re-ask on a band-only inventory
# --------------------------------------------------------------------------- #


def test_band_only_is_the_grounding_predicate() -> None:
    """script.py's re-ask reads the SAME predicate as grounding's first ask."""
    assert website_review_script.inventory_is_band_only is inventory_is_band_only
    band = RedesignInventory(
        sections=(
            RedesignSection(id="redesign-section-band-01", name="band-01"),
            RedesignSection(id="redesign-section-band-02", name="band-02"),
        )
    )
    band_with_heading = RedesignInventory(
        sections=(RedesignSection(id="redesign-section-band-01", name="band-01", heading="Plans"),)
    )
    # A NAMED section with no readable HTML is not band-only: its name is groundable.
    named_empty = RedesignInventory(
        sections=(RedesignSection(id="redesign-section-s0", name="s0"),)
    )
    mixed = RedesignInventory(sections=(*band.sections, *named_empty.sections))
    assert inventory_is_band_only(band)
    assert not inventory_is_band_only(band_with_heading)
    assert not inventory_is_band_only(named_empty)
    assert not inventory_is_band_only(mixed)
    assert not inventory_is_band_only(RedesignInventory())
    # The live one-section run (slug only, no HTML) keeps the strict named rule.
    stage = WebsiteReviewScriptStage(client=_WireWM(empty_calls=0).client())
    assert not inventory_is_band_only(stage._inventory_for(_one_section_ctx()))
