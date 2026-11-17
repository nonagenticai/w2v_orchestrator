"""website_review_script — the EMPTY draw, driven through the REAL ``WMClient`` wire path.

The incident (live, paid, 2026-10-06, run ``01a10e19-145f-7475-a34a-aa1fc8d3113f``,
50s order): the script call answered HTTP 200 with ``{"lines": []}`` — 2132 prompt
tokens, 7 completion tokens — and nothing rejected it. The stage persisted 0 lines
against a 165-word budget as a SUCCESS, ``visual_director`` / ``slide_plan`` /
``record`` / ``slides`` each no-op'd in under a second, and the run died three
stages later in the wrong place: ``website_review_compile: no shots to compile``.

A zero-word draw is the limiting case of the short draw
(``tests/unit/test_website_review_short_draw_floor.py``) and it was the one case
that floor skipped — ``_reconcile`` returned early on "no lines".

Same shape as ``action_reel/test_segment_select_empty_draw_wire.py``: the floor
cannot live in the grammar (worker-management strips ``minItems``;
``tests/test_output_schema_bounds.py`` bans it), so it lives in the PARSER, and
these tests go through the Chat Completions body the stage actually POSTs (``LLM_BACKEND=vllm``, the
backend prod runs) rather than an ``AsyncMock`` that skips ``WMClient.message``.
"""

from __future__ import annotations

import json
import re
from typing import Any
from uuid import uuid7

import pytest
from w2v_common.stages import StageRetriesExhaustedError

import src.services.stages.website_review.script as website_review_script
from src.services.llm import WMClient
from src.services.runs.run_context import RunContext
from src.services.stages.llm_budget import llm_retries
from src.services.stages.primitives.script_stage import ScriptDrawUnderFloorError
from src.services.stages.shared.degradation import DURATION_OUT_OF_TOLERANCE
from src.services.stages.shared.empty_draw import EmptyScriptDrawError
from src.services.stages.website_review.script import WebsiteReviewScriptStage
from tests.support.platform_llm import (
    CHAT_PATH,
    PlatformChat,
    posted_schema,
    system_text,
    user_text,
)

pytestmark = pytest.mark.usefixtures("platform_llm_socket")


_VALID_LINES: list[dict[str, Any]] = [
    {
        "text": "The current hero buries the offer under three competing banners.",
        "seconds": 4.0,
        "phase": "before",
        "speaker": "a",
    },
    {
        "text": "Here is the new redesign with one clear headline and one action.",
        "seconds": 4.0,
        "phase": "after",
        "speaker": "a",
    },
]


def _ctx(*, site_capture: dict[str, Any] | None = None) -> RunContext:
    outputs: dict[str, Any] = {
        "website_critique": {
            "url": "https://acme.test",
            "overall_score": 70.0,
            "summary": "Slow and unclear.",
            "findings": [],
        },
        "website_redesign": {"style": "modern", "visuals": []},
    }
    if site_capture is not None:
        outputs["site_capture"] = site_capture
    return RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot={"brief": {"duration_seconds": 50}, "voice": {"descriptor": "warm"}},
        prior_run_outputs=outputs,
    )


class _WireWM:
    """The platform LLM (Chat Completions) behind an ``httpx2.MockTransport``.

    Answers the first ``empty_calls`` calls with ``{"lines": []}`` — the
    schema-valid, HTTP-200 answer the live run got — and every later one with a
    real two-act script. Records every request body.
    """

    def __init__(self, *, empty_calls: int) -> None:
        self.empty_calls = empty_calls
        self.chat = PlatformChat(self.answer)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return self.chat.bodies

    @property
    def paths(self) -> list[str]:
        return self.chat.paths

    def answer(self, _body: dict[str, Any], n: int) -> dict[str, Any]:
        return {"lines": [] if n < self.empty_calls else _VALID_LINES}

    def client(self) -> WMClient:
        # Route the platform socket here; everything above the transport (body
        # building, schema strip, response parsing) is real.
        self.chat.activate()
        return WMClient(timeout=30.0)


def test_the_wire_schema_puts_no_length_floor_on_lines() -> None:
    """The grammar CANNOT forbid ``{"lines": []}`` — the parser is the floor."""
    stage = WebsiteReviewScriptStage(client=_WireWM(empty_calls=0).client())
    assert stage.min_elements == 1
    assert not {"minItems", "maxItems"} & set(stage.element_schema)


async def test_the_posted_schema_requires_one_line_in_the_grammar() -> None:
    """The body WM receives carries ``prefixItems: [ITEM]`` under ``lines``.

    The plain ``items``-only array let xgrammar steer ``[`` straight to ``]`` (6/6
    empty on the live model; ``primitives.script_stage.non_empty_array``).
    """
    wire = _WireWM(empty_calls=0)
    await WebsiteReviewScriptStage(client=wire.client()).run({}, ctx=_ctx())
    assert wire.bodies
    for body in wire.bodies:
        schema = posted_schema(body)
        assert schema is not None, "the call must carry the json_schema grammar"
        lines = schema["properties"]["lines"]
        assert lines["prefixItems"] == [lines["items"]]
        assert lines["items"] == WebsiteReviewScriptStage.element_schema


async def test_an_all_empty_wire_answer_fails_at_the_script_stage_never_shipped() -> None:
    attempts = llm_retries() + 1
    assert attempts >= 2, "this test needs a retry budget to have anything to prove"
    wire = _WireWM(empty_calls=10**6)
    stage = WebsiteReviewScriptStage(client=wire.client())

    # Base commit: this returned a WebsiteReviewScriptOutput(lines=[]) — a success.
    with pytest.raises(StageRetriesExhaustedError, match="empty script draw") as caught:
        await stage.run({}, ctx=_ctx())

    # The failure is the SCRIPT stage's, and it names the real problem.
    assert caught.value.stage_name == "website_review_script"
    assert caught.value.attempts == attempts
    assert isinstance(caught.value.last_error, EmptyScriptDrawError)
    assert isinstance(caught.value.__cause__, StageRetriesExhaustedError)
    assert isinstance(caught.value.__cause__.last_error, ScriptDrawUnderFloorError)
    # Re-asked exactly the retry budget, then stopped: no grounding re-ask, no
    # duration re-ask, no measurement — nothing downstream of a script that is not
    # there.
    assert len(wire.bodies) == attempts
    assert set(wire.paths) == {CHAT_PATH}
    # The scrape is never in this prompt ⇒ nothing is claimed about one.
    assert "site capture" not in str(caught.value)


async def test_an_empty_first_attempt_recovers_on_the_re_prompt() -> None:
    wire = _WireWM(empty_calls=1)
    stage = WebsiteReviewScriptStage(client=wire.client())

    out = await stage.run({}, ctx=_ctx())

    assert [line.text for line in out.lines] == [line["text"] for line in _VALID_LINES]
    # Exactly ONE extra call for the empty draw, inside the first draw's retry
    # budget. Compared against the same model answering validly from the start, so
    # the corrective re-asks this short script also earns cancel out.
    baseline = _WireWM(empty_calls=0)
    await WebsiteReviewScriptStage(client=baseline.client()).run({}, ctx=_ctx())
    assert len(wire.bodies) == len(baseline.bodies) + 1


def _one_section_ctx() -> RunContext:
    """The live shape: example.com, ONE rendered redesign section, a 50s order."""
    ctx = _ctx(
        site_capture={
            "clips": [],
            "page_text": "x" * 171,
            "page_text_status": "thin",
            "capture_status": "degraded",
            "notes": ["page scrape read only 171 characters from https://acme.test"],
        }
    )
    ctx.prior_run_outputs["website_redesign"]["visuals"] = [
        {
            "id": "redesign-section-example-domain",
            "role": "redesign_section:example-domain",
            "image_url": "http://minio.test/r/sections/00_example-domain.png",
        }
    ]
    return ctx


async def test_the_empty_draw_error_names_what_the_prompt_had_not_the_page_scrape() -> None:
    """The scrape is not in this prompt, so a thin one explains nothing.

    #1068 appended "site capture read 171 chars … the model had almost nothing to
    ground the script on" here. ``build_prompt`` never contains ``page_text``; the
    model's actual inputs were one rendered section, a word budget, and the line
    cap its rules state. Those are what the error names. (The cap WAS two lines —
    one after-act line per section — until the allowance began scaling with the
    order: ``tests/unit/test_website_review_script_rules_satisfiable.py``.)
    """
    wire = _WireWM(empty_calls=10**6)
    stage = WebsiteReviewScriptStage(client=wire.client())

    with pytest.raises(StageRetriesExhaustedError, match="empty script draw") as caught:
        await stage.run({}, ctx=_one_section_ctx())

    message = str(caught.value)
    assert "site capture" not in message
    assert "ground the script on" not in message
    assert message.endswith(
        "; the script prompt had 1 rendered redesign section(s) for a 50s order — "
        "50s of narration, 125 words at 150 wpm — and its rules allow at most 10 "
        "line(s) (up to 5 after-act line(s) over 1 section(s), at least half of all "
        "lines after-act)"
    )
    # Every number in that clause is one the prompt the model received states.
    prompt = user_text(wire.bodies[0])
    assert "x" * 171 not in prompt, "the scrape must not be in this prompt at all"
    assert "Write a 50-second website review" in prompt
    assert "50 seconds is 125 words" in prompt
    assert "Write at most 5 after-act lines" in prompt
    assert "at least HALF of all lines are after-act lines" in prompt


async def test_the_empty_draw_error_says_so_when_no_sections_cap_the_script() -> None:
    stage = WebsiteReviewScriptStage(client=_WireWM(empty_calls=10**6).client())

    with pytest.raises(StageRetriesExhaustedError, match="empty script draw") as caught:
        await stage.run({}, ctx=_ctx())

    assert "0 rendered redesign section(s)" in str(caught.value)
    assert "no rendered sections, so its rules cap no line count" in str(caught.value)


@pytest.mark.parametrize("sections", list(range(1, 9))[:1])
def test_the_line_cap_is_the_two_caps_the_prompt_states(sections: int) -> None:
    """``max_lines_by_rules`` re-derived from the PROMPT TEXT, not from its own formula."""
    ctx = _ctx()
    ctx.prior_run_outputs["website_redesign"]["visuals"] = [
        {
            "id": f"redesign-section-s{i}",
            "role": f"redesign_section:s{i}",
            "image_url": f"u/{i:02d}_s{i}.png",
        }
        for i in range(sections)
    ]
    stage = WebsiteReviewScriptStage(client=_WireWM(empty_calls=0).client())

    prompt = stage.build_prompt(ctx)
    stated = re.search(r"at most (\d+) after-act lines", prompt)
    assert stated is not None
    after_cap = int(stated.group(1))
    # One line per section, or — when the page has too few sections for a 50s order
    # (125 words at the 150 wpm class default) — the order-scaled allowance.
    allowance = website_review_script.after_line_allowance(sections, 125)
    assert after_cap == allowance == max(sections, 5)
    assert ("ONE after-act line per section" in prompt) == (sections >= 5)
    assert "at least HALF of all lines are after-act lines" in prompt
    # after <= cap and after >= total / 2  ⇒  total <= 2 x cap.
    assert (
        website_review_script.max_lines_by_rules(sections, after_allowance=allowance)
        == 2 * after_cap
    )

    # An authored before act: the model writes the after act alone.
    ctx.config_snapshot["brief"]["shotlist"] = [
        {"narration": "The hero buries the offer.", "seconds": 4.0, "visual": "screen"},
    ]
    # A fresh stage: the allowance is held at the largest one execution has stated.
    authored = WebsiteReviewScriptStage(client=_WireWM(empty_calls=0).client()).build_prompt(ctx)
    assert "Write ONLY the AFTER act" in authored
    assert "at least HALF" not in authored
    stated = re.search(r"at most (\d+) after-act lines", authored)
    assert stated is not None
    after_alone = int(stated.group(1))
    # The model writes the after act alone, so the WHOLE remaining budget is its.
    assert after_alone == website_review_script.after_line_allowance(
        sections, 125, authored_before=1, authored_words=5
    )
    assert (
        website_review_script.max_lines_by_rules(
            sections, authored_before=1, after_allowance=after_alone
        )
        == after_alone
    )
    # The legacy two-argument forms still answer the one-line-per-section rule.
    assert website_review_script.max_lines_by_rules(sections) == 2 * sections
    assert website_review_script.max_lines_by_rules(sections, authored_before=1) == sections


def test_no_rendered_sections_means_no_line_cap() -> None:
    stage = WebsiteReviewScriptStage(client=_WireWM(empty_calls=0).client())
    assert "AFTER-ACT RULES" not in stage.build_prompt(_ctx())
    assert website_review_script.max_lines_by_rules(0) is None


async def test_an_empty_duration_re_ask_keeps_the_draw_in_hand() -> None:
    """Only the FIRST draw is fatal — a correction that draws empty is a failed correction.

    The valid script here is far short of a 50s order, so the stage re-asks. Every
    re-ask answers empty; the stage must still ship the script it already holds
    (degraded for length by the short-draw floor), never raise and never ship zero.
    """

    # WORST CASE: the before act narrates a FIX, so the grounding re-ask fires too
    # — every corrective lever this stage has is pulled, and each answers empty.
    first = [
        {**_VALID_LINES[0], "text": "The hero is cluttered, but we fixed the headline."},
        _VALID_LINES[1],
    ]

    class _ValidThenEmpty(_WireWM):
        def answer(self, _body: dict[str, Any], n: int) -> dict[str, Any]:
            return {"lines": first if n == 0 else []}

    wire = _ValidThenEmpty(empty_calls=0)
    out = await WebsiteReviewScriptStage(client=wire.client()).run({}, ctx=_ctx())

    assert [line.text for line in out.lines] == [raw["text"] for raw in first]
    # EXACTLY origin/main's spend on this sequence: the draw, the grounding re-ask,
    # the duration re-ask and the short-side second re-ask. Each empty correction
    # costs ONE call — it is parsed with no floor (``corrective_draw``), never
    # re-prompted. With the floor applied to corrections too this was 7.
    assert len(wire.bodies) == 4
    assert "Rewrite the complete script" in user_text(wire.bodies[1]), "grounding re-ask fired"
    # The first re-ask WAS sent and WAS discarded — never "no re-ask happened".
    estimate = out.narration_estimate
    assert estimate is not None
    assert (estimate.reasked, estimate.reask_kept) == (True, False)
    # And the short film it ships is a DEGRADATION that names what happened.
    short = [e for e in out.degradations if e["code"] == DURATION_OUT_OF_TOLERANCE]
    assert short and "re-ask came back empty" in short[0]["detail"]


async def test_no_narration_budget_is_the_only_unreconciled_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one early return left in ``_reconcile``: no order ⇒ no band ⇒ no verdict.

    "No lines" used to leave by the same door. It no longer can: a script with
    lines and no budget ships unreconciled (``narration_estimate is None``); a
    script with NO lines never reaches ``_reconcile`` at all (tests above).
    """
    monkeypatch.setattr(website_review_script, "_target_seconds", lambda _snapshot: 0.0)
    wire = _WireWM(empty_calls=0)

    out = await WebsiteReviewScriptStage(client=wire.client()).run({}, ctx=_ctx())

    assert len(out.lines) == len(_VALID_LINES)
    assert out.narration_estimate is None
    assert len(wire.bodies) == 1


async def test_an_authored_shotlist_does_not_make_an_empty_after_act_draw_legitimate() -> None:
    """The operator wrote the BEFORE act; the model owes the AFTER act — the reveal.

    The after-only prompt says the model's "FIRST line must clearly PRESENT the
    redesign", so "nothing to add" is not an answer it admits: shipping the
    authored before lines alone would be a review that never shows the redesign
    the run already rendered. The authored lines must not mask the empty draw.
    """
    attempts = llm_retries() + 1
    ctx = _ctx()
    ctx.config_snapshot["brief"]["shotlist"] = [
        {"narration": "The hero buries the offer.", "seconds": 4.0, "visual": "screen"},
    ]
    assert website_review_script.normalize_shotlist(ctx.config_snapshot["brief"]), (
        "the shotlist must be recognised or this test proves nothing about it"
    )
    wire = _WireWM(empty_calls=10**6)
    stage = WebsiteReviewScriptStage(client=wire.client())

    with pytest.raises(StageRetriesExhaustedError, match="empty script draw"):
        await stage.run({}, ctx=ctx)

    assert len(wire.bodies) == attempts
    assert "Write ONLY the AFTER act" in user_text(wire.bodies[0])


# --- the first draw states a MINIMUM; a retry is not the same ask again --------------
#
# Live 2026-10-06 (website_review E2E, https://example.com): ``{"lines": []}`` on
# EVERY attempt. Every length rule in the first-draw prompt was an upper bound
# ("HARD CEILING … write fewer … never more", "at most ONE after-act line per
# section"), so an empty list broke none of them; and the retries POSTed the
# byte-identical prompt, so they asked the same question and got the same answer.

_CEILING_RE = re.compile(r"Write (\d+) words in\s+total")
_FLOOR_RE = re.compile(r"HARD MINIMUM: .*?at least (\d+) words in total", re.DOTALL)
_RETRY_MARKER = "RETRY — your previous answer"


def _sections_ctx(sections: int, *, duration: int = 50) -> RunContext:
    ctx = _ctx()
    ctx.config_snapshot["brief"]["duration_seconds"] = duration
    ctx.prior_run_outputs["website_redesign"]["visuals"] = [
        {
            "id": f"redesign-section-s{i}",
            "role": f"redesign_section:s{i}",
            "image_url": f"u/{i:02d}_s{i}.png",
        }
        for i in range(sections)
    ]
    return ctx


def test_the_first_draw_prompt_states_a_minimum_not_only_ceilings() -> None:
    stage = WebsiteReviewScriptStage(client=_WireWM(empty_calls=0).client())

    prompt = stage.build_prompt(_one_section_ctx())

    assert "HARD MINIMUM" in prompt
    assert "at least ONE line" in prompt
    assert 'an empty "lines" list is never a valid answer' in prompt
    floor = _FLOOR_RE.search(prompt)
    assert floor is not None, "the first draw must state a minimum WORD count"
    # The helper the short-draw re-ask already floors with — one number, one rule.
    assert int(floor.group(1)) == website_review_script.short_draw_floor_words(50, stage._wpm)
    # One vocabulary for it: the re-ask's own OVERRIDES wording.
    assert "This minimum is binding" in prompt
    # It is a first draw: nothing claims a previous answer existed.
    assert _RETRY_MARKER not in prompt
    assert "REWRITE" not in prompt


@pytest.mark.parametrize("wpm", [94.0])
@pytest.mark.parametrize("sections", [0])
def test_the_stated_minimum_never_contradicts_the_stated_ceiling(sections: int, wpm: float) -> None:
    """Read both numbers back OUT OF THE PROMPT TEXT: floor < ceiling, always."""
    stage = WebsiteReviewScriptStage(client=_WireWM(empty_calls=0).client())
    stage._wpm = wpm
    with_words = 0
    for duration in [*range(8, 61), *range(65, 301, 5), 450, 600]:
        prompt = stage.build_prompt(_sections_ctx(sections, duration=duration))
        where = f"duration={duration} sections={sections} wpm={wpm}"
        assert prompt.count("HARD MINIMUM") == 1, where
        assert "at least ONE line" in prompt, where
        ceiling = _CEILING_RE.search(prompt)
        assert ceiling is not None, where
        floor = _FLOOR_RE.search(prompt)
        if floor is None:
            # Too small an order to hold a word floor strictly under its ceiling:
            # the line minimum stands alone rather than a floor that contradicts.
            assert int(ceiling.group(1)) <= 2, where
            continue
        with_words += 1
        assert 1 <= int(floor.group(1)) < int(ceiling.group(1)), where
    assert with_words > 50, "the word floor must actually be stated across the range"


def test_an_authored_before_act_is_not_double_counted_into_the_minimum() -> None:
    """The model writes the AFTER act alone, so its floor is net of the authored words."""
    stage = WebsiteReviewScriptStage(client=_WireWM(empty_calls=0).client())
    ctx = _one_section_ctx()
    whole = _FLOOR_RE.search(stage.build_prompt(ctx))
    ctx.config_snapshot["brief"]["shotlist"] = [
        {"narration": "The hero buries the offer.", "seconds": 4.0, "visual": "screen"},
    ]
    authored = _FLOOR_RE.search(stage.build_prompt(ctx))

    assert whole is not None and authored is not None
    assert int(authored.group(1)) == int(whole.group(1)) - 5


async def test_a_retry_after_an_empty_draw_is_a_different_corrective_ask() -> None:
    wire = _WireWM(empty_calls=10**6)
    stage = WebsiteReviewScriptStage(client=wire.client())

    with pytest.raises(StageRetriesExhaustedError, match="empty script draw"):
        await stage.run({}, ctx=_one_section_ctx())

    first, *retries = [user_text(body) for body in wire.bodies]
    assert retries, "this test needs a retry budget to have anything to prove"
    assert _RETRY_MARKER not in first
    floor = _FLOOR_RE.search(first)
    assert floor is not None
    for retry in retries:
        # origin/main: ``retry == first`` — the same question, asked again.
        assert retry != first
        assert retry.startswith(first), "the corrective block is APPENDED to the same ask"
        block = retry[len(first) :]
        assert _RETRY_MARKER in block
        assert "0 `lines`" in block and "at least 1" in block
        assert "HARD MINIMUM" in block
        assert f"at least {floor.group(1)} words in total" in block
    # Still one model call per attempt.
    assert len(wire.bodies) == llm_retries() + 1


async def test_an_empty_draw_logs_the_prompt_and_the_completion_for_diagnosis(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The prompt was persisted NOWHERE, so the live failure could not be replayed."""
    import hashlib

    wire = _WireWM(empty_calls=1)
    stage = WebsiteReviewScriptStage(client=wire.client())

    with caplog.at_level("WARNING", logger="src.services.stages.primitives.script_stage"):
        await stage.run({}, ctx=_one_section_ctx())

    records = [
        r
        for r in caplog.records
        if r.name == "src.services.stages.primitives.script_stage"
        and "under-floor script draw" in r.getMessage()
    ]
    assert len(records) == 1, "exactly the one empty attempt is logged"
    assert records[0].levelname == "WARNING"
    message = records[0].getMessage()
    assert message.startswith("website_review_script: under-floor script draw ")
    payload = json.loads(message.split("under-floor script draw ", 1)[1])
    # What the model received, rebuilt from the log line alone: the prompt is the
    # user turn and the system prompt closes the system turn (after the worker seed).
    assert user_text(wire.bodies[0]) == payload["prompt"]
    assert system_text(wire.bodies[0]).endswith(payload["system_prompt"])
    prompt = payload["prompt"]
    assert prompt and "Total spoken length" in prompt
    assert payload["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert payload["prompt_chars"] == len(prompt)
    assert payload["completion"] == '{"lines": []}'
    assert (payload["attempt"], payload["count"], payload["required"]) == (1, 0, 1)
    assert payload["worker"] == "website-review-script"
    # Never the credential the client authenticates with.
    assert "test-token" not in caplog.text
