"""website_review wave 2 — the defects prod run 01a0a054 (cliffordtraining.nl) shipped.

Each test is written against the run's own evidence and fails on origin/main 0a896d6.
"""

from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.generic.site_capture import SiteCaptureStage
from src.services.stages.primitives.dialogue import assign_dialogue_speakers, voice_for_speaker
from src.services.stages.website_review.grounding import (
    RedesignInventory,
    ScriptLineView,
    find_script_violations,
)
from src.services.stages.website_review.script import ReviewLine, WebsiteReviewScriptStage
from src.services.stages.website_review.slides import WebsiteReviewSlidesStage
from src.services.stages.website_review.visual_director import WebsiteReviewVisualDirectorStage

UNGROUNDED = "script_claims_ungrounded"


@pytest.fixture(autouse=True)
def _no_duration_reconciliation(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests are about WHAT the script says, not how long it runs.

    The length loop has its own suite; left on, its re-ask would spend a WM call
    and make every call count below about the order's arithmetic instead.
    """

    async def _as_drawn(self: Any, first: Any, payload: Any, ctx: Any) -> Any:
        return first

    monkeypatch.setattr(WebsiteReviewScriptStage, "_reconcile", _as_drawn)


IMBALANCE = "script_act_imbalance"

# The prod redesign, reduced: nine rendered sections, NO <form>, NO search input.
_REDESIGN_HTML = """
<html><body>
<section data-section="hero"><section><h1>Leiders die echt bewegen.</h1>
  <a class="btn btn-primary" href="/contact">Plan een gesprek</a></section></section>
<section data-section="diensten"><section><h2>Zes routes, een bestemming.</h2></section></section>
<section data-section="testimonials"><section><h2>75 LinkedIn-aanbevelingen.</h2></section></section>
<section data-section="cta"><section><h2>Klaar voor jouw volgende stap?</h2>
  <a class="cta-button" href="/contact">Plan een gesprek</a></section></section>
<section data-section="footer"><section><h2>Coaching voor leiders</h2></section></section>
</body></html>
"""
_SECTIONS = ["hero", "diensten", "testimonials", "cta", "footer"]


def _redesign_output() -> dict[str, Any]:
    # Deliberately NOT in page order (prod listed cta first) — file names carry it.
    order = {name: i for i, name in enumerate(_SECTIONS)}
    visuals = [
        {
            "id": f"redesign-section-{name}",
            "label": f"Redesign — {name.title()}",
            "role": f"redesign_section:{name}",
            "image_url": f"https://minio.test/sections/{order[name]:02d}_{name}.png",
        }
        for name in reversed(_SECTIONS)
    ]
    visuals.append(
        {
            "id": "redesign-desktop",
            "label": "Redesign Desktop",
            "role": "redesign_desktop",
            "image_url": "https://minio.test/desktop.png",
        }
    )
    return {"style": "modern", "redesign_job_id": "rd-1", "visuals": visuals}


def _findings() -> list[dict[str, Any]]:
    lows = [
        {
            "id": f"L{i}",
            "category": "seo",
            "severity": "low",
            "title": f"Low seo nit {i}",
            "recommendation": "tidy up",
        }
        for i in range(20)
    ]
    critical = {
        "id": "C1",
        "category": "ux",
        "severity": "critical",
        "title": "No contact form on homepage",
        "recommendation": "Add a contact form.",
    }
    return [*lows, critical]


def _ctx(brief: dict | None = None, voice: dict | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "type": "website_review",
            "brief": {"duration_seconds": 60, **(brief or {})},
            "voice": voice if voice is not None else {"voice_id": "af_heart"},
        },
        prior_run_outputs={
            "website_critique": {
                "url": "https://www.cliffordtraining.nl/",
                "overall_score": 65.2,
                "summary": "Security headers missing; no form on the homepage.",
                "findings": _findings(),
            },
            "website_redesign": _redesign_output(),
        },
    )


def _resp(lines: list[dict[str, Any]]) -> WMResponse:
    return WMResponse(
        output={"lines": lines},
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _design(html: str = _REDESIGN_HTML) -> AsyncMock:
    design = AsyncMock()
    design.get_design_html = AsyncMock(return_value=html)
    return design


def _line(text: str, phase: str, section: str | None = None, speaker: str = "a") -> dict:
    return {
        "text": text,
        "seconds": 3.0,
        "phase": phase,
        "speaker": speaker,
        "section_id": f"redesign-section-{section}" if section else None,
    }


# The prod draft, verbatim.
_PROD_DRAFT = [
    _line(
        "Clifford Training's homepage misses the mark. No security headers, and that "
        "render-blocking script stalls the load.",
        "before",
    ),
    _line(
        "Right. We added the missing headers and deferred that script so the page renders "
        "instantly.",
        "before",
    ),
    _line("And the contact button just sends users away. Where's the form?", "before"),
    _line("Now it's embedded right on the homepage. No extra click, no drop-off.", "before"),
    _line("Okay, show me the new redesign.", "before"),
    _line(
        "Here is the new redesign. Security headers are live, scripts are deferred, and the "
        "form sits front and center.",
        "after",
    ),
]

# A grounded draft: problems before, one line per rendered section after.
_GROUNDED_DRAFT = [
    _line("The current homepage buries its one call to action.", "before"),
    _line("And there is no way to get in touch without leaving the page.", "before"),
    _line("Here is the new redesign, opening on a calm hero.", "after", "hero"),
    _line("Six services now read as clear routes.", "after", "diensten"),
    _line("Seventy-five recommendations carry the proof.", "after", "testimonials"),
    _line("And a single closing call to action.", "after", "cta"),
]


# ---------------------------------------------------------------------------
# 1. grounding
# ---------------------------------------------------------------------------


async def test_prompt_carries_the_rendered_section_inventory_and_what_is_absent() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(_GROUNDED_DRAFT)
    design = _design()
    await WebsiteReviewScriptStage(client=fake, design=design).run({}, ctx=_ctx())

    design.get_design_html.assert_awaited_once_with("rd-1")
    prompt = fake.message.call_args_list[0].kwargs["message"]
    # sections in PAGE order with their real headings, by the id the director maps to
    hero = prompt.index("redesign-section-hero")
    footer = prompt.index("redesign-section-footer")
    assert hero < footer
    assert "Klaar voor jouw volgende stap?" in prompt
    assert "does NOT contain" in prompt
    assert "a form (contact/sign-up)" in prompt.split("does NOT contain", 1)[1].split("\n")[0]
    assert "describe ONLY what is listed" in prompt


async def test_an_ungrounded_draft_is_re_asked_once_and_the_grounded_rewrite_ships() -> None:
    prompts: list[str] = []

    async def _answer(*_args: object, **kwargs: object) -> object:
        # The model answers the prompt it is given: the grounding correction gets
        # the grounded draft, anything else the prod draft.
        prompt = str(kwargs["message"])
        prompts.append(prompt)
        return _resp(_GROUNDED_DRAFT if "REWRITE" in prompt else _PROD_DRAFT)

    fake = AsyncMock()
    fake.message.side_effect = _answer
    out = await WebsiteReviewScriptStage(client=fake, design=_design()).run({}, ctx=_ctx())

    reasks = [p for p in prompts if "REWRITE" in p]
    assert len(reasks) == 1  # exactly ONE grounding correction
    reask = reasks[0]
    assert "security headers" in reask
    assert "a form (contact/sign-up)" in reask
    assert "describes a fix over the ORIGINAL page" in reask
    assert [line.text for line in out.lines] == [d["text"] for d in _GROUNDED_DRAFT]
    assert UNGROUNDED not in {d["code"] for d in out.degradations}


async def test_a_draft_still_ungrounded_after_the_re_ask_records_a_degradation() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(_PROD_DRAFT)
    out = await WebsiteReviewScriptStage(client=fake, design=_design()).run({}, ctx=_ctx())

    assert fake.message.await_count == 2  # the draw + exactly ONE grounding re-ask
    codes = {d["code"]: d["detail"] for d in out.degradations}
    assert UNGROUNDED in codes
    assert "security headers" in codes[UNGROUNDED]
    assert "a form" in codes[UNGROUNDED]
    assert IMBALANCE in {a["code"] for a in out.advisories}  # 1 after line of 6


async def test_security_headers_are_never_grounded_even_without_the_html() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        [
            _line("The page ships without protection.", "before"),
            _line("Here is the redesign: security headers are live.", "after", "hero"),
        ]
    )
    broken = AsyncMock()
    broken.get_design_html = AsyncMock(side_effect=RuntimeError("html_render down"))
    out = await WebsiteReviewScriptStage(client=fake, design=broken).run({}, ctx=_ctx())
    assert UNGROUNDED in {d["code"] for d in out.degradations}


@pytest.mark.parametrize(
    ("text", "feature"),
    [("We added security headers.", "security_headers")],
)
def test_server_side_phrases_are_always_ungrounded(text: str, feature: str) -> None:
    lines = [ScriptLineView(text=text, phase="after")]
    # features=None: even with NO inventory, a server-side claim is never groundable.
    found = find_script_violations(lines, RedesignInventory())
    assert (0, feature) in found.ungrounded_claims, text


async def test_a_grounded_draft_costs_one_call_and_records_nothing() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(_GROUNDED_DRAFT)
    out = await WebsiteReviewScriptStage(client=fake, design=_design()).run({}, ctx=_ctx())
    assert fake.message.await_count == 1
    assert UNGROUNDED not in {d["code"] for d in out.degradations}
    assert IMBALANCE not in {a["code"] for a in out.advisories}
    assert [line.section_id for line in out.lines if line.phase == "after"] == [
        "redesign-section-hero",
        "redesign-section-diensten",
        "redesign-section-testimonials",
        "redesign-section-cta",
    ]


# ---------------------------------------------------------------------------
# 2 + 3. dialogue
# ---------------------------------------------------------------------------


async def test_dialogue_is_off_unless_explicitly_enabled_prompt_and_speakers_agree() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        [
            _line("Why is this so slow?", "before", speaker="a"),
            _line("Huge images.", "before", speaker="b"),
            *_GROUNDED_DRAFT[2:],
        ]
    )
    ctx = _ctx()  # no brief.dialogue at all — the prod shape
    out = await WebsiteReviewScriptStage(client=fake, design=_design()).run({}, ctx=ctx)

    prompt = fake.message.call_args_list[0].kwargs["message"]
    assert "SINGLE narrator" in prompt
    assert "TWO-PERSON" not in prompt
    assert {line.speaker for line in out.lines} == {"a"}
    # ...and record reads every line in the single voice, matching the prompt
    voices = {
        voice_for_speaker({"speaker": line.speaker}, {"voice_id": "af_heart"}, {})
        for line in out.lines
    }
    assert voices == {"af_heart"}


def test_dialogue_enabled_but_one_voice_configured_is_a_single_narrator() -> None:
    ctx = _ctx(
        brief={"dialogue": {"enabled": True}},
        voice={"voice_id": "af_heart", "cast": {"a": "af_heart", "b": "af_heart"}},
    )
    prompt = WebsiteReviewScriptStage(client=AsyncMock()).build_prompt(ctx)
    assert "SINGLE narrator" in prompt


def test_dialogue_enabled_with_two_voices_is_a_two_hander() -> None:
    ctx = _ctx(brief={"dialogue": {"enabled": True}})
    prompt = WebsiteReviewScriptStage(client=AsyncMock()).build_prompt(ctx)
    assert "TWO-PERSON" in prompt


def test_explicit_valid_speaker_tags_are_kept_not_re_derived_from_question_marks() -> None:
    lines = [
        ReviewLine(id="1", text="This homepage misses the mark.", seconds=3, speaker="a"),
        ReviewLine(id="2", text="Where would you start?", seconds=3, speaker="b"),
        ReviewLine(id="3", text="With the hero.", seconds=3, speaker="a"),
    ]
    assert [line.speaker for line in assign_dialogue_speakers(lines)] == ["a", "b", "a"]


# ---------------------------------------------------------------------------
# 4. findings by severity
# ---------------------------------------------------------------------------


def test_the_critical_finding_reaches_the_prompt_even_past_index_twelve() -> None:
    prompt = WebsiteReviewScriptStage(client=AsyncMock()).build_prompt(_ctx())
    findings = prompt.split("Key findings", 1)[1].split("\n", 1)[0]
    assert "No contact form on homepage" in findings
    assert findings.index("'critical'") < findings.index("'low'")


# ---------------------------------------------------------------------------
# 5. before/after balance
# ---------------------------------------------------------------------------


def test_prompt_confines_the_before_act_to_problems_and_sizes_the_after_act() -> None:
    prompt = WebsiteReviewScriptStage(client=AsyncMock()).build_prompt(_ctx())
    assert "PROBLEMS ONLY" in prompt
    assert "at least HALF of all lines are after-act lines" in prompt
    # 60s at the 150 wpm class default is 150 words: five sections are one short of
    # carrying the after act's half at a comfortable line, so the allowance is 6.
    from src.services.stages.website_review.script import after_line_allowance

    allowance = after_line_allowance(len(_SECTIONS), 150)
    assert allowance == len(_SECTIONS) + 1
    assert f"at most {allowance} after-act lines" in prompt


async def test_visual_director_honours_the_scripts_section_ids_without_an_llm_call() -> None:
    script = {
        "lines": [
            {"id": "b1", "text": "old", "phase": "before", "speaker": "a"},
            {
                "id": "a1",
                "text": "Here is the redesign.",
                "phase": "after",
                "speaker": "a",
                "section_id": "redesign-section-hero",
            },
            {
                "id": "a2",
                "text": "The closing call?",
                "phase": "after",
                "speaker": "a",
                "section_id": "redesign-section-cta",
            },
        ]
    }
    ctx = MagicMock()
    ctx.config_snapshot = {"brief": {}}
    ctx.prior_output.side_effect = lambda key: {
        "website_review_script": script,
        "website_redesign": _redesign_output(),
    }.get(key)
    wm = MagicMock()
    wm.message = AsyncMock(side_effect=AssertionError("the LLM must not be asked"))
    with patch(
        "src.services.stages.website_review.visual_director.require_video", lambda _ctx: None
    ):
        out = await WebsiteReviewVisualDirectorStage(client=wm).handle(
            cast("Any", SimpleNamespace()),
            ctx,
        )
    assert {a.line_id: a.section_id for a in out.assignments} == {
        "a1": "redesign-section-hero",
        "a2": "redesign-section-cta",
    }


# ---------------------------------------------------------------------------
# 6. tone
# ---------------------------------------------------------------------------


def test_voice_direction_tone_reaches_the_prompt() -> None:
    ctx = _ctx(voice={"voice_id": "af_heart", "direction": {"tone": "confident"}})
    prompt = WebsiteReviewScriptStage(client=AsyncMock()).build_prompt(ctx)
    assert "confident" in prompt
    assert "Voice/tone: neutral" not in prompt


# ---------------------------------------------------------------------------
# 7. slides stub is advisory
# ---------------------------------------------------------------------------


async def test_a_plain_run_without_slide_shots_is_not_degraded_by_the_slides_stage() -> None:
    from src.services.stages.shared.degradation import is_degraded

    ctx = MagicMock()
    ctx.config_snapshot = {"brief": {"generate_video": True}, "deliverable": {}}
    with patch("src.services.stages.website_review.slides.require_video", lambda _ctx: None):
        out = await WebsiteReviewSlidesStage(client=MagicMock(), video_client=MagicMock()).handle(
            cast("Any", SimpleNamespace()),
            ctx,
        )
    assert out.slides_stubbed is False  # advisory tier, not the authored-lines flag
    assert out.degradations == []
    assert [a["code"] for a in out.advisories] == ["slides_not_authored"]
    assert is_degraded(out.model_dump()) is False


# ---------------------------------------------------------------------------
# 8. site_capture on a website_review
# ---------------------------------------------------------------------------


def _zip(ids: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for sid in ids:
            zf.writestr(f"{sid}.png", b"png")
    return buf.getvalue()


async def test_site_capture_captures_the_review_target_url() -> None:
    client = AsyncMock()
    client.capture_screens = AsyncMock(return_value="screens-job-1")
    client.wait_for_content_recording = AsyncMock(
        return_value={"status": "completed", "duration_seconds": 0.0}
    )
    client.get_recording_video = AsyncMock(
        return_value=(_zip(["live-top", "live-bottom"]), "application/zip", None)
    )
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "type": "website_review",
            "brief": {"target_url": "https://site.test/"},
        },
    )
    with patch(
        "src.services.stages.generic.site_capture.store_capture_image",
        new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['image_id']}.png"),
    ):
        out = await SiteCaptureStage(client=client).run({}, ctx=ctx)

    call = client.capture_screens.await_args
    assert call is not None
    assert call.kwargs["url"] == "https://site.test/"
    assert {s.id for s in out.screens} == {"live-top", "live-bottom"}


async def test_site_capture_still_skips_a_non_review_brief_without_source_url() -> None:
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"type": "promo", "brief": {"target_url": "https://site.test/"}},
    )
    with pytest.raises(StageSkippedError):
        await SiteCaptureStage(client=AsyncMock()).run({}, ctx=ctx)
