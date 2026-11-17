"""website_review_slide_plan — the deck a website_review actually shows.

Covers the three things that make this stage worth having: it authors a card per
BEFORE-act line from the model, it NEVER ships a hollow deck silently (a malformed
or failed reply falls back to a deterministic critique projection and SAYS SO), and
the deck it produces is what quiets ``website_review_slides``' ``slides_not_authored``
advisory — which fired on every single run before this stage existed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.shared.degradation import SLIDE_PLAN_LLM_FALLBACK
from src.services.stages.website_review.slide_plan import (
    WebsiteReviewSlidePlanInput,
    WebsiteReviewSlidePlanStage,
)
from src.services.stages.website_review.slides import WebsiteReviewSlidesStage

_FINDINGS = [
    {
        "id": "f1",
        "category": "Navigation",
        "severity": "high",
        "title": "The primary nav hides the pricing page",
        "description": "Pricing sits three clicks deep behind a hamburger menu.",
        "recommendation": "Promote Pricing to a top-level nav item.",
    },
    {
        "id": "f2",
        "category": "Performance",
        "severity": "medium",
        "title": "The hero image is a 4 MB PNG",
        "description": "First contentful paint lands at 4.1s on a 4G connection.",
        "recommendation": "Ship a 120 KB WebP at two densities.",
    },
]

_LINES = [
    {"id": "l1", "text": "Let's start with how you find anything here.", "phase": "before"},
    {"id": "l2", "text": "Pricing is buried.", "phase": "before", "critique_item_id": "f1"},
    {
        "id": "l3",
        "text": "And the page takes forever.",
        "phase": "before",
        "critique_item_id": "f2",
    },
    {"id": "l4", "text": "Here is the redesign.", "phase": "after"},
]


def _ctx(
    *,
    lines: list[dict[str, Any]] | None = None,
    findings: list[dict[str, Any]] | None = None,
    brief: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> RunContext:
    prior: dict[str, dict[str, Any]] = {
        "website_review_script": {"lines": _LINES if lines is None else lines},
        "website_critique": {"findings": _FINDINGS if findings is None else findings},
    }
    if plan is not None:
        prior["website_review_slide_plan"] = plan
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": brief or {}, "deliverable": {"title": "Acme"}},
        prior_run_outputs=prior,
    )


def _wm(output: Any = None, *, error: BaseException | None = None) -> MagicMock:
    fake = MagicMock()
    if error is not None:
        fake.message = AsyncMock(side_effect=error)
    else:
        fake.message = AsyncMock(return_value=SimpleNamespace(output=output))
    return fake


# --------------------------------------------------------------------------- #
# 1. The authoring stage
# --------------------------------------------------------------------------- #


async def test_authors_one_card_per_before_line_from_the_model() -> None:
    wm = _wm(
        {
            "slides": [
                # Deliberately out of script order — the stage must re-sort.
                {
                    "line_id": "l2",
                    "template": "bullets",
                    "title": "Pricing is three clicks deep",
                    "bullets": ["Promote it to the top nav"],
                },
                {"line_id": "l1", "template": "title", "title": "Finding your way around"},
                # An after-act line the stage was never given → dropped.
                {"line_id": "l4", "template": "title", "title": "The redesign"},
            ]
        }
    )
    out = await WebsiteReviewSlidePlanStage(client=wm).handle(WebsiteReviewSlidePlanInput(), _ctx())

    assert out.source == "llm"
    assert not out.advisories
    # Script order, before-act lines only, unknown/after lines dropped.
    assert [s.line_id for s in out.slides] == ["l1", "l2"]
    assert out.slides[1].slide == {
        "template": "bullets",
        "eyebrow": "Acme",
        "title": "Pricing is three clicks deep",
        "bullets": ["Promote it to the top nav"],
    }
    assert out.slides[1].critique_item_id == "f1"
    # The prompt carried the critique findings, so the card can be grounded.
    message = wm.message.await_args.kwargs["message"]
    assert "Promote Pricing to a top-level nav item." in message
    assert wm.message.await_args.kwargs["output_schema"]["required"] == ["slides"]


async def test_authored_shotlist_deck_is_left_alone() -> None:
    brief = {
        "shotlist": [
            {"kind": "slide", "narration": "Three problems.", "slide": {"title": "Problems"}}
        ]
    }
    wm = _wm({"slides": [{"line_id": "l1", "title": "x"}]})
    out = await WebsiteReviewSlidePlanStage(client=wm).handle(
        WebsiteReviewSlidePlanInput(), _ctx(brief=brief)
    )
    assert out.source == "authored_shotlist"
    assert out.slides == []
    wm.message.assert_not_awaited()  # the operator's deck is never second-guessed


async def test_no_before_act_lines_plans_nothing() -> None:
    wm = _wm({"slides": []})
    out = await WebsiteReviewSlidePlanStage(client=wm).handle(
        WebsiteReviewSlidePlanInput(), _ctx(lines=[{"id": "a1", "text": "after", "phase": "after"}])
    )
    assert out.source == "no_lines"
    assert out.slides == []
    wm.message.assert_not_awaited()


# --------------------------------------------------------------------------- #
# 2. The malformed-reply fallback — never a hollow deck, never a silent one
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "reply",
    [pytest.param(None, id="null-output")],
)
async def test_malformed_reply_falls_back_to_the_critique_projection(reply: Any) -> None:
    out = await WebsiteReviewSlidePlanStage(client=_wm(reply)).handle(
        WebsiteReviewSlidePlanInput(), _ctx()
    )

    assert out.source == "critique_fallback"
    # A REAL deck, not an empty one: a card per before-act line.
    assert [s.line_id for s in out.slides] == ["l1", "l2", "l3"]
    # Grounded card: finding.title → title, recommendation + description → bullets.
    assert out.slides[1].slide == {
        "template": "bullets",
        "eyebrow": "Navigation",
        "title": "The primary nav hides the pricing page",
        "bullets": [
            "Promote Pricing to a top-level nav item.",
            "Pricing sits three clicks deep behind a hamburger menu.",
        ],
    }
    # Ungrounded line (cites no finding) → its own narration as a title card.
    assert out.slides[0].slide == {
        "template": "title",
        "eyebrow": "Acme",
        "title": "Let's start with how you find anything here.",
    }
    # ...and the fallback is ANSWERABLE FROM THE RUN RECORD. html_render #132 was
    # exactly this path returning a derived result with no trace.
    assert [a["code"] for a in out.advisories] == [SLIDE_PLAN_LLM_FALLBACK]


async def test_llm_failure_falls_back_and_records_the_cause() -> None:
    out = await WebsiteReviewSlidePlanStage(client=_wm(error=RuntimeError("WM 402"))).handle(
        WebsiteReviewSlidePlanInput(), _ctx()
    )
    assert out.source == "critique_fallback"
    assert len(out.slides) == 3
    assert "WM 402" in out.advisories[0]["detail"]


async def test_fallback_without_findings_still_authors_a_card_per_line() -> None:
    out = await WebsiteReviewSlidePlanStage(client=_wm({})).handle(
        WebsiteReviewSlidePlanInput(), _ctx(findings=[])
    )
    assert [s.slide["title"] for s in out.slides] == [line["text"] for line in _LINES[:3]]
    assert all(s.slide["template"] == "title" for s in out.slides)


# --------------------------------------------------------------------------- #
# 3. The seam into website_review_slides — the advisory goes quiet
# --------------------------------------------------------------------------- #


class _FakeStorage:
    async def initialize(self) -> None:
        pass

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        return f"artifacts/{job_id}/{stage_name}/{filename}"

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        return f"https://minio.test/{path}"


@pytest.fixture
def fake_storage():
    tts_storage.set_storage_for_tests(_FakeStorage())
    yield
    tts_storage.set_storage_for_tests(None)


def _slide_client(count: int) -> AsyncMock:
    fake = AsyncMock()
    fake.resolve_template_id.side_effect = lambda name: {
        "title": "tpl-title",
        "cta": "tpl-cta",
        "feature_spotlight": "tpl-feature",
    }[name]
    fake.start_deck.return_value = "deck-1"
    fake.wait_for_deck.return_value = {"status": "completed", "slide_count": count}
    fake.get_slide_png.side_effect = [f"\x89PNG-{i}".encode() for i in range(count)]
    return fake


_PLAN = {
    "slides": [
        {
            "line_id": "l2",
            "narration": "Pricing is buried.",
            "slide": {
                "template": "bullets",
                "eyebrow": "Navigation",
                "title": "Pricing is three clicks deep",
                "bullets": ["Promote it to the top nav"],
            },
        },
        {
            "line_id": "l3",
            "narration": "And the page takes forever.",
            "slide": {"template": "title", "eyebrow": "Performance", "title": "4 MB hero image"},
        },
    ]
}


async def test_planned_deck_renders_and_quiets_the_advisory(fake_storage) -> None:
    client = _slide_client(2)
    out = await WebsiteReviewSlidesStage(client=client).handle(MagicMock(), _ctx(plan=_PLAN))

    # The deck is REAL and keyed by SCRIPT line id, which is what compile binds on.
    assert [s.line_id for s in out.slides] == ["l2", "l3"]
    assert out.slides_planned is True
    assert out.slides_stubbed is False
    # The advisory that used to fire on EVERY run is now silent.
    assert [a["code"] for a in out.advisories] == []
    payload = client.start_deck.await_args.args[0]
    assert payload["slides"][0]["template_id"] == "tpl-feature"  # bullets → feature_spotlight
    assert payload["slides"][0]["content"]["bullets"] == ["Promote it to the top nav"]


async def test_advisory_still_fires_when_neither_producer_supplied_a_deck(fake_storage) -> None:
    client = AsyncMock()
    out = await WebsiteReviewSlidesStage(client=client).handle(MagicMock(), _ctx())

    assert out.slides == []
    assert out.slides_stubbed is False  # advisory tier, not the authored-lines flag
    assert [a["code"] for a in out.advisories] == ["slides_not_authored"]
    client.start_deck.assert_not_called()


async def test_hollow_planned_cards_are_dropped_not_rendered(fake_storage) -> None:
    # A card whose every authored field is blank normalises to {"template": ...}
    # alone. Rendering it would put an empty slide on screen — the hollow deck.
    client = AsyncMock()
    hollow = {"slides": [{"line_id": "l2", "narration": "x", "slide": {"template": "title"}}]}
    out = await WebsiteReviewSlidesStage(client=client).handle(MagicMock(), _ctx(plan=hollow))

    assert out.slides == []
    # Hollow cards are dropped before render, so this is the same empty-deck branch:
    # recorded by the advisory, not by the authored-lines `slides_stubbed` flag.
    assert out.slides_stubbed is False
    assert [a["code"] for a in out.advisories] == ["slides_not_authored"]
    client.start_deck.assert_not_called()


async def test_authored_shotlist_outranks_the_plan(fake_storage) -> None:
    client = _slide_client(1)
    brief = {
        "shotlist": [
            {
                "kind": "slide",
                "narration": "Three problems jump out.",
                "slide": {"template": "title", "title": "Operator's own card"},
            }
        ]
    }
    out = await WebsiteReviewSlidesStage(client=client).handle(
        MagicMock(), _ctx(brief=brief, plan=_PLAN)
    )

    assert [s.line_id for s in out.slides] == ["shot-000"]  # positional shotlist key
    assert out.slides_planned is False
    payload = client.start_deck.await_args.args[0]
    assert payload["slides"][0]["content"]["title"] == "Operator's own card"
