"""Deterministic shotlist-driven promo path.

Covers the normalizer plus each stage's shotlist branch (script / storyboard /
visual_director / slides) and an end-to-end render-binding test proving footage
and slides join 1:1 to narration in authored order. All pure/mocked — no DB,
no live services.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.promo.render import PromoRenderStage
from src.services.stages.promo.script import PromoScriptStage
from src.services.stages.promo.slides import PromoSlidesStage
from src.services.stages.promo.storyboard import PromoStoryboardStage
from src.services.stages.promo.visual_director import PromoVisualDirectorStage
from src.services.stages.shared.shotlist import normalize_shotlist, shot_line_id
from src.services.stages.verbatim import seconds_for

# A mixed authored shotlist used across the stage tests: url, slide, url, url.
_SHOTLIST = [
    {
        "kind": "url",
        "url": "https://app.example.com/",
        "narration": "Here's the dashboard the moment you log in.",
        "directives": {"hold_ms": 4000, "viewport": "desktop"},
    },
    {
        "kind": "slide",
        "narration": "Three reasons teams switch.",
        "slide": {
            "template": "bullets",
            "title": "Why teams switch",
            "bullets": ["Fast", "Simple"],
        },
    },
    {
        "kind": "url",
        "url": "https://app.example.com/pricing",
        "narration": "Pricing is dead simple.",
        "directives": {"click": '.nav >> text="Pricing"', "hold_ms": 3000},
    },
    {
        "kind": "url",
        "url": "https://app.example.com/signup",
        "narration": "Sign up in seconds.",
        "directives": {},
    },
]


def _brief(shotlist: list | None = None) -> dict:
    b: dict = {"value_proposition": "v", "call_to_action": "Go"}
    if shotlist is not None:
        b["shotlist"] = shotlist
    return b


# ── normalizer ─────────────────────────────────────────────────────────


def test_normalize_absent_or_malformed_returns_empty() -> None:
    assert normalize_shotlist({}) == []
    assert normalize_shotlist({"shotlist": None}) == []
    assert normalize_shotlist({"shotlist": "nope"}) == []
    assert normalize_shotlist({"shotlist": []}) == []


def test_normalize_preserves_order_and_deterministic_ids() -> None:
    shots = normalize_shotlist(_brief(_SHOTLIST))
    assert [s.line_id for s in shots] == ["shot-000", "shot-001", "shot-002", "shot-003"]
    assert [s.kind for s in shots] == ["url", "slide", "url", "url"]
    assert all(s.line_id == shot_line_id(s.index) for s in shots)


def test_normalize_seconds_match_verbatim_helper() -> None:
    shots = normalize_shotlist(_brief(_SHOTLIST))
    assert shots[0].seconds == seconds_for("Here's the dashboard the moment you log in.")


def test_normalize_drops_blank_narration_and_non_dict_and_reindexes() -> None:
    raw = [
        "not-a-dict",
        {"kind": "url", "url": "https://x", "narration": "   "},  # blank narration → dropped
        {"kind": "url", "url": "https://y", "narration": "kept one"},
        {"kind": "slide", "narration": "kept two", "slide": {}},
    ]
    shots = normalize_shotlist(_brief(raw))
    # Two survivors, re-indexed with no gaps.
    assert [s.line_id for s in shots] == ["shot-000", "shot-001"]
    assert [s.narration for s in shots] == ["kept one", "kept two"]


def test_normalize_directive_defaults_and_cleaning() -> None:
    shots = normalize_shotlist(
        _brief(
            [
                {
                    "kind": "url",
                    "url": "https://x",
                    "narration": "n",
                    "directives": {
                        "viewport": "bogus",
                        "scroll_to": "  ",
                        "click": "#go",
                    },
                }
            ]
        )
    )
    d = shots[0].directives
    assert d["hold_ms"] == 2500  # default
    assert d["viewport"] == "desktop"  # invalid → default
    assert d["click"] == "#go"
    assert "scroll_to" not in d  # blank dropped


# blanks dropped


# ── promo_script: deterministic verbatim lines ──────────────────────────


def test_script_shotlist_emits_deterministic_lines_in_order() -> None:
    stage = PromoScriptStage()
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": _brief(_SHOTLIST), "verbatim_items": []},
    )
    out = stage.short_circuit(ctx)
    assert out is not None
    assert [ln.id for ln in out.lines] == ["shot-000", "shot-001", "shot-002", "shot-003"]
    assert out.lines[1].text == "Three reasons teams switch."


def test_script_shotlist_wins_over_source_script() -> None:
    stage = PromoScriptStage()
    brief = _brief(_SHOTLIST)
    brief["source_script"] = "ignored line one\nignored line two"
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": brief, "verbatim_items": []},
    )
    out = stage.short_circuit(ctx)
    assert out is not None
    assert [ln.id for ln in out.lines] == ["shot-000", "shot-001", "shot-002", "shot-003"]


def test_script_shotlist_verbatim_gate_fires() -> None:
    from src.services.stages.base import StageError

    stage = PromoScriptStage()
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": _brief(_SHOTLIST),
            "verbatim_items": [{"text": "a phrase never spoken anywhere"}],
        },
    )
    with pytest.raises(StageError):
        stage.short_circuit(ctx)


# ── promo_storyboard: explicit per-shot visual_type ─────────────────────


@pytest.mark.asyncio
async def test_storyboard_shotlist_honors_per_shot_kind_with_mid_slide() -> None:
    stage = PromoStoryboardStage()
    lines = [
        {"id": shot_line_id(i), "text": s["narration"], "seconds": 2.0}
        for i, s in enumerate(_SHOTLIST)
    ]
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": _brief(_SHOTLIST)},
        prior_run_outputs={"promo_script": {"lines": lines, "total_seconds": 8.0}},
    )
    out = await stage.run({}, ctx=ctx)
    # url→footage, slide→slide; the slide sits in the INTERIOR (index 1), which the
    # legacy plan_visual_types would have forced to footage in demo_sync mode.
    assert [s.visual_type for s in out.shots] == ["footage", "slide", "footage", "footage"]


# ── promo_visual_director: deterministic identity, no LLM ────────────────


@pytest.mark.asyncio
async def test_visual_director_shotlist_is_identity_and_skips_llm() -> None:
    wm = AsyncMock()
    stage = PromoVisualDirectorStage(client=wm)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": _brief(_SHOTLIST)},
    )
    out = await stage.run({}, ctx=ctx)
    # One assignment per URL shot, screen_id == line_id (the join the site_capture
    # branch guarantees by using line_id as capture_id).
    assert {(a.line_id, a.screen_id) for a in out.assignments} == {
        ("shot-000", "shot-000"),
        ("shot-002", "shot-002"),
        ("shot-003", "shot-003"),
    }
    wm.message.assert_not_awaited()


# ── promo_slides: explicit slide content, url fallback ───────────────────


@pytest.mark.asyncio
async def test_slides_shotlist_uses_explicit_content_for_slide_shots() -> None:
    client = AsyncMock()
    client.resolve_template_id = AsyncMock(side_effect=lambda name: f"tid-{name}")
    client.start_deck = AsyncMock(return_value="deck-1")
    client.wait_for_deck = AsyncMock(return_value=None)
    client.get_slide_png = AsyncMock(return_value=b"png")
    stage = PromoSlidesStage(client=client)
    lines = [
        {"id": shot_line_id(i), "text": s["narration"], "seconds": 2.0}
        for i, s in enumerate(_SHOTLIST)
    ]
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": _brief(_SHOTLIST)},
        prior_run_outputs={"promo_script": {"lines": lines, "total_seconds": 8.0}},
    )
    with patch(
        "src.services.stages.shared.slide_deck.store_slide_image",
        new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['image_id']}.png"),
    ):
        out = await stage.run({}, ctx=ctx)

    # Every line gets a slide (universal fallback contract).
    assert {s.line_id for s in out.slides} == {"shot-000", "shot-001", "shot-002", "shot-003"}
    deck_call = client.start_deck.await_args
    assert deck_call is not None
    deck = deck_call.args[0]
    entries = deck["slides"]
    # Slide-shot (index 1) rendered on the bullets→feature_spotlight template with
    # its explicit content; url shots fall back to the title card.
    assert entries[1]["template_id"] == "tid-feature_spotlight"
    assert entries[1]["content"]["title"] == "Why teams switch"
    assert entries[1]["content"]["bullets"] == ["Fast", "Simple"]
    assert entries[0]["template_id"] == "tid-title"


# ── render: end-to-end 1:1 binding in authored order ────────────────────


def test_render_shotlist_binds_footage_and_slide_in_order() -> None:
    stage = PromoRenderStage()
    shots = normalize_shotlist(_brief(_SHOTLIST))
    lines = [{"id": s.line_id, "text": s.narration, "seconds": s.seconds} for s in shots]
    storyboard_shots = [
        {
            "id": str(uuid4()),
            "line_id": s.line_id,
            "description": s.narration,
            "seconds": s.seconds,
            "visual_style": "x",
            "visual_type": "footage" if s.kind == "url" else "slide",
        }
        for s in shots
    ]
    # site_capture produced a still per URL shot (id == line_id); slides produced a
    # slide per line (the universal fallback + the real slide for the slide shot);
    # the visual director mapped each URL line to its own screen (identity).
    screens = [
        {"id": s.line_id, "label": s.narration, "image_url": f"https://minio/{s.line_id}.png"}
        for s in shots
        if s.kind == "url"
    ]
    director = [
        {"id": str(uuid4()), "line_id": s.line_id, "screen_id": s.line_id}
        for s in shots
        if s.kind == "url"
    ]
    slides = [
        {
            "id": str(uuid4()),
            "line_id": s.line_id,
            "image_url": f"https://minio/slide-{s.line_id}.png",
        }
        for s in shots
    ]
    narrate = [
        {
            "line_id": s.line_id,
            "artifact_url": f"https://minio/audio-{s.line_id}.mp3",
            "seconds": s.seconds,
        }
        for s in shots
    ]
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": _brief(_SHOTLIST), "deliverable": {}},
        prior_run_outputs={
            "promo_storyboard": {"shots": storyboard_shots},
            "promo_script": {"lines": lines, "total_seconds": sum(s.seconds for s in shots)},
            "promo_narrate": {"clips": narrate},
            "site_capture": {"clips": [], "page_text": "", "notes": [], "screens": screens},
            "promo_visual_director": {"assignments": director, "notes": []},
            "promo_slides": {"slides": slides, "notes": []},
        },
    )

    built = stage.build_shots(ctx)
    # Authored order preserved, one shot per line.
    assert [b["line_id"] for b in built] == ["shot-000", "shot-001", "shot-002", "shot-003"]
    # URL shots show their captured still; the slide shot shows its slide image.
    assert built[0]["image_url"] == "https://minio/shot-000.png"
    assert built[2]["image_url"] == "https://minio/shot-002.png"
    assert built[3]["image_url"] == "https://minio/shot-003.png"
    assert built[1]["image_url"] == "https://minio/slide-shot-001.png"
    # Narration audio joins 1:1.
    assert all(b.get("audio_url") for b in built)
