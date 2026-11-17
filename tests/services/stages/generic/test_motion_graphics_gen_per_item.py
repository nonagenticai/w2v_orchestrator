"""motion_graphics_gen — one template per ITEM, the motion brief's items, one brand.

Run 01a11082 (2026-10-06): the request could name one ``template_id`` for the
whole run, so three different lines all became ``kinetic_typography`` cards.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.base import StagePreconditionError, StageSkippedError
from src.services.stages.generic.motion_graphics_gen import MotionGraphicsGenStage
from src.services.stages.shared.motion_slides import (
    deck_motion_params,
    run_motion_brand_params,
)
from src.services.stages.shared.slide_deck import _BRAND_DEFAULTS
from src.services.stages.shared.title_cards import build_title_card_shots
from src.services.visual_media import collect_motion_clips_by_line

_GEN = "src.services.stages.generic.motion_graphics_gen"
_SCRIPT: dict[str, object] = {
    "lines": [
        {"id": "l1", "text": "Users can set autonomy levels for 30+ capabilities."},
        {"id": "l2", "text": "Decisions are made by deterministic code, not language models."},
    ]
}
_BRIEF_ITEMS: list[dict[str, object]] = [
    {
        "id": "i1",
        "line_id": "l1",
        "template_id": "stat",
        "params": {"value": "30", "suffix": "+", "label": "Capabilities", "kicker": ""},
        "duration_ms": 8955,
        "authored_by": "fallback",
    },
    {
        "id": "i2",
        "line_id": "l2",
        "template_id": "two_up",
        "params": {"two_up": {"left": {"title": "Deterministic code"}, "right": {"title": "Not"}}},
        "duration_ms": 7982,
        "authored_by": "model",
    },
]


def _client() -> AsyncMock:
    client = AsyncMock()
    client.record_animation = AsyncMock(side_effect=lambda **kw: f"job-{kw['template_id']}")
    client.wait_for_content_recording = AsyncMock(
        return_value={"status": "completed", "duration_seconds": 4.0}
    )
    client.get_recording_video = AsyncMock(return_value=(b"\x00mp4", "video/mp4", None))
    return client


def _ctx(cfg: dict[str, object], prior: dict[str, dict[str, object]] | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(), deliverable_id=uuid4(), config_snapshot=cfg, prior_run_outputs=prior or {}
    )


@pytest.fixture(autouse=True)
def _storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{_GEN}.find_existing_media", AsyncMock(return_value=None))
    monkeypatch.setattr(
        f"{_GEN}.store_generated_video",
        AsyncMock(side_effect=lambda **kw: f"https://minio.test/v/{kw['video_id']}.mp4"),
    )


async def test_an_item_renders_its_own_template_and_the_block_template_is_the_default() -> None:
    client = _client()
    ctx = _ctx(
        {
            "motion_graphics": {
                "template_id": "section_divider",
                "duration_ms": 4000,
                "items": [
                    {"line_id": "a", "template_id": "stat", "params": {"value": "30"}},
                    {"line_id": "b", "params": {"title": "Control"}},
                    {"line_id": "c", "template_id": "timeline", "duration_ms": 9000},
                ],
            }
        }
    )

    out = await MotionGraphicsGenStage(client=client).run({}, ctx=ctx)

    sent = {
        call.kwargs["template_id"]: call.kwargs for call in client.record_animation.await_args_list
    }
    assert sorted(sent) == ["section_divider", "stat", "timeline"]
    assert {clip.line_id: clip.template_id for clip in out.clips} == {
        "a": "stat",
        "b": "section_divider",
        "c": "timeline",
    }
    assert sent["stat"]["params"] == {"value": "30"}
    assert sent["stat"]["duration_ms"] == 4000  # the block's default
    assert sent["timeline"]["duration_ms"] == 9000  # the item's own
    keyed, pool = collect_motion_clips_by_line(out.model_dump())
    assert sorted(keyed) == ["a", "b", "c"]
    assert pool == []


async def test_the_clip_id_depends_on_the_items_template() -> None:
    """Same params, same line, another template: another clip — never a reused one."""

    async def _clip_id(template_id: str) -> str:
        block = {"items": [{"line_id": "a", "template_id": template_id, "params": {"title": "X"}}]}
        ctx = RunContext(
            run_id=run_id,
            deliverable_id=uuid4(),
            config_snapshot={"motion_graphics": block},
            prior_run_outputs={},
        )
        out = await MotionGraphicsGenStage(client=_client()).run({}, ctx=ctx)
        return out.clips[0].id

    run_id = uuid4()
    divider, quote, again = (
        await _clip_id("section_divider"),
        await _clip_id("quote"),
        await _clip_id("section_divider"),
    )

    assert divider != quote
    assert divider == again


async def test_an_item_that_names_a_template_does_not_inherit_the_blocks_raw_html() -> None:
    client = _client()
    ctx = _ctx(
        {
            "motion_graphics": {
                "html": "<html>custom</html>",
                "items": [{"line_id": "a", "template_id": "stat"}, {"line_id": "b"}],
            }
        }
    )

    await MotionGraphicsGenStage(client=client).run({}, ctx=ctx)

    calls = {
        call.kwargs["part_id"]: call.kwargs for call in client.record_animation.await_args_list
    }
    by_template = {kw["template_id"]: kw["html"] for kw in calls.values()}
    assert by_template == {"stat": None, None: "<html>custom</html>"}


async def test_an_item_with_no_template_and_no_default_fails_loudly() -> None:
    ctx = _ctx(
        {"motion_graphics": {"items": [{"line_id": "a", "template_id": "stat"}, {"line_id": "b"}]}}
    )

    with pytest.raises(StagePreconditionError, match="item for line 'b' names no template_id"):
        await MotionGraphicsGenStage(client=_client()).run({}, ctx=ctx)


async def test_the_motion_briefs_items_are_rendered_without_a_motion_graphics_block() -> None:
    client = _client()
    ctx = _ctx(
        {
            "brand": {"name": "Merqi"},
            "theme": {},
            "deliverable": {"format": "16x9"},
            "resolution": "4k",
            # The brief only ever runs for a per-line motion request.
            "direction": {"motion": "per_line"},
        },
        prior={"promo_script": _SCRIPT, "promo_motion_brief": {"items": _BRIEF_ITEMS}},
    )

    out = await MotionGraphicsGenStage(client=client).run({}, ctx=ctx)

    assert [(clip.line_id, clip.template_id) for clip in out.clips] == [
        ("l1", "stat"),
        ("l2", "two_up"),
    ]
    sent = {
        call.kwargs["template_id"]: call.kwargs for call in client.record_animation.await_args_list
    }
    assert sent["stat"]["duration_ms"] == 8955
    assert sent["two_up"]["duration_ms"] == 7982
    assert (sent["stat"]["width"], sent["stat"]["height"]) == (3840, 2160)
    # Content from the brief, over the run's ONE brand resolution.
    brand = run_motion_brand_params(ctx.config_snapshot)
    assert sent["stat"]["params"] == {
        **brand,
        "value": "30",
        "suffix": "+",
        "label": "Capabilities",
        "kicker": "",
    }
    assert out.advisories == []


async def test_items_the_request_authored_win_over_the_motion_briefs() -> None:
    client = _client()
    ctx = _ctx(
        {
            "motion_graphics": {
                "template_id": "quote",
                "items": [{"line_id": "l1", "params": {"quote": "30+ capabilities"}}],
            }
        },
        prior={"promo_script": _SCRIPT, "promo_motion_brief": {"items": _BRIEF_ITEMS}},
    )

    out = await MotionGraphicsGenStage(client=client).run({}, ctx=ctx)

    assert [(clip.line_id, clip.template_id) for clip in out.clips] == [("l1", "quote")]
    assert client.record_animation.await_count == 1


@pytest.mark.parametrize("brief", [None])
async def test_without_a_block_and_without_brief_items_the_stage_still_skips(
    brief: dict[str, object] | None,
) -> None:
    prior: dict[str, dict[str, object]] = {"promo_script": _SCRIPT}
    if brief is not None:
        prior["promo_motion_brief"] = brief
    client = _client()

    with pytest.raises(StageSkippedError) as raised:
        await MotionGraphicsGenStage(client=client).run({}, ctx=_ctx({}, prior))

    assert raised.value.missing == ["motion_graphics"]
    assert client.record_animation.await_count == 0


# ── one brand resolution for cards, animated slides and the end card ────────


async def test_motion_cards_the_end_card_and_the_slides_get_the_same_brand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``theme == {}`` and a name-only brand (run 01a11082): one palette, one font
    — for a run that opted in to per-line motion (the default path is pinned to
    origin/main in ``test_promo_per_line_gates``)."""
    snapshot: dict[str, object] = {
        "direction": {"motion": "per_line"},
        "title": "Merqi",
        "brand": {"name": "Merqi"},
        "theme": {},
        "deliverable": {"format": "16x9"},
        "resolution": "1080p",
        "title_card": {"enabled": True, "intro": False, "outro": True},
    }
    expected = {
        "accent": "#2BB6A3",
        "accent2": "#F26B3A",
        "brand_mark": "Merqi",
        "bg": "#0B1F3A",
        "bg2": "#0B1F3A",
        "text": "#F8FAFC",
        "font_display": "'Inter','IBM Plex Sans',-apple-system,sans-serif",
        "font_sans": "'Inter','IBM Plex Sans',-apple-system,sans-serif",
    }
    monkeypatch.setattr(
        "src.services.stages.shared.title_cards.find_existing_media", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "src.services.stages.shared.title_cards.store_generated_video",
        AsyncMock(return_value="https://minio.test/v/card.mp4"),
    )

    # 1. the motion cards
    cards = _client()
    ctx = _ctx(
        {**snapshot, "motion_graphics": {"items": [{"line_id": "l1", "template_id": "stat"}]}},
        prior={"promo_script": _SCRIPT},
    )
    await MotionGraphicsGenStage(client=cards).run({}, ctx=ctx)
    card_params = cards.record_animation.await_args.kwargs["params"]

    # 2. the end card
    end = _client()
    _intro, outro = await build_title_card_shots(ctx, ordered_seconds=None, client=end)
    end_params = end.record_animation.await_args.kwargs["params"]

    # 3. the animated slides (what render_deck_motion_clips resolves for its deck)
    slide_params = deck_motion_params({**_BRAND_DEFAULTS, "name": "Merqi"}, {})

    assert outro is not None
    assert card_params == expected
    assert {key: end_params[key] for key in expected} == expected
    assert slide_params == expected

    # 4. NOT a promo (no promo_script): the end card keeps the params every other
    # product always sent — no font keys, so its cached cards still address.
    other = _client()
    plain = _ctx(snapshot, prior={})
    _intro, plain_outro = await build_title_card_shots(plain, ordered_seconds=None, client=other)
    plain_params = other.record_animation.await_args.kwargs["params"]
    assert plain_outro is not None
    assert "font_display" not in plain_params and "font_sans" not in plain_params


def test_a_project_brand_and_theme_override_the_shared_default() -> None:
    params = run_motion_brand_params(
        {
            "brand": {"name": "Acme", "secondary_color": "#00FFAA", "font_family": "Poppins"},
            "theme": {"gradient_start": "#101010", "gradient_end": "#202020"},
        }
    )

    assert params["accent"] == "#00FFAA"
    assert (params["bg"], params["bg2"]) == ("#101010", "#202020")
    assert params["font_display"] == "'Poppins','IBM Plex Sans',-apple-system,sans-serif"
    assert params["accent2"] == _BRAND_DEFAULTS["accent_color"]
