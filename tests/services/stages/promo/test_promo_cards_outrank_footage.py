"""Per-line motion: a line's recorded card IS its shot (audit of #1074 at 4c18e6e3).

The auditor's probe: a real brief, a real ``motion_graphics_gen``, a real
``PromoRenderStage.run``, every storyboard line typed ``footage`` and one capture
clip — what a marketing request produces, since it always sends
``brief.source_url``. Seven cards were recorded and none was placed: every shot
was the site clip, the last line ran 12.1 s of footage, and the film had no end
card at all, because the end card was absorbed on the brief's OUTPUT alone.

The rule these tests hold: under ``direction.motion == "per_line"`` a line's
card outranks captured footage, html and curated media for that line; footage is
for a line with no card; the end card is absorbed only when the closing SHOT
carries its card; a recorded card no shot carries is recorded. Without the flag
the old ranking stands.

Every test drives the real ``PromoRenderStage.run`` after the real brief and
card stages; the capture/director/footage-direction outputs are built from those
stages' own output models.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.generic.motion_graphics_gen import MotionGraphicsGenStage
from src.services.stages.generic.site_capture import SiteCaptureOutput, SiteClip, SiteScreen
from src.services.stages.promo.fill import CLOSING_TAIL_PLANNED_SECONDS, CTA_MIN_SECONDS
from src.services.stages.promo.visual_director import LineScreen, PromoVisualDirectorOutput
from src.services.stages.shared.degradation import MOTION_CLIP_UNUSED
from tests.services.stages.promo.test_motion_brief import _brief_and_cards
from tests.services.stages.promo.test_promo_frame_qa_2026_10_05 import (
    _CTA,
    _LINES,
    _kinetic,
    _promo_cfg,
    _render,
    _vp,
)

_LINE_IDS = [line_id for line_id, _text, _seconds in _LINES]
_NARRATED = {line_id: seconds for line_id, _text, seconds in _LINES}
_PER_LINE = {"motion": "per_line"}
_HOLD = 2.0  # what the fill plan gives each line of this 60 s order
# ... and its end card: on a per-line film the closing shot ends <= 3.0 s after
# the CTA line's narration as DELIVERED (fill.CLOSING_TAIL_MAX_SECONDS); the fill
# plans 2.957 s (one 24 fps frame + 1 ms under it, fill.CLOSING_TAIL_PLANNED_SECONDS,
# run 01a11511) - the 2.0 s hold.
_END_CARD = round(CLOSING_TAIL_PLANNED_SECONDS - 2.0, 3)

_SITE_CLIP = "https://minio.test/v/site-walkthrough.mp4"
_SCREEN_CLIP = "https://minio.test/v/screen-pricing.mp4"
_SCREEN_STILL = "https://minio.test/i/screen-pricing.png"
_FOOTAGE = {_SITE_CLIP, _SCREEN_CLIP, _SCREEN_STILL}
#: Keys that frame FOOTAGE on a shot; a card shot carries none of them.
_FOOTAGE_KEYS = (
    "focus_track",
    "cursor_track",
    "footage_viewport",
    "callouts",
    "split",
    "clip_lead_seconds",
    "clip_out_seconds",
    "clip_anchor",
)


@pytest.fixture(autouse=True)
def _gen_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = "src.services.stages.generic.motion_graphics_gen"
    monkeypatch.setattr(f"{gen}.find_existing_media", AsyncMock(return_value=None))
    monkeypatch.setattr(
        f"{gen}.store_generated_video",
        AsyncMock(side_effect=lambda **kw: f"https://minio.test/v/{kw['video_id']}.mp4"),
    )


def _site_footage(ctx: RunContext, *, directed: bool = False) -> None:
    """What a ``brief.source_url`` request leaves behind: every storyboard line
    typed ``footage`` and a captured primary clip. ``directed`` adds the
    per-screen capture (a screen with a reveal clip + recorded focus/cursor
    geometry), the visual director's line->screen map for l2/l3 and a
    footage-direction plan — everything ``build_shots`` keys to a footage line."""
    for shot in ctx.prior_run_outputs["promo_storyboard"]["shots"]:
        shot["visual_type"] = "footage"
    screens = (
        [
            SiteScreen(
                id="pricing",
                label="Pricing",
                image_url=_SCREEN_STILL,
                clip_url=_SCREEN_CLIP,
                focus_track=[
                    {"t": 2.0, "bbox": {"x": 10, "y": 20, "w": 100, "h": 40}, "action": "click"}
                ],
                cursor_track=[{"t": 1.6, "x": 1, "y": 1}, {"t": 2.0, "x": 9, "y": 9}],
                viewport={"width": 1280, "height": 720},
            )
        ]
        if directed
        else []
    )
    ctx.prior_run_outputs["site_capture"] = SiteCaptureOutput(
        clips=[
            SiteClip(id="clip-1", video_url=_SITE_CLIP, seconds=42.0, path="/", is_primary=True)
        ],
        page_text="Go-to-market that runs itself.",
        page_text_status="ok",
        capture_status="ok",
        notes=[],
        screens=screens,
        used_real_footage=True,
    ).model_dump(mode="json")
    if not directed:
        return
    ctx.prior_run_outputs["promo_visual_director"] = PromoVisualDirectorOutput(
        assignments=[
            LineScreen(id="a-l2", line_id="l2", screen_id="pricing"),
            LineScreen(id="a-l3", line_id="l3", screen_id="pricing"),
        ]
    ).model_dump(mode="json")
    ctx.prior_run_outputs["promo_footage_direction"] = {
        "plans": [
            {"line_id": "l2", "zoom": "auto", "callout": {"type": "badge", "label": "Click"}},
            {"line_id": "l3", "zoom": "auto", "split": {"layout": "h", "second_clip": "l2"}},
        ]
    }
    for shot in ctx.prior_run_outputs["promo_storyboard"]["shots"]:
        if shot["line_id"] in ("l2", "l3"):
            # An author's footage framing on the storyboard line.
            shot.update(clip_anchor="start", clip_lead_seconds=1.5, fit="cover")


def _cards(ctx: RunContext) -> dict[str, str]:
    """``line_id -> card url`` exactly as ``motion_graphics_gen`` emitted them."""
    return {
        clip["line_id"]: clip["video_url"]
        for clip in ctx.prior_run_outputs["motion_graphics_gen"]["clips"]
    }


def _drop_card(ctx: RunContext, line_id: str) -> None:
    """The card of ``line_id`` was never recorded (its clip is not in the output)."""
    output = ctx.prior_run_outputs["motion_graphics_gen"]
    output["clips"] = [clip for clip in output["clips"] if clip["line_id"] != line_id]


def _unused(out: Any) -> list[str]:
    return [a["detail"] for a in out.advisories if a["code"] == MOTION_CLIP_UNUSED]


# ── 1. every line has a card: the film is its cards, footage or not ─────────


async def test_per_line_cards_are_the_film_even_when_every_line_is_typed_footage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, RunContext] = {}

    async def _prepare(ctx: RunContext) -> None:
        _site_footage(ctx, directed=True)
        await _brief_and_cards(ctx, [])
        seen["ctx"] = ctx

    cfg = _promo_cfg(direction={**_PER_LINE, "footage": {"auto_zoom": True}})
    out, captured, title_vp = await _render(monkeypatch, cfg, 57.4, prepare=_prepare)
    shots = captured["shots"]
    cards = _cards(seen["ctx"])

    assert sorted(cards) == _LINE_IDS  # the brief authored, and the stage recorded, all seven
    # One shot per line, each one its own card; no separate end card.
    assert [shot["line_id"] for shot in shots] == _LINE_IDS
    assert {shot["line_id"]: shot["video_url"] for shot in shots} == cards
    assert not _FOOTAGE & {shot["video_url"] for shot in shots}
    # The closing card is there and its shot runs the line, the hold AND the end
    # card's seconds: the closing beat is the end card.
    closing = shots[-1]
    assert closing["video_url"] == cards["l7"]
    assert closing["seconds"] == pytest.approx(_NARRATED["l7"] + _HOLD + _END_CARD)
    assert closing["max_seconds"] == pytest.approx(closing["seconds"])
    title_vp.record_animation.assert_not_awaited()
    assert _unused(out) == []  # nothing recorded was dropped


async def test_a_card_shot_carries_no_footage_framing(monkeypatch: pytest.MonkeyPatch) -> None:
    """l2 and l3 are keyed to a captured screen (reveal clip, focus/cursor track,
    viewport), planned by footage direction (callout, split) and framed by the
    author (anchor, lead, ``fit: cover``). l2 has a card, l3 does not: only l3
    may carry any of it, and l3 — now the first FOOTAGE line on that screen —
    gets its reveal clip."""

    async def _prepare(ctx: RunContext) -> None:
        _site_footage(ctx, directed=True)
        await _brief_and_cards(ctx, [])
        _drop_card(ctx, "l3")

    cfg = _promo_cfg(direction={**_PER_LINE, "footage": {"auto_zoom": True}})
    _out, captured, _title_vp = await _render(monkeypatch, cfg, 57.4, prepare=_prepare)
    by_line = {shot["line_id"]: shot for shot in captured["shots"]}

    card = by_line["l2"]
    assert [key for key in _FOOTAGE_KEYS if key in card] == []
    assert card["fit"] == "contain"  # the card's own fit, not the author's footage "cover"
    footage = by_line["l3"]
    assert footage["video_url"] == _SCREEN_CLIP
    assert footage["fit"] == "cover" and footage["clip_lead_seconds"] == 1.5
    assert footage["focus_track"] and footage["cursor_track"]
    assert footage["footage_viewport"] == [1280, 720]
    # l3's split names l2 as its second pane; l2 shows a card, not footage, so
    # the card is never used as footage beside it.
    assert "split" not in footage


# ── 2. footage is for the line WITHOUT a card ───────────────────────────────


async def test_a_line_without_a_card_uses_footage_and_the_rest_keep_their_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, RunContext] = {}

    async def _prepare(ctx: RunContext) -> None:
        _site_footage(ctx)
        await _brief_and_cards(ctx, [])
        _drop_card(ctx, "l4")
        seen["ctx"] = ctx

    cfg = _promo_cfg(direction=_PER_LINE)
    out, captured, title_vp = await _render(monkeypatch, cfg, 57.4, prepare=_prepare)
    shots = captured["shots"]
    cards = _cards(seen["ctx"])

    assert [shot["line_id"] for shot in shots] == _LINE_IDS
    assert {shot["line_id"]: shot["video_url"] for shot in shots} == {**cards, "l4": _SITE_CLIP}
    # The closing shot still carries its card, so it is still the end card.
    assert shots[-1]["seconds"] == pytest.approx(_NARRATED["l7"] + _HOLD + _END_CARD)
    title_vp.record_animation.assert_not_awaited()
    assert _unused(out) == []


async def test_a_closing_line_without_its_card_keeps_the_separate_end_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The brief sized the closing beat to swallow the end card, but its card was
    not recorded: the closing shot is footage at its own length and the end card
    is rendered and spliced as it is without the brief."""
    seen: dict[str, RunContext] = {}

    async def _prepare(ctx: RunContext) -> None:
        _site_footage(ctx)
        await _brief_and_cards(ctx, [])
        _drop_card(ctx, "l7")
        seen["ctx"] = ctx

    cfg = _promo_cfg(direction=_PER_LINE)
    _out, captured, title_vp = await _render(monkeypatch, cfg, 57.4, prepare=_prepare)
    shots = captured["shots"]
    brief = seen["ctx"].prior_run_outputs["promo_motion_brief"]

    assert brief["closing_beat"]["end_card_seconds"] == _END_CARD  # the brief DID plan it
    assert [shot["line_id"] for shot in shots] == [*_LINE_IDS, "__title_outro__"]
    closing, end_card = shots[-2], shots[-1]
    assert closing["video_url"] == _SITE_CLIP
    assert closing["seconds"] == pytest.approx(_NARRATED["l7"] + _HOLD)  # NOT + the end card
    # The separate card is the only place the CTA is shown: it keeps the CTA floor
    # (2.5 s), not the ~1.0 s remainder of the closing tail the absorbed card gets.
    assert end_card["seconds"] == pytest.approx(CTA_MIN_SECONDS)
    assert end_card["video_url"] == "https://minio.test/v/outro.mp4"
    title_vp.record_animation.assert_awaited_once()
    params = title_vp.record_animation.await_args.kwargs["params"]
    assert (params["title"], params["subtitle"]) == ("Merqi", _CTA)
    # The film's planned total is what it is with every card: line + hold, + the card.
    assert sum(float(shot["seconds"]) for shot in shots) == pytest.approx(
        sum(_NARRATED.values()) + 7 * _HOLD + CTA_MIN_SECONDS
    )


async def test_a_recorded_card_no_shot_carries_is_recorded_by_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paid for and not shown is never silent: the storyboard has no shot for l4,
    so l4's recorded card has nowhere to go — exactly one advisory, naming l4."""

    async def _prepare(ctx: RunContext) -> None:
        _site_footage(ctx)
        await _brief_and_cards(ctx, [])
        story = ctx.prior_run_outputs["promo_storyboard"]
        story["shots"] = [shot for shot in story["shots"] if shot["line_id"] != "l4"]

    cfg = _promo_cfg(direction=_PER_LINE)
    out, captured, _title_vp = await _render(monkeypatch, cfg, 57.4, prepare=_prepare)

    assert "l4" not in [shot["line_id"] for shot in captured["shots"]]
    (detail,) = _unused(out)
    assert detail.startswith("motion clip keyed to line l4 was rendered but no shot")


# ── 3. without the flag the old ranking stands ──────────────────────────────


async def test_without_per_line_a_keyed_card_does_not_displace_footage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request's own keyed kinetic cards on footage lines, no ``per_line``:
    footage outranks them as it always did, the separate end card is spliced, and
    each unplaced card is recorded with the reason it always had."""

    async def _prepare(ctx: RunContext) -> None:
        _site_footage(ctx)
        out = await MotionGraphicsGenStage(client=_vp()).run({}, ctx=ctx)
        ctx.prior_run_outputs["motion_graphics_gen"] = out.model_dump(mode="json")
        assert sorted(_cards(ctx)) == ["l1", "l6"]

    cfg = _promo_cfg(motion_graphics=_kinetic(_LINES[0][1], _LINES[5][1]))
    out, captured, title_vp = await _render(monkeypatch, cfg, 57.4, prepare=_prepare)
    shots = captured["shots"]

    assert [shot["line_id"] for shot in shots] == [*_LINE_IDS, "__title_outro__"]
    assert {shot["video_url"] for shot in shots[:-1]} == {_SITE_CLIP}
    assert shots[-2]["seconds"] == pytest.approx(_NARRATED["l7"] + _HOLD)
    title_vp.record_animation.assert_awaited_once()
    assert _unused(out) == [
        f"motion clip keyed to line {line_id} was rendered but line {line_id} shipped "
        "another visual (footage or curated media outranks motion)"
        for line_id in ("l1", "l6")
    ]
