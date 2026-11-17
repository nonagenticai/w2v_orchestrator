"""Caption typography for per-line promos (frame QA of run 01a11295, 2026-10-06).

The burned-in captions were DejaVu Sans on a black plate under Inter cards on navy.
A per-line promo now sends the cards' font and palette (video_compiler #246
fields); every other run sends nothing new.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.primitives.compile_stage import CompileStageBase
from src.services.stages.promo.render import CAPTION_BOX_OPACITY, PromoRenderStage


def _ctx(config: dict[str, Any]) -> RunContext:
    return RunContext(
        run_id=uuid4(), deliverable_id=uuid4(), config_snapshot=config, prior_run_outputs={}
    )


_PER_LINE = {"brand": {"name": "Merqi"}, "theme": {}, "direction": {"motion": "per_line"}}


def test_a_per_line_promo_captions_in_the_cards_type_and_palette() -> None:
    assert PromoRenderStage().caption_look(_ctx(_PER_LINE)) == {
        "caption_font_family": "Inter",
        "caption_text_colour": "#F8FAFC",
        "caption_box_colour": "#0B1F3A",
        "caption_box_opacity": CAPTION_BOX_OPACITY,
    }


def test_the_look_follows_the_brand_not_a_constant() -> None:
    look = PromoRenderStage().caption_look(
        _ctx({**_PER_LINE, "brand": {"name": "Acme", "primary_color": "#101820"}})
    )
    assert look is not None and look["caption_box_colour"] == "#101820"


@pytest.mark.parametrize("config", [{"brand": {"name": "Merqi"}}])
def test_without_the_per_line_brief_nothing_new_is_sent(config: dict[str, Any]) -> None:
    assert PromoRenderStage().caption_look(_ctx(config)) is None


def test_other_products_send_no_look() -> None:
    assert CompileStageBase.caption_look(PromoRenderStage(), _ctx(_PER_LINE)) is None


def test_the_client_sends_each_set_key_and_only_with_cues() -> None:
    from src.clients import video_compile

    payload: dict[str, object] = {}
    look = {
        "caption_font_family": "Inter",
        "caption_text_colour": None,
        "caption_box_opacity": 0.85,
    }
    video_compile._fold_captions(payload, [{"shot_index": 0}], "burned_in", None, False, look)
    assert payload["caption_font_family"] == "Inter"
    assert payload["caption_box_opacity"] == 0.85
    assert "caption_text_colour" not in payload and "caption_box_colour" not in payload

    bare: dict[str, object] = {}
    video_compile._fold_captions(bare, None, "burned_in", None, False, look)
    assert bare == {}  # no cues: no caption fields at all

    plain: dict[str, object] = {}
    video_compile._fold_captions(plain, [{"shot_index": 0}], "burned_in", None, False, None)
    assert plain == {"captions": [{"shot_index": 0}], "captions_render": "burned_in"}


@pytest.mark.asyncio
async def test_the_real_promo_compile_sends_its_look(monkeypatch: pytest.MonkeyPatch) -> None:
    """Through ``PromoRenderStage.run``: the compile call carries the hook's look,
    and a run without the per-line brief carries none."""
    from tests.services.stages.promo.test_promo_frame_qa_2026_10_05 import _promo_cfg, _render

    _out, captured, _vp = await _render(
        monkeypatch, {**_promo_cfg(), "direction": {"motion": "per_line"}}, 57.4
    )
    assert captured["caption_look"] == {
        "caption_font_family": "Inter",
        "caption_text_colour": "#F8FAFC",
        # This fixture's brand sets primary_color #000000: the box follows it.
        "caption_box_colour": "#000000",
        "caption_box_opacity": CAPTION_BOX_OPACITY,
    }
    _out, captured, _vp = await _render(monkeypatch, _promo_cfg(), 57.4)
    assert captured["caption_look"] is None
