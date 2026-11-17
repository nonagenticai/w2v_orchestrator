"""Without ``direction.motion == "per_line"`` a promo renders exactly as on main.

The per-line motion work (#1074) changed four things for EVERY promo before it
was gated: the end card's brand params, the brand params sent with each motion
card, a no-text clause on every image prompt, and the local Ken Burns move of a
same-aspect slide. Each is now behind the flag.

The expected values of the default path are LITERALS recorded by running
origin/main (``797d3cbe``) on these exact inputs — not this branch's own
resolver, which is what let the change through unnoticed. The mirrored
``per_line`` tests show each of the four does change once the flag is set.
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from PIL import Image

from src.services.runs.run_context import RunContext
from src.services.stages.generic import motion_graphics_gen as gen
from src.services.stages.generic.motion_graphics_gen import MotionGraphicsGenStage
from src.services.stages.promo.image_prompts import PromoImagePromptsInput, PromoImagePromptsStage
from src.services.stages.promo.render import PromoRenderStage
from src.services.stages.shared import ken_burns_local, title_cards
from src.services.stages.shared.degradation import Degradation

_PER_LINE: dict[str, object] = {"motion": "per_line"}
_LINES = [
    ("l1", "Go-to-market that runs itself, under your control.", 2.778),
    ("l7", "Request access at merqi.nl", 2.039),
]
# The brand resolution a per-line motion run shares between its cards and its
# end card (``run_motion_brand_params`` of a name-only brand and ``theme == {}``).
_PER_LINE_BRAND = {
    "accent": "#2BB6A3",
    "accent2": "#F26B3A",
    "bg": "#0B1F3A",
    "bg2": "#0B1F3A",
    "brand_mark": "Merqi",
    "font_display": "'Inter','IBM Plex Sans',-apple-system,sans-serif",
    "font_sans": "'Inter','IBM Plex Sans',-apple-system,sans-serif",
    "text": "#F8FAFC",
}


# Content addresses (sha1 media ids, not secrets) origin/main 797d3cbe gives the
# three recordings of these inputs. Equal ids mean a recording cached before this
# change still resolves, and nothing is paid for twice.
_MAIN_END_CARD_ID = "f1c4cd7e2cbada39a272e4e448bcf8c1a9d4bb1b"  # pragma: allowlist secret
_MAIN_KINETIC_CARD_ID = "c2372e15b40e30c877e5f024a027018f7363d1de"  # pragma: allowlist secret
_MAIN_LOCAL_CLIP_ID = "69f92dc2b403d150bdcce9348e9069ba56807d27"  # pragma: allowlist secret


def _ctx(direction: dict[str, object] | None, **extra: object) -> RunContext:
    """Name-only brand, ``theme == {}`` — the request of run 01a11082."""
    snapshot: dict[str, object] = {
        "brand": {"name": "Merqi"},
        "theme": {},
        "brief": {"call_to_action": "Request access at merqi.nl", "target_duration_seconds": 60},
        "deliverable": {"format": "16x9", "target_duration_seconds": 60},
        "resolution": "4k",
        **extra,
    }
    if direction is not None:
        snapshot["direction"] = direction
    return RunContext(
        run_id=UUID(int=1),
        deliverable_id=UUID(int=2),
        config_snapshot=snapshot,
        prior_run_outputs={
            "promo_script": {
                "lines": [{"id": i, "text": t, "seconds": 3.0} for i, t, _s in _LINES]
            },
            "promo_storyboard": {
                "shots": [
                    {"id": f"s-{i}", "line_id": i, "seconds": 3.0, "visual_type": "ai_image"}
                    for i, _t, _s in _LINES
                ]
            },
            "promo_narrate": {"clips": [{"line_id": i, "seconds": s} for i, _t, s in _LINES]},
        },
    )


def _vp() -> AsyncMock:
    client = AsyncMock()
    client.record_animation = AsyncMock(return_value="job")
    client.wait_for_content_recording = AsyncMock(
        return_value={"status": "completed", "duration_seconds": 4.0}
    )
    client.get_recording_video = AsyncMock(return_value=(b"\x00mp4", "video/mp4", None))
    return client


# ── (a) the end card ────────────────────────────────────────────────────────


async def _end_card(
    monkeypatch: pytest.MonkeyPatch, direction: dict[str, object] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(record_animation kwargs, outro shot)`` of the promo's default end card,
    rendered the way the promo compile renders it."""
    monkeypatch.setattr(title_cards, "find_existing_media", AsyncMock(return_value=None))
    monkeypatch.setattr(
        title_cards, "store_generated_video", AsyncMock(return_value="https://m/end.mp4")
    )
    ctx, stage, client = _ctx(direction), PromoRenderStage(), _vp()
    _intro, outro = await title_cards.render_title_card_plan(
        ctx,
        stage.title_card_plan(ctx),
        client=client,
        cfg=stage.title_card_config(ctx),
        settled_lead_seconds=stage.title_card_settled_lead_seconds(ctx),
    )
    assert outro is not None
    return client.record_animation.await_args.kwargs, outro


async def test_default_end_card_params_are_mains(monkeypatch: pytest.MonkeyPatch) -> None:
    sent, shot = await _end_card(monkeypatch, None)

    # Recorded from origin/main 797d3cbe: brand {"name": "Merqi"}, theme {}.
    assert sent["params"] == {
        "brand_mark": "Merqi",
        "eyebrow": "",
        "subtitle": "Request access at merqi.nl",
        "title": "Merqi",
    }
    assert sent["template_id"] == "animated_title_card"
    assert sent["duration_ms"] == 10800
    assert sent["part_id"] == "title-outro"
    # Same content address as main's card, so a cached recording still resolves.
    assert sent["idempotency_key"].endswith(f":title_card:{_MAIN_END_CARD_ID}")
    assert (shot["clip_lead_seconds"], shot["seconds"]) == (2.8, 8.0)


# ── (b) the motion cards ────────────────────────────────────────────────────


async def _kinetic_card(
    monkeypatch: pytest.MonkeyPatch, direction: dict[str, object] | None
) -> tuple[dict[str, Any], list[str]]:
    """``(record_animation kwargs, stored video ids)`` of one kinetic card from a
    single-template ``motion_graphics`` block."""
    stored: list[str] = []

    async def _store(**kw: Any) -> str:
        stored.append(kw["video_id"])
        return f"https://minio.test/v/{kw['video_id']}.mp4"

    monkeypatch.setattr(gen, "find_existing_media", AsyncMock(return_value=None))
    monkeypatch.setattr(gen, "store_generated_video", _store)
    client = _vp()
    ctx = _ctx(
        direction,
        motion_graphics={
            "template_id": "kinetic_typography",
            "items": [{"params": {"words": _LINES[0][1].split()}}],
        },
    )
    out = await MotionGraphicsGenStage(client=client).run({}, ctx=ctx)
    assert [clip.line_id for clip in out.clips] == ["l1"]
    return client.record_animation.await_args.kwargs, stored


_WORDS = ["Go-to-market", "that", "runs", "itself,", "under", "your", "control."]


async def test_default_motion_card_params_are_mains(monkeypatch: pytest.MonkeyPatch) -> None:
    sent, stored = await _kinetic_card(monkeypatch, None)

    # Recorded from origin/main 797d3cbe: exactly the block's params, nothing merged.
    assert sent["params"] == {"words": _WORDS}
    assert sent["template_id"] == "kinetic_typography"
    assert (sent["width"], sent["height"], sent["fps"]) == (3840, 2160, 30)
    assert sent["duration_ms"] is None and sent["html"] is None
    # Same content address as main's clip: no second charge for a cached card.
    assert stored == [_MAIN_KINETIC_CARD_ID]


async def test_per_line_motion_card_takes_the_shared_brand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent, stored = await _kinetic_card(monkeypatch, _PER_LINE)

    assert sent["params"] == {**_PER_LINE_BRAND, "words": _WORDS}
    assert stored != [_MAIN_KINETIC_CARD_ID]


# ── (c) the image prompts ───────────────────────────────────────────────────

_MODEL_PROMPTS = ["A sleek interface of glowing nodes", "A doorway opening onto a teal horizon."]


async def _image_prompts(direction: dict[str, object] | None) -> list[str]:
    wm = MagicMock()
    wm.message = AsyncMock(
        return_value=SimpleNamespace(
            output={
                "prompts": [
                    {"line_id": i, "prompt": prompt}
                    for (i, _t, _s), prompt in zip(_LINES, _MODEL_PROMPTS, strict=True)
                ]
            },
            usage=None,
        )
    )
    out = await PromoImagePromptsStage(client=wm).handle(PromoImagePromptsInput(), _ctx(direction))
    return [prompt.prompt for prompt in out.prompts]


async def test_default_image_prompts_are_returned_verbatim() -> None:
    # Recorded from origin/main 797d3cbe: the model's prompt, untouched.
    assert await _image_prompts(None) == [
        "A sleek interface of glowing nodes",
        "A doorway opening onto a teal horizon.",
    ]


async def test_per_line_image_prompts_end_by_forbidding_text() -> None:
    clause = "No text, no letters, no words, no numbers, no signage, no logos, no UI."
    assert await _image_prompts(_PER_LINE) == [
        f"A sleek interface of glowing nodes. {clause}",
        f"A doorway opening onto a teal horizon. {clause}",
    ]


# ── (d) the local Ken Burns move of a same-aspect slide ─────────────────────

# origin/main 797d3cbe, a 16:9 slide in a 16:9 4K film (rendered at 1080p), 4.0 s
# at 30 fps: the never-crop ``contain`` graph — inset over a blurred copy.
_MAIN_CONTAIN_GRAPH = (
    "[0:v]split[bg_in][fg_in];"
    "[bg_in]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,boxblur=20:2[bg];"
    "[fg_in]scale=1812:1018:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
    "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,"
    "perspective=x0='W*(1-1/(1+0.0596026490066226*on/119))/2'"
    ":y0='H*(1-1/(1+0.0596026490066226*on/119))/2'"
    ":x1='W*(1-(1-1/(1+0.0596026490066226*on/119))/2)'"
    ":y1='H*(1-1/(1+0.0596026490066226*on/119))/2'"
    ":x2='W*(1-1/(1+0.0596026490066226*on/119))/2'"
    ":y2='H*(1-(1-1/(1+0.0596026490066226*on/119))/2)'"
    ":x3='W*(1-(1-1/(1+0.0596026490066226*on/119))/2)'"
    ":y3='H*(1-(1-1/(1+0.0596026490066226*on/119))/2)'"
    ":sense=source:eval=frame:interpolation=cubic,fps=30,format=yuv420p[out]"
)


async def _rescued_slide(
    monkeypatch: pytest.MonkeyPatch, direction: dict[str, object] | None
) -> tuple[str, list[Degradation], list[str]]:
    """``(ffmpeg filter graph, advisories, stored video ids)`` of the local rescue
    of one 16:9 slide the recorder failed — the real ``rescue_frozen_stills`` and
    ``render_local_ken_burns``; only ffmpeg, the fetch and the store are faked."""
    slide = io.BytesIO()
    Image.new("RGB", (1920, 1080), (11, 31, 58)).save(slide, format="PNG")
    commands: list[list[str]] = []
    stored: list[str] = []

    async def _run(cmd: list[str], _timeout_s: float) -> bytes:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"\x00mp4")
        return b""

    async def _fetch(_url: str, _client: object) -> tuple[bytes, str]:
        return slide.getvalue(), "image/png"

    async def _store(**kw: Any) -> str:
        stored.append(kw["video_id"])
        return f"https://minio.test/v/{kw['video_id']}.mp4"

    monkeypatch.setattr(ken_burns_local, "_run", _run)
    monkeypatch.setattr(ken_burns_local, "_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(ken_burns_local.kb, "_fetch_own_asset", _fetch)
    monkeypatch.setattr(ken_burns_local, "store_generated_video", _store)
    advisories: list[Degradation] = []
    out = await ken_burns_local.rescue_frozen_stills(
        _ctx(direction),
        [{"id": "s-l7", "line_id": "l7", "image_url": "https://m/slide-l7.png", "seconds": 4.0}],
        notes=["ken_burns: shot s-l7 soft-failed, kept as a still: RuntimeError: QUEUE_SATURATED"],
        advisories=advisories,
        budget_seconds=600.0,
    )
    assert out[0]["video_url"]
    (command,) = commands
    return command[command.index("-filter_complex") + 1], advisories, stored


async def test_default_same_aspect_slide_takes_mains_contain_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, advisories, stored = await _rescued_slide(monkeypatch, None)

    assert graph == _MAIN_CONTAIN_GRAPH
    # Recorded from origin/main 797d3cbe: the record and the clip's address too.
    assert [a["detail"] for a in advisories] == [
        "shot s-l7: recorder clip failed (RuntimeError: QUEUE_SATURATED); animated locally "
        "(contain-safe centred push-in) instead of a frozen still"
    ]
    assert stored == [_MAIN_LOCAL_CLIP_ID]


async def test_per_line_same_aspect_slide_fills_the_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    graph, advisories, _stored = await _rescued_slide(monkeypatch, _PER_LINE)

    assert graph == (
        "[0:v]scale=1920:1080:flags=lanczos,setsar=1,"
        "eq=brightness='0.03*sin(PI*n/119)':eval=frame,fps=30,format=yuv420p[out]"
    )
    assert "full-frame, no geometric move" in advisories[0]["detail"]
