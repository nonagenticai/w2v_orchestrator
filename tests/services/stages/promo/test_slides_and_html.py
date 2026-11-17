"""Unit tests for promo_slides + promo_html helpers (pure logic, no I/O)."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.schemas.design import DesignJobResult, DesignJobStatus, DesignVisualAsset
from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.promo.html_segments import PromoHtmlStage
from src.services.stages.promo.slides import (
    _BRAND_DEFAULTS,
    PromoSlidesStage,
)


def _design_result(**kwargs) -> DesignJobResult:
    return DesignJobResult(
        job_id="job-1", url="https://example.com", scope="hero", style="modern", **kwargs
    )


def _asset(
    *,
    asset_id: str = "a1",
    title: str = "Hero",
    kind: str = "screenshot",
    role: str = "after",
    url: str = "",
    source: str = "design",
    page_url: str | None = None,
    metadata: dict[str, object] | None = None,
) -> DesignVisualAsset:
    return DesignVisualAsset(
        asset_id=asset_id,
        title=title,
        kind=kind,
        role=role,
        url=url,
        source=source,
        page_url=page_url,
        metadata=metadata,
    )


def test_collect_after_images_prefers_visual_assets() -> None:
    result = _design_result(
        visual_assets=[
            _asset(asset_id="a1", role="after", url="https://x/after.png"),
            _asset(asset_id="b1", role="before", url="https://x/before.png"),
        ],
        screenshots={"desktop": "https://x/shot.png"},
    )
    segs = PromoHtmlStage._collect_after_images(result)
    urls = [s.image_url for s in segs]
    assert urls == ["https://x/after.png"]  # only "after", asset-source preferred


def test_collect_after_images_falls_back_to_screenshots() -> None:
    result = _design_result(visual_assets=[], screenshots={"desktop": "https://x/shot.png"})
    segs = PromoHtmlStage._collect_after_images(result)
    assert [s.image_url for s in segs] == ["https://x/shot.png"]


def test_collect_after_images_drops_unfetchable() -> None:
    result = _design_result(
        visual_assets=[_asset(role="after", url="/local/only.png")], screenshots={}
    )
    assert PromoHtmlStage._collect_after_images(result) == []


def _promo_html_ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"source_url": "https://example.com", "redesign_style": "modern"}
        },
        prior_run_outputs={},
    )


@pytest.mark.asyncio
async def test_promo_html_requests_lean_render_shape() -> None:
    """The promo opts into a single desktop hero shot — no original capture,
    no comparison, no full-page — so html-render-service doesn't OOM."""
    fake = AsyncMock()
    fake.start_design.return_value = "job-1"
    fake.wait_for_design.return_value = DesignJobStatus(job_id="job-1", status="completed")
    fake.get_design_result.return_value = _design_result(visual_assets=[], screenshots={})
    stage = PromoHtmlStage(client=fake)

    await stage.handle(PromoHtmlStage.input_model(), _promo_html_ctx())

    kwargs = fake.start_design.await_args.kwargs
    assert kwargs["scope"] == "hero"
    assert kwargs["capture_original"] is False
    assert kwargs["generate_comparison"] is False
    assert kwargs["screenshot_viewports"] == ["desktop"]
    assert kwargs["full_page"] is False


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


def _promo_ctx(brand: dict | None = None) -> RunContext:
    # Pin the base tier so the dims assertions below test the 16:9→1920x1080
    # mapping; the 2K/4K scaling is covered in test_resolution_tier.py.
    snapshot: dict = {"deliverable": {"title": "Common Practice"}, "resolution": "1080p"}
    if brand is not None:
        snapshot["brand"] = brand
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=snapshot,
        prior_run_outputs={
            "promo_script": {
                "lines": [
                    {"id": "l1", "text": "Vote on what ships."},
                    {
                        "id": "l2",
                        "text": "The people closest to the work decide what is worth improving.",
                    },
                ]
            }
        },
    )


@pytest.mark.asyncio
async def test_promo_slides_routes_title_and_cta_templates_and_forwards_brand(fake_storage) -> None:
    fake = AsyncMock()
    fake.resolve_template_id.side_effect = lambda name: {"title": "tpl-title", "cta": "tpl-cta"}[
        name
    ]
    fake.start_deck.return_value = "deck-1"
    fake.wait_for_deck.return_value = {"status": "completed", "slide_count": 2}
    fake.get_slide_png.side_effect = [b"\x89PNG-0", b"\x89PNG-1"]
    stage = PromoSlidesStage(client=fake)

    out = await stage.run({}, ctx=_promo_ctx())

    # Both the title and the cta templates are resolved.
    resolved = {c.args[0] for c in fake.resolve_template_id.await_args_list}
    assert resolved == {"title", "cta"}
    payload = fake.start_deck.await_args.args[0]
    # Brand defaults are forwarded so slides render on-brand, not slide_service defaults.
    assert payload["brand"]["primary_color"] == _BRAND_DEFAULTS["primary_color"]
    assert payload["brand"]["secondary_color"] == _BRAND_DEFAULTS["secondary_color"]
    # Opening line → title template, full text untruncated, no image/bullets.
    first = payload["slides"][0]
    assert first["template_id"] == "tpl-title"
    assert first["content"]["title"] == "Vote on what ships."
    assert "bullets" not in first["content"]
    # Closing line → cta template.
    last = payload["slides"][-1]
    assert last["template_id"] == "tpl-cta"
    assert last["role"] == "cta"
    assert len(out.slides) == 2


@pytest.mark.asyncio
async def test_promo_slides_project_brand_overrides_defaults(fake_storage) -> None:
    fake = AsyncMock()
    fake.resolve_template_id.return_value = "tpl-title"
    fake.start_deck.return_value = "deck-1"
    fake.wait_for_deck.return_value = {"status": "completed", "slide_count": 2}
    fake.get_slide_png.side_effect = [b"\x89PNG-0", b"\x89PNG-1"]
    stage = PromoSlidesStage(client=fake)

    await stage.run({}, ctx=_promo_ctx(brand={"primary_color": "#123456"}))

    payload = fake.start_deck.await_args.args[0]
    assert payload["brand"]["primary_color"] == "#123456"  # project wins
    assert payload["brand"]["secondary_color"] == _BRAND_DEFAULTS["secondary_color"]  # default kept


# --- motion-slide projection (pure) -----------------------------------------


# --- motion-slide rendering (opt-in) ----------------------------------------


def _video_fake() -> AsyncMock:
    v = AsyncMock()
    v.record_animation.return_value = "vjob-1"
    v.wait_for_content_recording.return_value = {"duration_seconds": 2.0}
    v.get_recording_video.return_value = (b"\x00mp4", "video/mp4", None)
    return v


def _slide_client_fake() -> AsyncMock:
    fake = AsyncMock()
    fake.resolve_template_id.side_effect = lambda name: {"title": "tpl-title", "cta": "tpl-cta"}[
        name
    ]
    fake.start_deck.return_value = "deck-1"
    fake.wait_for_deck.return_value = {"status": "completed", "slide_count": 2}
    fake.get_slide_png.side_effect = [b"\x89PNG-0", b"\x89PNG-1"]
    return fake


@pytest.mark.asyncio
async def test_motion_disabled_does_not_render_clips(fake_storage) -> None:
    video = _video_fake()
    stage = PromoSlidesStage(client=_slide_client_fake(), video_client=video)
    out = await stage.run({}, ctx=_promo_ctx())  # no motion_graphics config
    video.record_animation.assert_not_awaited()
    assert all(s.motion_video_url is None for s in out.slides)


@pytest.mark.asyncio
async def test_motion_graphics_on_still_submits_no_slide_recording(fake_storage) -> None:
    """promo_render never lays the animated slide variant (its text builds in from
    an empty frame under narration that starts at the cut), so promo_slides must
    not pay the recorder for it. Run 01a10fcf (2026-10-06) recorded seven such
    clips, used none, and they held the recorder slots Ken Burns starved for."""
    video = _video_fake()
    ctx = _promo_ctx()
    ctx.config_snapshot["motion_graphics"] = {"enabled": True, "template_id": "kinetic_typography"}
    stage = PromoSlidesStage(client=_slide_client_fake(), video_client=video)

    out = await stage.run({}, ctx=ctx)

    video.record_animation.assert_not_awaited()
    assert len(out.slides) == 2
    assert all(s.image_url and s.motion_video_url is None for s in out.slides)


@pytest.mark.asyncio
async def test_motion_slides_false_keeps_stills(fake_storage) -> None:
    video = _video_fake()
    ctx = _promo_ctx()
    ctx.config_snapshot["motion_graphics"] = {"enabled": True, "slides": False}
    stage = PromoSlidesStage(client=_slide_client_fake(), video_client=video)
    out = await stage.run({}, ctx=ctx)
    video.record_animation.assert_not_awaited()
    assert all(s.motion_video_url is None for s in out.slides)


@pytest.mark.asyncio
async def test_promo_html_reuses_captured_html_without_an_anonymous_brand_fetch(monkeypatch):
    captured = "<main><section><h1>Authenticated product</h1></section></main>"
    ctx = _promo_html_ctx()
    ctx.prior_run_outputs["site_capture"] = {"html_content": captured}
    extract = AsyncMock()
    monkeypatch.setattr("src.services.stages.promo.html_segments.resolve_brand_context", extract)
    fake = AsyncMock()
    fake.start_design.return_value = "job-1"
    fake.wait_for_design.return_value = DesignJobStatus(job_id="job-1", status="completed")
    fake.get_design_result.return_value = _design_result(
        screenshots={"desktop": "https://images.example.com/hero.png"}
    )
    out = await PromoHtmlStage(client=fake).handle(PromoHtmlStage.input_model(), ctx)
    assert fake.start_design.call_args.kwargs["html_content"] == captured
    extract.assert_not_awaited()
    assert out.used_html is True


@pytest.mark.asyncio
async def test_authenticated_html_design_uses_supplied_brief_when_old_capture_lacks_content(
    monkeypatch,
):
    ctx = _promo_html_ctx()
    ctx.config_snapshot["brief"].update(
        {
            "auth_info": {"type": "basic"},
            "value_proposition": "Report bottlenecks & build solutions",
        }
    )
    extract = AsyncMock()
    monkeypatch.setattr("src.services.stages.promo.html_segments.resolve_brand_context", extract)
    fake = AsyncMock()
    fake.start_design.return_value = "job-1"
    fake.wait_for_design.return_value = DesignJobStatus(job_id="job-1", status="completed")
    fake.get_design_result.return_value = _design_result(
        screenshots={"desktop": "https://images.example.com/hero.png"}
    )
    await PromoHtmlStage(client=fake).handle(PromoHtmlStage.input_model(), ctx)
    assert (
        "Report bottlenecks &amp; build solutions"
        in fake.start_design.call_args.kwargs["html_content"]
    )
    assert "auth_info" not in fake.start_design.call_args.kwargs
    extract.assert_not_awaited()
