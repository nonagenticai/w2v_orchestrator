"""Slide-capture ``render_scale`` must be a flat 2.0 at every resolution tier.

``slide_service`` raised its own ``device_scale_factor`` default to 2.0 (its
PR #128), but this repo sends ``render_scale`` explicitly on every deck, which
overrides that default. The three sites used to send ``tier_scale(quality_tier(...))``,
which is **1.0 at the 1080p tier** — the tier most runs use — so the sibling fix
bought nothing here.

``tier_scale`` is a *delivery* scale (motion-clip / compile-canvas dimensions);
``render_scale`` is a *capture* device-pixel-ratio and must stay pinned at 2.0
regardless of delivery tier — the video compiler's ken_burns motion crops INTO
the slide, so extra source pixels are real delivered detail even for a 1080p
deliverable.

These tests assert the actual payload each call site sends the slide_service
client — not just the constant's value — because the bug was in the three call
sites overriding the constant, not in the constant itself.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.services import tts_storage
from src.services.stages.shared.motion_slides import SLIDE_CAPTURE_SCALE


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


def test_slide_capture_scale_constant_is_flat_two() -> None:
    """Sanity: the constant itself, not 4/3 (which yields an odd 2559px width)."""
    assert SLIDE_CAPTURE_SCALE == 2.0


def _slide_client(png_count: int = 2) -> AsyncMock:
    fake = AsyncMock()
    fake.resolve_template_id.return_value = "tpl-feature"
    fake.start_deck.return_value = "deck-1"
    fake.wait_for_deck.return_value = {"status": "completed", "slide_count": png_count}
    fake.get_slide_png.side_effect = [b"\x89PNG" + bytes([i]) for i in range(png_count)]
    return fake
