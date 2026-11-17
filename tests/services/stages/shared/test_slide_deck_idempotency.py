"""Idempotency of the shared ``render_deck_slide_images`` helper.

Covers the reuse contract shared by all four slide stages (promo / lesson /
website_review / demo_video): a resumed run reuses already-stored slide PNGs and
skips the deck render entirely when every slide is present.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.services.stages.shared import slide_deck
from src.services.stages.shared.slide_deck import render_deck_slide_images

_DECK = {
    "title": "T",
    "brand": {"primary_color": "#000"},
    "render_scale": 2,
    "slides": [{"content": {"title": "a"}}, {"content": {"title": "b"}}],
}


def _client() -> AsyncMock:
    client = AsyncMock()
    client.start_deck = AsyncMock(return_value="deck-1")
    client.wait_for_deck = AsyncMock(return_value=None)
    client.get_slide_png = AsyncMock(return_value=b"png-bytes")
    return client


@pytest.mark.asyncio
async def test_renders_and_stores_when_nothing_exists() -> None:
    client = _client()
    with (
        patch.object(slide_deck, "find_existing_media", AsyncMock(return_value=None)),
        patch.object(
            slide_deck,
            "store_slide_image",
            AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['image_id']}.png"),
        ) as store,
    ):
        urls = await render_deck_slide_images(
            client, deck_payload=_DECK, run_id="run-1", stage_name="promo_slides", count=2
        )

    assert len(urls) == 2
    client.start_deck.assert_awaited_once()  # deck rendered
    assert store.await_count == 2  # both slides stored
    # Stored under a deterministic (content-addressed) id, not a random uuid.
    stored_ids = {call.kwargs["image_id"] for call in store.await_args_list}
    assert len(stored_ids) == 2


@pytest.mark.asyncio
async def test_full_reuse_skips_deck_render() -> None:
    client = _client()
    with (
        patch.object(
            slide_deck,
            "find_existing_media",
            AsyncMock(side_effect=lambda run_id, stage, mid, ext: f"https://minio/{mid}.{ext}"),
        ),
        patch.object(slide_deck, "store_slide_image", AsyncMock()) as store,
    ):
        urls = await render_deck_slide_images(
            client, deck_payload=_DECK, run_id="run-1", stage_name="promo_slides", count=2
        )

    assert len(urls) == 2
    assert all(u.endswith(".png") for u in urls)
    client.start_deck.assert_not_awaited()  # deck render skipped entirely
    client.get_slide_png.assert_not_awaited()
    store.assert_not_awaited()  # nothing re-stored


@pytest.mark.asyncio
async def test_partial_hit_renders_deck_but_reuses_present_slide() -> None:
    client = _client()
    # slide 0 already stored; slide 1 missing. The helper probes indices in order,
    # so return a hit for the first probe and a miss for the second.
    hits = ["https://minio/existing-0.png", None]
    probe_calls = iter(hits)

    async def _find(run_id, stage, media_id, ext):
        return next(probe_calls, None)

    with (
        patch.object(slide_deck, "find_existing_media", _find),
        patch.object(
            slide_deck,
            "store_slide_image",
            AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['image_id']}.png"),
        ) as store,
    ):
        urls = await render_deck_slide_images(
            client, deck_payload=_DECK, run_id="run-1", stage_name="promo_slides", count=2
        )

    assert urls[0] == "https://minio/existing-0.png"  # reused
    client.start_deck.assert_awaited_once()  # deck still rendered for slide 1
    assert store.await_count == 1  # only the missing slide stored
    client.get_slide_png.assert_awaited_once_with("deck-1", 1)
