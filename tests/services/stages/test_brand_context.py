"""Tests for best-effort brand extraction (src/services/stages/brand_context.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.clients.schemas.brand_extractor import BrandExtraction
from src.services.stages.brand_context import resolve_brand_context


def _client(resp: BrandExtraction) -> AsyncMock:
    fake = AsyncMock()
    fake.extract_brand.return_value = resp
    return fake


@pytest.mark.asyncio
async def test_maps_real_palette_fonts_logo() -> None:
    brand, site_analysis = await resolve_brand_context(
        "https://acme.test",
        client=_client(
            BrandExtraction.model_validate(
                {
                    "name": "Acme",
                    "primary_color": "#0a1f44",
                    "secondary_color": "#e8b923",
                    "font_family_primary": "Inter",
                    "font_family_secondary": "Lora",
                    "logo_url": "https://cdn.acme.test/logo.svg",
                    "brand_voice": "confident",
                }
            )
        ),
    )
    assert brand == {
        "primary_color": "#0a1f44",
        "secondary_color": "#e8b923",
        "name": "Acme",
        "font_family": "Inter",
        "display_font": "Inter",
        "body_font": "Lora",
        "logo_url": "https://cdn.acme.test/logo.svg",
    }
    assert site_analysis == {
        "tone": "confident",
        "visual_identity": {"primary_color": "#0a1f44", "secondary_color": "#e8b923"},
    }


@pytest.mark.asyncio
async def test_drops_default_blue_sentinels() -> None:
    # Extractor couldn't detect a real palette → returns the #2563EB/#1E40AF defaults.
    # We must NOT forward them (that would make html_render print the generic blue).
    brand, _site = await resolve_brand_context(
        "https://acme.test",
        client=_client(
            BrandExtraction.model_validate(
                {"name": "Acme", "primary_color": "#2563EB", "secondary_color": "#1E40AF"}
            )
        ),
    )
    # Name-only carries no colour/font signal → no usable brand dict.
    assert brand is None


@pytest.mark.asyncio
async def test_font_only_signal_is_kept() -> None:
    brand, _site = await resolve_brand_context(
        "https://acme.test",
        client=_client(
            BrandExtraction.model_validate(
                {"primary_color": "#2563EB", "font_family_primary": "Georgia"}
            )
        ),
    )
    assert brand == {"font_family": "Georgia", "display_font": "Georgia"}


@pytest.mark.asyncio
async def test_returns_none_on_extractor_error() -> None:
    fake = AsyncMock()
    fake.extract_brand.side_effect = RuntimeError("upstream 502")
    brand, site_analysis = await resolve_brand_context("https://acme.test", client=fake)
    assert brand is None
    assert site_analysis is None


@pytest.mark.asyncio
async def test_returns_none_for_empty_url() -> None:
    fake = AsyncMock()
    brand, site_analysis = await resolve_brand_context("", client=fake)
    assert (brand, site_analysis) == (None, None)
    fake.extract_brand.assert_not_called()
