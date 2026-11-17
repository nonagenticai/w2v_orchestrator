"""Best-effort brand extraction for redesign stages.

The redesign stages (``website_redesign``, ``promo_html``) historically passed no
brand data to html_render_service, so every redesign fell back to a hardcoded blue.
This helper extracts the source site's real palette / fonts / logo via the existing
``BrandExtractorClient`` (analysis_service ``POST /api/brand/from-url``) and maps it into
the ``brand`` / ``site_analysis`` shapes html_render_service already consumes.

It is strictly **best-effort**: any failure, timeout, or low-signal result returns
``(None, None)`` so the caller proceeds exactly as before. It must never raise.
"""

from __future__ import annotations

import logging
from typing import Any

from src.clients.brand_extractor_client import BrandExtractorClient
from src.clients.schemas.brand_extractor import BrandExtraction

logger = logging.getLogger(__name__)

# The brand extractor returns these sentinels when it could NOT detect a real palette.
# Passing them downstream would make html_render treat the palette as "known" and print
# the generic blue — exactly the bug we're fixing — so we drop colours equal to them.
_DEFAULT_PRIMARY = "#2563EB"
_DEFAULT_SECONDARY = "#1E40AF"

# Generous cap: the extraction crawls a page (+ optional LLM). It is cached, and the
# redesign stages already wait minutes, so a slow extraction simply falls back.
_EXTRACT_TIMEOUT_SECONDS = 90.0


def _clean(value: object) -> str | None:
    """Return a non-empty stripped string, else None."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _map_brand(resp: BrandExtraction) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Map a BrandExtraction into (brand, site_analysis), dropping empty/default fields."""
    brand: dict[str, str] = {}

    primary = _clean(resp.primary_color)
    if primary and primary != _DEFAULT_PRIMARY:
        brand["primary_color"] = primary
    secondary = _clean(resp.secondary_color)
    if secondary and secondary != _DEFAULT_SECONDARY:
        brand["secondary_color"] = secondary

    if name := _clean(resp.name):
        brand["name"] = name  # also feeds the ORCH-0.7 asset-rights ledger (brand.name row)
    # BrandResponse exposes font_family_primary / font_family_secondary. html_render now
    # reloads a real display+body pairing (Google Fonts) instead of system-ui, so forward
    # both: font_family (back-compat / display fallback) + display_font + body_font.
    if font := _clean(resp.font_family_primary):
        brand["font_family"] = font
        brand["display_font"] = font
    if body := _clean(resp.font_family_secondary):
        brand["body_font"] = body
    if logo := _clean(resp.logo_url):
        brand["logo_url"] = logo  # also feeds the ORCH-0.7 asset-rights ledger (brand.logo row)

    site_analysis: dict[str, Any] = {}
    if tone := _clean(resp.brand_voice):
        site_analysis["tone"] = tone
    # Mirror the real palette into visual_identity too — html_render reads either source.
    visual_identity = {k: brand[k] for k in ("primary_color", "secondary_color") if k in brand}
    if visual_identity:
        site_analysis["visual_identity"] = visual_identity

    # Only return a brand dict if it carries something html_render can actually use
    # beyond a name (a name alone won't change colours/imagery).
    has_signal = any(k in brand for k in ("primary_color", "secondary_color", "font_family"))
    return (brand or None if has_signal else None, site_analysis or None)


async def resolve_brand_context(
    url: str,
    *,
    client: BrandExtractorClient | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Best-effort: extract the site's brand and return (brand, site_analysis).

    Returns ``(None, None)`` on any failure or low-signal result. Never raises.
    """
    if not url:
        return None, None
    extractor = client or BrandExtractorClient(timeout=_EXTRACT_TIMEOUT_SECONDS)
    try:
        resp = await extractor.extract_brand(url, download_logo=False)
    except Exception as exc:  # noqa: BLE001 — best-effort enhancement, must never abort the stage
        logger.warning("brand extraction skipped for %s: %s", url, exc)
        return None, None
    brand, site_analysis = _map_brand(resp)
    if brand or site_analysis:
        logger.info("brand extracted for %s: brand=%s", url, brand)
    return brand, site_analysis
