"""Shared slide-deck helpers — the deterministic narration → slide projection.

Lifted out of ``promo_slides`` so any stage that renders authored shotlist slides
(``promo_slides`` for the whole promo deck, ``website_review_slides`` for the
authored BEFORE-act slides) builds the same deck-entry shapes on the same seeded
slide_service templates. The content is a deterministic projection of the
narration / authored shot — no extra LLM.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from src.services.stages.shared.shotlist import ShotSpec
from src.services.tts_storage import (
    deterministic_media_id,
    find_existing_media,
    store_slide_image,
)

if TYPE_CHECKING:
    from src.clients.slide_service_client import SlideServiceClient

# Interior/opening lines render on the seeded ``title`` template; the closing
# line renders on the ``cta`` template so a URL lands in slide_service's
# url-safe CTA layout (overflow-wrap, never a mid-word break). slide_service
# rasterises HTML/CSS with Chromium. Fallback titles use compact display copy;
# the original narration stays in the script and voiceover.
_TITLE_TEMPLATE_NAME = "title"
_CTA_TEMPLATE_NAME = "cta"
# Authored bullets slides render on the seeded ``feature_spotlight`` template —
# the same title+bullets layout lesson_record uses (proven content schema).
_FEATURE_TEMPLATE_NAME = "feature_spotlight"
_DECK_TIMEOUT_SECONDS = 600.0

# Common Practice palette. Passed in full so both the HTML preview and the pptx
# path are on-brand. Per-project ``brand`` keys override these at render time.
_BRAND_DEFAULTS: dict[str, str] = {
    "primary_color": "#0B1F3A",  # navy
    "secondary_color": "#2BB6A3",  # teal
    "accent_color": "#F26B3A",  # orange
    "font_family": "Inter",
}


def display_headline(text: str, fallback: str = "") -> str:
    """A compact excerpt for the screen; narration is preserved separately."""
    first = next((line.strip() for line in text.splitlines() if line.strip()), fallback)
    first = re.split(r"(?<=[.!?])\s+", first, maxsplit=1)[0]
    words = first.split()
    headline = " ".join(words[:10])
    if len(headline) > 90:
        headline = headline[:90].rsplit(" ", 1)[0]
    if len(words) > 10 or len(first) > 90:
        headline = headline.rstrip(".,;: ") + "…"
    return headline or fallback


def _slide_content(text: str, anchor: str) -> dict[str, object]:
    """Display a short headline rather than a whole spoken paragraph."""
    return {"eyebrow": anchor, "title": display_headline(text, anchor), "subtitle": ""}


def _shotlist_deck_slide(
    shot: ShotSpec | None,
    narration: str,
    anchor: str,
    *,
    title_id: str,
    cta_id: str,
    feature_id: str,
) -> dict[str, object]:
    """Build one deck entry for an authored shot.

    URL shots (and any line with no matching shot) get the title card as their
    universal fallback — if capture succeeds the render shows the real screen and
    this is never seen; if it fails, the narration still has a slide. Slide shots
    render their explicit content on the requested template (``title``/``cta``/
    ``bullets``→feature_spotlight), falling back to the narration for any field
    the operator left blank.
    """
    if shot is None or shot.kind != "slide":
        return {
            "template_id": title_id,
            "role": "content",
            "content": _slide_content(narration, anchor),
        }
    slide = shot.slide
    template = str(slide.get("template") or "title")
    title = str(slide.get("title") or display_headline(shot.narration, anchor))
    eyebrow = str(slide.get("eyebrow") or anchor)
    if template == "cta":
        content: dict[str, object] = {"eyebrow": eyebrow, "title": title}
        if slide.get("cta_url"):
            content["cta_url"] = str(slide["cta_url"])
        return {"template_id": cta_id, "role": "cta", "content": content}
    if template == "bullets" and feature_id:
        bullets = [str(b) for b in (slide.get("bullets") or [])] or [shot.narration]
        return {
            "template_id": feature_id,
            "role": "content",
            "content": {"eyebrow": eyebrow, "title": title, "bullets": bullets},
        }
    return {
        "template_id": title_id,
        "role": "content",
        "content": {"eyebrow": eyebrow, "title": title, "subtitle": str(slide.get("body") or "")},
    }


async def render_deck_slide_images(
    client: SlideServiceClient,
    *,
    deck_payload: dict[str, Any],
    run_id: str,
    stage_name: str,
    count: int,
    token_fp: str | None = None,
    tokens: dict[str, str] | None = None,
) -> list[str]:
    """Render a slide deck to per-index PNG URLs, idempotently.

    Content-addresses each slide PNG by ``(stage, deck slide content, brand,
    render_scale, index)`` so a resumed/retried run reuses already-stored slides
    instead of re-minting a random uuid per PNG. When EVERY slide is already
    present the whole ``start_deck`` render is skipped; on a partial hit the deck
    is rendered once (needed for the missing slides) but present slides are reused
    and not re-stored. Replaces the per-stage ``uuid7()`` loop the four slide
    stages (promo/lesson/website_review/demo_video) duplicated.

    ``token_fp`` (the C23 design/motion fingerprint) is folded into each media id
    ONLY when present; it is ``None`` for a run with no ``design``/``motion`` block,
    so conditional omission keeps those media ids byte-identical to today.

    ``tokens`` (the resolved C23 design-token bundle) is forwarded to slide_service
    on the deck submit, again only when present. It is NOT folded into the media id
    separately: it is a pure function of the same ``design`` block ``token_fp``
    already covers, so a token change moves the fingerprint and the ids follow.
    """
    slides = list(deck_payload.get("slides") or [])
    brand_key = json.dumps(deck_payload.get("brand"), sort_keys=True, default=str)
    scale = deck_payload.get("render_scale")
    media_ids = [
        deterministic_media_id(
            stage_name,
            json.dumps(slides[i] if i < len(slides) else None, sort_keys=True, default=str),
            brand_key,
            scale,
            i,
            *((token_fp,) if token_fp else ()),
        )
        for i in range(count)
    ]
    existing = [await find_existing_media(run_id, stage_name, mid, "png") for mid in media_ids]
    if all(url is not None for url in existing):
        # Full reuse — skip the deck render entirely.
        return [url for url in existing if url is not None]

    job_id = await client.start_deck(deck_payload, tokens=tokens)
    await client.wait_for_deck(job_id, timeout=_DECK_TIMEOUT_SECONDS)
    urls: list[str] = []
    for index in range(count):
        hit = existing[index]
        if hit is not None:
            urls.append(hit)
            continue
        png_bytes = await client.get_slide_png(job_id, index)
        urls.append(
            await store_slide_image(
                run_id=run_id,
                stage_name=stage_name,
                image_id=media_ids[index],
                image_bytes=png_bytes,
                image_format="png",
            )
        )
    return urls
