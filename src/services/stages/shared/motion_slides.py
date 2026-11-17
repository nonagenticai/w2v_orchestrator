"""Shared animated-slide rendering.

Projects a rendered slide deck onto motion templates and renders a branded clip
per slide via video_production_service. Used by ``promo_slides`` and
``website_review_slides`` — each stage owns its slide model, so this returns a
``{index: video_url}`` map (index-aligned to the deck) and the stage attaches the
clip to its own ``motion_video_url`` field. Per-clip soft-fail: a slide whose clip
fails is simply absent from the map and the render falls back to its still PNG.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.services.stages.shared.design_system import (
    _DARK_TEXT,
    _LIGHT_TEXT,
    _MIN_MUTED_CONTRAST,
    _MIN_TEXT_CONTRAST,
    _contrast_ratio,
    derive_tokens,
    token_fingerprint,
)
from src.services.stages.shared.motion_plan import (
    MotionPlan,
    MotionSignature,
    Timeline,
    motion_signature,
    plan_card_timing,
    stable_seed,
)
from src.services.stages.shared.slide_deck import _BRAND_DEFAULTS
from src.services.tts_storage import (
    deterministic_media_id,
    find_existing_media,
    store_generated_video,
)

if TYPE_CHECKING:
    from src.clients.video_production_client import VideoProductionClient

log = logging.getLogger(__name__)

# Render tuning.
_MOTION_CONCURRENCY = 2
_MOTION_RECORD_TIMEOUT_S = 600.0
_MOTION_PER_ITEM_TIMEOUT_S = 900.0
# Overall wall-clock budget for the whole motion-render phase. The phase runs
# INSIDE a stage with its own timeout (promo_slides = 900s, website_review_slides
# = 600s); a deck with many slides on a throughput-limited renderer used to blow
# that budget and HARD-FAIL the entire stage — the ~9-12 motion-slide cap that
# forced SECTION mode as a workaround. We now bound the phase to this budget,
# which MUST stay below the caller's stage timeout (with headroom for the stills
# deck): clips that finish in time get their motion video; the rest gracefully
# keep their still PNG (the same per-clip soft-fail the caller already handles).
# The stage always completes — no hard cap; the motion-clip count just scales
# with render throughput. Each caller passes a stage-appropriate `phase_budget_s`;
# this conservative default stays under the smallest (600s) stage. Env-tunable.
_MOTION_PHASE_BUDGET_S = float(os.getenv("MOTION_PHASE_BUDGET_S", "450"))
# Animated-slide template ids (registered in video_production_service).
_MOTION_SLIDE_TEMPLATE = "animated_slide"
_MOTION_TITLE_TEMPLATE = "animated_title_card"
# Deliverable format → motion clip dimensions (the still PNG is already on-brand;
# the motion clip just needs to fill the same frame). Default to 16:9.
_FORMAT_DIMS: dict[str, tuple[int, int]] = {
    "16x9": (1920, 1080),
    "9x16": (1080, 1920),
    "1x1": (1080, 1080),
    "4x5": (1080, 1350),
}


# Output resolution tier → linear scale applied to the base 1080p dims. The
# whole pipeline (motion clips, slide PNGs, the terminal compile canvas) is
# raised together by this one factor so a final video is sharp end-to-end, not
# an upscale of 1080p sources. 2K uses 2560/1920 so 16x9 → 2560x1440 exactly.
_TIER_SCALE: dict[str, float] = {
    "1080p": 1.0,
    "2k": 2560 / 1920,
    "4k": 2.0,
}
_DEFAULT_TIER = "2k"

# Slide-capture device-pixel-ratio sent to slide_service as `render_scale`.
# This is DELIBERATELY separate from `tier_scale` above: `tier_scale` sizes the
# *delivered* dimensions (motion clip / compile canvas) per resolution tier, but
# `render_scale` controls the Chromium capture DPR — how many source pixels we
# ask slide_service to rasterize before any downscale. Even a 1080p deliverable
# wants a 2x capture: the video compiler's ken_burns motion crops INTO the slide,
# so the extra source pixels are real delivered detail, not waste. Do not send
# `tier_scale("1080p")` (1.0) here — that was the bug (slide_service #128 raised
# its own render default to 2.0, but this repo's explicit render_scale overrode
# it back down to 1x on every 1080p run, which is most runs).
#
# Kept flat at 2.0 for every tier rather than matching tier_scale's 1.333 (2k)
# because 1.333 * 1920 = 2559.36 -> an ODD 2559px width, hostile to yuv420p
# encoders downstream. A flat 2.0 means 4k delivery gets an exact 3840x2160
# capture with no upscale at all.
SLIDE_CAPTURE_SCALE = 2.0


def quality_tier(config_snapshot: dict[str, Any] | None) -> str:
    """Resolution tier for this run: one of ``1080p`` | ``2k`` | ``4k``.

    Read off the per-run ``config_snapshot`` (set in ``run_setup`` from the
    request override or the project brief). Unknown/missing → the default tier
    so a run never silently drops to 1080p by typo.
    """
    raw = str((config_snapshot or {}).get("resolution") or "").strip().lower()
    return raw if raw in _TIER_SCALE else _DEFAULT_TIER


def tier_scale(tier: str) -> float:
    """Linear pixel-scale for a resolution tier (1.0 / 1.333 / 2.0).

    Used as the slide-service ``render_scale`` (Chromium device pixel ratio) so
    slide PNGs match the motion clips' resolution for the same tier.
    """
    return _TIER_SCALE.get(str(tier or "1080p").lower(), 1.0)


def _even(n: float) -> int:
    """Round to the nearest even int — yuv420p/H.264 require even W/H."""
    return round(n / 2) * 2


def format_dims(fmt: str | None, tier: str = "1080p") -> tuple[int, int]:
    """Deliverable ``format`` + resolution ``tier`` → motion clip ``(w, h)``.

    Single source of the format→dims mapping for every motion renderer (this
    module and ``title_cards``), so an intro card and a body slide always fill
    the same frame. ``tier`` scales the base 1080p dims up to 2K/4K; both
    dimensions are forced even for the encoder.
    """
    base_w, base_h = _FORMAT_DIMS.get(str(fmt or "16x9"), _FORMAT_DIMS["16x9"])
    scale = _TIER_SCALE.get(str(tier or "1080p").lower(), 1.0)
    if scale == 1.0:
        return base_w, base_h
    return _even(base_w * scale), _even(base_h * scale)


def film_frame_dims(config_snapshot: Mapping[str, object] | None) -> tuple[int, int]:
    """The run's output frame ``(w, h)``: its deliverable format at its quality tier."""
    snapshot = dict(config_snapshot or {})
    deliverable = snapshot.get("deliverable")
    fmt = deliverable.get("format") if isinstance(deliverable, Mapping) else None
    return format_dims(str(fmt) if fmt else None, quality_tier(snapshot))


def motion_brand_params(deck_brand: dict[str, object]) -> dict[str, str]:
    """Map brand tokens onto the motion templates' accent + brand-mark params.

    The accent colours are always mapped; the canvas/text surface is left to the
    template default here (the templates keep their dark canvas + light text so
    contrast is guaranteed regardless of a brand's primary colour, and the still
    PNG carries the full brand palette via slide_service). For a contrast-checked
    full-surface override, see :func:`theme_motion_params`.

    The brand ``name`` (when set) is carried as ``brand_mark`` so the title card
    shows the tenant's mark instead of the template's hardcoded ``nonagentic.ai``
    default. When the brand has no name the param is omitted, leaving the template
    default untouched (no global behaviour change). ``animated_slide`` ignores the
    key, so it is harmless on bullet cards."""
    params: dict[str, str] = {}
    if deck_brand.get("secondary_color"):
        params["accent"] = str(deck_brand["secondary_color"])
    if deck_brand.get("accent_color"):
        params["accent2"] = str(deck_brand["accent_color"])
    if deck_brand.get("name"):
        params["brand_mark"] = str(deck_brand["name"])
    return params


# The WCAG contrast thresholds + helpers (``_MIN_TEXT_CONTRAST`` /
# ``_MIN_MUTED_CONTRAST`` / ``_LIGHT_TEXT`` / ``_DARK_TEXT`` / ``_contrast_ratio``)
# and the light/dark text defaults now live in ``design_system`` — their canonical
# home — and are RE-IMPORTED above (byte-identical: they were lifted verbatim).
# ``_hex_to_rgb`` / ``_relative_luminance`` are used transitively via
# ``_contrast_ratio`` and no longer referenced directly here.


def _resolve_surface_bg(
    deck_brand: dict[str, object], deck_theme: dict[str, object] | None
) -> str | None:
    """First usable background colour for the motion card surface.

    Tries the theme's background keys in priority order, then falls back to the
    brand's ``primary_color`` so a brand-only project (no theme) still gets an
    on-brand card. Returns ``None`` when nothing usable is present."""
    theme = deck_theme or {}
    candidates = (
        theme.get("gradient_start"),
        theme.get("background_color"),
        theme.get("background"),  # the key app_video_docs generate.py emits
        deck_brand.get("primary_color"),  # brand fallback
    )
    for cand in candidates:
        if isinstance(cand, str) and cand:
            return cand
    return None


def _resolve_surface_text(deck_theme: dict[str, object] | None, bg: str) -> str | None:
    """Contrast-safe text colour for ``bg``.

    Uses the theme's explicit ``text_color`` when it clears the body-text
    threshold; otherwise auto-picks the higher-contrast of the light/dark template
    defaults. Returns ``None`` only when even the better default can't clear the
    threshold (e.g. ``bg`` is unparseable), so the caller falls back to the
    template's own surface and the "never unreadable" invariant holds."""
    text = (deck_theme or {}).get("text_color")
    if isinstance(text, str) and text:
        ratio = _contrast_ratio(text, bg)
        if ratio is not None and ratio >= _MIN_TEXT_CONTRAST:
            return text
    light = _contrast_ratio(_LIGHT_TEXT, bg)
    dark = _contrast_ratio(_DARK_TEXT, bg)
    if light is None or dark is None:
        return None
    best, ratio = (_LIGHT_TEXT, light) if light >= dark else (_DARK_TEXT, dark)
    return best if ratio >= _MIN_TEXT_CONTRAST else None


def theme_motion_params(
    deck_brand: dict[str, object], deck_theme: dict[str, object] | None = None
) -> dict[str, str]:
    """Brand accents + brand mark (always) plus a contrast-checked branded surface.

    Extends :func:`motion_brand_params` with full ``bg`` / ``bg2`` / ``text``
    overrides whenever a usable background can be resolved from the theme (any of
    ``gradient_start`` / ``background_color`` / ``background``) **or**, failing
    that, the brand's ``primary_color`` — and a contrast-safe text colour can be
    paired with it (the theme's ``text_color`` if it clears WCAG AA 4.5:1, else the
    higher-contrast light/dark template default). When no usable background exists,
    or even the best text default can't clear the threshold, the surface override
    is skipped and the template keeps its built-in dark canvas + light text. This
    preserves the "contrast guaranteed — a brand can never produce a white-on-white
    card" invariant while letting any tenant's background drive the surface. ``bg2``
    comes from ``gradient_end`` (else ``bg``); ``muted`` from ``subtitle_color``
    only when it clears the large-text threshold against ``bg``."""
    params = motion_brand_params(deck_brand)
    bg = _resolve_surface_bg(deck_brand, deck_theme)
    if bg is None:
        return params
    text = _resolve_surface_text(deck_theme, bg)
    if text is None:
        return params
    params["bg"] = bg
    params["text"] = text
    bg2 = (deck_theme or {}).get("gradient_end")
    params["bg2"] = str(bg2) if isinstance(bg2, str) and bg2 else bg
    muted = (deck_theme or {}).get("subtitle_color")
    if isinstance(muted, str) and muted:
        muted_ratio = _contrast_ratio(muted, bg)
        if muted_ratio is not None and muted_ratio >= _MIN_MUTED_CONTRAST:
            params["muted"] = muted
    return params


# The recorder embeds Fraunces and IBM Plex Sans, and (video_production_service#188)
# Inter 400/600/700 for the film templates. The default brand font is Inter
# (``_BRAND_DEFAULTS``), so the stack names 'Inter' first unless the brand names
# another family; a family the recorder does not have falls through to the
# embedded sans, never to a system serif.
_MOTION_FONT_FALLBACK = "'IBM Plex Sans',-apple-system,sans-serif"
_FONT_NAME_CHARS = re.compile(r"[^\w \-]")


def deck_motion_params(
    deck_brand: dict[str, object], deck_theme: dict[str, object] | None = None
) -> dict[str, str]:
    """:func:`theme_motion_params` plus the deck's font — every motion surface's look.

    ``font_display`` / ``font_sans`` both carry the deck brand's ``font_family``
    (the family slide_service sets the still slides in), so a motion card's
    headline is no longer the templates' serif next to a sans slide (frame QA of
    run 01a11082: "two unrelated design systems"). Templates that do not read
    the keys (``data_viz``, ``kinetic_typography``) ignore them.
    """
    params = theme_motion_params(deck_brand, deck_theme)
    family = _FONT_NAME_CHARS.sub("", str(deck_brand.get("font_family") or "")).strip()
    stack = f"'{family}',{_MOTION_FONT_FALLBACK}" if family else _MOTION_FONT_FALLBACK
    params["font_display"] = stack
    params["font_sans"] = stack
    return params


#: ``direction.motion`` value that opts a promo in to one designed card per line.
PER_LINE = "per_line"


def motion_per_line_requested(config_snapshot: Mapping[str, object]) -> bool:
    """Whether the run opted in to per-line motion (``direction.motion == "per_line"``).

    THE gate of everything the per-line motion work changes: a run without it
    renders exactly as it did before that work (same card params, same image
    prompts, same local Ken Burns move).
    """
    direction = config_snapshot.get("direction")
    return isinstance(direction, Mapping) and direction.get("motion") == PER_LINE


def run_motion_brand_params(config_snapshot: Mapping[str, object]) -> dict[str, str]:
    """THE brand params of a run's motion surfaces: cards, animated slides, end card.

    One resolution for all three, from the same deck brand the slide stages
    build (``_BRAND_DEFAULTS`` under the project's ``brand``) and the run's
    ``theme``. With ``theme == {}`` and a name-only brand every surface gets the
    same default palette and font instead of three different ones.
    """
    brand = config_snapshot.get("brand")
    theme = config_snapshot.get("theme")
    deck_brand: dict[str, object] = {
        **_BRAND_DEFAULTS,
        **(brand if isinstance(brand, dict) else {}),
    }
    return deck_motion_params(deck_brand, theme if isinstance(theme, dict) else None)


def motion_for_deck_slide(
    deck_slide: dict[str, Any], brand_params: dict[str, str]
) -> tuple[str, dict[str, Any]]:
    """Project a deck entry onto a motion template id + params.

    A bullets slide animates on ``animated_slide`` (kinetic title + build-on
    bullets); every other content/cta card animates on ``animated_title_card``
    (eyebrow + title + subtitle, the CTA url as the subtitle). The content dict
    is the same deterministic projection the still PNG uses, so the animated and
    still versions of a line always say the same thing."""
    content = deck_slide.get("content") or {}
    eyebrow = str(content.get("eyebrow") or "")
    title = str(content.get("title") or "")
    bullets = content.get("bullets")
    if bullets:
        return _MOTION_SLIDE_TEMPLATE, {
            "kicker": eyebrow,
            "title": title,
            "bullets": [str(b) for b in bullets],
            **brand_params,
        }
    subtitle = str(content.get("subtitle") or content.get("cta_url") or "")
    return _MOTION_TITLE_TEMPLATE, {
        "eyebrow": eyebrow,
        "title": title,
        "subtitle": subtitle,
        **brand_params,
    }


def _run_motion_plan(config_snapshot: dict[str, Any], motion_block: dict[str, Any]) -> MotionPlan:
    """A run-level representative :class:`MotionPlan` for the fingerprint seam.

    The plan hashed into :func:`token_fingerprint` only needs to change iff the
    resolved motion intent (director ``style`` + the per-video seed) changes, so a
    single run-level signature suffices — per-card timing is applied separately to
    the actual render params. Pure/deterministic (seed = ``sha256(project_id)``)."""
    style = str(motion_block.get("style") or "")
    seed = stable_seed(str(config_snapshot.get("project_id") or ""))
    sig = motion_signature(style, seed)
    return MotionPlan(
        signature=sig, layout="title", icon=None, timeline=plan_card_timing(0, 0, sig)
    )


def token_fp_for_snapshot(config_snapshot: dict[str, Any] | None) -> str | None:
    """Resolve the C23 design+motion media-id fingerprint for a run, or ``None``.

    The single seam every deck/ken-burns media id folds on. Returns ``None`` iff
    NEITHER a ``design`` nor a ``motion`` author-block is present in the run's
    ``config_snapshot`` — the inert / byte-identical path. Only a present block
    builds the derived tokens / motion plan it hashes, so the fingerprint (and
    therefore the media id) moves iff the resolved tokens change (idempotency)."""
    cfg = config_snapshot or {}
    # A malformed (non-dict) block from unvalidated ``config_overrides`` is treated
    # as absent — degrade to the inert path, never crash the run.
    design_block = cfg.get("design") if isinstance(cfg.get("design"), dict) else None
    motion_block = cfg.get("motion") if isinstance(cfg.get("motion"), dict) else None
    if not design_block and not motion_block:
        return None
    tokens = (
        derive_tokens(cfg.get("brand"), cfg.get("theme"), design_block) if design_block else None
    )
    plan = _run_motion_plan(cfg, motion_block) if motion_block else None
    return token_fingerprint(design_block, motion_block, tokens, plan)


def _resolve_run_motion(
    config_snapshot: dict[str, Any] | None,
) -> tuple[str | None, MotionSignature | None]:
    """Resolve the run's ``(token_fp, motion_signature)`` pair for the render loop.

    ``token_fp`` is the C23 media-id fingerprint (``None`` → inert/byte-identical);
    ``motion_signature`` is non-``None`` only when a ``motion`` author-block is
    present, so an un-opted run leaves the render params untouched."""
    cfg = config_snapshot or {}
    motion_block = cfg.get("motion") if isinstance(cfg.get("motion"), dict) else None
    motion_sig = _run_motion_plan(cfg, motion_block).signature if motion_block else None
    return token_fp_for_snapshot(config_snapshot), motion_sig


def _motion_signature_params(sig: MotionSignature, timeline: Timeline) -> dict[str, str]:
    """Flatten a motion signature + per-card timeline into template render params.

    Only merged into a clip's params when a ``motion`` block is present, so an
    un-opted run's params (and its media id) are byte-identical to today. When
    present, ``video_production_service``'s motion-graphics templates consume
    these through the shared ``window.MotionLib`` (VP-5): ``easing`` shapes the
    entrance/exit curve, ``stagger_ms`` the per-element rhythm,
    ``entrance_ms``/``hold_ms``/``exit_ms``/``marks`` the clip timeline (so a card
    animates entrance → hold → exit instead of freezing mid-clip), and
    ``parallax``/``glow_drift`` the depth motion. Each template gates every value
    on its own key, so a partial signature — and any key a given template does
    not consume — is harmless."""
    return {
        "easing": sig.easing,
        "stagger_ms": str(sig.stagger_ms),
        "entrance": sig.entrance,
        "exit": sig.exit_,
        "parallax": repr(sig.parallax),
        "glow_drift": repr(sig.glow_drift),
        "entrance_ms": str(timeline.entrance_ms),
        "hold_ms": str(timeline.hold_ms),
        "exit_ms": str(timeline.exit_ms),
        "marks": json.dumps(list(timeline.marks)),
    }


def _apply_card_motion(
    params: dict[str, Any], duration_ms: int | None, motion_sig: MotionSignature | None
) -> dict[str, Any]:
    """Merge the run's motion signature + this card's timeline into render params.

    A no-op returning ``params`` unchanged when ``motion_sig`` is ``None`` (no
    ``motion`` author-block) → the render params and media id stay byte-identical."""
    if motion_sig is None:
        return params
    n_elements = len(params.get("bullets") or []) or 1
    timeline = plan_card_timing(duration_ms or 0, n_elements, motion_sig)
    return {**params, **_motion_signature_params(motion_sig, timeline)}


@dataclass(frozen=True)
class DeckMotionResult:
    """Outcome of a motion-render phase — the clip urls plus WHY any are missing.

    ``urls`` maps deck index → rendered clip url (a slide absent from the map keeps
    its still PNG). ``budget_exceeded`` is ``True`` when the phase hit its wall-clock
    ``phase_budget_s`` and in-flight clips were cancelled; ``failed_indices`` lists
    every deck index that never produced a url (budget-cancelled OR per-clip
    soft-failed). Callers that carry a ``notes`` channel surface these so a dropped
    motion clip degrades the run (diagnosable) instead of shipping a silent ``None``.
    """

    urls: dict[int, str]
    budget_exceeded: bool
    failed_indices: tuple[int, ...]


async def render_deck_motion_clips(
    video_client: VideoProductionClient,
    *,
    line_ids: list[str],
    deck_slides: list[dict[str, Any]],
    deck_brand: dict[str, object],
    motion_cfg: dict[str, Any],
    deliverable: dict[str, Any],
    run_id: str,
    stage_name: str,
    deck_theme: dict[str, object] | None = None,
    phase_budget_s: float = _MOTION_PHASE_BUDGET_S,
    tier: str = "1080p",
    duration_ms_by_index: dict[int, int] | None = None,
    master_fps: int | None = None,
    config_snapshot: dict[str, Any] | None = None,
) -> DeckMotionResult:
    """Render a branded motion clip per deck slide; return a :class:`DeckMotionResult`.

    ``line_ids`` and ``deck_slides`` are index-aligned (one per rendered slide).
    Per-clip soft-fail: a slide whose clip fails is omitted from ``result.urls`` so
    the caller leaves its ``motion_video_url`` unset and the render keeps the PNG.
    ``result.budget_exceeded`` / ``result.failed_indices`` tell the caller WHY a
    slide is missing (phase-budget cancel vs per-clip soft-fail) so it can surface a
    typed degradation note instead of a silent ``None``.
    ``deck_theme`` (optional) lifts the clip's full bg/text surface to the project
    theme when it clears the WCAG contrast check; absent → accent-only as before.
    """
    brand_params = theme_motion_params(deck_brand, deck_theme)
    default_w, default_h = format_dims(deliverable.get("format"), tier)
    width = int(motion_cfg.get("width") or default_w)
    height = int(motion_cfg.get("height") or default_h)
    fps = int(motion_cfg.get("fps") or master_fps or 30)

    # C23 seam: the design+motion fingerprint (``None`` when no author block →
    # inert / byte-identical). When a ``motion`` block is present, resolve the
    # run-wide signature once so each card's timing is applied per its clip length.
    token_fp, motion_sig = _resolve_run_motion(config_snapshot)

    async def _gen(item: tuple[int, str]) -> tuple[int, str | None]:
        index, line_id = item
        if index >= len(deck_slides):
            return (index, None)
        template_id, params = motion_for_deck_slide(deck_slides[index], brand_params)
        # Per-card timeline length: when known (the caller passed the card's
        # narration duration), render the clip that long so its animated background
        # keeps moving the whole time it's on screen (no last-frame freeze). Unset →
        # the template's own RENDER_DURATION_MS default (unchanged behaviour).
        duration_ms = (duration_ms_by_index or {}).get(index)
        # Opt-in motion signature: when a ``motion`` block is present, apply the
        # run's signature + this card's timeline to the render params. A no-op
        # (params returned unchanged) when un-opted → byte-identical.
        params = _apply_card_motion(params, duration_ms, motion_sig)
        # Content-address the clip by everything that determines its pixels, so a
        # resumed/retried render reuses an already-recorded clip instead of
        # re-driving video_production_service for it. ``token_fp`` is folded ONLY
        # when present (design/motion opted in) → conditional omission keeps the
        # no-block media id byte-identical.
        media_id = deterministic_media_id(
            stage_name,
            template_id,
            json.dumps(params, sort_keys=True, default=str),
            width,
            height,
            fps,
            duration_ms,
            *((token_fp,) if token_fp else ()),
        )
        try:
            existing = await find_existing_media(run_id, stage_name, media_id, "mp4")
            if existing is not None:
                return (index, existing)
            job_id = await video_client.record_animation(
                template_id=template_id,
                params=params,
                part_id=f"slide-{line_id}",
                fps=fps,
                width=width,
                height=height,
                output_format="mp4",
                duration_ms=duration_ms,
            )
            await video_client.wait_for_content_recording(job_id, timeout=_MOTION_RECORD_TIMEOUT_S)
            video_bytes, _ctype, _disp = await video_client.get_recording_video(job_id)
            url = await store_generated_video(
                run_id=run_id,
                stage_name=stage_name,
                video_id=media_id,
                video_bytes=video_bytes,
                video_format="mp4",
            )
            return (index, url)
        except Exception as exc:  # noqa: BLE001 -- soft-fail: this slide keeps its PNG
            log.warning("%s: motion clip failed for line %s: %s", stage_name, line_id, exc)
            return (index, None)

    # Self-bounded concurrent render with graceful degradation (see
    # _MOTION_PHASE_BUDGET_S). Accumulate clips into `rendered` as they finish so
    # partial results survive a budget cancellation; on budget exceed, cancel the
    # in-flight/queued clips and leave their slides to fall back to the still PNG.
    semaphore = asyncio.Semaphore(_MOTION_CONCURRENCY)
    rendered: dict[int, str] = {}

    async def _run_one(item: tuple[int, str]) -> None:
        async with semaphore:
            try:
                # Per-clip hard cap (defence-in-depth; _gen also bounds its own
                # record wait). Only TimeoutError is swallowed here → that slide
                # keeps its still PNG; a phase-budget CancelledError propagates so
                # the task is cancelled cleanly (_gen's `except Exception` can't
                # swallow it — CancelledError is BaseException).
                index, url = await asyncio.wait_for(_gen(item), timeout=_MOTION_PER_ITEM_TIMEOUT_S)
            except TimeoutError:
                return  # this slide keeps its still PNG
            if url is not None:
                rendered[index] = url

    budget_exceeded = False
    tasks = [asyncio.create_task(_run_one((i, lid))) for i, lid in enumerate(line_ids)]
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=phase_budget_s)
    except TimeoutError:
        budget_exceeded = True
        log.warning(
            "%s: motion-render phase hit budget %.0fs — %d/%d clips rendered; "
            "remaining slides keep their still PNG",
            stage_name,
            phase_budget_s,
            len(rendered),
            len(line_ids),
        )
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    failed_indices = tuple(i for i in range(len(line_ids)) if i not in rendered)
    return DeckMotionResult(
        urls=dict(rendered),
        budget_exceeded=budget_exceeded,
        failed_indices=failed_indices,
    )
