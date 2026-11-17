"""media_discovery — find + rank real media (images/clips) for the content.

media-discovery (``POST /api/search``, called directly — KAG, which used to proxy
it, is retired) was built for w2v: it
federates across media sources and returns ranked items with ``thumbnail_url`` /
``quality_score``. This stage runs the *pipeline-auto* surface — during a run it
derives a query from the brief + ``promo_analyze`` insights, searches, keeps the
top-N, and records them as ``discovered_media`` so the visual stages (and the run
detail) have real candidate imagery alongside AI-gen + extracted images.

Opt-in: runs only when ``brief.media_discovery`` is present (a dict, possibly
empty ``{}``) so existing runs are unaffected. Soft-fail: any search error degrades
to zero discovered items + a note rather than failing the run.

The discovered items are persisted + visible in the run detail, AND the top
*image* items are re-published to run storage (``composite_url``) so
``promo_render`` can lay them over image-suitable lines (see
``src/services/visual_media.py``). The companion *operator-curated* surface is
the wizard's "Find media" panel, served by ``POST /api/media/search``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.clients.media_discovery_client import MediaDiscoveryClient, clamp_search_limit
from src.clients.schemas.media_discovery import MediaItem
from src.services.stages.base import (
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StageSkippedError,
)
from src.services.stages.shared.degradation import (
    MEDIA_DISCOVERY_INCOMPLETE,
    MEDIA_NOT_COMPOSITABLE,
    MEDIA_NOT_COMPOSITED,
    Degradation,
    advisory,
    degradation,
)
from src.services.stages.shared.motion_slides import film_frame_dims
from src.services.visual_media import (
    NotMediaContentError,
    republish_image,
    republish_video,
    video_page_host,
)

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

_SEARCH_TIMEOUT_S = 60.0
_DEFAULT_LIMIT = 12


def _brief_limit(raw: object) -> int:
    """The brief's ``media_discovery.limit`` as a search limit media-discovery accepts.

    Unset/zero or non-numeric falls back to the default; anything else is clamped
    to media-discovery's ``[1, 100]`` (above it the search is a 422).
    """
    if not raw or not isinstance(raw, int | float | str) or isinstance(raw, bool):
        return _DEFAULT_LIMIT
    try:
        return clamp_search_limit(int(raw))
    except ValueError:
        return _DEFAULT_LIMIT


_MAX_KEEP = 12
_QUERY_MAX_CHARS = 240
# Re-publishing fetches each image, so cap how many we composite per run.
_MAX_COMPOSITE = 8

#: ``DiscoveredMedia.composite_status`` values. ``not_compositable`` is a
#: CLASSIFICATION (the item is a web page, not a media file) and never degrades;
#: ``failed`` is a real media file that could not be re-published and does.
COMPOSITED = "composited"
NOT_COMPOSITABLE = "not_compositable"
COMPOSITE_FAILED = "failed"
NOT_ATTEMPTED = "not_attempted"


class MediaDiscoveryInput(StageInput):
    pass  # All inputs come from ctx.config_snapshot + prior outputs.


class DiscoveredMedia(StageOutput):
    id: str
    title: str = ""
    url: str = ""
    thumbnail_url: str = ""
    media_type: str = ""
    source: str = ""
    quality_score: float = 0.0
    # Compiler-fetchable URL after re-publishing the external image/video to run
    # storage; "" for unsupported items or when the re-publish failed (the item
    # is still recorded via ``url``/``thumbnail_url`` for the wizard/run detail).
    composite_url: str = ""
    #: Why ``composite_url`` is what it is — see the module constants. ``""`` on
    #: rows persisted before the field existed.
    composite_status: str = ""
    #: One sentence a human can read in the run detail ("YouTube watch page — not
    #: a downloadable clip; recorded for reference, not shown on screen").
    composite_detail: str = ""


class MediaDiscoveryOutput(StageOutputWithNotes):
    query: str = ""
    items: list[DiscoveredMedia] = []  # noqa: RUF012 -- pydantic per-instance default
    sources_queried: list[str] = []  # noqa: RUF012


class MediaDiscoveryStage(StageHandler[MediaDiscoveryInput, MediaDiscoveryOutput]):
    name = "media_discovery"
    input_model = MediaDiscoveryInput
    output_model = MediaDiscoveryOutput

    def __init__(self, client: MediaDiscoveryClient | None = None) -> None:
        self._client = client or MediaDiscoveryClient(timeout=_SEARCH_TIMEOUT_S)

    async def handle(
        self,
        payload: MediaDiscoveryInput,
        ctx: RunContext,
    ) -> MediaDiscoveryOutput:
        cfg = ctx.config_snapshot
        brief = cfg.get("brief") or {}
        md_cfg = brief.get("media_discovery")
        if not isinstance(md_cfg, dict):
            raise StageSkippedError(
                "media_discovery: no brief.media_discovery (opt-in stage not requested)",
                missing=["brief.media_discovery"],
            )

        query = self._build_query(cfg, brief, md_cfg, ctx)
        if not query:
            return MediaDiscoveryOutput(
                degradations=[
                    degradation(
                        MEDIA_DISCOVERY_INCOMPLETE,
                        "media_discovery: no search query could be derived from the brief "
                        "(no media_discovery.query, analyze hooks, value proposition or "
                        "title), so no media was searched",
                    )
                ]
            )

        source_filters = md_cfg.get("sources") if isinstance(md_cfg.get("sources"), list) else None
        limit = _brief_limit(md_cfg.get("limit"))
        degradations: list[Degradation] = []
        try:
            resp = await self._client.search(query, sources=source_filters, limit=limit)
        except Exception as exc:  # noqa: BLE001 -- soft-fail: degrade to zero items
            log.warning("media_discovery: search failed: %s", exc)
            return MediaDiscoveryOutput(
                query=query,
                degradations=[
                    degradation(
                        MEDIA_DISCOVERY_INCOMPLETE,
                        f"media_discovery: the media search failed ({exc}); no media was found",
                    )
                ],
            )

        items = self._to_items(resp.results)[:_MAX_KEEP]
        if resp.sources_failed:
            degradations.append(
                degradation(
                    MEDIA_DISCOVERY_INCOMPLETE,
                    "media_discovery: media source(s) unavailable during search: "
                    + ", ".join(resp.sources_failed),
                )
            )
        await self._republish_images(
            str(ctx.run_id), items, degradations, frame=film_frame_dims(ctx.config_snapshot)
        )
        return MediaDiscoveryOutput(
            query=query,
            items=items,
            sources_queried=resp.sources_queried,
            degradations=degradations,
            advisories=_not_compositable_advisories(items),
        )

    async def _republish_images(
        self,
        run_id: str,
        items: list[DiscoveredMedia],
        degradations: list[Degradation],
        *,
        frame: tuple[int, int] | None = None,
    ) -> None:
        """Re-publish the top image/video items to run storage so a render can use them.

        Every item leaves with a ``composite_status``:

        * ``not_compositable`` — the URL is not a media FILE: an explicit
          non-image/-video ``media_type``, a known video WATCH page (YouTube,
          Vimeo…) classified up front WITHOUT a request, or a URL that answered
          with a non-media content type. A classification, recorded on the item
          and never a degradation: prod run 01a09f82 reported a pinned YouTube
          result as a failure-looking ``not a video (text/html)`` note.
        * ``failed`` — a real media file that could not be re-published; the one
          case that raises :data:`MEDIA_NOT_COMPOSITED`.
        * ``composited`` / ``not_attempted`` (no url, or past the composite cap).
        """
        published = 0
        failures: list[str] = []
        for item in items:
            if not item.url:
                item.composite_status = NOT_ATTEMPTED
                item.composite_detail = "no source url"
                continue
            if item.media_type and item.media_type not in ("image", "video"):
                item.composite_status = NOT_COMPOSITABLE
                item.composite_detail = f"{item.media_type} item — not an image or clip"
                continue
            page_host = video_page_host(item.url)
            if page_host is not None:
                item.composite_status = NOT_COMPOSITABLE
                item.composite_detail = (
                    f"{page_host} watch page, not a downloadable clip — recorded for "
                    "reference, not shown on screen"
                )
                continue
            if published >= _MAX_COMPOSITE:
                item.composite_status = NOT_ATTEMPTED
                item.composite_detail = f"past the {_MAX_COMPOSITE}-item composite cap"
                continue
            try:
                if item.media_type == "video":
                    item.composite_url = await republish_video(
                        run_id, self.name, item.url, clip_id=item.id
                    )
                else:
                    # Normalised to the film frame at ingest (see still_fit).
                    item.composite_url = await republish_image(
                        run_id, self.name, item.url, image_id=item.id, fit_to=frame
                    )
                item.composite_status = COMPOSITED
                published += 1
            except NotMediaContentError as exc:
                log.info("media_discovery: %s is not media (%s)", item.url, exc.content_type)
                item.composite_status = NOT_COMPOSITABLE
                item.composite_detail = (
                    f"answered {exc.content_type}, not a{'n' if exc.expected == 'image' else ''} "
                    f"{exc.expected} file — recorded for reference, not shown on screen"
                )
            except Exception as exc:  # noqa: BLE001 -- soft-fail: keep external url only
                log.info("media_discovery: republish of %s failed: %s", item.url, exc)
                item.composite_status = COMPOSITE_FAILED
                item.composite_detail = str(exc)
                failures.append(f"{item.source or 'source'} {item.url}: {exc}")
        if failures:
            degradations.append(
                degradation(
                    MEDIA_NOT_COMPOSITED,
                    f"media_discovery: {len(failures)} media item(s) could not be "
                    "re-published and are not on screen — " + "; ".join(failures),
                )
            )

    @staticmethod
    def _build_query(cfg: dict, brief: dict, md_cfg: dict, ctx: RunContext) -> str:
        """Derive the search query: explicit override → analyze hooks → brief value."""
        explicit = str(md_cfg.get("query") or "").strip()
        if explicit:
            return explicit[:_QUERY_MAX_CHARS]
        parts: list[str] = []
        analyze = ctx.prior_output("promo_analyze") or {}
        for hook in (analyze.get("hook_candidates") or [])[:2]:
            text = hook.get("text") if isinstance(hook, dict) else None
            if text:
                parts.append(str(text))
        if not parts:
            for key in ("value_proposition", "angle", "audience", "outcome"):
                val = brief.get(key)
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
                    break
        if not parts and cfg.get("title"):
            parts.append(str(cfg["title"]))
        return " ".join(parts)[:_QUERY_MAX_CHARS].strip()

    @staticmethod
    def _to_items(results: list[MediaItem]) -> list[DiscoveredMedia]:
        out = [
            DiscoveredMedia(
                id=r.id,
                title=r.title,
                url=r.url,
                thumbnail_url=r.thumbnail_url or "",
                media_type=r.media_type,
                source=r.source,
                # Nullable server-side (unscored items) — rank those last.
                quality_score=r.quality_score or 0.0,
            )
            for r in results
            if r.id
        ]
        out.sort(key=lambda m: m.quality_score, reverse=True)
        return out


def _not_compositable_advisories(items: list[DiscoveredMedia]) -> list[Degradation]:
    """One ADVISORY naming every item classified as not-a-media-file, or ``[]``.

    Queryable from the run record (``advisories``) without flipping ``degraded``:
    a YouTube watch page in the search results is a fact about the results, not a
    defect in the video.
    """
    skipped = [item for item in items if item.composite_status == NOT_COMPOSITABLE]
    if not skipped:
        return []
    detail = "; ".join(f"{item.url}: {item.composite_detail}" for item in skipped)
    return [
        advisory(
            MEDIA_NOT_COMPOSITABLE,
            f"media_discovery: {len(skipped)} result(s) are not media files and were "
            f"kept for reference only — {detail}",
        )
    ]
