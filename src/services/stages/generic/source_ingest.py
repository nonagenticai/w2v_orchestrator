"""source_ingest — ground a project in a *combination* of sources.

Opt-in (like ``site_capture``): runs only when the brief carries
``brief.sources`` (the universal multi-source list — see
``src/models/source_spec.py``); otherwise raises ``StageSkippedError`` and the
run proceeds without it. When active it, per source:

  * ``url``      → webscraper ``POST /api/scrape/url`` via w2v_common's
                   ``WebscraperClient`` (skipping the one already recorded by
                   ``site_capture`` to avoid a double fetch),
  * ``document`` → ``fetch_document(ref, tenant)`` (MinIO read-back by key,
                   confined to the run's own tenant upload prefix) then
                   ``extract_document`` (in-process pdf/docx/epub/txt/md parse,
                   OCR for scanned PDF pages — ``src/services/documents``),
  * ``media``    → re-published to run storage (image pins) and passed through to
                   ``pinned_media`` (``composite_url``) for promo_render.

It aggregates the extracted text into ``grounding_text`` (capped), collects any
extracted/illustrative images (re-published so the compiler can fetch them
anonymously), and surfaces references + soft-failure notes.

**Persistent cross-run cache.** Each ``url``/``document`` ingest is memoized in a
tenant-scoped DB row (``source_cache.SourceCacheRepository``) so many runs —
notably concurrent course runs — over the SAME source don't each re-scrape/re-parse.
Active only when the dispatcher supplied a ``session_factory`` AND the run has a
tenant AND ``STORAGE_SOURCE_INGEST_CACHE_ENABLED``; otherwise (tests / no tenant /
flag off) the direct per-run path runs exactly as before. Cached images are stored
under the durable ``media-cache/`` prefix (re-minted at consume), so a cache HIT
reproduces the same output a fresh ingest would. Any cache-layer error degrades to
a direct ingest — the cache never fails a run. ``brief.source_ingest_refresh``
forces a bypass+refresh.

Failure semantics are deliberately soft (mirrors ``site_capture``): a scrape /
parse error degrades to fewer sources and a recorded note rather than failing
the run — grounding is an enhancement, not a precondition. Only "no
``brief.sources``" short-circuits, as a clean skip.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid7

from w2v_common.clients import ImageRef, WebscraperClient

from src.clients.downstream_credential import hop_lane
from src.config import get_settings
from src.models.source_spec import SourceSpec, parse_sources
from src.services.auth_secrets import auth_for_origin, unseal_auth
from src.services.document_storage import fetch_document
from src.services.documents.extract import ExtractedImage, extract_document, normalise_image
from src.services.source_cache import (
    CachedSource,
    SourceCacheRepository,
    auth_id_for,
    document_ref_key,
    normalize_url,
    source_fingerprint,
)
from src.services.stages.base import (
    SLOT_USE_OPTIONAL,
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithDegradations,
    StageSkippedError,
)
from src.services.stages.generic.grounding import (  # re-exported (pure readers)
    _GROUNDING_PROMPT_MAX_CHARS,  # noqa: F401
    grounding_prompt_block,  # noqa: F401
    grounding_sentences,  # noqa: F401
)
from src.services.stages.shared.degradation import SOURCE_GROUNDING_MISSING, degradation
from src.services.stages.shared.motion_slides import film_frame_dims
from src.services.tts_storage import (
    ingest_cache_image_url,
    store_capture_image,
    store_ingest_cache_image,
)
from src.services.visual_media import (
    NotMediaContentError,
    republish_image,
    republish_video,
    video_page_host,
)

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

# A live page fetch (Playwright, possibly a login flow) can be slow, so the
# scrape gets a generous per-page deadline (webscraper accepts 1-300 s). The
# stage's own DAG timeout bounds the whole fan-out.
_SCRAPE_TIMEOUT_S = 180.0
# Bound the grounding payload fed to the LLM, and the work of re-publishing
# extracted images.
_GROUNDING_MAX_CHARS = 24000
_PER_SOURCE_TEXT_MAX_CHARS = 12000
_MAX_IMAGES = 12
_MAX_REFERENCES = 50


class SourceIngestInput(StageInput):
    pass  # All inputs come from ctx.config_snapshot["brief"].


class IngestedImage(StageOutput):
    id: str
    url: str = ""
    caption: str = ""
    alt_text: str = ""
    source_ref: str = ""


class SourceIngestOutput(StageOutputWithDegradations):
    # Concatenated, source-labelled extracted text the LLM stages ground on.
    grounding_text: str = ""
    references: list[str] = []  # noqa: RUF012 -- pydantic per-instance default
    images: list[IngestedImage] = []  # noqa: RUF012
    # media-kind sources pinned in the wizard, passed through for visual stages.
    pinned_media: list[dict[str, object]] = []  # noqa: RUF012
    notes: list[str] = []  # noqa: RUF012
    sources_ingested: int = 0


def make_source_ingest_stage(deps: dict) -> StageHandler:
    """Blueprint factory: wire the persistent source-ingest cache with the
    dispatcher-provided ``session_factory`` (present for every real run). A caller
    that constructs ``SourceIngestStage()`` directly (tests / non-DAG use) gets the
    cache-disabled direct path — behavior-identical to before the cache existed."""
    return SourceIngestStage(session_factory=deps.get("session_factory"))


class SourceIngestStage(StageHandler[SourceIngestInput, SourceIngestOutput]):
    name = "source_ingest"
    input_model = SourceIngestInput
    output_model = SourceIngestOutput
    #: OCR is needed only for a scanned (text-less) PDF page, so it never gates the
    #: run up front; a byo_required tenant without an OCR connection soft-fails that
    #: document with a note naming the slot, only when such a page is actually met.
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("ocr", when=SLOT_USE_OPTIONAL),)

    def __init__(
        self, client: WebscraperClient | None = None, *, session_factory: Any = None
    ) -> None:
        # Built lazily (``_scraper``): WebscraperClient refuses construction without
        # a licence, and a stage instance must not fail a run that has no url source.
        self._webscraper = client
        self._session_factory = session_factory
        self._instance_id = uuid7().hex
        # Per-``handle`` cache context (one instance per run, set at handle start).
        self._repo: SourceCacheRepository | None = None
        self._tenant_id: Any = None
        self._refresh = False
        self._ttl = 0

    async def handle(
        self,
        payload: SourceIngestInput,
        ctx: RunContext,
    ) -> SourceIngestOutput:
        # Course-level in-run cache (#293): a course run ingests its sources once and
        # hands the result to each lesson child via config_overrides. When present,
        # reuse it verbatim — this short-circuits BEFORE the persistent cache, so a
        # child never re-fetches (nor re-consults the DB cache).
        precomputed = ctx.config_snapshot.get("precomputed_source_ingest")
        if isinstance(precomputed, dict) and precomputed.get("sources_ingested"):
            reused = SourceIngestOutput(**precomputed)
            reused.notes = [
                *reused.notes,
                "reused course-level source_ingest (child skipped re-fetch)",
            ]
            return reused

        brief = ctx.config_snapshot.get("brief") or {}
        sources = parse_sources(brief)
        if not sources:
            raise StageSkippedError(
                "source_ingest: no brief.sources (opt-in multi-source grounding not requested)",
                missing=["brief.sources"],
            )

        self._init_cache(ctx, brief)

        # Reuse only a completed, usable capture. Promo orders this stage after
        # site_capture; other blueprints can still ingest independently.
        site_url = (brief.get("source_url") or "").strip()
        capture = ctx.prior_output("site_capture")
        run_id = str(ctx.run_id)

        text_parts: list[str] = []
        references: list[str] = []
        images: list[IngestedImage] = []
        pinned_media: list[dict[str, Any]] = []
        notes: list[str] = []

        for src in sources:
            if src.kind == "media":
                pinned_media.append(
                    await self._pin_media(
                        src, run_id, notes, frame=film_frame_dims(ctx.config_snapshot)
                    )
                )
                continue
            if src.kind == "url":
                if site_url and src.ref == site_url and capture:
                    text = str(capture.get("page_text") or "").strip()
                    status = capture.get("page_text_status")
                    if text and (status == "ok" or (status is None and len(text) >= 200)):
                        text_parts.append(text)
                        references.append(src.ref)
                        continue
                await self._ingest_url(src, run_id, text_parts, references, images, notes)
            elif src.kind == "document":
                await self._ingest_document(src, run_id, text_parts, references, images, notes)

        grounding_text = "\n\n".join(text_parts)[:_GROUNDING_MAX_CHARS]
        if not text_parts and not pinned_media:
            notes.append("no source content ingested; run proceeds without grounding")
        return SourceIngestOutput(
            grounding_text=grounding_text,
            degradations=[
                degradation(
                    SOURCE_GROUNDING_MISSING,
                    "no source content ingested; run proceeds without grounding",
                )
            ]
            if not text_parts and not pinned_media
            else [],
            references=references[:_MAX_REFERENCES],
            images=images[:_MAX_IMAGES],
            pinned_media=pinned_media,
            notes=notes,
            sources_ingested=len(text_parts) + len(pinned_media),
        )

    def _init_cache(self, ctx: RunContext, brief: dict[str, Any]) -> None:
        """Resolve whether the persistent cache is active for this run and stash the
        context the per-source loop reads. Active only with a session_factory + a
        bound tenant + the flag on; else the direct per-run path runs."""
        storage = get_settings().storage
        self._tenant_id = ctx.tenant_id
        self._refresh = bool(brief.get("source_ingest_refresh"))
        self._ttl = storage.source_ingest_cache_ttl_seconds
        if self._session_factory and self._tenant_id and storage.source_ingest_cache_enabled:
            self._repo = SourceCacheRepository(self._session_factory, instance_id=self._instance_id)
        else:
            self._repo = None

    async def _pin_media(
        self,
        src: SourceSpec,
        run_id: str,
        notes: list[str],
        *,
        frame: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Pass a wizard-pinned media item through, re-publishing image OR video.

        ``composite_url`` (compiler-fetchable, in-cluster) is populated when the
        pin resolves to a fetchable image or video; ``kind`` records which
        ("image"/"video") so the render routes it to an image line or a clip
        line. We try an image re-publish first (the cheap content-type check
        rejects non-images before the body downloads), then fall back to a
        streamed, size-capped video re-publish. HTML / failures leave
        ``composite_url`` "" and ``kind`` "". The ``ref`` is always kept for the
        run detail. Soft-fail. (Media pins are pass-through, not cached.)

        ``composite_status`` says WHY: ``composited``, ``not_compositable`` (a
        video WATCH page such as a pinned YouTube result — classified up front,
        no request — or a URL that answered with non-media content; recorded, no
        note, because it is a classification and not a failure) or ``failed`` (a
        real fetch/store error, which keeps its note).

        ``frame`` is the film's output size: an image pin is normalised to it at
        ingest (``republish_image(fit_to=...)``) so a full-size phone photo never
        reaches the render stages at camera resolution.
        """
        page_host = video_page_host(src.ref)
        if page_host is not None:
            return {
                "ref": src.ref,
                "label": src.label,
                "composite_url": "",
                "kind": "",
                "composite_status": "not_compositable",
                "composite_detail": (
                    f"{page_host} watch page, not a downloadable clip — ingested for "
                    "reference, not shown on screen"
                ),
            }
        composite_url = ""
        kind = ""
        status = "composited"
        detail = ""
        try:
            composite_url = await republish_image(
                run_id, self.name, src.ref, image_id=str(uuid7()), fit_to=frame
            )
            kind = "image"
        except Exception as img_exc:  # noqa: BLE001 -- not an image: try video, else soft-fail
            try:
                composite_url = await republish_video(
                    run_id, self.name, src.ref, clip_id=str(uuid7())
                )
                kind = "video"
            except Exception as vid_exc:  # noqa: BLE001 -- soft-fail: keep the ref only
                log.info(
                    "source_ingest: pinned media %s not composited (image: %s; video: %s)",
                    src.ref,
                    img_exc,
                    vid_exc,
                )
                if isinstance(img_exc, NotMediaContentError) and isinstance(
                    vid_exc, NotMediaContentError
                ):
                    status = "not_compositable"
                    detail = (
                        f"answered {vid_exc.content_type}, not an image or video file — "
                        "ingested for reference, not shown on screen"
                    )
                else:
                    status = "failed"
                    # The IMAGE error is the reason when the pin IS an image: a
                    # too-large or undecodable photo also fails the video path with
                    # "not a video (image/jpeg)", which hid the real cause.
                    reason = vid_exc if isinstance(img_exc, NotMediaContentError) else img_exc
                    detail = str(reason)
                    notes.append(
                        f"pinned media not composited for {src.ref}: "
                        f"{type(reason).__name__}: {reason}"
                    )
        return {
            "ref": src.ref,
            "label": src.label,
            "composite_url": composite_url,
            "kind": kind,
            "composite_status": status,
            "composite_detail": detail,
        }

    async def _ingest_url(
        self,
        src: SourceSpec,
        run_id: str,
        text_parts: list[str],
        references: list[str],
        images: list[IngestedImage],
        notes: list[str],
    ) -> None:
        try:
            cached = await self._cached_or_direct(
                src, "url", lambda fp: self._produce_url(src, run_id, fp)
            )
        except Exception as exc:  # noqa: BLE001 -- soft-fail per source; record, don't abort
            log.warning("source_ingest: scrape of %s failed: %s", src.ref, exc)
            notes.append(f"scrape failed for {src.ref}: {exc}")
            return
        await self._apply_cached(cached, src, text_parts, references, images, notes)

    async def _ingest_document(
        self,
        src: SourceSpec,
        run_id: str,
        text_parts: list[str],
        references: list[str],
        images: list[IngestedImage],
        notes: list[str],
    ) -> None:
        try:
            # The ref rides in on the brief, so it is confined to THIS tenant's
            # upload prefix inside fetch_document — a brief naming another
            # tenant's key soft-fails like any missing document.
            data = await fetch_document(src.ref, str(self._tenant_id or ""))
        except Exception as exc:  # noqa: BLE001 -- soft-fail per source
            log.warning("source_ingest: document fetch %s failed: %s", src.ref, exc)
            notes.append(f"document not found for {src.ref}: {exc}")
            return
        try:
            cached = await self._cached_or_direct(
                src,
                "document",
                lambda fp: self._produce_document(src, run_id, data, fp),
                ref_key=document_ref_key(data),
            )
        except Exception as exc:  # noqa: BLE001 -- soft-fail per source
            log.warning("source_ingest: document parse %s failed: %s", src.ref, exc)
            notes.append(f"document parse failed for {src.ref}: {exc}")
            return
        await self._apply_cached(cached, src, text_parts, references, images, notes)

    async def _cached_or_direct(
        self,
        src: SourceSpec,
        kind: str,
        producer_for_fp: Any,
        *,
        ref_key: str | None = None,
    ) -> CachedSource:
        """Return the per-source contribution — from the persistent cache when
        active (single-flight; degrade-safe), else by producing it directly.

        ``producer_for_fp(fingerprint)`` runs the real scrape / parse and stores
        images (durably under the ``fingerprint`` when caching, per-run when
        ``None``); it raises on a scrape/parse error, which the caller turns into a
        soft-fail note.
        """
        if self._repo is None:
            return await producer_for_fp(None)
        rk = ref_key if ref_key is not None else normalize_url(src.ref)
        fingerprint = source_fingerprint(
            self._tenant_id, kind, rk, auth_id=auth_id_for(src.auth_info)
        )
        result, _ = await self._repo.get_or_produce(
            fingerprint,
            self._tenant_id,
            kind,
            src.ref,
            ttl_seconds=self._ttl,
            refresh=self._refresh,
            producer=lambda: producer_for_fp(fingerprint),
        )
        return result

    def _scraper(self) -> WebscraperClient:
        """The webscraper client (``WEBSCRAPER_AUTH_MODE`` lane), built on first use."""
        if self._webscraper is None:
            self._webscraper = WebscraperClient(licence_lane=hop_lane("webscraper"))
        return self._webscraper

    async def _produce_url(
        self, src: SourceSpec, run_id: str, fingerprint: str | None
    ) -> CachedSource:
        """Scrape a URL via webscraper and pack the per-source contribution. Raises
        on a scrape error (the caller soft-fails). ``fingerprint`` set → images
        stored durably for cross-run reuse; ``None`` → per-run storage (direct path)."""
        page = await self._scraper().scrape_url(
            src.ref,
            download_images=True,
            # Sealed at rest (auth_secrets); unsealed in memory for the scrape only.
            auth_config=auth_for_origin(unseal_auth(src.auth_info), src.ref, {"ref": src.ref}),
            timeout=_SCRAPE_TIMEOUT_S,
        )
        if page.error:
            # webscraper answers an unscrapable page with HTTP 200 + status=failed.
            raise RuntimeError(f"webscraper could not scrape {src.ref}: {page.error}")
        text = (page.markdown or page.text).strip()
        images = await self._store_images(_scraped_images(page.images), run_id, fingerprint)
        return CachedSource(text=text, references=[], images=images)

    async def _produce_document(
        self, src: SourceSpec, run_id: str, data: bytes, fingerprint: str | None
    ) -> CachedSource:
        """Parse a document in-process and pack the contribution. Raises
        ``DocumentExtractError`` when no text can be extracted (the caller turns it
        into a visible soft-fail note naming why)."""
        doc = await extract_document(data, _document_name(src))
        images = await self._store_images(doc.images, run_id, fingerprint)
        return CachedSource(
            text=doc.text, references=list(doc.references), images=images, notes=doc.notes
        )

    async def _store_images(
        self, assets: list[ExtractedImage], run_id: str, fingerprint: str | None
    ) -> list[dict[str, Any]]:
        """Persist extracted images and return their descriptors (capped). Durable
        ``media-cache/`` blobs (``cache_key``) when caching; per-run URLs otherwise."""
        descriptors: list[dict[str, Any]] = []
        for idx, asset in enumerate(assets):
            if len(descriptors) >= _MAX_IMAGES:
                break
            desc = await self._store_one_image(asset, run_id, fingerprint, idx)
            if desc is not None:
                descriptors.append(desc)
        return descriptors

    async def _store_one_image(
        self, asset: ExtractedImage, run_id: str, fingerprint: str | None, idx: int
    ) -> dict[str, Any] | None:
        """One extracted image → a stored descriptor, or ``None`` when publishing
        failed (soft-fail per image: the image is dropped, never kept as a dead URL)."""
        image_id = str(uuid7())
        base = {
            "id": image_id,
            "caption": asset.caption,
            "alt_text": asset.alt_text,
        }
        fmt = _image_format(asset.mime_type)
        try:
            if fingerprint is not None:
                key = await store_ingest_cache_image(
                    str(self._tenant_id), fingerprint, idx, asset.data, fmt
                )
                if key:
                    return {**base, "cache_key": key}
                # Durable store unavailable (partial backend) → per-run fallback.
            url = await store_capture_image(
                run_id=run_id,
                stage_name="source_ingest",
                image_id=image_id,
                image_bytes=asset.data,
                image_format=fmt,
            )
        except Exception as exc:  # noqa: BLE001 -- soft-fail per image
            log.info("source_ingest: image publish failed: %s", exc)
            return None
        return {**base, "url": url}

    async def _apply_cached(
        self,
        cached: CachedSource,
        src: SourceSpec,
        text_parts: list[str],
        references: list[str],
        images: list[IngestedImage],
        notes: list[str],
    ) -> None:
        """Fold a per-source contribution (cache hit OR fresh) into the aggregation
        — the same labelled-text/references/images shape a direct ingest built.
        Image URLs carrying a durable ``cache_key`` are re-minted here (dropped if
        the blob was GC'd); per-run ``url`` descriptors pass through."""
        if cached.text:
            text_parts.append(_labelled_block(src, cached.text))
        notes.extend(cached.notes)
        for ref in cached.references:
            references.append(str(ref))
        for desc in cached.images:
            if len(images) >= _MAX_IMAGES:
                break
            key = desc.get("cache_key")
            if key:
                url = await ingest_cache_image_url(str(key))
                if url is None:  # blob GC'd / unresolvable → drop this image
                    continue
            else:
                url = str(desc.get("url") or "")
            images.append(
                IngestedImage(
                    id=str(desc.get("id") or uuid7()),
                    url=url,
                    caption=str(desc.get("caption") or ""),
                    alt_text=str(desc.get("alt_text") or ""),
                    source_ref=src.ref,
                )
            )


def _labelled_block(src: SourceSpec, text: str) -> str:
    head = src.label or src.ref
    return f"# Source: {head}\n{text[:_PER_SOURCE_TEXT_MAX_CHARS].strip()}"


def _image_format(mime_type: str | None) -> str:
    mt = (mime_type or "").lower()
    if "png" in mt:
        return "png"
    if "webp" in mt:
        return "webp"
    return "jpg"


def _document_name(src: SourceSpec) -> str:
    """A filename for parser/extension sniffing: the label when it names a file,
    else the storage ref (upload keys keep the original filename)."""
    label = (src.label or "").strip()
    if "." in label.rsplit("/", 1)[-1]:
        return label
    return src.ref.rsplit("/", 1)[-1] or label or "document"


def _scraped_images(refs: list[ImageRef]) -> list[ExtractedImage]:
    """Downloaded page images that are big enough to illustrate (icons dropped)."""
    out: list[ExtractedImage] = []
    for ref in refs:
        raw = ref.content
        if raw is None:
            continue
        image = normalise_image(raw, alt_text=ref.alt)
        if image is not None:
            out.append(image)
        if len(out) >= _MAX_IMAGES:
            break
    return out
