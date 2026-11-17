"""Generate a storyboard panel's image and write it onto the panel.

This is the storyboard-side sibling of the ``ai_image_gen`` run stage: same
downstream (``AIContentGenerationClient``), same content-addressed reuse, same
loud validation, same mandatory re-publish — but keyed on a PANEL, not a run.
Panels live on ``project_id`` alone (no run, no deliverable), so nothing here
can borrow the run's artifact namespace or its SSE channel.

Three things in here are load-bearing and are NOT optional cleanup targets:

* **The idempotency key.** A resubmit without one starts a SECOND paid
  provider generation for the same prompt. The key is derived from the
  content-addressed media id, so a retry addresses the ORIGINAL job.
* **The loud url validation.** ACG can answer HTTP 200 with an ``image_url``
  of ``None`` / ``"none"`` / ``"null"``. Writing that into ``media_ref`` is how
  dead images previously reached video_render; here it would put a broken
  image on the board with a green status. It fails instead.
* **The re-publish.** ACG hands back a license-gated, ACG-relative path
  (``/api/concept-images/assets/{id}/download``) that no anonymous consumer —
  including the operator's browser — can fetch. The bytes are downloaded with
  the licensed client and re-stored, and the FETCHABLE url is what lands in
  ``media_ref``. That url is presigned with a 1h TTL, so it is a durable object
  IDENTITY, not a durable link — the read path re-signs it per response (see
  ``storyboard/panel_presign``). Do not treat the stored string as servable.

Submissions are serialised in TWO layers, because the orchestrator runs
``replicas: 2`` (HPA to 5) and a process-wide gate alone would bound concurrency
at the replica count rather than at one:

* :func:`_submit_gate` — an in-pod, per-event-loop semaphore. Cheap, and it
  absorbs a whole-board burst without hammering Redis.
* :class:`~src.services.storyboard.panel_submit_lock.PanelSubmitLock` — a
  TTL'd Redis key, so the guarantee actually holds ACROSS replicas. When Redis
  is unreachable it falls open and logs, degrading to per-pod serialisation.

The image runner behind ACG is single-threaded and already carries the live
``ai_image_gen`` stage, which fans out three-wide; a whole-board panel generate
that also fanned out would head-of-line block real runs behind board previews.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel

from src.clients.ai_content_generation_client import AIContentGenerationClient
from src.database.repositories.storyboard_repository import (
    StoryboardAssetRepository,
    StoryboardPanelRepository,
)
from src.models.showcase_line import MediaRef, PanelGen
from src.services.concept_images import DEFAULT_BUDGET_S, generate_concept_image
from src.services.storyboard.panel_prompt import PanelPrompt, build_panel_prompt
from src.services.storyboard.panel_submit_lock import PanelSubmitLock
from src.services.tts_storage import (
    deterministic_media_id,
    find_existing_media,
    store_generated_image,
)

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.database.models import StoryboardPanel

log = logging.getLogger(__name__)

#: Artifact-store stage name for panel images (the namespace's second segment).
STAGE_NAME = "storyboard_panel_image"
_IMAGE_FORMAT = "png"
_DEFAULT_SIZE = "1024x1024"
# Whole budget per panel image (queue + render + same-key retry), handed to the
# shared helper ``src.services.concept_images``. The panel gates below only
# serialise THIS pod's panels; ACG's one-at-a-time backend still queues them
# behind every run's images, so a render-sized deadline is the wrong unit.
_DEFAULT_BUDGET_S = DEFAULT_BUDGET_S

# An ``image_url`` that is present but semantically empty. ACG 200s with these
# when the provider failed mid-job; they must never reach ``media_ref``.
_DEAD_URLS = frozenset({"", "none", "null", "nil", "undefined"})

# PNG signature + the fixed-offset IHDR header: 8-byte signature, 4-byte chunk
# length, the 4-byte type "IHDR", then width and height as big-endian uint32s at
# offsets 16 and 20. IHDR is required by the spec to be the FIRST chunk, so those
# offsets are exact for every valid PNG.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_IHDR_END = 24


def _png_dimensions(data: bytes) -> str | None:
    """``"{width}x{height}"`` read out of a PNG's IHDR, or ``None``.

    ``None`` for anything that is not a PNG whose header we can read — a
    truncated body, a JPEG, a provider error page. The caller records ``None``
    rather than guessing, because the point of the field is to be measured.

    Header-only, so it costs nothing and adds no image dependency; it does not
    decode the image and cannot detect corruption past the first chunk.
    """
    if len(data) < _PNG_IHDR_END or not data.startswith(_PNG_SIGNATURE):
        return None
    if data[12:16] != b"IHDR":
        return None
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    if width <= 0 or height <= 0:
        return None
    return f"{width}x{height}"


# One in-flight panel submission per process, created PER EVENT LOOP. This is
# only the IN-POD layer — PanelSubmitLock is what makes the guarantee hold
# across replicas. Do not describe this semaphore alone as "serialising
# submissions": with replicas:2 that claim would be false.
# asyncio primitives bind to the first loop that awaits them (3.12+), and the
# test suite runs a fresh loop per test — a module-level Semaphore built at
# import time would raise "attached to a different loop" on the second test.
_SUBMIT_GATES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)


def _submit_gate() -> asyncio.Semaphore:
    """The process-wide, per-loop serialisation gate for panel submissions."""
    loop = asyncio.get_running_loop()
    gate = _SUBMIT_GATES.get(loop)
    if gate is None:
        gate = asyncio.Semaphore(1)
        _SUBMIT_GATES[loop] = gate
    return gate


class PanelImageError(Exception):
    """Panel image generation failed for a reason the operator should see."""


class PanelNotFoundError(PanelImageError):
    """The panel vanished (or is not this tenant's) between submit and run."""


class PanelImageDownstreamError(PanelImageError):
    """ACG answered, but with nothing usable (dead url, failed job, no bytes)."""


@dataclass(frozen=True, slots=True)
class PanelImageOptions:
    """Per-request generation knobs. Everything is optional; defaults are the norm."""

    size: str | None = None
    provider: str | None = None
    seed: int | None = None
    negative_prompt: str | None = None


class PanelJobConfig(BaseModel):
    """The ACG concept-image request body this service sends.

    Typed rather than a loose dict so the ONE consistency lever the runner
    exposes cannot be silently dropped: the seed rides in ``extra_params``
    because ``ConceptImageRequest`` has no top-level ``seed`` field, and the
    local image provider forwards ``extra_params``' ``steps``/``guidance``/
    ``seed`` to the runner verbatim. There is no reference-image field here on
    purpose — the runner is text→image, so nothing can condition it on a picture.
    """

    size: str
    extra_params: dict[str, int]
    provider: str | None = None
    negative_prompt: str | None = None

    def to_payload(self) -> dict[str, str | int | dict[str, int]]:
        """The client kwarg form — unset overrides omitted, not sent as null."""
        return self.model_dump(exclude_none=True)


@dataclass(slots=True)
class PanelImageResult:
    """What the caller (the job runner) reports back and publishes."""

    panel_id: UUID
    media_ref: MediaRef
    gen: PanelGen
    reused: bool = False


class StoryboardPanelImageService:
    """Generate one panel's image via ACG and persist it onto the panel row.

    The session is supplied by the caller and is expected to already be bound to
    the tenant (RLS ``app.current_tenant``) — the background runner does that
    before handing it over.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        client: AIContentGenerationClient | None = None,
        redis: Redis | None = None,
        budget_s: float = _DEFAULT_BUDGET_S,
    ) -> None:
        self._session = session
        self._client = client or AIContentGenerationClient()
        # Without a redis handle the cross-replica lock cannot be taken and
        # serialisation is per-pod only. The job runner (the only production
        # caller) always supplies one; it is optional so a direct/service-level
        # caller isn't forced to invent a Redis dependency.
        self._redis = redis
        self._budget_s = budget_s

    async def generate(
        self,
        panel_id: UUID,
        *,
        tenant_id: UUID,
        options: PanelImageOptions | None = None,
    ) -> PanelImageResult:
        opts = options or PanelImageOptions()
        panels = StoryboardPanelRepository(self._session)
        panel = await panels.get(panel_id, tenant_id=tenant_id)
        if panel is None:
            raise PanelNotFoundError(f"panel {panel_id} not found for this tenant")

        assets = {
            asset.id: asset
            for asset in await StoryboardAssetRepository(self._session).list_for_project(
                panel.project_id, tenant_id=tenant_id
            )
        }
        built = build_panel_prompt(panel, assets, seed_override=opts.seed)
        job_config = self._build_job_config(built.seed, opts)
        namespace = self._namespace(panel.project_id)
        # Content-addressed over EVERYTHING that determines the pixels: the
        # prompt text (which already folds in the assets' descriptors), the job
        # config (seed/size/provider), and the panel identity. Change any of
        # them and the id changes → a real regenerate. Change none and the id
        # is identical → the stored image is reused and nothing is paid for.
        media_id = deterministic_media_id(built.prompt, job_config.model_dump_json(), str(panel_id))

        existing = await find_existing_media(namespace, STAGE_NAME, media_id, _IMAGE_FORMAT)
        if existing is not None:
            return await self._persist(
                panels,
                panel,
                tenant_id=tenant_id,
                url=existing,
                asset_id=None,
                provider=None,
                built=built,
                media_id=media_id,
                job_config=job_config,
                reused=True,
                # A reused object is not downloaded, so its pixels are never in
                # this process — nothing to measure.
                measured_size=None,
            )

        url, acg_asset_id, provider = await self._submit_and_wait(
            built.prompt, job_config, media_id=media_id, project_id=panel.project_id
        )
        url, measured_size = await self._republish_relative(
            url, namespace=namespace, media_id=media_id
        )
        return await self._persist(
            panels,
            panel,
            tenant_id=tenant_id,
            url=url,
            asset_id=acg_asset_id,
            provider=provider,
            built=built,
            media_id=media_id,
            job_config=job_config,
            reused=False,
            measured_size=measured_size,
        )

    # ─── downstream ──────────────────────────────────────────────────────
    async def _submit_and_wait(
        self,
        prompt: str,
        job_config: PanelJobConfig,
        *,
        media_id: str,
        project_id: UUID,
    ) -> tuple[str, str | None, str | None]:
        """Submit to ACG under BOTH serialisation gates and validate the answer LOUDLY.

        In-pod semaphore first, then the cross-replica Redis lock. The order is
        fixed and there is only one lock pair, so it cannot deadlock; holding the
        local slot while queueing for the global one IS the queue.
        """
        async with _submit_gate(), self._cross_replica_gate():
            # Idempotency: a retry of this call (client reconnect, runner
            # restart, operator double-click) MUST address the original job.
            # Without the key ACG starts a second paid provider generation for
            # the identical prompt — the exact double-bill ai_image_gen's
            # docstring calls out. The key is content-addressed, so it is stable
            # across processes, not just across retries in one task.
            result = await generate_concept_image(
                self._client,
                prompt,
                job_config.to_payload(),
                idempotency_key=f"storyboard:{project_id}:{media_id}",
                budget_s=self._budget_s,
            )
        raw = result.image_url or result.download_url
        # ACG can answer 200 with a semantically empty url when the provider
        # failed mid-job. Writing that onto the panel yields a broken image with
        # a green status — fail instead, so the operator retries a visibly
        # failed panel rather than staring at a dead thumbnail.
        if raw is None or str(raw).strip().lower() in _DEAD_URLS:
            raise PanelImageDownstreamError(
                f"ai_content_generation returned no usable image_url for panel image "
                f"{media_id} (job={result.job_id}, status={result.status!r})"
            )
        return str(raw).strip(), result.asset_id, result.provider

    def _cross_replica_gate(self) -> AbstractAsyncContextManager[object]:
        """The estate-wide submission slot, or a no-op when no Redis is wired."""
        if self._redis is None:
            return nullcontext()
        return PanelSubmitLock(self._redis)

    async def _republish_relative(
        self, url: str, *, namespace: str, media_id: str
    ) -> tuple[str, str | None]:
        """Re-publish an ACG-relative download path to a fetchable artifact url.

        Returns ``(url, measured_size)``, where ``measured_size`` is
        ``"{width}x{height}"`` read out of the bytes we downloaded, or ``None``
        when we never held them (the absolute-url passthrough below) or could not
        read a PNG header. The size is measured HERE because this is the only
        place in the flow that has the pixels: the generator does not report the
        size it actually produced, and the requested size is not it — a request
        for ``1024x1024`` has been observed coming back 1536x1536.

        NOT best-effort here, unlike the run stage. ``ai_image_gen`` can leave
        the original path in place because its consumer (video-compile) has a
        fallback; a storyboard panel's consumer is the operator's browser, which
        has none — a license-gated relative path renders as a broken image.
        A failed re-publish is therefore a failed generation.

        An absolute ``http(s)`` url (a cloud provider's CDN link) passes through
        untouched; there is nothing license-gated to unwrap.
        """
        if not url.startswith("/"):
            return url, None
        try:
            data = await self._client.download_concept_image(url)
        except Exception as exc:
            raise PanelImageDownstreamError(
                f"could not download the generated panel image from {url}: {exc}"
            ) from exc
        if not data:
            raise PanelImageDownstreamError(
                f"ai_content_generation served zero bytes for panel image at {url}"
            )
        stored = await store_generated_image(
            run_id=namespace,
            stage_name=STAGE_NAME,
            image_id=media_id,
            image_bytes=data,
            image_format=_IMAGE_FORMAT,
        )
        return stored, _png_dimensions(data)

    # ─── persistence ─────────────────────────────────────────────────────
    async def _persist(
        self,
        panels: StoryboardPanelRepository,
        panel: StoryboardPanel,
        *,
        tenant_id: UUID,
        url: str,
        asset_id: str | None,
        provider: str | None,
        built: PanelPrompt,
        media_id: str,
        job_config: PanelJobConfig,
        reused: bool,
        measured_size: str | None,
    ) -> PanelImageResult:
        """Write ``media_ref`` + ``gen`` (+ ``visual_type``/``source``) onto the panel.

        ``gen.size`` is *measured_size* — the dimensions read off the delivered
        bytes — and ``None`` when they were never in this process. It is
        deliberately NOT ``job_config.size``: that is what we ASKED for, and the
        generator overrides it (a 1024x1024 request came back 1536x1536 in prod),
        so recording it made ``gen`` — the reproducibility record — assert
        something untrue about the image beside it.

        NO ``expected_lock_version`` here on purpose. The CAS is asserted at
        SUBMIT time, against the panel the operator was looking at; by the time
        this runs (minutes later) a legitimate text edit may have bumped the
        token, and 409ing then would silently discard an image that was already
        paid for. The write touches ONLY the four generated-media fields, so it
        cannot clobber an author's concurrent edit to the dialogue or the shot.
        """
        media_ref = MediaRef(type="ai_image", asset_id=asset_id, url=url)
        gen = PanelGen(
            prompt=built.prompt,
            # The seed is the whole reproducibility contract: it is what makes a
            # regenerate of an unchanged panel return the SAME image, and what a
            # sibling panel copies to keep a character on-model. Dropping it
            # leaves an image nobody can reproduce.
            seed=built.seed,
            model=provider,
            provider=provider,
            size=measured_size,
            style_token=built.style_token,
            asset_ids=list(built.asset_ids),
            media_id=media_id,
            reused=reused,
            generated_at=datetime.now(UTC).isoformat(),
        )
        row = await panels.update(
            panel.id,
            tenant_id=tenant_id,
            fields={
                "media_ref": media_ref.model_dump(),
                "gen": gen.model_dump(),
                "visual_type": "ai_image",
                "source": "ai",
            },
        )
        if row is None:
            raise PanelNotFoundError(f"panel {panel.id} disappeared before the image landed")
        return PanelImageResult(panel_id=panel.id, media_ref=media_ref, gen=gen, reused=reused)

    # ─── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _namespace(project_id: UUID) -> str:
        """Artifact namespace for a project's panel images.

        The storage helpers name this argument ``run_id`` because runs were the
        first caller, but it is only ever a path prefix (see
        ``store_thumbnail_override``, which passes a tenant). Storyboard tables
        carry no run_id at all, so the project is the only stable scope.
        """
        return f"storyboard/{project_id}"

    @staticmethod
    def _build_job_config(seed: int, opts: PanelImageOptions) -> PanelJobConfig:
        """The ACG concept-image config for this panel, with the seed pinned."""
        return PanelJobConfig(
            size=opts.size or _DEFAULT_SIZE,
            extra_params={"seed": seed},
            provider=opts.provider or None,
            negative_prompt=opts.negative_prompt or None,
        )
