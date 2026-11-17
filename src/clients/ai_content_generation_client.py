"""HTTP client for AI Content Generation Service (port 6310)."""

from __future__ import annotations

import asyncio
import re
from collections.abc import MutableMapping
from typing import Any

import httpx
from w2v_common.clients.base import raise_for_status_with_body

from src.clients.base import LicensedServiceClient
from src.clients.gpu_busy import BUSY_STATUSES, BusyRetry, GpuBusyError
from src.clients.schemas.ai_content_generation import (
    AcgConceptImageAsset,
    AcgConceptImageBatch,
    AcgConceptImageBatchSubmission,
    AcgConceptImageJob,
    AcgVideoGenerationResult,
    AcgVideoJobStatus,
    AcgVideoProviderList,
)
from src.services.providers.errors import SlotNotConfiguredError
from src.services.providers.slot_payload import PROVIDER_SLOT_FIELD, build_provider_slot

#: How ACG's ``local_image`` provider reports modality_gateway's **402** (a
#: ``byo_required`` tenant with no image provider) as the job's ``error``.
#: Vendored verbatim from ai_content_generation_service
#: ``src/services/image_providers/local_image.py`` (``backend returned {status}``).
#:
#: ⚠️ PROSE, because it is the only signal: the job envelope carries a string
#: ``error`` and no status code. Pinned by a test so a reword fails CI here.
IMAGE_BYO_REFUSAL_MARKER = "local_image: backend returned 402"

#: How ACG's ``local_image`` provider reports a GPU backend that stayed BUSY
#: (429/503, the admission contract's "not now") through ACG's own busy retries:
#: the same vendored ``backend returned {status}`` prose. The image lane was full,
#: the request was fine — raised as :class:`GpuBusyError` (transient, ``external``)
#: so the stage re-drives it instead of failing the run terminally.
IMAGE_BUSY_MARKER = re.compile(r"local_image: backend returned (429|503)\b")

#: The service name busy errors are attributed to.
_SERVICE = "ai_content_generation"

# Indirection so tests patch the busy wait instead of sleeping through it.
_sleep = asyncio.sleep


async def _attach_provider_slot(payload: MutableMapping[str, Any], slot: str) -> None:
    """Add ``provider_slot`` for a run-pinned tenant connection; leave ``payload`` alone else.

    Set AFTER the caller's config merge so a stage config can never forge or
    override it. The body is never logged by ``submit_job`` (only the endpoint), and
    an upstream error body that echoes it is redacted by w2v_common's ``secret``/
    pydantic-``input`` rules."""
    payload.pop(PROVIDER_SLOT_FIELD, None)
    provider_slot = await build_provider_slot(slot)
    if provider_slot is not None:
        payload[PROVIDER_SLOT_FIELD] = provider_slot
        return
    # No tenant connection ⇒ ACG uses OUR provider: the platform choke point.
    from src.services.providers.platform_guard import refuse_platform_for_bound_tenant

    await refuse_platform_for_bound_tenant(slot)


class AssetTooLargeError(RuntimeError):
    """A downloaded asset exceeded the caller's byte ceiling (never buffered whole)."""


class AIContentGenerationClient(LicensedServiceClient):
    """Client for the AI Content Generation microservice."""

    def __init__(
        self,
        base_url: str | None = None,
        license_token: str | None = None,
        **kwargs: Any,  # noqa: ANN401 -- forwarded verbatim to LicensedServiceClient
    ) -> None:
        super().__init__(
            service_name="ai_content_generation",
            base_url=base_url,
            license_token=license_token,
            **kwargs,
        )

    async def list_video_providers(self) -> AcgVideoProviderList:
        """List the video providers this ACG deployment actually registered.

        ACG builds its registry at start-up and only registers a provider whose
        credentials/health check passed, so this is a real capability probe, not
        a static catalogue: on a deployment with no working cloud key and no
        local GPU the ``providers`` list comes back EMPTY. Callers use it to
        avoid submitting a generation that can never be served.
        """
        data = await self._request("GET", "/api/content/videos/providers")
        return AcgVideoProviderList.model_validate(data)

    async def start_generation(
        self,
        prompt: str,
        config: dict | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> str:
        """Start AI video generation job, returns job_id.

        ``idempotency_key`` (sent as the Idempotency-Key header) makes a
        re-submit of the SAME generation — e.g. ``ai_video_gen`` retrying its
        per-item callable after a transient wait/poll failure — return the
        ORIGINAL job instead of spawning a SECOND paid provider render (ACG
        dedupes on it server-side). Without it, the re-submit double-bills.

        When the bound run pinned a TENANT ``video_gen`` connection, its resolved
        slot rides as ``provider_slot`` (P5 contract); otherwise the body is
        byte-identical to the pre-P5 one.
        """
        payload: dict[str, Any] = {"prompt": prompt}
        if config:
            payload.update(config)
        await _attach_provider_slot(payload, "video_gen")
        return await self.submit_job(
            "/api/content/videos/generate", payload, idempotency_key=idempotency_key
        )

    async def get_generation(self, job_id: str) -> AcgVideoJobStatus:
        """Get an AI video job's THIN status envelope (no asset URL/duration)."""
        data = await self._request("GET", f"/api/content/videos/jobs/{job_id}")
        return AcgVideoJobStatus.model_validate(data)

    async def wait_for_generation(self, job_id: str, timeout: float = 300.0) -> AcgVideoJobStatus:
        """Poll until an AI video job reaches a terminal status.

        Returns the THIN status envelope. The asset URL + duration live on the
        ``/result`` endpoint — call :meth:`get_generation_result` once this
        reports ``status == "completed"``.
        """
        data = await self.poll_job("/api/content/videos/jobs", job_id, timeout=timeout)
        return AcgVideoJobStatus.model_validate(data)

    async def get_generation_result(self, job_id: str) -> AcgVideoGenerationResult:
        """Get the FAT result (video_url + actual_duration_s) for a completed job.

        The downstream ``/result`` route 409s unless the job is fully
        ``completed``; only call this after :meth:`wait_for_generation` reports
        that status.
        """
        data = await self._request("GET", f"/api/content/videos/jobs/{job_id}/result")
        return AcgVideoGenerationResult.model_validate(data)

    async def create_concept_image(
        self,
        prompt: str,
        config: dict | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> str:
        """Create a concept-image job and return the downstream job_id.

        ``idempotency_key`` (sent as the Idempotency-Key header) makes a
        re-submit of the SAME image — e.g. ``ai_image_gen`` retrying its
        per-item callable after a transient wait failure — return the ORIGINAL
        job instead of spawning a SECOND paid provider generation (ACG dedupes
        on it server-side). Without it, the re-submit double-bills.

        When the bound run pinned a TENANT ``image`` connection, its resolved slot
        rides as ``provider_slot`` (P5 contract); otherwise the body is
        byte-identical to the pre-P5 one.
        """
        payload: dict[str, Any] = {"prompt": prompt}
        if config:
            payload.update(config)
        await _attach_provider_slot(payload, "image")
        return await self.submit_job(
            "/api/concept-images", payload, idempotency_key=idempotency_key
        )

    async def create_concept_image_batch(
        self,
        items: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> AcgConceptImageBatchSubmission:
        """Submit a batch of concept-image requests."""
        payload: dict[str, Any] = {"items": items}
        if config:
            payload.update(config)
        data = await self._request("POST", "/api/concept-images/batches", json=payload)
        return AcgConceptImageBatchSubmission.model_validate(data)

    async def get_concept_image_job(self, job_id: str) -> AcgConceptImageJob:
        """Get current concept-image job status (FAT: carries the asset inline)."""
        data = await self._request("GET", f"/api/concept-images/jobs/{job_id}")
        return AcgConceptImageJob.model_validate(data)

    async def wait_for_concept_image(
        self, job_id: str, timeout: float = 300.0
    ) -> AcgConceptImageJob:
        """Poll until a concept-image job completes.

        Returns the FAT status envelope, which already carries ``asset_id`` /
        ``image_url`` / ``download_url`` — no separate result fetch needed.
        """
        try:
            data = await self.poll_job("/api/concept-images/jobs", job_id, timeout=timeout)
        except RuntimeError as exc:
            # The job failed on modality_gateway's 402: the tenant is byo_required
            # with no image provider. Customer input, not retryable — raised as the
            # structured refusal (``byo_provider_required`` / ``image``).
            if IMAGE_BYO_REFUSAL_MARKER in str(exc):
                raise SlotNotConfiguredError("image", upstream_detail=str(exc)[:500]) from exc
            busy = IMAGE_BUSY_MARKER.search(str(exc))
            if busy is not None:
                raise GpuBusyError(
                    f"image lane stayed busy: {str(exc)[:500]}",
                    service=_SERVICE,
                    status_code=int(busy.group(1)),
                    lane="image",
                ) from exc
            raise
        return AcgConceptImageJob.model_validate(data)

    async def cancel_concept_image(self, job_id: str) -> bool:
        """Cancel a pending or in-flight concept-image job. True when ACG cancelled it.

        ACG answers 409 for an unknown, foreign or already-finished job (one answer
        for all three, so cancel never confirms another tenant's job exists); that
        is returned as False, not raised: the job is not rendering for us either way.
        A cancelled job releases its Idempotency-Key, so a later submit with the
        same key starts a FRESH job. Only cancel when nothing will retry it.
        Other failures raise; the caller decides that a cancel is best-effort.
        """
        try:
            await self._request("POST", f"/api/concept-images/jobs/{job_id}/cancel")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                return False
            raise
        return True

    async def get_concept_image_job_result(self, job_id: str) -> AcgConceptImageAsset:
        """Get the completed result payload for a concept-image job."""
        data = await self._request("GET", f"/api/concept-images/jobs/{job_id}/result")
        return AcgConceptImageAsset.model_validate(data)

    async def get_concept_image_batch(self, batch_id: str) -> AcgConceptImageBatch:
        """Get concept-image batch status and per-item results."""
        data = await self._request("GET", f"/api/concept-images/batches/{batch_id}")
        return AcgConceptImageBatch.model_validate(data)

    async def get_concept_image_batch_result(self, batch_id: str) -> AcgConceptImageBatch:
        """Get the completed result payload for a concept-image batch."""
        data = await self._request("GET", f"/api/concept-images/batches/{batch_id}/result")
        return AcgConceptImageBatch.model_validate(data)

    async def get_concept_image_asset(self, asset_id: str) -> AcgConceptImageAsset:
        """Look up concept-image metadata by stable asset identifier."""
        data = await self._request("GET", f"/api/concept-images/assets/{asset_id}")
        return AcgConceptImageAsset.model_validate(data)

    async def download_concept_image(
        self, url_or_path: str, *, max_bytes: int | None = None
    ) -> bytes:
        """Download concept-image bytes from an ACG download URL/path.

        ACG returns a license-gated, ACG-relative download URL
        (``/api/concept-images/assets/{id}/download``) that the anonymous
        video-compile service cannot fetch. Callers download the bytes here
        (the client carries the licence token) and re-publish them to the run's
        artifact store — the same pattern slide/site-capture assets use.
        """
        return await self._download(url_or_path, max_bytes=max_bytes)

    async def download_generated_video(
        self, url_or_path: str, *, max_bytes: int | None = None
    ) -> bytes:
        """Download generated-video bytes from an ACG download URL/path.

        Video sibling of :meth:`download_concept_image`: ACG returns a
        license-gated, ACG-relative video download path
        (``/api/content/videos/jobs/{id}/download``) the anonymous
        video-compile service cannot fetch. ``ai_video_gen`` downloads the
        bytes here (the client carries the licence token) and re-publishes
        them to the run's artifact store.
        """
        return await self._download(url_or_path, max_bytes=max_bytes)

    async def _download(self, url_or_path: str, *, max_bytes: int | None = None) -> bytes:
        """GET asset bytes under the GPU admission contract (#1106, #1111).

        429/503 → honour ``Retry-After`` (header, then body) within
        ``GPU_BUSY_DEADLINE_SECONDS``, then :class:`GpuBusyError` (transient). Any
        other >= 400 → ``raise_for_status_with_body`` as before.
        """
        client = await self._get_client()
        busy = BusyRetry(_SERVICE, sleep=_sleep)
        while True:
            if max_bytes is None:
                response = await client.get(url_or_path)
                if response.status_code in BUSY_STATUSES:
                    await busy.wait(response)
                    continue
                raise_for_status_with_body(response, _SERVICE)
                return response.content
            # A ceiling means the bytes must never be materialised whole: stream and
            # abort the moment the provider crosses it, so one oversized asset cannot
            # be held in the single orchestrator pod.
            async with client.stream("GET", url_or_path) as streamed:
                if streamed.status_code in BUSY_STATUSES:
                    await streamed.aread()
                    await busy.wait(streamed)
                    continue
                if streamed.status_code >= 400:
                    await streamed.aread()
                raise_for_status_with_body(streamed, _SERVICE)
                chunks: list[bytes] = []
                total = 0
                async for chunk in streamed.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise AssetTooLargeError(f"asset exceeds the {max_bytes} byte ceiling")
                    chunks.append(chunk)
                return b"".join(chunks)
