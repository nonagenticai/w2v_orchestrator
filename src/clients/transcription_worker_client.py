"""HTTP client for ASR — via modality_gateway ``/api/asr/transcribe`` (#1106).

It used to POST straight to ``asr.test:9201/api/transcribe`` (the
``transcription_worker`` discovery entry), which skipped modality_gateway's
per-tenant BYO check: a ``byo_required`` tenant reached the platform whisper box.
It now speaks the gateway's route, which resolves the tenant's provider (its own
OpenAI Whisper / Deepgram key, a self-hosted URL, or — for a tenant allowed the
platform — a verbatim proxy to ``asr.test``) and answers 402 for a
``byo_required`` tenant without one. Request (multipart ``file`` + form
``language`` / ``word_timestamps`` / ``return_srt``) and response
(``TranscriptionResponse``) are the same contract on both hops.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx
from pydantic import ValidationError

from src.auth.run_providers import current_run_slot
from src.clients.base import LicensedServiceClient
from src.clients.gpu_busy import (
    BUSY_STATUSES,
    BusyRetry,
    byo_refusal,
    get_gpu_admission_settings,
)
from src.clients.schemas.transcription_worker import TranscriptionResult

logger = logging.getLogger(__name__)

#: Provider slot this client serves (``src/services/providers/registry.py``).
ASR_SLOT = "asr"

#: The gateway route (multipart). NOT ``/api/transcribe`` on ``asr.test``.
ASR_TRANSCRIBE_PATH = "/api/asr/transcribe"

# Indirection so tests patch the busy wait instead of sleeping through it.
_sleep = asyncio.sleep

_asr_semaphore: asyncio.Semaphore | None = None


def _asr_gate() -> asyncio.Semaphore:
    """Per-process ceiling on in-flight ASR requests (``ASR_MAX_CONCURRENCY``, default 2).

    Callers own their concurrency under the GPU admission contract; the gate is
    released while a busy answer is waited out, so a sleeper never holds a slot.
    """
    global _asr_semaphore
    if _asr_semaphore is None:
        _asr_semaphore = asyncio.Semaphore(get_gpu_admission_settings().asr_max_concurrency)
    return _asr_semaphore


class TranscriptionWorkerClient(LicensedServiceClient):
    """Async ASR client: ``POST /api/asr/transcribe`` on modality_gateway.

    Transcription is synchronous — the response body is the full result.
    The gateway is behind license-authz like every peer, so this extends
    ``LicensedServiceClient`` (service-account token + identity hooks; the
    gateway is a METERED destination, so the run's tenant is stamped as
    ``X-Effective-Tenant-Id`` and the BYO check applies to THAT tenant).
    The base URL resolves via discovery ``modality_gateway``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        license_token: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(
            service_name="transcription_worker",
            discovery_name="modality_gateway",
            base_url=base_url,
            license_token=license_token,
            timeout=timeout,
            licence_hop="modality_gateway",  # MODALITY_GATEWAY_AUTH_MODE
        )

    async def _post_once(
        self,
        client: httpx.AsyncClient,
        *,
        file_path: Path,
        content_type: str,
        data: dict[str, str],
    ) -> httpx.Response:
        async with _asr_gate():
            with file_path.open("rb") as fh:
                files = {"file": (file_path.name, fh, content_type)}
                try:
                    return await client.post(ASR_TRANSCRIBE_PATH, files=files, data=data)
                except (httpx.HTTPError, httpx.InvalidURL) as exc:
                    # InvalidURL (a misconfigured discovery base_url) is NOT an
                    # httpx.HTTPError — include it so a bad URL surfaces as the
                    # client's RuntimeError, not a raw crash to the caller.
                    raise RuntimeError(f"transcription request failed: {exc}") from exc

    async def transcribe_file(
        self,
        *,
        file_path: Path,
        language: str | None = None,
        word_timestamps: bool = True,
        return_srt: bool = False,
        content_type: str = "audio/mpeg",
    ) -> TranscriptionResult:
        """POST the audio (multipart) and return the transcription result.

        The gateway returns ``TranscriptionResponse``: ``backend``, ``model``,
        ``language``, ``duration``, ``text``, ``segments`` (with per-word timings
        when the serving backend produces them) and ``srt`` when requested.

        429/503 → honour ``Retry-After`` within ``GPU_BUSY_DEADLINE_SECONDS``, then
        ``GpuBusyError`` (transient). 402 → ``SlotNotConfiguredError("asr")``.
        Any other >= 400 → ``RuntimeError`` as before.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"audio file not found: {file_path}")

        # Slot dispatch: a run that pinned a TENANT ``asr`` connection transcribes on
        # it in-process (never falling back to ours); no pin / the deployment pin takes
        # the modality-gateway path below, unchanged.
        ref = current_run_slot(ASR_SLOT)
        if ref is not None and ref.source == "tenant":
            from src.services.providers.asr_inprocess import ASRRequest, transcribe_on_tenant_slot

            return await transcribe_on_tenant_slot(
                ref,
                ASRRequest(
                    file_path=file_path,
                    content_type=content_type,
                    language=language,
                    word_timestamps=word_timestamps,
                    return_srt=return_srt,
                ),
                timeout=self.timeout,
            )

        from src.services.providers.platform_guard import refuse_platform_for_bound_tenant

        await refuse_platform_for_bound_tenant(ASR_SLOT)
        data: dict[str, str] = {
            "word_timestamps": "true" if word_timestamps else "false",
            "return_srt": "true" if return_srt else "false",
        }
        if language is not None:
            data["language"] = language

        client = await self._get_client()
        busy = BusyRetry("asr", sleep=_sleep)
        while True:
            resp = await self._post_once(
                client, file_path=file_path, content_type=content_type, data=data
            )
            refusal = byo_refusal(resp, "asr")
            if refusal is not None:
                raise refusal
            if resp.status_code in BUSY_STATUSES:
                logger.warning(
                    "asr transcribe %d (busy) busy #%d, honouring Retry-After",
                    resp.status_code,
                    busy.busy_count + 1,
                )
                await busy.wait(resp)
                continue
            break

        if resp.status_code >= 400:
            raise RuntimeError(f"transcription returned {resp.status_code}: {resp.text[:500]}")
        try:
            return TranscriptionResult.model_validate(resp.json())
        except (ValueError, ValidationError) as exc:
            # A 2xx with an unparseable body (JSONDecodeError ⊂ ValueError) or schema
            # drift (ValidationError) must surface as the client's RuntimeError, not a
            # raw 500 — this parse used to sit OUTSIDE the request guard.
            raise RuntimeError(f"transcription returned an unparseable body: {exc}") from exc
