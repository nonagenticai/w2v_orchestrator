"""Client for the pooling TTS gateway (claim-queue).

The gateway fans audiobook synthesis across many remote kokoro workers. narrate
submits a book as one idempotent batch, then polls until every chunk is ``done``.
Because submit is idempotent and poll is stateless, a reaped+resumed narrate just
re-submits (no-op) and re-polls a batch that kept draining on the workers — true
resumability the single sync box can't give.

``tts_gateway`` is not yet in ``w2v_common.discovery``; the base URL comes from
``settings.downstream.tts_gateway_url`` (a follow-up adds it to discovery).
"""

from __future__ import annotations

import asyncio

import httpx

from src.clients.base import LicensedServiceClient
from src.clients.gpu_busy import BUSY_STATUSES, BusyRetry, byo_refusal
from src.clients.schemas.tts_gateway import BatchStatusResult, SubmitBatchResult
from src.config import get_settings

_REQUEST_TIMEOUT_SECONDS = 60.0

# Indirection so tests patch the busy wait instead of sleeping through it.
_sleep = asyncio.sleep

#: The prefix tts_gateway's worker writes as a job's ``error`` when its synth
#: target (modality_gateway) answered **402** — a ``byo_required`` tenant with no
#: TTS provider. Vendored verbatim from ``tts_gateway/worker/worker.py``
#: ``SynthConfigError`` (tts_gateway#46: reported with ``terminal=true``).
#:
#: ⚠️ PROSE, because it is the only signal: the batch poll exposes a job as
#: ``{status: "failed", error: <str>}`` with no structured code or terminal flag.
#: The real cure is a code field on tts_gateway's ``JobStatus``; until then this
#: prefix is pinned by a test so a reword upstream fails CI here, loudly.
BYO_REFUSAL_ERROR_PREFIX = "synth refused with HTTP 402 (customer configuration)"


def is_byo_refusal(error: str | None) -> bool:
    """True when a failed gateway job's ``error`` is the worker's 402 BYO refusal."""
    return bool(error) and str(error).startswith(BYO_REFUSAL_ERROR_PREFIX)


class TTSGatewayClient(LicensedServiceClient):
    """Submit/poll client for the TTS gateway claim-queue."""

    def __init__(
        self,
        base_url: str | None = None,
        license_token: str | None = None,
        timeout: float = _REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(
            service_name="tts_gateway",
            base_url=base_url or get_settings().downstream.tts_gateway_url,
            license_token=license_token,
            timeout=timeout,
        )

    async def _send(
        self, method: str, path: str, *, json: dict[str, object] | None = None
    ) -> httpx.Response:
        """One gateway call under the GPU admission contract (#1106).

        429/503 → honour ``Retry-After`` within ``GPU_BUSY_DEADLINE_SECONDS``, then
        ``GpuBusyError`` (transient). 402 → ``SlotNotConfiguredError("tts")`` (the
        tenant is ``byo_required``). Any other >= 400 → ``raise_for_status`` as before.
        """
        client = await self._get_client()
        busy = BusyRetry("tts_gateway", sleep=_sleep)
        while True:
            resp = await client.request(method, path, json=json)
            refusal = byo_refusal(resp, "tts")
            if refusal is not None:
                raise refusal
            if resp.status_code in BUSY_STATUSES:
                await busy.wait(resp)
                continue
            resp.raise_for_status()
            return resp

    async def submit_batch(
        self, batch_id: str, chunks: list[dict[str, object]]
    ) -> SubmitBatchResult:
        """Enqueue a book's chunks (idempotent). Returns ``{batch_id, jobs:[...]}``.

        The claim-queue is the PLATFORM TTS: a byo_required tenant is refused here
        (``platform_guard``) before anything is enqueued."""
        from src.services.providers.platform_guard import refuse_platform_for_bound_tenant

        await refuse_platform_for_bound_tenant("tts")
        resp = await self._send(
            "POST", "/api/tts/batches", json={"batch_id": batch_id, "chunks": chunks}
        )
        return SubmitBatchResult.model_validate(resp.json())

    async def get_batch(self, batch_id: str) -> BatchStatusResult:
        """Poll a batch. Returns ``{batch_id, jobs:[{job_id, idempotency_key,
        status, result_url, duration_seconds, error}]}`` with URLs presigned now."""
        resp = await self._send("GET", f"/api/tts/batches/{batch_id}")
        return BatchStatusResult.model_validate(resp.json())
