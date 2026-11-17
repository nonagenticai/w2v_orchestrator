"""Speech-to-text and video-analysis routes — proxies for merqi marketing's recon A/V leg.

merqi marketing's recon called the ASR (``transcription.test:6318``) and
video-analyzer services DIRECTLY with a bundle-minted licence. A tenant API key
(``w2v_sk_``) cannot authenticate there (only the orchestrator verifies keys),
and decision 18 makes the orchestrator the cell's only ingress. These routes
re-expose exactly what recon uses, behind the full ``TENANT_GATE`` (mounted in
``src/asgi.py``) and every auth lane the orchestrator accepts (API key, JWT,
licence):

* ``POST /api/transcribe`` (multipart: ``file`` + ``language`` /
  ``word_timestamps`` / ``return_srt``) → the ASR peer's ``POST /api/transcribe``
  (discovery ``transcription_worker`` — the same peer the orchestrator's own
  transcription stages use); the response body is returned unchanged
  (``text``, ``segments``, ``language``, ``duration``, optional ``srt``, …).
* ``POST /api/analyze`` ``{"video_url"}`` → video-analyzer ``POST /api/analyze``
  (202 ``{"job_id", ...}``); ``GET /api/analyze/{job_id}`` and
  ``GET /api/analyze/{job_id}/result`` → the same paths. video-analyzer scopes
  every job to the delegated tenant (RLS), so a job id of another tenant is a 404.

Outbound credential: ``_proxy.forward_headers`` — the orchestrator's service
licence (or a licence-lane caller's own), NEVER the ``w2v_sk_`` key, plus
``X-Effective-Tenant-Id`` = the resolved tenant (``TenantIdDep``) and, for the
in-cell video-analyzer, a fresh cell token for that tenant. Upstream failures are
mapped (``_proxy.map_upstream_error``), never echoed.
"""

from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field

from src.api.routes._common import TenantIdDep
from src.api.routes._proxy import proxy_request
from src.middleware.proxy_body_limit import TRANSCRIBE_MAX_AUDIO_BYTES as BODY_CAP_AUDIO_BYTES
from src.services.providers.byo_posture import ByoPostureReader, get_byo_posture
from src.services.providers.resolver import refuse_platform_if_byo_required

router = APIRouter(tags=["media-understanding"])


def get_media_byo_posture() -> ByoPostureReader:
    """The licences ``byo_mode`` reader (a dependency so tests can pin a posture)."""
    return get_byo_posture()


ByoPostureDep = Annotated[ByoPostureReader, Depends(get_media_byo_posture)]

TRANSCRIPTION_PEER = "transcription_worker"
VIDEO_ANALYZER_PEER = "video_analyzer"

#: Largest audio upload accepted here. merqi recon caps its own fetch at the
#: same 25 MiB (``RECON_W2V_TRANSCRIPTION_MAX_AUDIO_BYTES``); the ASR peer
#: accepts up to 200 MiB, but this process buffers the body, so it stays small.
#: ``ProxyBodyLimitMiddleware`` enforces it (plus a multipart envelope) as a
#: streaming count before anything is buffered.
TRANSCRIBE_MAX_AUDIO_BYTES = BODY_CAP_AUDIO_BYTES
#: ASR is synchronous: a long clip decodes inside the request. recon waits 120 s.
TRANSCRIBE_TIMEOUT = httpx.Timeout(150.0, connect=5.0)

#: video-analyzer job ids (``uuid4().hex`` / uuid-ish); never a path fragment.
JobId = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,100}$")]


@router.post("/api/transcribe", operation_id="media.transcribe")
async def transcribe(
    tenant_id: TenantIdDep,
    posture: ByoPostureDep,
    file: Annotated[UploadFile, File(description="Audio file to transcribe")],
    language: Annotated[str | None, Form(max_length=16)] = None,
    word_timestamps: Annotated[bool, Form()] = True,
    return_srt: Annotated[bool, Form()] = False,
) -> dict:
    """Transcribe an uploaded audio file (ASR ``POST /api/transcribe``).

    The ASR peer is a PLATFORM GPU (``asr`` is a BYO modality), so the same
    #1106 refusal the run resolver applies to every platform-GPU call runs here
    first: a licences ``byo_required`` tenant gets 402 ``byo_provider_required``
    (modality ``asr``), a posture read failure 503 + ``Retry-After`` — before the
    audio is forwarded or any GPU second is spent. This route has no tenant-connection
    path, so a byo_required tenant cannot be served here at all.
    """
    await refuse_platform_if_byo_required(posture, tenant_id, "asr")
    content = await file.read(TRANSCRIBE_MAX_AUDIO_BYTES + 1)
    if not content:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty audio payload")
    if len(content) > TRANSCRIBE_MAX_AUDIO_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Audio payload exceeds size limit")
    data = {
        "word_timestamps": "true" if word_timestamps else "false",
        "return_srt": "true" if return_srt else "false",
    }
    if language:
        data["language"] = language
    return await proxy_request(
        "POST",
        "/api/transcribe",
        discovery_name=TRANSCRIPTION_PEER,
        effective_tenant=tenant_id,
        timeout=TRANSCRIBE_TIMEOUT,
        files={
            "file": (
                file.filename or "audio.mp3",
                content,
                file.content_type or "application/octet-stream",
            )
        },
        data=data,
    )


class VideoAnalyzeCreate(BaseModel):
    """What recon sends. Deliberately ONLY ``video_url``: video-analyzer's other
    request fields (pipeline/verification config) are not a public surface."""

    model_config = ConfigDict(extra="forbid")

    video_url: str = Field(min_length=1, max_length=2048, pattern=r"^https?://")


@router.post(
    "/api/analyze", operation_id="media.analyze.create", status_code=status.HTTP_202_ACCEPTED
)
async def analyze_video(tenant_id: TenantIdDep, body: VideoAnalyzeCreate) -> dict:
    """Start a video-analysis job (video-analyzer ``POST /api/analyze``)."""
    return await proxy_request(
        "POST",
        "/api/analyze",
        discovery_name=VIDEO_ANALYZER_PEER,
        effective_tenant=tenant_id,
        json=body.model_dump(),
    )


@router.get("/api/analyze/{job_id}", operation_id="media.analyze.status")
async def analyze_status(tenant_id: TenantIdDep, job_id: JobId) -> dict:
    """Poll a video-analysis job (tenant-scoped by video-analyzer)."""
    return await proxy_request(
        "GET",
        f"/api/analyze/{job_id}",
        discovery_name=VIDEO_ANALYZER_PEER,
        effective_tenant=tenant_id,
    )


@router.get("/api/analyze/{job_id}/result", operation_id="media.analyze.result")
async def analyze_result(tenant_id: TenantIdDep, job_id: JobId) -> dict:
    """Fetch a video-analysis job's result (tenant-scoped by video-analyzer)."""
    return await proxy_request(
        "GET",
        f"/api/analyze/{job_id}/result",
        discovery_name=VIDEO_ANALYZER_PEER,
        effective_tenant=tenant_id,
    )
