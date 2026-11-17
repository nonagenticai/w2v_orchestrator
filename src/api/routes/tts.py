"""TTS routes — voice preview for the Settings page.

The orchestrator owns the public ``w2v.nonagentic.ai`` API surface; the Istio
VirtualService routes only ``w2v-frontend → w2v-orchestrator``. The TTS service
(``tts.test``) is reachable only S2S from here. This router re-exposes a
small ``preview`` endpoint so the Settings UI can audition a voice before saving.

It deliberately reuses :class:`TTSClient` — the exact client every narrate stage
uses (``POST /api/tts/synthesize`` → WAV bytes) — so a preview hits the same live
backend narration does, with identity forwarded by the shared ContextVar hook.
"""

from __future__ import annotations

import base64
import binascii
import logging
import math

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.api.routes._common import TenantIdDep
from src.clients.gpu_busy import GpuBusyError
from src.clients.tts_client import TTSClient, TTSSynthesisError
from src.config import get_settings
from src.services.providers.errors import SlotNotConfiguredError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tts", tags=["tts"])

# Bump this when the preview sample text or audio encoding changes so stale
# cached clips are abandoned rather than served. See `_cache_key`.
_PREVIEW_CACHE_VERSION = "v1"

# A short, neutral sample per language. Keyed on the Kokoro voice-id prefix
# (af/am/bf/bm → English, e → Spanish, f → French, h → Hindi, i → Italian,
# j → Japanese, p → Portuguese, z → Chinese). Non-Kokoro providers and unknown
# prefixes fall back to English — the worst case is an English sentence read in
# the voice's accent, which is still a useful audition.
_SAMPLE_EN = "This is a preview of the selected narration voice."
_SAMPLES: dict[str, str] = {
    "e": "Esta es una muestra de la voz de narración seleccionada.",
    "f": "Ceci est un aperçu de la voix de narration sélectionnée.",
    "h": "यह चयनित वर्णन आवाज़ का एक नमूना है।",
    "i": "Questa è un'anteprima della voce di narrazione selezionata.",
    "j": "これは選択したナレーション音声のプレビューです。",
    "p": "Esta é uma prévia da voz de narração selecionada.",
    "z": "这是所选旁白语音的预览。",
}


def _sample_text(voice: str | None) -> str:
    """Pick a sample sentence matching the voice's language, defaulting to English."""
    if not voice:
        return _SAMPLE_EN
    # American/British English voices start with af/am/bf/bm — keep them English.
    if voice[:1] in {"a", "b"}:
        return _SAMPLE_EN
    return _SAMPLES.get(voice[:1], _SAMPLE_EN)


class TTSPreviewRequest(BaseModel):
    """Inbound body for a voice preview. All fields optional — the TTS service
    applies its own provider/voice defaults when omitted."""

    provider: str | None = None
    voice: str | None = None
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


def _cache_key(body: TTSPreviewRequest) -> str:
    """Redis key for a preview clip.

    The sample sentence is fixed per language and carries no tenant data, so a
    given ``(provider, voice, speed)`` always yields the same audio — the cache
    is global, not per-tenant. ``provider`` ``None`` and ``"auto"`` collapse to
    one key since both let the backend pick the same default."""
    provider = body.provider or "auto"
    voice = body.voice or "default"
    return f"tts:preview:{_PREVIEW_CACHE_VERSION}:{provider}:{voice}:{body.speed:.2f}"


@router.post("/preview", operation_id="tts.preview")
async def preview_voice(
    body: TTSPreviewRequest,
    request: Request,
    tenant_id: TenantIdDep,
) -> Response:
    """Return a short audio sample of the given voice for in-browser playback.

    The clip is deterministic in ``(provider, voice, speed)``, so we cache it in
    Redis: a repeat audition of any voice is served instantly without touching
    the single shared TTS box (and without queueing behind live narration). On a
    cache miss we synthesize via the shared :class:`TTSClient`, using a short
    retry budget so a broken backend fails fast (502), while a BUSY one (GPU
    admission 429/503) is waited out within ``tts_preview_busy_budget_seconds``
    honouring ``Retry-After``, then answered 503 + ``Retry-After``.
    """
    settings = get_settings().pipeline
    redis = getattr(request.app.state, "redis", None)
    key = _cache_key(body)

    if redis is not None:
        cached = await redis.get(key)
        if isinstance(cached, dict) and cached.get("audio_base64"):
            try:
                audio_bytes = base64.b64decode(cached["audio_base64"])
            except binascii.Error, ValueError:
                logger.warning("Discarding corrupt cached preview for %s", key)
            else:
                fmt = cached.get("format") or "wav"
                return Response(content=audio_bytes, media_type=f"audio/{fmt}")

    # Reuse the pod-lifetime preview client when wired (amortizes DNS + TCP +
    # TLS to the off-cluster TTS box); fall back to a transient client we close.
    shared = getattr(request.app.state, "tts_preview_client", None)
    client = shared or TTSClient()
    try:
        result = await client.synthesize(
            text=_sample_text(body.voice),
            voice=body.voice,
            provider=body.provider,
            speed=body.speed,
            attempts=settings.tts_preview_retry_attempts,
            busy_deadline_seconds=settings.tts_preview_busy_budget_seconds,
        )
    except TTSSynthesisError as exc:
        raise HTTPException(status_code=502, detail=f"voice preview failed: {exc}") from exc
    except GpuBusyError as exc:
        # The lane stayed full through the whole busy budget: busy, not broken.
        # Pass the backend's own hint on (else 30s) so the UI can retry on time.
        retry = math.ceil(exc.retry_after) if exc.retry_after is not None else 30
        raise HTTPException(
            status_code=503,
            detail=f"voice preview busy: {exc}",
            headers={"Retry-After": str(retry)},
        ) from exc
    except SlotNotConfiguredError as exc:
        raise HTTPException(status_code=402, detail=exc.failure_detail) from exc
    finally:
        if shared is None:
            await client.close()

    audio_bytes = result.audio_bytes
    audio_format = result.format

    if redis is not None:
        await redis.set(
            key,
            {
                "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
                "format": audio_format,
            },
            ttl=settings.tts_preview_cache_ttl_seconds,
        )

    return Response(content=audio_bytes, media_type=f"audio/{audio_format}")
