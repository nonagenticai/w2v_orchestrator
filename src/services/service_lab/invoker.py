"""The service lab: perform ONE real provider call and hand back the media.

The bench exists to answer "does MY GPU service actually work?" — so every slot below
makes the SAME call a run would make, with the SAME resolution rules, and returns the
real bytes (an image, a clip, speech, a transcript, an answer, markdown).

Resolution is the live, outside-a-run path (``POST /api/internal/provider-slots/resolve``
with ``run_id: null``): a tenant-bound session, ``build_resolver_deps``, then
``resolve_slot(..., run_snapshot=None, ...)``. That is the ONLY place ``byo_mode`` is
applied — a ``byo_required`` tenant with no connection raises ``SlotNotConfiguredError``,
``platform_only`` ignores the tenant's connection, and ``local_allowed`` prefers it.
There is deliberately no ``if byo_mode`` in this module.

Tenant vs platform, per slot (never a silent fallback — a tenant that configured its
own provider gets exactly that provider, or the error):

===========  ======================================  =================================
slot         tenant connection                       platform default
===========  ======================================  =================================
image        ACG submit + ``provider_slot``          ACG submit, no ``provider_slot``
video        same, plus a provider-availability gate  same
tts          ``call_tenant_tts`` (in-process)        ``TTSClient().synthesize``
asr          ``call_tenant_asr`` (in-process)        ``TranscriptionWorkerClient``
llm          ``call_tenant_llm`` (in-process)        ``WMClient.message`` dispatch
vlm          ``call_tenant_llm`` + image block       ``WMClient`` + image block
ocr          the tenant's OCR endpoint              ``W2V_PROVIDER_OCR_URL``
===========  ======================================  =================================

Errors are the repo's own: ``SlotNotConfiguredError`` / ``SlotUnavailableError`` /
``OcrNotConfiguredError`` / ``GpuBusyError`` / ``TransientUpstreamError`` … This module
NEVER maps them to HTTP codes and never converts one provider into another; the route
layer owns that translation.

Nothing here commits: ``invoke`` uses the caller's session (which the router binds to
the tenant) and only FLUSHES. Media bytes and the history row are written by
``services.service_lab.store``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from uuid import UUID, uuid7

from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.run_providers import (
    PROVIDER_SLOTS_KEY,
    bind_run_provider_slots,
    clear_run_provider_slots,
)
from src.clients.ai_content_generation_client import (
    AIContentGenerationClient,
    AssetTooLargeError,
)
from src.clients.ocr_client import OcrTarget, platform_target, tenant_target
from src.clients.schemas.ai_content_generation import (
    AcgVideoProviderInfo,
    AcgVideoProviderList,
)
from src.clients.transcription_worker_client import TranscriptionWorkerClient
from src.clients.tts_client import TTSClient
from src.config import get_settings
from src.database.repositories.service_run_repository import (
    DEFAULT_RETENTION_DAYS,
    ServiceRunRepository,
)
from src.database.session import bind_session_tenant
from src.services.concept_images import generate_concept_image
from src.services.llm.dispatch import WMClient
from src.services.llm.in_process import LLMRequest, call_tenant_llm
from src.services.providers.adapters import build_resolver_deps
from src.services.providers.asr_inprocess import ASRRequest, call_tenant_asr
from src.services.providers.errors import SlotUnavailableError
from src.services.providers.resolver import ResolvedSlot, resolve_slot
from src.services.providers.settings import get_provider_policy
from src.services.providers.slot_payload import (
    PROVIDER_SLOT_FIELD,
    slot_payload,
)
from src.services.providers.tts_inprocess import TTSRequest, call_tenant_tts
from src.services.service_lab.ocr import require_target, run_ocr
from src.services.stages.llm_budget import llm_client_timeout_seconds, output_token_budget
from src.services.stages.upstream import TransientUpstreamError
from src.services.visual_media import fetch_image_bytes, fetch_video_bytes, sniff_image_format

__all__ = [
    "SLOTS",
    "InvokeRequest",
    "InvokeResult",
    "ProviderInfo",
    "ServiceInputError",
    "enhance_prompt",
    "enhance_targets",
    "invoke",
]

log = logging.getLogger(__name__)


class ServiceInputError(ValueError):
    """The CALLER's input cannot be run — a wrong file type, a missing prompt.

    Distinct from a bare ``ValueError`` on purpose: the route maps this class to
    **422**, while an unknown slot or enhance target is a defect here and must
    stay a 500. Before this existed, posting an ``image/png`` to ``/asr`` was an
    internal server error with no ``service_runs`` row — "The service call
    failed." for a mistake the user could fix in one click.
    """


#: The bench's slot vocabulary. ``video`` is the ONE word the UI, the API and this
#: module use; the provider registry spells that slot ``video_gen`` (see
#: ``_REGISTRY_SLOT`` below). Never leak the registry spelling to the frontend.
SLOTS: Final = ("image", "tts", "asr", "llm", "vlm", "ocr", "video")

#: Bench slot → provider-registry slot. Anything absent is its own name.
_REGISTRY_SLOT: Final[dict[str, str]] = {"video": "video_gen"}

#: ``WMClient.message`` worker name for a bench call. Not a seeded worker: the LLM
#: path only uses the name for its usage row and its (auto-create) sampling defaults.
_WORKER: Final = "service_lab_bench"

#: The enhance button's worker name (same auto-create defaults — temperature 0.1).
_ENHANCE_WORKER: Final = "service_lab_enhance"

#: ``max_tokens`` for an enhance rewrite: a rewritten prompt, not an essay.
#: Built through ``output_token_budget`` — a literal here is a reasoning model's
#: thinking budget, and a value below the floor is worse than passing nothing.
_ENHANCE_TOKEN_BUDGET: Final = output_token_budget(answer_tokens=600)

#: ``max_tokens`` for a bench answer (llm / vlm). Same rule as above.
_ANSWER_TOKEN_BUDGET: Final = output_token_budget(answer_tokens=1200)

#: The fixed system instruction per enhance target. Each asks for the SAME thing —
#: rewrite the user's prompt, keep the intent, return nothing but the prompt — and
#: differs only in what "better" means for that modality.
_ENHANCE_INSTRUCTIONS: Final[dict[str, str]] = {
    "image": (
        "Rewrite the user's image prompt so a text-to-image model produces a stronger "
        "picture. Keep the user's subject and intent. Make it specific and visually "
        "concrete: subject, setting, composition, camera angle and lens, lighting, "
        "colour palette and artistic style. Return ONLY the rewritten prompt — no "
        "preamble, no quotes, no explanation."
    ),
    "video": (
        "Rewrite the user's video prompt so a text-to-video model produces a stronger "
        "clip. Keep the user's subject and intent. Make it specific and visually "
        "concrete: subject and action, how motion unfolds over time, camera movement "
        "(pan, dolly, handheld), shot framing, pacing and duration, lighting and style. "
        "Return ONLY the rewritten prompt — no preamble, no quotes, no explanation."
    ),
    "llm": (
        "Rewrite the user's prompt so a language model answers it better. Keep the "
        "user's intent. Clarify the task, state the context and constraints explicitly, "
        "and specify the desired output format and length. Return ONLY the rewritten "
        "prompt — no preamble, no quotes, no explanation."
    ),
    "vlm": (
        "Rewrite the user's question about an image so a vision-language model can "
        "answer it from the image alone. Keep the user's intent. Make the question "
        "specific and answerable from what is visible: name the objects, regions and "
        "details to look at, and ask for the exact answer format wanted. Return ONLY "
        "the rewritten question — no preamble, no quotes, no explanation."
    ),
}

#: The targets ``enhance_prompt`` accepts (the router validates against these).
enhance_targets: Final = tuple(_ENHANCE_INSTRUCTIONS)


@dataclass(frozen=True)
class InvokeRequest:
    """One bench call. Every field is optional except the slot (validated by the route)."""

    slot: str
    prompt: str | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    text: str | None = None  # tts
    voice: str | None = None  # tts
    speed: float | None = None  # tts
    language: str | None = None  # asr
    system: str | None = None  # llm
    seconds: int | None = None  # video
    image_bytes: bytes | None = None  # vlm / ocr upload (asr: the audio clip)
    image_content_type: str | None = None
    image_url: str | None = None  # vlm by URL


@dataclass(frozen=True)
class ProviderInfo:
    """Who served the call: the tenant's own connection, or the platform default."""

    source: str  # "tenant" | "deployment"
    kind: str | None
    model: str | None


@dataclass(frozen=True)
class InvokeResult:
    """What the bench shows. ``media_bytes`` is the real payload, inline."""

    output_kind: str  # image|audio|text|transcript|markdown|video
    text: str | None
    media_bytes: bytes | None
    media_content_type: str | None  # "image/png", "audio/wav", "video/mp4"
    provider: ProviderInfo


async def invoke(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    request: InvokeRequest,
    timeout_seconds: float = 180.0,
) -> InvokeResult:
    """Resolve the slot live for ``tenant_id`` and make ONE real provider call.

    Raises the provider layer's own errors (``SlotNotConfiguredError``,
    ``SlotUnavailableError``, ``OcrNotConfiguredError``, ``GpuBusyError``, upstream
    errors) — never an HTTP code: the route maps them.

    ``timeout_seconds`` bounds the WHOLE call (a queued video render included) and
    surfaces as a :class:`TimeoutError` with a sentence, never as a hung request.
    That type is deliberate: the route maps it to 504 ``upstream_timeout``, which a
    generic upstream error would not.
    """
    if request.slot not in SLOTS:
        raise ValueError(f"unknown service-lab slot {request.slot!r}; known: {', '.join(SLOTS)}")
    resolved = await _resolve(session, tenant_id, _REGISTRY_SLOT.get(request.slot, request.slot))
    await _prune_history(session, tenant_id)
    try:
        async with asyncio.timeout(timeout_seconds):
            return await _dispatch(request, resolved, timeout_seconds)
    except TimeoutError as exc:
        raise TimeoutError(
            f"The {request.slot} provider did not answer within {timeout_seconds:.0f}s"
        ) from exc


async def enhance_prompt(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    prompt: str,
    target: str,
    timeout_seconds: float = 60.0,
) -> str:
    """Rewrite ``prompt`` into a better prompt for ``target`` (the "Enhance" button).

    Uses the SAME ``llm`` resolution as :func:`invoke`, so ``byo_mode`` applies: a
    tenant with its own LLM connection enhances on THAT provider, everyone else on
    the platform default — the inference-router when ``LLM_BACKEND=ir``, the vLLM
    when ``LLM_BACKEND=vllm``. Nothing here hardcodes either URL.

    ``IRNotConfiguredError`` (platform LLM has no credentials) and
    ``SlotNotConfiguredError`` (``byo_required`` with no connection) propagate
    UNCHANGED. An empty/blank answer raises :class:`TransientUpstreamError` — the
    button must never quietly replace a prompt with nothing.

    Nothing is persisted: an enhance call is not a run and touches no media.
    """
    if target not in _ENHANCE_INSTRUCTIONS:
        raise ValueError(
            f"unknown enhance target {target!r}; known: {', '.join(_ENHANCE_INSTRUCTIONS)}"
        )
    instruction = _ENHANCE_INSTRUCTIONS[target]
    resolved = await _resolve(session, tenant_id, "llm")
    try:
        async with asyncio.timeout(timeout_seconds):
            if resolved.source == "tenant":
                result = await call_tenant_llm(
                    resolved,
                    LLMRequest(
                        worker_name=_ENHANCE_WORKER,
                        message=prompt,
                        system_prompt=instruction,
                        max_tokens=_ENHANCE_TOKEN_BUDGET,
                    ),
                    timeout=timeout_seconds,
                )
                text = _text_of(result.response.output)
            else:
                response = await WMClient(timeout=llm_client_timeout_seconds()).message(
                    _ENHANCE_WORKER,
                    prompt,
                    system_prompt=instruction,
                    max_tokens=_ENHANCE_TOKEN_BUDGET,
                )
                text = _text_of(response.output)
    except TimeoutError as exc:
        # Re-raised as a TimeoutError on purpose: the route maps that to 504
        # ``upstream_timeout``. Wrapping it in an upstream error would report a
        # timeout as a generic provider failure.
        raise TimeoutError(
            f"The prompt enhancer did not answer within {timeout_seconds:.0f}s"
        ) from exc
    if not text:
        raise TransientUpstreamError("The prompt enhancer returned an empty prompt")
    return text


# ── resolution ──────────────────────────────────────────────────────────────


async def _resolve(session: AsyncSession, tenant_id: UUID, slot: str) -> ResolvedSlot:
    """Live resolution for ``tenant_id`` — the ``/internal/provider-slots`` precedent.

    Tenant-bound session + ``build_resolver_deps`` + ``resolve_slot`` with no run
    snapshot: the tenant's own connection when ``byo_mode`` allows it, else the
    platform default; ``byo_required`` without one raises rather than falling back.
    """
    await bind_session_tenant(session, tenant_id)
    deps = build_resolver_deps(session, get_settings())
    return await resolve_slot(
        deps.repo,
        deps.decryptor,
        tenant_id,
        slot,
        run_snapshot=None,
        policy=deps.policy,
        posture=deps.posture,
    )


async def _prune_history(session: AsyncSession, tenant_id: UUID) -> None:
    """Opportunistic retention: drop rows past the media-cache horizon.

    Same 30-day window as the ``media-cache/`` objects, so a row never outlives its
    media, and no new background loop has to exist for it.

    Best-effort INSIDE A SAVEPOINT: a failed DELETE must not fail a bench call, and
    must not leave the caller's transaction aborted either (the route inserts the
    history row into this same transaction afterwards — a poisoned session would
    turn a working generation into a 500). The tenant GUC is transaction-local and
    was set before the savepoint, so it survives the rollback.
    """
    try:
        async with session.begin_nested():
            await ServiceRunRepository(session).delete_older_than(tenant_id, DEFAULT_RETENTION_DAYS)
    except Exception as exc:  # noqa: BLE001 -- retention never fails the call
        log.warning("service-lab retention prune failed: %s", type(exc).__name__)


# ── dispatch ────────────────────────────────────────────────────────────────


async def _dispatch(
    request: InvokeRequest, resolved: ResolvedSlot, timeout_seconds: float
) -> InvokeResult:
    if request.slot == "image":
        return await _image(request, resolved, timeout_seconds)
    if request.slot == "video":
        return await _video(request, resolved, timeout_seconds)
    if request.slot == "tts":
        return await _tts(request, resolved, timeout_seconds)
    if request.slot == "asr":
        return await _asr(request, resolved, timeout_seconds)
    if request.slot == "llm":
        return await _llm(request, resolved, timeout_seconds)
    if request.slot == "vlm":
        return await _vlm(request, resolved, timeout_seconds)
    if request.slot == "ocr":
        return await _ocr(request, resolved, timeout_seconds)
    raise ValueError(f"unknown service-lab slot {request.slot!r}")  # pragma: no cover


def _provider(resolved: ResolvedSlot, *, kind: str | None = None) -> ProviderInfo:
    """The provider that served a call. ``kind`` overrides the registry kind where the
    downstream service names the actual serving backend (ACG's provider name)."""
    return ProviderInfo(source=resolved.source, kind=kind or resolved.kind, model=resolved.model)


def _text_of(output: object) -> str:
    """A model answer as text, stripped (an empty answer is ``""``, not ``"None"``)."""
    if output is None:
        return ""
    return (output if isinstance(output, str) else str(output)).strip()


def _prompt_of(request: InvokeRequest) -> str:
    prompt = (request.prompt or "").strip()
    if not prompt:
        raise ServiceInputError("this slot needs a prompt")
    return prompt


def _acg_payload(
    request_prompt: str, config: dict[str, object], resolved: ResolvedSlot
) -> dict[str, object]:
    """The ACG body: ``{"prompt", **config}`` plus the tenant ``provider_slot``.

    This is exactly what ``AIContentGenerationClient._attach_provider_slot`` adds for
    a RUN-pinned tenant connection. It is done here because the bench has no run to
    pin: ``build_provider_slot`` reads the run's slot from the dispatch ContextVar and
    would find nothing, silently sending the platform's provider instead of the one
    the tenant configured. The contract shape (and its decrypted secret) comes from
    ``slot_payload`` — never logged, never persisted.
    """
    payload: dict[str, object] = {"prompt": request_prompt, **config}
    if resolved.source == "tenant":
        payload[PROVIDER_SLOT_FIELD] = slot_payload(resolved)
    return payload


# ── image ───────────────────────────────────────────────────────────────────


@contextmanager
def _bench_pin(resolved: ResolvedSlot) -> Iterator[None]:
    """Make the bench's resolved slot look like a RUN pin for one call.

    The shared ACG helpers read the tenant's pin from the dispatch ContextVar
    (``build_provider_slot``), and a bench request has no run behind it: without
    this the call silently uses the PLATFORM provider instead of the connection the
    tenant registered — the exact thing the page exists to test. Cleared on exit; a
    request context carries no other pin to restore.
    """
    bind_run_provider_slots({PROVIDER_SLOTS_KEY: {resolved.ref.slot: resolved.ref}})
    try:
        yield
    finally:
        clear_run_provider_slots()


async def _image(
    request: InvokeRequest, resolved: ResolvedSlot, timeout_seconds: float
) -> InvokeResult:
    """Submit a concept image, wait, download its bytes."""
    client = AIContentGenerationClient()
    config: dict[str, object] = {}
    if request.width and request.height:
        config["size"] = f"{request.width}x{request.height}"
    with _bench_pin(resolved):
        job = await generate_concept_image(
            client,
            _prompt_of(request),
            config,
            idempotency_key=f"service-lab:{uuid7()}",
            budget_s=timeout_seconds,
        )
    url = (job.download_url or job.image_url or job.url or "").strip()
    if job.status != "completed" or not url:
        raise TransientUpstreamError(
            f"The image provider finished with status {job.status!r} and no image"
            + (f": {job.error}" if job.error else "")
        )
    data, content_type = _bounded(await _fetch_media(client, url, kind="image"), kind="image")
    return InvokeResult(
        output_kind="image",
        text=None,
        media_bytes=data,
        media_content_type=content_type,
        provider=_provider(resolved, kind=job.provider),
    )


# ── video ───────────────────────────────────────────────────────────────────


async def _video(
    request: InvokeRequest, resolved: ResolvedSlot, timeout_seconds: float
) -> InvokeResult:
    """Submit a video generation, wait, download its bytes.

    ACG only REGISTERS a video provider whose credentials/health checks passed, so an
    empty list is the truthful "this deployment cannot generate video" answer: refuse
    BEFORE submitting a job that can never be served.
    """
    client = AIContentGenerationClient()
    catalogue = await client.list_video_providers()
    available = [p for p in catalogue.providers if p.available]
    if not available:
        raise SlotUnavailableError(
            "video_gen",
            "no AI video provider is registered on this deployment "
            "(GET /api/content/videos/providers returned none available)",
        )
    config: dict[str, object] = {}
    if request.seconds:
        config["duration_s"] = request.seconds
    if request.width and request.height:
        config["width"] = request.width
        config["height"] = request.height
    if request.seed is not None:
        config["seed"] = request.seed
    job_id = await client.submit_job(
        "/api/content/videos/generate",
        _acg_payload(_prompt_of(request), config, resolved),
        idempotency_key=f"service-lab:{uuid7()}",
    )
    status = await client.wait_for_generation(job_id, timeout=timeout_seconds)
    serving = status.provider or _catalogue_default(catalogue, available)
    if status.status != "completed":
        raise TransientUpstreamError(
            f"The video provider finished with status {status.status!r}"
            + (f": {status.error}" if status.error else "")
        )
    result = await client.get_generation_result(job_id)
    url = (result.video_url or "").strip()
    if not url:
        raise TransientUpstreamError("The video provider completed without a video URL")
    data, content_type = _bounded(await _fetch_media(client, url, kind="video"), kind="video")
    return InvokeResult(
        output_kind="video",
        text=None,
        media_bytes=data,
        media_content_type=content_type,
        provider=_provider(resolved, kind=result.provider or serving),
    )


def _catalogue_default(
    catalogue: AcgVideoProviderList, available: list[AcgVideoProviderInfo]
) -> str:
    """The provider ACG will most likely use when the job envelope names none.

    ``default_provider`` is echoed from ACG's config and may name a provider that
    never registered, so it only wins when it is actually AVAILABLE; otherwise the
    first available provider (ACG's list is in registration order).
    """
    names = [provider.name for provider in available]
    if catalogue.default_provider in names:
        return catalogue.default_provider
    return names[0] if names else "unknown"


# ── tts ─────────────────────────────────────────────────────────────────────


async def _tts(
    request: InvokeRequest, resolved: ResolvedSlot, timeout_seconds: float
) -> InvokeResult:
    """Synthesize speech: in-process on the tenant's connection, else the platform TTS."""
    text = (request.text or request.prompt or "").strip()
    if not text:
        raise ServiceInputError("tts needs text")
    speed = request.speed if request.speed is not None else 1.0
    if resolved.source == "tenant":
        result = await call_tenant_tts(
            resolved,
            TTSRequest(text=text, voice=request.voice, speed=speed),
            timeout=timeout_seconds,
        )
        synthesis = result.synthesis
    else:
        synthesis = await TTSClient().synthesize(text=text, voice=request.voice, speed=speed)
    return InvokeResult(
        output_kind="audio",
        text=None,
        media_bytes=synthesis.audio_bytes,
        media_content_type=f"audio/{(synthesis.format or 'wav').lower()}",
        provider=_provider(resolved),
    )


# ── asr ─────────────────────────────────────────────────────────────────────


async def _asr(
    request: InvokeRequest, resolved: ResolvedSlot, timeout_seconds: float
) -> InvokeResult:
    """Transcribe the uploaded clip.

    The upload arrives in ``image_bytes``/``image_content_type`` (the ONE byte channel
    the frozen request shape has). A browser's upload often arrives as
    ``application/octet-stream`` or ``audio/webm``, so only an ``image/*`` type is
    refused here as the wrong media; anything else is forwarded as declared and the
    provider's own answer is the verdict.
    """
    data = request.image_bytes
    if not data:
        raise ServiceInputError("asr needs an audio upload")
    content_type = (request.image_content_type or "audio/wav").split(";")[0].strip().lower()
    if content_type.startswith("image/"):
        raise ServiceInputError(f"asr needs an audio upload, got {content_type!r}")
    path = _write_temp_audio(data, content_type)
    try:
        if resolved.source == "tenant":
            transcription = await call_tenant_asr(
                resolved,
                ASRRequest(file_path=path, content_type=content_type, language=request.language),
                timeout=timeout_seconds,
            )
        else:
            transcription = await TranscriptionWorkerClient().transcribe_file(
                file_path=path, language=request.language, content_type=content_type
            )
    finally:
        path.unlink(missing_ok=True)
    return InvokeResult(
        output_kind="transcript",
        text=_text_of(transcription.text),
        media_bytes=None,
        media_content_type=None,
        provider=_provider(resolved),
    )


def _write_temp_audio(data: bytes, content_type: str) -> Path:
    """Spool the upload to a temp file: the ASR contract is multipart-from-a-path."""
    suffix = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/wave": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/mp4": ".m4a",
        "audio/ogg": ".ogg",
    }.get(content_type, ".bin")
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)  # noqa: SIM115
    try:
        handle.write(data)
    finally:
        handle.close()
    return Path(handle.name)


# ── llm / vlm ───────────────────────────────────────────────────────────────


async def _llm(
    request: InvokeRequest, resolved: ResolvedSlot, timeout_seconds: float
) -> InvokeResult:
    """Answer a prompt: in-process on the tenant's LLM, else the platform dispatch."""
    prompt = _prompt_of(request)
    response = await _ask_model(
        resolved,
        prompt=prompt,
        content_blocks=None,
        system=request.system,
        timeout_seconds=timeout_seconds,
    )
    return InvokeResult(
        output_kind="text",
        text=_text_of(response),
        media_bytes=None,
        media_content_type=None,
        provider=_provider(resolved),
    )


async def _vlm(
    request: InvokeRequest, resolved: ResolvedSlot, timeout_seconds: float
) -> InvokeResult:
    """Answer a question ABOUT an image: an image content block through the same LLM
    path. The platform has no separate VLM: it is the platform LLM dispatch carrying
    the image block, so its own error surfaces unchanged."""
    # The platform default for this slot is the LLM itself, which can only
    # answer if the served model is vision-capable. On 2026-11-02 it was not
    # (pgx vLLM served a text-only model): the image part came back a 400 and
    # the page told the user the provider had failed. Refuse the platform path
    # instead, as "unavailable" — the same shape video uses for a missing
    # renderer — until an operator serves a vision model and enables it.
    # A TENANT connection is unaffected: that one is the tenant's own claim.
    if resolved.source != "tenant" and not get_provider_policy().platform_vlm_enabled:
        raise SlotUnavailableError(
            "vlm",
            "no vision model is configured on this deployment — the platform "
            "text model cannot read images",
        )
    prompt = _prompt_of(request)
    image = await _image_block(request)
    response = await _ask_model(
        resolved,
        prompt=prompt,
        content_blocks=[image, {"type": "text", "text": prompt}],
        system=request.system,
        timeout_seconds=timeout_seconds,
    )
    return InvokeResult(
        output_kind="text",
        text=_text_of(response),
        media_bytes=None,
        media_content_type=None,
        provider=_provider(resolved),
    )


async def _ask_model(
    resolved: ResolvedSlot,
    *,
    prompt: str,
    content_blocks: list[dict[str, object]] | None,
    system: str | None,
    timeout_seconds: float,
) -> object:
    """One generation on the resolved ``llm``/``vlm`` slot; the model's raw output."""
    if resolved.source == "tenant":
        result = await call_tenant_llm(
            resolved,
            LLMRequest(
                worker_name=_WORKER,
                message=prompt if content_blocks is None else None,
                content_blocks=content_blocks,
                system_prompt=system,
                max_tokens=_ANSWER_TOKEN_BUDGET,
            ),
            timeout=timeout_seconds,
        )
        return result.response.output
    # The platform path a run uses: WMClient dispatches to inference-router or the
    # vLLM by LLM_BACKEND — never a hardcoded URL.
    response = await WMClient(timeout=llm_client_timeout_seconds()).message(
        _WORKER,
        prompt if content_blocks is None else None,
        content_blocks=content_blocks,
        system_prompt=system,
        max_tokens=_ANSWER_TOKEN_BUDGET,
        slot=resolved.ref.slot if resolved.ref.slot == "vlm" else None,
    )
    return response.output


async def _image_block(request: InvokeRequest) -> dict[str, object]:
    """The ``{"type": "image"}`` content block the in-process LLM path consumes."""
    if request.image_bytes:
        media_type = (request.image_content_type or "image/png").split(";")[0].strip().lower()
        if not media_type.startswith("image/"):
            raise ServiceInputError(f"vlm needs an image upload, got {media_type!r}")
        data = request.image_bytes
    elif request.image_url:
        data, ext = await fetch_image_bytes(request.image_url)
        media_type = "image/jpeg" if ext == "jpg" else f"image/{ext}"
    else:
        raise ServiceInputError("vlm needs an image (upload or url)")
    return {
        "type": "image",
        "media_type": media_type,
        "data": base64.b64encode(data).decode("ascii"),
    }


# ── ocr ─────────────────────────────────────────────────────────────────────


async def _ocr(
    request: InvokeRequest, resolved: ResolvedSlot, timeout_seconds: float
) -> InvokeResult:
    """OCR an uploaded image/PDF into markdown — the tenant's endpoint, else the platform's.

    ``W2V_PROVIDER_OCR_URL`` empty AND no tenant connection ⇒ the existing
    ``OcrNotConfiguredError`` ("OCR is not configured").
    """
    settings = get_settings().provider_ocr
    if request.image_bytes:
        data = request.image_bytes
        content_type = (request.image_content_type or "application/pdf").split(";")[0].strip()
    elif request.image_url:
        data, ext = await fetch_image_bytes(request.image_url)
        content_type = "image/jpeg" if ext == "jpg" else f"image/{ext}"
    else:
        raise ServiceInputError("ocr needs a document upload (or a url)")
    target: OcrTarget | None
    if resolved.source == "tenant":
        target = tenant_target(resolved, settings)
    else:
        target = platform_target(settings)
    markdown = await run_ocr(require_target(target), data, content_type, timeout=timeout_seconds)
    return InvokeResult(
        output_kind="markdown",
        text=markdown,
        media_bytes=None,
        media_content_type=None,
        provider=_provider(resolved),
    )


# ── media fetch ─────────────────────────────────────────────────────────────


#: Ceiling on ONE generated asset we will accept from a provider. Uploads are
#: capped at 25 MiB on the way in; without this the same 60 calls/hour can pull a
#: multi-GB asset from a misbehaving (or hostile) tenant provider and buffer it
#: whole in the single orchestrator pod.
MAX_MEDIA_BYTES: Final = 64 * 1024 * 1024


def _bounded(result: tuple[bytes, str], *, kind: str) -> tuple[bytes, str]:
    """Belt-and-braces: the streaming download already refuses an oversized asset.

    Kept because a third-party absolute URL goes through ``fetch_*_bytes``, which the
    bench does not control.
    """
    data, content_type = result
    if len(data) > MAX_MEDIA_BYTES:
        raise TransientUpstreamError(
            f"The {kind} provider returned {len(data)} bytes, over the "
            f"{MAX_MEDIA_BYTES} byte bench ceiling"
        )
    return data, content_type


async def _fetch_media(
    client: AIContentGenerationClient, url: str, *, kind: str
) -> tuple[bytes, str]:
    """Download a generated asset: ACG's own relative path through the ACG client
    (it carries the licence), an absolute third-party URL through the SSRF-guarded
    media sink."""
    if url.startswith("/"):
        try:
            data = (
                await client.download_concept_image(url, max_bytes=MAX_MEDIA_BYTES)
                if kind == "image"
                else await client.download_generated_video(url, max_bytes=MAX_MEDIA_BYTES)
            )
        except AssetTooLargeError as exc:
            raise TransientUpstreamError(str(exc)) from exc
        return data, _sniff_content_type(data, kind)
    if kind == "image":
        data, ext = await fetch_image_bytes(url)
        return data, "image/jpeg" if ext == "jpg" else f"image/{ext}"
    data, ext = await fetch_video_bytes(url)
    return data, _VIDEO_CONTENT_TYPES.get(ext, "video/mp4")


_VIDEO_CONTENT_TYPES: Final[dict[str, str]] = {
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mov": "video/quicktime",
    "ogv": "video/ogg",
}


def _sniff_content_type(data: bytes, kind: str) -> str:
    """The content type of downloaded bytes, by magic — the URL lies often enough.

    ACG's download path carries no trustworthy extension, and the bench shows the
    bytes back in the browser, so guessing wrong means a broken tile.
    """
    if kind == "video":
        if data[4:12].startswith(b"ftyp"):
            return "video/mp4"
        if data.startswith(b"\x1a\x45\xdf\xa3"):
            return "video/webm"
        return "video/mp4"
    fmt = sniff_image_format(data)
    if fmt == "png":
        return "image/png"
    if fmt == "jpg":
        return "image/jpeg"
    if fmt == "webp":
        return "image/webp"
    if fmt == "gif":
        return "image/gif"
    return "image/png"
