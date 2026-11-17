"""Service lab — ad-hoc image / speech / text / video generation over the provider slots.

One endpoint per registry slot (``image``, ``tts``, ``asr``, ``llm``, ``vlm``,
``ocr``, ``video``) plus a prompt enhancer, and the history + media surface for
what was generated. All of it is a thin HTTP edge over
``src/services/service_lab/``: this module owns the WIRE (validation, tenant
scoping, rate limit, error mapping, commit) and nothing else — no provider
resolution, no upstream calls, no retries, no storage layout. The invoker does
the call; this module writes the media object (``StorageService.put_object``) and
the history row (``service_run_repository.insert_run``).

Routing (bare paths, never ``/api/v1``):

    POST /api/services/image
    POST /api/services/tts
    POST /api/services/llm
    POST /api/services/video
    POST /api/services/asr
    POST /api/services/vlm
    POST /api/services/ocr
    POST /api/services/enhance
    GET  /api/services/{slot}/history
    GET  /api/services/{slot}/history/{run_id}
    GET  /api/services/{slot}/history/{run_id}/media

Access posture: ``TenantIdDep`` only — a plain MEMBER may generate; the admin
tier is NOT required. The ``(tenant)`` page gates admins in the UI, and the
resolver in the service layer already enforces the tenant's BYO/platform posture
on the way out. Mounting this router under ``TENANT_GATE`` (``asgi.py``) is what
binds RLS and refuses an unknown/unentitled tenant.

WHY THE ERRORS ARE ``JSONResponse`` AND NOT ``HTTPException``
-------------------------------------------------------------
The app's error envelope renders an ``HTTPException`` as
``{"error": {"code": "HTTP_<status>", ...}}`` (``src/api/error_details.py``) —
the app-wide ``code`` is derived from the STATUS, so the semantic code
(``provider_unavailable``, ``SERVICE_LAB_RATE_LIMITED``…) would be lost and the
frontend would have to branch on message text. ``src/api/byo_errors.py`` already
solves this for the BYO refusals by returning a raw response with
``{"detail": {...}}``; this module keeps the whole service-lab surface in that
same shape so one frontend branch covers all of it.

WHY THE ROUTE COMMITS
---------------------
``get_db_session`` commits in its post-``yield`` teardown, which FastAPI runs
AFTER the response is sent (see the repo CLAUDE.md). A caller's very next request
(``GET …/history/{id}``, ``GET …/media``) can therefore 404 on the id it was just
handed. Every mutating path below commits before it returns — including the
FAILURE path, so the history shows what was attempted.

The media read is the mirror image: it commits BEFORE the body streams, so the
pgbouncer server slot is released for the length of a video download instead of
being held ``idle in transaction`` (the same defect ``routes/artifact_proxy.py``
documents at length).

VIDEO'S MISSING-RENDERER ANSWER
-------------------------------
"Called the video slot and this deployment has no renderer" is a 409
``provider_unavailable`` — the same code as ``SlotUnavailableError`` — NOT the
BYO 402 every other slot returns for ``SlotNotConfiguredError``. The video page
keys its "no renderer" state off exactly that pair, and a 402 would send the
user to a connection screen that cannot help.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID, uuid7

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.minio_stream import stream_object
from src.api.routes._common import SessionDep, SubjectIdDep, TenantIdDep
from src.api.schemas import (
    EnhancePromptRequest,
    EnhancePromptResponse,
    ImageGenerateRequest,
    LlmGenerateRequest,
    ServiceLabSlot,
    ServiceProviderView,
    ServiceRunHistory,
    ServiceRunView,
    TtsGenerateRequest,
    VideoGenerateRequest,
)
from src.cache.raw import raw_redis
from src.clients.gpu_busy import GpuBusyError
from src.clients.ocr_client import OcrError, OcrNotConfiguredError
from src.config import get_settings
from src.database.models import ServiceRun
from src.database.repositories.service_run_repository import get_run, insert_run, list_runs
from src.middleware.proxy_body_limit import SERVICE_LAB_MAX_UPLOAD_BYTES
from src.services.providers.errors import SlotNotConfiguredError, SlotUnavailableError
from src.services.providers.rate_limit import ProviderTestRateLimiter
from src.services.providers.settings import get_provider_policy
from src.services.service_lab.invoker import (
    InvokeRequest,
    InvokeResult,
    ServiceInputError,
    enhance_prompt,
    invoke,
)
from src.services.stages.upstream import PermanentUpstreamError, TransientUpstreamError
from src.services.tts_storage import _get_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/services", tags=["service-lab"])

#: Redis namespace for the service-lab budget. Shares the limiter implementation
#: with the live-provider-test budget but NOT its key: a generation is a much
#: bigger spend than a connectivity probe, and the two must not drain each other.
RATE_LIMIT_PREFIX = "service-lab-rl"

#: Nothing this edge accepts carries a real file bigger than a page scan or a
#: short clip. 25 MiB is the whole-file cap; past it the upload is refused with
#: 413 rather than buffered and OOM'd.
#: ONE definition, shared with ``ProxyBodyLimitMiddleware`` so the size that stops a
#: body being spooled and the size the route enforces can never drift apart.
MAX_UPLOAD_BYTES = SERVICE_LAB_MAX_UPLOAD_BYTES

#: Per-slot deadlines handed to the service layer. Video is the only long one —
#: a render queues behind the shared GPU lane.
_TIMEOUT_SECONDS: dict[str, float] = {
    "image": 180.0,
    "video": 300.0,
    "tts": 120.0,
    "asr": 120.0,
    "llm": 120.0,
    "vlm": 120.0,
    "ocr": 120.0,
}
_ENHANCE_TIMEOUT_SECONDS = 60.0

#: ``Retry-After`` when the service layer's busy answer carries no hint of its own.
_DEFAULT_BUSY_RETRY_AFTER = 30

#: Extension per stored media type. Anything unmapped is stored as ``.bin`` — the
#: served Content-Type comes from the ROW, so an odd type is still served right;
#: the extension only keeps the object key readable.
_EXT_BY_CONTENT_TYPE: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "video/mp4": "mp4",
    "video/webm": "webm",
}

#: ``InvokeRequest`` fields echoed into the persisted row. The uploaded bytes are
#: deliberately absent — a 25 MiB blob has no business in a JSONB column, and the
#: echo is what the history UI renders.
_ECHOED_FIELDS: tuple[str, ...] = (
    "prompt",
    "width",
    "height",
    "seed",
    "text",
    "voice",
    "speed",
    "language",
    "system",
    "seconds",
    "image_content_type",
    "image_url",
)

#: What a 502 says. NEVER the upstream's own body: a provider's error text can
#: echo the tenant's key or request payload, and this response is browser-facing.
_UPSTREAM_FAILED_MESSAGE = "The provider could not complete the request. Try again shortly."


# ─── Rate limit ──────────────────────────────────────────────────────────


def get_service_lab_limiter(request: Request) -> ProviderTestRateLimiter:
    """Per-tenant generation budget, on the app's Redis (``app.state.redis``).

    Resolved exactly like ``connections.get_provider_test_limiter``: the app may
    hold the w2v_common wrapper (prod) or a raw client (tests); ``raw_redis``
    normalises both, and a never-connected wrapper is treated as "no Redis" —
    which ALLOWS the call (abuse damping is not an authorization boundary).
    """
    redis = getattr(request.app.state, "redis", None)
    raw = None
    if redis is not None:
        try:
            raw = raw_redis(redis)
        except RuntimeError:  # wrapper never connected — same as no Redis
            raw = None
    policy = get_provider_policy()
    return ProviderTestRateLimiter(
        raw,
        limit=policy.service_lab_rate_limit,
        window_seconds=policy.service_lab_rate_window_seconds,
        key_prefix=RATE_LIMIT_PREFIX,
    )


LimiterDep = Annotated[ProviderTestRateLimiter, Depends(get_service_lab_limiter)]


# ─── Error mapping ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Refusal:
    """A service-layer failure as it reaches the caller AND the history row."""

    status_code: int
    code: str
    message: str
    headers: Mapping[str, str] | None = None
    #: The full wire body when it is richer than ``{code, message}`` (the BYO 402).
    detail: Mapping[str, object] | None = None

    def body(self) -> dict[str, object]:
        if self.detail is not None:
            return dict(self.detail)
        return {"code": self.code, "message": self.message}

    def response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code,
            content={"detail": self.body()},
            headers=dict(self.headers or {}),
        )


def _is_timeout(exc: BaseException) -> bool:
    """True when ``exc`` IS a timeout or was RAISED FROM one.

    ``invoker.invoke`` bounds the call with ``asyncio.timeout`` and re-raises the
    expiry as ``TransientUpstreamError(...) from exc``, so the bare type check is
    not enough — the cause chain is what carries the timeout through. Walking it
    structurally beats matching the message text.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, TimeoutError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def map_service_error(exc: BaseException, *, slot: str) -> _Refusal | None:
    """Translate a service-layer failure into the ONE HTTP answer for this edge.

    Returns ``None`` for an exception this edge does not own (a genuine bug) —
    the caller re-raises it and the app's own 500 handling takes over. Never
    let a mapping fall through to a generic 500: a provider outage is not a
    defect here, and "internal server error" tells the user nothing.
    """
    if isinstance(exc, ServiceInputError):
        # The CALLER's mistake — a PNG posted to /asr, an empty prompt. 422, not
        # a 500: the message names what to change. Only this class is mapped;
        # a bare ValueError (unknown slot / enhance target) is a defect here and
        # must keep surfacing as one. It still persists a failed run through the
        # normal refusal path, so the history shows what was attempted.
        return _Refusal(status_code=422, code="invalid_service_input", message=str(exc))
    if isinstance(exc, SlotNotConfiguredError):
        # EVERY slot, video included: this error means "you must register your own
        # connection", and answering it with "no renderer on this deployment" would
        # send a byo_required tenant looking for a deployment fix that does not exist.
        # The genuine no-renderer case raises SlotUnavailableError (mapped below).
        # The exact body ``src/api/byo_errors.py`` returns for a run-start refusal.
        return _Refusal(
            status_code=402,
            code=str(exc.failure_code),
            message=str(exc),
            detail=dict(exc.failure_detail),
        )
    if isinstance(exc, OcrNotConfiguredError):
        return _Refusal(
            status_code=409,
            code="ocr_not_configured",
            message="OCR is not configured on this deployment.",
        )
    # BEFORE the OcrError branch: an OCR transport timeout is wrapped as an
    # OcrError, and "the provider did not answer in time" is the truer answer.
    if _is_timeout(exc):
        return _Refusal(
            status_code=504,
            code="upstream_timeout",
            message="The provider did not answer in time.",
        )
    if isinstance(exc, OcrError):
        # OcrNotConfiguredError is handled above; anything else is the provider
        # failing after its own retries — map it like every other upstream fault so
        # the failure is also persisted instead of escaping as a bare 500.
        return _Refusal(status_code=502, code="upstream_failed", message=_UPSTREAM_FAILED_MESSAGE)
    if isinstance(exc, SlotUnavailableError):
        return _Refusal(status_code=409, code="provider_unavailable", message=str(exc))
    if isinstance(exc, GpuBusyError):
        retry_after = int(exc.retry_after) if exc.retry_after else _DEFAULT_BUSY_RETRY_AFTER
        return _Refusal(
            status_code=503,
            code="provider_busy",
            message="The provider is busy. Try again shortly.",
            headers={"Retry-After": str(max(1, retry_after))},
        )
    if isinstance(exc, PermanentUpstreamError | TransientUpstreamError):
        return _Refusal(status_code=502, code="upstream_failed", message=_UPSTREAM_FAILED_MESSAGE)
    return None


# ─── Views + request echo ───────────────────────────────────────────────


def _request_echo(request: InvokeRequest) -> dict[str, object]:
    """The sanitised request record for ``ServiceRun.request``.

    Only the caller's own inputs, only the non-empty ones, and never the upload
    itself (its SIZE is recorded instead — enough to explain a run, useless as a
    copy of the file).
    """
    echo: dict[str, object] = {"slot": request.slot}
    for name in _ECHOED_FIELDS:
        value = getattr(request, name, None)
        if value is not None and value != "":
            echo[name] = value
    if request.image_bytes is not None:
        echo["upload_bytes"] = len(request.image_bytes)
    return echo


def _provider_dict(provider: object) -> dict[str, object]:
    """``{"source","kind","model"}`` from a ``ProviderInfo`` OR a stored mapping.

    The dataclass is NOT a ``Mapping``, so a single ``isinstance`` branch would
    silently drop every fresh run's provider — the exact bug this helper exists
    to prevent. Only the three known keys travel: JSONB may hold more later.
    """
    if isinstance(provider, Mapping):
        return {
            "source": provider.get("source"),
            "kind": provider.get("kind"),
            "model": provider.get("model"),
        }
    if provider is None:
        return {}
    return {
        "source": getattr(provider, "source", None),
        "kind": getattr(provider, "kind", None),
        "model": getattr(provider, "model", None),
    }


def _text_value(value: object) -> str | None:
    return None if value is None else str(value)


def _provider_view(provider: object) -> ServiceProviderView:
    """The persisted provider record as the wire model."""
    data = _provider_dict(provider)
    return ServiceProviderView(
        source=_text_value(data["source"]),
        kind=_text_value(data["kind"]),
        model=_text_value(data["model"]),
    )


def _media_url(slot: str, run_id: object) -> str:
    return f"/api/services/{slot}/history/{run_id}/media"


def _view(row: ServiceRun) -> ServiceRunView:
    """The wire projection of a persisted run. Pure — reads the row, writes nothing."""
    error = row.error if isinstance(row.error, Mapping) else None
    return ServiceRunView(
        id=row.id,
        slot=row.slot,
        status=row.status,
        created_at=row.created_at,
        duration_ms=row.duration_ms,
        provider=_provider_view(row.provider),
        request=dict(row.request or {}),
        output_kind=row.output_kind,
        text=row.text_output,
        media_url=_media_url(row.slot, row.id) if row.media_object_key else None,
        media_content_type=row.media_content_type,
        error=(
            {
                "code": str(error.get("code", "upstream_failed")),
                "message": str(error.get("message", "")),
            }
            if error is not None
            else None
        ),
    )


# ─── Uploads ─────────────────────────────────────────────────────────────


async def _read_upload(file: UploadFile) -> bytes:
    """The uploaded bytes, capped at :data:`MAX_UPLOAD_BYTES` (413 past it).

    Reads at most cap+1 bytes, so an oversized upload is refused without ever
    being fully materialised — the cap has to bound MEMORY, not just say no.
    """
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. The limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB.",
        )
    return data


# ─── Persistence ─────────────────────────────────────────────────────────


def _media_key(tenant_id: UUID, run_id: str, content_type: str | None) -> str:
    """``media-cache/tenants/{tenant}/service-lab/run/{xx}/{run_id}.{ext}``.

    Inside the ``media-cache/`` prefix the bucket lifecycle rule already expires,
    so the objects and the rows share one horizon. The two-character shard
    directory keeps a tenant with thousands of runs from putting them all in one
    listing prefix.
    """
    base = (content_type or "").split(";", 1)[0].strip().lower()
    ext = _EXT_BY_CONTENT_TYPE.get(base, "bin")
    return f"media-cache/tenants/{tenant_id}/service-lab/run/{run_id[:2]}/{run_id}.{ext}"


async def _store_media(tenant_id: UUID, run_id: str, data: bytes, content_type: str | None) -> str:
    """Write the generated bytes with the orchestrator's own credentials; return the key."""
    storage = await _get_storage()
    put = getattr(storage, "put_object", None)
    if put is None:  # a storage backend without the write surface
        raise RuntimeError("storage backend has no put_object")
    media_type = content_type or "application/octet-stream"
    key = _media_key(tenant_id, run_id, media_type)
    await put(key, data, media_type)
    return key


async def _persist_failure(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    subject_id: UUID,
    request: InvokeRequest,
    refusal: _Refusal,
    duration_ms: int,
) -> None:
    """Record the attempt so the history explains a failure the caller just saw.

    Best effort BY DESIGN: the service-layer exception may have left the
    transaction aborted, so the rollback comes first, and if the insert still
    fails we log and let the mapped refusal through — losing a history row must
    never turn a 409 into a 500.
    """
    try:
        await session.rollback()
        await insert_run(
            session,
            tenant_id=tenant_id,
            subject_id=subject_id,
            slot=request.slot,
            status="failed",
            request=_request_echo(request),
            duration_ms=duration_ms,
            error={"code": refusal.code, "message": refusal.message},
        )
        await session.commit()
    except Exception:
        logger.exception("service lab: could not persist the failed %s run", request.slot)


# ─── The one execution path ──────────────────────────────────────────────


async def _execute(
    *,
    session: AsyncSession,
    tenant_id: UUID,
    subject_id: UUID,
    request: InvokeRequest,
    limiter: ProviderTestRateLimiter,
) -> ServiceRunView | JSONResponse:
    """Rate limit → invoke → store → persist → commit, or a mapped refusal.

    Every generation endpoint funnels through here so the budget, the timeout
    table, the commit-before-respond rule and the error mapping have exactly ONE
    implementation.
    """
    decision = await limiter.hit(tenant_id)
    if not decision.allowed:
        return _Refusal(
            status_code=429,
            code="SERVICE_LAB_RATE_LIMITED",
            message="Too many generations. Try again later.",
            headers={"Retry-After": str(decision.retry_after)},
        ).response()

    slot = request.slot
    started = time.perf_counter()
    try:
        result: InvokeResult = await invoke(
            session,
            tenant_id=tenant_id,
            request=request,
            timeout_seconds=_TIMEOUT_SECONDS[slot],
        )
    except Exception as exc:
        refusal = map_service_error(exc, slot=slot)
        if refusal is None:
            raise
        await _persist_failure(
            session,
            tenant_id=tenant_id,
            subject_id=subject_id,
            request=request,
            refusal=refusal,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        return refusal.response()

    duration_ms = int((time.perf_counter() - started) * 1000)

    # The media key is derived from the run id, so the id is minted here and both
    # the object and the row carry it. The object goes first: a row that pointed
    # at an absent object would render a permanently broken <img>.
    run_id = uuid7()
    media_key: str | None = None
    if result.media_bytes is not None:
        try:
            media_key = await _store_media(
                tenant_id, str(run_id), result.media_bytes, result.media_content_type
            )
        except Exception:
            logger.exception("service lab: storing the %s media failed", slot)
            refusal = _Refusal(
                status_code=502,
                code="upstream_failed",
                message="The generated media could not be stored.",
            )
            await _persist_failure(
                session,
                tenant_id=tenant_id,
                subject_id=subject_id,
                request=request,
                refusal=refusal,
                duration_ms=duration_ms,
            )
            return refusal.response()

    row = await insert_run(
        session,
        run_id=run_id,
        tenant_id=tenant_id,
        subject_id=subject_id,
        slot=slot,
        status="succeeded",
        output_kind=result.output_kind,
        request=_request_echo(request),
        text_output=result.text,
        media_object_key=media_key,
        media_content_type=result.media_content_type if media_key else None,
        duration_ms=duration_ms,
        provider=_provider_dict(result.provider),
    )

    # Build the response BEFORE the commit: a prod session has expire_on_commit
    # on, so touching the row afterwards would re-open a transaction the caller
    # is about to be handed a response for.
    view = _view(row)
    await session.commit()
    return view


# ─── Request builders ────────────────────────────────────────────────────


def _invoke_request(
    slot: str,
    *,
    prompt: str | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    text: str | None = None,
    voice: str | None = None,
    speed: float | None = None,
    language: str | None = None,
    system: str | None = None,
    seconds: int | None = None,
    image_bytes: bytes | None = None,
    image_content_type: str | None = None,
    image_url: str | None = None,
) -> InvokeRequest:
    """One place that names every field of the frozen ``InvokeRequest``."""
    return InvokeRequest(
        slot=slot,
        prompt=prompt,
        width=width,
        height=height,
        seed=seed,
        text=text,
        voice=voice,
        speed=speed,
        language=language,
        system=system,
        seconds=seconds,
        image_bytes=image_bytes,
        image_content_type=image_content_type,
        image_url=image_url,
    )


# ─── Generation ──────────────────────────────────────────────────────────


@router.post("/image", operation_id="services.image", response_model=ServiceRunView)
async def generate_image(
    body: ImageGenerateRequest,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    limiter: LimiterDep,
) -> ServiceRunView | JSONResponse:
    """Generate an image from a prompt (``image`` slot)."""
    return await _execute(
        session=session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        request=_invoke_request(
            "image",
            prompt=body.prompt,
            width=body.width,
            height=body.height,
            seed=body.seed,
        ),
        limiter=limiter,
    )


@router.post("/tts", operation_id="services.tts", response_model=ServiceRunView)
async def generate_tts(
    body: TtsGenerateRequest,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    limiter: LimiterDep,
) -> ServiceRunView | JSONResponse:
    """Synthesize speech from text (``tts`` slot)."""
    return await _execute(
        session=session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        request=_invoke_request("tts", text=body.text, voice=body.voice, speed=body.speed),
        limiter=limiter,
    )


@router.post("/llm", operation_id="services.llm", response_model=ServiceRunView)
async def generate_llm(
    body: LlmGenerateRequest,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    limiter: LimiterDep,
) -> ServiceRunView | JSONResponse:
    """Complete a prompt with the tenant's language model (``llm`` slot)."""
    return await _execute(
        session=session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        request=_invoke_request("llm", prompt=body.prompt, system=body.system),
        limiter=limiter,
    )


@router.post("/video", operation_id="services.video", response_model=ServiceRunView)
async def generate_video(
    body: VideoGenerateRequest,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    limiter: LimiterDep,
) -> ServiceRunView | JSONResponse:
    """Generate a short video clip (``video`` slot, 300 s deadline)."""
    return await _execute(
        session=session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        request=_invoke_request("video", prompt=body.prompt, seconds=body.seconds),
        limiter=limiter,
    )


@router.post("/asr", operation_id="services.asr", response_model=ServiceRunView)
async def transcribe_audio(
    file: Annotated[UploadFile, File()],
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    limiter: LimiterDep,
    language: Annotated[str | None, Form()] = None,
) -> ServiceRunView | JSONResponse:
    """Transcribe an audio upload (``asr`` slot)."""
    audio = await _read_upload(file)
    return await _execute(
        session=session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        request=_invoke_request(
            "asr",
            image_bytes=audio,
            image_content_type=file.content_type,
            language=language,
        ),
        limiter=limiter,
    )


@router.post("/vlm", operation_id="services.vlm", response_model=ServiceRunView)
async def describe_image(
    prompt: Annotated[str, Form()],
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    limiter: LimiterDep,
    file: Annotated[UploadFile | None, File()] = None,
    image_url: Annotated[str | None, Form()] = None,
) -> ServiceRunView | JSONResponse:
    """Answer a prompt about an image — uploaded OR by URL, exactly one."""
    has_file = file is not None
    has_url = bool(image_url and image_url.strip())
    if has_file == has_url:
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of 'file' or 'image_url'.",
        )
    image = await _read_upload(file) if file is not None else None
    return await _execute(
        session=session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        request=_invoke_request(
            "vlm",
            prompt=prompt,
            image_bytes=image,
            image_content_type=file.content_type if file is not None else None,
            image_url=image_url.strip() if has_url else None,
        ),
        limiter=limiter,
    )


@router.post("/ocr", operation_id="services.ocr", response_model=ServiceRunView)
async def extract_text(
    file: Annotated[UploadFile, File()],
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    limiter: LimiterDep,
) -> ServiceRunView | JSONResponse:
    """Extract text from a document or image upload (``ocr`` slot)."""
    document = await _read_upload(file)
    return await _execute(
        session=session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        request=_invoke_request("ocr", image_bytes=document, image_content_type=file.content_type),
        limiter=limiter,
    )


@router.post("/enhance", operation_id="services.enhance", response_model=EnhancePromptResponse)
async def enhance(
    body: EnhancePromptRequest,
    session: SessionDep,
    tenant_id: TenantIdDep,
    limiter: LimiterDep,
) -> EnhancePromptResponse | JSONResponse:
    """Rewrite a prompt for the target modality. Persists no run — nothing is generated."""
    decision = await limiter.hit(tenant_id)
    if not decision.allowed:
        return _Refusal(
            status_code=429,
            code="SERVICE_LAB_RATE_LIMITED",
            message="Too many generations. Try again later.",
            headers={"Retry-After": str(decision.retry_after)},
        ).response()
    try:
        rewritten = await enhance_prompt(
            session,
            tenant_id=tenant_id,
            prompt=body.prompt,
            target=body.target,
            timeout_seconds=_ENHANCE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        refusal = map_service_error(exc, slot=body.target)
        if refusal is None:
            raise
        return refusal.response()
    return EnhancePromptResponse(prompt=rewritten)


# ─── History ─────────────────────────────────────────────────────────────


@router.get(
    "/{slot}/history",
    operation_id="services.history",
    response_model=ServiceRunHistory,
)
async def list_history(
    slot: ServiceLabSlot,
    session: SessionDep,
    tenant_id: TenantIdDep,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> ServiceRunHistory:
    """This tenant's runs for ``slot``, newest first.

    Tenant scoping is the repository's ``tenant_id`` filter AND RLS (the router is
    mounted under ``TENANT_GATE``): a caller can never page into another tenant's
    history by asking for it.
    """
    rows = await list_runs(session, tenant_id, slot, limit)
    return ServiceRunHistory(items=[_view(row) for row in rows])


@router.get(
    "/{slot}/history/{run_id}",
    operation_id="services.run",
    response_model=ServiceRunView,
)
async def get_history_run(
    slot: ServiceLabSlot,
    run_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> ServiceRunView:
    """One run. Another tenant's (or another slot's) id is a 404, never a 403."""
    row = await get_run(session, tenant_id, slot, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _view(row)


@router.get("/{slot}/history/{run_id}/media", operation_id="services.media")
async def get_history_media(
    slot: ServiceLabSlot,
    run_id: UUID,
    request: Request,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> Response:
    """Stream the run's stored bytes with the orchestrator's own credentials.

    Loaded straight by ``<img>`` / ``<audio>`` / ``<video>`` markup, which cannot
    set ``X-Active-Tenant-Id`` — the scope-switch therefore rides ``?active_tenant=``,
    which ``get_tenant_id`` already reads (each/header are the same chokepoint;
    see ``routes/artifact_proxy.py``). Nothing needs declaring here: the query
    parameter is consumed by the dependency, not by this handler.

    ``media_object_key`` comes FROM THE ROW — never from the caller — and the row
    was looked up tenant-scoped, so owning a run is the only way to name an object.
    ``stream_object`` maps a missing object to 404 (never 500) and forwards
    ``Range`` / ``ETag`` / ``Content-Type`` for seeking and revalidation.
    """
    row = await get_run(session, tenant_id, slot, run_id)
    if row is None or not row.media_object_key:
        raise HTTPException(status_code=404, detail="media not found")
    object_key = row.media_object_key
    media_type = row.media_content_type
    bucket = get_settings().storage.minio_bucket

    # END THE TRANSACTION BEFORE THE BODY STREAMS: the lookup above autobegan one,
    # and the dependency's teardown commit runs AFTER the whole response is sent,
    # which would hold the pgbouncer server slot for the length of a video.
    await session.commit()
    return await stream_object(bucket, object_key, request, media_type=media_type)
