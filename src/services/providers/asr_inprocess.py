"""Run one ``TranscriptionWorkerClient.transcribe_file()`` IN-PROCESS on a tenant ``asr`` slot.

Used only when the run pinned a TENANT connection for ``asr``. The caller contract is
``transcribe_file``'s — audio file, language, word_timestamps, return_srt — and the
result is the same :class:`TranscriptionResult` the modality-gateway path returns, so
the caption / audiobook-QA call sites cannot tell the difference.

Provider kinds:

* ``nonagentic_endpoint`` — the PLATFORM protocol at the tenant's ``base_url``:
  ``POST {base_url}/api/asr/transcribe`` with the multipart body the platform client
  sends (``file`` + ``language`` / ``word_timestamps`` / ``return_srt``), plus
  ``Authorization: Bearer <secret>`` when one is stored. The answer is parsed as the
  platform ``TranscriptionResponse``.
* ``openai-compatible`` — ``POST {base_url}/audio/transcriptions`` (OpenAI Whisper API
  shape) with ``response_format=verbose_json`` and, when word timings are wanted,
  ``timestamp_granularities[]=word`` + ``segment``. OpenAI returns words at the TOP
  level; they are folded into the segment whose time span holds them (one synthetic
  segment when the server returned none). ``srt`` is not requested from this kind.

Egress (pinned IP, origin lock, no redirects) and retries are the in-process LLM's
(``llm/in_process``). A failure raises :class:`TransientUpstreamError` /
:class:`PermanentUpstreamError` and NEVER falls back to the modality gateway.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Final

import httpx2
from pydantic import ValidationError
from w2v_common.auth.identity_context import (
    get_actor_id_from_context,
    get_stage_from_context,
    get_subject_id_from_context,
    get_tenant_id_from_context,
    get_w2v_project_id_from_context,
)

from src.clients.schemas.transcription_worker import (
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
)
from src.models.failure_class import CUSTOMER_INPUT, EXTERNAL, FailureClass
from src.services.llm.in_process import _AUTH_HTTP, _TRANSIENT_HTTP, _http_client, sanitise
from src.services.llm.metering import FUNDING_TENANT_BYO, _run_id
from src.services.providers.errors import SlotUnavailableError
from src.services.providers.policy import redact_url
from src.services.providers.registry import (
    ASR_DEFAULT_MODEL,
    KIND_NONAGENTIC_ENDPOINT,
    KIND_OPENAI_COMPATIBLE,
)
from src.services.providers.resolver import PROVIDER_SLOTS_KEY, ResolvedSlot, SlotRef, resolve_slot
from src.services.stages.upstream import PermanentUpstreamError, TransientUpstreamError

log = logging.getLogger(__name__)

SLOT: Final = "asr"
#: The platform transcribe route (modality-gateway), under the tenant base URL.
NONAGENTIC_ASR_PATH: Final = "/api/asr/transcribe"
OPENAI_ASR_PATH: Final = "/audio/transcriptions"


@dataclass(frozen=True, slots=True)
class ASRRequest:
    """The ``transcribe_file`` contract, as the in-process path consumes it."""

    file_path: Path
    content_type: str = "audio/mpeg"
    language: str | None = None
    word_timestamps: bool = True
    return_srt: bool = False


# ── resolution ──────────────────────────────────────────────────────────────


async def resolve_pinned_asr(tenant_id: uuid.UUID, ref: SlotRef) -> ResolvedSlot:
    """Load + validate exactly the pinned ``asr`` version for ``tenant_id``."""
    from src.config import get_settings
    from src.database.session import bind_session_tenant, get_db_session
    from src.services.providers.adapters import build_resolver_deps

    async with get_db_session() as session:
        await bind_session_tenant(session, tenant_id)
        deps = build_resolver_deps(session, get_settings())
        return await resolve_slot(
            deps.repo,
            deps.decryptor,
            tenant_id,
            SLOT,
            run_snapshot={PROVIDER_SLOTS_KEY: {SLOT: ref.to_dict()}},
            policy=deps.policy,
            posture=deps.posture,
        )


def _bound_tenant() -> uuid.UUID:
    raw = get_tenant_id_from_context()
    try:
        return uuid.UUID(str(raw))
    except ValueError, TypeError:
        raise SlotUnavailableError(
            SLOT, "no tenant is bound to this call, so its provider cannot be used"
        ) from None


# ── errors ──────────────────────────────────────────────────────────────────


def _stamp[E: TransientUpstreamError | PermanentUpstreamError](err: E, fault: FailureClass) -> E:
    err.failure_class = fault
    return err


def _where(kind: str) -> str:
    return f"Tenant speech-to-text provider ({kind})"


def _status_error(response: httpx2.Response, *, where: str, secret: str | None) -> Exception:
    status = response.status_code
    if status in _AUTH_HTTP:
        return _stamp(
            PermanentUpstreamError(
                f"{where} rejected the credentials (HTTP {status}). "
                "Update the token on the Providers page."
            ),
            CUSTOMER_INPUT,
        )
    detail = sanitise(response.text, secret)
    if status in _TRANSIENT_HTTP:
        return _stamp(TransientUpstreamError(f"{where} failed (HTTP {status}): {detail}"), EXTERNAL)
    return _stamp(
        PermanentUpstreamError(f"{where} rejected the request (HTTP {status}): {detail}"),
        CUSTOMER_INPUT,
    )


def _malformed(where: str, what: str) -> PermanentUpstreamError:
    return _stamp(
        PermanentUpstreamError(f"{where} returned an unusable response: {what}"), EXTERNAL
    )


# ── response mapping ────────────────────────────────────────────────────────


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _words(raw: object) -> list[TranscriptWord]:
    out: list[TranscriptWord] = []
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        return out
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        start, end, word = _float(entry.get("start")), _float(entry.get("end")), entry.get("word")
        if start is None or end is None or not isinstance(word, str):
            continue
        out.append(
            TranscriptWord(
                start=start, end=end, word=word, probability=_float(entry.get("probability"))
            )
        )
    return out


def _segments(raw: object) -> list[TranscriptSegment]:
    out: list[TranscriptSegment] = []
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        return out
    for i, seg in enumerate(raw):
        if not isinstance(seg, Mapping):
            continue
        start, end = _float(seg.get("start")), _float(seg.get("end"))
        seg_text = seg.get("text")
        if start is None or end is None or not isinstance(seg_text, str):
            continue
        raw_id = seg.get("id")
        out.append(
            TranscriptSegment(
                id=raw_id if isinstance(raw_id, int) and not isinstance(raw_id, bool) else i,
                start=start,
                end=end,
                text=seg_text,
                words=_words(seg.get("words")),
                avg_logprob=_float(seg.get("avg_logprob")),
                no_speech_prob=_float(seg.get("no_speech_prob")),
            )
        )
    return out


def _assign_words(segments: list[TranscriptSegment], words: list[TranscriptWord]) -> None:
    """Fold top-level ``words`` into the segment holding each word's midpoint."""
    for word in words:
        mid = (word.start + word.end) / 2
        target = next((s for s in segments if s.start <= mid <= s.end), None)
        if target is None:
            before = [s for s in segments if s.start <= mid]
            target = before[-1] if before else segments[0]
        target.words.append(word)


def openai_transcript(
    payload: Mapping[str, object], *, model: str, backend: str
) -> TranscriptionResult:
    """OpenAI ``verbose_json`` (or plain ``json``) → the internal transcript shape.

    Top-level ``words`` are assigned to the segment whose ``[start, end]`` holds the
    word's midpoint (else the last segment starting at or before it); segment-level
    ``words`` (faster-whisper servers) are kept as they are. With no segments, one
    synthetic segment spans every word (or the whole duration)."""
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("no text")
    top_words = _words(payload.get("words"))
    segments = _segments(payload.get("segments"))
    duration = _float(payload.get("duration"))
    if not segments and (top_words or text.strip()):
        first = top_words[0].start if top_words else 0.0
        last = top_words[-1].end if top_words else (duration or 0.0)
        segments.append(TranscriptSegment(id=0, start=first, end=last, text=text))
    if top_words and not any(seg.words for seg in segments):
        _assign_words(segments, top_words)
    if duration is None:
        duration = segments[-1].end if segments else 0.0
    language = payload.get("language")
    return TranscriptionResult(
        backend=backend,
        model=model,
        language=language if isinstance(language, str) else None,
        duration=duration,
        text=text,
        segments=segments,
    )


# ── the call ────────────────────────────────────────────────────────────────


def _secret(resolved: ResolvedSlot) -> str | None:
    value = resolved.secret.get_secret_value() if resolved.secret else None
    return value or None


async def _post_multipart(
    resolved: ResolvedSlot,
    path: str,
    request: ASRRequest,
    data: Mapping[str, str | list[str]],
    *,
    timeout: float,
    where: str,
) -> httpx2.Response:
    if not resolved.base_url:
        raise SlotUnavailableError(SLOT, f"kind {resolved.kind!r} requires a base URL")
    secret = _secret(resolved)
    base_url = resolved.base_url.rstrip("/")
    url = f"{base_url}{path}"
    headers = {"Authorization": f"Bearer {secret}"} if secret else {}
    try:
        audio = request.file_path.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"audio file not readable: {request.file_path}") from exc
    files = {"file": (request.file_path.name, audio, request.content_type)}
    try:
        async with _http_client(
            base_url, resolved.pinned_ip, timeout, retry_read_timeout=False
        ) as http:
            response = await http.post(url, files=files, data=dict(data), headers=headers)
            await response.aread()
    except (httpx2.TransportError, TimeoutError) as exc:
        raise _stamp(
            TransientUpstreamError(
                f"{where} unreachable: {type(exc).__name__}: {sanitise(str(exc), secret)}"
            ),
            EXTERNAL,
        ) from None
    if response.status_code >= 300:  # redirects are refused, not followed
        err = _status_error(response, where=where, secret=secret)
        log.warning(
            "in-process ASR call failed endpoint=%s status=%d error=%s",
            redact_url(url),
            response.status_code,
            err,
        )
        raise err
    return response


def _json(response: httpx2.Response, where: str) -> Mapping[str, object]:
    try:
        payload = response.json()
    except ValueError:
        raise _malformed(where, "body is not JSON") from None
    if not isinstance(payload, Mapping):
        raise _malformed(where, "body is not a JSON object")
    return payload


async def call_tenant_asr(
    resolved: ResolvedSlot, request: ASRRequest, *, timeout: float
) -> TranscriptionResult:
    """Transcribe ``request`` on the tenant connection ``resolved``; client-shaped result."""
    if resolved.source != "tenant":
        raise ValueError("call_tenant_asr needs a tenant slot")
    where = _where(resolved.kind)
    if resolved.kind == KIND_NONAGENTIC_ENDPOINT:
        data: dict[str, str | list[str]] = {
            "word_timestamps": "true" if request.word_timestamps else "false",
            "return_srt": "true" if request.return_srt else "false",
        }
        if request.language is not None:
            data["language"] = request.language
        response = await _post_multipart(
            resolved, NONAGENTIC_ASR_PATH, request, data, timeout=timeout, where=where
        )
        try:
            return TranscriptionResult.model_validate(_json(response, where))
        except ValidationError as exc:
            raise _malformed(where, f"{exc.error_count()} schema error(s)") from None
    if resolved.kind == KIND_OPENAI_COMPATIBLE:
        model = resolved.model or ASR_DEFAULT_MODEL
        data = {"model": model, "response_format": "verbose_json"}
        if request.word_timestamps:
            data["timestamp_granularities[]"] = ["word", "segment"]
        if request.language is not None:
            data["language"] = request.language
        response = await _post_multipart(
            resolved, OPENAI_ASR_PATH, request, data, timeout=timeout, where=where
        )
        try:
            return openai_transcript(_json(response, where), model=model, backend=resolved.kind)
        except (ValueError, ValidationError) as exc:
            raise _malformed(where, str(exc)[:200]) from None
    raise SlotUnavailableError(SLOT, f"kind {resolved.kind!r} cannot run in-process")


# ── metering ────────────────────────────────────────────────────────────────


async def record_asr_usage(
    *, tenant_id: uuid.UUID, resolved: ResolvedSlot, result: TranscriptionResult
) -> bool:
    """One ``run_usage`` row per tenant transcription (``tenant_byo``, cost 0, seconds).

    Same best-effort contract as ``tts_inprocess.record_tts_usage``."""
    from src.database.models import RunUsage
    from src.database.repositories.run_usage_repository import (
        IN_PROCESS_USAGE_SOURCE,
        attribution_subject,
    )
    from src.database.session import bind_session_tenant, get_db_session

    run_id = _run_id()
    if run_id is None:
        log.warning("in-process ASR usage not recorded: no run bound (kind=%s)", resolved.kind)
        return False
    seconds = max(0.0, float(result.duration))
    metadata: dict[str, object] = {
        "source": IN_PROCESS_USAGE_SOURCE,
        "funding": FUNDING_TENANT_BYO,
        "slot": SLOT,
        "provider_kind": resolved.kind,
        "connection_id": str(resolved.ref.connection_id),
        "connection_version": resolved.ref.version,
        "unit": "seconds",
        "seconds": seconds,
    }
    project = get_w2v_project_id_from_context()
    if project:
        metadata["project_id"] = project
    actor = get_actor_id_from_context()
    if actor:
        metadata["actor_id"] = actor
    subject = get_subject_id_from_context()
    if subject:
        metadata["subject_id"] = subject
    stage = get_stage_from_context() or "asr"
    row = RunUsage(
        id=uuid.uuid7(),
        run_id=run_id,
        stage_name=stage,
        model=result.model,
        provider=resolved.kind,
        input_tokens=0,
        output_tokens=0,
        cost_usd=Decimal(0),
        metadata_=metadata,
        slot=SLOT,
        subject_id=attribution_subject(subject),
        funding=FUNDING_TENANT_BYO,
        units=Decimal(str(round(seconds, 3))),
        unit_kind="seconds",
        connection_id=resolved.ref.connection_id,
        connection_version=resolved.ref.version,
    )
    try:
        async with get_db_session() as session:
            await bind_session_tenant(session, tenant_id)
            session.add(row)
    except Exception:
        log.exception(
            "in-process ASR usage write FAILED run=%s stage=%s provider=%s seconds=%.3f "
            "— backfill needed",
            run_id,
            stage,
            resolved.kind,
            seconds,
        )
        return False
    return True


# ── entry point ─────────────────────────────────────────────────────────────


async def transcribe_on_tenant_slot(
    ref: SlotRef, request: ASRRequest, *, timeout: float
) -> TranscriptionResult:
    """Resolve the pinned tenant ``asr`` version, transcribe on it, meter it."""
    tenant_id = _bound_tenant()
    resolved = await resolve_pinned_asr(tenant_id, ref)
    result = await call_tenant_asr(resolved, request, timeout=timeout)
    await record_asr_usage(tenant_id=tenant_id, resolved=resolved, result=result)
    return result
