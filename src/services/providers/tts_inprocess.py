"""Run one ``TTSClient.synthesize()`` IN-PROCESS on a tenant ``tts`` slot (P5-1).

Used only when the run pinned a TENANT connection for ``tts``. The caller contract is
``TTSClient.synthesize``'s — text, voice, speed, ssml — and the result is the same
:class:`TTSSynthesis` the modality-gateway path returns, so the narrate call sites
cannot tell the difference. Fields the tenant provider cannot report stay ``None`` so
the existing fallbacks run: ``integrated_lufs=None`` → ``ensure_tts_loudness``
re-masters; ``word_timings=None`` → ASR captions; ``speech/pause_seconds=None`` →
authored-break-only pacing.

Provider kinds:

* ``openai`` — ``POST {base_url or https://api.openai.com/v1}/audio/speech``, WAV.
* ``openai-compatible`` — same request against the tenant's own ``base_url``.
* ``elevenlabs`` — ``POST {base_url or https://api.elevenlabs.io}/v1/text-to-speech/
  {voice_id}/with-timestamps?output_format=pcm_24000``; the PCM is wrapped into WAV
  and ``word_timings`` are derived from the character alignment.
* ``nonagentic_endpoint`` — the PLATFORM protocol at the tenant's ``base_url``:
  ``POST {base_url}/api/tts/synthesize`` with the body ``TTSClient`` sends the
  modality gateway (``text``/``speed``/``voice``/``ssml``, ``Accept: application/json``)
  plus ``Authorization: Bearer <secret>`` when one is stored. SSML is passed through
  (the platform parses it); ``X-TTS-LUFS``, ``word_timings`` and the pause split are
  read exactly as the gateway path reads them. A raw ``audio/wav`` answer is accepted.

Egress, retries and key redaction are the P3 in-process LLM's (``llm/in_process``):
the same pinned-IP, origin-locked, redirect-refusing ``httpx2`` client and the same
retry policy. A tenant slot failure raises the orchestrator's own
:class:`TransientUpstreamError` / :class:`PermanentUpstreamError` and NEVER falls back
to the modality gateway.
"""

from __future__ import annotations

import base64
import binascii
import html
import io
import logging
import re
import struct
import uuid
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Final
from urllib.parse import quote

import httpx2
from w2v_common.auth.identity_context import (
    get_actor_id_from_context,
    get_stage_from_context,
    get_subject_id_from_context,
    get_tenant_id_from_context,
    get_w2v_project_id_from_context,
)

from src.clients.schemas.tts import SynthesizeWire, TTSSynthesis, WordTiming
from src.models.failure_class import CUSTOMER_INPUT, EXTERNAL, FailureClass
from src.services.llm.in_process import _AUTH_HTTP, _TRANSIENT_HTTP, _http_client, sanitise
from src.services.llm.metering import FUNDING_TENANT_BYO, _run_id
from src.services.providers.errors import SlotUnavailableError
from src.services.providers.policy import redact_url
from src.services.providers.resolver import PROVIDER_SLOTS_KEY, ResolvedSlot, SlotRef, resolve_slot
from src.services.stages.upstream import PermanentUpstreamError, TransientUpstreamError

log = logging.getLogger(__name__)

SLOT: Final = "tts"

#: Tenant kinds of the ``tts`` slot (the registry lists the same strings).
KIND_OPENAI: Final = "openai"
KIND_OPENAI_COMPATIBLE: Final = "openai-compatible"
KIND_ELEVENLABS: Final = "elevenlabs"
KIND_NONAGENTIC_ENDPOINT: Final = "nonagentic_endpoint"
#: The platform synth route (modality-gateway / tts_backend), under the tenant base URL.
NONAGENTIC_TTS_PATH: Final = "/api/tts/synthesize"
_LUFS_HEADER: Final = "X-TTS-LUFS"

OPENAI_DEFAULT_BASE_URL: Final = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL: Final = "gpt-4o-mini-tts"
OPENAI_DEFAULT_VOICE: Final = "alloy"
ELEVENLABS_DEFAULT_BASE_URL: Final = "https://api.elevenlabs.io"
ELEVENLABS_DEFAULT_MODEL: Final = "eleven_multilingual_v2"
#: ``output_format=pcm_24000`` → raw signed 16-bit little-endian mono at 24 kHz.
ELEVENLABS_PCM_RATE: Final = 24_000
#: ElevenLabs ``voice_settings.speed`` accepted range.
_ELEVENLABS_SPEED_RANGE: Final = (0.7, 1.2)
#: OpenAI ``/audio/speech`` ``speed`` accepted range.
_OPENAI_SPEED_RANGE: Final = (0.25, 4.0)

_SSML_TAG: Final = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class TTSRequest:
    """The ``TTSClient.synthesize`` contract, as the in-process path consumes it."""

    text: str
    voice: str | None = None
    speed: float = 1.0
    ssml: bool = False


@dataclass(frozen=True, slots=True)
class TTSResult:
    synthesis: TTSSynthesis
    characters: int
    model: str
    provider_kind: str


# ── resolution ──────────────────────────────────────────────────────────────


async def resolve_pinned_tts(tenant_id: uuid.UUID, ref: SlotRef) -> ResolvedSlot:
    """Load + validate exactly the pinned ``tts`` version for ``tenant_id``."""
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


# ── audio helpers ───────────────────────────────────────────────────────────


def plain_text(text: str, *, ssml: bool) -> str:
    """The text a tenant provider should speak: SSML markup stripped, entities decoded.

    None of the tenant kinds parses ``tts_backend``'s ``<speak>`` dialect — sent raw,
    the tags would be read aloud. The authored pause/emphasis is lost (the clip's
    ``pause_seconds`` is ``None``, so pacing falls back to authored-break-only)."""
    if not ssml:
        return text
    return " ".join(html.unescape(_SSML_TAG.sub(" ", text)).split())


def pcm16_to_wav(pcm: bytes, sample_rate: int, *, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm)
    return buf.getvalue()


@dataclass(frozen=True, slots=True)
class _Wav:
    data: bytes
    sample_rate: int
    duration_seconds: float


def normalise_wav(raw: bytes) -> _Wav:
    """Parse a provider WAV and return a WAV with CORRECT chunk sizes + its duration.

    OpenAI(-compatible) servers stream WAV with placeholder RIFF/data sizes
    (``0xFFFFFFFF``); the real data length is what follows the ``data`` header. PCM is
    re-wrapped with true sizes; any other encoding is passed through untouched with
    the duration computed from the bytes actually present. Raises ``ValueError``."""
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("response is not a RIFF/WAVE file")
    pos = 12
    fmt: tuple[int, int, int, int] | None = None  # (tag, channels, rate, bits)
    while pos + 8 <= len(raw):
        chunk_id = raw[pos : pos + 4]
        (size,) = struct.unpack("<I", raw[pos + 4 : pos + 8])
        body = pos + 8
        if chunk_id == b"fmt ":
            if size < 16 or body + 16 > len(raw):
                raise ValueError("WAV fmt chunk is truncated")
            tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", raw[body : body + 16])
            fmt = (tag, channels, rate, bits)
        elif chunk_id == b"data":
            if fmt is None:
                raise ValueError("WAV data chunk precedes its fmt chunk")
            tag, channels, rate, bits = fmt
            data = raw[body : body + min(size, len(raw) - body)]
            frame_bytes = channels * bits // 8
            if rate <= 0 or frame_bytes <= 0:
                raise ValueError("WAV fmt chunk declares no audio")
            data = data[: len(data) - len(data) % frame_bytes]
            duration = len(data) / (rate * frame_bytes)
            if tag == 1 and bits == 16:
                return _Wav(pcm16_to_wav(data, rate, channels=channels), rate, duration)
            return _Wav(raw, rate, duration)
        pos = body + size + (size & 1)
    raise ValueError("WAV has no data chunk")


def word_timings_from_alignment(alignment: object) -> list[WordTiming] | None:
    """ElevenLabs character alignment → ``[{word, start, end}]``, grouped on whitespace.

    ``None`` when the alignment is absent or inconsistent (→ ASR caption fallback);
    never a partial or guessed list."""
    if not isinstance(alignment, Mapping):
        return None
    chars = alignment.get("characters")
    starts = alignment.get("character_start_times_seconds")
    ends = alignment.get("character_end_times_seconds")
    if not (
        isinstance(chars, Sequence)
        and isinstance(starts, Sequence)
        and isinstance(ends, Sequence)
        and not isinstance(chars, str)
        and len(chars) == len(starts) == len(ends)
        and chars
    ):
        return None
    words: list[WordTiming] = []
    current: list[str] = []
    first = last = 0.0
    try:
        for ch, start, end in zip(chars, starts, ends, strict=True):
            if not isinstance(ch, str):
                return None
            if ch.isspace() or not ch:
                if current:
                    words.append(WordTiming(word="".join(current), start=first, end=last))
                    current = []
                continue
            if not current:
                first = float(start)
            current.append(ch)
            last = float(end)
        if current:
            words.append(WordTiming(word="".join(current), start=first, end=last))
    except TypeError, ValueError:
        return None
    return words or None


# ── error mapping ───────────────────────────────────────────────────────────


def _stamp[E: TransientUpstreamError | PermanentUpstreamError](err: E, fault: FailureClass) -> E:
    err.failure_class = fault
    return err


def _where(kind: str, model: str) -> str:
    return f"Tenant TTS provider ({kind}, model {model})"


def _status_error(response: httpx2.Response, *, where: str, secret: str | None) -> Exception:
    status = response.status_code
    if status in _AUTH_HTTP:
        return _stamp(
            PermanentUpstreamError(
                f"{where} rejected the credentials (HTTP {status}). "
                "Update the API key on the Providers page."
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
    # A provider that answers 200 with something unusable does so deterministically.
    return _stamp(
        PermanentUpstreamError(f"{where} returned an unusable response: {what}"), EXTERNAL
    )


# ── the call ────────────────────────────────────────────────────────────────


def _secret(resolved: ResolvedSlot, *, required: bool) -> str | None:
    value = resolved.secret.get_secret_value() if resolved.secret else None
    if required and not value:
        raise SlotUnavailableError(SLOT, "the connection has no API key")
    return value or None


def _option(resolved: ResolvedSlot, *names: str) -> str | None:
    for name in names:
        value = resolved.options.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(max(value, bounds[0]), bounds[1])


async def _openai_speech(
    resolved: ResolvedSlot, request: TTSRequest, *, timeout: float
) -> TTSResult:
    if resolved.kind == KIND_OPENAI_COMPATIBLE and not resolved.base_url:
        raise SlotUnavailableError(SLOT, "kind 'openai-compatible' requires a base URL")
    secret = _secret(resolved, required=resolved.kind == KIND_OPENAI)
    base_url = (resolved.base_url or OPENAI_DEFAULT_BASE_URL).rstrip("/")
    pinned = resolved.pinned_ip if resolved.base_url else None
    model = resolved.model or OPENAI_DEFAULT_MODEL
    voice = _option(resolved, "voice") or request.voice or OPENAI_DEFAULT_VOICE
    text = plain_text(request.text, ssml=request.ssml)
    where = _where(resolved.kind, model)
    headers = {"Accept": "audio/wav"}
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    body = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "wav",
        "speed": _clamp(request.speed, _OPENAI_SPEED_RANGE),
    }
    response = await _post(
        base_url, pinned, f"{base_url}/audio/speech", body, headers, timeout, where, secret
    )
    try:
        wav = normalise_wav(response.content)
    except ValueError as exc:
        raise _malformed(where, str(exc)) from None
    synthesis = TTSSynthesis(
        audio_bytes=wav.data,
        sample_rate=wav.sample_rate,
        duration_seconds=wav.duration_seconds,
        provider=resolved.kind,
        voice=voice,
        format="wav",
    )
    return TTSResult(
        synthesis=synthesis, characters=len(text), model=model, provider_kind=resolved.kind
    )


async def _elevenlabs_speech(
    resolved: ResolvedSlot, request: TTSRequest, *, timeout: float
) -> TTSResult:
    secret = _secret(resolved, required=True)
    voice_id = _option(resolved, "voice_id", "voice") or request.voice
    if not voice_id:
        raise SlotUnavailableError(SLOT, "the connection has no ElevenLabs voice id configured")
    base_url = (resolved.base_url or ELEVENLABS_DEFAULT_BASE_URL).rstrip("/")
    pinned = resolved.pinned_ip if resolved.base_url else None
    model = resolved.model or ELEVENLABS_DEFAULT_MODEL
    text = plain_text(request.text, ssml=request.ssml)
    where = _where(resolved.kind, model)
    body: dict[str, object] = {"text": text, "model_id": model}
    if request.speed != 1.0:
        body["voice_settings"] = {"speed": _clamp(request.speed, _ELEVENLABS_SPEED_RANGE)}
    url = (
        f"{base_url}/v1/text-to-speech/{quote(voice_id, safe='')}/with-timestamps"
        f"?output_format=pcm_{ELEVENLABS_PCM_RATE}"
    )
    headers = {"xi-api-key": secret or "", "Accept": "application/json"}
    response = await _post(base_url, pinned, url, body, headers, timeout, where, secret)
    try:
        payload = response.json()
    except ValueError:
        raise _malformed(where, "body is not JSON") from None
    if not isinstance(payload, Mapping) or not isinstance(payload.get("audio_base64"), str):
        raise _malformed(where, "no audio_base64")
    try:
        pcm = base64.b64decode(payload["audio_base64"], validate=True)
    except binascii.Error, ValueError:
        raise _malformed(where, "audio_base64 is not valid base64") from None
    if not pcm:
        raise _malformed(where, "empty audio")
    pcm = pcm[: len(pcm) - len(pcm) % 2]
    synthesis = TTSSynthesis(
        audio_bytes=pcm16_to_wav(pcm, ELEVENLABS_PCM_RATE),
        sample_rate=ELEVENLABS_PCM_RATE,
        duration_seconds=len(pcm) / (2 * ELEVENLABS_PCM_RATE),
        provider=resolved.kind,
        voice=voice_id,
        format="wav",
        word_timings=word_timings_from_alignment(payload.get("alignment")),
    )
    return TTSResult(
        synthesis=synthesis, characters=len(text), model=model, provider_kind=resolved.kind
    )


def _lufs(headers: Mapping[str, str]) -> float | None:
    raw = headers.get(_LUFS_HEADER)
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


async def _nonagentic_speech(
    resolved: ResolvedSlot, request: TTSRequest, *, timeout: float
) -> TTSResult:
    if not resolved.base_url:
        raise SlotUnavailableError(SLOT, "kind 'nonagentic_endpoint' requires a base URL")
    secret = _secret(resolved, required=False)
    base_url = resolved.base_url.rstrip("/")
    voice = _option(resolved, "voice") or request.voice
    model = resolved.model or "platform"
    where = _where(resolved.kind, model)
    body: dict[str, object] = {"text": request.text, "speed": request.speed}
    if voice is not None:
        body["voice"] = voice
    if request.ssml:
        body["ssml"] = True
    headers = {"Accept": "application/json"}
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    response = await _post(
        base_url,
        resolved.pinned_ip,
        f"{base_url}{NONAGENTIC_TTS_PATH}",
        body,
        headers,
        timeout,
        where,
        secret,
    )
    lufs = _lufs(response.headers)
    characters = len(plain_text(request.text, ssml=request.ssml))
    if response.headers.get("content-type", "").startswith("audio/"):
        try:
            wav = normalise_wav(response.content)
        except ValueError as exc:
            raise _malformed(where, str(exc)) from None
        synthesis = TTSSynthesis(
            audio_bytes=wav.data,
            sample_rate=wav.sample_rate,
            duration_seconds=wav.duration_seconds,
            provider=resolved.kind,
            voice=voice or "",
            format="wav",
            integrated_lufs=lufs,
        )
        return TTSResult(
            synthesis=synthesis, characters=characters, model=model, provider_kind=resolved.kind
        )
    try:
        wire = SynthesizeWire.model_validate(response.json())
        audio = base64.b64decode(wire.audio_base64, validate=True)
    except (ValueError, binascii.Error) as exc:  # ValidationError ⊂ ValueError
        raise _malformed(where, f"not a synthesize response ({type(exc).__name__})") from None
    if not audio:
        raise _malformed(where, "empty audio")
    synthesis = TTSSynthesis(
        audio_bytes=audio,
        sample_rate=wire.sample_rate,
        duration_seconds=wire.duration_sec,
        provider=wire.provider,
        voice=wire.voice,
        format=wire.format,
        integrated_lufs=lufs,
        word_timings=wire.word_timings,
        speech_seconds=wire.speech_seconds,
        pause_seconds=wire.pause_seconds,
    )
    return TTSResult(
        synthesis=synthesis, characters=characters, model=model, provider_kind=resolved.kind
    )


async def _post(
    base_url: str,
    pinned_ip: str | None,
    url: str,
    body: Mapping[str, object],
    headers: Mapping[str, str],
    timeout: float,
    where: str,
    secret: str | None,
) -> httpx2.Response:
    try:
        async with _http_client(base_url, pinned_ip, timeout, retry_read_timeout=False) as http:
            response = await http.post(url, json=dict(body), headers=dict(headers))
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
            "in-process TTS call failed endpoint=%s status=%d error=%s",
            redact_url(url),
            response.status_code,
            err,
        )
        raise err
    return response


async def call_tenant_tts(
    resolved: ResolvedSlot, request: TTSRequest, *, timeout: float
) -> TTSResult:
    """Synthesize ``request`` on the tenant connection ``resolved``; client-shaped result."""
    if resolved.source != "tenant":
        raise ValueError("call_tenant_tts needs a tenant slot")
    if resolved.kind in {KIND_OPENAI, KIND_OPENAI_COMPATIBLE}:
        return await _openai_speech(resolved, request, timeout=timeout)
    if resolved.kind == KIND_ELEVENLABS:
        return await _elevenlabs_speech(resolved, request, timeout=timeout)
    if resolved.kind == KIND_NONAGENTIC_ENDPOINT:
        return await _nonagentic_speech(resolved, request, timeout=timeout)
    raise SlotUnavailableError(SLOT, f"kind {resolved.kind!r} cannot run in-process")


# ── metering ────────────────────────────────────────────────────────────────


async def record_tts_usage(
    *, tenant_id: uuid.UUID, resolved: ResolvedSlot, result: TTSResult
) -> bool:
    """One ``run_usage`` row per tenant synth (``tenant_byo``, cost 0, characters).

    Same contract as ``llm/metering.record_in_process_usage``: best-effort, the
    ``in_process`` source keeps the billing mirror off the row, a failed write is
    logged with what a backfill needs. Characters live in ``metadata.characters``;
    the token columns stay 0 so token aggregates are not polluted with characters."""
    from src.database.models import RunUsage
    from src.database.repositories.run_usage_repository import (
        IN_PROCESS_USAGE_SOURCE,
        attribution_subject,
    )
    from src.database.session import bind_session_tenant, get_db_session

    run_id = _run_id()
    if run_id is None:
        log.warning(
            "in-process TTS usage not recorded: no run bound (kind=%s chars=%d)",
            result.provider_kind,
            result.characters,
        )
        return False
    metadata: dict[str, object] = {
        "source": IN_PROCESS_USAGE_SOURCE,
        "funding": FUNDING_TENANT_BYO,
        "slot": SLOT,
        "provider_kind": result.provider_kind,
        "connection_id": str(resolved.ref.connection_id),
        "connection_version": resolved.ref.version,
        "unit": "characters",
        "characters": result.characters,
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
    stage = get_stage_from_context() or "tts"
    row = RunUsage(
        id=uuid.uuid7(),
        run_id=run_id,
        stage_name=stage,
        model=result.model,
        provider=result.provider_kind,
        input_tokens=0,
        output_tokens=0,
        cost_usd=Decimal(0),
        metadata_=metadata,
        slot=SLOT,
        subject_id=attribution_subject(subject),
        funding=FUNDING_TENANT_BYO,
        units=Decimal(result.characters),
        unit_kind="chars",
        connection_id=resolved.ref.connection_id,
        connection_version=resolved.ref.version,
    )
    try:
        async with get_db_session() as session:
            await bind_session_tenant(session, tenant_id)
            session.add(row)
    except Exception:
        log.exception(
            "in-process TTS usage write FAILED run=%s stage=%s model=%s provider=%s "
            "chars=%d — backfill needed",
            run_id,
            stage,
            result.model,
            result.provider_kind,
            result.characters,
        )
        return False
    return True


# ── entry points ────────────────────────────────────────────────────────────


def tenant_tts_bound() -> bool:
    """True when the bound run pinned a TENANT ``tts`` connection."""
    from src.auth.run_providers import current_run_slot

    ref = current_run_slot(SLOT)
    return ref is not None and ref.source == "tenant"


async def synthesize_on_tenant_slot(
    ref: SlotRef, request: TTSRequest, *, timeout: float
) -> TTSSynthesis:
    """Resolve the pinned tenant ``tts`` version, synthesize on it, meter it."""
    tenant_id = _bound_tenant()
    resolved = await resolve_pinned_tts(tenant_id, ref)
    result = await call_tenant_tts(resolved, request, timeout=timeout)
    await record_tts_usage(tenant_id=tenant_id, resolved=resolved, result=result)
    return result.synthesis
