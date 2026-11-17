"""HTTP client for the TTS Service (port 6194 / mini2 :9202).

The tts_backend is a **sync** service: POST ``/api/tts/synthesize`` returns
either ``audio/wav`` bytes or, when ``Accept: application/json`` is sent,
a ``SynthesizeResponse`` JSON document with base64-encoded WAV + duration.

There is no async-job surface (no ``/api/tts/generate``, no ``/api/tts/jobs/{id}``).
The orchestrator stages therefore call ``synthesize`` once per paragraph and
the caller is responsible for persisting the bytes (see
``src.services.tts_storage.store_tts_clip``).

License-authz on tts_backend (mini2 :9202 launchd unit) requires the
service-account licence token in ``X-Tenant-Id: Bearer <token>`` — baked in
by ``BaseServiceClient`` when ``license_token`` is set.

The default base URL is **modality_gateway** (discovery ``modality_gateway``),
never ``tts.test`` (#1106): the gateway applies the tenant's BYO posture
(402 for ``byo_required``) before proxying to the kokoro box, so the platform
path cannot bypass that gate by default. The gateway serves the same
``/api/tts/synthesize`` contract.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import ClassVar

import httpx
from httpx._client import UseClientDefault  # the type of httpx.USE_CLIENT_DEFAULT

from src.auth.run_providers import current_run_slot
from src.clients.base import LicensedServiceClient
from src.clients.gpu_busy import (
    BUSY_STATUSES,
    BusyRetry,
    GpuBusyError,
    busy_retry_after,
    byo_refusal,
)
from src.clients.schemas.tts import SynthesizeWire, TTSSynthesis
from src.models.failure_class import EXTERNAL, FailureClass

logger = logging.getLogger(__name__)

# mini2 runs kokoro on a single fixed-capacity mac mini (16 GiB). Narration is
# multiplicative: course_compile_lessons runs lessons concurrently and each
# lesson_narrate fans out NarrationStageBase concurrency, so without a shared
# ceiling one pod can open (lessons * concurrency) synth sockets at once,
# CPU/RAM-saturate the box, and trip it into dropping connections mid-request
# (an empty-message httpx.TransportError). Cap concurrent synth calls
# process-wide so every stage AND every concurrent lesson share one budget;
# tune via env without a code change.
_MAX_CONCURRENT_SYNTHESIZE = max(1, int(os.getenv("TTS_MAX_CONCURRENT_SYNTHESIZE", "3")))
_TRANSPORT_RETRY_ATTEMPTS = max(1, int(os.getenv("TTS_TRANSPORT_RETRY_ATTEMPTS", "4")))
_TRANSPORT_RETRY_BASE_SECONDS = float(os.getenv("TTS_TRANSPORT_RETRY_BASE_SECONDS", "1.5"))

# 429/503 are the GPU admission contract's "busy" (#1106): honoured via
# ``src.clients.gpu_busy.BusyRetry`` — Retry-After (header, then body), clamped
# 1..120 s, within ``GPU_BUSY_DEADLINE_SECONDS``, then ``GpuBusyError`` (transient).
# 502/504 keep the attempt-bounded budget below.
#
# A 429/502/503/504 from tts_backend is the box (or its license-authz sidecar)
# saying "not right now", not "never". Prod run 01a0a054 died on
# ``503 license_unavailable … retry shortly`` 26s into website_review_record,
# after ~15 minutes of paid upstream work, because this client raised on the
# first 503 and the stage's single ``per_item_retries=1`` retry fired ~1s later
# into the same outage. 502/504 get their OWN, slower budget: 5 attempts
# (4 retries) at 7.5/15/30/60s ≈ 112s, or the server's ``Retry-After`` (clamped
# 1..120 s). 429/503 are bounded by the busy deadline instead. A plain 500
# (e.g. unknown voice) is deliberately NOT here: it is deterministic and must
# surface immediately.
_RETRYABLE_STATUSES = frozenset({502, 504})
_STATUS_RETRY_ATTEMPTS = max(1, int(os.getenv("TTS_STATUS_RETRY_ATTEMPTS", "5")))
_STATUS_RETRY_BASE_SECONDS = float(os.getenv("TTS_STATUS_RETRY_BASE_SECONDS", "7.5"))

# Indirection so tests patch the backoff wait instead of sleeping through it.
_sleep = asyncio.sleep
#: Head-room past a caller's busy budget for the one synth still in flight when the
#: budget ends: budget + this is the hard ceiling of the whole call (preview: 90+5).
_BUSY_CALL_MARGIN_SECONDS = 5.0
#: Of a caller's busy budget, the part kept for the synth itself: busy waits stop
#: this early (preview: busy ≤ 60s, so the admitted synth gets ≥ 35s of 95).
_SYNTH_RESERVE_SECONDS = 30.0
# BaseServiceClient defaults to a 30s timeout — far too short for long-form
# synthesis. A single full ~2000-char audiobook chunk legitimately takes 20-40s
# to render (100s+ of audio) and, on mini2's single-lock kokoro box, queues
# behind any other in-flight synth. At 30s such a chunk ReadTimeouts and gets
# retried, piling more load on the box and failing the whole narrate stage.
# Give TTS a long read timeout so a slow-but-working synth is awaited, not aborted.
_REQUEST_TIMEOUT_SECONDS = float(os.getenv("TTS_REQUEST_TIMEOUT_SECONDS", "300"))

_synth_semaphore: asyncio.Semaphore | None = None

#: Provider slot this client serves (``src/services/providers/registry.py``).
TTS_SLOT = "tts"


def _synth_gate() -> asyncio.Semaphore:
    """Process-wide ceiling on in-flight synthesize calls.

    Lazily bound to the running loop on first use — the orchestrator runs a
    single event loop per pod, so one module-level gate is the right scope.
    """
    global _synth_semaphore
    if _synth_semaphore is None:
        _synth_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SYNTHESIZE)
    return _synth_semaphore


# tts_backend pre-masters every clip to -18 LUFS and reports the measured
# integrated loudness in this response header.
#
# ⚠️ An absent header means the LEVEL IS UNKNOWN — NOT "the premaster failed
# open". This comment used to claim the latter, and it is wrong: tts_backend omits
# the header when its POST-premaster RE-MEASUREMENT fails, which happens while the
# audio itself is normalized perfectly well. It also omits it on a genuine
# premaster failure. The two are indistinguishable from here, so the orchestrator
# treats absence as "unproven" and re-masters the clip itself — see
# ``src/services/stages/primitives/narration.py::ensure_tts_loudness``. Because the
# fallback runs a `linear=true` loudnorm, re-mastering an already-normalized clip
# is a near-no-op gain, so treating the ambiguous case as unproven is safe.
_LUFS_HEADER = "X-TTS-LUFS"


def _integrated_lufs(headers: httpx.Headers, *, voice: str | None) -> float | None:
    """Read the measured post-premaster loudness from ``X-TTS-LUFS``.

    ``None`` when the header is absent or unparseable — i.e. "this clip's level is
    UNPROVEN", which covers both a real tts_backend premaster failure and (the far
    more common case) tts_backend's post-premaster re-measurement failing on audio
    that IS normalized. Both are logged at WARNING and both make the orchestrator's
    ``ensure_tts_loudness`` re-master the clip rather than guess.
    """
    raw = headers.get(_LUFS_HEADER)
    if raw is None:
        logger.warning(
            "tts synthesize response has no %s header — clip loudness is UNPROVEN "
            "(tts_backend premaster failed open, or its re-measurement did); the "
            "orchestrator will re-master it voice=%s",
            _LUFS_HEADER,
            voice,
        )
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "tts synthesize returned unparseable %s=%r voice=%s", _LUFS_HEADER, raw, voice
        )
        return None


#: One attempt's timeout: the call budget's remainder, or the client's own default.
_AttemptTimeout = httpx.Timeout | UseClientDefault


class _NoBudget:
    """No whole-call budget (narration): the plain gate and the client's own timeout."""

    def note_busy(self, response: httpx.Response) -> None:
        """Nothing to track without a budget."""

    def check_transport(self, exc: httpx.TransportError) -> None:
        """A transport error keeps the attempt-bounded retry path."""

    @asynccontextmanager
    async def attempt(self) -> AsyncIterator[_AttemptTimeout]:
        async with _synth_gate():
            yield httpx.USE_CLIENT_DEFAULT


class _CallBudget(_NoBudget):
    """Whole-call ceiling for the voice preview (#1111): budget + margin, ~95s.

    Cloudflare cuts the browser's request at ~100s (524), so EVERYTHING counts
    against one deadline: busy waits, the wait for a ``_synth_gate`` slot, and the
    synth in flight (its timeout is what is left). Busy waits stop
    ``_SYNTH_RESERVE_SECONDS`` early so an admitted synth always gets real time.
    Running out after any busy answer is ``GpuBusyError`` (→ 503 + Retry-After) —
    the lane was busy, not broken — never a ``TTSSynthesisError`` (→ 502).
    """

    def __init__(self, budget: float, busy: BusyRetry) -> None:
        self._limit = budget + _BUSY_CALL_MARGIN_SECONDS
        self._busy = busy
        self._last_busy: httpx.Response | None = None

    def note_busy(self, response: httpx.Response) -> None:
        self._last_busy = response

    def _remaining(self) -> float:
        return self._limit - self._busy.elapsed_seconds

    def _spent(self, why: str) -> Exception:
        if self._last_busy is not None:
            return self._busy.exhausted(self._last_busy, why)
        return TTSSynthesisError(f"tts call budget exhausted ({why})")

    def check_transport(self, exc: httpx.TransportError) -> None:
        # The attempt's timeout WAS the rest of the budget: once it fires after a
        # busy answer the budget is spent — busy, not a transport fault.
        if isinstance(exc, httpx.TimeoutException) and self._last_busy is not None:
            raise self._spent("synth did not finish within the call budget") from exc

    @asynccontextmanager
    async def attempt(self) -> AsyncIterator[_AttemptTimeout]:
        gate = _synth_gate()
        if self._remaining() <= 0:
            raise self._spent("no time left for another attempt")
        try:
            await asyncio.wait_for(gate.acquire(), timeout=self._remaining())
        except TimeoutError as exc:
            if self._last_busy is not None:
                raise self._spent("waiting for a synth slot") from exc
            # Our own synth slots stayed full: local congestion — still "busy".
            raise GpuBusyError(
                "tts synth slots stayed full for the whole call budget",
                service="tts",
                status_code=503,
                waited_seconds=self._busy.elapsed_seconds,
            ) from exc
        try:
            remaining = self._remaining()
            if remaining <= 0:
                raise self._spent("no time left after waiting for a synth slot")
            yield httpx.Timeout(min(remaining, _REQUEST_TIMEOUT_SECONDS))
        finally:
            gate.release()


class TTSSynthesisError(RuntimeError):
    """Raised when the TTS service returns no audio."""


class TTSUnavailableError(TTSSynthesisError):
    """tts_backend kept answering 502/504 through the whole retry budget.

    (429/503 — the GPU admission "busy" — exhaust as ``GpuBusyError`` instead.)

    ``failure_class = external``, stamped on the CLASS (same shape as
    ``VideoCompileJobFailedError``): a busy/rate-limiting/unavailable dependency
    is ``fault_for_status``'s ``external`` verdict for exactly these statuses.
    Without it the bare ``TTSSynthesisError`` (a ``RuntimeError``) attributes to
    ``unknown``. Subclasses ``TTSSynthesisError`` so every existing catcher
    (``api/routes/tts.py``) keeps working — attribution changes, control flow
    does not.
    """

    failure_class: ClassVar[FailureClass] = EXTERNAL

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class TTSClient(LicensedServiceClient):
    """Client for the TTS (Text-to-Speech) microservice."""

    def __init__(
        self,
        base_url: str | None = None,
        license_token: str | None = None,
        timeout: float = _REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(
            service_name="tts",
            discovery_name="modality_gateway",
            base_url=base_url,
            license_token=license_token,
            timeout=timeout,
            licence_hop="modality_gateway",  # MODALITY_GATEWAY_AUTH_MODE
        )

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        provider: str | None = None,
        speed: float = 1.0,
        attempts: int | None = None,
        ssml: bool = False,
        busy_deadline_seconds: float | None = None,
    ) -> TTSSynthesis:
        """Synthesize speech synchronously and return audio bytes + metadata.

        Returns a :class:`TTSSynthesis` with ``audio_bytes`` (raw WAV bytes),
        ``sample_rate`` (int), ``duration_seconds`` (float), ``provider``,
        ``voice``, ``format``, and ``integrated_lufs`` (``float | None``). Audio is
        fetched via ``Accept: application/json`` so the response carries the base64
        payload + duration metadata in a single round-trip; we decode the base64
        once here.

        ``integrated_lufs`` is the loudness tts_backend measured AFTER its -18 LUFS
        premaster, read from the ``X-TTS-LUFS`` response header. It is **not** used
        to mix — it is a PROOF-OF-LEVEL: a value means this clip's level is known,
        and ``None`` means it is UNPROVEN (tts_backend omits the header both when
        its premaster failed open and — more commonly — when its post-premaster
        re-measurement failed on audio that is in fact normalized). Absence is
        logged at WARNING and drives the orchestrator's own re-master in
        ``services/stages/primitives/narration.py::ensure_tts_loudness``, so an
        unproven clip no longer reaches the timeline at an arbitrary level.

        ``attempts`` overrides the transport-retry budget for this one call
        (default: the module-level narration budget). Interactive callers like
        the voice-preview endpoint pass a small value so a saturated backend
        surfaces a fast error instead of blocking a waiting human behind the
        full exponential-backoff chain.

        ``busy_deadline_seconds`` (with ``attempts``) gives a BUSY lane (429/503)
        its own total wait budget instead of the ``attempts`` cap: the voice
        preview fails fast on a broken box but sits out a full one, honouring
        ``Retry-After``. Exhausted → :class:`GpuBusyError` (``.retry_after`` set).

        ``ssml`` requests SSML-aware synthesis: when ``True`` the payload carries
        ``ssml: true`` so tts_backend parses ``text`` as SSML. When ``False``
        (the default) the ``ssml`` key is omitted entirely, keeping the plain
        request body byte-identical to callers that never pass it.
        """
        # P5-1 slot dispatch: a run that pinned a TENANT ``tts`` connection synthesizes
        # in-process on it (never falling back to ours); no pin / the deployment pin
        # takes the modality-gateway path below, byte-for-byte unchanged. ``provider``
        # and ``attempts`` are modality-gateway knobs: the tenant's connection decides
        # the provider and the in-process client carries its own retry policy.
        ref = current_run_slot(TTS_SLOT)
        if ref is not None and ref.source == "tenant":
            from src.services.providers.tts_inprocess import TTSRequest, synthesize_on_tenant_slot

            return await synthesize_on_tenant_slot(
                ref,
                TTSRequest(text=text, voice=voice, speed=speed, ssml=ssml),
                timeout=self.timeout,
            )

        from src.services.providers.platform_guard import refuse_platform_for_bound_tenant

        await refuse_platform_for_bound_tenant(TTS_SLOT)
        payload: dict[str, object] = {"text": text, "speed": speed}
        if voice is not None:
            payload["voice"] = voice
        if provider is not None and provider != "auto":
            payload["provider"] = provider
        if ssml:
            payload["ssml"] = True

        body, headers = await self._post_synthesize_json(
            payload, attempts=attempts, busy_deadline_seconds=busy_deadline_seconds
        )
        if not body.audio_base64:
            raise TTSSynthesisError("tts_backend returned no audio_base64")
        return TTSSynthesis(
            audio_bytes=base64.b64decode(body.audio_base64),
            sample_rate=body.sample_rate,
            duration_seconds=body.duration_sec,
            provider=body.provider,
            voice=body.voice,
            format=body.format,
            integrated_lufs=_integrated_lufs(headers, voice=voice),
            word_timings=body.word_timings,
            # None on any tts_backend deployed before the companion A/V quality
            # wave 8 change — see schemas/tts.py. Passed through verbatim, never
            # defaulted to 0.0, so "unknown" stays distinguishable from "none".
            speech_seconds=body.speech_seconds,
            pause_seconds=body.pause_seconds,
        )

    async def _post_synthesize_json(
        self,
        payload: dict[str, object],
        attempts: int | None = None,
        busy_deadline_seconds: float | None = None,
    ) -> tuple[SynthesizeWire, httpx.Headers]:
        # tts_backend's /synthesize returns audio/wav by default; only
        # ``Accept: application/json`` switches it to the JSON-with-base64
        # variant. BaseServiceClient._request() can't inject Accept, so we
        # drop a level and call the underlying httpx client directly while
        # keeping the auth + identity hooks the base client installed.
        #
        # Dropping below _request() also bypasses its transport retry, so we
        # retry here: a saturated mini2 drops connections mid-request, and an
        # exponential-backoff loop turns those transient drops into a success
        # once the box drains. The shared _synth_gate keeps us from being the
        # cause of that saturation in the first place. The gate is released
        # during backoff (it wraps only the request) so retries don't hog a
        # slot while sleeping. 429/502/503/504 are retried on a separate, slower
        # budget (see ``_RETRYABLE_STATUSES``); every other 4xx/5xx is NOT
        # retried — an unknown-voice 500 must surface, not loop.
        max_attempts = _TRANSPORT_RETRY_ATTEMPTS if attempts is None else max(1, attempts)
        max_status_attempts = _STATUS_RETRY_ATTEMPTS if attempts is None else max(1, attempts)
        chars = len(str(payload.get("text") or ""))
        voice = payload.get("voice")
        client = await self._get_client()
        attempt = 0
        status_attempt = 0
        # An explicit ``attempts`` also caps how many busy answers we sit through —
        # unless the caller gave busy its OWN deadline (the voice preview, #1111),
        # which then bounds busy by time, honouring each Retry-After. The total
        # deadline always applies.
        budget: _NoBudget
        if busy_deadline_seconds is not None:
            busy = BusyRetry(
                "tts",
                deadline_seconds=max(busy_deadline_seconds - _SYNTH_RESERVE_SECONDS, 0.0),
                sleep=_sleep,
            )
            budget = _CallBudget(busy_deadline_seconds, busy)
        else:
            busy = BusyRetry(
                "tts", max_attempts=None if attempts is None else max(1, attempts), sleep=_sleep
            )
            budget = _NoBudget()
        while attempt < max_attempts:
            # With a busy budget (the voice preview) the WHOLE call — busy waits, the
            # gate wait and the in-flight synth — ends by budget + margin (#1111).
            try:
                async with budget.attempt() as request_timeout:
                    response = await client.request(
                        "POST",
                        "/api/tts/synthesize",
                        json=payload,
                        headers={"Accept": "application/json"},
                        timeout=request_timeout,
                    )
            except httpx.TransportError as exc:
                budget.check_transport(exc)
                if attempt + 1 >= max_attempts:
                    logger.error(
                        "tts synthesize transport error voice=%s chars=%d after %d attempts: %r",
                        voice,
                        chars,
                        max_attempts,
                        exc,
                    )
                    raise TTSSynthesisError(
                        f"tts transport error after {max_attempts} attempts: {exc!r}"
                    ) from exc
                logger.warning(
                    "tts synthesize transport error voice=%s chars=%d attempt %d/%d, retrying: %r",
                    voice,
                    chars,
                    attempt + 1,
                    max_attempts,
                    exc,
                )
                await _sleep(_TRANSPORT_RETRY_BASE_SECONDS * (2**attempt))
                attempt += 1
                continue
            self._raise_on_redirect(response, "POST", "/api/tts/synthesize")
            refusal = byo_refusal(response, "tts")
            if refusal is not None:
                raise refusal
            if response.status_code in BUSY_STATUSES:
                logger.warning(
                    "tts /synthesize %d (busy) voice=%s chars=%d busy #%d, honouring Retry-After",
                    response.status_code,
                    voice,
                    chars,
                    busy.busy_count + 1,
                )
                budget.note_busy(response)
                await busy.wait(response)
                continue
            if response.status_code in _RETRYABLE_STATUSES:
                status_attempt += 1
                detail = f"tts /synthesize returned {response.status_code}: {response.text[:500]}"
                if status_attempt >= max_status_attempts:
                    logger.error(
                        "tts /synthesize %d voice=%s chars=%d after %d attempts body=%s",
                        response.status_code,
                        voice,
                        chars,
                        status_attempt,
                        response.text[:500],
                    )
                    raise TTSUnavailableError(
                        f"{detail} (after {status_attempt} attempts)",
                        status_code=response.status_code,
                    )
                retry_after = busy_retry_after(response)
                delay = (
                    retry_after
                    if retry_after is not None
                    else _STATUS_RETRY_BASE_SECONDS * (2 ** (status_attempt - 1))
                )
                logger.warning(
                    "tts /synthesize %d voice=%s chars=%d attempt %d/%d, retrying in %.1fs",
                    response.status_code,
                    voice,
                    chars,
                    status_attempt,
                    max_status_attempts,
                    delay,
                )
                await _sleep(delay)
                continue
            if response.status_code >= 400:
                logger.error(
                    "tts /synthesize %d voice=%s chars=%d body=%s",
                    response.status_code,
                    voice,
                    chars,
                    response.text[:500],
                )
                raise TTSSynthesisError(
                    f"tts /synthesize returned {response.status_code}: {response.text[:500]}"
                )
            return SynthesizeWire.model_validate(response.json()), response.headers
        # Unreachable: the loop returns on success or raises on the final
        # attempt; this satisfies the type checker's all-paths-return rule.
        raise TTSSynthesisError("tts synthesize: retry loop exited without a response")
