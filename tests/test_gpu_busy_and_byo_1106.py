"""#1106 — GPU admission back-pressure + the tenant BYO policy, end to end per client.

* busy (503/429) is waited out per ``Retry-After`` (header, then body), clamped 1..120,
  within a deadline; then ``GpuBusyError`` — TRANSIENT, ``external``, never permanent;
* ASR goes through modality_gateway ``/api/asr/transcribe`` — never ``asr.test`` —
  under a per-process concurrency cap;
* a ``byo_required`` tenant: modality_gateway's 402 becomes
  ``SlotNotConfiguredError`` (``customer_input``, code ``byo_provider_required``) on
  every client, and a tenant slot that points at a platform-internal engine is
  refused before ANY socket opens;
* the licences-backed posture reader fails CLOSED.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar
from uuid import UUID

import httpx
import httpx2
import pytest
from w2v_common.auth.identity_context import set_correlation_context, set_identity_context
from w2v_common.stages import StageRetriesExhaustedError

import src.clients.transcription_worker_client as asr_mod
import src.clients.tts_client as tts_mod
import src.clients.tts_gateway_client as gw_mod
from src.auth.run_providers import bind_run_provider_slots, clear_run_provider_slots
from src.clients.gpu_busy import GpuBusyError, busy_retry_after, get_gpu_admission_settings
from src.clients.licenses_client import LicensesClient
from src.clients.transcription_worker_client import TranscriptionWorkerClient
from src.clients.tts_client import TTSClient
from src.clients.tts_gateway_client import TTSGatewayClient
from src.models.failure_class import (
    CUSTOMER_INPUT,
    EXTERNAL,
    attribute_exception,
    failure_detail_for,
)
from src.services.llm import in_process
from src.services.providers import tts_inprocess
from src.services.providers.byo_posture import (
    LicencesByoPosture,
    StaticByoPosture,
    clear_byo_posture_cache,
)
from src.services.providers.errors import ByoPostureUnavailableError, SlotNotConfiguredError
from src.services.providers.resolver import (
    PROVIDER_SLOTS_KEY,
    ResolvedSlot,
    SlotRef,
    resolve_slot,
)
from src.services.providers.settings import ProviderPolicySettings
from src.services.stages.upstream import PermanentUpstreamError, TransientUpstreamError

TENANT = UUID("01990000-0000-7000-8000-0000000011aa")
CONN = UUID("01990000-0000-7000-8000-0000000011bb")
GATEWAY = "http://modality-gateway.test"
BYO_DETAIL = {
    "code": "byo_provider_required",
    "modality": "tts",
    "slots": "tts",
    "message": "Configure your own TTS provider to continue",
}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    """Instant sleeps (recorded), a known deadline, fresh caches/gates per test."""
    slept: list[float] = []

    async def _sleep(seconds: float) -> None:
        slept.append(seconds)
        # Yield to the loop: an instant fake sleep that never yields would let a
        # deadline-never-fires mutant spin forever where ``wait_for`` cannot cancel it.
        await asyncio.sleep(0)

    for mod in (tts_mod, gw_mod, asr_mod):
        monkeypatch.setattr(mod, "_sleep", _sleep)
    monkeypatch.setenv("GPU_BUSY_DEADLINE_SECONDS", "60")
    monkeypatch.setenv("ASR_MAX_CONCURRENCY", "2")
    get_gpu_admission_settings.cache_clear()
    monkeypatch.setattr(asr_mod, "_asr_semaphore", None)
    clear_byo_posture_cache()
    clear_run_provider_slots()
    yield slept
    get_gpu_admission_settings.cache_clear()
    clear_byo_posture_cache()
    clear_run_provider_slots()


def _busy(retry_after: str | None = None, body_retry: object = 3, status: int = 503):
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return httpx.Response(
        status,
        json={"error": "gpu_busy", "lane": "kokoro", "retry_after": body_retry},
        headers=headers,
    )


def _refused_402() -> httpx.Response:
    return httpx.Response(
        402,
        json={"detail": "tenant must use its own configured tts provider"},
    )


class _Seq:
    """MockTransport handler: answers ``responses`` in order (last repeats), records requests."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        idx = min(len(self.requests) - 1, len(self.responses) - 1)
        return self.responses[idx]


# ── Retry-After parsing ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("headers", "body", "want"),
    [({"Retry-After": "7"}, {"retry_after": 50}, 7.0)],
)
def test_retry_after_header_first_then_body_clamped(
    headers: dict[str, str], body: dict[str, Any], want: float | None
) -> None:
    assert busy_retry_after(httpx.Response(503, json=body, headers=headers)) == want


# ── TTSClient (modality_gateway synthesize) ─────────────────────────────────


def _tts(handler: _Seq) -> TTSClient:
    client = TTSClient(base_url=GATEWAY, license_token="svc")
    client.set_transport(httpx.MockTransport(handler))
    return client


def _tts_ok() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "audio_base64": "UklGRg==",
            "duration_sec": 1.0,
            "sample_rate": 24000,
            "provider": "kokoro",
            "voice": "af_heart",
            "format": "wav",
        },
        headers={"X-TTS-LUFS": "-18.0"},
    )


async def test_tts_busy_is_retried_per_retry_after_then_succeeds(_fresh: list[float]) -> None:
    seq = _Seq(_busy("7"), _busy(None, body_retry=5), _tts_ok())
    client = _tts(seq)

    result = await client.synthesize(text="hello")

    assert result.audio_bytes == b"RIFF"
    assert len(seq.requests) == 3
    assert _fresh == [7.0, 5.0]  # header, then body retry_after


async def test_tts_busy_past_deadline_is_a_transient_gpu_busy_error(_fresh: list[float]) -> None:
    seq = _Seq(_busy("25"))
    client = _tts(seq)

    # wait_for: a mutant whose deadline never fires must FAIL here, not hang the suite.
    with pytest.raises(GpuBusyError) as info:
        await asyncio.wait_for(client.synthesize(text="hello"), timeout=10)

    err = info.value
    # 25 + 25 fit the 60s deadline; a third 25 would not.
    assert _fresh == [25.0, 25.0] and len(seq.requests) == 3
    assert isinstance(err, TransientUpstreamError)
    assert not isinstance(err, PermanentUpstreamError)
    assert err.lane == "kokoro" and err.status_code == 503
    assert attribute_exception(err) == EXTERNAL
    assert failure_detail_for(err) is None


async def test_tts_402_is_byo_provider_required_and_not_retried(_fresh: list[float]) -> None:
    seq = _Seq(_refused_402(), _tts_ok())
    client = _tts(seq)

    with pytest.raises(SlotNotConfiguredError) as info:
        await client.synthesize(text="hello")

    assert len(seq.requests) == 1 and _fresh == []
    assert attribute_exception(info.value) == CUSTOMER_INPUT
    assert failure_detail_for(info.value) == BYO_DETAIL


# ── TTSGatewayClient (audiobook claim-queue) ────────────────────────────────


def _gateway(handler: _Seq) -> TTSGatewayClient:
    client = TTSGatewayClient(base_url="http://tts-gateway.test", license_token="svc")
    client.set_transport(httpx.MockTransport(handler))
    return client


async def test_tts_gateway_submit_busy_then_success(_fresh: list[float]) -> None:
    seq = _Seq(_busy("4", status=429), httpx.Response(200, json={"batch_id": "b1", "jobs": []}))

    result = await _gateway(seq).submit_batch("b1", [])

    assert result.batch_id == "b1"
    assert len(seq.requests) == 2 and _fresh == [4.0]


async def test_tts_gateway_busy_past_deadline_raises_gpu_busy(_fresh: list[float]) -> None:
    seq = _Seq(_busy("40"))

    with pytest.raises(GpuBusyError):
        await asyncio.wait_for(_gateway(seq).get_batch("b1"), timeout=10)

    assert _fresh == [40.0] and len(seq.requests) == 2


async def test_tts_gateway_402_is_byo_provider_required(_fresh: list[float]) -> None:
    seq = _Seq(_refused_402())

    with pytest.raises(SlotNotConfiguredError) as info:
        await _gateway(seq).submit_batch("b1", [])

    assert info.value.modality == "tts" and len(seq.requests) == 1


async def test_tts_gateway_other_errors_still_raise_for_status() -> None:
    seq = _Seq(httpx.Response(422, json={"detail": "bad chunk"}))

    with pytest.raises(httpx.HTTPStatusError):
        await _gateway(seq).submit_batch("b1", [])


# ── ASR → modality_gateway ──────────────────────────────────────────────────


def _transcription() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "backend": "mlx-whisper",
            "model": "large-v3",
            "language": "en",
            "duration": 1.5,
            "text": "hello world",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.5,
                    "text": "hello world",
                    "words": [{"start": 0.0, "end": 0.6, "word": "hello", "probability": 0.9}],
                }
            ],
        },
    )


def _audio(tmp_path: Path) -> Path:
    p = tmp_path / "clip.wav"
    p.write_bytes(b"RIFF....WAVE")
    return p


async def test_asr_posts_the_gateway_route_with_the_multipart_contract(tmp_path: Path) -> None:
    seq = _Seq(_transcription())
    client = TranscriptionWorkerClient(base_url=GATEWAY, license_token="svc")
    client.set_transport(httpx.MockTransport(seq))

    result = await client.transcribe_file(
        file_path=_audio(tmp_path), language="en", content_type="audio/wav"
    )

    [req] = seq.requests
    assert str(req.url) == f"{GATEWAY}/api/asr/transcribe"
    assert req.url.host != "asr.test"
    body = req.content
    assert b'name="file"; filename="clip.wav"' in body
    assert b'name="word_timestamps"\r\n\r\ntrue' in body
    assert b'name="return_srt"\r\n\r\nfalse' in body
    assert b'name="language"\r\n\r\nen' in body
    assert result.segments[0].words[0].word == "hello"


async def test_asr_busy_then_success_and_deadline_raises_gpu_busy(
    tmp_path: Path, _fresh: list[float]
) -> None:
    ok = _Seq(_busy("6"), _transcription())
    client = TranscriptionWorkerClient(base_url=GATEWAY, license_token="svc")
    client.set_transport(httpx.MockTransport(ok))
    assert (await client.transcribe_file(file_path=_audio(tmp_path))).text == "hello world"
    assert _fresh == [6.0] and len(ok.requests) == 2

    _fresh.clear()
    stuck = _Seq(_busy("50"))
    client2 = TranscriptionWorkerClient(base_url=GATEWAY, license_token="svc")
    client2.set_transport(httpx.MockTransport(stuck))
    with pytest.raises(GpuBusyError) as info:
        await asyncio.wait_for(client2.transcribe_file(file_path=_audio(tmp_path)), timeout=10)
    assert info.value.service == "asr" and _fresh == [50.0]


async def test_asr_402_is_byo_provider_required_for_asr(tmp_path: Path) -> None:
    seq = _Seq(_refused_402())
    client = TranscriptionWorkerClient(base_url=GATEWAY, license_token="svc")
    client.set_transport(httpx.MockTransport(seq))

    with pytest.raises(SlotNotConfiguredError) as info:
        await client.transcribe_file(file_path=_audio(tmp_path))

    assert failure_detail_for(info.value) == {
        "code": "byo_provider_required",
        "modality": "asr",
        "slots": "asr",
        "message": "Configure your own ASR provider to continue",
    }
    assert len(seq.requests) == 1


async def test_asr_concurrency_is_capped_per_process(tmp_path: Path) -> None:
    in_flight = 0
    peak = 0
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await release.wait()
        in_flight -= 1
        return _transcription()

    client = TranscriptionWorkerClient(base_url=GATEWAY, license_token="svc")
    client.set_transport(httpx.MockTransport(handler))
    audio = _audio(tmp_path)
    tasks = [asyncio.create_task(client.transcribe_file(file_path=audio)) for _ in range(5)]
    for _ in range(20):
        await asyncio.sleep(0)
    assert peak == 2 and in_flight == 2  # ASR_MAX_CONCURRENCY=2, the rest queue
    release.set()
    results = await asyncio.gather(*tasks)
    assert len(results) == 5 and peak == 2


# ── in-process TTS: a tenant slot on a platform engine is refused ───────────


def _private_ok_policy() -> ProviderPolicySettings:
    return ProviderPolicySettings.model_validate(
        {"ssrf_mode": "private_ok", "egress_allowlist": "tts.test:9202"}
    )


class _KokoroRec:
    """A tenant ``openai-compatible`` tts connection aimed at OUR kokoro box."""

    tenant_id = TENANT
    slot = "tts"
    connection_id = CONN
    version = 1
    kind = "openai-compatible"
    base_url = "http://tts.test:9202/v1"
    model = "kokoro"
    options: ClassVar[dict[str, object]] = {"voice": "af_heart"}
    secret_ciphertext = None
    purged_at = None


class _Repo:
    async def get_active(self, tenant_id: UUID, slot: str) -> _KokoroRec | None:
        return None

    async def get_version(self, connection_id: UUID, version: int) -> _KokoroRec | None:
        return _KokoroRec()


class _NoDecrypt:
    def decrypt(self, ciphertext: bytes, aad: bytes) -> str:
        raise AssertionError("no secret on this record")


async def test_byo_required_tenant_slot_on_platform_kokoro_opens_no_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posture = StaticByoPosture("byo_required")

    async def _resolve(tenant_id: UUID, ref: SlotRef) -> ResolvedSlot:
        return await resolve_slot(
            _Repo(),
            _NoDecrypt(),
            tenant_id,
            "tts",
            run_snapshot={PROVIDER_SLOTS_KEY: {"tts": ref.to_dict()}},
            policy=_private_ok_policy(),
            posture=posture,
        )

    monkeypatch.setattr(tts_inprocess, "resolve_pinned_tts", _resolve)
    provider_sockets: list[httpx2.Request] = []
    monkeypatch.setattr(
        in_process,
        "inner_transport_factory",
        lambda: httpx2.MockTransport(lambda r: provider_sockets.append(r) or httpx2.Response(599)),
    )
    gateway = _Seq(_tts_ok())
    client = _tts(gateway)
    set_identity_context(str(TENANT), None, None)
    set_correlation_context("run-1", "lesson_narrate")
    bind_run_provider_slots(
        {
            "provider_slots": {
                "tts": {
                    "source": "tenant",
                    "kind": "openai-compatible",
                    "connection_id": str(CONN),
                    "version": 1,
                }
            }
        }
    )
    try:
        with pytest.raises(SlotNotConfiguredError) as info:
            await client.synthesize(text="hello")
    finally:
        set_correlation_context(None, None)

    assert info.value.modality == "tts"
    assert posture.calls == [TENANT]
    assert gateway.requests == [] and provider_sockets == []  # nothing reached any engine


async def test_local_allowed_tenant_may_point_its_slot_at_an_allowlisted_host() -> None:
    got = await resolve_slot(
        _Repo(),
        _NoDecrypt(),
        TENANT,
        "tts",
        run_snapshot={
            PROVIDER_SLOTS_KEY: {
                "tts": {
                    "source": "tenant",
                    "kind": "openai-compatible",
                    "connection_id": str(CONN),
                    "version": 1,
                }
            }
        },
        policy=_private_ok_policy(),
        posture=StaticByoPosture("local_allowed"),
    )
    assert got.source == "tenant" and got.base_url == "http://tts.test:9202/v1"


# ── licences-backed posture: fails CLOSED ───────────────────────────────────


class _Auth:
    async def token(self) -> str:
        return "sa-token"


def _licences(handler: Any) -> tuple[LicencesByoPosture, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def _record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = LicensesClient(
        authenticator=_Auth(),
        base_url="http://licenses.test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_record)),
    )
    return LicencesByoPosture(client), seen


def _mode(mode: str, resolution: str = "row") -> Any:
    return lambda r: httpx.Response(
        200,
        json={"tenant_id": str(TENANT), "byo_mode": mode, "app": "w2v", "resolution": resolution},
    )


@pytest.mark.parametrize("mode", ["local_allowed"])
async def test_licences_posture_reads_the_w2v_per_app_mode(mode: str) -> None:
    reader, seen = _licences(_mode(mode))

    assert await reader.byo_mode(TENANT, slot="tts") == mode
    [req] = seen
    assert req.url.path == f"/tenants/{TENANT}/byo-mode"
    assert req.url.params["app"] == "w2v"
    assert req.url.params["on_unresolved"] == "error"


async def test_tenant_missing_from_the_registry_is_byo_required() -> None:
    reader, _ = _licences(
        lambda r: httpx.Response(
            404, json={"detail": {"error": "tenant_not_found_in_registry", "tenant_selector": "x"}}
        )
    )
    assert await reader.byo_mode(TENANT, slot="image") == "byo_required"


@pytest.mark.parametrize(
    "handler",
    [lambda r: httpx.Response(503, json={"detail": {"error": "byo_mode_unresolved"}})],
)
async def test_licences_failure_fails_closed_and_is_not_cached(handler: Any) -> None:
    reader, seen = _licences(handler)

    for _ in range(2):
        with pytest.raises(ByoPostureUnavailableError) as info:
            await reader.byo_mode(TENANT, slot="asr")
        assert isinstance(info.value, TransientUpstreamError)
        assert attribute_exception(info.value) == EXTERNAL
    assert len(seen) == 2  # a failure is retried on the next call, never remembered


@pytest.mark.parametrize("mode", ["local_allowed"])
async def test_a_licences_outage_after_a_read_serves_the_last_known_posture(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """CommonPractice posture cache: once seen, a licences blip (past the TTL) serves
    the LAST-KNOWN posture instead of refusing every model call of the tenant."""
    monkeypatch.setenv("BYO_POSTURE_CACHE_SECONDS", "0")  # every call re-reads licences
    down = {"on": False}

    def _flaky(request: httpx.Request) -> httpx.Response:
        if down["on"]:
            return httpx.Response(503, json={"detail": {"error": "byo_mode_unresolved"}})
        return _mode(mode)(request)

    reader, seen = _licences(_flaky)
    assert await reader.byo_mode(TENANT, slot="llm") == mode
    down["on"] = True
    assert await reader.byo_mode(TENANT, slot="ocr") == mode  # last-known, not a refusal
    assert len(seen) == 2, "licences WAS asked (outage), the fallback is the last answer"


async def test_a_licences_outage_for_a_never_seen_tenant_fails_closed() -> None:
    reader, _ = _licences(lambda r: httpx.Response(503, json={"detail": {"error": "x"}}))
    with pytest.raises(ByoPostureUnavailableError):
        await reader.byo_mode(TENANT, slot="ocr")


async def test_licences_success_is_cached() -> None:
    reader, seen = _licences(_mode("byo_required"))
    assert await reader.byo_mode(TENANT, slot="tts") == "byo_required"
    assert await reader.byo_mode(TENANT, slot="image") == "byo_required"
    assert len(seen) == 1


# ── failure surfacing through the stage wrapper ─────────────────────────────


def test_byo_refusal_survives_the_stage_retry_wrapper() -> None:
    cause = SlotNotConfiguredError("image")
    try:
        raise StageRetriesExhaustedError(
            stage_name="panel_images", attempts=1, last_error=cause
        ) from cause
    except StageRetriesExhaustedError as wrapped:
        assert attribute_exception(wrapped) == CUSTOMER_INPUT
        assert failure_detail_for(wrapped) == {
            "code": "byo_provider_required",
            "modality": "image",
            "slots": "image",
            "message": "Configure your own image provider to continue",
        }


# ── audiobook QA: busy is an outage (soft-skip), a BYO refusal is not ───────


def _qa_stage(asr_error: BaseException):
    from unittest.mock import AsyncMock

    from src.services.stages.audiobook.qa import AudiobookQAStage

    storage = AsyncMock()
    storage.retrieve_stage_artifact.return_value = b"RIFF....WAVE"
    asr = AsyncMock()
    asr.transcribe_file.side_effect = asr_error
    return AudiobookQAStage(client=AsyncMock(), asr=asr, storage=storage)


def _qa_clips():
    from src.services.stages.audiobook.qa import _QAClip

    return [_QAClip(chunk_id="c1", clip_id="k1", artifact_url="x", manuscript_text="hi")]


async def test_audiobook_qa_soft_skips_a_busy_asr_lane() -> None:
    stage = _qa_stage(GpuBusyError("asr stayed busy", service="asr", status_code=503))
    assert await stage._transcribe_all("run-1", _qa_clips(), {}) == {}


async def test_audiobook_qa_surfaces_a_byo_refusal_instead_of_skipping() -> None:
    stage = _qa_stage(SlotNotConfiguredError("asr"))
    with pytest.raises(SlotNotConfiguredError):
        await stage._transcribe_all("run-1", _qa_clips(), {})


async def test_posture_cache_is_keyed_by_app_as_well_as_tenant() -> None:
    """The per-app override makes the posture app-specific: one app's cached answer
    must never be served for another app of the same tenant."""

    def _by_app(request: httpx.Request) -> httpx.Response:
        mode = "byo_required" if request.url.params["app"] == "w2v" else "local_allowed"
        return httpx.Response(
            200,
            json={"tenant_id": str(TENANT), "byo_mode": mode, "app": request.url.params["app"]},
        )

    w2v_reader, _ = _licences(_by_app)
    other = LicencesByoPosture(w2v_reader._client, app_key="other-app")

    assert await w2v_reader.byo_mode(TENANT, slot="tts") == "byo_required"
    assert await other.byo_mode(TENANT, slot="tts") == "local_allowed"
