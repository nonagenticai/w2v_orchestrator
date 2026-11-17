"""P5-1 — ``TTSClient.synthesize`` dispatches a run's ``tts`` slot.

* unpinned / deployment pin → the modality-gateway request, byte-identical to today;
  the in-process transport is never opened.
* tenant pin → in-process synthesis on the pinned connection through a fake provider;
  the modality gateway is never called, failures never fall back to it, the key never
  leaks, usage is metered.
* audiobook_narrate: a tenant pin bypasses the tts_gateway claim-queue.
"""

from __future__ import annotations

import base64
import io
import json
import struct
import wave
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4, uuid7

import httpx
import httpx2
import pytest
from pydantic import SecretStr
from w2v_common.auth.identity_context import set_correlation_context, set_identity_context

from src.auth.run_providers import bind_run_provider_slots, clear_run_provider_slots
from src.clients.schemas.tts import TTSSynthesis
from src.clients.tts_client import TTSClient
from src.models.failure_class import CUSTOMER_INPUT, EXTERNAL
from src.services.llm import in_process
from src.services.providers import tts_inprocess
from src.services.providers.errors import SlotUnavailableError
from src.services.providers.resolver import ResolvedSlot, SlotRef
from src.services.stages.upstream import PermanentUpstreamError, TransientUpstreamError

TENANT = UUID("01990000-0000-7000-8000-0000000000bb")
CONN_ID = UUID("01990000-0000-7000-8000-000000000002")
RUN_ID = uuid7()
EL_VOICE = "21m00Tcm4TlvDq8ikWAM"  # pragma: allowlist secret -- public ElevenLabs sample voice id
SECRET = "sk-tts-SECRET-abcdef012345"  # pragma: allowlist secret - fake fixture key
GATEWAY = "http://tts.test"


def _pin(kind: str) -> dict[str, Any]:
    return {
        "provider_slots": {
            "tts": {
                "source": "tenant",
                "kind": kind,
                "connection_id": str(CONN_ID),
                "version": 2,
            }
        }
    }


DEPLOYMENT_PIN = {"provider_slots": {"tts": {"source": "deployment", "kind": "nonagentic"}}}


def _slot(
    kind: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    options: dict[str, object] | None = None,
    pinned_ip: str | None = None,
    secret: str | None = SECRET,
) -> ResolvedSlot:
    return ResolvedSlot(
        ref=SlotRef(slot="tts", source="tenant", kind=kind, connection_id=CONN_ID, version=2),
        base_url=base_url,
        model=model,
        options=options or {},
        secret=SecretStr(secret) if secret else None,
        pinned_ip=pinned_ip,
    )


def _pcm(n_frames: int) -> bytes:
    return b"".join(struct.pack("<h", (i * 37) % 2000 - 1000) for i in range(n_frames))


def _streamed_wav(pcm: bytes, rate: int = 24_000) -> bytes:
    """A WAV the way OpenAI streams it: RIFF and data sizes are 0xFFFFFFFF placeholders."""
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    return (
        b"RIFF"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + fmt
        + b"data"
        + struct.pack("<I", 0xFFFFFFFF)
        + pcm
    )


class _Fake:
    """httpx2 provider fake: records requests, answers with ``reply``."""

    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.requests: list[httpx2.Request] = []

    def transport(self) -> httpx2.MockTransport:
        def handler(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            return self.reply(request, len(self.requests) - 1)

        return httpx2.MockTransport(handler)


@pytest.fixture(autouse=True)
def _context(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    clear_run_provider_slots()
    set_identity_context(str(TENANT), None, None)
    set_correlation_context(str(RUN_ID), "lesson_narrate")

    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(in_process, "_sleep", _no_sleep)
    yield
    clear_run_provider_slots()
    set_correlation_context(None, None)


@pytest.fixture
def no_in_process(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Any socket the in-process path opens is recorded (and refused)."""
    opened: list[int] = []

    def _factory() -> httpx2.AsyncBaseTransport:
        opened.append(1)
        return httpx2.MockTransport(lambda r: httpx2.Response(599))

    monkeypatch.setattr(in_process, "inner_transport_factory", _factory)
    return opened


@pytest.fixture
def usage(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def _record(**kwargs: Any) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr(tts_inprocess, "record_tts_usage", _record)
    return calls


def _use_slot(monkeypatch: pytest.MonkeyPatch, slot: ResolvedSlot) -> list[SlotRef]:
    seen: list[SlotRef] = []

    async def _resolve(tenant_id: UUID, ref: SlotRef) -> ResolvedSlot:
        assert tenant_id == TENANT
        seen.append(ref)
        return slot

    monkeypatch.setattr(tts_inprocess, "resolve_pinned_tts", _resolve)
    return seen


def _gateway_body() -> dict[str, Any]:
    return {
        "audio_base64": base64.b64encode(b"RIFFgateway").decode(),
        "duration_sec": 1.5,
        "sample_rate": 24000,
        "provider": "kokoro",
        "voice": "af_heart",
        "format": "wav",
    }


async def _client_capturing(captured: list[httpx.Request]) -> TTSClient:
    client = TTSClient(base_url=GATEWAY, license_token="svc-token")
    real = await client._get_client()  # real headers + identity hooks, fake socket

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_gateway_body(), headers={"X-TTS-LUFS": "-18.0"})

    real._transport = httpx.MockTransport(handler)
    return client


async def _gateway_request(**kwargs: Any) -> tuple[httpx.Request, TTSSynthesis]:
    captured: list[httpx.Request] = []
    client = await _client_capturing(captured)
    try:
        result = await client.synthesize(**kwargs)
    finally:
        await client.close()
    assert len(captured) == 1
    return captured[0], result


def _gateway_refusing() -> TTSClient:
    client = TTSClient(base_url=GATEWAY, license_token="svc-token")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"modality gateway must not be called: {request.url}")

    client.set_transport(httpx.MockTransport(handler))
    return client


# ── unpinned / deployment: the modality-gateway request, unchanged ──────────


async def test_unpinned_synthesize_is_todays_gateway_request(no_in_process: list[int]) -> None:
    request, result = await _gateway_request(text="Hello world.", voice="af_heart", speed=1.1)

    assert request.method == "POST"
    assert str(request.url) == f"{GATEWAY}/api/tts/synthesize"
    assert json.loads(request.content) == {
        "text": "Hello world.",
        "speed": 1.1,
        "voice": "af_heart",
    }
    assert request.headers["Accept"] == "application/json"
    assert request.headers["X-Tenant-Id"] == "Bearer svc-token"
    assert result.audio_bytes == b"RIFFgateway"
    assert result.integrated_lufs == -18.0
    assert no_in_process == []


async def test_deployment_pin_request_is_byte_identical_to_unpinned(
    no_in_process: list[int],
) -> None:
    call = {"text": "<speak>Hi</speak>", "voice": "am_adam", "speed": 0.9, "ssml": True}
    unpinned, _ = await _gateway_request(**call)
    bind_run_provider_slots(DEPLOYMENT_PIN)
    pinned, _ = await _gateway_request(**call)

    assert pinned.method == unpinned.method
    assert pinned.url == unpinned.url
    assert pinned.content == unpinned.content
    assert dict(pinned.headers) == dict(unpinned.headers)
    assert no_in_process == []


# ── tenant: in-process on the pinned connection ─────────────────────────────


async def test_tenant_openai_slot_synthesizes_via_provider_not_gateway(
    monkeypatch: pytest.MonkeyPatch, usage: list[dict[str, Any]]
) -> None:
    pcm = _pcm(12_000)  # 0.5 s at 24 kHz
    fake = _Fake(lambda req, n: httpx2.Response(200, content=_streamed_wav(pcm)))
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)
    seen = _use_slot(monkeypatch, _slot("openai", options={"voice": "nova"}))
    bind_run_provider_slots(_pin("openai"))

    result = await _gateway_refusing().synthesize(
        "Hello there world.", voice="af_heart", speed=1.25, provider="kokoro"
    )

    assert [r.version for r in seen] == [2]
    assert len(fake.requests) == 1
    req = fake.requests[0]
    assert str(req.url) == "https://api.openai.com/v1/audio/speech"
    assert req.headers["Authorization"] == f"Bearer {SECRET}"
    assert json.loads(req.content) == {
        "model": "gpt-4o-mini-tts",
        "input": "Hello there world.",
        "voice": "nova",  # the connection's voice wins over the kokoro id
        "response_format": "wav",
        "speed": 1.25,
    }
    # Same response type as the gateway path; the placeholder sizes are repaired.
    assert isinstance(result, TTSSynthesis)
    assert result.sample_rate == 24_000
    assert result.duration_seconds == pytest.approx(0.5)
    with wave.open(io.BytesIO(result.audio_bytes)) as wav:
        assert wav.getnframes() == 12_000
        assert wav.readframes(12_000) == pcm
    assert result.provider == "openai" and result.voice == "nova"
    # Unknown → None so the existing fallbacks run.
    assert result.integrated_lufs is None
    assert result.word_timings is None
    assert result.speech_seconds is None and result.pause_seconds is None
    # Metered: slot provider + characters.
    assert len(usage) == 1
    assert usage[0]["tenant_id"] == TENANT
    assert usage[0]["result"].characters == len("Hello there world.")
    assert usage[0]["result"].provider_kind == "openai"


async def test_tenant_openai_compatible_connects_to_pinned_ip_and_strips_ssml(
    monkeypatch: pytest.MonkeyPatch, usage: list[dict[str, Any]]
) -> None:
    fake = _Fake(lambda req, n: httpx2.Response(200, content=_streamed_wav(_pcm(2400))))
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)
    _use_slot(
        monkeypatch,
        _slot(
            "openai-compatible",
            base_url="https://tts.tenant.example/v1",
            model="kokoro-82m",
            options={"voice": "af_sky"},
            pinned_ip="203.0.113.9",
            secret=None,
        ),
    )
    bind_run_provider_slots(_pin("openai-compatible"))

    await _gateway_refusing().synthesize(
        '<speak><break time="400ms"/>R&amp;D is <emphasis level="strong">fun</emphasis></speak>',
        ssml=True,
    )

    req = fake.requests[0]
    assert req.url.host == "203.0.113.9"  # DNS-rebinding pin
    assert req.headers["Host"] == "tts.tenant.example"
    assert req.url.path == "/v1/audio/speech"
    assert "Authorization" not in req.headers  # keyless self-hosted server
    body = json.loads(req.content)
    assert body["input"] == "R&D is fun"
    assert body["model"] == "kokoro-82m" and body["voice"] == "af_sky"


async def test_tenant_elevenlabs_alignment_becomes_word_timings(
    monkeypatch: pytest.MonkeyPatch, usage: list[dict[str, Any]]
) -> None:
    text = "Hi  there, you"
    chars = list(text)
    starts = [round(i * 0.1, 2) for i in range(len(chars))]
    ends = [round(s + 0.08, 2) for s in starts]
    pcm = _pcm(48_000)  # 2.0 s

    def reply(req: httpx2.Request, n: int) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "audio_base64": base64.b64encode(pcm).decode(),
                "alignment": {
                    "characters": chars,
                    "character_start_times_seconds": starts,
                    "character_end_times_seconds": ends,
                },
            },
        )

    fake = _Fake(reply)
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)
    _use_slot(monkeypatch, _slot("elevenlabs", options={"voice": EL_VOICE}))
    bind_run_provider_slots(_pin("elevenlabs"))

    result = await _gateway_refusing().synthesize(text, voice="af_heart", speed=1.0)

    req = fake.requests[0]
    assert req.url.host == "api.elevenlabs.io"
    assert req.url.path == f"/v1/text-to-speech/{EL_VOICE}/with-timestamps"
    assert req.url.params["output_format"] == "pcm_24000"
    assert req.headers["xi-api-key"] == SECRET
    assert json.loads(req.content) == {"text": text, "model_id": "eleven_multilingual_v2"}
    assert result.sample_rate == 24_000
    assert result.duration_seconds == pytest.approx(2.0)
    with wave.open(io.BytesIO(result.audio_bytes)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (24_000, 1, 2)
        assert wav.readframes(48_000) == pcm
    assert result.word_timings is not None
    assert [(w.word, w.start, w.end) for w in result.word_timings] == [
        ("Hi", 0.0, 0.18),
        ("there,", 0.4, 0.98),
        ("you", 1.1, 1.38),
    ]


def test_malformed_alignment_yields_none_not_a_guess() -> None:
    assert tts_inprocess.word_timings_from_alignment(None) is None
    assert (
        tts_inprocess.word_timings_from_alignment(
            {
                "characters": ["a", "b"],
                "character_start_times_seconds": [0.0],
                "character_end_times_seconds": [0.1, 0.2],
            }
        )
        is None
    )


# ── tenant failures: mapped, no fallback, no key leak ───────────────────────


async def test_tenant_auth_failure_is_permanent_and_never_falls_back(
    monkeypatch: pytest.MonkeyPatch, usage: list[dict[str, Any]]
) -> None:
    fake = _Fake(
        lambda req, n: httpx2.Response(401, json={"error": {"message": f"bad key {SECRET}"}})
    )
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)
    _use_slot(monkeypatch, _slot("openai", options={"voice": "nova"}))
    bind_run_provider_slots(_pin("openai"))

    with pytest.raises(PermanentUpstreamError) as info:
        await _gateway_refusing().synthesize("Hello.")

    assert info.value.failure_class == CUSTOMER_INPUT
    assert SECRET not in str(info.value)
    assert len(fake.requests) == 1  # auth errors are not retried
    assert usage == []


async def test_tenant_outage_is_transient_after_retries_without_fallback(
    monkeypatch: pytest.MonkeyPatch, usage: list[dict[str, Any]]
) -> None:
    fake = _Fake(lambda req, n: httpx2.Response(503, text=f"overloaded key={SECRET}"))
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)
    _use_slot(monkeypatch, _slot("elevenlabs", options={"voice": "v1"}))
    bind_run_provider_slots(_pin("elevenlabs"))

    with pytest.raises(TransientUpstreamError) as info:
        await _gateway_refusing().synthesize("Hello.")

    assert info.value.failure_class == EXTERNAL
    assert SECRET not in str(info.value)
    assert len(fake.requests) > 1  # the shared in-process retry policy ran
    assert usage == []


async def test_tenant_transport_error_is_transient(
    monkeypatch: pytest.MonkeyPatch, usage: list[dict[str, Any]]
) -> None:
    def reply(req: httpx2.Request, n: int) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=req)

    fake = _Fake(reply)
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)
    _use_slot(monkeypatch, _slot("openai", options={"voice": "nova"}))
    bind_run_provider_slots(_pin("openai"))

    with pytest.raises(TransientUpstreamError, match="unreachable"):
        await _gateway_refusing().synthesize("Hello.")


async def test_tenant_non_wav_reply_is_permanent(
    monkeypatch: pytest.MonkeyPatch, usage: list[dict[str, Any]]
) -> None:
    fake = _Fake(lambda req, n: httpx2.Response(200, content=b"ID3 not a wav"))
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)
    _use_slot(monkeypatch, _slot("openai", options={"voice": "nova"}))
    bind_run_provider_slots(_pin("openai"))

    with pytest.raises(PermanentUpstreamError, match="unusable response"):
        await _gateway_refusing().synthesize("Hello.")


async def test_tenant_pin_without_bound_tenant_fails_closed(no_in_process: list[int]) -> None:
    set_identity_context(None, None, None)
    bind_run_provider_slots(_pin("openai"))

    with pytest.raises(SlotUnavailableError):
        await _gateway_refusing().synthesize("Hello.")
    assert no_in_process == []


# ── metering row ────────────────────────────────────────────────────────────


async def test_record_tts_usage_writes_tenant_byo_characters_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.database.session as session_mod
    from src.database.repositories.run_usage_repository import IN_PROCESS_USAGE_SOURCE

    added: list[Any] = []
    bound: list[UUID] = []

    class _Session:
        def add(self, row: Any) -> None:
            added.append(row)

    class _Ctx:
        async def __aenter__(self) -> _Session:
            return _Session()

        async def __aexit__(self, *exc: object) -> None:
            return None

    async def _bind(_session: Any, tenant_id: UUID) -> None:
        bound.append(tenant_id)

    monkeypatch.setattr(session_mod, "get_db_session", lambda: _Ctx())
    monkeypatch.setattr(session_mod, "bind_session_tenant", _bind)
    slot = _slot("elevenlabs", model="eleven_turbo_v2")
    result = tts_inprocess.TTSResult(
        synthesis=TTSSynthesis(
            audio_bytes=b"x", sample_rate=24_000, duration_seconds=0.1, provider="e", voice="v"
        ),
        characters=42,
        model="eleven_turbo_v2",
        provider_kind="elevenlabs",
    )

    assert await tts_inprocess.record_tts_usage(tenant_id=TENANT, resolved=slot, result=result)

    assert bound == [TENANT]
    (row,) = added
    assert row.run_id == RUN_ID
    assert row.stage_name == "lesson_narrate"
    assert row.provider == "elevenlabs" and row.model == "eleven_turbo_v2"
    assert row.input_tokens == 0 and row.output_tokens == 0 and row.cost_usd == 0
    assert row.metadata_["source"] == IN_PROCESS_USAGE_SOURCE
    assert row.metadata_["funding"] == "tenant_byo"
    assert row.metadata_["slot"] == "tts"
    assert row.metadata_["characters"] == 42
    assert row.metadata_["connection_id"] == str(CONN_ID)
    assert SECRET not in json.dumps(row.metadata_)
    # P6 ORCH-3 attribution columns.
    assert row.slot == "tts" and row.funding == "tenant_byo"
    assert row.units == 42 and row.unit_kind == "chars"
    # P7 / R3 (migration 0088): the serving connection + pinned version, as columns.
    assert row.connection_id == CONN_ID and row.connection_version == 2


# ── audiobook: tenant slot bypasses the tts_gateway claim-queue ─────────────


def _audiobook_ctx() -> Any:
    from src.services.runs.run_context import RunContext

    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"voice_id": "af_heart", "speed": 1.0}},
        prior_run_outputs={
            "audiobook_plan": {
                "chapters": [
                    {
                        "id": "ch1",
                        "title": "One",
                        "sequence": 0,
                        "chunks": [
                            {"id": "k1", "text": "Alpha.", "sequence": 0},
                            {"id": "k2", "text": "Beta.", "sequence": 1},
                        ],
                    }
                ],
                "total_chunks": 2,
            }
        },
    )


class _FakeStorage:
    async def initialize(self) -> None:
        pass

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        return f"artifacts/{job_id}/{stage_name}/{filename}"

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        return f"https://minio.test/{path}"


@pytest.fixture
def gateway_flag_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    from src.config import clear_settings_cache
    from src.services import tts_storage

    monkeypatch.setenv("DOWNSTREAM_AUDIOBOOK_NARRATE_VIA_GATEWAY", "true")
    clear_settings_cache()
    tts_storage.set_storage_for_tests(_FakeStorage())
    constructed: list[int] = []

    def _gateway(*_a: object, **_k: object) -> AsyncMock:
        constructed.append(1)
        raise RuntimeError("tts_gateway constructed")

    monkeypatch.setattr("src.services.stages.audiobook.narrate.TTSGatewayClient", _gateway)
    yield constructed
    tts_storage.set_storage_for_tests(None)
    clear_settings_cache()


def _direct_client() -> AsyncMock:
    direct = AsyncMock()
    direct.synthesize.return_value = TTSSynthesis(
        audio_bytes=b"\x01", sample_rate=24_000, duration_seconds=2.0, provider="openai", voice="v"
    )
    return direct


async def test_audiobook_tenant_tts_bypasses_gateway(gateway_flag_on: list[int]) -> None:
    from src.services.stages.audiobook.narrate import AudiobookNarrateStage

    bind_run_provider_slots(_pin("openai"))
    direct = _direct_client()

    out = await AudiobookNarrateStage(client=direct).run({}, ctx=_audiobook_ctx())

    assert gateway_flag_on == []
    assert direct.synthesize.await_count == 2
    assert len(out.clips) == 2


async def test_audiobook_deployment_tts_still_uses_gateway(gateway_flag_on: list[int]) -> None:
    """Control: same flag, deployment pin → the claim-queue path is taken."""
    from src.services.stages.audiobook.narrate import AudiobookNarrateStage

    bind_run_provider_slots(DEPLOYMENT_PIN)
    direct = _direct_client()

    with pytest.raises(RuntimeError, match="tts_gateway constructed"):
        await AudiobookNarrateStage(client=direct).run({}, ctx=_audiobook_ctx())

    assert gateway_flag_on == [1]
    assert direct.synthesize.await_count == 0
