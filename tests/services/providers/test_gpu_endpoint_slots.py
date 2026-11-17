"""Tenant GPU slots: ``nonagentic_endpoint`` (tts/asr/image) + ``openai-compatible`` asr/vlm.

Covers the whole chain a tenant-admin's registration travels:

* registry / manifest — the new kinds, their field rules, and the JSON the FE renders;
* resolver — ``api.nonagentic.ai`` (public) is accepted even for a ``byo_required``
  tenant; a platform-internal allow-listed host still is not;
* runtime — ``TTSClient`` / ``TranscriptionWorkerClient`` route a run's TENANT pin to
  the tenant endpoint (platform protocol or OpenAI shape) and never to the gateway;
  a deployment pin keeps the gateway request; an image-bearing LLM call goes to a
  tenant ``vlm`` pin;
* tester — the live probe per new kind, with no secret / upstream body in the verdict;
* ``provider_slot`` payload — the new kind travels to ACG like the existing ones.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID, uuid7

import httpx
import httpx2
import pytest
from pydantic import SecretStr
from w2v_common.auth.identity_context import set_correlation_context, set_identity_context

from src.api.schemas.provider_requirements import ProviderRequirements
from src.auth.run_providers import bind_run_provider_slots, clear_run_provider_slots
from src.clients.transcription_worker_client import TranscriptionWorkerClient
from src.clients.tts_client import TTSClient
from src.services.blueprints.registry import get_registry
from src.services.llm import dispatch, in_process
from src.services.providers import asr_inprocess, tts_inprocess
from src.services.providers import policy as policy_mod
from src.services.providers.byo_posture import StaticByoPosture
from src.services.providers.errors import SlotNotConfiguredError
from src.services.providers.manifest import build_manifest
from src.services.providers.registry import (
    KIND_NONAGENTIC_ENDPOINT,
    KIND_OPENAI_COMPATIBLE,
    SLOTS,
    writable_slots,
)
from src.services.providers.resolver import ResolvedSlot, SlotRef
from src.services.providers.resolver import resolve_slot as _resolve_slot
from src.services.providers.settings import ProviderPolicySettings
from src.services.providers.slot_payload import slot_payload
from src.services.providers.tester import HttpProviderTester
from src.services.stages.upstream import PermanentUpstreamError

TENANT = UUID("01990000-0000-7000-8000-0000000000cc")
CONN = UUID("01990000-0000-7000-8000-000000000003")
SECRET = "lic-tok-SECRET-0123456789abcdef"  # pragma: allowlist secret - fake fixture token
GATEWAY = "http://gateway.test"
PUBLIC_IP = "104.21.15.162"  # what api.nonagentic.ai resolves to (Cloudflare)


@pytest.fixture(autouse=True)
def _context() -> Iterator[None]:
    clear_run_provider_slots()
    set_identity_context(str(TENANT), None, None)
    set_correlation_context(str(uuid7()), "lesson_narrate")
    yield
    clear_run_provider_slots()
    set_correlation_context(None, None)


def _pin(slot: str, kind: str) -> dict[str, Any]:
    return {
        "provider_slots": {
            slot: {"source": "tenant", "kind": kind, "connection_id": str(CONN), "version": 3}
        }
    }


def _slot(
    slot: str,
    kind: str,
    *,
    base_url: str | None = "https://api.nonagentic.ai/tts",
    model: str | None = None,
    options: dict[str, object] | None = None,
    secret: str | None = SECRET,
) -> ResolvedSlot:
    return ResolvedSlot(
        ref=SlotRef(slot=slot, source="tenant", kind=kind, connection_id=CONN, version=3),
        base_url=base_url,
        model=model,
        options=options or {},
        secret=SecretStr(secret) if secret else None,
        pinned_ip=None,
    )


class _Fake:
    """httpx2 provider fake: records requests, answers with ``reply``."""

    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.requests: list[httpx2.Request] = []

    def transport(self) -> httpx2.MockTransport:
        def handler(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            return self.reply(request)

        return httpx2.MockTransport(handler)


def _refusing(cls: type[TTSClient] | type[TranscriptionWorkerClient]) -> Any:
    client = cls(base_url=GATEWAY, license_token="svc-token")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"modality gateway must not be called: {request.url}")

    client.set_transport(httpx.MockTransport(handler))
    return client


# ── registry + manifest ─────────────────────────────────────────────────────


def test_registry_new_kinds_and_writable_gpu_slots() -> None:
    assert {"vlm", "asr"} <= set(writable_slots())
    for slot in ("tts", "asr", "image"):
        ks = SLOTS[slot].kind_spec(KIND_NONAGENTIC_ENDPOINT)
        assert ks is not None, slot
        assert ks.base_url == "required" and not ks.secret_required
        assert ks.label == "Nonagentic-protocol GPU endpoint"
        assert ks.secret_label == "Bearer token"  # pragma: allowlist secret
    assert SLOTS["asr"].tenant_kinds == {KIND_NONAGENTIC_ENDPOINT, KIND_OPENAI_COMPATIBLE}
    assert SLOTS["vlm"].tenant_kinds == {KIND_OPENAI_COMPATIBLE}
    vlm = SLOTS["vlm"].kind_spec(KIND_OPENAI_COMPATIBLE)
    assert vlm is not None and vlm.model_required and vlm.base_url == "required"
    # The platform stays the default for both newly writable slots.
    assert SLOTS["asr"].deployment_kinds == SLOTS["vlm"].deployment_kinds == {"platform"}


def test_manifest_renders_the_new_kind_from_the_registry() -> None:
    m = build_manifest(ProviderPolicySettings(), get_registry())
    ProviderRequirements.model_validate(m.model_dump())  # round-trips the FE contract
    slots = {s.slot: s for s in m.slots}
    assert slots["asr"].allowed_kinds == ["nonagentic_endpoint", "openai-compatible"]
    assert slots["asr"].default_kind == "platform"
    kind = slots["asr"].kinds["nonagentic_endpoint"].model_dump()
    assert kind == {
        "label": "Nonagentic-protocol GPU endpoint",
        "help": (
            "A GPU service speaking the Nonagentic platform protocol (POST "
            "/api/asr/transcribe under the base URL). The token, if set, is sent as "
            "'Authorization: Bearer <token>'."
        ),
        "base_url": "required",
        "default_base_url": None,
        "model_required": False,
        "secret_required": False,
        "secret_label": "Bearer token",  # pragma: allowlist secret - a form label
        "required_options": [],
        "fields": [
            {
                "key": "base_url",
                "label": "Base URL",
                "required": True,
                "help": "https://api.nonagentic.ai/asr",
                "format": "url",
            }
        ],
    }
    assert "nonagentic_endpoint" in slots["tts"].kinds
    assert "nonagentic_endpoint" in slots["image"].kinds
    assert slots["vlm"].kinds["openai-compatible"].model_required


# ── resolver: the public gateway is allowed, platform-internal hosts are not ─


class _Rec:
    def __init__(self, slot: str, kind: str, base_url: str) -> None:
        self.tenant_id = TENANT
        self.slot = slot
        self.connection_id = CONN
        self.version = 1
        self.kind = kind
        self.base_url = base_url
        self.model = None
        self.options: dict[str, object] = {}
        self.secret_ciphertext = None
        self.purged_at = None


class _Repo:
    def __init__(self, rec: _Rec) -> None:
        self.rec = rec

    async def get_active(self, tenant_id: UUID, slot: str) -> _Rec | None:
        return self.rec if slot == self.rec.slot else None

    async def get_version(self, connection_id: UUID, version: int) -> _Rec | None:
        return self.rec


class _NoDecrypt:
    def decrypt(self, ciphertext: bytes, aad: bytes) -> str:  # pragma: no cover
        raise AssertionError("no secret stored")


@pytest.mark.parametrize("slot", ["tts"])
async def test_byo_required_tenant_may_point_at_the_public_api_gateway(
    monkeypatch: pytest.MonkeyPatch, slot: str
) -> None:
    def _validate(url: str, *, allow_hosts: object = None) -> str:
        assert url.startswith("https://api.nonagentic.ai/")
        return PUBLIC_IP

    monkeypatch.setattr(policy_mod, "validate_public_url", _validate)
    rec = _Rec(slot, KIND_NONAGENTIC_ENDPOINT, f"https://api.nonagentic.ai/{slot}")
    got = await _resolve_slot(
        _Repo(rec),
        _NoDecrypt(),
        TENANT,
        slot,
        policy=ProviderPolicySettings(),
        posture=StaticByoPosture("byo_required"),
    )
    assert got.source == "tenant" and got.kind == KIND_NONAGENTIC_ENDPOINT
    assert got.pinned_ip == PUBLIC_IP


async def test_byo_required_tenant_still_cannot_wear_a_platform_internal_host() -> None:
    rec = _Rec("asr", KIND_NONAGENTIC_ENDPOINT, "http://asr.test:9201")
    pol = ProviderPolicySettings(
        ssrf_mode="private_ok", egress_allowlist=frozenset({"asr.test:9201"})
    )
    with pytest.raises(SlotNotConfiguredError):
        await _resolve_slot(
            _Repo(rec), _NoDecrypt(), TENANT, "asr", policy=pol,
            posture=StaticByoPosture("byo_required"),
        )  # fmt: skip


# ── runtime: tts ────────────────────────────────────────────────────────────


def _use(monkeypatch: pytest.MonkeyPatch, module: Any, name: str, slot: ResolvedSlot) -> None:
    async def _resolve(tenant_id: UUID, ref: SlotRef, **_: object) -> ResolvedSlot:
        assert tenant_id == TENANT and ref.connection_id == CONN
        return slot

    monkeypatch.setattr(module, name, _resolve)


@pytest.fixture
def no_meter(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def _record(**kwargs: Any) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr(tts_inprocess, "record_tts_usage", _record)
    monkeypatch.setattr(asr_inprocess, "record_asr_usage", _record)
    return calls


async def test_tenant_nonagentic_tts_posts_the_platform_body_with_bearer(
    monkeypatch: pytest.MonkeyPatch, no_meter: list[dict[str, Any]]
) -> None:
    wire = {
        "audio_base64": base64.b64encode(b"RIFFtenant").decode(),
        "duration_sec": 0.8,
        "sample_rate": 24000,
        "provider": "kokoro",
        "voice": "af_heart",
        "format": "wav",
        "word_timings": [{"word": "Hi", "start": 0.0, "end": 0.3}],
        "speech_seconds": 0.6,
        "pause_seconds": 0.2,
    }
    fake = _Fake(lambda r: httpx2.Response(200, json=wire, headers={"X-TTS-LUFS": "-18.2"}))
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)
    _use(monkeypatch, tts_inprocess, "resolve_pinned_tts", _slot("tts", KIND_NONAGENTIC_ENDPOINT))
    bind_run_provider_slots(_pin("tts", KIND_NONAGENTIC_ENDPOINT))

    result = await _refusing(TTSClient).synthesize(
        "<speak>Hi</speak>", voice="af_heart", speed=1.1, ssml=True
    )

    [req] = fake.requests
    assert str(req.url) == "https://api.nonagentic.ai/tts/api/tts/synthesize"
    assert req.headers["authorization"] == f"Bearer {SECRET}"
    assert req.headers["accept"] == "application/json"
    assert json.loads(req.content) == {
        "text": "<speak>Hi</speak>",
        "speed": 1.1,
        "voice": "af_heart",
        "ssml": True,
    }
    assert result.audio_bytes == b"RIFFtenant"
    assert result.integrated_lufs == -18.2
    assert result.word_timings is not None and result.word_timings[0].word == "Hi"
    assert (result.speech_seconds, result.pause_seconds) == (0.6, 0.2)
    assert len(no_meter) == 1


async def test_tenant_nonagentic_tts_without_a_token_sends_no_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = {"audio_base64": base64.b64encode(b"x").decode(), "duration_sec": 0.1,
            "sample_rate": 24000, "provider": "p", "voice": "v"}  # fmt: skip
    fake = _Fake(lambda r: httpx2.Response(200, json=wire))
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)
    got = await tts_inprocess.call_tenant_tts(
        _slot("tts", KIND_NONAGENTIC_ENDPOINT, secret=None),
        tts_inprocess.TTSRequest(text="hi"),
        timeout=5,
    )
    assert "authorization" not in fake.requests[0].headers
    assert got.synthesis.integrated_lufs is None  # absent header = unproven, not 0


async def test_tenant_nonagentic_tts_rejected_token_never_leaks_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _Fake(lambda r: httpx2.Response(401, text=f"bad token {SECRET}"))
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)
    with pytest.raises(PermanentUpstreamError) as info:
        await tts_inprocess.call_tenant_tts(
            _slot("tts", KIND_NONAGENTIC_ENDPOINT),
            tts_inprocess.TTSRequest(text="hi"),
            timeout=5,
        )
    assert SECRET not in str(info.value)


async def test_tenant_openai_compatible_tts_is_honoured(
    monkeypatch: pytest.MonkeyPatch, no_meter: list[dict[str, Any]]
) -> None:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes(b"\x00\x00" * 2400)
    fake = _Fake(lambda r: httpx2.Response(200, content=buf.getvalue()))
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)
    resolved = _slot(
        "tts", KIND_OPENAI_COMPATIBLE, base_url="https://api.nonagentic.ai/vllm/v1",
        model="kokoro", options={"voice": "af"},
    )  # fmt: skip
    _use(monkeypatch, tts_inprocess, "resolve_pinned_tts", resolved)
    bind_run_provider_slots(_pin("tts", KIND_OPENAI_COMPATIBLE))
    result = await _refusing(TTSClient).synthesize("Hello")
    assert str(fake.requests[0].url) == "https://api.nonagentic.ai/vllm/v1/audio/speech"
    assert abs(result.duration_seconds - 0.1) < 1e-6


# ── runtime: asr ────────────────────────────────────────────────────────────


def _audio(tmp_path: Path) -> Path:
    p = tmp_path / "clip.wav"
    p.write_bytes(b"RIFF....WAVEfmt ")
    return p


async def test_tenant_nonagentic_asr_posts_the_platform_multipart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_meter: list[dict[str, Any]]
) -> None:
    body = {
        "backend": "whisper", "model": "large-v3", "language": "en", "duration": 1.0,
        "text": "hello world",
        "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hello world",
                      "words": [{"start": 0.0, "end": 0.4, "word": "hello"}]}],
    }  # fmt: skip
    fake = _Fake(lambda r: httpx2.Response(200, json=body))
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)
    resolved = _slot("asr", KIND_NONAGENTIC_ENDPOINT, base_url="https://api.nonagentic.ai/asr")
    _use(monkeypatch, asr_inprocess, "resolve_pinned_asr", resolved)
    bind_run_provider_slots(_pin("asr", KIND_NONAGENTIC_ENDPOINT))

    result = await _refusing(TranscriptionWorkerClient).transcribe_file(
        file_path=_audio(tmp_path), language="en", content_type="audio/wav"
    )

    [req] = fake.requests
    assert str(req.url) == "https://api.nonagentic.ai/asr/api/asr/transcribe"
    assert req.headers["authorization"] == f"Bearer {SECRET}"
    content = req.content
    assert b'name="file"; filename="clip.wav"' in content
    assert b'name="word_timestamps"\r\n\r\ntrue' in content
    assert b'name="language"\r\n\r\nen' in content
    assert result.text == "hello world" and result.segments[0].words[0].word == "hello"
    assert len(no_meter) == 1


async def test_tenant_openai_compatible_asr_maps_verbose_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_meter: list[dict[str, Any]]
) -> None:
    body = {
        "task": "transcribe", "language": "english", "duration": 2.0, "text": "one two",
        "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "one"},
                     {"id": 1, "start": 1.0, "end": 2.0, "text": "two"}],
        "words": [{"word": "one", "start": 0.1, "end": 0.5},
                  {"word": "two", "start": 1.2, "end": 1.6}],
    }  # fmt: skip
    fake = _Fake(lambda r: httpx2.Response(200, json=body))
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)
    resolved = _slot("asr", KIND_OPENAI_COMPATIBLE, base_url="https://asr.example.com/v1")
    _use(monkeypatch, asr_inprocess, "resolve_pinned_asr", resolved)
    bind_run_provider_slots(_pin("asr", KIND_OPENAI_COMPATIBLE))

    result = await _refusing(TranscriptionWorkerClient).transcribe_file(file_path=_audio(tmp_path))

    [req] = fake.requests
    assert str(req.url) == "https://asr.example.com/v1/audio/transcriptions"
    content = req.content
    assert b'name="model"\r\n\r\nwhisper-1' in content
    assert b'name="response_format"\r\n\r\nverbose_json' in content
    assert content.count(b'name="timestamp_granularities[]"') == 2
    assert [w.word for w in result.segments[0].words] == ["one"]
    assert [w.word for w in result.segments[1].words] == ["two"]
    assert result.duration == 2.0 and result.backend == KIND_OPENAI_COMPATIBLE


def test_openai_transcript_without_segments_builds_one() -> None:
    got = asr_inprocess.openai_transcript(
        {"text": "hi there", "words": [{"word": "hi", "start": 0.0, "end": 0.2},
                                       {"word": "there", "start": 0.3, "end": 0.7}]},
        model="m", backend="b",
    )  # fmt: skip
    assert len(got.segments) == 1
    assert [w.word for w in got.segments[0].words] == ["hi", "there"]
    assert got.duration == 0.7


async def test_deployment_asr_pin_keeps_the_gateway_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    opened: list[int] = []

    def _factory() -> httpx2.AsyncBaseTransport:
        opened.append(1)
        return httpx2.MockTransport(lambda r: httpx2.Response(599))

    monkeypatch.setattr(in_process, "inner_transport_factory", _factory)
    bind_run_provider_slots(
        {"provider_slots": {"asr": {"source": "deployment", "kind": "platform"}}}
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, json={"backend": "w", "model": "m", "duration": 1.0, "text": "ok"}
        )

    client = TranscriptionWorkerClient(base_url=GATEWAY, license_token="svc")
    client.set_transport(httpx.MockTransport(handler))
    result = await client.transcribe_file(file_path=_audio(tmp_path))
    assert result.text == "ok"
    assert str(seen[0].url) == f"{GATEWAY}/api/asr/transcribe"
    assert opened == []


# ── runtime: vlm ────────────────────────────────────────────────────────────


async def test_image_call_goes_to_the_tenant_vlm_pin_text_call_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved_for: list[str] = []
    calls: list[ResolvedSlot] = []

    async def _resolve(tenant_id: UUID, ref: SlotRef, *, slot: str = "llm") -> ResolvedSlot:
        resolved_for.append(slot)
        return _slot(
            slot, KIND_OPENAI_COMPATIBLE, base_url="https://vlm.example.com/v1", model="qwen-vl"
        )

    async def _call(resolved: ResolvedSlot, request: Any, *, timeout: float) -> Any:
        calls.append(resolved)
        return in_process.LLMResult(
            response=dispatch.WMResponse(
                output="a cat",
                output_type="string",
                output_schema_used=None,
                metadata={},
                usage_metrics={},
            ),
            input_tokens=1, output_tokens=1, model="qwen-vl", provider_kind=resolved.kind,
        )  # fmt: skip

    async def _meter(**_: object) -> bool:
        return True

    monkeypatch.setattr(dispatch, "resolve_pinned_llm", _resolve)
    monkeypatch.setattr(in_process, "call_tenant_llm", _call)
    monkeypatch.setattr(dispatch, "record_in_process_usage", _meter)
    bind_run_provider_slots(_pin("vlm", KIND_OPENAI_COMPATIBLE))
    client = dispatch.WMClient(base_url="http://wm.test", license_token="t")
    blocks: list[dict[str, object]] = [
        {"type": "text", "text": "what is this?"},
        {"type": "image", "data": base64.b64encode(b"png").decode(), "media_type": "image/png"},
    ]
    out = await client.message("vision_worker", content_blocks=blocks)
    assert out.output == "a cat"
    assert resolved_for == ["vlm"] and calls[0].ref.slot == "vlm"

    platform: list[str] = []

    async def _platform(_settings: Any, request: Any, *, timeout: float) -> Any:
        platform.append(request.worker_name)
        return out

    # A text-only call takes the PLATFORM backend (LLM_BACKEND=vllm in the suite).
    monkeypatch.setattr(dispatch.vllm_direct, "call_platform_vllm", _platform)
    monkeypatch.setattr(dispatch, "record_platform_vllm_usage", _meter)
    await client.message("text_worker", "no image here")
    assert platform == ["text_worker"] and resolved_for == ["vlm"]  # vlm pin untouched


# ── provider_slot payload (ACG) ─────────────────────────────────────────────


def test_provider_slot_payload_carries_the_new_kind() -> None:
    got = slot_payload(
        _slot("image", KIND_NONAGENTIC_ENDPOINT, base_url="https://api.nonagentic.ai/image")
    )
    assert got["kind"] == "nonagentic_endpoint"
    assert got["base_url"] == "https://api.nonagentic.ai/image"
    assert got["secret"] == SECRET
    assert got["slot"] == "image"


# ── tester ──────────────────────────────────────────────────────────────────

_ALLOWED = ProviderPolicySettings(ssrf_mode="private_ok", egress_allowlist=frozenset({"gpu.test"}))


class _Upstream:
    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responses.pop(0)


async def _probe(up: _Upstream, slot: str, kind: str, **kw: Any) -> Any:
    tester = HttpProviderTester(_ALLOWED, transport=httpx.MockTransport(up))
    return await tester.test(
        slot=slot,
        kind=kind,
        base_url=kw.get("base_url", "https://gpu.test/x"),
        model=kw.get("model"),
        options=kw.get("options", {}),
        secret=kw.get("secret", SECRET),
    )


async def test_probe_tts_nonagentic_synthesizes_one_word() -> None:
    up = _Upstream(httpx.Response(200, json={"audio_base64": "UklGRg=="}))
    result = await _probe(up, "tts", KIND_NONAGENTIC_ENDPOINT)
    assert result.ok, result
    [req] = up.requests
    assert (req.method, str(req.url)) == ("POST", "https://gpu.test/x/api/tts/synthesize")
    assert json.loads(req.content) == {"text": "test"}
    assert req.headers["authorization"] == f"Bearer {SECRET}"


async def test_probe_asr_nonagentic_is_an_authenticated_post_without_audio() -> None:
    up = _Upstream(httpx.Response(422, json={"detail": "file required"}))
    result = await _probe(up, "asr", KIND_NONAGENTIC_ENDPOINT)
    assert result.ok, result
    [post] = up.requests
    assert (post.method, str(post.url)) == ("POST", "https://gpu.test/x/api/asr/transcribe")
    assert post.headers["authorization"] == f"Bearer {SECRET}"
    assert post.headers["content-type"].startswith("multipart/form-data; boundary=")
    # No audio part: the server must refuse it before transcribing anything.
    assert b'name="file"' not in post.content and b"RIFF" not in post.content


async def test_probe_asr_openai_compatible_falls_back_to_transcriptions() -> None:
    up = _Upstream(httpx.Response(404), httpx.Response(200, json={"text": ""}))
    assert (await _probe(up, "asr", KIND_OPENAI_COMPATIBLE)).ok
    assert str(up.requests[1].url) == "https://gpu.test/x/audio/transcriptions"
    assert b'name="model"\r\n\r\nwhisper-1' in up.requests[1].content


async def test_probe_image_nonagentic_is_an_authenticated_empty_generate() -> None:
    up = _Upstream(httpx.Response(422, json={"detail": "prompt required"}))
    result = await _probe(up, "image", KIND_NONAGENTIC_ENDPOINT)
    assert result.ok, result
    [req] = up.requests
    assert (req.method, str(req.url)) == ("POST", "https://gpu.test/x/api/image/generate")
    assert req.headers["authorization"] == f"Bearer {SECRET}"
    assert json.loads(req.content) == {}  # no prompt: nothing can be generated


@pytest.mark.parametrize("slot", ["image"])
@pytest.mark.parametrize(
    ("status", "ok", "fragment"),
    [(401, False, "token rejected")],
)
async def test_probe_gpu_nonagentic_auth_verdicts(
    slot: str, status: int, ok: bool, fragment: str
) -> None:
    """A wrong token must FAIL: /health (unauthenticated) used to pass it."""
    up = _Upstream(httpx.Response(status, text=f"nope {SECRET}"))
    result = await _probe(up, slot, KIND_NONAGENTIC_ENDPOINT)
    assert result.ok is ok, result
    assert fragment in result.message
    assert result.upstream_status == status
    assert SECRET not in result.message and "nope" not in result.message
    [req] = up.requests
    assert req.method == "POST" and "health" not in str(req.url)


async def test_probe_vlm_chat_fallback_carries_an_image_part() -> None:
    up = _Upstream(httpx.Response(404), httpx.Response(200, json={"choices": []}))
    result = await _probe(up, "vlm", KIND_OPENAI_COMPATIBLE, model="qwen-vl")
    assert result.ok, result
    chat = json.loads(up.requests[1].content)
    parts = chat["messages"][0]["content"]
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_probe_rejection_never_echoes_the_token_or_body() -> None:
    up = _Upstream(httpx.Response(401, text=f"invalid {SECRET}"))
    result = await _probe(up, "tts", KIND_NONAGENTIC_ENDPOINT)
    assert not result.ok and result.upstream_status == 401
    assert SECRET not in result.message and "invalid" not in result.message


async def test_probe_without_base_url_fails_fast() -> None:
    up = _Upstream()
    result = await _probe(up, "asr", KIND_NONAGENTIC_ENDPOINT, base_url=None)
    assert not result.ok and up.requests == []


# ── preflight: ASR is required only where a stage cannot do without it ─────


def test_preflight_requires_asr_for_audiobook_qa_but_not_for_caption_only_compiles() -> None:
    from src.services.providers.preflight import required_byo_modalities

    reg = get_registry()
    by_id = {bp.id: bp for bp in reg.all()}
    assert "asr" in required_byo_modalities(by_id["audiobook_v1"], {})
    # Compile stages transcribe for word-timed captions only, soft-failing without it.
    assert "asr" not in required_byo_modalities(by_id["website_review_v1"], {})
