"""Service lab invoker: the per-slot call, the provider it reports, and Enhance.

No database and no network. ``_resolve`` (the resolver seam) is patched to a fixed
``ResolvedSlot`` and every client is a fake, so these pin the WIRING — which provider
path ran, what rode on the wire, what came back, and what is reported as the serving
provider — without re-testing the clients themselves.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from src.clients.schemas.ai_content_generation import (
    AcgConceptImageJob,
    AcgVideoGenerationResult,
    AcgVideoJobStatus,
    AcgVideoProviderInfo,
    AcgVideoProviderList,
)
from src.clients.schemas.transcription_worker import TranscriptionResult
from src.clients.schemas.tts import TTSSynthesis
from src.config import ProviderOcrSettings
from src.services.providers.errors import SlotUnavailableError
from src.services.providers.resolver import ResolvedSlot, SlotRef
from src.services.service_lab import invoker
from src.services.service_lab.invoker import (
    InvokeRequest,
    ServiceInputError,
    enhance_prompt,
    invoke,
)
from src.services.stages.upstream import TransientUpstreamError

pytestmark = pytest.mark.asyncio

TENANT = UUID("11111111-1111-4111-8111-111111111111")
CONN = UUID("33333333-3333-4333-8333-333333333333")

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32
WAV = b"RIFF" + b"\x00" * 32


def _resolved(
    slot: str, *, source: str = "tenant", kind: str = "openai-compatible"
) -> ResolvedSlot:
    if source == "deployment":
        return ResolvedSlot(ref=SlotRef(slot=slot, source="deployment", kind="nonagentic"))
    return ResolvedSlot(
        ref=SlotRef(slot=slot, source="tenant", kind=kind, connection_id=CONN, version=1),
        base_url="https://provider.example.com/v1",
        model="model-1",
        options={},
        secret=SecretStr("sk-secret"),  # pragma: allowlist secret
        pinned_ip=None,
    )


@pytest.fixture
def no_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """``invoke``/``enhance_prompt`` take a session; these tests never touch it."""
    monkeypatch.setattr(invoker, "_prune_history", _noop)


async def _noop(*_args: object, **_kwargs: object) -> None:
    return None


def _session() -> AsyncSession:
    """``invoke``/``enhance_prompt`` take a session these tests never use: ``_resolve``
    and ``_prune_history`` are patched, so nothing opens a transaction."""
    return cast("AsyncSession", None)


def _patch_resolved(monkeypatch: pytest.MonkeyPatch, resolved: ResolvedSlot) -> None:
    async def _fake_resolve(*_args: object, **_kwargs: object) -> ResolvedSlot:
        return resolved

    monkeypatch.setattr(invoker, "_resolve", _fake_resolve)


# ── image ───────────────────────────────────────────────────────────────────


class _FakeImageClient:
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = None
        self.downloaded: list[str] = []
        self.download_ceiling: int | None = None

    async def create_concept_image(
        self, prompt: str, config: dict | None, *, idempotency_key: str | None = None
    ) -> str:
        assert idempotency_key
        self.payload = {"prompt": prompt, **(config or {})}
        return "job-1"

    async def wait_for_concept_image(self, job_id: str, timeout: float) -> AcgConceptImageJob:
        return AcgConceptImageJob(
            job_id=job_id,
            status="completed",
            provider="local_image",
            download_url="/api/concept-images/assets/a1/download",
        )

    async def download_concept_image(self, url: str, *, max_bytes: int | None = None) -> bytes:
        self.downloaded.append(url)
        self.download_ceiling = max_bytes
        return PNG


async def test_image_tenant_call_sends_the_provider_slot_and_returns_png(
    monkeypatch: pytest.MonkeyPatch, no_session: None
) -> None:
    _patch_resolved(monkeypatch, _resolved("image"))
    client = _FakeImageClient()
    monkeypatch.setattr(invoker, "AIContentGenerationClient", lambda: client)

    bound: dict[str, object] = {}
    monkeypatch.setattr(invoker, "bind_run_provider_slots", lambda snapshot: bound.update(snapshot))

    result = await invoke(
        _session(),
        tenant_id=TENANT,
        request=InvokeRequest(slot="image", prompt="a red fox", width=512, height=512),
    )

    assert client.payload is not None
    assert client.payload["prompt"] == "a red fox"
    assert client.payload["size"] == "512x512"
    # The tenant's connection rides as the run PIN: the ACG client's own
    # _attach_provider_slot reads it from context and puts it on the wire.
    pins = bound["provider_slots"]
    assert isinstance(pins, dict)
    assert pins["image"] == _resolved("image").ref
    assert result.output_kind == "image"
    assert result.media_bytes == PNG
    assert result.media_content_type == "image/png"
    assert result.provider.source == "tenant"
    assert result.provider.kind == "local_image"
    assert client.downloaded == ["/api/concept-images/assets/a1/download"]
    # The ceiling rides the download itself, so an oversized asset is never buffered.
    assert client.download_ceiling == invoker.MAX_MEDIA_BYTES


# ── video ───────────────────────────────────────────────────────────────────


class _FakeVideoClient:
    def __init__(self, providers: AcgVideoProviderList) -> None:
        self.providers = providers
        self.submits = 0
        self.payload: dict[str, object] | None = None

    async def list_video_providers(self) -> AcgVideoProviderList:
        return self.providers

    async def submit_job(
        self, endpoint: str, payload: dict, idempotency_key: str | None = None
    ) -> str:
        self.submits += 1
        self.payload = payload
        return "vid-1"

    async def wait_for_generation(self, job_id: str, timeout: float) -> AcgVideoJobStatus:
        return AcgVideoJobStatus(job_id=job_id, status="completed", provider="stub")

    async def get_generation_result(self, job_id: str) -> AcgVideoGenerationResult:
        return AcgVideoGenerationResult(
            job_id=job_id,
            status="completed",
            video_url=f"/api/content/videos/jobs/{job_id}/download",
        )

    async def download_generated_video(self, url: str, *, max_bytes: int | None = None) -> bytes:
        return MP4


async def test_video_refuses_before_submitting_when_no_provider_is_registered(
    monkeypatch: pytest.MonkeyPatch, no_session: None
) -> None:
    _patch_resolved(monkeypatch, _resolved("video_gen", source="deployment"))
    client = _FakeVideoClient(AcgVideoProviderList(providers=[], default_provider="runway"))
    monkeypatch.setattr(invoker, "AIContentGenerationClient", lambda: client)

    with pytest.raises(SlotUnavailableError) as raised:
        await invoke(
            _session(),
            tenant_id=TENANT,
            request=InvokeRequest(slot="video", prompt="a wave"),
        )

    assert client.submits == 0, "a job must never be submitted when nothing can serve it"
    assert raised.value.slot == "video_gen"


async def test_video_reports_the_serving_provider_name_and_bytes(
    monkeypatch: pytest.MonkeyPatch, no_session: None
) -> None:
    _patch_resolved(monkeypatch, _resolved("video_gen", source="deployment"))
    client = _FakeVideoClient(
        AcgVideoProviderList(
            providers=[AcgVideoProviderInfo(name="stub", available=True)], default_provider="stub"
        )
    )
    monkeypatch.setattr(invoker, "AIContentGenerationClient", lambda: client)

    result = await invoke(
        _session(),
        tenant_id=TENANT,
        request=InvokeRequest(slot="video", prompt="a wave", seconds=4),
    )

    assert client.payload is not None
    assert client.payload["duration_s"] == 4
    assert result.output_kind == "video"
    assert result.media_bytes == MP4
    assert result.media_content_type == "video/mp4"
    assert result.provider.kind == "stub", "the UI states which provider served the clip"
    assert result.provider.source == "deployment"


# ── tts / asr ───────────────────────────────────────────────────────────────


class _FakeSynthesis:
    audio_bytes = WAV
    format = "wav"


class _FakeTtsResult:
    synthesis = _FakeSynthesis()


async def test_tts_tenant_runs_in_process(
    monkeypatch: pytest.MonkeyPatch, no_session: None
) -> None:
    _patch_resolved(monkeypatch, _resolved("tts"))
    seen: dict[str, object] = {}

    async def _fake_call(resolved: ResolvedSlot, request: Any, *, timeout: float) -> Any:
        seen["text"] = request.text
        seen["voice"] = request.voice
        return _FakeTtsResult()

    monkeypatch.setattr(invoker, "call_tenant_tts", _fake_call)

    result = await invoke(
        _session(),
        tenant_id=TENANT,
        request=InvokeRequest(slot="tts", text="hello", voice="af_heart"),
    )

    assert seen == {"text": "hello", "voice": "af_heart"}
    assert result.output_kind == "audio"
    assert result.media_content_type == "audio/wav"
    assert result.provider.source == "tenant"


async def test_tts_platform_uses_the_modality_gateway_client(
    monkeypatch: pytest.MonkeyPatch, no_session: None
) -> None:
    _patch_resolved(monkeypatch, _resolved("tts", source="deployment"))
    used: dict[str, object] = {}

    class _Client:
        async def synthesize(
            self, text: str, voice: str | None = None, speed: float = 1.0
        ) -> TTSSynthesis:
            used["text"] = text
            return TTSSynthesis(
                audio_bytes=WAV,
                sample_rate=24_000,
                duration_seconds=1.0,
                provider="kokoro",
                voice=voice or "default",
                format="wav",
            )

    monkeypatch.setattr(invoker, "TTSClient", _Client)

    result = await invoke(
        _session(),
        tenant_id=TENANT,
        request=InvokeRequest(slot="tts", text="hello"),
    )

    assert used["text"] == "hello"
    assert result.media_bytes == WAV
    assert result.provider.source == "deployment"


async def test_asr_needs_an_audio_upload(monkeypatch: pytest.MonkeyPatch, no_session: None) -> None:
    _patch_resolved(monkeypatch, _resolved("asr"))

    # ServiceInputError, not a bare ValueError: the route maps THIS class to 422.
    # A plain ValueError is a defect here and stays a 500 — the distinction is
    # the whole point, so it is asserted rather than left to `match=`.
    with pytest.raises(ServiceInputError, match="audio upload"):
        await invoke(
            _session(),
            tenant_id=TENANT,
            request=InvokeRequest(slot="asr", image_bytes=PNG, image_content_type="image/png"),
        )


async def test_vlm_refuses_the_platform_path_without_a_vision_model(
    monkeypatch: pytest.MonkeyPatch, no_session: None
) -> None:
    """The platform default behind ``vlm`` IS the text LLM.

    On 2026-11-02 it could not read images (pgx vLLM served a text-only model),
    the image part was a 400 upstream and the page said "the provider failed".
    The bench now refuses the platform path as UNAVAILABLE — 409, never 502 —
    until a deployment actually serves a vision model. See
    ``ProviderPolicySettings.platform_vlm_enabled``.
    """
    _patch_resolved(monkeypatch, _resolved("vlm", source="deployment"))

    with pytest.raises(SlotUnavailableError, match="vision model"):
        await invoke(
            _session(),
            tenant_id=TENANT,
            request=InvokeRequest(
                slot="vlm",
                prompt="what does this say?",
                image_bytes=PNG,
                image_content_type="image/png",
            ),
        )


async def test_asr_transcribes_the_uploaded_clip(
    monkeypatch: pytest.MonkeyPatch, no_session: None
) -> None:
    _patch_resolved(monkeypatch, _resolved("asr"))
    seen: dict[str, object] = {}

    async def _fake_call(resolved: ResolvedSlot, request: Any, *, timeout: float) -> Any:
        seen["path"] = request.file_path
        seen["exists"] = request.file_path.exists()
        seen["language"] = request.language
        return TranscriptionResult(
            text="hello world", backend="whisper", model="large", duration=1.0
        )

    monkeypatch.setattr(invoker, "call_tenant_asr", _fake_call)

    result = await invoke(
        _session(),
        tenant_id=TENANT,
        request=InvokeRequest(
            slot="asr", image_bytes=WAV, image_content_type="audio/wav", language="en"
        ),
    )

    assert seen["exists"] is True
    assert str(seen["path"]).endswith(".wav")
    assert seen["language"] == "en"
    assert result.output_kind == "transcript"
    assert result.text == "hello world"


async def test_asr_accepts_a_browser_octet_stream_upload(
    monkeypatch: pytest.MonkeyPatch, no_session: None
) -> None:
    """A browser upload is often ``application/octet-stream``: forward it, don't refuse."""
    _patch_resolved(monkeypatch, _resolved("asr", source="deployment"))
    seen: dict[str, object] = {}

    class _Client:
        async def transcribe_file(self, **kwargs: object) -> TranscriptionResult:
            seen.update(kwargs)
            return TranscriptionResult(text="ok", backend="whisper", model="large", duration=1.0)

    monkeypatch.setattr(invoker, "TranscriptionWorkerClient", _Client)

    result = await invoke(
        _session(),
        tenant_id=TENANT,
        request=InvokeRequest(
            slot="asr",
            image_bytes=WAV,
            image_content_type="application/octet-stream",
            language="nl",
        ),
    )

    assert seen["language"] == "nl"
    assert seen["content_type"] == "application/octet-stream"
    assert result.text == "ok"


# ── ocr ─────────────────────────────────────────────────────────────────────


async def test_ocr_without_a_provider_raises_the_existing_not_configured_error(
    monkeypatch: pytest.MonkeyPatch, no_session: None
) -> None:
    from src.clients.ocr_client import OcrNotConfiguredError

    _patch_resolved(monkeypatch, _resolved("ocr", source="deployment"))

    class _Settings:
        provider_ocr = ProviderOcrSettings()

    monkeypatch.setattr(invoker, "get_settings", lambda: _Settings())

    with pytest.raises(OcrNotConfiguredError):
        await invoke(
            _session(),
            tenant_id=TENANT,
            request=InvokeRequest(slot="ocr", image_bytes=PNG, image_content_type="image/png"),
        )


# ── timeout ─────────────────────────────────────────────────────────────────


async def test_a_slow_slot_raises_a_clear_timeout_error(
    monkeypatch: pytest.MonkeyPatch, no_session: None
) -> None:
    _patch_resolved(monkeypatch, _resolved("tts"))

    async def _slow(*_args: object, **_kwargs: object) -> Any:
        import asyncio

        await asyncio.sleep(1.0)
        return _FakeTtsResult()

    monkeypatch.setattr(invoker, "call_tenant_tts", _slow)

    with pytest.raises(TimeoutError, match="did not answer within"):
        await invoke(
            _session(),
            tenant_id=TENANT,
            request=InvokeRequest(slot="tts", text="hello"),
            timeout_seconds=0.05,
        )


# ── enhance ─────────────────────────────────────────────────────────────────


async def test_enhance_uses_the_target_instruction_on_the_tenant_llm(
    monkeypatch: pytest.MonkeyPatch, no_session: None
) -> None:
    _patch_resolved(monkeypatch, _resolved("llm"))
    seen: dict[str, object] = {}

    class _Response:
        output = "  a lone lighthouse at dusk, 35mm, golden hour  "

    class _Result:
        response = _Response()

    async def _fake_call(resolved: ResolvedSlot, request: Any, *, timeout: float) -> Any:
        seen["system"] = request.system_prompt
        seen["message"] = request.message
        seen["max_tokens"] = request.max_tokens
        return _Result()

    monkeypatch.setattr(invoker, "call_tenant_llm", _fake_call)

    text = await enhance_prompt(
        _session(),
        tenant_id=TENANT,
        prompt="a lighthouse",
        target="image",
    )

    assert text == "a lone lighthouse at dusk, 35mm, golden hour"
    system = str(seen["system"])
    assert "camera" in system.lower() and "composition" in system.lower()
    assert "ONLY the rewritten prompt" in system
    assert seen["message"] == "a lighthouse"
    assert seen["max_tokens"] == invoker._ENHANCE_TOKEN_BUDGET


async def test_enhance_platform_goes_through_the_wm_dispatch_not_a_hardcoded_url(
    monkeypatch: pytest.MonkeyPatch, no_session: None
) -> None:
    _patch_resolved(monkeypatch, _resolved("llm", source="deployment"))
    seen: dict[str, object] = {}

    class _Response:
        output = "clarified task with explicit output constraints"

    class _Wm:
        def __init__(self, timeout: float | None = None) -> None:
            seen["timeout"] = timeout

        async def message(
            self, worker_name: str, message: str | None = None, **kwargs: object
        ) -> Any:
            seen["worker"] = worker_name
            seen["message"] = message
            seen["system"] = kwargs.get("system_prompt")
            return _Response()

    monkeypatch.setattr(invoker, "WMClient", _Wm)

    text = await enhance_prompt(
        _session(),
        tenant_id=TENANT,
        prompt="summarise this",
        target="llm",
    )

    assert text == "clarified task with explicit output constraints"
    assert seen["worker"] == "service_lab_enhance"
    assert "output format" in str(seen["system"]).lower()


async def test_enhance_blank_reply_raises_instead_of_returning_empty(
    monkeypatch: pytest.MonkeyPatch, no_session: None
) -> None:
    _patch_resolved(monkeypatch, _resolved("llm"))

    class _Response:
        output = "   \n "

    class _Result:
        response = _Response()

    async def _fake_call(resolved: ResolvedSlot, request: Any, *, timeout: float) -> Any:
        return _Result()

    monkeypatch.setattr(invoker, "call_tenant_llm", _fake_call)

    with pytest.raises(TransientUpstreamError, match="empty prompt"):
        await enhance_prompt(
            _session(),
            tenant_id=TENANT,
            prompt="a lighthouse",
            target="video",
        )


async def test_every_slot_declares_its_registry_name() -> None:
    assert invoker.SLOTS == ("image", "tts", "asr", "llm", "vlm", "ocr", "video")
    assert invoker._REGISTRY_SLOT == {"video": "video_gen"}
    assert tuple(invoker.enhance_targets) == ("image", "video", "llm", "vlm")
