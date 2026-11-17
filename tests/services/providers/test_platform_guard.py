"""The in-process PLATFORM choke point (review of #1150, defect 1).

A byo_required tenant must never reach a platform model, whatever the entry:
an unbound (request-context) call, an optional slot the preflight skips, or a run
pinned to the platform before the flip. A call with NO tenant bound (cron /
background) stays on the platform — CommonPractice: no tenant ⇒ platform.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from w2v_common.auth.identity_context import clear_identity_context, set_identity_context

from src.auth.run_providers import bind_run_provider_slots, clear_run_provider_slots
from src.services.providers.byo_posture import ByoMode, StaticByoPosture
from src.services.providers.errors import SlotNotConfiguredError
from src.services.providers.platform_guard import refuse_platform_for_bound_tenant
from src.services.providers.resolver import PROVIDER_SLOTS_KEY

TENANT = UUID("0199bbbb-0000-7000-8000-0000000000aa")
GOVERNED = ("llm", "vlm", "tts", "asr", "image", "video_gen", "ocr")


@pytest.fixture(autouse=True)
def _clean_context() -> Iterator[None]:
    clear_identity_context()
    clear_run_provider_slots()
    yield
    clear_identity_context()
    clear_run_provider_slots()


def _posture(monkeypatch: pytest.MonkeyPatch, mode: ByoMode) -> StaticByoPosture:
    posture = StaticByoPosture(mode)
    monkeypatch.setattr("src.services.providers.adapters.get_byo_posture", lambda: posture)
    return posture


def _pin(slot: str, source: str) -> None:
    kind = "openai" if source == "tenant" else "platform"
    entry: dict[str, object] = {"source": source, "kind": kind}
    if source == "tenant":
        entry |= {"connection_id": str(UUID(int=7)), "version": 1}
    bind_run_provider_slots({PROVIDER_SLOTS_KEY: {slot: entry}})


@pytest.mark.parametrize("slot", ["llm"])
async def test_an_unbound_call_with_a_tenant_is_refused_under_byo_required(
    monkeypatch: pytest.MonkeyPatch, slot: str
) -> None:
    _posture(monkeypatch, "byo_required")
    set_identity_context(str(TENANT), None, None)
    with pytest.raises(SlotNotConfiguredError) as info:
        await refuse_platform_for_bound_tenant(slot)
    assert info.value.slots == (slot,)


@pytest.mark.parametrize("slot", ["llm"])
async def test_a_deployment_pin_after_the_flip_is_refused(
    monkeypatch: pytest.MonkeyPatch, slot: str
) -> None:
    _posture(monkeypatch, "byo_required")
    set_identity_context(str(TENANT), None, None)
    _pin(slot, "deployment")
    with pytest.raises(SlotNotConfiguredError):
        await refuse_platform_for_bound_tenant(slot)


async def test_a_no_tenant_background_call_stays_on_the_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posture = _posture(monkeypatch, "byo_required")
    await refuse_platform_for_bound_tenant("llm")  # no raise
    assert posture.calls == [], "no tenant ⇒ platform; licences is not even asked"


@pytest.mark.parametrize("mode", ["local_allowed"])
async def test_owner_postures_pass(monkeypatch: pytest.MonkeyPatch, mode: ByoMode) -> None:
    posture = _posture(monkeypatch, mode)
    set_identity_context(str(TENANT), None, None)
    await refuse_platform_for_bound_tenant("tts")
    assert posture.calls == [TENANT]


async def test_a_tenant_pin_is_not_the_platform_path(monkeypatch: pytest.MonkeyPatch) -> None:
    posture = _posture(monkeypatch, "byo_required")
    set_identity_context(str(TENANT), None, None)
    _pin("tts", "tenant")
    await refuse_platform_for_bound_tenant("tts")
    assert posture.calls == []


# ── wired into each in-process platform client, before anything is sent ─────


async def test_wm_client_platform_llm_is_refused_before_any_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.llm import dispatch, ir_direct, vllm_direct

    _posture(monkeypatch, "byo_required")
    set_identity_context(str(TENANT), None, None)
    ir = AsyncMock()
    vl = AsyncMock()
    monkeypatch.setattr(ir_direct, "call_platform_ir", ir)
    monkeypatch.setattr(vllm_direct, "call_platform_vllm", vl)
    with pytest.raises(SlotNotConfiguredError):
        await dispatch.WMClient().message("storyboard_assist", "hi")
    ir.assert_not_called()
    vl.assert_not_called()


async def test_tts_client_platform_path_is_refused_before_any_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.clients.tts_client import TTSClient

    _posture(monkeypatch, "byo_required")
    set_identity_context(str(TENANT), None, None)
    client = TTSClient(license_token="svc")
    post = AsyncMock()
    monkeypatch.setattr(client, "_post_synthesize_json", post)
    with pytest.raises(SlotNotConfiguredError):
        await client.synthesize(text="hello")
    post.assert_not_called()


async def test_asr_client_platform_path_is_refused_before_any_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from src.clients.transcription_worker_client import TranscriptionWorkerClient

    _posture(monkeypatch, "byo_required")
    set_identity_context(str(TENANT), None, None)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF....WAVE")
    client = TranscriptionWorkerClient(base_url="http://gw.test", license_token="svc")
    sent = AsyncMock()
    monkeypatch.setattr(client, "_get_client", sent)
    with pytest.raises(SlotNotConfiguredError):
        await client.transcribe_file(file_path=audio)
    sent.assert_not_called()


async def test_tts_gateway_batch_is_refused_before_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.clients.tts_gateway_client import TTSGatewayClient

    _posture(monkeypatch, "byo_required")
    set_identity_context(str(TENANT), None, None)
    client = TTSGatewayClient(license_token="svc")
    send = AsyncMock()
    monkeypatch.setattr(client, "_send", send)
    with pytest.raises(SlotNotConfiguredError):
        await client.submit_batch("b1", [])
    send.assert_not_called()


async def test_ocr_with_no_tenant_uses_the_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.clients.ocr_client import resolve_ocr_target
    from src.config import ProviderOcrSettings

    _posture(monkeypatch, "byo_required")
    settings = ProviderOcrSettings.model_validate({"W2V_PROVIDER_OCR_URL": "http://ocr.p:9131"})
    target = await resolve_ocr_target(settings)
    assert target is not None and target.source == "deployment"
