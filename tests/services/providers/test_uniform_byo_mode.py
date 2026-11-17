"""ONE provider-access policy for every model slot (owner decision 2026-10-22).

Licences ``byo_mode`` decides, for EVERY ``byo_governed`` registry slot (llm, vlm,
tts, asr, image, video_gen, ocr), at pin, resolve and preflight:

* ``local_allowed`` — tenant connection if active, else the platform default;
* ``byo_required`` — tenant connection or a refusal naming the slot, never platform;
* ``platform_only`` — the platform default; a saved tenant connection is IGNORED.

Non-governed slots (device_farm, scrape, …) never read the posture.
"""

from __future__ import annotations

import base64
import json
from typing import Any, get_args
from uuid import UUID

import httpx2
import pytest
from pydantic import SecretStr

from src.config import ProviderOcrSettings
from src.services.providers import access
from src.services.providers.byo_posture import StaticByoPosture
from src.services.providers.errors import (
    BYO_MODALITIES,
    ByoModality,
    SlotNotConfiguredError,
    SlotUnavailableError,
)
from src.services.providers.registry import (
    BYO_GOVERNED_SLOTS,
    KIND_NONAGENTIC_ENDPOINT,
    KIND_OPENAI_COMPATIBLE,
    KIND_PLATFORM,
    PINNED_SLOT_NAMES,
    SLOTS,
)
from src.services.providers.resolver import (
    PROVIDER_SLOTS_KEY,
    ResolvedSlot,
    SlotRef,
    pin_slots,
    resolve_slot,
)
from tests.services.providers.test_resolver import DEC, PUBLIC_URL, TENANT, FakeRepo, Rec, policy

GOVERNED = ("llm", "vlm", "tts", "asr", "image", "video_gen", "ocr")

#: A minimal valid tenant connection per governed slot (kind + required fields).
_KIND: dict[str, tuple[str, dict[str, object]]] = {
    "llm": ("openai-compatible", {}),
    "vlm": ("openai-compatible", {}),
    "tts": ("openai", {"voice": "alloy"}),
    "asr": ("openai-compatible", {}),
    "image": ("openai", {}),
    "video_gen": ("runway", {}),
    "ocr": ("openai-compatible", {}),
}


def _rec(slot: str, version: int = 1) -> Rec:
    kind, options = _KIND[slot]
    return Rec(version, slot=slot, kind=kind, base_url=PUBLIC_URL, options=options)


class _AllSlotsRepo(FakeRepo):
    """A tenant with an active connection on EVERY pinned slot."""

    def __init__(self) -> None:
        super().__init__()
        self.asked: list[str] = []

    async def get_active(self, tenant_id: UUID, slot: str) -> Any:
        self.asked.append(slot)
        return Rec(1, slot=slot, kind="x")


# ── the table is the registry ───────────────────────────────────────────────


def test_governed_slots_come_from_the_registry_and_match_the_error_literal() -> None:
    assert BYO_GOVERNED_SLOTS == GOVERNED
    assert set(get_args(ByoModality)) == set(GOVERNED) == BYO_MODALITIES
    assert {n for n, s in SLOTS.items() if s.byo_governed} == set(GOVERNED)
    for slot in ("device_farm", "publish.youtube", "scrape", "media_search", "doc_parse"):
        assert not access.governed(slot)


@pytest.mark.parametrize("slot", ["llm"])
def test_access_table_per_mode(slot: str) -> None:
    assert access.tenant_connection_used("local_allowed", slot)
    assert access.platform_allowed("local_allowed", slot)
    assert not access.tenant_writes_refused("local_allowed", slot)
    assert access.tenant_connection_used("byo_required", slot)
    assert not access.platform_allowed("byo_required", slot)
    assert not access.tenant_writes_refused("byo_required", slot)
    assert not access.tenant_connection_used("platform_only", slot)
    assert access.platform_allowed("platform_only", slot)
    assert access.tenant_writes_refused("platform_only", slot)  # video_gen + ocr too


def test_required_slots() -> None:
    assert access.required_slots("byo_required") == frozenset(GOVERNED)
    assert access.required_slots("local_allowed") == frozenset()
    assert access.required_slots("platform_only") == frozenset()


# ── resolve_slot: mode x slot ───────────────────────────────────────────────


@pytest.mark.parametrize("slot", ["llm"])
async def test_local_allowed_prefers_the_tenant_connection_else_platform(slot: str) -> None:
    posture = StaticByoPosture("local_allowed")
    got = await resolve_slot(
        FakeRepo(active=_rec(slot)), DEC, TENANT, slot, policy=policy(), posture=posture
    )
    assert got.source == "tenant"
    got = await resolve_slot(FakeRepo(), DEC, TENANT, slot, policy=policy(), posture=posture)
    assert got.source == "deployment"


@pytest.mark.parametrize("slot", ["llm"])
async def test_byo_required_uses_the_tenant_connection_and_never_the_platform(slot: str) -> None:
    posture = StaticByoPosture("byo_required")
    got = await resolve_slot(
        FakeRepo(active=_rec(slot)), DEC, TENANT, slot, policy=policy(), posture=posture
    )
    assert got.source == "tenant"
    with pytest.raises(SlotNotConfiguredError) as info:
        await resolve_slot(FakeRepo(), DEC, TENANT, slot, policy=policy(), posture=posture)
    assert info.value.slots == (slot,)
    assert info.value.failure_detail["code"] == "byo_provider_required"
    assert info.value.failure_detail["modality"] == slot


@pytest.mark.parametrize("slot", ["llm"])
async def test_byo_required_refuses_a_pinned_platform_ref_too(slot: str) -> None:
    """Defence in depth: a run pinned to the platform before the flip is refused."""
    ref = SlotRef(
        slot=slot, source="deployment", kind=SLOTS[slot].deployment_kinds.__iter__().__next__()
    )
    snap = {PROVIDER_SLOTS_KEY: {slot: ref.to_dict()}}
    with pytest.raises(SlotNotConfiguredError):
        await resolve_slot(
            FakeRepo(),
            DEC,
            TENANT,
            slot,
            run_snapshot=snap,
            policy=policy(),
            posture=StaticByoPosture("byo_required"),
        )


@pytest.mark.parametrize("slot", ["llm"])
async def test_platform_only_ignores_a_saved_tenant_connection(slot: str) -> None:
    repo = FakeRepo(active=_rec(slot))
    got = await resolve_slot(
        repo, DEC, TENANT, slot, policy=policy(), posture=StaticByoPosture("platform_only")
    )
    assert got.source == "deployment"
    assert repo.active_calls == [], "the saved connection is not even loaded"


@pytest.mark.parametrize("slot", ["llm"])
async def test_platform_only_refuses_a_run_pinned_to_a_tenant_connection(slot: str) -> None:
    """A run pinned to the tenant's own provider BEFORE the operator flip is refused
    (permanent, tenant-facing), never silently re-routed mid-run."""
    rec = _rec(slot)
    ref = SlotRef(
        slot=slot,
        source="tenant",
        kind=rec.kind,
        connection_id=rec.connection_id,
        version=rec.version,
    )
    with pytest.raises(SlotUnavailableError, match="platform providers only"):
        await resolve_slot(
            FakeRepo(versions=[rec]),
            DEC,
            TENANT,
            slot,
            run_snapshot={PROVIDER_SLOTS_KEY: {slot: ref.to_dict()}},
            policy=policy(),
            posture=StaticByoPosture("platform_only"),
        )


# ── pin_slots ───────────────────────────────────────────────────────────────


async def test_pin_slots_under_platform_only_pins_every_governed_slot_to_the_platform() -> None:
    posture = StaticByoPosture("platform_only")
    pins = await pin_slots(_AllSlotsRepo(), TENANT, policy=policy(), posture=posture)
    for slot in PINNED_SLOT_NAMES:
        want = "deployment" if slot in GOVERNED else "tenant"
        assert pins[slot].source == want, slot
    assert posture.calls == [TENANT], "the posture is read ONCE per pin"


@pytest.mark.parametrize("mode", ["local_allowed"])
async def test_pin_slots_keeps_tenant_connections_otherwise(mode: str) -> None:
    posture = StaticByoPosture(mode)  # ty: ignore[invalid-argument-type]  # why: parametrized literal
    pins = await pin_slots(_AllSlotsRepo(), TENANT, policy=policy(), posture=posture)
    assert all(pins[s].source == "tenant" for s in PINNED_SLOT_NAMES)


async def test_pin_slots_without_connections_never_reads_the_posture() -> None:
    posture = StaticByoPosture("byo_required")
    pins = await pin_slots(FakeRepo(), TENANT, policy=policy(), posture=posture)
    assert all(p.source == "deployment" for p in pins.values())
    assert posture.calls == []


# ── the ocr slot ────────────────────────────────────────────────────────────


def test_ocr_slot_shape() -> None:
    spec = SLOTS["ocr"]
    assert spec.deployment_kinds == {KIND_PLATFORM}
    assert spec.tenant_kinds == {KIND_NONAGENTIC_ENDPOINT, KIND_OPENAI_COMPATIBLE}
    assert spec.writable and spec.pinned and spec.byo_governed
    ne = spec.tenant_kind_specs[KIND_NONAGENTIC_ENDPOINT]
    assert ne.base_url == "required" and not ne.secret_required and not ne.model_required
    assert "POST /v1/ocr" in ne.help
    oc = spec.tenant_kind_specs[KIND_OPENAI_COMPATIBLE]
    assert oc.base_url == "required" and oc.model_required
    assert "{base_url}/ocr" in oc.help
    assert policy().deployment_kind("ocr") == KIND_PLATFORM


def _settings() -> ProviderOcrSettings:
    return ProviderOcrSettings.model_validate(
        {
            "W2V_PROVIDER_OCR_URL": "http://ocr.platform:9131",
            "W2V_PROVIDER_OCR_API_KEY": "platform-key",  # pragma: allowlist secret
            "W2V_PROVIDER_OCR_MODEL": "PaddleOCR-VL",
        }
    )


def _resolved(kind: str, base: str, model: str | None) -> ResolvedSlot:
    return ResolvedSlot(
        ref=SlotRef(
            slot="ocr",
            source="tenant",
            kind=kind,
            connection_id=UUID(int=9),
            version=3,
        ),
        base_url=base,
        model=model,
        secret=SecretStr("tenant-key"),  # pragma: allowlist secret
        pinned_ip="1.1.1.1",
    )


def test_tenant_target_routes_per_kind() -> None:
    from src.clients.ocr_client import tenant_target

    ne = tenant_target(
        _resolved(KIND_NONAGENTIC_ENDPOINT, "https://ocr.t.example/", None), _settings()
    )
    assert ne.endpoint == "https://ocr.t.example/v1/ocr"
    assert ne.model == "PaddleOCR-VL" and ne.api_key == "tenant-key"  # pragma: allowlist secret
    oc = tenant_target(
        _resolved(KIND_OPENAI_COMPATIBLE, "https://api.mistral.example/v1", "mistral-ocr-latest"),
        _settings(),
    )
    assert oc.endpoint == "https://api.mistral.example/v1/ocr"
    assert oc.model == "mistral-ocr-latest" and oc.pinned_ip == "1.1.1.1"


async def test_ocr_goes_through_the_resolved_tenant_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OcrClient.for_bound_run`` → resolver → the TENANT endpoint with the TENANT key,
    over the pinned transport — never the platform URL or the platform key."""
    from src.clients import ocr_client
    from src.services.llm import in_process

    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={"pages": [{"index": 0, "markdown": "hi"}]})

    monkeypatch.setattr(
        in_process, "inner_transport_factory", lambda: httpx2.MockTransport(handler)
    )

    async def _resolve(settings: ProviderOcrSettings) -> ocr_client.OcrTarget | None:
        return ocr_client.tenant_target(
            _resolved(KIND_NONAGENTIC_ENDPOINT, "https://ocr.t.example", None), settings
        )

    client = ocr_client.OcrClient.for_bound_run(_settings(), resolve_target=_resolve)
    pages = await client.ocr_pdf(b"%PDF-1.4 x")
    assert [(p.index, p.markdown) for p in pages] == [(0, "hi")]
    [req] = seen
    # Connected to the resolver's vetted IP (DNS-rebinding pin), Host kept.
    assert str(req.url) == "https://1.1.1.1/v1/ocr"
    assert req.headers["host"] == "ocr.t.example"
    assert req.headers["authorization"] == "Bearer tenant-key"
    assert "platform-key" not in str(req.headers)
    body = json.loads(req.content)
    assert body["model"] == "PaddleOCR-VL"
    assert body["document"]["document_url"].endswith(base64.b64encode(b"%PDF-1.4 x").decode())


async def test_resolve_ocr_target_applies_the_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production resolver path: run pin → ``resolve_slot`` (byo_mode applied)."""
    from contextlib import asynccontextmanager

    from w2v_common.auth.identity_context import set_identity_context

    from src.clients import ocr_client
    from src.services.providers import adapters, resolver

    calls: list[dict[str, Any]] = []

    async def _fake_resolve(repo, dec, tenant_id, slot, **kw):
        calls.append({"tenant": tenant_id, "slot": slot, **kw})
        raise SlotNotConfiguredError(slot)

    @asynccontextmanager
    async def _session():
        yield object()

    async def _bind(_s: object, _t: object) -> None:
        return None

    class _Deps:
        repo = decryptor = policy = posture = None

    monkeypatch.setattr(resolver, "resolve_slot", _fake_resolve)
    monkeypatch.setattr("src.database.session.get_db_session", _session)
    monkeypatch.setattr("src.database.session.bind_session_tenant", _bind)
    monkeypatch.setattr(adapters, "build_resolver_deps", lambda *_a: _Deps())
    set_identity_context(str(TENANT), None, None)
    with pytest.raises(SlotNotConfiguredError) as info:
        await ocr_client.resolve_ocr_target(_settings())
    assert info.value.slots == ("ocr",)
    assert calls and calls[0]["slot"] == "ocr" and calls[0]["tenant"] == TENANT


async def test_a_text_layer_pdf_never_resolves_the_ocr_slot() -> None:
    """Only a page that NEEDS OCR consults the slot (and so the tenant's posture)."""
    from io import BytesIO

    from pypdf import PdfWriter

    from src.clients.ocr_client import OcrClient
    from src.services.documents.extract import extract_document

    asked: list[object] = []

    async def _resolve(_s: ProviderOcrSettings) -> None:
        asked.append(1)
        raise SlotNotConfiguredError("ocr")

    # A blank (text-less) page DOES ask — and the refusal propagates, naming ocr.
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)
    with pytest.raises(SlotNotConfiguredError):
        await extract_document(
            buf.getvalue(),
            "scan.pdf",
            ocr=OcrClient.for_bound_run(_settings(), resolve_target=_resolve),
        )
    assert asked == [1]


# ── loop guard: a tenant connection may not point at w2v itself ─────────────


@pytest.mark.parametrize(
    "url",
    ["https://studio.nonagentic.ai/api"],
)
def test_w2v_own_hosts_are_refused(url: str) -> None:
    from src.services.providers.errors import SlotEgressRefusedError
    from src.services.providers.policy import check_not_self, points_at_w2v_itself

    assert points_at_w2v_itself(url)
    with pytest.raises(SlotEgressRefusedError, match="request loop"):
        check_not_self("llm", url)


@pytest.mark.parametrize("url", ["https://api.nonagentic.ai/tts"])
def test_the_gateway_and_public_hosts_stay_allowed(url: str) -> None:
    from src.services.providers.policy import check_not_self, points_at_w2v_itself

    assert not points_at_w2v_itself(url)
    check_not_self("tts", url)


async def test_a_stored_connection_pointing_at_w2v_is_refused_at_resolve() -> None:
    from src.services.providers.errors import SlotEgressRefusedError

    rec = Rec(1, base_url="https://studio.nonagentic.ai/v1")
    with pytest.raises(SlotEgressRefusedError):
        await resolve_slot(
            FakeRepo(active=rec),
            DEC,
            TENANT,
            "llm",
            policy=policy(),
            posture=StaticByoPosture("local_allowed"),
        )
