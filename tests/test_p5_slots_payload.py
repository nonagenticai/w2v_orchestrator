"""P5-2 — the run-pinned ``provider_slot`` reaches ACG and app_inspector, and only then.

Real pieces: the run-slot ContextVar (``bind_run_provider_slots``), the identity
ContextVar, :func:`build_provider_slot`, the REAL ``AIContentGenerationClient`` /
``AppInspectorClient`` request path down to the httpx wire (an ``httpx.MockTransport``
captures the exact request bytes). Faked: only ``resolve_pinned_slot`` (the DB +
keyring read, covered by the resolver/route suites) — it records the ref it was asked
for so a test that "passes" without resolving the pinned version cannot hide.

Non-vacuity: each "absent" case is paired with the tenant case through the same
call, and the byte-identity check compares against the pre-P5 encoding of the body.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import SecretStr
from w2v_common.auth.identity_context import set_identity_context

from src.auth.run_providers import bind_run_provider_slots, clear_run_provider_slots
from src.clients.ai_content_generation_client import AIContentGenerationClient
from src.clients.app_inspector_client import AppInspectorClient
from src.clients.schemas.app_inspector import InspectionResult
from src.services.providers import slot_payload
from src.services.providers.errors import SlotUnavailableError
from src.services.providers.resolver import PROVIDER_SLOTS_KEY, ResolvedSlot, SlotRef
from src.services.providers.slot_payload import build_provider_slot, redact_provider_slot
from src.services.runs.run_context import RunContext
from src.services.stages.app_review.inspect import AppInspectStage

pytestmark = pytest.mark.asyncio

TENANT = uuid.UUID("0199aaaa-0000-7000-8000-000000000001")
CONN = uuid.UUID("0199bbbb-0000-7000-8000-000000000002")
SECRET = "sk-tenant-NEVER-LOG-3b9e"  # pragma: allowlist secret
BS_KEY = "bs-access-NEVER-LOG-77aa"  # pragma: allowlist secret


class FakeResolve:
    """Stands in for ``resolve_pinned_slot``; returns per-slot ``ResolvedSlot``s."""

    def __init__(self, by_slot: dict[str, ResolvedSlot]) -> None:
        self.by_slot = by_slot
        self.calls: list[tuple[uuid.UUID, str, SlotRef]] = []

    async def __call__(self, tenant_id: uuid.UUID, slot: str, ref: SlotRef) -> ResolvedSlot:
        self.calls.append((tenant_id, slot, ref))
        return self.by_slot[slot]


def _tenant_ref(slot: str, kind: str, version: int = 3) -> SlotRef:
    return SlotRef(slot=slot, source="tenant", kind=kind, connection_id=CONN, version=version)


def _resolved(slot: str, kind: str, **kw: Any) -> ResolvedSlot:
    return ResolvedSlot(ref=_tenant_ref(slot, kind), **kw)


def _bind(pins: dict[str, SlotRef] | None) -> None:
    """Bind a run's pins (``None`` = a legacy run that pinned nothing)."""
    if pins is None:
        bind_run_provider_slots({})
    else:
        bind_run_provider_slots({PROVIDER_SLOTS_KEY: {s: r.to_dict() for s, r in pins.items()}})
    set_identity_context(TENANT, None, None)


@pytest.fixture(autouse=True)
def _unbind() -> Iterator[None]:
    yield
    clear_run_provider_slots()


@pytest.fixture
def fake_resolve(monkeypatch: pytest.MonkeyPatch) -> FakeResolve:
    fake = FakeResolve(
        {
            "image": _resolved(
                "image",
                "openai",
                base_url=None,
                model="gpt-image-1",
                options={"quality": "high"},
                secret=SecretStr(SECRET),
            ),
            "video_gen": _resolved(
                "video_gen",
                "runway",
                base_url="https://api.runway.example/v1",
                model="gen4",
                options={},
                secret=SecretStr(SECRET),
            ),
            "device_farm": _resolved(
                "device_farm",
                "browserstack",
                options={"username": "acme-bs"},
                secret=SecretStr(BS_KEY),
            ),
        }
    )
    monkeypatch.setattr(slot_payload, "resolve_pinned_slot", fake)
    return fake


class Wire:
    """Captures every request the client puts on the wire."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(202, json={"job_id": "job-1"})


def _acg(wire: Wire) -> AIContentGenerationClient:
    client = AIContentGenerationClient(base_url="http://acg.test", license_token="lic")
    client._client = httpx.AsyncClient(
        base_url="http://acg.test", transport=httpx.MockTransport(wire)
    )
    return client


def _pre_p5_bytes(payload: dict[str, Any]) -> bytes:
    """How the client encoded this body before P5 (httpx ``json=``)."""
    return httpx.Request("POST", "http://acg.test/x", json=payload).content


# ─── build_provider_slot ─────────────────────────────────────────────────


async def test_build_provider_slot_returns_the_contract_only_for_a_tenant_pin(
    fake_resolve: FakeResolve,
) -> None:
    _bind({"image": _tenant_ref("image", "openai")})
    got = await build_provider_slot("image")
    assert got == {
        "slot": "image",
        "kind": "openai",
        "base_url": None,
        "model": "gpt-image-1",
        "options": {"quality": "high"},
        "secret": SECRET,
        "ref": {"connection_id": str(CONN), "version": 3},
    }
    [(tenant, slot, ref)] = fake_resolve.calls
    assert (tenant, slot, ref) == (TENANT, "image", _tenant_ref("image", "openai"))
    redacted = redact_provider_slot(got)
    assert redacted is not None and redacted["secret"] == "[REDACTED]"
    assert got["secret"] == SECRET  # the redaction copied, it did not mutate


@pytest.mark.parametrize(
    "pins",
    [None],
)
async def test_build_provider_slot_is_none_without_a_tenant_pin_for_that_slot(
    fake_resolve: FakeResolve, pins: dict[str, SlotRef] | None
) -> None:
    _bind(pins)
    assert await build_provider_slot("image") is None
    assert fake_resolve.calls == []


async def test_build_provider_slot_unbound_context_is_none(fake_resolve: FakeResolve) -> None:
    clear_run_provider_slots()
    assert await build_provider_slot("image") is None
    assert fake_resolve.calls == []


async def test_a_broken_tenant_pin_raises_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(*_a: object) -> ResolvedSlot:
        raise SlotUnavailableError("image", "the pinned connection version has been purged")

    monkeypatch.setattr(slot_payload, "resolve_pinned_slot", boom)
    _bind({"image": _tenant_ref("image", "openai")})
    with pytest.raises(SlotUnavailableError, match="purged"):
        await build_provider_slot("image")


# ─── ACG client ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("method", "slot", "kind", "path"),
    [("create_concept_image", "image", "openai", "/api/concept-images")],
)
async def test_acg_body_carries_provider_slot_only_for_a_tenant_pin(
    fake_resolve: FakeResolve, method: str, slot: str, kind: str, path: str
) -> None:
    config = {"aspect_ratio": "16:9", "provider": "stub"}
    expected_plain = {"prompt": "a red fox", **config}

    # 1) No pin at all → byte-identical to the pre-P5 body.
    wire = Wire()
    _bind(None)
    await getattr(_acg(wire), method)("a red fox", dict(config), idempotency_key="k1")
    # 2) Deployment pin for the slot → still byte-identical.
    _bind({slot: SlotRef(slot=slot, source="deployment", kind="nonagentic")})
    await getattr(_acg(wire), method)("a red fox", dict(config), idempotency_key="k1")
    # 3) Tenant pin → the contract object rides along; nothing else changes.
    _bind({slot: _tenant_ref(slot, kind)})
    await getattr(_acg(wire), method)("a red fox", dict(config), idempotency_key="k1")

    plain_a, plain_b, tenant = wire.requests
    assert plain_a.url.path == plain_b.url.path == tenant.url.path == path
    assert plain_a.content == plain_b.content == _pre_p5_bytes(expected_plain)
    assert "provider_slot" not in json.loads(plain_a.content)
    body = json.loads(tenant.content)
    assert body.pop("provider_slot") == {
        "slot": slot,
        "kind": kind,
        "base_url": fake_resolve.by_slot[slot].base_url,
        "model": fake_resolve.by_slot[slot].model,
        "options": dict(fake_resolve.by_slot[slot].options),
        "secret": SECRET,
        "ref": {"connection_id": str(CONN), "version": 3},
    }
    assert body == expected_plain
    assert tenant.headers["idempotency-key"] == "k1"
    assert [c[1] for c in fake_resolve.calls] == [slot]


async def test_acg_strips_a_forged_provider_slot_from_stage_config(
    fake_resolve: FakeResolve,
) -> None:
    wire = Wire()
    _bind(None)
    forged = {"secret": "forged"}  # pragma: allowlist secret -- forged fixture
    await _acg(wire).create_concept_image("p", {"provider_slot": forged})
    assert json.loads(wire.requests[0].content) == {"prompt": "p"}


async def test_acg_upstream_error_does_not_log_the_secret(
    fake_resolve: FakeResolve, caplog: pytest.LogCaptureFixture
) -> None:
    """A FastAPI 422 echoes the request under ``input`` — the log line must not carry it."""

    def echo(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            422,
            json={"detail": [{"loc": ["body", "provider_slot"], "input": body["provider_slot"]}]},
        )

    client = AIContentGenerationClient(base_url="http://acg.test", license_token="lic")
    client._client = httpx.AsyncClient(
        base_url="http://acg.test", transport=httpx.MockTransport(echo)
    )
    _bind({"image": _tenant_ref("image", "openai")})
    caplog.set_level(logging.DEBUG)
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.create_concept_image("p")
    assert SECRET not in caplog.text
    assert SECRET not in str(exc_info.value)
    assert "provider_slot" in caplog.text  # the error WAS logged — just redacted


# ─── app_inspect stage → AppInspectorClient device_farm ──────────────────


def _ctx(competitors: list[str] | None = None) -> RunContext:
    return RunContext(
        run_id=uuid.uuid4(),
        deliverable_id=uuid.uuid4(),
        config_snapshot={
            "brief": {"target_url": "http://target", "competitor_urls": competitors or []}
        },
    )


def _inspection() -> InspectionResult:
    return InspectionResult(estimated_purpose="x", screens=[], screens_discovered=0)


async def test_inspect_stage_sends_no_device_farm_without_a_tenant_pin(
    fake_resolve: FakeResolve,
) -> None:
    fake = AsyncMock()
    fake.inspect_app.return_value = _inspection()
    _bind({"device_farm": SlotRef(slot="device_farm", source="deployment", kind="local")})
    await AppInspectStage(client=fake).run({}, ctx=_ctx(["http://rival"]))
    # Exactly the pre-P5 call shape: url only, no config.
    assert [c.args for c in fake.inspect_app.await_args_list] == [
        ("http://target",),
        ("http://rival",),
    ]
    assert all(not c.kwargs for c in fake.inspect_app.await_args_list)
    assert fake_resolve.calls == []


async def test_inspect_stage_maps_a_tenant_browserstack_slot_to_device_farm(
    fake_resolve: FakeResolve,
) -> None:
    fake = AsyncMock()
    fake.inspect_app.return_value = _inspection()
    _bind({"device_farm": _tenant_ref("device_farm", "browserstack")})
    await AppInspectStage(client=fake).run({}, ctx=_ctx(["http://rival"]))
    farm = {
        "device_farm": {
            "provider": "browserstack",
            "browserstack_username": "acme-bs",
            "browserstack_access_key": BS_KEY,
        }
    }
    # Target AND competitors run on the tenant's farm; resolved once per stage.
    assert [c.args for c in fake.inspect_app.await_args_list] == [
        ("http://target", farm),
        ("http://rival", farm),
    ]
    assert len(fake_resolve.calls) == 1


async def test_app_inspector_client_puts_device_farm_on_the_wire_without_logging_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    wire = Wire()
    client = AppInspectorClient(base_url="http://ai.test", license_token="lic")
    client._client = httpx.AsyncClient(
        base_url="http://ai.test", transport=httpx.MockTransport(wire)
    )
    farm = {
        "provider": "browserstack",
        "browserstack_username": "acme-bs",
        "browserstack_access_key": BS_KEY,
    }
    caplog.set_level(logging.DEBUG)
    await client.start_inspection("http://target", {"device_farm": farm})
    assert json.loads(wire.requests[0].content) == {
        "app_path": "http://target",
        "device_farm": farm,
    }
    assert BS_KEY not in caplog.text
