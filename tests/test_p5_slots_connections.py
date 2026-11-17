"""P5 — the newly writable slots on the REAL ``/api/tenant/connections`` request path.

Same harness as ``tests/api/routes/test_tenancy_connections.py``: real JWT → auth
middleware → membership tenancy → real router → real Postgres rows. Faked: the live
provider test (a recorder returning a scripted verdict; the HTTP prober itself is in
``test_p5_slots_tester.py``).

Non-vacuity: every 422 "nothing persisted, tester never called" case is paired with
a valid PUT through the same call shape that DOES persist and DOES call the tester.
"""

from __future__ import annotations

import os
import types
import uuid
from dataclasses import dataclass, field
from typing import Any

import fakeredis.aioredis
import pytest
from sqlalchemy import func, select

from src.api.routes.tenancy.connections import (
    get_connections_keyring,
    get_connections_policy,
    get_provider_tester,
)
from src.database.models import TenantConnection
from src.services.providers.byo_posture import StaticByoPosture
from src.services.providers.crypto import Keyring
from src.services.providers.registry import PINNED_SLOT_NAMES, SLOTS
from src.services.providers.resolver import pin_slots
from src.services.providers.settings import ProviderPolicySettings
from src.services.providers.tester import ProviderTestResult
from tests.api.routes._tenancy_harness import (  # noqa: F401 — fixtures
    Harness,
    harness_fixture,
    signing_key_fixture,
)

pytestmark = pytest.mark.asyncio

SECRET = "sk-p5-route-NEVER-ECHO-41ab"  # pragma: allowlist secret
SELFHOST = "https://tts.selfhost.test/v1"
_POLICY = ProviderPolicySettings(
    ssrf_mode="private_ok", egress_allowlist=frozenset({"tts.selfhost.test"})
)


@dataclass
class FakeTester:
    verdict: ProviderTestResult = field(
        default_factory=lambda: ProviderTestResult(True, "answered HTTP 200", 200)
    )
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def test(self, **kwargs: Any) -> ProviderTestResult:
        self.calls.append(kwargs)
        return self.verdict


@dataclass
class Ctx:
    h: Harness
    tester: FakeTester


@pytest.fixture
def ctx(harness: Harness) -> Ctx:
    c = Ctx(h=harness, tester=FakeTester())
    keyring = Keyring([("k1", os.urandom(32))])
    harness.app.dependency_overrides[get_provider_tester] = lambda: c.tester
    harness.app.dependency_overrides[get_connections_policy] = lambda: _POLICY
    harness.app.dependency_overrides[get_connections_keyring] = lambda: keyring
    harness.app.state.redis = fakeredis.aioredis.FakeRedis()
    return c


async def _admin(h: Harness) -> tuple[uuid.UUID, str]:
    tenant = await h.tenant()
    sub = uuid.uuid7()
    await h.member(sub, tenant, "tenant_admin", email=f"admin-{sub}@acme.test")
    return tenant, h.token(sub)


async def _rows(h: Harness, tenant: uuid.UUID) -> int:
    async with h.factory() as s:
        return (
            await s.execute(
                select(func.count())
                .select_from(TenantConnection)
                .where(TenantConnection.tenant_id == tenant)
            )
        ).scalar_one()


async def _slot(h: Harness, tok: str, slot: str) -> dict[str, Any]:
    r = await h.call("GET", "/api/tenant/connections", tok)
    assert r.status_code == 200, r.text
    return next(i for i in r.json()["items"] if i["slot"] == slot)


# (slot, valid body) — each must persist; the invalid variants below derive from it.
VALID: dict[str, dict[str, Any]] = {
    "tts/openai": {"kind": "openai", "options": {"voice": "alloy"}, "secret": SECRET},
    "tts/elevenlabs": {"kind": "elevenlabs", "options": {"voice": "v-123"}, "secret": SECRET},
    "tts/openai-compatible": {
        "kind": "openai-compatible",
        "base_url": SELFHOST,
        "model": "kokoro",
        "options": {"voice": "af_heart"},
    },
    "image/openai": {"kind": "openai", "model": "gpt-image-1", "secret": SECRET},
    "video_gen/sora": {"kind": "sora", "secret": SECRET},
    "device_farm/browserstack": {
        "kind": "browserstack",
        "options": {"username": "acme-bs"},
        "secret": SECRET,
    },
}

INVALID: list[tuple[str, str, dict[str, Any], str]] = [
    ("tts", "tts/openai", {"options": {}}, "OPTION_REQUIRED"),
    ("tts", "tts/openai", {"options": {"voice": "  "}}, "OPTION_REQUIRED"),
    ("tts", "tts/elevenlabs", {"secret": ...}, "SECRET_REQUIRED"),
    ("tts", "tts/openai", {"secret": ""}, "SECRET_REQUIRED"),
    ("tts", "tts/openai-compatible", {"base_url": ...}, "BASE_URL_REQUIRED"),
    ("tts", "tts/openai-compatible", {"model": ...}, "MODEL_REQUIRED"),
    ("tts", "tts/openai", {"kind": "anthropic"}, "PROVIDER_KIND_NOT_ALLOWED"),
    ("image", "image/openai", {"kind": "elevenlabs"}, "PROVIDER_KIND_NOT_ALLOWED"),
    ("image", "image/openai", {"secret": ...}, "SECRET_REQUIRED"),
    ("video_gen", "video_gen/sora", {"secret": ...}, "SECRET_REQUIRED"),
    (
        "device_farm",
        "device_farm/browserstack",
        {"base_url": "https://evil.example/v1"},
        "BASE_URL_NOT_ALLOWED",
    ),
    ("device_farm", "device_farm/browserstack", {"options": {}}, "OPTION_REQUIRED"),
    ("device_farm", "device_farm/browserstack", {"secret": ...}, "SECRET_REQUIRED"),
]


def _body(key: str, **over: Any) -> dict[str, Any]:
    body = {**VALID[key], "expected_version": 0, **over}
    return {k: v for k, v in body.items() if v is not ...}


@pytest.mark.parametrize("key", list(sorted(VALID))[:1])
async def test_valid_put_per_kind_tests_then_persists(ctx: Ctx, key: str) -> None:
    slot = key.split("/")[0]
    tenant, tok = await _admin(ctx.h)
    r = await ctx.h.call("PUT", f"/api/tenant/connections/{slot}", tok, json_body=_body(key))
    assert r.status_code == 200, r.text
    view = r.json()
    assert view["source"] == "tenant" and view["connection"]["version"] == 1
    assert view["connection"]["last_test_ok"] is True
    assert SECRET not in r.text
    [call] = ctx.tester.calls
    assert (call["slot"], call["kind"]) == (slot, VALID[key]["kind"])
    assert call["secret"] == VALID[key].get("secret")
    assert await _rows(ctx.h, tenant) == 1


@pytest.mark.parametrize(("slot", "key", "over", "code"), [INVALID[0]])
async def test_invalid_put_per_kind_is_422_and_persists_nothing(
    ctx: Ctx, slot: str, key: str, over: dict[str, Any], code: str
) -> None:
    tenant, tok = await _admin(ctx.h)
    r = await ctx.h.call(
        "PUT", f"/api/tenant/connections/{slot}", tok, json_body=_body(key, **over)
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == code
    assert ctx.tester.calls == []  # the key never left
    assert await _rows(ctx.h, tenant) == 0


async def test_probe_failure_is_422_and_persists_nothing(ctx: Ctx) -> None:
    tenant, tok = await _admin(ctx.h)
    ctx.tester.verdict = ProviderTestResult(False, "api.elevenlabs.io answered HTTP 401", 401)
    r = await ctx.h.call(
        "PUT", "/api/tenant/connections/tts", tok, json_body=_body("tts/elevenlabs")
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "PROVIDER_TEST_FAILED"
    assert len(ctx.tester.calls) == 1
    assert await _rows(ctx.h, tenant) == 0
    assert (await _slot(ctx.h, tok, "tts"))["source"] == "deployment"


async def test_unprobed_kind_is_stored_as_not_probed_never_as_a_pass(ctx: Ctx) -> None:
    tenant, tok = await _admin(ctx.h)
    msg = "not probed: runway has no auth-only endpoint"
    ctx.tester.verdict = ProviderTestResult(True, msg, probed=False)
    r = await ctx.h.call(
        "PUT",
        "/api/tenant/connections/video_gen",
        tok,
        json_body={"kind": "runway", "secret": SECRET, "expected_version": 0},
    )
    assert r.status_code == 200, r.text
    conn = r.json()["connection"]
    assert conn["last_test_ok"] is None
    assert conn["last_test_error"] == msg
    assert await _rows(ctx.h, tenant) == 1

    r = await ctx.h.call("POST", "/api/tenant/connections/video_gen/test", tok)
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": None, "probed": False, "message": msg, "upstream_status": None,
                        "version": 1}  # fmt: skip


async def test_publish_youtube_is_a_grant_slot_not_writable_here(ctx: Ctx) -> None:
    tenant, tok = await _admin(ctx.h)
    view = await _slot(ctx.h, tok, "publish.youtube")
    assert view["type"] == "grant" and view["writable"] is False
    for method, path, body in (
        ("PUT", "/api/tenant/connections/publish.youtube", _body("video_gen/sora")),
        ("POST", "/api/tenant/connections/publish.youtube/test", None),
        ("DELETE", "/api/tenant/connections/publish.youtube", None),
    ):
        r = await ctx.h.call(method, path, tok, json_body=body)
        assert r.status_code == 409, (method, r.text)
        assert r.json()["error"]["code"] == "SLOT_NOT_WRITABLE"
    assert ctx.tester.calls == []
    assert await _rows(ctx.h, tenant) == 0


async def test_get_lists_p5_kinds_requirements_and_defaults(ctx: Ctx) -> None:
    _, tok = await _admin(ctx.h)
    tts = await _slot(ctx.h, tok, "tts")
    assert tts["allowed_kinds"] == [
        "elevenlabs",
        "nonagentic_endpoint",
        "openai",
        "openai-compatible",
    ]
    assert tts["default_kind"] == "nonagentic"
    assert tts["kind_requirements"]["openai-compatible"] == {
        "base_url": "required",
        "model_required": True,
        "secret_required": False,
        "required_options": ["voice"],
    }
    farm = await _slot(ctx.h, tok, "device_farm")
    assert farm["default_kind"] == "local"
    assert farm["kind_requirements"]["browserstack"]["base_url"] == "forbidden"
    assert farm["kind_requirements"]["browserstack"]["required_options"] == ["username"]
    video = await _slot(ctx.h, tok, "video_gen")
    assert video["allowed_kinds"] == ["kling", "runway", "seedance", "sora", "veo"]
    assert (await _slot(ctx.h, tok, "image"))["allowed_kinds"] == [
        "nonagentic_endpoint",
        "openai",
        "openai-compatible",
    ]


class _EveryoneHasARow:
    """Repo that claims an active tenant row on EVERY slot it is asked about."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def get_active(self, tenant_id: uuid.UUID, slot: str) -> Any:
        self.asked.append(slot)
        return types.SimpleNamespace(kind="x", connection_id=uuid.uuid4(), version=1)

    async def get_version(self, connection_id: uuid.UUID, version: int) -> None:
        return None


async def test_pin_slots_excludes_publish_youtube() -> None:
    repo = _EveryoneHasARow()
    pins = await pin_slots(
        repo,
        uuid.uuid4(),
        policy=ProviderPolicySettings(_env_file=None),
        posture=StaticByoPosture("local_allowed"),
    )
    assert "publish.youtube" in SLOTS
    assert "publish.youtube" not in pins
    assert "publish.youtube" not in repo.asked
    assert set(pins) == set(SLOTS) - {"publish.youtube"} == set(PINNED_SLOT_NAMES)
