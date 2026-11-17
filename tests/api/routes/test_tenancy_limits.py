"""P6 ORCH-3 — GET /api/tenant/limits on the REAL request path.

Real RS256 JWT → ``TenancyMiddleware`` (membership mode) → the real router. Only the
JWKS socket and the licences service (an ``httpx.MockTransport``) are faked. The
view is read-only and must say truthfully what is enforced: the licence quota is not,
the provider policy is.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from src.api.routes._common import get_licenses_client
from src.clients.licenses_client import LicensesClient
from src.services.providers.settings import get_provider_policy
from tests.api.routes._tenancy_harness import (  # noqa: F401 — fixtures
    Harness,
    harness_fixture,
    signing_key_fixture,
)

pytestmark = pytest.mark.asyncio

PATH = "/api/tenant/limits"
APP_ID, PLAN_ID = "app-w2v", "plan-pro"


class _Auth:
    async def token(self) -> str:
        return "sa-token"


def _licences(handler: Any) -> LicensesClient:
    return LicensesClient(
        authenticator=_Auth(),
        base_url="http://licenses.test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _posture(mode: str):
    def _answer(request: httpx.Request) -> httpx.Response | None:
        if request.url.path.endswith("/byo-mode"):
            assert request.url.params.get("app") == "w2v"
            tenant = request.url.path.split("/")[2]
            return httpx.Response(
                200, json={"tenant_id": tenant, "byo_mode": mode, "app": "w2v", "resolution": "row"}
            )
        return None

    return _answer


def _healthy(request: httpx.Request, *, byo_mode: str = "local_allowed") -> httpx.Response:
    posture = _posture(byo_mode)(request)
    if posture is not None:
        return posture
    if request.url.path == "/licenses/active":
        return httpx.Response(
            200,
            json={
                "active": True,
                "license_id": "lic-1",
                "app": {"id": APP_ID, "key": "w2v"},
                "plan": {"id": PLAN_ID, "code": "pro"},
            },
        )
    if request.url.path == f"/apps/{APP_ID}/plans":
        return httpx.Response(
            200,
            json=[
                {
                    "id": "plan-free",
                    "app_id": APP_ID,
                    "code": "free",
                    "name": "Free",
                    "quota_monthly": 10,
                },
                {
                    "id": PLAN_ID,
                    "app_id": APP_ID,
                    "code": "pro",
                    "name": "Pro",
                    "quota_monthly": 5000,
                    "token_budget_monthly": 2_000_000,
                },
            ],
        )
    return httpx.Response(404)


@pytest.fixture(autouse=True)
def _fresh_policy() -> Iterator[None]:
    get_provider_policy.cache_clear()
    yield
    get_provider_policy.cache_clear()


async def _admin(h: Harness, role: str = "tenant_admin") -> str:
    tenant = await h.tenant()
    sub = uuid.uuid7()
    await h.member(sub, tenant, role, email=f"{role}-{sub}@acme.test")
    return h.token(sub)


def _by_key(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["key"]: item for item in body["limits"]}


async def test_tenant_admin_sees_limits_with_truthful_enforcement(harness: Harness) -> None:
    harness.app.dependency_overrides[get_licenses_client] = lambda: _licences(_healthy)
    r = await harness.call("GET", PATH, await _admin(harness))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["licence_lookup"] == "ok" and body["plan_code"] == "pro"
    assert body["editable"] is False
    limits = _by_key(body)

    quota = limits["licence.quota_monthly"]
    assert quota["value"] == 5000  # the ACTIVE plan's, not the first plan listed
    assert quota["enforced"] is False and quota["enforced_at"] is None
    assert "Not enforced" in quota["note"]
    assert limits["licence.token_budget_monthly"]["value"] == 2_000_000
    assert limits["licence.token_budget_monthly"]["enforced"] is False

    byo = limits["provider.byo_required"]
    assert byo["value"] == [] and byo["enforced"] is True
    kinds = limits["provider.allowed_kinds"]
    assert kinds["enforced"] is True and {"openai", "elevenlabs"} <= set(kinds["value"])
    rate = limits["provider.test_rate_limit"]
    assert rate["value"] == 10 and rate["enforced"] is True


async def test_policy_env_is_reflected(harness: Harness, monkeypatch) -> None:
    monkeypatch.setenv("PROVIDER_ALLOWED_KINDS", "openai")
    monkeypatch.setenv("PROVIDER_TEST_RATE_LIMIT", "0")
    get_provider_policy.cache_clear()
    harness.app.dependency_overrides[get_licenses_client] = lambda: _licences(
        lambda req: _healthy(req, byo_mode="byo_required")
    )
    r = await harness.call("GET", PATH, await _admin(harness))
    assert r.status_code == 200, r.text
    limits = _by_key(r.json())
    # #1106: the posture is the tenant's licences byo_mode (app w2v), not an env var.
    assert limits["provider.byo_required"]["value"] == [
        "asr", "image", "llm", "ocr", "tts", "video_gen", "vlm",
    ]  # fmt: skip
    assert limits["provider.byo_required"]["source"] == "licence"
    assert limits["provider.allowed_kinds"]["value"] == ["openai"]
    rate = limits["provider.test_rate_limit"]
    assert rate["value"] == 0 and rate["enforced"] is False


@pytest.mark.parametrize(
    ("handler", "lookup"),
    [(lambda req: httpx.Response(404), "no_active_licence")],
)
async def test_licence_lookup_failures_still_answer(
    harness: Harness, handler: Any, lookup: str
) -> None:
    harness.app.dependency_overrides[get_licenses_client] = lambda: _licences(handler)
    r = await harness.call("GET", PATH, await _admin(harness))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["licence_lookup"] == lookup
    quota = _by_key(body)["licence.quota_monthly"]
    assert quota["value"] is None and quota["enforced"] is False
    # The BYO posture comes from licences too: when it cannot be read the view still
    # answers, showing it as UNKNOWN (None) — never a fabricated "nothing required".
    byo = _by_key(body)["provider.byo_required"]
    assert byo["enforced"] is True and byo["value"] is None


async def test_a_member_is_refused_and_there_is_no_write_route(harness: Harness) -> None:
    calls: list[str] = []

    def _record(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _healthy(request)

    harness.app.dependency_overrides[get_licenses_client] = lambda: _licences(_record)
    member = await _admin(harness, "member")
    r = await harness.call("GET", PATH, member)
    assert r.status_code == 403, r.text
    assert calls == [], "a refused caller must not trigger the licence lookup"
    admin = await _admin(harness)
    for method in ("PUT", "POST", "PATCH", "DELETE"):
        r = await harness.call(method, PATH, admin, json_body={})
        assert r.status_code == 405, (method, r.text)
    # Control: the same admin can read.
    assert (await harness.call("GET", PATH, admin)).status_code == 200
