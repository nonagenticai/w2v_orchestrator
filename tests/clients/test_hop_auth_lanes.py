"""Out-of-cell hops switched by ``<HOP>_AUTH_MODE`` (webscraper, media_discovery, modality_gateway).

``shared`` (default) must be byte-for-byte today's wire: the shared
``DOWNSTREAM_LICENSE_TOKEN`` on ``X-Tenant-Id: Bearer``. ``tenant_licence`` must
send the bound tenant's own minted licence for the hop's app and nothing else.
Assertions are on the request the recorded downstream actually received, with the
production client's real event hooks installed.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from prometheus_client import REGISTRY
from w2v_common.auth.bundle_tokens import (
    BundleMinterNotConfiguredError,
    BundleTokenMinter,
    reset_bundle_token_minter,
    set_bundle_token_minter,
)
from w2v_common.auth.identity_context import (
    NoTenantForLicenceError,
    clear_identity_context,
    set_identity_context,
)
from w2v_common.clients import WebscraperClient
from w2v_common.tenant import TenantSource

from src.clients.billing_client import (
    BillingCostUnavailableError,
    BillingServiceClient,
    BillingTenantLicenceMismatchError,
)
from src.clients.downstream_credential import assert_hop_lanes_boot_config, hop_lane
from src.clients.media_discovery_client import MediaDiscoveryClient
from src.clients.tts_client import TTSClient
from src.config import get_settings

TENANT = "0199a1b2-0000-7000-8000-00000000000a"
SHARED = "test-downstream-license-token"  # pragma: allowlist secret — conftest default
TOKEN_URL = "http://kc.test/realms/secure-apps/protocol/openid-connect/token"
_REAL_ASYNC_CLIENT = httpx.AsyncClient
_MODES = (
    "WEBSCRAPER_AUTH_MODE",
    "MEDIA_DISCOVERY_AUTH_MODE",
    "MODALITY_GATEWAY_AUTH_MODE",
    "BILLING_AUTH_MODE",
)


def _metric(app: str, outcome: str) -> float:
    v = REGISTRY.get_sample_value("w2v_bundle_token_mint_total", {"app": app, "outcome": outcome})
    return 0.0 if v is None else v


class _Wire:
    """Records downstream requests; answers the minter's Keycloak + licences calls."""

    def __init__(self) -> None:
        self.downstream: list[httpx.Request] = []
        self.minted_apps: list[str] = []
        #: Status codes the downstream answers, in order; then 200.
        self.statuses: list[int] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_URL:
            return httpx.Response(200, json={"access_token": "minter-sa", "expires_in": 300})
        if request.url.host == "lic.test":
            app = request.url.params["app"]
            self.minted_apps.append(app)
            n = len(self.minted_apps)
            return httpx.Response(
                201,
                json={
                    "token": f"lic-{app}-{n}",
                    "expires_at": datetime.fromtimestamp(4_000_000_000, tz=UTC).isoformat(),
                    "tenant_id": request.url.path.split("/")[3],
                    "app": app,
                    "license_id": f"lid-{app}-{n}",
                },
            )
        self.downstream.append(request)
        status = self.statuses.pop(0) if self.statuses else 200
        return httpx.Response(status, json={"text_content": "page", "title": "t"})


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Wire]:
    for name in _MODES:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    w = _Wire()

    def _factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(w.handler)
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _factory)
    reset_bundle_token_minter()
    clear_identity_context()
    yield w
    clear_identity_context()
    reset_bundle_token_minter()
    get_settings.cache_clear()


@pytest.fixture
def minter(wire: _Wire) -> None:
    set_bundle_token_minter(
        BundleTokenMinter(
            licenses_url="http://lic.test",
            token_url=TOKEN_URL,
            client_id="w2v-bundle-minter",
            client_secret="s",  # pragma: allowlist secret
            transport=httpx.MockTransport(wire.handler),
        )
    )


def _bind() -> None:
    set_identity_context(
        TENANT, "inbound-licence", "subject-1", tenant_source=TenantSource.LICENSE_TOKEN
    )


# -- default (shared): unchanged ---------------------------------------------------


def _webscraper() -> WebscraperClient:
    return WebscraperClient(base_url="http://ws.test:9119", licence_lane=hop_lane("webscraper"))


async def test_webscraper_default_is_shared_token(wire: _Wire) -> None:
    _bind()
    ws = _webscraper()
    resp = await ws.scrape_url("https://example.com")
    assert resp.text == "page"
    req = wire.downstream[-1]
    assert req.headers["X-Tenant-Id"] == f"Bearer {SHARED}"
    assert wire.minted_apps == []
    await ws.close()


async def test_media_discovery_default_is_shared_token(wire: _Wire) -> None:
    _bind()
    md = MediaDiscoveryClient(base_url="http://md.test:8809")
    pooled = await md._get_client()
    await pooled.post("/api/search", json={"query": "x"})
    assert wire.downstream[-1].headers["X-Tenant-Id"] == f"Bearer {SHARED}"
    assert wire.minted_apps == []
    await md.close()


@pytest.mark.parametrize("cls", [TTSClient])
async def test_modality_default_is_shared_token(wire: _Wire, cls: type) -> None:
    _bind()
    client = cls(base_url="http://modality.test:8890")
    pooled = await client._get_client()
    await pooled.post("/api/tts/synthesize", json={})
    assert wire.downstream[-1].headers["X-Tenant-Id"] == f"Bearer {SHARED}"
    assert wire.minted_apps == []
    await client.close()


# -- tenant_licence: the tenant's own licence, never the shared token ----------------


async def test_webscraper_tenant_licence_sends_minted_w2v_licence(
    wire: _Wire, minter: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # webscraper has no app of its own: its lane app key is ``w2v`` (w2v_common HOP_APP_KEYS).
    monkeypatch.setenv("WEBSCRAPER_AUTH_MODE", "tenant_licence")
    _bind()
    before = _metric("w2v", "minted")
    ws = _webscraper()
    await ws.scrape_url("https://example.com")
    req = wire.downstream[-1]
    assert req.headers["X-Tenant-Id"] == "Bearer lic-w2v-1"
    assert req.headers["X-License-Id"] == "lid-w2v-1"
    assert "X-Effective-Tenant-Id" not in req.headers
    assert SHARED not in str(req.headers)
    assert wire.minted_apps == ["w2v"]
    assert _metric("w2v", "minted") == before + 1
    await ws.close()


async def test_media_discovery_tenant_licence_sends_minted_w2v_licence(
    wire: _Wire, minter: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEDIA_DISCOVERY_AUTH_MODE", "tenant_licence")
    _bind()
    md = MediaDiscoveryClient(base_url="http://md.test:8809")
    pooled = await md._get_client()
    await pooled.post("/api/search", json={"query": "x"})
    req = wire.downstream[-1]
    assert req.headers["X-Tenant-Id"] == "Bearer lic-w2v-1"
    assert SHARED not in str(req.headers)
    assert wire.minted_apps == ["w2v"]
    await md.close()


@pytest.mark.parametrize("cls", [TTSClient])
async def test_modality_tenant_licence_sends_minted_w2v_licence(
    wire: _Wire, minter: None, monkeypatch: pytest.MonkeyPatch, cls: type
) -> None:
    monkeypatch.setenv("MODALITY_GATEWAY_AUTH_MODE", "tenant_licence")
    _bind()
    client = cls(base_url="http://modality.test:8890")
    pooled = await client._get_client()
    await pooled.post("/api/tts/synthesize", json={})
    req = wire.downstream[-1]
    assert req.headers["X-Tenant-Id"] == "Bearer lic-w2v-1"
    assert "X-Effective-Tenant-Id" not in req.headers
    assert SHARED not in str(req.headers)
    assert wire.minted_apps == ["w2v"]
    await client.close()


async def test_modality_flip_leaves_webscraper_shared(
    wire: _Wire, minter: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One hop per flip: MODALITY_GATEWAY_AUTH_MODE must not move webscraper.
    monkeypatch.setenv("MODALITY_GATEWAY_AUTH_MODE", "tenant_licence")
    _bind()
    ws = _webscraper()
    await ws.scrape_url("https://example.com")
    assert wire.downstream[-1].headers["X-Tenant-Id"] == f"Bearer {SHARED}"
    await ws.close()


async def test_tenant_licence_without_tenant_raises_and_sends_nothing(
    wire: _Wire, minter: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WEBSCRAPER_AUTH_MODE", "tenant_licence")
    ws = _webscraper()
    with pytest.raises(NoTenantForLicenceError):
        await ws.scrape_url("https://example.com")
    assert wire.downstream == []
    await ws.close()


# -- boot ------------------------------------------------------------------------------


def test_boot_rejects_a_typo_mode(wire: _Wire, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBSCRAPER_AUTH_MODE", "tenant-licence")
    with pytest.raises(ValueError, match="WEBSCRAPER_AUTH_MODE"):
        assert_hop_lanes_boot_config()


def test_boot_rejects_tenant_licence_without_minter(
    wire: _Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "BUNDLE_MINTER_CLIENT_ID",
        "BUNDLE_MINTER_CLIENT_SECRET",
        "BUNDLE_MINTER_TOKEN_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODALITY_GATEWAY_AUTH_MODE", "tenant_licence")
    with pytest.raises(BundleMinterNotConfiguredError):
        assert_hop_lanes_boot_config()


def test_boot_default_is_fine(wire: _Wire) -> None:
    assert_hop_lanes_boot_config()


# -- billing (BILLING_AUTH_MODE) ------------------------------------------------------


def _billing(wire: _Wire) -> BillingServiceClient:
    return BillingServiceClient(
        base_url="http://billing.test", transport=httpx.MockTransport(wire.handler)
    )


async def test_billing_default_is_shared_token(wire: _Wire) -> None:
    _bind()
    await _billing(wire).costs_by_correlation(UUID(TENANT), [uuid4()])
    req = wire.downstream[-1]
    assert req.headers["Authorization"] == f"Bearer {SHARED}"
    assert req.headers["X-Tenant-Id"] == TENANT
    assert wire.minted_apps == []


async def test_billing_tenant_licence_sends_minted_w2v_licence(
    wire: _Wire, minter: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BILLING_AUTH_MODE", "tenant_licence")
    _bind()
    await _billing(wire).costs_by_correlation(UUID(TENANT), [uuid4()])
    req = wire.downstream[-1]
    assert req.headers["Authorization"] == "Bearer lic-w2v-1"
    assert req.headers["X-License-Id"] == "lid-w2v-1"
    assert req.headers["X-Tenant-Id"] == TENANT  # plain uuid, billing's require_tenant
    assert SHARED not in str(req.headers)
    assert wire.minted_apps == ["w2v"]


async def test_billing_tenant_licence_refuses_another_tenants_read(
    wire: _Wire, minter: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BILLING_AUTH_MODE", "tenant_licence")
    _bind()
    with pytest.raises(BillingTenantLicenceMismatchError):
        await _billing(wire).costs_by_correlation(uuid4(), [uuid4()])
    assert wire.downstream == []


async def test_billing_tenant_licence_401_remints_next_read(
    wire: _Wire, minter: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BILLING_AUTH_MODE", "tenant_licence")
    _bind()
    wire.statuses = [401]
    client = _billing(wire)
    with pytest.raises(BillingCostUnavailableError, match="401"):
        await client.costs_by_correlation(UUID(TENANT), [uuid4()])
    await client.costs_by_correlation(UUID(TENANT), [uuid4()])
    sent = [r.headers["Authorization"] for r in wire.downstream]
    assert sent == ["Bearer lic-w2v-1", "Bearer lic-w2v-2"]
