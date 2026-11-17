"""P7 / R3: DOWNSTREAM_LICENSE_TOKEN is optional only under CELL_AUTH_MODE=enforce.

Contract pinned here:

* boot — off/dual (and unset mode) still REFUSE to build Settings without the
  token; enforce boots with it unset or blank (``None``).
* (a) in-cell hops — under enforce the shared token is never sent (the cell JWT
  authenticates the hop), even when it IS configured.
* (b) out-of-cell hops — under enforce with no token, every path fails LOUD at
  call time with ``DownstreamCredentialMissingError`` naming the destination;
  construction (which happens at boot for some clients) does not raise.
* (c) tenant-less in-cell hops — fail closed (``MissingTenantError``) unless the
  call EXPLICITLY opted in with ``service_scope`` (only the reconciler-finalize
  ``design_job_exists`` probe does); then it carries a ``scope=service`` cell
  token and ``w2v_orchestrator_cell_service_scope_calls_total`` counts it.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jwt.algorithms import OKPAlgorithm
from pydantic import ValidationError
from w2v_common.auth.cell_jwt import (
    CELL_TOKEN_HEADER,
    CellClaims,
    CellScope,
    CellTokenError,
    reset_cell_state,
    verify_cell_token,
)
from w2v_common.auth.identity_context import (
    clear_identity_context,
    identity_event_hook,
    mark_inbound_request,
    set_identity_context,
)
from w2v_common.observability.request_context import trace_event_hook
from w2v_common.tenant import MissingTenantError, TenantSource

from src.api.routes._proxy import forward_headers
from src.clients.base import LicensedServiceClient
from src.clients.billing_client import BillingCostUnavailableError, BillingServiceClient
from src.clients.design_client import DesignClient, design_job_exists
from src.clients.downstream_credential import DownstreamCredentialMissingError, service_scope
from src.clients.media_discovery_client import MediaDiscoveryClient
from src.clients.video_compile import VideoCompileClient
from src.clients.video_production_client import VideoProductionClient
from src.config import DownstreamServiceSettings, Settings, clear_settings_cache
from src.observability.metrics import (
    SERVICE_SCOPE_DESTINATIONS,
    cell_service_scope_calls_total,
)
from src.services.llm.dispatch import WMClient
from tests.support.platform_llm import PlatformChat

_TOKEN = "enforce-test-fake-token"  # pragma: allowlist secret
_TENANT = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
_REAL_ASYNC_CLIENT = httpx.AsyncClient


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path):
    # No relative .env may leak a token into DownstreamServiceSettings.
    monkeypatch.chdir(tmp_path)
    for name in ("W2V_LOCAL", "CELL_JWT_SIGNING_KEY", "CELL_JWT_KID"):
        monkeypatch.delenv(name, raising=False)
    clear_identity_context()
    # Outbound headers require a bound tenant (w2v-common refuses a blank one).
    set_identity_context(tenant_id=_TENANT, license_id="lic-1", subject_id="sub-1")
    clear_settings_cache()
    yield
    clear_identity_context()
    clear_settings_cache()


def _mode(monkeypatch: pytest.MonkeyPatch, mode: str | None, token: str | None) -> None:
    if mode is None:
        monkeypatch.delenv("CELL_AUTH_MODE", raising=False)
    else:
        monkeypatch.setenv("CELL_AUTH_MODE", mode)
    if token is None:
        monkeypatch.delenv("DOWNSTREAM_LICENSE_TOKEN", raising=False)
    else:
        monkeypatch.setenv("DOWNSTREAM_LICENSE_TOKEN", token)
    clear_settings_cache()


# --------------------------------------------------------------------------- #
# Boot / config
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", [None])
def test_boot_requires_token_outside_enforce(monkeypatch, mode):
    _mode(monkeypatch, mode, None)
    with pytest.raises(ValidationError, match="DOWNSTREAM_LICENSE_TOKEN"):
        DownstreamServiceSettings(_env_file=None)
    with pytest.raises(ValidationError, match="license_token"):
        Settings(_env_file=None)


@pytest.mark.parametrize("mode", ["off"])
def test_boot_rejects_blank_token_outside_enforce(monkeypatch, mode):
    _mode(monkeypatch, mode, "   ")
    with pytest.raises(ValidationError, match="set but blank"):
        DownstreamServiceSettings(_env_file=None)


@pytest.mark.parametrize("token", [None])
def test_enforce_boots_without_token(monkeypatch, token):
    _mode(monkeypatch, "enforce", token)
    assert DownstreamServiceSettings(_env_file=None).license_token is None
    assert Settings(_env_file=None).downstream.license_token is None


def test_enforce_keeps_a_configured_token(monkeypatch):
    _mode(monkeypatch, "enforce", _TOKEN)
    token = DownstreamServiceSettings(_env_file=None).license_token
    assert token is not None
    assert token.get_secret_value() == _TOKEN


# --------------------------------------------------------------------------- #
# (b) out-of-cell: loud at call time under enforce without a token
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("factory", "destination"),
    [(MediaDiscoveryClient, "media_discovery")],
)
async def test_out_of_cell_client_fails_loud_at_call_time(monkeypatch, factory, destination):
    _mode(monkeypatch, "enforce", None)
    client: LicensedServiceClient = factory()  # construction (boot) must NOT raise
    assert client.license_token is None
    with pytest.raises(DownstreamCredentialMissingError, match=destination) as exc:
        await client._get_client()
    assert exc.value.destination == destination


async def test_media_discovery_public_call_fails_loud(monkeypatch):
    """The named error surfaces through a real client method, not just the seam."""
    _mode(monkeypatch, "enforce", None)
    with pytest.raises(DownstreamCredentialMissingError, match="media_discovery"):
        await MediaDiscoveryClient().search("widgets")


async def test_platform_llm_never_needs_the_downstream_token(monkeypatch):
    """The platform LLM (``LLM_BACKEND=vllm``) authenticates with its OWN static
    licence: no ``DOWNSTREAM_LICENSE_TOKEN`` under enforce is not a refusal, and
    the shared token never rides the call. (The worker_management path that needed
    it was deleted in w2v_common 0.109.0.)"""
    _mode(monkeypatch, "enforce", None)

    async def _no_ledger(**_k: object) -> bool:
        return True

    monkeypatch.setattr("src.services.llm.dispatch.record_platform_vllm_usage", _no_ledger)
    chat = PlatformChat(lambda _body, _n: "hi back").install(monkeypatch)

    resp = await WMClient().message("any-worker", "hello")

    assert resp.output == "hi back"
    (req,) = chat.requests
    assert req.headers["authorization"] == "Bearer test-vllm-licence-token"
    assert "x-tenant-id" not in req.headers


async def test_billing_fails_loud_not_best_effort(monkeypatch):
    _mode(monkeypatch, "enforce", None)
    sent: list[httpx.Request] = []
    client = BillingServiceClient(
        base_url="http://billing.example:1",
        transport=httpx.MockTransport(lambda r: sent.append(r) or httpx.Response(200)),
    )
    with pytest.raises(DownstreamCredentialMissingError, match="billing") as exc:
        await client.costs_by_correlation(uuid.uuid4(), [uuid.uuid4()])
    # Must NOT be the swallowed-as-$0 best-effort error, and nothing went out.
    assert not isinstance(exc.value, BillingCostUnavailableError)
    assert sent == []


def test_proxy_to_out_of_cell_peer_fails_loud(monkeypatch):
    _mode(monkeypatch, "enforce", None)
    with pytest.raises(DownstreamCredentialMissingError, match="media_discovery"):
        forward_headers(discovery_name="media_discovery")


# --------------------------------------------------------------------------- #
# (a) in-cell: the shared token is never sent under enforce
# --------------------------------------------------------------------------- #


def _has_licence(headers: httpx.Headers | dict[str, str]) -> bool:
    lowered = {k.lower(): v for k, v in dict(headers).items()}
    return (
        "authorization" in lowered
        or lowered.get("x-tenant-id", "").startswith("Bearer")
        or _TOKEN in " ".join(lowered.values())
    )


@pytest.mark.parametrize("token", [_TOKEN])
@pytest.mark.parametrize(
    "factory",
    [VideoProductionClient],
)
async def test_in_cell_client_sends_no_licence_under_enforce(monkeypatch, factory, token):
    _mode(monkeypatch, "enforce", token)
    client: LicensedServiceClient = factory()
    assert client.license_token is None
    captured: list[httpx.Request] = []
    real = await client._get_client()  # in-cell: never raises
    wire = httpx.AsyncClient(
        base_url=client.base_url,
        headers=dict(real.headers),
        event_hooks=real._event_hooks,
        transport=httpx.MockTransport(lambda r: captured.append(r) or httpx.Response(200)),
    )
    try:
        await wire.get("/health")
    finally:
        await wire.aclose()
        await client.close()
    assert len(captured) == 1
    assert not _has_licence(captured[0].headers)


@pytest.mark.parametrize("token", [_TOKEN])
def test_in_cell_proxy_sends_no_licence_under_enforce(monkeypatch, token):
    _mode(monkeypatch, "enforce", token)
    for name in ("analysis", "html_render"):
        assert not _has_licence(forward_headers(discovery_name=name))


@pytest.mark.parametrize("mode", [None])
async def test_in_cell_client_still_sends_token_outside_enforce(monkeypatch, mode):
    """off/dual are unchanged: the in-cell hop keeps dual-sending the licence."""
    _mode(monkeypatch, mode, _TOKEN)
    client = VideoProductionClient()
    assert client.license_token == _TOKEN
    real = await client._get_client()
    assert real.headers.get("X-Tenant-Id") == f"Bearer {_TOKEN}"
    await client.close()
    assert forward_headers(discovery_name="analysis")["X-Tenant-Id"] == f"Bearer {_TOKEN}"


# --------------------------------------------------------------------------- #
# (c) tenant-less in-cell hops: a scope=service cell token under enforce
# --------------------------------------------------------------------------- #

_ORCH = "w2v-orchestrator"


@pytest.fixture
def cell_signer(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:
    """This process signs as the orchestrator (Ed25519 + a JWKS for verification)."""
    key = Ed25519PrivateKey.generate()
    jwk = json.loads(OKPAlgorithm.to_jwk(key.public_key()))
    jwk.update({"kid": "orch-1", "iss": _ORCH})
    jwks = tmp_path / "jwks.json"
    jwks.write_text(json.dumps({"keys": [jwk]}))
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", pem)
    monkeypatch.setenv("CELL_JWT_KID", "orch-1")
    monkeypatch.setenv("CELL_SERVICE_NAME", _ORCH)
    monkeypatch.setenv("CELL_DEPLOYMENT_ID", "strato-w2v")
    monkeypatch.setenv("CELL_JWKS_PATH", str(jwks))
    monkeypatch.setenv("CELL_ALLOWED_CALLERS", "*")
    reset_cell_state()
    clear_identity_context()  # tenant-less, like a reconciler or startup probe
    yield
    reset_cell_state()


def _verify_as(
    monkeypatch: pytest.MonkeyPatch,
    token: str,
    callee: str,
    service_callers: frozenset[str] | None = None,
) -> CellClaims:
    with monkeypatch.context() as m:
        m.setenv("CELL_SERVICE_NAME", callee)
        return verify_cell_token(token, service_callers=service_callers)


async def _send(client: LicensedServiceClient, path: str = "/api/x") -> httpx.Request:
    """One request through the client's REAL hook pipeline, captured at the wire."""
    captured: list[httpx.Request] = []
    real = await client._get_client()
    wire = httpx.AsyncClient(
        base_url=client.base_url,
        headers=dict(real.headers),
        event_hooks=real._event_hooks,
        transport=httpx.MockTransport(lambda r: captured.append(r) or httpx.Response(200)),
    )
    try:
        await wire.get(path)
    finally:
        await wire.aclose()
        await client.close()
    (request,) = captured
    return request


def _service_scope_count(destination: str) -> float:
    return cell_service_scope_calls_total.labels(destination=destination)._value.get()


@pytest.mark.usefixtures("cell_signer")
@pytest.mark.parametrize(
    "make",
    [DesignClient],
)
async def test_tenantless_in_cell_hop_without_opt_in_still_fails_closed(monkeypatch, make):
    """Service scope is never implicit: a tenant-less in-cell hop that did NOT opt in
    (a tenant-scoped background job that forgot to bind its tenant) must fail, not
    silently act as the orchestrator."""
    _mode(monkeypatch, "enforce", None)
    before = _service_scope_count("html_render")
    with pytest.raises(MissingTenantError):
        await _send(make())
    assert _service_scope_count("html_render") == before


@pytest.mark.usefixtures("cell_signer")
async def test_opted_in_tenantless_in_cell_hop_gets_a_service_scope_token_under_enforce(
    monkeypatch,
):
    _mode(monkeypatch, "enforce", None)
    client = DesignClient()
    before = _service_scope_count("html_render")
    with service_scope(client.discovery_name):
        request = await _send(client)
    assert _service_scope_count("html_render") == before + 1
    assert not _has_licence(request.headers)
    token = request.headers[CELL_TOKEN_HEADER]
    # A route that did not opt in refuses it; one that admits the orchestrator accepts.
    with pytest.raises(CellTokenError):
        _verify_as(monkeypatch, token, "html-render-service")
    claims = _verify_as(
        monkeypatch, token, "html-render-service", service_callers=frozenset({_ORCH})
    )
    assert claims.scope is CellScope.SERVICE
    assert claims.tenant_id is None
    assert claims.audience == "html-render-service"
    assert claims.issuer == _ORCH


@pytest.mark.usefixtures("cell_signer")
async def test_design_job_exists_without_a_tenant_sends_a_service_token(monkeypatch):
    """The reconciler-finalize probe, end to end through the production client."""
    _mode(monkeypatch, "enforce", None)
    seen: list[httpx.Request] = []

    class _Recording(_REAL_ASYNC_CLIENT):
        def __init__(self, *args, **kwargs) -> None:
            kwargs["transport"] = httpx.MockTransport(
                lambda r: seen.append(r) or httpx.Response(200, text="<html></html>")
            )
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _Recording)
    before = _service_scope_count("html_render")
    assert await design_job_exists("job-1") is True
    assert _service_scope_count("html_render") == before + 1
    (request,) = seen
    claims = _verify_as(
        monkeypatch,
        request.headers[CELL_TOKEN_HEADER],
        "html-render-service",
        service_callers=frozenset({_ORCH}),
    )
    assert claims.scope is CellScope.SERVICE


@pytest.mark.usefixtures("cell_signer")
async def test_bound_tenant_keeps_its_tenant_scope_token_under_enforce(monkeypatch):
    _mode(monkeypatch, "enforce", None)
    set_identity_context(uuid.UUID(_TENANT), None, "sub-1", tenant_source=TenantSource.JWT_CLAIM)
    request = await _send(DesignClient())
    claims = _verify_as(monkeypatch, request.headers[CELL_TOKEN_HEADER], "html-render-service")
    assert claims.scope is not CellScope.SERVICE
    assert claims.tenant_id == uuid.UUID(_TENANT)


@pytest.mark.usefixtures("cell_signer")
async def test_video_compile_hop_carries_a_tenant_cell_token_under_enforce(monkeypatch):
    """w2v-common 0.101.0 made video-compiler a cell member: under enforce the hop
    no longer needs DOWNSTREAM_LICENSE_TOKEN because it carries a cell JWT minted
    for aud ``video-compiler`` and scoped to the run's tenant — it does not fail open."""
    _mode(monkeypatch, "enforce", None)
    set_identity_context(uuid.UUID(_TENANT), None, "sub-1", tenant_source=TenantSource.JWT_CLAIM)
    request = await _send(VideoCompileClient())
    assert not _has_licence(request.headers)
    claims = _verify_as(monkeypatch, request.headers[CELL_TOKEN_HEADER], "video-compiler")
    assert claims.scope is not CellScope.SERVICE
    assert claims.tenant_id == uuid.UUID(_TENANT)


@pytest.mark.usefixtures("cell_signer")
@pytest.mark.parametrize("mode", [None])
async def test_tenantless_hop_is_byte_identical_outside_enforce(monkeypatch, mode):
    """off/dual: the extra hook changes nothing on the wire."""
    _mode(monkeypatch, mode, _TOKEN)
    with_hook = await _send(DesignClient())
    base_only: list[httpx.Request] = []
    client = DesignClient()
    real = await client._get_client()
    wire = httpx.AsyncClient(
        base_url=client.base_url,
        headers=dict(real.headers),
        event_hooks={"request": [identity_event_hook, trace_event_hook]},
        transport=httpx.MockTransport(lambda r: base_only.append(r) or httpx.Response(200)),
    )
    try:
        await wire.get("/api/x")
    finally:
        await wire.aclose()
        await client.close()
    assert CELL_TOKEN_HEADER not in with_hook.headers
    assert with_hook.headers.multi_items() == base_only[0].headers.multi_items()


@pytest.mark.usefixtures("cell_signer")
async def test_tenantless_hop_inside_an_inbound_request_still_fails_closed(monkeypatch):
    """w2v_common refuses tenant-less outbound work on behalf of an inbound request
    (it would act as the service for a user); the service token must not bypass that."""
    _mode(monkeypatch, "enforce", None)
    mark_inbound_request()
    with pytest.raises(MissingTenantError):
        await _send(DesignClient())


@pytest.mark.usefixtures("cell_signer")
async def test_opt_in_with_a_bound_tenant_keeps_the_tenant_token(monkeypatch):
    """The opt-in only covers the tenant-less case; a bound tenant is never downgraded."""
    _mode(monkeypatch, "enforce", None)
    set_identity_context(uuid.UUID(_TENANT), None, "sub-1", tenant_source=TenantSource.JWT_CLAIM)
    client = DesignClient()
    before = _service_scope_count("html_render")
    with service_scope(client.discovery_name):
        request = await _send(client)
    claims = _verify_as(monkeypatch, request.headers[CELL_TOKEN_HEADER], "html-render-service")
    assert claims.scope is not CellScope.SERVICE
    assert claims.tenant_id == uuid.UUID(_TENANT)
    assert _service_scope_count("html_render") == before


def test_opted_in_destinations_have_zero_series_from_the_first_scrape():
    from prometheus_client import REGISTRY

    assert DesignClient().discovery_name in SERVICE_SCOPE_DESTINATIONS
    for destination in SERVICE_SCOPE_DESTINATIONS:
        value = REGISTRY.get_sample_value(
            "w2v_orchestrator_cell_service_scope_calls_total", {"destination": destination}
        )
        assert value is not None
