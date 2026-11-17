from uuid import uuid4

import httpx
import pytest

from src.clients.billing_client import BillingCostUnavailableError, BillingServiceClient

pytestmark = pytest.mark.asyncio


async def test_empty_ids_short_circuits_without_a_call() -> None:
    called = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        called["n"] += 1
        return httpx.Response(200, json={"data": {}})

    client = BillingServiceClient(
        base_url="http://billing.invalid", transport=httpx.MockTransport(handler)
    )
    out = await client.costs_by_correlation(uuid4(), [])
    assert out["event_count"] == 0
    assert called["n"] == 0


async def test_parses_envelope_and_sends_auth_and_tenant_headers() -> None:
    tenant = uuid4()
    run_id = uuid4()
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        seen["__params__"] = request.url.query.decode()
        return httpx.Response(
            200,
            json={
                "data": {
                    "total_cost_usd": "0.15",
                    "groups": [{"key": {"stage": "narration"}, "cost_usd": "0.15"}],
                },
                "meta": {},
            },
        )

    client = BillingServiceClient(
        base_url="http://billing.invalid",
        license_token="svc-tok",
        transport=httpx.MockTransport(handler),
    )
    data = await client.costs_by_correlation(tenant, [run_id], group_by=["stage"])
    assert data["total_cost_usd"] == "0.15"
    assert data["groups"][0]["key"]["stage"] == "narration"
    # Auth: Bearer service token + plain X-Tenant-Id (fr's billing pattern).
    assert seen["authorization"] == "Bearer svc-tok"
    assert seen["x-tenant-id"] == str(tenant)
    assert f"ids={run_id}" in seen["__params__"]
    assert "group_by=stage" in seen["__params__"]


async def test_http_error_raises_unavailable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    client = BillingServiceClient(
        base_url="http://billing.invalid", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(BillingCostUnavailableError):
        await client.costs_by_correlation(uuid4(), [uuid4()])


async def test_transport_error_raises_unavailable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = BillingServiceClient(
        base_url="http://billing.invalid", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(BillingCostUnavailableError):
        await client.costs_by_correlation(uuid4(), [uuid4()])


async def test_decoding_error_degrades_not_500() -> None:
    """A DecodingError (bad content-encoding on billing's response) is a RequestError
    but NOT a TransportError, so the old `except httpx.TransportError` let it escape
    as a 500. The docstring promises degrade-on-any-HTTP-error, so it must surface
    as BillingCostUnavailableError."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-encoding": "gzip"}, content=b"not-actually-gzip"
        )

    client = BillingServiceClient(
        base_url="http://billing.invalid", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(BillingCostUnavailableError):
        await client.costs_by_correlation(uuid4(), [uuid4()])


async def test_non_json_body_degrades_not_500() -> None:
    """billing returning a 200 with a non-JSON body (e.g. an HTML error page from
    an upstream proxy) makes resp.json() raise JSONDecodeError. That must degrade
    to BillingCostUnavailableError, not escape as a raw 500."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway error</html>")

    client = BillingServiceClient(
        base_url="http://billing.invalid", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(BillingCostUnavailableError):
        await client.costs_by_correlation(uuid4(), [uuid4()])
