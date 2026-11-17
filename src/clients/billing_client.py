"""Read client for billing-service's per-correlation cost endpoint.

The orchestrator's ``/usage``, ``/admin`` and run-viewer cost surfaces show $0
because the local ``run_usage`` table is never written. billing-service is the
authoritative, effective-dated pricer of every LLM cost event; since the
correlation-cost rollout it stores ``correlation_id = run_id`` (+ ``stage``) on
each event (worker_management / inference-router emit; the orchestrator stamps).
This client reads that priced cost back so the orchestrator can populate
``run_usage`` from the source of truth.

AUTH (matches fr's ``backend/src/integrations/metering/usage.ts``):
- ``Authorization: Bearer <service-account license token>`` so the in-cluster
  license-authz ext_authz admits the call (the standard platform-service path —
  NOT the w2v ``X-Tenant-Id: Bearer`` S2S convention, which billing's
  ``require_tenant`` would misread as the tenant string).
- a **plain** ``X-Tenant-Id: <uuid>`` override so billing's ``require_tenant``
  scopes the read to that exact tenant (defense-in-depth: billing's
  by-correlation endpoint filters by this tenant too).

``BILLING_AUTH_MODE`` (w2v_common ``TenantLicenceLane``, app ``w2v``): ``shared``
(default) is the above, unchanged. ``tenant_licence`` puts the BOUND tenant's own
minted ``w2v`` licence on ``Authorization`` instead of the shared token, and only
for a read of that same tenant: a read of any other tenant, no bound tenant, or a
refused mint RAISES (never the best-effort ``BillingCostUnavailableError``, never
the shared token).
"""

from __future__ import annotations

import json
import os
from typing import Any
from uuid import UUID

import httpx
from w2v_common.auth.bundle_tokens import is_licence_rejection

from src.clients.downstream_credential import (
    DownstreamCredentialMissingError,
    hop_lane,
    optional_out_of_cell_license_token,
)

_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_PATH = "/api/costs/by-correlation"


def _default_billing_base_url() -> str:
    """billing-service base URL from env (fr parity).

    billing is a cross-namespace platform service, not a w2v in-mesh service, so
    it is NOT in w2v_common discovery. Mirror fr's
    ``backend/src/integrations/metering/usage.ts``: prefer ``BILLING_BASE_URL``,
    fall back to ``BILLING_SERVICE_URL``. Empty when neither is set → the client
    degrades to a no-op (cost reconcile is best-effort; the page shows $0 rather
    than failing), and gitops wires the env in to light it up.
    """
    return (
        os.environ.get("BILLING_BASE_URL") or os.environ.get("BILLING_SERVICE_URL") or ""
    ).strip()


class BillingCostUnavailableError(RuntimeError):
    """billing-service was unreachable or returned an error (read is best-effort)."""


class BillingTenantLicenceMismatchError(RuntimeError):
    """``BILLING_AUTH_MODE=tenant_licence``: the read names a tenant other than the bound one."""


class BillingServiceClient:
    """Read priced per-correlation cost from billing-service.

    One instance per request is fine (cheap); the connection is opened and
    closed per call, matching the orchestrator's other thin proxies. ``base_url``
    + ``transport`` are injectable for tests.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        license_token: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = (base_url or _default_billing_base_url()).rstrip("/")
        # billing is OUT of the w2v cell: it needs this service's licence token.
        # ``None`` only under CELL_AUTH_MODE=enforce with no token configured;
        # then a configured billing call fails LOUD in ``costs_by_correlation``
        # (DownstreamCredentialMissingError — deliberately NOT the best-effort
        # BillingCostUnavailableError, which callers swallow as $0).
        self._lane = hop_lane("billing")
        self._token = (
            license_token if license_token is not None else optional_out_of_cell_license_token()
        )
        self._transport = transport

    async def _headers(self, tenant_id: UUID) -> tuple[dict[str, str], str | None]:
        """Request headers and, in tenant_licence mode, the tenant the licence was minted for."""
        headers = {"X-Tenant-Id": str(tenant_id)}
        if self._lane.tenant_licence:
            licence_tenant, minted = await self._lane.licence()
            if UUID(licence_tenant) != tenant_id:
                raise BillingTenantLicenceMismatchError(
                    f"billing read for tenant {tenant_id} under a licence minted for "
                    f"{licence_tenant}: refusing (BILLING_AUTH_MODE=tenant_licence)"
                )
            headers["Authorization"] = f"Bearer {minted.token}"
            if minted.license_id:
                headers["X-License-Id"] = minted.license_id
            return headers, licence_tenant
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers, None

    async def costs_by_correlation(
        self,
        tenant_id: UUID,
        correlation_ids: list[UUID],
        *,
        group_by: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return billing's priced cost grouped over the given correlation ids.

        Returns the ``data`` envelope payload (totals + ``groups``). Raises
        ``BillingCostUnavailableError`` on any transport/HTTP error so callers can
        degrade gracefully (cost stays $0 rather than 500-ing the page).
        """
        if not self._base_url:
            # No BILLING_BASE_URL configured — cost reconcile is best-effort, so
            # signal "unavailable" rather than hit an empty URL. Callers catch it.
            raise BillingCostUnavailableError("BILLING_BASE_URL not configured")
        if not correlation_ids:
            return {
                "correlation_ids": [],
                "group_by": group_by or [],
                "total_cost_usd": "0",
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "event_count": 0,
                "groups": [],
            }
        params: dict[str, str] = {"ids": ",".join(str(c) for c in correlation_ids)}
        if group_by:
            params["group_by"] = ",".join(group_by)
        if self._token is None and not self._lane.tenant_licence:
            raise DownstreamCredentialMissingError("billing")
        headers, minted_for = await self._headers(tenant_id)
        url = f"{self._base_url}{_PATH}"
        try:
            async with httpx.AsyncClient(
                timeout=_TIMEOUT, follow_redirects=False, transport=self._transport
            ) as client:
                resp = await client.get(url, headers=headers, params=params)
            if minted_for is not None and is_licence_rejection(resp.status_code):
                # The minted licence was refused: the next read re-mints.
                self._lane.invalidate(minted_for)
            if resp.status_code >= 400:
                raise BillingCostUnavailableError(
                    f"billing returned {resp.status_code}: {resp.text[:200]}"
                )
            body = resp.json()
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            # Degrade (not 500), per the docstring, on the full failure surface:
            # TransportError (connect/timeout), UnsupportedProtocol (a scheme-less
            # base URL is a RequestError, NOT a TransportError), and InvalidURL
            # (not even an HTTPError subclass).
            raise BillingCostUnavailableError(f"billing unreachable: {exc}") from exc
        except json.JSONDecodeError as exc:
            # billing returned a non-JSON body (e.g. an upstream proxy's HTML error
            # page). resp.json() must not escape as a raw 500.
            raise BillingCostUnavailableError(f"billing returned non-JSON body: {exc}") from exc
        # billing wraps responses in an envelope ({"data": ..., "meta": ...}).
        return body.get("data", body) if isinstance(body, dict) else {}
