"""Shared thin HTTP proxy helpers for analysis_service-backed routers.

The orchestrator owns the public ``w2v.nonagentic.ai`` API surface; the Istio
VirtualService routes only to ``w2v-frontend → w2v-orchestrator``.
``analysis_service`` (critique + redesign domains) stays internal on port 6303.
The critique and redesign routers re-expose its endpoints and forward either the
inbound caller's license token (when present) or the orchestrator's own
``DOWNSTREAM_LICENSE_TOKEN`` (when the inbound auth is a Keycloak JWT from the
browser, which does not carry a license).

P7 (intra-cell auth): every proxied request ALSO carries a fresh
``X-W2V-Cell-Token`` minted by w2v_common's ``cell_token_event_hook`` on each
send (tenant = the request's ``TenantIdDep`` tenant, actor = the human ``sub``,
aud = the target's cell name). The licence headers above are still sent
(dual-send) until the receivers flip to ``CELL_AUTH_MODE=enforce``. Under
enforce the service-account fallback is NOT sent to an in-cell peer (see
``src/clients/downstream_credential.py``); a caller's own licence still is.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from uuid import UUID

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from w2v_common.auth.identity_context import (
    EFFECTIVE_TENANT_HEADER,
    cell_token_event_hook,
    get_caller_license_token,
    get_identity_headers,
)
from w2v_common.discovery import service_url
from w2v_common.tenant import parse_tenant_id

from src.auth.api_key_format import KEY_SCHEME
from src.auth.cell_tenant import send_as_tenant
from src.clients.downstream_credential import (
    in_cell_license_token,
    is_in_cell,
    require_out_of_cell_license_token,
)
from src.clients.gpu_busy import BUSY_STATUSES, busy_retry_after

ANALYSIS_TIMEOUT = httpx.Timeout(60.0, connect=5.0)
#: Brand ICP / extract scrape + LLM synchronously (P9-7): marketing's own client
#: waits 60 s (``RECON_W2V_ANALYSIS_TIMEOUT_SECONDS``); the extra margin is the
#: scrape, so this hop never times out before the caller does.
ANALYSIS_SLOW_TIMEOUT = httpx.Timeout(120.0, connect=5.0)

#: Signs EVERY send (no token is held, so a retry or a poll loop past the 60 s
#: TTL re-mints). Only the cell token: the legacy headers are passed per call by
#: ``forward_headers`` exactly as before.
_CELL_HOOKS: dict[str, list[Callable[[httpx.Request], Awaitable[None]]]] = {
    "request": [cell_token_event_hook]
}


def _effective(effective_tenant: UUID | str | None) -> UUID | None:
    """``effective_tenant`` (``TenantIdDep``) as a UUID; ``None`` = use the ambient one."""
    if effective_tenant is None:
        return None
    return parse_tenant_id(effective_tenant, source="proxy effective tenant")


def _service_account_token(discovery_name: str) -> str:
    """The orchestrator's shared licence token for a hop to ``discovery_name``.

    In-cell peer (every current proxy target: analysis, html_render) → the token
    under CELL_AUTH_MODE off/dual, ``""`` (send none — the cell JWT minted by
    ``cell_token_event_hook`` authenticates the hop) under enforce. Out-of-cell →
    the token, or ``DownstreamCredentialMissingError`` when none is configured.
    """
    if is_in_cell(service_url(discovery_name)):
        return in_cell_license_token() or ""
    return require_out_of_cell_license_token(discovery_name)


def forward_headers(
    effective_tenant: UUID | str | None = None, *, discovery_name: str = "analysis"
) -> dict[str, str]:
    """Outbound headers: license token + identity X-* headers + the caller's real tenant.

    ``X-Tenant-Id: Bearer <license token>`` is NOT a typo and must not be
    "cleaned up" to a bare UUID: it is the ONLY channel w2v_common's
    ``AuthMiddleware`` accepts an opaque license token on (see
    ``AuthMiddleware._extract_license_token`` — ``Authorization`` is reserved for
    the caller's own Keycloak JWT and is never introspected as a license). Every
    fleet client (WMClient, ArtifactClient, WebscraperClient, …) sends it this way.

    ``X-Effective-Tenant-Id`` is what makes the downstream job belong to the
    CUSTOMER. A browser user authenticates here with a Keycloak JWT, so there is
    no caller license token and the token below is the orchestrator's own
    service-account ``DOWNSTREAM_LICENSE_TOKEN`` — which introspects to the SA's
    tenant, collapsing every customer's jobs into one tenant downstream. The
    side-channel carries the real tenant alongside it; the receiver honours it
    only when the introspected subject is in ``AUTH_DELEGATING_SUBJECTS`` (the
    orchestrator SA is; a customer's own token never is), so it cannot be used to
    spoof a tenant. Stamped unconditionally, exactly as ``identity_event_hook``
    does for the cached fleet clients.

    ``effective_tenant`` MUST come from ``_common.get_tenant_id`` (``TenantIdDep``)
    — the single server-side chokepoint that resolves the caller's verified claim
    plus a ``super_admin``'s scope-switch. It is never read from a client-supplied
    header/param here: ``get_tenant_id`` silently ignores a non-super caller's
    override, so a customer cannot steer this value.
    """
    headers = {k: v for k, v in get_identity_headers().items() if v}
    # A tenant API key (``w2v_sk_``) is NEVER a caller licence token: the
    # tenancy middleware binds ``caller_license_token=None`` on that lane, so a
    # key caller falls to the service-account token + the delegated tenant below.
    # Only the orchestrator can verify a ``w2v_sk_`` key; forwarded, it would be
    # a 401 downstream at best and a leaked credential at worst.
    caller = get_caller_license_token()
    if caller and caller.startswith(KEY_SCHEME):  # belt-and-braces for the rule above
        caller = None
    token = caller or _service_account_token(discovery_name)
    if token:
        bearer = f"Bearer {token}"
        headers["Authorization"] = bearer
        headers["X-Tenant-Id"] = bearer
    tenant = str(effective_tenant).strip() if effective_tenant else ""
    if tenant:
        headers[EFFECTIVE_TENANT_HEADER] = tenant
    return headers


def _service_url(path: str, discovery_name: str = "analysis") -> str:
    """Build an upstream URL for ``path`` on the given discovery peer.

    Defaults to ``analysis`` (critique stays on analysis_service). The redesign
    facade passes ``discovery_name="html_render"`` so it forwards to the new
    web-design service instead.
    """
    return f"{service_url(discovery_name).rstrip('/')}{path}"


async def proxy_get(
    path: str,
    params: dict[str, str | int] | None = None,
    *,
    discovery_name: str = "analysis",
    effective_tenant: UUID | str | None = None,
    timeout: httpx.Timeout = ANALYSIS_TIMEOUT,
) -> dict:
    """Forward a GET to the upstream peer and return parsed JSON."""
    url = _service_url(path, discovery_name)
    headers = forward_headers(effective_tenant, discovery_name=discovery_name)
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=False, event_hooks=_CELL_HOOKS
    ) as client:
        try:
            resp = await send_as_tenant(
                _effective(effective_tenant), client.get(url, params=params, headers=headers)
            )
        except httpx.TransportError as exc:
            raise HTTPException(
                status_code=502, detail=f"{discovery_name} unreachable: {exc}"
            ) from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


async def proxy_post(
    path: str,
    json: dict,
    *,
    discovery_name: str = "analysis",
    effective_tenant: UUID | str | None = None,
    timeout: httpx.Timeout = ANALYSIS_TIMEOUT,
) -> dict:
    """Forward a POST (JSON body) to the upstream peer and return parsed JSON."""
    url = _service_url(path, discovery_name)
    headers = forward_headers(effective_tenant, discovery_name=discovery_name)
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=False, event_hooks=_CELL_HOOKS
    ) as client:
        try:
            resp = await send_as_tenant(
                _effective(effective_tenant), client.post(url, json=json, headers=headers)
            )
        except httpx.TransportError as exc:
            raise HTTPException(
                status_code=502, detail=f"{discovery_name} unreachable: {exc}"
            ) from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


async def proxy_stream(
    path: str,
    default_media_type: str,
    *,
    discovery_name: str = "analysis",
    effective_tenant: UUID | str | None = None,
) -> StreamingResponse:
    """Buffer-and-stream a binary/text artifact (PDF, HTML) from the upstream peer.

    Bodies here are modest (one report / one variant per job), so buffering keeps
    AsyncClient lifecycle management straightforward across error paths.
    """
    url = _service_url(path, discovery_name)
    headers = forward_headers(effective_tenant, discovery_name=discovery_name)
    async with httpx.AsyncClient(
        timeout=ANALYSIS_TIMEOUT, follow_redirects=False, event_hooks=_CELL_HOOKS
    ) as client:
        try:
            upstream = await send_as_tenant(
                _effective(effective_tenant), client.get(url, headers=headers)
            )
        except httpx.TransportError as exc:
            raise HTTPException(
                status_code=502, detail=f"{discovery_name} unreachable: {exc}"
            ) from exc
    if upstream.status_code >= 400:
        raise HTTPException(status_code=upstream.status_code, detail=upstream.text)

    return StreamingResponse(
        iter([upstream.content]),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", default_media_type),
        headers={
            k: v
            for k, v in upstream.headers.items()
            if k.lower() in {"content-disposition", "content-length"}
        },
    )


#: Upstream statuses passed through to the caller unchanged: they describe the
#: CALLER's request (bad input, unknown job, too large, unsupported media).
_PASS_THROUGH_4XX: frozenset[int] = frozenset({400, 404, 409, 413, 415, 422})
_DETAIL_MAX = 300
#: A passed-through ``gpu_busy`` lane name is bounded like any upstream string.
_LANE_MAX = 64


def _upstream_detail(resp: httpx.Response) -> str | None:
    """A short, string ``detail`` from the upstream body — never the raw body."""
    try:
        body = resp.json()
    except ValueError:
        return None
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str) and detail.strip():
        return detail.strip()[:_DETAIL_MAX]
    return None


def _busy_error(resp: httpx.Response, discovery_name: str) -> HTTPException:
    """429/503 → the SAME status + ``Retry-After`` (GPU admission contract, #1111).

    The hint is the upstream ``Retry-After`` header, else its body's
    ``retry_after`` (clamped ``1..120`` s, as every in-process caller reads it). A
    ``gpu_busy`` body keeps its contract shape (``error``/``lane``/``retry_after``
    plus a human ``message`` — never the raw body) so a caller can tell a full GPU
    lane from any other "not now"; anything else gets a short string.
    """
    delay = busy_retry_after(resp)
    retry = str(math.ceil(delay)) if delay is not None else None
    detail: str | dict[str, object] = (
        f"{discovery_name} is busy; retry shortly"
        if resp.status_code == 429
        else f"{discovery_name} is unavailable; retry shortly"
    )
    try:
        body = resp.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and isinstance(body.get("detail"), dict):
        body = body["detail"]  # FastAPI's ``HTTPException(detail={...})`` envelope
    if isinstance(body, dict) and body.get("error") == "gpu_busy":
        lane = body.get("lane")
        detail = {
            "error": "gpu_busy",
            "message": f"{discovery_name} is busy; retry shortly",
            "lane": lane[:_LANE_MAX] if isinstance(lane, str) else None,
            "retry_after": int(retry) if retry else None,
        }
    return HTTPException(
        status_code=resp.status_code,
        detail=detail,
        headers={"Retry-After": retry} if retry else None,
    )


def map_upstream_error(resp: httpx.Response, discovery_name: str) -> HTTPException:
    """The caller-facing answer for a failed upstream response.

    * 400/404/409/413/415/422 → the same status (about the caller's request),
      with the upstream's short string ``detail`` when it sent one;
    * 402 → 402 (the tenant's entitlement);
    * 429/503 → the SAME status + ``Retry-After`` (when the upstream gave one, in
      the header or a ``gpu_busy`` body): a busy GPU lane is "not now", never a
      5xx failure of ours (#1111). A non-busy 503 is still "unavailable, retry"
      and passes through the same way — it is never re-labelled 502;
    * 401/403/407 → **502**: the upstream refused THIS service's credential — a
      configuration fault of the deployment, never the caller's;
    * anything else (other 5xx) → **502**.

    The upstream body is never echoed beyond a bounded ``detail`` string.
    """
    code = resp.status_code
    if code in _PASS_THROUGH_4XX:
        return HTTPException(
            status_code=code,
            detail=_upstream_detail(resp) or f"{discovery_name} rejected the request",
        )
    if code == 402:
        return HTTPException(status_code=402, detail="This tenant has no active Studio licence.")
    if code in BUSY_STATUSES:
        return _busy_error(resp, discovery_name)
    if code in (401, 403, 407):
        return HTTPException(
            status_code=502,
            detail=f"{discovery_name} refused this service's credential (HTTP {code})",
        )
    return HTTPException(status_code=502, detail=f"{discovery_name} failed (HTTP {code})")


async def proxy_request(
    method: str,
    path: str,
    *,
    discovery_name: str,
    effective_tenant: UUID | str | None,
    timeout: httpx.Timeout = ANALYSIS_TIMEOUT,
    json: dict | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
    data: dict[str, str] | None = None,
) -> dict:
    """Forward one request (JSON or multipart) and return the parsed JSON body.

    Same credential rules as :func:`proxy_post` (``forward_headers`` + a fresh
    cell token for an in-cell peer); failures are mapped by
    :func:`map_upstream_error` instead of echoing the upstream body. A transport
    failure is 502, a timeout 504."""
    url = _service_url(path, discovery_name)
    headers = forward_headers(effective_tenant, discovery_name=discovery_name)
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=False, event_hooks=_CELL_HOOKS
    ) as client:
        try:
            resp = await send_as_tenant(
                _effective(effective_tenant),
                client.request(method, url, json=json, files=files, data=data, headers=headers),
            )
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail=f"{discovery_name} timed out") from exc
        except httpx.TransportError as exc:
            raise HTTPException(status_code=502, detail=f"{discovery_name} unreachable") from exc
    if resp.status_code >= 400:
        raise map_upstream_error(resp, discovery_name)
    try:
        body = resp.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502, detail=f"{discovery_name} returned a non-JSON body"
        ) from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=502, detail=f"{discovery_name} returned a non-object body")
    return body
