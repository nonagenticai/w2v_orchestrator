"""HTTP client for YouTube Upload Service (port 6315)."""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable, Iterator
from contextvars import ContextVar
from typing import Literal, cast
from uuid import UUID

import httpx
from w2v_common.auth.cell_jwt import (
    CELL_TOKEN_HEADER,
    cell_audience_for_url,
    cell_signer_configured,
    mint_service_cell_token,
)
from w2v_common.auth.identity_context import ACTOR_ID_HEADER, get_tenant_id_from_context
from w2v_common.clients.base import raise_for_status_with_body

from src.auth.cell_tenant import send_as_tenant
from src.auth.tenant_scope import tenant_scope_key
from src.clients.base import LicensedServiceClient

#: Owner tenant passed EXPLICITLY by a request-originated caller, bound only for
#: the duration of one client method call. It takes precedence over the ambient
#: identity ContextVar. It exists because ``BaseServiceClient._request`` has no
#: per-call ``headers`` argument, so the value has to reach the event hook (the
#: one place that stamps the header) some other way; unlike the ambient var it
#: is set from an argument the route supplies, never from inbound-request state.
_explicit_owner_tenant: ContextVar[str | None] = ContextVar(
    "youtube_upload_explicit_owner_tenant", default=None
)


#: Set for the duration of an OPERATOR call (platform channel, per-tenant policy):
#: those youtube_upload routes take the SA bearer and NO owner tenant, so the hook
#: must not stamp the ambient tenant either — an operator's own active tenant
#: leaking onto ``/api/admin/*`` would read upstream as "act for this tenant".
_operator_lane: ContextVar[bool] = ContextVar("youtube_upload_operator_lane", default=False)


@contextlib.contextmanager
def _no_owner() -> Iterator[None]:
    """Suppress ``X-Owner-Tenant-Id`` (explicit AND ambient) for the wrapped call."""
    token = _operator_lane.set(True)
    try:
        yield
    finally:
        _operator_lane.reset(token)


@contextlib.contextmanager
def _explicit_owner(owner_tenant_id: str | None) -> Iterator[None]:
    """Bind ``owner_tenant_id`` for the wrapped call; a no-op when it is ``None``."""
    if owner_tenant_id is None:
        yield
        return
    token = _explicit_owner_tenant.set(owner_tenant_id)
    try:
        yield
    finally:
        _explicit_owner_tenant.reset(token)


async def _as_owner[T](owner_tenant_id: str | None, send: Awaitable[T]) -> T:
    """Await ``send`` acting for ``owner_tenant_id`` (P7 + the legacy owner header).

    Binds the explicit owner for :func:`_owner_tenant_event_hook` (legacy
    ``X-Owner-Tenant-Id``) AND as the identity tenant of the send, so the
    ``X-W2V-Cell-Token`` that ``identity_event_hook`` mints names the SAME
    tenant — never the ambient one (a scope-switched ``super_admin``'s claim).
    Once youtube_upload reads the owner from the cell ``tenant`` claim, a
    token minted from the ambient tenant would make it read the wrong tenant's
    jobs and grant. ``None`` / an unusable owner: ambient, exactly as before.
    """
    owner = tenant_scope_key(owner_tenant_id) if owner_tenant_id is not None else None
    with _explicit_owner(owner_tenant_id):
        return await send_as_tenant(UUID(owner) if owner else None, send)


def _operator_cell_token(request: httpx.Request) -> None:
    """Replace the cell token on an OPERATOR call with a ``scope=service`` one.

    ``/api/admin/*`` on youtube_upload acts for NO tenant (``require_operator_sa``
    refuses an owner tenant). A tenant-scope token minted from the operator's
    own ambient tenant would read upstream as "act for this tenant" — the very
    escalation ``_no_owner`` exists to prevent. So the tenant token
    ``identity_event_hook`` stamped is dropped and a tenant-less service token
    for this callee is minted instead (``iss`` = this orchestrator). This is a
    deliberate, route-scoped exception to w2v_common's "a bound tenant never
    travels as the service" rule (``service_cell_token_hook`` refuses when a
    tenant is bound): the super_admin gate is the orchestrator route's, and the
    receiver must opt the operator routes in with
    ``require_cell_service_caller({"w2v-orchestrator"})``.
    """
    if CELL_TOKEN_HEADER in request.headers:
        del request.headers[CELL_TOKEN_HEADER]
    aud = cell_audience_for_url(str(request.url))
    if aud is None or not cell_signer_configured():
        return
    request.headers[CELL_TOKEN_HEADER] = mint_service_cell_token(aud)


async def _owner_tenant_event_hook(request: httpx.Request) -> None:
    """Stamp the REAL caller tenant as ``X-Owner-Tenant-Id``.

    The youtube_upload_service authorizes the orchestrator's service-account
    license token (carried in ``X-Tenant-Id: Bearer <SA>``), so its introspected
    tenant is the SAME for every real tenant — scoping upload jobs by it leaked
    every tenant's jobs to every other. The service scopes jobs by this header
    instead; only the SA-authenticated orchestrator can reach those routes, so
    the header is trusted (same model as ``X-Subject-Id`` / ``X-License-Id``).

    An explicitly-passed ``owner_tenant_id`` wins over the ambient ContextVar.
    Every request-originated caller now passes one (the HTTP routes already have
    the caller tenant in scope as ``TenantIdDep``), because relying on the
    ambient var is precisely what produced unreadable rows: when nothing is
    bound this hook no-ops, the service falls back to the channel tenant or a
    sentinel, and the resulting ``upload_jobs`` row has an ``owner_tenant_id``
    that can never match a real caller — measured 2026-08-20 on 35 of 48 prod
    rows, every one of which now 404s forever.

    The no-op is deliberately KEPT rather than made fail-closed: background
    tasks legitimately run with no bound tenant, and hard-failing them is a
    separate product decision.
    """
    # w2v-common 0.67.0: `get_tenant_id_from_context()` now returns
    # `uuid.UUID | None`, so the header value must be rendered explicitly.
    # `tenant_scope_key` also refuses the nil uuid BY VALUE: stamping
    # `X-Owner-Tenant-Id: 00000000-...` is precisely the "sentinel that can
    # never match a real caller" described above, and it would reinstate the
    # unreadable-rows bug in a form the existing `if owner:` guard cannot see,
    # because the nil uuid is truthy.
    if _operator_lane.get():
        request.headers.pop("X-Owner-Tenant-Id", None)
        _operator_cell_token(request)
        return
    owner = tenant_scope_key(_explicit_owner_tenant.get() or get_tenant_id_from_context())
    if owner:
        request.headers["X-Owner-Tenant-Id"] = owner


def _actor_headers(actor_id: str | None) -> dict[str, str] | None:
    """``X-Actor-Id`` naming the authenticated human behind a config write.

    youtube_upload records ``updated_by`` from this header (else the SA subject).
    The route passes the SAME subject it writes to ``admin_audit_log``. Set
    per-request, it wins over the ambient value ``identity_event_hook`` would
    stamp (that hook only fills an ABSENT header), so an inbound client-supplied
    ``X-Actor-Id`` can never be what reaches upstream on these writes.
    """
    return {ACTOR_ID_HEADER: actor_id} if actor_id else None


class YouTubeUploadClient(LicensedServiceClient):
    """Client for the YouTube Upload microservice.

    Extends ``LicensedServiceClient`` (not bare ``BaseServiceClient``) so the
    orchestrator's service-account ``DOWNSTREAM_LICENSE_TOKEN`` is baked in and
    forwarded as ``X-Tenant-Id: Bearer <token>``. The upload service enforces
    license-authz on ``/api/upload*``; browser-originated publish requests reach
    the orchestrator with a Keycloak JWT that carries no license token, so the
    service-account fallback is what authenticates the downstream call.
    """

    def __init__(
        self,
        base_url: str | None = None,
        license_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            service_name="youtube_upload",
            base_url=base_url,
            license_token=license_token,
            timeout=timeout,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Append the owner-tenant hook to the shared client once.

        ``BaseServiceClient._get_client`` builds the client with the canonical
        ``identity_event_hook`` / ``trace_event_hook``; we add ours so the real
        caller tenant rides every youtube-upload call.
        """
        client = await super()._get_client()
        hooks = client.event_hooks.setdefault("request", [])
        if _owner_tenant_event_hook not in hooks:
            # httpx types the request-hook list broadly (element type erases to
            # the untyped callable); our concrete coroutine fn satisfies it at
            # runtime (it's how identity_event_hook is registered too), so cast
            # to its exact request-hook signature to append without widening.
            hooks.append(
                cast("Callable[[httpx.Request], Awaitable[None]]", _owner_tenant_event_hook)
            )
        return client

    async def auth_call(
        self, method: Literal["GET", "POST", "DELETE"], action: str, owner_tenant_id: str
    ) -> httpx.Response:
        """Issue ``<method> /api/auth/<action>`` on the SA + owner-tenant lane; return it raw.

        The OAuth grant routes (connect / status / disconnect) belong to the
        tenant the orchestrator resolved from w2v MEMBERSHIP, not to the SA
        token's tenant, so ``owner_tenant_id`` is mandatory here — there is no
        ambient fallback on this lane. The response is returned unraised: the
        proxy route mirrors the upstream status + body (a 404 "no connection"
        is an answer, not a fault) and owns the 5xx -> 502 mapping. Redirects
        are never followed (``follow_redirects=False`` on the shared client).
        """
        client = await self._get_client()
        return await _as_owner(owner_tenant_id, client.request(method, f"/api/auth/{action}"))

    async def settings_call(
        self,
        method: Literal["GET", "PUT", "DELETE"],
        path: str,
        owner_tenant_id: str,
        json: dict[str, object] | None = None,
        *,
        actor_id: str | None = None,
    ) -> httpx.Response:
        """Issue ``<method> /api/settings<path>`` for ``owner_tenant_id``; return it raw.

        Same lane and contract as :meth:`auth_call`: the owner tenant is mandatory
        (membership-resolved by the route), the response is returned unraised so the
        route owns the status mapping, and NOTHING is logged here — the PUT body
        carries a tenant's OAuth ``client_secret`` and ``_request`` would log a 4xx
        body that echoes it.

        ``actor_id`` (writes): see :func:`_actor_headers`.
        """
        client = await self._get_client()
        return await _as_owner(
            owner_tenant_id,
            client.request(
                method, f"/api/settings{path}", json=json, headers=_actor_headers(actor_id)
            ),
        )

    async def operator_call(
        self,
        method: Literal["GET", "PUT"],
        path: str,
        json: dict[str, object] | None = None,
        *,
        actor_id: str | None = None,
    ) -> httpx.Response:
        """Issue ``<method> /api/admin<path>`` on the SA lane with NO owner tenant.

        Platform-wide operator config (the platform channel, a tenant's permission
        to use it). Role gating (super_admin) is the orchestrator route's; upstream
        only checks the SA bearer. Returned raw and unlogged, like
        :meth:`settings_call` (the PUT may carry the platform refresh token).
        ``actor_id`` (writes): see :func:`_actor_headers`.
        """
        client = await self._get_client()
        with _no_owner():
            return await client.request(
                method, f"/api/admin{path}", json=json, headers=_actor_headers(actor_id)
            )

    async def start_upload(
        self, payload: dict[str, object], owner_tenant_id: str | None = None
    ) -> dict[str, object]:
        """Start a YouTube upload job.

        ``owner_tenant_id``: the caller tenant that must own the created row.
        Omit it only from background/unauth callers — see ``_owner_tenant_event_hook``.
        """
        return await _as_owner(owner_tenant_id, self._request("POST", "/api/upload", json=payload))

    async def get_upload_job(
        self, job_id: str, owner_tenant_id: str | None = None
    ) -> dict[str, object]:
        """Get YouTube upload job status.

        A 404 must NOT log at ERROR *here*: the generic ``_request`` logs every 4xx
        at ERROR before raising, so a long-lived poller spams the orchestrator's
        logs (observed live, ~1 ERROR / 30s, indefinitely). The route — not the
        client — owns the 404 log: ``routes/youtube.get_upload_job`` emits a
        ``log.warning`` with the job_id + tenant before mapping it to a 404. (There
        is NO TTL or prune for ``upload_jobs``; a 404 usually means the row's
        ``owner_tenant_id`` cannot match the caller.) So issue the GET
        directly: a 404 raises a clean ``HTTPStatusError`` quietly; any OTHER status goes
        through ``raise_for_status_with_body`` which keeps the ERROR log for genuine
        faults. The shared client from ``_get_client`` still carries the SA license
        token + identity/owner-tenant hooks, so auth is unchanged.
        """
        client = await self._get_client()
        response = await _as_owner(owner_tenant_id, client.get(f"/api/upload/jobs/{job_id}"))
        if response.status_code == httpx.codes.NOT_FOUND:
            raise httpx.HTTPStatusError(
                f"youtube_upload upload job {job_id} not found",
                request=response.request,
                response=response,
            )
        raise_for_status_with_body(response, "youtube_upload")
        return cast("dict[str, object]", response.json())

    async def retract_upload_job(
        self, job_id: str, owner_tenant_id: str | None = None
    ) -> dict[str, object]:
        """Set a published upload job's YouTube video to private (reduce-only, reversible).

        Issued directly (like :meth:`get_upload_job`) so the route can map the
        upstream 404 (no such job for this owner) and 409 (job never published)
        as domain answers instead of the generic ``_request`` ERROR log + raise.
        """
        client = await self._get_client()
        response = await _as_owner(
            owner_tenant_id, client.post(f"/api/upload/jobs/{job_id}/retract")
        )
        if response.status_code in (httpx.codes.NOT_FOUND, httpx.codes.CONFLICT):
            raise httpx.HTTPStatusError(
                f"youtube_upload retract of job {job_id} returned {response.status_code}",
                request=response.request,
                response=response,
            )
        raise_for_status_with_body(response, "youtube_upload")
        return cast("dict[str, object]", response.json())

    async def set_upload_job_privacy(
        self, job_id: str, privacy_status: str, owner_tenant_id: str | None = None
    ) -> dict[str, object]:
        """Raise a published upload job's YouTube video to unlisted/public (governed release).

        Issued directly like :meth:`retract_upload_job` so the route can map the
        upstream 404 (no such job for this owner) and 409 (job never published) as
        domain answers.
        """
        client = await self._get_client()
        response = await _as_owner(
            owner_tenant_id,
            client.post(
                f"/api/upload/jobs/{job_id}/privacy", json={"privacy_status": privacy_status}
            ),
        )
        if response.status_code in (httpx.codes.NOT_FOUND, httpx.codes.CONFLICT):
            raise httpx.HTTPStatusError(
                f"youtube_upload privacy change of job {job_id} returned {response.status_code}",
                request=response.request,
                response=response,
            )
        raise_for_status_with_body(response, "youtube_upload")
        return cast("dict[str, object]", response.json())

    async def list_upload_jobs(
        self, limit: int = 50, offset: int = 0, owner_tenant_id: str | None = None
    ) -> dict[str, object]:
        """List YouTube upload jobs owned by ``owner_tenant_id`` (ambient when omitted)."""
        return await _as_owner(
            owner_tenant_id,
            self._request("GET", "/api/upload/jobs", params={"limit": limit, "offset": offset}),
        )
