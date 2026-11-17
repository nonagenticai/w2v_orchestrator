"""Base async HTTP clients for downstream microservices.

``BaseServiceClient`` is re-exported from w2v_common for backwards compatibility.

``LicensedServiceClient`` layers the orchestrator's standard wiring on top:
default the service-account license token from settings and resolve the base
URL via w2v_common discovery — so the per-service clients don't each repeat
that block.
"""

from __future__ import annotations

import ssl

import httpx
from w2v_common.auth.identity_context import identity_event_hook
from w2v_common.clients.base import (
    _MAX_RETRIES,
    _POLL_BACKOFF_MULTIPLIER,
    _POLL_MAX_INTERVAL,
    _RETRY_DELAY,
    _RETRYABLE_STATUS_CODES,
    BaseServiceClient,
)
from w2v_common.discovery import service_url

from src.clients.downstream_credential import (
    DownstreamCredentialMissingError,
    cell_identity_event_hook,
    hop_lane,
    in_cell_license_token,
    is_in_cell,
    optional_out_of_cell_license_token,
)

__all__ = [
    "_MAX_RETRIES",
    "_POLL_BACKOFF_MULTIPLIER",
    "_POLL_MAX_INTERVAL",
    "_RETRYABLE_STATUS_CODES",
    "_RETRY_DELAY",
    "BaseServiceClient",
    "DownstreamCredentialMissingError",
    "LicensedServiceClient",
]


class LicensedServiceClient(BaseServiceClient):
    """A ``BaseServiceClient`` pre-wired for the orchestrator's downstream calls.

    Every downstream service enforces license-authz (only ``/health*`` is
    exempt), so the orchestrator must present its service-account token on each
    call. Rather than repeat ``token = license_token or
    get_settings().downstream.license_token.get_secret_value()`` + ``service_url(...)`` in every
    client, subclasses just declare their ``service_name`` — and a
    ``discovery_name`` only when it differs from it (e.g. ``redesign`` and
    ``critique`` both resolve to the ``analysis`` service).

    The baked token is forwarded as ``X-Tenant-Id: Bearer <token>`` by
    ``BaseServiceClient``; per-request identity ContextVars (X-License-Id /
    X-Subject-Id) propagate via the shared ``identity_event_hook``.
    """

    def __init__(
        self,
        *,
        service_name: str,
        discovery_name: str | None = None,
        base_url: str | None = None,
        license_token: str | None = None,
        timeout: float = 30.0,
        ssl_context: ssl.SSLContext | bool = True,
        poll_interval: float = 2.0,
        max_poll_attempts: int = 150,
        licence_hop: str | None = None,
    ) -> None:
        #: Discovery name of the callee — the bounded ``destination`` label of an
        #: explicit service-scope send (``downstream_credential.service_scope``).
        self.discovery_name = discovery_name or service_name
        resolved_url = base_url or service_url(self.discovery_name)
        # P7 / R3: an explicit token is the caller's choice and is used as given.
        # Otherwise the shared DOWNSTREAM_LICENSE_TOKEN — never on an in-cell hop
        # under CELL_AUTH_MODE=enforce (the cell JWT authenticates it), and an
        # out-of-cell hop with none configured fails LOUD on first send
        # (``_get_client``), not at construction: clients are built at boot.
        self._missing_credential_for: str | None = None
        # An out-of-cell metered hop (``licence_hop``) follows ``<HOP>_AUTH_MODE``.
        # In ``tenant_licence`` mode NO shared or explicit token is baked in: every
        # request carries the bound tenant's minted licence (w2v_common lane hooks)
        # and a missing tenant / refused mint raises. ``shared`` changes nothing.
        lane = hop_lane(licence_hop) if licence_hop is not None else None
        token: str | None
        if lane is not None and lane.tenant_licence:
            token = None
        elif license_token is not None:
            token = license_token
        elif is_in_cell(resolved_url):
            token = in_cell_license_token()
        else:
            token = optional_out_of_cell_license_token()
            if token is None:
                self._missing_credential_for = service_name
        super().__init__(
            base_url=resolved_url,
            service_name=service_name,
            license_token=token,
            timeout=timeout,
            ssl_context=ssl_context,
            poll_interval=poll_interval,
            max_poll_attempts=max_poll_attempts,
            licence_lane=lane,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Refuse to send an out-of-cell call that has no credential (enforce only).

        Also swaps the base client's ``identity_event_hook`` for
        :func:`cell_identity_event_hook` (same position, so hook order is kept):
        identical unless the call EXPLICITLY opted in with
        :func:`~src.clients.downstream_credential.service_scope` — only then does
        a tenant-less in-cell hop under ``enforce`` leave with a ``scope=service``
        cell token; without it, it fails closed exactly as before. Idempotent;
        a mock client with no hook pipeline is left alone.
        """
        if self._missing_credential_for is not None:
            raise DownstreamCredentialMissingError(self._missing_credential_for)
        client = await super()._get_client()
        # Duck-typed, not ``isinstance(client, httpx.AsyncClient)``: tests swap
        # ``httpx.AsyncClient`` for a factory (isinstance would raise) or inject
        # an ``AsyncMock`` (whose ``event_hooks`` is not a dict — nothing to rewire).
        event_hooks = getattr(client, "event_hooks", None)
        if isinstance(event_hooks, dict):
            hooks = event_hooks.setdefault("request", [])
            for i, hook in enumerate(hooks):
                if hook is identity_event_hook:
                    hooks[i] = cell_identity_event_hook
        return client
