"""Which outbound hop may carry the shared ``DOWNSTREAM_LICENSE_TOKEN`` (P7 / R3).

Under ``CELL_AUTH_MODE`` ``off``/``dual`` the token is required at boot
(``DownstreamServiceSettings.license_token``) and every hop carries it exactly
as before — these helpers return it unchanged.

Under ``enforce`` the token is optional (an R3 deployment boots with env +
slots only; the shared platform licence is what the cell template retires):

* an IN-CELL hop (a w2v peer, per ``w2v_common`` ``cell_audience_for_url``) is
  authenticated by the per-request cell JWT and must NOT carry the shared
  token — :func:`in_cell_license_token` returns ``None``;
* an OUT-OF-CELL hop (webscraper, media-discovery, WM, video-compiler, tts/asr
  host services, billing) still needs a credential. With none configured it fails LOUD at call time
  with :class:`DownstreamCredentialMissingError` naming the destination —
  never an unauthenticated send, never a fallback to another identity.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from contextvars import ContextVar

import httpx
from w2v_common.auth.cell_jwt import (
    CellAuthMode,
    CellConfigError,
    cell_audience_for_url,
    cell_auth_mode,
    cell_signer_configured,
)
from w2v_common.auth.identity_context import (
    get_tenant_id_from_context,
    identity_event_hook,
    is_inbound_request,
    service_cell_token_hook,
)
from w2v_common.auth.tenant_licence_lane import TenantLicenceLane

from src.config import get_settings
from src.observability.metrics import cell_service_scope_calls_total

logger = logging.getLogger(__name__)


class CellBootConfigError(RuntimeError):
    """``CELL_AUTH_MODE=enforce`` without a usable cell signing key: refuse to boot."""


def assert_cell_boot_config() -> None:
    """Refuse to boot under ``enforce`` unless this service can mint cell JWTs.

    Under enforce an in-cell hop drops the shared token
    (:func:`in_cell_license_token` returns ``None``) and relies on the cell JWT
    alone. Without ``CELL_JWT_SIGNING_KEY`` nothing is minted, so every in-cell
    call would leave with NO credential at all and be rejected one request at a
    time. That is a deployment error, so it fails the pod at startup instead,
    naming the missing setting. off/dual: no-op (the shared token still rides).
    """
    if cell_auth_mode() is not CellAuthMode.ENFORCE:
        return
    try:
        configured = cell_signer_configured()
    except CellConfigError as exc:
        message = f"CELL_AUTH_MODE=enforce but cell signing is misconfigured: {exc}"
        logger.critical(message)
        raise CellBootConfigError(message) from exc
    if not configured:
        message = (
            "CELL_AUTH_MODE=enforce requires CELL_JWT_SIGNING_KEY (plus CELL_JWT_KID, "
            "CELL_SERVICE_NAME, CELL_DEPLOYMENT_ID): in-cell hops carry no shared "
            "token under enforce, so without a signing key they would send no "
            "credential at all. Refusing to boot."
        )
        logger.critical(message)
        raise CellBootConfigError(message)


class DownstreamCredentialMissingError(RuntimeError):
    """An out-of-cell call needs ``DOWNSTREAM_LICENSE_TOKEN`` and none is configured."""

    def __init__(self, destination: str) -> None:
        self.destination = destination
        super().__init__(
            f"outbound call to out-of-cell service {destination!r} needs a credential, but "
            "DOWNSTREAM_LICENSE_TOKEN is unset (optional only under CELL_AUTH_MODE=enforce, "
            "where in-cell hops use cell JWTs). Configure a credential for this destination "
            "or do not call it from this deployment."
        )


def is_in_cell(url: str) -> bool:
    """True when ``url`` is a w2v cell peer (it receives a cell JWT)."""
    return cell_audience_for_url(url) is not None


def in_cell_license_token() -> str | None:
    """The shared token for an in-cell hop: unchanged off/dual, ``None`` under enforce."""
    if cell_auth_mode() is CellAuthMode.ENFORCE:
        return None
    token = get_settings().downstream.license_token
    return token.get_secret_value() if token is not None else None


def optional_out_of_cell_license_token() -> str | None:
    """The shared token for an out-of-cell hop, or ``None`` if not configured.

    Callers that resolve at construction time use this and call
    :func:`require_out_of_cell_license_token` (or raise
    :class:`DownstreamCredentialMissingError`) when the request is actually sent.
    """
    token = get_settings().downstream.license_token
    return token.get_secret_value() if token is not None else None


def require_out_of_cell_license_token(destination: str) -> str:
    """The shared token for an out-of-cell hop; raises when it is not configured."""
    token = optional_out_of_cell_license_token()
    if not token:
        raise DownstreamCredentialMissingError(destination)
    return token


#: Out-of-cell licence-gated hops switched per hop by ``<HOP>_AUTH_MODE`` (w2v_common
#: :class:`~w2v_common.auth.tenant_licence_lane.TenantLicenceLane`): ``shared``
#: (default — the shared ``DOWNSTREAM_LICENSE_TOKEN``, exactly as before) or
#: ``tenant_licence`` (the run's tenant's own minted licence; never the shared
#: token, no fallback). WM keeps its own ``WM_AUTH_MODE`` inside ``WMClient``.
#: webscraper + media_discovery replaced ``kag`` (retired) — both licence-gated,
#: neither meters; their lane app key is ``w2v`` (w2v_common ``HOP_APP_KEYS``).
OUT_OF_CELL_LANE_HOPS: tuple[str, ...] = (
    "webscraper",
    "media_discovery",
    "modality_gateway",
    "billing",
)


def hop_lane(hop: str) -> TenantLicenceLane:
    """The auth lane for one out-of-cell hop, read from ``<HOP>_AUTH_MODE`` now.

    Raises ``ValueError`` on an unknown mode and ``BundleMinterNotConfiguredError``
    for ``tenant_licence`` without the ``BUNDLE_MINTER_*`` settings.
    """
    return TenantLicenceLane(hop, shared_token=optional_out_of_cell_license_token() or "")


def assert_hop_lanes_boot_config() -> None:
    """Refuse to boot on a bad ``<HOP>_AUTH_MODE`` or a tenant_licence hop without a minter.

    Clients are also built per request/stage, where the same error would only
    surface on the first metered call. Building every lane once here makes a
    typo or a missing ``BUNDLE_MINTER_*`` secret fail the pod at startup.
    """
    for hop in OUT_OF_CELL_LANE_HOPS:
        lane = hop_lane(hop)
        logger.info("out-of-cell hop %s auth mode=%s app=%s", hop, lane.mode.value, lane.app)


#: Discovery name of the destination whose call explicitly opted in to service
#: scope, bound only for the duration of that one call (:func:`service_scope`).
#: ``None`` — every hop by default — means a tenant-less in-cell hop under
#: ``enforce`` fails closed exactly as ``identity_event_hook`` decides.
_service_scope_destination: ContextVar[str | None] = ContextVar(
    "cell_service_scope_destination", default=None
)


@contextlib.contextmanager
def service_scope(destination: str) -> Iterator[None]:
    """Opt the wrapped call in to a ``scope=service`` cell token (tenant-less hops only).

    Same shape as ``youtube_upload_client._no_owner`` (a ContextVar lane bound
    for one client-method call and read by the request hook). w2v_common's rule
    is that service scope is never implicit — a tenant-scoped background job
    that lost its tenant must FAIL (``MissingTenantError``), not act as the
    service — so only a call site that names itself here gets one. ``destination``
    is the client's discovery name (the bounded label of
    ``w2v_orchestrator_cell_service_scope_calls_total``).
    """
    token = _service_scope_destination.set(destination)
    try:
        yield
    finally:
        _service_scope_destination.reset(token)


#: Audience resolved from the request URL (``None``): only ever called for an
#: in-cell destination (pre-checked), so nothing is minted for or counted against
#: an out-of-cell host.
_SERVICE_CELL_TOKEN = service_cell_token_hook(None)
#: Placeholder that walks ``identity_event_hook``'s tenant step down its
#: documented "service-credentialed, not inbound" branch. Removed before send.
_SERVICE_CREDENTIAL_PLACEHOLDER = "Bearer cell-service-scope"


def _opted_in_service_hop(request: httpx.Request) -> str | None:
    """The opted-in destination when this hop may go as the service, else ``None``.

    Explicit opt-in + enforce + no tenant + not inside an inbound request +
    in-cell + no SA bearer.
    """
    destination = _service_scope_destination.get()
    if destination is None:
        return None
    if cell_auth_mode() is not CellAuthMode.ENFORCE:
        return None
    if get_tenant_id_from_context() is not None or is_inbound_request():
        return None
    if request.headers.get("X-Tenant-Id", "").lower().startswith("bearer "):
        return None
    if cell_audience_for_url(str(request.url)) is None:
        return None
    return destination


async def cell_identity_event_hook(request: httpx.Request) -> None:
    """``identity_event_hook``, plus a ``scope=service`` token for an OPTED-IN tenant-less hop.

    Under ``enforce`` an in-cell client carries no shared licence
    (:func:`in_cell_license_token`), so a hop with NO tenant bound outside any
    inbound request fails closed in ``identity_event_hook`` with
    ``MissingTenantError``. That is deliberate (w2v_common
    ``service_cell_token_hook``: "service scope is never implicit; the tenant
    hook never downgrades a call that lost its tenant") and stays the default.

    Only a call wrapped in :func:`service_scope` — the reconciler-finalize
    ``design_job_exists`` probe, whose work genuinely is the orchestrator's —
    goes as the service: the canonical hook runs with the request marked
    service-credentialed (every non-tenant header is stamped by w2v_common), the
    marker is removed, a ``scope=service`` cell token is minted for the in-cell
    audience and the send is counted. Receivers accept it only on routes that
    opt in with ``require_cell_service_caller``.

    Every other case is ``identity_event_hook`` unchanged, byte for byte: no
    opt-in; off/dual; a bound tenant (an unvouched one is still never signed or
    laundered into the service); a tenant-less hop inside an inbound request;
    an out-of-cell destination.
    """
    destination = _opted_in_service_hop(request)
    if destination is None:
        await identity_event_hook(request)
        return
    request.headers["X-Tenant-Id"] = _SERVICE_CREDENTIAL_PLACEHOLDER
    try:
        await identity_event_hook(request)
    finally:
        del request.headers["X-Tenant-Id"]
    await _SERVICE_CELL_TOKEN(request)
    cell_service_scope_calls_total.labels(destination=destination).inc()
