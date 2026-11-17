"""Licenses service client (admin tier Phase C-2).

Ports fr's ``licenses.ts``: list apps + plans, mint a license for a subject, and
regenerate its raw token. The licenses service authorizes from the caller token's
``realm_access.roles`` (the w2v-backend SA holds ``license-reader`` +
``license-manager`` — gitops #1035) and requires an ``X-Tenant-Id`` header on
protected routes. Auth is a Keycloak **client-credentials Bearer** (NOT the
service-account license token), so this does not subclass ``LicensedServiceClient``.

A raw license token is returned ONLY by ``regenerate_token`` / ``create_token``
and is never persisted by licences — capture it at mint time or it is gone. The
orchestrator stores ONE per tenant, sealed (``src/services/tenant_licence.py``),
and verifies entitlement by ``introspect``-ing it.

The tenant's w2v licence is TENANT-level, held by the service subject
``svc:<tenant-uuid>:w2v`` — never by a person. Licences mints that holder only
through ``ensure_tenant_bundle`` (``POST /internal/tenants/{id}/bundles/w2v/ensure``,
w2v being the tenant's owner and a ``bundle_ensurer`` of ``w2v``, licences
migrations 050/052). That is the ONE internal route here, and only the in-cluster
operator lanes (super-admin provision, self-serve auto-issue) call it: an
off-cluster ingress refuses internal routes (400 ``missing_app_key``), and the
off-cluster partner lane never mints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

import httpx
import w2v_common.auth.middleware as _auth_middleware

from src.clients.schemas.licenses import (
    ActiveLicenseView,
    AppView,
    BundleEnsureView,
    LicenseView,
    PlanView,
    TenantAppByoModeSetView,
    TenantAppByoOverrideView,
    TenantByoModeView,
)
from src.clients.service_auth import TokenProvider

_TIMEOUT_SECONDS = 15.0
#: The bundle ensure's own read budget. Since licences #275 (app_requires) the
#: ensure also provisions the bundle's DEPENDENCY licences (workers,
#: inference-router, kag, mlx-*) in the same transaction: measured live
#: 2026-10-15 at 15.8 s server-side, so the 15 s default above timed out on a
#: request licences then completed. Configurable per deployment via
#: ``W2V_LICENCES_ENSURE_TIMEOUT_SECONDS`` (``Settings.licences_ensure_timeout_seconds``).
ENSURE_TIMEOUT_SECONDS = 90.0


class LicensesClientError(RuntimeError):
    """A licenses service call returned an unexpected status.

    Carries the upstream ``status_code`` as a first-class attribute: callers must
    be able to tell an AUTHZ rejection (401/403 — this service account is not
    permitted on that route, a permanent misconfiguration) apart from a transient
    5xx. The license gate branches on exactly that (see
    ``src.services.license_status``), and string-matching the message for a code
    is how that distinction silently rots.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        detail: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        #: licences' machine code (``detail.error``) when it sent one — e.g.
        #: ``tenant_not_accessible``. Never parsed out of ``message``.
        self.code = code
        #: licences' whole structured ``detail`` object when it sent one (e.g. the
        #: ``required_by`` rows of a 409 ``dependency_still_required``).
        self.detail = detail


class TenantNotOnboardedError(LicensesClientError):
    """403 ``tenant_not_accessible``: the tenant is not onboarded to w2v.

    Since licences migration 050 a tenant belongs to the app that registered it,
    and w2v may read or mint on another app's tenant only once that owner granted
    w2v access (by bundle ensure or ``PUT /tenants/{id}/grants/w2v``). This refusal
    means "no grant yet" — an ANSWER about the tenant, not a misconfiguration of
    this service account and not an outage. Callers treat the tenant as not
    onboarded to w2v (no entitlement), never as a crash.
    """


class LicencesAccessRefusedError(LicensesClientError):
    """403 from licences' ownership / issuer rules that IS a configuration fault:
    ``unknown_api_client`` (w2v-backend is not registered as an owning app),
    ``app_not_issuable_by_caller`` (w2v may not issue that app), or
    ``tenant_owner_only`` (only the tenant's owner may do this)."""


class LicencesTenantOwnedByOtherAppError(LicensesClientError):
    """409 ``tenant_owned_by_other_app`` on ``PUT /tenants/{id}``: the id is
    registered to a different app and w2v holds no grant on it. Never retried."""


class LicencesTenantNameTakenError(LicensesClientError):
    """409 ``tenant_name_taken`` on ``PUT /tenants/{id}``: w2v already owns a
    DIFFERENT licences tenant with this name (names are unique per owning app)."""


#: 403 ``detail.error`` codes licences emits for the ownership / issuer rules.
TENANT_NOT_ACCESSIBLE = "tenant_not_accessible"
_ACCESS_REFUSED_CODES = frozenset(
    {"unknown_api_client", "app_not_issuable_by_caller", "tenant_owner_only"}
)
_CONFLICT_ERRORS: dict[str, type[LicensesClientError]] = {
    "tenant_owned_by_other_app": LicencesTenantOwnedByOtherAppError,
    "tenant_name_taken": LicencesTenantNameTakenError,
}


def _error_detail(resp: httpx.Response) -> dict[str, object] | None:
    """licences' structured ``{"detail": {"error": ..., "message": ..., ...}}``."""
    try:
        body = resp.json()
    except ValueError:
        return None
    detail = body.get("detail") if isinstance(body, dict) else None
    return cast(dict[str, object], detail) if isinstance(detail, dict) else None


def _error_code(resp: httpx.Response) -> str | None:
    """``detail.error`` from licences' ``{"detail": {"error": ..., "message": ...}}``."""
    detail = _error_detail(resp)
    code = detail.get("error") if detail is not None else None
    return code if isinstance(code, str) else None


def _failure(op: str, resp: httpx.Response) -> LicensesClientError:
    """The most specific error for a failed licences call (the message keeps the
    long-standing ``"<op> returned <status>: <body excerpt>"`` shape)."""
    message = f"{op} returned {resp.status_code}: {resp.text[:200]}"
    code = _error_code(resp)
    cls: type[LicensesClientError] = LicensesClientError
    if resp.status_code == 403 and code == TENANT_NOT_ACCESSIBLE:
        cls = TenantNotOnboardedError
    elif resp.status_code == 403 and code in _ACCESS_REFUSED_CODES:
        cls = LicencesAccessRefusedError
    elif resp.status_code == 409 and code is not None:
        cls = _CONFLICT_ERRORS.get(code, LicensesClientError)
    return cls(message, status_code=resp.status_code, code=code, detail=_error_detail(resp))


def _is_ensure_result(resp: httpx.Response) -> bool:
    """A 409 that carries a ``BundleEnsureResponse`` (some apps refused) rather
    than a ``{"detail": {...}}`` refusal of the whole call."""
    try:
        body = resp.json()
    except ValueError:
        return False
    return isinstance(body, dict) and isinstance(body.get("refused"), list)


@dataclass(frozen=True, slots=True)
class RegisteredTenant:
    """The licences tenant after ``PUT /tenants/{id}``; ``created`` only on a 201
    (w2v now owns it). 200 = already w2v's, or another app's on which w2v holds a
    grant — returned unchanged, never renamed."""

    tenant_id: str
    name: str
    created: bool


@dataclass(frozen=True, slots=True)
class MintedToken:
    """A freshly minted raw token (shown once) and its own expiry (None = perpetual)."""

    token: str
    expires_at: str | None

    def __repr__(self) -> str:  # never render the token
        return f"MintedToken(expires_at={self.expires_at!r})"


class LicensesClient:
    def __init__(
        self,
        *,
        authenticator: TokenProvider,
        base_url: str,
        client: httpx.AsyncClient | None = None,
        introspect_url: str | None = None,
        ensure_timeout_seconds: float = ENSURE_TIMEOUT_SECONDS,
    ) -> None:
        self._auth = authenticator
        self._base = base_url.rstrip("/")
        self._client = client
        self._introspect_url = introspect_url
        self._ensure_timeout = ensure_timeout_seconds

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        tenant_id: UUID | str | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """``timeout`` overrides the default budget for this one call (only the
        bundle ensure needs it); None keeps ``_TIMEOUT_SECONDS`` (or an injected
        client's own timeout)."""
        token = await self._auth.token()
        headers = {"Authorization": f"Bearer {token}"}
        if tenant_id is not None:
            headers["X-Tenant-Id"] = str(tenant_id)
        url = f"{self._base}{path}"
        if self._client is not None:
            if timeout is None:
                return await self._client.request(method, url, json=json, headers=headers)
            return await self._client.request(
                method, url, json=json, headers=headers, timeout=timeout
            )
        async with httpx.AsyncClient(timeout=timeout or _TIMEOUT_SECONDS) as client:
            return await client.request(method, url, json=json, headers=headers)

    async def list_apps(self, *, tenant_id: UUID | str) -> list[AppView]:
        resp = await self._request("GET", "/apps", tenant_id=tenant_id)
        if resp.status_code != 200:
            raise _failure("list_apps", resp)
        apps = resp.json()
        return [AppView.model_validate(a) for a in apps] if isinstance(apps, list) else []

    async def list_plans(self, app_id: str, *, tenant_id: UUID | str) -> list[PlanView]:
        resp = await self._request("GET", f"/apps/{app_id}/plans", tenant_id=tenant_id)
        if resp.status_code != 200:
            raise _failure("list_plans", resp)
        plans = resp.json()
        return [PlanView.model_validate(p) for p in plans] if isinstance(plans, list) else []

    async def get_active_license(
        self, *, tenant_id: UUID | str, app_key: str
    ) -> ActiveLicenseView | None:
        """Resolve the currently-active license for ``(tenant_id, app_key)`` via the
        licenses JWT-mode ``GET /licenses/active`` route (the same endpoint fr-backend
        uses). Returns the introspect detail on 200 (its license id is
        ``license_id``, NOT ``id``), ``None`` on 404 (no active license), and an
        ``ActiveLicenseView(conflict=True)`` sentinel on 409 (multiple active — a
        data conflict the caller treats as entitled-with-unknown-expiry)."""
        resp = await self._request(
            "GET",
            f"/licenses/active?tenant_id={tenant_id}&app_key={app_key}",
            tenant_id=tenant_id,
        )
        if resp.status_code == 200:
            detail = resp.json()
            return ActiveLicenseView.model_validate(detail) if isinstance(detail, dict) else None
        if resp.status_code == 404:
            return None
        if resp.status_code == 409:
            return ActiveLicenseView(conflict=True)
        raise _failure("get_active_license", resp)

    async def get_tenant_byo_mode(
        self, *, tenant_id: UUID | str, app_key: str
    ) -> TenantByoModeView:
        """The tenant's BYO posture for ``app_key`` (per-app override, else tenant-wide).

        ``on_unresolved=error``: a licences DB fault answers 503 (raised here as
        :class:`LicensesClientError`) instead of a fabricated 200 the caller could
        not tell apart from a real verdict. Every non-200 raises; the caller
        decides what a 404 (tenant not in the registry) means.
        """
        resp = await self._request(
            "GET",
            f"/tenants/{tenant_id}/byo-mode?app={app_key}&on_unresolved=error",
            tenant_id=tenant_id,
        )
        if resp.status_code != 200:
            raise _failure("get_tenant_byo_mode", resp)
        return TenantByoModeView.model_validate(resp.json())

    async def get_tenant_app_byo_override(
        self, *, tenant_id: UUID | str, app_key: str
    ) -> TenantAppByoOverrideView | None:
        """The per-app BYO OVERRIDE row for ``(tenant_id, app_key)``, or None on
        404 (no override — the tenant-wide value applies; licences answers the
        same 404 for an unknown app). NOT the effective posture: use
        :meth:`get_tenant_byo_mode` for that. Needs ``license-reader``."""
        resp = await self._request(
            "GET", f"/tenants/{tenant_id}/apps/{app_key}/byo-mode", tenant_id=tenant_id
        )
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise _failure("get_tenant_app_byo_override", resp)
        return TenantAppByoOverrideView.model_validate(resp.json())

    async def set_tenant_app_byo_mode(
        self, *, tenant_id: UUID | str, app_key: str, byo_mode: str
    ) -> TenantAppByoModeSetView:
        """``PUT /tenants/{id}/apps/{app_key}/byo-mode`` ``{byo_mode}``: write the
        PER-APP override, so the tenant's other apps keep their posture. Needs
        ``license-admin`` or ``license-manager`` AND w2v issuing ``app_key`` (403
        ``app_not_administrable_by_caller`` otherwise). Every non-200 raises."""
        resp = await self._request(
            "PUT",
            f"/tenants/{tenant_id}/apps/{app_key}/byo-mode",
            json={"byo_mode": byo_mode},
            tenant_id=tenant_id,
        )
        if resp.status_code != 200:
            raise _failure("set_tenant_app_byo_mode", resp)
        return TenantAppByoModeSetView.model_validate(resp.json())

    async def list_user_licenses(
        self, subject_id: UUID | str, *, tenant_id: UUID | str
    ) -> list[LicenseView]:
        """List the licenses already minted for ``subject_id`` (the GET counterpart
        of ``create_license``'s POST). Used to make provisioning idempotent — a
        retry reuses an existing non-revoked license instead of double-minting.
        Each row carries ``id`` + ``revoked_at`` (revocation is a timestamp, not a
        bare ``revoked`` bool)."""
        resp = await self._request("GET", f"/licenses/user/{subject_id}", tenant_id=tenant_id)
        if resp.status_code != 200:
            raise _failure("list_user_licenses", resp)
        rows = resp.json()
        return [LicenseView.model_validate(r) for r in rows] if isinstance(rows, list) else []

    async def create_license(
        self,
        *,
        subject_id: UUID | str,
        app_id: str,
        plan_id: str,
        tenant_id: UUID | str,
        metadata: dict[str, Any] | None = None,
    ) -> LicenseView:
        """Mint a license for ``subject_id`` under ``tenant_id``. Returns the license
        row (carries ``id``); the raw token comes from ``regenerate_token``."""
        body: dict[str, Any] = {"app_id": app_id, "plan_id": plan_id}
        if metadata is not None:
            body["metadata"] = metadata
        resp = await self._request(
            "POST", f"/licenses/user/{subject_id}", json=body, tenant_id=tenant_id
        )
        if resp.status_code not in (200, 201):
            raise _failure("create_license", resp)
        return LicenseView.model_validate(resp.json())

    async def regenerate_token(self, license_id: str, *, tenant_id: UUID | str) -> str:
        """Mint the raw license token (returned ONCE, never persisted)."""
        resp = await self._request(
            "POST", f"/licenses/{license_id}/regenerate-token", json={}, tenant_id=tenant_id
        )
        if resp.status_code not in (200, 201):
            raise _failure("regenerate_token", resp)
        token = resp.json().get("token")
        if not token:
            raise LicensesClientError("regenerate_token response missing token")
        return str(token)

    async def create_token(self, license_id: UUID | str, *, tenant_id: UUID | str) -> MintedToken:
        """Mint an ADDITIONAL raw token for a licence (``POST /licenses/{id}/tokens``,
        ``token_type=api``): unlike ``regenerate_token`` it revokes nothing, so the
        tenant's stored, dedicated token never kills a token someone else holds.
        Needs ``license-manager``. Returned ONCE — the caller seals and stores it."""
        resp = await self._request(
            "POST", f"/licenses/{license_id}/tokens?token_type=api", tenant_id=tenant_id
        )
        if resp.status_code not in (200, 201):
            raise _failure("create_token", resp)
        body = resp.json()
        token = body.get("token") if isinstance(body, dict) else None
        if not token:
            raise LicensesClientError("create_token response missing token")
        return MintedToken(token=str(token), expires_at=body.get("expires_at"))

    async def ensure_tenant_bundle(
        self, tenant_id: UUID | str, bundle_key: str, *, plans: list[str]
    ) -> BundleEnsureView:
        """``POST /internal/tenants/{id}/bundles/{bundle}/ensure`` ``{plans}``: make
        the tenant hold ONE tenant-level licence per app of ``bundle_key``, held by
        ``svc:<tenant-uuid>:<app>`` (idempotent — ``exists`` / ``adopt`` / ``insert``).
        ``plans`` is the ordered plan-code preference for an inserted licence.

        200, and 409 with a ``refused`` list (the other apps were still ensured),
        both return the parsed result; the caller decides what a refusal means.
        Anything else raises (:class:`LicencesAccessRefusedError` for
        ``tenant_owner_only`` / ``unknown_api_client``, else :class:`LicensesClientError`).

        Runs on its own, longer timeout (``ensure_timeout_seconds``): the ensure
        also provisions the bundle's dependency licences. An ``httpx.TimeoutException``
        propagates — the caller re-reads to learn whether licences finished."""
        resp = await self._request(
            "POST",
            f"/internal/tenants/{tenant_id}/bundles/{bundle_key}/ensure",
            json={"plans": plans},
            tenant_id=tenant_id,
            timeout=self._ensure_timeout,
        )
        if resp.status_code == 200 or (resp.status_code == 409 and _is_ensure_result(resp)):
            return BundleEnsureView.model_validate(resp.json())
        raise _failure("ensure_tenant_bundle", resp)

    async def register_tenant(self, tenant_id: UUID | str, *, name: str) -> RegisteredTenant:
        """``PUT /tenants/{id}`` ``{name}``: register ``tenant_id`` in licences with
        w2v as its owning app (licences derives the owner from this service
        account's client — it is never sent). Idempotent: 201 creates, 200 returns
        the existing row unchanged (w2v's own tenant, or one w2v holds a grant on).

        Raises :class:`LicencesTenantOwnedByOtherAppError` /
        :class:`LicencesTenantNameTakenError` (409), :class:`LicencesAccessRefusedError`
        (403 ``unknown_api_client``), or :class:`LicensesClientError` for anything
        else (a 403 role refusal carries ``code=None``)."""
        resp = await self._request("PUT", f"/tenants/{tenant_id}", json={"name": name})
        if resp.status_code not in (200, 201):
            raise _failure("register_tenant", resp)
        body = resp.json()
        stored = body.get("name") if isinstance(body, dict) else None
        return RegisteredTenant(
            tenant_id=str(tenant_id),
            name=str(stored or name),
            created=resp.status_code == 201,
        )

    # ─── early-access removal (``src/services/signup/removal.py``) ────────

    async def list_tenant_licenses(self, *, tenant_id: UUID | str) -> list[LicenseView]:
        """``GET /licenses`` with ``X-Tenant-Id``: EVERY licence of the tenant (all
        apps, all holders, revoked ones included). Needs ``license-reader``."""
        resp = await self._request("GET", "/licenses", tenant_id=tenant_id)
        if resp.status_code != 200:
            raise _failure("list_tenant_licenses", resp)
        rows = resp.json()
        return [LicenseView.model_validate(r) for r in rows] if isinstance(rows, list) else []

    async def revoke_license(self, license_id: str, *, tenant_id: UUID | str) -> list[str]:
        """``POST /license/revoke`` ``{license_id}``: revoke the licence and every
        token of it (idempotent on licences' side). Needs ``license-manager`` and
        w2v issuing the app — or owning the tenant whose bundle minted it
        (licences ``_bundle_owner_may_revoke``). ``{"revoked": false}`` (licences'
        "License not found" for an unknown id OR another tenant's) raises, like
        every non-200.

        Returns the ids of the DEPENDENCY licences licences released in the same
        transaction (revoking the last licence that required them), ``[]`` when
        the answer names none (an older licences, or nothing orphaned). Revoking a
        dependency licence itself answers 409 ``dependency_still_required`` while
        another licence still needs it."""
        resp = await self._request(
            "POST", "/license/revoke", json={"license_id": license_id}, tenant_id=tenant_id
        )
        if resp.status_code != 200:
            raise _failure("revoke_license", resp)
        body = resp.json()
        if not (isinstance(body, dict) and body.get("revoked") is True):
            message = body.get("message") if isinstance(body, dict) else None
            raise LicensesClientError(
                f"revoke_license {license_id}: not revoked ({message or 'no message'})",
                status_code=200,
                code="licence_not_revoked",
            )
        released = body.get("released_dependency_ids", body.get("released_dependencies"))
        return [str(r) for r in released] if isinstance(released, list) else []

    async def archive_tenant(self, tenant_id: UUID | str) -> str | None:
        """``PATCH /tenants/{id}/archive`` ``{"archived": true}``: retire the tenant
        in licences (its licences stop authorizing; idempotent — re-archiving
        keeps the original timestamp). Needs ``license-manager`` on a tenant w2v
        OWNS. Returns ``archived_at``; every non-200 raises."""
        resp = await self._request(
            "PATCH", f"/tenants/{tenant_id}/archive", json={"archived": True}
        )
        if resp.status_code != 200:
            raise _failure("archive_tenant", resp)
        body = resp.json()
        archived_at = body.get("archived_at") if isinstance(body, dict) else None
        return str(archived_at) if archived_at else None

    async def introspect(self, token: str) -> ActiveLicenseView:
        """``POST {LICENSE_INTROSPECT_URL}`` ``{"token": …}`` — the PUBLIC
        introspection route (the token in the body is the credential, so no
        service-account Bearer is sent). This is the one licences call an
        off-cluster deployment's ingress admits for a token. 200 → the verdict
        (``active:false`` + ``reason`` is a verdict, not an error); anything else
        raises :class:`LicensesClientError` with the status. The token never
        appears in an error message."""
        # Read at call time (w2v_common resolves it once from env/discovery).
        url = self._introspect_url or _auth_middleware.LICENSE_INTROSPECT_URL
        if self._client is not None:
            resp = await self._client.post(url, json={"token": token})
        else:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                resp = await client.post(url, json={"token": token})
        if resp.status_code != 200:
            # Status + reason phrase ONLY: the request body carried the token, and an
            # error body may echo it (an opaque urlsafe string no redaction can spot).
            raise LicensesClientError(
                f"introspect returned {resp.status_code} {resp.reason_phrase}".rstrip(),
                status_code=resp.status_code,
            )
        body = resp.json()
        if not isinstance(body, dict):
            raise LicensesClientError(
                f"introspect: expected a JSON object, got {type(body).__name__}"
            )
        return ActiveLicenseView.model_validate(body)
