"""A licences-service fake for the P9 / licence-introspection tests — only the
two routes an off-cluster ingress admits (``src/services/tenant_licence.py``):

* ``get_active_license`` (``GET /licenses/active``): ``live`` maps tenant → its
  active w2v licence ids; the answer carries ``candidate_count``. The licence gate
  reads the same route, so a tenant absent from ``live`` is also unentitled there.
* ``introspect`` (``POST /license/introspect``): tokens minted with
  :meth:`P9Licences.token` resolve to their introspection; an unknown token is
  ``active:false`` ``token_not_found``.
* ``create_token`` (provisioning's dedicated mint).
* ``register_tenant`` (``PUT /tenants/{id}``, licences migration 050): records
  ``(tenant_id, name)`` in ``registered``; ``register_error`` makes it raise.
* ``ensure_tenant_bundle`` (+ ``list_apps`` / ``list_plans``), the partner
  create's licence ensure: a tenant with exactly one ``live`` w2v licence answers
  ``exists`` with it; several → refused ``several_candidates``; none → ``insert``
  a new licence, which then shows up in ``live``. ``ensure_refuse`` maps app key
  → refusal reason; ``ensure_error`` makes the call raise. Calls are recorded in
  ``ensure_calls``. ``ensure_error_after_commit`` raises only AFTER the licence
  was written (licences finished, the client timed out first); with it set,
  ``list_user_licenses`` answers the ``svc:<tenant>:w2v`` holder's licences.
* ``not_onboarded``: tenants whose ``/licenses/active`` answers 403
  ``tenant_not_accessible`` (another app owns them, no grant to w2v).

``down`` makes every read fail like an unreachable service; ``introspect_down``
(5xx) / ``introspect_refuse=<status>`` (definite 4xx) only the introspection (the
gate's ``/licenses/active`` still answers);
``refuse=<status>`` fails every read like a definite licences/gateway 4xx refusal.
"""

from __future__ import annotations

import secrets
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI

from src.clients.licenses_client import (
    LicensesClientError,
    MintedToken,
    RegisteredTenant,
    TenantNotOnboardedError,
)
from src.clients.schemas.licenses import (
    ActiveAppRef,
    ActiveLicenseView,
    ActivePlanRef,
    BundleEnsureAppView,
    BundleEnsureRefusalView,
    BundleEnsureView,
    LicenseView,
)
from src.services.providers.crypto import Keyring


class P9Licences:
    def __init__(
        self,
        live: dict[UUID, list[UUID]] | None = None,
        *,
        down: bool = False,
        introspect_down: bool = False,
        introspect_refuse: int | None = None,
        refuse: int | None = None,
    ) -> None:
        self.live = live or {}
        #: Tenants answered with the client's legacy 409 multi-active sentinel.
        self.conflict: set[UUID] = set()
        self.tokens: dict[str, ActiveLicenseView] = {}
        self.down = down
        self.introspect_down = introspect_down
        self.introspect_refuse = introspect_refuse
        self.refuse = refuse
        self.introspect_calls = 0
        self.active_calls = 0
        self.minted: list[tuple[str, str]] = []
        self.registered: list[tuple[str, str]] = []
        self.register_error: Exception | None = None
        self.not_onboarded: set[UUID] = set()
        self.ensure_calls: list[str] = []
        self.ensure_refuse: dict[str, str] = {}
        self.ensure_error: Exception | None = None
        self.ensure_error_after_commit: Exception | None = None
        self.reread_subjects: list[str] = []

    def token(
        self,
        tenant: UUID,
        licence: UUID,
        *,
        app: str = "w2v",
        active: bool = True,
        reason: str | None = None,
        token_expires_at: str | None = None,
    ) -> str:
        """Register a token and return its raw value."""
        raw = "lic_tok_" + secrets.token_hex(16)
        self.tokens[raw] = (
            ActiveLicenseView(
                active=True,
                tenant_id=str(tenant),
                license_id=str(licence),
                app=ActiveAppRef(key=app),
                plan=ActivePlanRef(code="standard"),
                token_expires_at=token_expires_at,
            )
            if active
            else ActiveLicenseView(active=False, reason=reason or "license_inactive")
        )
        return raw

    def revoke(self, raw: str) -> None:
        self.tokens[raw] = ActiveLicenseView(active=False, reason="token_not_found")

    def _check(self) -> None:
        if self.down:
            raise OSError("licences unreachable")
        if self.refuse is not None:
            raise LicensesClientError(
                f"licences read returned {self.refuse}: "
                '{"error":"missing_app_key","hint":"Bearer eyJhbGc.eyJzdWI.sig"}',
                status_code=self.refuse,
            )

    async def introspect(self, token: str) -> ActiveLicenseView:
        self._check()
        if self.introspect_down:
            raise LicensesClientError("introspect returned 503: down", status_code=503)
        if self.introspect_refuse is not None:
            raise LicensesClientError(
                f"introspect returned {self.introspect_refuse}: missing_app_key",
                status_code=self.introspect_refuse,
            )
        self.introspect_calls += 1
        return self.tokens.get(token) or ActiveLicenseView(active=False, reason="token_not_found")

    async def list_user_licenses(self, subject_id: object, *, tenant_id: object) -> list[Any]:
        if self.ensure_error_after_commit is None:
            return []
        self.reread_subjects.append(str(subject_id))
        if str(subject_id) != f"svc:{UUID(str(tenant_id))}:w2v":
            return []
        return [LicenseView(id=str(lid)) for lid in self.live.get(UUID(str(tenant_id)), [])]

    async def get_active_license(
        self, *, tenant_id: UUID | str, app_key: str
    ) -> ActiveLicenseView | None:
        self._check()
        self.active_calls += 1
        if UUID(str(tenant_id)) in self.not_onboarded:
            raise TenantNotOnboardedError(
                "get_active_license returned 403: tenant_not_accessible",
                status_code=403,
                code="tenant_not_accessible",
            )
        if UUID(str(tenant_id)) in self.conflict:
            return ActiveLicenseView(conflict=True)
        lics = self.live.get(UUID(str(tenant_id)), [])
        if not lics or app_key != "w2v":
            return None
        return ActiveLicenseView(
            active=True,
            tenant_id=str(tenant_id),
            license_id=str(lics[0]),
            app=ActiveAppRef(key="w2v"),
            plan=ActivePlanRef(code="standard"),
            candidate_count=len(lics),
        )

    async def create_token(self, license_id: UUID | str, *, tenant_id: UUID | str) -> MintedToken:
        self._check()
        raw = self.token(UUID(str(tenant_id)), UUID(str(license_id)))
        self.minted.append((str(license_id), raw))
        return MintedToken(token=raw, expires_at="2027-10-13T00:00:00+00:00")

    async def register_tenant(self, tenant_id: UUID | str, *, name: str) -> RegisteredTenant:
        self._check()
        if self.register_error is not None:
            raise self.register_error
        created = not any(t == str(tenant_id) for t, _ in self.registered)
        self.registered.append((str(tenant_id), name))
        return RegisteredTenant(tenant_id=str(tenant_id), name=name, created=created)

    async def list_apps(self, *, tenant_id: UUID | str) -> list[Any]:
        self._check()
        return []

    async def list_plans(self, app_id: str, *, tenant_id: UUID | str) -> list[Any]:
        return []

    async def ensure_tenant_bundle(
        self, tenant_id: UUID | str, bundle_key: str, *, plans: list[str]
    ) -> BundleEnsureView:
        self._check()
        self.ensure_calls.append(str(tenant_id))
        if self.ensure_error is not None:
            raise self.ensure_error
        tenant = UUID(str(tenant_id))
        apps: dict[str, BundleEnsureAppView] = {}
        refused: list[BundleEnsureRefusalView] = []
        for key, reason in self.ensure_refuse.items():
            refused.append(BundleEnsureRefusalView(app=key, reason=reason))
        if "w2v" not in self.ensure_refuse:
            live = self.live.get(tenant, [])
            if len(live) > 1:
                refused.append(BundleEnsureRefusalView(app="w2v", reason="several_candidates"))
            elif live:
                apps["w2v"] = BundleEnsureAppView(decision="exists", license_id=str(live[0]))
            else:
                lid = uuid4()
                self.live[tenant] = [lid]
                apps["w2v"] = BundleEnsureAppView(
                    decision="insert", license_id=str(lid), plan=plans[0]
                )
        if self.ensure_error_after_commit is not None:
            raise self.ensure_error_after_commit
        return BundleEnsureView(
            tenant_id=str(tenant), bundle=bundle_key, apps=apps, refused=refused
        )

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - a call nobody expected
        raise AttributeError(f"P9Licences has no {name!r} — the code under test called it")


#: A real AES-GCM keyring for sealing tenant licence tokens in tests.
TEST_KEYRING = Keyring([("lt1", bytes(range(32)))])


def override_licence_keyring(app: FastAPI, keyring: Keyring = TEST_KEYRING) -> None:
    from src.api.routes import _common

    app.dependency_overrides[_common.get_licence_keyring] = lambda: keyring
