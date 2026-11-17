"""A licences fake that models WHO HOLDS each licence.

``ensure_tenant_bundle`` behaves like licences' ``bundle_ensure``: one licence
per (tenant, app) of the ``w2v`` bundle, held by ``svc:<tenant>:<app>`` —
``insert`` the first time, ``exists`` after; an app with a person-held licence
is refused ``member_licence``. ``create_license`` records a licence held by the
path subject (a PERSON when w2v mints the old way), so a test reads
``holders`` to tell a tenant-level licence from a person-held one.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from src.clients.licenses_client import LicensesClientError, MintedToken
from src.clients.schemas.licenses import (
    AppView,
    BundleEnsureAppView,
    BundleEnsureRefusalView,
    BundleEnsureView,
    LicenseView,
    PlanView,
    TenantAppByoModeSetView,
)

W2V_BUNDLE_APPS = ("w2v", "w2v-image")


class TenantLicencesFake:
    def __init__(
        self,
        *,
        apps: list[AppView] | None = None,
        plans: list[PlanView] | None = None,
        bundle_apps: tuple[str, ...] = W2V_BUNDLE_APPS,
    ) -> None:
        self.apps = (
            apps
            if apps is not None
            else [
                AppView(id="app-w2v", key="w2v", name="W2V"),
                AppView(id="app-w2v-image", key="w2v-image", name="W2V Image"),
            ]
        )
        self.plans = (
            plans
            if plans is not None
            else [
                PlanView(id="plan-1", app_id="app-w2v", code="p1", name="P1", revoked=True),
                PlanView(id="plan-2", app_id="app-w2v", code="p2", name="P2"),
            ]
        )
        self.bundle_apps = bundle_apps
        #: licence id -> (tenant, app key, holder subject)
        self.holders: dict[str, tuple[str, str, str]] = {}
        self.ensure_calls: list[dict[str, object]] = []
        self.token_calls: list[str] = []
        self.registered: list[tuple[str, str]] = []
        self.ensure_error: Exception | None = None
        #: Raised AFTER the ensure wrote its licences — licences finished, the
        #: client gave up first (live 2026-10-15: 15.8 s against a 15 s budget).
        self.ensure_error_after_commit: Exception | None = None
        self.reread_calls: list[str] = []
        #: (tenant, app key) -> per-app byo_mode override written by w2v.
        self.byo_overrides: dict[tuple[str, str], str] = {}
        #: Raised by ``set_tenant_app_byo_mode`` (licences refused/failed the write).
        self.byo_set_error: Exception | None = None
        #: Raise ``byo_set_error`` only for the first N writes (None ⇒ every write).
        self.byo_set_fail_times: int | None = None
        self.byo_set_attempts = 0

    # -- helpers for assertions ------------------------------------------------
    def holder_of(self, license_id: str) -> str:
        return self.holders[license_id][2]

    def _find(self, tenant: str, app_key: str) -> list[str]:
        return [lid for lid, (t, a, _h) in self.holders.items() if t == tenant and a == app_key]

    # -- the LicensesClient surface ---------------------------------------------
    async def register_tenant(self, tenant_id, *, name: str):  # type: ignore[no-untyped-def]
        from src.clients.licenses_client import RegisteredTenant

        self.registered.append((str(tenant_id), name))
        return RegisteredTenant(tenant_id=str(tenant_id), name=name, created=True)

    async def list_apps(self, *, tenant_id) -> list[AppView]:  # type: ignore[no-untyped-def]
        return self.apps

    async def list_plans(self, app_id: str, *, tenant_id) -> list[PlanView]:  # type: ignore[no-untyped-def]
        return self.plans

    async def ensure_tenant_bundle(
        self, tenant_id: UUID | str, bundle_key: str, *, plans: list[str]
    ) -> BundleEnsureView:
        self.ensure_calls.append(
            {"tenant_id": str(tenant_id), "bundle": bundle_key, "plans": plans}
        )
        if self.ensure_error is not None:
            raise self.ensure_error
        tenant = str(UUID(str(tenant_id)))
        apps: dict[str, BundleEnsureAppView] = {}
        refused: list[BundleEnsureRefusalView] = []
        for key in self.bundle_apps:
            holder = f"svc:{tenant}:{key}"
            existing = self._find(tenant, key)
            tenant_level = [lid for lid in existing if self.holders[lid][2] == holder]
            if tenant_level:
                apps[key] = BundleEnsureAppView(decision="exists", license_id=tenant_level[0])
            elif existing:
                refused.append(BundleEnsureRefusalView(app=key, reason="member_licence"))
            else:
                lid = str(uuid4())
                self.holders[lid] = (tenant, key, holder)
                apps[key] = BundleEnsureAppView(decision="insert", license_id=lid, plan=plans[0])
        if self.ensure_error_after_commit is not None:
            raise self.ensure_error_after_commit
        return BundleEnsureView(tenant_id=tenant, bundle=bundle_key, apps=apps, refused=refused)

    async def create_license(  # the OLD person-held mint — recorded, never wanted
        self, *, subject_id, app_id, plan_id, tenant_id, metadata=None
    ) -> LicenseView:  # type: ignore[no-untyped-def]
        key = next(a.key for a in self.apps if a.id == app_id)
        lid = str(uuid4())
        self.holders[lid] = (str(tenant_id), key, str(subject_id))
        return LicenseView(id=lid)

    async def list_user_licenses(self, subject_id, *, tenant_id) -> list[LicenseView]:  # type: ignore[no-untyped-def]
        self.reread_calls.append(str(subject_id))
        return [
            LicenseView(id=lid)
            for lid, (t, _a, h) in self.holders.items()
            if h == str(subject_id) and t == str(tenant_id)
        ]

    async def regenerate_token(self, license_id: str, *, tenant_id) -> str:  # type: ignore[no-untyped-def]
        # Revokes every other token of the licence — w2v must never call it here.
        raise LicensesClientError("regenerate_token must not be called (it revokes tokens)")

    async def create_token(self, license_id: str, *, tenant_id) -> MintedToken:  # type: ignore[no-untyped-def]
        self.token_calls.append(license_id)
        n = self.token_calls.count(license_id)
        return MintedToken(token=f"TOKEN-{license_id}-{n}", expires_at="2027-10-13T00:00:00Z")

    async def set_tenant_app_byo_mode(  # type: ignore[no-untyped-def]
        self, *, tenant_id, app_key: str, byo_mode: str
    ) -> TenantAppByoModeSetView:
        self.byo_set_attempts += 1
        if self.byo_set_error is not None and (
            self.byo_set_fail_times is None or self.byo_set_attempts <= self.byo_set_fail_times
        ):
            raise self.byo_set_error
        key = (str(tenant_id), app_key)
        prior = self.byo_overrides.get(key)
        self.byo_overrides[key] = byo_mode
        return TenantAppByoModeSetView(
            tenant_id=str(tenant_id), app_key=app_key, byo_mode=byo_mode, prior_byo_mode=prior
        )
