"""Route-level tests for the self-serve lazy auto-issue branch of
``GET /api/license/status`` (``src/api/routes/auth.py``).

The handler is a thin wrapper over the real ``get_license_status``, so these call
it directly with a stateful fake client (no DB) and a stubbed settings object to
flip the two flags. They prove the branch fires only on ``state='none'`` with both
flags on, is idempotent, and is fail-soft (a licenses error never 500s the poll).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast
from uuid import UUID

import pytest

from src.api.routes import auth as auth_route
from src.clients.licenses_client import LicensesClient, LicensesClientError
from src.clients.schemas.licenses import (
    ActiveLicenseView,
    ActivePlanRef,
    AppInfoView,
    AppView,
    BundleEnsureAppView,
    BundleEnsureView,
    LicenseView,
    PlanView,
)
from src.services.license_status import clear_license_status_cache

_TENANT = UUID("0195f48b-d000-7000-8000-0000000000aa")
_SUBJECT = UUID("0195f48b-d000-7000-8000-0000000000bb")


class _Settings:
    def __init__(self, *, gate: bool, autoissue: bool) -> None:
        self.license_gate_enabled = gate
        self.license_autoissue_enabled = autoissue


def _patch_settings(monkeypatch: Any, *, gate: bool, autoissue: bool) -> None:
    monkeypatch.setattr(
        auth_route, "get_settings", lambda: _Settings(gate=gate, autoissue=autoissue)
    )


@pytest.fixture(autouse=True)
def _reset_status_cache() -> Iterator[None]:
    # get_license_status now memoizes per (tenant, app, subject); these tests
    # reuse a single identity, so a stale entry would bleed across cases.
    clear_license_status_cache()
    yield
    clear_license_status_cache()


class _RouteFake:
    """Returns no active license until ``create_license`` flips it to active."""

    def __init__(
        self,
        *,
        issue_succeeds: bool = True,
        inventory: list[LicenseView] | None = None,
    ) -> None:
        self._issued = False
        self._issue_succeeds = issue_succeeds
        self._inventory = inventory or []
        self.ensure_calls: list[tuple[str, str]] = []

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> ActiveLicenseView | None:
        if self._issued:
            return ActiveLicenseView(
                active=True, status="active", plan=ActivePlanRef(code="standard")
            )
        return None

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str) -> list[LicenseView]:
        return self._inventory

    async def list_apps(self, *, tenant_id: str) -> list[AppView]:
        return [AppView(id="app-w2v", key="w2v", name="W2V")]

    async def list_plans(self, app_id: str, *, tenant_id: str) -> list[PlanView]:
        return [PlanView(id="plan-std", app_id="app-w2v", code="std", name="Std")]

    async def ensure_tenant_bundle(
        self, tenant_id: UUID | str, bundle_key: str, *, plans: list[str]
    ) -> BundleEnsureView:
        """licences' bundle ensure: the tenant-level licence, held by svc:<tenant>:w2v."""
        self.ensure_calls.append((str(tenant_id), bundle_key))
        if not self._issue_succeeds:
            raise LicensesClientError("boom")
        self._issued = True
        return BundleEnsureView(
            tenant_id=str(tenant_id),
            bundle=bundle_key,
            apps={"w2v": BundleEnsureAppView(decision="insert", license_id="lic-new")},
        )

    async def create_license(self, **kwargs: Any) -> LicenseView:
        # The old per-PERSON mint (subject = the polling member). Never again.
        raise AssertionError(f"person-held licence minted: {kwargs.get('subject_id')}")


async def _status(fake: _RouteFake, *, tenant: UUID | None = _TENANT) -> Any:
    return await auth_route.license_status(
        tenant_id=cast(UUID, tenant),
        subject_id=_SUBJECT,
        licenses=cast(LicensesClient, fake),
    )


async def test_autoissue_mints_and_flips_to_active(monkeypatch: Any) -> None:
    _patch_settings(monkeypatch, gate=True, autoissue=True)
    fake = _RouteFake()
    resp = await _status(fake)
    assert resp.state == "active"
    # Ensured for the TENANT (bundle w2v → holder svc:<tenant>:w2v), not the poller.
    assert fake.ensure_calls == [(str(_TENANT), "w2v")]


async def test_autoissue_off_stays_none(monkeypatch: Any) -> None:
    _patch_settings(monkeypatch, gate=True, autoissue=False)
    fake = _RouteFake()
    resp = await _status(fake)
    assert resp.state == "none"
    assert fake.ensure_calls == []


async def test_gate_off_never_autoissues(monkeypatch: Any) -> None:
    # Gate OFF → status is always 'active' and the none-branch never runs.
    _patch_settings(monkeypatch, gate=False, autoissue=True)
    fake = _RouteFake()
    resp = await _status(fake)
    assert resp.state == "active"
    assert fake.ensure_calls == []


async def test_autoissue_failure_is_fail_soft(monkeypatch: Any) -> None:
    _patch_settings(monkeypatch, gate=True, autoissue=True)
    fake = _RouteFake(issue_succeeds=False)
    resp = await _status(fake)  # must NOT raise
    assert resp.state == "none"
    assert len(fake.ensure_calls) == 1


async def test_no_tenant_never_autoissues(monkeypatch: Any) -> None:
    _patch_settings(monkeypatch, gate=True, autoissue=True)
    fake = _RouteFake()
    resp = await _status(fake, tenant=None)
    assert resp.state == "none"
    assert fake.ensure_calls == []


async def test_expired_tenant_not_autoissued(monkeypatch: Any) -> None:
    # An expired inventory record → state 'expired', NOT 'none' → never minted
    # (re-issue is a renewal/billing decision, not onboarding auto-issue).
    _patch_settings(monkeypatch, gate=True, autoissue=True)
    fake = _RouteFake(
        inventory=[
            LicenseView(
                id="lic-1",
                app=AppInfoView(id="app-w2v", key="w2v", name="W2V"),
                status="expired",
                expires_at="2020-01-01T00:00:00+00:00",
            )
        ]
    )
    resp = await _status(fake)
    assert resp.state == "expired"
    assert fake.ensure_calls == []


async def test_revoked_tenant_not_autoissued(monkeypatch: Any) -> None:
    _patch_settings(monkeypatch, gate=True, autoissue=True)
    fake = _RouteFake(
        inventory=[
            LicenseView(
                id="lic-1",
                app=AppInfoView(id="app-w2v", key="w2v", name="W2V"),
                status="revoked",
                revoked_at="2020-01-01T00:00:00+00:00",
            )
        ]
    )
    resp = await _status(fake)
    assert resp.state == "revoked"
    assert fake.ensure_calls == []
