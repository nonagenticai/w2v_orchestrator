"""Pure derivation + fail-soft tests for the w2v license-status gate.

Mirrors fr's ``backend/tests/unit/lib/license-status.test.ts``. No DB/redis —
the derivation is pure and the orchestrator path is exercised with a fake client.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from src.clients.licenses_client import TenantNotOnboardedError
from src.clients.schemas.licenses import (
    ActiveLicenseView,
    ActivePlanRef,
    AppInfoView,
    LicenseView,
)
from src.services.license_status import (
    derive_active_state,
    derive_inactive_state,
    get_license_status,
)

NOW = datetime(2026, 6, 26, 12, 0, 0, tzinfo=UTC)


def _iso(days: int) -> str:
    return (NOW + timedelta(days=days)).isoformat()


# ─── derive_active_state (200 from /licenses/active) ──────────────────────


def test_active_no_expiry_is_active() -> None:
    r = derive_active_state({"status": "active", "plan": {"code": "standard"}}, NOW)
    assert r.state == "active"
    assert r.has_license is True
    assert r.days_remaining is None
    assert r.plan_code == "standard"


def test_active_far_future_is_active() -> None:
    r = derive_active_state({"expires_at": _iso(90)}, NOW)
    assert r.state == "active"
    assert r.days_remaining == 90


def test_active_within_window_is_expiring_soon() -> None:
    r = derive_active_state({"expires_at": _iso(10)}, NOW)
    assert r.state == "expiring_soon"
    assert r.has_license is True
    assert r.days_remaining == 10


def test_active_at_boundary_is_expiring_soon() -> None:
    r = derive_active_state({"expires_at": _iso(30)}, NOW)
    assert r.state == "expiring_soon"


def test_active_with_past_expiry_is_expired_not_expiring() -> None:
    # A 200 whose expiry is already past (stale 200 / clock skew) must hard-block
    # as expired, NOT show a dismissible expiring banner (fr negative-days fix).
    r = derive_active_state({"expires_at": _iso(-5)}, NOW)
    assert r.state == "expired"
    assert r.has_license is False
    assert r.days_remaining == -5


# ─── derive_inactive_state (404 → subject inventory) ──────────────────────


def _rec(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"app": {"key": "w2v"}, "status": "active"}
    base.update(kw)
    return base


def test_inactive_no_records_is_none() -> None:
    assert derive_inactive_state([], "w2v", NOW).state == "none"


def test_inactive_other_app_only_is_none() -> None:
    recs = [{"app": {"key": "fr"}, "status": "expired", "expires_at": _iso(-1)}]
    assert derive_inactive_state(recs, "w2v", NOW).state == "none"


def test_inactive_revoked_takes_precedence() -> None:
    recs = [
        _rec(status="expired", expires_at=_iso(-10)),
        _rec(status="revoked", revoked_at=_iso(-1), plan={"code": "standard"}),
    ]
    r = derive_inactive_state(recs, "w2v", NOW)
    assert r.state == "revoked"
    assert r.has_license is False
    assert r.plan_code == "standard"


def test_inactive_revoked_via_revoked_at_only() -> None:
    recs = [_rec(status="active", revoked_at=_iso(-2))]
    assert derive_inactive_state(recs, "w2v", NOW).state == "revoked"


def test_inactive_expired_by_status() -> None:
    r = derive_inactive_state([_rec(status="expired", expires_at=_iso(-3))], "w2v", NOW)
    assert r.state == "expired"
    assert r.days_remaining == -3


def test_inactive_expired_by_past_date() -> None:
    r = derive_inactive_state([_rec(status="active", expires_at=_iso(-1))], "w2v", NOW)
    assert r.state == "expired"


def test_inactive_picks_most_recently_expired() -> None:
    recs = [
        _rec(status="expired", expires_at=_iso(-30)),
        _rec(status="expired", expires_at=_iso(-2)),
    ]
    r = derive_inactive_state(recs, "w2v", NOW)
    assert r.state == "expired"
    assert r.days_remaining == -2


def test_inactive_pending_future_is_none() -> None:
    # A record exists but is neither active (we got a 404) nor expired/revoked.
    recs = [_rec(status="pending", expires_at=_iso(20))]
    assert derive_inactive_state(recs, "w2v", NOW).state == "none"


# ─── get_license_status (orchestrator path, fail-soft) ────────────────────


class _FakeClient:
    def __init__(
        self,
        *,
        active: ActiveLicenseView | None = None,
        inventory: list[LicenseView] | None = None,
        raises: bool = False,
    ) -> None:
        self._active = active
        self._inventory = inventory or []
        self._raises = raises

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> ActiveLicenseView | None:
        if self._raises:
            raise RuntimeError("licenses down")
        return self._active

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str) -> list[LicenseView]:
        return self._inventory


async def test_gate_disabled_is_always_active() -> None:
    client = _FakeClient(raises=True)  # would error if consulted
    r = await get_license_status(
        client=client, app_key="w2v", tenant_id="t", subject_id="s", gate_enabled=False
    )
    assert r.state == "active"
    assert r.has_license is True


async def test_no_tenant_is_none() -> None:
    r = await get_license_status(
        client=_FakeClient(), app_key="w2v", tenant_id=None, subject_id="s", gate_enabled=True
    )
    assert r.state == "none"


async def test_active_license_resolves_active() -> None:
    client = _FakeClient(
        active=ActiveLicenseView(expires_at=_iso(120), plan=ActivePlanRef(code="standard"))
    )
    r = await get_license_status(
        client=client, app_key="w2v", tenant_id="t", subject_id="s", gate_enabled=True, now=NOW
    )
    assert r.state == "active"
    assert r.plan_code == "standard"


async def test_conflict_resolves_active_unknown_expiry() -> None:
    client = _FakeClient(active=ActiveLicenseView(conflict=True))
    r = await get_license_status(
        client=client, app_key="w2v", tenant_id="t", subject_id="s", gate_enabled=True, now=NOW
    )
    assert r.state == "active"
    assert r.expires_at is None


async def test_404_then_expired_inventory() -> None:
    client = _FakeClient(
        active=None,
        inventory=[
            LicenseView(
                id="lic-1",
                app=AppInfoView(id="app-w2v", key="w2v", name="W2V"),
                status="expired",
                expires_at=_iso(-4),
            )
        ],
    )
    r = await get_license_status(
        client=client, app_key="w2v", tenant_id="t", subject_id="s", gate_enabled=True, now=NOW
    )
    assert r.state == "expired"


async def test_404_no_subject_is_none() -> None:
    client = _FakeClient(active=None)
    r = await get_license_status(
        client=client, app_key="w2v", tenant_id="t", subject_id=None, gate_enabled=True, now=NOW
    )
    assert r.state == "none"


async def test_upstream_error_is_unknown_not_blocking() -> None:
    client = _FakeClient(raises=True)
    r = await get_license_status(
        client=client, app_key="w2v", tenant_id="t", subject_id="s", gate_enabled=True, now=NOW
    )
    assert r.state == "unknown"
    assert r.has_license is False


class _NotOnboardedClient(_FakeClient):
    async def get_active_license(self, *, tenant_id: str, app_key: str) -> ActiveLicenseView | None:
        raise TenantNotOnboardedError(
            "get_active_license returned 403: tenant_not_accessible",
            status_code=403,
            code="tenant_not_accessible",
        )


async def test_tenant_not_accessible_is_not_onboarded_not_none_or_unknown() -> None:
    """G7: licences' 403 ``tenant_not_accessible`` is its own definitive state —
    not ``none`` (that would read as "buy a licence") and not ``unknown``."""
    r = await get_license_status(
        client=_NotOnboardedClient(),
        app_key="w2v",
        tenant_id="t",
        subject_id="s",
        gate_enabled=True,
        now=NOW,
    )
    assert r.state == "not_onboarded"
    assert r.has_license is False
