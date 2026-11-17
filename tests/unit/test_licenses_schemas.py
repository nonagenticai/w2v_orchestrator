"""Contract tests for the licenses typed response models.

These pin the two shape traps that shipped as live entitlement-gate bugs, each
now a type error instead of a silent ``None``:

- ``GET /licenses/active`` (``ActiveLicenseView``) is the introspect shape whose
  license id is ``license_id`` — NOT ``id``. ``license_provisioning`` used to read
  ``active["id"]`` off it, so an already-active license ALWAYS resolved to ``None``
  and could double-mint. It also has no ``revoked_at``.
- ``GET /licenses/user/{id}`` / ``POST`` (``LicenseView``) is the LicenseResponse
  shape: it DOES have ``id`` + ``revoked_at`` and has NO bare ``revoked`` bool.
  The admin idempotency filter used to read ``lic.get("revoked")`` — a field that
  never exists — so it never recognised a revoked license.

Source of truth: ``/nonagenticai/licenses/main.py``. DB-free, unit tier.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.clients.schemas.licenses import (
    ActiveLicenseView,
    AppView,
    LicenseView,
    PlanView,
)

# ── ActiveLicenseView: introspect shape has license_id, NOT id ──────────────


def test_active_license_uses_license_id_not_id() -> None:
    """The introspect payload's license id is ``license_id``; a stray ``id`` key
    is IGNORED (extra='ignore'). Reading ``id`` was the provisioning bug."""
    active = ActiveLicenseView.model_validate(
        {"active": True, "license_id": "lic-real", "id": "WRONG", "status": "active"}
    )
    assert active.license_id == "lic-real"
    # ``id`` is not a field on this shape — the old bug read it and got None.
    assert not hasattr(active, "id")
    assert not hasattr(active, "revoked_at")


def test_active_license_conflict_sentinel_defaults() -> None:
    """The client synthesises ``{"conflict": True}`` on a 409 multi-active; a
    minimal payload leaves every wire field None but the sentinel readable."""
    active = ActiveLicenseView(conflict=True)
    assert active.conflict is True
    assert active.active is False
    assert active.license_id is None
    assert active.plan is None


def test_active_license_nested_plan_and_app() -> None:
    active = ActiveLicenseView.model_validate(
        {
            "active": True,
            "license_id": "lic-1",
            "plan": {"code": "standard"},
            "app": {"key": "w2v"},
        }
    )
    assert active.plan is not None
    assert active.plan.code == "standard"
    assert active.app is not None
    assert active.app.key == "w2v"


# ── LicenseView: LicenseResponse shape has id + revoked_at, no `revoked` ─────


def test_license_view_has_id_and_revoked_at_no_revoked_bool() -> None:
    lic = LicenseView.model_validate(
        {
            "id": "lic-1",
            "app_id": "app-w2v",
            "status": "revoked",
            "revoked_at": "2020-01-01T00:00:00+00:00",
        }
    )
    assert lic.id == "lic-1"
    assert lic.revoked_at == "2020-01-01T00:00:00+00:00"
    # There is no bare ``revoked`` bool — the admin idempotency bug read it.
    assert not hasattr(lic, "revoked")


def test_license_view_only_id_required() -> None:
    """Every field but ``id`` is optional — a lean row must validate."""
    lic = LicenseView.model_validate({"id": "lic-1"})
    assert lic.id == "lic-1"
    assert lic.revoked_at is None
    assert lic.status is None


def test_license_view_requires_id() -> None:
    with pytest.raises(ValidationError):
        LicenseView.model_validate({"app_id": "app-w2v"})  # missing id


# ── AppView / PlanView: revocation is a bool on THESE (catalog rows) ─────────


def test_app_and_plan_views_carry_revoked_bool() -> None:
    app = AppView.model_validate({"id": "app-w2v", "key": "w2v", "name": "W2V"})
    assert app.revoked is False
    plan = PlanView.model_validate(
        {"id": "p1", "app_id": "app-w2v", "code": "std", "name": "Std", "revoked": True}
    )
    assert plan.revoked is True


def test_models_ignore_unknown_downstream_fields() -> None:
    """extra='ignore' (pydantic default): a downstream ADDING a field can never
    break the orchestrator."""
    lic = LicenseView.model_validate({"id": "lic-1", "brand_new_field": {"x": 1}})
    assert lic.id == "lic-1"
