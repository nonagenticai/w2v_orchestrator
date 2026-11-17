"""Shared LicensesClient stand-ins for route tests.

``require_license`` is ON by default and fails CLOSED, so any test app that
mounts a LICENCE-GATED route without a licences client gets a 503
``license_unavailable`` on every request — which masks whatever that test was
actually about. Tests whose subject is NOT entitlement override
``_common.get_licenses_client`` with :func:`override_entitled` so the gate stays
mounted in its real position and answers "entitled".

Tests whose subject IS entitlement build their own scripted client instead (see
``test_admin_license_gate.py`` / ``test_license_gate.py``) — never this one.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from src.api.routes import _common
from src.clients.schemas.licenses import ActiveLicenseView, ActivePlanRef


class EntitledLicenses:
    """Minimal LicensesClient stand-in reporting a perpetual active licence."""

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> ActiveLicenseView:
        return ActiveLicenseView(expires_at=None, plan=ActivePlanRef(code="standard"))

    async def list_user_licenses(self, subject_id: str | None, *, tenant_id: str) -> list[Any]:
        return []


def override_entitled(app: FastAPI) -> None:
    """Point the app's licences dependency at an entitled stub."""
    app.dependency_overrides[_common.get_licenses_client] = lambda: EntitledLicenses()
