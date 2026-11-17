"""Validation and serialization contract for AuthConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.auth_config import AuthConfig, AuthType


class TestAuthConfigValidation:
    def test_requires_credential_id(self) -> None:
        # Validate from dict so ty doesn't mistake this for a call-site error —
        # the point is that pydantic rejects the missing field at runtime.
        with pytest.raises(ValidationError):
            AuthConfig.model_validate({})

    def test_default_type_is_form(self) -> None:
        c = AuthConfig(credential_id="cred-1")
        assert c.type is AuthType.FORM

    def test_accepts_all_auth_types(self) -> None:
        for t in ("form", "storage_state", "bearer", "basic"):
            c = AuthConfig.model_validate({"credential_id": "cred-1", "type": t})
            assert c.type == t

    def test_wait_timeout_bounds_enforced(self) -> None:
        with pytest.raises(ValidationError):
            AuthConfig(credential_id="cred-1", wait_timeout_ms=500)
        with pytest.raises(ValidationError):
            AuthConfig(credential_id="cred-1", wait_timeout_ms=200_000)

    def test_serializes_compatible_with_kag(self) -> None:
        """Wire shape must match kag's AuthConfig 1:1 (verified 2026-04-18)."""
        c = AuthConfig.model_validate(
            {
                "credential_id": "cred-1",
                "login_url": "https://x/login",
                "wait_for_selector": "h1.dashboard",
            }
        )
        body = c.model_dump(mode="json", exclude_none=True)
        assert body["credential_id"] == "cred-1"
        assert body["type"] == "form"
        assert body["login_url"] == "https://x/login"
        assert body["wait_for_selector"] == "h1.dashboard"
        assert "username_selector" not in body
