"""Shared fixtures for the run-setup tests."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest

from src.services.providers.byo_posture import StaticByoPosture
from src.services.providers.settings import ProviderPolicySettings


class _NoConnections:
    """A tenant with no provider connections → every slot pins the deployment default."""

    async def get_active(self, tenant_id: UUID, slot: str) -> None:
        return None

    async def get_version(self, connection_id: UUID, version: int) -> None:
        return None


@pytest.fixture
def stub_provider_pins() -> Iterator[None]:
    """For build_config_snapshot tests that run on a SENTINEL session (no DB): the
    P3 provider pin reads tenant connections, so give it an empty repository. The
    pinning logic itself still runs (deployment refs for every slot). The P6 tenant
    defaults read gets "no row" (→ deployment defaults, byte-identical snapshot)."""
    deps = SimpleNamespace(
        repo=_NoConnections(),
        policy=ProviderPolicySettings(),
        posture=StaticByoPosture("local_allowed"),
    )
    with (
        patch("src.services.runs.run_setup.build_resolver_deps", return_value=deps),
        patch(
            "src.services.runs.run_setup.resolve_run_defaults",
            new=AsyncMock(return_value=None),
        ),
    ):
        yield
