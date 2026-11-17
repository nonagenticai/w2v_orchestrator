"""Unit tests for the ``require_createrole`` RLS-privilege tripwire.

The RLS isolation + rebind suites need ``CREATEROLE`` to forge a throwaway
``NOBYPASSRLS`` role and prove the shipped policies isolate. If that privilege
is ever missing, the suites MUST fail under CI (so tenant-isolation coverage
can't silently vanish while CI stays green) and skip only for a developer on a
locally-restricted role. This guards that branch logic without needing a DB.
"""

from __future__ import annotations

import pytest
from _pytest.outcomes import Failed  # what pytest.fail(...) raises

from tests.conftest import _running_in_ci, require_createrole

_CREATEROLE_ERR = Exception("permission denied to create role")


def test_fails_loud_under_ci(monkeypatch) -> None:
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert _running_in_ci() is True
    with pytest.raises(Failed):
        require_createrole(_CREATEROLE_ERR)


def test_skips_locally(monkeypatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert _running_in_ci() is False
    with pytest.raises(pytest.skip.Exception):
        require_createrole(_CREATEROLE_ERR)


def test_github_actions_env_also_trips_ci(monkeypatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert _running_in_ci() is True
    with pytest.raises(Failed):
        require_createrole(_CREATEROLE_ERR)
