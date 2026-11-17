"""Unit tests for the non-test-database guard.

Pure unit tests: no DB fixtures, no connection, no conftest DB machinery.
"""

from __future__ import annotations

import pytest

from tests.support.db_identity import (
    OVERRIDE_ENV_VAR,
    assert_test_database,
    database_name_from_dsn,
)

CI_DSN = (
    "postgresql+psycopg://ci_test:pw@127.0.0.1:5452/w2v_orchestrator_ci"  # pragma: allowlist secret
)
TEST_DSN = "postgresql+psycopg://ci_test:pw@127.0.0.1:5452/w2v_orchestrator_test"  # pragma: allowlist secret
PROD_DSN = (
    "postgresql+psycopg://w2v:pw@postgres.test:5432/w2v_orchestrator"  # pragma: allowlist secret
)


@pytest.fixture(autouse=True)
def _no_ambient_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let an operator's real override env var green these tests."""
    monkeypatch.delenv(OVERRIDE_ENV_VAR, raising=False)


@pytest.mark.parametrize(
    "dsn",
    [CI_DSN],
)
def test_accepts_test_databases(dsn: str) -> None:
    assert assert_test_database(dsn) is None


@pytest.mark.parametrize(
    ("dsn", "expected_name"),
    [(PROD_DSN, "w2v_orchestrator")],
)
def test_rejects_non_test_databases(dsn: str, expected_name: str) -> None:
    with pytest.raises(RuntimeError) as excinfo:
        assert_test_database(dsn)
    assert expected_name in str(excinfo.value)


@pytest.mark.parametrize(
    "dsn",
    ["postgresql+psycopg://user:pw@host:5432"],
)
def test_rejects_missing_or_empty_database_path(dsn: str) -> None:
    with pytest.raises(RuntimeError) as excinfo:
        assert_test_database(dsn)
    assert "no database path" in str(excinfo.value)


def test_override_env_var_allows_a_non_test_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OVERRIDE_ENV_VAR, "1")
    assert assert_test_database(PROD_DSN) is None


def test_override_is_ignored_when_allow_override_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OVERRIDE_ENV_VAR, "1")
    with pytest.raises(RuntimeError):
        assert_test_database(PROD_DSN, allow_override=False)


def test_override_only_honours_the_exact_value_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OVERRIDE_ENV_VAR, "0")
    with pytest.raises(RuntimeError):
        assert_test_database(PROD_DSN)


def test_error_message_never_leaks_the_dsn_or_the_password() -> None:
    password = "sup3rs3cr3t-live-pw"  # pragma: allowlist secret
    user = "w2v_prod_writer"
    dsn = f"postgresql+psycopg://{user}:{password}@postgres.test:5432/w2v_orchestrator"

    with pytest.raises(RuntimeError) as excinfo:
        assert_test_database(dsn)

    message = str(excinfo.value)
    assert password not in message
    assert user not in message
    assert dsn not in message
    assert "postgres.test" not in message
    # ...but it must still be actionable: it names the database it saw.
    assert "w2v_orchestrator" in message


@pytest.mark.parametrize(
    ("dsn", "expected"),
    [(CI_DSN, "w2v_orchestrator_ci")],
)
def test_database_name_from_dsn(dsn: str, expected: str) -> None:
    assert database_name_from_dsn(dsn) == expected
