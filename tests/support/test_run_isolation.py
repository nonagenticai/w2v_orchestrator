"""Unit gate on the per-run schema isolation helpers.

These run everywhere (no database needed), so CI — where the mechanism itself is
deliberately OFF — still proves the switch, the naming and, most importantly,
that the termination sweep is scoped to this run's own backends.
"""

from __future__ import annotations

import pytest

from tests.support import run_isolation


@pytest.mark.parametrize(
    ("env", "expected"),
    [({}, True)],
)
def test_isolation_defaults_off_in_ci_and_is_overridable(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: bool
) -> None:
    for key in ("CI", "GITHUB_ACTIONS", "W2V_TEST_SCHEMA_ISOLATION"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert run_isolation.isolation_enabled() is expected


def test_run_schema_and_app_name_share_one_token() -> None:
    """The GC recovers a schema's owner by re-deriving its application_name.

    If these two ever stop sharing a token, the liveness check silently finds no
    owner for every schema and the GC starts reclaiming LIVE runs.
    """
    token = run_isolation.RUN_SCHEMA.removeprefix(run_isolation.SCHEMA_PREFIX)
    assert f"{run_isolation.APP_NAME_PREFIX}{token}" == run_isolation.RUN_APP_NAME
    assert token == run_isolation.RUN_TOKEN
    assert len(run_isolation.RUN_SCHEMA) <= 63  # postgres identifier limit


def test_pgoptions_pins_both_search_path_and_application_name() -> None:
    value = run_isolation.pgoptions("some_schema")
    assert '-c search_path="some_schema"' in value
    assert f"-c application_name={run_isolation.RUN_APP_NAME}" in value


def test_terminate_sweep_is_scoped_to_this_run_when_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of the fix: the sweep cannot reach another run's backends.

    A regression here re-opens the original bug — run A executing run B's
    perfectly healthy `idle in transaction` fixture connection.
    """
    monkeypatch.setenv("W2V_TEST_SCHEMA_ISOLATION", "1")
    sql = run_isolation.terminate_scope_sql()
    assert f"application_name = '{run_isolation.RUN_APP_NAME}'" in sql
    assert "idle in transaction" in sql  # the leaked-lock guard is still there
    assert "'active'" not in sql


def test_terminate_sweep_is_unchanged_when_isolation_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CI must keep the exact pre-existing statement."""
    monkeypatch.setenv("W2V_TEST_SCHEMA_ISOLATION", "0")
    assert run_isolation.terminate_scope_sql() == (
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        "WHERE datname = current_database() "
        "AND pid <> pg_backend_pid() "
        "AND state IN ('idle in transaction', 'idle in transaction (aborted)')"
    )


def test_scoped_schema_is_a_noop_without_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("W2V_TEST_SCHEMA_ISOLATION", "0")
    assert run_isolation.scoped_schema("parity_alembic") == "parity_alembic"
    monkeypatch.setenv("W2V_TEST_SCHEMA_ISOLATION", "1")
    assert run_isolation.scoped_schema("parity_alembic") == (
        f"parity_alembic_{run_isolation.RUN_TOKEN}"
    )


def test_libpq_dsn_strips_the_sqlalchemy_driver_suffix() -> None:
    assert (
        run_isolation.libpq_dsn("postgresql+psycopg://u:p@h:5452/db")  # pragma: allowlist secret
        == "postgresql://u:p@h:5452/db"  # pragma: allowlist secret
    )
    assert run_isolation.libpq_dsn("postgresql://u@h/db") == "postgresql://u@h/db"


@pytest.mark.parametrize(
    ("comment", "older"),
    [("created_at=2000-01-01T00:00:00+00:00", True)],
)
def test_stale_detection_treats_an_unreadable_comment_as_ancient(
    comment: object, older: bool
) -> None:
    from datetime import UTC, datetime

    cutoff = datetime.now(UTC)
    assert run_isolation._is_older_than(comment, cutoff) is older
