"""The test-DSN resolution contract: env wins, else ephemeral, else fail closed."""

from __future__ import annotations

import subprocess

import pytest

from tests.support import test_db


@pytest.fixture(autouse=True)
def _no_leftover_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every case starts with no container and no inherited DSN."""
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(test_db, "_started_container", None)


def test_test_database_url_wins_over_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql+psycopg://u:p@h:1/one_ci")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@h:2/two_ci")
    assert test_db.resolve_test_database_url() == "postgresql+psycopg://u:p@h:1/one_ci"


def test_an_explicit_url_never_touches_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@h:2/two_ci")

    def _explode(*args: str, **kwargs: object) -> None:
        raise AssertionError(f"docker must not be invoked when a DSN is given: {args}")

    monkeypatch.setattr(test_db, "_docker", _explode)
    assert test_db.resolve_test_database_url() == "postgresql+psycopg://u:p@h:2/two_ci"


def test_no_env_starts_an_ephemeral_test_database(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    def _fake_docker(*args: str, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[0] == "run":
            return subprocess.CompletedProcess(args, 0, "cid123\n", "")
        if args[0] == "port":
            return subprocess.CompletedProcess(args, 0, "127.0.0.1:49876\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(test_db, "_docker", _fake_docker)
    monkeypatch.setattr(test_db, "_wait_until_queryable", lambda port, deadline: True)

    url = test_db.resolve_test_database_url()

    assert url.endswith("/w2v_orchestrator_ci"), url
    assert ":49876/" in url
    # The image is pinned to CI's, and the container is labelled for leak-finding.
    run_args = calls[0]
    assert test_db.EPHEMERAL_IMAGE in run_args
    assert test_db.EPHEMERAL_LABEL in run_args
    # Disk-backed, never --tmpfs: a full suite OOM-kills a tmpfs database.
    assert "--tmpfs" not in run_args


def test_teardown_removes_the_container_and_its_anonymous_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remove_args: list[tuple[str, ...]] = []

    def _fake_docker(*args: str, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[0] == "run":
            return subprocess.CompletedProcess(args, 0, "cid123\n", "")
        if args[0] == "port":
            return subprocess.CompletedProcess(args, 0, "127.0.0.1:49876\n", "")
        remove_args.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(test_db, "_docker", _fake_docker)
    monkeypatch.setattr(test_db, "_wait_until_queryable", lambda port, deadline: True)

    test_db.resolve_test_database_url()
    test_db.stop_ephemeral_postgres()

    # -v is what removes the anonymous data volume; without it, runs accumulate.
    assert remove_args == [("rm", "-f", "-v", "cid123")]


def test_a_slow_server_fails_closed_rather_than_hanging(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_docker(*args: str, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[0] == "run":
            return subprocess.CompletedProcess(args, 0, "cid123\n", "")
        if args[0] == "port":
            return subprocess.CompletedProcess(args, 0, "127.0.0.1:49876\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(test_db, "_docker", _fake_docker)
    monkeypatch.setattr(test_db, "_wait_until_queryable", lambda port, deadline: False)

    with pytest.raises(RuntimeError, match="never became queryable"):
        test_db.resolve_test_database_url()


def test_a_wedged_daemon_names_the_wedge_and_gives_a_manual_way_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured failure mode: when the daemon wedges, `docker ps` hangs too."""

    def _hangs(*args: str, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="docker", timeout=test_db.DOCKER_TIMEOUT_S)

    monkeypatch.setattr(test_db.shutil, "which", lambda name: f"/usr/local/bin/{name}")
    monkeypatch.setattr(test_db.subprocess, "run", _hangs)

    with pytest.raises(RuntimeError) as excinfo:
        test_db.resolve_test_database_url()

    message = str(excinfo.value)
    assert "did not answer" in message
    assert "docker run --rm -d" in message  # the manual escape hatch is included


def test_missing_docker_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(test_db.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="docker is not installed"):
        test_db.resolve_test_database_url()


def test_the_ephemeral_database_name_satisfies_the_destructive_suite_guard() -> None:
    """db_identity refuses anything not ending _ci/_test; this must keep passing."""
    from tests.support.db_identity import assert_test_database

    assert_test_database(
        f"postgresql+psycopg://u:p@127.0.0.1:1/{test_db.EPHEMERAL_DATABASE}",
        allow_override=False,
    )
