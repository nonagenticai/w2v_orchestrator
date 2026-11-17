"""Resolve the test database DSN, starting a throwaway postgres:18 when unset.

Estate standard (2026-10-21): ``TEST_DATABASE_URL`` > ``DATABASE_URL`` > an
ephemeral ``postgres:18`` this process starts and tears down > fail closed.
No host default, no magic port for a human to hand-start.

Two deliberate choices:

* **Disk-backed, not ``--tmpfs``** — a full suite writes more than tmpfs holds
  under memory pressure, and an OOM-killed database reads as a test failure.
* **Every docker call is timeout-bounded** — measured 2026-10-29: when the
  daemon wedges, ``docker ps`` itself hangs for minutes, so an unbounded call
  turns "docker is busy" into "the session hangs forever".

The DSN always names a ``_ci`` database, so ``db_identity.assert_test_database``
still recognises it: this does not widen what the destructive suite runs against.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import shutil
import subprocess
import time

import psycopg

#: Consulted in order. TEST_DATABASE_URL is the estate standard; DATABASE_URL is
#: what CI sets and what tests/conftest.py used to read alone.
PRECEDENCE = ("TEST_DATABASE_URL", "DATABASE_URL")

EPHEMERAL_IMAGE = "postgres:18"
#: Must end in ``_ci`` — see db_identity.TEST_DB_SUFFIXES. Matches CI's database.
EPHEMERAL_DATABASE = "w2v_orchestrator_ci"
EPHEMERAL_USER = "ci_test"
EPHEMERAL_PASSWORD = "ci_test_password_secure"  # pragma: allowlist secret

#: Label every container this module starts, so a leak (SIGKILL, no atexit) is
#: findable with `docker ps -a --filter label=...`.
EPHEMERAL_LABEL = "nonagentic.test-db=w2v_orchestrator"

#: How long to wait for the server to accept TCP. Generous for a cold pull
#: (the image is usually cached), tight enough that a broken box fails fast.
READY_TIMEOUT_S = 90
#: Cap on any single docker CLI call. A healthy call is well under a second.
DOCKER_TIMEOUT_S = 30

_started_container: str | None = None


def _docker(*args: str, timeout: float = DOCKER_TIMEOUT_S) -> subprocess.CompletedProcess[str]:
    """Run ``docker <args>``, converting a wedged daemon into a clean failure."""
    executable = shutil.which("docker")
    if executable is None:
        raise RuntimeError("docker is not installed")
    try:
        return subprocess.run(  # noqa: S603 - fixed argv tuple, no shell, absolute path
            (executable, *args),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"docker {args[0]} did not answer within {timeout:.0f}s — the daemon is "
            "wedged or overloaded (measured on this box 2026-10-29: `docker ps` "
            "also hangs). Not waiting."
        ) from exc


def missing_url_message() -> str:
    return (
        "No test database. Set TEST_DATABASE_URL (or DATABASE_URL), or make "
        "`docker` available so the suite starts an ephemeral postgres:18. "
        "There is no host default and no fixed port. Manually:\n"
        f"  CID=$(docker run --rm -d -e POSTGRES_USER={EPHEMERAL_USER} \\\n"
        f"        -e POSTGRES_PASSWORD={EPHEMERAL_PASSWORD} \\\n"
        f"        -e POSTGRES_DB={EPHEMERAL_DATABASE} -p 127.0.0.1::5432 {EPHEMERAL_IMAGE})\n"
        "  PORT=$(docker port \"$CID\" 5432/tcp | head -1 | awk -F: '{print $NF}')\n"
        f"  export TEST_DATABASE_URL=postgresql+psycopg://{EPHEMERAL_USER}:{EPHEMERAL_PASSWORD}"
        f"@127.0.0.1:$PORT/{EPHEMERAL_DATABASE}"
    )


def stop_ephemeral_postgres() -> None:
    """Remove the container and its anonymous data volume, if this process started it."""
    global _started_container
    container, _started_container = _started_container, None
    if container:
        # Teardown must never mask the real result of the suite. -v also removes
        # the anonymous volume, so nothing accumulates across runs.
        with contextlib.suppress(RuntimeError):
            _docker("rm", "-f", "-v", container)


def _wait_until_queryable(port: str, deadline: float) -> bool:
    """True once a client can actually CONNECT AND QUERY.

    Not a TCP probe: the postgres entrypoint starts a temporary init server that
    accepts and then drops connections before the real one is up, so a socket
    check passes while the database still answers nothing. Measured 2026-10-29 —
    that false positive surfaced as `server closed the connection unexpectedly`
    from the first real test.
    """
    dsn = (
        f"postgresql://{EPHEMERAL_USER}:{EPHEMERAL_PASSWORD}@127.0.0.1:{port}/{EPHEMERAL_DATABASE}"
    )
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(dsn, connect_timeout=3) as conn:
                conn.execute("select 1")
            return True
        except psycopg.OperationalError:
            time.sleep(0.5)
    return False


def start_ephemeral_postgres() -> str:
    """Start a disposable postgres:18 and return its SQLAlchemy URL."""
    global _started_container

    name = f"w2v-orch-test-pg-{os.getpid()}"
    try:
        run = _docker(
            "run",
            "--rm",
            "-d",
            "--name",
            name,
            "--label",
            EPHEMERAL_LABEL,
            # Anonymous volume, not --tmpfs and not a host bind mount. tmpfs is
            # OOM-prone for a full suite; a macOS bind mount cannot be owned by
            # the container's postgres user and initdb dies with "data directory
            # has wrong ownership" (measured 2026-10-29). An anonymous volume is
            # disk-backed, correctly owned, and `--rm` deletes it with the pod.
            "-v",
            "/var/lib/postgresql",
            "-p",
            "127.0.0.1::5432",
            "-e",
            f"POSTGRES_USER={EPHEMERAL_USER}",
            "-e",
            f"POSTGRES_PASSWORD={EPHEMERAL_PASSWORD}",
            "-e",
            f"POSTGRES_DB={EPHEMERAL_DATABASE}",
            EPHEMERAL_IMAGE,
        )
        if run.returncode != 0:
            raise RuntimeError(run.stderr.strip() or f"docker run exited {run.returncode}")
        container = run.stdout.strip()
        _started_container = container

        port_result = _docker("port", container, "5432/tcp")
        if port_result.returncode != 0 or not port_result.stdout.strip():
            raise RuntimeError(port_result.stderr.strip() or "could not read the mapped port")
        port = port_result.stdout.strip().splitlines()[0].rsplit(":", 1)[-1]

        if not _wait_until_queryable(port, time.monotonic() + READY_TIMEOUT_S):
            raise RuntimeError(
                f"the ephemeral postgres on 127.0.0.1:{port} never became queryable "
                f"within {READY_TIMEOUT_S}s"
            )
    except RuntimeError:
        stop_ephemeral_postgres()
        raise

    return (
        f"postgresql+psycopg://{EPHEMERAL_USER}:{EPHEMERAL_PASSWORD}"
        f"@127.0.0.1:{port}/{EPHEMERAL_DATABASE}"  # pragma: allowlist secret
    )


def resolve_test_database_url() -> str:
    """The test DSN: explicit env first, else an ephemeral container, else raise."""
    for name in PRECEDENCE:
        value = os.environ.get(name)
        if value:
            return value

    atexit.register(stop_ephemeral_postgres)
    try:
        return start_ephemeral_postgres()
    except RuntimeError as exc:
        raise RuntimeError(f"{exc}\n\n{missing_url_message()}") from exc
