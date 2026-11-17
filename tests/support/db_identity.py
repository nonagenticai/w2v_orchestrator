"""Refuse to run the destructive test suite against a non-test database.

Why this exists
---------------
``tests/conftest.py`` captures ``DATABASE_URL`` from the ambient environment
(``os.environ.setdefault`` only fills it in when it is *unset*, so a pre-set
production URL is silently adopted) and then, against whatever it captured,
runs ``DROP TABLE ... CASCADE``, ``Base.metadata.drop_all`` and
``TRUNCATE TABLE ... RESTART IDENTITY CASCADE``.

``tests/support/run_isolation.py`` normally contains that blast radius by
pinning every connection to a private schema — but ``isolation_enabled()``
returns ``False`` when ``CI=true`` / ``GITHUB_ACTIONS=true``, deliberately. So
in any environment that is both "CI" and carrying a production
``DATABASE_URL`` (an in-cluster pod, a self-hosted runner, a shell with prod
env sourced) those statements land on ``public`` with nothing in the way.
``projects -> deliverables -> runs -> stage_outputs -> run_usage`` all CASCADE
(migration ``0015``) and ``runs`` has no ``deleted_at``.

This is not hypothetical: orchestrator #878, "CI tests were WRITING TO
PRODUCTION".

The check is deliberately dumb: the database *name* must look like a test
database. Both the CI database and the local docker-compose default are
``w2v_orchestrator_ci``, so this accepts every legitimate caller today.

Secrecy rule
------------
The raised message reports the database NAME and nothing else. It must never
carry the DSN, the password or the userinfo — a psycopg exception in this
estate leaked a live password twice because its text carried the full
conninfo.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

#: A database is accepted iff its name ends with one of these.
TEST_DB_SUFFIXES = ("_ci", "_test")

#: Deliberate operator escape hatch. Accidents do not set env vars.
OVERRIDE_ENV_VAR = "W2V_ALLOW_NON_TEST_DATABASE"


def database_name_from_dsn(dsn: str) -> str:
    """Return the database name in ``dsn``, or ``""`` if it has none.

    Handles the SQLAlchemy dialect scheme (``postgresql+psycopg://``), a query
    string, and a missing or empty path.
    """
    path = urlsplit(dsn).path
    # urlsplit already strips ?query and #fragment. Leading "/" is the
    # authority/path separator; the dbname is the first path segment.
    return path.lstrip("/").split("/", 1)[0]


def assert_test_database(dsn: str, *, allow_override: bool = True) -> None:
    """Raise ``RuntimeError`` unless ``dsn`` names a test database.

    Args:
        dsn: The database URL the test suite is about to run destructive
            DDL/DML against.
        allow_override: When True, ``W2V_ALLOW_NON_TEST_DATABASE=1`` in the
            environment bypasses the check.

    Raises:
        RuntimeError: The DSN's database name does not end with ``_ci`` or
            ``_test``. The message names the database only — never the DSN,
            the userinfo or the password.
    """
    if allow_override and os.environ.get(OVERRIDE_ENV_VAR) == "1":
        return

    dbname = database_name_from_dsn(dsn)
    if dbname.endswith(TEST_DB_SUFFIXES):
        return

    shown = dbname if dbname else "<none: the DSN has no database path>"
    raise RuntimeError(
        f"Refusing to run the test suite against database {shown!r}: its name "
        f"does not end with {' or '.join(TEST_DB_SUFFIXES)}. This suite runs "
        "DROP TABLE ... CASCADE, Base.metadata.drop_all and TRUNCATE TABLE ... "
        "RESTART IDENTITY CASCADE against whatever DATABASE_URL points at, and "
        "per-run schema isolation is DISABLED under CI=true / "
        "GITHUB_ACTIONS=true — so this would destroy that database's public "
        "schema. Precedent: orchestrator #878, 'CI tests were WRITING TO "
        "PRODUCTION'. Point DATABASE_URL at a database whose name ends with "
        f"{' or '.join(TEST_DB_SUFFIXES)} (CI and the local docker-compose "
        "default both use 'w2v_orchestrator_ci'). If this database really is "
        f"disposable and you mean it, set {OVERRIDE_ENV_VAR}=1. "
        "(The DSN is withheld from this message on purpose: it carries "
        "credentials.)"
    )
