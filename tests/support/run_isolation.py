"""Per-run schema isolation for the shared local test database.

The problem this solves
----------------------
Every worktree of this repo points ``DATABASE_URL`` at the SAME container
(``w2v_orchestrator_test_postgres``, ``127.0.0.1:5452``, db
``w2v_orchestrator_ci``) and every run builds its tables in that database's
``public`` schema. Two concurrent runs then corrupt each other two ways:

1. ``tests/conftest.py``'s per-test ``TRUNCATE`` wipes the *other* run's fixture
   rows mid-test.
2. The ``pg_terminate_backend`` sweep that guards against a leaked SSE backend
   holding a lock reaps every ``idle in transaction`` backend in the database.
   A concurrent run's connection sitting between two statements of its own
   fixture is ``idle in transaction`` too — it is not a leak, it is another run
   working normally. It dies with *"terminating connection due to administrator
   command"*, which surfaces as ``connection refused``-shaped noise against a
   container that is healthy the whole time.

The mechanism
-------------
Each pytest process claims its own postgres SCHEMA and pins every libpq
connection it makes to that schema via ``PGOPTIONS``. Tables, ``TRUNCATE``
scope and lock scope all become private to the run. The same connections also
carry a per-run ``application_name``, which is what lets the termination sweep
be narrowed to *this run's own* backends (see ``terminate_scope_sql``).

Why ``PGOPTIONS`` and not ``DATABASE_URL?options=-csearch_path%3D...``
---------------------------------------------------------------------
This estate already burned that path. gitops-production
``79f59692e`` ("fix(archetype): pass schema search_path via PGOPTIONS env, not
URL options") reverted URL-encoded options because alembic's ``env.py`` and
every ``BaseSettings`` consumer read ``DATABASE_URL`` through ``ConfigParser``,
which interprets ``%2``/``%3`` as interpolation syntax and raises
``ValueError: invalid interpolation syntax``. ``PGOPTIONS`` is libpq's own
channel: psycopg (sync and async), alembic and psql all honour it at connect
time and nobody has to parse the URL. Keeping ``DATABASE_URL`` clean also means
the string comparisons other test modules do on it (e.g.
``TEST_DATABASE_URL.rsplit("/", 1)``) keep working.

Why schema-per-run and not database-per-run
-------------------------------------------
* ``CREATE DATABASE`` needs a server-level privilege that a developer connected
  as a restricted role does not necessarily have; ``CREATE SCHEMA`` only needs
  ``CREATE`` on the database we are already using.
* Teardown reliability: ``DROP DATABASE`` fails outright ("database is being
  accessed by other users") if a single leaked backend survives — which is
  exactly the failure mode this suite already documents. ``DROP SCHEMA ...
  CASCADE`` takes ordinary object locks, so it can be bounded with
  ``lock_timeout`` and retried by the GC instead of wedging teardown.
* It is the shape CI itself used in its shared-instance era, and the shape the
  sibling repos use per xdist worker.

CI is deliberately NOT isolated
-------------------------------
CI gets a per-job EPHEMERAL postgres sidecar (gitops-production
``python-service.yml``; ``postgres_isolation`` is documented there as a no-op
today), so there is exactly one run against a private database — isolation buys
nothing and any divergence would be pure risk. ``isolation_enabled()`` returns
False under CI, which restores the pre-existing code path byte for byte.
``W2V_TEST_SCHEMA_ISOLATION=1``/``=0`` overrides the default either way.

Crash-safe reclamation
----------------------
A run drops its own schema at interpreter exit. That covers a normal exit, a
``KeyboardInterrupt`` and an unhandled crash, but not ``SIGKILL`` / a killed
container. So every run also GCs on entry: any ``w2v_t_*`` schema whose
``created_at`` comment is older than ``W2V_TEST_SCHEMA_TTL_SECONDS`` (default 4h)
AND which has no live backend carrying its ``application_name`` is dropped.
The liveness half is exact rather than heuristic because each run holds one
long-lived "beacon" connection for its whole session — so a run that is merely
*slow* (or parked between tests with all its pooled engines disposed) is never
mistaken for a dead one.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import secrets
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg
    import psycopg.sql

# Prefix is the GC's namespace. Anything matching it is owned by this mechanism
# and reclaimable; nothing else in the database is ever touched.
SCHEMA_PREFIX = "w2v_t_"
APP_NAME_PREFIX = "w2v_test_"

# PID alone is not unique enough: PIDs are recycled, and a stale schema from a
# crashed run could collide with a live one. PID keeps the name diagnosable
# (`ps` it), the random suffix makes it unique.
RUN_TOKEN = f"{os.getpid()}_{secrets.token_hex(4)}"
RUN_SCHEMA = f"{SCHEMA_PREFIX}{RUN_TOKEN}"
RUN_APP_NAME = f"{APP_NAME_PREFIX}{RUN_TOKEN}"

_DEFAULT_TTL_SECONDS = 4 * 60 * 60

# The single long-lived connection that advertises "this run is alive" to other
# runs' GC sweeps. Opened once by ``ensure_run_schema``.
_beacon: psycopg.Connection[tuple[object, ...]] | None = None
_schema_ready = False


def isolation_enabled() -> bool:
    """True when this process should own a private schema.

    Off under CI (one run, private ephemeral database — see module docstring),
    on everywhere else. ``W2V_TEST_SCHEMA_ISOLATION`` forces either answer.
    """
    override = os.environ.get("W2V_TEST_SCHEMA_ISOLATION")
    if override in {"0", "1"}:
        return override == "1"
    return not (os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true")


def scoped_schema(base: str) -> str:
    """Make a fixed throwaway-schema name unique to this run.

    The suite builds several *named* throwaway schemas beyond the main one
    (``parity_create_all``, ``parity_alembic``, ``panel_uq_migtest``). Each is
    created with ``DROP SCHEMA IF EXISTS ... CASCADE`` first, so under
    concurrency one run destroys the other's half-built probe. Routing those
    names through here gives each run its own copy. Unchanged under CI.
    """
    if not isolation_enabled():
        return base
    return f"{base}_{RUN_TOKEN}"


def pgoptions(schema: str) -> str:
    """The ``PGOPTIONS`` value pinning libpq connections to ``schema``.

    ``application_name`` rides along so ``pg_stat_activity`` can attribute a
    backend to this run — used both by the narrowed termination sweep and by
    the GC's liveness check.
    """
    return f'-c search_path="{schema}" -c application_name={RUN_APP_NAME}'


def install_pgoptions() -> None:
    """Point every libpq connection this process makes at the run's schema.

    Called at conftest import time, BEFORE anything connects. Pure env-var
    mutation: no I/O, so a developer box with no postgres running is unaffected
    and still gets the existing loud "test database unreachable" failure.
    """
    if isolation_enabled():
        os.environ["PGOPTIONS"] = pgoptions(RUN_SCHEMA)


def terminate_scope_sql() -> str:
    """The ``pg_terminate_backend`` sweep, scoped to backends this run owns.

    The sweep itself is still needed and must not be deleted: ``src/api/sse.py``
    can still leak an un-awaited stream task whose event loop closes
    mid-transaction, leaving a server-side backend alive holding a row/table
    lock. ``TRUNCATE ... CASCADE`` needs ACCESS EXCLUSIVE and would block on it
    forever, hanging the whole suite. What was wrong was its *scope*, not its
    existence: ``datname = current_database()`` also matches every concurrently
    running suite, whose fixtures are legitimately ``idle in transaction``
    between two of their own statements.

    Under isolation the filter gains ``application_name = <this run>``, so it
    can only ever reap backends this process opened. Under CI (isolation off)
    the SQL is byte-identical to what it always was — there is only one run
    against a private ephemeral database, so database-wide scope is already
    self-scoped.
    """
    sql = (
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        "WHERE datname = current_database() "
        "AND pid <> pg_backend_pid() "
        "AND state IN ('idle in transaction', 'idle in transaction (aborted)')"
    )
    if isolation_enabled():
        sql += f" AND application_name = '{RUN_APP_NAME}'"
    return sql


def libpq_dsn(sqlalchemy_url: str) -> str:
    """Strip SQLAlchemy's ``+driver`` suffix so ``psycopg.connect`` accepts it."""
    for prefix in ("postgresql+psycopg://", "postgres+psycopg://"):
        if sqlalchemy_url.startswith(prefix):
            return "postgresql://" + sqlalchemy_url[len(prefix) :]
    return sqlalchemy_url


def _created_at_comment() -> str:
    """Schema comment carrying the creation time, for the age half of the GC.

    Same idiom as the archetype's ``postgres-ci-gc`` (``created_at=<iso8601>``)
    so the two are readable by the same eye.
    """
    return f"created_at={datetime.now(UTC).isoformat()}"


def _ttl() -> timedelta:
    raw = os.environ.get("W2V_TEST_SCHEMA_TTL_SECONDS")
    seconds = int(raw) if raw and raw.isdigit() else _DEFAULT_TTL_SECONDS
    return timedelta(seconds=seconds)


def _sweep_stale_schemas(conn: psycopg.Connection[tuple[object, ...]]) -> list[str]:
    """Drop abandoned run schemas. Returns the names actually dropped.

    A schema is abandoned only if BOTH hold:

    * its ``created_at`` comment is older than the TTL (a schema with no
      readable comment is treated as ancient — it means a run died between
      ``CREATE SCHEMA`` and ``COMMENT``), and
    * no backend in this database carries its ``application_name``, i.e. its
      beacon connection is gone.

    Requiring both means a long, slow run is never reclaimed out from under
    itself, and a ``SIGKILL``ed one is reclaimed by the next run that starts.
    """
    cutoff = datetime.now(UTC) - _ttl()
    dropped: list[str] = []
    with conn.cursor() as cur:
        # Session-level, not SET LOCAL: the beacon is autocommit, so there is no
        # surrounding transaction for a LOCAL setting to belong to.
        cur.execute("SET lock_timeout = '5s'")
        cur.execute(
            "SELECT n.nspname, obj_description(n.oid, 'pg_namespace') "
            "FROM pg_namespace n WHERE n.nspname LIKE %s",
            (f"{SCHEMA_PREFIX}%",),
        )
        candidates = [(str(row[0]), row[1]) for row in cur.fetchall()]

    for name, comment in candidates:
        if name == RUN_SCHEMA:
            continue
        if not _is_older_than(comment, cutoff):
            continue
        token = name[len(SCHEMA_PREFIX) :]
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_stat_activity "
                "WHERE datname = current_database() AND application_name = %s LIMIT 1",
                (f"{APP_NAME_PREFIX}{token}",),
            )
            if cur.fetchone() is not None:
                continue  # owner still alive — slow, not dead
        try:
            with conn.cursor() as cur:
                cur.execute(_drop_schema_sql(name))
            dropped.append(name)
        except Exception:  # noqa: BLE001 -- GC is best effort; the next run retries
            conn.rollback()
    return dropped


def _drop_schema_sql(name: str) -> psycopg.sql.Composed:
    """``DROP SCHEMA IF EXISTS <name> CASCADE`` with the name properly quoted."""
    from psycopg import sql

    return sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(name))


def _is_older_than(comment: object, cutoff: datetime) -> bool:
    """True when ``comment`` encodes a ``created_at`` at or before ``cutoff``.

    An unparseable/absent comment counts as older: it can only come from a run
    that died between ``CREATE SCHEMA`` and ``COMMENT ON SCHEMA``, and such a
    schema is empty anyway.
    """
    if not isinstance(comment, str) or not comment.startswith("created_at="):
        return True
    try:
        created = datetime.fromisoformat(comment.removeprefix("created_at="))
    except ValueError:
        return True
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return created <= cutoff


def ensure_run_schema(sqlalchemy_url: str) -> None:
    """Create this run's schema (idempotent), GC abandoned ones, hold the beacon.

    Called lazily from the DB fixtures rather than at import, so a process that
    never touches the database never opens a connection.

    Raises whatever ``psycopg.connect`` raises when the database is unreachable
    — callers route that into the suite's existing loud
    ``_fail_or_skip_without_database`` policy.
    """
    global _beacon, _schema_ready
    if not isolation_enabled() or _schema_ready:
        return

    import psycopg
    from psycopg import sql

    conn: psycopg.Connection[tuple[object, ...]] = psycopg.connect(
        libpq_dsn(sqlalchemy_url), autocommit=True
    )
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(RUN_SCHEMA)))
        cur.execute(
            sql.SQL("COMMENT ON SCHEMA {} IS {}").format(
                sql.Identifier(RUN_SCHEMA), sql.Literal(_created_at_comment())
            )
        )
    _sweep_stale_schemas(conn)
    # This connection stays open for the whole session: it is the run's liveness
    # beacon (it carries RUN_APP_NAME via PGOPTIONS). It is autocommit, so it
    # parks as `idle`, never `idle in transaction` — no sweep, ours or another
    # run's, can mistake it for a leak.
    _beacon = conn
    _schema_ready = True
    atexit.register(_release)


def _release() -> None:
    """Drop this run's schema and close the beacon. Registered via ``atexit``.

    Best effort by design: if a leaked backend still holds a lock the DROP times
    out rather than hanging the exit, and the schema is left for another run's
    GC to reclaim once the TTL elapses and the beacon is gone.
    """
    global _beacon, _schema_ready
    conn = _beacon
    _beacon = None
    _schema_ready = False
    if conn is None:
        return
    # Both suppressions are deliberate: an interpreter already on its way out
    # must not be turned into a nonzero exit by a teardown hiccup. Anything left
    # behind is reclaimed by the next run's GC.
    try:
        with contextlib.suppress(Exception), conn.cursor() as cur:
            cur.execute("SET lock_timeout = '10s'")
            cur.execute(_drop_schema_sql(RUN_SCHEMA))
    finally:
        with contextlib.suppress(Exception):
            conn.close()
