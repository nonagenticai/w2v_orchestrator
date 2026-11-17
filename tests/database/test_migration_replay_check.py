"""From-scratch migration replay + model-parity gate.

Proves the whole thing a fresh deploy / disaster-recovery / new-env bring-up
depends on: the Alembic chain replays cleanly from an EMPTY database to ``head``,
and the schema it produces matches the ORM models exactly.

Why this exists ALONGSIDE ``test_schema_parity``: that test compares the RLS
policy + SECURITY DEFINER resolver-function surface between a ``create_all`` build
and an ``alembic upgrade head`` build (the ``after_create`` registrar drift risk).
It does NOT assert that the *rest* of the schema — every table, column, type,
index, constraint, server default — reproduces the models. ``alembic check``
(autogenerate-produces-no-operations) closes exactly that gap, and does it against
a from-base replay so a migration that silently diverged from ``models.py`` (a
column added to a model but never migrated, or the reverse) fails HERE, at PR
time, instead of as a prod-only surprise.

It also guards the live-generator freeze class: several historical migrations
inline frozen literal DDL (``0030``/``0031``/``0035``/``0054``) precisely so a
later edit to ``src.database.rls`` can't retroactively rewrite their replayed
output. If a frozen snapshot ever drifts from what the current schema needs, the
replayed schema stops matching the models and ``alembic check`` goes red.

Build strategy — a throwaway *database* (not a schema), so ``alembic_version`` and
``command.check`` behave exactly as they do in prod. ``command.upgrade`` drives its
own async engine via ``env.py`` (which reads ``DATABASE_URL`` from the env); we
point that at the throwaway DB, run the chain, run ``check``, then drop the DB in a
``finally``. Gated on DB availability the same way the other DB tests are — an
unreachable ``TEST_DATABASE_URL`` FAILS via ``_fail_or_skip_without_database``
(``W2V_SKIP_DB_TESTS=1`` is the explicit opt-out), because a proof that never ran
must never look like a proof that passed.
"""

from __future__ import annotations

import os
import time
import warnings
from typing import cast

import pytest
from alembic import command
from alembic.config import Config
from psycopg.errors import QueryCanceled
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import OperationalError

from tests.conftest import TEST_DATABASE_URL, _fail_or_skip_without_database
from tests.support import run_isolation

# A throwaway DB named for the owning process, so a LOCAL parallel run never
# collides. It is NOT what the CI flake is about: CI gives each job its own
# ephemeral Postgres container and runs one pytest process (no xdist), so there
# is no second tenant for this name to collide WITH. That theory was investigated
# and refuted — see `_drop_probe_db` for what is and is not established.
_PROBE_DB = f"migration_replay_probe_{os.getpid()}"


def _probe_url() -> str:
    server = TEST_DATABASE_URL.rsplit("/", 1)[0]
    return f"{server}/{_PROBE_DB}"


def _admin_engine():
    # CREATE/DROP DATABASE is a server-level op that just needs SOME connection in
    # AUTOCOMMIT (it cannot run inside a transaction). Connect to the TEST DB itself
    # — it is guaranteed to exist and be reachable (conftest uses it) — rather than a
    # ``/postgres`` maintenance DB that a hardened server might not expose. Creating
    # the throwaway DB does not touch the test DB's data.
    return create_engine(TEST_DATABASE_URL, isolation_level="AUTOCOMMIT")


#: Teardown must not inherit CI's ambient `statement_timeout`. The archetype
#: exports `PGOPTIONS=-c statement_timeout=300000` (gitops `python-service.yml`)
#: for the whole job, so a blocked DROP sat there for FIVE MINUTES before the
#: suite failed — burning a 13-minute CI job to report a statement that takes
#: 27ms when it is not blocked. 30s is ~1000x a healthy drop and turns a hang
#: into a fast, legible failure. Worst case across all retries is
#: `_DROP_MAX_ATTEMPTS * (30s + _DROP_RETRY_SLEEP_SECONDS)` ≈ 96s, versus the
#: 300s a single inherited-timeout attempt used to burn.
_TEARDOWN_STATEMENT_TIMEOUT_MS = 30_000

#: Three attempts, because the observed stall is transient (runner disk/IO), not
#: a lock-holder — see `_drop_probe_db`. One retry could land inside the same
#: stall; three spread ~96s of wall clock across it. More would trade CI minutes
#: for a cleanup we have already decided is non-fatal.
_DROP_MAX_ATTEMPTS = 3

#: Breathing room between attempts. Long enough for a brief IO stall to clear,
#: short enough to be noise against a 475s suite.
_DROP_RETRY_SLEEP_SECONDS = 2.0


#: Diagnostics must not inherit the teardown timeout either — a 30s hang while
#: trying to explain a hang tells us nothing, and it would be paid once PER
#: retry. These are three tiny catalog reads; 5s is generous.
_DIAGNOSTIC_STATEMENT_TIMEOUT_MS = 5_000

#: What we ask Postgres when the DROP is blocked. Each entry is (label, SQL).
#: - `pg_stat_activity` filtered to the probe DB: if this returns ROWS there
#:   really is a live backend attached and we learn WHO (application_name),
#:   since when (backend_start), and what it is waiting on. `WITH (FORCE)` should
#:   have terminated exactly these, so rows here mean FORCE did not do its job.
#: - `pg_prepared_xacts` (whole table — it is normally EMPTY, and tiny when not):
#:   a prepared transaction (2PC) holds its locks with NO backend behind it, so
#:   `pg_terminate_backend` and `WITH (FORCE)` are both powerless against it and
#:   the DROP blocks forever. Rows here would BE the answer.
#: - `pg_locks`: we take ungranted locks server-wide rather than joining
#:   `pg_database` on the probe DB's oid. Two reasons: the blocked DROP is itself
#:   an ungranted lock waiter so it shows up regardless, and `l.database` on a
#:   database-level lock is the *target* db oid while the probe DB may already be
#:   gone from `pg_database` by the time we look — the join would then silently
#:   return nothing. Ungranted-only keeps the output small on a busy server.
_TEARDOWN_DIAGNOSTICS: tuple[tuple[str, str], ...] = (
    (
        "pg_stat_activity (backends attached to the probe DB)",
        "SELECT pid, state, application_name, backend_start, state_change, "
        "wait_event_type, wait_event, left(query, 200) AS query "
        "FROM pg_stat_activity WHERE datname = :db",
    ),
    (
        "pg_prepared_xacts (2PC — locks with no backend to terminate)",
        "SELECT gid, prepared, owner, database FROM pg_prepared_xacts",
    ),
    (
        "pg_locks (ungranted — who is waiting on what)",
        "SELECT locktype, database, relation, pid, mode, granted FROM pg_locks WHERE NOT granted",
    ),
)


def _print_drop_diagnostics(conn: Connection, attempt: int, max_attempts: int) -> None:
    """Print WHY the DROP was blocked. Must never raise — see the caller.

    Runs on the SAME connection as the failed DROP, which is connected to the
    TEST database (see `_admin_engine`), never to the probe DB, so it is not
    itself among the blocked/blocking backends. The engine is AUTOCOMMIT, so a
    statement that errors does not poison the ones after it, and `QueryCanceled`
    is not a disconnect — SQLAlchemy leaves the connection usable.

    `print()` on purpose: pytest captures stdout and replays it under the failing
    test, so this lands in the CI log exactly where the next reader will look.
    No logging config to get wrong, no dependency to add.
    """
    print(
        f"\n=== DROP DATABASE {_PROBE_DB} was blocked "
        f"(attempt {attempt}/{max_attempts}) — teardown diagnostics ==="
    )
    try:
        conn.execute(text(f"SET statement_timeout = {_DIAGNOSTIC_STATEMENT_TIMEOUT_MS}"))
    except Exception as exc:  # noqa: BLE001 — diagnostics are best-effort
        print(f"[diagnostic] could not lower statement_timeout: {type(exc).__name__}: {exc}")
    for label, sql in _TEARDOWN_DIAGNOSTICS:
        print(f"--- {label} ---")
        try:
            rows = conn.execute(text(sql), {"db": _PROBE_DB}).mappings().all()
        except Exception as exc:  # noqa: BLE001 — a dead probe must not mask the real error
            print(f"[diagnostic] query FAILED: {type(exc).__name__}: {exc}")
            continue
        if not rows:
            print("(no rows)")
            continue
        for row in rows:
            print(dict(row))
    print("=== end teardown diagnostics ===")


#: Prefix of the warning emitted when cleanup gives up. Named so the test that
#: proves the non-fatal contract can assert on it, and so a log search finds it.
_LEAK_WARNING_PREFIX = "MIGRATION REPLAY PROBE DB LEAKED"


def _drop_probe_db(conn: Connection) -> bool:
    """Drop the throwaway replay DB; on failure WARN and return False — never raise.

    ⚠️⚠️ THIS IS DELIBERATELY NON-FATAL. DO NOT "FIX" IT BACK INTO A `raise`. ⚠️⚠️

    Why (measured, three consecutive `main` runs up to 2026-08-16): the replay
    test's own assertions — `alembic upgrade head` from base, then
    `alembic check` finding no drift — ALL PASSED, and then the suite went red
    on this one teardown statement:

        E sqlalchemy.exc.OperationalError: (psycopg.errors.QueryCanceled)
          canceling statement due to statement timeout
        ==== 1 failed, 5473 passed ====

    Because `Lint, types, tests` failed, `build-and-push` and `deploy` were
    SKIPPED and every merge to main landed UNDEPLOYED. One cleanup line was
    holding the whole fleet's deploy path hostage while 5473 real assertions,
    including this test's own, were green.

    And the blocker hypothesis is REFUTED, not merely unproven. The diagnostics
    added in #822 (kept below — they are what produced this evidence) FIRED on
    the latest failure and reported `pg_stat_activity`, `pg_prepared_xacts` and
    ungranted `pg_locks` ALL EMPTY: no backend attached to the probe DB, no
    prepared transaction, nothing waiting on a lock. The DROP burned the full
    300s inherited timeout with nothing visibly contending. That is the shape of
    a runner disk/IO stall during the server's file deletion, not a lock-holder
    bug there is a code fix for. So: do NOT add more lock hunting here.

    The contract, therefore:
      * bound each attempt with our OWN short `statement_timeout` so a stall
        cannot burn CI minutes (the ambient `PGOPTIONS` 300s must not apply);
      * retry a small bounded number of times, since the stall is transient;
      * if it still will not drop, print the diagnostics, emit a LOUD warning
        NAMING the leaked database so it can be reaped, and let the test PASS —
        its assertions already succeeded, and a teardown that cannot clean up
        proves nothing about the migration chain.

    A leftover `migration_replay_probe_<pid>` database is acceptable fallout:
    it is a throwaway, the next run picks a fresh PID-keyed name, and CI's
    Postgres is an ephemeral per-job container that is discarded anyway.

    `WITH (FORCE)` (PG13+) terminates other backends and drops in one statement,
    closing the sweep/DROP race — keep it. It did NOT fix the flake, but it
    removes one class of cause for free.

    Two older theories are also dead, and stay dead: "CI contention between
    parallel Postgres suites" and "PID-keyed name collision". CI gives each job a
    PRIVATE ephemeral Postgres sidecar and there is no xdist in `pyproject.toml`,
    so there is exactly one pytest process and no second tenant on that server.

    Returns True if the database is gone (or was never there), False if cleanup
    was abandoned and the DB may still exist.
    """
    last_error: OperationalError | None = None
    for attempt in range(1, _DROP_MAX_ATTEMPTS + 1):
        # Re-set every attempt on purpose: `_print_drop_diagnostics` lowers this
        # session's `statement_timeout` to its own 5s budget, and the engine is
        # AUTOCOMMIT so that change is session-wide and sticky. Without this the
        # second attempt would silently run under the diagnostics' timeout.
        conn.execute(text(f"SET statement_timeout = {_TEARDOWN_STATEMENT_TIMEOUT_MS}"))
        try:
            conn.execute(text(f"DROP DATABASE IF EXISTS {_PROBE_DB} WITH (FORCE)"))
        except OperationalError as exc:
            # `psycopg.errors.QueryCanceled` (statement timeout) arrives wrapped
            # in SQLAlchemy's OperationalError. Capture the server-side state on
            # EVERY failed attempt — evidence taken at the moment of the stall is
            # what refuted the lock-holder theory, and it is cheap (three catalog
            # reads under a 5s cap).
            last_error = exc
            _print_drop_diagnostics(conn, attempt, _DROP_MAX_ATTEMPTS)
            if attempt < _DROP_MAX_ATTEMPTS:
                time.sleep(_DROP_RETRY_SLEEP_SECONDS)
            continue
        return True

    message = (
        f"{_LEAK_WARNING_PREFIX}: could not DROP DATABASE {_PROBE_DB} after "
        f"{_DROP_MAX_ATTEMPTS} attempts at {_TEARDOWN_STATEMENT_TIMEOUT_MS}ms each. "
        f"The replay assertions ALREADY PASSED, and the diagnostics above show no "
        f"backend, no prepared transaction and no ungranted lock — so this is "
        f"cleanup fallout, not a migration failure, and it is deliberately NOT "
        f"failing the suite (see _drop_probe_db). Reap it manually if it persists: "
        f"DROP DATABASE IF EXISTS {_PROBE_DB} WITH (FORCE);  Last error: "
        f"{type(last_error).__name__}: {last_error}"
    )
    # Both channels on purpose: `print` lands in pytest's captured stdout right
    # under the test (where the previous reader of this failure was looking), and
    # `warnings.warn` puts it in the terminal's warnings summary at the very end
    # of the run, where a green suite's only anomaly is easy to spot.
    print(f"\n*** {message}")
    warnings.warn(message, stacklevel=2)
    return False


def _require_database() -> None:
    """An unreachable test DB FAILS here (conftest policy) — it does not skip.

    This module used to call ``pytest.skip`` on its own, bypassing the suite-wide
    policy in ``tests/conftest.py``. With the DB down, the fresh-deploy /
    disaster-recovery / schema-drift proof below evaporated while pytest still
    exited 0. Route through ``_fail_or_skip_without_database`` so it fails loudly
    and the only way to skip is the deliberate, visible ``W2V_SKIP_DB_TESTS=1``
    operator opt-out.
    """
    engine = create_engine(TEST_DATABASE_URL)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:
        _fail_or_skip_without_database(exc)
    finally:
        engine.dispose()


def test_migration_chain_replays_from_scratch_and_matches_models() -> None:
    """`alembic upgrade head` from an EMPTY database succeeds AND the replayed
    schema matches the ORM models (``alembic check`` reports no operations).

    A failure means one of: a migration errors on a fresh replay; the migration
    chain and ``models.py`` have diverged (a model change never migrated, or a
    migration the models don't reflect); or a frozen historical migration's inline
    DDL no longer produces the schema the models describe. All three are
    fresh-deploy / disaster-recovery breakers that are otherwise invisible until a
    real bring-up fails.
    """
    _require_database()

    admin = _admin_engine()
    with admin.connect() as conn:
        # The PRE-create drop is non-fatal for the same reason the teardown one
        # is, and it costs no safety: this test's precondition is "the probe DB
        # is empty", and the `CREATE DATABASE` on the next line is what actually
        # enforces it. If a previous run leaked the name and it still cannot be
        # dropped, CREATE fails with "already exists" and this test goes red
        # HERE — before any assertion — which is correct and legible.
        _drop_probe_db(conn)
        conn.execute(text(f"CREATE DATABASE {_PROBE_DB}"))
    admin.dispose()

    cfg = Config()
    cfg.set_main_option("script_location", "src/database/migrations")
    prev_db_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = _probe_url()
    # The throwaway DB is a fresh `public`-only database. This process' global
    # PGOPTIONS pins search_path at the run's private schema in the TEST db (see
    # tests/support/run_isolation.py), which does not exist here — unqualified
    # migration DDL would fail with "no schema has been selected to create in".
    # Point it at `public` for the replay, keeping the run's application_name.
    prev_pgoptions = os.environ.get("PGOPTIONS")
    if prev_pgoptions is not None:
        os.environ["PGOPTIONS"] = run_isolation.pgoptions("public")
    try:
        # 1) Replay the whole chain from base. Any migration erroring here fails.
        command.upgrade(cfg, "head")
        # 2) The replayed schema must reproduce the models exactly. `command.check`
        # raises AutogenerateDiffsDetected (or another error) if autogenerate would
        # emit ANY operation — i.e. a table/column/type/index/default/constraint in
        # the models that the migrations don't build, or vice versa.
        try:
            command.check(cfg)
        except Exception as exc:  # noqa: BLE001 — surface the diff as a clear failure
            pytest.fail(
                "alembic check found drift between the from-scratch migration replay "
                "and the ORM models — the migration chain and models.py have diverged, "
                "or a frozen historical migration's DDL no longer matches the models:\n"
                f"{type(exc).__name__}: {exc}"
            )
    finally:
        if prev_db_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = prev_db_url
        if prev_pgoptions is not None:
            os.environ["PGOPTIONS"] = prev_pgoptions
        admin = _admin_engine()
        with admin.connect() as conn:
            # Non-fatal BY DESIGN. Everything this test asserts has already run
            # by the time we get here; a cleanup that cannot complete must not
            # turn a green proof red (and, via the skipped deploy job, block
            # every merge). `_drop_probe_db` warns loudly and names the leak.
            _drop_probe_db(conn)
        admin.dispose()


# --------------------------------------------------------------------------
# The non-fatal-teardown contract itself.
#
# These need NO database: they drive `_drop_probe_db` against a fake connection
# that reproduces the exact CI failure (a `QueryCanceled` on the DROP, with all
# three diagnostics empty). They exist so that anyone who "tidies up" the
# swallowed exception back into a `raise` gets an immediate red test explaining
# why the swallow is load-bearing — see `_drop_probe_db`'s docstring.
# --------------------------------------------------------------------------


class _EmptyResult:
    """Minimal stand-in for a SQLAlchemy result that yielded no rows."""

    def mappings(self) -> _EmptyResult:
        return self

    def all(self) -> list[dict[str, object]]:
        return []


class _StallingConnection:
    """A connection whose `DROP DATABASE` always times out, like CI's.

    Every other statement succeeds and every diagnostic returns "(no rows)" —
    exactly the evidence the #822 diagnostics reported on the real failure
    (no backend, no 2PC, no ungranted lock).
    """

    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(
        self, statement: object, parameters: dict[str, object] | None = None
    ) -> _EmptyResult:
        sql = str(statement)
        self.statements.append(sql)
        if sql.startswith("DROP DATABASE"):
            raise OperationalError(
                sql,
                parameters,
                QueryCanceled("canceling statement due to statement timeout"),
            )
        return _EmptyResult()


def test_drop_probe_db_timeout_is_non_fatal_and_warns_about_the_leak(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A DROP that keeps timing out must NOT propagate — it must warn and yield.

    This is the whole point of the design: the replay assertions have already
    passed by the time teardown runs, and a red `Lint, types, tests` job skips
    `build-and-push` and `deploy`, so a cleanup failure was landing every merge
    to main UNDEPLOYED. Assert the swallow, the retries, the bounded timeout,
    the preserved diagnostics, and the leak-naming warning.
    """
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    fake = _StallingConnection()

    # `catch_warnings(record=True)` rather than `pytest.warns`, deliberately: if
    # someone turns the swallow back into a `raise`, the OperationalError has to
    # surface as the test's error verbatim, not get reshaped into "DID NOT WARN".
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        dropped = _drop_probe_db(cast("Connection", fake))

    # 1) It did not raise, and it told the truth about not having cleaned up.
    assert dropped is False

    caught = [w for w in recorded if _LEAK_WARNING_PREFIX in str(w.message)]
    assert caught, f"no {_LEAK_WARNING_PREFIX} warning; got {[str(w.message) for w in recorded]}"

    # 2) It really tried, a BOUNDED number of times, under OUR short timeout —
    #    never CI's ambient 300s `PGOPTIONS` one.
    drops = [s for s in fake.statements if s.startswith("DROP DATABASE")]
    assert len(drops) == _DROP_MAX_ATTEMPTS
    timeouts = [s for s in fake.statements if s.startswith("SET statement_timeout")]
    assert timeouts.count(f"SET statement_timeout = {_TEARDOWN_STATEMENT_TIMEOUT_MS}") == (
        _DROP_MAX_ATTEMPTS
    )
    assert _TEARDOWN_STATEMENT_TIMEOUT_MS <= 60_000

    # 3) The warning NAMES the leaked database so it can be reaped.
    message = str(caught[0].message)
    assert _PROBE_DB in message
    assert "QueryCanceled" in message

    # 4) The #822 diagnostics are still emitted on the give-up path — they are
    #    what proved there was no lock-holder, and must survive this change.
    out = capsys.readouterr().out
    assert "pg_stat_activity" in out
    assert "pg_prepared_xacts" in out
    assert "pg_locks" in out
    assert "(no rows)" in out
    assert _PROBE_DB in out


def test_drop_probe_db_returns_true_when_the_drop_succeeds() -> None:
    """The happy path still drops on the first attempt and reports success."""

    class _WorkingConnection(_StallingConnection):
        def execute(
            self, statement: object, parameters: dict[str, object] | None = None
        ) -> _EmptyResult:
            self.statements.append(str(statement))
            return _EmptyResult()

    fake = _WorkingConnection()
    assert _drop_probe_db(cast("Connection", fake)) is True
    assert len([s for s in fake.statements if s.startswith("DROP DATABASE")]) == 1
