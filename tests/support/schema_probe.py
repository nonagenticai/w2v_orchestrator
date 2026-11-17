"""Dual-schema probe: build the SAME database twice — once from
``Base.metadata.create_all`` and once by replaying the Alembic chain — into two
throwaway postgres schemas, side by side, so a test can compare the two
authorities directly.

Why two builds and not one: the test suite's own database is built by
``tests/conftest.py``'s ``_db_schema_initialized`` with ``create_all``, i.e. from
``src/database/models.py``. Anything a test reads out of ``pg_constraint`` there
is therefore a *reflection of the models*, not of what prod runs. Prod is built
by ``alembic upgrade head``. Every guard that wants to make a claim about the
PRODUCTION schema has to read the replayed build, and every guard that wants to
claim the test schema is representative has to compare the two.

This module owns only the mechanics (build, drop, introspect). Availability
gating and the session-scoped single-build fixture live in ``tests/conftest.py``
(``parity_schemas``) so the two expensive builds happen exactly once per session
and are shared by every consumer:

* ``tests/database/test_schema_parity.py`` — RLS policy + SECURITY DEFINER
  resolver-function surface.
* ``tests/contracts/test_stage_status_check_constraint_parity.py`` — the
  ``ck_*`` CHECK-constraint surface + the ``StageStatus`` Literal.

Build strategy — two schemas in the ONE test database (not two databases), so no
``CREATE DATABASE`` privilege is needed and both builds are introspectable from a
single connection:

* ``parity_create_all`` — a connection with ``search_path`` set to it, then
  ``Base.metadata.create_all`` (which also fires the ``after_create`` registrars,
  so policies + functions land unqualified in this schema).
* ``parity_alembic`` — ``alembic upgrade head`` with ``PGOPTIONS`` pinning the
  libpq connection's ``search_path`` to it, so every unqualified migration DDL
  (including ``alembic_version``) lands there.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

from sqlalchemy import Connection, Engine, create_engine, text

from tests.support import run_isolation

_REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = _REPO_ROOT / "src" / "database" / "migrations"

# Both probe schemas start with `DROP SCHEMA IF EXISTS ... CASCADE`, so a fixed
# name means two concurrent local runs demolish each other's half-built probe.
# `scoped_schema` appends this run's token locally and leaves the name untouched
# under CI (one run, private database).
CREATE_ALL_SCHEMA = run_isolation.scoped_schema("parity_create_all")
ALEMBIC_SCHEMA = run_isolation.scoped_schema("parity_alembic")

# (table, policy) -> (cmd, USING predicate, WITH CHECK predicate)
PolicySnapshot = dict[tuple[str, str], tuple[str | None, str | None, str | None]]
# (proname, identity args) -> (SECURITY DEFINER, volatility, body)
FunctionSnapshot = dict[tuple[str, str], tuple[bool, str, str | None]]
# (table, constraint name) -> normalized CHECK predicate
CheckSnapshot = dict[tuple[str, str], str]


class ParityProbe(NamedTuple):
    """The two built schemas plus the engine to introspect them with.

    ``create_all`` is the models' schema (what the test suite runs against);
    ``alembic`` is the migration chain's schema (what PROD runs against).
    """

    engine: Engine
    create_all: str
    alembic: str


def sync_engine(url: str) -> Engine:
    """A synchronous psycopg engine on the test database.

    ``command.upgrade`` drives its own async engine internally via ``env.py``;
    this engine is only for the ``create_all`` build and the introspection
    snapshots, so a plain sync test with no event loop can orchestrate both
    builds deterministically.
    """
    return create_engine(url, poolclass=None)


def norm(sql: str | None, schema: str) -> str | None:
    """Drop ``schema``-qualification so predicates/bodies compare on shape only.

    Postgres renders stored predicates with schema-qualified table names
    (``parity_create_all.projects`` vs ``parity_alembic.projects``); stripping
    each build's own prefix leaves only the logic to compare.
    """
    if sql is None:
        return None
    return sql.replace(f"{schema}.", "").replace(f'"{schema}".', "")


def build_create_all(engine: Engine) -> None:
    from src.database.models import Base

    with engine.begin() as conn:
        conn.exec_driver_sql(f"DROP SCHEMA IF EXISTS {CREATE_ALL_SCHEMA} CASCADE")
        conn.exec_driver_sql(f"CREATE SCHEMA {CREATE_ALL_SCHEMA}")
        # Everything unqualified (tables, constraints, policies, SECURITY DEFINER
        # functions, and each function's ``SET search_path FROM CURRENT`` capture)
        # resolves into this schema for the life of the connection.
        conn.exec_driver_sql(f"SET search_path TO {CREATE_ALL_SCHEMA}")
        Base.metadata.create_all(conn)


def build_alembic(engine: Engine, database_url: str) -> None:
    from alembic import command
    from alembic.config import Config

    with engine.begin() as conn:
        conn.exec_driver_sql(f"DROP SCHEMA IF EXISTS {ALEMBIC_SCHEMA} CASCADE")
        conn.exec_driver_sql(f"CREATE SCHEMA {ALEMBIC_SCHEMA}")

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    # env.py reads DATABASE_URL from the environment; ``PGOPTIONS`` pins the
    # libpq connection's search_path so every unqualified migration DDL (incl.
    # alembic_version + the RLS policies/functions) lands in the alembic schema.
    prev_pgoptions = os.environ.get("PGOPTIONS")
    prev_db_url = os.environ.get("DATABASE_URL")
    # Route through the shared builder so the connection keeps this run's
    # `application_name` while its search_path points at the alembic probe.
    os.environ["PGOPTIONS"] = run_isolation.pgoptions(ALEMBIC_SCHEMA)
    os.environ["DATABASE_URL"] = database_url
    try:
        command.upgrade(cfg, "head")
    finally:
        if prev_pgoptions is None:
            os.environ.pop("PGOPTIONS", None)
        else:
            os.environ["PGOPTIONS"] = prev_pgoptions
        if prev_db_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = prev_db_url


def drop_probe_schemas(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(f"DROP SCHEMA IF EXISTS {CREATE_ALL_SCHEMA} CASCADE")
        conn.exec_driver_sql(f"DROP SCHEMA IF EXISTS {ALEMBIC_SCHEMA} CASCADE")


def snapshot_policies(conn: Connection, schema: str) -> PolicySnapshot:
    rows = conn.execute(
        text(
            "SELECT tablename, policyname, cmd, qual, with_check "
            "FROM pg_policies WHERE schemaname = :s "
            "ORDER BY tablename, policyname"
        ),
        {"s": schema},
    ).all()
    return {
        (r.tablename, r.policyname): (
            r.cmd,
            norm(r.qual, schema),
            norm(r.with_check, schema),
        )
        for r in rows
    }


def snapshot_functions(conn: Connection, schema: str) -> FunctionSnapshot:
    # Scope to SECURITY DEFINER functions: that is EXACTLY the registrar-emitted
    # resolver surface (``run_tenant``/``orphan_runs``/``admin_*``) — the drift
    # risk this snapshot exists for. It deliberately excludes noise that would
    # otherwise differ for reasons unrelated to the registrars: ``CREATE
    # EXTENSION pgcrypto`` functions (present only in the alembic build, and
    # SECURITY INVOKER) and the ``enforce_lesson_parent_is_course`` plpgsql
    # trigger fn (a SECURITY INVOKER migration artifact, not a resolver). The
    # filter stays bidirectional: a migration that adds a SECURITY DEFINER
    # resolver without its registrar (or the reverse) still diverges here.
    rows = conn.execute(
        text(
            "SELECT p.proname, "
            "pg_get_function_identity_arguments(p.oid) AS args, "
            "p.prosecdef, p.provolatile, p.prosrc "
            "FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
            "WHERE n.nspname = :s AND p.prosecdef "
            "ORDER BY p.proname, args"
        ),
        {"s": schema},
    ).all()
    return {(r.proname, r.args): (r.prosecdef, r.provolatile, norm(r.prosrc, schema)) for r in rows}


def snapshot_check_constraints(conn: Connection, schema: str) -> CheckSnapshot:
    """Every table CHECK constraint in ``schema``, keyed ``(table, name)``.

    ``contype = 'c'`` is table CHECKs only — postgres 17+ catalogues NOT NULL
    separately as ``contype = 'n'``, so column nullability (already covered by
    ``alembic check``) does not leak in here as noise.

    The value is the postgres-normalized predicate from
    ``pg_get_constraintdef``, NOT the source text, so source-level whitespace or
    ``IN (...)`` vs ``= ANY (ARRAY[...])`` spelling differences between a model
    and its migration compare equal — only a real difference in admitted values
    (or in the column type the predicate casts to) shows up.
    """
    rows = conn.execute(
        text(
            "SELECT c.relname AS table_name, con.conname, "
            "pg_get_constraintdef(con.oid) AS definition "
            "FROM pg_constraint con "
            "JOIN pg_class c ON c.oid = con.conrelid "
            "JOIN pg_namespace n ON n.oid = con.connamespace "
            "WHERE n.nspname = :s AND con.contype = 'c' "
            "ORDER BY c.relname, con.conname"
        ),
        {"s": schema},
    ).all()
    out: CheckSnapshot = {}
    for r in rows:
        definition = norm(r.definition, schema)
        assert definition is not None  # pg_get_constraintdef never returns NULL here
        out[(r.table_name, r.conname)] = definition
    return out
