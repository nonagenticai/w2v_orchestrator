"""Test fixtures for w2v_orchestrator."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from typing import NoReturn

from tests.support.test_db import resolve_test_database_url

# Resolve the test DB BEFORE any app imports, and export it as DATABASE_URL
# because the app reads that name. Precedence, and the reason there is no
# magic port any more, live in tests/support/test_db.py: TEST_DATABASE_URL >
# DATABASE_URL > an ephemeral postgres:18 this process starts and tears down >
# fail closed. There is deliberately no fixed port for a human to hand-start.
#
# Capture it into a module constant BEFORE the autouse _clear_settings fixture
# monkeypatches DATABASE_URL away.
TEST_DATABASE_URL = resolve_test_database_url()
os.environ["DATABASE_URL"] = TEST_DATABASE_URL

from tests.support.db_identity import (  # noqa: E402 -- must import after env setup above
    assert_test_database,
)

# Refuse to proceed unless that URL names a test database. Everything below
# runs DROP/TRUNCATE ... CASCADE against it, and the schema isolation installed
# next is a no-op under CI. See tests/support/db_identity.py (orchestrator #878).
assert_test_database(TEST_DATABASE_URL)

from tests.support import run_isolation  # noqa: E402 -- must import after env setup above

# Claim a private schema for THIS pytest process and pin every libpq connection
# it makes to it, before anything connects. The URL stays clean; the schema
# travels via PGOPTIONS (see tests/support/run_isolation.py for why, and for why
# this is a no-op under CI).
run_isolation.install_pgoptions()

import pytest  # noqa: E402 -- must import after env setup above
from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# pgbouncer caches backend-auth failures per-user under `server_login_retry`
# (default 15s). A single transient backend hiccup poisons CI for every
# concurrent ARC runner using ci_test until the cache expires. We can't fix
# that from this repo, but we can tolerate it: retry the first engine
# acquisition narrowly on the cached-login error string.
_LOGIN_RETRY_BACKOFFS_S = (1, 2, 3, 5, 8)


def _is_pgbouncer_login_retry(exc: BaseException) -> bool:
    """True iff `exc` is the pgbouncer `server_login_retry` cached-error flake.

    Narrow on purpose — any other OperationalError (wrong creds, wrong host,
    DB doesn't exist, postgres down) must fail fast so we don't mask real bugs.
    """
    return "server_login_retry" in str(getattr(exc, "orig", exc))


async def _wait_for_engine(engine: AsyncEngine) -> None:
    """Acquire+release one connection from `engine`, tolerating pgbouncer flakes.

    Used as a "ping" at fixture entry. Once pgbouncer has handed out a server
    connection the per-user auth cache is no longer in the path, so a single
    successful ping is enough to unblock the rest of the test session.
    """
    last_exc: OperationalError | None = None
    for attempt, backoff in enumerate((0, *_LOGIN_RETRY_BACKOFFS_S), start=1):
        if backoff:
            print(
                f"[conftest] pgbouncer server_login_retry flake — sleeping {backoff}s "
                f"before attempt {attempt}/{len(_LOGIN_RETRY_BACKOFFS_S) + 1}",
                flush=True,
            )
            await asyncio.sleep(backoff)
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return
        except OperationalError as exc:
            if not _is_pgbouncer_login_retry(exc):
                raise
            last_exc = exc
    # Exhausted retries on the pgbouncer cache flake — let the caller see
    # the real OperationalError so existing skip-on-OperationalError handlers
    # (e.g. dev-box without postgres.test) still fire.
    assert last_exc is not None  # loop ran at least once
    raise last_exc


def _running_in_ci() -> bool:
    """True when executing inside GitHub Actions (or any CI setting ``CI=true``).

    GitHub Actions sets both ``CI`` and ``GITHUB_ACTIONS`` in every job's
    environment automatically — no workflow wiring required.
    """
    return os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"


def require_createrole(exc: BaseException) -> NoReturn:
    """Handle a ``CREATE ROLE`` privilege failure in the RLS-privilege tests.

    The tenant-isolation tests forge a throwaway ``NOBYPASSRLS`` role and
    ``SET ROLE`` into it to prove the shipped RLS predicates isolate — the one
    thing the owner-bypass posture hides from every other test. That needs
    ``CREATEROLE`` on the connected DB user.

    - **Under CI** these tests MUST run: the ephemeral ``postgres:18`` CI
      container's ``ci_test`` user is a superuser (``POSTGRES_USER`` in the
      official image), so ``CREATE ROLE`` cannot legitimately fail. If it ever
      does — e.g. a future archetype change swaps CI onto a restricted role —
      ``fail`` loudly so tenant-isolation coverage can never silently vanish
      while CI stays green.
    - **Locally** a developer may be connected as the restricted runtime role
      (no ``CREATEROLE``); skip so the DB subset still passes for them.
    """
    reason = f"test DB user cannot CREATE ROLE (needs CREATEROLE): {getattr(exc, 'orig', exc)}"
    if _running_in_ci():
        pytest.fail(
            f"{reason}\nRLS isolation tests must execute under CI — the ephemeral "
            "postgres:18 ci_test user is a superuser and should have CREATEROLE. "
            "A silent skip here would let a cross-tenant regression ship green.",
            pytrace=False,
        )
    pytest.skip(reason)


def _ensure_run_schema() -> None:
    """Create this run's private schema, GC abandoned ones, hold the beacon.

    A no-op under CI. An unreachable database is routed into the suite-wide
    loud-failure policy rather than raising a psycopg error out of a fixture.
    """
    import psycopg

    try:
        run_isolation.ensure_run_schema(TEST_DATABASE_URL)
    except psycopg.OperationalError as exc:
        _fail_or_skip_without_database(exc)


def _fail_or_skip_without_database(exc: BaseException) -> NoReturn:
    """An unreachable test DB is a FAILURE, not a skip.

    This used to be an unconditional ``pytest.skip(..., allow_module_level=True)``.
    That is the most expensive kind of bug: with no database reachable, the whole
    DB-backed suite evaporated and pytest exited 0 — "43 skipped, 2 passed" reads
    exactly like a pass at a glance, and in CI-shaped output like a green run. A
    test that cannot run has proven nothing; it must never look like one that did.

    The same reasoning is already applied 20 lines up for CREATEROLE, which
    fails loudly rather than let tenant-isolation coverage vanish. Connectivity
    deserves the identical treatment: infrastructure that is misconfigured must
    be *loud*, because the failure mode of quiet infrastructure is a regression
    shipping green.

    Skipping remains available, but only as a deliberate, explicit choice:
    set ``W2V_SKIP_DB_TESTS=1``. Opting out is a decision a developer makes on
    purpose and can see in their own shell history — not a default the suite
    silently makes on their behalf.
    """
    detail = getattr(exc, "orig", exc)
    if os.environ.get("W2V_SKIP_DB_TESTS") == "1":
        pytest.skip(
            f"DB unavailable at {TEST_DATABASE_URL} and W2V_SKIP_DB_TESTS=1: {detail}",
            allow_module_level=True,
        )
    pytest.fail(
        f"Test database unreachable at {TEST_DATABASE_URL}: {detail}\n"
        "\n"
        "The DB-backed suite cannot run, so it must not report success.\n"
        "\n"
        "  Point it somewhere:  export TEST_DATABASE_URL=postgresql+psycopg://...\n"
        "  (or run with docker available and one is started for you)\n"
        "  Skip anyway (explicitly):  W2V_SKIP_DB_TESTS=1 uv run pytest\n",
        pytrace=False,
    )


# ---------------------------------------------------------------------------
# Compile-cancel egress guard
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_real_compile_cancel(monkeypatch):
    """Stop ``cancel_compiles_for_run`` from POSTing at the PRODUCTION compiler.

    It is the one edge in this repo that builds a REAL, license-authenticated
    HTTP client inside a best-effort ``try/except`` — so an escape is silent by
    construction: the call fails, the warning says "propagation failed", and the
    test still passes.

    Measured 2026-08-20: it was NOT silent in CI. The test job runs on an
    in-cluster self-hosted runner, where ``video-compiler.video-compiler.svc``
    resolves, so ~13 authenticated ``POST /api/jobs/cancel`` per CI run landed on
    the live compiler — 195 of them over 5h40m against 4 real compiles. Each was
    logged at INFO with ``n_cancelled: 0`` (a run id minted milliseconds earlier
    cannot match a job), so nothing anywhere went red.

    A test that wants to ASSERT the propagation applies its own
    ``monkeypatch.setattr`` after this one and is unaffected; everything else gets
    a stub that never opens a socket. Autouse rather than opt-in on purpose: the
    escape is invisible, so it cannot be left to each new test to remember.
    """

    class _StubCancelClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def cancel_run_jobs(self, _run_id: str) -> int:
            return 0

        async def close(self) -> None:
            return None

    from src.services import compile_cancel

    monkeypatch.setattr(compile_cancel, "VideoCompileClient", _StubCancelClient)


# ---------------------------------------------------------------------------
# worker_management egress guard
# ---------------------------------------------------------------------------

#: The message every blocked ``WMClient`` call raises. Public so the regression
#: test can match it exactly.
WM_EGRESS_BLOCKED = (
    "test tried to call worker_management through a real WMClient; "
    "patch WMClient (or pre-seed wm._client with an httpx.MockTransport client)"
)


class WMEgressBlockedError(RuntimeError):
    """Raised when a test reaches a real ``WMClient`` socket."""


#: Base URLs of every blocked attempt since the last per-test reset. Module state
#: (not fixture state) because the guard is installed ONCE for the whole session —
#: see ``_install_wm_egress_guard`` for why it must outlive any single test.
_WM_BLOCKED_ATTEMPTS: list[str] = []


@pytest.fixture(scope="session", autouse=True)
def _install_wm_egress_guard():
    """Stop any real ``WMClient`` from reaching the PRODUCTION worker_management.

    Same leak shape as ``_no_real_compile_cancel`` above. ``WMClient`` resolves
    ``service_url("worker_management")`` by default, and the CI test job runs on an
    in-cluster ARC runner where that name resolves — so a test that forgot to patch
    a stage's ``WMClient`` sent real, license-authenticated LLM traffic to prod.
    Measured 2026-10-07: 10 ``POST /workers/script-draft/message`` from CI runner
    pods in one day (all 400). The stage swallowed the failure, so nothing went red.

    The guard sits on the ONE chokepoint every import site shares — the w2v_common
    class's ``_get_client`` (``message()`` → ``_request`` → ``_get_client``) — instead
    of patching the ~40 ``from w2v_common.clients import WMClient`` sites one by one
    (a new stage would silently fall outside a per-module list). Construction stays
    legal (tests assert on ``.timeout``); only opening a connection is refused.
    Two things keep working:

    * a test that patches ``WMClient`` in its module, or ``_get_client`` on an
      instance — the class's ``_get_client`` is then never reached;
    * the ``*_empty_draw_wire`` tests, which pre-seed ``wm._client`` with an
      ``httpx.MockTransport`` client — a pre-seeded, open client is returned as is.

    SESSION scope and never restored, on purpose: the dispatcher runs stages on
    background tasks, and a task that outlives its test would otherwise find the
    real ``_get_client`` back in place the moment a function-scoped monkeypatch
    undid itself — the exact window in which a slow CI runner leaks.
    """
    from w2v_common.clients.wm_client import WMClient

    real_get_client = WMClient._get_client

    async def _guarded_get_client(self: WMClient) -> AsyncClient:
        client = self._client
        if client is not None and not client.is_closed:
            # A test pre-seeded the pool (MockTransport) — use it unchanged.
            return await real_get_client(self)
        _WM_BLOCKED_ATTEMPTS.append(str(self.base_url))
        raise WMEgressBlockedError(f"{WM_EGRESS_BLOCKED} (base_url={self.base_url})")

    WMClient._get_client = _guarded_get_client  # type: ignore[method-assign]
    _install_llm_egress_guard()
    yield


#: The message every blocked platform-LLM call raises (vLLM / inference-router).
LLM_EGRESS_BLOCKED = (
    "test tried to reach a real LLM endpoint (vLLM / inference-router); install a "
    "tests.support.platform_llm.PlatformChat (in_process.inner_transport_factory) "
    "or pass the client a MockTransport"
)


class LLMEgressBlockedError(RuntimeError):
    """Raised when a test reaches a real platform-LLM socket."""


def _install_llm_egress_guard() -> None:
    """The same guard for the LLM backends that REPLACED worker_management.

    ``w2v_common`` 0.109.0 serves ``WMClient.message`` from Chat Completions (vLLM
    or inference-router) and the orchestrator's platform slot calls them itself
    (``src/services/llm/vllm_direct.py`` / ``ir_direct.py``). On an in-cluster
    runner ``vllm.test`` / ``inference-router`` resolve, so an unpatched call
    is real, licence-authenticated egress — the 2026-10-07 leak with a new host.
    Two chokepoints, both refused BEFORE a socket and without a retry ladder (a
    ``RuntimeError``, not an ``httpx`` transport error, so ``RetryTransport`` does
    not back off on it):

    * the orchestrator: ``in_process.inner_transport_factory`` (every platform AND
      tenant in-process call). A test's ``monkeypatch.setattr`` of it wins and is
      restored back to this guard, never to the real transport;
    * ``w2v_common``: ``VLLMDirectClient._get_client`` (``IRDirectClient`` inherits
      it) when no transport was injected and no pool was pre-seeded.
    """
    import httpx
    import httpx2
    from w2v_common.clients.vllm_direct import VLLMDirectClient

    from src.services.llm import in_process

    class _BlockedTransport(httpx2.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
            _WM_BLOCKED_ATTEMPTS.append(f"{request.url.scheme}://{request.url.host}")
            raise LLMEgressBlockedError(f"{LLM_EGRESS_BLOCKED} (url={request.url})")

    in_process.inner_transport_factory = _BlockedTransport

    real_get_client = VLLMDirectClient._get_client

    def _guarded_vllm_client(self: VLLMDirectClient) -> httpx.AsyncClient:
        client = self._client
        if self._inner_transport is None and (client is None or client.is_closed):
            _WM_BLOCKED_ATTEMPTS.append(str(self.settings.base_url))
            raise LLMEgressBlockedError(f"{LLM_EGRESS_BLOCKED} (base_url={self.settings.base_url})")
        return real_get_client(self)

    VLLMDirectClient._get_client = _guarded_vllm_client  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _no_real_worker_management():
    """Fail LOUD on any blocked ``WMClient`` attempt, even a swallowed one.

    Stage code routinely catches broad exceptions and turns them into a 'failed'
    stage — exactly how the leak stayed silent. So the session guard records every
    attempt and this per-test check fails the test at teardown regardless. An
    attempt made by a task that outlived an EARLIER test lands here too: still
    loud, attributed to the next test. A test whose point is to provoke the
    guard clears ``_WM_BLOCKED_ATTEMPTS`` itself after asserting on it.
    """
    _WM_BLOCKED_ATTEMPTS.clear()
    yield _WM_BLOCKED_ATTEMPTS
    attempts = list(_WM_BLOCKED_ATTEMPTS)
    _WM_BLOCKED_ATTEMPTS.clear()
    if attempts:
        pytest.fail(
            f"{WM_EGRESS_BLOCKED}: {len(attempts)} blocked attempt(s) to {sorted(set(attempts))}",
            pytrace=False,
        )


@pytest.fixture
def platform_llm_socket(monkeypatch):
    """Snapshot ``in_process.inner_transport_factory`` and restore it at teardown.

    For tests whose wire helper calls ``PlatformChat.activate()`` (no monkeypatch
    in reach): whatever they install is undone, back to the egress guard.
    """
    from src.services.llm import in_process

    monkeypatch.setattr(in_process, "inner_transport_factory", in_process.inner_transport_factory)


# ---------------------------------------------------------------------------
# Settings cache management
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _owner_tenant_byo_posture(monkeypatch):
    """Default every test to an OWNER tenant's BYO posture (``local_allowed``).

    The production reader asks the licences service — reachable from the in-cluster
    CI runner, so a test that resolved a platform-GPU slot would otherwise make a
    real cross-service call. Tests of the byo_required behaviour (#1106) pass their
    own ``StaticByoPosture`` / override ``get_connections_byo_posture`` explicitly;
    the licences-backed reader is unit-tested against a MockTransport.
    """
    from src.services.providers.byo_posture import StaticByoPosture, clear_byo_posture_cache

    clear_byo_posture_cache()
    monkeypatch.setattr(
        "src.services.providers.adapters.get_byo_posture",
        lambda: StaticByoPosture("local_allowed"),
    )
    monkeypatch.setattr(
        "src.api.routes.tenancy.connections.get_byo_posture",
        lambda: StaticByoPosture("local_allowed"),
    )
    yield
    clear_byo_posture_cache()


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch):
    """Remove CI env vars that interfere with config-default tests."""
    for key in list(os.environ):
        if any(
            key.startswith(p)
            for p in (
                "DATABASE_",
                "AUTH_",
                "REDIS_",
                "API_",
                "STORAGE_",
                "LOG_",
                "CACHE_",
                "TRACING_",
                "LLM_",
                "FFMPEG_",
                "PIPELINE_",
                "DOWNSTREAM_",
            )
        ):
            monkeypatch.delenv(key, raising=False)
    # LLM_MODEL has no default in code — production sources from configmap.
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-flash")
    # The platform LLM backend prod runs (gitops w2v-orchestrator-config sets
    # LLM_BACKEND=vllm). w2v_common 0.109.0 deleted the worker_management path and
    # defaults an UNSET backend to `ir`, which without a licence fails every call
    # with IRNotConfiguredError — so the suite pins a backend. Fake token; the base
    # URL never resolves, and the egress guard above refuses it before a socket.
    from tests.support.platform_llm import VLLM_TEST_BASE_URL

    monkeypatch.setenv("LLM_BACKEND", "vllm")
    monkeypatch.setenv("LLM_VLLM_TOKEN", "test-vllm-licence-token")  # pragma: allowlist secret
    monkeypatch.setenv("LLM_VLLM_BASE_URL", VLLM_TEST_BASE_URL)
    # Required-env fields that no longer have a localhost fallback
    # (w2v_common 0.15.0 stripped DB/Redis/Storage defaults; orchestrator
    # also dropped its Redis url + minio_endpoint overrides). Tests need
    # *some* value to construct settings; use loopback-style URIs.
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6390/0")
    monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "localhost:9050")
    # DOWNSTREAM_LICENSE_TOKEN is required (no code default). Fake, not a credential.
    monkeypatch.setenv("DOWNSTREAM_LICENSE_TOKEN", "test-downstream-license-token")
    from src.config import clear_settings_cache

    clear_settings_cache()
    yield
    clear_settings_cache()


# ---------------------------------------------------------------------------
# Schema-authority probe (models vs migration chain, built side by side)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def parity_schemas():
    """Build BOTH schema authorities once per session and hand back both names.

    ``create_all`` (the models — and therefore the whole test suite's own
    database) and ``alembic upgrade head`` (the migration chain — and therefore
    PROD) are built into two throwaway schemas of the test database. Any guard
    that wants to make a claim about the production schema must read the
    ``alembic`` one: reading ``pg_constraint`` off the normal ``db_session``
    fixture only ever reflects ``src/database/models.py`` back at you, because
    ``_db_schema_initialized`` below builds that database with ``create_all``.

    Session-scoped because a full chain replay is the expensive part; both
    consumers (``tests/database/test_schema_parity.py`` and
    ``tests/contracts/test_stage_status_check_constraint_parity.py``) share the
    one build. Mechanics live in ``tests/support/schema_probe.py``.

    An unreachable database FAILS here, same policy as every other DB fixture —
    see ``_fail_or_skip_without_database``.
    """
    from tests.support import schema_probe

    _ensure_run_schema()
    engine = schema_probe.sync_engine(TEST_DATABASE_URL)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:
        engine.dispose()
        _fail_or_skip_without_database(exc)

    try:
        schema_probe.build_create_all(engine)
        schema_probe.build_alembic(engine, TEST_DATABASE_URL)
        yield schema_probe.ParityProbe(
            engine=engine,
            create_all=schema_probe.CREATE_ALL_SCHEMA,
            alembic=schema_probe.ALEMBIC_SCHEMA,
        )
    finally:
        schema_probe.drop_probe_schemas(engine)
        engine.dispose()


# ---------------------------------------------------------------------------
# Database fixtures (real postgres+psycopg, TRUNCATE-based isolation)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
async def _db_schema_initialized():
    """Drop+recreate model tables once per test session to absorb schema drift.

    The CI database may carry a stale schema from previous orchestrator
    versions. Drop and recreate all model-managed tables once at session
    start; per-test isolation is then handled by TRUNCATE in db_engine.

    If TEST_DATABASE_URL is unreachable this FAILS — see
    ``_fail_or_skip_without_database``. It does not skip.
    """
    from src.database.models import Base

    # Locally this claims a private schema for the run (and reclaims abandoned
    # ones) so a concurrent worktree's suite can neither TRUNCATE our tables nor
    # see our backends. Under CI it is a no-op — the job already owns an
    # ephemeral database.
    _ensure_run_schema()
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    try:
        await _wait_for_engine(engine)
        async with engine.begin() as conn:
            # Drop any orphan tables from retired models (curriculum_lessons +
            # curricula were removed when migration 0012 dropped the substrate;
            # CI databases that pre-date that change carry the tables as dead
            # weight with FKs pointing back at projects).
            await conn.execute(text("DROP TABLE IF EXISTS curriculum_lessons CASCADE"))
            await conn.execute(text("DROP TABLE IF EXISTS curricula CASCADE"))
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
    except OperationalError as exc:
        await engine.dispose()
        _fail_or_skip_without_database(exc)
    await engine.dispose()
    yield


@pytest.fixture
async def db_engine(_db_schema_initialized):
    """Per-test async postgres engine with TRUNCATE-based isolation.

    Connects to TEST_DATABASE_URL captured at module load (CI postgres in
    CI, configurable locally). Schema is created once per session by
    _db_schema_initialized; this fixture TRUNCATEs every model table so
    each test starts from a clean slate.
    """
    from src.database.models import Base

    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    await _wait_for_engine(engine)
    async with engine.begin() as conn:
        # Backstop, NOT the fix. The `app` fixture below now shuts its
        # RunDispatcher down in teardown, which is where dispatcher-leaked
        # connections actually got closed; `src/api/sse.py` still has an
        # un-awaited stream task that can leak one. A leaked task's loop closes
        # mid-transaction, so its psycopg backend survives server-side holding a
        # row/table lock. `TRUNCATE ... CASCADE` needs ACCESS EXCLUSIVE, so it
        # would block on that leaked session FOREVER — hanging the whole suite.
        #
        # Reap only backends parked IDLE while holding a lock: the original
        # `idle in transaction`, plus `idle in transaction (aborted)` (a leaked
        # task whose statement errored before its loop died). Both are stuck and
        # safe to terminate. Deliberately NOT `active`. `pid <> pg_backend_pid()`
        # keeps us from reaping ourselves. (No-op when nothing leaked.)
        #
        # The `datname = current_database()` scope used to argue that excluding
        # `active` made it safe for concurrent runs. It does not: a concurrent
        # run's connection parked between two statements of its OWN fixture is
        # `idle in transaction` too — a working run, not a leak — and we killed
        # it ("terminating connection due to administrator command"). The scope
        # now also pins `application_name` to this run, so the sweep can only
        # reach backends this process opened. See
        # tests/support/run_isolation.py::terminate_scope_sql.
        await conn.execute(text(run_isolation.terminate_scope_sql()))
        # Backstop: if something still holds a conflicting lock, fail FAST with a
        # diagnosable error instead of hanging the run indefinitely.
        await conn.execute(text("SET lock_timeout = '15s'"))
        if Base.metadata.sorted_tables:
            tables = ", ".join(f'"{t.name}"' for t in reversed(Base.metadata.sorted_tables))
            await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session_factory(db_engine):
    """Create a real async_sessionmaker bound to the test engine."""
    return async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def db_session(db_session_factory):
    """Create a real AsyncSession for a single test."""
    async with db_session_factory() as session:
        yield session


# ---------------------------------------------------------------------------
# Existing fixtures (mock-based, for API route tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    from src.config import (
        APISettings,
        AuthSettings,
        DatabaseSettings,
        PipelineSettings,
        RedisSettings,
        Settings,
        clear_settings_cache,
    )

    clear_settings_cache()
    return Settings(
        database=DatabaseSettings(url=TEST_DATABASE_URL),
        redis=RedisSettings(url="redis://localhost:6390/0"),
        api=APISettings(port=6300, host="127.0.0.1"),
        auth=AuthSettings(enabled=False),
        pipeline=PipelineSettings(mode="gateway"),
    )


@pytest.fixture
async def app(db_session_factory) -> AsyncGenerator[FastAPI]:
    """Studio v3 test app — fresh FastAPI wiring routers + test DB + run engine.

    A real RunEngine + RunDispatcher are wired so /api/runs routes work end-to-end.
    Inference + render clients are mocked at the stage-module level by individual
    tests that exercise the pause/resume flow.

    Async + generator on purpose. ``RunDispatcher.dispatch`` is fire-and-forget
    (``asyncio.create_task(self._run(run_id))``); ``_tasks`` is drained ONLY by
    ``shutdown()``. This fixture used to ``return`` with no teardown, so every
    test that dispatched a run orphaned its task: the function-scoped loop closed
    mid-transaction and the task's psycopg connection was never closed, leaving a
    backend alive server-side holding locks that hang the next TRUNCATE. Every
    other construction site in the suite already shuts down in a ``finally``
    (tests/integration/_dispatch_harness.py, tests/unit/test_run_dispatcher_*.py,
    tests/services/test_run_dispatcher.py); this one is now consistent with them.
    """
    import fakeredis.aioredis

    import src.services.blueprints  # noqa: F401
    from src.api.routes import (
        admin_router,
        annotations_router,
        audit_router,
        auth_router,
        blueprints_router,
        critique_router,
        deliverables_router,
        fixes_router,
        media_router,
        projects_router,
        runs_router,
        runs_usage_router,
        storyboard_router,
        uploads_router,
        uploads_video_router,
        verbatim_router,
        youtube_router,
    )
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    test_app = FastAPI(title="Test W2V Orchestrator")
    test_app.state.session_factory = db_session_factory
    redis = fakeredis.aioredis.FakeRedis()
    test_app.state.redis = redis
    # Keep local refs for teardown: tests may overwrite app.state (e.g. the
    # degraded-mode 503 tests set state.dispatcher = None), but the ORIGINAL
    # dispatcher is the one whose fire-and-forget tasks must be drained.
    dispatcher = RunDispatcher(
        session_factory=db_session_factory,
        registry=get_registry(),
        redis=redis,
    )
    test_app.state.dispatcher = dispatcher
    test_app.state.run_engine = RunEngine(
        session_factory=db_session_factory,
        registry=get_registry(),
        dispatch=dispatcher.dispatch,
    )
    test_app.include_router(auth_router)
    test_app.include_router(blueprints_router)
    test_app.include_router(projects_router)
    test_app.include_router(audit_router)
    test_app.include_router(deliverables_router)
    test_app.include_router(runs_router)
    test_app.include_router(runs_usage_router)
    test_app.include_router(annotations_router)
    test_app.include_router(verbatim_router)
    test_app.include_router(storyboard_router)
    test_app.include_router(media_router)
    test_app.include_router(critique_router)
    test_app.include_router(fixes_router)
    test_app.include_router(uploads_router)
    # Mounted separately in src/asgi.py too (it is a distinct router, not part of
    # uploads_router) — the streaming video-source routes are invisible to the
    # test app unless it is included here as well.
    test_app.include_router(uploads_video_router)
    test_app.include_router(youtube_router)
    test_app.include_router(admin_router)
    # The production app registers this in src/asgi.py; the test app is hand-built and
    # bypasses create_app, so register it here too or the storyboard mutators surface a
    # raw 500 instead of the 409 the ProjectArchivedError guard intends.
    from src.api.routes.storyboard import register_storyboard_exception_handlers

    register_storyboard_exception_handlers(test_app)
    from src.api.byo_errors import register_byo_exception_handlers

    register_byo_exception_handlers(test_app)
    try:
        yield test_app
    finally:
        # Dispatcher first: shutdown() cancels + awaits _tasks (and the control
        # listener), and those tasks publish terminal events to redis on their
        # way out. Closing redis before them would raise inside teardown.
        try:
            # Storyboard panel-image jobs are the same fire-and-forget shape as
            # the dispatcher's run tasks (lazily created by the route, so it may
            # be absent) — drain them before redis closes for the same reason.
            panel_jobs = getattr(test_app.state, "panel_image_jobs", None)
            if panel_jobs is not None:
                await panel_jobs.shutdown()
            await dispatcher.shutdown()
        finally:
            await redis.aclose()


def _build_route_client(test_app: FastAPI, tenant_id: str, subject_id: str) -> ASGITransport:
    """Mount a per-test middleware that seeds tenant/subject UUIDs.

    Every fixture invocation creates a NEW middleware closure so concurrent
    test executions don't bleed UUIDs into one another.
    """
    from starlette.middleware.base import BaseHTTPMiddleware

    class _SeedAuthState(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = tenant_id
            request.state.subject_id = subject_id
            return await call_next(request)

    test_app.add_middleware(_SeedAuthState)
    return ASGITransport(app=test_app)


class SeededClient(AsyncClient):
    """AsyncClient subclass that exposes the seeded tenant + subject UUIDs."""

    tenant_id: str
    subject_id: str


@pytest.fixture
async def client(app: FastAPI) -> AsyncGenerator[SeededClient]:
    """Async test client with a seeded tenant + subject pair."""
    from uuid import uuid4

    tenant_id = str(uuid4())
    subject_id = str(uuid4())
    transport = _build_route_client(app, tenant_id, subject_id)
    async with SeededClient(transport=transport, base_url="http://testserver") as c:
        c.tenant_id = tenant_id
        c.subject_id = subject_id
        yield c


@pytest.fixture
async def other_tenant_client(
    db_session_factory,
) -> AsyncGenerator[AsyncClient]:
    """A second client whose tenant_id differs from ``client``'s.

    Used to assert cross-tenant reads return 404 (never 200). Builds its
    own FastAPI app rather than reusing ``app`` so the seeded UUIDs are
    independent of the primary client.
    """
    from uuid import uuid4

    from src.api.routes.auth import router as auth_router
    from src.api.routes.deliverables import router as deliverables_router
    from src.api.routes.projects import router as projects_router
    from src.api.routes.storyboard import (
        register_storyboard_exception_handlers,
    )
    from src.api.routes.storyboard import (
        router as storyboard_router,
    )

    test_app = FastAPI(title="Test W2V Orchestrator (other tenant)")
    test_app.state.session_factory = db_session_factory
    test_app.include_router(auth_router)
    test_app.include_router(projects_router)
    test_app.include_router(deliverables_router)
    test_app.include_router(storyboard_router)
    register_storyboard_exception_handlers(test_app)
    from src.api.byo_errors import register_byo_exception_handlers

    register_byo_exception_handlers(test_app)
    transport = _build_route_client(test_app, str(uuid4()), str(uuid4()))
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.fixture
def sample_pipeline_config():
    return {
        "url": "https://example.com",
        "config": {
            "stages_enabled": ["crawl", "analyze", "storyline", "shotlist"],
            "timeout_seconds": 300,
        },
    }


@pytest.fixture
async def redis_client():
    """Async fakeredis client for dispatcher integration tests."""
    from fakeredis.aioredis import FakeRedis

    client = FakeRedis()
    try:
        yield client
    finally:
        await client.aclose()
