"""Database connection module with connection pooling and initialization."""

import asyncio
import logging
import os
import re
from urllib.parse import urlparse, urlunparse

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import AsyncAdaptedQueuePool, NullPool

from src.config import _DEFAULT_DATABASE_URL

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", _DEFAULT_DATABASE_URL)

parsed_url = urlparse(DATABASE_URL)
DB_HOST = parsed_url.hostname or "localhost"
DB_PORT = parsed_url.port or 5432
DB_NAME = parsed_url.path.lstrip("/") if parsed_url.path else "postgres"
DB_USER = parsed_url.username or "postgres"
DB_PASSWORD = parsed_url.password or ""

DB_POOL_MIN_SIZE = int(os.getenv("POSTGRES_POOL_MIN_SIZE", "2"))
DB_POOL_MAX_SIZE = int(os.getenv("POSTGRES_POOL_MAX_SIZE", "10"))
DB_POOL_TIMEOUT = int(os.getenv("POSTGRES_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE = int(os.getenv("POSTGRES_POOL_RECYCLE", "900"))
DB_ADMIN_DB = os.getenv("POSTGRES_ADMIN_DB", "postgres")
# The restricted runtime role (Phase D enforce) is a non-superuser that cannot
# CREATE DATABASE and has no business connecting to the admin DB. The DB always
# exists by then (the init container migrated it), so the auto-bootstrap is pure
# footgun — gate it off for that container. Default false (legacy behaviour).
DB_SKIP_BOOTSTRAP = os.getenv("POSTGRES_SKIP_BOOTSTRAP", "false").lower() == "true"

_engine: AsyncEngine | None = None
# Serialises lazy init in `get_engine()` so concurrent first-callers
# don't each spin up an engine and leak the discarded pool's
# connections. Lock is constructed lazily because `asyncio.Lock()`
# requires a running event loop.
_engine_init_lock: asyncio.Lock | None = None


def _create_or_verify_database(sync_admin_url: str) -> None:
    """Open admin engine and create the target database if missing."""
    engine = create_engine(sync_admin_url, isolation_level="AUTOCOMMIT", poolclass=NullPool)
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                {"db_name": DB_NAME},
            )
            if result.scalar() is not None:
                logger.info("Database '%s' already exists", DB_NAME)
                return
            try:
                conn.execute(text(f"CREATE DATABASE {DB_NAME}"))
                logger.info("Database '%s' created successfully", DB_NAME)
            except ProgrammingError as e:
                if "already exists" not in str(e).lower():
                    raise
    finally:
        try:
            engine.dispose()
        except OSError:
            logger.debug("Failed to dispose admin engine during cleanup")


async def _ensure_database_exists(max_retries: int = 3) -> None:
    """Ensure the target database exists, creating it if necessary."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", DB_NAME):
        raise ValueError(
            f"Database name '{DB_NAME}' contains invalid characters; "
            f"only alphanumeric and underscore are supported."
        )

    admin_url_parts = list(urlparse(DATABASE_URL))
    admin_url_parts[2] = f"/{DB_ADMIN_DB}"
    sync_admin_url = urlunparse(admin_url_parts)

    logger.info(
        "Ensuring PostgreSQL database '%s' exists using admin database '%s' at %s:%s",
        DB_NAME,
        DB_ADMIN_DB,
        DB_HOST,
        DB_PORT,
    )

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            _create_or_verify_database(sync_admin_url)
            return
        except (OperationalError, OSError, ConnectionRefusedError) as e:
            last_error = e
            if attempt < max_retries:
                await asyncio.sleep(attempt * 2)
        except (ProgrammingError, RuntimeError) as e:
            logger.error("Unexpected error ensuring database exists: %s", e, exc_info=True)
            raise

    raise ConnectionError(
        f"Failed to connect to admin database at {DB_HOST}:{DB_PORT} "
        f"after {max_retries} attempts. Last error: {last_error}"
    )


async def create_engine_with_retry(max_retries: int = 5, retry_delay: float = 2.0) -> AsyncEngine:
    """Create database engine with connection retry logic."""
    last_error = None
    current_delay = retry_delay

    for attempt in range(1, max_retries + 1):
        try:
            pool_config = {
                "poolclass": AsyncAdaptedQueuePool,
                "pool_size": DB_POOL_MIN_SIZE,
                "max_overflow": DB_POOL_MAX_SIZE - DB_POOL_MIN_SIZE,
                "pool_timeout": DB_POOL_TIMEOUT,
                "pool_recycle": DB_POOL_RECYCLE,
                "pool_pre_ping": True,
                "echo": os.getenv("SQL_ECHO", "false").lower() == "true",
            }
            engine = create_async_engine(DATABASE_URL, **pool_config)
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("Successfully created engine on attempt %d", attempt)
            return engine

        except (OperationalError, ProgrammingError) as e:
            error_msg = str(e).lower()
            # DB_SKIP_BOOTSTRAP (restricted runtime role): never attempt the
            # admin-DB CREATE DATABASE bootstrap — the DB already exists by then,
            # and the role can't connect to the admin DB. Falls through to the
            # normal retry/fail path instead.
            missing_db = "database" in error_msg and "does not exist" in error_msg
            if missing_db and not DB_SKIP_BOOTSTRAP:
                try:
                    await _ensure_database_exists()
                    pool_config = {
                        "poolclass": AsyncAdaptedQueuePool,
                        "pool_size": DB_POOL_MIN_SIZE,
                        "max_overflow": DB_POOL_MAX_SIZE - DB_POOL_MIN_SIZE,
                        "pool_timeout": DB_POOL_TIMEOUT,
                        "pool_recycle": DB_POOL_RECYCLE,
                        "pool_pre_ping": True,
                        "echo": os.getenv("SQL_ECHO", "false").lower() == "true",
                    }
                    engine = create_async_engine(DATABASE_URL, **pool_config)
                    async with engine.begin() as conn:
                        await conn.execute(text("SELECT 1"))
                    return engine
                except (OperationalError, ProgrammingError, OSError) as create_err:
                    last_error = create_err
            else:
                last_error = e
            if attempt < max_retries:
                await asyncio.sleep(current_delay)
                current_delay *= 2

        except (TimeoutError, OSError, ConnectionRefusedError) as e:
            last_error = e
            if attempt < max_retries:
                await asyncio.sleep(current_delay)
                current_delay *= 2

        except RuntimeError as e:
            last_error = e
            if attempt < max_retries:
                await asyncio.sleep(current_delay)
                current_delay *= 2

    raise ConnectionError(
        f"Failed to connect to PostgreSQL at {DB_HOST}:{DB_PORT}/{DB_NAME} "
        f"after {max_retries} attempts. Last error: {last_error}"
    ) from last_error


async def _probe_backend_recovery(engine: AsyncEngine) -> None:
    """Ask ONCE whether this engine's backend is a read-only standby, and record it.

    The ``after_begin`` RLS listener issues ``SET LOCAL transaction_read_only = off``
    as the first statement of every transaction to survive a leaked
    ``default_transaction_read_only = on`` on a shared PgBouncer backend. That
    statement is ILLEGAL during recovery (0A000), so on a standby it would kill
    every transaction — reads included. This answers the question once, here, rather
    than paying a ``pg_is_in_recovery()`` round-trip on every transaction in the
    service.

    Deliberately NOT fatal: a probe that cannot run must not stop the pod from
    starting. Failing to answer leaves the 'assume primary' default, which is the
    behaviour that shipped before the reset existed, and
    ``note_read_only_reset_rejected`` still latches from the first rejection.
    """
    from src.database.session import RECOVERY_PROBE_SQL, set_backend_in_recovery

    try:
        async with engine.connect() as conn:
            in_recovery = bool((await conn.execute(text(RECOVERY_PROBE_SQL))).scalar())
    # Deliberately ANY Exception, not a hand-listed tuple. A tuple only tolerates the
    # failures someone thought of, which makes a "non-fatal" probe fatal for every
    # other one (a driver returning an unexpected object, a wrapper that is not an
    # async context manager, a decode error). Connectivity is NOT this handler's job:
    # `create_engine_with_retry` has already opened a connection and run SELECT 1 with
    # retries before we are called, and it still raises ConnectionError when the
    # database is unreachable — so nothing swallowed here can hide a dead database.
    # BaseException (CancelledError on shutdown, KeyboardInterrupt, SystemExit) is
    # excluded on purpose: those are not probe failures and must keep propagating.
    except Exception as exc:  # the probe must never stop the pod booting
        logger.warning(
            "could not determine whether the database backend is in recovery (%s); "
            "assuming a primary",
            exc,
            exc_info=True,
        )
        return
    set_backend_in_recovery(in_recovery)
    if in_recovery:
        logger.warning(
            "the database backend is a read-only STANDBY (pg_is_in_recovery() = t). "
            "Writes will fail here; the leaked-read-only reset is disabled so reads "
            "keep working. If this pod is meant to write, its DSN is pointed at the "
            "wrong host."
        )


async def get_engine() -> AsyncEngine:
    """Get or create the global database engine instance.

    Concurrent first-callers are serialised behind ``_engine_init_lock``
    so only one engine is ever bound to the process; without it,
    coroutines that arrive during ``await create_engine_with_retry()``
    each construct their own engine and the slower one's pool leaks.
    """
    global _engine, _engine_init_lock
    if _engine is not None:
        return _engine
    if _engine_init_lock is None:
        _engine_init_lock = asyncio.Lock()
    async with _engine_init_lock:
        if _engine is None:
            engine = await create_engine_with_retry()
            await _probe_backend_recovery(engine)
            _engine = engine
    return _engine


async def health_check() -> bool:
    """Perform a health check on the database connection."""
    try:
        engine = await get_engine()
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except (OSError, OperationalError) as e:
        logger.error("Database health check failed: %s", e, exc_info=True)
        return False


async def close_engine() -> None:
    """Close the global database engine."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        logger.info("Database engine closed successfully")
