"""Tests for database connection module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

from src.database.connection import (
    DATABASE_URL,
    DB_HOST,
    DB_NAME,
    DB_POOL_MAX_SIZE,
    DB_POOL_MIN_SIZE,
    DB_POOL_RECYCLE,
    DB_POOL_TIMEOUT,
    DB_PORT,
    DB_USER,
    _ensure_database_exists,
    close_engine,
    create_engine_with_retry,
    health_check,
)

# ============================================================================
# Module-level constants
# ============================================================================


def test_database_url_configured():
    """DATABASE_URL is a non-empty string."""
    assert DATABASE_URL is not None
    assert len(DATABASE_URL) > 0
    assert "postgresql" in DATABASE_URL


def test_db_host_parsed():
    """DB_HOST is parsed from the URL."""
    assert DB_HOST is not None


def test_db_port_parsed():
    """DB_PORT is parsed from the URL."""
    assert isinstance(DB_PORT, int)


def test_db_name_parsed():
    """DB_NAME is parsed from the URL."""
    assert DB_NAME is not None
    assert len(DB_NAME) > 0


def test_db_user_parsed():
    """DB_USER is parsed from the URL."""
    assert DB_USER is not None


def test_pool_size_defaults():
    """Pool size defaults are reasonable integers."""
    assert DB_POOL_MIN_SIZE >= 1
    assert DB_POOL_MAX_SIZE >= DB_POOL_MIN_SIZE
    assert DB_POOL_TIMEOUT > 0
    assert DB_POOL_RECYCLE > 0


# ============================================================================
# _ensure_database_exists
# ============================================================================


@pytest.mark.asyncio
async def test_ensure_db_retries_on_connection_error():
    """OperationalError on first attempt retries and succeeds on second."""
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = 1  # DB exists
    mock_conn.execute.return_value = mock_result
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    mock_engine_fail = MagicMock()
    mock_engine_fail.connect.side_effect = OperationalError(
        statement="SELECT 1", params=None, orig=Exception("connection refused")
    )
    mock_engine_fail.dispose = MagicMock()

    mock_engine_ok = MagicMock()
    mock_engine_ok.connect.return_value = mock_conn
    mock_engine_ok.dispose = MagicMock()

    call_count = 0

    def create_engine_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_engine_fail
        return mock_engine_ok

    with (
        patch(
            "src.database.connection.create_engine",
            side_effect=create_engine_side_effect,
        ),
        patch("src.database.connection.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        await _ensure_database_exists(max_retries=3)

    mock_sleep.assert_awaited_once()
    mock_engine_fail.dispose.assert_called_once()
    mock_engine_ok.dispose.assert_called_once()


# ============================================================================
# create_engine_with_retry
# ============================================================================


@pytest.mark.asyncio
async def test_create_engine_exhausts_retries():
    """All attempts fail, ConnectionError is raised with last error info."""
    mock_engine = AsyncMock(spec=["begin", "dispose"])
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(side_effect=TimeoutError("always times out"))
    mock_cm.__aexit__ = AsyncMock(return_value=False)
    mock_engine.begin.return_value = mock_cm

    with (
        patch(
            "src.database.connection.create_async_engine",
            return_value=mock_engine,
        ),
        patch("src.database.connection.asyncio.sleep", new_callable=AsyncMock),
    ):
        with pytest.raises(ConnectionError, match="Failed to connect"):
            await create_engine_with_retry(max_retries=3)


@pytest.mark.asyncio
async def test_create_engine_backoff_doubles():
    """asyncio.sleep is called with exponentially increasing delays."""
    mock_engine = AsyncMock(spec=["begin", "dispose"])
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(side_effect=TimeoutError("timeout"))
    mock_cm.__aexit__ = AsyncMock(return_value=False)
    mock_engine.begin.return_value = mock_cm

    with (
        patch(
            "src.database.connection.create_async_engine",
            return_value=mock_engine,
        ),
        patch("src.database.connection.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        with pytest.raises(ConnectionError):
            await create_engine_with_retry(max_retries=4, retry_delay=2.0)

    # Delays: 2.0, 4.0, 8.0 (last attempt doesn't sleep)
    sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
    assert sleep_calls == [2.0, 4.0, 8.0]


# ============================================================================
# get_engine (singleton)
# ============================================================================


# ============================================================================
# health_check
# ============================================================================


@pytest.mark.asyncio
async def test_health_check_success():
    """health_check returns True when SELECT 1 succeeds."""
    import src.database.connection as conn_module

    original = conn_module._engine

    mock_conn = AsyncMock()
    mock_engine = AsyncMock(spec=["begin", "dispose"])
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_cm.__aexit__ = AsyncMock(return_value=False)
    mock_engine.begin.return_value = mock_cm
    conn_module._engine = mock_engine

    try:
        with patch(
            "src.database.connection.get_engine",
            new_callable=AsyncMock,
            return_value=mock_engine,
        ):
            result = await health_check()

        assert result is True
        mock_conn.execute.assert_awaited_once()
    finally:
        conn_module._engine = original


@pytest.mark.asyncio
async def test_health_check_failure():
    """health_check returns False when the database connection fails."""
    import src.database.connection as conn_module

    original = conn_module._engine

    try:
        with patch(
            "src.database.connection.get_engine",
            new_callable=AsyncMock,
            side_effect=ConnectionError("db down"),
        ):
            result = await health_check()

        assert result is False
    finally:
        conn_module._engine = original


# ============================================================================
# close_engine
# ============================================================================


@pytest.mark.asyncio
async def test_close_engine_when_none():
    """close_engine is safe when no engine exists."""
    import src.database.connection as conn_module

    original = conn_module._engine
    conn_module._engine = None
    await close_engine()
    assert conn_module._engine is None
    conn_module._engine = original


@pytest.mark.asyncio
async def test_close_engine_disposes():
    """close_engine disposes the engine and sets _engine to None."""
    import src.database.connection as conn_module

    mock_engine = AsyncMock()
    original = conn_module._engine
    conn_module._engine = mock_engine

    await close_engine()

    mock_engine.dispose.assert_awaited_once()
    assert conn_module._engine is None
    conn_module._engine = original
