"""Tests for database session management module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ============================================================================
# close_session_maker
# ============================================================================


@pytest.mark.asyncio
async def test_close_session_maker():
    """close_session_maker resets the global session maker."""
    import src.database.session as session_module

    # Set a mock value
    session_module._async_session_maker = MagicMock()
    await session_module.close_session_maker()
    assert session_module._async_session_maker is None


@pytest.mark.asyncio
async def test_close_session_maker_when_none():
    """close_session_maker is safe when already None."""
    import src.database.session as session_module

    session_module._async_session_maker = None
    await session_module.close_session_maker()
    assert session_module._async_session_maker is None


# ============================================================================
# get_session_maker
# ============================================================================


@pytest.mark.asyncio
async def test_get_session_maker_creates_once():
    """get_session_maker creates a session maker and caches it."""
    import src.database.session as session_module

    session_module._async_session_maker = None

    mock_engine = AsyncMock()
    with patch.object(session_module, "get_engine", return_value=mock_engine):
        maker = await session_module.get_session_maker()
        assert maker is not None
        # Second call should return same maker
        maker2 = await session_module.get_session_maker()
        assert maker2 is maker

    # Cleanup
    session_module._async_session_maker = None


# ============================================================================
# get_db (FastAPI dependency — async generator)
# ============================================================================


def _make_mock_session_maker():
    """Helper: build a mock session maker whose __call__ returns an async-context session."""
    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.close = AsyncMock()

    # session_maker() returns an async context manager that yields mock_session
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_session)
    ctx.__aexit__ = AsyncMock(return_value=False)

    mock_maker = MagicMock()
    mock_maker.return_value = ctx

    return mock_maker, mock_session


@pytest.mark.asyncio
async def test_get_db_commits_on_success():
    """get_db commits and closes the session on successful iteration."""
    import src.database.session as session_module

    mock_maker, mock_session = _make_mock_session_maker()

    with patch.object(session_module, "get_session_maker", return_value=mock_maker):
        gen = session_module.get_db()
        session = await gen.__anext__()
        assert session is mock_session

        # Signal generator that the consumer is done (StopAsyncIteration)
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()

    mock_session.commit.assert_awaited_once()
    mock_session.close.assert_awaited_once()
    mock_session.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_db_rollbacks_on_exception():
    """get_db rolls back and closes the session when an exception is thrown in."""
    import src.database.session as session_module

    mock_maker, mock_session = _make_mock_session_maker()

    with patch.object(session_module, "get_session_maker", return_value=mock_maker):
        gen = session_module.get_db()
        session = await gen.__anext__()
        assert session is mock_session

        # Throw an exception into the generator
        with pytest.raises(RuntimeError, match="db error"):
            await gen.athrow(RuntimeError("db error"))

    mock_session.rollback.assert_awaited_once()
    mock_session.close.assert_awaited_once()
    mock_session.commit.assert_not_awaited()


# ============================================================================
# get_db_session (async context manager)
# ============================================================================


@pytest.mark.asyncio
async def test_get_db_session_commits_on_success():
    """get_db_session commits and closes the session on success."""
    import src.database.session as session_module

    mock_maker, mock_session = _make_mock_session_maker()

    with patch.object(session_module, "get_session_maker", return_value=mock_maker):
        async with session_module.get_db_session() as session:
            assert session is mock_session

    mock_session.commit.assert_awaited_once()
    mock_session.close.assert_awaited_once()
    mock_session.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_db_session_rollbacks_on_exception():
    """get_db_session rolls back and closes the session on exception."""
    import src.database.session as session_module

    mock_maker, mock_session = _make_mock_session_maker()

    with patch.object(session_module, "get_session_maker", return_value=mock_maker):
        with pytest.raises(RuntimeError, match="db error"):
            async with session_module.get_db_session() as session:
                assert session is mock_session
                raise RuntimeError("db error")

    mock_session.rollback.assert_awaited_once()
    mock_session.close.assert_awaited_once()
    mock_session.commit.assert_not_awaited()
