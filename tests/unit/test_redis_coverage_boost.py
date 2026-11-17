"""Orchestrator-specific Redis client tests.

Connection/ping/disconnect lifecycle behavior is inherited from
w2v_common.cache.redis_client.RedisClient and is covered by
w2v_common's own test suite. This file focuses on the orchestrator's
additions: pubsub(), publish(), and singleton lifecycle helpers.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cache.redis_client import RedisClient, close_redis, get_redis_client, init_redis


def _mock_settings():
    """Helper to create mock settings for RedisClient."""
    mock = MagicMock()
    mock.redis.url = "redis://localhost:6390/0"
    mock.redis.max_connections = 10
    mock.redis.socket_timeout = 5.0
    mock.redis.socket_connect_timeout = 2.0
    mock.redis.decode_responses = True
    mock.redis.health_check_interval = 30
    return mock


# ============================================================================
# pubsub()
# ============================================================================


def test_pubsub_raises_when_not_connected():
    """pubsub() raises RuntimeError when not connected."""
    with patch("src.cache.redis_client.get_settings", return_value=_mock_settings()):
        client = RedisClient()

    with pytest.raises(RuntimeError, match="not connected"):
        client.pubsub()


# ============================================================================
# publish()
# ============================================================================


@pytest.mark.asyncio
async def test_publish_raises_when_not_connected():
    """publish() raises RuntimeError when not connected."""
    with patch("src.cache.redis_client.get_settings", return_value=_mock_settings()):
        client = RedisClient()

    with pytest.raises(RuntimeError, match="not connected"):
        await client.publish("ch", "msg")


# ============================================================================
# Settings → constructor wiring
# ============================================================================


def test_constructor_threads_health_check_interval_from_settings():
    """The orchestrator subclass must propagate health_check_interval from
    its typed settings into the base class — otherwise ConnectionPool.from_url
    would never receive it."""
    with patch("src.cache.redis_client.get_settings", return_value=_mock_settings()):
        client = RedisClient()
    assert client._health_check_interval == 30


# ============================================================================
# get_redis_client() singleton
# ============================================================================


def test_get_redis_client_creates_singleton():
    """get_redis_client() creates and returns a singleton instance."""
    import src.cache.redis_client as module

    original = module._redis_client
    try:
        module._redis_client = None
        with patch("src.cache.redis_client.get_settings", return_value=_mock_settings()):
            c1 = get_redis_client()
            c2 = get_redis_client()
        assert c1 is c2
        assert c1 is not None
    finally:
        module._redis_client = original


# ============================================================================
# init_redis()
# ============================================================================


@pytest.mark.asyncio
async def test_init_redis_connects_client():
    """init_redis() creates client and calls connect()."""
    import src.cache.redis_client as module

    original = module._redis_client
    try:
        module._redis_client = None
        with patch("src.cache.redis_client.get_settings", return_value=_mock_settings()):
            mock_client = RedisClient()
        mock_connect = AsyncMock()
        with patch.object(mock_client, "connect", mock_connect):
            module._redis_client = mock_client
            result = await init_redis()
            assert result is mock_client
            mock_connect.assert_awaited_once()
    finally:
        module._redis_client = original


# ============================================================================
# close_redis()
# ============================================================================


@pytest.mark.asyncio
async def test_close_redis_with_active_client():
    """close_redis() disconnects and discards the global client."""
    import src.cache.redis_client as module

    original = module._redis_client
    try:
        mock_client = MagicMock()
        mock_client.disconnect = AsyncMock()
        module._redis_client = mock_client

        await close_redis()

        mock_client.disconnect.assert_awaited_once()
        assert module._redis_client is None
    finally:
        module._redis_client = original
