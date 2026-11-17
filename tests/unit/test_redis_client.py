"""Tests for Redis client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import RedisError

from src.cache.redis_client import RedisClient, close_redis

# ---------- RedisClient unit tests ----------


def test_redis_client_init():
    """RedisClient initialises with settings from config."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        assert client._redis_url == "redis://localhost:6390/0"
        assert client._max_connections == 10
        assert client._pool is None
        assert client._client is None


def test_redis_client_init_custom_url():
    """RedisClient can be initialised with a custom URL."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://default:6390/0"
        mock_settings.return_value.redis.max_connections = 5
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient(redis_url="redis://custom:6399/1")
        assert client._redis_url == "redis://custom:6399/1"


def test_is_available_false_initially():
    """is_available returns False before connecting."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        assert client.is_available is False


def test_ensure_connected_raises_without_connect():
    """_ensure_connected raises RuntimeError before connect()."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        with pytest.raises(RuntimeError, match="not connected"):
            client._ensure_connected()


# ---------- Serialization tests ----------


def test_serialize_dict():
    """_serialize converts dict to JSON string."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        result = client._serialize({"key": "value"})
        assert result == '{"key": "value"}'


def test_serialize_pydantic_model():
    """_serialize handles objects with model_dump method."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        model = MagicMock()
        model.model_dump.return_value = {"name": "test"}
        result = client._serialize(model)
        assert '"name"' in result
        assert '"test"' in result


def test_deserialize_valid_json():
    """_deserialize parses valid JSON string."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        result = client._deserialize('{"key": "value"}')
        assert result == {"key": "value"}


def test_deserialize_none():
    """_deserialize returns None for None input."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        assert client._deserialize(None) is None


def test_deserialize_invalid_json():
    """_deserialize returns None for invalid JSON."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        assert client._deserialize("not valid json{") is None


# ---------- Async operations when not connected ----------


@pytest.mark.asyncio
async def test_get_returns_none_when_not_connected():
    """get() returns None when client is not connected."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        result = await client.get("some-key")
        assert result is None


@pytest.mark.asyncio
async def test_set_returns_false_when_not_connected():
    """set() returns False when client is not connected."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        result = await client.set("some-key", "value")
        assert result is False


@pytest.mark.asyncio
async def test_delete_returns_false_when_not_connected():
    """delete() returns False when client is not connected."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        result = await client.delete("some-key")
        assert result is False


@pytest.mark.asyncio
async def test_exists_returns_false_when_not_connected():
    """exists() returns False when client is not connected."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        result = await client.exists("some-key")
        assert result is False


@pytest.mark.asyncio
async def test_ping_returns_false_when_not_connected():
    """ping() returns False when client is not connected."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        result = await client.ping()
        assert result is False


# ---------- Async operations with mocked connected client ----------


@pytest.mark.asyncio
async def test_get_returns_deserialized_value():
    """get() returns deserialized value when key exists."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        mock_redis = AsyncMock()
        mock_redis.get.return_value = '{"status": "ok"}'
        client._client = mock_redis

        result = await client.get("my-key")
        assert result == {"status": "ok"}
        mock_redis.get.assert_awaited_once_with("my-key")


@pytest.mark.asyncio
async def test_get_returns_none_for_missing_key():
    """get() returns None when key does not exist."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        client._client = mock_redis

        result = await client.get("missing-key")
        assert result is None


@pytest.mark.asyncio
async def test_get_handles_redis_error():
    """get() returns None on Redis error."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        mock_redis = AsyncMock()
        mock_redis.get.side_effect = RedisError("Redis down")
        client._client = mock_redis

        result = await client.get("error-key")
        assert result is None


@pytest.mark.asyncio
async def test_set_without_ttl():
    """set() stores value without TTL."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        mock_redis = AsyncMock()
        mock_redis.set.return_value = True
        client._client = mock_redis

        result = await client.set("key", {"data": 1})
        assert result is True
        mock_redis.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_set_with_ttl():
    """set() stores value with TTL via setex."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        mock_redis = AsyncMock()
        mock_redis.setex.return_value = True
        client._client = mock_redis

        result = await client.set("key", {"data": 1}, ttl=300)
        assert result is True
        mock_redis.setex.assert_awaited_once()


@pytest.mark.asyncio
async def test_set_handles_error():
    """set() returns False on Redis error."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        mock_redis = AsyncMock()
        mock_redis.set.side_effect = RedisError("Redis down")
        client._client = mock_redis

        result = await client.set("key", "value")
        assert result is False


@pytest.mark.asyncio
async def test_delete_success():
    """delete() returns True when key is deleted."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        mock_redis = AsyncMock()
        mock_redis.delete.return_value = 1
        client._client = mock_redis

        result = await client.delete("key")
        assert result is True


@pytest.mark.asyncio
async def test_delete_handles_error():
    """delete() returns False on Redis error."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        mock_redis = AsyncMock()
        mock_redis.delete.side_effect = RedisError("Redis down")
        client._client = mock_redis

        result = await client.delete("key")
        assert result is False


@pytest.mark.asyncio
async def test_exists_success():
    """exists() returns True when key exists."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        mock_redis = AsyncMock()
        mock_redis.exists.return_value = 1
        client._client = mock_redis

        result = await client.exists("key")
        assert result is True


@pytest.mark.asyncio
async def test_exists_handles_error():
    """exists() returns False on error."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        mock_redis = AsyncMock()
        mock_redis.exists.side_effect = RedisError("Redis down")
        client._client = mock_redis

        result = await client.exists("key")
        assert result is False


@pytest.mark.asyncio
async def test_ping_success():
    """ping() returns True when Redis responds."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        mock_redis = MagicMock()
        mock_ping = AsyncMock(return_value=True)
        mock_redis.ping = mock_ping
        client._client = mock_redis

        result = await client.ping()
        assert result is True


@pytest.mark.asyncio
async def test_ping_handles_error():
    """ping() returns False on error."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = RedisError("connection lost")
        client._client = mock_redis

        result = await client.ping()
        assert result is False


# ---------- Context manager tests ----------


@pytest.mark.asyncio
async def test_context_manager():
    """RedisClient can be used as an async context manager."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        with patch.object(client, "connect", new_callable=AsyncMock) as mock_connect:
            with patch.object(client, "disconnect", new_callable=AsyncMock) as mock_disconnect:
                async with client as c:
                    assert c is client
                mock_connect.assert_awaited_once()
                mock_disconnect.assert_awaited_once()


# ---------- disconnect tests ----------


@pytest.mark.asyncio
async def test_disconnect_closes_resources():
    """disconnect() closes both client and pool."""
    with patch("src.cache.redis_client.get_settings") as mock_settings:
        mock_settings.return_value.redis.url = "redis://localhost:6390/0"
        mock_settings.return_value.redis.max_connections = 10
        mock_settings.return_value.redis.socket_timeout = 5.0
        mock_settings.return_value.redis.socket_connect_timeout = 2.0
        mock_settings.return_value.redis.decode_responses = True

        client = RedisClient()
        mock_redis_client = AsyncMock()
        mock_pool = AsyncMock()
        client._client = mock_redis_client
        client._pool = mock_pool

        await client.disconnect()
        mock_redis_client.aclose.assert_awaited_once()
        mock_pool.aclose.assert_awaited_once()
        assert client._client is None
        assert client._pool is None


# ---------- Module-level functions ----------


@pytest.mark.asyncio
async def test_close_redis_when_no_client():
    """close_redis() is safe when no global client exists."""
    import src.cache.redis_client as module

    original = module._redis_client
    module._redis_client = None
    await close_redis()
    assert module._redis_client is None
    module._redis_client = original
