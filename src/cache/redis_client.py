"""Redis client for w2v_orchestrator -- extends w2v_common with pub/sub.

Adds publish() and pubsub() on top of the shared w2v_common RedisClient.
All connection and pool configuration (including health_check_interval)
lives in w2v_common.
"""

from __future__ import annotations

import logging
from typing import Any

from w2v_common.cache.redis_client import RedisClient as _BaseRedisClient

from src.config import get_settings

logger = logging.getLogger(__name__)


class RedisClient(_BaseRedisClient):
    """Async Redis client with publish/pubsub support."""

    def __init__(
        self,
        redis_url: str | None = None,
        max_connections: int | None = None,
        socket_timeout: int | None = None,
        socket_connect_timeout: int | None = None,
        decode_responses: bool | None = None,
        health_check_interval: int | None = None,
    ) -> None:
        # Resolve from the orchestrator's typed settings when nothing is passed
        # in. The base class's env-var fallback would also work, but the
        # orchestrator's settings layer is the canonical config source for this
        # service (e.g. it overrides the default port to 6390).
        settings = get_settings()
        super().__init__(
            redis_url=redis_url or settings.redis.url,
            max_connections=max_connections or settings.redis.max_connections,
            socket_timeout=socket_timeout or settings.redis.socket_timeout,
            socket_connect_timeout=(
                socket_connect_timeout or settings.redis.socket_connect_timeout
            ),
            decode_responses=(
                decode_responses
                if decode_responses is not None
                else settings.redis.decode_responses
            ),
            health_check_interval=(
                health_check_interval
                if health_check_interval is not None
                else settings.redis.health_check_interval
            ),
        )

    def pubsub(self, **kwargs: Any) -> Any:
        """Return a Pub/Sub object from the underlying Redis client."""
        client = self._ensure_connected()
        return client.pubsub(**kwargs)

    async def publish(self, channel: str, message: str) -> int:
        """Publish a message to a Redis channel."""
        client = self._ensure_connected()
        return await client.publish(channel, message)


_redis_client: RedisClient | None = None


def get_redis_client() -> RedisClient:
    """Return the cached singleton RedisClient instance."""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client


async def init_redis() -> RedisClient:
    """Initialize and connect the global RedisClient."""
    client = get_redis_client()
    await client.connect()
    return client


async def close_redis() -> None:
    """Disconnect and discard the global RedisClient."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.disconnect()
        _redis_client = None
