"""``ProviderTestRateLimiter`` — fixed window, per tenant, fail-open on Redis loss."""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid7

import fakeredis.aioredis
import pytest
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from src.services.providers.rate_limit import ProviderTestRateLimiter

pytestmark = pytest.mark.asyncio


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


async def test_fixed_window_refuses_past_limit_and_resets_next_window() -> None:
    redis = fakeredis.aioredis.FakeRedis()
    clock = _Clock(100.0)
    limiter = ProviderTestRateLimiter(redis, limit=2, window_seconds=60, clock=clock)
    tenant, other = uuid7(), uuid7()

    assert (await limiter.hit(tenant)).allowed
    assert (await limiter.hit(tenant)).allowed
    refused = await limiter.hit(tenant)
    assert not refused.allowed
    assert refused.retry_after == 20  # window [60, 120) ends at 120
    assert (await limiter.hit(other)).allowed  # per tenant
    # The key carries a TTL so it never outlives its window.
    ttl = await redis.ttl(limiter.key(tenant, clock.now))
    assert 0 < ttl <= 60

    clock.now = 120.0  # next window
    assert (await limiter.hit(tenant)).allowed


async def test_limit_zero_disables() -> None:
    limiter = ProviderTestRateLimiter(fakeredis.aioredis.FakeRedis(), limit=0, window_seconds=60)
    tenant = uuid7()
    assert all([(await limiter.hit(tenant)).allowed for _ in range(50)])


class _BrokenPipe:
    async def __aenter__(self) -> _BrokenPipe:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def incr(self, key: str) -> None: ...

    def expire(self, key: str, seconds: int) -> None: ...

    async def execute(self) -> list[Any]:
        raise RedisConnectionError("down")


class _BrokenRedis:
    def pipeline(self, transaction: bool = True) -> _BrokenPipe:
        return _BrokenPipe()


async def test_redis_error_and_missing_redis_fail_open() -> None:
    tenant = uuid7()
    broken = ProviderTestRateLimiter(cast(Redis, _BrokenRedis()), limit=1, window_seconds=60)
    assert (await broken.hit(tenant)).allowed
    assert (await broken.hit(tenant)).allowed
    absent = ProviderTestRateLimiter(None, limit=1, window_seconds=60)
    assert (await absent.hit(tenant)).allowed
    assert (await absent.hit(tenant)).allowed
