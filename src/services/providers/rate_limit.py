"""Per-tenant rate limit on LIVE provider tests.

A live test sends a request (with the tenant's key) to a tenant-chosen
``base_url``. ``POST /api/tenant/connections/{slot}/test`` and the test inside
``PUT …/{slot}`` share ONE budget per tenant, so a tenant admin (or a stolen admin
session) cannot turn the orchestrator into a request amplifier against an
endpoint, or burn its own provider quota, by hammering either route.

Fixed window, Redis-backed (shared by every pod): the key carries the window
index, ``INCR`` + ``EXPIRE`` run in one MULTI, so a key whose ``EXPIRE`` was
somehow lost still cannot outlive its window's relevance — the next window uses a
new key. No existing limiter helper exists in this repo (the reconciler's
``incr`` counter is a re-dispatch tally, not a limiter), hence this small one.

Redis unavailable (no client, never connected, or a ``RedisError``/``OSError``):
the test is ALLOWED and a WARNING is logged. The limiter is abuse damping, not an
authorization boundary — authz (tenant_admin), the kind policy and the SSRF/egress
guard all run regardless — and refusing every provider test while Redis is down
would block tenants from configuring providers at all.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from redis.exceptions import RedisError

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

KEY_PREFIX = "provider-test-rl"


@dataclass(frozen=True, slots=True)
class RateDecision:
    allowed: bool
    #: Seconds until the current window ends (``Retry-After``); 0 when allowed.
    retry_after: int = 0


class ProviderTestRateLimiter:
    """``limit`` calls per tenant per ``window_seconds`` fixed window.

    ``key_prefix`` namespaces the budget. It defaults to :data:`KEY_PREFIX` so the
    live-test budget is unchanged, and a second caller (the service lab) passes its
    own prefix rather than sharing — and silently draining — the live-test budget.
    """

    def __init__(
        self,
        redis: Redis | None,
        *,
        limit: int,
        window_seconds: int,
        clock: Callable[[], float] = time.time,
        key_prefix: str = KEY_PREFIX,
    ) -> None:
        self._redis = redis
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._prefix = key_prefix

    def key(self, tenant_id: UUID, now: float) -> str:
        return f"{self._prefix}:{tenant_id}:{int(now // self._window)}"

    async def hit(self, tenant_id: UUID) -> RateDecision:
        """Count one live test for ``tenant_id``; refuse past the limit."""
        if self._limit <= 0:
            return RateDecision(True)
        if self._redis is None:
            logger.warning("provider test rate limit skipped: no redis connection")
            return RateDecision(True)
        now = self._clock()
        key = self.key(tenant_id, now)
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.incr(key)
                pipe.expire(key, self._window)
                count, _ = await pipe.execute()
        except (RedisError, OSError) as exc:
            logger.warning("provider test rate limit skipped: redis error: %s", exc)
            return RateDecision(True)
        if int(count) <= self._limit:
            return RateDecision(True)
        window_end = (now // self._window + 1) * self._window
        return RateDecision(False, max(1, math.ceil(window_end - now)))
