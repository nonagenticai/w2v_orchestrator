"""Unwrap the w2v_common ``RedisClient`` wrapper down to the raw redis-py client.

Production injects ``src.cache.redis_client.RedisClient`` (a ``w2v_common``
wrapper) wherever a "redis handle" is passed around — ``app.state.redis`` is
that wrapper. Unit tests inject a raw ``fakeredis.aioredis.FakeRedis``. The two
are NOT interchangeable, and code that is annotated ``Redis`` but handed the
wrapper breaks in three separate ways:

* **Different keyword.** The wrapper's ``set`` takes ``ttl=``; redis-py's takes
  ``ex=``. Passing ``ex=`` to the wrapper is a ``TypeError`` on the first write
  — which is exactly how the panel-image job store 500'd in production.
* **Different value.** The wrapper JSON-serialises on ``set`` and
  JSON-*deserialises* on ``get``, so what comes back is a ``dict``, not the
  stored JSON text. That breaks ``model_validate_json`` and any symmetric
  token compare.
* **Different failure mode.** The wrapper swallows ``RedisError`` and returns
  ``False``/``None``, so ``except RedisError`` around a wrapper call is dead
  code, and a caller cannot tell "key absent" from "Redis is down".

So the callers that need redis-py semantics unwrap ONCE, at their boundary, and
keep an honest ``Redis`` annotation from there inward. This is the single copy
of that unwrapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from w2v_common.cache.redis_client import RedisClient


def raw_redis(client: Redis | RedisClient) -> Redis:
    """Return the raw ``redis.asyncio.Redis`` underneath ``client``.

    Accepts either shape. A ``RedisClient`` wrapper is dropped to its live
    client via ``_ensure_connected`` — which raises ``RuntimeError`` when
    ``connect()`` was never called, so a never-connected wrapper fails loudly
    here rather than silently no-op'ing on every later call. An already-raw
    client (production's lower layers, and the ``FakeRedis`` used by tests) is
    returned unchanged.
    """
    ensure = getattr(client, "_ensure_connected", None)
    return cast("Redis", ensure() if callable(ensure) else client)
