"""Per-run execution lease — unit tests (no DB; fakeredis only).

The lease is what prevents the reconciler from double-executing a run that
is still live on a draining pod (see RunDispatcher._run / asgi reconciler).
"""

from uuid import uuid4, uuid7

import fakeredis.aioredis
import pytest

from src.services.runs.run_lease import RunLease

pytestmark = pytest.mark.asyncio


def _lease(redis: object) -> RunLease:
    # A fresh owner token per instance ≈ a distinct pod.
    return RunLease(redis, instance_id=uuid7().hex)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


async def test_lease_blocks_second_instance() -> None:
    """Two leases (≈ two pods) share one Redis; only one can hold a run's lease.
    The other's acquire returns False — it must not execute the run."""
    redis = fakeredis.aioredis.FakeRedis()
    l1, l2 = _lease(redis), _lease(redis)
    assert l1._instance_id != l2._instance_id
    run_id = uuid4()

    assert await l1.acquire(run_id) is True
    assert await l2.acquire(run_id) is False  # contended — l2 backs off


async def test_lease_release_is_owner_scoped() -> None:
    """A foreign release must not drop the owner's lease; the owner's release
    frees it for the next claimant. Guards against a TTL-reacquire clobber."""
    redis = fakeredis.aioredis.FakeRedis()
    l1, l2 = _lease(redis), _lease(redis)
    run_id = uuid4()

    assert await l1.acquire(run_id) is True
    await l2.release(run_id)  # not the owner — no-op
    assert await l2.acquire(run_id) is False  # l1 still owns it

    await l1.release(run_id)  # owner releases
    assert await l2.acquire(run_id) is True  # now claimable


async def test_lease_acquire_falls_open_on_redis_error() -> None:
    """If Redis is unavailable, acquire returns None (fall open) so a transient
    Redis blip doesn't wedge every run instead of executing it."""

    class _BrokenRedis:
        async def set(self, *_a: object, **_k: object) -> object:
            raise OSError("redis down")

    assert await _lease(_BrokenRedis()).acquire(uuid4()) is None


async def test_lease_present_reflects_any_owner() -> None:
    """``present`` is owner-agnostic: once *any* pod holds a run's lease, every
    lease instance sees it as present (the reaper's liveness signal)."""
    redis = fakeredis.aioredis.FakeRedis()
    l1, l2 = _lease(redis), _lease(redis)
    run_id = uuid4()

    assert await l2.present(run_id) is False  # nobody holds it yet
    assert await l1.acquire(run_id) is True
    assert await l2.present(run_id) is True  # l1's lease is visible to l2


async def test_lease_present_falls_closed_on_redis_error() -> None:
    """On a Redis error ``present`` returns True ("assume alive, skip") so the
    periodic reaper does NOT re-dispatch the whole fleet during an outage
    (which would double-execute, since acquire also falls open)."""

    class _BrokenRedis:
        async def get(self, *_a: object, **_k: object) -> object:
            raise OSError("redis down")

    assert await _lease(_BrokenRedis()).present(uuid4()) is True


async def test_release_swallows_the_same_error_shape_acquire_and_present_do() -> None:
    """`release` caught only `(RedisError, OSError)` while its two peers also
    catch `AttributeError` — and `release` reaches the SAME `raw.get`/`raw.delete`
    surface via `_owned`. An `AttributeError` escaping here lands in the
    dispatcher's `finally` and skips the DB-claim release that follows it, so the
    run keeps both ownership guards warm with no owner.

    `'_FakeRedis' object has no attribute 'set'` is a shape this fleet has
    actually logged, so the stub raises exactly that.
    """

    class _Rebound:
        async def get(self, *_a: object, **_k: object) -> object:
            raise AttributeError("'_FakeRedis' object has no attribute 'get'")

    lease = RunLease(_Rebound(), instance_id=uuid7().hex)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    raised: BaseException | None = None
    try:
        await lease.release(uuid4())
    except BaseException as exc:  # noqa: BLE001 — the assertion IS "nothing escapes"
        raised = exc

    assert raised is None, (
        f"release let {raised!r} escape into the dispatcher's finally, which skips "
        "the DB-claim release that follows it"
    )
