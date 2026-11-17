"""Two ways one run gets executed (and paid for) twice, or a cancel is lost.

* ``RunDbClaim`` heartbeat: losing the durable claim used to stop only the
  HEARTBEAT while this pod kept executing the DAG next to the replica that took
  the claim over.
* ``RunDispatcher.cancel``: a DB blip inside the unshielded CancelledError
  handler propagated out, 500'ing the cancel route before ``mark_cancelled``
  and the paid-child sweeps ever ran.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import src.services.runs.run_claim as run_claim_mod
from src.services.runs.run_claim import RunDbClaim

pytestmark = pytest.mark.asyncio


class _FakeResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _FakeSession:
    def __init__(self, behaviour) -> None:
        self._behaviour = behaviour

    async def execute(self, *_a, **_kw):
        return self._behaviour()

    async def commit(self) -> None:
        return None

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_a: object) -> bool:
        return False


def _factory_for(behaviour):
    def _factory() -> _FakeSession:
        return _FakeSession(behaviour)

    return _factory


@pytest.fixture
def _fast_claim(monkeypatch):
    monkeypatch.setattr(run_claim_mod, "_CLAIM_HEARTBEAT", 0.01)
    monkeypatch.setattr(run_claim_mod, "_CLAIM_TTL", 0.02)
    monkeypatch.setattr(run_claim_mod, "bind_run_tenant", AsyncMock())


async def test_heartbeat_reports_a_reclaimed_claim_to_on_lost(_fast_claim) -> None:
    """rowcount==0 means another replica stale-reclaimed us. The old code only
    ``return``ed — the run kept executing here alongside the new owner."""
    claim = RunDbClaim(_factory_for(lambda: _FakeResult(0)), instance_id="pod-a")
    lost = asyncio.Event()
    calls = []

    def _on_lost() -> None:
        calls.append(1)
        lost.set()

    task = claim.start_heartbeat(uuid4(), on_lost=_on_lost)
    await asyncio.wait_for(lost.wait(), timeout=3)
    await asyncio.wait_for(task, timeout=3)
    assert calls == [1], "on_lost must fire exactly once when the claim is reclaimed"
    assert task.done() and not task.cancelled()


async def test_heartbeat_treats_a_ttl_of_failures_as_a_lost_claim(_fast_claim) -> None:
    """The 2026-08-20 leaked ``row_security=off`` incident 500'd every statement
    while the run was live. Refreshes that keep erroring for a whole TTL mean the
    claim IS stale and another replica may take it — the old loop swallowed them
    forever."""

    def _boom():
        raise RuntimeError("row_security=off leaked onto this connection")

    claim = RunDbClaim(_factory_for(_boom), instance_id="pod-a")
    lost = asyncio.Event()
    calls = []

    def _on_lost() -> None:
        calls.append(1)
        lost.set()

    task = claim.start_heartbeat(uuid4(), on_lost=_on_lost)
    await asyncio.wait_for(lost.wait(), timeout=3)
    await asyncio.wait_for(task, timeout=3)
    assert calls == [1], "a TTL of failed refreshes must report the claim as lost"
    assert task.done() and not task.cancelled(), "the heartbeat must stop, not spin"


async def test_a_lost_claim_halts_the_running_dag(_fast_claim, monkeypatch) -> None:
    """Production call site: ``RunDispatcher._run`` on the Redis-fail-open path.

    The heartbeat losing the claim has to stop the WORK, not just itself — every
    non-idempotent submit stage (render / TTS / upload) otherwise runs on both
    replicas and is billed twice.
    """
    import fakeredis.aioredis

    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher

    redis = fakeredis.aioredis.FakeRedis()
    dispatcher = RunDispatcher(
        session_factory=_factory_for(lambda: _FakeResult(0)),
        registry=get_registry(),
        redis=redis,
    )
    try:
        run_id = uuid4()
        # Redis falls OPEN (acquire -> None) so the durable DB claim is the guard.
        monkeypatch.setattr(dispatcher._lease, "acquire", AsyncMock(return_value=None))
        monkeypatch.setattr(dispatcher._claim, "acquire", AsyncMock(return_value=True))
        monkeypatch.setattr(dispatcher._claim, "release", AsyncMock())

        started = asyncio.Event()

        async def _long_dag(_rid) -> None:
            started.set()
            await asyncio.sleep(30)

        monkeypatch.setattr(dispatcher, "_execute_run", _long_dag)

        task = asyncio.create_task(dispatcher._run(run_id))
        await asyncio.wait_for(started.wait(), timeout=3)
        for _ in range(300):
            if task.done():
                break
            await asyncio.sleep(0.01)
        assert task.done(), "losing the claim left this pod executing the DAG"
        assert task.cancelled()
    finally:
        await redis.aclose()


async def test_cancel_returns_false_when_the_run_task_handler_raises() -> None:
    """``cancel`` returned True on task EXISTENCE and let any non-CancelledError
    escape. A DB blip in the unshielded handler therefore 500'd the cancel route
    before ``mark_cancelled`` or either paid-child sweep ran."""
    import fakeredis.aioredis

    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher

    redis = fakeredis.aioredis.FakeRedis()
    dispatcher = RunDispatcher(
        session_factory=_factory_for(lambda: _FakeResult(1)),
        registry=get_registry(),
        redis=redis,
    )
    try:
        run_id = uuid4()

        async def _handler_blows_up() -> None:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise RuntimeError("connection reset in the CancelledError handler") from None

        task = asyncio.create_task(_handler_blows_up())
        await asyncio.sleep(0)
        dispatcher._tasks[run_id] = task

        assert await dispatcher.cancel(run_id) is False, (
            "a failed in-task cancel must route the caller onto the DB fallback"
        )
    finally:
        dispatcher._tasks.pop(run_id, None)
        await redis.aclose()


# ---------------------------------------------------------------------------
# RunLease — the SAME state machine as RunDbClaim above, and until now the only
# one of the two with no way to report a lost guard. The DB claim does NOT
# backstop it: ``RunDispatcher._run`` takes the claim ONLY when ``acquire`` fell
# open, so a run that started under a HEALTHY Redis lease has no claim at all.
# ---------------------------------------------------------------------------


@pytest.fixture
def _fast_lease(monkeypatch):
    import src.services.runs.run_lease as run_lease_mod

    monkeypatch.setattr(run_lease_mod, "_LEASE_HEARTBEAT", 0.01)
    monkeypatch.setattr(run_lease_mod, "_LEASE_TTL", 0.02)


class _StubRawRedis:
    """Raw-client stand-in for the two lease-loss shapes."""

    def __init__(self, *, get_value: object = None, raise_on_get: Exception | None = None) -> None:
        self._get_value = get_value
        self._raise = raise_on_get

    async def get(self, _key: str):
        if self._raise is not None:
            raise self._raise
        return self._get_value

    async def expire(self, _key: str, _ttl: int) -> bool:
        return True


async def _drive_lease_heartbeat(raw, monkeypatch) -> list[int]:
    from src.services.runs.run_lease import RunLease

    lease = RunLease(raw, instance_id="pod-a")
    monkeypatch.setattr(lease, "raw_redis", lambda: raw)
    lost = asyncio.Event()
    calls: list[int] = []

    def _on_lost() -> None:
        calls.append(1)
        lost.set()

    task = lease.start_heartbeat(uuid4(), _on_lost)
    await asyncio.wait_for(lost.wait(), timeout=3)
    await asyncio.wait_for(task, timeout=3)
    assert task.done() and not task.cancelled(), "the heartbeat must stop, not spin"
    return calls


async def test_lease_heartbeat_reports_a_reacquired_lease_to_on_lost(
    _fast_lease, monkeypatch
) -> None:
    """The key is gone (TTL-expired) or now holds another pod's token — that pod
    is entitled to execute this run. Stopping only the heartbeat left BOTH pods
    running the same non-idempotent submit stages."""
    assert await _drive_lease_heartbeat(_StubRawRedis(get_value=b"pod-b"), monkeypatch) == [1]


async def test_lease_heartbeat_treats_a_ttl_of_failures_as_a_lost_lease(
    _fast_lease, monkeypatch
) -> None:
    """A renew that has been failing for a whole TTL means the key MUST already
    have expired and is reacquirable. The old loop logged and span forever."""
    from redis.exceptions import ConnectionError as RedisConnectionError

    assert await _drive_lease_heartbeat(
        _StubRawRedis(raise_on_get=RedisConnectionError("redis is down")), monkeypatch
    ) == [1]


async def test_a_lost_lease_halts_the_running_dag(_fast_lease, monkeypatch) -> None:
    """Production call site: ``RunDispatcher._run`` on the NORMAL (lease-owned)
    path — the one with no DB claim behind it."""
    import fakeredis.aioredis

    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher

    redis = fakeredis.aioredis.FakeRedis()
    dispatcher = RunDispatcher(
        session_factory=_factory_for(lambda: _FakeResult(1)),
        registry=get_registry(),
        redis=redis,
    )
    try:
        run_id = uuid4()
        # Healthy lease acquire -> True, so `db_claimed` stays False and the Redis
        # lease is the ONLY guard on this run.
        monkeypatch.setattr(dispatcher._lease, "acquire", AsyncMock(return_value=True))
        monkeypatch.setattr(dispatcher._lease, "release", AsyncMock())
        # ...and then it is reacquired by pod-b under us.
        monkeypatch.setattr(
            dispatcher._lease, "raw_redis", lambda: _StubRawRedis(get_value=b"pod-b")
        )

        started = asyncio.Event()

        async def _long_dag(_rid) -> None:
            started.set()
            await asyncio.sleep(30)

        monkeypatch.setattr(dispatcher, "_execute_run", _long_dag)

        task = asyncio.create_task(dispatcher._run(run_id))
        await asyncio.wait_for(started.wait(), timeout=3)
        for _ in range(300):
            if task.done():
                break
            await asyncio.sleep(0.01)
        assert task.done(), "losing the Redis lease left this pod executing the DAG"
        assert task.cancelled()
    finally:
        await redis.aclose()
