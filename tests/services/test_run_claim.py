"""Durable per-run execution claim — DB-backed second guard behind the Redis
lease. Proves exactly one instance wins on the fail-open path, the claim's
freshness/staleness semantics, and that the dispatcher fail-open path skips a
second replica when Redis is down.
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid7

import pytest
from sqlalchemy import func, update

from src.database.models import Deliverable, Project, Run
from src.services.runs.run_claim import _CLAIM_TTL, RunDbClaim

# Needs a real DB (the claim is a DB row); skipped by default like other DB tests.
pytestmark = pytest.mark.asyncio


def _claim(session_factory) -> RunDbClaim:
    # A fresh owner token per instance ≈ a distinct pod.
    return RunDbClaim(session_factory, instance_id=uuid7().hex)


async def _seed_run(session_factory) -> UUID:
    """Insert one project→deliverable→run tree; return the run id."""
    async with session_factory() as s:
        project = Project(
            id=uuid7(),
            tenant_id=uuid7(),
            owner_subject_id=uuid7(),
            visibility="private",
            type="promo",
            title="claim-test",
        )
        deliverable = Deliverable(id=uuid7(), project_id=project.id, name="d1", format="16x9")
        run = Run(id=uuid7(), deliverable_id=deliverable.id, blueprint_id="promo_v1")
        s.add_all([project, deliverable, run])
        await s.commit()
        return run.id


async def _backdate_claim(session_factory, run_id: UUID, seconds: int) -> None:
    """Push exec_claimed_at into the past to simulate a stale (abandoned) claim."""
    async with session_factory() as s:
        await s.execute(
            update(Run)
            .where(Run.id == run_id)
            .values(exec_claimed_at=func.now() - func.make_interval(0, 0, 0, 0, 0, 0, seconds))
        )
        await s.commit()


async def test_claim_blocks_second_instance(db_session_factory) -> None:
    """Two claims (≈ two pods) over one DB — only one can win a run's claim."""
    run_id = await _seed_run(db_session_factory)
    c1, c2 = _claim(db_session_factory), _claim(db_session_factory)
    assert c1._instance_id != c2._instance_id

    assert await c1.acquire(run_id) is True
    assert await c2.acquire(run_id) is False  # c1 holds a fresh claim


async def test_claim_release_is_owner_scoped(db_session_factory) -> None:
    """A foreign release must not drop the owner's claim; the owner's release
    frees it for the next claimant."""
    run_id = await _seed_run(db_session_factory)
    c1, c2 = _claim(db_session_factory), _claim(db_session_factory)

    assert await c1.acquire(run_id) is True
    await c2.release(run_id)  # not the owner — no-op
    assert await c2.acquire(run_id) is False  # c1 still owns it

    await c1.release(run_id)  # owner releases
    assert await c2.acquire(run_id) is True  # now claimable


async def test_fresh_reflects_claim_lifecycle(db_session_factory) -> None:
    """fresh() is False when unclaimed, True while a live claim is held, False
    once released."""
    run_id = await _seed_run(db_session_factory)
    c1 = _claim(db_session_factory)

    assert await c1.fresh(run_id) is False  # never claimed
    assert await c1.acquire(run_id) is True
    assert await c1.fresh(run_id) is True  # claim held
    await c1.release(run_id)
    assert await c1.fresh(run_id) is False  # released


async def test_stale_claim_is_reclaimable_and_not_fresh(db_session_factory) -> None:
    """A claim older than the TTL (a hard-killed pod) is stale: fresh() is False
    and another instance can reclaim it."""
    run_id = await _seed_run(db_session_factory)
    c1, c2 = _claim(db_session_factory), _claim(db_session_factory)

    assert await c1.acquire(run_id) is True
    await _backdate_claim(db_session_factory, run_id, _CLAIM_TTL + 30)

    assert await c1.fresh(run_id) is False  # stale → not a liveness signal
    assert await c2.acquire(run_id) is True  # reclaimable by another pod


async def test_heartbeat_refreshes_claim(db_session_factory) -> None:
    """The heartbeat update refreshes exec_claimed_at so a long run's claim never
    goes stale under it (validated via a direct refresh, not a real sleep)."""
    run_id = await _seed_run(db_session_factory)
    c1 = _claim(db_session_factory)
    assert await c1.acquire(run_id) is True
    await _backdate_claim(db_session_factory, run_id, _CLAIM_TTL + 30)
    assert await c1.fresh(run_id) is False

    # One heartbeat cycle's worth of work (the UPDATE inside _heartbeat) re-stamps
    # exec_claimed_at to now() for the owning instance → fresh again.
    async with db_session_factory() as s:
        await s.execute(
            update(Run)
            .where(Run.id == run_id, Run.exec_pod == c1._instance_id)
            .values(exec_claimed_at=func.now())
        )
        await s.commit()
    assert await c1.fresh(run_id) is True


async def test_dispatcher_fail_open_path_skips_second_replica(db_session_factory) -> None:
    """When Redis is down (lease.acquire → None), the durable DB claim lets only
    ONE replica execute: while the first is mid-run (claim held), a second
    replica's dispatch on the same run must skip.

    (The claim-level contention is covered by test_claim_blocks_second_instance;
    this proves _run WIRES the claim on the fail-open path — acquires it, and
    skips execution when it's already held.)"""
    from unittest.mock import AsyncMock, patch

    import fakeredis.aioredis

    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher

    run_id = await _seed_run(db_session_factory)
    redis = fakeredis.aioredis.FakeRedis()
    registry = get_registry()

    d1 = RunDispatcher(session_factory=db_session_factory, registry=registry, redis=redis)
    d2 = RunDispatcher(session_factory=db_session_factory, registry=registry, redis=redis)

    executed: list[str] = []
    d1_executing = asyncio.Event()
    release_d1 = asyncio.Event()

    async def _exec_hold(_rid: UUID) -> None:
        executed.append(d1._instance_id)
        d1_executing.set()  # claim is held from here…
        await release_d1.wait()  # …until the test lets go

    async def _exec_noop(_rid: UUID) -> None:
        executed.append(d2._instance_id)

    # d1 dispatches (Redis down) and holds its DB claim mid-run.
    with (
        patch.object(d1._lease, "acquire", AsyncMock(return_value=None)),
        patch.object(d1, "_execute_run", side_effect=_exec_hold),
    ):
        t1 = asyncio.create_task(d1._run(run_id))
        await asyncio.wait_for(d1_executing.wait(), timeout=10)

        # While d1 holds the claim, d2 dispatches the SAME run (also Redis-down).
        # It must fail the DB claim and skip — _execute_run never runs.
        with (
            patch.object(d2._lease, "acquire", AsyncMock(return_value=None)),
            patch.object(d2, "_execute_run", side_effect=_exec_noop),
        ):
            await d2._run(run_id)

        assert executed == [d1._instance_id], f"only d1 must execute, got {executed}"

        release_d1.set()
        await t1

    assert executed == [d1._instance_id]
