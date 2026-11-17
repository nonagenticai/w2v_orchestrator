"""Persistent source-ingest cache — single-flight repository (DB-backed).

Proves the cache's contract against a real Postgres: hit/miss/expiry, the
single-flight coalescing that makes K concurrent identical ingests share ONE
producer call, stale-claim takeover, force-refresh, producer-error release, and
tenant isolation. Skipped by default like the other DB tests (needs the row).
"""

from __future__ import annotations

import asyncio
from uuid import uuid7

import pytest
from sqlalchemy import func, select, update

from src.database.models import SourceIngestCache
from src.services import source_cache as sc
from src.services.source_cache import CachedSource, SourceCacheRepository, source_fingerprint

pytestmark = pytest.mark.asyncio


def _repo(session_factory) -> SourceCacheRepository:
    return SourceCacheRepository(session_factory, instance_id=uuid7().hex)


def _producer(text: str = "grounding", calls: list | None = None, *, delay: float = 0.0):
    async def produce() -> CachedSource:
        if calls is not None:
            calls.append(1)
        if delay:
            await asyncio.sleep(delay)
        return CachedSource(
            text=text,
            references=["https://ref.example/1"],
            images=[{"cache_key": "media-cache/x/y.jpg", "ext": "jpg", "caption": "c"}],
        )

    return produce


async def _backdate(session_factory, fp: str, *, expires_secs=None, claimed_secs=None) -> None:
    vals: dict = {}
    if expires_secs is not None:
        vals["expires_at"] = func.now() - func.make_interval(0, 0, 0, 0, 0, 0, expires_secs)
    if claimed_secs is not None:
        vals["claimed_at"] = func.now() - func.make_interval(0, 0, 0, 0, 0, 0, claimed_secs)
    async with session_factory() as s:
        await s.execute(
            update(SourceIngestCache).where(SourceIngestCache.fingerprint == fp).values(**vals)
        )
        await s.commit()


async def _row_count(session_factory, fp: str) -> int:
    async with session_factory() as s:
        return (
            await s.execute(
                select(func.count())
                .select_from(SourceIngestCache)
                .where(SourceIngestCache.fingerprint == fp)
            )
        ).scalar_one()


async def test_miss_then_hit(db_session_factory) -> None:
    """First call produces + persists; the second serves from the row without
    calling the producer again."""
    repo = _repo(db_session_factory)
    tenant = uuid7()
    fp = source_fingerprint(tenant, "url", "https://example.com/a")
    calls: list = []

    result, source = await repo.get_or_produce(
        fp,
        tenant,
        "url",
        "https://example.com/a",
        ttl_seconds=3600,
        refresh=False,
        producer=_producer("first", calls),
    )
    assert source == "produced"
    assert result.text == "first"
    assert len(calls) == 1

    result2, source2 = await repo.get_or_produce(
        fp,
        tenant,
        "url",
        "https://example.com/a",
        ttl_seconds=3600,
        refresh=False,
        producer=_producer("SHOULD-NOT-RUN", calls),
    )
    assert source2 == "hit"
    assert result2.text == "first"  # served from the row
    assert result2.images[0]["cache_key"] == "media-cache/x/y.jpg"
    assert len(calls) == 1  # producer NOT called again


async def test_expired_row_is_refreshed(db_session_factory) -> None:
    repo = _repo(db_session_factory)
    tenant = uuid7()
    fp = source_fingerprint(tenant, "url", "https://example.com/exp")
    calls: list = []

    await repo.get_or_produce(
        fp,
        tenant,
        "url",
        "https://example.com/exp",
        ttl_seconds=3600,
        refresh=False,
        producer=_producer("stale", calls),
    )
    await _backdate(db_session_factory, fp, expires_secs=60)  # push expiry into the past

    result, source = await repo.get_or_produce(
        fp,
        tenant,
        "url",
        "https://example.com/exp",
        ttl_seconds=3600,
        refresh=False,
        producer=_producer("fresh", calls),
    )
    assert source == "produced"
    assert result.text == "fresh"
    assert len(calls) == 2  # re-ingested


async def test_force_refresh_bypasses_fresh_row(db_session_factory) -> None:
    repo = _repo(db_session_factory)
    tenant = uuid7()
    fp = source_fingerprint(tenant, "url", "https://example.com/r")
    await repo.get_or_produce(
        fp,
        tenant,
        "url",
        "https://example.com/r",
        ttl_seconds=3600,
        refresh=False,
        producer=_producer("v1"),
    )
    result, source = await repo.get_or_produce(
        fp,
        tenant,
        "url",
        "https://example.com/r",
        ttl_seconds=3600,
        refresh=True,
        producer=_producer("v2"),
    )
    assert source == "produced"
    assert result.text == "v2"
    fresh = await repo.get_fresh(fp, tenant)
    assert fresh is not None and fresh.text == "v2"


async def test_single_flight_coalesces_concurrent_ingests(db_session_factory, monkeypatch) -> None:
    """K concurrent identical ingests share ONE producer call — the winner
    produces, the losers poll the row and reuse the result."""
    monkeypatch.setattr(sc, "_POLL_INTERVAL", 0.02)
    tenant = uuid7()
    fp = source_fingerprint(tenant, "url", "https://example.com/hot")
    calls: list = []

    async def one():
        repo = _repo(db_session_factory)  # distinct instance ≈ distinct pod
        return await repo.get_or_produce(
            fp,
            tenant,
            "url",
            "https://example.com/hot",
            ttl_seconds=3600,
            refresh=False,
            producer=_producer("shared", calls, delay=0.15),
        )

    results = await asyncio.gather(*[one() for _ in range(6)])

    assert len(calls) == 1, f"producer must run once, ran {len(calls)}x"
    assert all(r.text == "shared" for r, _ in results)
    sources = sorted(s for _, s in results)
    assert sources.count("produced") == 1  # exactly one winner
    assert all(s in ("produced", "produced_after_wait") for _, s in results)
    assert await _row_count(db_session_factory, fp) == 1


async def test_stale_claim_is_taken_over(db_session_factory) -> None:
    """A pending claim older than the TTL (a crashed winner) is reclaimable — the
    next caller ingests rather than waiting forever."""
    tenant = uuid7()
    fp = source_fingerprint(tenant, "url", "https://example.com/stale")

    # A crashed winner: claim the row, then never complete it, and backdate it.
    dead = _repo(db_session_factory)
    assert await dead.try_claim(fp, tenant, "url", "https://example.com/stale") is True
    await _backdate(db_session_factory, fp, claimed_secs=sc._CLAIM_TTL + 30)

    calls: list = []
    result, source = await _repo(db_session_factory).get_or_produce(
        fp,
        tenant,
        "url",
        "https://example.com/stale",
        ttl_seconds=3600,
        refresh=False,
        producer=_producer("recovered", calls),
    )
    assert source == "produced"
    assert result.text == "recovered"
    assert len(calls) == 1


async def test_producer_error_releases_claim_and_propagates(db_session_factory) -> None:
    """A producer (KAG) failure is NOT swallowed: it propagates so the stage's
    per-source soft-fail records it, and the pending claim is released so a retry
    isn't blocked."""
    repo = _repo(db_session_factory)
    tenant = uuid7()
    fp = source_fingerprint(tenant, "url", "https://example.com/boom")

    async def boom() -> CachedSource:
        raise RuntimeError("kag exploded")

    with pytest.raises(RuntimeError, match="kag exploded"):
        await repo.get_or_produce(
            fp,
            tenant,
            "url",
            "https://example.com/boom",
            ttl_seconds=3600,
            refresh=False,
            producer=boom,
        )
    # Claim released → no ready row, and the pending row is gone (retryable).
    assert await repo.get_fresh(fp, tenant) is None
    assert await _row_count(db_session_factory, fp) == 0


async def test_release_failure_does_not_mask_the_producer_error(
    db_session_factory, monkeypatch
) -> None:
    """``_safe_release`` (called from ``get_or_produce``'s ``except Exception``
    arm around ``producer()``) must not let a release failure replace the real
    producer exception. ``release()`` internally calls ``_as_uuid()``, so a
    malformed tenant id raises a plain ``ValueError`` there — not a
    ``SQLAlchemyError`` — which is what this simulates directly on the bound
    method (a real malformed tenant_id fails earlier, in ``try_claim``, so
    monkeypatching ``release`` is the only way to exercise this specific
    failure point through ``get_or_produce``).

    Mutation check: reverting ``_safe_release``'s ``except Exception`` back to
    ``except SQLAlchemyError`` turns this red — ``excinfo.value`` becomes the
    injected ``ValueError`` instead of the original ``RuntimeError``.
    """
    repo = _repo(db_session_factory)
    tenant = uuid7()
    fp = source_fingerprint(tenant, "url", "https://example.com/boom-release")

    async def broken_release(*a, **kw):
        raise ValueError("not a real UUID")

    monkeypatch.setattr(repo, "release", broken_release)

    async def boom() -> CachedSource:
        raise RuntimeError("kag exploded")

    with pytest.raises(RuntimeError, match="kag exploded"):
        await repo.get_or_produce(
            fp,
            tenant,
            "url",
            "https://example.com/boom-release",
            ttl_seconds=3600,
            refresh=False,
            producer=boom,
        )


async def test_tenant_isolation_is_a_miss(db_session_factory) -> None:
    """The same source under a different tenant is a different fingerprint → the
    second tenant never sees the first tenant's cached row."""
    repo = _repo(db_session_factory)
    t1, t2 = uuid7(), uuid7()
    url = "https://example.com/shared"
    fp1 = source_fingerprint(t1, "url", url)
    fp2 = source_fingerprint(t2, "url", url)
    assert fp1 != fp2

    await repo.get_or_produce(
        fp1, t1, "url", url, ttl_seconds=3600, refresh=False, producer=_producer("t1-only")
    )
    assert await repo.get_fresh(fp2, t2) is None  # tenant 2: clean miss
