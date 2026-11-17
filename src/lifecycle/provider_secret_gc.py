"""Pod-lifetime loop purging superseded, unpinned provider secrets (P3 WP3).

Same ``asyncio.create_task`` + ``while True`` shape as ``audit_prune``: work
first, sleep after, a failed tick is logged and retried next interval. A pass
with nothing superseded is one cheap function call that purges 0 rows.
Interval: ``PROVIDER_GC_INTERVAL_SECONDS`` (``<= 0`` disables); grace:
``PROVIDER_GC_GRACE_SECONDS``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import TYPE_CHECKING

from src.services.providers.gc import purge_unpinned_secrets
from src.services.providers.settings import get_provider_policy

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

log = logging.getLogger(__name__)


async def run_provider_secret_gc_once(
    session_factory: async_sessionmaker[AsyncSession], grace: timedelta
) -> int:
    async with session_factory() as session:
        purged = await purge_unpinned_secrets(session, grace=grace)
        await session.commit()
    if purged:
        log.info("provider secret GC: purged %d superseded unpinned secret(s)", purged)
    return purged


async def _run_periodic_provider_secret_gc(
    session_factory: async_sessionmaker[AsyncSession],
    interval_seconds: int,
    grace: timedelta,
) -> None:
    while True:
        try:
            await run_provider_secret_gc_once(session_factory, grace)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("provider secret GC tick failed; nothing purged, retrying next tick")
        await asyncio.sleep(interval_seconds)


def start_provider_secret_gc_task(
    session_factory: async_sessionmaker[AsyncSession],
) -> asyncio.Task[None] | None:
    """Lifespan hook; ``None`` when ``PROVIDER_GC_INTERVAL_SECONDS <= 0``."""
    policy = get_provider_policy()
    if policy.gc_interval_seconds <= 0:
        log.info("provider secret GC disabled (interval=%s)", policy.gc_interval_seconds)
        return None
    return asyncio.create_task(
        _run_periodic_provider_secret_gc(
            session_factory,
            policy.gc_interval_seconds,
            timedelta(seconds=policy.gc_grace_seconds),
        ),
        name="provider-secret-gc",
    )
