"""``request_audit`` retention loop — carved out of ``src.asgi``.

Behaviour-preserving move: the pod-lifetime loop that prunes the high-volume
``request_audit`` table (migration 0026) inside its retention window, plus the
``_start_audit_prune_task`` helper the lifespan uses to spawn it (or skip it when
retention is disabled). ``src.asgi.lifespan`` calls ``_start_audit_prune_task``
exactly as before.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import TYPE_CHECKING

from src.config import get_settings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


async def _run_periodic_audit_prune(
    session_factory: async_sessionmaker[AsyncSession],
    retention_days: int,
    interval_seconds: int,
) -> None:
    """Prune ``request_audit`` rows older than ``retention_days`` every
    ``interval_seconds`` for the pod's whole lifetime.

    The middleware-wide audit (migration 0026) persists a row per mutation /
    auth-failure / cross-tenant access; the auth-failure rows make the table an
    unbounded disk-fill vector under an unauthenticated scanner. This loop keeps
    it inside the retention window. The DELETE is batched + index-supported so a
    sweep never holds a long lock. A single bad tick must never kill the loop —
    transient DB errors are logged and the next interval retries.

    Only started when pruning is enabled (``retention_days > 0``).
    """
    from src.database.repositories.request_audit_repository import RequestAuditRepository

    log = logging.getLogger(__name__)
    # Work FIRST, sleep after: a sleep-first loop never completes a single pass
    # in a pod that restarts more often than ``interval_seconds``, which is the
    # normal case during a rollout, so the prune silently never runs.
    while True:
        try:
            cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=retention_days)
            async with session_factory() as session:
                deleted = await RequestAuditRepository(session).prune_older_than(cutoff=cutoff)
                await session.commit()
            if deleted:
                log.info("request_audit retention: pruned %d rows older than %s", deleted, cutoff)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("request_audit prune tick failed; continuing")
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            raise


def _start_audit_prune_task(
    session_factory: async_sessionmaker[AsyncSession],
) -> asyncio.Task[None] | None:
    """Spawn the request_audit retention loop, or ``None`` when disabled.

    Disabled when ``request_audit_retention_days <= 0`` (an external log shipper
    owns retention). Extracted from ``lifespan`` to keep that function simple.
    """
    settings = get_settings()
    if settings.pipeline.request_audit_retention_days <= 0:
        return None
    return asyncio.create_task(
        _run_periodic_audit_prune(
            session_factory,
            settings.pipeline.request_audit_retention_days,
            settings.pipeline.request_audit_prune_interval_seconds,
        ),
        name="request-audit-prune",
    )
