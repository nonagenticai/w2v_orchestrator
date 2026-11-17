"""Pod-lifetime loop resuming tenant offboarding jobs whose pod died (P6 ORCH-6).

Export and delete jobs run as detached in-process tasks (``tenant_offboarding.spawn``)
and heartbeat ``updated_at``. A job still ``running`` past ``RUNNING_LEASE_SECONDS``
lost its worker (rollout, OOM, crash); this loop re-claims it atomically and resumes
it — a delete from the progress it recorded, an export from scratch. Runs at startup
and then every ``OFFBOARDING_RESUME_INTERVAL_SECONDS`` (default 60; ``<= 0`` disables).
Same shape as ``provider_secret_gc``: work first, sleep after, a failed tick is
logged and retried next interval.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from src.services import tenant_offboarding as svc
from src.services.tts_storage import _get_storage

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

log = logging.getLogger(__name__)


async def resume_offboarding_once(
    session_factory: async_sessionmaker[AsyncSession],
) -> list[UUID]:
    storage = svc.as_offboarding_storage(await _get_storage())
    if storage is None:
        log.warning("offboarding resume: storage lacks the offboarding methods; skipped")
        return []
    return await svc.resume_stale_jobs(session_factory, storage)


async def _run_periodic_offboarding_resume(
    session_factory: async_sessionmaker[AsyncSession], interval_seconds: int
) -> None:
    while True:
        try:
            await resume_offboarding_once(session_factory)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("offboarding resume tick failed; retrying next tick")
        await asyncio.sleep(interval_seconds)


def start_offboarding_resume_task(
    session_factory: async_sessionmaker[AsyncSession],
) -> asyncio.Task[None] | None:
    interval = int(os.getenv("OFFBOARDING_RESUME_INTERVAL_SECONDS", "60"))
    if interval <= 0:
        log.info("offboarding resume disabled (interval=%s)", interval)
        return None
    return asyncio.create_task(
        _run_periodic_offboarding_resume(session_factory, interval),
        name="offboarding-resume",
    )
