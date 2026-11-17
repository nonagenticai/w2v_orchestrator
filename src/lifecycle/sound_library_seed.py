"""Self-healing shared sound library — seed it at startup when it is missing.

On 2026-10-14 the whole ``w2v-orchestrator`` bucket was wiped by hand and
``sound_library/shared/`` stayed empty until someone re-ran
``python -m scripts.sync_sound_library``. The lifespan now spawns this one-shot
background task instead: it never blocks readiness and never fails startup (any
error is a WARNING); it uploads only when the shared manifest is definitively
absent, and never deletes.
"""

from __future__ import annotations

import asyncio
import logging

from src.services.stages.sound_catalog import clear_sound_library_cache
from src.services.stages.sound_library_sync import seed_shared_library_if_missing
from src.services.storage_service import StorageService

logger = logging.getLogger(__name__)


async def _seed_shared_sound_library() -> None:
    svc = StorageService()
    try:
        await svc.initialize()
        if await seed_shared_library_if_missing(svc):
            # This pod may already have cached the baseline-only library.
            clear_sound_library_cache()
            logger.info("sound_library: shared library seeded at startup")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("sound_library: startup seed failed: %s", exc, exc_info=True)
    finally:
        try:
            await svc.close()
        except Exception as exc:  # noqa: BLE001 -- teardown is best-effort
            logger.debug("sound_library: storage close after seed failed: %s", exc)


def start_sound_library_seed_task() -> asyncio.Task[None]:
    """Spawn the one-shot seed in the background (the lifespan does not await it)."""
    return asyncio.create_task(_seed_shared_sound_library(), name="sound-library-seed")
