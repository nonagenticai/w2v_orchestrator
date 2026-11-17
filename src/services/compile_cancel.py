"""Best-effort propagation of a run cancel into the video-compile service.

Cancelling a run used to stop the ORCHESTRATOR only. The render — the most expensive
stage in the product — kept going on the compiler until it finished or its lease expired,
and was billed in full. This module is the missing edge: whenever a run is cancelled or
terminalized-failed, it tells the compiler to drop that run's compiles.

Two properties are non-negotiable here, and both are about the CALLER, not the compiler:

* it must never raise. A cancel is a user action on a live request path; if a sick or
  absent compiler could raise, the run would be stranded in ``cancelling`` and the user's
  cancel would fail outright — a strictly worse outcome than a render that runs on.
* it must never hang. The client's own request timeout is 600s, which on a wedged
  compiler would park a cancel request (and the DB transaction behind it) for ten
  minutes; :data:`~src.clients.video_compile._CANCEL_TIMEOUT_SECONDS` caps the whole
  attempt far below that.

The compiler side is durable, so "best effort" costs less than it sounds: the cancel is a
terminal DB flip that survives a compiler restart, and a job that was never reached
simply stays subject to the pre-existing reaper/work-timeout backstops.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from src.clients.video_compile import _CANCEL_TIMEOUT_SECONDS, VideoCompileClient

logger = logging.getLogger(__name__)


async def cancel_compiles_for_run(run_id: UUID | str) -> int:
    """Tell the compiler to drop every live render belonging to ``run_id``.

    Returns the number of compile jobs the compiler reports cancelled — 0 both when the
    run had no live render (the common case: most runs are cancelled before or after the
    compile stage) and when the call failed. The two are deliberately not distinguished
    in the return value: no caller may branch on it, because doing so would make the
    cancel path depend on the compiler being up. The distinction is in the LOG.
    """
    client = None
    try:
        # Construction is INSIDE the guard on purpose: it resolves service discovery and
        # reads the service-account license token, either of which can raise on a
        # misconfigured pod — and a cancel must not fail on config.
        client = VideoCompileClient(timeout=_CANCEL_TIMEOUT_SECONDS)
        n = await asyncio.wait_for(
            client.cancel_run_jobs(str(run_id)), timeout=_CANCEL_TIMEOUT_SECONDS
        )
    except Exception as exc:  # noqa: BLE001 — a cancel must never fail on this
        # Deliberately WARNING, not ERROR: the run IS cancelled either way, and the
        # compiler's work timeout + reaper still bound the orphaned render. What is lost
        # is the money between now and then, which is worth a log line and not a page.
        logger.warning("compile-cancel propagation failed for run %s: %s", run_id, exc)
        return 0
    else:
        if n:
            logger.info("cancelled %d in-flight compile job(s) for run %s", n, run_id)
        return n
    finally:
        # The client owns an httpx.AsyncClient; not closing it leaks a connection pool per
        # cancelled run. Guarded for the same reason as everything else here.
        if client is not None:
            try:
                await client.close()
            except Exception:  # teardown must not resurrect a swallow
                logger.warning(
                    "compile-cancel client close failed for run %s", run_id, exc_info=True
                )
