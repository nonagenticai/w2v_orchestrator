"""Wait out html_render's claim queue BEFORE the generation budget starts.

Every design stage submits to ``html_render_service`` and then polls with a
budget sized for *generation* — an LLM redesign plus screenshot renders, tens of
minutes. html_render is a Postgres claim-queue (``pending`` -> ``processing``,
see its ``src/services/job_manager.py``) served by a small replica count, so a
submitted job can sit unclaimed for a long time while every replica is busy on
someone else's job.

Charging that queue time to the generation budget is what this module exists to
stop. Measured live on 2026-08-10 (website_review run
``019fea75-98ee-70d5-84bb-044e12fddd49``, design job ``019fec3f-…``):

    15:17:18Z  submitted            (uuid7 timestamp)
    15:40:26Z  claimed by a worker  -> 23 minutes queued
    16:38:32Z  stage failed         -> 8 x 590s budget spent, only 58 min of it
                                       on actual generation
    16:47:39Z  html_render finished that same job successfully

The run failed and a design that completed was thrown away. Waiting here is free
— no worker is running — so the queue gets its own generous ceiling and the
generation budget starts when generation does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from w2v_common.stages import stage_call

if TYPE_CHECKING:
    from src.clients.design_client import DesignClient

# Matches DesignClient._QUEUE_TIMEOUT_SECONDS; the outer stage_call needs a
# little more than the inner wait so the inner TimeoutError (which names the job
# and says "never claimed") is what surfaces, not a generic stage timeout.
_QUEUE_STAGE_TIMEOUT_SECONDS = 1810.0


async def await_design_start(client: DesignClient, job_id: str, stage_name: str) -> None:
    """Block until ``job_id`` leaves html_render's ``pending`` queue.

    ``retries=0``: a job that is never claimed is not made more likely to be
    claimed by asking again, and the wait is already long enough that re-running
    it would double an already-generous ceiling.

    Returns None rather than the queued duration on purpose — the caller has no
    use for it and the client logs it where the real elapsed time lives.
    """
    await stage_call(
        lambda: client.wait_until_started(job_id),
        timeout_seconds=_QUEUE_STAGE_TIMEOUT_SECONDS,
        retries=0,
        stage_name=f"{stage_name}.queue",
    )
