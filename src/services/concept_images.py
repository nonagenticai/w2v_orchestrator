"""The ONE way the orchestrator generates a concept image through ACG.

Every concept-image caller (``ai_image_gen``, ``course_thumbnail``, the
storyboard panel renderer) goes through :func:`generate_concept_image`.
``tests/unit/test_concept_image_single_path.py`` fails if any other module calls
the ACG client's concept-image methods directly, because each caller that held
its own submit/wait got the same thing wrong.

Why: the self-hosted image backend renders ONE image at a time (~20-30 s). A
job's wait is therefore mostly QUEUE time behind other runs' images, and ACG
cannot say when a job leaves that queue: it flips ``pending`` -> ``processing``
BEFORE its local_image provider takes the single in-pod slot, so ``processing``
includes the wait and there is no "rendering started" state to time a render
from. A deadline sized for one render (180 s) applied to queue + render timed
the 9th of 9 queued jobs out at 169 s+, and the retry rendered it a second time.

The contract, so no caller can hold its own short deadline again:

* the caller names its whole BUDGET for this image (queue + render + retries);
  the wait deadline is that budget minus :data:`DEADLINE_MARGIN_S`, so the
  client's own ``TimeoutError`` fires before the caller's outer timeout cancels
  the call mid-poll;
* every submit carries the caller's Idempotency-Key, so a retry re-attaches to
  the ORIGINAL job (ACG returns it) instead of rendering a second one;
* a retry only happens on a transient failure and only inside the same
  deadline; running out of deadline is final, never retried;
* only after the FINAL attempt is the job cancelled (best-effort). A cancel
  releases the Idempotency-Key, so cancelling before a retry would send the
  retry to the back of the queue as a fresh job.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import httpx

from src.services.stages.upstream import PermanentUpstreamError, TransientUpstreamError

if TYPE_CHECKING:
    from src.clients.ai_content_generation_client import AIContentGenerationClient
    from src.clients.schemas.ai_content_generation import AcgConceptImageJob

log = logging.getLogger(__name__)

#: Slack between the wait deadline and the caller's budget.
DEADLINE_MARGIN_S = 30.0
#: Default whole-image budget: a deep queue on the one-at-a-time backend
#: (~20 images at ~25 s) plus the render itself.
DEFAULT_BUDGET_S = 600.0
#: Submits per image (initial + retries), all with the same Idempotency-Key.
DEFAULT_ATTEMPTS = 2
#: Below this much deadline left, a retry cannot wait for anything useful.
_MIN_WAIT_S = 5.0
#: What a second attempt can fix. ``TimeoutError`` (an ``OSError``) is caught
#: before this and is final: the deadline is shared, so there is no time left.
#: ``PermanentUpstreamError`` (e.g. a byo_required tenant's
#: ``SlotNotConfiguredError``) is re-raised untouched: each retry is a paid submit.
_RETRY_ON: tuple[type[BaseException], ...] = (
    RuntimeError,
    OSError,
    ValueError,
    httpx.HTTPError,
    TransientUpstreamError,
)


def _now() -> float:
    return time.monotonic()


async def generate_concept_image(
    client: AIContentGenerationClient,
    prompt: str,
    config: dict | None,
    *,
    idempotency_key: str,
    budget_s: float = DEFAULT_BUDGET_S,
    attempts: int = DEFAULT_ATTEMPTS,
) -> AcgConceptImageJob:
    """Submit, wait (queue + render) and return the completed ACG job envelope.

    ``budget_s`` is everything the caller allows for this image. Raises
    ``TimeoutError`` when the deadline runs out, the last retryable error when
    every attempt failed, and any other error (incl. ``PermanentUpstreamError``)
    immediately.
    """
    if budget_s <= DEADLINE_MARGIN_S + _MIN_WAIT_S:
        raise ValueError(
            f"concept-image budget {budget_s}s must exceed the "
            f"{DEADLINE_MARGIN_S + _MIN_WAIT_S}s margin"
        )
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    deadline = _now() + budget_s - DEADLINE_MARGIN_S
    job_id: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            job_id = await client.create_concept_image(
                prompt, config, idempotency_key=idempotency_key
            )
            remaining = deadline - _now()
            if remaining < _MIN_WAIT_S:
                raise TimeoutError(
                    f"concept-image job {job_id}: deadline exhausted "
                    f"({budget_s - DEADLINE_MARGIN_S:.0f}s, queue + render)"
                )
            return await client.wait_for_concept_image(job_id, timeout=remaining)
        except TimeoutError:
            await _cancel_abandoned(client, job_id)
            raise
        except PermanentUpstreamError:
            raise
        except _RETRY_ON as exc:
            if attempt == attempts or deadline - _now() < _MIN_WAIT_S:
                await _cancel_abandoned(client, job_id)
                raise
            log.warning(
                "concept image %s: attempt %d/%d failed (%s); retrying with the same "
                "Idempotency-Key",
                job_id,
                attempt,
                attempts,
                exc,
            )
    raise AssertionError("unreachable: the loop returns or raises")  # pragma: no cover


async def _cancel_abandoned(client: AIContentGenerationClient, job_id: str | None) -> None:
    """Best-effort: stop ACG rendering an image nobody will wait for any more.

    A timed-out job otherwise keeps its place in the one-at-a-time GPU queue and
    renders anyway, delaying every other run's images. A 409 (already finished,
    or unknown) is fine. A cancel failure never replaces the error being raised.
    """
    if job_id is None:
        return
    try:
        cancelled = await client.cancel_concept_image(job_id)
    except Exception as exc:  # noqa: BLE001 -- best-effort cancel; the original error wins
        log.warning("concept image: cancel of abandoned job %s failed: %s", job_id, exc)
        return
    log.info(
        "concept image: abandoned job %s %s",
        job_id,
        "cancelled" if cancelled else "was not cancellable (already finished)",
    )
