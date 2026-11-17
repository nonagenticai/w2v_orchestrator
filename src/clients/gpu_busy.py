"""Caller half of the GPU admission contract (#1106): honour ``gpu_busy``, then give up transiently.

The GPU services and the gateways in front of them shed load with **503** (or 429)
``{"error": "gpu_busy", "lane": "<name>", "retry_after": N}`` + ``Retry-After: N``.
Callers own their own concurrency and must:

* wait what the server said — the ``Retry-After`` header first, then the body's
  ``retry_after`` — clamped to ``1..120`` s; with neither, exponential backoff
  (2 → 60 s, jittered);
* stop after a bounded TOTAL deadline (``GPU_BUSY_DEADLINE_SECONDS``), and then
* raise :class:`GpuBusyError` — a :class:`TransientUpstreamError` attributed
  ``external``: the stage layer retries/re-drives it, it never terminally fails
  a run as if the request were wrong.

The settings live here (not in ``src/config.py``) for the same reason
``providers/settings.py`` does: a small, self-contained knob set with no prefix.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import TYPE_CHECKING

import httpx
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.models.failure_class import EXTERNAL, FailureClass
from src.services.stages.upstream import TransientUpstreamError

if TYPE_CHECKING:
    from src.services.providers.errors import ByoModality

__all__ = [
    "BUSY_STATUSES",
    "RETRY_AFTER_MAX_SECONDS",
    "RETRY_AFTER_MIN_SECONDS",
    "BusyRetry",
    "GpuAdmissionSettings",
    "GpuBusyError",
    "busy_retry_after",
    "byo_refusal",
    "get_gpu_admission_settings",
]

#: Statuses the admission contract uses for "not now". 502/504 are NOT here: a
#: gateway that could not reach its backend is a different failure, with its own
#: (attempt-bounded) budget at the call sites that tolerate it.
BUSY_STATUSES = frozenset({429, 503})

RETRY_AFTER_MIN_SECONDS = 1.0
RETRY_AFTER_MAX_SECONDS = 120.0
_BACKOFF_BASE_SECONDS = 2.0
_BACKOFF_MAX_SECONDS = 60.0


class GpuAdmissionSettings(BaseSettings):
    """Env (no prefix): ``GPU_BUSY_DEADLINE_SECONDS``, ``ASR_MAX_CONCURRENCY``."""

    #: Total wall time one call may spend waiting out ``gpu_busy`` before it
    #: raises :class:`GpuBusyError`. Covers every busy answer of that call.
    gpu_busy_deadline_seconds: float = Field(default=300.0, gt=0)
    #: Per-process ceiling on in-flight ASR requests (``TranscriptionWorkerClient``).
    asr_max_concurrency: int = Field(default=2, ge=1)

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


@lru_cache(maxsize=1)
def get_gpu_admission_settings() -> GpuAdmissionSettings:
    """Process-wide admission settings (cached; tests ``cache_clear()`` it)."""
    return GpuAdmissionSettings()


class GpuBusyError(TransientUpstreamError):
    """A GPU lane kept answering busy (503/429) past the call's deadline.

    TRANSIENT by class (a retry later can succeed — the lane was full, the request
    was fine) and ``external`` by attribution (a dependency's capacity, not our
    code and not the customer's input).
    """

    failure_class: FailureClass = EXTERNAL

    def __init__(
        self,
        message: str,
        *,
        service: str,
        status_code: int,
        lane: str | None = None,
        waited_seconds: float = 0.0,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.service = service
        self.status_code = status_code
        self.lane = lane
        self.waited_seconds = waited_seconds
        #: The last busy answer's own hint (clamped ``1..120`` s), so a caller that
        #: surfaces the busy to ITS caller can pass ``Retry-After`` on; ``None`` if
        #: the server gave none.
        self.retry_after = retry_after


def _clamp(seconds: float) -> float:
    return min(max(seconds, RETRY_AFTER_MIN_SECONDS), RETRY_AFTER_MAX_SECONDS)


def _as_seconds(raw: object) -> float | None:
    if isinstance(raw, bool):  # a JSON ``true`` is not a duration
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        try:
            value = float(raw.strip())
        except ValueError:  # the HTTP-date form: our services never emit it
            return None
    else:
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _body(response: httpx.Response) -> dict[str, object]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    if not isinstance(payload, dict):
        return {}
    # FastAPI's HTTPException wraps a dict detail: ``{"detail": {...}}``.
    detail = payload.get("detail")
    return detail if isinstance(detail, dict) and "error" in detail else payload


def busy_retry_after(response: httpx.Response) -> float | None:
    """Seconds the server asked us to wait, clamped to ``1..120``; ``None`` if it said nothing.

    The ``Retry-After`` header wins; the body's ``retry_after`` is the fallback.
    """
    header = _as_seconds(response.headers.get("Retry-After"))
    if header is not None:
        return _clamp(header)
    body = _as_seconds(_body(response).get("retry_after"))
    return _clamp(body) if body is not None else None


def _lane(response: httpx.Response) -> str | None:
    lane = _body(response).get("lane")
    return lane if isinstance(lane, str) else None


def _backoff(busy_count: int) -> float:
    base = min(_BACKOFF_BASE_SECONDS * (2 ** (busy_count - 1)), _BACKOFF_MAX_SECONDS)
    return base * random.uniform(0.8, 1.2)  # noqa: S311 — jitter, not crypto


class BusyRetry:
    """One call's busy budget: a monotonic deadline + a count of busy answers seen.

    ``wait(response)`` either sleeps the server-requested delay (never past the
    deadline) or raises :class:`GpuBusyError` when the deadline cannot fit another
    attempt. ``max_attempts`` optionally caps the number of busy answers tolerated
    (an interactive caller that wants a fast failure); the deadline always applies.
    """

    def __init__(
        self,
        service: str,
        *,
        deadline_seconds: float | None = None,
        max_attempts: int | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.service = service
        self._budget = (
            deadline_seconds
            if deadline_seconds is not None
            else get_gpu_admission_settings().gpu_busy_deadline_seconds
        )
        self._clock = clock
        self._started = clock()
        self._max_attempts = max_attempts
        self._sleep = sleep or asyncio.sleep
        self.busy_count = 0
        #: Seconds this budget has slept. Elapsed time is the LARGER of the wall
        #: clock and this, so an injected (instant) sleep still spends the budget.
        self.slept_seconds = 0.0

    def _exhausted(self, response: httpx.Response, waited: float, why: str) -> GpuBusyError:
        lane = _lane(response)
        return GpuBusyError(
            f"{self.service} stayed busy ({response.status_code}"
            f"{f', lane={lane}' if lane else ''}) — {why} after {self.busy_count} busy "
            f"answer(s) and {waited:.0f}s",
            service=self.service,
            status_code=response.status_code,
            lane=lane,
            waited_seconds=waited,
            retry_after=busy_retry_after(response),
        )

    def exhausted(self, response: httpx.Response, why: str) -> GpuBusyError:
        """The :class:`GpuBusyError` for ``response`` when a CALLER's own budget ran out."""
        return self._exhausted(response, self.elapsed_seconds, why)

    @property
    def elapsed_seconds(self) -> float:
        """Budget spent so far: the larger of wall time and slept time."""
        return max(self._clock() - self._started, self.slept_seconds)

    async def wait(self, response: httpx.Response) -> None:
        """Sleep out one busy answer, or raise :class:`GpuBusyError` when out of budget."""
        self.busy_count += 1
        waited = self.elapsed_seconds
        if self._max_attempts is not None and self.busy_count >= self._max_attempts:
            raise self._exhausted(response, waited, "attempt budget exhausted")
        remaining = self._budget - waited
        delay = busy_retry_after(response)
        if delay is None:
            delay = _backoff(self.busy_count)
        if delay > remaining:
            raise self._exhausted(response, waited, "deadline exhausted")
        self.slept_seconds += delay
        await self._sleep(delay)


def byo_refusal(response: httpx.Response, modality: ByoModality) -> Exception | None:
    """modality_gateway's **402** for a ``byo_required`` tenant, as a typed refusal.

    Returns (does not raise) a :class:`SlotNotConfiguredError` — ``customer_input``,
    code ``byo_provider_required`` — or ``None`` for any other status. The gateway's
    own text is kept on the error for logs; the user sees the fixed sentence.
    """
    if response.status_code != 402:
        return None
    # Lazy: ``src.services.providers`` imports the licences client, whose package
    # imports this module — a top-level import here would be circular.
    from src.services.providers.errors import SlotNotConfiguredError

    return SlotNotConfiguredError(modality, upstream_detail=response.text[:500])
