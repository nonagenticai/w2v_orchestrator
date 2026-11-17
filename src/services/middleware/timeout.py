from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from w2v_common.stages import StageTimeoutError

from src.services.middleware.base import Middleware, NextFn
from src.services.stages.phase_tracker import current_phase

T = TypeVar("T")


class MiddlewareTimeoutError(StageTimeoutError, TimeoutError):
    """A ``TimeoutMiddleware`` ceiling was exceeded — messaged and attributable.

    ``asyncio.wait_for`` raises a bare, zero-argument ``TimeoutError`` on
    expiry: no ceiling, no phase, no message. Formatted downstream
    (``dag_executor.py``'s ``f"{type(exc).__name__}: {exc}"``) that becomes the
    literal string ``"TimeoutError: "`` — unreadable, and unattributable
    because a bare ``TimeoutError`` matches neither ``_OUR_DEFECT_TYPES`` nor
    any stamped ``failure_class`` in ``attribute_exception``'s chain walk.

    This subclasses BOTH ``StageTimeoutError`` (so
    ``dag_executor._RESUMABLE_TIMEOUT_EXC`` still catches it — resumability
    behaviour is unchanged) and the builtin ``TimeoutError`` (so anything
    that catches the builtin, including ``asyncio.wait_for`` callers
    upstream, still catches this). ``failure_class = "external"`` is stamped
    directly on the instance: a dependency/stage not answering within its
    budget is an external fault, not ours to fix by shipping code, and
    ``attribute_exception``'s explicit-attribute pass reads it directly
    rather than falling through to ``unknown``.
    """

    failure_class = "external"

    def __init__(self, seconds: float, phase: str | None) -> None:
        phase_desc = phase if phase is not None else "unknown phase"
        message = f"stage timed out after {seconds}s in phase {phase_desc!r}"
        super().__init__(message)
        self.seconds = seconds
        self.phase = phase


class TimeoutMiddleware(Middleware):
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    async def __call__(self, call: Callable[[], Awaitable[T]], next_: NextFn[T]) -> T:
        # ``asyncio.timeout`` rather than ``wait_for`` SPECIFICALLY so we can ask
        # ``cm.expired()`` whether OUR ceiling is what fired.
        #
        # A bare ``except TimeoutError`` here would be a regression, not a fix:
        # inner code raises bare ``TimeoutError`` too — ``BaseServiceClient.poll_job``
        # (w2v_common/clients/base.py:968,1016) raises
        # ``f"{service} job {id} timed out after {timeout}s"`` — and blindly
        # converting that would REPLACE a precise, correctly-scoped message with
        # this middleware's much larger outer ceiling. ``compile_stage`` relies on
        # the opposite ("stage_call's own asyncio timeout is the backstop, a bit
        # higher so poll_job's clearer TimeoutError surfaces first"). So an inner
        # timeout propagates UNTOUCHED; only our own expiry is rewritten.
        cm = asyncio.timeout(self.seconds)
        try:
            async with cm:
                return await next_(call)
        except TimeoutError as exc:
            if cm.expired():
                raise MiddlewareTimeoutError(self.seconds, current_phase()) from exc
            raise
