from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

# Stage metric definitions live in the single observability home; this
# middleware only observes them. (Names unchanged — same series as before.)
from src.observability.metrics import stage_duration_seconds, stage_total
from src.services.middleware.base import Middleware, NextFn

T = TypeVar("T")


class MetricsMiddleware(Middleware):
    def __init__(self, stage_name: str) -> None:
        self.stage_name = stage_name

    async def __call__(self, call: Callable[[], Awaitable[T]], next_: NextFn[T]) -> T:
        t0 = time.monotonic()
        # Initialise outside the try/except so the `finally` always has a
        # bound value, including the asyncio.CancelledError / BaseException
        # path (those skip the `except Exception` branch but still run
        # `finally`). The previous code surfaced UnboundLocalError ("local
        # variable 'outcome' where it is not associated with a value") which
        # masked the real cancellation cause; observed live on website_review
        # post-promote run c7dbe9f4-d3a1-41ab-8280-7097828a0669.
        outcome = "fail"
        try:
            res = await next_(call)
            outcome = "ok"
            return res
        finally:
            dt = time.monotonic() - t0
            stage_duration_seconds.labels(stage=self.stage_name, outcome=outcome).observe(dt)
            stage_total.labels(stage=self.stage_name, outcome=outcome).inc()
