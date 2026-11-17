import asyncio

import pytest

from src.services.middleware.chain import build_chain
from src.services.middleware.timeout import TimeoutMiddleware


@pytest.mark.asyncio
async def test_chain_does_not_retry_a_failing_call() -> None:
    """The stage chain must invoke the call exactly once on failure.

    Retry was removed from the chain (it stacked a cascade on top of the
    handlers' own call-boundary retries and re-ran non-idempotent renders).
    This pins that the chain is a pass-through on error: one attempt, propagate.
    """
    state = {"calls": 0}

    async def always_fail():
        state["calls"] += 1
        raise RuntimeError("boom")

    chain = build_chain([TimeoutMiddleware(seconds=5)])
    with pytest.raises(RuntimeError):
        await chain(always_fail)
    assert state["calls"] == 1


@pytest.mark.asyncio
async def test_timeout_fires() -> None:
    async def slow():
        await asyncio.sleep(2)
        return "never"

    chain = build_chain([TimeoutMiddleware(seconds=0.1)])
    with pytest.raises(asyncio.TimeoutError):
        await chain(slow)


@pytest.mark.asyncio
async def test_metrics_middleware_outcome_bound_on_cancellation() -> None:
    """Regression for UnboundLocalError on asyncio.CancelledError path.

    Live website_review post-promote run c7dbe9f4-d3a1-41ab-8280-7097828a0669
    crashed in the metrics middleware's `finally` because `outcome` was only
    bound inside try/except Exception — and CancelledError is a BaseException
    that skips both branches. The fix initialises `outcome = "fail"` before
    the try so `finally` always has a bound value. This regression test
    raises CancelledError mid-chain and asserts the middleware does NOT
    surface UnboundLocalError as a "secondary" exception masking the
    original cause.
    """
    from src.services.middleware.metrics import MetricsMiddleware

    async def cancels():
        raise asyncio.CancelledError("user-cancelled mid-stage")

    chain = build_chain([MetricsMiddleware(stage_name="test_stage")])
    with pytest.raises(asyncio.CancelledError):
        await chain(cancels)
