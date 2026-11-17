from __future__ import annotations

import asyncio

import pytest

from src.models.failure_class import attribute_exception
from src.services.middleware.timeout import MiddlewareTimeoutError, TimeoutMiddleware
from src.services.stages.phase_tracker import phase


async def _slow_call() -> None:
    await asyncio.sleep(10)


async def _next(call):  # type: ignore[no-untyped-def]
    return await call()


async def test_timeout_raises_messaged_error_with_ceiling_and_phase() -> None:
    middleware = TimeoutMiddleware(seconds=0.01)

    async def _run() -> None:
        async with phase("ken_burns"):
            await middleware(_slow_call, _next)

    with pytest.raises(MiddlewareTimeoutError) as exc_info:
        await _run()

    message = str(exc_info.value)
    assert message
    assert "0.01" in message
    assert "ken_burns" in message


async def test_timeout_renders_unknown_phase_when_none() -> None:
    middleware = TimeoutMiddleware(seconds=0.01)

    with pytest.raises(MiddlewareTimeoutError) as exc_info:
        await middleware(_slow_call, _next)

    message = str(exc_info.value)
    assert message
    assert "unknown phase" in message


async def test_timeout_error_is_instance_of_both_timeout_types() -> None:
    from w2v_common.stages import StageTimeoutError

    middleware = TimeoutMiddleware(seconds=0.01)

    with pytest.raises(MiddlewareTimeoutError) as exc_info:
        await middleware(_slow_call, _next)

    err = exc_info.value
    assert isinstance(err, TimeoutError)
    assert isinstance(err, StageTimeoutError)
    assert isinstance(err.__cause__, TimeoutError)


async def test_timeout_error_attributes_as_external_not_unknown() -> None:
    middleware = TimeoutMiddleware(seconds=0.01)

    with pytest.raises(MiddlewareTimeoutError) as exc_info:
        await middleware(_slow_call, _next)

    assert attribute_exception(exc_info.value) == "external"


@pytest.mark.asyncio
async def test_inner_timeout_error_propagates_untouched() -> None:
    """An inner bare TimeoutError must NOT be rewritten as a middleware timeout.

    ``BaseServiceClient.poll_job`` raises a bare ``TimeoutError`` carrying a
    precise message and its OWN (much smaller) budget. A blanket
    ``except TimeoutError`` in the middleware would replace that with this
    middleware's outer ceiling -- losing the real message and misreporting the
    number. ``compile_stage`` explicitly depends on the inner error surfacing.
    """
    inner = TimeoutError("video_compiler job abc123 timed out after 1200s")

    async def raises_inner() -> None:
        raise inner

    mw = TimeoutMiddleware(seconds=60)

    with pytest.raises(TimeoutError) as caught:
        await mw(lambda: raises_inner(), lambda c: c())

    assert caught.value is inner, "inner TimeoutError was replaced by the middleware"
    assert not isinstance(caught.value, MiddlewareTimeoutError)
    assert "job abc123" in str(caught.value)
    assert "1200s" in str(caught.value)
