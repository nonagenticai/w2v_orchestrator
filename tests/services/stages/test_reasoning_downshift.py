"""A retry that re-issues an identical call against an identical budget.

Measured on the production path 2026-08-20 (#873/#883): with chain-of-thought ON
the model burned the whole 950s deadline and returned NOTHING — twice, on both
output schemas — while the same call with reasoning OFF answered in 34-67s. So
``stage_call``'s retry, which re-runs the same request with the same budget, was
provably incapable of succeeding. These tests pin the one input that varies.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from src.services.stages.llm_budget import downshift_reasoning, reasoning_for_attempt


def _drive[R](factory: Callable[[], Awaitable[R]]) -> R:
    """Run one attempt of a downshift factory to completion.

    ``asyncio.run`` wants a ``Coroutine``; the factory's declared return is the more
    general ``Awaitable`` that ``stage_call`` accepts, so go through an ``async def``
    rather than widening the production signature to please one test.
    """

    async def _run() -> R:
        return await factory()

    return asyncio.run(_run())


def test_the_first_attempt_is_byte_identical_to_not_downshifting() -> None:
    """The happy path must not silently gain (or lose) reasoning.

    ``None`` — not ``True`` — so WM keeps its own default. Passing ``True`` here
    would turn the helper into a behaviour change on every stage that adopts it.
    """
    seen: list[bool | None] = []

    async def _attempt(enable_reasoning: bool | None) -> str:
        seen.append(enable_reasoning)
        return "ok"

    factory = downshift_reasoning(lambda think: _attempt(think))
    assert _drive(factory) == "ok"
    assert seen == [None]


def test_every_retry_tells_the_model_not_to_think() -> None:
    seen: list[bool | None] = []

    async def _attempt(enable_reasoning: bool | None) -> str:
        seen.append(enable_reasoning)
        return "ok"

    factory = downshift_reasoning(lambda think: _attempt(think))
    for _ in range(4):
        _drive(factory)

    assert seen == [None, False, False, False]


def test_the_retry_that_could_not_succeed_now_can() -> None:
    """The measured failure, end to end through ``stage_call``.

    Attempt 1 reasons and exhausts its budget without answering; attempt 2 is
    downshifted and answers. Before this helper both attempts reasoned, so the
    ladder was two guaranteed timeouts and a dead paid run.
    """
    from w2v_common.stages.execution import stage_call

    calls: list[bool | None] = []

    async def _attempt(enable_reasoning: bool | None) -> str:
        calls.append(enable_reasoning)
        if enable_reasoning is not False:
            await asyncio.sleep(30)  # thinks straight through its deadline
            raise AssertionError("unreachable — the attempt clock fires first")
        return "answered"

    async def _run() -> str:
        return await stage_call(
            downshift_reasoning(lambda think: _attempt(think)),
            timeout_seconds=1.0,
            retries=1,
            backoff_initial=0.001,
            backoff_jitter=0.0,
            stage_name="downshift-probe",
        )

    assert asyncio.run(_run()) == "answered"
    assert calls == [None, False]


def test_a_factory_is_single_use_per_ladder() -> None:
    """Two ladders must each start at ``None``.

    Caching the factory on ``self`` would start the second stage invocation
    already downshifted — a silent, permanent reasoning-off sweep by accident.
    """
    seen: list[bool | None] = []

    async def _attempt(enable_reasoning: bool | None) -> str:
        seen.append(enable_reasoning)
        return "ok"

    for _ in range(2):
        factory = downshift_reasoning(lambda think: _attempt(think))
        _drive(factory)
        _drive(factory)

    assert seen == [None, False, None, False]


@pytest.mark.parametrize("think", [None])
def test_the_flag_reaches_the_wrapped_awaitable_untouched(think: bool | None) -> None:
    """The helper passes the flag through; it does not interpret it."""
    received: list[bool | None] = []

    def _make(enable_reasoning: bool | None) -> Awaitable[None]:
        received.append(enable_reasoning)

        async def _inner() -> None:
            return None

        return _inner()

    factory = downshift_reasoning(_make)
    if think is False:
        _drive(factory)  # burn attempt 1
    _drive(factory)
    assert received[-1] is think


def test_a_call_too_deep_to_take_the_flag_reads_it_from_the_context() -> None:
    """The ``loop_engine`` shape: the closure's signature is not ours to change.

    ``critic_fn(draft)`` is called by ``run_evaluator_optimizer``, so the WM call
    inside it cannot receive the flag as an argument. It reads the ContextVar
    instead, and must see the SAME value the attempt was issued with.
    """
    seen: list[bool | None] = []

    async def _critic() -> str:
        seen.append(reasoning_for_attempt())  # nested arbitrarily deep
        return "ok"

    factory = downshift_reasoning(lambda _think: _critic())
    _drive(factory)
    _drive(factory)

    assert seen == [None, False]


def test_the_flag_never_outlives_its_attempt() -> None:
    """A leaked ContextVar would silently downshift the NEXT stage on the same task."""

    async def _attempt(_think: bool | None) -> None:
        return None

    async def _run() -> bool | None:
        factory = downshift_reasoning(_attempt)
        await factory()
        await factory()  # retry — binds False
        return reasoning_for_attempt()

    assert asyncio.run(_run()) is None


def test_the_context_default_is_none_outside_any_ladder() -> None:
    assert reasoning_for_attempt() is None


@pytest.mark.asyncio
async def test_loop_engine_binds_the_flag_for_its_critic_and_optimizer() -> None:
    """The seam that nine WM call sites depend on, driven for real.

    `claim_finish`, `reconcile_timeline` (x3), `retention_plan` (x2) and
    `script_doctor` (x2) call `reasoning_for_attempt()` inside a closure whose
    signature `run_evaluator_optimizer` owns. Nothing at those call sites shows
    where the value comes from, so this pins it at the source: if `loop_engine`
    stops wrapping, every one of them silently reverts to always-thinking.

    Caught exactly that once already — `footage_critique` read the ContextVar
    under its OWN `stage_call`, which no downshift factory wrapped, so the read
    returned `None` forever.
    """
    from src.services.stages.shared.loop_engine import Critique, run_evaluator_optimizer

    critic_saw: list[bool | None] = []
    passes = {"n": 0}

    async def _critic(draft: str) -> Critique:
        critic_saw.append(reasoning_for_attempt())
        passes["n"] += 1
        if passes["n"] == 1:
            raise ValueError("bad draw")  # forces the retry that must downshift
        return Critique(score=9.9)  # >= threshold, so the loop stops on pass 1

    async def _optimizer(draft: str, critique: Critique) -> str:  # pragma: no cover
        raise AssertionError("threshold met on pass 1; optimizer must not run")

    await run_evaluator_optimizer(
        "draft",
        critic_fn=_critic,
        optimizer_fn=_optimizer,
        timeout_seconds=5,
        retries=1,
        retry_on=(ValueError,),
        stage_name="downshift-loop-probe",
    )

    assert critic_saw == [None, False], (
        f"loop_engine did not downshift its critic retry: {critic_saw}"
    )
