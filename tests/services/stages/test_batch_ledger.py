"""A retry that re-pays answered batches cannot finish work that needs two attempts."""

from __future__ import annotations

import asyncio

import pytest

from src.services.stages.batch_ledger import BatchLedger


@pytest.mark.asyncio
async def test_a_retry_only_issues_what_did_not_answer() -> None:
    issued: list[str] = []
    fail_on = {"c"}

    async def _call(batch: str) -> str:
        issued.append(batch)
        if batch in fail_on:
            raise ValueError("bad draw")
        return batch.upper()

    ledger: BatchLedger[str, str] = BatchLedger()
    batches = ["a", "b", "c", "d"]

    with pytest.raises(ValueError, match="bad draw"):
        await ledger.run(batches, _call)
    # `run` fans out, so 'd' was already in flight when 'c' raised — a serial
    # loop would have stopped at 'c'. That is the bounded, deliberate cost of
    # concurrency: 'd' ANSWERED, so it is banked and the retry does not re-buy it.
    assert sorted(issued) == ["a", "b", "c", "d"]
    assert ledger.answered == 3

    fail_on.clear()  # the retry draws better
    assert await ledger.run(batches, _call) == ["A", "B", "C", "D"]

    # 'a', 'b' and 'd' were paid for once, not twice; only the batch that never
    # answered is re-issued. Without the ledger this reads
    # ['a','b','c','d', 'a','b','c','d'] — 3 wasted WM calls on a 4-batch stage.
    assert issued[4:] == ["c"]


@pytest.mark.asyncio
async def test_the_answer_is_in_batch_order_not_completion_order() -> None:
    async def _call(batch: int) -> int:
        await asyncio.sleep(0)
        return batch * 10

    ledger: BatchLedger[int, int] = BatchLedger()
    assert await ledger.run([0, 1, 2, 3], _call) == [0, 10, 20, 30]


@pytest.mark.asyncio
async def test_a_completed_ledger_issues_nothing() -> None:
    issued = 0

    async def _call(batch: int) -> int:
        nonlocal issued
        issued += 1
        return batch

    ledger: BatchLedger[int, int] = BatchLedger()
    await ledger.run([1, 2], _call)
    await ledger.run([1, 2], _call)
    assert issued == 2


@pytest.mark.asyncio
async def test_an_empty_batch_list_is_an_empty_answer() -> None:
    async def _call(batch: int) -> int:  # pragma: no cover -- must never be called
        raise AssertionError("called with no batches")

    ledger: BatchLedger[int, int] = BatchLedger()
    assert await ledger.run([], _call) == []


@pytest.mark.asyncio
async def test_run_parallel_keeps_the_siblings_that_answered_before_it_raises() -> None:
    """The load-bearing difference from a plain `gather`.

    `asyncio.gather(...)` without `return_exceptions` raises on the first
    exception and throws away the siblings that succeeded — which is exactly the
    cost this class exists to stop paying twice. Those siblings were already in
    flight and already billed; discarding their answers means the retry buys them
    again.
    """
    ledger: BatchLedger[str, str] = BatchLedger()
    attempts: list[str] = []

    async def _call(index: int, batch: str) -> str:
        attempts.append(batch)
        if batch == "b" and attempts.count("b") == 1:
            raise ValueError("batch b failed on the first attempt")
        return f"{batch}!"

    with pytest.raises(ValueError, match="batch b failed"):
        await ledger.run_parallel(["a", "b", "c"], _call, concurrency=3)

    assert ledger.answered == 2, "a and c answered and must be retained across the raise"

    # The retry: only the failed batch is re-issued.
    result = await ledger.run_parallel(["a", "b", "c"], _call, concurrency=3)

    assert result == ["a!", "b!", "c!"]
    assert attempts == ["a", "b", "c", "b"], (
        f"expected only batch b to be re-issued, got {attempts}"
    )


@pytest.mark.asyncio
async def test_run_parallel_reraises_the_first_failure_in_batch_order() -> None:
    """Not the first to FINISH — the retry classification upstream must see the
    same exception a serial loop would have raised."""
    ledger: BatchLedger[str, str] = BatchLedger()

    async def _call(index: int, batch: str) -> str:
        if batch == "b":
            await asyncio.sleep(0.02)  # loses the race, still wins the report
            raise ValueError("b")
        if batch == "c":
            raise RuntimeError("c")
        return batch

    with pytest.raises(ValueError, match=r"^b$"):
        await ledger.run_parallel(["a", "b", "c"], _call, concurrency=3)


@pytest.mark.asyncio
async def test_run_parallel_bounds_concurrency() -> None:
    ledger: BatchLedger[int, int] = BatchLedger()
    in_flight = 0
    peak = 0

    async def _call(index: int, batch: int) -> int:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return batch

    await ledger.run_parallel(list(range(10)), _call, concurrency=3)

    assert peak <= 3, f"concurrency ceiling breached: {peak} in flight"


@pytest.mark.asyncio
async def test_run_parallel_banks_finished_batches_when_the_attempt_is_cancelled() -> None:
    """The case the ledger exists for, and the one the first version got wrong.

    A stage times out via `stage_call`'s `asyncio.wait_for`, which CANCELS the
    attempt. The original implementation did
    `settled = await asyncio.gather(..., return_exceptions=True)` and then looped
    over `settled` — but an outer cancel propagates out of that `await`, so
    `settled` was never bound and the loop never ran. Every batch that had
    already answered was discarded, and the ledger delivered no resume benefit
    for a TIMEOUT at all.

    Measured before the fix: a gather of one instant and one slow task, cancelled
    by `wait_for`, recorded ZERO results.
    """
    ledger: BatchLedger[str, str] = BatchLedger()
    issued: list[str] = []

    async def _call(index: int, batch: str) -> str:
        issued.append(batch)
        if batch == "slow":
            await asyncio.sleep(5)
        return f"{batch}!"

    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await asyncio.wait_for(
            ledger.run_parallel(["fast", "slow"], _call, concurrency=2), timeout=0.2
        )

    assert ledger.answered == 1, (
        "the batch that ALREADY answered must be banked despite the cancellation — "
        "otherwise the retry re-buys it and the ledger is pointless for timeouts"
    )

    # The retry: only the batch that never answered is re-issued.
    result = await ledger.run_parallel(["fast", "slow"], _call, concurrency=2)

    assert result == ["fast!", "slow!"]
    assert issued == ["fast", "slow", "slow"], f"expected 'fast' to be bought once, got {issued}"


@pytest.mark.asyncio
async def test_run_issues_batches_concurrently() -> None:
    """`run` was serial until 2026-08-23 — concurrency exactly 1.

    The vLLM that serves every w2v stage is configured `--max-num-seqs=32` and
    was measured idle-waiting on us: zero preemptions, 0.70ms mean queue time,
    `Waiting: 0`, throughput scaling LINEARLY with concurrency. A stage of N
    items at batch size 5 was paying ceil(N/5) SEQUENTIAL GPU decodes.

    A serial loop cannot get past 1 in flight, so `peak == 1` fails this test.
    """
    ledger: BatchLedger[int, int] = BatchLedger()
    in_flight = 0
    peak = 0

    async def _call(batch: int) -> int:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return batch

    assert await ledger.run(list(range(10)), _call) == list(range(10))

    assert peak > 1, "run is still serial — the whole point of the change"
    assert peak <= 5, f"default concurrency ceiling breached: {peak} in flight"


@pytest.mark.asyncio
async def test_run_honours_a_per_call_concurrency_override() -> None:
    ledger: BatchLedger[int, int] = BatchLedger()
    in_flight = 0
    peak = 0

    async def _call(batch: int) -> int:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return batch

    await ledger.run(list(range(9)), _call, concurrency=2)

    assert peak == 2, f"expected the override to bound in-flight at 2, saw {peak}"


@pytest.mark.asyncio
async def test_run_does_not_reissue_an_answered_batch_on_retry() -> None:
    """The resume contract, under concurrency.

    Concurrency must not cost the ledger its reason to exist: a batch that
    answered on attempt 1 must never be bought again on attempt 2.
    """
    ledger: BatchLedger[str, str] = BatchLedger()
    issued: list[str] = []
    fail_on = {"c"}

    async def _call(batch: str) -> str:
        issued.append(batch)
        await asyncio.sleep(0)
        if batch in fail_on:
            raise ValueError("bad draw")
        return batch.upper()

    batches = ["a", "b", "c", "d", "e"]

    with pytest.raises(ValueError, match="bad draw"):
        await ledger.run(batches, _call)
    assert ledger.answered == 4, "every batch but 'c' answered and must be banked"

    fail_on.clear()
    assert await ledger.run(batches, _call) == ["A", "B", "C", "D", "E"]
    assert issued[5:] == ["c"], f"only 'c' may be re-issued, got {issued[5:]}"

    # A third call with a complete ledger issues nothing at all.
    assert await ledger.run(batches, _call) == ["A", "B", "C", "D", "E"]
    assert len(issued) == 6


@pytest.mark.asyncio
async def test_run_banks_finished_batches_when_the_attempt_is_cancelled() -> None:
    """Partial progress must survive a CANCELLATION, not just an in-band raise.

    `stage_call` times out via `asyncio.wait_for`, which CANCELS the attempt.
    `asyncio.gather(..., return_exceptions=True)` records NOTHING on that path —
    the `await` raises `CancelledError` and the collection loop never runs — so a
    ledger built on plain gather is INERT for exactly the timeout it exists to
    fix. `run` therefore harvests from the task objects in a `finally`.
    """
    ledger: BatchLedger[str, str] = BatchLedger()
    issued: list[str] = []

    async def _call(batch: str) -> str:
        issued.append(batch)
        if batch.startswith("slow"):
            await asyncio.sleep(5)
        return f"{batch}!"

    batches = ["fast1", "fast2", "slow1", "slow2"]

    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await asyncio.wait_for(ledger.run(batches, _call), timeout=0.2)

    assert ledger.answered == 2, (
        "both fast batches had already answered and must be banked despite the "
        f"cancellation; ledger holds {ledger.answered}"
    )

    # The retry resumes: only the two that never answered are re-issued.
    async def _call_fast(batch: str) -> str:
        issued.append(batch)
        return f"{batch}!"

    assert await ledger.run(batches, _call_fast) == [
        "fast1!",
        "fast2!",
        "slow1!",
        "slow2!",
    ]
    assert issued[4:] == ["slow1", "slow2"], f"fast batches were re-bought: {issued}"
