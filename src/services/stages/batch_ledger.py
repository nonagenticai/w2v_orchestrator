"""A retry must not re-pay the batches that already answered.

Several LLM stages split their work into bounded WM calls and run the whole loop
inside ONE ``stage_call`` slice::

    async def _attempt() -> Out:
        parts = []
        for batch in batched(items, N):
            parts.extend(await wm.message(...))   # k calls
        return assemble(parts)

    await stage_call(lambda: guarded(_attempt), timeout_seconds=950, retries=1)

The accumulator is local to ``_attempt``, so when attempt 1 times out having
finished k of N batches, attempt 2 starts again at batch 0. It re-pays k WM calls
it already has answers for, reaches roughly the same place, and times out again.
**A stage in that shape cannot complete unless ALL N batches fit inside a SINGLE
attempt** — the retry adds cost and no coverage. ``voice_director`` died exactly
this way in production on 2026-08-20, and #885 found and fixed the same shape by
hand in ``shared/footage_critique.py``.

The ledger outlives the attempt. It is created OUTSIDE ``_attempt`` and remembers
each batch's answer by index, so a retry only issues the batches that have not
answered yet. Progress accumulates across the whole ladder instead of resetting.

This is only sound because the batches are INDEPENDENT — each returns its own
slice of the answer and the parts are merged flat. Every current caller documents
that property at its own call site. Do NOT use this where batch i+1's prompt
depends on batch i's reply: the ledger would silently pin a stale prefix.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

# How many batches of ONE ledger may be in flight at once by default.
#
# The batches a ledger holds are INDEPENDENT by construction (see the module
# docstring), so nothing about the work forces them to be serial — and until
# 2026-08-23 :meth:`run` issued them one at a time anyway, i.e. concurrency
# exactly 1. Measured on the pgx vLLM that serves every w2v stage:
# ``--max-num-seqs=32``, KV cache at 1.1%, ZERO preemptions, mean queue time
# 0.70ms over 11,023 requests, ``Waiting: 0`` in every engine log line and
# ``Running`` never above 3. Throughput scaled LINEARLY with concurrency —
# 1 request 14.0 tok/s, 2 requests 27.5 tok/s, per-stream rate unchanged — so
# the second concurrent call was essentially free and ~30x of the server was
# sitting idle waiting on us. A stage of N items at batch size 5 was paying
# ceil(N/5) SEQUENTIAL decodes for no reason.
#
# 5 matches the ceiling the two hand-rolled fan-outs already converged on
# (``segment_select._SELECT_BATCH_CONCURRENCY``,
# ``footage_critique._CRITIC_BATCH_CONCURRENCY``) — read the derivation there.
# It is a per-ledger cap, and several stages can be in flight at once, so it is
# deliberately well under ``max-num-seqs``. Override per call when a stage has
# measured something better; do not raise this constant without re-measuring.
_DEFAULT_BATCH_CONCURRENCY = 5


class BatchLedger[T, R]:
    """Per-ladder record of which batches already answered.

    Create one per stage invocation, OUTSIDE the attempt factory. Creating it
    inside is the bug this exists to fix; caching it on ``self`` is worse — a
    second run of the same stage would inherit the first run's answers.
    """

    def __init__(self) -> None:
        self._answered: dict[int, R] = {}

    @property
    def answered(self) -> int:
        """How many batches are already paid for. For logging and for tests."""
        return len(self._answered)

    def reset(self) -> None:
        """Forget every banked answer so a retry RE-ISSUES the whole draw.

        The ledger banks any batch that returned without raising, and an EMPTY
        answer is a return, not a raise — legitimately so, because a batch that
        holds nothing worth picking is normal when its siblings carry the
        picks. But that makes an all-empty draw indistinguishable from a
        complete one to :meth:`run_parallel`: on the retry the early-out
        (``if not pending``) issues ZERO calls, the caller re-validates the same
        banked nothing, and raises the identical error. The re-prompt the
        caller's retry budget exists to buy can never happen.

        So a caller that has just discovered its banked answers are
        COLLECTIVELY unusable calls this immediately before raising into its
        retry ladder. The next attempt then re-issues every batch, which is
        exactly what a re-prompt means. Do NOT call it on a partial failure —
        resuming the batches that did answer is the ledger's whole purpose.
        """
        self._answered.clear()

    async def run(
        self,
        batches: Sequence[T],
        call: Callable[[T], Awaitable[R]],
        *,
        concurrency: int = _DEFAULT_BATCH_CONCURRENCY,
    ) -> list[R]:
        """Issue only the batches that have not answered, up to ``concurrency`` at once.

        Returns every batch's result in BATCH order regardless of which call
        finished first — results are keyed by index, never appended. Raises
        whatever ``call`` raises, having kept the results collected so far — that
        is the point: the raise propagates to ``stage_call``, which retries, and
        the retry resumes.

        This is :meth:`run_parallel` with the index dropped, and it inherits that
        method's whole contract: the semaphore, the ``finally`` harvest that
        survives cancellation, and re-raising the first failure IN BATCH ORDER.
        Use ``run_parallel`` directly when ``call`` needs to know its position.

        ⚠️ Two consequences of no longer being serial, both deliberate:

        * A failing draw now issues up to ``concurrency - 1`` batches that a
          serial loop would have skipped, because they were already in flight
          when the failure landed. Their answers are BANKED, so the retry does
          not re-buy them; the cost is bounded and the win is ceil(N/5)
          sequential decodes collapsing into ceil(N/5/concurrency) waves.
        * ``call`` must not depend on being invoked in order or on observing a
          previous batch's effects. That was already the ledger's stated
          precondition (batches are INDEPENDENT — see the module docstring); it
          is now load-bearing rather than merely documented.

        ``batches`` must be a materialised sequence, not the iterator ``batched()``
        returns: it is walked once per attempt, and an exhausted iterator on
        attempt 2 would look like "everything is done" and silently return a
        truncated answer.
        """

        async def _indexed(_index: int, batch: T) -> R:
            return await call(batch)

        return await self.run_parallel(batches, _indexed, concurrency=concurrency)

    async def run_parallel(
        self,
        batches: Sequence[T],
        call: Callable[[int, T], Awaitable[R]],
        *,
        concurrency: int,
    ) -> list[R]:
        """Same contract as :meth:`run`, for callers that fan their batches out.

        ``call`` receives ``(index, batch)`` with a ZERO-based index, because a
        fanned-out caller usually labels each prompt with its position and cannot
        recover that from the batch alone.

        ⚠️ **The harvest is in a ``finally``, and that is the whole point.** The
        first version of this method did::

            settled = await asyncio.gather(*coros, return_exceptions=True)
            for ... in zip(pending, settled): self._answered[index] = result

        which recovers in-band exceptions but loses EVERYTHING on cancellation —
        and cancellation is precisely how a stage times out. ``stage_call`` uses
        ``asyncio.wait_for``, which CANCELS the attempt; the ``await gather``
        then raises ``CancelledError``, ``settled`` is never bound, and the loop
        never runs. Measured: a gather of one instant task and one slow task,
        cancelled by ``wait_for``, records **zero** results. So the ledger gave
        no resume benefit at all for the timeout case it exists to fix.

        Harvesting from the task objects in a ``finally`` fixes it: a child that
        had already COMPLETED keeps its result even though ``gather`` cancels its
        children on the way out, so its answer is banked and the retry does not
        re-buy it. A child still in flight is genuinely lost — its call was
        already issued and nothing can recover that.

        The first exception IN BATCH ORDER is re-raised, unchanged, so the retry
        classification upstream sees the same exception a serial loop would have
        raised.
        """
        pending = [
            (index, batch) for index, batch in enumerate(batches) if index not in self._answered
        ]
        if not pending:
            return [self._answered[index] for index in range(len(batches))]

        limit = asyncio.Semaphore(concurrency)

        async def _one(index: int, batch: T) -> R:
            async with limit:
                return await call(index, batch)

        tasks: dict[int, asyncio.Task[R]] = {
            index: asyncio.ensure_future(_one(index, batch)) for index, batch in pending
        }
        try:
            await asyncio.gather(*tasks.values(), return_exceptions=True)
        finally:
            # Runs on the cancellation path too. `task.result()` is only touched
            # for a task that finished cleanly, so neither a cancelled nor a
            # failed child raises here.
            for index, task in tasks.items():
                if task.done() and not task.cancelled() and task.exception() is None:
                    self._answered[index] = task.result()

        first_error: BaseException | None = None
        for index, task in tasks.items():
            if index in self._answered:
                continue
            exc = task.exception() if task.done() and not task.cancelled() else None
            if exc is not None and first_error is None:
                first_error = exc
        if first_error is not None:
            raise first_error
        return [self._answered[index] for index in range(len(batches))]
