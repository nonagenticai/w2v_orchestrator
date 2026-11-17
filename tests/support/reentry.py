"""Re-entry harness: prove a stage does NOT repeat paid calls after a pod death.

A stage that dies half-way is re-entered FROM THE TOP by the reconciler. With its
expensive sub-steps wrapped in ``ctx.step_ledger(name).once(...)``, the second entry
must reuse every sub-step the first one finished. This module is the shared way to
assert that, so every wrapped call site is tested the same way.

Typical use::

    from tests.support.reentry import Outbound, run_with_reentry

    async def test_narrate_does_not_repay_finished_scenes() -> None:
        out = Outbound()
        tts = AsyncMock()
        tts.synthesize = out.fake("tts.synthesize", lambda text, **_: {"url": f"s3://{text}"})
        stage = NarrateStage(client=tts)

        result = await run_with_reentry(
            lambda ctx: stage.run({}, ctx),
            ctx=_ctx_with_three_scenes(),
            outbound=out,
            die_after=2,            # 2 outbound calls finish, the 3rd kills the pod
        )

        result.assert_calls({"tts.synthesize": 3})   # 3 scenes ⇒ 3 calls, NOT 2 + 3
        assert result.first_attempt_calls == {"tts.synthesize": 2}
        assert len(result.output.clips) == 3

What it simulates
-----------------
* **Death**: the ``die_after + 1``-th outbound call (counted across ALL fakes of the
  :class:`Outbound`) raises :class:`SimulatedPodDeath` INSTEAD of running, and is
  not counted. It models "the pod went away before this call returned": the sub-step
  never finished, so nothing may be in the ledger for it. ``SimulatedPodDeath`` is
  an ordinary ``Exception`` subclass on purpose — a real pod death runs no handler
  code at all, and a stage that catches ``Exception`` and carries on would mask the
  simulation, which :func:`run_with_reentry` reports as a failure rather than a pass.
* **Durability**: both attempts share one :class:`InMemoryStepStore`, the stand-in
  for ``stage_step_results``. Each attempt gets a FRESH ``StepLedger`` over it, as
  in production where nothing but the table survives the pod.

What it does NOT cover: the real table, RLS and the upsert — see
``tests/database/repositories/test_stage_step_repository.py``.
"""

from __future__ import annotations

import asyncio
import inspect
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import cast
from uuid import UUID

from src.services.runs.run_context import RunContext
from src.services.stages.shared.step_ledger import JsonValue, StepLedger, StoredStep


class SimulatedPodDeath(Exception):  # noqa: N818 — named for what it simulates
    """Raised by an :class:`Outbound` fake in place of the call that "killed the pod"."""


class InMemoryStepStore:
    """``StepStore`` backed by a dict — the test double for ``stage_step_results``.

    ``rows`` is keyed ``(run_id, stage_name, step_key)`` exactly like the table's
    unique constraint, so a second ``put`` REPLACES. Set ``fail_get`` / ``fail_put``
    to an exception to make that operation raise (a ledger DB outage).
    """

    def __init__(self) -> None:
        self.rows: dict[tuple[UUID, str, str], StoredStep] = {}
        self.gets: list[tuple[UUID, str, str]] = []
        self.puts: list[tuple[UUID, str, str]] = []
        self.fail_get: Exception | None = None
        self.fail_put: Exception | None = None

    async def get(self, run_id: UUID, stage_name: str, step_key: str) -> StoredStep | None:
        self.gets.append((run_id, stage_name, step_key))
        if self.fail_get is not None:
            raise self.fail_get
        return self.rows.get((run_id, stage_name, step_key))

    async def put(
        self, run_id: UUID, stage_name: str, step_key: str, input_hash: str, result: JsonValue
    ) -> None:
        self.puts.append((run_id, stage_name, step_key))
        if self.fail_put is not None:
            raise self.fail_put
        self.rows[(run_id, stage_name, step_key)] = StoredStep(input_hash=input_hash, result=result)

    def step_keys(self, stage_name: str | None = None) -> set[str]:
        """Step keys currently stored (optionally for one stage)."""
        return {key for (_, stage, key) in self.rows if stage_name in (None, stage)}


class Outbound:
    """A set of counting fakes for a stage's outbound (paid) calls, with a kill switch.

    ``fake(name, result)`` returns an async callable to drop in wherever the stage
    awaits a client method. ``result`` is either a plain value or a callable (sync or
    async) receiving the call's arguments. Every completed call is counted under
    ``name`` in :attr:`calls`; all fakes share one death budget (see :meth:`arm`).
    """

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self._budget: int | None = None

    def arm(self, die_after: int) -> None:
        """Let ``die_after`` more outbound calls finish; the next one raises."""
        if die_after < 0:
            raise ValueError("die_after must be >= 0")
        self._budget = die_after

    def disarm(self) -> None:
        self._budget = None

    @property
    def total(self) -> int:
        return sum(self.calls.values())

    def fake(self, name: str, result: object = None) -> Callable[..., Awaitable[object]]:
        async def _call(*args: object, **kwargs: object) -> object:
            if self._budget is not None:
                if self._budget == 0:
                    raise SimulatedPodDeath(f"simulated pod death at outbound call {name!r}")
                self._budget -= 1
            self.calls[name] += 1
            if not callable(result):
                return result
            value = cast("Callable[..., object]", result)(*args, **kwargs)
            return await value if inspect.isawaitable(value) else value

        _call.__name__ = f"fake_{name}"
        return _call


def ctx_with_ledger(ctx: RunContext, store: InMemoryStepStore) -> RunContext:
    """``ctx`` with ``step_ledger(...)`` backed by ``store`` (a fresh ledger per call)."""
    return replace(
        ctx,
        step_ledger_factory=lambda stage_name: StepLedger(store, ctx.run_id, stage_name),
    )


@dataclass(frozen=True)
class ReentryResult[T]:
    """Outcome of :func:`run_with_reentry`."""

    #: What the SECOND (surviving) attempt returned.
    output: T
    #: Per-fake completed calls when the first attempt died.
    first_attempt_calls: dict[str, int]
    #: Per-fake completed calls over BOTH attempts — what the tenant paid for.
    total_calls: dict[str, int]
    #: The shared store, for asserting which steps were recorded.
    store: InMemoryStepStore

    @property
    def second_attempt_calls(self) -> dict[str, int]:
        """Per-fake calls made by the re-entry alone (zero entries omitted)."""
        delta = Counter(self.total_calls)
        delta.subtract(self.first_attempt_calls)
        return {name: n for name, n in delta.items() if n}

    def assert_calls(self, expected: dict[str, int]) -> None:
        """Assert the TOTAL per-fake call counts across both attempts.

        For a fully-wrapped stage this equals what ONE uninterrupted run makes —
        that equality is the whole point of the ledger.
        """
        assert self.total_calls == expected, (
            f"outbound calls across death + re-entry were {self.total_calls}, expected "
            f"{expected} (first attempt alone made {self.first_attempt_calls}) — a count "
            "above the single-run figure means the re-entry REPEATED a finished, paid call"
        )


async def run_with_reentry[T](
    attempt: Callable[[RunContext], Awaitable[T]],
    *,
    ctx: RunContext,
    outbound: Outbound,
    die_after: int,
    store: InMemoryStepStore | None = None,
) -> ReentryResult[T]:
    """Run ``attempt`` until a simulated pod death, then re-enter it to completion.

    ``attempt`` receives a ``RunContext`` whose ``step_ledger`` is backed by one
    shared in-memory store, and is typically ``lambda c: stage.run({}, c)``. Pass
    ``store`` to pre-seed it or to share it across several stages.

    Fails (``AssertionError``) if the first attempt does NOT die — ``die_after`` is
    then at or beyond the stage's total outbound calls, or the stage swallowed
    :class:`SimulatedPodDeath`; either way the scenario under test never happened.
    """
    store = store if store is not None else InMemoryStepStore()
    ledger_ctx = ctx_with_ledger(ctx, store)

    outbound.arm(die_after)
    died = False
    try:
        await attempt(ledger_ctx)
    except SimulatedPodDeath:
        died = True
    finally:
        outbound.disarm()
    assert died, (
        f"the first attempt finished without dying (die_after={die_after}, outbound calls "
        f"made={dict(outbound.calls)}): lower die_after, or the stage swallowed "
        "SimulatedPodDeath — a re-entry was never exercised"
    )
    first = dict(outbound.calls)
    # Let anything the dead attempt scheduled settle, as a fresh pod would never see it.
    await asyncio.sleep(0)

    output = await attempt(ledger_ctx)
    return ReentryResult(
        output=output,
        first_attempt_calls=first,
        total_calls=dict(outbound.calls),
        store=store,
    )
