"""Script stages reuse the model replies they already paid for on re-entry.

``LLMScriptStageBase`` has ONE ``WMClient.message`` call site, reached once per
draw: the first draw, each ``stage_call`` retry of it, and every corrective re-ask
a blueprint makes by calling ``super().handle`` again. A pod death between two
draws used to re-send (and re-bill) the finished ones and mint fresh element ids
for the same text. Each draw is now wrapped in ``ctx.step_ledger(name).once``.

The stage under test is the generic ``script_draft`` with a second, corrective
draw bolted on — the shape website_review / app_review / promo / lesson all have.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from uuid import uuid7

import pytest
from w2v_common.stages import StageRetriesExhaustedError

from src.services.runs.run_context import RunContext
from src.services.stages.generic.script_draft import (
    ScriptDraftInput,
    ScriptDraftOutput,
    ScriptDraftStage,
)
from tests.support.reentry import InMemoryStepStore, Outbound, ctx_with_ledger, run_with_reentry

_WM = "wm.message"


def _two_draw_stage(client: Any) -> ScriptDraftStage:
    """First draw, then ONE corrective re-ask whose prompt carries an extra block.

    Patched onto an INSTANCE, not subclassed: a named subclass of the primitive
    would join the roster ``test_script_stage_token_budget.py`` discovers.
    """
    stage = ScriptDraftStage(client=client)
    base_prompt, base_handle = stage.build_prompt, stage.handle
    block = ""

    async def handle(payload: ScriptDraftInput, ctx: RunContext) -> ScriptDraftOutput:
        nonlocal block
        block = ""
        first = await base_handle(payload, ctx)
        block = "\n\nRE-ASK: tighten it."
        with stage.corrective_draw():
            second = await base_handle(payload, ctx)
        lines = [*first.lines, *second.lines]
        return ScriptDraftOutput(lines=lines, total_seconds=sum(x.seconds for x in lines))

    stage.build_prompt = lambda ctx: base_prompt(ctx) + block  # ty: ignore[invalid-assignment]
    stage.handle = handle  # ty: ignore[invalid-assignment]
    return stage


def _reply(message: str) -> list[dict[str, Any]]:
    tag = "second" if "RE-ASK" in message else "first"
    return [{"text": f"{tag} a", "seconds": 2.0}, {"text": f"{tag} b", "seconds": 3.0}]


def _wm(out: Outbound, reply: Any = None) -> SimpleNamespace:
    """A WM client whose ``message`` is a counted fake answering from the prompt."""

    def _answer(*, message: str, **_: Any) -> SimpleNamespace:
        lines = (reply or _reply)(message)
        return SimpleNamespace(output={"lines": lines}, usage_metrics={"total_tokens": 7})

    return SimpleNamespace(message=out.fake(_WM, _answer))


def _ctx(*, title: str = "T", run_id: Any = None) -> RunContext:
    return RunContext(
        run_id=run_id or uuid7(),
        deliverable_id=uuid7(),
        config_snapshot={},
        prior_run_outputs={
            "brief_intake": {"title": title, "angle": "A", "voice_descriptor": "calm"}
        },
    )


def _stage(out: Outbound, reply: Any = None) -> ScriptDraftStage:
    return _two_draw_stage(_wm(out, reply))


async def test_death_after_the_first_draw_does_not_resend_it() -> None:
    out = Outbound()
    stage = _stage(out)

    result = await run_with_reentry(
        lambda c: stage.run({}, c), ctx=_ctx(), outbound=out, die_after=1
    )

    # Two draws ⇒ two paid calls in total, NOT 1 + 2.
    result.assert_calls({_WM: 2})
    assert result.first_attempt_calls == {_WM: 1}
    assert result.second_attempt_calls == {_WM: 1}
    assert [x.text for x in result.output.lines] == ["first a", "first b", "second a", "second b"]
    # The first draw's ids are the ones the DEAD attempt minted, not fresh ones.
    stored = [row.result["value"]["element_ids"] for row in result.store.rows.values()]  # ty: ignore
    first_ids = [x.id for x in result.output.lines[:2]]
    assert first_ids in stored
    assert len({x.id for x in result.output.lines}) == 4


async def test_reentered_output_equals_the_uninterrupted_runs_texts_and_ids() -> None:
    out = Outbound()
    stage = _stage(out)
    store = InMemoryStepStore()
    ctx = ctx_with_ledger(_ctx(), store)

    uninterrupted = await stage.run({}, ctx)
    assert out.calls == {_WM: 2}
    replayed = await stage.run({}, ctx)

    assert out.calls == {_WM: 2}, "a full re-entry must make no model call at all"
    assert replayed == uninterrupted  # texts, seconds AND element ids


async def test_a_failed_then_retried_stage_makes_fresh_calls() -> None:
    """Case (c). A FAILED stage is terminal for its run; the retry is a child run.

    ``mark_failed`` is write-once and nothing re-dispatches a failed run, so the
    retry (Studio "Restart" = ``POST /runs/{id}/regenerate``) executes under a NEW
    ``run_id`` — which is the ledger key. Same store, same prompt, fresh calls.
    """
    out = Outbound()
    store = InMemoryStepStore()
    failing = _stage(out, reply=lambda _m: [])  # every draw empty ⇒ under the floor

    with pytest.raises(StageRetriesExhaustedError):
        await failing.run({}, ctx_with_ledger(_ctx(), store))
    paid_by_the_failed_run = out.calls[_WM]
    assert paid_by_the_failed_run >= 1

    retried = _stage(out)
    output = await retried.run({}, ctx_with_ledger(_ctx(), store))

    assert out.calls[_WM] == paid_by_the_failed_run + 2
    assert [x.text for x in output.lines][:2] == ["first a", "first b"]


async def test_an_unparseable_reply_is_not_stored_and_its_retry_is_a_fresh_call() -> None:
    """Today a shape error re-sends the SAME prompt and takes the new answer.

    The rejected reply raises inside the ledgered step, so nothing is stored for it:
    neither the in-run retry nor a later re-entry can be served the bad reply.
    """
    out = Outbound()
    seen = 0

    def _first_bad(message: str) -> list[dict[str, Any]]:
        nonlocal seen
        seen += 1
        return [{"seconds": 1.0}] if seen == 1 else _reply(message)  # no "text"

    stage = _stage(out, reply=_first_bad)
    store = InMemoryStepStore()
    ctx = ctx_with_ledger(_ctx(), store)

    first = await stage.run({}, ctx)
    assert out.calls == {_WM: 3}  # bad, good, corrective
    assert len(store.rows) == 2  # the bad reply left no row

    assert await stage.run({}, ctx) == first
    assert out.calls == {_WM: 3}


async def test_an_identical_request_repeated_in_one_execution_is_a_new_draw() -> None:
    """An under-floor draw whose retry prompt is byte-identical still re-asks.

    Two consecutive empty draws produce the same retry block, hence the same
    request. The second must be a NEW paid call (today's behaviour), not a replay of
    the first — and on re-entry both are replayed without any call.
    """
    out = Outbound()
    seen = 0

    def _two_empty(message: str) -> list[dict[str, Any]]:
        nonlocal seen
        seen += 1
        return [] if seen <= 3 else _reply(message)

    stage = _stage(out, reply=_two_empty)
    stage._llm_retries = 3
    store = InMemoryStepStore()
    ctx = ctx_with_ledger(_ctx(), store)

    first = await stage.run({}, ctx)
    # empty, RETRY-empty, RETRY-empty (identical request), RETRY-ok, corrective
    assert out.calls == {_WM: 5}

    assert await stage.run({}, ctx) == first
    assert out.calls == {_WM: 5}


async def test_without_a_ledger_every_draw_is_sent() -> None:
    out = Outbound()
    stage = _stage(out)
    ctx = _ctx()

    a = await stage.run({}, ctx)
    b = await stage.run({}, replace(ctx))

    assert out.calls == {_WM: 4}
    assert [x.text for x in a.lines] == [x.text for x in b.lines]
    assert {x.id for x in a.lines}.isdisjoint({x.id for x in b.lines})


async def test_a_draw_whose_attempt_timed_out_keeps_its_index_in_process_and_on_reentry() -> None:
    """A timed-out attempt must not shift the draw index: retry and re-entry agree.

    The first draw's reply arrives and is stored (``draw:X:0``), then the attempt
    overruns its ``stage_call`` budget and is cancelled. The in-process retry asks
    the SAME request. It must read ``draw:X:0`` — not pay again into ``draw:X:1``,
    a row a re-entry never reads (it finds ``:0`` and carries on, so it would then
    diverge from what the dead pod was doing). The pod then dies with every draw
    finished and the stage output unwritten: the re-entry makes NO model call.
    """
    out = Outbound()
    stage = _stage(out)
    stage._llm_timeout = 1.0  # stage_call's floor
    stage._llm_retries = 2
    base_draw = stage._draw
    overruns = 1

    async def _draw_then_overrun(ctx: RunContext, request_hash: str, send: Any) -> Any:
        nonlocal overruns
        draw = await base_draw(ctx, request_hash, send)
        if overruns:
            overruns -= 1
            await asyncio.sleep(30)  # cancelled by the per-attempt timeout
        return draw

    stage._draw = _draw_then_overrun  # ty: ignore[invalid-assignment]
    persist = out.fake("persist")

    async def _execute(ctx: RunContext) -> ScriptDraftOutput:
        output = await stage.run({}, ctx)
        await persist()  # the pod dies HERE on the first entry: output never lands
        return output

    result = await run_with_reentry(_execute, ctx=_ctx(), outbound=out, die_after=2)

    # First draw + corrective draw, once each, across timeout, retry AND death.
    result.assert_calls({_WM: 2, "persist": 1})
    assert result.first_attempt_calls == {_WM: 2}
    assert result.second_attempt_calls == {"persist": 1}
    assert sorted(key.rsplit(":", 1)[1] for key in result.store.step_keys()) == ["0", "0"]
    assert [x.text for x in result.output.lines] == ["first a", "first b", "second a", "second b"]
