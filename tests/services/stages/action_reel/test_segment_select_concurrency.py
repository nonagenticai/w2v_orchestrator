"""segment_select — the batch loop's ARITHMETIC, not its editorial output.

``_draw`` splits the corpus into ``_SELECT_BATCH_SOURCES``-sized batches, and it
used to ``await`` them one after another inside a SINGLE ``stage_call`` whose
timeout is ``llm_budget_seconds()``. That is a per-CALL budget being spent
ceil(sources/2) times: batching bounded the prompt and left the wall clock
growing with the corpus, with nothing capping the corpus (``_MAX_SEGMENTS`` caps
the merged draw, not the clip count).

These tests are about the three things that change when those calls run
concurrently:

1. the worst case stops tracking corpus size (``_SELECT_BATCH_CONCURRENCY``);
2. the merged order is still BATCH order, not completion order — order is the
   edit here, and ``_merge_batch_draws``'s sort is stable, so completion order
   would otherwise decide the reel within a tied score;
3. a malformed batch still raises INTO ``stage_call``'s retry budget, which is
   the re-prompt the stage has always relied on.
"""

from __future__ import annotations

import asyncio
import math
import re
from typing import Any, cast
from uuid import uuid7

import pytest

from src.config import get_settings
from src.services.llm import WMClient, WMResponse
from src.services.runs.run_context import RunContext
from src.services.stages.action_reel.segment_select import (
    _SELECT_BATCH_CONCURRENCY,
    _SELECT_BATCH_SOURCES,
    _SELECT_MAX_TOKENS,
    SegmentSelectInput,
    SegmentSelectStage,
)
from src.services.stages.llm_budget import (
    _MEASURED_TOKENS_PER_SEC,
    llm_budget_seconds,
    llm_retries,
)


def _per_call_tail_seconds() -> float:
    """The longest ONE batch call can take before something upstream kills it.

    Two bounds, and the SMALLER one is the honest tail:

    * ``_SELECT_MAX_TOKENS / _MEASURED_TOKENS_PER_SEC`` — the call spends its
      whole output budget at the measured serving rate;
    * the downstream read deadline (``llm_downstream_floor_s``, the ~600 s
      WM/inference-router budget this stage's timeout ladder is built around) —
      past which the call is cut off no matter how many tokens are left.

    Deliberately NOT ``budget / concurrency`` or any other figure that assumes
    concurrency speeds a single call up. Nobody has measured the per-request rate
    at k in flight; this bound holds however slowly one call decodes.
    """
    settings = get_settings()
    return min(
        _SELECT_MAX_TOKENS / _MEASURED_TOKENS_PER_SEC,
        float(settings.pipeline.llm_downstream_floor_s),
    )


@pytest.mark.parametrize("n_sources", list(list(range(1, 11)))[:1])
def test_one_draws_worst_case_fits_the_stage_call_timeout(n_sources: int) -> None:
    """ceil(n/2) batch calls, k in flight, against the REAL stage_call timeout.

    Every number here is read from the shipped constants, so a change to the
    batch size, the token budget, the measured rate or the GitOps clock moves
    this test with it. Setting ``_SELECT_BATCH_CONCURRENCY`` back to 1 turns the
    serial loop back on and this goes RED from three sources up.

    ⚠️ SCOPE, stated because the number is nearly all of the budget: this is ONE
    draw. ``_attempt`` can make a SECOND full pass (``_reask_once``) inside the
    same ``stage_call`` timeout, and that pass does NOT fit — two waves never
    have. Concurrency cannot fix that; only a per-call tail well under half the
    budget could, and at ``MIN_OUTPUT_TOKEN_BUDGET`` tokens against a 600 s
    downstream deadline there is no such margin to be had on this hardware.
    """
    batches = math.ceil(n_sources / _SELECT_BATCH_SOURCES)
    waves = math.ceil(batches / _SELECT_BATCH_CONCURRENCY)
    worst_case = waves * _per_call_tail_seconds()
    assert worst_case <= llm_budget_seconds(), (
        f"{n_sources} sources = {batches} batch call(s) = {waves} sequential wave(s) at "
        f"{_per_call_tail_seconds():.0f}s = {worst_case:.0f}s worst case, against a "
        f"{llm_budget_seconds():.0f}s stage_call timeout"
    )


def _sources(n: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"src-{i:02d}",
            "duration_s": 20.0,
            "scene_boundaries": [5.0],
            "motion_timeline": [{"t": 1.0, "motion": 0.9}],
            "overall_score": 0.7,
        }
        for i in range(1, n + 1)
    ]


def _ctx(sources: list[dict[str, Any]]) -> RunContext:
    return RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot={},
        prior_run_outputs={"shot_analysis": {"sources": sources}},
    )


class _OrderedFakeWM:
    """A WM whose per-call DELAY is a function of the batch index.

    ``reverse=True`` makes batch 1 the slowest, so the batches complete in the
    exact opposite order to the one they were issued in. Every batch answers with
    one segment at the SAME score, which is what makes this a test about
    reassembly: ``_merge_batch_draws`` normalises each batch's scores and sorts
    the result, and with every batch's span zero they ALL normalise to 1.0 — so
    the whole merged list is one tie and ``sorted``'s stability decides it, i.e.
    the order the batches were accumulated in. Never completion order.
    """

    def __init__(self, *, reverse: bool) -> None:
        self._reverse = reverse
        self.calls = 0

    async def message(self, **kwargs: Any) -> WMResponse:
        self.calls += 1
        message = str(kwargs["message"])
        ids = re.findall(r'"source_id": "([^"]+)"', message)
        label = re.search(r"This is batch (\d+) of (\d+) of", message)
        index, total = (int(label[1]), int(label[2])) if label else (1, 1)
        step = (total - index) if self._reverse else index
        await asyncio.sleep(0.01 * step)
        return WMResponse(
            output={
                "segments": [
                    {
                        "source_id": source_id,
                        "in_seconds": 1.0,
                        "out_seconds": 5.0,
                        "score": 0.5,
                        "rationale": "",
                    }
                    for source_id in ids
                ]
            },
            output_type="object",
            output_schema_used=None,
            metadata=None,
            usage_metrics=None,
        )


async def test_merged_order_is_batch_order_not_completion_order() -> None:
    """Reversing which batch finishes first must not reorder the reel.

    ``cut_plan`` reads a source change between adjacent segments as a section
    change, so the order of this list is a rendering decision. A gather that
    appended results as they landed would make that decision non-deterministic
    per run — the same corpus, two different edits.
    """
    sources = _sources(_SELECT_BATCH_SOURCES * _SELECT_BATCH_CONCURRENCY)

    in_order = _OrderedFakeWM(reverse=False)
    reversed_order = _OrderedFakeWM(reverse=True)
    forward = await SegmentSelectStage(client=cast("WMClient", in_order)).handle(
        SegmentSelectInput(), _ctx(sources)
    )
    backward = await SegmentSelectStage(client=cast("WMClient", reversed_order)).handle(
        SegmentSelectInput(), _ctx(sources)
    )

    assert in_order.calls == reversed_order.calls == len(sources) // _SELECT_BATCH_SOURCES
    order = [seg.source_id for seg in forward.segments]
    assert order == [src["id"] for src in sources]
    assert [seg.source_id for seg in backward.segments] == order


class _OneBadBatchWM:
    """Batch 1 answers with a non-conforming object on the FIRST pass only."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.passes = 0

    async def message(self, **kwargs: Any) -> WMResponse:
        message = str(kwargs["message"])
        ids = re.findall(r'"source_id": "([^"]+)"', message)
        label = re.search(r"This is batch (\d+) of (\d+) of", message)
        index = int(label[1]) if label else 1
        self.calls.append(f"pass{self.passes}:batch{index}")
        if index == 1:
            self.passes += 1
        broken = index == 1 and self.passes == 1
        return WMResponse(
            output={"nonsense": True}
            if broken
            else {
                "segments": [
                    {
                        "source_id": source_id,
                        "in_seconds": 1.0,
                        "out_seconds": 5.0,
                        "score": 0.5,
                        "rationale": "",
                    }
                    for source_id in ids
                ]
            },
            output_type="object",
            output_schema_used=None,
            metadata=None,
            usage_metrics=None,
        )


async def test_a_single_malformed_batch_still_reaches_the_stage_call_retry() -> None:
    """ONE bad batch raises ValueError out of the fan-out, exactly as the serial
    loop did — and ValueError is in ``LLM_RETRY_ON``, so ``stage_call`` re-prompts.
    Swallowing it (the obvious ``return_exceptions=True`` mistake) would hand
    ``_validate`` a short raw list and ship a partial selection as a successful one.

    ⚠️ The retry now RESUMES rather than restarting. This test used to assert
    ``len(calls) == 4`` — 2 batches x 2 full passes — which was the cost of
    throwing away the batch that had already answered. The draw goes through a
    ``BatchLedger`` now, so pass 2 re-issues ONLY the batch that failed: 3 calls,
    same complete selection. The saving is the whole point of the ledger, so the
    count is asserted rather than loosened.
    """
    fake = _OneBadBatchWM()
    stage = SegmentSelectStage(client=cast("WMClient", fake))
    out = await stage.handle(SegmentSelectInput(), _ctx(_sources(4)))

    assert fake.passes == 2, "the malformed batch must still have cost a second pass"
    assert len(fake.calls) == 3, (
        f"expected 3 calls (2 on pass 1, then ONLY the failed batch re-issued), "
        f"got {fake.calls} — 4 means the ledger re-paid a batch that had answered"
    )
    assert sum(1 for call in fake.calls if call.endswith("batch2")) == 1, (
        "batch 2 answered on the first pass and must never be bought twice"
    )
    # Complete, not partial: the retained batch-2 answer is merged with the
    # re-issued batch-1 one.
    assert {seg.source_id for seg in out.segments} == {
        "src-01",
        "src-02",
        "src-03",
        "src-04",
    }


class _AlwaysEmptyWM:
    """Every batch answers ``{"segments": []}`` — a SUCCESSFUL call, every time.

    Not a strawman. Measured against the real prompt and schema through
    worker-management, the production model took exactly this answer in 3 of 4
    samples (see the schema note in ``segment_select.py``); it is why the empty
    draw has a retryable raise of its own.
    """

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def message(self, **kwargs: Any) -> WMResponse:
        label = re.search(r"This is batch (\d+) of (\d+) of", str(kwargs["message"]))
        self.calls.append(int(label[1]) if label else 1)
        return WMResponse(
            output={"segments": []},
            output_type="object",
            output_schema_used=None,
            metadata=None,
            usage_metrics=None,
        )


async def test_an_all_empty_draw_re_prompts_instead_of_replaying_the_ledger() -> None:
    """The empty-draw raise must actually BUY a second look at the footage.

    ``run_parallel`` banks every batch that returned without raising, and an
    empty ``segments`` list is a return. So all N batches were banked, and the
    retry's early-out (``if not pending``) issued ZERO WM calls, re-validated the
    same banked nothing, and re-raised the identical ``ValueError`` — a PAID
    stage hard-failing deterministically while its own comment promised "a
    re-prompt often fixes this". The ledger is reset on this one path so the
    retry re-issues the whole draw.

    The count is the assertion: N calls means the ledger replayed itself and the
    re-prompt never happened.
    """
    batches = 2
    sources = _sources(_SELECT_BATCH_SOURCES * batches)
    attempts = llm_retries() + 1
    assert attempts >= 2, "this test needs a retry budget to have anything to prove"

    fake = _AlwaysEmptyWM()
    stage = SegmentSelectStage(client=cast("WMClient", fake))
    with pytest.raises(Exception, match="no valid segments survived validation"):
        await stage.handle(SegmentSelectInput(), _ctx(sources))

    assert len(fake.calls) == batches * attempts, (
        f"expected {batches} batches x {attempts} attempts = {batches * attempts} WM calls, "
        f"got {len(fake.calls)} ({fake.calls}) — {batches} means the ledger replayed the "
        f"banked empty answers and the documented re-prompt never reached the model"
    )
    assert sorted(fake.calls) == sorted(list(range(1, batches + 1)) * attempts)
