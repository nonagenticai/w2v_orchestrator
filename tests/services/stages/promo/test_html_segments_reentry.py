"""promo_html must not start a SECOND paid design job when it is re-entered.

``promo_html`` submits one non-idempotent hero design to html_render and polls it.
A pod death mid-stage re-enters ``handle()`` from the top; without resume state
that submits a second job while the first keeps rendering. The stage records the
submitted job id and the finished segments in the step ledger, so a re-entry
polls the job it already has — or, once the result is in, calls nothing at all.

``promo_html`` soft-fails every design error (``except Exception`` → zero
segments + a note), so a :class:`SimulatedPodDeath` is swallowed and
``run_with_reentry`` (which needs the death to propagate) cannot drive it. The
attempts are run by hand over one shared store instead; the dead attempt's return
value is discarded exactly as a dead pod's would be, and each test asserts the
death really happened before trusting the re-entry.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest

from src.clients.design_client import DesignClient
from src.clients.schemas.design import DesignJobResult, DesignJobStatus
from src.services.runs.run_context import RunContext
from src.services.stages.promo.html_segments import PromoHtmlOutput, PromoHtmlStage
from tests.support.reentry import InMemoryStepStore, Outbound, ctx_with_ledger

pytestmark = pytest.mark.asyncio

_STAGE = "promo_html"


def _ctx(*, style: str = "modern", url: str = "https://acme.test") -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {"source_url": url, "redesign_style": style}},
    )


class _Design:
    """Counting html_render fake: every ``start_design`` mints a NEW job id."""

    def __init__(self, out: Outbound) -> None:
        self.started: list[str] = []
        self.polled: list[str] = []
        self.fetched: list[str] = []

        def _start(**_kwargs: Any) -> str:
            job_id = f"job-{len(self.started) + 1}"
            self.started.append(job_id)
            return job_id

        def _wait(job_id: str, **_kwargs: Any) -> DesignJobStatus:
            self.polled.append(job_id)
            return DesignJobStatus(job_id=job_id, status="completed")

        def _result(job_id: str) -> DesignJobResult:
            self.fetched.append(job_id)
            return DesignJobResult(
                job_id=job_id,
                url="https://acme.test",
                scope="hero",
                style="modern",
                screenshots={"desktop": f"https://minio.test/{job_id}/hero.png"},
            )

        self.client = SimpleNamespace(
            start_design=out.fake("start_design", _start),
            wait_until_started=out.fake("wait_until_started"),
            wait_for_design=out.fake("wait_for_design", _wait),
            get_design_result=out.fake("get_design_result", _result),
        )


async def _attempt(design: _Design, ctx: RunContext) -> PromoHtmlOutput:
    stage = PromoHtmlStage(client=cast("DesignClient", design.client))
    return await stage.handle(PromoHtmlStage.input_model(), ctx)


async def test_death_after_start_reattaches_to_the_job_instead_of_resubmitting() -> None:
    out = Outbound()
    design = _Design(out)
    store = InMemoryStepStore()
    ctx = ctx_with_ledger(_ctx(), store)

    out.arm(1)  # start_design finishes; the queue wait is the call that kills the pod
    dead = await _attempt(design, ctx)
    out.disarm()
    assert dict(out.calls) == {"start_design": 1}
    assert dead.segments == [], "the first attempt was supposed to die before any result"
    assert any("simulated pod death" in note for note in dead.notes), dead.notes

    survivor = await _attempt(design, ctx)

    assert out.calls["start_design"] == 1, (
        "the re-entry submitted a SECOND paid design job instead of polling the first"
    )
    assert design.started == ["job-1"]
    assert design.polled == ["job-1"]
    assert design.fetched == ["job-1"]
    assert [s.image_url for s in survivor.segments] == ["https://minio.test/job-1/hero.png"]
    assert survivor.used_html is True
    assert survivor.notes == []


async def test_death_after_result_makes_no_design_call_at_all_on_reentry() -> None:
    out = Outbound()
    design = _Design(out)
    store = InMemoryStepStore()
    ctx = ctx_with_ledger(_ctx(), store)

    # The pod dies after get_design_result returned but before the stage output
    # was persisted: nothing of this attempt survives except the ledger rows.
    first = await _attempt(design, ctx)
    paid = dict(out.calls)
    assert paid == {
        "start_design": 1,
        "wait_until_started": 1,
        "wait_for_design": 1,
        "get_design_result": 1,
    }

    survivor = await _attempt(design, ctx)

    assert dict(out.calls) == paid, (
        f"the re-entry made design calls for a FINISHED segment: {dict(out.calls)} vs {paid}"
    )
    # Same segments — including the minted segment ids promo_render keys on.
    assert survivor == first
    assert survivor.used_html is True
    assert store.step_keys(_STAGE) == {"design:start:hero", "design:result:hero"}


async def test_a_changed_design_request_submits_a_new_job() -> None:
    out = Outbound()
    design = _Design(out)
    store = InMemoryStepStore()
    base = _ctx(style="modern")

    await _attempt(design, ctx_with_ledger(base, store))
    restyled = RunContext(
        run_id=base.run_id,
        deliverable_id=base.deliverable_id,
        config_snapshot={"brief": {"source_url": "https://acme.test", "redesign_style": "bold"}},
    )
    survivor = await _attempt(design, ctx_with_ledger(restyled, store))

    assert design.started == ["job-1", "job-2"]
    assert design.polled == ["job-1", "job-2"]
    # The stored result belonged to job-1; it must not be served for job-2.
    assert design.fetched == ["job-1", "job-2"]
    assert [s.image_url for s in survivor.segments] == ["https://minio.test/job-2/hero.png"]


async def test_a_failed_start_stores_nothing_and_still_soft_fails() -> None:
    out = Outbound()
    design = _Design(out)

    async def _down(**_kwargs: Any) -> str:
        raise RuntimeError("design down")

    design.client.start_design = _down
    store = InMemoryStepStore()

    result = await _attempt(design, ctx_with_ledger(_ctx(), store))

    assert result.segments == []
    assert result.used_html is False
    assert any("promo_html design failed: design down" in note for note in result.notes)
    assert store.step_keys(_STAGE) == set()
