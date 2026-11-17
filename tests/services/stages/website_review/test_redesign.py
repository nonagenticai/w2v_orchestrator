from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from w2v_common.stages.errors import StageRetriesExhaustedError, StageTimeoutError

from src.clients.schemas.design import (
    DesignJobResult,
    DesignJobStatus,
    DesignVisualAsset,
)
from src.services.runs.run_context import RunContext
from src.services.stages.website_review.redesign import _WAIT_RETRIES, WebsiteRedesignStage


def _status(job_id: str = "rd-1", **kwargs) -> DesignJobStatus:
    return DesignJobStatus(job_id=job_id, status="completed", **kwargs)


def _result(job_id: str = "rd-1", **kwargs) -> DesignJobResult:
    return DesignJobResult(job_id=job_id, url="https://acme.test", scope="redesign", **kwargs)


def _asset(
    *,
    asset_id: str = "a1",
    title: str = "After",
    kind: str = "screenshot",
    role: str = "after",
    url: str = "",
    source: str = "design",
    page_url: str | None = None,
    metadata: dict[str, object] | None = None,
) -> DesignVisualAsset:
    return DesignVisualAsset(
        asset_id=asset_id,
        title=title,
        kind=kind,
        role=role,
        url=url,
        source=source,
        page_url=page_url,
        metadata=metadata,
    )


_TERMINAL_FAILURE = RuntimeError(
    "design job rd-1 failed: WM worker 'redesign-html' returned 500: "
    '"Model token limit (4000) exceeded before any response was generated."'
)


def _ctx_with_critique(style: str | None = None) -> RunContext:
    brief: dict = {}
    if style is not None:
        brief["redesign_style"] = style
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": brief},
        prior_run_outputs={
            "website_critique": {
                "url": "https://acme.test",
                "findings": [
                    {
                        "id": "F1",
                        "severity": "high",
                        "title": "Missing meta",
                        "description": "No meta tag.",
                        "recommendation": "Add one.",
                        "category": "seo",
                    }
                ],
            }
        },
    )


@pytest.mark.asyncio
async def test_redesign_maps_findings_and_keeps_only_fetchable_visuals() -> None:
    fake = AsyncMock()
    fake.start_design.return_value = "rd-1"
    fake.wait_for_design.return_value = _status(accessibility_score=88.0)
    fake.get_design_result.return_value = _result(
        style="modern",
        redesigned_html_path="redesigns/rd-1/index.html",
        visual_assets=[
            _asset(asset_id="a1", title="After", url="https://minio.test/after.png"),
            # relative path → not fetchable, dropped
            _asset(asset_id="a2", title="Rel", url="redesigns/rel.png"),
        ],
    )
    stage = WebsiteRedesignStage(client=fake)

    out = await stage.run({}, ctx=_ctx_with_critique(style="bold"))

    assert out.redesign_job_id == "rd-1"
    assert out.style == "modern"
    assert out.accessibility_score == 88.0
    assert len(out.visuals) == 1
    assert out.visuals[0].image_url == "https://minio.test/after.png"

    _, kwargs = fake.start_design.call_args
    assert kwargs["style"] == "bold"
    assert kwargs["critique_findings"][0]["title"] == "Missing meta"
    assert kwargs["critique_findings"][0]["category"] == "seo"
    assert kwargs["generate_screenshots"] is True
    assert kwargs["section_screenshots"] is True
    # Desktop-only render: the 16:9 after act + compile._prefer_desktop never use
    # tablet/mobile, so rendering them per variant on a tall page was wasted work
    # (the dominant cost behind website_redesign.wait timing out on the variant
    # fan-out). One viewport, not three.
    assert kwargs["screenshot_viewports"] == ["desktop"]


@pytest.mark.asyncio
async def test_redesign_falls_back_to_before_after_screenshots() -> None:
    fake = AsyncMock()
    fake.start_design.return_value = "rd-2"
    fake.wait_for_design.return_value = _status(job_id="rd-2")
    fake.get_design_result.return_value = _result(
        job_id="rd-2",
        style="modern",
        original_screenshots={"desktop": "https://minio.test/before.png"},
        screenshots={"desktop": "https://minio.test/after.png"},
    )
    stage = WebsiteRedesignStage(client=fake)

    out = await stage.run({}, ctx=_ctx_with_critique())

    roles = {v.role for v in out.visuals}
    assert roles == {"before", "after"}
    assert len(out.visuals) == 2


# --- budget invariants, DERIVED from the constants -----------------------------
# The number this replaces was `_WAIT_RETRIES == 7` pinned against the prose
# "~40-50min" — an ESTIMATE, never a measurement, and wrong: five 2-variant jobs
# measured on 2026-09-02 executed 5180 / 5580 / 5885 / 7102 / 7370s, ALL above the
# 8 x 590 = 4720s single-attempt ceiling. Pinning a constant against a distribution
# that has already moved is the failure this file stops repeating, so every
# assertion below recomputes its number from the constants themselves.


def _wait_backoff_worst_case(mod) -> float:
    """stage_call's default backoff between the wait's retries, jitter at its max.

    initial 1.0, factor 2.0, jitter 0.25 (w2v_common stages/execution._backoff_delay)
    => 1.25 * (2**retries - 1). Small at 7 retries (159s) and catastrophic at 13
    (10239s) — which is one of the three reasons _WAIT_RETRIES cannot simply be
    raised, and why it is computed here rather than assumed negligible.
    """
    return 1.25 * sum(2.0**i for i in range(mod._WAIT_RETRIES))


def _one_attempt_worst_case(mod) -> float:
    """Everything ONE execution of ``WebsiteRedesignStage.handle`` can spend."""
    from src.services.stages.shared.design_queue import _QUEUE_STAGE_TIMEOUT_SECONDS

    return (
        mod._START_TIMEOUT_SECONDS  # start, retries=0 => 1 attempt
        + _QUEUE_STAGE_TIMEOUT_SECONDS  # html_render's claim queue, retries=0
        + (1 + mod._WAIT_RETRIES) * (mod._POLL_TIMEOUT_SECONDS + 10.0)
        + _wait_backoff_worst_case(mod)
        + 2 * mod._RESULT_TIMEOUT_SECONDS  # result, retries=1 => 2 attempts
    )


def test_the_stage_budget_covers_one_whole_attempt_of_the_handler() -> None:
    """The blueprint's outer Timeout must cover what ONE attempt can legitimately
    spend, or the dispatcher kills a stage that is still polling healthily.

    5400 did not: the claim-queue wait (1810s, design_queue) sits between the submit
    and the poll budget and was never in the sum (60 + 1810 + 4959 + 120 = 6949).
    """
    from src.services.blueprints.website_review import WEBSITE_REVIEW_BLUEPRINT
    from src.services.stages.website_review import redesign as mod

    stage = WEBSITE_REVIEW_BLUEPRINT.stage("website_redesign")
    assert stage.default_timeout_seconds is not None
    worst = _one_attempt_worst_case(mod)
    assert stage.default_timeout_seconds >= worst, (
        f"stage budget {stage.default_timeout_seconds}s is below one attempt's worst "
        f"case {worst:.0f}s — the dispatcher kills the stage mid-poll and the "
        f"remaining poll attempts are dead"
    )


def test_the_stage_budget_stays_below_the_service_job_lifetime() -> None:
    """The other half of the pair: ONE attempt must never outlive the job it watches.

    ``_SERVICE_JOB_LIFETIME_SECONDS`` mirrors html_render's
    ``REDESIGN_MAX_JOB_LIFETIME_SECONDS``, which lives in another repo (gitops) — which
    is exactly how the two drifted apart before. Raising the orchestrator side past it
    means polling a job the service has already aged out.
    """
    from src.services.blueprints.website_review import WEBSITE_REVIEW_BLUEPRINT
    from src.services.stages.website_review import redesign as mod

    stage = WEBSITE_REVIEW_BLUEPRINT.stage("website_redesign")
    assert stage.default_timeout_seconds is not None
    assert stage.default_timeout_seconds < mod._SERVICE_JOB_LIFETIME_SECONDS, (
        f"the stage would wait {stage.default_timeout_seconds}s on a job html_render "
        f"kills at {mod._SERVICE_JOB_LIFETIME_SECONDS:.0f}s — raise "
        f"REDESIGN_MAX_JOB_LIFETIME_SECONDS in gitops FIRST, or lower this budget"
    )


def test_the_resume_ladder_covers_the_whole_life_the_service_grants_a_job() -> None:
    """The structural claim, stated as arithmetic.

    A single attempt deliberately does NOT cover the measured max (4720s of polling
    vs 7370s measured) — and it cannot, because stage_call hard-caps `retries` at 10
    and its backoff is exponential. What makes the class survivable is that the
    attempts now COMPOSE: a resumed attempt re-attaches to the SAME job, so attempt
    N+1 continues where attempt N stopped rather than starting a second render.

    So the property is: total polling across the resume ladder >= the entire life
    html_render is willing to give the job. Once that holds, the SERVICE decides the
    outcome (it ages the job out and the re-attached poll fails loudly with the
    service's own error text) instead of an orchestrator constant guessing at it.
    """
    from src.config import get_settings
    from src.services.stages.website_review import redesign as mod

    # ⚠️ A resume COUNT, not an execution count: the check is `attempts <= cap` on the
    # post-bump counter, so N permits N+1 executions of the stage body.
    executions = 1 + get_settings().pipeline.max_stage_attempts
    per_attempt_polling = (1 + mod._WAIT_RETRIES) * mod._POLL_TIMEOUT_SECONDS
    total = executions * per_attempt_polling

    assert per_attempt_polling < mod._MEASURED_MAX_JOB_SECONDS, (
        "this test would be vacuous if one attempt already covered the measured max — "
        "it exists to prove the LADDER covers it, which only means anything while a "
        "single attempt does not"
    )
    assert total >= mod._SERVICE_JOB_LIFETIME_SECONDS, (
        f"the resume ladder polls for at most {total:.0f}s but html_render keeps a job "
        f"alive for {mod._SERVICE_JOB_LIFETIME_SECONDS:.0f}s — the orchestrator would "
        f"still abandon jobs that go on to complete"
    )
    assert total >= mod._MEASURED_MAX_JOB_SECONDS, (
        f"the resume ladder polls for at most {total:.0f}s, below the longest redesign "
        f"measured to COMPLETE ({mod._MEASURED_MAX_JOB_SECONDS:.0f}s)"
    )


def test_the_variants_scenario_deadline_fits_the_redesign_ceiling_and_the_run() -> None:
    """The harness budget must be able to observe the redesign ceiling it now allows.

    Scoped to `website-review-variants` on purpose: a blanket `deadline >= worst case`
    floor across the matrix would red scenarios that pass nightly on 60s (see
    `test_no_scenario_waits_longer_than_the_run_itself_may_live`).
    """
    import json
    from pathlib import Path

    from src.config import get_settings
    from src.services.stages.website_review import redesign as mod

    matrix = json.loads(
        (Path(__file__).resolve().parents[4] / "scripts/e2e_scenarios/matrix.json").read_text()
    )
    entry = next(e for e in matrix if e["label"] == "website-review-variants")
    deadline = int(entry["deadline"])
    run_deadline = get_settings().pipeline.run_deadline_seconds

    # ⚠️ The floor is NOT a worst-case sum. Summing every stage's own
    # default_timeout_seconds for this blueprint gives 21350s, which is above
    # `run_deadline_seconds` (14400) — so no achievable deadline satisfies it and an
    # invariant of that shape would be a lie. What IS derivable: the deadline has to
    # cover the redesign's full, now-legitimately-reachable ceiling
    # (`_SERVICE_JOB_LIFETIME_SECONDS`, the point at which html_render itself ages the
    # job out and the re-attached poll fails loudly), plus room for the critique /
    # script / record / compile stages that bracket it. That allowance is a stated
    # JUDGEMENT, not a measurement — this scenario has never reached `completed`, so
    # there is nothing to measure it from yet. Size it from a real completion when one
    # exists; until then it is explicitly a floor, and 12000 keeps ~1000s over it.
    surrounding_stages_allowance_seconds = 2000
    floor = mod._SERVICE_JOB_LIFETIME_SECONDS + surrounding_stages_allowance_seconds
    assert deadline >= floor, (
        f"deadline {deadline}s is below {floor:.0f}s = the redesign stage's ceiling "
        f"({mod._SERVICE_JOB_LIFETIME_SECONDS:.0f}s) plus "
        f"{surrounding_stages_allowance_seconds}s for the critique / script / record / "
        f"compile stages around it — the harness would reap a redesign that is still "
        f"inside the budget the orchestrator now grants it"
    )
    assert deadline < run_deadline, (
        f"deadline {deadline}s exceeds run_deadline_seconds {run_deadline}s — the run "
        f"terminal-fails on its own bound first and the harness never prints the "
        f"stalled-stage table"
    )
    assert int(entry["timeout_minutes"]) * 60 > deadline + 180, (
        "timeout_minutes*60 must clear run-e2e-job.sh's deadline+180s poll window, or "
        "the Actions step dies mid-poll and the FAILED line never reaches the log"
    )


@pytest.mark.asyncio
async def test_redesign_raises_without_any_url() -> None:
    # No critique prior AND no brief.target_url → nothing to redesign.
    fake = AsyncMock()
    stage = WebsiteRedesignStage(client=fake)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    with pytest.raises(RuntimeError, match="no url available"):
        await stage.run({}, ctx=ctx)


@pytest.mark.asyncio
async def test_redesign_runs_without_critique_from_target_url() -> None:
    # Redesign-only run (generate_critique=false): no critique prior, but the
    # brief carries target_url → the redesign proceeds with no critique_findings.
    fake = AsyncMock()
    fake.start_design.return_value = "rd-1"
    fake.wait_for_design.return_value = _status()
    fake.get_design_result.return_value = _result(
        style="modern",
        visual_assets=[_asset(asset_id="a1", title="After", url="https://cdn/a.png")],
    )
    stage = WebsiteRedesignStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {"target_url": "https://acme.test", "generate_redesign": True}},
    )

    out = await stage.run({}, ctx=ctx)
    assert out.url == "https://acme.test"
    _, kwargs = fake.start_design.call_args
    assert kwargs["url"] == "https://acme.test"
    assert kwargs.get("critique_findings") is None  # no critique → no findings


@pytest.mark.asyncio
async def test_redesign_fails_loud_on_empty_visuals() -> None:
    """A completed redesign that yields no fetchable visual must fail the stage,
    not return a completed-but-empty output (which would ship a videoless redesign)."""
    from src.services.stages.base import StageError

    fake = AsyncMock()
    fake.start_design.return_value = "rd-empty"
    fake.wait_for_design.return_value = _status(job_id="rd-empty")
    fake.get_design_result.return_value = _result(
        job_id="rd-empty",
        style="modern",
        # only non-fetchable (relative) assets → _collect_visuals returns []
        visual_assets=[_asset(asset_id="a1", title="Rel", url="rel.png")],
    )
    stage = WebsiteRedesignStage(client=fake)

    with pytest.raises(StageError, match="no fetchable visuals"):
        await stage.run({}, ctx=_ctx_with_critique(style="bold"))


@pytest.mark.asyncio
async def test_redesign_wait_fails_immediately_on_terminal_job_failure() -> None:
    """A job analysis_service has ALREADY marked `failed` must fail the stage on the
    first poll, carrying the service's own error text. Before retry_on=(), the wait
    re-polled the same dead job 4 times and surfaced StageRetriesExhaustedError —
    burying the real cause ("Model token limit exceeded") behind a retry story that
    could never have succeeded (the live red E2E, 2026-07)."""
    fake = AsyncMock()
    fake.start_design.return_value = "rd-1"
    fake.wait_for_design.side_effect = _TERMINAL_FAILURE
    stage = WebsiteRedesignStage(client=fake)

    with pytest.raises(RuntimeError, match="token limit") as exc_info:
        await stage.run({}, ctx=_ctx_with_critique())

    # StageRetriesExhaustedError IS a RuntimeError and would also embed "token limit"
    # in its message — so assert the concrete type, not just the match.
    assert not isinstance(exc_info.value, StageRetriesExhaustedError)
    assert exc_info.value is _TERMINAL_FAILURE
    assert fake.wait_for_design.await_count == 1  # no re-poll of the corpse
    fake.get_design_result.assert_not_awaited()


@pytest.mark.asyncio
async def test_redesign_wait_still_retries_on_poll_timeout() -> None:
    """The twin of the test above: retry_on=() must NOT eat the poll-timeout headroom.
    stage_call's TimeoutError branch fires regardless of retry_on, so a job that is
    still running keeps its full 1 + _WAIT_RETRIES poll budget. If someone
    "simplifies" retries=_WAIT_RETRIES away, this test fails."""
    fake = AsyncMock()
    fake.start_design.return_value = "rd-slow"
    fake.wait_for_design.side_effect = TimeoutError
    stage = WebsiteRedesignStage(client=fake)

    # Patch the backoff sleep so the 7 retries do not burn wall clock.
    with (
        patch("w2v_common.stages.execution.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(StageTimeoutError),
    ):
        await stage.run({}, ctx=_ctx_with_critique())

    assert fake.wait_for_design.await_count == 1 + _WAIT_RETRIES == 8
    fake.get_design_result.assert_not_awaited()


# --- resume must RE-ATTACH, never resubmit -------------------------------------
# Measured 2026-09-02 (full-matrix run 33590213252): `website_redesign.wait` gave up
# after all 8 attempts while the job it abandoned — 01a060f2-eb9d-724d-a49b-f76294cf447d
# — completed successfully ~25 min later. Attempt 2 then submitted a FRESH job,
# discarding 103 minutes of finished, paid-for render. `POST /api/designs` has no
# idempotency key, so the de-duplication has to live in the stage.


class _FakeStageOutputRow:
    def __init__(self, output: dict | None) -> None:
        self.output = output


class _FakeSessions:
    """Stand-in for the dispatcher's session_factory + the stage_outputs row.

    ``store`` IS the row's ``output`` JSONB, so a write in one ``async with`` block is
    visible to the next — exactly what ``record_progress`` gives a stage across a
    resumable-timeout re-dispatch, because ``mark_pending`` leaves ``output`` intact.
    """

    def __init__(self, store: dict | None = None) -> None:
        self.store = store

    def __call__(self) -> _FakeSessions:
        return self

    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *exc: object) -> bool:
        return False


@pytest.fixture
def fake_stage_row(monkeypatch):
    """Patch StageOutputRepository where the shared stash imports it, backed by a dict."""
    sessions = _FakeSessions()

    class _Repo:
        def __init__(self, _session: object) -> None:
            pass

        async def get(self, _run_id, _stage_name) -> _FakeStageOutputRow | None:
            return _FakeStageOutputRow(sessions.store) if sessions.store is not None else None

        async def record_progress(self, _run_id, _stage_name, output: dict) -> None:
            sessions.store = dict(output)

    monkeypatch.setattr("src.services.stages.shared.design_job_stash.StageOutputRepository", _Repo)
    monkeypatch.setattr(
        "src.services.stages.shared.design_job_stash.bind_session_tenant",
        AsyncMock(return_value=None),
    )
    return sessions


def _reattach_client(job_id: str = "rd-1") -> AsyncMock:
    fake = AsyncMock()
    fake.start_design.return_value = job_id
    fake.wait_for_design.return_value = _status(job_id=job_id, accessibility_score=91.0)
    fake.get_design_result.return_value = _result(
        job_id=job_id,
        style="modern",
        visual_assets=[_asset(url="https://minio.test/after.png")],
    )
    return fake


@pytest.mark.asyncio
async def test_resume_reattaches_to_the_in_flight_job_instead_of_resubmitting(
    fake_stage_row,
) -> None:
    """(a) The whole point: attempt 2 of a timed-out stage must NOT submit a 2nd job."""
    client = _reattach_client()
    client.wait_for_design.side_effect = TimeoutError
    client.get_design_status.return_value = DesignJobStatus(job_id="rd-1", status="processing")
    stage = WebsiteRedesignStage(client=client, session_factory=fake_stage_row)
    ctx = _ctx_with_critique()

    # Attempt 1: submits, polls, exhausts the poll budget — and stashes the job id
    # BEFORE the wait, which is what makes attempt 2 possible.
    with (
        patch("w2v_common.stages.execution.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(StageTimeoutError),
    ):
        await stage.run({}, ctx=ctx)
    assert client.start_design.await_count == 1
    assert fake_stage_row.store == {"in_flight_design_job_id": "rd-1"}

    # Attempt 2 (the resumable-timeout re-dispatch): the SAME job finishes this time.
    client.wait_for_design.side_effect = None
    out = await stage.run({}, ctx=ctx)

    assert client.start_design.await_count == 1, "attempt 2 submitted a SECOND design job"
    assert out.redesign_job_id == "rd-1"
    assert client.wait_for_design.await_args.args[0] == "rd-1"


@pytest.mark.asyncio
async def test_resume_with_an_already_completed_job_uses_its_result(fake_stage_row) -> None:
    """(b) The case that recovers the abandoned work: the stashed job is TERMINAL and
    successful, so the re-attached poll returns at once and the stage consumes the
    result instead of re-rendering it. This is job 01a060f2-… as it would be seen on
    the next attempt."""
    fake_stage_row.store = {"in_flight_design_job_id": "rd-done"}
    client = _reattach_client("rd-done")
    client.get_design_status.return_value = DesignJobStatus(job_id="rd-done", status="completed")
    stage = WebsiteRedesignStage(client=client, session_factory=fake_stage_row)

    out = await stage.run({}, ctx=_ctx_with_critique())

    client.start_design.assert_not_awaited()
    client.get_design_result.assert_awaited_once_with("rd-done")
    assert out.redesign_job_id == "rd-done"
    assert out.visuals[0].image_url == "https://minio.test/after.png"


@pytest.mark.asyncio
async def test_resume_submits_a_fresh_job_when_the_stashed_one_died(fake_stage_row) -> None:
    """A stashed job the service has since marked failed is not re-attachable."""
    fake_stage_row.store = {"in_flight_design_job_id": "rd-dead"}
    client = _reattach_client("rd-1")
    client.get_design_status.return_value = DesignJobStatus(job_id="rd-dead", status="failed")
    stage = WebsiteRedesignStage(client=client, session_factory=fake_stage_row)

    out = await stage.run({}, ctx=_ctx_with_critique())

    client.start_design.assert_awaited_once()
    assert out.redesign_job_id == "rd-1"
    assert fake_stage_row.store == {"in_flight_design_job_id": "rd-1"}


@pytest.mark.asyncio
async def test_a_fresh_stage_starts_a_job_normally_and_stashes_it(fake_stage_row) -> None:
    """(c) Nothing stashed => the ordinary first attempt, plus the pre-wait stash."""
    client = _reattach_client()
    stage = WebsiteRedesignStage(client=client, session_factory=fake_stage_row)

    out = await stage.run({}, ctx=_ctx_with_critique())

    client.start_design.assert_awaited_once()
    client.get_design_status.assert_not_awaited()  # nothing stashed => no probe
    assert out.redesign_job_id == "rd-1"
    assert fake_stage_row.store == {"in_flight_design_job_id": "rd-1"}


@pytest.mark.asyncio
async def test_no_session_factory_degrades_to_a_plain_submit() -> None:
    """Without the dispatcher dep the stage still runs — it just cannot re-attach."""
    client = _reattach_client()
    stage = WebsiteRedesignStage(client=client)

    out = await stage.run({}, ctx=_ctx_with_critique())

    assert out.redesign_job_id == "rd-1"
    client.get_design_status.assert_not_awaited()


def test_the_blueprint_wires_the_session_factory_into_the_stage() -> None:
    """The re-attach ships INERT if the blueprint keeps the bare ``handler(...)``
    factory: the stage would be constructed without a session factory and silently
    degrade to the old resubmit behaviour with every unit test still green."""
    from src.services.blueprints.website_review import WEBSITE_REVIEW_BLUEPRINT

    sentinel = object()
    stage = WEBSITE_REVIEW_BLUEPRINT.stage("website_redesign").handler_factory(
        {"session_factory": sentinel}
    )
    assert isinstance(stage, WebsiteRedesignStage)
    assert stage._sessions is sentinel
