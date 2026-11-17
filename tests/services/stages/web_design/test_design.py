"""web_design stage — greenfield HTML generation from a brief."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from w2v_common.stages.errors import StageRetriesExhaustedError, StageTimeoutError

from src.clients.schemas.design import DesignJobResult, DesignJobStatus, DesignVisualAsset
from src.services.blueprints.registry import get_registry
from src.services.blueprints.web_design import WEB_DESIGN_BLUEPRINT
from src.services.runs.run_context import RunContext
from src.services.stages.base import StageError
from src.services.stages.web_design.design import _WAIT_RETRIES, WebDesignStage


def _ctx(brief: dict) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": brief, "title": "Acme"},
        prior_run_outputs={},
    )


def _client() -> AsyncMock:
    c = AsyncMock()
    c.start_design.return_value = "wd-1"
    c.wait_for_design.return_value = DesignJobStatus(
        job_id="wd-1", status="completed", accessibility_score=92.0
    )
    c.get_design_result.return_value = DesignJobResult(
        job_id="wd-1",
        url="",
        scope="landing",
        style="bold",
        redesigned_html_path="jobs/wd-1/index.html",
        visual_assets=[
            DesignVisualAsset(
                asset_id="a1",
                title="Hero",
                kind="screenshot",
                role="after",
                url="https://cdn/a.png",
                source="design",
            )
        ],
    )
    return c


def test_web_design_registered() -> None:
    assert get_registry().resolve("web_design") is WEB_DESIGN_BLUEPRINT


@pytest.mark.asyncio
async def test_greenfield_landing_from_brief() -> None:
    client = _client()
    stage = WebDesignStage(client=client)
    out = await stage.run(
        {},
        ctx=_ctx(
            {
                "description": "A landing page for a bakery",
                "page_type": "landing",
                "redesign_style": "bold",
            }
        ),
    )
    assert out.redesign_job_id == "wd-1"
    assert out.redesigned_html_path == "jobs/wd-1/index.html"
    assert out.visuals  # screenshot/visual surfaced for the run-view
    _, kwargs = client.start_design.call_args
    assert kwargs["scope"] == "landing"  # greenfield (no reference URL)
    assert kwargs["url"] == ""
    assert kwargs["user_context"]["description"] == "A landing page for a bakery"
    assert kwargs["page_type"] == "landing"


@pytest.mark.asyncio
async def test_requires_description() -> None:
    stage = WebDesignStage(client=_client())
    with pytest.raises(StageError, match="description"):
        await stage.run({}, ctx=_ctx({"page_type": "landing"}))


@pytest.mark.asyncio
async def test_wait_fails_immediately_on_terminal_job_failure() -> None:
    """A design job the service already marked `failed` fails the stage on poll 1 with
    the service's own error text (retry_on=()) — no re-polling of a dead job."""
    client = _client()
    terminal = RuntimeError("design job wd-1 failed: WM worker returned 500: token limit")
    client.wait_for_design.side_effect = terminal
    stage = WebDesignStage(client=client)

    with pytest.raises(RuntimeError, match="token limit") as exc_info:
        await stage.run({}, ctx=_ctx({"description": "A bakery landing page"}))

    assert not isinstance(exc_info.value, StageRetriesExhaustedError)
    assert exc_info.value is terminal
    assert client.wait_for_design.await_count == 1
    client.get_design_result.assert_not_awaited()


@pytest.mark.asyncio
async def test_wait_still_retries_on_poll_timeout() -> None:
    """retry_on=() must not eat the poll-timeout headroom: a still-running job keeps
    its full 1 + _WAIT_RETRIES poll budget (stage_call retries TimeoutError anyway)."""
    client = _client()
    client.wait_for_design.side_effect = TimeoutError
    stage = WebDesignStage(client=client)

    with (
        patch("w2v_common.stages.execution.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(StageTimeoutError),
    ):
        await stage.run({}, ctx=_ctx({"description": "A bakery landing page"}))

    assert client.wait_for_design.await_count == 1 + _WAIT_RETRIES == 4


# --- resume must RE-ATTACH, never resubmit -------------------------------------
# Live 2026-08-10: html_render logged 6 distinct design job ids for 3 submits and
# 253 WM LLM calls, because each poll-deadline expiry reset the stage to 'pending'
# and re-entered handle() from the top with a brand-new submit. POST /api/designs
# has no idempotency key, so the de-dup lives in the stage.


class _FakeStageOutputRow:
    def __init__(self, output: dict | None) -> None:
        self.output = output


class _FakeSessions:
    """Minimal stand-in for the dispatcher's session_factory + stage_outputs row.

    ``store`` IS the row's ``output`` JSONB, so a write in one ``async with``
    block is visible to the next — exactly what ``record_progress`` gives a stage
    across a resumable-timeout re-dispatch (``mark_pending`` leaves ``output``).
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
    """Patch StageOutputRepository as the stage imports it, backed by a dict."""
    sessions = _FakeSessions()

    class _Repo:
        def __init__(self, _session: object) -> None:
            pass

        async def get(self, _run_id, _stage_name) -> _FakeStageOutputRow | None:
            return _FakeStageOutputRow(sessions.store) if sessions.store is not None else None

        async def record_progress(self, _run_id, _stage_name, output: dict) -> None:
            sessions.store = dict(output)

    # The stash moved to the shared helper (stages/shared/design_job_stash.py) so
    # website_redesign and web_design_v2 use the SAME implementation; patch it there.
    monkeypatch.setattr("src.services.stages.shared.design_job_stash.StageOutputRepository", _Repo)
    monkeypatch.setattr(
        "src.services.stages.shared.design_job_stash.bind_session_tenant",
        AsyncMock(return_value=None),
    )
    return sessions


@pytest.mark.asyncio
async def test_resume_reattaches_to_in_flight_job_instead_of_resubmitting(fake_stage_row) -> None:
    """The whole point: attempt 2 of a timed-out stage must NOT submit a 2nd job."""
    client = _client()
    client.wait_for_design.side_effect = TimeoutError
    client.get_design_status.return_value = DesignJobStatus(job_id="wd-1", status="processing")
    stage = WebDesignStage(client=client, session_factory=fake_stage_row)
    ctx = _ctx({"description": "A bakery landing page"})

    # Attempt 1: submits, polls, times out — and stashes the job id.
    with (
        patch("w2v_common.stages.execution.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(StageTimeoutError),
    ):
        await stage.run({}, ctx=ctx)
    assert client.start_design.await_count == 1
    assert fake_stage_row.store == {"in_flight_design_job_id": "wd-1"}

    # Attempt 2 (the resumable-timeout re-dispatch): the job finishes this time.
    client.wait_for_design.side_effect = None
    out = await stage.run({}, ctx=ctx)

    assert client.start_design.await_count == 1  # NOT resubmitted
    assert out.redesign_job_id == "wd-1"


@pytest.mark.asyncio
async def test_resume_submits_fresh_job_when_the_stashed_one_died(fake_stage_row) -> None:
    """A stashed job the service has since marked failed is not re-attachable."""
    fake_stage_row.store = {"in_flight_design_job_id": "wd-dead"}
    client = _client()
    client.get_design_status.return_value = DesignJobStatus(job_id="wd-dead", status="failed")
    stage = WebDesignStage(client=client, session_factory=fake_stage_row)

    out = await stage.run({}, ctx=_ctx({"description": "A bakery landing page"}))

    client.start_design.assert_awaited_once()
    assert out.redesign_job_id == "wd-1"
    assert fake_stage_row.store == {"in_flight_design_job_id": "wd-1"}


@pytest.mark.asyncio
async def test_timeout_message_names_the_last_seen_job_status() -> None:
    """ "queued 590s" and "rendering 590s" were indistinguishable — now they aren't."""
    client = _client()
    client.wait_for_design.side_effect = TimeoutError
    client.get_design_status.return_value = DesignJobStatus(job_id="wd-1", status="queued")
    stage = WebDesignStage(client=client)

    with (
        patch("w2v_common.stages.execution.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(StageTimeoutError) as exc_info,
    ):
        await stage.run({}, ctx=_ctx({"description": "A bakery landing page"}))

    assert "last-seen job status: queued" in str(exc_info.value)
    assert "wd-1" in str(exc_info.value)


@pytest.mark.asyncio
async def test_timeout_message_survives_a_failing_status_probe() -> None:
    """The diagnostic must never turn a resumable timeout into a different failure."""
    client = _client()
    client.wait_for_design.side_effect = TimeoutError
    client.get_design_status.side_effect = RuntimeError("html_render unreachable")
    stage = WebDesignStage(client=client)

    with (
        patch("w2v_common.stages.execution.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(StageTimeoutError) as exc_info,
    ):
        await stage.run({}, ctx=_ctx({"description": "A bakery landing page"}))

    assert "status probe failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_output_carries_live_html_url_for_the_render_job() -> None:
    """v1 must publish a SERVED ref for the generated HTML, not only a screenshot.

    Without it `brand_artifact.critique_score` stays NULL platform-wide: the consumer
    has only `redesigned_html_path` (an internal MinIO key it cannot fetch) and never
    reads the HTML it is supposed to critique. Same shape as web_design_v2 —
    the orchestrator redesign proxy at `/api/redesign/jobs/{job_id}/html`.
    """
    client = _client()
    stage = WebDesignStage(client=client)
    ctx = _ctx({"description": "A landing page for a bakery"})
    out = await stage.run({}, ctx=ctx)

    assert out.live_html_url == "/api/redesign/jobs/wd-1/html"
    # The id in the URL is the html_render DESIGN job id (what redesigned_html_path
    # was minted under), NOT the orchestrator run/deliverable uuid — using the latter
    # is exactly what 404s at the proxy.
    assert out.live_html_url.endswith(f"/{out.redesign_job_id}/html")
    assert str(ctx.run_id) not in out.live_html_url
    assert str(ctx.deliverable_id) not in out.live_html_url
