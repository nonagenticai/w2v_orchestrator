"""web_design_v2 stage handlers: brand_extract, design, verify, assemble."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from w2v_common.stages.errors import StageRetriesExhaustedError, StageTimeoutError

from src.clients.schemas.design import DesignJobResult, DesignJobStatus, DesignVisualAsset
from src.services.runs.run_context import RunContext
from src.services.stages.base import StageError
from src.services.stages.web_design_v2.assemble import WebDesignAssembleStage
from src.services.stages.web_design_v2.brand_extract import BrandExtractStage
from src.services.stages.web_design_v2.design import _WAIT_RETRIES, WebDesignV2Stage
from src.services.stages.web_design_v2.verify import WebDesignVerifyStage


def _ctx(brief: dict, prior: dict | None = None, brand: dict | None = None) -> RunContext:
    snap: dict = {"brief": brief, "title": "Acme"}
    if brand is not None:
        snap["brand"] = brand
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=snap,
        prior_run_outputs=prior or {},
    )


# --------------------------- brand_extract --------------------------------- #


@pytest.mark.asyncio
async def test_brand_extract_from_url_merges_project_over_url(monkeypatch) -> None:
    async def _fake(url, **kw):
        return ({"primary_color": "#aaa", "name": "SiteName"}, {"tone": "bold"})

    monkeypatch.setattr(
        "src.services.stages.web_design_v2.brand_extract.resolve_brand_context", _fake
    )
    out = await BrandExtractStage().run(
        {}, ctx=_ctx({"target_url": "https://x.example"}, brand={"primary_color": "#111"})
    )
    assert out.brand is not None
    assert out.brand["primary_color"] == "#111"  # project wins
    assert out.brand["name"] == "SiteName"  # url fills gap
    assert out.site_analysis == {"tone": "bold"}
    assert out.source == "url"


# ------------------------------- design ------------------------------------ #


def _design_client() -> AsyncMock:
    c = AsyncMock()
    c.start_design.return_value = "wd2-1"
    c.wait_for_design.return_value = DesignJobStatus(
        job_id="wd2-1", status="completed", accessibility_score=95.0
    )
    c.get_design_result.return_value = DesignJobResult(
        job_id="wd2-1",
        url="",
        scope="landing",
        style="assembly",
        redesigned_html_path="jobs/wd2-1/redesign.html",
        verification_status="passed",
        verification_report={"status": "passed", "findings": []},
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


@pytest.mark.asyncio
async def test_design_runs_assembly_and_passes_verification() -> None:
    client = _design_client()
    out = await WebDesignV2Stage(client=client).run(
        {},
        ctx=_ctx(
            {"description": "A bakery landing page", "interactivity": "interactive"},
            prior={"brand_extract": {"brand": {"primary_color": "#e11d48"}}},
        ),
    )
    assert out.redesign_job_id == "wd2-1"
    assert out.live_html_url == "/api/redesign/jobs/wd2-1/html"
    assert out.verification_status == "passed"
    assert out.interactivity == "interactive"
    # assembly mode + brand forwarded to the service
    kwargs = client.start_design.call_args.kwargs
    assert kwargs["mode"] == "assembly"
    assert kwargs["interactivity"] == "interactive"
    # brand_extract had no name and the brief names no business → no name is sent
    # (the project title "Acme" is NOT a business name; html_render's
    # canonical_brand adopts one name for the whole page)
    assert kwargs["brand"] == {"primary_color": "#e11d48"}
    assert kwargs["scope"] == "landing"
    # embeds must be a list even when the brief has none — html_render rejects a
    # JSON null for its `list[EmbedSpec]` field (422 list_type), so an embed-less
    # run must send [] (== absent), never None. Regression guard for the live 422.
    assert kwargs["embeds"] == []


@pytest.mark.asyncio
async def test_design_requires_description() -> None:
    with pytest.raises(StageError):
        await WebDesignV2Stage(client=_design_client()).run({}, ctx=_ctx({}))


# ------------------------------- verify ------------------------------------ #


@pytest.mark.asyncio
async def test_verify_fails_loud_on_failed_report() -> None:
    with pytest.raises(StageError, match="failed verification"):
        await WebDesignVerifyStage().run(
            {},
            ctx=_ctx(
                {},
                prior={
                    "design": {
                        "verification_status": "failed",
                        "verification_report": {
                            "status": "failed",
                            "findings": [
                                {"severity": "hard", "check": "console", "message": "JS error"}
                            ],
                        },
                    }
                },
            ),
        )


@pytest.mark.asyncio
async def test_verify_fails_loud_when_verification_absent() -> None:
    """The assembly verify lane silently didn't run (status absent) → refuse to
    complete an unverified design (default W2V_WEB_DESIGN_REQUIRE_VERIFICATION)."""
    with pytest.raises(StageError, match="no verification result"):
        await WebDesignVerifyStage().run(
            {},
            ctx=_ctx({}, prior={"design": {"redesign_job_id": "j1"}}),  # no verification_status
        )


@pytest.mark.asyncio
async def test_verify_fails_loud_when_verification_skipped() -> None:
    with pytest.raises(StageError, match="no verification result"):
        await WebDesignVerifyStage().run(
            {},
            ctx=_ctx({}, prior={"design": {"verification_status": "skipped"}}),
        )


# ------------------------------ assemble ----------------------------------- #


@pytest.mark.asyncio
async def test_assemble_requires_job_id() -> None:
    with pytest.raises(StageError):
        await WebDesignAssembleStage().run({}, ctx=_ctx({}, prior={"design": {}}))


@pytest.mark.asyncio
async def test_design_wait_fails_immediately_on_terminal_job_failure() -> None:
    """A design job the service already marked `failed` (e.g. verification failed, or
    the WM worker blew its token limit) fails the stage on poll 1 with the service's
    own error text — retry_on=() stops the dead job being re-polled 4 times."""
    client = _design_client()
    terminal = RuntimeError("design job wd2-1 failed: WM worker returned 500: token limit")
    client.wait_for_design.side_effect = terminal
    stage = WebDesignV2Stage(client=client)

    with pytest.raises(RuntimeError, match="token limit") as exc_info:
        await stage.run({}, ctx=_ctx({"description": "A bakery landing page"}))

    assert not isinstance(exc_info.value, StageRetriesExhaustedError)
    assert exc_info.value is terminal
    assert client.wait_for_design.await_count == 1
    client.get_design_result.assert_not_awaited()


@pytest.mark.asyncio
async def test_design_wait_still_retries_on_poll_timeout() -> None:
    """retry_on=() must not eat the poll-timeout headroom: a still-running job keeps
    its full 1 + _WAIT_RETRIES poll budget (stage_call retries TimeoutError anyway)."""
    client = _design_client()
    client.wait_for_design.side_effect = TimeoutError
    stage = WebDesignV2Stage(client=client)

    with (
        patch("w2v_common.stages.execution.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(StageTimeoutError),
    ):
        await stage.run({}, ctx=_ctx({"description": "A bakery landing page"}))

    assert client.wait_for_design.await_count == 1 + _WAIT_RETRIES == 4


# --------- design: resume must RE-ATTACH, never resubmit --------------------- #
# Same mechanism, same reasoning and the same shared helper as website_redesign
# (stages/shared/design_job_stash.py): html_render's POST /api/designs has no
# idempotency key, so a poll-budget expiry that re-enters handle() from the top would
# orphan a healthy render and pay for a second one.


class _FakeSessions:
    def __init__(self, store: dict | None = None) -> None:
        self.store = store

    def __call__(self) -> _FakeSessions:
        return self

    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *exc: object) -> bool:
        return False


@pytest.fixture
def v2_stage_row(monkeypatch):
    sessions = _FakeSessions()

    class _Row:
        def __init__(self, output: dict | None) -> None:
            self.output = output

    class _Repo:
        def __init__(self, _session: object) -> None:
            pass

        async def get(self, _run_id, _stage_name):
            return _Row(sessions.store) if sessions.store is not None else None

        async def record_progress(self, _run_id, _stage_name, output: dict) -> None:
            sessions.store = dict(output)

    monkeypatch.setattr("src.services.stages.shared.design_job_stash.StageOutputRepository", _Repo)
    monkeypatch.setattr(
        "src.services.stages.shared.design_job_stash.bind_session_tenant",
        AsyncMock(return_value=None),
    )
    return sessions


@pytest.mark.asyncio
async def test_v2_design_resume_reattaches_instead_of_resubmitting(v2_stage_row) -> None:
    client = _design_client()
    client.wait_for_design.side_effect = TimeoutError
    client.get_design_status.return_value = DesignJobStatus(job_id="wd2-1", status="processing")
    stage = WebDesignV2Stage(client=client, session_factory=v2_stage_row)
    ctx = _ctx({"description": "An acme landing page"})

    with (
        patch("w2v_common.stages.execution.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(StageTimeoutError),
    ):
        await stage.run({}, ctx=ctx)
    assert client.start_design.await_count == 1
    assert v2_stage_row.store == {"in_flight_design_job_id": "wd2-1"}

    client.wait_for_design.side_effect = None
    out = await stage.run({}, ctx=ctx)

    assert client.start_design.await_count == 1, "attempt 2 submitted a SECOND design job"
    assert out.redesign_job_id == "wd2-1"


@pytest.mark.asyncio
async def test_v2_design_fresh_attempt_submits_and_stashes(v2_stage_row) -> None:
    client = _design_client()
    stage = WebDesignV2Stage(client=client, session_factory=v2_stage_row)

    out = await stage.run({}, ctx=_ctx({"description": "An acme landing page"}))

    client.start_design.assert_awaited_once()
    assert out.redesign_job_id == "wd2-1"
    assert v2_stage_row.store == {"in_flight_design_job_id": "wd2-1"}


def test_v2_blueprint_wires_the_session_factory_into_the_design_stage() -> None:
    """Without this the re-attach ships INERT: the stage is built with no session
    factory and silently degrades to the old resubmit behaviour, every test green."""
    from src.services.blueprints.web_design_v2 import WEB_DESIGN_V2_BLUEPRINT

    sentinel = object()
    stage = WEB_DESIGN_V2_BLUEPRINT.stage("design").handler_factory({"session_factory": sentinel})
    assert isinstance(stage, WebDesignV2Stage)
    assert stage._sessions is sentinel


# P6: the tenant's default brand is merged UNDER ``snapshot.brand`` at run creation.
# It must stay a gap-filler: a brief-authored brand and the reference site's
# extracted brand both outrank it (explicit run/brief > tenant default).
_TENANT_BRAND = {"primary_color": "#aaaaaa", "font_family": "TenantSans"}


def _tenant_ctx(brief: dict) -> RunContext:
    ctx = _ctx(brief, brand=dict(_TENANT_BRAND))
    ctx.config_snapshot["tenant_defaults"] = {"version": 1, "brand": dict(_TENANT_BRAND)}
    return ctx


@pytest.mark.asyncio
async def test_brand_extract_brief_brand_outranks_tenant_default() -> None:
    out = await BrandExtractStage().run(
        {}, ctx=_tenant_ctx({"brand": {"primary_color": "#111111"}})
    )
    assert out.brand is not None
    assert out.brand["primary_color"] == "#111111"  # brief wins
    assert out.brand["font_family"] == "TenantSans"  # tenant fills the gap


@pytest.mark.asyncio
async def test_brand_extract_url_brand_outranks_tenant_default(monkeypatch) -> None:
    async def _fake(url: str):
        return {"primary_color": "#222222"}, {"tone": "x"}

    monkeypatch.setattr(
        "src.services.stages.web_design_v2.brand_extract.resolve_brand_context", _fake
    )
    out = await BrandExtractStage().run({}, ctx=_tenant_ctx({"target_url": "https://x.example"}))
    assert out.brand is not None
    assert out.brand["primary_color"] == "#222222"  # the site's own palette wins
    assert out.brand["font_family"] == "TenantSans"
    assert out.source == "url"


@pytest.mark.asyncio
async def test_brand_extract_tenant_default_alone_is_used() -> None:
    out = await BrandExtractStage().run({}, ctx=_tenant_ctx({}))
    assert out.brand == _TENANT_BRAND
    assert out.source == "tenant"


# No target_url ⇒ brand_extract yields no name; html_render then invented one
# (run 01a156a6: "Hearth & Crumb" header, "The Daily Crumb" footer). The design
# stage must send the name the operator wrote in a brief field — never the project
# title, which labels the PROJECT, not the business; with no name, html_render's
# canonical_brand adopts ONE invented name for the whole page.


async def _sent_brand(brief: dict, prior_brand: dict | None, title: str | None = "Acme"):
    client = _design_client()
    ctx = _ctx(
        {"description": "A bakery landing page", **brief},
        prior={"brand_extract": {"brand": prior_brand}},
    )
    if title is None:
        ctx.config_snapshot.pop("title")
    else:
        ctx.config_snapshot["title"] = title
    await WebDesignV2Stage(client=client).run({}, ctx=ctx)
    return client.start_design.call_args.kwargs["brand"]


@pytest.mark.asyncio
async def test_design_never_sends_the_project_title_as_the_brand() -> None:
    assert await _sent_brand({}, None, title="Bakery landing page") is None
    sent = await _sent_brand({}, {"primary_color": "#111"}, title="Bakery landing page")
    assert sent == {"primary_color": "#111"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "brief",
    [{"brand_name": "Crumb & Co"}],
)
async def test_design_structured_brief_name_outranks_title(brief: dict) -> None:
    assert await _sent_brand(brief, None) == {"name": "Crumb & Co"}


@pytest.mark.asyncio
async def test_design_extracted_brand_name_wins() -> None:
    sent = await _sent_brand({"brand_name": "Brief"}, {"name": "SiteName", "primary_color": "#111"})
    assert sent == {"name": "SiteName", "primary_color": "#111"}


@pytest.mark.asyncio
async def test_design_no_name_anywhere_sends_null_brand() -> None:
    assert await _sent_brand({}, None, title="  ") is None
    assert await _sent_brand({}, None, title=None) is None
