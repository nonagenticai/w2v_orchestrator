"""UI review 2026-10-17 backend fixes — resume 409, cancel-on-delete, clone, run totals,
failed_stage, and auth/licence failures classified permanent.

Each test names the review finding it closes (``reports/ui-review-2026-10-17``).
"""

from __future__ import annotations

from uuid import UUID

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError
from sqlalchemy import select, update
from w2v_common.stages import StageRetriesExhaustedError

from src.database.models import Project, Run
from src.database.repositories.run_repository import RunRepository
from src.models.failure_class import EXTERNAL, failure_detail_for
from src.services.llm import in_process, vllm_direct
from src.services.stages.llm_budget import LLM_RETRY_ON
from src.services.stages.upstream import (
    PermanentUpstreamError,
    TransientUpstreamError,
    UpstreamAuthError,
    classify,
)

pytestmark = pytest.mark.asyncio


# ── helpers ─────────────────────────────────────────────────────────────────


async def _make_project(client, **overrides) -> tuple[str, str]:
    body: dict = {
        "type": "generic",
        "title": "ui review",
        "brief": {"angle": "growth", "target_duration_seconds": 30},
        "voice": {"descriptor": "crisp"},
        "deliverables": [
            {"name": "d", "format": "16x9", "captions_mode": "none", "target_duration_seconds": 30}
        ],
    }
    body.update(overrides)
    r = await client.post("/api/projects", json=body)
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    detail = await client.get(f"/api/projects/{pid}")
    return pid, detail.json()["deliverables"][0]["id"]


async def _run_row(db_session_factory, deliv: str, *, status: str, **values) -> UUID:
    """A run created DIRECTLY in the DB (no POST → no executor race), forced to ``status``."""
    async with db_session_factory() as s:
        run = await RunRepository(s).create(
            deliverable_id=UUID(deliv),
            blueprint_id="generic_v1",
            config_snapshot={},
            pause_after_stages=[],
        )
        await s.commit()
        await s.execute(update(Run).where(Run.id == run.id).values(status=status, **values))
        await s.commit()
        return run.id


async def _status(db_session_factory, run_id: UUID) -> str:
    async with db_session_factory() as s:
        return (await s.execute(select(Run.status).where(Run.id == run_id))).scalar_one()


# ── 1c. resume on a non-paused run ──────────────────────────────────────────


@pytest.mark.parametrize("status", ["failed"])
async def test_resume_non_paused_run_is_a_coded_409(client, db_session_factory, status) -> None:
    _, deliv = await _make_project(client)
    run_id = await _run_row(db_session_factory, deliv, status=status)

    r = await client.post(f"/api/runs/{run_id}/resume")

    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "run_not_resumable"
    assert detail["status"] == status
    assert "paused" in detail["message"]
    assert await _status(db_session_factory, run_id) == status, "a refused resume mutates nothing"


# ── 0 / 6. auth + licence failures are permanent ────────────────────────────


def _status_error(status: int, body: str) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://worker-management/api/workers/x/message")
    resp = httpx.Response(status, text=body, request=req)
    return httpx.HTTPStatusError("boom", request=req, response=resp)


_WM_WRAPPED_401 = (
    '{"error": {"code": "INTERNAL_ERROR", "status": 500, "message": "status_code: 401, '
    "model_name: local-llm, body: {'detail': 'Caller licence could not be introspected; "
    "caller is not authorised'}\"}}"
)


@pytest.mark.parametrize(
    "body",
    [_WM_WRAPPED_401],
)
async def test_wm_500_wrapping_an_auth_refusal_is_permanent(body: str) -> None:
    err = classify(_status_error(500, body))

    assert isinstance(err, UpstreamAuthError)
    assert isinstance(err, PermanentUpstreamError)
    assert not isinstance(err, LLM_RETRY_ON), "a permanent auth failure must not burn retries"
    assert err.failure_class == EXTERNAL  # the fault axis is unchanged: a 5xx dependency
    detail = failure_detail_for(err)
    assert detail is not None
    assert detail["code"] == "upstream_auth"
    assert detail["scope"] == "platform"


async def test_wm_500_connection_error_stays_transient() -> None:
    err = classify(_status_error(500, '{"error": {"message": "Connection error."}}'))
    assert isinstance(err, TransientUpstreamError)
    assert failure_detail_for(err) is None


async def test_a_500_that_merely_mentions_401_stays_transient() -> None:
    err = classify(_status_error(500, '{"error": {"message": "retry 401 of 900 failed"}}'))
    assert isinstance(err, TransientUpstreamError)


@pytest.mark.parametrize("status", [401])
async def test_direct_auth_status_carries_upstream_auth_detail(status: int) -> None:
    err = classify(_status_error(status, '{"detail": "nope"}'))
    assert isinstance(err, UpstreamAuthError)
    assert failure_detail_for(err) == {
        "code": "upstream_auth",
        "scope": "platform",
        "message": err.failure_detail["message"],
    }


async def test_failure_detail_survives_the_retry_wrapper_chain() -> None:
    inner = classify(_status_error(500, _WM_WRAPPED_401))
    wrapped = StageRetriesExhaustedError(stage_name="s", attempts=1, last_error=inner)
    wrapped.__cause__ = inner
    detail = failure_detail_for(wrapped)
    assert detail is not None and detail["code"] == "upstream_auth"


@pytest.mark.parametrize("status", [401])
async def test_vllm_direct_auth_is_permanent_platform_auth(status: int) -> None:
    err = vllm_direct._http_error(status, "denied", "local-llm")
    assert isinstance(err, UpstreamAuthError)
    assert err.failure_detail["scope"] == "platform"


async def test_tenant_byo_key_rejection_is_tenant_scoped_auth() -> None:
    exc = ModelHTTPError(status_code=401, model_name="gpt", body=None)
    err = in_process.map_error(exc, kind="openai", model="gpt", secret=None)
    assert isinstance(err, UpstreamAuthError)
    assert err.failure_detail["scope"] == "tenant"


# ── X3. clone_from_project_id ───────────────────────────────────────────────


async def test_create_clones_brand_voice_theme_and_explicit_fields_win(client) -> None:
    src_id, _ = await _make_project(
        client,
        brand={"primary_color": "#112233", "logo": "s3://logo"},
        voice={"descriptor": "warm", "speed": 1.1},
        theme={"font": "Inter", "mood": "calm"},
        brief={
            "angle": "trust",
            "audience": "CFOs",
            "target_url": "https://source.example",
            "target_duration_seconds": 30,
        },
    )
    r = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "clone",
            "brief": {"angle": "growth", "target_duration_seconds": 45},
            "brand": {"primary_color": "#ffffff"},
            "voice": {},
            "theme": {},
            "clone_from_project_id": src_id,
            "deliverables": [
                {"name": "d", "format": "16x9", "captions_mode": "none"},
            ],
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["brand"] == {"primary_color": "#ffffff", "logo": "s3://logo"}
    assert body["voice"] == {"descriptor": "warm", "speed": 1.1}
    assert body["theme"] == {"font": "Inter", "mood": "calm"}
    # Same type: brief STYLE defaults copied, request keys win, content bindings never.
    assert body["brief"]["angle"] == "growth"
    assert body["brief"]["audience"] == "CFOs"
    assert "target_url" not in body["brief"]


async def test_clone_from_another_tenant_is_a_coded_404(client, other_tenant_client) -> None:
    r = await other_tenant_client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "foreign",
            "brief": {"angle": "x", "target_duration_seconds": 30},
            "deliverables": [{"name": "d", "format": "16x9", "captions_mode": "none"}],
        },
    )
    assert r.status_code == 201, r.text
    foreign_id = r.json()["id"]

    clone = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "steal",
            "clone_from_project_id": foreign_id,
            "deliverables": [{"name": "d", "format": "16x9", "captions_mode": "none"}],
        },
    )
    assert clone.status_code == 404, clone.text
    assert clone.json()["detail"]["code"] == "clone_source_not_found"


async def test_draft_and_finalize_patch_apply_the_clone(client) -> None:
    src_id, _ = await _make_project(
        client, brand={"primary_color": "#abcdef"}, theme={"font": "Inter"}
    )
    draft = await client.post(
        "/api/projects/draft",
        json={"type": "generic", "title": "draft", "clone_from_project_id": src_id},
    )
    assert draft.status_code == 201, draft.text
    assert draft.json()["brand"] == {"primary_color": "#abcdef"}

    blank = await client.post("/api/projects/draft", json={"type": "generic", "title": "blank"})
    pid = blank.json()["id"]
    patch = await client.patch(
        f"/api/projects/{pid}",
        json={"brand": {"logo": "s3://mine"}, "clone_from_project_id": src_id},
    )
    assert patch.status_code == 200, patch.text
    got = (await client.get(f"/api/projects/{pid}")).json()
    assert got["brand"] == {"primary_color": "#abcdef", "logo": "s3://mine"}
    assert got["theme"] == {"font": "Inter"}


# ── 3. cancel on delete ─────────────────────────────────────────────────────


async def test_delete_cancels_every_active_run_including_paused(client, db_session_factory) -> None:
    pid, deliv = await _make_project(client)
    running = await _run_row(db_session_factory, deliv, status="running")
    paused = await _run_row(db_session_factory, deliv, status="paused")
    done = await _run_row(db_session_factory, deliv, status="completed")

    r = await client.delete(f"/api/projects/{pid}")

    assert r.status_code == 204, r.text
    assert await _status(db_session_factory, running) == "cancelled"
    assert await _status(db_session_factory, paused) == "cancelled"
    assert await _status(db_session_factory, done) == "completed"
    async with db_session_factory() as s:
        status = (await s.execute(select(Project.status).where(Project.id == UUID(pid)))).scalar()
    assert status == "archived"


async def test_delete_degraded_records_cancel_and_does_not_archive(
    client, app, db_session_factory
) -> None:
    pid, deliv = await _make_project(client)
    running = await _run_row(db_session_factory, deliv, status="running")
    app.state.dispatcher = None

    r = await client.delete(f"/api/projects/{pid}")

    assert r.status_code == 503, r.text
    assert await _status(db_session_factory, running) == "cancelled"
    async with db_session_factory() as s:
        status = (await s.execute(select(Project.status).where(Project.id == UUID(pid)))).scalar()
    assert status != "archived", "a retryable 503 must not half-delete"


# ── M6. run list total ──────────────────────────────────────────────────────


async def test_project_runs_list_reports_the_total_beyond_the_page(
    client, db_session_factory
) -> None:
    pid, deliv = await _make_project(client)
    for _ in range(3):
        await _run_row(db_session_factory, deliv, status="completed")

    r = await client.get(f"/api/projects/{pid}/runs?limit=2")

    assert r.status_code == 200, r.text
    assert len(r.json()) == 2
    assert r.headers["x-total-count"] == "3"


# ── X7. failed_stage on the run detail ──────────────────────────────────────


async def test_run_detail_exposes_failed_stage(client, db_session_factory) -> None:
    _, deliv = await _make_project(client)
    failed = await _run_row(
        db_session_factory, deliv, status="failed", current_stage="script_draft", error="x"
    )
    ok = await _run_row(db_session_factory, deliv, status="paused", current_stage="script_draft")

    assert (await client.get(f"/api/runs/{failed}")).json()["failed_stage"] == "script_draft"
    assert (await client.get(f"/api/runs/{ok}")).json()["failed_stage"] is None
