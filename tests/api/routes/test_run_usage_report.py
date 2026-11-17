"""``POST /api/runs/{run_id}/usage`` — service-reported platform-vLLM usage.

Real postgres. Pins: a report lands as one ``local_gpu`` / ``service_reported`` row,
a retried ``call_id`` is a no-op (``recorded=false``, still one row), a run of
another tenant (or no run at all) is a 404 that writes nothing, the body is strict,
the row is durable before the response is built, and run detail's ``usage_total``
sums the reported tokens.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4, uuid7

import pytest
from sqlalchemy import select, text
from starlette.requests import Request

from src.api.routes.runs.usage import report_run_usage
from src.api.schemas import RunUsageReport
from src.database.models import Deliverable, Project, Run, RunUsage
from src.database.repositories.run_usage_repository import SERVICE_REPORTED_USAGE_SOURCE

pytestmark = pytest.mark.asyncio


def _body(call_id: str = "call-1", **over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "call_id": call_id,
        "stage": "analysis_summary",
        "model": "local-llm",
        "provider": "vllm",
        "input_tokens": 120,
        "output_tokens": 30,
        "total_tokens": 150,
        "service": "analysis-service",
    }
    body.update(over)
    return body


async def _run_for(db_session_factory: Any, tenant_id: str | UUID) -> UUID:
    """A run under a project owned by ``tenant_id``, created directly in the DB."""
    async with db_session_factory() as s:
        project = Project(
            id=uuid7(),
            tenant_id=UUID(str(tenant_id)),
            owner_subject_id=uuid7(),
            visibility="private",
            type="generic",
            title="usage-report",
        )
        deliverable = Deliverable(id=uuid7(), project_id=project.id, name="d", format="16x9")
        run = Run(
            id=uuid7(),
            deliverable_id=deliverable.id,
            blueprint_id="generic_v1",
            status="running",
            config_snapshot={},
            pause_after_stages=[],
        )
        s.add(project)
        await s.flush()
        s.add(deliverable)
        await s.flush()
        s.add(run)
        await s.commit()
        return run.id


async def _rows(db_session_factory: Any, run_id: UUID) -> list[RunUsage]:
    async with db_session_factory() as s:
        return list(
            (await s.execute(select(RunUsage).where(RunUsage.run_id == run_id))).scalars().all()
        )


async def test_report_writes_one_local_gpu_row(client: Any, db_session_factory: Any) -> None:
    run_id = await _run_for(db_session_factory, client.tenant_id)

    resp = await client.post(f"/api/runs/{run_id}/usage", json=_body())

    assert resp.status_code == 202, resp.text
    assert resp.json() == {"recorded": True}
    (row,) = await _rows(db_session_factory, run_id)
    assert (row.stage_name, row.model, row.provider) == ("analysis_summary", "local-llm", "vllm")
    assert (row.input_tokens, row.output_tokens) == (120, 30)
    assert row.cost_usd == 0
    assert row.funding == "local_gpu"
    assert row.slot is None and row.connection_id is None
    assert row.unit_kind == "tokens" and row.units == 150
    assert row.subject_id == UUID(client.subject_id)
    assert row.metadata_ is not None
    assert row.metadata_["source"] == SERVICE_REPORTED_USAGE_SOURCE
    assert row.metadata_["call_id"] == "call-1"
    assert row.metadata_["service"] == "analysis-service"


async def test_stage_falls_back_to_the_service_name(client: Any, db_session_factory: Any) -> None:
    run_id = await _run_for(db_session_factory, client.tenant_id)
    resp = await client.post(f"/api/runs/{run_id}/usage", json=_body(stage=None))
    assert resp.status_code == 202, resp.text
    (row,) = await _rows(db_session_factory, run_id)
    assert row.stage_name == "analysis-service"


async def test_duplicate_call_id_is_not_recorded_twice(
    client: Any, db_session_factory: Any
) -> None:
    run_id = await _run_for(db_session_factory, client.tenant_id)

    first = await client.post(f"/api/runs/{run_id}/usage", json=_body("dup"))
    retry = await client.post(f"/api/runs/{run_id}/usage", json=_body("dup", input_tokens=999))
    other = await client.post(f"/api/runs/{run_id}/usage", json=_body("other"))

    assert (first.status_code, first.json()) == (202, {"recorded": True})
    assert (retry.status_code, retry.json()) == (202, {"recorded": False})
    assert (other.status_code, other.json()) == (202, {"recorded": True})
    rows = await _rows(db_session_factory, run_id)
    assert sorted(str((r.metadata_ or {})["call_id"]) for r in rows) == ["dup", "other"]
    assert {r.input_tokens for r in rows} == {120}, "the retry must not overwrite the first"


async def test_same_call_id_on_another_run_is_its_own_row(
    client: Any, db_session_factory: Any
) -> None:
    run_a = await _run_for(db_session_factory, client.tenant_id)
    run_b = await _run_for(db_session_factory, client.tenant_id)
    assert (await client.post(f"/api/runs/{run_a}/usage", json=_body("x"))).json()["recorded"]
    assert (await client.post(f"/api/runs/{run_b}/usage", json=_body("x"))).json()["recorded"]


async def test_other_tenants_run_is_404_and_writes_nothing(
    client: Any, db_session_factory: Any
) -> None:
    foreign_run = await _run_for(db_session_factory, uuid4())

    resp = await client.post(f"/api/runs/{foreign_run}/usage", json=_body())

    assert resp.status_code == 404, resp.text
    assert await _rows(db_session_factory, foreign_run) == []


async def test_unknown_run_is_404(client: Any, db_session_factory: Any) -> None:
    ghost = uuid7()
    resp = await client.post(f"/api/runs/{ghost}/usage", json=_body())
    assert resp.status_code == 404, resp.text
    assert await _rows(db_session_factory, ghost) == []


@pytest.mark.parametrize(
    "bad",
    [{"unexpected": 1}],
)
async def test_body_is_strict(client: Any, db_session_factory: Any, bad: dict[str, Any]) -> None:
    run_id = await _run_for(db_session_factory, client.tenant_id)
    resp = await client.post(f"/api/runs/{run_id}/usage", json=_body(**bad))
    assert resp.status_code == 422, resp.text
    assert await _rows(db_session_factory, run_id) == []


async def test_missing_required_field_is_422(client: Any, db_session_factory: Any) -> None:
    run_id = await _run_for(db_session_factory, client.tenant_id)
    body = _body()
    del body["service"]
    assert (await client.post(f"/api/runs/{run_id}/usage", json=body)).status_code == 422


async def test_report_is_committed_before_the_response_is_built(db_session_factory: Any) -> None:
    """Called DIRECTLY and read through an independent connection (the test client
    commits in teardown, so it could not catch a flush-only handler)."""
    tenant_id, subject_id = uuid4(), uuid4()
    run_id = await _run_for(db_session_factory, tenant_id)
    request = cast(
        "Request", SimpleNamespace(headers={}, state=SimpleNamespace(subject_id=str(subject_id)))
    )

    async with db_session_factory() as session:
        result = await report_run_usage(
            run_id=run_id,
            body=RunUsageReport.model_validate(_body("durable")),
            request=request,
            session=session,
            tenant_id=tenant_id,
        )
        assert result.recorded is True
        async with db_session_factory() as probe:
            found = (
                await probe.execute(
                    text("select 1 from run_usage where run_id = :r"), {"r": str(run_id)}
                )
            ).first()
        assert found is not None, "the reported row is not durable at response-build time"


async def test_run_detail_usage_total_sums_reported_rows(
    client: Any, app: Any, db_session_factory: Any
) -> None:
    from src.api.routes._common import get_billing_client

    class _NoBilling:
        async def costs_by_correlation(self, tenant_id, correlation_ids, *, group_by=None):
            return {"groups": []}

    app.dependency_overrides[get_billing_client] = _NoBilling
    try:
        run_id = await _run_for(db_session_factory, client.tenant_id)
        await client.post(f"/api/runs/{run_id}/usage", json=_body("a"))
        await client.post(
            f"/api/runs/{run_id}/usage",
            json=_body("b", stage="script", input_tokens=10, output_tokens=5, total_tokens=15),
        )

        resp = await client.get(f"/api/runs/{run_id}")

        assert resp.status_code == 200, resp.text
        usage = resp.json()["usage_total"]
        assert (usage["input_tokens"], usage["output_tokens"]) == (130, 35)
        assert usage["cost_usd"] == 0
        stages = {r["stage_name"]: r for r in usage["by_stage"]}
        assert set(stages) == {"analysis_summary", "script"}
        assert stages["script"]["provider"] == "vllm"
    finally:
        app.dependency_overrides.pop(get_billing_client, None)
