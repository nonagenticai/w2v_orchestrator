"""P3 WP-LLM against REAL postgres: tenant connection → in-process call → run_usage.

* end to end: a real (encrypted) tenant ``anthropic`` connection pinned on a run;
  ``WMClient.message`` resolves exactly that version, decrypts the key, calls the
  (fake) provider with it, and writes one ``tenant_byo`` row into ``run_usage``;
* the billing mirror (``replace_for_run``) must not erase those rows, and a run whose
  only rows are in-process ones still counts as "missing usage" for the sweep.
"""

from __future__ import annotations

import base64
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid7

import pytest
from sqlalchemy import select, update
from w2v_common.auth.identity_context import set_correlation_context, set_identity_context

from src.auth.run_providers import bind_run_provider_slots, clear_run_provider_slots
from src.database.models import Deliverable, Project, Run, RunUsage
from src.database.repositories.admin_repository import AdminRepository
from src.database.repositories.connections_repository import ConnectionsRepository
from src.database.repositories.run_usage_repository import (
    IN_PROCESS_USAGE_SOURCE,
    SERVICE_REPORTED_USAGE_SOURCE,
    RunUsageRepository,
)
from src.services.llm import WMClient, in_process
from src.services.providers.crypto import Keyring
from tests.services.llm._fakes import SECRET, FakeProvider, anthropic_reply

pytestmark = pytest.mark.asyncio

KEY = os.urandom(32)


@pytest.fixture
def _db_sessions(monkeypatch: pytest.MonkeyPatch, db_session_factory: Any) -> None:
    """Point the production ``get_db_session`` at the test engine."""

    @asynccontextmanager
    async def _session() -> AsyncIterator[Any]:
        async with db_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    monkeypatch.setattr("src.database.session.get_db_session", _session)
    monkeypatch.setattr("src.services.llm.metering.get_db_session", _session)
    monkeypatch.setenv("PROVIDER_SECRET_KEYS", f"k1:{base64.b64encode(KEY).decode()}")
    from src.config import clear_settings_cache

    clear_settings_cache()


async def _tenant_run(session: Any) -> tuple[UUID, UUID]:
    tenant = uuid7()
    assert await AdminRepository(session).ensure_tenant(tenant, created_via="super_admin")
    project = Project(
        id=uuid7(),
        tenant_id=tenant,
        owner_subject_id=uuid7(),
        visibility="private",
        type="promo",
        title="p",
    )
    deliverable = Deliverable(id=uuid7(), project_id=project.id, name="d", format="16x9")
    run = Run(id=uuid7(), deliverable_id=deliverable.id, blueprint_id="promo_v1", status="running")
    session.add_all([project, deliverable, run])
    await session.commit()
    return tenant, run.id


@pytest.fixture(autouse=True)
def _unbind() -> Any:
    clear_run_provider_slots()
    yield
    clear_run_provider_slots()
    set_correlation_context(None, None)


async def test_tenant_call_resolves_the_pinned_version_and_writes_byo_usage(
    db_session: Any, _db_sessions: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    tenant, run_id = await _tenant_run(db_session)
    repo = ConnectionsRepository(db_session)
    ring = Keyring([("k1", KEY)])
    conn, v1 = await repo.create_version(
        tenant_id=tenant,
        slot="llm",
        kind="anthropic",
        model="claude-v1",
        secret=SECRET,
        keyring=ring,
    )
    # The tenant edits the connection AFTER the run pinned v1: the run keeps v1.
    await repo.create_version(
        tenant_id=tenant,
        slot="llm",
        kind="anthropic",
        model="claude-v2",
        secret="sk-other",  # pragma: allowlist secret - fake
        keyring=ring,
    )
    await db_session.commit()

    fake = FakeProvider(anthropic_reply("pinned answer", usage=(21, 9)))
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)
    set_identity_context(str(tenant), None, None)
    set_correlation_context(str(run_id), "promo_review")
    bind_run_provider_slots(
        {
            "provider_slots": {
                "llm": {
                    "source": "tenant",
                    "kind": "anthropic",
                    "connection_id": str(conn.id),
                    "version": v1.version,
                }
            }
        }
    )

    resp = await WMClient(timeout=5).message("promo-review", "review it")

    assert resp.output == "pinned answer"
    assert fake.bodies[0]["model"] == "claude-v1"
    # anthropic>=2 authenticates with `Authorization: Bearer`; 1.x used `x-api-key`.
    # Assert the tenant's key reaches the wire, whichever header the SDK uses.
    _req = fake.requests[0]
    assert (_req.headers.get("authorization") or _req.headers.get("x-api-key")) in (
        f"Bearer {SECRET}",
        SECRET,
    )

    rows = (
        (await db_session.execute(select(RunUsage).where(RunUsage.run_id == run_id)))
        .scalars()
        .all()
    )
    assert len(rows) == 1
    row = rows[0]
    assert (row.stage_name, row.model, row.provider) == ("promo_review", "claude-v1", "anthropic")
    assert (row.input_tokens, row.output_tokens, row.cost_usd) == (21, 9, Decimal(0))
    assert row.metadata_ is not None
    assert row.metadata_["funding"] == "tenant_byo"
    assert row.metadata_["source"] == IN_PROCESS_USAGE_SOURCE
    assert row.metadata_["connection_version"] == v1.version
    assert SECRET not in str(row.metadata_)


async def test_billing_mirror_keeps_in_process_rows(db_session: Any) -> None:
    _tenant, run_id = await _tenant_run(db_session)
    db_session.add(
        RunUsage(
            id=uuid7(),
            run_id=run_id,
            stage_name="promo_review",
            model="claude-v1",
            provider="anthropic",
            input_tokens=21,
            output_tokens=9,
            metadata_={"source": IN_PROCESS_USAGE_SOURCE, "funding": "tenant_byo"},
        )
    )
    await db_session.commit()
    repo = RunUsageRepository(db_session)

    await repo.replace_for_run(
        run_id=run_id,
        rows=[
            {"stage_name": "script", "model": "local-llm", "input_tokens": 5, "cost_usd": "0.01"}
        ],
    )
    await repo.replace_for_run(
        run_id=run_id,
        rows=[
            {"stage_name": "script", "model": "local-llm", "input_tokens": 6, "cost_usd": "0.02"}
        ],
    )
    await db_session.commit()

    rows = {(r.stage_name, r.input_tokens) for r in await repo.list_for_run(run_id)}
    assert rows == {("promo_review", 21), ("script", 6)}, "BYO row kept; mirror rows replaced"


async def test_run_with_only_in_process_rows_still_needs_reconciling(db_session: Any) -> None:
    tenant, run_id = await _tenant_run(db_session)
    await db_session.execute(update(Run).where(Run.id == run_id).values(status="completed"))
    db_session.add(
        RunUsage(
            id=uuid7(),
            run_id=run_id,
            stage_name="promo_review",
            metadata_={"source": IN_PROCESS_USAGE_SOURCE, "funding": "tenant_byo"},
        )
    )
    await db_session.commit()

    missing = await RunUsageRepository(db_session).run_ids_unsettled_usage(
        tenant, min_terminal_age_seconds=0
    )
    assert [r for r, _ in missing] == [run_id]


# ── service-reported + platform vLLM rows (LLM_BACKEND=vllm) ───────────────────


def _reported(run_id: UUID, call_id: str, tokens: int) -> RunUsage:
    return RunUsage(
        id=uuid7(),
        run_id=run_id,
        stage_name="analysis_summary",
        model="local-llm",
        provider="vllm",
        input_tokens=tokens,
        output_tokens=1,
        funding="local_gpu",
        metadata_={
            "source": SERVICE_REPORTED_USAGE_SOURCE,
            "funding": "local_gpu",
            "call_id": call_id,
            "service": "analysis-service",
        },
    )


async def test_billing_mirror_keeps_service_reported_rows(db_session: Any) -> None:
    _tenant, run_id = await _tenant_run(db_session)
    db_session.add(_reported(run_id, "c1", 40))
    db_session.add(
        RunUsage(id=uuid7(), run_id=run_id, stage_name="old", metadata_={"source": "billing"})
    )
    db_session.add(RunUsage(id=uuid7(), run_id=run_id, stage_name="legacy", metadata_=None))
    await db_session.commit()
    repo = RunUsageRepository(db_session)

    await repo.replace_for_run(
        run_id=run_id, rows=[{"stage_name": "script", "input_tokens": 6, "cost_usd": "0.02"}]
    )
    await db_session.commit()

    rows = {(r.stage_name, r.input_tokens) for r in await repo.list_for_run(run_id)}
    assert rows == {("analysis_summary", 40), ("script", 6)}, (
        "service-reported row kept; billing + NULL-source rows are the mirror's and replaced"
    )


async def test_run_with_only_service_reported_rows_still_needs_reconciling(
    db_session: Any,
) -> None:
    tenant, run_id = await _tenant_run(db_session)
    await db_session.execute(update(Run).where(Run.id == run_id).values(status="completed"))
    db_session.add(_reported(run_id, "c1", 40))
    await db_session.commit()

    missing = await RunUsageRepository(db_session).run_ids_unsettled_usage(
        tenant, min_terminal_age_seconds=0
    )
    assert [r for r, _ in missing] == [run_id]


async def test_insert_reported_is_idempotent_per_run_and_call_id(db_session: Any) -> None:
    _tenant, run_id = await _tenant_run(db_session)
    repo = RunUsageRepository(db_session)

    assert await repo.insert_reported(_reported(run_id, "same", 10)) is True
    assert await repo.insert_reported(_reported(run_id, "same", 99)) is False
    await db_session.commit()

    (row,) = await repo.list_for_run(run_id)
    assert row.input_tokens == 10
    assert (await repo.total(run_id))["input_tokens"] == 10


async def test_platform_vllm_call_writes_a_local_gpu_row(
    db_session: Any, _db_sessions: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLM_BACKEND=vllm: the orchestrator's OWN platform call never reaches WM or
    billing, so ``WMClient.message`` itself writes the ledger row."""
    import httpx2

    from src.config import clear_settings_cache

    tenant, run_id = await _tenant_run(db_session)
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    monkeypatch.setenv("LLM_VLLM_BASE_URL", "http://vllm.test:8001/v1")
    monkeypatch.setenv("LLM_VLLM_TOKEN", "vllm-licence-fake-0123456789")  # pragma: allowlist secret
    clear_settings_cache()
    fake = FakeProvider(
        reply=lambda _b, _n: httpx2.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 33, "completion_tokens": 7},
            },
        )
    )
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)
    set_identity_context(str(tenant), None, None)
    set_correlation_context(str(run_id), "demo_video_script")

    resp = await WMClient(timeout=5).message("film-critic", "hi")

    assert resp.output == "ok"
    (row,) = (
        (await db_session.execute(select(RunUsage).where(RunUsage.run_id == run_id)))
        .scalars()
        .all()
    )
    assert (row.stage_name, row.model, row.provider) == ("demo_video_script", "local-llm", "vllm")
    assert (row.input_tokens, row.output_tokens, row.cost_usd) == (33, 7, Decimal(0))
    assert row.funding == "local_gpu", "usage_metrics funding platform_direct → local_gpu"
    assert row.slot is None and row.connection_id is None
    assert row.metadata_ is not None
    assert row.metadata_["source"] == IN_PROCESS_USAGE_SOURCE
    assert row.metadata_["funding"] == "local_gpu"
    clear_settings_cache()


async def test_platform_vllm_call_without_a_run_writes_nothing_and_still_answers(
    _db_sessions: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from w2v_common.clients import WMResponse

    from src.services.llm import metering

    async def _boom() -> Any:
        raise AssertionError("no run bound: must not open a session")

    set_identity_context(str(uuid7()), None, None)
    set_correlation_context(None, None)
    monkeypatch.setattr(metering, "get_db_session", _boom)
    resp = WMResponse(
        output="x",
        output_type="string",
        output_schema_used=None,
        metadata=None,
        usage_metrics={"input_tokens": 1, "output_tokens": 1, "funding": "platform_direct"},
    )
    assert await metering.record_platform_vllm_usage(worker_name="w", response=resp) is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("platform_direct", "local_gpu")],
)
async def test_usage_metrics_funding_maps_to_the_taxonomy(raw: object, expected: str) -> None:
    from src.services.llm.metering import funding_from_usage_metrics

    assert funding_from_usage_metrics(raw) == expected
