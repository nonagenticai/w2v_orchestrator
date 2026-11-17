import datetime
import logging
from uuid import uuid4

import pytest
from sqlalchemy import select, update

from src.clients.billing_client import BillingCostUnavailableError
from src.database.models import Run
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository
from src.database.repositories.run_usage_repository import RunUsageRepository
from src.services.run_usage_reconciler import DEFAULT_BILLING_TIMEOUT_S, RunUsageReconciler

_T = DEFAULT_BILLING_TIMEOUT_S  # reconcile_one has no unbounded default

pytestmark = pytest.mark.asyncio


class _FakeBilling:
    def __init__(self, data: dict | None = None, *, raises: bool = False) -> None:
        self._data = data or {"groups": []}
        self._raises = raises
        self.calls: list = []

    async def costs_by_correlation(self, tenant_id, correlation_ids, *, group_by=None):
        self.calls.append((tenant_id, list(correlation_ids), group_by))
        if self._raises:
            raise BillingCostUnavailableError("boom")
        return self._data


async def _make_run(session, *, terminal: bool = True):
    tenant = uuid4()
    proj = await ProjectRepository(session).create_with_deliverables(
        tenant_id=tenant,
        owner_subject_id=uuid4(),
        type_="generic",
        title="x",
        brief={},
        brand={},
        voice={},
        theme={},
        visibility="tenant",
        deliverables=[
            {"name": "d", "format": "16x9", "captions_mode": "none", "target_duration_seconds": 60}
        ],
    )
    await session.commit()
    run = await RunRepository(session).create(
        deliverable_id=proj.deliverables[0].id,
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    await session.commit()
    if terminal:
        await RunRepository(session).mark_completed(run.id)
        # Backdate updated_at past the sweep's cost-flush grace so the tenant-wide
        # work-list includes it (simulates a run that terminated minutes ago).
        await session.execute(
            update(Run)
            .where(Run.id == run.id)
            .values(updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=2))
        )
        await session.commit()
    return tenant, run


async def test_reconcile_writes_billing_groups_into_run_usage(db_session) -> None:
    tenant, run = await _make_run(db_session)
    billing = _FakeBilling(
        {
            "groups": [
                {
                    "key": {
                        "correlation_id": str(run.id),
                        "stage": "narration",
                        "model": "claude-opus-4-8",
                        "provider": "anthropic",
                    },
                    "cost_usd": "0.12",
                    "input_tokens": 1000,
                    "output_tokens": 300,
                },
                {
                    "key": {
                        "correlation_id": str(run.id),
                        "stage": "render",
                        "model": "claude-opus-4-8",
                        "provider": "anthropic",
                    },
                    "cost_usd": "0.03",
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            ]
        }
    )
    reconciler = RunUsageReconciler(db_session, billing)
    written = await reconciler.reconcile_one(tenant, run.id, run.created_at, billing_timeout_s=_T)
    assert written == 2

    repo = RunUsageRepository(db_session)
    totals = await repo.total(run.id)
    assert totals["cost_usd"] == pytest.approx(0.15, rel=1e-4)
    assert totals["input_tokens"] == 1100
    stages = {r.stage_name for r in await repo.list_for_run(run.id)}
    assert stages == {"narration", "render"}
    # group_by must include correlation_id so per-run attribution survives.
    assert billing.calls[0][2] == ["correlation_id", "stage", "model", "provider"]


async def test_reconcile_unsettled_for_tenant_picks_up_terminal_run(db_session) -> None:
    tenant, run = await _make_run(db_session)
    billing = _FakeBilling(
        {
            "groups": [
                {
                    "key": {
                        "correlation_id": str(run.id),
                        "stage": "s",
                        "model": "m",
                        "provider": "p",
                    },
                    "cost_usd": "0.05",
                    "input_tokens": 10,
                    "output_tokens": 2,
                }
            ]
        }
    )
    reconciler = RunUsageReconciler(db_session, billing)
    written = await reconciler.reconcile_unsettled_for_tenant(tenant)
    assert written == 1
    assert float((await RunUsageRepository(db_session).total(run.id))["cost_usd"]) == pytest.approx(
        0.05, rel=1e-4
    )


async def test_reconcile_best_effort_on_billing_outage(db_session) -> None:
    tenant, run = await _make_run(db_session)
    reconciler = RunUsageReconciler(db_session, _FakeBilling(raises=True))
    written = await reconciler.reconcile_one(tenant, run.id, run.created_at, billing_timeout_s=_T)
    assert written == 0  # no exception bubbles; cost stays $0


async def test_sweep_marks_zero_cost_run_so_it_is_not_rebilled(db_session) -> None:
    # A genuinely zero-cost terminal run (billing has no cost for it) must be
    # marked reconciled by the tenant sweep so the next /usage load does NOT
    # re-query billing for it — the core efficiency fix.
    tenant, run = await _make_run(db_session)
    billing = _FakeBilling({"groups": []})
    reconciler = RunUsageReconciler(db_session, billing)

    written = await reconciler.reconcile_unsettled_for_tenant(tenant)
    assert written == 0  # no cost rows for a zero-cost run
    assert len(billing.calls) == 1  # billing queried once

    marked_at = (
        await db_session.execute(select(Run.usage_reconciled_at).where(Run.id == run.id))
    ).scalar_one()
    assert marked_at is not None  # the run is now marked reconciled

    # A second sweep must NOT re-query billing for this already-reconciled run.
    await reconciler.reconcile_unsettled_for_tenant(tenant)
    assert len(billing.calls) == 1  # still one — the run dropped out of the work-list


async def test_reconcile_coerces_null_token_and_cost_group(db_session) -> None:
    # A non-LLM cost event (TTS/render/upload) legitimately carries explicit
    # ``null`` token/cost fields. ``group.get(k, 0)`` returns None (the key
    # exists), so before the fix the repo's int(None)/Decimal(str(None)) raised
    # and the /usage + run-viewer cost tile 500'd instead of showing $0. Assert
    # the null group reconciles to 0 tokens / $0 with no exception.
    tenant, run = await _make_run(db_session)
    billing = _FakeBilling(
        {
            "groups": [
                {
                    "key": {
                        "correlation_id": str(run.id),
                        "stage": "narrate_tts",
                        "model": None,
                        "provider": "kokoro",
                    },
                    "cost_usd": None,
                    "input_tokens": None,
                    "output_tokens": None,
                }
            ]
        }
    )
    reconciler = RunUsageReconciler(db_session, billing)
    written = await reconciler.reconcile_one(tenant, run.id, run.created_at, billing_timeout_s=_T)
    assert written == 1  # the null row is written, coerced to zero

    totals = await RunUsageRepository(db_session).total(run.id)
    assert totals["input_tokens"] == 0
    assert totals["output_tokens"] == 0
    assert totals["cost_usd"] == pytest.approx(0.0, abs=1e-9)


# ── billing READ-CONTRACT pins ───────────────────────────────────────────────
# billing-service is developed on another machine and we cannot gate its
# changes, so we pin the exact response shape this reconciler consumes. It reads
# SEVEN field names and nothing else:
#   groups[].key.correlation_id / .stage / .model / .provider
#   groups[].input_tokens / .output_tokens / .cost_usd
# A rename on billing's side used to be SILENT here — ``group.get("cost_usd")
# or "0"`` turns an absent key into $0.00, so run_usage fills with zeros, the
# run is marked reconciled, and revenue attribution is permanently wrong with
# no error and no log.


async def test_reconcile_consumes_all_seven_billing_field_names(db_session) -> None:
    """Every one of the 7 consumed field names must reach a run_usage row.

    Renaming any of them on billing's side must break this test, not production
    revenue attribution.
    """
    tenant, run = await _make_run(db_session)
    billing = _FakeBilling(
        {
            "groups": [
                {
                    "key": {
                        "correlation_id": str(run.id),
                        "stage": "segment_select",
                        "model": "qwen3-27b",
                        "provider": "vllm",
                    },
                    "input_tokens": 4321,
                    "output_tokens": 765,
                    "cost_usd": "1.25",
                }
            ]
        }
    )
    written = await RunUsageReconciler(db_session, billing).reconcile_one(
        tenant, run.id, run.created_at, billing_timeout_s=_T
    )
    assert written == 1

    (row,) = await RunUsageRepository(db_session).list_for_run(run.id)
    # key.correlation_id — routed the group to THIS run.
    assert row.run_id == run.id
    assert row.stage_name == "segment_select"  # key.stage
    assert row.model == "qwen3-27b"  # key.model
    assert row.provider == "vllm"  # key.provider
    assert row.input_tokens == 4321  # input_tokens
    assert row.output_tokens == 765  # output_tokens
    assert float(row.cost_usd) == pytest.approx(1.25, rel=1e-9)  # cost_usd


async def test_renamed_price_field_is_not_silently_zeroed(db_session, caplog) -> None:
    """A group that carries cost under a DIFFERENT name must not record $0.00.

    This is the money bug: if billing renames ``cost_usd`` the key simply goes
    missing, ``.get()`` returns None and ``or "0"`` writes a confident $0.00.
    The reconciler must instead detect the unknown shape, refuse to write, and
    leave the run un-marked so it self-heals once the shape is understood.
    """
    tenant, run = await _make_run(db_session)
    billing = _FakeBilling(
        {
            "groups": [
                {
                    "key": {
                        "correlation_id": str(run.id),
                        "stage": "narration",
                        "model": "m",
                        "provider": "p",
                    },
                    "input_tokens": 1000,
                    "output_tokens": 300,
                    # billing renamed the price field — ``cost_usd`` is ABSENT.
                    "cash_cost_usd": "4.20",
                }
            ]
        }
    )
    with caplog.at_level(logging.ERROR):
        written = await RunUsageReconciler(db_session, billing).reconcile_one(
            tenant, run.id, run.created_at, billing_timeout_s=_T
        )

    assert written == 0, "an unknown billing shape must not be written as usage"
    assert await RunUsageRepository(db_session).list_for_run(run.id) == [], (
        "a group with cost data under an unknown name must not land as a $0.00 row"
    )
    assert "cost_usd" in caplog.text, "the missing field name must be named in the log"
    assert caplog.records and any(r.levelno >= logging.ERROR for r in caplog.records), (
        "a broken billing read contract must be logged loudly, not silently absorbed"
    )
    # Not marked reconciled ⇒ the run is retried once billing's shape is fixed.
    marked_at = (
        await db_session.execute(select(Run.usage_reconciled_at).where(Run.id == run.id))
    ).scalar_one()
    assert marked_at is None


async def test_renamed_token_field_is_not_silently_zeroed(db_session) -> None:
    """The same pin for the token counts, which drive the usage time-series."""
    tenant, run = await _make_run(db_session)
    billing = _FakeBilling(
        {
            "groups": [
                {
                    "key": {
                        "correlation_id": str(run.id),
                        "stage": "narration",
                        "model": "m",
                        "provider": "p",
                    },
                    "prompt_tokens": 1000,  # renamed from input_tokens
                    "output_tokens": 300,
                    "cost_usd": "0.12",
                }
            ]
        }
    )
    written = await RunUsageReconciler(db_session, billing).reconcile_one(
        tenant, run.id, run.created_at, billing_timeout_s=_T
    )
    assert written == 0
    assert await RunUsageRepository(db_session).list_for_run(run.id) == []


async def test_present_but_zero_cost_group_is_still_written(db_session) -> None:
    """KEY ABSENT is the defect; KEY PRESENT AND ZERO is legitimate.

    An unpriced event (a free/stub provider) really does cost $0 and must still
    reconcile to a $0 row — the strictness above must not swallow it.
    """
    tenant, run = await _make_run(db_session)
    billing = _FakeBilling(
        {
            "groups": [
                {
                    "key": {
                        "correlation_id": str(run.id),
                        "stage": "narrate_tts",
                        "model": None,
                        "provider": "kokoro",
                    },
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": "0",
                }
            ]
        }
    )
    written = await RunUsageReconciler(db_session, billing).reconcile_one(
        tenant, run.id, run.created_at, billing_timeout_s=_T
    )
    assert written == 1, "a genuinely zero-cost group must still be recorded"
    totals = await RunUsageRepository(db_session).total(run.id)
    assert totals["cost_usd"] == pytest.approx(0.0, abs=1e-9)


async def test_reconcile_one_does_not_mark_so_viewer_stays_live(db_session) -> None:
    # The on-demand run-viewer read must not pin the marker, so a run opened
    # before its cost flushed still reflects current billing on the next open.
    tenant, run = await _make_run(db_session)
    reconciler = RunUsageReconciler(db_session, _FakeBilling({"groups": []}))
    await reconciler.reconcile_one(tenant, run.id, run.created_at, billing_timeout_s=_T)

    marked_at = (
        await db_session.execute(select(Run.usage_reconciled_at).where(Run.id == run.id))
    ).scalar_one()
    assert marked_at is None


# ─── billing configured, no credential (CELL_AUTH_MODE=enforce, no token) ──────


class _CredentialMissingBilling:
    def __init__(self) -> None:
        self.calls = 0

    async def costs_by_correlation(self, tenant_id, correlation_ids, *, group_by=None):
        from src.clients.downstream_credential import DownstreamCredentialMissingError

        self.calls += 1
        raise DownstreamCredentialMissingError("billing")


def _unavailable_count() -> float:
    from prometheus_client import REGISTRY

    return (
        REGISTRY.get_sample_value(
            "w2v_orchestrator_cost_unavailable_total",
            {"reason": "billing_credential_missing"},
        )
        or 0.0
    )


async def test_credential_missing_propagates_never_reads_as_zero(
    db_session, monkeypatch, caplog
) -> None:
    """The bug this pins: a credential fault swallowed as ``return 0`` is a run that
    reads $0. It must propagate (callers turn it into cost_status=unavailable),
    be counted every time, logged at ERROR once per process, and mark nothing."""
    import src.services.run_usage_reconciler as mod
    from src.clients.downstream_credential import DownstreamCredentialMissingError

    monkeypatch.setattr(mod, "_credential_missing_logged", False)
    tenant, run = await _make_run(db_session)
    billing = _CredentialMissingBilling()
    reconciler = RunUsageReconciler(db_session, billing)
    before = _unavailable_count()

    with caplog.at_level(logging.ERROR, logger="src.services.run_usage_reconciler"):
        for _ in range(2):
            with pytest.raises(DownstreamCredentialMissingError):
                await reconciler.reconcile_one(
                    tenant, run.id, run.created_at, mark_reconciled=True, billing_timeout_s=_T
                )
        with pytest.raises(DownstreamCredentialMissingError):
            await reconciler.reconcile_unsettled_for_tenant(tenant)

    assert billing.calls == 3
    assert _unavailable_count() - before == 3, "every unavailable read is counted"
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1, f"ERROR once per process, got {len(errors)}"
    assert "no credential" in errors[0].getMessage()
    marked = (
        await db_session.execute(select(Run.usage_reconciled_at).where(Run.id == run.id))
    ).scalar_one()
    assert marked is None, "an unpriced run must stay unreconciled so it is retried"


async def test_run_usage_total_cannot_render_unknown_cost_as_zero() -> None:
    """``cost_status=unavailable`` and a numeric cost are mutually exclusive by
    construction — no code path can build an "unavailable but $0" payload."""
    from pydantic import ValidationError

    from src.api.schemas.runs import RunUsageTotal

    def build(**fields: object) -> RunUsageTotal:
        return RunUsageTotal.model_validate(
            {"input_tokens": 0, "output_tokens": 0, "by_stage": [], **fields}
        )

    with pytest.raises(ValidationError, match="must not carry a cost_usd"):
        build(
            cost_usd=0.0,
            cost_status="unavailable",
            cost_unavailable_reason="billing_credential_missing",
        )
    with pytest.raises(ValidationError, match="requires cost_unavailable_reason"):
        build(cost_usd=None, cost_status="unavailable")
    with pytest.raises(ValidationError, match="cost_status=ok requires cost_usd"):
        build(cost_usd=None)
    ok = build(
        cost_usd=None,
        cost_status="unavailable",
        cost_unavailable_reason="billing_credential_missing",
    )
    assert ok.model_dump()["cost_usd"] is None


async def test_billing_timeout_degrades_to_nothing_written(db_session) -> None:
    """``billing_timeout_s`` bounds the billing READ only: a hanging billing returns
    0 promptly (degrades like an outage) — nothing written, nothing marked."""
    import asyncio
    import time

    tenant, run = await _make_run(db_session)

    class _Hanging:
        async def costs_by_correlation(self, tenant_id, correlation_ids, *, group_by=None):
            await asyncio.sleep(60)
            return {"groups": []}

    started = time.perf_counter()
    n = await RunUsageReconciler(db_session, _Hanging()).reconcile_one(
        tenant, run.id, run.created_at, mark_reconciled=True, billing_timeout_s=0.05
    )
    assert n == 0
    assert time.perf_counter() - started < 5
    marked = (
        await db_session.execute(select(Run.usage_reconciled_at).where(Run.id == run.id))
    ).scalar_one()
    assert marked is None
    assert await RunUsageRepository(db_session).list_for_run(run.id) == []


async def test_sweep_writes_in_constant_statements_and_keeps_per_run_subject(db_session) -> None:
    """Reconciling N runs is a batched replace: the statement count does not grow
    with N (was DELETE + SELECT + UPDATE per run), and every run's rows still
    carry THAT run's own snapshot subject."""
    from sqlalchemy import event

    from src.database.models import RunUsage

    async def _reconcile_n(n: int) -> tuple[int, dict]:
        tenant, first = await _make_run(db_session)
        runs = [first]
        for _ in range(n - 1):
            runs.append(
                await RunRepository(db_session).create(
                    deliverable_id=first.deliverable_id,
                    blueprint_id="generic_v1",
                    config_snapshot={},
                    pause_after_stages=[],
                )
            )
        subjects = {r.id: uuid4() for r in runs}
        for r in runs:
            await db_session.execute(
                update(Run)
                .where(Run.id == r.id)
                .values(config_snapshot={"subject_id": str(subjects[r.id])})
            )
        await db_session.commit()
        groups = [
            {
                "key": {"correlation_id": str(r.id), "stage": "s", "model": "m", "provider": "p"},
                "cost_usd": "0.01",
                "input_tokens": 1,
                "output_tokens": 1,
            }
            for r in runs
        ]
        statements: list[str] = []

        def _count(_conn, _cursor, statement, *_a) -> None:
            statements.append(statement)

        sync_engine = db_session.bind.sync_engine
        event.listen(sync_engine, "before_cursor_execute", _count)
        try:
            written = await RunUsageReconciler(
                db_session, _FakeBilling({"groups": groups})
            ).reconcile(tenant, [(r.id, r.created_at) for r in runs], mark_reconciled=True)
        finally:
            event.remove(sync_engine, "before_cursor_execute", _count)
        assert written == n
        got = {
            row.run_id: row.subject_id
            for row in (
                await db_session.execute(
                    select(RunUsage).where(RunUsage.run_id.in_([r.id for r in runs]))
                )
            ).scalars()
        }
        assert got == subjects
        return len([s for s in statements if not s.lstrip().upper().startswith("INSERT")]), got

    two, _ = await _reconcile_n(2)
    six, _ = await _reconcile_n(6)
    assert six == two, f"non-INSERT statements grew with N: {two} for 2 runs, {six} for 6"


async def test_a_run_locked_by_the_sweep_is_skipped_not_waited_for(
    db_session, db_session_factory
) -> None:
    """Contention must cost a page NOTHING: the mirror lock is try-only.

    Measured before this fix: the viewer took the BLOCKING advisory lock BEFORE
    reading billing, so a run the tenant-wide sweep was holding made the run page
    wait for the sweep's whole 200-run transaction — 503 after 8s, versus <0.2s
    on main. Now the contended run is dropped instead: nothing written, nothing
    marked, and the page immediately serves the stored cost as not-final.
    """
    import asyncio
    import time

    from sqlalchemy import text

    from src.database.repositories.run_usage_repository import mirror_lock_key

    tenant, run = await _make_run(db_session)
    billing = _FakeBilling(
        {
            "groups": [
                {
                    "key": {
                        "correlation_id": str(run.id),
                        "stage": "s",
                        "model": "m",
                        "provider": "p",
                    },
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "cost_usd": "0.03",
                }
            ]
        }
    )

    async with db_session_factory() as holder:
        # A second connection holds this run's mirror lock for its whole
        # transaction, exactly as the sweep does.
        await holder.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"),
            {"k": mirror_lock_key(run.id)},
        )
        started = time.perf_counter()
        n = await asyncio.wait_for(
            RunUsageReconciler(db_session, billing).reconcile_one(
                tenant, run.id, run.created_at, mark_reconciled=True, billing_timeout_s=_T
            ),
            timeout=5.0,
        )
        elapsed = time.perf_counter() - started

    assert n == 0, "a run another caller holds must not be written"
    assert elapsed < 2.0, f"the viewer waited {elapsed:.1f}s on the sweep's lock"
    # Billing is read FIRST now; the drop happens after it answers. That is the
    # trade: a duplicate (cheap) billing read instead of a blocked page.
    assert billing.calls
    assert await RunUsageRepository(db_session).list_for_run(run.id) == []
    assert (
        await db_session.execute(select(Run.usage_reconciled_at).where(Run.id == run.id))
    ).scalar_one() is None

    # Once the sweep releases, the very same call converges: the skip is a delay,
    # never a loss.
    n = await RunUsageReconciler(db_session, billing).reconcile_one(
        tenant, run.id, run.created_at, mark_reconciled=True, billing_timeout_s=_T
    )
    assert n == 1
    assert (
        await db_session.execute(select(Run.usage_reconciled_at).where(Run.id == run.id))
    ).scalar_one() is not None


async def test_a_run_settled_while_billing_is_read_is_not_overwritten(
    db_session, db_session_factory
) -> None:
    """The unsettled re-check runs UNDER the lock, after billing answers.

    Billing is read with no lock held, so another caller can settle the run in
    that window (here: a second connection writes its mirror row and marks it
    settled, as a concurrent sweep would). This caller's billing answer is then
    the STALER one, and must be dropped — not replace the settled mirror."""
    tenant, run = await _make_run(db_session)

    class _SettledMeanwhile(_FakeBilling):
        async def costs_by_correlation(self, tenant_id, correlation_ids, *, group_by=None):
            async with db_session_factory() as other:
                await RunUsageRepository(other).replace_for_runs(
                    [
                        (
                            run.id,
                            [
                                {
                                    "stage_name": "fresh",
                                    "model": "m",
                                    "provider": "p",
                                    "input_tokens": 5,
                                    "output_tokens": 5,
                                    "cost_usd": 0.50,
                                }
                            ],
                            run.created_at,
                        )
                    ],
                    mark_reconciled=True,
                )
                await other.commit()
            return await super().costs_by_correlation(tenant_id, correlation_ids, group_by=group_by)

    stale = _SettledMeanwhile(
        {
            "groups": [
                {
                    "key": {
                        "correlation_id": str(run.id),
                        "stage": "stale",
                        "model": "m",
                        "provider": "p",
                    },
                    "cost_usd": "0.01",
                    "input_tokens": 1,
                    "output_tokens": 1,
                }
            ]
        }
    )
    n = await RunUsageReconciler(db_session, stale).reconcile_one(
        tenant, run.id, run.created_at, mark_reconciled=True, billing_timeout_s=_T
    )

    assert n == 0, "a run settled by another caller must not be re-written"
    assert {r.stage_name for r in await RunUsageRepository(db_session).list_for_run(run.id)} == {
        "fresh"
    }
