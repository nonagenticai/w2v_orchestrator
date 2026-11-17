"""Admin tier (Phase B, 6th increment) — GET /api/admin/usage.

The cost/token drill-down behind /api/admin/overview. Tenant-scoped via
``TenantAdminDep`` (so it honours the X-Active-Tenant-Id scope-switch), all-time
totals plus breakdowns by project type, usage stage, and model/provider — all
read from ``run_usage`` joined up the Run→Deliverable→Project chain.

Same per-test-app + ``_SeedRoles`` middleware idiom as ``test_admin_authz.py``
(the conftest ``client`` fixture seeds no roles).
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import update
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes.admin import router as admin_router
from src.database.models import Deliverable, Project, Run, RunUsage
from tests.api.routes._licence_stubs import override_entitled

pytestmark = pytest.mark.asyncio


def _build_app(db_session_factory, *, tenant_id: str | None, roles: list[str] | None) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = db_session_factory
    app.include_router(admin_router)
    # require_license is ON by default and fails CLOSED, so the licence-gated
    # admin read routes would 503 here without a licences client and mask what
    # this file tests. Hand the gate an entitled answer; it stays mounted.
    override_entitled(app)
    subject_id = str(uuid4())

    class _SeedRoles(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            if tenant_id is not None:
                request.state.tenant_id = tenant_id
            if roles is not None:
                request.state.roles = roles
            request.state.subject_id = subject_id
            return await call_next(request)

    app.add_middleware(_SeedRoles)
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def _seed_project(
    session, tenant_id: UUID, ptype: str, runs: list[list[dict[str, Any]]]
) -> None:
    """One project of ``ptype`` with one deliverable and N runs; each ``runs``
    entry is that run's list of usage-row kwargs (an empty list = a run with no
    usage rows)."""
    proj = Project(
        id=uuid7(),
        tenant_id=tenant_id,
        owner_subject_id=uuid7(),
        visibility="tenant",
        type=ptype,
        title="P",
    )
    session.add(proj)
    deliv = Deliverable(id=uuid7(), project_id=proj.id, name="d", format="16x9")
    session.add(deliv)
    await session.flush()
    for usage_rows in runs:
        run = Run(
            id=uuid7(), deliverable_id=deliv.id, blueprint_id=f"{ptype}_v1", status="completed"
        )
        session.add(run)
        await session.flush()
        for u in usage_rows:
            session.add(RunUsage(id=uuid7(), run_id=run.id, **u))
    await session.flush()


def _u(
    stage: str,
    model: str | None,
    provider: str | None,
    cost: str,
    i: int,
    o: int,
    created_at: datetime.datetime | None = None,
) -> dict:
    row: dict[str, Any] = {
        "stage_name": stage,
        "model": model,
        "provider": provider,
        "cost_usd": Decimal(cost),
        "input_tokens": i,
        "output_tokens": o,
    }
    # Only set created_at when a test pins a day for the time-series; otherwise
    # the server_default (now()) applies.
    if created_at is not None:
        row["created_at"] = created_at
    return row


async def _seed_tenant_a(session, tenant_a: UUID) -> None:
    # promo project: run1 (script gpt + narrate kokoro) + run3 (NO usage rows).
    await _seed_project(
        session,
        tenant_a,
        "promo",
        runs=[
            [
                _u("script", "gpt", "openai", "2.00", 100, 50),
                _u("narrate", "kokoro", "tts", "0.50", 0, 0),
            ],
            [],  # a completed run that recorded no usage
        ],
    )
    # course project: run2 (script gpt).
    await _seed_project(
        session,
        tenant_a,
        "course",
        runs=[[_u("script", "gpt", "openai", "3.00", 200, 100)]],
    )


# ─── authz ───────────────────────────────────────────────────────────────

_MATRIX = [
    ("anon", False, None, 401),
    ("member", True, ["member"], 403),
    ("tenant_admin", True, ["tenant_admin"], 200),
    ("super_admin", True, ["super_admin", "tenant_admin"], 200),
]


@pytest.mark.parametrize("label,has_tenant,roles,expect", [("anon", False, None, 401)])
async def test_usage_authz_matrix(
    db_session_factory, label: str, has_tenant: bool, roles: list[str] | None, expect: int
) -> None:
    tenant_id = str(uuid4()) if has_tenant else None
    app = _build_app(db_session_factory, tenant_id=tenant_id, roles=roles)
    async with _client(app) as c:
        r = await c.get("/api/admin/usage")
    assert r.status_code == expect, f"{label} GET /api/admin/usage"


# ─── correctness ───────────────────────────────────────────────────────────


async def test_usage_totals_and_breakdowns(db_session, db_session_factory) -> None:
    tenant_a = uuid4()
    await _seed_tenant_a(db_session, tenant_a)
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(tenant_a), roles=["tenant_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/usage")
    assert r.status_code == 200
    body = r.json()
    assert body["tenant_id"] == str(tenant_a)

    totals = body["totals"]
    assert totals["project_count"] == 2
    assert totals["run_count"] == 3  # run3 has no usage but still counts
    assert totals["input_tokens"] == 300
    assert totals["output_tokens"] == 150
    assert totals["cost_usd"] == pytest.approx(5.50)

    # by_type: ordered by cost desc → course (3.00) before promo (2.50).
    by_type = body["by_type"]
    assert [t["type"] for t in by_type] == ["course", "promo"]
    course = next(t for t in by_type if t["type"] == "course")
    promo = next(t for t in by_type if t["type"] == "promo")
    assert course["run_count"] == 1
    assert course["cost_usd"] == pytest.approx(3.00)
    assert promo["run_count"] == 2  # run1 + the usage-less run3
    assert promo["cost_usd"] == pytest.approx(2.50)
    assert promo["input_tokens"] == 100

    # by_stage: INNER-join usage (3 rows) → script (5.00) before narrate (0.50);
    # the usage-less run contributes nothing here.
    by_stage = body["by_stage"]
    assert [s["stage_name"] for s in by_stage] == ["script", "narrate"]
    script = by_stage[0]
    assert script["call_count"] == 2
    assert script["cost_usd"] == pytest.approx(5.00)
    assert script["input_tokens"] == 300
    assert script["output_tokens"] == 150
    assert by_stage[1]["call_count"] == 1
    assert by_stage[1]["cost_usd"] == pytest.approx(0.50)

    # by_model: gpt/openai (5.00) before kokoro/tts (0.50).
    by_model = body["by_model"]
    assert [(m["model"], m["provider"]) for m in by_model] == [
        ("gpt", "openai"),
        ("kokoro", "tts"),
    ]
    assert by_model[0]["call_count"] == 2
    assert by_model[0]["cost_usd"] == pytest.approx(5.00)
    assert by_model[1]["call_count"] == 1


async def test_usage_empty_tenant_returns_zeros(db_session_factory) -> None:
    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["tenant_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/usage")
    assert r.status_code == 200
    body = r.json()
    assert body["totals"]["run_count"] == 0
    assert body["totals"]["project_count"] == 0
    assert body["totals"]["cost_usd"] == 0.0
    assert body["by_type"] == []
    assert body["by_stage"] == []
    assert body["by_model"] == []


async def test_usage_scope_switch_reads_foreign_tenant(db_session, db_session_factory) -> None:
    """A super_admin with X-Active-Tenant-Id reads the FOREIGN tenant's usage,
    isolated from their own claim's tenant."""
    tenant_a, tenant_b = uuid4(), uuid4()
    await _seed_tenant_a(db_session, tenant_a)
    await _seed_project(
        db_session,
        tenant_b,
        "lesson",
        runs=[[_u("compile", "claude", "anthropic", "9.99", 10, 20)]],
    )
    await db_session.commit()

    app = _build_app(
        db_session_factory, tenant_id=str(tenant_a), roles=["super_admin", "tenant_admin"]
    )
    async with _client(app) as c:
        r = await c.get("/api/admin/usage", headers={"X-Active-Tenant-Id": str(tenant_b)})
    assert r.status_code == 200
    body = r.json()
    assert body["tenant_id"] == str(tenant_b)
    assert body["totals"]["project_count"] == 1  # B only, A excluded
    assert body["totals"]["cost_usd"] == pytest.approx(9.99)
    assert [t["type"] for t in body["by_type"]] == ["lesson"]


async def test_usage_window_days_scopes_totals_and_breakdowns(
    db_session, db_session_factory
) -> None:
    """?window_days=N scopes the WHOLE response — totals, project_count, and
    every by-* breakdown — to in-window run_usage, not just the chart. A
    40-day-old run and a usage-less run drop out when windowed; with no param
    everything (incl. the usage-less run) reappears. now()-relative, so asserted
    on values rather than dates."""
    now = datetime.datetime.now(datetime.UTC)
    recent = now - datetime.timedelta(days=2)
    old = now - datetime.timedelta(days=40)
    tenant = uuid4()
    # promo: a recent metered run (in window) + a usage-less run (no run_usage).
    await _seed_project(
        db_session,
        tenant,
        "promo",
        runs=[
            [
                _u("script", "gpt", "openai", "1.00", 10, 5, created_at=recent),
                _u("narrate", "kokoro", "tts", "0.50", 0, 0, created_at=recent),
            ],
            [],  # usage-less run — counts all-time, drops out when windowed
        ],
    )
    # course: a single 40-day-old run (outside a 7-day window).
    await _seed_project(
        db_session,
        tenant,
        "course",
        runs=[[_u("script", "gpt", "openai", "9.00", 90, 45, created_at=old)]],
    )
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(tenant), roles=["tenant_admin"])

    # ── windowed: only the recent promo run's metered usage ──────────────────
    async with _client(app) as c:
        r = await c.get("/api/admin/usage?window_days=7")
    assert r.status_code == 200
    w = r.json()
    assert w["totals"]["cost_usd"] == pytest.approx(1.50)  # 1.00 + 0.50, old 9.00 excluded
    assert w["totals"]["input_tokens"] == 10
    assert w["totals"]["output_tokens"] == 5
    assert w["totals"]["run_count"] == 1  # recent metered run only (old + usage-less excluded)
    assert w["totals"]["project_count"] == 1  # promo only (course's usage is out of window)
    assert [t["type"] for t in w["by_type"]] == ["promo"]  # course absent
    assert w["by_type"][0]["run_count"] == 1
    assert {s["stage_name"] for s in w["by_stage"]} == {"script", "narrate"}
    script = next(s for s in w["by_stage"] if s["stage_name"] == "script")
    assert script["cost_usd"] == pytest.approx(1.00)  # in-window only, NOT 10.00
    assert {m["model"] for m in w["by_model"]} == {"gpt", "kokoro"}
    gpt = next(m for m in w["by_model"] if m["model"] == "gpt")
    assert gpt["cost_usd"] == pytest.approx(1.00)  # old 9.00 gpt usage excluded

    # ── all-time (no param): the full figures, incl. the usage-less run ──────
    async with _client(app) as c:
        r = await c.get("/api/admin/usage")
    assert r.status_code == 200
    a = r.json()
    assert a["totals"]["cost_usd"] == pytest.approx(10.50)  # 1.50 + 9.00
    assert a["totals"]["run_count"] == 3  # recent + old + usage-less
    assert a["totals"]["project_count"] == 2  # promo + course
    assert {t["type"] for t in a["by_type"]} == {"promo", "course"}


# ─── repo unit ─────────────────────────────────────────────────────────────


async def test_usage_detail_repo_shape(db_session) -> None:
    from src.database.repositories.admin_repository import AdminRepository

    tenant_a = uuid4()
    await _seed_tenant_a(db_session, tenant_a)
    await db_session.commit()

    data = await AdminRepository(db_session).usage_detail(tenant_a)
    assert data["totals"]["run_count"] == 3
    assert data["totals"]["cost_usd"] == pytest.approx(5.50)
    assert {t["type"] for t in data["by_type"]} == {"promo", "course"}
    assert {s["stage_name"] for s in data["by_stage"]} == {"script", "narrate"}
    assert {m["model"] for m in data["by_model"]} == {"gpt", "kokoro"}
    assert "time_series" in data


# ─── time-series (increments 7 + 8) ──────────────────────────────────────────


async def test_usage_time_series_buckets_by_day_gap_filled(db_session, db_session_factory) -> None:
    """Daily date_trunc buckets, oldest first; per-day cost summed; gap-filled
    (the empty 06-14 between 06-13 and 06-15 appears at cost 0); a usage-less run
    contributes nothing."""
    tz = datetime.UTC
    d1 = datetime.datetime(2026, 6, 13, 12, 0, tzinfo=tz)
    d2 = datetime.datetime(2026, 6, 15, 12, 0, tzinfo=tz)
    tenant = uuid4()
    await _seed_project(
        db_session,
        tenant,
        "promo",
        runs=[
            # Day 1: two usage rows (same day) → one bucket summing 1.50.
            [
                _u("script", "gpt", "openai", "1.00", 10, 5, created_at=d1),
                _u("narrate", "kokoro", "tts", "0.50", 0, 0, created_at=d1),
            ],
            # Day 2 (skip 06-14): one usage row → 2.25.
            [_u("script", "gpt", "openai", "2.25", 20, 10, created_at=d2)],
            # A completed run with NO usage rows → absent from the series.
            [],
        ],
    )
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(tenant), roles=["tenant_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/usage")
    assert r.status_code == 200
    series = r.json()["time_series"]

    # Gap-filled + ascending: 06-14 is present (filler).
    assert [p["bucket"] for p in series] == ["2026-06-13", "2026-06-14", "2026-06-15"]
    by_day = {p["bucket"]: p for p in series}
    assert by_day["2026-06-13"]["cost_usd"] == pytest.approx(1.50)
    assert by_day["2026-06-13"]["input_tokens"] == 10
    assert by_day["2026-06-14"]["cost_usd"] == pytest.approx(0.0)  # zero-filled gap
    assert by_day["2026-06-14"]["input_tokens"] == 0
    assert by_day["2026-06-15"]["cost_usd"] == pytest.approx(2.25)
    # The series cost sums to the all-time total (fillers add nothing).
    assert sum(p["cost_usd"] for p in series) == pytest.approx(3.75)


async def test_usage_time_series_buckets_by_week(db_session, db_session_factory) -> None:
    """?bucket=week → buckets keyed on the Monday of each ISO week; same-week
    days collapse, costs sum per week."""
    tz = datetime.UTC
    # 2026-06-08 is a Monday. 06-09 + 06-10 are the same week (Mon 06-08);
    # 06-16 is the next week (Mon 06-15).
    w1a = datetime.datetime(2026, 6, 9, 9, 0, tzinfo=tz)
    w1b = datetime.datetime(2026, 6, 10, 9, 0, tzinfo=tz)
    w2 = datetime.datetime(2026, 6, 16, 9, 0, tzinfo=tz)
    tenant = uuid4()
    await _seed_project(
        db_session,
        tenant,
        "promo",
        runs=[
            [
                _u("script", "gpt", "openai", "1.00", 10, 5, created_at=w1a),
                _u("narrate", "kokoro", "tts", "0.50", 0, 0, created_at=w1b),
            ],
            [_u("script", "gpt", "openai", "2.00", 20, 10, created_at=w2)],
        ],
    )
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(tenant), roles=["tenant_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/usage?bucket=week")
    assert r.status_code == 200
    series = r.json()["time_series"]
    by_week = {p["bucket"]: p for p in series}
    assert [p["bucket"] for p in series] == ["2026-06-08", "2026-06-15"]  # Mondays
    assert by_week["2026-06-08"]["cost_usd"] == pytest.approx(1.50)  # both days collapse
    assert by_week["2026-06-15"]["cost_usd"] == pytest.approx(2.00)


async def test_usage_time_series_buckets_by_month(db_session, db_session_factory) -> None:
    """?bucket=month → buckets keyed on the 1st; gap-filled across months."""
    tz = datetime.UTC
    m1 = datetime.datetime(2026, 4, 20, 9, 0, tzinfo=tz)
    m3 = datetime.datetime(2026, 6, 5, 9, 0, tzinfo=tz)  # skips May
    tenant = uuid4()
    await _seed_project(
        db_session,
        tenant,
        "promo",
        runs=[
            [_u("script", "gpt", "openai", "1.00", 10, 5, created_at=m1)],
            [_u("script", "gpt", "openai", "3.00", 20, 10, created_at=m3)],
        ],
    )
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(tenant), roles=["tenant_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/usage?bucket=month")
    assert r.status_code == 200
    series = r.json()["time_series"]
    by_month = {p["bucket"]: p for p in series}
    # 1st-of-month buckets, gap-filled (May appears at 0).
    assert [p["bucket"] for p in series] == ["2026-04-01", "2026-05-01", "2026-06-01"]
    assert by_month["2026-04-01"]["cost_usd"] == pytest.approx(1.00)
    assert by_month["2026-05-01"]["cost_usd"] == pytest.approx(0.0)  # zero-filled gap
    assert by_month["2026-06-01"]["cost_usd"] == pytest.approx(3.00)


async def test_usage_time_series_window_days_excludes_old_rows(
    db_session, db_session_factory
) -> None:
    """?window_days=7 → the axis runs to today, an in-window row is present, a
    40-day-old row is excluded. Asserted structurally (window is now()-relative)."""
    now = datetime.datetime.now(datetime.UTC)
    recent = now - datetime.timedelta(days=2)
    old = now - datetime.timedelta(days=40)
    tenant = uuid4()
    await _seed_project(
        db_session,
        tenant,
        "promo",
        runs=[
            [_u("script", "gpt", "openai", "1.00", 10, 5, created_at=recent)],
            [_u("script", "gpt", "openai", "9.00", 90, 45, created_at=old)],
        ],
    )
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(tenant), roles=["tenant_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/usage?window_days=7&bucket=day")
    assert r.status_code == 200
    series = r.json()["time_series"]
    buckets = {p["bucket"]: p for p in series}

    today = now.date().isoformat()
    recent_day = recent.date().isoformat()
    old_day = old.date().isoformat()
    # The axis runs to today; the in-window day carries its cost; the gap days
    # are zero-filled; the 40-day-old row is outside the window entirely.
    assert today in buckets
    assert buckets[recent_day]["cost_usd"] == pytest.approx(1.00)
    assert old_day not in buckets
    # Windowed series excludes the old row; its cost (9.00) is absent.
    assert sum(p["cost_usd"] for p in series) == pytest.approx(1.00)
    # An 8-bucket window (today + 7 prior days) is fully gap-filled.
    assert len(series) == 8


async def test_usage_empty_tenant_has_empty_series(db_session_factory) -> None:
    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["tenant_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/usage")
    assert r.status_code == 200
    assert r.json()["time_series"] == []


# ─── P6 ORCH-3: by_user / by_slot ────────────────────────────────────────


async def test_usage_by_user_and_by_slot(db_session, db_session_factory) -> None:
    """Every usage row lands in exactly one by_user and one by_slot group, so both
    sum to the totals: a removed member is kept (as removed), a NULL subject is the
    "unattributed" user, a billing-mirrored row is the "platform" slot (route) and a
    row with neither slot nor mirror provenance is the "unattributed" slot. Funding is
    the taxonomy value, else "unattributed" (NULL = not known — never guessed). The member join is scoped to
    THIS tenant (the same subject's membership elsewhere never leaks its email)."""
    from src.database.models import TenantMember

    tenant, other = uuid4(), uuid4()
    alice, bob, svc = uuid4(), uuid4(), uuid4()
    now = datetime.datetime.now(datetime.UTC)
    db_session.add_all(
        [
            TenantMember(subject_id=alice, tenant_id=tenant, email="alice@acme.test"),
            TenantMember(
                subject_id=bob,
                tenant_id=tenant,
                email="bob@acme.test",
                status="removed",
                removed_at=now,
            ),
            TenantMember(subject_id=svc, tenant_id=other, email="leak@other.test"),
        ]
    )

    def row(cost: str, i: int, o: int, **attr: Any) -> dict[str, Any]:
        return {**_u("s", "m", "p", cost, i, o), **attr}

    await _seed_project(
        db_session,
        tenant,
        "promo",
        runs=[
            [
                row(
                    "0",
                    10,
                    5,
                    subject_id=alice,
                    slot="llm",
                    funding="tenant_byo",
                    units=Decimal(15),
                    unit_kind="tokens",
                ),
                row(
                    "0",
                    0,
                    0,
                    subject_id=alice,
                    slot="tts",
                    funding="tenant_byo",
                    units=Decimal(300),
                    unit_kind="chars",
                ),
                row(
                    "1.50",
                    100,
                    50,
                    subject_id=alice,
                    funding="local_gpu",
                    units=Decimal(150),
                    unit_kind="tokens",
                ),
                row(
                    "2.00",
                    20,
                    10,
                    subject_id=bob,
                    units=Decimal(30),
                    unit_kind="tokens",
                ),
                row("0.25", 1, 1, subject_id=svc),
                # in-process row with no slot/funding/subject: nothing attributes it
                row("0.75", 3, 3, metadata_={"source": "orchestrator_in_process"}),
            ]
        ],
    )
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(tenant), roles=["tenant_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/usage")
    assert r.status_code == 200, r.text
    body = r.json()
    totals = body["totals"]

    users = {u["label"]: u for u in body["by_user"]}
    assert set(users) == {"alice@acme.test", "bob@acme.test", str(svc), "unattributed"}
    assert users["alice@acme.test"]["call_count"] == 3
    assert users["alice@acme.test"]["cost_usd"] == pytest.approx(1.50)
    assert users["alice@acme.test"]["member_status"] == "active"
    assert users["bob@acme.test"]["member_status"] == "removed"
    assert users[str(svc)]["email"] is None and users[str(svc)]["member_status"] is None
    assert users["unattributed"]["subject_id"] is None
    assert users["unattributed"]["cost_usd"] == pytest.approx(0.75)
    for key in ("cost_usd", "input_tokens", "output_tokens"):
        assert sum(u[key] for u in body["by_user"]) == pytest.approx(totals[key]), key
    assert sum(u["call_count"] for u in body["by_user"]) == 6

    slots = {(s["slot"], s["funding"]): s for s in body["by_slot"]}
    assert set(slots) == {
        ("llm", "tenant_byo"),
        ("tts", "tenant_byo"),
        ("platform", "local_gpu"),
        ("platform", "unattributed"),
        ("unattributed", "unattributed"),
    }
    # no funding outside the agreed taxonomy (+ the unknown bucket) is ever reported
    assert {s["funding"] for s in body["by_slot"]} <= {
        "local_gpu",
        "platform_external",
        "tenant_byo",
        "unattributed",
    }
    assert slots[("llm", "tenant_byo")]["units"] == {"tokens": 15}
    assert slots[("tts", "tenant_byo")]["units"] == {"chars": 300}
    assert slots[("platform", "local_gpu")]["units"] == {"tokens": 150}
    unknown = slots[("platform", "unattributed")]
    assert unknown["call_count"] == 2
    assert unknown["units"] == {"tokens": 30}  # the svc row carries no unit
    assert unknown["cost_usd"] == pytest.approx(2.25)
    assert slots[("unattributed", "unattributed")]["units"] == {}
    for key in ("cost_usd", "input_tokens", "output_tokens"):
        assert sum(s[key] for s in body["by_slot"]) == pytest.approx(totals[key]), key
    assert sum(s["call_count"] for s in body["by_slot"]) == 6


async def test_usage_by_user_honours_the_window(db_session, db_session_factory) -> None:
    tenant, alice = uuid4(), uuid4()
    old = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=40)
    await _seed_project(
        db_session,
        tenant,
        "promo",
        runs=[
            [
                {**_u("s", "m", "p", "1.00", 1, 1), "subject_id": alice},
                {**_u("s", "m", "p", "9.00", 1, 1, created_at=old), "subject_id": alice},
            ]
        ],
    )
    await db_session.commit()
    app = _build_app(db_session_factory, tenant_id=str(tenant), roles=["tenant_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/usage", params={"window_days": 7})
    (only,) = r.json()["by_user"]
    assert only["subject_id"] == str(alice) and only["cost_usd"] == pytest.approx(1.00)


# ─── billing configured, no credential (CELL_AUTH_MODE=enforce, no token) ──────


class _CredentialMissingBilling:
    async def costs_by_correlation(self, tenant_id, correlation_ids, *, group_by=None):
        from src.clients.downstream_credential import DownstreamCredentialMissingError

        raise DownstreamCredentialMissingError("billing")


@pytest.mark.parametrize("path", ["/api/admin/usage"])
async def test_admin_cost_reads_report_unavailable_not_500(
    db_session, db_session_factory, path: str
) -> None:
    """A not-yet-mirrored terminal run + billing with no credential: the admin cost
    pages must 200 and SAY cost is unavailable, not 500 and not pass silently."""
    from src.api.routes._common import get_billing_client

    tenant = uuid4()
    await _seed_project(db_session, tenant, "promo", runs=[[]])
    # Past the sweep's freshness guard so the run is on the reconcile work-list.
    await db_session.execute(
        update(Run).values(
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1)
        )
    )
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(tenant), roles=["tenant_admin"])
    app.dependency_overrides[get_billing_client] = _CredentialMissingBilling
    async with _client(app) as c:
        r = await c.get(path)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cost_status"] == "unavailable"
    assert body["cost_unavailable_reason"] == "billing_credential_missing"
