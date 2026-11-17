"""P6 ORCH-2 — the pipeline reads tenant defaults, pinned into the run snapshot.

Real postgres throughout (``tenant_defaults`` rows written through the repository).
Claims, one per default the blueprint lists, plus the two cross-cutting ones:

* golden: no row → the pre-P6 snapshot (no new keys, deployment values); a row
  whose every field is NULL → the same snapshot plus only the version pin;
* each tenant field overrides the deployment default (brand, voice id/speed, voice
  direction tone, retention mode, captions karaoke, timing-director tier) and an
  explicit project / brief / run value still beats it;
* ``retention_mode=off`` seeds nothing and ``retention_plan`` then skips; the
  operator kill-switch stays global;
* captions mode: an omitted ``captions_mode`` on a new deliverable takes the
  tenant's, an explicit one wins, no row → ``none``;
* the pinned karaoke flag reaches the compile payload (``video_compile._fold_captions``);
* pin-on-create: a run started before the tenant changes its defaults keeps the old
  values in its stored snapshot; the next run gets the new ones;
* a run SPAWNED BY a run (a ``parent_run_id`` child, or a course lesson handed the
  course run's snapshot) inherits the parent's pinned defaults — including "none" —
  after the tenant changed them, and the failed-parent Restart check agrees.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid7

import pytest
from sqlalchemy import select, update

import src.services.blueprints  # noqa: F401  # registers blueprints
from src.config import get_settings
from src.database.models import Deliverable, Run
from src.database.repositories.admin_repository import AdminRepository
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.tenant_defaults_repository import TenantDefaultsRepository
from src.services.blueprints.registry import get_registry
from src.services.run_engine import RunEngine
from src.services.runs.run_context import RunContext
from src.services.runs.run_setup import build_config_snapshot, order_fingerprint
from src.services.stages.base import StageSkippedError
from src.services.stages.demo_video.retention_plan import RetentionPlanStage
from src.services.tenant_defaults import TENANT_DEFAULTS_KEY

pytestmark = pytest.mark.asyncio

_FULL: dict[str, Any] = {
    "voice": {"voice_id": "bf_emma", "speed": 1.2},
    "brand": {"primary_color": "#112233", "font_family": "Inter"},
    "captions": {"mode": "burned_in", "karaoke": True},
    "voice_direction": {"tone": "warm"},
    "retention_mode": "auto",
    "model_tier": {"timing_director": "tier-high"},
}
_NULL: dict[str, Any] = dict.fromkeys(_FULL)


async def _tenant_project(
    session,
    *,
    brand: dict[str, Any] | None = None,
    voice: dict[str, Any] | None = None,
    brief: dict[str, Any] | None = None,
    type_: str = "demo_video",
) -> tuple[UUID, Any]:
    tenant = uuid7()
    assert await AdminRepository(session).ensure_tenant(tenant, created_via="super_admin")
    proj = await ProjectRepository(session).create_with_deliverables(
        tenant_id=tenant,
        owner_subject_id=uuid7(),
        type_=type_,
        title="p",
        brief=brief or {},
        brand=brand or {},
        voice=voice or {},
        theme={},
        visibility="tenant",
        deliverables=[{"name": "d", "format": "16x9", "captions_mode": "none"}],
    )
    await session.commit()
    return tenant, proj


async def _set_defaults(session, tenant: UUID, values: dict[str, Any]) -> int:
    repo = TenantDefaultsRepository(session)
    current = await repo.get(tenant)
    if current is None:
        row = await repo.create(tenant, values, actor=None)
    else:
        row = await repo.update_if_version(tenant, current.version, values, actor=None)
    assert row is not None
    await session.commit()
    return row.version


_SUBJECT = uuid7()


async def _snapshot(session, tenant: UUID, proj: Any, **kw: Any) -> dict[str, Any]:
    return await build_config_snapshot(
        session,
        proj=proj,
        deliv=proj.deliverables[0],
        tenant_id=tenant,
        subject_id=_SUBJECT,
        config_overrides=kw.pop("config_overrides", None),
        **kw,
    )


def _settled(snap: dict[str, Any]) -> dict[str, Any]:
    """Drop the per-call ids (project/deliverable differ between two projects)."""
    out = dict(snap)
    for key in ("project_id", "tenant_id", "deliverable", "provider_slots"):
        out.pop(key, None)
    return out


# ─── golden ──────────────────────────────────────────────────────────────


async def test_no_row_and_an_all_null_row_both_give_the_deployment_snapshot(db_session) -> None:
    pipeline = get_settings().pipeline
    tenant_a, proj_a = await _tenant_project(db_session, brand={"name": "A"}, voice={"x": 1})
    none = await _snapshot(db_session, tenant_a, proj_a)

    assert TENANT_DEFAULTS_KEY not in none
    assert "captions_karaoke" not in none
    assert none["brand"] == {"name": "A"}
    assert none["voice"]["x"] == 1 and "voice_id" not in none["voice"]
    assert none["model_tier"] == {"timing_director": pipeline.timing_director_model_tier}
    if pipeline.retention_default_enabled:
        assert none["direction"]["retention"] == {"mode": "critique_only"}
    if pipeline.voice_direction_default_enabled:
        assert none["voice"]["direction"] == {"tone": "confident"}

    tenant_b, proj_b = await _tenant_project(db_session, brand={"name": "A"}, voice={"x": 1})
    assert await _set_defaults(db_session, tenant_b, _NULL) == 1
    null = await _snapshot(db_session, tenant_b, proj_b)
    assert null.pop(TENANT_DEFAULTS_KEY) == {"version": 1}
    assert _settled(null) == _settled(none), "an all-NULL row must equal no row"


# ─── each default overrides the deployment value ─────────────────────────


async def test_every_tenant_default_reaches_the_snapshot(db_session, monkeypatch) -> None:
    pipeline = get_settings().pipeline
    monkeypatch.setattr(pipeline, "retention_default_enabled", True)
    monkeypatch.setattr(pipeline, "voice_direction_default_enabled", True)
    monkeypatch.setattr(pipeline, "captions_karaoke", False)
    monkeypatch.setattr(pipeline, "timing_director_model_tier", None)
    tenant, proj = await _tenant_project(db_session)
    await _set_defaults(db_session, tenant, _FULL)

    snap = await _snapshot(db_session, tenant, proj)
    assert snap[TENANT_DEFAULTS_KEY] == {"version": 1, **_FULL}
    assert snap["brand"] == {"primary_color": "#112233", "font_family": "Inter"}
    assert snap["voice"]["voice_id"] == "bf_emma"
    assert snap["voice"]["speed"] == 1.2
    assert snap["voice"]["direction"] == {"tone": "warm"}
    assert snap["direction"]["retention"] == {"mode": "auto"}
    assert "retention_weights" in snap, "a seeded retention block still resolves weights"
    assert snap["captions_karaoke"] is True
    assert snap["model_tier"] == {"timing_director": "tier-high"}


async def test_explicit_project_brief_and_run_values_beat_the_tenant(
    db_session, monkeypatch
) -> None:
    pipeline = get_settings().pipeline
    monkeypatch.setattr(pipeline, "retention_default_enabled", True)
    monkeypatch.setattr(pipeline, "voice_direction_default_enabled", True)
    tenant, proj = await _tenant_project(
        db_session,
        brand={"primary_color": "#000000", "secondary_color": None},
        voice={"voice_id": "am_adam"},
        brief={"voice": {"direction": {"tone": "brisk"}}},
    )
    await _set_defaults(
        db_session, tenant, {**_FULL, "brand": {**_FULL["brand"], "secondary_color": "#ffffff"}}
    )
    snap = await _snapshot(db_session, tenant, proj)
    # Project keys win; tenant keys fill the gaps; a project None does not erase one.
    assert snap["brand"] == {
        "primary_color": "#000000",
        "secondary_color": "#ffffff",
        "font_family": "Inter",
    }
    assert snap["voice"]["voice_id"] == "am_adam"
    assert snap["voice"]["speed"] == 1.2
    assert snap["voice"]["direction"] == {"tone": "brisk"}, "brief-authored tone wins"

    run = await _snapshot(
        db_session,
        tenant,
        proj,
        config_overrides={
            "direction": {"retention": {"mode": "critique_only"}},
            "captions_karaoke": False,
            "brand": {"name": "override"},
        },
    )
    assert run["direction"]["retention"] == {"mode": "critique_only"}
    assert run["captions_karaoke"] is False
    assert run["brand"] == {"name": "override"}, "a run override replaces the block"


async def test_overrides_cannot_forge_the_version_pin(db_session) -> None:
    tenant, proj = await _tenant_project(db_session)
    snap = await _snapshot(
        db_session, tenant, proj, config_overrides={"tenant_defaults": {"version": 9, **_FULL}}
    )
    assert TENANT_DEFAULTS_KEY not in snap
    assert "voice_id" not in snap["voice"]


# ─── retention ───────────────────────────────────────────────────────────


async def test_retention_off_seeds_nothing_and_the_planner_skips(db_session, monkeypatch) -> None:
    monkeypatch.setattr(get_settings().pipeline, "retention_default_enabled", True)
    tenant, proj = await _tenant_project(db_session)
    await _set_defaults(db_session, tenant, {**_NULL, "retention_mode": "off"})
    snap = await _snapshot(db_session, tenant, proj)
    assert "retention" not in (snap.get("direction") or {})
    assert "retention_weights" not in snap

    wm = AsyncMock()
    stage = RetentionPlanStage(client=wm)
    lines = [{"id": "l1", "text": "Hello there.", "seconds": 3.0, "clip": None}]
    ctx = RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot=snap,
        prior_run_outputs={
            "demo_video_gen": {
                "lines": lines,
                "bdd_job_id": "",
                "features_generated": 0,
                "package_url": "",
            }
        },
    )
    with pytest.raises(StageSkippedError, match=r"no direction\.retention"):
        await stage.handle(stage.input_model(), ctx)
    wm.message.assert_not_called()


async def test_the_retention_kill_switch_stays_global(db_session, monkeypatch) -> None:
    monkeypatch.setattr(get_settings().pipeline, "retention_default_enabled", False)
    monkeypatch.setattr(get_settings().pipeline, "voice_direction_default_enabled", False)
    tenant, proj = await _tenant_project(db_session)
    await _set_defaults(db_session, tenant, _FULL)
    snap = await _snapshot(db_session, tenant, proj)
    assert "retention" not in (snap.get("direction") or {})
    assert "direction" not in snap["voice"]


# ─── captions ────────────────────────────────────────────────────────────


async def test_new_deliverable_captions_mode_resolves_explicit_then_tenant_then_none(
    client, db_session_factory
) -> None:
    tenant = UUID(client.tenant_id)

    async def create(**deliv: Any) -> str:
        r = await client.post(
            "/api/projects",
            json={
                "type": "generic",
                "title": "t",
                "brief": {},
                "deliverables": [{"name": "d", "format": "16x9", **deliv}],
            },
        )
        assert r.status_code == 201, r.text
        return r.json()["deliverables"][0]["id"]

    async def mode(deliverable_id: str) -> str:
        async with db_session_factory() as s:
            return (
                await s.execute(
                    select(Deliverable.captions_mode).where(Deliverable.id == UUID(deliverable_id))
                )
            ).scalar_one()

    assert await mode(await create()) == "none", "no row → deployment default"
    async with db_session_factory() as s:
        await AdminRepository(s).ensure_tenant(tenant, created_via="super_admin")
        await s.commit()
        await _set_defaults(s, tenant, {**_NULL, "captions": {"mode": "overlay"}})
    assert await mode(await create()) == "overlay", "omitted → tenant default"
    assert await mode(await create(captions_mode="none")) == "none", "explicit wins"
    assert await mode(await create(captions_mode="burned")) == "burned_in"


# ─── pin on create ───────────────────────────────────────────────────────


async def _start(
    session_factory,
    tenant: UUID,
    proj: Any,
    *,
    parent_run_id: UUID | None = None,
    inherit_defaults_from: dict[str, Any] | None = None,
) -> dict[str, Any]:
    engine = RunEngine(
        session_factory=session_factory, registry=get_registry(), dispatch=AsyncMock()
    )
    run = await engine.start_run(
        deliverable_id=proj.deliverables[0].id,
        tenant_id=tenant,
        subject_id=uuid7(),
        parent_run_id=parent_run_id,
        pause_after_stages=[],
        inherit_defaults_from=inherit_defaults_from,
    )
    snap = await _stored(session_factory, run.id)
    snap["_run_id"] = run.id
    return snap


async def _stored(session_factory, run_id: UUID) -> dict[str, Any]:
    async with session_factory() as s:
        row = (await s.execute(select(Run).where(Run.id == run_id))).scalar_one()
        return dict(row.config_snapshot)


async def test_a_created_run_keeps_its_defaults_when_the_tenant_changes_them(
    db_session_factory, monkeypatch
) -> None:
    monkeypatch.setattr(get_settings().pipeline, "voice_direction_default_enabled", True)
    async with db_session_factory() as s:
        tenant, proj = await _tenant_project(s)
        await _set_defaults(s, tenant, _FULL)
    first = await _start(db_session_factory, tenant, proj)
    assert first[TENANT_DEFAULTS_KEY]["version"] == 1
    assert first["voice"]["voice_id"] == "bf_emma"

    async with db_session_factory() as s:
        v2 = await _set_defaults(
            s,
            tenant,
            {
                **_FULL,
                "voice": {"voice_id": "am_adam"},
                "voice_direction": {"tone": "calm"},
                "model_tier": {"timing_director": "tier-medium"},
            },
        )
    assert v2 == 2

    deliverable_id = proj.deliverables[0].id
    async with db_session_factory() as s:
        stored_rows = (
            (await s.execute(select(Run).where(Run.deliverable_id == deliverable_id)))
            .scalars()
            .all()
        )
    assert len(stored_rows) == 1
    stored = dict(stored_rows[0].config_snapshot)
    assert stored == {k: v for k, v in first.items() if k != "_run_id"}, (
        "the existing run's snapshot must not move"
    )
    assert stored["voice"]["voice_id"] == "bf_emma"
    assert stored["voice"]["direction"] == {"tone": "warm"}
    assert stored["model_tier"] == {"timing_director": "tier-high"}

    second = await _start(db_session_factory, tenant, proj)
    assert second[TENANT_DEFAULTS_KEY]["version"] == 2
    assert second["voice"]["voice_id"] == "am_adam"
    assert second["voice"]["direction"] == {"tone": "calm"}
    assert second["model_tier"] == {"timing_director": "tier-medium"}
    # Changed VALUES are a changed order; the version pin alone is not.
    first.pop("_run_id")
    second.pop("_run_id")
    assert order_fingerprint(first) != order_fingerprint(second)
    bumped = {**first, TENANT_DEFAULTS_KEY: {"version": 7}}
    assert order_fingerprint(bumped) == order_fingerprint(first)


# ─── runs spawned by a run inherit the parent's pin ──────────────────────


_CHANGED: dict[str, Any] = {
    **_FULL,
    "voice": {"voice_id": "am_adam"},
    "voice_direction": {"tone": "calm"},
    "captions": {"karaoke": False},
    "model_tier": {"timing_director": "tier-medium"},
}


async def _finish(session_factory, run_id: UUID) -> None:
    """A child may only be spawned from a stopped (paused/completed/failed/cancelled) parent."""
    async with session_factory() as s:
        await s.execute(update(Run).where(Run.id == run_id).values(status="failed"))
        await s.commit()


def _values(snap: dict[str, Any]) -> tuple[Any, ...]:
    return (
        snap["voice"].get("voice_id"),
        (snap["voice"].get("direction") or {}).get("tone"),
        snap.get("captions_karaoke"),
        snap["model_tier"]["timing_director"],
        snap.get(TENANT_DEFAULTS_KEY),
    )


async def test_a_child_run_inherits_its_parents_defaults_not_the_tenants_current(
    db_session_factory, monkeypatch
) -> None:
    monkeypatch.setattr(get_settings().pipeline, "voice_direction_default_enabled", True)
    async with db_session_factory() as s:
        tenant, proj = await _tenant_project(s, type_="generic")
        await _set_defaults(s, tenant, _FULL)
    parent = await _start(db_session_factory, tenant, proj)
    assert _values(parent)[:4] == ("bf_emma", "warm", True, "tier-high")

    async with db_session_factory() as s:
        assert await _set_defaults(s, tenant, _CHANGED) == 2

    await _finish(db_session_factory, parent["_run_id"])
    child = await _start(db_session_factory, tenant, proj, parent_run_id=parent["_run_id"])
    assert _values(child) == _values(parent), "a parent_run_id child keeps the parent pin"
    # Control: a run started fresh reads the tenant's current defaults.
    fresh = await _start(db_session_factory, tenant, proj)
    assert _values(fresh)[:4] == ("am_adam", "calm", False, "tier-medium")


async def test_a_course_lesson_inherits_the_course_runs_defaults(
    db_session_factory, monkeypatch
) -> None:
    """Lessons are spawned with parent_run_id=None (another project's deliverable) and
    get the course run's snapshot instead — the compile_lessons call shape."""
    monkeypatch.setattr(get_settings().pipeline, "voice_direction_default_enabled", True)
    async with db_session_factory() as s:
        tenant, course = await _tenant_project(s, type_="generic")
        await _set_defaults(s, tenant, _FULL)
        lesson = await ProjectRepository(s).create_with_deliverables(
            tenant_id=tenant,
            owner_subject_id=uuid7(),
            type_="generic",
            title="lesson",
            brief={},
            brand={},
            voice={},
            theme={},
            visibility="tenant",
            parent_project_id=course.id,
            deliverables=[{"name": "d", "format": "16x9", "captions_mode": "none"}],
        )
        await s.commit()
    course_run = await _start(db_session_factory, tenant, course)
    async with db_session_factory() as s:
        await _set_defaults(s, tenant, _CHANGED)

    child = await _start(db_session_factory, tenant, lesson, inherit_defaults_from=course_run)
    assert _values(child) == _values(course_run)


async def test_a_parent_that_pinned_no_defaults_passes_none_down(
    db_session_factory, monkeypatch
) -> None:
    pipeline = get_settings().pipeline
    monkeypatch.setattr(pipeline, "voice_direction_default_enabled", True)
    async with db_session_factory() as s:
        tenant, proj = await _tenant_project(s, type_="generic")
    parent = await _start(db_session_factory, tenant, proj)
    assert TENANT_DEFAULTS_KEY not in parent
    async with db_session_factory() as s:
        await _set_defaults(s, tenant, _FULL)  # the tenant adds defaults mid-run

    await _finish(db_session_factory, parent["_run_id"])
    child = await _start(db_session_factory, tenant, proj, parent_run_id=parent["_run_id"])
    assert TENANT_DEFAULTS_KEY not in child
    assert child["voice"].get("voice_id") is None
    assert child["voice"]["direction"] == {"tone": "confident"}
    assert "captions_karaoke" not in child
    assert child["model_tier"] == {"timing_director": pipeline.timing_director_model_tier}


async def test_restart_check_rebuilds_with_the_parents_pin(db_session_factory) -> None:
    """The failed-parent Restart compares the parent to the snapshot its CHILD would
    get; with the inherited pin a tenant-defaults change is not an order change."""
    async with db_session_factory() as s:
        tenant, proj = await _tenant_project(s, type_="generic")
        await _set_defaults(s, tenant, _FULL)
        parent = await _snapshot(s, tenant, proj)
        await _set_defaults(s, tenant, _CHANGED)
        would_be = await _snapshot(s, tenant, proj, inherit_defaults_from=parent)
        live = await _snapshot(s, tenant, proj)
    assert order_fingerprint(would_be) == order_fingerprint(parent)
    assert order_fingerprint(live) != order_fingerprint(parent)
