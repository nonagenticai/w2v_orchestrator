"""P3 WP3 — a run pins its provider slots + model tiers at start, against real postgres.

* ``build_config_snapshot`` writes ``provider_slots`` (every registry slot) and
  ``model_tier`` AFTER the override merge — a caller cannot inject either;
* both keys are run-scoped, so a legacy (pre-P3) parent fingerprints equal to its
  would-be child;
* ``start_run(inherit_slots=…)`` keeps a parent's pins for system-spawned children;
  without it the run resolves the tenant's CURRENT connection;
* the adapters bridge WP1's repository + keyring to the resolver end to end.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid7

import pytest
from sqlalchemy import select

import src.services.blueprints  # noqa: F401  # registers blueprints
from src.config import clear_settings_cache, get_settings
from src.database.models import Run
from src.database.repositories.admin_repository import AdminRepository
from src.database.repositories.connections_repository import ConnectionsRepository
from src.database.repositories.project_repository import ProjectRepository
from src.services.blueprints.registry import get_registry
from src.services.providers import PINNED_SLOT_NAMES, resolve_slot
from src.services.providers.adapters import build_resolver_deps
from src.services.providers.crypto import Keyring
from src.services.run_engine import RunEngine
from src.services.runs.run_setup import build_config_snapshot, order_fingerprint

pytestmark = pytest.mark.asyncio

SECRET = "sk-pinned-0001"  # pragma: allowlist secret


@pytest.fixture
def keyring(monkeypatch: pytest.MonkeyPatch) -> Iterator[Keyring]:
    key = os.urandom(32)
    monkeypatch.setenv("PROVIDER_SECRET_KEYS", f"k1:{base64.b64encode(key).decode()}")
    clear_settings_cache()
    yield Keyring([("k1", key)])
    clear_settings_cache()


async def _tenant_project(session) -> tuple[UUID, Any]:
    tenant = uuid7()
    assert await AdminRepository(session).ensure_tenant(tenant, created_via="super_admin")
    proj = await ProjectRepository(session).create_with_deliverables(
        tenant_id=tenant,
        owner_subject_id=uuid7(),
        type_="promo",
        title="p",
        brief={},
        brand={},
        voice={},
        theme={},
        visibility="tenant",
        deliverables=[{"name": "d", "format": "16x9", "captions_mode": "none"}],
    )
    await session.commit()
    return tenant, proj


async def _connect(session, tenant: UUID, ring: Keyring, model: str) -> tuple[UUID, int]:
    conn, ver = await ConnectionsRepository(session).create_version(
        tenant_id=tenant,
        slot="llm",
        kind="openai-compatible",
        base_url="https://1.1.1.1/v1",
        model=model,
        secret=SECRET,
        keyring=ring,
    )
    await session.commit()
    return conn.id, ver.version


async def _snapshot(session, tenant: UUID, proj: Any, **kw: Any) -> dict[str, Any]:
    return await build_config_snapshot(
        session,
        proj=proj,
        deliv=proj.deliverables[0],
        tenant_id=tenant,
        subject_id=uuid7(),
        config_overrides=kw.pop("config_overrides", None),
        **kw,
    )


async def test_new_run_pins_deployment_refs_for_every_slot(db_session) -> None:
    tenant, proj = await _tenant_project(db_session)
    snap = await _snapshot(db_session, tenant, proj)
    pins = snap["provider_slots"]
    # Grant slots (publish.youtube) are not pinned per run (P5-3).
    assert set(pins) == set(PINNED_SLOT_NAMES)
    assert "publish.youtube" not in pins
    assert all(p["source"] == "deployment" and p["connection_id"] is None for p in pins.values())
    assert pins["llm"]["kind"] == "nonagentic-wm"
    assert pins["tts"]["kind"] == "nonagentic"
    assert pins["device_farm"]["kind"] == "local"
    assert pins["vlm"]["kind"] == "platform"


async def test_model_tier_is_pinned_from_settings(db_session, monkeypatch) -> None:
    monkeypatch.setattr(get_settings().pipeline, "timing_director_model_tier", "tier-high")
    tenant, proj = await _tenant_project(db_session)
    snap = await _snapshot(db_session, tenant, proj)
    assert snap["model_tier"] == {"timing_director": "tier-high"}


async def test_overrides_cannot_inject_provider_slots_or_model_tier(db_session) -> None:
    tenant, proj = await _tenant_project(db_session)
    forged = {
        "llm": {
            "source": "tenant",
            "kind": "openai-compatible",
            "connection_id": str(uuid7()),  # another tenant's connection
            "version": 1,
        }
    }
    snap = await _snapshot(
        db_session,
        tenant,
        proj,
        config_overrides={"provider_slots": forged, "model_tier": {"timing_director": "x"}},
    )
    assert snap["provider_slots"]["llm"]["source"] == "deployment"
    assert snap["provider_slots"]["llm"]["kind"] == "nonagentic-wm"
    assert snap["model_tier"] == {
        "timing_director": get_settings().pipeline.timing_director_model_tier
    }


async def test_tenant_connection_is_pinned_by_exact_version(db_session, keyring) -> None:
    tenant, proj = await _tenant_project(db_session)
    conn_id, version = await _connect(db_session, tenant, keyring, "m-1")
    snap = await _snapshot(db_session, tenant, proj)
    assert snap["provider_slots"]["llm"] == {
        "source": "tenant",
        "kind": "openai-compatible",
        "connection_id": str(conn_id),
        "version": version,
    }
    # And the adapters resolve that pin end to end (joined read + real keyring).
    deps = build_resolver_deps(db_session, get_settings())
    resolved = await resolve_slot(
        deps.repo,
        deps.decryptor,
        tenant,
        "llm",
        run_snapshot=snap,
        policy=deps.policy,
        posture=deps.posture,
    )
    assert resolved.model == "m-1" and resolved.secret is not None
    assert resolved.secret.get_secret_value() == SECRET


async def test_legacy_snapshot_order_fingerprint_is_unchanged_by_the_pins() -> None:
    legacy = {"project_id": "p", "brief": {"a": 1}, "deliverable": {"id": "d"}}
    pinned = {
        **legacy,
        "provider_slots": {"llm": {"source": "deployment", "kind": "nonagentic-wm"}},
        "model_tier": {"timing_director": None},
    }
    assert order_fingerprint(pinned) == order_fingerprint(legacy)


async def _start(
    session_factory, tenant: UUID, proj: Any, inherit: dict[str, Any] | None
) -> dict[str, Any]:
    engine = RunEngine(
        session_factory=session_factory, registry=get_registry(), dispatch=AsyncMock()
    )
    run = await engine.start_run(
        deliverable_id=proj.deliverables[0].id,
        tenant_id=tenant,
        subject_id=uuid7(),
        parent_run_id=None,
        pause_after_stages=[],
        inherit_slots=inherit,
    )
    async with session_factory() as s:
        row = (await s.execute(select(Run).where(Run.id == run.id))).scalar_one()
        return dict(row.config_snapshot)


async def test_child_inherits_parent_pins_while_a_fresh_run_resolves_live(
    db_session_factory, keyring
) -> None:
    async with db_session_factory() as s:
        tenant, proj = await _tenant_project(s)
        _conn_id, v1 = await _connect(s, tenant, keyring, "m-1")
    parent = await _start(db_session_factory, tenant, proj, None)
    assert parent["provider_slots"]["llm"]["version"] == v1

    # The tenant moves its LLM to a new version while the parent is still running.
    async with db_session_factory() as s:
        _, v2 = await _connect(s, tenant, keyring, "m-2")
    assert v2 == v1 + 1

    child = await _start(db_session_factory, tenant, proj, parent["provider_slots"])
    assert child["provider_slots"]["llm"]["version"] == v1, "system child keeps parent pin"
    assert child["provider_slots"] == parent["provider_slots"]

    fresh = await _start(db_session_factory, tenant, proj, None)
    assert fresh["provider_slots"]["llm"]["version"] == v2, "a new run resolves live"


async def test_child_of_a_legacy_parent_pins_fresh(db_session_factory) -> None:
    async with db_session_factory() as s:
        tenant, proj = await _tenant_project(s)
    child = await _start(db_session_factory, tenant, proj, {})
    assert set(child["provider_slots"]) == set(PINNED_SLOT_NAMES)


async def test_child_with_every_pinned_slot_inherited_does_not_re_pin(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child holding the parent's FULL pin set must not resolve anything live.

    The "anything left to pin?" check iterated ``SLOT_NAMES``, which includes the
    grant slot ``publish.youtube`` — never pinned, so never inherited — so EVERY
    inheriting child re-ran ``pin_slots`` (a live connection read per child)."""
    import src.services.runs.run_setup as run_setup

    tenant, proj = await _tenant_project(db_session)
    parent = await _snapshot(db_session, tenant, proj)
    spy = AsyncMock(side_effect=AssertionError("an inheriting child must not re-pin"))
    monkeypatch.setattr(run_setup, "pin_slots", spy)

    child = await _snapshot(db_session, tenant, proj, inherit_slots=parent["provider_slots"])

    spy.assert_not_awaited()
    assert child["provider_slots"] == parent["provider_slots"]
    assert "publish.youtube" not in child["provider_slots"]
