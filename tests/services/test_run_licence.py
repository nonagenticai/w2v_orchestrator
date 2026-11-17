"""A run's cost events carry the run TENANT's own licence (``run_licence``).

The run snapshot's ``license_id`` becomes every S2S hop's ``X-License-Id`` and so
the ``license_id`` of each ``billing:costs`` event downstream. These fail if the
snapshot goes back to copying the request ContextVar, which is EMPTY on the human
(membership) lane — the hole that let one umbrella licence absorb every tenant's
spend.
"""

from __future__ import annotations

import logging
from uuid import UUID, uuid4

import pytest
from w2v_common.auth.identity_context import clear_identity_context

from src.database.models import Tenant
from src.observability.metrics import run_licence_unattributed_total
from src.services.run_licence import resolve_run_licence

pytestmark = pytest.mark.asyncio


def _unattributed(reason: str) -> float:
    return run_licence_unattributed_total.labels(reason=reason)._value.get()


async def _tenant(db_session, licence_ref: UUID | None) -> UUID:
    tid = uuid4()
    db_session.add(Tenant(id=tid, name=f"t-{tid.hex[:8]}", licence_ref=licence_ref))
    await db_session.flush()
    return tid


async def test_the_tenants_own_licence_wins(db_session) -> None:
    ref = uuid4()
    tid = await _tenant(db_session, ref)
    assert await resolve_run_licence(db_session, tid, "") == str(ref)
    # ...even over a licence the request happened to carry.
    assert await resolve_run_licence(db_session, tid, str(uuid4())) == str(ref)


async def test_no_licence_ref_uses_the_tenant_bound_request_licence(db_session) -> None:
    tid, bound = await _tenant(db_session, None), uuid4()
    assert await resolve_run_licence(db_session, tid, str(bound)) == str(bound)


@pytest.mark.parametrize("context", [""])
async def test_no_licence_at_all_is_an_honest_blank_counted_and_logged(
    db_session, caplog, context: str | None
) -> None:
    tid = await _tenant(db_session, None)
    before = _unattributed("no_licence_ref")
    with caplog.at_level(logging.ERROR, logger="w2v_orchestrator.run_licence"):
        assert await resolve_run_licence(db_session, tid, context) == ""
    assert _unattributed("no_licence_ref") == before + 1
    assert str(tid) in caplog.text


async def test_the_nil_licence_ref_is_never_a_licence(db_session) -> None:
    tid = await _tenant(db_session, UUID(int=0))
    assert await resolve_run_licence(db_session, tid, "") == ""


async def test_an_unregistered_tenant_is_counted_as_such(db_session) -> None:
    before = _unattributed("no_tenant_row")
    assert await resolve_run_licence(db_session, uuid4(), "") == ""
    assert _unattributed("no_tenant_row") == before + 1


async def test_start_run_snapshots_the_tenants_licence_on_the_membership_lane(
    db_session,
) -> None:
    """End to end through ``RunEngine.start_run``: a person started the run, so the
    request ContextVar holds no licence — the persisted snapshot must still carry
    the tenant's ``licence_ref`` (what the dispatcher re-binds as X-License-Id)."""
    from src.database.repositories.project_repository import ProjectRepository
    from src.services.blueprints.registry import get_registry
    from src.services.run_engine import RunEngine

    clear_identity_context()  # the membership lane binds license_id=None
    ref = uuid4()
    tenant = await _tenant(db_session, ref)
    subject = uuid4()
    proj = await ProjectRepository(db_session).create_with_deliverables(
        tenant_id=tenant,
        owner_subject_id=subject,
        type_="generic",
        title="x",
        brief={"angle": "growth", "audience": "any", "target_duration_seconds": 30},
        brand={},
        voice={"descriptor": "crisp"},
        theme={},
        visibility="tenant",
        deliverables=[
            {"name": "d", "format": "16x9", "captions_mode": "none", "target_duration_seconds": 30}
        ],
    )
    await db_session.commit()

    class _SessionCtx:
        async def __aenter__(self):  # type: ignore[no-untyped-def]
            return db_session

        async def __aexit__(self, *exc):  # type: ignore[no-untyped-def]
            return False

    async def _dispatch(_run_id: UUID) -> None:
        return None

    engine = RunEngine(
        session_factory=lambda: _SessionCtx(), registry=get_registry(), dispatch=_dispatch
    )
    run = await engine.start_run(
        deliverable_id=proj.deliverables[0].id,
        tenant_id=tenant,
        subject_id=subject,
        parent_run_id=None,
        pause_after_stages=[],
    )
    assert run.config_snapshot["license_id"] == str(ref)
