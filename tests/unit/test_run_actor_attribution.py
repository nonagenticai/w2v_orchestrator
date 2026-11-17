"""A dispatched run carries its CALLER's actor, not the orchestrator's own.

w2v-common 0.88 uses ``SERVICE_ACTOR_ID`` only for SERVICE-originated work. A run
executes in a background task (and on the reconciler path with no inbound request
at all), where ``set_identity_context`` clears the actor — so without a snapshotted
actor every stage's WM call and cost event would carry the orchestrator's own sub.

* run created with an actor -> the dispatched stage's inference-router call carries THAT actor
  and fr project, even with a different ``SERVICE_ACTOR_ID`` configured;
* a legacy run (no stored actor) stays actor-less and is COUNTED — it never falls
  back to ``SERVICE_ACTOR_ID``, because it was caller-originated.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest
from w2v_common.auth import (
    ACTOR_ID_HEADER,
    PROJECT_ID_HEADER,
    bind_attribution_from_headers,
    clear_identity_context,
    set_identity_context,
)
from w2v_common.auth.identity_context import (
    ACTOR_MISSING_INBOUND,
    actor_missing_total,
    get_project_id_from_context,
    mark_inbound_request,
    resolve_actor_id,
)
from w2v_common.tenant import TenantSource

from src.auth.run_attribution import (
    ACTOR_ID_KEY,
    ATTRIBUTION_PROJECT_KEY,
    snapshot_run_attribution,
)
from src.auth.tenant_provenance import TENANT_SOURCE_KEY
from tests.support.platform_llm import common_ir_client, use_ir_backend
from tests.unit.test_run_dispatcher_failure import _drive_dispatcher

pytestmark = pytest.mark.asyncio

CALLER_ACTOR = "a1a1a1a1-1111-4111-8111-111111111111"
OWN_SERVICE_ACTOR = "5e5e5e5e-5555-4555-8555-555555555555"
FR_PROJECT = "b2b2b2b2-2222-4222-8222-222222222222"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("SERVICE_ACTOR_ID", OWN_SERVICE_ACTOR)
    use_ir_backend(monkeypatch)
    clear_identity_context()
    yield
    clear_identity_context()


async def _dispatch(snapshot_extra: dict) -> tuple[list[httpx.Request], dict]:
    seen: list[httpx.Request] = []
    observed: dict = {}
    # The real w2v_common client on the ir backend: its pooled client carries the
    # identity hook that stamps X-Actor-Id on the inference-router call.
    wm = common_ir_client(seen)

    async def _stage(*_args: object, **_kwargs: object) -> None:
        observed["actor"] = resolve_actor_id()
        observed["project"] = get_project_id_from_context()
        await wm.message("script-draft", "hello")
        raise RuntimeError("stop after the WM call")

    with patch("src.services.run_dispatcher.RunDispatcher._recompute_project", new=AsyncMock()):
        await _drive_dispatcher(
            executor_side_effect=_stage,
            project_id=uuid4(),
            # A run created from a licence-token request records that provenance;
            # inference-router delegation refuses an unvouched tenant.
            snapshot_extra={TENANT_SOURCE_KEY: TenantSource.LICENSE_TOKEN.value, **snapshot_extra},
        )
    return seen, observed


async def test_snapshot_captures_the_callers_actor_and_project() -> None:
    set_identity_context(uuid4(), "lic", "sub", actor_id=CALLER_ACTOR)
    mark_inbound_request()
    bind_attribution_from_headers({PROJECT_ID_HEADER: FR_PROJECT}, trusted=True)
    assert snapshot_run_attribution() == {
        ACTOR_ID_KEY: CALLER_ACTOR,
        ATTRIBUTION_PROJECT_KEY: FR_PROJECT,
    }


async def test_dispatched_run_carries_the_snapshotted_actor() -> None:
    seen, observed = await _dispatch(
        {ACTOR_ID_KEY: CALLER_ACTOR, ATTRIBUTION_PROJECT_KEY: FR_PROJECT}
    )

    assert observed["actor"] == (CALLER_ACTOR, None)
    assert observed["project"] == FR_PROJECT
    assert len(seen) == 1
    assert seen[0].headers[ACTOR_ID_HEADER] == CALLER_ACTOR
    # Bound inside the run task only — the dispatching context never sees it.
    assert resolve_actor_id() == (OWN_SERVICE_ACTOR, None)


async def test_legacy_run_without_actor_stays_actorless_and_is_counted() -> None:
    counter = actor_missing_total.labels(service="outbound", reason=ACTOR_MISSING_INBOUND)
    before = counter._value.get()

    seen, observed = await _dispatch({})

    assert observed["actor"] == (None, ACTOR_MISSING_INBOUND)
    assert len(seen) == 1
    assert ACTOR_ID_HEADER not in seen[0].headers
    assert seen[0].headers.get(ACTOR_ID_HEADER) != OWN_SERVICE_ACTOR
    assert counter._value.get() >= before + 1
