"""A dispatched run attributes its LLM calls to its OWN w2v project.

``RunDispatcher._execute_run`` binds the run's project (``projects.id`` of the
run's deliverable) with ``set_w2v_project_context``; ``WMClient.message`` then
sends it to inference-router as ``X-Project-Id``. Driven through the real dispatch
task with a real ``w2v_common`` ``WMClient`` on the ir backend over an
``httpx.MockTransport`` (the conftest egress guard blocks every other LLM socket).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import httpx
import pytest
from w2v_common.auth.identity_context import (
    PROJECT_ID_HEADER,
    get_w2v_project_id_from_context,
)
from w2v_common.tenant import TenantSource

from src.auth.tenant_provenance import TENANT_SOURCE_KEY
from tests.support.platform_llm import CHAT_PATH, common_ir_client, use_ir_backend
from tests.unit.test_run_dispatcher_failure import _drive_dispatcher

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _ir(monkeypatch: pytest.MonkeyPatch) -> None:
    use_ir_backend(monkeypatch)


async def _run_with_stage_wm_call(
    project_id: UUID | None,
) -> tuple[list[httpx.Request], list[str]]:
    """Drive one dispatch whose 'stage' makes a WM call; return requests + bound ids."""
    seen: list[httpx.Request] = []
    bound: list[str] = []
    wm = common_ir_client(seen)

    async def _stage(*_args: object, **_kwargs: object) -> None:
        bound.append(get_w2v_project_id_from_context())
        await wm.message("script-draft", "hello")
        raise RuntimeError("stop after the WM call")

    with patch("src.services.run_dispatcher.RunDispatcher._recompute_project", new=AsyncMock()):
        await _drive_dispatcher(
            executor_side_effect=_stage,
            project_id=project_id,
            # A run created from a licence-token request records that provenance;
            # inference-router delegation refuses an unvouched tenant.
            snapshot_extra={TENANT_SOURCE_KEY: TenantSource.LICENSE_TOKEN.value},
        )
    return seen, bound


async def test_dispatched_run_sends_its_w2v_project_as_x_project_id() -> None:
    project_id = uuid4()

    seen, bound = await _run_with_stage_wm_call(project_id)

    assert bound == [str(project_id)]
    assert len(seen) == 1
    assert seen[0].url.path == CHAT_PATH
    assert seen[0].headers[PROJECT_ID_HEADER] == str(project_id)


async def test_each_run_carries_its_own_project_and_none_leaks_to_the_caller() -> None:
    first, second = uuid4(), uuid4()

    seen_a, _ = await _run_with_stage_wm_call(first)
    seen_b, _ = await _run_with_stage_wm_call(second)

    assert seen_a[0].headers[PROJECT_ID_HEADER] == str(first)
    assert seen_b[0].headers[PROJECT_ID_HEADER] == str(second)
    # Bound inside the per-run task only — the dispatching context never sees it.
    assert get_w2v_project_id_from_context() == ""
