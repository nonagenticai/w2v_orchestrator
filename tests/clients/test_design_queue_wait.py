"""html_render queue time must not be charged to the design generation budget.

The defect these pin, measured live on 2026-08-10 (website_review run
``019fea75-98ee-70d5-84bb-044e12fddd49``): design job ``019fec3f-…`` was created
at 15:17:18Z and no worker claimed it until 15:40:26Z — 23 minutes in
html_render's ``pending`` claim queue. ``website_redesign.wait``'s 8 x 590s
budget starts at SUBMIT, so it expired at 16:38:32Z having permitted only 58 min
of actual generation. The service finished that very job successfully at
16:47:39Z (``completed_at``, read live off the API). The run failed and a
completed design was thrown away.

``wait_until_started`` is the pre-phase that fixes it: block for free while the
job is queued, then let the generation budget start when generation does.

No socket opens — ``_request`` and ``asyncio.sleep`` are stubbed.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.clients.design_client import DesignClient
from src.services.stages.shared.design_queue import await_design_start


def _client(monkeypatch: pytest.MonkeyPatch, statuses: list[str]) -> tuple[DesignClient, list[str]]:
    """A client whose status endpoint yields ``statuses`` in order, then repeats the last."""
    client = DesignClient(base_url="http://design.test", license_token="t")
    seen: list[str] = []
    remaining = list(statuses)

    async def _fake_request(method: str, endpoint: str, **_: Any) -> dict[str, Any]:
        seen.append(endpoint)
        status = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return {"job_id": "rd-1", "status": status}

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(client, "_request", _fake_request)
    monkeypatch.setattr("src.clients.design_client.asyncio.sleep", _no_sleep)
    return client, seen


async def test_returns_immediately_when_a_worker_already_has_the_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, seen = _client(monkeypatch, ["processing"])

    queued_for = await client.wait_until_started("rd-1")

    assert queued_for >= 0.0
    assert len(seen) == 1  # one status read, no waiting


async def test_blocks_while_pending_and_returns_once_claimed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Three polls queued, then a worker claims it — the shape of the live incident.
    client, seen = _client(monkeypatch, ["pending", "pending", "pending", "processing"])

    await client.wait_until_started("rd-1")

    assert len(seen) == 4
    assert seen[0] == "/api/designs/jobs/rd-1"


async def test_a_terminal_status_does_not_hold_the_queue_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``failed`` leaves ``pending``, so this returns — surfacing WHY is wait_for_design's job.

    Holding here would bury the service's own error text behind a 30-minute wait.
    """
    client, seen = _client(monkeypatch, ["failed"])

    await client.wait_until_started("rd-1")

    assert len(seen) == 1


async def test_a_never_claimed_job_times_out_naming_the_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = _client(monkeypatch, ["pending"])

    with pytest.raises(TimeoutError) as exc:
        await client.wait_until_started("rd-1", timeout=0.05, interval=0.01)

    assert "rd-1" in str(exc.value)
    assert "never claimed" in str(exc.value)


async def test_helper_waits_out_the_queue_before_the_caller_polls_for_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression itself: generation polling must not begin while still queued.

    Ordering is the assertion — with the pre-phase removed, ``wait_for_design``
    is entered while the job is ``pending`` and every second of that queue time
    comes out of the generation budget.
    """
    client, _ = _client(monkeypatch, ["pending", "pending", "processing"])
    order: list[str] = []

    real_wait_until_started = client.wait_until_started

    async def _tracked_wait_until_started(job_id: str, **kwargs: Any) -> float:
        order.append("queue")
        return await real_wait_until_started(job_id, **kwargs)

    async def _fake_wait_for_design(job_id: str, **_: Any) -> object:
        order.append("generate")
        return {"job_id": job_id, "status": "completed"}

    monkeypatch.setattr(client, "wait_until_started", _tracked_wait_until_started)
    monkeypatch.setattr(client, "wait_for_design", _fake_wait_for_design)

    await await_design_start(client, "rd-1", "website_redesign")
    await client.wait_for_design("rd-1")

    assert order == ["queue", "generate"]
