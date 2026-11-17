"""The website_review minimum-order floor is enforced at RUN START, not only at create.

#1009 added ``validate_website_review_order`` to ``RunEngine.start_run`` (prod
01a09f82 ordered 10s and shipped 15.1s), but only the create-door and the pure
function were tested. A deliverable whose order was shortened after creation (or a
row that predates the create-door check) reaches the run door, which must 400
before any run row exists or a single paid stage runs.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy import func, select, update

pytestmark = pytest.mark.asyncio


async def _website_review_deliverable(client) -> UUID:
    r = await client.post(
        "/api/projects",
        json={
            "type": "website_review",
            "title": "review",
            "brief": {"source_url": "https://example.com"},
            "deliverables": [{"name": "d", "format": "16x9", "target_duration_seconds": 60}],
        },
    )
    assert r.status_code == 201, r.text
    detail = await client.get(f"/api/projects/{r.json()['id']}")
    return UUID(detail.json()["deliverables"][0]["id"])


async def _shorten_order(db_session, deliverable_id: UUID, seconds: int) -> None:
    from src.database.models import Deliverable

    await db_session.execute(
        update(Deliverable)
        .where(Deliverable.id == deliverable_id)
        .values(target_duration_seconds=seconds)
    )
    await db_session.commit()


async def _run_count(db_session, deliverable_id: UUID) -> int:
    from src.database.models import Run

    stmt = select(func.count()).select_from(Run).where(Run.deliverable_id == deliverable_id)
    return int((await db_session.execute(stmt)).scalar_one())


async def test_run_start_rejects_an_order_below_the_physical_floor(client, db_session) -> None:
    deliv = await _website_review_deliverable(client)
    await _shorten_order(db_session, deliv, 1)

    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})

    assert r.status_code == 400, r.text
    assert "shortest possible review" in r.text
    assert await _run_count(db_session, deliv) == 0


async def test_run_start_accepts_an_order_above_the_floor(client, db_session) -> None:
    deliv = await _website_review_deliverable(client)

    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})

    assert r.status_code == 201, r.text
    assert await _run_count(db_session, deliv) == 1


# ── the 10s floor at the CREATE door, as a typed 400 ────────────────────────────
#
# Owner decision 2026-09-15: the floor is 10s everywhere. The server used to take 10s
# (and 2.4s with title cards off) while the wizard refused under 30s. It also refused
# through a pydantic ``model_validator``, so the client got a **422** whose body was a
# validator traceback — neither the floor nor the ordered value reached the operator.


def _create_body(seconds: float) -> dict:
    return {
        "type": "website_review",
        "title": "review",
        "brief": {"source_url": "https://example.com"},
        "deliverables": [{"name": "d", "format": "16x9", "target_duration_seconds": seconds}],
    }


async def test_create_refuses_nine_seconds_with_a_typed_400(client) -> None:
    r = await client.post("/api/projects", json=_create_body(9))

    assert r.status_code == 400, r.text  # NOT pydantic's 422, NOT a 500
    detail = r.json()["detail"]
    assert detail["reason"] == "order_below_floor"
    assert detail["ordered_seconds"] == 9.0
    assert detail["minimum_seconds"] == 10.0
    # The message names the floor AND the ordered value.
    assert "9s" in detail["message"]
    assert "10s" in detail["message"]


async def test_create_accepts_exactly_ten_seconds(client) -> None:
    r = await client.post("/api/projects", json=_create_body(10))

    assert r.status_code == 201, r.text
    detail = await client.get(f"/api/projects/{r.json()['id']}")
    assert detail.json()["deliverables"][0]["target_duration_seconds"] == 10


async def test_run_start_refuses_nine_seconds_with_the_same_typed_400(client, db_session) -> None:
    """The run door answers with the SAME body — an order shortened after creation."""
    deliv = await _website_review_deliverable(client)
    await _shorten_order(db_session, deliv, 9)

    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})

    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert detail["reason"] == "order_below_floor"
    assert detail["ordered_seconds"] == 9.0
    assert detail["minimum_seconds"] == 10.0
    assert await _run_count(db_session, deliv) == 0
