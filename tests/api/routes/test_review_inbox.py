"""The run owner's in-app "needs your review" inbox (``/api/review-inbox``).

Prod case: run 01a1770d (promo_v1) paused at its storyboard review checkpoint on
2026-10-26 07:15:21Z and waited 29h because nobody knew. The owner is now told inside
Studio. These pin the contract:

* a notice exists exactly while a run the caller OWNS is paused at a CHECKPOINT;
* never for a manual/unknown pause, another owner, another tenant, or a finished run;
* dismiss hides it once per pause; it returns as the 24h REMINDER, which only a
  dismiss of the reminder itself (``?reminder=true``) hides for good;
* a pause with no ``paused_at`` is shown as the reminder at once (never starved);
* resume re-arms it for the next checkpoint.
"""

from __future__ import annotations

import datetime
from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, update
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes.review_inbox import router as review_inbox_router
from src.database.models import Deliverable, Project, Run
from src.database.repositories.run_repository import RunRepository

pytestmark = pytest.mark.asyncio


def _client(db_session_factory, *, tenant_id: UUID, subject_id: UUID) -> AsyncClient:
    app = FastAPI()
    app.state.session_factory = db_session_factory
    app.include_router(review_inbox_router)

    class _Seed(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = str(tenant_id)
            request.state.subject_id = str(subject_id)
            return await call_next(request)

    app.add_middleware(_Seed)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def _run(
    session_factory,
    *,
    tenant_id: UUID,
    owner: UUID,
    status: str = "paused",
    reason: str | None = "checkpoint",
    paused_hours_ago: float | None = 0.5,
    title: str = "commonpractice-app.nonagentic.ai",
) -> UUID:
    async with session_factory() as s:
        proj = Project(
            id=uuid7(),
            tenant_id=tenant_id,
            owner_subject_id=owner,
            visibility="tenant",
            type="promo",
            title=title,
        )
        s.add(proj)
        deliv = Deliverable(id=uuid7(), project_id=proj.id, name="60s horizontal", format="16x9")
        s.add(deliv)
        await s.flush()
        run = Run(
            id=uuid7(),
            deliverable_id=deliv.id,
            blueprint_id="promo_v1",
            status=status,
            current_stage="promo_storyboard",
            pause_reason=reason,
            paused_at=None
            if paused_hours_ago is None
            else datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=paused_hours_ago),
        )
        s.add(run)
        await s.commit()
        return run.id


async def _inbox(c: AsyncClient) -> dict[str, dict]:
    r = await c.get("/api/review-inbox")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"items"}
    return {i["run_id"]: i for i in body["items"]}


async def test_owner_sees_a_notice_for_a_checkpoint_pause(db_session_factory) -> None:
    tenant, owner = uuid4(), uuid4()
    rid = await _run(db_session_factory, tenant_id=tenant, owner=owner)
    async with _client(db_session_factory, tenant_id=tenant, subject_id=owner) as c:
        items = await _inbox(c)
    item = items[str(rid)]
    assert item["project_title"] == "commonpractice-app.nonagentic.ai"
    assert item["deliverable_name"] == "60s horizontal"
    assert item["stage_name"] == "promo_storyboard"
    assert item["paused_at"] is not None
    assert item["reminder"] is False


@pytest.mark.parametrize(
    ("status", "reason"),
    [("paused", "manual")],
)
async def test_no_notice_unless_waiting_at_a_checkpoint(db_session_factory, status, reason) -> None:
    """A manual or unclassified pause has nothing to review; a run that is not paused
    (even with a stale stamp after a cancel) is not waiting for anyone."""
    tenant, owner = uuid4(), uuid4()
    rid = await _run(
        db_session_factory, tenant_id=tenant, owner=owner, status=status, reason=reason
    )
    async with _client(db_session_factory, tenant_id=tenant, subject_id=owner) as c:
        assert str(rid) not in await _inbox(c)


async def test_only_the_owner_in_the_owners_tenant_sees_it(db_session_factory) -> None:
    tenant, owner = uuid4(), uuid4()
    rid = await _run(db_session_factory, tenant_id=tenant, owner=owner)
    async with _client(db_session_factory, tenant_id=tenant, subject_id=uuid4()) as other:
        assert str(rid) not in await _inbox(other)
        assert (await other.post(f"/api/review-inbox/{rid}/dismiss")).status_code == 404
    async with _client(db_session_factory, tenant_id=uuid4(), subject_id=owner) as foreign:
        assert str(rid) not in await _inbox(foreign)
        assert (await foreign.post(f"/api/review-inbox/{rid}/dismiss")).status_code == 404


async def test_dismiss_hides_it_and_the_24h_reminder_brings_it_back(db_session_factory) -> None:
    tenant, owner = uuid4(), uuid4()
    rid = await _run(db_session_factory, tenant_id=tenant, owner=owner, paused_hours_ago=2)
    async with _client(db_session_factory, tenant_id=tenant, subject_id=owner) as c:
        assert (await c.post(f"/api/review-inbox/{rid}/dismiss")).status_code == 204
        assert str(rid) not in await _inbox(c), "a dismissed notice must stay hidden"
        # Idempotent: dismissing again changes nothing and still answers 204.
        assert (await c.post(f"/api/review-inbox/{rid}/dismiss")).status_code == 204

        # 24h after the pause the reminder resurfaces it, highlighted.
        async with db_session_factory() as s:
            await s.execute(
                update(Run)
                .where(Run.id == rid)
                .values(
                    paused_at=Run.paused_at - datetime.timedelta(hours=23),
                    review_notice_dismissed_at=Run.paused_at - datetime.timedelta(hours=22),
                )
            )
            await s.commit()
        items = await _inbox(c)
        assert items[str(rid)]["reminder"] is True

        # Dismissing the REMINDER hides it for good (for this pause).
        r = await c.post(f"/api/review-inbox/{rid}/dismiss", params={"reminder": "true"})
        assert r.status_code == 204
        assert str(rid) not in await _inbox(c)


async def test_a_stale_tab_dismiss_after_24h_does_not_kill_the_reminder(db_session_factory) -> None:
    """Tab A dismissed the notice early. Tab B loaded before the 24h mark still shows the
    plain (non-reminder) notice; when it dismisses after the mark, the reminder must still
    come back, because nobody has seen it yet."""
    tenant, owner = uuid4(), uuid4()
    rid = await _run(db_session_factory, tenant_id=tenant, owner=owner, paused_hours_ago=25)
    async with db_session_factory() as s:
        await s.execute(
            update(Run)
            .where(Run.id == rid)
            .values(review_notice_dismissed_at=Run.paused_at + datetime.timedelta(hours=1))
        )
        await s.commit()
    async with _client(db_session_factory, tenant_id=tenant, subject_id=owner) as c:
        assert (await _inbox(c))[str(rid)]["reminder"] is True
        # The stale tab's plain dismiss (no ``reminder`` flag).
        assert (await c.post(f"/api/review-inbox/{rid}/dismiss")).status_code == 204
        items = await _inbox(c)
        assert str(rid) in items, "a stale pre-reminder dismiss must not hide the reminder"
        assert items[str(rid)]["reminder"] is True


async def test_a_wait_past_24h_is_flagged_as_a_reminder(db_session_factory) -> None:
    tenant, owner = uuid4(), uuid4()
    rid = await _run(db_session_factory, tenant_id=tenant, owner=owner, paused_hours_ago=29)
    async with _client(db_session_factory, tenant_id=tenant, subject_id=owner) as c:
        assert (await _inbox(c))[str(rid)]["reminder"] is True


async def test_a_pause_without_paused_at_gets_the_reminder(db_session_factory) -> None:
    """No wait start on record: the wait may already be long, so the notice is the
    reminder straight away. Dismissing it is then dismissing the reminder: final."""
    tenant, owner = uuid4(), uuid4()
    rid = await _run(db_session_factory, tenant_id=tenant, owner=owner, paused_hours_ago=None)
    async with _client(db_session_factory, tenant_id=tenant, subject_id=owner) as c:
        item = (await _inbox(c))[str(rid)]
        assert item["paused_at"] is None
        assert item["reminder"] is True
        assert (await c.post(f"/api/review-inbox/{rid}/dismiss")).status_code == 204
        assert str(rid) not in await _inbox(c)


async def test_resume_rearms_the_notice_for_the_next_checkpoint(db_session_factory) -> None:
    tenant, owner = uuid4(), uuid4()
    rid = await _run(db_session_factory, tenant_id=tenant, owner=owner)
    async with _client(db_session_factory, tenant_id=tenant, subject_id=owner) as c:
        assert (await c.post(f"/api/review-inbox/{rid}/dismiss")).status_code == 204
        async with db_session_factory() as s:
            assert await RunRepository(s).mark_resumed(rid)
            await s.commit()
            row = (await s.execute(select(Run).where(Run.id == rid))).scalar_one()
            assert row.review_notice_dismissed_at is None
            assert await RunRepository(s).mark_paused(rid, reason="checkpoint")
            await s.commit()
        assert str(rid) in await _inbox(c), "the next checkpoint must notify again"


async def test_oldest_wait_comes_first(db_session_factory) -> None:
    tenant, owner = uuid4(), uuid4()
    newer = await _run(db_session_factory, tenant_id=tenant, owner=owner, paused_hours_ago=1)
    older = await _run(db_session_factory, tenant_id=tenant, owner=owner, paused_hours_ago=5)
    async with _client(db_session_factory, tenant_id=tenant, subject_id=owner) as c:
        r = await c.get("/api/review-inbox")
    ids = [i["run_id"] for i in r.json()["items"]]
    assert ids.index(str(older)) < ids.index(str(newer))
