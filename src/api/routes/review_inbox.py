"""The run owner's in-app "needs your review" inbox.

Why: run 01a1770d (promo_v1) waited 29h at its storyboard review checkpoint because
nobody knew it was waiting. Studio has no channel outside itself (and must not borrow
another app's), so the owner is told INSIDE Studio: the frontend polls this inbox and
shows a bell + a banner on the dashboard and the project list, each linking straight
to the checkpoint's review page.

The inbox is derived from the run rows, never stored separately, so it cannot drift:
a notice exists exactly while a run the caller OWNS (``projects.owner_subject_id``) in
the caller's tenant is paused at a review CHECKPOINT (``pause_reason='checkpoint'``,
migration 0102). A manual pause has nothing to review and never appears.

* Dismiss (``POST /api/review-inbox/{run_id}/dismiss``) hides a notice — until the run
  has waited ``REMINDER_AFTER`` (24h), when it comes back flagged ``reminder`` ONCE.
  Dismissing the reminder (``?reminder=true``: the caller says it was SHOWN the
  reminder) hides it for the rest of this pause (no repeat reminders). A plain dismiss
  keeps the FIRST dismissal time, so a stale tab still showing the pre-reminder notice
  cannot dismiss the reminder before anyone has seen it.
* A checkpoint pause with no ``paused_at`` stamp counts as having waited long: it
  is shown as the reminder straight away, so it can never be silently starved.
* Resume clears the dismissal, so the run's next checkpoint is a fresh notice.
"""

from __future__ import annotations

import datetime
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import ColumnElement, and_, func, literal, or_, select, update

from src.api.routes._common import SessionDep, SubjectIdDep, TenantIdDep
from src.api.schemas.review_inbox import ReviewInbox, ReviewInboxItem
from src.database.models import Deliverable, Project, Run

router = APIRouter(prefix="/api/review-inbox", tags=["review-inbox"])

#: How long a checkpoint waits before its notice becomes the reminder.
REMINDER_AFTER = datetime.timedelta(hours=24)


def _waiting_long() -> ColumnElement[bool]:
    # NULL paused_at = wait of unknown length: treat as long, so it gets the reminder.
    return or_(Run.paused_at.is_(None), Run.paused_at <= func.now() - literal(REMINDER_AFTER))


def _owned_review_pauses(tenant_id: UUID, subject_id: UUID) -> list[ColumnElement[bool]]:
    return [
        Project.tenant_id == tenant_id,
        Project.owner_subject_id == subject_id,
        Run.status == "paused",
        Run.pause_reason == "checkpoint",
    ]


@router.get("", response_model=ReviewInbox, operation_id="review_inbox.list")
async def list_review_inbox(
    session: SessionDep, tenant_id: TenantIdDep, subject_id: SubjectIdDep
) -> ReviewInbox:
    """Runs paused at a review checkpoint that the caller owns, oldest wait first."""
    visible = or_(
        Run.review_notice_dismissed_at.is_(None),
        # A dismissed notice returns as the reminder — unless it was dismissed AFTER
        # the reminder mark (the owner already saw the reminder). With no paused_at the
        # notice IS the reminder from the start, so its dismissal is final.
        and_(
            _waiting_long(),
            Run.review_notice_dismissed_at < Run.paused_at + literal(REMINDER_AFTER),
        ),
    )
    rows = (
        await session.execute(
            select(
                Run.id,
                Run.current_stage,
                Run.paused_at,
                _waiting_long().label("reminder"),
                Project.id.label("project_id"),
                Project.title,
                Deliverable.name.label("deliverable_name"),
            )
            .join(Deliverable, Deliverable.id == Run.deliverable_id)
            .join(Project, Project.id == Deliverable.project_id)
            .where(*_owned_review_pauses(tenant_id, subject_id), visible)
            .order_by(Run.paused_at.asc().nulls_last(), Run.id)
        )
    ).all()
    items = [
        ReviewInboxItem(
            run_id=r.id,
            project_id=r.project_id,
            project_title=r.title,
            deliverable_name=r.deliverable_name,
            stage_name=r.current_stage,
            paused_at=r.paused_at,
            reminder=bool(r.reminder),
        )
        for r in rows
    ]
    return ReviewInbox(items=items)


@router.post(
    "/{run_id}/dismiss",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="review_inbox.dismiss",
)
async def dismiss_review_notice(
    run_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    reminder: bool = Query(
        False,
        description="True when the caller is dismissing the item it was shown flagged "
        "`reminder`: that dismissal is final for this pause.",
    ),
) -> None:
    """Hide this notice until the run has waited ``REMINDER_AFTER``. Owner-only; 404 otherwise (a
    foreign or missing run, or one that is not waiting at a checkpoint)."""
    owned = (
        select(Run.id)
        .join(Deliverable, Deliverable.id == Run.deliverable_id)
        .join(Project, Project.id == Deliverable.project_id)
        .where(Run.id == run_id, *_owned_review_pauses(tenant_id, subject_id))
        .scalar_subquery()
    )
    result = await session.execute(
        update(Run)
        .where(Run.id == owned)
        .values(
            # Only a dismissal of the reminder the caller SAW moves the stamp past the
            # reminder mark. A plain dismiss keeps the first stamp: a stale tab that
            # dismisses the pre-reminder notice after 24h must not kill the reminder.
            review_notice_dismissed_at=func.now()
            if reminder
            else func.coalesce(Run.review_notice_dismissed_at, func.now())
        )
        .execution_options(synchronize_session=False)
    )
    if int(getattr(result, "rowcount", 0) or 0) == 0:
        raise HTTPException(404, "no review notice for this run")
    # Commit before responding (see deliverables.py: get_db_session commits AFTER
    # the response is sent).
    await session.commit()
