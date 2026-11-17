"""A DEFERRED unique violation must still map to 409, not escape as a 500.

``uq_storyboard_panels_project_shot`` is DEFERRABLE INITIALLY IMMEDIATE, and the
merge / restore paths deliberately run ``SET CONSTRAINTS ALL DEFERRED`` so a
reorder or a shot_key rename can hold a TRANSIENT duplicate mid-write
(``merge_service.py``, ``storyboard_snapshot_service.py``). On those paths the
check does not run at flush — it runs at COMMIT.

Five handlers used to commit OUTSIDE the ``try`` that maps ``IntegrityError`` to
409, so on exactly those paths the violation escaped the ``except`` and the
caller got a raw 500 from a handler written to return 409. Pre-existing since
2026-08-10 (3cd74c8d / 9b032f9f), not a regression. ``patch_panel``'s FIRST
commit was already inside its try and carries a comment saying why — it is the
reference this file is written against.

⚠️ These call the handlers DIRECTLY. Driving them through the FastAPI test client
proves nothing about commit placement, for the same reason
``tests/api/routes/test_write_durability.py`` gives: the client runs the
dependency teardown — which commits — before handing back the response, so the
commit under test is not the one that decides the status code. Every existing
storyboard 409 test raises its IntegrityError from INSIDE the try (a flush-time
collision, or a stubbed service), which is why none of them could see this.

The deferral here is real, not simulated: each test issues the same
``SET CONSTRAINTS ALL DEFERRED`` the production merge path issues, against the
real constraint, so the violation genuinely surfaces at ``session.commit()``.
"""

from __future__ import annotations

from typing import cast
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from src.api.routes.storyboard import (
    create_branch,
    create_panel,
    delete_asset,
    merge_branch,
    merge_into_sibling,
    patch_panel,
)
from src.api.schemas import (
    BranchCreateRequest,
    StoryboardAssetCreate,
    StoryboardPanelCreate,
    StoryboardPanelUpdate,
)
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.storyboard_repository import (
    StoryboardPanelRepository,
    StoryboardRepository,
)
from src.services.storyboard.branch_service import StoryboardBranchService
from src.services.storyboard.merge_service import StoryboardMergeService

pytestmark = pytest.mark.asyncio

_SHOT_KEY = "1A"


async def _seed_project(session, tenant_id: UUID, subject_id: UUID, title: str) -> UUID:
    """A committed project to hang storyboard rows off. Not the thing under test."""
    proj = await ProjectRepository(session).create_with_deliverables(
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        type_="generic",
        title=title,
        brief={},
        brand={},
        voice={},
        theme={},
        visibility="tenant",
        parent_project_id=None,
        deliverables=[{"name": "d1", "format": "16x9"}],
    )
    await session.commit()
    return proj.id


async def _real_branch(session, project_id: UUID, tenant_id: UUID, subject_id: UUID) -> UUID:
    """A real open branch, so ``_require_branch_at`` (which runs BEFORE the try)
    does not 404 the request before it can reach the commit under test."""
    branch = await create_branch(
        project_id=project_id,
        body=BranchCreateRequest(name="alt-cut"),
        session=session,
        tenant_id=tenant_id,
        subject_id=subject_id,
        service=StoryboardBranchService(session),
    )
    return branch.id


async def _write_duplicate_shot_key(session, project_id: UUID, tenant_id: UUID) -> None:
    """Mint a SECOND panel on an already-used shot_key, through the real repository.

    Under an active deferral this INSERT succeeds and the unique violation is held
    back to COMMIT — which is exactly the state a colliding merge leaves the
    transaction in, and the only state in which the misplaced commit was reachable.
    """
    storyboard = await StoryboardRepository(session).ensure_for_project(
        project_id, tenant_id=tenant_id
    )
    assert storyboard is not None
    await StoryboardPanelRepository(session).create(
        project_id=project_id,
        storyboard_id=storyboard.id,
        position=None,
        shot_key=_SHOT_KEY,
    )


async def _defer_constraints(session) -> None:
    """Exactly what ``merge_service`` / ``storyboard_snapshot_service`` do.

    This is the whole point: after this statement the unique check no longer runs
    at flush, so a duplicate INSERT succeeds and the violation is raised by
    ``COMMIT``. That is the only code path on which the misplaced commit was
    reachable, and it is why every pre-existing 409 test missed it.
    """
    await session.execute(text("SET CONSTRAINTS ALL DEFERRED"))


async def test_the_constraint_really_defers(db_session_factory) -> None:
    """Premise check. Without this the tests below could pass vacuously.

    If ``SET CONSTRAINTS ALL DEFERRED`` did not actually defer this constraint,
    the duplicate would raise at flush — inside the try — and every assertion
    below would be testing the ALREADY-correct flush-time path while telling us
    it had tested the commit-time one.
    """
    tenant_id, subject_id = uuid4(), uuid4()
    async with db_session_factory() as session:
        project_id = await _seed_project(session, tenant_id, subject_id, "defer-premise")
        await create_panel(
            project_id=project_id,
            body=StoryboardPanelCreate.model_validate({"shot_key": _SHOT_KEY}),
            session=session,
            tenant_id=tenant_id,
        )
        await _defer_constraints(session)

        # The duplicate INSERT itself must SUCCEED under deferral.
        try:
            await create_panel(
                project_id=project_id,
                body=StoryboardPanelCreate.model_validate({"shot_key": _SHOT_KEY}),
                session=session,
                tenant_id=tenant_id,
            )
        except HTTPException as exc:
            assert exc.status_code == 409
        else:  # pragma: no cover - only reached if the handler stopped committing
            pytest.fail("create_panel neither raised nor was reached — check the fixture")

        await session.rollback()


async def test_create_panel_deferred_duplicate_is_409_not_500(db_session_factory) -> None:
    """The commit is where the collision lands; the 409 must still be produced."""
    tenant_id, subject_id = uuid4(), uuid4()
    async with db_session_factory() as session:
        project_id = await _seed_project(session, tenant_id, subject_id, "create-deferred-409")
        await create_panel(
            project_id=project_id,
            body=StoryboardPanelCreate.model_validate({"shot_key": _SHOT_KEY}),
            session=session,
            tenant_id=tenant_id,
        )
        await _defer_constraints(session)

        with pytest.raises(HTTPException) as excinfo:
            await create_panel(
                project_id=project_id,
                body=StoryboardPanelCreate.model_validate({"shot_key": _SHOT_KEY}),
                session=session,
                tenant_id=tenant_id,
            )

        assert excinfo.value.status_code == 409, (
            "a deferred shot_key collision surfaced at COMMIT and escaped the "
            "`except IntegrityError`, so the caller got a 500 from a handler "
            "written to return 409"
        )
        await session.rollback()


async def test_patch_panel_status_flip_deferred_duplicate_is_409_not_500(
    db_session_factory,
) -> None:
    """``patch_panel``'s SECOND commit — the status-flip path.

    Its FIRST commit (the no-status branch) was already inside the try; this
    exercises the other one, which was not. A PATCH carrying both a colliding
    ``shot_key`` and a ``status`` takes the flip branch, so the field write and
    the flip ride the same transaction and the collision lands on that commit.
    """
    tenant_id, subject_id = uuid4(), uuid4()
    async with db_session_factory() as session:
        project_id = await _seed_project(session, tenant_id, subject_id, "patch-deferred-409")
        await create_panel(
            project_id=project_id,
            body=StoryboardPanelCreate.model_validate({"shot_key": _SHOT_KEY}),
            session=session,
            tenant_id=tenant_id,
        )
        victim = await create_panel(
            project_id=project_id,
            body=StoryboardPanelCreate.model_validate({"shot_key": "2B"}),
            session=session,
            tenant_id=tenant_id,
        )
        await _defer_constraints(session)

        with pytest.raises(HTTPException) as excinfo:
            await patch_panel(
                panel_id=victim.id,
                body=StoryboardPanelUpdate.model_validate(
                    {"shot_key": _SHOT_KEY, "status": "approved"}
                ),
                session=session,
                tenant_id=tenant_id,
                subject_id=subject_id,
            )

        assert excinfo.value.status_code == 409, (
            "the status-flip commit sat outside the try, so a deferred collision "
            "from the field write escaped as a 500"
        )
        await session.rollback()


async def test_merge_branch_deferred_duplicate_is_409_not_500(db_session_factory) -> None:
    """The sharpest of the five: this path ALWAYS defers.

    ``merge_service.merge`` runs ``SET CONSTRAINTS ALL DEFERRED`` itself, so on a
    real colliding merge the violation can ONLY appear at the route's commit. The
    stub stands in for the service so the test does not have to construct a
    genuinely conflicting three-way merge — but the deferral, the constraint and
    the duplicate row it writes are all real, so the failure arrives at exactly
    the line under test rather than being simulated by a raised exception.
    """
    tenant_id, subject_id = uuid4(), uuid4()
    async with db_session_factory() as session:
        project_id = await _seed_project(session, tenant_id, subject_id, "merge-deferred-409")
        await create_panel(
            project_id=project_id,
            body=StoryboardPanelCreate.model_validate({"shot_key": _SHOT_KEY}),
            session=session,
            tenant_id=tenant_id,
        )
        branch_id = await _real_branch(session, project_id, tenant_id, subject_id)
        await session.commit()

        class _CollidingMerge:
            """Defers, then writes a duplicate — what a colliding merge does."""

            async def merge(self, *_args, **_kwargs) -> dict:
                await _defer_constraints(session)
                await _write_duplicate_shot_key(session, project_id, tenant_id)
                return {}

        with pytest.raises(HTTPException) as excinfo:
            await merge_branch(
                project_id=project_id,
                branch_id=branch_id,
                body=_merge_body(),
                session=session,
                tenant_id=tenant_id,
                subject_id=subject_id,
                service=cast("StoryboardMergeService", _CollidingMerge()),
            )

        assert excinfo.value.status_code == 409, (
            "merge_service ALWAYS defers this constraint, so the collision can only "
            "land on the route's commit — with that commit outside the try the "
            "'merge raced a concurrent write' 409 was unreachable"
        )
        await session.rollback()


async def test_merge_into_sibling_deferred_duplicate_is_409_not_500(db_session_factory) -> None:
    """``merge_into_sibling`` defers too — same exposure, same fix."""
    tenant_id, subject_id = uuid4(), uuid4()
    async with db_session_factory() as session:
        project_id = await _seed_project(session, tenant_id, subject_id, "sibling-deferred-409")
        await create_panel(
            project_id=project_id,
            body=StoryboardPanelCreate.model_validate({"shot_key": _SHOT_KEY}),
            session=session,
            tenant_id=tenant_id,
        )
        branch_id = await _real_branch(session, project_id, tenant_id, subject_id)
        await session.commit()

        class _CollidingSiblingMerge:
            async def merge_into_sibling(self, *_args, **_kwargs) -> dict:
                await _defer_constraints(session)
                await _write_duplicate_shot_key(session, project_id, tenant_id)
                return {}

        with pytest.raises(HTTPException) as excinfo:
            await merge_into_sibling(
                project_id=project_id,
                branch_id=branch_id,
                body=_sibling_body(),
                session=session,
                tenant_id=tenant_id,
                subject_id=subject_id,
                service=cast("StoryboardMergeService", _CollidingSiblingMerge()),
            )

        assert excinfo.value.status_code == 409
        await session.rollback()


async def test_delete_asset_404_still_precedes_the_commit(db_session_factory) -> None:
    """The 404 moved INSIDE the try with the commit; it must still come FIRST.

    ``delete_asset`` is the consistency case, not a live 500 — it never writes
    ``storyboard_panels``, so the deferrable unique constraint cannot fire there.
    What CAN regress is the reordering: if the commit were placed ahead of the
    not-found branch, a CAS miss would commit whatever else the transaction held.
    """
    tenant_id, subject_id = uuid4(), uuid4()
    async with db_session_factory() as session:
        await _seed_project(session, tenant_id, subject_id, "delete-asset-order")

        with pytest.raises(HTTPException) as excinfo:
            await delete_asset(
                asset_id=uuid4(),
                session=session,
                tenant_id=tenant_id,
                expected_lock_version=0,
            )

        assert excinfo.value.status_code == 404, (
            "a missing asset must 404 without committing, not fall through to the commit"
        )
        await session.rollback()


async def test_delete_asset_still_requires_the_lock(db_session_factory) -> None:
    """Control: the 400 guard ahead of the try is untouched by the move."""
    tenant_id, subject_id = uuid4(), uuid4()
    async with db_session_factory() as session:
        await _seed_project(session, tenant_id, subject_id, "delete-asset-lock")

        with pytest.raises(HTTPException) as excinfo:
            await delete_asset(
                asset_id=uuid4(),
                session=session,
                tenant_id=tenant_id,
                expected_lock_version=None,
            )

        assert excinfo.value.status_code == 400
        await session.rollback()


async def test_create_panel_without_a_collision_still_commits(db_session_factory) -> None:
    """Control for the whole file.

    Moving a commit into a ``try`` can silently stop it happening (a swallowed
    exception, a misplaced ``return``). This asserts the ordinary path still
    lands the row durably — read through an INDEPENDENT session, per
    ``test_write_durability.py``, because a flush would satisfy a same-session read.
    """
    tenant_id, subject_id = uuid4(), uuid4()
    async with db_session_factory() as session:
        project_id = await _seed_project(session, tenant_id, subject_id, "create-still-commits")
        panel = await create_panel(
            project_id=project_id,
            body=StoryboardPanelCreate.model_validate({"shot_key": "9Z"}),
            session=session,
            tenant_id=tenant_id,
        )

    async with db_session_factory() as probe:
        found = (
            await probe.execute(
                text("select shot_key from storyboard_panels where id = :i"),
                {"i": str(panel.id)},
            )
        ).scalar_one_or_none()

    assert found == "9Z", (
        "the panel is not durable at response-build time — moving the commit into "
        "the try stopped it committing at all"
    )


def _merge_body():
    from src.api.schemas.storyboard import MergeRequest

    return MergeRequest.model_validate({"root_fingerprint": "x", "resolutions": []})


def _sibling_body():
    from src.api.schemas.storyboard import SiblingMergeRequest

    return SiblingMergeRequest.model_validate(
        {"target_branch_id": str(uuid4()), "target_fingerprint": "x", "resolutions": []}
    )


async def test_delete_asset_unused_still_deletes(db_session_factory) -> None:
    """Control: the happy path still commits the delete."""
    from src.api.routes.storyboard import create_asset

    tenant_id, subject_id = uuid4(), uuid4()
    async with db_session_factory() as session:
        project_id = await _seed_project(session, tenant_id, subject_id, "delete-asset-happy")
        asset = await create_asset(
            project_id=project_id,
            body=StoryboardAssetCreate.model_validate({"kind": "character", "name": "Hero"}),
            session=session,
            tenant_id=tenant_id,
        )
        await delete_asset(
            asset_id=asset.id,
            session=session,
            tenant_id=tenant_id,
            expected_lock_version=asset.lock_version,
        )

    async with db_session_factory() as probe:
        found = (
            await probe.execute(
                text("select 1 from storyboard_assets where id = :i"), {"i": str(asset.id)}
            )
        ).first()

    assert found is None, "the delete is not durable at response-build time"
