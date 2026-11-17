"""A create must be DURABLE by the time its response is built.

`create_with_deliverables` only FLUSHes, and the session dependency commits in its
``yield`` teardown — which FastAPI (0.136) runs AFTER the response is sent. So
without an explicit commit the caller is handed ``201`` and an id for a row that
is not committed yet, and its very next request reads a database where the
project does not exist → ``404 project not found``.

At idle the teardown commit wins that race in microseconds, which is why it never
showed up locally. Under the live 33-scenario e2e matrix it loses: `storyboard-merge`
and `panel-image` both failed exactly this way on 2026-08-10 — the row present in
the DB afterwards, tenant resolution correct (``claim == effective``), and the 404
returned in 6 ms because there was genuinely nothing to find. One of the two passed
on a re-run with no code change, which is the signature of a race.

⚠️ Driving this through the test client would prove NOTHING: the client runs the
dependency teardown before handing back the response, so the row is always
committed by the time an assertion could look. These tests therefore call the
handler directly and then read through an INDEPENDENT session — a connection that
never saw the handler's transaction and can only see committed data.
"""

from __future__ import annotations

import io
import os
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import UploadFile
from sqlalchemy import select, text
from starlette.datastructures import Headers
from starlette.requests import Request

from src.api.routes import uploads_video
from src.api.routes.admin import record_scope_switch
from src.api.routes.annotations import patch_annotation
from src.api.routes.audit import AuditRequest, create_audit
from src.api.routes.deliverables import patch_deliverable
from src.api.routes.projects import create_project
from src.api.routes.runs.stages import create_stage_annotation
from src.api.routes.storyboard import (
    create_asset,
    create_branch,
    create_panel,
    create_scene,
    patch_panel,
)
from src.api.routes.tenancy.connections import delete_connection, put_connection
from src.api.routes.uploads_video import upload_video_source
from src.api.routes.verbatim import create_item
from src.api.schemas import (
    AnnotationCreate,
    AnnotationUpdate,
    BranchCreateRequest,
    DeliverableUpdate,
    ProjectCreate,
    ScopeSwitchAuditRequest,
    StoryboardAssetCreate,
    StoryboardPanelCreate,
    StoryboardPanelUpdate,
    StoryboardSceneCreate,
    VerbatimItemCreate,
)
from src.api.schemas.connections import ConnectionPut
from src.database.models import (
    AdminAuditLog,
    Annotation,
    Deliverable,
    Run,
    StageOutput,
    StoryboardAsset,
    StoryboardBranch,
    StoryboardPanel,
    StoryboardScene,
    Tenant,
    TenantConnection,
    VerbatimItem,
    VideoSource,
)
from src.database.repositories.admin_audit_repository import AuditRepository
from src.database.repositories.annotation_repository import AnnotationRepository
from src.database.repositories.connections_repository import ConnectionsRepository
from src.database.repositories.deliverable_repository import DeliverableRepository
from src.database.repositories.project_repository import ProjectRepository
from src.services.providers.byo_posture import StaticByoPosture
from src.services.providers.crypto import Keyring
from src.services.providers.rate_limit import ProviderTestRateLimiter
from src.services.providers.settings import ProviderPolicySettings
from src.services.providers.tester import ProviderTestResult
from src.services.storyboard.branch_service import StoryboardBranchService

pytestmark = pytest.mark.asyncio


def _body(title: str) -> ProjectCreate:
    return ProjectCreate.model_validate(
        {
            "type": "generic",
            "title": title,
            "deliverables": [{"name": "d1", "format": "16x9"}],
        }
    )


async def _visible_to_another_connection(db_session_factory, project_id: UUID) -> bool:
    """Is the row visible to a session that never shared the writer's transaction?

    Uncommitted work is invisible across connections, so this is exactly the
    durability question — and it cannot be satisfied by a flush.
    """
    async with db_session_factory() as probe:
        found = (
            await probe.execute(
                text("select 1 from projects where id = :i"), {"i": str(project_id)}
            )
        ).first()
        return found is not None


async def test_create_project_is_committed_before_the_response_is_built(
    db_session_factory,
) -> None:
    """The id handed to the caller must address a row a SEPARATE connection can read.

    Fails against a handler that only flushes: the probe session sees nothing, which
    is precisely what the caller's next request experiences as a 404.
    """
    tenant_id, subject_id = uuid4(), uuid4()

    async with db_session_factory() as session:
        detail = await create_project(
            body=_body("durability-probe"),
            tenant_id=tenant_id,
            subject_id=subject_id,
            project_repo=ProjectRepository(session),
            session=session,
        )

        # Read it from a connection that never saw `session`'s transaction, while
        # `session` is still open — i.e. before any teardown could rescue us.
        assert await _visible_to_another_connection(db_session_factory, detail.id), (
            "the created project is not durable at response-build time; the caller's "
            "next request would 404 on an id we just handed it"
        )


async def test_flush_alone_is_not_durable(db_session_factory) -> None:
    """Pins WHY the commit is required, so nobody 'simplifies' it back to a flush.

    The repository on its own flushes and no more; that write must stay invisible
    to another connection until someone commits. If this ever starts passing
    without a commit, the premise of the fix above has changed and it should be
    re-derived rather than deleted.
    """
    tenant_id, subject_id = uuid4(), uuid4()

    async with db_session_factory() as session:
        proj = await ProjectRepository(session).create_with_deliverables(
            tenant_id=tenant_id,
            owner_subject_id=subject_id,
            type_="generic",
            title="flush-only-probe",
            brief={},
            brand={},
            voice={},
            theme={},
            visibility="tenant",
            parent_project_id=None,
            deliverables=[{"name": "d1", "format": "16x9"}],
        )

        assert not await _visible_to_another_connection(db_session_factory, proj.id), (
            "a flushed-but-uncommitted row was visible to another connection — the "
            "durability premise this fix rests on no longer holds"
        )

        await session.rollback()


# ─── The rest of the write surface (the #791 sweep) ──────────────────────────
# The same race lived in every other mutating handler in `routes/storyboard.py` and
# `routes/projects.py`. These cover a REPRESENTATIVE spread rather than one test per
# handler: one case per distinct repository/service that persists the row, since that
# is what a missing commit is a property of. Same rule as above — call the handler
# directly, then read through an independent session.


async def _row_visible(db_session_factory, model: type[Any], row_id: UUID) -> bool:
    """Is ``row_id`` readable by a session that never shared the writer's transaction?

    The ORM-typed sibling of :func:`_visible_to_another_connection`; the probe session
    is a separate connection either way, so it can only see COMMITTED rows.
    """
    async with db_session_factory() as probe:
        found = (await probe.execute(select(model.id).where(model.id == row_id))).first()
        return found is not None


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


async def test_create_scene_is_committed_before_the_response_is_built(
    db_session_factory,
) -> None:
    """StoryboardSceneRepository.create only flushes — the route must commit."""
    tenant_id, subject_id = uuid4(), uuid4()

    async with db_session_factory() as session:
        project_id = await _seed_project(session, tenant_id, subject_id, "scene-durability")
        scene = await create_scene(
            project_id=project_id,
            body=StoryboardSceneCreate.model_validate({"name": "INT. LAB - NIGHT"}),
            session=session,
            tenant_id=tenant_id,
        )

        assert await _row_visible(db_session_factory, StoryboardScene, scene.id), (
            "the created scene is not durable at response-build time"
        )


async def test_create_asset_is_committed_before_the_response_is_built(
    db_session_factory,
) -> None:
    """StoryboardAssetRepository.create — a different repository, same flush-only rule."""
    tenant_id, subject_id = uuid4(), uuid4()

    async with db_session_factory() as session:
        project_id = await _seed_project(session, tenant_id, subject_id, "asset-durability")
        asset = await create_asset(
            project_id=project_id,
            body=StoryboardAssetCreate.model_validate({"kind": "character", "name": "Hero"}),
            session=session,
            tenant_id=tenant_id,
        )

        assert await _row_visible(db_session_factory, StoryboardAsset, asset.id), (
            "the created asset is not durable at response-build time"
        )


async def test_patch_panel_change_is_committed_before_the_response_is_built(
    db_session_factory,
) -> None:
    """An UPDATE, not an INSERT: existence proves nothing, so assert the NEW VALUE.

    A patch that only flushed would leave the probe reading the pre-patch text — which
    is exactly what a client re-fetching the panel it just edited would see.
    """
    tenant_id, subject_id = uuid4(), uuid4()

    async with db_session_factory() as session:
        project_id = await _seed_project(session, tenant_id, subject_id, "panel-patch-durability")
        panel = await create_panel(
            project_id=project_id,
            body=StoryboardPanelCreate.model_validate({"action_description": "before"}),
            session=session,
            tenant_id=tenant_id,
        )

        await patch_panel(
            panel_id=panel.id,
            body=StoryboardPanelUpdate.model_validate({"action_description": "after"}),
            session=session,
            tenant_id=tenant_id,
            subject_id=subject_id,
        )

        async with db_session_factory() as probe:
            stored = (
                await probe.execute(
                    select(StoryboardPanel.action_description).where(StoryboardPanel.id == panel.id)
                )
            ).scalar_one_or_none()
        assert stored == "after", (
            f"the patched panel is not durable at response-build time (probe read {stored!r}); "
            "the caller's next read would serve the pre-patch value"
        )


async def test_create_branch_is_committed_before_the_response_is_built(
    db_session_factory,
) -> None:
    """The fork writes a snapshot, a child project, the seeded graph AND the link row
    through ``StoryboardBranchService`` — which documents that the ROUTE commits."""
    tenant_id, subject_id = uuid4(), uuid4()

    async with db_session_factory() as session:
        project_id = await _seed_project(session, tenant_id, subject_id, "branch-durability")
        await create_panel(
            project_id=project_id,
            body=StoryboardPanelCreate.model_validate({"action_description": "on the root"}),
            session=session,
            tenant_id=tenant_id,
        )

        branch = await create_branch(
            project_id=project_id,
            body=BranchCreateRequest(name="alt-cut"),
            session=session,
            tenant_id=tenant_id,
            subject_id=subject_id,
            service=StoryboardBranchService(session),
        )

        assert await _row_visible(db_session_factory, StoryboardBranch, branch.id), (
            "the branch link row is not durable at response-build time"
        )
        assert await _visible_to_another_connection(db_session_factory, branch.child_project_id), (
            "the branch's child project is not durable at response-build time; the client "
            "would 404 on the very project the branch response points it at"
        )


# ─── POST /api/audit — the deterministic (not racy) shape ────────────────────
# `create_audit` hands the deliverable straight to `RunEngine.start_run`, which opens
# its OWN session. A flush-only handler therefore fails EVERY time, not sometimes:
# the engine's tenant-scoped lookup finds nothing and raises
# "deliverable not found or not visible to tenant" → 400.


def _request_with_engine(engine: Any) -> Request:
    """A minimal Request whose ``app.state.run_engine`` is ``engine``.

    ``create_audit`` touches nothing else on the request.
    """
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/audit",
            "headers": [],
            "app": SimpleNamespace(state=SimpleNamespace(run_engine=engine)),
        }
    )


async def test_create_audit_deliverable_is_visible_to_start_run(db_session_factory) -> None:
    """At the instant ``start_run`` runs, its own session must find the deliverable.

    The probe reproduces the engine's lookup exactly — ``DeliverableRepository.get``
    under the same tenant scope, on an INDEPENDENT session — from inside a stub
    ``start_run``, i.e. at the real moment of truth. Against the flush-only handler
    the probe returns ``None``, which is the raise the route maps to
    ``400 deliverable not found or not visible to tenant``. Stubbing the engine keeps
    the test off the dispatcher/Redis path while still pinning the failure mode.
    """
    tenant_id, subject_id = uuid4(), uuid4()
    seen: dict[str, Any] = {}

    class _ProbingEngine:
        async def start_run(self, **kwargs: Any) -> SimpleNamespace:
            async with db_session_factory() as probe:
                found = await DeliverableRepository(probe).get(
                    kwargs["deliverable_id"], tenant_id=kwargs["tenant_id"]
                )
            seen["deliverable"] = found
            if found is None:
                # Exactly what the real engine does — so a broken handler surfaces as
                # the 400 users actually get, not as an assertion in a helper.
                raise ValueError("deliverable not found or not visible to tenant")
            return SimpleNamespace(id=uuid4())

    async with db_session_factory() as session:
        resp = await create_audit(
            body=AuditRequest.model_validate(
                {"url": "https://example.com", "auditType": "seo_audit"}
            ),
            request=_request_with_engine(_ProbingEngine()),
            tenant_id=tenant_id,
            subject_id=subject_id,
            project_repo=ProjectRepository(session),
            session=session,
        )

    assert seen["deliverable"] is not None, (
        "start_run's own session could not see the audit deliverable — POST /api/audit "
        "returns 400 'deliverable not found or not visible to tenant' every time"
    )
    assert seen["deliverable"].id == resp.deliverable_id
    assert await _visible_to_another_connection(db_session_factory, resp.project_id), (
        "the audit project is not durable at response-build time"
    )


# ─── The remaining write surface (admin / deliverables / verbatim / annotations /
#     video sources) ────────────────────────────────────────────────────────────
# Same rule and the same representative-spread policy as above: one case per
# distinct repository that persists the row, driven at the handler directly with a
# probe on an independent connection. Thirteen near-identical tests would prove
# nothing the five below do not.


async def test_patch_deliverable_change_is_committed_before_the_response_is_built(
    db_session_factory,
) -> None:
    """DeliverableRepository.update only flushes — assert the NEW VALUE, not existence."""
    tenant_id, subject_id = uuid4(), uuid4()

    async with db_session_factory() as session:
        proj = await ProjectRepository(session).create_with_deliverables(
            tenant_id=tenant_id,
            owner_subject_id=subject_id,
            type_="generic",
            title="deliverable-patch-durability",
            brief={},
            brand={},
            voice={},
            theme={},
            visibility="tenant",
            parent_project_id=None,
            deliverables=[{"name": "before", "format": "16x9"}],
        )
        await session.commit()
        deliverable_id = proj.deliverables[0].id

        await patch_deliverable(
            deliverable_id=deliverable_id,
            body=DeliverableUpdate.model_validate({"name": "after"}),
            tenant_id=tenant_id,
            deliverable_repo=DeliverableRepository(session),
            session=session,
        )

        async with db_session_factory() as probe:
            stored = (
                await probe.execute(
                    select(Deliverable.name).where(Deliverable.id == deliverable_id)
                )
            ).scalar_one_or_none()
        assert stored == "after", (
            f"the patched deliverable is not durable at response-build time (probe read "
            f"{stored!r}); the caller's next read would serve the pre-patch value"
        )


async def test_create_verbatim_item_is_committed_before_the_response_is_built(
    db_session_factory,
) -> None:
    """VerbatimItemRepository.create only flushes — the route must commit."""
    tenant_id, subject_id = uuid4(), uuid4()

    async with db_session_factory() as session:
        project_id = await _seed_project(session, tenant_id, subject_id, "verbatim-durability")
        item = await create_item(
            project_id=project_id,
            body=VerbatimItemCreate.model_validate({"text": "say it exactly like this"}),
            session=session,
            tenant_id=tenant_id,
        )

        assert await _row_visible(db_session_factory, VerbatimItem, item.id), (
            "the created verbatim item is not durable at response-build time"
        )


async def test_patch_annotation_change_is_committed_before_the_response_is_built(
    db_session_factory,
) -> None:
    """AnnotationRepository.update only flushes — assert the NEW VALUE."""
    tenant_id, subject_id = uuid4(), uuid4()

    async with db_session_factory() as session:
        proj = await ProjectRepository(session).create_with_deliverables(
            tenant_id=tenant_id,
            owner_subject_id=subject_id,
            type_="generic",
            title="annotation-patch-durability",
            brief={},
            brand={},
            voice={},
            theme={},
            visibility="tenant",
            parent_project_id=None,
            deliverables=[{"name": "d1", "format": "16x9"}],
        )
        run = Run(
            id=uuid4(),
            deliverable_id=proj.deliverables[0].id,
            blueprint_id="generic_v1",
            status="paused",
            config_snapshot={},
            pause_after_stages=[],
        )
        stage_output = StageOutput(
            id=uuid4(), run_id=run.id, stage_name="script_draft", status="completed", output={}
        )
        session.add_all([run, stage_output])
        annotation = await AnnotationRepository(session).create(
            stage_output_id=stage_output.id, text="before", element_id=None
        )
        await session.commit()

        await patch_annotation(
            annotation_id=annotation.id,
            body=AnnotationUpdate.model_validate({"text": "after"}),
            session=session,
            tenant_id=tenant_id,
        )

        async with db_session_factory() as probe:
            stored = (
                await probe.execute(select(Annotation.text).where(Annotation.id == annotation.id))
            ).scalar_one_or_none()
        assert stored == "after", (
            f"the patched annotation is not durable at response-build time (probe read "
            f"{stored!r}); the caller's next read would serve the pre-patch value"
        )


async def test_record_scope_switch_is_committed_before_the_response_is_built(
    db_session_factory,
) -> None:
    """The purest single-``add`` handler in the repo: the whole thing is one audit row.

    A security-audit record that only exists in a pending transaction is not a record.
    """
    actor, from_tenant, to_tenant = uuid4(), uuid4(), uuid4()
    request = cast("Request", SimpleNamespace(state=SimpleNamespace(tenant_id=from_tenant)))

    async with db_session_factory() as session:
        entry = await record_scope_switch(
            request=request,
            body=ScopeSwitchAuditRequest(to_tenant_id=to_tenant),
            actor=actor,
            _=None,
            audit=AuditRepository(session),
            session=session,
        )

        assert await _row_visible(db_session_factory, AdminAuditLog, entry.id), (
            "the scope-switch audit row is not durable at response-build time"
        )


async def test_upload_video_source_is_committed_before_the_response_is_built(
    db_session_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row must be durable before the response — and before the preview worker looks.

    That worker runs OUT OF PROCESS and reads ``video_sources`` on its own connection,
    so a merely-flushed row is invisible to it: the clip's bytes are in MinIO and
    nothing ever derives a preview from them.
    """

    class _FakeStorage:
        async def put_file(self, key: str, path: str, content_type: str) -> None:
            return None

        async def get_presigned_url(self, key: str) -> str:
            return f"https://minio.test/{key}"

    async def _get_storage() -> _FakeStorage:
        return _FakeStorage()

    monkeypatch.setattr(uploads_video, "_get_storage", _get_storage)

    tenant_id, subject_id = uuid4(), uuid4()
    # 12 bytes of ISO-BMFF head (size, ``ftyp``, brand) is all the magic-byte gate reads.
    clip = (32).to_bytes(4, "big") + b"ftyp" + b"isom" + b"\x00" * 4 + b"\x00" * 64
    upload = UploadFile(
        file=io.BytesIO(clip),
        filename="take-1.mp4",
        headers=Headers({"content-type": "video/mp4"}),
    )
    # The handler reads exactly one header off the request.
    request = cast("Request", SimpleNamespace(headers={}))

    async with db_session_factory() as session:
        project_id = await _seed_project(session, tenant_id, subject_id, "video-source-durability")
        resp = await upload_video_source(
            project_id=project_id,
            file=upload,
            request=request,
            session=session,
            tenant_id=tenant_id,
        )

        assert await _row_visible(db_session_factory, VideoSource, resp.source_id), (
            "the video source row is not durable at response-build time; the preview "
            "worker reads it from another connection and would never see it"
        )


async def test_create_stage_annotation_is_committed_before_the_response_is_built(
    db_session_factory,
) -> None:
    """The #804 sweep missed this handler; `AnnotationRepository.create` only flushes.

    Asserted through an INDEPENDENT session on purpose: driving this through the
    test client proves nothing, because the client runs the dependency teardown
    (which commits) before returning, so the row is always visible by the time
    you can assert and the test passes on broken code.
    """
    tenant_id, subject_id = uuid4(), uuid4()

    async with db_session_factory() as session:
        proj = await ProjectRepository(session).create_with_deliverables(
            tenant_id=tenant_id,
            owner_subject_id=subject_id,
            type_="generic",
            title="stage-annotation-create-durability",
            brief={},
            brand={},
            voice={},
            theme={},
            visibility="tenant",
            parent_project_id=None,
            deliverables=[{"name": "d1", "format": "16x9"}],
        )
        run = Run(
            id=uuid4(),
            deliverable_id=proj.deliverables[0].id,
            blueprint_id="generic_v1",
            status="paused",
            config_snapshot={},
            pause_after_stages=[],
        )
        stage_output = StageOutput(
            id=uuid4(), run_id=run.id, stage_name="script_draft", status="completed", output={}
        )
        session.add_all([run, stage_output])
        await session.commit()

        created = await create_stage_annotation(
            run_id=run.id,
            stage_name="script_draft",
            body=AnnotationCreate.model_validate({"text": "durable?"}),
            session=session,
            tenant_id=tenant_id,
        )

        async with db_session_factory() as probe:
            stored = (
                await probe.execute(select(Annotation.text).where(Annotation.id == created.id))
            ).scalar_one_or_none()

    assert stored == "durable?", (
        "annotation not visible to an independent connection: the handler returned "
        "before committing, so its 201 describes a row nobody else can read yet"
    )


# ─── P3 WP4: tenant provider connections ─────────────────────────────────────


class _PassingTester:
    async def test(self, **_: object) -> ProviderTestResult:
        return ProviderTestResult(True, "ok", 200)


_CONN_POLICY = ProviderPolicySettings(
    ssrf_mode="private_ok", egress_allowlist=frozenset({"llm.durable.test"})
)


async def _seed_tenant(db_session_factory) -> UUID:
    tenant_id = uuid7()
    async with db_session_factory() as s:
        s.add(Tenant(id=tenant_id, name=f"durable-{tenant_id}"))
        await s.commit()
    return tenant_id


async def _active_connection_audits(db_session_factory, tenant_id: UUID) -> tuple[int, list[str]]:
    async with db_session_factory() as probe:
        active = (
            await probe.execute(
                select(TenantConnection.id).where(
                    TenantConnection.tenant_id == tenant_id, TenantConnection.status == "active"
                )
            )
        ).all()
        actions = (
            (
                await probe.execute(
                    select(AdminAuditLog.action).where(AdminAuditLog.target_tenant_id == tenant_id)
                )
            )
            .scalars()
            .all()
        )
    return len(active), list(actions)


async def test_put_connection_is_committed_before_the_response_is_built(
    db_session_factory,
) -> None:
    """The version + its audit row must be readable by another connection while the
    handler's session is still open — i.e. before the dependency teardown commit."""
    tenant_id = await _seed_tenant(db_session_factory)
    async with db_session_factory() as session:
        view = await put_connection(
            slot="llm",
            body=ConnectionPut(
                kind="openai-compatible",
                base_url="https://llm.durable.test/v1",
                secret="sk-durable",  # pragma: allowlist secret
                expected_version=0,
            ),
            tenant_id=tenant_id,
            actor=uuid4(),
            audit=AuditRepository(session),
            session=session,
            policy=_CONN_POLICY,
            keyring=Keyring([("k1", os.urandom(32))]),
            tester=_PassingTester(),
            posture=StaticByoPosture("local_allowed"),
            limiter=ProviderTestRateLimiter(None, limit=0, window_seconds=60),
        )
        assert view.connection is not None and view.connection.version == 1
        active, actions = await _active_connection_audits(db_session_factory, tenant_id)

    assert active == 1, "PUT /api/tenant/connections/llm returned before committing"
    assert actions == ["connection.set"]


async def test_delete_connection_is_committed_before_the_response_is_built(
    db_session_factory,
) -> None:
    tenant_id = await _seed_tenant(db_session_factory)
    async with db_session_factory() as s:
        await ConnectionsRepository(s).create_version(
            tenant_id=tenant_id, slot="llm", kind="anthropic", secret=None
        )
        await s.commit()

    async with db_session_factory() as session:
        await delete_connection(
            slot="llm",
            tenant_id=tenant_id,
            actor=uuid4(),
            audit=AuditRepository(session),
            session=session,
        )
        active, actions = await _active_connection_audits(db_session_factory, tenant_id)

    assert active == 0, "DELETE /api/tenant/connections/llm returned before committing"
    assert actions == ["connection.delete"]


async def test_tenant_music_upload_and_delete_are_committed_before_the_response(
    db_session_factory,
) -> None:
    """P6 ORCH-4: the track row + its audit row (upload), and the delete + its audit
    row, are visible to another connection while the handler's session is open."""
    import wave

    from src.api.routes.tenancy.music import delete_tenant_music, upload_tenant_music
    from src.database.models import TenantSoundAsset
    from src.services import tts_storage

    class _Mem:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}

        async def put_object(self, key: str, data: bytes, content_type: str = "") -> str:
            self.objects[key] = data
            return key

        async def delete_object(self, key: str) -> bool:
            self.objects.pop(key, None)
            return True

        async def get_presigned_url(self, path: str, expires: int | None = None) -> str:
            return f"https://minio.test/{path}"

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 4000)
    tts_storage.set_storage_for_tests(cast(Any, _Mem()))
    try:
        tenant_id = await _seed_tenant(db_session_factory)
        async with db_session_factory() as session:
            track = await upload_tenant_music(
                request=cast("Request", SimpleNamespace(headers={})),
                tenant_id=tenant_id,
                actor=uuid4(),
                audit=AuditRepository(session),
                session=session,
                file=UploadFile(file=io.BytesIO(buf.getvalue()), filename="bed.wav"),
                rights_attested=True,
                category="music",
                description="",
                slug=None,
            )
            assert await _row_visible(db_session_factory, TenantSoundAsset, track.id), (
                "POST /api/tenant/music returned before committing"
            )
        async with db_session_factory() as session:
            await delete_tenant_music(
                track_id=track.id,
                tenant_id=tenant_id,
                actor=uuid4(),
                audit=AuditRepository(session),
                session=session,
            )
            assert not await _row_visible(db_session_factory, TenantSoundAsset, track.id), (
                "DELETE /api/tenant/music returned before committing"
            )
            async with db_session_factory() as probe:
                actions = (
                    (
                        await probe.execute(
                            select(AdminAuditLog.action).where(
                                AdminAuditLog.target_tenant_id == tenant_id
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
            assert sorted(actions) == ["music.delete", "music.upload"]
    finally:
        tts_storage.set_storage_for_tests(None)
