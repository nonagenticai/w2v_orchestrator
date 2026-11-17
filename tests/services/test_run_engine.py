from uuid import uuid4

import pytest

from src.services.run_engine import RunEngine

pytestmark = pytest.mark.asyncio


async def test_start_run_creates_pending_then_dispatches(db_session) -> None:
    from src.database.repositories.project_repository import ProjectRepository
    from src.services.blueprints.registry import get_registry

    tenant = uuid4()
    subject = uuid4()
    proj_repo = ProjectRepository(db_session)
    proj = await proj_repo.create_with_deliverables(
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
            {
                "name": "d",
                "format": "16x9",
                "captions_mode": "none",
                "target_duration_seconds": 30,
            }
        ],
    )
    await db_session.commit()
    deliverable_id = proj.deliverables[0].id

    dispatched: list = []

    async def fake_dispatch(run_id):
        dispatched.append(run_id)

    # The session_factory must return an async-context-manager-like object that yields db_session.
    # Use a small wrapper.
    class _SessionCtx:
        async def __aenter__(self):
            return db_session

        async def __aexit__(self, *exc):
            return False

    engine = RunEngine(
        session_factory=lambda: _SessionCtx(),
        registry=get_registry(),
        dispatch=fake_dispatch,
    )
    run = await engine.start_run(
        deliverable_id=deliverable_id,
        tenant_id=tenant,
        subject_id=subject,
        parent_run_id=None,
        pause_after_stages=[],
    )
    assert run.status == "pending"
    assert run.blueprint_id == "generic_v1"
    assert dispatched == [run.id]


async def test_start_run_loads_verbatim_items_into_config_snapshot(db_session) -> None:
    from src.database.repositories.project_repository import ProjectRepository
    from src.database.repositories.verbatim_item_repository import VerbatimItemRepository
    from src.services.blueprints.registry import get_registry

    tenant = uuid4()
    subject = uuid4()
    proj_repo = ProjectRepository(db_session)
    proj = await proj_repo.create_with_deliverables(
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
            {
                "name": "d",
                "format": "16x9",
                "captions_mode": "none",
                "target_duration_seconds": 30,
            }
        ],
    )
    await db_session.commit()
    deliverable_id = proj.deliverables[0].id

    vr = VerbatimItemRepository(db_session)
    first = await vr.create(project_id=proj.id, text="first phrase", position=0)
    second = await vr.create(
        project_id=proj.id,
        text="second phrase",
        position=1,
        kind="slide",
        scope="slide",
        scope_target="slide-3",
    )
    await db_session.commit()
    first_id, second_id = str(first.id), str(second.id)

    class _SessionCtx:
        async def __aenter__(self):
            return db_session

        async def __aexit__(self, *exc):
            return False

    async def fake_dispatch(run_id):
        return None

    engine = RunEngine(
        session_factory=lambda: _SessionCtx(),
        registry=get_registry(),
        dispatch=fake_dispatch,
    )
    run = await engine.start_run(
        deliverable_id=deliverable_id,
        tenant_id=tenant,
        subject_id=subject,
        parent_run_id=None,
        pause_after_stages=[],
    )

    items = run.config_snapshot["verbatim_items"]
    # Ordered by position then created_at.
    assert [it["text"] for it in items] == ["first phrase", "second phrase"]
    assert [it["position"] for it in items] == [0, 1]
    # The snapshot is the ONLY source of verbatim for stages (no stage queries the
    # repository), so it must carry every persisted discriminator — dropping `kind`
    # is what made a slide-verbatim phrase get hard-asserted against the narration.
    assert [it["id"] for it in items] == [first_id, second_id]
    assert [it["kind"] for it in items] == ["tts", "slide"]
    # `scope`/`scope_target` are deliberately NOT carried: nothing reads them yet.
    # A half-implemented payload invites the next author to assume it works.
    assert "scope" not in items[0]
    assert "scope_target" not in items[0]


async def test_start_run_verbatim_items_empty_when_none(db_session) -> None:
    from src.database.repositories.project_repository import ProjectRepository
    from src.services.blueprints.registry import get_registry

    tenant = uuid4()
    subject = uuid4()
    proj_repo = ProjectRepository(db_session)
    proj = await proj_repo.create_with_deliverables(
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
            {
                "name": "d",
                "format": "16x9",
                "captions_mode": "none",
                "target_duration_seconds": 30,
            }
        ],
    )
    await db_session.commit()
    deliverable_id = proj.deliverables[0].id

    class _SessionCtx:
        async def __aenter__(self):
            return db_session

        async def __aexit__(self, *exc):
            return False

    async def fake_dispatch(run_id):
        return None

    engine = RunEngine(
        session_factory=lambda: _SessionCtx(),
        registry=get_registry(),
        dispatch=fake_dispatch,
    )
    run = await engine.start_run(
        deliverable_id=deliverable_id,
        tenant_id=tenant,
        subject_id=subject,
        parent_run_id=None,
        pause_after_stages=[],
    )

    assert run.config_snapshot["verbatim_items"] == []


async def _make_project(db_session, *, brief: dict, type_: str = "generic"):
    from src.database.repositories.project_repository import ProjectRepository

    tenant = uuid4()
    subject = uuid4()
    proj_repo = ProjectRepository(db_session)
    proj = await proj_repo.create_with_deliverables(
        tenant_id=tenant,
        owner_subject_id=subject,
        type_=type_,
        title="x",
        brief=brief,
        brand={},
        voice={"descriptor": "crisp"},
        theme={},
        visibility="tenant",
        deliverables=[
            {
                "name": "d",
                "format": "16x9",
                "captions_mode": "none",
                "target_duration_seconds": 30,
            }
        ],
    )
    await db_session.commit()
    return tenant, subject, proj


def _engine_for(db_session):
    from src.services.blueprints.registry import get_registry

    class _SessionCtx:
        async def __aenter__(self):
            return db_session

        async def __aexit__(self, *exc):
            return False

    async def fake_dispatch(run_id):
        return None

    return RunEngine(
        session_factory=lambda: _SessionCtx(),
        registry=get_registry(),
        dispatch=fake_dispatch,
    )


async def test_start_run_lifts_ai_visual_config_from_brief(db_session) -> None:
    img_cfg = {"enabled": True, "style": "photoreal"}
    vid_cfg = {"enabled": True, "model": "veo"}
    tenant, subject, proj = await _make_project(
        db_session,
        brief={
            "angle": "growth",
            "audience": "any",
            "target_duration_seconds": 30,
            "ai_image_gen": img_cfg,
            "ai_video_gen": vid_cfg,
        },
    )
    engine = _engine_for(db_session)
    run = await engine.start_run(
        deliverable_id=proj.deliverables[0].id,
        tenant_id=tenant,
        subject_id=subject,
        parent_run_id=None,
        pause_after_stages=[],
    )

    assert run.config_snapshot["ai_image_gen"] == img_cfg
    assert run.config_snapshot["ai_video_gen"] == vid_cfg


async def test_start_run_no_ai_visual_keys_when_brief_absent(db_session) -> None:
    """The brief lift is presence-opt-in: a blueprint with no seeded default (promo)
    gets no AI visual block when the brief carries none."""
    tenant, subject, proj = await _make_project(
        db_session,
        brief={"angle": "growth", "audience": "any", "target_duration_seconds": 30},
        type_="promo",
    )
    engine = _engine_for(db_session)
    run = await engine.start_run(
        deliverable_id=proj.deliverables[0].id,
        tenant_id=tenant,
        subject_id=subject,
        parent_run_id=None,
        pause_after_stages=[],
    )

    assert "ai_image_gen" not in run.config_snapshot
    assert "ai_video_gen" not in run.config_snapshot


async def test_start_run_generic_seeds_only_default_ai_image_gen(db_session) -> None:
    """generic is seeded a per-shot still default (storyboard prompts) and nothing
    else: ai_video_gen stays presence-opt-in."""
    tenant, subject, proj = await _make_project(
        db_session,
        brief={"angle": "growth", "audience": "any", "target_duration_seconds": 30},
    )
    engine = _engine_for(db_session)
    run = await engine.start_run(
        deliverable_id=proj.deliverables[0].id,
        tenant_id=tenant,
        subject_id=subject,
        parent_run_id=None,
        pause_after_stages=[],
    )

    assert run.config_snapshot["ai_image_gen"]["prompts_from_stage"] == "storyboard_draft"
    assert "ai_video_gen" not in run.config_snapshot


async def test_start_run_config_overrides_win_over_brief(db_session) -> None:
    brief_cfg = {"enabled": True, "style": "from_brief"}
    override_cfg = {"enabled": True, "style": "from_override"}
    tenant, subject, proj = await _make_project(
        db_session,
        brief={
            "angle": "growth",
            "audience": "any",
            "target_duration_seconds": 30,
            "ai_image_gen": brief_cfg,
        },
    )
    engine = _engine_for(db_session)
    run = await engine.start_run(
        deliverable_id=proj.deliverables[0].id,
        tenant_id=tenant,
        subject_id=subject,
        parent_run_id=None,
        pause_after_stages=[],
        config_overrides={"ai_image_gen": override_cfg},
    )

    assert run.config_snapshot["ai_image_gen"] == override_cfg


async def test_start_run_seeds_pause_defaults_from_blueprint_when_none(db_session) -> None:
    """When pause_after_stages is None (caller omitted), the engine seeds it
    from the blueprint's default_pause_after specs. Regression: pre-2026-05-31
    the field defaulted to ``[]`` which silently put every run on autopilot,
    contradicting blueprint declarations and the run-monitor UI's intent.

    Promo's blueprint declares pauses after promo_script + promo_storyboard;
    that's the set we expect to land on the persisted run.
    """
    from src.database.repositories.project_repository import ProjectRepository
    from src.services.blueprints.registry import get_registry

    tenant = uuid4()
    subject = uuid4()
    proj_repo = ProjectRepository(db_session)
    proj = await proj_repo.create_with_deliverables(
        tenant_id=tenant,
        owner_subject_id=subject,
        type_="promo",
        title="seed-pause-from-bp",
        brief={
            "audience": "founders",
            "value_proposition": "shipit",
            "call_to_action": "go",
            "target_duration_seconds": 30,
        },
        brand={},
        voice={"voice_id": "af_heart", "provider": "kokoro"},
        theme={},
        visibility="tenant",
        deliverables=[
            {
                "name": "d",
                "format": "16x9",
                "captions_mode": "none",
                "target_duration_seconds": 30,
            }
        ],
    )
    await db_session.commit()

    class _SessionCtx:
        async def __aenter__(self):
            return db_session

        async def __aexit__(self, *exc):
            return False

    async def _noop_dispatch(run_id):
        pass

    engine = RunEngine(
        session_factory=lambda: _SessionCtx(),
        registry=get_registry(),
        dispatch=_noop_dispatch,
    )
    run = await engine.start_run(
        deliverable_id=proj.deliverables[0].id,
        tenant_id=tenant,
        subject_id=subject,
        parent_run_id=None,
        pause_after_stages=None,  # caller omitted → seed from blueprint
    )
    assert set(run.pause_after_stages) == {"promo_script", "promo_storyboard"}


async def test_start_run_respects_explicit_empty_pause_list_as_autopilot(db_session) -> None:
    """``pause_after_stages=[]`` (explicit empty) stays autopilot — only
    None triggers the blueprint-defaults seeding. This is the escape hatch
    for callers that genuinely want no pauses."""
    from src.database.repositories.project_repository import ProjectRepository
    from src.services.blueprints.registry import get_registry

    tenant = uuid4()
    subject = uuid4()
    proj_repo = ProjectRepository(db_session)
    proj = await proj_repo.create_with_deliverables(
        tenant_id=tenant,
        owner_subject_id=subject,
        type_="promo",
        title="explicit-autopilot",
        brief={
            "audience": "founders",
            "value_proposition": "shipit",
            "call_to_action": "go",
            "target_duration_seconds": 30,
        },
        brand={},
        voice={"voice_id": "af_heart", "provider": "kokoro"},
        theme={},
        visibility="tenant",
        deliverables=[
            {
                "name": "d",
                "format": "16x9",
                "captions_mode": "none",
                "target_duration_seconds": 30,
            }
        ],
    )
    await db_session.commit()

    class _SessionCtx:
        async def __aenter__(self):
            return db_session

        async def __aexit__(self, *exc):
            return False

    async def _noop_dispatch(run_id):
        pass

    engine = RunEngine(
        session_factory=lambda: _SessionCtx(),
        registry=get_registry(),
        dispatch=_noop_dispatch,
    )
    run = await engine.start_run(
        deliverable_id=proj.deliverables[0].id,
        tenant_id=tenant,
        subject_id=subject,
        parent_run_id=None,
        pause_after_stages=[],  # explicit autopilot — keep empty
    )
    assert run.pause_after_stages == []


def _website_review_proj_kwargs(tenant, subject, brief):
    return dict(
        tenant_id=tenant,
        owner_subject_id=subject,
        type_="website_review",
        title="wr",
        brief={"target_url": "https://acme.test", **brief},
        brand={},
        voice={"voice_id": "af_heart", "provider": "kokoro"},
        theme={},
        visibility="tenant",
        deliverables=[
            {
                "name": "d",
                "format": "16x9",
                "captions_mode": "none",
                "target_duration_seconds": 60,
            }
        ],
    )


async def _start_seeded_website_review(db_session, tenant, subject, brief):
    from src.database.repositories.project_repository import ProjectRepository
    from src.services.blueprints.registry import get_registry

    proj = await ProjectRepository(db_session).create_with_deliverables(
        **_website_review_proj_kwargs(tenant, subject, brief)
    )
    await db_session.commit()

    class _SessionCtx:
        async def __aenter__(self):
            return db_session

        async def __aexit__(self, *exc):
            return False

    async def _noop_dispatch(run_id):
        pass

    engine = RunEngine(
        session_factory=lambda: _SessionCtx(),
        registry=get_registry(),
        dispatch=_noop_dispatch,
    )
    return await engine.start_run(
        deliverable_id=proj.deliverables[0].id,
        tenant_id=tenant,
        subject_id=subject,
        parent_run_id=None,
        pause_after_stages=None,  # caller omitted → seed from blueprint defaults
    )


async def test_website_review_video_seeds_no_review_checkpoints(db_session) -> None:
    """A website_review that WILL produce video now seeds NO checkpoints.

    Owner decision 2026-09-15: the redesign + script gates are removed from the
    blueprint — the UI never announced them, so a gated run read as stalled.
    """
    run = await _start_seeded_website_review(db_session, uuid4(), uuid4(), {"generate_video": True})
    assert run.pause_after_stages == []


# ``test_website_review_no_video_seeds_no_checkpoints`` DELETED 2026-09-15 for the same
# reason as the two opt-out tests in tests/services/blueprints/test_website_review.py:
# with the blueprint's gates removed the seed is ``[]`` for every brief, so asserting
# ``[] == []`` for ``generate_video: False`` pinned nothing and its name claimed a
# behaviour (`generate_video` drives autopilot) the code no longer has. Its branch was
# deleted from ``_seed_pause_after_stages`` with the gates.
#
# The test above covers what is left and CAN fail: restoring either
# ``default_pause_after=True`` reds it.


async def test_website_review_all_outputs_disabled_is_rejected(db_session) -> None:
    """The empty-run guard fires through start_run (ValueError → HTTP 400)."""
    with pytest.raises(ValueError, match="at least one output"):
        await _start_seeded_website_review(
            db_session,
            uuid4(),
            uuid4(),
            {
                "generate_critique": False,
                "generate_redesign": False,
                "generate_video": False,
            },
        )
