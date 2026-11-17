"""Service-level unit tests for the Storyboard Phase-1 AI authoring service.

Exercises ``StoryboardAssistService`` by direct construction with a fake
Worker-Management client (``AsyncMock`` returning a canned ``WMResponse``),
against a real postgres session (the ``db_session`` fixture). No HTTP, no LLM.

Covers the FROZEN CONTRACT ``draft()`` + ``rewrite()`` behaviours:
persistence, global ``clip-<n>`` spine numbering across scenes, asset dedup +
on-the-fly asset creation, enum coercion (draft drops-to-None; rewrite skips),
the 409 conflict / 422 empty-input guards, ``replace=True`` asset preservation,
revision bump, and cross-tenant / missing-panel ``None`` returns.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from w2v_common.auth.identity_context import clear_identity_context, set_identity_context
from w2v_common.clients import WMResponse
from w2v_common.tenant import TenantSource

from src.api.schemas import StoryboardDraftRequest
from src.database.models import Project
from src.database.repositories.storyboard_repository import (
    StoryboardAssetRepository,
    StoryboardPanelRepository,
    StoryboardRepository,
    StoryboardSceneRepository,
)
from src.services.llm import WMClient
from src.services.storyboard.assist import (
    DraftConflictError,
    DraftInputError,
    StoryboardAssistService,
)
from tests.support.platform_llm import CHAT_PATH, PlatformChat, use_ir_backend

pytestmark = pytest.mark.asyncio


@pytest.fixture
def client_ids() -> tuple[UUID, UUID]:
    """A (tenant_id, subject_id) pair for service-level tests (no HTTP client)."""
    return uuid4(), uuid4()


# ─── helpers ─────────────────────────────────────────────────────────────
async def _seed_project(
    db_session, *, tenant_id: UUID, subject_id: UUID, brief: dict[str, Any] | None = None
) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="storyboard-assist",
        brief=brief or {},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


def _resp(output: dict[str, Any]) -> WMResponse:
    return WMResponse(
        output=output,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _fake(output: dict[str, Any]) -> AsyncMock:
    fake = AsyncMock()
    fake.message.return_value = _resp(output)
    return fake


def _board() -> dict[str, Any]:
    """A 2-scene / 3-panel board: 'Alice' recurs in every panel (dedup → 1
    asset), 'Downtown' is both a declared asset and a panel location, 'sword'
    is a prop referenced but NOT pre-declared (created on the fly)."""
    return {
        "assets": [
            {"name": "Alice", "kind": "character", "canonical_descriptor": "brave hero"},
            {"name": "Downtown", "kind": "location", "canonical_descriptor": "busy street"},
        ],
        "scenes": [
            {
                "name": "Opening",
                "int_ext": "EXT",
                "location": "Downtown",
                "time_of_day": "day",
                "description": "Alice arrives.",
                "panels": [
                    {
                        "action_description": "Alice walks in",
                        "dialogue": "Here at last.",
                        "shot_number": "1A",
                        "shot_size": "MS",
                        "angle": "eye_level",
                        "movement": "static",
                        "transition_out": "cut",
                        "duration_seconds": 3.0,
                        "characters": ["Alice"],
                        "props": ["sword"],
                        "location": "Downtown",
                    },
                    {
                        "action_description": "Close on her face",
                        "dialogue": "",
                        "shot_size": "CU",
                        "characters": ["alice"],  # different case → same asset
                        "props": [],
                        "location": "Downtown",
                    },
                ],
            },
            {
                "name": "Reveal",
                "int_ext": "INT",
                "location": "Lair",
                "time_of_day": "night",
                "description": "The confrontation.",
                "panels": [
                    {
                        "action_description": "Alice faces the villain",
                        "dialogue": "It ends now.",
                        "shot_size": "WS",
                        "characters": ["Alice"],
                        "props": [],
                    },
                ],
            },
        ],
    }


def _svc(db_session, output: dict[str, Any]) -> StoryboardAssistService:
    return StoryboardAssistService(db_session, client=_fake(output))


# ─── draft: happy path ───────────────────────────────────────────────────
async def test_draft_persists_board_with_global_clip_spine(client_ids, db_session) -> None:
    tid, sid = client_ids
    project = await _seed_project(db_session, tenant_id=tid, subject_id=sid)
    svc = _svc(db_session, _board())

    scenes, panels, assets = await svc.draft(
        project_id=project.id,
        tenant_id=tid,
        req=StoryboardDraftRequest(script_or_idea="A hero saves the town"),
    )

    assert len(scenes) == 2
    assert len(panels) == 3
    # clip-<n> is a SINGLE global counter that spans scene boundaries.
    assert [p.shot_key for p in panels] == ["clip-1", "clip-2", "clip-3"]
    assert [p.position for p in panels] == [0, 1, 2]
    for p in panels:
        assert p.source == "ai"
        assert p.gen["worker"] == "storyboard-draft"

    # 'Alice' referenced 3x (incl. a lowercase variant) + declared once → ONE row.
    all_assets = await StoryboardAssetRepository(db_session).list_for_project(
        project.id, tenant_id=tid
    )
    chars = [a for a in all_assets if a.kind == "character" and a.name.lower() == "alice"]
    assert len(chars) == 1
    alice = chars[0]
    for p in panels:
        assert p.characters == [str(alice.id)]

    # panel 'location' name resolves to a REAL location asset UUID.
    downtown = next(a for a in all_assets if a.kind == "location" and a.name == "Downtown")
    assert panels[0].location_asset_id == downtown.id
    assert isinstance(panels[0].location_asset_id, UUID)

    # 'sword' was never declared but is referenced → created on the fly.
    assert any(a.name.lower() == "sword" for a in all_assets)
    # the service returns the full asset library.
    assert {a.id for a in assets} >= {alice.id, downtown.id}


# ─── draft: conflict / replace ───────────────────────────────────────────
async def test_draft_conflict_when_board_has_panels_and_no_replace(client_ids, db_session) -> None:
    tid, sid = client_ids
    project = await _seed_project(db_session, tenant_id=tid, subject_id=sid)
    sb = await StoryboardRepository(db_session).ensure_for_project(project.id, tenant_id=tid)
    assert sb is not None
    await StoryboardPanelRepository(db_session).create(
        project_id=project.id,
        storyboard_id=sb.id,
        position=0,
        shot_key=None,
        action_description="pre-existing",
    )
    await db_session.commit()

    svc = _svc(db_session, _board())
    with pytest.raises(DraftConflictError):
        await svc.draft(
            project_id=project.id,
            tenant_id=tid,
            req=StoryboardDraftRequest(script_or_idea="new idea", replace=False),
        )

    # Nothing replaced: the single pre-existing panel is intact.
    remaining = await StoryboardPanelRepository(db_session).list_for_project(
        project.id, tenant_id=tid
    )
    assert len(remaining) == 1
    assert remaining[0].action_description == "pre-existing"


async def test_draft_replace_swaps_panels_and_scenes_but_keeps_assets(
    client_ids, db_session
) -> None:
    tid, sid = client_ids
    project = await _seed_project(db_session, tenant_id=tid, subject_id=sid)

    # First draft populates the board + asset library.
    await _svc(db_session, _board()).draft(
        project_id=project.id,
        tenant_id=tid,
        req=StoryboardDraftRequest(script_or_idea="first"),
    )
    assets_before = await StoryboardAssetRepository(db_session).list_for_project(
        project.id, tenant_id=tid
    )
    alice_before = next(a for a in assets_before if a.name.lower() == "alice")

    # Second draft with replace=True: a smaller board.
    smaller = {
        "assets": [],
        "scenes": [
            {
                "name": "Solo",
                "panels": [
                    {"action_description": "Alice alone", "characters": ["Alice"], "props": []},
                ],
            },
        ],
    }
    scenes, panels, _ = await _svc(db_session, smaller).draft(
        project_id=project.id,
        tenant_id=tid,
        req=StoryboardDraftRequest(script_or_idea="redo", replace=True),
    )
    assert len(scenes) == 1
    assert len(panels) == 1
    # Old spine gone, replaced 1:1.
    live_panels = await StoryboardPanelRepository(db_session).list_for_project(
        project.id, tenant_id=tid
    )
    assert [p.shot_key for p in live_panels] == ["clip-1"]
    live_scenes = await StoryboardSceneRepository(db_session).list_for_project(
        project.id, tenant_id=tid
    )
    assert len(live_scenes) == 1

    # Assets preserved: 'Alice' is the SAME row (not duplicated, not deleted).
    assets_after = await StoryboardAssetRepository(db_session).list_for_project(
        project.id, tenant_id=tid
    )
    alice_after = [a for a in assets_after if a.name.lower() == "alice"]
    assert len(alice_after) == 1
    assert alice_after[0].id == alice_before.id
    assert panels[0].characters == [str(alice_before.id)]


# ─── draft: enum coercion + input guards ─────────────────────────────────
async def test_draft_invalid_enums_coerce_to_none_not_error(client_ids, db_session) -> None:
    tid, sid = client_ids
    project = await _seed_project(db_session, tenant_id=tid, subject_id=sid)
    board = {
        "assets": [],
        "scenes": [
            {
                "name": "S",
                "panels": [
                    {
                        "action_description": "x",
                        "shot_size": "WIDE",
                        "angle": "tilted",
                        "characters": [],
                        "props": [],
                    },
                ],
            }
        ],
    }

    _, panels, _ = await _svc(db_session, board).draft(
        project_id=project.id,
        tenant_id=tid,
        req=StoryboardDraftRequest(script_or_idea="idea"),
    )
    assert len(panels) == 1
    # 'WIDE' / 'tilted' are not valid enum members → coerced to None (no raise).
    assert panels[0].shot_size is None
    assert panels[0].angle is None


async def test_draft_empty_input_raises(client_ids, db_session) -> None:
    tid, sid = client_ids
    project = await _seed_project(db_session, tenant_id=tid, subject_id=sid, brief={})

    fake = _fake(_board())
    svc = StoryboardAssistService(db_session, client=fake)
    with pytest.raises(DraftInputError):
        await svc.draft(
            project_id=project.id,
            tenant_id=tid,
            req=StoryboardDraftRequest(script_or_idea=None),
        )
    # The LLM is never called when there is nothing to draft from.
    fake.message.assert_not_called()


async def test_draft_falls_back_to_project_brief(client_ids, db_session) -> None:
    tid, sid = client_ids
    project = await _seed_project(
        db_session, tenant_id=tid, subject_id=sid, brief={"idea": "A hero saves the town"}
    )

    _, panels, _ = await _svc(db_session, _board()).draft(
        project_id=project.id,
        tenant_id=tid,
        req=StoryboardDraftRequest(script_or_idea=None),  # brief supplies the text
    )
    assert len(panels) == 3


# ─── rewrite ─────────────────────────────────────────────────────────────
async def _seed_panel(db_session, *, tenant_id: UUID, subject_id: UUID, **fields: Any):
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    sb = await StoryboardRepository(db_session).ensure_for_project(project.id, tenant_id=tenant_id)
    assert sb is not None
    panel = await StoryboardPanelRepository(db_session).create(
        project_id=project.id, storyboard_id=sb.id, position=0, shot_key=None, **fields
    )
    await db_session.commit()
    return project, panel


async def test_rewrite_updates_fields_and_bumps_revision(client_ids, db_session) -> None:
    tid, sid = client_ids
    _, panel = await _seed_panel(
        db_session,
        tenant_id=tid,
        subject_id=sid,
        dialogue="old",
        shot_size="MS",
        status="draft",
    )
    original_key, original_pos = panel.shot_key, panel.position

    svc = _svc(db_session, {"dialogue": "new punchy line", "shot_size": "CU"})
    row = await svc.rewrite(panel.id, tenant_id=tid, instruction="make it punchy", fields=None)

    assert row is not None
    assert row.dialogue == "new punchy line"
    assert row.shot_size == "CU"
    assert row.revision == 2  # bumped from the default 1
    assert row.source == "ai"
    # Immutable spine fields are untouched.
    assert row.shot_key == original_key
    assert row.position == original_pos
    assert row.status == "draft"


async def test_rewrite_skips_invalid_enum_keeping_current_value(client_ids, db_session) -> None:
    tid, sid = client_ids
    _, panel = await _seed_panel(
        db_session,
        tenant_id=tid,
        subject_id=sid,
        shot_size="MS",
    )

    svc = _svc(db_session, {"shot_size": "HUGE"})  # invalid enum member
    row = await svc.rewrite(panel.id, tenant_id=tid, instruction="tighten", fields=None)

    assert row is not None
    # Invalid enum → field skipped, current value preserved (no exception).
    assert row.shot_size == "MS"
    assert row.revision == 2


async def test_rewrite_missing_panel_returns_none(client_ids, db_session) -> None:
    tid, _ = client_ids
    svc = _svc(db_session, {"dialogue": "x"})
    assert await svc.rewrite(uuid4(), tenant_id=tid, instruction="x", fields=None) is None


async def test_rewrite_cross_tenant_panel_returns_none(client_ids, db_session) -> None:
    tid, sid = client_ids
    _, panel = await _seed_panel(db_session, tenant_id=tid, subject_id=sid, dialogue="mine")

    svc = _svc(db_session, {"dialogue": "stolen"})
    other_tenant = uuid4()
    assert await svc.rewrite(panel.id, tenant_id=other_tenant, instruction="x", fields=None) is None


# ─── w2v project attribution on the LLM call ─────────────────────────────
#
# inference-router attributes a call by ``X-Project-Id``; storyboard assist runs in
# REQUEST scope (no run bound), so it names its W2V project explicitly. Driven
# through the orchestrator's REAL ``WMClient`` on the ir backend; only the socket
# (``in_process.inner_transport_factory``) is a PlatformChat.


@pytest.fixture
def ir_chat(monkeypatch: pytest.MonkeyPatch) -> Any:
    def _install(output: dict[str, Any]) -> PlatformChat:
        use_ir_backend(monkeypatch)
        # IR delegation refuses an unvouched tenant; a request-scope caller has one.
        set_identity_context(str(uuid4()), None, None, tenant_source=TenantSource.LICENSE_TOKEN)
        return PlatformChat(lambda _body, _n: output).install(monkeypatch)

    yield _install
    clear_identity_context()


async def test_draft_attributes_the_llm_call_to_the_storyboards_project(
    client_ids, db_session, ir_chat
) -> None:
    tid, sid = client_ids
    project = await _seed_project(db_session, tenant_id=tid, subject_id=sid)
    chat = ir_chat(_board())
    svc = StoryboardAssistService(db_session, client=WMClient(timeout=30.0))

    await svc.draft(
        project_id=project.id,
        tenant_id=tid,
        req=StoryboardDraftRequest(script_or_idea="A hero saves the town"),
    )

    assert chat.paths == [CHAT_PATH]
    assert chat.requests[0].headers["X-Worker-Name"] == "storyboard-draft"
    assert chat.requests[0].headers["X-Project-Id"] == str(project.id)


async def test_rewrite_attributes_the_llm_call_to_the_panels_project(
    client_ids, db_session, ir_chat
) -> None:
    tid, sid = client_ids
    project, panel = await _seed_panel(db_session, tenant_id=tid, subject_id=sid, dialogue="old")
    chat = ir_chat({"dialogue": "new"})
    svc = StoryboardAssistService(db_session, client=WMClient(timeout=30.0))

    row = await svc.rewrite(panel.id, tenant_id=tid, instruction="punchier", fields=["dialogue"])

    assert row is not None and row.dialogue == "new"
    assert len(chat.requests) == 1
    assert chat.requests[0].headers["X-Project-Id"] == str(project.id)
