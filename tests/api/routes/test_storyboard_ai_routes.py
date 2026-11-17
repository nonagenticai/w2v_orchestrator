"""HTTP-level tests for the Storyboard Phase-1 AI authoring routes.

Exercises the two new endpoints through the real FastAPI app (``client`` +
``app`` fixtures, real postgres via ``db_session``):

* ``POST /api/projects/{project_id}/storyboard/draft``
* ``POST /api/storyboard/panels/{panel_id}/rewrite``

The LLM is stubbed by OVERRIDING the ``get_assist_service`` dependency with a
fake-WM-backed service. The override declares ``session: SessionDep`` so it
shares the request's DB session (FastAPI caches ``get_db_session`` within a
request), meaning the service persists into the same transaction the route
commits. Mirrors the fixture usage of ``test_storyboard_routes.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.clients import WMResponse

from src.api.routes._common import get_db_session
from src.api.routes.storyboard import get_assist_service
from src.database.models import Project
from src.database.repositories.storyboard_repository import (
    StoryboardPanelRepository,
    StoryboardRepository,
)
from src.services.storyboard.assist import StoryboardAssistService

pytestmark = pytest.mark.asyncio


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
        title="storyboard-ai",
        brief=brief or {},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


def _fake(output: dict[str, Any]) -> AsyncMock:
    fake = AsyncMock()
    fake.message.return_value = WMResponse(
        output=output,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )
    return fake


def _install_fake(app: FastAPI, fake: AsyncMock) -> None:
    """Override ``get_assist_service`` with a fake-WM service that shares the
    request DB session (via ``Depends(get_db_session)``)."""

    def _factory(session: AsyncSession = Depends(get_db_session)) -> StoryboardAssistService:
        return StoryboardAssistService(session, client=fake)

    app.dependency_overrides[get_assist_service] = _factory


def _board() -> dict[str, Any]:
    return {
        "assets": [{"name": "Alice", "kind": "character", "canonical_descriptor": "hero"}],
        "scenes": [
            {
                "name": "Opening",
                "int_ext": "EXT",
                "location": "Downtown",
                "panels": [
                    {
                        "action_description": "Alice walks in",
                        "dialogue": "Hi.",
                        "shot_size": "MS",
                        "characters": ["Alice"],
                        "props": ["sword"],
                        "location": "Downtown",
                    },
                ],
            },
        ],
    }


# ─── draft ───────────────────────────────────────────────────────────────
async def test_draft_happy_returns_201_with_board(client, app, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    _install_fake(app, _fake(_board()))
    try:
        r = await client.post(
            f"/api/projects/{project.id}/storyboard/draft",
            json={"script_or_idea": "A hero saves the town"},
        )
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 201, r.text
    body = r.json()
    assert len(body["scenes"]) == 1
    assert len(body["panels"]) == 1
    assert body["panels"][0]["shot_key"] == "clip-1"
    assert body["panels"][0]["source"] == "ai"
    assert any(a["name"] == "Alice" for a in body["assets"])


async def test_draft_conflict_without_replace_returns_409(client, app, db_session) -> None:
    tid = UUID(client.tenant_id)
    project = await _seed_project(db_session, tenant_id=tid, subject_id=UUID(client.subject_id))
    sb = await StoryboardRepository(db_session).ensure_for_project(project.id, tenant_id=tid)
    assert sb is not None
    await StoryboardPanelRepository(db_session).create(
        project_id=project.id,
        storyboard_id=sb.id,
        position=0,
        shot_key=None,
        action_description="existing",
    )
    await db_session.commit()

    _install_fake(app, _fake(_board()))
    try:
        r = await client.post(
            f"/api/projects/{project.id}/storyboard/draft",
            json={"script_or_idea": "new idea", "replace": False},
        )
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 409, r.text


async def test_draft_no_text_and_empty_brief_returns_422(client, app, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id), brief={}
    )
    _install_fake(app, _fake(_board()))
    try:
        r = await client.post(f"/api/projects/{project.id}/storyboard/draft", json={})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 422, r.text


async def test_draft_cross_tenant_project_returns_404(client, app, db_session) -> None:
    # Project belongs to a DIFFERENT tenant → _require_project must 404 before
    # the service (and the LLM) is ever reached (tenant isolation invariant).
    other_project = await _seed_project(db_session, tenant_id=uuid4(), subject_id=uuid4())
    fake = _fake(_board())
    _install_fake(app, fake)
    try:
        r = await client.post(
            f"/api/projects/{other_project.id}/storyboard/draft",
            json={"script_or_idea": "A hero saves the town"},
        )
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 404, r.text
    fake.message.assert_not_called()


# ─── rewrite ─────────────────────────────────────────────────────────────
async def test_rewrite_happy_returns_200_with_revision_bump(client, app, db_session) -> None:
    tid = UUID(client.tenant_id)
    project = await _seed_project(db_session, tenant_id=tid, subject_id=UUID(client.subject_id))
    sb = await StoryboardRepository(db_session).ensure_for_project(project.id, tenant_id=tid)
    assert sb is not None
    panel = await StoryboardPanelRepository(db_session).create(
        project_id=project.id,
        storyboard_id=sb.id,
        position=0,
        shot_key=None,
        dialogue="old",
        shot_size="MS",
    )
    await db_session.commit()

    _install_fake(app, _fake({"dialogue": "new punchy line", "shot_size": "CU"}))
    try:
        r = await client.post(
            f"/api/storyboard/panels/{panel.id}/rewrite",
            json={"instruction": "make it punchy"},
        )
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dialogue"] == "new punchy line"
    assert body["shot_size"] == "CU"
    assert body["revision"] == 2
    assert body["source"] == "ai"


async def test_rewrite_missing_panel_returns_404(client, app, db_session) -> None:
    _install_fake(app, _fake({"dialogue": "x"}))
    try:
        r = await client.post(
            f"/api/storyboard/panels/{uuid4()}/rewrite",
            json={"instruction": "anything"},
        )
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 404, r.text


async def test_rewrite_missing_instruction_returns_422(client) -> None:
    # Pydantic validation (instruction has min_length=1) fires before the
    # handler runs — no dependency override needed.
    missing = await client.post(f"/api/storyboard/panels/{uuid4()}/rewrite", json={})
    assert missing.status_code == 422, missing.text
    empty = await client.post(f"/api/storyboard/panels/{uuid4()}/rewrite", json={"instruction": ""})
    assert empty.status_code == 422, empty.text
