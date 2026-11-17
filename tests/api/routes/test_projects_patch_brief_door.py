"""The typed-brief door on the wizard's draft-finalize path.

``ProjectCreate._validate_brief`` 422s a malformed typed brief, but the wizard
never POSTs ``/projects``: it creates a draft, PATCHes the brief in pieces, then
adds deliverables. That path skipped the door entirely, so an ``app_review``
draft with no ``target_url`` reached a paid stage.

Rule: a deliverable-less draft accepts partial briefs on PATCH; adding its FIRST
deliverable (the draft -> project transition) validates the stored brief, with the
same 422 body as create's. ONLY that transition validates: a later brief PATCH on a
project that already has a deliverable stays unvalidated, so live projects with
legacy partial briefs remain editable.
"""

import pytest

from tests.conftest import SeededClient

pytestmark = pytest.mark.asyncio

_DELIVERABLE = {"name": "main", "format": "16x9"}


async def _draft(client: SeededClient) -> str:
    r = await client.post("/api/projects/draft", json={"type": "app_review", "title": "door"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def test_draft_accepts_partial_brief_while_in_progress(client: SeededClient) -> None:
    pid = await _draft(client)
    r = await client.patch(f"/api/projects/{pid}", json={"brief": {"competitor_urls": []}})
    assert r.status_code == 200, r.text


async def test_first_deliverable_rejects_invalid_draft_brief(client: SeededClient) -> None:
    pid = await _draft(client)
    await client.patch(f"/api/projects/{pid}", json={"brief": {"competitor_urls": []}})
    r = await client.post(f"/api/projects/{pid}/deliverables", json=_DELIVERABLE)
    assert r.status_code == 422, r.text
    assert "target_url" in r.text
    detail = await client.get(f"/api/projects/{pid}")
    assert detail.json()["deliverables"] == []


async def test_first_deliverable_accepts_valid_draft_brief(client: SeededClient) -> None:
    pid = await _draft(client)
    await client.patch(f"/api/projects/{pid}", json={"brief": {"target_url": "https://a.example"}})
    r = await client.post(f"/api/projects/{pid}/deliverables", json=_DELIVERABLE)
    assert r.status_code == 201, r.text


async def test_first_deliverable_422_has_the_create_shape(client: SeededClient) -> None:
    pid = await _draft(client)
    await client.patch(f"/api/projects/{pid}", json={"brief": {"target_url": " "}})
    finalized = await client.post(f"/api/projects/{pid}/deliverables", json=_DELIVERABLE)
    assert finalized.status_code == 422, finalized.text

    created = await client.post(
        "/api/projects",
        json={
            "type": "app_review",
            "title": "door",
            "brief": {"target_url": " "},
            "deliverables": [_DELIVERABLE],
        },
    )
    assert created.status_code == 422, created.text
    assert finalized.json() == created.json()


async def test_patch_after_first_deliverable_stays_unvalidated(client: SeededClient) -> None:
    """A live project with a legacy partial brief must remain editable."""
    pid = await _draft(client)
    await client.patch(f"/api/projects/{pid}", json={"brief": {"target_url": "https://a.example"}})
    assert (await client.post(f"/api/projects/{pid}/deliverables", json=_DELIVERABLE)).is_success

    patched = await client.patch(f"/api/projects/{pid}", json={"brief": {"legacy": "no-url"}})
    assert patched.status_code == 200, patched.text
    # A second deliverable is not the draft -> project transition: no door either.
    second = await client.post(
        f"/api/projects/{pid}/deliverables", json={"name": "alt", "format": "9x16"}
    )
    assert second.status_code == 201, second.text
