"""Studio v3 — /api/projects route tests (T8).

Exercises atomic create-with-deliverables, the lesson→course parent guard
(app-layer), tenant scoping (cross-tenant 404), and soft-delete + listing
include_archived semantics.
"""

from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from src.database.repositories.run_repository import RunRepository
from src.database.repositories.run_usage_repository import RunUsageRepository

pytestmark = pytest.mark.asyncio

_GENERIC_PROJECT = {
    "type": "generic",
    "title": "runs-history",
    "deliverables": [
        {"name": "d", "format": "16x9", "captions_mode": "none", "target_duration_seconds": 60}
    ],
}


async def test_post_project_creates_with_deliverables(client: AsyncClient) -> None:
    body = {
        "type": "promo",
        "title": "Hello promo",
        "brief": {"angle": "growth"},
        "deliverables": [
            {
                "name": "16x9 hero",
                "format": "16x9",
                "captions_mode": "none",
                "target_duration_seconds": 60,
            },
            {
                "name": "9x16 vert",
                "format": "9x16",
                "captions_mode": "burned_in",
                "target_duration_seconds": 30,
            },
        ],
    }
    r = await client.post("/api/projects", json=body)
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["title"] == "Hello promo"
    assert len(data["deliverables"]) == 2
    assert {d["format"] for d in data["deliverables"]} == {"16x9", "9x16"}
    detail = await client.get(f"/api/projects/{data['id']}")
    assert detail.status_code == 200
    assert len(detail.json()["deliverables"]) == 2


async def test_post_project_rejects_zero_deliverables(client: AsyncClient) -> None:
    r = await client.post(
        "/api/projects",
        json={"type": "generic", "title": "x", "deliverables": []},
    )
    assert r.status_code == 422


async def test_list_excludes_archived_by_default(client: AsyncClient) -> None:
    r = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "to-archive",
            "deliverables": [
                {
                    "name": "d",
                    "format": "16x9",
                    "captions_mode": "none",
                    "target_duration_seconds": 60,
                }
            ],
        },
    )
    pid = r.json()["id"]
    archived = await client.delete(f"/api/projects/{pid}")
    assert archived.status_code == 204
    listed = await client.get("/api/projects")
    assert pid not in [p["id"] for p in listed.json()["items"]]
    listed_all = await client.get("/api/projects?include_archived=true")
    assert pid in [p["id"] for p in listed_all.json()["items"]]


async def test_list_returns_new_metadata_fields(client: AsyncClient) -> None:
    r = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "meta",
            "deliverables": [
                {"name": "a", "format": "16x9", "captions_mode": "none"},
                {"name": "b", "format": "9x16", "captions_mode": "none"},
            ],
        },
    )
    assert r.status_code == 201, r.text
    listed = await client.get("/api/projects")
    item = next(p for p in listed.json()["items"] if p["id"] == r.json()["id"])
    assert item["deliverable_count"] == 2
    assert item["last_run_status"] is None
    assert item["finished_at"] is None
    assert item["status"] == "drafting"
    assert item["thumbnail_url"] is None


async def test_list_sort_param_accepts_finished_at(client: AsyncClient) -> None:
    r = await client.get("/api/projects?sort_by=finished_at&order=asc")
    assert r.status_code == 200


async def test_list_sort_param_rejects_unknown_column(client: AsyncClient) -> None:
    r = await client.get("/api/projects?sort_by=bogus")
    assert r.status_code == 422


async def test_patch_project_accepts_failed_status(client: AsyncClient) -> None:
    created = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "x",
            "deliverables": [{"name": "d", "format": "16x9", "captions_mode": "none"}],
        },
    )
    pid = created.json()["id"]
    r = await client.patch(f"/api/projects/{pid}", json={"status": "failed"})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "failed"
    listed = await client.get("/api/projects?status=failed")
    assert pid in [p["id"] for p in listed.json()["items"]]


async def test_lesson_must_have_course_parent(client: AsyncClient) -> None:
    r = await client.post(
        "/api/projects",
        json={
            "type": "lesson",
            "title": "orphan",
            "deliverables": [
                {
                    "name": "d",
                    "format": "16x9",
                    "captions_mode": "none",
                    "target_duration_seconds": 60,
                }
            ],
        },
    )
    assert r.status_code == 400
    assert "parent" in r.json()["detail"].lower()


async def test_cross_tenant_get_returns_404(
    client: AsyncClient, other_tenant_client: AsyncClient
) -> None:
    r = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "private",
            "deliverables": [
                {
                    "name": "d",
                    "format": "16x9",
                    "captions_mode": "none",
                    "target_duration_seconds": 60,
                }
            ],
        },
    )
    pid = r.json()["id"]
    cross = await other_tenant_client.get(f"/api/projects/{pid}")
    assert cross.status_code == 404


async def test_patch_project_updates_title_and_brief(client: AsyncClient) -> None:
    created = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "old",
            "deliverables": [
                {
                    "name": "d",
                    "format": "16x9",
                    "captions_mode": "none",
                    "target_duration_seconds": 60,
                }
            ],
        },
    )
    pid = created.json()["id"]
    r = await client.patch(f"/api/projects/{pid}", json={"title": "new", "brief": {"k": "v"}})
    assert r.status_code == 200
    assert r.json()["title"] == "new"
    detail = await client.get(f"/api/projects/{pid}")
    assert detail.json()["brief"] == {"k": "v"}


async def test_children_create_list_reorder(client: AsyncClient) -> None:
    parent = await client.post(
        "/api/projects",
        json={
            "type": "course",
            "title": "course-1",
            "deliverables": [
                {
                    "name": "d",
                    "format": "16x9",
                    "captions_mode": "none",
                    "target_duration_seconds": 60,
                }
            ],
        },
    )
    parent_id = parent.json()["id"]
    child_a = await client.post(
        f"/api/projects/{parent_id}/children",
        json={
            "type": "lesson",
            "title": "lesson-A",
            "deliverables": [
                {
                    "name": "d",
                    "format": "16x9",
                    "captions_mode": "none",
                    "target_duration_seconds": 60,
                }
            ],
        },
    )
    assert child_a.status_code == 201
    a_id = child_a.json()["id"]
    child_b = await client.post(
        f"/api/projects/{parent_id}/children",
        json={
            "type": "lesson",
            "title": "lesson-B",
            "deliverables": [
                {
                    "name": "d",
                    "format": "16x9",
                    "captions_mode": "none",
                    "target_duration_seconds": 60,
                }
            ],
        },
    )
    b_id = child_b.json()["id"]
    listed = await client.get(f"/api/projects/{parent_id}/children")
    assert listed.status_code == 200
    assert {p["id"] for p in listed.json()} == {a_id, b_id}
    reorder = await client.patch(
        f"/api/projects/{parent_id}/children/reorder",
        json={"ordered_child_ids": [b_id, a_id]},
    )
    assert reorder.status_code == 204


async def test_child_inherits_parent_brand_and_theme(client: AsyncClient) -> None:
    """A course's lesson children must render on-brand: when a child is created
    without explicit branding it inherits the parent's brand/voice/theme; an
    explicit override still wins (BUG-1 — lessons used to hardcode empty brand)."""
    brand = {"primary_color": "#280983", "name": "Specra"}
    theme = {"background": "#19023f", "text_color": "#F8FAFC"}
    deliverable = {
        "name": "d",
        "format": "16x9",
        "captions_mode": "none",
        "target_duration_seconds": 60,
    }
    parent = await client.post(
        "/api/projects",
        json={
            "type": "course",
            "title": "branded-course",
            "brand": brand,
            "theme": theme,
            "deliverables": [deliverable],
        },
    )
    parent_id = parent.json()["id"]

    inherited = await client.post(
        f"/api/projects/{parent_id}/children",
        json={"type": "lesson", "title": "inherits", "deliverables": [deliverable]},
    )
    assert inherited.status_code == 201
    detail = await client.get(f"/api/projects/{inherited.json()['id']}")
    assert detail.json()["brand"] == brand
    assert detail.json()["theme"] == theme

    own_brand = {"primary_color": "#000000", "name": "Override"}
    overridden = await client.post(
        f"/api/projects/{parent_id}/children",
        json={
            "type": "lesson",
            "title": "overrides",
            "brand": own_brand,
            "deliverables": [deliverable],
        },
    )
    over_detail = await client.get(f"/api/projects/{overridden.json()['id']}")
    assert over_detail.json()["brand"] == own_brand
    # theme still inherited (only brand was overridden)
    assert over_detail.json()["theme"] == theme


async def test_source_url_projected_from_brief_url(client: AsyncClient) -> None:
    """`source_url` is a typed top-level projection of `brief.source_url`
    (or the legacy `brief.url`). The FE consumes this stable field so it can
    drop the `brief as Record<string, unknown>` cast in the run page."""
    legacy = await client.post(
        "/api/projects",
        json={
            "type": "app_review",
            "title": "legacy-key",
            "brief": {"url": "https://example.com/legacy"},
            "deliverables": [{"name": "d", "format": "16x9", "captions_mode": "none"}],
        },
    )
    assert legacy.status_code == 201, legacy.text
    assert legacy.json()["source_url"] == "https://example.com/legacy"

    canonical = await client.post(
        "/api/projects",
        json={
            "type": "app_review",
            "title": "canonical-key",
            "brief": {"source_url": "https://example.com/canonical"},
            "deliverables": [{"name": "d", "format": "16x9", "captions_mode": "none"}],
        },
    )
    assert canonical.status_code == 201, canonical.text
    assert canonical.json()["source_url"] == "https://example.com/canonical"

    # canonical key wins when both are present (no surprise demotion if the
    # caller writes a NEW `source_url` over a stale `url`).
    both = await client.post(
        "/api/projects",
        json={
            "type": "app_review",
            "title": "both",
            "brief": {
                "url": "https://example.com/old",
                "source_url": "https://example.com/new",
            },
            "deliverables": [{"name": "d", "format": "16x9", "captions_mode": "none"}],
        },
    )
    assert both.status_code == 201, both.text
    assert both.json()["source_url"] == "https://example.com/new"

    # absent → null sourceUrl (not omitted) so the FE can rely on the key.
    missing = await client.post(
        "/api/projects",
        json={
            "type": "promo",
            "title": "no-url",
            "brief": {"angle": "growth"},
            "deliverables": [{"name": "d", "format": "16x9", "captions_mode": "none"}],
        },
    )
    assert missing.status_code == 201, missing.text
    assert missing.json()["source_url"] is None

    # list endpoint also serializes source_url so the FE list view shows it.
    listing = await client.get("/api/projects?type=app_review")
    assert listing.status_code == 200
    titles_to_url = {item["title"]: item["source_url"] for item in listing.json()["items"]}
    assert titles_to_url["legacy-key"] == "https://example.com/legacy"
    assert titles_to_url["canonical-key"] == "https://example.com/canonical"
    assert titles_to_url["both"] == "https://example.com/new"


async def test_source_url_projected_from_brief_target_url(client: AsyncClient) -> None:
    """website_review / app_review store their URL under `brief.target_url`
    (not `source_url`). The top-level `source_url` projection falls back to it
    so the gallery URL chip works for those types too — mirroring the FE run
    page's `target_url ?? source_url ?? url` cascade. `source_url` still wins
    when both are set."""
    target_only = await client.post(
        "/api/projects",
        json={
            "type": "website_review",
            "title": "target-only",
            "brief": {"target_url": "https://example.com/review-target"},
            "deliverables": [{"name": "d", "format": "16x9", "captions_mode": "none"}],
        },
    )
    assert target_only.status_code == 201, target_only.text
    assert target_only.json()["source_url"] == "https://example.com/review-target"

    # source_url wins over target_url when both are present.
    both = await client.post(
        "/api/projects",
        json={
            "type": "website_review",
            "title": "source-and-target",
            "brief": {
                "source_url": "https://example.com/source",
                "target_url": "https://example.com/target",
            },
            "deliverables": [{"name": "d", "format": "16x9", "captions_mode": "none"}],
        },
    )
    assert both.status_code == 201, both.text
    assert both.json()["source_url"] == "https://example.com/source"

    # list view projects target_url too.
    listing = await client.get("/api/projects?type=website_review")
    assert listing.status_code == 200
    titles_to_url = {item["title"]: item["source_url"] for item in listing.json()["items"]}
    assert titles_to_url["target-only"] == "https://example.com/review-target"
    assert titles_to_url["source-and-target"] == "https://example.com/source"


async def test_get_unknown_project_returns_404(client: AsyncClient) -> None:
    from uuid import uuid4

    r = await client.get(f"/api/projects/{uuid4()}")
    assert r.status_code == 404


# ─── child-lifecycle SSE + publishing ────────────────────────────────────


async def _drain_messages(pubsub, *, deadline_s: float = 2.0) -> list[dict]:
    """Pull every message available on the pubsub before the deadline."""
    import asyncio
    import json as _json

    out: list[dict] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + deadline_s
    while loop.time() < deadline:
        msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
        if msg is None:
            if out:
                break
            continue
        data = msg["data"]
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        out.append(_json.loads(data))
    return out


async def test_create_child_publishes_child_added(client, app) -> None:
    """POST /api/projects/{parent}/children publishes a child_added event
    to the parent's projects:<parent_id> Redis channel — what the FE
    EventSource subscription consumes."""
    parent = await client.post(
        "/api/projects",
        json={
            "type": "course",
            "title": "course-publish",
            "deliverables": [{"name": "d", "format": "16x9"}],
        },
    )
    parent_id = parent.json()["id"]

    pubsub = app.state.redis.pubsub()
    await pubsub.subscribe(f"projects:{parent_id}")
    try:
        child = await client.post(
            f"/api/projects/{parent_id}/children",
            json={
                "type": "lesson",
                "title": "lesson-publish",
                "deliverables": [{"name": "d", "format": "16x9"}],
            },
        )
        assert child.status_code == 201
        child_id = child.json()["id"]

        messages = await _drain_messages(pubsub)
        assert len(messages) == 1, messages
        msg = messages[0]
        assert msg["event"] == "child_added"
        assert msg["project_id"] == parent_id
        assert msg["child_id"] == child_id
    finally:
        await pubsub.unsubscribe(f"projects:{parent_id}")
        await pubsub.aclose()


async def test_reorder_children_publishes_child_reordered(client, app) -> None:
    """PATCH /api/projects/{parent}/children/reorder publishes a
    child_reordered event carrying the new order."""
    parent = await client.post(
        "/api/projects",
        json={
            "type": "course",
            "title": "course-reorder-publish",
            "deliverables": [{"name": "d", "format": "16x9"}],
        },
    )
    parent_id = parent.json()["id"]
    a = await client.post(
        f"/api/projects/{parent_id}/children",
        json={
            "type": "lesson",
            "title": "lesson-A",
            "deliverables": [{"name": "d", "format": "16x9"}],
        },
    )
    b = await client.post(
        f"/api/projects/{parent_id}/children",
        json={
            "type": "lesson",
            "title": "lesson-B",
            "deliverables": [{"name": "d", "format": "16x9"}],
        },
    )
    a_id = a.json()["id"]
    b_id = b.json()["id"]

    pubsub = app.state.redis.pubsub()
    await pubsub.subscribe(f"projects:{parent_id}")
    try:
        resp = await client.patch(
            f"/api/projects/{parent_id}/children/reorder",
            json={"ordered_child_ids": [b_id, a_id]},
        )
        assert resp.status_code == 204

        messages = await _drain_messages(pubsub)
        assert len(messages) == 1, messages
        msg = messages[0]
        assert msg["event"] == "child_reordered"
        assert msg["project_id"] == parent_id
        assert msg["order"] == [b_id, a_id]
    finally:
        await pubsub.unsubscribe(f"projects:{parent_id}")
        await pubsub.aclose()


async def test_project_events_unknown_project_returns_404(client: AsyncClient) -> None:
    """Synthetic project_id (or cross-tenant project_id) → 404. Same status
    code for both, so no info leak across the tenant boundary."""
    from uuid import uuid4

    r = await client.get(f"/api/projects/{uuid4()}/events")
    assert r.status_code == 404


async def test_project_events_cross_tenant_returns_404(
    client, other_tenant_client: AsyncClient
) -> None:
    """A project owned by tenant A cannot be subscribed to by tenant B."""
    # tenant A creates the project
    a_proj = await client.post(
        "/api/projects",
        json={
            "type": "course",
            "title": "owned-by-A",
            "deliverables": [{"name": "d", "format": "16x9"}],
        },
    )
    assert a_proj.status_code == 201
    a_id = a_proj.json()["id"]
    # tenant B's events subscription must 404 — same as for a missing id
    r = await other_tenant_client.get(f"/api/projects/{a_id}/events")
    assert r.status_code == 404


async def test_list_project_runs_newest_first(client: AsyncClient, db_session) -> None:
    """GET /api/projects/{id}/runs returns the project's runs newest-first with
    the RunSummary shape. Runs are seeded directly via the repository (sharing
    the test engine) so no dispatcher fires — deterministic, no stage mocks."""
    created = await client.post("/api/projects", json=_GENERIC_PROJECT)
    assert created.status_code == 201, created.text
    pid = created.json()["id"]
    detail = await client.get(f"/api/projects/{pid}")
    did = UUID(detail.json()["deliverables"][0]["id"])

    repo = RunRepository(db_session)
    run_a = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    run_b = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await db_session.commit()

    res = await client.get(f"/api/projects/{pid}/runs")
    assert res.status_code == 200, res.text
    rows = res.json()
    assert [r["id"] for r in rows] == [str(run_b.id), str(run_a.id)]
    first = rows[0]
    assert first["project_id"] == pid
    assert first["deliverable_id"] == str(did)
    assert first["blueprint_id"] == "generic_v1"
    assert first["status"] == "pending"
    assert first["created_at"] is not None


async def test_list_project_runs_carries_token_totals(client: AsyncClient, db_session) -> None:
    """Each RunSummary in the list carries its run's summed input/output tokens
    (the deliverable page showed none); a run with no usage rows reads 0/0."""
    created = await client.post("/api/projects", json=_GENERIC_PROJECT)
    assert created.status_code == 201, created.text
    pid = created.json()["id"]
    detail = await client.get(f"/api/projects/{pid}")
    did = UUID(detail.json()["deliverables"][0]["id"])

    repo = RunRepository(db_session)
    used = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    unused = await repo.create(
        deliverable_id=did, blueprint_id="generic_v1", config_snapshot={}, pause_after_stages=[]
    )
    await RunUsageRepository(db_session).replace_for_run(
        run_id=used.id,
        rows=[
            {"stage_name": "analyze", "input_tokens": 1200, "output_tokens": 300},
            {"stage_name": "script_draft", "input_tokens": 800, "output_tokens": 450},
        ],
    )
    await db_session.commit()

    res = await client.get(f"/api/projects/{pid}/runs")
    assert res.status_code == 200, res.text
    by_id = {r["id"]: r for r in res.json()}
    assert (by_id[str(used.id)]["input_tokens"], by_id[str(used.id)]["output_tokens"]) == (
        2000,
        750,
    )
    assert (by_id[str(unused.id)]["input_tokens"], by_id[str(unused.id)]["output_tokens"]) == (
        0,
        0,
    )


async def test_list_project_runs_unknown_project_404(client: AsyncClient) -> None:
    r = await client.get(f"/api/projects/{uuid4()}/runs")
    assert r.status_code == 404


async def test_list_project_runs_cross_tenant_404(
    client: AsyncClient, other_tenant_client: AsyncClient
) -> None:
    created = await client.post("/api/projects", json=_GENERIC_PROJECT)
    pid = created.json()["id"]
    cross = await other_tenant_client.get(f"/api/projects/{pid}/runs")
    assert cross.status_code == 404


# ─── Showcase brief: door validation + typed projection ──────────────────

_SHOWCASE_DELIVERABLE = [
    {"name": "16x9", "format": "16x9", "captions_mode": "none", "target_duration_seconds": 60}
]


def _showcase_body(brief: dict) -> dict:
    return {
        "type": "showcase",
        "title": "showcase test",
        "brief": brief,
        "deliverables": _SHOWCASE_DELIVERABLE,
    }


async def test_showcase_generated_brief_passes(client: AsyncClient) -> None:
    # Generated mode: target_url + focus areas, no package/lines.
    r = await client.post(
        "/api/projects",
        json=_showcase_body(
            {"target_url": "https://app.example.com", "focus_areas": ["onboarding"]}
        ),
    )
    assert r.status_code == 201, r.text
    assert r.json()["showcase_brief"]["target_url"] == "https://app.example.com"


async def test_showcase_authored_brief_passes(client: AsyncClient) -> None:
    # Authored mode: lines present ⇒ package_url required (and here supplied).
    body = _showcase_body(
        {
            "target_url": "https://app.example.com",
            "package_url": "https://minio.test/pkg.zip",
            "lines": [
                {"id": "clip-1", "text": "Open the dashboard.", "seconds": 4, "clip": "@clip-1"},
                {"text": "Create a project.", "clip_anchor": "end"},
            ],
        }
    )
    r = await client.post("/api/projects", json=body)
    assert r.status_code == 201, r.text
    sb = r.json()["showcase_brief"]
    assert sb["package_url"] == "https://minio.test/pkg.zip"
    assert len(sb["lines"]) == 2
    # The '@clip-1' tag is normalized to the 'clip-1' FTR join key at the door.
    assert sb["lines"][0]["clip"] == "clip-1"


async def test_showcase_missing_target_url_422(client: AsyncClient) -> None:
    r = await client.post("/api/projects", json=_showcase_body({"focus_areas": ["x"]}))
    assert r.status_code == 422, r.text


@pytest.mark.parametrize("blank", [""])
async def test_showcase_empty_target_url_422(client: AsyncClient, blank: str) -> None:
    # A present-but-blank target_url must 422 at the door too — min_length=1 rejects
    # the empty string (str_strip_whitespace collapses whitespace first) instead of
    # letting it survive to a RuntimeError three stages deep in gen/record.
    r = await client.post("/api/projects", json=_showcase_body({"target_url": blank}))
    assert r.status_code == 422, r.text


async def test_showcase_line_empty_text_422(client: AsyncClient) -> None:
    body = _showcase_body(
        {
            "target_url": "https://app.example.com",
            "package_url": "https://minio.test/pkg.zip",
            "lines": [{"text": ""}],
        }
    )
    r = await client.post("/api/projects", json=body)
    assert r.status_code == 422, r.text


async def test_showcase_lines_without_package_422(client: AsyncClient) -> None:
    body = _showcase_body(
        {
            "target_url": "https://app.example.com",
            "lines": [{"text": "Do a thing."}],
        }
    )
    r = await client.post("/api/projects", json=body)
    assert r.status_code == 422, r.text


async def test_showcase_valid_voice_cast_passes(client: AsyncClient) -> None:
    # A named multi-voice cast of in-catalog kokoro ids is accepted at the door.
    body = _showcase_body({"target_url": "https://app.example.com"})
    body["voice"] = {
        "provider": "kokoro",
        "voice_id": "af_heart",
        "cast": {"mark": "am_michael", "lisa": "af_bella", "daniel": "bm_daniel"},
    }
    r = await client.post("/api/projects", json=body)
    assert r.status_code == 201, r.text


async def test_showcase_unknown_voice_id_422(client: AsyncClient) -> None:
    # A typo'd top-level voice_id would 500 the TTS backend three stages deep;
    # reject it at the door instead.
    body = _showcase_body({"target_url": "https://app.example.com"})
    body["voice"] = {"voice_id": "af_notreal"}
    r = await client.post("/api/projects", json=body)
    assert r.status_code == 422, r.text


async def test_showcase_unknown_cast_id_422(client: AsyncClient) -> None:
    # A typo'd cast entry silently falls back to the wrong voice today; catch it.
    body = _showcase_body({"target_url": "https://app.example.com"})
    body["voice"] = {"voice_id": "af_heart", "cast": {"mark": "xx_not_a_voice"}}
    r = await client.post("/api/projects", json=body)
    assert r.status_code == 422, r.text


async def test_non_kokoro_provider_skips_catalog_check(client: AsyncClient) -> None:
    # openai/elevenlabs voices use their own namespaces — not validated against
    # the kokoro catalog, so a non-kokoro voice_id is accepted.
    body = _showcase_body({"target_url": "https://app.example.com"})
    body["voice"] = {"provider": "openai", "voice_id": "shimmer"}
    r = await client.post("/api/projects", json=body)
    assert r.status_code == 201, r.text


async def test_showcase_brief_projection_round_trips_on_read(client: AsyncClient) -> None:
    created = await client.post(
        "/api/projects",
        json=_showcase_body({"target_url": "https://app.example.com", "focus_areas": ["a"]}),
    )
    pid = created.json()["id"]
    got = await client.get(f"/api/projects/{pid}")
    assert got.status_code == 200
    assert got.json()["showcase_brief"]["target_url"] == "https://app.example.com"


async def test_showcase_extra_brief_keys_preserved_in_projection(client: AsyncClient) -> None:
    # extra="allow": non-typed brief keys (e.g. ken_burns) round-trip on read.
    r = await client.post(
        "/api/projects",
        json=_showcase_body({"target_url": "https://app.example.com", "ken_burns": True}),
    )
    assert r.status_code == 201, r.text
    assert r.json()["showcase_brief"]["ken_burns"] is True


async def test_legacy_partial_showcase_brief_reads_without_500(client: AsyncClient) -> None:
    # A brief that no longer validates (e.g. a pre-typed legacy project, simulated
    # via PATCH which does not re-run the create-time door validator) must degrade
    # to showcase_brief=None on read, never 500.
    created = await client.post(
        "/api/projects",
        json=_showcase_body({"target_url": "https://app.example.com"}),
    )
    pid = created.json()["id"]
    patched = await client.patch(f"/api/projects/{pid}", json={"brief": {"legacy": "no-url"}})
    assert patched.status_code == 200, patched.text
    got = await client.get(f"/api/projects/{pid}")
    assert got.status_code == 200
    assert got.json()["showcase_brief"] is None


# ─── brief.legal door validation (claim_finish / A18 opt-in gate) ────────────
# claim_finish wakes ONLY on an explicit legal.mode of auto|critique_only, and it
# runs in BOTH the showcase and demo_video blueprints. Until this door existed a
# typo ("critique-only", "Auto") got a 201 and then silently skipped the legal
# stage for the whole run — the operator only found out by reading stage rows.
# The block must stay OPT-IN though: the legal block already ships in prod as
# {"trademarks": [...], "defensibility": "..."} with NO mode, and that shape must
# keep validating (and keep claim_finish inert).


def _demo_video_body(brief: dict) -> dict:
    return {
        "type": "demo_video",
        "title": "demo_video test",
        "brief": brief,
        "deliverables": _SHOWCASE_DELIVERABLE,
    }


@pytest.mark.parametrize("mode", ["auto"])
async def test_showcase_valid_legal_mode_passes(client: AsyncClient, mode: str) -> None:
    r = await client.post(
        "/api/projects",
        json=_showcase_body({"target_url": "https://app.example.com", "legal": {"mode": mode}}),
    )
    assert r.status_code == 201, r.text
    assert r.json()["showcase_brief"]["legal"]["mode"] == mode


@pytest.mark.parametrize("bad_mode", ["critique-only"])
async def test_showcase_typod_legal_mode_422(client: AsyncClient, bad_mode: str) -> None:
    # The whole point: a mode the stage does not recognise must fail at the door,
    # not silently disable the legal gate for the entire run.
    r = await client.post(
        "/api/projects",
        json=_showcase_body({"target_url": "https://app.example.com", "legal": {"mode": bad_mode}}),
    )
    assert r.status_code == 422, r.text


async def test_showcase_legal_block_without_mode_still_passes(client: AsyncClient) -> None:
    # REGRESSION GUARD for the legal block already shipping in prod: no `mode` key,
    # other consumers' keys only. Must validate, round-trip verbatim, and leave
    # claim_finish inert (mode absent ⇒ stage skips, unchanged behaviour).
    legal = {"trademarks": ["Acme®", "Acme Cloud™"], "defensibility": "reviewed by counsel"}
    created = await client.post(
        "/api/projects",
        json=_showcase_body({"target_url": "https://app.example.com", "legal": legal}),
    )
    assert created.status_code == 201, created.text
    got = await client.get(f"/api/projects/{created.json()['id']}")
    assert got.json()["brief"]["legal"] == legal
    assert "mode" not in got.json()["brief"]["legal"]


async def test_showcase_brief_without_legal_passes(client: AsyncClient) -> None:
    r = await client.post(
        "/api/projects", json=_showcase_body({"target_url": "https://app.example.com"})
    )
    assert r.status_code == 201, r.text
    assert r.json()["showcase_brief"]["legal"] is None


async def test_showcase_legal_claims_block_passes(client: AsyncClient) -> None:
    # The full opt-in shape the stage reads: mode + posture + claims knobs.
    brief = {
        "target_url": "https://app.example.com",
        "legal": {
            "mode": "critique_only",
            "posture": "conservative",
            "claims": {
                "allow_superlatives": False,
                "on_unsupported": "flag",
                "iterations": 2,
                "score_threshold": 4.0,
            },
        },
    }
    created = await client.post("/api/projects", json=_showcase_body(brief))
    assert created.status_code == 201, created.text
    got = await client.get(f"/api/projects/{created.json()['id']}")
    # Persisted verbatim — the stage reads this dict, not the typed model.
    assert got.json()["brief"]["legal"] == brief["legal"]


@pytest.mark.parametrize("claims", [["we are the fastest"]])
async def test_showcase_legal_non_dict_claims_still_passes(
    client: AsyncClient, claims: object
) -> None:
    # NON-TIGHTENING guard. `mode` is the ONLY typed key on LegalBrief: claim_finish
    # itself does `if not isinstance(claims_cfg, dict): claims_cfg = {}`, so a
    # non-dict legal.claims is anticipated in the wild. Typing the sub-block would
    # 422 briefs that create fine today — it must not.
    r = await client.post(
        "/api/projects",
        json=_showcase_body({"target_url": "https://app.example.com", "legal": {"claims": claims}}),
    )
    assert r.status_code == 201, r.text


async def test_showcase_legal_unknown_keys_preserved(client: AsyncClient) -> None:
    # extra='allow' on LegalBrief: every non-`mode` key round-trips untouched.
    brief = {
        "target_url": "https://app.example.com",
        "legal": {"mode": "auto", "future_knob": 1, "claims": {"future_claims_knob": True}},
    }
    created = await client.post("/api/projects", json=_showcase_body(brief))
    assert created.status_code == 201, created.text
    got = await client.get(f"/api/projects/{created.json()['id']}")
    assert got.json()["brief"]["legal"] == brief["legal"]


async def test_demo_video_valid_legal_mode_passes(client: AsyncClient) -> None:
    r = await client.post("/api/projects", json=_demo_video_body({"legal": {"mode": "auto"}}))
    assert r.status_code == 201, r.text


@pytest.mark.parametrize("bad_mode", ["critique-only"])
async def test_demo_video_typod_legal_mode_422(client: AsyncClient, bad_mode: str) -> None:
    # demo_video runs claim_finish too, so its door must reject a bad mode as well.
    r = await client.post("/api/projects", json=_demo_video_body({"legal": {"mode": bad_mode}}))
    assert r.status_code == 422, r.text


async def test_demo_video_arbitrary_brief_still_passes(client: AsyncClient) -> None:
    # demo_video had NO door model before this change. The new one must reject a bad
    # legal.mode and NOTHING else — no required fields, extra='allow'.
    r = await client.post(
        "/api/projects",
        json=_demo_video_body(
            {
                "source_url": "https://app.example.com",
                "legal": {"trademarks": ["Acme®"]},
                "anything": {"nested": [1, 2, 3]},
            }
        ),
    )
    assert r.status_code == 201, r.text


async def test_demo_video_empty_brief_passes(client: AsyncClient) -> None:
    r = await client.post("/api/projects", json=_demo_video_body({}))
    assert r.status_code == 201, r.text


# ─── seo_audit / cro_audit / marketing_scorecard door validation ─────────────
# All single-URL audit types share the AuditBrief door model, so a missing/empty/
# non-string target_url must 422 at create instead of surviving to a StagePrecondition
# failure one stage deep — the same fail-at-the-door contract showcase enforces.


def _audit_body(audit_type: str, brief: dict) -> dict:
    return {
        "type": audit_type,
        "title": f"{audit_type} test",
        "brief": brief,
        "deliverables": _SHOWCASE_DELIVERABLE,
    }


@pytest.mark.parametrize("audit_type", ["seo_audit"])
async def test_audit_valid_brief_passes(client: AsyncClient, audit_type: str) -> None:
    r = await client.post(
        "/api/projects",
        json=_audit_body(audit_type, {"target_url": "https://example.com", "include_llm": True}),
    )
    assert r.status_code == 201, r.text
    # extra='allow' + source_url cascade: target_url is projected on read.
    assert r.json()["source_url"] == "https://example.com"


@pytest.mark.parametrize("audit_type", ["seo_audit"])
async def test_audit_missing_target_url_422(client: AsyncClient, audit_type: str) -> None:
    r = await client.post("/api/projects", json=_audit_body(audit_type, {"include_llm": True}))
    assert r.status_code == 422, r.text


@pytest.mark.parametrize("audit_type", ["seo_audit"])
@pytest.mark.parametrize("blank", [""])
async def test_audit_blank_target_url_422(client: AsyncClient, audit_type: str, blank: str) -> None:
    r = await client.post("/api/projects", json=_audit_body(audit_type, {"target_url": blank}))
    assert r.status_code == 422, r.text


@pytest.mark.parametrize("audit_type", ["seo_audit"])
async def test_audit_non_string_target_url_422(client: AsyncClient, audit_type: str) -> None:
    r = await client.post("/api/projects", json=_audit_body(audit_type, {"target_url": 123}))
    assert r.status_code == 422, r.text


@pytest.mark.parametrize("audit_type", ["seo_audit"])
async def test_audit_extra_brief_keys_preserved(client: AsyncClient, audit_type: str) -> None:
    # extra='allow': non-typed keys round-trip (the door only rejects a bad target_url).
    created = await client.post(
        "/api/projects",
        json=_audit_body(audit_type, {"target_url": "https://example.com", "custom": "keep"}),
    )
    assert created.status_code == 201, created.text
    got = await client.get(f"/api/projects/{created.json()['id']}")
    assert got.json()["brief"]["custom"] == "keep"


@pytest.mark.parametrize("audit_type", ["seo_audit"])
async def test_audit_generate_video_flag_accepted(client: AsyncClient, audit_type: str) -> None:
    # generate_video is now a typed AuditBrief field (drives the optional video chain);
    # both truthy and omitted must pass the door and round-trip in the persisted brief.
    on = await client.post(
        "/api/projects",
        json=_audit_body(audit_type, {"target_url": "https://example.com", "generate_video": True}),
    )
    assert on.status_code == 201, on.text
    got = await client.get(f"/api/projects/{on.json()['id']}")
    assert got.json()["brief"]["generate_video"] is True

    # Omitted → scan-only (default False), still valid.
    off = await client.post(
        "/api/projects",
        json=_audit_body(audit_type, {"target_url": "https://example.com"}),
    )
    assert off.status_code == 201, off.text


@pytest.mark.parametrize("audit_type", ["seo_audit"])
async def test_audit_brief_projection_round_trips_on_read(
    client: AsyncClient, audit_type: str
) -> None:
    # The typed audit_brief projection surfaces the validated flags on read (same
    # pattern as showcase_brief) so the FE gets a generated contract.
    created = await client.post(
        "/api/projects",
        json=_audit_body(audit_type, {"target_url": "https://example.com", "generate_video": True}),
    )
    assert created.status_code == 201, created.text
    got = await client.get(f"/api/projects/{created.json()['id']}")
    assert got.json()["audit_brief"]["target_url"] == "https://example.com"
    assert got.json()["audit_brief"]["generate_video"] is True


@pytest.mark.parametrize(
    ("status", "stored_reason", "expected"),
    [("paused", "checkpoint", "checkpoint")],
)
async def test_list_reports_the_latest_runs_pause_reason_only_while_paused(
    client: AsyncClient, db_session_factory, status: str, stored_reason: str, expected
) -> None:
    from sqlalchemy import update

    from src.database.models import Run

    r = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "pause reason",
            "deliverables": [{"name": "a", "format": "16x9", "captions_mode": "none"}],
        },
    )
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    did = (await client.get(f"/api/projects/{pid}")).json()["deliverables"][0]["id"]
    async with db_session_factory() as s:
        run = await RunRepository(s).create(
            deliverable_id=UUID(did),
            blueprint_id="generic_v1",
            config_snapshot={},
            pause_after_stages=[],
        )
        await s.execute(
            update(Run).where(Run.id == run.id).values(status=status, pause_reason=stored_reason)
        )
        await s.commit()
    listed = await client.get("/api/projects")
    item = next(p for p in listed.json()["items"] if p["id"] == pid)
    assert item["last_run_status"] == status
    assert item["last_run_pause_reason"] == expected
