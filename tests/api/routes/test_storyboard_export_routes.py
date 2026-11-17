"""DB-backed integration tests for the storyboard export endpoints.

Covers ``GET /api/projects/{project_id}/storyboard/export.{csv,edl,fcpxml,pdf}``
mounted by ``src/api/routes/storyboard.py``: the download shape (content-type +
``attachment`` filename), a parseable body, and the ``_require_project``
ownership 404 (missing + cross-tenant). Mirrors the seeding + fixtures of
``test_storyboard_routes.py``.
"""

from __future__ import annotations

import csv
import io
from typing import Any, cast
from uuid import UUID, uuid4
from xml.etree.ElementTree import fromstring

import pytest
from httpx import AsyncClient
from w2v_common.net import SSRFError

from src.api.routes import storyboard as storyboard_routes
from src.config import clear_settings_cache
from src.database.models import Project
from src.database.repositories.storyboard_repository import StoryboardPanelRepository
from src.services.storyboard.csv_export import CSV_COLUMNS
from src.services.tts_storage import set_storage_for_tests
from src.services.visual_media import fetch_image_bytes
from tests.support.tiny_image import tiny_png as _png

pytestmark = pytest.mark.asyncio


async def _seed_project(db_session, *, tenant_id: UUID, subject_id: UUID) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="storyboard-csv-export",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def test_export_csv_download_shape(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    await client.post(
        f"/api/projects/{project.id}/storyboard/scenes",
        json={"name": "Opening", "int_ext": "INT", "location": "Office"},
    )
    a = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"vo_text": "hello, world", "shot_size": "MS", "duration_seconds": 3.5},
    )
    b = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"shot_key": "clip-9", "action_description": "second"},
    )
    assert a.status_code == 201 and b.status_code == 201, (a.text, b.text)

    resp = await client.get(f"/api/projects/{project.id}/storyboard/export.csv")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/csv")
    assert (
        resp.headers["content-disposition"] == f'attachment; filename="storyboard-{project.id}.csv"'
    )

    rows = list(csv.reader(io.StringIO(resp.text)))
    assert rows[0] == CSV_COLUMNS
    # header + one row per panel
    assert len(rows) == 3
    shot_keys = [r[CSV_COLUMNS.index("shot_key")] for r in rows[1:]]
    assert shot_keys == ["clip-1", "clip-9"]


async def test_export_csv_empty_board_header_only(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    resp = await client.get(f"/api/projects/{project.id}/storyboard/export.csv")
    assert resp.status_code == 200, resp.text
    rows = list(csv.reader(io.StringIO(resp.text)))
    assert rows == [CSV_COLUMNS]


async def test_export_csv_unknown_project_returns_404(client) -> None:
    resp = await client.get(f"/api/projects/{uuid4()}/storyboard/export.csv")
    assert resp.status_code == 404


async def test_export_csv_cross_tenant_returns_404(
    client, other_tenant_client: AsyncClient, db_session
) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    resp = await other_tenant_client.get(f"/api/projects/{project.id}/storyboard/export.csv")
    assert resp.status_code == 404


async def test_export_edl_download_shape(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    a = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"action_description": "hero enters", "duration_seconds": 4.0},
    )
    b = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"shot_key": "clip-9", "action_description": "second", "duration_seconds": 2.0},
    )
    assert a.status_code == 201 and b.status_code == 201, (a.text, b.text)

    resp = await client.get(f"/api/projects/{project.id}/storyboard/export.edl")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/plain")
    assert (
        resp.headers["content-disposition"] == f'attachment; filename="storyboard-{project.id}.edl"'
    )
    lines = resp.text.splitlines()
    assert lines[0] == "TITLE: STORYBOARD"
    assert lines[1] == "FCM: NON-DROP FRAME"
    events = [ln for ln in lines if ln[:3].isdigit()]
    assert len(events) == 2


async def test_export_edl_cross_tenant_returns_404(
    client, other_tenant_client: AsyncClient, db_session
) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    resp = await other_tenant_client.get(f"/api/projects/{project.id}/storyboard/export.edl")
    assert resp.status_code == 404


async def test_export_fcpxml_download_shape(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    a = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"shot_key": "clip-1", "duration_seconds": 4.0},
    )
    b = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"shot_key": "clip-2", "duration_seconds": 2.0},
    )
    assert a.status_code == 201 and b.status_code == 201, (a.text, b.text)

    resp = await client.get(f"/api/projects/{project.id}/storyboard/export.fcpxml")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/xml")
    assert resp.headers["content-disposition"] == (
        f'attachment; filename="storyboard-{project.id}.fcpxml"'
    )
    root = fromstring(resp.text)  # noqa: S314 — trusted, self-generated
    assert root.tag == "fcpxml"
    spine = root.find("./library/event/project/sequence/spine")
    assert spine is not None
    assert len(list(spine)) == 2


async def test_export_fcpxml_cross_tenant_returns_404(
    client, other_tenant_client: AsyncClient, db_session
) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    resp = await other_tenant_client.get(f"/api/projects/{project.id}/storyboard/export.fcpxml")
    assert resp.status_code == 404


# ─── PDF board ───────────────────────────────────────────────────────────────
# The PDF is the only export that wants BYTES, so these also pin the frame-
# resolution contract: SSRF-guarded, best-effort, never fatal.


async def test_export_pdf_download_shape(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    scene = await client.post(
        f"/api/projects/{project.id}/storyboard/scenes",
        json={"name": "Opening", "int_ext": "INT", "location": "Office"},
    )
    assert scene.status_code == 201, scene.text
    a = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={
            "scene_id": scene.json()["id"],
            "action_description": "hero enters",
            "shot_size": "MS",
            "duration_seconds": 3.5,
        },
    )
    b = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"shot_key": "clip-9", "dialogue": "second"},
    )
    assert a.status_code == 201 and b.status_code == 201, (a.text, b.text)

    resp = await client.get(f"/api/projects/{project.id}/storyboard/export.pdf")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/pdf")
    assert (
        resp.headers["content-disposition"] == f'attachment; filename="storyboard-{project.id}.pdf"'
    )
    assert resp.content.startswith(b"%PDF")
    assert resp.content.rstrip().endswith(b"%%EOF")


async def test_export_pdf_empty_board_still_renders(client, db_session) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    resp = await client.get(f"/api/projects/{project.id}/storyboard/export.pdf")
    assert resp.status_code == 200, resp.text
    assert resp.content.startswith(b"%PDF")


async def test_export_pdf_panel_without_image_uses_placeholder(client, db_session) -> None:
    """The common case today — image generation is not built, so no panel has a frame."""
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    created = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={"action_description": "no frame yet"},
    )
    assert created.status_code == 201, created.text

    resp = await client.get(f"/api/projects/{project.id}/storyboard/export.pdf")
    assert resp.status_code == 200, resp.text
    assert resp.content.startswith(b"%PDF")
    # No raster was embedded → the placeholder well was drawn.
    assert b"/Subtype /Image" not in resp.content


async def test_export_pdf_unfetchable_image_still_produces_a_pdf(client, db_session) -> None:
    """A dead/blocked media URL degrades to the placeholder — it must not 500."""
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    created = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={
            "action_description": "frame that will not load",
            "media_ref": {
                "type": "ai_image",
                "url": "https://storyboard-pdf-export.invalid/frame.png",
            },
        },
    )
    assert created.status_code == 201, created.text

    resp = await client.get(f"/api/projects/{project.id}/storyboard/export.pdf")
    assert resp.status_code == 200, resp.text
    assert resp.content.startswith(b"%PDF")
    assert b"/Subtype /Image" not in resp.content


async def test_export_pdf_embeds_a_resolved_frame(client, db_session, monkeypatch) -> None:
    """The positive half of the frame-resolution wiring.

    Without this, the "degrades to placeholder" tests would still pass if the
    route never resolved an image at all. The SSRF-guarded fetch itself is stubbed
    (no network in unit tests); everything else — media_ref → URL → bytes → PDF —
    is the real path.
    """
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    created = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={
            "action_description": "framed",
            "media_ref": {"type": "ai_image", "url": "https://cdn.example.com/frame.png"},
        },
    )
    assert created.status_code == 201, created.text

    seen: list[str] = []

    async def _fake_fetch(url: str, **kwargs: object) -> tuple[bytes, str]:
        seen.append(url)
        return _png(), "png"

    monkeypatch.setattr(storyboard_routes, "fetch_image_bytes", _fake_fetch)

    resp = await client.get(f"/api/projects/{project.id}/storyboard/export.pdf")
    assert resp.status_code == 200, resp.text
    assert seen == ["https://cdn.example.com/frame.png"]
    assert resp.content.startswith(b"%PDF")
    assert b"/Subtype /Image" in resp.content


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:6300/api/projects"],
)
async def test_pdf_frame_fetch_rejects_private_targets(url: str) -> None:
    """The SSRF guard must REJECT, not fetch.

    ``fetch_image_bytes`` → ``w2v_common.net.safe_get`` is the chokepoint the PDF
    route uses; asserting ``SSRFError`` here proves the route cannot be turned
    into an in-cluster read primitive by a crafted ``media_ref.url``. Note
    ``ip.is_private`` is False for CGNAT 100.64/10 — the whole Tailnet — so a
    hand-rolled check would have let that row through.
    """
    with pytest.raises(SSRFError):
        await fetch_image_bytes(url, timeout=2.0, max_bytes=1024)


async def test_pdf_frame_fetch_rejects_non_http_scheme() -> None:
    with pytest.raises(ValueError, match="non-http"):
        await fetch_image_bytes("file:///etc/passwd", timeout=2.0, max_bytes=1024)


async def test_export_pdf_unknown_project_returns_404(client) -> None:
    resp = await client.get(f"/api/projects/{uuid4()}/storyboard/export.pdf")
    assert resp.status_code == 404


async def test_export_pdf_cross_tenant_returns_404(
    client, other_tenant_client: AsyncClient, db_session
) -> None:
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    resp = await other_tenant_client.get(f"/api/projects/{project.id}/storyboard/export.pdf")
    assert resp.status_code == 404


async def test_export_pdf_resolves_frames_only_through_the_ssrf_chokepoint(
    client, db_session, monkeypatch
) -> None:
    """The route must reach the network THROUGH the SSRF guard, not beside it.

    ``test_pdf_frame_fetch_rejects_private_targets`` proves the guard REFUSES a
    private target; it does not prove this route ever asks it. That gap is real
    and mutation-visible: swapping ``fetch_image_bytes`` for a raw
    ``client.get`` leaves every other test in this file green, because a raw
    fetch of ``169.254.169.254`` also fails and BOTH paths degrade to the same
    placeholder. "No image in the PDF" cannot tell a refusal apart from a failed
    connection, so no assertion about the OUTPUT can catch the bypass.

    Assert the call instead. Together with the guard's own test this closes the
    chain: route -> guarded chokepoint -> chokepoint refuses. Deliberately NOT
    asserted via a real hostname — ``safe_get`` rejects at DNS resolution, so
    any "control" host that does not resolve records no send either, and the
    counter would prove nothing. The raw-send count is kept as the second half:
    the guard is asked AND nothing goes around it.
    """
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    created = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={
            "action_description": "hostile frame",
            "media_ref": {"type": "ai_image", "url": "http://169.254.169.254/latest/meta-data/"},
        },
    )
    assert created.status_code == 201, created.text

    import httpx as _httpx

    asked: list[str] = []
    raw_sends: list[str] = []
    real_send = _httpx.AsyncClient.send

    async def _spy_fetch(url: str, **kwargs: object) -> tuple[bytes, str]:
        asked.append(url)
        # What the real guard does with this target, proven separately by
        # test_pdf_frame_fetch_rejects_private_targets.
        raise SSRFError(f"URL {url} resolves to blocked address")

    async def _counting_send(self, request, **kwargs):  # type: ignore[no-untyped-def]
        # The ASGI test client is itself an httpx.AsyncClient, so its own call
        # INTO the app must not be counted as an outbound frame fetch.
        if request.url.host != "testserver":
            raw_sends.append(str(request.url))
        return await real_send(self, request, **kwargs)

    monkeypatch.setattr(storyboard_routes, "fetch_image_bytes", _spy_fetch)
    monkeypatch.setattr(_httpx.AsyncClient, "send", _counting_send)

    resp = await client.get(f"/api/projects/{project.id}/storyboard/export.pdf")

    assert asked == ["http://169.254.169.254/latest/meta-data/"], (
        "the route did not resolve the frame through the SSRF-guarded chokepoint"
    )
    assert raw_sends == [], f"the route opened a connection around the guard: {raw_sends}"
    # And a refused frame is still a usable board, not a 500.
    assert resp.status_code == 200, resp.text
    assert resp.content.startswith(b"%PDF")
    assert b"/Subtype /Image" not in resp.content


# ─── the artifact host is estate-internal, and the guard blocks it ───────────
_PDF_BUCKET = "w2v-orchestrator"


def _own_frame_url(project_id: UUID) -> str:
    """A frame url exactly as panel generation writes one for this project."""
    return (
        f"https://minio.test/{_PDF_BUCKET}/artifacts/storyboard/{project_id}"
        f"/storyboard_panel_image/frame-1.png"
    )


def _guarded_fetch(calls: list[tuple[str, tuple[str, ...] | None]]):
    """A ``fetch_image_bytes`` stand-in that enforces the ONE rule under test.

    The real guard lives in ``w2v_common.net.safe_get`` and its ``allow_hosts``
    behaviour is proven end-to-end in ``tests/services/test_visual_media.py``;
    what these route tests must pin is the DECISION the export makes — which url
    gets the allowlist and which does not — so the stub refuses any estate name
    that was not explicitly allowed, exactly as the real guard does.
    """

    async def _fetch(url: str, **kwargs: object) -> tuple[bytes, str]:
        allow = cast("tuple[str, ...] | None", kwargs.get("allow_hosts"))
        calls.append((url, allow))
        host = url.split("//", 1)[-1].split("/", 1)[0]
        permitted = {h.split(":", 1)[0] for h in (allow or ())}
        if host.split(":", 1)[0].endswith(".test") and host.split(":", 1)[0] not in permitted:
            raise SSRFError(f"Blocked hostname: {host} (estate-internal / metadata name)")
        return _png(), "png"

    return _fetch


async def test_export_pdf_embeds_a_frame_from_our_own_artifact_store(
    client, db_session, monkeypatch
) -> None:
    """The board PDF had NEVER embedded a frame.

    Panel frames live on ``minio.test`` — an estate-internal name the SSRF
    guard blocks by design — and ``fetch_image_bytes`` exposed no way to permit
    it, so every frame raised ``SSRFError``, landed in ``unavailable``, and the
    export still returned 200 with a placeholder. Nothing ever failed; the frame
    slot was simply inert.
    """
    monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "minio.test:8389")
    clear_settings_cache()
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    created = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={
            "action_description": "a frame that really exists",
            "media_ref": {"type": "ai_image", "url": _own_frame_url(project.id)},
        },
    )
    assert created.status_code == 201, created.text

    calls: list[tuple[str, tuple[str, ...] | None]] = []
    monkeypatch.setattr(storyboard_routes, "fetch_image_bytes", _guarded_fetch(calls))

    panels = await StoryboardPanelRepository(db_session).list_for_project(
        project.id, tenant_id=UUID(client.tenant_id)
    )
    images, unavailable = await storyboard_routes._resolve_panel_images(
        panels, tenant_id=UUID(client.tenant_id)
    )

    assert unavailable == set(), "our own frame was still refused"
    assert set(images) == {p.id for p in panels}
    assert calls and calls[0][1] is not None, "the fetch was not given the artifact allowlist"
    assert "minio.test" in (calls[0][1] or ()), (
        "the allowlist was not derived from STORAGE_MINIO_ENDPOINT"
    )

    resp = await client.get(f"/api/projects/{project.id}/storyboard/export.pdf")
    assert resp.status_code == 200, resp.text
    assert b"/Subtype /Image" in resp.content, "the frame still degraded to a placeholder"


async def test_export_pdf_refuses_a_client_pointed_internal_url(
    client, db_session, monkeypatch
) -> None:
    """The allowlist must not become a client-steerable SSRF hole.

    ``media_ref`` is writable on panel create AND patch, so a tenant can point a
    panel at any internal host. The allowlist is therefore applied per-url and
    only after ``panel_object_ref`` confirms the object sits under this panel's
    own project/tenant prefix; anything else keeps the full guard and degrades to
    the ``FRAME_UNAVAILABLE_CAPTION`` placeholder.

    Two shapes, both refused: another internal service entirely, and a
    well-formed minio.test key belonging to a DIFFERENT project.
    """
    monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "minio.test:8389")
    clear_settings_cache()
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    for poisoned in (
        "https://licenses.test/admin/secrets.png",
        _own_frame_url(uuid4()),  # a stranger's project prefix, right host
    ):
        created = await client.post(
            f"/api/projects/{project.id}/storyboard/panels",
            json={
                "action_description": "poisoned",
                "media_ref": {"type": "ai_image", "url": poisoned},
            },
        )
        assert created.status_code == 201, created.text

    calls: list[tuple[str, tuple[str, ...] | None]] = []
    monkeypatch.setattr(storyboard_routes, "fetch_image_bytes", _guarded_fetch(calls))

    panels = await StoryboardPanelRepository(db_session).list_for_project(
        project.id, tenant_id=UUID(client.tenant_id)
    )
    images, unavailable = await storyboard_routes._resolve_panel_images(
        panels, tenant_id=UUID(client.tenant_id)
    )

    assert images == {}, "a client-supplied internal url was fetched"
    assert unavailable == {p.id for p in panels}
    assert all(allow is None for _url, allow in calls), (
        "an unconfined url was handed the artifact allowlist"
    )


# ─── presigned-frame expiry on the export path ───────────────────────────────
class _ResigningStorage:
    """Storage that re-mints an owned MinIO URL with a visibly fresh signature.

    Mirrors ``tests/api/routes/test_storyboard_panel_image_routes.py``'s fake:
    an owned URL gets a new signature, a foreign one is declined (``None``).
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def resign_presigned_url(self, url: str) -> str | None:
        self.calls.append(url)
        if "minio.test" not in url:
            return None
        return f"{url.split('?', 1)[0]}?X-Amz-Signature=fresh"


async def test_export_pdf_re_signs_an_expired_frame_url(client, db_session, monkeypatch) -> None:
    """A generated frame is presigned for ONE HOUR; a board is exported for days.

    ``_panel_resp`` already re-signs ``media_ref.url`` on the API read path, so the
    board UI shows the image. The PDF export reads the URL straight off the ORM row,
    so an hour after generation it fetches an EXPIRED signature, the fetch fails, and
    every frame silently degrades to a placeholder captioned "no frame yet" — telling
    a director the frame was never rendered when in fact it exists.
    """
    storage = _ResigningStorage()
    set_storage_for_tests(cast("Any", storage))
    try:
        project = await _seed_project(
            db_session,
            tenant_id=UUID(client.tenant_id),
            subject_id=UUID(client.subject_id),
        )
        stale = "https://minio.test/buckets/concept/frame.png?X-Amz-Signature=expired"
        created = await client.post(
            f"/api/projects/{project.id}/storyboard/panels",
            json={
                "action_description": "generated an hour ago",
                "media_ref": {"type": "ai_image", "url": stale},
            },
        )
        assert created.status_code == 201, created.text

        seen: list[str] = []

        async def _fake_fetch(url: str, **kwargs: object) -> tuple[bytes, str]:
            seen.append(url)
            if "X-Amz-Signature=fresh" not in url:
                # What MinIO really answers an expired signature with.
                raise RuntimeError("Request has expired")
            return _png(), "png"

        monkeypatch.setattr(storyboard_routes, "fetch_image_bytes", _fake_fetch)

        resp = await client.get(f"/api/projects/{project.id}/storyboard/export.pdf")
        assert resp.status_code == 200, resp.text
        assert seen, "the export never attempted a frame fetch"
        assert all("X-Amz-Signature=fresh" in url for url in seen), (
            f"the export fetched a STALE presigned URL: {seen}"
        )
        assert b"/Subtype /Image" in resp.content, "the frame degraded to a placeholder"
    finally:
        set_storage_for_tests(None)


async def test_export_pdf_resigns_the_stored_frame_url_before_fetching(
    client, db_session, monkeypatch
) -> None:
    """The stored frame url is presigned for ONE HOUR; a board is read for days.

    ``_panel_resp`` re-signs on the API response path, so the board's own UI
    shows images fine. This export read ``media_ref["url"]`` straight off the
    ORM row, so every real export more than an hour after generation fetched an
    EXPIRED signature, every fetch failed, and every frame silently degraded to
    a placeholder — a board that looks entirely un-rendered while the UI beside
    it shows the art. Nothing 500s, so the caption is the only symptom.

    Asserts the fetch is handed the RE-SIGNED url, not the stored one. Reverting
    the re-sign in ``_resolve_panel_images`` fails exactly this.
    """
    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    stored = "https://minio.example.com/w2v-media/frame.png?X-Amz-Signature=EXPIRED"
    created = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={
            "action_description": "an hour-old frame",
            "media_ref": {"type": "ai_image", "url": stored},
        },
    )
    assert created.status_code == 201, created.text

    fetched: list[str] = []

    async def _fake_fetch(url: str, **kwargs: object) -> tuple[bytes, str]:
        fetched.append(url)
        return _png(), "png"

    class _Resigner:
        async def resign_presigned_url(self, url: str) -> str:
            assert url == stored
            return "https://minio.example.com/w2v-media/frame.png?X-Amz-Signature=FRESH"

    async def _fake_storage() -> object:
        return _Resigner()

    monkeypatch.setattr(storyboard_routes, "fetch_image_bytes", _fake_fetch)
    monkeypatch.setattr(
        "src.services.storyboard.panel_presign._get_storage",
        _fake_storage,
    )

    resp = await client.get(f"/api/projects/{project.id}/storyboard/export.pdf")
    assert resp.status_code == 200, resp.text
    assert fetched == ["https://minio.example.com/w2v-media/frame.png?X-Amz-Signature=FRESH"], (
        f"the export fetched a stale signature: {fetched}"
    )
    assert b"/Subtype /Image" in resp.content


async def test_export_pdf_distinguishes_a_failed_frame_from_a_missing_one(
    client, db_session, monkeypatch
) -> None:
    """ "no frame yet" must not be said about a shot that HAS a frame.

    Both cases render a placeholder and both 200 — so the caption is the only
    thing telling a director whether the shot was never generated or whether we
    simply could not fetch it. Saying "no frame yet" for a rendered shot is a
    false statement about their own board.
    """
    from src.services.storyboard.pdf_export import (
        FRAME_UNAVAILABLE_CAPTION,
        PLACEHOLDER_CAPTION,
    )

    assert FRAME_UNAVAILABLE_CAPTION != PLACEHOLDER_CAPTION

    project = await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )
    created = await client.post(
        f"/api/projects/{project.id}/storyboard/panels",
        json={
            "action_description": "a frame we cannot reach",
            "media_ref": {"type": "ai_image", "url": "https://cdn.example.com/gone.png"},
        },
    )
    assert created.status_code == 201, created.text

    async def _boom(url: str, **kwargs: object) -> tuple[bytes, str]:
        raise RuntimeError("403 Request has expired")

    monkeypatch.setattr(storyboard_routes, "fetch_image_bytes", _boom)

    resp = await client.get(f"/api/projects/{project.id}/storyboard/export.pdf")
    assert resp.status_code == 200, resp.text
    assert resp.content.startswith(b"%PDF")

    # The route's own resolver must REPORT the failure, not swallow it silently.
    panels = await StoryboardPanelRepository(db_session).list_for_project(
        project.id, tenant_id=UUID(client.tenant_id)
    )
    images, unavailable = await storyboard_routes._resolve_panel_images(
        panels, tenant_id=UUID(client.tenant_id)
    )
    assert images == {}
    assert {p.id for p in panels} == unavailable, (
        "a frame that failed to fetch was not reported as unavailable, so the "
        "board would caption it 'no frame yet'"
    )
