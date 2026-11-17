"""The panel image proxy — the route that makes a generated panel visible at all.

A panel's ``media_ref.url`` is a presigned ``minio.test`` URL. That host is
Tailnet split-DNS: it resolves inside the cluster and nowhere else, so an
``<img src>`` pointed at it never loads in the user's browser. Run artifacts have
been rewritten to an orchestrator-proxied path for exactly this reason
(``api/artifact_url_rewriter``); storyboard panels were not, so the feature was
invisible to real users however well the generation worked.

Panels carry no ``run_id``, so the run-scoped ``/api/runs/{id}/artifact/...``
proxy cannot serve them. These cover the panel-scoped replacement:

* the read path emits a RELATIVE proxy path and never a ``minio.test`` host;
* the proxy streams the panel's own object, for the tenant that owns it, with the
  content type the object KEY implies rather than the one MinIO has stored — every
  artifact written before ``storage_service.guess_artifact_content_type`` existed is
  labelled ``application/json`` in object metadata, PNGs included, and those objects
  are not being re-written;
* **the key is confined to the panel's own project/tenant prefix.** ``media_ref``
  is client-writable (it is a field on both the panel create and update schemas),
  and MinIO is read with the ORCHESTRATOR's credentials, so authorising the panel
  alone would make this route a confused deputy: point your own panel at a
  stranger's key and the proxy hands you their bytes. The cross-project test below
  is that regression — it must go red if the prefix check is removed;
* a cross-tenant panel is a 404 (not a 403 — matching every sibling panel route);
* a panel with no image is a 404;
* the PDF export still receives a FETCHABLE ABSOLUTE url, because it fetches
  server-side where ``minio.test`` *is* reachable — regressing it to the proxy
  path would silently turn every exported frame into a "frame unavailable" card;
* an unusable storage backend is a loud 503, not a 200 with an empty body (which
  renders as a broken image and looks like a generation bug).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from src.api.routes import storyboard as storyboard_routes
from src.database.models import Project
from src.services.tts_storage import set_storage_for_tests

pytestmark = pytest.mark.asyncio

_BUCKET = "w2v-orchestrator"
_HOST = f"https://minio.test/{_BUCKET}/"
_PIXELS = b"\x89PNG\r\n\x1a\n-fake-image-bytes"


def _owned_key(project_id: UUID, name: str = "frame-1") -> str:
    """The key a real panel generation writes for this project.

    ``StoryboardPanelImageService._namespace`` (``storyboard/{project_id}``) +
    ``STAGE_NAME`` + the content-addressed media id, as
    ``storage_service.store_stage_artifact`` lays it out.
    """
    return f"artifacts/storyboard/{project_id}/storyboard_panel_image/{name}.png"


def _stored_url(project_id: UUID, name: str = "frame-1") -> str:
    """What the row actually holds: that key, presigned, on the in-cluster host."""
    return f"{_HOST}{_owned_key(project_id, name)}?X-Amz-Signature=expired"


def _proxy_path(panel_id: str) -> str:
    return f"/api/storyboard/panels/{panel_id}/image/content"


class _FakeStream:
    """The three members ``minio_stream.StreamedLike`` reads, plus a close flag."""

    def __init__(self, payload: bytes, *, content_type: str) -> None:
        self.status = 200
        self.headers = {"content-type": content_type, "content-length": str(len(payload))}
        self._payload = payload
        self.closed = False

    async def iter_chunks(self, chunk_size: int = 8192) -> AsyncIterator[bytes]:
        for start in range(0, len(self._payload), chunk_size):
            yield self._payload[start : start + chunk_size]

    async def aclose(self) -> None:
        self.closed = True


class _StreamingStorage:
    """Storage that can stream one object and declines to re-sign anything.

    Declining the re-sign (``None``) is the honest fake for the proxy tests: it
    proves the proxy path is emitted by the REWRITE, not accidentally by a
    re-signer that happened to return something.
    """

    def __init__(self, *, content_type: str = "image/png") -> None:
        self.opened: list[tuple[str, str]] = []
        self.streams: list[_FakeStream] = []
        self._content_type = content_type

    async def resign_presigned_url(self, url: str) -> str | None:
        return None

    async def get_object_stream(
        self,
        bucket: str,
        object_key: str,
        request_headers: dict[str, str] | None = None,
    ) -> _FakeStream:
        self.opened.append((bucket, object_key))
        stream = _FakeStream(_PIXELS, content_type=self._content_type)
        self.streams.append(stream)
        return stream


class _NoStreamStorage:
    """A backend with no streaming support at all — the uninitialised-storage case."""

    async def resign_presigned_url(self, url: str) -> str | None:
        return None


@pytest.fixture
def streaming_storage():
    storage = _StreamingStorage()
    set_storage_for_tests(cast("Any", storage))
    try:
        yield storage
    finally:
        set_storage_for_tests(None)


async def _seed_project(db_session, *, tenant_id: UUID, subject_id: UUID) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="panel-image-proxy",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def _seed_panel(client, project_id: UUID, *, url: str | None = None) -> dict[str, Any]:
    """Create a panel; by default one holding this project's own generated frame."""
    body: dict[str, Any] = {"action_description": "Mara steps into the light"}
    if url is not None:
        body["media_ref"] = {"type": "ai_image", "url": url}
    resp = await client.post(f"/api/projects/{project_id}/storyboard/panels", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _read_panel(client, project_id: UUID, panel_id: str) -> dict[str, Any]:
    """Re-read the panel off the list route (there is no by-id panel GET)."""
    resp = await client.get(f"/api/projects/{project_id}/storyboard/panels")
    assert resp.status_code == 200, resp.text
    rows = [p for p in resp.json()["items"] if p["id"] == panel_id]
    assert rows, f"panel {panel_id} vanished"
    return rows[0]


async def test_panel_read_emits_a_relative_proxy_path(
    client, db_session, streaming_storage
) -> None:
    """The one property the whole change exists for: nothing leaving the public
    API may carry a host the user's browser cannot resolve."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id, url=_stored_url(project.id))

    url = (await _read_panel(client, project.id, panel["id"]))["media_ref"]["url"]

    assert url == _proxy_path(panel["id"])
    assert url.startswith("/"), "an absolute url pins the SPA to one origin"
    assert "minio.test" not in url


async def test_the_proxy_streams_the_panels_own_object(
    client, db_session, streaming_storage
) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id, url=_stored_url(project.id))

    resp = await client.get(_proxy_path(panel["id"]))

    assert resp.status_code == 200, resp.text
    assert resp.content == _PIXELS
    assert resp.headers["content-type"].startswith("image/png")
    # Derived from the panel row — bucket and key both come off media_ref, and
    # the expired signature on the stored url is discarded (we read with our own
    # credentials, which have no TTL).
    assert streaming_storage.opened == [(_BUCKET, _owned_key(project.id))]
    assert streaming_storage.streams[0].closed, "the upstream connection was never released"


class _JsonTypedStream(_FakeStream):
    """A stored object whose metadata says ``application/json`` — with the header
    name cased the way a real server sends it.

    aiohttp yields ``istr`` keys carrying the server's casing (``Content-Type``),
    and Starlette lowercases raw header names only after the response is built. A
    case-sensitive strip of the forwarded type would therefore leave it in place and
    emit two ``Content-Type`` headers, MinIO's wrong one first — so the casing here
    is load-bearing, not decoration.
    """

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload, content_type="application/json")
        self.headers = {"Content-Type": "application/json", "Content-Length": str(len(payload))}


class _HistoricalJsonStorage(_StreamingStorage):
    """Storage holding a PNG that was written back when every stage artifact was
    stamped ``application/json`` — i.e. every panel image generated to date."""

    async def get_object_stream(
        self,
        bucket: str,
        object_key: str,
        request_headers: dict[str, str] | None = None,
    ) -> _FakeStream:
        self.opened.append((bucket, object_key))
        stream = _JsonTypedStream(_PIXELS)
        self.streams.append(stream)
        return stream


async def test_a_png_stored_as_json_is_still_served_as_an_image(client, db_session) -> None:
    """THE demo-visible regression, and the one a re-write cannot fix.

    ``store_stage_artifact`` hardcoded ``application/json`` for every artifact it
    ever wrote, so every existing panel PNG carries that type in its MinIO object
    metadata. Those objects are not being re-written, so forwarding stored metadata
    keeps serving a 1.8 MB PNG as JSON — browsers sniff past it, a download, a CDN
    or an ``Accept``-negotiating client does not.

    The proxy takes the type from the object KEY (``…/frame-1.png``), which is the
    format the writer chose. Delete the ``media_type=`` argument in
    ``get_panel_image_content`` and this goes red.
    """
    storage = _HistoricalJsonStorage()
    set_storage_for_tests(cast("Any", storage))
    try:
        project = await _seed_project(
            db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
        )
        panel = await _seed_panel(client, project.id, url=_stored_url(project.id))

        resp = await client.get(_proxy_path(panel["id"]))

        assert resp.status_code == 200, resp.text
        assert resp.content == _PIXELS
        assert resp.headers["content-type"].startswith("image/"), (
            f"a PNG was served as {resp.headers['content-type']!r}"
        )
        # Exactly one Content-Type reaches the wire — a surviving forwarded copy
        # would win over ours and the assertion above would still pass on the
        # merged value in some clients.
        assert (
            len([1 for name, _ in resp.headers.raw if name.decode().lower() == "content-type"]) == 1
        )
    finally:
        set_storage_for_tests(None)


async def test_the_cross_run_media_cache_key_is_also_served(
    client, db_session, streaming_storage
) -> None:
    """With ``storage.media_cache_cross_run`` on, ``_store_media_asset`` writes the
    frame under the TENANT-scoped ``media-cache/tenants/{tenant}/`` prefix instead
    of the per-project one. Confining to the project prefix alone would 404 every
    real thumbnail the day that flag is flipped — silently, since a broken <img>
    looks like a generation bug."""
    tenant_id = UUID(client.tenant_id)
    project = await _seed_project(
        db_session, tenant_id=tenant_id, subject_id=UUID(client.subject_id)
    )
    key = f"media-cache/tenants/{tenant_id}/ab/abcdef0123.png"
    panel = await _seed_panel(client, project.id, url=f"{_HOST}{key}")

    assert (await _read_panel(client, project.id, panel["id"]))["media_ref"]["url"] == _proxy_path(
        panel["id"]
    )
    assert (await client.get(_proxy_path(panel["id"]))).status_code == 200
    assert streaming_storage.opened == [(_BUCKET, key)]


async def test_a_ref_naming_another_projects_object_is_refused(
    client, db_session, streaming_storage
) -> None:
    """THE confused-deputy regression.

    ``media_ref`` is client-writable, so a caller can point its OWN panel at a
    well-formed key under someone else's prefix. The proxy reads MinIO with the
    orchestrator's credentials, which can read every tenant's objects — so if the
    key is not confined to the panel's own project/tenant, authorising the panel
    hands the caller a stranger's bytes.

    Delete the prefix condition in ``panel_image_url.panel_object_ref`` and exactly
    this test goes red: the read path starts advertising a proxy path and the GET
    starts returning 200 with the victim's object.

    This test used to assert ``served == poisoned`` — that a refused ref comes back
    UNCHANGED. That was the leak, encoded as the expectation. Unchanged meant the
    read path emitted the foreign key, and (because ``refresh_panel_media_url`` ran
    FIRST and re-signs on bucket ownership alone) emitted it carrying a fresh,
    valid, 1-hour signature. The proxy was refusing to serve the object while the
    JSON handed over a working pointer to it. ``public_media_ref`` now withholds the
    url entirely; the assertion below is the strictly stronger one.
    """
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    victim_project_id = uuid4()
    poisoned = f"{_HOST}{_owned_key(victim_project_id, 'victim-frame')}"
    panel = await _seed_panel(client, project.id, url=poisoned)

    # The API must not advertise a working proxy path for someone else's object,
    # and must not hand back the foreign url either — neither the signature (which
    # grants READ on the victim's object) nor the bare key (which discloses it).
    served = (await _read_panel(client, project.id, panel["id"]))["media_ref"]["url"]
    assert served != _proxy_path(panel["id"]), "a foreign ref was advertised as proxyable"
    assert served is None, "a foreign ref's url was emitted instead of withheld"
    assert str(victim_project_id) not in str(served)

    # … and the proxy must refuse to fetch it, without ever reaching storage.
    assert (await client.get(_proxy_path(panel["id"]))).status_code == 404
    assert streaming_storage.opened == [], "the orchestrator read a foreign object"


async def test_a_ref_that_traverses_out_of_the_prefix_is_refused(
    client, db_session, streaming_storage
) -> None:
    """``..`` segments are collapsed BEFORE the containment test, and containment is
    segment-wise — so neither a traversal nor a sibling prefix that merely shares a
    string prefix can escape."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    escapes = [
        f"artifacts/storyboard/{project.id}/../{uuid4()}/storyboard_panel_image/x.png",
        f"artifacts/storyboard/{project.id}evil/storyboard_panel_image/x.png",
        f"artifacts/storyboard/{project.id}%2F..%2F{uuid4()}/x.png",
    ]
    for key in escapes:
        panel = await _seed_panel(client, project.id, url=f"{_HOST}{key}")
        assert (await client.get(_proxy_path(panel["id"]))).status_code == 404, key

    assert streaming_storage.opened == [], "a traversing key reached storage"


async def test_a_cross_tenant_panel_is_404(
    client, other_tenant_client, db_session, streaming_storage
) -> None:
    """404, not 403: the panel repository is tenant-scoped, so a foreign panel
    does not resolve at all — and the proxy must never open the object first."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id, url=_stored_url(project.id))

    resp = await other_tenant_client.get(_proxy_path(panel["id"]))

    assert resp.status_code == 404, resp.text
    assert streaming_storage.opened == [], "a foreign tenant reached storage"


async def test_a_panel_with_no_image_is_404(client, db_session, streaming_storage) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)

    resp = await client.get(_proxy_path(panel["id"]))

    assert resp.status_code == 404, resp.text
    assert streaming_storage.opened == []


async def test_an_external_image_is_not_rewritten_or_proxied(
    client, db_session, streaming_storage
) -> None:
    """An operator-pasted external image is already browser-reachable. Rewriting
    it to a proxy path would point the SPA at an object our MinIO does not have."""
    external = "https://cdn.example.com/art/reference.png"
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id, url=external)

    assert (await _read_panel(client, project.id, panel["id"]))["media_ref"]["url"] == external
    assert (await client.get(_proxy_path(panel["id"]))).status_code == 404
    assert streaming_storage.opened == []


async def test_pdf_export_still_fetches_an_absolute_in_cluster_url(
    client, db_session, monkeypatch, streaming_storage
) -> None:
    """The PDF export fetches frame bytes SERVER-side, where ``minio.test`` is
    reachable, so it must keep receiving an absolute url.

    It resolves frames through ``_resolve_panel_images`` — a different path from
    ``_panel_resp`` — and that separation is the thing under test: hand it the
    proxy path and ``fetch_image_bytes`` would refuse a relative string, every
    frame would degrade to a "frame unavailable" card, and the export would still
    200 so nothing would flag it.
    """
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    await _seed_panel(client, project.id, url=_stored_url(project.id))

    fetched: list[str] = []

    async def _fake_fetch(url: str, **kwargs: object) -> tuple[bytes, str]:
        fetched.append(url)
        return _PIXELS, "png"

    monkeypatch.setattr(storyboard_routes, "fetch_image_bytes", _fake_fetch)

    resp = await client.get(f"/api/projects/{project.id}/storyboard/export.pdf")

    assert resp.status_code == 200, resp.text
    assert fetched, "the export never attempted a frame fetch"
    assert fetched[0].startswith("https://minio.test/"), (
        f"the export was handed a url it cannot fetch server-side: {fetched[0]}"
    )
    assert "/api/storyboard/panels/" not in fetched[0], (
        "the export was handed the browser proxy path — every frame would degrade to a placeholder"
    )


async def test_unusable_storage_is_a_503_not_an_empty_200(client, db_session) -> None:
    """A silent empty body renders as a broken image and reads as a generation
    bug; the failure has to name itself."""
    set_storage_for_tests(cast("Any", _NoStreamStorage()))
    try:
        project = await _seed_project(
            db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
        )
        panel = await _seed_panel(client, project.id, url=_stored_url(project.id))

        resp = await client.get(_proxy_path(panel["id"]))

        assert resp.status_code == 503, resp.text
        assert resp.content, "a 503 with no body tells the operator nothing"
    finally:
        set_storage_for_tests(None)
