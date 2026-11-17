"""The youtube-upload client forwards the real caller tenant as X-Owner-Tenant-Id.

The orchestrator authorizes youtube_upload_service with its service-account
license token, so the downstream-introspected tenant is the SAME for every
real tenant. The owner hook carries the real caller tenant so the service can
scope upload jobs to the actual owner instead of leaking them across tenants.
"""

from __future__ import annotations

import json

import httpx
import pytest
from w2v_common.auth.identity_context import clear_identity_context, set_identity_context
from w2v_common.tenant import InvalidTenantIdError

from src.clients.youtube_upload_client import (
    YouTubeUploadClient,
    _explicit_owner_tenant,
    _owner_tenant_event_hook,
)

pytestmark = pytest.mark.asyncio


#: w2v-common 0.67.0 parses the ambient tenant as a `uuid.UUID`; the former
#: "real-tenant-7" slug is now REJECTED at `set_identity_context`.
_REAL_TENANT = "77777777-7777-7777-7777-777777777777"
#: The nil uuid -- the pre-0.67 `auth.default_tenant_id` fallback.
_NIL_TENANT = "00000000-0000-0000-0000-000000000000"
#: Ambient vs explicit owner, both real uuids for the same reason.
_AMBIENT = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_EXPLICIT = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"


@pytest.mark.parametrize(
    "spelling",
    [_NIL_TENANT],
    ids=["hyphenated"],
)
async def test_the_ambient_lane_can_no_longer_carry_the_nil_tenant(spelling: str) -> None:
    """w2v-common 0.68.0 closed the ambient lane at the door.

    ``set_identity_context`` runs the value through ``parse_tenant_id``, which
    now refuses the nil uuid on both of its paths. So the hook can never be
    handed a nil tenant *via the ContextVar* — the earlier version of the test
    above bound it exactly that way, and that setup now raises.

    Pinned rather than merely dropped: this is the boundary that makes the
    ambient half of ``tenant_scope_key``'s nil check defence-in-depth, and if
    the library ever stops refusing, the coverage above must go back to
    exercising this lane too.
    """
    with pytest.raises(InvalidTenantIdError, match=r"(?i)nil uuid"):
        set_identity_context(tenant_id=spelling, license_id="lic", subject_id="sub")


async def test_client_registers_the_hook_on_its_http_client() -> None:
    client = YouTubeUploadClient(base_url="http://yt", license_token="sa-token")
    try:
        http = await client._get_client()
        assert _owner_tenant_event_hook in http.event_hooks["request"]
        # idempotent — a second resolution doesn't double-register
        http2 = await client._get_client()
        assert http2.event_hooks["request"].count(_owner_tenant_event_hook) == 1
    finally:
        await client.close()


def _client_returning(monkeypatch, status: int, body: dict | None = None):
    """A YouTubeUploadClient whose http client is a MockTransport returning `status`."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body if body is not None else {"detail": "x"})

    client = YouTubeUploadClient(base_url="http://yt", license_token="sa-token")
    mock_http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://yt")

    async def fake_get_client() -> httpx.AsyncClient:
        return mock_http

    monkeypatch.setattr(client, "_get_client", fake_get_client)
    return client, mock_http


async def test_get_upload_job_404_is_quiet(monkeypatch, caplog) -> None:
    """A pruned job (404) raises HTTPStatusError but logs NO ERROR (poller spam)."""
    client, http = _client_returning(monkeypatch, 404)
    try:
        with caplog.at_level("ERROR", logger="w2v_common.clients.base"):
            with pytest.raises(httpx.HTTPStatusError):
                await client.get_upload_job("gone")
        errors = [r for r in caplog.records if r.name == "w2v_common.clients.base"]
        assert errors == [], f"a 404 status-poll must not log ERROR, got: {errors}"
    finally:
        await http.aclose()


async def test_get_upload_job_200_returns_dict(monkeypatch) -> None:
    client, http = _client_returning(monkeypatch, 200, {"job_id": "j1", "status": "completed"})
    try:
        job = await client.get_upload_job("j1")
        assert job["status"] == "completed"
    finally:
        await http.aclose()


async def test_get_upload_job_502_still_logs_error(monkeypatch, caplog) -> None:
    """A genuine upstream fault (non-404) keeps the ERROR log + raises."""
    client, http = _client_returning(monkeypatch, 502)
    try:
        with caplog.at_level("ERROR", logger="w2v_common.clients.base"):
            with pytest.raises(httpx.HTTPStatusError):
                await client.get_upload_job("boom")
        errors = [r for r in caplog.records if r.name == "w2v_common.clients.base"]
        assert errors, "a non-404 fault must still log ERROR"
    finally:
        await http.aclose()


def _recording_client(
    monkeypatch,
) -> tuple[YouTubeUploadClient, httpx.AsyncClient, list[httpx.Request]]:
    """A client whose transport records every outgoing request (hooks still run)."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"job_id": "j", "status": "completed", "jobs": []})

    client = YouTubeUploadClient(base_url="http://yt", license_token="sa-token")
    mock_http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://yt",
        event_hooks={"request": [_owner_tenant_event_hook]},
    )

    async def fake_get_client() -> httpx.AsyncClient:
        return mock_http

    monkeypatch.setattr(client, "_get_client", fake_get_client)
    return client, mock_http, seen


@pytest.fixture
def _clean_identity():
    """No ambient identity bound before or after — these tests bind it explicitly."""
    clear_identity_context()
    try:
        yield
    finally:
        clear_identity_context()


async def test_explicit_owner_beats_ambient_tenant(monkeypatch, _clean_identity) -> None:
    """An explicitly-passed owner_tenant_id WINS over a different ambient tenant.

    This is the whole point of the fix: the route knows the caller tenant, and
    the ambient var may hold something else (or nothing) by the time the client
    runs. Explicit must win, or upload_jobs rows get an owner no caller matches.
    """
    set_identity_context(tenant_id=_AMBIENT, license_id="lic", subject_id="sub")
    client, http, seen = _recording_client(monkeypatch)
    try:
        await client.get_upload_job("j1", owner_tenant_id=_EXPLICIT)
    finally:
        await http.aclose()

    assert len(seen) == 1
    assert seen[0].headers["X-Owner-Tenant-Id"] == _EXPLICIT


async def test_omitted_owner_falls_back_to_ambient(monkeypatch, _clean_identity) -> None:
    """With no explicit arg the ambient tenant is still used — background callers unchanged."""
    set_identity_context(tenant_id=_AMBIENT, license_id="lic", subject_id="sub")
    client, http, seen = _recording_client(monkeypatch)
    try:
        await client.get_upload_job("j1")
    finally:
        await http.aclose()

    assert len(seen) == 1
    assert seen[0].headers["X-Owner-Tenant-Id"] == _AMBIENT


async def test_neither_explicit_nor_ambient_omits_the_header(monkeypatch, _clean_identity) -> None:
    """No ambient + no explicit => header ABSENT. The no-op is deliberate, not fail-closed."""
    client, http, seen = _recording_client(monkeypatch)
    try:
        await client.get_upload_job("j1")
    finally:
        await http.aclose()

    assert len(seen) == 1
    assert "X-Owner-Tenant-Id" not in seen[0].headers


async def test_explicit_owner_does_not_leak_into_the_next_call(
    monkeypatch, _clean_identity
) -> None:
    """The ContextVar must be RESET after the call — the real risk of this approach.

    If ``_explicit_owner`` leaked, a later background/unauth call would stamp the
    previous caller's tenant onto somebody else's job. Asserted twice: with no
    ambient fallback at all (header must vanish) and with an ambient tenant
    bound (the AMBIENT one must win again, not the stale explicit one).
    """
    client, http, seen = _recording_client(monkeypatch)
    try:
        await client.get_upload_job("j1", owner_tenant_id=_EXPLICIT)
        # 1) nothing bound at all afterwards -> no header
        await client.get_upload_job("j2")
        # 2) ambient bound afterwards -> ambient wins, not the stale explicit value
        set_identity_context(tenant_id=_AMBIENT, license_id="lic", subject_id="sub")
        await client.get_upload_job("j3")
    finally:
        await http.aclose()

    assert len(seen) == 3
    assert seen[0].headers["X-Owner-Tenant-Id"] == _EXPLICIT
    assert "X-Owner-Tenant-Id" not in seen[1].headers, (
        f"explicit owner leaked into the next call: {seen[1].headers.get('X-Owner-Tenant-Id')!r}"
    )
    assert seen[2].headers["X-Owner-Tenant-Id"] == _AMBIENT


async def test_explicit_owner_is_reset_when_the_call_raises(monkeypatch, _clean_identity) -> None:
    """A failing call must still reset the ContextVar (the finally path)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = YouTubeUploadClient(base_url="http://yt", license_token="sa-token")
    mock_http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://yt")

    async def fake_get_client() -> httpx.AsyncClient:
        return mock_http

    monkeypatch.setattr(client, "_get_client", fake_get_client)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_upload_job("boom", owner_tenant_id=_EXPLICIT)
    finally:
        await mock_http.aclose()

    assert _explicit_owner_tenant.get() is None


async def test_start_and_list_also_forward_the_explicit_owner(monkeypatch, _clean_identity) -> None:
    """start_upload and list_upload_jobs carry the explicit owner too, not just the getter."""
    set_identity_context(tenant_id=_AMBIENT, license_id="lic", subject_id="sub")
    client, http, seen = _recording_client(monkeypatch)
    try:
        await client.start_upload({"title": "t"}, owner_tenant_id=_EXPLICIT)
        await client.list_upload_jobs(limit=10, offset=0, owner_tenant_id=_EXPLICIT)
    finally:
        await http.aclose()

    assert len(seen) == 2
    assert [r.headers["X-Owner-Tenant-Id"] for r in seen] == [
        _EXPLICIT,
        _EXPLICIT,
    ]


async def test_retract_posts_to_the_job_with_the_explicit_owner(
    monkeypatch, _clean_identity
) -> None:
    """Retraction hits ``POST /api/upload/jobs/{id}/retract`` scoped to the explicit owner."""
    client, http, seen = _recording_client(monkeypatch)
    try:
        await client.retract_upload_job("j1", owner_tenant_id=_EXPLICIT)
    finally:
        await http.aclose()

    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert seen[0].url.path == "/api/upload/jobs/j1/retract"
    assert seen[0].headers["X-Owner-Tenant-Id"] == _EXPLICIT


async def test_set_privacy_posts_target_to_the_job_with_the_explicit_owner(
    monkeypatch, _clean_identity
) -> None:
    """The release hits ``POST /api/upload/jobs/{id}/privacy`` with the target, owner-scoped."""
    client, http, seen = _recording_client(monkeypatch)
    try:
        await client.set_upload_job_privacy("j1", "public", owner_tenant_id=_EXPLICIT)
    finally:
        await http.aclose()

    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert seen[0].url.path == "/api/upload/jobs/j1/privacy"
    assert json.loads(seen[0].content) == {"privacy_status": "public"}
    assert seen[0].headers["X-Owner-Tenant-Id"] == _EXPLICIT
