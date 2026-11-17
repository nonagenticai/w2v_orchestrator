"""HTTP edge of the service lab — validation, error mapping, persistence, media.

Two things in here are load-bearing and easy to lose in a refactor:

* **Durability is asserted against a session the route never touched.** The test
  app overrides ``get_db_session`` with a dependency that does NOT commit in its
  teardown, so a row is visible afterwards only because the ROUTE committed it.
  Running the same assertions through the stock dependency would pass on a route
  that never commits at all (see the repo CLAUDE.md).
* **Cross-tenant requests use a SECOND seeded app.** ``X-Active-Tenant-Id`` /
  ``?active_tenant=`` are deliberately ignored for a non-super_admin caller, so
  re-pointing a header from one test client would silently keep the original
  tenant and assert nothing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes import service_lab as service_lab_routes
from src.api.routes._common import get_db_session
from src.api.routes.service_lab import get_service_lab_limiter
from src.clients.gpu_busy import GpuBusyError
from src.clients.ocr_client import OcrError, OcrNotConfiguredError
from src.database.models import ServiceRun, Tenant
from src.services.providers.errors import SlotNotConfiguredError, SlotUnavailableError
from src.services.providers.rate_limit import KEY_PREFIX, ProviderTestRateLimiter
from src.services.service_lab.invoker import InvokeResult, ProviderInfo, ServiceInputError
from src.services.stages.upstream import PermanentUpstreamError
from src.services.tts_storage import set_storage_for_tests

_TENANT = uuid4()
_SUBJECT = uuid4()
_OTHER_TENANT = uuid4()

_PNG = b"\x89PNG\r\n\x1a\n-fake-image-bytes"
_WAV = b"RIFF....WAVE-fake-audio"

#: The limiter namespace the route asks for; read from the module under test so a
#: rename cannot leave this assertion pinning a stale string.
RATE_LIMIT_PREFIX = service_lab_routes.RATE_LIMIT_PREFIX


# ─── Fakes ───────────────────────────────────────────────────────────────


class _FakeStream:
    """The ``StreamedLike`` protocol ``minio_stream.open_object`` expects."""

    status = 200

    def __init__(self, data: bytes, content_type: str) -> None:
        self._data = data
        self.headers = {
            "content-type": content_type,
            "accept-ranges": "bytes",
            "etag": '"test-etag"',
        }

    async def iter_chunks(self, chunk_size: int) -> AsyncIterator[bytes]:
        for start in range(0, len(self._data), chunk_size):
            yield self._data[start : start + chunk_size]

    async def aclose(self) -> None:
        return None


class _FakeStorage:
    """Stands in for BOTH storage singletons (the writer AND the media reader)."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.puts: list[tuple[str, bytes, str]] = []
        self.fail_writes = False

    async def put_object(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        if self.fail_writes:
            raise OSError("storage unavailable")
        self.objects[key] = (data, content_type)
        self.puts.append((key, data, content_type))
        return key

    async def get_object_stream(
        self, bucket: str, object_key: str, request_headers: Any = None
    ) -> _FakeStream:
        if object_key not in self.objects:
            from miniopy_async.error import S3Error

            # The SDK types ``response`` as a live aiohttp response; the object is
            # never touched on the 404 path, so ``None`` is honest here.
            error = S3Error("NoSuchKey", "missing", object_key, "req", "host", cast("Any", None))
            # The 403/404 mapping in ``minio_stream.open_object`` reads this off the
            # instance; S3Error does not declare it.
            cast("Any", error).status_code = 404
            raise error
        data, content_type = self.objects[object_key]
        return _FakeStream(data, content_type)


async def _non_committing_session(request: Request) -> AsyncIterator[AsyncSession]:
    """``get_db_session`` WITHOUT the teardown commit — see the module docstring."""
    factory = request.app.state.session_factory
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


class _Api(AsyncClient):
    """Client that carries its app, so a test can override a dependency."""

    app: FastAPI


def _seed_auth(tenant_id: UUID):
    from starlette.middleware.base import BaseHTTPMiddleware

    class _Seed(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = str(tenant_id)
            request.state.subject_id = str(_SUBJECT)
            return await call_next(request)

    return _Seed


@asynccontextmanager
async def _api_as(db_session_factory, tenant_id: UUID) -> AsyncIterator[_Api]:
    """The router on a minimal app acting as ``tenant_id``."""
    from fakeredis.aioredis import FakeRedis

    app = FastAPI()
    app.state.session_factory = db_session_factory
    app.state.redis = FakeRedis()
    app.include_router(service_lab_routes.router)
    app.dependency_overrides[get_db_session] = _non_committing_session
    app.add_middleware(_seed_auth(tenant_id))
    client = _Api(transport=ASGITransport(app=app), base_url="http://testserver")
    client.app = app
    async with client:
        yield client


@pytest.fixture
def storage() -> Iterator[_FakeStorage]:
    fake = _FakeStorage()
    # The writer (this route) and the reader (minio_stream) share the process-wide
    # storage singleton, so one injection covers both directions. The fake is
    # deliberately narrower than the ``_StorageLike`` protocol: the routes only
    # reach ``put_object`` / ``get_object_stream``.
    set_storage_for_tests(cast("Any", fake))
    try:
        yield fake
    finally:
        set_storage_for_tests(None)


@pytest.fixture(autouse=True)
async def _tenants(db_session_factory) -> None:
    """``service_runs.tenant_id`` is a real FK — the tenants must exist."""
    async with db_session_factory() as session:
        for tenant_id in (_TENANT, _OTHER_TENANT):
            session.add(Tenant(id=tenant_id, name=f"service-lab-{tenant_id}", status="active"))
        await session.commit()


@pytest.fixture
async def api(db_session_factory, storage: _FakeStorage) -> AsyncIterator[_Api]:
    async with _api_as(db_session_factory, _TENANT) as client:
        yield client


# ─── Helpers ─────────────────────────────────────────────────────────────


def _result(
    *,
    output_kind: str = "image",
    text: str | None = None,
    media: bytes | None = None,
    content_type: str | None = None,
) -> InvokeResult:
    return InvokeResult(
        output_kind=output_kind,
        provider=ProviderInfo(source="deployment", kind="nonagentic", model="test-model"),
        text=text,
        media_bytes=media,
        media_content_type=content_type,
    )


def _stub_invoke(monkeypatch: pytest.MonkeyPatch, behaviour: Any) -> list[dict[str, Any]]:
    """Patch the route's ``invoke``; ``behaviour`` returns an InvokeResult or raises it."""
    calls: list[dict[str, Any]] = []

    async def fake_invoke(
        session: AsyncSession,
        *,
        tenant_id: UUID,
        request: Any,
        timeout_seconds: float = 180.0,
    ) -> InvokeResult:
        calls.append(
            {"tenant_id": tenant_id, "request": request, "timeout_seconds": timeout_seconds}
        )
        outcome = behaviour(request)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(service_lab_routes, "invoke", fake_invoke)
    return calls


async def _rows(db_session_factory, tenant_id: UUID = _TENANT) -> list[ServiceRun]:
    """Read the persisted rows through a session the route never touched."""
    async with db_session_factory() as session:
        result = await session.execute(
            select(ServiceRun)
            .where(ServiceRun.tenant_id == tenant_id)
            .order_by(ServiceRun.created_at.asc())
        )
        return list(result.scalars().all())


# ─── Validation ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "body"),
    [("/api/services/image", {})],
)
async def test_invalid_bodies_are_422(api: AsyncClient, path: str, body: dict[str, Any]) -> None:
    response = await api.post(path, json=body)
    assert response.status_code == 422, response.text


async def test_history_rejects_an_unknown_slot(api: AsyncClient) -> None:
    assert (await api.get("/api/services/nope/history")).status_code == 422


async def test_history_limit_is_bounded(api: AsyncClient) -> None:
    assert (await api.get("/api/services/image/history?limit=0")).status_code == 422
    assert (await api.get("/api/services/image/history?limit=51")).status_code == 422


async def test_asr_requires_a_file(api: AsyncClient) -> None:
    assert (await api.post("/api/services/asr", data={"language": "en"})).status_code == 422


async def test_vlm_needs_exactly_one_image_source(api: AsyncClient) -> None:
    neither = await api.post("/api/services/vlm", data={"prompt": "what is this?"})
    assert neither.status_code == 422
    both = await api.post(
        "/api/services/vlm",
        data={"prompt": "what is this?", "image_url": "https://example.test/a.png"},
        files={"file": ("a.png", _PNG, "image/png")},
    )
    assert both.status_code == 422


async def test_upload_over_the_cap_is_413(
    api: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_lab_routes, "MAX_UPLOAD_BYTES", 8)
    response = await api.post("/api/services/asr", files={"file": ("a.wav", _WAV * 4, "audio/wav")})
    assert response.status_code == 413, response.text


# ─── Error mapping ───────────────────────────────────────────────────────

_ERROR_CASES = [
    (
        "SlotUnavailableError",
        lambda: SlotUnavailableError("tts", "secret undecryptable"),
        409,
        "provider_unavailable",
    ),
    (
        "OcrNotConfiguredError",
        lambda: OcrNotConfiguredError("no ocr"),
        409,
        "ocr_not_configured",
    ),
    (
        "GpuBusyError",
        lambda: GpuBusyError("busy", service="tts", status_code=503, retry_after=12),
        503,
        "provider_busy",
    ),
    ("TimeoutError", TimeoutError, 504, "upstream_timeout"),
    (
        "PermanentUpstreamError",
        lambda: PermanentUpstreamError("upstream said: key sk-live-SECRET"),
        502,
        "upstream_failed",
    ),
]


@pytest.mark.parametrize(
    ("name", "factory", "status", "code"),
    [
        (
            "SlotUnavailableError",
            lambda: SlotUnavailableError("tts", "secret undecryptable"),
            409,
            "provider_unavailable",
        ),
        (
            "ServiceInputError",
            lambda: ServiceInputError("asr needs an audio upload, got 'image/png'"),
            422,
            "invalid_service_input",
        ),
    ],
)
async def test_service_errors_map_to_coded_http(
    api: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    factory: Any,
    status: int,
    code: str,
) -> None:
    _stub_invoke(monkeypatch, lambda _request: factory())
    response = await api.post("/api/services/llm", json={"prompt": "hello"})
    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code
    # A 502 must never echo the upstream's own text (it can carry a secret).
    if status == 502:
        assert "SECRET" not in response.text


async def test_a_wrapped_timeout_is_504_not_502(
    api: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``invoke`` re-raises its asyncio.timeout expiry as a transient upstream error."""

    def behaviour(_request: Any) -> BaseException:
        try:
            raise TimeoutError("deadline")
        except TimeoutError as cause:
            try:
                raise PermanentUpstreamError("did not answer in time") from cause
            except PermanentUpstreamError as wrapped:
                return wrapped

    _stub_invoke(monkeypatch, behaviour)
    response = await api.post("/api/services/tts", json={"text": "hi"})
    assert response.status_code == 504, response.text
    assert response.json()["detail"]["code"] == "upstream_timeout"


async def test_byo_required_is_the_402_byo_errors_body(
    api: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_invoke(monkeypatch, lambda _request: SlotNotConfiguredError("image"))
    response = await api.post("/api/services/image", json={"prompt": "a cat"})
    assert response.status_code == 402, response.text
    detail = response.json()["detail"]
    # The SAME object src/api/byo_errors.py puts on the wire, so the frontend has
    # one branch for a run-start refusal and a service-lab refusal.
    assert detail["code"] == "byo_provider_required"
    assert detail["modality"] == "image"
    assert "image" in detail["slots"]
    assert detail["message"]


async def test_video_with_no_registered_renderer_is_409_provider_unavailable(
    api: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ACG registers no video provider: that is an unavailable provider, not BYO."""
    _stub_invoke(
        monkeypatch,
        lambda _request: SlotUnavailableError("video_gen", "no renderer is registered"),
    )
    response = await api.post("/api/services/video", json={"prompt": "a wave"})
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "provider_unavailable"
    assert detail["message"]


async def test_video_under_byo_required_still_answers_the_byo_body(
    api: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A byo_required tenant with no video connection must hear BYO, not "no renderer":
    the deployment is fine, the tenant simply has not registered a provider."""
    _stub_invoke(monkeypatch, lambda _request: SlotNotConfiguredError("video_gen"))
    response = await api.post("/api/services/video", json={"prompt": "a wave"})
    assert response.status_code == 402, response.text
    assert response.json()["detail"]["code"] == "byo_provider_required"


async def test_an_ocr_provider_fault_is_502_not_an_unmapped_500(
    api: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OcrError is what the provider raises after its own retries. It used to escape
    the mapper and surface as a bare 500 with no history row."""
    _stub_invoke(monkeypatch, lambda _request: OcrError("ocr provider said 500"))
    response = await api.post(
        "/api/services/ocr",
        files={"file": ("page.png", _PNG, "image/png")},
    )
    assert response.status_code == 502, response.text
    assert response.json()["detail"]["code"] == "upstream_failed"


async def test_busy_answers_carry_retry_after(
    api: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_invoke(
        monkeypatch,
        lambda _request: GpuBusyError("busy", service="tts", status_code=503, retry_after=17),
    )
    response = await api.post("/api/services/tts", json={"text": "hi"})
    assert response.status_code == 503
    assert response.headers["retry-after"] == "17"


async def test_rate_limit_exceeded_is_429(api: _Api, monkeypatch: pytest.MonkeyPatch) -> None:
    from fakeredis.aioredis import FakeRedis

    _stub_invoke(monkeypatch, lambda _request: _result(output_kind="text", text="ok"))
    limiter = ProviderTestRateLimiter(
        FakeRedis(), limit=1, window_seconds=3600, key_prefix=RATE_LIMIT_PREFIX
    )
    api.app.dependency_overrides[get_service_lab_limiter] = lambda: limiter

    first = await api.post("/api/services/llm", json={"prompt": "one"})
    second = await api.post("/api/services/llm", json={"prompt": "two"})
    assert first.status_code == 200, first.text
    assert second.status_code == 429, second.text
    assert second.json()["detail"]["code"] == "SERVICE_LAB_RATE_LIMITED"
    assert int(second.headers["retry-after"]) >= 1


def test_the_service_lab_limiter_has_its_own_redis_namespace() -> None:
    """The budget must not drain — or be drained by — the live-provider-test budget."""
    live_test = ProviderTestRateLimiter(None, limit=1, window_seconds=60)
    assert live_test.key(_TENANT, 0).startswith(f"{KEY_PREFIX}:")
    namespaced = ProviderTestRateLimiter(
        None, limit=1, window_seconds=60, key_prefix=RATE_LIMIT_PREFIX
    )
    assert namespaced.key(_TENANT, 0).startswith("service-lab-rl:")
    assert namespaced.key(_TENANT, 0) != live_test.key(_TENANT, 0)


# ─── Success paths ───────────────────────────────────────────────────────


async def test_image_run_persists_stores_and_returns_media_url(
    api: AsyncClient, db_session_factory, storage: _FakeStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_invoke(
        monkeypatch,
        lambda _request: _result(output_kind="image", media=_PNG, content_type="image/png"),
    )
    response = await api.post(
        "/api/services/image",
        json={"prompt": "a lighthouse at dusk", "width": 512, "height": 640, "seed": 7},
    )
    assert response.status_code == 200, response.text
    view = response.json()
    assert view["slot"] == "image"
    assert view["status"] == "succeeded"
    assert view["output_kind"] == "image"
    assert view["media_content_type"] == "image/png"
    assert view["media_url"] == f"/api/services/image/history/{view['id']}/media"
    assert view["provider"] == {
        "source": "deployment",
        "kind": "nonagentic",
        "model": "test-model",
    }

    # The bytes went to storage under the documented media-cache layout.
    run_id = view["id"]
    expected_key = f"media-cache/tenants/{_TENANT}/service-lab/run/{run_id[:2]}/{run_id}.png"
    assert storage.objects[expected_key] == (_PNG, "image/png")

    # The row is durable WITHOUT the dependency's teardown commit, and the echo
    # carries the caller's parameters but never the upload itself.
    rows = await _rows(db_session_factory)
    assert len(rows) == 1
    row = rows[0]
    assert str(row.id) == run_id
    assert row.media_object_key == expected_key
    assert row.request["prompt"] == "a lighthouse at dusk"
    assert row.request["width"] == 512
    assert "image_bytes" not in row.request


async def test_llm_run_persists_text_and_no_media(
    api: AsyncClient, db_session_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_invoke(monkeypatch, lambda _request: _result(output_kind="text", text="four candles"))
    response = await api.post("/api/services/llm", json={"prompt": "count", "system": "be terse"})
    assert response.status_code == 200, response.text
    view = response.json()
    assert view["text"] == "four candles"
    assert view["media_url"] is None
    rows = await _rows(db_session_factory)
    assert [row.text_output for row in rows] == ["four candles"]
    assert rows[0].request["system"] == "be terse"


async def test_video_gets_the_long_deadline(
    api: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_invoke(monkeypatch, lambda _request: _result(output_kind="video", media=b"mp4"))
    response = await api.post("/api/services/video", json={"prompt": "a wave", "seconds": 5})
    assert response.status_code == 200, response.text
    assert calls[0]["timeout_seconds"] == 300.0


async def test_multipart_runs_pass_the_uploaded_bytes(
    api: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_invoke(
        monkeypatch, lambda _request: _result(output_kind="transcript", text="transcript")
    )
    response = await api.post(
        "/api/services/asr",
        files={"file": ("clip.wav", _WAV, "audio/wav")},
        data={"language": "nl"},
    )
    assert response.status_code == 200, response.text
    request = calls[0]["request"]
    assert request.image_bytes == _WAV
    assert request.image_content_type == "audio/wav"
    assert request.language == "nl"


async def test_enhance_returns_the_rewritten_prompt(
    api: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_enhance(session, *, tenant_id, prompt, target, timeout_seconds=60.0):
        assert (prompt, target, timeout_seconds) == ("a cat", "image", 60.0)
        return "a photorealistic cat, golden hour"

    monkeypatch.setattr(service_lab_routes, "enhance_prompt", fake_enhance)
    response = await api.post("/api/services/enhance", json={"prompt": "a cat", "target": "image"})
    assert response.status_code == 200, response.text
    assert response.json() == {"prompt": "a photorealistic cat, golden hour"}


async def test_a_failed_run_is_recorded_in_the_history(
    api: AsyncClient, db_session_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_invoke(monkeypatch, lambda _request: SlotUnavailableError("llm", "pinned version gone"))
    response = await api.post("/api/services/llm", json={"prompt": "hello"})
    assert response.status_code == 409

    rows = await _rows(db_session_factory)
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert rows[0].error is not None
    assert rows[0].error["code"] == "provider_unavailable"
    assert rows[0].media_object_key is None


async def test_a_storage_failure_is_a_502_and_still_records_the_run(
    api: AsyncClient,
    db_session_factory,
    storage: _FakeStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_invoke(
        monkeypatch,
        lambda _request: _result(output_kind="image", media=_PNG, content_type="image/png"),
    )
    storage.fail_writes = True
    response = await api.post("/api/services/image", json={"prompt": "x"})
    assert response.status_code == 502, response.text
    assert response.json()["detail"]["code"] == "upstream_failed"

    rows = await _rows(db_session_factory)
    assert [row.status for row in rows] == ["failed"]
    assert rows[0].error is not None
    assert rows[0].error["code"] == "upstream_failed"


# ─── History ─────────────────────────────────────────────────────────────


async def test_history_is_newest_first_and_tenant_scoped(
    api: AsyncClient, db_session_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = iter(["first", "second", "third"])

    def behaviour(_request: Any) -> InvokeResult:
        return _result(output_kind="text", text=next(prompts))

    _stub_invoke(monkeypatch, behaviour)
    for _ in range(3):
        assert (await api.post("/api/services/llm", json={"prompt": "x"})).status_code == 200

    response = await api.get("/api/services/llm/history")
    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert [item["text"] for item in items] == ["third", "second", "first"]

    # Another tenant sees nothing, even for the same slot.
    async with _api_as(db_session_factory, _OTHER_TENANT) as other:
        assert (await other.get("/api/services/llm/history")).json()["items"] == []


async def test_another_tenants_run_is_404(
    api: AsyncClient, db_session_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_invoke(monkeypatch, lambda _request: _result(output_kind="text", text="mine"))
    created = await api.post("/api/services/llm", json={"prompt": "x"})
    run_id = created.json()["id"]

    assert (await api.get(f"/api/services/llm/history/{run_id}")).status_code == 200

    async with _api_as(db_session_factory, _OTHER_TENANT) as other:
        assert (await other.get(f"/api/services/llm/history/{run_id}")).status_code == 404
    # The slot is part of the identity: the same id is not an image run.
    assert (await api.get(f"/api/services/image/history/{run_id}")).status_code == 404


async def test_history_carries_the_media_url_only_when_media_exists(
    api: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_invoke(
        monkeypatch,
        lambda _request: _result(output_kind="image", media=_PNG, content_type="image/png"),
    )
    await api.post("/api/services/image", json={"prompt": "x"})
    items = (await api.get("/api/services/image/history")).json()["items"]
    assert items[0]["media_url"] == f"/api/services/image/history/{items[0]['id']}/media"
    assert items[0]["text"] is None
    assert items[0]["error"] is None


# ─── Media ───────────────────────────────────────────────────────────────


async def _image_run(
    api: AsyncClient, monkeypatch: pytest.MonkeyPatch, storage: _FakeStorage
) -> dict[str, Any]:
    _stub_invoke(
        monkeypatch,
        lambda _request: _result(output_kind="image", media=_PNG, content_type="image/png"),
    )
    response = await api.post("/api/services/image", json={"prompt": "x"})
    assert response.status_code == 200, response.text
    return response.json()


async def test_media_streams_the_owners_bytes(
    api: AsyncClient, storage: _FakeStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = await _image_run(api, monkeypatch, storage)
    response = await api.get(view["media_url"])
    assert response.status_code == 200, response.text
    assert response.content == _PNG
    assert response.headers["content-type"] == "image/png"
    assert response.headers["etag"] == '"test-etag"'


async def test_media_is_404_for_another_tenant(
    api: AsyncClient,
    db_session_factory,
    storage: _FakeStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = await _image_run(api, monkeypatch, storage)
    async with _api_as(db_session_factory, _OTHER_TENANT) as other:
        stolen = await other.get(view["media_url"])
    assert stolen.status_code == 404
    assert stolen.content != _PNG


async def test_media_is_404_when_the_object_is_gone(
    api: AsyncClient, storage: _FakeStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = await _image_run(api, monkeypatch, storage)
    storage.objects.clear()  # the row survived, the object did not
    response = await api.get(view["media_url"])
    assert response.status_code == 404, response.text
