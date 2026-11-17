"""Regression: PR-2 downstream service clients carry auth.

These clients used to be bare ``httpx.AsyncClient`` instances with no
``Authorization`` / ``X-Tenant-Id`` headers, so every outbound call was
anonymous and license-authz returned 401. These tests pin the contract:

- The license-token from ``Settings.downstream.license_token`` is baked in
- Per-request identity ContextVars (X-License-Id / X-Subject-Id) propagate
  via the shared ``identity_event_hook``

(LLM calls no longer go through a service-local client — they route through
``worker_management`` via ``w2v_common.clients.WMClient``, whose own base-auth
coverage lives in ``w2v_common``.)
"""

from __future__ import annotations

import contextlib

import httpx
import pytest
from w2v_common.auth.identity_context import (
    clear_identity_context,
    set_identity_context,
)
from w2v_common.tenant import TenantSource, UnvouchedTenantError

from src.clients.ai_content_generation_client import AIContentGenerationClient
from src.clients.app_inspector_client import AppInspectorClient
from src.clients.base import LicensedServiceClient
from src.clients.bdd_playwright_client import BDDPlaywrightClient
from src.clients.frontend_test_runner_client import FrontendTestRunnerClient
from src.clients.media_discovery_client import MediaDiscoveryClient
from src.clients.transcription_worker_client import TranscriptionWorkerClient
from src.clients.video_analyzer_client import VideoAnalyzerClient
from src.clients.video_compile import VideoCompileClient
from src.clients.video_production_client import VideoProductionClient
from src.config import get_settings

_IDEM_KEY = "11111111-1111-7111-8111-111111111111:demo_video_compile"
#: A real uuid: w2v-common 0.67.0 parses the ambient tenant and REJECTS a slug.
_RUN_TENANT = "cccccccc-cccc-cccc-cccc-cccccccccccc"

pytestmark = pytest.mark.asyncio


def _sa_token() -> str:
    """The configured service-account token (always set off/dual — boot requires it)."""
    token = get_settings().downstream.license_token
    assert token is not None
    return token.get_secret_value()


@pytest.fixture(autouse=True)
def _clear_context():
    clear_identity_context()
    yield
    clear_identity_context()


async def _capture_outbound(client_factory, call):
    """Run ``call(client)`` against a MockTransport and return the request."""
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["req"] = request
        # job_id + terminal status let VideoCompileClient's submit→poll flow
        # complete in two requests (POST submit → GET poll); the other clients
        # ignore these extra fields and read their own.
        return httpx.Response(
            200,
            json={
                "job_id": "job-1",
                "status": "completed",
                # stage + progress are required on video_analyzer's JobStatusResponse
                # (VideoAnalyzerJobResult now validates the body); other clients ignore them.
                "stage": "done",
                "progress": 1.0,
                "artifact_url": "http://x/y.mp4",
                "duration_seconds": 1.0,
                "choices": [{"message": {"content": "hi"}}],
                # MediaDiscoveryClient.search validates a full SearchResponse body
                # (MediaSearchResponse — all five fields required); other
                # clients ignore these extras.
                "results": [],
                "total": 0,
                "sources_queried": [],
                "sources_failed": [],
                "search_time_ms": 1.0,
            },
        )

    client = client_factory()
    # set_transport rebuilds _client without baked headers/event_hooks, so
    # we instead patch in a transport on the lazily-built real client.
    real = await client._get_client()
    await real.aclose()
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        timeout=client.timeout,
        headers=dict(real.headers),
        event_hooks=real._event_hooks,
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    try:
        await call(client)
    finally:
        await client.close()
    return captured["req"]


async def test_video_compile_client_sends_license_token():
    """X-Tenant-Id: Bearer <opaque-license-token> is baked in by BaseServiceClient."""
    expected = _sa_token()

    req = await _capture_outbound(
        VideoCompileClient,
        lambda c: c.compile(shots=[], idempotency_key=_IDEM_KEY),
    )

    # Opaque license token travels via X-Tenant-Id (not Authorization);
    # see w2v_common.clients.base._get_client.
    assert req.headers.get("X-Tenant-Id") == f"Bearer {expected}"


async def test_app_inspector_client_sends_license_token():
    """Regression: app_inspect 401'd live because this client sent no token."""
    expected = _sa_token()

    req = await _capture_outbound(
        AppInspectorClient,
        lambda c: c.get_inspection("job-1"),
    )

    assert req.headers.get("X-Tenant-Id") == f"Bearer {expected}"


async def test_ai_content_generation_client_sends_license_token():
    """ai_image_gen → ACG needs the token so promo/generic/course images render."""
    expected = _sa_token()

    req = await _capture_outbound(
        AIContentGenerationClient,
        lambda c: c.get_generation("job-1"),
    )

    assert req.headers.get("X-Tenant-Id") == f"Bearer {expected}"


async def test_video_production_client_sends_license_token():
    """Regression: site_capture → video-production /api/record/url 401'd live
    because this client (uniquely among downstream clients) baked in no token."""
    expected = _sa_token()

    req = await _capture_outbound(
        VideoProductionClient,
        lambda c: c.record_url(url="https://example.com/", part_id="t"),
    )

    assert req.headers.get("X-Tenant-Id") == f"Bearer {expected}"


async def test_frontend_test_runner_client_sends_license_token():
    """frontend_test_runner is license-gated; this client must carry the token."""
    expected = _sa_token()

    req = await _capture_outbound(FrontendTestRunnerClient, lambda c: c.get_job("job-1"))

    assert isinstance(FrontendTestRunnerClient(), LicensedServiceClient)
    assert req.headers.get("X-Tenant-Id") == f"Bearer {expected}"


async def test_bdd_playwright_client_sends_license_token():
    """bdd_playwright_service is license-gated; BDDPlaywrightClient must carry the token."""
    expected = _sa_token()

    req = await _capture_outbound(BDDPlaywrightClient, lambda c: c.get_job("job-1"))

    assert isinstance(BDDPlaywrightClient(), LicensedServiceClient)
    assert req.headers.get("X-Tenant-Id") == f"Bearer {expected}"


async def test_media_discovery_client_sends_license_token():
    """media-discovery is license-gated; MediaDiscoveryClient must carry the token."""
    expected = _sa_token()

    req = await _capture_outbound(MediaDiscoveryClient, lambda c: c.search("widgets"))

    assert isinstance(MediaDiscoveryClient(), LicensedServiceClient)
    assert req.headers.get("X-Tenant-Id") == f"Bearer {expected}"


async def test_video_analyzer_client_sends_license_token():
    """video_analyzer_service is license-gated; VideoAnalyzerClient must carry the
    token (it used to extend the bare BaseServiceClient → a wired caller would 401).

    Instantiating with no ``base_url`` also proves discovery resolves the URL — the
    bare BaseServiceClient required a positional ``base_url`` and would TypeError here.
    """
    expected = _sa_token()

    req = await _capture_outbound(VideoAnalyzerClient, lambda c: c.get_result("job-1"))

    client = VideoAnalyzerClient()
    assert isinstance(client, LicensedServiceClient)
    assert client.timeout == 60.0  # custom non-default timeout survives the conversion
    assert req.headers.get("X-Tenant-Id") == f"Bearer {expected}"


async def test_transcription_worker_client_sends_license_token(tmp_path):
    """ASR (now via modality_gateway, #1106) is license-gated;
    TranscriptionWorkerClient must carry the token
    (it used to extend the bare BaseServiceClient → a wired caller would 401)."""
    expected = _sa_token()
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")

    async def _transcribe(c):
        # The capture transport answers a body that is not a TranscriptionResponse;
        # only the outbound request matters here.
        with contextlib.suppress(RuntimeError):
            await c.transcribe_file(file_path=audio)

    req = await _capture_outbound(TranscriptionWorkerClient, _transcribe)

    client = TranscriptionWorkerClient()
    assert isinstance(client, LicensedServiceClient)
    assert client.timeout == 300.0  # custom non-default timeout survives the conversion
    assert req.headers.get("X-Tenant-Id") == f"Bearer {expected}"


async def test_video_compile_forwards_identity_context_headers():
    """video_compile is a DELEGATING destination (w2v-common 0.80.2): the hop
    must name the RUN's tenant on ``X-Effective-Tenant-Id`` while still
    authenticating with the service-account token on ``X-Tenant-Id``.

    A delegated hop is only allowed for a tenant of VOUCHED provenance, so the
    seeded context carries ``tenant_source`` — exactly as the run task does
    (see ``tests/clients/test_design_client_effective_tenant.py``).
    """
    set_identity_context(
        tenant_id=_RUN_TENANT,
        license_id="lic-1",
        subject_id="sub-7",
        tenant_source=TenantSource.LICENSE_TOKEN,
    )

    req = await _capture_outbound(
        VideoCompileClient,
        lambda c: c.compile(shots=[], idempotency_key=_IDEM_KEY),
    )

    assert req.headers.get("X-License-Id") == "lic-1"
    assert req.headers.get("X-Subject-Id") == "sub-7"
    # The compile job video-compiler creates is scoped to THIS value.
    assert req.headers.get("X-Effective-Tenant-Id") == _RUN_TENANT
    # The constructor-baked Bearer header wins over the bare-UUID ContextVar
    # value (preserving the service-account introspection path).
    assert req.headers.get("X-Tenant-Id", "").lower().startswith("bearer ")


async def test_video_compile_refuses_an_unvouched_tenant():
    """Fail closed. A tenant of unrecorded provenance must never be delegated
    to video-compiler — the guard raises before any request reaches the wire."""
    set_identity_context(tenant_id=_RUN_TENANT, license_id="lic-1", subject_id="sub-7")
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"job_id": "job-1", "status": "completed"})

    client = VideoCompileClient()
    real = await client._get_client()
    await real.aclose()
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        timeout=client.timeout,
        headers=dict(real.headers),
        event_hooks=real._event_hooks,
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    try:
        with pytest.raises(UnvouchedTenantError):
            await client.compile(shots=[], idempotency_key=_IDEM_KEY)
    finally:
        await client.close()
    assert not sent, "an unvouched hop must not reach the wire"
