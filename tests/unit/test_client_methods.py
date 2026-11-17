"""Tests for service client subclass methods.

Tests the actual request building (correct HTTP method, endpoint, payload
structure, query params) by inspecting what the injected mock httpx client
received — NOT by echoing mock return values.
"""

from __future__ import annotations

import itertools
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients.ai_content_generation_client import AIContentGenerationClient
from src.clients.app_inspector_client import AppInspectorClient
from src.clients.brand_extractor_client import BrandExtractorClient
from src.clients.critique_client import CritiqueClient
from src.clients.design_client import DesignClient
from src.clients.tts_client import TTSClient


def _mock_client_internals(client):
    """Inject a mock httpx client into a service client."""
    mock_httpx = AsyncMock()
    mock_httpx.is_closed = False
    client._client = mock_httpx
    return mock_httpx


def _make_json_response(data: dict, status=200):
    """Create a mock httpx response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _extract_request_call(mock_httpx):
    """Extract (method, endpoint, json, params) from the mock's last request call."""
    call = mock_httpx.request.call_args
    method = call.args[0] if call.args else call.kwargs.get("method")
    endpoint = call.args[1] if len(call.args) > 1 else call.kwargs.get("url")
    sent_json = call.kwargs.get("json")
    sent_params = call.kwargs.get("params")
    return method, endpoint, sent_json, sent_params


# ============================================================================
# TTSClient
# ============================================================================


def _make_synth_response(audio_b64: str = "AAA=", duration: float = 1.5) -> MagicMock:
    """tts_backend /synthesize JSON-mode response."""
    return _make_json_response(
        {
            "audio_base64": audio_b64,
            "sample_rate": 24000,
            "duration_sec": duration,
            "provider": "kokoro",
            "voice": "af_heart",
            "format": "wav",
        }
    )


@pytest.mark.asyncio
async def test_tts_synthesize_posts_correct_payload():
    """TTSClient.synthesize POSTs to /api/tts/synthesize with Accept: application/json."""
    client = TTSClient()
    mock_httpx = _mock_client_internals(client)
    mock_httpx.request.return_value = _make_synth_response()

    result = await client.synthesize("Hello world", voice="en-US", provider="kokoro")

    call = mock_httpx.request.call_args
    method = call.args[0] if call.args else call.kwargs.get("method")
    endpoint = call.args[1] if len(call.args) > 1 else call.kwargs.get("url")
    sent_json = call.kwargs.get("json")
    sent_headers = call.kwargs.get("headers") or {}
    assert method == "POST"
    assert endpoint == "/api/tts/synthesize"
    assert sent_headers.get("Accept") == "application/json"
    assert sent_json["text"] == "Hello world"
    assert sent_json["voice"] == "en-US"
    assert sent_json["provider"] == "kokoro"
    assert sent_json["speed"] == 1.0
    # Decoded bytes round-trip + metadata propagated. "AAA=" → 2 NUL bytes.
    assert result.audio_bytes == b"\x00\x00"
    assert result.duration_seconds == 1.5
    assert result.sample_rate == 24000
    assert result.format == "wav"
    await client.close()


@pytest.mark.asyncio
async def test_tts_synthesize_omits_voice_when_none():
    """TTSClient.synthesize omits voice and 'auto' provider from the payload."""
    client = TTSClient()
    mock_httpx = _mock_client_internals(client)
    mock_httpx.request.return_value = _make_synth_response()

    await client.synthesize("No voice")

    _, _, sent_json, _ = _extract_request_call(mock_httpx)
    assert "voice" not in sent_json
    # "auto" provider is the implicit default; the client suppresses it so
    # tts_backend uses its own configured default (no Literal["auto"] in schema).
    assert "provider" not in sent_json
    await client.close()


@pytest.mark.asyncio
async def test_tts_synthesize_includes_speed_when_non_default():
    """TTSClient.synthesize forwards a non-default speed."""
    client = TTSClient()
    mock_httpx = _mock_client_internals(client)
    mock_httpx.request.return_value = _make_synth_response()

    await client.synthesize("Fast speech", speed=1.5)

    _, _, sent_json, _ = _extract_request_call(mock_httpx)
    assert sent_json["speed"] == 1.5
    await client.close()


@pytest.mark.asyncio
async def test_tts_synthesize_raises_when_no_audio():
    """TTSClient.synthesize raises if the backend returned empty audio_base64."""
    from src.clients.tts_client import TTSSynthesisError

    client = TTSClient()
    mock_httpx = _mock_client_internals(client)
    mock_httpx.request.return_value = _make_json_response(
        {
            "audio_base64": "",
            "sample_rate": 0,
            "duration_sec": 0,
            "provider": "",
            "voice": "",
            "format": "wav",
        }
    )

    with pytest.raises(TTSSynthesisError):
        await client.synthesize("anything")
    await client.close()


# ============================================================================
# CritiqueClient
# ============================================================================


@pytest.mark.asyncio
async def test_critique_start_posts_url_to_critique_endpoint():
    """CritiqueClient.start_critique POSTs url to /api/critique."""
    client = CritiqueClient()
    mock_httpx = _mock_client_internals(client)
    mock_httpx.request.return_value = _make_json_response({"job_id": "cr-1"})

    job_id = await client.start_critique("https://example.com")

    method, endpoint, sent_json, _ = _extract_request_call(mock_httpx)
    assert method == "POST"
    assert endpoint == "/api/critique"
    assert sent_json["url"] == "https://example.com"
    assert "categories" not in sent_json
    assert job_id == "cr-1"
    await client.close()


@pytest.mark.asyncio
async def test_critique_start_includes_categories():
    """CritiqueClient.start_critique includes categories in payload."""
    client = CritiqueClient()
    mock_httpx = _mock_client_internals(client)
    mock_httpx.request.return_value = _make_json_response({"job_id": "cr-cat"})

    await client.start_critique("https://example.com", categories=["ux", "a11y"])

    _, _, sent_json, _ = _extract_request_call(mock_httpx)
    assert sent_json["categories"] == ["ux", "a11y"]
    await client.close()


@pytest.mark.asyncio
async def test_critique_compare_sends_query_params():
    """CritiqueClient.compare_critiques GETs /api/critique/compare with job_id params."""
    client = CritiqueClient()
    mock_httpx = _mock_client_internals(client)
    mock_httpx.request.return_value = _make_json_response(
        {
            "job_id_1": "cr-1",
            "job_id_2": "cr-2",
            "url_1": "https://a.example",
            "url_2": "https://b.example",
            "overall_delta": 8.5,
        }
    )

    result = await client.compare_critiques("cr-1", "cr-2")

    method, endpoint, _, sent_params = _extract_request_call(mock_httpx)
    assert method == "GET"
    assert endpoint == "/api/critique/compare"
    assert sent_params == {"job_id_1": "cr-1", "job_id_2": "cr-2"}
    assert result.overall_delta == 8.5
    await client.close()


@pytest.mark.asyncio
async def test_critique_wait_raises_on_failure():
    """CritiqueClient.wait_for_critique raises RuntimeError on 'failed' status."""
    client = CritiqueClient()
    mock_httpx = _mock_client_internals(client)
    mock_httpx.request.return_value = _make_json_response(
        {"status": "failed", "error": "model unavailable"}
    )

    with pytest.raises(RuntimeError, match="failed"):
        await client.wait_for_critique("cr-fail")
    await client.close()


# ============================================================================
# BrandExtractorClient
# ============================================================================


@pytest.mark.asyncio
async def test_brand_extractor_posts_url():
    """BrandExtractorClient.extract_brand POSTs to /api/brand/from-url, returns a typed model."""
    client = BrandExtractorClient()
    mock_httpx = _mock_client_internals(client)
    mock_httpx.request.return_value = _make_json_response(
        {"name": "TestCo", "primary_color": "#abcdef"}
    )

    result = await client.extract_brand("https://example.com")

    method, endpoint, sent_json, _ = _extract_request_call(mock_httpx)
    assert method == "POST"
    assert endpoint == "/api/brand/from-url"
    assert sent_json["url"] == "https://example.com"
    # download_logo is always sent on the wire (revert-check: the client must forward it).
    assert sent_json["download_logo"] is False
    # extract_brand returns a validated BrandExtraction, not a raw dict.
    assert result.name == "TestCo"
    assert result.primary_color == "#abcdef"
    await client.close()


# ============================================================================
# AIContentGenerationClient
# ============================================================================


@pytest.mark.asyncio
async def test_ai_gen_start_generation_posts_prompt_and_config():
    """AIContentGenerationClient.start_generation POSTs prompt and config."""
    client = AIContentGenerationClient()
    mock_httpx = _mock_client_internals(client)
    mock_httpx.request.return_value = _make_json_response({"job_id": "aig-1"})

    job_id = await client.start_generation("Generate a demo video", config={"style": "modern"})

    method, _, sent_json, _ = _extract_request_call(mock_httpx)
    assert method == "POST"
    assert sent_json["prompt"] == "Generate a demo video"
    # config is spread into the payload via update(), not nested as "config"
    assert sent_json["style"] == "modern"
    assert job_id == "aig-1"
    await client.close()


@pytest.mark.asyncio
async def test_ai_gen_wait_timeout():
    """AIContentGenerationClient.wait_for_generation raises TimeoutError on timeout."""
    client = AIContentGenerationClient()
    mock_httpx = _mock_client_internals(client)
    mock_httpx.request.return_value = _make_json_response({"status": "processing"})

    with (
        patch("time.monotonic", side_effect=itertools.chain([0.0, 0.0], itertools.repeat(500.0))),
        patch("w2v_common.clients.base.asyncio.sleep", new_callable=AsyncMock),
    ):
        with pytest.raises(TimeoutError, match="timed out"):
            await client.wait_for_generation("aig-timeout", timeout=5.0)
    await client.close()


# ============================================================================
# DesignClient
# ============================================================================


@pytest.mark.asyncio
async def test_redesign_start_posts_url_and_findings():
    """DesignClient.start_design POSTs url and critique_findings."""
    client = DesignClient()
    mock_httpx = _mock_client_internals(client)
    mock_httpx.request.return_value = _make_json_response({"job_id": "rd-1"})

    job_id = await client.start_design(
        "https://example.com",
        critique_findings=[{"issue": "contrast"}],
        site_analysis={"framework": "react"},
        html_content="<html>Hello</html>",
    )

    method, _, sent_json, _ = _extract_request_call(mock_httpx)
    assert method == "POST"
    assert sent_json["url"] == "https://example.com"
    assert sent_json["critique_findings"] == [{"issue": "contrast"}]
    assert sent_json["site_analysis"] == {"framework": "react"}
    assert sent_json["html_content"] == "<html>Hello</html>"
    assert job_id == "rd-1"
    await client.close()


@pytest.mark.asyncio
async def test_redesign_start_refinement_posts_feedback_and_focus_areas():
    """DesignClient.start_refinement POSTs feedback and focus_areas."""
    client = DesignClient()
    mock_httpx = _mock_client_internals(client)
    mock_httpx.request.return_value = _make_json_response({"job_id": "rd-refine"})

    job_id = await client.start_refinement(
        "rd-parent",
        feedback="Fix contrast and headings",
        focus_areas=["a11y", "seo"],
    )

    _, _, sent_json, _ = _extract_request_call(mock_httpx)
    assert sent_json["feedback"] == "Fix contrast and headings"
    assert sent_json["focus_areas"] == ["a11y", "seo"]
    assert job_id == "rd-refine"
    await client.close()


@pytest.mark.asyncio
async def test_redesign_get_html_returns_text():
    """DesignClient.get_design_html returns the HTML text response body."""
    client = DesignClient()
    mock_httpx = _mock_client_internals(client)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<html>ok</html>"
    mock_httpx.get.return_value = mock_resp

    html = await client.get_design_html("rd-1")

    assert html == "<html>ok</html>"
    await client.close()


@pytest.mark.asyncio
async def test_redesign_wait_polls_then_returns():
    """DesignClient.wait_for_design polls then returns the typed status body.

    The wire payloads stay raw dicts (the httpx layer is what's mocked); the
    client validates the final body into ``DesignJobStatus``. Extra keys the
    server model doesn't declare never arrive (FastAPI response_model strips
    them), so only real ``JobStatusResponse`` fields appear here.
    """
    client = DesignClient()
    mock_httpx = _mock_client_internals(client)
    mock_httpx.request.side_effect = [
        _make_json_response({"job_id": "rd-wait", "status": "processing"}),
        _make_json_response(
            {"job_id": "rd-wait", "status": "completed", "accessibility_score": 91.5}
        ),
    ]

    with patch("w2v_common.clients.base.asyncio.sleep", new_callable=AsyncMock):
        result = await client.wait_for_design("rd-wait")

    assert result.job_id == "rd-wait"
    assert result.status == "completed"
    assert result.accessibility_score == 91.5
    await client.close()


# ============================================================================
# AppInspectorClient
# ============================================================================


@pytest.mark.asyncio
async def test_app_inspector_start_inspection_posts_url():
    """AppInspectorClient.start_inspection POSTs url to the inspection endpoint."""
    client = AppInspectorClient()
    mock_httpx = _mock_client_internals(client)
    mock_httpx.request.return_value = _make_json_response({"job_id": "insp-1"})

    job_id = await client.start_inspection("http://example.com")

    method, _, sent_json, _ = _extract_request_call(mock_httpx)
    assert method == "POST"
    # AppInspectorClient maps url to "app_path" in the payload
    assert sent_json["app_path"] == "http://example.com"
    assert job_id == "insp-1"
    await client.close()


@pytest.mark.asyncio
async def test_app_inspector_start_includes_config():
    """AppInspectorClient.start_inspection includes config when provided."""
    client = AppInspectorClient()
    mock_httpx = _mock_client_internals(client)
    mock_httpx.request.return_value = _make_json_response({"job_id": "insp-2"})

    # Config keys like platform, max_screens etc are spread into payload
    # Unrecognized keys like "depth" are NOT included
    await client.start_inspection(
        "http://example.com", config={"platform": "android", "max_screens": 10}
    )

    _, _, sent_json, _ = _extract_request_call(mock_httpx)
    assert sent_json["app_path"] == "http://example.com"
    assert sent_json["platform"] == "android"
    assert sent_json["max_screens"] == 10
    await client.close()


@pytest.mark.asyncio
async def test_app_inspector_inspect_app_submits_polls_then_fetches_result():
    """inspect_app submits, polls the THIN status, then GETs the FAT /result.

    Three HTTP calls: POST /api/inspect, GET /api/inspect/{id} (status until
    completed), GET /api/inspect/{id}/result. The old code stopped at the status
    poll and returned that body — the dead-signal bug this fix closes.
    """
    client = AppInspectorClient()
    mock_httpx = _mock_client_internals(client)
    mock_httpx.request.side_effect = [
        _make_json_response({"job_id": "insp-full"}),
        _make_json_response(
            {
                "job_id": "insp-full",
                "status": "completed",
                "result_url": "/api/inspect/insp-full/result",
            }
        ),
        _make_json_response(
            {
                "job_id": "insp-full",
                "status": "completed",
                "estimated_purpose": "A note-taking app",
                "screens_discovered": 1,
                "overall_accessibility_score": 95,
                "screens": [{"screen_id": "s1", "screen_type": "home", "accessibility_score": 95}],
            }
        ),
    ]

    result = await client.inspect_app("https://example.com", config={"depth": 5})

    assert mock_httpx.request.call_count == 3
    assert result.job_id == "insp-full"
    assert result.status == "completed"
    assert result.estimated_purpose == "A note-taking app"
    assert result.screens_discovered == 1
    await client.close()
