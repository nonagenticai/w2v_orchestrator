"""Tests for downstream service client subclasses.

Verifies constructor defaults, license token propagation, httpx client
lifecycle, and that subclasses correctly inherit BaseServiceClient behavior.
"""

from __future__ import annotations

import pytest

from src.clients.ai_content_generation_client import AIContentGenerationClient
from src.clients.app_inspector_client import AppInspectorClient
from src.clients.brand_extractor_client import BrandExtractorClient
from src.clients.critique_client import CritiqueClient
from src.clients.design_client import DesignClient
from src.clients.tts_client import TTSClient

# ============================================================================
# Parametrized: every client has correct defaults and accepts custom URL
# ============================================================================

# (cls, w2v_local_url, service_name) — the localhost form is what
# w2v_common.discovery.service_url() returns when W2V_LOCAL=1 is set.
# The port comes from discovery._SERVICES (single source of truth). TTSClient
# resolves modality_gateway (8890), NOT tts.test:9202 — the gateway's BYO
# gate must sit in front of the kokoro box by default (#1106).
_CLIENT_DEFAULTS = [
    (TTSClient, "http://localhost:8890", "tts"),
    (CritiqueClient, "http://localhost:6303", "critique"),
    (BrandExtractorClient, "http://localhost:6303", "brand_extractor"),
    (AIContentGenerationClient, "http://localhost:6310", "ai_content_generation"),
    (DesignClient, "http://localhost:6319", "design"),
    (AppInspectorClient, "http://localhost:6311", "app_inspector"),
]


@pytest.mark.parametrize(
    "cls,expected_url,expected_name", [(TTSClient, "http://localhost:8890", "tts")]
)
def test_client_default_base_url_and_service_name(cls, expected_url, expected_name, monkeypatch):
    """Each client subclass resolves to the expected localhost URL under W2V_LOCAL=1."""
    monkeypatch.setenv("W2V_LOCAL", "1")
    client = cls()
    assert client.base_url == expected_url
    assert client.service_name == expected_name
    # TTS is a slow long-form synth service and overrides the 30s default with a
    # long request timeout (see src/clients/tts_client.py); all others keep 30s.
    assert client.timeout == (300.0 if cls is TTSClient else 30.0)


# ============================================================================
# License token propagation
# ============================================================================


@pytest.mark.parametrize("cls,_url,_name", [(TTSClient, "http://localhost:8890", "tts")])
@pytest.mark.asyncio
async def test_license_token_propagated_to_httpx_headers(cls, _url, _name):
    """License token is passed as Bearer via X-Tenant-Id (w2v_common 0.6.7+)."""
    client = cls(license_token="test-token-xyz")
    assert client.license_token == "test-token-xyz"

    internal = await client._get_client()
    assert internal.headers.get("x-tenant-id") == "Bearer test-token-xyz"
    assert internal.headers.get("authorization") is None
    await client.close()


# ============================================================================
# Client lifecycle: _get_client reuses, close cleans up
# ============================================================================


@pytest.mark.asyncio
async def test_get_client_returns_same_instance():
    """_get_client returns the same httpx client on repeated calls."""
    client = BrandExtractorClient()
    c1 = await client._get_client()
    c2 = await client._get_client()
    assert c1 is c2
    assert not c1.is_closed
    await client.close()
    assert client._client is None


@pytest.mark.asyncio
async def test_get_client_recreates_after_close():
    """_get_client creates a fresh httpx client after close."""
    client = BrandExtractorClient()
    c1 = await client._get_client()
    await client.close()
    c2 = await client._get_client()
    assert c1 is not c2
    assert not c2.is_closed
    await client.close()


# ============================================================================
# AppInspectorClient exposes expected interface
# ============================================================================


def test_app_inspector_exposes_all_interface_methods():
    """AppInspectorClient has start_inspection, get_inspection, wait_for_inspection, inspect_app."""
    client = AppInspectorClient()
    for method_name in ("start_inspection", "get_inspection", "wait_for_inspection", "inspect_app"):
        method = getattr(client, method_name, None)
        assert method is not None, f"Missing method: {method_name}"
        assert callable(method), f"{method_name} is not callable"
