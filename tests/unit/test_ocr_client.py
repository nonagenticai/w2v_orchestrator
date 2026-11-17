"""OcrClient — provider_contracts ``POST /v1/ocr`` wire, retry policy, key hygiene."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from pydantic import SecretStr

from src.clients.ocr_client import MAX_ATTEMPTS, OcrClient, OcrError, OcrNotConfiguredError
from src.config import ProviderOcrSettings

pytestmark = pytest.mark.asyncio

_KEY = "sk-ocr-test-key"  # pragma: allowlist secret


def _settings(url: str = "http://ocr.test:9131/", key: str = _KEY) -> ProviderOcrSettings:
    return ProviderOcrSettings(
        W2V_PROVIDER_OCR_URL=url,
        W2V_PROVIDER_OCR_API_KEY=SecretStr(key),
        W2V_PROVIDER_OCR_MODEL="PaddleOCR-VL-1.5-bf16",
    )


class _Script:
    """Answers each request with the next scripted (status, body) and records it."""

    def __init__(self, *answers: tuple[int, object]) -> None:
        self.answers = list(answers)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, body = self.answers.pop(0)
        return httpx.Response(status, json=body)


_OK = {
    "model": "PaddleOCR-VL-1.5-bf16",
    "pages": [
        {"index": 1, "markdown": "second", "images": [], "dimensions": {"dpi": 200}},
        {"index": 0, "markdown": "first", "images": []},
    ],
    "usage_info": {"pages_processed": 2, "doc_size_bytes": 10},
}


def _err(status: int, kind: str) -> tuple[int, object]:
    return status, {"error": {"message": f"{kind} {status}", "type": kind, "code": "c"}}


def _client(
    script: _Script,
    sleeps: list[float] | None = None,
    *,
    settings: ProviderOcrSettings | None = None,
) -> OcrClient:
    async def _sleep(s: float) -> None:
        if sleeps is not None:
            sleeps.append(s)

    return OcrClient(
        settings if settings is not None else _settings(),
        transport=httpx.MockTransport(script),
        sleep=_sleep,
    )


async def test_wire_shape_and_bearer_auth() -> None:
    script = _Script((200, _OK))
    pages = await _client(script).ocr_pdf(b"%PDF-1.4 x", pages=[0, 1])
    assert [(p.index, p.markdown) for p in pages] == [(0, "first"), (1, "second")]
    [req] = script.requests
    assert str(req.url) == "http://ocr.test:9131/v1/ocr"
    assert req.headers["authorization"] == f"Bearer {_KEY}"
    # mlx_api is not a w2v mesh peer: no identity triple goes out.
    assert "x-tenant-id" not in req.headers
    body = json.loads(req.content)
    assert body["model"] == "PaddleOCR-VL-1.5-bf16"
    assert body["document"]["type"] == "document_url"
    assert body["document"]["document_url"] == "data:application/pdf;base64,JVBERi0xLjQgeA=="
    assert body["pages"] == [0, 1]
    assert body["include_image_base64"] is False


async def test_not_configured_raises() -> None:
    client = OcrClient(ProviderOcrSettings())
    assert client.enabled is False
    with pytest.raises(OcrNotConfiguredError, match="W2V_PROVIDER_OCR_URL"):
        await client.ocr_pdf(b"%PDF")


@pytest.mark.parametrize("status", [408])
async def test_retryable_statuses_are_retried(status: int) -> None:
    sleeps: list[float] = []
    script = _Script(_err(status, "retryable"), (200, _OK))
    pages = await _client(script, sleeps).ocr_pdf(b"%PDF")
    assert len(pages) == 2
    assert len(script.requests) == 2
    assert len(sleeps) == 1


@pytest.mark.parametrize(
    ("status", "kind"),
    [(400, "invalid_request"), (401, "auth"), (404, "not_found"), (413, "invalid_request")],
)
async def test_non_retryable_statuses_fail_at_once(status: int, kind: str) -> None:
    script = _Script(_err(status, kind))
    with pytest.raises(OcrError, match=f"HTTP {status}.*{kind} {status}"):
        await _client(script).ocr_pdf(b"%PDF")
    assert len(script.requests) == 1


async def test_retry_budget_is_bounded() -> None:
    script = _Script(*[_err(503, "retryable")] * MAX_ATTEMPTS)
    with pytest.raises(OcrError, match="HTTP 503"):
        await _client(script).ocr_pdf(b"%PDF")
    assert len(script.requests) == MAX_ATTEMPTS


async def test_transport_errors_are_retried_then_raise() -> None:
    calls = 0

    def boom(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused", request=request)

    client = OcrClient(_settings(), transport=httpx.MockTransport(boom), sleep=_noop)
    with pytest.raises(OcrError, match="unreachable after 3 attempts"):
        await client.ocr_pdf(b"%PDF")
    assert calls == MAX_ATTEMPTS


async def _noop(_s: float) -> None:
    return None


@pytest.mark.parametrize(
    "body",
    [[]],
)
async def test_malformed_response_raises(body: object) -> None:
    with pytest.raises(OcrError):
        await _client(_Script((200, body))).ocr_pdf(b"%PDF")


async def test_key_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    script = _Script(_err(503, "retryable"), _err(401, "auth"))
    with pytest.raises(OcrError) as info:
        await _client(script).ocr_pdf(b"%PDF")
    assert _KEY not in caplog.text
    assert _KEY not in str(info.value)


async def test_no_key_sends_no_authorization_header() -> None:
    script = _Script((200, _OK))
    await _client(script, settings=_settings(key="")).ocr_pdf(b"%PDF")
    assert "authorization" not in script.requests[0].headers


async def test_settings_reject_a_non_http_url() -> None:
    with pytest.raises(ValueError, match="http"):
        ProviderOcrSettings(W2V_PROVIDER_OCR_URL="mlx-api-media.test:9131")


async def test_settings_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("W2V_PROVIDER_OCR_URL", "http://mlx-api-media.test:9131")
    monkeypatch.setenv("W2V_PROVIDER_OCR_API_KEY", _KEY)
    monkeypatch.setenv("W2V_PROVIDER_OCR_MODEL", "mistral-ocr-latest")
    s = ProviderOcrSettings()
    assert s.enabled
    assert s.provider_ocr_api_key.get_secret_value() == _KEY
    assert s.provider_ocr_model == "mistral-ocr-latest"
    assert _KEY not in repr(s)
