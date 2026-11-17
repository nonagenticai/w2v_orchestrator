"""One OCR call for the service lab, for an IMAGE as well as a PDF.

``src/clients/ocr_client.py`` is the production OCR lane and is built around the
document-extraction use case: its only entry point is :meth:`OcrClient.ocr_pdf`,
which sends the bytes as ``data:application/pdf;base64,…``. The service lab's bench
uploads a SCREENSHOT or a photo, so the same provider contract
(``POST {endpoint}``, ``{"document": {"type": "document_url", "document_url":
"data:<type>;base64,…"}}``, answer ``pages[].markdown``) has to be posted with the
upload's REAL media type.

Everything else is deliberately the same as the client: the SAME resolved
:class:`~src.clients.ocr_client.OcrTarget` (a tenant connection or the platform
``W2V_PROVIDER_OCR_URL``), the same pinned-IP/no-redirect transport for a tenant
origin, the same retry policy (408/429/5xx, ``Retry-After`` honoured and capped) and
the same error types. Only the data-URL media type differs — this module exists so
that difference lives in ONE place instead of forking the client.

The client itself is not edited: document extraction has no reason to grow an image
path, and its tests pin the PDF wire shape byte-for-byte.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Final

import httpx2

from src.clients.ocr_client import OcrError, OcrNotConfiguredError, OcrTarget
from src.services.llm import in_process

__all__ = ["MAX_ATTEMPTS", "run_ocr"]

log = logging.getLogger(__name__)

#: Same budget as the production client (``ocr_client.MAX_ATTEMPTS``).
MAX_ATTEMPTS: Final = 3
#: Contract status classes that are ``retryable`` (``CONTRACTS.md`` common rules).
_RETRYABLE_STATUS: Final = frozenset({408, 429})
_BACKOFF_SECONDS: Final = 2.0
_RETRY_AFTER_CAP_SECONDS: Final = 30.0
_ERROR_DETAIL_MAX_CHARS: Final = 300


async def run_ocr(
    target: OcrTarget,
    data: bytes,
    content_type: str,
    *,
    timeout: float,
    transport: httpx2.AsyncBaseTransport | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> str:
    """OCR ``data`` (an image or a PDF) and return the pages' markdown, joined.

    Raises :class:`~src.clients.ocr_client.OcrError` on a non-retryable answer, an
    exhausted retry budget or a malformed response. Never raises
    ``OcrNotConfiguredError`` — the caller decides that (it may have a tenant
    target even when the platform URL is empty).
    """
    if not data:
        raise OcrError("OCR was asked for an empty document")
    media_type = (content_type or "application/octet-stream").split(";")[0].strip().lower()
    payload: dict[str, object] = {
        "model": target.model,
        "document": {
            "type": "document_url",
            "document_url": f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}",
        },
        "include_image_base64": False,
    }
    headers = {"Accept": "application/json"}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"

    if target.source == "tenant" and target.base_url:
        # The tenant's endpoint: one origin, the resolver's vetted IP, no redirects.
        async with httpx2.AsyncClient(
            transport=in_process.PinnedTransport(
                target.base_url, target.pinned_ip, in_process.inner_transport_factory()
            ),
            timeout=httpx2.Timeout(timeout, connect=min(10.0, timeout)),
            follow_redirects=False,
            headers=headers,
        ) as tenant_client:
            return await _post_with_retry(tenant_client, target.endpoint, payload, sleep=sleep)

    async with httpx2.AsyncClient(
        timeout=httpx2.Timeout(timeout, connect=min(10.0, timeout)),
        transport=transport,
        follow_redirects=False,
        headers=headers,
    ) as client:
        return await _post_with_retry(client, target.endpoint, payload, sleep=sleep)


async def _post_with_retry(
    client: httpx2.AsyncClient,
    url: str,
    payload: Mapping[str, object],
    *,
    sleep: Callable[[float], Awaitable[None]],
) -> str:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = await client.post(url, json=dict(payload))
        except (httpx2.TimeoutException, httpx2.TransportError) as exc:
            if attempt >= MAX_ATTEMPTS:
                raise OcrError(
                    f"OCR provider unreachable after {attempt} attempts: {type(exc).__name__}"
                ) from exc
            delay = _BACKOFF_SECONDS * attempt
            log.warning(
                "service-lab OCR transport error (attempt %d/%d), retrying in %.1fs: %s",
                attempt,
                MAX_ATTEMPTS,
                delay,
                type(exc).__name__,
            )
            await sleep(delay)
            continue
        status = response.status_code
        if status < 300:
            try:
                return _pages_markdown(response.json())
            except ValueError as exc:
                raise OcrError("OCR response is not JSON") from exc
        if status not in _RETRYABLE_STATUS and status < 500:
            raise OcrError(f"OCR provider answered HTTP {status}: {_error_detail(response)}")
        if attempt >= MAX_ATTEMPTS:
            raise OcrError(
                f"OCR provider answered HTTP {status} (attempt {attempt}/{MAX_ATTEMPTS}): "
                f"{_error_detail(response)}"
            )
        delay = _retry_after(response, attempt)
        log.warning(
            "service-lab OCR provider HTTP %d (attempt %d/%d), retrying in %.1fs",
            status,
            attempt,
            MAX_ATTEMPTS,
            delay,
        )
        await sleep(delay)
    raise OcrError("OCR retry loop exhausted")  # pragma: no cover -- loop returns or raises


def _retry_after(response: httpx2.Response, attempt: int) -> float:
    raw = response.headers.get("retry-after", "")
    try:
        return min(max(float(raw), 0.0), _RETRY_AFTER_CAP_SECONDS)
    except ValueError:
        return _BACKOFF_SECONDS * attempt


def _error_detail(response: httpx2.Response) -> str:
    """The contract error message (or a bounded body excerpt), never the request."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:_ERROR_DETAIL_MAX_CHARS]
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            message = str(err.get("message") or "")
            code = err.get("code")
            return (f"{message} (code={code})" if code else message)[:_ERROR_DETAIL_MAX_CHARS]
    return str(body)[:_ERROR_DETAIL_MAX_CHARS]


def _pages_markdown(body: object) -> str:
    """``pages[].markdown`` in page order — the OCR contract's answer (as the client parses it)."""
    if not isinstance(body, dict):
        raise OcrError(f"OCR response is {type(body).__name__}, not an object")
    pages = body.get("pages")
    if not isinstance(pages, list):
        raise OcrError("OCR response has no 'pages' list")
    parsed: list[tuple[int, str]] = []
    for raw in pages:
        if not isinstance(raw, dict):
            raise OcrError("OCR response page is not an object")
        index, markdown = raw.get("index"), raw.get("markdown")
        if not isinstance(index, int) or isinstance(index, bool) or not isinstance(markdown, str):
            raise OcrError("OCR response page lacks an int 'index' / str 'markdown'")
        parsed.append((index, markdown))
    return "\n\n".join(markdown for _, markdown in sorted(parsed)).strip()


def require_target(target: OcrTarget | None) -> OcrTarget:
    """The resolved target, or the existing "OCR is off" error (platform URL empty)."""
    if target is None:
        raise OcrNotConfiguredError("OCR is not configured (W2V_PROVIDER_OCR_URL is empty)")
    return target
