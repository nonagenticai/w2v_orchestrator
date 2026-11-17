"""OCR client for the provider_contracts ``POST /v1/ocr`` wire (Mistral OCR shape).

Used by :mod:`src.services.documents.extract` to OCR the pages of an uploaded PDF
that carry no text layer (scans). Owner decision 2026-10-17: the provider is
``mlx_api`` for now (``mlx-api-media.test:9131``, PaddleOCR-VL); because the
wire is the Mistral OCR contract, a commercial vendor is a config change
(``W2V_PROVIDER_OCR_{URL,API_KEY,MODEL}`` — :class:`src.config.ProviderOcrSettings`).

Wire (provider_contracts ``CONTRACTS.md`` § ocr; mlx_api ``app.py`` ``/v1/ocr``):

* request ``{"model", "document": {"type": "document_url", "document_url":
  "data:application/pdf;base64,…"}, "pages"?: [0-based], "include_image_base64": false}``
  — internal documents are always sent as ``data:`` URLs (mlx_api refuses to fetch
  a non-public URL, by design);
* response ``{"model", "pages": [{"index", "markdown", ...}], "usage_info": {...}}``,
  one entry per requested page, ascending ``index``;
* auth ``Authorization: Bearer <key>`` on every path but ``/health``;
* errors carry ``{"error": {"message", "type", "code"}}``; only ``retryable``
  (408 / 429 / 5xx) is retried — 4xx (auth, invalid request, unknown model) is a
  configuration or input error and fails at once.

**Provider slot ``ocr``** (owner decision 2026-10-22): OCR is a plain web provider
like every other model slot. :meth:`OcrClient.for_bound_run` (the production
default of ``extract_document``) resolves the bound run's ``ocr`` pin through
``providers/resolver.resolve_slot`` — so the tenant's licences ``byo_mode`` applies
exactly as for llm/tts/…: ``local_allowed`` → the tenant's own OCR connection if
any, else the platform URL above; ``byo_required`` → the tenant's connection or a
:class:`~src.services.providers.errors.SlotNotConfiguredError` (``ocr``), never the
platform; ``platform_only`` → the platform URL. A tenant connection is either
``nonagentic_endpoint`` (``POST {base_url}/v1/ocr`` — a copy of mlx_api) or
``openai-compatible`` (``POST {base_url}/ocr`` with ``base_url`` ending ``/v1``,
e.g. Mistral), same wire. Its egress is pinned to the resolver's vetted IP
(``in_process.PinnedTransport``), redirects are refused.

mlx_api is not a w2v mesh peer: the bearer key is its only credential and it reads
no ``X-Tenant-Id`` / ``X-License-Id``, so the w2v identity triple is deliberately
NOT forwarded (``tests/test_no_bare_httpx_client.py`` NON_W2V_PEER). The key is
never logged, and neither is a request body (it carries the tenant's document).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, cast

import httpx
import httpx2

from src.config import ProviderOcrSettings, get_settings
from src.services.providers.registry import KIND_NONAGENTIC_ENDPOINT, OCR_ROUTE

if TYPE_CHECKING:
    from src.services.providers.resolver import ResolvedSlot

log = logging.getLogger(__name__)

#: Contract status classes that are ``retryable`` (CONTRACTS.md common rules).
_RETRYABLE_STATUS: Final = frozenset({408, 429})
MAX_ATTEMPTS: Final = 3
_BACKOFF_SECONDS: Final = 2.0
_RETRY_AFTER_CAP_SECONDS: Final = 30.0
#: One OCR request carries a handful of scanned pages; a VLM OCR page is seconds.
DEFAULT_TIMEOUT_SECONDS: Final = 180.0
_ERROR_DETAIL_MAX_CHARS: Final = 300


class OcrError(RuntimeError):
    """The OCR provider could not OCR the document (after any retries)."""


class OcrNotConfiguredError(OcrError):
    """No OCR provider is configured (``W2V_PROVIDER_OCR_URL`` is empty)."""


#: The registry slot OCR is resolved through.
SLOT: Final = "ocr"
_Response = httpx.Response | httpx2.Response


@dataclass(frozen=True, slots=True)
class OcrTarget:
    """Where one OCR call goes: the platform URL, or the tenant's resolved connection."""

    endpoint: str
    model: str
    api_key: str | None
    source: Literal["deployment", "tenant"] = "deployment"
    #: Tenant origin + the resolver's vetted IP (DNS-rebinding pin; None = allowlisted).
    base_url: str | None = None
    pinned_ip: str | None = None


def platform_target(settings: ProviderOcrSettings) -> OcrTarget | None:
    """The platform OCR provider (``W2V_PROVIDER_OCR_*``), or ``None`` when OCR is off."""
    if not settings.enabled:
        return None
    key = settings.provider_ocr_api_key.get_secret_value()
    return OcrTarget(
        endpoint=f"{settings.provider_ocr_url.strip().rstrip('/')}{OCR_ROUTE}",
        model=settings.provider_ocr_model,
        api_key=key or None,
    )


def tenant_target(resolved: ResolvedSlot, settings: ProviderOcrSettings) -> OcrTarget:
    """The tenant ``ocr`` connection as an :class:`OcrTarget` (registry kinds only)."""
    from src.services.providers.errors import SlotUnavailableError

    if not resolved.base_url:
        raise SlotUnavailableError(SLOT, f"kind {resolved.kind!r} requires a base URL")
    base = resolved.base_url.rstrip("/")
    # nonagentic_endpoint = a copy of the platform service (route under the root);
    # openai-compatible = a /v1 base URL (``POST {base_url}/ocr``, Mistral-style).
    endpoint = f"{base}{OCR_ROUTE}" if resolved.kind == KIND_NONAGENTIC_ENDPOINT else f"{base}/ocr"
    model = resolved.model or settings.provider_ocr_model
    secret = resolved.secret.get_secret_value() if resolved.secret is not None else ""
    return OcrTarget(
        endpoint=endpoint,
        model=model,
        api_key=secret or None,
        source="tenant",
        base_url=resolved.base_url,
        pinned_ip=resolved.pinned_ip,
    )


async def resolve_ocr_target(settings: ProviderOcrSettings) -> OcrTarget | None:
    """Resolve the BOUND tenant's ``ocr`` slot (the run's pin, else live) — byo_mode applied.

    Raises :class:`~src.services.providers.errors.SlotNotConfiguredError` for a
    ``byo_required`` tenant without its own OCR connection and
    :class:`~src.services.providers.errors.SlotUnavailableError` for an unusable one
    — never a silent platform fallback. ``None`` = the platform default with OCR off."""
    from w2v_common.auth.identity_context import get_tenant_id_from_context

    from src.auth.run_providers import current_run_slot
    from src.database.session import bind_session_tenant, get_db_session
    from src.services.providers.adapters import build_resolver_deps
    from src.services.providers.errors import SlotUnavailableError
    from src.services.providers.resolver import PROVIDER_SLOTS_KEY, resolve_slot

    raw_tenant = get_tenant_id_from_context()
    if not raw_tenant:
        # No tenant (a system / background call): CommonPractice — no tenant ⇒ platform.
        return platform_target(settings)
    try:
        tenant_id = uuid.UUID(str(raw_tenant))
    except ValueError, TypeError:
        raise SlotUnavailableError(SLOT, "the bound tenant id is malformed") from None
    ref = current_run_slot(SLOT)
    snapshot = {PROVIDER_SLOTS_KEY: {SLOT: ref.to_dict()}} if ref is not None else None
    async with get_db_session() as session:
        await bind_session_tenant(session, tenant_id)
        deps = build_resolver_deps(session, get_settings())
        resolved = await resolve_slot(
            deps.repo,
            deps.decryptor,
            tenant_id,
            SLOT,
            run_snapshot=snapshot,
            policy=deps.policy,
            posture=deps.posture,
        )
    if resolved.source == "tenant":
        return tenant_target(resolved, settings)
    return platform_target(settings)


TargetResolver = Callable[[ProviderOcrSettings], Awaitable[OcrTarget | None]]


@dataclass(frozen=True, slots=True)
class OcrPage:
    """One OCR'd page: its 0-based ``index`` in the SENT document and its text."""

    index: int
    markdown: str


def _is_retryable(status: int) -> bool:
    return status in _RETRYABLE_STATUS or status >= 500


def _retry_after(response: _Response, attempt: int) -> float:
    raw = response.headers.get("retry-after", "")
    try:
        return min(max(float(raw), 0.0), _RETRY_AFTER_CAP_SECONDS)
    except ValueError:
        return _BACKOFF_SECONDS * attempt


def _error_detail(response: _Response) -> str:
    """The contract error message (or a bounded body excerpt), never the request."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:_ERROR_DETAIL_MAX_CHARS]
    if isinstance(body, dict):
        err = cast("dict[str, object]", body).get("error")
        if isinstance(err, dict):
            err_d = cast("dict[str, object]", err)
            msg = str(err_d.get("message") or "")
            code = err_d.get("code")
            return (f"{msg} (code={code})" if code else msg)[:_ERROR_DETAIL_MAX_CHARS]
    return str(body)[:_ERROR_DETAIL_MAX_CHARS]


def _parse_pages(body: object) -> list[OcrPage]:
    if not isinstance(body, dict):
        raise OcrError(f"OCR response is {type(body).__name__}, not an object")
    pages = cast("dict[str, object]", body).get("pages")
    if not isinstance(pages, list):
        raise OcrError("OCR response has no 'pages' list")
    out: list[OcrPage] = []
    for raw in cast("list[object]", pages):
        if not isinstance(raw, dict):
            raise OcrError("OCR response page is not an object")
        page = cast("dict[str, object]", raw)
        index, markdown = page.get("index"), page.get("markdown")
        if not isinstance(index, int) or isinstance(index, bool) or not isinstance(markdown, str):
            raise OcrError("OCR response page lacks an int 'index' / str 'markdown'")
        out.append(OcrPage(index=index, markdown=markdown))
    return sorted(out, key=lambda p: p.index)


class OcrClient:
    """Async client for the ``ocr`` provider slot (``POST …/v1/ocr``).

    Built with explicit ``settings`` (and no ``resolve_target``) it always calls the
    platform URL — the unit-test shape. :meth:`for_bound_run` resolves the bound
    run's ``ocr`` slot instead (production). ``transport`` (platform path, an
    ``httpx.MockTransport``) and ``sleep`` are test seams; the tenant path's
    transport is ``in_process.inner_transport_factory``.
    """

    def __init__(
        self,
        settings: ProviderOcrSettings | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        resolve_target: TargetResolver | None = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings().provider_ocr
        self._timeout = timeout
        self._transport = transport
        self._sleep = sleep
        self._resolve_target = resolve_target

    @classmethod
    def for_bound_run(
        cls,
        settings: ProviderOcrSettings | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        resolve_target: TargetResolver | None = None,
    ) -> OcrClient:
        """The production client: the bound run's ``ocr`` slot via the resolver
        (:func:`resolve_ocr_target`, looked up at call time so tests can pin it)."""
        return cls(
            settings,
            timeout=timeout,
            sleep=sleep,
            resolve_target=resolve_target or (lambda s: resolve_ocr_target(s)),
        )

    @property
    def enabled(self) -> bool:
        """True when the PLATFORM OCR provider URL is configured."""
        return self._settings.enabled

    @property
    def endpoint(self) -> str:
        """The platform ``/v1/ocr`` URL (no credential in it)."""
        return f"{self._settings.provider_ocr_url.strip().rstrip('/')}{OCR_ROUTE}"

    async def target(self) -> OcrTarget | None:
        """Where OCR goes for this call, or ``None`` (OCR off). May raise the slot's
        BYO / unavailable errors (see :func:`resolve_ocr_target`)."""
        if self._resolve_target is None:
            return platform_target(self._settings)
        return await self._resolve_target(self._settings)

    async def ocr_pdf(
        self, pdf: bytes, *, pages: list[int] | None = None, target: OcrTarget | None = None
    ) -> list[OcrPage]:
        """OCR ``pdf`` (all pages, or the 0-based ``pages``) and return its pages.

        ``target`` defaults to :meth:`target`. Raises :class:`OcrNotConfiguredError`
        when no provider is configured and :class:`OcrError` on a non-retryable
        answer, an exhausted retry budget or a malformed response.
        """
        if target is None:
            target = await self.target()
        if target is None:
            raise OcrNotConfiguredError("OCR is not configured (W2V_PROVIDER_OCR_URL is empty)")
        payload: dict[str, object] = {
            "model": target.model,
            "document": {
                "type": "document_url",
                "document_url": "data:application/pdf;base64,"
                + base64.b64encode(pdf).decode("ascii"),
            },
            "include_image_base64": False,
        }
        if pages is not None:
            payload["pages"] = list(pages)
        headers = {"Accept": "application/json"}
        if target.api_key:
            headers["Authorization"] = f"Bearer {target.api_key}"
        if target.source == "tenant" and target.base_url:
            from src.services.llm import in_process

            # The tenant's endpoint: one origin, the resolver's vetted IP, no redirects.
            async with httpx2.AsyncClient(
                transport=in_process.PinnedTransport(
                    target.base_url, target.pinned_ip, in_process.inner_transport_factory()
                ),
                timeout=httpx2.Timeout(self._timeout, connect=min(10.0, self._timeout)),
                follow_redirects=False,
                headers=headers,
            ) as tenant_client:
                return _parse_pages(
                    await self._post_with_retry(tenant_client, target.endpoint, payload)
                )
        async with httpx.AsyncClient(
            timeout=self._timeout, transport=self._transport, headers=headers
        ) as client:
            return _parse_pages(await self._post_with_retry(client, target.endpoint, payload))

    async def _post_with_retry(
        self,
        client: httpx.AsyncClient | httpx2.AsyncClient,
        url: str,
        payload: dict[str, object],
    ) -> object:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = await client.post(url, json=payload)
            except (
                httpx.TimeoutException,
                httpx.TransportError,
                httpx2.TimeoutException,
                httpx2.TransportError,
            ) as exc:
                if attempt >= MAX_ATTEMPTS:
                    raise OcrError(
                        f"OCR provider unreachable after {attempt} attempts: {type(exc).__name__}"
                    ) from exc
                delay = _BACKOFF_SECONDS * attempt
                log.warning(
                    "OCR transport error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt,
                    MAX_ATTEMPTS,
                    delay,
                    type(exc).__name__,
                )
                await self._sleep(delay)
                continue
            status = response.status_code
            if status < 300:
                try:
                    return response.json()
                except ValueError as exc:
                    raise OcrError("OCR response is not JSON") from exc
            if not _is_retryable(status) or attempt >= MAX_ATTEMPTS:
                raise OcrError(
                    f"OCR provider answered HTTP {status} (attempt {attempt}/{MAX_ATTEMPTS}): "
                    f"{_error_detail(response)}"
                )
            delay = _retry_after(response, attempt)
            log.warning(
                "OCR provider HTTP %d (attempt %d/%d), retrying in %.1fs",
                status,
                attempt,
                MAX_ATTEMPTS,
                delay,
            )
            await self._sleep(delay)
        raise OcrError("OCR retry loop exhausted")  # pragma: no cover -- loop always returns/raises
