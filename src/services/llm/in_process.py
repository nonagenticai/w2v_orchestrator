"""Run one ``WMClient.message()``-shaped request IN-PROCESS on a tenant ``llm`` slot.

Used only when the run pinned a TENANT connection for ``llm`` (P3 WP-LLM). The
request contract is WM's — ``system_prompt``, ``message`` / ``content_blocks``,
``output_schema``, ``max_tokens`` — and the result is the same :class:`WMResponse`
the WM path returns, so the ~43 call sites cannot tell the difference.

Provider kinds:

* ``openai-compatible`` — ``base_url`` + bearer key + model (OpenAI Chat Completions).
* ``anthropic`` — key + model (``base_url`` optional, defaults to Anthropic's API).

Egress: the resolver validated ``base_url`` against the SSRF policy and returned the
IP it resolved to (``ResolvedSlot.pinned_ip``). :class:`PinnedTransport` connects to
exactly that IP (Host header + TLS SNI keep the hostname, so the certificate is still
verified against the name), so there is no DNS-rebinding window between the check and
the connect. A request for any other origin is refused before it leaves the process.

Errors are raised as the orchestrator's own :class:`TransientUpstreamError` /
:class:`PermanentUpstreamError` (``guarded`` passes them through untouched), with a
sanitised message: the key is never part of an error or a log line.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, cast
from urllib.parse import urlsplit

import httpx2
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic_ai import Agent, BinaryContent, ModelRetry, ModelSettings, StructuredDict
from pydantic_ai.exceptions import (
    ContentFilterError,
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RunUsage
from w2v_common.clients import WMResponse
from w2v_common.clients import base as _wm_retry  # the WM path's retry policy, verbatim

from src.models.failure_class import CUSTOMER_INPUT, EXTERNAL, OUR_DEFECT, FailureClass
from src.services.llm.worker_defaults import get_worker_defaults
from src.services.providers.errors import SlotUnavailableError
from src.services.providers.registry import KIND_ANTHROPIC, KIND_OPENAI_COMPATIBLE
from src.services.providers.resolver import ResolvedSlot
from src.services.stages.upstream import (
    PermanentUpstreamError,
    TransientUpstreamError,
    UpstreamAuthError,
)

log = logging.getLogger(__name__)

#: A JSON schema / JSON object as WM receives it (untyped JSON by contract).
JsonSchema = dict[str, object]

SLOT: Final = "llm"

#: pydantic-ai output-validation retries — WM's agents run with 3 ("Exceeded maximum
#: output retries (3)"), so a bad draw gets the same number of re-asks here.
OUTPUT_RETRIES: Final = 3

#: Statuses a retry can fix (provider rate limit / overload / outage).
_TRANSIENT_HTTP: Final = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_AUTH_HTTP: Final = frozenset({401, 403})

#: Same prose markers ``upstream.py`` uses for WM's deterministic 5xx/422 bodies.
_TOKEN_LIMIT_MARKERS: Final = ("token limit", "exceeded before any response", "context length")

#: Anything key-shaped in a provider error body (``sk-…``, ``sk-ant-…``, ``Bearer …``).
_KEYISH: Final = re.compile(r"(?i)(bearer\s+\S+|\b(?:sk|pk|rk|key)-[A-Za-z0-9_\-*.]{4,})")
_MAX_DETAIL: Final = 300

#: Factory for the transport that actually opens sockets (tests swap in a mock).
InnerTransportFactory = Callable[[], httpx2.AsyncBaseTransport]


def _default_inner_transport() -> httpx2.AsyncBaseTransport:
    return httpx2.AsyncHTTPTransport()


inner_transport_factory: InnerTransportFactory = _default_inner_transport


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """The WM ``message()`` contract, as the in-process path consumes it."""

    worker_name: str
    message: str | None = None
    content_blocks: Sequence[Mapping[str, Any]] | None = None
    system_prompt: str | None = None
    output_schema: Mapping[str, Any] | None = None
    max_tokens: int = 8_000
    model_override: str | None = None
    work_tier: str | None = None
    enable_reasoning: bool | None = None
    context: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class LLMResult:
    response: WMResponse
    input_tokens: int
    output_tokens: int
    model: str
    provider_kind: str
    overrides_ignored: Mapping[str, object] = field(default_factory=dict)


# ── egress ──────────────────────────────────────────────────────────────────


class PinnedTransport(httpx2.AsyncBaseTransport):
    """Send every request to ONE origin, connecting to the resolver's pinned IP.

    ``pinned_ip=None`` means the host was allow-listed by policy (or is the provider's
    fixed default endpoint): the origin is still enforced, DNS is not pinned.
    """

    def __init__(
        self, base_url: str, pinned_ip: str | None, inner: httpx2.AsyncBaseTransport
    ) -> None:
        parts = urlsplit(base_url)
        if not parts.hostname or parts.scheme not in {"http", "https"}:
            raise SlotUnavailableError(SLOT, "the provider endpoint is not a valid URL")
        self._scheme = parts.scheme
        self._host = parts.hostname.lower().rstrip(".")
        self._port = parts.port or (443 if parts.scheme == "https" else 80)
        self._ip = pinned_ip
        self._inner = inner

    def _same_origin(self, url: httpx2.URL) -> bool:
        port = url.port or (443 if url.scheme == "https" else 80)
        return (
            url.scheme == self._scheme
            and (url.host or "").lower().rstrip(".") == self._host
            and port == self._port
        )

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        if not self._same_origin(request.url):
            raise httpx2.ConnectError(
                "refused: request left the tenant provider's origin", request=request
            )
        if self._ip is not None:
            # Host header was set from the ORIGINAL url when the request was built;
            # only the connect target changes. SNI keeps cert verification on the name.
            request.url = request.url.copy_with(host=self._ip)
            request.extensions = {**request.extensions, "sni_hostname": self._host}
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


#: Patched by tests; the retry sleeps go through here.
_sleep: Callable[[float], Any] = asyncio.sleep


def _retry_sleep_seconds(response: httpx2.Response, attempt: int) -> float:
    """``BaseServiceClient``'s ``_retry_sleep_seconds`` for an ``httpx2`` response.

    Linear backoff from the per-status budget; a 429's ``Retry-After`` (capped at
    ``_RETRY_AFTER_MAX_DELAY``) may only make the wait LONGER, never shorter.
    """
    _, base_delay = _wm_retry._retry_budget(response.status_code)
    default_delay = base_delay * attempt
    if response.status_code in _wm_retry._RETRY_AFTER_STATUS_CODES:
        honoured = _wm_retry._parse_retry_after(response.headers.get("Retry-After"))
        if honoured is not None:
            return max(honoured, default_delay)
    return default_delay


class RetryTransport(httpx2.AsyncBaseTransport):
    """The WM path's retry policy (``BaseServiceClient._request``) for the tenant call.

    The WM path's ``message()`` sends an idempotency key, so ``_request`` retries it:
    429 up to 6 attempts (linear 5s, ``Retry-After`` honoured, capped), 500/502/503/504
    up to 3 attempts (linear 1s), connect errors up to 3 attempts (linear 1s),
    all inside one 75s cumulative sleep budget.

    Which transport errors are re-sent is an explicit per-construction choice,
    ``retry_read_timeout``:

    * ``False`` — TENANT providers (paid, non-idempotent). Only ``ConnectError``/
      ``ConnectTimeout`` are retried: the request provably never reached the
      provider. A ``ReadTimeout``/``WriteTimeout`` means the POST may already be
      executing upstream, and this call carries no idempotency key the provider
      honours, so re-sending it can run (and bill) the completion twice; those
      propagate on the first failure.
    * ``True`` — the PLATFORM's own unmetered local vLLM (``vllm_direct``). Every
      ``TimeoutException`` is retried too, as the WM path does: a duplicate there
      costs GPU time, not a tenant's money, and a lost retry fails a stage.

    The same constants are read here, at call time, from ``w2v_common.clients.base`` so the two paths cannot drift. Any
    other status — 4xx auth/validation included — is returned untouched (no retry).
    """

    def __init__(self, inner: httpx2.AsyncBaseTransport, *, retry_read_timeout: bool) -> None:
        self._inner = inner
        self._retryable_errors: tuple[type[Exception], ...] = (
            (httpx2.ConnectError, httpx2.TimeoutException)
            if retry_read_timeout
            else (httpx2.ConnectError, httpx2.ConnectTimeout)
        )

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        slept = 0.0
        attempt = 1
        while True:
            try:
                response = await self._inner.handle_async_request(request)
            except self._retryable_errors as exc:
                delay = _wm_retry._RETRY_DELAY * attempt
                if (
                    attempt >= _wm_retry._MAX_RETRIES
                    or slept + delay > _wm_retry._RETRY_SLEEP_BUDGET
                ):
                    raise
                slept += delay
                log.warning(
                    "LLM provider transport error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt,
                    _wm_retry._MAX_RETRIES,
                    delay,
                    type(exc).__name__,
                )
                await _sleep(delay)
                attempt += 1
                continue
            status = response.status_code
            max_attempts, _ = _wm_retry._retry_budget(status)
            if status not in _wm_retry._RETRYABLE_STATUS_CODES or attempt >= max_attempts:
                return response
            delay = _retry_sleep_seconds(response, attempt)
            if slept + delay > _wm_retry._RETRY_SLEEP_BUDGET:
                return response
            slept += delay
            await response.aclose()
            log.warning(
                "tenant LLM provider returned %d (attempt %d/%d), retrying in %.1fs",
                status,
                attempt,
                max_attempts,
                delay,
            )
            await _sleep(delay)
            attempt += 1

    async def aclose(self) -> None:
        await self._inner.aclose()


async def _strip_authorization(request: httpx2.Request) -> None:
    """Request hook for a keyless connection: no ``Authorization`` reaches the wire."""
    request.headers.pop("Authorization", None)


def _http_client(
    base_url: str,
    pinned_ip: str | None,
    timeout: float,
    *,
    retry_read_timeout: bool,
    keyless: bool = False,
) -> httpx2.AsyncClient:
    """``retry_read_timeout``: ``True`` only for the platform's own unmetered vLLM,
    ``False`` for every tenant (paid) provider — see :class:`RetryTransport`."""
    return httpx2.AsyncClient(
        # Origin enforcement OUTSIDE the retries: a refused origin is never retried.
        transport=PinnedTransport(
            base_url,
            pinned_ip,
            RetryTransport(inner_transport_factory(), retry_read_timeout=retry_read_timeout),
        ),
        timeout=httpx2.Timeout(timeout, connect=min(10.0, timeout)),
        follow_redirects=False,
        event_hooks={"request": [_strip_authorization]} if keyless else None,
    )


# ── model construction ──────────────────────────────────────────────────────

_ANTHROPIC_DEFAULT_BASE_URL: Final = "https://api.anthropic.com"


def _secret(resolved: ResolvedSlot) -> str:
    if resolved.secret is None or not resolved.secret.get_secret_value():
        raise SlotUnavailableError(SLOT, "the connection has no API key")
    return resolved.secret.get_secret_value()


#: Placeholder ``api_key`` for a keyless (self-hosted) openai-compatible endpoint.
#: The SDK (openai 3.x) refuses an empty key at construction AND refuses a request
#: with no ``Authorization`` unless omitted per call (which pydantic-ai cannot do);
#: ``api_key=None`` is worse — it reads ``OPENAI_API_KEY`` from the process env, i.e.
#: would send a PLATFORM key to a tenant's endpoint. So the client gets this
#: placeholder and ``_strip_authorization`` removes the header before it is sent.
_KEYLESS_PLACEHOLDER: Final = "keyless"  # pragma: allowlist secret


def build_model(resolved: ResolvedSlot, *, timeout: float) -> tuple[Model, httpx2.AsyncClient]:
    """The pydantic-ai model for ``resolved`` plus the client the caller must close."""
    if not resolved.model:
        raise SlotUnavailableError(SLOT, "the connection has no model configured")
    if resolved.kind == KIND_OPENAI_COMPATIBLE:
        if not resolved.base_url:
            raise SlotUnavailableError(SLOT, "kind 'openai-compatible' requires a base URL")
        # The registry declares the key OPTIONAL for openai-compatible (llm + vlm):
        # a self-hosted vLLM without auth is a valid connection, so no key means
        # "send no Authorization header", never a refusal.
        has_key = resolved.secret is not None and bool(resolved.secret.get_secret_value())
        http = _http_client(
            resolved.base_url,
            resolved.pinned_ip,
            timeout,
            retry_read_timeout=False,  # tenant provider: paid, non-idempotent
            keyless=not has_key,
        )
        # max_retries=0: retries live in ``RetryTransport`` (the WM path's
        # BaseServiceClient policy — 429 w/ Retry-After, 5xx, connect/timeouts), so the
        # SDK's own retry loop must not multiply the attempts on top of it.
        client = AsyncOpenAI(
            base_url=resolved.base_url,
            api_key=_secret(resolved) if has_key else _KEYLESS_PLACEHOLDER,
            http_client=http,
            max_retries=0,
        )
        # OpenAI itself takes ``max_completion_tokens``; most OpenAI-COMPATIBLE servers
        # (vLLM, Ollama, OpenRouter, LM Studio) only honour the older ``max_tokens``.
        is_openai = (urlsplit(resolved.base_url).hostname or "").lower() == "api.openai.com"
        model: Model = OpenAIChatModel(
            resolved.model,
            provider=OpenAIProvider(openai_client=client),
            profile=None
            if is_openai
            else OpenAIModelProfile(openai_chat_supports_max_completion_tokens=False),
        )
        return model, http
    if resolved.kind == KIND_ANTHROPIC:
        key = _secret(resolved)
        base_url = resolved.base_url or _ANTHROPIC_DEFAULT_BASE_URL
        pinned = resolved.pinned_ip if resolved.base_url else None
        http = _http_client(base_url, pinned, timeout, retry_read_timeout=False)  # tenant, paid
        # max_retries=0: see the openai-compatible branch — RetryTransport owns retries.
        client = AsyncAnthropic(base_url=base_url, api_key=key, http_client=http, max_retries=0)
        return AnthropicModel(
            resolved.model, provider=AnthropicProvider(anthropic_client=client)
        ), http
    raise SlotUnavailableError(SLOT, f"kind {resolved.kind!r} cannot run in-process")


# ── request mapping ─────────────────────────────────────────────────────────

_JSON_TO_PY: Final[dict[str, type]] = {
    "boolean": bool,
    "integer": int,
    "number": float,
    "string": str,
}
_LIST_OF: Final[dict[str, object]] = {
    "boolean": list[bool],
    "integer": list[int],
    "number": list[float],
    "string": list[str],
}


def output_type_for(schema: Mapping[str, object] | None) -> object:
    """WM's ``_output_type_from_schema`` (tool-output mode, the non-vLLM branch)."""
    if not schema:
        return str
    raw_type = schema.get("type")
    if isinstance(raw_type, str) and raw_type in _JSON_TO_PY:
        return _JSON_TO_PY[raw_type]
    if raw_type == "array":
        items = schema.get("items")
        item_name = items.get("type") if isinstance(items, Mapping) else None
        return _LIST_OF.get(item_name, list[Any]) if isinstance(item_name, str) else list[Any]
    return StructuredDict(dict(schema), name="OutputData", description="Structured output")


def schema_violations(schema: Mapping[str, Any] | None, output: object) -> list[str]:
    """The contract checks WM's post-hoc validator applies that tool mode does not.

    ``required`` keys of an object root and ``enum`` / ``const`` of the root. Kept
    deliberately small: the provider's tool/JSON mode already enforces the shape.
    """
    if not schema:
        return []
    problems: list[str] = []
    if "enum" in schema and output not in schema["enum"]:
        problems.append(f"value {output!r} is not one of {schema['enum']!r}")
    if "const" in schema and output != schema["const"]:
        problems.append(f"value {output!r} is not {schema['const']!r}")
    required = schema.get("required")
    if isinstance(output, Mapping) and isinstance(required, list):
        missing = [k for k in required if isinstance(k, str) and k not in output]
        if missing:
            problems.append(f"missing required keys: {', '.join(missing)}")
    return problems


def user_content(request: LLMRequest) -> str | list[str | BinaryContent]:
    """``message`` as-is, or ``content_blocks`` as multimodal parts (WM block shape)."""
    if request.message is not None and request.content_blocks is not None:
        raise ValueError("Pass exactly one of 'message' or 'content_blocks', not both")
    if request.message is not None:
        return request.message
    if request.content_blocks is None:
        raise ValueError("Either 'message' or 'content_blocks' must be provided")
    parts: list[str | BinaryContent] = []
    for block in request.content_blocks:
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind == "image":
            try:
                data = base64.b64decode(str(block.get("data", "")), validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("image content block is not valid base64") from exc
            media_type = str(block.get("media_type") or "image/jpeg")
            parts.append(BinaryContent(data=data, media_type=media_type))
        else:
            raise ValueError(f"unsupported content block type {kind!r}")
    return parts


def instructions_for(request: LLMRequest) -> list[str]:
    """Seed prompt (what WM's worker row carries) then the caller's per-call prompt."""
    seed = get_worker_defaults(request.worker_name).system_prompt
    return [p for p in (seed, request.system_prompt) if p]


def output_type_label(output: object, schema: Mapping[str, Any] | None) -> str:
    """WM's ``output_type`` derivation (``message_endpoints.py``)."""
    if isinstance(output, bool):
        return "boolean"
    if isinstance(output, str):
        return "enum" if schema and "enum" in schema else "string"
    if isinstance(output, dict):
        return "object"
    if isinstance(output, list):
        return "array"
    return "other"


# ── error mapping ───────────────────────────────────────────────────────────


def sanitise(text: str, secret: str | None) -> str:
    """Provider text safe for users and logs: no key, nothing key-shaped, bounded."""
    if secret:
        text = text.replace(secret, "<redacted>")
    text = _KEYISH.sub("<redacted>", text)
    text = " ".join(text.split())
    return text[:_MAX_DETAIL] + ("…" if len(text) > _MAX_DETAIL else "")


def _stamp[E: TransientUpstreamError | PermanentUpstreamError](err: E, fault: FailureClass) -> E:
    err.failure_class = fault
    return err


def map_error(exc: BaseException, *, kind: str, model: str, secret: str | None) -> Exception:
    """Translate a pydantic-ai / transport failure into the orchestrator's taxonomy."""
    where = f"Tenant LLM provider ({kind}, model {model})"
    if isinstance(exc, ModelHTTPError):
        status = exc.status_code
        if status in _AUTH_HTTP:
            return _stamp(
                UpstreamAuthError(
                    f"{where} rejected the credentials (HTTP {status}). "
                    "Update the API key on the Providers page.",
                    scope="tenant",
                ),
                CUSTOMER_INPUT,
            )
        detail = sanitise(str(exc.body or exc.message or ""), secret)
        if status in _TRANSIENT_HTTP:
            if any(m in detail.lower() for m in _TOKEN_LIMIT_MARKERS):
                return _stamp(
                    PermanentUpstreamError(f"{where} failed (HTTP {status}): {detail}"),
                    CUSTOMER_INPUT,
                )
            return _stamp(
                TransientUpstreamError(f"{where} failed (HTTP {status}): {detail}"), EXTERNAL
            )
        return _stamp(
            PermanentUpstreamError(f"{where} rejected the request (HTTP {status}): {detail}"),
            CUSTOMER_INPUT,
        )
    if isinstance(exc, UsageLimitExceeded | ContentFilterError):
        return _stamp(
            PermanentUpstreamError(f"{where}: {sanitise(str(exc), secret)}"), CUSTOMER_INPUT
        )
    if isinstance(exc, UnexpectedModelBehavior):
        detail = sanitise(str(exc), secret)
        # A bad DRAW, not a bad request — both "Exceeded maximum output retries" (the
        # verdict ``upstream.classify`` gives WM's 422 for it) and pydantic-ai's
        # ``finish_reason=length`` truncation ("Model token limit (N) exceeded before
        # any response was generated" / ``IncompleteToolCall``). The latter used to be
        # PERMANENT/CUSTOMER_INPUT here; #1126 made the platform path
        # (``vllm_direct._complete_once``) retryable/EXTERNAL and BYOK now matches it.
        return _stamp(TransientUpstreamError(f"{where}: {detail}"), EXTERNAL)
    if isinstance(exc, ModelAPIError | httpx2.TransportError | TimeoutError):
        return _stamp(
            TransientUpstreamError(
                f"{where} unreachable: {type(exc).__name__}: {sanitise(str(exc), secret)}"
            ),
            EXTERNAL,
        )
    if isinstance(exc, UserError):
        return _stamp(PermanentUpstreamError(f"{where}: {sanitise(str(exc), secret)}"), OUR_DEFECT)
    return exc if isinstance(exc, Exception) else RuntimeError(str(exc))


# ── the call ────────────────────────────────────────────────────────────────


def _ignored_overrides(request: LLMRequest, resolved: ResolvedSlot) -> dict[str, object]:
    """WM routing knobs a tenant connection does not honour (its model wins); recorded."""
    overrides: dict[str, object] = {}
    if request.model_override and request.model_override != resolved.model:
        overrides["model_override"] = request.model_override
    if request.work_tier:
        overrides["work_tier"] = request.work_tier
    if request.enable_reasoning is not None:
        overrides["enable_reasoning"] = request.enable_reasoning
    return overrides


def _agent(model: Model, request: LLMRequest, schema: JsonSchema | None) -> Agent[None, Any]:
    agent: Agent[None, Any] = Agent(
        model,
        # A runtime-built spec (scalar type / list[...] / StructuredDict) — opaque to
        # the type checker by construction, exactly as in WM's agent builder.
        output_type=cast("Any", output_type_for(schema)),
        instructions=instructions_for(request) or None,
        retries=OUTPUT_RETRIES,
    )

    @agent.output_validator
    def _contract(output: object) -> object:
        problems = schema_violations(schema, output)
        if problems:
            raise ModelRetry("Output does not match the schema: " + "; ".join(problems))
        return output

    return agent


def _result(
    output: object,
    usage: RunUsage,
    *,
    resolved: ResolvedSlot,
    request: LLMRequest,
    schema: JsonSchema | None,
    overrides: dict[str, object],
) -> LLMResult:
    model_name = resolved.model or "?"
    in_tok = int(usage.input_tokens or 0)
    out_tok = int(usage.output_tokens or 0)
    metadata: dict[str, object] = {
        "source": "tenant",
        "slot": resolved.ref.slot,
        "provider_kind": resolved.kind,
        "served_model": model_name,
        "connection_id": str(resolved.ref.connection_id),
        "connection_version": resolved.ref.version,
        "worker_name": request.worker_name,
        "seed_prompt_applied": get_worker_defaults(request.worker_name).system_prompt is not None,
    }
    if overrides:
        metadata["overrides_ignored"] = overrides
    if request.context is not None:
        metadata["context"] = dict(request.context)
    response = WMResponse(
        output=output,
        output_type=output_type_label(output, schema),
        output_schema_used=schema,
        metadata=metadata,
        usage_metrics={
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            "requests": usage.requests,
            "provider": resolved.kind,
            "model": model_name,
            "funding": "tenant_byo",
        },
    )
    return LLMResult(
        response=response,
        input_tokens=in_tok,
        output_tokens=out_tok,
        model=model_name,
        provider_kind=resolved.kind,
        overrides_ignored=overrides,
    )


async def call_tenant_llm(
    resolved: ResolvedSlot, request: LLMRequest, *, timeout: float
) -> LLMResult:
    """Run ``request`` against the tenant connection ``resolved``; WM-shaped result."""
    if resolved.source != "tenant":
        raise ValueError("call_tenant_llm needs a tenant slot")
    secret = resolved.secret.get_secret_value() if resolved.secret else None
    model_name = resolved.model or "?"
    try:
        prompt = user_content(request)
    except ValueError as exc:
        # A caller bug, deterministic: WM answers the same malformed body with 422.
        raise _stamp(PermanentUpstreamError(str(exc)), OUR_DEFECT) from None

    schema = dict(request.output_schema) if request.output_schema else None
    model, http = build_model(resolved, timeout=timeout)
    agent = _agent(model, request, schema)
    settings = ModelSettings(
        temperature=get_worker_defaults(request.worker_name).temperature,
        max_tokens=request.max_tokens,
        timeout=timeout,
    )
    try:
        async with http:
            run = await agent.run(prompt, model_settings=settings)
    except TransientUpstreamError, PermanentUpstreamError:
        raise
    except Exception as exc:
        mapped = map_error(exc, kind=resolved.kind, model=model_name, secret=secret)
        if mapped is exc:
            raise
        log.warning(
            "in-process LLM call failed worker=%s kind=%s model=%s error=%s",
            request.worker_name,
            resolved.kind,
            model_name,
            mapped,
        )
        raise mapped from None
    return _result(
        run.output,
        run.usage,
        resolved=resolved,
        request=request,
        schema=schema,
        overrides=_ignored_overrides(request, resolved),
    )
