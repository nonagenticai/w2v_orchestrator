"""Platform-default LLM calls straight to the pgx vLLM with a static licence token.

Owner decision 2026-10-17: the WM → inference-router hop mints a per-tenant
``workers`` bundle token on every call, and licences answering 409
``ambiguous_bundle`` took every w2v LLM stage down. This backend is the
care-bus-agents shape instead: ONE OpenAI-compatible Chat Completions call to the
vLLM (model ``local-llm``) with a static ``vllm`` licence (``LLM_VLLM_TOKEN``).
Selected by ``LLM_BACKEND=vllm`` (:class:`src.config.PlatformLLMSettings`);
``wm`` stays the default, so rollback is a config flip.

Deliberately LEAN (owner guidance): no pydantic-ai agent, no multi-provider layer,
no port of WM's error-correction loop. What it keeps from the WM path, because the
call sites depend on it:

* the worker's seed prompt + temperature (``worker_defaults.yaml``), then the
  per-call ``system_prompt`` — the same order WM applies;
* the ``WM_MIN_OUTPUT_TOKENS`` floor, clamped to WM's 16384 ceiling for this vLLM;
* ``chat_template_kwargs.enable_thinking = bool(enable_reasoning)`` — what
  inference-router forwarded (thinking OFF unless a caller opts in);
* structured output as vLLM guided decoding (``response_format=json_schema``) over
  the xgrammar-STRIPPED schema, validated (Draft 2020-12) against that same
  stripped schema, re-asked with the violations on failure (``OUTPUT_RETRIES``).
  ``prefixItems`` survives the strip, so ``non_empty_array`` (#1076) stays a
  grammar-level guarantee. Count/length bounds the grammar cannot carry are
  checked against the ORIGINAL schema and reported, never fatal (WM's default);
* the WM path's HTTP retry policy (``in_process.RetryTransport``) and the same
  :class:`WMResponse` shape, so the ~44 call sites cannot tell the difference.

No metering is emitted (billing is out of this lane): usage is returned in
``usage_metrics`` with ``funding="platform_direct"`` and logged.

The same round-trip machinery (:func:`run_chat_completions`) also serves the
``ir`` backend (:mod:`src.services.llm.ir_direct`), which differs only in the
endpoint, the headers, top-level ``enable_reasoning`` and who meters the call.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, cast

import httpx2
from jsonschema import Draft202012Validator
from w2v_common.clients import WMResponse

from src.config import PlatformLLMSettings
from src.models.failure_class import CUSTOMER_INPUT, EXTERNAL, OUR_DEFECT
from src.services.llm import in_process
from src.services.llm.in_process import LLMRequest, output_type_label, sanitise
from src.services.llm.worker_defaults import get_worker_defaults
from src.services.llm.xgrammar_schema import BOUND_KEYWORDS, strip_xgrammar_unsupported
from src.services.stages.upstream import (
    PermanentUpstreamError,
    TransientUpstreamError,
    UpstreamAuthError,
)

log = logging.getLogger(__name__)

BACKEND: Final = "vllm"
#: Re-asks after a reply that is not valid JSON / violates the schema (WM: 3).
OUTPUT_RETRIES: Final = in_process.OUTPUT_RETRIES
#: WM's ``max_tokens`` ceiling (``workers/models.py``) protecting the self-hosted
#: vLLM from the FP4-MoE kernel crash near the full context (incident 2026-05-13).
VLLM_MAX_OUTPUT_TOKENS: Final = 16_384
#: Key a non-object root is wrapped under, so the grammar can carry it.
_WRAP_KEY: Final = "response"
_AUTH_HTTP: Final = frozenset({401, 403})
_TOKEN_LIMIT_MARKERS: Final = ("context length", "maximum context", "token limit", "too long")


@dataclass(frozen=True, slots=True)
class _OutputContract:
    """What the decoder is told, and how to read its answer back."""

    grammar: dict[str, object] | None  # the response_format schema; None ⇒ plain text
    wrapped: bool  # root was not an object: answer lives under ``_WRAP_KEY``


def output_contract(schema: Mapping[str, object] | None) -> _OutputContract:
    """Decoder-visible schema for ``schema`` (xgrammar-stripped, object-rooted)."""
    if not schema:
        return _OutputContract(grammar=None, wrapped=False)
    stripped = cast("dict[str, object]", strip_xgrammar_unsupported(dict(schema)))
    if schema.get("type") == "object" or (
        "type" not in schema and isinstance(schema.get("properties"), Mapping)
    ):
        return _OutputContract(grammar=stripped, wrapped=False)
    return _OutputContract(
        grammar={
            "type": "object",
            "properties": {_WRAP_KEY: stripped},
            "required": [_WRAP_KEY],
            "additionalProperties": False,
        },
        wrapped=True,
    )


def contract_violations(grammar: Mapping[str, object], output: object) -> list[str]:
    """Draft 2020-12 errors of ``output`` against the DECODER-visible schema."""
    validator = Draft202012Validator(dict(grammar))
    problems: list[str] = []
    for err in sorted(validator.iter_errors(output), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(p) for p in err.absolute_path) or "<root>"
        problems.append(f"{where}: {err.message}")
    return problems[:10]


def bound_violations(schema: Mapping[str, object] | None, output: object) -> list[dict[str, str]]:
    """Advisory: count/length bounds of the ORIGINAL schema the grammar could not carry."""
    if not schema:
        return []
    out: list[dict[str, str]] = []
    for err in Draft202012Validator(dict(schema)).iter_errors(output):
        if err.validator in BOUND_KEYWORDS:
            out.append(
                {
                    "path": "/" + "/".join(str(p) for p in err.absolute_path),
                    "keyword": str(err.validator),
                    "message": err.message[:200],
                }
            )
    return out


def _user_content(request: LLMRequest) -> str | list[dict[str, object]]:
    """``message`` as-is, or ``content_blocks`` as OpenAI chat content parts."""
    if request.message is not None and request.content_blocks is not None:
        raise ValueError("Pass exactly one of 'message' or 'content_blocks', not both")
    if request.message is not None:
        return request.message
    if request.content_blocks is None:
        raise ValueError("Either 'message' or 'content_blocks' must be provided")
    parts: list[dict[str, object]] = []
    for block in request.content_blocks:
        kind = block.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": str(block.get("text", ""))})
        elif kind == "image":
            media_type = str(block.get("media_type") or "image/jpeg")
            data = str(block.get("data", ""))
            parts.append(
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
            )
        else:
            raise ValueError(f"unsupported content block type {kind!r}")
    return parts


def build_body(
    request: LLMRequest,
    *,
    model: str,
    contract: _OutputContract,
    reasoning_top_level: bool = False,
) -> dict[str, object]:
    """The Chat Completions body for one attempt (messages appended by the caller).

    ``reasoning_top_level``: inference-router's contract — ``enable_reasoning`` as a
    TOP-LEVEL field, which IR translates for the backend it routes to. The direct
    vLLM takes ``chat_template_kwargs.enable_thinking`` instead."""
    messages: list[dict[str, object]] = []
    instructions = in_process.instructions_for(request)
    if instructions:
        messages.append({"role": "system", "content": "\n\n".join(instructions)})
    messages.append({"role": "user", "content": _user_content(request)})
    defaults = get_worker_defaults(request.worker_name)
    body: dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": defaults.temperature,
        "max_tokens": min(request.max_tokens, VLLM_MAX_OUTPUT_TOKENS),
    }
    if defaults.frequency_penalty is not None:
        # Anti-loop sampling for a worker measured to degenerate (worker_defaults.yaml).
        body["frequency_penalty"] = defaults.frequency_penalty
    # inference-router's contract: thinking OFF unless the caller opts in.
    if reasoning_top_level:
        body["enable_reasoning"] = bool(request.enable_reasoning)
    else:
        body["chat_template_kwargs"] = {"enable_thinking": bool(request.enable_reasoning)}
    if contract.grammar is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "OutputData", "schema": contract.grammar, "strict": False},
        }
    return body


def _where(model: str, backend: str = BACKEND) -> str:
    return f"Platform LLM ({backend}, model {model})"


def _http_error(
    status: int,
    detail: str,
    model: str,
    *,
    backend: str = BACKEND,
    credential: str = "LLM_VLLM_TOKEN",
    licence_app: str = "vllm",
) -> Exception:
    where = _where(model, backend)
    if status in _AUTH_HTTP:
        # UpstreamAuthError (a PermanentUpstreamError): never retried, and it carries
        # ``failure_detail.code == "upstream_auth"`` (scope platform) — the same code
        # the WM path now stamps on a wrapped 401/403, so the FE maps one code.
        err: TransientUpstreamError | PermanentUpstreamError = UpstreamAuthError(
            f"{where} rejected {credential} (HTTP {status}); "
            f"it must be a licence for app `{licence_app}`.",
            scope="platform",
        )
        err.failure_class = OUR_DEFECT
        return err
    if status == 429 or status >= 500:
        err = TransientUpstreamError(f"{where} failed (HTTP {status}): {detail}")
        err.failure_class = EXTERNAL
        return err
    if any(m in detail.lower() for m in _TOKEN_LIMIT_MARKERS):
        err = PermanentUpstreamError(f"{where} rejected the request (HTTP {status}): {detail}")
        err.failure_class = CUSTOMER_INPUT
        return err
    err = PermanentUpstreamError(f"{where} rejected the request (HTTP {status}): {detail}")
    err.failure_class = OUR_DEFECT
    return err


def _parse(content: str, contract: _OutputContract) -> tuple[object, list[str]]:
    """``(output, problems)`` for one completion's ``content``."""
    if contract.grammar is None:
        return content, []
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, [f"reply is not valid JSON ({exc.msg})"]
    problems = contract_violations(contract.grammar, parsed)
    if problems:
        return None, problems
    if contract.wrapped:
        return parsed[_WRAP_KEY], []
    return parsed, []


@dataclass(frozen=True, slots=True)
class ChatEndpoint:
    """Where one platform Chat Completions backend lives and how it is called."""

    backend: str  # ``usage_metrics.provider`` / ``metadata.backend``
    base_url: str  # origin the pinned transport is bound to
    url: str  # full Chat Completions URL
    token: str  # the static credential (redacted from every error)
    model: str
    credential: str  # env name, for the 401/403 message
    licence_app: str  # the licence app that credential must be for
    headers: dict[str, str] = field(default_factory=dict)
    #: Re-send a read timeout — ONLY for an unmetered backend (no double bill).
    retry_read_timeout: bool = True
    #: ``enable_reasoning`` top-level (IR) vs ``chat_template_kwargs`` (vLLM).
    reasoning_top_level: bool = False
    #: ``usage_metrics.funding``; ``None`` ⇒ the key is omitted (someone else meters).
    funding: str | None = "platform_direct"


async def _complete_once(
    http: httpx2.AsyncClient, ep: ChatEndpoint, body: dict[str, object]
) -> tuple[str, int, int]:
    """One Chat Completions round trip: ``(content, prompt_tokens, completion_tokens)``."""
    model = ep.model
    try:
        resp = await http.post(ep.url, json=body, headers=ep.headers)
    except httpx2.TransportError as exc:
        err = TransientUpstreamError(
            f"{_where(model, ep.backend)} unreachable: {type(exc).__name__}: "
            f"{sanitise(str(exc), ep.token)}"
        )
        err.failure_class = EXTERNAL
        raise err from None
    if resp.status_code != 200:
        raise _http_error(
            resp.status_code,
            sanitise(resp.text, ep.token),
            model,
            backend=ep.backend,
            credential=ep.credential,
            licence_app=ep.licence_app,
        )
    data = resp.json()
    usage = data.get("usage") or {}
    choice = (data.get("choices") or [{}])[0]
    if choice.get("finish_reason") == "length":
        # A truncated reply is a bad DRAW of the platform model, not a bad request:
        # live runs 01a158a5-c749 / 01a158a9-5a9b (2026-10-20) died here on a 30s
        # generic script whose reply was ONE unterminated `text` string of ~43k chars
        # (a repetition loop), while the byte-identical request finishes in 115-275
        # tokens ~92% of the time. Retryable, and never the customer's fault — the
        # same verdict a reply that keeps violating the schema gets (EXTERNAL).
        err = TransientUpstreamError(
            f"{_where(model, ep.backend)}: output token limit reached before the reply finished "
            f"(max_tokens={body['max_tokens']}, completion_tokens="
            f"{int(usage.get('completion_tokens') or 0)})"
        )
        err.failure_class = EXTERNAL
        raise err
    content = (choice.get("message") or {}).get("content") or ""
    return content, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def _reask(content: str, problems: list[str]) -> list[dict[str, str]]:
    """The assistant's rejected reply plus the violations, for the next attempt."""
    return [
        {"role": "assistant", "content": content},
        {
            "role": "user",
            "content": "Your reply does not match the required JSON schema: "
            + "; ".join(problems)
            + ". Reply again with ONLY JSON that matches the schema.",
        },
    ]


async def call_platform_vllm(
    settings: PlatformLLMSettings, request: LLMRequest, *, timeout: float
) -> WMResponse:
    """Run ``request`` on the pgx vLLM; the same :class:`WMResponse` WM returns."""
    token = settings.llm_vllm_token.get_secret_value()
    endpoint = ChatEndpoint(
        backend=BACKEND,
        base_url=settings.llm_vllm_base_url,
        url=settings.llm_vllm_base_url.rstrip("/") + "/chat/completions",
        token=token,
        model=settings.model_for(request.worker_name),
        credential="LLM_VLLM_TOKEN",
        licence_app="vllm",
        headers={"Authorization": f"Bearer {token}"},
        # The platform's own unmetered vLLM: a read timeout is re-sent (no tenant bill).
        retry_read_timeout=True,
    )
    return await run_chat_completions(endpoint, request, timeout=timeout)


async def run_chat_completions(
    ep: ChatEndpoint, request: LLMRequest, *, timeout: float
) -> WMResponse:
    """Run ``request`` against ``ep`` (schema re-asks included); a :class:`WMResponse`."""
    model = ep.model
    schema = dict(request.output_schema) if request.output_schema else None
    contract = output_contract(schema)
    try:
        body = build_body(
            request, model=model, contract=contract, reasoning_top_level=ep.reasoning_top_level
        )
    except ValueError as exc:
        err = PermanentUpstreamError(str(exc))
        err.failure_class = OUR_DEFECT
        raise err from None

    in_tok = out_tok = requests = 0
    problems: list[str] = []
    output: object = None
    http = in_process._http_client(
        ep.base_url, None, timeout, retry_read_timeout=ep.retry_read_timeout
    )
    async with http:
        for attempt in range(OUTPUT_RETRIES + 1):
            content, used_in, used_out = await _complete_once(http, ep, body)
            requests += 1
            in_tok += used_in
            out_tok += used_out
            output, problems = _parse(content, contract)
            if not problems:
                break
            log.warning(
                "platform %s reply violates the schema worker=%s attempt=%d/%d: %s",
                ep.backend,
                request.worker_name,
                attempt + 1,
                OUTPUT_RETRIES + 1,
                "; ".join(problems)[:500],
            )
            # A fresh list each attempt: the transport serialised the previous one.
            body["messages"] = [*cast("list[object]", body["messages"]), *_reask(content, problems)]
    if problems:
        # A bad DRAW, not a bad request — the verdict upstream.classify gives WM's 422.
        err = TransientUpstreamError(
            f"{_where(model, ep.backend)}: Exceeded maximum output retries ({OUTPUT_RETRIES}): "
            + "; ".join(problems)[:300]
        )
        err.failure_class = EXTERNAL
        raise err

    usage_metrics: dict[str, object] = {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "requests": requests,
        "provider": ep.backend,
        "model": model,
    }
    if ep.funding is not None:
        usage_metrics["funding"] = ep.funding
    bounds = bound_violations(schema, output)
    if bounds:
        usage_metrics["bound_violations"] = bounds
    metadata: dict[str, object] = {
        "source": "platform",
        "backend": ep.backend,
        "served_model": model,
        "worker_name": request.worker_name,
        "seed_prompt_applied": get_worker_defaults(request.worker_name).system_prompt is not None,
    }
    ignored = {
        k: v
        for k, v in (("model_override", request.model_override), ("work_tier", request.work_tier))
        if v
    }
    if ignored:
        metadata["overrides_ignored"] = ignored
    if request.context is not None:
        metadata["context"] = dict(request.context)
    log.info(
        "platform %s call worker=%s model=%s requests=%d input_tokens=%d output_tokens=%d",
        ep.backend,
        request.worker_name,
        model,
        requests,
        in_tok,
        out_tok,
    )
    return WMResponse(
        output=output,
        output_type=output_type_label(output, schema),
        output_schema_used=schema,
        metadata=metadata,
        usage_metrics=usage_metrics,
    )


__all__ = [
    "BACKEND",
    "VLLM_MAX_OUTPUT_TOKENS",
    "ChatEndpoint",
    "bound_violations",
    "build_body",
    "call_platform_vllm",
    "contract_violations",
    "output_contract",
    "run_chat_completions",
]
