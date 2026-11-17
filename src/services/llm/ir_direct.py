"""Platform-default LLM calls to inference-router (``LLM_BACKEND=ir``).

Owner decision 2026-10-20: w2v's platform LLM traffic goes to inference-router
(IR) directly — no worker_management hop. ONE OpenAI-compatible Chat Completions
call, ``POST {LLM_IR_BASE_URL}/v1/chat/completions``, model ``local-llm``,
authenticated by ONE static licence for app ``inference-router``
(``LLM_IR_TOKEN``) sent as ``X-Tenant-Id: Bearer``. The run's tenant rides as
``X-Effective-Tenant-Id`` (IR's delegation contract), so IR bills the tenant the
work is for.

Self-contained on purpose: ``w2v_common`` 0.109.0 is adding the same backend in
parallel; this copy does not depend on it. Everything except the endpoint and the
headers is the vLLM-direct machinery (:func:`vllm_direct.run_chat_completions`:
seed prompt + temperature, the 16384 ``max_tokens`` ceiling, xgrammar-stripped
``json_schema`` with schema re-asks, error mapping without leaking the token).
The two differences of substance:

* ``enable_reasoning`` is a TOP-LEVEL body field (IR's contract), never
  ``chat_template_kwargs`` — IR ignores the latter.
* IR METERS the call. So a read timeout is NOT re-sent (it may already have been
  billed), ``usage_metrics`` carries no ``funding`` and no local ``run_usage`` row
  is written: the cost event IR emits carries ``X-Correlation-Id`` (the run id) and
  ``X-Stage``, and the billing mirror (``RunUsageReconciler``) copies it into
  ``run_usage`` exactly as it does for the WM path. Writing a row here as well
  would count every token twice.

Headers (IR ``routers/completions.py``): ``X-App-Id: w2v`` (the per-app
accounting key, as ``WMClient`` sends), ``X-Worker-Name`` (worker label),
``X-Project-Id`` (the W2V project, :func:`project_header`), ``X-Job-Id`` = the
run's correlation id (IR forbids a forged default, so it is OMITTED when no run is
bound), ``X-Subject-Id`` when a subject is bound, and the
cost-correlation pair. Delegation follows the ``identity_event_hook`` rule: the
bound tenant must have VOUCHED provenance, otherwise the call is refused before
anything is sent (``resolve_licence_tenant``).
"""

from __future__ import annotations

import uuid
from typing import Final

from w2v_common.auth.identity_context import (
    PROJECT_ID_HEADER,
    get_attribution_subject_id_from_context,
    get_correlation_id_from_context,
    get_stage_from_context,
    get_tenant_id_from_context,
    get_w2v_project_id_from_context,
    parse_attribution_uuid,
    resolve_licence_tenant,
    tenant_level_project_id,
)
from w2v_common.clients import WMResponse
from w2v_common.clients.wm_client import APP_ID_HEADER, DEFAULT_APP_ID, JOB_ID_HEADER

from src.config import PlatformLLMSettings
from src.services.llm.in_process import LLMRequest
from src.services.llm.vllm_direct import ChatEndpoint, run_chat_completions

BACKEND: Final = "ir"
#: ``usage_metrics.provider`` / ``metadata.backend`` for an IR call.
PROVIDER: Final = "inference-router"
CHAT_PATH: Final = "/v1/chat/completions"
EFFECTIVE_TENANT_HEADER: Final = "X-Effective-Tenant-Id"


def project_header(project_id: uuid.UUID | str | None) -> dict[str, str]:
    """``X-Project-Id`` for one IR call: the W2V project of the work.

    The ``w2v_common`` ``WMClient`` rule (``wm_client._project_header``), which the
    deleted WM path applied too: the caller's ``project_id`` (strict canonical
    UUID), else the W2V project bound by the run dispatcher, else the bound
    tenant's tenant-level project, else nothing. Never an fr project.
    """
    if project_id is not None:
        canonical = parse_attribution_uuid(str(project_id))
        if canonical is None:
            raise ValueError(f"project_id is not a canonical non-nil uuid: {project_id!r}")
        return {PROJECT_ID_HEADER: canonical}
    project = get_w2v_project_id_from_context() or tenant_level_project_id(
        get_tenant_id_from_context()
    )
    return {PROJECT_ID_HEADER: project} if project else {}


def ir_headers(
    token: str, worker_name: str, project_id: uuid.UUID | str | None = None
) -> dict[str, str]:
    """The request headers for one IR call, from the bound execution context.

    Raises ``NoTenantForLicenceError`` / ``UnvouchedTenantError`` (``w2v_common``)
    when no tenant, or an unvouched one, is bound — the same refusal the WM path's
    delegation hook makes."""
    headers = {
        "X-Tenant-Id": f"Bearer {token}",
        EFFECTIVE_TENANT_HEADER: resolve_licence_tenant(),
        APP_ID_HEADER: DEFAULT_APP_ID,
        "X-Worker-Name": worker_name,
        **project_header(project_id),
    }
    correlation = get_correlation_id_from_context()
    if correlation:
        headers["X-Correlation-Id"] = correlation
        headers[JOB_ID_HEADER] = correlation
    stage = get_stage_from_context()
    if stage:
        headers["X-Stage"] = stage
    subject = get_attribution_subject_id_from_context()
    if subject:
        headers["X-Subject-Id"] = subject
    return headers


def ir_endpoint(
    settings: PlatformLLMSettings, worker_name: str, project_id: uuid.UUID | str | None = None
) -> ChatEndpoint:
    token = settings.llm_ir_token.get_secret_value()
    base = settings.llm_ir_base_url.rstrip("/")
    return ChatEndpoint(
        backend=PROVIDER,
        base_url=settings.llm_ir_base_url,
        url=base + CHAT_PATH,
        token=token,
        model=settings.llm_ir_model,
        credential="LLM_IR_TOKEN",
        licence_app="inference-router",
        headers=ir_headers(token, worker_name, project_id),
        retry_read_timeout=False,
        reasoning_top_level=True,
        funding=None,
    )


async def call_platform_ir(
    settings: PlatformLLMSettings,
    request: LLMRequest,
    *,
    timeout: float,
    project_id: uuid.UUID | str | None = None,
) -> WMResponse:
    """Run ``request`` on inference-router; the same :class:`WMResponse` WM returns."""
    return await run_chat_completions(
        ir_endpoint(settings, request.worker_name, project_id), request, timeout=timeout
    )


__all__ = [
    "BACKEND",
    "CHAT_PATH",
    "PROVIDER",
    "call_platform_ir",
    "ir_endpoint",
    "ir_headers",
    "project_header",
]
