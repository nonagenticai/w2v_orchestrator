"""The orchestrator's ``WMClient``: platform backend or in-process tenant slot.

Every LLM call site in this service constructs ``WMClient`` and awaits
``.message(...)``. This subclass is the single injection point (ruff ``TID251``
bans importing ``w2v_common.clients.WMClient`` anywhere else in ``src/``):

* the bound run pinned NO ``llm`` slot (unbound context, legacy run, API request)
  or pinned the deployment kind (``nonagentic-wm``) → the PLATFORM default, whose
  backend is ``LLM_BACKEND`` (:class:`src.config.PlatformLLMSettings`):

  - ``ir`` (default) → inference-router's ``/v1/chat/completions`` with a static
    ``inference-router`` licence, delegating the run's tenant
    (:mod:`src.services.llm.ir_direct`), same :class:`WMResponse`. IR meters it;
    the billing mirror brings the usage into ``run_usage``.
  - ``vllm`` → the pgx vLLM directly with a static licence token
    (:mod:`src.services.llm.vllm_direct`), same :class:`WMResponse`. The kind name
    ``nonagentic-wm`` deliberately keeps meaning "the platform default, whatever
    env says", so runs pinned before the flip resume on the new backend.

  The worker_management backend is gone (``w2v_common`` 0.109.0 deleted
  ``POST /workers/{name}/message``): ``super().message`` is never called. The
  inherited WM pool (``_get_client``) survives only for the tenant-config mirror.
* the run pinned a TENANT ``llm`` connection → that exact version is resolved
  (secret decrypted, egress re-checked, IP pinned) and the call runs in-process via
  pydantic-ai (:mod:`src.services.llm.in_process`), returning the same
  :class:`WMResponse`. Usage is written to ``run_usage`` as ``tenant_byo``.
* the call carries an IMAGE content block and the run pinned a TENANT ``vlm``
  connection → it runs in-process on THAT connection (``openai-compatible``: Chat
  Completions with ``image_url`` parts), ahead of the ``llm`` rules above. A
  text-only call never touches the ``vlm`` slot.
"""

from __future__ import annotations

import uuid
from typing import Literal

from w2v_common.auth.identity_context import get_tenant_id_from_context
from w2v_common.clients import WMClient as _WMClient  # noqa: TID251 - the one wrapper
from w2v_common.clients import WMResponse
from w2v_common.clients.ir_direct import IRNotConfiguredError
from w2v_common.clients.wm_client import WM_MIN_OUTPUT_TOKENS

from src.auth.run_providers import current_run_slot
from src.services.llm import in_process, ir_direct, vllm_direct
from src.services.llm.in_process import LLMRequest
from src.services.llm.metering import record_in_process_usage, record_platform_vllm_usage
from src.services.providers.errors import SlotUnavailableError
from src.services.providers.platform_guard import refuse_platform_for_bound_tenant
from src.services.providers.resolver import (
    PROVIDER_SLOTS_KEY,
    ResolvedSlot,
    SlotRef,
    resolve_slot,
)

__all__ = ["WMClient", "WMResponse", "resolve_pinned_llm"]

SLOT = in_process.SLOT
VLM_SLOT = "vlm"


def _has_image(content_blocks: list[dict[str, object]] | None) -> bool:
    return bool(content_blocks) and any(
        b.get("type") == "image" for b in content_blocks or () if isinstance(b, dict)
    )


async def resolve_pinned_llm(
    tenant_id: uuid.UUID, ref: SlotRef, *, slot: str = SLOT
) -> ResolvedSlot:
    """Load + validate exactly the pinned ``llm`` (or ``vlm``) version for ``tenant_id``."""
    from src.config import get_settings
    from src.database.session import bind_session_tenant, get_db_session
    from src.services.providers.adapters import build_resolver_deps

    async with get_db_session() as session:
        await bind_session_tenant(session, tenant_id)
        deps = build_resolver_deps(session, get_settings())
        return await resolve_slot(
            deps.repo,
            deps.decryptor,
            tenant_id,
            slot,
            run_snapshot={PROVIDER_SLOTS_KEY: {slot: ref.to_dict()}},
            policy=deps.policy,
            posture=deps.posture,
        )


def _bound_tenant() -> uuid.UUID:
    raw = get_tenant_id_from_context()
    try:
        return uuid.UUID(str(raw))
    except ValueError, TypeError:
        raise SlotUnavailableError(
            SLOT, "no tenant is bound to this call, so its provider cannot be used"
        ) from None


def _platform_backend() -> str:
    """``LLM_BACKEND`` for the platform-default slot (``ir`` | ``vllm``)."""
    from src.config import get_settings

    return get_settings().platform_llm.llm_backend


def _request(
    worker_name: str,
    message: str | None,
    *,
    content_blocks: list[dict[str, object]] | None,
    system_prompt: str | None,
    output_schema: dict[str, object] | None,
    max_tokens: int | None,
    model_override: str | None,
    work_tier: str | None,
    enable_reasoning: bool | None,
    context: dict[str, object] | None,
) -> LLMRequest:
    """The in-process request for a ``message()`` call (tenant slot or direct vLLM)."""
    if message is None and content_blocks is None:
        raise ValueError("Either 'message' or 'content_blocks' must be provided")
    if message is not None and content_blocks is not None:
        raise ValueError("Pass exactly one of 'message' or 'content_blocks', not both")
    return LLMRequest(
        worker_name=worker_name,
        message=message,
        content_blocks=content_blocks,
        system_prompt=system_prompt,
        output_schema=output_schema,
        # Same floor as the WM path (WM_MIN_OUTPUT_TOKENS): a reasoning model spends
        # this budget thinking, and the tenant (or the platform vLLM) may run one.
        max_tokens=max(max_tokens or WM_MIN_OUTPUT_TOKENS, WM_MIN_OUTPUT_TOKENS),
        model_override=model_override,
        work_tier=work_tier,
        enable_reasoning=enable_reasoning,
        context=context,
    )


class WMClient(_WMClient):
    """``w2v_common`` ``WMClient`` + per-run ``llm`` slot dispatch (module doc)."""

    async def message(
        self,
        worker_name: str,
        message: str | None = None,
        *,
        content_blocks: list[dict[str, object]] | None = None,
        context: dict[str, object] | None = None,
        system_prompt: str | None = None,
        output_schema: dict[str, object] | None = None,
        model_override: str | None = None,
        max_tokens: int | None = None,
        work_tier: Literal["small", "medium", "high"] | None = None,
        product: str | None = None,
        enable_reasoning: bool | None = None,
        disable_state_storage: bool = True,
        project_id: uuid.UUID | str | None = None,
        slot: Literal["llm", "vlm"] | None = None,
    ) -> WMResponse:
        # WM-era knobs, kept for signature compatibility with the ~44 call sites.
        # ``project_id`` is NOT one: inference-router attributes by it (ir path).
        del product, disable_state_storage
        # ``slot`` (w2v_common 0.113.0) selects the run slot exactly as the base
        # class does: explicit wins, else ``vlm`` iff an image part is present. The
        # base class's own slot resolution (``W2V_PROVIDER_SLOTS=enforce``) is never
        # reached: ``super().message`` is not called, and this service's pinned-slot
        # dispatch (``current_run_slot`` + ``platform_guard``) stays authoritative.
        wants_vlm = slot == VLM_SLOT if slot is not None else _has_image(content_blocks)
        if wants_vlm:
            vlm_ref = current_run_slot(VLM_SLOT)
            if vlm_ref is not None and vlm_ref.source == "tenant":
                request = _request(
                    worker_name,
                    message,
                    content_blocks=content_blocks,
                    system_prompt=system_prompt,
                    output_schema=output_schema,
                    max_tokens=max_tokens,
                    model_override=model_override,
                    work_tier=work_tier,
                    enable_reasoning=enable_reasoning,
                    context=context,
                )
                tenant_id = _bound_tenant()
                resolved = await resolve_pinned_llm(tenant_id, vlm_ref, slot=VLM_SLOT)
                result = await in_process.call_tenant_llm(resolved, request, timeout=self.timeout)
                await record_in_process_usage(
                    tenant_id=tenant_id, worker_name=worker_name, resolved=resolved, result=result
                )
                return result.response
        ref = current_run_slot(SLOT)
        platform = ref is None or ref.source == "deployment"
        if platform:
            # The choke point: a byo_required tenant never reaches the platform LLM,
            # pinned or not (request-context calls, resumed runs). No tenant ⇒ platform.
            await refuse_platform_for_bound_tenant(SLOT)
        backend = _platform_backend() if platform else None
        if backend == ir_direct.BACKEND:
            from src.config import get_settings

            if not get_settings().platform_llm.llm_ir_token.get_secret_value().strip():
                # LLM_BACKEND defaulted to ir with no licence: fail the CALL, loudly,
                # with the w2v_common error — never an unauthenticated request.
                raise IRNotConfiguredError(
                    "platform LLM backend is ir (LLM_BACKEND unset) but LLM_IR_TOKEN is "
                    "not configured; set LLM_IR_TOKEN or LLM_BACKEND=vllm"
                )
            # IR meters the call; its cost event carries X-Correlation-Id (the run)
            # and the billing mirror copies it into run_usage — no local row here.
            return await ir_direct.call_platform_ir(
                get_settings().platform_llm,
                _request(
                    worker_name,
                    message,
                    content_blocks=content_blocks,
                    system_prompt=system_prompt,
                    output_schema=output_schema,
                    max_tokens=max_tokens,
                    model_override=model_override,
                    work_tier=work_tier,
                    enable_reasoning=enable_reasoning,
                    context=context,
                ),
                timeout=self.timeout,
                project_id=project_id,
            )
        if backend == vllm_direct.BACKEND:
            from src.config import get_settings

            response = await vllm_direct.call_platform_vllm(
                get_settings().platform_llm,
                _request(
                    worker_name,
                    message,
                    content_blocks=content_blocks,
                    system_prompt=system_prompt,
                    output_schema=output_schema,
                    max_tokens=max_tokens,
                    model_override=model_override,
                    work_tier=work_tier,
                    enable_reasoning=enable_reasoning,
                    context=context,
                ),
                timeout=self.timeout,
            )
            # vLLM-direct skips WM and billing, so the reconciler never sees this
            # call: the ledger row is written here (funding local_gpu), best-effort.
            await record_platform_vllm_usage(worker_name=worker_name, response=response)
            return response
        if platform or ref is None:
            # Unreachable while PlatformLLMSettings.llm_backend is ir | vllm; a new
            # value must get its own branch above, never fall through to a tenant path.
            raise AssertionError(f"no dispatch branch for platform LLM backend {backend!r}")
        request = _request(
            worker_name,
            message,
            content_blocks=content_blocks,
            system_prompt=system_prompt,
            output_schema=output_schema,
            max_tokens=max_tokens,
            model_override=model_override,
            work_tier=work_tier,
            enable_reasoning=enable_reasoning,
            context=context,
        )
        tenant_id = _bound_tenant()
        resolved = await resolve_pinned_llm(tenant_id, ref)
        result = await in_process.call_tenant_llm(resolved, request, timeout=self.timeout)
        await record_in_process_usage(
            tenant_id=tenant_id, worker_name=worker_name, resolved=resolved, result=result
        )
        return result.response
