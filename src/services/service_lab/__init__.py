"""Service lab — the "test this GPU service" bench.

Two entry points:

* :func:`~src.services.service_lab.invoker.invoke` — resolve a provider slot LIVE for
  a tenant (``byo_mode`` honoured through the resolver, never a local ``if``) and make
  ONE real call, returning the produced media inline.
* :func:`~src.services.service_lab.invoker.enhance_prompt` — rewrite a short prompt
  into a better one for a target modality, on the same ``llm`` slot.

The provider layer's own errors (``SlotNotConfiguredError``, ``SlotUnavailableError``,
``OcrNotConfiguredError``, ``GpuBusyError``, upstream errors) propagate untouched — the
route maps them to HTTP. Nothing here invents an error code.

Persistence lives in ``database.repositories.service_run_repository`` (``insert_run`` /
``list_runs`` / ``get_run``); this package never commits.
"""

from __future__ import annotations

from src.services.service_lab.invoker import (
    SLOTS,
    InvokeRequest,
    InvokeResult,
    ProviderInfo,
    enhance_prompt,
    enhance_targets,
    invoke,
)

__all__ = [
    "SLOTS",
    "InvokeRequest",
    "InvokeResult",
    "ProviderInfo",
    "enhance_prompt",
    "enhance_targets",
    "invoke",
]
