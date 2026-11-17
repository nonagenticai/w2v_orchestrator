"""HTTP mapping for the #1106 BYO refusals raised outside a run (e.g. run-start preflight).

* :class:`SlotNotConfiguredError` → **402** ``{"detail": {"code": "byo_provider_required",
  "modality": ..., "message": ...}}`` — the same object ``RunDetail.failure_detail``
  carries, so the frontend handles both with one branch.
* :class:`ByoPostureUnavailableError` → **503** + ``Retry-After`` — licences could not
  be read; retryable, never a platform fallback.

Registered on the production app (``src/asgi.py``) AND the hand-built test apps
(``tests/conftest.py``), like the storyboard/tenancy handlers.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from src.services.providers.errors import ByoPostureUnavailableError, SlotNotConfiguredError

__all__ = ["POSTURE_RETRY_AFTER_SECONDS", "register_byo_exception_handlers"]

POSTURE_RETRY_AFTER_SECONDS = 30


async def _byo_required(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, SlotNotConfiguredError)  # noqa: S101 — registered for this type only
    return JSONResponse(
        status_code=status.HTTP_402_PAYMENT_REQUIRED, content={"detail": exc.failure_detail}
    )


async def _posture_unavailable(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
        headers={"Retry-After": str(POSTURE_RETRY_AFTER_SECONDS)},
    )


def register_byo_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(SlotNotConfiguredError, _byo_required)
    app.add_exception_handler(ByoPostureUnavailableError, _posture_unavailable)
