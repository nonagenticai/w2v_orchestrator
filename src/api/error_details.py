"""Keep a STRUCTURED ``HTTPException`` detail structured on the wire.

The problem
-----------
``w2v_common.asgi.error_envelope`` renders every routed ``HTTPException`` through
``_http_response``, which does::

    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)

and never populates ``details``. So a route that deliberately raises a TYPED body —
``HTTPException(400, detail=refusal.model_dump())``, the shape
``routes/projects.py`` and ``routes/deliverables.py`` use for the website_review
order floor — reaches the client as a Python ``repr`` STRING::

    {"error": {"message": "{'reason': 'order_below_floor', 'ordered_seconds': 9.0, ...}"}}

Every number is there and none of it is parseable. The frontend (PR #448) surfaces
``ApiError.detail`` inline and cannot branch on ``reason`` against a repr.

Why the fix lives here
----------------------
The stringification is in the SHARED library, not at the raise site, and this repo
cannot change that library. ``register_error_envelope_handlers`` is documented as
idempotent — "services can register extra handlers after" — so the orchestrator
re-registers ``StarletteHTTPException`` with a handler that keeps a structured
detail structured. Everything else about the envelope is unchanged: ``code``,
``status``, ``request_id``, ``timestamp``, and a string detail still land in
``message`` exactly as before. Non-orchestrator services keep the library default
until the fix lands upstream in ``w2v_common``.

⚠️ Registration ORDER matters: this must be registered AFTER
``register_error_envelope_handlers`` (which ``create_app`` calls), or the library's
handler replaces it again.

⚠️ KNOWN GAP, deliberately NOT fixed here (recorded so the next reader does not have
to rediscover it): the **ExceptionGroup path is still stringified**. An
``HTTPException`` raised inside a ``TaskGroup`` arrives wrapped, and only the
library's bare-``Exception`` handler unwraps it (``_find_http_exception``) — it then
renders via the library's own ``_http_response``, which this module does not replace.
So a structured detail raised from inside a task group would still reach the wire as a
``repr`` string.

It is unreachable on the three doors this fixes (``create_project``,
``create_project_from_intent``, the deliverable run-start) — none raises the typed
refusal from inside a task group; all three raise directly in the request handler. The
real fix belongs upstream in ``w2v_common`` alongside ``_find_http_exception``, where
one change covers every service. Chasing it here would mean re-implementing the
unwrap in every service that ever wants a typed body.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from w2v_common.asgi.error_envelope import make_error_payload

#: Fallback ``message`` when a structured detail carries no human line of its own.
#: Never a ``repr`` of the body — the body itself is on the wire under ``details``.
_STRUCTURED_FALLBACK = "Request refused; see error.details"


def _request_id(request: Request) -> str:
    """Mirror of the library's request-id resolution (state → header → uuid7)."""
    import uuid as _uuid

    rid = getattr(request.state, "request_id", None)
    if rid:
        return str(rid)
    return request.headers.get("x-request-id") or str(_uuid.uuid7())


def _structured_detail(detail: object) -> dict[str, object] | list[object] | None:
    """The detail when it is a JSON OBJECT/ARRAY worth preserving, else ``None``.

    A ``str`` detail (the overwhelming majority — ``HTTPException(404, "run not
    found")``) is NOT structured and keeps the library's exact behaviour. ``bytes``
    is excluded deliberately: it is a ``Sequence`` but not JSON-serialisable.
    """
    if isinstance(detail, Mapping):
        return dict(detail)
    if isinstance(detail, Sequence) and not isinstance(detail, str | bytes | bytearray):
        return list(detail)
    return None


def _message_for(detail: dict[str, object] | list[object]) -> str:
    """Human line for a structured detail — its own ``message``, never a ``repr``."""
    if isinstance(detail, Mapping):
        text = detail.get("message")
        if isinstance(text, str) and text:
            return text
    return _STRUCTURED_FALLBACK


def register_object_detail_http_handler(app: FastAPI) -> None:
    """Re-register the ``HTTPException`` handler so a dict/list detail stays JSON.

    Call AFTER ``register_error_envelope_handlers(app)``.
    """

    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        if not isinstance(exc, StarletteHTTPException):
            raise TypeError(f"object-detail handler got non-HTTP exception: {type(exc).__name__}")
        structured = _structured_detail(exc.detail)
        message = (
            _message_for(structured)
            if structured is not None
            else (exc.detail if isinstance(exc.detail, str) else str(exc.detail))
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=make_error_payload(
                code=f"HTTP_{exc.status_code}",
                message=message,
                status=exc.status_code,
                request_id=_request_id(request),
                details=structured,
            ),
            headers=dict(exc.headers or {}),
        )

    app.add_exception_handler(StarletteHTTPException, _handler)
