"""Graceful-shutdown middleware.

``app.state.shutting_down = True`` is owned by the lifespan handler in
``asgi.py``. This middleware short-circuits *new* mutating requests once the
flag is set so they fail fast with a retriable 503 + ``Retry-After`` header
during the drain window, instead of getting cut at SIGKILL when k8s decides
the pod has used up ``terminationGracePeriodSeconds``.

GETs and HEADs are allowed through during drain — the frontend retries them
idempotently and the operational cost of a half-served read is bounded. Only
POST / PATCH / PUT / DELETE return 503 during drain.

⚠️ **Under uvicorn this middleware is effectively UNREACHABLE in production.**
Verified 2026-08-23 against ``uvicorn/server.py::Server.shutdown``: on SIGTERM
uvicorn (1) closes the listening sockets, (2) asks every open connection to
shut down, (3) ``await _wait_tasks_to_complete()`` — blocking until every
in-flight connection AND request task is done — and only THEN (4) sends the
lifespan shutdown event, which is where ``asgi.py`` sets ``shutting_down``.
So by the time the flag flips, no new request can arrive (listeners closed)
and none is in flight (drained). The 503 arm can essentially never fire; the
same is true of the ``shutting_down`` guard in ``middleware/audit.py::_persist``.

It is kept, not deleted, because it is CORRECT and becomes live the moment the
flag is set EARLIER than the lifespan — which is a deployment change, not a code
change: a k8s ``preStop`` hook (or a SIGTERM handler) that flips the flag before
uvicorn begins its drain, so the pod sheds writes while the endpoints controller
removes it from the Service. That hook lives in ``gitops-production``, not this
repo. Until it exists, do NOT read a green deploy as evidence that this drain
behaviour was exercised — nothing exercises it but the unit tests, which set the
flag by hand.

The lifespan handler installs the flag, optionally waits a bounded window
for in-flight dispatch tasks to drain, then proceeds with the existing
``dispatcher.shutdown()`` + DB/Redis close path.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_DRAINABLE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class ShutdownMiddleware(BaseHTTPMiddleware):
    """Refuse new mutating requests once the lifespan flips ``shutting_down``."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Return 503 + Retry-After during drain for write methods; pass otherwise."""
        if request.method in _DRAINABLE_METHODS and getattr(
            request.app.state, "shutting_down", False
        ):
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Service is draining for shutdown; retry shortly.",
                    "code": "shutting_down",
                },
                headers={"Retry-After": "5"},
            )
        return await call_next(request)
