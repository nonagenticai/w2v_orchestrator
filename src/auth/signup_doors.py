"""The two early-access service doors bypass ``AuthMiddleware`` — and nothing else.

``POST /api/signup-approvals/intake`` and
``POST /api/signup-approvals/{id}/telegram-decision`` are called with a Keycloak
SERVICE-ACCOUNT token (early-access-intake / chat-bridge). ``AuthMiddleware``
would refuse it (its audience is pinned to ``w2v-backend``), and it cannot admit a
templated path (``extra_public_endpoints`` is exact-match only). So these two
paths — POST only, exact shape — skip it, and the routes verify the token
themselves: pinned issuer, ``exp``/``iss``/``sub`` required, ``azp`` allow-list,
``aud`` contains ``azp``, the door's realm role, never ``super_admin``
(``src/services/signup/service_token.py``). An unauthenticated caller therefore
reaches the route and is answered 401 there; every other path is untouched.

Because the doors are public (ingress) and FastAPI reads + parses the JSON body
BEFORE the route's own token check, their bodies are capped at
:data:`SIGNUP_DOOR_BODY_LIMIT_BYTES` (16 KiB — the largest real intake body, a
2000-character message plus a Turnstile token, is a few KiB) by
:class:`SignupDoorBodyLimitMiddleware`, OUTER of Auth: a declared
``Content-Length`` over it is 413'd before a byte is read, and a streamed /
chunked body stops being read at the cap (413), so an anonymous caller can never
make the process buffer or parse more than that.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from src.auth.middleware import AuthMiddleware
from src.middleware.proxy_body_limit import ProxyBodyLimitMiddleware

logger = logging.getLogger(__name__)

#: ``/api/signup-approvals/intake`` and ``/api/signup-approvals/<one segment>/telegram-decision``.
_DOOR = re.compile(r"^/api/signup-approvals/(?:intake|[^/]+/telegram-decision)/?$")
#: Every path under it is a service door (the super_admin routes live under
#: ``/api/admin/signup-approvals``).
SIGNUP_DOOR_PREFIX = "/api/signup-approvals/"
SIGNUP_DOOR_BODY_LIMIT_BYTES = 16 * 1024


class SignupDoorBodyLimitMiddleware(ProxyBodyLimitMiddleware):
    """The streaming body cap (Content-Length + counted bytes → 413), scoped to
    the signup service doors at 16 KiB, no large-body slots."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(
            app,
            limit=SIGNUP_DOOR_BODY_LIMIT_BYTES,
            prefixes=(SIGNUP_DOOR_PREFIX,),
            overrides={},
            large_body_slots=0,
        )


def is_signup_service_door(method: str, path: str) -> bool:
    return method == "POST" and _DOOR.fullmatch(path) is not None


class SignupDoorAwareAuthMiddleware(AuthMiddleware):
    """``AuthMiddleware`` that hands the two signup service doors straight to the
    route (which verifies its own service token)."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if is_signup_service_door(request.method, request.url.path):
            return await call_next(request)
        return await super().dispatch(request, call_next)


def install_signup_service_doors(app: Starlette) -> None:
    """Swap the registered ``AuthMiddleware`` for the door-aware subclass (same
    position, same arguments). Must run before the first request (Starlette
    builds the stack lazily)."""
    for index, entry in enumerate(app.user_middleware):
        if entry.cls is AuthMiddleware:
            app.user_middleware[index] = Middleware(
                SignupDoorAwareAuthMiddleware, *entry.args, **entry.kwargs
            )
            return
    logger.warning("AuthMiddleware not registered; signup service doors not installed")
