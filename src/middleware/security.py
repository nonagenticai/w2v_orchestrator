"""Frame policy for the few responses the SPA legitimately embeds.

The LIVE security-header middleware is ``w2v_common.middleware.SecurityHeadersMiddleware``,
installed by ``create_app``. It stamps ``X-Frame-Options: DENY`` on EVERY response and
it OVERWRITES (``response.headers[...] = ...``), so a route cannot opt out by setting
its own header — the canonical middleware runs after the handler and clobbers it.

That broke the redesign preview: the Studio renders
``GET /api/redesign/jobs/{id}/html`` (and the per-variant html) in a same-origin
``<iframe sandbox="allow-scripts">``, and the browser refused to paint a ``DENY``
document in a frame. This module used to hold a byte-for-byte local copy of the
canonical middleware that nothing mounted (only a unit test imported it), which made
the file look like the place the header came from.

:class:`FramePolicyMiddleware` is a pure-ASGI wrapper registered OUTER of the
canonical stack (``src/asgi.py`` ``extra_middleware``), so it sees the response start
AFTER ``DENY`` was stamped and relaxes it to ``SAMEORIGIN`` +
``Content-Security-Policy: frame-ancestors 'self'`` for exactly the paths in
:data:`FRAMEABLE_PATH_PATTERN`. Every other response keeps ``DENY``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: The ONLY responses allowed to render inside a frame — the redesign html previews.
#: Anchored both ends; one path segment per id so ``/html/../x`` style paths never match.
#: Pinned against the redesign router's real route paths by
#: ``tests/unit/test_frame_policy_middleware.py`` so a renamed route cannot silently
#: fall back to DENY (or a new route silently become frameable).
FRAMEABLE_PATH_PATTERN: Final = re.compile(r"^/api/redesign/jobs/[^/]+(?:/variants/[^/]+)?/html$")

FRAME_OPTIONS_SAMEORIGIN: Final = b"SAMEORIGIN"
FRAME_ANCESTORS_SELF: Final = b"frame-ancestors 'self'"


def is_frameable_path(path: str) -> bool:
    return FRAMEABLE_PATH_PATTERN.match(path) is not None


class FramePolicyMiddleware:
    """Relax ``X-Frame-Options`` to same-origin for :data:`FRAMEABLE_PATH_PATTERN` GETs."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") not in {"GET", "HEAD"}
            or not is_frameable_path(scope.get("path", ""))
        ):
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() not in {b"x-frame-options", b"content-security-policy"}
                ]
                headers.append((b"x-frame-options", FRAME_OPTIONS_SAMEORIGIN))
                headers.append((b"content-security-policy", FRAME_ANCESTORS_SELF))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)
