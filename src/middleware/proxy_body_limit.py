"""Request-body cap for the analysis PROXY routes (P9 review, fix 3).

``/api/brand/*``, ``/api/critique*``, ``/api/analyze*`` and ``/api/transcribe``
forward a body to an upstream peer (analysis, video-analyzer, ASR)
for any API-key / licence / browser caller. Nothing else in the app bounds a
request body (uvicorn does not), so a multi-MB body would be buffered here and
re-sent downstream. Most of these routes carry a URL plus a few options —
256 KiB is orders of magnitude above any real payload.

The ONE exception is the critique create (``POST /api/critique``, exact path):
its ``html`` mode (P9-7, merqi marketing) carries a whole rendered page, and
analysis_service scores up to its own ``MAX_HTML_SIZE`` (10 MiB,
``critique_service/helpers.py``). That route's cap is analysis's limit plus a
small JSON envelope allowance, so the proxy never refuses a body analysis would
score — a 256 KiB cap there 413'd real rendered pages into silent null scores.

Pure ASGI (not ``BaseHTTPMiddleware``), registered OUTER of Auth, and it NEVER
buffers — so what an unauthenticated caller can make this process hold is
bounded by what the route stack itself reads, not by the cap:

* a declared ``Content-Length`` over the cap → 413 before a byte is read;
* otherwise ``receive`` is wrapped with a STREAMING byte counter and the app
  consumes the body as it arrives. Auth runs before anything reads the body, so
  an anonymous caller is 401'd with ~0 body bytes read whatever the cap. Once
  an (authenticated) read crosses the cap, the counter stops reading, hands the
  app ``http.disconnect`` instead of the over-cap chunk, and this middleware
  SUBSTITUTES a 413 for whatever the app then answers or raises (FastAPI turns
  a failed body read into a 400 "error parsing the body", and a
  ``BaseHTTPMiddleware`` in between may raise instead — both are replaced). The
  body is read before any response is started, so the substitution is clean;
  if the app had somehow already started a response, it is left alone.

Large-body slots (override routes only): a body that streams past the DEFAULT
cap (256 KiB) on an override route must take one of ``large_body_slots``
process-wide slots — acquired without waiting at the moment the count crosses
256 KiB; none free → 429 + ``Retry-After`` before another byte is read. The slot
is held until the app returns (the upstream call is when the copies are live)
and released in ``finally`` — on success, 413, 429, disconnect or exception.
Small bodies never take a slot. The slot is taken on the STREAMED count, not on
a declared ``Content-Length``: the body is only read after Auth, so an anonymous
caller can never occupy a slot (taking it on the header would let anonymous
large-declared requests exhaust the slots pre-401).

An earlier version pre-read the whole body here (up to the cap) before Auth —
harmless at 256 KiB, but at the 10 MiB critique cap it let an anonymous POST
pin ~20 MiB (chunks + join) per connection pre-401 (P9-7 review blocker).

Scoped by path prefix so upload routes (``/api/uploads*``, music, video ingest),
which have their own, larger, purpose-built limits, are untouched.
"""

from __future__ import annotations

import time
from uuid import uuid7

from fastapi import status
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: 256 KiB — default cap for every proxied analysis route.
PROXY_BODY_LIMIT_BYTES = 256 * 1024
#: Path prefixes whose bodies are proxied verbatim to analysis.
PROXY_BODY_LIMIT_PREFIXES: tuple[str, ...] = (
    "/api/brand/",
    "/api/critique",
    "/api/transcribe",
    "/api/analyze",
    # The service-lab bench takes multipart uploads on three slots. Without this
    # the guard would only run AFTER Starlette has spooled the whole body.
    "/api/services/",
)
#: analysis_service ``CritiqueHelpersMixin.MAX_HTML_SIZE`` (origin/main) — the
#: largest ``html`` analysis scores. Keep in lockstep with analysis.
ANALYSIS_CRITIQUE_MAX_HTML_BYTES = 10 * 1024 * 1024
#: Allowance for the JSON envelope around ``html`` (keys, categories,
#: instructions, auth_info, escaping of quotes/backslashes).
CRITIQUE_JSON_ENVELOPE_BYTES = 64 * 1024
#: Exact-path overrides of the default cap (checked before the prefix default).
#: ``POST /api/transcribe`` (merqi recon's ASR leg): the route's own audio cap
#: (``media_understanding.TRANSCRIBE_MAX_AUDIO_BYTES``, 25 MiB) + the multipart
#: envelope (boundaries, part headers, the three small form fields).
TRANSCRIBE_MAX_AUDIO_BYTES = 25 * 1024 * 1024

#: The bench's own upload ceiling (asr / vlm / ocr). The route enforces the same
#: number on the bytes it reads; this one stops the body before it is spooled.
SERVICE_LAB_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MULTIPART_ENVELOPE_BYTES = 64 * 1024
PROXY_BODY_LIMIT_OVERRIDES: dict[str, int] = {
    "/api/critique": ANALYSIS_CRITIQUE_MAX_HTML_BYTES + CRITIQUE_JSON_ENVELOPE_BYTES,
    "/api/transcribe": TRANSCRIBE_MAX_AUDIO_BYTES + MULTIPART_ENVELOPE_BYTES,
    "/api/services/asr": SERVICE_LAB_MAX_UPLOAD_BYTES + MULTIPART_ENVELOPE_BYTES,
    "/api/services/vlm": SERVICE_LAB_MAX_UPLOAD_BYTES + MULTIPART_ENVELOPE_BYTES,
    "/api/services/ocr": SERVICE_LAB_MAX_UPLOAD_BYTES + MULTIPART_ENVELOPE_BYTES,
}

_CODE = "REQUEST_BODY_TOO_LARGE"
_BUSY_CODE = "LARGE_BODY_SLOTS_BUSY"
#: Default process-wide slots for over-default bodies on override routes.
LARGE_BODY_SLOTS = 4
#: ``Retry-After`` on a 429 — about half the upstream analysis timeout.
LARGE_BODY_RETRY_AFTER_SECONDS = 30


def _too_large(limit: int) -> str:
    return f"Request body exceeds {limit} bytes."


class ProxyBodyLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        limit: int = PROXY_BODY_LIMIT_BYTES,
        prefixes: tuple[str, ...] = PROXY_BODY_LIMIT_PREFIXES,
        overrides: dict[str, int] | None = None,
        large_body_slots: int = LARGE_BODY_SLOTS,
    ) -> None:
        self.app = app
        self.limit = limit
        self.prefixes = prefixes
        self.overrides = PROXY_BODY_LIMIT_OVERRIDES if overrides is None else overrides
        #: One per app instance == process-wide (one app per worker process).
        self.large_body_slots = LargeBodySlots(large_body_slots)

    def _limit_for(self, path: str) -> int:
        return self.overrides.get(path.rstrip("/") or path, self.limit)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = str(scope.get("path", ""))
        if scope["type"] != "http" or not path.startswith(self.prefixes):
            await self.app(scope, receive, send)
            return
        limit = self._limit_for(path)
        declared = _content_length(scope)
        if declared is not None and declared > limit:
            await _reject(scope, receive, send, limit)
            return

        slots = self.large_body_slots if limit > self.limit else None
        guard = _StreamingGuard(scope, receive, send, limit, self.limit, slots)
        try:
            await self.app(scope, guard.receive, guard.send)
        except Exception:
            if not guard.must_reject:
                raise
        finally:
            guard.release()
        if guard.must_reject and not guard.rejected:
            await guard.reject()


class LargeBodySlots:
    """A non-waiting counting semaphore. ``try_acquire`` never suspends, so the
    check-and-take is atomic on the event loop (no lock needed)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.in_use = 0

    def try_acquire(self) -> bool:
        if self.in_use >= self.capacity:
            return False
        self.in_use += 1
        return True

    def release(self) -> None:
        self.in_use -= 1


class _StreamingGuard:
    """Per-request streaming byte counter, large-body slot, and 413/429
    substitution (see module doc)."""

    def __init__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        limit: int,
        slot_threshold: int,
        slots: LargeBodySlots | None,
    ) -> None:
        self._scope = scope
        self._receive = receive
        self._send = send
        self._limit = limit
        self._slot_threshold = slot_threshold
        self._slots = slots
        self.holds_slot = False
        #: Body bytes actually pulled off the wire (tests assert on this).
        self.received = 0
        #: Why reading stopped: 413 (over the cap) or 429 (no large-body slot).
        self.abort_status: int | None = None
        self.app_started = False
        self.rejected = False

    @property
    def must_reject(self) -> bool:
        return self.abort_status is not None and not self.app_started

    def _take_slot(self) -> bool:
        if self._slots is None or not self._slots.try_acquire():
            return False
        self.holds_slot = True
        return True

    def release(self) -> None:
        if self.holds_slot and self._slots is not None:
            self.holds_slot = False
            self._slots.release()

    async def receive(self) -> Message:
        if self.abort_status is not None:
            return {"type": "http.disconnect"}
        message = await self._receive()
        if message["type"] != "http.request":
            return message
        self.received += len(message.get("body", b""))
        if self.received > self._limit:
            self.abort_status = status.HTTP_413_CONTENT_TOO_LARGE
        elif (
            self._slots is not None
            and not self.holds_slot
            and self.received > self._slot_threshold
            and not self._take_slot()
        ):
            self.abort_status = status.HTTP_429_TOO_MANY_REQUESTS
        if self.abort_status is not None:
            return {"type": "http.disconnect"}  # stop reading; never hand the app this chunk
        return message

    async def send(self, message: Message) -> None:
        if self.must_reject:
            if not self.rejected:
                await self.reject()
            return  # swallow the app's own answer to the aborted read
        if message["type"] == "http.response.start":
            self.app_started = True
        await self._send(message)

    async def reject(self) -> None:
        self.rejected = True
        if self.abort_status == status.HTTP_429_TOO_MANY_REQUESTS:
            await _busy(self._scope, _disconnected, self._send)
        else:
            await _reject(self._scope, _disconnected, self._send, self._limit)


async def _disconnected() -> Message:
    return {"type": "http.disconnect"}


def _content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


async def _reject(scope: Scope, receive: Receive, send: Send, limit: int) -> None:
    response = JSONResponse(
        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        content={
            "error": {
                "code": _CODE,
                "message": _too_large(limit),
                "status": status.HTTP_413_CONTENT_TOO_LARGE,
                "request_id": str(uuid7()),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        },
    )
    await response(scope, receive, send)


async def _busy(scope: Scope, receive: Receive, send: Send) -> None:
    response = JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        headers={"Retry-After": str(LARGE_BODY_RETRY_AFTER_SECONDS)},
        content={
            "error": {
                "code": _BUSY_CODE,
                "message": "Too many large request bodies in flight; retry later.",
                "status": status.HTTP_429_TOO_MANY_REQUESTS,
                "request_id": str(uuid7()),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        },
    )
    await response(scope, receive, send)
