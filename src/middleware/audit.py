"""Audit middleware: log + durably persist state-changing / auth-failure /
cross-tenant requests.

Mounted OUTER to ``AuthMiddleware`` (see ``asgi.py``) — Auth returns its own 401s
WITHOUT calling ``call_next``, so an inner middleware would never see them; an
outer one does, and still reads the final response status. ``request.state`` is
the shared per-request ``scope["state"]`` dict, so identity that Auth (inner)
stashes (``tenant_id``/``roles``/``user``) is visible here AFTER ``call_next``
returns. ``subject_id`` is NOT on ``request.state`` (only in a ContextVar that
does not cross the BaseHTTPMiddleware task boundary), so the actor is derived from
``request.state.user.sub``.

The DB write is **fail-safe**: a fresh short session, time-bounded, and any error
is swallowed + logged — a failed audit insert never breaks the request it is
auditing. See ``request_audit`` (migration 0026) and ``RequestAuditRepository``.

⚠️ It is **NOT off the hot path**, despite what this docstring claimed until
2026-08-18. ``await self._persist(...)`` runs before ``return response``, so
``_AUDIT_WRITE_TIMEOUT_S`` is not a bound on background latency — it is added
directly to p99 request latency whenever the DB pool is contended. The 15
``request_audit write failed`` warnings/24h observed in prod each represent a
request that also waited up to 2s. Moving the write to a tracked background task
is the real fix, but it changes the request lifecycle and wants its own PR; until
then, do not read this as "free".
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from uuid import UUID

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.api.routes._common import resolve_active_tenant
from src.auth.membership import RESOLVED_BY_MEMBERSHIP
from src.database.repositories.request_audit_repository import RequestAuditRepository
from src.utils import safe_exc_message

logger = logging.getLogger("w2v_orchestrator.audit")

_MUTATION_METHODS = frozenset({"POST", "PUT", "DELETE", "PATCH"})
_AUTH_FAILURE_CODES = frozenset({401, 403})
# Cap the off-hot-path write so a hung DB can never add unbounded latency.
_AUDIT_WRITE_TIMEOUT_S = 2.0


def fingerprint_api_key(api_key: str | None) -> str | None:
    """Return a short, non-reversible fingerprint of an API key."""
    if not api_key:
        return None
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def _resource_from_path(path: str) -> str | None:
    """Extract a logical resource name from a URL path."""
    parts = path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "api":
        resource = parts[1]
        if resource.endswith("s"):
            resource = resource[:-1]
        return resource
    return None


def _best_effort_uuid(val: object) -> UUID | None:
    if val is None:
        return None
    if isinstance(val, UUID):
        return val
    try:
        return UUID(str(val))
    except TypeError, ValueError:
        return None


def _credential(request: Request) -> str | None:
    """The caller's bearer credential: license token (``X-Tenant-Id: Bearer ..``)
    or the Keycloak JWT (``Authorization: Bearer ..``)."""
    xt = request.headers.get("x-tenant-id", "")
    if xt.lower().startswith("bearer "):
        return xt[7:].strip() or None
    return request.headers.get("authorization", "").removeprefix("Bearer ").strip() or None


def _build_detail(
    request: Request,
    *,
    claim: UUID | None,
    cross_tenant: bool,
    exc: Exception | None,
) -> dict[str, object]:
    """The row's free-form ``detail`` column. Never carries a body, a header or
    a raw credential — only derived, bounded facts."""
    detail: dict[str, object] = {}
    if cross_tenant and claim is not None:
        detail["from_tenant"] = str(claim)
    # A non-UUID tenant claim (e.g. the route-dep 401 "malformed claim"):
    # keep the raw value for forensics since it can't go in the uuid column.
    raw_claim = getattr(request.state, "tenant_id", None)
    if claim is None and raw_claim is not None:
        detail["raw_tenant_claim"] = str(raw_claim)
    if request.headers.get("idempotency-key"):
        detail["idempotent_replay"] = True
    if exc is not None:
        # Type + message ONLY. No body, no headers, no token — `_persist`'s
        # shape is unchanged and `detail` is the one free-form column. The
        # message is scrubbed and truncated because an exception string is
        # attacker-adjacent data (a driver error can echo a DSN, an httpx
        # error a credentialed URL).
        detail["exception_type"] = type(exc).__name__
        detail["exception_message"] = safe_exc_message(exc)
    return detail


class AuditMiddleware(BaseHTTPMiddleware):
    """Log + persist mutations, auth failures, and cross-tenant access."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception as exc:
            # An unhandled exception does NOT come back as a response: it
            # propagates past this frame to ServerErrorMiddleware (Starlette
            # mounts that OUTSIDE every user middleware, and w2v_common's
            # `Exception` envelope handler is installed there), which renders
            # the 500 and re-raises. Until 2026-08-23 this frame had no
            # `except`, so every 5xx bypassed `_record` entirely — the live
            # `request_audit` table held 11,465 rows and not one with
            # `status_code >= 500`, ever. The instrument was blind, not the
            # service healthy.
            #
            # `except Exception`, deliberately NOT `BaseException`:
            # `asyncio.CancelledError` is a BaseException in modern Python and
            # is a control-flow signal (client disconnect / ASGI abort /
            # shutdown), not a server error. Recording it as a 500 would
            # manufacture fake outages, and this repo has already been bitten
            # by treating an abort signal as an error. It therefore passes
            # through untouched and unrecorded.
            await self._record_unhandled(request, exc, (time.monotonic() - start) * 1000)
            # Bare `raise`: re-raises the ORIGINAL exception unchanged, with its
            # traceback, __cause__ and __context__ intact. The error response
            # shape and status code are decided downstream exactly as before —
            # we added a record, we did not handle the error.
            raise
        await self._record(request, response.status_code, (time.monotonic() - start) * 1000)
        return response

    async def _record_unhandled(self, request: Request, exc: Exception, duration_ms: float) -> None:
        """Record an unhandled exception as a 500, never masking the original.

        Every failure mode here is swallowed: if the audit path itself raises,
        the caller's bare `raise` must still surface the ORIGINAL exception, so
        the substitute must never escape. `CancelledError` is intentionally NOT
        caught — if the request task is being cancelled while we write, letting
        the cancellation win is correct.
        """
        try:
            await self._record(
                request,
                500,
                duration_ms,
                exc=exc,
            )
        except Exception:  # must never mask the original exception
            logger.warning(
                "request_audit unhandled-exception record failed (request_id=%s)",
                getattr(request.state, "request_id", None) or "-",
                exc_info=True,
            )

    async def _record(
        self,
        request: Request,
        status_code: int,
        duration_ms: float,
        *,
        exc: Exception | None = None,
    ) -> None:
        """Classify one finished request and log + persist it if it qualifies."""
        method = request.method

        # Effective-tenant + scope-switch, derived from the SAME rule the route
        # dependency uses (resolve_active_tenant) so the two never drift.
        claim = _best_effort_uuid(getattr(request.state, "tenant_id", None))
        roles = set(getattr(request.state, "roles", []) or [])
        # Read BOTH channels, exactly as `get_tenant_id` does.
        # A browser-initiated load (<img>/<video>/<a download>/EventSource) cannot
        # set the header and carries the switch as `?active_tenant=` instead — if
        # this middleware looked at the header alone it would score those requests
        # `cross_tenant=False`, and since they are GETs `should_record` would be
        # False. A super-admin could then read another tenant's artifacts with no
        # durable audit row at all, which is precisely the trail this table exists
        # to keep. Header still wins, same precedence as the dep.
        override = request.headers.get("x-active-tenant-id") or request.query_params.get(
            "active_tenant"
        )
        cross_tenant = False
        effective = claim
        if getattr(request.state, "tenant_resolved_by", None) == RESOLVED_BY_MEMBERSHIP:
            # Membership mode: TenancyMiddleware already rewrote state.tenant_id
            # to the selected tenant, so comparing it with the override is always
            # equal. It flags the one cross-tenant shape explicitly: a
            # super_admin acting in a tenant where they hold no membership.
            cross_tenant = getattr(request.state, "cross_tenant_scope", False) is True
        elif claim is not None:
            effective, cross_tenant = resolve_active_tenant(claim, override, roles)

        is_auth_failure = status_code in _AUTH_FAILURE_CODES
        is_mutation = method in _MUTATION_METHODS
        is_server_error = status_code >= 500
        # OPTIONS/HEAD are never recorded; everything else is recorded only if it
        # mutates, fails auth, crosses a tenant boundary (incl. a GET read), or
        # is a server error. The 5xx arm covers BOTH shapes: an unhandled
        # exception (routed here by dispatch's except) and a 500-status response
        # that came back normally (e.g. `HTTPException(500)`, which the inner
        # ExceptionMiddleware turns into a response). Without it a failing GET
        # would still leave no trace.
        should_record = method not in ("OPTIONS", "HEAD") and (
            is_mutation or is_auth_failure or cross_tenant or is_server_error
        )
        if not should_record:
            return

        # event is the trigger class (precedence: auth_failure → mutation → read);
        # cross_tenant is an orthogonal dimension stored alongside.
        if is_auth_failure:
            event = "auth_failure"
        elif is_mutation:
            event = "mutation"
        else:
            event = "read"

        actor = _best_effort_uuid(getattr(getattr(request.state, "user", None), "sub", None))
        resource = _resource_from_path(request.url.path)
        ip = request.client.host if request.client else None
        request_id = getattr(request.state, "request_id", None)

        detail = _build_detail(request, claim=claim, cross_tenant=cross_tenant, exc=exc)

        log_data: dict[str, object] = {
            "tenant_id": str(effective) if effective is not None else "anonymous",
            "method": method,
            "path": request.url.path,
            "resource": resource,
            "status_code": status_code,
            "ip": ip or "unknown",
            "duration_ms": round(duration_ms, 1),
            "cross_tenant": cross_tenant,
            "event": event,
        }
        if exc is not None:
            log_data["exception"] = type(exc).__name__
        if is_auth_failure:
            logger.warning("AUTH_FAILURE: %s", log_data)
        elif exc is not None:
            # ERROR, no exc_info: the traceback is already logged once by
            # w2v_common's `Exception` envelope handler (outermost). This line
            # is the audit projection, not a second stack dump.
            logger.error("AUDIT_UNHANDLED: %s", log_data)
        else:
            logger.info("AUDIT: %s", log_data)

        await self._persist(
            request,
            actor_subject_id=actor,
            claim_tenant_id=claim,
            effective_tenant_id=effective,
            cross_tenant=cross_tenant,
            event=event,
            method=method,
            path=request.url.path,
            resource=resource,
            status_code=status_code,
            ip=ip,
            duration_ms=round(duration_ms, 1),
            credential_fingerprint=fingerprint_api_key(_credential(request)),
            request_id=request_id,
            detail=detail,
        )

    async def _persist(
        self,
        request: Request,
        *,
        actor_subject_id: UUID | None,
        claim_tenant_id: UUID | None,
        effective_tenant_id: UUID | None,
        cross_tenant: bool,
        event: str,
        method: str,
        path: str,
        resource: str | None,
        status_code: int,
        ip: str | None,
        duration_ms: float | None,
        credential_fingerprint: str | None,
        request_id: str | None,
        detail: dict[str, object] | None = None,
    ) -> None:
        """Write one request_audit row, fail-safe. Never raises into dispatch."""
        app_state = request.app.state
        factory = getattr(app_state, "session_factory", None)
        if factory is None or getattr(app_state, "shutting_down", False):
            # Degraded/import-time app, or draining (engine may be closing) — the
            # log line above is the only record; do not risk a hot-path error.
            return
        try:
            async with asyncio.timeout(_AUDIT_WRITE_TIMEOUT_S):
                async with factory() as session:
                    await RequestAuditRepository(session).record(
                        actor_subject_id=actor_subject_id,
                        claim_tenant_id=claim_tenant_id,
                        effective_tenant_id=effective_tenant_id,
                        cross_tenant=cross_tenant,
                        event=event,
                        method=method,
                        path=path,
                        resource=resource,
                        status_code=status_code,
                        ip=ip,
                        duration_ms=duration_ms,
                        credential_fingerprint=credential_fingerprint,
                        request_id=request_id,
                        detail=detail,
                    )
                    await session.commit()
        except Exception as exc:
            # Audit must never break the request it audits. Swallow + log.
            # (CancelledError is BaseException — left to propagate so a cancelled
            # request task isn't masked; drain is handled by the shutting_down
            # guard above, and asyncio.timeout raises TimeoutError, caught here.)
            # `%r`, NOT `%s`, deliberately: the dominant failure here is the
            # asyncio.timeout above, and CPython raises a BARE `TimeoutError()`
            # (asyncio/timeouts.py: `raise TimeoutError from exc_val`, no args),
            # so `%s` renders an EMPTY string and the log line says nothing at
            # all. That shipped, and produced blind "write failed: " warnings in
            # prod. `%r` renders `TimeoutError()`; exc_info gives the traceback;
            # request_id is passed explicitly because AuditMiddleware is a
            # BaseHTTPMiddleware mounted OUTSIDE RequestIdMiddleware, so the
            # ContextVar set by the inner app does not propagate back out here.
            # Do not "tidy" %r back to %s.
            logger.warning(
                "request_audit write failed: %r (request_id=%s)",
                exc,
                request_id or "-",
                exc_info=True,
            )
