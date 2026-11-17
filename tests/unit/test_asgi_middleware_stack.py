"""The create_app migration's central claim: the canonical + extra middleware
end up in the exact request-time order the orchestrator's invariants require.

After migrating onto ``w2v_common.asgi.create_app`` the stack is assembled from
two sources — the factory's canonical middleware (SecurityHeaders + RequestId +
Auth + Idempotency) and the service-specific ``extra_middleware`` (Metrics,
Audit, Shutdown, CORS). Starlette's ``user_middleware`` is ordered outer→inner
(index 0 = outermost, seen first inbound). We assert the relative order that
encodes the invariants:

* CORS outermost  → a preflight OPTIONS is answered before auth can 401 it.
* Shutdown next   → drain 503s skip auth + DB-session cost.
* Audit outer of RequestId/Auth → it sees Auth's 401s and reads request_id
  (which the inner canonical RequestId sets) after call_next.
* RequestId outer of Auth (w2v_common 0.34.0 fix) → 401s still carry X-Request-Id.
* Idempotency innermost / inner of Auth → the tenant is known when it keys.

``src.asgi`` is imported inside each test (not at module import) so the autouse
env fixture has set REDIS_URL before the app is built — the pattern the other
``test_asgi_*`` modules use.
"""

from __future__ import annotations


def _order() -> list[str]:
    from src.asgi import app

    # user_middleware is outer→inner (index 0 = outermost). ``m.cls`` is typed as
    # a middleware factory by Starlette, so read ``__name__`` via getattr.
    # The early-access service doors swap in an AuthMiddleware SUBCLASS (same slot,
    # same arguments; src/auth/signup_doors.py) — it IS the AuthMiddleware here.
    names = [getattr(m.cls, "__name__", "") for m in app.user_middleware]
    return ["AuthMiddleware" if n == "SignupDoorAwareAuthMiddleware" else n for n in names]


def _rank(order: list[str], name: str) -> int:
    assert name in order, f"{name} missing from middleware stack: {order}"
    return order.index(name)


def test_idempotency_is_enabled_and_registered() -> None:
    """The canonical (tenant-keyed, 0.34.0-fixed) IdempotencyMiddleware is wired."""
    assert "IdempotencyMiddleware" in _order()


def test_request_id_is_the_shared_middleware() -> None:
    from src.asgi import app

    req_id = next(
        m for m in app.user_middleware if getattr(m.cls, "__name__", "") == "RequestIdMiddleware"
    )
    assert getattr(req_id.cls, "__module__", "").startswith("w2v_common"), (
        "RequestId must be the shared w2v_common middleware, not a local clone"
    )


def test_middleware_order_encodes_invariants() -> None:
    order = _order()
    # CORS outermost, then Shutdown, then Audit.
    assert _rank(order, "CORSMiddleware") < _rank(order, "ShutdownMiddleware")
    assert _rank(order, "ShutdownMiddleware") < _rank(order, "AuditMiddleware")
    # Audit outer of RequestId (so request_id is set when Audit records post-call_next).
    assert _rank(order, "AuditMiddleware") < _rank(order, "RequestIdMiddleware")
    # RequestId outer of Idempotency; Idempotency is the innermost canonical.
    assert _rank(order, "RequestIdMiddleware") < _rank(order, "IdempotencyMiddleware")
    # SecurityHeaders (canonical, forced) sits inside the extra group.
    assert _rank(order, "MetricsMiddleware") < _rank(order, "SecurityHeadersMiddleware")


def test_tenancy_resolves_inside_auth_and_before_idempotency() -> None:
    """P2 WP3: membership resolution needs Auth's verified identity, and must bind
    the tenant before Idempotency keys on it or any route/client runs."""
    order = _order()
    assert _rank(order, "AuthMiddleware") + 1 == _rank(order, "TenancyMiddleware")
    assert _rank(order, "TenancyMiddleware") < _rank(order, "IdempotencyMiddleware")
