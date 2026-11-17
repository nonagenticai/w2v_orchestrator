"""Pod-lifecycle machinery carved out of ``src.asgi``.

Behaviour-preserving decomposition of the asgi god-module's lifespan clusters:

- ``reconciler`` — orphan-run reconciler (durable resume + periodic timer).
- ``audit_prune`` — ``request_audit`` retention loop.
- ``metrics`` — dedicated plaintext Prometheus metrics listener.

``src.asgi.lifespan`` imports from these and calls them exactly as before; this
package holds no state and starts nothing at import time.
"""
