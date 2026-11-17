"""Dedicated plaintext Prometheus metrics listener — carved out of ``src.asgi``.

Behaviour-preserving move of the second (bare WSGI) metrics server that serves
the default ``prometheus_client`` REGISTRY on ``API_METRICS_PORT`` so the
cluster's non-ambient (plaintext) Prometheus can scrape it without relaxing the
STRICT-mTLS API port (:6300). The in-mesh ``GET /metrics`` route stays on
``app`` in ``src.asgi`` and is unaffected; ``src.asgi.lifespan`` calls
``_start_metrics_server`` / ``_stop_metrics_server`` exactly as before.
"""

import logging
from wsgiref.simple_server import WSGIServer

from fastapi import FastAPI
from prometheus_client import start_http_server

logger = logging.getLogger(__name__)


def _start_metrics_server(port: int) -> WSGIServer | None:
    """Start a dedicated plaintext Prometheus metrics listener on ``port``.

    The main API port (:6300) is under STRICT mTLS, so the cluster's
    non-ambient (plaintext) Prometheus cannot scrape its /metrics. This binds a
    second, bare WSGI server serving the same default prometheus_client
    REGISTRY; gitops marks the port PERMISSIVE (workload-scoped PeerAuth) and
    admits it via the w2v NetworkPolicy, keeping :6300 STRICT. The in-mesh
    ``GET /metrics`` route on :6300 is retained for HBONE callers.

    Returns the server so the lifespan can shut it down, or ``None`` if the bind
    fails (e.g. the port is already taken under a test harness) — a metrics
    listener must never crash the app.
    """
    try:
        httpd, _thread = start_http_server(port)
    except OSError as exc:
        logger.warning("Metrics server failed to bind :%d: %s", port, exc)
        return None
    logger.info("Prometheus metrics server listening on :%d", port)
    return httpd


def _stop_metrics_server(app: FastAPI) -> None:
    """Shut down the dedicated metrics listener if one was started.

    Called unconditionally from the lifespan teardown (the None-check lives here,
    not inline, to keep ``lifespan`` under ruff's C901 complexity ceiling).
    """
    httpd = getattr(app.state, "metrics_httpd", None)
    if httpd is not None:
        httpd.shutdown()
