"""Prometheus metrics middleware."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from w2v_common.observability.metrics import (
    http_request_duration_seconds,
    http_requests_total,
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record HTTP request metrics."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Record request count and duration histograms in Prometheus."""
        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start

        # Label by the matched route template (``/api/runs/{run_id}``) rather
        # than the raw path. The raw path embeds the run/project UUID, so every
        # request would mint a fresh Prometheus series — unbounded cardinality
        # that bloats the TSDB and eventually OOMs the scrape. Unmatched paths
        # (404s, scanners) collapse to a single ``<unmatched>`` series.
        route = request.scope.get("route")
        endpoint = getattr(route, "path_format", None) or "<unmatched>"
        method = request.method
        status = str(response.status_code)

        http_requests_total.labels(method=method, endpoint=endpoint, status_code=status).inc()
        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)

        return response
