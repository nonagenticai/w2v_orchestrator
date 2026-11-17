"""Unit tests for OpenTelemetry tracing setup + export.

The cluster ConfigMap has set OTLP_TRACES_ENDPOINT for months but the
production Dockerfile never installed the optional `otel` extra, so
setup_tracing() silently degraded to ImportError → tracing disabled.
The fix moved opentelemetry-{api,sdk} + the OTLP HTTP exporter into main
deps. These tests pin that wiring so a future deps-prune can't silently
re-break it.

Note: OTel installs a global TracerProvider and refuses to replace it
once set. To avoid cross-test bleed we (a) test setup_tracing() through
its boolean return only and (b) test span emission with a locally-created
TracerProvider rather than the global.
"""

from __future__ import annotations

import os
from unittest.mock import patch


def test_setup_tracing_returns_true_when_sdk_available() -> None:
    from src.observability.tracing import setup_tracing

    env = {k: v for k, v in os.environ.items() if not k.startswith(("OTEL_", "OTLP_"))}
    with patch.dict(os.environ, env, clear=True):
        assert setup_tracing("test-svc", "test") is True


def test_setup_tracing_respects_otel_sdk_disabled() -> None:
    from src.observability.tracing import setup_tracing

    with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}, clear=False):
        assert setup_tracing("test-svc", "test") is False


def test_otlp_exporter_package_importable() -> None:
    """The OTLP HTTP exporter MUST be in main deps (not optional) so the
    production image installs it. Pre-fix, this import would ImportError
    inside setup_tracing() and tracing would silently degrade."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    assert OTLPSpanExporter is not None
    assert BatchSpanProcessor is not None


def test_spans_emit_via_in_memory_exporter_using_local_provider() -> None:
    """End-to-end span lifecycle smoke against a LOCAL TracerProvider so we
    don't collide with the OTel global. Validates the SDK + InMemoryExporter
    integration our trace probe depends on."""
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test-svc"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = trace.get_tracer("orchestrator-probe", tracer_provider=provider)
    with tracer.start_as_current_span("orchestrator.smoke") as span:
        span.set_attribute("job_id", "test-123")
        span.set_attribute("stage", "smoke")

    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].name == "orchestrator.smoke"
    attrs = finished[0].attributes
    assert attrs is not None
    assert attrs["job_id"] == "test-123"
