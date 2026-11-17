"""OpenTelemetry tracing for pipeline stages.

setup_tracing() wires both a TracerProvider AND an OTLP HTTP exporter when
the OTLP_TRACES_ENDPOINT env var (or its standard OTEL_EXPORTER_OTLP_ENDPOINT
equivalent) is set. Without the exporter the SDK creates spans locally but
never ships them — which is exactly the silent failure mode the sibling
trace probe (scripts/verify_otel_export.py) was added to catch.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


class PipelineTracer:
    """Trace pipeline stage execution with OpenTelemetry."""

    def __init__(self, service_name: str = "w2v-pipeline", enabled: bool = False) -> None:
        self.service_name = service_name
        self.enabled = enabled
        self._tracer: Any = None

        if enabled:
            try:
                self._setup_otel()
            except ImportError:
                logger.warning("OpenTelemetry not installed, tracing disabled")
                self.enabled = False

    def _setup_otel(self) -> None:
        """Set up OpenTelemetry tracer."""
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        resource = Resource.create({"service.name": self.service_name})
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(self.service_name)

    @contextmanager
    def span(self, name: str, attributes: dict[str, str] | None = None) -> Generator[Any]:
        """Create a trace span."""
        if not self.enabled or not self._tracer:
            yield None
            return

        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span

    def stage_span(self, job_id: str, stage_name: str) -> Any:
        """Create a span for a pipeline stage."""
        return self.span(
            f"pipeline.stage.{stage_name}",
            attributes={"job_id": job_id, "stage": stage_name},
        )


class NoOpTracer:
    """No-op tracer when tracing is disabled."""

    @contextmanager
    def span(self, name: str, attributes: dict | None = None) -> Generator[None]:
        """Yield None without creating a real span."""
        yield None

    def stage_span(self, job_id: str, stage_name: str) -> Any:
        """Return a no-op span context manager for the given stage."""
        return self.span(f"pipeline.stage.{stage_name}")


def _resolve_otlp_endpoint() -> str | None:
    """Pick the OTLP traces endpoint from env. None disables the exporter."""
    # OTLP_TRACES_ENDPOINT is the w2v-common-config name; OTEL_EXPORTER_*
    # variants are the OTel-spec names. Accept both so we don't have to
    # touch the ConfigMap if a future fleet renames.
    for key in (
        "OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ):
        value = os.environ.get(key)
        if value:
            return value
    return None


def setup_tracing(service_name: str = "w2v-orchestrator", environment: str = "development") -> bool:
    """Configure OpenTelemetry TracerProvider + OTLP HTTP exporter.

    Returns True if tracing + export were configured, False if the OTel
    packages are missing (degraded local mode). When OTLP_TRACES_ENDPOINT
    is unset the provider is still installed but no exporter is attached
    — spans are recorded locally and dropped on flush (useful for tests).
    """
    if os.environ.get("OTEL_SDK_DISABLED", "").lower() == "true":
        logger.info("OTEL_SDK_DISABLED=true — tracing skipped at setup_tracing()")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        resource = Resource.create(
            {
                "service.name": service_name,
                "deployment.environment": environment,
            }
        )

        provider = TracerProvider(resource=resource)

        endpoint = _resolve_otlp_endpoint()
        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
                logger.info("OTLP span exporter wired: endpoint=%s", endpoint)
            except ImportError:
                logger.warning(
                    "OTLP HTTP exporter package missing — spans will be created "
                    "but not exported. Install opentelemetry-exporter-otlp-proto-http."
                )

        trace.set_tracer_provider(provider)

        logger.info(
            "OpenTelemetry tracing configured: service=%s environment=%s export=%s",
            service_name,
            environment,
            "on" if endpoint else "off",
        )
        return True

    except ImportError:
        logger.warning(
            "OpenTelemetry SDK not installed — tracing disabled. "
            "Add opentelemetry-api + opentelemetry-sdk to main deps."
        )
        return False
    except (OSError, RuntimeError) as exc:
        logger.warning("Failed to configure OpenTelemetry tracing: %s", exc)
        return False
