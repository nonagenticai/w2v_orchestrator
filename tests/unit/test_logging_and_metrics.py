"""Tests for logging configuration and Prometheus metrics definitions.

Metrics tests verify correct metric types, label names, and that counters/
histograms can actually be incremented/observed without error.
Logging tests verify the JSONFormatter produces spec-compliant output.
"""

from __future__ import annotations

import json
import logging

from w2v_common.observability.metrics import (
    http_request_duration_seconds,
    http_requests_total,
)

from src.observability import (
    JSONFormatter,
    active_runs,
    configure_logging,
    db_errors_total,
    notify_errors_total,
    runs_total,
    stage_duration_seconds,
    stage_total,
)

# ============================================================================
# JSONFormatter
# ============================================================================


def test_json_formatter_produces_valid_json_with_all_required_fields():
    """JSONFormatter output is valid JSON with timestamp, level, logger, message."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="test message",
        args=None,
        exc_info=None,
    )
    output = formatter.format(record)
    data = json.loads(output)

    assert data["level"] == "INFO"
    assert data["logger"] == "test_logger"
    assert data["message"] == "test message"
    assert "timestamp" in data
    # timestamp should be a non-empty string
    assert len(data["timestamp"]) > 0


def test_json_formatter_resolves_format_args():
    """JSONFormatter resolves %-style message arguments into final message."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="test.py",
        lineno=10,
        msg="count is %d, name is %s",
        args=(42, "alice"),
        exc_info=None,
    )
    output = formatter.format(record)
    data = json.loads(output)

    assert data["message"] == "count is 42, name is alice"
    assert data["level"] == "WARNING"


def test_json_formatter_includes_exception_traceback():
    """JSONFormatter includes exception info with class name in the output."""
    formatter = JSONFormatter()
    try:
        raise ValueError("test error")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=10,
        msg="something failed",
        args=None,
        exc_info=exc_info,
    )
    output = formatter.format(record)
    data = json.loads(output)

    assert "exception" in data
    assert "ValueError" in data["exception"]
    assert "test error" in data["exception"]


def test_json_formatter_includes_request_id_when_present():
    """JSONFormatter includes request_id if the attribute exists on the record."""
    formatter = JSONFormatter()
    record = logging.makeLogRecord(
        {
            "name": "test",
            "levelno": logging.INFO,
            "levelname": "INFO",
            "pathname": "test.py",
            "lineno": 10,
            "msg": "with request id",
            "args": None,
            "exc_info": None,
            "request_id": "req-abc-123",
        }
    )
    output = formatter.format(record)
    data = json.loads(output)

    assert data["request_id"] == "req-abc-123"


def test_json_formatter_omits_request_id_when_absent():
    """JSONFormatter does not include request_id when not set on record."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="no request id",
        args=None,
        exc_info=None,
    )
    output = formatter.format(record)
    data = json.loads(output)

    assert "request_id" not in data or data.get("request_id") is None


# ============================================================================
# configure_logging
# ============================================================================


def test_configure_logging_text_mode_sets_level():
    """configure_logging in text mode sets root logger level."""
    configure_logging(json_logs=False, level="DEBUG")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) >= 1
    # Verify no JSONFormatter in text mode
    json_formatters = [h for h in root.handlers if isinstance(h.formatter, JSONFormatter)]
    assert len(json_formatters) == 0
    configure_logging(json_logs=False, level="INFO")


def test_configure_logging_json_mode_installs_json_formatter():
    """configure_logging in JSON mode installs a JSON-shaped formatter on a handler."""
    from src.observability.logging import CorrelatedJSONFormatter

    configure_logging(json_logs=True, level="INFO")
    root = logging.getLogger()
    json_formatters = [
        h for h in root.handlers if isinstance(h.formatter, JSONFormatter | CorrelatedJSONFormatter)
    ]
    assert len(json_formatters) >= 1
    configure_logging(json_logs=False, level="INFO")


def test_configure_logging_does_not_accumulate_handlers():
    """Calling configure_logging twice does not double the handler count."""
    configure_logging(json_logs=False, level="INFO")
    root = logging.getLogger()
    count_after_first = len(root.handlers)
    configure_logging(json_logs=False, level="INFO")
    count_after_second = len(root.handlers)
    assert count_after_second <= count_after_first


# ============================================================================
# Prometheus metrics: type, labels, and functional increment/observe
# ============================================================================


def test_http_requests_total_is_canonical_w2v_family():
    """Inbound HTTP uses the fleet-canonical w2v_http_* family (not a fork)."""
    assert http_requests_total._type == "counter"
    assert http_requests_total._name == "w2v_http_requests"
    label_names = http_requests_total._labelnames
    assert "method" in label_names
    assert "endpoint" in label_names
    assert "status_code" in label_names


def test_http_request_duration_is_histogram_with_sub_second_buckets():
    """w2v_http_request_duration_seconds has sub-second buckets for fast endpoints."""
    assert http_request_duration_seconds._type == "histogram"
    buckets = list(http_request_duration_seconds._kwargs.get("buckets", []))
    assert any(b < 1.0 for b in buckets), "Expected sub-second buckets"


def test_runs_total_is_counter_with_outcome_label():
    """w2v_orchestrator_runs_total has an 'outcome' label and increments."""
    assert runs_total._type == "counter"
    assert "outcome" in runs_total._labelnames
    before = runs_total.labels(outcome="completed")._value.get()
    runs_total.labels(outcome="completed").inc()
    assert runs_total.labels(outcome="completed")._value.get() == before + 1


def test_active_runs_is_gauge_and_supports_inc_dec():
    """w2v_orchestrator_active_runs is a Gauge that can inc/dec."""
    assert active_runs._type == "gauge"
    before = active_runs._value.get()
    active_runs.inc()
    assert active_runs._value.get() == before + 1
    active_runs.dec()
    assert active_runs._value.get() == before


def test_stage_duration_is_histogram_with_stage_and_outcome_labels():
    """w2v_orchestrator_stage_duration_seconds has stage + outcome labels."""
    assert stage_duration_seconds._type == "histogram"
    label_names = stage_duration_seconds._labelnames
    assert "stage" in label_names
    assert "outcome" in label_names


def test_stage_duration_histogram_has_long_buckets():
    """Stage durations can be minutes-long; buckets should go up to 600s."""
    buckets = list(stage_duration_seconds._kwargs.get("buckets", []))
    assert max(buckets) >= 600.0


def test_stage_total_is_counter_with_stage_and_outcome():
    """w2v_orchestrator_stage_total has stage and outcome labels."""
    assert stage_total._type == "counter"
    label_names = stage_total._labelnames
    assert "stage" in label_names
    assert "outcome" in label_names


def test_notify_errors_total_is_counter():
    """w2v_orchestrator_notify_errors_total counts silent SSE-publish failures."""
    assert notify_errors_total._type == "counter"
    before = notify_errors_total._value.get()
    notify_errors_total.inc()
    assert notify_errors_total._value.get() == before + 1


def test_db_errors_total_is_counter_with_op_label():
    """w2v_orchestrator_db_errors_total has an 'op' label."""
    assert db_errors_total._type == "counter"
    assert "op" in db_errors_total._labelnames
    before = db_errors_total.labels(op="reconciler_tick")._value.get()
    db_errors_total.labels(op="reconciler_tick").inc()
    assert db_errors_total.labels(op="reconciler_tick")._value.get() == before + 1


# ============================================================================
# request_id binding — regression: the filter must ride on the HANDLER
# ============================================================================


def test_configure_logging_binds_request_id_on_child_logger_records(capsys):
    """A CHILD logger's record carries ``request_id`` in the emitted JSON.

    Regression guard. ``RequestIdLogFilter`` used to be registered with
    ``logging.getLogger().addFilter(...)`` in ``src/asgi.py``. Logger-level
    filters only run in ``Logger.handle()`` for the logger the record was
    logged ON; ``Logger.callHandlers`` walks ancestors and invokes their
    *handlers* without re-applying the ancestors' logger filters. Every line
    this service actually emits comes from a child logger (``src.*``,
    ``w2v_common.*``), so that placement bound ``request_id`` on nothing —
    live pod lines carried ``correlation_id`` and no ``request_id`` at all.

    Asserting on a CHILD logger is the whole point: the same assertion made on
    the ROOT logger passes against the broken placement, which is why the bug
    survived a test file that already covered ``configure_logging``.
    """
    from w2v_common.observability.request_context import clear_request_id, set_request_id

    configure_logging(json_logs=True, level="INFO")
    set_request_id("req-abc-123")
    try:
        logging.getLogger("w2v_common.storage.service").info("child record")
    finally:
        clear_request_id()
        captured = capsys.readouterr()
        configure_logging(json_logs=False, level="INFO")

    lines = [line for line in captured.out.splitlines() if "child record" in line]
    assert lines, f"no log line emitted; stdout={captured.out!r}"
    entry = json.loads(lines[-1])
    assert entry["logger"] == "w2v_common.storage.service"
    assert entry.get("request_id") == "req-abc-123", (
        "request_id was not bound onto a CHILD logger's record — the "
        f"RequestIdLogFilter is not on the stdout handler. entry={entry}"
    )
