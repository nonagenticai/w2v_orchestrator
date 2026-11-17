"""Structured logging: JSON formatter, correlation-ID formatter, and setup helpers."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from urllib.parse import urlsplit

from w2v_common.observability.request_context import RequestIdLogFilter, get_request_id

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def _correlation_id() -> str:
    """Correlation id for the current context.

    Background run execution binds ``correlation_id_var`` to the run_id (via
    ``set_correlation_id``). HTTP requests no longer set it — the shared
    ``RequestIdMiddleware`` (installed by ``create_app``) binds the request-id
    ContextVar instead — so fall back to that so HTTP log lines keep a
    correlation id (equal to the request id, as before the create_app migration).
    """
    return correlation_id_var.get("") or get_request_id()


def set_correlation_id(cid: str | None = None) -> str:
    """Set the correlation ID for the current async context.

    Args:
        cid: Optional correlation ID. A UUID is generated if None.

    Returns:
        The correlation ID that was set.
    """
    import uuid

    cid = cid or str(uuid.uuid7())
    correlation_id_var.set(cid)
    return cid


@contextmanager
def bound_correlation_id(cid: str) -> Iterator[str]:
    """Bind ``correlation_id_var`` to ``cid`` for the block, then restore it.

    The scoped form of ``set_correlation_id`` for run execution. The run
    dispatcher wraps each run's task in this with the run id, so every log line a
    stage emits (and every task it spawns — ``asyncio.create_task`` copies the
    context) carries ``correlation_id=<run_id>`` and a run's stage events can be
    found by grepping the pod log for the run id.

    Before this was called anywhere, ``set_correlation_id`` had NO caller: stage
    lines carried ``correlation_id=""`` (reconciler-resumed runs) or the id of
    the HTTP request that created the run task — never the run id — so an
    investigation grepping a run id found only access lines.

    ``reset`` (not a second ``set``) on exit so an inline caller gets its prior
    value back instead of inheriting this run's id.
    """
    token = correlation_id_var.set(cid)
    try:
        yield cid
    finally:
        correlation_id_var.reset(token)


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines for structured log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record to a single-line JSON string."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log_entry["request_id"] = str(record.request_id)
        return json.dumps(log_entry, default=str)


class CorrelatedFormatter(logging.Formatter):
    """Log formatter that includes correlation ID."""

    def format(self, record: logging.LogRecord) -> str:
        record.correlation_id = _correlation_id()
        return super().format(record)


def _quiet_request_loggers() -> None:
    """Pin httpx/httpcore to WARNING.

    They log every request line ("HTTP Request: GET <url>") at INFO with the URL
    verbatim — userinfo included — so a password-protected site_capture
    ``source_url`` put its password into the pod log. Warnings still pass.
    """
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def configure_logging(json_logs: bool = False, level: str = "INFO") -> None:
    """Configure root logger with either JSON or text formatting."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    _quiet_request_loggers()

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level.upper())
    # The request-id filter MUST ride on the HANDLER, not on the root logger.
    # ``Logger.filter()`` only runs for records logged *directly* on that
    # logger; ``Logger.callHandlers`` walks ancestors and invokes their
    # *handlers* without re-applying the ancestors' logger-level filters. Every
    # line this service emits comes from a child logger (``src.*``,
    # ``w2v_common.*``), so a filter registered via
    # ``logging.getLogger().addFilter(...)`` never runs and ``request_id`` is
    # never bound — which is exactly what shipped: live pod lines carried
    # ``correlation_id`` but no ``request_id``. Handler filters DO run for
    # propagated records, so this placement is the one that works.
    handler.addFilter(RequestIdLogFilter())

    if json_logs:
        handler.setFormatter(CorrelatedJSONFormatter())
    else:
        handler.setFormatter(
            CorrelatedFormatter(
                "%(asctime)s [%(correlation_id)s] %(name)s %(levelname)s %(message)s"
            )
        )

    root.addHandler(handler)


class CorrelatedJSONFormatter(logging.Formatter):
    """JSON log formatter with correlation ID."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": _correlation_id(),
        }
        # ``request_id`` is bound by w2v_common.observability.RequestIdLogFilter
        # (registered on the root logger in asgi.py lifespan). Surfacing both
        # ``correlation_id`` (orchestrator-local) and ``request_id`` (the
        # cross-service trace id) makes it explicit during log review which
        # field can be grepped against peer-service logs.
        if hasattr(record, "request_id") and record.request_id:
            entry["request_id"] = str(record.request_id)
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "job_id"):
            entry["job_id"] = str(record.job_id)
        if hasattr(record, "stage"):
            entry["stage"] = str(record.stage)
        return json.dumps(entry, default=str)


def setup_correlated_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure logging with correlation ID support."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    _quiet_request_loggers()

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(CorrelatedJSONFormatter())
    else:
        handler.setFormatter(
            CorrelatedFormatter(
                "%(asctime)s [%(correlation_id)s] %(name)s %(levelname)s %(message)s"
            )
        )

    root.addHandler(handler)


def database_endpoint(url: str) -> str:
    """Render a database DSN as ``host:port/dbname`` for logging — never userinfo.

    WHY: this line used to log ``settings.database.url`` raw, i.e. the full
    ``postgresql+psycopg://<user>:<password>@host:port/db`` DSN, at INFO on the
    connect path. That was inert only because gitops pinned the orchestrator to
    ``LOG_LEVEL=WARNING``; the moment the level was raised to INFO (gitops #2952,
    so that this service logged anything at all) the live ``w2v_orch_app``
    password began landing in plaintext in every pod's stdout and in anything
    shipping those logs. A log line whose safety depends on the log LEVEL is not
    safe, it is merely unexercised.

    Mirrors ``w2v_common.metering.cost_bus._redis_endpoint``, which fixed the
    identical defect for the metering Redis DSN (w2v_common #119). Only the
    ``urlsplit`` hostname, port and path are used: those components structurally
    cannot carry the ``user:pass@`` userinfo, so no redaction PATTERN is needed —
    and a pattern is exactly what tends to rot. The scheme is kept because
    ``postgresql+psycopg`` vs anything else is the driver assertion this fleet
    cares about, and it carries no secret.

    Fails CLOSED: a malformed DSN renders ``"<unparseable>"`` rather than falling
    back to the raw URL, because a redactor that reverts to the original on a
    parse error leaks precisely when it is least expected. A log line must also
    never be able to break service startup, so nothing here raises.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return "<unparseable>"
    if not host:
        return "<unparseable>"
    where = f"{host}:{port}" if port is not None else host
    dbname = parts.path.lstrip("/")
    if dbname:
        where = f"{where}/{dbname}"
    return f"{parts.scheme}://{where}" if parts.scheme else where
