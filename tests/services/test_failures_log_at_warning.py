"""Failure paths must be VISIBLE in production — level-pinning tests.

The fleet runs ``LOG_LEVEL=WARNING``. Anything a failure path emits at INFO or
DEBUG does not exist in prod: the 484-count 404 storm ran for two months partly
because its only trace was an INFO line nobody could see. So each site below is
pinned by its numeric level, not by "something was logged" — a test that only
asserts ``caplog.records`` stays green when someone drops the level back to
DEBUG, which is precisely the regression this file exists to catch.

``caplog.set_level(logging.DEBUG, logger=...)`` on purpose: the handler must
accept DEBUG so a *regressed* (lowered) level still produces a record, and the
assertion then fails on ``levelno`` rather than on an empty ``records`` list.
Catching the regression as "wrong level" instead of "nothing logged" is what
makes the failure message point at the actual defect.
"""

from __future__ import annotations

import logging
import sys
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from src.observability import tracing as tracing_module
from src.services import compile_cancel as compile_cancel_module
from src.services import tts_storage as tts_storage_module
from src.services.compile_cancel import cancel_compiles_for_run
from src.services.stages.brand_context import resolve_brand_context
from src.services.tts_storage import find_existing_clip, find_existing_media


def _only(caplog: pytest.LogCaptureFixture, needle: str) -> logging.LogRecord:
    """The single captured record whose message contains ``needle``."""
    hits = [r for r in caplog.records if needle in r.getMessage()]
    assert len(hits) == 1, f"expected exactly one {needle!r} record, got {hits!r}"
    return hits[0]


# ── sites 2 + 3: tts_storage resume probes — indeterminate, not absent ───────


def _exploding_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``_get_storage`` raise, i.e. a genuine probe FAILURE.

    Not the same thing as a miss: absence returns ``None`` from ``get_bytes``
    and leaves the ``except`` untouched, so this branch only ever fires on a
    real defect (transport down, corrupt sidecar) — which is why it can afford
    WARNING without flooding.
    """

    async def _boom() -> Any:
        raise RuntimeError("minio unreachable")

    monkeypatch.setattr(tts_storage_module, "_get_storage", _boom)


async def test_find_existing_clip_indeterminate_logs_at_warning(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _exploding_storage(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="src.services.tts_storage")

    assert await find_existing_clip("run-1", "narrate", "clip-1") is None

    record = _only(caplog, "find_existing_clip(run-1/narrate/clip-1) indeterminate")
    assert record.levelno == logging.WARNING
    assert "minio unreachable" in record.getMessage()


async def test_find_existing_media_indeterminate_logs_at_warning(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _exploding_storage(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="src.services.tts_storage")

    assert await find_existing_media("run-1", "ken_burns", "media-1", "mp4") is None

    record = _only(caplog, "find_existing_media(run-1/ken_burns/media-1) indeterminate")
    assert record.levelno == logging.WARNING
    assert "minio unreachable" in record.getMessage()


# ── site 4: brand_context — a stage failure that read as a routine skip ──────


async def test_brand_extraction_failure_logs_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = AsyncMock()
    client.extract_brand.side_effect = RuntimeError("analysis_service 503")

    caplog.set_level(logging.DEBUG, logger="src.services.stages.brand_context")
    assert await resolve_brand_context("https://acme.test", client=client) == (None, None)

    record = _only(caplog, "brand extraction skipped for https://acme.test")
    assert record.levelno == logging.WARNING, (
        "a swallowed extractor exception means the redesign silently fell back to "
        f"the hardcoded blue; at {record.levelname} nobody in prod ever sees it"
    )
    assert "analysis_service 503" in record.getMessage()


# ── site 5: compile_cancel — client close failure (leaks a connection pool) ──


class _UncloseableClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def cancel_run_jobs(self, run_id: str) -> int:
        return 0

    async def close(self) -> None:
        raise RuntimeError("event loop is closed")


async def test_compile_cancel_close_failure_logs_at_warning(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compile_cancel_module, "VideoCompileClient", _UncloseableClient)
    caplog.set_level(logging.DEBUG, logger="src.services.compile_cancel")

    assert await cancel_compiles_for_run(uuid7()) == 0

    record = _only(caplog, "compile-cancel client close failed for run")
    assert record.levelno == logging.WARNING, (
        "a failed close leaks an httpx connection pool per cancelled run; the "
        f"traceback was already captured and then discarded at {record.levelname}"
    )
    assert record.exc_info is not None, "exc_info must survive the level change"


# ── site 6: tracing silently disabled ───────────────────────────────────────


async def test_tracing_disabled_logs_at_warning(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    # A ``None`` entry in sys.modules makes the import machinery raise
    # ImportError without touching the real (installed) package.
    monkeypatch.setitem(sys.modules, "opentelemetry", None)

    caplog.set_level(logging.DEBUG, logger="src.observability.tracing")
    assert tracing_module.setup_tracing() is False

    record = _only(caplog, "OpenTelemetry SDK not installed")
    assert record.levelno == logging.WARNING, (
        "running with no tracing at all is a degraded mode an operator must be "
        f"able to see; at {record.levelname} the pod looks fully instrumented"
    )
