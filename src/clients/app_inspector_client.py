"""HTTP client for App Inspector Service (port 6311)."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar

import httpx

from src.clients.base import LicensedServiceClient
from src.clients.schemas.app_inspector import InspectionResult
from src.services.stages.upstream import classify

# /result 409s (Retry-After: 30) until the job is completed/awaiting_selection.
# poll_job already blocks until a terminal-success status, so a 409 here is only
# a brief status/result write-race — bounded-retry it rather than fail the stage.
_RESULT_MAX_ATTEMPTS = 3
_RESULT_RETRY_SLEEP_SECONDS = 2.0

#: The ``device_farm`` body keys that carry a credential in clear.
_DEVICE_FARM_SECRET_KEYS = ("browserstack_access_key",)
#: w2v_common's client logger: it logs every >=400 response body BEFORE raising, and
#: its name-based redaction (``\baccess[_-]?key``) misses ``browserstack_access_key``,
#: so a pydantic 422 echoing the farm config would log the key verbatim.
_W2V_CLIENT_LOGGER = "w2v_common.clients.base"
_WITHHELD = "[response body withheld: the request carried device-farm credentials]"
_REDACTED = "[REDACTED]"
#: ``... returned 422: <body>`` — everything after the status is the upstream body.
_STATUS_BODY_RE = re.compile(r"(returned \d{3}): .*", re.DOTALL)

#: Secrets in flight on THIS task's inspection (a ContextVar, so a concurrent
#: inspection without a farm is untouched and the filter needs no lock).
_ACTIVE_SECRETS: ContextVar[tuple[str, ...]] = ContextVar(
    "app_inspector_device_farm_secrets", default=()
)


def _scrub(text: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        text = text.replace(secret, _REDACTED)
    return text


class _DeviceFarmSecretLogFilter(logging.Filter):
    """While a device-farm secret is in flight, never let a client log line carry it.

    A >=WARNING record (the ``returned <status>: <body>`` line) loses its body
    outright — a TRUNCATED body can end in a partial key that no value-match finds —
    and loses its traceback (the exception message embeds the same body)."""

    def filter(self, record: logging.LogRecord) -> bool:
        secrets = _ACTIVE_SECRETS.get()
        if not secrets:
            return True
        message = record.getMessage()
        if record.levelno >= logging.WARNING:
            message = _STATUS_BODY_RE.sub(rf"\1: {_WITHHELD}", message)
            record.exc_info = None
            record.exc_text = None
        record.msg = _scrub(message, secrets)
        record.args = None
        return True


def _install_log_filter() -> None:
    target = logging.getLogger(_W2V_CLIENT_LOGGER)
    if not any(isinstance(f, _DeviceFarmSecretLogFilter) for f in target.filters):
        target.addFilter(_DeviceFarmSecretLogFilter())


_install_log_filter()


def _device_farm_secrets(config: Mapping[str, object] | None) -> tuple[str, ...]:
    farm = config.get("device_farm") if config else None
    if not isinstance(farm, Mapping):
        return ()
    return tuple(
        value
        for key in _DEVICE_FARM_SECRET_KEYS
        if isinstance(value := farm.get(key), str) and value
    )


class DeviceFarmInspectionError(RuntimeError):
    """An inspection that carried device-farm credentials failed upstream.

    Status code + a generic sentence only: the upstream body (a pydantic 422 echoes
    the request's ``device_farm`` object, key included) is never carried. A
    ``RuntimeError`` so the per-competitor degrade in ``app_review/inspect.py`` still
    catches it."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@asynccontextmanager
async def _guard_device_farm_secrets(secrets: tuple[str, ...]) -> AsyncIterator[None]:
    """Sanitize every failure (raised AND logged) of a call that sent ``secrets``."""
    if not secrets:
        yield
        return
    token = _ACTIVE_SECRETS.set(secrets)
    try:
        yield
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        raise DeviceFarmInspectionError(
            f"app_inspector returned {status} for an inspection on the tenant device "
            f"farm; {_WITHHELD}",
            status_code=status,
        ) from None
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        if any(secret in text for secret in secrets):
            raise DeviceFarmInspectionError(
                f"app_inspector inspection on the tenant device farm failed: "
                f"{_scrub(text, secrets)}"
            ) from None
        raise
    finally:
        _ACTIVE_SECRETS.reset(token)


class AppInspectorClient(LicensedServiceClient):
    """Client for the App Inspector microservice."""

    def __init__(
        self,
        base_url: str | None = None,
        license_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            service_name="app_inspector",
            base_url=base_url,
            license_token=license_token,
            timeout=timeout,
        )

    async def start_inspection(self, url: str, config: dict | None = None) -> str:
        """Start an inspection job, returns job_id."""
        payload: dict[str, object] = {"app_path": url}
        if config:
            # Map config keys to InspectRequest fields instead of nesting as "config"
            for key in (
                "platform",
                "automation_backend",
                "max_screens",
                "enable_recording",
                "require_selection",
                "device_farm",
            ):
                if key in config:
                    payload[key] = config[key]
        async with _guard_device_farm_secrets(_device_farm_secrets(config)):
            return await self.submit_job("/api/inspect", payload)

    async def _fetch_result(self, job_id: str) -> InspectionResult:
        """GET the FAT ``/result`` body and type it.

        Constructed deterministically — the status body's ``result_url`` is null
        for ``partial``/``completed_with_errors`` (server only sets it on
        ``completed``), so trusting it would drop the result. Bounded-retries a
        transient 409 (result not yet readable) and CLASSIFIES anything else
        through ``stages/upstream.classify`` instead of re-raising it raw.
        """
        last_exc: httpx.HTTPStatusError | None = None
        for attempt in range(_RESULT_MAX_ATTEMPTS):
            try:
                raw = await self._request("GET", f"/api/inspect/{job_id}/result")
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 409:
                    # Everything that is NOT the 409 write-race goes through the ONE
                    # policy in stages/upstream.py rather than escaping as a bare
                    # HTTPStatusError that every caller then re-interprets. A 402
                    # becomes PermanentUpstreamError carrying the real upstream
                    # sentence; a 503/429 becomes TransientUpstreamError, so a caller
                    # that retries knows it is worth retrying.
                    raise classify(exc) from exc
                last_exc = exc
                await asyncio.sleep(_RESULT_RETRY_SLEEP_SECONDS * (attempt + 1))
                continue
            return InspectionResult.model_validate(raw)
        raise RuntimeError(
            f"app_inspector /result for {job_id} still 409 after {_RESULT_MAX_ATTEMPTS} attempts"
        ) from last_exc

    async def get_inspection(self, job_id: str) -> InspectionResult:
        """Fetch the full, typed inspection result for a completed job."""
        return await self._fetch_result(job_id)

    async def wait_for_inspection(self, job_id: str, timeout: float = 300.0) -> InspectionResult:
        """Poll until the job reaches a terminal state, then fetch the FAT result.

        ``poll_job`` GETs the THIN status endpoint until the job completes (or
        raises on a failed job); the inspection data itself lives at ``/result``.
        """
        await self.poll_job("/api/inspect", job_id, timeout=timeout)
        return await self._fetch_result(job_id)

    async def inspect_app(self, url: str, config: dict | None = None) -> InspectionResult:
        """Inspect an app — submit and wait for the full result."""
        async with _guard_device_farm_secrets(_device_farm_secrets(config)):
            job_id = await self.start_inspection(url, config)
            return await self.wait_for_inspection(job_id)
