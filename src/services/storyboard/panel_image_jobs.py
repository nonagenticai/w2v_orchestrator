"""Fire-and-forget job surface for storyboard panel image generation.

Every one of the 29 storyboard routes is synchronous today, and that is fine for
LLM drafts (seconds). An image generation is minutes, so it cannot ride the
request — hence a job. This is deliberately the SMALLEST job surface that works:

* **No table, no migration.** The job record is a short-lived Redis key with a
  TTL. It is progress metadata, not a business record — the durable outcome is
  the panel's own ``media_ref`` / ``gen``, which is what the board reads. A
  dropped Redis key costs the operator a page refresh, not an image.
* **No new stream.** Storyboard rows are keyed on ``project_id`` only (no
  ``run_id``), so ``runs:{run_id}`` is unusable. Progress is published to the
  EXISTING ``projects:{project_id}`` channel that ``GET /api/projects/{id}/events``
  already streams, exactly like the child-lifecycle events do.
* **No new executor.** Tasks are tracked in a dict and drained by ``shutdown()``,
  the same shape ``RunDispatcher`` uses, so a pod SIGTERM cancels them instead of
  orphaning a DB connection.

The tasks run in whichever replica accepted the POST, while the job record lives
in Redis, so a poll served by any replica sees the truth.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid7

from pydantic import ValidationError
from redis.exceptions import RedisError

from src.api.panel_image_url import proxied_job
from src.api.schemas import PanelImageJobResponse
from src.database.session import bind_session_tenant
from src.services.storyboard.render import (
    PanelImageError,
    PanelImageOptions,
    StoryboardPanelImageService,
)

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.clients.ai_content_generation_client import AIContentGenerationClient

log = logging.getLogger(__name__)

#: How long a finished job stays pollable. Long enough for a browser that was
#: closed mid-generation to come back and read the terminal state; short enough
#: that Redis never becomes an accidental job archive.
JOB_TTL_SECONDS = 3600

_KEY_PREFIX = "storyboard:panel-image"

#: SSE event name on the ``projects:{project_id}`` channel.
PANEL_IMAGE_EVENT = "panel_image_status"


def _key(job_id: str) -> str:
    return f"{_KEY_PREFIX}:{job_id}"


class PanelImageJobStore:
    """Redis-backed job records, TTL-bounded.

    The record IS the wire model (:class:`PanelImageJobResponse`) — one typed
    thing rather than a dict that gets validated at the route boundary, so a
    field can never drift between what the runner writes and what the client
    reads.

    ``redis`` must be the RAW ``redis.asyncio`` client, not the w2v_common
    ``RedisClient`` wrapper: this store uses redis-py's ``ex=`` TTL keyword,
    reads the stored JSON back verbatim for ``model_validate_json``, and relies
    on ``RedisError`` propagating so the ``except`` clauses below are live. The
    route unwraps once, via :func:`src.cache.raw.raw_redis`.
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def create(self, *, panel_id: UUID, project_id: UUID) -> PanelImageJobResponse:
        record = PanelImageJobResponse(
            job_id=str(uuid7()),
            panel_id=panel_id,
            project_id=project_id,
            status="queued",
            updated_at=datetime.now(UTC).isoformat(),
        )
        await self._write(record)
        return record

    async def get(self, job_id: str) -> PanelImageJobResponse | None:
        try:
            raw = await self._redis.get(_key(job_id))
        except (RedisError, OSError) as exc:
            log.warning("panel-image job %s unreadable: %s", job_id, exc)
            return None
        if raw is None:
            return None
        try:
            return PanelImageJobResponse.model_validate_json(raw)
        except ValidationError:
            # A record written by an older/newer pod shape is unreadable, not a
            # 500: the panel row still carries the real outcome.
            log.warning("panel-image job %s has an unreadable record", job_id)
            return None

    async def save(self, record: PanelImageJobResponse) -> None:
        record.updated_at = datetime.now(UTC).isoformat()
        await self._write(record)

    async def _write(self, record: PanelImageJobResponse) -> None:
        # Best-effort: the panel row is the durable outcome, so a Redis blip must
        # not abort a generation that is already paid for. It only costs the
        # client its progress view.
        try:
            await self._redis.set(_key(record.job_id), record.model_dump_json(), ex=JOB_TTL_SECONDS)
        except (RedisError, OSError) as exc:
            log.warning("failed to persist panel-image job %s: %s", record.job_id, exc)


class PanelImageJobRunner:
    """Owns the in-flight panel-image tasks for this process.

    ``redis`` is the RAW ``redis.asyncio`` client (see
    :class:`PanelImageJobStore`); it is used for the job record, for the
    progress ``publish``, and is handed to
    :class:`~src.services.storyboard.render.StoryboardPanelImageService` for the
    cross-replica submit lock, which wants the raw client for the same reasons.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[Any],
        redis: Redis,
        client: AIContentGenerationClient | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._redis = redis
        self._client = client
        self.store = PanelImageJobStore(redis)
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def submit(
        self,
        *,
        panel_id: UUID,
        project_id: UUID,
        tenant_id: UUID,
        options: PanelImageOptions,
    ) -> PanelImageJobResponse:
        record = await self.store.create(panel_id=panel_id, project_id=project_id)
        task = asyncio.create_task(
            self._run(record, tenant_id=tenant_id, options=options),
            name=f"panel-image-{record.job_id}",
        )
        self._tasks[record.job_id] = task
        task.add_done_callback(lambda _t, jid=record.job_id: self._tasks.pop(jid, None))
        return record

    async def shutdown(self) -> None:
        """Cancel + await every in-flight task (pod SIGTERM / test teardown)."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            # ``asyncio.wait`` retrieves neither result nor exception, so a task
            # that died on its own never re-raises into teardown.
            await asyncio.wait(tasks)
        self._tasks.clear()

    async def wait_for(self, job_id: str) -> None:
        """Await one job's task if it is still running here (tests + graceful drain).

        Uses ``asyncio.wait`` for the same reason ``shutdown`` does: the caller
        wants "it finished", and the outcome is read from the job record.
        """
        task = self._tasks.get(job_id)
        if task is not None:
            await asyncio.wait([task])

    # ─── the task body ───────────────────────────────────────────────────
    async def _run(
        self, record: PanelImageJobResponse, *, tenant_id: UUID, options: PanelImageOptions
    ) -> None:
        record.status = "running"
        await self._advance(record, tenant_id)
        try:
            async with self._session_factory() as session:
                # RLS is enforced in prod and binds per transaction; a background
                # unit of work must bind its tenant BEFORE its first query or the
                # policies admit zero rows and the panel reads as "not found".
                await bind_session_tenant(session, tenant_id)
                # Redis is passed for the cross-replica submission lock, NOT for
                # the job record (that is `self.store`'s business). Without it a
                # second replica would submit concurrently — see
                # panel_submit_lock's module docstring.
                service = StoryboardPanelImageService(
                    session, client=self._client, redis=self._redis
                )
                result = await service.generate(
                    record.panel_id, tenant_id=tenant_id, options=options
                )
                await session.commit()
        except asyncio.CancelledError:
            # Deliberately no Redis write: we are inside a cancellation (pod
            # SIGTERM / teardown), so any further await would just raise again
            # and swallow the cancel. The record's TTL reaps the stale "running".
            raise
        except PanelImageError as exc:
            log.warning("panel image job %s failed: %s", record.job_id, exc)
            await self._fail(record, str(exc), tenant_id)
        except Exception as exc:
            log.exception("panel image job %s crashed", record.job_id)
            await self._fail(record, f"{type(exc).__name__}: {exc}", tenant_id)
        else:
            record.status = "completed"
            record.media_ref = result.media_ref
            record.gen = result.gen
            await self._advance(record, tenant_id)

    async def _fail(self, record: PanelImageJobResponse, error: str, tenant_id: UUID) -> None:
        record.status = "failed"
        record.error = error
        await self._advance(record, tenant_id)

    async def _advance(self, record: PanelImageJobResponse, tenant_id: UUID) -> None:
        await self.store.save(record)
        await self._publish(record, tenant_id)

    async def _publish(self, record: PanelImageJobResponse, tenant_id: UUID) -> None:
        """Best-effort progress push on ``projects:{project_id}``.

        Notifications are not source of truth — the panel row is. A Redis
        failure here is logged and swallowed; EventSource reconnects and the
        client resyncs from the panel/job fetch.

        A panel image this tenant owns is published as the browser-reachable proxy
        path rather than the stored ``minio.test`` one — this event is consumed
        by an SPA that renders it directly, and that host does not resolve there.
        Any other url (an external image; a ref outside the project's own prefix)
        is published unchanged, exactly as ``panel_image_url.proxied_job`` decides
        for the poll route. The record Redis holds is untouched either way.

        ``tenant_id`` is the one the job was submitted for — the same value
        ``_run`` binds the RLS session to.
        """
        payload = proxied_job(record, tenant_id=tenant_id).model_dump(mode="json")
        payload["event"] = PANEL_IMAGE_EVENT
        try:
            await self._redis.publish(f"projects:{record.project_id}", json.dumps(payload))
        except (RedisError, OSError) as exc:
            log.warning("failed to publish panel-image event for %s: %s", record.job_id, exc)
