"""Preview-proxy sweeper — the missing WRITER for ``video_sources.preview_key``.

An upload commits with ``preview_status='pending'`` and three NULL derivative
keys; ``footage_probe`` already presigns ``preview_key`` into ``preview_url``,
``shot_analysis`` already PREFERS that proxy over the 4K original, and the
frontend already renders a click-to-play ``<video>`` off it. Nothing produced the
file, so every row in the fleet sat at ``pending`` and the first gate a reviewer
sees showed metadata and no footage. This module is that producer, and it is the
only thing the feature was missing — no schema, contract or frontend change.

Shape decisions worth not re-deriving:

**A sweeper loop, not a request-time background task.** ``BackgroundTasks`` has
zero occurrences in this repo and would tie the transcode's lifetime to the
request that happened to upload the clip. This is the same lifespan-started
``asyncio.create_task`` + ``while True`` idiom as ``reconciler`` and
``audit_prune``, and it is what lets a pod restart pick up work an earlier pod
never finished — including the 137 rows that predate this code.

**The claim is an advisory lock PLUS a durable lease.** It was the lock alone
until 2026-08-22, on the reasoning that a crashed pod drops its connection,
Postgres releases the lock, and the next tick re-picks the row — no new column and
no expiry arithmetic. That reasoning had a hole big enough to wedge the sweeper:
the attempt counter was a Python local written only AFTER the transcode returned
or raised, so a pod SIGKILLed mid-ffmpeg (OOMKill of PID 1, node eviction/drain,
node reboot) wrote NOTHING. The row came back ``pending`` with its counter
unchanged and was re-picked forever, and ``max_attempts`` — enforced only on the
caught-exception path — could never engage for that class. The counter bump and
``video_sources.preview_leased_until`` (migration 0065) are now written and
COMMITTED before ffmpeg starts, the resolver skips rows leased into the future,
and the cap is also checked at claim time. There is still no reaper to write: a
dead pod's lease EXPIRES, and the row it re-offers has its attempt already counted.
The lock's remaining job is to cheaply de-duplicate the scan; the conditional
``WHERE`` on that committed claim is what actually fences two pods, because the
commit itself releases the lock. Deliberately the ``_xact_`` variant and NOT
session-level: the pods talk to
Postgres through PgBouncer in ``pool_mode=transaction`` (see the leaked-GUC banner
in ``src.database.session``), where a session-level lock outlives the transaction
on a SHARED server connection and a later ``pg_advisory_unlock`` may land on a
different one — leaking a lock that would block that source id forever. The price
is that the claiming transaction stays open for the duration of the transcode;
that is bounded by ``preview_proxy_timeout_seconds`` and by the semaphore below,
and if a server-side ``idle_in_transaction_session_timeout`` kills it the row is
simply left ``pending`` and retried, because the status flip is the LAST statement.
(Since the durable claim above, the transaction no longer spans the transcode at
all — the claim commits first — so that timeout is not even reachable here.)

**Retries are bounded in the existing ``probe`` JSONB.** ``preview_worker.attempts``
rides the blob the upload already writes (``models.VideoSource.probe``); on
exhaustion the row gets ``preview_status='failed'`` plus ``error``, which
``footage_probe`` already surfaces as a visible note and the frontend already
renders in a danger tone.

**RLS.** The sweep spans all tenants on a session with no tenant bound, so the
scan goes through the ``pending_video_sources`` SECURITY DEFINER resolver
(migration 0064) — the exact pattern ``orphan_runs()`` established. Every WRITE
is then issued under ``bind_session_tenant`` with the tenant the resolver
returned, so the data-plane policy is unchanged and still the thing that decides
what the worker may touch.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

# RUNTIME import, deliberately not under TYPE_CHECKING: pydantic resolves
# ``PendingSource``'s annotations when the model is first used, and a
# TYPE_CHECKING-only ``UUID`` makes it "not fully defined" at that moment.
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import text

from src.config import get_settings
from src.utils import safe_exc_message

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

# How many proxies this pod builds at once. A MEMORY-SAFETY bound, not a
# throughput knob, and 2 for the same measured reason ``shot_analysis._CONCURRENCY``
# is 2: one 4K HEVC ffmpeg decode peaks at ~444 MiB RSS, cost is linear in job
# count, 3 concurrent measured safe inside a 4Gi limit and 4 measured OOMKill.
# Explicitly NOT derived from replica count — a Service does not promise an even
# split, so "2 per pod" is the only number that survives a request landing badly,
# and the replica count of this deployment has drifted before with no HPA in the
# manifest. Module-level (not per-tick) so a slow tick overlapping the next one
# still shares one budget rather than doubling it.
_CONCURRENCY = 2
_SEMAPHORE = asyncio.Semaphore(_CONCURRENCY)

# ffmpeg writes here and the object is stored under this leaf inside the source's
# own prefix. ``.mp4`` + h264 + AAC is the point of the whole exercise: the
# originals are hevc/hvc1 and browser HEVC support is unreliable, which is why
# "just redirect the reviewer at the original" is not a substitute.
_PREVIEW_LEAF = "preview.mp4"
_PREVIEW_CONTENT_TYPE = "video/mp4"


class _FileStorage(Protocol):
    """The streaming large-artifact pair this module needs from storage."""

    async def get_file(self, key: str, local_path: str) -> bool: ...
    async def put_file(self, key: str, local_path: str, content_type: str = ...) -> str: ...
    #: Metadata HEAD (``stat_object``) — bytes-free and milliseconds. The disk
    #: budget has to be checked BEFORE the download, so this is the only member of
    #: the trio that must not transfer anything.
    async def object_size(self, key: str) -> int | None: ...


class PreviewProxyError(RuntimeError):
    """A proxy could not be built for one source. Carries the operator-facing text
    written to ``video_sources.error`` on the final attempt."""


class PendingSource(BaseModel):
    """One row of ``pending_video_sources(:limit)``.

    A pydantic model rather than a raw ``Row``/``dict[str, Any]``: the resolver is
    a JSON-free but still untyped SQL boundary, and this repo's ``Any`` ratchet is
    lower-only, so the narrow happens once here instead of at every field read.
    """

    source_id: UUID
    tenant_id: UUID
    project_id: UUID
    object_key: str
    attempts: int


class SweepTally(BaseModel):
    """Per-tick outcome counts (returned so tests assert on the DECISION, not on
    log plumbing)."""

    ready: int = 0
    retry: int = 0
    failed: int = 0
    skipped: int = 0
    contended: int = 0
    #: Terminal writes the ownership fence rejected — this worker's lease had
    #: expired and someone else owns the row's verdict now. Its OWN counter, not
    #: folded into ``skipped``: ``skipped`` means "never started the work", while
    #: this means "did the whole expensive transcode and threw the verdict away",
    #: and only the second one tells you the lease is too short.
    lost_lease: int = 0


def _preview_key(source: PendingSource) -> str:
    """Object key for this source's proxy — the SAME tenant/project-confined tree
    the original lives in.

    ``_source_prefix`` is imported rather than re-spelled: that prefix is what
    keeps one tenant's footage out of another's listing, and it is declared
    never-expired in the gitops minio-lifecycle CronJob. A second hand-written
    copy here would be a second thing to get wrong AND a key the retention job
    does not know about.
    """
    from src.api.routes.uploads_video import _source_prefix

    return _source_prefix(source.tenant_id, source.project_id, source.source_id) + _PREVIEW_LEAF


def _ffmpeg_args(src_path: str, out_path: str) -> list[str]:
    """Single-pass 4K/HEVC → 720p h264 proxy.

    ``scale=-2:min(720\\,ih)`` never UPSCALES an already-small clip (and ``-2``
    keeps the width even, which h264 requires); ``-threads 2`` matches the CPU
    quota this pod actually has, so ffmpeg does not spawn ``nproc`` frame-threads
    each holding a full 4K DPB; ``+faststart`` moves the moov atom to the front so
    the reviewer's ``<video>`` can start playing on the first range request
    instead of after the whole file. ``0:a?`` is optional — silent drone footage
    is the common case and a missing audio stream must not fail the transcode.
    """
    return [
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        src_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        "scale=-2:'min(720,ih)'",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "26",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-threads",
        "2",
        out_path,
    ]


async def _transcode(src_path: str, out_path: str, timeout_seconds: int) -> None:
    """Run ffmpeg, or raise ``PreviewProxyError``.

    Every failure is raised, not swallowed: unlike ``_probe_video`` (whose bytes
    are already durably stored by the time it runs) this IS the unit of work, and
    a silent failure would leave the row ``pending`` forever with no attempt
    counted. Only the stderr tail reaches the message — never the argv — so a
    path never lands in an operator-visible column.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise PreviewProxyError("no ffmpeg on PATH")
    proc = await asyncio.create_subprocess_exec(
        ffmpeg,
        *_ffmpeg_args(src_path, out_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(timeout_seconds):
            _, stderr = await proc.communicate()
    except TimeoutError as exc:
        # Reap it: an abandoned ffmpeg on a pathological file would keep decoding
        # for the pod's lifetime AND hold this row's advisory lock with it.
        proc.kill()
        await proc.wait()
        raise PreviewProxyError(f"transcode timed out after {timeout_seconds}s") from exc
    if proc.returncode != 0:
        tail = stderr.decode("utf-8", "replace").strip()[-300:]
        raise PreviewProxyError(f"ffmpeg exited {proc.returncode}: {tail}")


def _make_staging_dir() -> str:
    """``mkdtemp`` behind a typed wrapper — ``tempfile.mkdtemp`` is overloaded on
    its prefix type and comes back as ``str | bytes`` through ``to_thread``."""
    return tempfile.mkdtemp(prefix="preview-proxy-")


async def _build_proxy(source: PendingSource, timeout_seconds: int) -> str:
    """Download the original, transcode it, upload the proxy; return its key.

    The staging directory is reclaimed in a ``finally`` whether or not the
    transcode raised — a leaked 134 MiB original plus its partial output per
    attempt fills the pod's ephemeral disk within a single shoot, exactly as the
    upload path's own ``finally`` exists to prevent.
    """
    from src.services.tts_storage import _get_storage

    # ``_get_storage`` is typed as ``_StorageLike`` — deliberately the NARROW
    # tts-shaped Protocol, because every in-memory storage fake in the test suite
    # is structurally checked against it and widening it there breaks 20-odd of
    # them. The live singleton is a ``StorageService``, which does have the
    # streaming file pair; ``_FileStorage`` states exactly the two methods this
    # module needs and the cast is the narrow. (``uploads_video`` solves the same
    # mismatch with ``getattr(storage, "put_file", None)`` + a 503; a background
    # loop has no request to 503, so a missing method surfaces as a counted,
    # retried attempt instead.)
    storage = cast("_FileStorage", await _get_storage())
    pipeline = get_settings().pipeline
    max_bytes = pipeline.preview_proxy_max_source_bytes
    transfer_timeout = pipeline.preview_proxy_transfer_timeout_seconds

    # ═══ DISK BUDGET, CHECKED BEFORE A SINGLE BYTE MOVES ═══
    # A metadata HEAD, not a transfer. The worker stages the original AND the proxy
    # side by side in the pod's /tmp emptyDir at _CONCURRENCY=2, so an oversized
    # source does not fail cheaply: it fills the volume, takes the OTHER concurrent
    # job and every unrelated /tmp writer in the pod down with it, and — if it is
    # larger than the volume — can NEVER succeed, burning every attempt until the
    # row lands permanently 'failed' with an ffmpeg/ENOSPC error that says nothing
    # about the real cause. Refusing up front turns a physical impossibility into
    # one clear sentence an operator can act on.
    size = await storage.object_size(source.object_key)
    if size is None:
        raise PreviewProxyError("source object is missing from storage")
    if size > max_bytes:
        raise PreviewProxyError(
            f"source object is {size} bytes, above the {max_bytes}-byte preview-proxy "
            "budget; the worker must stage the original and its proxy on the pod's "
            "ephemeral disk, so this clip cannot be transcoded here no matter how "
            "many times it is retried. Raise preview_proxy_max_source_bytes together "
            "with the /tmp volume sizeLimit, or supply a smaller master."
        )

    tmpdir = await asyncio.to_thread(_make_staging_dir)
    try:
        src_path = str(Path(tmpdir) / "source")
        out_path = str(Path(tmpdir) / _PREVIEW_LEAF)
        # get_file, never get_object_bytes: streaming straight to disk is the only
        # way to hand a 4K original to a subprocess without an OOMKill.
        #
        # BOTH transfers are wall-clock bounded. fget_object/fput_object have no
        # timeout of their own, so a stalled transfer previously hung the worker
        # forever — past the row's lease, which is precisely how a second worker
        # ends up transcoding the same row while the first is still holding a stale
        # verdict. `asyncio.timeout` raises the builtin TimeoutError, which the
        # caller's `except Exception` counts as a normal failed attempt.
        async with asyncio.timeout(transfer_timeout):
            found = await storage.get_file(source.object_key, src_path)
        if not found:
            raise PreviewProxyError("source object is missing from storage")
        await _transcode(src_path, out_path, timeout_seconds)
        key = _preview_key(source)
        async with asyncio.timeout(transfer_timeout):
            await storage.put_file(key, out_path, _PREVIEW_CONTENT_TYPE)
        return key
    finally:
        await asyncio.to_thread(shutil.rmtree, tmpdir, True)


_CLAIM_SQL = "SELECT pg_try_advisory_xact_lock(hashtext(:k))"
_RECHECK_SQL = "SELECT preview_status FROM video_sources WHERE id = :id"
# ``probe || jsonb_build_object(...)`` MERGES: the ffprobe blob the upload wrote is
# the cut planner's input and must survive every attempt counter bump.
_ATTEMPTS_JSONB = (
    "probe || jsonb_build_object('preview_worker', jsonb_build_object('attempts', :attempts))"
)
# How much longer than the transcode itself the lease runs. Derived from
# ``timeout_seconds`` at call time and NEVER a second absolute number: a hard-coded
# lease would silently become SHORTER than the timeout the moment
# ``preview_proxy_timeout_seconds`` is raised in gitops, and a lease that expires
# while ffmpeg is still running re-offers a row that is actively being transcoded —
# reintroducing the duplicate-work bug the lease exists to prevent. The margin
# covers the download and the upload that bracket ffmpeg inside ``_build_proxy``.
#
# It is DERIVED from the two transfer timeouts rather than being a flat 300, for the
# same reason the lease is derived from the transcode timeout: the margin exists to
# cover the download and the upload that bracket ffmpeg, and now that both of those
# are individually bounded, their bound is exactly the number that makes the lease
# strictly longer than this worker's worst-case runtime. A flat constant smaller
# than the transfer budget would expire the lease while a slow-but-healthy transfer
# was still legitimately running — re-offering the row and making the ownership
# fence fire on ordinary work instead of on genuine lease loss.
_LEASE_MARGIN_MULTIPLIER = 2  # one download + one upload


def _lease_seconds(timeout_seconds: int, transfer_timeout_seconds: int) -> int:
    """Worst-case wall clock for one attempt: two bounded transfers + ffmpeg."""
    return timeout_seconds + _LEASE_MARGIN_MULTIPLIER * transfer_timeout_seconds


# The DURABLE claim, and the whole point of this file's 2026-08-22 change. It bumps
# the attempt counter AND stamps the lease in ONE statement, which the caller
# COMMITS BEFORE ffmpeg starts — so a pod SIGKILLed mid-transcode (OOMKill of PID 1,
# node drain, node reboot) leaves a counted attempt behind instead of a row that is
# re-picked forever with the cap unable to engage.
#
# Committing releases ``pg_try_advisory_xact_lock`` (it is transaction-scoped), so
# the WHERE clause — not the lock — is what keeps a second pod off the row: it
# re-asserts BOTH ``pending`` and "not currently leased" at write time, and
# ``RETURNING id`` tells the caller whether it actually won. A row that a sibling
# claimed between the scan and here yields no row and is reported as skipped.
_CLAIM_LEASE_SQL = (
    f"UPDATE video_sources SET probe = {_ATTEMPTS_JSONB}, "  # noqa: S608
    "preview_leased_until = now() + make_interval(secs => :lease_seconds), "
    "updated_at = now() WHERE id = :id AND preview_status = 'pending' "
    "AND (preview_leased_until IS NULL OR preview_leased_until < now()) "
    "RETURNING id, preview_leased_until"
)
# Text for a row whose attempts were already exhausted at CLAIM time. Distinct from
# the per-attempt ffmpeg error on purpose: it is the operator-visible fingerprint of
# a pod that died mid-transcode, which produces no exception and therefore no error
# string of its own.
_EXHAUSTED_AT_CLAIM_ERROR = (
    "preview proxy exhausted {max_attempts} attempts; the previous attempts ended "
    "without recording an outcome (the worker pod most likely died mid-transcode)"
)
# S608 on the statements below is a false positive: the only interpolation is
# ``_ATTEMPTS_JSONB``, itself a module constant of hard-coded column names, and every
# value (:key/:attempts/:error/:id/:lease_seconds) is a BOUND parameter, never
# string-formatted.
#
# Every terminal write CLEARS the lease. A finished row must never read as leased:
# ``ready``/``failed`` rows are no longer offered by the resolver anyway, but a
# ``retry`` row that kept its lease would sit out however much of the transcode
# budget was left before anyone could look at it again.
#
# ═══ OWNERSHIP FENCE ═══
# Every terminal write below re-asserts ``preview_leased_until = :lease``, the exact
# lease timestamp THIS worker's claim wrote and got back via ``RETURNING``. Without
# it the terminal writes predicated on ``id`` alone while only the CLAIM re-asserted
# ownership, so a worker whose lease had already expired still landed its verdict:
#
#   A claims -> A's lease expires mid-transcode -> B claims and finishes 'ready'
#   -> A's ffmpeg finally errors -> A writes 'failed' + an error string OVER B's
#   'ready', while ``preview_key`` still points at B's perfectly good proxy.
#
# The reviewer then sees "preview failed" for a preview that plays. Worse,
# ``_MARK_RETRY_SQL`` from A also CLEARED B's live lease, re-offering a row B was
# actively transcoding and inviting a third concurrent transcode of the same source.
#
# The lease timestamp IS the claim token — no new column needed. It is unique per
# claim by construction: a second claim can only be granted once the first lease has
# EXPIRED (the claim's own WHERE says so), so the two ``now() + interval`` values are
# necessarily distinct. A fenced write from an expired claim matches 0 rows, which
# the caller LOGS rather than swallows: a lost race is information about lease
# sizing, not noise.
_LEASE_FENCE = " AND preview_leased_until = :lease"
_MARK_READY_SQL = (
    "UPDATE video_sources SET preview_key = :key, preview_status = 'ready', error = NULL, "  # noqa: S608
    f"probe = {_ATTEMPTS_JSONB}, preview_leased_until = NULL, updated_at = now() WHERE id = :id"
    + _LEASE_FENCE
)
_MARK_RETRY_SQL = (
    f"UPDATE video_sources SET probe = {_ATTEMPTS_JSONB}, error = :error, "  # noqa: S608
    "preview_leased_until = NULL, updated_at = now() WHERE id = :id" + _LEASE_FENCE
)
_MARK_FAILED_SQL = (
    "UPDATE video_sources SET preview_status = 'failed', error = :error, "  # noqa: S608
    f"probe = {_ATTEMPTS_JSONB}, preview_leased_until = NULL, updated_at = now() WHERE id = :id"
    + _LEASE_FENCE
)
# The exhausted-at-CLAIM retirement runs BEFORE any lease of ours exists, so it
# cannot use the fence above. It is still fenced — on the same predicate the claim
# itself uses — so it can never retire a row a live worker is mid-transcode on, nor
# re-retire one somebody else already finished.
_MARK_EXHAUSTED_SQL = (
    "UPDATE video_sources SET preview_status = 'failed', error = :error, "  # noqa: S608
    f"probe = {_ATTEMPTS_JSONB}, preview_leased_until = NULL, updated_at = now() "
    "WHERE id = :id AND preview_status = 'pending' "
    "AND (preview_leased_until IS NULL OR preview_leased_until < now())"
)


async def _process_source(
    session_factory: async_sessionmaker[AsyncSession],
    source: PendingSource,
    *,
    max_attempts: int,
    timeout_seconds: int,
) -> str:
    """Claim one row, build its proxy, record the outcome.

    Returns 'ready' | 'retry' | 'failed' | 'skipped' | 'contended'.

    Its OWN session, because the advisory lock is a property of the connection
    holding it — two rows worked concurrently on one session would share (and
    commit away) each other's claim.
    """
    async with _SEMAPHORE, session_factory() as session:
        locked = (await session.execute(text(_CLAIM_SQL), {"k": str(source.source_id)})).scalar()
        if not locked:
            return "contended"
        return await _work_locked_source(
            session, source, max_attempts=max_attempts, timeout_seconds=timeout_seconds
        )


async def _work_locked_source(
    session: AsyncSession,
    source: PendingSource,
    *,
    max_attempts: int,
    timeout_seconds: int,
) -> str:
    """The body of ``_process_source`` once the row's advisory lock is held."""
    from src.database.session import bind_session_tenant

    await bind_session_tenant(session, source.tenant_id)
    # Re-read UNDER the lock. Between the resolver's scan and this claim another
    # pod may have finished the row; without this a second worker would rebuild
    # and re-upload a proxy that already exists, and (worse) reset a 'failed' row
    # nobody asked it to retry.
    status = (
        await session.execute(text(_RECHECK_SQL), {"id": str(source.source_id)})
    ).scalar_one_or_none()
    if status != "pending":
        await session.commit()
        return "skipped"

    # Enforce the cap HERE, not only on the caught-exception path. A pod killed
    # mid-transcode raises nothing, so the old cap could not see that class at all;
    # with the durable claim below, such a row comes back with its attempt already
    # counted and must eventually be retired rather than retried forever.
    if source.attempts >= max_attempts:
        result = await session.execute(
            text(_MARK_EXHAUSTED_SQL),
            {
                "id": str(source.source_id),
                "attempts": source.attempts,
                "error": _EXHAUSTED_AT_CLAIM_ERROR.format(max_attempts=max_attempts),
            },
        )
        await session.commit()
        if _rows_affected(result) == 0:
            # The fence rejected us: between the resolver's scan and this write a
            # sibling worker claimed or finished the row. Reporting "failed" and
            # logging "giving up" here would ASSERT a retirement that never
            # happened — the same lie every other terminal write in this file
            # already refuses to tell. ``skipped``, not ``lost_lease``: no lease of
            # ours existed and no transcode was burned, which is exactly the
            # distinction the two counters carry.
            logger.info(
                "preview proxy did NOT retire source=%s at %d attempts: the row was "
                "no longer pending-and-unleased, so another worker owns its verdict",
                source.source_id,
                source.attempts,
            )
            return "skipped"
        logger.warning(
            "preview proxy giving up on source=%s after %d attempts with no recorded outcome",
            source.source_id,
            source.attempts,
        )
        return "failed"

    attempts = source.attempts + 1
    # DURABLE CLAIM, committed before a single frame is decoded. This commit ends the
    # transaction holding the advisory lock; the conditional WHERE in
    # ``_CLAIM_LEASE_SQL`` is what carries mutual exclusion from here on, and the
    # ``after_begin`` RLS listener re-binds this tenant for the transaction the
    # terminal write below runs in (``bind_session_tenant`` recorded it in
    # ``session.info``).
    claim_row = (
        await session.execute(
            text(_CLAIM_LEASE_SQL),
            {
                "id": str(source.source_id),
                "attempts": attempts,
                "lease_seconds": _lease_seconds(
                    timeout_seconds,
                    get_settings().pipeline.preview_proxy_transfer_timeout_seconds,
                ),
            },
        )
    ).one_or_none()
    await session.commit()
    if claim_row is None:
        # A sibling pod won the row between the resolver's scan and this write.
        return "skipped"
    # The lease we were granted — our claim token for every terminal write below.
    lease = claim_row.preview_leased_until

    try:
        key = await _build_proxy(source, timeout_seconds)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 -- one bad clip must not stop the sweep
        # The PreviewProxyError branch is deliberately untouched: its text is
        # authored HERE (argv and paths excluded, ffmpeg stderr capped at 300
        # chars), so "ffmpeg exited 1: moov atom not found" still reaches the
        # operator intact. Only the generic branch carries foreign exception text.
        reason = str(exc) if isinstance(exc, PreviewProxyError) else _operator_safe_reason(exc)
        exhausted = attempts >= max_attempts
        sql = _MARK_FAILED_SQL if exhausted else _MARK_RETRY_SQL
        result = await session.execute(
            text(sql),
            {
                "id": str(source.source_id),
                "attempts": attempts,
                "error": reason,
                "lease": lease,
            },
        )
        await session.commit()
        if _rows_affected(result) == 0:
            return _lost_race(source, attempts, "failed" if exhausted else "retry", reason)
        logger.warning(
            "preview proxy attempt %d/%d failed for source=%s: %s",
            attempts,
            max_attempts,
            source.source_id,
            reason,
        )
        return "failed" if exhausted else "retry"

    result = await session.execute(
        text(_MARK_READY_SQL),
        {"id": str(source.source_id), "key": key, "attempts": attempts, "lease": lease},
    )
    await session.commit()
    if _rows_affected(result) == 0:
        # Our proxy is still in storage under the deterministic ``_preview_key`` and
        # is byte-identical to whatever the winner uploaded, so dropping this write
        # loses nothing but the wasted CPU.
        return _lost_race(source, attempts, "ready", None)
    logger.info("preview proxy ready: source=%s key=%s", source.source_id, key)
    return "ready"


#: Credential-bearing URL QUERY parameters — the gap ``safe_exc_message`` does not
#: cover. That helper scrubs DSN userinfo, ``Bearer …`` and
#: ``password|token|api_key=…``; MEASURED against a real presigned MinIO URL, it
#: leaves ``X-Amz-Credential`` and ``X-Amz-Signature`` fully intact:
#:
#:   safe_exc_message(Exception("S3 error on https://…?X-Amz-Credential=AKIA…"))
#:     -> "S3 error on https://…?X-Amz-Credential=AKIA…"   (unchanged)
#:
#: which is exactly the shape this call site produces: ``_build_proxy`` talks to
#: MinIO, and a storage/transport exception routinely carries the presigned URL it
#: failed on. SigV4 puts the access key id and the signature in the query string, so
#: that string is a live credential.
_URL_CREDENTIAL_PARAM = re.compile(
    r"(?i)([?&][\w.-]*(?:signature|credential|security-token|access[_-]?key|token|secret|password|sig)[\w.-]*=)[^&\s]+"
)


def _operator_safe_reason(exc: Exception) -> str:
    """The text written to ``video_sources.error`` for a NON-``PreviewProxyError``.

    That column is not an operator-only surface: ``footage_probe`` interpolates it
    verbatim into a stage note and the frontend renders that note verbatim in the
    review gate, so this string reaches a human's browser. The raw
    ``f"{type(exc).__name__}: {exc}"`` it replaces was both UNBOUNDED (a large repr
    became an unbounded row and an unbounded DOM string) and UNSCRUBBED.

    REUSES ``safe_exc_message`` rather than reimplementing it — it already handles
    DSN userinfo, ``Bearer …``, ``password|token|api_key=…``, the empty-``str`` case
    (a bare ``TimeoutError()`` stringifies to ``""``, so it falls back to ``repr``),
    and the 500-char bound. The query-parameter pass is layered on top for the
    credentials that helper misses; it runs AFTER, so a signature that survived at
    the truncation boundary is still redacted.

    The exception TYPE is kept: it is the single most useful diagnostic token and
    is a Python identifier, not attacker-controlled text.
    """
    # ``safe_exc_message`` lives in ``src.utils`` (stdlib-only) precisely so this
    # background sweeper can import it at module level. It used to be the PRIVATE
    # ``src.middleware.audit._safe_exc_message``, reached by a function-local import
    # to keep the Starlette middleware stack out of this loop's import graph — which
    # worked, but made a rename fail only at exception time in production.
    return (
        f"{type(exc).__name__}: {_URL_CREDENTIAL_PARAM.sub(r'\1[redacted]', safe_exc_message(exc))}"
    )


def _rows_affected(result: object) -> int:
    """``rowcount`` off an UPDATE result, as an int.

    The repo-wide idiom (``run_repository``, ``stage_output_repository``,
    ``storyboard_repository``): ``AsyncSession.execute`` is typed as returning
    ``Result``, which has no ``rowcount`` — only the ``CursorResult`` a DML
    statement actually produces does. ``getattr`` keeps that honest without a
    ``cast`` that would lie about the static type or a ``# type: ignore``.
    """
    return int(getattr(result, "rowcount", 0) or 0)


def _lost_race(source: PendingSource, attempts: int, intended: str, reason: str | None) -> str:
    """Report a terminal write that the ownership fence rejected (0 rows).

    Reached when this worker's lease expired mid-transcode and another worker
    re-claimed the row: our verdict is stale and must NOT overwrite theirs. Logged
    rather than swallowed — a steady stream of these means
    ``preview_proxy_timeout_seconds`` + the lease margin are under-sized for the
    real footage, which is a tuning signal you cannot get any other way.
    """
    logger.warning(
        "preview proxy DISCARDED a stale '%s' verdict for source=%s (attempt %d): the "
        "lease had expired and another worker re-claimed the row, so this write was "
        "fenced off at 0 rows. Underlying reason: %s",
        intended,
        source.source_id,
        attempts,
        reason if reason is not None else "transcode succeeded",
    )
    return "lost_lease"


_SCAN_SQL = (
    "SELECT source_id, tenant_id, project_id, object_key, attempts FROM pending_video_sources(:lim)"
)


async def _scan_pending(
    session_factory: async_sessionmaker[AsyncSession], batch_size: int
) -> list[PendingSource]:
    """Cross-tenant scan of pending sources via the SECURITY DEFINER resolver.

    A short, self-contained transaction: it must NOT stay open across the
    transcodes, or the scan would hold a connection for the whole tick.
    """
    async with session_factory() as session:
        rows = (await session.execute(text(_SCAN_SQL), {"lim": batch_size})).all()
        await session.commit()
    return [
        PendingSource(
            source_id=row.source_id,
            tenant_id=row.tenant_id,
            project_id=row.project_id,
            object_key=row.object_key,
            attempts=row.attempts,
        )
        for row in rows
    ]


def _tally_outcome(tally: SweepTally, outcome: str) -> None:
    """Fold one row's outcome into the tick tally. Explicit rather than
    ``setattr``: an outcome string that stopped matching a field would silently
    vanish from the counts instead of failing loudly."""
    match outcome:
        case "ready":
            tally.ready += 1
        case "retry":
            tally.retry += 1
        case "failed":
            tally.failed += 1
        case "skipped":
            tally.skipped += 1
        case "contended":
            tally.contended += 1
        case "lost_lease":
            tally.lost_lease += 1
        case _:  # pragma: no cover - unreachable; guards a future outcome string
            raise ValueError(f"unknown preview-proxy outcome {outcome!r}")


async def sweep_pending_previews(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    batch_size: int,
    max_attempts: int,
    timeout_seconds: int,
) -> SweepTally:
    """One tick: claim up to ``batch_size`` pending sources and build their proxies."""
    sources = await _scan_pending(session_factory, batch_size)
    tally = SweepTally()
    if not sources:
        return tally
    outcomes = await asyncio.gather(
        *(
            _process_source(
                session_factory,
                source,
                max_attempts=max_attempts,
                timeout_seconds=timeout_seconds,
            )
            for source in sources
        ),
        return_exceptions=True,
    )
    for source, outcome in zip(sources, outcomes, strict=True):
        if isinstance(outcome, asyncio.CancelledError):
            # ``return_exceptions=True`` COLLECTS a cancelled child instead of
            # propagating it. During a pod drain that would be silently tallied as
            # a retry and the loop would look like it completed a tick; re-raise so
            # the lifespan's cancel actually unwinds this task.
            raise outcome
        if isinstance(outcome, BaseException):
            # A row whose bookkeeping itself blew up (DB blip mid-UPDATE). It stays
            # `pending` and is re-claimed next tick — never allowed to take the rest
            # of the batch down with it.
            #
            # ``exc_info=outcome`` is LOAD-BEARING, not decoration. ``gather`` RETURNS
            # the exception here rather than raising it, so there is no exception
            # being handled — ``logger.exception()`` (which reads ``sys.exc_info()``)
            # renders NOTHING and this line is a bare sentence with the cause thrown
            # away. That blinded a live incident on 2026-08-23: the sweep logged this
            # message once per row for half an hour and the actual error
            # (``ReadOnlySqlTransaction`` from a PgBouncer backend carrying a leaked
            # ``default_transaction_read_only=on``) could only be recovered by
            # re-running the work by hand inside a pod.
            logger.error(
                "preview proxy bookkeeping failed for source=%s",
                source.source_id,
                exc_info=outcome,
            )
            tally.retry += 1
            continue
        _tally_outcome(tally, outcome)
    logger.info(
        "preview proxy sweep: ready=%d retry=%d failed=%d skipped=%d contended=%d lost_lease=%d",
        tally.ready,
        tally.retry,
        tally.failed,
        tally.skipped,
        tally.contended,
        tally.lost_lease,
    )
    return tally


async def _run_preview_proxy_loop(
    session_factory: async_sessionmaker[AsyncSession],
    interval_seconds: int,
    batch_size: int,
    max_attempts: int,
    timeout_seconds: int,
) -> None:
    """Sweep for pending previews every ``interval_seconds`` for the pod's lifetime.

    A single bad tick (transient DB/storage error) must never kill the loop — the
    whole point is that a row nobody has managed to transcode yet is retried by
    whichever pod is alive later.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await sweep_pending_previews(
                session_factory,
                batch_size=batch_size,
                max_attempts=max_attempts,
                timeout_seconds=timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("preview proxy tick failed; continuing")


def _start_preview_proxy_task(
    session_factory: async_sessionmaker[AsyncSession],
) -> asyncio.Task[None] | None:
    """Spawn the preview-proxy sweeper, or ``None`` when disabled.

    Disabled by default — see ``PipelineSettings.preview_proxy_enabled`` for why
    (ffmpeg RSS versus this pod's memory limit).
    """
    settings = get_settings()
    if not settings.pipeline.preview_proxy_enabled:
        return None
    return asyncio.create_task(
        _run_preview_proxy_loop(
            session_factory,
            settings.pipeline.preview_proxy_interval_seconds,
            settings.pipeline.preview_proxy_batch_size,
            settings.pipeline.preview_proxy_max_attempts,
            settings.pipeline.preview_proxy_timeout_seconds,
        ),
        name="preview-proxy",
    )
