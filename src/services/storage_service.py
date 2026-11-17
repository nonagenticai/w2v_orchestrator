"""Pipeline artifact storage — a thin domain adapter over the shared
``w2v_common.storage.StorageService`` facade.

This module used to carry a full hand-rolled local+MinIO backend (it was, in
fact, the ancestor the facade was consolidated from). It now keeps only the
orchestrator's *domain* surface — run/stage-scoped key shaping
(``artifacts/{job}/{stage}/...``, ``videos/{job}/final.mp4``), ``file://`` URL
construction for the local backend, the cross-run media-cache lifecycle GC, and
the browser-download bucket predicate — and delegates every object-store
primitive to the shared facade. Public API + reference formats are unchanged, so
no call site changes.

A handful of operations still touch the facade's MinIO client directly because
the shared facade deliberately does not model them: the best-effort
``media-cache/`` bucket lifecycle rule (``_ensure_media_cache_lifecycle``,
init-time), ``close_session`` (``close``, for one-shot scripts), and the chunked
upload's three reaches — server-side ``compose_object``, the streaming
``get_file`` download-to-path, and the ``object_size`` metadata HEAD that makes
``complete`` idempotent against an already-composed object. They all reach the
client through the SAME accessor (``_minio_client_or_raise`` / the guarded
``getattr`` in the two best-effort paths); do not add another way in. The hot per-request streaming path
(the artifact proxy) uses the facade's first-class :meth:`get_object_stream`
instead of reaching in.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from w2v_common.storage import StorageService as _FacadeStorage

from src.config import get_settings

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    # Type-only: the lifecycle classes are imported LAZILY at their single call
    # site (see ``_ensure_media_cache_lifecycle``) so a ``local``-backend consumer
    # never needs ``miniopy-async`` installed. Naming them here costs nothing at
    # runtime (``from __future__ import annotations`` above) and lets the
    # lifecycle helpers below be typed without ``Any``, which ANN401 rejects in
    # this file.
    from miniopy_async.lifecycleconfig import LifecycleConfig, Rule
    from w2v_common.storage import StreamedObject


class _ObjectStat(Protocol):
    """The one field compose validation reads off a ``stat_object`` result."""

    size: int


class _MinioClientLike(Protocol):
    """The narrow slice of ``miniopy_async.Minio`` this module reaches for.

    A structural Protocol rather than the concrete class (or a bare ``Any``,
    which ANN401 rightly rejects for this file): ``miniopy-async`` is imported
    lazily so a ``local``-backend consumer never needs it installed, and naming
    the real type at module scope would make this module unimportable there.
    Declaring only the four operations we use also documents the blast radius of
    the escape hatch — if this Protocol grows, the reach into the facade's
    private client is growing with it, and that is the moment to push the
    operation into ``w2v_common.storage`` instead.
    """

    async def fget_object(self, bucket_name: str, object_name: str, file_path: str) -> object: ...

    async def stat_object(self, bucket_name: str, object_name: str) -> _ObjectStat: ...

    async def compose_object(
        self,
        bucket_name: str,
        object_name: str,
        sources: Sequence[object],
        metadata: dict[str, str] | None = None,
    ) -> object: ...


logger = logging.getLogger(__name__)

# Object-key prefix for the P3b cross-run, tenant-scoped media cache. Single
# source of truth shared with ``tts_storage._media_cache_key`` (the writer) and
# the lifecycle rule below (the GC). Mirrors ``sound_catalog.LIBRARY_ROOT``.
MEDIA_CACHE_ROOT = "media-cache"

#: The one lifecycle rule id this service OWNS. Everything else in the bucket's
#: lifecycle configuration belongs to somebody else (see
#: ``_ensure_media_cache_lifecycle`` for why that matters) and is preserved
#: verbatim. Changing this string orphans the previously-written rule rather than
#: updating it — the old id would then read as "foreign" and be kept forever.
MEDIA_CACHE_LIFECYCLE_RULE_ID = "w2v-media-cache-expiry"

#: S3 error codes that mean "this bucket simply has no lifecycle configuration",
#: which is a normal empty state, NOT a failure. ``miniopy-async``'s
#: ``get_bucket_lifecycle`` swallows ``NoSuchLifecycleConfiguration`` and returns
#: ``None``, but the sync ``minio`` client (and older/other S3 implementations)
#: raise it instead, so the read path below handles both shapes. Any OTHER error
#: means we do not know what is in the bucket, and a blind write would then be
#: destructive — see the read-failure branch.
_ABSENT_LIFECYCLE_CODES = frozenset(
    {"NoSuchLifecycleConfiguration", "NoSuchLifecycleConfig", "NoSuchLifecyclePolicy"}
)


def _is_absent_lifecycle_error(exc: BaseException) -> bool:
    """True when ``exc`` means "no lifecycle configuration exists" (not an error)."""
    code = getattr(exc, "code", None)
    return isinstance(code, str) and code in _ABSENT_LIFECYCLE_CODES


def _rule_id_of(rule: object) -> str | None:
    """The ``rule_id`` of a lifecycle rule, tolerating a client that omits it."""
    rule_id = getattr(rule, "rule_id", None)
    return rule_id if isinstance(rule_id, str) else None


def _media_cache_rule_matches(rule: object, ttl_days: int) -> bool:
    """True when an existing rule already says exactly what we would write.

    Compared field-by-field rather than with ``==`` because ``miniopy_async``'s
    ``Rule`` does not define ``__eq__`` — two structurally identical rules are
    never equal objects, so an identity/equality check would report "changed" on
    every startup and turn the no-op path into a write on every pod boot. That is
    the whole point of the check: each write is another chance to lose somebody
    else's rule (below), so we take that chance only when something differs.
    """
    expiration = getattr(rule, "expiration", None)
    rule_filter = getattr(rule, "rule_filter", None)
    return (
        getattr(rule, "status", None) == "Enabled"
        and getattr(rule_filter, "prefix", None) == f"{MEDIA_CACHE_ROOT}/"
        and getattr(expiration, "days", None) == ttl_days
        and getattr(expiration, "date", None) is None
    )


class _LifecycleClientLike(Protocol):
    """The single bucket-lifecycle READ the media-cache GC reaches for.

    A structural Protocol for the same reason as :class:`_MinioClientLike`:
    ``miniopy-async`` is imported lazily, so the concrete client type cannot be
    named at module scope, and ``Any`` is not available here (ANN401 is enforced
    in this file). The WRITE (``set_bucket_lifecycle``) stays on the caller's
    side, where the client object is already in hand.
    """

    async def get_bucket_lifecycle(self, bucket_name: str) -> LifecycleConfig | None: ...


async def _read_existing_lifecycle_rules(
    client: _LifecycleClientLike, bucket: str
) -> list[Rule] | None:
    """The bucket's current lifecycle rules, or ``None`` meaning **abstain**.

    ``[]`` and ``None`` are deliberately different answers. ``[]`` is "the bucket
    has no lifecycle configuration", which is a normal empty state we can safely
    write over. ``None`` is "we do not KNOW what is in the bucket" — and a write
    from a position of ignorance is exactly the destructive act the caller exists
    to avoid, because ``set_bucket_lifecycle`` REPLACES the whole configuration
    and would drop rules we never saw. The caller returns without writing; the
    cache simply does not GC until the next boot.

    "Absent" arrives in two shapes depending on the client: ``None`` (this client
    swallows ``NoSuchLifecycleConfiguration``) or that S3Error raised. Both mean
    "empty bucket policy", not failure. Any OTHER read error — or a config object
    whose ``rules`` we cannot read as a sequence, which is the same epistemic
    position — abstains.
    """
    try:
        current = await client.get_bucket_lifecycle(bucket)
    except Exception as exc:  # noqa: BLE001 -- best-effort, never fatal
        if not _is_absent_lifecycle_error(exc):
            logger.warning(
                "media-cache lifecycle: could not READ existing config on %s (%s); "
                "refusing to write, because set_bucket_lifecycle REPLACES the whole "
                "configuration and would drop rules we cannot see",
                bucket,
                exc,
            )
            return None
        current = None

    raw_rules = getattr(current, "rules", None) if current is not None else None
    if current is not None and not isinstance(raw_rules, list | tuple):
        logger.warning(
            "media-cache lifecycle: existing config on %s has unreadable rules (%r); "
            "refusing to write rather than replace rules we cannot enumerate",
            bucket,
            type(raw_rules).__name__,
        )
        return None
    return list(raw_rules or [])


def _merge_media_cache_rule(existing_rules: list[Rule], desired: Rule) -> list[Rule]:
    """``existing_rules`` with ``desired`` swapped in for the rule we own.

    Positional replacement (rather than "filter ours out, append") keeps the
    document byte-stable across restarts, so a diff of the bucket policy shows
    only what actually changed. Every foreign rule is carried through as the SAME
    object, never a rebuilt copy. Appended only when the bucket does not already
    carry our id — two rules with one id is an invalid document, and appending
    blindly would leave the OLD ttl live alongside the new one.
    """
    merged = [
        desired if _rule_id_of(rule) == MEDIA_CACHE_LIFECYCLE_RULE_ID else rule
        for rule in existing_rules
    ]
    if not any(_rule_id_of(rule) == MEDIA_CACHE_LIFECYCLE_RULE_ID for rule in existing_rules):
        merged.append(desired)
    return merged


#: Minimum size of every compose source part EXCEPT the last one. This is an
#: S3/MinIO server-side-copy constraint (each non-final source becomes a
#: multipart part, and a multipart part may not be smaller than 5 MiB), NOT a
#: preference of ours — see :class:`ComposeConstraintError` for why we check it
#: ourselves instead of letting MinIO answer.
MIN_COMPOSE_PART_BYTES = 5 * 1024 * 1024

#: Maximum number of source objects in a single compose. S3 multipart allows
#: 10,000 parts; a compose with more sources is rejected server-side. At the
#: 32 MiB client chunk size the chunked-upload route uses, this ceiling is ~312
#: GiB, so hitting it means the caller is chunking far too finely — that is a
#: caller bug worth surfacing loudly rather than a limit to work around.
MAX_COMPOSE_PARTS = 10_000


def _s3_error_types() -> tuple[type[BaseException], ...]:
    """The ``miniopy_async`` error class to catch for "object is not there".

    Resolved once at import, mirroring ``w2v_common.storage.service._MINIO_EXC``:
    ``S3Error`` is not an ``OSError``, and the import is lazy because a
    ``local``-backend consumer never installs ``miniopy-async``. An empty tuple is
    a legal ``except`` target, so the call sites need no ``if`` around it.
    """
    try:
        from miniopy_async.error import S3Error
    except ImportError:  # pragma: no cover -- a minio-backend pod always has it
        return ()
    return (S3Error,)


_S3_ERRORS: tuple[type[BaseException], ...] = _s3_error_types()


class ComposeConstraintError(ValueError):
    """A :meth:`StorageService.compose_object` call violates an S3/MinIO
    server-side-compose constraint (part too small, too many parts, no parts, or
    a source object that is not there).

    Raised **before** any compose request is issued, and that timing is the whole
    point. MinIO only reports ``EntityTooSmall`` / ``InvalidPart`` when it tries
    to assemble the object — i.e. after the client has already spent minutes
    PUTting every chunk of a 134 MB source clip over a metered link. Learning
    "part 3 was 4 MiB" at that moment is unactionable: the upload is spent and the
    only recovery is to redo all of it. Failing here instead lets the route
    reject a bad chunk plan at the *first* part, or (for a mis-sized final
    assembly) return a precise, human-readable 4xx naming the offending key.

    It is a ``ValueError`` subclass deliberately: this is always a caller/plan
    bug, never a transport condition, so a retry loop must NOT treat it as
    transient.
    """


def _assert_compose_part_sizes(source_keys: list[str], sizes: list[int]) -> None:
    """Enforce the S3 compose part-size floor across ``source_keys``.

    Every source EXCEPT the last must be at least :data:`MIN_COMPOSE_PART_BYTES`.
    The final part is exempt (a multipart upload's last part may be any size) —
    which is exactly why the check is positional and why a caller must never
    reorder parts to "fix" a size complaint: moving the short tail into the middle
    turns a legal compose into an illegal one.

    Applied to BOTH backends on purpose. The floor is meaningless to a filesystem
    concat, but a developer running ``STORAGE_BACKEND=local`` must hit the same
    wall they would hit in prod; a guard that only fires against real MinIO is a
    guard that ships broken chunk plans.
    """
    for index, (source_key, size) in enumerate(zip(source_keys[:-1], sizes[:-1], strict=True)):
        if size < MIN_COMPOSE_PART_BYTES:
            raise ComposeConstraintError(
                f"compose source part {index} ({source_key!r}) is {size} bytes; every part "
                f"except the last must be >= {MIN_COMPOSE_PART_BYTES} bytes "
                f"({MIN_COMPOSE_PART_BYTES // (1024 * 1024)} MiB) or MinIO rejects the "
                "compose with EntityTooSmall after the whole upload has been paid for"
            )


def is_readable_w2v_bucket(bucket: str, own_bucket: str) -> bool:
    """Buckets the orchestrator user may read / re-sign / proxy.

    True for the service's own configured bucket, or **any** ``w2v-*`` artifact
    bucket — narration/slide/AI-image (``w2v-orchestrator``), redesign sections
    (``w2v-html-render``), compiled videos (``w2v-compiled-videos``), course
    stitches (``w2v-courses``), audiobooks (``w2v-audiobooks``), etc. The
    orchestrator's MinIO user can read every w2v bucket, so re-signing or
    proxying any of them with our own credentials is safe.

    Single source of truth shared by the compile-side re-sign path and
    ``artifact_proxy`` (the browser-download gate), so the two can never drift —
    the historical drift is exactly why the delivered audiobook 404'd through the
    proxy while it re-signed fine on the compile side. Tenant isolation is
    enforced separately (the proxy checks run ownership; the re-sign path checks
    the object lives in our MinIO host).
    """
    return bool(bucket) and (bucket == own_bucket or bucket.startswith("w2v-"))


class StorageService:
    """Manages pipeline artifact storage (local or MinIO/S3) over the shared
    ``w2v_common`` facade. Public API + reference formats are unchanged."""

    def __init__(self) -> None:
        settings = get_settings()
        self._backend = settings.storage.backend
        self._local_path = Path(settings.storage.local_path)
        self._minio_bucket = settings.storage.minio_bucket
        self._media_cache_cross_run = getattr(settings.storage, "media_cache_cross_run", False)
        self._media_cache_ttl_days = getattr(settings.storage, "media_cache_ttl_days", 30)
        self._facade = _FacadeStorage(settings.storage)

    async def initialize(self) -> None:
        """Initialize the storage backend (facade) + the media-cache GC rule."""
        await self._facade.initialize()
        if self._backend == "minio":
            await self._ensure_media_cache_lifecycle()

    async def _ensure_media_cache_lifecycle(self) -> None:
        """Best-effort: expire the cross-run media cache under ``media-cache/``.

        The P3b cross-run cache stores content-addressed media under a
        ``media-cache/`` prefix that no run owns (runs are soft-deleted and never
        reclaim media, and a ``_MEDIA_GEN_VERSION`` bump orphans old ids), so a
        bucket lifecycle expiration is the only reclamation path. Only applied
        when the cache is enabled and a positive TTL is set. Never fatal to
        startup — a lifecycle-set failure (permissions, older MinIO) must not
        keep the service from serving; the cache still works, it just won't GC.

        Uses the facade's initialized MinIO client: the shared facade does not
        model bucket-lifecycle rules, and this is a one-time, best-effort,
        orchestrator-specific GC config — not a hot path.

        ⚠️ **``set_bucket_lifecycle`` REPLACES the bucket's whole lifecycle
        configuration — it does not merge.** S3 models the configuration as one
        document, so PUTting a single-rule document deletes every other rule in
        the bucket. This function therefore READS the current configuration first
        and writes back ``foreign rules + ours``.

        That is not theoretical. As of 2026-08-18 ``w2v-orchestrator`` carries a
        single operator-applied whole-bucket rule (empty prefix, 60-day expiry,
        MinIO-generated id ``d8129rcus7pm44a84qu0``) that exists in **no git
        repo** — it cannot be recreated from a manifest if we drop it. The same
        bucket holds ``tenants/`` (uploaded source clips) and ``sound_library/``
        (the music beds every reel depends on), so a lost or wrongly-widened rule
        destroys user data. The pre-merge version of this function was dormant
        only because ``media_cache_cross_run`` is currently false; flipping
        ``STORAGE_MEDIA_CACHE_CROSS_RUN=true`` would have silently deleted that
        retention rule at startup, best-effort, with nothing but a warning.

        Hence also the loud logging: the failure mode here is *silence*. Every
        foreign rule id we preserve is logged, so if this ever does drop one, the
        log line before the write names what was there.
        """
        if not self._media_cache_cross_run or self._media_cache_ttl_days <= 0:
            return
        client = getattr(self._facade, "_minio_client", None)
        if client is None:
            return
        try:
            from miniopy_async.lifecycleconfig import Expiration, Filter, LifecycleConfig, Rule
        except ImportError:  # pragma: no cover -- prod always has it
            logger.warning("media-cache lifecycle: miniopy_async lifecycle API unavailable")
            return

        # --- READ the existing configuration -----------------------------
        # ``None`` back means "we could not establish what is in the bucket" —
        # abstain rather than replace. See the helper for the two shapes "absent"
        # arrives in and why an unclassifiable read must not become a write.
        existing_rules = await _read_existing_lifecycle_rules(client, self._minio_bucket)
        if existing_rules is None:
            return
        foreign_ids = [
            _rule_id_of(rule) or "<unnamed>"
            for rule in existing_rules
            if _rule_id_of(rule) != MEDIA_CACHE_LIFECYCLE_RULE_ID
        ]
        if foreign_ids:
            logger.info(
                "media-cache lifecycle: preserving %d rule(s) we do not own on %s: %s",
                len(foreign_ids),
                self._minio_bucket,
                ", ".join(foreign_ids),
            )

        try:
            desired = Rule(
                rule_id=MEDIA_CACHE_LIFECYCLE_RULE_ID,
                status="Enabled",
                rule_filter=Filter(prefix=f"{MEDIA_CACHE_ROOT}/"),
                expiration=Expiration(days=self._media_cache_ttl_days),
            )

            # --- MERGE: replace ours in place, keep everyone else's order ----
            ours = next(
                (r for r in existing_rules if _rule_id_of(r) == MEDIA_CACHE_LIFECYCLE_RULE_ID),
                None,
            )
            if ours is not None and _media_cache_rule_matches(ours, self._media_cache_ttl_days):
                logger.info(
                    "media-cache lifecycle: already correct on %s (%s/ expires after %dd, "
                    "%d foreign rule(s) intact) — skipping write",
                    self._minio_bucket,
                    MEDIA_CACHE_ROOT,
                    self._media_cache_ttl_days,
                    len(foreign_ids),
                )
                return

            merged = _merge_media_cache_rule(existing_rules, desired)

            await client.set_bucket_lifecycle(self._minio_bucket, LifecycleConfig(merged))
            logger.info(
                "media-cache lifecycle set: %s/ expires after %dd (%d rule(s) written, "
                "%d preserved: %s)",
                MEDIA_CACHE_ROOT,
                self._media_cache_ttl_days,
                len(merged),
                len(foreign_ids),
                ", ".join(foreign_ids) or "none",
            )
        except Exception as exc:  # noqa: BLE001 -- GC config is best-effort, never fatal
            logger.warning("media-cache lifecycle: could not set rule (%s)", exc)

    async def close(self) -> None:
        """Close the MinIO client's aiohttp session.

        One-shot scripts (e.g. ``scripts/sync_sound_library.py``) build a
        ``StorageService`` and exit; without this the aiohttp session is GC'd
        unclosed and prints ``Unclosed client session`` warnings. Idempotent and
        a no-op for the local backend / uninitialised client. The shared facade
        does not expose a close, so reach its client for this one shutdown op.
        """
        client = getattr(self._facade, "_minio_client", None)
        if self._backend == "minio" and client is not None:
            await client.close_session()

    # --- Run / stage-scoped artifacts -----------------------------------
    # (module-level helper `guess_artifact_content_type` lives at the bottom)

    async def store_stage_artifact(
        self,
        job_id: str,
        stage_name: str,
        data: bytes,
        filename: str = "output.json",
    ) -> str:
        """Store a stage artifact. Returns the storage path/key.

        The content type is derived from ``filename``, NOT hardcoded. It used to be
        an unconditional ``application/json``, which was true for the default
        ``output.json`` and a lie for everything else this helper stores — the
        ``promo_slides`` PNGs, the ``*_narrate`` audio clips, manuscripts.

        That lie was product-visible, not cosmetic. MinIO persists the declared
        type, the artifact proxy faithfully passes it back, and a browser will NOT
        render an ``<img>`` whose response is ``Content-Type: application/json`` —
        so every slide thumbnail on every run page decoded to nothing while the
        storyline counter happily reported "8/8 visuals". Measured live on run
        019efbb0 (2026-08-09): 8 images, 8 valid PNG bodies, 0 with naturalWidth>0.
        """
        path = f"artifacts/{job_id}/{stage_name}/{filename}"
        return await self._facade.put_bytes(
            path, data, content_type=guess_artifact_content_type(filename)
        )

    async def retrieve_stage_artifact(
        self,
        job_id: str,
        stage_name: str,
        filename: str = "output.json",
    ) -> bytes | None:
        """Retrieve a stage artifact."""
        path = f"artifacts/{job_id}/{stage_name}/{filename}"
        return await self._facade.get_bytes(path)

    async def stage_artifact_url(self, job_id: str, stage_name: str, filename: str) -> str | None:
        """Fetchable URL for a stage artifact WITHOUT (re)writing it.

        Presigned http(s):// for MinIO, ``file://`` for local — so an idempotent
        caller that already knows the object exists (a resumed narrate reusing an
        already-synthesized clip) can reconstruct the URL without re-storing the
        bytes. Does NOT verify existence; the caller gates on a separate presence
        check (the duration sidecar).
        """
        path = f"artifacts/{job_id}/{stage_name}/{filename}"
        if self._backend == "minio":
            return await self._facade.presign_get(path)
        if self._backend == "local":
            return f"file://{self._local_path / path}"
        return None

    async def object_url(self, key: str) -> str | None:
        """Fetchable URL for an arbitrary own-bucket ``key`` WITHOUT (re)writing it.

        The generic, non-run-scoped analog of ``stage_artifact_url``: presigned
        http(s):// for MinIO, ``file://`` for the local backend. Used by the P3b
        cross-run media cache, whose key (``media-cache/tenants/...``) is not the
        ``artifacts/{run}/{stage}/`` shape ``stage_artifact_url`` builds. Does NOT
        verify existence; callers gate on a separate ``object_exists``.
        """
        if self._backend == "minio":
            return await self._facade.presign_get(key)
        if self._backend == "local":
            return f"file://{self._local_path / key}"
        return None

    async def retrieve(self, ref: str) -> bytes | None:
        """Read raw bytes back by the storage reference ``store_*`` returned
        (a MinIO object key, or an absolute/relative local path). Returns
        ``None`` if the object is absent. Read by key, never by an arbitrary URL
        (no SSRF surface)."""
        return await self._facade.get_bytes(ref)

    async def store_final_video(self, job_id: str, video_data: bytes, fmt: str = "mp4") -> str:
        """Store the final compiled video."""
        path = f"videos/{job_id}/final.{fmt}"
        return await self._facade.put_bytes(path, video_data, content_type=f"video/{fmt}")

    # --- Generic keyed-object access (own bucket; NOT run/job-scoped) ---

    async def put_object(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Write ``data`` at an arbitrary object ``key`` in the own bucket."""
        return await self._facade.put_bytes(key, data, content_type=content_type)

    async def put_file(
        self,
        key: str,
        local_path: str | Path,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Write the file already on disk at ``local_path`` to ``key`` — the
        LARGE-artifact path, which never lands the file in RAM.

        The streaming sibling of :meth:`put_object`: the facade ``fput_object``s
        it (chunked multipart, constant memory) for the minio backend and does a
        buffered ``copyfile`` for the local one, so a multi-GB source clip costs
        a fixed buffer instead of its own size in heap. Every ``put_bytes``-based
        helper on this class materialises the whole object first, which is fine
        for a JSON stage artifact or a WAV clip and is exactly what a 4K video
        upload must not do — use this instead whenever the caller already has (or
        can stage) the bytes on disk.
        """
        return await self._facade.put_file(key, local_path, content_type=content_type)

    async def get_object_bytes(self, key: str) -> bytes | None:
        """Read raw bytes at an arbitrary object ``key``, or ``None`` if absent."""
        return await self._facade.get_bytes(key)

    async def get_file(self, key: str, local_path: str) -> bool:
        """Download the object at ``key`` to ``local_path`` on disk. True on
        success, False when the object is absent — the READ mirror of
        :meth:`put_file`, and the only safe way to hand a large object to a
        subprocess.

        NEVER materialises the object in memory: the minio backend uses
        ``fget_object`` (chunked streaming write straight to the file) and the
        local backend a buffered ``copyfile``. The caller for this is the chunked
        upload's post-compose probe, which ``ffprobe``s the assembled source clip
        — a 134 MiB 4K drone file. ``get_object_bytes`` would work functionally and
        is the wrong answer: the orchestrator pod is memory-sized for JSON stage
        artifacts, so one concurrent upload of a real camera file is an OOMKill,
        and OOMKills in this service are indistinguishable from an unrelated crash
        loop when you are reading the events afterwards. If you are tempted to
        "simplify" this into ``local_path.write_bytes(await get_object_bytes(key))``,
        that is the regression this method exists to prevent.

        Absence is a RETURN VALUE, not an exception, because the callers here are
        probes ("did the parts survive the compose?") that must branch, not
        unwind. Genuine transport / permission failures still raise — a
        connectivity outage must not be silently reported as "the object is not
        there", which would let a caller delete parts it never actually read.
        """
        dest = Path(local_path)
        await asyncio.to_thread(dest.parent.mkdir, parents=True, exist_ok=True)

        if self._backend == "local":
            src = Path(key)
            if not src.is_absolute():
                src = self._local_path / key
            if not await asyncio.to_thread(src.is_file):
                return False
            await asyncio.to_thread(shutil.copyfile, src, dest)
            return True

        if self._backend == "minio":
            client = self._minio_client_or_raise()
            try:
                await client.fget_object(self._minio_bucket, key, str(dest))
            except _S3_ERRORS as exc:
                if getattr(exc, "code", None) in ("NoSuchKey", "NoSuchObject"):
                    logger.info("get_file: object absent: %s", key)
                    return False
                raise
            return True

        raise ValueError(f"Unsupported storage backend: {self._backend}")

    async def object_size(self, key: str) -> int | None:
        """Size in bytes of the object at ``key``, or ``None`` when it is absent.

        A metadata HEAD (``stat_object``) — bytes-free and milliseconds — NOT a
        read. It exists for exactly one caller: the chunked upload's ``complete``,
        which must be able to ask "is the final object already there?" before it
        composes.

        Why that question needs answering at all: ``complete`` composes, then
        INSERTs the ``video_sources`` row, then commits. A SIGTERM in that window
        is routine (``maxSurge: 1`` rolling deploys) and rolls the row back while
        the composed object survives. The client retries, and without this probe
        the retry recomposes from parts — which is fine only for as long as the
        parts still exist. Skipping the compose when the object is already there
        makes the retry converge instead of depending on staging that a previous
        attempt may already have reclaimed.

        Deliberately NOT :meth:`object_exists`: that one takes a presigned/absolute
        URL and re-derives ``{bucket}/{key}`` from it, and it answers a TRI-state
        (``None`` = "don't know") because its callers must not fail a run on an
        indeterminate probe. Here the input is a bare own-bucket key we minted
        ourselves, and an indeterminate answer is not survivable as a silent
        "absent": a transport error must RAISE (the caller then 5xx's and the
        client retries) rather than be reported as "not composed yet", which would
        recompose over a good object or, worse, over already-swept parts. So the
        only ``None`` this returns is a genuine ``NoSuchKey``.
        """
        if self._backend == "local":
            path = Path(key)
            if not path.is_absolute():
                path = self._local_path / key
            if not await asyncio.to_thread(path.is_file):
                return None
            return (await asyncio.to_thread(path.stat)).st_size

        if self._backend == "minio":
            client = self._minio_client_or_raise()
            try:
                stat = await client.stat_object(self._minio_bucket, key)
            except _S3_ERRORS as exc:
                if getattr(exc, "code", None) in ("NoSuchKey", "NoSuchObject"):
                    return None
                raise
            return int(stat.size or 0)

        raise ValueError(f"Unsupported storage backend: {self._backend}")

    # --- Server-side compose (chunked upload assembly) -------------------

    def _minio_client_or_raise(self) -> _MinioClientLike:
        """The facade's initialized MinIO client, or a loud failure.

        Typed as the narrow :class:`_MinioClientLike` Protocol, not ``Any`` — the
        facade stores the client as ``Any`` because it is lazily imported, but
        that looseness must not leak into this module (ANN401 is enforced here).

        Same sanctioned escape hatch as ``_ensure_media_cache_lifecycle`` and
        ``close``: the shared ``w2v_common`` facade deliberately models neither
        server-side compose, nor a download-to-path, nor a key-addressed
        ``stat``, so those three reach its client. Deliberately ONE hatch, not a
        second pattern — if the facade grows ``compose``/``fget``/``stat``
        first-class, every reach lives here and in those three methods, and
        nowhere else.

        Unlike the GC/close paths this raises rather than no-ops: those are
        best-effort housekeeping, whereas a compose that quietly did nothing would
        hand the caller a key for an object that does not exist, and the failure
        would surface much later as an unplayable video.
        """
        client = getattr(self._facade, "_minio_client", None)
        if client is None:
            raise RuntimeError(
                "MinIO backend is not initialized; call StorageService.initialize() first"
            )
        return client

    async def compose_object(
        self, key: str, source_keys: list[str], content_type: str = "application/octet-stream"
    ) -> str:
        """Assemble one object at ``key`` from ``source_keys``, server-side.

        Returns the same reference shape every other writer on this class returns
        (the object ``key`` for minio, the absolute path for local), so a caller
        can feed the result straight to :meth:`object_url` / :meth:`get_file`.

        This is the second half of the chunked upload: Cloudflare 413s any body
        over 100 MB, so the client PUTs 32 MiB parts as individual objects and we
        stitch them here. MinIO does this with a server-side multipart copy — the
        bytes never traverse this pod, which is the entire reason we do not simply
        read the parts and re-``put_bytes`` them.

        ⚠️ **``source_keys`` order IS the byte order, and we do NOT sort it.**
        The composed object is the parts concatenated in exactly the order given.
        Do not "helpfully" sort the keys anywhere up the call chain: a
        lexicographic sort puts ``part10`` before ``part2``, which produces an
        object that is the right *size*, uploads fine, and is a shuffled,
        undecodable video. That failure is silent — ``ffprobe`` returns ``{}``, the
        duration lands as 0, and the DAG carries a permanently zeroed clip length
        downstream. The caller must order parts by their numeric index.

        Raises :class:`ComposeConstraintError` — before any compose request — when
        the plan cannot legally be composed (no parts, more than
        :data:`MAX_COMPOSE_PARTS`, a non-final part below
        :data:`MIN_COMPOSE_PART_BYTES`, or a missing source).
        """
        if not source_keys:
            raise ComposeConstraintError("compose_object requires at least one source key")
        if len(source_keys) > MAX_COMPOSE_PARTS:
            raise ComposeConstraintError(
                f"compose_object got {len(source_keys)} source parts; S3/MinIO allows at most "
                f"{MAX_COMPOSE_PARTS}. Re-chunk the upload with a larger part size."
            )

        if self._backend == "local":
            return await self._compose_local(key, source_keys)
        if self._backend == "minio":
            return await self._compose_minio(key, source_keys, content_type)
        raise ValueError(f"Unsupported storage backend: {self._backend}")

    async def _compose_minio(self, key: str, source_keys: list[str], content_type: str) -> str:
        """MinIO server-side compose, size-validated first.

        The ``stat_object`` sweep before the compose is a deliberate extra
        round-trip per part: it is a metadata HEAD (bytes-free, milliseconds) and
        it is the only way to know a part's size without trusting the caller's
        bookkeeping. It buys us the precondition check that turns a post-upload
        ``EntityTooSmall`` into an actionable, key-named error. Do not drop it to
        "save a call" — the call it saves costs nothing and the failure it
        prevents costs the whole upload.
        """
        client = self._minio_client_or_raise()
        try:
            from miniopy_async.commonconfig import ComposeSource
        except ImportError as exc:  # pragma: no cover -- a minio-backend pod always has it
            raise RuntimeError("miniopy-async is required for server-side compose_object") from exc

        sizes: list[int] = []
        for source_key in source_keys:
            try:
                stat = await client.stat_object(self._minio_bucket, source_key)
            except _S3_ERRORS as exc:
                raise ComposeConstraintError(
                    f"compose source {source_key!r} is not readable "
                    f"(code={getattr(exc, 'code', '?')}); refusing to compose a partial object"
                ) from exc
            sizes.append(int(stat.size or 0))

        _assert_compose_part_sizes(source_keys, sizes)

        sources = [ComposeSource(self._minio_bucket, source_key) for source_key in source_keys]
        await client.compose_object(
            self._minio_bucket,
            key,
            sources,
            metadata={"Content-Type": content_type},
        )
        logger.info("Composed %s from %d parts (%d bytes total)", key, len(source_keys), sum(sizes))
        return key

    async def _compose_local(self, key: str, source_keys: list[str]) -> str:
        """Local-backend compose: a streaming concat, in the given order.

        Exists so the chunked-upload route is exercisable without a live MinIO,
        and so ``STORAGE_BACKEND=local`` development sees the same ordering and
        part-size behaviour as prod. Concatenates through a buffered
        ``copyfileobj`` rather than reading parts into memory, for the same reason
        :meth:`get_file` streams: the parts are 32 MiB each and the whole point of
        this feature is files too big to hold.
        """
        resolved: list[Path] = []
        for source_key in source_keys:
            candidate = Path(source_key)
            resolved.append(candidate if candidate.is_absolute() else self._local_path / source_key)

        sizes: list[int] = []
        for source_key, path in zip(source_keys, resolved, strict=True):
            if not await asyncio.to_thread(path.is_file):
                raise ComposeConstraintError(
                    f"compose source {source_key!r} does not exist at {path}; "
                    "refusing to compose a partial object"
                )
            sizes.append((await asyncio.to_thread(path.stat)).st_size)

        _assert_compose_part_sizes(source_keys, sizes)

        dest = self._local_path / key

        def _concat() -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as out:
                for path in resolved:
                    with path.open("rb") as part:
                        shutil.copyfileobj(part, out)

        await asyncio.to_thread(_concat)
        logger.info("Composed %s from %d local parts", dest, len(source_keys))
        return str(dest)

    async def list_object_keys(self, prefix: str) -> list[str]:
        """List object keys under ``prefix`` in the own bucket."""
        return await self._facade.list_prefix(prefix)

    async def delete_object_prefix(self, prefix: str) -> bool:
        """Delete every object under an arbitrary ``prefix`` in the own bucket.

        The generic, non-run-scoped analog of :meth:`delete_job_artifacts` (which
        hardcodes the ``artifacts/{job}/`` shape). Used by resource deletes whose
        objects live under their own key tree — e.g. an uploaded video source's
        ``.../video-sources/{id}/`` folder, which holds the original plus every
        derived preview/poster/sprite. Callers treat the result as best-effort:
        a stranded object is bucket-lifecycle garbage, not a failed request.
        """
        return await self._facade.delete_prefix(prefix)

    async def delete_object(self, key: str) -> bool:
        """Delete ONE object at ``key`` in the own bucket (idempotent: a missing
        object is success). ``False`` on a storage error. Unlike
        :meth:`delete_object_prefix`, never touches a sibling that merely shares
        the key as a prefix."""
        return await self._facade.delete_object(key)

    async def object_exists(self, url: str) -> bool | None:
        """Probe whether the artifact behind ``url`` exists in storage.

        Returns ``True`` (present), ``False`` (definitively absent — the only
        signal a caller may fail a run on), or ``None`` (indeterminate: foreign
        host, missing bucket, permissions/transport error — the caller MUST treat
        as "don't know" and not fail the run). Delegates to the facade, which
        re-derives ``{bucket}/{key}`` and ``stat_object``s with our own creds
        (no HEAD on the SigV4-GET-only presigned URL).
        """
        return await self._facade.object_exists(url)

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        """Generate a presigned GET URL (MinIO only); ``None`` otherwise.

        ``bucket`` defaults to the own bucket; pass an explicit bucket to presign
        an object the orchestrator can read but did not write (e.g.
        ``w2v-compiled-videos`` for the YouTube publish path). The in-cluster host
        is rewritten to ``STORAGE_MINIO_EXTERNAL_ENDPOINT`` when configured.
        """
        return await self._facade.presign_get(path, expires=expires, bucket=bucket)

    async def resign_presigned_url(self, url: str) -> str | None:
        """Re-mint a fresh presigned GET URL for a (possibly expired) URL we own.

        Returns ``None`` (caller keeps the original) unless the URL is
        unambiguously ours: minio backend, http(s) scheme, a host we presign for,
        and a bucket we own (own bucket or any ``w2v-*``). Re-signing a foreign
        URL would mint a valid-looking pointer into our MinIO for an absent object
        (a guaranteed 404), so the guard is load-bearing.
        """
        return await self._facade.resign_presigned_url(url)

    async def delete_job_artifacts(self, job_id: str) -> bool:
        """Delete all artifacts for a job."""
        return await self._facade.delete_prefix(f"artifacts/{job_id}/")

    async def list_job_artifacts(self, job_id: str) -> list[str]:
        """List all artifact paths for a job."""
        return await self._facade.list_prefix(f"artifacts/{job_id}/")

    async def check_health(self) -> bool:
        """Check storage backend health."""
        return await self._facade.check_health()

    # --- Streaming (artifact proxy) -------------------------------------

    async def get_object_stream(
        self,
        bucket: str,
        object_key: str,
        request_headers: Mapping[str, str] | None = None,
    ) -> StreamedObject:
        """Open a range-aware streaming read for the browser artifact proxy.

        Delegates to the facade's first-class streaming get (status/headers +
        async chunk iterator + ``aclose``), forwarding ``Range``/``If-Range``/
        ``If-None-Match`` — replacing the proxy's old reach into the private
        MinIO client. ``bucket`` may be any ``w2v-*`` bucket we can read.
        """
        return await self._facade.get_object_stream(
            object_key, bucket=bucket, request_headers=request_headers
        )


#: Extensions whose type we pin ourselves rather than trusting the platform
#: mimetypes registry, which varies by host and has burned this fleet before
#: (a bare `mimetypes.guess_type` returns None for .webm on some minimal images,
#: and .wav resolves to audio/x-wav on others -- browsers want audio/wav).
_PINNED_CONTENT_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".json": "application/json",
    ".srt": "application/x-subrip",
    ".vtt": "text/vtt",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
}


def guess_artifact_content_type(filename: str) -> str:
    """Best content type for a stored artifact, derived from its NAME.

    Pinned table first, then the platform registry, then a neutral fallback.

    ``application/octet-stream`` is the fallback rather than ``application/json``
    because an unknown type must not claim to be something a consumer will try to
    parse. A browser treats octet-stream as "download it"; it treats a mislabelled
    JSON image as "this is broken", which is how the slide thumbnails silently
    stopped rendering.
    """
    suffix = Path(filename).suffix.lower()
    if suffix in _PINNED_CONTENT_TYPES:
        return _PINNED_CONTENT_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"
