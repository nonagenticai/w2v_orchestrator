"""Persistent, cross-run, per-source ingest cache.

``source_ingest`` grounds a project in ``brief.sources[]`` by scraping/parsing
each source (webscraper scrape / in-process parse) and republishing extracted media. That is expensive, and
a user running MANY courses concurrently over the SAME reference sources
re-ingests each one N times. This module memoizes each source's ingest in a
tenant-scoped DB row (``source_ingest_cache``) so identical sources reuse one
ingest across runs.

Three jobs, mirroring existing fleet primitives:

* **Fingerprint** — a content/identity key. Folds ``tenant_id`` in, so a second
  tenant's identical source is a different row (an automatic miss even before
  RLS); folds the auth identity in, so a credentialed fetch never serves an
  anonymous one's body; for documents, keys on the CONTENT hash (not the ref) so
  the same file uploaded twice hits.
* **Single-flight** — a DB-backed claim (INSERT ``pending`` … ON CONFLICT DO
  NOTHING, else reclaim a stale/expired row) so K concurrent identical ingests
  coalesce onto ONE ingest; the losers poll for the winner's result. Same
  serialize-on-row model as ``runs.run_claim.RunDbClaim``.
* **Degrade-safe** — ANY cache-layer error (DB down) or a poll timeout falls
  through to a direct ingest, so the cache can never fail a run. A per-source scrape/parse
  failure propagates to the stage's existing soft-fail handler unchanged.

The cache stores each cached image's DURABLE media-cache object key (see
``tts_storage.store_ingest_cache_image``), never a presigned URL — the consume
path re-mints a fresh URL so cross-run reuse survives presign expiry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from hashlib import sha256
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import and_, delete, func, or_, select, true, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from src.database.models import SourceIngestCache
from src.database.session import bind_session_tenant

log = logging.getLogger(__name__)

# A ``pending`` claim not completed within this window is treated as abandoned (a
# crashed winner) and reclaimable — must exceed the scrape per-call budget
# (``source_ingest._KAG_TIMEOUT_S`` = 180s) so a slow-but-live ingest is never
# stolen out from under itself.
_CLAIM_TTL = 200  # seconds
# A loser polls the row this often, up to this long, before giving up and
# ingesting directly (never wedge a run on a stuck winner).
_POLL_INTERVAL = 1.0  # seconds
_MAX_WAIT = 180.0  # seconds

# Bump to invalidate every cached entry when the extraction/packing semantics
# change (mirrors tts_storage._MEDIA_GEN_VERSION / PREPROC_VERSION).
CACHE_SCHEMA_VERSION = "1"


class CachedSource(BaseModel):
    """The per-source contribution cached for reuse — exactly what a fresh ingest
    would append to the stage's aggregation. ``images`` carry a durable
    ``cache_key`` (re-minted to a URL at consume time), never a presigned URL."""

    text: str = ""
    references: list[str] = Field(default_factory=list)
    images: list[dict[str, object]] = Field(default_factory=list)
    #: Partial-extraction facts (e.g. scanned pages left un-OCR'd) re-surfaced on
    #: every reuse. Absent on rows written before it existed → ``[]``.
    notes: list[str] = Field(default_factory=list)


# ─── Fingerprint ───────────────────────────────────────────────────────────────


def normalize_url(url: str) -> str:
    """Conservative URL normalization for the cache key.

    Lowercase scheme + host, strip a default port, drop the fragment; KEEP path +
    query (they change the fetched content) and the trailing slash (aggressive
    canonicalization risks false collisions between distinct pages). Two requests
    that differ only in casing/host/fragment share a cache entry; anything that
    could change the body does not.
    """
    p = urlsplit(url.strip())
    scheme = (p.scheme or "http").lower()
    host = (p.hostname or "").lower()
    port = p.port
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = host if (port is None or default_port) else f"{host}:{port}"
    return urlunsplit((scheme, netloc, p.path, p.query, ""))


def document_ref_key(data: bytes) -> str:
    """Cache ref-key for a document source: the CONTENT hash, not the ref/filename,
    so the same bytes uploaded under two different keys share one cache entry."""
    return "doc:" + sha256(data).hexdigest()


def auth_id_for(auth_info: dict[str, object] | None) -> str:
    """Stable identity token for a source's auth. ``anon`` when absent, else a
    short digest of the auth payload — folded into the fingerprint so a
    credentialed fetch and an anonymous fetch of the same URL never cross-serve."""
    if not auth_info:
        return "anon"
    try:
        raw = json.dumps(auth_info, sort_keys=True, default=str)
    except TypeError, ValueError:
        raw = str(auth_info)
    return "auth:" + sha256(raw.encode("utf-8")).hexdigest()[:16]


def source_fingerprint(
    tenant_id: UUID | str,
    kind: str,
    ref_key: str,
    *,
    opts_token: str = "img=1",  # noqa: S107 -- a scrape-options token, not a secret
    auth_id: str = "anon",
) -> str:
    """64-hex content/identity fingerprint (the cache PK).

    ``ref_key`` is ``normalize_url(url)`` for urls or ``document_ref_key(bytes)``
    for documents. ``opts_token`` encodes scrape options that change output (today
    the stage always requests images → ``img=1``); a future flag change bumps it so
    a stale body is never served. ``tenant_id`` folds in for isolation.
    """
    raw = "|".join([f"v{CACHE_SCHEMA_VERSION}", str(tenant_id), kind, ref_key, opts_token, auth_id])
    return sha256(raw.encode("utf-8")).hexdigest()


# ─── Single-flight repository ───────────────────────────────────────────────────


def _stale_threshold():  # noqa: ANN202 -- SQLAlchemy expression, not a value
    """``now() - _CLAIM_TTL`` computed on the DB server (skew-free)."""
    return func.now() - func.make_interval(0, 0, 0, 0, 0, 0, _CLAIM_TTL)


Producer = Callable[[], Awaitable[CachedSource]]


class SourceCacheRepository:
    """Tenant-scoped, single-flight persistence for source ingests.

    One instance per stage handler; ``instance_id`` identifies this claimant (like
    ``RunDbClaim``) so a claim is released/reclaimed by the right owner. Every
    method binds the tenant GUC before its query so the row is admitted under RLS
    enforce (a hard no-op off enforce / in tests running as the table owner).
    """

    def __init__(self, session_factory, *, instance_id: str) -> None:  # noqa: ANN001
        self._sessions = session_factory
        self._instance_id = instance_id

    async def get_fresh(self, fingerprint: str, tenant_id: UUID | str) -> CachedSource | None:
        """Return the cached contribution for a ``ready``, non-expired row, else ``None``."""
        async with self._sessions() as s:
            await bind_session_tenant(s, _as_uuid(tenant_id))
            payload = (
                await s.execute(
                    select(SourceIngestCache.payload).where(
                        SourceIngestCache.fingerprint == fingerprint,
                        SourceIngestCache.state == "ready",
                        or_(
                            SourceIngestCache.expires_at.is_(None),
                            SourceIngestCache.expires_at > func.now(),
                        ),
                    )
                )
            ).scalar_one_or_none()
        if payload is None:
            return None
        return CachedSource(**payload)

    async def try_claim(
        self,
        fingerprint: str,
        tenant_id: UUID | str,
        kind: str,
        ref: str,
        *,
        refresh: bool = False,
    ) -> bool:
        """Claim the ingest for this instance. ``True`` — we won (must produce);
        ``False`` — another instance holds a fresh claim / fresh ready row (poll).

        Two racing pods serialize on the row: the INSERT..ON CONFLICT DO NOTHING
        lets exactly one create the ``pending`` row; the loser's reclaim UPDATE
        matches only a STALE pending claim (crashed winner) or an EXPIRED ready row
        (lazy refresh) — or anything when ``refresh`` forces it.
        """
        async with self._sessions() as s:
            await bind_session_tenant(s, _as_uuid(tenant_id))
            won = (
                await s.execute(
                    pg_insert(SourceIngestCache)
                    .values(
                        fingerprint=fingerprint,
                        tenant_id=_as_uuid(tenant_id),
                        kind=kind,
                        ref=ref,
                        state="pending",
                        claimed_by=self._instance_id,
                        claimed_at=func.now(),
                    )
                    .on_conflict_do_nothing(index_elements=["fingerprint"])
                    .returning(SourceIngestCache.fingerprint)
                )
            ).scalar_one_or_none()
            if won is None:
                reclaimable = (
                    true()
                    if refresh
                    else or_(
                        and_(
                            SourceIngestCache.state == "pending",
                            SourceIngestCache.claimed_at < _stale_threshold(),
                        ),
                        and_(
                            SourceIngestCache.state == "ready",
                            or_(
                                SourceIngestCache.expires_at.is_(None),
                                SourceIngestCache.expires_at < func.now(),
                            ),
                        ),
                    )
                )
                won = (
                    await s.execute(
                        update(SourceIngestCache)
                        .where(SourceIngestCache.fingerprint == fingerprint, reclaimable)
                        .values(
                            state="pending",
                            claimed_by=self._instance_id,
                            claimed_at=func.now(),
                            expires_at=None,
                        )
                        .returning(SourceIngestCache.fingerprint)
                    )
                ).scalar_one_or_none()
            await s.commit()
            return won is not None

    async def complete(
        self,
        fingerprint: str,
        tenant_id: UUID | str,
        result: CachedSource,
        ttl_seconds: int,
    ) -> None:
        """Flip our claimed row to ``ready`` with the cached payload + TTL."""
        async with self._sessions() as s:
            await bind_session_tenant(s, _as_uuid(tenant_id))
            await s.execute(
                update(SourceIngestCache)
                .where(SourceIngestCache.fingerprint == fingerprint)
                .values(
                    state="ready",
                    payload=result.model_dump(),
                    claimed_by=None,
                    claimed_at=None,
                    expires_at=func.now() + func.make_interval(0, 0, 0, 0, 0, 0, ttl_seconds),
                )
            )
            await s.commit()

    async def release(self, fingerprint: str, tenant_id: UUID | str) -> None:
        """Drop our OWN pending claim so a retry can proceed immediately (used when
        the producer failed). Owner-scoped so a stale-reclaim by another pod is
        never clobbered."""
        async with self._sessions() as s:
            await bind_session_tenant(s, _as_uuid(tenant_id))
            await s.execute(
                delete(SourceIngestCache).where(
                    SourceIngestCache.fingerprint == fingerprint,
                    SourceIngestCache.claimed_by == self._instance_id,
                    SourceIngestCache.state == "pending",
                )
            )
            await s.commit()

    async def get_or_produce(
        self,
        fingerprint: str,
        tenant_id: UUID | str,
        kind: str,
        ref: str,
        *,
        ttl_seconds: int,
        refresh: bool,
        producer: Producer,
    ) -> tuple[CachedSource, str]:
        """Return the cached ingest, producing it via ``producer`` under
        single-flight on a miss. Returns ``(result, source)`` where ``source`` is
        ``hit`` / ``produced`` / ``produced_after_wait`` / ``degraded``.

        Degrade contract: a cache-LAYER failure (any ``SQLAlchemyError``) or a
        poll timeout falls through to a direct ``producer()`` (uncached) — the
        cache never fails a run. A ``producer`` failure (a scrape/parse error) is NOT
        swallowed here: we release our claim and re-raise so the stage's existing
        per-source soft-fail records the note, and the producer is never called
        twice.
        """
        try:
            if not refresh:
                hit = await self.get_fresh(fingerprint, tenant_id)
                if hit is not None:
                    return hit, "hit"
            deadline = time.monotonic() + _MAX_WAIT
            force = refresh
            while True:
                won = await self.try_claim(fingerprint, tenant_id, kind, ref, refresh=force)
                force = False  # only the first attempt force-reclaims
                if won:
                    try:
                        result = await producer()
                    except Exception:
                        await self._safe_release(fingerprint, tenant_id)
                        raise
                    await self._safe_complete(fingerprint, tenant_id, result, ttl_seconds)
                    return result, "produced"
                hit = await self.get_fresh(fingerprint, tenant_id)
                if hit is not None:
                    return hit, "produced_after_wait"
                if time.monotonic() >= deadline:
                    break
                await asyncio.sleep(_POLL_INTERVAL)
        except SQLAlchemyError as exc:
            log.warning("source-ingest cache degraded (%s); ingesting %s directly", exc, ref)
        return await producer(), "degraded"

    async def _safe_complete(
        self, fingerprint: str, tenant_id: UUID | str, result: CachedSource, ttl_seconds: int
    ) -> None:
        try:
            await self.complete(fingerprint, tenant_id, result, ttl_seconds)
        except SQLAlchemyError as exc:  # best-effort: an uncached result is still correct
            log.warning("source-ingest cache complete failed (%s); result uncached", exc)

    async def _safe_release(self, fingerprint: str, tenant_id: UUID | str) -> None:
        """Best-effort claim release, called from ``get_or_produce``'s ``except
        Exception`` arm around ``producer()``. Widened from ``except
        SQLAlchemyError`` to ``except Exception``: this method's own docstring
        already names the stale-claim TTL as the backstop for ANY release
        failure, not just a DB-layer one — ``release()`` also calls
        ``_as_uuid()``, so a malformed ``tenant_id`` raises a plain
        ``ValueError`` here, which the narrower guard let escape and REPLACE
        the real ``producer()`` exception on the caller's `raise`. Cost of the
        degraded state: our pending claim is left in place until the
        stale-claim TTL expires (``_CLAIM_TTL``), so a retry has to wait out
        that window instead of reclaiming immediately — never worse than that.
        ``BaseException`` (``CancelledError``, etc.) still escapes.
        """
        try:
            await self.release(fingerprint, tenant_id)
        except Exception as exc:  # noqa: BLE001 — the stale-claim TTL is the backstop
            log.debug("source-ingest cache release failed (%s)", exc)


def _as_uuid(tenant_id: UUID | str) -> UUID:
    return tenant_id if isinstance(tenant_id, UUID) else UUID(str(tenant_id))
