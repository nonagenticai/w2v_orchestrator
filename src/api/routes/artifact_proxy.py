"""Artifact proxy — streams a stage's stored MinIO bytes to the browser.

The SPA renders compiled videos + narration audio + slide PNGs via
``<video>`` / ``<audio>`` / ``<img>`` tags. Stage outputs persist presigned
URLs hosted at ``minio.test:8389`` so the in-cluster video-compile
service can fetch them anonymously over cluster DNS. That host is unreachable
from the user's browser, so this router exposes a tenant-isolated proxy:

    GET /api/runs/{run_id}/artifact/{bucket}/{object_key:path}

Three gates, in order: the bucket must be one the orchestrator may read, the
caller's tenant must own ``run_id``, and — the gate that makes the first two
mean anything — ``object_key`` must be one of the objects ``run_id``'s own
persisted stage outputs reference (:func:`_confine_object_key`). Owning *a* run
is not permission to name *any* key: the shared ``StorageService`` has no tenant
concept, so whatever key reaches it is fetched with the orchestrator's own
credentials no matter whose object it is, and ``object_key`` is a path param.

Only past all three do we **use the orchestrator's own MinIO credentials** to
``get_object`` directly — the inbound query string (and
the presigned signature on the stored URL) is ignored on purpose. We
discovered the hard way that stored presigned URLs carry the issuing
service's signature with a 1-hour TTL; after that hour every video became
"unavailable" again no matter how the URL was forwarded. The MinIO-client
path has no TTL.

Range / If-Range / If-Modified-Since / If-None-Match round-trip so ``<video>``
seeking works without a server-side cache; Content-Range / Accept-Ranges /
ETag / Last-Modified / Cache-Control come back — including the ``304 Not
Modified`` a satisfied conditional earns, which ``miniopy_async`` reports by
*raising* (it returns a response only for 200/204/206) and which we therefore
have to recognise as the success it is rather than a transport failure. Bodies stream in 64 KiB
chunks; the underlying aiohttp connection releases when the browser
disconnects via the StreamingResponse body generator's ``finally``.

Why not host-rewrite via Cloudflare tunnel? Public ``minio.nonagentic.ai``
would need a new tunnel + ingress + CORS, and MinIO has no
Keycloak/license-token boundary — direct browser-to-MinIO would lose
tenant isolation. Routing through the orchestrator reuses the existing
``AuthMiddleware`` for free.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response
from starlette.responses import StreamingResponse

from src.api.routes._common import SessionDep, TenantIdDep
from src.api.routes.runs import _tenant_owns_run
from src.config import get_settings
from src.services.artifact_ref_guard import run_artifact_refs as _run_artifact_refs
from src.services.object_key_guard import KeyConfinementError, normalize_key
from src.services.storage_service import guess_artifact_content_type, is_readable_w2v_bucket
from src.services.tts_storage import (
    _get_storage,  # why: reuse the process-wide lazy MinIO singleton already used by TTS/slide storage
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/runs", tags=["artifacts"])
log = logging.getLogger(__name__)

# Buckets the proxy is willing to fetch from: any bucket in the orchestrator's
# own ``w2v-*`` MinIO namespace (its configured bucket, ``w2v-compiled-videos``,
# ``w2v-html-render``, ``w2v-audiobooks``, ``w2v-courses`` …). This is validated
# by the shared ``is_readable_w2v_bucket`` predicate — the SAME one
# ``StorageService`` uses to decide what it may re-sign — so the proxy gate and
# the compile-side re-sign path can never drift (the drift is exactly why a
# delivered audiobook re-signed fine on the compile side yet 404'd here). The
# orchestrator's MinIO user can read all w2v buckets, so re-fetching with our own
# creds means the baked presign expiry never bites the browser. This gate is a
# namespace bound and NOTHING else — the tenant boundary is ``_tenant_owns_run``
# plus ``_confine_object_key`` below, and it was the missing second half of that
# pair (bucket + run checked, key waved through) that let an owned run address a
# stranger's object.

# Headers we forward from the inbound request to MinIO so video ``Range``
# seeks and conditional fetches work. Whitelist only — anything else
# (cookie, authorization) would be a leak.
_FORWARDED_REQUEST_HEADERS = frozenset({"range", "if-range", "if-modified-since", "if-none-match"})

# Headers we forward from MinIO back to the browser. Whitelist again — MinIO
# stamps various ``X-Amz-*`` headers that mean nothing to the SPA.
_FORWARDED_RESPONSE_HEADERS = frozenset(
    {
        "content-type",
        "content-length",
        "content-range",
        "accept-ranges",
        "etag",
        "last-modified",
        "cache-control",
    }
)

# Headers a bodiless ``304 Not Modified`` may carry. Deliberately NOT
# content-type / content-length / content-range: a 304 has no body, and a
# content-length copied off the would-have-been 200 misdescribes it.
_NOT_MODIFIED_RESPONSE_HEADERS = frozenset({"etag", "last-modified", "cache-control"})

# Stream in modest chunks — 64 KiB keeps memory bounded on long videos
# without paying syscall overhead per packet.
_STREAM_CHUNK_BYTES = 64 * 1024

# ``304 Not Modified`` is the SUCCESS answer to the conditional request we
# forwarded, not a failure — see ``_open_minio_object``.
_NOT_MODIFIED = 304

# S3 error codes that mean "absent, or not ours to read". ``S3Error`` carries no
# ``status_code`` (only ``ServerError`` does), and when miniopy_async synthesises
# the error itself from a bodiless 403/404 it may hand us a response object we
# cannot read a status off — the code is then the only structured signal left.
_NOT_FOUND_S3_CODES = frozenset({"NoSuchKey", "NoSuchBucket", "ResourceNotFound", "AccessDenied"})

# ``416 Range Not Satisfiable`` is the one upstream 4xx on this path the BROWSER
# both caused and can correct: a ``<video>`` probing past EOF, or seeking against
# a stale cached duration, asks for ``Range: bytes=99999999-`` and MinIO answers
# ``InvalidRange``. Collapsing that into 502 turned a self-correcting protocol
# exchange into a dead media element — exactly the seek a reviewer performs on a
# footage thumbnail. DECISION: 416 passes through; the other statuses this branch
# can see (400 InvalidArgument, 405 MethodNotAllowed) stay 502, because those
# describe a request THIS service composed, not one the browser can fix — telling
# the SPA "400, your fault" would blame the client for our bug and invite a retry
# loop. 403/404 still collapse to 404 so a probe cannot tell absent from forbidden.
_RANGE_NOT_SATISFIABLE = 416


def _filter_headers(items: Mapping[str, str], allowed: frozenset[str]) -> dict[str, str]:
    """Whitelist-filter request or response headers (case-insensitive).

    Anything outside ``allowed`` is dropped — keeps cookie / authorization
    off the upstream and ``X-Amz-*`` off the downstream so the SPA never
    sees storage-backend implementation details. Accepts both
    ``starlette.datastructures.Headers`` and ``multidict.CIMultiDict``
    (aiohttp), which both satisfy ``Mapping[str, str]``.
    """
    return {name: value for name, value in items.items() if name.lower() in allowed}


def _s3_error_status(exc: Exception) -> int:
    """The HTTP status behind a miniopy ``S3Error``.

    ``S3Error`` exposes ``.code`` / ``.message`` / ``.response`` and — unlike
    ``ServerError`` — **no** ``status_code``, so reading ``exc.status_code`` is
    always the default and every S3Error would be treated as a 5xx. Read the
    aiohttp response's ``status`` first; fall back to the S3 error *code*, which
    is structured too (``NoSuchKey`` / ``AccessDenied`` / …), never to the
    message string.
    """
    status = getattr(getattr(exc, "response", None), "status", None)
    if isinstance(status, int):
        return status
    return 404 if getattr(exc, "code", None) in _NOT_FOUND_S3_CODES else 500


def _upstream_content_range(exc: Exception) -> dict[str, str] | None:
    """The upstream ``Content-Range`` carried by an ``S3Error``, if it has one.

    ``S3Error.response`` is the aiohttp ``ClientResponse``; its headers are the
    only place the origin's ``bytes */<length>`` survives. ``getattr`` because
    miniopy sometimes synthesises the error with a response object that carries
    no readable headers at all.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not isinstance(headers, Mapping):
        return None
    forwarded = _filter_headers(cast("Mapping[str, str]", headers), frozenset({"content-range"}))
    return forwarded or None


def _not_modified_response(
    request_headers: Mapping[str, str],
    upstream_headers: Mapping[str, str] | None = None,
) -> Response:
    """A bodiless ``304``, carrying the validators RFC 9110 §15.4.5 asks for.

    On the normal path MinIO *raised* rather than returning a response, so there
    are no upstream headers to forward and the validators are echoed from what
    the client itself sent: an ``If-None-Match`` naming exactly one entity tag is
    by definition the ``ETag`` the client would have been handed back, and
    ``If-Modified-Since`` is the ``Last-Modified`` it was derived from. ``*`` and
    multi-tag lists are NOT echoed — neither names a single entity, so echoing
    one would assert a validator the origin never issued.
    """
    headers = dict(_filter_headers(upstream_headers or {}, _NOT_MODIFIED_RESPONSE_HEADERS))
    present = {name.lower() for name in headers}

    if "etag" not in present:
        if_none_match = (request_headers.get("if-none-match") or "").strip()
        if if_none_match and if_none_match != "*" and "," not in if_none_match:
            headers["etag"] = if_none_match
    if "last-modified" not in present:
        if_modified_since = (request_headers.get("if-modified-since") or "").strip()
        if if_modified_since:
            headers["last-modified"] = if_modified_since

    # No media_type / content-length: Starlette omits both for a 304, and the
    # body is empty by construction.
    return Response(status_code=_NOT_MODIFIED, headers=headers)


# The run's artifact list, and the ONE place it is derived — shared with the
# write-path guard (``assert_refs_owned``) so "what a caller may store" and "what
# the proxy will serve" are answered by the same walk. Bound to a module global
# so the route reads it late (tests substitute it).
#
# ⚠️ PERF (follow-up, not fixed here): this runs on EVERY request, including every
# Range request of a playback — an audiobook re-parses that run's whole JSONB
# output set dozens of times per listen. Index-bounded (``run_id``) but not
# size-bounded. Deliberately NOT cached here: the docstring's own reason stands —
# a just-written artifact must be servable immediately — and the safe bound
# (memoise per ``run_id`` keyed on ``max(stage_outputs.updated_at)``) costs a
# second query and real staleness reasoning that does not belong in a security fix.
#
# One extra query per artifact request (the ownership check already costs one).


async def _confine_object_key(
    session: AsyncSession, run_id: UUID, bucket: str, object_key: str
) -> str:
    """Return the normalised ``object_key``, proven to belong to ``run_id``.

    Ownership of the run is necessary but nowhere near sufficient: the key is a
    path param, and MinIO is read with the orchestrator's own credentials, so
    without this the caller picks any object in any readable ``w2v-*`` bucket.
    The key is re-derived — judged against its fully decoded form, path-normalised,
    then matched as a whole ``{bucket}/{key}`` against the refs the run itself
    persisted — so a caller-supplied string can only ever name something the run
    already owns. What is matched and fetched is the identifier the caller
    *named*, not the decoded probe: over-decoding would turn a real
    ``discount%2Ddeal.pdf`` into a key MinIO has never heard of (see
    ``object_key_guard``). Starlette has already stripped the one encoding layer
    the path parameter carried, so the stored refs are stripped exactly once too.

    That the persisted refs are trustworthy is not this gate's doing — every
    caller-facing write into ``StageOutput.output`` runs
    ``artifact_ref_guard.assert_refs_owned`` first, or a caller could simply patch
    the victim's URL into its own run and hand this check a forged allowlist. This
    gate remains as the backstop for any write path that guard does not cover.

    A whole-ref match rather than a prefix rule because there is no single "the
    run's prefix": a run's own objects sit under ``artifacts/{run_id}/`` and
    ``videos/{run_id}/``, but the compiled video, audiobook and html-render
    artifacts it legitimately serves are written by *other* services into
    ``w2v-*`` buckets whose key shapes this repo does not define. The persisted
    refs are the only record that spans all of them.

    Raises 404 ("artifact not found") on refusal: same body as an absent object,
    so a probe learns nothing from the difference.
    """
    try:
        normalized = normalize_key(object_key)
    except KeyConfinementError as exc:
        log.info("artifact proxy refused malformed key for run %s: %s", run_id, exc)
        raise HTTPException(404, "artifact not found") from exc

    if f"{bucket}/{normalized}" not in await _run_artifact_refs(session, run_id):
        log.info(
            "artifact proxy refused key not owned by run %s: %s/%s",
            run_id,
            bucket,
            normalized,
        )
        raise HTTPException(404, "artifact not found")
    return normalized


async def _open_minio_object(
    bucket: str,
    object_key: str,
    request_headers: dict[str, str],
) -> Any:
    """Open a range-aware streaming read against MinIO; raises HTTPException on failure.

    Returns the facade's ``StreamedObject`` (status / headers / async chunk
    iterator). The caller releases it via ``aclose()`` in a ``finally`` so the
    underlying connection releases when the browser disconnects.

    Returns ``None`` — not a stream — when the upstream answered **304 Not
    Modified**, i.e. the conditional request we forwarded was satisfied. That is
    a success with no body to open, so there is nothing to hand back and nothing
    to close; the caller turns it into a bodiless 304.
    """
    storage = await _get_storage()
    # ``get_object_stream`` is beyond the minimal ``_StorageLike`` protocol, so
    # reach it via getattr (same house pattern as the other non-protocol storage
    # methods) — a storage without it is treated as uninitialised.
    open_stream = getattr(storage, "get_object_stream", None)
    if open_stream is None:
        raise HTTPException(503, "artifact storage not initialised")

    # miniopy_async raises on EVERY non-2xx — its ``_execute`` returns the
    # response only for 200/204/206 and raises for anything else. Two classes
    # come out of that branch: ``S3Error`` (an XML error body, or a status it has
    # a code mapping for — ``code`` = "NoSuchKey" / "AccessDenied" / …, and NO
    # ``status_code``), and ``ServerError`` (a bodiless status it has no mapping
    # for — carries ``status_code``, and is what a **304** arrives as).
    try:
        from miniopy_async.error import S3Error, ServerError
    except ImportError as exc:  # pragma: no cover -- prod always has it
        raise HTTPException(500, "minio sdk missing") from exc

    try:
        stream = await open_stream(
            bucket,
            object_key,
            request_headers=request_headers or None,
        )
    except RuntimeError as exc:
        # Backend not initialised (or not MinIO) — the facade raises RuntimeError.
        raise HTTPException(503, "artifact storage not initialised") from exc
    except ServerError as exc:
        # A conditional GET the object satisfied: MinIO answered 304, which
        # miniopy turns into an exception purely because it is not 2xx. This is
        # the success the ``If-None-Match`` / ``If-Modified-Since`` we forwarded
        # was asking for — the whole point of forwarding them. Matched on the
        # structured ``status_code``, never the message text, and NOT logged at
        # WARNING: it fires on every cached asset the SPA re-requests and used to
        # bury real transport failures (and answer 502 to a valid request).
        if exc.status_code == _NOT_MODIFIED:
            return None
        log.warning(
            "artifact proxy MinIO server error for %s/%s: status=%s",
            bucket,
            object_key,
            exc.status_code,
        )
        raise HTTPException(502, "artifact unreachable") from exc
    except S3Error as exc:
        # 403 / 404 collapse to 404 — browser only cares whether the artifact
        # is reachable; not leaking whether-vs-permissions matches the
        # bucket-allowlist 404 behaviour above.
        status = _s3_error_status(exc)
        log.info(
            "artifact proxy MinIO S3Error for %s/%s: code=%s status=%s",
            bucket,
            object_key,
            getattr(exc, "code", "?"),
            status,
        )
        if status in (403, 404):
            raise HTTPException(404, "artifact not found") from exc
        if status == _RANGE_NOT_SATISFIABLE:
            raise HTTPException(
                _RANGE_NOT_SATISFIABLE,
                "requested range not satisfiable",
                # RFC 9110 §15.5.17: a 416 SHOULD carry ``Content-Range:
                # bytes */<complete-length>`` so the player learns the real size
                # in the same round trip instead of guessing again. Forwarded
                # from upstream when MinIO sent one; omitted (not invented) when
                # it did not — a fabricated length is worse than none.
                headers=_upstream_content_range(exc),
            ) from exc
        raise HTTPException(502, "artifact unreachable") from exc
    except Exception as exc:
        # Transport-level / connection / DNS / TLS — never let an opaque
        # 500 surface; the browser would just retry, hammering MinIO.
        log.warning(
            "artifact proxy MinIO transport error for %s/%s: %s",
            bucket,
            object_key,
            exc,
        )
        raise HTTPException(502, "artifact unreachable") from exc

    return stream


@router.get(
    "/{run_id}/artifact/{bucket}/{object_key:path}",
    operation_id="runs.artifact_proxy",
    summary="Stream a stage artifact (video / audio / image) from in-cluster MinIO",
)
async def get_artifact(
    run_id: UUID,
    bucket: str,
    object_key: str,
    request: Request,
    session: SessionDep,
    # why: loaded straight by <img>/<video>/<audio> markup, which cannot set the
    # X-Active-Tenant-Id header — the scope-switch rides ?active_tenant= instead.
    # get_tenant_id reads both channels, so the router-level gates see it too.
    tenant_id: TenantIdDep,
) -> Response:
    # 404 over 400 on bucket — don't leak whether the bucket exists.
    if not is_readable_w2v_bucket(bucket, get_settings().storage.minio_bucket):
        raise HTTPException(404, "artifact not found")
    if (await _tenant_owns_run(session, run_id, tenant_id)) is None:
        raise HTTPException(404, "run not found")
    # Owning the run is not permission to name any key — tie the key to the run.
    object_key = await _confine_object_key(session, run_id, bucket, object_key)

    # END THE TRANSACTION BEFORE THE BODY STREAMS.
    #
    # The two gates above autobegin a transaction (SessionDep is autocommit=False),
    # and `get_db_session` only commits in its post-`yield` teardown — which FastAPI
    # 0.136 runs AFTER `await response(scope, receive, send)`, i.e. after the whole
    # artifact has streamed. So postgres sits `idle in transaction` for the length of
    # a video download, and pgbouncer (pool_mode=transaction) cannot recycle the
    # server slot: concurrent clip downloads drain the pool and every other endpoint
    # — including the DAG dispatcher — dies on `query_wait_timeout`. Same defect
    # class the SSE routes were fixed for (see `src/api/sse.py`: "held for the whole
    # response … that is exactly how the pool exhausted in prod").
    #
    # Committing here hands the server slot back immediately; the session held until
    # teardown then costs only a client-side pool entry. Nothing is written on this
    # path, so the commit is purely a transaction terminator.
    await session.commit()

    request_headers = _filter_headers(request.headers, _FORWARDED_REQUEST_HEADERS)
    upstream_response = await _open_minio_object(bucket, object_key, request_headers)
    if upstream_response is None:
        # The forwarded conditional was satisfied upstream — nothing was opened.
        return _not_modified_response(request_headers)

    response_headers = _filter_headers(upstream_response.headers, _FORWARDED_RESPONSE_HEADERS)
    upstream_status = upstream_response.status

    if upstream_status == _NOT_MODIFIED:
        # A backend that returns 304 instead of raising (the local-file stream,
        # or a future non-raising client). Still bodiless: streaming a body under
        # a 304 is a protocol violation, so release the handle and answer flat.
        with contextlib.suppress(Exception):
            await upstream_response.aclose()
        return _not_modified_response(request_headers, response_headers)

    async def _body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream_response.iter_chunks(_STREAM_CHUNK_BYTES):
                yield chunk
        finally:
            # Best-effort connection release — the body iter is already
            # exhausted, so any error here is the aiohttp internal pool
            # tearing down a half-spent connection. Nothing actionable.
            with contextlib.suppress(Exception):
                await upstream_response.aclose()

    # The object key's extension WINS over the stored content type.
    #
    # Not a stylistic preference — the stored type is demonstrably unreliable.
    # `store_stage_artifact` declared `application/json` for every artifact it
    # wrote, so every historical `promo_slides/*.png` and `*_narrate/*.wav` in
    # MinIO carries that type today. Browsers refuse to render an <img> served as
    # application/json, so run pages showed empty thumbnails while the storyline
    # counter reported "8/8 visuals" (measured live on run 019efbb0, 2026-08-09:
    # 8 valid PNG bodies, 0 with naturalWidth > 0).
    #
    # Fixing only the writer would repair new runs and leave every existing run
    # broken forever, since the type is baked into the stored object's metadata.
    # Deriving it here repairs the whole history on the next request.
    #
    # `guess_artifact_content_type` falls back to octet-stream for unknown
    # extensions, so we only override when we actually recognise the suffix —
    # an unrecognised key keeps whatever upstream said.
    derived = guess_artifact_content_type(object_key)
    if derived == "application/octet-stream":
        derived = response_headers.get("content-type", "application/octet-stream")
    response_headers["content-type"] = derived

    return StreamingResponse(
        _body(),
        status_code=upstream_status,
        media_type=derived,
        headers=response_headers,
    )
