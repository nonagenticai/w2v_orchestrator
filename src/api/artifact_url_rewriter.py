"""Rewrite cluster-internal MinIO URLs in stage-output payloads to
orchestrator-proxied URLs the browser can reach.

Stage outputs persist presigned MinIO URLs (``http://minio.test:8389/...``)
because that host resolves inside the cluster (worker1 dnsmasq) and the
video-compile service fetches the bytes from there. From outside the
cluster — i.e. the user's browser — that hostname does not resolve, so the
SPA's ``<video>`` tag fails to load. Rewrite the URL on the way out so the
browser hits the orchestrator's artifact-proxy endpoint instead.

DB representation is intentionally unchanged: downstream stage handlers
(``video_compile``, ``growth_bundle``) continue to consume the original
presigned URLs through their direct database/handler-deps path. The rewrite
only applies to JSON returned over the public ``/api/runs/...`` surface.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import cast
from urllib.parse import urlparse, urlsplit
from uuid import UUID

from src.config import get_settings

# The host this module matched unconditionally before W3-1a. Kept in the set so
# deriving from settings can only ever ADD hosts, never remove one.
_BASELINE_HOST = "minio.test"


def _endpoint_host(endpoint: str) -> str:
    """The bare hostname of a configured endpoint, port and scheme stripped.

    Mirrors ``StorageService._endpoint_netloc`` in w2v_common, which accepts a
    bare ``host:port``, a bare ``host``, or a full ``scheme://host:port``.
    """
    netloc = urlparse(endpoint if "://" in endpoint else f"scheme://{endpoint}").netloc
    return netloc.rsplit("@", 1)[-1].split(":", 1)[0]


@lru_cache(maxsize=8)
def _host_pattern(minio_endpoint: str, minio_external_endpoint: str | None) -> re.Pattern[str]:
    """Compile the internal-MinIO matcher from the CONFIGURED storage endpoints.

    Why this is derived rather than hardcoded (W3-1a). Two host sets decide
    whether a client-supplied URL is safe to re-sign with our own credentials:

    * ``StorageService._is_own_minio_host`` (w2v_common) — what
      ``resign_presigned_url`` will sign for. It reads
      ``STORAGE_MINIO_ENDPOINT`` and ``STORAGE_MINIO_EXTERNAL_ENDPOINT``.
    * this pattern — what ``assert_refs_owned`` collects and therefore
      ownership-checks.

    Safety rests on the second being a SUPERSET of the first: every URL that
    could be re-signed must first have been ownership-checked. That held only by
    coincidence while this module hardcoded ``minio.test`` and production
    happened to set ``STORAGE_MINIO_ENDPOINT=minio.test:8389`` with no
    external endpoint. Setting ``STORAGE_MINIO_EXTERNAL_ENDPOINT`` — a supported
    ConfigMap edit — widened the re-signer past this literal and armed a real
    hole with no test failing. Reading the same two settings closes that by
    construction.

    The change is deliberately ADDITIVE in two ways, because widening this guard
    is always safe and narrowing it never is:

    * ``minio.test`` stays in the set unconditionally (``_BASELINE_HOST``),
      so a process with no storage configured — every unit test under the
      ``STORAGE_``-stripping ``_clear_settings`` fixture, and any local-backend
      deployment — keeps matching exactly what it matched before.
    * the port stays optional per host rather than pinned to the configured one.
      Pinning would NARROW today's matcher (production stores both
      ``minio.test`` and ``minio.test:8389`` forms) and break artifact
      proxying.
    """
    hosts = {_BASELINE_HOST, _endpoint_host(minio_endpoint)}
    if minio_external_endpoint:
        hosts.add(_endpoint_host(minio_external_endpoint))
    alternatives = "|".join(re.escape(h) for h in sorted(h for h in hosts if h))
    return re.compile(rf"^https?://(?:{alternatives})(?::\d+)?/")


def internal_host_pattern() -> re.Pattern[str]:
    """The live internal-MinIO matcher for this process's storage settings."""
    storage = get_settings().storage
    return _host_pattern(storage.minio_endpoint, storage.minio_external_endpoint)


#: SigV4 presign parameters. Dropped from every emitted proxy path — see
#: :func:`proxy_path`.
_SIGV4_PARAM_PREFIX = "x-amz-"


def _without_presign_params(query: str) -> str:
    """*query* with the SigV4 presign parameters removed, order preserved."""
    kept = [
        pair
        for pair in query.split("&")
        if pair and not pair.split("=", 1)[0].lower().startswith(_SIGV4_PARAM_PREFIX)
    ]
    return "&".join(kept)


def proxy_path(run_id: UUID | str, bucket_and_key: str, query: str) -> str:
    """Return the orchestrator-proxy path for one ``{bucket}/{key}`` pair.

    ``bucket_and_key`` is the URL path stripped of its leading slash;
    ``query`` is the raw query string (already URL-encoded). The result is a
    relative path the SPA mounts directly into ``<video src>`` / ``<img src>``
    — no host needed because the SPA and orchestrator share the
    ``w2v.nonagentic.ai`` origin via the SSR proxy.

    ⚠️ The SigV4 presign parameters are STRIPPED, and that is a bug fix, not
    tidiness. ``artifact_proxy`` ignores the inbound query on purpose and
    re-fetches with the orchestrator's own credentials, so a proxy path can
    never expire — but it used to carry the original ``X-Amz-Date`` /
    ``X-Amz-Expires`` verbatim, and the SPA's ``isPresignExpired`` regexes any
    URL string for exactly those. One hour after a stage ran, every
    ``cut_plan`` shot thumbnail on the REVIEW GATE rendered as "source link
    expired" — 20 of 20, measured in a real browser on 2026-08-20 — with a
    clean console and zero failed requests, because the SPA never even set
    ``src``. A human was being asked to approve an edit they could not see.

    So the URL no longer says something about itself that is false. The SPA is
    being fixed too (``w2v_frontend``: proxy paths are exempt from the expiry
    check), and deliberately so — either half fixes it alone, and neither
    deploy has to land first.
    """
    suffix = bucket_and_key.lstrip("/")
    kept = _without_presign_params(query)
    if kept:
        return f"/api/runs/{run_id}/artifact/{suffix}?{kept}"
    return f"/api/runs/{run_id}/artifact/{suffix}"


def internal_minio_ref(url: object) -> str | None:
    """The ``{bucket}/{key}`` an internal-MinIO URL addresses, else ``None``.

    Shared with the artifact proxy's key-confinement gate so the set of refs this
    module *emits* proxy paths for is exactly the set the proxy will *serve* —
    the same anti-drift argument the shared bucket predicate makes. Still
    percent-encoded here (it is a URL path); the proxy normalises before
    comparing.
    """
    if not isinstance(url, str) or not internal_host_pattern().match(url):
        return None
    return urlsplit(url).path.lstrip("/")


def rewrite_minio_url(run_id: UUID | str, url: str) -> str:
    """Rewrite a single ``http://minio.test/...`` URL to its proxy form.

    Leaves any URL that does not match the internal host unchanged (e.g.,
    already-proxied URLs, ``file://`` paths from the local backend, or any
    other CDN-hosted asset).
    """
    if not isinstance(url, str) or internal_minio_ref(url) is None:
        return url
    parts = urlsplit(url)
    return proxy_path(run_id, parts.path, parts.query)


def collect_internal_minio_refs(value: object, into: set[str]) -> set[str]:
    """Gather every ``{bucket}/{key}`` a stage-output payload references.

    The read-only twin of :func:`rewrite_payload` — same walk, same matcher — so
    "what the API turned into a proxy path" and "what the proxy will serve" can
    never diverge. Kept beside its twin for exactly that reason.
    """
    if isinstance(value, dict):
        for v in cast("dict[str, object]", value).values():
            collect_internal_minio_refs(v, into)
    elif isinstance(value, list):
        for v in cast("list[object]", value):
            collect_internal_minio_refs(v, into)
    else:
        ref = internal_minio_ref(value)
        if ref:
            into.add(ref)
    return into


def rewrite_payload(run_id: UUID | str, value: object) -> object:
    """Recursively rewrite ``minio.test`` URLs inside a stage-output value.

    Walks dicts and lists; replaces matching string values in place and
    returns the mutated container. Non-string scalars are passed through
    unchanged. The function returns the input reference so callers can chain
    or assign back.
    """
    if isinstance(value, dict):
        mapping = cast("dict[str, object]", value)
        for k, v in list(mapping.items()):
            mapping[k] = rewrite_payload(run_id, v)
        return mapping
    if isinstance(value, list):
        seq = cast("list[object]", value)
        for i, v in enumerate(seq):
            seq[i] = rewrite_payload(run_id, v)
        return seq
    if isinstance(value, str):
        return rewrite_minio_url(run_id, value)
    return value
