"""Configuration management for w2v_orchestrator using Pydantic Settings."""

import base64
import binascii
import json
import logging
import math
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from w2v_common.auth.cell_jwt import CellAuthMode, cell_auth_mode
from w2v_common.config import APISettings as BaseAPISettings
from w2v_common.config import AuthSettings as AuthSettings
from w2v_common.config import CacheSettings as CacheSettings
from w2v_common.config import DatabaseSettings as BaseDatabaseSettings
from w2v_common.config import DownstreamSettings as _BaseDownstreamSettings
from w2v_common.config import IdempotencySettings as IdempotencySettings
from w2v_common.config import LoggingSettings as LoggingSettings
from w2v_common.config import RedisSettings as BaseRedisSettings
from w2v_common.config import StorageSettings as BaseStorageSettings
from w2v_common.config import TracingSettings as TracingSettings

# Imported from the library rather than re-declared, so the config bound and the
# runtime check CANNOT drift. `stage_call(retries=...)` runs `_validate_retries`,
# which raises ValueError above this value INSIDE the stage handler — i.e. at
# runtime, in production, across the ~40 call sites that pass
# `retries=self._llm_retries`. Binding the field to the same number moves that
# failure to config load, where a bad `PIPELINE_LLM_RETRIES` is refused at boot.
# The name is module-private upstream; that is deliberate here. If w2v_common
# renames or drops it, this import fails LOUDLY at orchestrator startup, which is
# strictly better than a silently-stale copy of the number that lets >cap through
# to crash a stage mid-run.
from w2v_common.stages.execution import _MAX_RETRIES as STAGE_CALL_MAX_RETRIES

logger = logging.getLogger(__name__)

# ── A11 ship-gate ceilings (PIPELINE_SHIP_GATE_DEFAULT_CEILINGS) ─────────────────
# The four calibratable gestalt families, split by SHAPE: three are scalar thresholds,
# `loudness_lufs` is a two-element [lo, hi] WINDOW (see
# `gestalt_metrics.loudness_window`). Anything outside this set is dropped — a typo'd
# key must not silently ride into the gate config as a no-op the operator believes is live.
_SHIP_GATE_SCALAR_CEILINGS = frozenset({"momentum_gap_max", "continuity_l2", "cohesion_floor"})
_SHIP_GATE_WINDOW_CEILING = "loudness_lufs"


def _finite_float(value: object) -> float | None:
    """Coerce a JSON scalar to a finite float, else None.

    `bool` is rejected explicitly: it is an `int` subclass, so a stray `true` would
    otherwise become the ceiling `1.0` and silently gate every render.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _coerce_window(value: object) -> list[float] | None:
    """Coerce a JSON `[lo, hi]` pair to two finite floats with lo <= hi, else None."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    items = list(value)
    if len(items) != 2:
        return None
    low, high = _finite_float(items[0]), _finite_float(items[1])
    if low is None or high is None or low > high:
        return None
    return [low, high]


_DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5450/w2v_orchestrator"  # pragma: allowlist secret
# There is deliberately NO default for DOWNSTREAM_LICENSE_TOKEN (see
# DownstreamServiceSettings.license_token). A hardcoded default used to live
# here; it was a live kag service credential, and being a default meant an
# unset env var silently authenticated as kag instead of failing. Removed from
# source 2026-10-08; it remains in git history (no rotation/rewrite decided).

# LLM-stage timeout ladder — single source of truth (see PipelineSettings.llm_budget_seconds).
# Every LLM text stage routes WM -> inference-router -> pgx and must wait out the same
# downstream read budget. The per-attempt budget MUST sit in the half-open window
# [downstream floor, outer per-stage ceiling): at or above the ~600s downstream read budget
# so a still-working call is never cut off early, and strictly below the outer
# TimeoutMiddleware ceiling so stage_call's own timeout fires first (a clean
# StageTimeoutError) rather than the blunt outer wrapper. These two constants are the
# single-source DEFAULTS for the env-configurable PipelineSettings.llm_downstream_floor_s /
# .llm_stage_base_ceiling_s — an LLM stage's StageSpec.default_timeout_seconds is None
# ("inherit"), which the DAG executor resolves to the runtime ceiling. A future slow-vLLM
# raise lands in GitOps (raise both together); the boot validator rejects a budget that
# forgets the ceiling instead of silently degrading to outer-wrapper timeouts.
LLM_DOWNSTREAM_FLOOR_S = 600.0
# The shipped per-attempt budget, as an IMPORTABLE module constant rather than only a
# Field default. services/stages/llm_budget.py sizes its output-token ceiling from this
# (tokens = seconds * measured tok/s) and must be able to read it WITHOUT instantiating
# Settings — those budgets are module-level constants at their call sites, evaluated at
# import, and reading get_settings() there would freeze whatever env happened to be set
# first. Keeping the two in one place is what stops the wall clock and the token budget
# drifting apart, which is exactly how #851 and #854 arrived at three different numbers.
LLM_BUDGET_SECONDS = 950.0
# The HTTP-client rung: a WMClient read timeout derived from whatever per-attempt
# budget a call site ACTUALLY uses, by subtracting a small ABSOLUTE delta.
#
# ⚠️ THE ORIGINAL JUSTIFICATION FOR THIS RUNG WAS FALSE. Read this before touching it.
#
# It shipped as a 0.9 FRACTION on this reasoning: WMClient's read timeout and
# stage_call's per-attempt budget were both 950.0; the stage clock starts first; so
# httpx.ReadTimeout could never be raised and "the in-stage retry was DEAD CODE —
# segment_select could only ever die terminally".
#
# The last clause is not true. ``stage_call`` (w2v_common/stages/execution.py:110-122)
# runs every attempt under ``asyncio.wait_for`` and catches its OWN ``TimeoutError``:
#
#     except TimeoutError as exc:
#         logger.warning("stage_call %s: attempt %d/%d timed out after %.1fs", ...)
#         if attempt == attempts:
#             raise StageTimeoutError(f"stage {stage_name!r}: all {attempts} attempts ...")
#
# A per-attempt timeout is therefore ALREADY retried; only the last one is terminal. The
# very sentence quoted as evidence — "all 2 attempts timed out" — is stage_call
# reporting that it DID retry, twice. The tie cost nothing, and this rung does not
# convert a terminal failure into a retryable one.
#
# What it does buy, stated honestly and narrowly: a typed ``httpx.ReadTimeout``
# (classified ``TransientUpstreamError`` by ``stages/upstream.py``) instead of a bare
# ``asyncio.wait_for`` cancellation, and a bound on the socket for any call path that is
# NOT wrapped in ``stage_call`` / ``asyncio.timeout``. Defence in depth, not a fix for a
# dead retry.
#
# ⚠️ WHY AN ABSOLUTE DELTA AND NOT A FRACTION — the slack is GENERATION time.
#
# At 0.9 the rung was 855 s under a 950 s budget. A call needing 855-950 s used to
# SUCCEED (generation ~900 s, budget 950 ⇒ returns); under the fraction it is aborted at
# 855 and retried, and ``stage_call`` gives attempt 2 a FRESH FULL budget — so the same
# prompt goes to the same model at the same token rate, blows 855 again, and the stage
# dies ``StageRetriesExhausted`` at ~1711 s instead of returning at ~900 s. The deployed
# local model was just swapped for a much slower one, so that is exactly the band live
# calls now sit in. A fraction also GROWS the theft as the budget rises: the slower the
# model, the more generation time it discards.
#
# The rung only has to break a TIE and cover PROPAGATION — httpx raising ReadTimeout,
# ``guarded()`` classifying it, ``stage_call`` recording the attempt and starting the
# next — which is sub-millisecond. 20 s is three orders of magnitude more than that, and
# its cost is bounded and small:
#
#     at the measured 11.5 tok/s (services/stages/llm_budget.py), a 20 s delta is
#         20 * 11.5   ~=   230 output tokens forfeited on the tail
#     against the 0.9 fraction's
#         95 * 11.5   ~=  1092 output tokens
#     i.e. 2.1% of the 950 s budget instead of 10.0%.
#
# 20 s also has to cover httpx's CONNECT and WRITE phases, because ``timeout=<float>``
# sets all four httpx timeouts to the same value — the rung must leave room for a
# connect on a saturated pool, not only for the read.
LLM_CLIENT_TIMEOUT_DELTA_S = 20.0
# Guard for SUBDIVIDED slices. ``client_timeout_for`` is applied to a SITE's own
# per-attempt timeout, which can be far smaller than the global budget (a per-batch share
# of a critic deadline, an inner ladder sized to fit inside one). Subtracting a flat 20 s
# from a 30 s slice would leave 10 s and make the CLIENT the binding constraint — the
# fraction's mistake, inverted. So the rung never drops below half its input: for every
# effective timeout ``t > 0``,
#
#     t * LLM_MIN_CLIENT_SHARE  <=  client_timeout_for(t)  <  t
#
# The absolute delta dominates at every realistic stage budget (950 - 20 = 930, far above
# 475); the share only engages below 2 * delta = 40 s, where a flat delta is the worse of
# the two evils.
LLM_MIN_CLIENT_SHARE = 0.5


def client_timeout_for(
    effective_timeout_seconds: float,
    *,
    delta_seconds: float = LLM_CLIENT_TIMEOUT_DELTA_S,
    min_share: float = LLM_MIN_CLIENT_SHARE,
) -> float:
    """The HTTP-client rung for a site whose OWN per-attempt budget is the argument.

    ⚠️ Pass the timeout THAT SITE hands ``stage_call`` / ``asyncio.timeout`` — never the
    global ``llm_budget_seconds`` unless the site actually uses it. A site that
    SUBDIVIDES the budget (a per-batch share of a shared critic deadline, an inner retry
    ladder sized to fit inside one slice) runs on a much smaller clock, and a rung
    derived from the global sits ABOVE that clock rather than below it — inert exactly
    where the ordering matters.

    Guarantees ``min_share * t <= result < t`` for every ``t > 0``.
    """
    if effective_timeout_seconds <= 0.0:
        raise ValueError(
            f"effective_timeout_seconds must be positive, got {effective_timeout_seconds}"
        )
    return max(
        effective_timeout_seconds - delta_seconds,
        effective_timeout_seconds * min_share,
    )


# 2026-08-19: 700 -> 1950, in lockstep with llm_budget_seconds 660 -> 950.
#
# The ceiling is a PER-STAGE budget and a stage is `llm_retries + 1` attempts, so
# it has to cover ALL of them. It never did: 660 * 2 = 1320 against a live
# ceiling of 900 meant the outer TimeoutMiddleware always killed attempt 2
# mid-flight — the blunt abort this ladder exists to prevent. 950 * 2 + 50 of
# margin = 1950, and `_validate_llm_timeout_ladder` now enforces that product
# rather than only the single-attempt bound, so the bug cannot come back.
LLM_STAGE_BASE_CEILING_S = 1950


# ---------------------------------------------------------------------------
# Shared infrastructure settings (imported from w2v_common, with overrides)
# ---------------------------------------------------------------------------


class DatabaseSettings(BaseDatabaseSettings):
    """Database settings with w2v_orchestrator defaults."""

    url: str = Field(
        default=_DEFAULT_DATABASE_URL,
        description="PostgreSQL database connection URL",
    )


class RedisSettings(BaseRedisSettings):
    """Redis settings with w2v_orchestrator defaults.

    Inherits ``health_check_interval`` from the shared base; the orchestrator
    opts in to a non-zero interval via the ``REDIS_HEALTH_CHECK_INTERVAL``
    env var (set to ``30`` in the orchestrator's K8s manifest).
    """

    # url is required (no localhost fallback) — REDIS_URL set via Secret in
    # the deployment manifests

    # The shared base defaults max_connections=10, which is far too small for
    # THIS service: every live SSE stream (GET /api/runs/{id}/events) holds one
    # pool connection for the stream's whole lifetime (redis pubsub), competing
    # with cache ops + the 30s health-ping for the same pool. With only 10, a
    # handful of concurrent run-viewers exhausts it → "Redis ping failed: Too
    # many connections" and 500s on /events → the SPA's EventSource retry-storms
    # and the run page hangs (this is what timed out the post-promote E2E's
    # multi-stage promo/website specs). Raise the ceiling well above realistic
    # concurrent-SSE + cache load; connections are cheap and Redis maxclients is
    # 10000. Overridable via REDIS_MAX_CONNECTIONS.
    max_connections: int = Field(default=100, description="Redis connection pool size")


#: Default hosts ``GET /api/media/thumb`` will proxy. DERIVED (2026-09-15) by
#: reading the media-discovery adapters (called directly since the KAG removal) — ``nonagenticai/media-discovery``
#: ``src/adapters/`` — not from general knowledge. Of the five registered adapters,
#: only three emit a ``thumbnail_url`` on a FIXED CDN:
#:   * ``youtube``   — ``snippet.thumbnails.{maxres,high,medium,default}.url``
#:                     (``youtube.py:294-299``) → ``i.ytimg.com`` / ``i9.ytimg.com``
#:   * ``vimeo``     — ``pictures.sizes[-1]["link"]`` (``vimeo.py:141``) → ``i.vimeocdn.com``
#:   * ``wikimedia`` — Commons ``imageinfo.url`` (``wikimedia.py:209,241``) → ``upload.wikimedia.org``
#: The other two are deliberately NOT here, and cannot be:
#:   * ``podcast``   — feed ``image``/``artwork`` (``podcast.py:213,274``) is whatever
#:                     host the RSS feed names. Unbounded. It is ENABLED by default
#:                     (``media-discovery`` config ``enable_podcast=True``), so podcast
#:                     hits are the bulk of the "broken thumbnail" the operator sees —
#:                     they now get a typed, explainable 400 instead of a bare one.
#:   * ``web_scraper`` — ``search()`` returns ``[]``, so it never reaches this proxy
#:                     at all; its ``get_item`` thumbnail is likewise arbitrary.
#: Allow-listing either would turn this route into an OPEN image proxy, so the
#: widening lever is CONFIG (``API_THUMB_ALLOWED_HOST_SUFFIXES``) — an operator can add
#: one known CDN without a code deploy, and the matching stays an ALLOW-list.
# Characters that cannot occur in a parsed hostname: they mean the operator wrote a
# scheme/port/path/query/userinfo instead of a bare suffix, which would match nothing.
_NON_HOSTNAME_CHARS: frozenset[str] = frozenset("/:@?#\\[]")

DEFAULT_THUMB_ALLOWED_HOST_SUFFIXES: tuple[str, ...] = (
    "ytimg.com",
    "vimeocdn.com",
    "upload.wikimedia.org",
)


class APISettings(BaseAPISettings):
    """API settings with w2v_orchestrator default port."""

    port: int = Field(default=6300, description="API server port")
    # Dedicated plaintext Prometheus metrics port. The main API port (:6300) is
    # under STRICT mTLS, so the cluster's non-ambient (plaintext) Prometheus
    # cannot scrape its /metrics. A second bare listener on this port — marked
    # PERMISSIVE by a workload-scoped PeerAuth and admitted by the w2v
    # NetworkPolicy in gitops — lets Prometheus scrape while :6300 stays STRICT.
    metrics_port: int = Field(default=6301, description="Dedicated Prometheus metrics port")
    # P9-7: POST /api/critique accepts up to analysis's 10 MiB html, and each such
    # request holds ~40-70 MiB (body, parsed str, re-encoded forward) for up to the
    # 60 s upstream call. Process-wide slots for bodies over the 256 KiB default;
    # a caller over the bound gets 429 + Retry-After instead of OOMing the pod.
    critique_large_body_slots: int = Field(
        default=4,
        ge=1,
        description="Concurrent POST /api/critique bodies over 256 KiB (429 when full)",
    )
    cors_origin_regex: str | None = Field(
        default=None,
        description=(
            "Regex for allowed CORS origins (used by Starlette's allow_origin_regex). "
            "Defaults to None so production must set API_CORS_ORIGIN_REGEX explicitly "
            "via the gitops ConfigMap; the dev compose file sets it to a localhost regex."
        ),
    )
    # SSRF boundary, PARTIAL — read this before trusting it.
    # `GET /api/media/thumb` fetches an ARBITRARY caller-supplied URL server-side. This
    # list is an ALLOW-list and must stay one, but it guards HOP 0 ONLY: the URL the
    # caller handed us. `w2v_common.net.safe_get` then follows up to
    # `DEFAULT_MAX_REDIRECTS=5` hops, and `_validate_redirect` re-checks each hop for
    # public-vs-internal but NEVER re-checks this allow-list. So an open redirector on
    # an allow-listed CDN lets a caller make us fetch an arbitrary PUBLIC url from our
    # egress — request laundering, not SSRF-to-internal (internal/link-local/loopback
    # targets stay blocked on every hop). Pre-existing, not a regression of the config
    # move; closing it needs a per-hop allow-list re-check in `w2v_common.net`, which is
    # a separate unit of work.
    #
    # ⚠️ Setting this env var REPLACES the default list — it does NOT add to it. An
    # operator who sets it to just `'["cdn.newcdn.example"]'` drops ytimg/vimeocdn/
    # wikimedia and every existing thumbnail 400s fleet-wide. To admit one CDN, repeat
    # ALL the defaults (`DEFAULT_THUMB_ALLOWED_HOST_SUFFIXES` above) plus the new entry.
    # Replacement is deliberate (the conventional shape for a settings list) and pinned
    # by `tests/api/routes/test_media.py::
    # test_thumb_allow_list_comes_from_config_not_a_module_literal`.
    #
    # Config, not a module literal, so a newly federated media adapter's CDN can be
    # admitted from the gitops ConfigMap instead of a code deploy. Set it as a JSON
    # array — `API_THUMB_ALLOWED_HOST_SUFFIXES=
    # '["ytimg.com","vimeocdn.com","upload.wikimedia.org","cdn.example.org"]'`.
    # (The base class ships a `_CsvFriendlyEnvSource`
    # meant to accept bare CSV too; measured 2026-09-15 against this pinned
    # pydantic-settings it no longer works for ANY list field — `API_CORS_ORIGINS="a,b"`
    # raises the same SettingsError — so JSON is the format that actually loads. That is
    # an upstream w2v_common bug, out of this change's lane; the validator below still
    # accepts CSV if it is ever fixed.)
    thumb_allowed_host_suffixes: list[str] = Field(
        default_factory=lambda: list(DEFAULT_THUMB_ALLOWED_HOST_SUFFIXES),
        description=(
            "Hosts GET /api/media/thumb may proxy, as registrable suffixes matched "
            "EXACTLY or on a dot-boundary (API_THUMB_ALLOWED_HOST_SUFFIXES, JSON array). "
            "Defaults to the fixed CDNs of the media-discovery adapters. An entry with "
            "no dot is REFUSED: a bare TLD like 'com' would allow-list the internet."
        ),
    )

    @field_validator("thumb_allowed_host_suffixes", mode="before")
    @classmethod
    def _normalize_thumb_suffixes(cls, value: object) -> list[str]:
        """Lower-case, strip stray leading dots/wildcards, and refuse a too-broad entry.

        An operator writing ``*.ytimg.com`` or ``.ytimg.com`` means the same suffix, and
        silently keeping the raw string would make the entry match NOTHING — a hole that
        LOOKS closed. A single-label entry (``com``, ``localhost``) is the opposite
        failure — it would match a whole TLD — so it raises at config load rather than
        widening the proxy at runtime.

        Same reasoning for anything that is not a bare hostname. The matcher in
        ``api/routes/media.py`` compares ``urlsplit(...).hostname`` — no scheme, no
        port, no path, no userinfo — so ``https://ytimg.com``, ``ytimg.com:8080``,
        ``evil.example/path`` or an entry with embedded whitespace can never equal or
        dot-suffix ANY host. Those pass a naive dot check and then silently allow
        nothing, so they raise at load too: a dead allow-list entry reads to the
        operator as an admitted CDN.
        """
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError, ValueError:
                parsed = None
            items = parsed if isinstance(parsed, list) else value.split(",")
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            msg = f"API_THUMB_ALLOWED_HOST_SUFFIXES must be a list of host suffixes, got {value!r}"
            raise TypeError(msg)
        cleaned: list[str] = []
        for item in items:
            suffix = str(item).strip().lower().removeprefix("*").lstrip(".").rstrip(".")
            if not suffix:
                continue
            if "." not in suffix:
                msg = (
                    f"API_THUMB_ALLOWED_HOST_SUFFIXES entry {item!r} has no dot; a "
                    "single-label suffix would allow-list an entire TLD"
                )
                raise ValueError(msg)
            if set(suffix) & _NON_HOSTNAME_CHARS or any(ch.isspace() for ch in suffix):
                msg = (
                    f"API_THUMB_ALLOWED_HOST_SUFFIXES entry {item!r} is not a bare "
                    "hostname; a scheme, port, path, userinfo or whitespace can never "
                    "match a parsed host, so the entry would allow nothing at all"
                )
                raise ValueError(msg)
            if suffix not in cleaned:
                cleaned.append(suffix)
        return cleaned


class StorageSettings(BaseStorageSettings):
    """Storage settings with w2v_orchestrator defaults."""

    # minio_endpoint is required (no localhost fallback) — STORAGE_MINIO_ENDPOINT
    # set via per-service ConfigMap envFrom in the fleet deployment configuration.
    minio_bucket: str = Field(default="w2v-orchestrator", description="MinIO bucket name")
    # Kept at the base 3600s (1h) default — deliberately. Browser-facing presigns
    # (uploads_video original_url, storyboard panel thumbnails, the final-artifact
    # rewrite path) must NOT get the long compile-input TTL below; scoping the
    # long TTL to browser links too would silently double how long a leaked link
    # stays valid. The compile-input TTL lives ONLY in
    # ``services/stages/shot_presign.py::refresh_shot_asset_urls`` (via
    # ``compile_budget.PRESIGN_TTL_SECONDS``) and ``tts_storage.presign_sound_asset``
    # (the music/ambient bed and per-shot SFX), which are the presign paths a
    # compile job's own inputs go through — see shot_presign for the 2026-09-17
    # incident (``shot_asset_unfetchable``) this exists to fix.
    minio_external_endpoint: str | None = Field(
        default=None,
        description="External MinIO endpoint for presigned URLs (replaces internal host when set)",
    )
    # P3b — cross-run content-addressed media cache (env STORAGE_MEDIA_CACHE_*).
    media_cache_cross_run: bool = Field(
        default=False,
        description=(
            "When true, content-addressed media (AI images/videos, slides) is stored under a "
            "cross-run, TENANT-scoped 'media-cache/tenants/<tenant_id>/' prefix so identical "
            "generative inputs reuse a single blob across runs — not just within one run's retries "
            "(P2). Off = per-run storage (exact P2 behavior). Requires a tenant bound in the "
            "identity context; with no tenant, storage falls back to per-run. Scoped per tenant "
            "deliberately: a prompt hash is not an authorization boundary, so media is never shared "
            "across tenants."
        ),
    )
    media_cache_ttl_days: int = Field(
        default=30,
        description=(
            "TTL (days) for the 'media-cache/' prefix bucket lifecycle rule; 0 disables GC. A "
            "cross-run cache blob has no owning run to reclaim it (runs are soft-deleted and never "
            "delete media), and a _MEDIA_GEN_VERSION bump orphans old ids, so a bucket lifecycle "
            "expiration is the only reclamation path."
        ),
    )
    # P6 ORCH-4 tenant music library (env STORAGE_TENANT_MUSIC_MAX_BYTES).
    tenant_music_max_bytes: int = Field(
        default=50 * 1024 * 1024,
        gt=0,
        description=(
            "Largest audio file a tenant_admin may upload to POST /api/tenant/music. The "
            "body is read into memory once (sniff + ffprobe), so this bounds per-request "
            "heap. 50 MiB holds ~5 min of 16-bit stereo WAV or a long 320 kbps MP3 bed."
        ),
    )
    # Persistent cross-run source-ingest cache (env STORAGE_SOURCE_INGEST_CACHE_*).
    source_ingest_cache_enabled: bool = Field(
        default=True,
        description=(
            "When true, source_ingest memoizes each brief.sources[] ingest (webscraper scrape / "
            "in-process document parse + "
            "republished media) in a persistent, tenant-scoped DB cache so many runs — notably "
            "concurrent course runs — over the same source reuse one ingest instead of each "
            "re-scraping/re-parsing. Off = every run ingests directly. Any cache-layer "
            "error degrades to a direct ingest, so this never fails a run."
        ),
    )
    source_ingest_cache_ttl_seconds: int = Field(
        default=604800,
        description=(
            "TTL (seconds, default 7d) for a cached source ingest before it is refreshed on next "
            "use. Source reference material is stable, so a week comfortably spans a concurrent "
            "course burst. Keep <= media_cache_ttl_days so a still-'ready' row never references a "
            "lifecycle-GC'd image blob (the consume-time object_exists probe is the backstop)."
        ),
    )


# ---------------------------------------------------------------------------
# Orchestrator-specific settings (not from w2v_common)
# ---------------------------------------------------------------------------


class PipelineSettings(BaseSettings):
    """Pipeline execution settings."""

    mode: str = Field(default="gateway", description="gateway or monolith")
    default_timeout_seconds: int = Field(default=7200)
    max_poll_attempts: int = Field(default=500)
    # ⛔ SEVEN FIELDS WERE DELETED FROM THIS BLOCK 2026-08-20, and the reason is worth
    # keeping: `max_concurrent_pipelines`, `retry_max_attempts`, `retry_delay_seconds`,
    # `poll_interval_seconds`, `max_pages_per_crawl`, `crawl_batch_size` and
    # `bundle_signing_secret` were read NOWHERE in `src/`. Setting any of them — via env or
    # the gitops ConfigMap — did exactly nothing, silently: no error, no log line. An
    # operator reaching for MAX_CONCURRENT_PIPELINES to throttle a saturated pod got
    # silence. The block looked alive because `max_poll_attempts` beside them genuinely is
    # read (`src/clients/base.py`), and `tests/test_config.py` asserted the dead ones'
    # defaults, so CI was green and reinforced the belief.
    #
    # `tests/test_no_dead_settings_surface.py` now fails on any field with no reader in
    # `src/`. Before adding a knob here, add its consumer — or the gate will say so.
    # Single source of truth for the LLM-stage timeout ladder. Every LLM text
    # stage routes WM -> inference-router -> pgx and must wait out the same
    # downstream read budget instead of cutting a still-working call off early.
    #
    # 660 -> 950 on 2026-08-19, and this number is NOT a comfort margin — it is
    # what decides how many tokens a reasoning model is allowed to think for.
    # A WM call that exhausts `max_tokens` while still emitting ThinkingParts
    # dies with a TERMINAL 422 ("Model token limit (N) exceeded before any
    # response was generated"), which kills the whole run. The budget therefore
    # has to be large enough that `max_tokens` can be large enough. At the
    # measured 11.5 tok/s of the deployed dense Qwen3.8-27B on pgx:
    #
    #     660s -> ~7_590 tokens   <- barely above the 6000 that ALREADY 422s
    #     950s -> ~10_925 tokens
    #
    # 6000 is worker-management's own default and is an OBSERVED production
    # failure on promo_script, promo_analyze, website_review_script,
    # growth_bundle, retention_plan and segment_select. So the old window was
    # (6000, 7590) — a fix had nowhere to land. This is the wall-clock half of
    # opening it; the token half lives in services/stages/llm_budget.py.
    #
    # Bounded above by the DEPLOYED downstream caps, not by code defaults:
    # worker-management AGENT_RUN_TIMEOUT_S=1200 and inference-router
    # INFERENCE_ROUTER_CHAIN_DEADLINE_SECONDS=1200 (both raised 2026-08-19 for
    # this same slow local model). 950 leaves 250s of margin under them.
    llm_budget_seconds: float = Field(
        default=LLM_BUDGET_SECONDS,
        description="Per-attempt wall-clock budget (s) for every LLM stage's "
        "WMClient + stage_call. Env PIPELINE_LLM_BUDGET_SECONDS.",
    )
    # The HTTP-client rung of the ladder, STRICTLY below the per-attempt timeout it is
    # derived FROM. An absolute delta, not a fraction: the slack is generation time, and
    # a fraction scales the theft up precisely when the model gets slower. `gt=0.0` is
    # what makes the strict ordering structural — no value of this field can recreate the
    # tie. See LLM_CLIENT_TIMEOUT_DELTA_S for the arithmetic and for the false premise
    # ("the retry was dead code") this replaces.
    llm_client_timeout_delta_s: float = Field(
        default=LLM_CLIENT_TIMEOUT_DELTA_S,
        gt=0.0,
        description="Seconds subtracted from a call site's OWN per-attempt timeout to "
        "get its WMClient read timeout. Must be > 0 so the client rung is strictly below "
        "the budget the stage clock is counting down. Env "
        "PIPELINE_LLM_CLIENT_TIMEOUT_DELTA_S.",
    )
    # ge/le are the SAME bounds `stage_call`'s `_validate_retries` enforces, so an
    # out-of-range PIPELINE_LLM_RETRIES is refused at config load instead of raising
    # ValueError inside a stage handler at runtime. Before this, the only thing
    # bounding the field from above was `_validate_llm_timeout_ladder`'s
    # `budget x attempts < ceiling` product — which fails for the WRONG reason
    # (it names the ceiling, never the retry cap), and whose own error text invites
    # raising the ceiling in lockstep. Doing that made >cap REACHABLE, trading a
    # boot failure for ~40 call sites that crash mid-run.
    llm_retries: int = Field(
        default=1,
        ge=0,
        le=STAGE_CALL_MAX_RETRIES,
        description="stage_call retries for LLM stages (attempts = retries+1). "
        "1 = one patient attempt + one opportunistic fast-fail retry; avoids "
        "re-queue thrash under vLLM contention. Hard-capped at "
        f"{STAGE_CALL_MAX_RETRIES} (w2v_common stage_call's own limit). "
        "Env PIPELINE_LLM_RETRIES.",
    )
    # The two ends of the LLM-stage timeout ladder, now env-configurable so a
    # slow-vLLM raise lands in GitOps (not a code edit). The module constants
    # LLM_DOWNSTREAM_FLOOR_S / LLM_STAGE_BASE_CEILING_S are the single-source
    # defaults. The ceiling is the outer per-stage TimeoutMiddleware budget for
    # every LLM stage that does not override it (StageSpec.default_timeout_seconds
    # is None -> the DAG executor resolves it to this value); heavier stages
    # (redesign, record, child-run fanout) still pin their own explicit ceiling.
    # _validate_llm_timeout_ladder keeps llm_budget_seconds inside
    # [floor, ceiling) so raising one without the other fails at boot.
    llm_stage_base_ceiling_s: int = Field(
        default=LLM_STAGE_BASE_CEILING_S,
        description="Outer per-stage TimeoutMiddleware ceiling (s) for LLM stages "
        "that don't override it; must stay strictly above llm_budget_seconds. "
        "Env PIPELINE_LLM_STAGE_BASE_CEILING_S.",
    )
    llm_downstream_floor_s: float = Field(
        default=LLM_DOWNSTREAM_FLOOR_S,
        description="Lower bound of the llm_budget_seconds window — the downstream "
        "read budget below which a still-working call is cut off early. "
        "Env PIPELINE_LLM_DOWNSTREAM_FLOOR_S.",
    )
    course_outline_split: bool = Field(
        default=False,
        description=(
            "Pillar-2 vLLM split for course_outline. False (default) => ONE WM "
            "call, byte-identical. True => a structure call (section titles + "
            "intro_summary) + an objective-batch call, stitched deterministically "
            "so each draw stays small on the single slow GB10 vLLM. Empty/malformed "
            "structure, or objectives-vs-titles length mismatch, falls back to the "
            "one-shot path. Env PIPELINE_COURSE_OUTLINE_SPLIT."
        ),
    )
    timing_director_model_tier: str | None = Field(
        default=None,
        description=(
            "Optional model tier for the reconcile_timeline C13 relief loop's "
            "timing-director + beat-rewriter WM calls. None (default) => the "
            "workers fall through the inference-router default_chain to the small "
            "local model (byte-identical to today). Set to an inference-router "
            "tier alias (e.g. 'tier-medium' => deepseek-v4-pro, 'tier-high' => "
            "grok-4.5) to route the directive-proposal + per-beat rewrite to a "
            "more capable model when the local model under-performs. This is an "
            "operational escape hatch flipped in GitOps, not a code change. Env "
            "PIPELINE_TIMING_DIRECTOR_MODEL_TIER."
        ),
    )
    max_stage_attempts: int = Field(
        default=2,
        description="Durability P1: how many times a genuinely-slow stage that "
        "times out may be RESUMED before the run terminal-fails. ⚠️ THIS IS A "
        "RESUME COUNT, NOT AN EXECUTION COUNT, despite the field name: the check "
        "is `attempts <= cap` on the POST-bump counter, so N permits N+1 RUNS of "
        "the stage body. At the default 2 that is THREE executions. Sizing a "
        "wall-clock budget from the name gives 2x the stage ceiling; the truth is "
        "3x — for an LLM stage inheriting LLM_STAGE_BASE_CEILING_S (1950s) that is "
        "5850s, not 3900s, and the e2e harness deadline for the same blueprint is "
        "3600s. Pinned by "
        "tests/services/runs/test_dag_executor.py::"
        "test_the_cap_is_a_resume_count_so_n_permits_n_plus_1_executions. "
        "⚠️ The resume ladder is bounded against the RUN by RUN_DEADLINE_SECONDS "
        "(below) and by nothing else — this knob alone multiplies across stages. "
        "A stage that timed out because the model was slow is re-run with a "
        "DOWNSHIFTED reasoning budget (see llm_budget.downshift_reasoning), not an "
        "identical one, so a later attempt is a different call rather than the same "
        "unanswerable one. Each timeout increments "
        "stage_outputs.attempts; while under the cap the run is left 'running' for "
        "the reconciler to resume from the last completed stage. "
        "Env PIPELINE_MAX_STAGE_ATTEMPTS.",
    )
    run_deadline_seconds: int = Field(
        default=14400,
        ge=0,
        description="Wall clock a whole RUN may not outlive, measured from "
        "`runs.started_at`. 0 disables it (pre-2026-08-21 behaviour). "
        "⚠️ RE-SIZED 2026-08-23 FOR THE NEW MODEL, 7200 -> 14400. The original "
        "7200 was derived from 30 days of runs that were almost entirely BEFORE "
        "the pgx vLLM swap (MoE gemma-4-26B-A4B -> dense Qwen3.8-27B-FP8, unit "
        "started 2026-08-18 23:15). Post-swap the LLM path is 5-23x slower by "
        "blueprint, and `website_review_v1` now COMPLETES at up to 7814s — i.e. "
        "the old bound sat BELOW a legitimate completion and would have "
        "terminal-failed it. The slowdown is an accepted trade (the new model is "
        "slower but better); the budget has to follow it. "
        "Post-swap completed maxima, measured: website_review 7814, demo_video "
        "3121, promo 2708, action_reel 1396. 14400 gives the worst of those 1.8x "
        "headroom and still cuts the post-swap failed tail (website_review burned "
        "19407s = 5.4h). "
        "⚠️ READ THIS BEFORE COMPARING IT TO RUN DURATIONS: it is checked at ONE "
        "point, the resume decision in dag_executor, so it can only ever fire on a "
        "run that has ALREADY had a stage time out. A run that succeeds never "
        "reaches the check, and a stage already in flight is never cancelled "
        "mid-work. That is why a value below the longest observed SUCCESSFUL run is "
        "still safe: the two populations do not meet. "
        "⚠️ This is the ONLY bound on total run wall-clock. `max_stage_attempts` is "
        "per-STAGE, so N slow stages multiply: nothing stopped a run resuming its "
        "way through hours. Measured over 30 days of production runs before this "
        "existed: FIVE runs burned more than two hours each, ALL of them already "
        "lost, the worst 34909s = 9.7 HOURS (website_review_v1, "
        "`website_redesign.wait`, 'all 8 attempts timed out'); then 10969s, 9666s, "
        "7947s, 7261s. The longest run that ever SUCCEEDED in that same window was "
        "5176s (1.44h, website_review_v1) and every blueprint's p95 completed "
        "duration is under 2400s. So the failed tail ran 6.7x the successful "
        "maximum. 7200s would have terminated all five and, per the paragraph "
        "above, could not have touched any of the runs that completed. A tighter "
        "per-blueprint value belongs in GitOps once each blueprint's own p99 is "
        "watched — website_review_v1 and course_v1 are the only two whose p95 "
        "approaches 2400s. "
        "Env PIPELINE_RUN_DEADLINE_SECONDS.",
    )
    demo_qa_enabled: bool = Field(
        default=True,
        description=(
            "Per-clip visual QA in demo_video_record: each recorded clip is run "
            "through video_analyzer's quality path (frame-extract + OpenCV "
            "sharpness) and a blank/flat clip is surfaced via the stage's `notes` "
            "→ `degraded` (the v12 fix: a white/short-rendered clip at the right "
            "*length* is otherwise byte-indistinguishable from a good one). "
            "Best-effort — an analyzer outage never fails the run. Disable with "
            "PIPELINE_DEMO_QA_ENABLED=false."
        ),
    )
    demo_qa_strict: bool = Field(
        default=False,
        description=(
            "When true, a clip that fails visual QA FAILS the run instead of only "
            "degrading it. Default false = notes + degraded (soft). "
            "Env PIPELINE_DEMO_QA_STRICT."
        ),
    )
    demo_qa_min_sharpness: float = Field(
        default=3.0,
        description=(
            "min_sharpness_score criterion for per-clip QA — video_analyzer's "
            "mean Laplacian variance. A flat white/black frame is ≈0; a real UI "
            "is well above this. Deliberately conservative so a legitimately "
            "sparse UI doesn't false-positive (a false positive only *degrades*, "
            "never blocks). Env PIPELINE_DEMO_QA_MIN_SHARPNESS."
        ),
    )
    demo_qa_fail_sharpness: float = Field(
        default=1.0,
        description=(
            "Blank-clip HARD-FAIL threshold on video_analyzer's mean Laplacian "
            "variance. A clip whose measured sharpness is below this is a genuinely "
            "BLANK recording (the app never loaded — prod blanks measure exactly "
            "0.0) and FAILS the run outright, independent of `demo_qa_strict`, "
            "rather than shipping `completed`+degraded. Distinct from and BELOW "
            "`demo_qa_min_sharpness` (the softer 3.0 degrade band): [fail, min) "
            "still only degrades. Default 1.0 sits in the wide gap between blank "
            "(~0) and any real UI (>=3), so it fires on real blanks with ~zero "
            "false-fail risk. Effective value is clamped to never exceed "
            "`demo_qa_min_sharpness`, so a hard-fail can never fire on a clip the "
            "degrade band would pass. Set 0.0 to disable the hard gate (degrade-"
            "only, the pre-fix behaviour). Env PIPELINE_DEMO_QA_FAIL_SHARPNESS."
        ),
    )
    demo_timeline_underfull_fail_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "GROSS-UNDERFILL HARD-FAIL threshold for `reconcile_timeline`, as a "
            "fraction of the deliverable's target duration. When the reconciled "
            "timeline fills LESS than this share of the target (production run "
            "01a00b81 shipped 2.32s of a 60.00s target = 0.039 and reported "
            "`completed`), the stage FAILS the run instead of shipping a video that "
            "is a different deliverable from the one that was ordered. Distinct "
            "from and BELOW the soft UNDERFULL note, which keeps firing for every "
            "shortfall outside the `tolerance_pct` band: a MILDLY short timeline "
            "(55s of 60s = 0.92) is a pacing preference and must only degrade. "
            "Default 0.5 sits in the wide gap between the two — half the ordered "
            "runtime missing cannot be a preference, and no legitimate spine lands "
            "there — so it fires on real collapses with ~zero false-fail risk. Set "
            "0.0 to disable the hard gate (note-only, the pre-fix behaviour); the "
            "note itself is emitted either way. Freeze pad (held frames) counts "
            "toward the fill only up to `time_budget.MAX_PAD_CREDIT_RATIO` (10%) of "
            "target, so padding can never by itself carry a run over this floor — "
            "the effective requirement is captured content >= (ratio - 0.10) x "
            "target. Live run 01a059a2 shipped a blank 5s order on 1.40s of footage "
            "because uncapped pad scored it 58%. "
            "Env PIPELINE_DEMO_TIMELINE_UNDERFULL_FAIL_RATIO."
        ),
    )
    demo_qa_min_overall_quality: float = Field(
        default=0.0,
        description=(
            "Optional secondary min_overall_quality criterion (0-1 blend). "
            "Left at 0.0 (disabled) by default — the sharpness gate is the "
            "primary blank detector and overall_score normalisation is looser. "
            "Set >0 to enable. Env PIPELINE_DEMO_QA_MIN_OVERALL_QUALITY."
        ),
    )
    demo_qa_ocr_enabled: bool = Field(
        default=False,
        description=(
            "O1b wrong-screen detection: when a clip's authored line carries an "
            "`expected_text`, assert it via video_analyzer OCR (`required_ocr_text`) "
            "so a clip that recorded the WRONG screen (e.g. a 'Logging out' "
            "interstitial) at the right *length* is flagged — the other half of the "
            "v12 fix. OFF by default: OCR is tesseract-backed and a mis-provisioned "
            "analyzer without it yields empty OCR → a spurious 'text not found'. Even "
            "when enabled, an OCR miss only DEGRADES (never hard-fails under strict). "
            "Env PIPELINE_DEMO_QA_OCR_ENABLED."
        ),
    )
    demo_require_footage: bool = Field(
        default=True,
        description=(
            "Total-footage-loss gate in demo_video_record: when a footage SOURCE was "
            "present (brief.package_url or a demo_video_gen bdd_job_id) but "
            "frontend_test_runner returned ZERO scenario clips, FAIL the run instead of "
            "degrading it to narrated text cards. FTR reports status='completed' with an "
            "empty scenario_clips when its Playwright run times out and salvages "
            "nothing, so the wait succeeds and the customer is otherwise charged "
            "`completed`+degraded for a video with no footage. NARROW by design: a "
            "PARTIAL harvest, a tag-naming mismatch and a storage outage all still "
            "degrade (soft footage-gap notes), because clips FTR did return are not a "
            "capture failure. Set false to restore the degrade-only behaviour. "
            "Env PIPELINE_DEMO_REQUIRE_FOOTAGE."
        ),
    )
    promo_require_site_grounding: bool = Field(
        default=True,
        description=(
            "Site-grounding gate in promo_script: when true, a promo run whose site "
            "scrape was blocked or failed hard-fails instead of silently producing an "
            "ungrounded script. site_capture reports `page_text_status`; on 'blocked' "
            "or 'fetch_failed' the prompt would otherwise drop the grounding block "
            "without a trace and the LLM invents product features. An 'empty' scrape "
            "(the site really has no readable copy) is NOT a failure — it states the "
            "absence in the prompt and degrades the run via a stage note. Set false to "
            "generate an ungrounded script anyway. "
            "Env PIPELINE_PROMO_REQUIRE_SITE_GROUNDING."
        ),
    )
    # ── WS-1: opt-in stage activation defaults ───────────────────────────────
    # Eight stages guard themselves on a presence-is-opt-in config block and raise
    # StageSkippedError when it is absent. Nothing in the product ever authored those
    # blocks, so all eight had NEVER completed in production. `build_config_snapshot`
    # now injects a conservative default block per stage, each behind its OWN flag so
    # a misbehaving stage is killable from the GitOps configmap without a code deploy
    # (and without taking the other seven down with it). A per-run `config_overrides`
    # value or a brief-authored block always WINS — injection is a leaf-level
    # `setdefault`, never a clobber.
    ship_gate_default_enabled: bool = Field(
        default=True,
        description=(
            "Seed a default `ship_gate: {mode: critique_only}` when the run authored "
            "none, so the A11 advisory ship gate (`final_critique`) actually runs. "
            "`critique_only` is strictly non-mutating: only `mode: auto` triggers the "
            "in-stage re-drive, so the delivered artifact stays byte-identical and the "
            "stage contributes scores/findings only. Env "
            "PIPELINE_SHIP_GATE_DEFAULT_ENABLED."
        ),
    )
    ship_gate_default_enforce: bool = Field(
        default=False,
        description=(
            "Seed `ship_gate.enforce = True` (an OPT-IN HARD gate) into the default "
            "`ship_gate` block. Enforce arms EVERY ship-gate term on every mode the "
            "`final_critique` stage does not skip: a draft with an armed term that "
            "affirmatively FAILS is FAILED OUTRIGHT (no artifact ship). `mode` is only the "
            "repair axis — `auto` first runs the bounded ≤2-pass re-drive and judges the "
            "best draft; `critique_only` judges the first draft with no re-drive. An "
            "insufficient-evidence term never blocks. OFF by default: with enforce false "
            "only individually armed terms can block. "
            "Like `ceilings`, this rides INSIDE the seeded default block and is "
            "seeded only when True, so the default path stays byte-identical; a per-run "
            "`ship_gate.enforce` override always wins. Env PIPELINE_SHIP_GATE_DEFAULT_ENFORCE."
        ),
    )
    ship_gate_default_ceilings: str = Field(
        default="",
        description=(
            "JSON object of CALIBRATED A11 ship-gate ceilings, folded into the seeded "
            '`ship_gate` default block. Example: \'{"momentum_gap_max": 0.12, '
            '"continuity_l2": 0.34, "cohesion_floor": 0.71, "loudness_lufs": [-16.5, -12.1]}\'. '
            "Empty (the default) seeds NO `ceilings` key, so the gate keeps its code "
            "defaults and stays advisory — byte-identical to not setting this at all. "
            "Values MUST come from `scripts/gestalt_calibration_report.py`, which refuses "
            "below n=20 samples and validates that every suggestion rejects ZERO observed "
            "samples; hand-picked numbers can blind-block real videos. Malformed JSON, "
            "unknown keys, non-finite/boolean numbers and bad windows are logged and "
            "DROPPED rather than raised — this is parsed on the run-setup path, so a bad "
            "configmap value must never break run creation. Env "
            "PIPELINE_SHIP_GATE_DEFAULT_CEILINGS."
        ),
    )

    # ── A11 ship-gate ARMING (per TERM), one axis per concept ────────────────
    # ⛔ TWO SEPARATE AXES, deliberately not folded into one field:
    #   * `ship_gate.mode` (critique_only / auto / off) says what the gate DOES about a
    #     failure — nothing, or a bounded in-stage re-drive. It is a REPAIR axis.
    #   * these booleans say whether an AFFIRMATIVE failure of one term BLOCKS the run.
    # Before this, neither axis could block: the seeded default block is
    # `{"mode": "critique_only"}` with no `enforce` key, and the only hard-fail site
    # (`_enforced_accept`) was reachable ONLY from the `auto` re-drive AND required
    # `ship_gate.enforce`. Both conditions were false in production simultaneously, so
    # `ship_ready` was a verdict that explained and never decided.
    #
    # Each term is armed by its OWN switch so a miscalibrated term is backed out from
    # the GitOps configmap (env `PIPELINE_SHIP_GATE_<TERM>_ARMED=false`) without a code
    # change or a rollout — the `mastering_gate_*_enabled` shape in
    # `video_compiler_service/src/services/mastering.py::gate_armed`. A per-run
    # `ship_gate.enforce: true` still escalates and arms EVERY term.
    #
    # ⚠️ Read at STAGE time, not seeded into `config_snapshot`: a run created before a
    # switch flipped must obey the flip, which a snapshot copy would not.
    ship_gate_loudness_armed: bool = Field(
        default=False,
        description=(
            "ARM the ship gate's loudness term: an AFFIRMATIVE `loudness_verdict == "
            "'fail'` (the compiler's measured integrated LUFS outside the window) FAILS "
            "the run. `insufficient_evidence` (unmeasurable loudness) can NEVER fail "
            "— it is a third state, not a failure. Env PIPELINE_SHIP_GATE_LOUDNESS_ARMED."
            "\n\n"
            "⛔ DISARMED 2026-09-16 — the fail rate this switch was armed without is now "
            "MEASURED, and it is 100%. When armed it read: 'the live fail RATE is "
            "UNMEASURED (n=0); this is the accepted false-failure risk, and this switch "
            "is the back-out.' Measured on RS w2v-orchestrator-7b5bbcf5cf (#1025 live, "
            "gate-proven in-pod): 3 of 3 demo_video runs failed final_critique at "
            "-14.2 / -14.1 / -14.1 LUFS against the (-14, -11) window. "
            "\n\n"
            "The window is NOT the defect and must not be widened to accommodate this: "
            "the ratified master target is -12.5 LUFS (`mastering_plan.py`), so delivery "
            "is 1.6-1.7 LU BELOW target — an order of magnitude past single-pass "
            "loudnorm tolerance. The arming rationale assumed loudnorm 'lands inside by "
            "design'; measurement falsifies that assumption, not the band. The gate was "
            "convicting a real defect — a systematically quiet speech bus — that nothing "
            "downstream of it can fix. "
            "\n\n"
            "Why disarm rather than leave it enforcing: `final_critique` is the "
            "`delivered_artifact_stage` for BOTH demo_video and showcase, so a fail "
            "cancels growth_bundle + render_manifest on both; and structural relief is "
            "only spawned after `mark_completed` on a transitioned run "
            "(`run_dispatcher.py`), so an enforcing loudness fail also silently disables "
            "demo_video auto-convergence entirely. RE-ARM once the speech-bus level is "
            "fixed — this is a one-line change and the term is otherwise sound."
        ),
    )
    ship_gate_momentum_armed: bool = Field(
        default=True,
        description=(
            "ARM the ship gate's momentum term. Safe to arm by construction: "
            "`momentum_verdict` returns `unconfigured` whenever no "
            "`ceilings['momentum_gap_max']` is set — which is the live state, nothing "
            "anywhere configures it — and `unconfigured` reads as insufficient evidence, "
            "never a fail. So arming this today cannot fail a single run; it is the SEAM "
            "that goes live the moment a calibrated ceiling is configured, instead of "
            "needing a code change then. Env PIPELINE_SHIP_GATE_MOMENTUM_ARMED."
        ),
    )
    ship_gate_duration_armed: bool = Field(
        default=False,
        description=(
            "ARM the ship gate's DURATION term — the delivered-vs-ORDERED verdict from "
            "`stages/shared/duration_conformance`, READ not re-invented (this gate "
            "defines no duration ceiling of its own). A 218.47s film against a 30.0s "
            "order shipped green because the gate had no duration term at all; wiring "
            "the existing verdict closes that blindness. NOT ARMED by default, and that "
            "is a MEASURED refusal, not caution: that module records 273% of order on "
            "`website_review`, 216% on `lesson`, 140% on `app_review` and 129% on "
            "`promo` from live artifacts (2026-09-10). Arming it now would fail the "
            "MAJORITY of deliveries over an overshoot the ship gate cannot remediate "
            "(`critique_only` runs no re-drive) — the 'a signal that fires on everything "
            "is not a signal' trap. Flip this to true once the duration wave lands and "
            "the rate is re-measured. Env PIPELINE_SHIP_GATE_DURATION_ARMED."
        ),
    )
    ship_gate_continuity_armed: bool = Field(
        default=False,
        description=(
            "ARM the ship gate's continuity term. NOT ARMED, and unarmABLE today: "
            "`video_analyzer_service` emits no per-frame brightness / sharpness / "
            "dominant_colour, so `_feature_vector` substitutes whole-video means and "
            "pads [0, 0, 0] — five of six components are constant BY CONSTRUCTION and a "
            "static screen capture zeroes the sixth. The 0.5 ceiling is hardcoded and "
            "was never calibrated (`continuity_stats` exists precisely because the "
            "evidence to calibrate it does not exist). Arming a term over noise "
            "manufactures failures. Flip only after the analyzer emits real per-frame "
            "visual features. Env PIPELINE_SHIP_GATE_CONTINUITY_ARMED."
        ),
    )
    ship_gate_cohesion_armed: bool = Field(
        default=False,
        description=(
            "ARM the ship gate's cohesion term. NOT ARMED — same structural "
            "unobservability as continuity, plus a recorded measurement that settles it: "
            "`gestalt_metrics` documents a measured cohesion of 0.0082 against a "
            "`_DEFAULT_COHESION_FLOOR` of 0.5, i.e. arming this would fail ~100% of "
            "masters on an axis carrying no signal. Env "
            "PIPELINE_SHIP_GATE_COHESION_ARMED."
        ),
    )

    def resolved_ship_gate_ceilings(self) -> dict[str, object]:
        """Parse `ship_gate_default_ceilings` into a validated ceilings mapping.

        Fail-SOFT by construction: every rejection path logs and returns/skips rather
        than raising, because this runs while building a run's `config_snapshot`. The
        w2v_common 0.49.0 lesson applies — an unguarded `float(os.getenv(...))` on a
        request path turned a typo'd env var into a 500.
        """
        raw = self.ship_gate_default_ceilings.strip()
        if not raw:
            return {}
        try:
            parsed: object = json.loads(raw)
        except ValueError:
            logger.warning(
                "PIPELINE_SHIP_GATE_DEFAULT_CEILINGS is not valid JSON; ignoring it and "
                "leaving the ship gate on its code defaults."
            )
            return {}
        if not isinstance(parsed, dict):
            logger.warning(
                "PIPELINE_SHIP_GATE_DEFAULT_CEILINGS must be a JSON object, got %s; ignoring.",
                type(parsed).__name__,
            )
            return {}
        ceilings: dict[str, object] = {}
        for key, value in parsed.items():
            if key == _SHIP_GATE_WINDOW_CEILING:
                window = _coerce_window(value)
                if window is None:
                    logger.warning(
                        "ship_gate ceiling %r must be a [lo, hi] pair of finite numbers "
                        "with lo <= hi; dropping it.",
                        key,
                    )
                    continue
                ceilings[key] = window
            elif key in _SHIP_GATE_SCALAR_CEILINGS:
                number = _finite_float(value)
                if number is None:
                    logger.warning(
                        "ship_gate ceiling %r must be a finite number; dropping it.", key
                    )
                    continue
                ceilings[key] = number
            else:
                logger.warning(
                    "unknown ship_gate ceiling %r; dropping it (known: %s).",
                    key,
                    ", ".join(sorted({*_SHIP_GATE_SCALAR_CEILINGS, _SHIP_GATE_WINDOW_CEILING})),
                )
        return ceilings

    footage_direction_default_enabled: bool = Field(
        default=True,
        description=(
            "Seed a default `direction.footage` block when the run authored none, so "
            "the directed-footage planner (`*_footage_direction`) runs instead of "
            "skipping. Note this block is ALSO the compile-side gate for binding the "
            "VP-2 focus/cursor sidecar onto shots; `auto_zoom` is deliberately left "
            "unset so no capture_scale bump is requested. Env "
            "PIPELINE_FOOTAGE_DIRECTION_DEFAULT_ENABLED."
        ),
    )
    footage_critique_default_enabled: bool = Field(
        default=True,
        description=(
            "Seed `direction.footage.critique = true` when the run authored no "
            "critique sub-flag, so the footage-quality critique loop runs. Nested "
            "UNDER the footage block, so it only takes effect on a run that has one "
            "(authored, or injected by the sibling flag above). Env "
            "PIPELINE_FOOTAGE_CRITIQUE_DEFAULT_ENABLED."
        ),
    )
    retention_default_enabled: bool = Field(
        default=True,
        description=(
            "Seed a default `direction.retention: {mode: critique_only}` when the run "
            "authored none, so the retention planner runs advisory-only (it also "
            "materializes the flat `retention_weights` block the growth bundle reads). "
            "Env PIPELINE_RETENTION_DEFAULT_ENABLED."
        ),
    )
    timing_default_enabled: bool = Field(
        default=True,
        description=(
            "Seed a default `direction.timing: {mode: auto}` when the run authored "
            "none, so the deterministic time-budget reconciler runs. `auto` is safe as "
            "a default here: this stage makes NO LLM call — it reconciles the spine "
            "against the deliverable's target duration arithmetically — and still "
            "skips clean when neither `direction.timing.target_seconds` nor the "
            "deliverable's `target_duration_seconds` gives it a budget. Env "
            "PIPELINE_TIMING_DEFAULT_ENABLED."
        ),
    )
    timing_relief_default_enabled: bool = Field(
        default=False,
        description=(
            "In ADDITION to the default `direction.timing` seed, also seed a "
            "`direction.timing.relief: {mode: auto, iterations: 2, score_threshold: 4.0}` "
            "block, ARMING the opt-in WM relief loop that shortens over-long narration on "
            "an INFEASIBLE timeline. Unlike the deterministic reconciler this makes real "
            "LLM (beat-rewriter) calls and only fires on already-infeasible runs, so it is "
            "default OFF and rolled out via canary. Note: this seeds `relief` under a "
            "`setdefault('direction','timing',...)`, so a run that authors its OWN "
            "`direction.timing` block WITHOUT a `relief` key is left untouched (conservative "
            "— an explicit authored timing wins). Env PIPELINE_TIMING_RELIEF_DEFAULT_ENABLED."
        ),
    )
    structural_relief_autoconverge_enabled: bool = Field(
        default=False,
        description=(
            "KILL-SWITCH (default OFF) for PR-C bounded auto-convergence of footage-pinned "
            "over-budget demo_video timelines. When a run completes with reconcile_timeline's "
            "convergence_outcome=='cant_converge' (every over-budget beat protected → the "
            "in-stage relief loop provably cannot converge), a post-completion dispatcher hook "
            "spawns a bounded child regenerate run from demo_video_gen that cuts the over-budget "
            "beats to shorten the spine — the only remaining lever. Each child is a fresh run_id "
            "(no double-pay); a unique index on (parent_run_id, structural_relief_pass) prevents "
            "a replicas:2 double-spawn; bounded to `structural_relief_max_passes` EXTRA paid "
            "renders. OFF until live-proven per deliverable class; flip via gitops. Env "
            "PIPELINE_STRUCTURAL_RELIEF_AUTOCONVERGE_ENABLED."
        ),
    )
    structural_relief_max_passes: int = Field(
        default=2,
        ge=0,
        le=4,
        description=(
            "Max bounded structural-relief regenerate passes (each an EXTRA paid render) before "
            "the run terminates completed+degraded. 0 disables spawning even when the kill-switch "
            "is on. Env PIPELINE_STRUCTURAL_RELIEF_MAX_PASSES."
        ),
    )
    voice_direction_default_enabled: bool = Field(
        default=True,
        description=(
            "Seed a default `voice.direction: {tone: confident}` when neither the "
            "project voice row nor the brief authored one, so the ORCH-1.4 voice "
            "director emits per-line performance directives instead of skipping. Env "
            "PIPELINE_VOICE_DIRECTION_DEFAULT_ENABLED."
        ),
    )
    claim_finish_default_enabled: bool = Field(
        default=True,
        description=(
            "Seed `legal.mode = critique_only` when the run authored no explicit mode, "
            "so the claim-grounding pass runs advisory-only. This alone does NOT make "
            "the stage live: it also requires a non-empty ingested source corpus "
            "(`grounding_sentences`), without which it still skips clean rather than "
            "call every claim unsourced. Env PIPELINE_CLAIM_FINISH_DEFAULT_ENABLED."
        ),
    )
    script_doctor_default_enabled: bool = Field(
        default=True,
        description=(
            "Seed a default top-level `script_doctor: {enabled: true}` when the run "
            "authored none, so the WS-2 script doctor actually runs instead of "
            "skipping. Unlike its siblings this one is NOT advisory: the stage runs a "
            "full LLM critique/rewrite loop over the generated script — up to "
            "`max_passes` critique+rewrite round-trips against worker_management, "
            "inside a 900s stage timeout — on EVERY demo_video run, and it REWRITES "
            "line text that every downstream script consumer then reads. That is the "
            "cost of turning it on; flip this off to stop paying it without a code "
            "deploy. Env PIPELINE_SCRIPT_DOCTOR_DEFAULT_ENABLED."
        ),
    )
    grade_default_on_showcase: bool = Field(
        default=False,
        description=(
            "ORCH-3.3 grade kill-switch. Two effects, both gated on this one flag: "
            "(1) a showcase run with no explicit `finish` in its brief gets a "
            "default enabled `finish` block injected; (2) `build_finish_spec` only "
            "runs the deterministic colorist / emits the whole-film grade knobs "
            "(look/tints/vignette/grain/glow/sharpen/grade_intensity, off the run's "
            "mood/style + design tokens) when this is true. OFF by default: with it "
            "off NO run emits a grade — not even one whose brief authored "
            "`finish.enabled: true` — so the feature is fully killable without a "
            "code deploy while the parallel VC grade-chain rolls out. A brief-"
            "authored grade knob still WINS over the table when the flag is on. "
            "Env PIPELINE_GRADE_DEFAULT_ON_SHOWCASE."
        ),
    )
    captions_default_on_showcase: bool = Field(
        default=False,
        description=(
            "ORCH-4.2 word-timed burned-in captions kill-switch. When False "
            "(default) the live caption stage is inert and renders are "
            "byte-identical. Flip PIPELINE_CAPTIONS_DEFAULT_ON_SHOWCASE=true in "
            "the GitOps configmap to activate word-timed captions on the "
            "showcase path (also requires deliverable captions_mode=burned_in)."
        ),
    )
    captions_karaoke: bool = Field(
        default=False,
        description=(
            "Word-timed karaoke ({\\k}) highlight for burned-in captions. When True "
            "the orchestrator forwards captions_karaoke to the compiler so each "
            "cue's words highlight in time (burned_in only; the soft overlay track "
            "cannot carry {\\k} and ignores it). False (default) => plain phrase "
            "captions, byte-identical. Env PIPELINE_CAPTIONS_KARAOKE. Requires "
            "captions_default_on_showcase to be on for captions to build at all."
        ),
    )
    sound_default_on_showcase: bool = Field(
        default=True,
        description=(
            "ORCH-4.1 T1: every showcase run ships with sound design ON by "
            "default (`sound_effects={'enabled': True, 'ambient': True}`) so a "
            "showcase brief that never authors a `sound_effects` block no "
            "longer renders silent. Only fills the block when the project type "
            "is 'showcase' AND the brief/override didn't already set "
            "`sound_effects` — an authored block always wins. Kill-switch for "
            "a bad default without a code deploy. Env "
            "PIPELINE_SOUND_DEFAULT_ON_SHOWCASE."
        ),
    )
    planning_from_intent_enabled: bool = Field(
        default=True,
        description=(
            "A0 (great-video PR 2.2): the authoring front door. When a brief "
            "carries an `intent` block (from POST /api/projects/from-intent or "
            "an authored brief), build_config_snapshot resolves it into a "
            "PlanDocument and merges the computed knobs (gains, cast, "
            "resolution, footage-bound lines) into brief presence-opt-in — an "
            "authored value always wins per knob, and a brief with no `intent` "
            "is byte-identical. Kill-switch for a bad default without a code "
            "deploy. Env PIPELINE_PLANNING_FROM_INTENT_ENABLED."
        ),
    )
    provenance_ledger_universal_enabled: bool = Field(
        default=True,
        description=(
            "Great-video A20 follow-up: record an AUDIT-ONLY provenance ledger "
            "on every demo_video/showcase run that has NO `intent` block, so the "
            "render_manifest override-ledger is populated on ordinary runs (not "
            "just intent/storyboard-authored ones). build_config_snapshot "
            "resolves the brief with an empty intent and writes ONLY "
            "`plan_document` — never the brief — so the delivered video is "
            "byte-identical; render_manifest is the sole reader. Kill-switch to "
            "disable the ledger without touching intent planning or a code "
            "deploy. Env PIPELINE_PROVENANCE_LEDGER_UNIVERSAL_ENABLED."
        ),
    )
    # ⛔ `webhook_allow_private` DELETED 2026-08-20. It was a SECURITY-SHAPED knob — "Allow
    # private/internal IPs for webhook callbacks" — in a service with no webhook feature at
    # all (`git grep -rn webhook -- src/` returned only this field and one unrelated prose
    # string). A reader, or a future SSRF audit, would conclude callback egress is gated.
    # Nothing gated anything. Outbound URL safety lives in `w2v_common.net` — see the
    # "Outbound URL fetching / SSRF" section of this repo's CLAUDE.md.
    shutdown_grace_seconds: int = Field(
        default=100,
        description=(
            "Wall-clock seconds the asgi lifespan waits for in-flight run "
            "dispatch tasks to drain before force-cancelling them. With the "
            "RunDispatcher._draining flag, the CancelledError handler leaves "
            "drained runs in ``running`` instead of marking them ``cancelled`` "
            "— the next pod's startup reconciler resumes them. Must stay "
            "strictly shorter than k8s ``terminationGracePeriodSeconds`` so "
            "the Redis/DB close path still has headroom before SIGKILL. The "
            "production ConfigMap pins this to 100s paired with "
            "terminationGrace=120s; default here matches so local dev "
            "behaves like prod."
        ),
    )
    reconciler_max_age_seconds: int = Field(
        default=7_200,
        description=(
            "Startup reconciler resumes runs whose status is `running`/`pending` "
            "and whose ``updated_at`` is within this many seconds. Runs older "
            "than this cutoff are considered abandoned (user has moved on, or "
            "the run was stuck before durable resume existed) and are marked "
            "``failed`` with a descriptive error rather than re-dispatched. "
            "Default 2h covers the slowest blueprint (website_review ≈ 75 min) "
            "with substantial headroom for a pod outage longer than the run's "
            "remaining work."
        ),
    )
    reconciler_interval_seconds: int = Field(
        default=90,
        description=(
            "After the startup reconcile, re-run the orphan reconciler every "
            "this many seconds for the pod's lifetime. Must stay strictly "
            "greater than the RunDispatcher per-run lease TTL (60s) so a "
            "hard-killed pod's lease has TTL-expired before the next tick "
            "attempts to re-acquire and resume the run; small enough that a "
            "wedged run self-heals within ~2 intervals (≲3 min) rather than "
            "hanging until an unrelated pod restart. The reaper skips any run "
            "whose lease is still held (a live pod owns it), so each tick is a "
            "cheap Redis GET per running run."
        ),
    )
    tts_preview_cache_ttl_seconds: int = Field(
        default=604_800,
        description=(
            "TTL for cached voice-preview audio keyed by (provider, voice, "
            "speed). The preview sample sentence is fixed and tenant-agnostic, "
            "so the cached WAV is shared across all tenants and a long TTL is "
            "safe. 7 days keeps the catalogue of previously-auditioned voices "
            "warm without pinning Redis forever; bump the key's ``v1`` prefix "
            "in routes/tts.py if the sample text ever changes."
        ),
    )
    tts_preview_retry_attempts: int = Field(
        default=2,
        description=(
            "Transport-error and 502/504 retry budget for the interactive "
            "voice-preview path (a human is waiting on the click). Deliberately "
            "far below the narration default so an unreachable or broken TTS box "
            "fails fast with a 502 → toast instead of blocking ~10s through the "
            "full exponential-backoff chain. 2 = first try + one retry. It does "
            "NOT bound a BUSY backend (429/503 + Retry-After, the GPU admission "
            "contract): that waits within ``tts_preview_busy_budget_seconds``."
        ),
    )
    tts_preview_busy_budget_seconds: float = Field(
        default=90.0,
        gt=0,
        le=300,
        description=(
            "Total wall time one voice preview may spend waiting out a BUSY TTS "
            "lane (429/503, honouring each answer's Retry-After). 90s, NOT the "
            "backends' 120s lane queue: the browser reaches us through "
            "Cloudflare, whose edge cuts w2v.nonagentic.ai at ~100s and answers "
            "524 — worse than our own 503. The in-flight synth's timeout is "
            "capped to what is left of budget + 5s, so the whole call ends by "
            "~95s. When the budget is exhausted the route answers 503 + "
            "Retry-After (the backend's last hint, else 30s) so the UI can say "
            '"busy, try again" — never a 502.'
        ),
    )
    idempotency_ttl_seconds: int = Field(
        default=86_400,
        description=(
            "TTL for cached POST responses keyed by ``Idempotency-Key``. "
            "24h matches Stripe's de-facto behaviour and is long enough to "
            "absorb the worst-case client retry chain (DLQ replay, manual "
            "re-drive) without pinning Redis indefinitely."
        ),
    )
    request_audit_retention_days: int = Field(
        default=90,
        description=(
            "Retention window for the high-volume ``request_audit`` table "
            "(migration 0026): rows older than this many days are pruned by a "
            "pod-lifetime background loop. Every 401/403 persists a row, so an "
            "unauthenticated scanner is a disk-fill vector — a finite window "
            "bounds the table. 90 days keeps a useful incident-response trail. "
            "Set to 0 (or negative) to DISABLE pruning entirely (e.g. when an "
            "external log shipper owns retention)."
        ),
    )
    request_audit_prune_interval_seconds: int = Field(
        default=86_400,
        description=(
            "How often the ``request_audit`` retention loop sweeps for expired "
            "rows. Daily (86_400s) is ample — the prune is a batched, "
            "index-supported DELETE on ``created_at`` and the table only needs "
            "to stay within ``request_audit_retention_days``. Ignored when "
            "retention is disabled."
        ),
    )
    preview_proxy_enabled: bool = Field(
        default=False,
        description=(
            "Run the pod-lifetime preview-proxy sweeper that turns each "
            "``video_sources`` original into a 720p h264 proxy (``preview_key`` + "
            "``preview_status='ready'``), which is what makes the "
            "``footage_probe`` review gate actually PLAY the footage instead of "
            "listing metadata. DEFAULT OFF on purpose: the transcode runs ffmpeg "
            "INSIDE the API pod, and a 4K HEVC seek-decode measures ~444 MiB RSS "
            "per concurrent job (see ``shot_analysis._CONCURRENCY``'s derivation) "
            "against a pod sized for JSON stage artifacts. Raise the deployment's "
            "memory/cpu limits, then set PIPELINE_PREVIEW_PROXY_ENABLED=true in "
            "the gitops ConfigMap."
        ),
    )
    preview_proxy_interval_seconds: int = Field(
        default=60,
        ge=1,
        description=(
            "How often the preview-proxy sweeper looks for ``preview_status="
            "'pending'`` video sources. An upload is a human handing us footage "
            "and then waiting to review it, so a minute is the slowest tolerable "
            "time-to-first-frame; the scan itself is one indexed SELECT through "
            "the ``pending_video_sources`` resolver and costs nothing when idle. "
            "Ignored when ``preview_proxy_enabled`` is false."
        ),
    )
    preview_proxy_batch_size: int = Field(
        default=4,
        ge=1,
        description=(
            "Maximum pending sources one sweeper tick claims. Bounds the work a "
            "single tick queues behind the module-level semaphore — without it a "
            "bulk upload of 137 clips would build a 137-deep queue of pending "
            "tasks in one tick, and a pod drain mid-tick would abandon all of "
            "them at once. Throughput is governed by the semaphore, not by this; "
            "this only keeps the queue shallow. Unclaimed rows are simply picked "
            "up by the next tick — nothing is lost by a small batch."
        ),
    )
    preview_proxy_max_attempts: int = Field(
        default=3,
        ge=1,
        description=(
            "How many times the sweeper will try to build one source's proxy "
            "before writing ``preview_status='failed'`` + ``error``. The counter "
            "lives in the existing ``probe`` JSONB under ``preview_worker."
            "attempts`` — no new column, no lease-expiry arithmetic. Without a "
            "cap a permanently-undecodable clip is re-claimed on every tick "
            "forever and starves the batch; with it the reviewer sees a visible "
            "'failed' note (which ``footage_probe`` already surfaces) instead of "
            "a spinner that never resolves."
        ),
    )
    preview_proxy_timeout_seconds: int = Field(
        default=900,
        ge=1,
        description=(
            "Hard wall-clock bound on ONE ffmpeg proxy transcode. Load-bearing "
            "beyond the usual reason: the row's claim is a "
            "``pg_try_advisory_xact_lock`` held for the whole transcode, so a "
            "hung ffmpeg would pin an open transaction (and a pgbouncer server "
            "connection) indefinitely. 15 minutes comfortably covers the 512 MiB "
            "upload ceiling at ``-preset veryfast -threads 2``; past it the "
            "process is killed, the attempt counted, and the lock released."
        ),
    )
    preview_proxy_max_source_bytes: int = Field(
        default=1_500_000_000,
        ge=1,
        description=(
            "Largest source object the preview-proxy sweeper will DOWNLOAD. A "
            "disk-safety bound, not a product limit. The worker stages the whole "
            "original plus its output side by side in a ``mkdtemp`` under the pod's "
            "``/tmp`` emptyDir (4Gi) at a concurrency of 2, so two large masters "
            "claimed in one tick can ENOSPC the volume — which burns both attempts "
            "AND breaks unrelated ``/tmp`` writers in the same pod. Worse, a source "
            "bigger than the volume can never succeed at all: it burns every attempt "
            "and lands permanently ``failed``. Checked with a metadata HEAD before a "
            "single byte moves, so an oversized clip is retired immediately with an "
            "operator-readable reason instead of failing expensively N times. Sized "
            "so that 2 concurrent jobs (source + ~10%% proxy each) fit the volume "
            "with headroom; raise it only together with the ``/tmp`` emptyDir "
            "``sizeLimit`` in gitops, never alone."
        ),
    )
    preview_proxy_transfer_timeout_seconds: int = Field(
        default=600,
        ge=1,
        description=(
            "Wall-clock bound on EACH of the two object transfers bracketing the "
            "proxy transcode (the source download and the proxy upload). Only "
            "ffmpeg was bounded before: ``get_file``/``put_file`` go to "
            "``fget_object``/``fput_object`` with no timeout of their own, so a "
            "stalled MinIO transfer hung the worker indefinitely — well past the "
            "row's lease, at which point another worker re-claims the row and the "
            "first one's eventual verdict is a stale write racing the second's. "
            "Bounding the transfers is what keeps a worker's runtime inside the "
            "lease it holds, and therefore what makes the ownership fence on the "
            "terminal writes reachable rather than theoretical."
        ),
    )
    storyboard_snapshot_max_auto_per_project: int = Field(
        default=10,
        ge=0,
        description=(
            "Retention cap on machine-minted ``auto`` storyboard snapshots per "
            "project (migration 0050). Every restore first mints a pre-restore "
            "safety copy of the WHOLE storyboard graph as JSONB, so a project "
            "that is rolled back repeatedly grows an unbounded pile of blobs. "
            "Right after each ``auto`` snapshot is written, every ``auto`` row "
            "beyond the newest N for that project is deleted. ONLY ``kind='auto'`` "
            "rows are pruned — user-created ``manual`` snapshots are kept forever. "
            "Set to 0 to DISABLE pruning entirely (0 means 'never prune', NOT "
            "'delete them all')."
        ),
    )
    # ─── e2e project retention sweep ────────────────────────────────────────
    # The end-to-end suites create a project per spec and NEVER delete it: 46
    # live create-sites (13 Playwright specs in w2v_frontend/e2e/tests + 33
    # happy-path scenarios in scripts/e2e_validate.py), zero with project
    # teardown. `_teardown_run` only CANCELS the run. The result accumulates in
    # the owner's tenant UI at a measured ~11 projects / 30 minutes.
    #
    # ⛔ BOTH ids must be set or the sweep does not run. A sweep that activates
    # on a partial config is the dangerous version: with only an age predicate
    # it would archive the owner's REAL projects, which live in the same tenant
    # under a different subject. There is deliberately no "prune everything
    # older than N" mode and no wildcard — the subject is the whole safety
    # argument, not a filter layered on top of one.
    e2e_project_prune_subject_id: str | None = Field(
        default=None,
        description=(
            "``projects.owner_subject_id`` whose terminal projects the retention "
            "sweep archives. This is the e2e suite's subject. Unset (the "
            "default) DISABLES the sweep; so does leaving "
            "``e2e_project_prune_tenant_id`` unset. Must be a UUID."
        ),
    )
    e2e_project_prune_tenant_id: str | None = Field(
        default=None,
        description=(
            "``projects.tenant_id`` the retention sweep is confined to. Required "
            "alongside ``e2e_project_prune_subject_id`` — the pair is the sweep's "
            "entire scope. Unset DISABLES the sweep. Must be a UUID."
        ),
    )
    e2e_project_prune_min_age_hours: int = Field(
        default=6,
        gt=0,
        description=(
            "How long a TERMINAL e2e project must sit before the sweep archives "
            "it. 6h is past the longest nightly shard (72 min) plus a working "
            "morning, so a failure is still inspectable in the UI when someone "
            "comes to look at it. Only ``completed``/``failed`` projects are "
            "ever eligible — a ``drafting``/``producing`` project belongs to a "
            "live run and is never touched, at any age."
        ),
    )
    e2e_project_prune_interval_seconds: int = Field(
        default=3600,
        gt=0,
        description=(
            "How often the e2e retention sweep runs. Hourly against a ~22/hour "
            "arrival rate: this drains a trickle, it is not a backlog job."
        ),
    )
    e2e_project_prune_max_per_sweep: int = Field(
        default=50,
        gt=0,
        description=(
            "Hard cap on projects archived in ONE sweep. Sized at ~2x the "
            "measured arrival rate so the steady state never hits it — which is "
            "the point: if this cap is ever reached it means the scope is wrong, "
            "and the cap turns that from a mass-archive into a logged anomaly "
            "someone can catch on the next tick."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="PIPELINE_", env_file=".env", case_sensitive=False, extra="ignore"
    )

    def client_timeout_for(self, effective_timeout_seconds: float) -> float:
        """This instance's client rung for a site whose per-attempt budget is the arg.

        The name deliberately shadows the module-level :func:`client_timeout_for` (which
        this delegates to) so there is exactly ONE spelling of the derivation in the
        codebase. Call sites that subdivide the budget must go through this, not through
        :attr:`llm_client_timeout_seconds` — that property is only correct for a site
        whose effective per-attempt timeout IS ``llm_budget_seconds``.
        """
        return client_timeout_for(
            effective_timeout_seconds,
            delta_seconds=self.llm_client_timeout_delta_s,
        )

    @property
    def llm_client_timeout_seconds(self) -> float:
        """The WMClient read timeout for a PLAIN site — one whose per-attempt timeout is
        the global ``llm_budget_seconds``.

        Derived, never configured directly, so the two can't be edited out of order.
        """
        return self.client_timeout_for(self.llm_budget_seconds)

    @model_validator(mode="after")
    def _validate_llm_timeout_ladder(self) -> PipelineSettings:
        """Enforce the LLM-stage timeout ladder invariant at boot.

        Two bounds, and the second one is the reason this validator was wrong.

        LOWER: ``llm_budget_seconds >= llm_downstream_floor_s``. Below the floor a
        still-working downstream call is cut off early.

        UPPER: ``llm_budget_seconds * (llm_retries + 1) < llm_stage_base_ceiling_s``.
        The ceiling is the outer per-stage TimeoutMiddleware budget, and a stage is
        ``llm_retries + 1`` sequential ``stage_call`` attempts — so the ceiling has to
        cover the WHOLE stage, not one attempt. Checking only the single-attempt bound
        (what this did until 2026-08-19) let a config through in which the retry could
        never finish: the shipped 660 * 2 = 1320 against a live ceiling of 900 meant the
        outer wrapper always killed attempt 2 mid-flight, replacing the clean
        StageTimeoutError this ladder exists to produce with a blunt abort. That was
        latent in production, not hypothetical.

        Raising the budget for a slow vLLM without raising the ceiling is the footgun
        this guards; all three bounds are env-configurable, so the guard moves with them.
        """
        # ── CLIENT RUNG, UPPER BOUND: strictly below the per-attempt budget. ────────
        # `gt=0.0` on the delta already guarantees this, so reaching the raise means
        # someone widened the field constraint. Belt-and-braces on purpose: an equality
        # here is not a config smell, it is the exact tie this rung exists to break.
        if self.llm_client_timeout_seconds >= self.llm_budget_seconds:
            raise ValueError(
                f"llm_client_timeout_seconds ({self.llm_client_timeout_seconds}) must be "
                f"STRICTLY below the per-attempt budget llm_budget_seconds "
                f"({self.llm_budget_seconds}). The stage clock starts first, so on a tie "
                "stage_call's timeout always fires before the HTTP read timeout. Env "
                "PIPELINE_LLM_CLIENT_TIMEOUT_DELTA_S."
            )
        # ── CLIENT RUNG, LOWER BOUND: at or above the downstream read floor. ────────
        # ⚠️ THIS BOUND WAS MISSING, and its absence is the same defect class the upper
        # bound guards. The validator rejected `client >= budget` and said nothing about
        # `client` being far too SMALL: `PIPELINE_LLM_CLIENT_TIMEOUT_FRACTION=0.5` booted
        # clean at 475 s — a configuration that starts up happily and then cuts every
        # still-working call off 125 s under the downstream read budget. A guard that
        # validates one direction only is how a knob ships that can quietly recreate the
        # thing it was added to prevent.
        #
        # The bound belongs on the CLIENT rung, not on the budget, because the client is
        # what actually holds the socket. It subsumes the old `budget >= floor` check
        # (client = budget - delta >= floor implies budget >= floor + delta > floor), so
        # there is one lower bound here rather than two that can disagree.
        if self.llm_client_timeout_seconds < self.llm_downstream_floor_s:
            raise ValueError(
                f"llm_client_timeout_seconds ({self.llm_client_timeout_seconds}) is below "
                f"the downstream read budget llm_downstream_floor_s "
                f"({self.llm_downstream_floor_s}) — the HTTP client would cut off a "
                f"still-working call. llm_budget_seconds ({self.llm_budget_seconds}) must "
                f"be at least llm_downstream_floor_s + llm_client_timeout_delta_s "
                f"({self.llm_downstream_floor_s} + {self.llm_client_timeout_delta_s} = "
                f"{self.llm_downstream_floor_s + self.llm_client_timeout_delta_s}). Env "
                "PIPELINE_LLM_BUDGET_SECONDS / PIPELINE_LLM_CLIENT_TIMEOUT_DELTA_S / "
                "PIPELINE_LLM_DOWNSTREAM_FLOOR_S."
            )
        attempts = self.llm_retries + 1
        stage_worst_case = self.llm_budget_seconds * attempts
        if stage_worst_case >= self.llm_stage_base_ceiling_s:
            raise ValueError(
                f"llm_budget_seconds ({self.llm_budget_seconds}) x {attempts} attempts "
                f"(llm_retries={self.llm_retries}) = {stage_worst_case}s, which is not "
                f"strictly below the outer per-stage ceiling llm_stage_base_ceiling_s "
                f"({self.llm_stage_base_ceiling_s}). The ceiling covers the whole stage, "
                "not one attempt — at or above it the outer TimeoutMiddleware fires "
                "before stage_call's own timeout and the retry can never complete. Raise "
                "PIPELINE_LLM_STAGE_BASE_CEILING_S in lockstep, or lower "
                "PIPELINE_LLM_RETRIES."
            )
        return self

    @property
    def effective_demo_qa_fail_sharpness(self) -> float:
        """The blank hard-fail threshold, clamped to never exceed the softer
        `demo_qa_min_sharpness` degrade band.

        Guarantees the invariant that a blank hard-fail can NEVER fire on a clip
        the degrade band would pass (`sharpness >= demo_qa_min_sharpness`): even a
        fat-fingered `demo_qa_fail_sharpness` set above the degrade threshold is
        pulled back to it, so a real (merely-soft) UI is never hard-failed. `0.0`
        (disable) is preserved unchanged since `min(0.0, x) == 0.0`.
        """
        return min(self.demo_qa_fail_sharpness, self.demo_qa_min_sharpness)


class DownstreamServiceSettings(_BaseDownstreamSettings):
    """Downstream-service settings for the orchestrator.

    Peer service URLs are now resolved at call sites via
    :func:`w2v_common.discovery.service_url`. Only the cross-cutting
    ``license_token`` is kept here (read from ``DOWNSTREAM_LICENSE_TOKEN``).
    """

    # SecretStr, not str: pydantic renders a plain str verbatim in the model
    # repr, so ANY assertion that happens to show `Settings` republishes this
    # credential into the CI log. That is not hypothetical — a failing
    # tests/test_config.py assertion printed the live token on 2026-08-04.
    # Read it with `.get_secret_value()`, which is greppable in review.
    #
    # REQUIRED unless CELL_AUTH_MODE=enforce, no code default. Under off|dual an
    # unset or blank DOWNSTREAM_LICENSE_TOKEN makes constructing Settings raise,
    # so the service fails at boot instead of calling peers unauthenticated (or
    # as somebody else). Under enforce (P7: in-cell hops carry per-service cell
    # JWTs; an R3 deployment boots with env + slots only) it is OPTIONAL and
    # ``None`` when unset/blank: in-cell clients never send it, and every
    # out-of-cell call that needs it raises ``DownstreamCredentialMissingError``
    # at call time (src/clients/downstream_credential.py) — never a silent
    # unauthenticated send, never a fallback identity.
    license_token: SecretStr | None = Field(
        default=None,
        validate_default=True,
        description=(
            "License token for authenticating with downstream services "
            "(DOWNSTREAM_LICENSE_TOKEN; required unless CELL_AUTH_MODE=enforce)"
        ),
    )

    @field_validator("license_token")
    @classmethod
    def _license_token_not_blank(cls, value: SecretStr | None) -> SecretStr | None:
        blank = value is None or not value.get_secret_value().strip()
        if cell_auth_mode() is CellAuthMode.ENFORCE:
            return None if blank else value
        if value is None:
            raise ValueError(
                "DOWNSTREAM_LICENSE_TOKEN is unset; it is required unless CELL_AUTH_MODE=enforce"
            )
        if blank:
            raise ValueError("DOWNSTREAM_LICENSE_TOKEN is set but blank; it is required")
        return value

    # tts_gateway is a new peer not yet in w2v_common.discovery; resolved via this
    # setting until it lands there (then this becomes an override). In-cluster DNS.
    tts_gateway_url: str = Field(
        default="http://tts-gateway.w2v:6320",
        description="Base URL of the pooling TTS gateway (claim-queue).",
    )
    # Feature flag: route audiobook narrate through the TTS gateway (submit/poll)
    # instead of the direct synchronous synthesize_and_store path. Off by default
    # until the worker fleet + gateway have soaked.
    audiobook_narrate_via_gateway: bool = Field(
        default=False,
        description="Route audiobook_narrate through the TTS gateway claim-queue.",
    )


class ProvisioningSettings(BaseSettings):
    """Client-credentials config for provisioning, licenses and identity invites.

    The admin-tier provisioning flow calls provisioning's identity invite API
    (find-or-create a Keycloak user + its action email) and the licenses service
    (mint a license, regenerate its token) as the ``w2v-backend`` service account. Both authorize
    a Keycloak **client-credentials** Bearer token — NOT the service-account
    license token used for the license-authz peers. Field names match the env
    vars already wired on the orchestrator deployment (``KEYCLOAK_URL`` etc.,
    case-insensitive); the client secret lives in the SOPS w2v-orchestrator
    Secret. Peer URLs (licenses) still resolve via ``service_url`` at call time.

    The early-access approve uses the Keycloak Admin REST API
    (``{keycloak_url}/admin/realms/{user_realm}``; realm-management ``manage-users``)
    to mint the requester's own user and send its one action email for
    ``identity_invite_client_id`` (``src/services/signup/identity.py``). The
    administered realm is ``KEYCLOAK_USER_REALM`` (default ``KEYCLOAK_REALM``); it
    authenticates as ``KEYCLOAK_USER_REALM_ADMIN_CLIENT_ID`` in that realm when set,
    else as this same service account.
    """

    keycloak_url: str = Field(
        default="", description="Keycloak base URL (e.g. http://keycloak.keycloak.svc:8080)"
    )
    keycloak_realm: str = Field(
        default="portal",
        description=(
            "Realm the service token (KEYCLOAK_CLIENT_ID) is minted from — licenses, identity "
            "invites (unless KEYCLOAK_SERVICE_REALM is set), and (unless "
            "KEYCLOAK_USER_REALM_ADMIN_CLIENT_ID is set) the admin REST"
        ),
    )
    # Realm-per-app P7 chain C ("realm portal = staff only"): the realm the w2v-backend S2S
    # token (licenses, identity invites) is minted from. KEYCLOAK_REALM cannot simply be
    # moved to secure-apps: w2v_common's inbound JWT validator also reads it as its primary
    # issuer realm. Same client id + secret in both realms (gitops keycloak
    # PATCH_ONLY=p7-s2s-secure-apps). Empty ⇒ KEYCLOAK_REALM, i.e. byte-identical to before.
    keycloak_service_realm: str = Field(
        default="",
        description="Realm the S2S token is minted from (KEYCLOAK_SERVICE_REALM); empty → KEYCLOAK_REALM",
    )
    # Realm-per-app (commonpractice #480 pattern): the realm Studio's USERS live in is
    # separate from the realm the w2v-backend service token is minted from. licenses
    # only knows the portal w2v-backend client, so KEYCLOAK_REALM stays portal while the
    # early-access approve / removal / backfill create, look up, mail and delete users in
    # KEYCLOAK_USER_REALM. Empty ⇒ KEYCLOAK_REALM, i.e. byte-identical to before.
    keycloak_user_realm: str = Field(
        default="",
        description="Realm Studio users are created/looked up in; empty → KEYCLOAK_REALM",
    )
    # Optional: the user realm's OWN admin client (e.g. realm w2v's w2v-admin). Set both or
    # neither. Unset ⇒ the admin REST authenticates as today (KEYCLOAK_REALM +
    # KEYCLOAK_CLIENT_ID/SECRET) and only the administered realm follows KEYCLOAK_USER_REALM.
    keycloak_user_realm_admin_client_id: str = Field(
        default="", description="Admin client in KEYCLOAK_USER_REALM (both-or-neither)"
    )
    keycloak_user_realm_admin_client_secret: SecretStr = Field(
        default=SecretStr(""), description="Secret of KEYCLOAK_USER_REALM_ADMIN_CLIENT_ID"
    )
    keycloak_client_id: str = Field(
        default="w2v-backend", description="Client-credentials client id"
    )
    # SecretStr for the same reason as DownstreamServiceSettings.license_token:
    # a plain str is rendered verbatim by pydantic's repr and leaks into CI logs.
    keycloak_client_secret: SecretStr = Field(
        default=SecretStr(""), description="Client-credentials client secret"
    )
    license_app_keys: list[str] = Field(
        default_factory=lambda: ["w2v"],
        description="App key(s) to mint a license for on provision (licenses GET /apps key).",
    )
    # P2 tenancy invites (WP4a): the invite link is `<w2v_frontend_url>/invite/accept?token=…`
    # (relative when unset).
    w2v_frontend_url: str = Field(
        default="", description="Public w2v SPA base URL (invite links); empty → relative link"
    )
    # Identity invites: w2v never creates Keycloak users itself. Provisioning owns
    # user creation behind `POST {provisioning_url}/api/identity/invites` (find or
    # create by email + the Keycloak action email), authorized by this service's
    # client-credentials token (realm role `identity-inviter`). `client_id` +
    # `redirect_uri` must match this caller's allow-list entry in provisioning
    # (prefix match on the redirect).
    provisioning_url: str = Field(
        default="", description="Provisioning base URL (identity invite API); empty → refused"
    )
    identity_invite_client_id: str = Field(
        default="w2v-frontend",
        description="Public login client the invite action email is issued for",
    )
    identity_invite_redirect_uri: str = Field(
        default="",
        description=(
            "Where the invite email lands once its actions are done when there is no "
            "per-invite link (first admin); empty → `<w2v_frontend_url>/`"
        ),
    )

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    @model_validator(mode="after")
    def _user_realm_admin_both_or_neither(self) -> ProvisioningSettings:
        has_id = bool(self.keycloak_user_realm_admin_client_id.strip())
        has_secret = bool(self.keycloak_user_realm_admin_client_secret.get_secret_value().strip())
        if has_id != has_secret:
            raise ValueError(
                "KEYCLOAK_USER_REALM_ADMIN_CLIENT_ID and KEYCLOAK_USER_REALM_ADMIN_CLIENT_SECRET: "
                "set both or neither"
            )
        return self

    @property
    def service_realm(self) -> str:
        """The realm the S2S token is minted from (``KEYCLOAK_SERVICE_REALM``, default
        ``KEYCLOAK_REALM``)."""
        return self.keycloak_service_realm.strip() or self.keycloak_realm

    @property
    def user_realm(self) -> str:
        """The realm Studio's users live in (``KEYCLOAK_USER_REALM``, default ``KEYCLOAK_REALM``)."""
        return self.keycloak_user_realm.strip() or self.keycloak_realm

    @property
    def user_realm_has_admin_client(self) -> bool:
        """True when the user realm has its own admin client (both fields set)."""
        return bool(self.keycloak_user_realm_admin_client_id.strip())


#: The eight languages the website, the Keycloak email theme and the
#: ``signup_approvals.locale`` CHECK share (early-access v2, contract C6).
SIGNUP_LOCALES: tuple[str, ...] = ("ar", "de", "en", "es", "fr", "he", "nl", "pl")


def _csv(raw: str, fallback: str) -> list[str]:
    items = [s.strip() for s in raw.split(",") if s.strip()]
    return items or [fallback]


class SignupSettings(BaseSettings):
    """Early-access signup (v2, ``~/.claude/plans/w2v-early-access.md``).

    ⛔ OFF BY DEFAULT: every signup route answers 503 ``signup_not_configured``
    until ``W2V_SIGNUP_ENABLED=true``. Read PER CALL through
    :func:`current_signup_settings` (never frozen in the cached ``Settings``), so
    an env-only rollout flips it and tests can drive it.

    The two service doors admit ONLY a Keycloak service-account token whose
    ``azp`` is allow-listed, whose ``aud`` contains that ``azp``, which carries
    the door's realm role and NOT ``super_admin``; the issuer is pinned
    (``W2V_SIGNUP_TOKEN_ISSUER``, else ``KEYCLOAK_ISSUER``; empty ⇒ every token
    refused).

    Realm-per-app P0: each door can pin its OWN issuer —
    ``W2V_SIGNUP_INTAKE_ISSUER`` / ``W2V_SIGNUP_DECIDER_ISSUER`` (full issuer
    URL; default = the shared pin above, so unset changes nothing). Same shape
    as CommonPractice ``CP_SIGNUP_INTAKE_ISSUER`` / ``CP_SIGNUP_DECIDER_ISSUER``.
    P1 flips a door to ``secure-apps`` once its client moves there; the JWKS is
    then fetched from THAT issuer's realm (``service_token._validator``).
    """

    enabled: bool = Field(default=False, description="W2V_SIGNUP_ENABLED")
    intake_azp: str = Field(
        default="w2v-signup-intake", description="CSV allow-list of intake client ids (azp)"
    )
    intake_role: str = Field(default="w2v-signup-intake", description="Realm role of the intake")
    decider_azp: str = Field(
        default="chat-bridge-signup-decider",
        description="CSV allow-list of Telegram-decision client ids (azp)",
    )
    decider_role: str = Field(
        default="w2v-signup-decide", description="Realm role of the Telegram decision door"
    )
    token_issuer: str = Field(
        default="",
        validation_alias=AliasChoices("W2V_SIGNUP_TOKEN_ISSUER", "KEYCLOAK_ISSUER"),
        description="The issuer every signup service token must carry (pinned)",
    )
    intake_issuer: str = Field(
        default="",
        description="W2V_SIGNUP_INTAKE_ISSUER — the intake door's pinned issuer (default token_issuer)",
    )
    decider_issuer: str = Field(
        default="",
        description="W2V_SIGNUP_DECIDER_ISSUER — the decider door's pinned issuer (default token_issuer)",
    )
    # @nonagenticaibot (whose webhook chat-bridge owns — w2v only ever sendMessage's).
    telegram_bot_token: SecretStr = Field(default=SecretStr(""))
    telegram_chat_id: str = Field(default="")
    telegram_thread_id: str = Field(default="")
    telegram_api_url: str = Field(default="https://api.telegram.org")
    #: Pending rows past which the intake refuses with 503 ``signup_queue_full``.
    queue_max_pending: int = Field(default=500, ge=1)
    #: The retry tick (approved rows with unfinished steps).
    tick_interval_seconds: int = Field(default=300)
    #: How long the requester's tenant_admin invite stays acceptable.
    invite_ttl_days: int = Field(default=14, ge=1, le=60)
    #: The new tenant's LLM access posture when the approver chose none (an approve /
    #: invite body without ``llmAccess``). The Telegram one-tap approve does NOT use
    #: it: it always passes ``local_allowed`` explicitly.
    llm_access: Literal["local_allowed", "byo_required", "platform_only"] = "local_allowed"

    model_config = SettingsConfigDict(
        env_prefix="W2V_SIGNUP_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def intake_azps(self) -> list[str]:
        return _csv(self.intake_azp, "w2v-signup-intake")

    @property
    def decider_azps(self) -> list[str]:
        return _csv(self.decider_azp, "chat-bridge-signup-decider")

    @property
    def intake_token_issuer(self) -> str:
        """The intake door's pinned issuer: ``W2V_SIGNUP_INTAKE_ISSUER`` else the shared pin."""
        return (self.intake_issuer.strip() or self.token_issuer.strip()).rstrip("/")

    @property
    def decider_token_issuer(self) -> str:
        """The decider door's pinned issuer: ``W2V_SIGNUP_DECIDER_ISSUER`` else the shared pin."""
        return (self.decider_issuer.strip() or self.token_issuer.strip()).rstrip("/")

    @property
    def telegram_enabled(self) -> bool:
        return bool(
            self.telegram_bot_token.get_secret_value()
            and self.telegram_chat_id
            and re.fullmatch(r"\d+", self.telegram_thread_id or "")
        )


def current_signup_settings() -> SignupSettings:
    """A FRESH read of the signup settings (env at call time, not the cache)."""
    return SignupSettings()


class AuthSecretSettings(BaseSettings):
    """Key material that seals login secrets inside project briefs at rest.

    A brief's ``auth_config`` / ``auth_info`` carries the password the recording
    stage types into a tenant's login form. It is sealed with Fernet before it is
    persisted (``src/services/auth_secrets.py``) and only unsealed in memory at the
    moment a stage dispatches it. Comma-separated keys rotate: the first encrypts,
    every key decrypts. EMPTY ⇒ fail closed — a brief carrying a plaintext secret is
    refused (503) rather than stored readable.
    """

    auth_config_secret_keys: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="AUTH_CONFIG_SECRET_KEYS",
        description="Comma-separated Fernet keys sealing brief login secrets (first encrypts).",
    )

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


#: A provider-secret key id: short, printable, and free of the ``|`` separator the
#: ciphertext envelope uses (``v1|kid|nonce|ct``).
PROVIDER_KEY_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,32}")
#: AES-256 ⇒ exactly 32 key bytes.
PROVIDER_KEY_BYTES = 32


def parse_provider_secret_keys(raw: str) -> list[tuple[str, bytes]]:
    """Parse ``"kid:b64,kid2:b64"`` into ``[(kid, 32-byte key), …]`` (order kept).

    Accepts standard or urlsafe base64. Raises ``ValueError`` on a malformed entry,
    a duplicate kid, or a key that is not 32 bytes — the message names the kid or
    the entry POSITION, never key material."""
    entries: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for position, item in enumerate(raw.split(","), start=1):
        item = item.strip()
        if not item:
            continue
        kid, sep, b64 = item.partition(":")
        kid, b64 = kid.strip(), b64.strip()
        if not sep or not PROVIDER_KEY_ID_PATTERN.fullmatch(kid):
            raise ValueError(f"PROVIDER_SECRET_KEYS entry #{position} is not 'kid:base64'")
        if kid in seen:
            raise ValueError(f"PROVIDER_SECRET_KEYS repeats key id {kid!r}")
        try:
            key = base64.b64decode(b64.replace("-", "+").replace("_", "/"), validate=True)
        except binascii.Error, ValueError:
            raise ValueError(f"PROVIDER_SECRET_KEYS key {kid!r} is not valid base64") from None
        if len(key) != PROVIDER_KEY_BYTES:
            raise ValueError(
                f"PROVIDER_SECRET_KEYS key {kid!r} must decode to {PROVIDER_KEY_BYTES} bytes"
            )
        seen.add(kid)
        entries.append((kid, key))
    return entries


class ProviderSecretSettings(BaseSettings):
    """Keyring sealing tenant provider-connection secrets at rest (P3 provider slots).

    ``PROVIDER_SECRET_KEYS="kid:base64,kid2:base64"`` — AES-256 keys (32 bytes  # pragma: allowlist secret
    each). The FIRST entry encrypts; every entry decrypts, so rotation is "prepend
    a new key, run ``scripts/rewrap_provider_secrets.py``, then drop the old one".
    Empty ⇒ secret writes are refused (503) and nothing can be decrypted; a
    malformed value fails config load (it must never degrade to "no keys"). See
    ``src/services/providers/crypto.py``.
    """

    provider_secret_keys: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="PROVIDER_SECRET_KEYS",
        description="Comma-separated kid:base64 AES-256 keys for provider secrets (first encrypts).",
    )

    @field_validator("provider_secret_keys")
    @classmethod
    def _keys_parse(cls, value: SecretStr) -> SecretStr:
        parse_provider_secret_keys(value.get_secret_value())
        return value

    def keyring_entries(self) -> list[tuple[str, bytes]]:
        """The parsed keyring, primary (encrypting) key first."""
        return parse_provider_secret_keys(self.provider_secret_keys.get_secret_value())

    # ``hide_input_in_errors`` is load-bearing: by default a ValidationError embeds
    # ``input_value`` — the RAW env string, i.e. key material in the pod's crash log.
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore", hide_input_in_errors=True
    )


def _discovered_url(name: str) -> str:
    from w2v_common.discovery import service_url

    return service_url(name)


class PlatformLLMSettings(BaseSettings):
    """Which backend serves the PLATFORM-default ``llm`` slot (owner decision 2026-10-17).

    ``ir`` (the default, owner decision 2026-10-20 — the same default as
    ``w2v_common`` >= 0.109.0) calls inference-router's OpenAI-compatible
    ``POST {LLM_IR_BASE_URL}/v1/chat/completions`` with ONE static licence for app
    ``inference-router`` (``LLM_IR_TOKEN``, sent as ``X-Tenant-Id: Bearer``) and
    delegates the run's tenant via ``X-Effective-Tenant-Id``. IR meters the call.
    ``vllm`` calls the pgx vLLM directly (OpenAI Chat Completions) with ONE static
    licence token, the care-bus-agents shape, and is the rollback. Tenant BYO slots
    are unaffected.

    The worker_management backend (``wm``) is GONE: ``w2v_common`` 0.109.0 deleted
    ``POST /workers/{name}/message`` and refuses ``LLM_BACKEND=wm`` by name, so it
    is refused here too rather than routed to a client that cannot serve it.

    An EXPLICIT ``LLM_BACKEND`` without its credential refuses to boot. A DEFAULTED
    ``ir`` without ``LLM_IR_TOKEN`` boots (jobs/tools that never call the LLM build
    these settings too) and every platform LLM call then raises
    ``IRNotConfiguredError`` — the ``w2v_common`` rule.

    ``LLM_VLLM_TOKEN`` must be a licence for app ``vllm``: both vLLM front doors
    bind it (``vllm.test`` is a license-authz STRICT_APP_HOST, the public
    ``/vllm/`` prefix is route-bound) — the w2v ``DOWNSTREAM_LICENSE_TOKEN`` is an
    app ``w2v`` licence and gets 403.
    """

    llm_backend: Literal["ir", "vllm"] = Field(
        default="ir",
        validation_alias="LLM_BACKEND",
        description="Platform-default LLM backend: ir (inference-router, default) or vllm.",
    )
    llm_ir_base_url: str = Field(
        default_factory=lambda: _discovered_url("inference_router"),
        validation_alias="LLM_IR_BASE_URL",
        description=(
            "inference-router origin (default: w2v_common discovery); the call is "
            "POST {base}/v1/chat/completions."
        ),
    )
    llm_ir_token: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="LLM_IR_TOKEN",
        description="Static `inference-router` licence token sent as X-Tenant-Id: Bearer.",
    )
    llm_ir_model: str = Field(
        default="local-llm",
        validation_alias="LLM_IR_MODEL",
        description="Model name sent to inference-router.",
    )
    llm_vllm_base_url: str = Field(
        default="http://localhost:8001/v1",
        validation_alias="LLM_VLLM_BASE_URL",
        description="OpenAI-compatible base URL of the pgx vLLM (licence-gated front door).",
    )
    llm_vllm_token: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="LLM_VLLM_TOKEN",
        description="Static `vllm` licence token sent as Authorization: Bearer.",
    )
    llm_vllm_model_default: str = Field(
        default="local-llm",
        validation_alias="LLM_VLLM_MODEL_DEFAULT",
        description="Served model for generation calls (and every call by default).",
    )
    llm_vllm_model_fast: str = Field(
        default="",
        validation_alias="LLM_VLLM_MODEL_FAST",
        description=(
            "Served model for fast/choice workers (LLM_VLLM_FAST_WORKERS). Empty ⇒ "
            "those workers use the default model. Must be served by the SAME "
            "OpenAI-compatible endpoint."
        ),
    )
    llm_vllm_fast_workers: Annotated[frozenset[str], NoDecode] = Field(
        default_factory=lambda: frozenset[str](),
        validation_alias="LLM_VLLM_FAST_WORKERS",
        description="Comma-separated worker names routed to LLM_VLLM_MODEL_FAST.",
    )

    @field_validator("llm_vllm_fast_workers", mode="before")
    @classmethod
    def _parse_fast_workers(cls, value: object) -> frozenset[str]:
        if value is None:
            return frozenset()
        if isinstance(value, frozenset | set | list | tuple):
            return frozenset(str(v).strip() for v in value if str(v).strip())
        return frozenset(item.strip() for item in str(value).split(",") if item.strip())

    @model_validator(mode="after")
    def _backend_needs_its_credential(self) -> PlatformLLMSettings:
        if self.llm_backend == "ir":
            explicit = "llm_backend" in self.model_fields_set
            if explicit and not self.llm_ir_token.get_secret_value().strip():
                raise ValueError(
                    "LLM_BACKEND=ir requires LLM_IR_TOKEN (an `inference-router` licence token)"
                )
            if not self.llm_ir_base_url.startswith(("http://", "https://")):
                raise ValueError("LLM_IR_BASE_URL must be an http(s) URL")
            if not self.llm_ir_model.strip():
                raise ValueError("LLM_IR_MODEL must not be empty")
            return self
        if self.llm_backend != "vllm":
            return self
        if not self.llm_vllm_token.get_secret_value().strip():
            raise ValueError("LLM_BACKEND=vllm requires LLM_VLLM_TOKEN (a `vllm` licence token)")
        if not self.llm_vllm_base_url.startswith(("http://", "https://")):
            raise ValueError("LLM_VLLM_BASE_URL must be an http(s) URL")
        if not self.llm_vllm_model_default.strip():
            raise ValueError("LLM_VLLM_MODEL_DEFAULT must not be empty")
        return self

    def model_for(self, worker_name: str) -> str:
        """The served model for ``worker_name``: fast when listed and configured."""
        if self.llm_vllm_model_fast and worker_name in self.llm_vllm_fast_workers:
            return self.llm_vllm_model_fast
        return self.llm_vllm_model_default

    # ``hide_input_in_errors``: a ValidationError must never echo the raw token.
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore", hide_input_in_errors=True
    )


class ProviderSlotResolveSettings(BaseSettings):
    """Who may call ``POST /api/internal/provider-slots/resolve`` (satellite slot lookup).

    ``W2V_PROVIDER_SLOT_CALLERS`` — CSV of cell-token ISSUERS (service names, e.g.
    ``html-render-service``) admitted on top of ``CELL_ALLOWED_CALLERS``. Explicit
    names only: ``*`` is REFUSED at startup, because the answer can carry a tenant's
    decrypted provider secret. Empty (the default) ⇒ nobody may call it.
    """

    provider_slot_callers: Annotated[frozenset[str], NoDecode] = Field(
        default_factory=frozenset,
        validation_alias="W2V_PROVIDER_SLOT_CALLERS",
    )

    @field_validator("provider_slot_callers", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return frozenset(p.strip() for p in value.split(",") if p.strip())
        return value

    @field_validator("provider_slot_callers")
    @classmethod
    def _no_wildcard(cls, value: frozenset[str]) -> frozenset[str]:
        if any("*" in name for name in value):
            raise ValueError(
                "W2V_PROVIDER_SLOT_CALLERS must list explicit issuers; a wildcard is refused "
                "(the endpoint hands out tenant provider secrets)"
            )
        return value

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class ProviderOcrSettings(BaseSettings):
    """OCR provider for scanned (no text layer) PDF pages in ``source_ingest``.

    Owner decision 2026-10-17 (KAG removal): OCR is served by ``mlx_api``
    (``mlx-api-media.test:9131``, mini1) for now. The wire is the
    provider_contracts ``POST /v1/ocr`` contract — the Mistral OCR shape — so a
    later move to a commercial vendor is a config change (URL + key + model).

    ``W2V_PROVIDER_OCR_URL`` empty (the default) ⇒ OCR is OFF: a PDF page with
    no text layer is not OCR'd, and a PDF with no text at all fails its parse
    with an explicit "OCR is not configured" error (never a silent empty
    document). The URL is the provider ROOT; the client appends ``/v1/ocr``.
    """

    provider_ocr_url: str = Field(
        default="",
        validation_alias="W2V_PROVIDER_OCR_URL",
        description="OCR provider root URL (POST {url}/v1/ocr). Empty = OCR off.",
    )
    provider_ocr_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="W2V_PROVIDER_OCR_API_KEY",
        description="Sent as Authorization: Bearer <key> (mlx_api: MLX_API_KEY).",
    )
    provider_ocr_model: str = Field(
        default="PaddleOCR-VL-1.5-bf16",
        validation_alias="W2V_PROVIDER_OCR_MODEL",
        description=(
            "OCR model id. mlx_api accepts its configured id or the short alias "
            "(default = mlx_api's PaddleOCR-VL alias)."
        ),
    )

    @model_validator(mode="after")
    def _url_is_http(self) -> ProviderOcrSettings:
        url = self.provider_ocr_url.strip()
        if url and not url.startswith(("http://", "https://")):
            raise ValueError("W2V_PROVIDER_OCR_URL must be an http(s) URL")
        if url and not self.provider_ocr_model.strip():
            raise ValueError("W2V_PROVIDER_OCR_MODEL must not be empty when OCR is enabled")
        return self

    @property
    def enabled(self) -> bool:
        """True when an OCR provider URL is configured."""
        return bool(self.provider_ocr_url.strip())

    # ``hide_input_in_errors``: a ValidationError must never echo the raw key.
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore", hide_input_in_errors=True
    )


# ---------------------------------------------------------------------------
# Main settings class
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Root settings aggregating all sub-configurations for the orchestrator."""

    app_name: str = Field(default="W2V Orchestrator Service")
    app_version: str = Field(default="0.1.0")
    environment: str = Field(default="development")

    # Phase C C-3: fail-closed tenant-existence enforcement on every tenant-scoped
    # request. `enforce` (default) → a caller whose effective tenant is not known
    # (no `tenants` registry row and no projects) gets 403. `warn` → allow but log
    # would-be-blocks (blast-radius telemetry). `off` → no check. Dialable via the
    # gitops ConfigMap (TENANT_ENFORCEMENT_MODE) WITHOUT an image rebuild — the
    # emergency valve if a legitimate caller is ever blocked.
    tenant_enforcement_mode: Literal["off", "warn", "enforce"] = Field(
        default="enforce",
        description="Tenant-existence enforcement on tenant-scoped routes (off|warn|enforce).",
    )

    # Phase D (RLS) — whether to bind `app.current_tenant` (the GUC the projects
    # RLS policy reads) per request via `SET LOCAL` on tenant-scoped data routes.
    # `off` (default) → no GUC is set; the projects policy stays inert (the app's
    # DB user owns the table + RLS is not FORCEd, so the owner bypasses it). A
    # later increment flips this to `enforce` together with FORCE + a
    # no-`BYPASSRLS` DB role; until then this is dormant plumbing. Dialable via
    # the gitops ConfigMap (TENANT_RLS_MODE) WITHOUT an image rebuild.
    tenant_rls_mode: Literal["off", "enforce"] = Field(
        default="off",
        description="Bind app.current_tenant per request for Postgres RLS (off|enforce).",
    )

    # P2 tenancy cutover dial (src/auth/tenancy_middleware.py). Where a human
    # (Keycloak JWT) caller's tenant comes from:
    #   * `claim` (default) — the JWT `tenant_id` claim, exactly as before P2.
    #   * `shadow` — still served by the claim; the membership answer is computed
    #     alongside, a disagreement logs + increments
    #     w2v_orchestrator_tenancy_shadow_mismatch_total, and a missing membership
    #     for the caller's OWN claim tenant is lazily created.
    #   * `membership` — w2v's tenant_members table is the authority; the claim is
    #     ignored and a caller with no usable membership gets 403 NO_MEMBERSHIP.
    # Cutover: backfill_tenancy.py --apply (HARD gate) → shadow until the mismatch
    # counter stays 0 → membership. In membership AND shadow mode the orchestrator
    # forces AUTH_JWT_TENANT_REQUIRED=false itself (tenancy_middleware.
    # align_jwt_tenant_requirement): a claimless identity is resolved from
    # tenant_members (shadow counts it `claim_missing`), and in membership the
    # claim is never read and the only realm role honoured is super_admin. Dialable via the gitops
    # ConfigMap without a rebuild (pod restart); rollback = claim.
    w2v_tenancy_source: Literal["claim", "shadow", "membership"] = Field(
        default="claim",
        validation_alias=AliasChoices("W2V_TENANCY_SOURCE"),
        description="Tenant source for JWT callers: claim | shadow | membership (default claim).",
    )

    # P2 WP4b partner lane: `POST /api/tenants` is open ONLY to a caller whose
    # introspected w2v licence id is in this operator-managed allowlist (gitops
    # ConfigMap) AND is active. Empty (default) = no partner may create tenants.
    # CSV or JSON array of licence UUIDs; a non-UUID entry fails config load so a
    # typo can never silently allow-list nothing.
    w2v_partner_license_ids: Annotated[frozenset[UUID], NoDecode] = Field(
        default_factory=lambda: frozenset[UUID](),
        validation_alias=AliasChoices("W2V_PARTNER_LICENSE_IDS"),
        description="Licence ids allowed to create tenants via POST /api/tenants (CSV or JSON).",
    )

    # P9: FIRST-PARTY partner licences — our own provisioning / control_service,
    # which also mint the tenant's w2v licence. Such a partner MAY claim a tenant
    # id that already holds a live w2v licence (provisioning mints the licence
    # BEFORE it creates the w2v tenant, and must be able to redrive), and its
    # tenants get ``licence_ref`` auto-set from that licence. A third-party
    # partner may do neither (the licences service records no issuer, so it could
    # be claiming someone else's licence). MUST be a subset of
    # W2V_PARTNER_LICENSE_IDS: extras are dropped with a warning at load.
    w2v_partner_first_party_license_ids: Annotated[frozenset[UUID], NoDecode] = Field(
        default_factory=lambda: frozenset[UUID](),
        validation_alias=AliasChoices("W2V_PARTNER_FIRST_PARTY_LICENSE_IDS"),
        description=(
            "First-party partner licence ids (⊆ W2V_PARTNER_LICENSE_IDS) that may claim "
            "an already-licensed tenant and auto-record its licence_ref (CSV or JSON)."
        ),
    )

    @field_validator(
        "w2v_partner_license_ids", "w2v_partner_first_party_license_ids", mode="before"
    )
    @classmethod
    def _parse_partner_license_ids(cls, value: object) -> frozenset[UUID]:
        if value is None or isinstance(value, frozenset | set | list | tuple):
            return frozenset(UUID(str(v)) for v in value or ())
        raw = str(value).strip()
        if not raw:
            return frozenset()
        items = json.loads(raw) if raw.startswith("[") else raw.split(",")
        return frozenset(UUID(str(item).strip()) for item in items if str(item).strip())

    # P2 WP4b licence refresh: how often the pod-lifetime loop re-checks every
    # tenant with a licence_ref against the licences service (active ↔
    # licence_lapsed). 0 or negative disables the loop.
    licence_refresh_interval_seconds: int = Field(
        default=900,
        validation_alias=AliasChoices("W2V_LICENCE_REFRESH_INTERVAL_SECONDS"),
        description="Tenant licence refresh interval in seconds (default 900; <=0 disables).",
    )

    # The licences bundle ensure (POST /internal/tenants/{id}/bundles/w2v/ensure)
    # gets its own read budget: since licences #275 it also provisions the
    # bundle's dependency licences, measured live at 15.8 s (2026-10-15) against
    # the 15 s budget every other licences call keeps.
    licences_ensure_timeout_seconds: float = Field(
        default=90.0,
        gt=0,
        validation_alias=AliasChoices("W2V_LICENCES_ENSURE_TIMEOUT_SECONDS"),
        description="Timeout in seconds for the licences bundle ensure call (default 90).",
    )

    # WS-B license gate. Default ON (owner ruling: ALL w2v usage is licence-gated
    # and an unsanctioned request is blocked). ON means two things at once:
    #   * GET /api/license/status reports the REAL per-tenant state
    #     (active|expiring_soon|expired|revoked|none|unknown) so the SPA can
    #     render the renewal block / expiring banner; and
    #   * require_license (_common.py) enforces it on every tenant-scoped data
    #     router — 402 for a blocking state, 503 when entitlement is
    #     INDETERMINATE (fail-CLOSED; see _enforce_license_state).
    # Setting W2V_LICENSE_GATE_ENABLED=false via the gitops ConfigMap (no
    # rebuild) is the break-glass switch that restores the old no-op behaviour.
    # Per-request license-token auth remains the real enforcer regardless.
    # ROUTE-LOCAL delegation for POST /api/runs/{run_id}/usage ONLY. w2v_common's
    # AuthMiddleware delegation (AUTH_DELEGATING_SUBJECTS) is GLOBAL — a listed
    # subject could re-scope to any tenant on EVERY route — so it stays off here.
    # A licence-lane caller whose INTROSPECTED subject is on this CSV list may name
    # the run's tenant in X-Effective-Tenant-Id on the usage route and nowhere else.
    # Empty (default) = no delegation anywhere.
    usage_report_delegating_subjects: str = Field(
        default="",
        validation_alias=AliasChoices("USAGE_REPORT_DELEGATING_SUBJECTS"),
        description="CSV of service-account subjects allowed to report usage for another tenant's run.",
    )

    license_gate_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("W2V_LICENSE_GATE_ENABLED", "LICENSE_GATE_ENABLED"),
        description=(
            "Enforce the w2v license on tenant-scoped routers and report the real "
            "per-tenant state on /api/license/status (default ON)."
        ),
    )

    # Self-serve lazy auto-issue. When OFF (default) /api/license/status never
    # mints — identical behaviour to today. When ON (and the gate is ON), a
    # portal tenant polling with state='none' (self-registered, never licensed)
    # gets a w2v license minted on the spot so self-serve onboarding self-heals
    # instead of hard-blocking. Only issues on 'none' — never resurrects expired
    # (a renewal decision) or revoked (deliberate) tenants. Flag flips via the
    # gitops ConfigMap (no rebuild); the lazy hook is fail-soft (a status poll
    # never 500s on a licenses blip).
    license_autoissue_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("W2V_LICENSE_AUTOISSUE_ENABLED", "LICENSE_AUTOISSUE_ENABLED"),
        description="Self-serve: lazily issue a w2v license for an authenticated tenant with state='none' (default OFF).",
    )

    # web_design_v2 always drives html_render in mode="assembly", whose verify
    # lane must run — the product guarantee is "only verified designs complete".
    # When ON (default), a design that comes back with NO verification result
    # (status absent/"skipped" — the lane silently didn't run, e.g. an html_render
    # rollback to a pre-verification image) FAILS the run instead of quietly
    # completing an unverified design. Flip OFF via the gitops ConfigMap (no
    # rebuild) to consciously tolerate a degraded verify lane.
    web_design_require_verification: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "W2V_WEB_DESIGN_REQUIRE_VERIFICATION", "WEB_DESIGN_REQUIRE_VERIFICATION"
        ),
        description="Fail a web_design_v2 run whose design has no verification result (default ON).",
    )

    # TTSB-3 — lower voice_director performance directives (pre-line pause +
    # per-word emphasis) into SSML on the narrate path. OFF by default: the
    # plain-text synth path is byte-identical to today (same TTS payload, same
    # content-addressed clip id → cache stays warm). When flipped ON via the
    # gitops ConfigMap (no rebuild), demo_video_narrate wraps each line that has
    # a matching directive in `<speak>…</speak>`; a line with no directive (or an
    # inert directive) still takes the plain path. SSML lines self-invalidate the
    # clip cache because the SSML string IS the hashed `text`.
    narration_ssml_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("W2V_NARRATION_SSML_ENABLED", "NARRATION_SSML_ENABLED"),
        description="Lower voice_director directives into SSML on the narrate path (default OFF).",
    )

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api: APISettings = Field(default_factory=APISettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    downstream: DownstreamServiceSettings = Field(default_factory=DownstreamServiceSettings)
    provisioning: ProvisioningSettings = Field(default_factory=ProvisioningSettings)
    # Early-access signup. Request paths read current_signup_settings() (per call);
    # this copy is only the snapshot the lifespan uses to start the retry tick.
    signup: SignupSettings = Field(default_factory=SignupSettings)
    auth_secrets: AuthSecretSettings = Field(default_factory=AuthSecretSettings)
    provider_secrets: ProviderSecretSettings = Field(default_factory=ProviderSecretSettings)
    platform_llm: PlatformLLMSettings = Field(default_factory=PlatformLLMSettings)
    provider_ocr: ProviderOcrSettings = Field(default_factory=ProviderOcrSettings)
    provider_slot_resolve: ProviderSlotResolveSettings = Field(
        default_factory=ProviderSlotResolveSettings
    )
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    # Canonical idempotency block read by w2v_common.asgi.create_app. Always on
    # for the orchestrator (the old asgi always registered IdempotencyMiddleware);
    # PIPELINE_IDEMPOTENCY_TTL_SECONDS stays the single ttl source, mirrored here.
    idempotency: IdempotencySettings = Field(default_factory=IdempotencySettings)

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    @model_validator(mode="after")
    def _first_party_partners_are_partners(self) -> Settings:
        """A first-party licence that is not ALSO an allow-listed partner licence
        would grant claim rights to a caller the partner gate refuses anyway — a
        config error. Drop it loudly rather than fail the pod."""
        extras = self.w2v_partner_first_party_license_ids - self.w2v_partner_license_ids
        if extras:
            logger.warning(
                "W2V_PARTNER_FIRST_PARTY_LICENSE_IDS not in W2V_PARTNER_LICENSE_IDS, ignored: %s",
                sorted(str(e) for e in extras),
            )
            self.w2v_partner_first_party_license_ids = (
                self.w2v_partner_first_party_license_ids & self.w2v_partner_license_ids
            )
        return self

    @model_validator(mode="after")
    def _wire_idempotency(self) -> Settings:
        """Enable canonical idempotency and source its TTL from the pipeline block."""
        self.idempotency.enabled = True
        self.idempotency.ttl_seconds = self.pipeline.idempotency_ttl_seconds
        return self

    def get_database_url(self) -> str:
        """Return the configured database connection URL."""
        return self.database.url

    def get_redis_url(self) -> str:
        """Return the configured Redis connection URL."""
        return self.redis.url

    def is_production(self) -> bool:
        """Return True when running in production environment."""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Return True when running in development environment."""
        return self.environment.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()


def clear_settings_cache() -> None:
    """Invalidate the cached Settings so the next call re-reads env vars."""
    get_settings.cache_clear()
