"""ASGI application — Studio v3 (PR-1).

Lifespan opens the postgres engine + Redis client and exposes them on
``app.state.engine`` / ``app.state.session_factory`` / ``app.state.redis``,
which the route dependencies in ``src.api.routes._common`` consume. Routers
mount the 5 PR-1 surfaces; future surfaces (annotations, runs, stages,
elements, verbatim) ship 503 stubs from their owning routers in PR-2..6.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING

from redis.exceptions import RedisError

# Close the SSR-proxy bypass; must run BEFORE w2v_common.auth is imported.
os.environ["AUTH_TRUST_INCLUSTER_IDENTITY"] = "false"

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.responses import Response
from w2v_common.asgi import create_app
from w2v_common.auth import register_api_key_verifier

from src.api.byo_errors import register_byo_exception_handlers
from src.api.error_details import register_object_detail_http_handler
from src.api.routes import (
    admin_router,
    annotations_router,
    artifact_proxy_router,
    audit_router,
    auth_router,
    blueprints_router,
    brand_router,
    critique_router,
    deliverables_router,
    fixes_router,
    health_router,
    internal_provider_slots_router,
    license_router,
    media_router,
    media_understanding_router,
    music_router,
    projects_router,
    projects_sse_router,
    provider_requirements_router,
    redesign_router,
    review_inbox_router,
    runs_router,
    runs_sse_router,
    runs_usage_router,
    service_lab_router,
    storyboard_router,
    tts_router,
    uploads_router,
    uploads_video_router,
    verbatim_router,
    youtube_admin_router,
    youtube_router,
)
from src.api.routes._common import TENANT_GATE
from src.api.routes.signup_approvals import (
    SIGNUP_ROUTERS,
    register_signup_exception_handlers,
)
from src.api.routes.storyboard import register_storyboard_exception_handlers
from src.api.routes.tenancy import ROUTERS as TENANCY_ROUTERS
from src.api.routes.tenancy.errors import register_tenancy_exception_handlers
from src.auth.signup_doors import SignupDoorBodyLimitMiddleware, install_signup_service_doors
from src.auth.tenancy_middleware import install_tenancy_middleware, make_api_key_verifier
from src.cache.redis_client import close_redis, init_redis
from src.clients.downstream_credential import (
    assert_cell_boot_config,
    assert_hop_lanes_boot_config,
)
from src.config import get_settings
from src.database.connection import close_engine, get_engine
from src.database.rls_events import install_rls_tenant_rebind
from src.lifecycle.audit_prune import _start_audit_prune_task
from src.lifecycle.e2e_project_prune import _start_e2e_project_prune_task
from src.lifecycle.metrics import _start_metrics_server, _stop_metrics_server
from src.lifecycle.offboarding_resume import start_offboarding_resume_task
from src.lifecycle.preview_proxy import _start_preview_proxy_task
from src.lifecycle.provider_secret_gc import start_provider_secret_gc_task
from src.lifecycle.reconciler import _reconcile_orphan_runs, _run_periodic_reconciler
from src.lifecycle.signup_tick import start_signup_tick_task
from src.lifecycle.sound_library_seed import start_sound_library_seed_task
from src.middleware.audit import AuditMiddleware
from src.middleware.metrics import MetricsMiddleware
from src.middleware.proxy_body_limit import ProxyBodyLimitMiddleware
from src.middleware.security import FramePolicyMiddleware
from src.middleware.shutdown import ShutdownMiddleware
from src.observability import configure_logging
from src.observability.logging import database_endpoint
from src.services.licence_refresh import start_licence_refresh_task

if TYPE_CHECKING:
    from src.services.run_dispatcher import RunDispatcher

load_dotenv()


async def _drain_dispatcher(dispatcher: RunDispatcher, grace_seconds: int) -> None:
    """Wait up to ``grace_seconds`` for active dispatch tasks to finish.

    Helper extracted from the asgi lifespan to keep the lifespan generator
    under ruff's C901 complexity ceiling. Returns once all tasks have
    drained or the deadline elapses (the caller then issues the
    force-cancel ``dispatcher.shutdown()``).
    """
    if grace_seconds <= 0:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        active = dispatcher.active_run_count()
        if active == 0:
            return
        logging.getLogger(__name__).info(
            "Shutdown drain: %d dispatch task(s) still in flight; waiting", active
        )
        await asyncio.sleep(0.5)
    logging.getLogger(__name__).warning(
        "Shutdown drain window (%ds) elapsed with active dispatch tasks; cancelling",
        grace_seconds,
    )


async def _close_state_clients(app: FastAPI) -> None:
    """Close pod-lifetime downstream clients created in the lifespan.

    Extracted from the lifespan generator to keep it under ruff's C901 ceiling.
    Both clients are BaseServiceClient subclasses whose only teardown method is
    async ``close()`` (there is no ``aclose()``); failures are suppressed so one
    stuck client never blocks the rest of shutdown. Calling the WRONG method here
    would raise AttributeError that the suppress() swallows, silently leaking the
    httpx connection pool on every pod shutdown — so the method name must be right.
    """
    for attr, closer in (("tts_preview_client", "close"), ("youtube_client", "close")):
        client = getattr(app.state, attr, None)
        if client is not None:
            with suppress(Exception):
                await getattr(client, closer)()


async def _drain_panel_image_jobs(app: FastAPI) -> None:
    """Cancel + await the storyboard panel-image tasks, if any were started.

    Same fire-and-forget shape as the dispatcher's run tasks, so the same rule
    applies: a SIGTERM must not orphan a task holding a DB connection. The
    runner is created lazily by the route, so it is often absent entirely.
    Extracted from the lifespan generator to keep it under ruff's C901 ceiling.
    """
    runner = getattr(app.state, "panel_image_jobs", None)
    if runner is not None:
        with suppress(Exception):
            await runner.shutdown()


settings = get_settings()
configure_logging(json_logs=settings.logging.json_logs, level=settings.logging.level)
# ``configure_logging`` installs ``RequestIdLogFilter`` on the stdout HANDLER
# it creates. It must not be re-registered on the root logger here: logger-level
# filters do not run for records propagated up from child loggers, so that
# placement bound ``request_id`` on nothing this service actually logs.
logger = logging.getLogger(__name__)


async def _cancel_lifespan_tasks(app: FastAPI, *names: str) -> None:
    """Cancel and await the named ``app.state`` background tasks, in order.

    One helper rather than a copy of the cancel/suppress/await triple per loop:
    each copy is a branch inside ``lifespan``, which is already at the C901
    ceiling, so the next background loop added would have had to refactor this
    anyway. Order is the CALLER's — the reconciler must stop before the
    dispatcher drains, so pass it first. A task that was never started (its
    feature is disabled) is ``None`` and skipped.
    """
    for name in names:
        task = getattr(app.state, name, None)
        if task is None:
            continue
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open DB engine + Redis client; close on shutdown."""
    # CI's export-openapi step runs `app.openapi()` against this image
    # without DB or Redis available — short-circuit so the import side-effect
    # of opening connections doesn't fail the docker run.
    if os.getenv("ORCHESTRATOR_SKIP_LIFESPAN") == "1":
        yield
        return

    logger.info("Starting w2v_orchestrator (Studio v3)")
    # P7 / R3: under CELL_AUTH_MODE=enforce in-cell hops send no shared token, so a
    # missing cell signing key would make them send NO credential. Fail the pod here,
    # before anything is opened, naming the missing setting.
    assert_cell_boot_config()
    # <HOP>_AUTH_MODE (webscraper / media_discovery / modality_gateway / billing): a
    # typo, or tenant_licence without the BUNDLE_MINTER_* settings, fails the pod
    # here, not on the first call.
    assert_hop_lanes_boot_config()

    engine = await get_engine()
    app.state.engine = engine
    app.state.db_engine = engine  # /readyz reads this name
    app.state.session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    # Re-bind app.current_tenant on every transaction so an intra-request commit
    # (e.g. the usage reconciler inside get_run) can't drop the RLS GUC and make
    # a subsequent owned-row read 404 under enforce. No-op unless enforce + tenant
    # in context. See src/database/rls_events.py.
    install_rls_tenant_rebind()
    # Tenant API keys (P2 WP3): w2v_common routes every `Bearer w2v_sk_...` to
    # this verifier instead of licence introspection. Unregistered → every key
    # is a 401, so it must be in place before the first request.
    register_api_key_verifier(make_api_key_verifier(app.state.session_factory))

    try:
        app.state.redis = await init_redis()
    except (RedisError, OSError) as exc:
        logger.warning("Redis unavailable at startup: %s", exc)
        app.state.redis = None

    # Pod-lifetime client for the voice-preview endpoint so each audition
    # reuses one keep-alive connection to the off-cluster TTS box instead of
    # paying a fresh DNS + TCP + TLS handshake per click. See routes/tts.py.
    from src.clients.tts_client import TTSClient

    app.state.tts_preview_client = TTSClient()

    # Pod-lifetime client for the YouTube publish route — reuses one keep-alive
    # connection to youtube-upload-service across uploads/polls. Closed below.
    from src.clients.youtube_upload_client import YouTubeUploadClient

    app.state.youtube_client = YouTubeUploadClient()

    import src.services.blueprints  # noqa: F401  triggers blueprint registration
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    dispatcher: RunDispatcher | None = None
    if app.state.redis is not None:
        dispatcher = RunDispatcher(
            session_factory=app.state.session_factory,
            registry=get_registry(),
            redis=app.state.redis,  # ty: ignore[invalid-argument-type]  # why: app.state.redis narrowed to Redis by the surrounding `if app.state.redis is not None` guard; ty doesn't carry state-attr narrowing across blocks
        )
        app.state.dispatcher = dispatcher
        app.state.run_engine = RunEngine(
            session_factory=app.state.session_factory,
            registry=get_registry(),
            dispatch=dispatcher.dispatch,
        )
        # Late-bind so course_compile_lessons can spawn child runs through the
        # engine without an import cycle.
        dispatcher.bind_run_engine(app.state.run_engine)

        # Subscribe to the cross-pod run-control channel so a cancel/pause that
        # lands on a non-owning replica still stops the run on whichever pod
        # actually holds its execution task. See RunDispatcher.broadcast_interrupt.
        await dispatcher.start_control_listener()

        # Resume any runs the previous pod left in `running`/`pending`
        # after a drain (RunDispatcher._draining behaviour). Brief
        # advisory lock serialises across the 2 replicas; runs older
        # than reconciler_max_age_seconds are marked `failed` instead
        # of resumed. The re-dispatch attempt count lives in Redis (shared
        # across replicas) so a wedged run is bounded. See _reconcile_orphan_runs.
        await _reconcile_orphan_runs(
            app.state.session_factory,
            dispatcher,
            settings.pipeline.reconciler_max_age_seconds,
            redis=app.state.redis,
        )
        # Then keep reconciling on a timer so a run wedged by a hard-killed
        # pod (whose lease lingered past this one startup reconcile) self-heals
        # instead of hanging in `running` forever. See _run_periodic_reconciler.
        app.state.reconciler_task = asyncio.create_task(
            _run_periodic_reconciler(
                app.state.session_factory,
                dispatcher,
                settings.pipeline.reconciler_max_age_seconds,
                settings.pipeline.reconciler_interval_seconds,
                app.state.redis,
            ),
            name="periodic-reconciler",
        )
    else:
        app.state.dispatcher = None
        app.state.run_engine = None
        app.state.reconciler_task = None

    # Pod-lifetime retention loop for the high-volume request_audit table.
    # Independent of Redis/the dispatcher — it only needs the session factory.
    app.state.audit_prune_task = _start_audit_prune_task(app.state.session_factory)

    # Pod-lifetime retention sweep for the projects the e2e suites abandon (46
    # create-sites, zero with project teardown). Same shape and same dependency
    # as the audit prune above; disabled unless BOTH scope ids are configured,
    # and it logs which way it went at startup. See
    # src/lifecycle/e2e_project_prune.py for why this is a sweep and not 46
    # per-spec teardowns.
    app.state.e2e_project_prune_task = _start_e2e_project_prune_task(app.state.session_factory)

    # P2 tenancy: re-check every licence-backed tenant (active <-> licence_lapsed)
    # every W2V_LICENCE_REFRESH_INTERVAL_SECONDS. See src/services/licence_refresh.py.
    app.state.licence_refresh_task = start_licence_refresh_task(
        app.state.session_factory, settings.licence_refresh_interval_seconds
    )

    # P3 WP3: purge superseded provider-connection secrets no non-terminal run pins.
    # PROVIDER_GC_INTERVAL_SECONDS (<=0 disables). See src/services/providers/gc.py.
    app.state.provider_secret_gc_task = start_provider_secret_gc_task(app.state.session_factory)

    # P6 ORCH-6: resume tenant export/delete jobs whose pod died mid-job (stale lease).
    app.state.offboarding_resume_task = start_offboarding_resume_task(app.state.session_factory)

    # Early-access signup: finish approved signups' unfinished provisioning steps
    # (the set-password email waits for them). Each tick is inert unless
    # W2V_SIGNUP_ENABLED=true. See src/lifecycle/signup_tick.py.
    app.state.signup_tick_task = start_signup_tick_task(app.state.session_factory)

    # Pod-lifetime sweeper that builds the browser-playable h264 proxy for each
    # uploaded video source (video_sources.preview_key / preview_status). Without
    # it every row stays 'pending' and the footage_probe review gate — whose whole
    # job is to LOOK at the footage — shows metadata and no video. Like the prune
    # loop it only needs the session factory. Disabled by default; see
    # PipelineSettings.preview_proxy_enabled.
    app.state.preview_proxy_task = _start_preview_proxy_task(app.state.session_factory)

    # Self-healing shared sound library: re-seed sound_library/shared/ from the
    # bundled assets/ when its manifest is absent (bucket wiped 2026-10-14).
    # Background, best-effort — never blocks readiness. See src/lifecycle/sound_library_seed.py.
    app.state.sound_library_seed_task = start_sound_library_seed_task()

    # Dedicated plaintext metrics listener so the non-ambient Prometheus can
    # scrape /metrics without relaxing the STRICT-mTLS API port (:6300). See
    # _start_metrics_server + the w2v gitops PeerAuth/NetworkPolicy.
    app.state.metrics_httpd = _start_metrics_server(settings.api.metrics_port)

    logger.info("Database connected: %s", database_endpoint(settings.database.url))
    app.state.shutting_down = False
    yield

    # Graceful drain: tell ShutdownMiddleware to refuse new mutating
    # requests, then wait up to ``shutdown_grace_seconds`` for in-flight
    # dispatcher work to finish before cancelling tasks. k8s sends SIGTERM
    # ``terminationGracePeriodSeconds`` before SIGKILL — pick a drain
    # window a few seconds shorter so the dispatcher cancellation +
    # connection close path always has headroom to complete.
    app.state.shutting_down = True
    # Stop the periodic reconciler first so it can't re-dispatch a run while
    # the dispatcher is draining.
    await _cancel_lifespan_tasks(
        app,
        "reconciler_task",
        "audit_prune_task",
        "preview_proxy_task",
        "e2e_project_prune_task",
        "licence_refresh_task",
        "provider_secret_gc_task",
        "offboarding_resume_task",
        "signup_tick_task",
        "sound_library_seed_task",
    )
    await _drain_panel_image_jobs(app)
    _stop_metrics_server(app)
    if dispatcher is not None:
        await _drain_dispatcher(dispatcher, settings.pipeline.shutdown_grace_seconds)
        await dispatcher.shutdown()

    await _close_state_clients(app)
    # The verifier closes over this engine's session factory; drop it before
    # the engine goes so a late key request 401s instead of hitting a dead pool.
    register_api_key_verifier(None)

    if app.state.redis is not None:
        try:
            await close_redis()
        except OSError as exc:
            logger.error("Error closing Redis: %s", exc, exc_info=True)
    try:
        await close_engine()
    except OSError as exc:
        logger.error("Error closing database: %s", exc, exc_info=True)
    logger.info("w2v_orchestrator shutdown complete")


# Built on the shared w2v_common.asgi.create_app factory. It supplies the
# canonical middleware stack — SecurityHeaders + RequestId (forced) + Auth (from
# settings.auth) + Idempotency (from settings.idempotency, tenant-keyed & fixed
# in w2v_common 0.34.0) — and the license-token OpenAPI. The orchestrator's
# service-specific middleware (bespoke browser CORS, drain Shutdown, DB-backed
# Audit, route-template Metrics) are layered via extra_middleware.
#
# create_app registers RequestId OUTER of Auth (0.34.0 fix), so request.state.
# request_id is set even on a 401 Auth short-circuits. extra_middleware registers
# OUTER of the canonical stack; the LAST tuple is the outermost. Ordering the
# list [Metrics, Audit, Shutdown, CORS] yields the request-time (outer→inner)
# stack:  CORS → Shutdown → Audit → Metrics → SecurityHeaders → RequestId →
# Auth → Idempotency → route. Invariants preserved: CORS outermost (preflight
# answered before auth); Shutdown drains 503s before auth/DB cost; Audit outer
# of Auth (sees 401s) and reads request_id AFTER call_next (canonical RequestId,
# inner, has set it); Idempotency inner of Auth (tenant known when keying).
app = create_app(
    settings,
    routers=[
        health_router,
        auth_router,
        license_router,
        blueprints_router,
        # P8 requirements manifest: tenant-agnostic, any authenticated caller.
        provider_requirements_router,
        runs_sse_router,
        projects_sse_router,
        # POST /api/runs/{id}/usage: route-local delegation, binds RLS itself — see
        # routes/runs/usage.py for why it cannot sit under TENANT_GATE.
        runs_usage_router,
        admin_router,
        # Operator YouTube config under /api/admin — per-route gates, like admin_router.
        youtube_admin_router,
        # P2 tenancy (WP4) routers carry PER-ROUTE gates — see routes/tenancy.
        *TENANCY_ROUTERS,
        # Early-access signup: two service doors (own token check) + super_admin
        # routes. Ungated — they are how a tenant comes to exist. Every route 503s
        # unless W2V_SIGNUP_ENABLED=true. See routes/signup_approvals.py.
        *SIGNUP_ROUTERS,
    ],
    title=settings.app_name,
    description="Studio video pipeline orchestration microservice (v3)",
    version=settings.app_version,
    cors_enabled=False,  # browser-facing CORS is bespoke (regex origins) — see below
    extra_middleware=[
        # Innermost extra, but still OUTER of the canonical SecurityHeadersMiddleware
        # (every extra is): it must see the response AFTER `X-Frame-Options: DENY`
        # was stamped, to relax it for the redesign html previews only.
        (FramePolicyMiddleware, {}),
        (MetricsMiddleware, {}),
        (AuditMiddleware, {}),
        (ShutdownMiddleware, {}),
        # P9: body cap on the analysis proxy routes only (413 on exceed): 256 KiB,
        # except POST /api/critique = analysis MAX_HTML_SIZE (10 MiB) + 64 KiB.
        # Streaming count, never a pre-read buffer — it sits OUTER of Auth.
        # Bodies over 256 KiB on that route also take one of N process-wide
        # slots (no waiting; 429 + Retry-After when full).
        (
            ProxyBodyLimitMiddleware,
            {"large_body_slots": settings.api.critique_large_body_slots},
        ),
        # Early-access service doors: public, and FastAPI parses JSON before the
        # route's own token check — 16 KiB streaming cap (413), outer of Auth.
        (SignupDoorBodyLimitMiddleware, {}),
        (
            CORSMiddleware,
            {
                "allow_origin_regex": settings.api.cors_origin_regex,
                "allow_credentials": True,
                "allow_methods": ["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": [
                    "Authorization",
                    "X-Tenant-Id",
                    "X-Active-Tenant-Id",
                    "X-License-Id",
                    "X-Subject-Id",
                    "X-Request-Id",
                    "Content-Type",
                    "Accept",
                ],
                "expose_headers": ["X-Request-Id"],
            },
        ),
    ],
)
# create_app builds a hook-based lifespan; the orchestrator's lifespan is a rich
# single context manager (ORCHESTRATOR_SKIP_LIFESPAN short-circuit + reconciler/
# audit-prune task spawning + graceful drain), so we install it verbatim rather
# than split it into on_startup/on_shutdown hooks.
app.router.lifespan_context = lifespan

# P2 tenancy (WP3): membership resolution, DIRECTLY inside AuthMiddleware and
# outside Idempotency + every route/client, so the tenant a request acts in is
# bound before anything keys on it or calls a peer with it. W2V_TENANCY_SOURCE
# (claim|shadow|membership) picks the behaviour per request; `claim` is a no-op
# for the human lane. Request-time order is now:
#   CORS → Shutdown → Audit → Metrics → SecurityHeaders → RequestId → Auth →
#   Tenancy → Idempotency → route.
install_tenancy_middleware(app)
# The two early-access service doors verify their own service-account token and
# skip AuthMiddleware (exact POST paths only) — see src/auth/signup_doors.py.
install_signup_service_doors(app)

# The gated tenant routers + the /metrics endpoint are mounted AFTER create_app,
# below, so the per-router dependencies and the mount order are preserved.


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus scrape endpoint (exempt from auth via PUBLIC_ENDPOINTS)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Fail-closed tenant-existence gate (Phase C-3) on every tenant-scoped data
# router: a request is served only if its effective tenant is known (see
# require_known_tenant). EXEMPT: health (probes), auth (/api/auth/me must stay
# reachable so a blocked user can load identity → FE tenant-unavailable banner),
# blueprints (tenant-agnostic catalogue), and — as a ROUTER only — admin, whose
# routes carry their own per-route gate (see the mount comment below).
# apply_tenant_rls runs AFTER require_known_tenant (order preserved): it binds
# app.current_tenant for the request transaction when TENANT_RLS_MODE=enforce,
# and is a no-op at the default `off`. Dormant Phase D plumbing (see _common.py).
# require_license (Gate 2) runs last and is ON BY DEFAULT: it 402s a tenant with
# no entitled w2v license (none/expired/revoked) and 503s (license_unavailable)
# when entitlement cannot be established at all — fail-CLOSED. Set
# W2V_LICENSE_GATE_ENABLED=false to break-glass back to a no-op. Exempt:
# health/auth/license/blueprints/SSE, plus the four circular PROVISIONING routes
# on the admin router (per-route — see the mount comment below).
# See _common._enforce_license_state for the 402-vs-503 rationale.
# Single definition, in _common.py, shared with the PER-ROUTE gates on the
# admin router (see TENANT_GATE / LICENSE_GATE_ONLY there), so a router-level
# mount and a per-route mount can never enforce different things.
_tenant_gate = TENANT_GATE

# EXEMPT routers (no _tenant_gate) are mounted by create_app via routers= above:
# health, auth, license, blueprints, runs_sse, projects_sse, runs_usage, admin.
# health/auth/license/blueprints are tenant-agnostic; the SSE routers do their own
# tenant-ownership check on a short-lived RLS-bound session (mounting them under
# the gate would pin a pool connection for the stream's lifetime — the prod
# pool-exhaustion bug). create_app still installs AuthMiddleware, so every one of
# them requires the caller's tenant claim.
#
# `admin` is exempt ONLY as a ROUTER: it mixes two kinds of route and cannot be
# flipped wholesale, so its gates are declared PER-ROUTE in routes/admin.py.
#   * Legitimately exempt (circular — these are how a tenant GETS a licence, so
#     gating them would deadlock onboarding): POST /api/admin/provision,
#     GET/PATCH /api/admin/tenants[/{id}], POST /api/admin/scope-switch.
#   * Gated (product capability that must stop when entitlement stops):
#     GET /api/admin/overview, /usage, /runs, /audit — all with
#     LICENSE_GATE_ONLY, not the full gate; see _common.LICENSE_GATE_ONLY for
#     why require_known_tenant + the RLS bind are deliberately omitted there.
# TenantAdminDep/SuperAdminDep on those routes answer AUTHENTICATION/authz; the
# gate answers ENTITLEMENT. They are not substitutes for one another.
# The GATED data routers are mounted here with the fail-closed tenant gate.
app.include_router(projects_router, dependencies=_tenant_gate)
app.include_router(audit_router, dependencies=_tenant_gate)
app.include_router(deliverables_router, dependencies=_tenant_gate)
app.include_router(review_inbox_router, dependencies=_tenant_gate)
app.include_router(runs_router, dependencies=_tenant_gate)
# Mount after runs_router so the more-specific `/api/runs/{id}/artifact/...`
# path co-exists with runs.py's `/api/runs/{id}/...` routes; FastAPI's
# trie matches by literal segment first.
app.include_router(artifact_proxy_router, dependencies=_tenant_gate)
app.include_router(annotations_router, dependencies=_tenant_gate)
app.include_router(verbatim_router, dependencies=_tenant_gate)
app.include_router(storyboard_router, dependencies=_tenant_gate)
app.include_router(media_router, dependencies=_tenant_gate)
app.include_router(music_router, dependencies=_tenant_gate)
app.include_router(critique_router, dependencies=_tenant_gate)
# P9-7: brand extraction / ICP proxied to analysis (merqi's only route to them).
app.include_router(brand_router, dependencies=_tenant_gate)
# merqi recon A/V leg: ASR + video-analyzer proxied (its only route to them).
app.include_router(media_understanding_router, dependencies=_tenant_gate)
# Cell lane only (the route refuses any other lane with 403); the gate still binds RLS.
app.include_router(internal_provider_slots_router, dependencies=_tenant_gate)
app.include_router(fixes_router, dependencies=_tenant_gate)
app.include_router(redesign_router, dependencies=_tenant_gate)
app.include_router(tts_router, dependencies=_tenant_gate)
# Ad-hoc slot generation (image/tts/llm/video/asr/vlm/ocr) + its history and media
# proxy. Tenant-gated like its neighbours: the gate binds RLS and refuses an
# unknown/unentitled tenant; a plain MEMBER is allowed through (no admin tier).
app.include_router(service_lab_router, dependencies=_tenant_gate)
app.include_router(uploads_router, dependencies=_tenant_gate)
# Project-scoped video-source ingest. Separate router (not uploads_router) because
# every handler in uploads.py buffers the whole body in RAM; see uploads_video.py.
app.include_router(uploads_video_router, dependencies=_tenant_gate)
app.include_router(youtube_router, dependencies=_tenant_gate)

# Map the storyboard lifecycle guard (ProjectArchivedError, raised deep in the repository
# by the by-id mutators) to a uniform 409. Registered here rather than in create_app
# because it is orchestrator-specific; see register_storyboard_exception_handlers.
register_storyboard_exception_handlers(app)
register_tenancy_exception_handlers(app)
register_signup_exception_handlers(app)
register_byo_exception_handlers(app)

# Keep a STRUCTURED HTTPException detail (the typed order-floor refusal, the
# regenerate-scope refusal) a JSON OBJECT on the wire instead of a Python repr
# string. MUST come after create_app's register_error_envelope_handlers, which
# this re-registration deliberately replaces — see src/api/error_details.py.
register_object_detail_http_handler(app)


def main() -> None:
    """Main entry point for running the ASGI application."""
    import uvicorn

    uvicorn.run(
        "src.asgi:app",
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.logging.level.lower(),
        reload=settings.api.reload,
    )


if __name__ == "__main__":
    main()
