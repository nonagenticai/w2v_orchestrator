# W2V Orchestrator

Control plane for the web-to-video Studio: turns a tenant's brief into finished media by executing a blueprint — a DAG of stages — against the downstream w2v microservices. Port 6300.

## What W2V Orchestrator Does

W2V Orchestrator is the single entry point tenants use to produce media on the web-to-video platform. Since the Studio v3 reset the API is organised around **projects → deliverables → runs**: a tenant creates a project of a given `type` (promo, course, lesson, app_review, website_review, demo_video, web_design, audiobook, or the plain `generic` type), the project holds one or more deliverables, and each deliverable is executed by a **run**. Every project type resolves to exactly one **blueprint** — a directed acyclic graph of stages such as `brief_intake`, `script_draft`, `narrate`, `record`, `compile`, `analyze`, and `upload` — registered at import time in `src/services/blueprints/registry.py`.

When a run is submitted, the run engine resolves the deliverable's blueprint, topologically walks the stage DAG honouring each stage's `depends_on`, and executes stages in order. Some stages run locally in-process (the LLM stages, which call `worker_management`'s `POST /workers/{name}/message` with a per-call `output_schema` for structured output — never an inference router or a provider directly); the rest dispatch over HTTP to a peer w2v service — text-to-speech, recording, compilation, verification, slide generation, image/video generation, transcription, or upload. Peer URLs are resolved through `w2v_common.discovery.service_url(<name>)`, never hardcoded, and the caller's identity (`X-Tenant-Id` / `X-License-Id` / `X-Subject-Id`) is forwarded on every service-to-service call so the downstream service authorizes the original tenant.

Each stage's output and artifact URLs are persisted to PostgreSQL as the run advances, and live progress is streamed to the SPA over Server-Sent Events (`GET /api/runs/{id}/events`) fanned out from Redis Pub/Sub. `pause_after_stages` checkpoints let a run halt for human review and resume; annotations, element restamps, and per-stage regeneration let a tenant refine a specific stage's output without re-running the whole DAG. A run also carries a single resolution tier (`1080p`, `2k` default, or `4k`) threaded end-to-end so the final video is sharp at the source rather than an upscale of 1080p assets.

The orchestrator is the most database-backed w2v service and the only one that owns a `src/database/models/` layer — projects, deliverables, runs, stage outputs, annotations, run usage, verbatim items, tenants, members, and audit tables all live in its schema. **Row-Level Security is live in production (enforce mode).** Runtime connects as the restricted role `w2v_orch_app` with `app.current_tenant` bound per transaction, so a tenant can only ever read its own rows; the privileged owner `w2v_service` is reserved for alembic/DDL. It is also the **only multi-replica w2v service** — it runs `replicas:2` behind an HPA (min 2, max 5), so every cross-pod concern (run dispatch, cancel/pause broadcast, idempotency cache, orphan-run reconciliation) is engineered to work across replicas via Redis leases and a periodic reconciler. `replicas:1` is the video-compiler's invariant, not the orchestrator's.

## Architecture

```
                                    ┌──────────────────┐
   SPA / tenant ──X-Tenant-Id──────►│ W2V Orchestrator │◄──── SSE run progress
                                    │   (replicas 2-5) │
                                    └────────┬─────────┘
             ┌──────────────┬──────────────┼──────────────┬──────────────┐
             ▼              ▼              ▼              ▼              ▼
      ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
      │worker_mgmt │ │tts_gateway │ │video_prod +│ │video_      │ │slide +     │
      │  (LLM)     │ │            │ │video_compile│ │analyzer    │ │html_render │
      └────────────┘ └────────────┘ └────────────┘ └────────────┘ └─────┬──────┘
                                                        youtube_upload ──┘
             │                                                    │
             ▼                                                    ▼
   ┌────────────────────┐                                ┌────────────────┐
   │ PostgreSQL 18 (RLS)│                                │ MinIO artifacts│
   │  via PgBouncer     │        Redis (SSE + leases) ───┤                │
   └────────────────────┘                                └────────────────┘
```

## Features

- **Project → deliverable → run model** — the Studio v3 API structures every job as a typed project holding deliverables that are executed by runs, so multi-output and parent/child (course → lesson) work is first-class rather than bolted onto a flat pipeline.
- **Blueprint DAG execution** — each project type resolves to one registered blueprint whose stages run in topological order honouring `depends_on`, so adding a project type means adding a blueprint module, not editing an executor.
- **Local LLM + HTTP dispatch stages** — LLM stages run in-process via `worker_management` with per-call JSON-schema output; all other stages dispatch to peer services resolved through service discovery, so the orchestrator owns coordination while specialised work stays in the owning service.
- **Durable, cross-pod runs** — a per-run Redis execution lease plus a startup and periodic orphan reconciler resume runs left `running` by a drained or hard-killed pod, so the 2-5 replica fleet survives rollouts and OOM kills without wedging runs.
- **Live SSE progress** — run progress publishes to Redis `runs:<id>` and streams to clients as `text/event-stream`, so the SPA renders live stage status without polling or WebSockets.
- **Pause / resume / regenerate** — `pause_after_stages` checkpoints halt a run for review; annotations, element restamps, and per-stage regeneration refine one stage's output without re-running the DAG.
- **Resolution tiers** — a single `1080p` / `2k` / `4k` tier is resolved once per run and threaded through every pixel source (slides, motion clips, recordings, final compile CRF) so uploads exceed soft 1080p.
- **Idempotency** — mutating POSTs keyed by `Idempotency-Key` are cached (24h) so client/DLQ retries don't double-submit non-idempotent stages.
- **Row-Level Security** — the restricted `w2v_orch_app` runtime role plus a per-transaction `app.current_tenant` GUC enforce tenant isolation at the database, not just the application layer.
- **Admin tier** — super/tenant-admin routes browse runs, tenants, members, usage, and audit logs, with a scope-switch (`X-Active-Tenant-Id`) that lets a super-admin operate as a chosen tenant.
- **Request audit + retention** — every mutation and auth failure persists a `request_audit` row; a pod-lifetime loop prunes rows past the retention window so an unauthenticated scanner can't fill the disk.
- **Graceful drain** — the ASGI lifespan refuses new mutating requests on SIGTERM and waits for in-flight dispatch to finish (leaving drained runs `running` for the next pod's reconciler) before force-cancelling.
- **Dedicated metrics listener** — a plaintext Prometheus server on port 6301 lets the non-ambient scraper read `/metrics` while the API port 6300 stays under STRICT mTLS.

## Authentication

Inbound requests present a license token as `X-Tenant-Id: Bearer <license_token>` — **not** `Authorization`, which is reserved for Keycloak JWTs emitted by the SPA. The `AuthMiddleware` (re-mounting `w2v_common.auth.middleware.AuthMiddleware`) validates the token, introspecting against the licenses service and caching results for 5 minutes. `AUTH_TRUST_INCLUSTER_IDENTITY` is forced to `false` at import time, so the SSR-proxy identity bypass is closed. Auth is enabled by default.

| Priority | Method | Details |
|----------|--------|---------|
| 1 | License token | `X-Tenant-Id: Bearer <token>` introspected against the licenses service (5-min cache) |
| 2 | Keycloak JWT | `Authorization: Bearer <jwt>` from the SPA (portal realm) |
| 3 | Trusted network | Bypass for internal cluster CIDRs → default tenant (no token) |

Beyond authentication, tenant-scoped data routers pass through a three-gate dependency chain: `require_known_tenant` (fail-closed tenant-existence check, `TENANT_ENFORCEMENT_MODE=enforce`), `apply_tenant_rls` (binds `app.current_tenant` for Postgres RLS, `TENANT_RLS_MODE=enforce` in prod), and `require_license` (per-tenant license gate, default off). **RLS is enforced in production** via the restricted `w2v_orch_app` role. Admin and BYO-LLM routers skip the shared gate and carry their own super/tenant-admin dependencies; a super-admin may operate against another tenant by sending `X-Active-Tenant-Id`.

Exempt from the tenant gate: health probes, `/api/auth/me` (so a blocked user can still load identity), `/api/license/status`, `/api/blueprints` (tenant-agnostic catalogue), the SSE stream routers (which do their own short-lived ownership check to avoid pinning a pool connection), the admin routers, and `/metrics`.

## Quick Start

```bash
uv sync --all-extras
uv run uvicorn src.asgi:app --host 0.0.0.0 --port 6300
uv run pytest tests/ -v --cov=src
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

## API Endpoints

All routes are unversioned (`/api/<resource>`, never `/api/v1/`).

### Auth & License

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/auth/me` | Echo identity claims for the inbound caller |
| GET | `/api/license/status` | Per-tenant license state (active / expiring / expired / none) |

### Blueprints `/api/blueprints`

| Method | Path | Description |
|--------|------|-------------|
| GET | `` | List registered blueprints and their stage DAGs |

### Projects `/api/projects`

| Method | Path | Description |
|--------|------|-------------|
| GET | `` | List projects (filters: type, status, parent_id, include_archived) |
| POST | `` | Create a project + N deliverables atomically |
| GET | `/{project_id}` | Project detail with deliverables + children |
| PATCH | `/{project_id}` | Update project fields |
| DELETE | `/{project_id}` | Soft-delete (archive) a project |
| GET | `/{project_id}/runs` | List runs for a project |
| GET | `/{project_id}/children` | List child projects |
| POST | `/{project_id}/children` | Create child project under parent |
| PATCH | `/{project_id}/children/reorder` | Reorder children |
| GET | `/{project_id}/events` | SSE stream of child-run status |

### Deliverables `/api/deliverables`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/{deliverable_id}` | Get a deliverable |
| PATCH | `/{deliverable_id}` | Update deliverable fields |
| DELETE | `/{deliverable_id}` | Delete a deliverable |
| POST | `/{deliverable_id}/runs` | Submit a run (accepts `pause_after_stages`, `config_overrides`) |

### Runs `/api/runs`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/{run_id}` | Run detail with stage outputs + token usage |
| GET | `/{run_id}/status` | Lightweight run status |
| GET | `/{run_id}/events` | SSE stream for live progress |
| GET | `/{run_id}/captions.srt` | Download soft-CC captions |
| GET | `/{run_id}/ancestors` | Ancestor run chain |
| POST | `/{run_id}/cancel` | Cancel a running run |
| POST | `/{run_id}/pause` | Pause a running run |
| POST | `/{run_id}/resume` | Resume a paused run |
| GET | `/{run_id}/checkpoints` | List pause checkpoints |
| PATCH | `/{run_id}/checkpoints` | Update pause checkpoints |
| POST | `/{run_id}/regenerate` | Regenerate from a stage |
| GET | `/{run_id}/stages/{stage_name}/output` | Get a stage's output |
| PATCH | `/{run_id}/stages/{stage_name}/elements/{element_id}` | Edit a stage element |
| POST | `/{run_id}/stages/{stage_name}/elements/{element_id}/restamp` | Restamp a stage element |
| PATCH | `/{run_id}/stages/course_stitch/chapters` | Edit course chapters |
| PATCH | `/{run_id}/stages/course_stitch/thumbnail` | Set course thumbnail |
| GET | `/{run_id}/stages/{stage_name}/annotations` | List stage-output annotations |
| POST | `/{run_id}/stages/{stage_name}/annotations` | Create a stage-output annotation |
| GET | `/{run_id}/artifact/{bucket}/{object_key:path}` | Proxy-download a run artifact from MinIO |

### Annotations `/api/annotations`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/{annotation_id}` | Get an annotation |
| PATCH | `/{annotation_id}` | Update an annotation |
| DELETE | `/{annotation_id}` | Delete an annotation |

### Verbatim Items

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/projects/{project_id}/verbatim-items` | List verbatim items for a project |
| POST | `/api/projects/{project_id}/verbatim-items` | Create a verbatim item |
| PATCH | `/api/verbatim-items/{item_id}` | Update a verbatim item |
| DELETE | `/api/verbatim-items/{item_id}` | Delete a verbatim item |

### Proxies (→ analysis_service)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/critique` | Create a critique job |
| GET | `/api/critique` | List critique jobs (translates `page_size`→`per_page`) |
| GET | `/api/critique/{job_id}` | Get a critique job |
| GET | `/api/critique/{job_id}/report/pdf` | Download critique PDF report |
| POST | `/api/redesign` | Create a redesign job |
| GET | `/api/redesign/jobs/{job_id}` | Get a redesign job |
| GET | `/api/redesign/jobs/{job_id}/result` | Get redesign result |
| GET | `/api/redesign/jobs/{job_id}/html` | Get redesign HTML |
| GET | `/api/redesign/jobs/{job_id}/pdf` | Get redesign PDF |
| GET | `/api/redesign/jobs/{job_id}/variants/{style}/html` | Get a redesign style variant |

### Media, TTS & Uploads

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/media/search` | Search stock/royalty-free media |
| POST | `/api/tts/preview` | Cached voice-preview audio (provider, voice, speed) |
| POST | `/api/uploads/manuscript` | Upload an audiobook manuscript (multipart) |
| POST | `/api/uploads/thumbnail` | Upload a thumbnail image |
| POST | `/api/uploads/document` | Upload a source document |

### YouTube `/api/youtube`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload` | Submit a YouTube upload |
| GET | `/upload/jobs/{job_id}` | Get an upload job |
| GET | `/upload/jobs` | List upload jobs |

### Tenant provider connections `/api/tenant/connections` (tenant_admin)

| Method | Path | Description |
|--------|------|-------------|
| GET | `` | One entry per provider slot (deployment default or the tenant's connection; never the secret) |
| PUT | `/{slot}` | Set a connection — policy + SSRF check, live test, then a new immutable version |
| POST | `/{slot}/test` | Test a draft (body) or the stored version |
| DELETE | `/{slot}` | Remove the tenant's connection (back to the deployment default) |

### Admin `/api/admin`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/runs` | List runs across tenants |
| GET | `/tenants` | List tenants |
| PATCH | `/tenants/{tenant_id}` | Update a tenant |
| POST | `/scope-switch` | Switch super-admin active tenant |
| GET | `/audit` | Admin audit log |
| GET | `/request-audit` | Request audit log |
| GET | `/members` | List tenant members |
| GET | `/overview` | Admin overview rollup |
| GET | `/usage` | Usage detail |
| POST | `/provision` | Provision a tenant (Keycloak + license) |
| POST | `/members/{subject_id}/roles` | Set member roles |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/healthz` | Liveness probe |
| GET | `/readyz` | Readiness probe (DB + Redis reachability) |
| GET | `/health`, `/health/live`, `/health/ready` | Legacy probe aliases (hidden from schema) |
| GET | `/metrics` | Prometheus scrape (also on dedicated port 6301) |

## Configuration

All configuration via environment variables. Infrastructure settings are inherited from `w2v_common` with orchestrator overrides.

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Listen host |
| `API_PORT` | `6300` | API server port |
| `API_METRICS_PORT` | `6301` | Dedicated plaintext Prometheus metrics port |
| `API_DEBUG` | `false` | Enable /docs and /redoc |
| `API_CORS_ORIGIN_REGEX` | `None` | Allowed CORS origin regex (set via gitops ConfigMap) |
| `AUTH_ENABLED` | `true` | Toggle license/JWT auth middleware |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_JSON_LOGS` | `false` | Structured JSON logging |
| `ENVIRONMENT` | `development` | Deployment environment |
| `TENANT_ENFORCEMENT_MODE` | `enforce` | Fail-closed tenant-existence gate (off / warn / enforce) |
| `TENANT_RLS_MODE` | `off`* | Bind `app.current_tenant` per request (off / enforce). *Prod ConfigMap sets `enforce`. |
| `DATABASE_URL` | `postgresql+psycopg://localhost:5450/w2v_orchestrator` | Postgres URL (privileged owner path) |
| `RUNTIME_DATABASE_URL` | — | Restricted `w2v_orch_app` role URL used at runtime under RLS |
| `REDIS_URL` | — | Redis URL (required; SSE fan-out + run leases) |
| `REDIS_MAX_CONNECTIONS` | `100` | Redis pool size (SSE streams hold one conn each) |
| `STORAGE_MINIO_ENDPOINT` | — | MinIO endpoint (required) |
| `STORAGE_MINIO_BUCKET` | `w2v-orchestrator` | MinIO bucket |
| `DOWNSTREAM_LICENSE_TOKEN` | (dev default) | Service-to-service license token |
| `DOWNSTREAM_TTS_GATEWAY_URL` | `http://tts-gateway.w2v:6320` | TTS gateway (claim-queue) base URL |
| `PIPELINE_MAX_CONCURRENT_PIPELINES` | `10` | Max concurrent runs |
| `PIPELINE_STAGE_DEFAULT_TIMEOUT` | `1800` | Default per-stage timeout (seconds) |
| `PIPELINE_SHUTDOWN_GRACE_SECONDS` | `100` | Drain window for in-flight dispatch on SIGTERM |
| `PIPELINE_RECONCILER_MAX_AGE_SECONDS` | `7200` | Orphan-run resume cutoff |
| `PIPELINE_IDEMPOTENCY_TTL_SECONDS` | `86400` | Cached-response TTL for `Idempotency-Key` |
| `PIPELINE_REQUEST_AUDIT_RETENTION_DAYS` | `90` | `request_audit` retention (0 disables prune) |
| `W2V_LICENSE_GATE_ENABLED` | `false` | Report real per-tenant license state on /api/license/status |
| `W2V_WEB_DESIGN_REQUIRE_VERIFICATION` | `true` | Fail a web_design_v2 run with no verification result |
| `KEYCLOAK_URL` / `KEYCLOAK_CLIENT_SECRET` | — | Admin-tier provisioning (client-credentials) |
| `OTLP_TRACES_ENDPOINT` | — | OpenTelemetry OTLP HTTP endpoint |

## Architecture

```
src/
├── asgi.py                       # FastAPI app, lifespan, middleware stack, orphan reconciler, metrics server
├── config.py                     # Pydantic settings (API 6300 / metrics 6301, DB, Redis, MinIO, pipeline, downstream)
├── utils.py                      # Cross-cutting helpers
├── api/
│   ├── artifact_url_rewriter.py  # Rewrites internal MinIO URLs to external presigned
│   ├── schemas.py                # Request/response Pydantic models
│   ├── sse.py                    # Server-Sent Events fan-out from Redis Pub/Sub
│   └── routes/                   # 20 routers mounted in asgi.py
│       ├── _common.py            # TenantIdDep, SessionDep, tenant-gate deps, repo factories
│       ├── _proxy.py             # Shared proxy helper (critique/redesign → analysis_service)
│       ├── auth.py               # /api/auth/me + /api/license/status
│       ├── blueprints.py         # /api/blueprints
│       ├── projects.py           # /api/projects (+ SSE child-status router)
│       ├── deliverables.py       # /api/deliverables (+ run submit)
│       ├── runs.py               # /api/runs (+ SSE events router)
│       ├── artifact_proxy.py     # /api/runs/{id}/artifact/... MinIO proxy
│       ├── annotations.py        # /api/annotations
│       ├── verbatim.py           # /api/projects/{id}/verbatim-items
│       ├── critique.py           # /api/critique proxy
│       ├── redesign.py           # /api/redesign proxy
│       ├── media.py              # /api/media/search
│       ├── tts.py                # /api/tts/preview
│       ├── uploads.py            # /api/uploads (manuscript/thumbnail/document)
│       ├── youtube.py            # /api/youtube upload
│       ├── admin.py              # /api/admin (runs, tenants, members, audit, provision)
│       └── health.py             # /healthz, /readyz, legacy aliases
├── auth/                         # AuthMiddleware re-mount + orchestrator subject_id/ContextVar deps
├── cache/redis_client.py         # Redis client init/close
├── clients/                      # ~25 HTTP clients for downstream services (base.py = BaseServiceClient)
├── database/
│   ├── connection.py             # Async engine (owner + restricted runtime roles)
│   ├── models.py                 # SQLAlchemy models (projects, deliverables, runs, stage_outputs, ...)
│   ├── repositories/             # Repository per aggregate (Project, Run, StageOutput, Annotation, ...)
│   ├── rls.py / rls_events.py    # RLS GUC bind + after_begin re-bind listener
│   ├── session.py                # bind_session_tenant helper
│   └── migrations/versions/      # Alembic (linear head; 0015 = Studio v3 reset, 0027-0031 = RLS)
├── middleware/                   # audit, idempotency, metrics, request_id, security, shutdown
├── observability/                # Structured logging, Prometheus metrics, OTel tracing
├── models/                       # auth_config, source_spec value objects
├── security/                     # SSRF validator for outbound URLs
└── services/
    ├── run_dispatcher.py         # Tenant-aware dispatch, cross-pod control listener, run leases
    ├── run_engine.py             # Run execution loop (resolves blueprint, walks DAG)
    ├── runs/                     # dag_executor, sub_dag, run_setup, run_lease, regenerate, elements
    ├── blueprints/               # One blueprint module per project type + registry
    ├── stages/                   # Stage handlers grouped by project type + shared/primitives
    ├── evaluation/               # Deterministic script-craft rubric (script_doctor's gate)
    └── middleware/               # Stage-level retry / timeout / metrics chain
```

> `package.json` (name `web-to-video`) declares only `@playwright/test` + `typescript` dev-deps — it is a TypeScript Playwright recording/scaffold toolkit for generating recording scripts, **not** a runnable Node service.

## Testing

```bash
uv run pytest tests/ -v --cov=src          # All tests with coverage (needs CI postgres + fakeredis)
uv run pytest tests/unit/ -v               # Unit tests
uv run pytest tests/integration/ -v        # Integration tests (marker: integration; skipped by default)
uv run pytest tests/contracts/ -v          # Contract tests
uv run pytest tests/e2e/ -v                # In-repo e2e scenarios
uv run ruff check src/ tests/              # Lint
uv run ruff format --check src/ tests/     # Format check
uv run ty check .                          # Type check
```

Coverage threshold: 80% (enforced in pyproject.toml). Tests run against real Postgres (`DATABASE_URL` captured before the autouse settings-clear fixture) plus `fakeredis`; the `integration` and `cross_service_monorepo` markers are de-selected by default.

The end-to-end suite that exercised a live deployment in the private estate is not part of this published source tree; per-PR CI runs the unit suite only.

## CI/CD

`.github/workflows/ci-public.yml` is self-contained and runs on GitHub-hosted runners: a secret scan (betterleaks), the `no-private-refs` guard, then `ruff check`, `ruff format --check`, `pytest` against a Postgres 18 service container, and the repository's `scripts/pip-audit.sh`.

No image is published from this repository. Build it locally with the Dockerfile (see Docker below).

## Docker

```bash
docker build -t w2v-orchestrator .
docker run -p 6300:6300 w2v-orchestrator
docker run -p 6300:6300 -e LOG_LEVEL=DEBUG -e AUTH_ENABLED=false \
  -e DATABASE_URL=postgresql+psycopg://host/w2v_orchestrator w2v-orchestrator
```

These are for local testing only. Production deployment is via CI/CD (push → GitHub Actions → registry.test:5000 → Flux → K3s).

## Dependencies

Cross-service (URLs resolved via `w2v_common.discovery.service_url`, never hardcoded):

- **worker_management** (port 8887) — all LLM stages route here via `POST /workers/{name}/message` with per-call `output_schema`
- **tts_gateway** (port 6320) — pooling TTS claim-queue for narration
- **video_production** (port 6308) — browser/Playwright recording + compilation
- **video_compiler** (port 8200) — terminal video encode (per-tier CRF)
- **video_analyzer** (port 6317) — recorded-video verification/analysis
- **html_render** (port 6319) — HTML→image/PDF rendering (web_design assembly + verify)
- **slide_service** (port 6312) — slide-deck PNG generation
- **youtube_upload** (port 6315) — YouTube publish + poll
- **analysis_service**, **content**, **ai_content_generation**, **app_inspector**, **bdd_playwright**, **course**, **frontend_test_runner**, **transcription_worker**, **kag/webscraper** — additional stage-dispatch peers
- **licenses** (port 8544) — license-token introspection (auth) + admin-tier license minting
- **worker/artifacts** platform services — cross-namespace peers via discovery

Infrastructure:

- **PostgreSQL 18** — persistent store via PgBouncer (`pgbouncer.pgbouncer:6432`); RLS enforced with a restricted runtime role
- **Redis** — SSE Pub/Sub fan-out, run execution leases, idempotency + reconcile counters
- **MinIO** — artifact storage (`w2v-orchestrator` bucket + cross-`w2v-*` reads)

Libraries (from `pyproject.toml`): **FastAPI** · **Uvicorn** · **Pydantic v2 / pydantic-settings** · **SQLAlchemy (async)** + **Alembic** · **psycopg 3** (binary) · **redis** · **miniopy-async** · **httpx** · **sse-starlette** · **prometheus-client** · **PyJWT** · **python-docx / EbookLib** (manuscript parsing) · **python-multipart** · **w2v-common ≥ 0.31.0** · **OpenTelemetry** (API/SDK/OTLP-HTTP).

## Known Issues

- **Mixed DB-access pattern** — the repository layer (`src/database/repositories/`) is the intended path, but a few route and stage handlers still issue direct `sqlalchemy.select()` calls that bypass it (e.g. `_tenant_owns_run` in `routes/runs.py`). Known tech debt; new code goes through repositories.
- **`pip-audit` ignore: PYSEC-2025-183** — pyjwt 2.12.1 advisory has no upstream fix; ignored in CI until the next dependency sweep bumps the pin.
- **Auto-commit hook may split commits** — the repo may split in-flight changes into multiple commits authored as `georgiedekker`; be aware when committing (does not affect runtime).
```

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Licensing

This service contains no licence enforcement of its own and runs no licence server.
Tenant and licence identity is forwarded to peer services, which perform the checks.
It runs with no licence switch set; unset means off.
