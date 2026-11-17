"""Prometheus metrics for the orchestrator — the single home for run + stage
+ silent-failure telemetry.

Layering (one coherent surface, no forks):

* **Inbound HTTP** — the fleet-canonical ``w2v_http_*`` family from
  ``w2v_common.observability.metrics``, applied by ``src/middleware/metrics.py``.
  This module no longer defines its own ``orchestrator_requests_total`` copy.
* **Run lifecycle + stage execution** — orchestrator-specific, all prefixed
  ``w2v_orchestrator_*`` and defined HERE so there is one place to find them.
  ``src/services/middleware/metrics.py`` imports the stage pair from here
  rather than minting its own.

All series register on the prometheus_client default ``REGISTRY`` at import and
are exposed on ``GET /metrics`` (``asgi.py``).
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------
active_runs = Gauge(
    "w2v_orchestrator_active_runs",
    "Runs currently executing on this pod (per-run lease held).",
)

runs_total = Counter(
    "w2v_orchestrator_runs_total",
    "Run lifecycle terminal transitions, by outcome.",
    # completed | failed | cancelled | paused | stale_failed
    ["outcome"],
)

# ---------------------------------------------------------------------------
# Stage execution — defined here, observed by the stage MetricsMiddleware
# (src/services/middleware/metrics.py). Names are unchanged from when they
# lived there, so existing scrapes/dashboards keep working.
# ---------------------------------------------------------------------------
stage_duration_seconds = Histogram(
    "w2v_orchestrator_stage_duration_seconds",
    "Stage handler duration in seconds.",
    ["stage", "outcome"],
    # Stages routinely run for minutes (render/compile/upload), so the default
    # ≤10s buckets were near-useless — extend to 10 minutes.
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)

stage_total = Counter(
    "w2v_orchestrator_stage_total",
    "Stage handler invocations.",
    ["stage", "outcome"],
)

# ---------------------------------------------------------------------------
# GET /api/runs/{run_id} phase breakdown — pure measurement, no behaviour.
#
# That endpoint intermittently 524s. The 524 is Cloudflare's 100s
# origin-response timeout (the tunnel only sets connectTimeout: 30s and the w2v
# VirtualService route allows 3600s, so Istio is not the cutter). Nothing today
# says WHERE those seconds go, so before any fix we split the handler into
# labelled phases whose sum is the whole handler body — see
# src/api/routes/runs/read.py::get_run for the observation sites and for how the
# post-handler serialization cost is derived from these plus the middleware
# total.
#
# Buckets run out to 100s deliberately: the interesting question is "which phase
# is anywhere near the Cloudflare ceiling", which a default <=10s ladder cannot
# answer.
# ---------------------------------------------------------------------------
run_detail_phase_seconds = Histogram(
    "w2v_orchestrator_run_detail_phase_seconds",
    "Wall time of one phase of the GET /api/runs/{run_id} handler body, in "
    "seconds. Phases partition the handler: ownership (tenant-ownership fetch), "
    "stage_query (stage-output SELECT, whole ORM rows incl. the full JSONB "
    "output), other_queries (run-usage read-through + totals + the "
    "deliverable→project lookup), response_build (StageOutputResponse "
    "model_validate + artifact-URL rewrite + RunDetail construction).",
    ["phase"],
    buckets=(0.005, 0.025, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 100.0),
)

run_detail_stages = Histogram(
    "w2v_orchestrator_run_detail_stages",
    "Number of stage-output rows returned by one GET /api/runs/{run_id}. A "
    "cheap volume proxy for the payload: it is the multiplier on both the "
    "stage_query row set and the response_build loop. Deliberately NOT a byte "
    "count — psycopg has already deserialized the JSONB by the time the handler "
    "sees it, so bytes would cost a re-serialize or a second query.",
    buckets=(0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0),
)

# ---------------------------------------------------------------------------
# Silent-failure counters — surface paths that are otherwise best-effort
# swallowed (log-only), so they become alertable instead of invisible.
# ---------------------------------------------------------------------------
notify_errors_total = Counter(
    "w2v_orchestrator_notify_errors_total",
    "Run/pipeline event publish failures (Redis pub/sub). Delivery is "
    "best-effort, so a non-zero rate means live SSE clients may stop receiving "
    "updates while a run keeps progressing.",
)

db_errors_total = Counter(
    "w2v_orchestrator_db_errors_total",
    "Errors caught on a best-effort background path that would otherwise be "
    "swallowed (e.g. a reconciler tick). Distinguished by op.",
    ["op"],
)

cost_unavailable_total = Counter(
    "w2v_orchestrator_cost_unavailable_total",
    "A cost read (run detail / admin overview / admin usage) that could NOT be "
    "priced and was served with cost_status=unavailable instead of a fabricated "
    "$0. reason: billing_credential_missing (BILLING_BASE_URL set but no "
    "out-of-cell credential — CELL_AUTH_MODE=enforce with "
    "DOWNSTREAM_LICENSE_TOKEN unset).",
    ["reason"],
)
cost_unavailable_total.labels(reason="billing_credential_missing")

provider_slot_resolve_total = Counter(
    "w2v_orchestrator_provider_slot_resolve_total",
    "POST /api/internal/provider-slots/resolve answers (satellite provider-slot lookups). "
    "source: tenant | deployment | none; caller: the verified cell issuer (none = "
    "refused lane); outcome: ok | forbidden | unknown_slot | run_not_found | "
    "byo_provider_required | provider_unavailable | keyring_unavailable.",
    ["slot", "source", "caller", "outcome"],
)

# ---------------------------------------------------------------------------
# P7 cell auth (src/clients/downstream_credential.py)
# ---------------------------------------------------------------------------
cell_service_scope_calls_total = Counter(
    "w2v_orchestrator_cell_service_scope_calls_total",
    "CELL_AUTH_MODE=enforce: a tenant-less in-cell call that EXPLICITLY opted in "
    "(downstream_credential.service_scope) and was sent with a scope=service cell "
    "token. destination: the callee's discovery name (bounded: only opted-in "
    "call sites). Every other tenant-less in-cell hop fails closed.",
    ["destination"],
)
# Zero-valued from the first scrape for every opted-in destination.
SERVICE_SCOPE_DESTINATIONS: tuple[str, ...] = ("html_render",)
for _destination in SERVICE_SCOPE_DESTINATIONS:
    cell_service_scope_calls_total.labels(destination=_destination)

# ---------------------------------------------------------------------------
# P2 tenancy cutover (src/auth/tenancy_middleware.py)
# ---------------------------------------------------------------------------
tenancy_shadow_mismatch_total = Counter(
    "w2v_orchestrator_tenancy_shadow_mismatch_total",
    "W2V_TENANCY_SOURCE=shadow: a JWT request the claim served one way and "
    "membership resolution would have served differently (TENANCY_SHADOW_MISMATCH "
    "log line). The shadow -> membership cutover waits for this to stay 0. "
    "reason: no_membership | not_a_member | tenant_differs | "
    "super_admin_unknown_tenant | error (the shadow check itself failed).",
    ["reason"],
)
tenancy_shadow_checked_total = Counter(
    "w2v_orchestrator_tenancy_shadow_checked_total",
    "W2V_TENANCY_SOURCE=shadow: every shadow comparison, once each — the "
    "denominator for tenancy_shadow_mismatch_total, so 'no mismatches' can be "
    "told apart from 'no shadow traffic'. result: match | mismatch | error "
    "(the check failed before reaching a verdict) | claim_missing (the token "
    "carried no tenant claim; served by membership, nothing to compare).",
    ["result"],
)
# Zero-valued from the first scrape: an absent series must mean "not deployed",
# never "nothing compared".
for _result in ("match", "mismatch", "error", "claim_missing"):
    tenancy_shadow_checked_total.labels(result=_result)

licence_refresh_untrusted_total = Counter(
    "w2v_orchestrator_licence_refresh_untrusted_total",
    "Licence refresh (src/services/licence_refresh.py): a TENANT whose status the "
    "sweep could not judge and left unchanged. reason: unavailable (licences "
    "unreachable / 5xx / bad shape) | refused (a definite 4xx from licences or its "
    "gateway — a configuration fault) | unreadable (the stored licence token will "
    "not open — rotated/missing key or corrupt row; needs an operator) | raced (the "
    "token / ref / lease changed while the sweep was checking; the next sweep re-judges).",
    ["reason"],
)

api_key_unattributable_total = Counter(
    "w2v_orchestrator_api_key_unattributable_total",
    "API-key lane (P9-6): a request refused 503 license_unavailable because its "
    "spend could not be attributed to a verified licence. reason (= the response "
    "detail's state): no_ref (the tenant has no licence_ref recorded yet) | "
    "ref_ambiguous (no usable stored token and GET /licenses/active returned "
    "several candidates or the legacy 409, so the ref cannot be singled out — "
    "fixed by storing the tenant's licence token).",
    ["reason"],
)
#: Bounded label set, zero-valued from the first scrape.
API_KEY_UNATTRIBUTABLE_REASONS: tuple[str, ...] = ("no_ref", "ref_ambiguous")
for _reason in API_KEY_UNATTRIBUTABLE_REASONS:
    api_key_unattributable_total.labels(reason=_reason)

tenant_licence_token_expiring = Gauge(
    "w2v_orchestrator_tenant_licence_token_expiring",
    "Tenants whose STORED licence token expires within 30 days (set by every "
    "licence-refresh sweep). An expired stored token is treated as no token — the "
    "tenant falls back to GET /licenses/active — so re-mint and re-send it "
    "(PUT /api/tenants/{id}/licence-token) before it lapses.",
)

tenant_licence_token_expired_total = Counter(
    "w2v_orchestrator_tenant_licence_token_expired_total",
    "A stored tenant licence token found EXPIRED at use; the check fell back to "
    "GET /licenses/active. site: sweep | api_key_gate.",
    ["site"],
)

run_licence_unattributed_total = Counter(
    "w2v_orchestrator_run_licence_unattributed_total",
    "A run started with NO licence of its own tenant to attribute its cost events "
    "to (tenants.licence_ref unset and no tenant-bound request licence). The run's "
    "S2S hops carry an empty X-License-Id — never an umbrella licence. reason: "
    "no_tenant_row | no_licence_ref.",
    ["reason"],
)

request_licence_unattributed_total = Counter(
    "w2v_orchestrator_request_licence_unattributed_total",
    "A membership-resolved (human JWT) request bound NO licence of its tenant to "
    "attribute its request-time cost events to: same resolution as runs "
    "(run_licence) — tenants.licence_ref, never an umbrella licence or the nil UUID. "
    "Its S2S hops carry an empty X-License-Id. reason: no_tenant_row | "
    "no_licence_ref | read_failed.",
    ["reason"],
)
#: Bounded label set, zero-valued from the first scrape.
REQUEST_LICENCE_UNATTRIBUTED_REASONS: tuple[str, ...] = (
    "no_tenant_row",
    "no_licence_ref",
    "read_failed",
)
for _reason in REQUEST_LICENCE_UNATTRIBUTED_REASONS:
    request_licence_unattributed_total.labels(reason=_reason)

growth_bundle_advisory_errors_total = Counter(
    "w2v_orchestrator_growth_bundle_advisory_errors_total",
    "Advisory-insight failures in growth_bundle that are best-effort swallowed "
    "(the bundle still ships). A non-zero rate means the published deliverable is "
    "missing an advisory enrichment. Distinguished by site.",
    ["site"],
)

# ---------------------------------------------------------------------------
# Gestalt ship-gate telemetry — the numeric terms behind final_critique's A11
# advisory ship gate, emitted per demo-video render so the ship-gate ceilings
# can be calibrated from real footage. Observing these does NOT change the gate;
# ``ship_ready`` is decided in gestalt_summary and only *read* here.
# ---------------------------------------------------------------------------
gestalt_momentum_gap = Histogram(
    "w2v_orchestrator_gestalt_momentum_gap",
    "Assembled-vs-predicted watch-curve deficit (worst normalized per-beat "
    "under-delivery, 0..1) measured per render — calibration evidence for the "
    "ship-gate momentum_gap_max ceiling.",
    buckets=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0),
)

gestalt_cohesion = Histogram(
    "w2v_orchestrator_gestalt_cohesion",
    "Whole-film shot-to-shot cohesion score (0..1) measured per render — "
    "calibration evidence for the ship-gate cohesion_floor.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

gestalt_continuity_max_delta = Histogram(
    "w2v_orchestrator_gestalt_continuity_max_delta",
    "Worst shot-to-shot continuity delta (uncensored L2 distribution max) per "
    "render — calibration evidence for the ship-gate continuity_l2 ceiling.",
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
)

gestalt_loudness_lufs = Histogram(
    "w2v_orchestrator_gestalt_loudness_lufs",
    "Compiler-measured integrated loudness (LUFS) of the final mux per render — "
    "calibration evidence for the ship-gate loudness_lufs window.",
    buckets=(-25.0, -20.0, -17.0, -15.0, -14.0, -13.0, -11.0, -8.0),
)

gestalt_ship_gate_total = Counter(
    "w2v_orchestrator_gestalt_ship_gate_total",
    "Ship-gate verdicts per render, labelled by the gate's boolean terms — the "
    "denominator for calibrating how often each ceiling would flip ship_ready.",
    ["ship_ready", "momentum_ok"],
)

gestalt_metrics_emit_errors_total = Counter(
    "w2v_orchestrator_gestalt_metrics_emit_errors_total",
    "Failures while emitting the gestalt ship-gate telemetry above; the emit is "
    "best-effort swallowed (the render still ships) so a non-zero rate means the "
    "calibration series is under-counting, not that any run was affected.",
)


# ---------------------------------------------------------------------------
# TTS clip loudness — the narration level contract, made machine-readable.
#
# The timeline mixes every clip at unity gain and the music duck back-solves from
# a -18 LUFS voiceover, so a clip of UNKNOWN level is a silent product defect.
# `ensure_tts_loudness` (services/stages/primitives/narration.py) either proves a
# level or does not — and EVERY clip lands in exactly one of these two series:
# observed on the histogram when a level is known, counted below otherwise.
#
# That invariant is load-bearing, not decorative: `histogram_count + counter` is
# the total clip population, so a clip missing from both would be invisible in
# exactly the place an auditor looks. It was false once — a normalized clip whose
# trailing advisory measurement failed had no level to observe AND no reason to
# count, while its sidecar claimed `normalized: true`. That case is now the
# `measure_failed` reason. Adding a new outcome to `ensure_tts_loudness` means
# adding it to one of these two series; there is no third option.
# A WARNING log was the only signal before; these are the alertable version.
# ---------------------------------------------------------------------------
tts_clip_loudness_lufs = Histogram(
    "w2v_orchestrator_tts_clip_loudness_lufs",
    "Integrated loudness (LUFS) of each stored narration clip, labelled by who "
    "measured it: 'tts_backend' (X-TTS-LUFS header present) or 'orchestrator' "
    "(header absent, re-mastered here). The distribution should sit on -18.",
    ["source"],
    buckets=(-30.0, -24.0, -21.0, -19.0, -18.5, -17.5, -17.0, -15.0, -12.0, -8.0),
)

tts_clip_unnormalized_total = Counter(
    "w2v_orchestrator_tts_clip_unnormalized_total",
    "Narration clips stored WITHOUT a proven loudness, by why the orchestrator "
    "fallback did not produce one and which stage synthesized it. Most reasons "
    "mean the clip hits the timeline at an arbitrary level; 'measure_failed' is "
    "the exception — loudnorm DID run and its output was kept, only the "
    "confirming measurement is missing, so that clip is very probably correct but "
    "unproven. 'unmeasurable_level' is not a fault at all: a sub-0.5s or "
    "near-silent clip has no integrated loudness to normalize toward.",
    # disabled | unmeasurable_level | premaster_failed_open | duration_drift |
    # measure_failed  -- see ensure_tts_loudness's docstring for what each means.
    ["reason", "stage"],
)
