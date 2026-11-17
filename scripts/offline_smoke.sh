#!/usr/bin/env bash
# offline_smoke.sh -- end-to-end smoke test against the local docker-compose
# stack with EVERY LLM / TTS / image / video / vision call routed to a
# deterministic stub provider. No paid APIs, no GPU, no LAN dependencies.
#
# Prerequisites:
#   - Docker + docker compose v2 available on PATH
#   - All five stub provider features present in the local working trees:
#       analysis_service:           LLM_PROVIDER=stub
#       content_service:            LLM_PROVIDER=stub
#       tts_service:                TTS_PROVIDER=stub + STUB_ENABLED=true
#       ai_content_generation:      IMAGE_PROVIDER=stub + VIDEO_PROVIDER=stub
#       video_analyzer_service:     ANALYZER_PROVIDER=stub
#     The script verifies each by grepping the service source tree and aborts
#     loudly if any stub provider is missing.
#
# Exit codes:
#   0   Pipeline reached a terminal success state and the artifact assertion
#       passed.
#   2   Missing prerequisite (stub provider not found, docker not running).
#   1   Anything else -- pipeline failure, timeout, HTTP error.

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths (script lives in w2v_orchestrator/scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCHESTRATOR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$ORCHESTRATOR_DIR/.." && pwd)"
COMPOSE_BASE="$WORKSPACE_DIR/docker-compose.yml"
COMPOSE_OVERRIDE="$WORKSPACE_DIR/docker-compose.override.yml"
COMPOSE_OFFLINE="$SCRIPT_DIR/docker-compose.offline.yml"
SCENARIO="$SCRIPT_DIR/e2e_scenarios/offline_smoke.json"

ORCH_HOST="${ORCH_HOST:-http://localhost:6300}"
TENANT="${TENANT:-0195f48b-d000-7000-8000-000000000001}"
DEADLINE="${DEADLINE_SECONDS:-300}"

# ---------------------------------------------------------------------------
# Stub provider env vars -- exported so docker compose interpolates them.
# ---------------------------------------------------------------------------
export LLM_PROVIDER=stub
export TTS_PROVIDER=stub
export STUB_ENABLED=true
export IMAGE_PROVIDER=stub
export VIDEO_PROVIDER=stub
export ANALYZER_PROVIDER=stub
# Belt-and-suspenders: blank any paid API keys so a leaked default cannot
# accidentally hit a real endpoint.
export DEEPSEEK_API_KEY=""
export OPENAI_API_KEY=""
export ANTHROPIC_API_KEY=""
export FAL_KEY=""
export RUNWAY_API_KEY=""
export GOOGLE_AI_API_KEY=""
export GOOGLE_API_KEY=""
export ELEVENLABS_API_KEY=""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { printf '[offline_smoke] %s\n' "$*" >&2; }
fail() { printf '[offline_smoke] FAIL: %s\n' "$*" >&2; exit "${2:-1}"; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1" 2
}

verify_stub_present() {
    local service_dir="$1" label="$2"
    shift 2
    local found=0
    for pattern in "$@"; do
        if grep -rqs --include='*.py' -E "$pattern" "$WORKSPACE_DIR/$service_dir/src"; then
            found=1
            break
        fi
    done
    if (( found == 0 )); then
        fail "stub provider missing in $service_dir (none of: $*). Ensure the $label PR is on the local branch." 2
    fi
}

compose() {
    docker compose \
        -f "$COMPOSE_BASE" \
        -f "$COMPOSE_OVERRIDE" \
        -f "$COMPOSE_OFFLINE" \
        --project-name w2v_offline_smoke \
        "$@"
}

cleanup() {
    local rc=$?
    log "cleaning up compose stack (exit=$rc)"
    compose down --remove-orphans --volumes >/dev/null 2>&1 || true
    exit "$rc"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
require_cmd docker
require_cmd curl
require_cmd python3

[[ -f "$COMPOSE_BASE" ]]    || fail "compose file not found: $COMPOSE_BASE" 2
[[ -f "$COMPOSE_OFFLINE" ]] || fail "offline override missing: $COMPOSE_OFFLINE" 2
[[ -f "$SCENARIO" ]]        || fail "scenario file missing: $SCENARIO" 2

if ! docker info >/dev/null 2>&1; then
    fail "docker daemon is not reachable" 2
fi

verify_stub_present "analysis_service"              "analysis_service LLM stub" \
    'is_stub_enabled' 'LLM_PROVIDER.*stub' 'llm_stub'
verify_stub_present "content_service"               "content_service LLM stub" \
    'stub_client' 'is_stub_enabled' 'LLM_PROVIDER.*stub'
verify_stub_present "tts_service"                   "tts_service stub" \
    'StubProvider' 'TTS_PROVIDER.*stub' 'STUB_ENABLED'
verify_stub_present "ai_content_generation_service" "ai_content_generation_service stub" \
    'IMAGE_PROVIDER' 'VIDEO_PROVIDER' 'StubImageProvider' 'StubVideoProvider'
verify_stub_present "video_analyzer_service"        "video_analyzer_service stub" \
    'ANALYZER_PROVIDER' 'StubAnalyzer' 'analyzer.*stub'

# ---------------------------------------------------------------------------
# Bring up the stack
# ---------------------------------------------------------------------------
export COMPOSE_PROFILES=services
log "compose: building & starting infra + microservices (this can take a while on first run)"
compose up -d --build --quiet-pull

# ---------------------------------------------------------------------------
# Wait for orchestrator readiness (and downstreams via /health/downstream)
# ---------------------------------------------------------------------------
wait_for_http() {
    local url="$1" timeout="$2" elapsed=0 sleep_s=3
    while (( elapsed < timeout )); do
        if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep "$sleep_s"
        elapsed=$(( elapsed + sleep_s ))
    done
    return 1
}

log "waiting for orchestrator at $ORCH_HOST/health (up to 180s)"
if ! wait_for_http "$ORCH_HOST/health" 180; then
    log "orchestrator failed to become ready -- dumping logs"
    compose logs --tail=200 orchestrator >&2 || true
    fail "orchestrator never returned 200 on /health"
fi

log "checking downstream readiness via $ORCH_HOST/health/downstream"
if ! curl -fsS --max-time 10 "$ORCH_HOST/health/downstream" >/dev/null 2>&1; then
    log "WARNING: /health/downstream did not return 200; some peers may be unhealthy"
fi

# ---------------------------------------------------------------------------
# Submit pipeline + poll
# ---------------------------------------------------------------------------
log "submitting pipeline against $ORCH_HOST (tenant=$TENANT)"
SUBMIT_RESP="$(curl -fsS -X POST \
    -H 'Content-Type: application/json' \
    -H "X-Tenant-Id: $TENANT" \
    --data @"$SCENARIO" \
    "$ORCH_HOST/api/pipeline")"
JOB_ID="$(printf '%s' "$SUBMIT_RESP" | python3 -c 'import json,sys;print(json.load(sys.stdin)["job_id"])')"
log "submitted job_id=$JOB_ID"

deadline=$(( SECONDS + DEADLINE ))
final_body=""
final_status=""
while (( SECONDS < deadline )); do
    body="$(curl -fsS "$ORCH_HOST/api/pipeline/$JOB_ID" || true)"
    if [[ -n "$body" ]]; then
        status="$(printf '%s' "$body" | python3 -c 'import json,sys;
try:
    print(json.load(sys.stdin).get("status",""))
except Exception:
    print("")')"
        case "$status" in
            succeeded)
                final_body="$body"; final_status="$status"; break ;;
            failed|cancelled)
                final_body="$body"; final_status="$status"; break ;;
        esac
    fi
    sleep 5
done

if [[ -z "$final_status" ]]; then
    log "timeout waiting $DEADLINE s for job $JOB_ID; last body: $body"
    fail "pipeline did not reach a terminal state within ${DEADLINE}s"
fi

if [[ "$final_status" != "succeeded" ]]; then
    log "pipeline terminated with status=$final_status"
    printf '%s\n' "$final_body" | python3 -m json.tool >&2 || printf '%s\n' "$final_body" >&2
    fail "pipeline reached terminal state but not 'succeeded' (got: $final_status)"
fi

# ---------------------------------------------------------------------------
# Artifact assertion: at least one stage must have produced an output
# artifact (a non-empty 'output' / 'artifacts' / 'result' field). For
# dry_run mode the terminal artifact is the generated shotlist + script
# bundle; we accept any of these as proof-of-work.
# ---------------------------------------------------------------------------
ARTIFACT_OK="$(printf '%s' "$final_body" | python3 -c '
import json,sys
body = json.load(sys.stdin)
stages = body.get("stages") or []
succeeded = [s for s in stages if (s.get("status") or "").lower() == "succeeded"]
if not succeeded:
    print("no_succeeded_stages"); sys.exit(0)
# Look for any non-trivial artifact field
for s in succeeded:
    for k in ("output","artifacts","result","artifact_url","artifact_path"):
        v = s.get(k)
        if v not in (None, "", [], {}):
            print("ok"); sys.exit(0)
print("no_artifact")
')"

case "$ARTIFACT_OK" in
    ok)
        log "artifact assertion passed; pipeline $JOB_ID succeeded with $(printf '%s' "$final_body" | python3 -c 'import json,sys;print(len((json.load(sys.stdin).get("stages") or [])))' ) stages."
        ;;
    no_succeeded_stages)
        fail "pipeline reported succeeded but no stages succeeded"
        ;;
    no_artifact)
        printf '%s\n' "$final_body" | python3 -m json.tool >&2 || true
        fail "pipeline succeeded but no stage exposed an artifact / output / result field"
        ;;
    *)
        fail "unexpected artifact-assertion output: $ARTIFACT_OK"
        ;;
esac

log "OFFLINE SMOKE SUCCESS"
exit 0
