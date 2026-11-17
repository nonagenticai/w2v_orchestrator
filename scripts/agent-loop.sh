#!/usr/bin/env bash
set -euo pipefail

# web_to_video agent orchestration loop
# Runs Claude Code agents in parallel for script generation

TOOL="${TOOL:-claude}"
PARALLEL="${PARALLEL:-1}"
PROMPT_DIR="${PROMPT_DIR:-output/prompts}"
OUTPUT_DIR="${OUTPUT_DIR:-output/scripts/parts}"
TIMEOUT="${TIMEOUT:-300}"
MAX_RETRIES="${MAX_RETRIES:-2}"
ON_THROTTLE="${ON_THROTTLE:-wait}"  # wait | stop | retry
STOP_REQUESTED=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tool) TOOL="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --prompt-dir) PROMPT_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --max-retries) MAX_RETRIES="$2"; shift 2 ;;
    --on-throttle) ON_THROTTLE="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

is_throttled() {
  echo "$1" | grep -iE "rate limit|rate-limited|throttl|429|too many requests" >/dev/null 2>&1
}

run_with_timeout() {
  local timeout_cmd=""
  if command -v timeout >/dev/null 2>&1; then
    timeout_cmd="timeout"
  elif command -v gtimeout >/dev/null 2>&1; then
    timeout_cmd="gtimeout"
  fi

  if [[ -n "$timeout_cmd" && "$TIMEOUT" -gt 0 ]]; then
    "$timeout_cmd" "$TIMEOUT" "$@"
    return $?
  fi
  "$@"
}

run_tool_prompt() {
  local prompt_file="$1"
  local output=""
  if [[ "$TOOL" == "claude" ]]; then
    output=$(run_with_timeout claude --dangerously-skip-permissions --print < "$prompt_file" 2>&1) || true
  elif [[ "$TOOL" == "amp" ]]; then
    output=$(cat "$prompt_file" | run_with_timeout amp --dangerously-allow-all 2>&1) || true
  elif [[ "$TOOL" == "gemini" ]]; then
    output=$(run_with_timeout gemini --yolo < "$prompt_file" 2>&1) || true
  elif [[ "$TOOL" == "codex" ]]; then
    output=$(run_with_timeout codex exec --dangerously-bypass-approvals-and-sandbox - < "$prompt_file" 2>&1) || true
  else
    echo "Unknown tool: $TOOL" >&2
    exit 1
  fi
  echo "$output"
}

process_prompt() {
  local prompt_file="$1"
  local part_id
  part_id=$(basename "$prompt_file" .md | sed 's/^\.prompt-//')
  local output_file="${OUTPUT_DIR}/${part_id}.demo.spec.ts"
  local lock_file="${OUTPUT_DIR}/.lock-${part_id}"

  # File-based locking
  if [[ -f "$lock_file" ]]; then
    echo "Skipping $part_id (locked by another worker)"
    return 0
  fi
  echo "$$" > "$lock_file"

  local attempt=0
  local success=false

  while [[ $attempt -lt $((MAX_RETRIES + 1)) ]]; do
    attempt=$((attempt + 1))
    echo "[$part_id] Attempt $attempt (tool=$TOOL)"

    OUTPUT=$(run_tool_prompt "$prompt_file")

    # Check for throttling
    if is_throttled "$OUTPUT"; then
      echo "[$part_id] Throttle detected"
      if [[ "$ON_THROTTLE" == "stop" ]]; then
        rm -f "$lock_file"
        return 1
      elif [[ "$ON_THROTTLE" == "wait" ]]; then
        echo "[$part_id] Waiting 60s before retry..."
        sleep 60
        continue
      else
        sleep 10
        continue
      fi
    fi

    # Check for completion token
    if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
      # Extract TypeScript code
      SCRIPT=$(echo "$OUTPUT" | sed -n '/```typescript/,/```/p' | sed '1d;$d')
      if [[ -n "$SCRIPT" ]]; then
        echo "$SCRIPT" > "$output_file"
        echo "[$part_id] Generated successfully"
        success=true
        break
      else
        echo "[$part_id] Completion token found but no TypeScript block"
      fi
    else
      echo "[$part_id] Missing completion token"
    fi

    if [[ $attempt -lt $((MAX_RETRIES + 1)) ]]; then
      echo "[$part_id] Retrying..."
      sleep $((RANDOM % 3 + 1))
    fi
  done

  rm -f "$lock_file"

  if [[ "$success" != "true" ]]; then
    echo "[$part_id] Failed after $attempt attempts"
    return 1
  fi
}

run_worker() {
  local worker_id="$1"
  local jitter=$((RANDOM % 3))
  sleep "$jitter"

  for prompt_file in "$PROMPT_DIR"/.prompt-*.md; do
    [[ -f "$prompt_file" ]] || continue
    [[ "$STOP_REQUESTED" == "true" ]] && return 0
    process_prompt "$prompt_file" || true
  done
}

# Main execution
if [[ "$PARALLEL" -le 1 ]]; then
  run_worker 1
else
  pids=()
  cleanup_all() {
    STOP_REQUESTED=true
    for pid in "${pids[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
    pkill -P $$ 2>/dev/null || true
  }
  trap cleanup_all INT TERM

  for w in $(seq 1 "$PARALLEL"); do
    run_worker "$w" &
    pids+=($!)
  done

  exit_code=0
  for pid in "${pids[@]}"; do
    wait "$pid" || exit_code=1
  done
  exit "$exit_code"
fi

echo "Agent loop complete."
