#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IO_DIR="$ROOT_DIR/interactive_io"
PID_FILE="$IO_DIR/dit_stream.pid"
LOG_FILE="$IO_DIR/dit_stream.log"
STATE_FILE="$IO_DIR/heavy_launch_state.json"
DRY_RUN=0
SCENE=""
QUALITY=""
STEPS=""
GATE_GEMMA=1
LAUNCH_ID="$(date +%s)-$$"

json_emit() {
  local status="$1"
  local reason="${2:-}"
  local gated="${3:-false}"
  local pid="${4:-}"
  python3 - "$STATE_FILE" "$status" "$reason" "$gated" "$pid" "$SCENE" "$QUALITY" "$STEPS" "$LAUNCH_ID" <<'PY'
import json, os, sys, time
state_file, status, reason, gated, pid, scene, quality, steps, launch_id = sys.argv[1:10]
data = {
    "status": status,
    "reason": reason,
    "gated_gemma": gated.lower() == "true",
    "pid": pid or None,
    "scene": scene,
    "quality": quality,
    "steps": int(steps) if str(steps).isdigit() else steps,
    "launch_id": launch_id,
    "timestamp": time.time(),
}
os.makedirs(os.path.dirname(state_file), exist_ok=True)
with open(state_file, 'w') as f:
    json.dump(data, f, indent=2)
print(json.dumps(data))
PY
}

deny() {
  json_emit "denied" "$1" "${2:-false}" ""
  exit 3
}

detect_log_reason() {
  local log_file="${1:-$LOG_FILE}"
  if [[ ! -f "$log_file" ]]; then
    return 1
  fi

  if tail -n 120 "$log_file" 2>/dev/null | grep -qiE 'CUDA error: out of memory|cudaErrorMemoryAllocation|torch\.OutOfMemoryError'; then
    printf '%s' 'cuda_oom'
    return 0
  fi

  return 1
}

crash() {
  local reason="$1"
  local gated="${2:-false}"
  local pid="${3:-}"
  local detected_reason=""
  detected_reason="$(detect_log_reason || true)"
  if [[ -n "$detected_reason" ]] && [[ "$reason" == "heavy_stream_pid_not_alive_after_launch" || "$reason" == "worker_exited" ]]; then
    reason="${reason}:${detected_reason}"
  fi
  if [[ "${GEMMA_GATED:-0}" -eq 1 ]]; then
    systemctl --user start gemma-e2b.service >/dev/null 2>&1 || true
  fi
  json_emit "crashed" "$reason" "$gated" "$pid"
  exit 1
}

service_active() {
  [[ "$(systemctl --user is-active "$1" 2>/dev/null || true)" == "active" ]]
}

port_listening() {
  local port="$1"
  ss -ltn "( sport = :$port )" | grep -q ":$port"
}

container_pid_alive() {
  local pid="$1"
  docker exec inspatio-world bash -lc "kill -0 '$pid' 2>/dev/null && cmd=\$(tr '\\0' ' ' < /proc/'$pid'/cmdline 2>/dev/null || true); case \"\$cmd\" in *dit_stream.py*) exit 0 ;; *) exit 1 ;; esac" >/dev/null 2>&1
}

start_restore_watcher() {
  local pid="$1"
  local gated="$2"
  local launch_id="$3"
  local log_file="$4"
  nohup bash -lc '
    ROOT_DIR="$1"
    STATE_FILE="$2"
    PID="$3"
    GATED="$4"
    LAUNCH_ID="$5"
    LOG_FILE="$6"
    STATUS_FILE="$ROOT_DIR/interactive_io/status.json"
    while docker exec inspatio-world bash -lc "kill -0 \"$PID\" 2>/dev/null && cmd=\\$(tr '\''\\0'\'' '\'' '\'' < /proc/\"$PID\"/cmdline 2>/dev/null || true); case \"\\$cmd\" in *dit_stream.py*) exit 0 ;; *) exit 1 ;; esac" >/dev/null 2>&1; do
      sleep 2
    done
    if [[ "$GATED" == "1" ]]; then
      systemctl --user start gemma-e2b.service >/dev/null 2>&1 || true
    fi
    python3 - "$STATE_FILE" "$STATUS_FILE" "$LAUNCH_ID" "$GATED" "$LOG_FILE" <<'"'"'PY'"'"'
import json, os, sys, time
state_file, status_file, launch_id, gated, log_file = sys.argv[1:6]

def detect_reason(path):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 16384))
            tail = f.read().decode('utf-8', errors='ignore')
    except Exception:
        return None
    lowered = tail.lower()
    if 'cuda error: out of memory' in lowered or 'cudaerrormemoryallocation' in lowered or 'torch.outofmemoryerror' in lowered:
        return 'worker_exited:cuda_oom'
    return None

try:
    with open(state_file, 'r') as f:
        data = json.load(f)
except Exception:
    raise SystemExit(0)
if data.get('launch_id') != launch_id:
    raise SystemExit(0)
status = 'crashed'
reason = detect_reason(log_file) or 'worker_exited'
try:
    with open(status_file, 'r') as f:
        raw = json.load(f)
    current = raw.get('status')
    if current in {'stopped', 'ended'}:
        status = 'stopped'
        reason = 'worker_exited_after_cleanup'
    elif current == 'crashed':
        status = 'crashed'
        reason = raw.get('error') or detect_reason(log_file) or 'worker_crashed'
except Exception:
    pass
data.update({
    'status': status,
    'reason': reason,
    'gated_gemma': gated == '1',
    'restored_gemma': gated == '1',
    'timestamp': time.time(),
})
with open(state_file, 'w') as f:
    json.dump(data, f, indent=2)
PY
  ' _ "$ROOT_DIR" "$STATE_FILE" "$pid" "$gated" "$launch_id" "$log_file" >/dev/null 2>&1 &
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)
      SCENE="${2:-}"
      shift 2
      ;;
    --quality)
      QUALITY="${2:-}"
      shift 2
      ;;
    --steps)
      STEPS="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

[[ -n "$SCENE" ]] || deny "missing_scene_argument"
[[ -n "$QUALITY" ]] || deny "missing_quality_argument"
[[ -n "$STEPS" ]] || deny "missing_steps_argument"

LAUNCH_CMD="cd /workspace/inspatio-world && rm -f /workspace/inspatio-world/interactive_io/dit_stream.pid && INSPATIO_USE_TORCH_COMPILE=0 TORCH_CUDA_ARCH_LIST=12.1a TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache nohup python3 dit_stream.py > /workspace/inspatio-world/interactive_io/dit_stream.log 2>&1 & echo \$! > /workspace/inspatio-world/interactive_io/dit_stream.pid"

service_active llama-main.service || deny "llama_main_service_inactive"
port_listening 18080 || deny "llama_main_port_18080_not_listening"
service_active openclaw-gateway.service || deny "openclaw_gateway_service_inactive"
port_listening 18789 || deny "openclaw_gateway_port_18789_not_listening"
command -v docker >/dev/null 2>&1 || deny "docker_unavailable"

docker start inspatio-world >/dev/null 2>&1 || deny "inspatio_container_unavailable"

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(tr -d '[:space:]' < "$PID_FILE" || true)"
  if [[ -n "$EXISTING_PID" ]] && container_pid_alive "$EXISTING_PID"; then
    deny "heavy_worker_already_active"
  fi
  rm -f "$PID_FILE"
fi

if docker exec inspatio-world bash -lc "pgrep -af '[d]it_stream.py' >/dev/null 2>&1"; then
  deny "heavy_worker_already_active"
fi

GEMMA_GATED=0
if [[ "$DRY_RUN" -eq 1 ]]; then
  if [[ "$GATE_GEMMA" -eq 1 ]] && service_active gemma-e2b.service; then
    json_emit "degraded" "dry_run_preflight_ok_gemma_would_be_gated" true ""
  else
    json_emit "launching" "dry_run_preflight_ok" false ""
  fi
  exit 0
fi

if [[ "$GATE_GEMMA" -eq 1 ]] && service_active gemma-e2b.service; then
  systemctl --user stop gemma-e2b.service >/dev/null 2>&1 || deny "failed_to_gate_gemma" true
  sleep 2
  if service_active gemma-e2b.service || port_listening 18081; then
    deny "gemma_lane_did_not_stop_cleanly" true
  fi
  GEMMA_GATED=1
fi

json_emit "launching" "preflight_ok" "$([[ "$GEMMA_GATED" -eq 1 ]] && echo true || echo false)" "" >/dev/null

if ! docker exec inspatio-world bash -lc "$LAUNCH_CMD" >/dev/null 2>&1; then
  crash "docker_exec_launch_failed" "$([[ "$GEMMA_GATED" -eq 1 ]] && echo true || echo false)"
fi

NEW_PID=""
if [[ -f "$PID_FILE" ]]; then
  NEW_PID="$(tr -d '[:space:]' < "$PID_FILE" || true)"
fi

if [[ -z "$NEW_PID" ]]; then
  crash "heavy_stream_launch_did_not_produce_pid" "$([[ "$GEMMA_GATED" -eq 1 ]] && echo true || echo false)"
fi

if ! container_pid_alive "$NEW_PID"; then
  crash "heavy_stream_pid_not_alive_after_launch" "$([[ "$GEMMA_GATED" -eq 1 ]] && echo true || echo false)" "$NEW_PID"
fi

start_restore_watcher "$NEW_PID" "$GEMMA_GATED" "$LAUNCH_ID" "$LOG_FILE"

if [[ "$GEMMA_GATED" -eq 1 ]]; then
  json_emit "degraded" "gemma_lane_temporarily_gated" true "$NEW_PID"
else
  json_emit "running" "launch_ok" false "$NEW_PID"
fi
