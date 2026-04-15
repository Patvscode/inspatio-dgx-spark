#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IO_DIR="$ROOT_DIR/interactive_io"
PID_FILE="$IO_DIR/dit_stream.pid"
LOG_FILE="$IO_DIR/dit_stream.log"
DRY_RUN=0
SCENE=""
QUALITY=""
STEPS=""

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

docker start inspatio-world >/dev/null 2>&1 || true

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(tr -d '[:space:]' < "$PID_FILE" || true)"
  if [[ -n "$EXISTING_PID" ]]; then
    if docker exec inspatio-world bash -lc "kill -0 '$EXISTING_PID' 2>/dev/null && cmd=\$(tr '\\0' ' ' < /proc/'$EXISTING_PID'/cmdline 2>/dev/null || true); case \"\$cmd\" in *dit_stream.py*) exit 0 ;; *) exit 1 ;; esac"; then
      echo "dit_stream already running with pid $EXISTING_PID" >&2
      exit 0
    fi
  fi
  rm -f "$PID_FILE"
fi

LAUNCH_CMD="cd /workspace/inspatio-world && rm -f /workspace/inspatio-world/interactive_io/dit_stream.pid && INSPATIO_USE_TORCH_COMPILE=0 TORCH_CUDA_ARCH_LIST=12.1a TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache nohup python3 dit_stream.py > /workspace/inspatio-world/interactive_io/dit_stream.log 2>&1 & echo \$! > /workspace/inspatio-world/interactive_io/dit_stream.pid"

if [[ "$DRY_RUN" -eq 1 ]]; then
  python3 - <<'PY' "$SCENE" "$QUALITY" "$STEPS" "$LAUNCH_CMD"
import json, sys
scene, quality, steps, command = sys.argv[1:5]
print(json.dumps({
    "scene": scene,
    "quality": quality,
    "steps": steps,
    "command": command,
}, indent=2))
PY
  exit 0
fi

docker exec inspatio-world bash -lc "$LAUNCH_CMD"

NEW_PID=""
if [[ -f "$PID_FILE" ]]; then
  NEW_PID="$(tr -d '[:space:]' < "$PID_FILE" || true)"
fi

if [[ -z "$NEW_PID" ]]; then
  echo "heavy stream launch did not produce a pid file" >&2
  exit 1
fi

printf 'launched dit_stream pid=%s scene=%s quality=%s steps=%s log=%s\n' "$NEW_PID" "$SCENE" "$QUALITY" "$STEPS" "$LOG_FILE"
