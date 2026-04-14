#!/bin/bash
# InSpatio-World — Run Pipeline
# Handles llama-server stop/start automatically
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_NAME="inspatio-world"
INPUT_DIR="./inspatio-world/test/example"
TRAJ="orbit"
CONFIG="configs/inference_1.3b.yaml"
MASTER_PORT=29515
SKIP_STEPS=""
COMPILE_DIT=true
USE_TAE=true

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --input DIR       Input directory with .mp4 files (default: example)"
    echo "  --traj TRAJ       Trajectory: orbit | zoom (default: orbit)"
    echo "  --skip-step1      Skip Florence-2 captioning"
    echo "  --skip-step2      Skip DA3 depth + point cloud rendering"
    echo "  --no-restart      Don't stop/restart llama-servers
  --no-compile      Disable torch.compile (slower but no warmup)
  --no-tae          Use full VAE instead of TAE"
    echo "  -h, --help        Show this help"
    exit 0
}

NO_RESTART=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --input) INPUT_DIR="$2"; shift 2 ;;
        --traj)
            case $2 in
                orbit) TRAJ="orbit"; shift 2 ;;
                zoom) TRAJ="zoom"; shift 2 ;;
                *) echo "Unknown trajectory: $2 (use: orbit, zoom)"; exit 1 ;;
            esac ;;
        --skip-step1) SKIP_STEPS="$SKIP_STEPS --skip_step1"; shift ;;
        --skip-step2) SKIP_STEPS="$SKIP_STEPS --skip_step1 --skip_step2"; shift ;;
        --no-restart) NO_RESTART=true; shift ;;
        --no-compile) COMPILE_DIT=false; shift ;;
        --no-tae) USE_TAE=false; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Map trajectory name to file
case $TRAJ in
    orbit) TRAJ_FILE="./traj/x_y_circle_cycle.txt" ;;
    zoom)  TRAJ_FILE="./traj/zoom_out_in.txt" ;;
esac

# Resolve input path relative to inspatio-world dir
if [[ "$INPUT_DIR" == ./* ]]; then
    DOCKER_INPUT="$INPUT_DIR"
elif [[ "$INPUT_DIR" == /* ]]; then
    # Absolute path — must be under the mounted volume
    DOCKER_INPUT="/workspace/inspatio-world${INPUT_DIR#$SCRIPT_DIR/inspatio-world}"
else
    DOCKER_INPUT="./$INPUT_DIR"
fi

echo "============================================"
echo " InSpatio-World — DGX Spark"
echo "============================================"
echo " Input:      $INPUT_DIR"
echo " Trajectory: $TRAJ ($TRAJ_FILE)"
echo "============================================"
echo ""

# Ensure container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo ">>> Starting container..."
    docker start "$CONTAINER_NAME" 2>/dev/null || {
        echo "ERROR: Container '$CONTAINER_NAME' not found. Run setup.sh first."
        exit 1
    }
    sleep 2
fi

# Stop llama-servers to free GPU memory
LLAMA_PIDS=""
if [ "$NO_RESTART" = false ]; then
    LLAMA_PIDS=$(pgrep -f "llama-server" 2>/dev/null || true)
    if [ -n "$LLAMA_PIDS" ]; then
        echo ">>> Stopping llama-servers to free GPU memory..."
        pkill -f "llama-server" 2>/dev/null || true
        sleep 3
        # Force kill if still running
        pkill -9 -f "llama-server" 2>/dev/null || true
        sleep 2
        echo "  Freed $(free -g | awk '/Mem:/{print $4}')GB memory"
    fi
fi

# Run pipeline
echo ">>> Running InSpatio-World pipeline..."
echo ""

docker exec -w /workspace/inspatio-world "$CONTAINER_NAME" bash -c "
export TORCH_CUDA_ARCH_LIST='12.1a'
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
bash run_test_pipeline.sh \\
    --input_dir $DOCKER_INPUT \\
    --traj_txt_path $TRAJ_FILE \\
    --config_path $CONFIG \\
    --master_port $MASTER_PORT \\
    $SKIP_STEPS \
    $([ "$COMPILE_DIT" = true ] && echo '--compile_dit') \
    $([ "$USE_TAE" = true ] && echo '--use_tae')
"

EXIT_CODE=$?

# Restart llama-servers
if [ "$NO_RESTART" = false ] && [ -n "$LLAMA_PIDS" ]; then
    echo ""
    echo ">>> Restarting llama-servers..."
    systemctl --user start llama-main.service 2>/dev/null || true
    sleep 3
    echo "  llama-servers restored"
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================"
    echo " Done! Check output in:"
    echo " inspatio-world/output/"
    echo "============================================"
else
    echo ""
    echo "Pipeline failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
