#!/bin/bash
# Launch InSpatio-World Interactive Streaming Viewer
# This starts both the DiT server (in Docker) and the viewer (on host)

set -e

echo "============================================"
echo "  InSpatio-World Interactive Launcher"
echo "============================================"

# Clean old frames
rm -rf ~/Desktop/AI-apps-workspace/inspatio-world/interactive_io/frames/*
echo '{"yaw":0,"pitch":0,"zoom":1.0}' > ~/Desktop/AI-apps-workspace/inspatio-world/interactive_io/pose.json

# Kill any existing viewers
kill $(lsof -ti:7861) 2>/dev/null || true
sleep 1

# Make sure container is running
docker start inspatio-world 2>/dev/null || true
sleep 1

# Stop llama-servers to free GPU
echo "Freeing GPU..."
systemctl --user stop llama-main.service 2>/dev/null || true
pkill -f llama-server 2>/dev/null || true
sleep 2

# Start DiT streaming server in Docker (background)
echo "Starting DiT streaming server..."
docker exec inspatio-world bash -c "cd /workspace/inspatio-world && \
    TORCH_CUDA_ARCH_LIST=12.1a \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
    python3 dit_stream.py" &
DIT_PID=$!

# Wait a moment then start viewer
echo "Starting viewer..."
sleep 3
python3 ~/Desktop/AI-apps-workspace/inspatio-world/interactive_stream.py &
VIEWER_PID=$!

echo ""
echo "  Viewer: http://100.109.173.109:7861"
echo "  DiT server warming up (1-2 min for first run)..."
echo "  Press Ctrl+C to stop everything"
echo ""

# Wait for either to exit
trap "kill $DIT_PID $VIEWER_PID 2>/dev/null; systemctl --user start llama-main.service; echo 'Cleaned up.'" EXIT
wait
