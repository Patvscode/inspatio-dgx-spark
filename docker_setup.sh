#!/bin/bash
# InSpatio-World on DGX Spark — Docker setup
# Uses NVIDIA PyTorch container (pre-built flash-attn, Triton, CUDA 13)

set -e

WORK_DIR="$HOME/Desktop/AI-apps-workspace/inspatio-world"
CONTAINER_NAME="inspatio-world"
IMAGE="nvcr.io/nvidia/pytorch:25.09-py3"

echo "=== InSpatio-World Docker Setup ==="

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' already exists."
    echo "Starting it..."
    docker start -i "${CONTAINER_NAME}"
    exit 0
fi

echo "Creating new container..."
docker run \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=16g \
    -v "${WORK_DIR}:/workspace/inspatio-world" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e TORCH_CUDA_ARCH_LIST="12.1a" \
    -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -it "${IMAGE}" bash
