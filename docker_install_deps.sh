#!/bin/bash
# Run INSIDE the Docker container to install InSpatio-World deps
# The NVIDIA container already has: PyTorch, flash-attn 2, Triton, CUDA 13

set -e
cd /workspace/inspatio-world

echo "=== Installing InSpatio-World dependencies ==="

# Install Python deps (PyTorch already in container, skip it)
pip install --no-cache-dir \
    einops>=0.7.0 \
    omegaconf>=2.3.0 \
    safetensors>=0.7.0 \
    "transformers>=4.38.0,<5.0.0" \
    accelerate>=0.27.0 \
    "diffusers>=0.27.0" \
    timm>=0.9.0 \
    "numpy<2.0" \
    opencv-python>=4.8.0 \
    Pillow>=10.0.0 \
    tqdm>=4.65.0 \
    plyfile>=1.0 \
    easydict>=1.9 \
    imageio>=2.28.0 \
    ftfy>=6.0.0 \
    av>=12.0.0 \
    scipy>=1.10.0 \
    open3d>=0.17.0 \
    hf_transfer

# Install depth-anything-3
pip install --no-cache-dir depth_anything_3>=0.1.0

# Create mock decord module (InSpatio uses pyAV for actual video work)
echo "=== Creating mock decord module ==="
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p "${SITE_PACKAGES}/decord"
cat > "${SITE_PACKAGES}/decord/__init__.py" << 'MOCK'
"""Mock decord module for InSpatio-World on ARM64.
Real video loading uses pyAV. This satisfies import checks."""

class bridge:
    @staticmethod
    def set_bridge(name):
        pass

class VideoReader:
    """Minimal mock — raises clear error if actually called."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "decord.VideoReader is not available on ARM64. "
            "Use av (pyAV) for video loading instead."
        )

class AVVideoReader(VideoReader):
    pass
MOCK

echo "=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

try:
    import flash_attn
    print(f'Flash Attention: {flash_attn.__version__}')
except:
    print('Flash Attention: not available (will use SDPA fallback)')

import decord
print('decord mock: OK')

import diffusers, transformers, einops, av
print(f'diffusers: {diffusers.__version__}')
print(f'transformers: {transformers.__version__}')
print('All core deps: OK')
"

echo ""
echo "=== Dependencies installed! ==="
echo "Next: download models with: bash scripts/download.sh"
echo "Then run: bash run_test_pipeline.sh --input_dir ./test/example --traj_txt_path ./traj/x_y_circle_cycle.txt"
