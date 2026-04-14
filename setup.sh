#!/bin/bash
# InSpatio-World DGX Spark Setup
# One-command setup: pulls container, clones repo, installs deps, downloads models
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/inspatio-world"
CONTAINER_NAME="inspatio-world"
IMAGE="nvcr.io/nvidia/pytorch:25.09-py3"

echo "============================================"
echo " InSpatio-World — DGX Spark Setup"
echo "============================================"
echo ""

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Install Docker + NVIDIA Container Toolkit first."
    exit 1
fi

if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "WARNING: NVIDIA runtime not detected in docker info. GPU access may not work."
    echo "Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Step 1: Clone InSpatio-World
if [ ! -d "$WORK_DIR" ]; then
    echo ">>> Cloning InSpatio-World..."
    git clone https://github.com/inspatio/inspatio-world.git "$WORK_DIR"
else
    echo ">>> InSpatio-World already cloned at $WORK_DIR"
fi

# Step 2: Apply patches
echo ">>> Applying ARM64 patches..."

# Patch render_point_cloud.py (open3d → plyfile)
if grep -q "import open3d" "$WORK_DIR/scripts/render_point_cloud.py" 2>/dev/null; then
    echo "  Patching render_point_cloud.py (open3d → plyfile)..."
    sed -i 's/^import open3d as o3d$/from plyfile import PlyData/' "$WORK_DIR/scripts/render_point_cloud.py"

    # Replace the load_ply_data function
    python3 -c "
import re
with open('$WORK_DIR/scripts/render_point_cloud.py', 'r') as f:
    content = f.read()

old_func = re.search(r'def load_ply_data\(ply_path, device\):.*?return torch\.from_numpy\(pts\)\.to\(device\), torch\.from_numpy\(colors\)\.to\(device\)', content, re.DOTALL)
if old_func:
    new_func = '''def load_ply_data(ply_path, device):
    \"\"\"Load point cloud from PLY file. Returns (points, colors) tensors.
    Uses plyfile instead of open3d for ARM64 compatibility.\"\"\"
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    if len(vertex) == 0:
        logger.warning(f\"Point cloud has no points: {ply_path}\")
        return None, None
    pts = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1).astype(np.float32)
    # Colors may be 0-255 uint8 or 0-1 float
    if 'red' in vertex:
        r, g, b = vertex['red'], vertex['green'], vertex['blue']
        colors = np.stack([r, g, b], axis=-1).astype(np.float32)
        if colors.max() > 1.0:
            colors /= 255.0
    else:
        colors = np.ones_like(pts) * 0.5  # default gray
    return torch.from_numpy(pts).to(device), torch.from_numpy(colors).to(device)'''
    content = content[:old_func.start()] + new_func + content[old_func.end():]
    with open('$WORK_DIR/scripts/render_point_cloud.py', 'w') as f:
        f.write(content)
    print('  render_point_cloud.py patched')
else:
    print('  load_ply_data not found or already patched')
"
else
    echo "  render_point_cloud.py already patched"
fi

# Step 3: Pull container
echo ">>> Pulling NVIDIA PyTorch container (~18GB)..."
docker pull "$IMAGE"

# Step 4: Create container
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo ">>> Container '$CONTAINER_NAME' already exists. Starting it..."
    docker start "$CONTAINER_NAME" 2>/dev/null || true
else
    echo ">>> Creating container..."
    docker run \
        --name "$CONTAINER_NAME" \
        --gpus all \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --shm-size=16g \
        -v "$WORK_DIR:/workspace/inspatio-world" \
        -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
        -e TORCH_CUDA_ARCH_LIST="12.1a" \
        -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
        -e HF_HUB_ENABLE_HF_TRANSFER=1 \
        -d "$IMAGE" \
        sleep infinity
fi

# Step 5: Install deps inside container
echo ">>> Installing dependencies inside container..."
docker exec "$CONTAINER_NAME" bash -c '
set -e
apt-get update -qq && apt-get install -y -qq git-lfs ffmpeg > /dev/null 2>&1
git lfs install > /dev/null 2>&1

pip install --no-cache-dir \
    einops omegaconf safetensors "transformers>=4.38.0,<5.0.0" \
    accelerate "diffusers>=0.27.0" timm "numpy<2.0" \
    opencv-python-headless Pillow tqdm plyfile easydict \
    imageio ftfy av scipy hf_transfer \
    "moviepy==1.0.3" addict evo 2>&1 | tail -3

pip install --no-cache-dir --no-deps depth_anything_3 2>&1 | tail -1
'

# Step 6: Install decord mock
echo ">>> Installing decord mock (pyAV-based)..."
docker exec "$CONTAINER_NAME" bash -c '
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p "${SITE}/decord"
cat > "${SITE}/decord/__init__.py" << '\''MOCK'\''
"""Functional decord mock using pyAV for ARM64 compatibility."""
import av
import torch
import numpy as np

class bridge:
    @staticmethod
    def set_bridge(name):
        pass

class VideoReader:
    """Drop-in replacement for decord.VideoReader using pyAV."""
    def __init__(self, uri, height=-1, width=-1, **kwargs):
        self.uri = uri
        self.frames = []
        container = av.open(str(uri))
        for frame in container.decode(video=0):
            self.frames.append(frame.to_ndarray(format="rgb24"))
        container.close()
        self._len = len(self.frames)

    def __len__(self):
        return self._len

    def get_batch(self, indices):
        return torch.from_numpy(np.stack([self.frames[i] for i in indices]))

class AVVideoReader(VideoReader):
    pass
MOCK
'

# Step 7: Patch depth_anything_3 export chain
echo ">>> Patching depth_anything_3 lazy imports..."
docker exec "$CONTAINER_NAME" bash -c '
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
cat > "${SITE}/depth_anything_3/utils/export/__init__.py" << '\''PATCH'\''
# Patched for ARM64/DGX Spark — lazy imports for unavailable deps (pycolmap, gsplat)
from depth_anything_3.specs import Prediction

try:
    from .depth_vis import export_to_depth_vis
except ImportError:
    export_to_depth_vis = None

try:
    from .feat_vis import export_to_feat_vis
except ImportError:
    export_to_feat_vis = None

try:
    from .npz import export_to_mini_npz, export_to_npz
except ImportError:
    export_to_mini_npz = None
    export_to_npz = None


def _lazy_import(name):
    try:
        if name == "gs":
            from depth_anything_3.utils.export.gs import export_to_gs_ply, export_to_gs_video
            return export_to_gs_ply, export_to_gs_video
        elif name == "colmap":
            from .colmap import export_to_colmap
            return export_to_colmap
        elif name == "glb":
            from .glb import export_to_glb
            return export_to_glb
    except ImportError:
        return None


def export(prediction: Prediction, export_format: str, export_dir: str, **kwargs):
    if "-" in export_format:
        for fmt in export_format.split("-"):
            export(prediction, fmt, export_dir, **kwargs)
        return

    if export_format == "depth_vis" and export_to_depth_vis:
        export_to_depth_vis(prediction, export_dir)
    elif export_format == "npz" and export_to_npz:
        export_to_npz(prediction, export_dir)
    elif export_format == "mini_npz" and export_to_mini_npz:
        export_to_mini_npz(prediction, export_dir)
    elif export_format == "feat_vis" and export_to_feat_vis:
        export_to_feat_vis(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format in ("gs_ply", "gs_video", "colmap", "glb"):
        result = _lazy_import(export_format.split("_")[0] if "gs" in export_format else export_format)
        if result is None:
            raise ImportError(f"Export format {export_format} requires deps unavailable on ARM64")
    else:
        raise ValueError(f"Unsupported export format: {export_format}")


__all__ = [export]
PATCH
'

# Step 8: Download models
echo ">>> Downloading model checkpoints (~69GB)..."
docker exec -w /workspace/inspatio-world "$CONTAINER_NAME" bash -c '
set -e
mkdir -p checkpoints && cd checkpoints

echo "  [1/5] InSpatio-World-1.3B..."
if [ ! -f "InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors" ] || [ $(stat -c%s "InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors" 2>/dev/null || echo 0) -lt 1000000 ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/inspatio/world 2>/dev/null || true
    cd world && git lfs pull && cd ..
    mkdir -p InSpatio-World-1.3B
    cp world/InSpatio-World-1.3B.safetensors InSpatio-World-1.3B/
fi
echo "  InSpatio-World-1.3B: $(du -sh InSpatio-World-1.3B/ | cut -f1)"

echo "  [2/5] Wan2.1-T2V-1.3B..."
if [ ! -d "Wan2.1-T2V-1.3B/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B 2>/dev/null || true
    cd Wan2.1-T2V-1.3B && git lfs pull && cd ..
fi
echo "  Wan2.1-T2V-1.3B: $(du -sh Wan2.1-T2V-1.3B/ | cut -f1)"

echo "  [3/5] DA3 (Depth-Anything-3)..."
if [ ! -d "DA3/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE DA3 2>/dev/null || true
    cd DA3 && git lfs pull && cd ..
fi
echo "  DA3: $(du -sh DA3/ | cut -f1)"

echo "  [4/5] Florence-2-large..."
if [ ! -d "Florence-2-large/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/microsoft/Florence-2-large 2>/dev/null || true
    cd Florence-2-large && git lfs pull && cd ..
fi
echo "  Florence-2-large: $(du -sh Florence-2-large/ | cut -f1)"

echo "  [5/5] TAEHV (optional speed-up)..."
if [ ! -d "taehv/.git" ]; then
    git clone https://github.com/madebyollin/taehv.git 2>/dev/null || true
fi
echo "  TAEHV: $(du -sh taehv/ | cut -f1)"
'

# Step 9: Verify
echo ""
echo ">>> Verifying installation..."
docker exec "$CONTAINER_NAME" python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')
import decord; print('decord mock: OK')
import diffusers, transformers, depth_anything_3
print(f'diffusers: {diffusers.__version__}, transformers: {transformers.__version__}')
from depth_anything_3.api import DepthAnything3; print('DA3 import chain: OK')
print()
print('All dependencies verified ✓')
" 2>&1 | grep -v FutureWarning | grep -v pynvml

echo ""
echo "============================================"
echo " Setup complete!"
echo " Run: bash run.sh"
echo "============================================"
