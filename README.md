# InSpatio-World on DGX Spark (ARM64 + Blackwell)

Running [InSpatio-World](https://github.com/inspatio/inspatio-world) — a real-time 4D world simulator — on the NVIDIA DGX Spark (GB10 Grace Blackwell, aarch64, CUDA 13.0, 128GB unified memory).

InSpatio-World takes a regular video and generates novel-view videos from camera trajectories you never filmed. Think: give it a street scene, get back a cinematic orbit shot.

## The Problem

InSpatio-World is designed for x86_64 + standard NVIDIA GPUs. The DGX Spark's combination of **ARM64 CPU + Blackwell GPU + CUDA 13.0** breaks several dependencies:

- `flash-attn` — no aarch64 prebuilt wheels, source build fails (CUDA version mismatch)
- `open3d` — no Python 3.12 aarch64 wheel on PyPI
- `decord` — no ARM64 wheel, C++ source build needs native libs
- `xformers` — x86_64 only
- `pycolmap` / `gsplat` — no aarch64 builds
- Default `pip install torch` on aarch64 gives CPU-only PyTorch

## The Solution

Use NVIDIA's PyTorch container (`nvcr.io/nvidia/pytorch:25.09-py3`) which ships with Blackwell-optimized PyTorch, Flash Attention 2, and Triton pre-compiled for aarch64 + GB10. Then patch around the remaining ARM64-incompatible dependencies.

### What's in this repo

```
├── README.md                 # This file
├── setup.sh                  # One-command setup (pull container, install deps, download models)
├── run.sh                    # One-command run (handles server stop/start + pipeline)
├── patches/
│   ├── decord_mock.py        # Functional decord replacement using pyAV
│   ├── render_point_cloud.patch  # open3d → plyfile replacement
│   └── da3_export_lazy.py    # Lazy imports for depth_anything_3 export chain
├── docker/
│   └── install_deps.sh       # Dependency installation inside container
└── docs/
    └── SETUP_NOTES.md        # Detailed walkthrough of every issue and fix
```

## Quick Start

### Prerequisites
- NVIDIA DGX Spark (or any aarch64 + Blackwell GPU system)
- Docker with NVIDIA Container Toolkit (`nvidia-docker`)
- ~80GB free disk space (container + models)
- git-lfs installed on host

### Setup (one time, ~30 min depending on download speed)

```bash
git clone https://github.com/Patvscode/inspatio-dgx-spark.git
cd inspatio-dgx-spark
bash setup.sh
```

This will:
1. Pull the NVIDIA PyTorch container (~18GB)
2. Clone InSpatio-World and install all dependencies
3. Apply ARM64 patches automatically
4. Download all model checkpoints (~69GB)

### Run

```bash
# Run on the included example video
bash run.sh

# Run on your own video
bash run.sh --input ./my_videos --traj orbit

# Available trajectories: orbit, zoom
```

> **Note:** The run script automatically stops any running llama-server processes to free GPU memory, and restarts them after inference completes. InSpatio-World needs ~97GB VRAM for 301-frame inference.

> **Performance:** `torch.compile` + TAE are enabled by default. First run includes a ~5 min one-time kernel compilation warmup (cached for subsequent runs). Use `--no-compile` to skip compilation if you want faster startup at the cost of slower inference.

## Performance on DGX Spark

| Step | Task | Time | Peak Memory |
|------|------|------|-------------|
| 1 | Florence-2 captioning | <1s | ~4GB |
| 2a | DA3 depth estimation | 75s | 25.5GB |
| 2b | Point cloud rendering | 14s | ~2GB |
| 3 | v2v inference (301 frames) | ~2 min (compiled) | ~97GB |
| 3 | v2v inference (first run, includes compilation) | ~7 min | ~97GB |

## Patches Explained

### 1. decord → pyAV (`patches/decord_mock.py`)

decord has no ARM64 wheel. InSpatio uses `decord.VideoReader` in its data loading pipeline (`datasets/utils.py`). We replace it with a drop-in mock that uses pyAV (which is pure Python + FFmpeg, works everywhere):

```python
class VideoReader:
    def __init__(self, uri, height=-1, width=-1, **kwargs):
        container = av.open(str(uri))
        self.frames = [frame.to_ndarray(format="rgb24") 
                       for frame in container.decode(video=0)]
        container.close()

    def get_batch(self, indices):
        return torch.from_numpy(np.stack([self.frames[i] for i in indices]))
```

### 2. open3d → plyfile (`patches/render_point_cloud.patch`)

open3d has no Python 3.12 aarch64 wheel. The point cloud renderer (`scripts/render_point_cloud.py`) only uses `o3d.io.read_point_cloud()` to load PLY files. We replace it with plyfile:

```python
from plyfile import PlyData

def load_ply_data(ply_path, device):
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    pts = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1).astype(np.float32)
    colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=-1).astype(np.float32)
    if colors.max() > 1.0:
        colors /= 255.0
    return torch.from_numpy(pts).to(device), torch.from_numpy(colors).to(device)
```

### 3. depth_anything_3 export chain (`patches/da3_export_lazy.py`)

The `depth_anything_3` package eagerly imports all export modules at package load time, pulling in `pycolmap` and `gsplat` which have no ARM64 builds. InSpatio only uses the depth estimation, not the Gaussian splatting export. We replace `depth_anything_3/utils/export/__init__.py` with lazy imports that skip unavailable modules.

### 4. Additional missing deps

The `depth_anything_3` pip package declares `xformers` as a dependency but doesn't actually need it for inference (Flash Attention / SDPA handles attention). We install with `--no-deps` and add the real requirements manually. Also needed: `moviepy==1.0.3`, `addict`, `evo`, `ffmpeg`.

## Container Details

The NVIDIA PyTorch container (`nvcr.io/nvidia/pytorch:25.09-py3`) provides:

| Component | Version |
|-----------|---------|
| PyTorch | 2.9.0 (Blackwell-optimized) |
| CUDA | 13.0 |
| Flash Attention | 2.7.4 |
| Triton | 3.4.0 |
| Python | 3.12.3 |

This container is the key — it eliminates the need to build flash-attn, triton, or PyTorch from source on ARM64.

## Architecture

InSpatio-World's inference pipeline:

```
Input Video (.mp4)
    │
    ▼
[Step 1] Florence-2 → Video captions (JSON)
    │
    ▼
[Step 2a] DA3 → Per-frame depth maps + camera poses
    │
    ▼
[Step 2b] Point cloud renderer → Rough novel-view renders + masks
    │
    ▼
[Step 3] InSpatio-World 1.3B (DiT) → High-quality novel-view video
    │
    ▼
Output Video (.mp4)
```

## Credits

- [InSpatio-World](https://github.com/inspatio/inspatio-world) — the original project (Apache-2.0)
- [NVIDIA PyTorch Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) — pre-built ARM64 + Blackwell environment
- [natolambert/dgx-spark-setup](https://github.com/natolambert/dgx-spark-setup) — DGX Spark ML setup reference
- [RobG-git's DGX Spark gist](https://gist.github.com/RobG-git/fd1739c79e2405eb56032dee7901421e) — container approach inspiration

## License

The patches and scripts in this repo are MIT licensed. InSpatio-World itself is Apache-2.0. Model weights have their own licenses — check the respective HuggingFace pages.
