# Setup Notes — InSpatio-World on DGX Spark

Detailed walkthrough of every issue encountered and how it was resolved.

## System Specs

- **Hardware:** NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)
- **CPU:** ARM64 (aarch64), Grace
- **GPU:** Blackwell (sm_121a), compute capability 12.1
- **Memory:** 128GB unified (shared CPU+GPU)
- **CUDA:** 13.0
- **OS:** Ubuntu 24.04 LTS

## Why Not Native Install?

We initially planned a native conda environment with cu130 PyTorch and SDPA fallback (no flash-attn). Here's why we switched to Docker:

1. **PyTorch cu130:** Available but untested with InSpatio's dependency tree
2. **flash-attn:** Would need to be skipped entirely (SDPA fallback exists but is slower)
3. **decord, open3d, xformers:** All needed workarounds regardless
4. **The NVIDIA container already solved all the hard parts** — Blackwell-optimized PyTorch, Flash Attention 2, Triton, all pre-compiled for aarch64

The Docker approach took ~30 min vs an estimated 2+ hours for native, with higher confidence of success.

## Issue-by-Issue Breakdown

### Issue 1: PyTorch CPU-only on aarch64

**Problem:** `pip install torch` on aarch64 installs CPU-only PyTorch by default. The official wheels with CUDA support require specifying `--index-url https://download.pytorch.org/whl/cu128` (or cu130).

**Solution:** Used NVIDIA's container which has a custom Blackwell-optimized PyTorch build (2.9.0).

### Issue 2: flash-attn no aarch64 wheel

**Problem:** InSpatio's `requirements.txt` references flash-attn, and the official install URL points to an x86_64 `.whl`. No aarch64 prebuilt exists, and building from source fails due to CUDA 13.0 vs PyTorch's expected CUDA version.

**Solution:** The NVIDIA container ships flash-attn 2.7.4 pre-compiled. InSpatio's attention code (`wan/modules/attention.py`) has a 3-tier fallback: SageAttn → Flash Attn 2/3 → `torch.nn.functional.scaled_dot_product_attention`. With flash-attn available in the container, it uses the optimal path.

### Issue 3: open3d no Python 3.12 aarch64 wheel

**Problem:** `pip install open3d` returns "No matching distribution found" on Python 3.12 aarch64. Open3D has experimental ARM64 support but wheels aren't available for all Python versions.

**Solution:** The only open3d usage is in `scripts/render_point_cloud.py` — specifically `o3d.io.read_point_cloud()` to load PLY files. Replaced with `plyfile` (pure Python, works everywhere). The replacement reads vertex positions and RGB colors from the PLY binary format directly.

### Issue 4: decord no ARM64 wheel

**Problem:** decord (video reader library) has no ARM64 wheel and building from source requires native C++ dependencies.

**Solution:** Created a functional drop-in mock using pyAV. The mock implements `VideoReader.__init__()`, `__len__()`, and `get_batch()` — the three methods InSpatio actually calls. pyAV uses FFmpeg under the hood and works on all platforms.

**Important:** A dummy mock (raising NotImplementedError) is NOT sufficient. InSpatio's inference pipeline (`datasets/utils.py` → `read_frames()`) calls `decord.VideoReader()` directly during Step 3. The mock must actually read video frames.

### Issue 5: depth_anything_3 eager imports

**Problem:** The `depth_anything_3` package eagerly imports its entire export chain at package load time:
```
depth_anything_3.api
  → depth_anything_3.utils.export
    → .gs → gsplat (no ARM64 build)
    → .colmap → pycolmap (no ARM64 build)
```

InSpatio only uses `DepthAnything3.from_pretrained()` for depth estimation — none of the export functionality.

**Solution:** Replaced `depth_anything_3/utils/export/__init__.py` with lazy imports that silently skip unavailable modules. Core exports (depth_vis, npz) still work; 3D reconstruction exports (gs_ply, colmap) raise clear errors if attempted.

### Issue 6: Missing transitive dependencies

**Problem:** Several packages needed by depth_anything_3 aren't declared in its pip metadata or are version-pinned incorrectly:
- `moviepy` — needed but not installed
- `moviepy==1.0.3` — required specifically (2.x is incompatible with the import style used)
- `addict` — needed by DA3's model code
- `evo` — needed by DA3's pose alignment code
- `ffmpeg` — needed for point cloud rendering (video encoding)

**Solution:** Installed each manually. The `depth_anything_3` package was installed with `--no-deps` to avoid pulling in `xformers` (x86 only, not actually needed).

### Issue 7: torchrun port conflict

**Problem:** Step 3 uses `torchrun` for distributed inference, which binds to a default master port (29513). If another process already uses that port, it fails with `EADDRINUSE`.

**Solution:** Pass `--master_port 29515` (or any free port) to the pipeline script.

### Issue 8: GPU memory contention

**Problem:** First inference attempt saw only 15.4GB free VRAM (out of 128GB unified). The InSpatio model + 301 frames of VAE encoding + DiT inference needs ~97GB. The container was killed by OOM.

**Root cause:** Two llama-server processes (Gemma 4 26B on :18080 and Gemma E2B on :18081) were holding GPU memory.

**Solution:** Stop llama-servers before running InSpatio, restart after. The run script handles this automatically. On the second attempt with 97GB free, inference completed in ~5 minutes.

## Performance Results

| Step | Description | Time | Peak GPU Memory |
|------|-------------|------|-----------------|
| Step 1 | Florence-2 video captioning | <1s | ~4GB |
| Step 2a | DA3 depth estimation (247 frames) | 75s | 25.5GB |
| Step 2b | Point cloud rendering (247 frames) | 14s | ~2GB |
| Step 3 | v2v inference (301 frames, 25 blocks) | 280s (~5min) | ~97GB |
| **Total** | Full pipeline | **~6.5 min** | **~97GB peak** |

Step 3 breakdown:
- VAE Encode: 110s
- DiT + VAE Decode: 170s
- VAE Decode (final): 0.5s

## Container Choice

We tested `nvcr.io/nvidia/pytorch:25.09-py3` (September 2025 release). The November 2025 release (`25.11-py3`) would also work but is ~18GB to download and we already had 25.09 cached.

Key container contents:
- PyTorch 2.9.0 (Blackwell-optimized, not stock pip)
- CUDA 13.0 toolkit
- Flash Attention 2.7.4 (pre-compiled for aarch64 + sm_121)
- Triton 3.4.0
- Python 3.12.3
- cuBLAS, cuDNN, NCCL (all Blackwell-optimized)

## Files Modified from Upstream

1. `scripts/render_point_cloud.py` — replaced open3d import with plyfile
2. Container-level: `decord/__init__.py` — functional pyAV mock (not in repo files)
3. Container-level: `depth_anything_3/utils/export/__init__.py` — lazy imports

The upstream InSpatio-World code is otherwise unmodified. All patches are minimal and targeted.
