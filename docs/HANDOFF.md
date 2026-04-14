# InSpatio-World — Project Handoff Document

**Last updated:** 2026-04-14 00:05 EDT by main  
**For:** Any agent picking up this project

---

## What Is This Project?

InSpatio-World is a 4D world simulator that takes a video and lets you explore the scene from new camera angles. We're building an interactive web app around it on the DGX Spark.

## Current State

### What Works ✅
1. **Full pipeline running in Docker** — upload video → get novel-view video back
2. **Gradio batch UI on port 7860** — resolution presets, quality controls, Scout Preview button
3. **Resolution/quality parameterized** — DA3, point cloud render, and DiT all accept custom width/height/steps
4. **GitHub repo:** https://github.com/Patvscode/inspatio-dgx-spark

### What's Being Built 🔧
1. **Interactive viewer** — real-time 3D navigation with Viser, DiT refinement overlay
2. **spark-viewer framework** — reusable modular components (separate repo)

### What's Parked 📋
1. **Path B speed optimizations** — optimized PyTorch wheels, flash-attn, WorldFM (see `docs/INSPATIO_PATH_B_EXPERIMENTAL.md`)

## Key Files — Read These First

| File | What it is | Read when |
|------|-----------|-----------|
| `docs/ARCHITECTURE.md` | Full modular architecture with code sketches | Before building anything |
| `docs/INTERACTIVE_APP_PLAN.md` | Detailed build plan with Viser shortcut | Before building the interactive app |
| `docs/PRODUCT_VISION.md` | Speed research, GB10 analysis, optimization priorities | When working on performance |
| `docs/INSPATIO_PATH_B_EXPERIMENTAL.md` | Parked speed optimization leads | When ready to push speed further |
| `docs/SETUP_NOTES.md` | How the Docker container was set up | If container needs recreation |
| `app.py` | Working Gradio batch UI | To understand current UI |

## Infrastructure

### Docker Container: `inspatio-world`
- **Image:** nvidia/pytorch:25.09-py3
- **No port mappings** — Gradio runs on host, calls Docker via `docker exec`
- **Model checkpoints:** 69GB in container at `/workspace/inspatio-world/checkpoints/`
- **ARM64 patches applied:** decord→pyAV, open3d→plyfile, DA3 lazy imports
- **Patched pipeline script:** `run_test_pipeline.sh` accepts `--gen_width`, `--gen_height`, `--denoising_steps`
- **Patch saved on host:** `patches/patch_resolution.py` — reapply after container recreation

### To start the container:
```bash
docker start inspatio-world
```

### To check it's running:
```bash
docker ps --filter name=inspatio-world
```

### To run the Gradio app:
```bash
python3 ~/Desktop/AI-apps-workspace/inspatio-world/app.py
# Opens on http://localhost:7860
```

## The Pipeline (3 stages)

### Stage 1: Preprocessing (run once per video, ~30-60s)
- **Florence-2** captions the video (text prompt)
- **DA3** estimates depth maps → PLY point clouds
- Output: `user_input/<name>/new.json` + `user_input/<name>/<name>_da3_tmp/`
- Point cloud: `user_input/<name>/<name>_da3_tmp/point_cloud.ply` (3.7MB)

### Stage 2: Point Cloud Rendering (fast, ~100ms/frame)
- Takes camera pose (angles + radius) → renders point cloud from that viewpoint
- Pure geometry, no neural network
- Outputs: `render_offline.mp4` + `mask_offline.mp4`
- This is the "instant preview" layer

### Stage 3: DiT Neural Refinement (slow, the quality pass)
- Takes rough point cloud render + text → photorealistic output
- `CausalInferencePipeline` processes in blocks of 3 frames
- At 480p, 4 steps: ~2 min. At 240p, 2 steps: ~15-30s
- Config: `configs/inference_1.3b.yaml` — video_size, denoising_step_list, guidance_scale

## Camera Model

InSpatio uses **spherical coordinates**:
- `x_up_angle` — pitch (look up/down), degrees
- `y_left_angle` — yaw (look left/right), degrees
- `r` — radius (distance from scene center, multiplied by depth)

Trajectory files (`traj/*.txt`) have 3 lines:
```
line 1: x_up_angles (space-separated)
line 2: y_left_angles (space-separated)
line 3: radius multipliers (space-separated)
```

These get interpolated to match the frame count via `datasets/utils.py → generate_traj_txt()`.

## GPU Memory Management

The DGX Spark runs llama-servers for the AI agents on ports 18080 and 18081. InSpatio needs the GPU too. The Gradio app's `GPUMemoryManager` class:
1. Stops llama-servers before inference
2. Runs InSpatio pipeline
3. Restarts llama-servers after

**Critical:** Never leave llama-servers stopped. The GPUMemoryManager has a watchdog timer that auto-restores after a configurable delay.

## Performance Numbers

| Setting | Time | Notes |
|---------|------|-------|
| Full (480p, 4 steps) | ~2 min | Best quality |
| Draft (360p, 3 steps) | ~1 min | Good balance |
| Scout (240p, 2 steps) | ~15-30s | Quick preview |
| Point cloud render | ~50ms/frame | Instant (no DiT) |

GB10 specs: ~12 TFLOPS BF16, 273 GB/s bandwidth. Bandwidth-bound, not compute-bound.

## What NOT To Do

1. **Don't modify the Docker container directly** — patches get lost on recreation. Save patches to `patches/` on host.
2. **Don't break `app.py`** — it's the stable batch mode. Build interactive stuff separately.
3. **Don't run InSpatio while llama-servers are running** — GPU contention. Use GPUMemoryManager.
4. **Don't hardcode localhost** — use 0.0.0.0 for servers so Tailscale/phone access works.
5. **Don't skip torch.compile warmup** — first run is slow (30-60s). Subsequent runs are faster. The warmup is cached in `/dev/shm/`.

## Next Steps (for whoever picks this up)

1. **Install viser:** `pip install viser plyfile`
2. **Read:** `docs/ARCHITECTURE.md` — understand the modular split
3. **Build:** `interactive.py` using spark-viewer framework + InSpatio backend
4. **Test:** Load existing PLY from previous run, verify point cloud displays in browser
5. **Wire:** Camera pose tracking → DiT refinement → background image overlay
6. **Estimated time:** ~60-70 minutes for a working interactive viewer
