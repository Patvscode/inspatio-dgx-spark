# InSpatio-World — Product Vision & Research Notes

## Scout & Render Concept

The core UX idea: separate **exploration** (fast, rough, interactive) from **production** (slow, beautiful, automated).

### Scout Mode (Interactive Preview)
- Low resolution (240x416 instead of 480x832) = 4x fewer pixels
- User controls camera with mouse/touch (drag to orbit, scroll to zoom)
- Model generates 3 frames per block, streams to browser
- Expected: ~2-3 FPS at low res on DGX Spark
- Camera trajectory is automatically recorded

### Render Mode (Post-Processing)
After scouting, user picks options:
1. **Re-render at full res** — same trajectory, 480x832, ~2 min
2. **AI upscale** — take low-res output, upscale to 1080p/4K (Real-ESRGAN or similar)
3. **Frame interpolation** — RIFE to go from 2 FPS to 24 FPS
4. **Combo pipeline** — scout → interpolate → upscale (all automated)

### Gallery
- Keep all renders — different angles, quality levels
- Compare side-by-side
- Re-render from saved trajectories

### User Flow
```
Upload video
    ↓
Scout Mode (live, grainy, ~2-3 FPS)
  - drag/orbit to explore
  - trajectory auto-recorded
    ↓
Pick from options:
  ☑️ Re-render (full quality, 480p, ~2 min)
  ☑️ Upscale to 1080p (+30 sec)  
  ☑️ Smooth to 24 FPS (+15 sec)
  ☑️ All of the above
    ↓
Final video plays back smooth in gallery
```

---

## GB10 vs H100 Performance Analysis

### Raw Numbers
| Spec | GB10 (Spark) | H100 (SXM) | Ratio |
|------|-------------|-------------|-------|
| BF16 TFLOPs | ~11-12 | ~1000 | **~83x** |
| Memory | 128GB LPDDR5x | 80GB HBM3 | 1.6x more on Spark |
| Bandwidth | 273 GB/s | 3,400 GB/s | **12.4x less** |
| TDP | 140W (SoC) | 700W | 5x less power |

### Why the 35x Inference Gap (Not 83x)
1. **Memory bandwidth is the real bottleneck, not compute.** Diffusion inference is memory-bound (moving weights through attention), not compute-bound. The 12.4x bandwidth gap matters more than the 83x TFLOP gap.
2. **torch.compile helps us more proportionally** — compiled kernels reduce memory round-trips.
3. **Flash Attention 2** is pre-built in the container — this is bandwidth-optimized attention.
4. **The model is only 1.3B params** — fits easily in our 128GB, no memory pressure. H100 gains less from its huge bandwidth when the model is small.

### The Real Bottleneck
- **273 GB/s LPDDR5x bandwidth** is the #1 limiting factor
- Each DiT block needs to read all model weights + KV cache through memory
- 25 blocks × 4 denoising steps × ~1.3B params (BF16) = lots of memory reads
- This is why NVIDIA pushes FP4 on Spark — 4-bit reduces bandwidth pressure by 4x

### Why InSpatio's Paper Says "Real-Time"
They tested on H100 SXM with HBM3 (3400 GB/s bandwidth). At 24 FPS, each 3-frame block takes 0.125s. On RTX 4090 (1TB/s bandwidth) they get 10 FPS. On our Spark (273 GB/s) we get ~0.65 FPS. The ratio tracks bandwidth almost linearly.

---

## Speed Optimization Research

### Things We Can Try

#### 1. FP8 Quantization (HIGH POTENTIAL)
- **Wan2.2_FP8 repo** (github.com/mali-afridi/Wan2.2_FP8) — Transformer Engine based FP8 for Wan models
- **Aquiles-ai/Wan2.1-Turbo-fp8** on HuggingFace — pre-quantized Wan2.1 with LoRA
- FP8 reduces memory bandwidth by 2x vs BF16 → could nearly double our FPS
- GB10's Blackwell tensor cores natively support FP8
- **Risk:** InSpatio's fine-tuned weights may not survive quantization cleanly

#### 2. NVFP4 Quantization (HIGHEST POTENTIAL, HARDEST)
- GB10 has 1 PFLOP FP4 performance — specifically designed for this
- NVIDIA's NVFP4 with block size 16 reduces quantization error
- 4x bandwidth reduction vs BF16 → theoretical 4x speedup
- **Risk:** Video diffusion models are sensitive to quantization artifacts. Need testing.
- Overworld's world_engine already uses nvfp4 successfully on Spark

#### 3. Fewer Denoising Steps
- Currently 4 steps (1000, 750, 500, 250)
- Could try 2 steps for scout mode — half the DiT compute
- Quality degrades but acceptable for preview

#### 4. Diffusion Caching
- Wan2.2's optimization: skip redundant DiT blocks when adjacent frames are similar
- 1.62x speedup reported
- Would need code changes to InSpatio's inference loop

#### 5. TensorRT Export
- NVIDIA blog shows TensorRT + FP8 for video diffusion models
- Could export the DiT to TensorRT for optimized inference
- Complex but potentially 2-3x speedup

#### 6. Resolution Scaling for Scout Mode
- 240x416 = 4x fewer pixels = 4x less latent computation
- Model supports variable resolution (latent space is resolution-independent)
- Easiest win — just change config values

#### 7. SageAttention
- InSpatio already has SageAttention support in attention.py
- Faster than Flash Attention for some workloads
- Check if it's available in the NVIDIA container

### Priority Order for Implementation
1. **Resolution scaling** (easy, immediate 4x for scout mode)
2. **Fewer denoising steps** (config change, 2x for scout mode)
3. **FP8 quantization** (moderate effort, 2x for everything)
4. **Diffusion caching** (code changes, 1.5x)
5. **NVFP4** (hard, 4x theoretical)
6. **TensorRT** (hardest, 2-3x)

Combined scout mode: half res + 2 steps = potentially **8x faster** = ~5 FPS on Spark.
That's genuinely interactive.

---

## Video Length Capabilities

### Current Limits
- Input: any length .mp4 (pipeline extracts frames)
- The model processes in 3-frame blocks with KV cache
- Adaptive frame expansion: if trajectory needs more frames, it interpolates (bounce)
- Our test: 247 input frames → expanded to 301 → 25 blocks

### Memory Scaling
- Each frame in latent space: [1, 16, 60, 104] BF16 ≈ 200KB
- KV cache: fixed at 9360 entries regardless of video length
- VAE encode is the memory hog: all frames encoded upfront
- **Rough limits on Spark (97GB free):**
  - ~600-800 frames at 480x832 (20-27 seconds at 30fps)
  - ~2000+ frames at 240x416 (67+ seconds at 30fps)

### Practical Recommendations
- Short clips (3-10 sec) work best — less memory, faster processing
- Trim input videos before upload
- Scout mode at low res can handle longer clips
