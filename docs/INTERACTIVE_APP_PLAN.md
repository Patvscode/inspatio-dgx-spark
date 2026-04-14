# InSpatio-World Interactive App — Full Build Plan

**Created:** 2026-04-13 23:45 EDT  
**Updated:** 2026-04-14 00:00 EDT — MASSIVE SHORTCUT FOUND (Viser)  
**Status:** PLANNED — execute next session  
**Goal:** Real-time interactive camera control of InSpatio-World generated worlds

---

## THE SHORTCUT: Viser (`pip install viser`)

**Viser** is a Python 3D visualization library built by the nerfstudio team for exactly this use case — interactive neural rendering viewers.

### What Viser gives us FOR FREE:
1. **Point cloud rendering** — `server.scene.add_point_cloud(points, colors, point_size)` — loads our PLY files directly, renders in browser at 60fps via Three.js
2. **Orbit/pan/zoom camera controls** — built-in, mouse + touch + keyboard, momentum
3. **Camera pose tracking** — `client.camera.wxyz`, `client.camera.position`, `client.camera.update_timestamp` — know exactly where user is looking and when they stopped
4. **GUI building blocks** — `server.gui.add_slider()`, `add_button()`, `add_checkbox()` — our entire settings panel
5. **Background image overlay** — `server.scene.set_background_image(numpy_array)` — overlay DiT results directly behind point cloud
6. **Get renders** — `client.get_render(height, width)` — capture viewport
7. **WebSocket transport** — works over SSH, Tailscale, remote
8. **Theming** — dark mode, notifications, modal dialogs

### What we still need to build:
1. **Scene loading** — preprocess video (Florence + DA3), load PLY into Viser
2. **DiT refinement loop** — watch camera pose, debounce, run inference, overlay result
3. **Camera pose conversion** — Viser quaternion+position → InSpatio spherical angles
4. **GPU memory management** — same pattern as Gradio app

### Estimated build time: ~45-60 minutes (down from 3 hours)

The architecture collapses into a single Python file:
```python
import viser
import numpy as np
from plyfile import PlyData

server = viser.ViserServer(port=7861)
ply = PlyData.read("point_cloud.ply")
points = np.stack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']], axis=-1)
colors = np.stack([ply['vertex']['red'], ply['vertex']['green'], ply['vertex']['blue']], axis=-1)
server.scene.add_point_cloud("/world", points=points, colors=colors, point_size=0.02)

quality = server.gui.add_slider("Quality", min=1, max=4, step=1, initial_value=2)
auto_refine = server.gui.add_checkbox("Auto-Refine", initial_value=True)
refine_btn = server.gui.add_button("Refine Now")
# Camera tracking + DiT trigger loop in background thread
```

### Source projects / prebuilt components:
| Component | Source | What it gives us |
|-----------|--------|-----------------|
| 3D viewer + controls + GUI | **viser** (pip) | ~90% of frontend |
| Point cloud I/O | **plyfile** (pip, already installed) | PLY loading |
| Camera pose math | **viser + InSpatio utils.py** | Pose conversion |
| DiT inference | **InSpatio CausalInferencePipeline** (in Docker) | Neural refinement |
| GPU management | **Our GPUMemoryManager** (from app.py) | Server stop/start |
| Viewer architecture | **nerfstudio** (reference) | Camera→render→display loop pattern |

---

## How InSpatio-World Actually Works (the key insight)

The pipeline has **3 stages**:

### Stage 1: Preprocessing (run ONCE per input video)
1. **Florence-2** captions the video (text prompt)
2. **DA3** estimates depth maps for every frame
3. Depth maps → PLY point clouds

This takes ~30-60s. Only needs to happen once per video.

### Stage 2: Point Cloud Rendering (FAST, can run per-frame)
Given a camera pose (x_angle, y_angle, radius), renders the point cloud from that viewpoint.
- Outputs: `render_video` (what the camera "sees" from the point cloud) + `mask_video` (valid/invalid pixels)
- This is **pure geometry** — no neural network, just z-buffer splatting
- Takes ~10-50ms per frame depending on resolution
- This is our "instant preview" — the raw point cloud render

### Stage 3: DiT Neural Refinement (SLOW, the quality pass)
Takes the rough point cloud render + text prompt → produces photorealistic output.
- The `CausalInferencePipeline` processes in **blocks of 3 frames**
- Each block: context encoding + N denoising steps
- At 480p with 4 steps: ~2 min for a full video
- At 240p with 2 steps: ~15-30 sec
- Single block (3 frames): maybe 2-5 sec

**The key insight for interactivity:**
- Point cloud render = instant (~50ms) but rough (visible holes, artifacts)
- DiT refinement = slow but photorealistic
- We can show the POINT CLOUD render in real-time as the user moves, then run DiT on whatever they're looking at to "fill in" quality

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    BROWSER (React)                    │
│                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ Point Cloud  │  │  DiT Refined │  │  Controls   │ │
│  │ Live Canvas  │  │  Overlay     │  │  Panel      │ │
│  │ (instant)    │  │  (delayed)   │  │             │ │
│  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘ │
│         │                │                  │        │
│         │    WebSocket (camera pose + settings)      │
│         └────────────────┼──────────────────┘        │
└──────────────────────────┼───────────────────────────┘
                           │
                    ┌──────┴──────┐
                    │  FastAPI    │
                    │  Server     │
                    │  (host)     │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Renderer │ │   DiT    │ │ Preproc  │
        │ (CPU/GPU)│ │  (GPU)   │ │ (once)   │
        │ ~50ms/f  │ │ ~2-5s/b  │ │ ~60s     │
        └──────────┘ └──────────┘ └──────────┘
              ▲            ▲
              │            │
        Docker container (inspatio-world)
```

### Two-Layer Rendering Strategy

1. **Layer 1: Point Cloud (instant)**  
   As the user moves the camera, render the point cloud at that pose.
   This is purely geometric — fast enough for real-time even on CPU.
   Shows: rough 3D view with holes/artifacts where the point cloud is sparse.

2. **Layer 2: DiT Refinement (background)**  
   When the user pauses or settles on a view, queue a DiT refinement pass.
   Once complete, crossfade from the rough render to the photorealistic output.
   If the user moves again before it finishes, cancel and restart.

This gives:
- **Instant feedback** when moving (point cloud)
- **Beautiful output** when still (DiT)
- The DiT acts like "autofocus" — it takes a moment to sharpen

---

## Camera Control System

The camera model is **spherical** (already how InSpatio works):
- `x_up_angle` — pitch (look up/down), degrees
- `y_left_angle` — yaw (look left/right), degrees  
- `r` — radius/zoom (distance from scene center)

### Input Mappings

**Desktop:**
| Input | Action |
|-------|--------|
| Click + drag | Orbit (yaw + pitch) |
| Scroll wheel | Zoom in/out |
| WASD | Orbit (keyboard) |
| Q/E | Zoom in/out |
| Space | Pause/resume DiT refinement |

**Mobile/Touch:**
| Input | Action |
|-------|--------|
| Single finger drag | Orbit |
| Pinch | Zoom |
| Double tap | Reset to center |

**Gamepad (stretch goal):**
| Input | Action |
|-------|--------|
| Left stick | Orbit |
| Right trigger | Zoom in |
| Left trigger | Zoom out |

### Control Feel
- **Momentum/inertia** — camera keeps drifting slightly after drag release (like Google Maps)
- **Smoothing** — lerp between current and target pose (60fps UI, model gets interpolated poses)
- **Deadzone** — small movements don't trigger re-render
- **Angle limits** — clamp pitch to ±45°, yaw ±180° (matches trajectory file ranges)

---

## Interactive Settings Panel

### Primary Controls (always visible, draggable sliders)

1. **Quality ↔ Speed slider**  
   Single slider that controls resolution + denoising steps together.
   - Left (Fast): 240p, 1 step → fastest DiT pass
   - Middle: 360p, 2 steps
   - Right (Quality): 480p, 4 steps → best quality  
   Shows live estimate: "DiT: ~3s" / "DiT: ~15s" / "DiT: ~2 min"

2. **Auto-Refine toggle**  
   ON: DiT automatically runs when camera stops moving  
   OFF: Only renders point cloud, manual "Refine" button to trigger DiT

3. **Refine Delay slider** (0.5s - 5s)  
   How long to wait after camera stops before triggering DiT.
   Short = more responsive, burns more GPU. Long = only refines deliberate pauses.

### Secondary Controls (expandable panel)

4. **Resolution** — independent override (240×416, 360×624, 480×832)
5. **Denoising Steps** — independent override (1, 2, 3, 4)
6. **Guidance Scale** — 1.0-5.0 (default 3.0, from config)
7. **Point Cloud Render** — show/hide the rough layer
8. **Crossfade Duration** — how fast the DiT result fades in (0.1-2.0s)

### Status Bar
- Current FPS (point cloud layer)
- DiT queue status: "Idle" / "Rendering 2/4 steps..." / "Done (3.2s)"
- GPU memory usage
- Camera: yaw 15° pitch -5° zoom 1.2x

---

## Technical Implementation Plan

### File Structure
```
~/Desktop/AI-apps-workspace/inspatio-world/
├── interactive/
│   ├── server.py            # FastAPI + WebSocket server
│   ├── renderer.py          # Point cloud renderer (extracted from render_point_cloud.py)
│   ├── dit_worker.py        # DiT inference worker (extracted from inference_causal_test.py)
│   ├── scene_manager.py     # Manages loaded scene state (point clouds, depth, captions)
│   ├── static/
│   │   ├── index.html       # Single-page app
│   │   ├── app.js           # Main app logic
│   │   ├── camera.js        # Camera controls (orbit, zoom, momentum)
│   │   ├── renderer.js      # Canvas rendering + crossfade
│   │   ├── controls.js      # Settings panel + sliders
│   │   └── style.css        # Clean dark theme
│   └── launch.sh            # One-click launcher
├── app.py                   # Existing Gradio app (untouched)
└── docs/
    └── INTERACTIVE_APP_PLAN.md  # This file
```

### Backend: `server.py` (FastAPI, runs on host)

```python
# Endpoints:
POST /api/load          # Load a video → run preprocessing (Florence + DA3 + point clouds)
                        # Returns scene_id when done

WS   /ws/{scene_id}     # WebSocket for real-time control
                        # Client sends: {"pose": {"yaw": 15, "pitch": -5, "zoom": 1.0}, "settings": {...}}
                        # Server sends: {"type": "pointcloud", "frame": base64_jpeg}
                        #               {"type": "dit_result", "frame": base64_jpeg}
                        #               {"type": "status", "dit": "rendering", "step": 2, "total": 4}

GET  /api/scenes        # List loaded scenes
GET  /api/status        # GPU/memory status
```

### Key Backend Design Decisions

1. **Point cloud rendering on HOST** (not in Docker)
   - The renderer is pure numpy/torch — no special deps
   - Already patched for ARM64 (plyfile instead of open3d)
   - Avoids Docker exec overhead for the latency-critical path
   - We extract `render_point_cloud.py` logic into `renderer.py`

2. **DiT runs IN Docker** via subprocess or direct Python import
   - Model weights are in the container
   - We keep the container running with model loaded
   - Alternative: run a persistent inference server inside Docker (better)

3. **Persistent model loading**
   - The biggest time waste currently: loading models every run
   - The interactive server loads once at startup, keeps warm
   - DiT + TAE + text encoder stay in GPU memory
   - Point clouds stay in CPU memory (they're small)

4. **Frame delivery: JPEG over WebSocket**
   - Encode each frame as JPEG (quality 70-85)
   - At 240p: ~15-30KB per frame
   - At 480p: ~40-80KB per frame
   - WebSocket handles the streaming naturally

### Frontend: Camera Controls

Using vanilla JS (no heavy framework needed for this):

```javascript
// Camera state
let camera = { yaw: 0, pitch: 0, zoom: 1.0 };
let velocity = { yaw: 0, pitch: 0, zoom: 0 };
const FRICTION = 0.92;  // momentum decay
const SENSITIVITY = 0.3;  // degrees per pixel of drag

// Input handling
canvas.onmousedown → start tracking
canvas.onmousemove → update velocity from delta
canvas.onmouseup → release (momentum continues)
canvas.onwheel → zoom velocity

// Animation loop (60fps)
function tick() {
    camera.yaw += velocity.yaw;
    camera.pitch += velocity.pitch;
    camera.zoom += velocity.zoom;
    velocity.yaw *= FRICTION;
    velocity.pitch *= FRICTION;
    velocity.zoom *= FRICTION;
    
    // Clamp
    camera.pitch = clamp(camera.pitch, -45, 45);
    camera.zoom = clamp(camera.zoom, 0.3, 3.0);
    
    // Send to server if changed significantly
    if (poseChanged(camera, lastSent)) {
        ws.send(JSON.stringify({pose: camera}));
        lastSent = {...camera};
    }
    
    requestAnimationFrame(tick);
}
```

### Frontend: Dual-Canvas Rendering

```html
<div id="viewport">
    <canvas id="pointcloud-layer"></canvas>   <!-- always updating -->
    <canvas id="dit-layer"></canvas>           <!-- crossfade overlay -->
</div>
```

- Point cloud frames arrive fast → draw immediately on bottom canvas
- DiT frames arrive slowly → fade in on top canvas over 0.3-1s
- When camera moves again → fade out DiT layer, show point cloud

### Frontend: Settings Panel (inspired by game settings)

```
┌─────────────────────────────────────┐
│ ⚡ Quality ◀━━━━━━●━━━━━━━▶ ✨      │  ← Single combined slider
│            ~3s          ~2min       │
│                                     │
│ [Auto-Refine: ON]  Delay: [1.5s]  │
│                                     │
│ ▸ Advanced Settings                 │
│   Resolution: [240p ▾]              │
│   Steps: [2 ▾]                      │
│   Guidance: [3.0 ━━━●━━]           │
│   Show Point Cloud: [✓]            │
│   Crossfade: [0.5s ━━●━━]          │
└─────────────────────────────────────┘
```

Dark theme, semi-transparent, positioned bottom-right like game HUD.
Draggable/collapsible.

---

## Execution Steps (next session) — REVISED with Viser shortcut

### Phase 1: Viser Point Cloud Viewer (~15 min)
1. `pip install viser` on host
2. Write `interactive/viewer.py` — load PLY, add to scene, add GUI sliders
3. Launch, verify point cloud renders in browser with orbit controls
4. Test on phone via Tailscale

### Phase 2: GUI Controls (~10 min)
5. Add quality slider, auto-refine checkbox, refine button, status text
6. Add resolution dropdown, denoising steps slider
7. Add point size slider for visual tuning

### Phase 3: Camera Pose Tracking + DiT Integration (~25 min)
8. Background thread: poll `client.camera.wxyz/position/update_timestamp`
9. Debounce: detect when camera stops moving (no pose change for N seconds)
10. Convert Viser quaternion+position → InSpatio spherical angles (x_up, y_left, r)
11. Run point cloud render (Python, from our patched renderer) at target resolution
12. Run DiT inference (docker exec or persistent server) with rendered input
13. Overlay result via `server.scene.set_background_image(dit_output)`

### Phase 4: GPU Management + Scene Loading (~10 min)
14. Port GPUMemoryManager from app.py (stop/start llama-servers)
15. Add "Load Video" button — runs preprocessing pipeline (Florence + DA3)
16. Show progress/status during preprocessing

### Phase 5: Polish (~10 min)
17. Loading states and notifications
18. Record trajectory button (save camera path for batch render)
19. Error handling

**Total estimated: ~60-70 minutes**

---

## Performance Expectations

| Layer | Resolution | Expected FPS | Feel |
|-------|-----------|-------------|------|
| Point cloud | 240p | 15-20 FPS | Smooth orbit |
| Point cloud | 480p | 8-12 FPS | Usable |
| DiT single block | 240p, 2 steps | ~2-5 sec | "Loading..." |
| DiT single block | 480p, 4 steps | ~30-60 sec | Background task |

The point cloud layer is what makes it feel interactive.
The DiT layer is what makes it look amazing.
Together: you orbit freely, pause, and the world sharpens around you.

---

## Open Questions / Risks

1. **Point cloud renderer performance on host** — currently uses numpy. Could be GPU-accelerated with PyTorch for more FPS.
2. **DiT single-block inference** — we need to test if we can run just 1 block (3 frames) independently, or if it needs full video context. The causal pipeline suggests yes, but needs verification.
3. **Memory management** — keeping DiT loaded alongside llama-servers may not fit. The GPU manager from the Gradio app should handle this.
4. **Docker networking** — WebSocket from host to Docker inference. May need to expose a port or use docker exec.
5. **First-frame warmup** — torch.compile warmup takes 30-60s on first inference. Need to handle this gracefully (show "Warming up..." on first DiT pass).

---

## Stretch Goals (future)

- **Record trajectory** — record your camera path, then render full quality video along that path
- **Multiple viewpoints** — split screen showing different angles
- **Interpolation** — smooth between discrete DiT frames
- **WorldFM integration** — use WorldFM for the "instant" layer instead of point clouds (better quality, still fast)
- **VR mode** — WebXR for headset viewing
