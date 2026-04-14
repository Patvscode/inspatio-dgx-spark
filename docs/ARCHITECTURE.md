# InSpatio-World Interactive App — Modular Architecture

**Created:** 2026-04-14  
**Goal:** Build it modular so the viewer framework can be reused for any 3D/neural rendering app

---

## Core Idea: Separate the Viewer Framework from the AI Backend

```
┌─────────────────────────────────────────────────────────────┐
│                    spark-viewer (reusable)                    │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ SceneManager │  │  GUIBuilder  │  │  RefinementLoop   │  │
│  │              │  │              │  │                   │  │
│  │ • load PLY   │  │ • sliders    │  │ • watch camera    │  │
│  │ • add meshes │  │ • buttons    │  │ • debounce        │  │
│  │ • manage     │  │ • presets    │  │ • call backend     │  │
│  │   scenes     │  │ • status bar │  │ • overlay result  │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘  │
│         │                 │                    │             │
│         └─────────────────┼────────────────────┘             │
│                           │                                  │
│                    ┌──────┴───────┐                          │
│                    │ ViserServer  │  (pip install viser)      │
│                    │ port 7861    │                          │
│                    └──────────────┘                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    │  Backend    │  ← THIS is what changes per project
                    │  (plugin)   │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  ┌───────────┐    ┌───────────┐     ┌───────────────┐
  │ InSpatio  │    │ WorldFM   │     │ Future App X  │
  │ Backend   │    │ Backend   │     │ Backend       │
  │           │    │           │     │               │
  │ • DiT     │    │ • Frame   │     │ • Whatever    │
  │ • render  │    │   model   │     │   model       │
  │ • preproc │    │ • preproc │     │ • preproc     │
  └───────────┘    └───────────┘     └───────────────┘
```

---

## File Structure

```
~/Desktop/AI-apps-workspace/
├── spark-viewer/                    # REUSABLE FRAMEWORK
│   ├── __init__.py
│   ├── viewer.py                    # Main viewer class (wraps Viser)
│   ├── scene.py                     # SceneManager — load PLY, manage 3D objects
│   ├── gui.py                       # GUIBuilder — standard controls factory
│   ├── refinement.py                # RefinementLoop — camera tracking + backend trigger
│   ├── gpu.py                       # GPUMemoryManager (copied from existing app.py)
│   ├── pose.py                      # Camera pose conversions (quaternion ↔ euler ↔ spherical)
│   └── presets.py                   # Resolution/quality presets (shared across backends)
│
├── inspatio-world/                  # THIS PROJECT
│   ├── app.py                       # Existing Gradio app (batch mode, untouched)
│   ├── interactive.py               # NEW — interactive mode using spark-viewer
│   ├── backend.py                   # NEW — InSpatio-specific backend (implements Backend interface)
│   ├── patches/                     # Container patches
│   ├── docs/                        # Plans, architecture
│   ├── traj/                        # Trajectory files
│   └── ...                          # Existing files
│
├── world_engine/                    # Overworld (existing)
│   └── interactive.py               # FUTURE — could reuse spark-viewer + own backend
│
└── [future-app]/
    └── interactive.py               # FUTURE — same pattern
```

---

## The Backend Interface (what each AI project implements)

```python
# spark-viewer/backend.py — Abstract interface

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class CameraPose:
    """Universal camera pose representation."""
    yaw: float        # degrees, left/right
    pitch: float      # degrees, up/down
    zoom: float       # distance multiplier
    # Raw Viser data preserved for backends that want it
    wxyz: tuple = None       # quaternion
    position: tuple = None   # xyz

@dataclass  
class RenderSettings:
    """What the user has configured via GUI."""
    width: int = 832
    height: int = 480
    denoising_steps: list = None  # e.g. [1000, 750, 500, 250]
    guidance_scale: float = 3.0
    use_tae: bool = True
    compile_dit: bool = True

@dataclass
class RenderResult:
    """What the backend returns."""
    image: np.ndarray          # HxWx3 uint8
    elapsed_seconds: float
    metadata: dict = None      # backend-specific info

class Backend(ABC):
    """Interface that each AI project implements."""
    
    @abstractmethod
    def load_scene(self, input_path: str, progress_callback=None) -> dict:
        """Preprocess input (video/image/etc) → return scene data.
        Returns dict with at least 'point_cloud_path' key.
        Called once per input. Can be slow (preprocessing)."""
        pass
    
    @abstractmethod
    def get_point_cloud(self, scene_data: dict) -> tuple:
        """Return (points_Nx3, colors_Nx3_uint8) for the viewer."""
        pass
    
    @abstractmethod  
    def render(self, scene_data: dict, pose: CameraPose, 
               settings: RenderSettings) -> RenderResult:
        """Run the neural refinement pass at the given pose.
        Called when camera stops. Should return a refined image."""
        pass
    
    @abstractmethod
    def get_presets(self) -> dict:
        """Return available resolution/quality presets for GUI.
        E.g. {"Scout 240p": {"width": 416, "height": 240, "steps": [1000, 250]}}"""
        pass
    
    def cleanup(self):
        """Optional cleanup when viewer shuts down."""
        pass
    
    def supports_streaming(self) -> bool:
        """If True, render() may be called rapidly for streaming frames.
        If False (default), only called after camera stops."""
        return False
```

---

## How the Pieces Connect

### 1. `spark-viewer/viewer.py` — The Main Entry Point

```python
class SparkViewer:
    """Reusable interactive 3D viewer with neural refinement."""
    
    def __init__(self, backend: Backend, port: int = 7861, title: str = "Spark Viewer"):
        self.backend = backend
        self.server = viser.ViserServer(port=port)
        self.scene_mgr = SceneManager(self.server)
        self.gui = GUIBuilder(self.server, backend.get_presets())
        self.gpu_mgr = GPUMemoryManager()
        self.refine_loop = RefinementLoop(
            server=self.server,
            backend=backend,
            gui=self.gui,
            gpu_mgr=self.gpu_mgr,
        )
    
    def load(self, input_path: str):
        """Load an input (video/image) → preprocess → show point cloud."""
        scene_data = self.backend.load_scene(input_path, self.gui.update_progress)
        points, colors = self.backend.get_point_cloud(scene_data)
        self.scene_mgr.set_point_cloud(points, colors)
        self.refine_loop.set_scene(scene_data)
    
    def run(self):
        """Start the viewer (blocks)."""
        self.refine_loop.start()
        print(f"Viewer running at http://localhost:{self.server.port}")
        while True:
            time.sleep(1.0)
```

### 2. `spark-viewer/refinement.py` — The Camera → Render Loop

```python
class RefinementLoop:
    """Watches camera, triggers backend render when camera stops."""
    
    def __init__(self, server, backend, gui, gpu_mgr):
        self.server = server
        self.backend = backend
        self.gui = gui
        self.gpu_mgr = gpu_mgr
        self.scene_data = None
        self._last_pose = None
        self._last_move_time = 0
        self._rendering = False
    
    def start(self):
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()
    
    def _loop(self):
        while True:
            time.sleep(0.1)  # Check 10x/sec
            
            clients = self.server.get_clients()
            if not clients or self.scene_data is None:
                continue
            
            client = list(clients.values())[0]  # Single user for now
            current_pose = self._extract_pose(client)
            
            # Did camera move?
            if self._pose_changed(current_pose, self._last_pose):
                self._last_move_time = time.time()
                self._last_pose = current_pose
                # Clear old refinement when moving
                if self._rendering:
                    pass  # Could cancel in-progress render
                continue
            
            # Camera stopped — should we refine?
            if not self.gui.auto_refine:
                continue
            
            idle_time = time.time() - self._last_move_time
            if idle_time < self.gui.refine_delay:
                continue
            
            if self._rendering or self._last_pose is None:
                continue
            
            # REFINE
            self._rendering = True
            self.gui.set_status("🔄 Refining...")
            
            try:
                self.gpu_mgr.stop_servers()
                settings = self.gui.get_render_settings()
                result = self.backend.render(self.scene_data, self._last_pose, settings)
                
                # Overlay the result
                self.server.scene.set_background_image(result.image, format="jpeg")
                self.gui.set_status(f"✅ Refined in {result.elapsed_seconds:.1f}s")
            except Exception as e:
                self.gui.set_status(f"❌ {str(e)[:50]}")
            finally:
                if not self.gui.keep_gpu:
                    self.gpu_mgr.restart_servers()
                self._rendering = False
```

### 3. `spark-viewer/gui.py` — Standard Controls

```python
class GUIBuilder:
    """Standard GUI controls that any backend can use."""
    
    def __init__(self, server, presets):
        # Quality preset (combined slider)
        preset_names = list(presets.keys())
        self._quality = server.gui.add_dropdown(
            "Quality Preset", preset_names, 
            initial_value=preset_names[len(preset_names)//2]
        )
        
        # Auto-refine
        self._auto_refine = server.gui.add_checkbox("Auto-Refine", initial_value=True)
        self._refine_delay = server.gui.add_slider(
            "Refine Delay (s)", min=0.5, max=5.0, step=0.5, initial_value=1.5
        )
        
        # Manual trigger
        self._refine_btn = server.gui.add_button("⚡ Refine Now")
        
        # Status
        self._status = server.gui.add_text("Status", initial_value="Ready", disabled=True)
        
        # Point cloud appearance
        self._point_size = server.gui.add_slider(
            "Point Size", min=0.005, max=0.1, step=0.005, initial_value=0.02
        )
        
        # GPU
        self._keep_gpu = server.gui.add_checkbox("Keep GPU (batch mode)", initial_value=False)
        
        self.presets = presets
    
    @property
    def auto_refine(self): return self._auto_refine.value
    
    @property
    def refine_delay(self): return self._refine_delay.value
    
    @property
    def keep_gpu(self): return self._keep_gpu.value
    
    def get_render_settings(self) -> RenderSettings:
        preset = self.presets[self._quality.value]
        return RenderSettings(**preset)
    
    def set_status(self, text): 
        self._status.value = text
    
    def update_progress(self, msg):
        self._status.value = msg
```

### 4. `inspatio-world/backend.py` — InSpatio-Specific Implementation

```python
class InSpatioBackend(Backend):
    """InSpatio-World specific backend."""
    
    def __init__(self, container_name="inspatio-world"):
        self.container = container_name
    
    def load_scene(self, input_path, progress_callback=None):
        # Run Steps 1+2 (Florence + DA3 + point cloud render prep)
        if progress_callback:
            progress_callback("Running Florence-2 captioning...")
        # docker exec ... run_test_pipeline.sh --skip_step3
        # Returns paths to point cloud, caption, depth data
        return {
            "point_cloud_path": "..._da3_tmp/point_cloud.ply",
            "json_path": ".../new.json",
            "caption": "...",
            "depth_path": "...",
            "radius": ...,
        }
    
    def get_point_cloud(self, scene_data):
        ply = PlyData.read(scene_data["point_cloud_path"])
        v = ply['vertex']
        points = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
        colors = np.stack([v['red'], v['green'], v['blue']], axis=-1).astype(np.uint8)
        return points, colors
    
    def render(self, scene_data, pose, settings):
        t0 = time.time()
        
        # 1. Generate trajectory file from single pose
        traj = self._pose_to_trajectory(pose, n_frames=3)
        
        # 2. Run point cloud render at target resolution
        self._run_render(scene_data, traj, settings.width, settings.height)
        
        # 3. Run DiT inference (Step 3 only)
        result_path = self._run_dit(scene_data, traj, settings)
        
        # 4. Read result
        image = read_video_first_frame(result_path)
        return RenderResult(image=image, elapsed_seconds=time.time() - t0)
    
    def get_presets(self):
        return {
            "⚡ Scout (240p, 2 steps)": {
                "width": 416, "height": 240,
                "denoising_steps": [1000, 250],
            },
            "🔄 Draft (360p, 3 steps)": {
                "width": 624, "height": 360,
                "denoising_steps": [1000, 500, 250],
            },
            "✨ Full (480p, 4 steps)": {
                "width": 832, "height": 480,
                "denoising_steps": [1000, 750, 500, 250],
            },
        }
    
    def _pose_to_trajectory(self, pose, n_frames=3):
        """Convert CameraPose → InSpatio trajectory (3-line txt format)."""
        # Line 1: x_up_angles (pitch) — repeated n_frames times
        # Line 2: y_left_angles (yaw) — repeated n_frames times  
        # Line 3: radius values — repeated n_frames times
        x = " ".join([str(pose.pitch)] * n_frames)
        y = " ".join([str(pose.yaw)] * n_frames)
        r = " ".join([str(pose.zoom)] * n_frames)
        return f"{x}\n{y}\n{r}"
```

### 5. `inspatio-world/interactive.py` — Launch Script

```python
"""InSpatio-World Interactive Viewer — launch script."""
from spark_viewer import SparkViewer
from backend import InSpatioBackend

def main():
    backend = InSpatioBackend(container_name="inspatio-world")
    viewer = SparkViewer(backend, port=7861, title="InSpatio-World Explorer")
    
    # Auto-load the last processed scene if available
    import glob
    scenes = glob.glob("user_input/*/new.json")
    if scenes:
        viewer.load(scenes[-1])
    
    viewer.run()

if __name__ == "__main__":
    main()
```

---

## Reuse Examples (future)

### WorldFM Backend (the faster scout model)
```python
class WorldFMBackend(Backend):
    def render(self, scene_data, pose, settings):
        # WorldFM uses 1-step DMD — much faster
        # Different model, same viewer interface
        ...
    
    def supports_streaming(self):
        return True  # Fast enough for continuous frames
```

### Overworld/Waypoint Backend
```python  
class OverworldBackend(Backend):
    def load_scene(self, input_path, progress_callback=None):
        # Overworld uses a seed image, not a video
        ...
    
    def render(self, scene_data, pose, settings):
        # Waypoint model generates next frame from pose
        ...
```

### Generic Point Cloud Viewer (no AI, just viewing)
```python
class StaticBackend(Backend):
    def render(self, scene_data, pose, settings):
        return None  # No refinement, just point cloud viewing
```

---

## Port Assignments
| Service | Port | Notes |
|---------|------|-------|
| Gradio (batch mode) | 7860 | Existing, untouched |
| Viser (interactive) | 7861 | New |
| DiT inference server | (docker exec) | No port needed if using subprocess |

---

## Data Flow for a Single Interaction

```
User drags camera
       │
       ▼
Viser tracks pose (client.camera.wxyz/position)
       │
       ▼
RefinementLoop detects camera stopped (1.5s no movement)
       │
       ▼
GPUMemoryManager stops llama-servers
       │
       ▼
Backend.render(pose, settings) called
       │
       ├── 1. Pose → trajectory file (instant)
       ├── 2. Point cloud render at target resolution (~100ms)
       ├── 3. DiT inference on rendered input (~3-30s depending on settings)
       └── 4. Return refined image
       │
       ▼
server.scene.set_background_image(result)
       │
       ▼
User sees photorealistic view fade in behind point cloud
       │
       ▼
GPUMemoryManager restarts llama-servers (unless batch mode)
```
