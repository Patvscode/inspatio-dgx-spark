#!/usr/bin/env python3
"""InSpatio-World Interactive Viewer — real-time 3D exploration with DiT refinement.

Browse your scene at 60fps, pause to get AI-refined views.
Uses Viser for the 3D viewer and InSpatio-World for neural refinement.

Usage:
    python3 interactive.py
    # Then open http://localhost:7861 (or Tailscale IP)
"""

import glob
import os
import shutil
import subprocess
import threading
import time

import numpy as np
import viser
from plyfile import PlyData

# ── Config ──
HOST_DIR = os.path.expanduser("~/Desktop/AI-apps-workspace/inspatio-world")
CONTAINER_NAME = "inspatio-world"
CONTAINER_WORK = "/workspace/inspatio-world"
CONTAINER_CONFIG = "configs/inference_1.3b.yaml"
PORT = 7861

# Resolution presets: name → (height, width, denoising_steps, description)
PRESETS = {
    "⚡ Scout (240p)": (240, 416, "1000,250", "~15-30s"),
    "🔄 Draft (360p)": (360, 624, "1000,500,250", "~1 min"),
    "✨ Full (480p)": (480, 832, "1000,750,500,250", "~2 min"),
}


# ── GPU Memory Manager ──
class GPUManager:
    def __init__(self):
        self.lock = threading.Lock()

    def _count_servers(self):
        try:
            r = subprocess.run(["pgrep", "-c", "llama-server"],
                               capture_output=True, text=True, timeout=5)
            return int(r.stdout.strip()) if r.returncode == 0 else 0
        except Exception:
            return 0

    def stop(self):
        with self.lock:
            if self._count_servers() == 0:
                return
            subprocess.run(["systemctl", "--user", "stop", "llama-main.service"],
                           timeout=10, capture_output=True)
            subprocess.run(["pkill", "-f", "llama-server"], timeout=5, capture_output=True)
            time.sleep(3)
            subprocess.run(["pkill", "-9", "-f", "llama-server"], timeout=5, capture_output=True)
            time.sleep(2)

    def restart(self):
        with self.lock:
            if self._count_servers() > 0:
                return
            subprocess.run(["systemctl", "--user", "start", "llama-main.service"],
                           timeout=15, capture_output=True)
            time.sleep(2)
            # Also restart E2B helper
            r = subprocess.run(["pgrep", "-af", "llama-server.*18081"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode != 0:
                subprocess.Popen(
                    ["/home/pmello/llama.cpp-new/build-cuda/bin/llama-server",
                     "-m", "/home/pmello/models/gemma-4/E2B-it/gemma-4-E2B-it-Q8_0.gguf",
                     "--mmproj", "/home/pmello/models/gemma-4/E2B-it/mmproj-BF16.gguf",
                     "--host", "127.0.0.1", "--port", "18081",
                     "--ctx-size", "262144", "--n-gpu-layers", "999",
                     "--threads", "8", "--jinja", "--reasoning-format", "deepseek"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

    def status(self):
        n = self._count_servers()
        return f"{n} model(s) active" if n > 0 else "GPU free"


# ── Scene Manager ──
class SceneData:
    """Holds all data for a loaded scene."""
    def __init__(self, scene_dir):
        self.scene_dir = scene_dir
        self.points = None
        self.colors = None
        self.json_path = None
        self.caption = ""
        self.radius = 1.0
        self.depth_min = 1.0

    def load(self):
        # Find point cloud
        ply_path = os.path.join(self.scene_dir, "point_cloud.ply")
        if not os.path.exists(ply_path):
            # Search in subdirs
            plys = glob.glob(os.path.join(self.scene_dir, "**/point_cloud.ply"), recursive=True)
            if plys:
                ply_path = plys[0]
            else:
                raise FileNotFoundError(f"No point_cloud.ply found in {self.scene_dir}")

        ply = PlyData.read(ply_path)
        v = ply['vertex']
        self.points = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
        self.colors = np.stack([v['red'], v['green'], v['blue']], axis=-1).astype(np.uint8)

        # Find json_path (for DiT inference)
        parent = os.path.dirname(os.path.dirname(ply_path))  # up from da3_tmp
        jsons = glob.glob(os.path.join(parent, "*.json"))
        if jsons:
            self.json_path = jsons[0]

        print(f"Loaded {len(self.points)} points from {ply_path}")
        return self


def find_latest_scene():
    """Find the most recently processed scene."""
    base = os.path.join(HOST_DIR, "user_input")
    if not os.path.isdir(base):
        return None
    # Look for da3_tmp dirs with point_cloud.ply
    plys = glob.glob(os.path.join(base, "**/*_da3_tmp/point_cloud.ply"), recursive=True)
    if not plys:
        return None
    # Return the directory of the most recent one
    newest = max(plys, key=os.path.getmtime)
    return os.path.dirname(newest)


def pose_to_spherical(wxyz, position):
    """Convert Viser camera (quaternion + position) to spherical angles for InSpatio."""
    w, x, y, z = wxyz

    # Yaw from quaternion
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

    # Pitch from quaternion
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))

    # Zoom from distance to origin
    pos = np.array(position)
    zoom = float(np.linalg.norm(pos))
    if zoom < 0.01:
        zoom = 1.0

    return yaw, pitch, zoom


def run_dit_refinement(scene: SceneData, yaw, pitch, zoom, preset_name, gpu_mgr):
    """Run the DiT pipeline for a single viewpoint. Returns image path or None."""
    if scene.json_path is None:
        return None

    h, w, steps, _ = PRESETS[preset_name]

    # Create a temporary trajectory file for this single pose
    traj_dir = os.path.join(HOST_DIR, "traj")
    traj_path = os.path.join(traj_dir, "_interactive_pose.txt")

    # Clamp angles to reasonable ranges
    pitch_clamped = np.clip(pitch, -45, 45)
    yaw_clamped = np.clip(yaw, -60, 60)

    # Write 3-frame static trajectory (model needs at least 3 frames per block)
    with open(traj_path, 'w') as f:
        angles_x = " ".join([f"{pitch_clamped:.1f}"] * 4)
        angles_y = " ".join([f"{yaw_clamped:.1f}"] * 4)
        radii = " ".join(["1.0"] * 4)
        f.write(f"{angles_x}\n{angles_y}\n{radii}\n")

    # Clean stale output
    traj_name = "_interactive_pose"
    json_name = os.path.splitext(os.path.basename(scene.json_path))[0]
    # The output goes to output/<input_dir_name>/<traj_name>/
    input_dir_name = os.path.basename(os.path.dirname(scene.json_path))
    output_dir = os.path.join(HOST_DIR, "output", input_dir_name, traj_name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Get the input dir name for the pipeline
    input_rel = os.path.relpath(os.path.dirname(scene.json_path), HOST_DIR)

    # Build docker exec command
    cmd = (
        f"cd {CONTAINER_WORK} && "
        f"TORCH_CUDA_ARCH_LIST=12.1a "
        f"TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas "
        f"TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache "
        f"bash run_test_pipeline.sh "
        f"--input_dir ./{input_rel} "
        f"--traj_txt_path ./traj/_interactive_pose.txt "
        f"--config_path {CONTAINER_CONFIG} "
        f"--master_port 29516 "
        f"--gen_width {w} --gen_height {h} "
        f"--denoising_steps {steps} "
        f"--skip_step1 --skip_step2 "
        f"--use_tae --compile_dit"
    )

    gpu_mgr.stop()
    try:
        result = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "bash", "-c", cmd],
            capture_output=True, text=True, timeout=600)
    except Exception as e:
        gpu_mgr.restart()
        return None

    if result.returncode != 0:
        gpu_mgr.restart()
        return None

    # Find output video
    pred_videos = sorted(glob.glob(os.path.join(output_dir, "**/*pred_video*.mp4"), recursive=True))
    if not pred_videos:
        gpu_mgr.restart()
        return None

    return pred_videos[0]


def extract_frame_from_video(video_path):
    """Extract first frame from video as numpy array."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass

    # Fallback: use ffmpeg
    tmp_path = "/tmp/inspatio_frame.png"
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-frames:v", "1", "-q:v", "2", tmp_path],
        capture_output=True, timeout=10)
    if os.path.exists(tmp_path):
        from PIL import Image
        img = np.array(Image.open(tmp_path))
        os.remove(tmp_path)
        return img
    return None


# ── Main App ──
def main():
    gpu_mgr = GPUManager()

    # Find latest scene
    scene_dir = find_latest_scene()
    if scene_dir is None:
        print("No preprocessed scenes found. Run the Gradio app first to process a video.")
        print("Then restart this viewer.")
        return

    print(f"Loading scene from: {scene_dir}")
    scene = SceneData(scene_dir).load()

    # Start Viser
    server = viser.ViserServer(port=PORT)
    print(f"\n{'='*60}")
    print(f"  InSpatio-World Interactive Viewer")
    print(f"  Local:     http://localhost:{PORT}")
    print(f"  Tailscale: http://100.109.173.109:{PORT}")
    print(f"  Points:    {len(scene.points):,}")
    print(f"{'='*60}\n")

    # ── Scene ──
    server.scene.add_point_cloud(
        "/world",
        points=scene.points,
        colors=scene.colors,
        point_size=0.015,
    )

    # ── GUI ──
    # Title
    server.gui.add_markdown("## InSpatio-World Explorer")

    # Quality preset
    preset_names = list(PRESETS.keys())
    gui_preset = server.gui.add_dropdown(
        "Quality", preset_names, initial_value=preset_names[0]
    )

    # Auto-refine
    gui_auto_refine = server.gui.add_checkbox("Auto-Refine", initial_value=True)
    gui_refine_delay = server.gui.add_slider(
        "Refine Delay (s)", min=0.5, max=5.0, step=0.5, initial_value=1.5
    )

    # Manual refine
    gui_refine_btn = server.gui.add_button("⚡ Refine Now")

    # Point cloud appearance
    gui_point_size = server.gui.add_slider(
        "Point Size", min=0.005, max=0.05, step=0.005, initial_value=0.015
    )

    # Keep GPU toggle
    gui_keep_gpu = server.gui.add_checkbox("Keep GPU (batch mode)", initial_value=False)

    # Status
    gui_status = server.gui.add_markdown("*Status: Ready — orbit to explore*")

    # GPU controls
    gui_free_gpu = server.gui.add_button("🔓 Free GPU")
    gui_restore_gpu = server.gui.add_button("🔒 Restore Models")

    # ── Point size callback ──
    @gui_point_size.on_update
    def _(_):
        server.scene.add_point_cloud(
            "/world", points=scene.points, colors=scene.colors,
            point_size=gui_point_size.value,
        )

    # ── GPU buttons ──
    @gui_free_gpu.on_click
    def _(_):
        gpu_mgr.stop()
        gui_status.content = f"*Status: GPU freed — {gpu_mgr.status()}*"

    @gui_restore_gpu.on_click
    def _(_):
        gpu_mgr.restart()
        gui_status.content = f"*Status: Models restored — {gpu_mgr.status()}*"

    # ── Refinement state ──
    refine_state = {
        "last_pose": None,
        "last_move_time": time.time(),
        "rendering": False,
        "last_refined_pose": None,
    }

    def do_refine(yaw, pitch, zoom):
        """Run DiT refinement in background."""
        if refine_state["rendering"]:
            return
        refine_state["rendering"] = True

        preset = gui_preset.value
        _, _, _, est = PRESETS[preset]
        gui_status.content = f"*Status: 🔄 Refining ({preset.split('(')[0].strip()}, est {est})...*"

        t0 = time.time()
        video_path = run_dit_refinement(scene, yaw, pitch, zoom, preset, gpu_mgr)

        if video_path:
            frame = extract_frame_from_video(video_path)
            if frame is not None:
                server.scene.set_background_image(frame, format="jpeg")
                elapsed = time.time() - t0
                gui_status.content = f"*Status: ✅ Refined in {elapsed:.1f}s — move to explore more*"
                refine_state["last_refined_pose"] = (yaw, pitch, zoom)
            else:
                gui_status.content = "*Status: ⚠️ Refined but couldn't extract frame*"
        else:
            gui_status.content = "*Status: ❌ Refinement failed — check container*"

        if not gui_keep_gpu.value:
            gpu_mgr.restart()
        refine_state["rendering"] = False

    # ── Manual refine button ──
    @gui_refine_btn.on_click
    def _(event: viser.GuiEvent):
        client = event.client
        if client is None:
            return
        yaw, pitch, zoom = pose_to_spherical(client.camera.wxyz, client.camera.position)
        threading.Thread(target=do_refine, args=(yaw, pitch, zoom), daemon=True).start()

    # ── Camera tracking + auto-refine loop ──
    def camera_loop():
        while True:
            time.sleep(0.2)

            clients = server.get_clients()
            if not clients:
                continue

            client = list(clients.values())[0]
            current = pose_to_spherical(client.camera.wxyz, client.camera.position)

            # Did camera move?
            last = refine_state["last_pose"]
            if last is None:
                refine_state["last_pose"] = current
                refine_state["last_move_time"] = time.time()
                continue

            moved = (
                abs(current[0] - last[0]) > 0.5
                or abs(current[1] - last[1]) > 0.5
                or abs(current[2] - last[2]) > 0.02
            )

            if moved:
                refine_state["last_pose"] = current
                refine_state["last_move_time"] = time.time()
                # Clear refined background when moving
                if refine_state["last_refined_pose"] is not None:
                    # Could clear background here if desired
                    pass
                if not refine_state["rendering"]:
                    gui_status.content = "*Status: 🔭 Exploring — pause to refine*"
                continue

            # Camera stopped — should we auto-refine?
            if not gui_auto_refine.value:
                continue
            if refine_state["rendering"]:
                continue

            idle = time.time() - refine_state["last_move_time"]
            if idle < gui_refine_delay.value:
                continue

            # Don't re-refine same pose
            yaw, pitch, zoom = current
            last_ref = refine_state["last_refined_pose"]
            if last_ref is not None:
                if (abs(yaw - last_ref[0]) < 1.0
                        and abs(pitch - last_ref[1]) < 1.0
                        and abs(zoom - last_ref[2]) < 0.05):
                    continue

            # Trigger refinement
            threading.Thread(target=do_refine, args=(yaw, pitch, zoom), daemon=True).start()

    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    # ── Keep alive ──
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down...")
        gpu_mgr.restart()


if __name__ == "__main__":
    main()
